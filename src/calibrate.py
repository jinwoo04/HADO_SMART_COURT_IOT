"""캘리브레이션 도구.

사용법
------
1. 코트 4 코너에 컬러 마커 부착
2. 카메라를 최종 거치 위치에 고정
3. python -m src.calibrate
4. 화면 안내:
   - SPACE : 현재 프레임 캡처 (정지)
   - 마우스 좌클릭 4번 : 좌상 → 우상 → 우하 → 좌하 순서 (시계방향)
   - U : 마지막 점 취소
   - R : 다시 찍기
   - ENTER : 저장
   - ESC : 취소
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.homography import (
    Calibration, Intrinsics,
    build_undistort_maps, compute_homography, pixel_to_court,
)


CORNER_NAMES = ["좌상 (Top-Left)", "우상 (Top-Right)", "우하 (Bottom-Right)", "좌하 (Bottom-Left)"]
CORNER_COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 100, 0)]


class CalibrationTool:
    def __init__(self, court_width_m: float, court_height_m: float):
        self.court_width_m = court_width_m
        self.court_height_m = court_height_m
        self.frozen_frame: np.ndarray | None = None
        self.clicked_points: list[tuple[int, int]] = []

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.frozen_frame is not None:
            if len(self.clicked_points) < 4:
                self.clicked_points.append((x, y))
                print(f"  ✓ {CORNER_NAMES[len(self.clicked_points)-1]} = ({x}, {y})")

    def _draw_overlay(self, frame: np.ndarray, live: bool) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # 상단 안내 텍스트
        if live:
            msg = "라이브 모드 — SPACE를 눌러 프레임 캡처 | ESC: 종료"
            color = (0, 255, 0)
        else:
            if len(self.clicked_points) < 4:
                msg = f"클릭: {CORNER_NAMES[len(self.clicked_points)]} ({len(self.clicked_points)}/4) | U: 취소 | R: 리셋"
            else:
                msg = "ENTER: 저장 | R: 다시 | ESC: 취소"
            color = (255, 255, 0)
        cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(out, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        # 클릭된 점 표시
        for i, (x, y) in enumerate(self.clicked_points):
            cv2.circle(out, (x, y), 8, CORNER_COLORS[i], -1)
            cv2.circle(out, (x, y), 10, (255, 255, 255), 2)
            label = f"{i+1}"
            cv2.putText(out, label, (x + 14, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_COLORS[i], 2)

        # 4점 다 클릭하면 폴리곤으로 연결
        if len(self.clicked_points) >= 2:
            pts = np.array(self.clicked_points, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=(len(self.clicked_points) == 4),
                          color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        return out

    def _validate_calibration(self, calib: Calibration) -> float:
        """4 코너의 변환 오차 평균을 m 단위로 반환. 클수록 부정확."""
        transformed = pixel_to_court(calib.corners_pixel, calib)
        expected = calib.corners_court_m
        errors = np.linalg.norm(transformed - expected, axis=1)
        return float(np.mean(errors))

    def run(self, camera: Camera, output_path: Path) -> bool:
        window = "HADO Calibration"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse_callback)

        print("=" * 60)
        print(" HADO 캘리브레이션 도구")
        print(f" 코트 규격: {self.court_width_m} m × {self.court_height_m} m")
        print(" 카메라를 거치 위치에 고정한 상태에서 진행하세요.")
        print("=" * 60)

        image_size: tuple[int, int] = (0, 0)

        while True:
            if self.frozen_frame is None:
                ok, frame = camera.read()
                if not ok:
                    print("[!] 카메라 프레임 읽기 실패")
                    break
                image_size = (frame.shape[1], frame.shape[0])
                display = self._draw_overlay(frame, live=True)
            else:
                display = self._draw_overlay(self.frozen_frame, live=False)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[Calibrate] 취소됨")
                cv2.destroyAllWindows()
                return False

            elif key == 32:  # SPACE
                if self.frozen_frame is None:
                    self.frozen_frame = frame.copy()
                    print("[Calibrate] 프레임 캡처 완료. 4 코너를 시계방향으로 클릭하세요.")
                    print(f"  순서: {' → '.join(CORNER_NAMES)}")

            elif key == ord('u'):  # 마지막 점 취소
                if self.clicked_points:
                    removed = self.clicked_points.pop()
                    print(f"[Calibrate] 점 취소: {removed}")

            elif key == ord('r'):  # 리셋
                self.clicked_points.clear()
                self.frozen_frame = None
                print("[Calibrate] 리셋. 다시 캡처하세요.")

            elif key == 13:  # ENTER
                if len(self.clicked_points) == 4 and self.frozen_frame is not None:
                    corners_px = np.array(self.clicked_points, dtype=np.float32)
                    calib = compute_homography(
                        corners_pixel=corners_px,
                        court_width_m=self.court_width_m,
                        court_height_m=self.court_height_m,
                        image_size=image_size,
                    )
                    err = self._validate_calibration(calib)
                    print(f"[Calibrate] 변환 평균 오차: {err*1000:.2f} mm")
                    if err > 0.05:
                        print("  ⚠ 오차가 5cm 초과 — 코너 클릭이 부정확할 수 있습니다.")

                    calib.to_json(output_path)
                    print(f"[Calibrate] ✓ 저장 완료: {output_path}")

                    # 미리보기 이미지 저장
                    preview_path = output_path.with_suffix(".preview.jpg")
                    cv2.imwrite(str(preview_path), self._draw_overlay(self.frozen_frame, live=False))
                    print(f"[Calibrate] 미리보기 이미지: {preview_path}")

                    cv2.destroyAllWindows()
                    return True
                else:
                    print(f"[!] 아직 {4 - len(self.clicked_points)}개 코너가 남았습니다.")

        cv2.destroyAllWindows()
        return False


class IntrinsicCalibrationTool:
    """체커보드를 이용한 렌즈 왜곡 보정 파라미터 추출.

    사용법
    ------
    1. 9×6 체커보드 A4 출력 (config/checkerboard_9x6.png 생성 후 인쇄)
    2. 카메라 최종 거치 위치에 고정
    3. 체커보드를 들고 화면 앞에서 좌/우/상/하 + 기울기 방향으로 15~20장 캡처
       - SPACE : 체커보드 인식 시 캡처 (빨간 테두리가 초록으로 바뀔 때)
       - ENTER : 15장 이상 캡처 시 계산 시작
       - ESC   : 취소
    4. config/cam{id}_intrinsics.json 저장
    """

    BOARD_W = 9    # 내부 코너 가로 수
    BOARD_H = 6    # 내부 코너 세로 수
    MIN_FRAMES = 15

    _CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, cam_id: int = 0):
        self.cam_id = cam_id
        self._obj_pts: list[np.ndarray] = []   # 3D 기준점
        self._img_pts: list[np.ndarray] = []   # 2D 감지 코너

        # 체커보드 3D 기준점 (z=0 평면)
        objp = np.zeros((self.BOARD_W * self.BOARD_H, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.BOARD_W, 0:self.BOARD_H].T.reshape(-1, 2)
        self._objp = objp

    @staticmethod
    def generate_checkerboard_image(out_path: str = "config/checkerboard_9x6.png") -> None:
        """9×6 체커보드 PNG 생성 (A4 출력용)."""
        sq = 80   # 픽셀당 정사각형 크기
        w = (IntrinsicCalibrationTool.BOARD_W + 1) * sq
        h = (IntrinsicCalibrationTool.BOARD_H + 1) * sq
        img = np.ones((h, w), dtype=np.uint8) * 255
        for r in range(IntrinsicCalibrationTool.BOARD_H + 1):
            for c in range(IntrinsicCalibrationTool.BOARD_W + 1):
                if (r + c) % 2 == 0:
                    img[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 0
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, img)
        print(f"[Intrinsic] 체커보드 이미지 저장: {out_path}  (A4 인쇄 권장)")

    def _try_detect(self, frame: np.ndarray):
        """체커보드 코너 감지. 반환: (corners_subpix or None, gray)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (self.BOARD_W, self.BOARD_H),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None, gray
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._CRITERIA)
        return corners_sub, gray

    def run(self, source, output_path: Path) -> bool:
        window = "HADO Intrinsic Calibration"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 600)

        print("=" * 60)
        print(f" 렌즈 왜곡 캘리브레이션  (카메라 ID: {self.cam_id})")
        print(f" 체커보드: {self.BOARD_W}×{self.BOARD_H} 내부 코너")
        print(f" 목표: {self.MIN_FRAMES}장 이상 캡처 후 ENTER")
        print("=" * 60)
        print(" SPACE : 체커보드 감지 시 캡처")
        print(" ENTER : 계산 (15장 이상일 때)")
        print(" ESC   : 취소")
        print("-" * 60)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[!] 카메라 열기 실패: {source}")
            return False

        image_size: Tuple[int, int] = (0, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            image_size = (w, h)

            corners, gray = self._try_detect(frame)
            detected = corners is not None

            display = frame.copy()

            # 테두리 색으로 인식 여부 표시
            border_color = (0, 220, 80) if detected else (0, 60, 220)
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), border_color, 6)

            if detected:
                cv2.drawChessboardCorners(display, (self.BOARD_W, self.BOARD_H), corners, True)

            # 상단 HUD
            n = len(self._obj_pts)
            status = f"캡처: {n}/{self.MIN_FRAMES}장"
            hint   = "SPACE: 캡처" if detected else "체커보드를 화면에 맞춰주세요"
            ready  = "  |  ENTER: 계산 시작" if n >= self.MIN_FRAMES else ""
            cv2.rectangle(display, (0, 0), (w, 46), (0, 0, 0), -1)
            cv2.putText(display, f"{status}  {hint}{ready}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            # 캡처된 프레임 수 시각화 (하단 바)
            bar_w = int(w * min(n, self.MIN_FRAMES) / self.MIN_FRAMES)
            cv2.rectangle(display, (0, h - 8), (bar_w, h), (0, 200, 80), -1)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # ESC
                print("[Intrinsic] 취소됨")
                break

            elif key == 32 and detected:   # SPACE + 체커보드 감지됨
                self._obj_pts.append(self._objp)
                self._img_pts.append(corners)
                print(f"  [{n+1:02d}] 캡처 완료 — 다른 각도로 이동하세요")

            elif key == 13 and len(self._obj_pts) >= self.MIN_FRAMES:   # ENTER
                cap.release()
                cv2.destroyAllWindows()
                return self._compute_and_save(image_size, output_path)

        cap.release()
        cv2.destroyAllWindows()
        return False

    def _compute_and_save(self, image_size: Tuple[int, int], output_path: Path) -> bool:
        print(f"\n[Intrinsic] {len(self._obj_pts)}장으로 캘리브레이션 계산 중...")
        ret, K, dist, _, _ = cv2.calibrateCamera(
            self._obj_pts, self._img_pts, image_size, None, None
        )
        print(f"[Intrinsic] 재투영 오차(RMS): {ret:.4f} px", end="  ")
        if ret < 0.5:
            print("(양호 ✓)")
        elif ret < 1.0:
            print("(보통 — 더 많은 각도로 재시도 권장)")
        else:
            print("(불량 ⚠ — 체커보드가 구겨지지 않았는지 확인 후 재시도)")

        intrinsics = Intrinsics(
            cam_id=self.cam_id,
            image_size=image_size,
            camera_matrix=K,
            dist_coeffs=dist.ravel(),
            reprojection_error=float(ret),
        )
        intrinsics.to_json(output_path)
        print(f"[Intrinsic] ✓ 저장: {output_path}")
        print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
        print(f"  왜곡계수: {dist.ravel().round(4).tolist()}")

        # 보정 전후 비교 이미지 저장
        preview_path = output_path.with_suffix(".undistort_preview.jpg")
        self._save_preview(intrinsics, preview_path)
        return True

    def _save_preview(self, intrinsics: Intrinsics, out_path: Path) -> None:
        """마지막 캡처 프레임의 보정 전/후 비교 이미지 저장."""
        if not self._img_pts:
            return
        # 마지막 캡처 이미지를 재구성 (corners로 임시 역산 불가 → 스킵)
        # 실제로는 캡처된 원본 프레임을 보관해야 하므로 여기선 안내만 출력
        print(f"[Intrinsic] 보정 전후 비교는 --preview 옵션으로 확인하세요")
        print(f"  python -m src.calibrate --intrinsic --preview --cam-id {intrinsics.cam_id}")


def main():
    parser = argparse.ArgumentParser(description="HADO 캘리브레이션 도구")
    parser.add_argument("--source", default=0, help="카메라 소스 (정수=장치번호, 문자열=영상 경로)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--court-width",  type=float, default=10.0)
    parser.add_argument("--court-height", type=float, default=6.0)
    parser.add_argument("--output", default="config/calibration.json")
    # ArUco 관련
    parser.add_argument("--aruco",       action="store_true", help="ArUco 마커 자동 캘리브레이션")
    parser.add_argument("--gen-markers", action="store_true", help="코트용 ArUco 마커 이미지 생성")
    # 렌즈 왜곡 보정 관련
    parser.add_argument("--intrinsic",       action="store_true",
                        help="렌즈 왜곡 보정 캘리브레이션 (체커보드 촬영)")
    parser.add_argument("--cam-id",          type=int, default=0,
                        help="카메라 ID (멀티카메라 시 구분용, 기본값=0)")
    parser.add_argument("--gen-checkerboard", action="store_true",
                        help="9×6 체커보드 이미지 생성 (A4 출력 후 사용)")
    parser.add_argument("--preview",         action="store_true",
                        help="저장된 intrinsics로 왜곡 보정 전후 비교 (--intrinsic과 함께)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    output_path = Path(args.output)

    # ── 체커보드 이미지 생성 ────────────────────────────────────
    if args.gen_checkerboard:
        IntrinsicCalibrationTool.generate_checkerboard_image()
        sys.exit(0)

    # ── 렌즈 왜곡 보정 캘리브레이션 ────────────────────────────
    if args.intrinsic:
        intrinsic_path = Path(f"config/cam{args.cam_id}_intrinsics.json")

        if args.preview:
            # 보정 전후 실시간 비교
            if not intrinsic_path.exists():
                print(f"[!] intrinsics 파일 없음: {intrinsic_path}")
                print(f"    먼저 --intrinsic --cam-id {args.cam_id} 로 캘리브레이션 진행")
                sys.exit(1)
            from src.homography import build_undistort_maps, Intrinsics
            intr = Intrinsics.from_json(intrinsic_path)
            map1, map2 = build_undistort_maps(intr)
            cap = cv2.VideoCapture(source)
            print("왜곡 보정 미리보기 — ESC 종료  (왼쪽: 원본 / 오른쪽: 보정)")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                import src.homography as _hg
                undist = _hg.undistort_frame(frame, map1, map2)
                h, w = frame.shape[:2]
                comp = np.hstack([
                    cv2.putText(frame.copy(),  "원본",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,80,255), 2),
                    cv2.putText(undist.copy(), "보정후", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,220,80), 2),
                ])
                cv2.imshow("Undistort Preview (ESC to quit)", comp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        tool = IntrinsicCalibrationTool(cam_id=args.cam_id)
        success = tool.run(source, intrinsic_path)
        sys.exit(0 if success else 1)

    # ── ArUco 마커 생성 ─────────────────────────────────────────
    if args.gen_markers:
        from src.aruco_calibrate import generate_marker_images
        generate_marker_images()
        sys.exit(0)

    # ── ArUco 코트 캘리브레이션 ────────────────────────────────
    if args.aruco:
        from src.aruco_calibrate import auto_calibrate_loop
        calib = auto_calibrate_loop(
            source=source,
            court_width_m=args.court_width,
            court_height_m=args.court_height,
            output_path=output_path,
            save_on_detect=True,
        )
        sys.exit(0 if calib else 1)

    # ── 기본: 4점 클릭 코트 캘리브레이션 ───────────────────────
    with Camera(source=source, width=args.width, height=args.height) as cam:
        tool = CalibrationTool(args.court_width, args.court_height)
        success = tool.run(cam, output_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
