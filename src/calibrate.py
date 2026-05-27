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
from src.homography import Calibration, compute_homography, pixel_to_court


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


def main():
    parser = argparse.ArgumentParser(description="HADO 코트 캘리브레이션 도구")
    parser.add_argument("--source", default=0, help="카메라 소스")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--court-width", type=float, default=10.0, help="코트 가로 (m)")
    parser.add_argument("--court-height", type=float, default=6.0, help="코트 세로 (m)")
    parser.add_argument("--output", default="config/calibration.json")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    output_path = Path(args.output)

    with Camera(source=source, width=args.width, height=args.height) as cam:
        tool = CalibrationTool(args.court_width, args.court_height)
        success = tool.run(cam, output_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
