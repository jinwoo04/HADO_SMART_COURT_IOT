"""ArUco 마커 기반 자동 캘리브레이션.

코트 4 코너에 ArUco 마커(DICT_4X4_50, ID 0-3)를 부착하면
카메라 영상에서 자동으로 Homography를 계산한다.

마커 배치:
    ID 0 (0m, 0m) ─────────── ID 1 (10m, 0m)
        │      TeamA │ TeamB       │
    ID 3 (0m, 6m) ─────────── ID 2 (10m, 6m)

사용법:
    # 단독 실행 — 카메라/영상에서 마커 감지 테스트
    python -m src.aruco_calibrate
    python -m src.aruco_calibrate --source data/영상.mp4

    # 다른 모듈에서 import
    from src.aruco_calibrate import calibrate_from_aruco, draw_aruco_overlay
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.homography import Calibration, compute_homography

# ── 마커 ID → 코트 좌표 (m) ──────────────────────────────────────
# CLAUDE.md 좌표 규칙: 좌상 원점, x→오른쪽(10m), y↓아래(6m)
MARKER_COURT_M: dict[int, tuple[float, float]] = {
    0: (0.0,  0.0),   # 좌상 (Team A 왼쪽 끝)
    1: (10.0, 0.0),   # 우상 (Team B 오른쪽 끝)
    2: (10.0, 6.0),   # 우하
    3: (0.0,  6.0),   # 좌하
}

# ArUco 딕셔너리 — 4×4 격자 50개 마커, 오감지 적음
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR   = cv2.aruco.ArucoDetector(
    _ARUCO_DICT,
    cv2.aruco.DetectorParameters(),
)

# 마커 색상 (ID별)
_MARKER_COLORS = {
    0: (0,   0,   255),  # 빨강
    1: (0,   255, 255),  # 노랑
    2: (0,   255,   0),  # 초록
    3: (255, 100,   0),  # 파랑
}


# ── 핵심 함수 ─────────────────────────────────────────────────────

def detect_markers(
    frame: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """프레임에서 ArUco 마커를 감지하고 {id: (center_x, center_y)} 반환.

    MARKER_COURT_M에 정의된 ID(0~3)만 반환.
    """
    corners, ids, _ = _DETECTOR.detectMarkers(frame)
    result: dict[int, tuple[float, float]] = {}
    if ids is None:
        return result
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in MARKER_COURT_M:
            continue
        # corners[i] shape: (1, 4, 2) — 시계방향 4꼭짓점
        pts = corners[i].reshape(4, 2)
        cx, cy = pts.mean(axis=0)
        result[int(marker_id)] = (float(cx), float(cy))
    return result


def calibrate_from_aruco(
    frame: np.ndarray,
    court_width_m: float = 10.0,
    court_height_m: float = 6.0,
) -> Optional[Calibration]:
    """4개 마커가 모두 감지되면 Calibration 반환, 아니면 None."""
    detected = detect_markers(frame)
    required = set(MARKER_COURT_M.keys())
    if not required.issubset(detected.keys()):
        return None

    h, w = frame.shape[:2]
    # 시계방향 순서: 0(좌상) → 1(우상) → 2(우하) → 3(좌하)
    order = [0, 1, 2, 3]
    corners_px = np.array([detected[i] for i in order], dtype=np.float32)

    return compute_homography(
        corners_pixel=corners_px,
        court_width_m=court_width_m,
        court_height_m=court_height_m,
        image_size=(w, h),
    )


def draw_aruco_overlay(
    frame: np.ndarray,
    detected: dict[int, tuple[float, float]],
    calib: Optional[Calibration] = None,
) -> np.ndarray:
    """감지된 마커를 프레임에 오버레이 (in-place 아닌 복사본 반환)."""
    out = frame.copy()
    required = set(MARKER_COURT_M.keys())
    found = set(detected.keys())

    for marker_id, (cx, cy) in detected.items():
        color  = _MARKER_COLORS.get(marker_id, (200, 200, 200))
        court  = MARKER_COURT_M[marker_id]
        label  = f"ID{marker_id} ({court[0]:.0f},{court[1]:.0f})m"
        # 원 + 라벨
        cv2.circle(out, (int(cx), int(cy)), 12, color, -1, cv2.LINE_AA)
        cv2.circle(out, (int(cx), int(cy)), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (int(cx) + 16, int(cy) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # 4점 모두 감지되면 코트 윤곽 폴리곤 그리기
    if required.issubset(found):
        order = [0, 1, 2, 3]
        pts = np.array([[int(detected[i][0]), int(detected[i][1])] for i in order],
                       dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 180), thickness=2,
                      lineType=cv2.LINE_AA)

    # 상태 배너
    n = len(found & required)
    if n == 4:
        banner, bcolor = "ArUco 캘리브레이션 완료", (0, 200, 80)
    else:
        missing = required - found
        banner  = f"ArUco: {n}/4 감지  (미감지 ID: {sorted(missing)})"
        bcolor  = (0, 120, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, bcolor, 2, cv2.LINE_AA)

    return out


def auto_calibrate_loop(
    source,
    court_width_m: float = 10.0,
    court_height_m: float = 6.0,
    output_path: Optional[Path] = None,
    save_on_detect: bool = True,
) -> Optional[Calibration]:
    """카메라/영상을 열고 4개 마커가 감지될 때까지 대기, 감지되면 Calibration 반환.

    Parameters
    ----------
    source          : cv2.VideoCapture 소스 (int 또는 파일 경로)
    save_on_detect  : True면 output_path에 자동 저장
    """
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        print(f"[ArUco] 소스 열기 실패: {source}")
        return None

    cv2.namedWindow("ArUco Calibration", cv2.WINDOW_NORMAL)
    calib: Optional[Calibration] = None
    print("[ArUco] 4개 마커(ID 0-3)가 모두 보이도록 카메라를 조정하세요. ESC: 취소")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected = detect_markers(frame)
        display  = draw_aruco_overlay(frame, detected)

        if len(detected) == 4 and calib is None:
            calib = calibrate_from_aruco(frame, court_width_m, court_height_m)
            if calib and save_on_detect and output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                calib.to_json(output_path)
                print(f"[ArUco] 자동 저장: {output_path}")
            print("[ArUco] 캘리브레이션 완료. ENTER로 저장 확인, ESC로 종료.")

        cv2.imshow("ArUco Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:  # ESC 또는 ENTER
            break

    cap.release()
    cv2.destroyAllWindows()
    return calib


# ── 마커 이미지 생성 유틸 ──────────────────────────────────────────

def generate_marker_images(
    out_dir: str | Path = "config/aruco_markers",
    size_px: int = 300,
) -> None:
    """코트 설치용 마커 이미지(ID 0~3) PNG 생성."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for marker_id, court_pos in MARKER_COURT_M.items():
        img = cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, size_px)
        # 흰 여백 추가 (인쇄 시 여백 필요)
        pad = size_px // 6
        img_pad = np.ones((size_px + pad * 2, size_px + pad * 2), dtype=np.uint8) * 255
        img_pad[pad:pad + size_px, pad:pad + size_px] = img
        fname = out / f"marker_id{marker_id}_court{court_pos[0]:.0f}x{court_pos[1]:.0f}.png"
        cv2.imwrite(str(fname), img_pad)
        print(f"  {fname}")
    print(f"[ArUco] 마커 {len(MARKER_COURT_M)}개 생성 → {out}")
    print("  → A4 출력 후 코트 각 코너 바닥에 부착하세요.")


# ── 단독 실행 ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ArUco 마커 캘리브레이션")
    parser.add_argument("--source",  default="0",            help="카메라 인덱스 또는 영상 경로")
    parser.add_argument("--output",  default="config/calibration.json")
    parser.add_argument("--gen-markers", action="store_true", help="마커 이미지만 생성하고 종료")
    parser.add_argument("--marker-dir",  default="config/aruco_markers")
    args = parser.parse_args()

    if args.gen_markers:
        generate_marker_images(args.marker_dir)
        return

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    calib = auto_calibrate_loop(
        source=source,
        output_path=Path(args.output),
        save_on_detect=True,
    )
    if calib:
        print(f"[ArUco] 성공. calibration.json 저장됨.")
    else:
        print("[ArUco] 캘리브레이션 실패 또는 취소.")


if __name__ == "__main__":
    main()
