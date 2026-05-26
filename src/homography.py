"""픽셀 ↔ 코트 좌표 변환 (Homography).

캘리브레이션 단계에서 저장된 4점 매핑을 이용해
- 카메라 픽셀 좌표 (x_px, y_px) → 실제 코트 좌표 (x_m, y_m) 변환
- 그 반대 (m → px) 도 가능
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Calibration:
    """캘리브레이션 데이터 컨테이너."""
    court_width_m: float
    court_height_m: float
    image_size: Tuple[int, int]   # (width, height)
    corners_pixel: np.ndarray     # shape (4, 2), 시계방향 [TL, TR, BR, BL]
    corners_court_m: np.ndarray   # shape (4, 2), 위와 같은 순서로 매핑되는 실제 좌표
    homography: np.ndarray        # 3x3, 픽셀 → 코트 변환
    homography_inv: np.ndarray    # 3x3, 코트 → 픽셀 변환

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            court_width_m=float(data["court_width_m"]),
            court_height_m=float(data["court_height_m"]),
            image_size=tuple(data["image_size"]),
            corners_pixel=np.array(data["corners_pixel"], dtype=np.float32),
            corners_court_m=np.array(data["corners_court_m"], dtype=np.float32),
            homography=np.array(data["homography"], dtype=np.float64),
            homography_inv=np.array(data["homography_inv"], dtype=np.float64),
        )

    def to_json(self, path: str | Path):
        data = {
            "court_width_m": self.court_width_m,
            "court_height_m": self.court_height_m,
            "image_size": list(self.image_size),
            "corners_pixel": self.corners_pixel.tolist(),
            "corners_court_m": self.corners_court_m.tolist(),
            "homography": self.homography.tolist(),
            "homography_inv": self.homography_inv.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def compute_homography(
    corners_pixel: np.ndarray,
    court_width_m: float,
    court_height_m: float,
    image_size: Tuple[int, int],
) -> Calibration:
    """4점 픽셀 좌표 → 코트 실제 좌표 매핑으로 Homography 계산.

    Parameters
    ----------
    corners_pixel : (4, 2) 시계방향 [좌상, 우상, 우하, 좌하] 픽셀 좌표
    court_width_m : 코트 가로 길이 (m)
    court_height_m : 코트 세로 길이 (m)
    image_size : (W, H) 캘리브레이션 시 이미지 크기
    """
    corners_pixel = np.asarray(corners_pixel, dtype=np.float32).reshape(4, 2)

    # 코트 좌표계: 좌상단(0,0), 우하단(W, H)
    corners_court = np.array([
        [0.0, 0.0],                         # 좌상
        [court_width_m, 0.0],               # 우상
        [court_width_m, court_height_m],    # 우하
        [0.0, court_height_m],              # 좌하
    ], dtype=np.float32)

    H, _ = cv2.findHomography(corners_pixel, corners_court, method=0)
    H_inv = np.linalg.inv(H)

    return Calibration(
        court_width_m=court_width_m,
        court_height_m=court_height_m,
        image_size=image_size,
        corners_pixel=corners_pixel,
        corners_court_m=corners_court,
        homography=H,
        homography_inv=H_inv,
    )


def pixel_to_court(points_px: np.ndarray, calib: Calibration) -> np.ndarray:
    """픽셀 좌표를 코트 좌표(m)로 변환.

    Parameters
    ----------
    points_px : shape (N, 2) — [(x_px, y_px), ...]
    calib : Calibration

    Returns
    -------
    shape (N, 2) — [(x_m, y_m), ...]
    """
    points_px = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_px, calib.homography)
    return transformed.reshape(-1, 2)


def court_to_pixel(points_m: np.ndarray, calib: Calibration) -> np.ndarray:
    """코트 좌표(m)를 픽셀 좌표로 역변환."""
    points_m = np.asarray(points_m, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_m, calib.homography_inv)
    return transformed.reshape(-1, 2)


def is_inside_court(points_m: np.ndarray, calib: Calibration, margin_m: float = 0.5) -> np.ndarray:
    """코트 안쪽인지 boolean 마스크 반환 (out-of-bounds 필터링용).

    margin_m : 코트 외곽 허용 마진 (선수가 라인을 넘는 순간을 위한 여유)
    """
    points_m = np.asarray(points_m, dtype=np.float32).reshape(-1, 2)
    in_x = (points_m[:, 0] >= -margin_m) & (points_m[:, 0] <= calib.court_width_m + margin_m)
    in_y = (points_m[:, 1] >= -margin_m) & (points_m[:, 1] <= calib.court_height_m + margin_m)
    return in_x & in_y


# ---------- 단독 테스트 ----------
def main():
    """단독 실행: 더미 데이터로 변환 정확도 확인."""
    # 시뮬레이션: 1280x720 카메라가 비스듬히 6m x 2.66m 코트를 본다고 가정
    fake_corners_px = np.array([
        [200, 200],   # 좌상
        [1080, 250],  # 우상 (원근감 때문에 약간 안쪽)
        [1180, 600],  # 우하
        [100, 580],   # 좌하
    ], dtype=np.float32)

    calib = compute_homography(
        corners_pixel=fake_corners_px,
        court_width_m=6.0,
        court_height_m=2.66,
        image_size=(1280, 720),
    )
    print("[Homography] 계산 완료")
    print(f"  H =\n{calib.homography}")

    # 코너 4점을 변환해서 코트 좌표가 정확히 (0,0)~(6,2.66) 모서리에 떨어지는지 확인
    transformed = pixel_to_court(fake_corners_px, calib)
    print("[Homography] 4 코너 변환 결과 (m 단위):")
    for i, p in enumerate(transformed):
        print(f"  corner {i}: ({p[0]:.3f}, {p[1]:.3f})")

    # 중앙점 테스트
    center_px = np.array([[640, 400]], dtype=np.float32)
    center_m = pixel_to_court(center_px, calib)
    print(f"[Homography] 화면 중앙 (640, 400) → 코트 ({center_m[0,0]:.2f}, {center_m[0,1]:.2f}) m")

    # 역변환 라운드트립
    back = court_to_pixel(center_m, calib)
    err = np.linalg.norm(back[0] - center_px[0])
    print(f"[Homography] 역변환 오차: {err:.4f} px (이상적으로 0)")


if __name__ == "__main__":
    main()
