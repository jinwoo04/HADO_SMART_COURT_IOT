"""Homography 모듈 정합성 테스트.

pytest 또는 단독 실행 가능: python -m tests.test_homography
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path에 추가 (단독 실행 대응)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.homography import (
    Calibration,
    compute_homography,
    pixel_to_court,
    court_to_pixel,
    is_inside_court,
)


def test_corners_map_to_court_corners():
    """4 코너 픽셀이 정확히 코트 모서리 좌표(m)로 변환되어야 한다."""
    corners_px = np.array([
        [100, 100], [1180, 120], [1200, 600], [80, 580]
    ], dtype=np.float32)
    calib = compute_homography(corners_px, 6.0, 2.66, (1280, 720))
    transformed = pixel_to_court(corners_px, calib)
    expected = np.array([[0, 0], [6, 0], [6, 2.66], [0, 2.66]], dtype=np.float32)
    err = np.linalg.norm(transformed - expected)
    assert err < 1e-3, f"코너 변환 오차 과다: {err}"
    print(f"  ✓ 코너 변환 오차: {err:.6f} m")


def test_inverse_roundtrip():
    """픽셀 → 코트 → 픽셀 라운드트립 후 원래 점과 일치해야 한다."""
    corners_px = np.array([
        [100, 100], [1180, 120], [1200, 600], [80, 580]
    ], dtype=np.float32)
    calib = compute_homography(corners_px, 6.0, 2.66, (1280, 720))

    test_points = np.array([[640, 360], [200, 200], [1000, 500]], dtype=np.float32)
    court = pixel_to_court(test_points, calib)
    back = court_to_pixel(court, calib)
    err = np.linalg.norm(back - test_points, axis=1).max()
    assert err < 1e-2, f"라운드트립 오차: {err}"
    print(f"  ✓ 라운드트립 최대 오차: {err:.6f} px")


def test_inside_court_mask():
    corners_px = np.array([
        [100, 100], [1180, 100], [1180, 600], [100, 600]
    ], dtype=np.float32)
    calib = compute_homography(corners_px, 6.0, 2.66, (1280, 720))

    # 코트 중앙은 안쪽
    center_m = np.array([[3.0, 1.33]], dtype=np.float32)
    assert is_inside_court(center_m, calib)[0]

    # 코트 밖
    out_m = np.array([[10.0, 5.0]], dtype=np.float32)
    assert not is_inside_court(out_m, calib)[0]
    print("  ✓ inside_court 마스크 정상")


def test_json_save_load():
    """JSON 직렬화/역직렬화 후 데이터 일치."""
    corners_px = np.array([
        [100, 100], [1180, 100], [1180, 600], [100, 600]
    ], dtype=np.float32)
    calib = compute_homography(corners_px, 6.0, 2.66, (1280, 720))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calib.json"
        calib.to_json(path)
        loaded = Calibration.from_json(path)

    assert np.allclose(calib.homography, loaded.homography)
    assert np.allclose(calib.corners_pixel, loaded.corners_pixel)
    assert calib.court_width_m == loaded.court_width_m
    print("  ✓ JSON 저장/로드 일치")


def run_all():
    print("=" * 50)
    print(" Homography 테스트")
    print("=" * 50)
    test_corners_map_to_court_corners()
    test_inverse_roundtrip()
    test_inside_court_mask()
    test_json_save_load()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
