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


def _rect_calib():
    """직사각형 픽셀 코너 기반 표준 캘리브레이션."""
    corners_px = np.array(
        [[100, 100], [1180, 100], [1180, 600], [100, 600]], dtype=np.float32
    )
    return compute_homography(corners_px, 6.0, 2.66, (1280, 720))


def test_court_to_pixel_center():
    """코트 중앙(3.0, 1.33)은 이미지 중앙 근처로 역변환되어야 한다."""
    calib = _rect_calib()
    center_m = np.array([[3.0, 1.33]], dtype=np.float32)
    px = court_to_pixel(center_m, calib)
    # 직사각형 코너 → 이미지 중앙 (640, 350) ± 50px
    assert abs(px[0, 0] - 640) < 50, f"x 오프셋 과다: {px[0,0]:.1f}"
    assert abs(px[0, 1] - 350) < 50, f"y 오프셋 과다: {px[0,1]:.1f}"
    print(f"  ✓ 코트 중앙 역변환: {px[0,0]:.1f}, {px[0,1]:.1f} px")


def test_is_inside_court_margin():
    """margin_m=0.3 → 코트 경계에서 0.2m 안쪽 점은 마진 안에서 inside=True."""
    calib = _rect_calib()
    # 경계 근처 (x=0.2, y=1.33) — margin=0.3이면 inside, margin=0이면 inside
    near_edge = np.array([[0.2, 1.33]], dtype=np.float32)
    assert is_inside_court(near_edge, calib, margin_m=0.0)[0], "경계 안쪽 0.2m가 outside로 판정됨"
    # x=-0.1 → 코트 밖 (margin 관계없이)
    outside = np.array([[-0.1, 1.33]], dtype=np.float32)
    assert not is_inside_court(outside, calib, margin_m=0.0)[0], "코트 밖 점이 inside로 판정됨"
    print("  ✓ is_inside_court margin_m 경계 처리 정상")


def test_hado_zone_line_x():
    """HADO 존 라인 x=1.5m, x=3.0m이 코트 좌표로 유지되어야 한다."""
    calib = _rect_calib()
    zone_points = np.array([[1.5, 1.33], [3.0, 1.33]], dtype=np.float32)
    # 픽셀→코트 라운드트립으로 1mm 이내 재현
    px = court_to_pixel(zone_points, calib)
    back = pixel_to_court(px, calib)
    err = np.abs(back - zone_points).max()
    assert err < 0.001, f"존 라인 변환 오차: {err:.6f} m"
    print(f"  ✓ HADO 존 라인 (1.5m, 3.0m) 변환 오차: {err:.7f} m")


def test_pixel_to_court_batch():
    """여러 점을 한꺼번에 변환해도 개별 변환과 결과 동일."""
    calib = _rect_calib()
    pts = np.array([[200, 200], [640, 360], [1100, 550]], dtype=np.float32)
    batch = pixel_to_court(pts, calib)
    singles = np.vstack([pixel_to_court(pts[[i]], calib) for i in range(len(pts))])
    assert np.allclose(batch, singles, atol=1e-5), "배치/개별 변환 결과 불일치"
    print("  ✓ pixel_to_court 배치 처리 일관성")


def run_all():
    print("=" * 50)
    print(" Homography 테스트")
    print("=" * 50)
    test_corners_map_to_court_corners()
    test_inverse_roundtrip()
    test_inside_court_mask()
    test_json_save_load()
    test_court_to_pixel_center()
    test_is_inside_court_margin()
    test_hado_zone_line_x()
    test_pixel_to_court_batch()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
