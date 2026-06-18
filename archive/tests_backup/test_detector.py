"""Detection 데이터클래스 단위 테스트.

pytest 또는 단독 실행: python -m tests.test_detector

PersonDetector(YOLOv8)는 모델 파일이 필요해 CI에서 제외.
Detection 프로퍼티만 테스트 (하드웨어/모델 불필요).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import Detection


# ── bbox ──────────────────────────────────────────────────────────────────────

def test_bbox_returns_float32_array():
    d = Detection(x1=10, y1=20, x2=50, y2=80, confidence=0.9)
    assert d.bbox.dtype == np.float32
    assert d.bbox.shape == (4,)
    print("  ✓ bbox: float32 shape (4,)")


def test_bbox_values():
    d = Detection(x1=10.0, y1=20.0, x2=50.0, y2=80.0, confidence=0.9)
    expected = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    assert np.array_equal(d.bbox, expected)
    print("  ✓ bbox: 값 정확")


# ── foot_point ────────────────────────────────────────────────────────────────

def test_foot_point_is_bottom_center():
    """발 위치 = 하단 중앙: ((x1+x2)/2, y2)."""
    d = Detection(x1=100, y1=200, x2=200, y2=500, confidence=0.9)
    fx, fy = d.foot_point
    assert fx == 150.0
    assert fy == 500.0
    print("  ✓ foot_point: 하단 중앙 (150, 500)")


def test_foot_point_square_box():
    d = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.8)
    assert d.foot_point == (50.0, 100.0)
    print("  ✓ foot_point: 정사각형 박스")


def test_foot_point_single_pixel():
    d = Detection(x1=5, y1=5, x2=5, y2=5, confidence=0.7)
    assert d.foot_point == (5.0, 5.0)
    print("  ✓ foot_point: 점(단일 픽셀) 처리")


# ── center ────────────────────────────────────────────────────────────────────

def test_center_is_midpoint():
    d = Detection(x1=0, y1=0, x2=200, y2=100, confidence=0.9)
    cx, cy = d.center
    assert cx == 100.0
    assert cy == 50.0
    print("  ✓ center: 중앙점 정확")


def test_center_non_square():
    d = Detection(x1=10, y1=30, x2=90, y2=70, confidence=0.85)
    cx, cy = d.center
    assert cx == 50.0
    assert cy == 50.0
    print("  ✓ center: 비정사각형 박스")


# ── area ──────────────────────────────────────────────────────────────────────

def test_area_rectangle():
    d = Detection(x1=0, y1=0, x2=40, y2=80, confidence=0.9)
    assert d.area == 3200.0
    print("  ✓ area: 40×80 = 3200")


def test_area_zero_for_degenerate():
    """역방향 좌표(x2 < x1)는 면적 0."""
    d = Detection(x1=100, y1=100, x2=50, y2=50, confidence=0.5)
    assert d.area == 0.0
    print("  ✓ area: 역방향 좌표 → 0")


def test_area_square():
    d = Detection(x1=10, y1=10, x2=110, y2=110, confidence=0.9)
    assert d.area == 10000.0
    print("  ✓ area: 100×100 정사각형")


# ── 좌표 일관성 ───────────────────────────────────────────────────────────────

def test_foot_point_y_equals_y2():
    """foot_point y는 항상 y2여야 한다 (homography 입력 규약)."""
    for y2 in [100, 300, 720]:
        d = Detection(x1=0, y1=0, x2=100, y2=y2, confidence=0.9)
        assert d.foot_point[1] == float(y2), f"y2={y2}일 때 foot_point.y 불일치"
    print("  ✓ foot_point.y == y2 규약 준수")


def test_foot_point_x_is_horizontal_center():
    """foot_point x는 (x1+x2)/2와 정확히 일치."""
    d = Detection(x1=150, y1=200, x2=250, y2=480, confidence=0.88)
    assert d.foot_point[0] == (150 + 250) / 2.0
    print("  ✓ foot_point.x == (x1+x2)/2 규약 준수")


def test_center_and_foot_x_share_horizontal_center():
    """center.x == foot_point.x (둘 다 수평 중앙)."""
    d = Detection(x1=200, y1=100, x2=400, y2=500, confidence=0.9)
    assert d.center[0] == d.foot_point[0]
    print("  ✓ center.x == foot_point.x")


def test_foot_point_y_below_center_y():
    """foot_point.y > center.y (발이 중심보다 아래)."""
    d = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.9)
    assert d.foot_point[1] > d.center[1]
    print("  ✓ foot_point.y > center.y (발은 중심보다 아래)")


def test_confidence_stored():
    d = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.75)
    assert d.confidence == 0.75
    print("  ✓ confidence 값 보존")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 50)
    print(" Detection 테스트")
    print("=" * 50)
    test_bbox_returns_float32_array()
    test_bbox_values()
    test_foot_point_is_bottom_center()
    test_foot_point_square_box()
    test_foot_point_single_pixel()
    test_center_is_midpoint()
    test_center_non_square()
    test_area_rectangle()
    test_area_zero_for_degenerate()
    test_area_square()
    test_foot_point_y_equals_y2()
    test_foot_point_x_is_horizontal_center()
    test_center_and_foot_x_share_horizontal_center()
    test_foot_point_y_below_center_y()
    test_confidence_stored()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
