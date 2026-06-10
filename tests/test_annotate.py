"""annotate.put_text_kr ROI 렌더링 테스트."""
from __future__ import annotations

import numpy as np

from src.annotate import put_text_kr


def _blank(h: int = 120, w: int = 300) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_put_text_kr_modifies_pixels():
    """텍스트 렌더링 후 이미지가 실제로 변해야 한다."""
    img = _blank()
    before = img.copy()
    put_text_kr(img, "전술 가이드", (10, 40), 18, (0, 255, 0))
    assert not np.array_equal(img, before), "텍스트가 렌더링되지 않음"
    print("  ✓ put_text_kr: 픽셀 변경 확인")


def test_put_text_kr_only_touches_text_region():
    """텍스트 bbox 바깥(반대쪽 모서리) 픽셀은 변하지 않아야 한다."""
    img = _blank()
    put_text_kr(img, "ABC", (5, 5), 14, (255, 255, 255))
    # 우하단 구석은 텍스트와 무관 — 변경 없어야 함
    assert np.array_equal(img[-20:, -20:], np.zeros((20, 20, 3), dtype=np.uint8))
    print("  ✓ put_text_kr: ROI 외부 픽셀 보존")


def test_put_text_kr_empty_text_noop():
    """빈 문자열은 no-op."""
    img = _blank()
    before = img.copy()
    put_text_kr(img, "", (10, 10), 14, (0, 0, 255))
    assert np.array_equal(img, before)
    print("  ✓ put_text_kr: 빈 텍스트 no-op")


def test_put_text_kr_out_of_bounds_safe():
    """이미지 밖 좌표에서도 크래시 없이 동작해야 한다."""
    img = _blank()
    before = img.copy()
    put_text_kr(img, "밖", (1000, 1000), 14, (0, 0, 255))   # 완전 화면 밖
    put_text_kr(img, "걸침", (290, 110), 14, (0, 0, 255))   # 경계 걸침 — 크래시만 없으면 됨
    assert np.array_equal(img[:50, :50], before[:50, :50])
    print("  ✓ put_text_kr: 화면 밖 좌표 안전")


def run_all():
    print("=" * 55)
    print(" Annotate 테스트")
    print("=" * 55)
    test_put_text_kr_modifies_pixels()
    test_put_text_kr_only_touches_text_region()
    test_put_text_kr_empty_text_noop()
    test_put_text_kr_out_of_bounds_safe()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
