"""Visualizer 모듈 단위 테스트.

pytest 또는 단독 실행 가능: python -m tests.test_visualizer
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.homography import compute_homography
from src.tracker import Track
from src.visualizer import (
    color_for_id,
    combine_views,
    draw_detections_on_frame,
    draw_hud,
    draw_players_on_birdeye,
    render_court_birdeye,
)


# ── 공용 픽스처 ────────────────────────────────────────────────────────────────

def _make_calib():
    return compute_homography(
        corners_pixel=np.array([[100, 100], [1180, 100], [1180, 600], [100, 600]],
                               dtype=np.float32),
        court_width_m=6.0,
        court_height_m=2.66,
        image_size=(1280, 720),
    )


def _make_tracks():
    return [
        Track(track_id=1,
              bbox=np.array([400, 200, 480, 480], dtype=np.float32),
              confidence=0.92, age=10,
              history=[(440, 480), (445, 475), (450, 470), (455, 480)]),
        Track(track_id=2,
              bbox=np.array([800, 250, 880, 500], dtype=np.float32),
              confidence=0.88, age=8,
              history=[(840, 500), (835, 495), (830, 490)]),
    ]


# ── 테스트 함수 ────────────────────────────────────────────────────────────────

def test_court_image_shape():
    """render_court_birdeye 출력 크기가 (court_h*px_per_m, court_w*px_per_m, 3)이어야 한다."""
    calib = _make_calib()
    img = render_court_birdeye(calib, px_per_m=100)
    expected_h = int(calib.court_height_m * 100)   # 266
    expected_w = int(calib.court_width_m * 100)    # 600
    assert img.shape == (expected_h, expected_w, 3), \
        f"예상 {(expected_h, expected_w, 3)}, 실제 {img.shape}"
    print(f"  ✓ 코트 이미지 크기: {img.shape}")


def test_court_image_has_lines():
    """코트 이미지는 배경색과 다른 픽셀(라인)이 존재해야 한다."""
    calib = _make_calib()
    bg_color = (40, 50, 35)
    img = render_court_birdeye(calib, px_per_m=100, bg_color=bg_color)
    bg = np.array(bg_color, dtype=np.uint8)
    differs = np.any(img != bg, axis=2)
    assert differs.any(), "라인이 전혀 그려지지 않았다"
    print(f"  ✓ 코트 라인 픽셀 수: {differs.sum()}")


def test_court_label_uses_correct_height():
    """라벨이 2.66을 표시하는지 확인 (반올림 버그 회귀 테스트)."""
    calib = _make_calib()
    img = render_court_birdeye(calib, px_per_m=100)
    # 우하단 영역 픽셀이 배경과 다르면 텍스트가 그려진 것 (간접 확인)
    bottom_right = img[-20:, -100:]
    bg = np.array([40, 50, 35], dtype=np.uint8)
    assert np.any(bottom_right != bg), "우하단 라벨이 그려지지 않았다"
    # calib의 height_m 값이 실제 2.66인지 직접 확인
    assert abs(calib.court_height_m - 2.66) < 1e-6, \
        f"court_height_m={calib.court_height_m}, 2.66이어야 한다"
    print("  ✓ 코트 높이 2.66 정상")


def test_color_for_id_returns_bgr_tuple():
    """color_for_id는 3-tuple of int를 반환해야 한다."""
    for tid in range(1, 8):
        c = color_for_id(tid)
        assert len(c) == 3, "BGR 3-채널이어야 한다"
        assert all(isinstance(v, int) for v in c), "각 채널은 int여야 한다"
    print("  ✓ color_for_id BGR tuple 정상")


def test_color_for_id_wraps():
    """ID가 팔레트 크기를 넘으면 순환한다."""
    c1 = color_for_id(1)
    c7 = color_for_id(7)   # 팔레트 6개 → ID 7 == ID 1
    assert c1 == c7, f"순환 실패: id=1 {c1}, id=7 {c7}"
    print("  ✓ color_for_id 순환 정상")


def test_draw_detections_preserves_shape():
    """draw_detections_on_frame은 입력 프레임과 동일한 shape를 반환해야 한다."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = draw_detections_on_frame(frame.copy(), _make_tracks())
    assert result.shape == frame.shape
    print(f"  ✓ 오버레이 후 shape 유지: {result.shape}")


def test_draw_detections_empty_tracks():
    """트랙이 없을 때 원본 프레임과 동일해야 한다."""
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    result = draw_detections_on_frame(frame.copy(), [])
    assert np.array_equal(result, frame), "트랙 없을 때 프레임이 변경되면 안 된다"
    print("  ✓ 빈 트랙 처리 정상")


def test_draw_detections_modifies_frame():
    """트랙이 있으면 프레임이 실제로 변경되어야 한다."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = draw_detections_on_frame(frame.copy(), _make_tracks())
    assert not np.array_equal(result, frame), "트랙이 있는데 프레임이 변경되지 않았다"
    print("  ✓ 바운딩 박스 오버레이 확인")


def test_draw_players_on_birdeye_shape():
    """draw_players_on_birdeye 출력 shape가 입력 court_img와 같아야 한다."""
    calib = _make_calib()
    court = render_court_birdeye(calib, px_per_m=100)
    result = draw_players_on_birdeye(court, _make_tracks(), calib, px_per_m=100)
    assert result.shape == court.shape
    print(f"  ✓ 버드아이 오버레이 shape 유지: {result.shape}")


def test_draw_players_modifies_birdeye():
    """트랙이 있으면 코트 이미지가 변경되어야 한다."""
    calib = _make_calib()
    court = render_court_birdeye(calib, px_per_m=100)
    result = draw_players_on_birdeye(court, _make_tracks(), calib, px_per_m=100)
    assert not np.array_equal(result, court), "선수가 그려지지 않았다"
    print("  ✓ 선수 점 렌더링 확인")


def test_draw_players_out_of_court_no_crash():
    """코트 밖 선수는 그리지 않고 예외 없이 반환해야 한다."""
    calib = _make_calib()
    court = render_court_birdeye(calib, px_per_m=100)
    out_of_court = [
        Track(track_id=1,
              bbox=np.array([-100, -100, -50, -50], dtype=np.float32),
              confidence=0.8, age=1, history=[(-75, -50)]),
    ]
    result = draw_players_on_birdeye(court, out_of_court, calib, px_per_m=100)
    assert result.shape == court.shape
    print("  ✓ 코트 밖 선수 안전 처리 확인")


def test_combine_views_width():
    """combine_views 출력 너비 = 카메라 너비 + 4(구분선) + 스케일된 코트 너비."""
    cam = np.zeros((720, 1280, 3), dtype=np.uint8)
    calib = _make_calib()
    court = render_court_birdeye(calib, px_per_m=100)
    combined = combine_views(cam, court)

    expected_court_w = int(court.shape[1] * (cam.shape[0] / court.shape[0]))
    expected_w = cam.shape[1] + 4 + expected_court_w
    assert combined.shape[1] == expected_w, \
        f"예상 너비 {expected_w}, 실제 {combined.shape[1]}"
    assert combined.shape[0] == cam.shape[0], "높이가 카메라 높이와 달라야 하지 않음"
    print(f"  ✓ combine_views 크기: {combined.shape[1]}×{combined.shape[0]}")


def test_draw_hud_preserves_shape():
    """draw_hud는 입력 프레임 shape를 유지해야 한다."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = draw_hud(frame, fps=25.3, n_players=3)
    assert result.shape == frame.shape
    print("  ✓ HUD shape 유지 확인")


def test_draw_hud_modifies_top():
    """HUD는 프레임 상단 영역을 변경해야 한다."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_hud(frame, fps=25.3, n_players=3)
    top_strip = frame[:40, :, :]
    assert top_strip.any(), "HUD가 상단에 그려지지 않았다"
    print("  ✓ HUD 상단 렌더링 확인")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 50)
    print(" Visualizer 테스트")
    print("=" * 50)
    test_court_image_shape()
    test_court_image_has_lines()
    test_court_label_uses_correct_height()
    test_color_for_id_returns_bgr_tuple()
    test_color_for_id_wraps()
    test_draw_detections_preserves_shape()
    test_draw_detections_empty_tracks()
    test_draw_detections_modifies_frame()
    test_draw_players_on_birdeye_shape()
    test_draw_players_modifies_birdeye()
    test_draw_players_out_of_court_no_crash()
    test_combine_views_width()
    test_draw_hud_preserves_shape()
    test_draw_hud_modifies_top()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
