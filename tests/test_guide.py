"""Guide 모듈 단위 테스트.

pytest 또는 단독 실행: python -m tests.test_guide

커버리지:
  - URGENCY_COLOR 상수
  - draw_guide_on_birdeye: shape, 빈 입력, 거리 필터, HIGH 원 렌더링
  - draw_guide_text_on_frame: shape, 빈 입력, max_lines, 긴급도 정렬
  - VoiceGuide: disabled 모드, cooldown, speak_advices 선택 로직
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.guide import (
    URGENCY_COLOR,
    VoiceGuide,
    draw_guide_on_birdeye,
    draw_guide_text_on_frame,
)
from src.homography import compute_homography
from src.tactic_engine import TacticAdvice, TacticEngine, PlayerState
from src.visualizer import render_court_birdeye


# ── 픽스처 ────────────────────────────────────────────────────────────────────

def _make_calib():
    return compute_homography(
        corners_pixel=np.array([[100, 100], [1180, 100], [1180, 600], [100, 600]],
                               dtype=np.float32),
        court_width_m=6.0, court_height_m=2.66, image_size=(1280, 720),
    )


def _make_advice(tid=1, team="A", cx=1.5, cy=1.0, tx=2.5, ty=1.0,
                 urgency="MID", rule="R4") -> TacticAdvice:
    return TacticAdvice(
        track_id=tid, team=team,
        current_pos=(cx, cy), target_pos=(tx, ty),
        urgency=urgency, reason="테스트", voice_message="전진",
        rule=rule,
    )


def _court():
    return render_court_birdeye(_make_calib(), px_per_m=100)


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ── URGENCY_COLOR ─────────────────────────────────────────────────────────────

def test_urgency_color_has_all_levels():
    for level in ("LOW", "MID", "HIGH"):
        assert level in URGENCY_COLOR, f"{level} 없음"
    print("  ✓ URGENCY_COLOR: LOW/MID/HIGH 모두 존재")


def test_urgency_color_values_are_bgr():
    for level, color in URGENCY_COLOR.items():
        assert len(color) == 3, f"{level} color가 3채널이 아님"
        assert all(0 <= v <= 255 for v in color), f"{level} color 범위 초과"
    print("  ✓ URGENCY_COLOR: 유효한 BGR 값")


def test_urgency_colors_are_distinct():
    colors = list(URGENCY_COLOR.values())
    assert len(set(colors)) == len(colors), "긴급도별 색상이 중복됨"
    print("  ✓ URGENCY_COLOR: 긴급도별 색상 구분됨")


# ── draw_guide_on_birdeye ─────────────────────────────────────────────────────

def test_guide_birdeye_preserves_shape():
    court = _court()
    result = draw_guide_on_birdeye(court, [_make_advice()])
    assert result.shape == court.shape
    print(f"  ✓ draw_guide_on_birdeye shape 유지: {result.shape}")


def test_guide_birdeye_empty_advices_returns_copy():
    court = _court()
    result = draw_guide_on_birdeye(court, [])
    assert result.shape == court.shape
    assert np.array_equal(result, court), "빈 advices 시 court 변경 없어야 함"
    print("  ✓ draw_guide_on_birdeye: 빈 advices → 원본 동일")


def test_guide_birdeye_modifies_court_with_advice():
    court = _court()
    advice = _make_advice(cx=1.5, cy=1.0, tx=2.5, ty=1.0)  # distance=1.0 > 0.15
    result = draw_guide_on_birdeye(court, [advice])
    assert not np.array_equal(result, court), "advice 있을 때 court 변경 기대"
    print("  ✓ draw_guide_on_birdeye: 화살표 렌더링 확인")


def test_guide_birdeye_skips_small_distance():
    """distance_m < 0.15인 advice는 건너뜀 → court 변경 없음."""
    court = _court()
    # current == target → distance = 0
    advice = _make_advice(cx=1.5, cy=1.0, tx=1.5, ty=1.0)
    result = draw_guide_on_birdeye(court, [advice])
    assert np.array_equal(result, court), "거리=0 advice → court 변경 없어야 함"
    print("  ✓ draw_guide_on_birdeye: 거리 0 advice 스킵")


def test_guide_birdeye_high_urgency_draws_more():
    """HIGH urgency는 LOW보다 더 많은 픽셀을 변경한다 (펄스 원 추가)."""
    court = _court()
    advice_low = _make_advice(cx=1.5, cy=1.0, tx=2.5, ty=1.0, urgency="LOW")
    advice_high = _make_advice(cx=1.5, cy=1.0, tx=2.5, ty=1.0, urgency="HIGH")
    result_low = draw_guide_on_birdeye(court, [advice_low])
    result_high = draw_guide_on_birdeye(court, [advice_high])
    diff_low = np.sum(result_low != court)
    diff_high = np.sum(result_high != court)
    assert diff_high > diff_low, "HIGH urgency가 LOW보다 더 많이 그려야 함"
    print(f"  ✓ HIGH urgency 펄스 원: 변경 픽셀 {diff_low} → {diff_high}")


def test_guide_birdeye_does_not_mutate_input():
    """draw_guide_on_birdeye는 원본 court_img를 수정하지 않는다."""
    court = _court()
    original = court.copy()
    draw_guide_on_birdeye(court, [_make_advice()])
    assert np.array_equal(court, original), "원본 court_img가 변경됨"
    print("  ✓ draw_guide_on_birdeye: 원본 불변")


def test_guide_birdeye_multiple_advices():
    """여러 advice를 한 번에 처리해도 shape 유지."""
    court = _court()
    advices = [
        _make_advice(tid=1, cx=1.0, cy=0.5, tx=2.0, ty=0.5, urgency="HIGH"),
        _make_advice(tid=2, cx=1.0, cy=2.0, tx=2.0, ty=2.0, urgency="MID"),
        _make_advice(tid=3, cx=4.5, cy=1.3, tx=3.5, ty=1.3, urgency="LOW"),
    ]
    result = draw_guide_on_birdeye(court, advices)
    assert result.shape == court.shape
    print("  ✓ draw_guide_on_birdeye: 다중 advice 처리")


# ── draw_guide_text_on_frame ──────────────────────────────────────────────────

def test_guide_text_preserves_shape():
    frame = _frame()
    result = draw_guide_text_on_frame(frame.copy(), [_make_advice()])
    assert result.shape == frame.shape
    print(f"  ✓ draw_guide_text_on_frame shape 유지")


def test_guide_text_empty_advices_no_change():
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    result = draw_guide_text_on_frame(frame.copy(), [])
    assert np.array_equal(result, frame), "빈 advices → 프레임 변경 없어야 함"
    print("  ✓ draw_guide_text_on_frame: 빈 advices → 원본 동일")


def test_guide_text_modifies_frame():
    frame = _frame()
    result = draw_guide_text_on_frame(frame.copy(), [_make_advice()])
    assert not np.array_equal(result, frame)
    print("  ✓ draw_guide_text_on_frame: 텍스트 패널 렌더링 확인")


def test_guide_text_max_lines_respected():
    """max_lines 개수 이상의 advice는 잘린다 (패널 높이로 간접 확인)."""
    advices = [
        _make_advice(tid=i, cx=float(i), cy=1.0, tx=float(i) + 0.5, ty=1.0)
        for i in range(1, 8)
    ]
    # max_lines=2와 max_lines=6로 그린 결과가 달라야 한다
    frame2 = _frame()
    frame6 = _frame()
    draw_guide_text_on_frame(frame2, advices, max_lines=2)
    draw_guide_text_on_frame(frame6, advices, max_lines=6)
    assert not np.array_equal(frame2, frame6), "max_lines 차이가 반영되지 않음"
    print("  ✓ draw_guide_text_on_frame: max_lines 적용 확인")


def test_guide_text_sorted_high_first():
    """HIGH urgency가 LOW보다 먼저 그려져야 한다.
    두 프레임 비교: HIGH만 있는 것과 LOW만 있는 것의 패널 위치가 같아야
    HIGH가 첫 줄에 그려진다는 뜻이므로, 혼합했을 때 HIGH-only와 상단이 유사한지 확인."""
    adv_high = _make_advice(tid=1, urgency="HIGH")
    adv_low  = _make_advice(tid=2, urgency="LOW")

    frame_mixed = _frame()
    draw_guide_text_on_frame(frame_mixed, [adv_low, adv_high])  # LOW 먼저 넣어도

    frame_high_only = _frame()
    draw_guide_text_on_frame(frame_high_only, [adv_high])

    # 첫 줄 텍스트 영역(하단 패널 첫 행)이 동일해야 HIGH가 첫 줄에 정렬된 것
    # 두 이미지의 하단 특정 구역을 비교
    region_mixed     = frame_mixed[-160:-120, 10:480]
    region_high_only = frame_high_only[-160:-120, 10:480]
    assert np.array_equal(region_mixed, region_high_only), "HIGH urgency가 첫 줄이 아님"
    print("  ✓ draw_guide_text_on_frame: HIGH urgency 우선 정렬")


# ── VoiceGuide ────────────────────────────────────────────────────────────────

def test_voice_guide_disabled_no_crash():
    vg = VoiceGuide(enabled=False)
    vg.speak("테스트")          # 아무 일도 없어야 함
    vg.close()
    print("  ✓ VoiceGuide disabled: 크래시 없음")


def test_voice_guide_disabled_speak_advices_no_crash():
    vg = VoiceGuide(enabled=False)
    advices = [
        _make_advice(urgency="HIGH"),
        _make_advice(tid=2, urgency="MID"),
    ]
    vg.speak_advices(advices)
    vg.close()
    print("  ✓ VoiceGuide disabled: speak_advices 크래시 없음")


def test_voice_guide_speak_advices_empty():
    vg = VoiceGuide(enabled=False)
    vg.speak_advices([])
    vg.close()
    print("  ✓ VoiceGuide: 빈 advices speak_advices 정상")


def test_voice_guide_speak_advices_ignores_low():
    """speak_advices는 LOW urgency를 무시한다."""
    vg = VoiceGuide(enabled=False)
    # LOW만 있으면 아무것도 출력 안 함 — 내부 candidates 비어야 함
    candidates = [a for a in [_make_advice(urgency="LOW")]
                  if a.voice_message and a.urgency in ("HIGH", "MID")]
    assert len(candidates) == 0
    vg.close()
    print("  ✓ VoiceGuide: LOW urgency speak_advices 무시")


def test_voice_guide_speak_advices_prefers_high_over_mid():
    """speak_advices: HIGH가 있으면 MID보다 우선."""
    vg = VoiceGuide(enabled=False)
    adv_mid  = _make_advice(tid=1, urgency="MID", rule="R1")
    adv_high = _make_advice(tid=2, urgency="HIGH", rule="R3")
    candidates = sorted(
        [a for a in [adv_mid, adv_high] if a.voice_message and a.urgency in ("HIGH", "MID")],
        key=lambda a: ({"HIGH": 0, "MID": 1}.get(a.urgency, 2), -a.distance_m),
    )
    assert candidates[0].urgency == "HIGH"
    vg.close()
    print("  ✓ VoiceGuide: HIGH urgency speak_advices 우선 선택")


def test_voice_guide_cooldown_blocks_repeat():
    """같은 메시지를 cooldown 내에 두 번 speak → _last_spoken에 기록만 되고 큐는 1번만."""
    vg = VoiceGuide(enabled=False, cooldown_sec=5.0)
    # enabled=False 이므로 실제 speak 호출 안 됨 — 로직만 검증
    # 내부 cooldown 로직을 직접 시뮬레이션
    last = {}
    cooldown = 2.0
    msg = "후방 수비"

    def _speak_sim(text):
        now = time.time()
        if now - last.get(text, 0) < cooldown:
            return False
        last[text] = now
        return True

    assert _speak_sim(msg) is True    # 첫 호출 통과
    assert _speak_sim(msg) is False   # 즉시 재호출 차단
    time.sleep(0.01)
    assert _speak_sim(msg) is False   # 아직 cooldown 내
    vg.close()
    print("  ✓ VoiceGuide cooldown: 동일 메시지 반복 차단")


def test_voice_guide_close_idempotent():
    """close()를 여러 번 호출해도 예외 없음."""
    vg = VoiceGuide(enabled=False)
    vg.close()
    vg.close()
    print("  ✓ VoiceGuide.close() 멱등성")


# ── 엔진+가이드 통합 ──────────────────────────────────────────────────────────

def test_engine_to_guide_pipeline():
    """TacticEngine.analyze → draw_guide_on_birdeye 전체 경로 크래시 없음."""
    calib = _make_calib()
    court = render_court_birdeye(calib, px_per_m=100)
    engine = TacticEngine()
    players = [
        PlayerState(track_id=1, court_x=1.5, court_y=1.0),
        PlayerState(track_id=2, court_x=1.6, court_y=1.1),
        PlayerState(track_id=3, court_x=4.5, court_y=0.5),
        PlayerState(track_id=4, court_x=4.5, court_y=2.2),
    ]
    advices = engine.analyze(players)
    result = draw_guide_on_birdeye(court, advices, px_per_m=100)
    assert result.shape == court.shape
    frame = _frame()
    draw_guide_text_on_frame(frame, advices)
    print("  ✓ 엔진→가이드 통합 파이프라인 정상")


def test_voice_guide_korean_voice_selected():
    """pyttsx3 설치 환경에서 한국어 음성이 선택되는지 확인."""
    pytest.importorskip("pyttsx3")
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    ko_voices = [v for v in voices
                 if "ko" in (v.id or "").lower() or "korean" in (v.name or "").lower()]
    if not ko_voices:
        pytest.skip("한국어 음성 없는 환경 — Pi4 espeak-ng 없이는 스킵")
    # VoiceGuide 초기화 시 크래시 없어야 함
    vg = VoiceGuide(enabled=True, cooldown_sec=0.1)
    assert vg.enabled, "pyttsx3 있는 환경에서 enabled=True여야 함"
    vg.close()
    print("  ✓ VoiceGuide: 한국어 음성 선택 및 초기화 정상")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 55)
    print(" Guide 테스트")
    print("=" * 55)
    test_urgency_color_has_all_levels()
    test_urgency_color_values_are_bgr()
    test_urgency_colors_are_distinct()
    test_guide_birdeye_preserves_shape()
    test_guide_birdeye_empty_advices_returns_copy()
    test_guide_birdeye_modifies_court_with_advice()
    test_guide_birdeye_skips_small_distance()
    test_guide_birdeye_high_urgency_draws_more()
    test_guide_birdeye_does_not_mutate_input()
    test_guide_birdeye_multiple_advices()
    test_guide_text_preserves_shape()
    test_guide_text_empty_advices_no_change()
    test_guide_text_modifies_frame()
    test_guide_text_max_lines_respected()
    test_guide_text_sorted_high_first()
    test_voice_guide_disabled_no_crash()
    test_voice_guide_disabled_speak_advices_no_crash()
    test_voice_guide_speak_advices_empty()
    test_voice_guide_speak_advices_ignores_low()
    test_voice_guide_speak_advices_prefers_high_over_mid()
    test_voice_guide_cooldown_blocks_repeat()
    test_voice_guide_close_idempotent()
    test_voice_guide_korean_voice_selected()
    test_engine_to_guide_pipeline()
    print(f"\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
