"""TacticEngine 단위 테스트.

pytest 또는 단독 실행: python -m tests.test_tactic_engine

커버리지:
  - 헬퍼 함수 (_dist, _direction_name, _depth_in_own_half)
  - PlayerState / TacticAdvice 프로퍼티
  - 팀 배정 로직
  - 규칙 R1~R5 발화 조건 + 미발화 조건
  - 규칙 우선순위 (R5 > R3 > R1 > R4 > R2 > BASE)
  - analyze() 인터페이스
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tactic_engine import (
    TacticAdvice,
    TacticEngine,
    PlayerState,
    _direction_name,
    _dist,
)


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _engine() -> TacticEngine:
    return TacticEngine(
        court_width_m=6.0,
        court_height_m=2.66,
        spacing_min_m=1.0,
        counter_range_m=2.5,
        gap_min_m=2.0,
        backline_depth_m=1.5,
    )


def _ps(tid, x, y) -> PlayerState:
    return PlayerState(track_id=tid, court_x=x, court_y=y)


# ── _dist ─────────────────────────────────────────────────────────────────────

def test_dist_zero():
    assert _dist((1.0, 2.0), (1.0, 2.0)) == 0.0
    print("  ✓ _dist: 같은 점 거리 0")


def test_dist_known_value():
    d = _dist((0.0, 0.0), (3.0, 4.0))
    assert abs(d - 5.0) < 1e-9
    print("  ✓ _dist: 3-4-5 직각삼각형")


def test_dist_symmetric():
    a, b = (1.5, 0.8), (3.2, 2.1)
    assert abs(_dist(a, b) - _dist(b, a)) < 1e-12
    print("  ✓ _dist: 대칭성")


# ── _direction_name ───────────────────────────────────────────────────────────

def test_direction_name_stationary():
    assert _direction_name(0.0, 0.0) == "현 위치 유지"
    assert _direction_name(0.1, 0.0) == "현 위치 유지"   # abs < 0.15
    print("  ✓ _direction_name: 정지 → 현 위치 유지")


def test_direction_name_right():
    assert _direction_name(1.0, 0.0) == "오른쪽"
    print("  ✓ _direction_name: 오른쪽")


def test_direction_name_left():
    assert _direction_name(-1.0, 0.0) == "왼쪽"
    print("  ✓ _direction_name: 왼쪽")


def test_direction_name_forward():
    # dy 음수 = 위쪽 = 앞으로 (y축 아래 증가 좌표계에서 y감소=앞)
    assert _direction_name(0.0, -1.0) == "앞으로"
    print("  ✓ _direction_name: 앞으로")


def test_direction_name_backward():
    assert _direction_name(0.0, 1.0) == "뒤로"
    print("  ✓ _direction_name: 뒤로")


def test_direction_name_diagonals():
    assert _direction_name(1.0, -1.0) == "오른쪽 앞"
    assert _direction_name(-1.0, -1.0) == "왼쪽 앞"
    assert _direction_name(1.0, 1.0) == "오른쪽 뒤"
    assert _direction_name(-1.0, 1.0) == "왼쪽 뒤"
    print("  ✓ _direction_name: 사선 4방향")


# ── PlayerState / TacticAdvice 프로퍼티 ───────────────────────────────────────

def test_player_state_pos():
    p = _ps(1, 2.5, 1.3)
    assert p.pos == (2.5, 1.3)
    print("  ✓ PlayerState.pos 정상")


def test_advice_distance_m():
    a = TacticAdvice(
        track_id=1, team="A",
        current_pos=(0.0, 0.0), target_pos=(3.0, 4.0),
        urgency="MID", reason="", voice_message="", rule="R4",
    )
    assert abs(a.distance_m - 5.0) < 1e-9
    print("  ✓ TacticAdvice.distance_m 정상")


def test_advice_direction_vec_unit():
    a = TacticAdvice(
        track_id=1, team="A",
        current_pos=(1.0, 1.0), target_pos=(4.0, 5.0),
        urgency="LOW", reason="", voice_message="", rule="BASE",
    )
    vx, vy = a.direction_vec
    mag = math.hypot(vx, vy)
    assert abs(mag - 1.0) < 1e-9, f"단위벡터 크기 {mag}"
    print("  ✓ TacticAdvice.direction_vec 단위벡터")


def test_advice_direction_vec_zero_when_stationary():
    a = TacticAdvice(
        track_id=1, team="A",
        current_pos=(2.0, 1.0), target_pos=(2.0, 1.0),
        urgency="LOW", reason="", voice_message="", rule="BASE",
    )
    assert a.direction_vec == (0.0, 0.0)
    print("  ✓ TacticAdvice.direction_vec: 이동 없으면 (0,0)")


# ── 팀 배정 ───────────────────────────────────────────────────────────────────

def test_team_left_is_A():
    e = _engine()
    e.assign_team(1, court_x=1.0)
    assert e.get_team(1) == "A"
    print("  ✓ 왼쪽(x<3) → Team A")


def test_team_right_is_B():
    e = _engine()
    e.assign_team(1, court_x=5.0)
    assert e.get_team(1) == "B"
    print("  ✓ 오른쪽(x≥3) → Team B")


def test_team_boundary_is_B():
    e = _engine()
    e.assign_team(1, court_x=3.0)   # exactly half → B
    assert e.get_team(1) == "B"
    print("  ✓ 경계값(x=3.0) → Team B")


def test_team_assignment_not_overwritten():
    """한 번 배정된 팀은 위치가 바뀌어도 유지된다."""
    e = _engine()
    e.assign_team(1, court_x=1.0)   # → A
    e.assign_team(1, court_x=5.0)   # 재호출 → 변경 없어야 함
    assert e.get_team(1) == "A"
    print("  ✓ 팀 배정 고정(재호출 무시)")


def test_reset_teams():
    e = _engine()
    e.assign_team(1, court_x=1.0)
    e.reset_teams()
    assert e.get_team(1) == "A"   # 기본값 "A"
    assert 1 not in e._team_assignment
    print("  ✓ reset_teams 후 배정 초기화")


def test_get_team_default_is_A():
    e = _engine()
    assert e.get_team(99) == "A"   # 미등록 ID
    print("  ✓ 미등록 ID get_team 기본값 A")


# ── analyze 인터페이스 ─────────────────────────────────────────────────────────

def test_analyze_empty():
    e = _engine()
    result = e.analyze([])
    assert result == []
    print("  ✓ analyze([]) → 빈 리스트")


def test_analyze_returns_one_advice_per_player():
    e = _engine()
    players = [_ps(1, 1.0, 1.0), _ps(2, 1.5, 1.8), _ps(3, 4.5, 1.0)]
    advices = e.analyze(players)
    assert len(advices) == 3
    ids = {a.track_id for a in advices}
    assert ids == {1, 2, 3}
    print("  ✓ analyze: 선수당 advice 1개")


def test_analyze_single_player_no_crash():
    e = _engine()
    advices = e.analyze([_ps(1, 1.5, 1.0)])
    assert len(advices) == 1
    assert advices[0].rule == "BASE"
    print("  ✓ 선수 1명 단독 → BASE 규칙")


def test_analyze_team_assignment_auto():
    e = _engine()
    players = [_ps(1, 1.0, 1.0), _ps(2, 5.0, 1.0)]
    e.analyze(players)
    assert e.get_team(1) == "A"
    assert e.get_team(2) == "B"
    print("  ✓ analyze 호출 시 팀 자동 배정")


# ── R1: Spacing ───────────────────────────────────────────────────────────────

def test_r1_fires_when_teammates_too_close():
    """같은 팀 두 선수가 1.0m 이내 → R1 발화."""
    e = _engine()
    # 두 선수 거리 ≈ 0.14m < 1.0m
    players = [
        _ps(1, 1.5, 1.0),
        _ps(2, 1.6, 1.1),
        _ps(3, 4.5, 1.3),   # 상대팀 (멀리)
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    # #1 또는 #2 중 하나 이상 R1
    r1_fired = any(a.rule == "R1" for a in [advices[1], advices[2]])
    assert r1_fired, f"R1 미발화: {advices[1].rule}, {advices[2].rule}"
    print("  ✓ R1 발화: 팀원 0.14m 근접")


def test_r1_urgency_is_mid():
    e = _engine()
    players = [_ps(1, 1.5, 1.0), _ps(2, 1.6, 1.1)]
    advices = {a.track_id: a for a in e.analyze(players)}
    r1_advices = [a for a in advices.values() if a.rule == "R1"]
    assert all(a.urgency == "MID" for a in r1_advices)
    print("  ✓ R1 긴급도 MID")


def test_r1_no_fire_when_teammates_far():
    """팀원 거리 > 1.0m → R1 미발화."""
    e = _engine()
    # 두 선수 거리 ≈ 1.1m (상대팀 없음 → 팀끼리만 비교)
    players = [_ps(1, 1.0, 1.0), _ps(2, 2.1, 1.0)]
    advices = {a.track_id: a for a in e.analyze(players)}
    # 상대팀 없으면 R4(gap) 조건도 미충족 → R2 or BASE
    assert advices[1].rule != "R1"
    assert advices[2].rule != "R1"
    print("  ✓ R1 미발화: 팀원 1.1m 이격")


# ── R2: Coverage ──────────────────────────────────────────────────────────────

def test_r2_fires_when_team_bunched_in_y():
    """같은 팀이 y축으로 0.8m 이내에 밀집 → R2 발화."""
    e = _engine()
    # 팀A: x축 이격(R1 미발화)이지만 y축 밀집(spread=0)
    # 상대: 멀리 배치, gap 작음(R4 미발화)
    players = [
        _ps(1, 1.0, 1.0),
        _ps(2, 2.1, 1.0),   # dist=1.1 → R1 없음, y동일 → spread=0 → R2
        _ps(3, 4.5, 0.8),
        _ps(4, 4.5, 1.5),   # gap=0.7 < 2.0 → R4 없음
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R2" or advices[2].rule == "R2", \
        f"R2 미발화: #{1}={advices[1].rule}, #{2}={advices[2].rule}"
    print("  ✓ R2 발화: 팀 y축 밀집(spread=0)")


def test_r2_no_fire_when_spread():
    """팀이 y축으로 충분히 퍼져 있으면 R2 미발화."""
    e = _engine()
    players = [
        _ps(1, 1.0, 0.3),
        _ps(2, 2.1, 2.3),   # y spread = 2.0 > 0.8
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule != "R2"
    assert advices[2].rule != "R2"
    print("  ✓ R2 미발화: y축 충분히 분산")


# ── R3: Counter ───────────────────────────────────────────────────────────────

def test_r3_fires_when_frontal_threat():
    """정면 2.5m 이내 적 → R3 발화."""
    e = _engine()
    players = [
        _ps(1, 2.0, 1.3),   # Team A
        _ps(2, 3.5, 1.3),   # Team B, dist=1.5 < 2.5, |dy|=0 < 0.6
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R3", f"R3 미발화: {advices[1].rule}"
    print("  ✓ R3 발화: 정면 1.5m 위협")


def test_r3_urgency_is_high():
    e = _engine()
    players = [_ps(1, 2.0, 1.3), _ps(2, 3.5, 1.3)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].urgency == "HIGH"
    print("  ✓ R3 긴급도 HIGH")


def test_r3_target_moves_lateral():
    """R3 목표 위치: x는 후퇴, y는 측면 이동."""
    e = _engine()
    players = [
        _ps(1, 2.0, 0.5),   # 위쪽에 있는 Team A → 아래쪽으로 피함
        _ps(2, 3.5, 0.5),   # Team B 정면 위협
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    a1 = advices[1]
    assert a1.rule == "R3"
    # y가 0.5(위쪽)이므로 court_height/2(=1.33)보다 작 → avoid_y = court_h - 0.4 = 2.26
    assert a1.target_pos[1] > a1.current_pos[1], "측면 이동(y 증가) 기대"
    print("  ✓ R3 목표: 측면 y 이동 확인")


def test_r3_no_fire_when_far():
    """적이 2.5m 초과 → R3 미발화."""
    e = _engine()
    players = [
        _ps(1, 1.5, 1.3),
        _ps(2, 4.1, 1.3),   # dist=2.6 > 2.5
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule != "R3", f"R3 오발화: {advices[1].rule}"
    print("  ✓ R3 미발화: 적 2.6m 이격")


def test_r3_no_fire_when_off_axis():
    """적이 가깝지만 y축 오프셋 > 0.6 → R3 미발화."""
    e = _engine()
    players = [
        _ps(1, 2.0, 1.0),
        _ps(2, 3.5, 1.7),   # dist≈1.55 < 2.5 but |dy|=0.7 > 0.6
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule != "R3"
    print("  ✓ R3 미발화: y오프셋 0.7 > 0.6")


# ── R4: Gap Attack ────────────────────────────────────────────────────────────

def test_r4_fires_when_gap_large():
    """적 팀 사이 y 갭 > 2.0m → R4 발화."""
    e = _engine()
    players = [
        _ps(1, 1.5, 1.3),   # Team A
        _ps(2, 4.5, 0.2),   # Team B
        _ps(3, 4.5, 2.4),   # Team B — gap = 2.2 > 2.0
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R4", f"R4 미발화: {advices[1].rule}"
    print("  ✓ R4 발화: 적 y갭 2.2m")


def test_r4_urgency_is_mid():
    e = _engine()
    players = [_ps(1, 1.5, 1.3), _ps(2, 4.5, 0.2), _ps(3, 4.5, 2.4)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].urgency == "MID"
    print("  ✓ R4 긴급도 MID")


def test_r4_no_fire_when_gap_small():
    """적 갭 < 2.0m → R4 미발화."""
    e = _engine()
    players = [
        _ps(1, 1.5, 1.3),
        _ps(2, 4.5, 0.5),
        _ps(3, 4.5, 2.4),   # gap = 1.9 < 2.0
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule != "R4", f"R4 오발화: {advices[1].rule}"
    print("  ✓ R4 미발화: 적 y갭 1.9m")


def test_r4_target_advances_toward_gap():
    """R4 목표 x좌표: Team A는 오른쪽으로 전진."""
    e = _engine()
    players = [
        _ps(1, 1.5, 1.3),
        _ps(2, 4.5, 0.2),
        _ps(3, 4.5, 2.4),
    ]
    advices = {a.track_id: a for a in e.analyze(players)}
    a1 = advices[1]
    assert a1.rule == "R4"
    assert a1.target_pos[0] >= a1.current_pos[0], "Team A R4: x 전진 기대"
    print("  ✓ R4 Team A: x 전진 방향 확인")


# ── R5: Backline ──────────────────────────────────────────────────────────────

def test_r5_fires_when_two_opponents_deep():
    """적 2명이 자기 진영 깊이 침투(depth > 1.5m) → R5 발화."""
    e = _engine()
    # 1프레임: 팀 배정 확립 (B선수를 오른쪽에서 시작)
    e.analyze([_ps(1, 2.0, 1.3), _ps(2, 4.5, 0.7), _ps(3, 4.5, 2.0)])
    # 2프레임: B선수가 A 진영 깊이 침투 (depth = 3.0 - 0.8 = 2.2 > 1.5)
    players = [_ps(1, 2.0, 1.3), _ps(2, 0.8, 0.7), _ps(3, 0.8, 2.0)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R5", f"R5 미발화: {advices[1].rule}"
    print("  ✓ R5 발화: 적 2명 진영 깊이 침투")


def test_r5_urgency_is_high():
    e = _engine()
    e.analyze([_ps(1, 2.0, 1.3), _ps(2, 4.5, 0.7), _ps(3, 4.5, 2.0)])
    players = [_ps(1, 2.0, 1.3), _ps(2, 0.8, 0.7), _ps(3, 0.8, 2.0)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].urgency == "HIGH"
    print("  ✓ R5 긴급도 HIGH")


def test_r5_no_fire_when_only_one_deep():
    """적 1명만 깊이 침투 → R5 미발화."""
    e = _engine()
    e.analyze([_ps(1, 2.0, 1.3), _ps(2, 4.5, 1.3), _ps(3, 4.5, 1.5)])
    # #2 깊이 침투 (depth=2.2), #3 얕게 (x=2.5, depth=0.5 < 1.5)
    players = [_ps(1, 2.0, 1.3), _ps(2, 0.8, 1.3), _ps(3, 2.5, 1.3)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule != "R5", f"R5 오발화: {advices[1].rule}"
    print("  ✓ R5 미발화: 적 1명만 침투")


def test_r5_team_b_fires():
    """Team B 관점에서도 R5 정상 발화."""
    e = _engine()
    # Team B 진영: x > 3.0, 깊이 = x - 3.0 > 1.5 → x > 4.5
    # 팀 배정: #1=B, #2&#3=A 먼저 확립
    e.analyze([_ps(1, 5.0, 1.3), _ps(2, 1.5, 0.5), _ps(3, 1.5, 2.1)])
    # A팀 선수들이 B 진영 깊이 침투 (depth = 4.8 - 3.0 = 1.8 > 1.5)
    players = [_ps(1, 5.0, 1.3), _ps(2, 4.8, 0.5), _ps(3, 4.8, 2.1)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R5", f"Team B R5 미발화: {advices[1].rule}"
    print("  ✓ R5 Team B 관점 발화")


# ── _depth_in_own_half ────────────────────────────────────────────────────────

def test_depth_team_a():
    e = _engine()
    assert abs(e._depth_in_own_half(0.5, "A") - 2.5) < 1e-9
    assert abs(e._depth_in_own_half(2.0, "A") - 1.0) < 1e-9
    print("  ✓ _depth_in_own_half Team A 정상")


def test_depth_team_b():
    e = _engine()
    assert abs(e._depth_in_own_half(4.0, "B") - 1.0) < 1e-9
    assert abs(e._depth_in_own_half(5.5, "B") - 2.5) < 1e-9
    print("  ✓ _depth_in_own_half Team B 정상")


# ── 우선순위 ──────────────────────────────────────────────────────────────────

def test_priority_r5_over_r3():
    """R5 조건 + R3 조건 동시 성립 → R5가 우선."""
    e = _engine()
    e.analyze([_ps(1, 1.8, 1.0), _ps(2, 4.5, 1.0), _ps(3, 4.5, 2.0)])
    # B선수들이 A 진영 깊이 침투(R5) + #2는 정면 근접(R3도 성립)
    players = [_ps(1, 1.8, 1.0), _ps(2, 0.6, 1.0), _ps(3, 0.6, 2.0)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R5", f"우선순위 실패: {advices[1].rule}"
    print("  ✓ 우선순위: R5 > R3")


def test_priority_r3_over_r1():
    """R3 조건 + R1 조건 동시 성립 → R3가 우선."""
    e = _engine()
    # #3을 Team B로 먼저 배정
    e.analyze([_ps(1, 1.5, 1.0), _ps(2, 1.5, 1.8), _ps(3, 4.5, 1.0)])
    # #3이 A 진영으로 이동: dist=1.2 < 2.5, |dy|=0 → R3 성립
    # #1/#2 거리 0.14m → R1도 성립 → R3가 이겨야 함
    players = [_ps(1, 1.5, 1.0), _ps(2, 1.6, 1.1), _ps(3, 2.7, 1.0)]
    advices = {a.track_id: a for a in e.analyze(players)}
    assert advices[1].rule == "R3", f"우선순위 실패: {advices[1].rule}"
    print("  ✓ 우선순위: R3 > R1")


def test_base_returns_current_position():
    """규칙 미발화 시 target = current_pos."""
    e = _engine()
    players = [_ps(1, 1.5, 1.3)]   # 혼자, 적 없음
    advices = e.analyze(players)
    a = advices[0]
    assert a.rule == "BASE"
    assert a.target_pos == a.current_pos
    print("  ✓ BASE: target = current_pos")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 55)
    print(" TacticEngine 테스트")
    print("=" * 55)
    test_dist_zero()
    test_dist_known_value()
    test_dist_symmetric()
    test_direction_name_stationary()
    test_direction_name_right()
    test_direction_name_left()
    test_direction_name_forward()
    test_direction_name_backward()
    test_direction_name_diagonals()
    test_player_state_pos()
    test_advice_distance_m()
    test_advice_direction_vec_unit()
    test_advice_direction_vec_zero_when_stationary()
    test_team_left_is_A()
    test_team_right_is_B()
    test_team_boundary_is_B()
    test_team_assignment_not_overwritten()
    test_reset_teams()
    test_get_team_default_is_A()
    test_analyze_empty()
    test_analyze_returns_one_advice_per_player()
    test_analyze_single_player_no_crash()
    test_analyze_team_assignment_auto()
    test_r1_fires_when_teammates_too_close()
    test_r1_urgency_is_mid()
    test_r1_no_fire_when_teammates_far()
    test_r2_fires_when_team_bunched_in_y()
    test_r2_no_fire_when_spread()
    test_r3_fires_when_frontal_threat()
    test_r3_urgency_is_high()
    test_r3_target_moves_lateral()
    test_r3_no_fire_when_far()
    test_r3_no_fire_when_off_axis()
    test_r4_fires_when_gap_large()
    test_r4_urgency_is_mid()
    test_r4_no_fire_when_gap_small()
    test_r4_target_advances_toward_gap()
    test_r5_fires_when_two_opponents_deep()
    test_r5_urgency_is_high()
    test_r5_no_fire_when_only_one_deep()
    test_r5_team_b_fires()
    test_depth_team_a()
    test_depth_team_b()
    test_priority_r5_over_r3()
    test_priority_r3_over_r1()
    test_base_returns_current_position()
    print(f"\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
