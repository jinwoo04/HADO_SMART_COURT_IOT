"""Level 2 전술 분석 엔진.

선수 좌표(m 단위)를 입력받아 각 선수에게 전술 조언(추천 위치 + 이동 방향 + 음성 안내)을 생성.

설계 원칙
---------
1. **Rule-based 우선**: ML 없이도 의미 있는 가이드가 가능해야 한다.
2. **HADO 특화**: 10m × 6m 코트 (높이 2.66m), 3 vs 3 기준.
3. **확장 가능**: 누적 데이터가 쌓이면 ML 규칙으로 교체할 수 있게 인터페이스 분리.

기본 전술 규칙
-------------
- (R1) Spacing: 같은 팀끼리 1.0 m 이내로 붙으면 분산
- (R2) Coverage: 한 팀이 코트 반쪽을 동시에 비워두면 안 됨
- (R3) Counter: 정면 가까이(2.5m 이내)에 적이 있으면 측면 회피
- (R4) Gap Attack: 적팀 사이 공간(>2m)이 보이면 그쪽으로 전진
- (R5) Backline: 적이 자기 진영에 너무 깊이 들어오면 후방으로 후퇴
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------- 데이터 클래스 ----------
@dataclass
class PlayerState:
    """단일 선수의 현재 상태 (Level 2 입력)."""
    track_id: int
    court_x: float       # 코트 X 좌표 (m)
    court_y: float       # 코트 Y 좌표 (m)
    confidence: float = 1.0
    role: str = ""       # "technician" | "defender" | "main_attacker" | "" (미지정)

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.court_x, self.court_y)


@dataclass
class TacticAdvice:
    """단일 선수에게 주는 가이드 (Level 2 출력)."""
    track_id: int
    team: str                       # "A" (왼쪽) | "B" (오른쪽)
    current_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    urgency: str                    # "LOW" | "MID" | "HIGH"
    reason: str                     # 화면 텍스트용 한글
    voice_message: str              # 음성 안내용 짧은 한글
    rule: str                       # 어떤 규칙이 발동했는지 (R1~R5 | ML:패턴ID)
    ml_confidence: float = 0.0     # MovementModel 기여도 (0=미사용, >0=블렌딩됨)

    @property
    def direction_vec(self) -> Tuple[float, float]:
        """현재 → 목표 단위 벡터."""
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    @property
    def distance_m(self) -> float:
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        return math.hypot(dx, dy)


# ---------- 헬퍼 ----------
def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _direction_name(dx: float, dy: float) -> str:
    """이동 방향을 한글 8방위로 변환."""
    if abs(dx) < 0.15 and abs(dy) < 0.15:
        return "현 위치 유지"
    angle = math.degrees(math.atan2(dy, dx))
    # 0도 = 오른쪽(+x), 시계방향
    if -22.5 <= angle < 22.5:
        return "오른쪽"
    elif 22.5 <= angle < 67.5:
        return "오른쪽 뒤"
    elif 67.5 <= angle < 112.5:
        return "뒤로"
    elif 112.5 <= angle < 157.5:
        return "왼쪽 뒤"
    elif angle >= 157.5 or angle < -157.5:
        return "왼쪽"
    elif -157.5 <= angle < -112.5:
        return "왼쪽 앞"
    elif -112.5 <= angle < -67.5:
        return "앞으로"
    else:  # -67.5 ~ -22.5
        return "오른쪽 앞"


def _rule_to_context(rule: str) -> str:
    """발동된 규칙 → MovementModel context 문자열 변환."""
    if rule in ("R3", "R5"):
        return "defend"
    if rule == "R4":
        return "attack"
    if rule in ("R1", "R2"):
        return "transition"
    return "attack"   # BASE


# ---------- 엔진 ----------
class TacticEngine:
    """규칙 기반 전술 분석기.

    매 프레임 호출 가능. 팀 배정은 첫 등장 시 자동(코트 절반 기준).
    """

    # ML 블렌딩 가중치 상수
    _ML_WEIGHT_BASE = 0.7   # BASE(규칙 미발동) 시 ML 최대 기여 비율
    _ML_WEIGHT_MID  = 0.4   # MID urgency 시 ML 최대 기여 비율
    _ML_CONF_MIN    = 0.5   # 이 신뢰도 미만은 ML 무시

    def __init__(
        self,
        court_width_m: float = 10.0,
        court_height_m: float = 6.0,
        spacing_min_m: float = 1.5,
        counter_range_m: float = 3.0,
        counter_y_offset_m: float = 1.0,
        gap_min_m: float = 2.5,
        backline_depth_m: float = 2.0,
        coverage_spread_min_m: float = 1.5,
        movement_model=None,    # MovementModel | None (순환 import 방지로 타입 미지정)
    ):
        self.court_width = court_width_m
        self.court_height = court_height_m
        self.spacing_min = spacing_min_m
        self.counter_range = counter_range_m
        self.counter_y_offset = counter_y_offset_m
        self.gap_min = gap_min_m
        self.backline_depth = backline_depth_m
        self.coverage_spread_min = coverage_spread_min_m
        self._movement_model = movement_model
        self._team_assignment: Dict[int, str] = {}

    @classmethod
    def from_config(cls, config: dict, movement_model=None) -> "TacticEngine":
        """court_config.yaml 딕셔너리에서 파라미터를 읽어 인스턴스 생성."""
        court = config.get("court", {})
        tc    = config.get("tactic", {})
        return cls(
            court_width_m         = court.get("width_m",               10.0),
            court_height_m        = court.get("height_m",               6.0),
            spacing_min_m         = tc.get("spacing_min_m",             1.5),
            counter_range_m       = tc.get("counter_range_m",           3.0),
            counter_y_offset_m    = tc.get("counter_y_offset_m",        1.0),
            gap_min_m             = tc.get("gap_min_m",                 2.5),
            backline_depth_m      = tc.get("backline_depth_m",          2.0),
            coverage_spread_min_m = tc.get("coverage_spread_min_m",     1.5),
            movement_model        = movement_model,
        )

    # ----- 팀 배정 -----
    def assign_team(self, track_id: int, court_x: float):
        """첫 등장 시 코트 어느 쪽에 있냐로 팀 결정 후 고정."""
        if track_id in self._team_assignment:
            return
        self._team_assignment[track_id] = "A" if court_x < self.court_width / 2 else "B"

    def reset_teams(self):
        self._team_assignment.clear()

    def get_team(self, track_id: int) -> str:
        return self._team_assignment.get(track_id, "A")

    # ----- 메인 분석 -----
    def analyze(self, players: List[PlayerState]) -> List[TacticAdvice]:
        """프레임 단위 분석.

        Parameters
        ----------
        players : 모든 선수 상태 (양 팀)

        Returns
        -------
        list[TacticAdvice]  (선수당 1개)
        """
        # 팀 배정 확정
        for p in players:
            self.assign_team(p.track_id, p.court_x)

        teams: Dict[str, List[PlayerState]] = {"A": [], "B": []}
        for p in players:
            teams[self.get_team(p.track_id)].append(p)

        advices: List[TacticAdvice] = []
        for p in players:
            team = self.get_team(p.track_id)
            opp_team = "B" if team == "A" else "A"
            teammates = [t for t in teams[team] if t.track_id != p.track_id]
            opponents = teams[opp_team]

            advice = self._advise_for_player(p, team, teammates, opponents)
            advices.append(advice)

        return advices

    def _advise_for_player(
        self,
        me: PlayerState,
        team: str,
        teammates: List[PlayerState],
        opponents: List[PlayerState],
    ) -> TacticAdvice:
        """단일 선수에 대한 우선순위 기반 규칙 적용 + MovementModel 블렌딩."""

        # 기본값
        target = me.pos
        urgency = "LOW"
        reason = "위치 유지"
        voice = ""
        rule = "BASE"

        # R5: 적이 자기 진영 깊이 들어옴 → 후퇴 (HIGH — ML 무시)
        own_half_x_range = (0, self.court_width / 2) if team == "A" else (self.court_width / 2, self.court_width)
        opps_in_own_half_deep = [
            o for o in opponents
            if own_half_x_range[0] <= o.court_x <= own_half_x_range[1]
            and self._depth_in_own_half(o.court_x, team) > self.backline_depth
        ]
        if len(opps_in_own_half_deep) >= 2:
            target_x = own_half_x_range[0] + 0.4 if team == "A" else own_half_x_range[1] - 0.4
            target = (target_x, me.court_y)
            urgency, reason, voice, rule = "HIGH", "적 다수 침투 — 후방 수비", "후방 수비", "R5"

        # R3: 정면 카운터 위협 (HIGH — ML 무시)
        elif [o for o in opponents
              if _dist(o.pos, me.pos) < self.counter_range
              and abs(o.court_y - me.court_y) < self.counter_y_offset]:
            threats = [o for o in opponents
                       if _dist(o.pos, me.pos) < self.counter_range
                       and abs(o.court_y - me.court_y) < self.counter_y_offset]
            nearest = min(threats, key=lambda o: _dist(o.pos, me.pos))
            avoid_y = self.court_height - 0.4 if me.court_y < self.court_height / 2 else 0.4
            if team == "A":
                target = (max(0.3, me.court_x - 0.3), avoid_y)
            else:
                target = (min(self.court_width - 0.3, me.court_x + 0.3), avoid_y)
            urgency, voice, rule = "HIGH", "측면 회피", "R3"
            reason = f"정면 위협 (적 #{nearest.track_id})"

        # R1: 팀원 너무 가까움 → 분산 (MID)
        elif teammates and any(_dist(m.pos, me.pos) < self.spacing_min for m in teammates):
            mate = min(teammates, key=lambda m: _dist(m.pos, me.pos))
            if me.court_y < mate.court_y:
                target = (me.court_x, max(0.3, me.court_y - 0.6))
            else:
                target = (me.court_x, min(self.court_height - 0.3, me.court_y + 0.6))
            urgency, voice, rule = "MID", "거리 확보", "R1"
            reason = f"팀원과 너무 가까움 (#{mate.track_id})"

        # R4: 적 라인 사이 공간 공격 (MID)
        elif len(opponents) >= 2:
            opp_sorted = sorted(opponents, key=lambda o: o.court_y)
            _r4_applied = False
            for i in range(len(opp_sorted) - 1):
                gap_size = opp_sorted[i + 1].court_y - opp_sorted[i].court_y
                if gap_size > self.gap_min:
                    gap_y = (opp_sorted[i].court_y + opp_sorted[i + 1].court_y) / 2
                    opp_x_avg = sum(o.court_x for o in opponents) / len(opponents)
                    half = self.court_width / 2
                    if team == "A":
                        target_x = max(me.court_x, min(opp_x_avg - 0.4, half - 0.1))
                    else:
                        target_x = min(me.court_x, max(opp_x_avg + 0.4, half + 0.1))
                    target = (target_x, gap_y)
                    urgency, voice, rule = "MID", "공격 전진", "R4"
                    reason = f"적 라인 공간 공격 ({gap_size:.1f}m gap)"
                    _r4_applied = True
                    break
            # R4 미적용 → R2 체크
            if not _r4_applied and teammates:
                ys = [t.court_y for t in teammates] + [me.court_y]
                if max(ys) - min(ys) < self.coverage_spread_min:
                    team_y_avg = sum(ys) / len(ys)
                    if me.court_y >= team_y_avg:
                        target = (me.court_x, min(self.court_height - 0.3, team_y_avg + 0.9))
                    else:
                        target = (me.court_x, max(0.3, team_y_avg - 0.9))
                    urgency, voice, rule = "MID", "측면 커버", "R2"
                    reason = "팀이 한쪽 쏠림 — 커버 분담"

        # R2: 팀 쏠림 커버 (opponents < 2인 경우)
        elif teammates:
            ys = [t.court_y for t in teammates] + [me.court_y]
            if max(ys) - min(ys) < self.coverage_spread_min:
                team_y_avg = sum(ys) / len(ys)
                if me.court_y >= team_y_avg:
                    target = (me.court_x, min(self.court_height - 0.3, team_y_avg + 0.9))
                else:
                    target = (me.court_x, max(0.3, team_y_avg - 0.9))
                urgency, voice, rule = "MID", "측면 커버", "R2"
                reason = "팀이 한쪽 쏠림 — 커버 분담"

        # ── MovementModel 블렌딩 (HIGH urgency 제외) ──────────────
        ml_conf = 0.0
        if self._movement_model and me.role and urgency != "HIGH":
            ctx = _rule_to_context(rule)
            pred = self._movement_model.predict(me.pos, me.role, ctx, team)
            if pred and pred.confidence >= self._ML_CONF_MIN:
                w_ml   = pred.confidence * (self._ML_WEIGHT_BASE if rule == "BASE"
                                             else self._ML_WEIGHT_MID)
                w_rule = 1.0 - w_ml
                tx = target[0] * w_rule + pred.target_pos[0] * w_ml
                ty = target[1] * w_rule + pred.target_pos[1] * w_ml
                # 진영 클리핑
                if team == "A":
                    tx = max(0.0, min(4.99, tx))
                else:
                    tx = max(5.01, min(self.court_width, tx))
                ty = max(0.0, min(self.court_height, ty))
                target  = (round(tx, 3), round(ty, 3))
                ml_conf = pred.confidence
                reason  = f"{reason} [ML {pred.confidence:.2f}]"

        return self._make_advice(me, team, target, urgency, reason, voice, rule, ml_conf)

    def _make_advice(
        self, me: PlayerState, team: str, target: Tuple[float, float],
        urgency: str, reason: str, voice: str, rule: str,
        ml_confidence: float = 0.0,
    ) -> TacticAdvice:
        dx, dy = target[0] - me.court_x, target[1] - me.court_y
        if not voice and _dist(target, me.pos) > 0.15:
            voice = _direction_name(dx, dy)
        return TacticAdvice(
            track_id=me.track_id,
            team=team,
            current_pos=me.pos,
            target_pos=target,
            urgency=urgency,
            reason=reason,
            voice_message=voice,
            rule=rule,
            ml_confidence=ml_confidence,
        )

    def _depth_in_own_half(self, x: float, team: str) -> float:
        """팀 진영에서 자기 골 라인까지의 깊이 (m)."""
        if team == "A":
            return self.court_width / 2 - x   # A는 좌측이 자기 진영, x 작을수록 깊다
        else:
            return x - self.court_width / 2


# ---------- 단독 테스트 ----------
def main():
    engine = TacticEngine()

    # 2 vs 2 시나리오: 팀 A가 코트 왼쪽, 팀 B가 오른쪽
    players = [
        PlayerState(track_id=1, court_x=1.5, court_y=1.0),   # A1
        PlayerState(track_id=2, court_x=1.6, court_y=1.1),   # A2 — 너무 붙음
        PlayerState(track_id=3, court_x=4.5, court_y=0.5),   # B1
        PlayerState(track_id=4, court_x=4.5, court_y=2.2),   # B2 — 사이 갭 큼
    ]

    advices = engine.analyze(players)
    print("=" * 60)
    print(" Tactic Engine 단독 테스트")
    print("=" * 60)
    for a in advices:
        print(f"\n[Player #{a.track_id} | Team {a.team}]")
        print(f"  현재: ({a.current_pos[0]:.2f}, {a.current_pos[1]:.2f})")
        print(f"  추천: ({a.target_pos[0]:.2f}, {a.target_pos[1]:.2f})")
        print(f"  거리: {a.distance_m:.2f} m, 긴급도: {a.urgency}, 규칙: {a.rule}")
        print(f"  사유: {a.reason}")
        print(f"  음성: \"{a.voice_message}\"")


if __name__ == "__main__":
    main()
