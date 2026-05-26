"""Level 2 전술 분석 엔진.

선수 좌표(m 단위)를 입력받아 각 선수에게 전술 조언(추천 위치 + 이동 방향 + 음성 안내)을 생성.

설계 원칙
---------
1. **Rule-based 우선**: ML 없이도 의미 있는 가이드가 가능해야 한다.
2. **HADO 특화**: 6m × 2.66m 코트, 2 vs 2 또는 3 vs 3 기준.
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
from typing import Dict, List, Tuple


# ---------- 데이터 클래스 ----------
@dataclass
class PlayerState:
    """단일 선수의 현재 상태 (Level 2 입력)."""
    track_id: int
    court_x: float       # 코트 X 좌표 (m)
    court_y: float       # 코트 Y 좌표 (m)
    confidence: float = 1.0

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
    rule: str                       # 어떤 규칙이 발동했는지 (R1~R5)

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


# ---------- 엔진 ----------
class TacticEngine:
    """규칙 기반 전술 분석기.

    매 프레임 호출 가능. 팀 배정은 첫 등장 시 자동(코트 절반 기준).
    """

    def __init__(
        self,
        court_width_m: float = 6.0,
        court_height_m: float = 2.66,
        spacing_min_m: float = 1.0,
        counter_range_m: float = 2.5,
        gap_min_m: float = 2.0,
        backline_depth_m: float = 1.5,
    ):
        self.court_width = court_width_m
        self.court_height = court_height_m
        self.spacing_min = spacing_min_m
        self.counter_range = counter_range_m
        self.gap_min = gap_min_m
        self.backline_depth = backline_depth_m
        self._team_assignment: Dict[int, str] = {}

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
        """단일 선수에 대한 우선순위 기반 규칙 적용."""

        # 기본값: 현재 위치 유지
        target = me.pos
        urgency = "LOW"
        reason = "위치 유지"
        voice = ""
        rule = "BASE"

        # R5: 적이 자기 진영 깊이 들어옴 → 후퇴
        own_half_x_range = (0, self.court_width / 2) if team == "A" else (self.court_width / 2, self.court_width)
        opps_in_own_half_deep = [
            o for o in opponents
            if own_half_x_range[0] <= o.court_x <= own_half_x_range[1]
            and self._depth_in_own_half(o.court_x, team) > self.backline_depth
        ]
        if len(opps_in_own_half_deep) >= 2:
            # 자기 진영 가장 안쪽으로 후퇴
            target_x = own_half_x_range[0] + 0.4 if team == "A" else own_half_x_range[1] - 0.4
            target = (target_x, me.court_y)
            urgency = "HIGH"
            reason = "적 다수 침투 — 후방 수비"
            voice = "후방 수비"
            rule = "R5"
            return self._make_advice(me, team, target, urgency, reason, voice, rule)

        # R3: 정면 카운터 위협 (가까운 적이 같은 y선)
        threats = [
            o for o in opponents
            if _dist(o.pos, me.pos) < self.counter_range
            and abs(o.court_y - me.court_y) < 0.6
        ]
        if threats:
            nearest = min(threats, key=lambda o: _dist(o.pos, me.pos))
            # 측면(y 방향)으로 회피 — 코트 중심에서 먼 쪽
            avoid_y = self.court_height - 0.4 if me.court_y < self.court_height / 2 else 0.4
            target = (me.court_x, avoid_y)
            # x 후퇴도 살짝
            if team == "A":
                target = (max(0.3, me.court_x - 0.3), avoid_y)
            else:
                target = (min(self.court_width - 0.3, me.court_x + 0.3), avoid_y)
            urgency = "HIGH"
            reason = f"정면 위협 (적 #{nearest.track_id})"
            voice = "측면 회피"
            rule = "R3"
            return self._make_advice(me, team, target, urgency, reason, voice, rule)

        # R1: 팀원과 너무 가깝다 → 분산
        if teammates:
            for mate in teammates:
                if _dist(mate.pos, me.pos) < self.spacing_min:
                    # 중심에서 더 먼 쪽으로
                    if me.court_y < mate.court_y:
                        target = (me.court_x, max(0.3, me.court_y - 0.6))
                    else:
                        target = (me.court_x, min(self.court_height - 0.3, me.court_y + 0.6))
                    urgency = "MID"
                    reason = f"팀원과 너무 가까움 (#{mate.track_id})"
                    voice = "거리 확보"
                    rule = "R1"
                    return self._make_advice(me, team, target, urgency, reason, voice, rule)

        # R4: 적 팀 사이 공간 공격
        if len(opponents) >= 2:
            opp_sorted = sorted(opponents, key=lambda o: o.court_y)
            for i in range(len(opp_sorted) - 1):
                gap_y = (opp_sorted[i].court_y + opp_sorted[i + 1].court_y) / 2
                gap_size = opp_sorted[i + 1].court_y - opp_sorted[i].court_y
                if gap_size > self.gap_min:
                    # 적 라인까지 전진하되 적 평균 x보다 살짝 뒤
                    opp_x_avg = sum(o.court_x for o in opponents) / len(opponents)
                    if team == "A":
                        target_x = max(me.court_x, min(opp_x_avg - 0.4, self.court_width - 0.5))
                    else:
                        target_x = min(me.court_x, max(opp_x_avg + 0.4, 0.5))
                    target = (target_x, gap_y)
                    urgency = "MID"
                    reason = f"적 라인 공간 공격 ({gap_size:.1f}m gap)"
                    voice = "공격 전진"
                    rule = "R4"
                    return self._make_advice(me, team, target, urgency, reason, voice, rule)

        # R2: 같은 팀이 좌우 한쪽으로 쏠림 → 반대편 커버
        if teammates:
            ys = [t.court_y for t in teammates] + [me.court_y]
            spread = max(ys) - min(ys)
            if spread < 0.8:
                # 내가 그룹의 위/아래 극단이면 그 방향으로 더 이동
                team_y_avg = sum(ys) / len(ys)
                if me.court_y >= team_y_avg:
                    target = (me.court_x, min(self.court_height - 0.3, team_y_avg + 0.9))
                else:
                    target = (me.court_x, max(0.3, team_y_avg - 0.9))
                urgency = "MID"
                reason = "팀이 한쪽 쏠림 — 커버 분담"
                voice = "측면 커버"
                rule = "R2"
                return self._make_advice(me, team, target, urgency, reason, voice, rule)

        # 기본 — 유지
        return self._make_advice(me, team, target, urgency, reason, voice, rule)

    def _make_advice(
        self, me: PlayerState, team: str, target: Tuple[float, float],
        urgency: str, reason: str, voice: str, rule: str,
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
