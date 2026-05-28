"""k-NN 기반 움직임 패턴 매칭 모델.

movement_data.csv에 저장된 전문가 어노테이션 데이터를 로드하고,
선수의 현재 위치·포지션·컨텍스트를 입력받아 가장 유사한 패턴의
다음 목표 위치를 반환한다.

매칭 알고리즘
-----------
1. (role, context) 로 후보 패턴 필터링 (exact match)
2. 각 패턴의 모든 스텝 중 현재 위치에 가장 가까운 스텝을 찾음
3. 그 거리로 k-NN 순위 결정 (상위 k개)
4. 역거리 가중 평균(IDW)으로 다음 목표 위치 산출

실행:
    python -m src.movement_model
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "movement_data.csv"

COURT_W = 10.0
COURT_H = 6.0


# ---------- 내부 데이터 구조 ----------

@dataclass
class _Step:
    from_pos: Tuple[float, float]
    to_pos:   Tuple[float, float]


@dataclass
class _Pattern:
    pattern_id: int
    team:       str
    role:       str
    context:    str
    intent:     str = ""
    steps:      List[_Step] = field(default_factory=list)

    def closest_step_idx(self, pos: Tuple[float, float]) -> Tuple[int, float]:
        """현재 위치와 가장 가까운 스텝 인덱스와 거리를 반환."""
        best_i, best_d = 0, float("inf")
        for i, s in enumerate(self.steps):
            d = _dist(pos, s.from_pos)
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d


# ---------- 출력 ----------

@dataclass
class MovementPrediction:
    """predict() 반환값."""
    target_pos:        Tuple[float, float]        # 즉시 이동 목표 (1스텝)
    full_path:         List[Tuple[float, float]]  # 매칭 패턴의 남은 경로
    confidence:        float                      # 0.0~1.0 (거리 기반)
    matched_pattern_id: int                       # 디버깅용
    intent:            str = ""                   # 매칭된 패턴의 의도


# ---------- 헬퍼 ----------

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clip(pos: Tuple[float, float], team: str) -> Tuple[float, float]:
    """진영 불가침: 팀A x<5.0, 팀B x>5.0."""
    x, y = pos
    y = max(0.0, min(COURT_H, y))
    if team == "A":
        x = max(0.0, min(4.99, x))
    else:
        x = max(5.01, min(COURT_W, x))
    return (round(x, 3), round(y, 3))


# ---------- 모델 ----------

class MovementModel:
    """전문가 어노테이션 기반 k-NN 이동 패턴 모델.

    Parameters
    ----------
    csv_path : path-like
        movement_data.csv 경로. 기본값은 data/movement_data.csv.
    k : int
        참조할 최근접 패턴 수. 데이터가 적으면 자동 조정.
    """

    def __init__(
        self,
        csv_path: Path | str = _DEFAULT_CSV,
        k: int = 3,
    ) -> None:
        self.k = k
        # (role, context) → List[_Pattern]
        self._index: Dict[Tuple[str, str], List[_Pattern]] = defaultdict(list)
        self._load(Path(csv_path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        raw: Dict[int, _Pattern] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid  = int(row["pattern_id"])
                step = int(row["step"])
                if pid not in raw:
                    raw[pid] = _Pattern(
                        pattern_id=pid,
                        team=row["team"],
                        role=row["role"],
                        context=row["context"],
                        intent=row.get("intent", ""),
                    )
                # step 순서가 뒤섞여 있을 수 있으므로 인덱스로 삽입
                p = raw[pid]
                s = _Step(
                    from_pos=(float(row["from_x"]), float(row["from_y"])),
                    to_pos  =(float(row["to_x"]),   float(row["to_y"])),
                )
                # step 번호 순 유지
                while len(p.steps) < step:
                    p.steps.append(s)   # 빈 자리 채우기 (정상 CSV는 발생 안 함)
                p.steps[step - 1] = s

        for p in raw.values():
            if p.steps:
                self._index[(p.role, p.context)].append(p)

    @property
    def pattern_count(self) -> int:
        return sum(len(v) for v in self._index.values())

    def predict(
        self,
        pos:     Tuple[float, float],
        role:    str,
        context: str,
        team:    str,
        intent:  str = "",
    ) -> Optional[MovementPrediction]:
        """현재 위치·포지션·상황을 받아 다음 목표 위치를 반환.

        Parameters
        ----------
        pos     : (court_x, court_y) in metres
        role    : "technician" | "defender" | "main_attacker"
        context : "attack" | "defend" | "transition"
        team    : "A" | "B"
        intent  : 선택적. 지정 시 해당 의도 패턴만 후보로 사용.

        Returns
        -------
        MovementPrediction or None (매칭 패턴 없음)
        """
        candidates = self._index.get((role, context), [])
        # intent 지정 시 해당 의도 패턴 우선 사용 (없으면 전체로 fallback)
        if intent:
            intent_filtered = [p for p in candidates if p.intent == intent]
            if intent_filtered:
                candidates = intent_filtered
        if not candidates:
            return None

        # 각 패턴에서 현재 위치와 가장 가까운 스텝 탐색
        scored: List[Tuple[float, _Pattern, int]] = []  # (distance, pattern, step_idx)
        for pat in candidates:
            step_i, dist = pat.closest_step_idx(pos)
            scored.append((dist, pat, step_i))

        scored.sort(key=lambda x: x[0])
        top_k = scored[: max(1, min(self.k, len(scored)))]

        # 역거리 가중 평균으로 목표 위치 결정
        tx, ty, weight_sum = 0.0, 0.0, 0.0
        for dist, pat, step_i in top_k:
            w = 1.0 / (dist + 1e-6)
            next_step = pat.steps[step_i]
            tx += next_step.to_pos[0] * w
            ty += next_step.to_pos[1] * w
            weight_sum += w

        raw_target = (tx / weight_sum, ty / weight_sum)
        target = _clip(raw_target, team)

        # 신뢰도: 가장 가까운 패턴의 거리 기반 (3m 이상이면 0)
        best_dist = top_k[0][0]
        confidence = max(0.0, 1.0 - best_dist / 3.0)

        # 베스트 패턴의 남은 경로 (matched step 이후)
        _, best_pat, best_step_i = top_k[0]
        remaining = [s.to_pos for s in best_pat.steps[best_step_i:]]

        return MovementPrediction(
            target_pos=target,
            full_path=remaining,
            confidence=round(confidence, 3),
            matched_pattern_id=best_pat.pattern_id,
            intent=best_pat.intent,
        )

    def coverage(self) -> Dict[Tuple[str, str], int]:
        """(role, context) → 패턴 수 딕셔너리."""
        return {k: len(v) for k, v in self._index.items()}


# ---------- 독립 실행 ----------

def main() -> None:
    model = MovementModel()
    print(f"MovementModel 로드 완료 — {model.pattern_count}개 패턴\n")

    print("=== 커버리지 ===")
    roles    = ["technician", "defender", "main_attacker"]
    contexts = ["attack", "defend", "transition"]
    cov = model.coverage()
    print(f"{'':20} {'attack':>8} {'defend':>8} {'transition':>12}")
    for role in roles:
        row = f"{role:20}"
        for ctx in contexts:
            row += f"  {cov.get((role, ctx), 0):>6}"
        print(row)

    print("\n=== 예측 테스트 ===")
    test_cases = [
        # (pos,        role,           context,     team, 설명)
        ((1.0, 1.0),  "technician",   "attack",    "A",  "팀A 테크니션 — 왼쪽 상단"),
        ((3.5, 3.0),  "main_attacker","attack",    "A",  "팀A 메인공격수 — 중앙 전방"),
        ((2.0, 5.0),  "defender",     "defend",    "A",  "팀A 디펜더 — 왼쪽 하단"),
        ((8.0, 1.0),  "technician",   "defend",    "B",  "팀B 테크니션 — 오른쪽 상단"),
        ((6.5, 3.0),  "main_attacker","transition","B",  "팀B 메인공격수 — 전환 상황"),
    ]

    for pos, role, ctx, team, desc in test_cases:
        pred = model.predict(pos, role, ctx, team)
        if pred:
            print(f"  {desc}")
            print(f"    현재: {pos}  →  목표: {pred.target_pos}"
                  f"  (신뢰도 {pred.confidence:.2f}, 패턴#{pred.matched_pattern_id})")
        else:
            print(f"  {desc}: 매칭 없음")


if __name__ == "__main__":
    main()
