"""movement_data.csv 패턴에 intent를 일괄 지정하는 레이블링 도구.

각 패턴의 화살표를 코트 위에 시각화하고 z/x/c/v 키로 의도를 지정.
의도 지정 후 자동으로 다음 패턴으로 넘어가며, ESC/q 에서 CSV에 일괄 저장.

조작법
------
z/x/c/v  : 의도 지정 → 다음 패턴으로 자동 이동
n / Space / → : 다음 패턴 (의도 지정 없이 넘김)
p / ←    : 이전 패턴
s        : 현재까지 중간 저장
ESC / q  : 저장 후 종료

실행:
    python -m src.label_intents                 # 전체 패턴
    python -m src.label_intents --only-empty    # intent 미지정 패턴만
    ./run.sh label
    ./run.sh label --only-empty
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.annotate import (
    CH, CW, INTENT_KO, INTENT_MAP, LINE_C, PLAYER_COLORS, ROLE_KO,
    STATUS_BG, STATUS_H, WIN_H,
    Arrow, _draw_arrow, _make_court, put_text_kr,
)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "movement_data.csv"

PatternRows = List[Dict[str, str]]


def _load(only_empty: bool) -> tuple[list[PatternRows], list[dict]]:
    if not CSV_PATH.exists():
        print(f"[Label] CSV 없음: {CSV_PATH}")
        sys.exit(1)

    all_rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))

    groups: dict[int, PatternRows] = defaultdict(list)
    for row in all_rows:
        groups[int(row["pattern_id"])].append(row)

    patterns = [groups[pid] for pid in sorted(groups)]
    if only_empty:
        patterns = [p for p in patterns if not p[0]["intent"]]

    return patterns, all_rows


def _render_pattern(court_base: np.ndarray, rows: PatternRows) -> np.ndarray:
    court = court_base.copy()

    # step 번호 표시
    for i, row in enumerate(rows, start=1):
        pid = int(row["player_id"])
        arrow = Arrow(
            player_id=pid,
            from_m=(float(row["from_x"]), float(row["from_y"])),
            to_m=(float(row["to_x"]), float(row["to_y"])),
            role=row["role"],
            context=row["context"],
        )
        _draw_arrow(court, arrow)

        # 스텝 번호 — 출발점 옆에 작게 표시
        fx = int(float(row["from_x"]) * 80)
        fy = int(float(row["from_y"]) * 80)
        cv2.putText(court, str(i), (fx - 14, fy + 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.9,
                    PLAYER_COLORS.get(pid, (200, 200, 200)), 1, cv2.LINE_AA)

    return court


def _draw_panel(canvas: np.ndarray, rows: PatternRows,
                intent: str, idx: int, total: int, saved: bool) -> None:
    y0 = CH
    canvas[y0:, :] = STATUS_BG

    row0 = rows[0]
    role    = row0["role"]
    context = row0["context"]
    team    = row0["team"]
    pid     = row0["player_id"]
    pat_id  = row0["pattern_id"]
    steps   = len(rows)

    role_ko   = ROLE_KO.get(role, role)
    intent_ko = INTENT_KO.get(intent, "미지정")
    color     = PLAYER_COLORS.get(int(pid), (200, 200, 200))

    role_intents = INTENT_MAP.get(role, {})
    if role_intents:
        hint = "  ".join(f"[{k}]{INTENT_KO.get(v,'?')}"
                         for k, v in role_intents.items())
    else:
        hint = "포지션 없음 — q/w/e로 포지션 있는 데이터만 가능"

    saved_mark = " ✓저장됨" if saved else ""
    progress_bar = _progress(idx + 1, total)

    lines = [
        (f"패턴 {idx+1}/{total}  #{pat_id}  선수{pid}({team}팀)  "
         f"포지션: {role_ko}  컨텍스트: {context}  "
         f"스텝: {steps}개  |  의도: {intent_ko}{saved_mark}",
         color),
        (f"의도키: {hint}", (180, 200, 120)),
        (f"[z/x/c/v]지정+다음  [n/Space/→]다음  [p/←]이전  [s]저장  [ESC/q]종료"
         f"   {progress_bar}", LINE_C),
    ]
    for i, (line, clr) in enumerate(lines):
        put_text_kr(canvas, line, (10, y0 + 10 + i * 34), 16, clr)


def _progress(current: int, total: int, width: int = 30) -> str:
    filled = round(width * current / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {current}/{total}"


def _save(all_rows: list[dict], updates: dict[int, str]) -> None:
    header = list(all_rows[0].keys())
    for row in all_rows:
        pid = int(row["pattern_id"])
        if pid in updates:
            row["intent"] = updates[pid]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    n_labeled = sum(1 for r in all_rows if r["intent"])
    n_patterns = len({r["pattern_id"] for r in all_rows})
    n_updated  = sum(1 for v in updates.values() if v)
    print(f"[Label] 저장 — 이번 세션 {n_updated}개 패턴 지정 | "
          f"전체 {n_labeled}/{len(all_rows)}행 intent 있음 "
          f"({n_patterns}패턴 기준)")


def run(only_empty: bool = False) -> None:
    patterns, all_rows = _load(only_empty)
    if not patterns:
        print("[Label] 레이블링할 패턴 없음")
        return

    court_base = _make_court()
    updates: dict[int, str] = {}   # pattern_id → intent (이번 세션 변경분)

    idx      = 0
    total    = len(patterns)
    win_name = "HADO Intent Labeling"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    suffix = " (intent 미지정만)" if only_empty else ""
    print(f"\n[Label] {total}개 패턴 레이블링 시작{suffix}")
    print("[Label] z/x/c/v로 의도 지정 → 자동으로 다음 패턴 이동\n")

    dirty = False   # 저장 안 된 변경 있음

    while True:
        rows    = patterns[idx]
        pat_id  = int(rows[0]["pattern_id"])
        intent  = updates.get(pat_id, rows[0]["intent"])
        saved   = bool(rows[0]["intent"]) or pat_id in updates

        canvas = np.zeros((WIN_H, CW, 3), dtype=np.uint8)
        canvas[:CH, :CW] = _render_pattern(court_base, rows)
        _draw_panel(canvas, rows, intent, idx, total, saved)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q')):           # ESC / q — 저장 후 종료
            if dirty:
                _save(all_rows, updates)
            break

        elif key == ord('s'):               # 중간 저장
            _save(all_rows, updates)
            dirty = False

        elif key in (ord('n'), ord(' '), 83):   # n / Space / → 키코드83
            idx = min(idx + 1, total - 1)

        elif key in (ord('p'), 81):         # p / ← 키코드81
            idx = max(idx - 1, 0)

        elif chr(key) in "zxcv":
            role       = rows[0]["role"]
            new_intent = INTENT_MAP.get(role, {}).get(chr(key), "")
            if new_intent:
                updates[pat_id] = new_intent
                dirty = True
                print(f"[Label] #{pat_id:>3}  {ROLE_KO.get(role, role):6}  "
                      f"→  {INTENT_KO[new_intent]}  ({idx+1}/{total})")
                idx = min(idx + 1, total - 1)   # 자동으로 다음으로
            else:
                print(f"[Label] 포지션 '{role}' 에 해당 키 없음")

    cv2.destroyAllWindows()
    print("\n[Label] 종료")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="movement_data.csv intent 일괄 레이블링")
    parser.add_argument(
        "--only-empty", action="store_true",
        help="intent가 비어있는 패턴만 표시 (기본: 전체)")
    args = parser.parse_args()
    run(only_empty=args.only_empty)


if __name__ == "__main__":
    main()
