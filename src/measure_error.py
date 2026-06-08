"""W5 위치 오차 측정 도구.

코트에서 알려진 N개 좌표에 서서 CSV를 녹화한 뒤,
이 스크립트에 기준 좌표 목록을 넣으면 RMS 오차(m)와 각 점별 오차를 출력한다.

사용법
------
1. 코트에서 알려진 위치(예: 코너, 라인 교차점)에 마커를 붙인다.
2. `./run.sh main --level 1` 실행 후 `r` 키로 CSV 녹화 시작.
3. 마커 위치를 순서대로 (각 3~5초씩) 서고, `r` 키로 녹화 중지.
4. 이 스크립트 실행:
       python -m src.measure_error --csv data/position_logs/positions_XXXXXX.csv \\
                                   --ref 0.0,0.0 1.5,0.0 3.0,3.0 ...

기준점 형식: x_m,y_m (공백으로 구분, 서있던 순서와 동일하게)

실제 사용 예:
    python -m src.measure_error \\
        --csv data/position_logs/positions_20260613_143200.csv \\
        --ref 0.0,0.0  5.0,0.0  10.0,0.0  0.0,3.0  5.0,3.0  10.0,3.0  \\
               0.0,6.0  5.0,6.0  10.0,6.0  5.0,1.5
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load_positions(csv_path: Path) -> list[dict]:
    """positions_*.csv → 레코드 리스트."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "timestamp": float(row["timestamp"]),
                    "track_id":  int(row["track_id"]),
                    "x_m":       float(row["x_m"]),
                    "y_m":       float(row["y_m"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def segment_by_gap(rows: list[dict], gap_s: float = 2.0) -> list[list[dict]]:
    """타임스탬프 갭으로 레코드를 구간 분리. 각 구간 = 한 기준점에 서있던 시간."""
    if not rows:
        return []
    segments: list[list[dict]] = [[rows[0]]]
    for r in rows[1:]:
        if r["timestamp"] - segments[-1][-1]["timestamp"] > gap_s:
            segments.append([r])
        else:
            segments[-1].append(r)
    return segments


def median_pos(seg: list[dict]) -> tuple[float, float]:
    """구간 중앙값 위치 (outlier에 강건)."""
    xs = sorted(r["x_m"] for r in seg)
    ys = sorted(r["y_m"] for r in seg)
    mid = len(xs) // 2
    return xs[mid], ys[mid]


def run(args) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[Error] CSV 없음: {csv_path}")
        return

    # 기준점 파싱
    refs: list[tuple[float, float]] = []
    for token in args.ref:
        parts = token.split(",")
        if len(parts) != 2:
            print(f"[Error] 기준점 형식 오류: '{token}' → x_m,y_m 형식 필요")
            return
        refs.append((float(parts[0]), float(parts[1])))

    if not refs:
        print("[Error] --ref 기준점을 최소 1개 이상 지정하세요.")
        return

    rows = load_positions(csv_path)
    if not rows:
        print("[Error] CSV에 유효한 레코드가 없습니다.")
        return

    # 주 트랙 ID 선택 (가장 많이 등장한 ID)
    id_count: dict[int, int] = defaultdict(int)
    for r in rows:
        id_count[r["track_id"]] += 1
    main_id = max(id_count, key=id_count.__getitem__)
    player_rows = [r for r in rows if r["track_id"] == main_id]
    print(f"[MeasureError] CSV: {csv_path.name}")
    print(f"[MeasureError] 주 트랙 ID: #{main_id} ({len(player_rows)}개 레코드)")

    segments = segment_by_gap(player_rows, gap_s=args.gap)
    print(f"[MeasureError] 감지된 구간 수: {len(segments)}  (기준점 수: {len(refs)})\n")

    if len(segments) < len(refs):
        print(f"[Warning] 구간({len(segments)})이 기준점({len(refs)})보다 적습니다.")
        print("  → CSV 녹화가 짧거나 gap 파라미터 조정 필요 (--gap 현재값: {args.gap}s)")

    n = min(len(segments), len(refs))
    errors: list[float] = []

    print(f"{'#':>3}  {'기준 (m)':>14}  {'측정 (m)':>14}  {'오차 (m)':>10}")
    print("-" * 52)
    for i in range(n):
        ref_x, ref_y = refs[i]
        meas_x, meas_y = median_pos(segments[i])
        err = math.sqrt((meas_x - ref_x) ** 2 + (meas_y - ref_y) ** 2)
        errors.append(err)
        print(f"{i+1:>3}  ({ref_x:5.2f}, {ref_y:5.2f})  "
              f"({meas_x:5.2f}, {meas_y:5.2f})  {err:8.3f} m")

    if not errors:
        print("[Error] 유효한 비교 쌍이 없습니다.")
        return

    rms = math.sqrt(mean(e ** 2 for e in errors))
    avg = mean(errors)
    mx  = max(errors)

    print("-" * 52)
    print(f"\n  RMS 오차  : {rms*100:.1f} cm   ← QA_PREP.md에 기입")
    print(f"  평균 오차 : {avg*100:.1f} cm")
    print(f"  최대 오차 : {mx*100:.1f} cm")
    print(f"  샘플 수   : {n}")
    if len(errors) > 1:
        print(f"  표준편차  : {stdev(errors)*100:.1f} cm")

    # 판정
    print()
    if rms <= 0.10:
        print("  ✅ 목표 달성 (≤ 10 cm)")
    elif rms <= 0.25:
        print("  ⚠  하드 한계 이내 (10~25 cm) — 캘리브레이션 재시도 권장")
    else:
        print("  ❌ 하드 한계 초과 (> 25 cm) — 캘리브레이션 필수 재실행")


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO 위치 오차 측정")
    parser.add_argument("--csv",  required=True, help="positions_*.csv 경로")
    parser.add_argument("--ref",  nargs="+", required=True,
                        metavar="x,y",
                        help="기준점 목록 (서있던 순서대로, 예: 0.0,0.0 5.0,3.0 ...)")
    parser.add_argument("--gap",  type=float, default=2.0,
                        help="구간 분리 타임스탬프 갭 (초, 기본 2.0)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
