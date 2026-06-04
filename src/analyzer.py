"""경기 후 분석 모듈.

positions.csv를 읽어 statistics.json + 히트맵 이미지를 생성한다.

실행:
    python -m src.analyzer --match-dir data/matches/20260604_180000
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.homography import Calibration
from src.visualizer import render_court_birdeye


def analyze_match(match_dir: Path, calib: Calibration, px_per_m: float = 60.0) -> dict:
    """match_dir 안의 positions.csv를 분석해 statistics.json + heatmap.png 생성."""
    csv_path = match_dir / "positions.csv"
    if not csv_path.exists():
        print(f"[Analyzer] positions.csv 없음: {csv_path}")
        return {}

    print(f"[Analyzer] 분석 시작: {match_dir.name}")
    stats = _compute_statistics(csv_path, calib)

    stats_path = match_dir / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Analyzer] statistics.json 저장")

    _generate_heatmap(csv_path, calib, match_dir, px_per_m)
    _print_summary(stats)
    return stats


def _load_positions(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "t": float(row["timestamp"]),
                    "frame": int(row["frame_idx"]),
                    "track_id": int(row["track_id"]),
                    "x": float(row["x_m"]),
                    "y": float(row["y_m"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _compute_statistics(csv_path: Path, calib: Calibration) -> dict:
    rows = _load_positions(csv_path)
    if not rows:
        return {}

    duration = rows[-1]["t"] - rows[0]["t"] if len(rows) > 1 else 0.0
    total_frames = rows[-1]["frame"] + 1 if rows else 0

    by_track: Dict[int, List[dict]] = {}
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r)

    player_stats = {}
    for tid, pts in by_track.items():
        pts.sort(key=lambda p: p["t"])

        dist = 0.0
        speeds = []
        for i in range(1, len(pts)):
            dx = pts[i]["x"] - pts[i-1]["x"]
            dy = pts[i]["y"] - pts[i-1]["y"]
            dt = pts[i]["t"] - pts[i-1]["t"]
            if 0 < dt < 2.0:
                d = (dx**2 + dy**2) ** 0.5
                dist += d
                speeds.append(d / dt)

        half_x = calib.court_width_m / 2
        attack_frames = sum(1 for p in pts if p["x"] > half_x)
        defend_frames = len(pts) - attack_frames

        # 구역별 체류 시간 (x축 3구간)
        zone_w = calib.court_width_m / 3
        zone_counts = [0, 0, 0]
        for p in pts:
            z = min(2, int(p["x"] / zone_w))
            zone_counts[z] += 1

        avg_x = sum(p["x"] for p in pts) / len(pts)
        avg_y = sum(p["y"] for p in pts) / len(pts)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        player_stats[str(tid)] = {
            "total_distance_m": round(dist, 2),
            "avg_speed_m_s": round(avg_speed, 3),
            "attack_zone_ratio": round(attack_frames / max(1, len(pts)), 3),
            "defend_zone_ratio": round(defend_frames / max(1, len(pts)), 3),
            "zone_distribution": [round(z / max(1, len(pts)), 3) for z in zone_counts],
            "avg_position_m": [round(avg_x, 2), round(avg_y, 2)],
            "detection_frames": len(pts),
        }

    return {
        "duration_sec": round(duration, 1),
        "total_frames": total_frames,
        "unique_players": len(by_track),
        "players": player_stats,
    }


def _generate_heatmap(
    csv_path: Path,
    calib: Calibration,
    out_dir: Path,
    px_per_m: float,
) -> None:
    rows = _load_positions(csv_path)
    if not rows:
        return

    W = int(calib.court_width_m * px_per_m)
    H = int(calib.court_height_m * px_per_m)

    heatmap = np.zeros((H, W), dtype=np.float32)
    for r in rows:
        px = int(np.clip(r["x"] * px_per_m, 0, W - 1))
        py = int(np.clip(r["y"] * px_per_m, 0, H - 1))
        heatmap[py, px] += 1.0

    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    court = render_court_birdeye(calib, px_per_m=px_per_m)
    court_r = cv2.resize(court, (W, H))
    gray = cv2.cvtColor(court_r, cv2.COLOR_BGR2GRAY)
    _, line_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    colored[line_mask > 0] = (220, 220, 220)

    out_path = out_dir / "heatmap.png"
    cv2.imwrite(str(out_path), colored)
    print(f"[Analyzer] heatmap.png 저장")


def _print_summary(stats: dict) -> None:
    print(f"\n{'='*50}")
    print(f" 경기 분석 요약")
    print(f"{'='*50}")
    print(f"  경기 시간   : {stats.get('duration_sec', 0):.1f}초")
    print(f"  총 프레임   : {stats.get('total_frames', 0)}")
    print(f"  감지 선수   : {stats.get('unique_players', 0)}명")
    for tid, p in stats.get("players", {}).items():
        print(f"  선수 #{tid}:")
        print(f"    이동거리  : {p['total_distance_m']:.1f} m")
        print(f"    평균속도  : {p['avg_speed_m_s']:.2f} m/s")
        print(f"    공격구역  : {p['attack_zone_ratio']:.0%}")
    print(f"{'='*50}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="경기 후 분석")
    parser.add_argument("--match-dir", required=True, help="match 폴더 경로")
    parser.add_argument("--calib", default="config/calibration.json")
    parser.add_argument("--px-per-m", type=float, default=60.0)
    args = parser.parse_args()

    from src.homography import Calibration
    calib = Calibration.from_json(args.calib)
    match_dir = Path(args.match_dir)
    analyze_match(match_dir, calib, px_per_m=args.px_per_m)


if __name__ == "__main__":
    main()
