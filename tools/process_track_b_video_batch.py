"""Batch intake pipeline for Track B match videos.

This utility scans a directory of match videos, runs the current player-only
detector on each file, mines hard frames for relabeling, and writes a batch
manifest so the next Roboflow pass can be tracked cleanly.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2

from track_b_runtime import PROJECT_ROOT, resolve_device

RUN_VIDEO_SCRIPT = PROJECT_ROOT / "tools" / "run_player_model_on_video.py"
MINE_HARD_SCRIPT = PROJECT_ROOT / "tools" / "mine_video_hard_frames.py"


def _video_meta(video_path: Path) -> dict[str, float | int | str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0
    return {
        "video_path": str(video_path),
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def _summarize_csv(csv_path: Path) -> dict[str, float | int]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = [int(row["detections"]) for row in rows]
    avg_confs = [float(row["avg_conf"]) for row in rows]
    nonzero_confs = [value for value in avg_confs if value > 0]
    processed = len(rows)
    zero_frames = sum(1 for value in counts if value == 0)
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    avg_count = round(sum(counts) / processed, 3) if processed else 0.0
    avg_conf = round(sum(nonzero_confs) / len(nonzero_confs), 4) if nonzero_confs else 0.0
    return {
        "processed_frames": processed,
        "zero_frames": zero_frames,
        "min_count": min_count,
        "max_count": max_count,
        "avg_count": avg_count,
        "avg_nonzero_conf": avg_conf,
    }


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run(args: argparse.Namespace) -> int:
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    detections_dir = Path(args.out_dir) / "detections"
    hard_frames_root = Path(args.out_dir) / "hard_frames"
    detections_dir.mkdir(parents=True, exist_ok=True)
    hard_frames_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    seen: set[Path] = set()
    videos: list[Path] = []
    for pattern in args.pattern:
        for video_path in sorted(video_dir.glob(pattern)):
            if video_path in seen:
                continue
            seen.add(video_path)
            videos.append(video_path)
    if not videos:
        raise FileNotFoundError(f"No videos matched {args.pattern} under {video_dir}")

    manifest_rows: list[dict[str, str | int | float]] = []
    for video_path in videos:
        meta = _video_meta(video_path)
        stem = video_path.stem.replace(" ", "_")
        csv_path = detections_dir / f"{stem}_player_only_detections.csv"
        hard_dir = hard_frames_root / stem

        try:
            detect_cmd = [
                sys.executable,
                str(RUN_VIDEO_SCRIPT),
                "--video",
                str(video_path),
                "--out-dir",
                str(detections_dir),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--device",
                device,
                "--frame-stride",
                str(args.frame_stride),
                "--max-det",
                str(args.max_det),
                "--no-video",
            ]
            if args.model:
                detect_cmd.extend(["--model", args.model])
            if args.max_frames:
                detect_cmd.extend(["--max-frames", str(args.max_frames)])
            _run(detect_cmd)

            mine_cmd = [
                sys.executable,
                str(MINE_HARD_SCRIPT),
                "--video",
                str(video_path),
                "--csv",
                str(csv_path),
                "--out-dir",
                str(hard_dir),
                "--low-count",
                str(args.low_count),
                "--high-count",
                str(args.high_count),
                "--low-conf",
                str(args.low_conf),
                "--min-gap-frames",
                str(args.min_gap_frames),
                "--max-samples",
                str(args.max_samples),
            ]
            _run(mine_cmd)

            summary = _summarize_csv(csv_path)
            hard_count = len(list((hard_dir / "frames").glob("*.jpg")))
            manifest_rows.append(
                {
                    **meta,
                    **summary,
                    "status": "ok",
                    "error": "",
                    "csv_path": str(csv_path),
                    "hard_frame_dir": str(hard_dir),
                    "hard_frame_count": hard_count,
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    **meta,
                    "processed_frames": 0,
                    "zero_frames": 0,
                    "min_count": 0,
                    "max_count": 0,
                    "avg_count": 0.0,
                    "avg_nonzero_conf": 0.0,
                    "status": "error",
                    "error": str(exc),
                    "csv_path": str(csv_path),
                    "hard_frame_dir": str(hard_dir),
                    "hard_frame_count": 0,
                }
            )
            print(f"warning: failed to process {video_path}: {exc}")
            if args.stop_on_error:
                break

    manifest_path = Path(args.out_dir) / "batch_manifest.csv"
    fieldnames = [
        "video_path",
        "fps",
        "total_frames",
        "duration_sec",
        "width",
        "height",
        "processed_frames",
        "zero_frames",
        "min_count",
        "max_count",
        "avg_count",
        "avg_nonzero_conf",
        "status",
        "error",
        "hard_frame_count",
        "csv_path",
        "hard_frame_dir",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"videos_processed: {len(manifest_rows)}")
    print(f"manifest: {manifest_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process Track B match videos.")
    parser.add_argument("--video-dir", required=True, help="Directory containing match videos.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for videos. Repeat this flag to include multiple groups.",
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "track_b_batch_review"))
    parser.add_argument("--model", default=None, help="Optional YOLO checkpoint path.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:0")
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--low-count", type=int, default=1)
    parser.add_argument("--high-count", type=int, default=10)
    parser.add_argument("--low-conf", type=float, default=0.18)
    parser.add_argument("--min-gap-frames", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the batch on the first error.")
    parsed = parser.parse_args()
    if not parsed.pattern:
        parsed.pattern = ["*.mp4"]
    raise SystemExit(run(parsed))


if __name__ == "__main__":
    main()
