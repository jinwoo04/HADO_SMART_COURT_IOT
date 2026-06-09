"""Orchestrate the full Track B batch-review loop.

This wrapper ties together three existing utilities:

1. `process_track_b_video_batch.py`
2. `summarize_track_b_batch.py`
3. `prepare_roboflow_upload_bundle.py`

It exists to make the recurring Track B workflow easy to repeat on each new
video batch without retyping the same command chain.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from track_b_runtime import PROJECT_ROOT, resolve_model_path


PROCESS_SCRIPT = PROJECT_ROOT / "tools" / "process_track_b_video_batch.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "tools" / "summarize_track_b_batch.py"
BUNDLE_SCRIPT = PROJECT_ROOT / "tools" / "prepare_roboflow_upload_bundle.py"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _match_order(batch_manifest: Path, summary_csv: Path) -> list[str]:
    if summary_csv.exists():
        rows = _read_csv(summary_csv)
        matches = [row["match_name"] for row in rows if row.get("match_name")]
        if matches:
            return matches

    rows = _read_csv(batch_manifest)
    matches: list[str] = []
    seen: set[str] = set()
    for row in rows:
        match_name = Path(row["video_path"]).stem
        if match_name in seen:
            continue
        seen.add(match_name)
        matches.append(match_name)
    return matches


def _hard_manifests(hard_root: Path) -> list[Path]:
    return sorted(hard_root.glob("*/hard_frame_manifest.csv"))


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_manifest = out_dir / "batch_manifest.csv"
    summary_md = Path(args.summary_md) if args.summary_md else out_dir / "batch_summary.md"
    summary_csv = Path(args.summary_csv) if args.summary_csv else out_dir / "batch_priority.csv"
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else out_dir / "roboflow_upload_bundle"
    hard_root = out_dir / "hard_frames"

    if not args.skip_process:
        if not args.video_dir:
            raise ValueError("--video-dir is required unless --skip-process is used.")

        resolved_model = resolve_model_path(args.model)
        cmd = [
            sys.executable,
            str(PROCESS_SCRIPT),
            "--video-dir",
            args.video_dir,
            "--out-dir",
            str(out_dir),
            "--model",
            str(resolved_model),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--device",
            args.device,
            "--frame-stride",
            str(args.frame_stride),
            "--max-det",
            str(args.max_det),
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
        for pattern in args.pattern:
            cmd.extend(["--pattern", pattern])
        if args.max_frames:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.stop_on_error:
            cmd.append("--stop-on-error")
        _run(cmd)

    if not batch_manifest.exists():
        raise FileNotFoundError(batch_manifest)

    if not args.skip_summary:
        cmd = [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--batch-manifest",
            str(batch_manifest),
            "--out-md",
            str(summary_md),
            "--out-csv",
            str(summary_csv),
        ]
        _run(cmd)

    if not args.skip_bundle:
        manifest_paths = _hard_manifests(hard_root)
        if not manifest_paths:
            raise FileNotFoundError(f"No hard frame manifests were found under {hard_root}")

        cmd = [sys.executable, str(BUNDLE_SCRIPT)]
        for manifest_path in manifest_paths:
            cmd.extend(["--manifest", str(manifest_path)])
        for match_name in _match_order(batch_manifest, summary_csv):
            cmd.extend(["--match", match_name])
        for action in args.bundle_allowed_action:
            cmd.extend(["--allowed-action", action])
        cmd.extend(
            [
                "--max-per-match",
                str(args.bundle_max_per_match),
                "--out-dir",
                str(bundle_dir),
                "--overwrite",
            ]
        )
        if args.bundle_make_zip:
            cmd.append("--make-zip")
        _run(cmd)

    print(f"out_dir: {out_dir}")
    print(f"batch_manifest: {batch_manifest}")
    if summary_md.exists():
        print(f"summary_md: {summary_md}")
    if summary_csv.exists():
        print(f"summary_csv: {summary_csv}")
    if bundle_dir.exists():
        print(f"bundle_dir: {bundle_dir}")
        if args.bundle_make_zip:
            print(f"bundle_zip: {bundle_dir.parent / (bundle_dir.name + '.zip')}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a full Track B batch review flow.")
    parser.add_argument("--video-dir", default="", help="Directory containing raw match videos.")
    parser.add_argument("--pattern", action="append", default=[], help="Video glob. Repeat to add more.")
    parser.add_argument("--out-dir", required=True, help="Output directory for the batch review.")
    parser.add_argument("--model", default=None, help="Optional YOLO checkpoint path.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--low-count", type=int, default=1)
    parser.add_argument("--high-count", type=int, default=10)
    parser.add_argument("--low-conf", type=float, default=0.18)
    parser.add_argument("--min-gap-frames", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=18)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true")
    parser.add_argument("--summary-md", default="", help="Optional markdown summary output path.")
    parser.add_argument("--summary-csv", default="", help="Optional priority CSV output path.")
    parser.add_argument("--bundle-dir", default="", help="Optional Roboflow bundle output path.")
    parser.add_argument(
        "--bundle-allowed-action",
        action="append",
        default=[],
        help="Allowed bundle action. Repeat to keep multiple action types.",
    )
    parser.add_argument("--bundle-max-per-match", type=int, default=4)
    parser.add_argument("--bundle-make-zip", action="store_true")
    parsed = parser.parse_args()
    if not parsed.pattern:
        parsed.pattern = ["*.mp4"]
    if not parsed.bundle_allowed_action:
        parsed.bundle_allowed_action = ["label_priority"]
    raise SystemExit(run(parsed))


if __name__ == "__main__":
    main()
