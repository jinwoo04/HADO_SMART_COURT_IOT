"""Mine hard review frames from a video detection CSV.

The companion script `run_player_model_on_video.py` writes per-frame detection
counts. This script turns suspicious rows into frame images and a contact sheet
for Roboflow relabeling.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    with Path(args.csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for row in rows:
        detections = int(row["detections"])
        avg_conf = float(row["avg_conf"])
        reason = ""
        if detections <= args.low_count:
            reason = f"low_count_{detections}"
        elif detections >= args.high_count:
            reason = f"high_count_{detections}"
        elif 0 < avg_conf <= args.low_conf:
            reason = f"low_conf_{avg_conf:.2f}"
        if reason:
            row = dict(row)
            row["reason"] = reason
            candidates.append(row)

    # Keep diverse timestamps instead of adjacent near-duplicates.
    selected: list[dict[str, str]] = []
    last_frame = -10**9
    for row in candidates:
        frame_idx = int(row["frame_idx"])
        if frame_idx - last_frame < args.min_gap_frames:
            continue
        selected.append(row)
        last_frame = frame_idx
        if len(selected) >= args.max_samples:
            break
    return selected


def _save_contact_sheet(image_paths: list[Path], out_path: Path, thumb_w: int = 320) -> None:
    if not image_paths:
        return
    thumbs = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        thumb_h = int(img.shape[0] * scale)
        img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.putText(
            img,
            p.stem[-32:],
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(img)
    if not thumbs:
        return

    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    h = max(t.shape[0] for t in thumbs)
    sheet = np.zeros((rows * h, cols * thumb_w, 3), dtype=np.uint8)
    for i, img in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h:r * h + img.shape[0], c * thumb_w:(c + 1) * thumb_w] = img
    cv2.imwrite(str(out_path), sheet)


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    selected = _read_candidates(args)
    wanted = {int(row["frame_idx"]): row for row in selected}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved_paths: list[Path] = []
    frame_idx = -1
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        row = wanted.pop(frame_idx, None)
        if row is None:
            continue
        reason = row["reason"]
        time_sec = float(row["time_sec"])
        name = f"frame_{frame_idx:06d}_{time_sec:07.3f}s_{reason}.jpg"
        out_path = frames_dir / name
        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

    cap.release()

    manifest_path = out_dir / "hard_frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_idx", "time_sec", "detections", "avg_conf", "reason", "image_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, img_path in zip(selected, saved_paths):
            row = dict(row)
            row["image_path"] = str(img_path)
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    contact_sheet_path = out_dir / "hard_frame_contact_sheet.jpg"
    _save_contact_sheet(saved_paths, contact_sheet_path)

    print(f"selected_rows: {len(selected)}")
    print(f"saved_frames: {len(saved_paths)}")
    print(f"manifest: {manifest_path}")
    print(f"contact_sheet: {contact_sheet_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard frames from player detector CSV.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "track_b_reviews" / "hard_frames"))
    parser.add_argument("--low-count", type=int, default=1)
    parser.add_argument("--high-count", type=int, default=8)
    parser.add_argument("--low-conf", type=float, default=0.18)
    parser.add_argument("--min-gap-frames", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=40)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
