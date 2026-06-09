"""Prepare a small Track B occlusion dataset from HADO match video.

This tool creates a YOLO-style dataset for the long-term AR/effect-overlap
project. Player labels are auto-labeled with YOLO. Effect labels are weak
color/brightness candidates intended for Roboflow review, not final truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = ["player", "effect"]


@dataclass
class FrameRecord:
    path: Path
    frame_idx: int
    time_sec: float
    split: str
    player_count: int
    effect_count: int
    effect_score: float


def _load_yolo(model_path: str):
    from ultralytics import YOLO  # type: ignore

    return YOLO(model_path)


def _xyxy_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / width, cy / height, bw / width, bh / height


def _detect_players(model, frame: np.ndarray, conf: float, imgsz: int) -> list[tuple[float, float, float, float, float]]:
    results = model(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False)
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    out = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        if (x2 - x1) * (y2 - y1) < 250:
            continue
        out.append((x1, y1, x2, y2, float(score)))
    return out


def _parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--effect-roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--effect-roi must have x2>x1 and y2>y1")
    return x1, y1, x2, y2


def _effect_candidates(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    max_boxes: int = 6,
) -> tuple[list[tuple[float, float, float, float]], float]:
    """Find saturated/bright AR effect candidates.

    This intentionally favors recall. The output should be reviewed in Roboflow.
    """
    height, width = frame.shape[:2]
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        rx1 = max(0, min(width - 1, rx1))
        ry1 = max(0, min(height - 1, ry1))
        rx2 = max(rx1 + 1, min(width, rx2))
        ry2 = max(ry1 + 1, min(height, ry2))
        work = frame[ry1:ry2, rx1:rx2]
    else:
        rx1 = ry1 = 0
        work = frame

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    saturated = (s > 95) & (v > 130)
    bright = (v > 225) & (s > 45)
    mask = np.where(saturated | bright, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_h, roi_w = work.shape[:2]
    boxes: list[tuple[float, float, float, float]] = []
    total_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 120:
            continue
        x, y, w, hh = cv2.boundingRect(contour)
        if w < 8 or hh < 8:
            continue
        if area > roi_w * roi_h * 0.18:
            continue
        # Ignore tiny UI-like strips near exact borders.
        if (y < 6 or y + hh > roi_h - 6) and area < 1500:
            continue
        boxes.append((float(x + rx1), float(y + ry1), float(x + w + rx1), float(y + hh + ry1)))
        total_area += area

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:max_boxes], total_area / float(width * height)


def _split_for_index(i: int, rng: random.Random) -> str:
    value = rng.random()
    if value < 0.75:
        return "train"
    if value < 0.90:
        return "valid"
    return "test"


def _write_data_yaml(out_dir: Path) -> None:
    data = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def prepare_dataset(args: argparse.Namespace) -> list[FrameRecord]:
    video_path = Path(args.video)
    out_dir = Path(args.out)
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    sample_every = max(1, int(round(fps / args.sample_fps)))
    max_frames = args.max_frames
    stride = sample_every
    if max_frames and math.ceil(frame_count / stride) > max_frames:
        stride = max(stride, math.ceil(frame_count / max_frames))

    model = _load_yolo(args.model)
    rng = random.Random(args.seed)
    effect_roi = _parse_roi(args.effect_roi)
    records: list[FrameRecord] = []
    metadata_rows: list[dict] = []

    frame_idx = 0
    sampled_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        split = _split_for_index(sampled_idx, rng)
        stem = f"hado_{sampled_idx:05d}_f{frame_idx:06d}_t{frame_idx / fps:07.2f}"
        image_path = out_dir / split / "images" / f"{stem}.jpg"
        label_path = out_dir / split / "labels" / f"{stem}.txt"

        players = _detect_players(model, frame, conf=args.player_conf, imgsz=args.imgsz)
        effects, effect_score = _effect_candidates(frame, roi=effect_roi, max_boxes=args.max_effect_boxes)
        height, width = frame.shape[:2]

        lines: list[str] = []
        for x1, y1, x2, y2, score in players:
            cx, cy, bw, bh = _xyxy_to_yolo((x1, y1, x2, y2), width, height)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        for box in effects:
            cx, cy, bw, bh = _xyxy_to_yolo(box, width, height)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        records.append(FrameRecord(
            path=image_path,
            frame_idx=frame_idx,
            time_sec=frame_idx / fps,
            split=split,
            player_count=len(players),
            effect_count=len(effects),
            effect_score=effect_score,
        ))
        metadata_rows.append({
            "image": str(image_path.relative_to(out_dir)),
            "frame_idx": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "split": split,
            "player_count": len(players),
            "effect_count": len(effects),
            "effect_score": round(effect_score, 6),
            "label_quality": "weak_auto_effect_labels_review_required",
        })

        sampled_idx += 1
        if max_frames and sampled_idx >= max_frames:
            break
        frame_idx += 1

    cap.release()
    _write_data_yaml(out_dir)

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [])
        if metadata_rows:
            writer.writeheader()
            writer.writerows(metadata_rows)

    summary = {
        "source_video": str(video_path),
        "duration_sec": duration,
        "source_fps": fps,
        "source_frames": frame_count,
        "sample_stride_frames": stride,
        "sampled_frames": len(records),
        "effect_roi": list(effect_roi) if effect_roi else None,
        "classes": CLASS_NAMES,
        "player_labels": sum(r.player_count for r in records),
        "effect_candidate_labels": sum(r.effect_count for r in records),
        "splits": {split: sum(1 for r in records if r.split == split) for split in ["train", "valid", "test"]},
        "notes": [
            "Player labels are YOLO auto-labels.",
            "Effect labels are weak HSV/brightness candidates and must be reviewed before serious training.",
            "Use this dataset first for Roboflow review and baseline Track B experiments.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Track B occlusion YOLO dataset from video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path for player auto-labeling")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Target extraction FPS")
    parser.add_argument("--max-frames", type=int, default=80, help="Hard cap for extracted frames")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--player-conf", type=float, default=0.25, help="Player auto-label confidence threshold")
    parser.add_argument("--effect-roi", default=None, help="Restrict weak effect boxes to x1,y1,x2,y2 pixels")
    parser.add_argument("--max-effect-boxes", type=int, default=6, help="Max weak effect boxes per frame")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Delete output directory before writing")
    args = parser.parse_args()

    records = prepare_dataset(args)
    print(f"Prepared {len(records)} frames at {Path(args.out).resolve()}")
    print(f"Players: {sum(r.player_count for r in records)}")
    print(f"Effect candidates: {sum(r.effect_count for r in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
