"""Run the Track B player-only detector on a video sample.

This is a lightweight review utility for the Roboflow-trained player detector.
It samples frames from a match video, writes an annotated preview video, and
exports per-frame detection counts to CSV so hard segments can be reviewed.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from track_b_runtime import PROJECT_ROOT, resolve_device, resolve_model_path

DEFAULT_MODEL = None


def _draw_box(frame, box, conf: float, class_name: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (80, 255, 120)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {conf:.2f}"
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 112, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(14, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    model_path = resolve_model_path(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem.replace(" ", "_")
    preview_path = out_dir / f"{stem}_player_only_preview.mp4"
    csv_path = out_dir / f"{stem}_player_only_detections.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = max(1.0, fps / max(1, args.frame_stride))
    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_path), fourcc, out_fps, (width, height))

    device = resolve_device(args.device)
    model = YOLO(str(model_path))
    rows: list[dict[str, str | int | float]] = []
    frame_idx = -1
    processed = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        if frame_idx % args.frame_stride != 0:
            continue

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
            max_det=args.max_det,
        )[0]

        boxes = result.boxes
        count = 0
        avg_conf = 0.0
        if boxes is not None and len(boxes) > 0:
            count = len(boxes)
            confs = boxes.conf.detach().cpu().tolist()
            avg_conf = sum(confs) / len(confs)
            for xyxy, conf, cls_id in zip(
                boxes.xyxy.detach().cpu().tolist(),
                confs,
                boxes.cls.detach().cpu().tolist(),
            ):
                name = result.names.get(int(cls_id), "player")
                _draw_box(frame, xyxy, float(conf), name)

        rows.append(
            {
                "frame_idx": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "detections": count,
                "avg_conf": round(avg_conf, 4),
            }
        )
        processed += 1

        cv2.putText(
            frame,
            f"frame {frame_idx}/{total_frames} | players {count}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame_idx", "time_sec", "detections", "avg_conf"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    elapsed = time.time() - start
    print(f"video: {video_path}")
    print(f"model: {model_path}")
    print(f"device: {device}")
    print(f"processed_frames: {processed}/{total_frames} stride={args.frame_stride}")
    print(f"elapsed_sec: {elapsed:.1f}")
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"preview: {preview_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track B player-only detector on a video.")
    parser.add_argument("--video", required=True, help="Input match video path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO checkpoint path.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "track_b_reviews"))
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:0")
    parser.add_argument("--frame-stride", type=int, default=5, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this source frame index.")
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--no-video", action="store_true", help="Only write CSV metrics.")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
