"""Export skeleton/action/court-position review artifacts from a match video.

This is the next-layer utility after player-only detection review. It runs a
pose model on a video, tracks players, estimates court coordinates, classifies
HADO actions, and writes:

- an overlay review video (optional)
- a flat CSV for quick filtering
- a JSONL file with raw keypoints per tracked player frame
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "track_b_pose_review"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detection, PersonDetector
from src.homography import Calibration, compute_homography, pixel_to_court
from src.pose import analyze_pose, classify_hado_action, draw_posture_label, draw_skeleton, sample_vest_hue
from src.tracker import IoUTracker, Track


def _resolve_pose_model(force_pt: bool) -> str:
    ncnn_path = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx_path = PROJECT_ROOT / "yolov8n-pose.onnx"
    pt_path = PROJECT_ROOT / "yolov8n-pose.pt"
    if not force_pt and ncnn_path.exists():
        return str(ncnn_path)
    if not force_pt and onnx_path.exists():
        return str(onnx_path)
    return str(pt_path if pt_path.exists() else Path("yolov8n-pose.pt"))


def _approx_calib(frame_w: int, frame_h: int) -> Calibration:
    margin = 0.10
    corners = np.array(
        [
            [frame_w * margin, frame_h * margin],
            [frame_w * (1 - margin), frame_h * margin],
            [frame_w * (1 - margin), frame_h * (1 - margin)],
            [frame_w * margin, frame_h * (1 - margin)],
        ],
        dtype=np.float32,
    )
    return compute_homography(
        corners_pixel=corners,
        court_width_m=10.0,
        court_height_m=6.0,
        image_size=(frame_w, frame_h),
    )


def _match_detection(track: Track, detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return min(
        detections,
        key=lambda d: abs(d.foot_point[0] - track.foot_point[0]) + abs(d.foot_point[1] - track.foot_point[1]),
    )


def _team_from_center(det: Detection, frame_center_x: float) -> str:
    return "A" if det.center[0] < frame_center_x else "B"


def _draw_overlay(
    frame: np.ndarray,
    track: Track,
    det: Detection,
    court_xy: tuple[float, float] | None,
    action_label: str,
    action_conf: float,
    posture_label: str,
) -> None:
    team_color = (90, 210, 120) if det.center[0] < frame.shape[1] / 2 else (100, 170, 255)
    cv2.rectangle(frame, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), team_color, 2)
    draw_skeleton(frame, det, base_color=team_color)
    posture = analyze_pose(det)
    if posture:
        draw_posture_label(frame, det, posture)

    info_y = max(14, int(det.y1) - 8)
    court_txt = ""
    if court_xy is not None:
        court_txt = f" ({court_xy[0]:.1f}m,{court_xy[1]:.1f}m)"
    label = f"#{track.track_id} {action_label} {action_conf:.2f}{court_txt}"
    cv2.putText(
        frame,
        label,
        (int(det.x1), info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        team_color,
        1,
        cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem.replace(" ", "_")

    pose_model = args.model or _resolve_pose_model(args.pt)
    detector = PersonDetector(
        model_path=pose_model,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        device=args.device,
    )
    tracker = IoUTracker(iou_threshold=args.iou, max_lost_frames=args.max_lost, max_history=90)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    calib = _approx_calib(frame_w, frame_h)

    overlay_path = out_dir / f"{stem}_pose_overlay.mp4"
    csv_path = out_dir / f"{stem}_pose_tracks.csv"
    jsonl_path = out_dir / f"{stem}_pose_tracks.jsonl"
    summary_path = out_dir / f"{stem}_pose_summary.json"

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            min(fps, 30.0),
            (frame_w, frame_h),
        )

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "frame_idx",
        "time_sec",
        "track_id",
        "team_hint",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "det_conf",
        "court_x_m",
        "court_y_m",
        "posture",
        "crouch_score",
        "arm_spread",
        "action",
        "action_conf",
        "vest_hue",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    action_counts: dict[str, int] = {}
    posture_counts: dict[str, int] = {}
    processed_frames = 0

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.frame_stride != 0:
            continue
        if args.max_frames and processed_frames >= args.max_frames:
            break
        processed_frames += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        foot_pts = (
            np.array([[t.foot_point[0], t.foot_point[1]] for t in tracks], dtype=np.float32)
            if tracks
            else np.empty((0, 2), dtype=np.float32)
        )
        court_pts = pixel_to_court(foot_pts, calib) if len(foot_pts) else np.empty((0, 2), dtype=np.float32)

        overlay = frame.copy() if writer is not None else frame
        for i, track in enumerate(tracks):
            det = _match_detection(track, detections)
            if det is None or det.keypoints is None:
                continue
            posture = analyze_pose(det)
            action = classify_hado_action(det, frame_center_x=frame_w / 2)
            vest_hue = sample_vest_hue(frame, det)
            court_xy = None
            if i < len(court_pts):
                court_xy = (float(court_pts[i][0]), float(court_pts[i][1]))
            action_label = action.action if action else "unknown"
            action_conf = float(action.confidence) if action else 0.0
            posture_label = posture.posture if posture else "unknown"
            crouch_score = float(posture.crouch_score) if posture else 0.0
            arm_spread = float(posture.arm_spread) if posture else 0.0
            team_hint = _team_from_center(det, frame_w / 2)

            action_counts[action_label] = action_counts.get(action_label, 0) + 1
            posture_counts[posture_label] = posture_counts.get(posture_label, 0) + 1

            row = {
                "frame_idx": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "track_id": track.track_id,
                "team_hint": team_hint,
                "bbox_x1": round(det.x1, 2),
                "bbox_y1": round(det.y1, 2),
                "bbox_x2": round(det.x2, 2),
                "bbox_y2": round(det.y2, 2),
                "det_conf": round(det.confidence, 4),
                "court_x_m": round(court_xy[0], 4) if court_xy else "",
                "court_y_m": round(court_xy[1], 4) if court_xy else "",
                "posture": posture_label,
                "crouch_score": round(crouch_score, 4),
                "arm_spread": round(arm_spread, 4),
                "action": action_label,
                "action_conf": round(action_conf, 4),
                "vest_hue": vest_hue,
            }
            csv_writer.writerow(row)

            keypoints_payload = det.keypoints.tolist() if det.keypoints is not None else None
            jsonl_file.write(
                json.dumps(
                    {
                        **row,
                        "keypoints": keypoints_payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if writer is not None:
                _draw_overlay(overlay, track, det, court_xy, action_label, action_conf, posture_label)

        if writer is not None:
            writer.write(overlay)

    cap.release()
    csv_file.close()
    jsonl_file.close()
    if writer is not None:
        writer.release()

    summary = {
        "video": str(video_path),
        "model": pose_model,
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "frame_stride": args.frame_stride,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "overlay_path": str(overlay_path) if not args.no_video else "",
        "action_counts": action_counts,
        "posture_counts": posture_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"processed_frames: {processed_frames}")
    print(f"csv: {csv_path}")
    print(f"jsonl: {jsonl_path}")
    if not args.no_video:
        print(f"overlay: {overlay_path}")
    print(f"summary: {summary_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Track B skeleton/action/court review artifacts from a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default="", help="Optional pose model path")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--max-lost", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--pt", action="store_true", help="Force .pt model instead of NCNN/ONNX auto choice")
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video export")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
