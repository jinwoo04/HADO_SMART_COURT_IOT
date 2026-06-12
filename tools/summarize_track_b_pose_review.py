"""Summarize Track B pose/skeleton review artifacts.

`export_track_b_pose_review.py` writes one directory per video probe. This
script scans those outputs and creates a compact table for comparing clean,
heavy-effect, and follow-up runs without opening each CSV by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from track_b_runtime import PROJECT_ROOT


DEFAULT_ROOT = PROJECT_ROOT / "data" / "track_b_pose_review"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value: str | int | float | None) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _resolve_path(value: str, summary_path: Path) -> Path:
    if not value:
        return Path()
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        PROJECT_ROOT / path,
        summary_path.parent / path,
        summary_path.parent / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def _top_counts(counter: Counter[str], limit: int = 4) -> str:
    if not counter:
        return ""
    return ", ".join(f"{key}:{count}" for key, count in counter.most_common(limit))


def _review_label(summary_path: Path, video_path: str) -> str:
    parent = summary_path.parent.name
    video_stem = Path(video_path).stem if video_path else summary_path.stem.replace("_pose_summary", "")
    if parent and parent != DEFAULT_ROOT.name:
        return parent
    return video_stem


def _recommendation(row: dict[str, object]) -> str:
    processed_frames = int(row["processed_frames"])
    rows_per_frame = float(row["rows_per_processed_frame"])
    churn = float(row["track_churn_ratio"])
    avg_conf = float(row["avg_det_conf"])
    ready_ratio = float(row["ready_ratio"])
    oob_ratio = float(row["court_oob_ratio"])

    if processed_frames < 10:
        return "Sample is too small; use at least 30 processed frames before judging."
    if rows_per_frame < 0.5:
        return "Pose coverage is sparse; improve player detection before action review."
    if churn > 0.8:
        return "Track IDs churn heavily; compare again after player-only relabel/retrain."
    if avg_conf < 0.35:
        return "Detector confidence is weak; prioritize cleaner player boxes and low-conf frames."
    if oob_ratio > 0.25:
        return "Court coordinates are noisy; use real calibration before position judgments."
    if ready_ratio > 0.8:
        return "Actions are mostly ready; collect deliberate action clips or use denser sampling."
    return "Good candidate for deeper skeleton/action verification."


def _extract_row(summary_path: Path) -> dict[str, object]:
    summary = _read_json(summary_path)
    csv_path = _resolve_path(str(summary.get("csv_path", "")), summary_path)
    rows = _read_csv(csv_path) if csv_path.exists() else []

    frames = {row.get("frame_idx", "") for row in rows if row.get("frame_idx", "") != ""}
    tracks = {row.get("track_id", "") for row in rows if row.get("track_id", "") != ""}
    actions = Counter(row.get("action", "unknown") or "unknown" for row in rows)
    postures = Counter(row.get("posture", "unknown") or "unknown" for row in rows)
    det_conf = [_as_float(row.get("det_conf")) for row in rows if row.get("det_conf", "") != ""]
    action_conf = [_as_float(row.get("action_conf")) for row in rows if row.get("action_conf", "") != ""]

    court_rows = [
        (_as_float(row.get("court_x_m")), _as_float(row.get("court_y_m")))
        for row in rows
        if row.get("court_x_m", "") != "" and row.get("court_y_m", "") != ""
    ]
    court_oob = [
        (x, y)
        for x, y in court_rows
        if x < 0.0 or x > 10.0 or y < 0.0 or y > 6.0
    ]

    processed_frames = int(summary.get("processed_frames") or 0)
    track_rows = len(rows)
    frames_with_tracks = len(frames)
    unique_tracks = len(tracks)
    rows_per_processed_frame = track_rows / processed_frames if processed_frames else 0.0
    tracks_per_track_frame = track_rows / frames_with_tracks if frames_with_tracks else 0.0
    track_churn_ratio = unique_tracks / frames_with_tracks if frames_with_tracks else 0.0
    avg_det_conf = sum(det_conf) / len(det_conf) if det_conf else 0.0
    avg_action_conf = sum(action_conf) / len(action_conf) if action_conf else 0.0
    ready_ratio = actions.get("ready", 0) / track_rows if track_rows else 0.0
    court_oob_ratio = len(court_oob) / len(court_rows) if court_rows else 0.0

    row: dict[str, object] = {
        "review": _review_label(summary_path, str(summary.get("video", ""))),
        "video": summary.get("video", ""),
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
        "processed_frames": processed_frames,
        "track_rows": track_rows,
        "frames_with_tracks": frames_with_tracks,
        "unique_tracks": unique_tracks,
        "rows_per_processed_frame": rows_per_processed_frame,
        "tracks_per_track_frame": tracks_per_track_frame,
        "track_churn_ratio": track_churn_ratio,
        "avg_det_conf": avg_det_conf,
        "avg_action_conf": avg_action_conf,
        "ready_ratio": ready_ratio,
        "court_oob_ratio": court_oob_ratio,
        "top_actions": _top_counts(actions),
        "top_postures": _top_counts(postures),
    }
    row["recommendation"] = _recommendation(row)
    return row


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "review",
        "processed_frames",
        "track_rows",
        "frames_with_tracks",
        "unique_tracks",
        "rows_per_processed_frame",
        "tracks_per_track_frame",
        "track_churn_ratio",
        "avg_det_conf",
        "avg_action_conf",
        "ready_ratio",
        "court_oob_ratio",
        "top_actions",
        "top_postures",
        "recommendation",
        "video",
        "csv_path",
        "summary_path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for key in (
                "rows_per_processed_frame",
                "tracks_per_track_frame",
                "track_churn_ratio",
                "avg_det_conf",
                "avg_action_conf",
                "ready_ratio",
                "court_oob_ratio",
            ):
                payload[key] = f"{float(payload[key]):.4f}"
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def _write_markdown(rows: list[dict[str, object]], out_path: Path, csv_path: Path) -> None:
    lines = [
        "# Track B Pose Review Summary",
        "",
        "This summary aggregates `export_track_b_pose_review.py` outputs.",
        "",
        f"- Reviews found: `{len(rows)}`",
        f"- CSV: `{csv_path}`",
        "",
        "## Review Table",
        "",
        "| Review | Frames | Rows | Track Frames | Unique Tracks | Rows/Frame | Churn | Det Conf | Action Conf | Ready % | Top Actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {review} | {frames} | {rows} | {track_frames} | {tracks} | {rpf:.2f} | {churn:.2f} | {det:.2f} | {act:.2f} | {ready:.0%} | {actions} |".format(
                review=row["review"],
                frames=int(row["processed_frames"]),
                rows=int(row["track_rows"]),
                track_frames=int(row["frames_with_tracks"]),
                tracks=int(row["unique_tracks"]),
                rpf=float(row["rows_per_processed_frame"]),
                churn=float(row["track_churn_ratio"]),
                det=float(row["avg_det_conf"]),
                act=float(row["avg_action_conf"]),
                ready=float(row["ready_ratio"]),
                actions=row["top_actions"] or "-",
            )
        )

    lines.extend(["", "## Recommendations", ""])
    for row in rows:
        lines.append(f"- `{row['review']}`: {row['recommendation']}")

    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "- Compare the same match before and after a player-only retrain.",
            "- A good follow-up should increase `Rows/Frame` while reducing churn.",
            "- Treat court coordinates as approximate unless the review used real calibration.",
            "- If `Ready %` stays very high, collect deliberate action clips or sample more densely.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    summary_paths = sorted(root.rglob("*_pose_summary.json"))
    rows = [_extract_row(path) for path in summary_paths]
    rows.sort(
        key=lambda row: (
            str(row["review"]),
            str(row["video"]),
        )
    )

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md, out_csv)

    print(f"reviews: {len(rows)}")
    print(f"summary_csv: {out_csv}")
    print(f"summary_md: {out_md}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Track B pose review outputs.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Root directory containing pose review outputs.")
    parser.add_argument(
        "--out-csv",
        default=str(DEFAULT_ROOT / "pose_review_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_ROOT / "pose_review_summary.md"),
        help="Output markdown path.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
