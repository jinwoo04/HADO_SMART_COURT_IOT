"""Summarize a Track B batch evaluation directory.

This reads the batch manifest produced by `process_track_b_video_batch.py`,
aggregates hard-frame reasons per match, and writes a compact markdown summary
plus an optional CSV priority table for the next annotation pass.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hard_reason_counts(hard_manifest_path: Path) -> Counter[str]:
    if not hard_manifest_path.exists():
        return Counter()
    rows = _read_csv(hard_manifest_path)
    return Counter(row["reason"] for row in rows if row.get("reason"))


def _match_row(batch_row: dict[str, str]) -> dict[str, object]:
    hard_dir = Path(batch_row["hard_frame_dir"])
    hard_manifest_path = hard_dir / "hard_frame_manifest.csv"
    reasons = _hard_reason_counts(hard_manifest_path)

    hard_count = int(batch_row.get("hard_frame_count") or 0)
    zero_frames = int(batch_row.get("zero_frames") or 0)
    max_count = int(batch_row.get("max_count") or 0)
    avg_count = float(batch_row.get("avg_count") or 0.0)
    avg_conf = float(batch_row.get("avg_nonzero_conf") or 0.0)
    overdetect = sum(count for reason, count in reasons.items() if reason.startswith("high_count_"))
    underdetect = sum(count for reason, count in reasons.items() if reason.startswith("low_count_"))
    lowconf = sum(count for reason, count in reasons.items() if reason.startswith("low_conf_"))

    severity = (overdetect * 3) + (lowconf * 2) + underdetect + zero_frames
    return {
        "match_name": Path(batch_row["video_path"]).stem,
        "video_path": batch_row["video_path"],
        "processed_frames": int(batch_row.get("processed_frames") or 0),
        "hard_frame_count": hard_count,
        "zero_frames": zero_frames,
        "max_count": max_count,
        "avg_count": avg_count,
        "avg_nonzero_conf": avg_conf,
        "overdetect_frames": overdetect,
        "underdetect_frames": underdetect,
        "lowconf_frames": lowconf,
        "reason_counts": reasons,
        "severity_score": severity,
    }


def _top_reason_label(reasons: Counter[str]) -> str:
    if not reasons:
        return "none"
    reason, count = reasons.most_common(1)[0]
    return f"{reason} ({count})"


def _recommendation(row: dict[str, object]) -> str:
    overdetect = int(row["overdetect_frames"])
    underdetect = int(row["underdetect_frames"])
    lowconf = int(row["lowconf_frames"])

    if overdetect >= max(3, underdetect, lowconf):
        return "Focus on AR/effect overlap cleanup; keep player boxes tight."
    if lowconf >= max(3, underdetect):
        return "Prioritize blurry or partial players; relabel visible body only."
    if underdetect > 0:
        return "Add missed-player frames if the court scene is still playable."
    return "Low urgency; review only if more data is needed."


def _write_priority_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "match_name",
        "severity_score",
        "hard_frame_count",
        "overdetect_frames",
        "lowconf_frames",
        "underdetect_frames",
        "zero_frames",
        "max_count",
        "avg_count",
        "avg_nonzero_conf",
        "top_reason",
        "recommendation",
        "video_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "match_name": row["match_name"],
                    "severity_score": row["severity_score"],
                    "hard_frame_count": row["hard_frame_count"],
                    "overdetect_frames": row["overdetect_frames"],
                    "lowconf_frames": row["lowconf_frames"],
                    "underdetect_frames": row["underdetect_frames"],
                    "zero_frames": row["zero_frames"],
                    "max_count": row["max_count"],
                    "avg_count": f"{float(row['avg_count']):.3f}",
                    "avg_nonzero_conf": f"{float(row['avg_nonzero_conf']):.4f}",
                    "top_reason": _top_reason_label(row["reason_counts"]),
                    "recommendation": _recommendation(row),
                    "video_path": row["video_path"],
                }
            )


def _write_markdown(
    rows: list[dict[str, object]],
    batch_rows: list[dict[str, str]],
    out_path: Path,
    priority_csv_path: Path | None,
) -> None:
    total_videos = len(batch_rows)
    total_hard = sum(int(row["hard_frame_count"]) for row in rows)
    total_processed = sum(int(row["processed_frames"]) for row in rows)
    reason_totals: Counter[str] = Counter()
    for row in rows:
        reason_totals.update(row["reason_counts"])

    lines = [
        "# Track B Batch Summary",
        "",
        "## Overview",
        "",
        f"- Videos processed: `{total_videos}`",
        f"- Sampled frames processed: `{total_processed}`",
        f"- Hard frames mined: `{total_hard}`",
        "",
        "## Priority Matches",
        "",
        "| Match | Severity | Hard Frames | Overdetect | Low Conf | Underdetect | Top Reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {match} | {severity} | {hard} | {over} | {lowconf} | {under} | {reason} |".format(
                match=row["match_name"],
                severity=row["severity_score"],
                hard=row["hard_frame_count"],
                over=row["overdetect_frames"],
                lowconf=row["lowconf_frames"],
                under=row["underdetect_frames"],
                reason=_top_reason_label(row["reason_counts"]),
            )
        )

    lines.extend(["", "## Global Reasons", ""])
    for reason, count in reason_totals.most_common(12):
        lines.append(f"- `{reason}`: `{count}`")

    lines.extend(["", "## Recommendations", ""])
    for row in rows[:5]:
        lines.append(f"- `{row['match_name']}`: {_recommendation(row)}")

    if priority_csv_path:
        lines.extend(["", "## Artifact", "", f"- Priority CSV: `{priority_csv_path}`"])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    batch_manifest = Path(args.batch_manifest)
    if not batch_manifest.exists():
        raise FileNotFoundError(batch_manifest)

    batch_rows = _read_csv(batch_manifest)
    rows = [_match_row(batch_row) for batch_row in batch_rows if batch_row.get("status") == "ok"]
    rows.sort(
        key=lambda row: (
            -int(row["severity_score"]),
            -int(row["hard_frame_count"]),
            str(row["match_name"]),
        )
    )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    priority_csv_path: Path | None = None
    if args.out_csv:
        priority_csv_path = Path(args.out_csv)
        priority_csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_priority_csv(rows, priority_csv_path)

    _write_markdown(rows, batch_rows, out_md, priority_csv_path)

    print(f"videos: {len(batch_rows)}")
    print(f"summary_md: {out_md}")
    if priority_csv_path:
        print(f"priority_csv: {priority_csv_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Track B batch evaluation directory.")
    parser.add_argument("--batch-manifest", required=True, help="Path to batch_manifest.csv")
    parser.add_argument("--out-md", required=True, help="Path to write the markdown summary")
    parser.add_argument("--out-csv", default="", help="Optional path to write the priority CSV")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
