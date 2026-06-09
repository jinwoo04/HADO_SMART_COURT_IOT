"""Prepare a Roboflow relabeling bundle from Track B hard-frame manifests.

The bundle keeps the selected JPGs together with a flat CSV queue, a short
markdown brief, and an optional ZIP archive so the next manual Roboflow pass is
easy to pick up.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "track_b_batch_review"
REASON_RE = re.compile(r"^(low_count|high_count|low_conf)_(.+)$")


def _reason_priority(reason: str) -> tuple[float, str, str]:
    match = REASON_RE.match(reason)
    if not match:
        return 0.0, "review", "Unrecognized reason; review manually."

    kind, value = match.groups()
    if kind == "low_conf":
        conf = float(value)
        score = 100.0 + max(0.0, 0.25 - conf) * 100.0
        return score, "label_priority", "Gameplay frame with weak confidence; relabel tightly."
    if kind == "high_count":
        count = int(float(value))
        score = 80.0 + count
        return score, "label_priority", "Likely over-detection or duplicate boxes; clean player labels."

    count = int(float(value))
    if count == 0:
        return 55.0, "review_or_skip", "No detections; keep only if frame is playable and players are visible."
    if count == 1:
        return 65.0, "review_or_skip", "Very low count; use if a player is visible but missed by the detector."
    return 50.0, "review", "Low-count frame; confirm whether this should enter training."


def _load_rows(manifest_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest_path in manifest_paths:
        match_name = manifest_path.parent.name
        with manifest_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_path = Path(row["image_path"])
                if not image_path.exists():
                    continue
                row = dict(row)
                row["match_name"] = match_name
                row["manifest_path"] = str(manifest_path)
                row["image_name"] = image_path.name
                rows.append(row)
    return rows


def _filter_rows(
    rows: list[dict[str, str]],
    matches: list[str],
    allowed_actions: set[str],
    max_per_match: int,
) -> list[dict[str, str]]:
    match_filter = set(matches)
    selected: list[dict[str, str]] = []
    per_match_count: Counter[str] = Counter()

    for row in rows:
        match_name = row["match_name"]
        if match_filter and match_name not in match_filter:
            continue
        score, action, note = _reason_priority(row["reason"])
        if action not in allowed_actions:
            continue
        row["priority_score"] = f"{score:.2f}"
        row["suggested_action"] = action
        row["review_note"] = note
        if max_per_match and per_match_count[match_name] >= max_per_match:
            continue
        selected.append(row)
        per_match_count[match_name] += 1
    return selected


def _sort_rows(rows: list[dict[str, str]], match_order: list[str]) -> list[dict[str, str]]:
    order = {match: index for index, match in enumerate(match_order)}
    return sorted(
        rows,
        key=lambda row: (
            order.get(row["match_name"], 10**6),
            -float(row["priority_score"]),
            float(row["time_sec"]),
        ),
    )


def _write_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "match_name",
        "frame_idx",
        "time_sec",
        "detections",
        "avg_conf",
        "reason",
        "priority_score",
        "suggested_action",
        "review_note",
        "image_name",
        "image_path",
        "copied_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_brief(rows: list[dict[str, str]], out_path: Path) -> None:
    action_counts = Counter(row["suggested_action"] for row in rows)
    per_match = defaultdict(int)
    per_reason = Counter(row["reason"] for row in rows)
    for row in rows:
        per_match[row["match_name"]] += 1

    lines = [
        "# Roboflow Upload Bundle",
        "",
        "This bundle was prepared from Track B hard-frame manifests.",
        "",
        "## Summary",
        "",
        f"- Frames in bundle: `{len(rows)}`",
        f"- `label_priority`: `{action_counts.get('label_priority', 0)}`",
        f"- `review_or_skip`: `{action_counts.get('review_or_skip', 0)}`",
        f"- `review`: `{action_counts.get('review', 0)}`",
        "",
        "## Per-match counts",
        "",
    ]
    for match_name, count in sorted(per_match.items()):
        lines.append(f"- `{match_name}`: `{count}`")

    lines.extend(["", "## Most common reasons", ""])
    for reason, count in per_reason.most_common(10):
        lines.append(f"- `{reason}`: `{count}`")

    lines.extend(
        [
            "",
            "## Labeling reminder",
            "",
            "- Keep this pass player-only.",
            "- Prefer playable court frames over intro or scoreboard screens.",
            "- If the frame is non-playable and contains no useful player visibility, skip it.",
            "- For shield overlap, draw the player box as tightly and consistently as possible around the visible body.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    manifest_paths = [Path(path) for path in args.manifest]
    rows = _load_rows(manifest_paths)
    if not rows:
        raise FileNotFoundError("No valid rows were found in the provided manifests.")

    allowed_actions = set(args.allowed_action)
    filtered = _filter_rows(rows, args.match, allowed_actions, args.max_per_match)
    if not filtered:
        raise RuntimeError("No rows matched the current filters.")
    filtered = _sort_rows(filtered, args.match)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for row in filtered:
        src = Path(row["image_path"])
        dst_name = f"{row['match_name']}__{row['image_name']}"
        dst = frames_dir / dst_name
        shutil.copy2(src, dst)
        row["copied_path"] = str(dst)

    manifest_out = out_dir / "upload_manifest.csv"
    _write_manifest(filtered, manifest_out)
    brief_out = out_dir / "README.md"
    _write_brief(filtered, brief_out)

    if args.make_zip:
        archive_base = out_dir.parent / out_dir.name
        shutil.make_archive(str(archive_base), "zip", out_dir)

    print(f"rows_selected: {len(filtered)}")
    print(f"bundle_dir: {out_dir}")
    print(f"manifest: {manifest_out}")
    print(f"brief: {brief_out}")
    if args.make_zip:
        print(f"zip: {out_dir.parent / (out_dir.name + '.zip')}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Roboflow upload bundle from Track B manifests.")
    parser.add_argument("--manifest", action="append", required=True, help="Path to hard_frame_manifest.csv")
    parser.add_argument("--match", action="append", default=[], help="Optional match filter and sort order.")
    parser.add_argument(
        "--allowed-action",
        action="append",
        default=[],
        help="Suggested actions to keep in the bundle.",
    )
    parser.add_argument("--max-per-match", type=int, default=0, help="Optional cap per match.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "roboflow_upload_bundle"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle directory.")
    parser.add_argument("--make-zip", action="store_true", help="Create a sibling zip archive.")
    parsed = parser.parse_args()
    if not parsed.allowed_action:
        parsed.allowed_action = ["label_priority", "review_or_skip"]
    raise SystemExit(run(parsed))


if __name__ == "__main__":
    main()
