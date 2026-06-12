"""Build a human review packet for Track B Roboflow relabeling.

The Roboflow upload bundle is optimized for upload, not for human pre-review.
This script reads an `upload_manifest.csv` and creates a compact packet with:

- `relabel_contact_sheet.jpg`
- `relabel_checklist.csv`
- `README.md`

Use it before a manual Roboflow pass so collaborators can quickly understand
which frames are most important and why they were selected.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from track_b_runtime import PROJECT_ROOT


REASON_RE = re.compile(r"^(low_count|high_count|low_conf)_(.+)$")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _issue_label(reason: str) -> str:
    match = REASON_RE.match(reason)
    if not match:
        return "review"
    kind, value = match.groups()
    if kind == "high_count":
        return "overdetect_effect_or_duplicate"
    if kind == "low_count":
        return "possible_missed_player"
    if kind == "low_conf":
        return "weak_or_partial_player_visibility"
    return value


def _action_hint(reason: str) -> str:
    issue = _issue_label(reason)
    if issue == "overdetect_effect_or_duplicate":
        return "Delete fake player boxes on effects/shields; keep real players tight."
    if issue == "possible_missed_player":
        return "Add missing visible players only if the frame is playable."
    if issue == "weak_or_partial_player_visibility":
        return "Tighten boxes around visible body; skip unclear non-play frames."
    return "Review manually and keep player-only labels."


def _priority_key(row: dict[str, str]) -> tuple[float, int, str, float]:
    score = float(row.get("priority_score") or 0.0)
    detections = int(float(row.get("detections") or 0))
    time_sec = float(row.get("time_sec") or 0.0)
    return (-score, -detections, row.get("match_name", ""), time_sec)


def _load_rows(manifest_path: Path, max_rows: int) -> list[dict[str, str]]:
    rows = _read_csv(manifest_path)
    rows.sort(key=_priority_key)
    if max_rows:
        rows = rows[:max_rows]
    for index, row in enumerate(rows, start=1):
        row["review_order"] = str(index)
        row["likely_issue"] = _issue_label(row.get("reason", ""))
        row["relabel_hint"] = _action_hint(row.get("reason", ""))
    return rows


def _put_label(img: np.ndarray, text: str, xy: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _make_thumb(row: dict[str, str], thumb_w: int, thumb_h: int) -> np.ndarray:
    img_path = _resolve_path(row["copied_path"])
    img = cv2.imread(str(img_path))
    if img is None:
        thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        _put_label(thumb, "missing image", (12, thumb_h // 2), 0.65, (80, 80, 255))
        return thumb

    h, w = img.shape[:2]
    scale = min(thumb_w / max(1, w), (thumb_h - 44) / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    thumb = np.full((thumb_h, thumb_w, 3), 18, dtype=np.uint8)
    x0 = (thumb_w - new_w) // 2
    y0 = 26
    thumb[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    reason = row.get("reason", "")
    color = (80, 210, 255)
    if reason.startswith("high_count"):
        color = (0, 120, 255)
    elif reason.startswith("low_count"):
        color = (255, 180, 80)
    elif reason.startswith("low_conf"):
        color = (120, 220, 120)

    header = "#{order} {match} f{frame} det={det}".format(
        order=row.get("review_order", ""),
        match=row.get("match_name", ""),
        frame=row.get("frame_idx", ""),
        det=row.get("detections", ""),
    )
    footer = f"{reason}  score={row.get('priority_score', '')}"
    cv2.rectangle(thumb, (0, 0), (thumb_w - 1, 24), (35, 35, 35), -1)
    cv2.rectangle(thumb, (0, thumb_h - 20), (thumb_w - 1, thumb_h - 1), (35, 35, 35), -1)
    _put_label(thumb, header, (8, 17), 0.48, color)
    _put_label(thumb, footer[:62], (8, thumb_h - 6), 0.43, (220, 220, 220))
    cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), color, 1)
    return thumb


def _write_contact_sheet(rows: list[dict[str, str]], out_path: Path, cols: int, thumb_w: int, thumb_h: int) -> None:
    if not rows:
        raise ValueError("No rows available for contact sheet.")
    cols = max(1, cols)
    sheet_rows = math.ceil(len(rows) / cols)
    sheet = np.full((sheet_rows * thumb_h, cols * thumb_w, 3), 28, dtype=np.uint8)
    for index, row in enumerate(rows):
        r = index // cols
        c = index % cols
        thumb = _make_thumb(row, thumb_w, thumb_h)
        y0 = r * thumb_h
        x0 = c * thumb_w
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _write_checklist(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "review_order",
        "match_name",
        "frame_idx",
        "time_sec",
        "detections",
        "avg_conf",
        "reason",
        "priority_score",
        "likely_issue",
        "relabel_hint",
        "image_name",
        "copied_path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_readme(rows: list[dict[str, str]], out_path: Path, contact_sheet: Path, checklist: Path) -> None:
    reasons = Counter(row.get("reason", "") for row in rows)
    issues = Counter(row.get("likely_issue", "") for row in rows)
    per_match: dict[str, int] = defaultdict(int)
    for row in rows:
        per_match[row.get("match_name", "")] += 1

    lines = [
        "# Track B Relabel Review Packet",
        "",
        "This packet is for human pre-review before the Roboflow player-only relabel pass.",
        "",
        "## Artifacts",
        "",
        f"- Contact sheet: `{contact_sheet}`",
        f"- Checklist CSV: `{checklist}`",
        "",
        "## Summary",
        "",
        f"- Frames: `{len(rows)}`",
        "",
        "## Likely Issues",
        "",
    ]
    for issue, count in issues.most_common():
        lines.append(f"- `{issue}`: `{count}`")

    lines.extend(["", "## Reasons", ""])
    for reason, count in reasons.most_common():
        lines.append(f"- `{reason}`: `{count}`")

    lines.extend(["", "## Per Match", ""])
    for match_name, count in sorted(per_match.items()):
        lines.append(f"- `{match_name}`: `{count}`")

    lines.extend(
        [
            "",
            "## Relabel Rules",
            "",
            "- Keep this pass `player-only`.",
            "- Delete fake player boxes on AR effects, shields, projectiles, glow, or duplicated detections.",
            "- Add missing `player` boxes only when a real player is clearly visible.",
            "- Keep boxes tight around the visible body; do not include the surrounding effect glow.",
            "- Skip non-playable frames such as intro, roster, scoreboard, or full-screen UI.",
            "",
            "## Suggested Order",
            "",
            "Review the contact sheet in numeric `#order` order, then use the same `review_order` in the checklist CSV.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(manifest_path, args.max_rows)
    if not rows:
        raise ValueError("No manifest rows were found.")

    contact_sheet = out_dir / "relabel_contact_sheet.jpg"
    checklist = out_dir / "relabel_checklist.csv"
    readme = out_dir / "README.md"

    _write_contact_sheet(rows, contact_sheet, args.cols, args.thumb_width, args.thumb_height)
    _write_checklist(rows, checklist)
    _write_readme(rows, readme, contact_sheet, checklist)

    print(f"rows: {len(rows)}")
    print(f"contact_sheet: {contact_sheet}")
    print(f"checklist: {checklist}")
    print(f"readme: {readme}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Track B Roboflow relabel review packet.")
    parser.add_argument("--manifest", required=True, help="Path to Roboflow upload_manifest.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for the review packet")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional limit for smoke tests")
    parser.add_argument("--cols", type=int, default=4, help="Contact sheet columns")
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=250)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
