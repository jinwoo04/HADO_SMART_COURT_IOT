"""Build an OpenAI Batch API request file for Track B frame QA.

This utility converts a Track B `upload_manifest.csv` into a `.jsonl` file that
can be uploaded to the OpenAI Batch API. Each request uses the Responses API
with image input plus a strict JSON schema so the returned QA can be merged back
into the relabeling queue.

The default mode embeds local images as base64 data URLs for convenience.
For larger batches, prefer `--image-ref-mode file-id` with an uploaded file-id
mapping to keep the batch input file smaller.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
from pathlib import Path

from track_b_runtime import PROJECT_ROOT


SCHEMA_NAME = "track_b_frame_review"
SCHEMA = {
    "type": "object",
    "properties": {
        "visible_player_count": {"type": "integer", "minimum": 0, "maximum": 6},
        "likely_playable": {"type": "boolean"},
        "occlusion_level": {
            "type": "string",
            "enum": ["none", "partial", "heavy", "unclear"],
        },
        "detector_issue": {
            "type": "string",
            "enum": ["none", "overdetect", "underdetect", "lowconf", "unclear"],
        },
        "relabel_decision": {
            "type": "string",
            "enum": ["label_priority", "review", "review_or_skip"],
        },
        "priority_adjustment": {"type": "integer", "minimum": -3, "maximum": 3},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": [
        "visible_player_count",
        "likely_playable",
        "occlusion_level",
        "detector_issue",
        "relabel_decision",
        "priority_adjustment",
        "confidence",
        "notes",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You review HADO match frames for player-only relabeling. "
    "Focus on whether visible real players should be relabeled, whether the "
    "frame is playable, and whether the detector likely over-detected effects "
    "or under-detected players. Ignore non-player AR effects unless they block "
    "player visibility. Return only the requested JSON fields."
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_file_id_map(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    mapping: dict[str, str] = {}
    for row in rows:
        file_id = row.get("file_id", "").strip()
        if not file_id:
            continue
        for key in ("image_name", "copied_path", "image_path", "custom_id"):
            value = row.get(key, "").strip()
            if value:
                mapping[value] = file_id
    return mapping


def _image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_ref(
    row: dict[str, str],
    image_path: Path,
    mode: str,
    file_id_map: dict[str, str],
    image_url_prefix: str,
) -> dict[str, str]:
    if mode == "base64":
        return {"type": "input_image", "image_url": _image_data_url(image_path)}

    if mode == "file-id":
        for key in (
            row.get("image_name", ""),
            row.get("copied_path", ""),
            row.get("image_path", ""),
            row.get("custom_id", ""),
        ):
            value = key.strip()
            if value and value in file_id_map:
                return {"type": "input_image", "file_id": file_id_map[value]}
        raise KeyError(f"No file_id mapping found for {row.get('image_name', image_path.name)}")

    if mode == "image-url":
        if not image_url_prefix:
            raise ValueError("--image-url-prefix is required with --image-ref-mode image-url")
        return {
            "type": "input_image",
            "image_url": f"{image_url_prefix.rstrip('/')}/{image_path.name}",
        }

    raise ValueError(f"Unsupported image ref mode: {mode}")


def _user_prompt(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "Review this HADO frame for Track B player-only relabeling.",
            "",
            "Frame metadata:",
            f"- match_name: {row.get('match_name', '')}",
            f"- frame_idx: {row.get('frame_idx', '')}",
            f"- time_sec: {row.get('time_sec', '')}",
            f"- detector_count: {row.get('detections', '')}",
            f"- avg_conf: {row.get('avg_conf', '')}",
            f"- mined_reason: {row.get('reason', '')}",
            f"- current_suggested_action: {row.get('suggested_action', '')}",
            f"- review_note: {row.get('review_note', '')}",
            "",
            "Return a strict JSON review using this guidance:",
            "- likely_playable=true only if the frame shows a real playable court moment.",
            "- detector_issue=overdetect when player boxes probably include AR effects or duplicates.",
            "- detector_issue=underdetect when visible players appear to be missing.",
            "- detector_issue=lowconf when the main problem looks like weak confidence or blurry visibility.",
            "- relabel_decision=label_priority for useful playable relabel frames.",
            "- relabel_decision=review_or_skip for non-playable or low-value frames.",
            "- priority_adjustment should be between -3 and 3.",
        ]
    )


def _request_body(
    model: str,
    row: dict[str, str],
    image_ref: dict[str, str],
    max_output_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _user_prompt(row)},
                    image_ref,
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "max_output_tokens": max_output_tokens,
    }


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    rows = _read_csv(manifest_path)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No manifest rows were found.")

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_manifest = Path(args.out_manifest) if args.out_manifest else out_jsonl.with_suffix(".manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    file_id_map: dict[str, str] = {}
    if args.file_id_manifest:
        file_id_map = _load_file_id_map(Path(args.file_id_manifest))

    request_rows: list[dict[str, str]] = []
    total_image_bytes = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            image_path = Path(row["copied_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            custom_id = f"trackb-{row.get('match_name', 'match')}-{row.get('frame_idx', idx)}-{idx:04d}"
            image_ref = _image_ref(row, image_path, args.image_ref_mode, file_id_map, args.image_url_prefix)
            if args.image_ref_mode == "base64":
                total_image_bytes += image_path.stat().st_size

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": _request_body(args.model, row, image_ref, args.max_output_tokens),
            }
            f.write(json.dumps(request, ensure_ascii=True) + "\n")

            request_rows.append(
                {
                    "custom_id": custom_id,
                    "match_name": row.get("match_name", ""),
                    "frame_idx": row.get("frame_idx", ""),
                    "time_sec": row.get("time_sec", ""),
                    "reason": row.get("reason", ""),
                    "priority_score": row.get("priority_score", ""),
                    "image_name": row.get("image_name", ""),
                    "copied_path": str(image_path),
                    "image_ref_mode": args.image_ref_mode,
                }
            )

    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "custom_id",
                "match_name",
                "frame_idx",
                "time_sec",
                "reason",
                "priority_score",
                "image_name",
                "copied_path",
                "image_ref_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(request_rows)

    print(f"requests: {len(request_rows)}")
    print(f"out_jsonl: {out_jsonl}")
    print(f"out_manifest: {out_manifest}")
    print(f"image_ref_mode: {args.image_ref_mode}")
    if args.image_ref_mode == "base64":
        print(f"embedded_image_bytes: {total_image_bytes}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OpenAI Batch API input file for Track B frame QA.")
    parser.add_argument("--manifest", required=True, help="Track B upload_manifest.csv path")
    parser.add_argument("--out-jsonl", required=True, help="Batch input .jsonl output path")
    parser.add_argument("--out-manifest", default="", help="Optional custom_id mapping CSV path")
    parser.add_argument("--model", default="gpt-5.5", help="Responses model to use for frame QA")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row limit for smoke tests")
    parser.add_argument(
        "--image-ref-mode",
        choices=["base64", "file-id", "image-url"],
        default="base64",
        help="How to reference images inside each batch request",
    )
    parser.add_argument("--file-id-manifest", default="", help="CSV mapping image_name or path to file_id")
    parser.add_argument("--image-url-prefix", default="", help="URL prefix for image-url mode")
    parser.add_argument("--max-output-tokens", type=int, default=250, help="Responses max_output_tokens")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
