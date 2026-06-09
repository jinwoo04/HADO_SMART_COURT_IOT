"""Convert YOLO polygon/segment labels to YOLO bbox labels.

Roboflow exports may contain segmentation polygons even when the downstream
pipeline uses object detection. This script keeps images as-is and converts each
polygon line into its tight normalized bbox.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


SPLITS = ("train", "valid", "test")


def _line_to_box(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]

    if len(vals) == 4:
        cx, cy, w, h = vals
        return cls, cx, cy, w, h
    if len(vals) % 2 != 0:
        return None

    xs = vals[0::2]
    ys = vals[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cls, cx, cy, w, h


def _parse_class_map(value: str | None) -> dict[int, int | None]:
    if not value:
        return {}
    mapping: dict[int, int | None] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        src_cls = int(left.strip())
        right = right.strip().lower()
        mapping[src_cls] = None if right in {"drop", "none", "-"} else int(right)
    return mapping


def convert_dataset(
    src: Path,
    dst: Path,
    names: list[str],
    copy_images: bool,
    class_map: dict[int, int | None] | None = None,
) -> dict:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    summary = {
        "source": str(src),
        "destination": str(dst),
        "names": names,
        "splits": {},
    }
    total_classes: Counter[int] = Counter()

    for split in SPLITS:
        src_img_dir = src / split / "images"
        src_label_dir = src / split / "labels"
        dst_img_dir = dst / split / "images"
        dst_label_dir = dst / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        split_classes: Counter[int] = Counter()
        image_count = 0
        label_count = 0
        skipped_lines = 0

        for img_path in sorted(src_img_dir.glob("*")):
            if not img_path.is_file():
                continue
            image_count += 1
            target_img = dst_img_dir / img_path.name
            if copy_images:
                shutil.copy2(img_path, target_img)
            else:
                try:
                    target_img.symlink_to(img_path)
                except FileExistsError:
                    pass

            src_label = src_label_dir / f"{img_path.stem}.txt"
            out_lines: list[str] = []
            if src_label.exists():
                for line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
                    box = _line_to_box(line)
                    if box is None:
                        skipped_lines += 1
                        continue
                    cls, cx, cy, w, h = box
                    if class_map and cls in class_map:
                        mapped = class_map[cls]
                        if mapped is None:
                            continue
                        cls = mapped
                    out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    split_classes[cls] += 1
                    total_classes[cls] += 1
                    label_count += 1
            (dst_label_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""),
                encoding="utf-8",
            )

        summary["splits"][split] = {
            "images": image_count,
            "labels": label_count,
            "classes": {str(k): v for k, v in sorted(split_classes.items())},
            "skipped_lines": skipped_lines,
        }

    data = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    summary["classes_total"] = {str(k): v for k, v in sorted(total_classes.items())}
    (dst / "conversion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO segment dataset to bbox dataset")
    parser.add_argument("--src", required=True, help="Source YOLO segment dataset root")
    parser.add_argument("--dst", required=True, help="Destination YOLO detect dataset root")
    parser.add_argument("--names", default="player,effect", help="Comma-separated class names")
    parser.add_argument(
        "--class-map",
        default=None,
        help="Optional class remap, e.g. '0:0,2:0,1:drop'. Use 'drop' to discard a class.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    names = [name.strip() for name in args.names.split(",") if name.strip()]
    summary = convert_dataset(Path(args.src), Path(args.dst), names, args.copy_images, _parse_class_map(args.class_map))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
