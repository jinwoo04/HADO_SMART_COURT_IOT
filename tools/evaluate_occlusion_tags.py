"""Evaluate a YOLO model by filename occlusion tags.

The Track B Roboflow export encodes occlusion difficulty in image filenames
(`clean`, `partial`, `heavy`). This tool builds lightweight symlink subsets for
each tag and evaluates a trained YOLO checkpoint on each subset.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

from track_b_runtime import resolve_device


DEFAULT_TAGS = ("clean", "partial", "heavy")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_tag(path: Path, tags: tuple[str, ...]) -> str | None:
    name = path.name.lower()
    for tag in tags:
        if f"_{tag}_" in name or f"_{tag}." in name:
            return tag
    return None


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", ["player", "effect"])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def _count_labels(label_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = parts[0]
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def build_tag_subsets(
    dataset: Path,
    out_dir: Path,
    split: str,
    tags: tuple[str, ...],
    names: list[str],
) -> dict[str, dict]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    summary: dict[str, dict] = {}
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"

    for tag in tags:
        tag_root = out_dir / tag
        tag_img = tag_root / "test" / "images"
        tag_label = tag_root / "test" / "labels"
        tag_img.mkdir(parents=True)
        tag_label.mkdir(parents=True)
        summary[tag] = {
            "images": 0,
            "labels": 0,
            "classes": {str(idx): 0 for idx, _ in enumerate(names)},
            "data_yaml": str(tag_root / "data.yaml"),
        }

        data = {
            "path": str(tag_root.resolve()),
            "train": "test/images",
            "val": "test/images",
            "test": "test/images",
            "nc": len(names),
            "names": {idx: name for idx, name in enumerate(names)},
        }
        (tag_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    for image in sorted(image_dir.iterdir()):
        if not image.is_file() and not image.is_symlink():
            continue
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        tag = _image_tag(image, tags)
        if tag is None:
            continue

        tag_root = out_dir / tag
        label = label_dir / f"{image.stem}.txt"
        _safe_symlink(image.resolve(), tag_root / "test" / "images" / image.name)
        if label.exists():
            _safe_symlink(label.resolve(), tag_root / "test" / "labels" / label.name)

        summary[tag]["images"] += 1
        label_counts = _count_labels(label)
        for cls, count in label_counts.items():
            summary[tag]["classes"][cls] = summary[tag]["classes"].get(cls, 0) + count
            summary[tag]["labels"] += count

    (out_dir / "tag_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_tags(model_path: Path, subset_summary: dict[str, dict], project: Path, imgsz: int, device: str) -> list[dict]:
    from ultralytics import YOLO  # type: ignore

    device = resolve_device(device)
    model = YOLO(str(model_path))
    rows: list[dict] = []
    for tag, item in subset_summary.items():
        if item["images"] == 0:
            rows.append({
                "tag": tag,
                "images": 0,
                "instances": 0,
                "precision": "",
                "recall": "",
                "map50": "",
                "map50_95": "",
                "save_dir": "",
            })
            continue
        results = model.val(
            data=item["data_yaml"],
            split="test",
            imgsz=imgsz,
            device=device,
            project=str(project),
            name=f"tag_{tag}",
            plots=True,
            verbose=False,
        )
        rows.append({
            "tag": tag,
            "images": item["images"],
            "instances": item["labels"],
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "save_dir": str(results.save_dir),
        })
    return rows


def write_metrics(rows: list[dict], out_dir: Path) -> None:
    (out_dir / "tag_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "tag_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO checkpoint by occlusion filename tags")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root")
    parser.add_argument("--data-yaml", default=None, help="Dataset data.yaml; defaults to <dataset>/data.yaml")
    parser.add_argument("--model", required=True, help="YOLO checkpoint path")
    parser.add_argument("--out", required=True, help="Output directory for subsets and metrics")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Split to evaluate")
    parser.add_argument("--tags", default="clean,partial,heavy", help="Comma-separated filename tags")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:0")
    parser.add_argument("--build-only", action="store_true", help="Only build tag subsets, do not run model evaluation")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    data_yaml = Path(args.data_yaml) if args.data_yaml else dataset / "data.yaml"
    out_dir = Path(args.out)
    tags = tuple(tag.strip().lower() for tag in args.tags.split(",") if tag.strip())
    names = _load_names(data_yaml)

    summary = build_tag_subsets(dataset, out_dir / "subsets", args.split, tags, names)
    if args.build_only:
        print(json.dumps(summary, indent=2))
        return 0

    rows = evaluate_tags(Path(args.model), summary, out_dir / "eval", args.imgsz, args.device)
    write_metrics(rows, out_dir)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
