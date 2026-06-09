"""Mine hard player-detection samples from a YOLO dataset.

This finds images where player ground-truth boxes are missed or weakly matched
by a trained model. The output is intended for Roboflow relabeling/review and
for oversampling in the next Track B dataset version.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from track_b_runtime import resolve_device, resolve_model_path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _tag_for_name(name: str) -> str:
    lowered = name.lower()
    for tag in ("clean", "partial", "heavy"):
        if f"_{tag}_" in lowered or f"_{tag}." in lowered:
            return tag
    return "unknown"


def _load_yolo_boxes(label_path: Path, width: int, height: int, cls_filter: int = 0) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if cls != cls_filter:
            continue
        cx, cy, bw, bh = [float(v) for v in parts[1:]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _draw_boxes(
    image: np.ndarray,
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float, float]],
    title: str,
) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 80, 255), 2)
    for x1, y1, x2, y2, conf in pred_boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 60), 2)
        cv2.putText(canvas, f"{conf:.2f}", (int(x1), max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 60), 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def _make_contact_sheet(previews: list[Path], out_path: Path, thumb_w: int = 320, cols: int = 3) -> None:
    if not previews:
        return
    thumbs: list[np.ndarray] = []
    for path in previews:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        thumb_h = max(1, int(img.shape[0] * scale))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h)))
    if not thumbs:
        return
    thumb_h = max(img.shape[0] for img in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 24, dtype=np.uint8)
    for idx, img in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * thumb_h:r * thumb_h + img.shape[0], c * thumb_w:c * thumb_w + img.shape[1]] = img
    cv2.imwrite(str(out_path), sheet)


def _predict_players(model, image: np.ndarray, args: argparse.Namespace):
    return model(
        image,
        imgsz=args.imgsz,
        conf=args.conf,
        classes=[args.player_class],
        device=args.device,
        verbose=False,
    )[0]


def mine(args: argparse.Namespace) -> list[dict]:
    from ultralytics import YOLO  # type: ignore

    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    review_img_dir = out_dir / "review_subset" / "images"
    review_label_dir = out_dir / "review_subset" / "labels"
    preview_dir.mkdir(parents=True, exist_ok=True)
    review_img_dir.mkdir(parents=True, exist_ok=True)
    review_label_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_subset" / "data.yaml").write_text(
        "\n".join([
            f"path: {(out_dir / 'review_subset').resolve()}",
            "train: images",
            "val: images",
            "test: images",
            "nc: 2",
            "names:",
            "  0: player",
            "  1: effect",
            "",
        ]),
        encoding="utf-8",
    )

    args.device = resolve_device(args.device)
    model = YOLO(str(resolve_model_path(args.model)))
    rows: list[dict] = []
    processed = 0
    for split in args.splits.split(","):
        split = split.strip()
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            gt_boxes = _load_yolo_boxes(label_dir / f"{image_path.stem}.txt", width, height, cls_filter=args.player_class)
            if not gt_boxes:
                continue

            processed += 1
            if args.progress and processed % args.progress == 0:
                print(f"processed {processed} images...", flush=True)

            pred = _predict_players(model, image, args)
            pred_boxes: list[tuple[float, float, float, float, float]] = []
            if pred.boxes is not None and len(pred.boxes) > 0:
                xyxy = pred.boxes.xyxy.cpu().numpy()
                confs = pred.boxes.conf.cpu().numpy()
                for box, conf in zip(xyxy, confs):
                    pred_boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))

            matched_ious: list[float] = []
            matched_confs: list[float] = []
            missed = 0
            for gt in gt_boxes:
                best_iou = 0.0
                best_conf = 0.0
                for pred_box in pred_boxes:
                    iou = _iou(gt, pred_box[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = pred_box[4]
                matched_ious.append(best_iou)
                matched_confs.append(best_conf)
                if best_iou < args.iou:
                    missed += 1

            avg_iou = sum(matched_ious) / len(matched_ious)
            avg_conf = sum(matched_confs) / len(matched_confs)
            miss_rate = missed / len(gt_boxes)
            # Higher score means more useful for relabeling/oversampling.
            hard_score = miss_rate * 2.0 + (1.0 - avg_iou) + max(0.0, args.target_conf - avg_conf)

            rows.append({
                "image": str(image_path),
                "label": str(label_dir / f"{image_path.stem}.txt"),
                "split": split,
                "tag": _tag_for_name(image_path.name),
                "gt_players": len(gt_boxes),
                "pred_players": len(pred_boxes),
                "missed_players": missed,
                "miss_rate": round(miss_rate, 6),
                "avg_best_iou": round(avg_iou, 6),
                "avg_match_conf": round(avg_conf, 6),
                "hard_score": round(hard_score, 6),
            })

    rows.sort(key=lambda item: (float(item["hard_score"]), float(item["miss_rate"])), reverse=True)
    selected = rows[: args.top_k]

    preview_paths: list[Path] = []
    for rank, row in enumerate(selected, start=1):
        src_image = Path(row["image"])
        src_label = Path(row["label"])
        image = cv2.imread(str(src_image))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_boxes = _load_yolo_boxes(src_label, width, height, cls_filter=args.player_class)
        pred = _predict_players(model, image, args)
        pred_boxes = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            xyxy = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                pred_boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))

        dst_image = review_img_dir / src_image.name
        dst_label = review_label_dir / src_label.name
        if dst_image.exists() or dst_image.is_symlink():
            dst_image.unlink()
        dst_image.symlink_to(src_image.resolve())
        if src_label.exists():
            if dst_label.exists() or dst_label.is_symlink():
                dst_label.unlink()
            dst_label.symlink_to(src_label.resolve())

        title = f"{rank:02d} {row['tag']} miss={row['missed_players']}/{row['gt_players']} iou={row['avg_best_iou']}"
        preview = _draw_boxes(image, gt_boxes, pred_boxes, title)
        preview_path = preview_dir / f"{rank:03d}_{src_image.stem}.jpg"
        cv2.imwrite(str(preview_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        preview_paths.append(preview_path)

    with (out_dir / "hard_player_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "model": args.model,
        "dataset": str(dataset),
        "splits": args.splits,
        "iou_threshold": args.iou,
        "prediction_conf": args.conf,
        "total_candidates": len(rows),
        "selected_for_review": len(selected),
        "tag_counts_selected": {},
        "top_csv": str(out_dir / "hard_player_samples.csv"),
        "review_subset": str(out_dir / "review_subset"),
        "preview_dir": str(preview_dir),
    }
    for row in selected:
        summary["tag_counts_selected"][row["tag"]] = summary["tag_counts_selected"].get(row["tag"], 0) + 1
    (out_dir / "hard_player_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _make_contact_sheet(preview_paths[: min(len(preview_paths), args.sheet_count)], out_dir / "hard_player_contact_sheet.jpg")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine hard player samples for Track B")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root")
    parser.add_argument("--model", default=None, help="YOLO checkpoint path; defaults to Track B best checkpoint")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--splits", default="valid,test", help="Comma-separated splits")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--target-conf", type=float, default=0.5)
    parser.add_argument("--player-class", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--sheet-count", type=int, default=24)
    parser.add_argument("--progress", type=int, default=25)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    selected = mine(args)
    print(json.dumps(selected[:10], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
