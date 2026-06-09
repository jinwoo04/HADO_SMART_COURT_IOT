"""Track B export ingestion and retrain orchestration.

This script turns a reviewed Roboflow export into a player-only YOLO detect
dataset, then optionally trains, evaluates by occlusion tag, and mines hard
player samples for the next relabeling loop.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from convert_yolo_segments_to_boxes import convert_dataset
from evaluate_occlusion_tags import build_tag_subsets, evaluate_tags, write_metrics
from mine_hard_player_samples import mine
from track_b_runtime import PROJECT_ROOT, resolve_device, resolve_model_path


DEFAULT_CLASS_MAP = "0:0,2:0,1:drop"
DEFAULT_PLAYER_NAMES = ["player"]
COMPARE_RUNS_SCRIPT = PROJECT_ROOT / "tools" / "compare_track_b_runs.py"


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


def _names_from_arg(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--names must contain at least one class name")
    return names


def _load_yaml_names(data_yaml: Path) -> list[str]:
    import yaml

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", DEFAULT_PLAYER_NAMES)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def _find_train_save_dir(train_result, fallback_project: Path, fallback_name: str) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir:
        return Path(save_dir)

    trainer = getattr(train_result, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir:
        return Path(trainer_save_dir)

    candidates = sorted(fallback_project.glob(f"{fallback_name}*"))
    if candidates:
        return candidates[-1]
    return fallback_project / fallback_name


def _train_model(args: argparse.Namespace, data_yaml: Path, run_root: Path) -> tuple[Path, dict]:
    from ultralytics import YOLO  # type: ignore

    train_project = run_root / "train"
    train_project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.init_model)
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(train_project),
        name=args.train_name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        pretrained=True,
        verbose=False,
    )
    save_dir = _find_train_save_dir(train_result, train_project, args.train_name)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best checkpoint not found: {best_path}")
    summary = {
        "train_save_dir": str(save_dir),
        "best_checkpoint": str(best_path),
    }
    return best_path, summary


def _run_eval(model_path: Path, dataset_dir: Path, data_yaml: Path, out_dir: Path, split: str, imgsz: int, device: str) -> list[dict]:
    names = _load_yaml_names(data_yaml)
    subset_summary = build_tag_subsets(dataset_dir, out_dir / "subsets", split, ("clean", "partial", "heavy"), names)
    rows = evaluate_tags(model_path, subset_summary, out_dir / "eval", imgsz, device)
    write_metrics(rows, out_dir)
    return rows


def _run_hard_mining(args: argparse.Namespace, dataset_dir: Path, model_path: Path, out_dir: Path) -> dict:
    mine_args = argparse.Namespace(
        dataset=str(dataset_dir),
        model=str(model_path),
        out=str(out_dir),
        splits=args.mine_splits,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.mine_conf,
        iou=args.mine_iou,
        target_conf=args.mine_target_conf,
        player_class=0,
        top_k=args.mine_top_k,
        sheet_count=args.mine_sheet_count,
        progress=args.mine_progress,
        clean=True,
    )
    selected = mine(mine_args)
    summary_path = out_dir / "hard_player_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["selected_rows_preview"] = selected[:5]
    return summary


def _write_markdown_summary(summary: dict, out_path: Path) -> None:
    lines = [
        "# Track B Export Retrain Summary",
        "",
        f"- export source: `{summary['export_source']}`",
        f"- converted dataset: `{summary['converted_dataset']}`",
        f"- class map: `{summary['class_map']}`",
        f"- device: `{summary['device']}`",
        f"- trained: `{summary['trained']}`",
        f"- model used: `{summary['model_used']}`",
    ]
    train = summary.get("train")
    if train:
        lines.extend(
            [
                f"- train save dir: `{train['train_save_dir']}`",
                f"- best checkpoint: `{train['best_checkpoint']}`",
            ]
        )
    eval_rows = summary.get("eval_rows", [])
    if eval_rows:
        lines.extend(["", "## Tag Metrics", "", "| tag | images | precision | recall | map50 | map50-95 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in eval_rows:
            lines.append(
                f"| {row['tag']} | {row['images']} | {row['precision']} | {row['recall']} | {row['map50']} | {row['map50_95']} |"
            )
    hard = summary.get("hard_mining")
    if hard:
        lines.extend(
            [
                "",
                "## Hard Mining",
                "",
                f"- selected for review: `{hard.get('selected_for_review', 0)}`",
                f"- top csv: `{hard.get('top_csv', '')}`",
                f"- review subset: `{hard.get('review_subset', '')}`",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_leaderboard(out_root: Path) -> None:
    if not COMPARE_RUNS_SCRIPT.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(COMPARE_RUNS_SCRIPT),
            "--runs-root",
            str(out_root),
            "--out-dir",
            str(out_root),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert, retrain, evaluate, and mine Track B Roboflow exports")
    parser.add_argument("--export-src", required=True, help="Roboflow YOLO export root")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs" / "track_b_retrain_runs"), help="Output root for converted data and reports")
    parser.add_argument("--run-name", default="latest_export", help="Subdirectory name under out-root")
    parser.add_argument("--names", default="player", help="Comma-separated destination class names")
    parser.add_argument("--class-map", default=DEFAULT_CLASS_MAP, help="Class remap, e.g. 0:0,2:0,1:drop")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and reuse an existing model")
    parser.add_argument("--model", default=None, help="Model checkpoint to evaluate/mine when --skip-train is used")
    parser.add_argument("--init-model", default="yolov8n.pt", help="Ultralytics init model for retraining")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-name", default="yolov8n_player_only")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:0")
    parser.add_argument("--eval-split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--skip-eval", action="store_true", help="Skip occlusion-tag evaluation")
    parser.add_argument("--skip-mine", action="store_true", help="Skip hard-sample mining")
    parser.add_argument("--mine-splits", default="valid,test")
    parser.add_argument("--mine-conf", type=float, default=0.25)
    parser.add_argument("--mine-iou", type=float, default=0.3)
    parser.add_argument("--mine-target-conf", type=float, default=0.5)
    parser.add_argument("--mine-top-k", type=int, default=80)
    parser.add_argument("--mine-sheet-count", type=int, default=24)
    parser.add_argument("--mine-progress", type=int, default=25)
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    export_src = Path(args.export_src).expanduser().resolve()
    run_root = Path(args.out_root).expanduser().resolve() / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = run_root / "dataset_player_only"
    names = _names_from_arg(args.names)
    conversion_summary = convert_dataset(
        export_src,
        dataset_dir,
        names=names,
        copy_images=args.copy_images,
        class_map=_parse_class_map(args.class_map),
    )

    if args.skip_train:
        model_path = resolve_model_path(args.model)
        train_summary = None
    else:
        model_path, train_summary = _train_model(args, dataset_dir / "data.yaml", run_root)

    eval_rows: list[dict] = []
    if not args.skip_eval:
        eval_rows = _run_eval(model_path, dataset_dir, dataset_dir / "data.yaml", run_root / "tag_eval", args.eval_split, args.imgsz, args.device)

    hard_summary: dict | None = None
    if not args.skip_mine:
        hard_summary = _run_hard_mining(args, dataset_dir, model_path, run_root / "hard_mining")

    summary = {
        "export_source": str(export_src),
        "converted_dataset": str(dataset_dir),
        "class_map": args.class_map,
        "device": args.device,
        "trained": not args.skip_train,
        "model_used": str(model_path),
        "conversion": conversion_summary,
        "train": train_summary,
        "eval_rows": eval_rows,
        "hard_mining": hard_summary,
    }
    (run_root / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_summary(summary, run_root / "run_summary.md")
    _refresh_leaderboard(Path(args.out_root).expanduser().resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
