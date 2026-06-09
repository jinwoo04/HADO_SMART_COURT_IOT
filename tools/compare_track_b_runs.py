"""Compare Track B retrain runs and recommend the current best checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from track_b_runtime import PROJECT_ROOT


DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs" / "track_b_retrain_runs"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: object) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _best_epoch_row(results_csv: Path) -> dict[str, str] | None:
    if not results_csv.exists():
        return None
    rows = _read_csv(results_csv)
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _to_float(row.get("metrics/mAP50-95(B)")),
            _to_float(row.get("metrics/mAP50(B)")),
            _to_float(row.get("metrics/recall(B)")),
            _to_float(row.get("metrics/precision(B)")),
        ),
    )


def _extract_row(run_dir: Path) -> dict[str, object] | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None

    summary = _read_json(summary_path)
    train_save_dir = Path(summary.get("train", {}).get("train_save_dir", "")) if summary.get("train") else None
    results_csv = train_save_dir / "results.csv" if train_save_dir else None
    best_epoch = _best_epoch_row(results_csv) if results_csv else None

    conversion = summary.get("conversion", {})
    splits = conversion.get("splits", {})
    total_images = sum(int(split.get("images", 0)) for split in splits.values())
    total_labels = sum(int(split.get("labels", 0)) for split in splits.values())
    hard = summary.get("hard_mining", {}) or {}

    model_used = Path(summary.get("model_used", ""))
    if not model_used.exists():
        model_used = Path(summary.get("train", {}).get("best_checkpoint", "")) if summary.get("train") else model_used

    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "export_source": summary.get("export_source", ""),
        "model_used": str(model_used),
        "trained": bool(summary.get("trained", False)),
        "total_images": total_images,
        "total_labels": total_labels,
        "train_images": int(splits.get("train", {}).get("images", 0)),
        "valid_images": int(splits.get("valid", {}).get("images", 0)),
        "test_images": int(splits.get("test", {}).get("images", 0)),
        "hard_selected": int(hard.get("selected_for_review", 0) or 0),
        "peak_epoch": int(float(best_epoch.get("epoch", 0))) if best_epoch else 0,
        "precision": _to_float(best_epoch.get("metrics/precision(B)")) if best_epoch else 0.0,
        "recall": _to_float(best_epoch.get("metrics/recall(B)")) if best_epoch else 0.0,
        "map50": _to_float(best_epoch.get("metrics/mAP50(B)")) if best_epoch else 0.0,
        "map50_95": _to_float(best_epoch.get("metrics/mAP50-95(B)")) if best_epoch else 0.0,
    }
    row["score"] = round(
        (row["map50_95"] * 100.0)
        + (row["map50"] * 20.0)
        + (row["recall"] * 5.0)
        + min(row["total_images"], 200) * 0.02,
        4,
    )
    return row


def _sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            -float(row["map50_95"]),
            -float(row["map50"]),
            -float(row["recall"]),
            str(row["run_name"]),
        ),
    )


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "run_name",
        "score",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "peak_epoch",
        "total_images",
        "total_labels",
        "train_images",
        "valid_images",
        "test_images",
        "hard_selected",
        "model_used",
        "export_source",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown(rows: list[dict[str, object]], out_path: Path, best_txt: Path) -> None:
    best = rows[0] if rows else None
    lines = [
        "# Track B Run Leaderboard",
        "",
        "## Current Best",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- run: `{best['run_name']}`",
                f"- checkpoint: `{best['model_used']}`",
                f"- peak epoch: `{best['peak_epoch']}`",
                f"- precision: `{float(best['precision']):.4f}`",
                f"- recall: `{float(best['recall']):.4f}`",
                f"- mAP50: `{float(best['map50']):.4f}`",
                f"- mAP50-95: `{float(best['map50_95']):.4f}`",
                f"- scoreboard score: `{float(best['score']):.4f}`",
                f"- best checkpoint pointer: `{best_txt}`",
            ]
        )
    else:
        lines.append("- no valid runs found")

    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Run | Score | Precision | Recall | mAP50 | mAP50-95 | Images | Hard Selected |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {run} | {score:.4f} | {precision:.4f} | {recall:.4f} | {map50:.4f} | {map50_95:.4f} | {images} | {hard} |".format(
                run=row["run_name"],
                score=float(row["score"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                map50=float(row["map50"]),
                map50_95=float(row["map50_95"]),
                images=int(row["total_images"]),
                hard=int(row["hard_selected"]),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(runs_root)

    rows = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        row = _extract_row(run_dir)
        if row:
            rows.append(row)

    rows = _sort_rows(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "track_b_run_leaderboard.csv"
    out_md = out_dir / "track_b_run_leaderboard.md"
    best_txt = out_dir / "track_b_current_best_checkpoint.txt"
    best_json = out_dir / "track_b_current_best_checkpoint.json"

    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md, best_txt)

    best_payload = {}
    if rows:
        best = rows[0]
        best_txt.write_text(str(best["model_used"]) + "\n", encoding="utf-8")
        best_payload = {
            "run_name": best["run_name"],
            "checkpoint": best["model_used"],
            "precision": best["precision"],
            "recall": best["recall"],
            "map50": best["map50"],
            "map50_95": best["map50_95"],
            "score": best["score"],
        }
    else:
        best_txt.write_text("", encoding="utf-8")
    best_json.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"runs: {len(rows)}")
    print(f"leaderboard_csv: {out_csv}")
    print(f"leaderboard_md: {out_md}")
    print(f"best_checkpoint_txt: {best_txt}")
    print(f"best_checkpoint_json: {best_json}")
    if rows:
        print(f"recommended_run: {rows[0]['run_name']}")
        print(f"recommended_checkpoint: {rows[0]['model_used']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Track B retrain runs and recommend the best checkpoint.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Root directory containing run subdirectories.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory where the leaderboard and best-checkpoint pointer will be written.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
