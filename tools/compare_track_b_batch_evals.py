"""Compare Track B batch evaluation directories.

Use this after retraining a player-only model and re-running
`prepare_track_b_batch_review.py` on the same video batch. The script reads
each evaluation directory's `batch_manifest.csv`, aggregates hard-frame
severity, and writes:

- a run-level comparison table
- an optional per-match delta table against a baseline eval

The main intended comparison is `batch02_v4_eval` vs `batch02_v5_eval`.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_int(value: object) -> int:
    if value in ("", None):
        return 0
    return int(float(str(value)))


def _as_float(value: object) -> float:
    if value in ("", None):
        return 0.0
    return float(str(value))


def _reason_counts(hard_frame_dir: str) -> Counter[str]:
    if not hard_frame_dir:
        return Counter()
    manifest = Path(hard_frame_dir) / "hard_frame_manifest.csv"
    if not manifest.exists():
        return Counter()
    return Counter(row.get("reason", "") for row in _read_csv(manifest) if row.get("reason"))


def _match_name(row: dict[str, str]) -> str:
    return Path(row.get("video_path", "")).stem


def _severity(reason_counts: Counter[str], zero_frames: int) -> int:
    overdetect = sum(count for reason, count in reason_counts.items() if reason.startswith("high_count_"))
    underdetect = sum(count for reason, count in reason_counts.items() if reason.startswith("low_count_"))
    lowconf = sum(count for reason, count in reason_counts.items() if reason.startswith("low_conf_"))
    return (overdetect * 3) + (lowconf * 2) + underdetect + zero_frames


def _match_row(eval_name: str, row: dict[str, str]) -> dict[str, object]:
    reasons = _reason_counts(row.get("hard_frame_dir", ""))
    zero_frames = _as_int(row.get("zero_frames"))
    overdetect = sum(count for reason, count in reasons.items() if reason.startswith("high_count_"))
    underdetect = sum(count for reason, count in reasons.items() if reason.startswith("low_count_"))
    lowconf = sum(count for reason, count in reasons.items() if reason.startswith("low_conf_"))
    hard = _as_int(row.get("hard_frame_count"))
    processed = _as_int(row.get("processed_frames"))
    max_count = _as_int(row.get("max_count"))
    avg_count = _as_float(row.get("avg_count"))
    avg_conf = _as_float(row.get("avg_nonzero_conf"))
    return {
        "eval": eval_name,
        "match_name": _match_name(row),
        "video_path": row.get("video_path", ""),
        "processed_frames": processed,
        "hard_frame_count": hard,
        "hard_rate": hard / processed if processed else 0.0,
        "zero_frames": zero_frames,
        "max_count": max_count,
        "avg_count": avg_count,
        "avg_nonzero_conf": avg_conf,
        "overdetect_frames": overdetect,
        "underdetect_frames": underdetect,
        "lowconf_frames": lowconf,
        "severity_score": _severity(reasons, zero_frames),
        "top_reason": _top_reason(reasons),
    }


def _top_reason(reasons: Counter[str]) -> str:
    if not reasons:
        return "none"
    reason, count = reasons.most_common(1)[0]
    return f"{reason} ({count})"


def _collect_eval(eval_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = eval_dir / "batch_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    eval_name = eval_dir.name
    rows = [
        _match_row(eval_name, row)
        for row in _read_csv(manifest)
        if row.get("status", "ok") in ("", "ok")
    ]
    reason_totals: Counter[str] = Counter()
    for row in _read_csv(manifest):
        if row.get("status", "ok") not in ("", "ok"):
            continue
        reason_totals.update(_reason_counts(row.get("hard_frame_dir", "")))

    processed = sum(int(row["processed_frames"]) for row in rows)
    hard = sum(int(row["hard_frame_count"]) for row in rows)
    severity = sum(int(row["severity_score"]) for row in rows)
    overdetect = sum(int(row["overdetect_frames"]) for row in rows)
    underdetect = sum(int(row["underdetect_frames"]) for row in rows)
    lowconf = sum(int(row["lowconf_frames"]) for row in rows)
    avg_count = (
        sum(float(row["avg_count"]) * int(row["processed_frames"]) for row in rows) / processed
        if processed
        else 0.0
    )
    avg_conf_den = sum(int(row["processed_frames"]) for row in rows if float(row["avg_nonzero_conf"]) > 0.0)
    avg_conf = (
        sum(float(row["avg_nonzero_conf"]) * int(row["processed_frames"]) for row in rows) / avg_conf_den
        if avg_conf_den
        else 0.0
    )
    summary = {
        "eval": eval_name,
        "eval_dir": str(eval_dir),
        "videos": len(rows),
        "processed_frames": processed,
        "hard_frame_count": hard,
        "hard_rate": hard / processed if processed else 0.0,
        "severity_score": severity,
        "overdetect_frames": overdetect,
        "underdetect_frames": underdetect,
        "lowconf_frames": lowconf,
        "avg_count": avg_count,
        "avg_nonzero_conf": avg_conf,
        "top_reason": _top_reason(reason_totals),
    }
    return summary, rows


def _delta(current: float, baseline: float) -> float:
    return current - baseline


def _pct_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline


def _write_run_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "eval",
        "videos",
        "processed_frames",
        "hard_frame_count",
        "hard_rate",
        "severity_score",
        "overdetect_frames",
        "underdetect_frames",
        "lowconf_frames",
        "avg_count",
        "avg_nonzero_conf",
        "top_reason",
        "eval_dir",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for key in ("hard_rate", "avg_count", "avg_nonzero_conf"):
                payload[key] = f"{float(payload[key]):.4f}"
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def _write_match_delta_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "match_name",
        "baseline_eval",
        "current_eval",
        "severity_delta",
        "severity_pct_delta",
        "hard_delta",
        "overdetect_delta",
        "underdetect_delta",
        "lowconf_delta",
        "avg_count_delta",
        "hard_rate_delta",
        "baseline_top_reason",
        "current_top_reason",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for key in ("severity_pct_delta", "avg_count_delta", "hard_rate_delta"):
                payload[key] = f"{float(payload[key]):.4f}"
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def _match_deltas(
    baseline_eval: str,
    current_eval: str,
    baseline_rows: list[dict[str, object]],
    current_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    baseline_by_match = {str(row["match_name"]): row for row in baseline_rows}
    deltas: list[dict[str, object]] = []
    for current in current_rows:
        match = str(current["match_name"])
        baseline = baseline_by_match.get(match)
        if not baseline:
            continue
        deltas.append(
            {
                "match_name": match,
                "baseline_eval": baseline_eval,
                "current_eval": current_eval,
                "severity_delta": int(current["severity_score"]) - int(baseline["severity_score"]),
                "severity_pct_delta": _pct_delta(float(current["severity_score"]), float(baseline["severity_score"])),
                "hard_delta": int(current["hard_frame_count"]) - int(baseline["hard_frame_count"]),
                "overdetect_delta": int(current["overdetect_frames"]) - int(baseline["overdetect_frames"]),
                "underdetect_delta": int(current["underdetect_frames"]) - int(baseline["underdetect_frames"]),
                "lowconf_delta": int(current["lowconf_frames"]) - int(baseline["lowconf_frames"]),
                "avg_count_delta": _delta(float(current["avg_count"]), float(baseline["avg_count"])),
                "hard_rate_delta": _delta(float(current["hard_rate"]), float(baseline["hard_rate"])),
                "baseline_top_reason": baseline["top_reason"],
                "current_top_reason": current["top_reason"],
            }
        )
    deltas.sort(key=lambda row: (int(row["severity_delta"]), int(row["hard_delta"]), str(row["match_name"])))
    return deltas


def _improvement_note(delta: int) -> str:
    if delta < 0:
        return "improved"
    if delta > 0:
        return "worse"
    return "same"


def _write_markdown(
    summaries: list[dict[str, object]],
    match_deltas: list[dict[str, object]],
    out_path: Path,
    run_csv: Path,
    match_csv: Path | None,
) -> None:
    lines = [
        "# Track B Batch Eval Comparison",
        "",
        f"- Run CSV: `{run_csv}`",
    ]
    if match_csv:
        lines.append(f"- Match delta CSV: `{match_csv}`")

    lines.extend(
        [
            "",
            "## Eval Summary",
            "",
            "| Eval | Videos | Frames | Hard | Hard % | Severity | Over | Under | Low Conf | Avg Count | Top Reason |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {eval} | {videos} | {frames} | {hard} | {hard_rate:.1%} | {severity} | {over} | {under} | {lowconf} | {avg_count:.2f} | {reason} |".format(
                eval=row["eval"],
                videos=int(row["videos"]),
                frames=int(row["processed_frames"]),
                hard=int(row["hard_frame_count"]),
                hard_rate=float(row["hard_rate"]),
                severity=int(row["severity_score"]),
                over=int(row["overdetect_frames"]),
                under=int(row["underdetect_frames"]),
                lowconf=int(row["lowconf_frames"]),
                avg_count=float(row["avg_count"]),
                reason=row["top_reason"],
            )
        )

    if match_deltas:
        lines.extend(
            [
                "",
                "## Per-Match Delta",
                "",
                "Negative deltas are good: fewer hard frames or lower severity than the baseline.",
                "",
                "| Match | Severity Δ | Hard Δ | Over Δ | Under Δ | LowConf Δ | Status |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in match_deltas:
            severity_delta = int(row["severity_delta"])
            lines.append(
                "| {match} | {severity} | {hard} | {over} | {under} | {lowconf} | {status} |".format(
                    match=row["match_name"],
                    severity=severity_delta,
                    hard=int(row["hard_delta"]),
                    over=int(row["overdetect_delta"]),
                    under=int(row["underdetect_delta"]),
                    lowconf=int(row["lowconf_delta"]),
                    status=_improvement_note(severity_delta),
                )
            )

    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "- Use the same raw video batch and sampling settings when comparing model versions.",
            "- For Track B v5, compare `batch02_v4_eval` against `batch02_v5_eval`.",
            "- The main success signal is lower `severity_score`, especially lower `overdetect_frames`.",
            "- If mAP improves but over-detection severity rises, the retrain did not solve the Track B problem.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_eval_dirs(args: argparse.Namespace) -> list[Path]:
    eval_dirs = [Path(path) for path in args.eval_dir]
    if args.root:
        root = Path(args.root)
        eval_dirs.extend(sorted(path for path in root.iterdir() if (path / "batch_manifest.csv").exists()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in eval_dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def run(args: argparse.Namespace) -> int:
    eval_dirs = _resolve_eval_dirs(args)
    if not eval_dirs:
        raise ValueError("Provide at least one --eval-dir or --root containing batch_manifest.csv files.")

    summaries: list[dict[str, object]] = []
    match_rows_by_eval: dict[str, list[dict[str, object]]] = {}
    for eval_dir in eval_dirs:
        summary, match_rows = _collect_eval(eval_dir)
        summaries.append(summary)
        match_rows_by_eval[str(summary["eval"])] = match_rows

    summaries.sort(key=lambda row: str(row["eval"]))
    baseline_eval = args.baseline or str(summaries[0]["eval"])
    current_eval = args.current or (str(summaries[-1]["eval"]) if len(summaries) > 1 else "")

    match_deltas: list[dict[str, object]] = []
    if current_eval and baseline_eval != current_eval:
        baseline_rows = match_rows_by_eval.get(baseline_eval, [])
        current_rows = match_rows_by_eval.get(current_eval, [])
        if baseline_rows and current_rows:
            match_deltas = _match_deltas(baseline_eval, current_eval, baseline_rows, current_rows)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    match_csv = Path(args.out_match_csv) if args.out_match_csv else None
    _write_run_csv(summaries, out_csv)
    if match_csv:
        _write_match_delta_csv(match_deltas, match_csv)
    _write_markdown(summaries, match_deltas, out_md, out_csv, match_csv)

    print(f"evals: {len(summaries)}")
    print(f"run_csv: {out_csv}")
    if match_csv:
        print(f"match_delta_csv: {match_csv}")
    print(f"summary_md: {out_md}")
    if match_deltas:
        print(f"baseline: {baseline_eval}")
        print(f"current: {current_eval}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Track B batch evaluation directories.")
    parser.add_argument("--eval-dir", action="append", default=[], help="Evaluation directory with batch_manifest.csv.")
    parser.add_argument("--root", default="", help="Optional root containing multiple eval directories.")
    parser.add_argument("--baseline", default="", help="Baseline eval name. Defaults to first sorted eval.")
    parser.add_argument("--current", default="", help="Current eval name. Defaults to last sorted eval.")
    parser.add_argument(
        "--out-csv",
        default="outputs/track_b_eval_comparison/batch_eval_comparison.csv",
        help="Run-level comparison CSV path.",
    )
    parser.add_argument(
        "--out-match-csv",
        default="outputs/track_b_eval_comparison/batch_eval_match_deltas.csv",
        help="Per-match delta CSV path.",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/track_b_eval_comparison/batch_eval_comparison.md",
        help="Markdown summary path.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
