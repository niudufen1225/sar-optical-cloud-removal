#!/usr/bin/env python3
"""Batch-evaluate ALLClear training runs.

For each run directory, this script:

1. analyzes train_log.csv and saved epoch visualizations;
2. optionally evaluates checkpointed models on val/test with region-aware metrics;
3. optionally exports balanced low/medium/high/heavy visual grids for val/test;
4. writes cross-run CSV summaries and one Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.eval_metrics import paired_bootstrap


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str], *, dry_run: bool = False) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def discover_runs(root: Path, explicit: list[Path], pattern: str) -> list[Path]:
    runs = [p.expanduser().resolve() for p in explicit]
    if not runs:
        runs = sorted(p.parent.resolve() for p in root.expanduser().resolve().glob(f"{pattern}/train_log.csv"))
    return [p for p in runs if (p / "train_log.csv").exists()]


def find_checkpoint(run_dir: Path, preferred: str) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    preferred_path = ckpt_dir / preferred
    if preferred_path.exists():
        return preferred_path
    best = sorted(ckpt_dir.glob("best_epoch_*.pt"))
    if best:
        return best[-1]
    last = ckpt_dir / "last.pt"
    return last if last.exists() else None


def latest_split_rows(log_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in log_rows:
        split = row.get("split", "")
        if split:
            out[split] = row
    return out


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def summarize_run(run_dir: Path, analysis_dir: Path, checkpoint: Path | None) -> dict[str, Any]:
    cfg = read_json(run_dir / "config.resolved.json")
    latest = read_json(run_dir / "metrics" / "latest.json")
    log_rows = read_csv_rows(run_dir / "train_log.csv")
    latest_rows = latest_split_rows(log_rows)
    val_row = latest_rows.get("val", {})
    train_row = latest_rows.get("train", {})
    anomaly_rows = read_csv_rows(analysis_dir / "log_visual" / "anomalies.csv")
    visual_rows = read_csv_rows(analysis_dir / "log_visual" / "visualization_inventory.csv")
    complete_visuals = sum(1 for row in visual_rows if str(row.get("complete", "")).lower() in {"true", "1"})
    row: dict[str, Any] = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "framework": cfg.get("model", {}).get("framework", ""),
        "config_run_name": cfg.get("run_name", ""),
        "epochs_logged": len({r.get("epoch", "") for r in log_rows if r.get("epoch", "")}),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "has_checkpoint": checkpoint is not None,
        "analysis_dir": str(analysis_dir),
        "visual_epochs": len(visual_rows),
        "visual_epochs_complete": complete_visuals,
        "anomaly_count": len(anomaly_rows),
        "error_anomaly_count": sum(1 for r in anomaly_rows if r.get("severity") == "error"),
        "best_metric": latest.get("best_metric_name", cfg.get("train", {}).get("best_metric", "")),
        "best_metric_value": latest.get("best_metric", ""),
    }
    for prefix, source in [("latest_train", train_row), ("latest_val", val_row)]:
        for key in [
            "total",
            "recon_total",
            "gan_total",
            "final_l1",
            "shadow_removal",
            "shadow_mask",
            "shadow_penumbra",
            "cloud_l1",
            "cloud_kl",
            "cloud_adv",
            "disc_total",
        ]:
            row[f"{prefix}_{key}"] = source.get(key, "")
    best_val = None
    for r in log_rows:
        if r.get("split") != "val":
            continue
        value = float_or_none(r.get(str(row["best_metric"]), ""))
        if value is not None and (best_val is None or value < best_val):
            best_val = value
    row["best_val_from_log"] = best_val if best_val is not None else ""
    return row


def collect_csv_with_run(run_dir: Path, source: Path, out_rows: list[dict[str, Any]]) -> None:
    for row in read_csv_rows(source):
        row = dict(row)
        row["run"] = run_dir.name
        row["source_file"] = str(source)
        out_rows.append(row)


def collect_branch_summary(run_dir: Path, source: Path, split: str, out_rows: list[dict[str, Any]]) -> None:
    data = read_json(source)
    if not data:
        return
    base = {
        "run": run_dir.name,
        "split": split,
        "samples": data.get("samples", ""),
        "source_file": str(source),
    }
    for candidate, regions in data.get("metrics", {}).items():
        for region, metrics in regions.items():
            row = dict(base)
            row.update({"candidate": candidate, "region": region})
            row.update(metrics)
            out_rows.append(row)
    for key, value in data.get("softshadow", {}).items():
        row = dict(base)
        row.update({"candidate": "softshadow", "region": key, "mae": value})
        out_rows.append(row)
    for key, value in data.get("branch_improvement", {}).items():
        row = dict(base)
        row.update({"candidate": "branch_improvement", "region": key, "mae": value})
        out_rows.append(row)


def paired_metric_names(rows1: list[dict[str, str]], rows2: list[dict[str, str]]) -> list[str]:
    """Select quality metrics that are meaningful for S2-S1 paired deltas."""

    common = set(rows1[0] if rows1 else {}).intersection(rows2[0] if rows2 else {})
    selected = []
    for name in sorted(common):
        if not name.startswith("final_"):
            continue
        if name.endswith(("_mae", "_rmse", "_psnr", "_ssim", "_mean_abs_channel_bias")):
            selected.append(name)
        elif "_wavelet_" in name and name.endswith(("_mae", "_abs_delta")):
            selected.append(name)
    return selected


def paired_comparison(
    s1_csv: Path,
    s2_csv: Path,
    *,
    run1: Path,
    run2: Path,
    split: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rows1 = read_csv_rows(s1_csv)
    rows2 = read_csv_rows(s2_csv)
    by_id_1 = {str(row.get("sample_id", "")): row for row in rows1 if str(row.get("sample_id", ""))}
    by_id_2 = {str(row.get("sample_id", "")): row for row in rows2 if str(row.get("sample_id", ""))}
    shared_ids = sorted(set(by_id_1).intersection(by_id_2))
    metrics: dict[str, Any] = {}
    for name in paired_metric_names(rows1, rows2):
        values1 = []
        values2 = []
        for sample_id in shared_ids:
            try:
                value1 = float(by_id_1[sample_id].get(name, "nan"))
                value2 = float(by_id_2[sample_id].get(name, "nan"))
            except (TypeError, ValueError):
                value1, value2 = math.nan, math.nan
            values1.append(value1)
            values2.append(value2)
        higher = name.endswith(("_psnr", "_ssim"))
        metrics[name] = paired_bootstrap(
            values1,
            values2,
            higher_is_better=higher,
            resamples=resamples,
            seed=seed,
        )
    return {
        "status": "ok" if shared_ids else "no_shared_samples",
        "split": split,
        "run_s1": str(run1),
        "run_s2": str(run2),
        "per_sample_join": "exact sample_id intersection; row order is ignored",
        "shared_samples": len(shared_ids),
        "bootstrap": {"resamples": int(resamples), "seed": int(seed), "delta": "metric_s2 - metric_s1", "ci": "percentile 95%"},
        "metrics": metrics,
    }
def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# ALLClear Batch Evaluation",
        "",
        f"- Runs: `{len(rows)}`",
        f"- Root: `{args.root}`",
        f"- Branch eval: `{not args.skip_branch_eval}`",
        f"- Split visuals: `{not args.skip_split_visuals}`",
        "",
        "## Run Summary",
        "",
        "| run | framework | epochs | checkpoint | latest_val_recon_total | anomalies | visuals |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {framework} | {epochs_logged} | {ckpt} | {val} | {anom} | {vis}/{vis_all} |".format(
                run=row.get("run", ""),
                framework=row.get("framework", ""),
                epochs_logged=row.get("epochs_logged", ""),
                ckpt="yes" if row.get("has_checkpoint") else "no",
                val=row.get("latest_val_recon_total", ""),
                anom=row.get("anomaly_count", ""),
                vis=row.get("visual_epochs_complete", ""),
                vis_all=row.get("visual_epochs", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `runs_summary.csv`: one row per run.",
            "- `metric_summary_all.csv`: train/val loss statistics from every run.",
            "- `loss_contributions_all.csv`: weighted contribution ratios for every loss term.",
            "- `generalization_gaps_all.csv`: train/val gap table.",
            "- `visualization_inventory_all.csv`: saved low/medium/high/heavy visualization coverage.",
            "- `visual_candidate_comparison_all.csv`: image-panel proxy metrics extracted from saved visual grids.",
            "- `branch_metrics_all.csv`: checkpoint-based val/test metrics by candidate and region when branch eval is enabled.",
            "- `paired_comparison.json`: paired S2-S1 bootstrap when `--paired-run` is supplied twice.",
            "- `<run>/log_visual/report.md`: detailed per-run log and visualization diagnosis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/allclear"), help="Directory containing run subdirectories.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Specific run directory. Can be repeated.")
    parser.add_argument("--pattern", default="*", help="Glob pattern under --root when --run-dir is not provided.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/allclear_batch_eval"), help="Cross-run report directory.")
    parser.add_argument("--analysis-name", default="analysis_batch", help="Per-run analysis subdirectory name.")
    parser.add_argument("--checkpoint", default="last.pt", help="Preferred checkpoint file under checkpoints/. Falls back to best_epoch_*.pt.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split sample limit for branch metrics.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-visuals", type=int, default=9, help="Per-split branch visual examples to save.")
    parser.add_argument("--samples-per-bucket", type=int, default=5, help="Balanced split visualization samples per cloud bucket.")
    parser.add_argument("--skip-branch-eval", action="store_true")
    parser.add_argument("--skip-split-visuals", action="store_true")
    parser.add_argument("--sar-counterfactual", action="store_true", help="Enable the five-forward SAR counterfactual evaluator.")
    parser.add_argument("--sar-batch-size", type=int, default=4)
    parser.add_argument("--sar-low-pass-kernel", type=int, default=5)
    parser.add_argument("--paired-run", type=Path, action="append", default=[], help="Repeat exactly twice to produce paired S2-S1 bootstrap.")
    parser.add_argument("--paired-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.root, args.run_dir, args.pattern)
    paired_runs = [path.expanduser().resolve() for path in args.paired_run]
    if paired_runs and len(paired_runs) != 2:
        raise ValueError("--paired-run must be supplied exactly twice")
    if paired_runs:
        known = {path.resolve() for path in runs}
        runs.extend(path for path in paired_runs if path not in known and (path / "train_log.csv").exists())
    if not runs:
        raise FileNotFoundError(f"No runs with train_log.csv found under {args.root}")

    run_rows: list[dict[str, Any]] = []
    metric_summary_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    visual_inventory_rows: list[dict[str, Any]] = []
    visual_panel_rows: list[dict[str, Any]] = []
    visual_comparison_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []

    for run_dir in runs:
        analysis_dir = run_dir / args.analysis_name
        log_visual_dir = analysis_dir / "log_visual"
        log_visual_dir.mkdir(parents=True, exist_ok=True)
        code = run_cmd(
            [
                sys.executable,
                "scripts/evaluate_training_run.py",
                "--run-dir",
                str(run_dir),
                "--out-dir",
                str(log_visual_dir),
            ],
            dry_run=args.dry_run,
        )
        checkpoint = find_checkpoint(run_dir, args.checkpoint)
        if code != 0:
            print(f"[warn] log/visual analysis failed for {run_dir}", file=sys.stderr)

        if checkpoint and not args.skip_branch_eval:
            for split in args.splits:
                branch_dir = analysis_dir / f"branch_{split}"
                cmd = [
                    sys.executable,
                    "scripts/evaluate_stage1_branches.py",
                    "--config",
                    str(run_dir / "config.resolved.json"),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    split,
                    "--output-dir",
                    str(branch_dir),
                    "--gpu",
                    str(args.gpu),
                    "--num-workers",
                    str(args.num_workers),
                    "--save-visuals",
                    str(args.save_visuals),
                ]
                if args.limit:
                    cmd.extend(["--limit", str(args.limit)])
                if args.sar_counterfactual:
                    cmd.extend(["--sar-counterfactual", "--sar-batch-size", str(args.sar_batch_size), "--sar-low-pass-kernel", str(args.sar_low_pass_kernel)])
                run_cmd(cmd, dry_run=args.dry_run)
                collect_branch_summary(run_dir, branch_dir / f"{split}_branch_metrics_summary.json", split, branch_rows)

        if checkpoint and not args.skip_split_visuals:
            cmd = [
                sys.executable,
                "scripts/export_run_split_visuals.py",
                "--run-dir",
                str(run_dir),
                "--checkpoint",
                str(checkpoint),
                "--splits",
                *args.splits,
                "--samples-per-bucket",
                str(args.samples_per_bucket),
                "--gpu",
                str(args.gpu),
                "--output-dir",
                str(analysis_dir / "split_visuals"),
            ]
            run_cmd(cmd, dry_run=args.dry_run)

        run_rows.append(summarize_run(run_dir, analysis_dir, checkpoint))
        collect_csv_with_run(run_dir, log_visual_dir / "metric_summary.csv", metric_summary_rows)
        collect_csv_with_run(run_dir, log_visual_dir / "generalization_gaps.csv", gap_rows)
        collect_csv_with_run(run_dir, log_visual_dir / "loss_contributions.csv", contribution_rows)
        collect_csv_with_run(run_dir, log_visual_dir / "anomalies.csv", anomaly_rows)
        collect_csv_with_run(run_dir, log_visual_dir / "visualization_inventory.csv", visual_inventory_rows)
        collect_csv_with_run(run_dir, log_visual_dir / "visual_panel_quality.csv", visual_panel_rows)
        collect_csv_with_run(run_dir, log_visual_dir / "visual_candidate_comparison.csv", visual_comparison_rows)

    write_csv(out_dir / "runs_summary.csv", run_rows)
    write_csv(out_dir / "metric_summary_all.csv", metric_summary_rows)
    write_csv(out_dir / "generalization_gaps_all.csv", gap_rows)
    write_csv(out_dir / "loss_contributions_all.csv", contribution_rows)
    write_csv(out_dir / "anomalies_all.csv", anomaly_rows)
    write_csv(out_dir / "visualization_inventory_all.csv", visual_inventory_rows)
    write_csv(out_dir / "visual_panel_quality_all.csv", visual_panel_rows)
    write_csv(out_dir / "visual_candidate_comparison_all.csv", visual_comparison_rows)
    write_csv(out_dir / "branch_metrics_all.csv", branch_rows)
    if paired_runs and not args.dry_run and not args.skip_branch_eval:
        pair_path_1 = paired_runs[0] / args.analysis_name / f"branch_{args.paired_split}" / f"{args.paired_split}_branch_metrics_per_sample.csv"
        pair_path_2 = paired_runs[1] / args.analysis_name / f"branch_{args.paired_split}" / f"{args.paired_split}_branch_metrics_per_sample.csv"
        if not pair_path_1.exists() or not pair_path_2.exists():
            raise FileNotFoundError(f"Paired evaluator CSV missing: {pair_path_1} or {pair_path_2}")
        comparison = paired_comparison(
            pair_path_1,
            pair_path_2,
            run1=paired_runs[0],
            run2=paired_runs[1],
            split=args.paired_split,
            resamples=max(2000, int(args.bootstrap_resamples)),
            seed=int(args.bootstrap_seed),
        )
        (out_dir / "paired_comparison.json").write_text(json.dumps(json_safe(comparison), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_rows, args)
    print(f"batch_analysis_dir={out_dir}")
    print(f"report={out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
