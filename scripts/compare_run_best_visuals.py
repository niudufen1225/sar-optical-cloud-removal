#!/usr/bin/env python3
"""Compare best checkpoints from multiple ALLClear runs on identical samples.

The normal per-run visualization pipeline selects samples independently, which
makes qualitative ablations hard to judge.  This script first selects one fixed
set of medium/high/heavy samples from a reference config, then runs every best
checkpoint on that same set and writes side-by-side contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.dataset import cloud_fraction
from src.allclear.train import (
    _display_limits,
    _mask_panel,
    _panel_to_image,
    _sar_panel,
    _save_titled_grid,
    _shared_rgb_limits,
    _stretch_rgb,
    apply_model_band_indices,
    build_model,
    cloud_bucket_name,
    keep_top_visual_candidate,
    load_checkpoint,
    make_loader,
    model_band_indices_from_cfg,
    model_reflectance_range_from_cfg,
    move_batch,
    select_visual_candidates,
    slice_tensor_batch,
    visual_dedup_keys,
    visual_structure_score,
)


DEFAULT_BUCKETS = ("medium", "high", "heavy")


def load_run_config(run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / "config.resolved.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    yaml_path = run_dir / "config.resolved.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing resolved config in {run_dir}")


def find_best_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("best_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No best_epoch_*.pt found under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def discover_runs(root: Path, pattern: str) -> list[Path]:
    runs: list[Path] = []
    for path in sorted(root.glob(pattern)):
        if path.is_dir() and (path / "config.resolved.json").exists() and (path / "checkpoints").exists():
            try:
                find_best_checkpoint(path)
            except FileNotFoundError:
                continue
            runs.append(path.resolve())
    return runs


def short_run_label(run_dir: Path) -> str:
    name = run_dir.name
    match = re.search(r"stage1__(.+)$", name)
    label = match.group(1) if match else name
    if len(label) > 28:
        return f"{label[:12]}...{label[-12:]}"
    return label


def raw_rgb_indices_from_cfg(cfg: dict[str, Any]) -> tuple[int, int, int]:
    rgb = tuple(int(v) for v in cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    model_indices = model_band_indices_from_cfg(cfg)
    if model_indices is not None and max(rgb) < len(model_indices):
        mapped = tuple(int(model_indices[idx]) for idx in rgb)
        return mapped  # type: ignore[return-value]
    return rgb  # type: ignore[return-value]


def output_rgb_indices_from_cfg(cfg: dict[str, Any], channels: int) -> tuple[int, int, int]:
    rgb = tuple(int(v) for v in cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    if channels >= 3 and max(rgb) < channels:
        return rgb  # type: ignore[return-value]
    if channels >= 3:
        return (0, 1, 2)
    return (0, 0, 0)


def rgb_panel(x: Tensor, indices: tuple[int, int, int]) -> Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape={tuple(x.shape)}")
    c = x.shape[1]
    safe = tuple(idx if 0 <= idx < c else 0 for idx in indices)
    panel = x[0, list(safe)].detach().float().cpu()
    if c == 1:
        panel = panel[:1].repeat(3, 1, 1)
    return panel


def undo_model_reflectance(x: Tensor, cfg: dict[str, Any]) -> Tensor:
    value_range = model_reflectance_range_from_cfg(cfg)
    if value_range is None:
        return x
    lo, hi = value_range
    return (x.detach().float() * (hi - lo) + lo).clamp_min(0.0)


def stretch_rgb_row(
    panels: list[Tensor],
    *,
    mode: str,
    gamma: float,
    gain: float,
) -> list[Tensor]:
    if mode == "panel":
        out = []
        for panel in panels:
            lo, hi = _display_limits(panel)
            out.append(_stretch_rgb(panel, lo, hi, gamma=gamma, gain=gain))
        return out
    if mode == "target":
        ref = panels[1] if len(panels) > 1 else panels[0]
        lo, hi = _display_limits(ref)
        return [_stretch_rgb(panel, lo, hi, gamma=gamma, gain=gain) for panel in panels]
    if mode == "shared_all":
        lo, hi = _shared_rgb_limits(panels)
        return [_stretch_rgb(panel, lo, hi, gamma=gamma, gain=gain) for panel in panels]
    raise ValueError(f"Unsupported visual RGB stretch mode: {mode}")


def resize_panel(panel: Tensor, tile_size: int) -> Tensor:
    if panel.shape[-2:] == (tile_size, tile_size):
        return panel
    return F.interpolate(
        panel.unsqueeze(0),
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=False,
    )[0]


def first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""


def sample_metadata(batch: dict[str, Any], *, bucket: str, score: float, cloud_frac: float) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "score": score,
        "cloud_fraction": cloud_frac,
        "sample_id": first_text(batch.get("sample_id")),
        "roi_id": first_text(batch.get("roi_id")),
        "clear_date": first_text(batch.get("clear_date")),
        "cloudy_date": first_text(batch.get("cloudy_date")),
        "clear_s2_path": first_text(batch.get("clear_s2_path")),
        "cloudy_s2_path": first_text(batch.get("cloudy_s2_path")),
        "dedup_keys": "|".join(visual_dedup_keys(batch)),
    }


def buckets_complete(
    buckets: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]],
    requested: int,
) -> bool:
    return all(len(items) >= requested for items in buckets.values())


def select_reference_samples(
    *,
    cfg: dict[str, Any],
    split: str,
    buckets: tuple[str, ...],
    samples_per_bucket: int,
    candidate_pool_per_bucket: int,
) -> dict[str, list[tuple[dict[str, Tensor], dict[str, Any]]]]:
    loader = make_loader(cfg, split, distributed=False, rank=0, world_size=1)
    cloud_index = int(cfg.get("data", {}).get("cloud_index", 1))
    raw_rgb_indices = raw_rgb_indices_from_cfg(cfg)
    pool_size = max(samples_per_bucket, candidate_pool_per_bucket)
    candidates: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]] = {name: [] for name in buckets}
    seen = {name: 0 for name in buckets}

    pbar = tqdm(loader, desc=f"select {split} samples", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        frac = cloud_fraction(batch["cld_shdw"], cloud_index=cloud_index)
        for i, value in enumerate(frac):
            bucket = cloud_bucket_name(float(value.item()))
            if bucket not in candidates:
                continue
            seen[bucket] += 1
            score = visual_structure_score(batch, i, raw_rgb_indices)
            raw_item = slice_tensor_batch(batch, i, cpu=True)
            meta = sample_metadata(raw_item, bucket=bucket, score=score, cloud_frac=float(value.item()))
            keep_top_visual_candidate(
                candidates[bucket],
                (score, raw_item, meta),
                pool_size,
            )
        pbar.set_postfix({key: f"{min(len(value), samples_per_bucket)}/{samples_per_bucket}" for key, value in candidates.items()})
        if buckets_complete(candidates, pool_size):
            break

    selected = select_visual_candidates(candidates, buckets, samples_per_bucket)
    return {
        bucket: [(raw_item, meta) for _, raw_item, meta in selected.get(bucket, [])]
        for bucket in buckets
    }


def model_forward(
    model: torch.nn.Module,
    raw_batch: dict[str, Tensor],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    model_indices = model_band_indices_from_cfg(cfg)
    reflectance_range = model_reflectance_range_from_cfg(cfg)
    batch = apply_model_band_indices(move_batch(raw_batch, device), model_indices, reflectance_range)
    outputs = model(
        batch["s2_toa"],
        batch.get("s1"),
        batch["cld_shdw"],
        softshadow_bbox=batch.get("bbox"),
        softshadow_case=batch.get("shadow_case"),
        return_intermediates=True,
    )
    if not isinstance(outputs, dict):
        outputs = {"I_hat": outputs}
    return {k: v.detach().cpu() for k, v in outputs.items() if isinstance(v, Tensor)}, {
        k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in batch.items()
    }


def save_selected_samples_csv(selected: dict[str, list[tuple[dict[str, Tensor], dict[str, Any]]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "bucket",
        "row",
        "score",
        "cloud_fraction",
        "sample_id",
        "roi_id",
        "clear_date",
        "cloudy_date",
        "clear_s2_path",
        "cloudy_s2_path",
        "dedup_keys",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bucket, samples in selected.items():
            for row, (_, meta) in enumerate(samples):
                record = {key: meta.get(key, "") for key in fields}
                record["row"] = row
                writer.writerow(record)


def make_contact_sheets(
    *,
    selected: dict[str, list[tuple[dict[str, Tensor], dict[str, Any]]]],
    reference_cfg: dict[str, Any],
    run_results: dict[str, dict[str, list[Tensor]]],
    out_dir: Path,
    split: str,
    tile_size: int,
    rgb_stretch: str,
    rgb_gamma: float,
    rgb_gain: float,
) -> dict[str, str]:
    saved: dict[str, str] = {}
    raw_rgb_indices = raw_rgb_indices_from_cfg(reference_cfg)
    for bucket, samples in selected.items():
        if not samples:
            continue
        rows: list[list[Tensor]] = []
        for row_idx, (raw_batch, _) in enumerate(samples):
            cloudy = rgb_panel(raw_batch["s2_toa"], raw_rgb_indices)
            target = rgb_panel(raw_batch["target"], raw_rgb_indices)
            rgb_panels = [cloudy, target]

            run_panels: list[Tensor] = []
            for label, by_bucket in run_results.items():
                del label
                run_panels.append(by_bucket[bucket][row_idx])
            stretched = stretch_rgb_row(
                [*rgb_panels, *run_panels],
                mode=rgb_stretch,
                gamma=rgb_gamma,
                gain=rgb_gain,
            )
            mask = raw_batch["cld_shdw"]
            cloud_index = int(reference_cfg.get("data", {}).get("cloud_index", 1))
            if mask.ndim == 4:
                if mask.shape[1] == 1:
                    mask_panel = (mask.round().long() == cloud_index).float()
                else:
                    mask_panel = mask[:, cloud_index : cloud_index + 1]
            elif mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask_panel = (mask.round().long() == cloud_index).float()
                else:
                    mask_panel = mask[cloud_index : cloud_index + 1]
            else:
                mask_panel = mask
            common_panels = [
                stretched[0],
                stretched[1],
                _sar_panel(raw_batch["s1"]) if isinstance(raw_batch.get("s1"), Tensor) else stretched[0].new_zeros(stretched[0].shape),
                _mask_panel(mask_panel, (0.10, 0.78, 1.00)),
            ]
            row = [*common_panels, *stretched[2:]]
            rows.append([resize_panel(panel, tile_size) for panel in row])

        titles = ["Cloudy S2", "Target", "SAR", "Restore Mask", *run_results.keys()]
        path = out_dir / f"{split}_{bucket}_best_compare.png"
        _save_titled_grid(rows, titles, path)
        saved[bucket] = str(path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=None, help="Explicit run directories to compare.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/allclear"), help="Root used when --runs is omitted.")
    parser.add_argument("--run-glob", default="20*_stage1__*", help="Glob under --runs-root used when --runs is omitted.")
    parser.add_argument("--reference-run", type=Path, default=None, help="Run whose config selects the fixed samples.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS), choices=["low", "medium", "high", "heavy"])
    parser.add_argument("--samples-per-bucket", type=int, default=5)
    parser.add_argument("--candidate-pool-per-bucket", type=int, default=80)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--visual-rgb-stretch", default="target", choices=["panel", "target", "shared_all"])
    parser.add_argument("--visual-rgb-gamma", type=float, default=0.72)
    parser.add_argument("--visual-rgb-gain", type=float, default=1.08)
    parser.add_argument("--strict", action="store_true", help="Raise on incompatible runs instead of skipping them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs:
        run_dirs = [path.expanduser().resolve() for path in args.runs]
    else:
        run_dirs = discover_runs(args.runs_root.expanduser().resolve(), args.run_glob)
    if args.max_runs is not None:
        run_dirs = run_dirs[: int(args.max_runs)]
    if not run_dirs:
        raise SystemExit("No run directories with best checkpoints were found.")

    reference_run = (args.reference_run.expanduser().resolve() if args.reference_run else run_dirs[-1])
    reference_cfg = load_run_config(reference_run)
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else reference_run / "analysis_best_checkpoint_compare"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_reference_samples(
        cfg=reference_cfg,
        split=args.split,
        buckets=tuple(args.buckets),
        samples_per_bucket=int(args.samples_per_bucket),
        candidate_pool_per_bucket=int(args.candidate_pool_per_bucket),
    )
    save_selected_samples_csv(selected, out_dir / f"selected_samples_{args.split}.csv")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    summary: dict[str, Any] = {
        "reference_run": str(reference_run),
        "split": args.split,
        "buckets": args.buckets,
        "samples_per_bucket": args.samples_per_bucket,
        "device": str(device),
        "runs": [],
        "skipped": [],
    }
    run_results: dict[str, dict[str, list[Tensor]]] = {}

    for run_dir in run_dirs:
        label = short_run_label(run_dir)
        try:
            cfg = load_run_config(run_dir)
            checkpoint = find_best_checkpoint(run_dir)
            model = build_model(cfg).to(device)
            load_checkpoint(checkpoint, model)
            model.eval()
            outputs_by_bucket: dict[str, list[Tensor]] = {bucket: [] for bucket in args.buckets}
            with torch.no_grad():
                for bucket in args.buckets:
                    for raw_batch, _ in selected.get(bucket, []):
                        outputs, _ = model_forward(model, raw_batch, cfg, device)
                        pred = outputs.get("I_hat", outputs.get("I_cloud", outputs.get("I_cloud_raw")))
                        if pred is None:
                            raise KeyError("Model output does not contain I_hat/I_cloud/I_cloud_raw.")
                        pred = undo_model_reflectance(pred, cfg)
                        rgb_indices = output_rgb_indices_from_cfg(cfg, pred.shape[1])
                        outputs_by_bucket[bucket].append(rgb_panel(pred, rgb_indices))
            run_results[label] = outputs_by_bucket
            summary["runs"].append(
                {
                    "label": label,
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "config": str(run_dir / "config.resolved.json"),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            if args.strict:
                raise
            summary["skipped"].append({"run_dir": str(run_dir), "label": label, "error": repr(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not run_results:
        raise SystemExit("No run could be evaluated successfully.")

    saved = make_contact_sheets(
        selected=selected,
        reference_cfg=reference_cfg,
        run_results=run_results,
        out_dir=out_dir,
        split=args.split,
        tile_size=int(args.tile_size),
        rgb_stretch=str(args.visual_rgb_stretch),
        rgb_gamma=float(args.visual_rgb_gamma),
        rgb_gain=float(args.visual_rgb_gain),
    )
    summary["saved"] = saved
    summary_path = out_dir / "best_checkpoint_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
