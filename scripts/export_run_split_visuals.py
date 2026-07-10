#!/usr/bin/env python3
"""Export balanced val/test visualizations for one ALLClear training run.

The training loop already writes per-epoch validation grids.  This script is
for fixed-checkpoint reporting: load a run's resolved config and checkpoint,
then save low/medium/high/heavy visual grids for the requested dataset splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.dataset import cloud_fraction
from src.allclear.train import (
    apply_model_band_indices,
    build_model,
    CLOUD_VISUAL_BUCKETS,
    cloud_bucket_name,
    concat_tensor_batches,
    load_checkpoint,
    make_loader,
    model_band_indices_from_cfg,
    model_reflectance_range_from_cfg,
    move_batch,
    save_visuals,
    select_visual_candidates,
    slice_tensor_batch,
    keep_top_visual_candidate,
    visual_structure_score,
)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.resolved.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    yaml_path = run_dir / "config.resolved.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing resolved config in {run_dir}")


def find_checkpoint(run_dir: Path, name: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if name in {"last", "last.pt"}:
        path = ckpt_dir / "last.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if name in {"best", "best.pt"}:
        best = sorted(ckpt_dir.glob("best_epoch_*.pt"))
        if not best:
            raise FileNotFoundError(f"No best_epoch_*.pt in {ckpt_dir}")
        return best[-1]
    path = Path(name).expanduser()
    return path if path.is_absolute() else run_dir / path


def buckets_complete(buckets: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]], n: int) -> bool:
    return all(len(items) >= n for items in buckets.values())


def export_split(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    split: str,
    device: torch.device,
    out_dir: Path,
    checkpoint_label: str,
    samples_per_bucket: int,
    candidate_pool_per_bucket: int,
) -> dict[str, Any]:
    loader = make_loader(cfg, split, distributed=False, rank=0, world_size=1)
    cloud_index = int(cfg.get("data", {}).get("cloud_index", 1))
    rgb_indices = tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    model_band_indices = model_band_indices_from_cfg(cfg)
    model_reflectance_range = model_reflectance_range_from_cfg(cfg)
    pool_size = max(int(samples_per_bucket), int(candidate_pool_per_bucket))
    buckets: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]] = {name: [] for name in CLOUD_VISUAL_BUCKETS}
    seen = {name: 0 for name in CLOUD_VISUAL_BUCKETS}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"visual {split}", unit="batch", dynamic_ncols=True)
        for batch in pbar:
            batch = apply_model_band_indices(move_batch(batch, device), model_band_indices, model_reflectance_range)
            outputs = model(
                batch["s2_toa"],
                batch.get("s1"),
                batch["cld_shdw"],
                softshadow_bbox=batch.get("bbox"),
                softshadow_case=batch.get("shadow_case"),
                return_intermediates=True,
            )
            frac = cloud_fraction(batch["cld_shdw"], cloud_index=cloud_index).to(device)
            for i, value in enumerate(frac):
                bucket = cloud_bucket_name(value)
                seen[bucket] += 1
                score = visual_structure_score(batch, i, rgb_indices)
                keep_top_visual_candidate(
                    buckets[bucket],
                    (score, slice_tensor_batch(outputs, i, cpu=True), slice_tensor_batch(batch, i, cpu=True)),
                    pool_size,
                )
            pbar.set_postfix({key: f"{min(len(value), samples_per_bucket)}/{samples_per_bucket}" for key, value in buckets.items()})
            if buckets_complete(buckets, pool_size):
                break

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    eval_cfg = cfg.get("eval", {})
    visual_profile = str(eval_cfg.get("visual_profile", cfg.get("loss", {}).get("profile", "stage1")))
    visual_rgb_gamma = eval_cfg.get("visual_rgb_gamma")
    visual_rgb_gain = eval_cfg.get("visual_rgb_gain")
    visual_rgb_stretch = eval_cfg.get("visual_rgb_stretch")
    selected_by_bucket = select_visual_candidates(buckets, CLOUD_VISUAL_BUCKETS, samples_per_bucket)
    for bucket, selected in selected_by_bucket.items():
        if not selected:
            continue
        out = concat_tensor_batches([pair[1] for pair in selected])
        batch = concat_tensor_batches([pair[2] for pair in selected])
        path = split_dir / f"{checkpoint_label}_{split}_{bucket}.png"
        save_visuals(
            out,
            batch,
            path,
            rgb_indices,
            max_items=samples_per_bucket,
            visual_profile=visual_profile,
            visual_rgb_gamma=float(visual_rgb_gamma) if visual_rgb_gamma is not None else None,
            visual_rgb_gain=float(visual_rgb_gain) if visual_rgb_gain is not None else None,
            visual_rgb_stretch=str(visual_rgb_stretch) if visual_rgb_stretch is not None else None,
        )
        saved[bucket] = str(path)
    return {
        "split": split,
        "seen": seen,
        "saved": saved,
        "requested_per_bucket": samples_per_bucket,
        "candidate_pool_per_bucket": pool_size,
        "selection": "top_target_rgb_gradient_contrast",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best", help="best, last, or checkpoint path relative to run-dir.")
    parser.add_argument("--splits", nargs="+", default=None, choices=["train", "val", "test"])
    parser.add_argument("--samples-per-bucket", type=int, default=None)
    parser.add_argument("--candidate-pool-per-bucket", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    cfg = load_run_config(run_dir)
    eval_cfg = cfg.get("eval", {})
    splits = args.splits or eval_cfg.get("visualize_splits", ["val", "test"])
    samples_per_bucket = int(args.samples_per_bucket or eval_cfg.get("visual_samples_per_bucket", 5))
    candidate_pool_per_bucket = int(
        args.candidate_pool_per_bucket
        or eval_cfg.get("visual_candidate_pool_per_bucket", max(samples_per_bucket * 5, samples_per_bucket))
    )
    checkpoint = find_checkpoint(run_dir, args.checkpoint)
    checkpoint_label = checkpoint.stem
    out_dir = (args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "visualizations" / "split_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint, model)

    summaries = []
    for split in splits:
        summaries.append(
            export_split(
                model=model,
                cfg=cfg,
                split=split,
                device=device,
                out_dir=out_dir,
                checkpoint_label=checkpoint_label,
                samples_per_bucket=samples_per_bucket,
                candidate_pool_per_bucket=candidate_pool_per_bucket,
            )
        )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(out_dir),
        "splits": summaries,
    }
    summary_path = out_dir / "split_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
