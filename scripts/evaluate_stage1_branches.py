#!/usr/bin/env python3
"""Evaluate Stage1 restoration branches on ALLClear splits.

This script loads a trained checkpoint, runs the model on a manifest split, and
reports region-aware metrics for the candidates that exist in the selected
configuration:

- final Stage1 output on full / clear / shadow / cloud regions;
- cloud branch raw fill inside the cloud region;
- cloudy input baseline in each same region;
- SoftShadow mask quality only for configs that actually use SoftShadow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.config import load_config
from src.allclear.dataset import AllClearDataset
from src.allclear.eval_metrics import (
    METRIC_DATA_DOMAIN,
    METRIC_DATA_RANGE,
    SSIM_WINDOW_SIZE,
    channel_bias_stats,
    haar_swt2,
    hf_magnitude,
    pearson_correlation,
    region_ssim,
    sar_counterfactuals,
    wavelet_metrics,
)
from src.allclear.modules.common import masks_from_cld_shdw
from src.allclear.modules.softshadow import soft_shadow_target
from src.allclear.train import (
    CLOUD_VISUAL_BUCKETS,
    apply_model_band_indices,
    build_model,
    cloud_bucket_name,
    load_checkpoint,
    model_band_indices_from_cfg,
    model_reflectance_range_from_cfg,
    save_visuals,
)


class RegionAccumulator:
    def __init__(self, rgb_indices: tuple[int, int, int] = (0, 1, 2)) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.bias_sum = 0.0
        self.count = 0.0
        self.sam_sum = 0.0
        self.sam_count = 0.0
        self.ssim_sum = 0.0
        self.ssim_count = 0.0
        self.region_frac_sum = 0.0
        self.region_frac_count = 0.0
        self.rgb_indices = tuple(int(v) for v in rgb_indices)
        self.rgb_bias_sum = [0.0, 0.0, 0.0]
        self.rgb_count = [0.0, 0.0, 0.0]

    def update(self, pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
        pred = pred.detach().float()
        target = target.detach().float()
        mask = mask.detach().float().clamp(0.0, 1.0)
        if mask.shape[1] == 1 and pred.shape[1] != 1:
            mask_c = mask.expand(-1, pred.shape[1], -1, -1)
        else:
            mask_c = mask
        count = float(mask_c.sum().item())
        if count <= 1.0e-6:
            return {
                "mae": math.nan,
                "rmse": math.nan,
                "psnr": math.nan,
                "bias": math.nan,
                "sam_deg": math.nan,
                "ssim": math.nan,
                "bias_r": math.nan,
                "bias_g": math.nan,
                "bias_b": math.nan,
                "mean_abs_channel_bias": math.nan,
                "region_frac": 0.0,
                "count": 0.0,
            }
        diff = pred - target
        abs_sum = float((diff.abs() * mask_c).sum().item())
        sq_sum = float((diff.square() * mask_c).sum().item())
        bias_sum = float((diff * mask_c).sum().item())
        self.abs_sum += abs_sum
        self.sq_sum += sq_sum
        self.bias_sum += bias_sum
        self.count += count
        sample_region_frac = float(mask[:, :1].mean().item())
        self.region_frac_sum += sample_region_frac
        self.region_frac_count += 1.0

        sample_mae = abs_sum / count
        sample_rmse = math.sqrt(sq_sum / count)
        sample_psnr = 20.0 * math.log10(METRIC_DATA_RANGE / max(sample_rmse, 1.0e-12))
        sample_bias = bias_sum / count
        rgb = channel_bias_stats(pred, target, mask, self.rgb_indices)
        rgb_mask = mask[:, :1]
        rgb_denom = float(rgb_mask.sum().item())
        for index, channel in enumerate(self.rgb_indices):
            channel_sum = float((diff[:, channel : channel + 1] * rgb_mask).sum().item())
            self.rgb_bias_sum[index] += channel_sum
            self.rgb_count[index] += rgb_denom

        pixel_mask = mask[:, :1] > 0.5
        sample_sam = math.nan
        if pixel_mask.any():
            p = pred.permute(0, 2, 3, 1)[pixel_mask[:, 0]]
            t = target.permute(0, 2, 3, 1)[pixel_mask[:, 0]]
            denom = p.norm(dim=1) * t.norm(dim=1)
            valid = denom > 1.0e-8
            if valid.any():
                cos = (p[valid] * t[valid]).sum(dim=1) / denom[valid]
                angle = torch.acos(cos.clamp(-1.0, 1.0)) * (180.0 / math.pi)
                self.sam_sum += float(angle.sum().item())
                self.sam_count += float(angle.numel())
                sample_sam = float(angle.mean().item())

        ssim_values = []
        for sample_index in range(pred.shape[0]):
            value = region_ssim(
                pred[sample_index : sample_index + 1],
                target[sample_index : sample_index + 1],
                mask[sample_index : sample_index + 1],
                rgb_indices=self.rgb_indices,
                data_range=METRIC_DATA_RANGE,
                window_size=SSIM_WINDOW_SIZE,
            )
            if math.isfinite(value):
                ssim_values.append(value)
        sample_ssim = float(sum(ssim_values) / len(ssim_values)) if ssim_values else math.nan
        if math.isfinite(sample_ssim):
            self.ssim_sum += sample_ssim
            self.ssim_count += 1.0

        return {
            "mae": sample_mae,
            "rmse": sample_rmse,
            "psnr": sample_psnr,
            "bias": sample_bias,
            "sam_deg": sample_sam,
            "ssim": sample_ssim,
            **rgb,
            "region_frac": sample_region_frac,
            "count": count,
        }

    def current(self, region_frac: float | None = None) -> dict[str, float]:
        if self.count <= 1.0e-6:
            return {
                "mae": math.nan,
                "rmse": math.nan,
                "psnr": math.nan,
                "bias": math.nan,
                "sam_deg": math.nan,
                "ssim": math.nan,
                "bias_r": math.nan,
                "bias_g": math.nan,
                "bias_b": math.nan,
                "mean_abs_channel_bias": math.nan,
                "region_frac": region_frac or 0.0,
                "count": 0.0,
            }
        mae = self.abs_sum / self.count
        rmse = math.sqrt(self.sq_sum / self.count)
        psnr = 20.0 * math.log10(METRIC_DATA_RANGE / max(rmse, 1.0e-12))
        sam = self.sam_sum / self.sam_count if self.sam_count > 0 else math.nan
        rgb_bias = [
            self.rgb_bias_sum[i] / self.rgb_count[i] if self.rgb_count[i] > 1.0e-6 else math.nan
            for i in range(3)
        ]
        return {
            "mae": mae,
            "rmse": rmse,
            "psnr": psnr,
            "bias": self.bias_sum / self.count,
            "sam_deg": sam,
            "ssim": self.ssim_sum / self.ssim_count if self.ssim_count > 0 else math.nan,
            "bias_r": rgb_bias[0],
            "bias_g": rgb_bias[1],
            "bias_b": rgb_bias[2],
            "mean_abs_channel_bias": float(sum(abs(v) for v in rgb_bias) / 3.0) if all(math.isfinite(v) for v in rgb_bias) else math.nan,
            "region_frac": self.region_frac_sum / self.region_frac_count if self.region_frac_count > 0 else (region_frac if region_frac is not None else math.nan),
            "count": self.count,
        }


class ScalarAccumulator:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: Tensor, mask: Tensor | None = None) -> float:
        value = value.detach().float()
        if mask is not None:
            mask = mask.detach().float()
            while mask.shape[1] != value.shape[1]:
                mask = mask.expand(-1, value.shape[1], -1, -1)
                break
            denom = float(mask.sum().item())
            if denom <= 1.0e-6:
                return math.nan
            total = float((value * mask).sum().item())
            self.sum += total
            self.count += denom
            return total / denom
        total = float(value.sum().item())
        self.sum += total
        self.count += float(value.numel())
        return total / max(float(value.numel()), 1.0)

    def current(self) -> float:
        return self.sum / self.count if self.count > 0 else math.nan


class WaveletAccumulator:
    """Sample-mean accumulator for the fixed Haar metrics."""

    KEYS = (
        "ll_mae",
        "lh_mae",
        "hl_mae",
        "hh_mae",
        "hf_energy_ratio_pred",
        "hf_energy_ratio_target",
        "hf_energy_ratio_abs_delta",
    )

    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {key: [] for key in self.KEYS}

    def update(self, values: dict[str, float]) -> None:
        for key in self.KEYS:
            value = safe_float(values.get(key))
            if math.isfinite(value):
                self.values[key].append(value)

    def current(self) -> dict[str, float | int]:
        return {
            key: (sum(values) / len(values) if values else math.nan)
            for key, values in self.values.items()
        } | {"valid_samples": max((len(values) for values in self.values.values()), default=0)}


def batch_sample_ids(batch: dict[str, Any]) -> list[str]:
    value = batch.get("sample_id", [])
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if torch.is_tensor(value):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [str(value)] if value not in (None, "") else []


def _model_forward(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Tensor]:
    return model(
        batch["s2_toa"],
        batch.get("s1"),
        batch["cld_shdw"],
        softshadow_bbox=batch.get("bbox"),
        softshadow_case=batch.get("shadow_case"),
        return_intermediates=True,
    )


def _sar_output_delta_stats(real_output: Tensor, alternative_output: Tensor) -> dict[str, float]:
    delta = real_output - alternative_output
    bands = haar_swt2(delta.float())
    ll = bands["LL"].abs().mean().item()
    hf = torch.cat([bands[name].abs() for name in ("LH", "HL", "HH")], dim=1).mean().item()
    return {
        "mae": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "ll_mae": float(ll),
        "hf_mae": float(hf),
    }


def evaluate_sar_counterfactuals(
    model: torch.nn.Module,
    dataset: AllClearDataset,
    cfg: dict[str, Any],
    *,
    split: str,
    device: torch.device,
    model_band_indices: tuple[int, ...] | None,
    model_reflectance_range: tuple[float, float] | None,
    shadow_index: int,
    cloud_index: int,
    batch_size: int,
    num_workers: int,
    limit: int,
    low_pass_kernel: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Run real/zero/shuffle/low/high SAR counterfactuals in eval mode.

    This is intentionally a second pass.  The regular evaluator keeps its
    historical batch-size-one ordering and visualization behavior; the second
    pass uses batches so that a true batch shuffle exists.
    """

    from torch.utils.data import DataLoader

    if not bool(getattr(dataset, "load_sar", False)):
        return {}, {"status": "no_sar", "split": split, "valid_samples": 0, "shuffle_valid_samples": 0}
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    per_sample: dict[str, dict[str, float]] = {}
    aggregate: dict[str, list[float]] = defaultdict(list)
    valid_samples = 0
    shuffle_valid_samples = 0
    processed = 0
    with torch.inference_mode():
        for batch_cpu in loader:
            if limit and processed >= limit:
                break
            current_batch = batch_cpu
            if limit and processed + len(batch_sample_ids(current_batch)) > limit:
                keep = limit - processed
                current_batch = {
                    key: (value[:keep] if torch.is_tensor(value) else value[:keep] if isinstance(value, list) else value)
                    for key, value in current_batch.items()
                }
            batch = apply_model_band_indices(move_batch(current_batch, device), model_band_indices, model_reflectance_range)
            sar = batch.get("s1")
            if not torch.is_tensor(sar):
                continue
            scenarios = sar_counterfactuals(sar, low_pass_kernel=low_pass_kernel)
            scenario_outputs: dict[str, Tensor] = {}
            for name in ("real", "zero", "shuffle", "low_pass", "high_pass"):
                scenario_batch = dict(batch)
                scenario_batch["s1"] = scenarios[name]
                scenario_outputs[name] = _model_forward(model, scenario_batch)["I_hat"]
            masks = masks_from_cld_shdw(batch["cld_shdw"], shadow_index=shadow_index, cloud_index=cloud_index)
            error_hf = hf_magnitude(scenario_outputs["real"] - batch["target"])
            sar_hf = hf_magnitude(scenarios["high_pass"])
            ids = batch_sample_ids(current_batch)
            for index, sample_id in enumerate(ids):
                if not sample_id:
                    sample_id = f"sample_{processed + index:06d}"
                row: dict[str, float] = {}
                real = scenario_outputs["real"][index : index + 1]
                target = batch["target"][index : index + 1]
                sar_item = scenarios["real"][index : index + 1]
                row["D_LF_SAR"] = float(scenarios["low_pass"][index : index + 1].square().mean().sqrt().item() / max(sar_item.square().mean().sqrt().item(), 1.0e-12))
                row["D_HF_SAR"] = float(scenarios["high_pass"][index : index + 1].square().mean().sqrt().item() / max(sar_item.square().mean().sqrt().item(), 1.0e-12))
                row["sar_lf_energy_ratio"] = row["D_LF_SAR"]
                row["sar_hf_energy_ratio"] = row["D_HF_SAR"]
                row["sar_error_hf_corr_full"] = pearson_correlation(error_hf[index], sar_hf[index])
                cloud_mask = masks.cloud[index : index + 1]
                error_hf_cloud = error_hf[index : index + 1] * cloud_mask
                sar_hf_cloud = sar_hf[index : index + 1] * cloud_mask
                row["sar_error_hf_corr_cloud"] = pearson_correlation(error_hf_cloud, sar_hf_cloud)
                for name, label in (("shuffle", "shuffle"), ("low_pass", "lf"), ("high_pass", "hf"), ("zero", "zero")):
                    delta_stats = _sar_output_delta_stats(real, scenario_outputs[name][index : index + 1])
                    for key, value in delta_stats.items():
                        row[f"sar_real_vs_{label}_{key}"] = value
                row["sar_real_output_mae"] = float((real - target).abs().mean().item())
                row["sar_real_output_rmse"] = float((real - target).square().mean().sqrt().item())
                is_shuffle_valid = bool(scenarios["shuffle_valid"])
                row["sar_shuffle_valid"] = 1.0 if is_shuffle_valid else 0.0
                if not is_shuffle_valid:
                    for key in tuple(row):
                        if key.startswith("sar_real_vs_shuffle_"):
                            row[key] = math.nan
                else:
                    shuffle_valid_samples += 1
                valid_samples += 1
                per_sample[sample_id] = row
                for key, value in row.items():
                    if math.isfinite(value):
                        aggregate[key].append(value)
            processed += len(ids)
    summary = {
        "status": "ok" if valid_samples else "no_valid_samples",
        "split": split,
        "low_pass": {"operator": "5x5 uniform average", "padding": "reflect", "stride": 1},
        "high_pass": "SAR_HF = SAR_real - SAR_LF",
        "no_counterfactual_renormalization": True,
        "valid_samples": valid_samples,
        "shuffle_valid_samples": shuffle_valid_samples,
        "metrics": {key: (sum(values) / len(values) if values else math.nan) for key, values in sorted(aggregate.items())},
    }
    return per_sample, summary


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def metric_row(prefix: str, stats: dict[str, float]) -> dict[str, float | str]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def json_safe(value: Any) -> Any:
    """Convert non-finite floats to JSON null without changing legacy files."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def build_dataset(cfg: dict[str, Any], split: str) -> AllClearDataset:
    data = cfg["data"]
    def split_value(key: str) -> Any:
        return data.get(f"{key}_{split}", data.get(key))
    return AllClearDataset(
        root=data["root"],
        manifest=data[f"{split}_manifest"],
        optical_scale=float(data.get("optical_scale", 10000.0)),
        image_size=data.get("image_size"),
        shadow_index=int(data.get("shadow_index", 3)),
        cloud_index=int(data.get("cloud_index", 1)),
        prefer_original_cld_shdw=bool(data.get("prefer_original_cld_shdw", True)),
        load_sar=bool(data.get("load_sar", True)),
        cache_dir=data.get("cache_dir"),
        band_indices=data.get("band_indices"),
        softshadow_mask_dir=split_value("softshadow_mask_dir"),
        softshadow_bbox_path=split_value("softshadow_bbox_path"),
        softshadow_bbox_space=str(cfg.get("model", {}).get("softshadow_bbox_space", data.get("softshadow_bbox_space", "image"))),
        softshadow_sam_input_size=int(cfg.get("model", {}).get("softshadow_sam_input_size", data.get("softshadow_sam_input_size", 1024))),
        softshadow_shadow_case_enabled=bool(data.get("softshadow_shadow_case_enabled", False)),
        softshadow_shadow_case_positive_threshold=float(data.get("softshadow_shadow_case_positive_threshold", 0.05)),
        softshadow_absent_shadow_threshold=float(data.get("softshadow_absent_shadow_threshold", 0.002)),
        softshadow_absent_division_threshold=float(data.get("softshadow_absent_division_threshold", 0.002)),
        softshadow_valid_shadow_threshold=float(data.get("softshadow_valid_shadow_threshold", 0.002)),
        softshadow_valid_division_threshold=float(data.get("softshadow_valid_division_threshold", 0.005)),
        softshadow_max_bbox_area=float(data.get("softshadow_max_bbox_area", 0.95)),
        softshadow_max_clear_leakage=float(data.get("softshadow_max_clear_leakage", 0.30)),
        softshadow_max_cloud_leakage=float(data.get("softshadow_max_cloud_leakage", 0.30)),
        softshadow_min_division_shadow_precision_dilated=float(
            data.get("softshadow_min_division_shadow_precision_dilated", 0.30)
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help=(
            "Checkpoint path, or one of: best, last, none. "
            "When best/last is used, the run directory is inferred from --run-dir "
            "or from config.resolved.json."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run directory used to resolve best/last checkpoints.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/allclear_branch_eval"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-visuals", type=int, default=0, help="Legacy mode: save the first N per-sample visual grids.")
    parser.add_argument("--visual-samples-per-bucket", type=int, default=None, help="Save N low/medium/high/heavy visual samples.")
    parser.add_argument(
        "--sar-counterfactual",
        action="store_true",
        help="Run the additional real/zero/shuffle/low-pass/high-pass SAR evaluation pass.",
    )
    parser.add_argument("--sar-batch-size", type=int, default=4, help="Batch size for the SAR counterfactual pass.")
    parser.add_argument("--sar-low-pass-kernel", type=int, default=5, help="Odd reflect-padded box kernel for SAR_LF.")
    return parser.parse_args()


def infer_run_dir(config_path: str | Path, run_dir: Path | None) -> Path | None:
    if run_dir is not None:
        return run_dir.expanduser().resolve()
    cfg_path = Path(config_path).expanduser().resolve()
    if cfg_path.name == "config.resolved.json" and cfg_path.parent.name:
        return cfg_path.parent
    return None


def resolve_checkpoint(checkpoint: str, config_path: str | Path, run_dir: Path | None) -> Path | None:
    value = str(checkpoint)
    if value.lower() in {"", "none", "null", "random"}:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    run = infer_run_dir(config_path, run_dir)
    if run is None:
        raise FileNotFoundError(
            f"Cannot resolve checkpoint={checkpoint!r}; pass a checkpoint path or --run-dir."
        )
    ckpt_dir = run / "checkpoints"
    if value.lower() == "last":
        last = ckpt_dir / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"Missing last checkpoint: {last}")
        return last.resolve()
    if value.lower() == "best":
        best = sorted(ckpt_dir.glob("best_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        if not best:
            raise FileNotFoundError(f"No best_epoch_*.pt found in {ckpt_dir}")
        return best[-1].resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def resolve_rgb_indices(
    cfg: dict[str, Any],
    model_band_indices: tuple[int, ...] | None,
    channels: int,
) -> tuple[int, int, int]:
    """Map manifest/original RGB indices into the current model tensor."""

    configured = tuple(int(v) for v in cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    if model_band_indices is not None and all(value in model_band_indices for value in configured):
        mapped = tuple(model_band_indices.index(value) for value in configured)
    elif all(0 <= value < channels for value in configured):
        mapped = configured
    elif channels >= 3:
        mapped = (0, 1, 2)
    else:
        raise ValueError(f"Cannot resolve RGB indices {configured} for {channels} model channels")
    if max(mapped) >= channels:
        raise ValueError(f"Resolved RGB indices {mapped} exceed {channels} model channels")
    return mapped


def checkpoint_metadata(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"path": None, "epoch": None, "best_metric": None}
    ckpt = torch.load(checkpoint, map_location="cpu")
    return {
        "path": str(checkpoint),
        "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) and "epoch" in ckpt else None,
        "best_metric": float(ckpt.get("best_metric", math.nan)) if isinstance(ckpt, dict) else math.nan,
        "keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else [],
    }


def cloud_bucket_from_fraction(value: float) -> str:
    return cloud_bucket_name(value)


def candidate_roles(framework: str) -> dict[str, str]:
    fw = framework.lower()
    if fw in {"softshadow_dadigan_baseline", "softshadow_dadigan", "sam_dadigan"}:
        return {
            "final": "final composite from one DaDiGAN restoration mask",
            "cloudy": "input cloudy S2 baseline",
            "shadow_branch": "not an independent branch in this framework; I_shadow aliases I_hat",
            "cloud_branch_raw": "DaDiGAN raw fill",
            "cloud_branch_composite": "DaDiGAN composite with M_restore",
        }
    if fw in {"dadigan_baseline", "strict_dadigan", "dadigan"}:
        return {
            "final": "DADIGAN baseline output",
            "cloudy": "input cloudy S2 baseline",
            "shadow_branch": "not available; I_shadow aliases input S2",
            "cloud_branch_raw": "DADIGAN raw fill",
            "cloud_branch_composite": "DADIGAN mask composite",
        }
    return {
        "final": "hard semantic composite",
        "cloudy": "input cloudy S2 baseline",
        "shadow_branch": "SoftShadow shadow removal candidate",
        "cloud_branch_raw": "DaDiGAN cloud raw fill",
        "cloud_branch_composite": "DaDiGAN cloud composite",
    }


def is_cloud_only_profile(cfg: dict[str, Any]) -> bool:
    profile = str(cfg.get("loss", {}).get("profile", "")).lower()
    framework = str(cfg.get("model", {}).get("framework", "")).lower()
    return profile in {"cloud_only", "dadigan_lama_ffc", "dadigan_baseline"} or framework in {
        "dadigan_baseline",
        "strict_dadigan",
        "dadigan",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_model(cfg).to(device).eval()
    ckpt_path = resolve_checkpoint(args.checkpoint, args.config, args.run_dir)
    ckpt_meta = checkpoint_metadata(ckpt_path)
    if ckpt_path is not None:
        load_checkpoint(ckpt_path, model)

    model_band_indices = model_band_indices_from_cfg(cfg)
    model_reflectance_range = model_reflectance_range_from_cfg(cfg)
    model_channels = int(cfg.get("model", {}).get("s2_channels", len(model_band_indices) if model_band_indices is not None else 13))
    rgb_indices = resolve_rgb_indices(cfg, model_band_indices, model_channels)
    soft_target_kernel = int(cfg.get("loss", {}).get("shadow_soft_target_low_pass_kernel", 5))
    shadow_index = int(cfg.get("data", {}).get("shadow_index", cfg.get("model", {}).get("shadow_index", 3)))
    cloud_index = int(cfg.get("data", {}).get("cloud_index", cfg.get("model", {}).get("cloud_index", 1)))
    framework = str(cfg.get("model", {}).get("framework", "stage1")).lower()
    cloud_only = is_cloud_only_profile(cfg)
    visual_profile = str(cfg.get("eval", {}).get("visual_profile", cfg.get("loss", {}).get("profile", "stage1")))
    visual_rgb_gamma = cfg.get("eval", {}).get("visual_rgb_gamma")
    visual_rgb_gain = cfg.get("eval", {}).get("visual_rgb_gain")
    visual_rgb_stretch = cfg.get("eval", {}).get("visual_rgb_stretch")
    visual_samples_per_bucket = args.visual_samples_per_bucket
    if visual_samples_per_bucket is None:
        visual_samples_per_bucket = int(cfg.get("eval", {}).get("visual_samples_per_bucket", 5))

    regions = ["full", "clear", "shadow", "cloud"]
    candidates = {
        "final": "I_hat",
        "cloudy": "s2_toa",
        "cloud_branch_raw": "I_cloud_raw",
        "cloud_branch_composite": "I_cloud",
    }
    if not cloud_only:
        candidates["shadow_branch"] = "I_shadow"
    acc: dict[str, RegionAccumulator] = defaultdict(lambda: RegionAccumulator(rgb_indices))
    bucket_acc: dict[str, RegionAccumulator] = defaultdict(lambda: RegionAccumulator(rgb_indices))
    wavelet_acc: dict[str, WaveletAccumulator] = defaultdict(WaveletAccumulator)
    bucket_wavelet_acc: dict[str, WaveletAccumulator] = defaultdict(WaveletAccumulator)
    scalar_acc: dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
    bucket_scalar_acc: dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
    case_scalar_acc: dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
    sample_rows: list[dict[str, Any]] = []
    saved_by_bucket: dict[str, int] = {name: 0 for name in CLOUD_VISUAL_BUCKETS}
    saved_legacy = 0

    pbar = tqdm(enumerate(loader), total=min(len(loader), args.limit) if args.limit else len(loader), desc=f"eval {args.split}", unit="sample")
    with torch.no_grad():
        for idx, batch_cpu in pbar:
            if args.limit and idx >= args.limit:
                break
            row_meta = dataset.rows[idx]
            batch = apply_model_band_indices(move_batch(batch_cpu, device), model_band_indices, model_reflectance_range)
            outputs = _model_forward(model, batch)
            masks = masks_from_cld_shdw(batch["cld_shdw"], shadow_index=shadow_index, cloud_index=cloud_index)
            region_masks = {
                "full": torch.ones_like(masks.clear),
                "clear": masks.clear,
                "known": masks.clear,
                "shadow": masks.shadow,
                "cloud": masks.cloud,
            }
            image_map = {
                "I_hat": outputs["I_hat"],
                "s2_toa": batch["s2_toa"],
                "I_cloud_raw": outputs["I_cloud_raw"],
                "I_cloud": outputs["I_cloud"],
            }
            if not cloud_only:
                image_map["I_shadow"] = outputs["I_shadow"]

            sample: dict[str, Any] = {
                "sample_id": batch_cpu.get("sample_id", [""])[0] if isinstance(batch_cpu.get("sample_id"), list) else batch_cpu.get("sample_id", ""),
                "manifest_bucket": row_meta.get("bucket", ""),
                "cloud_bucket": cloud_bucket_from_fraction(float(masks.cloud.mean().item())),
                "target_degraded_ratio": safe_float(row_meta.get("target_degraded_ratio", row_meta.get("official_target_metric", ""))),
                "cloud_frac": float(masks.cloud.mean().item()),
                "shadow_frac": float(masks.shadow.mean().item()),
                "clear_frac": float(masks.clear.mean().item()),
                "shadow_case": int(batch_cpu["shadow_case"][0].item()) if "shadow_case" in batch_cpu else -1,
                "shadow_case_name": {0: "no_shadow", 1: "valid_shadow", 2: "ambiguous"}.get(
                    int(batch_cpu["shadow_case"][0].item()) if "shadow_case" in batch_cpu else -1,
                    "disabled",
                ),
                "m_restore_mean": float(outputs.get("M_restore", outputs.get("M_cloud", masks.cloud)).float().mean().item()),
                "m_cloud_output_mean": float(outputs.get("M_cloud", masks.cloud).float().mean().item()),
            }
            if not cloud_only:
                sample.update(
                    {
                        "m_shadow_soft_mean": float(outputs.get("M_shadow_soft", masks.shadow.new_zeros(masks.shadow.shape)).float().mean().item()),
                        "m_shadow_soft_max": float(outputs.get("M_shadow_soft", masks.shadow.new_zeros(masks.shadow.shape)).float().max().item()),
                        "m_shadow_soft_raw_mean": float(outputs.get("M_shadow_soft_raw", outputs.get("M_shadow_soft", masks.shadow.new_zeros(masks.shadow.shape))).float().mean().item()),
                        "m_shadow_soft_eff_mean": float(outputs.get("M_shadow_soft_eff", outputs.get("M_shadow_soft", masks.shadow.new_zeros(masks.shadow.shape))).float().mean().item()),
                    }
                )
            for region_name, region_mask in region_masks.items():
                for cand_name, output_key in candidates.items():
                    if cand_name == "shadow_branch" and region_name not in {"shadow", "full"}:
                        continue
                    if cand_name.startswith("cloud_branch") and region_name not in {"cloud", "full"}:
                        continue
                    stats = acc[f"{cand_name}/{region_name}"].update(image_map[output_key], batch["target"], region_mask)
                    bucket_acc[f"{sample['cloud_bucket']}/{cand_name}/{region_name}"].update(image_map[output_key], batch["target"], region_mask)
                    sample.update(metric_row(f"{cand_name}_{region_name}", stats))

            for cand_name, output_key in candidates.items():
                for region_name in ("full", "cloud"):
                    wavelet = wavelet_metrics(
                        image_map[output_key],
                        batch["target"],
                        region_masks[region_name],
                    )
                    prefix = f"{cand_name}_{region_name}_wavelet"
                    sample.update({f"{prefix}_{key}": value for key, value in wavelet.items()})
                    wavelet_acc[f"{cand_name}/{region_name}"].update(wavelet)
                    bucket_wavelet_acc[f"{sample['cloud_bucket']}/{cand_name}/{region_name}"].update(wavelet)

            cloud_base = sample.get("cloudy_cloud_mae", math.nan)
            cloud_pred = sample.get("cloud_branch_raw_cloud_mae", math.nan)
            shadow_base = sample.get("cloudy_shadow_mae", math.nan)
            shadow_pred = sample.get("shadow_branch_shadow_mae", math.nan)
            sample["cloud_mae_improve_vs_cloudy_pct"] = 100.0 * (cloud_base - cloud_pred) / cloud_base if math.isfinite(cloud_base) and cloud_base > 1.0e-12 and math.isfinite(cloud_pred) else math.nan
            sample["shadow_mae_improve_vs_cloudy_pct"] = 100.0 * (shadow_base - shadow_pred) / shadow_base if math.isfinite(shadow_base) and shadow_base > 1.0e-12 and math.isfinite(shadow_pred) else math.nan

            if not cloud_only:
                if "sam_mask" in batch:
                    soft_gt = batch["sam_mask"].float()
                    if soft_gt.shape[-2:] != masks.shadow.shape[-2:]:
                        soft_gt = torch.nn.functional.interpolate(
                            soft_gt,
                            size=masks.shadow.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    soft_gt = soft_gt[:, :1].clamp(0.0, 1.0)
                else:
                    soft_gt = soft_shadow_target(
                        batch["s2_toa"],
                        batch["target"],
                        masks.shadow,
                        rgb_indices=rgb_indices,
                        low_pass_kernel=soft_target_kernel,
                    )
                soft_eff = outputs.get("M_shadow_soft_eff", outputs.get("M_shadow_soft", torch.zeros_like(masks.shadow))).float()
                soft_raw = outputs.get("M_shadow_soft_raw", soft_eff).float()
                case_name = str(sample["shadow_case_name"])
                bucket_name = str(sample["cloud_bucket"])
                soft_metrics = {
                    "soft_shadow_eff_mae_in_shadow": ((soft_eff - soft_gt).abs(), masks.shadow),
                    "soft_shadow_eff_mse_in_shadow": ((soft_eff - soft_gt).square(), masks.shadow),
                    "soft_shadow_eff_leakage_clear": (soft_eff, masks.clear),
                    "soft_shadow_raw_mae_in_shadow": ((soft_raw - soft_gt).abs(), masks.shadow),
                    "soft_shadow_raw_mse_in_shadow": ((soft_raw - soft_gt).square(), masks.shadow),
                    "soft_shadow_raw_leakage_clear": (soft_raw, masks.clear),
                    "soft_shadow_eff_penumbra_frac": (((soft_eff > 0.01) & (soft_eff < 0.99)).float(), masks.shadow),
                    "soft_shadow_raw_penumbra_frac": (((soft_raw > 0.01) & (soft_raw < 0.99)).float(), masks.shadow),
                }
                for metric_name, (value, mask) in soft_metrics.items():
                    sample[metric_name] = scalar_acc[metric_name].update(value, mask)
                    bucket_scalar_acc[f"{bucket_name}/{metric_name}"].update(value, mask)
                    case_scalar_acc[f"{case_name}/{metric_name}"].update(value, mask)
                # Backward-compatible aliases used by earlier summaries.
                sample["soft_shadow_mae_in_shadow"] = sample["soft_shadow_eff_mae_in_shadow"]
                sample["soft_shadow_mse_in_shadow"] = sample["soft_shadow_eff_mse_in_shadow"]
                sample["soft_shadow_leakage_clear"] = sample["soft_shadow_eff_leakage_clear"]
                sample["soft_shadow_penumbra_frac"] = sample["soft_shadow_eff_penumbra_frac"]

            sample_rows.append(sample)

            if args.save_visuals > 0 and saved_legacy < args.save_visuals:
                save_visuals(
                    {k: v.detach().cpu() for k, v in outputs.items()},
                    {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                    vis_dir / f"first_{idx:04d}_{sample['cloud_bucket']}_{sample['manifest_bucket']}.png",
                    rgb_indices,
                    max_items=1,
                    visual_profile=visual_profile,
                    visual_rgb_gamma=float(visual_rgb_gamma) if visual_rgb_gamma is not None else None,
                    visual_rgb_gain=float(visual_rgb_gain) if visual_rgb_gain is not None else None,
                    visual_rgb_stretch=str(visual_rgb_stretch) if visual_rgb_stretch is not None else None,
                )
                saved_legacy += 1
            bucket_name = str(sample["cloud_bucket"])
            if visual_samples_per_bucket > 0 and saved_by_bucket.get(bucket_name, 0) < visual_samples_per_bucket:
                bucket_count = saved_by_bucket[bucket_name]
                save_visuals(
                    {k: v.detach().cpu() for k, v in outputs.items()},
                    {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                    vis_dir / f"{bucket_name}_{bucket_count:02d}_idx{idx:04d}_{sample['manifest_bucket']}.png",
                    rgb_indices,
                    max_items=1,
                    visual_profile=visual_profile,
                    visual_rgb_gamma=float(visual_rgb_gamma) if visual_rgb_gamma is not None else None,
                    visual_rgb_gain=float(visual_rgb_gain) if visual_rgb_gain is not None else None,
                    visual_rgb_stretch=str(visual_rgb_stretch) if visual_rgb_stretch is not None else None,
                )
                saved_by_bucket[bucket_name] = bucket_count + 1

    # The counterfactual pass is opt-in because it requires five inference
    # forwards per batch.  Its output file is always written so downstream
    # tooling can distinguish disabled from unavailable SAR.
    sar_rows: dict[str, dict[str, float]] = {}
    if args.sar_counterfactual:
        sar_rows, sar_summary = evaluate_sar_counterfactuals(
            model,
            dataset,
            cfg,
            split=args.split,
            device=device,
            model_band_indices=model_band_indices,
            model_reflectance_range=model_reflectance_range,
            shadow_index=shadow_index,
            cloud_index=cloud_index,
            batch_size=args.sar_batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            low_pass_kernel=args.sar_low_pass_kernel,
        )
        for row in sample_rows:
            sample_id = str(row.get("sample_id", ""))
            row.update(sar_rows.get(sample_id, {}))
    else:
        sar_summary = {
            "status": "disabled",
            "split": args.split,
            "valid_samples": 0,
            "shuffle_valid_samples": 0,
            "metrics": {},
        }

    summary: dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": ckpt_meta,
        "framework": framework,
        "cloud_only": cloud_only,
        "candidate_roles": candidate_roles(framework),
        "split": args.split,
        "samples": len(sample_rows),
        "metric_domain": METRIC_DATA_DOMAIN,
        "data_range": METRIC_DATA_RANGE,
        "model_reflectance_stretch": list(model_reflectance_range) if model_reflectance_range is not None else None,
        "optical_scale_before_model_stretch": float(cfg.get("data", {}).get("optical_scale", 10000.0)),
        "ssim": {
            "data_range": METRIC_DATA_RANGE,
            "window_size": SSIM_WINDOW_SIZE,
            "window": "uniform",
            "channel_aggregation": "mean over RGB channels and spatial SSIM map",
            "region_rule": "tight bbox of mask > 0.5; empty or <3x3 crop is NaN",
        },
        "metrics": {},
        "bucket_metrics": {},
        "wavelet": {
            "filter": "one-level undecimated Haar SWT",
            "bands": ["LL", "LH", "HL", "HH"],
            "boundary": "one-pixel right/bottom reflect padding; no decimation",
            "low_high_orientation": "LH=low-y/high-x, HL=high-y/low-x",
        },
        "sar_counterfactual": sar_summary,
        "softshadow": {} if cloud_only else {name: meter.current() for name, meter in scalar_acc.items()},
        "bucket_softshadow": {},
        "softshadow_by_case": {},
        "visualizations": {
            "saved_by_cloud_bucket": saved_by_bucket,
            "legacy_first_n_saved": saved_legacy,
            "visual_samples_per_bucket": visual_samples_per_bucket,
        },
    }
    for name, meter in sorted(acc.items()):
        cand, region = name.split("/")
        summary["metrics"].setdefault(cand, {})[region] = meter.current()
    for name, meter in sorted(bucket_acc.items()):
        bucket_name, cand, region = name.split("/")
        summary["bucket_metrics"].setdefault(bucket_name, {}).setdefault(cand, {})[region] = meter.current()
    wavelet_summary: dict[str, Any] = {
        "metric_domain": METRIC_DATA_DOMAIN,
        "data_range": METRIC_DATA_RANGE,
        "model_reflectance_stretch": list(model_reflectance_range) if model_reflectance_range is not None else None,
        "optical_scale_before_model_stretch": float(cfg.get("data", {}).get("optical_scale", 10000.0)),
        "filter": "one-level undecimated Haar SWT",
        "boundary": "one-pixel right/bottom reflect padding; no decimation",
        "metrics": {},
        "bucket_metrics": {},
    }
    for name, meter in sorted(wavelet_acc.items()):
        cand, region = name.split("/")
        wavelet_summary["metrics"].setdefault(cand, {})[region] = meter.current()
    for name, meter in sorted(bucket_wavelet_acc.items()):
        bucket_name, cand, region = name.split("/", 2)
        wavelet_summary["bucket_metrics"].setdefault(bucket_name, {}).setdefault(cand, {})[region] = meter.current()
    if not cloud_only:
        for name, meter in sorted(bucket_scalar_acc.items()):
            bucket_name, metric_name = name.split("/", 1)
            summary["bucket_softshadow"].setdefault(bucket_name, {})[metric_name] = meter.current()
        for name, meter in sorted(case_scalar_acc.items()):
            case_name, metric_name = name.split("/", 1)
            summary["softshadow_by_case"].setdefault(case_name, {})[metric_name] = meter.current()

    # Aggregate improvement from global accumulators.
    cloud_base = summary["metrics"].get("cloudy", {}).get("cloud", {}).get("mae", math.nan)
    cloud_pred = summary["metrics"].get("cloud_branch_raw", {}).get("cloud", {}).get("mae", math.nan)
    shadow_base = summary["metrics"].get("cloudy", {}).get("shadow", {}).get("mae", math.nan)
    shadow_pred = summary["metrics"].get("shadow_branch", {}).get("shadow", {}).get("mae", math.nan)
    summary["branch_improvement"] = {
        "cloud_mae_improve_vs_cloudy_pct": 100.0 * (cloud_base - cloud_pred) / cloud_base if math.isfinite(cloud_base) and cloud_base > 1.0e-12 and math.isfinite(cloud_pred) else math.nan,
        "shadow_mae_improve_vs_cloudy_pct": 100.0 * (shadow_base - shadow_pred) / shadow_base if math.isfinite(shadow_base) and shadow_base > 1.0e-12 and math.isfinite(shadow_pred) else math.nan,
    }
    target_values = [safe_float(row.get("target_degraded_ratio")) for row in sample_rows]
    target_values = [v for v in target_values if math.isfinite(v)]
    summary["target_degradation_audit"] = {
        "max": max(target_values) if target_values else math.nan,
        "mean": sum(target_values) / len(target_values) if target_values else math.nan,
        "nonzero_count": sum(v > 0 for v in target_values),
        "gt_0_01_count": sum(v > 0.01 for v in target_values),
    }

    sample_csv = out_dir / f"{args.split}_branch_metrics_per_sample.csv"
    if sample_rows:
        fields = sorted({key for row in sample_rows for key in row})
        with sample_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sample_rows)
    (out_dir / f"{args.split}_branch_metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "metrics_summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "wavelet_summary.json").write_text(json.dumps(json_safe(wavelet_summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "sar_counterfactual_summary.json").write_text(json.dumps(json_safe(sar_summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
