"""Stage1 trainer for ALLClear cloud/shadow removal."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.allclear.config import load_config, make_run_dir, save_config
from src.allclear.dataset import AllClearDataset, cloud_fraction
from src.allclear.losses import (
    AllClearStageLoss,
    CloudOnlyRestorationLoss,
    CloudDiscriminatorLoss,
    HingeDiscriminatorLoss,
    LossWeights,
    R1DiscriminatorLoss,
)
from src.allclear.model import AllClearTGDADSoftShadow, DADIGANBaseline, SoftShadowDADIGANBaseline
from src.allclear.modules.dadigan import make_cloud_discriminator
from src.allclear.modules.pix2pixhd import make_pix2pixhd_nlayer_discriminator
from src.allclear.modules.sn_patchgan import make_sn_patchgan_discriminator

logger = logging.getLogger("allclear.train")


def distributed_env() -> tuple[bool, int, int, int]:
    """Return (enabled, rank, local_rank, world_size) for torchrun-style DDP."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def is_main_process(rank: int = 0) -> bool:
    return rank == 0


def maybe_barrier(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, stripping DataParallel wrapper if present."""
    return getattr(m, "module", m)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
    return out


def model_band_indices_from_cfg(cfg: dict[str, Any]) -> tuple[int, ...] | None:
    """Return S2 band indices used by the model, or None for all bands."""

    values = cfg.get("data", {}).get("model_band_indices")
    if values is None:
        return None
    indices = tuple(int(v) for v in values)
    if not indices:
        raise ValueError("data.model_band_indices must be omitted or contain at least one band index.")
    if min(indices) < 0:
        raise ValueError("data.model_band_indices must use non-negative band indices.")
    return indices


def model_reflectance_range_from_cfg(cfg: dict[str, Any]) -> tuple[float, float] | None:
    """Return optional model-domain reflectance stretch range.

    This is intentionally separate from ``data.optical_scale``.  The dataset
    first loads physical reflectance, then model-level RGB ablations may map the
    selected channels to a LaMa-style visual domain such as [0, 0.35] -> [0, 1].
    """

    data = cfg.get("data", {})
    max_value = data.get("model_reflectance_max", data.get("model_rgb_reflectance_max"))
    if max_value is None:
        return None
    min_value = data.get("model_reflectance_min", data.get("model_rgb_reflectance_min", 0.0))
    lo = float(min_value)
    hi = float(max_value)
    if not hi > lo:
        raise ValueError("data.model_reflectance_max must be greater than data.model_reflectance_min.")
    return lo, hi


def apply_model_band_indices(
    batch: dict[str, Any],
    indices: tuple[int, ...] | None,
    reflectance_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Slice and optionally stretch S2 input/target channels for ablations.

    ``cld_shdw`` and SAR are deliberately left untouched: the cloud/shadow
    labels and radar auxiliary modality keep their original semantics.
    """

    if indices is None and reflectance_range is None:
        return batch
    if "s2_toa" not in batch:
        raise KeyError("Batch is missing required key 's2_toa'.")
    out = dict(batch)

    index = None
    max_index = None
    if indices is not None:
        index = torch.as_tensor(indices, device=batch["s2_toa"].device, dtype=torch.long)
        max_index = int(index.max().item())

    for key in ("s2_toa", "target"):
        value = out.get(key)
        if not isinstance(value, Tensor):
            continue
        if (indices is not None or reflectance_range is not None) and value.ndim < 2:
            raise ValueError(f"Batch key {key!r} must have channel dimension before applying model_band_indices.")
        if index is not None and max_index is not None:
            if value.shape[1] <= max_index:
                raise ValueError(
                    f"data.model_band_indices={list(indices or ())} cannot be applied to {key!r} "
                    f"with {value.shape[1]} channels."
                )
            value = value.index_select(1, index)
        if reflectance_range is not None:
            lo, hi = reflectance_range
            value = ((value.float() - lo) / (hi - lo)).clamp(0.0, 1.0)
        out[key] = value
    return out


def amp_dtype_from_cfg(name: str) -> torch.dtype:
    if str(name).lower() == "bf16":
        return torch.bfloat16
    if str(name).lower() in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError("train.amp_dtype must be 'fp16' or 'bf16'")


def autocast_context(enabled: bool, dtype: torch.dtype):
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def configure_torch_backend(train_cfg: dict[str, Any]) -> None:
    """Apply backend knobs commonly used in PyTorch training recipes."""

    if bool(train_cfg.get("allow_tf32", False)) and torch.cuda.is_available():
        precision = str(train_cfg.get("matmul_precision", "high"))
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))


def make_optimizer(params, *, lr: float, weight_decay: float, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the requested optimizer.

    DADIGAN reports Adam in the paper; AdamW remains the default for older
    ALLClear configs unless ``train.optimizer: adam`` is explicitly set.
    """

    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name != "adamw":
        raise ValueError("train.optimizer must be one of: adamw, adam")
    if bool(train_cfg.get("optimizer_fused", False)) and torch.cuda.is_available():
        kwargs["fused"] = True
    elif bool(train_cfg.get("optimizer_foreach", False)):
        kwargs["foreach"] = True
    try:
        return torch.optim.AdamW(params, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        kwargs.pop("foreach", None)
        return torch.optim.AdamW(params, **kwargs)


def maybe_compile_submodules(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    """Optionally compile stable submodules, following PyTorch torch.compile usage."""

    train_cfg = cfg.get("train", {})
    if not bool(train_cfg.get("torch_compile", False)):
        return
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile requested, but this PyTorch build does not expose torch.compile.")
        return
    mode = str(train_cfg.get("torch_compile_mode", "reduce-overhead"))
    backend = str(train_cfg.get("torch_compile_backend", "inductor"))
    targets = train_cfg.get("torch_compile_modules", ["shadow_removal"])
    base = _unwrap(model)
    for target in targets:
        try:
            if target == "shadow_removal":
                if not hasattr(base, "shadow_branch"):
                    logger.warning("Skipping torch_compile target shadow_removal: model has no shadow_branch.")
                    continue
                base.shadow_branch.removal = torch.compile(base.shadow_branch.removal, mode=mode, backend=backend)
            elif target == "cloud_branch":
                base.cloud_branch = torch.compile(base.cloud_branch, mode=mode, backend=backend)
            else:
                logger.warning("Unknown torch_compile module target: %s", target)
                continue
            logger.info("torch.compile enabled for %s (mode=%s backend=%s)", target, mode, backend)
        except Exception as exc:
            logger.warning("Skipping torch.compile for %s: %s", target, exc)


def weights_from_cfg(cfg: dict[str, Any]) -> LossWeights:
    loss = cfg.get("loss", {})
    defaults = LossWeights()
    weights = LossWeights(
        final_l1=float(loss.get("final_l1", defaults.final_l1)),
        grad=float(loss.get("grad", defaults.grad)),
        shadow_removal=float(loss.get("shadow_removal", defaults.shadow_removal)),
        shadow_mask=float(loss.get("shadow_mask", defaults.shadow_mask)),
        shadow_penumbra=float(loss.get("shadow_penumbra", defaults.shadow_penumbra)),
        cloud_l1=float(loss.get("cloud_l1", defaults.cloud_l1)),
        cloud_l1_missing=float(loss.get("cloud_l1_missing", defaults.cloud_l1_missing)),
        cloud_l1_known=float(loss.get("cloud_l1_known", defaults.cloud_l1_known)),
        cloud_kl=float(loss.get("cloud_kl", defaults.cloud_kl)),
        cloud_adv=float(loss.get("cloud_adv", defaults.cloud_adv)),
        feature_matching=float(loss.get("feature_matching", defaults.feature_matching)),
        perceptual=float(loss.get("perceptual", defaults.perceptual)),
    )
    profile = str(loss.get("profile", "")).lower()
    if profile in {"cloud_only", "dadigan_lama_ffc", "dadigan_baseline"}:
        weights.final_l1 = 0.0
        weights.grad = 0.0
        weights.shadow_removal = 0.0
        weights.shadow_mask = 0.0
        weights.shadow_penumbra = 0.0
    return weights


LOSS_WEIGHT_FIELDS = tuple(LossWeights.__dataclass_fields__.keys())


def _scheduled_scale(cfg: dict[str, Any], name: str, epoch: int) -> float:
    schedule = cfg.get("loss_schedule", {}).get(name, {})
    start = int(schedule.get("start_epoch", 1))
    ramp = int(schedule.get("ramp_epochs", 0))
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch - start + 1) / float(ramp)))


def scheduled_weights_from_cfg(cfg: dict[str, Any], epoch: int) -> LossWeights:
    base = weights_from_cfg(cfg)
    values = {}
    for name in LOSS_WEIGHT_FIELDS:
        values[name] = float(getattr(base, name)) * _scheduled_scale(cfg, name, epoch)
    return LossWeights(**values)


def weights_log_dict(weights: LossWeights) -> dict[str, float]:
    return {f"w_{name}": float(getattr(weights, name)) for name in LOSS_WEIGHT_FIELDS}


def split_data_value(data: dict[str, Any], key: str, split: str) -> Any:
    return data.get(f"{key}_{split}", data.get(key))


def optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)


def optional_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    return (float(value),)


def optional_bool(value: Any) -> bool | None:
    """Parse a nullable compatibility switch without turning null into false."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError("nullable boolean config values must be true, false, or null")


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    framework = str(model_cfg.get("framework", "stage1")).lower()
    if framework in {"softshadow_dadigan_baseline", "softshadow_dadigan", "sam_dadigan"}:
        return SoftShadowDADIGANBaseline(
            s2_channels=int(model_cfg.get("s2_channels", 13)),
            sar_channels=int(model_cfg.get("sar_channels", 2)),
            dim=int(model_cfg.get("dim", 64)),
            shadow_index=int(cfg.get("data", {}).get("shadow_index", 3)),
            cloud_index=int(cfg.get("data", {}).get("cloud_index", 1)),
            rgb_indices=tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1])),
            softshadow_repo=model_cfg.get("softshadow_repo"),
            sam_checkpoint=model_cfg.get("sam_checkpoint"),
            softshadow_checkpoint=model_cfg.get("softshadow_checkpoint"),
            softshadow_sam_model_type=str(model_cfg.get("softshadow_sam_model_type", "vit_h")),
            softshadow_sam_lora_rank=int(model_cfg.get("softshadow_sam_lora_rank", 8)),
            softshadow_sam_lora_layers=model_cfg.get("softshadow_sam_lora_layers"),
            softshadow_sam_input_size=int(model_cfg.get("softshadow_sam_input_size", 1024)),
            softshadow_sam_checkpoint_blocks=bool(model_cfg.get("softshadow_sam_checkpoint_blocks", True)),
            softshadow_bbox_space=str(model_cfg.get("softshadow_bbox_space", "image")),
            softshadow_use_hard_support_gate=bool(model_cfg.get("softshadow_use_hard_support_gate", True)),
            softshadow_forward_valid_only=bool(model_cfg.get("softshadow_forward_valid_only", False)),
            restore_mask_mode=str(model_cfg.get("restore_mask_mode", "cloud_plus_soft_shadow")),
            cloud_ddin_steps=int(model_cfg.get("cloud_ddin_steps", 3)),
            cloud_prox_blocks=int(model_cfg.get("cloud_prox_blocks", 5)),
            cloud_reconstruct_blocks=int(model_cfg.get("cloud_reconstruct_blocks", 2)),
            cloud_cab_sr_ratio=int(model_cfg.get("cloud_cab_sr_ratio", 8)),
            cloud_cab_attention_mode=str(model_cfg.get("cloud_cab_attention_mode", "standard")),
            cloud_msab_mode=str(model_cfg.get("cloud_msab_mode", "restormer_mdta")),
            cloud_mask_input_mode=str(model_cfg.get("cloud_mask_input_mode", "learned")),
            cloud_append_mask=bool(model_cfg.get("cloud_append_mask", True)),
            cloud_mask_fill_value=float(model_cfg.get("cloud_mask_fill_value", 0.0)),
            cloud_output_activation=str(model_cfg.get("cloud_output_activation", "none")),
        )
    if framework in {"dadigan_baseline", "strict_dadigan", "dadigan"}:
        return DADIGANBaseline(
            s2_channels=int(model_cfg.get("s2_channels", 13)),
            sar_channels=int(model_cfg.get("sar_channels", 2)),
            dim=int(model_cfg.get("dim", 64)),
            shadow_index=int(cfg.get("data", {}).get("shadow_index", 3)),
            cloud_index=int(cfg.get("data", {}).get("cloud_index", 1)),
            cloud_ddin_steps=int(model_cfg.get("cloud_ddin_steps", 3)),
            cloud_prox_blocks=int(model_cfg.get("cloud_prox_blocks", 2)),
            cloud_reconstruct_blocks=int(model_cfg.get("cloud_reconstruct_blocks", 2)),
            cloud_bottleneck_context=str(model_cfg.get("cloud_bottleneck_context", "none")),
            cloud_pre_pda_context=str(model_cfg.get("cloud_pre_pda_context", "none")),
            cloud_pre_pda_ffc_blocks=int(model_cfg.get("cloud_pre_pda_ffc_blocks", 0)),
            cloud_pre_pda_ffc_ratio=float(model_cfg.get("cloud_pre_pda_ffc_ratio", model_cfg.get("cloud_ffc_ratio", 0.75))),
            cloud_pre_pda_ffc_enable_lfu=bool(model_cfg.get("cloud_pre_pda_ffc_enable_lfu", model_cfg.get("cloud_ffc_enable_lfu", False))),
            cloud_pre_pda_ffc_downsample=int(model_cfg.get("cloud_pre_pda_ffc_downsample", 4)),
            cloud_pre_pda_ffc_residual_scale=float(model_cfg.get("cloud_pre_pda_ffc_residual_scale", 0.05)),
            cloud_prefusion_context=str(model_cfg.get("cloud_prefusion_context", "none")),
            cloud_prefusion_blocks=int(model_cfg.get("cloud_prefusion_blocks", 0)),
            cloud_prefusion_kernel_size=int(model_cfg.get("cloud_prefusion_kernel_size", 5)),
            cloud_prefusion_reduction=int(model_cfg.get("cloud_prefusion_reduction", 16)),
            cloud_lowres_glfcr_coupled=bool(model_cfg.get("cloud_lowres_glfcr_coupled", False)),
            cloud_lowres_enabled=optional_bool(model_cfg.get("cloud_lowres_enabled")),
            cloud_ddin_glfcr_coupled=optional_bool(model_cfg.get("cloud_ddin_glfcr_coupled")),
            cloud_lowres_factor=int(model_cfg.get("cloud_lowres_factor", 2)),
            cloud_lowres_opt_ffc_blocks=int(model_cfg.get("cloud_lowres_opt_ffc_blocks", 0)),
            cloud_lowres_opt_ffc_ratio=float(model_cfg.get("cloud_lowres_opt_ffc_ratio", model_cfg.get("cloud_ffc_ratio", 0.75))),
            cloud_lowres_opt_ffc_enable_lfu=bool(model_cfg.get("cloud_lowres_opt_ffc_enable_lfu", model_cfg.get("cloud_ffc_enable_lfu", False))),
            cloud_lowres_opt_ffc_spatial_transform_layers=optional_int_tuple(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_layers")),
            cloud_lowres_opt_ffc_spatial_transform_pad_coef=float(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_pad_coef", 0.5)),
            cloud_lowres_opt_ffc_spatial_transform_angle_init_range=float(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_angle_init_range", 80.0)),
            cloud_lowres_opt_ffc_spatial_transform_train_angle=bool(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_train_angle", True)),
            cloud_lowres_glfcr_kernel_size=int(model_cfg.get("cloud_lowres_glfcr_kernel_size", 5)),
            cloud_cab_sr_ratio=int(model_cfg.get("cloud_cab_sr_ratio", 8)),
            cloud_cab_attention_mode=str(model_cfg.get("cloud_cab_attention_mode", "standard")),
            cloud_msab_mode=str(model_cfg.get("cloud_msab_mode", "efficient")),
            cloud_cab2_residual_source=str(model_cfg.get("cloud_cab2_residual_source", "query")),
            cloud_cab2_update_scale=float(model_cfg.get("cloud_cab2_update_scale", 1.0)),
            cloud_post_ddin_sar_filter=str(model_cfg.get("cloud_post_ddin_sar_filter", "none")),
            cloud_post_ddin_sar_filter_kernel_size=(
                int(model_cfg["cloud_post_ddin_sar_filter_kernel_size"])
                if model_cfg.get("cloud_post_ddin_sar_filter_kernel_size") is not None
                else None
            ),
            cloud_ffc_blocks=int(model_cfg.get("cloud_ffc_blocks", 0)),
            cloud_ffc_blocks_per_scale=optional_int_tuple(model_cfg.get("cloud_ffc_blocks_per_scale")),
            cloud_ffc_ratio=float(model_cfg.get("cloud_ffc_ratio", 0.75)),
            cloud_ffc_enable_lfu=bool(model_cfg.get("cloud_ffc_enable_lfu", False)),
            cloud_ffc_downsample=int(model_cfg.get("cloud_ffc_downsample", 1)),
            cloud_ffc_downsamples=optional_int_tuple(model_cfg.get("cloud_ffc_downsamples")),
            cloud_ffc_residual_scale=float(model_cfg.get("cloud_ffc_residual_scale", 0.1)),
            cloud_ffc_residual_scales=optional_float_tuple(model_cfg.get("cloud_ffc_residual_scales")),
            cloud_ffc_spatial_transform_layers=optional_int_tuple(model_cfg.get("cloud_ffc_spatial_transform_layers")),
            cloud_ffc_spatial_transform_pad_coef=float(model_cfg.get("cloud_ffc_spatial_transform_pad_coef", 0.5)),
            cloud_ffc_spatial_transform_angle_init_range=float(model_cfg.get("cloud_ffc_spatial_transform_angle_init_range", 80.0)),
            cloud_ffc_spatial_transform_train_angle=bool(model_cfg.get("cloud_ffc_spatial_transform_train_angle", True)),
            cloud_mask_input_mode=str(model_cfg.get("cloud_mask_input_mode", "raw")),
            cloud_append_mask=bool(model_cfg.get("cloud_append_mask", False)),
            cloud_mask_fill_value=float(model_cfg.get("cloud_mask_fill_value", 0.0)),
            cloud_output_activation=str(model_cfg.get("cloud_output_activation", "none")),
            baseline_mask_mode=str(model_cfg.get("baseline_mask_mode", "full")),
            baseline_output_mode=str(model_cfg.get("baseline_output_mode", "raw")),
        )
    return AllClearTGDADSoftShadow(
        s2_channels=int(model_cfg.get("s2_channels", 13)),
        sar_channels=int(model_cfg.get("sar_channels", 2)),
        dim=int(model_cfg.get("dim", 48)),
        shadow_backend=str(model_cfg.get("shadow_backend", "conv")),
        shadow_removal_backend=model_cfg.get("shadow_removal_backend"),
        shadow_hidden_channels=model_cfg.get("shadow_hidden_channels"),
        shadow_restormer_hidden_channels=int(model_cfg.get("shadow_restormer_hidden_channels", 64)),
        shadow_restormer_blocks=int(model_cfg.get("shadow_restormer_blocks", 2)),
        shadow_restormer_heads=int(model_cfg.get("shadow_restormer_heads", 4)),
        shadow_nafnet_hidden_channels=model_cfg.get("shadow_nafnet_hidden_channels"),
        shadow_nafnet_blocks=int(model_cfg.get("shadow_nafnet_blocks", 3)),
        softshadow_repo=model_cfg.get("softshadow_repo"),
        sam_checkpoint=model_cfg.get("sam_checkpoint"),
        softshadow_checkpoint=model_cfg.get("softshadow_checkpoint"),
        softshadow_sam_model_type=str(model_cfg.get("softshadow_sam_model_type", "vit_h")),
        softshadow_sam_lora_rank=int(model_cfg.get("softshadow_sam_lora_rank", 8)),
        softshadow_sam_lora_layers=model_cfg.get("softshadow_sam_lora_layers"),
        softshadow_sam_input_size=int(model_cfg.get("softshadow_sam_input_size", 1024)),
        softshadow_sam_checkpoint_blocks=bool(model_cfg.get("softshadow_sam_checkpoint_blocks", False)),
        softshadow_bbox_space=str(model_cfg.get("softshadow_bbox_space", "image")),
        softshadow_efficientvit_repo=model_cfg.get("softshadow_efficientvit_repo"),
        softshadow_efficientvit_checkpoint=model_cfg.get("softshadow_efficientvit_checkpoint"),
        softshadow_efficientvit_model=str(model_cfg.get("softshadow_efficientvit_model", "efficientvit-sam-xl0")),
        softshadow_efficientvit_adapter_rank=int(model_cfg.get("softshadow_efficientvit_adapter_rank", 8)),
        softshadow_efficientvit_adapter_layers=model_cfg.get("softshadow_efficientvit_adapter_layers"),
        softshadow_efficientvit_train_mask_decoder=bool(model_cfg.get("softshadow_efficientvit_train_mask_decoder", True)),
        softshadow_efficientvit_force_fp32=bool(model_cfg.get("softshadow_efficientvit_force_fp32", False)),
        softshadow_use_hard_support_gate=bool(model_cfg.get("softshadow_use_hard_support_gate", True)),
        softshadow_forward_valid_only=bool(model_cfg.get("softshadow_forward_valid_only", False)),
        rgb_indices=tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1])),
        shadow_index=int(model_cfg.get("shadow_index", 3)),
        cloud_index=int(model_cfg.get("cloud_index", 1)),
        cloud_bottleneck_context=str(model_cfg.get("cloud_bottleneck_context", "none")),
        cloud_pre_pda_context=str(model_cfg.get("cloud_pre_pda_context", "none")),
        cloud_pre_pda_ffc_blocks=int(model_cfg.get("cloud_pre_pda_ffc_blocks", 0)),
        cloud_pre_pda_ffc_ratio=float(model_cfg.get("cloud_pre_pda_ffc_ratio", model_cfg.get("cloud_ffc_ratio", 0.75))),
        cloud_pre_pda_ffc_enable_lfu=bool(model_cfg.get("cloud_pre_pda_ffc_enable_lfu", model_cfg.get("cloud_ffc_enable_lfu", False))),
        cloud_pre_pda_ffc_downsample=int(model_cfg.get("cloud_pre_pda_ffc_downsample", 4)),
        cloud_pre_pda_ffc_residual_scale=float(model_cfg.get("cloud_pre_pda_ffc_residual_scale", 0.05)),
        cloud_prefusion_context=str(model_cfg.get("cloud_prefusion_context", "none")),
        cloud_prefusion_blocks=int(model_cfg.get("cloud_prefusion_blocks", 0)),
        cloud_prefusion_kernel_size=int(model_cfg.get("cloud_prefusion_kernel_size", 5)),
        cloud_prefusion_reduction=int(model_cfg.get("cloud_prefusion_reduction", 16)),
        cloud_lowres_glfcr_coupled=bool(model_cfg.get("cloud_lowres_glfcr_coupled", False)),
        cloud_lowres_enabled=optional_bool(model_cfg.get("cloud_lowres_enabled")),
        cloud_ddin_glfcr_coupled=optional_bool(model_cfg.get("cloud_ddin_glfcr_coupled")),
        cloud_lowres_factor=int(model_cfg.get("cloud_lowres_factor", 2)),
        cloud_lowres_opt_ffc_blocks=int(model_cfg.get("cloud_lowres_opt_ffc_blocks", 0)),
        cloud_lowres_opt_ffc_ratio=float(model_cfg.get("cloud_lowres_opt_ffc_ratio", model_cfg.get("cloud_ffc_ratio", 0.75))),
        cloud_lowres_opt_ffc_enable_lfu=bool(model_cfg.get("cloud_lowres_opt_ffc_enable_lfu", model_cfg.get("cloud_ffc_enable_lfu", False))),
        cloud_lowres_opt_ffc_spatial_transform_layers=optional_int_tuple(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_layers")),
        cloud_lowres_opt_ffc_spatial_transform_pad_coef=float(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_pad_coef", 0.5)),
        cloud_lowres_opt_ffc_spatial_transform_angle_init_range=float(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_angle_init_range", 80.0)),
        cloud_lowres_opt_ffc_spatial_transform_train_angle=bool(model_cfg.get("cloud_lowres_opt_ffc_spatial_transform_train_angle", True)),
        cloud_lowres_glfcr_kernel_size=int(model_cfg.get("cloud_lowres_glfcr_kernel_size", 5)),
        cloud_backend=str(model_cfg.get("cloud_backend", "dadigan")),
        cloud_ddin_steps=int(model_cfg.get("cloud_ddin_steps", 3)),
        cloud_prox_blocks=int(model_cfg.get("cloud_prox_blocks", 2)),
        cloud_reconstruct_blocks=int(model_cfg.get("cloud_reconstruct_blocks", 2)),
        cloud_cab_sr_ratio=int(model_cfg.get("cloud_cab_sr_ratio", 8)),
        cloud_cab_attention_mode=str(model_cfg.get("cloud_cab_attention_mode", "standard")),
        cloud_msab_mode=str(model_cfg.get("cloud_msab_mode", "efficient")),
        cloud_cab2_residual_source=str(model_cfg.get("cloud_cab2_residual_source", "query")),
        cloud_cab2_update_scale=float(model_cfg.get("cloud_cab2_update_scale", 1.0)),
        cloud_post_ddin_sar_filter=str(model_cfg.get("cloud_post_ddin_sar_filter", "none")),
        cloud_post_ddin_sar_filter_kernel_size=(
            int(model_cfg["cloud_post_ddin_sar_filter_kernel_size"])
            if model_cfg.get("cloud_post_ddin_sar_filter_kernel_size") is not None
            else None
        ),
        cloud_ffc_blocks=int(model_cfg.get("cloud_ffc_blocks", 0)),
        cloud_mask_input_mode=str(model_cfg.get("cloud_mask_input_mode", "raw")),
        cloud_append_mask=bool(model_cfg.get("cloud_append_mask", False)),
        cloud_mask_fill_value=float(model_cfg.get("cloud_mask_fill_value", 0.0)),
        cloud_output_activation=str(model_cfg.get("cloud_output_activation", "none")),
        cloud_ffc_ratio=float(model_cfg.get("cloud_ffc_ratio", 0.75)),
        cloud_ffc_enable_lfu=bool(model_cfg.get("cloud_ffc_enable_lfu", False)),
        cloud_ffc_downsample=int(model_cfg.get("cloud_ffc_downsample", 1)),
        cloud_ffc_residual_scale=float(model_cfg.get("cloud_ffc_residual_scale", 0.1)),
        cloud_ffc_spatial_transform_layers=optional_int_tuple(model_cfg.get("cloud_ffc_spatial_transform_layers")),
        cloud_ffc_spatial_transform_pad_coef=float(model_cfg.get("cloud_ffc_spatial_transform_pad_coef", 0.5)),
        cloud_ffc_spatial_transform_angle_init_range=float(model_cfg.get("cloud_ffc_spatial_transform_angle_init_range", 80.0)),
        cloud_ffc_spatial_transform_train_angle=bool(model_cfg.get("cloud_ffc_spatial_transform_train_angle", True)),
        cloud_lama_ngf=int(model_cfg.get("cloud_lama_ngf", 64)),
        cloud_lama_downs=int(model_cfg.get("cloud_lama_downs", 3)),
        cloud_lama_blocks=int(model_cfg.get("cloud_lama_blocks", 9)),
        cloud_lama_pretrained=model_cfg.get("cloud_lama_pretrained"),
        cloud_lama_use_sar=bool(model_cfg.get("cloud_lama_use_sar", False)),
        cloud_lama_mask_input=bool(model_cfg.get("cloud_lama_mask_input", True)),
        cloud_lama_enable_lfu=bool(model_cfg.get("cloud_lama_enable_lfu", False)),
    )


def make_loader(cfg: dict[str, Any], split: str, *, distributed: bool = False, rank: int = 0, world_size: int = 1) -> DataLoader:
    data = cfg["data"]
    train = cfg.get("train", {})
    dataset = AllClearDataset(
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
        softshadow_mask_dir=split_data_value(data, "softshadow_mask_dir", split),
        softshadow_bbox_path=split_data_value(data, "softshadow_bbox_path", split),
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
    eval_cfg = cfg.get("eval", {})
    if split == "train":
        batch_size = int(train.get("batch_size", 2))
        num_workers = int(train.get("num_workers", 4))
        pin_memory = bool(train.get("pin_memory", True))
        persistent_workers = bool(train.get("persistent_workers", True))
        prefetch_factor = int(train.get("prefetch_factor", 2))
    else:
        batch_size = int(eval_cfg.get("batch_size", train.get("val_batch_size", 1)))
        num_workers = int(eval_cfg.get("num_workers", train.get("num_workers", 4)))
        pin_memory = bool(eval_cfg.get("pin_memory", train.get("pin_memory", True)))
        persistent_workers = bool(eval_cfg.get("persistent_workers", train.get("persistent_workers", True)))
        prefetch_factor = int(eval_cfg.get("prefetch_factor", train.get("prefetch_factor", 2)))
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=split == "train",
            drop_last=split == "train",
        )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": (split == "train" and sampler is None),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": split == "train",
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if discriminator is not None:
        payload["discriminator"] = _unwrap(discriminator).state_dict()
    if disc_optimizer is not None:
        payload["disc_optimizer"] = disc_optimizer.state_dict()
    return payload


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model_keys = _unwrap(model).load_state_dict(ckpt["model"], strict=False)
    if model_keys.missing_keys or model_keys.unexpected_keys:
        logger.warning(
            "Checkpoint model keys are not a strict match: missing=%d unexpected=%d. "
            "This is expected only for intentional architecture changes.",
            len(model_keys.missing_keys),
            len(model_keys.unexpected_keys),
        )
        if model_keys.missing_keys:
            logger.warning("First missing model keys: %s", ", ".join(model_keys.missing_keys[:12]))
        if model_keys.unexpected_keys:
            logger.warning("First unexpected model keys: %s", ", ".join(model_keys.unexpected_keys[:12]))
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if discriminator is not None and "discriminator" in ckpt:
        disc_keys = _unwrap(discriminator).load_state_dict(ckpt["discriminator"], strict=False)
        if disc_keys.missing_keys or disc_keys.unexpected_keys:
            logger.warning(
                "Checkpoint discriminator keys are not a strict match: missing=%d unexpected=%d.",
                len(disc_keys.missing_keys),
                len(disc_keys.unexpected_keys),
            )
    if disc_optimizer is not None and "disc_optimizer" in ckpt:
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_metric", float("inf")))


def set_requires_grad(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = trainable


def no_sync_if_needed(module: torch.nn.Module | None, enabled: bool):
    if enabled and module is not None and hasattr(module, "no_sync"):
        return module.no_sync()
    return nullcontext()


def cloud_composite_real(cloudy: Tensor, target: Tensor, cloud_mask: Tensor) -> Tensor:
    mask = cloud_mask.float().clamp(0.0, 1.0)
    return (1.0 - mask) * cloudy.float() + mask * target.float()


def cloud_gan_pair(outputs: dict[str, Tensor], batch: dict[str, Tensor], mode: str) -> tuple[Tensor, Tensor]:
    """Return real/fake images for the cloudy branch discriminator."""

    mode = str(mode).lower()
    cloud_mask = outputs["M_cloud"].float()
    if mode in {"raw", "cloud_raw", "predicted", "predicted_image", "lama_raw"}:
        return batch["target"].float(), outputs.get("I_cloud_raw", outputs["I_cloud"])
    if mode in {"composite", "cloud_context", "inpainted"}:
        real = cloud_composite_real(batch["s2_toa"], batch["target"], cloud_mask)
        return real, outputs["I_cloud"]
    raise ValueError("loss.cloud_gan_image must be one of: raw, composite")


def discriminator_input(cloudy: Tensor, cloud_mask: Tensor, image: Tensor, *, condition_mask: bool = True) -> Tensor:
    parts = [cloudy.float()]
    if condition_mask:
        parts.append(cloud_mask.float().clamp(0.0, 1.0))
    parts.append(image.float())
    return torch.cat(parts, dim=1)


def _call_discriminator(
    discriminator: torch.nn.Module,
    cloudy: Tensor,
    cloud_mask: Tensor,
    image: Tensor,
    *,
    return_features: bool = False,
    condition_mask: bool = True,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Call discriminator.  When ``return_features=True`` and the discriminator
    supports feature extraction, returns ``(scores, features)`` so the caller can
    compute a feature-matching loss.  Otherwise returns only the scores tensor.
    """
    from src.allclear.modules.pix2pixhd import NLayerDiscriminator
    from src.allclear.modules.sn_patchgan import SNPatchGANDiscriminator
    base_disc = _unwrap(discriminator)
    if isinstance(base_disc, SNPatchGANDiscriminator):
        scores, feats = discriminator(image, cloud_mask)
        return (scores, feats) if return_features else scores
    if isinstance(base_disc, NLayerDiscriminator):
        scores, feats = discriminator(image)
        return (scores, feats) if return_features else scores
    scores = discriminator(discriminator_input(cloudy, cloud_mask, image, condition_mask=condition_mask))
    return (scores, []) if return_features else scores


def to_rgb(x: Tensor, rgb_indices: tuple[int, int, int]) -> Tensor:
    rgb = x[:, list(rgb_indices)].detach().float().clamp(0, 1).cpu()
    return rgb


def _rgb_panel(x: Tensor, rgb_indices: tuple[int, int, int]) -> Tensor:
    return x[:, list(rgb_indices)].detach().float().cpu()[0]


def _display_limits(panel: Tensor, low_q: float = 0.01, high_q: float = 0.995) -> tuple[float, float]:
    values = panel.flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0, 1.0
    lo = float(torch.quantile(values, low_q).item())
    hi = float(torch.quantile(values, high_q).item())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1.0e-6:
        return 0.0, 1.0
    return lo, hi


def _shared_rgb_limits(panels: list[Tensor], low_q: float = 0.01, high_q: float = 0.995) -> tuple[float, float]:
    values = torch.cat([panel.detach().float().flatten() for panel in panels])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0, 1.0
    lo = float(torch.quantile(values, low_q).item())
    hi = float(torch.quantile(values, high_q).item())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1.0e-6:
        return 0.0, 1.0
    return lo, hi


def _stretch_rgb(panel: Tensor, lo: float, hi: float, gamma: float = 0.85, gain: float = 1.0) -> Tensor:
    image = ((panel - lo) / max(hi - lo, 1.0e-6)).clamp(0.0, 1.0)
    if gamma > 0 and abs(gamma - 1.0) > 1.0e-6:
        image = image.pow(gamma)
    if abs(gain - 1.0) > 1.0e-6:
        image = image * gain
    return image.clamp(0.0, 1.0)


def _visual_rgb_params(
    visual_profile: str,
    visual_rgb_gamma: float | None = None,
    visual_rgb_gain: float | None = None,
) -> tuple[float, float]:
    profile = str(visual_profile).lower()
    if visual_rgb_gamma is None:
        visual_rgb_gamma = 0.72 if profile == "dadigan_lama_ffc" else 0.85
    if visual_rgb_gain is None:
        visual_rgb_gain = 1.08 if profile == "dadigan_lama_ffc" else 1.0
    return float(visual_rgb_gamma), float(visual_rgb_gain)


def _visual_rgb_stretch_mode(visual_profile: str, visual_rgb_stretch: str | None = None) -> str:
    if visual_rgb_stretch is not None:
        mode = str(visual_rgb_stretch).lower()
    else:
        mode = "panel" if str(visual_profile).lower() == "dadigan_lama_ffc" else "shared_reference"
    if mode not in {"shared_reference", "panel"}:
        raise ValueError(f"Unsupported eval.visual_rgb_stretch: {mode}")
    return mode


def _stretch_rgb_panels(panels: list[Tensor], mode: str, gamma: float, gain: float) -> list[Tensor]:
    if mode == "panel":
        out = []
        for panel in panels:
            lo, hi = _display_limits(panel)
            out.append(_stretch_rgb(panel, lo, hi, gamma=gamma, gain=gain))
        return out
    shared_lo, shared_hi = _shared_rgb_limits(panels[:2])
    return [_stretch_rgb(panel, shared_lo, shared_hi, gamma=gamma, gain=gain) for panel in panels]


def _sar_panel(sar: Tensor) -> Tensor:
    """Convert 2-channel SAR (VV, VH) to a 3-channel pseudo-RGB panel.

    Mapping:  VV → Red,  VH → Green,  (VV·VH)^0.5 → Blue.
    Each channel is independently contrast-stretched to [0,1].
    """
    vv = sar[:, 0:1].detach().float().cpu()       # [B, 1, H, W]
    vh = sar[:, 1:2].detach().float().cpu()       # [B, 1, H, W]
    vv_lo, vv_hi = _display_limits(vv)
    vh_lo, vh_hi = _display_limits(vh)
    gm = (vv * vh).clamp_min(0).sqrt()             # geometric mean
    gm_lo, gm_hi = _display_limits(gm)
    r = _stretch_rgb(vv, vv_lo, vv_hi, gamma=0.85)
    g = _stretch_rgb(vh, vh_lo, vh_hi, gamma=0.85)
    b = _stretch_rgb(gm, gm_lo, gm_hi, gamma=0.85)
    return torch.cat([r, g, b], dim=1)[0]          # [3, H, W]


def _empty_panel_like(panel: Tensor) -> Tensor:
    return panel.new_full((3, panel.shape[-2], panel.shape[-1]), 0.025)


def _optional_sar_panel(batch: dict[str, Any], index: int, ref_panel: Tensor) -> Tensor:
    sar = batch.get("s1")
    if not isinstance(sar, Tensor):
        return _empty_panel_like(ref_panel)
    return _sar_panel(sar[index : index + 1])


def _mask_panel(mask: Tensor, color: tuple[float, float, float], gamma: float = 0.75) -> Tensor:
    m = mask.detach().float().cpu()
    if m.ndim == 3:
        m = m[:1]
    elif m.ndim == 2:
        m = m.unsqueeze(0)
    else:
        m = m.reshape(1, *m.shape[-2:])
    m = m.clamp(0.0, 1.0)
    if gamma > 0 and abs(gamma - 1.0) > 1.0e-6:
        m = m.pow(gamma)
    tint = torch.tensor(color, dtype=m.dtype).view(3, 1, 1)
    return (m * tint + (1.0 - m) * 0.025).clamp(0.0, 1.0)


def _panel_to_image(panel: Tensor):
    from PIL import Image

    array = (panel.clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).round().byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _save_titled_grid(panel_rows: list[list[Tensor]], titles: list[str], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not panel_rows:
        return
    images = [[_panel_to_image(panel) for panel in row] for row in panel_rows]
    tile_w, tile_h = images[0][0].size
    pad = 6
    title_h = 26
    cols = len(titles)
    rows = len(images)
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (tile_h + title_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 16, 22))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:  # pragma: no cover - font availability varies by system
        font = ImageFont.load_default()

    for row_idx, row in enumerate(images):
        y = pad + row_idx * (tile_h + title_h + pad)
        for col_idx, image in enumerate(row):
            if image.size != (tile_w, tile_h):
                image = image.resize((tile_w, tile_h), Image.BILINEAR)
            x = pad + col_idx * (tile_w + pad)
            draw.rectangle((x, y, x + tile_w, y + title_h), fill=(28, 35, 46))
            draw.text((x + 7, y + 6), titles[col_idx], fill=(238, 242, 247), font=font)
            canvas.paste(image, (x, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_visuals_fallback(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    path: Path,
    rgb_indices: tuple[int, int, int],
    max_items: int,
    visual_profile: str = "stage1",
    visual_rgb_gamma: float | None = None,
    visual_rgb_gain: float | None = None,
    visual_rgb_stretch: str | None = None,
) -> None:
    try:
        from torchvision.utils import make_grid, save_image  # type: ignore
    except Exception:  # pragma: no cover - optional dependency path
        return
    rows = []
    n = min(max_items, outputs["I_hat"].shape[0])
    profile = str(visual_profile).lower()
    cloud_only = profile in {"cloud_only", "dadigan_lama_ffc", "dadigan_baseline"}
    rgb_gamma, rgb_gain = _visual_rgb_params(profile, visual_rgb_gamma, visual_rgb_gain)
    rgb_stretch = _visual_rgb_stretch_mode(profile, visual_rgb_stretch)
    for i in range(n):
        rgb_panels = [
            _rgb_panel(batch["s2_toa"][i : i + 1], rgb_indices),
            _rgb_panel(batch["target"][i : i + 1], rgb_indices),
            _rgb_panel(outputs["I_hat"][i : i + 1], rgb_indices),
        ]
        row = [
            *_stretch_rgb_panels(rgb_panels, rgb_stretch, rgb_gamma, rgb_gain),
            _optional_sar_panel(batch, i, batch["s2_toa"][i]),
            outputs["M_cloud"][i].repeat(3, 1, 1).detach().cpu(),
            outputs["M_shadow"][i].repeat(3, 1, 1).detach().cpu(),
        ]
        if not cloud_only:
            row.extend(
                [
                    outputs.get("M_shadow_soft_raw", outputs["M_shadow_soft"])[i].repeat(3, 1, 1).detach().cpu(),
                    outputs.get("M_shadow_soft_eff", outputs["M_shadow_soft"])[i].repeat(3, 1, 1).detach().cpu(),
                ]
            )
        rows.extend(row)
    grid = make_grid(rows, nrow=6 if cloud_only else 8, padding=2)
    save_image(grid, path)


def save_visuals(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    path: Path,
    rgb_indices: tuple[int, int, int],
    max_items: int = 5,
    visual_profile: str = "stage1",
    visual_rgb_gamma: float | None = None,
    visual_rgb_gain: float | None = None,
    visual_rgb_stretch: str | None = None,
) -> None:
    try:
        profile = str(visual_profile).lower()
        cloud_only = profile in {"cloud_only", "dadigan_lama_ffc", "dadigan_baseline"}
        rgb_gamma, rgb_gain = _visual_rgb_params(profile, visual_rgb_gamma, visual_rgb_gain)
        rgb_stretch = _visual_rgb_stretch_mode(profile, visual_rgb_stretch)
        mask_title = "Restore Mask" if profile == "dadigan_lama_ffc" else "Cloud Mask"
        titles = ["Cloudy S2", "Target", "Stage1 Output", "SAR", mask_title, "Hard Shadow"]
        if not cloud_only:
            titles.extend(["Soft Raw", "Soft Eff"])
        panel_rows: list[list[Tensor]] = []
        n = min(max_items, outputs["I_hat"].shape[0])
        cloud_mask = outputs.get("M_cloud_vis", outputs["M_cloud"])
        for i in range(n):
            rgb_panels = [
                _rgb_panel(batch["s2_toa"][i : i + 1], rgb_indices),
                _rgb_panel(batch["target"][i : i + 1], rgb_indices),
                _rgb_panel(outputs["I_hat"][i : i + 1], rgb_indices),
            ]
            row = [
                *_stretch_rgb_panels(rgb_panels, rgb_stretch, rgb_gamma, rgb_gain),
                _optional_sar_panel(batch, i, rgb_panels[0]),
                _mask_panel(cloud_mask[i], (0.10, 0.78, 1.00)),
                _mask_panel(outputs["M_shadow"][i], (1.00, 0.62, 0.10)),
            ]
            if not cloud_only:
                row.extend(
                    [
                        _mask_panel(outputs.get("M_shadow_soft_raw", outputs["M_shadow_soft"])[i], (1.00, 0.68, 0.16)),
                        _mask_panel(outputs.get("M_shadow_soft_eff", outputs["M_shadow_soft"])[i], (1.00, 0.68, 0.16)),
                    ]
                )
            panel_rows.append(row)
        _save_titled_grid(panel_rows, titles, path)
    except Exception:
        _save_visuals_fallback(
            outputs,
            batch,
            path,
            rgb_indices,
            max_items,
            visual_profile=visual_profile,
            visual_rgb_gamma=visual_rgb_gamma,
            visual_rgb_gain=visual_rgb_gain,
            visual_rgb_stretch=visual_rgb_stretch,
        )


LOG_FIELDS = [
    "epoch",
    "split",
    "total",
    "recon_total",
    "gan_total",
    "pixel_total",
    "perceptual_total",
    "final_l1",
    "grad",
    "shadow_removal",
    "shadow_mask",
    "shadow_penumbra",
    "shadow_valid_frac",
    "shadow_no_shadow_frac",
    "shadow_ambiguous_frac",
    "cloud_l1",
    "cloud_l1_missing",
    "cloud_l1_known",
    "cloud_kl",
    "cloud_adv",
    "feature_matching",
    "perceptual",
    "disc_total",
    "disc_real_loss",
    "disc_fake_loss",
    "disc_real_gp",
    "disc_real_logit",
    "disc_fake_logit",
    *[f"w_{name}" for name in LOSS_WEIGHT_FIELDS],
]

CLOUD_ONLY_LOG_FIELDS = [
    "epoch",
    "split",
    "total",
    "recon_total",
    "gan_total",
    "pixel_total",
    "perceptual_total",
    "cloud_l1",
    "cloud_l1_missing",
    "cloud_l1_known",
    "cloud_kl",
    "cloud_adv",
    "feature_matching",
    "perceptual",
    "disc_total",
    "disc_real_loss",
    "disc_fake_loss",
    "disc_real_gp",
    "disc_real_logit",
    "disc_fake_logit",
    "w_cloud_l1",
    "w_cloud_l1_missing",
    "w_cloud_l1_known",
    "w_cloud_kl",
    "w_cloud_adv",
    "w_feature_matching",
    "w_perceptual",
]


def log_fields_for_profile(profile: str) -> list[str]:
    profile = str(profile).lower()
    if profile in {"cloud_only", "dadigan_lama_ffc", "dadigan_baseline"}:
        return CLOUD_ONLY_LOG_FIELDS
    return LOG_FIELDS


def append_log(path: Path, row: dict[str, float | int | str], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or LOG_FIELDS
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
        if existing:
            fieldnames = existing
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(terms: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(vals) / max(1, len(vals)) for key, vals in terms.items()}


def progress_postfix(running: dict[str, float]) -> dict[str, str]:
    """Compact tqdm postfix with the restoration losses that affect training."""

    fields = [
        ("loss", "total"),
        ("recon", "recon_total"),
        ("gan", "gan_total"),
        ("l1", "cloud_l1"),
        ("l1m", "cloud_l1_missing"),
        ("l1k", "cloud_l1_known"),
        ("kl", "cloud_kl"),
        ("perc", "perceptual"),
        ("fm", "feature_matching"),
        ("adv", "cloud_adv"),
    ]
    postfix: dict[str, str] = {}
    for label, key in fields:
        if key not in running:
            continue
        value = float(running.get(key, 0.0))
        if label not in {"loss", "recon", "gan"} and abs(value) < 1.0e-12:
            continue
        postfix[label] = f"{value:.4f}"
    return postfix


def reduce_metrics(metrics: dict[str, float], device: torch.device, *, distributed: bool) -> dict[str, float]:
    if not distributed or not metrics:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(value.item()) for key, value in zip(keys, values)}


def assert_finite_terms(loss: Tensor, terms: dict[str, Tensor], batch: dict[str, Any], *, epoch: int, split: str) -> None:
    bad = []
    if not torch.isfinite(loss.detach()).all():
        bad.append("total")
    for key, value in terms.items():
        if torch.is_tensor(value) and not torch.isfinite(value.detach()).all():
            bad.append(key)
    if not bad:
        return
    sample_id = batch.get("sample_id", "<unknown>")
    if isinstance(sample_id, (list, tuple)):
        sample_id = ",".join(str(x) for x in sample_id[:8])
    raise FloatingPointError(
        f"Non-finite loss at epoch={epoch} split={split}: {sorted(set(bad))}; sample_id={sample_id}"
    )


def assert_finite_gradients(module: torch.nn.Module, batch: dict[str, Any], *, epoch: int, split: str) -> None:
    """Fail before optimizer.step() if any trainable gradient is non-finite."""

    for name, param in _unwrap(module).named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if torch.isfinite(grad.detach()).all():
            continue
        sample_id = batch.get("sample_id", "<unknown>")
        if isinstance(sample_id, (list, tuple)):
            sample_id = ",".join(str(x) for x in sample_id[:8])
        raise FloatingPointError(
            f"Non-finite gradient at epoch={epoch} split={split}: parameter={name}; sample_id={sample_id}"
        )


def slice_tensor_batch(items: dict[str, Any], index: int, *, cpu: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items.items():
        if isinstance(value, Tensor):
            if value.ndim == 0:
                continue
            item = value[index : index + 1]
            out[key] = item.detach().cpu() if cpu else item
        elif isinstance(value, (list, tuple)):
            out[key] = value[index] if index < len(value) else value
        else:
            out[key] = value
    return out


def concat_tensor_batches(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    out: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in items))
    for key in keys:
        values = [item[key] for item in items if key in item]
        if values and all(isinstance(value, Tensor) for value in values):
            out[key] = torch.cat(values, dim=0)
        elif values:
            out[key] = values[0]
    return out


def visual_structure_score(batch: dict[str, Any], index: int, rgb_indices: tuple[int, int, int]) -> float:
    """Score a sample for visualization by visible land-cover structure.

    We rank candidates by target RGB gradients while penalizing panels that are
    mostly saturated, too dark, or visually low-detail.  This keeps validation
    grids useful for inspection instead of repeatedly selecting bright snow,
    cloud glare, or near-constant targets that happen to have large gradients.
    """

    source = batch.get("target")
    if not isinstance(source, Tensor):
        source = batch.get("s2_toa")
    if not isinstance(source, Tensor) or source.ndim < 4:
        return 0.0
    rgb = source[index : index + 1, list(rgb_indices)].detach().float().clamp(0.0, 1.0)
    lum = rgb.mean(dim=1, keepdim=True)
    if lum.shape[-1] < 2 or lum.shape[-2] < 2:
        return float(lum.std().cpu().item())
    values = lum.flatten()
    p05 = torch.quantile(values, 0.05)
    p50 = torch.quantile(values, 0.50)
    p95 = torch.quantile(values, 0.95)
    contrast = (p95 - p05).clamp_min(0.0)
    sat_high = (rgb.amax(dim=1, keepdim=True) > 0.98).float().mean()
    sat_low = (rgb.amin(dim=1, keepdim=True) < 0.02).float().mean()
    visible_mid = ((lum > 0.06) & (lum < 0.94)).float().mean()

    # Hard reject panels that are dominated by clipped highlights/shadows; they
    # are bad qualitative probes even when they contain mountain/snow edges.
    if float(sat_high.cpu()) > 0.35 or float(sat_low.cpu()) > 0.55 or float(visible_mid.cpu()) < 0.25:
        return -1.0e6 + float(contrast.cpu().item())

    dx = lum[..., :, 1:] - lum[..., :, :-1]
    dy = lum[..., 1:, :] - lum[..., :-1, :]
    gradient = dx.abs().mean() + dy.abs().mean()
    exposure_penalty = (1.0 - sat_high).clamp(0.05, 1.0) * (1.0 - 0.5 * sat_low).clamp(0.05, 1.0)
    midtone_bonus = visible_mid.clamp(0.05, 1.0)
    # Favor rich but inspectable structure; de-emphasize almost all-white/all-black
    # samples even if their target has strong edges.
    score = (gradient + 0.08 * contrast + 0.03 * lum.std()) * exposure_penalty * midtone_bonus
    if float(p50.cpu()) > 0.82:
        score = score * 0.35
    return float(score.cpu().item())


CLOUD_VISUAL_BUCKETS = ("low", "medium", "high", "heavy")


def cloud_bucket_name(value: float | Tensor) -> str:
    """Bucket cloud fraction for balanced visualizations.

    The dataset curation scripts use heavy=[0.90, 1.01). Keeping heavy separate
    prevents complete/near-complete cloud cases from being hidden inside high.
    """

    val = float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)
    if val < 0.1:
        return "low"
    if val < 0.4:
        return "medium"
    if val < 0.9:
        return "high"
    return "heavy"


def keep_top_visual_candidate(
    bucket: list[tuple[float, dict[str, Tensor], dict[str, Any]]],
    candidate: tuple[float, dict[str, Tensor], dict[str, Any]],
    limit: int,
) -> None:
    """Keep only top-scoring, non-duplicate visual candidates in a per-bucket pool."""

    limit = max(1, int(limit))
    candidate_keys = set(visual_dedup_keys(candidate[2]))
    if candidate_keys:
        for idx, existing in enumerate(bucket):
            if candidate_keys.intersection(visual_dedup_keys(existing[2])):
                if candidate[0] > existing[0]:
                    bucket[idx] = candidate
                return
    if len(bucket) < limit:
        bucket.append(candidate)
        return
    min_idx = min(range(len(bucket)), key=lambda idx: bucket[idx][0])
    if candidate[0] > bucket[min_idx][0]:
        bucket[min_idx] = candidate


def _first_scalar_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""


def visual_dedup_keys(batch: dict[str, Any]) -> tuple[str, ...]:
    """Return visualization identity keys ordered from strict to broad.

    ALLClear often contains multiple cloudy observations for the same clear
    target or ROI.  For visualization, that is redundant; we want different
    targets/ROIs so each grid covers more land-cover types and cloud patterns.
    """

    keys: list[str] = []
    clear_path = _first_scalar_text(batch.get("clear_s2_path"))
    cloudy_path = _first_scalar_text(batch.get("cloudy_s2_path"))
    roi = _first_scalar_text(batch.get("roi_id"))
    clear_date = _first_scalar_text(batch.get("clear_date"))
    sample = _first_scalar_text(batch.get("sample_id"))
    if clear_path:
        keys.append(f"target:{clear_path}")
    if roi and clear_date:
        keys.append(f"roi_clear:{roi}:{clear_date}")
    if roi:
        keys.append(f"roi:{roi}")
    if cloudy_path:
        keys.append(f"cloudy:{cloudy_path}")
    if sample:
        keys.append(f"sample:{sample}")
    return tuple(dict.fromkeys(keys))


def visual_dedup_key(batch: dict[str, Any]) -> str:
    """Identity used only for visualization de-duplication.

    Multiple ALLClear pairs can share the same cloud-free target but have
    different cloudy inputs.  Without this key, top-structure sampling can pick
    the same target several times, especially in heavy buckets.
    """

    keys = visual_dedup_keys(batch)
    return keys[0] if keys else ""


def select_visual_candidates(
    buckets: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]],
    bucket_order: tuple[str, ...] | list[str],
    samples_per_bucket: int,
) -> dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]]:
    """Select top candidates while avoiding repeats across all saved buckets."""

    selected: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]] = {}
    global_keys: set[str] = set()
    for name in bucket_order:
        ranked = sorted(buckets.get(name, []), key=lambda item: item[0], reverse=True)
        chosen: list[tuple[float, dict[str, Tensor], dict[str, Any]]] = []
        for item in ranked:
            keys = set(visual_dedup_keys(item[2]))
            if keys and keys.intersection(global_keys):
                continue
            chosen.append(item)
            global_keys.update(keys)
            if len(chosen) >= samples_per_bucket:
                break
        if len(chosen) < samples_per_bucket:
            local_keys = set().union(*(visual_dedup_keys(item[2]) for item in chosen))
            for item in ranked:
                keys = set(visual_dedup_keys(item[2]))
                if keys and keys.intersection(local_keys):
                    continue
                chosen.append(item)
                local_keys.update(keys)
                if len(chosen) >= samples_per_bucket:
                    break
        selected[name] = chosen[:samples_per_bucket]
    return selected


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    discriminator: torch.nn.Module | None,
    disc_optimizer: torch.optim.Optimizer | None,
    disc_criterion: CloudDiscriminatorLoss | HingeDiscriminatorLoss | R1DiscriminatorLoss,
    adv_weight: float,
    train: bool,
    epoch: int = 0,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    patchgan_condition_mask: bool = True,
    cloud_gan_image: str = "composite",
    model_band_indices: tuple[int, ...] | None = None,
    model_reflectance_range: tuple[float, float] | None = None,
    rank: int = 0,
) -> dict[str, float]:
    model.train(train)
    if discriminator is not None:
        discriminator.train(train)
    totals: dict[str, list[float]] = {}
    cloud_branch_trainable = any(param.requires_grad for param in _unwrap(model).cloud_branch.parameters())
    effective_adv_weight = adv_weight if cloud_branch_trainable else 0.0

    # AMP: automatic mixed precision via torch.amp.
    # ``autocast`` runs eligible ops (matmul, conv) in fp16 for speed +
    # memory savings; ``GradScaler`` prevents small-gradient underflow.
    use_amp = bool(amp_enabled) and train and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    grad_accum_steps = max(1, int(grad_accum_steps))

    desc = f"Epoch {epoch:4d} [train]" if train else f"Epoch {epoch:4d} [val  ]"
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True, disable=not is_main_process(rank))

    num_batches = len(loader)
    for batch_idx, batch in enumerate(pbar, start=1):
        group_start = ((batch_idx - 1) // grad_accum_steps) * grad_accum_steps + 1
        group_end = min(group_start + grad_accum_steps - 1, num_batches)
        group_size = max(1, group_end - group_start + 1)
        start_group = batch_idx == group_start
        update_now = batch_idx == group_end

        batch = apply_model_band_indices(move_batch(batch, device), model_band_indices, model_reflectance_range)
        if train and optimizer is not None and start_group:
            optimizer.zero_grad(set_to_none=True)

        # ── generator forward (inside autocast when training) ──────
        with torch.set_grad_enabled(train):
            with autocast_context(use_amp, amp_dtype):
                outputs = model(
                    batch["s2_toa"],
                    batch.get("s1"),
                    batch["cld_shdw"],
                    softshadow_bbox=batch.get("bbox"),
                    softshadow_case=batch.get("shadow_case"),
                    return_intermediates=True,
                )
                fake_logits = None

            # ── discriminator step (train only) ───────────────────
            real_feats: list[Tensor] | None = None
            fake_feats: list[Tensor] | None = None
            if train and discriminator is not None and disc_optimizer is not None and effective_adv_weight > 0:
                set_requires_grad(discriminator, True)
                if start_group:
                    disc_optimizer.zero_grad(set_to_none=True)
                with autocast_context(use_amp, amp_dtype):
                    cloud_mask = outputs["M_cloud"].float().detach()
                    real_cloud, fake_cloud = cloud_gan_pair(outputs, batch, cloud_gan_image)
                    real_for_disc = real_cloud.detach()
                    if isinstance(disc_criterion, R1DiscriminatorLoss):
                        real_for_disc.requires_grad_(True)
                    real_logits, real_feats_raw = _call_discriminator(
                        discriminator,
                        batch["s2_toa"],
                        cloud_mask,
                        real_for_disc,
                        return_features=True,
                        condition_mask=patchgan_condition_mask,
                    )
                    fake_logits_d, _ = _call_discriminator(
                        discriminator,
                        batch["s2_toa"],
                        cloud_mask,
                        fake_cloud.detach(),
                        return_features=True,
                        condition_mask=patchgan_condition_mask,
                    )
                if isinstance(disc_criterion, HingeDiscriminatorLoss):
                    d_loss, d_terms = disc_criterion(real_logits, fake_logits_d, cloud_mask)
                elif isinstance(disc_criterion, R1DiscriminatorLoss):
                    d_loss, d_terms = disc_criterion(real_logits, fake_logits_d, real_for_disc, cloud_mask)
                else:
                    d_loss, d_terms = disc_criterion(real_logits, fake_logits_d)
                assert_finite_terms(d_loss, d_terms, batch, epoch=epoch, split="disc")
                d_loss_for_backward = d_loss / float(group_size)
                with no_sync_if_needed(discriminator, train and not update_now):
                    if scaler is not None:
                        scaler.scale(d_loss_for_backward).backward()
                        if update_now:
                            scaler.unscale_(disc_optimizer)
                            assert_finite_gradients(discriminator, batch, epoch=epoch, split="disc")
                            scaler.step(disc_optimizer)
                    else:
                        d_loss_for_backward.backward()
                        if update_now:
                            assert_finite_gradients(discriminator, batch, epoch=epoch, split="disc")
                            disc_optimizer.step()
                for key, value in d_terms.items():
                    totals.setdefault(key, []).append(float(value.detach().cpu()))

                set_requires_grad(discriminator, False)
                with autocast_context(use_amp, amp_dtype):
                    with torch.no_grad():
                        _, real_feats_raw = _call_discriminator(
                            discriminator,
                            batch["s2_toa"],
                            cloud_mask,
                            real_cloud,
                            return_features=True,
                            condition_mask=patchgan_condition_mask,
                        )
                    real_feats = [f.detach() for f in real_feats_raw] if real_feats_raw else None
                    fake_logits, fake_feats_raw = _call_discriminator(
                        discriminator,
                        batch["s2_toa"],
                        cloud_mask,
                        fake_cloud,
                        return_features=True,
                        condition_mask=patchgan_condition_mask,
                    )
                    fake_feats = fake_feats_raw if fake_feats_raw else None
            elif discriminator is not None and effective_adv_weight > 0:
                with autocast_context(use_amp, amp_dtype):
                    cloud_mask = outputs["M_cloud"].float().detach()
                    _, fake_cloud = cloud_gan_pair(outputs, batch, cloud_gan_image)
                    fake_logits, fake_feats_raw = _call_discriminator(
                        discriminator,
                        batch["s2_toa"],
                        cloud_mask,
                        fake_cloud,
                        return_features=True,
                        condition_mask=patchgan_condition_mask,
                    )
                    fake_feats = fake_feats_raw if fake_feats_raw else None

            # ── generator loss + backward ─────────────────────────
            with autocast_context(use_amp, amp_dtype):
                loss, terms = criterion(outputs, batch, fake_logits=fake_logits,
                                       real_features=real_feats, fake_features=fake_feats)
            assert_finite_terms(loss, terms, batch, epoch=epoch, split="train" if train else "val")
            if train and optimizer is not None:
                loss_for_backward = loss / float(group_size)
                with no_sync_if_needed(model, train and not update_now):
                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                        if update_now:
                            scaler.unscale_(optimizer)
                            if grad_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            assert_finite_gradients(model, batch, epoch=epoch, split="train")
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        loss_for_backward.backward()
                        if update_now:
                            if grad_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            assert_finite_gradients(model, batch, epoch=epoch, split="train")
                            optimizer.step()
                set_requires_grad(discriminator, True)

        for key, value in terms.items():
            totals.setdefault(key, []).append(float(value.detach().cpu()))

        # Update postfix with running averages for the active restoration terms.
        running = aggregate(totals)
        pbar.set_postfix(progress_postfix(running))

    return aggregate(totals)


def validate_and_visualize(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    rgb_indices: tuple[int, int, int],
    visual_samples_per_bucket: int = 5,
    cloud_index: int = 1,
    visual_candidate_pool_per_bucket: int | None = None,
    visual_buckets: list[str] | tuple[str, ...] | None = None,
    visual_profile: str = "stage1",
    visual_rgb_gamma: float | None = None,
    visual_rgb_gain: float | None = None,
    visual_rgb_stretch: str | None = None,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    model_band_indices: tuple[int, ...] | None = None,
    model_reflectance_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    selected_buckets = tuple(str(name) for name in (visual_buckets or CLOUD_VISUAL_BUCKETS))
    invalid_buckets = [name for name in selected_buckets if name not in CLOUD_VISUAL_BUCKETS]
    if invalid_buckets:
        raise ValueError(f"eval.visual_buckets contains invalid cloud buckets: {invalid_buckets}")
    pool_size = max(
        int(visual_samples_per_bucket),
        int(visual_candidate_pool_per_bucket or max(visual_samples_per_bucket * 5, visual_samples_per_bucket)),
    )
    buckets: dict[str, list[tuple[float, dict[str, Tensor], dict[str, Any]]]] = {name: [] for name in selected_buckets}
    totals: dict[str, list[float]] = {}
    model.eval()
    use_amp = bool(amp_enabled) and torch.cuda.is_available()
    with torch.no_grad(), autocast_context(use_amp, amp_dtype):
        pbar = tqdm(loader, desc=f"Epoch {epoch:4d} [val  ]", unit="batch", leave=False, dynamic_ncols=True)
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
            loss, terms = criterion(outputs, batch, fake_logits=None)
            assert_finite_terms(loss, terms, batch, epoch=epoch, split="val")
            for key, value in terms.items():
                totals.setdefault(key, []).append(float(value.detach().cpu()))
            # Use the original ALLClear cloud label for visualization buckets.
            # DADIGANBaseline intentionally sets outputs["M_cloud"] to all-ones
            # for full-image loss, which would otherwise put every sample in
            # the high-cloud bucket and hide low/medium validation cases.
            frac = cloud_fraction(batch["cld_shdw"], cloud_index=cloud_index).to(device)
            for i, val in enumerate(frac):
                name = cloud_bucket_name(val)
                if name not in buckets:
                    continue
                score = visual_structure_score(batch, i, rgb_indices)
                keep_top_visual_candidate(
                    buckets[name],
                    (score, slice_tensor_batch(outputs, i, cpu=True), slice_tensor_batch(batch, i, cpu=True)),
                    pool_size,
                )
            running = aggregate(totals)
            pbar.set_postfix(progress_postfix(running))
    selected_by_bucket = select_visual_candidates(buckets, selected_buckets, visual_samples_per_bucket)
    for name, selected in selected_by_bucket.items():
        if not selected:
            continue
        out = concat_tensor_batches([pair[1] for pair in selected])
        batch = concat_tensor_batches([pair[2] for pair in selected])
        save_visuals(
            out,
            batch,
            run_dir / "visualizations" / f"epoch_{epoch:04d}_stage1_{name}.png",
            rgb_indices,
            max_items=visual_samples_per_bucket,
            visual_profile=visual_profile,
            visual_rgb_gamma=visual_rgb_gamma,
            visual_rgb_gain=visual_rgb_gain,
            visual_rgb_stretch=visual_rgb_stretch,
        )
    return aggregate(totals)


def _setup_logging(run_dir: Path, *, rank: int = 0) -> None:
    """Configure console and file logging."""
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    if not is_main_process(rank):
        logger.addHandler(logging.NullHandler())
        return

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler (full debug log)
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    logger.addHandler(fh)


def _run_dir_from_checkpoint(path: str | Path) -> Path:
    ckpt_path = Path(path).expanduser().resolve()
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """Format metrics dict into a compact one-line string."""
    parts = [f"{prefix}total={metrics.get('total', 0):.4f}"]
    for k in sorted(metrics):
        if k == "total":
            continue
        short = k.replace("_", ".")
        parts.append(f"{short}={metrics[k]:.4f}")
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ALLClear Stage1 restoration model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GPU 0 only, save to ./my_runs/
  python -m src.allclear.train --config configs/allclear_tgdad_softshadow_stage1.yaml --stage stage1 --gpu 0 -o ./my_runs

  # Use two GPUs, custom run name
  python -m src.allclear.train --config configs/allclear_tgdad_softshadow_stage1.yaml --stage stage1 --gpu 0,1 -n my_experiment

  # Resume from checkpoint
  python -m src.allclear.train --config configs/allclear_tgdad_softshadow_stage1.yaml --stage stage1 --gpu 0,1 --resume outputs/allclear/.../checkpoints/last.pt
""",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stage", default=None, help="Compatibility flag; model type is controlled by config model.framework.")
    parser.add_argument("--resume", help="Resume from a checkpoint .pt file")
    parser.add_argument(
        "--resume-in-place",
        action="store_true",
        help="When --resume is set, append logs/checkpoints to the original run directory instead of creating a new run.",
    )
    parser.add_argument(
        "--gpu", "--gpu-ids", dest="gpu_ids", default=None,
        help="GPU IDs to use, e.g. '0' for single GPU or '0,1' for two GPUs. "
             "If not set, uses all available GPUs via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--output-dir", "-o", dest="output_dir", default=None,
        help="Override output root directory (default: from config output_root, or 'outputs/allclear').",
    )
    parser.add_argument(
        "--run-name", "-n", dest="run_name", default=None,
        help="Override run name for the output subdirectory (default: from config run_name).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    model_band_indices = model_band_indices_from_cfg(cfg)
    model_reflectance_range = model_reflectance_range_from_cfg(cfg)

    # --- GPU selection (must happen BEFORE any torch.cuda.* call) ---
    # Query physical GPUs via nvidia-smi for logging (avoids initializing CUDA early)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        physical_gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    except Exception:
        physical_gpus = []

    logger.info("System has %d physical GPU(s): %s",
                len(physical_gpus),
                ", ".join(g.split(",")[1].strip() for g in physical_gpus) if physical_gpus else "unknown")

    # Set CUDA_VISIBLE_DEVICES *before* CUDA initializes
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info("Requested GPU(s): %s", args.gpu_ids)

    ddp_enabled, rank, local_rank, world_size = distributed_env()
    requested_distributed = str(train_cfg.get("distributed", "auto")).lower()
    if requested_distributed == "ddp" and not ddp_enabled and args.gpu_ids and "," in args.gpu_ids:
        raise RuntimeError(
            "train.distributed=ddp requires torchrun for multi-GPU training. "
            "Use: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m src.allclear.train ..."
        )
    if ddp_enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA in this trainer.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=str(train_cfg.get("dist_backend", "nccl")))

    # Now it's safe to query torch.cuda (CUDA initializes with the restricted device set)
    selected_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if selected_gpus == 0:
        device = torch.device("cpu")
        logger.warning("No GPU available, falling back to CPU.")
    elif ddp_enabled:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        gpu_names = [torch.cuda.get_device_name(i) for i in range(selected_gpus)]
        logger.info("Training on %d GPU(s): %s", selected_gpus, ", ".join(gpu_names))

    configure_torch_backend(train_cfg)
    set_seed(int(cfg.get("seed", 2026)) + rank)

    run_dir_obj: list[str | None] = [None]
    if is_main_process(rank):
        if args.resume_in_place:
            if not args.resume:
                raise ValueError("--resume-in-place requires --resume PATH")
            run_dir = _run_dir_from_checkpoint(args.resume)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        else:
            # Resolve output path: CLI overrides > config file > defaults
            output_root = args.output_dir or cfg.get("output_root", "outputs/allclear")
            run_name = args.run_name or cfg.get("run_name")
            run_dir = make_run_dir(output_root, "stage1", run_name)
            save_config(cfg, run_dir)
        run_dir_obj[0] = str(run_dir)
    if ddp_enabled:
        dist.broadcast_object_list(run_dir_obj, src=0)
    run_dir = Path(str(run_dir_obj[0]))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir, rank=rank)

    logger.info("=" * 70)
    logger.info("ALLClear Training - Stage1")
    logger.info("Config: %s", args.config)
    logger.info("Run directory: %s", run_dir)
    if args.resume_in_place:
        logger.info("Resume in place: appending to existing train.log/train_log.csv")
    logger.info("Device: %s (%d visible GPU(s), world_size=%d, rank=%d)", device, selected_gpus, world_size, rank)
    if model_band_indices is not None:
        logger.info("Model S2 band subset enabled: %s", list(model_band_indices))
    if model_reflectance_range is not None:
        logger.info(
            "Model reflectance stretch enabled: [%.4f, %.4f] -> [0, 1]",
            model_reflectance_range[0],
            model_reflectance_range[1],
        )
    logger.info("=" * 70)

    train_loader = make_loader(cfg, "train", distributed=ddp_enabled, rank=rank, world_size=world_size)
    val_loader = make_loader(cfg, "val", distributed=False, rank=0, world_size=1) if is_main_process(rank) else None
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader) if val_loader is not None else 0)

    model = build_model(cfg).to(device)
    maybe_compile_submodules(model, cfg)

    # Multi-GPU: prefer torchrun + DDP. DataParallel is kept only for legacy single-process runs.
    if ddp_enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", True)),
        )
        logger.info("Model wrapped with DistributedDataParallel across %d ranks", world_size)
    elif selected_gpus > 1:
        model = torch.nn.DataParallel(model)
        logger.warning("Model wrapped with legacy DataParallel. Use torchrun for DDP performance.")

    disc = None
    disc_opt = None
    base_loss_weights = weights_from_cfg(cfg)
    disc_type = str(cfg.get("model", {}).get("discriminator", "patchgan")).lower()
    patchgan_condition_mask = bool(cfg.get("model", {}).get("cloud_discriminator_condition_mask", True))
    if base_loss_weights.cloud_adv > 0:
        model_cfg = cfg.get("model", {})
        s2c = int(model_cfg.get("s2_channels", 13))
        if disc_type == "sn_patchgan":
            disc = make_sn_patchgan_discriminator(
                input_nc=s2c, ndf=64, n_layers=6, cond_mask=True,
            ).to(device)
        elif disc_type == "pix2pixhd_nlayer":
            disc = make_pix2pixhd_nlayer_discriminator(
                input_nc=s2c,
                ndf=int(model_cfg.get("cloud_discriminator_base_channels", 64)),
                n_layers=int(model_cfg.get("cloud_discriminator_layers", 4)),
            ).to(device)
        else:
            disc = make_cloud_discriminator(
                s2c,
                base_channels=int(model_cfg.get("cloud_discriminator_base_channels", 32)),
                num_layers=int(model_cfg.get("cloud_discriminator_layers", 5)),
                condition_channels=1 if patchgan_condition_mask else 0,
                norm_type=str(model_cfg.get("cloud_discriminator_norm", "batch")),
                output_mode=str(model_cfg.get("cloud_discriminator_output", "patch")),
            ).to(device)
        if ddp_enabled:
            disc = DistributedDataParallel(
                disc,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        elif selected_gpus > 1:
            disc = torch.nn.DataParallel(disc)
        disc_opt = make_optimizer(
            disc.parameters(),
            lr=float(cfg.get("train", {}).get("disc_lr", 1.0e-4)),
            weight_decay=0.0,
            train_cfg=train_cfg,
        )

    optimizer = make_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.get("train", {}).get("lr", 1.0e-4)),
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 1.0e-4)),
        train_cfg=train_cfg,
    )
    start_epoch = 1
    best_metric = float("inf")
    if args.resume:
        start_epoch, best_metric = load_checkpoint(args.resume, model, optimizer, disc, disc_opt)

    adversarial_loss = str(cfg.get("loss", {}).get("adversarial_loss", "bce"))
    loss_profile = str(cfg.get("loss", {}).get("profile", "stage1")).lower()
    common_loss_kwargs = dict(
        rgb_indices=tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1])),
        adversarial_loss=adversarial_loss,
        perceptual_type=str(cfg.get("loss", {}).get("perceptual_type", "rgb_l1")),
        perceptual_lama_repo=cfg.get("loss", {}).get("perceptual_lama_repo"),
        perceptual_weights_path=cfg.get("loss", {}).get("perceptual_weights_path"),
        cloud_l1_reduction=str(cfg.get("loss", {}).get("cloud_l1_reduction", "mask_mean")),
        cloud_l1_region=str(cfg.get("loss", {}).get("cloud_l1_region", "cloud")),
        cloud_kl_reduction=str(cfg.get("loss", {}).get("cloud_kl_reduction", "image_mean")),
        cloud_kl_mode=str(cfg.get("loss", {}).get("cloud_kl_mode", "softmax")),
        perceptual_input=str(cfg.get("loss", {}).get("perceptual_input", "cloud_context")),
        feature_matching_loss_type=str(cfg.get("loss", {}).get("feature_matching_loss_type", "mse")),
    )
    if loss_profile in {"cloud_only", "dadigan_lama_ffc", "dadigan_baseline"}:
        criterion = CloudOnlyRestorationLoss(base_loss_weights, **common_loss_kwargs).to(device)
        logger.info("Loss profile: %s (cloud/degraded restoration terms only)", loss_profile)
    else:
        criterion = AllClearStageLoss(
            base_loss_weights,
            **common_loss_kwargs,
            final_mask_mode=str(cfg.get("loss", {}).get("final_mask_mode", "degraded")),
            shadow_mask_outside_weight=float(cfg.get("loss", {}).get("shadow_mask_outside_weight", 0.05)),
            shadow_mask_region=str(cfg.get("loss", {}).get("shadow_mask_region", "support")),
            shadow_removal_region=str(cfg.get("loss", {}).get("shadow_removal_region", "shadow")),
            shadow_removal_loss_type=str(cfg.get("loss", {}).get("shadow_removal_loss_type", "l1")),
            shadow_penumbra_mode=str(cfg.get("loss", {}).get("shadow_penumbra_mode", "softshadow_no_penumbra")),
            shadow_soft_target_low_pass_kernel=int(cfg.get("loss", {}).get("shadow_soft_target_low_pass_kernel", 5)),
            shadow_soft_target_mode=str(cfg.get("loss", {}).get("shadow_soft_target_mode", "hard_support")),
            shadow_soft_target_division_threshold=float(cfg.get("loss", {}).get("shadow_soft_target_division_threshold", 0.05)),
            shadow_case_gating=bool(cfg.get("loss", {}).get("shadow_case_gating", False)),
        ).to(device)
    log_fields = log_fields_for_profile(loss_profile)
    adversarial_loss = str(cfg.get("loss", {}).get("adversarial_loss", "bce"))
    disc_criterion: CloudDiscriminatorLoss | HingeDiscriminatorLoss | R1DiscriminatorLoss
    if adversarial_loss == "hinge":
        disc_criterion = HingeDiscriminatorLoss(
            mask_as_fake_target=bool(cfg.get("loss", {}).get("mask_as_fake_target", False)),
            allow_scale_mask=bool(cfg.get("loss", {}).get("allow_scale_mask", True)),
            mask_scale_mode=str(cfg.get("loss", {}).get("mask_scale_mode", "nearest")),
        ).to(device)
    elif adversarial_loss in {"r1", "non_saturating", "non_saturating_r1", "softplus"}:
        disc_criterion = R1DiscriminatorLoss(
            gp_coef=float(cfg.get("loss", {}).get("r1_gp_coef", 0.001)),
            mask_as_fake_target=bool(cfg.get("loss", {}).get("mask_as_fake_target", True)),
            allow_scale_mask=bool(cfg.get("loss", {}).get("allow_scale_mask", True)),
            mask_scale_mode=str(cfg.get("loss", {}).get("mask_scale_mode", "nearest")),
        ).to(device)
    else:
        disc_criterion = CloudDiscriminatorLoss().to(device)
    log_path = run_dir / "train_log.csv"
    val_every = int(cfg.get("train", {}).get("val_every", 1))
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    keep_best = int(cfg.get("train", {}).get("keep_best", 3))
    best_metric_name = str(cfg.get("train", {}).get("best_metric", "recon_total"))
    amp_enabled = bool(cfg.get("train", {}).get("amp", True))
    amp_dtype = amp_dtype_from_cfg(str(cfg.get("train", {}).get("amp_dtype", "fp16")))
    grad_clip_norm = float(cfg.get("train", {}).get("grad_clip_norm", 1.0))
    grad_accum_steps = max(1, int(cfg.get("train", {}).get("grad_accum_steps", 1)))
    cloud_gan_image = str(cfg.get("loss", {}).get("cloud_gan_image", "composite"))
    best_paths: list[tuple[float, Path]] = []
    logger.info(
        "Training: epochs=%d (start=%d)  batch_size_per_rank=%d  grad_accum=%d  effective_global_batch=%d  lr=%.1e  val_every=%d  keep_best=%d",
        epochs,
        start_epoch,
        int(cfg.get("train", {}).get("batch_size", 1)),
        grad_accum_steps,
        int(cfg.get("train", {}).get("batch_size", 1)) * grad_accum_steps * world_size,
        float(cfg.get("train", {}).get("lr", 1.0e-4)),
        val_every,
        keep_best,
    )
    logger.info("Best checkpoint metric: %s", best_metric_name)

    # --- LR scheduler: Cosine Annealing with Linear Warmup ---
    # Loshchilov & Hutter (ICLR 2017) + warmup (Vaswani et al., NeurIPS 2017).
    # Used by SwinIR, Restormer, HAT, and most modern image-restoration models.
    lr_scheduler_name = str(cfg.get("train", {}).get("lr_scheduler", "none"))
    gen_scheduler = None
    disc_scheduler = None
    if lr_scheduler_name == "cosine_warmup":
        lr_warmup_epochs = int(cfg.get("train", {}).get("lr_warmup_epochs", 5))
        lr_min = float(cfg.get("train", {}).get("lr_min", 1.0e-6))
        # Total epochs for scheduler (may differ from epochs for warmup accounting)
        sched_start = start_epoch  # account for resume
        sched_epochs = max(1, epochs - sched_start + 1)
        warmup_iters = max(1, min(lr_warmup_epochs, sched_epochs - 1))
        cosine_iters = sched_epochs - warmup_iters
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, cosine_iters), eta_min=lr_min)
        gen_scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
        logger.info("LR scheduler: cosine_warmup  warmup=%d  min=%.1e  total=%d", warmup_iters, lr_min, sched_epochs)
        if disc_opt is not None:
            disc_warmup = LinearLR(disc_opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
            disc_cosine = CosineAnnealingLR(disc_opt, T_max=max(1, cosine_iters), eta_min=lr_min)
            disc_scheduler = SequentialLR(disc_opt, schedulers=[disc_warmup, disc_cosine], milestones=[warmup_iters])
    elif lr_scheduler_name != "none":
        logger.warning("Unknown lr_scheduler '%s'; using constant LR.", lr_scheduler_name)

    train_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        epoch_start = time.time()
        epoch_weights = scheduled_weights_from_cfg(cfg, epoch)
        criterion.weights = epoch_weights

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            discriminator=disc,
            disc_optimizer=disc_opt,
            disc_criterion=disc_criterion,
            adv_weight=epoch_weights.cloud_adv,
            train=True,
            epoch=epoch,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip_norm=grad_clip_norm,
            grad_accum_steps=grad_accum_steps,
            patchgan_condition_mask=patchgan_condition_mask,
            cloud_gan_image=cloud_gan_image,
            model_band_indices=model_band_indices,
            model_reflectance_range=model_reflectance_range,
            rank=rank,
        )
        train_metrics = reduce_metrics(train_metrics, device, distributed=ddp_enabled)
        if is_main_process(rank):
            row: dict[str, float | int | str] = {"epoch": epoch, "split": "train", **train_metrics, **weights_log_dict(epoch_weights)}
            append_log(log_path, row, fieldnames=log_fields)

        train_time = time.time() - epoch_start
        seconds_per_batch = train_time / max(1, len(train_loader))
        elapsed = time.time() - train_start
        eta = timedelta(seconds=int((elapsed / (epoch - start_epoch + 1)) * (epochs - epoch))) if epoch >= start_epoch else timedelta(0)

        logger.info(
            "Epoch %4d/%d | %s | %s | train_time=%s  sec/batch=%.3f  elapsed=%s  eta=%s",
            epoch,
            epochs,
            _format_metrics(train_metrics, "T "),
            "    ",
            timedelta(seconds=int(train_time)),
            seconds_per_batch,
            timedelta(seconds=int(elapsed)),
            eta,
        )

        val_metric = float("inf")
        if is_main_process(rank) and val_loader is not None and epoch % val_every == 0:
            val_start = time.time()
            val_metrics = validate_and_visualize(
                _unwrap(model),
                val_loader,
                criterion,
                device,
                run_dir,
                epoch,
                tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1])),
                int(cfg.get("eval", {}).get("visual_samples_per_bucket", 5)),
                int(cfg.get("data", {}).get("cloud_index", 1)),
                visual_candidate_pool_per_bucket=int(cfg.get("eval", {}).get("visual_candidate_pool_per_bucket", 0)) or None,
                visual_buckets=cfg.get("eval", {}).get("visual_buckets"),
                visual_profile=str(cfg.get("eval", {}).get("visual_profile", cfg.get("loss", {}).get("profile", "stage1"))),
                visual_rgb_gamma=float(cfg.get("eval", {}).get("visual_rgb_gamma", 0.72))
                if cfg.get("eval", {}).get("visual_rgb_gamma") is not None
                else None,
                visual_rgb_gain=float(cfg.get("eval", {}).get("visual_rgb_gain", 1.08))
                if cfg.get("eval", {}).get("visual_rgb_gain") is not None
                else None,
                visual_rgb_stretch=cfg.get("eval", {}).get("visual_rgb_stretch"),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                model_band_indices=model_band_indices,
                model_reflectance_range=model_reflectance_range,
            )
            val_time = time.time() - val_start
            val_metric = val_metrics.get(best_metric_name, val_metrics.get("total", float("inf")))
            append_log(log_path, {"epoch": epoch, "split": "val", **val_metrics, **weights_log_dict(epoch_weights)}, fieldnames=log_fields)
            logger.info(
                "Epoch %4d/%d | %s | val_time=%s",
                epoch,
                epochs,
                _format_metrics(val_metrics, "V "),
                timedelta(seconds=int(val_time)),
            )

        if is_main_process(rank):
            payload = checkpoint_payload(model, optimizer, epoch, min(best_metric, val_metric), disc, disc_opt)
            torch.save(payload, run_dir / "checkpoints" / "last.pt")
            if val_metric < best_metric:
                best_metric = val_metric
                metric_tag = best_metric_name.replace("/", "_").replace(" ", "_")
                best_path = run_dir / "checkpoints" / f"best_epoch_{epoch:04d}_{metric_tag}_{val_metric:.6f}.pt"
                torch.save(payload, best_path)
                best_paths.append((val_metric, best_path))
                best_paths = sorted(best_paths, key=lambda x: x[0])
                for _, path in best_paths[keep_best:]:
                    if path.exists():
                        path.unlink()
                best_paths = best_paths[:keep_best]
                logger.info("Epoch %4d/%d | *** New best: %s=%.6f ***", epoch, epochs, best_metric_name, best_metric)

            with (run_dir / "metrics" / "latest.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "best_metric_name": best_metric_name,
                        "last_train": train_metrics,
                        "loss_weights": weights_log_dict(epoch_weights),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    f,
                    indent=2,
                )
        maybe_barrier(ddp_enabled)

        # Step LR schedulers after each epoch
        if gen_scheduler is not None:
            gen_scheduler.step()
        if disc_scheduler is not None:
            disc_scheduler.step()

    total_time = time.time() - train_start
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("Total time: %s", timedelta(seconds=int(total_time)))
    logger.info("Best metric (%s): %.6f", best_metric_name, best_metric)
    logger.info("Run directory: %s", run_dir)
    logger.info("=" * 70)

    if is_main_process(rank):
        print(f"\nrun_dir={run_dir}")
        print(f"best_metric_name={best_metric_name}")
        print(f"best_metric={best_metric:.6f}")
        print(f"total_time={timedelta(seconds=int(total_time))}")
    cleanup_distributed(ddp_enabled)


if __name__ == "__main__":
    main()
