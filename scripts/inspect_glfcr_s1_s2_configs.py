#!/usr/bin/env python3
"""Inspect the Phase 3 S1/S2 structure without loading checkpoints.

The default mode only parses the two YAML files, constructs the configured
models, counts modules/parameters, compares canonicalized configuration trees,
and compares same-seed shared parameter initialization.  Random tensor
forward/backward is deliberately opt-in through ``--smoke``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.config import load_config
from src.allclear.modules.dadigan import (
    DDIN,
    DDINStep,
    GLFCRCoupledDDINStep,
    GLFCRDynamicFilterGenerator,
    GLFCRFusionStep,
)
from src.allclear.modules.lama_ffc import FFCResnetBlock, LearnableSpatialTransformWrapper
from src.allclear.paired_initialization import (
    POST_FILTER_KEY_PREFIX,
    load_model_initialization,
    validate_paired_metadata,
)
from src.allclear.train import build_model


ALLOWED_CONFIG_DIFFS = {
    ("run_name",),
    ("model", "cloud_post_ddin_sar_filter"),
    ("model", "cloud_post_ddin_sar_filter_kernel_size"),
}
CONFIG_DEFAULTS = {
    ("model", "cloud_lowres_enabled"): None,
    ("model", "cloud_ddin_glfcr_coupled"): None,
    ("model", "cloud_post_ddin_sar_filter"): "none",
    ("model", "cloud_post_ddin_sar_filter_kernel_size"): None,
}


def _canonical(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        out = {str(key): _canonical(item, path + (str(key),)) for key, item in value.items()}
        for key_path, default in CONFIG_DEFAULTS.items():
            if path == key_path[:-1] and key_path[-1] not in out:
                out[key_path[-1]] = default
        return {key: out[key] for key in sorted(out)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item, path + (str(index),)) for index, item in enumerate(value)]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, str) and (
        path[-1:] == ("root",)
        or path[-1:] == ("cache_dir",)
        or path[-1:] == ("perceptual_weights_path",)
        or (bool(path) and path[-1].endswith(("_path", "_manifest")))
    ):
        return str(Path(value).expanduser().resolve())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        out: dict[tuple[str, ...], Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, prefix + (str(key),)))
        return out
    return {prefix: value}


def config_differences(config_s1: dict[str, Any], config_s2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    left = _flatten(_canonical(config_s1))
    right = _flatten(_canonical(config_s2))
    differences: dict[str, dict[str, Any]] = {}
    for path in sorted(set(left) | set(right)):
        if left.get(path) == right.get(path):
            continue
        key = ".".join(path)
        differences[key] = {"s1": left.get(path), "s2": right.get(path), "allowed": path in ALLOWED_CONFIG_DIFFS}
    return differences


def _count(module: nn.Module, cls: type[nn.Module]) -> int:
    return sum(1 for item in module.modules() if isinstance(item, cls))


def _parameter_counts(module: nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
    }


def inspect_model(model: nn.Module) -> dict[str, Any]:
    branch = getattr(model, "cloud_branch", None)
    if branch is None:
        raise TypeError("The configured model does not expose cloud_branch; Phase 3 requires DADIGANBaseline.")
    ddin = getattr(branch, "ddin", None)
    post_filter = getattr(branch, "post_ddin_sar_filter", None)
    pdafm = getattr(branch, "pdafm", None)
    cab1 = getattr(pdafm, "cab_ps", None)
    cab1_attention = getattr(getattr(cab1, "attn", None), "attention_mode", None)
    cab2_source = getattr(pdafm, "cab2_residual_source", None)
    cab2_scale = getattr(pdafm, "cab2_update_scale", None)
    post_dfg = _count(post_filter, GLFCRDynamicFilterGenerator) if isinstance(post_filter, nn.Module) else 0
    ordinary_steps = len(getattr(ddin, "steps", ())) if isinstance(ddin, DDIN) else 0
    return {
        **_parameter_counts(model),
        "pixel_unshuffle_count": _count(branch, nn.PixelUnshuffle),
        "pixel_shuffle_count": _count(branch, nn.PixelShuffle),
        "ddin_type": type(ddin).__name__ if ddin is not None else None,
        "ordinary_ddin_step_count": ordinary_steps,
        "glfcr_coupled_ddin_step_count": _count(branch, GLFCRCoupledDDINStep),
        "glfcr_fusion_step_count": _count(branch, GLFCRFusionStep),
        "glfcr_dynamic_filter_generator_count": _count(branch, GLFCRDynamicFilterGenerator),
        "post_ddin_dynamic_filter_generator_count": post_dfg,
        "ffc_block_count": _count(branch, FFCResnetBlock),
        "spatial_wrapper_count": _count(branch, LearnableSpatialTransformWrapper),
        "cab_attention_mode": cab1_attention,
        "cab2_residual_source": cab2_source,
        "cab2_update_scale": cab2_scale,
        "lowres_enabled": getattr(branch, "lowres_enabled", None),
        "ddin_glfcr_coupled": getattr(branch, "ddin_glfcr_coupled", None),
        "post_ddin_sar_filter_mode": getattr(branch, "post_ddin_sar_filter_mode", None),
    }


def _structure_errors(label: str, stats: dict[str, Any]) -> list[str]:
    expected = {
        "lowres_enabled": True,
        "ordinary_ddin_step_count": 3,
        "glfcr_coupled_ddin_step_count": 0,
        "glfcr_fusion_step_count": 0,
        "ffc_block_count": 0,
        "spatial_wrapper_count": 0,
        "cab_attention_mode": "standard",
        "cab2_residual_source": "reference",
        "cab2_update_scale": 0.1,
    }
    errors = []
    for key, wanted in expected.items():
        got = stats.get(key)
        if got != wanted:
            errors.append(f"{label}: {key} expected {wanted!r}, got {got!r}")
    wanted_post_dfg = 0 if label == "S1" else 1
    if stats.get("post_ddin_dynamic_filter_generator_count") != wanted_post_dfg:
        errors.append(
            f"{label}: post_ddin_dynamic_filter_generator_count expected {wanted_post_dfg}, "
            f"got {stats.get('post_ddin_dynamic_filter_generator_count')!r}"
        )
    if label == "S2" and stats.get("glfcr_coupled_ddin_step_count") != 0:
        errors.append("S2: coupled DDIN is forbidden in the post-DDIN-only ablation")
    return errors


def _build(config: dict[str, Any], seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(int(seed))
    return build_model(config).to(device)


def compare_initialization(model_s1: nn.Module, model_s2: nn.Module, tolerance: float = 1.0e-7) -> dict[str, Any]:
    state_s1 = model_s1.state_dict()
    state_s2 = model_s2.state_dict()
    keys_s1 = set(state_s1)
    keys_s2 = set(state_s2)
    shared = sorted(keys_s1 & keys_s2)
    differences: dict[str, float] = {}
    for key in shared:
        left = state_s1[key].detach().float().cpu()
        right = state_s2[key].detach().float().cpu()
        if left.shape != right.shape:
            differences[key] = math.inf
        else:
            differences[key] = float((left - right).abs().max().item()) if left.numel() else 0.0
    global_max = max(differences.values(), default=0.0)
    return {
        "s1_key_count": len(keys_s1),
        "s2_key_count": len(keys_s2),
        "shared_key_count": len(shared),
        "s1_only_keys": sorted(keys_s1 - keys_s2),
        "s2_only_keys": sorted(keys_s2 - keys_s1),
        "shared_tensor_max_abs_diff": differences,
        "global_max_abs_diff": global_max,
        "tolerance": tolerance,
        "match": bool(global_max <= tolerance),
    }


def make_smoke_inputs(
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    height: int,
    width: int,
) -> dict[str, Tensor]:
    if height % 2 or width % 2:
        raise ValueError("height and width must be divisible by 2 when low-resolution PixelUnshuffle is enabled")
    torch.manual_seed(int(seed) + 1)
    s2 = torch.rand(batch_size, 3, height, width, device=device)
    sar = torch.rand(batch_size, 2, height, width, device=device)
    cld_shdw = torch.zeros(batch_size, 4, height, width, device=device)
    cloud = torch.rand(batch_size, height, width, device=device) > 0.5
    shadow = (torch.rand(batch_size, height, width, device=device) > 0.8) & ~cloud
    cld_shdw[:, 1] = cloud.to(dtype=s2.dtype)
    cld_shdw[:, 3] = shadow.to(dtype=s2.dtype)
    return {"s2": s2, "sar": sar, "cld_shdw": cld_shdw}


def smoke(model: nn.Module, *, device: torch.device, seed: int, batch_size: int, height: int, width: int) -> dict[str, Any]:
    model.train()
    inputs = make_smoke_inputs(
        device=device,
        seed=seed,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    s2, sar, cld_shdw = inputs["s2"], inputs["sar"], inputs["cld_shdw"]
    outputs = model(s2, sar, cld_shdw, return_intermediates=True)
    tensors = list(outputs.values()) if isinstance(outputs, dict) else [outputs]
    tensors = [value for value in tensors if torch.is_tensor(value)]
    scalar = sum(value.float().square().mean() for value in tensors)
    scalar.backward()
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in tensors)
    finite_grads = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "output_tensor_count": len(tensors),
        "scalar_loss_finite": bool(torch.isfinite(scalar.detach())),
        "outputs_finite": finite_outputs,
        "gradients_finite": finite_grads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-config", required=True, type=Path)
    parser.add_argument("--s2-config", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--s1-init", type=Path, default=None)
    parser.add_argument("--s2-init", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_s1 = load_config(args.s1_config)
    config_s2 = load_config(args.s2_config)
    seed = int(args.seed if args.seed is not None else config_s1.get("seed", 2026))
    device = torch.device(args.device)
    if not args.smoke and device.type != "cpu":
        raise ValueError("--device must be cpu unless --smoke is explicitly supplied")

    differences = config_differences(config_s1, config_s2)
    disallowed = {key: value for key, value in differences.items() if not value["allowed"]}
    if disallowed:
        report = {"status": "fail_config_diff", "config_differences": differences}
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    model_s1 = _build(config_s1, seed, device)
    model_s2 = _build(config_s2, seed, device)
    structure_s1 = inspect_model(model_s1)
    structure_s2 = inspect_model(model_s2)
    structure_errors = _structure_errors("S1", structure_s1) + _structure_errors("S2", structure_s2)
    raw_initialization = compare_initialization(model_s1, model_s2)
    if bool(args.s1_init) != bool(args.s2_init):
        raise ValueError("--s1-init and --s2-init must be supplied together")
    paired_initialization: dict[str, Any] | None = None
    paired_metadata: dict[str, Any] | None = None
    strict_load: dict[str, Any] = {"S1": False, "S2": False}
    paired_error: str | None = None
    if args.s1_init and args.s2_init and not structure_errors:
        try:
            meta_s1 = load_model_initialization(args.s1_init, model_s1, config_s1, expected_seed=seed)
            meta_s2 = load_model_initialization(args.s2_init, model_s2, config_s2, expected_seed=seed)
            validate_paired_metadata(meta_s1, meta_s2)
            paired_initialization = compare_initialization(model_s1, model_s2)
            invalid_s2_only = [key for key in paired_initialization["s2_only_keys"] if not key.startswith(POST_FILTER_KEY_PREFIX)]
            if paired_initialization["s1_only_keys"]:
                raise ValueError(f"Paired initialization has S1-only keys: {paired_initialization['s1_only_keys']}")
            if invalid_s2_only:
                raise ValueError(f"Paired initialization has invalid S2-only keys: {invalid_s2_only}")
            strict_load = {"S1": True, "S2": True}
            paired_metadata = {
                "checkpoint_hashes": {"S1": meta_s1["sha256"], "S2": meta_s2["sha256"]},
                "pair_id": meta_s1["pair_id"],
                "config_hashes": {"S1": meta_s1["config_sha256"], "S2": meta_s2["config_sha256"]},
                "structure_signatures": {
                    "S1": meta_s1["model_structure_signature"],
                    "S2": meta_s2["model_structure_signature"],
                },
                "seed": meta_s1["seed"],
            }
        except Exception as exc:
            paired_error = str(exc)
    report: dict[str, Any] = {
        "status": "ok",
        "device": str(device),
        "seed": seed,
        "s1_config": str(args.s1_config.expanduser().resolve()),
        "s2_config": str(args.s2_config.expanduser().resolve()),
        "allowed_config_differences": sorted(".".join(path) for path in ALLOWED_CONFIG_DIFFS),
        "config_differences": differences,
        "structure_errors": structure_errors,
        "S1": structure_s1,
        "S2": structure_s2,
        "raw_initialization": raw_initialization,
        "paired_initialization": paired_initialization,
        "paired_metadata": paired_metadata,
        "strict_load": strict_load,
    }
    if structure_errors:
        report["status"] = "fail_structure"
    if (args.s1_init and args.s2_init and paired_error) or (not args.s1_init and not raw_initialization["match"]):
        report["status"] = "fail_initialization"
        report["initialization_message"] = paired_error or "需要另行实现共同初始化机制，当前不得开始严格消融训练。"
    if args.s1_init and args.s2_init and paired_initialization is not None and not paired_initialization["match"]:
        report["status"] = "fail_initialization"
        report["initialization_message"] = "paired initialization shared parameters are not identical"

    if args.smoke and report["status"] == "ok":
        report["smoke"] = {
            "S1": smoke(model_s1, device=device, seed=seed, batch_size=args.batch_size, height=args.height, width=args.width),
            "S2": smoke(model_s2, device=device, seed=seed, batch_size=args.batch_size, height=args.height, width=args.width),
        }
        if not all(
            item["scalar_loss_finite"] and item["outputs_finite"] and item["gradients_finite"]
            for item in report["smoke"].values()
        ):
            report["status"] = "fail_smoke"

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
