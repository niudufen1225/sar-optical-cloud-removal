"""Numerical metrics used by the checkpoint evaluators.

This module is evaluation-only.  It deliberately does not import the model,
dataset, training configuration, or loss code so that metric definitions can be
tested independently and cannot change the training path.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


METRIC_DATA_DOMAIN = "model_supervision"
METRIC_DATA_RANGE = 1.0
SSIM_WINDOW_SIZE = 7
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _as_bchw(x: Tensor) -> Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    return x


def _hard_region_bbox(mask: Tensor) -> tuple[int, int, int, int] | None:
    """Return an inclusive-exclusive bbox for mask > 0.5, or None if empty."""

    mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"expected a single [H,W] mask, got {tuple(mask.shape)}")
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _effective_ssim_window(height: int, width: int, requested: int = SSIM_WINDOW_SIZE) -> int | None:
    limit = min(int(height), int(width), int(requested))
    if limit < 3:
        return None
    return limit if limit % 2 == 1 else limit - 1


def ssim_score(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = METRIC_DATA_RANGE,
    window_size: int = SSIM_WINDOW_SIZE,
) -> float:
    """Compute channel-mean SSIM with a uniform window.

    The evaluator fixes ``data_range=1`` and a 7x7 uniform window.  For a
    smaller crop the largest odd window in [3, 7] is used; crops smaller than
    3x3 are undefined and return NaN.  Spatial and channel means are taken
    after the SSIM map is formed.  This is the standard local SSIM formula,
    with ``count_include_pad=False`` at image boundaries.
    """

    pred = _as_bchw(pred).float()
    target = _as_bchw(target).float()
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    win = _effective_ssim_window(pred.shape[-2], pred.shape[-1], window_size)
    if win is None:
        return math.nan
    pad = win // 2
    mu_x = F.avg_pool2d(pred, win, stride=1, padding=pad, count_include_pad=False)
    mu_y = F.avg_pool2d(target, win, stride=1, padding=pad, count_include_pad=False)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.avg_pool2d(pred.square(), win, stride=1, padding=pad, count_include_pad=False) - mu_x2
    sigma_y = F.avg_pool2d(target.square(), win, stride=1, padding=pad, count_include_pad=False) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, win, stride=1, padding=pad, count_include_pad=False) - mu_xy
    c1 = (SSIM_K1 * float(data_range)) ** 2
    c2 = (SSIM_K2 * float(data_range)) ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)).clamp_min(1.0e-12)
    return float(score.mean().item())


def region_ssim(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    rgb_indices: tuple[int, int, int] = (0, 1, 2),
    data_range: float = METRIC_DATA_RANGE,
    window_size: int = SSIM_WINDOW_SIZE,
) -> float:
    """Compute RGB SSIM on the tight bbox of a hard region mask.

    Full-image SSIM uses the complete image.  Region SSIM uses the bbox of
    ``mask > 0.5`` and never pads an empty background into the crop.  An empty
    region or a crop smaller than 3x3 yields NaN and is excluded from means.
    """

    pred = _as_bchw(pred)
    target = _as_bchw(target)
    mask = _as_bchw(mask)
    if pred.shape[0] != 1 or target.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError("region_ssim expects one sample at a time")
    if max(rgb_indices) >= pred.shape[1] or max(rgb_indices) >= target.shape[1]:
        raise ValueError(f"RGB indices {rgb_indices} exceed image channels")
    bbox = _hard_region_bbox(mask[0, 0])
    if bbox is None:
        return math.nan
    y0, y1, x0, x1 = bbox
    return ssim_score(
        pred[:, list(rgb_indices), y0:y1, x0:x1],
        target[:, list(rgb_indices), y0:y1, x0:x1],
        data_range=data_range,
        window_size=window_size,
    )


def channel_bias_stats(pred: Tensor, target: Tensor, mask: Tensor, rgb_indices: tuple[int, int, int]) -> dict[str, float]:
    """Return signed RGB channel bias and mean absolute signed bias."""

    pred = _as_bchw(pred).float()
    target = _as_bchw(target).float()
    mask = _as_bchw(mask).float().clamp(0.0, 1.0)
    if max(rgb_indices) >= pred.shape[1] or max(rgb_indices) >= target.shape[1]:
        raise ValueError(f"RGB indices {rgb_indices} exceed image channels")
    values = []
    for channel in rgb_indices:
        weight = mask[:, :1]
        denom = weight.sum()
        values.append(float(((pred[:, channel : channel + 1] - target[:, channel : channel + 1]) * weight).sum().item() / denom.item()) if denom.item() > 1.0e-8 else math.nan)
    finite = [value for value in values if math.isfinite(value)]
    return {
        "bias_r": values[0],
        "bias_g": values[1],
        "bias_b": values[2],
        "mean_abs_channel_bias": float(np.mean(np.abs(finite))) if len(finite) == 3 else math.nan,
    }


def _haar_kernels(dtype: torch.dtype, device: torch.device) -> Tensor:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    lo = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=dtype, device=device)
    hi = torch.tensor([-inv_sqrt2, inv_sqrt2], dtype=dtype, device=device)
    # First axis is y, second axis is x. LH therefore detects x-direction
    # changes (vertical edges), while HL detects y-direction changes.
    return torch.stack(
        (
            torch.outer(lo, lo),
            torch.outer(lo, hi),
            torch.outer(hi, lo),
            torch.outer(hi, hi),
        ),
        dim=0,
    )


def haar_swt2(x: Tensor) -> dict[str, Tensor]:
    """One-level, undecimated 2-D Haar SWT with fixed FP32-compatible filters.

    No decimation is performed: every band is [B,C,H,W].  A one-pixel
    right/bottom ``reflect`` pad is used before the 2x2 convolution, so the
    output aligns with the input's top-left pixel and has exactly the input
    spatial shape.  This boundary rule is intentionally explicit and is shared
    by every candidate and counterfactual.
    """

    x = _as_bchw(x)
    if x.shape[-2] < 2 or x.shape[-1] < 2:
        raise ValueError("Haar SWT requires height and width >= 2")
    b, channels, height, width = x.shape
    padded = F.pad(x, (0, 1, 0, 1), mode="reflect")
    kernels = _haar_kernels(x.dtype, x.device)
    weight = kernels.unsqueeze(1).repeat(channels, 1, 1, 1)
    bands = F.conv2d(padded, weight, groups=channels)
    bands = bands.reshape(b, channels, 4, height, width)
    return {name: bands[:, :, index] for index, name in enumerate(("LL", "LH", "HL", "HH"))}


def wavelet_metrics(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> dict[str, float]:
    """Return fixed-Haar subband MAE and normalized HF energy statistics."""

    pred = _as_bchw(pred).float()
    target = _as_bchw(target).float()
    if pred.shape != target.shape:
        raise ValueError(f"wavelet shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.shape[0] != 1:
        raise ValueError("wavelet_metrics expects one sample at a time")
    if mask is None:
        mask = torch.ones_like(pred[:, :1])
    mask = _as_bchw(mask).float().clamp(0.0, 1.0)
    if mask.shape[0] != 1 or mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError("wavelet mask must be one sample with matching spatial shape")
    weight = mask.expand(-1, pred.shape[1], -1, -1)
    denom = weight.sum()
    if denom.item() <= 1.0e-8:
        return {key: math.nan for key in ("ll_mae", "lh_mae", "hl_mae", "hh_mae", "hf_energy_ratio_pred", "hf_energy_ratio_target", "hf_energy_ratio_abs_delta")}
    pred_bands = haar_swt2(pred)
    target_bands = haar_swt2(target)
    out: dict[str, float] = {}
    for name in ("LL", "LH", "HL", "HH"):
        diff = (pred_bands[name] - target_bands[name]).abs()
        out[f"{name.lower()}_mae"] = float((diff * weight).sum().item() / denom.item())
    pred_energy = sum(float((pred_bands[name].square() * weight).sum().item()) for name in ("LL", "LH", "HL", "HH"))
    target_energy = sum(float((target_bands[name].square() * weight).sum().item()) for name in ("LL", "LH", "HL", "HH"))
    pred_hf = sum(float((pred_bands[name].square() * weight).sum().item()) for name in ("LH", "HL", "HH"))
    target_hf = sum(float((target_bands[name].square() * weight).sum().item()) for name in ("LH", "HL", "HH"))
    pred_ratio = pred_hf / max(pred_energy, 1.0e-12)
    target_ratio = target_hf / max(target_energy, 1.0e-12)
    out["hf_energy_ratio_pred"] = pred_ratio
    out["hf_energy_ratio_target"] = target_ratio
    out["hf_energy_ratio_abs_delta"] = abs(pred_ratio - target_ratio)
    return out


def _reflect_box_low_pass(x: Tensor, kernel_size: int = 5) -> Tensor:
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("SAR low-pass kernel_size must be a positive odd integer")
    if kernel_size == 1:
        return x
    pad = kernel_size // 2
    if x.shape[-2] <= pad or x.shape[-1] <= pad:
        raise ValueError("SAR low-pass input is smaller than its reflect padding")
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0, count_include_pad=False)


def sar_counterfactuals(sar: Tensor, *, low_pass_kernel: int = 5) -> dict[str, Tensor | bool]:
    """Construct deterministic SAR counterfactuals without re-normalization.

    ``shuffle`` is a one-position batch roll.  It is invalid for a singleton
    batch because it equals the real SAR; callers must exclude that sample from
    shuffle statistics using ``shuffle_valid``.
    """

    sar = _as_bchw(sar).float()
    low = _reflect_box_low_pass(sar, low_pass_kernel)
    high = sar - low
    valid = sar.shape[0] > 1
    shuffled = torch.roll(sar, shifts=1, dims=0) if valid else sar.clone()
    return {"real": sar, "zero": torch.zeros_like(sar), "shuffle": shuffled, "low_pass": low, "high_pass": high, "shuffle_valid": valid}


def pearson_correlation(x: Tensor, y: Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    valid = torch.isfinite(x) & torch.isfinite(y)
    if int(valid.sum().item()) < 2:
        return math.nan
    x, y = x[valid], y[valid]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() <= 1.0e-12:
        return math.nan
    return float((x * y).sum().item() / denom.item())


def hf_magnitude(x: Tensor) -> Tensor:
    bands = haar_swt2(x)
    return torch.cat([bands[name].square() for name in ("LH", "HL", "HH")], dim=1).mean(dim=1, keepdim=True).sqrt()


def paired_bootstrap(
    s1: list[float] | np.ndarray,
    s2: list[float] | np.ndarray,
    *,
    higher_is_better: bool | None,
    resamples: int = 2000,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Paired bootstrap of per-sample S2-S1 differences.

    Non-finite pairs are removed before resampling.  Confidence intervals are
    percentile intervals computed from paired resamples, never from two
    independent summary means.  ``direction`` reports whether positive delta
    is an improvement for the metric.
    """

    if int(resamples) < 1:
        raise ValueError("resamples must be >= 1")
    a = np.asarray(s1, dtype=np.float64).reshape(-1)
    b = np.asarray(s2, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n:
        a, b = a[:n], b[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b[valid] - a[valid]
    if delta.size == 0:
        return {
            "n_valid": 0,
            "n_total": int(n),
            "mean_delta": math.nan,
            "median_delta": math.nan,
            "mean_ci95": [math.nan, math.nan],
            "median_ci95": [math.nan, math.nan],
            "higher_is_better": bool(higher_is_better),
            "direction": "undefined",
            "resamples": int(resamples),
            "seed": int(seed),
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, delta.size, size=(int(resamples), delta.size))
    sampled = delta[indices]
    means = sampled.mean(axis=1)
    medians = np.median(sampled, axis=1)
    mean_delta = float(delta.mean())
    median_delta = float(np.median(delta))
    if higher_is_better is None:
        direction = "diagnostic_only"
    else:
        improvement = mean_delta > 0.0 if higher_is_better else mean_delta < 0.0
        direction = "improved" if improvement else "worsened_or_no_change"
    return {
        "n_valid": int(delta.size),
        "n_total": int(n),
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "mean_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "median_ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "higher_is_better": bool(higher_is_better),
        "direction": direction,
        "resamples": int(resamples),
        "seed": int(seed),
    }
