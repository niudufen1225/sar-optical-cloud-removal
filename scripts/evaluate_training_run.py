#!/usr/bin/env python3
"""Analyze one ALLClear training run from logs and saved visualizations."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import shannon_entropy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_COLUMNS = [
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
    "cloud_l1",
    "known_l1",
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
    "disc_real",
    "disc_fake",
]

LOSS_COLUMNS = [
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
    "cloud_l1",
    "known_l1",
    "cloud_kl",
    "cloud_adv",
    "feature_matching",
    "perceptual",
    "disc_total",
    "disc_real_loss",
    "disc_fake_loss",
    "disc_real_gp",
]

SCHEDULE_COLUMNS = [
    "final_l1",
    "grad",
    "shadow_removal",
    "shadow_mask",
    "shadow_penumbra",
    "cloud_l1",
    "known_l1",
    "cloud_kl",
    "cloud_adv",
    "feature_matching",
    "perceptual",
    "disc_total",
]

WEIGHTED_LOSS_TERMS = [
    "final_l1",
    "grad",
    "shadow_removal",
    "shadow_mask",
    "shadow_penumbra",
    "cloud_l1",
    "known_l1",
    "cloud_kl",
    "cloud_adv",
    "feature_matching",
    "perceptual",
]

EXPECTED_BUCKETS = ("low", "medium", "high", "heavy")

PANEL_LABELS_6 = [
    "cloudy_s2",
    "target",
    "stage1_output",
    "sar",
    "cloud_mask",
    "hard_shadow",
]
PANEL_LABELS_7 = [
    "cloudy_s2",
    "target",
    "stage1_output",
    "cloud_mask",
    "hard_shadow",
    "soft_raw",
    "soft_eff",
]  # old format (without SAR), kept for backward compat
PANEL_LABELS_8 = [
    "cloudy_s2",
    "target",
    "stage1_output",
    "sar",
    "cloud_mask",
    "hard_shadow",
    "soft_raw",
    "soft_eff",
]


@dataclass
class RunContext:
    run_dir: Path
    out_dir: Path
    config: dict[str, Any]
    latest: dict[str, Any]
    train_batches: int | None
    val_batches: int | None


def expected_visual_buckets(config: dict[str, Any]) -> tuple[str, ...]:
    buckets = config.get("eval", {}).get("visual_buckets")
    if not buckets:
        return EXPECTED_BUCKETS
    selected = tuple(str(bucket) for bucket in buckets)
    invalid = [bucket for bucket in selected if bucket not in EXPECTED_BUCKETS]
    if invalid:
        raise ValueError(f"Invalid eval.visual_buckets in config: {invalid}")
    return selected


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def pct_change(first: float, last: float) -> float:
    if not math.isfinite(first) or abs(first) < 1.0e-12:
        return math.nan
    return 100.0 * (last - first) / abs(first)


def improvement_pct(first: float, last: float) -> float:
    if not math.isfinite(first) or abs(first) < 1.0e-12:
        return math.nan
    return 100.0 * (first - last) / abs(first)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_log(run_dir: Path) -> pd.DataFrame:
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing train log: {log_path}")
    df = pd.read_csv(log_path)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    for col in df.columns:
        if col not in {"split"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["epoch", "split"]).reset_index(drop=True)


def parse_timedelta_seconds(text: str) -> float:
    parts = text.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    return math.nan


def parse_train_log(run_dir: Path) -> tuple[pd.DataFrame, int | None, int | None]:
    path = run_dir / "train.log"
    if not path.exists():
        return pd.DataFrame(), None, None
    timestamp_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
    train_batches = None
    val_batches = None
    training_start: datetime | None = None
    last_epoch_end: datetime | None = None
    rows: list[dict[str, Any]] = []
    train_line_re = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|\s+T ")
    val_line_re = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|\s+V .*val_time=([0-9:]+)")
    batches_re = re.compile(r"Train batches:\s*(\d+)\s*\|\s*Val batches:\s*(\d+)")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m_ts = timestamp_re.match(line)
            ts = None
            if m_ts:
                ts = datetime.strptime(".".join(m_ts.groups()), "%Y-%m-%d %H:%M:%S.%f")
            m_batches = batches_re.search(line)
            if m_batches:
                train_batches = int(m_batches.group(1))
                val_batches = int(m_batches.group(2))
            if "Training: epochs=" in line and ts is not None:
                training_start = ts
                last_epoch_end = ts
            m_train = train_line_re.search(line)
            if m_train and ts is not None:
                epoch = int(m_train.group(1))
                train_seconds = math.nan
                if last_epoch_end is not None:
                    train_seconds = max(0.0, (ts - last_epoch_end).total_seconds())
                rows.append(
                    {
                        "epoch": epoch,
                        "split": "train",
                        "seconds": train_seconds,
                        "sec_per_batch": train_seconds / train_batches if train_batches else math.nan,
                    }
                )
            m_val = val_line_re.search(line)
            if m_val and ts is not None:
                epoch = int(m_val.group(1))
                val_seconds = parse_timedelta_seconds(m_val.group(3))
                rows.append(
                    {
                        "epoch": epoch,
                        "split": "val",
                        "seconds": val_seconds,
                        "sec_per_batch": val_seconds / val_batches if val_batches else math.nan,
                    }
                )
                last_epoch_end = ts
    if training_start is None:
        last_epoch_end = None
    return pd.DataFrame(rows), train_batches, val_batches


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    for split, part in df.groupby("split"):
        part = part.sort_values("epoch")
        for col in metric_cols:
            values = part[["epoch", col]].dropna()
            if values.empty:
                continue
            first = float(values[col].iloc[0])
            last = float(values[col].iloc[-1])
            row: dict[str, Any] = {
                "split": split,
                "metric": col,
                "first_epoch": int(values["epoch"].iloc[0]),
                "last_epoch": int(values["epoch"].iloc[-1]),
                "first": first,
                "last": last,
                "delta": last - first,
                "delta_pct": pct_change(first, last),
                "improvement_pct_if_lower_better": improvement_pct(first, last) if col in LOSS_COLUMNS else math.nan,
                "num_points": int(values.shape[0]),
            }
            if col in LOSS_COLUMNS:
                best_idx = values[col].idxmin()
                row["best_epoch"] = int(values.loc[best_idx, "epoch"])
                row["best"] = float(values.loc[best_idx, col])
            else:
                row["best_epoch"] = math.nan
                row["best"] = math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def generalization_gaps(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train"].set_index("epoch")
    val = df[df["split"] == "val"].set_index("epoch")
    common_epochs = sorted(set(train.index).intersection(set(val.index)))
    rows: list[dict[str, Any]] = []
    for epoch in common_epochs:
        row: dict[str, Any] = {"epoch": int(epoch)}
        for metric in [c for c in LOSS_COLUMNS if c in df.columns]:
            tr = safe_float(train.loc[epoch, metric])
            va = safe_float(val.loc[epoch, metric])
            row[f"{metric}_train"] = tr
            row[f"{metric}_val"] = va
            row[f"{metric}_gap_val_minus_train"] = va - tr if math.isfinite(tr) and math.isfinite(va) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def schedule_status(ctx: RunContext, df: pd.DataFrame) -> pd.DataFrame:
    cfg = ctx.config
    max_epoch = int(df["epoch"].max()) if not df.empty else 0
    loss_cfg = cfg.get("loss", {})
    schedule = cfg.get("loss_schedule", {})
    rows: list[dict[str, Any]] = []
    for metric in [m for m in SCHEDULE_COLUMNS if m in df.columns]:
        schedule_metric = "cloud_adv" if metric.startswith("disc_") else metric
        weight_col = f"w_{schedule_metric}"
        target_weight = safe_float(loss_cfg.get(schedule_metric, math.nan))
        start = int(schedule.get(schedule_metric, {}).get("start_epoch", 1))
        ramp = int(schedule.get(schedule_metric, {}).get("ramp_epochs", 0))
        current_weight = safe_float(df[weight_col].dropna().iloc[-1]) if weight_col in df.columns and not df[weight_col].dropna().empty else math.nan
        if metric.startswith("disc_"):
            active_rows = df[df[metric].notna()]
        else:
            active_rows = df[df[weight_col].fillna(0.0) > 0] if weight_col in df.columns else pd.DataFrame()
        if max_epoch < start:
            status = "not_started"
        elif metric.startswith("disc_") and active_rows.empty:
            status = "not_started"
        elif ramp > 0 and current_weight < target_weight:
            status = "warming_up"
        elif target_weight == 0:
            status = "disabled"
        else:
            status = "active"
        rows.append(
            {
                "metric": metric,
                "target_weight": target_weight,
                "current_weight": current_weight,
                "start_epoch": start,
                "ramp_epochs": ramp,
                "active_epoch_count": int(active_rows["epoch"].nunique()) if not active_rows.empty else 0,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def loss_contributions(ctx: RunContext, df: pd.DataFrame) -> pd.DataFrame:
    loss_cfg = ctx.config.get("loss", {})
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        weighted: dict[str, float] = {}
        for term in WEIGHTED_LOSS_TERMS:
            if term not in df.columns:
                continue
            value = safe_float(row.get(term))
            if not math.isfinite(value):
                continue
            weight = safe_float(row.get(f"w_{term}"))
            if not math.isfinite(weight):
                weight = safe_float(loss_cfg.get(term, 0.0))
            if not math.isfinite(weight):
                weight = 0.0
            weighted[term] = weight * value
        recon = safe_float(row.get("recon_total"))
        if not math.isfinite(recon):
            recon = sum(value for term, value in weighted.items() if term != "cloud_adv")
        gan = safe_float(row.get("gan_total"))
        if not math.isfinite(gan):
            gan = weighted.get("cloud_adv", 0.0)
        total = safe_float(row.get("total"))
        if not math.isfinite(total):
            total = recon + gan
        for term, contribution in weighted.items():
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "split": row["split"],
                    "term": term,
                    "raw_value": safe_float(row.get(term)),
                    "weight": safe_float(row.get(f"w_{term}")),
                    "weighted_contribution": contribution,
                    "share_of_recon_total": contribution / recon if term != "cloud_adv" and abs(recon) > 1.0e-12 else math.nan,
                    "share_of_total": contribution / total if abs(total) > 1.0e-12 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_metric_curves(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    def _plot(cols: list[str], title: str, filename: str, logy: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        plotted = False
        for col in cols:
            if col not in df.columns or df[col].dropna().empty:
                continue
            for split, part in df.groupby("split"):
                values = part[["epoch", col]].dropna().sort_values("epoch")
                if values.empty:
                    continue
                ax.plot(values["epoch"], values[col], marker="o", linewidth=1.8, label=f"{split}:{col}")
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    _plot(
        ["total", "recon_total", "gan_total", "cloud_l1", "shadow_removal", "shadow_mask", "feature_matching", "perceptual"],
        "Main Loss Curves",
        "loss_curves.png",
    )
    _plot(
        [
            "shadow_penumbra",
            "cloud_kl",
            "cloud_adv",
            "feature_matching",
            "disc_total",
            "disc_real_loss",
            "disc_fake_loss",
            "disc_real_gp",
        ],
        "Scheduled / GAN Loss Curves",
        "scheduled_gan_curves.png",
    )
    weight_cols = [c for c in df.columns if c.startswith("w_")]
    _plot(weight_cols, "Loss Weight Schedule Actually Used", "weight_schedule.png")


def plot_gaps(gaps: pd.DataFrame, out_dir: Path) -> None:
    if gaps.empty:
        return
    cols = [c for c in gaps.columns if c.endswith("_gap_val_minus_train")]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for col in cols:
        if gaps[col].dropna().empty:
            continue
        ax.plot(gaps["epoch"], gaps[col], marker="o", label=col.replace("_gap_val_minus_train", ""))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_title("Validation - Training Gap")
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap.png")
    plt.close(fig)


def plot_timing(timing: pd.DataFrame, out_dir: Path) -> None:
    if timing.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    for split, part in timing.groupby("split"):
        axes[0].plot(part["epoch"], part["seconds"] / 60.0, marker="o", label=split)
        axes[1].plot(part["epoch"], part["sec_per_batch"], marker="o", label=split)
    axes[0].set_title("Epoch Time")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("minutes")
    axes[1].set_title("Seconds per Batch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("seconds")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "timing.png")
    plt.close(fig)


def image_to_float_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def crop_panels(path: Path) -> dict[str, np.ndarray]:
    arr = image_to_float_rgb(path)
    h, w = arr.shape[:2]
    panels: list[np.ndarray] = []
    labels: list[str] = []
    pad = 6
    title_h = 26
    for cols in (8, 7):
        if w <= (cols + 1) * pad:
            continue
        tile = (w - (cols + 1) * pad) // cols
        if tile <= 0:
            continue
        expected_w = cols * tile + (cols + 1) * pad
        row_stride = tile + title_h + pad
        rows = (h - pad) // row_stride if h > pad else 0
        expected_h = rows * (tile + title_h) + (rows + 1) * pad
        if rows > 0 and abs(expected_w - w) <= cols and abs(expected_h - h) <= rows + 2:
            if cols == 8:
                base_labels = PANEL_LABELS_8
            else:
                base_labels = PANEL_LABELS_7  # old 7-col format
            grouped: dict[str, list[np.ndarray]] = {label: [] for label in base_labels}
            for row in range(rows):
                y0 = pad + row * row_stride + title_h
                for col, label in enumerate(base_labels):
                    x0 = pad + col * (tile + pad)
                    grouped[label].append(arr[y0 : y0 + tile, x0 : x0 + tile])
            return {label: np.concatenate(parts, axis=0) for label, parts in grouped.items() if parts}
    if h <= 320:
        old_pad = 2
        old_tile = h - 2 * old_pad
        old_cols = int((w - old_pad) // (old_tile + old_pad)) if old_tile > 0 else 0
        if old_cols in {6, 7, 8}:
            labels = {6: PANEL_LABELS_6, 7: PANEL_LABELS_7, 8: PANEL_LABELS_8}[old_cols]
            for col in range(old_cols):
                x0 = old_pad + col * (old_tile + old_pad)
                panels.append(arr[old_pad : old_pad + old_tile, x0 : x0 + old_tile])
        else:
            pad = 6
            title_h = 26
            tile = h - 2 * pad - title_h
            cols = int((w - pad) // (tile + pad)) if tile > 0 else 0
            labels = {6: PANEL_LABELS_6, 8: PANEL_LABELS_8, 7: PANEL_LABELS_7}.get(cols, PANEL_LABELS_7[:cols])
            for col in range(cols):
                x0 = pad + col * (tile + pad)
                y0 = pad + title_h
                panels.append(arr[y0 : y0 + tile, x0 : x0 + tile])
    if not panels:
        labels = ["full_image"]
        panels = [arr]
    return dict(zip(labels, panels))


def gray(panel: np.ndarray) -> np.ndarray:
    return (0.299 * panel[..., 0] + 0.587 * panel[..., 1] + 0.114 * panel[..., 2]).astype(np.float32)


def panel_stats(panel: np.ndarray) -> dict[str, float]:
    g = gray(panel)
    lap = cv2.Laplacian((g * 255.0).astype(np.uint8), cv2.CV_64F)
    return {
        "brightness_mean": float(g.mean()),
        "brightness_std": float(g.std()),
        "p01": float(np.quantile(g, 0.01)),
        "p05": float(np.quantile(g, 0.05)),
        "p95": float(np.quantile(g, 0.95)),
        "p99": float(np.quantile(g, 0.99)),
        "contrast_p95_p05": float(np.quantile(g, 0.95) - np.quantile(g, 0.05)),
        "sharpness_laplacian_var": float(lap.var()),
        "entropy": float(shannon_entropy((g * 255.0).astype(np.uint8))),
    }


def compare_panels(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    mae = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    psnr = float(peak_signal_noise_ratio(b, a, data_range=1.0)) if mse > 0 else math.inf
    try:
        ssim = float(structural_similarity(b, a, data_range=1.0, channel_axis=-1))
    except Exception:
        ssim = math.nan
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def visual_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vis_dir = run_dir / "visualizations"
    panel_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    pattern = re.compile(r"epoch_(\d+).*_(low|medium|high|heavy)\.png$")
    for path in sorted(vis_dir.glob("*.png")):
        m = pattern.search(path.name)
        epoch = int(m.group(1)) if m else math.nan
        bucket = m.group(2) if m else "unknown"
        panels = crop_panels(path)
        for name, panel in panels.items():
            row = {"file": path.name, "epoch": epoch, "bucket": bucket, "panel": name, **panel_stats(panel)}
            if name in {"cloud_mask", "hard_shadow", "soft_raw", "soft_eff"}:
                g = gray(panel)
                row["mask_mean"] = float(g.mean())
                row["mask_frac_gt_0_5"] = float((g > 0.5).mean())
            panel_rows.append(row)
        if "target" in panels:
            target = panels["target"]
            baseline = None
            if "cloudy_s2" in panels:
                baseline = compare_panels(panels["cloudy_s2"], target)["mae"]
            for name in ["stage1_output", "cloudy_s2"]:
                if name not in panels:
                    continue
                comp = compare_panels(panels[name], target)
                improvement = math.nan
                if baseline is not None and baseline > 1.0e-12:
                    improvement = 100.0 * (baseline - comp["mae"]) / baseline
                compare_rows.append(
                    {
                        "file": path.name,
                        "epoch": epoch,
                        "bucket": bucket,
                        "candidate": name,
                        **comp,
                        "mae_improvement_vs_cloudy_pct": improvement,
                    }
                )
    return pd.DataFrame(panel_rows), pd.DataFrame(compare_rows)


def plot_visuals(panel_df: pd.DataFrame, comp_df: pd.DataFrame, out_dir: Path) -> None:
    if not comp_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
        for bucket, part in comp_df[comp_df["candidate"] == "stage1_output"].groupby("bucket"):
            part = part.sort_values("epoch")
            axes[0].plot(part["epoch"], part["mae"], marker="o", label=bucket)
            axes[1].plot(part["epoch"], part["ssim"], marker="o", label=bucket)
        axes[0].set_title("Visual Proxy: Output vs Target MAE")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("MAE on RGB PNG")
        axes[1].set_title("Visual Proxy: Output vs Target SSIM")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("SSIM on RGB PNG")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "visual_proxy_curves.png")
        plt.close(fig)
    if not panel_df.empty:
        sub = panel_df[panel_df["panel"].isin(["cloudy_s2", "target", "stage1_output"])]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            for panel, part in sub.groupby("panel"):
                part = part.groupby("epoch", as_index=False)["brightness_mean"].mean()
                ax.plot(part["epoch"], part["brightness_mean"], marker="o", label=panel)
            ax.set_title("Visualization Brightness by Panel")
            ax.set_xlabel("epoch")
            ax.set_ylabel("mean luma")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "visual_brightness.png")
            plt.close(fig)


def make_contact_sheet(run_dir: Path, out_dir: Path) -> None:
    paths = sorted((run_dir / "visualizations").glob("*.png"))
    if not paths:
        return
    latest_epoch = max(int(re.search(r"epoch_(\d+)", p.name).group(1)) for p in paths if re.search(r"epoch_(\d+)", p.name))
    latest = [p for p in paths if f"epoch_{latest_epoch:04d}" in p.name]
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for path in sorted(latest):
        im = Image.open(path).convert("RGB")
        scale = min(1.0, 900.0 / im.width)
        im = im.resize((int(im.width * scale), int(im.height * scale)))
        thumbs.append(im)
        labels.append(path.name)
    if not thumbs:
        return
    pad = 8
    title_h = 24
    width = max(im.width for im in thumbs) + 2 * pad
    height = sum(im.height + title_h + pad for im in thumbs) + pad
    canvas = Image.new("RGB", (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    y = pad
    for im, label in zip(thumbs, labels):
        draw.text((pad, y), label, fill=(235, 238, 245), font=font)
        y += title_h
        canvas.paste(im, (pad, y))
        y += im.height + pad
    canvas.save(out_dir / "latest_visual_contact_sheet.png")


def checkpoint_inventory(run_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    ckpt_dir = run_dir / "checkpoints"
    rows: list[dict[str, Any]] = []
    if not ckpt_dir.exists():
        return pd.DataFrame(rows)
    best_re = re.compile(r"best_epoch_(\d+)_([A-Za-z0-9_.-]+)_([+-]?(?:\d+(?:\.\d+)?|nan|inf|-inf))\.pt$")
    max_epoch = int(df["epoch"].max()) if not df.empty else math.nan
    for path in sorted(ckpt_dir.glob("*.pt")):
        stat = path.stat()
        kind = "last" if path.name == "last.pt" else "best" if path.name.startswith("best_") else "other"
        epoch = math.nan
        metric = ""
        value = math.nan
        m = best_re.match(path.name)
        if m:
            epoch = int(m.group(1))
            metric = m.group(2)
            value = safe_float(m.group(3))
        elif path.name == "last.pt":
            epoch = max_epoch
            metric = "last"
        rows.append(
            {
                "file": path.name,
                "kind": kind,
                "epoch": epoch,
                "metric": metric,
                "value": value,
                "size_mb": stat.st_size / (1024.0 * 1024.0),
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def visual_inventory(run_dir: Path, expected_buckets: tuple[str, ...] = EXPECTED_BUCKETS) -> pd.DataFrame:
    vis_dir = run_dir / "visualizations"
    if not vis_dir.exists():
        return pd.DataFrame()
    pattern = re.compile(r"epoch_(\d+)_stage1_(low|medium|high|heavy)\.png$")
    grouped: dict[int, dict[str, Any]] = {}
    for path in sorted(vis_dir.glob("*.png")):
        m = pattern.match(path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        bucket = m.group(2)
        stat = path.stat()
        row = grouped.setdefault(
            epoch,
            {
                "epoch": epoch,
                "file_count": 0,
                "total_size_mb": 0.0,
                "buckets_present": set(),
            },
        )
        row["file_count"] += 1
        row["total_size_mb"] += stat.st_size / (1024.0 * 1024.0)
        row["buckets_present"].add(bucket)
    rows: list[dict[str, Any]] = []
    for epoch, row in sorted(grouped.items()):
        present = tuple(sorted(row["buckets_present"]))
        missing = tuple(bucket for bucket in expected_buckets if bucket not in present)
        rows.append(
            {
                "epoch": epoch,
                "file_count": row["file_count"],
                "buckets_present": ",".join(present),
                "missing_buckets": ",".join(missing),
                "complete": len(missing) == 0,
                "total_size_mb": row["total_size_mb"],
            }
        )
    return pd.DataFrame(rows)


def transition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    if train.empty:
        return pd.DataFrame()
    starts: list[int] = []
    for weight_col in ("w_cloud_adv", "w_feature_matching"):
        if weight_col in train.columns:
            active = train[train[weight_col].fillna(0.0) > 0.0]
            if not active.empty:
                starts.append(int(active["epoch"].min()))
    if not starts:
        return pd.DataFrame()
    start_epoch = min(starts)
    pre = train[train["epoch"] < start_epoch]
    early = train[(train["epoch"] >= start_epoch) & (train["epoch"] <= start_epoch + 2)]
    if pre.empty or early.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    metrics = [
        "total",
        "recon_total",
        "gan_total",
        "pixel_total",
        "perceptual_total",
        "cloud_l1",
        "cloud_adv",
        "feature_matching",
        "perceptual",
        "shadow_removal",
        "shadow_mask",
        "shadow_penumbra",
        "disc_total",
        "disc_real_loss",
        "disc_fake_loss",
        "disc_real_gp",
        "disc_real_logit",
        "disc_fake_logit",
    ]
    for metric in metrics:
        if metric not in train.columns:
            continue
        pre_values = pre[metric].dropna()
        early_values = early[metric].dropna()
        if pre_values.empty or early_values.empty:
            continue
        pre_mean = float(pre_values.mean())
        early_mean = float(early_values.mean())
        rows.append(
            {
                "transition_epoch": start_epoch,
                "metric": metric,
                "pre_epoch_min": int(pre["epoch"].min()),
                "pre_epoch_max": int(pre["epoch"].max()),
                "early_epoch_min": int(early["epoch"].min()),
                "early_epoch_max": int(early["epoch"].max()),
                "pre_mean": pre_mean,
                "early_mean": early_mean,
                "delta": early_mean - pre_mean,
                "delta_pct": pct_change(pre_mean, early_mean),
            }
        )
    return pd.DataFrame(rows)


def shadow_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    terms = ("shadow_removal", "shadow_mask", "shadow_penumbra")
    for split, part in df.groupby("split"):
        part = part.sort_values("epoch")
        for term in terms:
            if term not in part.columns:
                continue
            values = part[["epoch", term]].dropna()
            if values.empty:
                continue
            weight_col = f"w_{term}"
            weights = part[weight_col] if weight_col in part.columns else pd.Series(np.nan, index=part.index)
            active = part[weights.fillna(0.0) > 0.0] if weight_col in part.columns else part
            nonzero = values[values[term].abs() > 1.0e-12]
            zero_while_weight_positive = 0
            if weight_col in part.columns:
                zero_while_weight_positive = int(((part[term].fillna(0.0).abs() <= 1.0e-12) & (weights.fillna(0.0) > 0.0)).sum())
            latest_value = safe_float(values[term].iloc[-1])
            latest_weight = safe_float(weights.dropna().iloc[-1]) if weights.notna().any() else math.nan
            tail = values.tail(min(10, len(values)))[term]
            rows.append(
                {
                    "split": split,
                    "term": term,
                    "epoch_count": int(values["epoch"].nunique()),
                    "weight_positive_epoch_count": int(active["epoch"].nunique()) if not active.empty else 0,
                    "nonzero_epoch_count": int(nonzero["epoch"].nunique()),
                    "zero_while_weight_positive_epoch_count": zero_while_weight_positive,
                    "first_nonzero_epoch": int(nonzero["epoch"].iloc[0]) if not nonzero.empty else math.nan,
                    "latest_epoch": int(values["epoch"].iloc[-1]),
                    "latest_value": latest_value,
                    "latest_weight": latest_weight,
                    "mean": float(values[term].mean()),
                    "std": float(values[term].std(ddof=0)),
                    "tail10_std": float(tail.std(ddof=0)) if len(tail) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_transition(transition: pd.DataFrame, out_dir: Path) -> None:
    if transition.empty:
        return
    sub = transition[transition["delta_pct"].replace([np.inf, -np.inf], np.nan).notna()].copy()
    if sub.empty:
        return
    sub = sub.sort_values("delta_pct", key=lambda s: s.abs(), ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    colors = ["#b91c1c" if v > 0 else "#2563eb" for v in sub["delta_pct"]]
    ax.barh(sub["metric"], sub["delta_pct"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_title("GAN / Feature Matching Start Impact")
    ax.set_xlabel("early mean vs pre mean delta (%)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "gan_transition_impact.png")
    plt.close(fig)


def plot_checkpoint_timeline(checkpoints: pd.DataFrame, out_dir: Path) -> None:
    if checkpoints.empty:
        return
    best = checkpoints[(checkpoints["kind"] == "best") & checkpoints["epoch"].notna() & checkpoints["value"].notna()].copy()
    if best.empty:
        return
    best = best.sort_values("epoch")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.plot(best["epoch"], best["value"], marker="o", linewidth=1.8)
    ax.set_title("Saved Best Checkpoints")
    ax.set_xlabel("epoch")
    ax.set_ylabel("checkpoint metric value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "checkpoint_timeline.png")
    plt.close(fig)


def add_anomaly(rows: list[dict[str, Any]], severity: str, topic: str, message: str) -> None:
    rows.append({"severity": severity, "topic": topic, "message": message})


def find_anomalies(
    ctx: RunContext,
    df: pd.DataFrame,
    schedule: pd.DataFrame,
    timing: pd.DataFrame,
    panel_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    checkpoints: pd.DataFrame,
    visuals: pd.DataFrame,
    transition: pd.DataFrame,
    shadow_diag: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        add_anomaly(rows, "critical", "log", "train_log.csv 为空，无法评估训练。")
        return pd.DataFrame(rows)

    target_epochs = int(ctx.config.get("train", {}).get("epochs", 0) or 0)
    max_epoch = int(df["epoch"].max()) if df["epoch"].notna().any() else 0
    if target_epochs and max_epoch < target_epochs:
        add_anomaly(rows, "info", "progress", f"当前日志到 epoch {max_epoch}/{target_epochs}，结论只覆盖未完成训练。")

    duplicates = int(df.duplicated(["epoch", "split"]).sum())
    if duplicates:
        add_anomaly(rows, "warn", "log", f"存在 {duplicates} 条重复 epoch/split 记录，曲线可能被重复行影响。")

    for col in ("total", "recon_total", "gan_total", "cloud_l1"):
        if col in df.columns and df[col].isna().any():
            add_anomaly(rows, "critical", "numeric", f"`{col}` 存在 NaN，说明训练日志中核心 loss 非有限。")
    for col in [c for c in df.columns if c not in {"split"}]:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            add_anomaly(rows, "critical", "numeric", f"`{col}` 存在 inf。")

    train_epochs = set(df[df["split"] == "train"]["epoch"].dropna().astype(int).tolist())
    val_epochs = set(df[df["split"] == "val"]["epoch"].dropna().astype(int).tolist())
    missing_val = sorted(train_epochs - val_epochs)
    if missing_val:
        sample = ", ".join(str(x) for x in missing_val[:8])
        suffix = "..." if len(missing_val) > 8 else ""
        add_anomaly(rows, "warn", "validation", f"{len(missing_val)} 个训练 epoch 没有对应验证记录：{sample}{suffix}")

    if {"gan_total", "w_cloud_adv", "w_feature_matching"}.issubset(df.columns):
        non_adv_gan = df[
            (df["gan_total"].fillna(0.0) > 1.0e-6)
            & (df["w_cloud_adv"].fillna(0.0) <= 0.0)
            & (df["w_feature_matching"].fillna(0.0) <= 0.0)
        ]
        if not non_adv_gan.empty and "perceptual_total" not in df.columns:
            add_anomaly(
                rows,
                "warn",
                "loss_semantics",
                "`gan_total` 在 adversarial/FM 权重为 0 时仍非零；这通常表示 perceptual/HRF 等项被归入 gan_total 命名桶，报告时不要把它解释成纯 GAN 损失。",
            )

    if not transition.empty:
        total_row = transition[transition["metric"] == "total"]
        if not total_row.empty:
            delta = safe_float(total_row["delta_pct"].iloc[0])
            if math.isfinite(delta) and delta > 25.0:
                add_anomaly(rows, "warn", "gan_transition", f"GAN/FM 启动后早期 train total 均值上升 {delta:.1f}%，需要检查 loss 权重尺度和视觉质量。")
        recon_row = transition[transition["metric"] == "recon_total"]
        if not recon_row.empty:
            delta = safe_float(recon_row["delta_pct"].iloc[0])
            if math.isfinite(delta) and delta > 10.0:
                add_anomaly(rows, "warn", "gan_transition", f"GAN/FM 启动后 recon_total 均值上升 {delta:.1f}%，可能出现对抗项干扰重建项。")

    if {"disc_real_logit", "disc_fake_logit"}.issubset(df.columns):
        disc = df[(df["split"] == "train") & df["disc_real_logit"].notna() & df["disc_fake_logit"].notna()].copy()
        if not disc.empty:
            tail = disc.tail(min(10, len(disc)))
            gap = tail["disc_real_logit"] - tail["disc_fake_logit"]
            if float(gap.mean()) < 0.02:
                add_anomaly(rows, "info", "discriminator", "最近若干 epoch 的 real/fake logit 非常接近；判别器区分度弱或处于均衡状态，需要结合视觉和 loss 振荡判断。")
            if int((gap < 0).sum()) > 0:
                add_anomaly(rows, "warn", "discriminator", "最近若干 epoch 中出现 fake logit 高于 real logit，可能有判别器训练不足或生成器过强的局部阶段。")

    shadow_terms_enabled = False
    if not schedule.empty:
        shadow_sched = schedule[schedule["metric"].isin(["shadow_removal", "shadow_mask", "shadow_penumbra"])]
        shadow_terms_enabled = bool((shadow_sched["target_weight"].fillna(0.0) > 0.0).any())
    if shadow_terms_enabled and not shadow_diag.empty:
        for _, item in shadow_diag.iterrows():
            if safe_float(item["latest_weight"]) > 0 and int(item["nonzero_epoch_count"]) == 0:
                severity = "critical" if item["split"] == "train" else "warn"
                add_anomaly(rows, severity, "softshadow", f"{item['split']} `{item['term']}` 权重为正但全程为 0，SoftShadow 该项没有产生有效 loss。")
            if item["term"] == "shadow_penumbra" and safe_float(item["latest_weight"]) > 0 and int(item["zero_while_weight_positive_epoch_count"]) > 0:
                add_anomaly(
                    rows,
                    "warn",
                    "softshadow",
                    f"{item['split']} `shadow_penumbra` 在权重为正时有 {int(item['zero_while_weight_positive_epoch_count'])} 个 epoch 为 0；可能是软阴影 mask 饱和或没有半影像素。",
                )
        train_shadow = shadow_diag[(shadow_diag["split"] == "train") & (shadow_diag["term"].isin(["shadow_removal", "shadow_mask"]))]
        if not train_shadow.empty and (train_shadow["tail10_std"].fillna(0.0) < 1.0e-8).all():
            add_anomaly(rows, "info", "softshadow", "最近 10 个 epoch 的主要 SoftShadow loss 几乎无波动；可能已经平台期，也可能验证/样本构成固定导致。")

    if visuals.empty:
        add_anomaly(rows, "warn", "visualization", "没有找到 stage1 可视化 PNG，无法做展示图质量检查。")
    elif (visuals["complete"] == False).any():  # noqa: E712
        bad = visuals[visuals["complete"] == False]  # noqa: E712
        expected = "/".join(expected_visual_buckets(ctx.config))
        add_anomaly(rows, "warn", "visualization", f"{len(bad)} 个 epoch 缺少 {expected} 中至少一个可视化 bucket。")

    if checkpoints.empty:
        add_anomaly(rows, "warn", "checkpoint", "没有找到 checkpoint 文件。")
    else:
        best = checkpoints[(checkpoints["kind"] == "best") & checkpoints["metric"].eq("total") & checkpoints["value"].notna()]
        val = df[(df["split"] == "val") & df["total"].notna()] if "total" in df.columns else pd.DataFrame()
        if not best.empty and not val.empty:
            best_ckpt = float(best["value"].min())
            best_val = float(val["total"].min())
            if abs(best_ckpt - best_val) > 5.0e-4:
                add_anomaly(rows, "warn", "checkpoint", f"best checkpoint total={best_ckpt:.6f} 与日志 best val total={best_val:.6f} 不一致。")
        if not (checkpoints["file"] == "last.pt").any():
            add_anomaly(rows, "warn", "checkpoint", "缺少 last.pt，续训便利性较差。")

    if not comp_df.empty:
        latest = comp_df[comp_df["candidate"] == "stage1_output"].sort_values(["epoch", "bucket"])
        if not latest.empty:
            latest = latest[latest["epoch"] == latest["epoch"].max()]
            degraded = latest[latest["mae_improvement_vs_cloudy_pct"] < 0]
            if len(degraded) == len(latest):
                add_anomaly(rows, "warn", "visual_proxy", "最新可视化中所有 bucket 的 Stage1 RGB proxy MAE 都没有优于 cloudy 输入。")
            elif not degraded.empty:
                buckets = ", ".join(degraded["bucket"].astype(str).tolist())
                add_anomaly(rows, "info", "visual_proxy", f"最新可视化中这些 bucket 的 Stage1 RGB proxy MAE 未优于 cloudy 输入：{buckets}")

    if not timing.empty:
        train_t = timing[timing["split"] == "train"]
        if not train_t.empty and train_t["sec_per_batch"].notna().any():
            p95 = float(train_t["sec_per_batch"].quantile(0.95))
            med = float(train_t["sec_per_batch"].median())
            if med > 0 and p95 / med > 1.5:
                add_anomaly(rows, "info", "speed", f"训练 sec/batch 的 p95/median={p95 / med:.2f}，存在明显耗时波动。")

    if not rows:
        add_anomaly(rows, "info", "summary", "未发现脚本规则覆盖范围内的硬性异常。")
    return pd.DataFrame(rows)


def fmt(x: Any, nd: int = 4) -> str:
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """Render a compact markdown table without pandas' optional tabulate dependency."""
    if df.empty:
        return ""

    def cell(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [str(c) for c in df.columns]
    rows = [[cell(value) for value in row] for row in df.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([render_row(headers), sep, *[render_row(row) for row in rows]])


def write_report(
    ctx: RunContext,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    schedule: pd.DataFrame,
    contributions: pd.DataFrame,
    timing: pd.DataFrame,
    panel_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    checkpoints: pd.DataFrame,
    visual_inv: pd.DataFrame,
    transition: pd.DataFrame,
    shadow_diag: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> Path:
    out_path = ctx.out_dir / "training_run_report.md"
    max_epoch = int(df["epoch"].max()) if not df.empty else 0
    target_epochs = int(ctx.config.get("train", {}).get("epochs", 0) or 0)
    completed_pct = 100.0 * max_epoch / target_epochs if target_epochs else math.nan
    report_metric = "recon_total" if "recon_total" in df.columns and df["recon_total"].notna().any() else "total"
    val_total = df[(df["split"] == "val") & df[report_metric].notna()].sort_values("epoch")
    train_total = df[(df["split"] == "train") & df[report_metric].notna()].sort_values("epoch")
    best_epoch = int(val_total.loc[val_total[report_metric].idxmin(), "epoch"]) if not val_total.empty else None
    best_val = float(val_total[report_metric].min()) if not val_total.empty else math.nan
    val_improve = improvement_pct(float(val_total[report_metric].iloc[0]), float(val_total[report_metric].iloc[-1])) if len(val_total) >= 2 else math.nan
    train_improve = improvement_pct(float(train_total[report_metric].iloc[0]), float(train_total[report_metric].iloc[-1])) if len(train_total) >= 2 else math.nan
    final_gap = math.nan
    gap_col = f"{report_metric}_gap_val_minus_train"
    if not gaps.empty and gap_col in gaps:
        final_gap = float(gaps.sort_values("epoch")[gap_col].iloc[-1])

    not_started = schedule[schedule["status"] == "not_started"]["metric"].tolist() if not schedule.empty else []
    active = schedule[schedule["status"].isin(["active", "warming_up"])]["metric"].tolist() if not schedule.empty else []
    shadow_terms_enabled = False
    if not schedule.empty:
        shadow_sched = schedule[schedule["metric"].isin(["shadow_removal", "shadow_mask", "shadow_penumbra"])]
        shadow_terms_enabled = bool((shadow_sched["target_weight"].fillna(0.0) > 0.0).any())

    latest_visual = comp_df[comp_df["candidate"] == "stage1_output"].sort_values(["epoch", "bucket"]) if not comp_df.empty else pd.DataFrame()
    if not latest_visual.empty:
        latest_visual = latest_visual[latest_visual["epoch"] == latest_visual["epoch"].max()]

    lines: list[str] = []
    lines.append(f"# Training Run Report: `{ctx.run_dir.name}`")
    lines.append("")
    lines.append("## 1. Run 状态")
    lines.append(f"- Run dir: `{ctx.run_dir}`")
    lines.append(f"- 已记录 epoch: `{max_epoch}` / 配置 epoch: `{target_epochs}` ({fmt(completed_pct, 2)}%)")
    lines.append(f"- Train batches: `{ctx.train_batches}` | Val batches: `{ctx.val_batches}`")
    lines.append(f"- 报告主指标: `{report_metric}`")
    lines.append(f"- Best val {report_metric}: `{fmt(best_val)}` at epoch `{best_epoch}`")
    lines.append(f"- Train {report_metric} improvement: `{fmt(train_improve, 2)}%` | Val {report_metric} improvement: `{fmt(val_improve, 2)}%`")
    lines.append(f"- Final generalization gap (val - train {report_metric}): `{fmt(final_gap)}`")
    if max_epoch < target_epochs:
        if math.isfinite(completed_pct) and completed_pct >= 80.0:
            lines.append("- 结论：当前日志已接近完整训练，但还不是最终完成状态；可评价主要训练趋势，最终 checkpoint 和末尾收敛仍需等训练结束后复核。")
        elif math.isfinite(completed_pct) and completed_pct >= 50.0:
            lines.append("- 结论：当前日志是中后期训练片段，可评价主要调度启动后的稳定性，但不能替代完整收敛结论。")
        else:
            lines.append("- 结论：当前日志是早期训练片段，只能评价 warm-up 初期趋势，不能评价完整收敛、GAN 稳定性或最终泛化。")
    lines.append("")

    lines.append("## 2. 核心异常与风险")
    if anomalies.empty:
        lines.append("- 未发现脚本规则覆盖范围内的硬性异常。")
    else:
        severity_order = {"critical": 0, "warn": 1, "info": 2}
        show_anom = anomalies.copy()
        show_anom["_rank"] = show_anom["severity"].map(severity_order).fillna(9)
        show_anom = show_anom.sort_values(["_rank", "topic"]).drop(columns=["_rank"])
        lines.append(dataframe_to_markdown(show_anom, floatfmt=".6f"))
    lines.append("")

    lines.append("## 3. Loss 调度状态")
    if active:
        lines.append(f"- 已参与训练的主要项: `{', '.join(active)}`")
    if not_started:
        lines.append(f"- 尚未开始的项: `{', '.join(not_started)}`")
        lines.append("- 注意：尚未开始的 loss 不能根据当前数值判断有效或无效，只能说明调度还没走到对应阶段。")
    lines.append("")
    if not schedule.empty:
        lines.append(dataframe_to_markdown(schedule, floatfmt=".6f"))
        lines.append("")

    if not transition.empty:
        lines.append("## 4. GAN/FM 启动冲击")
        lines.append("- 这里比较 adversarial 或 feature matching 权重首次大于 0 之前的 train 均值，与启动后前 3 个 epoch 的 train 均值。")
        show_trans = transition.copy()
        show_trans["_abs_delta_pct"] = show_trans["delta_pct"].abs()
        show_trans = show_trans.sort_values("_abs_delta_pct", ascending=False).drop(columns=["_abs_delta_pct"]).head(16)
        cols = ["transition_epoch", "metric", "pre_mean", "early_mean", "delta", "delta_pct"]
        lines.append(dataframe_to_markdown(show_trans[cols], floatfmt=".6f"))
        lines.append("")

    if shadow_terms_enabled and not shadow_diag.empty:
        lines.append("## 5. SoftShadow 诊断")
        lines.append("- 这部分只基于 epoch 聚合日志，不能直接证明每个 batch 都有梯度；但可以判断 loss 项是否长期为 0、是否在权重为正时失效。")
        cols = [
            "split",
            "term",
            "weight_positive_epoch_count",
            "nonzero_epoch_count",
            "zero_while_weight_positive_epoch_count",
            "first_nonzero_epoch",
            "latest_value",
            "latest_weight",
            "tail10_std",
        ]
        lines.append(dataframe_to_markdown(shadow_diag[cols], floatfmt=".6f"))
        lines.append("")

    lines.append("## 6. 指标趋势摘要")
    key_metrics = [
        "total",
        "recon_total",
        "gan_total",
        "cloud_l1",
        "known_l1",
        "shadow_removal",
        "shadow_mask",
        "cloud_kl",
        "cloud_adv",
        "disc_total",
    ]
    show = summary[summary["metric"].isin(key_metrics)].copy()
    if not show.empty:
        cols = ["split", "metric", "first", "last", "best_epoch", "best", "improvement_pct_if_lower_better", "num_points"]
        lines.append(dataframe_to_markdown(show[cols], floatfmt=".6f"))
    lines.append("")
    lines.append("诊断：")
    if val_improve > 0:
        lines.append(f"- Val {report_metric} 从 epoch 1 到当前下降 `{fmt(val_improve, 2)}%`，早期优化方向是正常的。")
    if math.isfinite(final_gap) and final_gap < 0:
        lines.append(f"- 当前 val {report_metric} 低于 train {report_metric}，这通常不是过拟合，可能来自训练/验证样本云量分布差异、训练 batch 的难度更高或训练态正则影响。")
    elif math.isfinite(final_gap) and final_gap > 0:
        lines.append(f"- 当前 val {report_metric} 高于 train {report_metric}，需要继续观察 gap 是否扩大；若持续扩大才考虑过拟合。")
    lines.append("")

    if not contributions.empty:
        lines.append("## 7. Loss 贡献比例")
        latest_epoch = int(contributions["epoch"].max())
        latest_contrib = contributions[contributions["epoch"] == latest_epoch].copy()
        latest_contrib = latest_contrib[latest_contrib["weighted_contribution"].abs() > 0]
        if not latest_contrib.empty:
            latest_contrib = latest_contrib.sort_values(["split", "share_of_total"], ascending=[True, False])
            cols = ["split", "term", "weighted_contribution", "share_of_recon_total", "share_of_total"]
            lines.append(dataframe_to_markdown(latest_contrib[cols], floatfmt=".6f"))
        lines.append("")

    lines.append("## 8. 训练速度")
    if not timing.empty:
        speed_rows: list[dict[str, Any]] = []
        for split, part in timing.groupby("split"):
            seconds = part["seconds"].dropna()
            spb = part["sec_per_batch"].dropna()
            speed_rows.append(
                {
                    "split": split,
                    "epochs": int(part["epoch"].nunique()),
                    "seconds_mean": float(seconds.mean()) if not seconds.empty else math.nan,
                    "seconds_median": float(seconds.median()) if not seconds.empty else math.nan,
                    "seconds_p95": float(seconds.quantile(0.95)) if not seconds.empty else math.nan,
                    "sec_per_batch_mean": float(spb.mean()) if not spb.empty else math.nan,
                    "sec_per_batch_median": float(spb.median()) if not spb.empty else math.nan,
                    "sec_per_batch_p95": float(spb.quantile(0.95)) if not spb.empty else math.nan,
                    "latest_sec_per_batch": float(spb.iloc[-1]) if not spb.empty else math.nan,
                }
            )
        lines.append(dataframe_to_markdown(pd.DataFrame(speed_rows), floatfmt=".3f"))
        train_t = timing[timing["split"] == "train"]
        if not train_t.empty:
            lines.append(f"- 平均训练耗时: `{fmt(train_t['seconds'].mean() / 60.0, 2)}` min/epoch, `{fmt(train_t['sec_per_batch'].mean(), 3)}` sec/batch")
        lines.append("- 完整逐 epoch 时间表见 `timing_summary.csv`。")
    else:
        lines.append("- 未从 train.log 解析到时间信息。")
    lines.append("")

    lines.append("## 9. 可视化结果评价")
    lines.append("- 评价对象是保存的 RGB PNG，可判断展示质量和视觉趋势；这不是完整 13 波段 checkpoint 定量评测。")
    if not visual_inv.empty:
        complete_count = int(visual_inv["complete"].sum())
        expected = "/".join(expected_visual_buckets(ctx.config))
        lines.append(f"- 可视化 epoch 覆盖: `{len(visual_inv)}` 个；配置要求的 `{expected}` 完整覆盖: `{complete_count}` 个。")
        show_vis = visual_inv.tail(12)
        lines.append(dataframe_to_markdown(show_vis, floatfmt=".6f"))
        lines.append("")
    if not latest_visual.empty:
        cols = ["bucket", "mae", "psnr", "ssim", "mae_improvement_vs_cloudy_pct"]
        lines.append(dataframe_to_markdown(latest_visual[cols], floatfmt=".6f"))
    if not panel_df.empty:
        latest_panel = panel_df[panel_df["epoch"] == panel_df["epoch"].max()]
        panel_show = latest_panel[latest_panel["panel"].isin(["cloudy_s2", "target", "stage1_output"])]
        if not panel_show.empty:
            cols = ["bucket", "panel", "brightness_mean", "contrast_p95_p05", "sharpness_laplacian_var", "entropy"]
            lines.append("")
            lines.append("Latest visualization panel quality:")
            lines.append(dataframe_to_markdown(panel_show[cols], floatfmt=".6f"))
    lines.append("")
    lines.append("可视化诊断：")
    if not panel_df.empty:
        old_no_title = any(Image.open(ctx.run_dir / "visualizations" / f).size[1] <= 270 for f in panel_df["file"].drop_duplicates())
        if old_no_title:
            lines.append("- 当前 run 的 PNG 是旧版 make_grid 输出，未带标题；后续训练脚本已改为带标题和显示拉伸。")
    if not latest_visual.empty and (latest_visual["mae_improvement_vs_cloudy_pct"] < 0).any():
        lines.append("- 至少一个云量 bucket 中 Stage1 Output 的 RGB proxy MAE 没有优于 Cloudy 输入；需要用后续 epoch 和真实 val/test 指标确认。")
    lines.append("- 建议同时查看 `latest_visual_contact_sheet.png`，用肉眼确认云区、阴影区和 hard mask 边界是否存在接缝或过暗问题。")
    lines.append("")

    lines.append("## 10. Checkpoint 完整性")
    if checkpoints.empty:
        lines.append("- 未发现 checkpoint。")
    else:
        show_ckpt = checkpoints.sort_values(["kind", "epoch", "file"]).copy()
        lines.append(dataframe_to_markdown(show_ckpt, floatfmt=".6f"))
    lines.append("")

    lines.append("## 11. 生成文件")
    for p in sorted(ctx.out_dir.glob("*")):
        lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append("## 12. 后续建议")
    if "cloud_adv" in not_started or "disc_total" in not_started:
        lines.append("- 继续训练到 `cloud_adv` 调度启动之后，才能评价 GAN 对云区纹理的影响。")
    else:
        lines.append("- `cloud_adv` 和判别器已经启动，应重点检查 GAN 开启前后 cloud 区视觉质量与重建指标是否劣化。")
    pending_regularizers = [name for name in ("cloud_kl",) if name in not_started]
    if shadow_terms_enabled and "shadow_penumbra" in not_started:
        pending_regularizers.append("shadow_penumbra")
    if pending_regularizers:
        lines.append(f"- 继续训练到 `{', '.join(pending_regularizers)}` 调度启动之后，再评价对应正则项。")
    else:
        if shadow_terms_enabled:
            lines.append("- `cloud_kl` 与 `shadow_penumbra` 已经启动，应结合曲线判断它们是否只增加 loss 尺度而没有改善可视化质量。")
        else:
            lines.append("- `cloud_kl` 已按当前配置启动后，应结合曲线判断它是否只增加 loss 尺度而没有改善光谱质量。")
    lines.append("- 若要严谨报告模型性能，应再运行 checkpoint 级别评测脚本，计算 global/clear/shadow/cloud 的 MAE、RMSE、PSNR，并补充 RGB SSIM/SAM。")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ALLClear training run from logs and visualizations.")
    parser.add_argument(
        "--run-dir",
        default="outputs/allclear/2026-06-26T21-52-22_stage1_allclear_tgdad_softshadow_lama_pix2pixhd_r1",
        help="Training run directory containing train_log.csv and visualizations/.",
    )
    parser.add_argument("--out-dir", default=None, help="Output analysis directory. Default: <run-dir>/analysis.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(run_dir / "config.resolved.json")
    latest = load_json(run_dir / "metrics" / "latest.json")
    df = load_log(run_dir)
    timing, train_batches, val_batches = parse_train_log(run_dir)
    ctx = RunContext(run_dir=run_dir, out_dir=out_dir, config=config, latest=latest, train_batches=train_batches, val_batches=val_batches)

    df.to_csv(out_dir / "training_log_numeric.csv", index=False)
    summary = metric_summary(df)
    summary.to_csv(out_dir / "metric_summary.csv", index=False)
    gaps = generalization_gaps(df)
    gaps.to_csv(out_dir / "generalization_gaps.csv", index=False)
    schedule = schedule_status(ctx, df)
    schedule.to_csv(out_dir / "loss_schedule_status.csv", index=False)
    contributions = loss_contributions(ctx, df)
    contributions.to_csv(out_dir / "loss_contributions.csv", index=False)
    if not timing.empty:
        timing.to_csv(out_dir / "timing_summary.csv", index=False)

    checkpoints = checkpoint_inventory(run_dir, df)
    checkpoints.to_csv(out_dir / "checkpoint_inventory.csv", index=False)
    visual_inv = visual_inventory(run_dir, expected_visual_buckets(config))
    visual_inv.to_csv(out_dir / "visualization_inventory.csv", index=False)
    transition = transition_analysis(df)
    transition.to_csv(out_dir / "gan_feature_matching_transition.csv", index=False)
    shadow_diag = shadow_diagnostics(df)
    shadow_diag.to_csv(out_dir / "softshadow_diagnostics.csv", index=False)

    plot_metric_curves(df, out_dir)
    plot_gaps(gaps, out_dir)
    plot_timing(timing, out_dir)
    plot_transition(transition, out_dir)
    plot_checkpoint_timeline(checkpoints, out_dir)

    panel_df, comp_df = visual_metrics(run_dir)
    panel_df.to_csv(out_dir / "visual_panel_quality.csv", index=False)
    comp_df.to_csv(out_dir / "visual_candidate_comparison.csv", index=False)
    plot_visuals(panel_df, comp_df, out_dir)
    make_contact_sheet(run_dir, out_dir)

    anomalies = find_anomalies(ctx, df, schedule, timing, panel_df, comp_df, checkpoints, visual_inv, transition, shadow_diag)
    anomalies.to_csv(out_dir / "anomalies.csv", index=False)

    if run_dir.joinpath("train.log").exists():
        shutil.copy2(run_dir / "train.log", out_dir / "source_train.log")
    if run_dir.joinpath("config.resolved.json").exists():
        shutil.copy2(run_dir / "config.resolved.json", out_dir / "source_config.resolved.json")
    if run_dir.joinpath("metrics", "latest.json").exists():
        shutil.copy2(run_dir / "metrics" / "latest.json", out_dir / "source_latest.json")
    report = write_report(
        ctx,
        df,
        summary,
        gaps,
        schedule,
        contributions,
        timing,
        panel_df,
        comp_df,
        checkpoints,
        visual_inv,
        transition,
        shadow_diag,
        anomalies,
    )
    print(f"analysis_dir={out_dir}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
