#!/usr/bin/env python3
"""Audit SoftShadow division masks and bbox prompts for ALLClear.

The script is intentionally independent from model checkpoints. It evaluates
whether offline SoftShadow supervision files are usable before SAM-LoRA
training:

- ``division_mask`` / ``sam_mask`` range, coverage, softness, and leakage;
- bbox validity and coverage of hard shadow / division-mask support;
- agreement with the paired-image brightness-ratio soft target;
- low/medium/high/heavy cloud-bucket visual overlays.
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.config import load_config
from src.allclear.dataset import AllClearDataset
from src.allclear.modules.common import bbox_from_mask, masks_from_cld_shdw
from src.allclear.modules.softshadow import soft_shadow_division_target, soft_shadow_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training/eval YAML or config.resolved.json.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--splits", nargs="+", default=None, choices=["train", "val", "test"], help="Audit multiple splits into output-dir/<split>.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/allclear_softshadow_prompt_audit"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--softshadow-mask-dir", type=Path, default=None, help="Override data.softshadow_mask_dir.")
    parser.add_argument("--softshadow-bbox-path", type=Path, default=None, help="Override data.softshadow_bbox_path.")
    parser.add_argument("--bbox-space", default=None, choices=["image", "sam_input"], help="Override model.softshadow_bbox_space.")
    parser.add_argument("--sam-input-size", type=int, default=None, help="Override model.softshadow_sam_input_size.")
    parser.add_argument("--positive-threshold", type=float, default=0.05)
    parser.add_argument("--hard-threshold", type=float, default=0.50)
    parser.add_argument("--reference-target-mode", default="paper_ratio", choices=["paper_ratio", "hard_support"])
    parser.add_argument("--reference-low-pass-kernel", type=int, default=5)
    parser.add_argument("--reference-division-threshold", type=float, default=0.05)
    parser.add_argument("--visual-samples-per-bucket", type=int, default=3)
    parser.add_argument(
        "--visual-buckets",
        nargs="+",
        default=["low", "medium"],
        choices=["low", "medium", "high", "heavy"],
        help="Cloud buckets to visualize. Prompt auditing skips high/heavy by default because high-cloud samples often contain no exclusive shadow.",
    )
    parser.add_argument(
        "--min-visual-shadow-frac",
        type=float,
        default=0.002,
        help="Only save visualizations for samples with at least this exclusive non-cloud shadow fraction.",
    )
    parser.add_argument(
        "--min-visual-division-pos-frac",
        type=float,
        default=0.0,
        help="Optional lower bound on positive division-mask area for visualization.",
    )
    parser.add_argument("--fallback-online-mask", action="store_true", help="Use paired-image soft target when offline sam_mask is missing.")
    parser.add_argument("--fallback-bbox-from-mask", action="store_true", help="Use hard-shadow bbox when offline bbox is missing.")
    parser.add_argument(
        "--allow-missing-offline",
        action="store_true",
        help="Allow empty offline sam_mask/bbox inputs. This is only useful for debugging missing-data behavior.",
    )
    return parser.parse_args()


def assert_not_placeholder_path(path: Path | None, name: str) -> None:
    if path is None:
        return
    text = str(path)
    if text.startswith("/path/to/") or text == "/path/to":
        raise SystemExit(
            f"{name} is still a placeholder: {text}\n"
            "Please pass the real SoftShadow preprocessing output path, or omit the argument and use "
            "--fallback-online-mask/--fallback-bbox-from-mask only for a non-official smoke check."
        )


def manifest_has_any_column(path: str | Path, keys: tuple[str, ...]) -> bool:
    manifest = Path(path).expanduser()
    if not manifest.is_absolute():
        return False
    if not manifest.exists():
        return False
    with manifest.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return any(key in header for key in keys)


def require_offline_or_fallback(cfg: dict[str, Any], split: str, args: argparse.Namespace) -> None:
    if args.allow_missing_offline:
        return
    data = cfg.get("data", {})
    manifest = data.get(f"{split}_manifest")
    has_mask = bool(data.get(f"softshadow_mask_dir_{split}") or data.get("softshadow_mask_dir"))
    has_bbox = bool(data.get(f"softshadow_bbox_path_{split}") or data.get("softshadow_bbox_path"))
    if manifest:
        has_mask = has_mask or manifest_has_any_column(manifest, AllClearDataset.SOFTSHADOW_MASK_KEYS)
        has_bbox = has_bbox or manifest_has_any_column(manifest, AllClearDataset.SOFTSHADOW_BBOX_KEYS)
    missing = []
    if not has_mask and not args.fallback_online_mask:
        missing.append("division mask: set data.softshadow_mask_dir[_split] or pass --softshadow-mask-dir")
    if not has_bbox and not args.fallback_bbox_from_mask:
        missing.append("bbox YAML: set data.softshadow_bbox_path[_split] or pass --softshadow-bbox-path")
    if missing:
        raise SystemExit(
            "No official SoftShadow offline prompt inputs were configured:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\nRun scripts/precompute_allclear_softshadow_prompts.py first, or use fallback flags only for a non-official smoke check."
        )


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


def cloud_bucket(value: float) -> str:
    if value < 0.1:
        return "low"
    if value < 0.4:
        return "medium"
    if value < 0.9:
        return "high"
    return "heavy"


def masked_mean(x: Tensor, mask: Tensor) -> float:
    mask = mask.float()
    denom = float(mask.sum().item())
    if denom < 1.0:
        return math.nan
    return float((x.float() * mask).sum().item() / denom)


def safe_div(num: float, den: float) -> float:
    return num / den if den > 1.0e-12 else math.nan


def pearson_corr(a: Tensor, b: Tensor, mask: Tensor) -> float:
    mask_bool = mask[0] > 0.5 if mask.ndim == 3 else mask > 0.5
    av = a.reshape(-1)[mask_bool.reshape(-1)]
    bv = b.reshape(-1)[mask_bool.reshape(-1)]
    if av.numel() < 2:
        return math.nan
    av = av.float()
    bv = bv.float()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = av.square().mean().sqrt() * bv.square().mean().sqrt()
    return float((av * bv).mean().div(denom.clamp_min(1.0e-8)).item())


def boundary(mask: Tensor) -> Tensor:
    m = mask.float().clamp(0.0, 1.0)
    dil = F.max_pool2d(m.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    ero = -F.max_pool2d((-m).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    return (dil - ero).clamp(0.0, 1.0)


def bbox_to_image_coords(bbox: Tensor, h: int, w: int, bbox_space: str, sam_input_size: int) -> Tensor:
    box = bbox.float().reshape(-1, 4)[0].clone()
    if str(bbox_space).lower() in {"sam", "sam_input", "input", "input_size"}:
        box[[0, 2]] *= float(w) / float(sam_input_size)
        box[[1, 3]] *= float(h) / float(sam_input_size)
    elif str(bbox_space).lower() not in {"image", "original", "source"}:
        raise ValueError("bbox_space must be image or sam_input")
    box[[0, 2]] = box[[0, 2]].clamp(0.0, float(w - 1))
    box[[1, 3]] = box[[1, 3]].clamp(0.0, float(h - 1))
    return box


def bbox_mask_from_xyxy(box: Tensor, h: int, w: int) -> Tensor:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    out = torch.zeros((1, h, w), dtype=torch.float32)
    if x2 <= x1 or y2 <= y1:
        return out
    out[:, max(0, y1) : min(h, y2 + 1), max(0, x1) : min(w, x2 + 1)] = 1.0
    return out


def render_rgb(x: Tensor, rgb_indices: tuple[int, int, int]) -> Image.Image:
    rgb = x[list(rgb_indices)].float().clamp(0.0, 1.0)
    lo = torch.quantile(rgb.flatten(1), 0.02, dim=1).view(3, 1, 1)
    hi = torch.quantile(rgb.flatten(1), 0.98, dim=1).view(3, 1, 1)
    rgb = ((rgb - lo) / (hi - lo).clamp_min(1.0e-4)).clamp(0.0, 1.0)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def render_mask(mask: Tensor, color: tuple[int, int, int]) -> Image.Image:
    m = mask[0].float().clamp(0.0, 1.0)
    arr = torch.zeros((m.shape[0], m.shape[1], 3), dtype=torch.uint8)
    for i, c in enumerate(color):
        arr[..., i] = (m * float(c)).round().to(torch.uint8)
    return Image.fromarray(arr.numpy(), mode="RGB")


def overlay_mask(base: Image.Image, mask: Tensor, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = base.convert("RGBA")
    m = mask[0].float().clamp(0.0, 1.0)
    rgba = torch.zeros((m.shape[0], m.shape[1], 4), dtype=torch.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (m * float(alpha) * 255.0).round().to(torch.uint8)
    return Image.alpha_composite(base, Image.fromarray(rgba.numpy(), mode="RGBA")).convert("RGB")


def draw_bbox(image: Image.Image, box: Tensor, color: tuple[int, int, int] = (255, 60, 60), width: int = 3) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    return out


def save_grid(panels: list[tuple[str, Image.Image]], path: Path, tile: int = 192) -> None:
    title_h = 22
    pad = 3
    cols = 4
    rows = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (cols * tile + (cols + 1) * pad, rows * (tile + title_h) + (rows + 1) * pad), (18, 24, 32))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for idx, (title, image) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = pad + c * (tile + pad)
        y = pad + r * (tile + title_h + pad)
        draw.rectangle((x, y, x + tile, y + title_h), fill=(31, 41, 55))
        draw.text((x + 6, y + 5), title[:36], fill=(240, 244, 248), font=font)
        canvas.paste(image.resize((tile, tile), Image.BILINEAR), (x, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def sample_metrics(
    item: dict[str, Any],
    cfg: dict[str, Any],
    *,
    bbox_space: str,
    sam_input_size: int,
    positive_threshold: float,
    hard_threshold: float,
    fallback_online_mask: bool,
    fallback_bbox_from_mask: bool,
    reference_target_mode: str,
    reference_low_pass_kernel: int,
    reference_division_threshold: float,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    shadow_index = int(cfg.get("data", {}).get("shadow_index", 3))
    cloud_index = int(cfg.get("data", {}).get("cloud_index", 1))
    rgb_indices = tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    cloudy = item["s2_toa"].float()
    target = item["target"].float()
    masks = masks_from_cld_shdw(item["cld_shdw"].unsqueeze(0), shadow_index=shadow_index, cloud_index=cloud_index)
    shadow = masks.shadow[0].float()
    cloud = masks.cloud[0].float()
    clear = masks.clear[0].float()
    h, w = shadow.shape[-2:]

    has_offline_mask = "sam_mask" in item
    if has_offline_mask:
        division = item["sam_mask"].float()
        if division.shape[-2:] != (h, w):
            division = F.interpolate(division.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        division = division[:1].clamp(0.0, 1.0)
    elif fallback_online_mask:
        if reference_target_mode == "paper_ratio":
            division = soft_shadow_division_target(
                cloudy.unsqueeze(0),
                target.unsqueeze(0),
                rgb_indices=rgb_indices,
                low_pass_kernel=reference_low_pass_kernel,
                threshold=reference_division_threshold,
            )[0].float()
        else:
            division = soft_shadow_target(
                cloudy.unsqueeze(0),
                target.unsqueeze(0),
                shadow.unsqueeze(0),
                rgb_indices=rgb_indices,
                low_pass_kernel=reference_low_pass_kernel,
            )[0].float()
    else:
        division = shadow.new_zeros(shadow.shape)

    has_offline_bbox = "bbox" in item
    if has_offline_bbox:
        bbox = bbox_to_image_coords(item["bbox"].float(), h, w, bbox_space, sam_input_size)
    elif fallback_bbox_from_mask:
        bbox = bbox_from_mask(shadow.unsqueeze(0), pad=4)[0].cpu()
    else:
        bbox = torch.tensor([math.nan, math.nan, math.nan, math.nan], dtype=torch.float32)
    bbox_valid = bool(torch.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1])
    box_mask = bbox_mask_from_xyxy(bbox, h, w) if bbox_valid else torch.zeros_like(shadow)

    pos = (division > float(positive_threshold)).float()
    hard = (division > float(hard_threshold)).float()
    shadow_dil = F.max_pool2d(shadow.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
    div_boundary = boundary(hard)
    hard_boundary = boundary(shadow)

    pos_sum = float(pos.sum().item())
    hard_sum = float(hard.sum().item())
    shadow_sum = float(shadow.sum().item())
    cloud_sum = float(cloud.sum().item())
    clear_sum = float(clear.sum().item())
    bbox_sum = float(box_mask.sum().item())
    div_in_shadow = float((pos * shadow_dil).sum().item())
    shadow_hit = float((pos * shadow).sum().item())
    union = float(((pos + shadow) > 0.5).float().sum().item())
    boundary_inter = float((div_boundary * hard_boundary).sum().item())
    boundary_union = float(((div_boundary + hard_boundary) > 0.5).float().sum().item())

    if reference_target_mode == "paper_ratio":
        reference = soft_shadow_division_target(
            cloudy.unsqueeze(0),
            target.unsqueeze(0),
            rgb_indices=rgb_indices,
            low_pass_kernel=reference_low_pass_kernel,
            threshold=reference_division_threshold,
        )[0].float()
    else:
        reference = soft_shadow_target(
            cloudy.unsqueeze(0),
            target.unsqueeze(0),
            shadow.unsqueeze(0),
            rgb_indices=rgb_indices,
            low_pass_kernel=reference_low_pass_kernel,
        )[0].float()
    diff_reference = (division - reference).abs()

    row = {
        "sample_id": item.get("sample_id", ""),
        "shadow_case": int(item["shadow_case"].item()) if "shadow_case" in item else -1,
        "shadow_case_name": {0: "no_shadow", 1: "valid_shadow", 2: "ambiguous"}.get(
            int(item["shadow_case"].item()) if "shadow_case" in item else -1,
            "disabled",
        ),
        "has_offline_division_mask": int(has_offline_mask),
        "has_offline_bbox": int(has_offline_bbox),
        "cloud_frac": cloud_sum / float(h * w),
        "shadow_frac": shadow_sum / float(h * w),
        "clear_frac": clear_sum / float(h * w),
        "cloud_bucket": cloud_bucket(cloud_sum / float(h * w)),
        "division_min": float(division.min().item()),
        "division_max": float(division.max().item()),
        "division_mean": float(division.mean().item()),
        "division_pos_frac": pos_sum / float(h * w),
        "division_hard_frac": hard_sum / float(h * w),
        "division_soft_frac": float(((division > 0.05) & (division < 0.95)).float().mean().item()),
        "division_shadow_precision_dilated": safe_div(div_in_shadow, pos_sum),
        "division_shadow_recall": safe_div(shadow_hit, shadow_sum),
        "division_shadow_iou": safe_div(shadow_hit, union),
        "division_cloud_leakage": safe_div(float((pos * cloud).sum().item()), cloud_sum),
        "division_clear_leakage": safe_div(float((pos * clear).sum().item()), clear_sum),
        "division_outside_bbox_frac": safe_div(float((pos * (1.0 - box_mask)).sum().item()), pos_sum),
        "bbox_valid": int(bbox_valid),
        "bbox_x1": float(bbox[0].item()) if bbox_valid else math.nan,
        "bbox_y1": float(bbox[1].item()) if bbox_valid else math.nan,
        "bbox_x2": float(bbox[2].item()) if bbox_valid else math.nan,
        "bbox_y2": float(bbox[3].item()) if bbox_valid else math.nan,
        "bbox_area_frac": bbox_sum / float(h * w),
        "bbox_shadow_recall": safe_div(float((box_mask * shadow).sum().item()), shadow_sum),
        "bbox_division_recall": safe_div(float((box_mask * pos).sum().item()), pos_sum),
        "bbox_cloud_frac_inside": safe_div(float((box_mask * cloud).sum().item()), bbox_sum),
        "bbox_clear_frac_inside": safe_div(float((box_mask * clear).sum().item()), bbox_sum),
        "boundary_iou": safe_div(boundary_inter, boundary_union),
        "reference_target_mae_full": float(diff_reference.mean().item()),
        "reference_target_mae_shadow": masked_mean(diff_reference, shadow),
        "reference_target_mae_bbox": masked_mean(diff_reference, box_mask),
        "reference_target_corr_shadow": pearson_corr(division, reference, shadow),
        # Backward-compatible column names for older analysis notebooks.
        "online_target_mae_full": float(diff_reference.mean().item()),
        "online_target_mae_shadow": masked_mean(diff_reference, shadow),
        "online_target_mae_bbox": masked_mean(diff_reference, box_mask),
        "online_target_corr_shadow": pearson_corr(division, reference, shadow),
    }
    aux = {
        "division": division,
        "online": reference,
        "shadow": shadow,
        "cloud": cloud,
        "clear": clear,
        "bbox_mask": box_mask,
        "bbox": bbox,
        "diff_online": diff_reference,
    }
    return row, aux


def save_sample_visual(
    item: dict[str, Any],
    aux: dict[str, Tensor],
    row: dict[str, Any],
    path: Path,
    rgb_indices: tuple[int, int, int],
) -> None:
    cloudy_rgb = render_rgb(item["s2_toa"], rgb_indices)
    target_rgb = render_rgb(item["target"], rgb_indices)
    bbox_rgb = draw_bbox(cloudy_rgb, aux["bbox"]) if int(row["bbox_valid"]) else cloudy_rgb
    hard_overlay = overlay_mask(cloudy_rgb, aux["cloud"], (0, 200, 255), alpha=0.45)
    hard_overlay = overlay_mask(hard_overlay, aux["shadow"], (255, 170, 0), alpha=0.45)
    div_overlay = overlay_mask(cloudy_rgb, aux["division"], (255, 120, 0), alpha=0.55)
    if int(row["bbox_valid"]):
        div_overlay = draw_bbox(div_overlay, aux["bbox"])
        hard_overlay = draw_bbox(hard_overlay, aux["bbox"])
    panels = [
        ("Cloudy + bbox", bbox_rgb),
        ("Target", target_rgb),
        ("Hard cloud/shadow", hard_overlay),
        ("Division mask", render_mask(aux["division"], (255, 150, 0))),
        ("Division overlay", div_overlay),
        ("Reference target", render_mask(aux["online"], (180, 255, 80))),
        ("Abs diff vs ref", render_mask(aux["diff_online"], (255, 50, 50))),
        ("BBox area", render_mask(aux["bbox_mask"], (255, 60, 60))),
    ]
    save_grid(panels, path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(rows), "metrics": {}, "by_cloud_bucket": {}}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))] if rows else []
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        if values:
            summary["metrics"][key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    for bucket in ("low", "medium", "high", "heavy"):
        bucket_rows = [row for row in rows if row.get("cloud_bucket") == bucket]
        if not bucket_rows:
            continue
        summary["by_cloud_bucket"][bucket] = {"samples": len(bucket_rows), "metrics": {}}
        for key in numeric_keys:
            values = [float(row[key]) for row in bucket_rows if math.isfinite(float(row[key]))]
            if values:
                summary["by_cloud_bucket"][bucket]["metrics"][key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slice_collated_item(batch: dict[str, Any], index: int) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            item[key] = value[index]
        elif isinstance(value, (list, tuple)):
            item[key] = value[index]
        else:
            item[key] = value
    return item


def main() -> None:
    args = parse_args()
    assert_not_placeholder_path(args.softshadow_mask_dir, "--softshadow-mask-dir")
    assert_not_placeholder_path(args.softshadow_bbox_path, "--softshadow-bbox-path")
    cfg = load_config(args.config)
    splits = list(args.splits) if args.splits else [args.split]
    if len(splits) > 1 and (args.softshadow_mask_dir is not None or args.softshadow_bbox_path is not None):
        raise SystemExit("When using --splits, configure split-specific prompt paths in YAML instead of one global CLI override.")
    if args.softshadow_mask_dir is not None:
        cfg.setdefault("data", {})["softshadow_mask_dir"] = str(args.softshadow_mask_dir.expanduser().resolve())
    if args.softshadow_bbox_path is not None:
        cfg.setdefault("data", {})["softshadow_bbox_path"] = str(args.softshadow_bbox_path.expanduser().resolve())

    model_cfg = cfg.get("model", {})
    bbox_space = args.bbox_space or str(model_cfg.get("softshadow_bbox_space", "image"))
    sam_input_size = int(args.sam_input_size or model_cfg.get("softshadow_sam_input_size", 1024))
    rgb_indices = tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))

    root_out = args.output_dir.expanduser().resolve()
    all_summaries: dict[str, Any] = {}
    for split in splits:
        require_offline_or_fallback(cfg, split, args)
        dataset = build_dataset(cfg, split)
        out_dir = root_out if len(splits) == 1 else root_out / split
        vis_dir = out_dir / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        saved_by_bucket = defaultdict(int)
        total = min(len(dataset), args.limit) if args.limit else len(dataset)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=False,
            drop_last=False,
            persistent_workers=bool(args.num_workers and args.num_workers > 0),
        )
        seen = 0
        with torch.no_grad():
            pbar = tqdm(total=total, desc=f"audit {split}", unit="sample")
            for batch in loader:
                batch_size = int(next(value for value in batch.values() if isinstance(value, Tensor)).shape[0])
                for j in range(batch_size):
                    if seen >= total:
                        break
                    item = slice_collated_item(batch, j)
                    row, aux = sample_metrics(
                        item,
                        cfg,
                        bbox_space=bbox_space,
                        sam_input_size=sam_input_size,
                        positive_threshold=args.positive_threshold,
                        hard_threshold=args.hard_threshold,
                        fallback_online_mask=args.fallback_online_mask,
                        fallback_bbox_from_mask=args.fallback_bbox_from_mask,
                        reference_target_mode=str(args.reference_target_mode),
                        reference_low_pass_kernel=int(args.reference_low_pass_kernel),
                        reference_division_threshold=float(args.reference_division_threshold),
                    )
                    rows.append(row)
                    bucket = str(row["cloud_bucket"])
                    should_visualize = (
                        bucket in set(args.visual_buckets)
                        and float(row["shadow_frac"]) >= float(args.min_visual_shadow_frac)
                        and float(row["division_pos_frac"]) >= float(args.min_visual_division_pos_frac)
                    )
                    if should_visualize and saved_by_bucket[bucket] < args.visual_samples_per_bucket:
                        filename = f"{bucket}_{saved_by_bucket[bucket]:02d}_idx{seen:04d}_{str(row['sample_id']).replace('/', '_')}.png"
                        save_sample_visual(item, aux, row, vis_dir / filename, rgb_indices)
                        saved_by_bucket[bucket] += 1
                    seen += 1
                    pbar.update(1)
                if seen >= total:
                    break
            pbar.close()

        summary = summarize(rows)
        summary["config"] = str(Path(args.config).expanduser().resolve())
        summary["split"] = split
        summary["bbox_space"] = bbox_space
        summary["sam_input_size"] = sam_input_size
        summary["thresholds"] = {
            "positive": args.positive_threshold,
            "hard": args.hard_threshold,
        }
        summary["reference_target"] = {
            "mode": str(args.reference_target_mode),
            "low_pass_kernel": int(args.reference_low_pass_kernel),
            "division_threshold": float(args.reference_division_threshold),
        }
        summary["fallbacks"] = {
            "online_mask": bool(args.fallback_online_mask),
            "bbox_from_mask": bool(args.fallback_bbox_from_mask),
        }
        summary["visualizations"] = {
            "saved_by_cloud_bucket": dict(saved_by_bucket),
            "visual_samples_per_bucket": args.visual_samples_per_bucket,
            "visual_buckets": list(args.visual_buckets),
            "min_visual_shadow_frac": float(args.min_visual_shadow_frac),
            "min_visual_division_pos_frac": float(args.min_visual_division_pos_frac),
        }

        write_csv(out_dir / f"{split}_softshadow_prompt_quality.csv", rows)
        (out_dir / f"{split}_softshadow_prompt_quality_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_summaries[split] = summary

    if len(splits) > 1:
        root_out.mkdir(parents=True, exist_ok=True)
        (root_out / "softshadow_prompt_quality_summary_all_splits.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(all_summaries[splits[0]], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
