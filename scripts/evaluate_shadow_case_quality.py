#!/usr/bin/env python3
"""Audit SoftShadow shadow_case labels and prompt quality on ALLClear splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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
from src.allclear.dataset import AllClearDataset, SHADOW_CASE_NAMES
from src.allclear.modules.common import masks_from_cld_shdw


def split_value(data: dict[str, Any], key: str, split: str) -> Any:
    return data.get(f"{key}_{split}", data.get(key))


def build_dataset(cfg: dict[str, Any], split: str) -> AllClearDataset:
    data = cfg["data"]
    data["softshadow_shadow_case_enabled"] = True
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
        softshadow_mask_dir=split_value(data, "softshadow_mask_dir", split),
        softshadow_bbox_path=split_value(data, "softshadow_bbox_path", split),
        softshadow_bbox_space=str(cfg.get("model", {}).get("softshadow_bbox_space", data.get("softshadow_bbox_space", "image"))),
        softshadow_sam_input_size=int(cfg.get("model", {}).get("softshadow_sam_input_size", data.get("softshadow_sam_input_size", 1024))),
        softshadow_shadow_case_enabled=True,
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


def scalar(x: Any, default: float = math.nan) -> float:
    if isinstance(x, Tensor):
        return float(x.detach().float().reshape(-1)[0].item())
    try:
        return float(x)
    except Exception:
        return default


def row_from_item(item: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    shadow_index = int(cfg.get("data", {}).get("shadow_index", 3))
    cloud_index = int(cfg.get("data", {}).get("cloud_index", 1))
    masks = masks_from_cld_shdw(item["cld_shdw"].unsqueeze(0), shadow_index=shadow_index, cloud_index=cloud_index)
    case = int(item["shadow_case"].item())
    cloud_frac = float(masks.cloud.mean().item())
    shadow_frac = float(masks.shadow.mean().item())
    clear_frac = float(masks.clear.mean().item())
    return {
        "sample_id": item.get("sample_id", ""),
        "shadow_case": case,
        "shadow_case_name": SHADOW_CASE_NAMES.get(case, "unknown"),
        "cloud_bucket": cloud_bucket(cloud_frac),
        "cloud_frac": cloud_frac,
        "shadow_frac": shadow_frac,
        "clear_frac": clear_frac,
        "division_pos_frac": scalar(item.get("shadow_case_division_pos_frac")),
        "bbox_valid": scalar(item.get("shadow_case_bbox_valid")),
        "bbox_area_frac": scalar(item.get("shadow_case_bbox_area_frac")),
        "division_shadow_precision_dilated": scalar(item.get("shadow_case_division_shadow_precision_dilated")),
        "division_clear_leakage": scalar(item.get("shadow_case_division_clear_leakage")),
        "division_cloud_leakage": scalar(item.get("shadow_case_division_cloud_leakage")),
    }


def render_rgb(x: Tensor, rgb_indices: tuple[int, int, int]) -> Image.Image:
    rgb = x[list(rgb_indices)].float().clamp(0.0, 1.0)
    lo = torch.quantile(rgb.flatten(1), 0.02, dim=1).view(3, 1, 1)
    hi = torch.quantile(rgb.flatten(1), 0.98, dim=1).view(3, 1, 1)
    rgb = ((rgb - lo) / (hi - lo).clamp_min(1.0e-4)).clamp(0.0, 1.0).pow(0.85)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def render_mask(mask: Tensor, color: tuple[int, int, int]) -> Image.Image:
    m = mask.float()
    if m.ndim == 3:
        m = m[:1]
    m = m.clamp(0.0, 1.0)[0]
    arr = torch.zeros((m.shape[0], m.shape[1], 3), dtype=torch.uint8)
    for idx, value in enumerate(color):
        arr[..., idx] = (m * float(value)).round().to(torch.uint8)
    return Image.fromarray(arr.numpy(), mode="RGB")


def overlay_mask(base: Image.Image, mask: Tensor, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    out = base.convert("RGBA")
    m = mask.float()
    if m.ndim == 3:
        m = m[:1]
    m = m.clamp(0.0, 1.0)[0]
    rgba = torch.zeros((m.shape[0], m.shape[1], 4), dtype=torch.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (m * alpha * 255.0).round().to(torch.uint8)
    return Image.alpha_composite(out, Image.fromarray(rgba.numpy(), mode="RGBA")).convert("RGB")


def bbox_to_image_coords(bbox: Tensor, h: int, w: int, bbox_space: str, sam_input_size: int) -> Tensor | None:
    box = bbox.float().reshape(-1, 4)[0].clone()
    if not torch.isfinite(box).all():
        return None
    if bbox_space == "sam_input":
        box[[0, 2]] *= float(w) / float(sam_input_size)
        box[[1, 3]] *= float(h) / float(sam_input_size)
    box[[0, 2]] = box[[0, 2]].clamp(0.0, float(w - 1))
    box[[1, 3]] = box[[1, 3]].clamp(0.0, float(h - 1))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def draw_bbox(image: Image.Image, bbox: Tensor | None) -> Image.Image:
    out = image.copy()
    if bbox is None:
        return out
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    for offset in range(3):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 60, 60))
    return out


def save_visual(item: dict[str, Any], row: dict[str, Any], cfg: dict[str, Any], path: Path) -> None:
    rgb_indices = tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    shadow_index = int(cfg.get("data", {}).get("shadow_index", 3))
    cloud_index = int(cfg.get("data", {}).get("cloud_index", 1))
    bbox_space = str(cfg.get("model", {}).get("softshadow_bbox_space", cfg.get("data", {}).get("softshadow_bbox_space", "image"))).lower()
    sam_input_size = int(cfg.get("model", {}).get("softshadow_sam_input_size", cfg.get("data", {}).get("softshadow_sam_input_size", 1024)))
    masks = masks_from_cld_shdw(item["cld_shdw"].unsqueeze(0), shadow_index=shadow_index, cloud_index=cloud_index)
    cloudy = render_rgb(item["s2_toa"], rgb_indices)
    target = render_rgb(item["target"], rgb_indices)
    h, w = item["s2_toa"].shape[-2:]
    bbox = bbox_to_image_coords(item["bbox"], h, w, bbox_space, sam_input_size) if "bbox" in item else None
    division = item.get("sam_mask", masks.shadow[0].new_zeros(masks.shadow[0].shape))
    if division.shape[-2:] != (h, w):
        division = F.interpolate(division.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    hard = overlay_mask(render_mask(masks.cloud[0], (0, 190, 255)), masks.shadow[0], (255, 165, 0), alpha=0.85)
    overlay = overlay_mask(cloudy, division, (255, 165, 0), alpha=0.45)
    panels = [
        ("Cloudy + bbox", draw_bbox(cloudy, bbox)),
        ("Target", target),
        ("Hard cloud/shadow", hard),
        ("Division mask", render_mask(division, (255, 165, 0))),
        ("Division overlay", draw_bbox(overlay, bbox)),
        (f"{row['shadow_case_name']}", case_card(row, h, w)),
    ]
    save_grid(panels, path)


def case_card(row: dict[str, Any], h: int, w: int) -> Image.Image:
    image = Image.new("RGB", (w, h), (18, 24, 32))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    lines = [
        f"case: {row['shadow_case_name']}",
        f"cloud: {row['cloud_frac']:.3f}",
        f"shadow: {row['shadow_frac']:.3f}",
        f"division+: {row['division_pos_frac']:.3f}",
        f"bbox area: {row['bbox_area_frac']:.3f}",
        f"shadow prec: {row['division_shadow_precision_dilated']:.3f}",
        f"clear leak: {row['division_clear_leakage']:.3f}",
        f"cloud leak: {row['division_cloud_leakage']:.3f}",
    ]
    for idx, line in enumerate(lines):
        draw.text((10, 12 + idx * 20), line, fill=(238, 242, 247), font=font)
    return image


def save_grid(panels: list[tuple[str, Image.Image]], path: Path, tile: int = 192) -> None:
    title_h = 24
    pad = 4
    cols = 3
    rows = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (cols * tile + (cols + 1) * pad, rows * (tile + title_h) + (rows + 1) * pad), (12, 16, 22))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for idx, (title, image) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = pad + c * (tile + pad)
        y = pad + r * (tile + title_h + pad)
        draw.rectangle((x, y, x + tile, y + title_h), fill=(31, 41, 55))
        draw.text((x + 6, y + 5), title[:34], fill=(238, 242, 247), font=font)
        canvas.paste(image.resize((tile, tile), Image.BILINEAR), (x, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    by_case = Counter(row["shadow_case_name"] for row in rows)
    by_bucket = Counter(row["cloud_bucket"] for row in rows)
    by_case_bucket = Counter((row["shadow_case_name"], row["cloud_bucket"]) for row in rows)
    means: dict[str, float] = {}
    for key in (
        "cloud_frac",
        "shadow_frac",
        "division_pos_frac",
        "bbox_area_frac",
        "division_shadow_precision_dilated",
        "division_clear_leakage",
        "division_cloud_leakage",
    ):
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        means[key] = sum(values) / max(1, len(values))
    return {
        "count": len(rows),
        "case_count": dict(by_case),
        "case_ratio": {key: value / total for key, value in by_case.items()},
        "cloud_bucket_count": dict(by_bucket),
        "cloud_bucket_ratio": {key: value / total for key, value in by_bucket.items()},
        "case_by_cloud_bucket": {f"{case}/{bucket}": value for (case, bucket), value in by_case_bucket.items()},
        "means": means,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/allclear/shadow_case_quality"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visual-samples-per-case", type=int, default=3)
    parser.add_argument("--visual-buckets", nargs="+", default=["low", "medium"], choices=["low", "medium", "high", "heavy"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root_out = args.output_dir.expanduser().resolve()
    all_summaries: dict[str, Any] = {}
    for split in args.splits:
        dataset = build_dataset(cfg, split)
        out_dir = root_out / split if len(args.splits) > 1 else root_out
        vis_dir = out_dir / "visualizations"
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=False,
            drop_last=False,
        )
        rows: list[dict[str, Any]] = []
        saved: defaultdict[tuple[str, str], int] = defaultdict(int)
        total = min(len(dataset), args.limit) if args.limit else len(dataset)
        seen = 0
        pbar = tqdm(total=total, desc=f"shadow-case {split}", unit="sample")
        for batch in loader:
            batch_size = int(next(value for value in batch.values() if isinstance(value, Tensor)).shape[0])
            for i in range(batch_size):
                if seen >= total:
                    break
                item = {
                    key: (value[i] if isinstance(value, Tensor) else value[i] if isinstance(value, list) else value)
                    for key, value in batch.items()
                }
                row = row_from_item(item, cfg)
                rows.append(row)
                case_name = str(row["shadow_case_name"])
                bucket = str(row["cloud_bucket"])
                save_key = (case_name, bucket)
                if bucket in set(args.visual_buckets) and saved[save_key] < args.visual_samples_per_case:
                    sample_id = str(row["sample_id"]).replace("/", "_")
                    save_visual(
                        item,
                        row,
                        cfg,
                        vis_dir / f"{case_name}_{bucket}_{saved[save_key]:02d}_idx{seen:04d}_{sample_id}.png",
                    )
                    saved[save_key] += 1
                seen += 1
                pbar.update(1)
            if seen >= total:
                break
        pbar.close()
        summary = summarize(rows)
        summary["config"] = str(Path(args.config).expanduser().resolve())
        summary["split"] = split
        summary["thresholds"] = {
            key: cfg.get("data", {}).get(key)
            for key in (
                "softshadow_shadow_case_positive_threshold",
                "softshadow_absent_shadow_threshold",
                "softshadow_absent_division_threshold",
                "softshadow_valid_shadow_threshold",
                "softshadow_valid_division_threshold",
                "softshadow_max_bbox_area",
                "softshadow_min_division_shadow_precision_dilated",
                "softshadow_max_clear_leakage",
                "softshadow_max_cloud_leakage",
            )
        }
        summary["visualizations"] = {
            "saved_by_case_bucket": {f"{case}/{bucket}": count for (case, bucket), count in saved.items()},
            "visual_buckets": list(args.visual_buckets),
            "visual_samples_per_case": int(args.visual_samples_per_case),
        }
        write_csv(out_dir / f"{split}_shadow_case_quality.csv", rows)
        (out_dir / f"{split}_shadow_case_quality_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        all_summaries[split] = summary
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "shadow_case_quality_summary_all_splits.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
