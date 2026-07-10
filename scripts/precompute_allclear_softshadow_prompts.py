#!/usr/bin/env python3
"""Precompute SoftShadow-style division masks and bbox prompts for ALLClear.

SoftShadow's official dataloader expects two offline assets:

- ``gt_mask_dir`` / ``division_filter_results``: grayscale mask images loaded
  as ``sam_mask`` and resized to 256x256;
- ``bbox_path``: a YAML mapping from image stem to an ``[x1, y1, x2, y2]`` box
  passed directly to SAM after the input image is resized to 1024x1024.

ALLClear does not ship these SoftShadow files, so this script generates an
ALLClear-specific official-format equivalent from paired cloudy/clear S2 using
the SoftShadow paper's Y-channel division target. The ALLClear hard shadow mask
is not used to crop or constrain the division mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.config import load_config
from src.allclear.dataset import AllClearDataset
from src.allclear.modules.common import bbox_from_mask, masks_from_cld_shdw
from src.allclear.modules.softshadow import soft_shadow_division_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training/eval YAML or config.resolved.json.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed/softshadow_prompts"),
    )
    parser.add_argument("--bbox-space", default="sam_input", choices=["sam_input", "image"])
    parser.add_argument("--sam-input-size", type=int, default=1024)
    parser.add_argument("--mask-size", type=int, default=256)
    parser.add_argument("--bbox-pad", type=int, default=4, help="Padding in image-coordinate pixels before optional SAM scaling.")
    parser.add_argument("--low-pass-kernel", type=int, default=5)
    parser.add_argument("--division-threshold", type=float, default=0.05, help="SoftShadow-style threshold for suppressing lit/outlier pixels.")
    parser.add_argument("--bbox-source", default="division", choices=["division", "hard_shadow"], help="Default division avoids using ALLClear hard shadow as a prompt source.")
    parser.add_argument("--bbox-threshold", type=float, default=0.05)
    parser.add_argument("--min-bbox-pixels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def build_dataset(cfg: dict[str, Any], split: str) -> AllClearDataset:
    data = cfg["data"]
    return AllClearDataset(
        root=data["root"],
        manifest=data[f"{split}_manifest"],
        optical_scale=float(data.get("optical_scale", 10000.0)),
        image_size=data.get("image_size"),
        shadow_index=int(data.get("shadow_index", 3)),
        cloud_index=int(data.get("cloud_index", 1)),
        # SoftShadow division masks and bbox prompts only need paired optical
        # images plus cld_shdw metadata; reading SAR here is pure IO overhead.
        load_sar=False,
        cache_dir=data.get("cache_dir"),
        band_indices=data.get("band_indices"),
    )


def safe_name(value: Any) -> str:
    text = str(value)
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def scale_bbox(box: torch.Tensor, h: int, w: int, bbox_space: str, sam_input_size: int) -> list[int]:
    out = box.float().reshape(4).clone()
    if bbox_space == "sam_input":
        out[[0, 2]] *= float(sam_input_size) / float(w)
        out[[1, 3]] *= float(sam_input_size) / float(h)
        max_x = max_y = sam_input_size - 1
    else:
        max_x = w - 1
        max_y = h - 1
    out[[0, 2]] = out[[0, 2]].clamp(0.0, float(max_x))
    out[[1, 3]] = out[[1, 3]].clamp(0.0, float(max_y))
    vals = [int(round(float(v))) for v in out]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return [0, 0, int(max_x), int(max_y)]
    return vals


def save_mask_png(mask: torch.Tensor, path: Path) -> None:
    arr = (mask[0].float().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def write_yaml(path: Path, data: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    except Exception:
        # JSON is valid YAML 1.2 and can still be read by yaml.safe_load.
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_split(cfg: dict[str, Any], split: str, args: argparse.Namespace) -> dict[str, Any]:
    dataset = build_dataset(cfg, split)
    out_dir = args.output_root.expanduser().resolve() / split
    mask_dir = out_dir / "division_filter_results"
    bbox_path = out_dir / "bounding_boxes.yaml"
    stats_path = out_dir / "prompt_precompute_stats.csv"
    manifest_path = out_dir / "softshadow_manifest.csv"
    if out_dir.exists() and not args.overwrite:
        existing = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
        if existing and bbox_path.exists():
            raise SystemExit(f"Output exists for split={split}: {out_dir}. Pass --overwrite to regenerate.")
    mask_dir.mkdir(parents=True, exist_ok=True)

    rgb_indices = tuple(cfg.get("data", {}).get("rgb_indices", [3, 2, 1]))
    shadow_index = int(cfg.get("data", {}).get("shadow_index", 3))
    cloud_index = int(cfg.get("data", {}).get("cloud_index", 1))
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
    bbox_data: dict[str, list[int]] = {}
    stats_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    seen = 0
    with torch.no_grad():
        pbar = tqdm(total=total, desc=f"precompute {split}", unit="sample")
        for batch in loader:
            if seen >= total:
                break
            remaining = total - seen
            bsz = min(int(batch["s2_toa"].shape[0]), remaining)
            s2 = batch["s2_toa"][:bsz].float()
            target = batch["target"][:bsz].float()
            cld_shdw = batch["cld_shdw"][:bsz].float()
            sample_ids = batch.get("sample_id", [f"{split}_{seen + i:06d}" for i in range(bsz)])
            masks = masks_from_cld_shdw(cld_shdw, shadow_index=shadow_index, cloud_index=cloud_index)
            shadow = masks.shadow.float()
            cloud = masks.cloud.float()
            divisions = soft_shadow_division_target(
                s2,
                target,
                rgb_indices=rgb_indices,
                low_pass_kernel=int(args.low_pass_kernel),
                threshold=float(args.division_threshold),
            )
            h, w = shadow.shape[-2:]
            if args.mask_size and (h, w) != (args.mask_size, args.mask_size):
                divisions_to_save = torch.nn.functional.interpolate(
                    divisions,
                    size=(args.mask_size, args.mask_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                divisions_to_save = divisions
            for i in range(bsz):
                sample_id = safe_name(sample_ids[i])
                division = divisions[i]
                mask_path = mask_dir / f"{sample_id}.png"
                save_mask_png(divisions_to_save[i], mask_path)

                shadow_i = shadow[i : i + 1]
                cloud_i = cloud[i : i + 1]
                shadow_pixels = int((shadow_i > 0.5).sum().item())
                if args.bbox_source == "division":
                    bbox_support = (division.unsqueeze(0) > float(args.bbox_threshold)).float()
                else:
                    bbox_support = shadow_i
                bbox_pixels = int((bbox_support > 0.5).sum().item())
                if bbox_pixels >= int(args.min_bbox_pixels):
                    box_image = bbox_from_mask(bbox_support, pad=int(args.bbox_pad))[0]
                else:
                    box_image = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)])
                bbox = scale_bbox(box_image, h=h, w=w, bbox_space=str(args.bbox_space), sam_input_size=int(args.sam_input_size))
                bbox_data[sample_id] = bbox

                cloud_frac = float(cloud_i.mean().item())
                shadow_frac = float(shadow_i.mean().item())
                div_pos_frac = float((division > 0.05).float().mean().item())
                stats_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "mask_path": str(mask_path),
                        "bbox": json.dumps(bbox),
                        "bbox_space": args.bbox_space,
                        "bbox_source": args.bbox_source,
                        "cloud_frac": f"{cloud_frac:.8f}",
                        "shadow_frac": f"{shadow_frac:.8f}",
                        "division_mean": f"{float(division.mean().item()):.8f}",
                        "division_max": f"{float(division.max().item()):.8f}",
                        "division_pos_frac": f"{div_pos_frac:.8f}",
                        "shadow_pixels": shadow_pixels,
                        "bbox_pixels": bbox_pixels,
                    }
                )
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "sam_mask_path": str(mask_path),
                        "bbox": json.dumps(bbox),
                        "bbox_space": args.bbox_space,
                        "bbox_source": args.bbox_source,
                    }
                )
            seen += bsz
            pbar.update(bsz)
        pbar.close()

    write_yaml(bbox_path, bbox_data)
    write_csv(stats_path, stats_rows)
    write_csv(manifest_path, manifest_rows)
    summary = {
        "split": split,
        "samples": total,
        "division_mask_dir": str(mask_dir),
        "bbox_path": str(bbox_path),
        "bbox_space": args.bbox_space,
        "bbox_source": args.bbox_source,
        "sam_input_size": int(args.sam_input_size),
        "mask_size": int(args.mask_size),
        "division_threshold": float(args.division_threshold),
        "low_pass_kernel": int(args.low_pass_kernel),
        "stats_csv": str(stats_path),
        "manifest_csv": str(manifest_path),
        "mean_shadow_frac": sum(float(r["shadow_frac"]) for r in stats_rows) / max(len(stats_rows), 1),
        "mean_division_pos_frac": sum(float(r["division_pos_frac"]) for r in stats_rows) / max(len(stats_rows), 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summaries = [process_split(cfg, split, args) for split in args.splits]
    root = args.output_root.expanduser().resolve()
    (root / "summary.json").write_text(json.dumps({"splits": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"splits": summaries}, indent=2, ensure_ascii=False))
    print("\nConfig fields to use:")
    for summary in summaries:
        split = summary["split"]
        print(f"  data.softshadow_mask_dir_{split}: {summary['division_mask_dir']}")
        print(f"  data.softshadow_bbox_path_{split}: {summary['bbox_path']}")
    if summaries:
        print(f"  model.softshadow_bbox_space: {summaries[0]['bbox_space']}")


if __name__ == "__main__":
    main()
