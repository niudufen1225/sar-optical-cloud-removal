#!/usr/bin/env python3
"""Visualize every OmniCloudMask ALLClear sample.

The script is read-only for the dataset. It writes per-sample PNG panels and an
``index.csv`` with mask fractions, so the generated OmniCloudMask products can be
audited before training.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import tifffile  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires tifffile. Install it in the cr env first.") from exc


DEFAULT_OMNI_ROOT = Path(
    "/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed/omnicloudmask_v4_low300_mh1000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-root", type=Path, default=DEFAULT_OMNI_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/allclear/omnicloudmask_visual_audit"))
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated subset of train,val,test.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap per split. 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rgb-indices", default="3,2,1", help="0-based S2 bands to render as RGB. Default B04,B03,B02.")
    parser.add_argument("--optical-scale", type=float, default=10000.0)
    parser.add_argument("--rgb-max", type=float, default=0.35, help="Reflectance value mapped to white.")
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--title-height", type=int, default=22)
    parser.add_argument("--official-cloud-index", type=int, default=1)
    parser.add_argument("--official-shadow-index", type=int, default=3)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_stem(text: str) -> str:
    text = text.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)[:220]


def to_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape={arr.shape}")
    if arr.shape[0] <= 32 and arr.shape[-1] > 32:
        return np.transpose(arr, (1, 2, 0))
    return arr


def read_tif_hwc(path: str | Path) -> np.ndarray:
    return to_hwc(tifffile.imread(Path(path)))


def derive_official_cld_shdw_from_s2(path: str | Path) -> Path | None:
    p = Path(path)
    parts = list(p.parts)
    try:
        idx = parts.index("s2_toa")
    except ValueError:
        return None
    parts[idx] = "cld_shdw"
    derived = Path(*parts)
    derived = derived.with_name(derived.name.replace("_s2_toa_", "_cld_shdw_")).with_suffix(".tif")
    return derived if derived.exists() else None


def resize_rgb(arr: np.ndarray, size: int, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return image.resize((size, size), resample)


def render_s2_rgb(arr: np.ndarray, indices: tuple[int, int, int], scale: float, rgb_max: float, size: int) -> Image.Image:
    hwc = to_hwc(arr).astype(np.float32)
    channels = []
    for idx in indices:
        if idx >= hwc.shape[-1]:
            channels.append(np.zeros(hwc.shape[:2], dtype=np.float32))
        else:
            channels.append(hwc[..., idx] / max(scale, 1e-6))
    rgb = np.stack(channels, axis=-1)
    rgb = np.clip(rgb / max(rgb_max, 1e-6), 0.0, 1.0)
    rgb = np.power(rgb, 0.85)
    return resize_rgb((rgb * 255.0 + 0.5).astype(np.uint8), size)


def render_mask(mask: np.ndarray, color: tuple[int, int, int], size: int) -> Image.Image:
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if np.nanmax(m) > 1.5:
        m = m / 255.0
    m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c, value in enumerate(color):
        rgb[..., c] = (m * float(value)).astype(np.uint8)
    return resize_rgb(rgb, size, resample=Image.Resampling.NEAREST)


def render_class_map(classes: np.ndarray, size: int) -> Image.Image:
    cls = np.asarray(classes)
    if cls.ndim == 3:
        cls = cls[..., 0]
    cls = cls.astype(np.int64)
    palette = np.array(
        [
            [0, 0, 0],        # clear
            [0, 210, 255],    # thick cloud
            [70, 120, 255],   # thin cloud
            [255, 170, 40],   # shadow
        ],
        dtype=np.uint8,
    )
    cls = np.clip(cls, 0, len(palette) - 1)
    return resize_rgb(palette[cls], size, resample=Image.Resampling.NEAREST)


def mask_channel(mask_hwc: np.ndarray | None, index: int) -> np.ndarray | None:
    if mask_hwc is None or mask_hwc.shape[-1] <= index:
        return None
    m = mask_hwc[..., index].astype(np.float32)
    if np.nanmax(m) > 1.5:
        m = m / 255.0
    return np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def damage_from_mask(mask_hwc: np.ndarray | None, cloud_index: int, shadow_index: int) -> np.ndarray | None:
    cloud = mask_channel(mask_hwc, cloud_index)
    shadow = mask_channel(mask_hwc, shadow_index)
    if cloud is None and shadow is None:
        return None
    if cloud is None:
        cloud = np.zeros_like(shadow)
    if shadow is None:
        shadow = np.zeros_like(cloud)
    return np.maximum(cloud, shadow)


def add_title(tile: Image.Image, title: str, title_height: int) -> Image.Image:
    out = Image.new("RGB", (tile.width, tile.height + title_height), (18, 24, 32))
    out.paste(tile, (0, title_height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((5, 4), title, fill=(235, 240, 245), font=font)
    return out


def make_panel(tiles: list[tuple[str, Image.Image]], cols: int, title_height: int) -> Image.Image:
    rows = math.ceil(len(tiles) / cols)
    titled = [add_title(tile, title, title_height) for title, tile in tiles]
    w, h = titled[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (8, 10, 14))
    for idx, tile in enumerate(titled):
        y, x = divmod(idx, cols)
        canvas.paste(tile, (x * w, y * h))
    return canvas


def row_path(row: dict[str, str], *keys: str) -> Path | None:
    for key in keys:
        value = row.get(key, "")
        if value:
            path = Path(value)
            if path.exists():
                return path
    return None


def fraction(mask: np.ndarray | None) -> float:
    if mask is None:
        return float("nan")
    return float(np.asarray(mask, dtype=np.float32).clip(0.0, 1.0).mean())


def process_row(
    row: dict[str, str],
    split: str,
    args: argparse.Namespace,
    rgb_indices: tuple[int, int, int],
) -> dict[str, Any]:
    sample_id = row.get("sample_id") or Path(row.get("cloudy_s2_path", "sample")).stem
    bucket = row.get("bucket") or "unknown"
    out_dir = args.output_dir / split / bucket
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{safe_stem(sample_id)}.png"

    cloudy_path = row_path(row, "cloudy_s2_path", "s2_toa", "input")
    target_path = row_path(row, "clear_s2_path", "target_path", "target")
    class_path = row_path(row, "omnicloudmask_class_path")
    omni_mask_path = row_path(row, "omnicloudmask_cld_shdw_path", "cloudy_mask_path", "cld_shdw_path")
    official_mask_path = derive_official_cld_shdw_from_s2(cloudy_path) if cloudy_path else None

    if cloudy_path is None or target_path is None or class_path is None or omni_mask_path is None:
        missing = [
            name
            for name, value in {
                "cloudy_s2_path": cloudy_path,
                "clear_s2_path": target_path,
                "omnicloudmask_class_path": class_path,
                "omnicloudmask_cld_shdw_path": omni_mask_path,
            }.items()
            if value is None
        ]
        raise FileNotFoundError(f"{sample_id}: missing {missing}")

    class_map = read_tif_hwc(class_path)[..., 0]
    omni_mask = read_tif_hwc(omni_mask_path)
    official_mask = read_tif_hwc(official_mask_path) if official_mask_path else None
    omni_cloud = mask_channel(omni_mask, args.official_cloud_index)
    omni_shadow = mask_channel(omni_mask, args.official_shadow_index)
    omni_damage = damage_from_mask(omni_mask, args.official_cloud_index, args.official_shadow_index)
    official_cloud = mask_channel(official_mask, args.official_cloud_index)
    official_shadow = mask_channel(official_mask, args.official_shadow_index)
    official_damage = damage_from_mask(official_mask, args.official_cloud_index, args.official_shadow_index)
    diff_damage = None
    if omni_damage is not None and official_damage is not None:
        diff_damage = np.abs(omni_damage - official_damage)

    if args.overwrite or not out_png.exists():
        cloudy_rgb = render_s2_rgb(read_tif_hwc(cloudy_path), rgb_indices, args.optical_scale, args.rgb_max, args.tile_size)
        target_rgb = render_s2_rgb(read_tif_hwc(target_path), rgb_indices, args.optical_scale, args.rgb_max, args.tile_size)
        blank = Image.new("RGB", (args.tile_size, args.tile_size), (0, 0, 0))
        tiles = [
            ("Cloudy S2", cloudy_rgb),
            ("Target", target_rgb),
            ("Omni Class", render_class_map(class_map, args.tile_size)),
            ("Omni Cloud", render_mask(omni_cloud, (0, 210, 255), args.tile_size) if omni_cloud is not None else blank),
            ("Omni Shadow", render_mask(omni_shadow, (255, 170, 40), args.tile_size) if omni_shadow is not None else blank),
            ("Omni Damage", render_mask(omni_damage, (235, 235, 235), args.tile_size) if omni_damage is not None else blank),
            ("ALLClear Cloud", render_mask(official_cloud, (0, 210, 255), args.tile_size) if official_cloud is not None else blank),
            ("ALLClear Shadow", render_mask(official_shadow, (255, 170, 40), args.tile_size) if official_shadow is not None else blank),
            ("Damage Diff", render_mask(diff_damage, (255, 80, 80), args.tile_size) if diff_damage is not None else blank),
        ]
        make_panel(tiles, cols=3, title_height=args.title_height).save(out_png)

    return {
        "split": split,
        "bucket": bucket,
        "sample_id": sample_id,
        "png_path": str(out_png.resolve()),
        "cloudy_s2_path": str(cloudy_path),
        "target_path": str(target_path),
        "omnicloudmask_class_path": str(class_path),
        "omnicloudmask_cld_shdw_path": str(omni_mask_path),
        "official_cld_shdw_path": str(official_mask_path) if official_mask_path else "",
        "omni_cloud_frac": fraction(omni_cloud),
        "omni_shadow_frac": fraction(omni_shadow),
        "omni_damage_frac": fraction(omni_damage),
        "official_cloud_frac": fraction(official_cloud),
        "official_shadow_frac": fraction(official_shadow),
        "official_damage_frac": fraction(official_damage),
        "damage_diff_frac": fraction(diff_damage),
    }


def write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "bucket",
        "sample_id",
        "png_path",
        "cloudy_s2_path",
        "target_path",
        "omnicloudmask_class_path",
        "omnicloudmask_cld_shdw_path",
        "official_cld_shdw_path",
        "omni_cloud_frac",
        "omni_shadow_frac",
        "omni_damage_frac",
        "official_cloud_frac",
        "official_shadow_frac",
        "official_damage_frac",
        "damage_diff_frac",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rgb_indices = tuple(int(x.strip()) for x in args.rgb_indices.split(",") if x.strip())
    if len(rgb_indices) != 3:
        raise ValueError("--rgb-indices must contain exactly three comma-separated 0-based band indices")
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    all_index: list[dict[str, Any]] = []
    for split in splits:
        manifest = args.omni_root / "manifests" / f"pairs_{split}.csv"
        rows = read_csv(manifest)
        if args.max_samples > 0:
            rows = rows[: args.max_samples]
        split_index: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            record = process_row(row, split, args, rgb_indices)  # type: ignore[arg-type]
            split_index.append(record)
            all_index.append(record)
            if idx == 1 or idx % 100 == 0 or idx == len(rows):
                print(f"[{split}] {idx}/{len(rows)} visualized")
        write_index(args.output_dir / split / "index.csv", split_index)
    write_index(args.output_dir / "index.csv", all_index)
    print(f"Done. Wrote {len(all_index)} panels under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
