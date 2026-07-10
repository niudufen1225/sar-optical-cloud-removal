#!/usr/bin/env python3
"""Generate OmniCloudMask masks for ALLClear manifest rows.

Outputs:
  1. ``class_masks/*.tif``: uint8 OmniCloudMask classes
     0=clear, 1=thick cloud, 2=thin cloud, 3=cloud shadow.
  2. ``cld_shdw/*.tif``: five-channel ALLClear-compatible mask:
     channel 0 cloud confidence/proxy, channel 1 binary cloud,
     channels 2/3/4 binary shadow copies.
  3. ``manifests/pairs_{split}.csv``: input manifest rows with
     ``cloudy_mask_path`` replaced by the Omni-compatible mask path.

The script does not overwrite official ALLClear ``data/.../cld_shdw`` files.
Use ``data.prefer_original_cld_shdw: false`` in a training YAML to consume the
generated manifest masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tifffile  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires tifffile. Install it with `pip install tifffile`.") from exc


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True, help="Directory containing pairs_train/val/test.csv.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for masks, manifests, previews, reports.")
    parser.add_argument("--model-dir", type=Path, default=Path("pretrained/omnicloudmask/v4"))
    parser.add_argument("--model-version", type=float, default=4.0)
    parser.add_argument("--download-source", choices=("hugging_face", "google_drive"), default="hugging_face")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=("float32", "fp32", "float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--red-index", type=int, default=3, help="0-based ALLClear S2 index for red/B04.")
    parser.add_argument("--green-index", type=int, default=2, help="0-based ALLClear S2 index for green/B03.")
    parser.add_argument("--nir-index", type=int, default=8, help="0-based ALLClear S2 index for NIR/B8A. Use 7 for B08.")
    parser.add_argument("--input-scale", type=float, default=10000.0, help="Divide ALLClear S2 bands by this value before inference.")
    parser.add_argument("--confidence", action="store_true", help="Also request class confidence maps from OmniCloudMask.")
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="Debug cap; 0 means all rows.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(text: str) -> str:
    text = text.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)


def to_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D S2 array, got {arr.shape}")
    if arr.shape[-1] <= 32:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.moveaxis(arr, 0, -1))


def read_rgbnir(path: Path, red: int, green: int, nir: int, scale: float) -> np.ndarray:
    arr = to_hwc(tifffile.imread(path)).astype(np.float32)
    max_idx = max(red, green, nir)
    if arr.shape[-1] <= max_idx:
        raise ValueError(f"{path} has {arr.shape[-1]} bands, cannot read index {max_idx}")
    x = np.stack([arr[..., red], arr[..., green], arr[..., nir]], axis=0)
    x = np.nan_to_num(x, nan=0.0, posinf=scale, neginf=0.0)
    if scale > 0:
        x = x / float(scale)
    return np.clip(x, 0.0, 1.5).astype(np.float32)


def class_to_allclear_compat(pred_class: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
    cls = np.asarray(pred_class)
    if cls.ndim == 3 and cls.shape[0] == 1:
        cls = cls[0]
    if cls.ndim != 2:
        raise ValueError(f"Expected class mask [H,W] or [1,H,W], got {cls.shape}")
    cloud = np.isin(cls, [1, 2]).astype(np.float32)
    shadow = (cls == 3).astype(np.float32)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.ndim != 3 or conf.shape[0] < 4:
            raise ValueError(f"Expected confidence [4,H,W], got {conf.shape}")
        cloud_prob = np.clip(conf[1] + conf[2], 0.0, 1.0)
        shadow = (conf[3] >= np.maximum.reduce([conf[0], conf[1], conf[2]])).astype(np.float32)
    else:
        cloud_prob = cloud
    compat = np.stack([cloud_prob, cloud, shadow, shadow, shadow], axis=0).astype(np.float32)
    return compat


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        band = rgb[c]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.quantile(finite, [0.01, 0.995])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 0.35
        out[c] = np.clip((band - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0) ** 0.85
    return out


def save_preview(path: Path, rgbnir: np.ndarray, pred_class: np.ndarray, compat: np.ndarray) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rgb = stretch_rgb(rgbnir[[0, 1, 2]])
    cls = pred_class[0] if pred_class.ndim == 3 else pred_class
    cloud = compat[1]
    shadow = compat[3]
    panels = [
        np.moveaxis(rgb, 0, -1),
        np.dstack([cloud * 0.10, cloud * 0.78, cloud * 1.00]),
        np.dstack([shadow * 1.00, shadow * 0.62, shadow * 0.10]),
        np.dstack([
            np.isin(cls, [1]).astype(np.float32),
            np.isin(cls, [2]).astype(np.float32),
            (cls == 3).astype(np.float32),
        ]),
    ]
    titles = ["RGB/NIR", "Cloud", "Shadow", "Class RGB"]
    tile_h, tile_w = panels[0].shape[:2]
    pad, title_h = 5, 24
    canvas = Image.new("RGB", (4 * tile_w + 5 * pad, tile_h + title_h + 2 * pad), (12, 16, 22))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for idx, panel in enumerate(panels):
        x = pad + idx * (tile_w + pad)
        y = pad
        draw.rectangle((x, y, x + tile_w, y + title_h), fill=(28, 35, 46))
        draw.text((x + 6, y + 5), titles[idx], fill=(238, 242, 247), font=font)
        img = Image.fromarray((np.clip(panel, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        canvas.paste(img, (x, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def output_paths(output_root: Path, split: str, sample_id: str) -> tuple[Path, Path, Path]:
    stem = safe_stem(sample_id)
    return (
        output_root / "class_masks" / split / f"{stem}_omnicloudmask_class.tif",
        output_root / "cld_shdw" / split / f"{stem}_omnicloudmask_cld_shdw.tif",
        output_root / "previews" / split / f"{stem}_preview.png",
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    from omnicloudmask import predict_from_array
    from omnicloudmask.__version__ import __version__ as omnicloudmask_version

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    manifest_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "omnicloudmask_version": omnicloudmask_version,
        "model_version": float(args.model_version),
        "model_dir": str(args.model_dir.resolve()),
        "source_manifest_dir": str(args.manifest_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "band_indices_0_based": {"red": args.red_index, "green": args.green_index, "nir": args.nir_index},
        "input_scale": float(args.input_scale),
        "splits": {},
    }

    for split in splits:
        rows = read_csv(args.manifest_dir / f"pairs_{split}.csv")
        if args.max_rows_per_split > 0:
            rows = rows[: args.max_rows_per_split]
        out_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        previews_saved = 0
        for idx, row in enumerate(rows, start=1):
            sample_id = row.get("sample_id") or f"{row.get('roi_id', 'sample')}_{idx:06d}"
            cloudy_path = Path(row["cloudy_s2_path"])
            class_path, compat_path, preview_path = output_paths(args.output_root, split, sample_id)
            if compat_path.exists() and class_path.exists() and not args.overwrite:
                pred_class = tifffile.imread(class_path)
                compat = tifffile.imread(compat_path)
                if compat.ndim == 3 and compat.shape[-1] == 5:
                    compat = np.moveaxis(compat, -1, 0)
            else:
                rgbnir = read_rgbnir(cloudy_path, args.red_index, args.green_index, args.nir_index, args.input_scale)
                confidence = None
                if args.confidence:
                    confidence = predict_from_array(
                        rgbnir,
                        patch_size=int(args.patch_size),
                        patch_overlap=int(args.patch_overlap),
                        batch_size=int(args.batch_size),
                        inference_device=args.device,
                        mosaic_device="cpu",
                        inference_dtype=args.dtype,
                        export_confidence=True,
                        softmax_output=True,
                        no_data_value=0,
                        apply_no_data_mask=True,
                        destination_model_dir=args.model_dir,
                        model_download_source=args.download_source,
                        model_version=float(args.model_version),
                    ).astype(np.float32)
                    pred_class = np.argmax(confidence, axis=0).astype(np.uint8)[None, ...]
                else:
                    pred_class = predict_from_array(
                        rgbnir,
                        patch_size=int(args.patch_size),
                        patch_overlap=int(args.patch_overlap),
                        batch_size=int(args.batch_size),
                        inference_device=args.device,
                        mosaic_device="cpu",
                        inference_dtype=args.dtype,
                        export_confidence=False,
                        no_data_value=0,
                        apply_no_data_mask=True,
                        destination_model_dir=args.model_dir,
                        model_download_source=args.download_source,
                        model_version=float(args.model_version),
                    ).astype(np.uint8)
                compat = class_to_allclear_compat(pred_class, confidence)
                class_path.parent.mkdir(parents=True, exist_ok=True)
                compat_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(class_path, pred_class.astype(np.uint8))
                tifffile.imwrite(compat_path, compat.astype(np.float32))
                if previews_saved < args.preview_count:
                    save_preview(preview_path, rgbnir, pred_class, compat)
                    previews_saved += 1

            cloud_frac = float(np.mean(compat[1] >= 0.5))
            shadow_frac = float(np.mean((compat[3] >= 0.5) & ~(compat[1] >= 0.5)))
            damage_frac = float(np.mean((compat[1] >= 0.5) | (compat[3] >= 0.5)))
            counts["rows"] += 1
            counts["cloud_pixels"] += int(np.sum(compat[1] >= 0.5))
            counts["shadow_pixels"] += int(np.sum((compat[3] >= 0.5) & ~(compat[1] >= 0.5)))

            new_row = dict(row)
            new_row["cloudy_mask_path"] = str(compat_path.resolve())
            new_row["omnicloudmask_class_path"] = str(class_path.resolve())
            new_row["omnicloudmask_cld_shdw_path"] = str(compat_path.resolve())
            new_row["omnicloudmask_cloud_frac"] = f"{cloud_frac:.8f}"
            new_row["omnicloudmask_shadow_noncloud_frac"] = f"{shadow_frac:.8f}"
            new_row["omnicloudmask_damage_frac"] = f"{damage_frac:.8f}"
            out_rows.append(new_row)
            audit_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "cloudy_s2_path": str(cloudy_path),
                    "class_path": str(class_path.resolve()),
                    "cld_shdw_path": str(compat_path.resolve()),
                    "cloud_frac": f"{cloud_frac:.8f}",
                    "shadow_noncloud_frac": f"{shadow_frac:.8f}",
                    "damage_frac": f"{damage_frac:.8f}",
                }
            )
            if idx % 100 == 0:
                print(f"[{split}] {idx}/{len(rows)} rows processed")

        fieldnames: list[str] = []
        for row in out_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(args.output_root / "manifests" / f"pairs_{split}.csv", out_rows, fieldnames)
        write_csv(args.output_root / "audit" / f"{split}_mask_audit.csv", audit_rows, list(audit_rows[0].keys()) if audit_rows else ["split"])
        summary["splits"][split] = {
            "rows": int(counts["rows"]),
            "previews_saved": previews_saved,
            "mean_cloud_frac": float(np.mean([float(r["cloud_frac"]) for r in audit_rows])) if audit_rows else 0.0,
            "mean_shadow_noncloud_frac": float(np.mean([float(r["shadow_noncloud_frac"]) for r in audit_rows])) if audit_rows else 0.0,
            "mean_damage_frac": float(np.mean([float(r["damage_frac"]) for r in audit_rows])) if audit_rows else 0.0,
        }
        manifest_rows.extend(out_rows)

    if manifest_rows:
        fieldnames = []
        for row in manifest_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(args.output_root / "manifests" / "pairs_all.csv", manifest_rows, fieldnames)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
