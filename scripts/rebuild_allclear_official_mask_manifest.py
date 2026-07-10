#!/usr/bin/env python3
"""Rebuild ALLClear manifests from official cloud/shadow TIFF masks.

The script is intentionally non-destructive: it writes a new manifest directory
and never deletes image files.  It recomputes cloud/shadow statistics from the
official ``data/.../cld_shdw/*.tif`` products, filters invalid pairs, assigns
new buckets, and keeps an audit trail for rows that are rejected or suspicious.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed")
DEFAULT_INPUT_DIR = ROOT / "manifests_final_clean_plus_unused300heavy_all_nonheavy"
DEFAULT_OUTPUT_DIR = ROOT / "manifests_official_mask_filtered"
SPLITS = ("train", "val", "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ordered_fieldnames(*row_groups: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for rows in row_groups:
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    return fields


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def derive_cld_shdw_from_s2(path: Path | None) -> Path | None:
    if path is None:
        return None
    parts = list(path.parts)
    try:
        idx = parts.index("s2_toa")
    except ValueError:
        return None
    parts[idx] = "cld_shdw"
    derived = Path(*parts)
    return derived.with_name(derived.name.replace("_s2_toa_", "_cld_shdw_")).with_suffix(".tif")


def official_mask_path(
    row: dict[str, str],
    *,
    root: Path,
    s2_key: str,
    manifest_mask_key: str,
    allow_manifest_mask_fallback: bool,
) -> Path | None:
    """Prefer official ``data/.../cld_shdw`` derived from S2, matching Dataset."""

    derived = derive_cld_shdw_from_s2(resolve_path(root, row.get(s2_key)))
    if derived is not None and derived.exists():
        return derived.resolve()
    fallback = resolve_path(root, row.get(manifest_mask_key))
    if allow_manifest_mask_fallback and fallback is not None and fallback.exists():
        return fallback.resolve()
    return derived.resolve() if derived is not None else fallback


def to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {arr.shape}")
    if arr.shape[0] <= 32:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("This script needs tifffile: pip install tifffile") from exc

    arr = tifffile.imread(path)
    return to_chw(np.asarray(arr, dtype=np.float32))


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mask.size == 0:
        return mask
    out = mask.copy()
    if out.shape[0] > 1:
        for channel in range(out.shape[0]):
            channel_max = float(np.nanmax(out[channel]))
            if channel_max > 1.0:
                denom = 100.0 if channel_max <= 100.0 else 255.0
                out[channel] = out[channel] / denom
    elif float(np.nanmax(out)) > 4.0:
        denom = 100.0 if float(np.nanmax(out)) <= 100.0 else 255.0
        out = out / denom
    return np.clip(out, 0.0, 1.0)


def normalize_optical(image: np.ndarray, optical_scale: float) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=optical_scale, neginf=0.0)
    return np.clip(image, 0.0, optical_scale) / optical_scale


def mean_or_nan(x: np.ndarray) -> float:
    if x.size == 0:
        return math.nan
    return float(np.mean(x))


def mask_stats(
    path: Path,
    *,
    cloud_prob_channel: int,
    cloud_channel: int,
    shadow_channel: int,
    cloud_core_threshold: float,
) -> dict[str, float]:
    mask = normalize_mask(read_tif(path))
    max_channel = max(cloud_prob_channel, cloud_channel, shadow_channel)
    if mask.shape[0] <= max_channel:
        raise ValueError(f"{path} has {mask.shape[0]} channels, needs channel {max_channel}")

    cloud_prob = np.clip(mask[cloud_prob_channel], 0.0, 1.0)
    cloud_bin = np.clip(mask[cloud_channel], 0.0, 1.0)
    shadow_raw = np.clip(mask[shadow_channel], 0.0, 1.0)
    shadow_noncloud = np.clip(shadow_raw * (1.0 - cloud_bin), 0.0, 1.0)
    damage = np.maximum(cloud_bin, shadow_noncloud)
    clear = np.clip(1.0 - damage, 0.0, 1.0)

    return {
        "cloud_prob_mean": mean_or_nan(cloud_prob),
        "cloud_bin_frac": mean_or_nan(cloud_bin),
        "cloud_core_frac": mean_or_nan((cloud_prob >= cloud_core_threshold).astype(np.float32)),
        "shadow_raw_frac": mean_or_nan(shadow_raw),
        "shadow_noncloud_frac": mean_or_nan(shadow_noncloud),
        "damage_frac": mean_or_nan(damage),
        "clear_frac": mean_or_nan(clear),
    }


def parse_bins(spec: str) -> list[tuple[str, float, float]]:
    bins: list[tuple[str, float, float]] = []
    for item in spec.split(","):
        if not item.strip():
            continue
        try:
            name, lo, hi = item.split(":")
            bins.append((name.strip(), float(lo), float(hi)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid bin spec item: {item!r}") from exc
    if not bins:
        raise argparse.ArgumentTypeError("At least one bucket bin is required.")
    return bins


def assign_bucket(value: float, bins: list[tuple[str, float, float]]) -> str | None:
    if not math.isfinite(value):
        return None
    for idx, (name, lo, hi) in enumerate(bins):
        is_last = idx == len(bins) - 1
        if lo <= value <= hi if is_last else lo <= value < hi:
            return name
    return None


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def prefixed(prefix: str, stats: dict[str, float]) -> dict[str, str]:
    return {f"{prefix}_{key}": f"{value:.8f}" for key, value in stats.items()}


def compute_pair_delta(
    row: dict[str, str],
    *,
    root: Path,
    cloud_mask_path: Path,
    cloud_channel: int,
    shadow_channel: int,
    optical_scale: float,
) -> dict[str, str]:
    cloudy_path = resolve_path(root, row.get("cloudy_s2_path"))
    clear_path = resolve_path(root, row.get("clear_s2_path"))
    if cloudy_path is None or clear_path is None or not cloudy_path.exists() or not clear_path.exists():
        return {"paired_delta_status": "missing_s2"}

    cloudy = normalize_optical(read_tif(cloudy_path), optical_scale)
    clear = normalize_optical(read_tif(clear_path), optical_scale)
    mask = normalize_mask(read_tif(cloud_mask_path))
    if cloudy.shape != clear.shape or cloudy.shape[-2:] != mask.shape[-2:]:
        return {
            "paired_delta_status": "shape_mismatch",
            "cloudy_shape": "x".join(map(str, cloudy.shape)),
            "clear_shape": "x".join(map(str, clear.shape)),
            "mask_shape": "x".join(map(str, mask.shape)),
        }
    if mask.shape[0] <= max(cloud_channel, shadow_channel):
        return {"paired_delta_status": "mask_channel_missing"}

    cloud = np.clip(mask[cloud_channel], 0.0, 1.0)
    shadow = np.clip(mask[shadow_channel], 0.0, 1.0) * (1.0 - cloud)
    damage = np.maximum(cloud, shadow)
    clear_support = 1.0 - damage
    delta = np.mean(np.abs(cloudy - clear), axis=0)

    cloud_weight = float(np.mean(cloud))
    damage_weight = float(np.mean(damage))
    clear_weight = float(np.mean(clear_support))
    masked_delta = float(np.sum(delta * cloud) / max(float(np.sum(cloud)), 1.0e-6))
    damage_delta = float(np.sum(delta * damage) / max(float(np.sum(damage)), 1.0e-6))
    clear_delta = float(np.sum(delta * clear_support) / max(float(np.sum(clear_support)), 1.0e-6))
    ratio = masked_delta / max(clear_delta, 1.0e-6)

    return {
        "paired_delta_status": "ok",
        "cloud_mask_weight": f"{cloud_weight:.8f}",
        "damage_mask_weight": f"{damage_weight:.8f}",
        "clear_support_weight": f"{clear_weight:.8f}",
        "cloudy_target_cloud_l1": f"{masked_delta:.8f}",
        "cloudy_target_damage_l1": f"{damage_delta:.8f}",
        "cloudy_target_clear_l1": f"{clear_delta:.8f}",
        "cloudy_target_cloud_to_clear_l1_ratio": f"{ratio:.8f}",
    }


def reliability_flag(row: dict[str, Any], args: argparse.Namespace) -> str:
    if row.get("paired_delta_status") != "ok":
        return "not_computed"
    cloud_frac = safe_float(str(row.get("cloudy_cloud_bin_frac", "")))
    masked_delta = safe_float(str(row.get("cloudy_target_cloud_l1", "")))
    ratio = safe_float(str(row.get("cloudy_target_cloud_to_clear_l1_ratio", "")))
    if (
        cloud_frac >= args.suspect_full_cloud_min
        and masked_delta <= args.suspect_delta_max
        and ratio <= args.suspect_delta_ratio_max
    ):
        return "suspect_full_cloud_low_pair_delta"
    return "ok"


def reject_reason(row: dict[str, Any], args: argparse.Namespace, active_bucket: str | None) -> str | None:
    if row.get("missing_cloudy_mask") == "1":
        return "missing_cloudy_official_mask"
    if row.get("missing_clear_mask") == "1" and args.target_mask_missing_policy == "reject":
        return "missing_clear_official_mask"
    if args.require_sar:
        sar_path = Path(str(row.get("sar_s1_path", ""))).expanduser()
        if not str(row.get("sar_s1_path", "")) or not sar_path.exists():
            return "missing_sar"
    if active_bucket is None:
        return "source_fraction_outside_bins"
    if safe_float(str(row.get("target_cloud_bin_frac", ""))) > args.target_max_cloud:
        return "target_cloud_too_high"
    if safe_float(str(row.get("target_shadow_noncloud_frac", ""))) > args.target_max_shadow:
        return "target_shadow_too_high"
    if safe_float(str(row.get("target_damage_frac", ""))) > args.target_max_damage:
        return "target_damage_too_high"
    if args.exclude_suspect and row.get("mask_reliability_flag") != "ok":
        return str(row.get("mask_reliability_flag"))
    return None


def process_row(row: dict[str, str], args: argparse.Namespace, bins: list[tuple[str, float, float]]) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    out: dict[str, Any] = dict(row)
    out["original_bucket"] = row.get("bucket", "")
    out["original_degraded_ratio"] = row.get("degraded_ratio", "")
    out["original_target_degraded_ratio"] = row.get("target_degraded_ratio", "")

    cloudy_mask_path = official_mask_path(
        row,
        root=root,
        s2_key="cloudy_s2_path",
        manifest_mask_key="cloudy_mask_path",
        allow_manifest_mask_fallback=args.allow_manifest_mask_fallback,
    )
    clear_mask_path = official_mask_path(
        row,
        root=root,
        s2_key="clear_s2_path",
        manifest_mask_key="clear_mask_path",
        allow_manifest_mask_fallback=args.allow_manifest_mask_fallback,
    )
    out["official_cloudy_mask_path"] = str(cloudy_mask_path) if cloudy_mask_path else ""
    out["official_clear_mask_path"] = str(clear_mask_path) if clear_mask_path else ""
    out["missing_cloudy_mask"] = "0" if cloudy_mask_path and cloudy_mask_path.exists() else "1"
    out["missing_clear_mask"] = "0" if clear_mask_path and clear_mask_path.exists() else "1"

    if out["missing_cloudy_mask"] == "0":
        cloudy_stats = mask_stats(
            cloudy_mask_path,
            cloud_prob_channel=args.cloud_prob_channel,
            cloud_channel=args.cloud_channel,
            shadow_channel=args.shadow_channel,
            cloud_core_threshold=args.cloud_core_threshold,
        )
        out.update(prefixed("cloudy", cloudy_stats))
    if out["missing_clear_mask"] == "0":
        target_stats = mask_stats(
            clear_mask_path,
            cloud_prob_channel=args.cloud_prob_channel,
            cloud_channel=args.cloud_channel,
            shadow_channel=args.shadow_channel,
            cloud_core_threshold=args.cloud_core_threshold,
        )
        out.update(prefixed("target", target_stats))
        out["target_stats_source"] = "official_mask"
    elif args.target_mask_missing_policy == "use_manifest_ratio":
        target_ratio = safe_float(row.get("target_degraded_ratio"))
        if not math.isfinite(target_ratio):
            target_ratio = math.nan
        # The current pruned dataset may no longer contain clear cld_shdw TIFFs.
        # The manifest ratio was produced during curation from the official mask,
        # so we keep it as an explicit fallback instead of silently accepting.
        for key, value in {
            "target_cloud_prob_mean": target_ratio,
            "target_cloud_bin_frac": target_ratio,
            "target_cloud_core_frac": target_ratio,
            "target_shadow_raw_frac": 0.0,
            "target_shadow_noncloud_frac": 0.0,
            "target_damage_frac": target_ratio,
            "target_clear_frac": 1.0 - target_ratio if math.isfinite(target_ratio) else math.nan,
        }.items():
            out[key] = f"{value:.8f}" if math.isfinite(value) else ""
        out["target_stats_source"] = "manifest_target_degraded_ratio"
    else:
        out["target_stats_source"] = "missing_allowed"

    cloud_frac = safe_float(str(out.get("cloudy_cloud_bin_frac", "")))
    damage_frac = safe_float(str(out.get("cloudy_damage_frac", "")))
    target_cloud_frac = safe_float(str(out.get("target_cloud_bin_frac", "")))
    target_damage_frac = safe_float(str(out.get("target_damage_frac", "")))
    bucket_cloud = assign_bucket(cloud_frac, bins)
    bucket_damage = assign_bucket(damage_frac, bins)
    out["official_bucket_cloud"] = bucket_cloud or ""
    out["official_bucket_damage"] = bucket_damage or ""

    active_metric = cloud_frac if args.bucket_mode == "cloud" else damage_frac
    active_target_metric = target_cloud_frac if args.bucket_mode == "cloud" else target_damage_frac
    active_bucket = assign_bucket(active_metric, bins)
    out["official_bucket_mode"] = args.bucket_mode
    out["official_bucket_metric"] = f"{active_metric:.8f}" if math.isfinite(active_metric) else ""
    out["official_target_metric"] = f"{active_target_metric:.8f}" if math.isfinite(active_target_metric) else ""

    if args.compute_paired_delta and out["missing_cloudy_mask"] == "0":
        out.update(
            compute_pair_delta(
                row,
                root=root,
                cloud_mask_path=cloudy_mask_path,
                cloud_channel=args.cloud_channel,
                shadow_channel=args.shadow_channel,
                optical_scale=args.optical_scale,
            )
        )
    else:
        out["paired_delta_status"] = "disabled"
    out["mask_reliability_flag"] = reliability_flag(out, args)

    reason = reject_reason(out, args, active_bucket)
    out["official_filter_status"] = "rejected" if reason else "accepted"
    out["official_reject_reason"] = reason or ""
    if reason is None:
        out["bucket"] = active_bucket or ""
        out["degraded_ratio"] = out["official_bucket_metric"]
        out["target_degraded_ratio"] = out["official_target_metric"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-file", type=Path, default=None, help="Optional pairs_all.csv path; overrides --input-dir.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bucket-mode",
        choices=("cloud", "damage"),
        default="cloud",
        help="cloud is best for pure LaMa inpainting; damage includes cloud + non-cloud shadow for DADIGAN/Stage1 routing.",
    )
    parser.add_argument("--bins", type=str, default="low:0.01:0.20,medium:0.20:0.60,high:0.60:0.90,heavy:0.90:1.01")
    parser.add_argument("--cloud-prob-channel", type=int, default=0)
    parser.add_argument("--cloud-channel", type=int, default=1)
    parser.add_argument("--shadow-channel", type=int, default=3)
    parser.add_argument("--cloud-core-threshold", type=float, default=0.60)
    parser.add_argument("--target-max-cloud", type=float, default=0.01)
    parser.add_argument("--target-max-shadow", type=float, default=0.03)
    parser.add_argument("--target-max-damage", type=float, default=0.03)
    parser.add_argument(
        "--target-mask-missing-policy",
        choices=("use_manifest_ratio", "reject", "allow"),
        default="use_manifest_ratio",
        help=(
            "Policy when clear/target official cld_shdw TIFF is missing. "
            "The current pruned dataset keeps cloudy masks only, so the default "
            "uses manifest target_degraded_ratio as an explicit fallback."
        ),
    )
    parser.add_argument("--optical-scale", type=float, default=10000.0)
    parser.add_argument("--require-sar", action="store_true", help="Reject rows without an existing sar_s1_path.")
    parser.add_argument("--allow-manifest-mask-fallback", action="store_true", help="Use manifest mask path if derived official cld_shdw is missing.")
    parser.add_argument("--compute-paired-delta", action="store_true", help="Read cloudy/clear S2 and compute mask-region pair consistency audit.")
    parser.add_argument("--exclude-suspect", action="store_true", help="Reject rows flagged by the paired-delta reliability audit.")
    parser.add_argument("--suspect-full-cloud-min", type=float, default=0.90)
    parser.add_argument("--suspect-delta-max", type=float, default=0.035)
    parser.add_argument("--suspect-delta-ratio-max", type=float, default=1.50)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bins = parse_bins(args.bins)
    input_file = args.input_file.expanduser().resolve() if args.input_file else args.input_dir.expanduser().resolve() / "pairs_all.csv"
    out_dir = args.out_dir.expanduser().resolve()
    if not input_file.exists():
        raise SystemExit(f"Missing input manifest: {input_file}")

    rows = read_csv(input_file)
    if args.limit > 0:
        rows = rows[: args.limit]
    processed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        try:
            processed.append(process_row(row, args, bins))
        except Exception as exc:
            failed = dict(row)
            failed["official_filter_status"] = "rejected"
            failed["official_reject_reason"] = f"exception:{type(exc).__name__}:{exc}"
            processed.append(failed)
        if idx % 500 == 0:
            print(f"processed {idx}/{len(rows)}", file=sys.stderr)

    accepted = [row for row in processed if row.get("official_filter_status") == "accepted"]
    rejected = [row for row in processed if row.get("official_filter_status") == "rejected"]
    suspects = [row for row in processed if row.get("mask_reliability_flag", "").startswith("suspect")]

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(processed)
    write_csv(out_dir / "pairs_all.csv", accepted, fieldnames)
    write_csv(out_dir / "pairs_rejected.csv", rejected, fieldnames)
    write_csv(out_dir / "pairs_suspect.csv", suspects, fieldnames)
    for split in SPLITS:
        write_csv(out_dir / f"pairs_{split}.csv", [row for row in accepted if row.get("split") == split], fieldnames)
    (out_dir / "selected_rois.txt").write_text(
        "\n".join(sorted({str(row.get("roi_id", "")) for row in accepted if row.get("roi_id")})) + "\n",
        encoding="utf-8",
    )

    changed_bucket = [
        row
        for row in accepted
        if row.get("original_bucket") and row.get("bucket") and row.get("original_bucket") != row.get("bucket")
    ]
    summary = {
        "input_file": str(input_file),
        "out_dir": str(out_dir),
        "rows_input": len(rows),
        "rows_accepted": len(accepted),
        "rows_rejected": len(rejected),
        "rows_suspect": len(suspects),
        "bucket_mode": args.bucket_mode,
        "bins": args.bins,
        "channels": {
            "cloud_prob_channel": args.cloud_prob_channel,
            "cloud_channel": args.cloud_channel,
            "shadow_channel": args.shadow_channel,
            "cloud_core_threshold": args.cloud_core_threshold,
        },
        "target_thresholds": {
            "target_max_cloud": args.target_max_cloud,
            "target_max_shadow": args.target_max_shadow,
            "target_max_damage": args.target_max_damage,
        },
        "paired_delta": {
            "compute_paired_delta": args.compute_paired_delta,
            "exclude_suspect": args.exclude_suspect,
            "suspect_full_cloud_min": args.suspect_full_cloud_min,
            "suspect_delta_max": args.suspect_delta_max,
            "suspect_delta_ratio_max": args.suspect_delta_ratio_max,
        },
        "accepted_by_bucket": dict(Counter(str(row.get("bucket", "")) for row in accepted)),
        "accepted_by_split": dict(Counter(str(row.get("split", "")) for row in accepted)),
        "accepted_split_bucket_counts": {
            split: dict(Counter(str(row.get("bucket", "")) for row in accepted if row.get("split") == split))
            for split in SPLITS
        },
        "rejected_by_reason": dict(Counter(str(row.get("official_reject_reason", "")) for row in rejected)),
        "suspect_by_bucket": dict(Counter(str(row.get("bucket", row.get("original_bucket", ""))) for row in suspects)),
        "accepted_bucket_changed_from_original": len(changed_bucket),
        "outputs": {
            "pairs_all": str(out_dir / "pairs_all.csv"),
            "pairs_rejected": str(out_dir / "pairs_rejected.csv"),
            "pairs_suspect": str(out_dir / "pairs_suspect.csv"),
            "pairs_train": str(out_dir / "pairs_train.csv"),
            "pairs_val": str(out_dir / "pairs_val.csv"),
            "pairs_test": str(out_dir / "pairs_test.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
