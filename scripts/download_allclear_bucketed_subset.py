#!/usr/bin/env python3
"""Download an AllClear ROI subset and build cloud-bucketed pair manifests.

The script is intentionally conservative:
  * It reads an official AllClear task metadata JSON, such as
    train_tx3_s2-s1_10pct.json.
  * It downloads ROI archives from the official AllClear layout:
    http://allclear.cs.cornell.edu/dataset/allclear/data/{roi_id}.tar.gz
  * It computes cloud/shadow ratio from AllClear official cld_shdw masks.
  * It creates balanced pair manifests for cloud-removal experiments.

Default bucket policy used by the current ALLClear cloud-removal configs:
  low:    [0.01, 0.20)
  medium: [0.20, 0.60)
  high:   [0.60, 0.90)
  heavy:  [0.90, 1.01)

Use ``--bucket-target-counts low:100,heavy:1500`` when the final manifest
should cap low-cloud pairs while continuing to collect heavy-cloud pairs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import shutil
import tarfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_URL = "http://allclear.cs.cornell.edu/dataset/allclear"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    split: str
    bucket: str
    roi_id: str
    cloudy_date: str
    clear_date: str
    date_delta_days: int
    degraded_ratio: float
    target_degraded_ratio: float
    cloudy_s2_path: str
    clear_s2_path: str
    cloudy_mask_path: str
    clear_mask_path: str
    sar_s1_path: str
    sar_s1_date: str
    sar_cloudy_delta_days: str
    latitude: str
    longitude: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-json", required=True, help="Official AllClear metadata JSON.")
    parser.add_argument("--output-root", required=True, help="Output root. ROI archives/data and manifests are written here.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--download-timeout", type=int, default=180)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--keep-modalities",
        default="s2_toa,s1,cld_shdw",
        help="Comma-separated AllClear modality folders to keep/extract. Default excludes landsat8/landsat9/dw.",
    )
    parser.add_argument(
        "--no-prune-existing",
        action="store_true",
        help="Do not remove unneeded modality folders from already extracted ROI directories.",
    )
    parser.add_argument(
        "--redownload-existing-rois",
        action="store_true",
        help="Force re-download selected ROI archives even when ROI directories already exist. Use for recovery only.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually download missing ROI archives. Omit for dry-run.")
    parser.add_argument(
        "--download-until-balanced-pairs",
        type=int,
        default=0,
        help="Incrementally download ROI batches until the strict balanced manifest reaches this many pairs.",
    )
    parser.add_argument(
        "--check-every-rois",
        type=int,
        default=50,
        help="When --download-until-balanced-pairs is set, rebuild manifests after this many newly attempted ROI downloads.",
    )
    parser.add_argument(
        "--prune-unreferenced-on-stop",
        action="store_true",
        help="After reaching target pair count, delete local data/*.tif not referenced by the final manifest.",
    )
    parser.add_argument(
        "--prune-during-download",
        action="store_true",
        help="After each incremental check, delete TIFFs from surplus qualified pairs and non-kept modalities.",
    )
    parser.add_argument(
        "--prune-buffer-ratio",
        type=float,
        default=0.15,
        help="Extra per-bucket candidate buffer kept during online pruning, relative to target bucket count.",
    )
    parser.add_argument("--build-manifest-only", action="store_true", help="Skip download step and only use existing extracted data.")
    parser.add_argument("--selected-rois", default="", help="Optional ROI txt list. Overrides metadata ROI sampling.")
    parser.add_argument(
        "--preserve-selected-roi-order",
        action="store_true",
        help="Do not shuffle --selected-rois during incremental download. Use with a priority ROI list.",
    )
    parser.add_argument("--roi-list-out", default="", help="Optional selected ROI list output path.")
    parser.add_argument("--max-rois", type=int, default=0, help="Cap ROI count from metadata. 0 means all metadata ROIs.")
    parser.add_argument(
        "--require-s1",
        action="store_true",
        help="Keep only pairs with a local Sentinel-1 SAR file near the cloudy timestamp.",
    )
    parser.add_argument(
        "--no-local-rois-only",
        action="store_true",
        help="With --build-manifest-only, do not restrict selected ROIs to already extracted local ROI folders.",
    )
    parser.add_argument("--split-ratios", default="0.8,0.1,0.1")
    parser.add_argument("--split-unit", choices=["roi", "sample"], default="roi")
    parser.add_argument("--bucket-bins", default="low:0.01:0.20,medium:0.20:0.60,high:0.60:0.90,heavy:0.90:1.01")
    parser.add_argument("--bucket-fractions", default="low:0.10,medium:0.40,high:0.40,heavy:0.10")
    parser.add_argument(
        "--bucket-target-counts",
        default="",
        help=(
            "Optional final pair caps/targets by bucket, e.g. low:100,heavy:1500. "
            "Buckets not listed are kept unless --drop-unlisted-target-buckets is set. "
            "This bypasses fraction balancing."
        ),
    )
    parser.add_argument(
        "--drop-unlisted-target-buckets",
        action="store_true",
        help="With --bucket-target-counts, drop buckets not explicitly listed instead of keeping all of them.",
    )
    parser.add_argument(
        "--download-until-bucket-targets",
        default="",
        help=(
            "Incremental download stop condition by bucket, e.g. heavy:1500. "
            "Defaults to --bucket-target-counts when omitted and target counts are provided."
        ),
    )
    parser.add_argument(
        "--keep-all-qualified",
        action="store_true",
        help="Do not downsample by bucket fractions; keep every accepted local pair after split assignment.",
    )
    parser.add_argument("--balance-each-split", action="store_true", default=True)
    parser.add_argument("--no-balance-each-split", dest="balance_each_split", action="store_false")
    parser.add_argument("--max-pairs-per-split", type=int, default=0, help="Optional cap after balancing. 0 keeps maximum possible.")
    parser.add_argument("--target-max-degraded", type=float, default=0.10, help="Reject target images above this cloud+shadow ratio.")
    parser.add_argument("--cloud-channel", type=int, default=1, help="AllClear cld_shdw binary cloud channel.")
    parser.add_argument("--shadow-channel", type=int, default=3, help="AllClear cld_shdw shadow channel. 3 matches dark-pixel 0.25 in prior scripts.")
    parser.add_argument("--exclude-shadow", action="store_true", help="Use only cloud channel, ignoring shadow.")
    parser.add_argument("--cache-visible-masks", action="store_true", help="Write visible-white PNG masks for downstream LaMa conversion.")
    parser.add_argument("--manifest-dir", default="", help="Defaults to <output-root>/manifests.")
    parser.add_argument(
        "--base-manifest-dir",
        default="",
        help=(
            "Optional existing manifest directory whose pairs_all.csv is used as a non-downloaded base pool. "
            "Useful when local ROI folders were pruned but a previously validated manifest is still valid."
        ),
    )
    parser.add_argument(
        "--base-manifest-csv",
        default="",
        help="Optional existing pairs_all.csv path. Overrides --base-manifest-dir.",
    )
    return parser.parse_args()


def _date(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


def _parse_splits(text: str) -> Tuple[float, float, float]:
    values = tuple(float(x) for x in text.split(","))
    if len(values) != 3:
        raise ValueError("--split-ratios must be train,val,test")
    total = sum(values)
    if total <= 0:
        raise ValueError("split ratios must sum to > 0")
    return tuple(v / total for v in values)  # type: ignore[return-value]


def _parse_bins(text: str) -> List[Tuple[str, float, float]]:
    bins: List[Tuple[str, float, float]] = []
    for item in text.split(","):
        name, lo, hi = item.split(":")
        bins.append((name, float(lo), float(hi)))
    return bins


def _parse_fractions(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in text.split(","):
        name, value = item.split(":")
        out[name] = float(value)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("bucket fractions must sum to > 0")
    return {k: v / total for k, v in out.items()}


def _parse_counts(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", 1)
        count = int(value)
        if count < 0:
            raise ValueError(f"Bucket target count must be >= 0: {item}")
        out[name.strip()] = count
    return out


def _bucket(score: float, bins: Sequence[Tuple[str, float, float]]) -> Optional[str]:
    for name, lo, hi in bins:
        if lo <= score < hi:
            return name
    return None


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict metadata JSON, got {type(data)}")
    return data


def _metadata_roi_id(item: dict) -> str:
    roi = item.get("roi")
    if isinstance(roi, list) and roi:
        return str(roi[0])
    if isinstance(roi, str):
        return roi
    raise ValueError(f"Cannot parse ROI from metadata item: {item.keys()}")


def _metadata_latlon(item: dict) -> Tuple[str, str]:
    roi = item.get("roi")
    if isinstance(roi, list) and len(roi) > 1 and isinstance(roi[1], list) and len(roi[1]) >= 2:
        return str(roi[1][0]), str(roi[1][1])
    return "", ""


def _load_selected_rois(path: str) -> Optional[List[str]]:
    if not path:
        return None
    rois: List[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        roi = line.strip()
        if not roi or roi in seen:
            continue
        rois.append(roi)
        seen.add(roi)
    return rois


def _select_rois(metadata: dict, args: argparse.Namespace) -> List[str]:
    requested = _load_selected_rois(args.selected_rois)
    if requested is not None:
        return requested
    roi_ids = sorted({_metadata_roi_id(item) for item in metadata.values()})
    if args.max_rois and args.max_rois < len(roi_ids):
        rng = random.Random(args.seed)
        roi_ids = sorted(rng.sample(roi_ids, args.max_rois))
    return roi_ids


def _local_roi_ids(output_root: Path) -> set[str]:
    data_dir = output_root / "data"
    if not data_dir.exists():
        return set()
    return {path.name for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("roi")}


def _localize_path(path_text: str, output_root: Path) -> Path:
    path = Path(path_text)
    parts = path.parts
    if "dataset_30k_v4" in parts:
        idx = parts.index("dataset_30k_v4")
        rel = Path(*parts[idx + 1 :])
    else:
        roi_idx = next((i for i, part in enumerate(parts) if part.startswith("roi")), None)
        if roi_idx is None:
            raise ValueError(f"Cannot map AllClear path to local data root: {path_text}")
        rel = Path(*parts[roi_idx:])
    return output_root / "data" / rel


def _official_mask_path(s2_path: Path, output_root: Path) -> Optional[Path]:
    try:
        rel = s2_path.relative_to(output_root / "data")
    except ValueError:
        return None
    parts = list(rel.parts)
    if "s2_toa" not in parts:
        return None
    filename = rel.name.replace("_s2_toa_", "_cld_shdw_")
    mask_rel = Path(*parts[:-2]) / "cld_shdw" / filename
    candidate = output_root / "data" / mask_rel
    return candidate if candidate.exists() else None


def _read_tiff_hwc(path: Path) -> np.ndarray:
    import numpy as np
    from PIL import Image

    try:
        import tifffile  # type: ignore

        arr = tifffile.imread(path)
    except Exception:
        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported raster shape {arr.shape} for {path}")
    return arr.astype(np.float32)


def _read_degraded_ratio(mask_path: Path, cloud_channel: int, shadow_channel: int, include_shadow: bool) -> float:
    arr = _read_tiff_hwc(mask_path)
    if arr.shape[-1] <= cloud_channel:
        raise ValueError(f"Mask {mask_path} has shape {arr.shape}; no cloud channel {cloud_channel}")
    cloud = arr[..., cloud_channel] >= 0.5
    if include_shadow:
        if arr.shape[-1] <= shadow_channel:
            raise ValueError(f"Mask {mask_path} has shape {arr.shape}; no shadow channel {shadow_channel}")
        cloud = cloud | (arr[..., shadow_channel] >= 0.5)
    return float(cloud.mean())


def _cache_visible_mask(mask_path: Path, output_root: Path, manifest_dir: Path, cloud_channel: int, shadow_channel: int, include_shadow: bool) -> str:
    import numpy as np
    from PIL import Image

    arr = _read_tiff_hwc(mask_path)
    cloud = arr[..., cloud_channel] >= 0.5
    if include_shadow:
        cloud = cloud | (arr[..., shadow_channel] >= 0.5)
    visible = (~cloud).astype(np.uint8) * 255
    try:
        rel = mask_path.relative_to(output_root / "data").with_suffix(".png")
    except ValueError:
        rel = Path(mask_path.stem + ".png")
    out = manifest_dir / "visible_masks" / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visible, mode="L").save(out)
    return str(out)


def _verify_tar(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as tar:
            tar.next()
        return True
    except Exception:
        return False


def _parse_keep_modalities(text: str) -> set[str]:
    keep = {item.strip() for item in text.split(",") if item.strip()}
    if not keep:
        raise ValueError("--keep-modalities cannot be empty")
    return keep


def _tar_member_allowed(member_name: str, keep_modalities: set[str]) -> bool:
    parts = Path(member_name).parts
    # Keep top-level ROI/year directories so allowed modality members have parents.
    if len(parts) < 3:
        return True
    modality = parts[2]
    return modality in keep_modalities


def _safe_extract_selected(tar: tarfile.TarFile, dest: Path, keep_modalities: set[str]) -> int:
    members = [m for m in tar.getmembers() if _tar_member_allowed(m.name, keep_modalities)]
    tar.extractall(path=dest, members=members)
    return len(members)


def _prune_roi_dir(roi_dir: Path, keep_modalities: set[str]) -> int:
    """Remove modality folders not requested under an extracted AllClear ROI."""
    removed = 0
    if not roi_dir.exists():
        return 0
    for month_dir in roi_dir.iterdir():
        if not month_dir.is_dir():
            continue
        for child in month_dir.iterdir():
            if child.is_dir() and child.name not in keep_modalities:
                shutil.rmtree(child)
                removed += 1
    return removed


def _download_file(url: str, dest: Path, timeout: int, retries: int, retry_sleep: float) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - report and retry network failures.
            last_exc = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < max(1, retries):
                time.sleep(retry_sleep)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}") from last_exc


def _download_one(
    roi_id: str,
    output_root: Path,
    base_url: str,
    keep_archives: bool,
    execute: bool,
    timeout: int,
    retries: int,
    retry_sleep: float,
    keep_modalities: set[str],
    prune_existing: bool,
    redownload_existing: bool,
) -> Tuple[str, str]:
    data_dir = output_root / "data"
    roi_dir = data_dir / roi_id
    archive = data_dir / f"{roi_id}.tar.gz"
    if roi_dir.exists():
        if redownload_existing:
            if not execute:
                return roi_id, "dry_run_redownload_existing"
            shutil.rmtree(roi_dir)
        else:
            # A failed or over-aggressive cleanup can leave an ROI directory with
            # only *_metadata.csv files. Treat that state as incomplete; otherwise
            # recovery runs will skip the ROI forever while manifests still point to
            # missing .tif payloads.
            has_tif = any(roi_dir.rglob("*.tif"))
            if not has_tif:
                if not execute:
                    return roi_id, "dry_run_incomplete_no_tif"
                shutil.rmtree(roi_dir)
            else:
                removed = _prune_roi_dir(roi_dir, keep_modalities) if prune_existing else 0
                return roi_id, f"skip_extracted_pruned_{removed}" if removed else "skip_extracted"
    if not execute:
        return roi_id, "dry_run_missing"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or not _verify_tar(archive):
        if archive.exists():
            archive.unlink()
        _download_file(f"{base_url}/data/{roi_id}.tar.gz", archive, timeout, retries, retry_sleep)
        time.sleep(0.05)
    if not _verify_tar(archive):
        raise RuntimeError(f"Invalid archive after download: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract_selected(tar, data_dir, keep_modalities)
    removed = _prune_roi_dir(roi_dir, keep_modalities) if prune_existing else 0
    if not keep_archives:
        archive.unlink(missing_ok=True)
    return roi_id, f"downloaded_pruned_{removed}" if removed else "downloaded"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_base_manifest_records(args: argparse.Namespace) -> List[PairRecord]:
    csv_path_text = args.base_manifest_csv
    if not csv_path_text and args.base_manifest_dir:
        csv_path_text = str(Path(args.base_manifest_dir) / "pairs_all.csv")
    if not csv_path_text:
        return []
    path = Path(csv_path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing base manifest CSV: {path}")
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    records: List[PairRecord] = []
    for row in rows:
        sample_id = row.get("sample_id") or f"{row.get('roi_id', '')}_{Path(row.get('cloudy_s2_path', '')).stem}"
        if not sample_id or not row.get("bucket"):
            continue
        records.append(
            PairRecord(
                sample_id=sample_id,
                split=row.get("split", ""),
                bucket=row.get("bucket", ""),
                roi_id=row.get("roi_id", ""),
                cloudy_date=row.get("cloudy_date", ""),
                clear_date=row.get("clear_date", ""),
                date_delta_days=_safe_int(row.get("date_delta_days"), 0),
                degraded_ratio=_safe_float(row.get("degraded_ratio"), 0.0),
                target_degraded_ratio=_safe_float(row.get("target_degraded_ratio"), 0.0),
                cloudy_s2_path=row.get("cloudy_s2_path", ""),
                clear_s2_path=row.get("clear_s2_path", ""),
                cloudy_mask_path=row.get("cloudy_mask_path", ""),
                clear_mask_path=row.get("clear_mask_path", ""),
                sar_s1_path=row.get("sar_s1_path", ""),
                sar_s1_date=row.get("sar_s1_date", ""),
                sar_cloudy_delta_days=row.get("sar_cloudy_delta_days", ""),
                latitude=row.get("latitude", ""),
                longitude=row.get("longitude", ""),
            )
        )
    return records


def _dedupe_records(records: Sequence[PairRecord]) -> List[PairRecord]:
    deduped: List[PairRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (record.sample_id, record.cloudy_s2_path, record.clear_s2_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _append_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _nearest_s1(item: dict, cloudy_dt: dt.datetime, output_root: Path) -> Tuple[str, str, str]:
    best: Optional[Tuple[int, str, Path]] = None
    for date_text, path_text in item.get("s1", []):
        sar_path = _localize_path(path_text, output_root)
        if not sar_path.exists():
            continue
        delta = abs((_date(date_text) - cloudy_dt).days)
        if best is None or delta < best[0]:
            best = (delta, date_text, sar_path)
    if best is None:
        return "", "", ""
    return str(best[2]), best[1], str(best[0])


def _split_keys(keys: Sequence[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    items = list(keys)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n_train + n_val >= n and n >= 3:
        n_val = max(1, n_val)
        n_train = n - n_val - 1
    out = {}
    for idx, key in enumerate(items):
        out[key] = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
    return out


def _balance(records: Sequence[PairRecord], fractions: Dict[str, float], max_pairs: int, seed: int) -> List[PairRecord]:
    by_bucket: Dict[str, List[PairRecord]] = defaultdict(list)
    for record in records:
        by_bucket[record.bucket].append(record)
    if not records:
        return []
    available_limits = []
    for bucket, frac in fractions.items():
        if frac <= 0:
            continue
        available_limits.append(len(by_bucket[bucket]) / frac)
    if not available_limits:
        return list(records)
    target_total = int(min(available_limits))
    if max_pairs > 0:
        target_total = min(target_total, max_pairs)
    rng = random.Random(seed)
    selected: List[PairRecord] = []
    for bucket, frac in fractions.items():
        n = min(len(by_bucket[bucket]), int(round(target_total * frac)))
        items = list(by_bucket[bucket])
        rng.shuffle(items)
        selected.extend(items[:n])
    return sorted(selected, key=lambda r: (r.split, r.bucket, r.roi_id, r.sample_id))


def _select_bucket_target_counts(
    records: Sequence[PairRecord],
    target_counts: Dict[str, int],
    *,
    keep_unlisted: bool,
    seed: int,
) -> List[PairRecord]:
    if not target_counts:
        return list(records)
    by_bucket: Dict[str, List[PairRecord]] = defaultdict(list)
    for record in records:
        by_bucket[record.bucket].append(record)

    rng = random.Random(seed)
    selected: List[PairRecord] = []
    for bucket in sorted(by_bucket):
        items = list(by_bucket[bucket])
        if bucket not in target_counts:
            if keep_unlisted:
                selected.extend(items)
            continue
        target = target_counts[bucket]
        if target <= 0:
            continue
        rng.shuffle(items)
        selected.extend(items[: min(len(items), target)])
    return sorted(selected, key=lambda r: (r.bucket, r.roi_id, r.sample_id))


def _bucket_targets_reached(summary: dict, target_counts: Dict[str, int]) -> bool:
    if not target_counts:
        return False
    bucket_counts = summary.get("bucket_counts", {})
    return all(int(bucket_counts.get(bucket, 0)) >= int(target) for bucket, target in target_counts.items())


def _write_manifests_and_summary(
    metadata: dict,
    selected_rois: Sequence[str],
    args: argparse.Namespace,
    manifest_dir: Path,
    local_rois: set[str],
    include_shadow: bool,
) -> Tuple[dict, List[PairRecord]]:
    scanned_records, candidate_stats = _build_candidates(metadata, set(selected_rois), args, manifest_dir)
    base_records = _load_base_manifest_records(args)
    records = _dedupe_records([*base_records, *scanned_records])
    raw_candidate_count = len(records)
    raw_candidate_bucket_counts = dict(Counter(r.bucket for r in records))
    target_counts = _parse_counts(args.bucket_target_counts)
    target_count_selection = bool(target_counts)
    if target_count_selection:
        records = _select_bucket_target_counts(
            records,
            target_counts,
            keep_unlisted=not args.drop_unlisted_target_buckets,
            seed=args.seed,
        )
    split_ratios = _parse_splits(args.split_ratios)
    split_unit = args.split_unit
    split_keys = sorted({r.roi_id if split_unit == "roi" else r.sample_id for r in records})
    split_map = _split_keys(split_keys, split_ratios, args.seed)
    split_records = [
        PairRecord(**{**asdict(r), "split": split_map[r.roi_id if split_unit == "roi" else r.sample_id]})
        for r in records
    ]

    fractions = _parse_fractions(args.bucket_fractions)
    final_records: List[PairRecord] = []
    if target_count_selection:
        final_records = sorted(split_records, key=lambda r: (r.split, r.bucket, r.roi_id, r.sample_id))
    elif args.keep_all_qualified:
        final_records = sorted(split_records, key=lambda r: (r.split, r.bucket, r.roi_id, r.sample_id))
        if args.max_pairs_per_split > 0:
            capped: List[PairRecord] = []
            rng = random.Random(args.seed)
            for split in ("train", "val", "test"):
                items = [r for r in final_records if r.split == split]
                rng.shuffle(items)
                capped.extend(items[: args.max_pairs_per_split])
            final_records = sorted(capped, key=lambda r: (r.split, r.bucket, r.roi_id, r.sample_id))
    elif args.balance_each_split:
        for split in ("train", "val", "test"):
            final_records.extend(
                _balance([r for r in split_records if r.split == split], fractions, args.max_pairs_per_split, args.seed + len(split))
            )
    else:
        final_records = _balance(split_records, fractions, args.max_pairs_per_split, args.seed)

    all_rows = [asdict(r) for r in final_records]
    _write_csv(manifest_dir / "pairs_all.csv", all_rows)
    for split in ("train", "val", "test"):
        _write_csv(manifest_dir / f"pairs_{split}.csv", [r for r in all_rows if r["split"] == split])

    summary = {
        "metadata_json": str(Path(args.metadata_json).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "manifest_dir": str(manifest_dir),
        "selected_rois": len(selected_rois),
        "local_extracted_rois": len(local_rois),
        "base_manifest_pairs": len(base_records),
        "scanned_candidate_pairs": len(scanned_records),
        "raw_candidate_pairs": raw_candidate_count,
        "raw_candidate_bucket_counts": raw_candidate_bucket_counts,
        "selected_candidate_pairs": len(records),
        "selected_candidate_bucket_counts": dict(Counter(r.bucket for r in records)),
        "balanced_pairs": len(final_records),
        "candidate_filter_stats": candidate_stats,
        "bucket_bins": args.bucket_bins,
        "bucket_fractions": fractions,
        "bucket_target_counts": target_counts,
        "drop_unlisted_target_buckets": bool(args.drop_unlisted_target_buckets),
        "target_count_selection": target_count_selection,
        "keep_all_qualified": bool(args.keep_all_qualified),
        "split_counts": dict(Counter(r.split for r in final_records)),
        "bucket_counts": dict(Counter(r.bucket for r in final_records)),
        "split_bucket_counts": {
            split: dict(Counter(r.bucket for r in final_records if r.split == split))
            for split in ("train", "val", "test")
        },
        "target_max_degraded": args.target_max_degraded,
        "include_shadow": include_shadow,
        "cloud_channel": args.cloud_channel,
        "shadow_channel": args.shadow_channel,
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary, final_records


def _referenced_data_tifs(records: Sequence[PairRecord], output_root: Path) -> set[Path]:
    refs: set[Path] = set()
    for record in records:
        for text in (record.cloudy_s2_path, record.clear_s2_path, record.sar_s1_path):
            if text:
                path = Path(text).resolve()
                if path.suffix.lower() in {".tif", ".tiff"}:
                    refs.add(path)
        for text in (record.cloudy_s2_path, record.clear_s2_path):
            if not text:
                continue
            mask = _official_mask_path(Path(text), output_root)
            if mask is not None:
                refs.add(mask.resolve())
    return refs


def _prune_unreferenced_tifs(output_root: Path, records: Sequence[PairRecord]) -> dict:
    data_dir = output_root / "data"
    refs = _referenced_data_tifs(records, output_root)
    deleted = 0
    deleted_bytes = 0
    scanned = 0
    if not data_dir.exists():
        return {"scanned_tifs": 0, "referenced_tifs": len(refs), "deleted_tifs": 0, "deleted_bytes": 0}
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        scanned += 1
        resolved = path.resolve()
        if resolved in refs:
            continue
        size = path.stat().st_size
        path.unlink()
        deleted += 1
        deleted_bytes += size
    # Remove empty modality/month/roi directories left by tif deletion.
    for path in sorted(data_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return {
        "scanned_tifs": scanned,
        "referenced_tifs": len(refs),
        "deleted_tifs": deleted,
        "deleted_bytes": deleted_bytes,
    }


def _select_bucket_buffer_records(records: Sequence[PairRecord], args: argparse.Namespace, seed: int) -> List[PairRecord]:
    """Select a bounded candidate pool for online pruning.

    Keeps enough records per bucket to reach the requested balanced target plus
    a buffer. This prevents heavy or other over-represented buckets from keeping
    thousands of unused TIFFs while the download continues.
    """
    if args.download_until_balanced_pairs <= 0:
        return list(records)
    fractions = _parse_fractions(args.bucket_fractions)
    by_bucket: Dict[str, List[PairRecord]] = defaultdict(list)
    for record in records:
        by_bucket[record.bucket].append(record)
    rng = random.Random(seed)
    kept: List[PairRecord] = []
    for bucket, frac in fractions.items():
        target = int(round(args.download_until_balanced_pairs * frac))
        keep_n = max(target, int(round(target * (1.0 + max(0.0, args.prune_buffer_ratio)))))
        items = list(by_bucket[bucket])
        # Prefer records already assigned to train/val/test deterministically, then randomize within bucket.
        rng.shuffle(items)
        kept.extend(items[: min(len(items), keep_n)])
    return sorted(kept, key=lambda r: (r.bucket, r.roi_id, r.sample_id))


def _prune_to_records(output_root: Path, records_to_keep: Sequence[PairRecord], label: str) -> dict:
    report = _prune_unreferenced_tifs(output_root, records_to_keep)
    report["label"] = label
    return report


def _download_batch(
    batch: Sequence[str],
    args: argparse.Namespace,
    output_root: Path,
    keep_modalities: set[str],
) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    errors: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.cpus)) as executor:
        futures = [
            executor.submit(
                _download_one,
                roi_id,
                output_root,
                args.base_url,
                args.keep_archives,
                args.execute,
                args.download_timeout,
                args.download_retries,
                args.retry_sleep,
                keep_modalities,
                not args.no_prune_existing,
                args.redownload_existing_rois,
            )
            for roi_id in batch
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                roi_id, status = future.result()
                rows.append({"roi_id": roi_id, "status": status})
                print(f"[batch {idx}/{len(futures)}] {status} {roi_id}")
            except Exception as exc:  # noqa: BLE001 - continue other ROI downloads.
                message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                errors.append({"roi_id": "", "status": "error", "error": message})
                print(f"[batch {idx}/{len(futures)}] ERROR {message}")
    return rows, errors


def _build_candidates(
    metadata: dict,
    selected_rois: set[str],
    args: argparse.Namespace,
    manifest_dir: Path,
) -> Tuple[List[PairRecord], Dict[str, int]]:
    output_root = Path(args.output_root).resolve()
    bins = _parse_bins(args.bucket_bins)
    include_shadow = not args.exclude_shadow
    records: List[PairRecord] = []
    stats: Counter[str] = Counter()
    for key, item in metadata.items():
        roi_id = _metadata_roi_id(item)
        if roi_id not in selected_rois:
            stats["skip_roi_not_selected"] += 1
            continue
        if not item.get("target"):
            stats["skip_no_target"] += 1
            continue
        clear_date, clear_path_text = item["target"][0]
        clear_path = _localize_path(clear_path_text, output_root)
        clear_mask_path = _official_mask_path(clear_path, output_root) if clear_path.exists() else None
        if not clear_path.exists() or clear_mask_path is None:
            stats["skip_missing_clear_or_clear_mask"] += 1
            continue
        target_ratio = _read_degraded_ratio(clear_mask_path, args.cloud_channel, args.shadow_channel, include_shadow)
        if target_ratio > args.target_max_degraded:
            stats["skip_target_too_cloudy"] += 1
            continue
        clear_dt = _date(clear_date)
        lat, lon = _metadata_latlon(item)
        stats["candidate_clear_samples"] += 1
        for cloudy_date, cloudy_path_text in item.get("s2_toa", []):
            stats["candidate_cloudy_frames"] += 1
            cloudy_path = _localize_path(cloudy_path_text, output_root)
            if not cloudy_path.exists():
                stats["skip_missing_cloudy_s2"] += 1
                continue
            cloudy_mask_path = _official_mask_path(cloudy_path, output_root)
            if cloudy_mask_path is None:
                stats["skip_missing_cloudy_mask"] += 1
                continue
            ratio = _read_degraded_ratio(cloudy_mask_path, args.cloud_channel, args.shadow_channel, include_shadow)
            bucket = _bucket(ratio, bins)
            if bucket is None:
                stats["skip_cloud_ratio_outside_bins"] += 1
                continue
            cloudy_dt = _date(cloudy_date)
            sar_path, sar_date, sar_delta = _nearest_s1(item, cloudy_dt, output_root)
            if args.require_s1 and not sar_path:
                stats["skip_missing_s1"] += 1
                continue
            cached_cloudy_mask = str(cloudy_mask_path)
            cached_clear_mask = str(clear_mask_path)
            if args.cache_visible_masks:
                cached_cloudy_mask = _cache_visible_mask(
                    cloudy_mask_path, output_root, manifest_dir, args.cloud_channel, args.shadow_channel, include_shadow
                )
                cached_clear_mask = _cache_visible_mask(
                    clear_mask_path, output_root, manifest_dir, args.cloud_channel, args.shadow_channel, include_shadow
                )
            sample_id = f"{key}_{Path(cloudy_path_text).stem}"
            records.append(
                PairRecord(
                    sample_id=sample_id,
                    split="",
                    bucket=bucket,
                    roi_id=roi_id,
                    cloudy_date=cloudy_date,
                    clear_date=clear_date,
                    date_delta_days=abs((cloudy_dt - clear_dt).days),
                    degraded_ratio=ratio,
                    target_degraded_ratio=target_ratio,
                    cloudy_s2_path=str(cloudy_path),
                    clear_s2_path=str(clear_path),
                    cloudy_mask_path=cached_cloudy_mask,
                    clear_mask_path=cached_clear_mask,
                    sar_s1_path=sar_path,
                    sar_s1_date=sar_date,
                    sar_cloudy_delta_days=sar_delta,
                    latitude=lat,
                    longitude=lon,
                )
            )
            stats["accepted_pairs"] += 1
    return records, dict(stats)


def main() -> None:
    args = _parse_args()
    metadata_json = Path(args.metadata_json).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else output_root / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(metadata_json)
    selected_rois = _select_rois(metadata, args)
    local_rois = _local_roi_ids(output_root)
    if args.build_manifest_only and not args.no_local_rois_only and not args.selected_rois:
        selected_rois = sorted(set(selected_rois) & local_rois)
    selected_set = set(selected_rois)
    include_shadow = not args.exclude_shadow
    keep_modalities = _parse_keep_modalities(args.keep_modalities)
    download_target_counts = _parse_counts(args.download_until_bucket_targets)
    if not download_target_counts and args.bucket_target_counts:
        download_target_counts = _parse_counts(args.bucket_target_counts)
    roi_list_out = Path(args.roi_list_out).resolve() if args.roi_list_out else manifest_dir / "selected_rois.txt"
    roi_list_out.parent.mkdir(parents=True, exist_ok=True)
    roi_list_out.write_text("\n".join(selected_rois) + "\n", encoding="utf-8")

    download_rows = []
    download_errors = []
    if not args.build_manifest_only:
        print(f"metadata_samples={len(metadata)} selected_rois={len(selected_rois)} execute={args.execute}")
        print(f"selected_roi_list={roi_list_out}")
        if args.download_until_balanced_pairs > 0 or download_target_counts:
            # Incremental continuation is usually driven by --selected-rois as a
            # priority list of *new* ROIs to try next. Existing local ROIs must
            # still remain active for manifest building and pruning; otherwise
            # online pruning can delete valid previously downloaded samples.
            active_rois = sorted(set(selected_rois) | _local_roi_ids(output_root))
            current_local = sorted(set(active_rois) & _local_roi_ids(output_root))
            summary, final_records = _write_manifests_and_summary(metadata, current_local, args, manifest_dir, _local_roi_ids(output_root), include_shadow)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            reached_by_pairs = args.download_until_balanced_pairs > 0 and summary["balanced_pairs"] >= args.download_until_balanced_pairs
            reached_by_buckets = _bucket_targets_reached(summary, download_target_counts)
            if reached_by_pairs or reached_by_buckets:
                print(
                    "target_already_reached "
                    f"balanced_pairs={summary['balanced_pairs']} pair_target={args.download_until_balanced_pairs} "
                    f"bucket_counts={summary['bucket_counts']} bucket_targets={download_target_counts}"
                )
                if args.prune_unreferenced_on_stop:
                    prune_report = _prune_unreferenced_tifs(output_root, final_records)
                    (manifest_dir / "prune_unreferenced_report.json").write_text(
                        json.dumps(prune_report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    print(f"prune_unreferenced={json.dumps(prune_report, ensure_ascii=False)}")
                return

            remaining = [roi for roi in selected_rois if roi not in set(current_local)]
            if not args.preserve_selected_roi_order:
                rng = random.Random(args.seed)
                rng.shuffle(remaining)
            batch_size = max(1, args.check_every_rois)
            print(
                f"incremental_download_start local_rois={len(current_local)} remaining_rois={len(remaining)} "
                f"target_balanced_pairs={args.download_until_balanced_pairs} "
                f"bucket_targets={download_target_counts} batch_size={batch_size}"
            )
            for start in range(0, len(remaining), batch_size):
                batch = remaining[start : start + batch_size]
                rows, errors = _download_batch(batch, args, output_root, keep_modalities)
                _append_csv(manifest_dir / "download_log_incremental.csv", rows)
                _append_csv(manifest_dir / "download_errors_incremental.csv", errors)

                active_rois = sorted(set(selected_rois) | _local_roi_ids(output_root))
                current_local = sorted(set(active_rois) & _local_roi_ids(output_root))
                summary, final_records = _write_manifests_and_summary(
                    metadata,
                    current_local,
                    args,
                    manifest_dir,
                    _local_roi_ids(output_root),
                    include_shadow,
                )
                print(
                    "incremental_check "
                    f"attempted_rois={min(start + len(batch), len(remaining))}/{len(remaining)} "
                    f"local_rois={len(current_local)} raw_pairs={summary['raw_candidate_pairs']} "
                    f"balanced_pairs={summary['balanced_pairs']} "
                    f"bucket_counts={summary['bucket_counts']}"
                )
                if args.prune_during_download:
                    # Rebuild all qualified records, not only the strict balanced subset, then keep
                    # bounded per-bucket buffers needed to reach the target.
                    all_args = argparse.Namespace(**vars(args))
                    all_args.keep_all_qualified = True
                    all_args.bucket_target_counts = ""
                    qualified_summary, qualified_records = _write_manifests_and_summary(
                        metadata,
                        current_local,
                        all_args,
                        manifest_dir / "online_prune_qualified_pool",
                        _local_roi_ids(output_root),
                        include_shadow,
                    )
                    buffered_records = _select_bucket_buffer_records(qualified_records, args, args.seed + start)
                    prune_report = _prune_to_records(output_root, buffered_records, f"incremental_after_{start + len(batch)}")
                    _append_csv(manifest_dir / "online_prune_log.csv", [prune_report])
                    print(
                        "online_prune "
                        f"qualified_pairs={qualified_summary['balanced_pairs']} "
                        f"kept_buffer_pairs={len(buffered_records)} "
                        f"deleted_tifs={prune_report['deleted_tifs']} "
                        f"deleted_bytes={prune_report['deleted_bytes']}"
                    )
                reached_by_pairs = args.download_until_balanced_pairs > 0 and summary["balanced_pairs"] >= args.download_until_balanced_pairs
                reached_by_buckets = _bucket_targets_reached(summary, download_target_counts)
                if reached_by_pairs or reached_by_buckets:
                    print(
                        "target_reached "
                        f"balanced_pairs={summary['balanced_pairs']} pair_target={args.download_until_balanced_pairs} "
                        f"bucket_counts={summary['bucket_counts']} bucket_targets={download_target_counts}"
                    )
                    if args.prune_unreferenced_on_stop:
                        prune_report = _prune_unreferenced_tifs(output_root, final_records)
                        (manifest_dir / "prune_unreferenced_report.json").write_text(
                            json.dumps(prune_report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        print(f"prune_unreferenced={json.dumps(prune_report, ensure_ascii=False)}")
                    return
            print("target_not_reached_after_all_selected_rois")
            return

        rows, errors = _download_batch(selected_rois, args, output_root, keep_modalities)
        download_rows.extend(rows)
        download_errors.extend(errors)
        _write_csv(manifest_dir / "download_log.csv", download_rows)
        _write_csv(manifest_dir / "download_errors.csv", download_errors)
        if not args.execute:
            print("Dry run complete. Re-run with --execute to download and build complete manifests.")
            return

    summary, _final_records = _write_manifests_and_summary(metadata, selected_rois, args, manifest_dir, local_rois, include_shadow)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
