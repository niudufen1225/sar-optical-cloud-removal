#!/usr/bin/env python3
"""Prioritize remaining AllClear ROIs using empirical bucket yield.

This script learns a simple ROI-level prior from already-downloaded pairs:
month + latitude band -> expected medium/high yield. It then ranks not-yet-local
ROIs from the official metadata so download_allclear_bucketed_subset.py can
download more useful ROIs first.

It does not need remote access and does not download files.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--existing-pairs-csv", required=True, help="pairs_all.csv from a keep-all qualified manifest.")
    parser.add_argument("--output-root", required=True, help="AllClear local output root containing data/.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-roi-list", required=True)
    parser.add_argument("--target-buckets", default="medium,high")
    parser.add_argument("--lat-bin-size", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 writes all remaining ROIs.")
    return parser.parse_args()


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict metadata JSON, got {type(data)}")
    return data


def _roi_id(item: dict) -> str:
    roi = item.get("roi")
    if isinstance(roi, list) and roi:
        return str(roi[0])
    if isinstance(roi, str):
        return roi
    raise ValueError(f"Cannot parse ROI from {item.keys()}")


def _lat(item: dict) -> float:
    roi = item.get("roi")
    if isinstance(roi, list) and len(roi) > 1 and isinstance(roi[1], list) and roi[1]:
        return float(roi[1][0])
    return 0.0


def _lat_bin(lat: float, size: float) -> str:
    lo = int(lat // size) * int(size)
    hi = lo + int(size)
    return f"{lo}:{hi}"


def _month_from_date(text: str) -> int:
    return dt.datetime.fromisoformat(text).month


def _local_roi_ids(output_root: Path) -> set[str]:
    data_dir = output_root / "data"
    if not data_dir.exists():
        return set()
    return {path.name for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("roi")}


def _learn_strata(existing_pairs_csv: Path, targets: set[str], lat_bin_size: float) -> Tuple[dict, dict]:
    rows = list(csv.DictReader(existing_pairs_csv.open("r", encoding="utf-8")))
    roi_bucket = Counter()
    stratum_total = Counter()
    stratum_target = Counter()
    roi_strata: Dict[str, set[Tuple[int, str]]] = defaultdict(set)
    for row in rows:
        roi = row["roi_id"]
        bucket = row["bucket"]
        month = _month_from_date(row["cloudy_date"])
        lat = float(row["latitude"]) if row.get("latitude") else 0.0
        stratum = (month, _lat_bin(lat, lat_bin_size))
        stratum_total[stratum] += 1
        roi_strata[roi].add(stratum)
        if bucket in targets:
            stratum_target[stratum] += 1
            roi_bucket[(roi, bucket)] += 1
    prior = {}
    global_rate = (sum(stratum_target.values()) + 1.0) / (sum(stratum_total.values()) + 2.0)
    for stratum in stratum_total:
        # Beta smoothing avoids over-trusting tiny strata.
        prior[stratum] = (stratum_target[stratum] + 2.0 * global_rate) / (stratum_total[stratum] + 2.0)
    return prior, {"global_rate": global_rate, "roi_bucket": roi_bucket, "roi_strata": roi_strata}


def _score_roi(item: dict, prior: dict, lat_bin_size: float, global_rate: float) -> Tuple[float, str]:
    lat = _lat(item)
    latb = _lat_bin(lat, lat_bin_size)
    scores = []
    months = []
    for date_text, _path in item.get("s2_toa", []):
        month = _month_from_date(date_text)
        months.append(str(month))
        scores.append(prior.get((month, latb), global_rate))
    if not scores:
        return 0.0, ""
    # Max is appropriate because one good cloudy frame in tx3 is enough to yield a pair.
    return max(scores), ",".join(sorted(set(months), key=int))


def main() -> None:
    args = _parse_args()
    metadata = _load_metadata(Path(args.metadata_json))
    targets = {x.strip() for x in args.target_buckets.split(",") if x.strip()}
    output_root = Path(args.output_root)
    local = _local_roi_ids(output_root)
    prior, info = _learn_strata(Path(args.existing_pairs_csv), targets, args.lat_bin_size)
    global_rate = info["global_rate"]

    rows: List[dict] = []
    for key, item in metadata.items():
        roi = _roi_id(item)
        if roi in local:
            continue
        score, months = _score_roi(item, prior, args.lat_bin_size, global_rate)
        rows.append(
            {
                "roi_id": roi,
                "score": f"{score:.8f}",
                "latitude": f"{_lat(item):.6f}",
                "lat_bin": _lat_bin(_lat(item), args.lat_bin_size),
                "s2_months": months,
                "metadata_key": key,
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    if args.top_k > 0:
        rows = rows[: args.top_k]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["roi_id", "score"])
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_roi_list).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_roi_list).write_text("\n".join(row["roi_id"] for row in rows) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "metadata_rois": len({_roi_id(item) for item in metadata.values()}),
                "local_rois": len(local),
                "remaining_ranked": len(rows),
                "target_buckets": sorted(targets),
                "global_target_rate": global_rate,
                "out_csv": str(out_csv),
                "out_roi_list": args.out_roi_list,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
