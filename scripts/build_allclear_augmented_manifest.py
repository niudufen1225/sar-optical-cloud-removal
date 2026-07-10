#!/usr/bin/env python3
"""Build an augmented ALLClear manifest from final + unused qualified pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed")
FINAL_DIR = ROOT / "manifests_final_clean_low01_med60_high90"
QUALIFIED_DIR = Path("/home/students/sushaoqi/CR/main/outputs/dataset_curation/local_rois_keep_all_qualified")
OUT_DIR = ROOT / "manifests_final_clean_plus_unused1000heavy_all_nonheavy"
SPLITS = ("train", "val", "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_fieldnames(*row_groups: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for rows in row_groups:
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--qualified-dir", type=Path, default=QUALIFIED_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--heavy-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir = args.final_dir.expanduser().resolve()
    qualified_dir = args.qualified_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    final_rows = read_csv(final_dir / "pairs_all.csv")
    qualified_rows = read_csv(qualified_dir / "pairs_all.csv")
    final_ids = {row["sample_id"] for row in final_rows}

    unused = [row for row in qualified_rows if row["sample_id"] not in final_ids]
    unused_heavy = [row for row in unused if row.get("bucket") == "heavy"]
    unused_nonheavy = [row for row in unused if row.get("bucket") != "heavy"]
    if len(unused_heavy) < args.heavy_count:
        raise SystemExit(f"Requested {args.heavy_count} heavy rows, but only {len(unused_heavy)} are available.")

    rng = random.Random(args.seed)
    selected_heavy = sorted(rng.sample(unused_heavy, args.heavy_count), key=lambda r: r["sample_id"])
    selected_extra = sorted(unused_nonheavy + selected_heavy, key=lambda r: (r["split"], r["bucket"], r["roi_id"], r["sample_id"]))
    merged = sorted(final_rows + selected_extra, key=lambda r: (r["split"], r["bucket"], r["roi_id"], r["sample_id"]))

    fieldnames = ordered_fieldnames(final_rows, selected_extra)
    by_split = {split: [row for row in merged if row["split"] == split] for split in SPLITS}

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pairs_all.csv", merged, fieldnames)
    for split in SPLITS:
        write_csv(out_dir / f"pairs_{split}.csv", by_split[split], fieldnames)
    (out_dir / "selected_rois.txt").write_text(
        "\n".join(sorted({row["roi_id"] for row in merged})) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_final_dir": str(final_dir),
        "source_qualified_dir": str(qualified_dir),
        "seed": args.seed,
        "requested_unused_heavy": args.heavy_count,
        "final_rows": len(final_rows),
        "unused_qualified_rows": len(unused),
        "unused_available_by_bucket": dict(Counter(row["bucket"] for row in unused)),
        "added_rows": len(selected_extra),
        "added_by_bucket": dict(Counter(row["bucket"] for row in selected_extra)),
        "added_by_split": dict(Counter(row["split"] for row in selected_extra)),
        "total_rows": len(merged),
        "total_by_bucket": dict(Counter(row["bucket"] for row in merged)),
        "total_by_split": {split: len(rows) for split, rows in by_split.items()},
        "total_split_bucket_counts": {
            split: dict(Counter(row["bucket"] for row in rows))
            for split, rows in by_split.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
