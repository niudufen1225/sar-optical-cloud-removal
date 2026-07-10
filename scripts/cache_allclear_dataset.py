#!/usr/bin/env python3
"""Materialize ALLClear Dataset outputs as per-sample .pt cache files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.config import load_config
from src.allclear.dataset import AllClearDataset


def build_dataset(cfg: dict[str, Any], split: str, *, include_sar: bool) -> AllClearDataset:
    data = cfg["data"]
    return AllClearDataset(
        root=data["root"],
        manifest=data[f"{split}_manifest"],
        optical_scale=float(data.get("optical_scale", 10000.0)),
        image_size=data.get("image_size"),
        shadow_index=int(data.get("shadow_index", 3)),
        cloud_index=int(data.get("cloud_index", 1)),
        prefer_original_cld_shdw=bool(data.get("prefer_original_cld_shdw", True)),
        load_sar=include_sar,
        cache_dir=None,
        band_indices=data.get("band_indices"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training YAML whose manifests/preprocessing should be cached.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Output directory for per-sample .pt files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument(
        "--include-sar",
        action="store_true",
        help="Cache SAR even if the selected config has data.load_sar=false. Use this once to share cache with DADIGAN and LaMa configs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing cache files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    include_sar = bool(args.include_sar or cfg.get("data", {}).get("load_sar", True))

    summary: dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "cache_dir": str(cache_dir),
        "splits": list(args.splits),
        "include_sar": include_sar,
        "written": 0,
        "skipped": 0,
        "rows": {},
    }

    for split in args.splits:
        dataset = build_dataset(cfg, split, include_sar=include_sar)
        summary["rows"][split] = len(dataset)
        pbar = tqdm(range(len(dataset)), desc=f"cache {split}", unit="sample")
        for idx in pbar:
            row = dataset.rows[idx]
            sample_id = row.get("sample_id") or row.get("id") or f"sample_{idx:06d}"
            cache_path = cache_dir / f"{sample_id.replace('/', '_')}.pt"
            if cache_path.exists() and not args.overwrite:
                summary["skipped"] += 1
                continue
            item = dataset[idx]
            payload = {
                "sample_id": item["sample_id"],
                "s2_toa": item["s2_toa"].contiguous(),
                "target": item["target"].contiguous(),
                "cld_shdw": item["cld_shdw"].contiguous(),
            }
            if include_sar and "s1" in item:
                payload["s1"] = item["s1"].contiguous()
            tmp_path = cache_path.with_suffix(".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(cache_path)
            summary["written"] += 1

    (cache_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
