#!/usr/bin/env python3
"""Audit ALLClear manifest/cache storage and optionally delete redundant dirs.

Default mode is dry-run.  The script keeps anything referenced by the provided
configs and marks other manifest/cache directories as removable candidates.
It never marks the raw ``data`` directory for deletion unless explicitly asked
and every active config's manifests are fully covered by its cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.allclear.config import load_config


DEFAULT_CONFIGS = [
    PROJECT_ROOT / "configs/allclear_tgdad_softshadow_stage1.yaml",
    PROJECT_ROOT / "configs/allclear_dadigan_baseline.yaml",
    PROJECT_ROOT / "configs/allclear_tgdad_softshadow_stage1_lama.yaml",
]


def dir_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024.0 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def manifest_sample_ids(path: Path) -> list[str]:
    import csv

    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sample_id = row.get("sample_id") or row.get("id") or f"sample_{idx:06d}"
            ids.append(sample_id)
    return ids


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{sample_id.replace('/', '_')}.pt"


def collect_active_paths(config_paths: list[Path]) -> tuple[Path, set[Path], dict[str, Any]]:
    active: set[Path] = set()
    root: Path | None = None
    details: dict[str, Any] = {"configs": []}
    for cfg_path in config_paths:
        cfg = load_config(cfg_path)
        data = cfg.get("data", {})
        cfg_root = Path(data["root"]).expanduser().resolve()
        if root is None:
            root = cfg_root
        elif cfg_root != root:
            raise SystemExit(f"Configs point to different ALLClear roots: {root} vs {cfg_root}")
        cfg_detail = {"config": str(cfg_path), "root": str(cfg_root), "active": []}
        manifest_paths: list[Path] = []
        for key in ("train_manifest", "val_manifest", "test_manifest"):
            path = resolve_path(cfg_root, data.get(key))
            if path is not None:
                active.add(path.resolve())
                active.add(path.resolve().parent)
                cfg_detail["active"].append(str(path.resolve()))
                manifest_paths.append(path.resolve())
        cache_dir = resolve_path(cfg_root, data.get("cache_dir"))
        if cache_dir is not None:
            active.add(cache_dir.resolve())
            cfg_detail["active"].append(str(cache_dir.resolve()))
        cfg_detail["manifest_paths"] = [str(path) for path in manifest_paths]
        cfg_detail["cache_dir"] = str(cache_dir.resolve()) if cache_dir is not None else None
        details["configs"].append(cfg_detail)
    if root is None:
        raise SystemExit("No configs provided")
    active.add((root / "data").resolve())
    return root, active, details


def audit_cache_coverage(details: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in details.get("configs", []):
        cache_dir_text = cfg.get("cache_dir")
        cache_dir = Path(cache_dir_text) if cache_dir_text else None
        manifest_paths = [Path(path) for path in cfg.get("manifest_paths", [])]
        total = 0
        missing = 0
        missing_examples: list[str] = []
        for manifest in manifest_paths:
            ids = manifest_sample_ids(manifest)
            total += len(ids)
            if cache_dir is None:
                missing += len(ids)
                missing_examples.extend(ids[:5])
                continue
            for sample_id in ids:
                if not cache_path(cache_dir, sample_id).exists():
                    missing += 1
                    if len(missing_examples) < 12:
                        missing_examples.append(sample_id)
        rows.append(
            {
                "config": cfg.get("config"),
                "cache_dir": cache_dir_text,
                "manifest_count": len(manifest_paths),
                "sample_rows": total,
                "missing_cache_files": missing,
                "cache_complete": missing == 0 and total > 0 and cache_dir is not None,
                "missing_examples": missing_examples,
            }
        )
    return rows


def classify(root: Path, active: set[Path], *, allow_raw_data_candidate: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "data":
            if allow_raw_data_candidate:
                status = "candidate_redundant_raw_data"
                candidates.append(child)
            else:
                status = "keep_raw_data"
        elif child.name == "cache":
            status = "container"
        elif child.resolve() in active:
            status = "keep_active"
        elif child.name.startswith("manifests"):
            status = "candidate_redundant_manifest"
            candidates.append(child)
        else:
            status = "keep_unclassified"
        rows.append({"path": str(child), "kind": "top_level", "status": status, "size_bytes": dir_size_bytes(child)})

    cache_root = root / "cache"
    if cache_root.exists():
        for child in sorted(cache_root.iterdir()):
            if not child.is_dir():
                continue
            if child.resolve() in active:
                status = "keep_active_cache"
            else:
                status = "candidate_redundant_cache"
                candidates.append(child)
            rows.append({"path": str(child), "kind": "cache_child", "status": status, "size_bytes": dir_size_bytes(child)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs/dataset_storage_audit/allclear_storage_audit.json")
    parser.add_argument("--delete", action="store_true", help="Delete candidate_redundant_manifest/cache directories.")
    parser.add_argument(
        "--delete-raw-data-if-cache-complete",
        action="store_true",
        help="Also mark raw data/ as removable, but only if every active config has complete cache coverage.",
    )
    parser.add_argument("--yes", action="store_true", help="Required together with --delete.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = [path.expanduser().resolve() for path in args.configs]
    root, active, details = collect_active_paths(config_paths)
    cache_coverage = audit_cache_coverage(details)
    raw_delete_allowed = bool(args.delete_raw_data_if_cache_complete) and all(row["cache_complete"] for row in cache_coverage)
    rows = classify(root, active, allow_raw_data_candidate=raw_delete_allowed)
    candidates = [Path(row["path"]) for row in rows if str(row["status"]).startswith("candidate_redundant")]
    report = {
        "root": str(root),
        "active_paths": sorted(str(path) for path in active),
        "delete_requested": bool(args.delete),
        "dry_run": not (args.delete and args.yes),
        "raw_data_delete_requested": bool(args.delete_raw_data_if_cache_complete),
        "raw_data_delete_allowed": raw_delete_allowed,
        "cache_coverage": cache_coverage,
        "total_candidate_size_bytes": sum(int(row["size_bytes"]) for row in rows if str(row["status"]).startswith("candidate_redundant")),
        "rows": rows,
        **details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"root: {root}")
    print(f"active configs: {', '.join(str(path) for path in config_paths)}")
    print("cache coverage:")
    for row in cache_coverage:
        print(
            f"  complete={str(row['cache_complete']):5s} missing={row['missing_cache_files']:5d} "
            f"rows={row['sample_rows']:5d} config={row['config']}"
        )
    if args.delete_raw_data_if_cache_complete and not raw_delete_allowed:
        print("raw data deletion was requested but blocked because at least one active config cache is incomplete.")
    print(f"candidate redundant size: {human_size(report['total_candidate_size_bytes'])}")
    for row in rows:
        print(f"{row['status']:30s} {human_size(int(row['size_bytes'])):>8s}  {row['path']}")
    print(f"json: {args.output_json}")

    if args.delete and not args.yes:
        raise SystemExit("Refusing to delete without --yes. Re-run with --delete --yes after reviewing the JSON.")
    if args.delete and args.yes:
        for path in candidates:
            shutil.rmtree(path)
            print(f"deleted: {path}")


if __name__ == "__main__":
    main()
