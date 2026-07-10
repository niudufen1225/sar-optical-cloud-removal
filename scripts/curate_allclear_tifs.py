#!/usr/bin/env python3
"""Curate ALLClear tif files for the current Stage1 training manifests.

The script is conservative by default: it only writes reports. Destructive
deletion requires both --delete-extra-tif and --yes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed")
MANIFEST_DIR = ROOT / "manifests_final_clean_low01_med60_high90"
REPORT_DIR = Path("/home/students/sushaoqi/CR/main/outputs/dataset_curation")

SPLITS = ("train", "val", "test")
CORE_COLUMNS = ("cloudy_s2_path", "clear_s2_path", "sar_s1_path")


def derive_cld_shdw_from_s2(path: Path) -> Path | None:
    parts = list(path.parts)
    try:
        idx = parts.index("s2_toa")
    except ValueError:
        return None
    parts[idx] = "cld_shdw"
    derived = Path(*parts)
    return derived.with_name(derived.name.replace("_s2_toa_", "_cld_shdw_")).with_suffix(".tif")


def read_rows(manifest_dir: Path, splits: tuple[str, ...] = SPLITS) -> dict[str, list[dict[str, str]]]:
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in splits:
        path = manifest_dir / f"pairs_{split}.csv"
        if not path.exists():
            rows_by_split[split] = []
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            rows_by_split[split] = list(csv.DictReader(f))
    return rows_by_split


def row_required_tifs(row: dict[str, str]) -> dict[str, Path]:
    out = {name: Path(row[name]).expanduser().resolve() for name in CORE_COLUMNS if row.get(name)}
    mask = derive_cld_shdw_from_s2(out["cloudy_s2_path"]) if "cloudy_s2_path" in out else None
    if mask is not None:
        out["cloudy_cld_shdw_path"] = mask.expanduser().resolve()
    return out


def required_tifs(rows_by_split: dict[str, list[dict[str, str]]]) -> tuple[set[Path], dict[str, set[Path]], list[tuple[str, str, str]]]:
    by_kind: dict[str, set[Path]] = {
        "cloudy_s2_path": set(),
        "clear_s2_path": set(),
        "sar_s1_path": set(),
        "cloudy_cld_shdw_path": set(),
    }
    missing: list[tuple[str, str, str]] = []
    for split, rows in rows_by_split.items():
        for row in rows:
            sample_id = row.get("sample_id", "")
            for kind, path in row_required_tifs(row).items():
                by_kind.setdefault(kind, set()).add(path)
                if not path.exists():
                    missing.append((split, sample_id, str(path)))
    all_required: set[Path] = set()
    for paths in by_kind.values():
        all_required.update(paths)
    return all_required, by_kind, missing


def all_data_tifs(root: Path) -> set[Path]:
    return {p.resolve() for p in (root / "data").glob("**/*.tif")}


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(value)
            f.write("\n")


def write_manifest_dir(path: Path, rows_by_split: dict[str, list[dict[str, str]]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for split in SPLITS:
        for row in rows_by_split.get(split, []):
            all_rows.append(row)
            for key in row:
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)
    for split in SPLITS:
        split_rows = rows_by_split.get(split, [])
        with (path / f"pairs_{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
    with (path / "pairs_all.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "rows": {split: len(rows_by_split.get(split, [])) for split in SPLITS},
        "total": len(all_rows),
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_sources(
    base_rows: dict[str, list[dict[str, str]]],
    sources: list[Path],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    merged = {split: list(rows) for split, rows in base_rows.items()}
    seen = {row.get("sample_id") for rows in merged.values() for row in rows if row.get("sample_id")}
    accepted: list[dict[str, str]] = []
    for source in sources:
        source_rows = read_rows(source)
        for split in SPLITS:
            for row in source_rows.get(split, []):
                sample_id = row.get("sample_id", "")
                if not sample_id or sample_id in seen:
                    continue
                paths = row_required_tifs(row)
                if not paths or any(not path.exists() for path in paths.values()):
                    continue
                merged.setdefault(split, []).append(row)
                seen.add(sample_id)
                accepted.append({"source": str(source), "split": split, "sample_id": sample_id})
    return merged, accepted


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for directory in sorted((root / "data").glob("**/*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                continue
            removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--merge-source",
        type=Path,
        action="append",
        default=[],
        help="Additional manifest directory to merge by sample_id if all required tif files exist.",
    )
    parser.add_argument(
        "--merge-output",
        type=Path,
        default=None,
        help="Write merged pairs_{train,val,test,all}.csv here. Existing files may be overwritten.",
    )
    parser.add_argument("--delete-extra-tif", action="store_true", help="Delete data/**/*.tif files not required by the final or merged manifest.")
    parser.add_argument("--yes", action="store_true", help="Required together with --delete-extra-tif to actually unlink files.")
    parser.add_argument("--remove-empty-dirs", action="store_true", help="Remove empty data directories after deletion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    rows_by_split = read_rows(manifest_dir)

    merge_accepted: list[dict[str, str]] = []
    if args.merge_source:
        rows_by_split, merge_accepted = merge_sources(rows_by_split, [p.expanduser().resolve() for p in args.merge_source])
        if args.merge_output is not None:
            write_manifest_dir(args.merge_output.expanduser().resolve(), rows_by_split)

    required, by_kind, missing = required_tifs(rows_by_split)
    all_tifs = all_data_tifs(root)
    extra = sorted(all_tifs - required)

    report_dir.mkdir(parents=True, exist_ok=True)
    write_list(report_dir / "required_tifs.txt", sorted(str(p) for p in required))
    write_list(report_dir / "extra_tifs.txt", [str(p) for p in extra])
    write_list(report_dir / "missing_required_tifs.txt", [",".join(item) for item in missing])
    if merge_accepted:
        with (report_dir / "merged_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "split", "sample_id"])
            writer.writeheader()
            writer.writerows(merge_accepted)

    summary = {
        "root": str(root),
        "manifest_dir": str(manifest_dir),
        "rows": {split: len(rows_by_split.get(split, [])) for split in SPLITS},
        "required_tifs": len(required),
        "required_by_kind": {kind: len(paths) for kind, paths in by_kind.items()},
        "all_data_tifs": len(all_tifs),
        "extra_tifs": len(extra),
        "extra_by_parent": Counter(p.parent.name for p in extra),
        "missing_required_tifs": len(missing),
        "merge_accepted_rows": len(merge_accepted),
        "reports": {
            "required": str(report_dir / "required_tifs.txt"),
            "extra": str(report_dir / "extra_tifs.txt"),
            "missing": str(report_dir / "missing_required_tifs.txt"),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=dict))

    if args.delete_extra_tif:
        if not args.yes:
            raise SystemExit("Refusing to delete without --yes. Review extra_tifs.txt first.")
        for path in extra:
            path.unlink()
        print(f"Deleted {len(extra)} extra tif files.")
        if args.remove_empty_dirs:
            removed = remove_empty_dirs(root)
            print(f"Removed {removed} empty directories.")


if __name__ == "__main__":
    main()
