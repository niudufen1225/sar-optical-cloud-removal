#!/usr/bin/env python3
"""Reduce the extra heavy rows in the augmented ALLClear manifest and prune tif files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed")
FINAL_DIR = ROOT / "manifests_final_clean_low01_med60_high90"
AUGMENTED_DIR = ROOT / "manifests_final_clean_plus_unused1000heavy_all_nonheavy"
OUT_DIR = ROOT / "manifests_final_clean_plus_unused300heavy_all_nonheavy"
REPORT_DIR = Path("/home/students/sushaoqi/CR/main/outputs/dataset_curation/reduce_unused_heavy_keep300")
SPLITS = ("train", "val", "test")
CORE_COLUMNS = ("cloudy_s2_path", "clear_s2_path", "sar_s1_path")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def counter_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key, "") for row in rows))


def split_bucket_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    return {split: dict(Counter(row.get("bucket", "") for row in rows if row.get("split") == split)) for split in SPLITS}


def allocate_stratified(counts: dict[str, int], keep_total: int) -> dict[str, int]:
    total = sum(counts.values())
    if keep_total > total:
        raise ValueError(f"Cannot keep {keep_total} rows from only {total} candidates.")
    exact = {split: counts[split] * keep_total / total for split in counts}
    allocated = {split: int(exact[split]) for split in counts}
    remaining = keep_total - sum(allocated.values())
    order = sorted(counts, key=lambda split: (exact[split] - allocated[split], counts[split], split), reverse=True)
    for split in order[:remaining]:
        allocated[split] += 1
    return allocated


def select_added_heavy(rows: list[dict[str, str]], keep_total: int, seed: int) -> list[dict[str, str]]:
    by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    for row in rows:
        by_split.setdefault(row.get("split", ""), []).append(row)
    counts = {split: len(by_split.get(split, [])) for split in SPLITS if by_split.get(split)}
    allocated = allocate_stratified(counts, keep_total)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for split, split_rows in by_split.items():
        keep = allocated.get(split, 0)
        if keep <= 0:
            continue
        candidates = sorted(split_rows, key=lambda row: row["sample_id"])
        selected.extend(rng.sample(candidates, keep))
    return sorted(selected, key=lambda row: (row.get("split", ""), row.get("roi_id", ""), row["sample_id"]))


def derive_cld_shdw_from_s2(path: Path) -> Path | None:
    parts = list(path.parts)
    try:
        idx = parts.index("s2_toa")
    except ValueError:
        return None
    parts[idx] = "cld_shdw"
    derived = Path(*parts)
    return derived.with_name(derived.name.replace("_s2_toa_", "_cld_shdw_")).with_suffix(".tif")


def row_required_tifs(row: dict[str, str]) -> dict[str, Path]:
    out = {name: Path(row[name]).expanduser().resolve() for name in CORE_COLUMNS if row.get(name)}
    if "cloudy_s2_path" in out:
        mask = derive_cld_shdw_from_s2(out["cloudy_s2_path"])
        if mask is not None:
            out["cloudy_cld_shdw_path"] = mask.expanduser().resolve()
    return out


def required_tifs(rows: list[dict[str, str]]) -> tuple[set[Path], dict[str, set[Path]], list[tuple[str, str, str]]]:
    by_kind: dict[str, set[Path]] = {key: set() for key in (*CORE_COLUMNS, "cloudy_cld_shdw_path")}
    missing: list[tuple[str, str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        split = row.get("split", "")
        for kind, path in row_required_tifs(row).items():
            by_kind.setdefault(kind, set()).add(path)
            if not path.exists():
                missing.append((split, sample_id, str(path)))
    required: set[Path] = set()
    for paths in by_kind.values():
        required.update(paths)
    return required, by_kind, missing


def all_data_tifs(root: Path) -> set[Path]:
    return {path.resolve() for path in (root / "data").glob("**/*.tif")}


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(value)
            f.write("\n")


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for directory in sorted((root / "data").glob("**/*"), key=lambda p: len(p.parts), reverse=True):
        if not directory.is_dir():
            continue
        try:
            directory.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--augmented-dir", type=Path, default=AUGMENTED_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--keep-added-heavy", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--delete-extra-tif", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--remove-empty-dirs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    final_dir = args.final_dir.expanduser().resolve()
    augmented_dir = args.augmented_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()

    final_rows = read_csv(final_dir / "pairs_all.csv")
    augmented_rows = read_csv(augmented_dir / "pairs_all.csv")
    final_ids = {row["sample_id"] for row in final_rows}
    augmented_ids = {row["sample_id"] for row in augmented_rows}
    if len(augmented_ids) != len(augmented_rows):
        raise SystemExit("Augmented manifest has duplicated sample_id values.")

    added_rows = [row for row in augmented_rows if row["sample_id"] not in final_ids]
    added_heavy = [row for row in added_rows if row.get("bucket") == "heavy"]
    added_nonheavy = [row for row in added_rows if row.get("bucket") != "heavy"]
    keep_heavy = select_added_heavy(added_heavy, args.keep_added_heavy, args.seed)
    keep_heavy_ids = {row["sample_id"] for row in keep_heavy}
    removed_heavy = [row for row in added_heavy if row["sample_id"] not in keep_heavy_ids]
    removed_ids = {row["sample_id"] for row in removed_heavy}
    new_rows = [row for row in augmented_rows if row["sample_id"] not in removed_ids]
    new_rows = sorted(new_rows, key=lambda row: (row.get("split", ""), row.get("bucket", ""), row.get("roi_id", ""), row["sample_id"]))

    fieldnames = ordered_fieldnames(augmented_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pairs_all.csv", new_rows, fieldnames)
    for split in SPLITS:
        write_csv(out_dir / f"pairs_{split}.csv", [row for row in new_rows if row.get("split") == split], fieldnames)
    write_list(out_dir / "selected_rois.txt", sorted({row["roi_id"] for row in new_rows if row.get("roi_id")}))

    required, by_kind, missing = required_tifs(new_rows)
    all_tifs = all_data_tifs(root)
    extra = sorted(all_tifs - required)

    report_dir.mkdir(parents=True, exist_ok=True)
    write_list(report_dir / "kept_added_heavy_ids.txt", sorted(keep_heavy_ids))
    write_list(report_dir / "removed_added_heavy_ids.txt", sorted(removed_ids))
    write_list(report_dir / "required_tifs.txt", sorted(str(path) for path in required))
    write_list(report_dir / "extra_tifs.txt", [str(path) for path in extra])
    write_list(report_dir / "missing_required_tifs.txt", [",".join(item) for item in missing])

    summary = {
        "root": str(root),
        "source_final_dir": str(final_dir),
        "source_augmented_dir": str(augmented_dir),
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "seed": args.seed,
        "final_rows": len(final_rows),
        "augmented_rows_before": len(augmented_rows),
        "added_rows_before": len(added_rows),
        "added_nonheavy_kept": len(added_nonheavy),
        "added_heavy_before": len(added_heavy),
        "added_heavy_kept": len(keep_heavy),
        "added_heavy_removed": len(removed_heavy),
        "rows_after": len(new_rows),
        "before_by_bucket": counter_by(augmented_rows, "bucket"),
        "after_by_bucket": counter_by(new_rows, "bucket"),
        "before_split_bucket_counts": split_bucket_counts(augmented_rows),
        "after_split_bucket_counts": split_bucket_counts(new_rows),
        "kept_added_heavy_by_split": counter_by(keep_heavy, "split"),
        "removed_added_heavy_by_split": counter_by(removed_heavy, "split"),
        "required_tifs": len(required),
        "required_by_kind": {kind: len(paths) for kind, paths in by_kind.items()},
        "all_data_tifs": len(all_tifs),
        "extra_tifs": len(extra),
        "extra_by_parent": dict(Counter(path.parent.name for path in extra)),
        "missing_required_tifs": len(missing),
        "reports": {
            "kept_added_heavy_ids": str(report_dir / "kept_added_heavy_ids.txt"),
            "removed_added_heavy_ids": str(report_dir / "removed_added_heavy_ids.txt"),
            "extra_tifs": str(report_dir / "extra_tifs.txt"),
            "missing_required_tifs": str(report_dir / "missing_required_tifs.txt"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.delete_extra_tif:
        if not args.yes:
            raise SystemExit("Refusing to delete tif files without --yes.")
        for path in extra:
            path.unlink()
        print(f"Deleted {len(extra)} extra tif files.")
        if args.remove_empty_dirs:
            removed_dirs = remove_empty_dirs(root)
            print(f"Removed {removed_dirs} empty data directories.")


if __name__ == "__main__":
    main()
