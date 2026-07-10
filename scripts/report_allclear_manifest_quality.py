#!/usr/bin/env python3
"""Generate a Markdown quality report for filtered ALLClear manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/students/sushaoqi/CR/allclear_10pct_tx3_s2_s1_bucketed")
DEFAULT_MANIFEST_DIRS = [
    ROOT / "manifests_official_mask_lama_cloud",
    ROOT / "manifests_official_mask_dadigan_damage",
]
DEFAULT_OUT_DIR = Path("/home/students/sushaoqi/CR/main/outputs/dataset_quality/official_mask_report")
SPLITS = ("train", "val", "test")
BUCKETS = ("low", "medium", "high", "heavy")
METRIC_COLUMNS = (
    "cloudy_cloud_prob_mean",
    "cloudy_cloud_bin_frac",
    "cloudy_cloud_core_frac",
    "cloudy_shadow_noncloud_frac",
    "cloudy_damage_frac",
    "target_cloud_bin_frac",
    "target_damage_frac",
    "cloudy_target_cloud_l1",
    "cloudy_target_cloud_to_clear_l1_ratio",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def numeric_values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = [safe_float(row.get(column)) for row in rows]
    return np.asarray([v for v in values if math.isfinite(v)], dtype=np.float32)


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.4f}"


def metric_summary(rows: list[dict[str, str]], columns: tuple[str, ...] = METRIC_COLUMNS) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for column in columns:
        values = numeric_values(rows, column)
        if values.size == 0:
            continue
        out.append(
            {
                "metric": column,
                "count": str(int(values.size)),
                "mean": fmt_float(float(np.mean(values))),
                "std": fmt_float(float(np.std(values))),
                "p05": fmt_float(float(np.quantile(values, 0.05))),
                "p50": fmt_float(float(np.quantile(values, 0.50))),
                "p95": fmt_float(float(np.quantile(values, 0.95))),
            }
        )
    return out


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_bar(counter: Counter[str], title: str, path: Path, ordered: tuple[str, ...] | None = None) -> None:
    if ordered is None:
        labels = list(counter.keys())
    else:
        labels = [name for name in ordered if counter.get(name, 0) > 0]
        labels += [name for name in counter if name not in labels]
    values = [counter.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 1)))
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_split_bucket(rows: list[dict[str, str]], title: str, path: Path) -> None:
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row in rows:
        split = row.get("split", "")
        bucket = row.get("bucket", "")
        if split in counts:
            counts[split][bucket] += 1
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bottoms = np.zeros(len(SPLITS), dtype=np.float32)
    colors = {
        "low": "#66c2a5",
        "medium": "#fc8d62",
        "high": "#8da0cb",
        "heavy": "#e78ac3",
    }
    x = np.arange(len(SPLITS))
    for bucket in BUCKETS:
        values = np.asarray([counts[split].get(bucket, 0) for split in SPLITS], dtype=np.float32)
        ax.bar(x, values, bottom=bottoms, label=bucket, color=colors.get(bucket))
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.set_title(title)
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_histograms(rows: list[dict[str, str]], title: str, path: Path) -> None:
    cols = [
        ("cloudy_cloud_bin_frac", "cloud bin"),
        ("cloudy_damage_frac", "damage"),
        ("cloudy_shadow_noncloud_frac", "shadow non-cloud"),
        ("target_cloud_bin_frac", "target cloud"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)
    fig.suptitle(title)
    for ax, (column, label) in zip(axes.flat, cols):
        values = numeric_values(rows, column)
        if values.size:
            ax.hist(values, bins=30, range=(0, 1), color="#4c78a8", alpha=0.85)
            ax.axvline(float(np.median(values)), color="#f58518", linewidth=1.5, label="median")
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_delta_scatter(rows: list[dict[str, str]], title: str, path: Path) -> None:
    cloud = numeric_values(rows, "cloudy_cloud_bin_frac")
    delta_values: list[float] = []
    cloud_values: list[float] = []
    colors: list[str] = []
    for row in rows:
        x = safe_float(row.get("cloudy_cloud_bin_frac"))
        y = safe_float(row.get("cloudy_target_cloud_l1"))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        cloud_values.append(x)
        delta_values.append(y)
        colors.append("#d62728" if row.get("mask_reliability_flag", "").startswith("suspect") else "#4c78a8")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    if cloud.size and delta_values:
        ax.scatter(cloud_values, delta_values, s=14, c=colors, alpha=0.65, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("Official cloud mask fraction")
    ax.set_ylabel("Cloudy-target L1 inside cloud mask")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {arr.shape}")
    if arr.shape[0] <= 32:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def read_tif(path: Path) -> np.ndarray:
    import tifffile  # type: ignore

    return to_chw(np.asarray(tifffile.imread(path), dtype=np.float32))


def normalize_optical(image: np.ndarray, optical_scale: float) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=optical_scale, neginf=0.0)
    return np.clip(image, 0.0, optical_scale) / optical_scale


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = mask.copy()
    for channel in range(out.shape[0]):
        max_value = float(np.max(out[channel])) if out[channel].size else 0.0
        if max_value > 1.0:
            out[channel] = out[channel] / (100.0 if max_value <= 100.0 else 255.0)
    return np.clip(out, 0.0, 1.0)


def render_rgb(image: np.ndarray, rgb_indices: tuple[int, int, int], optical_scale: float) -> np.ndarray:
    image = normalize_optical(image, optical_scale)
    rgb = image[list(rgb_indices)]
    rgb = np.transpose(rgb, (1, 2, 0))
    lo = np.quantile(rgb, 0.01)
    hi = np.quantile(rgb, 0.995)
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def render_mask(mask: np.ndarray, cloud_channel: int, shadow_channel: int) -> tuple[np.ndarray, np.ndarray]:
    mask = normalize_mask(mask)
    cloud = mask[cloud_channel]
    shadow = mask[shadow_channel] * (1.0 - cloud)
    damage = np.maximum(cloud, shadow)
    cloud_rgb = np.zeros((*cloud.shape, 3), dtype=np.float32)
    cloud_rgb[..., 2] = cloud
    cloud_rgb[..., 1] = 0.75 * cloud
    damage_rgb = np.zeros((*damage.shape, 3), dtype=np.float32)
    damage_rgb[..., 0] = shadow
    damage_rgb[..., 1] = 0.55 * cloud
    damage_rgb[..., 2] = cloud
    return cloud_rgb, damage_rgb


def sample_balanced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bucket[row.get("bucket") or row.get("original_bucket") or "unknown"].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        added = False
        for bucket in (*BUCKETS, "unknown"):
            group = by_bucket.get(bucket, [])
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


def make_sample_grid(
    rows: list[dict[str, str]],
    title: str,
    path: Path,
    *,
    rgb_indices: tuple[int, int, int],
    cloud_channel: int,
    shadow_channel: int,
    optical_scale: float,
) -> bool:
    samples = rows
    if not samples:
        return False
    cols = ["cloudy", "target", "cloud mask", "damage mask", "abs diff"]
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(2.6 * len(cols), 2.4 * len(samples)), dpi=150)
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)
    fig.suptitle(title, fontsize=11)
    for row_idx, row in enumerate(samples):
        try:
            cloudy = read_tif(Path(row["cloudy_s2_path"]))
            target = read_tif(Path(row["clear_s2_path"]))
            mask_path = row.get("official_cloudy_mask_path") or row.get("cloudy_mask_path")
            mask = read_tif(Path(mask_path))
            cloudy_rgb = render_rgb(cloudy, rgb_indices, optical_scale)
            target_rgb = render_rgb(target, rgb_indices, optical_scale)
            cloud_rgb, damage_rgb = render_mask(mask, cloud_channel, shadow_channel)
            diff = np.mean(np.abs(normalize_optical(cloudy, optical_scale) - normalize_optical(target, optical_scale)), axis=0)
            diff_rgb = plt.cm.magma(np.clip(diff / max(float(np.quantile(diff, 0.995)), 1.0e-6), 0, 1))[..., :3]
            panels = [cloudy_rgb, target_rgb, cloud_rgb, damage_rgb, diff_rgb]
        except Exception as exc:
            panels = [np.zeros((64, 64, 3), dtype=np.float32) for _ in cols]
            panels[0][..., 0] = 0.3
            panels[0][..., 1] = 0.05
            panels[0][..., 2] = 0.05
            row["visualization_error"] = f"{type(exc).__name__}: {exc}"
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=8)
        label = row.get("sample_id", "")[:52]
        bucket = row.get("bucket") or row.get("original_bucket", "")
        cloud_frac = row.get("cloudy_cloud_bin_frac", "")
        axes[row_idx, 0].set_ylabel(
            textwrap.fill(f"{bucket} cloud={cloud_frac} {label}", width=30),
            fontsize=6,
            rotation=0,
            ha="right",
            va="center",
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def make_sample_pages(
    rows: list[dict[str, str]],
    label: str,
    group_dir: Path,
    *,
    visualize_all: bool,
    samples_per_group: int,
    samples_per_page: int,
    rgb_indices: tuple[int, int, int],
    cloud_channel: int,
    shadow_channel: int,
    optical_scale: float,
) -> list[Path]:
    """Write paged visualization grids for accepted/suspect/rejected rows."""

    if not rows:
        return []
    ensure_dir(group_dir)
    selected = rows if visualize_all else sample_balanced(rows, samples_per_group)
    if not selected:
        return []
    per_page = max(1, int(samples_per_page if visualize_all else samples_per_group))
    total_pages = int(math.ceil(len(selected) / per_page))
    pages: list[Path] = []
    for page_idx in range(total_pages):
        start = page_idx * per_page
        end = min(start + per_page, len(selected))
        page_rows = selected[start:end]
        page_path = group_dir / f"page_{page_idx + 1:04d}.png"
        title = f"{label}: samples {start + 1}-{end} / {len(selected)}"
        if make_sample_grid(
            page_rows,
            title,
            page_path,
            rgb_indices=rgb_indices,
            cloud_channel=cloud_channel,
            shadow_channel=shadow_channel,
            optical_scale=optical_scale,
        ):
            pages.append(page_path)
    return pages


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_无数据_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def write_manifest_report(manifest_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    name = manifest_dir.name
    manifest_out = out_dir / name
    figures_dir = manifest_out / "figures"
    ensure_dir(figures_dir)

    accepted = read_csv(manifest_dir / "pairs_all.csv")
    rejected = read_csv(manifest_dir / "pairs_rejected.csv")
    suspect = read_csv(manifest_dir / "pairs_suspect.csv")
    summary = read_summary(manifest_dir / "summary.json")

    bucket_counts = Counter(row.get("bucket", "") for row in accepted)
    split_counts = Counter(row.get("split", "") for row in accepted)
    reject_counts = Counter(row.get("official_reject_reason", "") for row in rejected)

    figures: list[Path] = []
    fig_path = figures_dir / "accepted_bucket_counts.png"
    plot_bar(bucket_counts, f"{name}: accepted buckets", fig_path, BUCKETS)
    figures.append(fig_path)
    fig_path = figures_dir / "accepted_split_bucket_counts.png"
    plot_split_bucket(accepted, f"{name}: split/bucket distribution", fig_path)
    figures.append(fig_path)
    fig_path = figures_dir / "metric_histograms.png"
    plot_histograms(accepted, f"{name}: accepted metric distributions", fig_path)
    figures.append(fig_path)
    fig_path = figures_dir / "paired_delta_scatter.png"
    plot_delta_scatter(accepted + suspect, f"{name}: paired-delta audit", fig_path)
    figures.append(fig_path)
    if rejected:
        fig_path = figures_dir / "rejected_reasons.png"
        plot_bar(reject_counts, f"{name}: rejected reasons", fig_path)
        figures.append(fig_path)

    sample_figures: dict[str, list[Path]] = {}
    for rows, label, dirname in [
        (accepted, "accepted samples", "accepted_samples"),
        (suspect, "suspect samples", "suspect_samples"),
        (rejected, "rejected samples", "rejected_samples"),
    ]:
        sample_figures[label] = make_sample_pages(
            rows,
            f"{name}: {label}",
            figures_dir / dirname,
            visualize_all=bool(args.visualize_all),
            samples_per_group=int(args.samples_per_group),
            samples_per_page=int(args.samples_per_page),
            rgb_indices=tuple(args.rgb_indices),
            cloud_channel=args.cloud_channel,
            shadow_channel=args.shadow_channel,
            optical_scale=args.optical_scale,
        )

    metrics = metric_summary(accepted)
    report_path = manifest_out / "quality_report.md"
    lines = [
        f"# {name} 数据集质量报告",
        "",
        "## 总览",
        "",
        f"- Manifest: `{manifest_dir}`",
        f"- 输入样本数: {summary.get('rows_input', len(accepted) + len(rejected))}",
        f"- 保留样本数: {len(accepted)}",
        f"- 拒绝样本数: {len(rejected)}",
        f"- 可疑样本数: {len(suspect)}",
        f"- 分桶模式: `{summary.get('bucket_mode', '')}`",
        f"- 旧分桶发生变化的保留样本数: {summary.get('accepted_bucket_changed_from_original', '')}",
        "",
        "## 保留样本分布",
        "",
        markdown_table(
            [{"bucket": bucket, "count": str(bucket_counts.get(bucket, 0))} for bucket in BUCKETS],
            ["bucket", "count"],
        ),
        markdown_table(
            [{"split": split, "count": str(split_counts.get(split, 0))} for split in SPLITS],
            ["split", "count"],
        ),
        "",
        "## 拒绝原因",
        "",
        markdown_table(
            [{"reason": key or "(empty)", "count": str(value)} for key, value in reject_counts.most_common()],
            ["reason", "count"],
        ),
        "",
        "## 关键指标统计",
        "",
        markdown_table(metrics, ["metric", "count", "mean", "std", "p05", "p50", "p95"]),
        "",
        "## 统计图",
        "",
    ]
    for figure in figures:
        lines.append(f"![{figure.stem}]({rel(figure, manifest_out)})")
        lines.append("")
    lines.append("## 样本可视化")
    lines.append("")
    if args.visualize_all:
        lines.append(
            f"全量可视化已启用；每页最多 {args.samples_per_page} 个样本，accepted/suspect/rejected 都会分页展示。"
        )
        lines.append("")
    for label, pages in sample_figures.items():
        if not pages:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for figure in pages:
            lines.append(f"![{figure.parent.name}/{figure.stem}]({rel(figure, manifest_out)})")
            lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    for filename in ("pairs_all.csv", "pairs_rejected.csv", "pairs_suspect.csv", "summary.json"):
        path = manifest_dir / filename
        if path.exists():
            lines.append(f"- `{path}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "name": name,
        "manifest_dir": str(manifest_dir),
        "report": str(report_path),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "suspect": len(suspect),
        "bucket_counts": dict(bucket_counts),
        "split_counts": dict(split_counts),
        "reject_counts": dict(reject_counts),
    }


def write_index(reports: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [
        {
            "manifest": report["name"],
            "accepted": str(report["accepted"]),
            "rejected": str(report["rejected"]),
            "suspect": str(report["suspect"]),
            "report": f"[quality_report.md]({rel(Path(report['report']), out_dir)})",
        }
        for report in reports
    ]
    lines = [
        "# ALLClear 官方 Mask 筛选质量总览",
        "",
        "该报告由 `scripts/report_allclear_manifest_quality.py` 生成，基于筛选脚本输出的 manifest 审计列和 TIFF 样本可视化。",
        "",
        markdown_table(rows, ["manifest", "accepted", "rejected", "suspect", "report"]),
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dirs", type=Path, nargs="+", default=DEFAULT_MANIFEST_DIRS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument(
        "--visualize-all",
        action="store_true",
        help="Visualize every accepted/suspect/rejected TIFF sample in paged grids instead of balanced samples only.",
    )
    parser.add_argument("--samples-per-page", type=int, default=20, help="Rows per visualization page when --visualize-all is set.")
    parser.add_argument("--rgb-indices", type=int, nargs=3, default=[3, 2, 1])
    parser.add_argument("--cloud-channel", type=int, default=1)
    parser.add_argument("--shadow-channel", type=int, default=3)
    parser.add_argument("--optical-scale", type=float, default=10000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    ensure_dir(out_dir)
    reports = []
    for manifest_dir in args.manifest_dirs:
        manifest_dir = manifest_dir.expanduser().resolve()
        if not (manifest_dir / "pairs_all.csv").exists():
            raise SystemExit(f"Missing pairs_all.csv in {manifest_dir}")
        reports.append(write_manifest_report(manifest_dir, out_dir, args))
    write_index(reports, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "reports": reports}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
