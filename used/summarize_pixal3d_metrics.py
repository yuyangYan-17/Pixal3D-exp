#!/usr/bin/env python3
"""Summarize Pixal3D multiresolution CSV results into normal and RGB tables.

The script reads the benchmark's ``all_views.csv`` and, optionally, its
``paired_resolution_comparison.csv``. It writes two CSV files:

  * normal_metrics_mean_summary.csv
  * rgb_metrics_mean_summary.csv

Each metric row contains the mean at every requested resolution, the
resolution-to-resolution delta, and direction-aware paired win/loss counts.
A final ``__overall__`` row gives a compact interpretation of the group.

Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


NORMAL_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "normal_iou_percent",
        "Normal silhouette IoU (%)",
        "higher",
        "GT 与 raw decoder mesh 的 nvdiffrast 几何轮廓重合率",
    ),
    (
        "normal_psnr_overlap_db",
        "Normal PSNR on overlap (dB)",
        "higher",
        "GT/预测法向量 RGB 在共同可见像素上的 PSNR",
    ),
    (
        "normal_ssim_overlap",
        "Normal SSIM on overlap",
        "higher",
        "GT/预测法向量 RGB 的局部结构相似度",
    ),
    (
        "normal_lpips_overlap",
        "Normal LPIPS on overlap",
        "lower",
        "LPIPS 网络衡量的法向量图感知距离",
    ),
    (
        "normal_mean_angular_error_deg",
        "Mean normal angular error (deg)",
        "lower",
        "共同可见像素上的平均法向量夹角",
    ),
    (
        "normal_median_angular_error_deg",
        "Median normal angular error (deg)",
        "lower",
        "共同可见像素上的法向量夹角中位数",
    ),
    (
        "normal_boundary_mean_angular_error_deg",
        "Boundary mean angular error (deg)",
        "lower",
        "GT 内轮廓边界带上的平均法向量夹角",
    ),
    (
        "normal_acc_11_25_percent",
        "Normal Acc@11.25deg (%)",
        "higher",
        "法向量误差小于 11.25° 的共同可见像素比例",
    ),
    (
        "normal_acc_22_5_percent",
        "Normal Acc@22.5deg (%)",
        "higher",
        "法向量误差小于 22.5° 的共同可见像素比例",
    ),
    (
        "normal_acc_30_percent",
        "Normal Acc@30deg (%)",
        "higher",
        "法向量误差小于 30° 的共同可见像素比例",
    ),
)

RGB_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "rgb_iou_percent",
        "RGB render silhouette IoU (%)",
        "higher",
        "源 GLB 与导出 Pixal3D GLB 的 Blender alpha mask IoU",
    ),
    (
        "rgb_psnr_overlap_db",
        "RGB PSNR on overlap (dB)",
        "higher",
        "两张 Blender beauty render 在 alpha mask 交集上的 PSNR",
    ),
    (
        "rgb_ssim_overlap",
        "RGB SSIM on overlap",
        "higher",
        "两张 Blender beauty render 在交集区域的局部结构相似度",
    ),
    (
        "rgb_lpips_overlap",
        "RGB LPIPS on overlap",
        "lower",
        "两张 Blender beauty render 的 LPIPS 感知距离",
    ),
)

VALID_STATUS = {"success", "skipped"}


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def fmt(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def requested_resolutions(rows: Sequence[Mapping[str, str]]) -> List[int]:
    resolutions = sorted(
        {
            int(value)
            for row in rows
            if (value := finite_float(row.get("pipeline_resolution"))) is not None
        }
    )
    if len(resolutions) < 2:
        raise ValueError(
            "Need at least two pipeline resolutions in all_views.csv; "
            f"found {resolutions}"
        )
    return resolutions


def successful_rows(rows: Sequence[Mapping[str, str]]) -> List[Mapping[str, str]]:
    return [row for row in rows if str(row.get("status", "")).strip() in VALID_STATUS]


def means_by_resolution(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    resolutions: Sequence[int],
) -> Tuple[Dict[int, Optional[float]], Dict[int, int]]:
    grouped: Dict[int, List[float]] = {resolution: [] for resolution in resolutions}
    for row in rows:
        resolution_value = finite_float(row.get("pipeline_resolution"))
        metric_value = finite_float(row.get(metric))
        if resolution_value is None or metric_value is None:
            continue
        resolution = int(resolution_value)
        if resolution in grouped:
            grouped[resolution].append(metric_value)
    return (
        {resolution: mean(values) for resolution, values in grouped.items()},
        {resolution: len(values) for resolution, values in grouped.items()},
    )


def group_pairs_from_all_views(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    reference_resolution: int,
    target_resolution: int,
) -> List[Tuple[float, float]]:
    grouped: Dict[Tuple[str, str, str], Dict[int, float]] = defaultdict(dict)
    for row in rows:
        metric_value = finite_float(row.get(metric))
        resolution_value = finite_float(row.get("pipeline_resolution"))
        if metric_value is None or resolution_value is None:
            continue
        key = (
            str(row.get("asset_id", "")),
            str(row.get("view_index", "")),
            str(row.get("seed", "")),
        )
        grouped[key][int(resolution_value)] = metric_value
    pairs: List[Tuple[float, float]] = []
    for by_resolution in grouped.values():
        if reference_resolution in by_resolution and target_resolution in by_resolution:
            pairs.append(
                (
                    by_resolution[reference_resolution],
                    by_resolution[target_resolution],
                )
            )
    return pairs


def pairs_from_paired_csv(
    paired_rows: Sequence[Mapping[str, str]],
    metric: str,
    reference_resolution: int,
    target_resolution: int,
) -> List[Tuple[float, float]]:
    reference_key = f"{metric}__r{reference_resolution}"
    target_key = f"{metric}__r{target_resolution}"
    pairs: List[Tuple[float, float]] = []
    for row in paired_rows:
        reference = finite_float(row.get(reference_key))
        target = finite_float(row.get(target_key))
        if reference is not None and target is not None:
            pairs.append((reference, target))
    return pairs


def paired_counts(
    pairs: Sequence[Tuple[float, float]],
    preferred_direction: str,
    tolerance: float,
) -> Dict[str, object]:
    better = worse = tied = 0
    deltas: List[float] = []
    for reference, target in pairs:
        delta = target - reference
        deltas.append(delta)
        if abs(delta) <= tolerance:
            tied += 1
        elif (delta > 0 and preferred_direction == "higher") or (
            delta < 0 and preferred_direction == "lower"
        ):
            better += 1
        else:
            worse += 1
    count = len(pairs)
    return {
        "paired_views": count,
        "target_better_views": better,
        "target_worse_views": worse,
        "tied_views": tied,
        "target_better_percent": (100.0 * better / count) if count else None,
        "paired_mean_delta": mean(deltas),
    }


def metric_conclusion(delta: Optional[float], direction: str, tolerance: float) -> str:
    if delta is None:
        return "无可用配对数据"
    if abs(delta) <= tolerance:
        return "基本持平"
    improved = (delta > 0 and direction == "higher") or (
        delta < 0 and direction == "lower"
    )
    return "1536 改善" if improved else "1536 变差"


def normal_overall(metric_deltas: Mapping[str, Optional[float]]) -> str:
    iou = metric_deltas.get("normal_iou_percent")
    fine_metrics = (
        ("normal_ssim_overlap", "higher"),
        ("normal_lpips_overlap", "lower"),
        ("normal_mean_angular_error_deg", "lower"),
        ("normal_median_angular_error_deg", "lower"),
        ("normal_acc_11_25_percent", "higher"),
    )
    fine_better = fine_worse = 0
    for metric, direction in fine_metrics:
        delta = metric_deltas.get(metric)
        if delta is None or abs(delta) <= 1e-12:
            continue
        improved = (delta > 0 and direction == "higher") or (
            delta < 0 and direction == "lower"
        )
        if improved:
            fine_better += 1
        else:
            fine_worse += 1
    boundary = metric_deltas.get("normal_boundary_mean_angular_error_deg")
    if iou is not None and iou > 0 and fine_worse > fine_better:
        suffix = "；边界角度误差改善" if boundary is not None and boundary < 0 else ""
        return f"Raw decoder mesh：轮廓明显改善，但精细法向量整体变差{suffix}。"
    if iou is not None and iou > 0 and fine_better > fine_worse:
        return "Raw decoder mesh：轮廓与精细法向量整体改善。"
    if iou is not None and iou < 0 and fine_worse >= fine_better:
        return "Raw decoder mesh：轮廓与精细法向量整体变差。"
    return "Raw decoder mesh：不同法向量指标方向混合，需逐项解释。"


def rgb_overall(metric_deltas: Mapping[str, Optional[float]]) -> str:
    directions = {
        "rgb_iou_percent": "higher",
        "rgb_psnr_overlap_db": "higher",
        "rgb_ssim_overlap": "higher",
        "rgb_lpips_overlap": "lower",
    }
    better = worse = 0
    for metric, direction in directions.items():
        delta = metric_deltas.get(metric)
        if delta is None or abs(delta) <= 1e-12:
            continue
        improved = (delta > 0 and direction == "higher") or (
            delta < 0 and direction == "lower"
        )
        if improved:
            better += 1
        else:
            worse += 1
    if better >= 3:
        return "最终导出 GLB：综合 RGB/外观质量整体改善。"
    if worse >= 3:
        return "最终导出 GLB：综合 RGB/外观质量整体变差。"
    return "最终导出 GLB：RGB/外观指标方向混合。"


def build_summary(
    all_rows: Sequence[Mapping[str, str]],
    paired_rows: Optional[Sequence[Mapping[str, str]]],
    metrics: Sequence[Tuple[str, str, str, str]],
    resolutions: Sequence[int],
    tolerance: float,
    overall_builder,
) -> List[Dict[str, str]]:
    reference_resolution = resolutions[0]
    target_resolution = resolutions[-1]
    valid_rows = successful_rows(all_rows)
    output: List[Dict[str, str]] = []
    deltas: Dict[str, Optional[float]] = {}

    for metric, label, direction, description in metrics:
        means, counts = means_by_resolution(valid_rows, metric, resolutions)
        reference_mean = means[reference_resolution]
        target_mean = means[target_resolution]
        delta = (
            target_mean - reference_mean
            if reference_mean is not None and target_mean is not None
            else None
        )
        deltas[metric] = delta

        if paired_rows is not None:
            pairs = pairs_from_paired_csv(
                paired_rows, metric, reference_resolution, target_resolution
            )
        else:
            pairs = group_pairs_from_all_views(
                valid_rows, metric, reference_resolution, target_resolution
            )
        pair_stats = paired_counts(pairs, direction, tolerance)
        relative_change = (
            100.0 * delta / abs(reference_mean)
            if delta is not None and reference_mean not in (None, 0.0)
            else None
        )

        row: Dict[str, str] = {
            "metric": metric,
            "metric_label": label,
            "description": description,
            "preferred_direction": direction,
            f"mean_r{reference_resolution}": fmt(reference_mean),
            f"n_r{reference_resolution}": str(counts[reference_resolution]),
            f"mean_r{target_resolution}": fmt(target_mean),
            f"n_r{target_resolution}": str(counts[target_resolution]),
            f"delta_r{target_resolution}_minus_r{reference_resolution}": fmt(delta),
            "relative_change_percent": fmt(relative_change),
            "paired_views": str(pair_stats["paired_views"]),
            f"r{target_resolution}_better_views": str(pair_stats["target_better_views"]),
            f"r{target_resolution}_worse_views": str(pair_stats["target_worse_views"]),
            "tied_views": str(pair_stats["tied_views"]),
            f"r{target_resolution}_better_percent": fmt(
                pair_stats["target_better_percent"]  # type: ignore[arg-type]
            ),
            "paired_mean_delta": fmt(
                pair_stats["paired_mean_delta"]  # type: ignore[arg-type]
            ),
            "conclusion": metric_conclusion(delta, direction, tolerance),
        }
        output.append(row)

    overall = overall_builder(deltas)
    output.append(
        {
            "metric": "__overall__",
            "metric_label": "整体结论",
            "description": overall,
            "preferred_direction": "",
            f"mean_r{reference_resolution}": "",
            f"n_r{reference_resolution}": "",
            f"mean_r{target_resolution}": "",
            f"n_r{target_resolution}": "",
            f"delta_r{target_resolution}_minus_r{reference_resolution}": "",
            "relative_change_percent": "",
            "paired_views": "",
            f"r{target_resolution}_better_views": "",
            f"r{target_resolution}_worse_views": "",
            "tied_views": "",
            f"r{target_resolution}_better_percent": "",
            "paired_mean_delta": "",
            "conclusion": overall,
        }
    )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-views-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-12,
        help="Absolute delta treated as a tie (default: 1e-12).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, all_rows = read_csv(args.all_views_csv)
    paired_rows: Optional[List[Dict[str, str]]] = None
    if args.paired_csv is not None:
        _, paired_rows = read_csv(args.paired_csv)

    resolutions = requested_resolutions(all_rows)
    normal_rows = build_summary(
        all_rows=all_rows,
        paired_rows=paired_rows,
        metrics=NORMAL_METRICS,
        resolutions=resolutions,
        tolerance=args.tie_tolerance,
        overall_builder=normal_overall,
    )
    rgb_rows = build_summary(
        all_rows=all_rows,
        paired_rows=paired_rows,
        metrics=RGB_METRICS,
        resolutions=resolutions,
        tolerance=args.tie_tolerance,
        overall_builder=rgb_overall,
    )

    normal_path = args.output_dir / "normal_metrics_mean_summary.csv"
    rgb_path = args.output_dir / "rgb_metrics_mean_summary.csv"
    write_csv(normal_path, normal_rows)
    write_csv(rgb_path, rgb_rows)

    print(f"resolutions={resolutions}")
    print(f"normal_summary={normal_path}")
    print(f"rgb_summary={rgb_path}")
    print(normal_rows[-1]["conclusion"])
    print(rgb_rows[-1]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
