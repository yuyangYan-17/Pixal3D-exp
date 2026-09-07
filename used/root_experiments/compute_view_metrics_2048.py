#!/usr/bin/env python3
"""Compare generated and baseline 2048 renders against the three input views."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from compute_input_view_metrics_1024 import _compute_lpips, _metric_pair


ANGLES = (0, 120, 240)


def _load_resized(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--baseline-render-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--foreground-threshold", type=float, default=4.0 / 255.0)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    args = parser.parse_args()
    generated_root = args.generated_root.resolve()
    baseline_root = args.baseline_render_root.resolve()
    rows: List[Dict[str, Any]] = []
    lpips_pairs: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for angle in ANGLES:
        reference = _load_resized(generated_root / "inputs" / f"view_{angle:03d}.png", args.resolution)
        baseline = _load_resized(baseline_root / "renders" / f"yaw{angle:03d}.png", args.resolution)
        generated = _load_resized(generated_root / "renders" / f"yaw{angle:03d}.png", args.resolution)
        foreground = np.max(reference, axis=-1) > float(args.foreground_threshold)
        baseline_fg = np.max(baseline, axis=-1) > float(args.foreground_threshold)
        generated_fg = np.max(generated, axis=-1) > float(args.foreground_threshold)
        baseline_metrics = _metric_pair(reference, baseline, foreground, baseline_fg)
        generated_metrics = _metric_pair(reference, generated, foreground, generated_fg)
        rows.append({
            "angle": angle,
            "reference": str(generated_root / "inputs" / f"view_{angle:03d}.png"),
            "baseline_render": str(baseline_root / "renders" / f"yaw{angle:03d}.png"),
            "generated_render": str(generated_root / "renders" / f"yaw{angle:03d}.png"),
            "resolution": [args.resolution, args.resolution],
            "baseline": baseline_metrics,
            "generated": generated_metrics,
            "delta_generated_minus_baseline": {
                section: {name: float(generated_metrics[section][name] - baseline_metrics[section][name])
                          for name in ("psnr_db", "ssim", "mae", "mse") if name in baseline_metrics[section]}
                for section in ("full_image", "input_foreground_only")
            },
            "silhouette_delta_iou": float(generated_metrics["silhouette"]["iou"] - baseline_metrics["silhouette"]["iou"]),
        })
        lpips_pairs.extend([(angle, reference, baseline), (angle, reference, generated)])

    def mean(method: str, section: str, name: str) -> float:
        return float(np.mean([row[method][section][name] for row in rows]))

    summary: Dict[str, Any] = {
        "full_image": {method: {name: mean(method, "full_image", name) for name in ("psnr_db", "ssim", "mae", "mse")} for method in ("baseline", "generated")},
        "input_foreground_only": {method: {name: mean(method, "input_foreground_only", name) for name in ("psnr_db", "ssim", "mae", "mse")} for method in ("baseline", "generated")},
        "silhouette": {method: float(np.mean([row[method]["silhouette"]["iou"] for row in rows])) for method in ("baseline", "generated")},
    }
    for section in ("full_image", "input_foreground_only"):
        summary[f"delta_generated_minus_baseline_{section}"] = {
            name: float(summary[section]["generated"][name] - summary[section]["baseline"][name])
            for name in ("psnr_db", "ssim", "mae", "mse")
        }
    summary["delta_generated_minus_baseline_silhouette_iou"] = summary["silhouette"]["generated"] - summary["silhouette"]["baseline"]
    lpips_values: Optional[List[float]] = None
    lpips_error: Optional[str] = None
    if args.lpips:
        values, lpips_error = _compute_lpips(lpips_pairs, args.lpips_net)
        if values is not None:
            lpips_values = values
            summary["lpips"] = {
                "baseline": float(np.mean(values[0::2])),
                "generated": float(np.mean(values[1::2])),
                "delta_generated_minus_baseline": float(np.mean(values[1::2]) - np.mean(values[0::2])),
            }
    result = {
        "status": "complete",
        "resolution": [args.resolution, args.resolution],
        "reference_resize": "input view resized with Lanczos only because source is 1024 and renders are 2048",
        "foreground_definition": f"max(R,G,B) > {args.foreground_threshold:.10f}",
        "rows": rows,
        "mean_over_three_views": summary,
        "lpips": {"requested": bool(args.lpips), "network": args.lpips_net if args.lpips else None, "values_interleaved_baseline_generated": lpips_values, "error": lpips_error},
    }
    out_json = generated_root / "metrics_2048_vs_baseline.json"
    out_csv = generated_root / "metrics_2048_vs_baseline.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["angle", "baseline_foreground_psnr_db", "generated_foreground_psnr_db", "delta_foreground_psnr_db", "baseline_foreground_ssim", "generated_foreground_ssim", "baseline_silhouette_iou", "generated_silhouette_iou"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "angle": row["angle"],
                "baseline_foreground_psnr_db": row["baseline"]["input_foreground_only"]["psnr_db"],
                "generated_foreground_psnr_db": row["generated"]["input_foreground_only"]["psnr_db"],
                "delta_foreground_psnr_db": row["delta_generated_minus_baseline"]["input_foreground_only"]["psnr_db"],
                "baseline_foreground_ssim": row["baseline"]["input_foreground_only"]["ssim"],
                "generated_foreground_ssim": row["generated"]["input_foreground_only"]["ssim"],
                "baseline_silhouette_iou": row["baseline"]["silhouette"]["iou"],
                "generated_silhouette_iou": row["generated"]["silhouette"]["iou"],
            })
    print(json.dumps({"json": str(out_json), "csv": str(out_csv), "mean_over_three_views": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
