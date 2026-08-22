#!/usr/bin/env python3
"""Compute 1024x1024 metrics between the three input views and renders.

The experiment writes RGB PNGs with a black background.  This script keeps the
images at their native 1024x1024 resolution (it never resizes them), reports
the usual full-image metrics and also reports foreground-only values so that
the black canvas does not hide reconstruction errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import convolve


ANGLES: Tuple[int, ...] = (0, 120, 240)
DEFAULT_ROOT = Path("outputs/global_c4096_visible_local_flow_cuda4")


def _load_rgb(path: Path, expected_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (expected_size, expected_size):
            raise ValueError(
                f"{path} is {rgb.size}, expected {expected_size}x{expected_size}; "
                "metrics are intentionally not resized"
            )
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(window_size, dtype=np.float64)
    coords -= (window_size - 1) / 2.0
    gaussian = np.exp(-(coords**2) / (2.0 * sigma**2))
    gaussian /= gaussian.sum()
    return np.outer(gaussian, gaussian).astype(np.float32)


def _blur_channels(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.stack(
        [convolve(image[..., channel], kernel, mode="constant", cval=0.0)
         for channel in range(image.shape[-1])],
        axis=-1,
    )


def _ssim_map(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    kernel = _gaussian_kernel()
    mu_ref = _blur_channels(reference, kernel)
    mu_pred = _blur_channels(prediction, kernel)
    mu_ref_sq = mu_ref * mu_ref
    mu_pred_sq = mu_pred * mu_pred
    mu_cross = mu_ref * mu_pred
    sigma_ref = _blur_channels(reference * reference, kernel) - mu_ref_sq
    sigma_pred = _blur_channels(prediction * prediction, kernel) - mu_pred_sq
    sigma_cross = _blur_channels(reference * prediction, kernel) - mu_cross
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_ref_sq + mu_pred_sq + c1) * (sigma_ref + sigma_pred + c2)
    )
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    return score.mean(axis=-1)


def _psnr(mse: float) -> float:
    return float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def _metric_pair(
    reference: np.ndarray,
    prediction: np.ndarray,
    foreground: np.ndarray,
    render_foreground: np.ndarray,
) -> Dict[str, Any]:
    diff = prediction - reference
    foreground = foreground.astype(bool)
    render_foreground = render_foreground.astype(bool)
    count = int(foreground.sum())
    if count == 0:
        raise ValueError("input foreground mask is empty")
    channel_count = int(reference.shape[-1])
    full_mse = float(np.mean(diff * diff))
    foreground_diff = diff[foreground]
    foreground_mse = float(np.mean(foreground_diff * foreground_diff))
    full_mae = float(np.mean(np.abs(diff)))
    foreground_mae = float(np.mean(np.abs(foreground_diff)))
    ssim = _ssim_map(reference, prediction)
    intersection = int(np.logical_and(foreground, render_foreground).sum())
    union = int(np.logical_or(foreground, render_foreground).sum())
    return {
        "full_image": {
            "psnr_db": _psnr(full_mse),
            "ssim": float(ssim.mean()),
            "mae": full_mae,
            "mse": full_mse,
        },
        "input_foreground_only": {
            "psnr_db": _psnr(foreground_mse),
            "ssim": float(ssim[foreground].mean()),
            "mae": foreground_mae,
            "mse": foreground_mse,
        },
        "silhouette": {
            "input_foreground_pixels": count,
            "render_foreground_pixels": int(render_foreground.sum()),
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": float(intersection / union) if union else 1.0,
            "input_fraction": float(foreground.mean()),
            "render_fraction": float(render_foreground.mean()),
        },
    }


def _mean_metric(rows: Sequence[Dict[str, Any]], section: str, name: str) -> float:
    values = [float(row["metrics"][section][name]) for row in rows]
    return float(np.mean(values))


def _compute_lpips(
    pairs: Sequence[Tuple[int, np.ndarray, np.ndarray]], network: str
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Run LPIPS when requested; return a useful error instead of failing PSNR/SSIM."""
    try:
        import torch
        import lpips

        model = lpips.LPIPS(net=network, verbose=False).eval().to("cpu")
        values: List[float] = []
        with torch.no_grad():
            for angle, reference, prediction in pairs:
                ref = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
                pred = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
                value = float(model(ref, pred).item())
                values.append(value)
                print(f"[lpips] yaw{angle:03d}: {value:.8f}", flush=True)
        return values, None
    except Exception as exc:  # pragma: no cover - depends on local torchvision/cache
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--foreground-threshold", type=float, default=4.0 / 255.0)
    parser.add_argument("--lpips", action="store_true", help="also run LPIPS using local weights")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    args = parser.parse_args()

    root = args.root.resolve()
    pairs: List[Tuple[int, np.ndarray, np.ndarray]] = []
    rows: List[Dict[str, Any]] = []
    for angle in ANGLES:
        input_path = root / "inputs" / f"view_{angle:03d}.png"
        render_path = root / "renders" / f"yaw{angle:03d}.png"
        reference = _load_rgb(input_path, args.resolution)
        prediction = _load_rgb(render_path, args.resolution)
        input_foreground = np.max(reference, axis=-1) > args.foreground_threshold
        render_foreground = np.max(prediction, axis=-1) > args.foreground_threshold
        metrics = _metric_pair(reference, prediction, input_foreground, render_foreground)
        rows.append({
            "angle": angle,
            "input": str(input_path),
            "render": str(render_path),
            "resolution": [args.resolution, args.resolution],
            "metrics": metrics,
        })
        pairs.append((angle, reference, prediction))

    lpips_values: Optional[List[float]] = None
    lpips_error: Optional[str] = None
    if args.lpips:
        lpips_values, lpips_error = _compute_lpips(pairs, args.lpips_net)
        if lpips_values is not None:
            for row, value in zip(rows, lpips_values):
                row["metrics"]["lpips"] = value

    mean: Dict[str, Any] = {
        "full_image": {
            name: _mean_metric(rows, "full_image", name)
            for name in ("psnr_db", "ssim", "mae", "mse")
        },
        "input_foreground_only": {
            name: _mean_metric(rows, "input_foreground_only", name)
            for name in ("psnr_db", "ssim", "mae", "mse")
        },
        "silhouette": {
            name: _mean_metric(rows, "silhouette", name)
            for name in ("iou", "input_fraction", "render_fraction")
        },
    }
    if lpips_values is not None:
        mean["lpips"] = float(np.mean(lpips_values))

    result: Dict[str, Any] = {
        "status": "complete",
        "root": str(root),
        "resolution": [args.resolution, args.resolution],
        "pairing": {f"yaw{angle:03d}": f"view_{angle:03d}" for angle in ANGLES},
        "foreground_definition": f"max(R,G,B) > {args.foreground_threshold:.10f} on each input/render RGB image",
        "resize_performed": False,
        "rows": rows,
        "mean_over_three_views": mean,
        "lpips": {
            "requested": bool(args.lpips),
            "network": args.lpips_net if args.lpips else None,
            "values": lpips_values,
            "mean": float(np.mean(lpips_values)) if lpips_values is not None else None,
            "error": lpips_error,
        },
    }
    json_path = root / "metrics_1024.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = root / "metrics_1024.csv"
    csv_fields = [
        "angle", "full_psnr_db", "full_ssim", "full_mae", "full_mse",
        "foreground_psnr_db", "foreground_ssim", "foreground_mae", "foreground_mse",
        "silhouette_iou", "input_foreground_pixels", "render_foreground_pixels",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            full = row["metrics"]["full_image"]
            foreground = row["metrics"]["input_foreground_only"]
            silhouette = row["metrics"]["silhouette"]
            writer.writerow({
                "angle": row["angle"],
                "full_psnr_db": full["psnr_db"],
                "full_ssim": full["ssim"],
                "full_mae": full["mae"],
                "full_mse": full["mse"],
                "foreground_psnr_db": foreground["psnr_db"],
                "foreground_ssim": foreground["ssim"],
                "foreground_mae": foreground["mae"],
                "foreground_mse": foreground["mse"],
                "silhouette_iou": silhouette["iou"],
                "input_foreground_pixels": silhouette["input_foreground_pixels"],
                "render_foreground_pixels": silhouette["render_foreground_pixels"],
            })

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
