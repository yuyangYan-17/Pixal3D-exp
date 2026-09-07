#!/usr/bin/env python3
"""Render the encoder-downsample experiment in its input view and annotate metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from render_pixal3d_raw_ovoxel import (
    load_mesh_checkpoint,
    render_static_ovoxel,
    save_render_outputs,
)


ROOT = Path("outputs/global_c256_encdown_context1024_flow_singleview_cuda4")
BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=ROOT / "final/final_material_mesh.pt")
    p.add_argument("--reference", type=Path, default=BASELINE / "canonical_1024.png")
    p.add_argument("--reference-mask", type=Path, default=BASELINE / "raw_ovoxel_render/alpha.png")
    p.add_argument("--camera", type=Path, default=BASELINE / "global_camera.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "aligned_input_view_1024")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--face-chunk-size", type=int, default=4_000_000)
    return p


def psnr(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> int:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera = json.loads(args.camera.read_text(encoding="utf-8"))

    mesh = load_mesh_checkpoint(args.mesh, device=device)
    renders = render_static_ovoxel(
        mesh,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        resolution=int(args.resolution),
        envmap="studio",
        ssaa=1,
        peel_layers=8,
        face_chunk_size=int(args.face_chunk_size),
        use_envmap_bg=False,
    )
    paths = save_render_outputs(renders, args.output_dir)
    del mesh, renders
    torch.cuda.empty_cache()

    size = (int(args.resolution), int(args.resolution))
    reference_image = Image.open(args.reference).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    prediction_image = Image.open(paths["render"]).convert("RGB")
    reference = np.asarray(reference_image, dtype=np.float32) / 255.0
    prediction = np.asarray(prediction_image, dtype=np.float32) / 255.0
    reference_mask = np.asarray(
        Image.open(args.reference_mask).convert("L").resize(size, Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0
    prediction_mask = np.asarray(Image.open(paths["alpha"]).convert("L"), dtype=np.float32) / 255.0
    foreground = reference_mask > 0.5
    diff = prediction - reference
    full_mse = float(np.mean(diff * diff))
    foreground_mse = float(np.mean(diff[foreground] ** 2))
    _, ssim_map = structural_similarity(reference, prediction, data_range=1.0, channel_axis=2, full=True)
    ref_alpha = foreground
    pred_alpha = prediction_mask > 0.5
    intersection = int(np.logical_and(ref_alpha, pred_alpha).sum())
    union = int(np.logical_or(ref_alpha, pred_alpha).sum())

    # Run LPIPS only after releasing the multi-gigabyte mesh tensors.
    from pixal3d_global4096_tile_endpoint_rollout_sync import _lpips_native_patches

    lpips = _lpips_native_patches(reference, prediction, patch_size=512)
    metrics = {
        "reference": str(args.reference.resolve()),
        "reference_foreground_mask": str(args.reference_mask.resolve()),
        "render": str(Path(paths["render"]).resolve()),
        "camera": camera,
        "resolution": list(size),
        "psnr_db": psnr(full_mse),
        "foreground_psnr_db": psnr(foreground_mse),
        "ssim": float(np.mean(ssim_map)),
        "foreground_ssim": float(np.mean(ssim_map[foreground])),
        "mae": float(np.mean(np.abs(diff))),
        "foreground_mae": float(np.mean(np.abs(diff[foreground]))),
        "lpips_alex_512_patches": None if lpips is None else float(lpips),
        "alpha_iou_vs_baseline_mask": float(intersection / max(union, 1)),
        "foreground_pixels": int(foreground.sum()),
        "metric_range": "RGB [0,1]",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    error = np.clip(np.mean(np.abs(diff), axis=2) * 4.0, 0.0, 1.0)
    heat = np.stack((error, error**2, np.zeros_like(error)), axis=2)
    heat_image = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255.0 + 0.5), "RGB")
    header_h = 150
    canvas = Image.new("RGB", (size[0] * 3, size[1] + header_h), (18, 21, 27))
    canvas.paste(reference_image, (0, header_h))
    canvas.paste(prediction_image, (size[0], header_h))
    canvas.paste(heat_image, (size[0] * 2, header_h))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(32)
    metric_font = load_font(25)
    draw.text((24, 14), "Input-aligned final render", fill=(245, 247, 250), font=title_font)
    line = (
        f"PSNR {metrics['psnr_db']:.3f} dB   FG-PSNR {metrics['foreground_psnr_db']:.3f} dB   "
        f"SSIM {metrics['ssim']:.4f}   FG-SSIM {metrics['foreground_ssim']:.4f}"
    )
    line2 = (
        f"MAE {metrics['mae']:.4f}   FG-MAE {metrics['foreground_mae']:.4f}   "
        f"LPIPS {metrics['lpips_alex_512_patches']:.4f}   Alpha IoU {metrics['alpha_iou_vs_baseline_mask']:.4f}"
    )
    draw.text((24, 58), line, fill=(114, 210, 255), font=metric_font)
    draw.text((24, 94), line2, fill=(255, 205, 112), font=metric_font)
    for x, label in zip((24, size[0] + 24, size[0] * 2 + 24), ("INPUT", "RENDER", "ABS ERROR x4")):
        draw.text((x, header_h + 18), label, fill=(255, 255, 255), font=metric_font, stroke_width=2, stroke_fill=(0, 0, 0))
    canvas.save(args.output_dir / "comparison_with_metrics.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"comparison={args.output_dir / 'comparison_with_metrics.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
