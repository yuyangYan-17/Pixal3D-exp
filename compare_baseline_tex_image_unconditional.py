#!/usr/bin/env python3
"""Compare baseline C64 texture flow with and without image conditioning."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity


COND = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
UNCOND = Path("outputs/baseline1024_tex_image_unconditional_cuda4_0_img")
OUT = UNCOND / "comparison"


def psnr(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def evaluate(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = pred - ref
    mse = float(np.mean(diff * diff))
    fg_mse = float(np.mean(diff[mask] ** 2))
    _, ssim_map = structural_similarity(ref, pred, data_range=1.0, channel_axis=2, full=True)
    return {
        "psnr_db": psnr(mse),
        "foreground_psnr_db": psnr(fg_mse),
        "ssim": float(np.mean(ssim_map)),
        "foreground_ssim": float(np.mean(ssim_map[mask])),
        "mae": float(np.mean(np.abs(diff))),
        "foreground_mae": float(np.mean(np.abs(diff[mask]))),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reference_image = Image.open(COND / "canonical_1024.png").convert("RGB")
    cond_image = Image.open(COND / "raw_ovoxel_render/render.png").convert("RGB")
    uncond_image = Image.open(UNCOND / "raw_ovoxel_render/render.png").convert("RGB")
    reference = np.asarray(reference_image, dtype=np.float32) / 255.0
    conditional = np.asarray(cond_image, dtype=np.float32) / 255.0
    unconditional = np.asarray(uncond_image, dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(COND / "raw_ovoxel_render/alpha.png").convert("L"), dtype=np.float32) > 127.5

    cond_metrics = evaluate(reference, conditional, mask)
    uncond_metrics = evaluate(reference, unconditional, mask)
    delta = {key: uncond_metrics[key] - cond_metrics[key] for key in cond_metrics}
    render_delta = np.abs(unconditional - conditional)
    render_delta_metrics = {
        "mae": float(render_delta.mean()),
        "foreground_mae": float(render_delta[mask].mean()),
        "psnr_db": psnr(float(np.mean((unconditional - conditional) ** 2))),
    }

    from pixal3d_global4096_tile_endpoint_rollout_sync import _lpips_native_patches

    cond_metrics["lpips_alex_512_patches"] = float(_lpips_native_patches(reference, conditional, 512))
    uncond_metrics["lpips_alex_512_patches"] = float(_lpips_native_patches(reference, unconditional, 512))
    delta["lpips_alex_512_patches"] = (
        uncond_metrics["lpips_alex_512_patches"] - cond_metrics["lpips_alex_512_patches"]
    )

    # The texture switch must not perturb geometry generated before texture flow.
    cond_mesh = torch.load(COND / "raw_ovoxel_mesh.pt", map_location="cpu", weights_only=False)
    uncond_mesh = torch.load(UNCOND / "raw_ovoxel_mesh.pt", map_location="cpu", weights_only=False)
    geometry = {
        "vertices_exactly_equal": bool(torch.equal(cond_mesh["vertices"], uncond_mesh["vertices"])),
        "faces_exactly_equal": bool(torch.equal(cond_mesh["faces"], uncond_mesh["faces"])),
        "coords_exactly_equal": bool(torch.equal(cond_mesh["coords"], uncond_mesh["coords"])),
        "conditional_vertices": int(cond_mesh["vertices"].shape[0]),
        "conditional_faces": int(cond_mesh["faces"].shape[0]),
    }
    del cond_mesh, uncond_mesh

    payload = {
        "experiment": "baseline1024 C64 texture image-condition ablation",
        "same_input_and_seed": True,
        "conditional": cond_metrics,
        "image_unconditional_shape_concat_retained": uncond_metrics,
        "delta_unconditional_minus_conditional": delta,
        "render_to_render": render_delta_metrics,
        "geometry": geometry,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    heat = np.stack(
        (np.clip(render_delta.mean(2) * 4, 0, 1), np.clip(render_delta.mean(2) * 2, 0, 1), np.zeros(mask.shape)),
        axis=2,
    )
    heat_image = Image.fromarray(np.uint8(heat * 255.0 + 0.5), "RGB")
    images = (reference_image, cond_image, uncond_image, heat_image)
    labels = ("INPUT", "TEX IMAGE CONDITIONAL", "TEX IMAGE UNCONDITIONAL", "COND vs UNCOND |DIFF| x4")
    w, h = reference_image.size
    header = 190
    canvas = Image.new("RGB", (4 * w, h + header), (18, 21, 27))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "Baseline 1024 · C64 Texture Flow image-condition ablation", fill="white", font=font(30))
    c, u = cond_metrics, uncond_metrics
    draw.text((20, 56), f"Conditional:   PSNR {c['psnr_db']:.3f}  FG {c['foreground_psnr_db']:.3f}  SSIM {c['ssim']:.4f}  LPIPS {c['lpips_alex_512_patches']:.4f}", fill=(100, 205, 255), font=font(23))
    draw.text((20, 91), f"Unconditional: PSNR {u['psnr_db']:.3f}  FG {u['foreground_psnr_db']:.3f}  SSIM {u['ssim']:.4f}  LPIPS {u['lpips_alex_512_patches']:.4f}", fill=(255, 197, 100), font=font(23))
    draw.text((20, 126), f"Delta U-C: PSNR {delta['psnr_db']:+.3f} dB  FG {delta['foreground_psnr_db']:+.3f} dB  SSIM {delta['ssim']:+.4f}  LPIPS {delta['lpips_alex_512_patches']:+.4f}", fill=(255, 130, 130), font=font(23))
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * w
        canvas.paste(image, (x, header))
        draw.text((x + 16, header + 14), label, fill="white", font=font(21), stroke_width=2, stroke_fill="black")
    canvas.save(OUT / "comparison_with_metrics.png")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
