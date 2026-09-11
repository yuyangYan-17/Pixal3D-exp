#!/usr/bin/env python3
"""Re-noise the final baseline C64 texture SLat and unconditionally denoise it.

The experiment first runs the complete native Pixal3D 1024 geometry and
texture flow.  Its final normalized texture SLat is treated as clean ``x0``.
For each of the 12 uniform sampler start times, the same fixed Gaussian noise
is mixed with ``x0`` using Pixal3D's native flow forward-process formula.  The
state is then integrated from that time to zero with zero image condition;
the fixed baseline shape concat condition is retained.

Only yaw-180 (back) PBR renders are evaluated.  Every variant is compared to
the back render of the baseline using the baseline foreground mask.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d.representations import MeshWithVoxel
from pixal3d.utils import render_utils
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT = "pixal3d_baseline1024_tex_slat_renoise_uncond_back_sweep_v1"
STEPS = 12
DEFAULT_BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
DEFAULT_OUTPUT = Path("outputs/baseline1024_tex_slat_renoise_uncond_back_cuda4")
RENDER_MODES = ("shaded", "base_color", "metallic", "roughness", "alpha", "normal", "mask")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def tensor_to_array(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    if value.ndim == 3:
        value = value.permute(1, 2, 0)
    elif value.ndim != 2:
        raise ValueError(f"unsupported render tensor shape: {tuple(value.shape)}")
    return np.nan_to_num(value.numpy(), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def array_to_image(value: np.ndarray) -> Image.Image:
    value = np.clip(value, 0.0, 1.0)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[..., 0]
    if value.ndim == 2:
        return Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="L")
    if value.ndim == 3 and value.shape[2] == 3:
        return Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    raise ValueError(f"unsupported image array shape: {value.shape}")


def save_render(result: Mapping[str, torch.Tensor], output_dir: Path) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, np.ndarray] = {}
    for mode in RENDER_MODES:
        if mode not in result:
            raise KeyError(f"renderer did not return {mode!r}")
        value = tensor_to_array(result[mode])
        saved[mode] = value
        array_to_image(value).save(output_dir / f"{mode}.png")
    array_to_image(saved["shaded"]).save(output_dir / "render.png")
    return saved


def back_camera(camera_angle_x: float, distance: float) -> tuple[torch.Tensor, torch.Tensor]:
    front, intrinsics = render_utils.proj_camera_to_render_params(
        camera_angle_x=float(camera_angle_x), distance=float(distance)
    )
    rotation_y_180 = torch.tensor(
        [[-1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=front.dtype,
        device=front.device,
    )
    return front @ rotation_y_180, intrinsics


def make_mesh(geometry: Any, tex_voxel: SparseTensor, resolution: int, layout: Mapping[str, slice]) -> MeshWithVoxel:
    return MeshWithVoxel(
        geometry.vertices,
        geometry.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / int(resolution),
        coords=tex_voxel.coords[:, 1:],
        attrs=tex_voxel.feats,
        voxel_shape=torch.Size([*tex_voxel.shape, *tex_voxel.spatial_shape]),
        layout=dict(layout),
    )


def ssim_map(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    from skimage.metrics import structural_similarity

    _, score = structural_similarity(
        reference,
        prediction,
        data_range=1.0,
        channel_axis=2,
        full=True,
    )
    return score.astype(np.float32)


def metrics(reference: np.ndarray, prediction: np.ndarray, foreground: np.ndarray) -> dict[str, float]:
    if reference.shape != prediction.shape:
        raise ValueError(f"render shape mismatch: {reference.shape} != {prediction.shape}")
    if foreground.shape != reference.shape[:2] or not foreground.any():
        raise ValueError("baseline foreground mask is empty or misaligned")
    difference = prediction - reference
    mse = float(np.mean(difference[foreground] ** 2))
    mae = float(np.mean(np.abs(difference[foreground])))
    score = ssim_map(reference, prediction)
    return {
        "foreground_psnr_db": float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse)),
        "foreground_ssim": float(np.mean(score[foreground])),
        "foreground_mae": mae,
        "foreground_mse": mse,
        "foreground_pixels": int(foreground.sum()),
    }


def error_heatmap(reference: np.ndarray, prediction: np.ndarray, foreground: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    error = np.mean(np.abs(prediction - reference), axis=2)
    value = np.clip(error / float(scale), 0.0, 1.0)
    # Black -> red -> yellow -> white; a fixed scale makes all 12 panels comparable.
    rgb = np.stack(
        [np.clip(3.0 * value, 0.0, 1.0),
         np.clip(3.0 * value - 1.0, 0.0, 1.0),
         np.clip(3.0 * value - 2.0, 0.0, 1.0)],
        axis=2,
    )
    rgb[~foreground] = 0.0
    return error.astype(np.float32), rgb.astype(np.float32)


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def make_grid(
    records: list[Mapping[str, Any]],
    output: Path,
    key: str,
    title: str,
    *,
    thumb: int = 420,
    label_h: int = 92,
) -> None:
    columns, rows = 4, 3
    scale = thumb / 420.0
    gap = max(12, int(round(12 * scale)))
    canvas = Image.new(
        "RGB",
        (columns * thumb + (columns + 1) * gap, rows * (thumb + label_h) + (rows + 1) * gap),
        (18, 20, 25),
    )
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = gap + column * (thumb + gap)
        y = gap + row * (thumb + label_h + gap)
        draw.text((x + int(5 * scale), y + int(5 * scale)), f"start {record['start_step']:02d}/12 · t={record['start_t']:.4f}", fill="white", font=font(max(20, int(20 * scale))))
        draw.text(
            (x + int(5 * scale), y + int(34 * scale)),
            f"uncond {record['unconditional_steps']:02d} steps",
            fill=(255, 205, 110),
            font=font(max(17, int(17 * scale))),
        )
        draw.text(
            (x + int(5 * scale), y + int(60 * scale)),
            f"FG PSNR {record['foreground_psnr_db']:.2f} · SSIM {record['foreground_ssim']:.4f}",
            fill=(110, 205, 255),
            font=font(max(16, int(16 * scale))),
        )
        with Image.open(record[key]) as source:
            tile = source.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y + label_h))
    canvas.save(output)
    print(f"[grid] {title}: {output}", flush=True)


def make_metric_plot(
    records: list[Mapping[str, Any]],
    output: Path,
    *,
    width: int = 1400,
    height: int = 760,
) -> None:
    scale = width / 1400.0
    margin_left = int(110 * scale)
    margin_right = int(110 * scale)
    margin_top = int(75 * scale)
    margin_bottom = int(105 * scale)
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    canvas = Image.new("RGB", (width, height), (18, 20, 25))
    draw = ImageDraw.Draw(canvas)
    steps = [int(row["start_step"]) for row in records]
    psnr = [float(row["foreground_psnr_db"]) for row in records]
    ssim = [float(row["foreground_ssim"]) for row in records]
    psnr_lo, psnr_hi = math.floor(min(psnr) - 1.0), math.ceil(max(psnr) + 1.0)
    ssim_lo, ssim_hi = max(0.0, min(ssim) - 0.04), 1.0

    def x_of(index: int) -> float:
        return margin_left + index * plot_w / (len(records) - 1)

    def y_psnr(value: float) -> float:
        return margin_top + (psnr_hi - value) * plot_h / (psnr_hi - psnr_lo)

    def y_ssim(value: float) -> float:
        return margin_top + (ssim_hi - value) * plot_h / (ssim_hi - ssim_lo)

    for line in range(6):
        y = margin_top + line * plot_h / 5
        draw.line((margin_left, y, width - margin_right, y), fill=(54, 58, 67), width=max(1, int(1 * scale)))
        p_value = psnr_hi - line * (psnr_hi - psnr_lo) / 5
        s_value = ssim_hi - line * (ssim_hi - ssim_lo) / 5
        draw.text((int(12 * scale), y - int(10 * scale)), f"{p_value:.1f}", fill=(90, 205, 255), font=font(max(12, int(17 * scale))))
        draw.text((width - margin_right + int(15 * scale), y - int(10 * scale)), f"{s_value:.3f}", fill=(255, 190, 90), font=font(max(12, int(17 * scale))))
    axis_width = max(2, int(2 * scale))
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill=(180, 185, 195), width=axis_width)
    draw.line((width - margin_right, margin_top, width - margin_right, height - margin_bottom), fill=(180, 185, 195), width=axis_width)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill=(180, 185, 195), width=axis_width)

    psnr_points = [(x_of(i), y_psnr(value)) for i, value in enumerate(psnr)]
    ssim_points = [(x_of(i), y_ssim(value)) for i, value in enumerate(ssim)]
    line_width = max(5, int(5 * scale))
    dot = max(6, int(6 * scale))
    draw.line(psnr_points, fill=(90, 205, 255), width=line_width, joint="curve")
    draw.line(ssim_points, fill=(255, 190, 90), width=line_width, joint="curve")
    for i, step in enumerate(steps):
        x = x_of(i)
        draw.ellipse((x - dot, psnr_points[i][1] - dot, x + dot, psnr_points[i][1] + dot), fill=(90, 205, 255))
        draw.ellipse((x - dot, ssim_points[i][1] - dot, x + dot, ssim_points[i][1] + dot), fill=(255, 190, 90))
        draw.text((x - int(8 * scale), height - margin_bottom + int(18 * scale)), str(step), fill="white", font=font(max(12, int(17 * scale))))
    draw.text((margin_left, int(20 * scale)), "Baseline-back foreground similarity after re-noise + unconditional denoise", fill="white", font=font(max(14, int(25 * scale))))
    draw.text((int(12 * scale), int(48 * scale)), "PSNR (dB)", fill=(90, 205, 255), font=font(max(12, int(18 * scale))))
    draw.text((width - int(92 * scale), int(48 * scale)), "SSIM", fill=(255, 190, 90), font=font(max(12, int(18 * scale))))
    draw.text((margin_left + plot_w // 2 - int(190 * scale), height - int(42 * scale)), "start step (later = less injected noise)", fill="white", font=font(max(13, int(19 * scale))))
    canvas.save(output)
    print(f"[plot] metrics: {output}", flush=True)


def save_latent(path: Path, slat: SparseTensor, *, normalized: bool, extra: Mapping[str, Any]) -> None:
    atomic_save(path, {
        "format": FORMAT,
        "coords": slat.coords.detach().to(torch.int32).cpu().contiguous(),
        "feats": slat.feats.detach().float().cpu().contiguous(),
        "normalized": bool(normalized),
        **dict(extra),
    })


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_BASELINE / "canonical_1024.png")
    parser.add_argument("--camera", type=Path, default=DEFAULT_BASELINE / "global_camera.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=1_000_042)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--ssaa", type=int, default=1)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=0)
    parser.add_argument("--error-scale", type=float, default=0.25)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_cuda):
        raise RuntimeError(
            f"set CUDA_VISIBLE_DEVICES={args.physical_cuda}; got {visible!r}. "
            "Inside the process, use --device cuda:0."
        )
    if args.resolution != 1024:
        raise ValueError("this experiment is fixed to the native 1024 decoder")
    if args.error_scale <= 0:
        raise ValueError("--error-scale must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    if image.size != (1024, 1024):
        raise ValueError(f"expected canonical 1024 image, got {image.size}")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.time()

    run_manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "running",
        "semantics": (
            "Run one complete conditional 1024 baseline; normalize its final texture SLat as x0; "
            "for start step k use t=(12-k+1)/12 and x_t=(1-t)x0+"
            "(sigma_min+(1-sigma_min)t)epsilon; execute steps k..12 with zero image condition "
            "while retaining the fixed baseline shape concat."
        ),
        "physical_cuda": int(args.physical_cuda),
        "logical_device": str(device),
        "seed": int(args.seed),
        "noise_seed": int(args.noise_seed),
        "image": str(args.image.resolve()),
        "camera": camera,
        "results": [],
    }
    atomic_json(output / "summary.json", run_manifest)

    print("[load] initializing full Pixal3D pipeline on physical CUDA 4", flush=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=False)
    ss_params = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    shape_params = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    texture_params = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.0, 1.0),
        "rescale_t": 1.0,
    }
    torch.manual_seed(int(args.seed))
    print("[baseline] running complete C64 geometry and texture flow", flush=True)
    generated, (shape_slat, baseline_tex_raw, baseline_resolution) = pipeline.run(
        image,
        camera_params=dict(camera),
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=1_000_000,
    )
    if int(baseline_resolution) != 1024:
        raise RuntimeError(f"baseline decoder resolution is {baseline_resolution}, expected 1024")
    del generated
    empty_cuda()

    shape_mean = torch.as_tensor(pipeline.shape_slat_normalization["mean"], device=device)[None]
    shape_std = torch.as_tensor(pipeline.shape_slat_normalization["std"], device=device)[None]
    tex_mean = torch.as_tensor(pipeline.tex_slat_normalization["mean"], device=device)[None]
    tex_std = torch.as_tensor(pipeline.tex_slat_normalization["std"], device=device)[None]
    shape_normalized = (shape_slat - shape_mean) / shape_std
    baseline_x0 = (baseline_tex_raw - tex_mean) / tex_std
    if not torch.equal(shape_slat.coords, baseline_tex_raw.coords):
        raise RuntimeError("baseline shape and texture SLat coordinates are not aligned")

    generator = torch.Generator(device=device).manual_seed(int(args.noise_seed))
    epsilon = baseline_x0.replace(
        torch.randn(
            baseline_x0.feats.shape,
            generator=generator,
            device=device,
            dtype=baseline_x0.feats.dtype,
        )
    )
    save_latent(output / "baseline_shape_slat.pt", shape_slat, normalized=False, extra={"kind": "baseline_shape"})
    save_latent(output / "baseline_texture_slat.pt", baseline_tex_raw, normalized=False, extra={"kind": "baseline_texture_x0"})
    save_latent(output / "fixed_texture_noise.pt", epsilon, normalized=True, extra={"kind": "epsilon", "seed": args.noise_seed})

    cond_tex = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        shape_slat.coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=64,
    )
    tex_model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    schedule = sampler.timestep_schedule(STEPS, rescale_t=1.0)
    sigma_min = float(sampler.sigma_min)

    print("[decode] decoding fixed baseline geometry once", flush=True)
    shape_meshes, subs = pipeline.decode_shape_slat(shape_slat, 1024)
    geometry = shape_meshes[0]
    extrinsics, intrinsics = back_camera(float(camera["camera_angle_x"]), float(camera["distance"]))
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": int(args.resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": int(args.ssaa),
            "peel_layers": int(args.peel_layers),
            "face_chunk_size": int(args.face_chunk_size),
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)

    def decode_and_render(tex_raw: SparseTensor, directory: Path, render_seed: int) -> dict[str, np.ndarray]:
        tex_voxel = pipeline.decode_tex_slat(tex_raw, subs)[0]
        mesh = make_mesh(geometry, tex_voxel, 1024, pipeline.pbr_attr_layout)
        torch.cuda.manual_seed_all(int(render_seed))
        rendered = renderer.render(
            mesh,
            extrinsics.to(device),
            intrinsics.to(device),
            envmap=envmap,
            use_envmap_bg=False,
        )
        saved = save_render(rendered, directory)
        del rendered, mesh, tex_voxel
        empty_cuda()
        return saved

    print("[render] baseline yaw=180", flush=True)
    baseline_render = decode_and_render(baseline_tex_raw, output / "baseline_back", args.seed + 100_180)
    reference = baseline_render["shaded"]
    foreground = baseline_render["mask"]
    if foreground.ndim == 3:
        foreground = foreground[..., 0]
    foreground = foreground > 0.5
    baseline_fg_path = output / "baseline_back" / "foreground_mask.png"
    array_to_image(foreground.astype(np.float32)).save(baseline_fg_path)

    results: list[dict[str, Any]] = []
    for start_step in range(1, STEPS + 1):
        variant_started = time.time()
        start_index = start_step - 1
        start_t = float(schedule[start_index])
        noise_scale = sigma_min + (1.0 - sigma_min) * start_t
        state = baseline_x0.replace(
            (1.0 - start_t) * baseline_x0.feats + noise_scale * epsilon.feats
        )
        print(
            f"[variant {start_step:02d}/12] t={start_t:.6f}, "
            f"noise_scale={noise_scale:.6f}, uncond_steps={STEPS - start_index}",
            flush=True,
        )
        for step_index in range(start_index, STEPS):
            t_now = float(schedule[step_index])
            t_next = float(schedule[step_index + 1])
            out = sampler.sample_once(
                tex_model,
                state,
                t_now,
                t_next,
                concat_cond=shape_normalized,
                **cond_tex,
                guidance_strength=0.0,
                guidance_rescale=0.0,
                guidance_interval=(0.0, 1.0),
            )
            state = out.pred_x_prev
        tex_raw = state * tex_std + tex_mean
        variant_dir = output / f"start_step_{start_step:02d}"
        latent_path = variant_dir / "final_texture_slat.pt"
        save_latent(
            latent_path,
            tex_raw,
            normalized=False,
            extra={
                "kind": "renoise_then_unconditional_texture",
                "start_step": start_step,
                "start_t": start_t,
                "unconditional_steps": STEPS - start_index,
            },
        )
        rendered = decode_and_render(tex_raw, variant_dir / "back", args.seed + 100_180)
        values = metrics(reference, rendered["shaded"], foreground)
        error, heatmap = error_heatmap(reference, rendered["shaded"], foreground, args.error_scale)
        error_npy = variant_dir / "back" / "absolute_error_float32.npy"
        np.save(error_npy, error)
        error_png = variant_dir / "back" / "error_map.png"
        array_to_image(heatmap).save(error_png)
        record = {
            "start_step": start_step,
            "start_t": start_t,
            "noise_clean_weight": 1.0 - start_t,
            "noise_epsilon_weight": noise_scale,
            "unconditional_steps": STEPS - start_index,
            "shape_concat_retained": True,
            "image_condition": "negative/zero image condition",
            "texture_slat": str(latent_path.resolve()),
            "back_render": str((variant_dir / "back" / "render.png").resolve()),
            "error_map": str(error_png.resolve()),
            "error_float32": str(error_npy.resolve()),
            "seconds": time.time() - variant_started,
            **values,
        }
        atomic_json(variant_dir / "metrics.json", record)
        results.append(record)
        run_manifest["results"] = results
        atomic_json(output / "summary.json", run_manifest)
        print(
            f"[metric {start_step:02d}] FG PSNR={values['foreground_psnr_db']:.4f} "
            f"SSIM={values['foreground_ssim']:.6f} MAE={values['foreground_mae']:.6f}",
            flush=True,
        )
        del state, tex_raw, rendered, error, heatmap
        empty_cuda()

    render_grid = output / "back_renders_3x4.png"
    error_grid = output / "back_error_maps_3x4.png"
    metric_plot = output / "foreground_metrics.png"
    make_grid(results, render_grid, "back_render", "back renders")
    make_grid(results, error_grid, "error_map", "back error maps")
    make_metric_plot(results, metric_plot)

    psnr_values = np.asarray([row["foreground_psnr_db"] for row in results], dtype=np.float64)
    ssim_values = np.asarray([row["foreground_ssim"] for row in results], dtype=np.float64)
    run_manifest.update({
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "texture_steps": STEPS,
        "texture_rescale_t": 1.0,
        "sigma_min": sigma_min,
        "timestep_schedule": schedule,
        "forward_noise_formula": "x_t=(1-t)*x0+(sigma_min+(1-sigma_min)*t)*epsilon",
        "baseline_shape_sha256": tensor_sha256(shape_slat.feats),
        "baseline_texture_x0_sha256": tensor_sha256(baseline_x0.feats),
        "fixed_epsilon_sha256": tensor_sha256(epsilon.feats),
        "baseline_back_render": str((output / "baseline_back" / "render.png").resolve()),
        "baseline_foreground_mask": str(baseline_fg_path.resolve()),
        "error_map_scale": {"metric": "mean absolute RGB error", "maximum": args.error_scale},
        "render_grid_3x4": str(render_grid.resolve()),
        "error_grid_3x4": str(error_grid.resolve()),
        "foreground_metric_plot": str(metric_plot.resolve()),
        "observed_monotonic_non_decreasing": {
            "foreground_psnr": bool(np.all(np.diff(psnr_values) >= -1e-9)),
            "foreground_ssim": bool(np.all(np.diff(ssim_values) >= -1e-9)),
        },
        "results": results,
    })
    atomic_json(output / "summary.json", run_manifest)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
