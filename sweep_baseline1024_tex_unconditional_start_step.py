#!/usr/bin/env python3
"""Sweep the first image-unconditional C64 texture-flow step from 1 through 12."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from inference import MODEL_PATH, init_pipeline
from render_pixal3d_raw_ovoxel import load_envmap, render_static_ovoxel, save_render_outputs


BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
DEFAULT_OUTPUT = Path("outputs/baseline1024_tex_unconditional_start_step_sweep_cuda4")
STEPS = 12


def timestep_schedule(steps: int = STEPS, rescale_t: float = 3.0) -> list[float]:
    """Exact FlowEuler schedule, including the final t=0 integration endpoint."""
    linear = np.linspace(1.0, 0.0, steps + 1)
    return [float(rescale_t * t / (1.0 + (rescale_t - 1.0) * t)) for t in linear]


def interval_for_first_unconditional_step(
    sampler: object,
    first_step: int,
    rescale_t: float = 3.0,
) -> tuple[float, float]:
    """Map a 1-indexed Euler step switch to the sampler's descending t interval."""
    if not 1 <= first_step <= STEPS:
        raise ValueError(f"first_step must be in [1,{STEPS}]")
    schedule = sampler.timestep_schedule(STEPS, rescale_t)
    if first_step == 1:
        upper = 1.0
    else:
        # Steps before the midpoint are outside the strength=0 interval and
        # therefore use the mixin's strength=1 conditional fast path.
        upper = (float(schedule[first_step - 2]) + float(schedule[first_step - 1])) / 2.0
    return (0.0, upper)


def metric(reference: np.ndarray, prediction: np.ndarray, foreground: np.ndarray) -> dict[str, float]:
    difference = prediction - reference
    full_mse = float(np.mean(difference * difference))
    fg_mse = float(np.mean(difference[foreground] ** 2))
    _, ssim_map = structural_similarity(reference, prediction, data_range=1.0, channel_axis=2, full=True)
    return {
        "psnr_db": float(10.0 * math.log10(1.0 / max(full_mse, 1e-12))),
        "foreground_psnr_db": float(10.0 * math.log10(1.0 / max(fg_mse, 1e-12))),
        "ssim": float(np.mean(ssim_map)),
        "foreground_ssim": float(np.mean(ssim_map[foreground])),
        "mae": float(np.mean(np.abs(difference))),
        "foreground_mae": float(np.mean(np.abs(difference[foreground]))),
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def write_grid(records: list[dict], output_dir: Path, rescale_t: float) -> Path:
    """Rebuild the requested 3x4 sheet from existing renders, with exact t labels."""
    schedule = timestep_schedule(rescale_t=rescale_t)
    thumb = 512
    label_h = 108
    margin = 12
    columns, rows = 3, 4
    sheet = Image.new(
        "RGB",
        (columns * thumb + (columns + 1) * margin, rows * (thumb + label_h) + (rows + 1) * margin),
        (18, 21, 27),
    )
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = margin + column * (thumb + margin)
        y = margin + row * (thumb + label_h + margin)
        step = int(record["first_unconditional_step"])
        t_now, t_next = schedule[step - 1 : step + 1]
        record["step_t"] = t_now
        record["step_t_next"] = t_next
        record["model_t"] = 1000.0 * t_now
        draw.text((x + 6, y + 4), f"UNCOND FROM STEP {step}", fill="white", font=font(22))
        draw.text(
            (x + 6, y + 34),
            f"t {t_now:.6f} -> {t_next:.6f}  (model t {1000.0 * t_now:.2f})",
            fill=(255, 208, 112),
            font=font(16),
        )
        draw.text(
            (x + 6, y + 64),
            f"cond {step-1}/12 · PSNR {record['psnr_db']:.2f} · SSIM {record['ssim']:.3f}",
            fill=(112, 205, 255),
            font=font(17),
        )
        with Image.open(record["render"]) as rendered:
            tile = rendered.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + label_h))
    sheet_path = output_dir / "unconditional_start_step_grid_3x4.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=BASELINE / "canonical_1024.png")
    parser.add_argument("--mask", type=Path, default=BASELINE / "raw_ovoxel_render/alpha.png")
    parser.add_argument("--camera", type=Path, default=BASELINE / "global_camera.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument("--grid-only", action="store_true", help="rebuild labels from existing summary/renders")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.grid_only:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rescale_t = float(summary.get("texture_rescale_t", args.texture_rescale_t))
        sheet_path = write_grid(summary["results"], args.output_dir, rescale_t)
        schedule = timestep_schedule(rescale_t=rescale_t)
        summary["texture_rescale_t"] = rescale_t
        summary["timestep_schedule"] = schedule
        summary["model_timestep_schedule"] = [1000.0 * t for t in schedule[:-1]]
        summary["grid"] = str(sheet_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[done] {sheet_path}")
        return 0

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    image = Image.open(args.image).convert("RGB")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    reference = np.asarray(image.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    foreground = np.asarray(
        Image.open(args.mask).convert("L").resize((args.resolution, args.resolution), Image.Resampling.NEAREST),
        dtype=np.float32,
    ) > 127.5

    pipeline = init_pipeline(str(MODEL_PATH), device=str(device), low_vram=False)
    envmap = load_envmap("studio", device=device)
    ss_params = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    shape_params = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    print("[sweep] bootstrapping one fixed C64 shape latent", flush=True)
    bootstrap_meshes, (shape_slat, bootstrap_tex, bootstrap_resolution) = pipeline.run(
        image,
        camera_params=dict(camera),
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params={
            "steps": STEPS,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "guidance_interval": (0.0, 1.0),
            "rescale_t": float(args.texture_rescale_t),
        },
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=1_000_000,
    )
    if int(bootstrap_resolution) != 1024:
        raise RuntimeError("bootstrap did not produce a 1024 baseline")
    del bootstrap_meshes, bootstrap_tex
    torch.cuda.empty_cache()

    # Compute the positive/zero image pair once for the exact fixed support.
    cond_tex = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        shape_slat.coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=64,
    )
    shape_mean = torch.as_tensor(pipeline.shape_slat_normalization["mean"], device=device)[None]
    shape_std = torch.as_tensor(pipeline.shape_slat_normalization["std"], device=device)[None]
    shape_normalized = (shape_slat - shape_mean) / shape_std
    tex_model = pipeline.models["tex_slat_flow_model_1024"]
    noise_channels = int(tex_model.in_channels) - int(shape_normalized.feats.shape[1])
    noise_seed = int(args.seed) + 1_000_000
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    fixed_noise = shape_normalized.replace(
        torch.randn(
            shape_normalized.feats.shape[0],
            noise_channels,
            device=device,
            generator=generator,
        )
    )
    tex_mean = torch.as_tensor(pipeline.tex_slat_normalization["mean"], device=device)[None]
    tex_std = torch.as_tensor(pipeline.tex_slat_normalization["std"], device=device)[None]
    fixed_shape_hash = __import__("hashlib").sha256(
        shape_slat.coords.detach().cpu().contiguous().numpy().tobytes()
        + shape_slat.feats.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    fixed_noise_hash = __import__("hashlib").sha256(
        fixed_noise.feats.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    records = []
    for first_unconditional in range(1, STEPS + 1):
        variant_dir = args.output_dir / f"unconditional_from_step_{first_unconditional:02d}"
        render_path = variant_dir / "render.png"
        interval = interval_for_first_unconditional_step(
            pipeline.tex_slat_sampler,
            first_unconditional,
            float(args.texture_rescale_t),
        )
        started = time.time()
        if not render_path.is_file():
            tex_params = {
                "steps": STEPS,
                "guidance_strength": 0.0,
                "guidance_rescale": 0.0,
                "guidance_interval": interval,
                "rescale_t": float(args.texture_rescale_t),
            }
            print(
                f"[sweep] unconditional_from_step={first_unconditional} "
                f"conditional_steps={first_unconditional - 1} interval={interval}",
                flush=True,
            )
            texture_normalized = pipeline.tex_slat_sampler.sample(
                tex_model,
                fixed_noise.replace(fixed_noise.feats.clone()),
                concat_cond=shape_normalized,
                **cond_tex,
                **tex_params,
                verbose=True,
                tqdm_desc=f"Texture: unconditional from step {first_unconditional}",
            ).samples
            texture_slat = texture_normalized * tex_std + tex_mean
            meshes = pipeline.decode_latent(shape_slat, texture_slat, 1024)
            if len(meshes) != 1:
                raise RuntimeError("unexpected decoder batch")
            renders = render_static_ovoxel(
                meshes[0],
                camera_angle_x=float(camera["camera_angle_x"]),
                distance=float(camera["distance"]),
                resolution=int(args.resolution),
                envmap=envmap,
                ssaa=1,
                peel_layers=8,
                face_chunk_size=4_000_000,
                use_envmap_bg=False,
                verbose=False,
            )
            save_render_outputs(renders, variant_dir)
            del meshes, renders, texture_normalized, texture_slat
            torch.cuda.empty_cache()
        prediction = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
        values = metric(reference, prediction, foreground)
        record = {
            "first_unconditional_step": first_unconditional,
            "conditional_steps": list(range(1, first_unconditional)),
            "unconditional_steps": list(range(first_unconditional, STEPS + 1)),
            "guidance_strength": 0.0,
            "guidance_interval_t": list(interval),
            "shape_concat_retained": True,
            "fixed_shape_sha256": fixed_shape_hash,
            "fixed_texture_noise_sha256": fixed_noise_hash,
            "fixed_texture_noise_seed": noise_seed,
            "render": str(render_path.resolve()),
            "seconds": time.time() - started,
            **values,
        }
        records.append(record)
        (variant_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    # Requested layout: three columns by four rows.
    sheet_path = write_grid(records, args.output_dir, float(args.texture_rescale_t))
    schedule = timestep_schedule(rescale_t=float(args.texture_rescale_t))
    summary = {
        "status": "complete",
        "semantics": "variant k uses conditional image for steps 1..k-1 and zero image condition for steps k..12; shape concat retained",
        "seed": int(args.seed),
        "texture_rescale_t": float(args.texture_rescale_t),
        "fixed_shape_sha256": fixed_shape_hash,
        "fixed_texture_noise_sha256": fixed_noise_hash,
        "fixed_texture_noise_seed": noise_seed,
        "timestep_schedule": schedule,
        "model_timestep_schedule": [1000.0 * t for t in schedule[:-1]],
        "results": records,
        "grid": str(sheet_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
