#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate exact crop cameras derived from one global MoGe camera.

Experiment
----------
1. Canonically preprocess one input image into 512/1024/4096 images.
2. Estimate ONLY the global camera from canonical_1024 with MoGe.
3. Run the unmodified Pixal3D 1024 cascade once:
       sparse structure -> shape 512 -> shape 1024 -> texture 1024 -> decode
4. Render the same decoded global model:
   a) once at 4096 with the original global camera;
   b) once per 1024 crop with analytically derived off-axis crop intrinsics.
5. Report:
   - global reference vs global render;
   - tile reference vs exact crop from the global render;
   - tile reference vs independently rendered derived crop camera;
   - derived crop-camera render vs exact global-render crop.

The last metric isolates camera derivation/rendering error. A correct crop camera
should closely reproduce the corresponding crop from the full global render.

This script expects to be placed in the Pixal3D-exp repository root beside:
    inference.py
    pixal3d_directory_texture_eval.py
    render_pixal3d_cache_no_uv.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw

from inference import (
    MODEL_PATH,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d_directory_texture_eval import save_to_glb_cache
import render_pixal3d_cache_no_uv as no_uv_renderer


CANONICAL_SIZE = 4096
PIPELINE_IMAGE_SIZE = 1024
TILE_SIZE = 1024
TILE_STRIDE = 512


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = {key for row in rows for key in row.keys()}
    preferred = [
        "tile_id",
        "box",
        "status",
        "derived_fov_deg",
        "derived_fx",
        "derived_fy",
        "derived_cx",
        "derived_cy",
        "blender_shift_x",
        "blender_shift_y",
        "reference_vs_global_crop_psnr",
        "reference_vs_global_crop_ssim",
        "reference_vs_global_crop_lpips",
        "reference_vs_derived_camera_psnr",
        "reference_vs_derived_camera_ssim",
        "reference_vs_derived_camera_lpips",
        "derived_vs_global_crop_psnr",
        "derived_vs_global_crop_ssim",
        "derived_vs_global_crop_lpips",
        "reference_path",
        "global_crop_path",
        "derived_render_path",
        "comparison_path",
        "error",
    ]
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(key for key in keys if key not in fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def composite_on_black(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def load_black_image(path: Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    with Image.open(path) as image:
        result = composite_on_black(image)
    if size is not None and result.size != size:
        result = result.resize(size, Image.Resampling.LANCZOS)
    return result


def save_triptych(
    reference: Image.Image,
    global_crop: Image.Image,
    derived: Image.Image,
    output_path: Path,
    title: str,
) -> None:
    images = [
        ("reference", reference.convert("RGB")),
        ("global render crop", global_crop.convert("RGB")),
        ("derived crop camera", derived.convert("RGB")),
    ]
    width, height = reference.size
    header = 58
    canvas = Image.new("RGB", (width * 3, height + header), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(images):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 12, 32), name, fill=(255, 255, 255))
    draw.text((12, 8), title, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_tile_ids(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--tile-ids must contain at least one integer")
    return sorted(set(values))


def tile_layout(
    canonical_size: int = CANONICAL_SIZE,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    starts = list(range(0, canonical_size - tile_size + 1, stride))
    if not starts or starts[-1] != canonical_size - tile_size:
        raise ValueError(
            f"tile layout does not terminate at the image edge: "
            f"canonical={canonical_size} tile={tile_size} stride={stride}"
        )
    return [
        (x0, y0, x0 + tile_size, y0 + tile_size)
        for y0 in starts
        for x0 in starts
    ]


# -----------------------------------------------------------------------------
# Camera derivation
# -----------------------------------------------------------------------------

def focal_pixels_from_fov(fov_x: float, image_width: int) -> float:
    return float(image_width) / (2.0 * math.tan(float(fov_x) / 2.0))


def derive_crop_camera(
    global_camera: Mapping[str, float],
    box: Sequence[int],
    *,
    full_width: int = CANONICAL_SIZE,
    full_height: int = CANONICAL_SIZE,
    output_width: int = TILE_SIZE,
    output_height: int = TILE_SIZE,
    blender_shift_y_sign: int = 1,
) -> Dict[str, float]:
    """
    Apply an exact pinhole crop + resize transform to the global camera.

    Global principal point is assumed to be at the canonical image center,
    matching Pixal3D's current camera representation.

    Pixel intrinsics after crop and resize:
        fx' = sx * fx
        fy' = sy * fy
        cx' = sx * (cx - x0)
        cy' = sy * (cy - y0)

    Blender, HORIZONTAL sensor fit, square pixels:
        cx = W/2 - shift_x * W
        cy = H/2 + shift_y * W

    Therefore:
        shift_x = (W/2 - cx) / W
        shift_y = (cy - H/2) / W
    """
    if blender_shift_y_sign not in (-1, 1):
        raise ValueError("blender_shift_y_sign must be +1 or -1")

    x0, y0, x1, y1 = (int(value) for value in box)
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"invalid crop box: {tuple(box)}")

    scale_x = float(output_width) / float(crop_width)
    scale_y = float(output_height) / float(crop_height)

    global_fov_x = float(global_camera["camera_angle_x"])
    global_fx = focal_pixels_from_fov(global_fov_x, full_width)

    # Pixal3D currently assumes square images/pixels and derives only horizontal FOV.
    global_fy = global_fx
    global_cx = float(full_width) / 2.0
    global_cy = float(full_height) / 2.0

    tile_fx = global_fx * scale_x
    tile_fy = global_fy * scale_y
    tile_cx = (global_cx - float(x0)) * scale_x
    tile_cy = (global_cy - float(y0)) * scale_y

    tile_fov_x = 2.0 * math.atan(float(output_width) / (2.0 * tile_fx))
    tile_fov_y = 2.0 * math.atan(float(output_height) / (2.0 * tile_fy))

    shift_x = (float(output_width) / 2.0 - tile_cx) / float(output_width)
    shift_y = (
        (tile_cy - float(output_height) / 2.0)
        / float(output_width)
        * float(blender_shift_y_sign)
    )

    return {
        "camera_angle_x": float(tile_fov_x),
        "camera_angle_y": float(tile_fov_y),
        "distance": float(global_camera["distance"]),
        "mesh_scale": float(global_camera["mesh_scale"]),
        "fx": float(tile_fx),
        "fy": float(tile_fy),
        "cx": float(tile_cx),
        "cy": float(tile_cy),
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
        "crop_scale_x": float(scale_x),
        "crop_scale_y": float(scale_y),
        "full_fx": float(global_fx),
        "full_fy": float(global_fy),
        "full_cx": float(global_cx),
        "full_cy": float(global_cy),
    }


# -----------------------------------------------------------------------------
# Original Pixal3D 1024 generation
# -----------------------------------------------------------------------------

def sampler_overrides(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ss = {
        "steps": int(args.ss_steps),
        "guidance_strength": float(args.ss_guidance_strength),
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    shape = {
        "steps": int(args.shape_steps),
        "guidance_strength": float(args.shape_guidance_strength),
        "guidance_rescale": float(args.shape_guidance_rescale),
        "rescale_t": float(args.shape_rescale_t),
    }
    texture = {
        "steps": int(args.texture_steps),
        "guidance_strength": float(args.texture_guidance_strength),
        "guidance_rescale": float(args.texture_guidance_rescale),
        "rescale_t": float(args.texture_rescale_t),
    }
    return ss, shape, texture


def estimate_global_camera(
    image_1024_path: Path,
    args: argparse.Namespace,
) -> Dict[str, float]:
    print(f"[MoGe] loading global camera model: {args.moge_model_path}")
    model = load_moge_model(device="cuda", model_name=str(args.moge_model_path))
    try:
        camera = get_camera_params_wild_moge(
            str(image_1024_path),
            model,
            device="cuda",
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            image_resolution=int(args.camera_image_resolution),
        )
    finally:
        model.cpu()
        del model
        empty_cuda_cache()

    result = {
        "camera_angle_x": float(camera["camera_angle_x"]),
        "distance": float(camera["distance"]),
        "mesh_scale": float(camera["mesh_scale"]),
    }
    print(
        "[global-camera] "
        f"fov={math.degrees(result['camera_angle_x']):.6f} deg "
        f"distance={result['distance']:.8f} "
        f"mesh_scale={result['mesh_scale']:.8f}"
    )
    return result


def generate_global_cache(
    *,
    image_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path, Dict[str, float], Dict[str, Any]]:
    """
    Run the repository's original pipeline.run(..., pipeline_type='1024_cascade')
    and save the raw undecimated decoder tensors to a postprocess cache.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_4096_path = output_dir / "canonical_4096.png"
    canonical_1024_path = output_dir / "canonical_1024.png"
    canonical_512_path = output_dir / "canonical_512.png"
    reference_4096_path = output_dir / "reference_global_4096.png"
    cache_dir = output_dir / "postprocess_cache"

    if (
        args.reuse_generation
        and (cache_dir / "READY").is_file()
        and canonical_4096_path.is_file()
        and canonical_1024_path.is_file()
        and (output_dir / "generation_summary.json").is_file()
    ):
        print(f"[reuse] generation cache: {cache_dir}")
        summary = json.loads((output_dir / "generation_summary.json").read_text(encoding="utf-8"))
        camera = {
            key: float(summary["camera"][key])
            for key in ("camera_angle_x", "distance", "mesh_scale")
        }
        return (
            cache_dir,
            canonical_4096_path,
            reference_4096_path,
            camera,
            summary,
        )

    print(f"[Pipeline] loading: {args.model_path}")
    pipeline = init_pipeline(
        str(args.model_path),
        device="cuda",
        low_vram=bool(args.low_vram),
    )

    print(f"[preprocess] {image_path}")
    canonical = pipeline.preprocess_canonical_images(Image.open(image_path))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    expected_sizes = {
        "image_4096": (4096, 4096),
        "image_1024": (1024, 1024),
        "image_512": (512, 512),
    }
    actual_sizes = {
        "image_4096": image_4096.size,
        "image_1024": image_1024.size,
        "image_512": image_512.size,
    }
    if actual_sizes != expected_sizes:
        raise RuntimeError(
            f"unexpected canonical image sizes: {actual_sizes}; "
            f"expected {expected_sizes}"
        )

    image_4096.save(canonical_4096_path)
    image_1024.save(canonical_1024_path)
    image_512.save(canonical_512_path)
    composite_on_black(image_4096).save(reference_4096_path)

    camera = estimate_global_camera(canonical_1024_path, args)
    ss_params, shape_params, texture_params = sampler_overrides(args)

    print("[generation] original Pixal3D 1024_cascade")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    started = time.perf_counter()
    mesh_list, latent_payload = pipeline.run(
        image_1024,
        camera_params=camera,
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    generation_seconds = time.perf_counter() - started

    if not mesh_list:
        raise RuntimeError("pipeline.run returned no mesh")
    mesh = mesh_list[0]
    shape_slat, tex_slat, decode_resolution = latent_payload
    decode_resolution = int(decode_resolution)

    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])
    print(
        f"[decode] resolution={decode_resolution} "
        f"vertices={vertices:,} faces={faces:,} "
        f"seconds={generation_seconds:.3f}"
    )

    export_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "grid_size": int(decode_resolution),
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        # The no-UV evaluator uses the raw tensors; these are retained here.
        "decimation_target": int(faces),
        "texture_size": int(args.texture_size),
        "remesh": False,
        "use_tqdm": True,
        "verbose": False,
    }

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(f"[cache] saving raw postprocess tensors: {cache_dir}")
    cache_manifest = save_to_glb_cache(
        cache_dir,
        export_kwargs,
        extra_metadata={
            "camera_params": dict(camera),
            "pipeline_resolution": 1024,
            "actual_grid_resolution": int(decode_resolution),
            "seed": int(args.seed),
            "decoder_vertices": vertices,
            "decoder_faces": faces,
            "experiment": "global_derived_crop_camera_eval",
            "source_image": str(image_path),
        },
    )

    summary = {
        "camera": camera,
        "pipeline_type": "1024_cascade",
        "seed": int(args.seed),
        "generation_seconds": float(generation_seconds),
        "decode_resolution": int(decode_resolution),
        "decoder_vertices": vertices,
        "decoder_faces": faces,
        "cache_dir": str(cache_dir),
        "cache_manifest": cache_manifest,
        "sampler_params": {
            "sparse_structure": ss_params,
            "shape": shape_params,
            "texture": texture_params,
        },
    }
    atomic_json(output_dir / "generation_summary.json", summary)

    del mesh, mesh_list, shape_slat, tex_slat, latent_payload
    del pipeline
    empty_cuda_cache()

    return cache_dir, canonical_4096_path, reference_4096_path, camera, summary


# -----------------------------------------------------------------------------
# Global render/package
# -----------------------------------------------------------------------------

def run_global_renderer(
    *,
    repository_dir: Path,
    cache_dir: Path,
    package_dir: Path,
    output_dir: Path,
    reference_4096_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    renderer_script = repository_dir / "render_pixal3d_cache_no_uv.py"
    if not renderer_script.is_file():
        raise FileNotFoundError(renderer_script)

    command = [
        sys.executable,
        str(renderer_script),
        "--cache-dir",
        str(cache_dir),
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(output_dir),
        "--reference-image",
        str(reference_4096_path),
        "--lights",
        str(args.light),
        "--engine",
        str(args.render_engine),
        "--material-mode",
        "pbr",
        "--base-color-space",
        "srgb",
        "--render-resolution",
        str(int(args.global_render_resolution)),
        "--metric-resolution",
        str(int(args.metric_resolution)),
        "--samples",
        str(int(args.global_blender_samples)),
        "--lpips-net",
        str(args.lpips_net),
        "--metric-device",
        str(args.metric_device),
        "--blender",
        str(args.blender),
        "--faces-per-shard",
        str(int(args.faces_per_shard)),
        "--copy-rows-per-chunk",
        str(int(args.copy_rows_per_chunk)),
        "--alignment-samples",
        str(int(args.alignment_samples)),
    ]
    if args.render_max_faces > 0:
        command.extend(["--max-faces", str(int(args.render_max_faces))])
    if args.skip_lpips:
        command.append("--skip-lpips")
    if args.overwrite_package or not (package_dir / "READY").is_file():
        command.append("--overwrite-package")
    else:
        command.append("--render-only")
    if args.overwrite_renders:
        command.append("--overwrite-renders")

    print("[global-render-command] " + shlex.join(command))
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"global renderer failed with exit code {process.returncode}")

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    successful = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") == "success"
    ]
    if not successful:
        raise RuntimeError(f"no successful global render metric in {metrics_path}")
    row = successful[0]
    render_path = Path(str(row["render_png"])).resolve()
    if not render_path.is_file():
        raise FileNotFoundError(render_path)
    return {
        "metrics_payload": payload,
        "metrics_row": row,
        "render_path": render_path,
        "package_dir": package_dir,
    }


# -----------------------------------------------------------------------------
# Per-job off-axis camera renderer
# -----------------------------------------------------------------------------

def patched_blender_helper_source(base_source: str) -> str:
    """
    Patch the repository's Blender helper in memory so each job can provide:
        camera_angle_x, distance, mesh_scale, shift_x, shift_y

    No repository source file is modified.
    """
    source = str(base_source)

    old_signature = "def create_aligned_camera(scene, distance, fov_rad):"
    new_signature = (
        "def create_aligned_camera(scene, distance, fov_rad, "
        "shift_x=0.0, shift_y=0.0):"
    )
    if source.count(old_signature) != 1:
        raise RuntimeError("unexpected renderer helper: camera function signature not found")
    source = source.replace(old_signature, new_signature, 1)

    old_lens = (
        '    camera_data.lens = 16.0 / math.tan(float(fov_rad) / 2.0)\n'
        '    camera_data.clip_start = 0.01'
    )
    new_lens = (
        '    camera_data.lens = 16.0 / math.tan(float(fov_rad) / 2.0)\n'
        '    camera_data.shift_x = float(shift_x)\n'
        '    camera_data.shift_y = float(shift_y)\n'
        '    camera_data.clip_start = 0.01'
    )
    if source.count(old_lens) != 1:
        raise RuntimeError("unexpected renderer helper: lens block not found")
    source = source.replace(old_lens, new_lens, 1)

    old_camera_block = """    camera = manifest["camera"]
    fov_rad = float(camera["camera_angle_x"])
    distance = float(camera["distance"]) * float(camera.get("mesh_scale", 1.0))
    create_aligned_camera(scene, distance, fov_rad)
    log("camera fov=%.12f distance=%.12f" % (fov_rad, distance))
"""
    new_camera_block = """    default_camera = manifest["camera"]
    fov_rad = float(default_camera["camera_angle_x"])
    distance = float(default_camera["distance"]) * float(default_camera.get("mesh_scale", 1.0))
    camera_object = create_aligned_camera(
        scene,
        distance,
        fov_rad,
        float(default_camera.get("shift_x", 0.0)),
        float(default_camera.get("shift_y", 0.0)),
    )
    log("default camera fov=%.12f distance=%.12f shift=(%.8f, %.8f)" % (
        fov_rad,
        distance,
        camera_object.data.shift_x,
        camera_object.data.shift_y,
    ))
"""
    if source.count(old_camera_block) != 1:
        raise RuntimeError("unexpected renderer helper: main camera block not found")
    source = source.replace(old_camera_block, new_camera_block, 1)

    old_loop_start = """        try:
            clear_lights()
"""
    new_loop_start = """        try:
            job_camera = job.get("camera", default_camera)
            fov_rad = float(job_camera["camera_angle_x"])
            distance = float(job_camera["distance"]) * float(job_camera.get("mesh_scale", 1.0))
            camera_object.matrix_world = Matrix((
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, -float(distance)),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ))
            camera_object.data.lens = 16.0 / math.tan(float(fov_rad) / 2.0)
            camera_object.data.shift_x = float(job_camera.get("shift_x", 0.0))
            camera_object.data.shift_y = float(job_camera.get("shift_y", 0.0))
            log("job camera fov=%.12f distance=%.12f shift=(%.8f, %.8f)" % (
                fov_rad,
                distance,
                camera_object.data.shift_x,
                camera_object.data.shift_y,
            ))
            clear_lights()
"""
    if source.count(old_loop_start) != 1:
        raise RuntimeError("unexpected renderer helper: render loop start not found")
    source = source.replace(old_loop_start, new_loop_start, 1)

    return source


def run_tile_camera_renders(
    *,
    package_dir: Path,
    output_dir: Path,
    jobs: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    original_source = no_uv_renderer.BLENDER_HELPER_SOURCE
    no_uv_renderer.BLENDER_HELPER_SOURCE = patched_blender_helper_source(
        original_source
    )
    try:
        render_args = argparse.Namespace(
            package_dir=package_dir.resolve(),
            output_dir=output_dir.resolve(),
            blender=str(args.blender),
            engine=str(args.render_engine),
            render_resolution=int(args.tile_render_resolution),
            samples=int(args.blender_samples),
            material_mode="pbr",
            base_color_space="srgb",
            flat_shading=False,
            use_alpha=False,
        )
        return no_uv_renderer.run_blender(render_args, jobs)
    finally:
        no_uv_renderer.BLENDER_HELPER_SOURCE = original_source


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def image_tensor(image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    rgb = image.convert("RGB")
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class MetricEvaluator:
    def __init__(
        self,
        *,
        metric_resolution: int,
        lpips_network: str,
        device: str,
        skip_lpips: bool,
    ) -> None:
        self.size = (int(metric_resolution), int(metric_resolution))
        self.skip_lpips = bool(skip_lpips)
        self.device = torch.device(
            device
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        self.lpips = (
            None
            if self.skip_lpips
            else no_uv_renderer.LPIPSEvaluator(lpips_network, self.device)
        )

    def evaluate(
        self,
        reference: Image.Image,
        prediction: Image.Image,
    ) -> Dict[str, Optional[float]]:
        ref = image_tensor(reference, self.size)
        pred = image_tensor(prediction, self.size)
        return {
            "psnr": float(no_uv_renderer.psnr_metric(ref, pred)),
            "ssim": float(no_uv_renderer.ssim_metric(ref, pred)),
            "lpips": (
                None
                if self.lpips is None
                else float(self.lpips.evaluate(ref, pred))
            ),
        }


def prefixed_metrics(
    prefix: str,
    values: Mapping[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    return {
        f"{prefix}_psnr": values["psnr"],
        f"{prefix}_ssim": values["ssim"],
        f"{prefix}_lpips": values["lpips"],
    }


# -----------------------------------------------------------------------------
# Complete experiment
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    repository_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    global_model_dir = output_dir / "global_model"
    (
        cache_dir,
        canonical_4096_path,
        reference_4096_path,
        global_camera,
        generation_summary,
    ) = generate_global_cache(
        image_path=image_path,
        output_dir=global_model_dir,
        args=args,
    )

    package_dir = global_model_dir / "no_uv_full_package"
    global_eval_dir = output_dir / "global_eval"
    global_render_result = run_global_renderer(
        repository_dir=repository_dir,
        cache_dir=cache_dir,
        package_dir=package_dir,
        output_dir=global_eval_dir,
        reference_4096_path=reference_4096_path,
        args=args,
    )
    global_render_path: Path = global_render_result["render_path"]
    global_row = dict(global_render_result["metrics_row"])

    print(
        "[global-metrics] "
        f"PSNR={global_row.get('psnr_db')} "
        f"SSIM={global_row.get('ssim')} "
        f"LPIPS={global_row.get('lpips')}"
    )

    canonical_4096 = load_black_image(canonical_4096_path)
    global_render_4096 = load_black_image(
        global_render_path,
        (int(args.global_render_resolution), int(args.global_render_resolution)),
    )

    # The crop boxes are defined in the canonical 4096 coordinate system.
    # Require the full render to have the same pixel grid for an exact oracle crop.
    if global_render_4096.size != (CANONICAL_SIZE, CANONICAL_SIZE):
        global_render_4096 = global_render_4096.resize(
            (CANONICAL_SIZE, CANONICAL_SIZE),
            Image.Resampling.LANCZOS,
        )

    boxes = tile_layout(
        canonical_size=CANONICAL_SIZE,
        tile_size=int(args.tile_size),
        stride=int(args.tile_stride),
    )
    selected_ids = parse_tile_ids(args.tile_ids)
    invalid_ids = [tile_id for tile_id in selected_ids if tile_id < 0 or tile_id >= len(boxes)]
    if invalid_ids:
        raise ValueError(
            f"tile IDs out of range 0..{len(boxes)-1}: {invalid_ids}"
        )

    tile_root = output_dir / "tiles"
    tile_root.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    camera_by_tile: Dict[int, Dict[str, float]] = {}
    paths_by_tile: Dict[int, Dict[str, Path]] = {}

    for tile_id in selected_ids:
        box = boxes[tile_id]
        tile_dir = tile_root / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)

        reference = canonical_4096.crop(box).resize(
            (int(args.tile_render_resolution), int(args.tile_render_resolution)),
            Image.Resampling.LANCZOS,
        )
        reference_path = tile_dir / "reference_tile.png"
        reference.save(reference_path)

        global_crop = global_render_4096.crop(box).resize(
            (int(args.tile_render_resolution), int(args.tile_render_resolution)),
            Image.Resampling.LANCZOS,
        )
        global_crop_path = tile_dir / "global_render_exact_crop.png"
        global_crop.save(global_crop_path)

        derived = derive_crop_camera(
            global_camera,
            box,
            full_width=CANONICAL_SIZE,
            full_height=CANONICAL_SIZE,
            output_width=int(args.tile_render_resolution),
            output_height=int(args.tile_render_resolution),
            blender_shift_y_sign=int(args.blender_shift_y_sign),
        )
        camera_by_tile[tile_id] = derived
        atomic_json(tile_dir / "derived_camera.json", derived)

        render_path = tile_dir / "derived_camera_render.png"
        status_path = tile_dir / "derived_camera_render_status.json"
        comparison_path = tile_dir / "comparison.png"
        paths_by_tile[tile_id] = {
            "reference": reference_path,
            "global_crop": global_crop_path,
            "render": render_path,
            "status": status_path,
            "comparison": comparison_path,
        }

        jobs.append(
            {
                "light_mode": str(args.light),
                "output_png": str(render_path),
                "status_json": str(status_path),
                "camera": {
                    "camera_angle_x": float(derived["camera_angle_x"]),
                    "distance": float(derived["distance"]),
                    "mesh_scale": float(derived["mesh_scale"]),
                    "shift_x": float(derived["shift_x"]),
                    "shift_y": float(derived["shift_y"]),
                },
            }
        )

        print(
            f"[tile {tile_id:02d}] box={box} "
            f"fov={math.degrees(derived['camera_angle_x']):.6f}deg "
            f"fx={derived['fx']:.3f} "
            f"c=({derived['cx']:.3f},{derived['cy']:.3f}) "
            f"shift=({derived['shift_x']:.6f},{derived['shift_y']:.6f})"
        )

    if args.overwrite_renders:
        for job in jobs:
            Path(job["output_png"]).unlink(missing_ok=True)
            Path(job["status_json"]).unlink(missing_ok=True)

    pending_jobs = [
        job
        for job in jobs
        if args.overwrite_renders or not Path(job["output_png"]).is_file()
    ]
    blender_results: Dict[str, Dict[str, Any]] = {}
    if pending_jobs:
        blender_results = run_tile_camera_renders(
            package_dir=package_dir,
            output_dir=tile_root / "_blender",
            jobs=pending_jobs,
            args=args,
        )

    metric_evaluator = MetricEvaluator(
        metric_resolution=int(args.metric_resolution),
        lpips_network=str(args.lpips_net),
        device=str(args.metric_device),
        skip_lpips=bool(args.skip_lpips),
    )

    tile_rows: List[Dict[str, Any]] = []
    for tile_id in selected_ids:
        box = boxes[tile_id]
        paths = paths_by_tile[tile_id]
        derived = camera_by_tile[tile_id]
        render_path = paths["render"]

        status = blender_results.get(str(render_path))
        if status is not None and status.get("status") != "success":
            row = {
                "tile_id": tile_id,
                "box": list(box),
                "status": "failed",
                "error": status.get("error"),
            }
            tile_rows.append(row)
            atomic_json(paths["reference"].parent / "metrics.json", row)
            continue
        if not render_path.is_file():
            row = {
                "tile_id": tile_id,
                "box": list(box),
                "status": "failed",
                "error": "derived camera render is missing",
            }
            tile_rows.append(row)
            atomic_json(paths["reference"].parent / "metrics.json", row)
            continue

        reference = load_black_image(
            paths["reference"],
            (int(args.tile_render_resolution), int(args.tile_render_resolution)),
        )
        global_crop = load_black_image(
            paths["global_crop"],
            (int(args.tile_render_resolution), int(args.tile_render_resolution)),
        )
        derived_render = load_black_image(
            render_path,
            (int(args.tile_render_resolution), int(args.tile_render_resolution)),
        )
        derived_render.save(render_path)

        metrics_ref_global = metric_evaluator.evaluate(reference, global_crop)
        metrics_ref_derived = metric_evaluator.evaluate(reference, derived_render)
        metrics_camera_consistency = metric_evaluator.evaluate(
            global_crop,
            derived_render,
        )

        save_triptych(
            reference,
            global_crop,
            derived_render,
            paths["comparison"],
            (
                f"tile {tile_id:02d} | "
                f"ref/global={metrics_ref_global['psnr']:.4f} dB | "
                f"ref/derived={metrics_ref_derived['psnr']:.4f} dB | "
                f"derived/global={metrics_camera_consistency['psnr']:.4f} dB"
            ),
        )

        row: Dict[str, Any] = {
            "tile_id": int(tile_id),
            "box": list(box),
            "status": "success",
            "derived_fov_deg": math.degrees(float(derived["camera_angle_x"])),
            "derived_fx": float(derived["fx"]),
            "derived_fy": float(derived["fy"]),
            "derived_cx": float(derived["cx"]),
            "derived_cy": float(derived["cy"]),
            "blender_shift_x": float(derived["shift_x"]),
            "blender_shift_y": float(derived["shift_y"]),
            **prefixed_metrics(
                "reference_vs_global_crop",
                metrics_ref_global,
            ),
            **prefixed_metrics(
                "reference_vs_derived_camera",
                metrics_ref_derived,
            ),
            **prefixed_metrics(
                "derived_vs_global_crop",
                metrics_camera_consistency,
            ),
            "reference_path": str(paths["reference"]),
            "global_crop_path": str(paths["global_crop"]),
            "derived_render_path": str(render_path),
            "comparison_path": str(paths["comparison"]),
            "error": None,
        }
        tile_rows.append(row)
        atomic_json(paths["reference"].parent / "metrics.json", row)

        print(
            f"[tile {tile_id:02d} metrics] "
            f"reference/global_crop={metrics_ref_global['psnr']:.4f} dB | "
            f"reference/derived={metrics_ref_derived['psnr']:.4f} dB | "
            f"derived/global_crop={metrics_camera_consistency['psnr']:.4f} dB"
        )

    successful_rows = [
        row for row in tile_rows if row.get("status") == "success"
    ]

    def mean_of(key: str) -> Optional[float]:
        values = [
            float(row[key])
            for row in successful_rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        return None if not values else float(np.mean(values))

    summary = {
        "experiment": "global_model_exact_derived_crop_camera",
        "input_image": str(image_path),
        "global_camera": global_camera,
        "global_generation": generation_summary,
        "global_metrics": global_row,
        "global_render_path": str(global_render_path),
        "package_dir": str(package_dir),
        "tile_layout": {
            "canonical_size": CANONICAL_SIZE,
            "tile_size": int(args.tile_size),
            "tile_stride": int(args.tile_stride),
            "num_tiles": len(boxes),
            "selected_tile_ids": selected_ids,
        },
        "metric_mean_over_successful_tiles": {
            "reference_vs_global_crop_psnr": mean_of(
                "reference_vs_global_crop_psnr"
            ),
            "reference_vs_derived_camera_psnr": mean_of(
                "reference_vs_derived_camera_psnr"
            ),
            "derived_vs_global_crop_psnr": mean_of(
                "derived_vs_global_crop_psnr"
            ),
            "reference_vs_global_crop_ssim": mean_of(
                "reference_vs_global_crop_ssim"
            ),
            "reference_vs_derived_camera_ssim": mean_of(
                "reference_vs_derived_camera_ssim"
            ),
            "derived_vs_global_crop_ssim": mean_of(
                "derived_vs_global_crop_ssim"
            ),
        },
        "interpretation": {
            "reference_vs_global_crop": (
                "Local quality of the ordinary global model, measured by an "
                "exact crop of the 4096 global render."
            ),
            "reference_vs_derived_camera": (
                "Local quality when the identical global model is rendered "
                "directly with the analytically derived crop camera."
            ),
            "derived_vs_global_crop": (
                "Camera/render consistency. This should be much higher than "
                "the two reference-based PSNR values when crop intrinsics and "
                "Blender shift signs are implemented correctly."
            ),
        },
        "tiles": tile_rows,
        "config": vars(args),
    }
    atomic_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "tile_metrics.csv", tile_rows)

    print(f"[done] summary: {output_dir / 'summary.json'}")
    print(f"[done] csv: {output_dir / 'tile_metrics.csv'}")
    print(
        "[mean tile PSNR] "
        f"reference/global_crop="
        f"{summary['metric_mean_over_successful_tiles']['reference_vs_global_crop_psnr']} | "
        f"reference/derived_camera="
        f"{summary['metric_mean_over_successful_tiles']['reference_vs_derived_camera_psnr']} | "
        f"derived/global_crop="
        f"{summary['metric_mean_over_successful_tiles']['derived_vs_global_crop_psnr']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument(
        "--moge-model-path",
        type=Path,
        default=Path("/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)

    # Original inference.py sampler defaults.
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)

    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)

    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=512)
    parser.add_argument("--texture-size", type=int, default=4096)

    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=TILE_STRIDE)
    parser.add_argument(
        "--tile-ids",
        default="24",
        help=(
            "Comma-separated IDs in the 7x7 layout produced by "
            "4096 canvas, 1024 tile, 512 stride. Examples: 24 or 0,6,24,42,48."
        ),
    )
    parser.add_argument(
        "--blender-shift-y-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help=(
            "Blender principal-point convention. Default +1 implements "
            "cy = H/2 + shift_y*W. The derived/global consistency metric "
            "will expose an incorrect sign immediately."
        ),
    )

    parser.add_argument("--light", default="studio")
    parser.add_argument(
        "--render-engine",
        choices=("cycles", "eevee"),
        default="cycles",
    )
    parser.add_argument("--global-render-resolution", type=int, default=4096)
    parser.add_argument("--tile-render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--global-blender-samples", type=int, default=64)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--render-max-faces", type=int, default=0)
    parser.add_argument("--faces-per-shard", type=int, default=2_000_000)
    parser.add_argument("--copy-rows-per-chunk", type=int, default=5_000_000)
    parser.add_argument("--alignment-samples", type=int, default=100_000)

    parser.add_argument(
        "--reuse-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing READY raw cache and generation_summary.json.",
    )
    parser.add_argument(
        "--overwrite-package",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--overwrite-renders",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if int(args.tile_size) != TILE_SIZE or int(args.tile_stride) != TILE_STRIDE:
        raise ValueError(
            "This diagnostic currently requires --tile-size 1024 "
            "and --tile-stride 512."
        )
    if int(args.global_render_resolution) != CANONICAL_SIZE:
        raise ValueError(
            "For exact pixel-coordinate crop comparison, "
            "--global-render-resolution must be 4096."
        )
    if int(args.tile_render_resolution) != TILE_SIZE:
        raise ValueError(
            "For the current crop-intrinsics equations and references, "
            "--tile-render-resolution must be 1024."
        )
    for name in (
        "ss_steps",
        "shape_steps",
        "texture_steps",
        "metric_resolution",
        "blender_samples",
        "global_blender_samples",
        "faces_per_shard",
        "copy_rows_per_chunk",
        "alignment_samples",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()