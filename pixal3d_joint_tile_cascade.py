#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixal3D 1024 joint tile cascade with a union master support.

Support preparation:
    full image -> sparse structure -> global shape-512 -> upsample -> C64_base

For every active 4K image tile (at least ``--min-tile-tokens`` input points):
    C64_tile_input -> floor(/2) -> unique C32_tile
    -> local shape-512 -> learned upsample -> C64_local

All locally proposed coordinates are inserted before the formal 1024 flows:
    C64_master = C64_base UNION C64_local(tile 0) UNION ...

The 1024 shape and texture flows then share one master state. At every step:
    1. Run the complete-image 1024 model on all C64_master points.
    2. During the first N steps, update only with the complete-image velocity.
    3. During the final 12-N steps, give each active tile the current master
       x_t restricted to that tile's C64_local coordinates and the same t.
    4. Each tile predicts velocity using its own crop DINO/projection features.
    5. Tent-weighted overlapping tile velocities replace the global velocity
       where available; all uncovered master points use global velocity.
    6. Update the single master state exactly once.

There are no private tile 1024 trajectories. Every tile coordinate has an exact
row in C64_master, so every tile velocity is evaluated on the current master
state and can be fused without state transplantation.

After GLB export, the script uses the aligned Pixal3D camera/transform from the
provided evaluation code to render with Blender Cycles, save an original/render
comparison, and compute PSNR, SSIM, and LPIPS.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

from inference import (
    MODEL_PATH,
    distance_from_fov,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d.modules.sparse import SparseTensor

try:
    import o_voxel  # type: ignore
except Exception:
    o_voxel = None

GRID_LR = 32
GRID_HR = 64
RESOLUTION_LR = 512
RESOLUTION_HR = 1024
CANONICAL_IMAGE_SIZE = 4096
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512

PIXAL3D_EXPORT_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
OVOXEL_DECODER_TO_GLTF = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
PIXAL3D_DECODER_TO_EXPORTED_GLTF = PIXAL3D_EXPORT_ROTATION @ OVOXEL_DECODER_TO_GLTF
PIXAL3D_EXPORTED_GLTF_TO_INTERNAL = np.linalg.inv(PIXAL3D_DECODER_TO_EXPORTED_GLTF)


@dataclass
class TileExpert:
    tile_id: int
    box: Tuple[int, int, int, int]
    projection_crop_box: Tuple[float, float, float, float]
    input_base_rows: torch.Tensor
    local_coords64: torch.Tensor
    shape_condition_cpu: Mapping[str, Any]
    texture_condition_cpu: Mapping[str, Any]
    lr_trace_path: Optional[str] = None
    # Filled only after all tile supports have been unioned into C64_master.
    master_rows: Optional[torch.Tensor] = None
    active_master_rows: Optional[torch.Tensor] = None
    active_local_rows: Optional[torch.Tensor] = None
    active_weights: Optional[torch.Tensor] = None


@dataclass
class OnlineFlowResult:
    samples: SparseTensor
    times: List[float]
    time_intervals: List[float]
    states: List[torch.Tensor]
    velocities: List[torch.Tensor]
    step_records: List[Dict[str, Any]]
    covered_rows_union: int

@dataclass
class RecordedFlow:
    result: Any
    name: str

    @property
    def trajectory(self) -> Any:
        trajectory = getattr(self.result, "trajectory", None)
        if trajectory is None:
            raise RuntimeError(f"{self.name} did not record a trajectory")
        return trajectory

    @property
    def samples(self) -> SparseTensor:
        samples = getattr(self.result, "samples", None)
        if not isinstance(samples, SparseTensor):
            raise TypeError(f"{self.name}.samples is not a SparseTensor")
        return samples


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _randn(
    rows: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int,
) -> torch.Tensor:
    if rows < 0 or channels <= 0:
        raise ValueError(f"invalid random tensor shape: rows={rows}, channels={channels}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (rows, channels),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _normalization(
    values: Mapping[str, Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    std = torch.as_tensor(values["std"], device=device, dtype=dtype)[None]
    mean = torch.as_tensor(values["mean"], device=device, dtype=dtype)[None]
    return std, mean


def _denormalize_sparse(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std, mean = _normalization(normalization, value.device, value.dtype)
    return value.replace(value.feats * std + mean)


def _normalize_sparse(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std, mean = _normalization(normalization, value.device, value.dtype)
    return value.replace((value.feats - mean) / std)


def _validate_trajectory(flow: RecordedFlow, expected_steps: int) -> None:
    trajectory = flow.trajectory
    states = trajectory.states
    velocities = trajectory.velocities
    if len(states) != expected_steps + 1 or len(velocities) != expected_steps:
        raise RuntimeError(
            f"{flow.name} trajectory must have {expected_steps + 1} states and "
            f"{expected_steps} velocities, got {len(states)} and {len(velocities)}"
        )
    if len(trajectory.times) != expected_steps + 1:
        raise RuntimeError(f"{flow.name} has an invalid mapped-time count")
    if len(trajectory.time_intervals) != expected_steps:
        raise RuntimeError(f"{flow.name} has an invalid dt count")


def _run_recorded_flow(
    *,
    pipeline: Any,
    sampler: Any,
    flow_model: torch.nn.Module,
    noise: SparseTensor,
    condition: Mapping[str, Any],
    sampler_params: Mapping[str, Any],
    description: str,
    concat_cond: Optional[SparseTensor] = None,
) -> RecordedFlow:
    if noise.coords.ndim != 2 or noise.coords.shape[1] != 4:
        raise ValueError(f"{description}: invalid sparse coordinates")
    if concat_cond is not None and not torch.equal(noise.coords, concat_cond.coords):
        raise RuntimeError(f"{description}: noise and concat condition coords differ")

    if pipeline.low_vram:
        flow_model.to(pipeline.device)

    call_kwargs: Dict[str, Any] = {
        **condition,
        **dict(sampler_params),
        "verbose": True,
        "tqdm_desc": description,
        "record_trajectory": True,
        "trajectory_device": "cpu",
        "return_model_history": False,
    }
    if concat_cond is not None:
        call_kwargs["concat_cond"] = concat_cond

    started = time.perf_counter()
    result = sampler.sample(flow_model, noise, **call_kwargs)
    _sync_cuda()
    print(
        f"[recorded-flow] name={description!r} tokens={noise.feats.shape[0]:,} "
        f"channels={noise.feats.shape[1]} seconds={time.perf_counter() - started:.3f}"
    )

    if pipeline.low_vram:
        flow_model.cpu()
        _empty_cuda_cache()

    wrapped = RecordedFlow(result=result, name=description)
    expected_steps = int(sampler_params.get("steps", 12))
    _validate_trajectory(wrapped, expected_steps)
    if not torch.equal(wrapped.samples.coords, noise.coords):
        raise RuntimeError(f"{description}: sampler changed coordinates or token order")
    return wrapped


def _learned_upsample_to_grid64(
    pipeline: Any,
    lr_shape_denormalized: SparseTensor,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    try:
        proposed = decoder.upsample(lr_shape_denormalized, upsample_times=4)
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
            _empty_cuda_cache()

    if proposed.ndim != 2 or proposed.shape[1] != 4:
        raise RuntimeError(
            "shape_slat_decoder.upsample must return [N,4] coordinates; "
            f"got {tuple(proposed.shape)}"
        )
    quantized = torch.cat(
        [
            proposed[:, :1],
            (
                (proposed[:, 1:] + 0.5)
                / float(RESOLUTION_LR)
                * float(GRID_HR)
            ).int(),
        ],
        dim=1,
    )
    coords64 = torch.unique(quantized, dim=0)
    if coords64.numel() == 0:
        raise RuntimeError("learned shape upsample produced an empty grid64 support")
    if torch.any(coords64[:, 1:] < 0) or torch.any(coords64[:, 1:] >= GRID_HR):
        bad = coords64[
            ((coords64[:, 1:] < 0) | (coords64[:, 1:] >= GRID_HR)).any(dim=1)
        ]
        raise RuntimeError(
            "learned shape upsample produced coordinates outside grid64: "
            f"{bad[:16].detach().cpu().tolist()}"
        )
    return coords64


def _downsample_grid64_to_grid32(coords64: torch.Tensor) -> torch.Tensor:
    if coords64.ndim != 2 or coords64.shape[1] != 4:
        raise ValueError("coords64 must have shape [N,4]")
    coords32 = coords64.clone()
    coords32[:, 1:] = torch.div(
        coords32[:, 1:], 2, rounding_mode="floor"
    )
    coords32 = torch.unique(coords32, dim=0)
    if coords32.numel() == 0:
        raise RuntimeError("grid64 -> grid32 downsample produced no coordinates")
    if torch.any(coords32[:, 1:] < 0) or torch.any(coords32[:, 1:] >= GRID_LR):
        raise RuntimeError("downsampled tile coordinates lie outside grid32")
    return coords32


def _coord_key_rows(coords: torch.Tensor) -> Dict[Tuple[int, int, int, int], int]:
    cpu = coords.detach().to(device="cpu", dtype=torch.int64)
    mapping: Dict[Tuple[int, int, int, int], int] = {}
    for row, values in enumerate(cpu.tolist()):
        key = tuple(int(v) for v in values)
        if key in mapping:
            raise RuntimeError(f"duplicate sparse coordinate: {key}")
        mapping[key] = row
    return mapping


def _exact_tile_intersection(
    *,
    global_coords: torch.Tensor,
    tile_input_global_rows: torch.Tensor,
    local_coords64: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact C64_tile_input ∩ C64_local row correspondences."""
    input_coords = global_coords.index_select(0, tile_input_global_rows)
    input_map = _coord_key_rows(input_coords)
    local_map = _coord_key_rows(local_coords64)
    common_keys = sorted(set(input_map).intersection(local_map))
    if not common_keys:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    input_rows_relative = torch.tensor(
        [input_map[key] for key in common_keys], dtype=torch.long
    )
    local_rows = torch.tensor(
        [local_map[key] for key in common_keys], dtype=torch.long
    )
    global_rows = tile_input_global_rows.detach().cpu().index_select(
        0, input_rows_relative
    )
    return global_rows, local_rows


def _tile_rows_and_weights(
    tile_ids: torch.Tensor,
    tile_weights: torch.Tensor,
    tile_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    membership = tile_ids == int(tile_id)
    row_mask = membership.any(dim=1)
    rows = torch.where(row_mask)[0]
    if rows.numel() == 0:
        return rows, torch.empty(0, device=tile_weights.device)
    slots = membership[rows].to(torch.int64).argmax(dim=1)
    weights = tile_weights[rows, slots]
    if torch.any(weights <= 0) or not torch.isfinite(weights).all():
        raise RuntimeError(f"tile {tile_id}: invalid membership weights")
    return rows, weights


def _parse_tile_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    output: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        output.add(int(item))
    return output


def _estimate_camera(
    *,
    image_1024: Image.Image,
    output_dir: Path,
    manual_fov: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if manual_fov > 0:
        camera_angle_x = float(manual_fov)
        distance = distance_from_fov(
            camera_angle_x,
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale,
            image_resolution,
        )["distance_from_x"]
        return {
            "camera_angle_x": camera_angle_x,
            "distance": float(distance),
            "mesh_scale": float(mesh_scale),
        }

    temporary = output_dir / f"_joint_tile_moge_{int(time.time() * 1000)}.png"
    image_1024.save(temporary)
    print("[MoGe-2] Loading model for camera estimation...")
    model = load_moge_model(device="cuda")
    try:
        params = get_camera_params_wild_moge(
            str(temporary),
            model,
            device="cuda",
            mesh_scale=mesh_scale,
            extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
    finally:
        model.cpu()
        del model
        temporary.unlink(missing_ok=True)
        _empty_cuda_cache()
    return {
        "camera_angle_x": float(params["camera_angle_x"]),
        "distance": float(params["distance"]),
        "mesh_scale": float(params["mesh_scale"]),
    }


def _build_sampler_params(args: argparse.Namespace, pipeline: Any) -> Dict[str, Dict[str, Any]]:
    ss = {
        **pipeline.sparse_structure_sampler_params,
        "steps": int(args.steps),
        "guidance_strength": float(args.ss_guidance_strength),
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    shape = {
        **pipeline.shape_slat_sampler_params,
        "steps": int(args.steps),
        "guidance_strength": float(args.shape_guidance_strength),
        "guidance_rescale": float(args.shape_guidance_rescale),
        "rescale_t": float(args.shape_rescale_t),
    }
    texture = {
        **pipeline.tex_slat_sampler_params,
        "steps": int(args.steps),
        "guidance_strength": float(args.texture_guidance_strength),
        "guidance_rescale": float(args.texture_guidance_rescale),
        "rescale_t": float(args.texture_rescale_t),
    }
    return {"ss": ss, "shape": shape, "texture": texture}


def _global_initial_support(
    *,
    pipeline: Any,
    image_512: Image.Image,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    sampler_params: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    RecordedFlow,
    SparseTensor,
]:
    """Run standard global SS + shape512 + learned upsample to C64_global."""
    cond_ss = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
    )
    _seed_everything(seed)
    coords32 = pipeline.sample_sparse_structure(
        cond_ss,
        resolution=GRID_LR,
        sampler_params=dict(sampler_params["ss"]),
    )
    del cond_ss
    if coords32.numel() == 0:
        raise RuntimeError("global sparse structure is empty")
    print(f"[global-support] C32 tokens={coords32.shape[0]:,}")

    cond_lr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image_512],
        coords32,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
        grid_resolution_override=GRID_LR,
    )
    flow_lr = pipeline.models["shape_slat_flow_model_512"]
    lr_noise = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(flow_lr.in_channels),
            device=pipeline.device,
            seed=seed + 101,
        ),
        coords=coords32,
    )
    lr_record = _run_recorded_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        flow_model=flow_lr,
        noise=lr_noise,
        condition=cond_lr,
        sampler_params=sampler_params["shape"],
        description="Global shape SLat 512 (recorded)",
    )
    lr_shape_denorm = _denormalize_sparse(
        lr_record.samples, pipeline.shape_slat_normalization
    )
    coords64 = _learned_upsample_to_grid64(pipeline, lr_shape_denorm)
    print(f"[global-support] C64_global tokens={coords64.shape[0]:,}")
    return coords32, coords64, lr_record, lr_shape_denorm

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(json_safe(value), file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def composite_on_black(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def save_metric_reference(
    condition_image: Image.Image,
    output_path: Path,
    render_resolution: int,
) -> Image.Image:
    reference = composite_on_black(condition_image)
    target_size = (int(render_resolution), int(render_resolution))
    if reference.size != target_size:
        reference = reference.resize(target_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.save(output_path)
    return reference


def save_black_composited_render(path: Path, expected_resolution: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = composite_on_black(image)
    target_size = (int(expected_resolution), int(expected_resolution))
    if rgb.size != target_size:
        rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)
    rgb.save(path)
    return rgb


def save_comparison(
    original: Image.Image,
    rendered: Image.Image,
    output_path: Path,
) -> None:
    original_rgb = original.convert("RGB")
    rendered_rgb = rendered.convert("RGB")
    if rendered_rgb.size != original_rgb.size:
        rendered_rgb = rendered_rgb.resize(
            original_rgb.size,
            Image.Resampling.LANCZOS,
        )
    width, height = original_rgb.size
    comparison = Image.new("RGB", (width * 2, height), (255, 255, 255))
    comparison.paste(original_rgb, (0, 0))
    comparison.paste(rendered_rgb, (width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)

BLENDER_HELPER_SOURCE = r"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import bpy
from mathutils import Matrix


def log(message):
    print("[blender-cycles] %s" % message, flush=True)


def require_single_visible_device():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != 1:
        raise RuntimeError(
            "Cycles helper requires exactly one CUDA_VISIBLE_DEVICES entry; "
            "got %r" % value
        )
    log(
        "CUDA_DEVICE_ORDER=%s CUDA_VISIBLE_DEVICES=%s"
        % (os.environ.get("CUDA_DEVICE_ORDER", "<unset>"), tokens[0])
    )


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.images,
    ):
        for block in list(datablocks):
            try:
                datablocks.remove(block)
            except Exception:
                pass


def clear_lights():
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    for light in list(bpy.data.lights):
        if light.users == 0:
            try:
                bpy.data.lights.remove(light)
            except Exception:
                pass


def configure_cycles_gpu(scene):
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise RuntimeError("Cycles addon is unavailable")

    preferences = addon.preferences
    selected_backend = None
    selected_device = None
    errors = []

    # A800 has no RT cores; try CUDA first, then OptiX as fallback.
    for backend in ("CUDA", "OPTIX"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            devices = list(preferences.devices)
            candidates = [device for device in devices if device.type == backend]
            if not candidates:
                continue

            for device in devices:
                device.use = False
            selected_device = candidates[0]
            selected_device.use = True
            selected_backend = backend
            break
        except Exception as exc:
            errors.append("%s: %s" % (backend, exc))

    if selected_backend is None or selected_device is None:
        raise RuntimeError(
            "No usable Cycles CUDA/OptiX device found: " + "; ".join(errors)
        )

    scene.cycles.device = "GPU"
    log(
        "cycles backend=%s selected=%s id=%s"
        % (
            selected_backend,
            selected_device.name,
            getattr(selected_device, "id", "<unknown>"),
        )
    )
    for device in preferences.devices:
        log(
            "cycles-device name=%s type=%s use=%s id=%s"
            % (
                device.name,
                device.type,
                device.use,
                getattr(device, "id", "<unknown>"),
            )
        )


def configure_render(resolution, samples):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    configure_cycles_gpu(scene)
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True

    # Reuse geometry/BVH across the light variants of the same imported model.
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True

    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    return scene

def set_world(scene, strength, color=(1.0, 1.0, 1.0, 1.0)):
    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = color
        background.inputs["Strength"].default_value = float(strength)


def import_glb(path):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(
            filepath=str(path),
            merge_vertices=False,
            import_shading="NORMALS",
        )
    except TypeError:
        bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    roots = [obj for obj in imported if obj.parent is None]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("temporary GLB contains no mesh objects")
    total_vertices = sum(len(obj.data.vertices) for obj in meshes)
    total_polygons = sum(len(obj.data.polygons) for obj in meshes)
    log(
        "imported meshes=%d vertices=%d polygons=%d merge_vertices=False"
        % (len(meshes), total_vertices, total_polygons)
    )
    return imported, roots


def apply_root_transform(roots, values):
    matrix_gltf = Matrix(values)
    gltf_to_blender = Matrix((
        (1.0, 0.0,  0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0,  0.0, 0.0),
        (0.0, 0.0,  0.0, 1.0),
    ))
    matrix_blender = gltf_to_blender @ matrix_gltf @ gltf_to_blender.inverted()
    for root in roots:
        root.matrix_world = matrix_blender @ root.matrix_world
    bpy.context.view_layer.update()


def add_area(scene, name, location, energy, size, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = float(energy)
    data.shape = "DISK"
    data.size = float(size)
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def add_light_rig(scene, mode):
    if mode == "studio":
        set_world(scene, 0.22)
        add_area(scene, "Key", (-3.5, -4.5, 4.5), 850.0, 4.5)
        add_area(scene, "Fill", (4.0, -2.5, 2.0), 430.0, 5.0)
        add_area(scene, "Top", (0.0, 0.5, 5.0), 300.0, 4.0)
        add_area(scene, "Rim", (0.5, 4.0, 3.0), 300.0, 3.0)
    elif mode == "three_point":
        set_world(scene, 0.12)
        add_area(scene, "Key", (-3.0, -4.0, 3.5), 1000.0, 3.0)
        add_area(scene, "Fill", (3.5, -2.5, 1.5), 350.0, 4.0)
        add_area(scene, "Rim", (1.0, 4.0, 3.0), 650.0, 2.5)
    elif mode == "softbox":
        set_world(scene, 0.30)
        add_area(scene, "SoftboxLeft", (-3.5, -3.5, 3.0), 650.0, 6.0)
        add_area(scene, "SoftboxRight", (3.5, -3.0, 2.5), 600.0, 6.0)
        add_area(scene, "SoftboxTop", (0.0, 0.0, 5.0), 250.0, 5.0)
    elif mode == "front":
        set_world(scene, 0.15)
        add_area(scene, "Front", (0.0, -4.5, 0.8), 1000.0, 5.0)
        add_area(scene, "FrontTop", (0.0, -2.5, 4.0), 250.0, 4.0)
    elif mode == "uniform":
        set_world(scene, 0.35)
        positions = (
            (0.0, -4.0, 0.0),
            (0.0, 4.0, 0.0),
            (-4.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (0.0, 0.0, 4.0),
            (0.0, 0.0, -4.0),
        )
        for index, position in enumerate(positions):
            add_area(scene, "Uniform%02d" % index, position, 230.0, 4.5)
    elif mode == "dramatic":
        set_world(scene, 0.035)
        add_area(scene, "HardKey", (-3.0, -3.5, 4.5), 1250.0, 1.5)
        add_area(
            scene,
            "CoolRim",
            (2.5, 4.0, 3.5),
            900.0,
            2.0,
            (0.65, 0.75, 1.0),
        )
        add_area(
            scene,
            "WarmFill",
            (3.5, -1.5, 0.5),
            120.0,
            3.0,
            (1.0, 0.72, 0.55),
        )
    else:
        raise ValueError("unsupported light mode: %s" % mode)


def create_aligned_camera(scene, distance, fov_rad):
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix((
        (1.0, 0.0,  0.0, 0.0),
        (0.0, 0.0, -1.0, -float(distance)),
        (0.0, 1.0,  0.0, 0.0),
        (0.0, 0.0,  0.0, 1.0),
    ))
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 32.0
    camera_data.sensor_height = 32.0
    camera_data.lens = 16.0 / math.tan(float(fov_rad) / 2.0)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    scene.camera = camera
    return camera


def write_status(path, payload):
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def group_key(job):
    return (
        job["input_glb"],
        int(job["resolution"]),
        float(job["fov_rad"]),
        float(job["distance"]),
        int(job.get("samples", 64)),
        json.dumps(job["transform"], separators=(",", ":")),
    )


def render_group(group_index, group_count, jobs):
    first = jobs[0]
    clear_scene()
    scene = configure_render(first["resolution"], first.get("samples", 64))

    started = time.perf_counter()
    _, roots = import_glb(first["input_glb"])
    apply_root_transform(roots, first["transform"])
    create_aligned_camera(scene, first["distance"], first["fov_rad"])
    log(
        "group=%d/%d imported_once jobs=%d seconds=%.3f input=%s"
        % (
            group_index,
            group_count,
            len(jobs),
            time.perf_counter() - started,
            first["input_glb"],
        )
    )

    for job_index, job in enumerate(jobs, start=1):
        try:
            clear_lights()
            add_light_rig(scene, job["light_mode"])
            output = Path(job["output_png"])
            output.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(output)
            render_started = time.perf_counter()
            log(
                "render group=%d job=%d/%d light=%s start"
                % (group_index, job_index, len(jobs), job["light_mode"])
            )
            bpy.ops.render.render(write_still=True)
            log(
                "render group=%d job=%d/%d light=%s seconds=%.3f done"
                % (
                    group_index,
                    job_index,
                    len(jobs),
                    job["light_mode"],
                    time.perf_counter() - render_started,
                )
            )
            write_status(
                job["status_json"],
                {"status": "success", "output_png": job["output_png"], "error": None},
            )
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
            write_status(
                job["status_json"],
                {
                    "status": "failed",
                    "output_png": job["output_png"],
                    "error": error,
                    "traceback": traceback.format_exc(),
                },
            )


def main():
    require_single_visible_device()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-json", required=True)
    args = parser.parse_args(argv)
    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))

    groups = OrderedDict()
    for job in jobs:
        groups.setdefault(group_key(job), []).append(job)

    group_values = list(groups.values())
    for group_index, group_jobs in enumerate(group_values, start=1):
        try:
            render_group(group_index, len(group_values), group_jobs)
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
            for job in group_jobs:
                write_status(
                    job["status_json"],
                    {
                        "status": "failed",
                        "output_png": job["output_png"],
                        "error": error,
                        "traceback": traceback.format_exc(),
                    },
                )


if __name__ == "__main__":
    main()
"""

def ensure_blender_helper(work_dir: Path) -> Path:
    helper_path = work_dir / "_pixal3d_blender_texture_render.py"
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.write_text(BLENDER_HELPER_SOURCE, encoding="utf-8")
    return helper_path


def run_blender_jobs(
    blender_executable: str,
    helper_path: Path,
    jobs: Sequence[Mapping[str, Any]],
    work_dir: Path,
    log_path: Path,
) -> Dict[str, Dict[str, Any]]:
    jobs = list(jobs)
    if not jobs:
        print("[render] no missing renders")
        return {}

    jobs_path = work_dir / "blender_jobs.json"
    atomic_json(jobs_path, jobs)
    command = [
        blender_executable,
        "--background",
        "--python",
        str(helper_path),
        "--",
        "--jobs-json",
        str(jobs_path),
    ]
    print(f"[render] {shlex.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()

    visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_tokens = [
        token.strip()
        for token in visible_devices.split(",")
        if token.strip()
    ]
    if len(visible_tokens) != 1:
        raise RuntimeError(
            "This renderer requires exactly one CUDA_VISIBLE_DEVICES entry; "
            f"got {visible_devices!r}"
        )

    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = visible_tokens[0]
    environment.pop("LIBGL_ALWAYS_SOFTWARE", None)

    for key in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(key, None)

    ld_library_path = environment.get("LD_LIBRARY_PATH", "")
    if "conda" in ld_library_path.lower() or "miniconda" in ld_library_path.lower():
        environment.pop("LD_LIBRARY_PATH", None)

    print(
        f"[blender-env] CUDA_DEVICE_ORDER={environment['CUDA_DEVICE_ORDER']} "
        f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}"
    )

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n# Blender command: {shlex.join(command)}\n")
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )

    results: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        output_png = str(job["output_png"])
        status_path = Path(job["status_json"])
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception as exc:
                status = {
                    "status": "failed",
                    "output_png": output_png,
                    "error": f"invalid Blender status JSON: {exc}",
                }
        elif Path(output_png).is_file():
            status = {
                "status": "success",
                "output_png": output_png,
                "error": None,
            }
        else:
            status = {
                "status": "failed",
                "output_png": output_png,
                "error": (
                    f"Blender exited with code {process.returncode} "
                    "without producing this render"
                ),
            }
        results[output_png] = status

    if process.returncode != 0:
        print(
            f"[render-warning] Blender exited with code {process.returncode}; "
            f"see {log_path}"
        )
    failures = sum(
        result.get("status") != "success" for result in results.values()
    )
    print(
        f"[render] requested={len(jobs)} "
        f"success={len(jobs) - failures} failed={failures}"
    )
    return results


def load_metric_tensor(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = composite_on_black(image)
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def psnr_metric(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction, reference).item())
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def gaussian_kernel(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def ssim_metric(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    x = reference.unsqueeze(0).float()
    y = prediction.unsqueeze(0).float()
    channels = int(x.shape[1])
    kernel = gaussian_kernel(window_size, sigma, x.device, x.dtype)
    kernel = kernel[None, None].expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = (
        (2.0 * mu_xy + c1)
        * (2.0 * sigma_xy + c2)
        / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12)
    )
    return float(ssim_map.mean().item())


class LPIPSEvaluator:
    def __init__(self, network: str, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "The lpips package is required. Install it with: pip install lpips"
            ) from exc
        self.device = device
        # lpips 0.1 still calls torchvision through the legacy `pretrained`
        # argument.  Suppress only those two compatibility warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            self.model = lpips.LPIPS(net=network).eval().to(device)

    @torch.inference_mode()
    def evaluate(self, reference: torch.Tensor, prediction: torch.Tensor) -> float:
        x = reference.unsqueeze(0).to(self.device)
        y = prediction.unsqueeze(0).to(self.device)
        value = self.model(x * 2.0 - 1.0, y * 2.0 - 1.0)
        return float(value.mean().item())


def evaluate_render(
    reference_cpu: torch.Tensor,
    prediction_path: Path,
    lpips_evaluator: LPIPSEvaluator,
) -> Dict[str, float]:
    height, width = int(reference_cpu.shape[1]), int(reference_cpu.shape[2])
    prediction_cpu = load_metric_tensor(prediction_path, (width, height))
    # PSNR/SSIM are inexpensive and stay on CPU. LPIPS uses the selected device.
    metrics = {
        "psnr_db": psnr_metric(reference_cpu, prediction_cpu),
        "ssim": ssim_metric(reference_cpu, prediction_cpu),
        "lpips": lpips_evaluator.evaluate(reference_cpu, prediction_cpu),
    }
    del prediction_cpu
    return metrics


def _features(value: Any) -> torch.Tensor:
    return value.feats if hasattr(value, "feats") else value


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.detach().to(device="cpu", copy=True),
            coords=value.coords.detach().to(device="cpu", copy=True),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    return value


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.to(device=device, non_blocking=False),
            coords=value.coords.to(device=device, non_blocking=False),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


def _sample_once_kwargs(sampler_params: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "steps", "rescale_t", "verbose", "tqdm_desc", "record_trajectory",
        "trajectory_device", "return_model_history",
    }
    return {key: value for key, value in sampler_params.items() if key not in excluded}


def _build_master_support(
    base_coords64: torch.Tensor,
    tile_experts: Sequence[TileExpert],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Preserve base row order and append the sorted union of tile-only coords."""
    if base_coords64.ndim != 2 or base_coords64.shape[1] != 4:
        raise ValueError("base_coords64 must have shape [N,4]")
    base_cpu = base_coords64.detach().to(device="cpu", dtype=torch.int64)
    base_keys = [tuple(int(v) for v in row) for row in base_cpu.tolist()]
    if len(set(base_keys)) != len(base_keys):
        raise RuntimeError("base C64 support contains duplicate coordinates")

    known = set(base_keys)
    extra_keys: set[Tuple[int, int, int, int]] = set()
    total_local_rows = 0
    for expert in tile_experts:
        local_cpu = expert.local_coords64.detach().to(device="cpu", dtype=torch.int64)
        total_local_rows += int(local_cpu.shape[0])
        for row in local_cpu.tolist():
            key = tuple(int(v) for v in row)
            if key not in known:
                extra_keys.add(key)

    ordered = base_keys + sorted(extra_keys)
    master_cpu = torch.tensor(ordered, dtype=base_coords64.dtype)
    master = master_cpu.to(device=base_coords64.device)
    return master, {
        "base_tokens": int(base_coords64.shape[0]),
        "tile_local_rows_total": int(total_local_rows),
        "added_unique_tokens": int(len(extra_keys)),
        "master_tokens": int(len(ordered)),
    }


def _bind_tile_experts_to_master(
    *,
    tile_experts: Sequence[TileExpert],
    master_coords64: torch.Tensor,
    master_tile_ids: torch.Tensor,
    master_tile_weights: torch.Tensor,
) -> List[TileExpert]:
    """Map every local coordinate to master and attach positive tent rows."""
    master_map = _coord_key_rows(master_coords64)
    usable: List[TileExpert] = []
    for expert in tile_experts:
        local_cpu = expert.local_coords64.detach().to(device="cpu", dtype=torch.int64)
        master_rows = []
        for row in local_cpu.tolist():
            key = tuple(int(v) for v in row)
            if key not in master_map:
                raise RuntimeError(
                    f"tile {expert.tile_id}: local coordinate missing from C64_master: {key}"
                )
            master_rows.append(master_map[key])
        expert.master_rows = torch.tensor(master_rows, dtype=torch.long)

        assigned_rows, assigned_weights = _tile_rows_and_weights(
            master_tile_ids, master_tile_weights, expert.tile_id
        )
        weight_by_master = {
            int(row): float(weight)
            for row, weight in zip(
                assigned_rows.detach().cpu().tolist(),
                assigned_weights.detach().cpu().tolist(),
            )
        }
        active_local: List[int] = []
        active_master: List[int] = []
        active_weights: List[float] = []
        for local_row, master_row in enumerate(master_rows):
            weight = weight_by_master.get(int(master_row), 0.0)
            if weight > 0.0 and math.isfinite(weight):
                active_local.append(local_row)
                active_master.append(master_row)
                active_weights.append(weight)

        if not active_local:
            print(
                f"[master-bind] tile={expert.tile_id:02d} has no positive tent rows; skipped"
            )
            continue
        expert.active_local_rows = torch.tensor(active_local, dtype=torch.long)
        expert.active_master_rows = torch.tensor(active_master, dtype=torch.long)
        expert.active_weights = torch.tensor(active_weights, dtype=torch.float32)
        print(
            f"[master-bind] tile={expert.tile_id:02d} local={len(master_rows):,} "
            f"active_tent={len(active_local):,}"
        )
        usable.append(expert)
    return usable


def _save_tile_lr_trace(
    *,
    path: Path,
    tile_id: int,
    box: Sequence[int],
    projection_crop_box: Sequence[float],
    input_coords64: torch.Tensor,
    coords32: torch.Tensor,
    local_coords64: torch.Tensor,
    lr_shape_flow: RecordedFlow,
    seeds: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "pixal3d_joint_tile_master_union_support_v3",
            "tile_id": int(tile_id),
            "box_4096": [int(v) for v in box],
            "projection_crop_box": [float(v) for v in projection_crop_box],
            "input_coords64": input_coords64.detach().cpu(),
            "coords32": coords32.detach().cpu(),
            "local_coords64": local_coords64.detach().cpu(),
            "seeds": {key: int(value) for key, value in seeds.items()},
            "shape_512": _serialize_trajectory(lr_shape_flow),
        },
        temporary,
    )
    temporary.replace(path)


def _serialize_trajectory(flow: RecordedFlow) -> Dict[str, Any]:
    trajectory = flow.trajectory
    return {
        "times": torch.as_tensor(trajectory.times, dtype=torch.float64),
        "time_intervals": torch.as_tensor(trajectory.time_intervals, dtype=torch.float64),
        "states": [state.detach().cpu() for state in trajectory.states],
        "velocities": [velocity.detach().cpu() for velocity in trajectory.velocities],
        "final_samples": flow.samples.feats.detach().cpu(),
    }


def _prepare_one_tile(
    *,
    pipeline: Any,
    tile_id: int,
    box: Tuple[int, int, int, int],
    projection_crop_box: Tuple[float, float, float, float],
    tile_image_1024: Image.Image,
    base_coords64: torch.Tensor,
    tile_input_rows: torch.Tensor,
    camera: Mapping[str, float],
    sampler_params: Mapping[str, Mapping[str, Any]],
    base_seed: int,
    trace_dir: Path,
    save_lr_trace: bool,
) -> TileExpert:
    started = time.perf_counter()
    tile_input_rows_device = tile_input_rows.to(
        device=base_coords64.device, dtype=torch.long
    )
    input_coords64 = base_coords64.index_select(0, tile_input_rows_device)
    coords32 = _downsample_grid64_to_grid32(input_coords64)
    tile_image_512 = tile_image_1024.resize(
        (RESOLUTION_LR, RESOLUTION_LR), Image.Resampling.LANCZOS
    )

    lr_seed = base_seed + tile_id * 10 + 1
    print(
        f"[tile-prepare] tile={tile_id:02d} box={box} "
        f"C64_input={input_coords64.shape[0]:,} C32={coords32.shape[0]:,}"
    )

    cond_lr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [tile_image_512],
        coords32,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
        grid_resolution_override=GRID_LR,
        projection_crop_box=projection_crop_box,
    )
    shape_lr_model = pipeline.models["shape_slat_flow_model_512"]
    lr_noise = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(shape_lr_model.in_channels),
            device=pipeline.device,
            seed=lr_seed,
        ),
        coords=coords32,
    )
    lr_flow = _run_recorded_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        flow_model=shape_lr_model,
        noise=lr_noise,
        condition=cond_lr,
        sampler_params=sampler_params["shape"],
        description=f"Tile {tile_id:02d} shape SLat 512",
    )
    lr_shape_denorm = _denormalize_sparse(
        lr_flow.samples, pipeline.shape_slat_normalization
    )
    local_coords64 = _learned_upsample_to_grid64(pipeline, lr_shape_denorm)
    print(
        f"[tile-prepare] tile={tile_id:02d} C64_local={local_coords64.shape[0]:,}"
    )

    shape_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [tile_image_1024],
        local_coords64,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
        grid_resolution_override=GRID_HR,
        projection_crop_box=projection_crop_box,
    )
    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [tile_image_1024],
        local_coords64,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
        grid_resolution_override=GRID_HR,
        projection_crop_box=projection_crop_box,
    )

    trace_path_value: Optional[str] = None
    if save_lr_trace:
        trace_path = trace_dir / f"tile_{tile_id:04d}.pt"
        _save_tile_lr_trace(
            path=trace_path,
            tile_id=tile_id,
            box=box,
            projection_crop_box=projection_crop_box,
            input_coords64=input_coords64,
            coords32=coords32,
            local_coords64=local_coords64,
            lr_shape_flow=lr_flow,
            seeds={"shape_512": lr_seed},
        )
        trace_path_value = str(trace_path.resolve())

    shape_condition_cpu = _tree_to_cpu(shape_condition)
    texture_condition_cpu = _tree_to_cpu(texture_condition)
    print(
        f"[tile-prepare] tile={tile_id:02d} input={input_coords64.shape[0]:,} "
        f"local={local_coords64.shape[0]:,} seconds={time.perf_counter()-started:.3f}"
    )
    del cond_lr, lr_noise, lr_shape_denorm, shape_condition, texture_condition, lr_flow
    _empty_cuda_cache()
    return TileExpert(
        tile_id=int(tile_id),
        box=tuple(int(v) for v in box),
        projection_crop_box=tuple(float(v) for v in projection_crop_box),
        input_base_rows=tile_input_rows.detach().cpu(),
        local_coords64=local_coords64,
        shape_condition_cpu=shape_condition_cpu,
        texture_condition_cpu=texture_condition_cpu,
        lr_trace_path=trace_path_value,
    )


@torch.no_grad()
def _run_online_fused_flow(
    *,
    pipeline: Any,
    sampler: Any,
    flow_model: torch.nn.Module,
    global_state: SparseTensor,
    global_condition: Mapping[str, Any],
    tile_experts: Sequence[TileExpert],
    sampler_params: Mapping[str, Any],
    replace_last_n: int,
    stage: str,
    global_concat_cond: Optional[SparseTensor] = None,
) -> OnlineFlowResult:
    if stage not in {"shape", "texture"}:
        raise ValueError(stage)
    steps = int(sampler_params.get("steps", 12))
    if not 0 <= replace_last_n <= steps:
        raise ValueError(f"{stage}: replace_last_n must be in [0,{steps}]")
    rescale_t = float(sampler_params.get("rescale_t", 1.0))
    times = [float(v) for v in sampler.timestep_schedule(steps, rescale_t)]
    intervals = [times[i] - times[i + 1] for i in range(steps)]
    start_step = steps - replace_last_n
    step_kwargs = _sample_once_kwargs(sampler_params)

    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    device = global_state.device
    global_condition_device = _tree_to_device(global_condition, device)
    if global_concat_cond is not None and not torch.equal(
        global_state.coords, global_concat_cond.coords
    ):
        raise RuntimeError(f"{stage}: global state/concat coords differ")

    states_cpu = [global_state.feats.detach().cpu().clone()]
    velocities_cpu: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    union_covered = torch.zeros(
        global_state.feats.shape[0], dtype=torch.bool, device=device
    )

    progress = tqdm(
        range(steps), desc=f"Master-union joint {stage} 1024", dynamic_ncols=True
    )
    for step in progress:
        t = times[step]
        t_next = times[step + 1]
        dt = intervals[step]

        # The complete-image model always predicts on every master point.
        global_call = {**global_condition_device, **step_kwargs}
        if global_concat_cond is not None:
            global_call["concat_cond"] = global_concat_cond
        global_out = sampler.sample_once(
            flow_model, global_state, t, t_next, **global_call
        )
        global_velocity = _features(global_out.pred_v).to(dtype=torch.float32)
        merged = global_velocity.clone()

        velocity_sum = torch.zeros_like(global_velocity)
        weight_sum = torch.zeros(
            (global_velocity.shape[0], 1), device=device, dtype=torch.float32
        )
        tile_calls = 0

        # First N steps are complete-image-only. Tile forwards begin exactly at
        # start_step and always receive the current master x_t at their rows.
        if step >= start_step:
            for expert in tile_experts:
                if (
                    expert.master_rows is None
                    or expert.active_master_rows is None
                    or expert.active_local_rows is None
                    or expert.active_weights is None
                ):
                    raise RuntimeError(
                        f"tile {expert.tile_id}: master row binding is incomplete"
                    )
                master_rows = expert.master_rows.to(device=device, dtype=torch.long)
                tile_state = SparseTensor(
                    feats=global_state.feats.index_select(0, master_rows),
                    coords=expert.local_coords64,
                )
                condition_cpu = (
                    expert.shape_condition_cpu
                    if stage == "shape"
                    else expert.texture_condition_cpu
                )
                condition_device = _tree_to_device(condition_cpu, device)
                tile_call = {**condition_device, **step_kwargs}
                if global_concat_cond is not None:
                    tile_call["concat_cond"] = SparseTensor(
                        feats=global_concat_cond.feats.index_select(0, master_rows),
                        coords=expert.local_coords64,
                    )
                tile_out = sampler.sample_once(
                    flow_model, tile_state, t, t_next, **tile_call
                )
                tile_velocity = _features(tile_out.pred_v).to(dtype=torch.float32)
                active_local = expert.active_local_rows.to(
                    device=device, dtype=torch.long
                )
                active_master = expert.active_master_rows.to(
                    device=device, dtype=torch.long
                )
                weights = expert.active_weights.to(
                    device=device, dtype=torch.float32
                )[:, None]
                selected = tile_velocity.index_select(0, active_local)
                velocity_sum.index_add_(0, active_master, selected * weights)
                weight_sum.index_add_(0, active_master, weights)
                tile_calls += 1
                del tile_state, condition_device, tile_call, tile_out

            covered = weight_sum[:, 0] > 0
            union_covered |= covered
            if torch.any(covered):
                # Simple replacement requested by the experiment: normalized
                # overlapping-tile mean replaces the complete-image velocity.
                merged[covered] = velocity_sum[covered] / weight_sum[covered]
        else:
            covered = torch.zeros(
                global_velocity.shape[0], dtype=torch.bool, device=device
            )

        next_global_feats = global_state.feats - float(dt) * merged.to(
            global_state.dtype
        )
        next_global = global_state.replace(next_global_feats)
        if not torch.isfinite(next_global.feats).all():
            raise RuntimeError(f"{stage}: non-finite master state at step {step}")

        covered_count = int(covered.sum().item())
        if covered_count:
            local_mean = velocity_sum[covered] / weight_sum[covered]
            global_cov = global_velocity[covered]
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    local_mean.flatten()[None], global_cov.flatten()[None]
                ).item()
            )
            norm_ratio = float(
                local_mean.norm().item() / max(global_cov.norm().item(), 1e-12)
            )
        else:
            cosine = 1.0
            norm_ratio = 1.0
        record = {
            "step": int(step),
            "t": float(t),
            "t_next": float(t_next),
            "dt": float(dt),
            "replacement_active": bool(step >= start_step),
            "covered_rows": covered_count,
            "covered_ratio": covered_count / float(global_state.feats.shape[0]),
            "tile_experts_called": int(tile_calls),
            "fallback_global_rows": int(global_state.feats.shape[0] - covered_count),
            "local_vs_global_cosine_covered": cosine,
            "local_to_global_norm_ratio_covered": norm_ratio,
        }
        records.append(record)
        progress.set_postfix(
            covered=f"{covered_count}/{global_state.feats.shape[0]}",
            tiles=tile_calls,
            replace=int(step >= start_step),
            cos=f"{cosine:.4f}",
        )
        global_state = next_global
        velocities_cpu.append(merged.detach().cpu().clone())
        states_cpu.append(global_state.feats.detach().cpu().clone())
        del global_out

    if pipeline.low_vram:
        flow_model.cpu()
        _empty_cuda_cache()
    return OnlineFlowResult(
        samples=global_state,
        times=times,
        time_intervals=intervals,
        states=states_cpu,
        velocities=velocities_cpu,
        step_records=records,
        covered_rows_union=int(union_covered.sum().item()),
    )


def _export_glb(
    *,
    pipeline: Any,
    shape_slat: SparseTensor,
    texture_slat: SparseTensor,
    output_path: Path,
    texture_size: int,
    decimation_target: int,
) -> Dict[str, int]:
    if o_voxel is None:
        raise RuntimeError("o_voxel is unavailable; use --no-decode or fix the environment")
    mesh_list = pipeline.decode_latent(shape_slat, texture_slat, RESOLUTION_HR)
    mesh = mesh_list[0]
    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])
    print(f"[Decoder mesh] vertices={vertices:,}, faces={faces:,}")
    # Match the supplied evaluation path: preserve the decoder mesh topology
    # instead of silently evaluating an aggressively decimated surrogate.
    effective_decimation_target = faces
    if int(decimation_target) != effective_decimation_target:
        print(
            f"[export] requested_decimation_target={int(decimation_target):,} "
            f"ignored; preserving decoder faces={effective_decimation_target:,}"
        )
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=RESOLUTION_HR,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=effective_decimation_target,
        texture_size=int(texture_size),
        remesh=False,
        use_tqdm=True,
        verbose=False,
    )
    glb.apply_transform(PIXAL3D_EXPORT_ROTATION)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output_path), extension_webp=False)
    print(f"[Done] GLB saved to: {output_path}")
    return {"decoder_vertices": vertices, "decoder_faces": faces, "effective_decimation_target": effective_decimation_target}


def _offload_pipeline_for_render(pipeline: Any) -> None:
    models = getattr(pipeline, "models", None)
    if models is not None and hasattr(models, "items"):
        for _, model in models.items():
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
    for name in (
        "image_cond_model_ss", "image_cond_model_shape_512",
        "image_cond_model_shape_1024", "image_cond_model_tex_1024",
    ):
        model = getattr(pipeline, name, None)
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
    _empty_cuda_cache()


def _render_and_evaluate(
    *,
    glb_path: Path,
    condition_image: Image.Image,
    camera: Mapping[str, float],
    output_dir: Path,
    blender: str,
    light: str,
    render_resolution: int,
    metric_resolution: int,
    blender_samples: int,
    lpips_net: str,
    metric_device: torch.device,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "original.png"
    render_path = output_dir / "render.png"
    comparison_path = output_dir / "comparison.png"
    metrics_path = output_dir / "metrics.json"
    blender_log = output_dir / "blender.log"
    reference_image = save_metric_reference(
        condition_image, reference_path, render_resolution
    )
    scratch = Path(tempfile.mkdtemp(prefix=".joint_tile_render_", dir=str(output_dir)))
    try:
        helper = ensure_blender_helper(scratch)
        status_path = scratch / "status.json"
        jobs = [{
            "input_glb": str(glb_path),
            "output_png": str(render_path),
            "status_json": str(status_path),
            "transform": PIXAL3D_EXPORTED_GLTF_TO_INTERNAL.tolist(),
            "resolution": int(render_resolution),
            "fov_rad": float(camera["camera_angle_x"]),
            "distance": float(camera["distance"] * camera["mesh_scale"]),
            "samples": int(blender_samples),
            "light_mode": str(light),
        }]
        result = run_blender_jobs(
            blender_executable=blender,
            helper_path=helper,
            jobs=jobs,
            work_dir=scratch,
            log_path=blender_log,
        ).get(str(render_path), {})
        if result.get("status") != "success" or not render_path.is_file():
            raise RuntimeError(f"aligned Blender render failed: {result.get('error')}")
        rendered = save_black_composited_render(render_path, render_resolution)
        save_comparison(reference_image, rendered, comparison_path)
        reference_cpu = load_metric_tensor(
            reference_path, (int(metric_resolution), int(metric_resolution))
        )
        evaluator = LPIPSEvaluator(lpips_net, metric_device)
        metrics = evaluate_render(reference_cpu, render_path, evaluator)
        payload = {
            "status": "success",
            **metrics,
            "light": light,
            "render_resolution": int(render_resolution),
            "metric_resolution": int(metric_resolution),
            "reference": str(reference_path),
            "render": str(render_path),
            "comparison": str(comparison_path),
            "blender_log": str(blender_log),
        }
        atomic_json(metrics_path, payload)
        print(
            f"[metrics] PSNR={metrics['psnr_db']:.4f} "
            f"SSIM={metrics['ssim']:.6f} LPIPS={metrics['lpips']:.6f}"
        )
        return payload
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_experiment(args: argparse.Namespace) -> None:
    if args.tile_size != DEFAULT_TILE_SIZE or args.tile_stride != DEFAULT_TILE_STRIDE:
        raise ValueError("master-union v3 requires tile-size=1024 and tile-stride=512")
    if args.steps != 12:
        raise ValueError("master-union v3 currently requires exactly 12 steps")
    if not math.isclose(float(args.replace_alpha), 1.0, abs_tol=0.0):
        raise ValueError(
            "master-union v3 implements simple hard replacement only; "
            "use --replace-alpha 1.0"
        )

    output_path = Path(args.output).expanduser().resolve()
    trace_dir = Path(args.trace_dir).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    source = Image.open(args.image)
    canonical = pipeline.preprocess_canonical_images(source)
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(trace_dir / "canonical_4096.png")
    image_1024.save(trace_dir / "canonical_1024.png")
    image_512.save(trace_dir / "canonical_512.png")

    camera = _estimate_camera(
        image_1024=image_1024,
        output_dir=trace_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
    )
    print(
        f"[camera] fov={camera['camera_angle_x']:.8f} "
        f"distance={camera['distance']:.8f} mesh_scale={camera['mesh_scale']:.8f}"
    )
    sampler_params = _build_sampler_params(args, pipeline)
    coords32_global, coords64_base, global_lr_flow, _ = _global_initial_support(
        pipeline=pipeline,
        image_512=image_512,
        image_1024=image_1024,
        camera=camera,
        sampler_params=sampler_params,
        seed=int(args.seed),
    )

    # Tile eligibility is determined from the original complete-image C64_base.
    base_projected_norm, base_projected_depth, base_projection_valid = (
        pipeline._project_sparse_coords_to_image_norm(
            image_cond_model=pipeline.image_cond_model_shape_1024,
            coords=coords64_base,
            camera_angle_x=camera["camera_angle_x"],
            distance=camera["distance"],
            mesh_scale=camera["mesh_scale"],
            grid_resolution=GRID_HR,
        )
    )
    boxes = pipeline.build_texture_image_tile_layout(
        canonical_size=CANONICAL_IMAGE_SIZE,
        tile_size=int(args.tile_size),
        tile_stride=int(args.tile_stride),
    )
    base_tile_ids, base_tile_weights, _ = pipeline.assign_texture_tiles(
        base_projected_norm * float(CANONICAL_IMAGE_SIZE),
        boxes,
        canonical_size=CANONICAL_IMAGE_SIZE,
        max_memberships=4,
    )

    requested = _parse_tile_ids(args.tile_ids)
    tile_experts: List[TileExpert] = []
    tile_metadata: List[Dict[str, Any]] = []
    processed = 0
    for tile_id, box in enumerate(boxes):
        if requested is not None and tile_id not in requested:
            continue
        rows, _ = _tile_rows_and_weights(
            base_tile_ids, base_tile_weights, tile_id
        )
        if rows.numel() < int(args.min_tile_tokens):
            tile_metadata.append({
                "tile_id": tile_id,
                "box": list(box),
                "status": "skipped_min_tokens",
                "input_tokens": int(rows.numel()),
            })
            continue
        if args.max_tiles is not None and processed >= int(args.max_tiles):
            break
        processed += 1
        x0, y0, x1, y1 = box
        tile_image = image_4096.crop((x0, y0, x1, y1)).convert("RGB")
        tile_image.save(trace_dir / f"tile_{tile_id:04d}.png")
        crop_box = (
            x0 / float(CANONICAL_IMAGE_SIZE),
            y0 / float(CANONICAL_IMAGE_SIZE),
            x1 / float(CANONICAL_IMAGE_SIZE),
            y1 / float(CANONICAL_IMAGE_SIZE),
        )
        expert = _prepare_one_tile(
            pipeline=pipeline,
            tile_id=tile_id,
            box=box,
            projection_crop_box=crop_box,
            tile_image_1024=tile_image,
            base_coords64=coords64_base,
            tile_input_rows=rows,
            camera=camera,
            sampler_params=sampler_params,
            base_seed=int(args.seed) + 10_000,
            trace_dir=trace_dir / "tile_support_traces",
            save_lr_trace=not bool(args.no_save_full_tile_traces),
        )
        tile_experts.append(expert)
        tile_metadata.append({
            "tile_id": tile_id,
            "box": list(box),
            "status": "prepared",
            "input_tokens": int(expert.input_base_rows.numel()),
            "local_tokens": int(expert.local_coords64.shape[0]),
            "lr_trace_path": expert.lr_trace_path,
        })
    if not tile_experts:
        raise RuntimeError("no tile passed --min-tile-tokens")

    # Insert every tile-proposed C64 point before the formal 1024 flow.
    coords64_master, master_stats = _build_master_support(
        coords64_base, tile_experts
    )
    print(
        f"[master-support] base={master_stats['base_tokens']:,} "
        f"added={master_stats['added_unique_tokens']:,} "
        f"master={master_stats['master_tokens']:,}"
    )

    # Reproject the union support. These memberships/weights are the only rows
    # a tile may replace; all remaining points fall back to global velocity.
    projected_norm, projected_depth, projection_valid = (
        pipeline._project_sparse_coords_to_image_norm(
            image_cond_model=pipeline.image_cond_model_shape_1024,
            coords=coords64_master,
            camera_angle_x=camera["camera_angle_x"],
            distance=camera["distance"],
            mesh_scale=camera["mesh_scale"],
            grid_resolution=GRID_HR,
        )
    )
    tile_ids, tile_weights, assignment_uv = pipeline.assign_texture_tiles(
        projected_norm * float(CANONICAL_IMAGE_SIZE),
        boxes,
        canonical_size=CANONICAL_IMAGE_SIZE,
        max_memberships=4,
    )
    tile_experts = _bind_tile_experts_to_master(
        tile_experts=tile_experts,
        master_coords64=coords64_master,
        master_tile_ids=tile_ids,
        master_tile_weights=tile_weights,
    )
    if not tile_experts:
        raise RuntimeError("no prepared tile has positive tent-weighted master rows")
    expert_by_id = {expert.tile_id: expert for expert in tile_experts}
    for record in tile_metadata:
        expert = expert_by_id.get(int(record["tile_id"]))
        if expert is None:
            if record.get("status") == "prepared":
                record["status"] = "skipped_no_positive_tent_rows"
            continue
        record["status"] = "complete"
        record["master_rows"] = int(expert.master_rows.numel())
        record["active_tent_rows"] = int(expert.active_master_rows.numel())
    print(
        f"[tile-prepare] usable_experts={len(tile_experts)} processed_tiles={processed}"
    )

    shape_hr_model = pipeline.models["shape_slat_flow_model_1024"]
    global_shape_noise = SparseTensor(
        feats=_randn(
            coords64_master.shape[0],
            int(shape_hr_model.in_channels),
            device=pipeline.device,
            seed=int(args.seed) + 202,
        ),
        coords=coords64_master,
    )
    cond_global_shape = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        coords64_master,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
        grid_resolution_override=GRID_HR,
    )
    shape_online = _run_online_fused_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        flow_model=shape_hr_model,
        global_state=global_shape_noise,
        global_condition=cond_global_shape,
        tile_experts=tile_experts,
        sampler_params=sampler_params["shape"],
        replace_last_n=int(args.shape_replace_last_n),
        stage="shape",
    )
    fused_shape_norm = shape_online.samples
    fused_shape_denorm = _denormalize_sparse(
        fused_shape_norm, pipeline.shape_slat_normalization
    )

    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(
        fused_shape_norm.feats.shape[1]
    )
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channels: {texture_channels}")
    global_texture_noise = SparseTensor(
        feats=_randn(
            coords64_master.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=int(args.seed) + 303,
        ),
        coords=coords64_master,
    )
    cond_global_texture = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024],
        coords64_master,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=camera["mesh_scale"],
        grid_resolution_override=GRID_HR,
    )
    texture_online = _run_online_fused_flow(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        flow_model=texture_model,
        global_state=global_texture_noise,
        global_condition=cond_global_texture,
        tile_experts=tile_experts,
        sampler_params=sampler_params["texture"],
        replace_last_n=int(args.texture_replace_last_n),
        stage="texture",
        global_concat_cond=fused_shape_norm,
    )
    fused_texture_norm = texture_online.samples
    fused_texture_denorm = _denormalize_sparse(
        fused_texture_norm, pipeline.tex_slat_normalization
    )

    global_trace_path = trace_dir / "global_joint_trace.pt"
    temporary_trace = global_trace_path.with_suffix(".pt.tmp")
    torch.save({
        "format": "pixal3d_joint_tile_master_union_fusion_v3",
        "image": str(Path(args.image).expanduser().resolve()),
        "camera": camera,
        "canonical_metadata": canonical["metadata"],
        "sampler_params": sampler_params,
        "coordinate_policy": {
            "grid": 64,
            "base_support_tokens": int(coords64_base.shape[0]),
            "master_support_tokens": int(coords64_master.shape[0]),
            "tile_new_points_inserted_before_1024_flow": True,
            "master_support": "C64_base union all valid C64_local",
            "tile_input_to_lr": "floor(C64_tile_input / 2), unique",
            "tile_local_translation": False,
        },
        "flow_policy": {
            "mode": "single_master_state_online_hard_velocity_replacement",
            "first_steps": "complete-image 1024 condition only",
            "last_steps": "global plus current-state tile forwards",
            "tile_private_1024_state": False,
            "tile_input_state": "current master x_t indexed by tile master rows",
            "uncovered_fallback": "current global velocity",
            "overlap_fusion": "normalized 2D tent mean",
        },
        "coords32_global": coords32_global.detach().cpu(),
        "coords64_base": coords64_base.detach().cpu(),
        "coords64_master": coords64_master.detach().cpu(),
        "master_stats": master_stats,
        "projection": {
            "normalized_xy": projected_norm.detach().cpu(),
            "assignment_uv_4096": assignment_uv.detach().cpu(),
            "depth": projected_depth.detach().cpu(),
            "valid": projection_valid.detach().cpu(),
            "tile_ids": tile_ids.detach().cpu(),
            "tile_weights": tile_weights.detach().cpu(),
            "boxes": [list(box) for box in boxes],
        },
        "tiles": tile_metadata,
        "tile_experts": [
            {
                "tile_id": int(expert.tile_id),
                "local_coords64": expert.local_coords64.detach().cpu(),
                "master_rows": expert.master_rows.detach().cpu(),
                "active_local_rows": expert.active_local_rows.detach().cpu(),
                "active_master_rows": expert.active_master_rows.detach().cpu(),
                "active_weights": expert.active_weights.detach().cpu(),
            }
            for expert in tile_experts
        ],
        "global_shape_512": _serialize_trajectory(global_lr_flow),
        "shape_online": {
            "times": shape_online.times,
            "time_intervals": shape_online.time_intervals,
            "states": shape_online.states,
            "velocities": shape_online.velocities,
            "steps": shape_online.step_records,
            "covered_rows_union": shape_online.covered_rows_union,
        },
        "texture_online": {
            "times": texture_online.times,
            "time_intervals": texture_online.time_intervals,
            "states": texture_online.states,
            "velocities": texture_online.velocities,
            "steps": texture_online.step_records,
            "covered_rows_union": texture_online.covered_rows_union,
        },
        "fused_shape_normalized": fused_shape_norm.feats.detach().cpu(),
        "fused_texture_normalized": fused_texture_norm.feats.detach().cpu(),
        "fused_shape_denormalized": fused_shape_denorm.feats.detach().cpu(),
        "fused_texture_denormalized": fused_texture_denorm.feats.detach().cpu(),
    }, temporary_trace)
    temporary_trace.replace(global_trace_path)
    print(f"[trace] saved={global_trace_path}")

    export_meta: Dict[str, Any] = {}
    metrics: Optional[Dict[str, Any]] = None
    if not args.no_decode:
        export_meta = _export_glb(
            pipeline=pipeline,
            shape_slat=fused_shape_denorm,
            texture_slat=fused_texture_denorm,
            output_path=output_path,
            texture_size=int(args.texture_size),
            decimation_target=int(args.decimation_target),
        )
        if args.render_eval:
            _offload_pipeline_for_render(pipeline)
            metric_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            evaluation_dir = trace_dir / "evaluation" / str(args.light)
            try:
                metrics = _render_and_evaluate(
                    glb_path=output_path,
                    condition_image=image_1024,
                    camera=camera,
                    output_dir=evaluation_dir,
                    blender=str(args.blender),
                    light=str(args.light),
                    render_resolution=int(args.render_resolution),
                    metric_resolution=int(args.metric_resolution),
                    blender_samples=int(args.blender_samples),
                    lpips_net=str(args.lpips_net),
                    metric_device=metric_device,
                )
            except Exception as exc:
                metrics = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "light": str(args.light),
                    "render_resolution": int(args.render_resolution),
                    "metric_resolution": int(args.metric_resolution),
                }
                atomic_json(evaluation_dir / "metrics.json", metrics)
                print(f"[evaluation-error] {metrics['error']}")
                if args.strict_eval:
                    raise
    else:
        print("[Done] --no-decode enabled; master trajectories and fused latents saved")

    summary = {
        "format": "pixal3d_joint_tile_master_union_summary_v3",
        "image": str(Path(args.image).expanduser().resolve()),
        "output": str(output_path),
        "trace": str(global_trace_path),
        "camera": camera,
        "base_global_tokens": int(coords64_base.shape[0]),
        "added_tile_tokens": int(master_stats["added_unique_tokens"]),
        "master_global_tokens": int(coords64_master.shape[0]),
        "processed_tiles": int(processed),
        "usable_tile_experts": int(len(tile_experts)),
        "min_tile_tokens": int(args.min_tile_tokens),
        "shape_replace_last_n": int(args.shape_replace_last_n),
        "texture_replace_last_n": int(args.texture_replace_last_n),
        "replacement": "hard normalized tent tile mean; global fallback",
        "shape_covered_rows": int(shape_online.covered_rows_union),
        "texture_covered_rows": int(texture_online.covered_rows_union),
        "tiles": tile_metadata,
        **export_meta,
        "evaluation": metrics,
    }
    atomic_json(trace_dir / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=512)

    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--tile-ids", type=str, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-tile-tokens", type=int, default=100)
    parser.add_argument("--no-save-full-tile-traces", action="store_true")

    parser.add_argument("--shape-replace-last-n", type=int, default=2)
    parser.add_argument("--texture-replace-last-n", type=int, default=2)
    parser.add_argument("--replace-alpha", type=float, default=1.0,
                        help="Compatibility flag; master-union v3 requires exactly 1.0")

    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--no-decode", action="store_true")

    parser.add_argument("--render-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--light", choices=("studio", "three_point", "softbox", "front", "uniform", "dramatic"), default="studio")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--strict-eval", action="store_true",
                        help="Fail the run if Blender rendering or metric computation fails")
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
