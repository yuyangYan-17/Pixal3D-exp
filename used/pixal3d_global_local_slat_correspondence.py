#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global/local C64 SLat correspondence analysis for Pixal3D.

This file is deliberately an analysis-only route.  It runs one ordinary
1024_cascade baseline, re-voxelizes the projected baseline mesh independently
for every canonical 4096 tile, encodes the local shape SLat, and samples the
local texture SLat from fresh native texture-flow noise.  It never changes a
sampler, adds a correction, fuses a trajectory, or puts local tokens into a
global integer grid.

The correspondence space is continuous normalized object space.  A C64 token
center is first computed in its own C1024/O-Voxel cube and local centers are
then mapped with the repository's exact local-camera -> global-camera
projective transform.  KD-tree neighborhoods therefore operate on physical
3-D centers, not on sparse row order or quantized global coordinates.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global_local_slat_correspondence_v1"
GLOBAL_IMAGE_SIZE = 1024
CANONICAL_IMAGE_SIZE = 4096
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
VOXEL_SIZE = 1.0 / LATENT_RESOLUTION
K_VALUES = (1, 4, 8, 16)
RADIUS_UNITS = (0.5, 1.0, 2.0)
EPSILON = 1e-6 * VOXEL_SIZE


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            _json_value(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _normalize_slat(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    features = value.feats.float()
    mean = torch.as_tensor(
        normalization["mean"], device=features.device, dtype=features.dtype
    ).reshape(1, -1)
    std = torch.as_tensor(
        normalization["std"], device=features.device, dtype=features.dtype
    ).reshape(1, -1)
    if mean.shape[1] != features.shape[1] or std.shape[1] != features.shape[1]:
        raise RuntimeError(
            "latent normalization channel mismatch: "
            f"features={features.shape[1]} mean={mean.shape[1]} std={std.shape[1]}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("latent normalization has non-finite values")
    if bool((std == 0).any().item()):
        raise RuntimeError("latent normalization has a zero standard deviation")
    return value.replace((features - mean) / std)


def _token_xyz(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"expected [N,3] or [N,4] coordinates, got {coords.shape}")
    xyz = coords[:, 1:4] if coords.shape[1] == 4 else coords[:, :3]
    return xyz.to(torch.float64)


def _token_centers_object(coords: torch.Tensor) -> torch.Tensor:
    """Continuous center in the token's own normalized [-0.5, 0.5] cube."""
    xyz = _token_xyz(coords)
    if xyz.numel() and (
        float(xyz.min().item()) < 0.0
        or float(xyz.max().item()) >= float(LATENT_RESOLUTION)
    ):
        raise RuntimeError("C64 coordinates are outside [0, 64)")
    return -0.5 + (xyz + 0.5) / float(LATENT_RESOLUTION)


def _map_local_centers_to_global(
    local_coords: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Map local C64 centers through the exact continuous camera transform."""
    local_object = _token_centers_object(local_coords)
    local_q = local_object * (2.0 * float(transform.mesh_scale))
    global_q, _ = core._local_q_to_global_q(
        local_q.to(torch.float32),
        global_camera=global_camera,
        transform=transform,
    )
    global_object = global_q.to(torch.float64) / (
        2.0 * float(global_camera["mesh_scale"])
    )
    local_q_roundtrip, _ = core._global_q_to_local_q(
        global_q,
        global_camera=global_camera,
        transform=transform,
    )
    error = (local_q_roundtrip - local_q.to(local_q_roundtrip.dtype)).abs()
    if not torch.isfinite(global_object).all():
        raise RuntimeError("continuous local->global center mapping is non-finite")
    stats = {
        "local_to_global_q_roundtrip_max_abs_error": float(error.max().item())
        if error.numel()
        else 0.0,
        "local_to_global_q_roundtrip_mean_abs_error": float(error.mean().item())
        if error.numel()
        else 0.0,
        "global_center_min": float(global_object.min().item())
        if global_object.numel()
        else 0.0,
        "global_center_max": float(global_object.max().item())
        if global_object.numel()
        else 0.0,
    }
    return local_object, global_object, stats


def _sampler_params(args: argparse.Namespace) -> Tuple[Dict[str, Any], ...]:
    return (
        {
            "steps": int(args.ss_steps),
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        {
            "steps": int(args.shape_steps),
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        {
            "steps": int(args.texture_steps),
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        },
    )


def _tensor_range(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().float()
    if value.numel() == 0:
        return {"min": [], "max": []}
    return {
        "min": value.amin(dim=0).cpu().tolist(),
        "max": value.amax(dim=0).cpu().tolist(),
    }


def _load_cached_global(
    output_dir: Path,
    *,
    shape_channels: int = 32,
    texture_channels: int = 32,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    mesh_path = output_dir / "global_baseline_mesh.pt"
    slat_path = output_dir / "global_baseline_slats.pt"
    if not mesh_path.is_file() or not slat_path.is_file():
        return None
    mesh_payload = torch.load(mesh_path, map_location="cpu", weights_only=False)
    slat_payload = torch.load(slat_path, map_location="cpu", weights_only=False)
    mesh = mesh_payload["mesh"] if isinstance(mesh_payload, Mapping) else mesh_payload
    if not isinstance(slat_payload, Mapping):
        raise RuntimeError(f"invalid global SLat checkpoint {slat_path}")
    for name, channels in (
        ("shape_norm", shape_channels),
        ("texture_norm", texture_channels),
    ):
        value = slat_payload[name]
        if value.ndim != 2 or value.shape[1] != channels:
            raise RuntimeError(
                f"cached {name} has unexpected shape {tuple(value.shape)}"
            )
    print(f"[global] reused baseline mesh and normalized SLat: {output_dir}")
    return mesh, dict(slat_payload)


def _run_global_baseline(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    image_1024: Image.Image,
    output_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    cached = _load_cached_global(output_dir)
    if cached is not None and bool(args.resume):
        return cached

    _seed_everything(int(args.global_seed))
    ss_params, shape_params, texture_params = _sampler_params(args)
    started = time.perf_counter()
    print("[global] ordinary Pixal3D 1024_cascade baseline")
    output, latents = pipeline.run(
        image_1024,
        camera_params=args.global_camera,
        seed=int(args.global_seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    if len(output) != 1:
        raise RuntimeError(f"global baseline returned {len(output)} meshes")
    mesh = core._validate_mesh(output[0], "global 1024_cascade baseline")
    shape_raw, texture_raw, resolution = latents
    if int(resolution) != OVOXEL_RESOLUTION:
        raise RuntimeError(f"baseline resolution={resolution}, expected 1024")
    if not torch.equal(shape_raw.coords, texture_raw.coords):
        raise RuntimeError("global shape/texture SLat supports are not identical")
    shape_norm = _normalize_slat(shape_raw, pipeline.shape_slat_normalization)
    texture_norm = _normalize_slat(texture_raw, pipeline.tex_slat_normalization)
    for name, value in (("shape", shape_norm), ("texture", texture_norm)):
        if value.coords.shape[1] != 4 or value.coords.shape[0] == 0:
            raise RuntimeError(f"global {name} C64 SLat has invalid support")
        if not torch.isfinite(value.feats).all():
            raise RuntimeError(f"global {name} normalized SLat is non-finite")

    mesh_cpu = mesh.to("cpu")
    payload = {
        "format": f"{FORMAT}_global_slats",
        "seed": int(args.global_seed),
        "resolution": int(resolution),
        "coords": shape_norm.coords.detach().cpu().to(torch.int32),
        "shape_raw": shape_raw.feats.detach().float().cpu(),
        "shape_norm": shape_norm.feats.detach().float().cpu(),
        "texture_raw": texture_raw.feats.detach().float().cpu(),
        "texture_norm": texture_norm.feats.detach().float().cpu(),
        "shape_normalization": dict(pipeline.shape_slat_normalization),
        "texture_normalization": dict(pipeline.tex_slat_normalization),
        "elapsed_seconds": float(time.perf_counter() - started),
        "normalized_endpoint": True,
        "coordinate_space": "global C64 sparse support; centers are continuous object space",
    }
    _atomic_torch_save(
        output_dir / "global_baseline_mesh.pt",
        {
            "format": f"{FORMAT}_global_mesh",
            "seed": int(args.global_seed),
            "mesh": mesh_cpu,
        },
    )
    _atomic_torch_save(output_dir / "global_baseline_slats.pt", payload)
    del output, latents, shape_raw, texture_raw, shape_norm, texture_norm, mesh
    _empty_cuda_cache()
    print(
        f"[global] saved tokens={int(payload['coords'].shape[0]):,} "
        f"seconds={payload['elapsed_seconds']:.1f}"
    )
    return mesh_cpu, payload


def _load_tile_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != f"{FORMAT}_tile":
        raise RuntimeError(f"invalid tile checkpoint {path}")
    for key in (
        "shape_norm",
        "texture_norm",
        "shape_coords",
        "texture_coords",
        "global_centers_object",
    ):
        if key not in payload:
            raise RuntimeError(f"tile checkpoint {path} lacks {key}")
    if payload["shape_norm"].shape != payload["texture_norm"].shape:
        raise RuntimeError(f"tile checkpoint {path} shape/texture dimensions differ")
    if not torch.equal(payload["shape_coords"], payload["texture_coords"]):
        raise RuntimeError(f"tile checkpoint {path} supports differ")
    return dict(payload)


def _collect_tile(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    shape_encoder: torch.nn.Module,
    baseline_mesh: Any,
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    output_dir: Path,
    tile_id: int,
    box: Sequence[int],
    face_min: torch.Tensor,
    face_max: torch.Tensor,
    face_finite: torch.Tensor,
) -> Dict[str, Any]:
    tile_dir = output_dir / "tiles" / f"tile_{int(tile_id):02d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tile_dir / "tile_latents.pt"
    cached = _load_tile_checkpoint(checkpoint_path)
    if cached is not None and bool(args.resume):
        record = dict(cached["record"])
        record["resumed"] = True
        print(
            f"[tile {int(tile_id):02d}] reused tokens="
            f"{int(cached['shape_coords'].shape[0]):,}"
        )
        return record

    box_tuple = tuple(int(value) for value in box)
    selected = core._tile_face_ids_from_bbox(
        face_min, face_max, face_finite, box_tuple
    )
    record: Dict[str, Any] = {
        "tile_id": int(tile_id),
        "box": list(box_tuple),
        "row": int(tile_id // 4),
        "column": int(tile_id % 4),
        "projected_bbox_faces": int(selected.shape[0]),
        "status": "started",
    }
    if selected.numel() == 0:
        record.update({"status": "skipped", "reason": "no projected triangle bbox"})
        _atomic_json(tile_dir / "summary.json", record)
        return record

    started = time.perf_counter()
    transform = core._derive_tile_camera(
        tile_id=int(tile_id),
        box=box_tuple,
        global_camera=global_camera,
        extend_pixel=int(args.extend_pixel),
    )
    geometry = core._prepare_tile_geometry(
        global_vertices=baseline_mesh.vertices,
        global_faces=baseline_mesh.faces,
        global_face_min=face_min,
        global_face_max=face_max,
        global_face_finite=face_finite,
        global_camera=global_camera,
        transform=transform,
    )
    local_image = image_4096.crop(box_tuple).convert("RGB")
    local_image.save(tile_dir / "tile_reference.png")

    try:
        shape_raw, shape_encoder_stats = core._encode_local_shape(
            encoder=shape_encoder,
            local_coords=geometry.coords,
            local_dual_vertices=geometry.dual_vertices,
            local_intersected=geometry.intersected,
            device=torch.device("cuda"),
            low_vram=bool(args.low_vram),
        )
        shape_norm = _normalize_slat(shape_raw, pipeline.shape_slat_normalization)
        shape_coords = shape_norm.coords
        if shape_norm.feats.shape[1] != 32:
            raise RuntimeError(
                f"local shape latent has {shape_norm.feats.shape[1]} channels, expected 32"
            )

        local_centers, global_centers, mapping_stats = (
            _map_local_centers_to_global(
                shape_coords.detach().cpu(),
                global_camera=global_camera,
                transform=transform,
            )
        )

        # This is the native texture route: fresh noise, local tile image
        # condition, normalized local shape concat condition, and all sampler
        # steps.  No reference texture latent, prefix/suffix, correction, or
        # velocity/x0 reuse is involved.
        texture_model = pipeline.models["tex_slat_flow_model_1024"]
        texture_channels = int(texture_model.in_channels) - int(
            shape_norm.feats.shape[1]
        )
        if texture_channels != 32:
            raise RuntimeError(
                f"native texture noise channels={texture_channels}, expected 32"
            )
        texture_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [local_image],
            shape_coords,
            camera_angle_x=float(transform.camera_angle_x),
            distance=float(transform.distance),
            mesh_scale=float(transform.mesh_scale),
            grid_resolution_override=LATENT_RESOLUTION,
        )
        local_seed = int(args.local_seed) + int(tile_id) * 100003
        _seed_everything(local_seed)
        texture_noise = SparseTensor(
            torch.randn(
                shape_coords.shape[0],
                texture_channels,
                device=shape_coords.device,
                dtype=torch.float32,
            ),
            shape_coords,
        )
        texture_params = {
            **dict(pipeline.tex_slat_sampler_params),
            "steps": int(args.texture_steps),
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        }
        if bool(args.low_vram):
            texture_model.to(torch.device("cuda"))
        texture_started = time.perf_counter()
        try:
            with torch.no_grad():
                result = pipeline.tex_slat_sampler.sample(
                    texture_model,
                    texture_noise,
                    concat_cond=shape_norm,
                    **texture_condition,
                    **texture_params,
                    verbose=bool(args.verbose_sampler),
                    tqdm_desc=f"tile {int(tile_id):02d} native texture SLat",
                )
        finally:
            if bool(args.low_vram):
                texture_model.cpu()
        _sync_cuda()
        texture_norm = getattr(result, "samples", result)
        if not isinstance(texture_norm, SparseTensor):
            raise RuntimeError("native texture sampler did not return SparseTensor")
        if not torch.equal(texture_norm.coords, shape_coords):
            raise RuntimeError("native texture flow changed local sparse support")
        if not torch.isfinite(texture_norm.feats).all():
            raise RuntimeError("native texture endpoint is non-finite")

        shape_coords_cpu = shape_coords.detach().cpu().to(torch.int32)
        texture_coords_cpu = texture_norm.coords.detach().cpu().to(torch.int32)
        shape_norm_cpu = shape_norm.feats.detach().float().cpu()
        texture_norm_cpu = texture_norm.feats.detach().float().cpu()
        record.update(
            {
                "status": "success",
                "tokens": int(shape_coords_cpu.shape[0]),
                "shape_channels": int(shape_norm_cpu.shape[1]),
                "texture_channels": int(texture_norm_cpu.shape[1]),
                "local_seed": local_seed,
                "shape_encoder": shape_encoder_stats,
                "texture_flow": {
                    "route": "native tex_slat_sampler.sample from fresh noise",
                    "image": "canonical_4096 1024x1024 tile crop",
                    "shape_concat_condition": "normalized local encoder shape SLat",
                    "reference_texture_used": False,
                    "prefix_suffix_used": False,
                    "velocity_or_x0_correction": False,
                    "fusion": False,
                    "sampler": texture_params,
                    "noise_channels": texture_channels,
                    "seconds": float(time.perf_counter() - texture_started),
                    "endpoint_space": "normalized texture SLat",
                },
                "geometry": geometry.stats,
                "coordinate_mapping": mapping_stats,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        payload = {
            "format": f"{FORMAT}_tile",
            "record": record,
            "tile_id": int(tile_id),
            "box": list(box_tuple),
            "transform": asdict(transform),
            "shape_coords": shape_coords_cpu,
            "texture_coords": texture_coords_cpu,
            "shape_norm": shape_norm_cpu,
            "texture_norm": texture_norm_cpu,
            "local_centers_object": local_centers,
            "global_centers_object": global_centers,
            "geometry": {
                "vertices": geometry.vertices,
                "faces": geometry.faces,
                "coords": geometry.coords,
                "dual_vertices": geometry.dual_vertices,
                "intersected": geometry.intersected,
                "selected_global_face_ids": geometry.selected_global_face_ids,
            },
            "normalization": {
                "shape": dict(pipeline.shape_slat_normalization),
                "texture": dict(pipeline.tex_slat_normalization),
            },
            "coordinate_space": (
                "local C64 centers mapped to continuous global normalized object "
                "space; no global C64/C256 quantization"
            ),
        }
        _atomic_torch_save(checkpoint_path, payload)
        _atomic_json(tile_dir / "summary.json", record)
        print(
            f"[tile {int(tile_id):02d}] saved tokens={record['tokens']:,} "
            f"seconds={record['elapsed_seconds']:.1f}"
        )
        return record
    finally:
        for name in (
            "shape_raw",
            "shape_norm",
            "texture_norm",
            "texture_noise",
            "texture_condition",
            "result",
        ):
            if name in locals():
                del locals()[name]
        _empty_cuda_cache()


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] requested={int(args.cuda_device)} current={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image = Path(args.image).expanduser().resolve()
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    canonical = pipeline.preprocess_canonical_images(Image.open(input_image))
    image_4096 = canonical["image_4096"]
    image_1024 = canonical["image_1024"]
    image_512 = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    if "foreground_mask_4096" in canonical:
        canonical["foreground_mask_4096"].save(
            output_dir / "canonical_foreground_mask_4096.png"
        )
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    camera_path = output_dir / "global_camera.json"
    if bool(args.resume) and camera_path.is_file():
        global_camera = json.loads(camera_path.read_text("utf-8"))
    elif args.camera_path:
        global_camera = json.loads(
            Path(args.camera_path).expanduser().read_text("utf-8")
        )
        _atomic_json(camera_path, global_camera)
    else:
        global_camera = core._estimate_camera(
            image_1024=image_1024,
            output_dir=output_dir,
            manual_fov=float(args.fov),
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            moge_model_path=args.moge_model_path,
        )
        _atomic_json(camera_path, global_camera)
    args.global_camera = global_camera

    baseline_mesh, baseline_slats = _run_global_baseline(
        args=args,
        pipeline=pipeline,
        image_1024=image_1024,
        output_dir=output_dir,
    )
    global_coords = baseline_slats["coords"]
    global_centers = _token_centers_object(global_coords)
    _atomic_torch_save(
        output_dir / "global_baseline_geometry.pt",
        {
            "coords": global_coords,
            "centers_object": global_centers,
            "format": f"{FORMAT}_global_centers",
        },
    )
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    shape_encoder = pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()

    boxes = core._tile_layout()
    requested = (
        None
        if args.tile_ids is None
        else {int(item.strip()) for item in args.tile_ids.split(",") if item.strip()}
    )
    if requested is not None:
        invalid = sorted(item for item in requested if item not in range(len(boxes)))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; valid ids=0..{len(boxes)-1}")
    records: List[Dict[str, Any]] = []
    for tile_id, box in enumerate(boxes):
        if requested is not None and tile_id not in requested:
            continue
        try:
            record = _collect_tile(
                args=args,
                pipeline=pipeline,
                shape_encoder=shape_encoder,
                baseline_mesh=baseline_mesh,
                global_camera=global_camera,
                image_4096=image_4096,
                output_dir=output_dir,
                tile_id=tile_id,
                box=box,
                face_min=face_min,
                face_max=face_max,
                face_finite=face_finite,
            )
        except Exception as error:
            record = {
                "tile_id": int(tile_id),
                "box": [int(value) for value in box],
                "row": int(tile_id // 4),
                "column": int(tile_id % 4),
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            _atomic_json(
                output_dir / "tiles" / f"tile_{tile_id:02d}" / "summary.json",
                record,
            )
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
            _empty_cuda_cache()
        records.append(record)
        _atomic_json(
            output_dir / "collection_manifest.json",
            {
                "format": FORMAT,
                "image": str(input_image),
                "cuda_device": int(args.cuda_device),
                "global_seed": int(args.global_seed),
                "local_seed": int(args.local_seed),
                "tile_layout": {
                    "canonical_size": CANONICAL_IMAGE_SIZE,
                    "tile_size": int(core.TILE_SIZE),
                    "stride": int(core.TILE_STRIDE),
                    "boxes": [list(item) for item in boxes],
                },
                "continuous_coordinate_space": True,
                "global_c64_voxel_size_object": VOXEL_SIZE,
                "tiles": records,
            },
        )

    manifest = {
        "format": FORMAT,
        "image": str(input_image),
        "cuda_device": int(args.cuda_device),
        "global_seed": int(args.global_seed),
        "local_seed": int(args.local_seed),
        "global_camera": global_camera,
        "global_baseline_slats": str(output_dir / "global_baseline_slats.pt"),
        "global_baseline_mesh": str(output_dir / "global_baseline_mesh.pt"),
        "tile_layout": {
            "canonical_size": CANONICAL_IMAGE_SIZE,
            "tile_size": int(core.TILE_SIZE),
            "stride": int(core.TILE_STRIDE),
            "count": len(boxes),
            "boxes": [list(item) for item in boxes],
        },
        "continuous_coordinate_space": {
            "object_cube": [-0.5, 0.5],
            "global_c64_voxel_size_object": VOXEL_SIZE,
            "local_centers_projected_with": "core._local_q_to_global_q",
            "global_integer_grid_quantization": False,
        },
        "successful_tiles": sum(row.get("status") == "success" for row in records),
        "skipped_tiles": sum(row.get("status") == "skipped" for row in records),
        "failed_tiles": sum(row.get("status") == "failed" for row in records),
        "tiles": records,
    }
    _atomic_json(output_dir / "collection_manifest.json", manifest)
    del shape_encoder, pipeline, baseline_mesh
    _empty_cuda_cache()
    return manifest


def _stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _row_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(
        right_centered, axis=1
    )
    output = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    output[valid] = numerator[valid] / denominator[valid]
    return output


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    output = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    output[valid] = numerator[valid] / denominator[valid]
    return output


def _feature_metrics(
    global_features: np.ndarray,
    aggregate_features: np.ndarray,
) -> Dict[str, Any]:
    if global_features.shape != aggregate_features.shape:
        raise ValueError(
            "feature comparison shape mismatch: "
            f"{global_features.shape} vs {aggregate_features.shape}"
        )
    difference = aggregate_features - global_features
    cosine_values = _cosine(global_features, aggregate_features)
    pearson_values = _row_pearson(global_features, aggregate_features)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    global_rms = float(np.sqrt(np.mean(np.square(global_features))))
    return {
        "tokens": int(global_features.shape[0]),
        "channels": int(global_features.shape[1]),
        "cosine_similarity": _stats(cosine_values),
        "pearson_correlation": {
            "flat": (
                float(
                    np.corrcoef(
                        global_features.reshape(-1),
                        aggregate_features.reshape(-1),
                    )[0, 1]
                )
                if global_features.size > 1
                and np.std(global_features) > 1e-12
                and np.std(aggregate_features) > 1e-12
                else None
            ),
            "per_token": _stats(pearson_values),
        },
        "rmse": rmse,
        "normalized_rmse": rmse / max(global_rms, 1e-12),
        "relative_l2": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(global_features), 1e-12)
        ),
        "_cosine_values": cosine_values,
        "_pearson_values": pearson_values,
    }


def _control_delta(
    matched: Mapping[str, Any],
    random: Mapping[str, Any],
) -> Dict[str, Any]:
    matched_cos = np.asarray(matched["_cosine_values"], dtype=np.float64)
    random_cos = np.asarray(random["_cosine_values"], dtype=np.float64)
    valid_cos = np.isfinite(matched_cos) & np.isfinite(random_cos)
    delta_cos = matched_cos[valid_cos] - random_cos[valid_cos]
    matched_p = np.asarray(matched["_pearson_values"], dtype=np.float64)
    random_p = np.asarray(random["_pearson_values"], dtype=np.float64)
    valid_p = np.isfinite(matched_p) & np.isfinite(random_p)
    delta_p = matched_p[valid_p] - random_p[valid_p]
    if delta_cos.size:
        cos_mean = float(delta_cos.mean())
        cos_se = (
            float(delta_cos.std(ddof=1) / math.sqrt(delta_cos.size))
            if delta_cos.size > 1
            else 0.0
        )
        cos_ci = [cos_mean - 1.96 * cos_se, cos_mean + 1.96 * cos_se]
        cos_z = cos_mean / max(cos_se, 1e-12)
        cos_p = math.erfc(abs(cos_z) / math.sqrt(2.0))
    else:
        cos_mean, cos_ci, cos_p = None, [None, None], None
    return {
        "paired_token_count_cosine": int(delta_cos.size),
        "cosine_matched_minus_random": {
            "mean": cos_mean,
            "ci95_normal_approx": cos_ci,
            "p_two_sided_normal_approx": cos_p,
            "positive_fraction": (
                float(np.mean(delta_cos > 0.0)) if delta_cos.size else None
            ),
        },
        "pearson_matched_minus_random_mean": (
            float(delta_p.mean()) if delta_p.size else None
        ),
        "paired_token_count_pearson": int(delta_p.size),
        "_delta_cosine_values": delta_cos,
    }


def _aggregate_by_indices(
    features: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    selected = features[indices]
    direct = selected.mean(axis=1)
    weights = 1.0 / (distances + EPSILON)
    weighted = np.sum(selected * weights[..., None], axis=1) / np.sum(
        weights, axis=1, keepdims=True
    )
    return direct, weighted


def _random_aggregate(
    global_positions: np.ndarray,
    local_positions: np.ndarray,
    local_features: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.integers(
        0,
        int(local_features.shape[0]),
        size=(int(global_positions.shape[0]), int(count)),
    )
    deltas = local_positions[indices] - global_positions[:, None, :]
    distances = np.linalg.norm(deltas, axis=2)
    direct, weighted = _aggregate_by_indices(local_features, distances, indices)
    return direct, weighted, distances


def _distance_bins(
    distances: np.ndarray,
    similarities: np.ndarray,
    bins: int = 10,
) -> List[Dict[str, Any]]:
    valid = np.isfinite(distances) & np.isfinite(similarities)
    distances = distances[valid]
    similarities = similarities[valid]
    if distances.size == 0:
        return []
    edges = np.quantile(distances, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    output = []
    for index in range(max(1, edges.size - 1)):
        low = float(edges[index])
        high = float(edges[index + 1]) if index + 1 < edges.size else low
        if index + 1 == edges.size - 1:
            mask = (distances >= low) & (distances <= high)
        else:
            mask = (distances >= low) & (distances < high)
        if not np.any(mask):
            continue
        output.append(
            {
                "bin": int(index),
                "distance_low_object": low,
                "distance_high_object": high,
                "distance_low_global_c64_voxel": low / VOXEL_SIZE,
                "distance_high_global_c64_voxel": high / VOXEL_SIZE,
                "count": int(mask.sum()),
                "cosine_mean": float(similarities[mask].mean()),
                "cosine_std": float(similarities[mask].std()),
            }
        )
    return output


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _public(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, tuple):
        return [_public(item) for item in value]
    return value


def _analyze_pool(
    *,
    global_positions: np.ndarray,
    global_features: np.ndarray,
    local_positions: np.ndarray,
    local_features: np.ndarray,
    local_tile_ids: np.ndarray,
    random_seed: int,
    name: str,
) -> Dict[str, Any]:
    if local_positions.shape[0] == 0:
        return {
            "name": name,
            "status": "empty",
            "local_tokens": 0,
            "global_tokens": int(global_positions.shape[0]),
        }
    if local_features.shape[0] != local_positions.shape[0]:
        raise ValueError("local positions and features are not aligned")
    tree = cKDTree(local_positions)
    max_k = min(max(K_VALUES), int(local_positions.shape[0]))
    distances, indices = tree.query(global_positions, k=max_k)
    if max_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    rng = np.random.default_rng(int(random_seed))
    nearest_tiles, nearest_counts = np.unique(
        local_tile_ids[indices[:, 0]], return_counts=True
    )
    result: Dict[str, Any] = {
        "name": name,
        "status": "success",
        "global_tokens": int(global_positions.shape[0]),
        "local_tokens": int(local_positions.shape[0]),
        "local_tile_count": int(np.unique(local_tile_ids).size),
        "nearest_distance_object": _stats(distances[:, 0]),
        "nearest_distance_global_c64_voxel": _stats(distances[:, 0] / VOXEL_SIZE),
        "nearest_neighbor_tile_histogram": {
            str(int(tile)): int(count)
            for tile, count in zip(nearest_tiles, nearest_counts)
        },
        "knn": {},
        "radius": {},
        "distance_similarity": {},
        "_nearest_distance": distances[:, 0],
        "_nearest_tile": local_tile_ids[indices[:, 0]],
    }
    if int(local_positions.shape[0]) < max(K_VALUES):
        result["knn_available_max"] = int(local_positions.shape[0])
    else:
        result["knn_available_max"] = int(max(K_VALUES))

    for k in K_VALUES:
        actual_k = min(int(k), int(local_positions.shape[0]))
        matched_direct, matched_weighted = _aggregate_by_indices(
            local_features,
            distances[:, :actual_k],
            indices[:, :actual_k],
        )
        random_direct, random_weighted, random_distances = _random_aggregate(
            global_positions,
            local_positions,
            local_features,
            actual_k,
            rng,
        )
        matched_direct_metrics = _feature_metrics(
            global_features, matched_direct
        )
        matched_weighted_metrics = _feature_metrics(
            global_features, matched_weighted
        )
        random_direct_metrics = _feature_metrics(global_features, random_direct)
        random_weighted_metrics = _feature_metrics(
            global_features, random_weighted
        )
        result["knn"][str(k)] = {
            "requested_k": int(k),
            "actual_k": int(actual_k),
            "neighbor_count": int(actual_k),
            "coverage": 1.0,
            "mean_neighbor_distance_object": float(
                distances[:, :actual_k].mean()
            ),
            "mean_neighbor_distance_global_c64_voxel": float(
                distances[:, :actual_k].mean() / VOXEL_SIZE
            ),
            "max_neighbor_distance_global_c64_voxel": float(
                distances[:, :actual_k].max() / VOXEL_SIZE
            ),
            "matched": {
                "direct_mean": matched_direct_metrics,
                "inverse_distance_weighted": matched_weighted_metrics,
            },
            "random": {
                "direct_mean": random_direct_metrics,
                "inverse_distance_weighted": random_weighted_metrics,
            },
            "control": {
                "direct_mean": _control_delta(
                    matched_direct_metrics, random_direct_metrics
                ),
                "inverse_distance_weighted": _control_delta(
                    matched_weighted_metrics, random_weighted_metrics
                ),
            },
            "_random_distances": random_distances,
        }

    for radius_units in RADIUS_UNITS:
        radius_object = float(radius_units) * VOXEL_SIZE
        neighborhoods = tree.query_ball_point(global_positions, radius_object)
        counts = np.asarray([len(item) for item in neighborhoods], dtype=np.int64)
        valid_rows = np.flatnonzero(counts > 0)
        matched_direct = np.empty(
            (valid_rows.size, local_features.shape[1]), dtype=np.float64
        )
        matched_weighted = np.empty_like(matched_direct)
        random_direct = np.empty_like(matched_direct)
        random_weighted = np.empty_like(matched_direct)
        rng_radius = np.random.default_rng(
            int(random_seed) + int(round(radius_units * 1000.0)) + 917
        )
        for output_row, global_row in enumerate(valid_rows.tolist()):
            neighbor_indices = np.asarray(
                neighborhoods[global_row], dtype=np.int64
            )
            neighbor_distances = np.linalg.norm(
                local_positions[neighbor_indices] - global_positions[global_row],
                axis=1,
            )
            # The index array is local to this row; distances remain the real
            # physical distances used for inverse-distance weighting.
            selected = local_features[neighbor_indices]
            direct = selected.mean(axis=0)
            weights = 1.0 / (neighbor_distances + EPSILON)
            weighted = np.sum(selected * weights[:, None], axis=0) / weights.sum()
            matched_direct[output_row] = direct
            matched_weighted[output_row] = weighted
            random_count = int(neighbor_indices.size)
            random_indices = rng_radius.integers(
                0, int(local_features.shape[0]), size=random_count
            )
            random_distances = np.linalg.norm(
                local_positions[random_indices] - global_positions[global_row],
                axis=1,
            )
            random_selected = local_features[random_indices]
            random_direct[output_row] = random_selected.mean(axis=0)
            random_weights = 1.0 / (random_distances + EPSILON)
            random_weighted[output_row] = np.sum(
                random_selected * random_weights[:, None], axis=0
            ) / random_weights.sum()
        if valid_rows.size:
            matched_direct_metrics = _feature_metrics(
                global_features[valid_rows], matched_direct
            )
            matched_weighted_metrics = _feature_metrics(
                global_features[valid_rows], matched_weighted
            )
            random_direct_metrics = _feature_metrics(
                global_features[valid_rows], random_direct
            )
            random_weighted_metrics = _feature_metrics(
                global_features[valid_rows], random_weighted
            )
        else:
            matched_direct_metrics = {}
            matched_weighted_metrics = {}
            random_direct_metrics = {}
            random_weighted_metrics = {}
        result["radius"][str(radius_units)] = {
            "radius_global_c64_voxel": float(radius_units),
            "radius_object": radius_object,
            "coverage": float(valid_rows.size / global_positions.shape[0]),
            "neighbor_count": _stats(counts),
            "matched_token_count": int(valid_rows.size),
            "matched": {
                "direct_mean": matched_direct_metrics,
                "inverse_distance_weighted": matched_weighted_metrics,
            },
            "random_count_matched": {
                "direct_mean": random_direct_metrics,
                "inverse_distance_weighted": random_weighted_metrics,
            },
            "control": (
                {
                    "direct_mean": _control_delta(
                        matched_direct_metrics, random_direct_metrics
                    ),
                    "inverse_distance_weighted": _control_delta(
                        matched_weighted_metrics, random_weighted_metrics
                    ),
                }
                if valid_rows.size
                else {}
            ),
        }

    nearest_features = local_features[indices[:, 0]]
    nearest_cosine = _cosine(global_features, nearest_features)
    distance_corr = (
        float(np.corrcoef(distances[:, 0], nearest_cosine)[0, 1])
        if distances.shape[0] > 1
        and np.std(distances[:, 0]) > 1e-12
        and np.std(nearest_cosine) > 1e-12
        else None
    )
    result["distance_similarity"] = {
        "nearest_pair_count": int(nearest_cosine.size),
        "distance_cosine_pearson": distance_corr,
        "distance_object": _stats(distances[:, 0]),
        "cosine": _stats(nearest_cosine),
        "bins": _distance_bins(distances[:, 0], nearest_cosine),
        "_distance": distances[:, 0],
        "_cosine": nearest_cosine,
    }
    return result


def _load_local_items(output_dir: Path) -> List[Dict[str, Any]]:
    manifest = json.loads(
        (output_dir / "collection_manifest.json").read_text("utf-8")
    )
    items = []
    for record in manifest["tiles"]:
        if record.get("status") != "success":
            continue
        path = (
            output_dir
            / "tiles"
            / f"tile_{int(record['tile_id']):02d}"
            / "tile_latents.pt"
        )
        payload = _load_tile_checkpoint(path)
        if payload is None:
            raise FileNotFoundError(f"missing successful tile checkpoint {path}")
        items.append(payload)
    if not items:
        raise RuntimeError("no successful local tile checkpoints")
    return items


def _analyze_latent(
    *,
    output_dir: Path,
    global_payload: Mapping[str, Any],
    local_items: Sequence[Mapping[str, Any]],
    latent_name: str,
    random_seed: int,
) -> Dict[str, Any]:
    del output_dir
    global_features = global_payload[f"{latent_name}_norm"].numpy().astype(
        np.float64, copy=False
    )
    global_positions = _token_centers_object(
        global_payload["coords"]
    ).numpy().astype(np.float64, copy=False)
    local_pools = []
    for item in local_items:
        feature = item[f"{latent_name}_norm"].numpy().astype(
            np.float64, copy=False
        )
        position = item["global_centers_object"].numpy().astype(
            np.float64, copy=False
        )
        tile_id = int(item["tile_id"])
        if feature.shape[0] != position.shape[0]:
            raise RuntimeError(
                f"tile {tile_id} {latent_name} features/centers differ"
            )
        local_pools.append((tile_id, position, feature))
    all_positions = np.concatenate([item[1] for item in local_pools], axis=0)
    all_features = np.concatenate([item[2] for item in local_pools], axis=0)
    all_tile_ids = np.concatenate(
        [
            np.full(item[1].shape[0], item[0], dtype=np.int32)
            for item in local_pools
        ],
        axis=0,
    )
    overall = _analyze_pool(
        global_positions=global_positions,
        global_features=global_features,
        local_positions=all_positions,
        local_features=all_features,
        local_tile_ids=all_tile_ids,
        random_seed=int(random_seed),
        name="all_tiles",
    )
    by_tile = {}
    for tile_id, position, feature in local_pools:
        tile_result = _analyze_pool(
            global_positions=global_positions,
            global_features=global_features,
            local_positions=position,
            local_features=feature,
            local_tile_ids=np.full(position.shape[0], tile_id, dtype=np.int32),
            random_seed=int(random_seed) + int(tile_id) * 1009,
            name=f"tile_{tile_id:02d}",
        )
        by_tile[str(tile_id)] = _public(tile_result)
    return {
        "format": FORMAT,
        "latent": latent_name,
        "feature_space": "fixed model-normalized SLat endpoint",
        "global_tokens": int(global_features.shape[0]),
        "local_tokens_all_tiles": int(all_features.shape[0]),
        "global_c64_voxel_size_object": VOXEL_SIZE,
        "epsilon_object": EPSILON,
        "overall": _public(overall),
        "by_tile": by_tile,
        "_overall_raw": overall,
    }


def _plot_latent_metrics(
    analysis: Mapping[str, Any],
    output_path: Path,
    *,
    latent_name: str,
) -> None:
    overall = analysis["_overall_raw"]
    if overall.get("status") != "success":
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    k_axis = np.asarray(K_VALUES, dtype=np.float64)
    for method, color in (
        ("direct_mean", "tab:blue"),
        ("inverse_distance_weighted", "tab:orange"),
    ):
        matched_cos = [
            overall["knn"][str(k)]["matched"][method][
                "cosine_similarity"
            ]["mean"]
            for k in K_VALUES
        ]
        random_cos = [
            overall["knn"][str(k)]["random"][method][
                "cosine_similarity"
            ]["mean"]
            for k in K_VALUES
        ]
        axes[0, 0].plot(
            k_axis,
            matched_cos,
            marker="o",
            color=color,
            label=f"matched {method}",
        )
        axes[0, 0].plot(
            k_axis,
            random_cos,
            marker="x",
            linestyle="--",
            color=color,
            label=f"random {method}",
        )
        matched_rmse = [
            overall["knn"][str(k)]["matched"][method]["normalized_rmse"]
            for k in K_VALUES
        ]
        random_rmse = [
            overall["knn"][str(k)]["random"][method]["normalized_rmse"]
            for k in K_VALUES
        ]
        axes[0, 1].plot(
            k_axis,
            matched_rmse,
            marker="o",
            color=color,
            label=f"matched {method}",
        )
        axes[0, 1].plot(
            k_axis,
            random_rmse,
            marker="x",
            linestyle="--",
            color=color,
            label=f"random {method}",
        )
    axes[0, 0].set_title(f"{latent_name}: K-neighbor cosine")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].set_ylabel("cosine similarity")
    axes[0, 1].set_title(f"{latent_name}: K-neighbor normalized RMSE")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].set_ylabel("RMSE / global RMS")
    for axis in axes[0]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    radius_axis = np.asarray(RADIUS_UNITS, dtype=np.float64)
    for method, color in (
        ("direct_mean", "tab:green"),
        ("inverse_distance_weighted", "tab:red"),
    ):
        matched = [
            overall["radius"][str(radius)]["matched"][method][
                "cosine_similarity"
            ]["mean"]
            for radius in RADIUS_UNITS
        ]
        random = [
            overall["radius"][str(radius)]["random_count_matched"][method][
                "cosine_similarity"
            ]["mean"]
            for radius in RADIUS_UNITS
        ]
        axes[1, 0].plot(
            radius_axis,
            matched,
            marker="o",
            color=color,
            label=f"matched {method}",
        )
        axes[1, 0].plot(
            radius_axis,
            random,
            marker="x",
            linestyle="--",
            color=color,
            label=f"random {method}",
        )
    coverage = [
        overall["radius"][str(radius)]["coverage"] for radius in RADIUS_UNITS
    ]
    axes[1, 1].plot(
        radius_axis,
        coverage,
        marker="o",
        color="black",
        label="coverage",
    )
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylabel("global-token coverage")
    axes[1, 0].set_title(f"{latent_name}: radius-neighbor cosine")
    axes[1, 0].set_xlabel("radius / global-C64 voxel")
    axes[1, 0].set_ylabel("cosine similarity")
    axes[1, 1].set_title(f"{latent_name}: radius coverage")
    axes[1, 1].set_xlabel("radius / global-C64 voxel")
    for axis in axes[1]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distance_similarity(
    shape_analysis: Mapping[str, Any],
    texture_analysis: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, analysis, label, color in (
        (axes[0], shape_analysis, "shape", "tab:blue"),
        (axes[1], texture_analysis, "texture", "tab:orange"),
    ):
        raw = analysis["_overall_raw"]
        if raw.get("status") != "success":
            axis.set_title(f"{label}: no data")
            continue
        distances = np.asarray(raw["distance_similarity"]["_distance"])
        cosine = np.asarray(raw["distance_similarity"]["_cosine"])
        rng = np.random.default_rng(20260811)
        if distances.size > 20000:
            sample = rng.choice(distances.size, 20000, replace=False)
        else:
            sample = np.arange(distances.size)
        axis.scatter(
            distances[sample] / VOXEL_SIZE,
            cosine[sample],
            s=2,
            alpha=0.15,
            color=color,
            rasterized=True,
        )
        bins = raw["distance_similarity"]["bins"]
        centers = [
            0.5
            * (
                row["distance_low_global_c64_voxel"]
                + row["distance_high_global_c64_voxel"]
            )
            for row in bins
        ]
        means = [row["cosine_mean"] for row in bins]
        axis.plot(centers, means, color="black", marker="o", linewidth=2)
        axis.set_title(f"{label}: nearest distance vs cosine")
        axis.set_xlabel("nearest distance / global-C64 voxel")
        axis.set_ylabel("global vs nearest-local cosine")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    shape_analysis: Mapping[str, Any],
    texture_analysis: Mapping[str, Any],
) -> Path:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    lines = [
        "# Pixal3D SLat global-local correspondence 实验",
        "",
        f"- 输入：{manifest['image']}",
        f"- CUDA device：{manifest['cuda_device']}",
        f"- global seed / local texture seed：{manifest['global_seed']} / "
        f"{manifest['local_seed']}",
        f"- tile：{manifest['tile_layout']['count']} 个 "
        f"{manifest['tile_layout']['tile_size']}×"
        f"{manifest['tile_layout']['tile_size']} tile，"
        f"stride={manifest['tile_layout']['stride']}",
        f"- 成功/跳过/失败：{manifest['successful_tiles']} / "
        f"{manifest['skipped_tiles']} / {manifest['failed_tiles']}",
        "",
        "## 方法核对",
        "",
        "所有比较使用模型固定 mean/std 标准化后的 endpoint。local C64 token "
        "中心先在各自 local C1024 cube 中连续计算，再经现有 local→global "
        "相机映射进入 normalized object space；分析阶段没有把 local token "
        "量化为 global C64/C256 坐标。",
        "",
        "local shape 来自每个裁剪 mesh 的新鲜 C1024 O-Voxel encoder；local "
        "texture 使用对应 1024 tile 图像、local normalized shape concat condition "
        "和原生 texture sampler 从 fresh noise 完整采样。没有 reference texture、"
        "G、prefix/suffix、velocity/x0 correction 或融合。",
        "",
        "## 全物体结果",
        "",
    ]
    for analysis in (shape_analysis, texture_analysis):
        latent = analysis["latent"]
        overall = analysis["overall"]
        lines.append(f"### {latent}")
        lines.append("")
        if overall.get("status") != "success":
            lines.append("无可用数据。")
            lines.append("")
            continue
        lines.append(
            f"- global tokens={overall['global_tokens']}，local tokens="
            f"{overall['local_tokens']}，最近距离中位数="
            f"{fmt(overall['nearest_distance_global_c64_voxel']['p50'])} "
            "global-C64 voxel"
        )
        lines.append(
            "- 最近邻距离与单 token cosine 的 Pearson="
            f"{fmt(overall['distance_similarity']['distance_cosine_pearson'])}"
        )
        distance_bins = overall["distance_similarity"]["bins"]
        if distance_bins:
            lines.append(
                "- 距离分桶 cosine（最近→最远）="
                f"{fmt(distance_bins[0]['cosine_mean'])} → "
                f"{fmt(distance_bins[-1]['cosine_mean'])}"
            )
        for k in (1, 4, 8, 16):
            row = overall["knn"][str(k)]
            delta = row["control"]["direct_mean"][
                "cosine_matched_minus_random"
            ]
            matched = row["matched"]["direct_mean"]
            lines.append(
                f"- K={k} direct mean：matched cosine="
                f"{fmt(matched['cosine_similarity']['mean'])}，"
                f"Pearson(flat)={fmt(matched['pearson_correlation']['flat'])}，"
                f"normalized RMSE={fmt(matched['normalized_rmse'])}；"
                f"inverse-distance cosine="
                f"{fmt(row['matched']['inverse_distance_weighted']['cosine_similarity']['mean'])}；"
                f"相对 random cosine Δ={fmt(delta['mean'])}，95% CI="
                f"[{fmt(delta['ci95_normal_approx'][0])}, "
                f"{fmt(delta['ci95_normal_approx'][1])}]"
            )
        for radius in RADIUS_UNITS:
            row = overall["radius"][str(radius)]
            matched = row["matched"]["direct_mean"]
            delta = row["control"]["direct_mean"][
                "cosine_matched_minus_random"
            ]
            lines.append(
                f"- radius={radius}：coverage={fmt(row['coverage'])}，"
                f"matched cosine={fmt(matched['cosine_similarity']['mean'])}，"
                f"normalized RMSE={fmt(matched['normalized_rmse'])}；"
                f"inverse-distance cosine="
                f"{fmt(row['matched']['inverse_distance_weighted']['cosine_similarity']['mean'])}；"
                f"相对 count-matched random cosine Δ={fmt(delta['mean'])}"
            )
        lines.append("")
    lines.extend(
        [
            "## 结论口径",
            "",
            "本实验只有一个物体；上面的 CI 是 token-level paired normal "
            "approximation，不能替代跨物体统计。判断“空间邻近是否高于随机”应同时看 "
            "matched-minus-random 的区间、Pearson/cosine 和距离分桶曲线，而不是只看 "
            "未校正的 token 数。",
            "",
            "完整 JSON、逐 tile checkpoint 和图表见同目录；distance_similarity.png "
            "给出最近距离—cosine 散点与分桶均值。",
        ]
    )
    report_path = output_dir / "analysis" / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads(
        (output_dir / "collection_manifest.json").read_text("utf-8")
    )
    global_payload = torch.load(
        output_dir / "global_baseline_slats.pt",
        map_location="cpu",
        weights_only=False,
    )
    local_items = _load_local_items(output_dir)
    shape_analysis = _analyze_latent(
        output_dir=output_dir,
        global_payload=global_payload,
        local_items=local_items,
        latent_name="shape",
        random_seed=int(args.random_seed),
    )
    texture_analysis = _analyze_latent(
        output_dir=output_dir,
        global_payload=global_payload,
        local_items=local_items,
        latent_name="texture",
        random_seed=int(args.random_seed) + 1000003,
    )
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(analysis_dir / "shape.json", _public(shape_analysis))
    _atomic_json(analysis_dir / "texture.json", _public(texture_analysis))
    _plot_latent_metrics(
        shape_analysis,
        analysis_dir / "shape_metrics.png",
        latent_name="shape",
    )
    _plot_latent_metrics(
        texture_analysis,
        analysis_dir / "texture_metrics.png",
        latent_name="texture",
    )
    _plot_distance_similarity(
        shape_analysis,
        texture_analysis,
        analysis_dir / "distance_similarity.png",
    )
    report_path = _write_report(
        output_dir,
        manifest=manifest,
        shape_analysis=shape_analysis,
        texture_analysis=texture_analysis,
    )
    summary = {
        "format": FORMAT,
        "collection_manifest": str(output_dir / "collection_manifest.json"),
        "cuda_device": manifest["cuda_device"],
        "successful_tiles": manifest["successful_tiles"],
        "global_normalized_endpoint": str(
            output_dir / "global_baseline_slats.pt"
        ),
        "shape_analysis": str(analysis_dir / "shape.json"),
        "texture_analysis": str(analysis_dir / "texture.json"),
        "plots": {
            "shape_metrics": str(analysis_dir / "shape_metrics.png"),
            "texture_metrics": str(analysis_dir / "texture_metrics.png"),
            "distance_similarity": str(analysis_dir / "distance_similarity.png"),
        },
        "report": str(report_path),
        "question": (
            "Continuous global 3-D center neighborhoods; matched K/radius "
            "aggregation versus count-matched random local tokens"
        ),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[analysis] saved {output_dir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/global_local_slat_correspondence_cuda4",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--local-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=20260811)
    parser.add_argument(
        "--mode", choices=("all", "collect", "analyze"), default="all"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verbose-sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shape-encoder",
        default=str(
            core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"
        ),
    )
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--camera-path", default=None)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
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
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if args.mode in ("all", "collect"):
        image = Path(args.image).expanduser()
        if not image.is_file():
            raise FileNotFoundError(str(image))
        base = Path(args.shape_encoder).expanduser()
        if not Path(f"{base}.json").is_file() or not Path(
            f"{base}.safetensors"
        ).is_file():
            raise FileNotFoundError(f"encoder checkpoint pair missing: {base}")
    for name in (
        "ss_steps",
        "shape_steps",
        "texture_steps",
        "max_num_tokens",
        "face_projection_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.mode in ("all", "collect"):
        collect(args)
    if args.mode in ("all", "analyze"):
        analyze(args)


if __name__ == "__main__":
    main()
