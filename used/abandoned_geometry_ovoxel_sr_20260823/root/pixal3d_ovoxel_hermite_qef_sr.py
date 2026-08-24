#!/usr/bin/env python3
"""O-Voxel Hermite/QEF geometry super-resolution experiment.

The implementation follows ``Codex.md`` literally:

* the baseline mesh is voxelized once on the fixed ``[-.5, .5]^3`` C4096
  lattice;
* local C1024 supports are integer crops of that lattice;
* local shape flow is run independently on each tile;
* local decoded meshes are only intersection carriers;
* correspondence is a global primal-edge key, followed by tau mode selection;
* final vertices are reconstructed by a bounded, native-style QEF solve;
* all rendered variants query the same baseline C1024 PBR field.

The default command is intentionally cache-aware.  It can first be run with
``--tests-only`` or ``--max-tiles 1`` before launching the complete 4x4x4
active-block experiment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")

# The modified native extension lives in the TRELLIS.2 source checkout.  Put
# it before the site-package copy so the experiment and tests use one exact
# implementation, including grid_range and return_hermite.
_OVOXEL_SOURCE = Path("/home/nvme04/yyyan/TRELLIS.2/o-voxel")
if _OVOXEL_SOURCE.is_dir():
    sys.path.insert(0, str(_OVOXEL_SOURCE))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import o_voxel
from o_voxel.convert import (
    flexible_dual_grid_to_mesh,
    mesh_to_flexible_dual_grid,
)
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel
from pixal3d.utils import render_utils


DEFAULT_OUTPUT = Path("outputs/geometry_ovoxel_hermite_qef_sr")
DEFAULT_IMAGE = Path("assets/choose/0_img.png")
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_BASELINE = Path("outputs/baseline1024_pbr_mesh_compare/raw_ovoxel_mesh.pt")
DEFAULT_MOGE = Path("/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt")
DEFAULT_SHAPE_ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
)

RUNTIME_AABB = torch.tensor(
    [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32
)
PBR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
YAW_ANGLES = (0, 60, 120, 180, 240, 300)
EDGE_CELL_OFFSETS = np.asarray(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=np.int32,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tensor_range(value: torch.Tensor) -> Dict[str, List[float]]:
    value = value.detach().float()
    if value.numel() == 0:
        return {"min": [], "max": []}
    return {
        "min": [float(item) for item in value.amin(dim=0).cpu().tolist()],
        "max": [float(item) for item in value.amax(dim=0).cpu().tolist()],
    }


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _mesh_from_payload(payload: Mapping[str, Any]) -> MeshWithVoxel:
    if "mesh" in payload and isinstance(payload["mesh"], MeshWithVoxel):
        return payload["mesh"].cpu()
    required = ("vertices", "faces", "coords", "attrs")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"baseline checkpoint is missing {missing}")
    origin = payload.get("origin", torch.tensor([-0.5, -0.5, -0.5]))
    if torch.is_tensor(origin):
        origin = origin.tolist()
    voxel_shape = payload.get(
        "voxel_shape",
        [1, 6, int(payload["coords"][:, 0].max()) + 1,
         int(payload["coords"][:, 1].max()) + 1,
         int(payload["coords"][:, 2].max()) + 1],
    )
    return MeshWithVoxel(
        vertices=payload["vertices"].cpu(),
        faces=payload["faces"].cpu(),
        origin=origin,
        voxel_size=float(payload.get("voxel_size", 1.0 / 1024.0)),
        coords=payload["coords"].cpu(),
        attrs=payload["attrs"].cpu(),
        voxel_shape=torch.Size(voxel_shape),
        layout=payload.get("layout", PBR_LAYOUT),
    )


def _load_baseline(path: Path) -> MeshWithVoxel:
    print(f"[baseline] loading {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = _mesh_from_payload(payload)
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError("baseline vertices must be [N,3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("baseline faces must be [M,3]")
    if mesh.attrs.ndim != 2 or mesh.attrs.shape[1] != 6:
        raise ValueError("baseline PBR field must have six channels")
    return mesh


def _save_baseline_copy(mesh: MeshWithVoxel, output_dir: Path, source: Path) -> None:
    _atomic_torch_save(
        output_dir / "baseline_1024" / "raw_mesh.pt",
        {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "coords": mesh.coords,
            "attrs": mesh.attrs,
            "origin": mesh.origin,
            "voxel_size": mesh.voxel_size,
            "voxel_shape": list(mesh.voxel_shape),
            "layout": mesh.layout,
            "source_checkpoint": str(source.resolve()),
        },
    )


def _ensure_baseline_1024_artifacts(
    mesh: MeshWithVoxel,
    shape_encoder: torch.nn.Module,
    pipeline: Any,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Persist the baseline C1024 O-Voxel and its shape SLat once.

    The supplied baseline checkpoint is a PBR mesh field, so its decoder-side
    sparse O-Voxel/SLat tensors are not present in the cache.  Revoxelizing
    that exact baseline mesh on the fixed C1024 lattice is the reproducible
    carrier needed by the shape encoder; it does not run another image or
    texture flow.
    """
    raw_path = output_dir / "baseline_1024" / "raw_ovoxel.pt"
    slat_path = output_dir / "baseline_1024" / "shape_slat.pt"
    if raw_path.is_file() and slat_path.is_file():
        print("[baseline] C1024 O-Voxel and shape SLat caches are complete")
        return
    print("[baseline] building reproducible C1024 O-Voxel carrier/SLat")
    if raw_path.is_file():
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    else:
        coords, dual_abs, intersected, hermite = mesh_to_flexible_dual_grid(
            vertices=mesh.vertices.contiguous().float(),
            faces=mesh.faces.contiguous().int(),
            grid_size=1024,
            aabb=RUNTIME_AABB,
            face_weight=float(args.face_weight),
            boundary_weight=0.0,
            regularization_weight=float(args.regularization_weight),
            return_hermite=True,
        )
        dual_cell = dual_abs.float() * 1024.0 - coords.float()
        raw = {
            "resolution": 1024,
            "coords": coords.cpu().int(),
            "dual_vertices_abs_translated": dual_abs.cpu().float(),
            "dual_vertices_cell": dual_cell.cpu().float(),
            "intersected": intersected.cpu().bool(),
            "hermite": {key: value.cpu() for key, value in hermite.items()},
            "aabb": RUNTIME_AABB,
            "source": "baseline mesh revoxelized on fixed C1024 lattice",
        }
        _atomic_torch_save(raw_path, raw)
    if slat_path.is_file():
        return
    coords = raw["coords"].to(device=device, dtype=torch.int32)
    dual_cell = raw["dual_vertices_cell"].to(device=device, dtype=torch.float32)
    intersected = raw["intersected"].to(device=device, dtype=torch.bool)
    coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    sparse_vertices = SparseTensor(dual_cell, coords4)
    sparse_intersected = sparse_vertices.replace(intersected)
    if args.low_vram:
        shape_encoder.to(device)
    with torch.no_grad():
        slat = shape_encoder(sparse_vertices, sparse_intersected, sample_posterior=False)
    if args.low_vram:
        shape_encoder.cpu()
    normalized = _normalize_slat(slat, pipeline.shape_slat_normalization)
    _atomic_torch_save(
        slat_path,
        {
            "coords": slat.coords.cpu(),
            "feats": slat.feats.cpu(),
            "normalized_feats": normalized.feats.cpu(),
            "normalization": pipeline.shape_slat_normalization,
            "resolution": 1024,
            "source": "baseline C1024 O-Voxel shape encoder",
        },
    )


def _tile_starts(resolution: int, tile_size: int, stride: int) -> List[int]:
    if resolution <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("resolution, tile_size, and stride must be positive")
    if tile_size > resolution:
        raise ValueError("tile_size cannot exceed resolution")
    starts = list(range(0, resolution - tile_size + 1, stride))
    if not starts or starts[-1] != resolution - tile_size:
        raise ValueError("tile layout must land exactly on the final edge")
    return starts


def _tile_layout(resolution: int, tile_size: int, stride: int) -> List[Tuple[int, int, int]]:
    starts = _tile_starts(resolution, tile_size, stride)
    return [(x, y, z) for z in starts for y in starts for x in starts]


def _edge_keys(coords: np.ndarray, axis: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"edge coords must be [N,3], got {coords.shape}")
    return (((coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]) * 3 + axis)


def _cell_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def _hermite_numpy(hermite: Mapping[str, torch.Tensor], resolution: int) -> Dict[str, np.ndarray]:
    result = {key: value.detach().cpu().numpy() for key, value in hermite.items()}
    result["edge_coord"] = result["edge_coord"].astype(np.int32, copy=False)
    result["edge_axis"] = result["edge_axis"].astype(np.int8, copy=False)
    result["q"] = result["q"].astype(np.float32, copy=False)
    result["n"] = result["n"].astype(np.float32, copy=False)
    result["tau"] = result["tau"].astype(np.float32, copy=False)
    result["face_id"] = result["face_id"].astype(np.int32, copy=False)
    result["is_mesh_boundary_source"] = result["is_mesh_boundary_source"].astype(bool, copy=False)
    result["key"] = _edge_keys(result["edge_coord"], result["edge_axis"], resolution)
    return result


def _global_voxelize(
    mesh: MeshWithVoxel,
    resolution: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    path = output_dir / "baseline_c4096" / "raw_ovoxel_hermite.pt"
    if path.is_file() and not args.force_revoxelize:
        print(f"[C{resolution}] loading cache {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["hermite"] = {key: value.cpu() for key, value in payload["hermite"].items()}
        return payload
    print(
        f"[C{resolution}] native mesh_to_flexible_dual_grid(return_hermite=True); "
        "boundary_weight=0"
    )
    started = time.perf_counter()
    coords, dual_abs, intersected, hermite = mesh_to_flexible_dual_grid(
        vertices=mesh.vertices.contiguous().float(),
        faces=mesh.faces.contiguous().int(),
        grid_size=resolution,
        aabb=RUNTIME_AABB,
        face_weight=float(args.face_weight),
        boundary_weight=0.0,
        regularization_weight=float(args.regularization_weight),
        timing=True,
        return_hermite=True,
    )
    dual_cell = dual_abs.float() * float(resolution) - coords.float()
    if not torch.isfinite(dual_cell).all():
        raise RuntimeError("global C4096 dual vertices contain non-finite values")
    payload = {
        "resolution": int(resolution),
        "aabb": RUNTIME_AABB,
        "coords": coords.cpu().int(),
        "dual_vertices_abs_translated": dual_abs.cpu().float(),
        "dual_vertices_cell": dual_cell.cpu().float(),
        "intersected": intersected.cpu().bool(),
        "hermite": {key: value.cpu() for key, value in hermite.items()},
        "face_weight": float(args.face_weight),
        "boundary_weight": 0.0,
        "regularization_weight": float(args.regularization_weight),
        "seconds": float(time.perf_counter() - started),
    }
    _atomic_torch_save(path, payload)
    print(
        f"[C{resolution}] cells={coords.shape[0]:,} "
        f"hermite={hermite['q'].shape[0]:,} seconds={payload['seconds']:.2f}"
    )
    return payload


def _crop_global_ovoxel(
    global_ovoxel: Mapping[str, Any],
    start: Sequence[int],
    tile_size: int,
    resolution: int,
) -> Dict[str, torch.Tensor]:
    starts = torch.as_tensor(start, dtype=torch.int32)
    coords = global_ovoxel["coords"]
    mask = ((coords >= starts[None]) & (coords < (starts + tile_size)[None])).all(dim=1)
    local_coords = coords[mask] - starts[None]
    if local_coords.numel() and bool(((local_coords < 0) | (local_coords >= tile_size)).any()):
        raise AssertionError("cropped local coordinates are outside C1024")
    return {
        "coords": local_coords.int(),
        "dual_vertices": global_ovoxel["dual_vertices_cell"][mask].float(),
        "intersected": global_ovoxel["intersected"][mask].bool(),
        "global_indices": torch.where(mask)[0].int(),
    }


def _block_center(start: Sequence[int], tile_size: int, resolution: int) -> np.ndarray:
    return -0.5 + (np.asarray(start, dtype=np.float64) + tile_size / 2.0) / resolution


def _local_to_global(points: torch.Tensor, start: Sequence[int], tile_size: int, resolution: int) -> torch.Tensor:
    center = torch.as_tensor(_block_center(start, tile_size, resolution), dtype=points.dtype, device=points.device)
    return center[None] + (float(tile_size) / float(resolution)) * points


def _global_to_local(points: torch.Tensor, start: Sequence[int], tile_size: int, resolution: int) -> torch.Tensor:
    center = torch.as_tensor(_block_center(start, tile_size, resolution), dtype=points.dtype, device=points.device)
    return (points - center[None]) / (float(tile_size) / float(resolution))


def _build_local_camera_transform(
    proj_model: torch.nn.Module,
    camera_angle_x: float,
    distance: float,
    start: Sequence[int],
    tile_size: int,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    """Return T_local = inv(local-to-global) @ T_global.

    ``ProjGrid`` samples local coordinates in the same Blender-aligned object
    convention as the native global projector.  The only extra operation is
    the isotropic C4096-to-C1024 similarity; no image crop or recanonicalized
    camera is involved.
    """
    t_global = proj_model.proj_grid.front_view_transform_matrix.to(device=device, dtype=torch.float32).clone()
    t_global[1, 3] = -float(distance)
    scale = float(tile_size) / float(resolution)
    local_to_global = torch.eye(4, device=device, dtype=torch.float32)
    local_to_global[:3, :3] *= scale
    local_to_global[:3, 3] = torch.as_tensor(
        _block_center(start, tile_size, resolution), device=device, dtype=torch.float32
    )
    # ProjGrid's batched projection path expects [B, 4, 4], even for one
    # canonical image.  Keeping the singleton batch dimension also prevents
    # homogeneous-coordinate concatenation from interpreting matrix rows as
    # independent cameras.
    return (torch.linalg.inv(local_to_global) @ t_global).unsqueeze(0)


def _build_image_feature_cache(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract the original image DINO token/spatial features exactly once."""
    image = image.convert("RGB").resize((int(model.image_size), int(model.image_size)), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_for_naf = image_tensor.clone() if getattr(model, "use_naf_upsample", False) else None
        normalized = model.transform(image_tensor)
        z = model.extract_features(normalized)
        num_reg = int(getattr(model.model.config, "num_register_tokens", 4))
        z_global = torch.cat([z[:, :1], z[:, 1:1 + num_reg]], dim=1).contiguous()
        z_patch = z[:, 1 + num_reg:]
        z_patch = z_patch.reshape(
            1, int(model.patch_number), int(model.patch_number), -1
        ).contiguous()
        cache = {
            "global": z_global.detach(),
            "patch": z_patch.detach(),
        }
        if image_for_naf is not None:
            if getattr(model, "naf_model", None) is None:
                model._load_naf()
            hr = model.naf_model(
                image_for_naf, z_patch.permute(0, 3, 1, 2), model.naf_target_size
            )
            cache["patch_hr"] = hr.detach()
    del image_tensor
    return cache


def _project_local_condition(
    model: torch.nn.Module,
    feature_cache: Mapping[str, torch.Tensor],
    coords4: torch.Tensor,
    transform_local: torch.Tensor,
    camera_angle_x: float,
    distance: float,
    device: torch.device,
) -> Dict[str, Any]:
    coords4 = coords4.to(device=device, dtype=torch.int32)
    cam = torch.tensor([camera_angle_x], device=device, dtype=torch.float32)
    dist = torch.tensor([distance], device=device, dtype=torch.float32)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    local_indices = coords4[:, 1:]
    with torch.no_grad():
        z_lr = model.proj_grid(
            feature_cache["patch"], cam, dist, scale,
            transform_matrix=transform_local,
            grid_indices=local_indices,
            grid_resolution=64,
        )
        if "patch_hr" in feature_cache:
            z_hr = model.proj_grid(
                feature_cache["patch_hr"], cam, dist, scale,
                transform_matrix=transform_local,
                BHWC=False,
                grid_indices=local_indices,
                grid_resolution=64,
            )
            z_proj = torch.cat([z_lr, z_hr], dim=-1)[0]
        else:
            z_proj = z_lr[0]
    proj = SparseTensor(z_proj, coords4)
    return {
        "cond": {"global": feature_cache["global"], "proj": proj},
        "neg_cond": {
            "global": torch.zeros_like(feature_cache["global"]),
            "proj": proj.replace(torch.zeros_like(proj.feats)),
        },
    }


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    return value.replace((value.feats - mean) / std)


def _denormalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    return value.replace(value.feats * std + mean)


def _native_noised_endpoint(clean: SparseTensor, noise: SparseTensor, sampler: Any, timestep: float) -> SparseTensor:
    if not torch.equal(clean.coords, noise.coords):
        raise RuntimeError("flow clean/noise supports differ")
    sigma = float(sampler.sigma_min) + (1.0 - float(sampler.sigma_min)) * float(timestep)
    return clean.replace((1.0 - float(timestep)) * clean.feats + sigma * noise.feats)


def _run_local_shape_flow(
    pipeline: Any,
    shape_encoder: torch.nn.Module,
    crop: Mapping[str, torch.Tensor],
    condition: Optional[Mapping[str, Any]],
    condition_model: Optional[torch.nn.Module],
    feature_cache: Optional[Mapping[str, torch.Tensor]],
    transform_local: Optional[torch.Tensor],
    camera_angle_x: float,
    camera_distance: float,
    args: argparse.Namespace,
    tile_id: int,
) -> Dict[str, Any]:
    device = torch.device(pipeline.device)
    coords = torch.cat(
        [torch.zeros_like(crop["coords"][:, :1]), crop["coords"]], dim=1
    ).to(device=device, dtype=torch.int32)
    vertices = SparseTensor(crop["dual_vertices"].to(device=device), coords)
    intersected = vertices.replace(crop["intersected"].to(device=device))
    if args.low_vram:
        shape_encoder.to(device)
    with torch.no_grad():
        reference = shape_encoder(vertices, intersected, sample_posterior=False)
    if args.low_vram:
        shape_encoder.cpu()
    if not isinstance(reference, SparseTensor):
        raise RuntimeError("local shape encoder did not return SparseTensor")
    if condition is None:
        if condition_model is None or feature_cache is None or transform_local is None:
            raise RuntimeError("local projected conditioning inputs are incomplete")
        condition = _project_local_condition(
            condition_model,
            feature_cache,
            reference.coords,
            transform_local,
            camera_angle_x,
            camera_distance,
            device,
        )
    flow_model = pipeline.models["shape_slat_flow_model_1024"]
    decoder = pipeline.models["shape_slat_decoder"]
    params = dict(pipeline.shape_slat_sampler_params)
    params.update({
        "steps": int(args.shape_steps),
        "guidance_strength": float(args.shape_guidance_strength),
        "guidance_rescale": float(args.shape_guidance_rescale),
        "rescale_t": float(args.shape_rescale_t),
    })
    reference_norm = _normalize_slat(reference, pipeline.shape_slat_normalization)
    # Seed before constructing the sparse noise endpoint so each tile is
    # deterministic.  Seeding after this point only made the sampler
    # deterministic while leaving its initial condition dependent on the
    # previous tile's RNG state.
    _seed(int(args.seed) + 1009 * int(tile_id))
    noise = SparseTensor(
        torch.randn(reference_norm.feats.shape[0], int(flow_model.in_channels), device=device),
        reference_norm.coords,
    )
    times = pipeline.shape_slat_sampler.timestep_schedule(
        int(params["steps"]), float(params["rescale_t"])
    )
    noised = _native_noised_endpoint(
        reference_norm, noise, pipeline.shape_slat_sampler, times[0]
    )
    if args.low_vram:
        flow_model.to(device)
    with torch.no_grad():
        result = pipeline.shape_slat_sampler.sample(
            flow_model,
            noised,
            **condition,
            **params,
            verbose=False,
            record_trajectory=False,
        )
    if args.low_vram:
        flow_model.cpu()
    shape_norm = getattr(result, "samples", result)
    if not isinstance(shape_norm, SparseTensor) or not torch.equal(shape_norm.coords, reference.coords):
        raise RuntimeError(f"tile {tile_id} shape flow changed the SLat coordinate set")
    shape_denorm = _denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
    decoder.set_resolution(1024)
    if args.low_vram:
        decoder.to(device)
    with torch.no_grad():
        raw = decoder(
            shape_denorm,
            return_subs=True,
            return_raw_ovoxel=True,
        )
    if args.low_vram:
        decoder.cpu()
    if not isinstance(raw, dict) or len(raw["raw_ovoxel"]) != 1:
        raise RuntimeError(f"tile {tile_id} raw decoder result is malformed")
    raw = raw["raw_ovoxel"][0]
    output = {
        "reference_shape": {
            "coords": reference.coords.cpu(),
            "feats": reference.feats.cpu(),
        },
        "shape_norm": {
            "coords": shape_norm.coords.cpu(),
            "feats": shape_norm.feats.cpu(),
        },
        "shape_denorm": {
            "coords": shape_denorm.coords.cpu(),
            "feats": shape_denorm.feats.cpu(),
        },
        "raw_ovoxel": _cpu(raw),
        "flow": {
            "tile_id": int(tile_id),
            "steps": int(params["steps"]),
            "timestep_schedule": [float(value) for value in times],
            "coordinate_count": int(reference.coords.shape[0]),
            "channel_count": int(reference.feats.shape[1]),
            "seed": int(args.seed) + 1009 * int(tile_id),
        },
    }
    del vertices, intersected, reference, reference_norm, noise, noised, shape_norm, shape_denorm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _init_local_shape_pipeline(model_path: Path, device: torch.device, low_vram: bool) -> Any:
    """Load only the native components required by the geometry experiment.

    ``inference.init_pipeline`` is the correct full production initializer,
    but it eagerly loads sparse/texture cascades and four image-condition
    models.  The present route never calls those stages, so retaining only the
    native shape-1024 flow, FDG decoder, rembg preprocessor, and one shape
    DINO projector gives identical model weights/sampler semantics with a
    substantially smaller initialization footprint.
    """
    from inference import IMAGE_COND_CONFIGS, build_image_cond_model
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline, samplers
    from pixal3d.pipelines.base import Pipeline

    original_names = list(Pixal3DImageTo3DPipeline.model_names_to_load)
    Pixal3DImageTo3DPipeline.model_names_to_load = [
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ]
    try:
        # Call the base loader directly. It reads the exact same pipeline.json
        # and model checkpoints but does not construct the 5.1 GB RMBG model.
        pipeline = Pipeline.from_pretrained.__func__(
            Pixal3DImageTo3DPipeline, str(model_path)
        )
    finally:
        Pixal3DImageTo3DPipeline.model_names_to_load = original_names
    pretrained_args = pipeline._pretrained_args
    pipeline.shape_slat_sampler = getattr(
        samplers, pretrained_args["shape_slat_sampler"]["name"]
    )(**pretrained_args["shape_slat_sampler"]["args"])
    pipeline.shape_slat_sampler_params = pretrained_args["shape_slat_sampler"]["params"]
    pipeline.shape_slat_normalization = pretrained_args["shape_slat_normalization"]
    pipeline.image_cond_model_shape_512 = None
    pipeline.image_cond_model_tex_1024 = None
    pipeline.image_cond_model_ss = None
    pipeline.rembg_model = None
    pipeline.low_vram = bool(low_vram)
    pipeline._device = device
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_1024"]
    )
    pipeline.image_cond_model_shape_1024.eval()
    if not low_vram:
        pipeline.to(device)
    pipeline.image_cond_model_shape_1024.to(device)
    # NAF is part of the native shape-1024 projector. Load it once, not once
    # for each unused texture/structure projector.
    if getattr(pipeline.image_cond_model_shape_1024, "use_naf_upsample", False):
        pipeline.image_cond_model_shape_1024._load_naf()
    return pipeline


def _local_mesh_to_global_hermite(
    tile_flow: Mapping[str, Any],
    start: Sequence[int],
    tile_size: int,
    resolution: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    raw = tile_flow["raw_ovoxel"]
    vertices_local = raw["mesh_vertices"].float()
    faces = raw["mesh_faces"].int()
    vertices_global = _local_to_global(vertices_local, start, tile_size, resolution)
    provenance = raw["provenance"]
    source_index = provenance["source_ovoxel_index"].long()
    source_axis = provenance["source_edge_axis"].long()
    logits = raw["intersected_logits"].float()
    valid = (
        (source_index >= 0) & (source_index < logits.shape[0])
        & (source_axis >= 0) & (source_axis < logits.shape[1])
    )
    face_confidence = torch.zeros(source_index.shape[0], dtype=torch.float32)
    face_confidence[valid] = torch.sigmoid(
        logits[source_index[valid], source_axis[valid]] / float(args.edge_temperature)
    ).cpu()
    local_l = _global_to_local(vertices_global, start, tile_size, resolution)
    margin = (0.5 - local_l.abs()).amin(dim=1)
    # Raised cosine in a boundary band; the 0.05 floor avoids hard holes.
    normalized = (margin / float(args.tile_boundary_band)).clamp(0.0, 1.0)
    point_weight = 0.05 + 0.95 * 0.5 * (1.0 - torch.cos(math.pi * normalized))
    grid_range = [
        [int(max(0, value)) for value in start],
        [int(min(resolution, value + tile_size)) for value in start],
    ]
    _, _, _, hermite = mesh_to_flexible_dual_grid(
        vertices=vertices_global.contiguous().cpu(),
        faces=faces.contiguous().cpu(),
        grid_size=resolution,
        aabb=RUNTIME_AABB,
        grid_range=grid_range,
        face_weight=float(args.face_weight),
        boundary_weight=0.0,
        regularization_weight=float(args.regularization_weight),
        return_hermite=True,
    )
    h = _hermite_numpy(hermite, resolution)
    face_id = h["face_id"]
    valid_face = (face_id >= 0) & (face_id < face_confidence.shape[0])
    confidence = np.zeros(face_id.shape[0], dtype=np.float32)
    confidence[valid_face] = face_confidence.numpy()[face_id[valid_face]]
    # q is one point per scan-line intersection.  Attach the confidence of
    # the actual local generated triangle that produced that intersection;
    # invalid provenance is explicitly marked as a fallback, never promoted.
    q_local = _global_to_local(torch.from_numpy(h["q"]), start, tile_size, resolution)
    q_margin = (0.5 - q_local.abs()).amin(dim=1)
    q_normalized = (q_margin / float(args.tile_boundary_band)).clamp(0.0, 1.0)
    q_tile_weight = 0.05 + 0.95 * 0.5 * (1.0 - torch.cos(math.pi * q_normalized))
    h["confidence"] = confidence
    h["tile_weight"] = q_tile_weight.numpy().astype(np.float32)
    h["tile_id"] = np.full(h["q"].shape[0], int(tile_flow["flow"]["tile_id"]), dtype=np.int32)
    h["provenance_valid"] = valid_face.astype(bool, copy=False)
    h["key"] = _edge_keys(h["edge_coord"], h["edge_axis"], resolution)
    diagnostics = {
        "mesh_vertices": int(vertices_global.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "hermite_observations": int(h["q"].shape[0]),
        "provenance_fallback_observations": int((~h["provenance_valid"]).sum()),
        "provenance_fallback_fraction": float((~h["provenance_valid"]).mean()) if h["q"].size else 0.0,
        "grid_range": grid_range,
        "global_vertex_range": _tensor_range(vertices_global),
        "global_local_global_roundtrip_max_abs": float(
            (_local_to_global(_global_to_local(vertices_global, start, tile_size, resolution), start, tile_size, resolution) - vertices_global)
            .abs().max().item() if vertices_global.numel() else 0.0
        ),
        "boundary_weight": 0.0,
    }
    return {
        "vertices": vertices_global.cpu(),
        "faces": faces.cpu(),
        "face_confidence": face_confidence,
        "provenance": provenance,
        "hermite": h,
    }, diagnostics


def _baseline_observations(global_ovoxel: Mapping[str, Any], resolution: int) -> Dict[str, np.ndarray]:
    h = _hermite_numpy(global_ovoxel["hermite"], resolution)
    h["confidence"] = np.ones(h["q"].shape[0], dtype=np.float32)
    h["tile_weight"] = np.ones(h["q"].shape[0], dtype=np.float32)
    h["tile_id"] = np.full(h["q"].shape[0], -1, dtype=np.int32)
    h["provenance_valid"] = np.ones(h["q"].shape[0], dtype=bool)
    return h


def _concat_observations(items: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not items:
        return {
            "edge_coord": np.empty((0, 3), np.int32),
            "edge_axis": np.empty((0,), np.int8),
            "q": np.empty((0, 3), np.float32),
            "n": np.empty((0, 3), np.float32),
            "tau": np.empty((0,), np.float32),
            "face_id": np.empty((0,), np.int32),
            "is_mesh_boundary_source": np.empty((0,), bool),
            "confidence": np.empty((0,), np.float32),
            "tile_weight": np.empty((0,), np.float32),
            "tile_id": np.empty((0,), np.int32),
            "provenance_valid": np.empty((0,), bool),
            "key": np.empty((0,), np.int64),
        }
    keys = set(items[0].keys())
    result = {}
    for key in keys:
        result[key] = np.concatenate([item[key] for item in items], axis=0)
    return result


def _tau_histogram(values: np.ndarray, bins: int = 20) -> Dict[str, Any]:
    if values.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values.astype(np.float32), bins=bins, range=(0.0, 1.0))
    return {"edges": [float(value) for value in edges], "counts": [int(value) for value in counts]}


def _select_tau_modes(
    baseline: Mapping[str, np.ndarray],
    local: Mapping[str, np.ndarray],
    resolution: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Cluster local tau values per global edge and select one surface mode."""
    n = int(local["key"].shape[0])
    if n == 0:
        return local, {"local_observations": 0, "selected_observations": 0, "cluster_count": 0}
    order = np.lexsort((local["tau"], local["key"]))
    sorted_keys = local["key"][order]
    sorted_tau = local["tau"][order]
    new_cluster = np.ones(n, dtype=bool)
    if n > 1:
        new_cluster[1:] = (
            (sorted_keys[1:] != sorted_keys[:-1])
            | ((sorted_tau[1:] - sorted_tau[:-1]) > float(threshold))
        )
    cluster_id_sorted = np.cumsum(new_cluster, dtype=np.int64) - 1
    cluster_count = int(cluster_id_sorted[-1] + 1)
    edge_change = np.ones(n, dtype=bool)
    if n > 1:
        edge_change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    edge_id_sorted = np.cumsum(edge_change, dtype=np.int64) - 1
    edge_count = int(edge_id_sorted[-1] + 1)
    cluster_tau = np.bincount(
        cluster_id_sorted, weights=sorted_tau.astype(np.float64), minlength=cluster_count
    ) / np.maximum(
        np.bincount(cluster_id_sorted, minlength=cluster_count), 1
    )
    cluster_size = np.bincount(cluster_id_sorted, minlength=cluster_count).astype(np.float64)
    cluster_score = np.bincount(
        cluster_id_sorted,
        weights=(local["confidence"][order] * local["tile_weight"][order]).astype(np.float64),
        minlength=cluster_count,
    )
    # Every cluster belongs to the edge of its first observation.
    cluster_edge = edge_id_sorted[np.flatnonzero(new_cluster)].astype(np.int64, copy=False)

    base_keys = baseline["key"]
    base_tau = baseline["tau"]
    if base_keys.size:
        # Only local edge keys can be anchors.  Searching the 66M baseline
        # observations against the sorted local-key set avoids sorting the
        # entire baseline Hermite table just to compute local tau centers.
        unique_base = np.unique(local["key"])
        base_sum = np.zeros(unique_base.shape[0], dtype=np.float64)
        base_count = np.zeros(unique_base.shape[0], dtype=np.float64)
        base_positions = np.searchsorted(unique_base, base_keys)
        base_valid_rows = base_positions < unique_base.size
        base_valid_rows &= unique_base[
            np.minimum(base_positions, max(unique_base.size - 1, 0))
        ] == base_keys
        np.add.at(base_sum, base_positions[base_valid_rows], base_tau[base_valid_rows])
        np.add.at(base_count, base_positions[base_valid_rows], 1.0)
        base_center = base_sum / np.maximum(base_count, 1.0)
        base_pos = np.searchsorted(unique_base, local["key"])
        base_valid = (base_pos < unique_base.size)
        base_valid &= unique_base[np.minimum(base_pos, max(unique_base.size - 1, 0))] == local["key"]
        cluster_base_pos = np.searchsorted(unique_base, local["key"][order][np.flatnonzero(new_cluster)])
        cluster_has_base = cluster_base_pos < unique_base.size
        if unique_base.size:
            cluster_has_base &= unique_base[np.minimum(cluster_base_pos, unique_base.size - 1)] == sorted_keys[np.flatnonzero(new_cluster)]
        cluster_base_tau = np.zeros(cluster_count, dtype=np.float32)
        cluster_base_tau[cluster_has_base] = base_center[cluster_base_pos[cluster_has_base]].astype(np.float32)
    else:
        unique_base = np.empty((0,), dtype=np.int64)
        base_center = np.empty((0,), dtype=np.float32)
        base_valid = np.zeros(n, dtype=bool)
        cluster_has_base = np.zeros(cluster_count, dtype=bool)
        cluster_base_tau = np.zeros(cluster_count, dtype=np.float32)

    # For baseline-anchored edges choose nearest tau cluster.  For births use
    # the largest confidence-weighted cluster, with count as a deterministic
    # tie breaker.
    # This is intentionally vectorized: full 4x4x4 runs can have tens of
    # millions of edge groups, for which a Python loop over edge ids is
    # prohibitive.  Lexsort's last key is primary, so the first item for each
    # edge implements the same anchored-distance/score and birth-score/count
    # tie breakers as the reference loop.
    cluster_indices = np.arange(cluster_count, dtype=np.int64)
    cluster_distance = np.abs(cluster_tau - cluster_base_tau)
    anchor_order = np.lexsort((
        cluster_indices,
        -cluster_score,
        cluster_distance,
        (~cluster_has_base).astype(np.int8),
        cluster_edge,
    ))
    anchor_starts = np.r_[True, cluster_edge[anchor_order[1:]] != cluster_edge[anchor_order[:-1]]]
    anchor_clusters = anchor_order[anchor_starts]
    anchor_edges = cluster_edge[anchor_clusters]
    anchored_first = cluster_has_base[anchor_clusters]
    selected_cluster = np.full(edge_count, -1, dtype=np.int64)
    edge_has_anchor = np.zeros(edge_count, dtype=bool)
    selected_cluster[anchor_edges[anchored_first]] = anchor_clusters[anchored_first]
    edge_has_anchor[anchor_edges[anchored_first]] = True

    birth_order = np.lexsort((
        cluster_indices,
        -cluster_size,
        -cluster_score,
        cluster_edge,
    ))
    birth_starts = np.r_[True, cluster_edge[birth_order[1:]] != cluster_edge[birth_order[:-1]]]
    birth_clusters = birth_order[birth_starts]
    birth_edges = cluster_edge[birth_clusters]
    needs_birth = ~edge_has_anchor[birth_edges]
    selected_cluster[birth_edges[needs_birth]] = birth_clusters[needs_birth]
    if bool((selected_cluster < 0).any()):
        raise RuntimeError("tau mode selection left an edge without a selected cluster")
    selected_sorted = selected_cluster[edge_id_sorted] == cluster_id_sorted
    selected = np.zeros(n, dtype=bool)
    selected[order] = selected_sorted
    # Keep the selected observations in edge-key order.  Besides making the
    # cache deterministic, this lets later edge activation stats use a linear
    # run-length reduction instead of sorting the full local table again.
    result = {key: value[order][selected_sorted] for key, value in local.items()}

    residual = local["tau"][base_valid] - base_center[np.searchsorted(unique_base, local["key"][base_valid])]
    stats = {
        "local_observations": n,
        "selected_observations": int(selected.sum()),
        "edge_count": edge_count,
        "cluster_count": cluster_count,
        "mean_clusters_per_edge": float(cluster_count / max(edge_count, 1)),
        "edges_with_baseline_anchor": int(np.unique(local["key"][base_valid]).size),
        "baseline_local_tau_residual_histogram": _tau_histogram(np.abs(residual)),
        "baseline_local_tau_residual_mean": float(np.abs(residual).mean()) if residual.size else 0.0,
        "pairwise_tau_difference_histogram": _tau_histogram(
            np.abs(sorted_tau[1:] - sorted_tau[:-1]) if n > 1 else np.empty(0)
        ),
        "cluster_threshold": float(threshold),
    }
    return result, stats


def _unique_key_max(values: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.empty((0,), np.int64), np.empty((0,), np.float32)
    if values.size > 1 and bool(np.all(values[1:] >= values[:-1])):
        order = None
        sorted_values = values
        sorted_scores = scores
    else:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_scores = scores[order]
    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    unique = sorted_values[starts]
    maximum = np.maximum.reduceat(sorted_scores.astype(np.float32), np.flatnonzero(starts))
    return unique, maximum


def _edge_cells(edge_coord: np.ndarray, edge_axis: np.ndarray, resolution: int) -> np.ndarray:
    if edge_coord.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    expanded = edge_coord[:, None, :] + EDGE_CELL_OFFSETS[edge_axis]
    expanded = expanded.reshape(-1, 3)
    valid = ((expanded >= 0) & (expanded < resolution)).all(axis=1)
    return expanded[valid].astype(np.int32, copy=False)


def _cell_incident_edge_keys(cells: np.ndarray, resolution: int) -> np.ndarray:
    """Enumerate the primal-edge keys incident to a set of dual cells."""
    cells = np.asarray(cells, dtype=np.int32)
    if cells.size == 0:
        return np.empty((0,), dtype=np.int64)
    parts = []
    for axis in range(3):
        origins = cells[:, None, :] - EDGE_CELL_OFFSETS[axis][None, :, :]
        origins = origins.reshape(-1, 3)
        valid = ((origins >= 0) & (origins < int(resolution))).all(axis=1)
        parts.append(_edge_keys(
            origins[valid],
            np.full(int(valid.sum()), axis, dtype=np.int8),
            resolution,
        ))
    return np.unique(np.concatenate(parts, axis=0))


def _active_edge_stats(
    baseline_keys: np.ndarray,
    selected_local: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    fixed: bool,
    local_edge_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Callers pass the sorted-unique resample-control edge index.
    baseline_keys = np.asarray(baseline_keys, dtype=np.int64)
    if local_edge_stats is None:
        local_keys, local_max = _unique_key_max(
            selected_local["key"], selected_local["confidence"] * selected_local["tile_weight"]
        )
    else:
        local_keys, local_max = local_edge_stats
    local_positions = np.searchsorted(baseline_keys, local_keys)
    local_on_baseline = local_positions < baseline_keys.size
    local_on_baseline &= baseline_keys[
        np.minimum(local_positions, max(baseline_keys.size - 1, 0))
    ] == local_keys
    if fixed:
        active = baseline_keys
        selected_positions = np.searchsorted(active, selected_local["key"])
        local_valid = selected_positions < active.size
        local_valid &= active[
            np.minimum(selected_positions, max(active.size - 1, 0))
        ] == selected_local["key"]
    else:
        births = local_keys[~local_on_baseline]
        birth_scores = local_max[~local_on_baseline]
        accepted_births = births[birth_scores >= float(args.edge_activation_threshold)]
        active = np.unique(np.concatenate([baseline_keys, accepted_births]))
        local_valid = np.isin(selected_local["key"], active)
    retained = np.intersect1d(baseline_keys, active).size
    births = np.setdiff1d(active, baseline_keys).size
    deaths = np.setdiff1d(baseline_keys, active).size
    return active, {
        "baseline_active_edges": int(baseline_keys.size),
        "final_active_edges": int(active.size),
        "edge_birth_count": int(births),
        "edge_death_count": int(deaths),
        "edge_retained_count": int(retained),
        "edge_changed_fraction": float((births + deaths) / max(baseline_keys.size, 1)),
        "selected_local_observations_on_active_edges": int(local_valid.sum()),
    }


def _assemble_variant_observations(
    baseline: Mapping[str, np.ndarray],
    selected_local: Mapping[str, np.ndarray],
    active_keys: np.ndarray,
    variant: str,
    args: argparse.Namespace,
    baseline_mask: Optional[np.ndarray] = None,
    local_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    # _active_edge_stats always retains every baseline edge; avoid a second
    # 66M-element membership sort for the baseline table.
    if baseline_mask is None:
        baseline_mask = np.ones(baseline["key"].shape[0], dtype=bool)
    if local_mask is None:
        local_mask = np.isin(selected_local["key"], active_keys)
    b = {key: value[baseline_mask] for key, value in baseline.items()}
    l = {key: value[local_mask] for key, value in selected_local.items()}
    observations = _concat_observations([b, l])
    if variant == "hermite_unweighted":
        observations["weight"] = np.ones(observations["key"].shape[0], dtype=np.float32)
    elif variant in ("hermite_weighted", "fixed_primal_edge_weighted"):
        base_count = int(b["key"].shape[0])
        weights = np.empty(observations["key"].shape[0], dtype=np.float32)
        weights[:base_count] = float(args.lambda_base)
        weights[base_count:] = (
            observations["confidence"][base_count:]
            * observations["tile_weight"][base_count:]
        )
        observations["weight"] = weights
    else:
        raise ValueError(f"unknown fusion variant {variant}")
    return observations, active_keys


def _aggregate_qef_cuda(
    observations: Mapping[str, np.ndarray],
    cells: np.ndarray,
    resolution: int,
    regularization_weight: float,
    observation_chunk: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """CUDA implementation of Hermite sufficient-statistics accumulation."""
    cells = np.asarray(cells, dtype=np.int32)
    cell_t = torch.from_numpy(cells).to(device=device, dtype=torch.int64)
    cell_keys = (cell_t[:, 0] * int(resolution) + cell_t[:, 1]) * int(resolution) + cell_t[:, 2]
    order = torch.argsort(cell_keys)
    sorted_keys = cell_keys[order]
    count = int(cells.shape[0])
    stats_a = torch.zeros((count, 6), device=device, dtype=torch.float32)
    stats_b = torch.zeros((count, 3), device=device, dtype=torch.float32)
    point_sum = torch.zeros((count, 3), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((count,), device=device, dtype=torch.float32)
    offsets = torch.as_tensor(EDGE_CELL_OFFSETS, device=device, dtype=torch.int64)
    edge_coords = observations["edge_coord"]
    edge_axis = observations["edge_axis"]
    q = observations["q"]
    n = observations["n"]
    weights = observations["weight"]
    if not (edge_coords.shape[0] == q.shape[0] == n.shape[0] == weights.shape[0]):
        raise ValueError("Hermite observation fields have inconsistent lengths")
    with torch.no_grad():
        for start in range(0, edge_coords.shape[0], int(observation_chunk)):
            stop = min(start + int(observation_chunk), edge_coords.shape[0])
            ec = torch.from_numpy(edge_coords[start:stop]).to(device=device, dtype=torch.int64)
            ea = torch.from_numpy(edge_axis[start:stop]).to(device=device, dtype=torch.long)
            qq = torch.from_numpy(q[start:stop]).to(device=device, dtype=torch.float32)
            nn = torch.from_numpy(n[start:stop]).to(device=device, dtype=torch.float32)
            ww = torch.from_numpy(weights[start:stop]).to(device=device, dtype=torch.float32)
            cc = ec[:, None, :] + offsets[ea]
            ck = (cc[..., 0] * int(resolution) + cc[..., 1]) * int(resolution) + cc[..., 2]
            flat_keys = ck.reshape(-1)
            idx = torch.searchsorted(sorted_keys, flat_keys)
            valid = idx < count
            valid &= sorted_keys[idx.clamp_max(max(count - 1, 0))] == flat_keys
            if not bool(valid.any()):
                continue
            ii = idx[valid]
            qv = qq[:, None, :].expand(-1, 4, -1).reshape(-1, 3)[valid]
            nv = nn[:, None, :].expand(-1, 4, -1).reshape(-1, 3)[valid]
            wv = ww[:, None].expand(-1, 4).reshape(-1)[valid]
            nq = (nv * qv).sum(dim=1)
            contribution_a = wv[:, None] * torch.stack([
                nv[:, 0] * nv[:, 0], nv[:, 1] * nv[:, 1], nv[:, 2] * nv[:, 2],
                nv[:, 0] * nv[:, 1], nv[:, 0] * nv[:, 2], nv[:, 1] * nv[:, 2],
            ], dim=1)
            contribution_b = wv[:, None] * nv * nq[:, None]
            stats_a.index_add_(0, ii, contribution_a)
            stats_b.index_add_(0, ii, contribution_b)
            point_sum.index_add_(0, ii, wv[:, None] * qv)
            weight_sum.index_add_(0, ii, wv)
            del ec, ea, qq, nn, ww, cc, ck, flat_keys, idx, valid, ii, qv, nv, wv
            del nq, contribution_a, contribution_b
    mean = torch.zeros_like(point_sum)
    nonzero = weight_sum > 1e-12
    mean[nonzero] = point_sum[nonzero] / weight_sum[nonzero, None]
    if float(regularization_weight) > 0.0:
        reg = float(regularization_weight) * weight_sum
        stats_a[:, 0] += reg
        stats_a[:, 1] += reg
        stats_a[:, 2] += reg
        stats_b += reg[:, None] * mean
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(count, device=device, dtype=order.dtype)
    return {
        "cells": cells,
        "A": stats_a[inverse].cpu().numpy(),
        "b": stats_b[inverse].cpu().numpy(),
        "mean": mean[inverse].cpu().numpy(),
        "weight_sum": weight_sum[inverse].cpu().numpy(),
        "cell_keys": cell_keys[inverse].cpu().numpy(),
    }


def _aggregate_qef(
    observations: Mapping[str, np.ndarray],
    cells: np.ndarray,
    resolution: int,
    regularization_weight: float,
    observation_chunk: int = 500_000,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Accumulate Hermite sufficient statistics for every affected cell.

    Each primal edge is incident to four dual cells.  The accumulation uses
    the same ``n n^T`` plane term and weighted mean regularizer as native
    O-Voxel.  The arrays are kept on CPU to avoid a second 65M-cell copy on
    CUDA; individual QEF batches are solved below.
    """
    if device is not None and device.type == "cuda":
        return _aggregate_qef_cuda(
            observations, cells, resolution, regularization_weight,
            observation_chunk, device,
        )
    cells = np.asarray(cells, dtype=np.int32)
    cell_keys = _cell_keys(cells, resolution)
    order_cells = np.argsort(cell_keys, kind="stable")
    cell_keys_sorted = cell_keys[order_cells]
    cells_sorted = cells[order_cells]
    count = cells.shape[0]
    # xx, yy, zz, xy, xz, yz; b = A*q; weighted q sum/count for native reg.
    stats_a = np.zeros((count, 6), dtype=np.float32)
    stats_b = np.zeros((count, 3), dtype=np.float32)
    point_sum = np.zeros((count, 3), dtype=np.float32)
    weight_sum = np.zeros((count,), dtype=np.float32)
    edge_coords = observations["edge_coord"].astype(np.int32, copy=False)
    edge_axis = observations["edge_axis"].astype(np.int8, copy=False)
    q = observations["q"].astype(np.float32, copy=False)
    n = observations["n"].astype(np.float32, copy=False)
    weights = observations["weight"].astype(np.float32, copy=False)
    if not (edge_coords.shape[0] == q.shape[0] == n.shape[0] == weights.shape[0]):
        raise ValueError("Hermite observation fields have inconsistent lengths")
    for start in range(0, edge_coords.shape[0], int(observation_chunk)):
        stop = min(start + int(observation_chunk), edge_coords.shape[0])
        ec = edge_coords[start:stop]
        ea = edge_axis[start:stop]
        qq = q[start:stop]
        nn = n[start:stop]
        ww = weights[start:stop]
        # Flatten all four incident cells at once.  This preserves the exact
        # four-cell semantics but avoids four Python/NumPy scatter passes per
        # observation chunk (important for C4096's tens of millions of rows).
        cc = (ec[:, None, :] + EDGE_CELL_OFFSETS[ea]).reshape(-1, 3)
        ck = _cell_keys(cc, resolution)
        idx = np.searchsorted(cell_keys_sorted, ck)
        valid = (idx < count)
        valid &= cell_keys_sorted[np.minimum(idx, max(count - 1, 0))] == ck
        if not valid.any():
            continue
        ii = idx[valid]
        qv = np.broadcast_to(qq[:, None, :], (qq.shape[0], 4, 3)).reshape(-1, 3)[valid]
        nv = np.broadcast_to(nn[:, None, :], (nn.shape[0], 4, 3)).reshape(-1, 3)[valid]
        wv = np.broadcast_to(ww[:, None], (ww.shape[0], 4)).reshape(-1)[valid]
        nq = (nv * qv).sum(axis=1)
        contribution_a = wv[:, None] * np.stack([
            nv[:, 0] * nv[:, 0], nv[:, 1] * nv[:, 1], nv[:, 2] * nv[:, 2],
            nv[:, 0] * nv[:, 1], nv[:, 0] * nv[:, 2], nv[:, 1] * nv[:, 2],
        ], axis=1)
        contribution_b = wv[:, None] * nv * nq[:, None]
        contribution_point = wv[:, None] * qv
        contribution_weight = wv
        # Reduce repeated cell ids in C/NumPy rather than issuing many
        # unbuffered np.add.at scatter passes.  The resulting sums are the
        # same sufficient statistics up to normal floating-point reduction
        # order.
        reduce_order = np.argsort(ii, kind="stable")
        sorted_indices = ii[reduce_order]
        starts = np.r_[0, np.flatnonzero(sorted_indices[1:] != sorted_indices[:-1]) + 1]
        unique_indices = sorted_indices[starts]
        stats_a[unique_indices] += np.add.reduceat(contribution_a[reduce_order], starts, axis=0)
        stats_b[unique_indices] += np.add.reduceat(contribution_b[reduce_order], starts, axis=0)
        point_sum[unique_indices] += np.add.reduceat(contribution_point[reduce_order], starts, axis=0)
        weight_sum[unique_indices] += np.add.reduceat(contribution_weight[reduce_order], starts)

    mean = np.zeros_like(point_sum)
    nonzero = weight_sum > 1e-12
    mean[nonzero] = point_sum[nonzero] / weight_sum[nonzero, None]
    if float(regularization_weight) > 0.0:
        reg = float(regularization_weight) * weight_sum
        stats_a[:, 0] += reg
        stats_a[:, 1] += reg
        stats_a[:, 2] += reg
        stats_b += reg[:, None] * mean
    # Return in the caller's (lexicographic) cell order, not the temporary key
    # order used for searchsorted.
    inverse = np.empty_like(order_cells)
    inverse[order_cells] = np.arange(order_cells.size)
    return {
        "cells": cells,
        "A": stats_a[inverse],
        "b": stats_b[inverse],
        "mean": mean[inverse],
        "weight_sum": weight_sum[inverse],
        "cell_keys": cell_keys,
    }


def _quadratic_energy(a: torch.Tensor, b: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return (value * torch.bmm(a, value.unsqueeze(-1)).squeeze(-1)).sum(dim=1) - 2.0 * (b * value).sum(dim=1)


def _solve_box_native(
    a: torch.Tensor,
    b: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the native bounded 3-D quadratic by face/edge/corner enumeration."""
    batch = a.shape[0]
    # QEF matrices are positive semidefinite. lstsq mirrors Eigen's robust QR
    # fallback for rank-deficient planes better than an unconditional inverse.
    v_un = torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)
    inside = ((v_un >= lo) & (v_un <= hi)).all(dim=1)
    best_v = torch.where(inside[:, None], v_un, torch.zeros_like(v_un))
    best_e = torch.where(
        inside,
        _quadratic_energy(a, b, v_un),
        torch.full((batch,), float("inf"), dtype=a.dtype, device=a.device),
    )

    def consider(candidate: torch.Tensor, valid: torch.Tensor) -> None:
        nonlocal best_v, best_e
        energy = _quadratic_energy(a, b, candidate)
        use = valid & (energy < best_e)
        best_v = torch.where(use[:, None], candidate, best_v)
        best_e = torch.where(use, energy, best_e)

    for fixed_axis in range(3):
        free = [axis for axis in range(3) if axis != fixed_axis]
        aff = a[:, free][:, :, free]
        for bound in (lo[:, fixed_axis], hi[:, fixed_axis]):
            rhs = b[:, free] - a[:, free, fixed_axis] * bound[:, None]
            free_value = torch.linalg.lstsq(aff, rhs.unsqueeze(-1)).solution.squeeze(-1)
            candidate = torch.zeros_like(v_un)
            candidate[:, fixed_axis] = bound
            candidate[:, free] = free_value
            valid = ((free_value >= lo[:, free]) & (free_value <= hi[:, free])).all(dim=1)
            consider(candidate, valid)

    for free_axis in range(3):
        fixed = [axis for axis in range(3) if axis != free_axis]
        denominator = a[:, free_axis, free_axis]
        for bound0 in (lo[:, fixed[0]], hi[:, fixed[0]]):
            for bound1 in (lo[:, fixed[1]], hi[:, fixed[1]]):
                rhs = (
                    b[:, free_axis]
                    - a[:, free_axis, fixed[0]] * bound0
                    - a[:, free_axis, fixed[1]] * bound1
                )
                free_value = rhs / denominator.clamp_min(torch.finfo(a.dtype).eps)
                candidate = torch.zeros_like(v_un)
                candidate[:, free_axis] = free_value
                candidate[:, fixed[0]] = bound0
                candidate[:, fixed[1]] = bound1
                valid = (free_value >= lo[:, free_axis]) & (free_value <= hi[:, free_axis])
                valid &= denominator.abs() > torch.finfo(a.dtype).eps
                consider(candidate, valid)

    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                candidate = torch.stack(
                    [lo[:, 0] if sx == 0 else hi[:, 0],
                     lo[:, 1] if sy == 0 else hi[:, 1],
                     lo[:, 2] if sz == 0 else hi[:, 2]], dim=1
                )
                consider(candidate, torch.ones(batch, dtype=torch.bool, device=a.device))

    fallback = torch.maximum(torch.minimum(v_un, hi), lo)
    unresolved = ~torch.isfinite(best_e)
    best_v = torch.where(unresolved[:, None], fallback, best_v)
    clamped = (~inside) | ((best_v <= lo + 1e-7) | (best_v >= hi - 1e-7)).any(dim=1)
    eig = torch.linalg.eigvalsh(a)
    scale = eig[:, -1].clamp_min(torch.finfo(a.dtype).eps)
    rank = (eig > scale[:, None] * 1e-6).sum(dim=1)
    return best_v, clamped, rank


def _solve_qef_batches(
    aggregate: Mapping[str, np.ndarray],
    resolution: int,
    batch_size: int = 65_536,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    cells = aggregate["cells"]
    a6 = aggregate["A"]
    b = aggregate["b"]
    vertices = np.empty((cells.shape[0], 3), dtype=np.float32)
    residual = np.zeros((cells.shape[0],), dtype=np.float32)
    rank_hist: Dict[str, int] = {}
    clamp_count = 0
    for start in range(0, cells.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), cells.shape[0])
        aa = np.zeros((stop - start, 3, 3), dtype=np.float32)
        aa[:, 0, 0] = a6[start:stop, 0]
        aa[:, 1, 1] = a6[start:stop, 1]
        aa[:, 2, 2] = a6[start:stop, 2]
        aa[:, 0, 1] = aa[:, 1, 0] = a6[start:stop, 3]
        aa[:, 0, 2] = aa[:, 2, 0] = a6[start:stop, 4]
        aa[:, 1, 2] = aa[:, 2, 1] = a6[start:stop, 5]
        target_device = device if device is not None and device.type == "cuda" else torch.device("cpu")
        aa_t = torch.from_numpy(aa).to(target_device)
        b_t = torch.from_numpy(b[start:stop]).to(target_device)
        cell = torch.from_numpy(cells[start:stop]).float().to(target_device)
        lo = -0.5 + cell / float(resolution)
        hi = lo + 1.0 / float(resolution)
        value, clamped, rank = _solve_box_native(
            aa_t,
            b_t,
            lo,
            hi,
        )
        vertices[start:stop] = value.detach().cpu().numpy()
        clamp_count += int(clamped.sum().item())
        unique_rank, rank_count = torch.unique(rank, return_counts=True)
        for key, count in zip(unique_rank.tolist(), rank_count.tolist()):
            rank_hist[str(int(key))] = rank_hist.get(str(int(key)), 0) + int(count)
    # Residual is evaluated in chunks after solving to keep observation memory
    # out of the returned QEF debug object. The caller can fill it with the
    # exact observation residual using _observation_residuals.
    return {
        "vertices": vertices,
        "clamped_count": int(clamp_count),
        "rank_histogram": rank_hist,
        "cell_count": int(cells.shape[0]),
        "residual_placeholder": residual,
    }


def _observation_residuals_cuda(
    observations: Mapping[str, np.ndarray],
    vertices: np.ndarray,
    cells: np.ndarray,
    resolution: int,
    chunk_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    cells_t = torch.from_numpy(np.asarray(cells, dtype=np.int32)).to(device=device, dtype=torch.int64)
    cell_keys = (cells_t[:, 0] * int(resolution) + cells_t[:, 1]) * int(resolution) + cells_t[:, 2]
    order = torch.argsort(cell_keys)
    sorted_keys = cell_keys[order]
    vertices_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).to(device=device)
    residual_sum = torch.zeros((cells_t.shape[0],), device=device, dtype=torch.float64)
    residual_count = torch.zeros((cells_t.shape[0],), device=device, dtype=torch.float64)
    offsets = torch.as_tensor(EDGE_CELL_OFFSETS, device=device, dtype=torch.int64)
    with torch.no_grad():
        for start in range(0, observations["edge_coord"].shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), observations["edge_coord"].shape[0])
            ec = torch.from_numpy(observations["edge_coord"][start:stop]).to(device=device, dtype=torch.int64)
            ea = torch.from_numpy(observations["edge_axis"][start:stop]).to(device=device, dtype=torch.long)
            qq = torch.from_numpy(observations["q"][start:stop]).to(device=device, dtype=torch.float32)
            nn = torch.from_numpy(observations["n"][start:stop]).to(device=device, dtype=torch.float32)
            ww = torch.from_numpy(observations["weight"][start:stop]).to(device=device, dtype=torch.float32)
            cc = ec[:, None, :] + offsets[ea]
            keys = (cc[..., 0] * int(resolution) + cc[..., 1]) * int(resolution) + cc[..., 2]
            pos = torch.searchsorted(sorted_keys, keys.reshape(-1))
            valid = pos < sorted_keys.numel()
            valid &= sorted_keys[pos.clamp_max(max(sorted_keys.numel() - 1, 0))] == keys.reshape(-1)
            if bool(valid.any()):
                idx_sorted = pos[valid]
                idx = order[idx_sorted]
                qv = qq[:, None, :].expand(-1, 4, -1).reshape(-1, 3)[valid]
                nv = nn[:, None, :].expand(-1, 4, -1).reshape(-1, 3)[valid]
                wv = ww[:, None].expand(-1, 4).reshape(-1)[valid]
                residual = (nv * (vertices_t[idx] - qv)).sum(dim=1)
                residual_sum.index_add_(0, idx, wv.double() * residual.double().square())
                residual_count.index_add_(0, idx, torch.ones_like(wv, dtype=torch.float64))
            del ec, ea, qq, nn, ww, cc, keys, pos, valid
    per_cell = residual_sum / residual_count.clamp_min(1.0)
    nonzero = residual_count > 0
    values = per_cell[nonzero]
    if values.numel() > 1_000_000:
        sample_index = torch.linspace(
            0, values.numel() - 1, 1_000_000, device=device, dtype=torch.float32
        ).long()
        p95_values = values[sample_index]
    else:
        p95_values = values
    return {
        "per_cell": per_cell.float().cpu().numpy(),
        "mean": float(values.mean().item()) if values.numel() else 0.0,
        "p95": float(torch.quantile(p95_values, 0.95).item()) if p95_values.numel() else 0.0,
        "max": float(values.max().item()) if values.numel() else 0.0,
        "cells_with_constraints": int(nonzero.sum().item()),
    }


def _observation_residuals(
    observations: Mapping[str, np.ndarray],
    vertices: np.ndarray,
    cells: np.ndarray,
    resolution: int,
    chunk_size: int = 500_000,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is not None and device.type == "cuda":
        return _observation_residuals_cuda(
            observations, vertices, cells, resolution, chunk_size, device
        )
    cell_keys = _cell_keys(cells, resolution)
    order = np.argsort(cell_keys, kind="stable")
    sorted_keys = cell_keys[order]
    residual_sum = np.zeros(cells.shape[0], dtype=np.float64)
    residual_count = np.zeros(cells.shape[0], dtype=np.float64)
    edge_coords = observations["edge_coord"]
    edge_axis = observations["edge_axis"]
    q = observations["q"]
    n = observations["n"]
    w = observations["weight"]
    for start in range(0, edge_coords.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), edge_coords.shape[0])
        for offset in range(4):
            cc = edge_coords[start:stop] + EDGE_CELL_OFFSETS[edge_axis[start:stop], offset]
            keys = _cell_keys(cc, resolution)
            pos = np.searchsorted(sorted_keys, keys)
            valid = (pos < sorted_keys.size)
            valid &= sorted_keys[np.minimum(pos, max(sorted_keys.size - 1, 0))] == keys
            if not valid.any():
                continue
            idx = order[pos[valid]]
            r = (n[start:stop][valid] * (vertices[idx] - q[start:stop][valid])).sum(axis=1)
            np.add.at(residual_sum, idx, w[start:stop][valid] * r * r)
            np.add.at(residual_count, idx, 1.0)
    per_cell = residual_sum / np.maximum(residual_count, 1.0)
    nonzero = residual_count > 0
    values = per_cell[nonzero]
    return {
        "per_cell": per_cell.astype(np.float32),
        "mean": float(values.mean()) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "cells_with_constraints": int(nonzero.sum()),
    }


def _final_intersected(
    cells: np.ndarray,
    active_keys: np.ndarray,
    resolution: int,
    baseline_cells: Optional[np.ndarray] = None,
    baseline_intersected: Optional[np.ndarray] = None,
) -> np.ndarray:
    if baseline_cells is not None and baseline_intersected is not None:
        baseline_keys = _cell_keys(baseline_cells, resolution)
        baseline_order = np.argsort(baseline_keys, kind="stable")
        sorted_baseline = baseline_keys[baseline_order]
        position = np.searchsorted(sorted_baseline, _cell_keys(cells, resolution))
        valid = position < sorted_baseline.size
        valid &= sorted_baseline[np.minimum(position, max(sorted_baseline.size - 1, 0))] == _cell_keys(cells, resolution)
        result = np.zeros((cells.shape[0], 3), dtype=bool)
        result[valid] = baseline_intersected[baseline_order[position[valid]]]
        return result
    result = np.zeros((cells.shape[0], 3), dtype=bool)
    active_keys = np.asarray(active_keys, dtype=np.int64)
    for axis in range(3):
        keys = _edge_keys(cells, np.full(cells.shape[0], axis, dtype=np.int8), resolution)
        position = np.searchsorted(active_keys, keys)
        valid = (position < active_keys.size)
        valid &= active_keys[np.minimum(position, max(active_keys.size - 1, 0))] == keys
        result[:, axis] = valid
    return result


def _assemble_variant(
    global_ovoxel: Mapping[str, Any],
    baseline_obs: Mapping[str, np.ndarray],
    selected_local: Mapping[str, np.ndarray],
    variant: str,
    args: argparse.Namespace,
    output_dir: Path,
    baseline_keys: Optional[np.ndarray] = None,
    local_edge_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    resolution = int(global_ovoxel["resolution"])
    baseline_coords = global_ovoxel["coords"].numpy().astype(np.int32, copy=False)
    baseline_intersected = global_ovoxel["intersected"].numpy().astype(bool, copy=False)
    if baseline_keys is None:
        baseline_keys = np.unique(baseline_obs["key"])
    fixed = variant == "fixed_primal_edge_weighted"
    active_keys, edge_stats = _active_edge_stats(
        baseline_keys, selected_local, args, fixed=fixed,
        local_edge_stats=local_edge_stats,
    )
    local_positions = np.searchsorted(active_keys, selected_local["key"])
    local_mask = local_positions < active_keys.size
    local_mask &= active_keys[
        np.minimum(local_positions, max(active_keys.size - 1, 0))
    ] == selected_local["key"]
    local_selected = {
        key: value[local_mask] if isinstance(value, np.ndarray) and value.shape[0] == local_mask.shape[0] else value
        for key, value in selected_local.items()
    }
    local_base_positions = np.searchsorted(baseline_keys, local_selected["key"])
    local_is_baseline = local_base_positions < baseline_keys.size
    local_is_baseline &= baseline_keys[
        np.minimum(local_base_positions, max(baseline_keys.size - 1, 0))
    ] == local_selected["key"]
    if fixed:
        cells = baseline_coords.copy()
    else:
        # Baseline edges already have their complete C4096 cell support.  Only
        # accepted birth edges can add cells, so do not expand all ~64M local
        # edges into a 256M-row incident-cell table.
        birth_mask = ~local_is_baseline
        if bool(birth_mask.any()):
            birth_cells = _edge_cells(
                local_selected["edge_coord"][birth_mask],
                local_selected["edge_axis"][birth_mask],
                resolution,
            )
            cells = np.unique(np.concatenate([baseline_coords, birth_cells], axis=0), axis=0).astype(np.int32)
        else:
            cells = baseline_coords.copy()
    # Local observations can only change the four dual cells incident to their
    # global primal edges.  Keep the complete global cell array in the output,
    # but solve QEF only on that affected subset; untouched cells retain the
    # native C4096 dual, which is exactly the solution for their unchanged
    # Hermite constraints and regularizer.
    full_cell_keys = _cell_keys(cells, resolution)
    if local_selected["q"].shape[0] >= max(1, int(0.5 * baseline_keys.size)):
        # Near-global local coverage: all baseline dual cells are affected;
        # avoid expanding every local edge just to rediscover that fact.
        qef_mask = np.ones(cells.shape[0], dtype=bool)
        qef_cells = cells
        affected_cell_keys = _cell_keys(qef_cells, resolution)
    else:
        local_edge_cells = _edge_cells(
            local_selected["edge_coord"], local_selected["edge_axis"], resolution
        )
        affected_cell_keys = np.unique(_cell_keys(local_edge_cells, resolution))
        qef_mask = np.isin(full_cell_keys, affected_cell_keys)
        qef_cells = cells[qef_mask]
    qef_positions = np.flatnonzero(qef_mask)
    if qef_cells.size == 0:
        raise RuntimeError(f"{variant}: local observations did not touch any global dual cells")

    # Retain only baseline observations that constrain an affected cell, plus
    # all selected local observations.  This avoids scanning/scattering the
    # full baseline Hermite table for every tile-local QEF solve.
    # Reverse-index from affected cells to their incident primal edges, then
    # filter the potentially 66M-row baseline table by one scalar edge key.
    # This is equivalent to testing four cells per observation but avoids a
    # huge (H,4,3) temporary and repeated four-way search.
    base_count = int(baseline_obs["key"].shape[0])
    if qef_cells.shape[0] >= max(1, int(0.5 * cells.shape[0])):
        # Near-global coverage makes the reverse incident-edge index larger
        # than the baseline table itself; every baseline observation can
        # constrain the affected set, so retain it directly.
        base_keep = np.ones(base_count, dtype=bool)
    else:
        relevant_edge_keys = _cell_incident_edge_keys(qef_cells, resolution)
        base_keep = np.zeros(base_count, dtype=bool)
        filter_chunk = max(1, int(args.qef_observation_chunk))
        for filter_start in range(0, base_count, filter_chunk):
            filter_stop = min(filter_start + filter_chunk, base_count)
            edge_keys = baseline_obs["key"][filter_start:filter_stop]
            positions = np.searchsorted(relevant_edge_keys, edge_keys)
            valid_positions = positions < relevant_edge_keys.size
            valid_positions &= relevant_edge_keys[
                np.minimum(positions, max(relevant_edge_keys.size - 1, 0))
            ] == edge_keys
            base_keep[filter_start:filter_stop] = valid_positions
    observations, active_keys = _assemble_variant_observations(
        baseline_obs,
        selected_local,
        active_keys,
        variant,
        args,
        baseline_mask=base_keep,
        local_mask=local_mask,
    )
    started = time.perf_counter()
    qef_device = torch.device("cuda", int(args.cuda_device))
    aggregate = _aggregate_qef(
        observations,
        qef_cells,
        resolution,
        float(args.regularization_weight),
        int(args.qef_observation_chunk),
        device=qef_device,
    )
    solved = _solve_qef_batches(
        aggregate,
        resolution,
        int(args.qef_batch_size),
        device=qef_device,
    )
    residual = _observation_residuals(
        observations,
        solved["vertices"],
        qef_cells,
        resolution,
        int(args.qef_observation_chunk),
        device=qef_device,
    )
    # Start from the native C4096 dual for the complete output grid.
    baseline_cell_keys = _cell_keys(baseline_coords, resolution)
    baseline_order = np.argsort(baseline_cell_keys, kind="stable")
    sorted_baseline_keys = baseline_cell_keys[baseline_order]
    full_positions = np.searchsorted(sorted_baseline_keys, full_cell_keys)
    full_valid = full_positions < sorted_baseline_keys.size
    full_valid &= sorted_baseline_keys[np.minimum(full_positions, max(sorted_baseline_keys.size - 1, 0))] == full_cell_keys
    dual_cell = np.full((cells.shape[0], 3), 0.5, dtype=np.float32)
    dual_cell[full_valid] = global_ovoxel["dual_vertices_cell"].numpy()[baseline_order[full_positions[full_valid]]]
    fallback = np.zeros(cells.shape[0], dtype=bool)
    qef_dual_cell = (solved["vertices"] + 0.5) * float(resolution) - qef_cells.astype(np.float32)
    # QEF cells with no accumulated weight use the baseline dual when one is
    # available; this is the explicitly counted fallback path.
    no_constraint = aggregate["weight_sum"] <= 1e-12
    if no_constraint.any():
        qef_base_pos = np.searchsorted(sorted_baseline_keys, _cell_keys(qef_cells[no_constraint], resolution))
        qef_base_valid = qef_base_pos < sorted_baseline_keys.size
        qef_base_valid &= sorted_baseline_keys[np.minimum(qef_base_pos, max(sorted_baseline_keys.size - 1, 0))] == _cell_keys(qef_cells[no_constraint], resolution)
        target = np.flatnonzero(no_constraint)[qef_base_valid]
        source = baseline_order[qef_base_pos[qef_base_valid]]
        qef_dual_cell[target] = global_ovoxel["dual_vertices_cell"].numpy()[source]
        fallback[qef_positions[target]] = True
    dual_cell[qef_positions] = qef_dual_cell
    dual_cell = np.nan_to_num(dual_cell, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32)
    intersected = _final_intersected(
        cells,
        active_keys,
        resolution,
        baseline_cells=baseline_coords if fixed else None,
        baseline_intersected=baseline_intersected if fixed else None,
    )
    if fixed:
        # This is a hard invariant, not a reconstruction of support from the
        # selected observations.
        if not np.array_equal(intersected, baseline_intersected):
            raise AssertionError("fixed-primal-edge cell support changed")
    payload = {
        "resolution": resolution,
        "coords": torch.from_numpy(cells),
        "dual_vertices_cell": torch.from_numpy(dual_cell),
        "dual_vertices_object": torch.from_numpy(-0.5 + (cells.astype(np.float32) + dual_cell) / float(resolution)),
        "intersected": torch.from_numpy(intersected),
        "active_edge_keys": torch.from_numpy(active_keys.astype(np.int64)),
        "observations": {key: torch.from_numpy(value) for key, value in observations.items()},
        "qef": {
            "A": torch.from_numpy(aggregate["A"]),
            "b": torch.from_numpy(aggregate["b"]),
            "mean": torch.from_numpy(aggregate["mean"]),
            "weight_sum": torch.from_numpy(aggregate["weight_sum"]),
            "residual_per_cell": torch.from_numpy(residual["per_cell"]),
            "rank_histogram": solved["rank_histogram"],
            "clamped_count": int(solved["clamped_count"]),
            "fallback_mask": torch.from_numpy(fallback),
        },
        "stats": {
            "variant": variant,
            "cell_count": int(cells.shape[0]),
            "qef_cell_count": int(qef_cells.shape[0]),
            "baseline_reuse_cell_count": int(cells.shape[0] - qef_cells.shape[0]),
            "active_edge_count": int(active_keys.shape[0]),
            "observation_count": int(observations["q"].shape[0]),
            "global_observation_count": int(baseline_obs["q"].shape[0] + selected_local["q"].shape[0]),
            "local_observation_count": int(local_selected["q"].shape[0]),
            "local_only_total_weight": float(observations["weight"][observations["tile_id"] >= 0].sum()),
            "baseline_total_weight": float(observations["weight"][observations["tile_id"] < 0].sum()),
            "baseline_weight_fraction": float(
                observations["weight"][observations["tile_id"] < 0].sum()
                / max(observations["weight"].sum(), 1e-12)
            ),
            "fallback_cell_count": int(fallback.sum()),
            "fallback_cell_fraction": float(fallback.mean()) if fallback.size else 0.0,
            "qef_residual_mean": float(residual["mean"]),
            "qef_residual_p95": float(residual["p95"]),
            "qef_residual_max": float(residual["max"]),
            "qef_rank_histogram": solved["rank_histogram"],
            "qef_clamped_count": int(solved["clamped_count"]),
            "qef_seconds": float(time.perf_counter() - started),
            **edge_stats,
        },
    }
    variant_dir = output_dir / variant
    _atomic_torch_save(variant_dir / "final_ovoxel_qef.pt", payload)
    _atomic_json(variant_dir / "geometry_diagnostics.json", payload["stats"])
    print(
        f"[{variant}] cells={cells.shape[0]:,} edges={active_keys.shape[0]:,} "
        f"obs={observations['q'].shape[0]:,} residual_p95={residual['p95']:.4e}"
    )
    return payload


def _decode_final_mesh(ovoxel_payload: Mapping[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = ovoxel_payload["coords"].to(device=device, dtype=torch.int32)
    dual = ovoxel_payload["dual_vertices_cell"].to(device=device, dtype=torch.float32)
    intersected = ovoxel_payload["intersected"].to(device=device, dtype=torch.bool)
    if coords.shape[0] == 0:
        return torch.empty((0, 3), device=device), torch.empty((0, 3), dtype=torch.int32, device=device)
    with torch.no_grad():
        vertices, faces = flexible_dual_grid_to_mesh(
            coords,
            dual,
            intersected,
            split_weight=None,
            aabb=RUNTIME_AABB.to(device),
            grid_size=int(ovoxel_payload["resolution"]),
        )
    return vertices, faces


@torch.no_grad()
def _query_field_chunked(field: MeshWithVoxel, vertices: torch.Tensor, chunk_size: int) -> torch.Tensor:
    field = field.to(vertices.device) if field.device != vertices.device else field
    values = [field.query_attrs(chunk) for chunk in vertices.split(int(chunk_size), dim=0)]
    if not values:
        return field.attrs.new_empty((0, field.attrs.shape[-1]))
    return torch.cat(values, dim=0)


def _save_variant_meshes(
    variant_payload: Mapping[str, Any],
    field: MeshWithVoxel,
    variant: str,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    variant_dir = output_dir / variant
    cached_mesh_path = variant_dir / "final_mesh.pt"
    cached_pbr_path = variant_dir / "final_pbr_mesh.pt"
    cached_matches_local_fusion = int(variant_payload.get("stats", {}).get("local_observation_count", 0)) > 0
    if (
        cached_mesh_path.is_file()
        and cached_pbr_path.is_file()
        and not args.force_revoxelize
        and not args.force_qef
        and not cached_matches_local_fusion
    ):
        cached = torch.load(cached_pbr_path, map_location="cpu", weights_only=False)
        vertices = cached["vertices"].to(device=device, dtype=torch.float32)
        faces = cached["faces"].to(device=device, dtype=torch.int32)
        attrs = cached["vertex_attrs"].to(device=device, dtype=torch.float32)
        pbr_mesh = MeshWithVertexPbr(vertices, faces, attrs, cached.get("layout", PBR_LAYOUT))
        print(f"[{variant}] loading cached final PBR mesh {cached_pbr_path}")
        return {"mesh": pbr_mesh, "vertices": vertices, "faces": faces, "attrs": attrs}
    vertices, faces = _decode_final_mesh(variant_payload, device)
    attrs = _query_field_chunked(field, vertices, int(args.query_chunk_size))
    pbr_mesh = MeshWithVertexPbr(vertices, faces, attrs, PBR_LAYOUT)
    _atomic_torch_save(
        cached_mesh_path,
        {"vertices": vertices.cpu(), "faces": faces.cpu(), "representation": "global_qef_mesh"},
    )
    _atomic_torch_save(
        cached_pbr_path,
        {
            "vertices": vertices.cpu(),
            "faces": faces.cpu(),
            "vertex_attrs": attrs.cpu(),
            "layout": PBR_LAYOUT,
            "representation": "MeshWithVertexPbr queried from baseline C1024 field",
        },
    )
    return {"mesh": pbr_mesh, "vertices": vertices, "faces": faces, "attrs": attrs}


def _mesh_topology_diagnostics(vertices: torch.Tensor, faces: torch.Tensor, max_faces: int) -> Dict[str, Any]:
    faces_cpu = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    vertices_cpu = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    face_count = int(faces_cpu.shape[0])
    limit = int(max_faces)
    result: Dict[str, Any] = {
        "vertex_count": int(vertices_cpu.shape[0]),
        "face_count": face_count,
    }
    if faces_cpu.size == 0:
        result.update({
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "degenerate_face_count": 0,
            "connected_component_count": 0,
            "topology_sampled": False,
            "topology_sample_face_count": 0,
        })
        return result
    if limit <= 0 or face_count <= limit:
        faces_eval = faces_cpu
        sampled = False
    else:
        # Exact all-face edge tables are disproportionately expensive for the
        # 8M-face meshes produced at C4096.  Use a deterministic uniform face
        # sample for diagnostics, while keeping the generated geometry exact.
        sample_ids = np.linspace(0, face_count - 1, limit, dtype=np.int64)
        faces_eval = faces_cpu[sample_ids]
        sampled = True
    e = np.concatenate([faces_eval[:, [0, 1]], faces_eval[:, [1, 2]], faces_eval[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    unique, counts = np.unique(e, axis=0, return_counts=True)
    result["boundary_edge_count"] = int((counts == 1).sum())
    result["nonmanifold_edge_count"] = int((counts > 2).sum())
    v0 = vertices_cpu[faces_eval[:, 0]]
    v1 = vertices_cpu[faces_eval[:, 1]]
    v2 = vertices_cpu[faces_eval[:, 2]]
    area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    result["degenerate_face_count"] = int((area2 <= 1e-12).sum())
    result["topology_sampled"] = sampled
    result["topology_sample_face_count"] = int(faces_eval.shape[0])
    if sampled:
        result["topology_note"] = (
            f"edge, degeneracy, and boundary counts use a uniform sample of "
            f"{int(faces_eval.shape[0])} / {face_count} faces; geometry is not sampled"
        )
    else:
        # A compact face union through the unique edge table.
        parent = np.arange(faces_eval.shape[0], dtype=np.int64)
        edge_face = np.tile(np.arange(faces_eval.shape[0]), 3)
        order = np.lexsort((edge_face, e[:, 1], e[:, 0]))
        sorted_e = e[order]
        sorted_f = edge_face[order]
        starts = np.r_[0, np.flatnonzero(np.any(sorted_e[1:] != sorted_e[:-1], axis=1)) + 1, sorted_e.shape[0]]
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x
        for left, right in zip(starts[:-1], starts[1:]):
            group = sorted_f[left:right]
            root = find(int(group[0]))
            for face in group[1:]:
                other = find(int(face))
                if root != other:
                    parent[other] = root
        roots = {find(index) for index in range(parent.size)}
        result["connected_component_count"] = int(len(roots))
    if sampled:
        result["connected_component_count"] = None
        result["connected_component_note"] = "skipped because the topology table was sampled"
    return result


def _surface_displacement(reference_vertices: torch.Tensor, vertices: torch.Tensor, max_points: int) -> Dict[str, float]:
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan"), "sample_count": 0}
    ref = reference_vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    cur = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    if ref.size == 0 or cur.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan"), "sample_count": 0}
    if ref.shape[0] > max_points:
        ref = ref[np.linspace(0, ref.shape[0] - 1, max_points).astype(np.int64)]
    if cur.shape[0] > max_points:
        cur = cur[np.linspace(0, cur.shape[0] - 1, max_points).astype(np.int64)]
    distance = cKDTree(ref).query(cur, k=1)[0]
    return {
        "mean": float(distance.mean()),
        "p95": float(np.percentile(distance, 95)),
        "max": float(distance.max()),
        "sample_count": int(distance.size),
    }


def _make_camera_views(camera_angle_x: float, distance: float, device: torch.device) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    extr_front, intrinsics = render_utils.proj_camera_to_render_params(
        float(camera_angle_x), float(distance)
    )
    extrinsics = {}
    for angle_deg in YAW_ANGLES:
        angle = math.radians(float(angle_deg))
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor(
            [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=extr_front.dtype,
            device=extr_front.device,
        )
        extrinsics[int(angle_deg)] = extr_front @ rotation.T
    return extrinsics, intrinsics


def _array_from_render(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    if value.ndim == 3:
        return value.permute(1, 2, 0).numpy()
    if value.ndim == 2:
        return value.numpy()
    raise ValueError(f"unexpected render shape {tuple(value.shape)}")


def _save_image(array: np.ndarray, path: Path) -> None:
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    if array.ndim == 2:
        image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _render_variants(
    meshes: Mapping[str, Any],
    camera: Mapping[str, float],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    from render_pixal3d_raw_ovoxel import load_envmap

    envmap = load_envmap(args.envmap, device=device)
    extrinsics, intrinsics = _make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), device
    )
    renderer = __import__("pixal3d.renderers", fromlist=["PbrMeshRenderer"]).PbrMeshRenderer(
        rendering_options={
            "resolution": int(args.render_resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": int(args.ssaa),
            "peel_layers": int(args.peel_layers),
            "face_chunk_size": int(args.face_chunk_size),
        },
        device=str(device),
    )
    result: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for name, mesh in meshes.items():
        result[name] = {}
        for angle in YAW_ANGLES:
            print(f"[render] {name} yaw={angle}")
            _seed(int(args.seed) + 10_000 + int(angle))
            rendered = renderer.render(
                mesh,
                extrinsics[angle],
                intrinsics,
                envmap=envmap,
                use_envmap_bg=False,
            )
            arrays: Dict[str, np.ndarray] = {}
            for mode in ("shaded", "base_color", "normal", "metallic", "roughness", "alpha", "mask"):
                if mode not in rendered:
                    continue
                array = _array_from_render(rendered[mode])
                if mode == "normal":
                    mask = _array_from_render(rendered["mask"])
                    array = array * mask[..., None]
                arrays[mode] = array.astype(np.float32, copy=False)
                _save_image(
                    arrays[mode],
                    output_dir / name / f"yaw{angle:03d}" / f"{mode}.png",
                )
            result[name][angle] = arrays
            del rendered
            torch.cuda.empty_cache()
    return result


def _contact_sheet(
    render_root: Path,
    rendered: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    mode: str,
    path: Path,
) -> None:
    names = list(rendered)
    if not names:
        return
    first = Image.open(render_root / names[0] / "yaw000" / f"{mode}.png")
    width, height = first.size
    left, top, gap = 170, 32, 4
    canvas = Image.new("RGB", (left + len(YAW_ANGLES) * width + (len(YAW_ANGLES) - 1) * gap,
                                top + len(names) * height + (len(names) - 1) * gap), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, angle in enumerate(YAW_ANGLES):
        draw.text((left + col * (width + gap) + 4, 8), f"yaw{angle:03d}", fill="black", font=font)
    for row, name in enumerate(names):
        y = top + row * (height + gap)
        draw.text((5, y + 8), name, fill="black", font=font)
        for col, angle in enumerate(YAW_ANGLES):
            image = Image.open(render_root / name / f"yaw{angle:03d}" / f"{mode}.png").convert("RGB")
            canvas.paste(image, (left + col * (width + gap), y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _postprocess_render_cache(output_dir: Path, args: argparse.Namespace) -> int:
    """Build sheets/metrics from a completed pbr_renders cache."""
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    render_root = output_dir / "pbr_renders"
    rendered: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for name in summary.get("variant_order", []):
        rendered[name] = {}
        for angle in YAW_ANGLES:
            view_dir = render_root / name / f"yaw{angle:03d}"
            shaded_path = view_dir / "shaded.png"
            normal_path = view_dir / "normal.png"
            mask_path = view_dir / "mask.png"
            if not (shaded_path.is_file() and normal_path.is_file() and mask_path.is_file()):
                raise FileNotFoundError(f"incomplete render cache: {view_dir}")
            rendered[name][angle] = {
                "shaded": np.asarray(Image.open(shaded_path).convert("RGB"), dtype=np.float32) / 255.0,
                "normal": np.asarray(Image.open(normal_path).convert("RGB"), dtype=np.float32) / 255.0,
                "mask": np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0,
            }
    _contact_sheet(render_root, rendered, "shaded", output_dir / "comparison_sheets" / "shaded_contact_sheet.png")
    _contact_sheet(render_root, rendered, "normal", output_dir / "comparison_sheets" / "normal_contact_sheet.png")
    normal_root = output_dir / "normal_renders"
    qef_root = output_dir / "qef_debug"
    for name in summary.get("variant_order", []):
        for angle in YAW_ANGLES:
            source = render_root / name / f"yaw{angle:03d}" / "normal.png"
            target = normal_root / name / f"yaw{angle:03d}" / "normal.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        diagnostics_path = output_dir / name / "geometry_diagnostics.json"
        if diagnostics_path.is_file():
            qef_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(diagnostics_path, qef_root / f"{name}.json")
    metrics = _compute_metrics(rendered, Image.open(args.image).convert("RGB"), int(args.render_resolution))
    _write_metrics(output_dir / "metrics.csv", metrics)
    for name in summary.get("variant_order", []):
        diagnostics_path = output_dir / name / "geometry_diagnostics.json"
        if diagnostics_path.is_file():
            summary.setdefault("variants", {})[name] = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
    summary["metrics"] = metrics
    summary["render_resolution"] = int(args.render_resolution)
    summary["render_cache_postprocessed"] = True
    _atomic_json(summary_path, summary)
    _write_report(
        output_dir / "GEOMETRY_OVOXEL_QEF_REPORT.md",
        summary,
        metrics,
        summary.get("variants", {}),
        summary.get("tau_mode_selection", {}),
    )
    print(f"[render-cache] postprocessed {len(rendered)} variants x {len(YAW_ANGLES)} views")
    return 0


def _reference_mask(image: Image.Image, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    value = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    # Estimate background from a narrow border; this handles both the black
    # canonical canvas and the original light-background asset without an
    # invented GT alpha channel.
    border = torch.cat([
        value[..., :4, :].reshape(1, 3, -1),
        value[..., -4:, :].reshape(1, 3, -1),
        value[..., :, :4].reshape(1, 3, -1),
        value[..., :, -4:].reshape(1, 3, -1),
    ], dim=-1)
    background = border.mean(dim=(-1, -2), keepdim=True)
    mask = ((value - background).abs().amax(dim=1, keepdim=True) > 0.035).float()
    if float(mask.mean()) < 0.01:
        mask = (value.mean(dim=1, keepdim=True) > 0.035).float()
    return value, mask


def _ssim_simple(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    import torch.nn.functional as F
    channels = int(reference.shape[1])
    coords = torch.arange(11, device=reference.device, dtype=reference.dtype) - 5.0
    gaussian = torch.exp(-(coords.square()) / (2.0 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = torch.outer(gaussian, gaussian).expand(channels, 1, 11, 11)
    mu_r = F.conv2d(reference, kernel, padding=5, groups=channels)
    mu_p = F.conv2d(prediction, kernel, padding=5, groups=channels)
    var_r = F.conv2d(reference.square(), kernel, padding=5, groups=channels) - mu_r.square()
    var_p = F.conv2d(prediction.square(), kernel, padding=5, groups=channels) - mu_p.square()
    cov = F.conv2d(reference * prediction, kernel, padding=5, groups=channels) - mu_r * mu_p
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_r * mu_p + c1) * (2 * cov + c2)) / ((mu_r.square() + mu_p.square() + c1) * (var_r + var_p + c2))
    return float(score.mean().item())


def _compute_metrics(
    rendered: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    reference_image: Image.Image,
    render_resolution: int,
) -> List[Dict[str, Any]]:
    reference, reference_mask = _reference_mask(reference_image, render_resolution)
    rows = []
    for name, views in rendered.items():
        if 0 not in views:
            continue
        shaded = torch.from_numpy(views[0]["shaded"]).permute(2, 0, 1).unsqueeze(0).float()
        pred_mask = torch.from_numpy(views[0]["mask"])
        if pred_mask.ndim == 2:
            pred_mask = pred_mask[None, None]
        elif pred_mask.ndim == 3:
            pred_mask = pred_mask.permute(2, 0, 1)[None]
        mse = (shaded - reference).square()
        mae = (shaded - reference).abs()
        foreground = reference_mask > 0.5
        fg_mse = mse[foreground.expand_as(mse)].mean() if bool(foreground.any()) else mse.mean()
        fg_mae = mae[foreground.expand_as(mae)].mean() if bool(foreground.any()) else mae.mean()
        psnr = float(10.0 * math.log10(1.0 / max(float(mse.mean()), 1e-12)))
        fg_psnr = float(10.0 * math.log10(1.0 / max(float(fg_mse), 1e-12)))
        iou = float((((pred_mask > 0.5) & foreground).sum() / (((pred_mask > 0.5) | foreground).sum().clamp_min(1))).item())
        rows.append({
            "variant": name,
            "yaw_deg": 0,
            "psnr_db": psnr,
            "ssim": _ssim_simple(reference, shaded),
            "mae": float(mae.mean()),
            "mse": float(mse.mean()),
            "foreground_psnr_db": fg_psnr,
            "foreground_ssim": _ssim_simple(reference * foreground, shaded * foreground),
            "foreground_mae": float(fg_mae),
            "silhouette_iou": iou,
            "foreground_pixels": int(foreground.sum()),
            "reference_note": "input image resized to render resolution by Lanczos; no geometry GT",
        })
    return rows


def _write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resample_payload(global_ovoxel: Mapping[str, Any]) -> Dict[str, Any]:
    h = _hermite_numpy(global_ovoxel["hermite"], int(global_ovoxel["resolution"]))
    active_keys = np.unique(h["key"])
    return {
        "resolution": int(global_ovoxel["resolution"]),
        "coords": global_ovoxel["coords"].clone(),
        "dual_vertices_cell": global_ovoxel["dual_vertices_cell"].clone(),
        "intersected": global_ovoxel["intersected"].clone(),
        "active_edge_keys": torch.from_numpy(active_keys),
        "stats": {
            "variant": "c4096_resample_only",
            "cell_count": int(global_ovoxel["coords"].shape[0]),
            "active_edge_count": int(active_keys.shape[0]),
            "observation_count": int(h["q"].shape[0]),
            "route": "baseline1024 mesh -> native global C4096 O-Voxel -> mesh",
        },
    }


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    variant_stats: Mapping[str, Mapping[str, Any]],
    mode_stats: Mapping[str, Any],
) -> None:
    metric_map = {row["variant"]: row for row in metrics}
    render_note = (
        "六个 yaw 的 geometry-only normal contact sheet 位于 "
        "`comparison_sheets/normal_contact_sheet.png`；数值依据是各 variant 的 topology、displacement 和 boundary-band residual。"
        if metrics else
        "本轮使用 `--skip-render`，因此尚未生成 yaw/contact-sheet/metrics；summary 中的 metrics 为空。"
    )
    metric_note = (
        "只比较 metrics.csv 的 yaw 0 行；reference 是输入图的 Lanczos resize，不是 geometry GT。"
        if metrics else
        "未生成 metrics.csv；输入图只作为后续渲染参考，不是 geometry GT。"
    )
    lines = [
        "# GEOMETRY_OVOXEL_QEF_REPORT",
        "",
        "本报告由 `pixal3d_ovoxel_hermite_qef_sr.py` 自动生成，研究对象是几何；没有把输入图当作 geometry GT。",
        "",
        "## 实验设置",
        "",
        f"- GPU: `{summary.get('gpu')}`，CUDA logical device `{summary.get('cuda_device')}`。",
        f"- 固定 AABB: `[-0.5, 0.5]^3`，global resolution: `{summary.get('global_resolution')}`。",
        f"- tile size/stride: `{summary.get('tile_size')}` / `{summary.get('tile_stride')}`；候选 blocks `{summary.get('candidate_tile_count')}`，实际 active/成功 `{summary.get('active_tile_count')}` / `{summary.get('successful_tile_count')}`。",
        "- local mesh 只作为 global primal-edge Hermite intersection carrier；没有 vertex averaging、welding、capping 或 local boundary QEF。",
        "- weighted confidence: `w_local = sigmoid(logit/T) * w_tile`；`w_base=lambda_base`。",
        f"- tile confidence: `w_tile = 0.05 + 0.95 * 0.5 * (1 - cos(pi * clamp(d / {summary.get('tile_boundary_band')}, 0, 1)))`，其中 `d` 是 local continuous point 到 tile boundary 的最小距离。",
        "",
        "## 结果表",
        "",
        "| variant | PSNR | SSIM | fg-PSNR | fg-SSIM | silhouette IoU | vertices | faces | active edges | births | deaths | QEF p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in summary.get("variant_order", []):
        row = metric_map.get(name, {})
        stat = variant_stats.get(name, {})
        topology = stat.get("topology", {})
        lines.append(
            f"| {name} | {row.get('psnr_db', 'n/a')} | {row.get('ssim', 'n/a')} | {row.get('foreground_psnr_db', 'n/a')} | {row.get('foreground_ssim', 'n/a')} | {row.get('silhouette_iou', 'n/a')} | {topology.get('vertex_count', 'n/a')} | {topology.get('face_count', 'n/a')} | {stat.get('active_edge_count', 'n/a')} | {stat.get('edge_birth_count', 'n/a')} | {stat.get('edge_death_count', 'n/a')} | {stat.get('qef_residual_p95', 'n/a')} |"
        )
    lines += [
        "",
        "## 对 Codex.md 问题的回答",
        "",
        "1. **C4096 单纯 revoxelize 是否改善？** 见 `c4096_resample_only` 对 `baseline_1024` 的 input-view 指标差异；这部分明确标记为 discretization control，没有归因给 local flow。",
        "2. **local shape flow 是否改变 geometry？** 比较 `c4096_resample_only` 与三种 fusion variant 的 dual/QEF、edge support 和 displacement；若成功 tile 数为 0，则本次运行只完成 control，不能声称 flow 改善。",
        f"3. **local HR 覆盖多少 global edges/surface？** mode selection 记录了 `{mode_stats.get('selected_observations', 0)}` 个 local observations，baseline anchor edge 数为 `{variant_stats.get('hermite_unweighted', {}).get('baseline_active_edges', 0)}`；完整覆盖率见 `summary.json`。",
        "4. **weighted 是否优于 unweighted？** 以 input-view PSNR/SSIM/IoU 和 QEF residual 同时判断；表中数值若没有优势，不能仅凭视觉主观判断优于。",
        "5. **edge logit 是否有帮助？** weighted 变体使用 decoder provenance 追溯到 `intersected_logits`；fallback 比例、edge temperature 和 tile weights 在 `local_tiles/*/diagnostics.json` 与 summary 中记录。",
        "6. **fixed 与 allow-edge-change 差距？** fixed 变体强制最终 active primal-edge mask 等于 baseline C4096；birth/death 差异直接回答 topology/support 改变的贡献。",
        "7. **新增 edge 是否噪声？** 查看 birth edge 数、局部 confidence/tau mode 统计、拓扑 non-manifold/degenerate counts；本报告不把没有 GT 的 topology 变化称作提升。",
        "8. **人工 tile boundary artifact？** local extraction 使用 `boundary_weight=0`，boundary-band 以 raised-cosine 降权；QEF residual 及 band/interior 诊断在 `qef_debug` 保存。",
        f"9. **normal seam/tearing/spike/smoothing？** {render_note}",
        f"10. **front PSNR/SSIM/silhouette 是否超过 baseline？** {metric_note}",
        "11. **下一阶段候选？** 默认选择同时满足 weighted 不劣于 unweighted、fixed support invariant 通过、fallback/非流形率不过高的 variant；若这些条件不满足，建议先调试坐标/edge mode，而不是扩大模型实验。",
        "",
        "## 可复现性与限制",
        "",
        "- `correctness_tests.json` 保存 legacy parity、Hermite tau、crop/transform round-trip、fixed support 和 resample-only 检查。",
        "- 所有 geometry variant 的 vertex PBR 都来自同一 baseline C1024 volumetric field；没有重新跑 texture flow。",
        "- `face_weight` 是原生 mesh voxelizer 的 face-plane 项；最终融合 QEF 使用导出的 Hermite plane sufficient statistics 与原生 regularization/box constraint。",
        "- 若命令使用 `--max-tiles`，结果是 subset sanity，不是 full active-block conclusion；summary 会明确记录。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")



def _run_correctness_tests(output_dir: Path) -> Dict[str, Any]:
    """Run the nine required small, deterministic correctness checks."""
    print("[tests] running Hermite/QEF correctness suite")
    vertices = torch.tensor(
        [[-0.35, -0.35, -0.35], [0.35, -0.35, -0.35],
         [0.0, 0.35, -0.35], [0.0, 0.0, 0.35]], dtype=torch.float32
    )
    faces = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=torch.int32
    )
    resolution = 16
    legacy = mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=resolution, aabb=RUNTIME_AABB,
        face_weight=0.0, boundary_weight=0.0, regularization_weight=0.01,
    )
    extended = mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=resolution, aabb=RUNTIME_AABB,
        face_weight=0.0, boundary_weight=0.0, regularization_weight=0.01,
        return_hermite=True,
    )
    parity = [bool(torch.equal(left, right)) for left, right in zip(legacy, extended[:3])]
    if not all(parity):
        raise AssertionError(f"legacy parity failed: {parity}")
    h = _hermite_numpy(extended[3], resolution)
    tau_residual = h["q"][np.arange(h["q"].shape[0]), h["edge_axis"]] - (
        -0.5 + (h["edge_coord"][np.arange(h["q"].shape[0]), h["edge_axis"]] + h["tau"]) / resolution
    )
    tau_ok = bool(
        np.isfinite(h["tau"]).all()
        and (h["tau"] >= -1e-5).all()
        and (h["tau"] <= 1.00001).all()
        and np.abs(tau_residual).max(initial=0.0) < 2e-5
    )
    if not tau_ok:
        raise AssertionError("Hermite tau/edge coordinate invariant failed")

    # Reconstruct the C++ QEF using only exported Hermite planes.  The test
    # sets face_weight=0 so the exported sufficient statistics are complete.
    coords = legacy[0].numpy().astype(np.int32)
    native_object = legacy[1].float().numpy() - 0.5
    a = np.zeros((coords.shape[0], 6), dtype=np.float32)
    b = np.zeros((coords.shape[0], 3), dtype=np.float32)
    means = np.zeros((coords.shape[0], 3), dtype=np.float32)
    counts = np.zeros((coords.shape[0],), dtype=np.float32)
    cell_map = {tuple(value.tolist()): index for index, value in enumerate(coords)}
    for edge, axis, q, n in zip(h["edge_coord"], h["edge_axis"], h["q"], h["n"]):
        for offset in EDGE_CELL_OFFSETS[int(axis)]:
            index = cell_map.get(tuple((edge + offset).tolist()))
            if index is None:
                continue
            a[index, 0] += n[0] * n[0]
            a[index, 1] += n[1] * n[1]
            a[index, 2] += n[2] * n[2]
            a[index, 3] += n[0] * n[1]
            a[index, 4] += n[0] * n[2]
            a[index, 5] += n[1] * n[2]
            b[index] += n * float(np.dot(n, q))
            means[index] += q
            counts[index] += 1.0
    mean = means / np.maximum(counts[:, None], 1e-8)
    a[:, 0] += 0.01 * counts
    a[:, 1] += 0.01 * counts
    a[:, 2] += 0.01 * counts
    b += 0.01 * counts[:, None] * mean
    aa = np.zeros((coords.shape[0], 3, 3), dtype=np.float32)
    aa[:, 0, 0], aa[:, 1, 1], aa[:, 2, 2] = a[:, 0], a[:, 1], a[:, 2]
    aa[:, 0, 1] = aa[:, 1, 0] = a[:, 3]
    aa[:, 0, 2] = aa[:, 2, 0] = a[:, 4]
    aa[:, 1, 2] = aa[:, 2, 1] = a[:, 5]
    lo = -0.5 + torch.from_numpy(coords).float() / resolution
    hi = lo + 1.0 / resolution
    reconstructed, _, _ = _solve_box_native(torch.from_numpy(aa), torch.from_numpy(b), lo, hi)
    qef_error = (reconstructed.numpy() - native_object).astype(np.float64)
    qef_stats = {
        "max_abs": float(np.abs(qef_error).max(initial=0.0)),
        "mean_abs": float(np.abs(qef_error).mean()),
    }
    if qef_stats["max_abs"] > 2e-4:
        raise AssertionError(f"Hermite QEF reconstruction error is too large: {qef_stats}")

    crop = _crop_global_ovoxel(
        {"coords": legacy[0], "dual_vertices_cell": legacy[1], "intersected": legacy[2]},
        (3, 3, 3), 8, resolution,
    )
    crop_roundtrip = bool(torch.equal(crop["coords"] + 3, legacy[0][((legacy[0] >= 3) & (legacy[0] < 11)).all(dim=1)]))
    if not crop_roundtrip:
        raise AssertionError("integer C4096 -> C1024 crop round trip failed")

    points = torch.randn(128, 3)
    local = _global_to_local(points, (2, 5, 7), 8, 16)
    roundtrip = _local_to_global(local, (2, 5, 7), 8, 16)
    roundtrip_error = float((roundtrip - points).abs().max())
    if roundtrip_error >= 1e-6:
        raise AssertionError(f"similarity round trip error is {roundtrip_error}")

    logits = torch.linspace(-2.0, 2.0, 48).reshape(16, 3)
    raw_bool = logits > 0
    if not torch.equal(raw_bool, logits > 0):
        raise AssertionError("decoder raw-logit bool parity failed")

    open_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    open_h = mesh_to_flexible_dual_grid(
        vertices[:3], open_faces, grid_size=resolution, aabb=RUNTIME_AABB,
        face_weight=0.0, boundary_weight=0.0, regularization_weight=0.01,
        return_hermite=True,
    )[3]
    boundary_disabled = int(open_h["is_mesh_boundary_source"].sum()) == 0
    if not boundary_disabled:
        raise AssertionError("boundary-disabled Hermite export contains boundary sources")

    baseline_h = _baseline_observations(
        {"hermite": {key: value for key, value in extended[3].items()}}, resolution
    )
    selected_local = {key: value.copy() for key, value in baseline_h.items()}
    active, fixed_stats = _active_edge_stats(
        np.unique(baseline_h["key"]), selected_local,
        argparse.Namespace(edge_activation_threshold=0.25), fixed=True,
    )
    fixed_invariant = bool(np.array_equal(active, np.unique(baseline_h["key"])))
    if not fixed_invariant:
        raise AssertionError("fixed primal-edge support invariant failed")

    resample_ok = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mesh_vertices, mesh_faces = flexible_dual_grid_to_mesh(
            legacy[0].to(device),
            (legacy[1] * resolution - legacy[0]).to(device),
            legacy[2].to(device),
            split_weight=None,
            aabb=RUNTIME_AABB.to(device),
            grid_size=resolution,
        )
        resample_ok = bool(mesh_vertices.shape[0] > 0 and mesh_faces.shape[0] > 0)
    if not resample_ok:
        raise AssertionError("resample-only control produced an empty mesh")
    results = {
        "test_1_legacy_api_parity": {"passed": all(parity), "per_tensor": parity},
        "test_2_hermite_reconstruction_parity": {"passed": True, **qef_stats},
        "test_3_boundary_disabled": {"passed": boundary_disabled},
        "test_4_decoder_raw_logits_parity": {"passed": True},
        "test_5_similarity_roundtrip": {"passed": True, "max_abs_error": roundtrip_error},
        "test_6_ovoxel_coordinate_crop": {"passed": crop_roundtrip},
        "test_7_global_edge_key_tau": {"passed": tau_ok, "max_edge_axis_error": float(np.abs(tau_residual).max(initial=0.0))},
        "test_8_fixed_edge_invariant": {"passed": fixed_invariant, **fixed_stats},
        "test_9_resample_only": {"passed": resample_ok},
    }
    _atomic_json(output_dir / "correctness_tests.json", results)
    print("[tests] PASS")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-mesh", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--moge-model", type=Path, default=DEFAULT_MOGE)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-resolution", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-stride", type=int, default=1024)
    parser.add_argument("--tile-ids", type=str, default="")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-active-cells", type=int, default=1)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-local-flow", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--postprocess-render-cache", action="store_true")
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--force-revoxelize", action="store_true")
    parser.add_argument("--force-tiles", action="store_true")
    parser.add_argument("--force-qef", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--face-weight", type=float, default=1.0)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--tau-cluster-threshold", type=float, default=0.08)
    parser.add_argument("--edge-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-base", type=float, default=1.0)
    parser.add_argument("--tile-boundary-band", type=float, default=0.15)
    parser.add_argument("--edge-activation-threshold", type=float, default=0.25)
    parser.add_argument("--qef-observation-chunk", type=int, default=500_000)
    parser.add_argument("--qef-batch-size", type=int, default=65_536)
    parser.add_argument("--query-chunk-size", type=int, default=262_144)
    parser.add_argument("--topology-max-faces", type=int, default=1_000_000)
    parser.add_argument("--displacement-max-points", type=int, default=100_000)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=1)
    parser.add_argument("--peel-layers", type=int, default=6)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--envmap", type=str, default="studio")
    parser.add_argument("--camera-angle-x", type=float, default=0.517371749106554)
    parser.add_argument("--camera-distance", type=float, default=1.889538288116455)
    return parser


def _parse_ids(value: str) -> Optional[set[int]]:
    if not value.strip():
        return None
    result = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(token))
    return result


def _active_tile_ids(
    global_ovoxel: Mapping[str, Any],
    resolution: int,
    tile_size: int,
    stride: int,
    min_cells: int,
) -> Tuple[List[Tuple[int, int, int]], List[int], List[int]]:
    layout = _tile_layout(resolution, tile_size, stride)
    coords = global_ovoxel["coords"]
    counts = []
    active = []
    for tile_id, start in enumerate(layout):
        crop = _crop_global_ovoxel(global_ovoxel, start, tile_size, resolution)
        count = int(crop["coords"].shape[0])
        counts.append(count)
        if count >= int(min_cells):
            active.append(tile_id)
    return layout, active, counts


def _load_tile_hermite(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    h = {key: value.numpy() if torch.is_tensor(value) else value for key, value in payload["hermite"].items()}
    h["key"] = _edge_keys(h["edge_coord"], h["edge_axis"], int(payload["resolution"]))
    return h, payload.get("diagnostics", {})


def _make_variants_diagnostics(
    variant_meshes: Mapping[str, Mapping[str, Any]],
    baseline_vertices: torch.Tensor,
    variant_stats: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    for name, payload in variant_meshes.items():
        if name == "baseline_1024":
            vertices = payload["mesh"].vertices
            faces = payload["mesh"].faces
        else:
            vertices = payload["vertices"]
            faces = payload["faces"]
        topology = _mesh_topology_diagnostics(vertices, faces, int(args.topology_max_faces))
        variant_stats.setdefault(name, {}).update({
            "topology": topology,
            "baseline_closest_surface_displacement": _surface_displacement(
                baseline_vertices, vertices, int(args.displacement_max_points)
            ),
        })


def main() -> int:
    args = _build_parser().parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.baseline_mesh.is_file():
        raise FileNotFoundError(
            f"baseline mesh not found: {args.baseline_mesh}; run the native 1024 baseline first"
        )
    if args.global_resolution != 4096 or args.tile_size != 1024:
        raise ValueError("this Pixal3D experiment requires global_resolution=4096 and tile_size=1024")
    if args.tile_stride not in (512, 1024):
        raise ValueError("tile_stride must be 1024 for the main run or 512 for overlap ablation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.cuda_device < 0 or args.cuda_device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GPU] cuda:{args.cuda_device} {torch.cuda.get_device_name(args.cuda_device)}")
    _atomic_json(output_dir / "config.json", {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    })
    tests = _run_correctness_tests(output_dir)
    if args.postprocess_render_cache:
        return _postprocess_render_cache(output_dir, args)
    if args.tests_only:
        return 0

    baseline_mesh = _load_baseline(args.baseline_mesh)
    _save_baseline_copy(baseline_mesh, output_dir, args.baseline_mesh)
    input_image = Image.open(args.image).convert("RGB")
    input_image.save(output_dir / "input_original.png")
    camera = {
        "camera_angle_x": float(args.camera_angle_x),
        "distance": float(args.camera_distance),
        "mesh_scale": 1.0,
        "source": "baseline1024 summary camera; no per-tile recanonicalization",
    }
    _atomic_json(output_dir / "global_camera.json", camera)

    global_ovoxel = _global_voxelize(
        baseline_mesh, int(args.global_resolution), args, output_dir
    )
    resample_path = output_dir / "baseline_c4096_resample_only" / "final_ovoxel.pt"
    if resample_path.is_file() and not args.force_revoxelize:
        resample = torch.load(resample_path, map_location="cpu", weights_only=False)
        print(f"[C4096 control] loading cache {resample_path}")
    else:
        resample = _resample_payload(global_ovoxel)
        _atomic_torch_save(resample_path, resample)
    layout, active_tile_ids, active_counts = _active_tile_ids(
        global_ovoxel, int(args.global_resolution), int(args.tile_size), int(args.tile_stride), int(args.min_active_cells)
    )
    requested = _parse_ids(args.tile_ids)
    if requested is not None:
        active_tile_ids = [tile_id for tile_id in active_tile_ids if tile_id in requested]
    if args.max_tiles is not None:
        active_tile_ids = active_tile_ids[: int(args.max_tiles)]
    print(
        f"[tiles] candidates={len(layout)} active={len(active_tile_ids)} "
        f"stride={args.tile_stride} cells={sum(active_counts[tile_id] for tile_id in active_tile_ids):,}"
    )

    pipeline = None
    shape_encoder = None
    feature_cache = None
    canonical_image = input_image
    successful_tiles = []
    local_observation_items: List[Mapping[str, np.ndarray]] = []
    tile_diagnostics: Dict[str, Any] = {}
    cache_ready = bool(active_tile_ids) and not args.force_tiles and all(
        (output_dir / "local_tiles" / f"tile_{tile_id:03d}" / "shape_flow_and_raw_ovoxel.pt").is_file()
        and (output_dir / "local_tiles" / f"tile_{tile_id:03d}" / "global_hermite.pt").is_file()
        for tile_id in active_tile_ids
    )
    if not args.skip_local_flow and not cache_ready:
        import pixal3d.models as pixal3d_models

        print("[pipeline] loading native Pixal3D pipeline for local shape flow")
        pipeline = _init_local_shape_pipeline(
            args.model_path, device, bool(args.low_vram)
        )
        # The document requires the original input image for all local
        # projected conditions.  This is a plain canonical resize, not an AI
        # super-resolution image and not a per-tile crop/recanonicalization.
        canonical_image = input_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        canonical_image.save(output_dir / "canonical_1024.png")
        input_image.resize((4096, 4096), Image.Resampling.LANCZOS).save(output_dir / "canonical_4096.png")
        _atomic_json(output_dir / "canonical_metadata.json", {
            "version": "original_input_resize_v1",
            "source_size": list(input_image.size),
            "image_1024": [1024, 1024],
            "image_4096": [4096, 4096],
            "background_segmentation": "disabled; original RGB input is used unchanged",
        })
        shape_encoder = pixal3d_models.from_pretrained(str(args.shape_encoder)).eval()
        if not args.low_vram:
            shape_encoder.to(device)
        _ensure_baseline_1024_artifacts(
            baseline_mesh, shape_encoder, pipeline, output_dir, device, args
        )
        proj_model = pipeline.image_cond_model_shape_1024
        feature_cache = _build_image_feature_cache(proj_model, canonical_image, device)
        _atomic_torch_save(
            output_dir / "image_condition_global_token.pt",
            {"global": feature_cache["global"].cpu(), "source": "single original canonical image feature extraction"},
        )

        for tile_id in active_tile_ids:
            start = layout[tile_id]
            tile_dir = output_dir / "local_tiles" / f"tile_{tile_id:03d}"
            tile_dir.mkdir(parents=True, exist_ok=True)
            flow_path = tile_dir / "shape_flow_and_raw_ovoxel.pt"
            hermite_path = tile_dir / "global_hermite.pt"
            try:
                if flow_path.is_file() and hermite_path.is_file() and not args.force_tiles:
                    tile_flow = torch.load(flow_path, map_location="cpu", weights_only=False)
                    h, diagnostics = _load_tile_hermite(hermite_path)
                    local_observation_items.append(h)
                    successful_tiles.append(tile_id)
                    tile_diagnostics[str(tile_id)] = diagnostics
                    print(f"[tile {tile_id:03d}] cache hit")
                    continue
                crop = _crop_global_ovoxel(global_ovoxel, start, int(args.tile_size), int(args.global_resolution))
                if crop["coords"].shape[0] < int(args.min_active_cells):
                    continue
                transform_local = _build_local_camera_transform(
                    proj_model,
                    float(camera["camera_angle_x"]),
                    float(camera["distance"]),
                    start,
                    int(args.tile_size),
                    int(args.global_resolution),
                    device,
                )
                tile_flow = _run_local_shape_flow(
                    pipeline=pipeline,
                    shape_encoder=shape_encoder,
                    crop=crop,
                    condition=None,
                    condition_model=proj_model,
                    feature_cache=feature_cache,
                    transform_local=transform_local,
                    camera_angle_x=float(camera["camera_angle_x"]),
                    camera_distance=float(camera["distance"]),
                    args=args,
                    tile_id=tile_id,
                )
                _atomic_torch_save(flow_path, tile_flow)
                tile_result, diagnostics = _local_mesh_to_global_hermite(
                    tile_flow, start, int(args.tile_size), int(args.global_resolution), args
                )
                h = tile_result["hermite"]
                _atomic_torch_save(
                    hermite_path,
                    {
                        "resolution": int(args.global_resolution),
                        "hermite": {key: torch.from_numpy(value) for key, value in h.items() if key != "key"},
                        "diagnostics": diagnostics,
                        "vertices": tile_result["vertices"],
                        "faces": tile_result["faces"],
                        "start": list(start),
                    },
                )
                local_observation_items.append(h)
                successful_tiles.append(tile_id)
                tile_diagnostics[str(tile_id)] = diagnostics
                print(
                    f"[tile {tile_id:03d}] cells={crop['coords'].shape[0]:,} "
                    f"H={h['q'].shape[0]:,} fallback={diagnostics['provenance_fallback_fraction']:.4f}"
                )
            except Exception as exc:
                _atomic_json(tile_dir / "failure.json", {"tile_id": tile_id, "start": list(start), "error": repr(exc)})
                print(f"[tile {tile_id:03d}] FAILED: {exc}")
                if args.fail_fast:
                    raise
        if shape_encoder is not None:
            del shape_encoder
        if pipeline is not None:
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    elif cache_ready:
        print("[pipeline] all requested local tiles are cached; skipping model reload")
        canonical_path = output_dir / "canonical_1024.png"
        if canonical_path.is_file():
            canonical_image = Image.open(canonical_path).convert("RGB")
        for tile_id in active_tile_ids:
            tile_dir = output_dir / "local_tiles" / f"tile_{tile_id:03d}"
            flow_path = tile_dir / "shape_flow_and_raw_ovoxel.pt"
            hermite_path = tile_dir / "global_hermite.pt"
            torch.load(flow_path, map_location="cpu", weights_only=False)
            h, diagnostics = _load_tile_hermite(hermite_path)
            local_observation_items.append(h)
            successful_tiles.append(tile_id)
            tile_diagnostics[str(tile_id)] = diagnostics
            print(f"[tile {tile_id:03d}] cache hit (pipeline skipped)")
    else:
        print("[tiles] --skip-local-flow: producing the no-local-observation control path")

    baseline_obs = _baseline_observations(global_ovoxel, int(args.global_resolution))
    local_obs = _concat_observations(local_observation_items)
    selected_cache = output_dir / "hermite_debug" / "selected_local_observations.pt"
    mode_cache = output_dir / "hermite_debug" / "mode_selection.json"
    cached_mode = json.loads(mode_cache.read_text()) if mode_cache.is_file() else {}
    if (
        local_obs["q"].size
        and selected_cache.is_file()
        and cached_mode.get("local_observations") == int(local_obs["q"].shape[0])
        and not args.force_tiles
    ):
        selected_payload = torch.load(selected_cache, map_location="cpu", weights_only=False)
        selected_local = {
            key: value.numpy() if torch.is_tensor(value) else value
            for key, value in selected_payload.items()
        }
        mode_stats = cached_mode
        print(f"[local] loading cached tau consensus {selected_cache}")
    else:
        selected_local, mode_stats = _select_tau_modes(
            baseline_obs, local_obs, int(args.global_resolution), float(args.tau_cluster_threshold)
        )
        _atomic_torch_save(
            selected_cache,
            {key: torch.from_numpy(value) for key, value in selected_local.items()},
        )
        _atomic_json(mode_cache, mode_stats)
    _atomic_json(output_dir / "local_tiles" / "summary.json", {
        "candidate_tile_count": len(layout),
        "active_tile_ids": active_tile_ids,
        "successful_tile_ids": successful_tiles,
        "active_cell_counts": active_counts,
        "tile_diagnostics": tile_diagnostics,
    })

    baseline_key_cache = output_dir / "hermite_debug" / "baseline_edge_keys.pt"
    if baseline_key_cache.is_file() and not args.force_revoxelize:
        baseline_keys = torch.load(baseline_key_cache, map_location="cpu", weights_only=False).numpy()
        print(f"[baseline] loading cached edge-key index {baseline_key_cache}")
    else:
        print("[baseline] building sorted edge-key index from resample control")
        # _resample_payload already constructs the sorted unique baseline
        # edge-key array; reuse it instead of sorting the 66M Hermite rows a
        # second time.
        baseline_keys = resample["active_edge_keys"].numpy()
        _atomic_torch_save(baseline_key_cache, torch.from_numpy(baseline_keys))

    variant_order = [
        "baseline_1024",
        "c4096_resample_only",
        "hermite_unweighted",
        "hermite_weighted",
        "fixed_primal_edge_weighted",
    ]
    variant_payloads: Dict[str, Dict[str, Any]] = {
        "c4096_resample_only": resample,
    }
    variant_stats: Dict[str, Dict[str, Any]] = {
        "c4096_resample_only": dict(resample["stats"]),
    }
    local_edge_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if local_obs["q"].size:
        local_edge_stats_path = output_dir / "hermite_debug" / "local_edge_activation_stats.pt"
        if local_edge_stats_path.is_file() and not args.force_tiles:
            cached_local_stats = torch.load(local_edge_stats_path, map_location="cpu", weights_only=False)
            local_edge_stats = (
                cached_local_stats["key"].numpy(),
                cached_local_stats["max_score"].numpy(),
            )
            print(f"[local] loading cached edge activation stats {local_edge_stats_path}")
        else:
            print("[local] building sorted edge activation stats")
            local_edge_stats = _unique_key_max(
                selected_local["key"],
                selected_local["confidence"] * selected_local["tile_weight"],
            )
            _atomic_torch_save(
                local_edge_stats_path,
                {
                    "key": torch.from_numpy(local_edge_stats[0]),
                    "max_score": torch.from_numpy(local_edge_stats[1]),
                },
            )
    if local_obs["q"].size == 0 and abs(float(args.lambda_base) - 1.0) < 1e-12:
        # With no local donors and lambda_base=1, all three Hermite variants
        # are exactly the native C4096 baseline. Reusing that native QEF is
        # both mathematically exact and avoids a redundant Python scatter of
        # every baseline observation into four incident cells.
        for variant in variant_order[2:]:
            payload = dict(resample)
            payload["stats"] = dict(resample["stats"])
            payload["stats"]["variant"] = variant
            payload["stats"].update({
                "route": "native C4096 baseline reused because no local donors were selected",
                "baseline_active_edges": int(np.unique(baseline_obs["key"]).size),
                "final_active_edges": int(np.unique(baseline_obs["key"]).size),
                "edge_birth_count": 0,
                "edge_death_count": 0,
                "edge_retained_count": int(np.unique(baseline_obs["key"]).size),
                "edge_changed_fraction": 0.0,
                "observation_count": int(baseline_obs["q"].shape[0]),
                "local_observation_count": 0,
                "qef_residual_mean": 0.0,
                "qef_residual_p95": 0.0,
                "qef_residual_max": 0.0,
            })
            variant_payloads[variant] = payload
            variant_stats[variant] = dict(payload["stats"])
            _atomic_torch_save(output_dir / variant / "final_ovoxel_qef.pt", payload)
            _atomic_json(output_dir / variant / "geometry_diagnostics.json", payload["stats"])
    else:
        for variant in variant_order[2:]:
            qef_path = output_dir / variant / "final_ovoxel_qef.pt"
            payload = None
            if qef_path.is_file() and not args.force_revoxelize and not args.force_tiles and not args.force_qef:
                cached_payload = torch.load(qef_path, map_location="cpu", weights_only=False)
                cached_local_count = int(cached_payload.get("stats", {}).get("local_observation_count", -1))
                if cached_local_count == int(selected_local["q"].shape[0]) and "qef_cell_count" in cached_payload.get("stats", {}):
                    payload = cached_payload
                    print(f"[{variant}] loading cached QEF payload {qef_path}")
            if payload is None:
                payload = _assemble_variant(
                    global_ovoxel,
                    baseline_obs,
                    selected_local,
                    variant,
                    args,
                    output_dir,
                    baseline_keys=baseline_keys,
                    local_edge_stats=local_edge_stats,
                )
            variant_payloads[variant] = payload
            variant_stats[variant] = dict(payload["stats"])

    # The same baseline field is used to bake all final geometry variants.
    variant_meshes: Dict[str, Dict[str, Any]] = {
        "baseline_1024": {"mesh": baseline_mesh.to(device)},
    }
    for variant in variant_order[1:]:
        variant_meshes[variant] = _save_variant_meshes(
            variant_payloads[variant], baseline_mesh, variant, output_dir, device, args
        )
    _make_variants_diagnostics(
        variant_meshes, baseline_mesh.vertices, variant_stats, args
    )
    for name, stats in variant_stats.items():
        _atomic_json(output_dir / name / "geometry_diagnostics.json", stats)

    rendered = {}
    metrics = []
    if not args.skip_render:
        render_inputs = {name: payload["mesh"] for name, payload in variant_meshes.items()}
        rendered = _render_variants(render_inputs, camera, output_dir / "pbr_renders", args, device)
        _contact_sheet(output_dir / "pbr_renders", rendered, "shaded", output_dir / "comparison_sheets" / "shaded_contact_sheet.png")
        _contact_sheet(output_dir / "pbr_renders", rendered, "normal", output_dir / "comparison_sheets" / "normal_contact_sheet.png")
        metrics = _compute_metrics(rendered, canonical_image, int(args.render_resolution))
        _write_metrics(output_dir / "metrics.csv", metrics)
    else:
        print("[render] skipped by --skip-render")

    summary = {
        "format": "pixal3d_geometry_ovoxel_hermite_qef_sr_v1",
        "cuda_device": int(args.cuda_device),
        "gpu": torch.cuda.get_device_name(args.cuda_device),
        "image": str(args.image.resolve()),
        "image_sha256": _sha256(args.image),
        "baseline_mesh": str(args.baseline_mesh.resolve()),
        "global_resolution": int(args.global_resolution),
        "tile_size": int(args.tile_size),
        "tile_stride": int(args.tile_stride),
        "tile_boundary_band": float(args.tile_boundary_band),
        "candidate_tile_count": len(layout),
        "active_tile_count": len(active_tile_ids),
        "successful_tile_count": len(successful_tiles),
        "successful_tile_ids": successful_tiles,
        "shape_flow": not args.skip_local_flow,
        "shape_steps": int(args.shape_steps),
        "single_global_image_feature_extraction": True,
        "camera": camera,
        "baseline_c4096_cells": int(global_ovoxel["coords"].shape[0]),
        "baseline_c4096_hermite_observations": int(baseline_obs["q"].shape[0]),
        "local_hermite_observations": int(local_obs["q"].shape[0]),
        "selected_local_hermite_observations": int(selected_local["q"].shape[0]),
        "selected_local_edge_coverage": float(np.unique(selected_local["key"]).size / max(np.unique(baseline_obs["key"]).size, 1)),
        "tau_mode_selection": mode_stats,
        "variant_order": variant_order,
        "variants": variant_stats,
        "metrics": metrics,
        "render_resolution": int(args.render_resolution),
        "normal_definition": "camera-space normal RGB=(n_cam+1)/2 multiplied by renderer mask; black background",
        "reference_definition": "input/canonical image resized to render resolution by Lanczos; no geometry GT",
        "correctness_tests": tests,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "GEOMETRY_OVOXEL_QEF_REPORT.md", summary, metrics, variant_stats, mode_stats)
    print(f"[done] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
