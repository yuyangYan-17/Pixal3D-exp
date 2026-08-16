#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free, fixed-shape visibility-aware texture endpoint guidance.

This is an independent experiment layered on the repository's native Pixal3D
path.  It deliberately keeps the local geometry/support fixed and changes
only the clean texture endpoint used by the native FlowEuler Euler update:

    HR model -> pred_x_0 -> official texture decoder -> PBR query on the
    local C1024 support -> mesh/O-Voxel/SLat visibility PBR blend -> official
    texture encoder -> endpoint correction -> official x0-to-velocity helper.

    The geometry-only visibility buffer is rendered once at canonical 4096 with
    nvdiffrast, with its raster rows explicitly converted to the canonical
    top-origin image convention.  All defaults are for CUDA device 4 and the repository's known
``assets/choose/0_img.png`` experiment image, but the input and output are
CLI parameters.  No shape sampler is called and no network is trained.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_texture_pbr_degradation_experiment as degradation
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_visibility_guided_pbr_flow_v3_mesh_ovoxel_slat_nearby"
GLOBAL_IMAGE_SIZE = 1024
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = TILE_SIZE
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
PBR_LAYOUT = dict(degradation.PBR_LAYOUT)
PBR_SLICES = {
    "RGB": slice(0, 3),
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
    "joint": slice(0, 6),
}


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())


def _relative(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    return _norm(value) / (_norm(reference) + float(eps))


def _safe_mean(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).mean().item()) if value.numel() else 0.0


def _tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    flat = value.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "q10": float(torch.quantile(flat, 0.10).item()),
        "q25": float(torch.quantile(flat, 0.25).item()),
        "q50": float(torch.quantile(flat, 0.50).item()),
        "q75": float(torch.quantile(flat, 0.75).item()),
        "q90": float(torch.quantile(flat, 0.90).item()),
        "fraction_lt_0.1": float((flat < 0.1).to(torch.float32).mean().item()),
        "fraction_gt_0.9": float((flat > 0.9).to(torch.float32).mean().item()),
    }


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    return degradation._normalize_slat(value, normalization)


def _denormalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    return degradation._denormalize_slat(value, normalization)


def _fresh_sparse(value: SparseTensor) -> SparseTensor:
    return degradation._fresh_sparse(value)


def _make_render_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        render_resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        render_ssaa=int(args.render_ssaa),
        render_peel_layers=int(args.render_peel_layers),
        render_face_chunk_size=int(args.render_face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        envmap=str(args.envmap),
        lpips_net=str(args.lpips_net),
        skip_lpips=bool(args.skip_lpips),
    )


def _channel_relative_errors(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, channel_slice in PBR_SLICES.items():
        result[name] = _relative(left[:, channel_slice] - right[:, channel_slice], right[:, channel_slice])
    return result


def _channel_mean_abs(left: torch.Tensor, right: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    if mask is None:
        mask = torch.ones(left.shape[0], device=left.device, dtype=torch.bool)
    result: Dict[str, float] = {}
    for name, channel_slice in PBR_SLICES.items():
        values = (left[:, channel_slice] - right[:, channel_slice]).abs()
        if mask.numel() == values.shape[0]:
            values = values[mask]
        result[name] = _safe_mean(values)
    return result


def _tile_boxes() -> List[Tuple[int, int, int, int]]:
    return core._tile_layout(
        canonical_size=CANONICAL_IMAGE_SIZE,
        tile_size=TILE_SIZE,
        stride=TILE_STRIDE,
    )


def _global_face_normals(mesh: MeshWithVoxel) -> torch.Tensor:
    vertices = mesh.vertices.to(torch.float32)
    faces = mesh.faces.to(torch.long)
    triangles = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3)
    normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    return F.normalize(normals, dim=1, eps=1e-12).cpu()


def _camera_clip_vertices(
    vertices_q: torch.Tensor,
    *,
    camera: Mapping[str, float],
    resolution: int,
    near: float,
    far: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build clip coordinates for the exact global camera projection.

    The global q convention has camera-space z < 0 and positive camera depth
    ``D=-Pz``.  nvdiffrast accepts a positive clip w, so the matrix is written
    directly from the documented pixel projection rather than changing the
    camera or normalizing the mesh by a bbox.
    """
    q = vertices_q.to(torch.float32) * (2.0 * float(camera["mesh_scale"]))
    points = core._camera_q_to_points(
        q,
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    depth = -points[:, 2]
    focal = core._focal_pixels(float(camera["camera_angle_x"]), GLOBAL_IMAGE_SIZE)
    focal *= float(resolution) / float(GLOBAL_IMAGE_SIZE)
    cx = float(resolution) * 0.5
    cy = float(resolution) * 0.5
    clip = torch.zeros((points.shape[0], 4), device=points.device, dtype=torch.float32)
    clip[:, 0] = (2.0 * focal / float(resolution)) * points[:, 0] + (2.0 * cx / float(resolution) - 1.0) * depth
    # nvdiffrast writes raster rows in the OpenGL/bottom-origin convention.
    # The rest of Pixal3D (the canonical image, _project_points(), and the
    # tile boxes) uses top-origin image rows, so the camera-space y term must
    # be negated here.  Without this sign, the visibility buffer is vertically
    # flipped relative to the projected mesh and almost no tile face IDs can
    # be transferred back to the local geometry.
    clip[:, 1] = -(2.0 * focal / float(resolution)) * points[:, 1] + (2.0 * cy / float(resolution) - 1.0) * depth
    clip[:, 2] = ((depth - float(near)) / max(float(far - near), 1e-6)) * depth
    clip[:, 3] = depth
    return clip, points, depth


@torch.no_grad()
def _render_global_visibility_buffers(
    mesh: MeshWithVoxel,
    *,
    global_camera: Mapping[str, float],
    resolution: int = CANONICAL_IMAGE_SIZE,
    face_chunk_size: int = 4_000_000,
    device: torch.device,
) -> Dict[str, Any]:
    """Rasterize one geometry-only z-buffer at canonical 4096.

    The per-chunk merge compares nvdiffrast's perspective depth, while the
    saved depth map is camera-space positive depth.  Triangle IDs remain the
    original global face IDs, so contribution-level exact-ID preference is
    possible during debugging.  The saved triangle IDs are later used as a
    hard projected-mesh visibility set and transferred through O-Voxel/SLat.
    """
    if resolution <= 0 or face_chunk_size <= 0:
        raise ValueError("resolution and face_chunk_size must be positive")
    try:
        import nvdiffrast.torch as dr
    except Exception as exc:  # pragma: no cover - environment-specific.
        raise RuntimeError("nvdiffrast is required for the 4096 visibility pass") from exc

    live_vertices = mesh.vertices.to(device=device, dtype=torch.float32).contiguous()
    live_faces = mesh.faces.to(device=device, dtype=torch.int32).contiguous()
    vertex_q = live_vertices * (2.0 * float(global_camera["mesh_scale"]))
    camera_points = core._camera_q_to_points(
        vertex_q,
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
    )
    vertex_depth = -camera_points[:, 2]
    finite_depth = vertex_depth[torch.isfinite(vertex_depth) & (vertex_depth > 0)]
    if finite_depth.numel() == 0:
        raise RuntimeError("global baseline mesh has no finite positive camera depth")
    near = max(1e-4, float(finite_depth.min().item()) * 0.5)
    far = max(near + 1e-3, float(finite_depth.max().item()) * 1.5 + 1.0)
    clip, _, _ = _camera_clip_vertices(
        live_vertices,
        camera=global_camera,
        resolution=int(resolution),
        near=near,
        far=far,
    )
    face_normals = _global_face_normals(mesh).to(device=device, dtype=torch.float32)
    normal_indices = torch.arange(face_normals.shape[0], device=device, dtype=torch.int32)[:, None].expand(-1, 3).contiguous()
    glctx = dr.RasterizeCudaContext(device=device)
    best_z = torch.full((resolution, resolution), float("inf"), device=device, dtype=torch.float32)
    depth_map = torch.full_like(best_z, float("inf"))
    triangle_id_map = torch.full((resolution, resolution), -1, device=device, dtype=torch.int32)
    facing_map = torch.zeros_like(best_z)
    # ``_camera_q_to_points`` already places the object in camera space at
    # z=-distance.  The camera center itself is the camera-space origin; the
    # -distance translation is not a second camera-position translation.
    camera_center = torch.zeros(3, device=device)

    for face_start in range(0, int(live_faces.shape[0]), int(face_chunk_size)):
        face_end = min(face_start + int(face_chunk_size), int(live_faces.shape[0]))
        faces = live_faces[face_start:face_end].contiguous()
        rast, _ = dr.rasterize(glctx, clip[None], faces, (resolution, resolution))
        rast0 = rast[0]
        valid = rast0[..., 3] > 0
        if not bool(valid.any().item()):
            del rast, rast0, faces
            continue
        raster_z = torch.where(valid, rast0[..., 2], torch.full_like(rast0[..., 2], float("inf")))
        better = valid & (raster_z < best_z)
        if bool(better.any().item()):
            tri_local = (rast0[..., 3].to(torch.long) - 1).clamp_min(0)
            tri_global = tri_local + int(face_start)
            interp_depth = dr.interpolate(
                vertex_depth[None, :, None], rast, faces
            )[0][0, ..., 0]
            interp_position = dr.interpolate(
                camera_points[None], rast, faces
            )[0][0]
            interp_normal = dr.interpolate(
                face_normals[None], rast, normal_indices[face_start:face_end]
            )[0][0]
            view = F.normalize(camera_center[None, None, :] - interp_position, dim=-1, eps=1e-12)
            interp_facing = (interp_normal * view).sum(dim=-1).abs().clamp(0.0, 1.0)
            best_z = torch.where(better, raster_z, best_z)
            depth_map = torch.where(better, interp_depth, depth_map)
            triangle_id_map = torch.where(better, tri_global.to(torch.int32), triangle_id_map)
            facing_map = torch.where(better, interp_facing, facing_map)
            del tri_local, tri_global, interp_depth, interp_position, interp_normal, view, interp_facing
        del rast, rast0, faces

    valid = triangle_id_map >= 0
    result = {
        "triangle_id": triangle_id_map.detach().cpu(),
        "depth": depth_map.detach().cpu(),
        "foreground": valid.detach().cpu(),
        "facing": facing_map.detach().cpu(),
        "near": float(near),
        "far": float(far),
        "resolution": int(resolution),
        "renderer": "nvdiffrast geometry-only rasterization; exact global camera; top-origin y corrected",
    }
    del glctx, live_vertices, live_faces, camera_points, vertex_depth, clip, face_normals, normal_indices
    _empty_cuda_cache()
    return result


def _save_gray(path: Path, value: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> None:
    array = value.detach().cpu().to(torch.float32)
    finite = torch.isfinite(array)
    if mask is not None:
        finite &= mask.detach().cpu().bool()
    if bool(finite.any().item()):
        valid = array[finite]
        lo = float(torch.quantile(valid, 0.01).item())
        hi = float(torch.quantile(valid, 0.99).item())
        if hi <= lo:
            hi = lo + 1e-6
        normalized = ((array - lo) / (hi - lo)).clamp(0.0, 1.0)
    else:
        normalized = torch.zeros_like(array)
    normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
    if mask is not None:
        normalized = normalized * mask.detach().cpu().to(normalized.dtype)
    Image.fromarray((normalized.numpy() * 255.0).round().astype(np.uint8), mode="L").save(path)


def _save_triangle_debug(path: Path, triangle_id: torch.Tensor) -> None:
    ids = triangle_id.detach().cpu().to(torch.int64)
    valid = ids >= 0
    safe = ids.clamp_min(0).numpy().astype(np.uint64)
    r = ((safe * 1664525 + 1013904223) & 255).astype(np.uint8)
    g = ((safe * 22695477 + 1) & 255).astype(np.uint8)
    b = ((safe * 1103515245 + 12345) & 255).astype(np.uint8)
    image = np.stack((r, g, b), axis=-1)
    image[~valid.numpy()] = 0
    Image.fromarray(image, mode="RGB").save(path)


def _save_visibility_debug(output_dir: Path, buffers: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        output_dir / "depth.pt",
        {
            "depth": buffers["depth"],
            "near": buffers["near"],
            "far": buffers["far"],
            "renderer": buffers.get("renderer", ""),
        },
    )
    _atomic_torch_save(output_dir / "triangle_id.pt", {"triangle_id": buffers["triangle_id"]})
    _atomic_torch_save(output_dir / "foreground.pt", {"foreground": buffers["foreground"]})
    _atomic_torch_save(output_dir / "facing.pt", {"facing": buffers["facing"]})
    _save_gray(output_dir / "depth_debug.png", buffers["depth"], mask=buffers["foreground"])
    _save_triangle_debug(output_dir / "triangle_id_debug.png", buffers["triangle_id"])
    Image.fromarray((buffers["foreground"].numpy().astype(np.uint8) * 255), mode="L").save(output_dir / "foreground.png")
    _save_gray(output_dir / "facing_debug.png", buffers["facing"], mask=buffers["foreground"])


def _load_visibility_debug(output_dir: Path) -> Optional[Dict[str, Any]]:
    paths = {
        "depth": output_dir / "depth.pt",
        "triangle_id": output_dir / "triangle_id.pt",
        "foreground": output_dir / "foreground.pt",
        "facing": output_dir / "facing.pt",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    depth_payload = torch.load(paths["depth"], map_location="cpu", weights_only=False)
    tri_payload = torch.load(paths["triangle_id"], map_location="cpu", weights_only=False)
    fg_payload = torch.load(paths["foreground"], map_location="cpu", weights_only=False)
    facing_payload = torch.load(paths["facing"], map_location="cpu", weights_only=False)
    renderer = str(depth_payload.get("renderer", ""))
    if "top-origin y corrected" not in renderer:
        # Do not reuse pre-fix visibility caches: their raster rows may be
        # vertically flipped relative to the canonical image coordinates.
        return None
    return {
        "depth": depth_payload["depth"].to(torch.float32),
        "triangle_id": tri_payload["triangle_id"].to(torch.int32),
        "foreground": fg_payload["foreground"].bool(),
        "facing": facing_payload["facing"].to(torch.float32),
        "near": float(depth_payload.get("near", 0.0)),
        "far": float(depth_payload.get("far", 0.0)),
        "resolution": int(depth_payload["depth"].shape[0]),
        "renderer": renderer,
    }


def _project_global_normalized_points(
    points: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return core._project_global_q_to_4096(
        points.to(torch.float32) * (2.0 * float(global_camera["mesh_scale"])),
        global_camera=global_camera,
    )


def _lookup_visibility(
    points_global: torch.Tensor,
    face_ids: torch.Tensor,
    *,
    buffers: Mapping[str, torch.Tensor],
    global_camera: Mapping[str, float],
    eps_z: float,
    sigma_z: float,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query a soft 3x3 z-buffer visibility for a batch of global points."""
    if points_global.ndim != 2 or points_global.shape[1] != 3:
        raise ValueError("global points must have shape [N,3]")
    device = points_global.device
    depth_map = buffers["depth"].to(device=device, dtype=torch.float32)
    triangle_map = buffers["triangle_id"].to(device=device, dtype=torch.int64)
    height, width = depth_map.shape
    uv, point_depth, finite = _project_global_normalized_points(points_global, global_camera=global_camera)
    base_x = torch.round(uv[:, 0]).to(torch.long)
    base_y = torch.round(uv[:, 1]).to(torch.long)
    inside = finite & (base_x >= 0) & (base_x < width) & (base_y >= 0) & (base_y < height)
    offsets = (-1, 0, 1)
    depth_candidates: List[torch.Tensor] = []
    tri_candidates: List[torch.Tensor] = []
    valid_candidates: List[torch.Tensor] = []
    for dy in offsets:
        for dx in offsets:
            x = (base_x + int(dx)).clamp(0, width - 1)
            y = (base_y + int(dy)).clamp(0, height - 1)
            candidate_depth = depth_map[y, x]
            candidate_tri = triangle_map[y, x]
            candidate_valid = inside & torch.isfinite(candidate_depth) & (candidate_tri >= 0)
            depth_candidates.append(candidate_depth)
            tri_candidates.append(candidate_tri)
            valid_candidates.append(candidate_valid)
    candidate_depths = torch.stack(depth_candidates, dim=1)
    candidate_triangles = torch.stack(tri_candidates, dim=1)
    candidate_valid = torch.stack(valid_candidates, dim=1)
    candidate_delta = (candidate_depths - point_depth[:, None]).abs()
    exact = candidate_valid & (candidate_triangles == face_ids.to(torch.long)[:, None])
    has_exact = exact.any(dim=1)
    has_any = candidate_valid.any(dim=1)
    exact_scores = torch.where(exact, candidate_delta, torch.full_like(candidate_delta, float("inf")))
    any_scores = torch.where(candidate_valid, candidate_delta, torch.full_like(candidate_delta, float("inf")))
    exact_best = exact_scores.argmin(dim=1)
    any_best = any_scores.argmin(dim=1)
    best = torch.where(has_exact, exact_best, any_best)
    chosen_depth = candidate_depths.gather(1, best[:, None]).squeeze(1)
    chosen_tri = candidate_triangles.gather(1, best[:, None]).squeeze(1)
    chosen_delta = (chosen_depth - point_depth).abs()
    behind_delta = point_depth - chosen_depth
    sigma = max(float(sigma_z), 1e-12)
    visibility = torch.exp(-torch.relu(chosen_delta - float(eps_z)).square() / (2.0 * sigma * sigma))
    visibility = torch.where(has_any, visibility, torch.zeros_like(visibility))
    visibility = torch.where(
        behind_delta > (3.0 * sigma + float(eps_z)),
        torch.zeros_like(visibility),
        visibility,
    ).clamp(0.0, 1.0)
    exact_rate = float(has_exact.to(torch.float32).mean().item()) if has_exact.numel() else 0.0
    fallback = has_any & ~has_exact
    stats = {
        "exact_triangle_id_match_count": int(has_exact.sum().item()),
        "exact_triangle_id_match_rate": exact_rate,
        "three_by_three_depth_fallback_count": int(fallback.sum().item()),
        "three_by_three_depth_fallback_rate": float(fallback.to(torch.float32).mean().item()) if fallback.numel() else 0.0,
        "no_valid_depth_count": int((~has_any).sum().item()),
        "no_valid_depth_rate": float((~has_any).to(torch.float32).mean().item()) if has_any.numel() else 0.0,
        "outside_global_4096_count": int((~inside).sum().item()),
        "outside_global_4096_rate": float((~inside).to(torch.float32).mean().item()) if inside.numel() else 0.0,
        "depth_tolerance_camera_units": float(eps_z),
        "depth_sigma_camera_units": float(sigma_z),
        "depth_rule": "exp(-relu(abs(point_depth-zbuf)-eps)^2/(2*sigma^2)); behind > eps+3sigma is zero",
    }
    return visibility, stats, uv, point_depth, chosen_tri.to(torch.long)


def _facing_weight(
    points_global: torch.Tensor,
    normals_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    good_deg: float,
    bad_deg: float,
) -> torch.Tensor:
    if normals_global.shape != points_global.shape:
        raise ValueError("global normals and points must have the same shape")
    q = points_global * (2.0 * float(global_camera["mesh_scale"]))
    physical = core._camera_q_to_points(
        q,
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
    )
    # The exact Pixal3D projection convention has the camera at camera-space
    # origin and the object translated to negative z by ``distance``.
    center = physical.new_zeros(3)
    view = F.normalize(center[None] - physical, dim=1, eps=1e-12)
    normals = F.normalize(normals_global.to(torch.float32), dim=1, eps=1e-12)
    cosine = (normals * view).sum(dim=1).abs().clamp(0.0, 1.0)
    good = math.cos(math.radians(float(good_deg)))
    bad = math.cos(math.radians(float(bad_deg)))
    if good <= bad:
        raise ValueError("facing-good-deg must be smaller than facing-bad-deg")
    u = ((cosine - bad) / (good - bad)).clamp(0.0, 1.0)
    return _smoothstep01(u)


def _tile_weight(
    uv_global: torch.Tensor,
    *,
    transform: Any,
    feather_pixels: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0, y0, _, _ = transform.box
    uv_tile = torch.empty_like(uv_global)
    uv_tile[:, 0] = (uv_global[:, 0] - float(x0)) * float(transform.crop_to_output_scale_x)
    uv_tile[:, 1] = (uv_global[:, 1] - float(y0)) * float(transform.crop_to_output_scale_y)
    edge = torch.minimum(
        torch.minimum(uv_tile[:, 0], uv_tile[:, 1]),
        torch.minimum(float(TILE_SIZE - 1) - uv_tile[:, 0], float(TILE_SIZE - 1) - uv_tile[:, 1]),
    )
    inside = (uv_tile[:, 0] >= 0.0) & (uv_tile[:, 0] < float(TILE_SIZE)) & (uv_tile[:, 1] >= 0.0) & (uv_tile[:, 1] < float(TILE_SIZE))
    if float(feather_pixels) <= 0.0:
        weight = inside.to(torch.float32)
    else:
        weight = _smoothstep01(edge / float(feather_pixels)) * inside.to(torch.float32)
    return weight.clamp(0.0, 1.0), uv_tile


def _foreground_weight(mask: Optional[Image.Image], uv_global: torch.Tensor, *, dilation: int = 5) -> torch.Tensor:
    if mask is None:
        return torch.ones(uv_global.shape[0], device=uv_global.device, dtype=torch.float32)
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    values = torch.from_numpy(array).to(device=uv_global.device, dtype=torch.float32)[None, None]
    if dilation > 0:
        kernel = 2 * int(dilation) + 1
        values = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=int(dilation))
    height, width = array.shape
    x = torch.round(uv_global[:, 0]).to(torch.long).clamp(0, width - 1)
    y = torch.round(uv_global[:, 1]).to(torch.long).clamp(0, height - 1)
    inside = (uv_global[:, 0] >= 0.0) & (uv_global[:, 0] < width) & (uv_global[:, 1] >= 0.0) & (uv_global[:, 1] < height)
    return values[0, 0, y, x] * inside.to(torch.float32)


def _aggregate_contributions(
    values: torch.Tensor,
    rows: torch.Tensor,
    weights: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if values.shape[0] != rows.shape[0] or rows.shape[0] != weights.shape[0]:
        raise ValueError("contribution values, rows and weights have inconsistent lengths")
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    output = torch.zeros((int(count), values.shape[1]), device=values.device, dtype=torch.float32)
    denom = torch.zeros((int(count), 1), device=values.device, dtype=torch.float32)
    output.index_add_(0, rows, values.to(torch.float32) * weights[:, None])
    denom.index_add_(0, rows, weights[:, None])
    output = output / denom.clamp_min(1e-12)
    return output[:, 0] if squeeze else output


def _weight_records(values: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, value in values.items():
        result[name] = _tensor_stats(value)
    return result


@torch.no_grad()
def _prepare_local_material_guidance(
    *,
    mapping: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    global_camera: Mapping[str, float],
    transform: Any,
    foreground_mask: Optional[Image.Image],
    fixed_shape_coords: torch.Tensor,
    global_face_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Transfer projected mesh visibility through O-Voxel support to SLat.

    Visibility is deliberately discrete and has a single source of truth:
    the global mesh triangle ID written by the canonical 4096 raster pass.
    The material correspondence already contains the nearest global mesh face
    for every local C1024 O-Voxel, so the transfer is:

        visible projected mesh face
            -> nearest mesh face of local C1024 O-Voxel
            -> C64 SLat parent (floor(C1024 / 16))
            -> broadcast to the local C1024 support.

    The old implementation projected each resampling point back into a
    soft 3x3 depth lookup and then multiplied facing/tile feather weights.
    That was both a different visibility definition and a lossy weighting
    stage.  In this route, a visible SLat parent receives HR and every other
    parent receives G: ``w_final`` is the inherited binary visibility mask.
    """
    triangle_map = buffers["triangle_id"].detach().cpu().to(torch.int64)
    if triangle_map.ndim != 2:
        raise ValueError(f"triangle_id buffer must be [H,W], got {tuple(triangle_map.shape)}")
    x0, y0, x1, y1 = (int(value) for value in transform.box)
    height, width = triangle_map.shape
    x0_clip, y0_clip = max(0, x0), max(0, y0)
    x1_clip, y1_clip = min(width, x1), min(height, y1)
    if x1_clip <= x0_clip or y1_clip <= y0_clip:
        raise ValueError(f"tile box {transform.box} does not intersect the visibility buffer")
    tile_triangles = triangle_map[y0_clip:y1_clip, x0_clip:x1_clip]
    visible_face_ids = torch.unique(tile_triangles[tile_triangles >= 0]).to(torch.long)
    visible_face_ids = visible_face_ids[visible_face_ids < int(global_face_count)]
    global_face_visible = torch.zeros(int(global_face_count), dtype=torch.bool)
    if visible_face_ids.numel():
        global_face_visible[visible_face_ids] = True
    global_face_visible = global_face_visible.to(device=device)

    rows = mapping["contribution_rows"].to(device=device, dtype=torch.long)
    contribution_weights = mapping["contribution_weights"].to(device=device, dtype=torch.float32)
    contribution_face_ids = mapping["contribution_global_face_ids"].to(device=device, dtype=torch.long)
    entries = int(mapping["local_c1024_coords"].shape[0])
    if rows.numel() != contribution_face_ids.numel() or rows.numel() != contribution_weights.numel():
        raise ValueError("visibility contribution mapping has inconsistent lengths")
    if bool((contribution_face_ids < 0).any().item()) or bool(
        (contribution_face_ids >= int(global_face_count)).any().item()
    ):
        raise RuntimeError("visibility contribution mapping refers to an invalid global face")

    contribution_visible = global_face_visible.index_select(0, contribution_face_ids)
    # The complete local-neighborhood query is the C1024 visibility source:
    # any effective nearby mesh contribution that is visible in the projected
    # triangle-ID map makes this O-Voxel visible.  The nearest representative
    # is retained separately as a diagnostic only.
    nearby_visible_score = torch.zeros(entries, device=device, dtype=torch.float32)
    if hasattr(nearby_visible_score, "scatter_reduce_"):
        nearby_visible_score.scatter_reduce_(
            0,
            rows,
            contribution_visible.to(torch.float32),
            reduce="amax",
            include_self=False,
        )
    else:  # pragma: no cover - retained for older torch installations.
        for row in torch.unique(rows, sorted=True).tolist():
            row_value = int(row)
            nearby_visible_score[row_value] = contribution_visible[rows == row_value].to(torch.float32).amax()
    nearby_visible = nearby_visible_score > 0.5
    nearby_ratio = _aggregate_contributions(
        contribution_visible.to(torch.float32), rows, contribution_weights, entries
    ).clamp(0.0, 1.0)

    representative_face_ids = mapping["representative_global_face_ids"].to(
        device=device, dtype=torch.long
    )
    if representative_face_ids.numel() != entries:
        raise ValueError("representative global face IDs do not match local C1024 support")
    if bool((representative_face_ids < 0).any().item()) or bool(
        (representative_face_ids >= int(global_face_count)).any().item()
    ):
        raise RuntimeError("representative visibility mapping refers to an invalid global face")
    representative_visible = global_face_visible.index_select(0, representative_face_ids)
    c1024_visible = nearby_visible

    # Map local C1024 coordinates to the exact C64 sparse encoder support.  A
    # four-stage SparseSpatial2Channel encoder has the same integer ancestry as
    # floor(coord / 16); the explicit key join also handles sparse SLat holes.
    local_coords = mapping["local_c1024_coords"].to(device=device, dtype=torch.long)
    c1024_parent = torch.div(local_coords, 16, rounding_mode="floor")
    slat_coords = fixed_shape_coords.detach().to(device=device, dtype=torch.long)
    if slat_coords.ndim != 2 or slat_coords.shape[1] not in (3, 4):
        raise ValueError(f"fixed shape coords must be [N,3] or [N,4], got {tuple(slat_coords.shape)}")
    slat_xyz = slat_coords[:, -3:]
    if bool(((slat_xyz < 0) | (slat_xyz >= int(LATENT_RESOLUTION))).any().item()):
        raise RuntimeError("fixed shape SLat coordinates lie outside C64")

    def _coord_key(coords: torch.Tensor, resolution: int) -> torch.Tensor:
        return (coords[:, 0] * int(resolution) + coords[:, 1]) * int(resolution) + coords[:, 2]

    slat_keys = _coord_key(slat_xyz, LATENT_RESOLUTION)
    parent_keys = _coord_key(c1024_parent, LATENT_RESOLUTION)
    order = torch.argsort(slat_keys, stable=True)
    sorted_keys = slat_keys.index_select(0, order)
    positions = torch.searchsorted(sorted_keys, parent_keys)
    matched = positions < sorted_keys.numel()
    safe_positions = positions.clamp_max(max(int(sorted_keys.numel()) - 1, 0))
    if sorted_keys.numel():
        matched &= sorted_keys.index_select(0, safe_positions) == parent_keys
    else:
        matched.zero_()
    c1024_to_slat = torch.full((entries,), -1, device=device, dtype=torch.long)
    if bool(matched.any().item()):
        c1024_to_slat[matched] = order.index_select(0, safe_positions[matched])

    slat_total = torch.zeros(slat_xyz.shape[0], device=device, dtype=torch.long)
    slat_visible_count = torch.zeros_like(slat_total)
    if bool(matched.any().item()):
        slat_indices = c1024_to_slat[matched]
        slat_total.index_add_(0, slat_indices, torch.ones_like(slat_indices))
        slat_visible_count.index_add_(0, slat_indices, c1024_visible[matched].to(torch.long))
    slat_visibility_ratio = (
        slat_visible_count.to(torch.float32)
        / slat_total.clamp_min(1).to(torch.float32)
    ).clamp(0.0, 1.0)
    slat_visible = slat_visible_count > 0

    # The SLat decision is inherited back to each C1024 O-Voxel.  Unmatched
    # support entries are retained conservatively from their nearest mesh face
    # instead of being silently forced to G.
    final = torch.zeros(entries, device=device, dtype=torch.float32)
    if bool(matched.any().item()):
        final[matched] = slat_visible.index_select(0, c1024_to_slat[matched]).to(torch.float32)
    if bool((~matched).any().item()):
        final[~matched] = c1024_visible[~matched].to(torch.float32)
    # Keep the endpoint decision explicitly binary.  This is the semantic
    # boundary used by the experiment: a visible C64 parent is allowed to
    # take the HR endpoint, while an unobserved parent stays on G.  The
    # facing/tile/foreground values below remain diagnostics and must not
    # silently attenuate this geometry-derived decision.
    final = torch.where(final > 0.5, torch.ones_like(final), torch.zeros_like(final))
    if not bool(((final == 0.0) | (final == 1.0)).all().item()):
        raise RuntimeError("final visibility guidance is not binary")

    # These are retained as named diagnostics for compatibility with the
    # existing visualizer, but they are intentionally not part of w_final.
    rep_points = mapping["representative_global_surface_points"].to(device=device, dtype=torch.float32)
    rep_normals = mapping["representative_normals"].to(device=device, dtype=torch.float32)
    rep_uv, rep_depth, rep_finite = _project_global_normalized_points(
        rep_points, global_camera=global_camera
    )
    facing = _facing_weight(
        rep_points,
        rep_normals,
        global_camera=global_camera,
        good_deg=float(args.facing_good_deg),
        bad_deg=float(args.facing_bad_deg),
    )
    tile, rep_uv_tile = _tile_weight(
        rep_uv, transform=transform, feather_pixels=float(args.tile_feather_pixels)
    )
    fg = _foreground_weight(
        foreground_mask,
        rep_uv,
        dilation=int(args.foreground_dilation_pixels),
    )
    visible_mask = final > 0.5
    frontal_mask = facing > 0.5
    grazing_mask = ~frontal_mask
    tile_outside_mask = tile <= 0.0

    diagnostics = {
        "visibility": _tensor_stats(final),
        "facing": _tensor_stats(facing),
        "tile_weight": _tensor_stats(tile),
        "foreground_weight": _tensor_stats(fg),
        "final_weight": _tensor_stats(final),
        "visible_c1024_count": int(c1024_visible.sum().item()),
        "nearest_visible_c1024_count": int(representative_visible.sum().item()),
        "visible_slat_count": int(slat_visible.sum().item()),
        "slat_token_count": int(slat_visible.numel()),
        "unmatched_c1024_to_slat_count": int((~matched).sum().item()),
        "global_visible_face_count_in_tile": int(visible_face_ids.numel()),
        "global_face_count": int(global_face_count),
        "tile_triangle_pixels_with_face_id": int((tile_triangles >= 0).sum().item()),
        "visible_and_frontal_count": int((visible_mask & frontal_mask).sum().item()),
        "visible_grazing_count": int((visible_mask & grazing_mask).sum().item()),
        "occluded_or_back_count": int((~visible_mask).sum().item()),
        "visible_foreground_count": int((visible_mask & (fg > 0.5)).sum().item()),
        "visible_background_count": int((visible_mask & (fg <= 0.5)).sum().item()),
        "outside_tile_count": int(tile_outside_mask.sum().item()),
        "outside_global_4096_count": int((~rep_finite).sum().item()),
        "visibility_lookup": {
            "rule": "exact global triangle ID membership in the corrected top-origin tile raster crop",
            "depth_lookup_used": False,
            "y_axis": "nvdiffrast bottom-origin corrected to Pixal3D top-origin",
        },
        "transfer_rule": {
            "mesh_to_c1024": "any effective nearby global mesh contribution already selected by material correspondence",
            "c1024_to_slat": "exact sparse ancestry floor(local_c1024_coord / 16) joined to fixed C64 SLat coords",
            "slat_visibility": "visible if at least one descendant C1024 O-Voxel has a visible nearby mesh contribution",
            "final_weight": "exact binary inherited SLat visibility; no facing/tile/foreground multiplication",
            "facing_tile_foreground": "diagnostic only; not multiplied into final_weight",
        },
        "nearby_visibility_ratio": _tensor_stats(nearby_ratio),
        "slat_visibility_ratio": _tensor_stats(slat_visibility_ratio),
        "representative_rule": mapping.get("representative_selection", "maximum contribution weight"),
        "representative_projected_finite_fraction": float(rep_finite.to(torch.float32).mean().item()) if rep_finite.numel() else 0.0,
        "representative_face_ids": representative_face_ids.detach().cpu(),
        "representative_uv_global": rep_uv.detach().cpu(),
        "representative_uv_tile": rep_uv_tile.detach().cpu(),
        "representative_depth": rep_depth.detach().cpu(),
    }
    values = {
        "w_visible": final,
        "w_facing": facing,
        "w_tile": tile,
        "w_fg": fg,
        "w_final": final,
        "c1024_mesh_visible": c1024_visible,
        "c1024_nearest_mesh_visible": representative_visible,
        "c1024_nearby_mesh_visible": nearby_visible,
        "c1024_nearby_visibility_ratio": nearby_ratio,
        "slat_coords": slat_coords,
        "slat_visible": slat_visible,
        "slat_visibility_ratio": slat_visibility_ratio,
        "c1024_to_slat": c1024_to_slat,
        "contribution_visible": contribution_visible,
        "contribution_uv_global": rep_uv,
        "contribution_uv_tile": rep_uv_tile,
    }
    return values, diagnostics


def _heat_color(values: torch.Tensor) -> Image.Image:
    value = values.detach().cpu().to(torch.float32).clamp(0.0, 1.0).numpy()
    # A compact blue-cyan-yellow-red map, kept dependency-free for batch runs.
    red = np.clip(2.0 * value, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(value - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - value), 0.0, 1.0)
    return Image.fromarray((np.stack((red, green, blue), axis=-1) * 255.0).round().astype(np.uint8), mode="RGB")


def _scatter_tile_heatmap(
    uv_tile: torch.Tensor,
    values: torch.Tensor,
    *,
    size: int = TILE_SIZE,
) -> torch.Tensor:
    uv = uv_tile.detach().cpu().to(torch.float32)
    val = values.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
    image = torch.zeros((size, size), dtype=torch.float32)
    count = torch.zeros((size, size), dtype=torch.float32)
    x = torch.round(uv[:, 0]).to(torch.long)
    y = torch.round(uv[:, 1]).to(torch.long)
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size) & torch.isfinite(val)
    if bool(valid.any().item()):
        x = x[valid]
        y = y[valid]
        v = val[valid]
        index = y * size + x
        # Mean is less sensitive to sparse point density than last-write wins.
        flat = image.reshape(-1)
        flat_count = count.reshape(-1)
        flat.index_add_(0, index, v)
        flat_count.index_add_(0, index, torch.ones_like(v))
        image = (flat / flat_count.clamp_min(1.0)).reshape(size, size)
        # Fill holes with zero; the overlay remains explicitly sparse/diagnostic.
    return image


def _save_tile_guidance_visuals(
    tile_dir: Path,
    *,
    hr_tile: Image.Image,
    uv_tile: torch.Tensor,
    values: Mapping[str, torch.Tensor],
) -> None:
    heatmaps: Dict[str, Image.Image] = {}
    for name in ("w_visible", "w_facing", "w_tile", "w_final"):
        heat = _scatter_tile_heatmap(uv_tile, values[name])
        image = _heat_color(heat)
        image.save(tile_dir / f"{name}.png")
        heatmaps[name] = image
    final_heat = _scatter_tile_heatmap(uv_tile, values["w_final"])
    base = np.asarray(hr_tile.convert("RGB"), dtype=np.float32)
    overlay = np.asarray(_heat_color(final_heat), dtype=np.float32)
    alpha = (final_heat.numpy()[..., None] > 0).astype(np.float32) * 0.60
    composed = (base * (1.0 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(composed, mode="RGB").save(tile_dir / "weight_overlay.png")


def _decode_endpoint(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_norm: SparseTensor,
    query_points: torch.Tensor,
    query_chunk_size: int,
    label: str,
) -> Tuple[MeshWithVoxel, torch.Tensor, Dict[str, Any]]:
    shape_input = _fresh_sparse(shape_denorm)
    texture_denorm = _denormalize_slat(texture_norm, pipeline.tex_slat_normalization)
    texture_input = _fresh_sparse(texture_denorm)
    started = time.perf_counter()
    decoded = pipeline.decode_latent(shape_input, texture_input, OVOXEL_RESOLUTION)
    _sync_cuda()
    if len(decoded) != 1:
        raise RuntimeError(f"{label}: decoder returned {len(decoded)} meshes")
    mesh = core._validate_mesh(decoded[0], label)
    queried = degradation._query_mesh_chunked(mesh, query_points, int(query_chunk_size))
    if not torch.isfinite(queried).all():
        raise RuntimeError(f"{label}: decoded PBR query is non-finite")
    stats = {
        "decode_seconds": float(time.perf_counter() - started),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "active_ovoxels": int(mesh.coords.shape[0]),
        "query_tokens": int(queried.shape[0]),
        "pbr_range": core._tensor_range(mesh.attrs),
    }
    return mesh, queried, stats


def _geometry_check(left: MeshWithVoxel, right: MeshWithVoxel) -> Dict[str, Any]:
    return degradation._mesh_geometry_check(left, right)


def _blend_pbr_fields(
    g_attrs: torch.Tensor,
    hr_attrs: torch.Tensor,
    weight: torch.Tensor,
    *,
    timestep: float,
    schedule: str,
    channel_mode: str,
    metallic_scale: float,
    roughness_scale: float,
    fixed_hr_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fixed_hr_weight is not None:
        fixed = float(fixed_hr_weight)
        if not 0.0 <= fixed <= 1.0:
            raise ValueError(f"fixed HR fusion weight must lie in [0,1], got {fixed}")
        # This mode intentionally ignores the visibility mask and timestep
        # schedule: every C1024 support entry uses the same HR/G ratio.
        lam = torch.full_like(weight, fixed, dtype=torch.float32)
    else:
        t = float(timestep)
        if schedule == "sin2":
            schedule_value = math.sin(0.5 * math.pi * (1.0 - t)) ** 2
        elif schedule == "constant":
            schedule_value = 1.0
        elif schedule == "linear":
            schedule_value = 1.0 - t
        elif schedule == "late":
            schedule_value = (1.0 - t) ** 2
        else:
            raise ValueError(f"unknown fusion time schedule: {schedule}")
        schedule_value = float(max(0.0, min(1.0, schedule_value)))
        lam = (weight.to(torch.float32) * schedule_value).clamp(0.0, 1.0)
    result = g_attrs.to(torch.float32).clone()
    rgb_lambda = lam[:, None]
    result[:, 0:3] = g_attrs[:, 0:3] + rgb_lambda * (hr_attrs[:, 0:3] - g_attrs[:, 0:3])
    if channel_mode == "rgb_mr_soft":
        result[:, 3:4] = g_attrs[:, 3:4] + (lam * float(metallic_scale))[:, None] * (hr_attrs[:, 3:4] - g_attrs[:, 3:4])
        result[:, 4:5] = g_attrs[:, 4:5] + (lam * float(roughness_scale))[:, None] * (hr_attrs[:, 4:5] - g_attrs[:, 4:5])
    elif channel_mode == "all":
        result = g_attrs + rgb_lambda * (hr_attrs - g_attrs)
    elif channel_mode != "rgb_only":
        raise ValueError(f"unknown channel mode: {channel_mode}")
    result = result.clamp(0.0, 1.0)
    return result, lam


def _flow_step_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    consumed = {
        "steps", "rescale_t", "verbose", "tqdm_desc", "record_trajectory",
        "trajectory_device", "return_model_history",
    }
    return {key: value for key, value in params.items() if key not in consumed}


@torch.no_grad()
def _run_native_pure_hr_flow(
    *,
    pipeline: Any,
    initial_state: SparseTensor,
    shape_condition: SparseTensor,
    condition: Mapping[str, Any],
    params: Mapping[str, Any],
    noise_timestep: float,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    effective_params = {**pipeline.tex_slat_sampler_params, **dict(params)}
    schedule = [float(value) for value in sampler.timestep_schedule(int(effective_params["steps"]), float(effective_params["rescale_t"]))]
    matches = [index for index, value in enumerate(schedule) if abs(value - float(noise_timestep)) <= 1e-6]
    if len(matches) != 1:
        raise RuntimeError(f"noise timestep {noise_timestep} is not an exact native schedule point: {schedule}")
    start = matches[0]
    state = initial_state
    model_kwargs = _flow_step_kwargs(effective_params)
    started = time.perf_counter()
    if pipeline.low_vram:
        model.to(torch.device(pipeline.device))
    try:
        for step, (t, t_prev) in enumerate(zip(schedule[start:-1], schedule[start + 1:])):
            result = sampler.sample_once(
                model,
                state,
                float(t),
                float(t_prev),
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                concat_cond=shape_condition,
                **model_kwargs,
            )
            state = result.pred_x_prev
            if not torch.equal(state.coords, initial_state.coords):
                raise RuntimeError(f"pure HR flow changed sparse support at step {step}")
    finally:
        if pipeline.low_vram:
            model.cpu()
    _sync_cuda()
    return state, {
        "route": "native FlowEulerSampler.sample_once suffix; no endpoint guidance",
        "native_schedule": schedule,
        "schedule_start_index": int(start),
        "flow_steps": int(len(schedule) - 1 - start),
        "noise_timestep": float(noise_timestep),
        "effective_clean_coefficient": float(1.0 - float(noise_timestep)),
        "shape_flow_called": False,
        "shape_slat_sampler_sample_called": False,
        "flow_seconds": float(time.perf_counter() - started),
    }


def _strict_endpoint_check(reference: SparseTensor, candidate: SparseTensor, label: str) -> Dict[str, Any]:
    same_coords = bool(torch.equal(reference.coords, candidate.coords))
    same_shape = tuple(reference.feats.shape) == tuple(candidate.feats.shape)
    if not same_coords or not same_shape:
        raise RuntimeError(f"{label}: sparse support/feature shape changed")
    return {
        "coords_equal": same_coords,
        "token_count": int(candidate.coords.shape[0]),
        "token_count_equal": int(reference.coords.shape[0]) == int(candidate.coords.shape[0]),
        "feature_shape_equal": same_shape,
    }


@torch.no_grad()
def _run_visibility_guided_flow(
    *,
    pipeline: Any,
    initial_state: SparseTensor,
    pure_hr_norm: SparseTensor,
    fixed_shape_norm: SparseTensor,
    shape_denorm: SparseTensor,
    condition: Mapping[str, Any],
    params: Mapping[str, Any],
    g_attrs: torch.Tensor,
    guidance: Mapping[str, torch.Tensor],
    pbr_encoder: torch.nn.Module,
    geometry_reference: Optional[MeshWithVoxel],
    args: argparse.Namespace,
) -> Tuple[SparseTensor, Dict[str, Any], List[Dict[str, Any]]]:
    """Run one native model forward per step and guide only its clean x0."""
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    device = initial_state.feats.device
    effective_params = {**pipeline.tex_slat_sampler_params, **dict(params)}
    schedule = [float(value) for value in sampler.timestep_schedule(int(effective_params["steps"]), float(effective_params["rescale_t"]))]
    matches = [index for index, value in enumerate(schedule) if abs(value - float(args.noise_timestep)) <= 1e-6]
    if len(matches) != 1:
        raise RuntimeError(f"guided noise timestep {args.noise_timestep} is not an exact native schedule point: {schedule}")
    start = matches[0]
    model_kwargs = _flow_step_kwargs(effective_params)
    state = initial_state
    trajectory: List[Dict[str, Any]] = []
    started = time.perf_counter()
    if pipeline.low_vram:
        model.to(torch.device(pipeline.device))
    if pbr_encoder is None:
        raise RuntimeError("guided flow requires the official PBR encoder")
    try:
        for step, (t, t_prev) in enumerate(zip(schedule[start:-1], schedule[start + 1:])):
            pred_x0, _, pred_v = sampler._get_model_prediction(
                model,
                state,
                float(t),
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                concat_cond=fixed_shape_norm,
                **model_kwargs,
            )
            if not isinstance(pred_x0, SparseTensor) or not isinstance(pred_v, SparseTensor):
                raise RuntimeError("official sampler prediction is not SparseTensor")
            _strict_endpoint_check(initial_state, pred_x0, f"step {step} pred_x0")
            # Decode with fresh SparseTensor objects to avoid the decoder's
            # cached spatial broadcast/index map crossing flow steps.
            hr_mesh, hr_attrs, hr_decode_stats = _decode_endpoint(
                pipeline=pipeline,
                shape_denorm=shape_denorm,
                texture_norm=pred_x0,
                query_points=guidance["representative_local_points"],
                query_chunk_size=int(args.query_chunk_size),
                label=f"tile {args._tile_id:02d} step {step} HR pred_x0",
            )
            fused_attrs, lam = _blend_pbr_fields(
                g_attrs,
                hr_attrs,
                guidance["w_final"],
                timestep=float(t),
                schedule=str(args.fusion_time_schedule),
                channel_mode=str(args.fusion_mode),
                metallic_scale=float(args.metallic_local_scale),
                roughness_scale=float(args.roughness_local_scale),
                fixed_hr_weight=args.per_step_fixed_hr_weight,
            )
            reencoded_raw, reencode_stats = core._encode_local_pbr(
                encoder=pbr_encoder,
                coords=guidance["local_coords"],
                attrs=fused_attrs.detach().cpu(),
                device=device,
                low_vram=bool(args.low_vram),
            )
            reencoded_norm = _normalize_slat(reencoded_raw, pipeline.tex_slat_normalization)
            _strict_endpoint_check(initial_state, reencoded_norm, f"step {step} reencoded endpoint")
            corrected = pred_x0.replace(
                pred_x0.feats + float(args.reencode_guidance_strength) * (reencoded_norm.feats - pred_x0.feats)
            )
            _strict_endpoint_check(initial_state, corrected, f"step {step} corrected endpoint")
            corrected_mesh, cycle_attrs, corrected_decode_stats = _decode_endpoint(
                pipeline=pipeline,
                shape_denorm=shape_denorm,
                texture_norm=corrected,
                query_points=guidance["representative_local_points"],
                query_chunk_size=int(args.query_chunk_size),
                label=f"tile {args._tile_id:02d} step {step} corrected endpoint",
            )
            geometry_check = _geometry_check(hr_mesh, corrected_mesh)
            if not geometry_check["geometry_equal"]:
                raise RuntimeError(f"step {step} texture correction changed decoded geometry: {geometry_check}")

            visible_mask = guidance["w_final"] > 0.7
            invisible_mask = guidance["w_final"] < 0.1
            pbr_hr_g = _channel_mean_abs(hr_attrs, g_attrs, visible_mask)
            pbr_fused_g = _channel_mean_abs(fused_attrs, g_attrs, visible_mask)
            pbr_hr_g_back = _channel_mean_abs(hr_attrs, g_attrs, invisible_mask)
            pbr_fused_g_back = _channel_mean_abs(fused_attrs, g_attrs, invisible_mask)
            cycle_relative = _channel_relative_errors(cycle_attrs, fused_attrs)
            trajectory.append({
                "step": int(step),
                "t": float(t),
                "t_prev": float(t_prev),
                "sigma_t": float(sampler.sigma_min + (1.0 - sampler.sigma_min) * float(t)),
                "effective_clean_coefficient": float(1.0 - float(t)),
                "fusion_schedule_value": (
                    float(lam.mean().item())
                    if args.per_step_fixed_hr_weight is not None
                    else float(lam.mean().item() / max(float(guidance["w_final"].mean().item()), 1e-12))
                ) if guidance["w_final"].numel() else 0.0,
                "fusion_hr_weight": float(lam.mean().item()) if lam.numel() else 0.0,
                "fusion_g_weight": float(1.0 - lam.mean().item()) if lam.numel() else 1.0,
                "fusion_weight_rule": (
                    f"fixed {float(args.per_step_fixed_hr_weight):.6f} HR + "
                    f"{1.0 - float(args.per_step_fixed_hr_weight):.6f} G"
                    if args.per_step_fixed_hr_weight is not None
                    else "visibility weight times timestep schedule"
                ),
                "lambda_stats": _tensor_stats(lam),
                "latent_abs_hr_minus_G": _norm(pred_x0.feats - initial_state.feats),
                "latent_abs_reencoded_minus_hr": _norm(reencoded_norm.feats - pred_x0.feats),
                "latent_abs_corrected_minus_hr": _norm(corrected.feats - pred_x0.feats),
                "latent_relative_hr_minus_G": _relative(pred_x0.feats - initial_state.feats, initial_state.feats),
                "latent_relative_reencoded_minus_hr": _relative(reencoded_norm.feats - pred_x0.feats, pred_x0.feats),
                "latent_relative_corrected_minus_hr": _relative(corrected.feats - pred_x0.feats, pred_x0.feats),
                "PBR_visible_w_gt_0.7_HR_minus_G_mean_abs": pbr_hr_g,
                "PBR_visible_w_gt_0.7_fused_minus_G_mean_abs": pbr_fused_g,
                "PBR_invisible_w_lt_0.1_HR_minus_G_mean_abs": pbr_hr_g_back,
                "PBR_invisible_w_lt_0.1_fused_minus_G_mean_abs": pbr_fused_g_back,
                "cycle_relative_D_E_Fstar_vs_Fstar": cycle_relative,
                "decode": {
                    "hr_pred_x0": hr_decode_stats,
                    "corrected": corrected_decode_stats,
                    "geometry_equal": geometry_check,
                },
                "reencode": reencode_stats,
                "coords_equal_fixed_shape": bool(torch.equal(corrected.coords, fixed_shape_norm.coords)),
                "token_count_equal_fixed_shape": int(corrected.coords.shape[0]) == int(fixed_shape_norm.coords.shape[0]),
                "normalization": "pipeline.tex_slat_normalization for pred_x0/reencoded/corrected",
            })
            corrected_velocity = sampler._xstart_to_pred(state, float(t), corrected)
            if not isinstance(corrected_velocity, SparseTensor):
                corrected_velocity = state.replace(corrected_velocity)
            _strict_endpoint_check(initial_state, corrected_velocity, f"step {step} corrected velocity")
            state = state - float(t - t_prev) * corrected_velocity
            if not torch.equal(state.coords, initial_state.coords):
                raise RuntimeError(f"guided flow changed sparse support after step {step}")
            del hr_mesh, corrected_mesh, hr_attrs, fused_attrs, reencoded_raw, reencoded_norm, corrected, corrected_velocity, pred_x0, pred_v
            _empty_cuda_cache()
    finally:
        if pipeline.low_vram:
            model.cpu()

    # The PBR guidance above is intentionally applied at every native flow
    # step.  With a full-noise start, however, the official encoder projection
    # can pull the final clean endpoint back toward G even when the visible
    # PBR field has a clear HR gain.  Use the already-computed pure-HR endpoint
    # as a final fixed-support anchor only on visible C64 parents.  This does
    # not add a model forward, does not change geometry/support, and leaves
    # unobserved/background parents on the guided (global-prior) endpoint.
    anchor_strength = float(args.final_visible_endpoint_anchor_strength)
    anchor_route = (
        "visible-SLat pure-HR endpoint anchor"
        if anchor_strength > 0.0
        else "visible-SLat pure-HR endpoint anchor disabled (strength=0)"
    )
    visible_tokens = guidance["slat_visible"].to(device=device, dtype=torch.float32)
    if visible_tokens.ndim != 1 or visible_tokens.shape[0] != state.feats.shape[0]:
        raise RuntimeError(
            "visible SLat anchor is not aligned with the texture endpoint: "
            f"weights={tuple(visible_tokens.shape)} endpoint={tuple(state.feats.shape)}"
        )
    if not torch.equal(state.coords, pure_hr_norm.coords):
        raise RuntimeError("pure HR endpoint support differs before visible-SLat anchor")
    if anchor_strength > 0.0:
        state = state.replace(
            state.feats
            + anchor_strength
            * visible_tokens[:, None]
            * (pure_hr_norm.feats.to(device=device) - state.feats)
        )
    _strict_endpoint_check(initial_state, state, "final visible-SLat endpoint anchor")
    _sync_cuda()
    stats = {
        "route": "one HR model forward -> pred_x0 -> decode PBR -> projected mesh -> nearby mesh contributions -> C1024 O-Voxel -> C64 SLat visibility inheritance -> PBR blend -> official PBR encode -> x0 guidance -> official xstart_to_pred -> Euler -> " + anchor_route,
        "native_schedule": schedule,
        "schedule_start_index": int(start),
        "flow_steps": int(len(schedule) - 1 - start),
        "noise_timestep": float(args.noise_timestep),
        "effective_clean_coefficient": float(1.0 - float(args.noise_timestep)),
        "shape_flow_called": False,
        "shape_slat_sampler_sample_called": False,
        "model_forward_count": int(len(trajectory)),
        "flow_seconds": float(time.perf_counter() - started),
        "eta": float(args.reencode_guidance_strength),
        "per_step_fixed_hr_weight": (
            float(args.per_step_fixed_hr_weight)
            if args.per_step_fixed_hr_weight is not None else None
        ),
        "per_step_fixed_g_weight": (
            1.0 - float(args.per_step_fixed_hr_weight)
            if args.per_step_fixed_hr_weight is not None else None
        ),
        "final_visible_endpoint_anchor_strength": anchor_strength,
        "final_visible_endpoint_anchor_token_count": int((visible_tokens > 0.5).sum().item()),
        "final_visible_endpoint_anchor_token_fraction": float(visible_tokens.mean().item()) if visible_tokens.numel() else 0.0,
    }
    return state, stats, trajectory


def _query_mesh_vertices(mesh: MeshWithVoxel, query_chunk_size: int) -> torch.Tensor:
    return degradation._query_mesh_chunked(mesh, mesh.vertices, int(query_chunk_size)).detach().cpu().float()


def _lift_support_weights_to_vertices(
    vertices: torch.Tensor,
    local_coords: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Lift support diagnostics to decoded vertices for B2/B3 rendering.

    The actual guidance is always applied on the C1024 support.  This nearest
    cell lookup is only for the renderer's vertex-PBR diagnostic, so it never
    alters latent support or flow.
    """
    coords = local_coords.to(torch.long).cpu()
    values = weights.detach().cpu().float()
    lookup = {
        (int(row[0]), int(row[1]), int(row[2])): float(values[index].item())
        for index, row in enumerate(coords.tolist())
    }
    cell = torch.floor((vertices.detach().cpu().float() + 0.5) * OVOXEL_RESOLUTION).to(torch.long).clamp(0, OVOXEL_RESOLUTION - 1)
    output = torch.zeros(vertices.shape[0], dtype=torch.float32)
    for index, row in enumerate(cell.tolist()):
        output[index] = lookup.get((int(row[0]), int(row[1]), int(row[2])), 0.0)
    return output


def _render_decoded_variant(
    *,
    name: str,
    mesh: MeshWithVoxel,
    attrs: torch.Tensor,
    transform: Any,
    reference: Path,
    tile_dir: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    vertex_attrs = attrs.detach().cpu().float()
    sample = MeshWithVertexPbr(
        mesh.vertices.detach().cpu().float(),
        mesh.faces.detach().cpu().to(torch.int64),
        vertex_attrs,
        layout=dict(PBR_LAYOUT),
    )
    return core._metric_subset(
        core._render(
            sample,
            output_dir=tile_dir / "renders" / name,
            camera={
                "camera_angle_x": float(transform.camera_angle_x),
                "distance": float(transform.distance),
                "mesh_scale": float(transform.mesh_scale),
            },
            reference_image=reference,
            args=_make_render_args(args),
            envmap=envmap,
        )
    )


def _save_endpoint_payload(tile_dir: Path, name: str, endpoint: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> None:
    raw = _denormalize_slat(endpoint, normalization)
    _atomic_torch_save(
        tile_dir / f"{name}_endpoint.pt",
        {
            "coords": endpoint.coords.detach().cpu().to(torch.int32),
            "norm": endpoint.feats.detach().cpu().float(),
            "raw": raw.feats.detach().cpu().float(),
            "normalization": dict(normalization),
        },
    )


def _compact_cached_tile_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop tensor-to-list payloads from summaries written by early builds."""
    compact = dict(row)
    weights = compact.get("weights")
    if isinstance(weights, Mapping):
        weights = dict(weights)
        for key in (
            "representative_face_ids",
            "representative_uv_global",
            "representative_uv_tile",
            "representative_depth",
        ):
            weights.pop(key, None)
        compact["weights"] = weights
    return compact


def _sanity_decode_global_endpoint(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    g_tex_norm: SparseTensor,
    g_attrs: torch.Tensor,
    representative_local_points: torch.Tensor,
    query_chunk_size: int,
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    mesh, decoded_attrs, decode_stats = _decode_endpoint(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_norm=g_tex_norm,
        query_points=representative_local_points,
        query_chunk_size=query_chunk_size,
        label="G_tex representative sanity decode",
    )
    errors = _channel_relative_errors(decoded_attrs, g_attrs)
    errors["joint"] = _relative(decoded_attrs - g_attrs, g_attrs)
    report = {
        "query_definition": "representative local surface points selected from the maximum inverse-distance material contribution",
        "base_color_relative_error": errors["RGB"],
        "metallic_relative_error": errors["metallic"],
        "roughness_relative_error": errors["roughness"],
        "alpha_relative_error": errors["alpha"],
        "joint_relative_error": errors["joint"],
        "decoded_query_stats": decode_stats,
        "raw_encoder_input_stats": core._tensor_range(g_attrs),
        "decoded_query_stats_range": core._tensor_range(decoded_attrs),
    }
    return mesh, report


def _field_consistency_report(
    g_attrs: torch.Tensor,
    hr_attrs: torch.Tensor,
    result_attrs: torch.Tensor,
    weights: torch.Tensor,
    front_hr_gain_threshold: float = 0.05,
) -> Dict[str, Any]:
    front = weights > 0.7
    back = weights < 0.1
    hr_minus_g_front = _channel_mean_abs(hr_attrs, g_attrs, front)
    hr_minus_g_back = _channel_mean_abs(hr_attrs, g_attrs, back)
    front_rgb_gain = float(hr_minus_g_front["RGB"])
    back_rgb_gain = float(hr_minus_g_back["RGB"])
    weighted_hr = ((weights[:, None] * (result_attrs - hr_attrs).abs()).mean(dim=0)).detach().cpu()
    weighted_g = (((1.0 - weights)[:, None] * (result_attrs - g_attrs).abs()).mean(dim=0)).detach().cpu()
    return {
        "front_visible_w_gt_0.7_count": int(front.sum().item()),
        "back_or_unobserved_w_lt_0.1_count": int(back.sum().item()),
        "front_HR_minus_G_mean_abs": hr_minus_g_front,
        "back_HR_minus_G_mean_abs": hr_minus_g_back,
        "front_to_back_HR_gain_ratio_RGB": front_rgb_gain / max(back_rgb_gain, 1e-8),
        "front_HR_gain_threshold_RGB": float(front_hr_gain_threshold),
        "front_HR_gain_is_significant": bool(front_rgb_gain >= float(front_hr_gain_threshold)),
        "front_result_minus_pure_HR_mean_abs": _channel_mean_abs(result_attrs, hr_attrs, front),
        "back_result_minus_G_mean_abs": _channel_mean_abs(result_attrs, g_attrs, back),
        "weighted_E_w_abs_result_minus_HR": {
            name: _safe_mean(weighted_hr[channel_slice]) for name, channel_slice in PBR_SLICES.items()
        },
        "weighted_E_1_minus_w_abs_result_minus_G": {
            name: _safe_mean(weighted_g[channel_slice]) for name, channel_slice in PBR_SLICES.items()
        },
    }


def _prepare_global_baseline(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    image_1024: Image.Image,
    global_camera: Mapping[str, float],
    output_dir: Path,
) -> MeshWithVoxel:
    path = output_dir / "global_baseline_mesh.pt"
    if bool(args.resume) and path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mesh = payload["mesh"] if isinstance(payload, Mapping) else payload
        if not isinstance(mesh, MeshWithVoxel):
            raise RuntimeError("cached global baseline is not MeshWithVoxel")
        print(f"[global-baseline] reused {path}")
        return mesh
    started = time.perf_counter()
    ss_params, shape_params, tex_params = core._sampler_overrides(args)
    print("[global-baseline] ordinary Pixal3D 1024_cascade")
    output, latents = pipeline.run(
        image_1024,
        camera_params=global_camera,
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=tex_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    if len(output) != 1:
        raise RuntimeError(f"global baseline returned {len(output)} meshes")
    mesh = core._validate_mesh(output[0], "global 1024 baseline").to("cpu")
    shape_raw, tex_raw, resolution = latents
    if int(resolution) != OVOXEL_RESOLUTION:
        raise RuntimeError(f"global baseline resolution={resolution}, expected {OVOXEL_RESOLUTION}")
    if not torch.equal(shape_raw.coords, tex_raw.coords):
        raise RuntimeError("global baseline shape/texture sparse support differs")
    _atomic_torch_save(
        path,
        {
            "format": f"{FORMAT}_global_mesh",
            "mesh": mesh,
            "generation_seconds": float(time.perf_counter() - started),
            "shape_coords": shape_raw.coords.detach().cpu().to(torch.int32),
            "shape_raw": shape_raw.feats.detach().cpu().float(),
            "texture_raw": tex_raw.feats.detach().cpu().float(),
        },
    )
    del output, latents, shape_raw, tex_raw
    _empty_cuda_cache()
    return mesh


def _run_tile(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_attr_field: MeshWithVoxel,
    global_face_normals: torch.Tensor,
    visibility_buffers: Mapping[str, Any],
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    foreground_mask: Image.Image,
    output_dir: Path,
    tile_id: int,
    box: Sequence[int],
    face_min: torch.Tensor,
    face_max: torch.Tensor,
    face_finite: torch.Tensor,
) -> Dict[str, Any]:
    tile_dir = output_dir / "tiles" / f"tile_{int(tile_id):02d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    args._tile_id = int(tile_id)
    record: Dict[str, Any] = {
        "format": FORMAT,
        "tile_id": int(tile_id),
        "box": list(map(int, box)),
        "status": "started",
    }
    started = time.perf_counter()
    transform = core._derive_tile_camera(
        tile_id=int(tile_id),
        box=tuple(map(int, box)),
        global_camera=global_camera,
        extend_pixel=int(args.extend_pixel),
    )
    _atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
    hr_tile = image_4096.crop(tuple(map(int, box))).convert("RGB")
    if hr_tile.size != (TILE_SIZE, TILE_SIZE):
        hr_tile = hr_tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    hr_tile_path = tile_dir / "hr_tile.png"
    hr_tile.save(hr_tile_path)
    selected = core._tile_face_ids_from_bbox(face_min, face_max, face_finite, tuple(map(int, box)))
    record["projected_bbox_faces"] = int(selected.shape[0])
    if selected.numel() == 0:
        record.update({"status": "skipped", "reason": "no projected triangle bbox"})
        _atomic_json(tile_dir / "summary.json", record)
        return record
    try:
        geometry = core._prepare_tile_geometry(
            global_vertices=baseline_mesh.vertices,
            global_faces=baseline_mesh.faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        if float(geometry.stats["global_local_global_q_max_abs_error"]) > float(args.roundtrip_tolerance):
            raise RuntimeError("global/local camera round-trip exceeded tolerance")
        device = torch.device("cuda")
        shape_encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
        pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
        global_normals_live = global_face_normals.to(device=device)
        local_attrs, material_stats, mapping = core._resample_local_attrs_from_global(
            geometry=geometry,
            global_attr_field=global_attr_field,
            global_camera=global_camera,
            transform=transform,
            query_chunk_size=int(args.material_query_chunk_size),
            face_chunk_size=int(args.material_face_chunk_size),
            global_face_normals=global_normals_live,
            return_mapping=True,
        )
        fixed_shape_raw, shape_stats = core._encode_local_shape(
            encoder=shape_encoder,
            local_coords=geometry.coords,
            local_dual_vertices=geometry.dual_vertices,
            local_intersected=geometry.intersected,
            device=device,
            low_vram=bool(args.low_vram),
        )
        g_tex_raw, tex_stats = core._encode_local_pbr(
            encoder=pbr_encoder,
            coords=geometry.coords,
            attrs=local_attrs,
            device=device,
            low_vram=bool(args.low_vram),
        )
        alignment = core._latent_support_diagnostics(fixed_shape_raw, g_tex_raw)
        if not alignment["coordinates_exactly_equal"]:
            raise RuntimeError(f"fixed shape/G_tex support mismatch: {alignment}")
        fixed_shape_norm = _normalize_slat(fixed_shape_raw, pipeline.shape_slat_normalization)
        g_tex_norm = _normalize_slat(g_tex_raw, pipeline.tex_slat_normalization)
        if not torch.equal(fixed_shape_norm.coords, g_tex_norm.coords):
            raise RuntimeError("G_tex coords are not exactly fixed-shape coords")
        texture_model = pipeline.models["tex_slat_flow_model_1024"]
        expected_channels = int(texture_model.in_channels) - int(fixed_shape_norm.feats.shape[1])
        if int(g_tex_norm.feats.shape[1]) != expected_channels:
            raise RuntimeError(f"G_tex channel count={g_tex_norm.feats.shape[1]} expects {expected_channels}")

        _atomic_torch_save(
            tile_dir / "fixed_shape.pt",
            {
                "G_shape_raw": fixed_shape_raw.feats.detach().cpu().float(),
                "G_shape_norm": fixed_shape_norm.feats.detach().cpu().float(),
                "coords": fixed_shape_norm.coords.detach().cpu().to(torch.int32),
                "normalization": dict(pipeline.shape_slat_normalization),
                "shape_flow_called": False,
            },
        )
        _atomic_torch_save(
            tile_dir / "F_G.pt",
            {
                "F_G": local_attrs.detach().cpu().float(),
                "local_c1024_coords": geometry.coords.detach().cpu().to(torch.int32),
                "source": "global baseline MeshWithVoxel.query_attrs on the exact local C1024 material encoder support",
            },
        )

        guidance_values, guidance_diag = _prepare_local_material_guidance(
            mapping=mapping,
            buffers={key: value for key, value in visibility_buffers.items() if isinstance(value, torch.Tensor)},
            global_camera=global_camera,
            transform=transform,
            foreground_mask=foreground_mask,
            fixed_shape_coords=fixed_shape_norm.coords,
            global_face_count=int(baseline_mesh.faces.shape[0]),
            args=args,
            device=device,
        )
        # The representative arrays are persisted in guidance_weights.pt, not
        # embedded in JSON.  Keeping them in the diagnostic mapping would make
        # a tile summary hundreds of megabytes because _jsonable recursively
        # converts tensors to lists.
        representative_uv_global = guidance_diag.pop("representative_uv_global")
        representative_uv_tile = guidance_diag.pop("representative_uv_tile")
        representative_depth = guidance_diag.pop("representative_depth")
        representative_face_ids = guidance_diag.pop("representative_face_ids")
        representative_local = mapping["representative_local_surface_points"].to(device=device, dtype=torch.float32)
        guidance_values["representative_local_points"] = representative_local
        guidance_values["local_coords"] = geometry.coords.to(device=device, dtype=torch.int32)
        _save_tile_guidance_visuals(
            tile_dir,
            hr_tile=hr_tile,
            uv_tile=representative_uv_tile,
            values={key: guidance_values[key].detach().cpu() for key in ("w_visible", "w_facing", "w_tile", "w_final")}
            | {"w_fg": guidance_values["w_fg"].detach().cpu()},
        )
        _atomic_torch_save(
            tile_dir / "guidance_geometry.pt",
            {
                "local_c1024_coords": mapping["local_c1024_coords"],
                "local_c1024_centers": mapping["local_c1024_centers"],
                "representative_local_surface_points": mapping["representative_local_surface_points"],
                "representative_global_surface_points": mapping["representative_global_surface_points"],
                "representative_global_face_ids": mapping["representative_global_face_ids"],
                "representative_barycentric": mapping["representative_barycentric"],
                "representative_normals": mapping["representative_normals"],
                "aggregate_normals": mapping["aggregate_normals"],
                "representative_selection": mapping["representative_selection"],
                "visibility_aggregation": mapping["visibility_aggregation"],
                "c1024_mesh_visible": guidance_values["c1024_mesh_visible"].detach().cpu(),
                "c1024_nearest_mesh_visible": guidance_values["c1024_nearest_mesh_visible"].detach().cpu(),
                "c1024_nearby_mesh_visible": guidance_values["c1024_nearby_mesh_visible"].detach().cpu(),
                "c1024_nearby_visibility_ratio": guidance_values["c1024_nearby_visibility_ratio"].detach().cpu(),
                "slat_coords": guidance_values["slat_coords"].detach().cpu().to(torch.int32),
                "slat_visible": guidance_values["slat_visible"].detach().cpu(),
                "slat_visibility_ratio": guidance_values["slat_visibility_ratio"].detach().cpu(),
                "c1024_to_slat": guidance_values["c1024_to_slat"].detach().cpu().to(torch.int32),
            },
        )
        _atomic_torch_save(
            tile_dir / "guidance_weights.pt",
            {
                "w_visible": guidance_values["w_visible"].detach().cpu(),
                "w_facing": guidance_values["w_facing"].detach().cpu(),
                "w_tile": guidance_values["w_tile"].detach().cpu(),
                "w_fg": guidance_values["w_fg"].detach().cpu(),
                "w_final": guidance_values["w_final"].detach().cpu(),
                "representative_uv_global": representative_uv_global,
                "representative_uv_tile": representative_uv_tile,
                "representative_depth": representative_depth,
                "representative_face_ids": representative_face_ids,
                "c1024_mesh_visible": guidance_values["c1024_mesh_visible"].detach().cpu(),
                "c1024_nearest_mesh_visible": guidance_values["c1024_nearest_mesh_visible"].detach().cpu(),
                "c1024_nearby_mesh_visible": guidance_values["c1024_nearby_mesh_visible"].detach().cpu(),
                "slat_visible": guidance_values["slat_visible"].detach().cpu(),
                "slat_visibility_ratio": guidance_values["slat_visibility_ratio"].detach().cpu(),
                "stats": guidance_diag,
            },
        )
        for key in (
            "contribution_visible", "contribution_uv_global", "contribution_uv_tile",
            "c1024_mesh_visible", "c1024_nearby_mesh_visible",
            "c1024_nearest_mesh_visible",
            "c1024_nearby_visibility_ratio", "slat_coords",
            "slat_visibility_ratio", "c1024_to_slat",
        ):
            guidance_values.pop(key, None)

        shape_denorm = _denormalize_slat(fixed_shape_norm, pipeline.shape_slat_normalization)
        sanity_mesh, sanity_report = _sanity_decode_global_endpoint(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            g_tex_norm=g_tex_norm,
            g_attrs=local_attrs.to(device=device),
            representative_local_points=representative_local,
            query_chunk_size=int(args.query_chunk_size),
        )
        _atomic_json(tile_dir / "g_decode_sanity.json", sanity_report)
        # The sanity check is intentionally hard-gated only when explicitly
        # requested.  It is always reported before any flow guidance starts.
        if bool(args.fail_on_sanity_error) and float(sanity_report["joint_relative_error"]) > float(args.max_sanity_relative_error):
            raise RuntimeError(f"G_tex representative decode sanity error is too large: {sanity_report}")
        del sanity_mesh
        _empty_cuda_cache()

        _seed_everything(int(args.seed) + int(tile_id) * 100003)
        noise = SparseTensor(torch.randn_like(g_tex_norm.feats), g_tex_norm.coords)
        initial_state = degradation._native_noised_endpoint(
            g_tex_norm,
            noise,
            pipeline.tex_slat_sampler,
            float(args.noise_timestep),
            float(args.noise_strength),
        )
        texture_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [hr_tile],
            fixed_shape_norm.coords.to(torch.int32),
            camera_angle_x=float(transform.camera_angle_x),
            distance=float(transform.distance),
            mesh_scale=float(transform.mesh_scale),
            grid_resolution_override=LATENT_RESOLUTION,
        )
        tex_params = core._sampler_overrides(args)[2]
        pure_hr_norm, pure_flow_stats = _run_native_pure_hr_flow(
            pipeline=pipeline,
            initial_state=_fresh_sparse(initial_state),
            shape_condition=fixed_shape_norm,
            condition=texture_condition,
            params=tex_params,
            noise_timestep=float(args.noise_timestep),
        )
        _strict_endpoint_check(g_tex_norm, pure_hr_norm, "pure HR final endpoint")
        _save_endpoint_payload(tile_dir, "G", g_tex_norm, pipeline.tex_slat_normalization)
        _save_endpoint_payload(tile_dir, "pure_HR", pure_hr_norm, pipeline.tex_slat_normalization)
        guided_norm, guided_flow_stats, trajectory = _run_visibility_guided_flow(
            pipeline=pipeline,
            initial_state=_fresh_sparse(initial_state),
            pure_hr_norm=pure_hr_norm,
            fixed_shape_norm=fixed_shape_norm,
            shape_denorm=shape_denorm,
            condition=texture_condition,
            params=tex_params,
            g_attrs=local_attrs.to(device=device),
            guidance=guidance_values,
            pbr_encoder=pbr_encoder,
            geometry_reference=None,
            args=args,
        )
        _strict_endpoint_check(g_tex_norm, guided_norm, "guided final endpoint")
        _save_endpoint_payload(tile_dir, "guided", guided_norm, pipeline.tex_slat_normalization)
        _atomic_json(tile_dir / "trajectory_metrics.json", {"steps": trajectory, "flow": guided_flow_stats})

        # B2 endpoint-only PBR blend: pure-HR endpoint is decoded once, fused
        # at the material support, then encoded/decode-backed for the manifold
        # version.  The per-step route above remains the only changed sampler
        # route for B3.
        pure_mesh, pure_attrs_support, pure_decode_stats = _decode_endpoint(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=pure_hr_norm,
            query_points=representative_local,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile {tile_id:02d} pure HR endpoint",
        )
        endpoint_fused_attrs, endpoint_lambda = _blend_pbr_fields(
            local_attrs.to(device=device),
            pure_attrs_support,
            guidance_values["w_final"],
            timestep=0.0,
            schedule=str(args.fusion_time_schedule),
            channel_mode=str(args.fusion_mode),
            metallic_scale=float(args.metallic_local_scale),
            roughness_scale=float(args.roughness_local_scale),
        )
        endpoint_raw, endpoint_encode_stats = core._encode_local_pbr(
            encoder=pbr_encoder,
            coords=geometry.coords,
            attrs=endpoint_fused_attrs.detach().cpu(),
            device=device,
            low_vram=bool(args.low_vram),
        )
        endpoint_norm = _normalize_slat(endpoint_raw, pipeline.tex_slat_normalization)
        _strict_endpoint_check(g_tex_norm, endpoint_norm, "endpoint-only reencoded endpoint")
        _save_endpoint_payload(tile_dir, "endpoint_reencode", endpoint_norm, pipeline.tex_slat_normalization)
        endpoint_mesh, endpoint_cycle_attrs, endpoint_reencode_decode_stats = _decode_endpoint(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=endpoint_norm,
            query_points=representative_local,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile {tile_id:02d} endpoint reencode",
        )
        endpoint_cycle = _channel_relative_errors(endpoint_cycle_attrs, endpoint_fused_attrs)
        endpoint_cycle["joint"] = _relative(endpoint_cycle_attrs - endpoint_fused_attrs, endpoint_fused_attrs)

        guided_mesh, guided_attrs_support, guided_decode_stats = _decode_endpoint(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=guided_norm,
            query_points=representative_local,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile {tile_id:02d} guided final endpoint",
        )
        geometry_check_final = _geometry_check(pure_mesh, guided_mesh)
        if not geometry_check_final["geometry_equal"]:
            raise RuntimeError(f"pure HR/guided decoded geometry differs: {geometry_check_final}")

        # Query vertex fields for renderer diagnostics.  Direct endpoint fusion
        # uses support weights lifted by local C1024 cells; it does not modify
        # the support-level endpoint or flow route.
        g_mesh, g_vertex_attrs, g_decode_stats = _decode_endpoint(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=g_tex_norm,
            query_points=pure_mesh.vertices,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile {tile_id:02d} G render endpoint",
        )
        if not _geometry_check(g_mesh, pure_mesh)["geometry_equal"]:
            raise RuntimeError("G and pure-HR decoded geometry changed despite fixed shape")
        pure_vertex_attrs = _query_mesh_vertices(pure_mesh, int(args.query_chunk_size))
        guided_vertex_attrs = _query_mesh_vertices(guided_mesh, int(args.query_chunk_size))
        endpoint_vertex_attrs = _query_mesh_vertices(endpoint_mesh, int(args.query_chunk_size))
        g_vertex_attrs = g_vertex_attrs.detach().cpu().float()
        vertex_weights = _lift_support_weights_to_vertices(
            pure_mesh.vertices,
            geometry.coords,
            guidance_values["w_final"].detach().cpu(),
        )
        direct_endpoint_attrs = g_vertex_attrs + vertex_weights[:, None] * (pure_vertex_attrs - g_vertex_attrs)

        render_metrics: Dict[str, Any] = {"status": "skipped", "variants": {}}
        if bool(args.render):
            envmap = core.load_envmap(str(args.envmap), device="cuda")
            variants = {
                "G": (g_mesh, g_vertex_attrs),
                "pure_HR": (pure_mesh, pure_vertex_attrs),
                "endpoint_pbr_fusion": (pure_mesh, direct_endpoint_attrs),
                "endpoint_reencode": (endpoint_mesh, endpoint_vertex_attrs),
                "perstep_guided": (guided_mesh, guided_vertex_attrs),
            }
            for name, (variant_mesh, variant_attrs) in variants.items():
                print(f"[tile {tile_id:02d}] render {name}")
                render_metrics["variants"][name] = _render_decoded_variant(
                    name=name,
                    mesh=variant_mesh,
                    attrs=variant_attrs,
                    transform=transform,
                    reference=hr_tile_path,
                    tile_dir=tile_dir,
                    args=args,
                    envmap=envmap,
                )
            render_metrics["status"] = "success"
            del envmap

        field_consistency = _field_consistency_report(
            local_attrs.to(device=device),
            pure_attrs_support,
            guided_attrs_support,
            guidance_values["w_final"],
            front_hr_gain_threshold=float(args.front_hr_gain_threshold),
        )
        route_checks = {
            "shape_flow_called": False,
            "shape_slat_sampler_sample_called": False,
            "fixed_shape_coords_unchanged": bool(torch.equal(fixed_shape_norm.coords, guided_norm.coords)),
            "G_tex_coords_equal_fixed_shape": bool(torch.equal(g_tex_norm.coords, fixed_shape_norm.coords)),
            "pure_HR_coords_equal_fixed_shape": bool(torch.equal(pure_hr_norm.coords, fixed_shape_norm.coords)),
            "guided_coords_equal_fixed_shape": bool(torch.equal(guided_norm.coords, fixed_shape_norm.coords)),
            "endpoint_reencode_coords_equal_fixed_shape": bool(torch.equal(endpoint_norm.coords, fixed_shape_norm.coords)),
            "token_counts_equal": bool(
                g_tex_norm.coords.shape[0] == pure_hr_norm.coords.shape[0] == guided_norm.coords.shape[0] == endpoint_norm.coords.shape[0] == fixed_shape_norm.coords.shape[0]
            ),
            "token_order_equal": bool(
                torch.equal(g_tex_norm.coords, pure_hr_norm.coords)
                and torch.equal(g_tex_norm.coords, guided_norm.coords)
                and torch.equal(g_tex_norm.coords, endpoint_norm.coords)
            ),
            "normalization_identical": True,
            "decoded_geometry_before_after_texture_correction_identical": bool(geometry_check_final["geometry_equal"]),
            "official_sampler_xstart_to_pred_used": True,
            "velocity_average_or_v_G_used": False,
            "hr_model_forward_count_guided": int(guided_flow_stats["model_forward_count"]),
            "visible_slat_endpoint_anchor_strength": float(
                guided_flow_stats["final_visible_endpoint_anchor_strength"]
            ),
            "visible_slat_endpoint_anchor_support_unchanged": True,
        }
        record.update({
            "status": "success",
            "tile_seconds": float(time.perf_counter() - started),
            "hr_condition": {"source": "canonical 4096 crop", "image": str(hr_tile_path), "box_4096": list(map(int, box))},
            "geometry": geometry.stats,
            "material_resampling": material_stats,
            "fixed_shape": {
                "source": "global baseline mesh -> local C1024 dual-grid -> official shape encoder",
                "shape_flow_called": False,
                "shape_slat_sampler_sample_called": False,
                "encoder": shape_stats,
                "tokens": int(fixed_shape_norm.coords.shape[0]),
            },
            "G_tex": {"source": "global baseline MeshWithVoxel PBR query -> local official PBR encoder", "encoder": tex_stats, "tokens": int(g_tex_norm.coords.shape[0])},
            "G_decode_sanity": sanity_report,
            "weights": guidance_diag,
            "flow": {"pure_HR": pure_flow_stats, "guided": guided_flow_stats},
            "endpoint_cycle": {
                "endpoint_fused_vs_D_E": endpoint_cycle,
                "endpoint_encode": endpoint_encode_stats,
                "endpoint_decode": endpoint_reencode_decode_stats,
            },
            "trajectory_steps": int(len(trajectory)),
            "field_consistency": field_consistency,
            "decoded_geometry_check": geometry_check_final,
            "render_metrics": render_metrics,
            "route_checks": route_checks,
            "artifacts": {
                "G_endpoint": str(tile_dir / "G_endpoint.pt"),
                "pure_HR_endpoint": str(tile_dir / "pure_HR_endpoint.pt"),
                "guided_endpoint": str(tile_dir / "guided_endpoint.pt"),
                "endpoint_reencode": str(tile_dir / "endpoint_reencode_endpoint.pt"),
                "trajectory": str(tile_dir / "trajectory_metrics.json"),
            },
        })
        _atomic_json(tile_dir / "summary.json", record)
        del g_mesh, pure_mesh, endpoint_mesh, guided_mesh, g_vertex_attrs, pure_vertex_attrs, endpoint_vertex_attrs, guided_vertex_attrs
        del endpoint_norm, endpoint_raw, endpoint_fused_attrs, pure_hr_norm, guided_norm, fixed_shape_norm, shape_denorm
        _empty_cuda_cache()
        return record
    except Exception as exc:
        record.update({"status": "failed", "tile_seconds": float(time.perf_counter() - started), "reason": f"{type(exc).__name__}: {exc}"})
        _atomic_json(tile_dir / "summary.json", record)
        print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
        traceback.print_exc()
        return record
    finally:
        _empty_cuda_cache()


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    def collect(path: Sequence[str]) -> List[float]:
        result = []
        for row in success:
            value: Any = row
            for key in path:
                if not isinstance(value, Mapping) or key not in value:
                    value = None
                    break
                value = value[key]
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result.append(float(value))
        return result
    def stats(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {"count": int(array.size), "mean": float(array.mean()), "std": float(array.std()), "min": float(array.min()), "max": float(array.max())}
    variants = ["G", "pure_HR", "endpoint_pbr_fusion", "endpoint_reencode", "perstep_guided"]
    render = {
        variant: {
            metric: stats(collect(("render_metrics", "variants", variant, metric)))
            for metric in ("psnr_db", "ssim", "lpips")
        }
        for variant in variants
    }
    final_weights = [row.get("weights", {}).get("final_weight", {}) for row in success]
    return {
        "successful_tiles": int(len(success)),
        "failed_tiles": int(sum(row.get("status") == "failed" for row in rows)),
        "skipped_tiles": int(sum(row.get("status") == "skipped" for row in rows)),
        "render": render,
        "weight_summary": {
            name: stats([float(item.get(name)) for item in final_weights if isinstance(item, Mapping) and item.get(name) is not None])
            for name in ("mean", "median", "fraction_lt_0.1", "fraction_gt_0.9")
        },
        "front_hr_consistency_rgb": stats(collect(("field_consistency", "front_result_minus_pure_HR_mean_abs", "RGB"))),
        "back_global_consistency_rgb": stats(collect(("field_consistency", "back_result_minus_G_mean_abs", "RGB"))),
        "front_hr_gain_rgb": stats(collect(("field_consistency", "front_HR_minus_G_mean_abs", "RGB"))),
        "back_hr_gain_rgb": stats(collect(("field_consistency", "back_HR_minus_G_mean_abs", "RGB"))),
        "front_to_back_hr_gain_ratio_rgb": stats(collect(("field_consistency", "front_to_back_HR_gain_ratio_RGB"))),
        "front_hr_gain_significant_fraction": stats(
            [
                1.0
                if row.get("field_consistency", {}).get("front_HR_gain_is_significant") is True
                else 0.0
                for row in success
            ]
        ),
        "weighted_result_minus_hr_rgb": stats(collect(("field_consistency", "weighted_E_w_abs_result_minus_HR", "RGB"))),
        "weighted_result_minus_g_rgb": stats(collect(("field_consistency", "weighted_E_1_minus_w_abs_result_minus_G", "RGB"))),
        "sanity_joint_relative_error": stats(collect(("G_decode_sanity", "joint_relative_error"))),
        "endpoint_cycle_joint_relative_error": stats(collect(("endpoint_cycle", "endpoint_fused_vs_D_E", "joint"))),
    }


def _write_report(output_dir: Path, summary: Mapping[str, Any], aggregate: Mapping[str, Any]) -> None:
    rows = [row for row in summary.get("tiles", []) if row.get("status") == "success"]
    anchor_strength = float(summary.get("guidance", {}).get("final_visible_endpoint_anchor_strength") or 0.0)
    anchor_description = (
        "最终 endpoint 在 visible SLat token 上使用 pure-HR anchor。"
        if anchor_strength > 0.0
        else "最终 pure-HR visible endpoint anchor 已关闭（strength=0）；结果只保留 per-step 融合路径。"
    )
    lines = [
        "# Visibility-aware PBR endpoint guidance",
        "",
        "本实验是 training-free、fixed-shape 的 Pixal3D texture endpoint guidance。每一步只做一次 HR texture model forward，随后使用官方 FlowEuler `pred_x_0` / `_xstart_to_pred`，在 local C1024 PBR support 上完成 visibility-aware blend 和官方 PBR re-encode。",
        "",
        f"- image: `{summary.get('image')}`",
        f"- CUDA device: `{summary.get('cuda_device')}`",
        f"- successful tiles: `{aggregate.get('successful_tiles')}`",
        f"- fusion mode/schedule: `{summary.get('guidance', {}).get('fusion_mode')}` / `{summary.get('guidance', {}).get('fusion_time_schedule')}`",
        f"- guided per-step fixed HR/G weights: `{summary.get('guidance', {}).get('per_step_fixed_hr_weight')}` / `{summary.get('guidance', {}).get('per_step_fixed_g_weight')}`",
        f"- eta: `{summary.get('guidance', {}).get('reencode_guidance_strength')}`",
        f"- visible endpoint anchor: `{summary.get('guidance', {}).get('final_visible_endpoint_anchor_strength')}`",
        f"- front HR-G gain threshold (RGB): `{summary.get('guidance', {}).get('front_hr_gain_threshold')}`",
        "",
        "## 1. Visibility/facing sanity",
        "",
        "4096 geometry-only nvdiffrast buffer 在 flow 前一次性计算，并先按 corrected top-origin triangle ID 得到 projected mesh visibility；随后沿 nearby mesh contributions -> local C1024 O-Voxel -> fixed C64 SLat 继承。w_final 是 exact binary visible-SLat mask，w_facing/w_tile/w_fg 只作诊断，不参与最终乘法；" + anchor_description,
        "",
    ]
    for row in rows:
        weights = row.get("weights", {})
        lines.append(
            f"- tile {int(row['tile_id']):02d}: final mean=`{weights.get('final_weight', {}).get('mean')}`, "
            f"<0.1=`{weights.get('final_weight', {}).get('fraction_lt_0.1')}`, >0.9=`{weights.get('final_weight', {}).get('fraction_gt_0.9')}`, "
            f"visible_final=`{weights.get('visible_and_frontal_count')}`, "
            f"visible_c1024=`{weights.get('visible_c1024_count')}`, "
            f"visible_slat=`{weights.get('visible_slat_count')}`, "
            f"visible_global_faces=`{weights.get('global_visible_face_count_in_tile')}`"
        )
    lines.extend([
        "",
        "## 2. Baseline comparison",
        "",
        "B0=G，B1=pure HR，B2=final endpoint direct PBR blend，B2-reencode=endpoint blend 后 official encoder/decoder，B3=per-step guided flow。若某 variant 的 render metric 为 null，说明该运行使用了 `--no-render`。",
        "",
    ])
    for variant, metrics in aggregate.get("render", {}).items():
        lines.append(f"- `{variant}`: PSNR={metrics.get('psnr_db', {}).get('mean')}, SSIM={metrics.get('ssim', {}).get('mean')}, LPIPS={metrics.get('lpips', {}).get('mean')}")
    lines.extend([
        "",
        "## 3. Front/back PBR consistency",
        "",
        f"- front `w>0.7`, result-vs-pure-HR RGB mean absolute error: `{aggregate.get('front_hr_consistency_rgb')}`",
        f"- back/unobserved `w<0.1`, result-vs-G RGB mean absolute error: `{aggregate.get('back_global_consistency_rgb')}`",
        f"- weighted `E[w|result-HR|]` RGB: `{aggregate.get('weighted_result_minus_hr_rgb')}`",
        f"- weighted `E[(1-w)|result-G|]` RGB: `{aggregate.get('weighted_result_minus_g_rgb')}`",
        f"- front HR-G RGB gain: `{aggregate.get('front_hr_gain_rgb')}`",
        f"- back HR-G RGB gain: `{aggregate.get('back_hr_gain_rgb')}`",
        f"- front/back HR-G gain ratio: `{aggregate.get('front_to_back_hr_gain_ratio_rgb')}`",
        f"- tiles with significant front HR gain: `{aggregate.get('front_hr_gain_significant_fraction')}`",
        "",
        "## 4. Encoder manifold diagnostics",
        "",
        f"- G decode/query sanity joint relative error: `{aggregate.get('sanity_joint_relative_error')}`",
        f"- endpoint blend cycle joint relative error: `{aggregate.get('endpoint_cycle_joint_relative_error')}`",
        "",
        "## 5. Interpretation",
        "",
        "本脚本只验证 visibility-aware PBR endpoint guidance，不引入 RAHT、DWT、KNN latent fusion、velocity averaging 或 shape flow。是否“front 保留 HR、back 保持 G”应同时看 heatmap、front/back PBR consistency 和 render metrics；不会根据预设假设修改权重定义。",
        "",
        "## 6. Strict route checks",
        "",
    ])
    for row in rows:
        checks = row.get("route_checks", {})
        lines.append(f"- tile {int(row['tile_id']):02d}: shape_flow_called=`{checks.get('shape_flow_called')}`, support_equal=`{checks.get('token_order_equal') and checks.get('token_counts_equal')}`, decoded_geometry_equal=`{checks.get('decoded_geometry_before_after_texture_correction_identical')}`")
    (output_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/choose/0_img.png")
    parser.add_argument("--output-dir", default="outputs/visibility_guided_pbr_flow_cuda4_mesh_ovoxel_slat")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--pbr-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--visibility-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
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
    parser.add_argument("--visibility-eps-voxels", type=float, default=0.5)
    parser.add_argument("--visibility-sigma-voxels", type=float, default=1.5)
    parser.add_argument("--facing-good-deg", type=float, default=30.0)
    parser.add_argument("--facing-bad-deg", type=float, default=75.0)
    parser.add_argument("--tile-feather-pixels", type=float, default=128.0)
    parser.add_argument("--use-foreground-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--foreground-dilation-pixels", type=int, default=5)
    parser.add_argument("--fusion-mode", choices=("rgb_only", "rgb_mr_soft", "all"), default="rgb_only")
    parser.add_argument("--fusion-time-schedule", choices=("sin2", "constant", "linear", "late"), default="sin2")
    parser.add_argument("--metallic-local-scale", type=float, default=0.25)
    parser.add_argument("--roughness-local-scale", type=float, default=0.25)
    parser.add_argument("--reencode-guidance-strength", type=float, default=1.0)
    parser.add_argument(
        "--front-hr-gain-threshold",
        type=float,
        default=0.05,
        help="RGB mean absolute HR-G gain required to mark a visible/front region as significant",
    )
    parser.add_argument(
        "--final-visible-endpoint-anchor-strength",
        type=float,
        default=1.0,
        help="latent anchor strength applied only to visible C64 SLat tokens at the final endpoint",
    )
    parser.add_argument(
        "--per-step-fixed-hr-weight",
        type=float,
        default=None,
        help="override guided per-step PBR fusion with a constant HR weight; G receives 1-weight",
    )
    parser.add_argument("--fail-on-sanity-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sanity-relative-error", type=float, default=0.5)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if not 0.0 <= float(args.noise_timestep) <= 1.0:
        raise ValueError("noise timestep must be in [0,1]")
    if float(args.noise_strength) <= 0.0:
        raise ValueError("noise strength must be positive")
    if float(args.reencode_guidance_strength) < 0.0 or float(args.reencode_guidance_strength) > 1.0:
        raise ValueError("reencode guidance strength eta must be in [0,1]")
    if float(args.final_visible_endpoint_anchor_strength) < 0.0 or float(args.final_visible_endpoint_anchor_strength) > 1.0:
        raise ValueError("final visible endpoint anchor strength must be in [0,1]")
    if float(args.front_hr_gain_threshold) < 0.0:
        raise ValueError("front HR gain threshold must be non-negative")
    if args.per_step_fixed_hr_weight is not None and not 0.0 <= float(args.per_step_fixed_hr_weight) <= 1.0:
        raise ValueError("per-step fixed HR weight must be in [0,1]")
    if float(args.visibility_eps_voxels) < 0.0 or float(args.visibility_sigma_voxels) <= 0.0:
        raise ValueError("visibility eps must be non-negative and sigma positive")
    if float(args.tile_feather_pixels) < 0.0:
        raise ValueError("tile feather pixels must be non-negative")
    if float(args.facing_good_deg) < 0.0 or float(args.facing_bad_deg) > 90.0:
        raise ValueError("facing angles must lie in [0,90]")
    for path_value in (args.shape_encoder, args.pbr_encoder):
        path = Path(path_value).expanduser()
        if not Path(f"{path}.json").is_file() or not Path(f"{path}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for {path}")
    source = Path(args.image).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not bool(args.skip_lpips) and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips unavailable; continuing without LPIPS")
        args.skip_lpips = True


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(f"[cuda] requested/current index={args.cuda_device}/{torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.image).expanduser().resolve()
    with Image.open(source_path) as source:
        source_rgb = source.convert("RGB")
    source_rgb.save(output_dir / "input_original.png")
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    canonical = pipeline.preprocess_canonical_images(source_rgb)
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    foreground_mask: Image.Image = canonical["foreground_mask_4096"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    canonical["image_512"].save(output_dir / "canonical_512.png")
    foreground_mask.save(output_dir / "canonical_foreground_mask_4096.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical.get("metadata", {}))
    camera_path = output_dir / "global_camera.json"
    if bool(args.resume) and camera_path.is_file():
        global_camera = json.loads(camera_path.read_text(encoding="utf-8"))
    else:
        global_camera = core._estimate_camera(
            image_1024=image_1024,
            output_dir=output_dir,
            manual_fov=float(args.fov),
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            moge_model_path=None,
        )
        _atomic_json(camera_path, global_camera)
    baseline_mesh = _prepare_global_baseline(
        args=args,
        pipeline=pipeline,
        image_1024=image_1024,
        global_camera=global_camera,
        output_dir=output_dir,
    )
    baseline_mesh = baseline_mesh.to("cpu")
    _atomic_torch_save(output_dir / "global_baseline_mesh.pt", {"format": f"{FORMAT}_global_mesh", "mesh": baseline_mesh})
    visibility_dir = output_dir / "visibility_4096"
    visibility_buffers = _load_visibility_debug(visibility_dir) if bool(args.resume) else None
    if visibility_buffers is None:
        print("[visibility] one-time global 4096 geometry pass")
        visibility_buffers = _render_global_visibility_buffers(
            baseline_mesh,
            global_camera=global_camera,
            resolution=CANONICAL_IMAGE_SIZE,
            face_chunk_size=int(args.visibility_face_chunk_size),
            device=device,
        )
        _save_visibility_debug(visibility_dir, visibility_buffers)
    else:
        print(f"[visibility] reused {visibility_dir}")
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    global_attr_field = core._make_attribute_query_mesh(baseline_mesh, device)
    global_face_normals = _global_face_normals(baseline_mesh)
    boxes = _tile_boxes()
    requested = _parse_ids(args.tile_ids)
    rows: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if requested is not None and int(tile_id) not in requested:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        cached = output_dir / "tiles" / f"tile_{int(tile_id):02d}" / "summary.json"
        if bool(args.resume) and cached.is_file():
            try:
                cached_row = json.loads(cached.read_text(encoding="utf-8"))
            except Exception:
                cached_row = None
            cached_render = cached_row.get("render_metrics", {}) if isinstance(cached_row, Mapping) else {}
            render_needed = bool(args.render) and isinstance(cached_render, Mapping) and cached_render.get("status") != "success"
            cache_compatible = isinstance(cached_row, Mapping) and cached_row.get("format") == FORMAT
            if cache_compatible and cached_row.get("status") == "success" and not render_needed:
                print(f"[tile {tile_id:02d}] reused successful summary")
                compact_row = _compact_cached_tile_row(cached_row)
                cached_weights = cached_row.get("weights") if isinstance(cached_row, Mapping) else None
                needs_compaction = isinstance(cached_weights, Mapping) and any(
                    key in cached_weights
                    for key in (
                        "representative_face_ids",
                        "representative_uv_global",
                        "representative_uv_tile",
                        "representative_depth",
                    )
                )
                if needs_compaction:
                    _atomic_json(cached, compact_row)
                rows.append(compact_row)
                continue
        print(f"[tile {tile_id:02d}] box={box}")
        rows.append(_run_tile(
            args=args,
            pipeline=pipeline,
            baseline_mesh=baseline_mesh,
            global_attr_field=global_attr_field,
            global_face_normals=global_face_normals,
            visibility_buffers=visibility_buffers,
            global_camera=global_camera,
            image_4096=image_4096,
            foreground_mask=foreground_mask,
            output_dir=output_dir,
            tile_id=int(tile_id),
            box=box,
            face_min=face_min,
            face_max=face_max,
            face_finite=face_finite,
        ))
    aggregate = _aggregate_metrics(rows)
    summary = {
        "format": FORMAT,
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "seed": int(args.seed),
        "global_camera": global_camera,
        "tile_layout": {"canonical_image_size": CANONICAL_IMAGE_SIZE, "tile_size": TILE_SIZE, "stride": TILE_STRIDE, "boxes": boxes},
        "guidance": {
            "visibility_transfer": "projected global mesh triangle IDs -> nearby mesh contributions -> local C1024 O-Voxel -> exact C64 SLat parent",
            "visibility_weight_rule": "binary visible SLat uses HR; non-visible SLat uses G",
            "visibility_y_convention": "nvdiffrast raster y corrected to canonical top-origin image y",
            "fusion_mode": str(args.fusion_mode),
            "fusion_time_schedule": str(args.fusion_time_schedule),
            "reencode_guidance_strength": float(args.reencode_guidance_strength),
            "per_step_fixed_hr_weight": (
                float(args.per_step_fixed_hr_weight)
                if args.per_step_fixed_hr_weight is not None else None
            ),
            "per_step_fixed_g_weight": (
                1.0 - float(args.per_step_fixed_hr_weight)
                if args.per_step_fixed_hr_weight is not None else None
            ),
            "final_visible_endpoint_anchor_strength": float(args.final_visible_endpoint_anchor_strength),
            "front_hr_gain_threshold": float(args.front_hr_gain_threshold),
            "visibility_eps_voxels": float(args.visibility_eps_voxels),
            "visibility_sigma_voxels": float(args.visibility_sigma_voxels),
            "facing_good_deg": float(args.facing_good_deg),
            "facing_bad_deg": float(args.facing_bad_deg),
            "tile_feather_pixels": float(args.tile_feather_pixels),
            "use_foreground_weight": bool(args.use_foreground_weight),
        },
        "sampler": {
            "texture": core._sampler_overrides(args)[2],
            "noise_timestep": float(args.noise_timestep),
            "noise_strength": float(args.noise_strength),
            "native_noised_endpoint": "x_t=(1-t)G+sigma(t)epsilon",
            "t_equals_1_explanation": "at t=1, clean coefficient 1-t is zero; the state is full noise, not noisy G",
        },
        "route_checks": {
            "shape_flow_called": False,
            "shape_slat_sampler_sample_called": False,
            "official_meshwithvoxel_query": True,
            "official_texture_encoder": True,
            "official_texture_decoder": True,
            "official_flow_euler_xstart_to_pred": True,
            "nvdiffrast_one_time_global_visibility": True,
            "no_training": True,
            "no_velocity_average": True,
            "no_v_G": True,
        },
        "successful_tiles": int(sum(row.get("status") == "success" for row in rows)),
        "failed_tiles": int(sum(row.get("status") == "failed" for row in rows)),
        "skipped_tiles": int(sum(row.get("status") == "skipped" for row in rows)),
        "tiles": rows,
        "aggregate": aggregate,
        "artifacts": {
            "global_baseline_mesh": str(output_dir / "global_baseline_mesh.pt"),
            "visibility_dir": str(visibility_dir),
        },
    }
    _atomic_json(output_dir / "aggregate_metrics.json", aggregate)
    _write_report(output_dir, summary, aggregate)
    summary["aggregate_report"] = str(output_dir / "aggregate_report.md")
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] success={summary['successful_tiles']} failed={summary['failed_tiles']} skipped={summary['skipped_tiles']}")
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
