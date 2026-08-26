#!/usr/bin/env python3
"""Helpers for the single-view shared-SLat experiment.

The module deliberately contains only the pieces that are new for
``Codex.md``:

* a frozen, baseline-mesh triangle visibility table for every front tile;
* a 180-degree baseline-material render used as a texture observation; and
* row-wise front/back projection-feature routing.

It never creates a second latent state.  The returned condition keeps the
front global tokens and replaces only the sparse ``proj`` rows selected by
the visibility table.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import open3d as o3d
import torch
from PIL import Image

import pixal3d_texture_visibility_guided_pbr_flow as visibility
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_singleview_shared_slat_support_v1"
CANONICAL_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
TILE_COUNT = 49


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
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


def _atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256_tensors(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        value = value.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tile_boxes() -> list[Tuple[int, int, int, int]]:
    starts = list(range(0, CANONICAL_SIZE - TILE_SIZE + 1, TILE_STRIDE))
    if len(starts) != 7 or starts[-1] != CANONICAL_SIZE - TILE_SIZE:
        raise RuntimeError("single-view support requires a 7x7 4096/1024/512 layout")
    return [
        (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)
        for y0 in starts
        for x0 in starts
    ]


def _nearest_triangle_mapping(
    baseline: MeshWithVoxel,
    master_q_world: torch.Tensor,
    output_path: Path,
    *,
    face_chunk_size: int = 250_000,
) -> Dict[str, torch.Tensor]:
    """Bind each shared row to one actual baseline triangle.

    ``master_q_world`` is the repository's doubled object-space camera q;
    baseline vertices are in the centred object frame, hence the division by
    two before the closest-point query.  Open3D supplies the CPU BVH for the
    large baseline meshes, and the repository kernel recomputes exact
    closest-point/barycentric values on the selected triangle.
    """
    baseline_hash = _sha256_tensors(baseline.vertices, baseline.faces)
    support_hash = _sha256_tensors(master_q_world)
    if output_path.is_file():
        cached = torch.load(output_path, map_location="cpu", weights_only=False)
        if (
            cached.get("baseline_hash") == baseline_hash
            and cached.get("support_hash") == support_hash
            and all(key in cached for key in ("nearest_face_id", "nearest_point", "nearest_bary", "face_distance"))
        ):
            return {key: cached[key] for key in ("nearest_face_id", "nearest_point", "nearest_bary", "face_distance")}

    points = master_q_world.detach().cpu().float() / 2.0
    vertices = baseline.vertices.detach().cpu().float()
    faces = baseline.faces.detach().cpu().long()
    triangles = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3)
    if triangles.shape[0] <= int(face_chunk_size):
        face, point, bary, distance = core._nearest_faces_by_surface_distance(
            points, triangles, chunk_size=int(face_chunk_size)
        )
    else:
        vertex_tensor = o3d.core.Tensor(
            vertices.numpy(), dtype=o3d.core.Dtype.Float32
        )
        face_tensor = o3d.core.Tensor(
            faces.numpy().astype(np.int32), dtype=o3d.core.Dtype.Int32
        )
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex["positions"] = vertex_tensor
        mesh.triangle["indices"] = face_tensor
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        face_parts = []
        for start in range(0, int(points.shape[0]), 32768):
            block = points[start : start + 32768]
            result = scene.compute_closest_points(
                o3d.core.Tensor(block.numpy(), dtype=o3d.core.Dtype.Float32)
            )
            face_parts.append(result["primitive_ids"].numpy().astype(np.int64))
        face = torch.from_numpy(np.concatenate(face_parts, axis=0)).long()
        selected = triangles.index_select(0, face)
        point, bary, distance = core._closest_points_on_triangles(points, selected)
        del scene, mesh, vertex_tensor, face_tensor

    if bool((face < 0).any()) or not torch.isfinite(point).all() or not torch.isfinite(distance).all():
        raise RuntimeError("baseline nearest-triangle binding returned invalid rows")
    payload = {
        "format": FORMAT,
        "baseline_hash": baseline_hash,
        "support_hash": support_hash,
        "nearest_face_id": face.to(torch.int64).cpu(),
        "nearest_point": point.to(torch.float32).cpu(),
        "nearest_bary": bary.to(torch.float32).cpu(),
        "face_distance": distance.to(torch.float32).cpu(),
        "primitive": "baseline triangle face",
        "tie_break": "smallest face id in the vectorized fallback; BVH primitive order for exact large-mesh ties",
    }
    _atomic_save(output_path, payload)
    return {key: payload[key] for key in ("nearest_face_id", "nearest_point", "nearest_bary", "face_distance")}


@torch.no_grad()
def build_front_visibility(
    *,
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    views: Mapping[int, Any],
    master_q_world: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    face_chunk_size: int = 4_000_000,
) -> Dict[str, Any]:
    """Freeze true baseline face visibility independently for every tile."""
    support_dir = Path(output_dir) / "support"
    visibility_dir = support_dir / "front_visibility"
    visibility_dir.mkdir(parents=True, exist_ok=True)
    nearest = _nearest_triangle_mapping(
        baseline,
        master_q_world,
        support_dir / "master_nearest_triangle.pt",
    )

    triangle_path = visibility_dir / "triangle_id.pt"
    depth_path = visibility_dir / "depth.pt"
    if triangle_path.is_file() and depth_path.is_file():
        triangle_id = torch.load(triangle_path, map_location="cpu", weights_only=False)["triangle_id"].to(torch.int32)
        depth = torch.load(depth_path, map_location="cpu", weights_only=False)["depth"].to(torch.float32)
        renderer_name = "cached nvdiffrast baseline triangle-id raster"
    else:
        buffers = visibility._render_global_visibility_buffers(
            baseline,
            global_camera=camera,
            resolution=CANONICAL_SIZE,
            face_chunk_size=int(face_chunk_size),
            device=device,
        )
        triangle_id = buffers["triangle_id"].to(torch.int32).cpu()
        depth = buffers["depth"].to(torch.float32).cpu()
        _atomic_save(triangle_path, {"triangle_id": triangle_id})
        _atomic_save(depth_path, {"depth": depth})
        _atomic_save(visibility_dir / "foreground.pt", {"foreground": buffers["foreground"].to(torch.bool).cpu()})
        _atomic_save(visibility_dir / "facing.pt", {"facing": buffers["facing"].to(torch.float32).cpu()})
        visibility._save_visibility_debug(visibility_dir, buffers)
        renderer_name = str(buffers.get("renderer", "nvdiffrast baseline triangle-id raster"))
        del buffers
        _empty_cuda_cache()

    master_count = int(master_q_world.shape[0])
    nearest_face = nearest["nearest_face_id"].to(torch.int64).contiguous()
    visible_matrix = torch.zeros((TILE_COUNT, master_count), dtype=torch.bool)
    mapping_matrix = torch.zeros_like(visible_matrix)
    face_visible_ids = []
    tile_stats: Dict[str, Dict[str, Any]] = {}
    for tile_id, box in enumerate(tile_boxes()):
        view = views.get(tile_id)
        crop = triangle_id[box[1] : box[3], box[0] : box[2]]
        face_ids = torch.unique(crop[crop >= 0].to(torch.int64), sorted=True)
        face_visible_ids.append(face_ids)
        if view is None:
            tile_stats[str(tile_id)] = {
                "status": "inactive",
                "face_visible_count": int(face_ids.numel()),
                "mapping_count": 0,
                "front_visible_count": 0,
            }
            continue
        ids = view.master_ids.to(torch.int64).cpu()
        if bool((ids < 0).any()) or bool((ids >= master_count).any()):
            raise RuntimeError(f"tile {tile_id}: support row id is outside master table")
        flags = torch.isin(nearest_face.index_select(0, ids), face_ids)
        mapping_matrix[tile_id, ids] = True
        visible_matrix[tile_id, ids] = flags
        tile_stats[str(tile_id)] = {
            "status": "active",
            "face_visible_count": int(face_ids.numel()),
            "mapping_count": int(ids.numel()),
            "front_visible_count": int(flags.sum()),
            "front_invisible_count": int((~flags).sum()),
            "front_visible_fraction": float(flags.float().mean()) if flags.numel() else 0.0,
            "rule": "nearest baseline triangle face id is present in this tile's exact raster crop",
        }

    payload = {
        "format": FORMAT,
        "frozen": True,
        "renderer": renderer_name,
        "visible": visible_matrix,
        "mapping_valid": mapping_matrix,
        "nearest_face_id": nearest_face,
        "face_visible_ids": face_visible_ids,
        "depth_shape": list(depth.shape),
        "tile_stats": tile_stats,
        "rule": "front-visible rows use front proj; front-invisible rows use back proj",
    }
    _atomic_save(support_dir / "frozen_visibility.pt", payload)
    _atomic_save(support_dir / "face_visibility_per_context.pt", {
        "format": FORMAT,
        "context_count": TILE_COUNT,
        "face_visible_ids": face_visible_ids,
        "visible": visible_matrix,
        "mapping_valid": mapping_matrix,
        "nearest_face_id": nearest_face,
    })
    _atomic_json(
        support_dir / "visibility_stats.json",
        {
            "format": FORMAT,
            "frozen_before_flow": True,
            "view_level_bit_broadcast": False,
            "context_count": TILE_COUNT,
            "master_count": master_count,
            "front_visible_rows": int(visible_matrix.sum()),
            "front_invisible_rows": int((mapping_matrix & ~visible_matrix).sum()),
            "mapping_rows": int(mapping_matrix.sum()),
            "tile_stats": tile_stats,
            "renderer": renderer_name,
        },
    )
    return {
        "visible": visible_matrix,
        "mapping_valid": mapping_matrix,
        "nearest": nearest,
        "face_visible_ids": face_visible_ids,
        "tile_stats": tile_stats,
        "depth": depth,
        "triangle_id": triangle_id,
    }


def _save_tensor_image(value: torch.Tensor, path: Path, *, channels: int = 3) -> None:
    data = value.detach().float().cpu()
    if data.ndim == 3 and data.shape[0] in (1, 3, 4):
        data = data.permute(1, 2, 0)
    if data.ndim == 2:
        data = data[..., None]
    array = np.nan_to_num(data.numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        Image.fromarray((array[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray((array[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


@torch.no_grad()
def render_back_material_observation(
    mesh: MeshWithVertexPbr,
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    *,
    resolution: int = CANONICAL_SIZE,
) -> Dict[str, Any]:
    """Render the baseline material mesh from the 180-degree camera."""
    output_dir = Path(output_dir)
    rgb_path = output_dir / "back_rgb_4096.png"
    mask_path = output_dir / "back_mask_4096.png"
    if rgb_path.is_file() and mask_path.is_file():
        with Image.open(rgb_path) as image:
            if image.size == (resolution, resolution):
                return {
                    "rgb_path": str(rgb_path),
                    "mask_path": str(mask_path),
                    "resolution": [resolution, resolution],
                    "yaw_deg": 180,
                    "cache_hit": True,
                }

    from pixal3d.renderers import PbrMeshRenderer
    from render_pixal3d_raw_ovoxel import load_envmap
    from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views

    extrinsics, intrinsics, _ = _make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), (180,)
    )
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": int(resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "peel_layers": 8,
            "face_chunk_size": 4_000_000,
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)
    live = mesh.to(device)
    result = renderer.render(
        live,
        extrinsics[180].to(device),
        intrinsics.to(device),
        envmap=envmap,
        use_envmap_bg=False,
    )
    _save_tensor_image(result["shaded"], rgb_path, channels=3)
    _save_tensor_image(result["mask"], mask_path, channels=1)
    _save_tensor_image(result["base_color"], output_dir / "back_base_color_4096.png", channels=3)
    _save_tensor_image(result["normal"], output_dir / "back_normal_camera_4096.png", channels=3)
    del result, live, envmap, renderer
    _empty_cuda_cache()
    return {
        "rgb_path": str(rgb_path),
        "mask_path": str(mask_path),
        "resolution": [resolution, resolution],
        "yaw_deg": 180,
        "cache_hit": False,
    }


def _feature_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, SparseTensor):
        return value.feats
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"expected a tensor or SparseTensor condition, got {type(value)!r}")


def route_proj_rows(
    front_proj: torch.Tensor,
    back_proj: torch.Tensor,
    front_visible: torch.Tensor,
) -> torch.Tensor:
    """Select only projection rows; global features never enter this route."""
    if front_proj.shape != back_proj.shape:
        raise ValueError(f"front/back projection feature shapes differ: {front_proj.shape} vs {back_proj.shape}")
    visible = front_visible.to(device=front_proj.device, dtype=torch.bool).reshape(-1)
    if visible.numel() != front_proj.shape[0]:
        raise ValueError(
            f"visibility rows {visible.numel()} do not match projection rows {front_proj.shape[0]}"
        )
    return torch.where(visible[:, None], front_proj, back_proj)


def route_texture_conditions(
    *,
    views: Mapping[int, Any],
    front_conditions: Mapping[int, Mapping[str, Any]],
    visibility_by_tile: Mapping[int, torch.Tensor],
    output_dir: Path,
    back_conditions: Optional[Mapping[int, Mapping[str, Any]]] = None,
    mode: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Build front-only or front-global/back-proj sparse condition caches."""
    if mode not in {"front_only", "front_global_back_proj"}:
        raise ValueError(f"unknown texture routing mode {mode!r}")
    if mode == "front_global_back_proj" and back_conditions is None:
        raise ValueError("back conditions are required for front_global_back_proj")
    root = Path(output_dir) / "conditions" / "texture"
    root.mkdir(parents=True, exist_ok=True)
    result: Dict[int, Dict[str, Any]] = {}
    total_visible = 0
    total_invisible = 0
    for tile_id, view in sorted(views.items()):
        front = front_conditions[tile_id]
        front_proj = _feature_tensor(front["cond"]["proj"]).detach().cpu().contiguous()
        front_global = _feature_tensor(front["cond"]["global"]).detach().cpu().contiguous()
        visible = visibility_by_tile[tile_id].detach().cpu().to(torch.bool).reshape(-1)
        if visible.numel() != front_proj.shape[0]:
            raise RuntimeError(
                f"tile {tile_id}: frozen visibility rows {visible.numel()} != condition rows {front_proj.shape[0]}"
            )
        if mode == "front_only":
            hybrid = front_proj
        else:
            back = back_conditions[tile_id]
            back_proj = _feature_tensor(back["cond"]["proj"]).detach().cpu().contiguous()
            hybrid = route_proj_rows(front_proj, back_proj, visible)
        payload = {
            "format": FORMAT,
            "tile_id": int(tile_id),
            "coords": view.local_coords.detach().cpu().clone(),
            "cond": {
                "global": front_global,
                "proj": hybrid,
            },
            "neg_cond": {
                "global": torch.zeros_like(front_global),
                "proj": torch.zeros_like(hybrid),
            },
            "routing": {
                "mode": mode,
                "global_source": "front_tile_image",
                "proj_source": "front_tile_image for front-visible rows; back_tile_image for front-invisible rows"
                if mode == "front_global_back_proj"
                else "front_tile_image for every row",
                "front_visible": visible,
                "front_visible_count": int(visible.sum()),
                "front_invisible_count": int((~visible).sum()),
            },
        }
        _atomic_save(root / f"tile_{tile_id:02d}.pt", payload)
        result[tile_id] = payload
        total_visible += int(visible.sum())
        total_invisible += int((~visible).sum())
    row_count = total_visible + total_invisible
    stats = {
        "mode": mode,
        "global_source": "front_tile_image",
        "proj_source": "front for visible rows; back for invisible rows"
        if mode == "front_global_back_proj"
        else "front for all rows",
        "front_visible_rows": total_visible,
        "front_invisible_rows": total_invisible,
        "front_proj_fraction": float(total_visible / max(1, row_count)) if mode == "front_global_back_proj" else 1.0,
        "back_proj_fraction": float(total_invisible / max(1, row_count)) if mode == "front_global_back_proj" else 0.0,
        "row_count": row_count,
        "single_texture_state": True,
        "second_global_token": False,
        "second_texture_latent": False,
    }
    _atomic_json(root / "routing_stats.json", stats)
    return result, stats


__all__ = [
    "FORMAT",
    "tile_boxes",
    "build_front_visibility",
    "render_back_material_observation",
    "route_proj_rows",
    "route_texture_conditions",
]
