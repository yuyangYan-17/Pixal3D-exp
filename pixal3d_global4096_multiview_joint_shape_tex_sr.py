#!/usr/bin/env python3
"""Joint Shape + Texture SR on one first-view dense SLat master support.

This is the executable implementation of ``Codex2.md``.  The first-view
4096 image creates the only global support.  The 120 and 240 degree panels
only create context mappings into that support; they never add global rows.

The two stages use the same Jacobi barrier:

    independent local x_t
    -> one current-time pred_x_0 per real context batch
    -> visible/Gaussian/fallback endpoint fusion by master id
    -> gather endpoint back to every local context
    -> official _xstart_to_pred and Euler update

Texture additionally decodes each frozen endpoint to PBR, fuses only direct
visible donors, re-encodes the complete local C1024 fields in real batches,
and applies the official cycle correction before the master endpoint fusion.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The sparse CUDA extensions read these during import.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import open3d as o3d
import torch
from PIL import Image

import pixal3d.models as pixal3d_models
import pixal3d_cross_tile_pbr_perstep as cross_tile
import pixal3d_global4096_tile_endpoint_rollout_sync as first_view_route
import pixal3d_global4096_tile_x0_consensus_sync as validation_route
import pixal3d_multiview_fixed_geometry_pbr_gaussian_sr as multiview_route
import pixal3d_texture_visibility_guided_pbr_flow as visibility
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithFacePbr, MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_global4096_multiview_joint_shape_tex_sr_cuda4_v1"
CANONICAL_SIZE = 4096
VIEW_SIZE = 1024
FIRST_TILE_SIZE = 1024
FIRST_TILE_STRIDE = 512
VIEW_TILE_SIZE = 256
VIEW_TILE_STRIDE = 128
MODEL_TILE_SIZE = 1024
TILE_GRID = 7
TILE_COUNT = 49
LATENT_SIZE = 64
LOCAL_OVOXEL = 1024
ANGLES = (0, 120, 240)
SIGMA_PIXELS = 256.0
FLOW_BATCH_SIZE = 44
DECODE_BATCH_SIZE = 12
PBR_ENCODE_BATCH_SIZE = 13
# The texture NAF image-condition branch is memory-sensitive at 1024^2.  The
# reference route keeps only this extractor at B=1; it does not change the
# required PBR sparse encoder batch of 13 used by texture fusion.
TEXTURE_CONDITION_BATCH_SIZE = 1

DEFAULT_IMAGE = Path("/home/nvme04/yyyan/Pixal3D/assets/images/0_img.png")
DEFAULT_MULTIVIEW = Path(
    "/home/nvme04/yyyan/Pixal3D/test_pic/mask_compare_output/image2_resized.png"
)
DEFAULT_FIRST_SUPPORT = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_tile_x0_consensus_sync_cuda5/support"
)
DEFAULT_BASELINE = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_tile_x0_consensus_sync_cuda5/baseline/global_baseline_mesh.pt"
)
DEFAULT_CAMERA = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_tile_x0_consensus_sync_cuda5/global_camera.json"
)
DEFAULT_OUTPUT = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_multiview_joint_shape_tex_sr_cuda4"
)
DEFAULT_ENCODER_ROOT = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts"
)
DEFAULT_SHAPE_ENCODER = DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"
DEFAULT_PBR_ENCODER = DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"


@dataclass
class PreparedContext:
    context_id: int
    angle: int
    angle_index: int
    tile_id: int
    source_box: Tuple[int, int, int, int]
    virtual_box: Tuple[int, int, int, int]
    transform: Any
    tile_image: Image.Image
    tile_dir: Path
    geometry: core.LocalGeometry
    baseline_pbr: torch.Tensor
    shape_full: SparseTensor
    texture_full: SparseTensor
    native_coords: torch.Tensor
    view: first_view_route.TileView
    master_ids: torch.Tensor
    local_coords: torch.Tensor
    uv_virtual: torch.Tensor
    gaussian_weight: torch.Tensor
    shape_reference: torch.Tensor
    texture_reference: torch.Tensor
    target_points: torch.Tensor
    target_world_points: torch.Tensor
    nearest_local_points: torch.Tensor
    nearest_local_uv: torch.Tensor
    visible: torch.Tensor
    mapping_valid_global: torch.Tensor
    shape_state: Optional[SparseTensor]
    texture_state: Optional[SparseTensor]
    condition_shape: Optional[Mapping[str, Any]]
    condition_texture: Optional[Mapping[str, Any]]
    support_stats: Dict[str, Any]


@dataclass
class DecodedTexture:
    pbr_at_master: torch.Tensor
    pbr_valid: torch.Tensor
    self_field: torch.Tensor
    self_valid: torch.Tensor
    decoded_stats: Dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _hash_many(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(name: str, item: Any) -> None:
        digest.update(name.encode("utf-8"))
        if isinstance(item, torch.Tensor):
            digest.update(_tensor_hash(item).encode("ascii"))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda x: str(x)):
                visit(str(key), item[key])
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(str(index), child)
        else:
            digest.update(repr(item).encode("utf-8"))

    for key in sorted(values, key=str):
        visit(str(key), values[key])
    return digest.hexdigest()


def _stats(value: torch.Tensor) -> Dict[str, Any]:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0}
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "norm": float(flat.norm()),
    }


def _tile_boxes(canonical: bool) -> List[Tuple[int, int, int, int]]:
    if canonical:
        boxes = first_view_route._tile_layout(
            CANONICAL_SIZE, FIRST_TILE_SIZE, FIRST_TILE_STRIDE
        )
    else:
        boxes = core._tile_layout(VIEW_SIZE, VIEW_TILE_SIZE, VIEW_TILE_STRIDE)
    if len(boxes) != TILE_COUNT:
        raise RuntimeError(f"fixed 7x7 layout produced {len(boxes)} boxes")
    return [tuple(int(v) for v in box) for box in boxes]


def _virtual_box(source_box: Sequence[int], canonical: bool) -> Tuple[int, int, int, int]:
    if canonical:
        return tuple(int(v) for v in source_box)
    return tuple(int(v) * 4 for v in source_box)


def _yaw_matrix(angle: int) -> torch.Tensor:
    theta = math.radians(float(angle))
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32
    )


def _world_to_view_q(q_world: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return q_world @ rotation.to(device=q_world.device, dtype=q_world.dtype)


def _view_to_world_q(q_view: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return q_view @ rotation.to(device=q_view.device, dtype=q_view.dtype).T


def _load_camera(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("camera"), Mapping):
        payload = payload["camera"]
    camera = {
        "camera_angle_x": float(payload["camera_angle_x"]),
        "distance": float(payload["distance"]),
        "mesh_scale": float(payload.get("mesh_scale", 1.0)),
    }
    expected = {
        "camera_angle_x": 0.517371749106554,
        "distance": 1.889538288116455,
        "mesh_scale": 1.0,
    }
    for key, value in expected.items():
        if abs(camera[key] - value) > 1e-9:
            raise RuntimeError(f"fixed camera mismatch for {key}: {camera[key]} != {value}")
    return camera


def _load_mesh(path: Path) -> MeshWithVoxel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if isinstance(mesh, MeshWithVoxel):
        return mesh.cpu()
    if not isinstance(mesh, Mapping):
        raise RuntimeError(f"baseline is not a MeshWithVoxel payload: {path}")
    return MeshWithVoxel(
        torch.as_tensor(mesh["vertices"]).float(),
        torch.as_tensor(mesh["faces"]).int(),
        torch.as_tensor(mesh["origin"]).tolist(),
        float(mesh["voxel_size"]),
        torch.as_tensor(mesh["coords"]).int(),
        torch.as_tensor(mesh["attrs"]).float(),
        torch.Size(mesh["voxel_shape"]),
        dict(mesh["layout"]),
    )


def _load_views(path: Path) -> Dict[int, Image.Image]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (3072, 1024):
        raise ValueError(f"multi-view composite must be 3072x1024, got {image.size}")
    return {
        0: image.crop((0, 0, 1024, 1024)),
        120: image.crop((1024, 0, 2048, 1024)),
        240: image.crop((2048, 0, 3072, 1024)),
    }


def _save_canonical(canonical: Mapping[str, Any], output_dir: Path) -> None:
    root = output_dir / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("source_square_rgba", "source_square_rgba.png"),
        ("foreground_mask_4096", "canonical_foreground_mask_4096.png"),
        ("image_4096", "canonical_foreground_rgb_4096.png"),
        ("image_1024", "global_input_1024.png"),
    ):
        if key in canonical:
            canonical[key].save(root / name)


def _geometry_payload(geometry: core.LocalGeometry) -> Dict[str, Any]:
    return {
        "vertices": geometry.vertices.cpu(),
        "faces": geometry.faces.cpu(),
        "coords": geometry.coords.cpu(),
        "dual_vertices": geometry.dual_vertices.cpu(),
        "dual_vertices_world": geometry.dual_vertices_world.cpu(),
        "intersected": geometry.intersected.cpu(),
        "selected_global_face_ids": geometry.selected_global_face_ids.cpu(),
        "stats": dict(geometry.stats),
    }


def _geometry_from_payload(payload: Mapping[str, Any]) -> core.LocalGeometry:
    return core.LocalGeometry(
        vertices=payload["vertices"].to(torch.float32),
        faces=payload["faces"].to(torch.int64),
        coords=payload["coords"].to(torch.int32),
        dual_vertices=payload["dual_vertices"].to(torch.float32),
        dual_vertices_world=payload["dual_vertices_world"].to(torch.float32),
        intersected=payload["intersected"],
        selected_global_face_ids=payload["selected_global_face_ids"].to(torch.long),
        stats=dict(payload.get("stats", {})),
    )


def _prepare_geometry(
    *,
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    transform: Any,
    face_bounds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rotated_vertices: torch.Tensor,
    output_path: Path,
    reuse_dual_path: Optional[Path] = None,
) -> core.LocalGeometry:
    if output_path.is_file():
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if payload.get("format") == FORMAT and payload.get("status") == "active":
            return _geometry_from_payload(payload["geometry"])
    # The previous first-view formal run persisted only C1024 dual rows.  It
    # is safe to reuse those rows after checking the fixed geometry/camera
    # route; local vertices/faces are reconstructed here, so no old feature
    # value or support identity crosses the boundary.
    if reuse_dual_path is not None and reuse_dual_path.is_file():
        cached = torch.load(reuse_dual_path, map_location="cpu", weights_only=False)
        if cached.get("status") == "active" and all(
            key in cached for key in ("coords", "dual_vertices", "intersected")
        ):
            face_min, face_max, face_finite = face_bounds
            face_ids = core._tile_face_ids_from_bbox(
                face_min, face_max, face_finite, transform.box
            )
            selected_faces = baseline.faces.index_select(0, face_ids.to(torch.long)).to(torch.int64)
            local_global_vertices, local_faces, _ = core._compact_submesh(
                rotated_vertices.cpu(), selected_faces
            )
            q_global = local_global_vertices * (2.0 * float(camera["mesh_scale"]))
            q_local, local_uv = core._global_q_to_local_q(
                q_global, global_camera=camera, transform=transform
            )
            local_vertices = q_local / (2.0 * float(transform.mesh_scale))
            coords = cached["coords"].to(torch.int32).cpu()
            dual_vertices = cached["dual_vertices"].to(torch.float32).cpu()
            intersected = cached["intersected"].cpu()
            if coords.shape[0] == dual_vertices.shape[0] == intersected.shape[0]:
                dual_vertices_world = (dual_vertices + coords.to(torch.float32)) / float(LOCAL_OVOXEL)
                stats = dict(cached.get("stats", {}))
                stats.update({
                    "geometry_cache_reused": str(reuse_dual_path),
                    "geometry_cache_feature_values_used": False,
                    "global_local_global_q_max_abs_error": float((core._local_q_to_global_q(q_local, global_camera=camera, transform=transform)[0] - q_global).abs().max()),
                    "selected_global_face_ids": int(face_ids.numel()),
                    "local_mesh_faces": int(local_faces.shape[0]),
                    "local_mesh_vertices": int(local_vertices.shape[0]),
                    "selected_local_uv_range": core._tensor_range(local_uv),
                })
                geometry = core.LocalGeometry(
                    vertices=local_vertices.cpu().float(),
                    faces=local_faces.cpu().long(),
                    coords=coords,
                    dual_vertices=dual_vertices,
                    dual_vertices_world=dual_vertices_world,
                    intersected=intersected,
                    selected_global_face_ids=face_ids.cpu().long(),
                    stats=stats,
                )
                _atomic_save(
                    output_path,
                    {"format": FORMAT, "status": "active", "geometry": _geometry_payload(geometry)},
                )
                return geometry
    face_min, face_max, face_finite = face_bounds
    geometry = core._prepare_tile_geometry(
        global_vertices=rotated_vertices.cpu(),
        global_faces=baseline.faces.cpu(),
        global_face_min=face_min,
        global_face_max=face_max,
        global_face_finite=face_finite,
        global_camera=camera,
        transform=transform,
    )
    _atomic_save(
        output_path,
        {"format": FORMAT, "status": "active", "geometry": _geometry_payload(geometry)},
    )
    return geometry


def _face_bounds(
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    path: Path,
    *,
    source_size: int,
    chunk_size: int,
    vertices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload["face_min"], payload["face_max"], payload["face_finite"]
    values = core._project_face_bboxes(
        baseline.vertices.cpu() if vertices is None else vertices.cpu(),
        baseline.faces.cpu(),
        mesh_scale=float(camera["mesh_scale"]),
        global_camera=camera,
        chunk_size=int(chunk_size),
        source_width=int(source_size),
        source_height=int(source_size),
    )
    _atomic_save(path, {"face_min": values[0], "face_max": values[1], "face_finite": values[2]})
    return values


def _support_from_source(
    source_dir: Path,
    camera: Mapping[str, float],
    transforms: Mapping[int, Any],
) -> Optional[first_view_route.MasterSupport]:
    master_path = source_dir / "master_support.pt"
    views_dir = source_dir / "tile_views"
    if not master_path.is_file() or not views_dir.is_dir():
        return None
    payload = torch.load(master_path, map_location="cpu", weights_only=False)
    # The previous CUDA5 entry used a different first-view encoder/support
    # route.  Its master rows are not safe to use as current encoder positions
    # (the check below would otherwise fail only after expensive encodes), so
    # accept a source cache only when it was produced by this exact entry.
    if payload.get("format") != FORMAT:
        print(
            f"[support] ignoring incompatible source format {payload.get('format')!r}; "
            "rebuilding first-view support from current shape encoder positions",
            flush=True,
        )
        return None
    if payload.get("encoder_feature_values_present", False):
        raise RuntimeError("first-view support source contains encoder feature values")
    q = payload.get("master_q_world", payload.get("master_q_global"))
    uv = payload.get("front_uv_4096", payload.get("master_uv_4096"))
    owner = payload.get("owner_front_tile_id", payload.get("owner_tile_id"))
    owner_coord = payload.get("owner_front_local_coord", payload.get("owner_local_coord_c64"))
    if not all(isinstance(v, torch.Tensor) for v in (q, uv, owner, owner_coord)):
        return None
    q, uv = q.float().contiguous(), uv.float().contiguous()
    owner, owner_coord = owner.to(torch.int16).contiguous(), owner_coord.to(torch.int32).contiguous()
    if q.ndim != 2 or q.shape[1] != 3 or uv.shape != (q.shape[0], 2):
        return None
    if owner.shape[0] != q.shape[0] or owner_coord.shape != (q.shape[0], 3):
        return None
    tile_views: Dict[int, first_view_route.TileView] = {}
    tile_stats: Dict[int, Dict[str, Any]] = {}
    for tile_id in range(TILE_COUNT):
        path = views_dir / f"tile_{tile_id:02d}.pt"
        if not path.is_file():
            tile_stats[tile_id] = {"status": "inactive", "reason": "source_missing"}
            continue
        row = torch.load(path, map_location="cpu", weights_only=False)
        ids = row["master_ids"].to(torch.int64).contiguous()
        coords = row["local_coords_c64"].to(torch.int32).contiguous()
        tile_uv = row["master_uv_4096"].to(torch.float32).contiguous()
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] != ids.numel():
            raise RuntimeError(f"invalid first-view source tile {tile_id}")
        if bool((ids < 0).any()) or bool((ids >= q.shape[0]).any()):
            raise RuntimeError(f"first-view source tile {tile_id} contains invalid master ids")
        box = tuple(int(v) for v in _tile_boxes(True)[tile_id])
        weights = first_view_route.gaussian_weights(tile_uv, box, SIGMA_PIXELS)
        view = first_view_route.TileView(
            tile_id=tile_id,
            box=box,
            transform=transforms[tile_id],
            master_ids=ids,
            local_coords=coords,
            master_uv_4096=tile_uv,
            gaussian_weight=weights,
            stats={"status": "active", "source": str(path)},
        )
        tile_views[tile_id] = view
        tile_stats[tile_id] = dict(view.stats)
    if not tile_views:
        return None
    coverage = torch.zeros((q.shape[0],), dtype=torch.int32)
    for view in tile_views.values():
        coverage.index_add_(0, view.master_ids, torch.ones_like(view.master_ids, dtype=torch.int32))
    if bool((coverage <= 0).any()):
        raise RuntimeError("first-view source support does not cover every master id")
    return first_view_route.MasterSupport(
        master_q_global=q,
        master_uv_4096=uv,
        owner_tile_id=owner,
        owner_local_coord_c64=owner_coord,
        tile_views=tile_views,
        tile_stats=tile_stats,
        collision_report=[],
        roundtrip_max_abs_error=0.0,
    )


def _save_master_support(
    support: first_view_route.MasterSupport,
    output_dir: Path,
    transforms: Mapping[int, Any],
) -> str:
    support_dir = output_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    master_ids = torch.arange(support.master_q_global.shape[0], dtype=torch.int64)
    support_hash = _hash_many(
        {
            "master_id": master_ids,
            "master_q_world": support.master_q_global,
            "owner": support.owner_tile_id,
            "owner_coord": support.owner_local_coord_c64,
            "uv": support.master_uv_4096,
        }
    )
    _atomic_save(
        support_dir / "master_support.pt",
        {
            "format": FORMAT,
            "master_id": master_ids,
            "master_q_world": support.master_q_global,
            "master_q_global": support.master_q_global,
            "owner_front_tile_id": support.owner_tile_id,
            "owner_front_local_coord": support.owner_local_coord_c64,
            "front_uv_4096": support.master_uv_4096,
            "encoder_feature_values_present": False,
            "support_sha256": support_hash,
        },
    )
    for tile_id, view in sorted(support.tile_views.items()):
        _atomic_save(
            support_dir / "tile_views" / f"tile_{tile_id:02d}.pt",
            {
                "tile_id": tile_id,
                "box": list(view.box),
                "master_ids": view.master_ids,
                "local_coords_c64": view.local_coords,
                "master_uv_4096": view.master_uv_4096,
                "gaussian_weight": view.gaussian_weight,
                "tile_camera": first_view_route._jsonable(transforms[tile_id].__dict__),
            },
        )
    first_view_route._owner_map_images(
        support, support_dir, sorted(support.tile_views)
    )
    _atomic_json(
        support_dir / "master_support.json",
        {
            "format": FORMAT,
            "master_token_count": int(master_ids.numel()),
            "active_tile_ids": sorted(support.tile_views),
            "inactive_tile_ids": [i for i in range(TILE_COUNT) if i not in support.tile_views],
            "ownership": "first-view 2-D first-owner half-open rectangles",
            "encoder_feature_values_used": False,
            "support_sha256": support_hash,
        },
    )
    _atomic_json(support_dir / "support_collision_report.json", {"collisions": [], "policy": "error"})
    return support_hash


def _coord_keys(coords: torch.Tensor) -> torch.Tensor:
    xyz = coords[:, -3:].to(torch.int64)
    return (xyz[:, 0] * LATENT_SIZE + xyz[:, 1]) * LATENT_SIZE + xyz[:, 2]


def _gather_coords(value: SparseTensor, coords: torch.Tensor, label: str) -> torch.Tensor:
    source = value.coords.detach().cpu().to(torch.int32)
    target = coords.detach().cpu().to(torch.int32)
    source_keys = _coord_keys(source)
    target_keys = _coord_keys(target)
    order = torch.argsort(source_keys, stable=True)
    sorted_keys = source_keys.index_select(0, order)
    positions = torch.searchsorted(sorted_keys, target_keys)
    valid = positions < sorted_keys.numel()
    safe = positions.clamp_max(max(0, sorted_keys.numel() - 1))
    valid &= sorted_keys.index_select(0, safe) == target_keys
    if not bool(valid.all()):
        missing = torch.where(~valid)[0][:16].tolist()
        raise RuntimeError(f"{label}: missing local C64 coordinates at rows {missing}")
    return value.feats.detach().cpu().float().index_select(0, order.index_select(0, safe))


def _map_master_to_context(
    *,
    master_q_world: torch.Tensor,
    native_coords: torch.Tensor,
    angle: int,
    transform: Any,
    camera: Mapping[str, float],
    virtual_box: Tuple[int, int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    rotation = _yaw_matrix(angle)
    q_view = _world_to_view_q(master_q_world, rotation)
    q_local, uv_tile = core._global_q_to_local_q(
        q_view, global_camera=camera, transform=transform
    )
    uv_full, _, finite = core._project_global_q_to_image(
        q_view,
        global_camera=camera,
        image_width=VIEW_SIZE,
        image_height=VIEW_SIZE,
    )
    local_xyz, local_valid = first_view_route._c64_coords_from_q(q_local)
    normalized = q_local / (2.0 * float(transform.mesh_scale))
    local_valid &= (
        torch.isfinite(uv_tile).all(dim=1)
        & torch.isfinite(uv_full).all(dim=1)
        & (normalized >= -0.5 - 1e-5).all(dim=1)
        & (normalized <= 0.5 + 1e-5).all(dim=1)
    )
    native = native_coords.detach().cpu().to(torch.int32)
    if native.numel() == 0:
        empty_ids = torch.empty((0,), dtype=torch.int64)
        empty_coords = torch.empty((0, 4), dtype=torch.int32)
        empty_uv = torch.empty((0, 2), dtype=torch.float32)
        return empty_ids, empty_coords, empty_uv, torch.zeros((master_q_world.shape[0],), dtype=torch.bool), {
            "candidate_count": 0,
            "selected_count": 0,
            "collision_count": 0,
            "roundtrip_max_abs_error": 0.0,
            "native_support_count": 0,
            "virtual_box": list(virtual_box),
        }
    native_keys = _coord_keys(native)
    native_order = torch.argsort(native_keys, stable=True)
    sorted_native = native_keys.index_select(0, native_order)
    local_coords = torch.cat((torch.zeros((local_xyz.shape[0], 1), dtype=torch.int32), local_xyz), dim=1)
    local_keys = _coord_keys(local_coords)
    positions = torch.searchsorted(sorted_native, local_keys)
    in_native = positions < sorted_native.numel()
    safe = positions.clamp_max(max(0, sorted_native.numel() - 1))
    in_native &= sorted_native.index_select(0, safe) == local_keys
    candidate = finite & local_valid & in_native
    candidate_ids = torch.where(candidate)[0]
    if not candidate_ids.numel():
        empty_ids = torch.empty((0,), dtype=torch.int64)
        empty_coords = torch.empty((0, 4), dtype=torch.int32)
        empty_uv = torch.empty((0, 2), dtype=torch.float32)
        return empty_ids, empty_coords, empty_uv, torch.zeros((master_q_world.shape[0],), dtype=torch.bool), {
            "candidate_count": 0,
            "collision_count": 0,
            "roundtrip_max_abs_error": 0.0,
        }
    q_back, _ = core._local_q_to_global_q(
        q_local.index_select(0, candidate_ids),
        global_camera=camera,
        transform=transform,
    )
    errors = (q_back - q_view.index_select(0, candidate_ids)).abs().amax(dim=1)
    keys = local_keys.index_select(0, candidate_ids)
    ids = candidate_ids.to(torch.int64)
    # Stable lexicographic sort: master id, round-trip error, local key.  The
    # final key sort groups equal quantized local coordinates; within a group
    # the smallest error wins and an exact tie keeps the smallest master id.
    order = torch.arange(ids.numel(), dtype=torch.long)
    order = order[torch.argsort(ids.index_select(0, order), stable=True)]
    order = order[torch.argsort(errors.index_select(0, order), stable=True)]
    order = order[torch.argsort(keys.index_select(0, order), stable=True)]
    sorted_keys = keys.index_select(0, order)
    first = torch.ones((order.numel(),), dtype=torch.bool)
    if order.numel() > 1:
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep = order[first]
    selected_master = ids.index_select(0, keep)
    selected_local = local_xyz.index_select(0, candidate_ids.index_select(0, keep))
    selected_uv = uv_full.index_select(0, candidate_ids.index_select(0, keep)) * 4.0
    selected_coords = torch.cat(
        (torch.zeros((selected_local.shape[0], 1), dtype=torch.int32), selected_local), dim=1
    )
    valid_global = torch.zeros((master_q_world.shape[0],), dtype=torch.bool)
    valid_global[selected_master] = True
    collision_count = int(sorted_keys.numel() - torch.unique(sorted_keys).numel())
    return (
        selected_master.contiguous(),
        selected_coords.contiguous(),
        selected_uv.to(torch.float32).contiguous(),
        valid_global,
        {
            "candidate_count": int(candidate_ids.numel()),
            "selected_count": int(selected_master.numel()),
            "collision_count": collision_count,
            "roundtrip_max_abs_error": float(errors.max()) if errors.numel() else 0.0,
            "native_support_count": int(native.shape[0]),
            "virtual_box": list(virtual_box),
        },
    )


def _nearest_triangle_mapping(
    baseline: MeshWithVoxel,
    master_q_world: torch.Tensor,
    output_path: Path,
    *,
    face_chunk_size: int,
) -> Dict[str, torch.Tensor]:
    baseline_hash = _hash_many({"vertices": baseline.vertices, "faces": baseline.faces})
    support_hash = _tensor_hash(master_q_world)
    if output_path.is_file():
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if payload.get("baseline_hash") == baseline_hash and payload.get("support_hash") == support_hash:
            return {key: payload[key] for key in ("nearest_face_id", "nearest_point", "nearest_bary", "face_distance")}
    points = master_q_world.cpu().float() / (2.0 * float(1.0))
    triangles = baseline.vertices.cpu().float().index_select(
        0, baseline.faces.cpu().long().reshape(-1)
    ).reshape(-1, 3, 3)
    if triangles.shape[0] <= 200_000:
        face, point, bary, distance = core._nearest_faces_by_surface_distance(
            points, triangles, chunk_size=int(face_chunk_size)
        )
    else:
        # Open3D builds a CPU BVH over the immutable baseline triangles.  The
        # returned primitive is an actual triangle, not a vertex/O-Voxel
        # parent.  Exact closest-point/barycentric values are recomputed on
        # the selected triangles with the repository's vectorized kernel.
        vertex_tensor = o3d.core.Tensor(
            baseline.vertices.cpu().numpy(), dtype=o3d.core.Dtype.Float32
        )
        face_tensor = o3d.core.Tensor(
            baseline.faces.cpu().numpy(), dtype=o3d.core.Dtype.Int32
        )
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex["positions"] = vertex_tensor
        mesh.triangle["indices"] = face_tensor
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        face_parts: List[torch.Tensor] = []
        for start in range(0, points.shape[0], 32768):
            block = points[start : start + 32768]
            result = scene.compute_closest_points(
                o3d.core.Tensor(block.numpy(), dtype=o3d.core.Dtype.Float32)
            )
            face_parts.append(result["primitive_ids"].numpy().astype(np.int64))
        face = torch.from_numpy(np.concatenate(face_parts, axis=0)).long()
        selected_triangles = triangles.index_select(0, face)
        point, bary, distance = core._closest_points_on_triangles(points, selected_triangles)
        del scene, mesh, vertex_tensor, face_tensor
    if bool((face < 0).any()) or not torch.isfinite(point).all() or not torch.isfinite(distance).all():
        raise RuntimeError("nearest triangle mapping returned invalid rows")
    result = {
        "format": FORMAT,
        "baseline_hash": baseline_hash,
        "support_hash": support_hash,
        "nearest_face_id": face.to(torch.int64).cpu(),
        "nearest_point": point.to(torch.float32).cpu(),
        "nearest_bary": bary.to(torch.float32).cpu(),
        "face_distance": distance.to(torch.float32).cpu(),
        "distance_space": "canonical baseline object coordinates",
        "primitive": "baseline triangle face",
        "tie_break": "smallest face id for the exact vectorized path; BVH primitive order for equal-distance large-mesh ties",
    }
    _atomic_save(output_path, result)
    return {key: result[key] for key in ("nearest_face_id", "nearest_point", "nearest_bary", "face_distance")}


def _build_visibility(
    *,
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    contexts: Sequence[PreparedContext],
    master_q_world: torch.Tensor,
    nearest: Mapping[str, torch.Tensor],
    output_dir: Path,
    render_face_chunk_size: int,
) -> Dict[str, Any]:
    count = int(master_q_world.shape[0])
    context_count = len(ANGLES) * TILE_COUNT
    visible_matrix = torch.zeros((context_count, count), dtype=torch.bool)
    mapping_matrix = torch.zeros((context_count, count), dtype=torch.bool)
    depth_error = torch.full((context_count, count), float("nan"), dtype=torch.float32)
    tile_center_distance = torch.full((context_count, count), float("inf"), dtype=torch.float32)
    face_visible_ids: List[torch.Tensor] = [torch.empty((0,), dtype=torch.int64) for _ in range(context_count)]
    context_lookup = {context.context_id: context for context in contexts}
    nearest_face = nearest["nearest_face_id"].long()
    for angle_index, angle in enumerate(ANGLES):
        rotation = _yaw_matrix(angle)
        native_visibility_resolution = CANONICAL_SIZE if angle == 0 else VIEW_SIZE
        native_boxes = _tile_boxes(angle == 0)
        root = output_dir / "support" / "face_visibility" / f"view_{angle:03d}"
        tri_path = root / "triangle_id.pt"
        depth_path = root / "depth.pt"
        if tri_path.is_file() and depth_path.is_file():
            tri = torch.load(tri_path, map_location="cpu", weights_only=False)["triangle_id"].to(torch.int32)
            depth = torch.load(depth_path, map_location="cpu", weights_only=False)["depth"].to(torch.float32)
        else:
            rotated = baseline.vertices.cpu() @ rotation
            view_mesh = MeshWithVoxel(
                rotated,
                baseline.faces.cpu(),
                baseline.origin.tolist(),
                float(baseline.voxel_size),
                baseline.coords.cpu(),
                baseline.attrs.cpu(),
                baseline.voxel_shape,
                dict(baseline.layout),
            )
            buffers = visibility._render_global_visibility_buffers(
                view_mesh,
                global_camera=camera,
                resolution=native_visibility_resolution,
                face_chunk_size=int(render_face_chunk_size),
                device=torch.device("cuda"),
            )
            visibility._save_visibility_debug(root, buffers)
            tri = buffers["triangle_id"].cpu().to(torch.int32)
            depth = buffers["depth"].cpu().to(torch.float32)
            _atomic_save(tri_path, {"triangle_id": tri})
            _atomic_save(depth_path, {"depth": depth})
            del buffers, view_mesh, rotated
            _empty_cuda_cache()
        for tile_id, source_box in enumerate(native_boxes):
            context_id = angle_index * TILE_COUNT + tile_id
            x0, y0, x1, y1 = source_box
            crop_tri = tri[y0:y1, x0:x1]
            faces = torch.unique(crop_tri[crop_tri >= 0].to(torch.int64), sorted=True)
            face_visible_ids[context_id] = faces
            context = context_lookup.get(context_id)
            if context is None:
                continue
            ids = context.master_ids
            mapping_matrix[context_id, ids] = True
            face_flags = torch.isin(nearest_face.index_select(0, ids), faces)
            context.visible = face_flags.bool().contiguous()
            visible_matrix[context_id, ids] = context.visible
            tile_center = torch.tensor(
                [(context.virtual_box[0] + context.virtual_box[2]) * 0.5,
                 (context.virtual_box[1] + context.virtual_box[3]) * 0.5],
                dtype=torch.float32,
            )
            tile_center_distance[context_id, ids] = torch.linalg.vector_norm(
                context.uv_virtual - tile_center[None], dim=1
            )
            q_view = _world_to_view_q(master_q_world, rotation)
            uv, point_depth, finite = core._project_global_q_to_image(
                q_view,
                global_camera=camera,
                image_width=native_visibility_resolution,
                image_height=native_visibility_resolution,
            )
            pixels = torch.round(uv).long()
            inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < native_visibility_resolution) & (pixels[:, 1] >= 0) & (pixels[:, 1] < native_visibility_resolution)
            safe = pixels.clamp(0, native_visibility_resolution - 1)
            sampled_depth = depth[safe[:, 1], safe[:, 0]]
            values = point_depth - sampled_depth
            values[~inside] = float("nan")
            depth_error[context_id, ids] = values.index_select(0, ids)
            context.support_stats.update({
                "face_visible_count": int(faces.numel()),
                "visible_master_count": int(context.visible.sum()),
                "mapping_master_count": int(ids.numel()),
                "visibility_rule": "independent source-256 crop of exact 1024 face-id/depth raster",
            })
    _atomic_save(
        output_dir / "support" / "face_visibility_per_context.pt",
        {
            "format": FORMAT,
            "context_count": context_count,
            "face_count": int(baseline.faces.shape[0]),
            "face_visible_ids": face_visible_ids,
            "visible": visible_matrix,
            "mapping_valid": mapping_matrix,
            "nearest_face_id": nearest_face,
        },
    )
    _atomic_save(
        output_dir / "support" / "frozen_visibility.pt",
        {
            "format": FORMAT,
            "visible": visible_matrix,
            "mapping_valid": mapping_matrix,
            "nearest_face_id": nearest_face,
            "face_visible_ids": face_visible_ids,
            "depth_error": depth_error,
            "tile_center_distance": tile_center_distance,
            "frozen": True,
            "donor_only": True,
        },
    )
    _atomic_json(
        output_dir / "support" / "visibility_stats.json",
        {
            "format": FORMAT,
            "frozen_before_flow": True,
            "independent_per_context_face_tables": True,
            "view_level_bit_broadcast": False,
            "contexts": [
                {
                    "context_id": int(context.context_id),
                    "angle": int(context.angle),
                    "tile_id": int(context.tile_id),
                    "mapping_count": int(context.master_ids.numel()),
                    "visible_count": int(context.visible.sum()),
                    "face_visible_count": int(face_visible_ids[context.context_id].numel()),
                }
                for context in contexts
            ],
        },
    )
    return {
        "visible": visible_matrix,
        "mapping_valid": mapping_matrix,
        "face_visible_ids": face_visible_ids,
        "depth_error": depth_error,
        "tile_center_distance": tile_center_distance,
    }


def _make_local_view(
    context_id: int,
    transform: Any,
    virtual_box: Tuple[int, int, int, int],
    master_ids: torch.Tensor,
    local_coords: torch.Tensor,
    uv_virtual: torch.Tensor,
    stats: Mapping[str, Any],
) -> first_view_route.TileView:
    return first_view_route.TileView(
        tile_id=int(context_id),
        box=virtual_box,
        transform=transform,
        master_ids=master_ids.to(torch.int64).contiguous(),
        local_coords=local_coords.to(torch.int32).contiguous(),
        master_uv_4096=uv_virtual.to(torch.float32).contiguous(),
        gaussian_weight=first_view_route.gaussian_weights(
            uv_virtual.to(torch.float32), virtual_box, SIGMA_PIXELS
        ).contiguous(),
        stats=dict(stats),
    )


def _save_context_mapping(
    prepared: Sequence[Mapping[str, Any]],
    output_dir: Path,
    master_count: int,
) -> str:
    rows: List[Dict[str, Any]] = []
    mapping_hash_values: Dict[str, Any] = {"master_count": master_count}
    for item in prepared:
        row = {
            "context_id": int(item["context_id"]),
            "angle": int(item["angle"]),
            "tile_id": int(item["tile_id"]),
            "source_box": list(item["source_box"]),
            "virtual_box": list(item["virtual_box"]),
            "status": item.get("status", "active"),
            "master_ids": item["master_ids"],
            "local_coords_c64": item["local_coords"],
            "uv_virtual_4096": item["uv_virtual"],
            "gaussian_weight": item["gaussian_weight"],
            "stats": dict(item.get("mapping_stats", {})),
        }
        rows.append(row)
        mapping_hash_values[str(item["context_id"])] = {
            "master_ids": item["master_ids"],
            "local_coords": item["local_coords"],
            "uv": item["uv_virtual"],
        }
    mapping_hash = _hash_many(mapping_hash_values)
    _atomic_save(
        output_dir / "support" / "context_mapping.pt",
        {
            "format": FORMAT,
            "master_count": master_count,
            "mapping_sha256": mapping_hash,
            "contexts": rows,
        },
    )
    _atomic_json(
        output_dir / "support" / "context_mapping_stats.json",
        {
            "format": FORMAT,
            "mapping_sha256": mapping_hash,
            "master_count": master_count,
            "contexts": [
                {
                    "context_id": row["context_id"],
                    "angle": row["angle"],
                    "tile_id": row["tile_id"],
                    "status": row["status"],
                    **row["stats"],
                }
                for row in rows
            ],
        },
    )
    return mapping_hash


def _pack_conditions(
    contexts: Sequence[PreparedContext],
    pipeline: Any,
    output_dir: Path,
    stage: str,
    device: torch.device,
    batch_size: int,
) -> Dict[int, Mapping[str, Any]]:
    if not contexts:
        return {}
    views = {context.context_id: context.view for context in contexts}
    images = {context.context_id: context.tile_image for context in contexts}
    model = (
        pipeline.image_cond_model_shape_1024
        if stage == "shape"
        else pipeline.image_cond_model_tex_1024
    )
    return first_view_route._build_batched_image_conditions(
        pipeline,
        model,
        views,
        images,
        output_dir,
        stage,
        {"camera_angle_x": 0.517371749106554, "distance": 1.889538288116455, "mesh_scale": 1.0},
        device,
        int(batch_size),
    )


def _pack_local_states(
    group: Sequence[PreparedContext],
    states: Mapping[int, SparseTensor],
    device: torch.device,
    label: str,
) -> SparseTensor:
    values = [states[context.context_id] for context in group]
    return first_view_route._pack_sparse_batch(values, label).to(device)


def _pack_endpoint_batch(
    group: Sequence[PreparedContext],
    global_endpoint: torch.Tensor,
    device: torch.device,
    label: str,
) -> SparseTensor:
    values = [
        SparseTensor(
            global_endpoint.index_select(0, context.master_ids),
            context.local_coords,
        )
        for context in group
    ]
    return first_view_route._pack_sparse_batch(values, label).to(device)


def _unpack_state_batch(value: SparseTensor, group: Sequence[PreparedContext], label: str) -> List[SparseTensor]:
    parts = first_view_route._split_sparse_batch(value, len(group), label)
    result: List[SparseTensor] = []
    for context, part in zip(group, parts):
        if not torch.equal(part.coords.cpu(), context.local_coords):
            raise RuntimeError(f"{label}: context {context.context_id} support changed")
        result.append(part.cpu())
    return result


def _predict_stage_batches(
    *,
    contexts: Sequence[PreparedContext],
    states: Mapping[int, SparseTensor],
    conditions: Mapping[int, Mapping[str, Any]],
    model: torch.nn.Module,
    sampler: Any,
    params: Mapping[str, Any],
    t: float,
    t_next: float,
    device: torch.device,
    concat: Optional[Mapping[int, SparseTensor]],
    stage: str,
) -> Tuple[Dict[int, SparseTensor], Dict[str, Any]]:
    predictions: Dict[int, SparseTensor] = {}
    batch_sizes: List[int] = []
    model.to(device)
    model.eval()
    for start in range(0, len(contexts), FLOW_BATCH_SIZE):
        group = contexts[start : start + FLOW_BATCH_SIZE]
        batch_sizes.append(len(group))
        state_batch = _pack_local_states(group, states, device, f"{stage} frozen state")
        condition = first_view_route._pack_flow_condition(
            [context.view for context in group],
            conditions,
            state_batch.coords,
            device,
        )
        concat_batch = None
        if concat is not None:
            concat_batch = _pack_local_states(group, concat, device, f"{stage} shape concat")
        out = sampler.sample_once(
            model,
            state_batch,
            float(t),
            float(t_next),
            cond=condition["cond"],
            neg_cond=condition["neg_cond"],
            concat_cond=concat_batch,
            **first_view_route._prediction_kwargs(params),
        )
        if not hasattr(out, "pred_x_0") or not isinstance(out.pred_x_0, SparseTensor):
            raise RuntimeError(f"{stage}: current prediction did not return SparseTensor pred_x_0")
        parts = _unpack_state_batch(out.pred_x_0, group, f"{stage} pred_x_0")
        for context, part in zip(group, parts):
            if not torch.isfinite(part.feats).all():
                raise FloatingPointError(f"{stage}: non-finite pred_x_0 for context {context.context_id}")
            predictions[context.context_id] = part
        del out, state_batch, condition, concat_batch, parts
        _empty_cuda_cache()
    model.cpu()
    _empty_cuda_cache()
    return predictions, {
        "physical_batches": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "logical_predictions": len(contexts),
        "current_prediction_only": True,
        "suffix_rollout_used": False,
    }


def _update_local_states(
    *,
    contexts: Sequence[PreparedContext],
    frozen: Mapping[int, SparseTensor],
    global_endpoint: torch.Tensor,
    sampler: Any,
    t: float,
    t_next: float,
    device: torch.device,
    label: str,
) -> Dict[int, SparseTensor]:
    result: Dict[int, SparseTensor] = {}
    for start in range(0, len(contexts), FLOW_BATCH_SIZE):
        group = contexts[start : start + FLOW_BATCH_SIZE]
        frozen_batch = _pack_local_states(group, frozen, device, f"{label} frozen batch")
        endpoint_batch = _pack_endpoint_batch(group, global_endpoint, device, f"{label} endpoint batch")
        velocity = sampler._xstart_to_pred(frozen_batch, float(t), endpoint_batch)
        if not isinstance(velocity, SparseTensor):
            raise RuntimeError(f"{label}: official _xstart_to_pred returned {type(velocity)!r}")
        parts = _unpack_state_batch(velocity, group, f"{label} velocity")
        dt = float(t - t_next)
        for context, part in zip(group, parts):
            next_value = frozen[context.context_id].replace(
                frozen[context.context_id].feats - dt * part.feats
            )
            if not torch.isfinite(next_value.feats).all():
                raise FloatingPointError(f"{label}: non-finite updated state for {context.context_id}")
            result[context.context_id] = next_value
        del frozen_batch, endpoint_batch, velocity, parts
        _empty_cuda_cache()
    return result


def _fuse_endpoint(
    *,
    contexts: Sequence[PreparedContext],
    predictions: Mapping[int, SparseTensor],
    fallback: torch.Tensor,
    master_count: int,
    channel_count: int,
    stage: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    sum_value = torch.zeros((master_count, channel_count), dtype=torch.float32)
    sum_weight = torch.zeros((master_count,), dtype=torch.float32)
    visible_count = torch.zeros((master_count,), dtype=torch.int32)
    for context in contexts:
        pred = predictions[context.context_id].feats.detach().cpu().float()
        if pred.shape[0] != context.master_ids.numel():
            raise RuntimeError(f"{stage}: prediction rows do not match context mapping")
        valid = context.visible & torch.isfinite(pred).all(dim=1)
        if bool(valid.any()):
            ids = context.master_ids[valid]
            weights = context.gaussian_weight[valid].float()
            values = pred[valid]
            sum_value.index_add_(0, ids, values * weights[:, None])
            sum_weight.index_add_(0, ids, weights)
            visible_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    if fallback.shape != sum_value.shape or not torch.isfinite(fallback).all():
        raise RuntimeError(f"{stage}: fallback endpoint has invalid shape or values")
    fallback_mask = sum_weight <= 0.0
    merged = fallback.clone()
    active = ~fallback_mask
    merged[active] = sum_value[active] / sum_weight[active, None]
    if not torch.isfinite(merged).all():
        raise FloatingPointError(f"{stage}: fused endpoint is non-finite")
    stats = {
        "stage": stage,
        "master_count": master_count,
        "visible_one": int((visible_count == 1).sum()),
        "visible_multiple": int((visible_count > 1).sum()),
        "visible_zero_fallback": int(fallback_mask.sum()),
        "visible_proposal_total": int(visible_count.sum()),
        "rule": "one direct proposal; Gaussian tile-center mean for multiple; baseline endpoint for zero",
    }
    return merged, visible_count, fallback_mask, stats


def _save_local_states(path: Path, states: Mapping[int, SparseTensor]) -> None:
    _atomic_save(
        path,
        {
            "context_ids": sorted(int(k) for k in states),
            "states": {
                str(int(k)): {
                    "coords": value.coords.cpu(),
                    "features": value.feats.cpu().float(),
                }
                for k, value in states.items()
            },
        },
    )


def _clone_sparse(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().clone(), value.coords.detach().clone())


def _load_local_states(path: Path) -> Dict[int, SparseTensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        int(key): SparseTensor(value["features"].float(), value["coords"].int())
        for key, value in payload["states"].items()
    }


def _schedule(sampler: Any, params: Mapping[str, Any]) -> Tuple[float, ...]:
    values = tuple(float(v) for v in sampler.timestep_schedule(int(params["steps"]), float(params.get("rescale_t", 1.0))))
    if len(values) != int(params["steps"]) + 1:
        raise RuntimeError("sampler schedule length mismatch")
    return values


def _run_shape_flow(
    *,
    contexts: Sequence[PreparedContext],
    pipeline: Any,
    output_dir: Path,
    device: torch.device,
    seed: int,
    steps_override: Optional[int],
    resume: bool,
    save_step_tensors: bool,
    shape_fallback: torch.Tensor,
    master_count: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model = pipeline.models["shape_slat_flow_model_1024"]
    sampler = pipeline.shape_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params)
    if steps_override is not None:
        params["steps"] = int(steps_override)
    schedule = _schedule(sampler, params)
    states: Dict[int, SparseTensor] = {}
    for context in contexts:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + int(context.angle) * 100003 + int(context.tile_id) * 1009 + 17
        )
        features = torch.randn(
            (context.master_ids.numel(), context.shape_full.feats.shape[1]),
            generator=generator,
            dtype=torch.float32,
        )
        states[context.context_id] = SparseTensor(features, context.local_coords.clone())
    stage_dir = output_dir / "shape"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _save_local_states(stage_dir / "initial_noise.pt", states)
    state_hash = _hash_many({str(k): v.feats for k, v in states.items()})
    checkpoint = stage_dir / "checkpoint.pt"
    start_step = 0
    records: List[Dict[str, Any]] = []
    final_endpoint: Optional[torch.Tensor] = None
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("support_hash") != state_hash or int(saved.get("steps", -1)) != len(schedule) - 1:
            raise RuntimeError("shape resume checkpoint rejected because support or schedule changed")
        states = _load_local_states(stage_dir / "state_next.pt")
        start_step = int(saved["next_step"])
        records = list(saved.get("records", []))
        if saved.get("global_endpoint") is not None:
            final_endpoint = saved["global_endpoint"].float()
    for step in range(start_step, len(schedule) - 1):
        t, t_next = schedule[step], schedule[step + 1]
        frozen = {cid: _clone_sparse(value) for cid, value in states.items()}
        predictions, prediction_stats = _predict_stage_batches(
            contexts=contexts,
            states=frozen,
            conditions={context.context_id: context.condition_shape for context in contexts},
            model=model,
            sampler=sampler,
            params=params,
            t=t,
            t_next=t_next,
            device=device,
            concat=None,
            stage="shape",
        )
        endpoint, visible_count, fallback_mask, fusion_stats = _fuse_endpoint(
            contexts=contexts,
            predictions=predictions,
            fallback=shape_fallback,
            master_count=master_count,
            channel_count=shape_fallback.shape[1],
            stage="shape",
        )
        # This is the single global barrier: all prediction batches have
        # completed before any local state is replaced.
        states = _update_local_states(
            contexts=contexts,
            frozen=frozen,
            global_endpoint=endpoint,
            sampler=sampler,
            t=t,
            t_next=t_next,
            device=device,
            label="shape",
        )
        final_endpoint = endpoint
        step_dir = stage_dir / f"step_{step:02d}"
        if save_step_tensors:
            _atomic_save(step_dir / "global_endpoint.pt", {"master_id": torch.arange(master_count), "features": endpoint})
            _atomic_save(step_dir / "visible_count.pt", visible_count)
            _atomic_save(step_dir / "fallback_mask.pt", fallback_mask)
        record = {
            "step": step,
            "t": t,
            "t_next": t_next,
            "contexts": len(contexts),
            "prediction": prediction_stats,
            "fusion": fusion_stats,
            "state_update": "gather global endpoint -> official _xstart_to_pred per physical batch -> local Euler",
            "jacobi_barrier": True,
            "local_state_independent_noise": True,
            "seconds": 0.0,
        }
        _atomic_json(step_dir / "stats.json", record)
        records.append(record)
        _save_local_states(stage_dir / "state_next.pt", states)
        _atomic_save(
            checkpoint,
            {
                "format": FORMAT,
                "stage": "shape",
                "support_hash": state_hash,
                "steps": len(schedule) - 1,
                "next_step": step + 1,
                "global_endpoint": endpoint,
                "records": records,
            },
        )
    if final_endpoint is None:
        raise RuntimeError("shape flow did not produce a final endpoint")
    _atomic_save(output_dir / "final" / "shape_global_final.pt", {"master_id": torch.arange(master_count), "features": final_endpoint})
    summary = {
        "format": FORMAT,
        "stage": "shape",
        "steps": len(schedule) - 1,
        "schedule": list(schedule),
        "records": records,
        "flow_batch_size": FLOW_BATCH_SIZE,
        "real_multi_batch": True,
        "suffix_rollout_used": False,
        "velocity_averaging": False,
        "final_endpoint": str(output_dir / "final" / "shape_global_final.pt"),
    }
    _atomic_json(stage_dir / "flow_summary.json", summary)
    return final_endpoint, summary


def _masked_query_pbr(
    mesh: MeshWithVoxel,
    points: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Query a decoded field with an explicit trilinear denominator gate."""
    if mesh.attrs.shape[1] != 6:
        raise RuntimeError(f"decoded PBR mesh must have six channels, got {mesh.attrs.shape}")
    mask = torch.ones((mesh.coords.shape[0],), dtype=torch.bool, device=mesh.attrs.device)
    attrs = torch.cat((mesh.attrs, mask[:, None].to(mesh.attrs.dtype)), dim=1)
    masked = MeshWithVoxel(
        mesh.vertices,
        mesh.faces,
        mesh.origin.tolist(),
        float(mesh.voxel_size),
        mesh.coords,
        attrs,
        torch.Size([1, 7, *mesh.voxel_shape[-3:]]),
        dict(mesh.layout),
    )
    queried = cross_tile._query_mesh_chunked(masked, points, int(chunk_size))
    queried = queried.detach().cpu().float()
    denominator = queried[:, 6]
    valid = torch.isfinite(queried).all(dim=1) & (denominator > 0.0)
    values = torch.zeros((points.shape[0], 6), dtype=torch.float32)
    if bool(valid.any()):
        values[valid] = queried[valid, :6] / denominator[valid, None]
    return values, valid


def _decode_texture_batches(
    *,
    contexts: Sequence[PreparedContext],
    predictions: Mapping[int, SparseTensor],
    shape_concat: Mapping[int, SparseTensor],
    pipeline: Any,
    device: torch.device,
    query_chunk_size: int,
) -> Dict[int, DecodedTexture]:
    """Decode all current texture endpoints in real B=12 batches."""
    snapshots: Dict[int, DecodedTexture] = {}
    for start in range(0, len(contexts), DECODE_BATCH_SIZE):
        group = contexts[start : start + DECODE_BATCH_SIZE]
        shape_values = [
            cross_tile._denormalize_slat(shape_concat[c.context_id], pipeline.shape_slat_normalization)
            for c in group
        ]
        texture_values = [
            cross_tile._denormalize_slat(predictions[c.context_id], pipeline.tex_slat_normalization)
            for c in group
        ]
        shape_batch = first_view_route._pack_sparse_batch(shape_values, "texture decode shape").to(device)
        texture_batch = first_view_route._pack_sparse_batch(texture_values, "texture decode endpoint").to(device)
        decoded = pipeline.decode_latent(shape_batch, texture_batch, LOCAL_OVOXEL)
        if len(decoded) != len(group):
            raise RuntimeError(f"texture decoder returned {len(decoded)} meshes for B={len(group)}")
        for context, mesh in zip(group, decoded):
            mesh = cross_tile._validate_decoded_mesh(
                mesh, f"texture context {context.context_id} endpoint"
            )
            mapped_values, mapped_valid = _masked_query_pbr(
                mesh,
                context.nearest_local_points.to(device),
                query_chunk_size,
            )
            self_values, self_valid = _masked_query_pbr(
                mesh,
                context.target_points.to(device),
                query_chunk_size,
            )
            self_field = context.baseline_pbr.clone()
            if bool(self_valid.any()):
                self_field[self_valid] = self_values[self_valid]
            if not torch.isfinite(self_field).all():
                raise FloatingPointError(f"texture context {context.context_id}: self PBR is non-finite")
            snapshots[context.context_id] = DecodedTexture(
                pbr_at_master=mapped_values,
                pbr_valid=mapped_valid & torch.isfinite(mapped_values).all(dim=1),
                self_field=self_field,
                self_valid=self_valid,
                decoded_stats={
                    "decoded_vertices": int(mesh.vertices.shape[0]),
                    "decoded_faces": int(mesh.faces.shape[0]),
                    "decoded_active_ovoxels": int(mesh.coords.shape[0]),
                    "mapped_query_valid": int(mapped_valid.sum()),
                    "self_query_valid": int(self_valid.sum()),
                    "query_rows": int(context.target_points.shape[0]),
                },
            )
            del mesh
        del shape_batch, texture_batch, decoded, shape_values, texture_values
        _empty_cuda_cache()
    return snapshots


@torch.no_grad()
def _encode_pbr_fields_batch(
    *,
    group: Sequence[PreparedContext],
    fields: Mapping[int, torch.Tensor],
    pbr_encoder: torch.nn.Module,
    pipeline: Any,
    device: torch.device,
    label: str,
) -> Dict[int, torch.Tensor]:
    """Encode complete local C1024 fields and gather their mapped C64 rows."""
    values: List[SparseTensor] = []
    for context in group:
        field = fields[context.context_id].detach().cpu().float()
        if field.shape != (context.geometry.coords.shape[0], 6):
            raise RuntimeError(
                f"{label} context {context.context_id}: field shape {field.shape} "
                f"does not match complete C1024 support {context.geometry.coords.shape}"
            )
        coords = torch.cat(
            (torch.zeros((field.shape[0], 1), dtype=torch.int32), context.geometry.coords.cpu()),
            dim=1,
        )
        values.append(SparseTensor(field * 2.0 - 1.0, coords))
    packed = first_view_route._pack_sparse_batch(values, label).to(device)
    pbr_encoder.to(device)
    pbr_encoder.eval()
    raw = pbr_encoder(packed, sample_posterior=False)
    if not isinstance(raw, SparseTensor) or not torch.isfinite(raw.feats).all():
        raise FloatingPointError(f"{label}: official PBR encoder returned non-finite output")
    parts = multiview_route._unpack_sparse_batch_parts(raw, len(group), label)
    result: Dict[int, torch.Tensor] = {}
    for context, part in zip(group, parts):
        normalized = cross_tile._normalize_slat(part.cpu(), pipeline.tex_slat_normalization)
        mapped = _gather_coords(normalized, context.local_coords, f"{label} context {context.context_id}")
        if not torch.isfinite(mapped).all():
            raise FloatingPointError(f"{label}: mapped endpoint is non-finite")
        result[context.context_id] = mapped
    del values, packed, raw, parts
    _empty_cuda_cache()
    return result


def _pbr_global_to_context(
    context: PreparedContext,
    global_pbr: torch.Tensor,
    master_count: int,
) -> torch.Tensor:
    """Continuously query fused C64 master anchors onto full local C1024.

    A small dense 64^3 value/denominator grid is materialized per context and
    queried with chunked trilinear interpolation at the actual local dual
    vertices.  Empty interpolation denominators retain the context's cached
    baseline PBR row; this is a value fallback, never a support mutation.
    """
    anchor_coords = context.local_coords[:, 1:].to(torch.long)
    if bool(((anchor_coords < 0) | (anchor_coords >= LATENT_SIZE)).any()):
        raise RuntimeError(f"context {context.context_id}: invalid C64 anchor coordinate")
    anchor_values = global_pbr.index_select(0, context.master_ids).float()
    if not torch.isfinite(anchor_values).all():
        raise FloatingPointError(f"context {context.context_id}: non-finite master PBR anchor")
    dense_values = torch.zeros((LATENT_SIZE ** 3, 6), dtype=torch.float32)
    dense_valid = torch.zeros((LATENT_SIZE ** 3,), dtype=torch.float32)
    anchor_keys = (anchor_coords[:, 0] * LATENT_SIZE + anchor_coords[:, 1]) * LATENT_SIZE + anchor_coords[:, 2]
    dense_values.index_copy_(0, anchor_keys, anchor_values)
    dense_valid.index_fill_(0, anchor_keys, 1.0)
    target_q = context.target_points.float() * (2.0 * float(context.transform.mesh_scale))
    fractional = ((target_q + 1.0) * (float(LATENT_SIZE - 1) / 2.0)).clamp(0.0, float(LATENT_SIZE - 1))
    lower = torch.floor(fractional).to(torch.long)
    upper = (lower + 1).clamp_max(LATENT_SIZE - 1)
    frac = fractional - lower.to(torch.float32)
    field = context.baseline_pbr.clone()
    for start in range(0, int(target_q.shape[0]), 262_144):
        end = min(start + 262_144, int(target_q.shape[0]))
        x0, y0, z0 = lower[start:end, 0], lower[start:end, 1], lower[start:end, 2]
        x1, y1, z1 = upper[start:end, 0], upper[start:end, 1], upper[start:end, 2]
        fx, fy, fz = frac[start:end, 0], frac[start:end, 1], frac[start:end, 2]
        rows: List[torch.Tensor] = []
        den_rows: List[torch.Tensor] = []
        for ix, wx in ((x0, 1.0 - fx), (x1, fx)):
            for iy, wy in ((y0, 1.0 - fy), (y1, fy)):
                for iz, wz in ((z0, 1.0 - fz), (z1, fz)):
                    keys = (ix * LATENT_SIZE + iy) * LATENT_SIZE + iz
                    weight = wx * wy * wz
                    rows.append(dense_values.index_select(0, keys) * weight[:, None])
                    den_rows.append(dense_valid.index_select(0, keys) * weight)
        denominator = torch.stack(den_rows, dim=1).sum(dim=1)
        interpolated = torch.stack(rows, dim=1).sum(dim=1)
        valid = denominator > 1e-8
        if bool(valid.any()):
            interpolated[valid] = interpolated[valid] / denominator[valid, None]
            valid &= torch.isfinite(interpolated).all(dim=1)
            field_rows = field[start:end]
            field_rows[valid] = interpolated[valid]
    if field.shape[0] != context.geometry.coords.shape[0] or not torch.isfinite(field).all():
        raise RuntimeError(
            f"context {context.context_id}: fused PBR query did not cover complete local support"
        )
    return field


def _fuse_pbr_at_master(
    contexts: Sequence[PreparedContext],
    snapshots: Mapping[int, DecodedTexture],
    baseline_pbr_global: torch.Tensor,
    master_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    pbr_sum = torch.zeros_like(baseline_pbr_global)
    weight_sum = torch.zeros((master_count,), dtype=torch.float32)
    donor_count = torch.zeros((master_count,), dtype=torch.int32)
    valid_query = 0
    for context in contexts:
        snapshot = snapshots[context.context_id]
        valid = context.visible & snapshot.pbr_valid
        valid_query += int(snapshot.pbr_valid.sum())
        if bool(valid.any()):
            ids = context.master_ids[valid]
            weights = context.gaussian_weight[valid].float()
            values = snapshot.pbr_at_master[valid]
            pbr_sum.index_add_(0, ids, values * weights[:, None])
            weight_sum.index_add_(0, ids, weights)
            donor_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    fused = baseline_pbr_global.clone()
    active = weight_sum > 0.0
    fused[active] = pbr_sum[active] / weight_sum[active, None]
    if not torch.isfinite(fused).all():
        raise FloatingPointError("global PBR fusion returned non-finite values")
    return fused, donor_count, {
        "valid_query_rows": valid_query,
        "donor_one": int((donor_count == 1).sum()),
        "donor_multiple": int((donor_count > 1).sum()),
        "donor_zero_baseline": int((donor_count == 0).sum()),
        "denominator_valid_rows": int(active.sum()),
        "visible_only": True,
        "gaussian_sigma_virtual_4096": SIGMA_PIXELS,
    }


def _run_texture_flow(
    *,
    contexts: Sequence[PreparedContext],
    shape_global_final: torch.Tensor,
    texture_fallback: torch.Tensor,
    baseline_pbr_global: torch.Tensor,
    pipeline: Any,
    pbr_encoder: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    seed: int,
    steps_override: Optional[int],
    resume: bool,
    save_step_tensors: bool,
    query_chunk_size: int,
    master_count: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    params = dict(pipeline.tex_slat_sampler_params)
    if steps_override is not None:
        params["steps"] = int(steps_override)
    schedule = _schedule(sampler, params)
    shape_concat: Dict[int, SparseTensor] = {
        context.context_id: SparseTensor(
            shape_global_final.index_select(0, context.master_ids),
            context.local_coords.clone(),
        )
        for context in contexts
    }
    states: Dict[int, SparseTensor] = {}
    for context in contexts:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + int(context.angle) * 100003 + int(context.tile_id) * 1009 + 71
        )
        features = torch.randn(
            (context.master_ids.numel(), context.texture_full.feats.shape[1]),
            generator=generator,
            dtype=torch.float32,
        )
        states[context.context_id] = SparseTensor(features, context.local_coords.clone())
    stage_dir = output_dir / "texture"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _save_local_states(stage_dir / "initial_noise.pt", states)
    state_hash = _hash_many({"state": {str(k): v.feats for k, v in states.items()}, "shape": shape_global_final})
    checkpoint = stage_dir / "checkpoint.pt"
    start_step = 0
    records: List[Dict[str, Any]] = []
    final_endpoint: Optional[torch.Tensor] = None
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("support_hash") != state_hash or int(saved.get("steps", -1)) != len(schedule) - 1:
            raise RuntimeError("texture resume checkpoint rejected because support or schedule changed")
        states = _load_local_states(stage_dir / "state_next.pt")
        start_step = int(saved["next_step"])
        records = list(saved.get("records", []))
        if saved.get("global_endpoint") is not None:
            final_endpoint = saved["global_endpoint"].float()
    for step in range(start_step, len(schedule) - 1):
        t, t_next = schedule[step], schedule[step + 1]
        frozen = {cid: _clone_sparse(value) for cid, value in states.items()}
        predictions, prediction_stats = _predict_stage_batches(
            contexts=contexts,
            states=frozen,
            conditions={context.context_id: context.condition_texture for context in contexts},
            model=model,
            sampler=sampler,
            params=params,
            t=t,
            t_next=t_next,
            device=device,
            concat=shape_concat,
            stage="texture",
        )
        snapshots = _decode_texture_batches(
            contexts=contexts,
            predictions=predictions,
            shape_concat=shape_concat,
            pipeline=pipeline,
            device=device,
            query_chunk_size=query_chunk_size,
        )
        fused_pbr, donor_count, pbr_stats = _fuse_pbr_at_master(
            contexts, snapshots, baseline_pbr_global, master_count
        )
        fused_fields = {
            context.context_id: _pbr_global_to_context(context, fused_pbr, master_count)
            for context in contexts
        }
        self_fields = {
            context.context_id: snapshots[context.context_id].self_field
            for context in contexts
        }
        corrected: Dict[int, SparseTensor] = {}
        cycle_rows: List[Dict[str, Any]] = []
        pbr_encoder.to(device)
        for start in range(0, len(contexts), PBR_ENCODE_BATCH_SIZE):
            group = contexts[start : start + PBR_ENCODE_BATCH_SIZE]
            fused_encoded = _encode_pbr_fields_batch(
                group=group,
                fields=fused_fields,
                pbr_encoder=pbr_encoder,
                pipeline=pipeline,
                device=device,
                label="texture fused PBR encode",
            )
            self_encoded = _encode_pbr_fields_batch(
                group=group,
                fields=self_fields,
                pbr_encoder=pbr_encoder,
                pipeline=pipeline,
                device=device,
                label="texture self PBR encode",
            )
            for context in group:
                pred = predictions[context.context_id]
                correction = fused_encoded[context.context_id] - self_encoded[context.context_id]
                corrected_features = pred.feats + correction
                if not torch.isfinite(corrected_features).all():
                    raise FloatingPointError(f"texture context {context.context_id}: cycle endpoint non-finite")
                corrected[context.context_id] = pred.replace(corrected_features)
                cycle_rows.append({
                    "context_id": int(context.context_id),
                    "correction": _stats(correction),
                    "cycle_coefficient": 1.0,
                })
            del fused_encoded, self_encoded
        pbr_encoder.cpu()
        endpoint, visible_count, fallback_mask, fusion_stats = _fuse_endpoint(
            contexts=contexts,
            predictions=corrected,
            fallback=texture_fallback,
            master_count=master_count,
            channel_count=texture_fallback.shape[1],
            stage="texture",
        )
        states = _update_local_states(
            contexts=contexts,
            frozen=frozen,
            global_endpoint=endpoint,
            sampler=sampler,
            t=t,
            t_next=t_next,
            device=device,
            label="texture",
        )
        final_endpoint = endpoint
        step_dir = stage_dir / f"step_{step:02d}"
        if save_step_tensors:
            _atomic_save(step_dir / "global_endpoint.pt", {"master_id": torch.arange(master_count), "features": endpoint})
            _atomic_save(step_dir / "visible_count.pt", visible_count)
            _atomic_save(step_dir / "fallback_mask.pt", fallback_mask)
        _atomic_json(step_dir / "pbr_fusion_stats.json", pbr_stats)
        _atomic_json(
            step_dir / "cycle_stats.json",
            {
                "cycle_coefficient": 1.0,
                "rows": cycle_rows,
                "correction_mean_abs": float(torch.cat([
                    (corrected[c.context_id].feats - predictions[c.context_id].feats).abs().reshape(-1)
                    for c in contexts
                ]).mean()),
                "self_endpoint_and_fused_endpoint_are_real_batched_encodes": True,
            },
        )
        record = {
            "step": step,
            "t": t,
            "t_next": t_next,
            "contexts": len(contexts),
            "prediction": prediction_stats,
            "decode_batch_size": DECODE_BATCH_SIZE,
            "pbr_encode_batch_size": PBR_ENCODE_BATCH_SIZE,
            "pbr_fusion": pbr_stats,
            "endpoint_fusion": fusion_stats,
            "cycle": "pred_x0 + normalize(E_fused) - normalize(E_self)",
            "jacobi_barrier": True,
            "local_state_independent_noise": True,
            "velocity_averaging": False,
        }
        _atomic_json(step_dir / "stats.json", record)
        records.append(record)
        _save_local_states(stage_dir / "state_next.pt", states)
        _atomic_save(
            checkpoint,
            {
                "format": FORMAT,
                "stage": "texture",
                "support_hash": state_hash,
                "steps": len(schedule) - 1,
                "next_step": step + 1,
                "global_endpoint": endpoint,
                "records": records,
            },
        )
        del predictions, snapshots, fused_fields, self_fields, corrected
        _empty_cuda_cache()
    if final_endpoint is None:
        raise RuntimeError("texture flow did not produce a final endpoint")
    _atomic_save(output_dir / "final" / "texture_global_final.pt", {"master_id": torch.arange(master_count), "features": final_endpoint})
    summary = {
        "format": FORMAT,
        "stage": "texture",
        "steps": len(schedule) - 1,
        "schedule": list(schedule),
        "records": records,
        "flow_batch_size": FLOW_BATCH_SIZE,
        "decode_batch_size": DECODE_BATCH_SIZE,
        "pbr_encode_batch_size": PBR_ENCODE_BATCH_SIZE,
        "real_multi_batch": True,
        "cycle_coefficient": 1.0,
        "suffix_rollout_used": False,
        "velocity_averaging": False,
        "final_endpoint": str(output_dir / "final" / "texture_global_final.pt"),
    }
    _atomic_json(stage_dir / "flow_summary.json", summary)
    return final_endpoint, summary


def _build_baseline_endpoints(
    *,
    contexts: Sequence[PreparedContext],
    baseline_pbr_global: torch.Tensor,
    output_dir: Path,
    master_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build shape/texture fallback endpoints from complete local subgraphs."""
    if not contexts:
        raise RuntimeError("cannot build fallback endpoints without active contexts")
    shape_channels = int(contexts[0].shape_reference.shape[1])
    texture_channels = int(contexts[0].texture_reference.shape[1])
    shape_sum = torch.zeros((master_count, shape_channels), dtype=torch.float32)
    texture_sum = torch.zeros((master_count, texture_channels), dtype=torch.float32)
    weight_sum = torch.zeros((master_count,), dtype=torch.float32)
    reference_count = torch.zeros((master_count,), dtype=torch.int32)
    source_mask = torch.zeros((len(contexts), master_count), dtype=torch.bool)
    shape_per_context: Dict[str, Any] = {}
    texture_per_context: Dict[str, Any] = {}
    for row, context in enumerate(contexts):
        ids = context.master_ids
        weight = context.gaussian_weight.float()
        shape_ref = context.shape_reference.float()
        texture_ref = context.texture_reference.float()
        valid = (
            torch.isfinite(shape_ref).all(dim=1)
            & torch.isfinite(texture_ref).all(dim=1)
            & torch.isfinite(weight)
            & (weight > 0.0)
        )
        if not bool(valid.all()):
            raise FloatingPointError(
                f"context {context.context_id}: baseline endpoint contains invalid rows"
            )
        source_mask[row, ids] = True
        shape_sum.index_add_(0, ids, shape_ref * weight[:, None])
        texture_sum.index_add_(0, ids, texture_ref * weight[:, None])
        weight_sum.index_add_(0, ids, weight)
        reference_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        shape_per_context[str(context.context_id)] = {
            "master_ids": ids,
            "endpoint": shape_ref,
            "valid": valid,
            "weight": weight,
        }
        texture_per_context[str(context.context_id)] = {
            "master_ids": ids,
            "endpoint": texture_ref,
            "valid": valid,
            "weight": weight,
        }
    if bool((weight_sum <= 0.0).any()):
        missing = torch.where(weight_sum <= 0.0)[0][:32].tolist()
        raise RuntimeError(f"baseline fallback has no reference for master IDs {missing}")
    shape_global = shape_sum / weight_sum[:, None]
    texture_global = texture_sum / weight_sum[:, None]
    if not torch.isfinite(shape_global).all() or not torch.isfinite(texture_global).all():
        raise FloatingPointError("baseline fallback endpoint is non-finite")
    baseline_dir = output_dir / "baseline"
    _atomic_save(
        baseline_dir / "shape_endpoint_per_context.pt",
        {"format": FORMAT, "contexts": shape_per_context},
    )
    _atomic_save(
        baseline_dir / "texture_endpoint_per_context.pt",
        {"format": FORMAT, "contexts": texture_per_context},
    )
    _atomic_save(
        baseline_dir / "shape_endpoint_global.pt",
        {"format": FORMAT, "master_id": torch.arange(master_count), "features": shape_global},
    )
    _atomic_save(
        baseline_dir / "texture_endpoint_global.pt",
        {"format": FORMAT, "master_id": torch.arange(master_count), "features": texture_global},
    )
    _atomic_save(
        baseline_dir / "reference_sources.pt",
        {
            "format": FORMAT,
            "source_context_ids": torch.tensor([c.context_id for c in contexts], dtype=torch.int64),
            "reference_mask": source_mask,
            "reference_count": reference_count,
            "sum_weight": weight_sum,
        },
    )
    _atomic_json(
        baseline_dir / "reference_count_and_weights.json",
        {
            "format": FORMAT,
            "master_count": master_count,
            "all_master_ids_covered": True,
            "reference_count": reference_count.tolist(),
            "sum_weight": weight_sum.tolist(),
            "source_context_ids": [int(c.context_id) for c in contexts],
            "encoder_unit": "one complete local C1024 subgraph per context before row gather",
            "shape_global_rule": "tile-center Gaussian mean of per-context normalized shape endpoints",
            "texture_global_rule": "tile-center Gaussian mean of per-context normalized PBR-encoded texture endpoints",
        },
    )
    return shape_global, texture_global, {
        "master_count": master_count,
        "context_count": len(contexts),
        "min_reference_count": int(reference_count.min()),
        "max_reference_count": int(reference_count.max()),
        "shape_channels": shape_channels,
        "texture_channels": texture_channels,
        "baseline_pbr_global_stats": _stats(baseline_pbr_global),
    }


def _load_or_query_baseline_pbr(
    *,
    baseline_field: MeshWithVoxel,
    nearest_point: torch.Tensor,
    output_path: Path,
    device: torch.device,
    query_chunk_size: int,
) -> torch.Tensor:
    cache_key = _hash_many({"nearest_point": nearest_point, "attrs": baseline_field.attrs})
    if output_path.is_file():
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if payload.get("cache_key") == cache_key:
            values = payload["pbr"].float()
            if values.shape == (nearest_point.shape[0], 6) and torch.isfinite(values).all():
                return values
    values = cross_tile._query_mesh_chunked(
        baseline_field,
        nearest_point.to(device),
        int(query_chunk_size),
    ).detach().cpu().float()
    if values.shape != (nearest_point.shape[0], 6) or not torch.isfinite(values).all():
        raise FloatingPointError("baseline PBR nearest-triangle query is invalid")
    _atomic_save(output_path, {"format": FORMAT, "cache_key": cache_key, "pbr": values})
    return values


@torch.no_grad()
def _query_baseline_pbr_on_dual_support(
    *,
    geometry: core.LocalGeometry,
    baseline_field: MeshWithVoxel,
    camera: Mapping[str, float],
    transform: Any,
    rotation: torch.Tensor,
    device: torch.device,
    query_chunk_size: int,
) -> torch.Tensor:
    """Query the immutable baseline field directly at every local C1024 dual row.

    The local C1024 support is complete before this function is called.  Each
    dual vertex is mapped through the official local-camera transform and the
    fixed view yaw into the baseline object frame; no local triangle
    correspondence or support mutation is involved.
    """
    if query_chunk_size <= 0:
        raise ValueError("material query chunk size must be positive")
    local_q = geometry.dual_vertices_world.to(device=device, dtype=torch.float32) * (
        2.0 * float(transform.mesh_scale)
    )
    q_view, _ = core._local_q_to_global_q(
        local_q, global_camera=camera, transform=transform
    )
    q_world = _view_to_world_q(q_view, rotation)
    points = q_world / (2.0 * float(camera["mesh_scale"]))
    values = cross_tile._query_mesh_chunked(
        baseline_field, points, int(query_chunk_size)
    ).detach().cpu().float()
    expected = (int(geometry.coords.shape[0]), 6)
    if values.shape != expected or not torch.isfinite(values).all():
        raise FloatingPointError(
            f"direct baseline PBR query returned {tuple(values.shape)}, expected {expected}"
        )
    return values


def _attach_nearest_context_points(
    contexts: Sequence[PreparedContext],
    nearest_point: torch.Tensor,
    camera: Mapping[str, float],
) -> None:
    for context in contexts:
        rotation = _yaw_matrix(context.angle)
        q_world = nearest_point.index_select(0, context.master_ids) * (
            2.0 * float(camera["mesh_scale"])
        )
        q_view = _world_to_view_q(q_world, rotation)
        q_local, uv = core._global_q_to_local_q(
            q_view, global_camera=camera, transform=context.transform
        )
        context.nearest_local_points = q_local / (2.0 * float(context.transform.mesh_scale))
        context.nearest_local_uv = uv.to(torch.float32)


def _load_prepared_context_cache(
    *,
    cache_dir: Path,
    pending: Sequence[Mapping[str, Any]],
    support: first_view_route.MasterSupport,
    camera: Mapping[str, float],
) -> Optional[List[PreparedContext]]:
    """Restore prepared context metadata without rerunning local encoders.

    This cache contains only current-run master mappings and baseline mapped
    endpoint rows.  It never imports encoder feature values into support
    construction; the dummy ``shape_full``/``texture_full`` tensors below are
    used only for their channel count after the mapped references have already
    been persisted by the completed preparation run.
    """
    cache_dir = cache_dir.expanduser().resolve()
    mapping_path = cache_dir / "support" / "context_mapping.pt"
    shape_path = cache_dir / "baseline" / "shape_endpoint_per_context.pt"
    texture_path = cache_dir / "baseline" / "texture_endpoint_per_context.pt"
    if not (mapping_path.is_file() and shape_path.is_file() and texture_path.is_file()):
        return None
    mapping_payload = torch.load(mapping_path, map_location="cpu", weights_only=False)
    shape_payload = torch.load(shape_path, map_location="cpu", weights_only=False)
    texture_payload = torch.load(texture_path, map_location="cpu", weights_only=False)
    if mapping_payload.get("format") != FORMAT:
        return None
    if int(mapping_payload.get("master_count", -1)) != int(support.master_q_global.shape[0]):
        return None
    pending_by_id = {int(item["context_id"]): item for item in pending}
    shape_rows = shape_payload.get("contexts", {})
    texture_rows = texture_payload.get("contexts", {})
    contexts: List[PreparedContext] = []
    for row in mapping_payload.get("contexts", []):
        if row.get("status") != "active":
            continue
        context_id = int(row["context_id"])
        item = pending_by_id.get(context_id)
        shape_row = shape_rows.get(str(context_id), shape_rows.get(context_id))
        texture_row = texture_rows.get(str(context_id), texture_rows.get(context_id))
        if item is None or shape_row is None or texture_row is None:
            return None
        master_ids = row["master_ids"].to(torch.int64).contiguous()
        local_coords = row["local_coords_c64"].to(torch.int32).contiguous()
        uv_virtual = row["uv_virtual_4096"].to(torch.float32).contiguous()
        shape_reference = shape_row["endpoint"].to(torch.float32).contiguous()
        texture_reference = texture_row["endpoint"].to(torch.float32).contiguous()
        if (
            shape_reference.shape[0] != master_ids.numel()
            or texture_reference.shape[0] != master_ids.numel()
            or shape_reference.ndim != 2
            or texture_reference.ndim != 2
            or not torch.isfinite(shape_reference).all()
            or not torch.isfinite(texture_reference).all()
        ):
            return None
        shape_full = SparseTensor(shape_reference.clone(), local_coords.clone())
        texture_full = SparseTensor(texture_reference.clone(), local_coords.clone())
        view = _make_local_view(
            context_id,
            item["transform"],
            tuple(int(v) for v in row["virtual_box"]),
            master_ids,
            local_coords,
            uv_virtual,
            dict(row.get("stats", {})),
        )
        rotation = _yaw_matrix(int(item["angle"]))
        local_q = item["geometry"].dual_vertices_world * (
            2.0 * float(item["transform"].mesh_scale)
        )
        q_view, _ = core._local_q_to_global_q(
            local_q, global_camera=camera, transform=item["transform"]
        )
        target_world_points = _view_to_world_q(q_view, rotation) / (
            2.0 * float(camera["mesh_scale"])
        )
        pbr_path = item["tile_dir"] / "baseline_pbr_field.pt"
        if not pbr_path.is_file():
            pbr_path = cache_dir / "contexts" / f"context_{context_id:03d}" / "baseline_pbr_field.pt"
        if not pbr_path.is_file():
            return None
        pbr_payload = torch.load(pbr_path, map_location="cpu", weights_only=False)
        baseline_pbr = pbr_payload.get("pbr")
        if not isinstance(baseline_pbr, torch.Tensor):
            return None
        baseline_pbr = baseline_pbr.float().contiguous()
        if baseline_pbr.shape != (item["geometry"].coords.shape[0], 6):
            return None
        valid_global = torch.zeros((int(support.master_q_global.shape[0]),), dtype=torch.bool)
        valid_global[master_ids] = True
        contexts.append(
            PreparedContext(
                context_id=context_id,
                angle=int(item["angle"]),
                angle_index=int(item["angle_index"]),
                tile_id=int(item["tile_id"]),
                source_box=tuple(int(v) for v in item["source_box"]),
                virtual_box=tuple(int(v) for v in row["virtual_box"]),
                transform=item["transform"],
                tile_image=item["tile_image"],
                tile_dir=item["tile_dir"],
                geometry=item["geometry"],
                baseline_pbr=baseline_pbr,
                shape_full=shape_full,
                texture_full=texture_full,
                native_coords=local_coords.clone(),
                view=view,
                master_ids=master_ids,
                local_coords=local_coords,
                uv_virtual=uv_virtual,
                gaussian_weight=row["gaussian_weight"].to(torch.float32).contiguous(),
                shape_reference=shape_reference,
                texture_reference=texture_reference,
                target_points=item["geometry"].dual_vertices_world.clone(),
                target_world_points=target_world_points.cpu(),
                nearest_local_points=torch.zeros((master_ids.numel(), 3), dtype=torch.float32),
                nearest_local_uv=torch.zeros((master_ids.numel(), 2), dtype=torch.float32),
                visible=torch.zeros((master_ids.numel(),), dtype=torch.bool),
                mapping_valid_global=valid_global,
                shape_state=None,
                texture_state=None,
                condition_shape=None,
                condition_texture=None,
                support_stats={
                    "status": "active",
                    "prepared_context_cache": str(cache_dir),
                    "mapping": dict(row.get("stats", {})),
                    "local_ovoxel_count": int(item["geometry"].coords.shape[0]),
                },
            )
        )
    if len(contexts) != len(pending_by_id):
        return None
    print(f"[prepared-cache] restored {len(contexts)} contexts from {cache_dir}", flush=True)
    return sorted(contexts, key=lambda context: context.context_id)


def _prepare_contexts(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    canonical: Mapping[str, Any],
    multiview_images: Mapping[int, Image.Image],
    output_dir: Path,
    device: torch.device,
) -> Tuple[first_view_route.MasterSupport, List[PreparedContext], Dict[str, Any], torch.nn.Module]:
    """Prepare every local C1024 subgraph and create only master mappings."""
    first_boxes = _tile_boxes(True)
    view_boxes = _tile_boxes(False)
    first_transforms = {
        tile_id: core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=camera,
            extend_pixel=0,
            source_width=CANONICAL_SIZE,
            source_height=CANONICAL_SIZE,
            model_width=MODEL_TILE_SIZE,
            model_height=MODEL_TILE_SIZE,
        )
        for tile_id, box in enumerate(first_boxes)
    }
    support = _support_from_source(args.first_view_support, camera, first_transforms)
    if support is None:
        print("[support] source support unavailable; first-view support will be built from C64 coordinates", flush=True)
    else:
        # Old formal support files are accepted only after a projective check.
        projected, _, finite = core._project_global_q_to_image(
            support.master_q_global,
            global_camera=camera,
            image_width=CANONICAL_SIZE,
            image_height=CANONICAL_SIZE,
        )
        if not bool(finite.all()) or float((projected - support.master_uv_4096).abs().max()) > 2e-3:
            raise RuntimeError("first-view source support failed its camera/schema check")
        print(f"[support] reuse first-view dense master rows={support.master_q_global.shape[0]}", flush=True)

    face_bounds_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    face_bounds_cache[0] = _face_bounds(
        baseline,
        camera,
        output_dir / "support" / "face_projection_bounds_4096.pt",
        source_size=CANONICAL_SIZE,
        chunk_size=int(args.face_projection_chunk_size),
    )
    rotations = {angle: _yaw_matrix(angle) for angle in ANGLES}
    for angle in ANGLES[1:]:
        face_bounds_cache[angle] = _face_bounds(
            baseline,
            camera,
            output_dir / "support" / f"face_projection_bounds_view_{angle:03d}.pt",
            source_size=VIEW_SIZE,
            chunk_size=int(args.face_projection_chunk_size),
            vertices=baseline.vertices.cpu() @ rotations[angle],
        )

    pending: List[Dict[str, Any]] = []
    preparation_rows: List[Dict[str, Any]] = []
    baseline_field: Optional[MeshWithVoxel] = None
    for angle_index, angle in enumerate(ANGLES):
        canonical_view = angle == 0
        boxes = first_boxes if canonical_view else view_boxes
        for tile_id, source_box in enumerate(boxes):
            context_id = angle_index * TILE_COUNT + tile_id
            if canonical_view and support is not None and tile_id not in support.tile_views:
                preparation_rows.append({
                    "context_id": context_id,
                    "angle": angle,
                    "tile_id": tile_id,
                    "source_box": source_box,
                    "virtual_box": _virtual_box(source_box, canonical_view),
                    "status": "inactive_first_view_support",
                    "master_ids": torch.empty((0,), dtype=torch.int64),
                    "local_coords": torch.empty((0, 4), dtype=torch.int32),
                    "uv_virtual": torch.empty((0, 2), dtype=torch.float32),
                    "gaussian_weight": torch.empty((0,), dtype=torch.float32),
                    "mapping_stats": {"reason": "not in immutable first-view support"},
                })
                continue
            transform = first_transforms[tile_id] if canonical_view else core._derive_tile_camera(
                tile_id=tile_id,
                box=source_box,
                global_camera=camera,
                extend_pixel=0,
                source_width=VIEW_SIZE,
                source_height=VIEW_SIZE,
                model_width=MODEL_TILE_SIZE,
                model_height=MODEL_TILE_SIZE,
            )
            if canonical_view:
                tile_image = canonical["image_4096"].crop(source_box).convert("RGB")
                if tile_image.size != (MODEL_TILE_SIZE, MODEL_TILE_SIZE):
                    raise RuntimeError(f"first-view tile {tile_id} is not native 1024")
            else:
                tile_image = multiview_images[angle].crop(source_box).convert("RGB")
                if tile_image.size != (VIEW_TILE_SIZE, VIEW_TILE_SIZE):
                    raise RuntimeError(f"view {angle} tile {tile_id} is not native 256")
                tile_image = tile_image.resize((MODEL_TILE_SIZE, MODEL_TILE_SIZE), Image.Resampling.BICUBIC)
            tile_dir = output_dir / "contexts" / f"context_{context_id:03d}"
            tile_dir.mkdir(parents=True, exist_ok=True)
            geometry_path = tile_dir / "geometry.pt"
            try:
                legacy_dual_path = None
                if angle == 0:
                    legacy_dual_path = (
                        args.first_view_support.expanduser().resolve().parent
                        / "tile_geometry"
                        / f"tile_{tile_id:02d}.pt"
                    )
                geometry = _prepare_geometry(
                    baseline=baseline,
                    camera=camera,
                    transform=transform,
                    face_bounds=face_bounds_cache[angle],
                    rotated_vertices=baseline.vertices.cpu() @ rotations[angle],
                    output_path=geometry_path,
                    reuse_dual_path=legacy_dual_path,
                )
            except RuntimeError as exc:
                if "no global triangle projection bbox intersects" in str(exc):
                    preparation_rows.append({
                        "context_id": context_id,
                        "angle": angle,
                        "tile_id": tile_id,
                        "source_box": source_box,
                        "virtual_box": _virtual_box(source_box, canonical_view),
                        "status": "empty_projective_support",
                        "master_ids": torch.empty((0,), dtype=torch.int64),
                        "local_coords": torch.empty((0, 4), dtype=torch.int32),
                        "uv_virtual": torch.empty((0, 2), dtype=torch.float32),
                        "gaussian_weight": torch.empty((0,), dtype=torch.float32),
                        "mapping_stats": {"reason": "no projected baseline face"},
                    })
                    continue
                raise
            pbr_cache = tile_dir / "baseline_pbr_field.pt"
            pbr_cache_key = _hash_many({
                "query_route": "direct_dual_support_v1",
                "coords": geometry.coords,
                "selected_faces": geometry.selected_global_face_ids,
                "angle": angle,
                "camera": dict(camera),
            })
            if pbr_cache.is_file():
                pbr_payload = torch.load(pbr_cache, map_location="cpu", weights_only=False)
                baseline_pbr = pbr_payload["pbr"].float() if pbr_payload.get("cache_key") == pbr_cache_key else None
            else:
                baseline_pbr = None
            if baseline_pbr is None:
                if baseline_field is None:
                    baseline_field = core._make_attribute_query_mesh(baseline, device)
                baseline_pbr = _query_baseline_pbr_on_dual_support(
                    geometry=geometry,
                    baseline_field=baseline_field,
                    camera=camera,
                    transform=transform,
                    rotation=rotations[angle],
                    device=device,
                    query_chunk_size=int(args.material_query_chunk_size),
                )
                _atomic_save(pbr_cache, {"format": FORMAT, "cache_key": pbr_cache_key, "pbr": baseline_pbr})
            if baseline_pbr.shape != (geometry.coords.shape[0], 6) or not torch.isfinite(baseline_pbr).all():
                raise FloatingPointError(f"context {context_id}: baseline PBR support is invalid")
            pending.append({
                "context_id": context_id,
                "angle": angle,
                "angle_index": angle_index,
                "tile_id": tile_id,
                "source_box": source_box,
                "virtual_box": _virtual_box(source_box, canonical_view),
                "transform": transform,
                "tile_image": tile_image,
                "tile_dir": tile_dir,
                "geometry": geometry,
                "local_attrs": baseline_pbr,
                "baseline_pbr": baseline_pbr,
            })
    if baseline_field is not None:
        del baseline_field
        _empty_cuda_cache()
    if not pending:
        raise RuntimeError("no local contexts have projective support")

    prepared_cache = getattr(args, "prepared_context_cache", None)
    if prepared_cache is not None and support is not None:
        cached_contexts = _load_prepared_context_cache(
            cache_dir=Path(prepared_cache),
            pending=pending,
            support=support,
            camera=camera,
        )
        if cached_contexts is not None:
            support_hash = _save_master_support(support, output_dir, first_transforms)
            pbr_encoder = pixal3d_models.from_pretrained(str(args.pbr_encoder)).eval().cpu()
            return support, cached_contexts, {"support_sha256": support_hash, "preparation_rows": TILE_COUNT * len(ANGLES)}, pbr_encoder

    shape_encoder = pixal3d_models.from_pretrained(str(args.shape_encoder)).eval()
    pbr_encoder = pixal3d_models.from_pretrained(str(args.pbr_encoder)).eval()
    encoded_contexts: List[PreparedContext] = []
    initial_batch = int(args.initial_context_encode_batch_size)
    if initial_batch <= 0:
        raise ValueError("initial context encode batch size must be positive")
    for start in range(0, len(pending), initial_batch):
        group = pending[start : start + initial_batch]
        encoded = multiview_route._encode_initial_batch(
            group,
            shape_encoder,
            pbr_encoder,
            pipeline,
            getattr(args, "_profile_records", None),
        )
        for item, (shape_raw, texture_raw, shape_stats, texture_stats) in zip(group, encoded):
            shape_full = cross_tile._normalize_slat(shape_raw.cpu(), pipeline.shape_slat_normalization)
            texture_full = cross_tile._normalize_slat(texture_raw.cpu(), pipeline.tex_slat_normalization)
            if not torch.equal(shape_full.coords, texture_full.coords):
                raise RuntimeError(f"context {item['context_id']}: shape/texture baseline support differs")
            virtual_box = item["virtual_box"]
            if item["angle"] == 0 and support is not None:
                source_view = support.tile_views[item["tile_id"]]
                master_ids = source_view.master_ids.clone()
                local_coords = source_view.local_coords.clone()
                uv_virtual = source_view.master_uv_4096.clone()
                mapping_stats = {"source": "immutable first-view support tile mapping"}
                valid_global = torch.zeros((int(support.master_q_global.shape[0]),), dtype=torch.bool)
                valid_global[master_ids] = True
            else:
                if support is None:
                    first_native = {
                        int(other["tile_id"]): other["shape_full"].coords
                        for other in []
                    }
                    # The support is built below after all first-view encoder
                    # rows are available; postpone this context's mapping.
                    master_ids = torch.empty((0,), dtype=torch.int64)
                    local_coords = torch.empty((0, 4), dtype=torch.int32)
                    uv_virtual = torch.empty((0, 2), dtype=torch.float32)
                    mapping_stats = {"postponed_until_support_build": True}
                    valid_global = torch.empty((0,), dtype=torch.bool)
                else:
                    master_ids, local_coords, uv_virtual, valid_global, mapping_stats = _map_master_to_context(
                        master_q_world=support.master_q_global,
                        native_coords=shape_full.coords,
                        angle=int(item["angle"]),
                        transform=item["transform"],
                        camera=camera,
                        virtual_box=virtual_box,
                    )
            rotation = rotations[item["angle"]]
            local_q = item["geometry"].dual_vertices_world * (2.0 * float(item["transform"].mesh_scale))
            q_view, _ = core._local_q_to_global_q(local_q, global_camera=camera, transform=item["transform"])
            target_world_points = _view_to_world_q(q_view, rotation) / (2.0 * float(camera["mesh_scale"]))
            if support is not None and master_ids.numel():
                shape_reference = _gather_coords(shape_full, local_coords, f"context {item['context_id']} shape baseline")
                texture_reference = _gather_coords(texture_full, local_coords, f"context {item['context_id']} texture baseline")
                view = _make_local_view(
                    item["context_id"], item["transform"], virtual_box,
                    master_ids, local_coords, uv_virtual,
                    {"shape_encoder": shape_stats, "texture_encoder": texture_stats, **mapping_stats},
                )
                context = PreparedContext(
                    context_id=item["context_id"],
                    angle=item["angle"],
                    angle_index=item["angle_index"],
                    tile_id=item["tile_id"],
                    source_box=item["source_box"],
                    virtual_box=virtual_box,
                    transform=item["transform"],
                    tile_image=item["tile_image"],
                    tile_dir=item["tile_dir"],
                    geometry=item["geometry"],
                    baseline_pbr=item["baseline_pbr"],
                    shape_full=shape_full,
                    texture_full=texture_full,
                    native_coords=shape_full.coords.clone(),
                    view=view,
                    master_ids=master_ids,
                    local_coords=local_coords,
                    uv_virtual=uv_virtual,
                    gaussian_weight=view.gaussian_weight,
                    shape_reference=shape_reference,
                    texture_reference=texture_reference,
                    target_points=item["geometry"].dual_vertices_world.clone(),
                    target_world_points=target_world_points.cpu(),
                    nearest_local_points=torch.zeros((master_ids.numel(), 3), dtype=torch.float32),
                    nearest_local_uv=torch.zeros((master_ids.numel(), 2), dtype=torch.float32),
                    visible=torch.zeros((master_ids.numel(),), dtype=torch.bool),
                    mapping_valid_global=valid_global,
                    shape_state=None,
                    texture_state=None,
                    condition_shape=None,
                    condition_texture=None,
                    support_stats={
                        "status": "active",
                        "shape_encoder": shape_stats,
                        "texture_encoder": texture_stats,
                        "mapping": mapping_stats,
                        "local_ovoxel_count": int(item["geometry"].coords.shape[0]),
                    },
                )
                encoded_contexts.append(context)
                preparation_rows.append({
                    "context_id": item["context_id"],
                    "angle": item["angle"],
                    "tile_id": item["tile_id"],
                    "source_box": item["source_box"],
                    "virtual_box": virtual_box,
                    "status": "active",
                    "master_ids": master_ids,
                    "local_coords": local_coords,
                    "uv_virtual": uv_virtual,
                    "gaussian_weight": view.gaussian_weight,
                    "mapping_stats": mapping_stats,
                })
            else:
                # This row is handled after support construction when the
                # first-view support was not available as a cache.
                item["shape_full"] = shape_full
                item["texture_full"] = texture_full
                item["shape_stats"] = shape_stats
                item["texture_stats"] = texture_stats
                item["target_world_points"] = target_world_points.cpu()
                if support is not None:
                    preparation_rows.append({
                        "context_id": item["context_id"],
                        "angle": item["angle"],
                        "tile_id": item["tile_id"],
                        "source_box": item["source_box"],
                        "virtual_box": item["virtual_box"],
                        "status": "empty_mapping",
                        "master_ids": master_ids,
                        "local_coords": local_coords,
                        "uv_virtual": uv_virtual,
                        "gaussian_weight": torch.empty((0,), dtype=torch.float32),
                        "mapping_stats": mapping_stats,
                    })
    shape_encoder.cpu()
    pbr_encoder.cpu()
    del shape_encoder
    _empty_cuda_cache()

    if support is None:
        native_by_tile: Dict[int, torch.Tensor] = {}
        for item in pending:
            if item["angle"] == 0 and "shape_full" in item:
                native_by_tile[item["tile_id"]] = item["shape_full"].coords
        support = first_view_route._build_master_support(
            native_by_tile,
            first_transforms,
            camera,
            sigma_pixels=SIGMA_PIXELS,
        )
        # The support builder derives master q/UV rows from first-owner
        # activations, then reprojects overlap rows into every tile.  A local
        # encoder may legitimately omit an overlap coordinate, so rebuild the
        # first-view tile mappings against the actual current native C64
        # supports before gathering endpoint features.  This preserves the
        # global master IDs and 2-D first-owner policy while never fabricating
        # a feature value at a missing local position.
        rebuilt_first_views: Dict[int, first_view_route.TileView] = {}
        for item in pending:
            if item["angle"] != 0:
                continue
            master_ids, local_coords, uv_virtual, _, mapping_stats = _map_master_to_context(
                master_q_world=support.master_q_global,
                native_coords=item["shape_full"].coords,
                angle=0,
                transform=item["transform"],
                camera=camera,
                virtual_box=item["virtual_box"],
            )
            if master_ids.numel():
                rebuilt_first_views[item["tile_id"]] = _make_local_view(
                    item["context_id"], item["transform"], item["virtual_box"],
                    master_ids, local_coords, uv_virtual,
                    {"source": "current first-view native C64 support", **mapping_stats},
                )
        if not rebuilt_first_views:
            raise RuntimeError("current first-view encoder positions produced no support views")
        support.tile_views = rebuilt_first_views
        support.tile_stats = {key: dict(value.stats) for key, value in rebuilt_first_views.items()}
        coverage = torch.zeros((support.master_q_global.shape[0],), dtype=torch.int32)
        for view in rebuilt_first_views.values():
            coverage.index_add_(0, view.master_ids, torch.ones_like(view.master_ids, dtype=torch.int32))
        if bool((coverage <= 0).any()):
            missing = torch.where(coverage <= 0)[0][:16].tolist()
            raise RuntimeError(
                "current first-view native supports do not cover immutable master rows; "
                f"missing examples={missing}"
            )
        encoded_contexts.clear()
        preparation_rows = [row for row in preparation_rows if row.get("angle") != 0]
        # Recreate all context mappings from the now immutable support.  The
        # endpoint values remain in ``pending`` and are gathered below.
        for item in pending:
            if item["angle"] == 0 and item["tile_id"] not in support.tile_views:
                continue
            shape_full = item["shape_full"]
            texture_full = item["texture_full"]
            if item["angle"] == 0:
                source_view = support.tile_views[item["tile_id"]]
                master_ids, local_coords, uv_virtual = source_view.master_ids, source_view.local_coords, source_view.master_uv_4096
                mapping_stats = {"source": "new first-view first-owner support"}
            else:
                master_ids, local_coords, uv_virtual, _, mapping_stats = _map_master_to_context(
                    master_q_world=support.master_q_global,
                    native_coords=shape_full.coords,
                    angle=item["angle"],
                    transform=item["transform"],
                    camera=camera,
                    virtual_box=item["virtual_box"],
                )
            if not master_ids.numel():
                preparation_rows.append({"context_id": item["context_id"], "angle": item["angle"], "tile_id": item["tile_id"], "source_box": item["source_box"], "virtual_box": item["virtual_box"], "status": "empty_mapping", "master_ids": master_ids, "local_coords": local_coords, "uv_virtual": uv_virtual, "gaussian_weight": torch.empty((0,), dtype=torch.float32), "mapping_stats": mapping_stats})
                continue
            view = _make_local_view(item["context_id"], item["transform"], item["virtual_box"], master_ids, local_coords, uv_virtual, mapping_stats)
            shape_reference = _gather_coords(shape_full, local_coords, f"context {item['context_id']} shape baseline")
            texture_reference = _gather_coords(texture_full, local_coords, f"context {item['context_id']} texture baseline")
            rotation = rotations[item["angle"]]
            local_q = item["geometry"].dual_vertices_world * (2.0 * float(item["transform"].mesh_scale))
            q_view, _ = core._local_q_to_global_q(local_q, global_camera=camera, transform=item["transform"])
            target_world_points = _view_to_world_q(q_view, rotation) / (2.0 * float(camera["mesh_scale"]))
            encoded_contexts.append(PreparedContext(
                context_id=item["context_id"], angle=item["angle"], angle_index=item["angle_index"], tile_id=item["tile_id"],
                source_box=item["source_box"], virtual_box=item["virtual_box"], transform=item["transform"], tile_image=item["tile_image"], tile_dir=item["tile_dir"], geometry=item["geometry"], baseline_pbr=item["baseline_pbr"], shape_full=shape_full, texture_full=texture_full, native_coords=shape_full.coords.clone(), view=view, master_ids=master_ids, local_coords=local_coords, uv_virtual=uv_virtual, gaussian_weight=view.gaussian_weight, shape_reference=shape_reference, texture_reference=texture_reference, target_points=item["geometry"].dual_vertices_world.clone(), target_world_points=target_world_points.cpu(), nearest_local_points=torch.zeros((master_ids.numel(), 3), dtype=torch.float32), nearest_local_uv=torch.zeros((master_ids.numel(), 2), dtype=torch.float32), visible=torch.zeros((master_ids.numel(),), dtype=torch.bool), mapping_valid_global=torch.zeros((support.master_q_global.shape[0],), dtype=torch.bool), shape_state=None, texture_state=None, condition_shape=None, condition_texture=None, support_stats={"status": "active", "mapping": mapping_stats, "local_ovoxel_count": int(item["geometry"].coords.shape[0]), "shape_encoder": item["shape_stats"], "texture_encoder": item["texture_stats"]},
            ))
            encoded_contexts[-1].mapping_valid_global[master_ids] = True
            preparation_rows.append({"context_id": item["context_id"], "angle": item["angle"], "tile_id": item["tile_id"], "source_box": item["source_box"], "virtual_box": item["virtual_box"], "status": "active", "master_ids": master_ids, "local_coords": local_coords, "uv_virtual": uv_virtual, "gaussian_weight": view.gaussian_weight, "mapping_stats": mapping_stats})

    if not encoded_contexts:
        raise RuntimeError("support mapping left no active contexts")
    support_hash = _save_master_support(support, output_dir, first_transforms)
    _save_context_mapping(preparation_rows, output_dir, int(support.master_q_global.shape[0]))
    _atomic_json(output_dir / "contexts" / "context_summary.json", {"format": FORMAT, "active_contexts": len(encoded_contexts), "contexts": [c.support_stats | {"context_id": c.context_id, "angle": c.angle, "tile_id": c.tile_id, "master_rows": int(c.master_ids.numel())} for c in encoded_contexts]})
    return support, sorted(encoded_contexts, key=lambda c: c.context_id), {"support_sha256": support_hash, "preparation_rows": len(preparation_rows)}, pbr_encoder


def _runtime(device: torch.device, physical_device: int) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return {
        "physical_cuda_device_requested": int(physical_device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "current_device": int(torch.cuda.current_device()),
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "total_memory_bytes": int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory),
        "free_memory_bytes": int(torch.cuda.mem_get_info(torch.cuda.current_device())[0]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
    }


def _file_identity(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat() if path.exists() else None
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size": int(stat.st_size) if stat is not None else None,
        "mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
    }


def _write_context_metadata(contexts: Sequence[PreparedContext], output_dir: Path) -> None:
    for context in contexts:
        _atomic_json(
            context.tile_dir / "tile_camera.json",
            first_view_route._jsonable(context.transform.__dict__),
        )
        _atomic_json(
            context.tile_dir / "support.json",
            {
                "format": FORMAT,
                "context_id": context.context_id,
                "angle": context.angle,
                "tile_id": context.tile_id,
                "source_box": list(context.source_box),
                "virtual_box": list(context.virtual_box),
                "master_rows": int(context.master_ids.numel()),
                "local_native_rows": int(context.native_coords.shape[0]),
                "visible_rows": int(context.visible.sum()),
                "support": context.support_stats,
                "master_id_only": True,
            },
        )
    _atomic_json(
        output_dir / "support" / "frozen_visibility_contexts.json",
        {
            "format": FORMAT,
            "context_count": len(contexts),
            "visibility_changes_during_flow": False,
            "contexts": [
                {
                    "context_id": c.context_id,
                    "angle": c.angle,
                    "tile_id": c.tile_id,
                    "mapping_rows": int(c.master_ids.numel()),
                    "visible_rows": int(c.visible.sum()),
                }
                for c in contexts
            ],
        },
    )


def _final_decode_and_render(
    *,
    support: first_view_route.MasterSupport,
    pipeline: Any,
    shape_global: torch.Tensor,
    texture_global: torch.Tensor,
    baseline: MeshWithVoxel,
    canonical: Mapping[str, Any],
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    decode_batch_size: int,
    render: bool,
) -> Dict[str, Any]:
    # The shared decoder performs only canonical first-view local->world
    # decoding and fixed 2-D face ownership.  It never sees attached-view
    # contexts and therefore cannot create or reorder master support rows.
    first_view_route.FORMAT = FORMAT
    decoded = first_view_route._decode_and_merge(
        pipeline=pipeline,
        support=support,
        shape_features=shape_global,
        texture_features=texture_global,
        camera=camera,
        output_dir=output_dir,
        device=device,
        decode_batch_size=decode_batch_size,
    )
    result: Dict[str, Any] = {
        "vertices": int(decoded["vertices"]),
        "faces": int(decoded["faces"]),
        "decoded_tiles": decoded.get("decoded_tiles", []),
        "mesh_decode": "official local C1024 decoder; first-view 2-D Gaussian face ownership",
    }
    if render:
        reference = np.asarray(canonical["image_4096"].convert("RGB"), dtype=np.float32) / 255.0
        foreground = np.asarray(canonical["foreground_mask_4096"].convert("L"), dtype=np.float32) / 255.0
        render_summary = first_view_route._render_one(
            decoded["vertex_mesh"],
            camera,
            output_dir / "final",
            device,
            "final",
            CANONICAL_SIZE,
            reference,
            foreground,
        )
        result["render"] = render_summary
        try:
            result["native_4096_validation"] = validation_route.validate_native_4096_outputs(
                output_dir, require_all=True
            )
        except Exception as exc:
            raise RuntimeError(f"final native 4096 artifact validation failed: {exc}") from exc
        first_view_route._compute_global_metrics(
            canonical["image_4096"],
            canonical["foreground_mask_4096"],
            output_dir,
            {"final": render_summary},
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--multiview-image", type=Path, default=DEFAULT_MULTIVIEW)
    parser.add_argument("--baseline-mesh", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--first-view-support", type=Path, default=DEFAULT_FIRST_SUPPORT)
    parser.add_argument("--prepared-context-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--pbr-encoder", type=Path, default=DEFAULT_PBR_ENCODER)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--shape-seed", type=int, default=20260823)
    parser.add_argument("--texture-seed", type=int, default=20260824)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--initial-context-encode-batch-size", type=int, default=PBR_ENCODE_BATCH_SIZE)
    parser.add_argument("--flow-batch-size", type=int, default=FLOW_BATCH_SIZE)
    parser.add_argument("--decode-batch-size", type=int, default=DECODE_BATCH_SIZE)
    parser.add_argument("--pbr-encode-batch-size", type=int, default=PBR_ENCODE_BATCH_SIZE)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--nearest-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-step-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.flow_batch_size != FLOW_BATCH_SIZE or args.decode_batch_size != DECODE_BATCH_SIZE or args.pbr_encode_batch_size != PBR_ENCODE_BATCH_SIZE:
        raise ValueError("Codex2 fixes flow/decode/pbr_encode batches to 44/12/13")
    if args.steps is not None and int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    for path in (args.image, args.multiview_image, args.baseline_mesh, args.camera):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(path)
    for path in (args.model_path, args.shape_encoder, args.pbr_encoder):
        candidate = Path(path).expanduser()
        if not candidate.exists() and not (
            Path(f"{candidate}.json").is_file()
            and Path(f"{candidate}.safetensors").is_file()
        ):
            raise FileNotFoundError(path)
    if not Path(args.first_view_support).expanduser().exists():
        # The documented cache is optional: the implementation can build the
        # first-view support after the one-time C64 activation encodes.
        print(f"[support] optional source cache not found: {args.first_view_support}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    requested = int(args.cuda_device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if entries != [str(requested)]:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES must expose only physical CUDA {requested}, got {visible!r}"
            )
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        if requested >= torch.cuda.device_count():
            raise RuntimeError(f"physical CUDA {requested} is unavailable")
        torch.cuda.set_device(requested)
        device = torch.device(f"cuda:{requested}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(device, requested)
    _seed(int(args.seed))
    camera = _load_camera(args.camera)
    baseline = _load_mesh(args.baseline_mesh)
    if baseline.faces.numel() == 0 or baseline.vertices.numel() == 0:
        raise RuntimeError("baseline mesh is empty")
    multiview_images = _load_views(args.multiview_image)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=bool(args.low_vram))
    source_image = Image.open(args.image)
    canonical = pipeline.preprocess_canonical_images(source_image)
    _save_canonical(canonical, output_dir)
    for angle, image in multiview_images.items():
        image.save(output_dir / "inputs" / f"view_{angle:03d}.png")
    _atomic_json(output_dir / "global_camera.json", camera)
    _atomic_json(output_dir / "runtime.json", runtime)
    _atomic_json(
        output_dir / "config.json",
        {
            "format": FORMAT,
            "input": str(args.image.resolve()),
            "multiview_input": str(args.multiview_image.resolve()),
            "baseline_mesh": _file_identity(args.baseline_mesh),
            "camera": camera,
            "args": vars(args),
            "runtime": runtime,
            "fixed_layout": {
                "first_view": {"size": CANONICAL_SIZE, "tile": FIRST_TILE_SIZE, "stride": FIRST_TILE_STRIDE, "grid": "7x7"},
                "attached_view": {"size": VIEW_SIZE, "tile": VIEW_TILE_SIZE, "stride": VIEW_TILE_STRIDE, "model_tile": MODEL_TILE_SIZE, "grid": "7x7"},
                "angles": list(ANGLES),
            },
            "batch_profile": {"flow": FLOW_BATCH_SIZE, "decode": DECODE_BATCH_SIZE, "pbr_encode": PBR_ENCODE_BATCH_SIZE, "initial_context_encode": int(args.initial_context_encode_batch_size)},
            "support_policy": "first-view-only dense master; attached views mapping-only",
        },
    )
    support, contexts, prep_meta, pbr_encoder = _prepare_contexts(
        args=args,
        pipeline=pipeline,
        baseline=baseline,
        camera=camera,
        canonical=canonical,
        multiview_images=multiview_images,
        output_dir=output_dir,
        device=device,
    )
    master_q_world = support.master_q_global.float().cpu().contiguous()
    master_count = int(master_q_world.shape[0])
    nearest = _nearest_triangle_mapping(
        baseline,
        master_q_world,
        output_dir / "support" / "master_nearest_triangle.pt",
        face_chunk_size=int(args.nearest_face_chunk_size),
    )
    _atomic_save(
        output_dir / "support" / "master_nearest_triangle.pt",
        {
            "format": FORMAT,
            "baseline_hash": _hash_many({"vertices": baseline.vertices, "faces": baseline.faces}),
            "support_hash": _tensor_hash(master_q_world),
            "nearest_face_id": nearest["nearest_face_id"],
            "nearest_point": nearest["nearest_point"],
            "nearest_bary": nearest["nearest_bary"],
            "face_distance": nearest["face_distance"],
            "master_q_world": master_q_world,
        },
    )
    master_payload = torch.load(
        output_dir / "support" / "master_support.pt", map_location="cpu", weights_only=False
    )
    master_payload.update(
        {
            "baseline_nearest_face_id": nearest["nearest_face_id"],
            "baseline_nearest_point": nearest["nearest_point"],
            "baseline_nearest_bary": nearest["nearest_bary"],
            "baseline_face_distance": nearest["face_distance"],
        }
    )
    _atomic_save(output_dir / "support" / "master_support.pt", master_payload)
    _attach_nearest_context_points(contexts, nearest["nearest_point"], camera)
    visibility_payload = _build_visibility(
        baseline=baseline,
        camera=camera,
        contexts=contexts,
        master_q_world=master_q_world,
        nearest=nearest,
        output_dir=output_dir,
        render_face_chunk_size=int(args.render_face_chunk_size),
    )
    _write_context_metadata(contexts, output_dir)
    active_views = {context.context_id: context.view for context in contexts}
    shape_conditions = _pack_conditions(
        contexts, pipeline, output_dir, "shape", device, PBR_ENCODE_BATCH_SIZE
    )
    texture_conditions = _pack_conditions(
        contexts, pipeline, output_dir, "texture", device, TEXTURE_CONDITION_BATCH_SIZE
    )
    for context in contexts:
        if context.context_id not in shape_conditions or context.context_id not in texture_conditions:
            raise RuntimeError(f"context {context.context_id}: missing shape/texture condition")
        context.condition_shape = shape_conditions[context.context_id]
        context.condition_texture = texture_conditions[context.context_id]
    baseline_field = core._make_attribute_query_mesh(baseline, device)
    baseline_pbr_global = _load_or_query_baseline_pbr(
        baseline_field=baseline_field,
        nearest_point=nearest["nearest_point"],
        output_path=output_dir / "baseline" / "pbr_at_nearest_master.pt",
        device=device,
        query_chunk_size=int(args.query_chunk_size),
    )
    shape_fallback, texture_fallback, fallback_summary = _build_baseline_endpoints(
        contexts=contexts,
        baseline_pbr_global=baseline_pbr_global,
        output_dir=output_dir,
        master_count=master_count,
    )
    _atomic_json(
        output_dir / "baseline" / "cache_metadata.json",
        {
            "format": FORMAT,
            "baseline_geometry_hash": _hash_many({"vertices": baseline.vertices, "faces": baseline.faces}),
            "master_support_hash": prep_meta["support_sha256"],
            "context_mapping_hash": _tensor_hash(visibility_payload["mapping_valid"]),
            "nearest_triangle_hash": _hash_many(nearest),
            "face_visibility_hash": _hash_many({"visible": visibility_payload["visible"], "faces": visibility_payload["face_visible_ids"]}),
            "shape_encoder": _file_identity(args.shape_encoder),
            "pbr_encoder": _file_identity(args.pbr_encoder),
            "shape_normalization": pipeline.shape_slat_normalization,
            "texture_normalization": pipeline.tex_slat_normalization,
            "fallback_summary": fallback_summary,
        },
    )
    shape_global, shape_summary = _run_shape_flow(
        contexts=contexts,
        pipeline=pipeline,
        output_dir=output_dir,
        device=device,
        seed=int(args.shape_seed),
        steps_override=args.steps,
        resume=bool(args.resume),
        save_step_tensors=bool(args.save_step_tensors),
        shape_fallback=shape_fallback,
        master_count=master_count,
    )
    texture_global, texture_summary = _run_texture_flow(
        contexts=contexts,
        shape_global_final=shape_global,
        texture_fallback=texture_fallback,
        baseline_pbr_global=baseline_pbr_global,
        pipeline=pipeline,
        pbr_encoder=pbr_encoder,
        output_dir=output_dir,
        device=device,
        seed=int(args.texture_seed),
        steps_override=args.steps,
        resume=bool(args.resume),
        save_step_tensors=bool(args.save_step_tensors),
        query_chunk_size=int(args.query_chunk_size),
        master_count=master_count,
    )
    final_summary = _final_decode_and_render(
        support=support,
        pipeline=pipeline,
        shape_global=shape_global,
        texture_global=texture_global,
        baseline=baseline,
        canonical=canonical,
        camera=camera,
        output_dir=output_dir,
        device=device,
        decode_batch_size=DECODE_BATCH_SIZE,
        render=bool(args.render),
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "runtime": runtime,
        "camera": camera,
        "master_support": {
            "count": master_count,
            "source": str(args.first_view_support.resolve()) if args.first_view_support.is_dir() else "built_from_first_view_only",
            "support_sha256": prep_meta["support_sha256"],
            "attached_views_created_support": False,
        },
        "contexts": {
            "active": len(contexts),
            "angles": list(ANGLES),
            "per_context_face_visibility": True,
            "mapping_only_attached_views": True,
        },
        "batch_profile": {
            "flow": FLOW_BATCH_SIZE,
            "decode": DECODE_BATCH_SIZE,
            "pbr_encode": PBR_ENCODE_BATCH_SIZE,
            "initial_context_encode": int(args.initial_context_encode_batch_size),
            "tail_batches_are_real": True,
            "serial_fallback": False,
        },
        "baseline_fallback": fallback_summary,
        "shape_flow": shape_summary,
        "texture_flow": texture_summary,
        "final": final_summary,
        "artifacts": {
            "master_support": str(output_dir / "support" / "master_support.pt"),
            "context_mapping": str(output_dir / "support" / "context_mapping.pt"),
            "nearest_triangle": str(output_dir / "support" / "master_nearest_triangle.pt"),
            "face_visibility": str(output_dir / "support" / "face_visibility_per_context.pt"),
            "frozen_visibility": str(output_dir / "support" / "frozen_visibility.pt"),
            "shape_global_final": str(output_dir / "final" / "shape_global_final.pt"),
            "texture_global_final": str(output_dir / "final" / "texture_global_final.pt"),
            "final_vertex_mesh": str(output_dir / "final" / "final_per_vertex_pbr_mesh.pt"),
            "final_face_mesh": str(output_dir / "final" / "final_per_face_pbr_mesh.pt"),
            "render_rgb_4096": str(output_dir / "final" / "final_render_rgb_4096.png"),
            "render_alpha_4096": str(output_dir / "final" / "final_render_alpha_4096.png"),
            "depth_4096": str(output_dir / "final" / "final_depth_4096.pt"),
        },
        "hard_assertions": {
            "single_first_view_master": True,
            "master_order_shape_texture_identical": True,
            "triangle_nearest_not_vertex": True,
            "per_context_face_visibility": True,
            "visibility_frozen": True,
            "one_current_prediction_per_context_timestep": True,
            "visible_gaussian_fallback_endpoint_rule": True,
            "independent_local_noise": True,
            "global_velocity_average": False,
            "finite_gate_for_pbr_query": True,
            "native_4096_final": bool(args.render),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"[done] {summary['status']} output={Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
