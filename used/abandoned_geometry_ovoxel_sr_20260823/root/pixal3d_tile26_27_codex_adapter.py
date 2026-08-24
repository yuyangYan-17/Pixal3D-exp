#!/usr/bin/env python3
"""Prepare the correct 2-D tile 26/27 caches for the Codex P0 merge.

The saved ``tile26_27_cuda4_correct_input_stride512`` directory contains the
post-flow local C64 SLATs, but not the decoder raw mesh/provenance required by
``pixal3d_ovoxel_global_mesh_revoxelize_merge.py``.  This adapter therefore:

1. decodes each saved local SLAT once with the native shape decoder;
2. applies the legacy tile-camera inverse exactly once *upstream*;
3. applies the old nearest-center 2-D ownership boxes to retain tile faces;
4. writes a provenance-complete, already-global-centered raw mesh payload and
   a manifest understood by the Codex global re-voxelizer.

The Codex merge receives only the pre-placed global mesh.  It does not perform
camera projection, depth backprojection, or image-plane ownership.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch

OVOXEL_SOURCE = Path("/home/nvme04/yyyan/TRELLIS.2/o-voxel")
if OVOXEL_SOURCE.is_dir():
    sys.path.insert(0, str(OVOXEL_SOURCE))

from o_voxel.convert import flexible_dual_grid_to_mesh


TILE_IDS = (26, 27)
DEFAULT_INPUT = Path("outputs/tile26_27_cuda4_correct_input_stride512")
DEFAULT_OUTPUT = DEFAULT_INPUT / "codex_adapter_raw"
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
GLOBAL_CAMERA = {
    "camera_angle_x": 0.517371749106554,
    "distance": 1.889538288116455,
    "mesh_scale": 1.0,
}


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _focal_pixels(angle: float, resolution: int) -> float:
    return float(resolution) / (2.0 * math.tan(float(angle) / 2.0))


def _camera_inverse_vertices(
    vertices_local: np.ndarray,
    tile_camera: Mapping[str, Any],
    chunk_size: int,
) -> np.ndarray:
    """Map local decoder object points to global object points once.

    This is intentionally kept in the input adapter.  The downstream Codex
    merge consumes the result under ``global_centered`` and has no camera path.
    """
    vertices_local = np.asarray(vertices_local, dtype=np.float32)
    if vertices_local.ndim != 2 or vertices_local.shape[1] != 3:
        raise ValueError(f"local vertices must be [N,3], got {vertices_local.shape}")
    tile_scale = float(tile_camera.get("mesh_scale", 1.0))
    global_scale = float(GLOBAL_CAMERA["mesh_scale"])
    tile_distance = float(tile_camera["distance"])
    global_distance = float(GLOBAL_CAMERA["distance"])
    fx = float(tile_camera["fx"])
    fy = float(tile_camera["fy"])
    cx = float(tile_camera["cx"])
    cy = float(tile_camera["cy"])
    full_fx = float(tile_camera["full_fx_4096"])
    full_fy = float(tile_camera["full_fy_4096"])
    full_cx = 2048.0
    full_cy = 2048.0
    x0, y0, _, _ = (float(v) for v in tile_camera["box"])
    crop_x = float(tile_camera.get("crop_to_output_scale_x", 1.0))
    crop_y = float(tile_camera.get("crop_to_output_scale_y", 1.0))
    if min(tile_scale, global_scale, tile_distance, global_distance, fx, fy, full_fx, full_fy) <= 0:
        raise ValueError("tile camera contains a non-positive scale, depth, or focal length")

    result = np.empty_like(vertices_local, dtype=np.float32)
    for start in range(0, vertices_local.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), vertices_local.shape[0])
        p = vertices_local[start:stop].astype(np.float64, copy=False)
        # q_local = p_local * (2 * tile_mesh_scale), and the centered local
        # camera point is [-distance] + q/(2*mesh_scale).
        local_depth = tile_distance - p[:, 2]
        if not np.isfinite(local_depth).all() or bool((local_depth <= 0).any()):
            raise ValueError(f"local decoder vertices contain non-positive depth in rows {start}:{stop}")
        uv_tile_x = fx * p[:, 0] / local_depth + cx
        uv_tile_y = -fy * p[:, 1] / local_depth + cy
        uv_full_x = uv_tile_x / crop_x + x0
        uv_full_y = uv_tile_y / crop_y + y0
        qz = p[:, 2] * (2.0 * tile_scale)
        global_depth = global_distance - qz / (2.0 * global_scale)
        if not np.isfinite(global_depth).all() or bool((global_depth <= 0).any()):
            raise ValueError(f"inverse global depth is invalid in rows {start}:{stop}")
        result[start:stop, 0] = ((uv_full_x - full_cx) * global_depth / full_fx).astype(np.float32)
        result[start:stop, 1] = (-(uv_full_y - full_cy) * global_depth / full_fy).astype(np.float32)
        # q_global_z is q_local_z; convert it back to object space.
        result[start:stop, 2] = (qz / (2.0 * global_scale)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("globalized decoder vertices contain NaN/Inf")
    return result


def _ownership_box(tile_camera: Mapping[str, Any], tile_stride: int = 512) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (int(v) for v in tile_camera["box"])
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    half = float(tile_stride) * 0.5
    left = 0.0 if x0 == 0 else cx - half
    right = 4096.0 if x1 == 4096 else cx + half
    top = 0.0 if y0 == 0 else cy - half
    bottom = 4096.0 if y1 == 4096 else cy + half
    return left, top, right, bottom


def _owned_face_mask(
    vertices_global: np.ndarray,
    faces: np.ndarray,
    tile_camera: Mapping[str, Any],
    chunk_size: int,
) -> np.ndarray:
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh faces must be [F,3], got {faces.shape}")
    n_vertices = int(vertices_global.shape[0])
    valid = ((faces >= 0) & (faces < n_vertices)).all(axis=1)
    keep = np.zeros((faces.shape[0],), dtype=bool)
    left, top, right, bottom = _ownership_box(tile_camera)
    full_fx = float(tile_camera["full_fx_4096"])
    full_fy = float(tile_camera["full_fy_4096"])
    global_distance = float(GLOBAL_CAMERA["distance"])
    for start in range(0, faces.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), faces.shape[0])
        local_faces = faces[start:stop]
        local_valid = valid[start:stop]
        if not bool(local_valid.any()):
            continue
        safe = np.clip(local_faces, 0, max(n_vertices - 1, 0)).astype(np.int64, copy=False)
        tri = vertices_global[safe].astype(np.float64, copy=False)
        centroid = tri.mean(axis=1)
        depth = global_distance - centroid[:, 2]
        uv_x = full_fx * centroid[:, 0] / np.maximum(depth, 1e-12) + 2048.0
        uv_y = -full_fy * centroid[:, 1] / np.maximum(depth, 1e-12) + 2048.0
        local_keep = (
            local_valid
            & np.isfinite(centroid).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0)
            & (uv_x >= left) & (uv_x < right)
            & (uv_y >= top) & (uv_y < bottom)
        )
        keep[start:stop] = local_keep
        if start == 0 or stop == faces.shape[0] or (start // max(1, int(chunk_size))) % 20 == 0:
            print(
                f"    ownership faces {stop:,}/{faces.shape[0]:,}; kept={int(keep[:stop].sum()):,}",
                flush=True,
            )
    return keep


def _subset_face_provenance(
    provenance: Mapping[str, Any],
    keep: np.ndarray,
    source_face_ids: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    face_count = int(keep.shape[0])
    for key, value in provenance.items():
        arr = value.detach().cpu() if torch.is_tensor(value) else value
        shape = getattr(arr, "shape", None)
        # quad_indices is a quad-level table and must remain aligned to the
        # original raw coords; explicit source_quad_id below handles the crop.
        if key != "quad_indices" and shape is not None and len(shape) >= 1 and int(shape[0]) == face_count:
            if torch.is_tensor(arr):
                arr = arr[torch.from_numpy(keep)]
            else:
                arr = np.asarray(arr)[keep]
        result[key] = arr
    qcount = 0
    if "quad_indices" in result:
        qindices = result["quad_indices"]
        qcount = int(qindices.shape[0]) if getattr(qindices, "ndim", 0) == 2 else 0
    if qcount * 2 == face_count:
        result["source_quad_id"] = torch.from_numpy((source_face_ids // 2).astype(np.int64, copy=False))
    else:
        result["source_quad_id"] = torch.from_numpy(source_face_ids.astype(np.int64, copy=False))
    return result


def _decode_raw_tile(
    pipeline: Any,
    cache_path: Path,
    device: torch.device,
    tile_id: int,
) -> Mapping[str, Any]:
    from pixal3d.modules.sparse import SparseTensor

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    coords = cache["local_coords64"].to(device=device, dtype=torch.int32)
    feats = cache["local_shape_denorm_feats"].to(device=device)
    sparse = SparseTensor(feats=feats, coords=coords)
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.set_resolution(1024)
    decoder.to(device)
    print(f"[tile {tile_id}] decoding raw mesh from C64={coords.shape[0]:,}", flush=True)
    with torch.no_grad():
        # Return only sparse decoder fields first. Keeping the VAE activations
        # resident while native provenance is materialized exceeds 80 GB for
        # these dense tiles.
        decoded = decoder(sparse, return_subs=False, return_ovoxel_fields=True)
    decoder.cpu()
    if not isinstance(decoded, Mapping) or len(decoded.get("ovoxel_fields", [])) != 1:
        raise RuntimeError(f"tile {tile_id} decoder O-Voxel field result is malformed")
    fields = _cpu(decoded["ovoxel_fields"][0])
    del decoded, sparse, coords, feats, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    native_coords = fields["coords"].to(device=device, dtype=torch.int32).contiguous()
    native_dual = fields["dual_vertices"].to(device=device, dtype=torch.float32).contiguous()
    native_intersected = fields["intersected"].to(device=device).contiguous()
    native_split = fields["quad_lerp"].to(device=device, dtype=torch.float32).contiguous()
    print(f"[tile {tile_id}] native mesh/provenance extraction from fields", flush=True)
    with torch.no_grad():
        mesh_vertices, mesh_faces, provenance = flexible_dual_grid_to_mesh(
            native_coords,
            native_dual,
            native_intersected,
            native_split,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=1024,
            train=False,
            return_provenance=True,
        )
    raw = {
        "coords": native_coords.detach().cpu(),
        "dual_vertices": native_dual.detach().cpu(),
        "intersected": native_intersected.detach().cpu(),
        "intersected_logits": fields["intersected_logits"],
        "quad_lerp": native_split.detach().cpu(),
        "mesh_vertices": mesh_vertices.detach().cpu(),
        "mesh_faces": mesh_faces.detach().cpu(),
        "provenance": _cpu(provenance),
    }
    del native_coords, native_dual, native_intersected, native_split
    del mesh_vertices, mesh_faces, provenance, fields
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw


def _prepare_one_tile(
    pipeline: Any,
    input_dir: Path,
    output_dir: Path,
    tile_id: int,
    face_chunk_size: int,
    vertex_chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    tile_dir = input_dir / "tiles" / f"tile_{tile_id}"
    cache_path = tile_dir / "velocity_synced_local_shape_texture_slat.pt"
    camera_path = tile_dir / "tile_camera.json"
    if not cache_path.is_file() or not camera_path.is_file():
        raise FileNotFoundError(f"tile {tile_id} cache/camera missing under {tile_dir}")
    tile_camera = _load_json(camera_path)
    raw = _decode_raw_tile(pipeline, cache_path, device, tile_id)
    vertices_local = raw["mesh_vertices"].detach().cpu().numpy().astype(np.float32, copy=False)
    faces_all = raw["mesh_faces"].detach().cpu().numpy().astype(np.int64, copy=False)
    print(
        f"[tile {tile_id}] decoder mesh vertices={vertices_local.shape[0]:,} faces={faces_all.shape[0]:,}",
        flush=True,
    )
    vertices_global = _camera_inverse_vertices(vertices_local, tile_camera, vertex_chunk_size)
    keep = _owned_face_mask(vertices_global, faces_all, tile_camera, face_chunk_size)
    retained = np.flatnonzero(keep).astype(np.int64, copy=False)
    if retained.size == 0:
        raise RuntimeError(f"tile {tile_id} ownership filter kept no triangles")
    selected_faces_source = faces_all[retained]
    used_vertices, inverse = np.unique(selected_faces_source.reshape(-1), return_inverse=True)
    compact_faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
    compact_vertices = vertices_global[used_vertices].astype(np.float32, copy=False)
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(f"tile {tile_id} raw decoder provenance is missing")
    raw_out: MutableMapping[str, Any] = dict(raw)
    raw_out["mesh_vertices"] = torch.from_numpy(compact_vertices)
    raw_out["mesh_faces"] = torch.from_numpy(compact_faces)
    raw_out["provenance"] = _subset_face_provenance(provenance, keep, retained)
    # These fields are explicit adapter provenance, not inputs to topology.
    raw_out["adapter_coordinate_convention"] = "global_centered"
    raw_out["adapter_source_face_ids"] = torch.from_numpy(retained)
    payload = {
        "format": "pixal3d_tile26_27_codex_preplaced_raw_mesh_v1",
        "tile_id": int(tile_id),
        "raw_ovoxel": _cpu(raw_out),
        "adapter": {
            "stage": "upstream_2d_tile_camera_inverse_and_ownership",
            "camera_projection_calls_in_codex_merge": 0,
            "coordinate_convention": "global_centered",
            "tile_camera": tile_camera,
            "ownership_box_4096": list(_ownership_box(tile_camera)),
            "input_decoder_vertices": int(vertices_local.shape[0]),
            "input_decoder_faces": int(faces_all.shape[0]),
            "owned_faces": int(retained.size),
            "compact_vertices": int(compact_vertices.shape[0]),
            "faces_dropped_outside_ownership": int(faces_all.shape[0] - retained.size),
        },
    }
    output_path = output_dir / f"tile_{tile_id:03d}" / "shape_flow_and_raw_ovoxel.pt"
    print(f"[tile {tile_id}] saving compact global raw payload -> {output_path}", flush=True)
    _atomic_torch_save(output_path, payload)
    del raw, raw_out, vertices_local, vertices_global, faces_all, compact_vertices, compact_faces, selected_faces_source
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "tile_id": int(tile_id),
        "output": str(output_path.resolve()),
        "origin": [0, 0, 0],
        "input_decoder_vertices": int(payload["adapter"]["input_decoder_vertices"]),
        "input_decoder_faces": int(payload["adapter"]["input_decoder_faces"]),
        "owned_faces": int(payload["adapter"]["owned_faces"]),
        "compact_vertices": int(payload["adapter"]["compact_vertices"]),
        "ownership_box_4096": payload["adapter"]["ownership_box_4096"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tile-ids", default="26,27")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--face-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--vertex-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if int(args.cuda_device) != 4:
        raise ValueError("the adapter is hard-pinned to physical CUDA 4")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {None, "4"}:
        raise RuntimeError(f"run the adapter with physical CUDA 4; got CUDA_VISIBLE_DEVICES={visible!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decoder reconstruction")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_name(device) == "":
        raise RuntimeError("CUDA device name is unavailable")
    tile_ids = tuple(int(v.strip()) for v in str(args.tile_ids).split(",") if v.strip())
    if not tile_ids:
        raise ValueError("no tile ids selected")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from pixal3d_ovoxel_hermite_qef_sr import _init_local_shape_pipeline

    print(
        f"[adapter] physical CUDA 4 via logical cuda:0 ({torch.cuda.get_device_name(device)})",
        flush=True,
    )
    pipeline = _init_local_shape_pipeline(args.model_path.expanduser().resolve(), device, bool(args.low_vram))
    # The saved SLAT cache is already post-flow.  The image projector is not
    # used for this reconstruction and keeping it on GPU steals the headroom
    # needed by the native provenance-producing mesher.
    unused_projector = getattr(pipeline, "image_cond_model_shape_1024", None)
    if unused_projector is not None:
        unused_projector.cpu()
        pipeline.image_cond_model_shape_1024 = None
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    try:
        for tile_id in tile_ids:
            output_path = output_dir / f"tile_{tile_id:03d}" / "shape_flow_and_raw_ovoxel.pt"
            if output_path.is_file() and not args.force:
                print(f"[tile {tile_id}] existing adapter payload; reusing {output_path}", flush=True)
                cached = _load_json(output_path.with_suffix(".json")) if output_path.with_suffix(".json").is_file() else None
                if cached is None:
                    cached = {"tile_id": int(tile_id), "output": str(output_path.resolve()), "reused": True}
                rows.append(cached)
                continue
            rows.append(_prepare_one_tile(
                pipeline, input_dir, output_dir, tile_id,
                int(args.face_chunk_size), int(args.vertex_chunk_size), device,
            ))
            _atomic_json(output_path.with_suffix(".json"), rows[-1])
    finally:
        pipeline.cpu()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {
        "format": "pixal3d_ovoxel_global_mesh_revoxelize_tile_manifest_v1",
        "global_resolution": 4096,
        "tile_size": 1024,
        "tile_stride": 512,
        "source_directory": str(input_dir),
        "adapter": {
            "stage": "premerge_camera_inverse",
            "camera_projection_calls_in_global_merge": 0,
            "coordinate_convention": "global_centered",
            "note": "2-D tile camera inverse is performed only while materializing this input adapter; global merge is mesh-only.",
        },
        "tiles": [],
    }
    for row in rows:
        tile_id = int(row["tile_id"])
        tile_camera = _load_json(input_dir / "tiles" / f"tile_{tile_id}" / "tile_camera.json")
        manifest["tiles"].append({
            "tile_id": tile_id,
            "origin": [0, 0, 0],
            "size": 1024,
            "stride": 512,
            "raw_ovoxel": row["output"],
            "coordinate_convention": "global_centered",
            "boundary_band": 0.0,
            "contribution_weight": 1.0,
            "premerge_camera_inverse": True,
            "source_2d_tile_box": tile_camera["box"],
        })
    _atomic_json(output_dir / "tile_manifest.json", manifest)
    _atomic_json(output_dir / "adapter_summary.json", {"tiles": rows, "manifest": str((output_dir / "tile_manifest.json").resolve())})
    print(f"[adapter] manifest ready: {output_dir / 'tile_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
