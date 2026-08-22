#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Return decoded local texture meshes to global space and render them together.

The preceding experiment stores sparse SLat endpoints rather than decoded
meshes.  This utility decodes the same nine PBR variants, maps every local
decoded vertex through Pixal3D's official local->global camera transform,
removes tile-overlap faces with the repository stitcher, welds nearby global
vertices, and renders the resulting whole-object mesh with the official PBR
renderer.

The global-camera render is evaluated against ``canonical_1024.png``.  The
same global camera trajectory is then used for fixed multi-view renders, so
the aligned input view and the multi-view images share one coordinate frame.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_texture_pbr_degradation_experiment as experiment
import pixal3d_texture_pbr_multiview as multiview
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr
from pixal3d.utils import render_utils


VARIANT_ORDER = tuple(multiview.VARIANT_ORDER)
VIEW_LABELS = tuple(label for label, _, _ in multiview.FIXED_VIEWS)
CANONICAL_REFERENCE = "canonical_1024.png"


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _render_args(args: argparse.Namespace) -> SimpleNamespace:
    """Arguments consumed by the existing single-view renderer wrapper."""
    return SimpleNamespace(
        render_resolution=int(args.aligned_resolution),
        metric_resolution=int(args.metric_resolution),
        render_ssaa=int(args.aligned_ssaa),
        render_peel_layers=int(args.peel_layers),
        render_face_chunk_size=int(args.face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        envmap=str(args.envmap),
        lpips_net="vgg",
        skip_lpips=True,
    )


def _make_global_patch(
    *,
    tile_id: int,
    box: Sequence[int],
    local_vertices: torch.Tensor,
    local_faces: torch.Tensor,
    local_attrs: torch.Tensor,
    transform: Any,
    global_camera: Mapping[str, float],
    query_chunk_size: int,
) -> core.ReturnedTilePatch:
    """Map one decoded local mesh to global normalized object coordinates."""
    local_vertices = local_vertices.detach().to(dtype=torch.float32)
    local_faces = local_faces.detach().to(dtype=torch.int32)
    local_attrs = local_attrs.detach().to(dtype=torch.float32)
    global_vertices = experiment._map_local_to_global_chunked(
        local_vertices,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(query_chunk_size),
    ).detach().cpu().float()

    # Check the actual camera transform in both directions before the patch is
    # admitted to the global stitch.  The q coordinates are the native camera
    # convention used by the official Pixal3D transform helpers.
    q_local = local_vertices * (2.0 * float(transform.mesh_scale))
    q_global = global_vertices * (2.0 * float(global_camera["mesh_scale"]))
    q_local_roundtrip, _ = core._global_q_to_local_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
    )
    roundtrip = (q_local_roundtrip.cpu() - q_local.cpu()).abs()
    if not torch.isfinite(global_vertices).all():
        raise RuntimeError(f"tile {tile_id}: local->global produced non-finite vertices")
    if float(roundtrip.max().item()) > float(1e-4):
        raise RuntimeError(
            f"tile {tile_id}: local/global round-trip error too large: "
            f"{float(roundtrip.max().item()):.3e}"
        )
    if local_faces.numel() and int(local_faces.max().item()) >= int(global_vertices.shape[0]):
        raise RuntimeError(f"tile {tile_id}: decoded face index exceeds vertex count")
    if local_attrs.shape[0] != local_vertices.shape[0]:
        raise RuntimeError(
            f"tile {tile_id}: PBR vertex attrs {tuple(local_attrs.shape)} do not "
            f"match vertices {tuple(local_vertices.shape)}"
        )

    stats = {
        "tile_id": int(tile_id),
        "local_vertices": int(local_vertices.shape[0]),
        "local_faces": int(local_faces.shape[0]),
        "global_vertices": int(global_vertices.shape[0]),
        "global_faces": int(local_faces.shape[0]),
        "local_to_global_roundtrip_max_abs_q": float(roundtrip.max().item()),
        "local_to_global_roundtrip_mean_abs_q": float(roundtrip.mean().item()),
        "coordinate_space": "global normalized object space",
        "transform": "official core._local_q_to_global_q",
    }
    return core.ReturnedTilePatch(
        tile_id=int(tile_id),
        box=tuple(int(v) for v in box),
        vertices=global_vertices,
        faces=local_faces.cpu(),
        vertex_attrs=local_attrs.cpu(),
        stats=stats,
    )


@torch.no_grad()
def _decode_tile_payload(
    *,
    pipeline: Any,
    source_dir: Path,
    tile_id: int,
    row: Mapping[str, Any],
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Decode one tile once and cache global geometry plus all variant attrs."""
    tile_dir = source_dir / "tiles" / f"tile_{tile_id:02d}"
    endpoints_path = tile_dir / "endpoints.pt"
    if not endpoints_path.is_file():
        raise FileNotFoundError(endpoints_path)
    endpoints = _load_torch(endpoints_path)
    device = torch.device("cuda")
    shape_coords = endpoints["shape_coords"].to(device=device, dtype=torch.int32)
    shape_norm = SparseTensor(
        endpoints["shape_norm"].to(device=device, dtype=torch.float32),
        shape_coords,
    )
    shape_denorm = experiment._denormalize_slat(
        shape_norm, pipeline.shape_slat_normalization
    )
    texture_coords = endpoints["g_tex_coords"].to(device=device, dtype=torch.int32)
    g_norm = endpoints["g_tex_norm"].to(dtype=torch.float32)
    hr_norm = endpoints["hr_tex_norm"].to(dtype=torch.float32)
    transform = core.TileCameraTransform(**endpoints["transform"])

    def decode(label: str, latent: torch.Tensor):
        return multiview._decode_variant(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=latent,
            coords=texture_coords,
            label=label,
            query_chunk_size=int(args.query_chunk_size),
        )

    g_vertices, g_faces, g_attrs, g_stats = decode(
        f"tile_{tile_id:02d} G global stitch", g_norm
    )
    hr_vertices, hr_faces, hr_attrs, hr_stats = decode(
        f"tile_{tile_id:02d} HR global stitch", hr_norm
    )
    if g_vertices.shape != hr_vertices.shape or not torch.equal(g_vertices, hr_vertices):
        raise RuntimeError(f"tile {tile_id}: G/HR decoded geometry mismatch")
    if not torch.equal(g_faces, hr_faces):
        raise RuntimeError(f"tile {tile_id}: G/HR decoded topology mismatch")

    global_positions = experiment._map_local_to_global_chunked(
        g_vertices,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    ).detach().cpu().float()
    ids = experiment._coarse_cell_ids(global_positions, 1)
    g_low, _ = experiment._project_cell_mean(g_attrs, ids)
    hr_low, _ = experiment._project_cell_mean(hr_attrs, ids)

    attrs: Dict[str, torch.Tensor] = {
        "G": g_attrs,
        "G_low": g_low,
        "G_low_HR_high": g_low + (hr_attrs - hr_low),
        "HR": hr_attrs,
        "HR_low": hr_low,
        "HR_low_G_high": hr_low + (g_attrs - g_low),
    }
    decode_stats: Dict[str, Any] = {"G": g_stats, "HR": hr_stats}
    for alpha, name in (
        (0.25, "latent_interp_25"),
        (0.50, "latent_interp_50"),
        (0.75, "latent_interp_75"),
    ):
        vertices, faces, interp_attrs, stats = decode(
            f"tile_{tile_id:02d} {name} global stitch",
            g_norm + float(alpha) * (hr_norm - g_norm),
        )
        if vertices.shape != g_vertices.shape or not torch.equal(vertices, g_vertices):
            raise RuntimeError(f"tile {tile_id}: {name} geometry differs from G")
        if not torch.equal(faces, g_faces):
            raise RuntimeError(f"tile {tile_id}: {name} topology differs from G")
        attrs[name] = interp_attrs
        decode_stats[name] = stats

    # Build each patch with the same global geometry tensor.  The attributes
    # differ by variant, but geometry is shared so the cache does not duplicate
    # the large position/face arrays nine times.
    patches: Dict[str, core.ReturnedTilePatch] = {}
    for variant in VARIANT_ORDER:
        patches[variant] = _make_global_patch(
            tile_id=tile_id,
            box=row["box"],
            local_vertices=g_vertices,
            local_faces=g_faces,
            local_attrs=attrs[variant],
            transform=transform,
            global_camera=global_camera,
            query_chunk_size=int(args.query_chunk_size),
        )

    payload = {
        "format": "pixal3d_texture_pbr_global_tile_cache_v1",
        "tile_id": int(tile_id),
        "box": [int(v) for v in row["box"]],
        "global_vertices": patches["G"].vertices,
        "faces": patches["G"].faces,
        "attrs": {variant: patches[variant].vertex_attrs for variant in VARIANT_ORDER},
        "decode_stats": decode_stats,
        "transform": endpoints["transform"],
        "row_status": row.get("status"),
    }
    del endpoints, shape_norm, shape_denorm, texture_coords, g_norm, hr_norm
    del g_vertices, g_faces, g_attrs, hr_attrs, g_low, hr_low, attrs, patches
    _empty_cuda_cache()
    return payload


def _patches_from_payloads(
    payloads: Mapping[int, Mapping[str, Any]],
    variant: str,
) -> List[core.ReturnedTilePatch]:
    patches: List[core.ReturnedTilePatch] = []
    for tile_id in sorted(payloads):
        payload = payloads[tile_id]
        patches.append(
            core.ReturnedTilePatch(
                tile_id=int(tile_id),
                box=tuple(int(v) for v in payload["box"]),
                vertices=payload["global_vertices"].to(torch.float32),
                faces=payload["faces"].to(torch.int32),
                vertex_attrs=payload["attrs"][variant].to(torch.float32),
                stats={"source": "cached local mesh returned to global space"},
            )
        )
    return patches


def _stitch_global_geometry(
    patches: Sequence[core.ReturnedTilePatch],
    *,
    global_camera: Mapping[str, float],
    face_chunk_size: int,
    weld_tolerance: float,
) -> Dict[str, Any]:
    """Compute tile ownership and the weld map once for all PBR variants.

    All nine variants share decoded geometry.  The repository stitcher also
    averages vertex attributes, so calling it nine times would repeat the
    expensive projection and ``torch.unique`` work.  This helper reproduces
    its documented face-owner/weld policy once, then ``_apply_stitched_attrs``
    reuses the map for each material field.
    """
    if not patches:
        raise RuntimeError("cannot stitch empty patch list")
    if len(patches) == 1:
        patch = patches[0]
        vertices = patch.vertices.to(torch.float32)
        faces = patch.faces.to(torch.int32)
        return {
            "welded_vertices": vertices,
            "welded_faces": faces,
            "inverse": None,
            "counts": None,
            "kept_face_ids": [torch.arange(faces.shape[0], dtype=torch.int32)],
            "raw_segment_ends": [int(faces.shape[0] * 3)],
            "stats": {
                "operation": "direct single-tile global mesh; overlap/weld not needed",
                "input_tiles": 1,
                "input_faces": int(faces.shape[0]),
                "overlap_faces_removed": 0,
                "kept_faces_before_weld": int(faces.shape[0]),
                "degenerate_faces_removed_after_weld": 0,
                "raw_face_corner_vertices": int(faces.shape[0] * 3),
                "welded_vertices": int(vertices.shape[0]),
                "welded_faces": int(faces.shape[0]),
                "vertices_welded": 0,
                "weld_tolerance_object": float(weld_tolerance),
                "face_chunk_size": int(face_chunk_size),
                "tile_overlap_policy": "not applicable for one tile",
            },
        }

    if face_chunk_size <= 0 or weld_tolerance <= 0.0:
        raise ValueError("face_chunk_size and weld_tolerance must be positive")

    patch_ids = torch.tensor([int(p.tile_id) for p in patches], dtype=torch.long)
    patch_boxes = torch.tensor([list(p.box) for p in patches], dtype=torch.float32)
    patch_centers = torch.stack(
        (
            (patch_boxes[:, 0] + patch_boxes[:, 2]) * 0.5,
            (patch_boxes[:, 1] + patch_boxes[:, 3]) * 0.5,
        ),
        dim=1,
    )
    global_scale = float(global_camera["mesh_scale"])
    raw_vertices: List[torch.Tensor] = []
    kept_face_ids: List[torch.Tensor] = []
    raw_segment_ends: List[int] = []
    per_tile: List[Dict[str, Any]] = []
    total_input_faces = 0
    total_kept_faces = 0
    total_invalid_projection_faces = 0
    raw_offset = 0

    for patch in patches:
        vertices = patch.vertices.to(device="cpu", dtype=torch.float32)
        faces = patch.faces.to(device="cpu", dtype=torch.long)
        face_count = int(faces.shape[0])
        selected_face_parts: List[torch.Tensor] = []
        invalid_projection = 0
        for face_start in range(0, face_count, int(face_chunk_size)):
            face_end = min(face_start + int(face_chunk_size), face_count)
            face_chunk = faces[face_start:face_end]
            corner_vertices = vertices.index_select(
                0, face_chunk.reshape(-1)
            ).reshape(-1, 3, 3)
            centroids = corner_vertices.mean(dim=1)
            uv, _, finite = core._project_global_q_to_4096(
                centroids * (2.0 * global_scale), global_camera=global_camera
            )
            inside = (
                finite[:, None]
                & (uv[:, None, 0] >= patch_boxes[None, :, 0])
                & (uv[:, None, 0] < patch_boxes[None, :, 2])
                & (uv[:, None, 1] >= patch_boxes[None, :, 1])
                & (uv[:, None, 1] < patch_boxes[None, :, 3])
            )
            distance2 = (uv[:, None, :] - patch_centers[None, :, :]).square().sum(dim=2)
            distance2 = torch.where(
                inside, distance2, torch.full_like(distance2, float("inf"))
            )
            nearest_distance2, nearest_patch = distance2.min(dim=1)
            has_owner = finite & torch.isfinite(nearest_distance2)
            owner_ids = patch_ids.index_select(0, nearest_patch)
            keep = (~has_owner) | (owner_ids == int(patch.tile_id))
            invalid_projection += int((~finite).sum().item())
            selected = torch.nonzero(keep, as_tuple=False).reshape(-1).to(torch.int32)
            if selected.numel():
                selected_face_parts.append(selected + int(face_start))

        selected_faces = (
            torch.cat(selected_face_parts, dim=0)
            if selected_face_parts
            else torch.empty((0,), dtype=torch.int32)
        )
        selected_face_long = selected_faces.to(torch.long)
        selected_face_ids = selected_faces
        selected_faces = (
            faces.index_select(0, selected_face_long)
            if selected_face_ids.numel()
            else faces[:0]
        )
        selected_vertices = vertices.index_select(
            0, selected_faces.reshape(-1)
        ).reshape(-1, 3)
        raw_vertices.append(selected_vertices)
        raw_offset += int(selected_vertices.shape[0])
        kept_face_ids.append(selected_face_ids)
        raw_segment_ends.append(raw_offset)
        kept_count = int(selected_faces.shape[0])
        total_input_faces += face_count
        total_kept_faces += kept_count
        total_invalid_projection_faces += invalid_projection
        per_tile.append(
            {
                "tile_id": int(patch.tile_id),
                "input_faces": face_count,
                "kept_faces": kept_count,
                "overlap_faces_removed": face_count - kept_count,
                "invalid_projection_faces_kept": invalid_projection,
            }
        )

    if not any(value.numel() for value in raw_vertices):
        raise RuntimeError("global overlap ownership removed every face")
    raw_vertices_cat = torch.cat(raw_vertices, dim=0)
    quantized = torch.round(raw_vertices_cat / float(weld_tolerance)).to(torch.int64)
    _, inverse = torch.unique(quantized, dim=0, sorted=True, return_inverse=True)
    del quantized
    welded_count = int(inverse.max().item()) + 1 if inverse.numel() else 0
    counts = torch.bincount(inverse, minlength=welded_count).to(torch.float32)
    welded_vertices = torch.zeros((welded_count, 3), dtype=torch.float32)
    welded_vertices.index_add_(0, inverse, raw_vertices_cat)
    welded_vertices = welded_vertices / counts[:, None].clamp_min(1.0)
    welded_faces = inverse.reshape(-1, 3)
    nondegenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 0] != welded_faces[:, 2])
        & (welded_faces[:, 1] != welded_faces[:, 2])
    )
    degenerate_removed = int((~nondegenerate).sum().item())
    welded_faces = welded_faces[nondegenerate].to(torch.int32)
    del raw_vertices, raw_vertices_cat
    return {
        "welded_vertices": welded_vertices,
        "welded_faces": welded_faces,
        "inverse": inverse,
        "counts": counts,
        "kept_face_ids": kept_face_ids,
        "raw_segment_ends": raw_segment_ends,
        "stats": {
            "operation": "shared projected nearest-owner overlap removal followed by spatial weld",
            "input_tiles": int(len(patches)),
            "input_faces": int(total_input_faces),
            "overlap_faces_removed": int(total_input_faces - total_kept_faces),
            "kept_faces_before_weld": int(total_kept_faces),
            "degenerate_faces_removed_after_weld": int(degenerate_removed),
            "raw_face_corner_vertices": int(inverse.shape[0]),
            "welded_vertices": int(welded_vertices.shape[0]),
            "welded_faces": int(welded_faces.shape[0]),
            "vertices_welded": int(inverse.shape[0] - welded_vertices.shape[0]),
            "weld_tolerance_object": float(weld_tolerance),
            "invalid_projection_faces_kept": int(total_invalid_projection_faces),
            "face_chunk_size": int(face_chunk_size),
            "nearest_owner": "successful tile center in projected 4096 image space",
            "weld": "round(object_xyz / tolerance) spatial hash; average positions per cell",
            "face_policy": "same as official core._stitch_tile_patches_nearest",
            "per_tile": per_tile,
        },
    }


def _apply_stitched_attrs(
    geometry: Mapping[str, Any],
    payloads: Mapping[int, Mapping[str, Any]],
    variant: str,
) -> torch.Tensor:
    """Average one variant's attrs using a geometry-only shared weld map."""
    inverse = geometry["inverse"]
    counts = geometry["counts"]
    if inverse is None:
        tile_id = sorted(payloads)[0]
        return payloads[tile_id]["attrs"][variant].to(torch.float32)
    attrs_dim = int(next(iter(payloads.values()))["attrs"][variant].shape[1])
    welded_count = int(geometry["welded_vertices"].shape[0])
    welded_attrs = torch.zeros((welded_count, attrs_dim), dtype=torch.float32)
    raw_start = 0
    for tile_id, kept_ids in zip(sorted(payloads), geometry["kept_face_ids"]):
        payload = payloads[tile_id]
        faces = payload["faces"].to(torch.long)
        selected_faces = faces.index_select(0, kept_ids.to(torch.long))
        source_ids = selected_faces.reshape(-1)
        raw_attrs = payload["attrs"][variant].index_select(0, source_ids).to(torch.float32)
        raw_end = raw_start + int(raw_attrs.shape[0])
        welded_attrs.index_add_(0, inverse[raw_start:raw_end], raw_attrs)
        raw_start = raw_end
        del faces, selected_faces, source_ids, raw_attrs
    if raw_start != int(inverse.shape[0]):
        raise RuntimeError(
            f"shared stitch attr map length mismatch: {raw_start} vs {inverse.shape[0]}"
        )
    welded_attrs = welded_attrs / counts[:, None].clamp_min(1.0)
    return welded_attrs


def _save_image(path: Path, image: np.ndarray | Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path)
    else:
        Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB").save(path)


def _save_global_variant_sheet(
    *,
    path: Path,
    reference_path: Path,
    aligned_paths: Mapping[str, Path],
    multiview_paths: Mapping[str, Mapping[str, Path]],
    panel: int,
) -> None:
    """Rows are variants; columns share aligned/global and multiview cameras."""
    columns = ["input_canonical", "aligned_front", *VIEW_LABELS]
    row_header = 128
    header = 44
    cell_h = panel + 28
    sheet = Image.new(
        "RGB",
        (row_header + len(columns) * panel, header + len(VARIANT_ORDER) * cell_h),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(VARIANT_ORDER):
        y = header + row_index * cell_h
        draw.text((7, y + panel // 2 - 8), variant, fill=(255, 255, 255))
        paths = [reference_path, aligned_paths[variant]] + [
            multiview_paths[variant][label] for label in VIEW_LABELS
        ]
        for col, image_path in enumerate(paths):
            with Image.open(image_path) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            sheet.paste(
                image,
                (x + (panel - image.width) // 2, y + (panel - image.height) // 2),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_front_overview(
    *,
    path: Path,
    reference_path: Path,
    aligned_paths: Mapping[str, Path],
    panel: int,
) -> None:
    columns = ["input_canonical", *VARIANT_ORDER]
    header = 44
    row_header = 128
    sheet = Image.new("RGB", (row_header + len(columns) * panel, header + panel), "black")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    paths = [reference_path] + [aligned_paths[variant] for variant in VARIANT_ORDER]
    for col, image_path in enumerate(paths):
        with Image.open(image_path) as source:
            image = ImageOps.contain(source.convert("RGB"), (panel, panel))
        x = row_header + col * panel
        sheet.paste(image, (x + (panel - image.width) // 2, header + (panel - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_global_multiview_sheet(
    *,
    path: Path,
    multiview_paths: Mapping[str, Mapping[str, Path]],
    panel: int,
) -> None:
    header = 44
    row_header = 128
    sheet = Image.new(
        "RGB",
        (row_header + len(VIEW_LABELS) * panel, header + len(VARIANT_ORDER) * (panel + 28)),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(VIEW_LABELS):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(VARIANT_ORDER):
        y = header + row_index * (panel + 28)
        draw.text((7, y + panel // 2 - 8), variant, fill=(255, 255, 255))
        for col, label in enumerate(VIEW_LABELS):
            with Image.open(multiview_paths[variant][label]) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            sheet.paste(image, (x + (panel - image.width) // 2, y + (panel - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _render_variant(
    *,
    variant: str,
    mesh: MeshWithVertexPbr,
    output_root: Path,
    reference_path: Path,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    variant_root = output_root / variant
    aligned_dir = variant_root / "aligned_view"
    render_result = core._render(
        mesh,
        output_dir=aligned_dir,
        camera=global_camera,
        reference_image=reference_path,
        args=_render_args(args),
        envmap=envmap,
    )
    aligned_path = Path(str(render_result["render_png"]))

    mv_args = SimpleNamespace(
        resolution=int(args.multiview_resolution),
        ssaa=int(args.multiview_ssaa),
        peel_layers=int(args.peel_layers),
        face_chunk_size=int(args.face_chunk_size),
        radius_scale=float(args.radius_scale),
        use_envmap_bg=bool(args.use_envmap_bg),
    )
    extrinsics, intrinsics, labels, options = multiview._fixed_cameras(
        global_camera, mv_args
    )
    renderer = render_utils.get_renderer(mesh, **options)
    frames = multiview._render_variant_frames(
        mesh,
        renderer=renderer,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        options=options,
        envmap=envmap,
        args=mv_args,
    )
    multiview_dir = variant_root / "multiview"
    view_paths: Dict[str, Path] = {}
    for frame, label in zip(frames, labels):
        path = multiview_dir / f"{label}.png"
        _save_image(path, frame)
        view_paths[label] = path

    result = {
        "variant": variant,
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "aligned_view": str(aligned_path),
        "aligned_metrics": str(aligned_dir / "metrics.json"),
        "multiview": {label: str(path) for label, path in view_paths.items()},
        "renderer": "official core._render + pixal3d.utils.render_utils.PbrMeshRenderer",
        "camera": dict(global_camera),
    }
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official Pixal3D decoder/renderer")
    torch.cuda.set_device(int(args.cuda_device))
    source_dir = Path(args.experiment_dir).expanduser().resolve()
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    reference_path = source_dir / CANONICAL_REFERENCE
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    selected = _parse_ids(args.tile_ids)
    rows = {
        int(row["tile_id"]): row
        for row in summary.get("tiles", [])
        if row.get("status") == "success"
        and (selected is None or int(row["tile_id"]) in selected)
    }
    if not rows:
        raise RuntimeError("no successful tiles selected")

    output_root = source_dir / "global_stitched_quality"
    cache_root = output_root / "decoded_global_tiles"
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    envmap = core.load_envmap(str(args.envmap), device="cuda")

    payloads: Dict[int, Dict[str, Any]] = {}
    tile_records: List[Dict[str, Any]] = []
    for tile_id in sorted(rows):
        cache_path = cache_root / f"tile_{tile_id:02d}.pt"
        started = time.perf_counter()
        try:
            if bool(args.resume) and cache_path.is_file():
                payload = _load_torch(cache_path)
                if payload.get("format") != "pixal3d_texture_pbr_global_tile_cache_v1":
                    raise RuntimeError(f"invalid cache format: {cache_path}")
                print(f"[global tile {tile_id:02d}] reused decoded global cache")
            else:
                print(f"[global tile {tile_id:02d}] decode and return local mesh to global")
                payload = _decode_tile_payload(
                    pipeline=pipeline,
                    source_dir=source_dir,
                    tile_id=tile_id,
                    row=rows[tile_id],
                    global_camera=global_camera,
                    args=args,
                )
                _atomic_torch_save(cache_path, payload)
            payloads[tile_id] = payload
            tile_records.append(
                {
                    "tile_id": int(tile_id),
                    "status": "success",
                    "cache": str(cache_path),
                    "global_vertices": int(payload["global_vertices"].shape[0]),
                    "global_faces": int(payload["faces"].shape[0]),
                    "seconds": float(time.perf_counter() - started),
                }
            )
        except Exception as exc:
            tile_records.append(
                {
                    "tile_id": int(tile_id),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "seconds": float(time.perf_counter() - started),
                }
            )
            print(f"[global tile {tile_id:02d}] FAILED: {type(exc).__name__}: {exc}")
        finally:
            _empty_cuda_cache()

    if not payloads:
        raise RuntimeError("all selected tiles failed before global stitching")

    aligned_paths: Dict[str, Path] = {}
    multiview_paths: Dict[str, Dict[str, Path]] = {}
    variant_records: Dict[str, Any] = {}
    geometry_patches = _patches_from_payloads(payloads, "G")
    print("[global stitch] computing shared tile ownership and global weld map")
    shared_geometry = _stitch_global_geometry(
        geometry_patches,
        global_camera=global_camera,
        face_chunk_size=int(args.stitch_face_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    for variant in VARIANT_ORDER:
        print(f"[global stitch] {variant}: applying global PBR attrs")
        merged_attrs = _apply_stitched_attrs(shared_geometry, payloads, variant)
        merged = MeshWithVertexPbr(
            vertices=shared_geometry["welded_vertices"],
            faces=shared_geometry["welded_faces"],
            vertex_attrs=merged_attrs,
            layout=dict(experiment.PBR_LAYOUT),
        )
        stitch_stats = dict(shared_geometry["stats"])
        stitch_stats["vertex_attrs_range"] = core._tensor_range(merged_attrs)
        variant_root = output_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        merged_path = variant_root / "global_merged_mesh.pt"
        if bool(args.save_merged_mesh) or not bool(args.resume) or not merged_path.is_file():
            _atomic_torch_save(
                merged_path,
                {
                    "format": "pixal3d_texture_pbr_global_merged_mesh_v1",
                    "variant": variant,
                    "mesh": merged,
                    "stitch_stats": stitch_stats,
                    "global_camera": global_camera,
                    "source_tiles": sorted(payloads),
                },
            )
        render_record = _render_variant(
            variant=variant,
            mesh=merged,
            output_root=output_root,
            reference_path=reference_path,
            global_camera=global_camera,
            args=args,
            envmap=envmap,
        )
        aligned_paths[variant] = Path(render_record["aligned_view"])
        multiview_paths[variant] = {
            label: Path(path) for label, path in render_record["multiview"].items()
        }
        variant_records[variant] = {
            "stitch": stitch_stats,
            "merged_mesh": str(merged_path),
            "render": render_record,
        }
        _write_json(variant_root / "global_stitch_render.json", variant_records[variant])
        del merged_attrs, merged
        _empty_cuda_cache()

    del geometry_patches, shared_geometry

    comparison_sheet = output_root / "global_aligned_view_and_multiview_sheet.png"
    _save_global_variant_sheet(
        path=comparison_sheet,
        reference_path=reference_path,
        aligned_paths=aligned_paths,
        multiview_paths=multiview_paths,
        panel=int(args.sheet_panel),
    )
    front_overview = output_root / "global_aligned_front_overview.png"
    _save_front_overview(
        path=front_overview,
        reference_path=reference_path,
        aligned_paths=aligned_paths,
        panel=int(args.overview_panel),
    )
    multiview_sheet = output_root / "global_multiview_sheet.png"
    _save_global_multiview_sheet(
        path=multiview_sheet,
        multiview_paths=multiview_paths,
        panel=int(args.sheet_panel),
    )
    output = {
        "format": "pixal3d_texture_pbr_global_stitch_v1",
        "source_experiment": str(source_dir),
        "cuda_device": int(args.cuda_device),
        "reference_view": str(reference_path),
        "camera": global_camera,
        "coordinate_transform": "official core._local_q_to_global_q",
        "stitch_policy": "official core._stitch_tile_patches_nearest: projected nearest-tile owner, then spatial weld",
        "weld_tolerance": float(args.stitch_tolerance),
        "successful_tiles": sorted(payloads),
        "failed_tiles": [row["tile_id"] for row in tile_records if row["status"] == "failed"],
        "tile_records": tile_records,
        "variants": list(VARIANT_ORDER),
        "views": list(VIEW_LABELS),
        "resolution": {
            "aligned_view": int(args.aligned_resolution),
            "multiview": int(args.multiview_resolution),
        },
        "variant_records": variant_records,
        "comparison_sheet": str(comparison_sheet),
        "front_overview": str(front_overview),
        "multiview_sheet": str(multiview_sheet),
    }
    _write_json(output_root / "global_stitch_summary.json", output)
    print(f"[done] global stitched output={output_root}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="outputs/codex_texture_pbr_degradation_cuda4_all_tiles",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-merged-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-chunk-size", type=int, default=32_768)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / 1024.0)
    parser.add_argument("--stitch-face-chunk-size", type=int, default=250_000)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--aligned-resolution", type=int, default=1024)
    parser.add_argument("--aligned-ssaa", type=int, default=1)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--peel-layers", type=int, default=4)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--sheet-panel", type=int, default=256)
    parser.add_argument("--overview-panel", type=int, default=192)
    return parser


if __name__ == "__main__":
    run(_build_parser().parse_args())
