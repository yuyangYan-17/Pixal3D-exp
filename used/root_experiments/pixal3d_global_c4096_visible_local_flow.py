#!/usr/bin/env python3
"""Strict global-C4096 carrier / local-C1024 multi-view texture flow.

This entry point is the executable version of ``Codex.md``.  Geometry is
always cut from the fixed baseline mesh and voxelized independently in each
native 1024 tile.  A sparse C4096 surface carrier is used only as the global
PBR/visibility join key.  After an independent local texture flow, only
carrier points that are directly visible in the current view and project into
the current tile are queried in the decoded local field.  The nearest tile
centre candidate wins; the baseline C4096 value is retained when no valid
visible candidate exists.

In particular, this file never maps a global C4096 integer coordinate to a
local C1024 integer coordinate and never concatenates DINO/SLat contexts.
The flow barrier itself remains per-context, while decoded local PBR values
are fused only through the shared global C4096 carrier before the corrected
endpoint is re-encoded.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import o_voxel
import torch
from PIL import Image

import pixal3d.models as pixal3d_models
import pixal3d_cross_tile_pbr_perstep as cross_tile
import pixal3d_multiview_fixed_geometry_pbr_gaussian_sr as mv
import pixal3d_texture_visibility_guided_pbr_flow as visibility
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.representations import MeshWithFacePbr, MeshWithVertexPbr, MeshWithVoxel
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global_c4096_visible_local_flow_v1"
ANGLES = (0, 120, 240)
GLOBAL_RESOLUTION = 4096
LOCAL_RESOLUTION = 1024
PBR_LAYOUT = dict(core.PBR_LAYOUT)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_baseline(path: Path) -> MeshWithVoxel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if isinstance(data, MeshWithVoxel):
        return data.to("cpu")
    if not isinstance(data, Mapping):
        raise RuntimeError(f"baseline payload is not a mesh mapping: {path}")
    required = ("vertices", "faces", "coords", "attrs", "origin", "voxel_size", "voxel_shape", "layout")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"baseline payload missing keys {missing}: {path}")
    return MeshWithVoxel(
        data["vertices"].to(torch.float32).cpu(),
        data["faces"].to(torch.int32).cpu(),
        torch.as_tensor(data["origin"]).tolist(),
        float(data["voxel_size"]),
        data["coords"].to(torch.int32).cpu(),
        data["attrs"].to(torch.float32).cpu(),
        torch.Size(data["voxel_shape"]),
        dict(data["layout"]),
    )


def _load_camera(baseline_dir: Path) -> Dict[str, float]:
    summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    camera = {
        "camera_angle_x": float(summary["camera_angle_x"]),
        "distance": float(summary["distance"]),
        "mesh_scale": 1.0,
    }
    expected = {"camera_angle_x": 0.517371749106554, "distance": 1.889538288116455, "mesh_scale": 1.0}
    for key, value in expected.items():
        if abs(camera[key] - value) > 1e-7:
            raise RuntimeError(f"baseline camera {key}={camera[key]} differs from Codex fixed value {value}")
    return camera


def _global_field(carrier: Mapping[str, torch.Tensor], device: torch.device) -> MeshWithVoxel:
    attrs = carrier["baseline_attrs"].to(device=device, dtype=torch.float32)
    coords = carrier["coords"].to(device=device, dtype=torch.int32)
    return MeshWithVoxel(
        torch.empty((1, 3), device=device, dtype=torch.float32),
        torch.empty((0, 3), device=device, dtype=torch.int32),
        [-0.5, -0.5, -0.5],
        1.0 / GLOBAL_RESOLUTION,
        coords,
        attrs,
        torch.Size([1, 6, GLOBAL_RESOLUTION, GLOBAL_RESOLUTION, GLOBAL_RESOLUTION]),
        dict(PBR_LAYOUT),
    )


@torch.no_grad()
def _query_field(field: MeshWithVoxel, points: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return core._query_mesh_attrs_chunked(field, points, chunk_size=int(chunk_size)).to(torch.float32)


def _build_or_load_carrier(
    baseline: MeshWithVoxel,
    output_dir: Path,
    query_chunk_size: int,
) -> Dict[str, torch.Tensor]:
    path = output_dir / "global_c4096_carrier.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("resolution", -1)) == GLOBAL_RESOLUTION and payload.get("surface_convention") == "object_centered_normalized":
            carrier = {
                "coords": payload["coords"].to(torch.int32).contiguous(),
                "surface_points": payload["surface_points"].to(torch.float32).contiguous(),
                "baseline_attrs": payload["baseline_attrs"].to(torch.float32).contiguous(),
            }
            if carrier["coords"].shape[0] == carrier["surface_points"].shape[0] == carrier["baseline_attrs"].shape[0]:
                print(f"[carrier] reuse {path} points={carrier['coords'].shape[0]}", flush=True)
                return carrier
        print("[carrier] cache schema/resolution mismatch; rebuilding", flush=True)

    started = time.perf_counter()
    print("[carrier] voxelize baseline mesh to sparse C4096 ...", flush=True)
    coords, dual_world, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=baseline.vertices.cpu().to(torch.float32),
        faces=baseline.faces.cpu().to(torch.int32),
        grid_size=GLOBAL_RESOLUTION,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )
    coords = coords.to(torch.int32).cpu().contiguous()
    # o_voxel returns dual vertices in the voxelizer AABB convention.  With
    # aabb=[0,1] after its internal translation this is [0,1], whereas every
    # Pixal3D mesh/query/camera routine uses the centered object frame
    # [-0.5,0.5].  The conversion is continuous and happens once here; it is
    # not a C4096->C1024 integer remap.
    dual_world = (dual_world.to(torch.float32).cpu() - 0.5).contiguous()
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise RuntimeError(f"global C4096 voxelization returned invalid coords {tuple(coords.shape)}")
    if not torch.isfinite(dual_world).all() or bool(((coords < 0) | (coords >= GLOBAL_RESOLUTION)).any()):
        raise RuntimeError("global C4096 carrier has non-finite/out-of-range surface points")
    print(f"[carrier] query baseline PBR at {coords.shape[0]} surface points", flush=True)
    field = core._make_attribute_query_mesh(baseline, torch.device("cuda"))
    attrs: List[torch.Tensor] = []
    for start in range(0, int(dual_world.shape[0]), int(query_chunk_size)):
        points = dual_world[start : start + int(query_chunk_size)].to("cuda")
        values = _query_field(field, points, query_chunk_size).cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError(f"baseline C1024 PBR query is non-finite at chunk {start}")
        attrs.append(values)
        if start == 0 or (start // int(query_chunk_size)) % 100 == 0:
            print(f"[carrier] queried {min(start + int(query_chunk_size), dual_world.shape[0])}/{dual_world.shape[0]}", flush=True)
    baseline_attrs = torch.cat(attrs, dim=0).to(torch.float32).contiguous()
    carrier = {"coords": coords, "surface_points": dual_world, "baseline_attrs": baseline_attrs}
    _atomic_save(path, {
        "format": FORMAT,
        "resolution": GLOBAL_RESOLUTION,
        "surface_convention": "object_centered_normalized",
        "coords": coords,
        "surface_points": dual_world,
        "baseline_attrs": baseline_attrs,
        "intersected": intersected.to(torch.bool).cpu(),
        "voxelizer": "o_voxel.mesh_to_flexible_dual_grid conservative surface",
        "seconds": time.perf_counter() - started,
    })
    del field, attrs, intersected
    _empty_cuda_cache()
    print(f"[carrier] ready points={coords.shape[0]} seconds={time.perf_counter() - started:.1f}", flush=True)
    return carrier


def _visibility_one_view(
    carrier: Mapping[str, torch.Tensor],
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    image: Image.Image,
    angle: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    root = output_dir / "global_c4096_visibility" / f"view_{int(angle):03d}"
    root.mkdir(parents=True, exist_ok=True)
    visible_path = root / "surface_visible.pt"
    rotation = mv._yaw_matrix(int(angle))
    q_world = carrier["surface_points"].to(torch.float32) * (2.0 * float(camera["mesh_scale"]))
    q_view = mv._world_to_view_q(q_world, rotation).contiguous()
    uv, _, finite = core._project_global_q_to_image(
        q_view, global_camera=camera, image_width=GLOBAL_RESOLUTION, image_height=GLOBAL_RESOLUTION
    )
    if visible_path.is_file():
        cached = torch.load(visible_path, map_location="cpu", weights_only=False)
        visible = cached["visible"].to(torch.bool).contiguous()
        if visible.shape[0] == q_view.shape[0] and cached.get("surface_convention") == "object_centered_normalized":
            stats = dict(cached.get("stats", {}))
            stats["cache_reused"] = True
            return q_view.cpu(), uv.cpu(), visible, stats

    rotated_vertices = baseline.vertices.cpu() @ rotation
    view_mesh = MeshWithVoxel(
        rotated_vertices,
        baseline.faces.cpu(),
        baseline.origin.tolist(),
        float(baseline.voxel_size),
        baseline.coords.cpu(),
        baseline.attrs.cpu(),
        baseline.voxel_shape,
        dict(baseline.layout),
    )
    print(f"[visibility] view={angle} render geometry z-buffer {GLOBAL_RESOLUTION}^2", flush=True)
    buffers = visibility._render_global_visibility_buffers(
        view_mesh,
        global_camera=camera,
        resolution=GLOBAL_RESOLUTION,
        face_chunk_size=int(args.render_face_chunk_size),
        device=torch.device("cuda"),
    )
    visibility._save_visibility_debug(root, buffers)
    camera_points = core._camera_q_to_points(q_view, distance=float(camera["distance"]), mesh_scale=float(camera["mesh_scale"]))
    depth = (-camera_points[:, 2]).cpu()
    uv_cpu = uv.cpu()
    finite_cpu = finite.cpu()
    pixels = torch.round(uv_cpu).to(torch.long)
    inside = finite_cpu & (pixels[:, 0] >= 0) & (pixels[:, 0] < GLOBAL_RESOLUTION) & (pixels[:, 1] >= 0) & (pixels[:, 1] < GLOBAL_RESOLUTION)
    safe = pixels.clamp(0, GLOBAL_RESOLUTION - 1)
    zbuf = buffers["depth"][safe[:, 1], safe[:, 0]].to(torch.float32)
    fg = buffers["foreground"][safe[:, 1], safe[:, 0]].bool()
    tolerance = max(0.0025, 4.0 * math.sqrt(3.0) / GLOBAL_RESOLUTION)
    visible = inside & fg & torch.isfinite(zbuf) & torch.isfinite(depth) & ((depth - zbuf).abs() <= tolerance)
    stats = {
        "angle": int(angle),
        "resolution": GLOBAL_RESOLUTION,
        "surface_points": int(q_view.shape[0]),
        "projected_finite": int(finite_cpu.sum()),
        "projected_inside": int(inside.sum()),
        "zbuffer_foreground_hits": int((inside & fg).sum()),
        "direct_visible_surface_points": int(visible.sum()),
        "direct_visible_fraction_of_surface": float(visible.to(torch.float32).mean()),
        "depth_tolerance": float(tolerance),
        "visibility_rule": "rounded 4096 pixel, foreground z-buffer, abs(camera-depth-zbuffer)<=tolerance",
        "cache_reused": False,
        "renderer": buffers.get("renderer", ""),
    }
    _atomic_save(visible_path, {"visible": visible, "stats": stats, "surface_convention": "object_centered_normalized"})

    # The audit image is deliberately downsampled for display only; all
    # decisions above use native 4096 coordinates and the full carrier.
    max_overlay = int(args.max_overlay_points)
    stride = max(1, int(math.ceil(q_view.shape[0] / max(1, max_overlay))))
    sample = torch.arange(0, q_view.shape[0], stride, dtype=torch.long)
    overlay_uv = uv_cpu.index_select(0, sample) / 4.0
    overlay_visible = visible.index_select(0, sample)
    overlay_stats = mv._overlay_points(
        image,
        overlay_uv,
        overlay_visible,
        root / "global_c4096_surface_visibility_overlay.png",
        label="global C4096 surface",
        radius=0,
    )
    stats["overlay"] = {**overlay_stats, "sample_stride": stride, "sampled_points": int(sample.numel())}
    _atomic_json(root / "summary.json", stats)
    del buffers, view_mesh, rotated_vertices, camera_points, zbuf, fg
    _empty_cuda_cache()
    return q_view.cpu(), uv_cpu, visible, stats


def _build_visibility(
    carrier: Mapping[str, torch.Tensor],
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    views: Mapping[int, Image.Image],
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
    q_views: Dict[int, torch.Tensor] = {}
    uvs: Dict[int, torch.Tensor] = {}
    visible: Dict[int, torch.Tensor] = {}
    rows: Dict[str, Any] = {}
    for angle in ANGLES:
        q_view, uv, flags, stats = _visibility_one_view(carrier, baseline, camera, views[int(angle)], int(angle), output_dir, args)
        q_views[int(angle)] = q_view
        uvs[int(angle)] = uv
        visible[int(angle)] = flags
        rows[str(int(angle))] = stats
    _atomic_json(output_dir / "global_c4096_visibility_stats.json", {
        "carrier": "surface points from conservative baseline mesh voxelization",
        "views": rows,
        "direct_visible_only": True,
        "local_slat_visibility_inheritance": False,
    })
    return q_views, uvs, visible, rows


def _make_context_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    # Reuse the repository's already-tested native context preparation and
    # camera/condition schema, but force one context per flow call below.
    native = mv._parser().parse_args([])
    for key, value in vars(native).items():
        setattr(args, key, value) if not hasattr(args, key) else None
    args.multiview_image = str(args.input)
    args.baseline_dir = str(args.baseline_dir)
    args.output_dir = str(output_dir)
    args.cuda_device = int(args.cuda_device)
    args.angles = list(ANGLES)
    args.selected_views = list(ANGLES)
    args.source_view_size = 1024
    args.source_tile_size = 256
    args.source_tile_stride = 128
    args.model_tile_size = 1024
    args.flow_batch_size = max(1, int(getattr(args, "flow_batch_size", 44)))
    args.initial_encode_batch_size = max(1, int(getattr(args, "initial_encode_batch_size", 1)))
    args.decode_batch_size = max(1, int(getattr(args, "decode_batch_size", 12)))
    args.pbr_encode_batch_size = max(1, int(getattr(args, "pbr_encode_batch_size", 13)))
    args.max_tiles = None
    args.tile_ids = list(args.requested_tile_ids) if args.requested_tile_ids is not None else None
    args.debug = bool(args.debug)
    args.render = False
    return args


def _direct_global_slat_visibility(
    contexts: Sequence[mv.TileContext],
    camera: Mapping[str, float],
    output_dir: Path,
) -> Dict[int, Dict[str, int]]:
    """Attach direct global 4096 z-buffer visibility for decoded donor gating.

    Flow never reads this flag.  The decoder helper uses it only to construct
    its diagnostic masked mesh; the global accumulator below performs the
    actual donor query against the direct global visibility table.
    """
    rows: Dict[int, Dict[str, int]] = {}
    by_angle: Dict[int, List[mv.TileContext]] = {}
    for context in contexts:
        by_angle.setdefault(int(context.angle), []).append(context)
    for angle, angle_contexts in by_angle.items():
        root = output_dir / "global_c4096_visibility" / f"view_{int(angle):03d}"
        depth_path = root / "depth.pt"
        foreground_path = root / "foreground.pt"
        if not depth_path.is_file() or not foreground_path.is_file():
            raise FileNotFoundError(f"missing direct visibility buffers for view {angle}: {depth_path}, {foreground_path}")
        depth_payload = torch.load(depth_path, map_location="cpu", weights_only=False)
        foreground_payload = torch.load(foreground_path, map_location="cpu", weights_only=False)
        depth_buffer = (depth_payload["depth"] if isinstance(depth_payload, Mapping) else depth_payload).to(torch.float32)
        foreground = (foreground_payload["foreground"] if isinstance(foreground_payload, Mapping) else foreground_payload).bool()
        rotation = mv._yaw_matrix(int(angle))
        total = visible_count = 0
        for context in angle_contexts:
            coords = context.shape_norm.coords[:, -3:].detach().cpu().to(torch.float32)
            local_center = (coords + 0.5) / float(mv.LATENT_RESOLUTION) - 0.5
            q_local = local_center * (2.0 * float(context.transform.mesh_scale))
            q_view, _ = core._local_q_to_global_q(q_local, global_camera=camera, transform=context.transform)
            q_view = q_view.to(torch.float32)
            uv, _, finite = core._project_global_q_to_image(
                q_view, global_camera=camera, image_width=GLOBAL_RESOLUTION, image_height=GLOBAL_RESOLUTION
            )
            camera_points = core._camera_q_to_points(q_view, distance=float(camera["distance"]), mesh_scale=float(camera["mesh_scale"]))
            point_depth = (-camera_points[:, 2]).to(torch.float32)
            pixels = torch.round(uv).to(torch.long)
            inside = finite.cpu() & (pixels[:, 0] >= 0) & (pixels[:, 0] < GLOBAL_RESOLUTION) & (pixels[:, 1] >= 0) & (pixels[:, 1] < GLOBAL_RESOLUTION)
            safe = pixels.clamp(0, GLOBAL_RESOLUTION - 1)
            zbuf = depth_buffer[safe[:, 1], safe[:, 0]]
            fg = foreground[safe[:, 1], safe[:, 0]]
            tolerance = max(0.0025, 4.0 * math.sqrt(3.0) / GLOBAL_RESOLUTION)
            flags = inside & fg & torch.isfinite(zbuf) & torch.isfinite(point_depth) & ((point_depth.cpu() - zbuf).abs() <= tolerance)
            context.slat_visible = flags.bool().contiguous()
            context.support_stats["slat_visibility_source"] = "direct global C4096 z-buffer at exact C64 center projection"
            context.support_stats["visible_slat_direct_global"] = int(flags.sum())
            context.support_stats["invisible_slat_direct_global"] = int((~flags).sum())
            total += int(flags.numel())
            visible_count += int(flags.sum())
        rows[int(angle)] = {"contexts": len(angle_contexts), "slat_points": total, "direct_visible_slat": visible_count}
        del depth_buffer, foreground
    _atomic_json(output_dir / "global_direct_slat_visibility.json", rows)
    return rows


def _global_accumulator_field(attrs: torch.Tensor, coords: torch.Tensor, device: torch.device, channels: int) -> MeshWithVoxel:
    return MeshWithVoxel(
        torch.empty((1, 3), device=device, dtype=torch.float32),
        torch.empty((0, 3), device=device, dtype=torch.int32),
        [-0.5, -0.5, -0.5],
        1.0 / GLOBAL_RESOLUTION,
        coords.to(device=device, dtype=torch.int32),
        attrs.to(device=device, dtype=torch.float32),
        torch.Size([1, channels, GLOBAL_RESOLUTION, GLOBAL_RESOLUTION, GLOBAL_RESOLUTION]),
        dict(PBR_LAYOUT),
    )


@torch.no_grad()
def _run_global_flow(
    contexts: Sequence[mv.TileContext],
    pipeline: Any,
    args: argparse.Namespace,
    output_dir: Path,
    carrier: Mapping[str, torch.Tensor],
    q_views: Mapping[int, torch.Tensor],
    uvs: Mapping[int, torch.Tensor],
    visible: Mapping[int, torch.Tensor],
    camera: Mapping[str, float],
) -> Dict[int, Any]:
    if not contexts:
        raise RuntimeError("no local contexts available for texture flow")
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {
        **pipeline.tex_slat_sampler_params,
        "steps": int(args.num_steps),
        "rescale_t": float(args.texture_rescale_t),
        "guidance_strength": float(args.texture_guidance_strength),
        "guidance_rescale": float(args.texture_guidance_rescale),
    }
    schedule = cross_tile._native_schedule(sampler, merged)
    start = cross_tile._schedule_start(schedule, float(args.noise_timestep))
    step_kwargs = cross_tile._sampler_step_kwargs(merged)
    flow_batch_size = max(1, int(args.flow_batch_size))
    decode_batch_size = max(1, int(args.decode_batch_size))
    pbr_encode_batch_size = max(1, int(args.pbr_encode_batch_size))
    states = {context.context_id: mv._sparse_cpu(context.initial_state) for context in contexts}
    flow_groups = list(mv._flow_groups(contexts, flow_batch_size))
    decode_groups = list(mv._flow_groups(contexts, decode_batch_size))
    pbr_encode_groups = list(mv._flow_groups(contexts, pbr_encode_batch_size))
    rows: List[Dict[str, Any]] = []
    carrier_attrs = carrier["baseline_attrs"]
    try:
        for step_index, (t, t_next) in enumerate(zip(schedule[start:-1], schedule[start + 1:]), start=start):
            started = time.perf_counter()
            print(
                f"[global flow] step={step_index} t={t:.8f} contexts={len(contexts)} "
                f"flow_batch={flow_batch_size} decode_batch={decode_batch_size} pbr_encode_batch={pbr_encode_batch_size}",
                flush=True,
            )
            predictions: Dict[int, Dict[str, Any]] = {}
            model.to("cuda")
            for group in flow_groups:
                predictions.update(mv._predict_flow_batch(group, states, model, sampler, float(t), step_kwargs))
            model.cpu()
            _empty_cuda_cache()

            snapshots = mv._decode_snapshots_batched(
                contexts,
                {key: value["x0"] for key, value in predictions.items()},
                pipeline,
                args,
                decode_batch_size,
            )

            # Barrier C/D: accumulate visible decoded PBR only at global C4096
            # carrier IDs, then query the global consensus back to every local
            # target support.  No local target visibility gate is applied.
            global_num = torch.zeros_like(carrier_attrs, dtype=torch.float32)
            global_den = torch.zeros((carrier_attrs.shape[0], 1), dtype=torch.float32)
            global_count = torch.zeros((carrier_attrs.shape[0],), dtype=torch.int32)
            # Keep a compact per-carrier view bitmask for diagnostics.  The
            # actual fusion uses global_num/global_den; this mask only tells
            # us whether receipts came from at least two distinct views.
            global_view_mask = torch.zeros((carrier_attrs.shape[0],), dtype=torch.uint8)
            for donor in contexts:
                ids, p_local, uv_tile = _query_table(
                    donor,
                    q_views[int(donor.angle)],
                    uvs[int(donor.angle)],
                    visible[int(donor.angle)],
                    camera,
                )
                if not ids.numel():
                    continue
                values = cross_tile._query_mesh_chunked(
                    snapshots[donor.context_id].mesh,
                    p_local.to("cuda"),
                    int(args.query_chunk_size),
                ).detach().cpu().to(torch.float32)
                valid = torch.isfinite(values).all(dim=1)
                if not bool(valid.any()):
                    continue
                ids = ids[valid]
                values = values[valid]
                weights = torch.exp(-mv._tile_distance(uv_tile[valid]).square() / (2.0 * float(args.gaussian_sigma) ** 2))
                global_num.index_add_(0, ids, values * weights[:, None])
                global_den.index_add_(0, ids, weights[:, None])
                global_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
                view_bit = {0: 1, 120: 2, 240: 4}.get(int(donor.angle))
                if view_bit is None:
                    raise RuntimeError(f"unsupported donor angle {donor.angle}")
                global_view_mask[ids] = global_view_mask[ids] | int(view_bit)

            num_field = _global_accumulator_field(global_num, carrier["coords"], torch.device("cuda"), 6)
            den_field = _global_accumulator_field(global_den, carrier["coords"], torch.device("cuda"), 1)
            fused_fields: Dict[int, torch.Tensor] = {}
            fusion_stats: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                target_world = context.target_world_q.to(torch.float32) / (2.0 * float(camera["mesh_scale"]))
                numerator = _query_field(num_field, target_world.to("cuda"), int(args.query_chunk_size)).cpu()
                denominator = _query_field(den_field, target_world.to("cuda"), int(args.query_chunk_size)).cpu()[:, :1]
                self_field = snapshots[context.context_id].target_field.cpu().to(torch.float32)
                valid_target = torch.isfinite(denominator[:, 0]) & (denominator[:, 0] > 1e-6) & torch.isfinite(numerator).all(dim=1)
                fused = self_field.clone()
                fused[valid_target] = numerator[valid_target] / denominator[valid_target]
                fused_fields[context.context_id] = fused
                fusion_stats[context.context_id] = {
                    "active_ovoxels": int(fused.shape[0]),
                    "global_visible_donor_receipts": int(valid_target.sum()),
                    "no_donor_keeps_self_prediction": int((~valid_target).sum()),
                    "global_carrier_donor_count_mean": float(global_count.to(torch.float32).mean()),
                    "global_carrier_donor_points": int((global_count > 0).sum()),
                }
            del num_field, den_field, global_num, global_den
            _empty_cuda_cache()

            corrected_x0: Dict[int, Any] = {}
            cycle_diagnostics: Dict[int, Dict[str, Any]] = {}
            pbr_encoder = getattr(args, "_pbr_encoder_instance", None)
            if pbr_encoder is None:
                raise RuntimeError("PBR encoder instance missing from global flow args")
            pbr_encoder.to("cuda")
            for group in pbr_encode_groups:
                encoded_result = mv._encode_fused_batch(
                    group,
                    fused_fields,
                    predictions,
                    pbr_encoder,
                    pipeline,
                    self_fields={
                        context.context_id: snapshots[context.context_id].target_field.cpu()
                        for context in group
                    },
                    cycle_correction=bool(args.cycle_correction),
                    return_diagnostics=bool(args.cycle_correction),
                )
                if bool(args.cycle_correction):
                    encoded_batch, diagnostics_batch = encoded_result
                    corrected_x0.update(encoded_batch)
                    cycle_diagnostics.update(diagnostics_batch)
                else:
                    corrected_x0.update(encoded_result)
            pbr_encoder.cpu()
            _empty_cuda_cache()

            correction_rows: List[Dict[str, Any]] = []
            for context in contexts:
                before = predictions[context.context_id]["x0"].feats.detach().cpu().to(torch.float32)
                after = corrected_x0[context.context_id].feats.detach().cpu().to(torch.float32)
                delta = (after - before).abs()
                correction_rows.append({
                    "context_id": int(context.context_id),
                    "angle": int(context.angle),
                    "tile_id": int(context.tile_id),
                    **mv._quantiles(delta),
                })
                if bool(args.cycle_correction):
                    correction_rows[-1].update(cycle_diagnostics[context.context_id])

            next_states: Dict[int, Any] = {}
            for group in flow_groups:
                old_values = [mv._sparse_cuda(states[c.context_id]) for c in group]
                endpoint_values = [mv._sparse_cuda(corrected_x0[c.context_id]) for c in group]
                old_batch = mv._pack_sparse_batch(old_values, "global Euler state")
                endpoint_batch = mv._pack_sparse_batch(endpoint_values, "global Euler endpoint")
                velocity_batch = sampler._xstart_to_pred(old_batch, float(t), endpoint_batch)
                next_batch = SparseTensor(old_batch.feats - float(t - t_next) * velocity_batch.feats, old_batch.coords.detach().clone())
                next_values = mv._unpack_sparse_batch(next_batch, old_values, "global Euler next state")
                for context, next_state in zip(group, next_values):
                    if not torch.isfinite(next_state.feats).all():
                        raise RuntimeError(f"non-finite global flow state at step {step_index}, context {context.context_id}")
                    next_states[context.context_id] = mv._sparse_cpu(next_state)
                del old_values, endpoint_values, old_batch, endpoint_batch, velocity_batch, next_batch, next_values
                _empty_cuda_cache()
            states = next_states
            record = {
                "step": int(step_index),
                "t": float(t),
                "t_next": float(t_next),
                "contexts": len(contexts),
                "seconds": time.perf_counter() - started,
                "flow_batch_size": flow_batch_size,
                "decode_batch_size": decode_batch_size,
                "pbr_encode_batch_size": pbr_encode_batch_size,
                "batch_calls": {"flow_forward": len(flow_groups), "endpoint_decode": len(decode_groups), "pbr_re_encode": len(pbr_encode_groups), "euler": len(flow_groups)},
                "all_contexts_completed_before_state_replace": True,
                "synchronous_jacobi": True,
                "global_carrier_donor_points": int((global_count > 0).sum()),
                "global_carrier_cross_view_receipts": int(((global_view_mask & (global_view_mask - 1)) != 0).sum()),
                "correction_norm": {
                    "mean": float(sum(row["mean"] for row in correction_rows) / max(1, len(correction_rows))),
                    "max": float(max((row["max"] for row in correction_rows), default=0.0)),
                },
                "correction_rows": correction_rows,
                "tiles": fusion_stats,
            }
            _atomic_json(output_dir / "steps" / f"step_{step_index:02d}_summary.json", record)
            rows.append(record)
            del predictions, snapshots, fused_fields, corrected_x0, cycle_diagnostics, global_count, global_view_mask, correction_rows
            _empty_cuda_cache()
    finally:
        model.cpu()
        _empty_cuda_cache()
    _atomic_json(output_dir / "flow_summary.json", {
        "schedule": schedule,
        "start_index": start,
        "steps": rows,
        "route": "all context flow forward -> endpoint decode -> global C4096 visible donor Gaussian fusion -> local query -> fused/self PBR re-encode -> cycle residual -> xstart_to_pred -> synchronous Euler" if bool(args.cycle_correction) else "all context flow forward -> endpoint decode -> global C4096 visible donor Gaussian fusion -> local query -> PBR re-encode -> xstart_to_pred -> synchronous Euler",
        "cycle_correction": bool(args.cycle_correction),
        "cycle_correction_coefficient": 1.0 if bool(args.cycle_correction) else 0.0,
        "flow_batch_size": flow_batch_size,
        "decode_batch_size": decode_batch_size,
        "pbr_encode_batch_size": pbr_encode_batch_size,
        "global_carrier_resolution": GLOBAL_RESOLUTION,
        "flow_visibility_gate": False,
        "donor_visibility_gate": "direct global C4096 z-buffer",
    })
    return states


def _query_table(
    context: mv.TileContext,
    q_view: torch.Tensor,
    uv_full: torch.Tensor,
    visible: torch.Tensor,
    camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = context.box
    finite = torch.isfinite(uv_full).all(dim=1) & torch.isfinite(q_view).all(dim=1)
    inside = finite & (uv_full[:, 0] >= float(x0)) & (uv_full[:, 0] < float(x1)) & (uv_full[:, 1] >= float(y0)) & (uv_full[:, 1] < float(y1))
    rows = torch.where(visible & inside)[0]
    if not rows.numel():
        return rows, torch.empty((0, 3)), torch.empty((0, 2))
    q_local, uv_tile = core._global_q_to_local_q(q_view.index_select(0, rows), global_camera=camera, transform=context.transform)
    p_local = q_local / (2.0 * float(context.transform.mesh_scale))
    valid = torch.isfinite(p_local).all(dim=1) & torch.isfinite(uv_tile).all(dim=1)
    valid &= (p_local >= -0.5 - 2e-4).all(dim=1) & (p_local <= 0.5 + 2e-4).all(dim=1)
    valid &= (uv_tile[:, 0] >= 0.0) & (uv_tile[:, 0] < 1024.0) & (uv_tile[:, 1] >= 0.0) & (uv_tile[:, 1] < 1024.0)
    return rows[valid], p_local[valid].to(torch.float32), uv_tile[valid].to(torch.float32)


@torch.no_grad()
def _decode_and_write_global(
    contexts: Sequence[mv.TileContext],
    states: Mapping[int, Any],
    pipeline: Any,
    carrier: Mapping[str, torch.Tensor],
    q_views: Mapping[int, torch.Tensor],
    uvs: Mapping[int, torch.Tensor],
    visible: Mapping[int, torch.Tensor],
    camera: Mapping[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    final_attrs = carrier["baseline_attrs"].clone()
    point_count = int(final_attrs.shape[0])
    best_distance = torch.full((point_count,), float("inf"), dtype=torch.float32)
    owner = torch.full((point_count,), -1, dtype=torch.int32)
    candidate_counts = torch.zeros((point_count,), dtype=torch.int16)
    context_rows: List[Dict[str, Any]] = []
    for index, context in enumerate(contexts):
        started = time.perf_counter()
        ids, p_local, uv_tile = _query_table(
            context,
            q_views[int(context.angle)],
            uvs[int(context.angle)],
            visible[int(context.angle)],
            camera,
        )
        if not ids.numel():
            context_rows.append({"context_id": context.context_id, "angle": context.angle, "tile_id": context.tile_id, "eligible_visible_carriers": 0, "valid_queries": 0})
            continue
        print(f"[decode] {index + 1}/{len(contexts)} view={context.angle} tile={context.tile_id} eligible={ids.numel()}", flush=True)
        shape = mv._sparse_cuda(context.shape_denorm)
        tex = mv._sparse_cuda(states[context.context_id])
        tex = cross_tile._denormalize_slat(tex, pipeline.tex_slat_normalization)
        decoded_list = pipeline.decode_latent(shape, tex, LOCAL_RESOLUTION)
        if len(decoded_list) != 1:
            raise RuntimeError(f"context {context.context_id} decoder returned {len(decoded_list)} meshes")
        decoded = cross_tile._validate_decoded_mesh(decoded_list[0], f"context {context.context_id} final")
        valid_query_count = 0
        local_distance = torch.linalg.vector_norm(uv_tile - uv_tile.new_tensor([511.5, 511.5]), dim=1).cpu()
        for start in range(0, int(ids.shape[0]), int(args.query_chunk_size)):
            stop = min(start + int(args.query_chunk_size), int(ids.shape[0]))
            values = _query_field(decoded, p_local[start:stop].to("cuda"), int(args.query_chunk_size)).cpu()
            finite_values = torch.isfinite(values).all(dim=1)
            if not bool(finite_values.any()):
                continue
            chunk_ids = ids[start:stop][finite_values]
            chunk_values = values[finite_values]
            chunk_dist = local_distance[start:stop][finite_values]
            old_dist = best_distance.index_select(0, chunk_ids)
            replace = chunk_dist < old_dist
            if bool(replace.any()):
                write_ids = chunk_ids[replace]
                final_attrs[write_ids] = chunk_values[replace]
                best_distance[write_ids] = chunk_dist[replace]
                owner[write_ids] = int(context.context_id)
            candidate_counts[chunk_ids] += 1
            valid_query_count += int(chunk_ids.numel())
        row = {
            "context_id": int(context.context_id),
            "angle": int(context.angle),
            "tile_id": int(context.tile_id),
            "eligible_visible_carriers": int(ids.numel()),
            "valid_queries": int(valid_query_count),
            "selected_as_nearest_center": int((owner.index_select(0, ids) == int(context.context_id)).sum()),
            "seconds": time.perf_counter() - started,
            "visibility_gate": "global C4096 z-buffer direct visibility",
            "global_to_local": "continuous exact camera projection/backprojection; no integer ID mapping",
        }
        context_rows.append(row)
        _atomic_json(context.tile_dir / "global_visible_query.json", row)
        del shape, tex, decoded_list, decoded
        _empty_cuda_cache()
    fallback = int((owner < 0).sum())
    chosen = int((owner >= 0).sum())
    _atomic_save(output_dir / "global_c4096_final_pbr.pt", {
        "format": FORMAT,
        "resolution": GLOBAL_RESOLUTION,
        "attrs": final_attrs,
        "owner_context": owner,
        "candidate_counts": candidate_counts,
        "best_tile_center_distance": best_distance,
        "carrier_points": int(point_count),
        "visible_donor_points": chosen,
        "baseline_fallback_points": fallback,
    })
    stats = {
        "carrier_points": point_count,
        "visible_donor_points": chosen,
        "baseline_fallback_points": fallback,
        "visible_donor_fraction": float(chosen / max(1, point_count)),
        "candidate_count_histogram": {str(k): int((candidate_counts == k).sum()) for k in range(0, int(candidate_counts.max().item()) + 1)} if candidate_counts.numel() else {},
        "contexts": context_rows,
        "selection": "among valid direct-visible candidates, minimum distance to current 1024 tile center; otherwise baseline C4096 PBR",
    }
    _atomic_json(output_dir / "global_pbr_assignment.json", stats)
    return final_attrs, stats


@torch.no_grad()
def _build_final_meshes(
    baseline: MeshWithVoxel,
    carrier: Mapping[str, torch.Tensor],
    final_attrs: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[MeshWithVertexPbr, MeshWithFacePbr, Dict[str, Any]]:
    field = _global_field({**carrier, "baseline_attrs": final_attrs}, torch.device("cuda"))
    vertices = baseline.vertices.cpu()
    faces = baseline.faces.cpu()
    vertex_parts: List[torch.Tensor] = []
    for start in range(0, int(vertices.shape[0]), int(args.final_query_chunk_size)):
        vertex_parts.append(_query_field(field, vertices[start : start + int(args.final_query_chunk_size)].to("cuda"), int(args.final_query_chunk_size)).cpu())
    vertex_attrs = torch.cat(vertex_parts, dim=0)
    face_points = vertices.index_select(0, faces.long().reshape(-1)).reshape(-1, 3, 3).mean(dim=1)
    face_parts: List[torch.Tensor] = []
    for start in range(0, int(face_points.shape[0]), int(args.final_query_chunk_size)):
        face_parts.append(_query_field(field, face_points[start : start + int(args.final_query_chunk_size)].to("cuda"), int(args.final_query_chunk_size)).cpu())
    face_attrs = torch.cat(face_parts, dim=0)
    vertex_mesh = MeshWithVertexPbr(vertices, faces, vertex_attrs, layout=dict(PBR_LAYOUT))
    face_mesh = MeshWithFacePbr(vertices, faces, face_attrs, layout=dict(PBR_LAYOUT))
    _atomic_save(output_dir / "final_per_vertex_pbr_mesh.pt", {"format": FORMAT, "representation": "per_vertex_pbr", "mesh": vertex_mesh})
    _atomic_save(output_dir / "final_per_face_pbr_mesh.pt", {"format": FORMAT, "representation": "per_face_pbr", "mesh": face_mesh})
    summary = {
        "geometry_source": "baseline raw_ovoxel_mesh.pt; vertices/faces copied without modification",
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "vertex_geometry_equal_baseline": bool(torch.equal(vertices, baseline.vertices.cpu()) and torch.equal(faces, baseline.faces.cpu())),
        "pbr_source": "global C4096 final carrier queried at baseline vertices/face centroids",
    }
    _atomic_json(output_dir / "final_mesh_summary.json", summary)
    del field, vertex_parts, face_parts
    _empty_cuda_cache()
    return vertex_mesh, face_mesh, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="test_pic/mask_compare_output/image2_resized.png")
    parser.add_argument("--baseline-dir", default="outputs/baseline1024_pbr_mesh_compare")
    parser.add_argument("--output-dir", default="outputs/global_c4096_visible_local_flow_cuda4")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--requested-tile-ids", nargs="+", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-batch-size", type=int, default=1)
    parser.add_argument("--initial-encode-batch-size", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=12)
    parser.add_argument("--pbr-encode-batch-size", type=int, default=13)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument(
        "--cycle-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use x0_pred + (E(P_fused)-E(P_self)); fixed coefficient 1.0",
    )
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--turntable-frames", type=int, default=24)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--carrier-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--final-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--max-overlay-points", type=int, default=500_000)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available() or int(args.cuda_device) >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.cuda_device} is unavailable; count={torch.cuda.device_count()}")
    if args.requested_tile_ids is not None and any(int(v) < 0 or int(v) >= 49 for v in args.requested_tile_ids):
        raise ValueError("requested tile IDs must lie in [0,48]")
    if (int(args.flow_batch_size) <= 0 or int(args.initial_encode_batch_size) <= 0
            or int(args.decode_batch_size) <= 0 or int(args.pbr_encode_batch_size) <= 0):
        raise ValueError("initial/flow/decode/pbr-encode batch sizes must be positive")
    torch.cuda.set_device(int(args.cuda_device))
    _seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    raw_mesh_path = baseline_dir / "raw_ovoxel_mesh.pt"
    if not input_path.is_file() or not raw_mesh_path.is_file() or not (baseline_dir / "summary.json").is_file():
        raise FileNotFoundError(f"missing input/baseline files: {input_path}, {raw_mesh_path}, {baseline_dir / 'summary.json'}")
    camera = _load_camera(baseline_dir)
    _atomic_json(output_dir / "global_camera.json", camera)
    _atomic_json(output_dir / "config.json", {
        "format": FORMAT,
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(int(args.cuda_device)),
        "input": str(input_path),
        "baseline_dir": str(baseline_dir),
        "camera_source": str(baseline_dir / "summary.json"),
        "camera": camera,
        "tile_ids": args.requested_tile_ids,
        "all_views": list(ANGLES),
        "local_geometry": "projected baseline submesh -> exact local camera -> local C1024 voxelize",
        "global_carrier": "sparse conservative C4096 surface voxelization",
        "flow": "global C4096 visible-donor Gaussian flow with isolated multi-context physical batches",
        "flow_batch_size": int(args.flow_batch_size),
        "initial_encode_batch_size": int(args.initial_encode_batch_size),
        "decode_batch_size": int(args.decode_batch_size),
        "pbr_encode_batch_size": int(args.pbr_encode_batch_size),
        "cycle_correction": bool(args.cycle_correction),
        "cycle_correction_coefficient": 1.0 if bool(args.cycle_correction) else 0.0,
        "global_to_local": "continuous point query only; no integer C4096/C1024 mapping",
    })
    baseline = _load_baseline(raw_mesh_path)
    shutil.copy2(raw_mesh_path, output_dir / "baseline_raw_ovoxel_mesh.pt")
    with Image.open(input_path) as source:
        composite = source.convert("RGB")
    views = mv._load_views(input_path, output_dir, ANGLES)
    print(f"[run] CUDA {args.cuda_device}: {torch.cuda.get_device_name(int(args.cuda_device))}", flush=True)
    print(f"[run] baseline vertices={baseline.vertices.shape[0]} faces={baseline.faces.shape[0]}", flush=True)
    carrier = _build_or_load_carrier(baseline, output_dir, int(args.carrier_query_chunk_size))
    _, _, visible, visibility_stats = _build_visibility(carrier, baseline, camera, views, output_dir, args)
    del composite
    _empty_cuda_cache()

    requested_render = bool(args.render)
    context_args = _make_context_args(args, output_dir)
    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    global_attr = _global_field(carrier, torch.device("cuda"))
    print("[contexts] build independent local C1024 tile contexts", flush=True)
    selected_views = {int(angle): views[int(angle)] for angle in ANGLES}
    contexts, dino, _ = mv._build_contexts(context_args, pipeline, baseline, camera, selected_views, output_dir, global_attr)
    del global_attr
    _empty_cuda_cache()
    if not contexts:
        raise RuntimeError("no requested tiles produced local contexts")
    _atomic_json(output_dir / "context_summary.json", {
        "active_contexts": len(contexts),
        "contexts": [{"context_id": c.context_id, "angle": c.angle, "tile_id": c.tile_id, "local_ovoxels": int(c.geometry.coords.shape[0]), "shape_slat": int(c.shape_norm.feats.shape[0])} for c in contexts],
        "dino": dino,
        "condition_fusion": False,
    })
    q_views: Dict[int, torch.Tensor] = {}
    uvs: Dict[int, torch.Tensor] = {}
    # Recreate continuous global-view projections once after context building;
    # no global integer coordinate is ever passed into local flow.
    for angle in ANGLES:
        q_world = carrier["surface_points"] * (2.0 * float(camera["mesh_scale"]))
        q_view = mv._world_to_view_q(q_world, mv._yaw_matrix(int(angle))).cpu()
        uv, _, _ = core._project_global_q_to_image(q_view, global_camera=camera, image_width=1024, image_height=1024)
        q_views[int(angle)] = q_view
        uvs[int(angle)] = uv.cpu()
        del q_world, q_view, uv
    _direct_global_slat_visibility(contexts, camera, output_dir)
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(context_args.pbr_encoder))).eval()
    context_args._pbr_encoder_instance = pbr_encoder
    states = _run_global_flow(
        contexts,
        pipeline,
        context_args,
        output_dir,
        carrier,
        q_views,
        uvs,
        visible,
        camera,
    )
    pbr_encoder.cpu()
    del pbr_encoder
    context_args._pbr_encoder_instance = None
    final_attrs, assignment_stats = _decode_and_write_global(contexts, states, pipeline, carrier, q_views, uvs, visible, camera, context_args, output_dir)
    vertex_mesh, face_mesh, mesh_stats = _build_final_meshes(baseline, carrier, final_attrs, context_args, output_dir)
    render_args = context_args
    render_args.render = requested_render
    render_args.render_resolution = int(args.render_resolution)
    render_args.render_ssaa = int(args.render_ssaa)
    render_args.render_peel_layers = int(args.render_peel_layers)
    render_args.turntable_frames = int(args.turntable_frames)
    render_args.envmap = args.envmap
    render_summary = mv._render_outputs(vertex_mesh, camera, render_args, output_dir)
    summary = {
        "format": FORMAT,
        "status": "complete",
        "config": str(output_dir / "config.json"),
        "camera": camera,
        "active_contexts": len(contexts),
        "carrier_points": int(carrier["coords"].shape[0]),
        "visibility": visibility_stats,
        "assignment": assignment_stats,
        "mesh": mesh_stats,
        "render": render_summary,
        "artifacts": {
            "global_carrier": str(output_dir / "global_c4096_carrier.pt"),
            "global_visibility": str(output_dir / "global_c4096_visibility"),
            "global_final_pbr": str(output_dir / "global_c4096_final_pbr.pt"),
            "vertex_mesh": str(output_dir / "final_per_vertex_pbr_mesh.pt"),
            "face_mesh": str(output_dir / "final_per_face_pbr_mesh.pt"),
            "renders": str(output_dir / "renders"),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] output={output_dir}", flush=True)
    return summary


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
