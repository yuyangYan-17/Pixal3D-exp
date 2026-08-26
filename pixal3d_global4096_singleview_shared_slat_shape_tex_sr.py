#!/usr/bin/env python3
"""Single-view shared-SLat 4096 Shape + Texture super-resolution.

This entry point implements the experiment described in ``Codex.md``.  The
front 4096 image creates one shared support from the baseline 1024 mesh.  All
shape and texture rows are gathered/scattered by that one support.  In the
texture stage Exp-B keeps the global image condition from the front tile and
selects only the sparse projection rows from the baseline-material back tile
when the frozen baseline triangle visibility says that a row is front
invisible.

The old multiview scripts are intentionally not imported into the flow path.
The existing tile support, native decoder and instantaneous ``pred_x_0``
consensus implementation are reused as low-level single-view components.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Sparse CUDA extensions read these before torch/model imports.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image

import pixal3d_global4096_tile_endpoint_rollout_sync as legacy
import pixal3d_global4096_tile_x0_consensus_sync as x0_route
import pixal3d_render_global4096_multiview as multiview_render
import pixal3d_singleview_shared_slat_support as singleview_support
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global4096_singleview_shared_slat_shape_tex_sr_cuda4_v1"
CANONICAL_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
TILE_COUNT = 49
LOCAL_OVOXEL = 1024
LATENT_SIZE = 64
FLOW_BATCH_SIZE = 44
DECODE_BATCH_SIZE = 12
CONDITION_BATCH_SIZE = 1

DEFAULT_IMAGE = Path("/home/nvme04/yyyan/Pixal3D/assets/images/0_img.png")
DEFAULT_BASELINE_DIR = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/global4096_tile_x0_consensus_sync_cuda5/baseline"
)
DEFAULT_SUPPORT_SOURCE = DEFAULT_BASELINE_DIR.parent / "support"
DEFAULT_C256_SOURCE = Path(
    "/home/nvme04/yyyan/Pixal3D/used/experiments_archive_20260802/outputs/"
    "global1024_to_c256_hr_tile_velocity_average/seed_42"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_singleview_shared_slat_shape_tex_sr_cuda4"
)
DEFAULT_ENCODER_ROOT = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts"
)
DEFAULT_SHAPE_ENCODER = DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"


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
    if isinstance(value, float) and not np.isfinite(value):
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


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _logical_cuda_device(physical_device: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the single-view shared-SLat experiment")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        ids = [part.strip() for part in visible.split(",") if part.strip()]
        if ids != [str(int(physical_device))]:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={visible!r} must expose only physical CUDA {physical_device}"
            )
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if int(physical_device) < 0 or int(physical_device) >= torch.cuda.device_count():
        raise RuntimeError(f"physical CUDA {physical_device} is unavailable")
    torch.cuda.set_device(int(physical_device))
    return torch.device("cuda", int(physical_device))


def _runtime(device: torch.device, physical_device: int) -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "physical_cuda_device_requested": int(physical_device),
        "cuda_visible_devices": visible,
        "logical_device": str(device),
        "current_device": int(torch.cuda.current_device()),
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "free_memory_bytes": int(torch.cuda.mem_get_info(device)[0]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_smi": _nvidia_smi(),
    }


def _nvidia_smi() -> Optional[str]:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return None


def _configure_routes() -> None:
    """Reuse the validated low-level route with a new format namespace."""
    legacy.FORMAT = FORMAT
    legacy.SHAPE_IMAGE_SIZE = TILE_SIZE
    legacy.TEXTURE_CONDITION_BATCH_SIZE = CONDITION_BATCH_SIZE
    legacy.FLOW_BATCH_SIZE = FLOW_BATCH_SIZE
    legacy.DECODE_BATCH_SIZE = DECODE_BATCH_SIZE
    x0_route.FORMAT = FORMAT
    x0_route.FLOW_BATCH_SIZE = FLOW_BATCH_SIZE
    x0_route.DECODE_BATCH_SIZE = DECODE_BATCH_SIZE
    x0_route.TEXTURE_CONDITION_BATCH_SIZE = CONDITION_BATCH_SIZE
    # x0_route delegates packing/serialization helpers to this module.
    x0_route._legacy.FORMAT = FORMAT


def _load_support_cache(
    source_dir: Path,
    transforms: Mapping[int, Any],
) -> legacy.MasterSupport:
    support_dir = Path(source_dir)
    master_path = support_dir / "master_support.pt"
    if not master_path.is_file():
        raise FileNotFoundError(master_path)
    payload = torch.load(master_path, map_location="cpu", weights_only=False)
    q = payload.get("master_q_world", payload.get("master_q_global"))
    uv = payload.get("front_uv_4096", payload.get("master_uv_4096"))
    owner = payload.get("owner_front_tile_id", payload.get("owner_tile_id"))
    owner_coord = payload.get("owner_front_local_coord", payload.get("owner_local_coord_c64"))
    if not all(isinstance(value, torch.Tensor) for value in (q, uv, owner, owner_coord)):
        raise RuntimeError(f"support cache {master_path} does not contain the required master schema")
    q = q.to(torch.float32).contiguous()
    uv = uv.to(torch.float32).contiguous()
    owner = owner.to(torch.int16).contiguous()
    owner_coord = owner_coord.to(torch.int32).contiguous()
    if q.ndim != 2 or q.shape[1] != 3 or uv.shape != (q.shape[0], 2):
        raise RuntimeError(f"support cache has invalid q/uv shapes: {q.shape}, {uv.shape}")
    tile_views: Dict[int, legacy.TileView] = {}
    tile_stats: Dict[int, Dict[str, Any]] = {}
    views_dir = support_dir / "tile_views"
    for tile_id, box in enumerate(legacy._tile_layout()):
        path = views_dir / f"tile_{tile_id:02d}.pt"
        if not path.is_file():
            tile_stats[tile_id] = {"status": "inactive", "reason": "cache_missing"}
            continue
        row = torch.load(path, map_location="cpu", weights_only=False)
        ids = row["master_ids"].to(torch.int64).contiguous()
        coords = row.get("local_coords_c64", row.get("local_coords")).to(torch.int32).contiguous()
        tile_uv = row.get("master_uv_4096", row.get("master_uv_world")).to(torch.float32).contiguous()
        weights = row.get("gaussian_weight")
        if weights is None:
            weights = legacy.gaussian_weights(tile_uv, box, legacy.SIGMA_PIXELS)
        weights = weights.to(torch.float32).contiguous()
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] != ids.numel():
            raise RuntimeError(f"support cache tile {tile_id} has misaligned ids/coords")
        if bool((ids < 0).any()) or bool((ids >= q.shape[0]).any()):
            raise RuntimeError(f"support cache tile {tile_id} has invalid master ids")
        view = legacy.TileView(
            tile_id=tile_id,
            box=tuple(int(value) for value in box),
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
        raise RuntimeError(f"support cache {support_dir} contains no active tile views")
    coverage = torch.zeros((q.shape[0],), dtype=torch.int32)
    for view in tile_views.values():
        coverage.index_add_(0, view.master_ids, torch.ones_like(view.master_ids, dtype=torch.int32))
    if bool((coverage <= 0).any()):
        raise RuntimeError("support cache does not cover every master row")
    return legacy.MasterSupport(
        master_q_global=q,
        master_uv_4096=uv,
        owner_tile_id=owner,
        owner_local_coord_c64=owner_coord,
        tile_views=tile_views,
        tile_stats=tile_stats,
        collision_report=[],
        roundtrip_max_abs_error=0.0,
    )


def _write_support_schema(
    support: legacy.MasterSupport,
    output_dir: Path,
    transforms: Mapping[int, Any],
    source: Optional[Path],
) -> None:
    legacy._save_master_support(support, output_dir, transforms)
    path = Path(output_dir) / "support" / "master_support.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    master_ids = torch.arange(support.master_q_global.shape[0], dtype=torch.int64)
    payload.update(
        {
            "format": FORMAT,
            "master_id": master_ids,
            "master_q_world": support.master_q_global,
            "master_q_global": support.master_q_global,
            "owner_front_tile_id": support.owner_tile_id,
            "owner_front_local_coord": support.owner_local_coord_c64,
            "front_uv_4096": support.master_uv_4096,
            "encoder_feature_values_present": False,
            "support_source": None if source is None else str(source.resolve()),
        }
    )
    _atomic_save(path, payload)
    coverage = torch.zeros((master_ids.numel(),), dtype=torch.int32)
    local_rows: Dict[str, int] = {}
    for tile_id, view in support.tile_views.items():
        coverage.index_add_(0, view.master_ids, torch.ones_like(view.master_ids, dtype=torch.int32))
        local_rows[str(tile_id)] = int(view.master_ids.numel())
    _atomic_json(
        Path(output_dir) / "support" / "support_diagnostics.json",
        {
            "format": FORMAT,
            "master_rows": int(master_ids.numel()),
            "active_tiles": sorted(int(key) for key in support.tile_views),
            "inactive_tiles": [tile for tile in range(TILE_COUNT) if tile not in support.tile_views],
            "local_rows_by_tile": local_rows,
            "overlap_rows": int((coverage > 1).sum()),
            "overlap_ratio": float((coverage > 1).float().mean()),
            "coverage_min": int(coverage.min()),
            "coverage_max": int(coverage.max()),
            "roundtrip_max_abs_error": float(support.roundtrip_max_abs_error),
            "support_identity": "single front-view shared SLat; first-owner overlap inheritance",
        },
    )


def _make_transforms(camera: Mapping[str, float]) -> Dict[int, Any]:
    return {
        tile_id: legacy.core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=camera,
            extend_pixel=0,
            source_width=CANONICAL_SIZE,
            source_height=CANONICAL_SIZE,
            model_width=TILE_SIZE,
            model_height=TILE_SIZE,
        )
        for tile_id, box in enumerate(legacy._tile_layout())
    }


def _prepare_support(
    *,
    args: argparse.Namespace,
    baseline: Any,
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
) -> Tuple[legacy.MasterSupport, Dict[int, Any], Dict[str, Any]]:
    transforms = _make_transforms(camera)
    source = Path(args.support_source).expanduser().resolve() if args.support_source else None
    if source is not None and (source / "master_support.pt").is_file():
        print(f"[support] reuse coordinate-only cache: {source}", flush=True)
        support = _load_support_cache(source, transforms)
        source_value: Optional[Path] = source
    else:
        print("[support] building single-view support from baseline mesh", flush=True)
        face_bounds = legacy._face_bounds(
            baseline,
            camera,
            output_dir / "support" / "face_projection_bounds.pt",
            int(args.face_projection_chunk_size),
        )
        geometries = legacy._prepare_native_geometries(
            baseline,
            camera,
            transforms,
            face_bounds,
            output_dir,
            list(range(TILE_COUNT)),
            int(args.face_projection_chunk_size),
        )
        native = legacy._encode_native_supports(
            geometries,
            Path(args.shape_encoder),
            output_dir,
            device,
            int(args.native_geometry_encode_batch_size),
        )
        support = legacy._build_master_support(
            native,
            transforms,
            camera,
            sigma_pixels=legacy.SIGMA_PIXELS,
        )
        source_value = None
    _write_support_schema(support, output_dir, transforms, source_value)
    _atomic_json(Path(output_dir) / "tile_cameras.json", {str(k): transforms[k].__dict__ for k in transforms})
    return support, transforms, {
        "source": None if source_value is None else str(source_value),
        "master_rows": int(support.master_q_global.shape[0]),
        "active_tiles": sorted(int(key) for key in support.tile_views),
        "inactive_tiles": [tile for tile in range(TILE_COUNT) if tile not in support.tile_views],
        "local_rows_by_tile": {str(key): int(value.master_ids.numel()) for key, value in support.tile_views.items()},
    }


def _tile_images(image: Image.Image, output_dir: Path, name: str) -> Dict[int, Image.Image]:
    root = Path(output_dir) / "inputs" / name
    root.mkdir(parents=True, exist_ok=True)
    images: Dict[int, Image.Image] = {}
    for tile_id, box in enumerate(singleview_support.tile_boxes()):
        crop = image.crop(box).convert("RGB")
        crop.save(root / f"tile_{tile_id:02d}.png")
        images[tile_id] = crop
    return images


def _build_visibility(
    *,
    args: argparse.Namespace,
    baseline: Any,
    camera: Mapping[str, float],
    support: legacy.MasterSupport,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    source = Path(args.visibility_source).expanduser().resolve() if args.visibility_source else None
    master_count = int(support.master_q_global.shape[0])
    if source is not None and (source / "frozen_visibility.pt").is_file():
        payload = torch.load(source / "frozen_visibility.pt", map_location="cpu", weights_only=False)
        visible = payload.get("visible")
        mapping = payload.get("mapping_valid")
        nearest_face = payload.get("nearest_face_id")
        if (
            not isinstance(visible, torch.Tensor)
            or visible.shape != (TILE_COUNT, master_count)
            or not isinstance(mapping, torch.Tensor)
            or mapping.shape != visible.shape
            or not isinstance(nearest_face, torch.Tensor)
            or nearest_face.shape[0] != master_count
        ):
            raise RuntimeError(f"visibility cache {source} is not aligned with the current support")
        print(f"[visibility] reuse frozen front table: {source}", flush=True)
        target = Path(output_dir) / "support"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("frozen_visibility.pt", "face_visibility_per_context.pt", "visibility_stats.json", "master_nearest_triangle.pt"):
            source_file = source / name
            if source_file.is_file():
                shutil.copy2(source_file, target / name)
        tile_stats = {}
        for tile_id, view in support.tile_views.items():
            flags = visible[tile_id, view.master_ids]
            tile_stats[str(tile_id)] = {
                "status": "active",
                "mapping_count": int(view.master_ids.numel()),
                "front_visible_count": int(flags.sum()),
                "front_invisible_count": int((~flags).sum()),
            }
        return {
            "visible": visible.to(torch.bool),
            "mapping_valid": mapping.to(torch.bool),
            "nearest": {"nearest_face_id": nearest_face.to(torch.int64)},
            "tile_stats": tile_stats,
            "source": str(source),
        }
    values = singleview_support.build_front_visibility(
        baseline=baseline,
        camera=camera,
        views=support.tile_views,
        master_q_world=support.master_q_global,
        output_dir=output_dir,
        device=device,
        face_chunk_size=int(args.visibility_face_chunk_size),
    )
    master_path = Path(output_dir) / "support" / "master_support.pt"
    master_payload = torch.load(master_path, map_location="cpu", weights_only=False)
    nearest = values["nearest"]
    master_payload.update(
        {
            "baseline_nearest_face_id": nearest["nearest_face_id"],
            "baseline_nearest_point": nearest["nearest_point"],
            "baseline_nearest_bary": nearest["nearest_bary"],
            "baseline_face_distance": nearest["face_distance"],
        }
    )
    _atomic_save(master_path, master_payload)
    values["source"] = "current-run baseline nvdiffrast raster"
    return values


def _build_conditions(
    *,
    pipeline: Any,
    support: legacy.MasterSupport,
    front_tiles: Mapping[int, Image.Image],
    back_tiles: Optional[Mapping[int, Image.Image]],
    visibility_payload: Mapping[str, Any],
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    mode: str,
    condition_batch_size: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    front_texture = legacy._build_batched_image_conditions(
        pipeline,
        pipeline.image_cond_model_tex_1024,
        support.tile_views,
        front_tiles,
        output_dir,
        "texture_front",
        camera,
        device,
        CONDITION_BATCH_SIZE,
    )
    back_texture = None
    if back_tiles is not None:
        back_texture = legacy._build_batched_image_conditions(
            pipeline,
            pipeline.image_cond_model_tex_1024,
            support.tile_views,
            back_tiles,
            output_dir,
            "texture_back",
            camera,
            device,
            CONDITION_BATCH_SIZE,
        )
    visibility_by_tile = {
        tile_id: visibility_payload["visible"][tile_id, view.master_ids].to(torch.bool)
        for tile_id, view in support.tile_views.items()
    }
    texture_conditions, route_stats = singleview_support.route_texture_conditions(
        views=support.tile_views,
        front_conditions=front_texture,
        back_conditions=back_texture,
        visibility_by_tile=visibility_by_tile,
        output_dir=output_dir,
        mode=mode,
    )
    return texture_conditions, {
        "front_texture_condition_tiles": len(front_texture),
        "back_texture_condition_tiles": 0 if back_texture is None else len(back_texture),
        "routing": route_stats,
    }


def _master_coords(count: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros((count, 1), dtype=torch.int32),
            legacy._master_index_coords(count),
        ),
        dim=1,
    )


def _run_flows(
    *,
    pipeline: Any,
    support: legacy.MasterSupport,
    shape_conditions: Mapping[int, Mapping[str, Any]],
    texture_conditions: Mapping[int, Mapping[str, Any]],
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    count = int(support.master_q_global.shape[0])
    master_coords = _master_coords(count)
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    shape_channels = int(shape_model.in_channels)
    texture_channels = int(texture_model.in_channels) - shape_channels
    if texture_channels <= 0:
        raise RuntimeError(
            f"texture model channels {texture_model.in_channels} do not exceed shape channels {shape_channels}"
        )
    shape_noise = torch.randn(
        (count, shape_channels),
        generator=torch.Generator(device="cpu").manual_seed(int(args.shape_seed)),
        dtype=torch.float32,
    )
    texture_noise = torch.randn(
        (count, texture_channels),
        generator=torch.Generator(device="cpu").manual_seed(int(args.texture_seed)),
        dtype=torch.float32,
    )
    _atomic_save(output_dir / "shape" / "shape_noise.pt", {"master_id": torch.arange(count), "features": shape_noise})
    _atomic_save(output_dir / "texture" / "texture_noise.pt", {"master_id": torch.arange(count), "features": texture_noise})
    shape_params = dict(pipeline.shape_slat_sampler_params)
    texture_params = dict(pipeline.tex_slat_sampler_params)
    if args.steps is not None:
        shape_params["steps"] = int(args.steps)
        texture_params["steps"] = int(args.steps)
    shape_final, shape_summary = x0_route._run_synchronized_x0_consensus_flow(
        stage="shape",
        initial_features=shape_noise,
        master_coords=master_coords,
        views=support.tile_views,
        conditions=shape_conditions,
        sampler=pipeline.shape_slat_sampler,
        model=shape_model,
        sampler_params=shape_params,
        output_dir=output_dir,
        device=device,
        flow_batch_size=int(args.flow_batch_size),
        resume=bool(args.resume),
        save_step_tensors=bool(args.save_step_tensors),
    )
    texture_final, texture_summary = x0_route._run_synchronized_x0_consensus_flow(
        stage="texture",
        initial_features=texture_noise,
        master_coords=master_coords,
        views=support.tile_views,
        conditions=texture_conditions,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        sampler_params=texture_params,
        output_dir=output_dir,
        device=device,
        flow_batch_size=int(args.flow_batch_size),
        concat_features=shape_final,
        resume=bool(args.resume),
        save_step_tensors=bool(args.save_step_tensors),
    )
    return shape_final, texture_final, shape_summary, texture_summary


def _render_final_and_metrics(
    *,
    output_dir: Path,
    camera_path: Path,
    mesh_path: Path,
    canonical: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    render_dir = output_dir / "multiview"
    render_args = SimpleNamespace(
        mesh=mesh_path,
        camera=camera_path,
        output_dir=render_dir,
        angles="0,120,240",
        resolution=int(args.render_resolution),
        face_chunk_size=int(args.render_face_chunk_size),
        device=str(device),
        force=False,
    )
    manifest = multiview_render.render(render_args)
    front_rgb = render_dir / "view_000_render_rgb_4096.png"
    front_alpha = render_dir / "view_000_render_alpha_4096.png"
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if front_rgb.is_file():
        shutil.copyfile(front_rgb, final_dir / "final_render_rgb_4096.png")
    if front_alpha.is_file():
        shutil.copyfile(front_alpha, final_dir / "final_render_alpha_4096.png")
    # Keep the existing repository metric schema and compare only the aligned
    # front view against the native preprocessed 4096 reference.
    legacy._compute_global_metrics(
        canonical["image_4096"],
        canonical["foreground_mask_4096"],
        output_dir,
        {
            "final": {
                "rgb_path": str(front_rgb),
                "alpha_path": str(front_alpha),
            }
        },
    )
    return {
        "manifest": manifest,
        "front_render": str(front_rgb),
        "front_alpha": str(front_alpha),
        "metrics": str(output_dir / "metrics_4096.json"),
    }


def _slat_payload(slat: SparseTensor) -> Dict[str, Any]:
    """Serialize a SLat without depending on the active sparse backend."""
    return {
        "coords": slat.coords.detach().cpu(),
        "feats": slat.feats.detach().cpu(),
    }


def _slat_stats(slat: SparseTensor, grid_resolution: int, stage: str) -> Dict[str, Any]:
    coords = slat.coords.detach()
    xyz = coords[:, 1:]
    return {
        "stage": stage,
        "grid_resolution": int(grid_resolution),
        "tokens": int(coords.shape[0]),
        "channels": int(slat.feats.shape[1]),
        "coord_min": [int(value) for value in xyz.amin(dim=0).cpu().tolist()],
        "coord_max": [int(value) for value in xyz.amax(dim=0).cpu().tolist()],
        "batch_ids": [int(value) for value in coords[:, 0].unique().cpu().tolist()],
    }


def _coord_stats(coords: torch.Tensor, grid_resolution: int, stage: str) -> Dict[str, Any]:
    coords = coords.detach()
    xyz = coords[:, 1:]
    return {
        "stage": stage,
        "grid_resolution": int(grid_resolution),
        "tokens": int(coords.shape[0]),
        "coord_min": [int(value) for value in xyz.amin(dim=0).cpu().tolist()],
        "coord_max": [int(value) for value in xyz.amax(dim=0).cpu().tolist()],
        "batch_ids": [int(value) for value in coords[:, 0].unique().cpu().tolist()],
    }


def _normalization_tensors(
    normalization: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    std = torch.as_tensor(normalization["std"], device=device, dtype=dtype)[None]
    mean = torch.as_tensor(normalization["mean"], device=device, dtype=dtype)[None]
    return std, mean


def _denormalize_slat(pipeline: Any, slat: SparseTensor) -> SparseTensor:
    std, mean = _normalization_tensors(
        pipeline.shape_slat_normalization,
        slat.device,
        slat.feats.dtype,
    )
    return slat.replace(slat.feats * std + mean)


def _normalize_slat(pipeline: Any, slat: SparseTensor) -> SparseTensor:
    std, mean = _normalization_tensors(
        pipeline.shape_slat_normalization,
        slat.device,
        slat.feats.dtype,
    )
    return slat.replace((slat.feats - mean) / std)


def _make_native_noise(
    token_count: int,
    channels: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        (int(token_count), int(channels)),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def _decoder_upsample_once(
    pipeline: Any,
    slat: SparseTensor,
    device: torch.device,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(device)
        decoder.low_vram = True
    coords = decoder.upsample(slat, upsample_times=1).int()
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise RuntimeError(f"decoder upsample returned invalid coords {tuple(coords.shape)}")
    if torch.any(coords[:, 0] != 0):
        raise RuntimeError("native Exp-C only supports the single input batch")
    return coords


def _run_native_shape_flow(
    *,
    pipeline: Any,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    coords: torch.Tensor,
    grid_resolution: int,
    output_dir: Path,
    stage_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    model = pipeline.models["shape_slat_flow_model_1024"]
    cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=int(grid_resolution),
    )
    noise_features = _make_native_noise(
        coords.shape[0],
        int(model.in_channels),
        seed,
        device,
    )
    noise = SparseTensor(feats=noise_features, coords=coords)
    _atomic_save(
        output_dir / "shape" / stage_name / "noise.pt",
        {"coords": coords.detach().cpu(), "features": noise_features.detach().cpu()},
    )
    sampler_params = dict(pipeline.shape_slat_sampler_params)
    if args.steps is not None:
        sampler_params["steps"] = int(args.steps)
    if pipeline.low_vram:
        model.to(device)
    started = time.perf_counter()
    sampled = pipeline.shape_slat_sampler.sample(
        model,
        noise,
        **cond,
        **sampler_params,
        verbose=True,
        tqdm_desc=f"Sampling native {stage_name} shape SLat (proj, {grid_resolution})",
    ).samples
    if pipeline.low_vram:
        model.cpu()
    slat = _denormalize_slat(pipeline, sampled)
    _atomic_save(
        output_dir / "shape" / stage_name / "final_state.pt",
        _slat_payload(slat),
    )
    summary = _slat_stats(slat, grid_resolution, stage_name)
    summary.update(
        {
            "flow_model": "shape_slat_flow_model_1024",
            "condition_model": "image_cond_model_shape_1024",
            "condition_image": "canonical image_1024",
            "sampler_params": sampler_params,
            "seed": int(seed),
            "seconds": float(time.perf_counter() - started),
            "mode": "native_global_sparse_flow",
        }
    )
    _atomic_json(output_dir / "shape" / stage_name / "summary.json", summary)
    del cond, noise, sampled, noise_features
    _empty_cuda_cache()
    return slat, summary


def _run_native_texture_flow(
    *,
    pipeline: Any,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    shape_slat: SparseTensor,
    grid_resolution: int,
    output_dir: Path,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    model = pipeline.models["tex_slat_flow_model_1024"]
    cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024],
        shape_slat.coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=int(grid_resolution),
    )
    shape_normalized = _normalize_slat(pipeline, shape_slat)
    texture_channels = int(model.in_channels) - int(shape_normalized.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(
            f"texture model channels {model.in_channels} do not exceed "
            f"shape channels {shape_normalized.feats.shape[1]}"
        )
    noise_features = _make_native_noise(
        shape_normalized.coords.shape[0],
        texture_channels,
        seed,
        device,
    )
    noise = shape_normalized.replace(feats=noise_features)
    _atomic_save(
        output_dir / "texture" / "c256_flow" / "noise.pt",
        {
            "coords": shape_normalized.coords.detach().cpu(),
            "features": noise_features.detach().cpu(),
        },
    )
    sampler_params = dict(pipeline.tex_slat_sampler_params)
    if args.steps is not None:
        sampler_params["steps"] = int(args.steps)
    if pipeline.low_vram:
        model.to(device)
    started = time.perf_counter()
    sampled = pipeline.tex_slat_sampler.sample(
        model,
        noise,
        concat_cond=shape_normalized,
        **cond,
        **sampler_params,
        verbose=True,
        tqdm_desc="Sampling native C256 texture SLat (proj)",
    ).samples
    if pipeline.low_vram:
        model.cpu()
    tex_std, tex_mean = _normalization_tensors(
        pipeline.tex_slat_normalization,
        sampled.device,
        sampled.feats.dtype,
    )
    tex_slat = sampled.replace(sampled.feats * tex_std + tex_mean)
    _atomic_save(
        output_dir / "texture" / "c256_flow" / "final_state.pt",
        _slat_payload(tex_slat),
    )
    summary = _slat_stats(tex_slat, grid_resolution, "texture_c256_flow")
    summary.update(
        {
            "flow_model": "tex_slat_flow_model_1024",
            "condition_model": "image_cond_model_tex_1024",
            "condition_image": "canonical image_1024",
            "concat_cond": "shape_c256_flow_final_normalized",
            "sampler_params": sampler_params,
            "seed": int(seed),
            "seconds": float(time.perf_counter() - started),
            "mode": "native_global_sparse_flow",
        }
    )
    _atomic_json(output_dir / "texture" / "c256_flow" / "summary.json", summary)
    del cond, shape_normalized, noise, sampled, noise_features
    _empty_cuda_cache()
    return tex_slat, summary


def _native_mesh_to_pbr(
    mesh: Any,
    device: torch.device,
) -> Tuple[Any, Any]:
    live = mesh.to(device)
    vertices = live.vertices.detach().cpu().float()
    faces = live.faces.detach().cpu().int()
    vertex_attrs = live.query_vertex_attrs().detach().cpu().float()
    face_indices = live.faces.to(torch.long)
    face_centers = live.vertices.index_select(0, face_indices.reshape(-1)).reshape(-1, 3, 3).mean(dim=1)
    face_attrs = live.query_attrs(face_centers).detach().cpu().float()
    vertex_mesh = legacy.MeshWithVertexPbr(vertices, faces, vertex_attrs, layout=dict(legacy.PBR_LAYOUT))
    face_mesh = legacy.MeshWithFacePbr(vertices, faces, face_attrs, layout=dict(legacy.PBR_LAYOUT))
    return vertex_mesh, face_mesh


def _run_native_cascade_exp_c(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Run the native Pixal3D C64→C128→C256→4096 cascade requested for Exp-C."""
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    device = _logical_cuda_device(int(args.cuda_device))
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not bool(args.resume):
        raise RuntimeError(
            f"refusing to reuse nonempty output {output_dir}; choose a new --output-dir or pass --resume"
        )
    started = time.perf_counter()
    _seed(int(args.seed))
    runtime = _runtime(device, int(args.cuda_device))
    print(
        f"[run] experiment=baseline4096_from1024 physical_cuda={args.cuda_device} "
        f"logical={device} output={output_dir}",
        flush=True,
    )
    pipeline = init_pipeline(
        str(args.model_path),
        device=str(device),
        low_vram=bool(args.low_vram),
    )
    source_image = Image.open(args.image).convert("RGB")
    canonical = pipeline.preprocess_canonical_images(source_image)
    preprocess_meta = legacy._save_canonical_images(canonical, output_dir)
    camera = legacy._load_camera(Path(args.baseline_dir).expanduser().resolve())
    _atomic_json(output_dir / "global_camera.json", camera)

    shape_params = {
        "steps": int(args.steps) if args.steps is not None else 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    texture_params = {
        "steps": int(args.steps) if args.steps is not None else 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    print("[exp-c] official baseline 1024_cascade → C64", flush=True)
    baseline_meshes, baseline_latents = pipeline.run(
        canonical["image_1024"],
        camera_params=dict(camera),
        seed=int(args.seed),
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=1_000_000,
    )
    if len(baseline_meshes) != 1:
        raise RuntimeError(f"official baseline returned {len(baseline_meshes)} meshes")
    baseline_mesh = baseline_meshes[0].cpu()
    shape_c64, texture_c64, baseline_resolution = baseline_latents
    if int(baseline_resolution) != 1024:
        raise RuntimeError(f"official baseline returned resolution {baseline_resolution}, expected 1024")
    _atomic_save(
        output_dir / "baseline" / "baseline_1024_mesh.pt",
        {"format": FORMAT, "mesh": baseline_mesh},
    )
    _atomic_save(
        output_dir / "baseline" / "baseline_c64_latents.pt",
        {
            "shape_slat": _slat_payload(shape_c64),
            "texture_slat": _slat_payload(texture_c64),
            "resolution": int(baseline_resolution),
        },
    )
    baseline_shape_stats = _slat_stats(shape_c64, 64, "shape_c64_baseline")
    baseline_texture_stats = _slat_stats(texture_c64, 64, "texture_c64_baseline")
    print(f"[exp-c] baseline C64 tokens={baseline_shape_stats['tokens']:,}", flush=True)

    print("[exp-c] decoder upsample(1): C64 → C128 support", flush=True)
    coords_c128 = _decoder_upsample_once(pipeline, shape_c64, device)
    if int(coords_c128[:, 1:].amax().item()) >= 128:
        raise RuntimeError("C128 upsample coordinates exceed the 128 grid")
    coords_c128_stats = _coord_stats(coords_c128, 128, "upsample_c64_to_c128")
    _atomic_save(output_dir / "shape" / "upsample_c128" / "coords.pt", {"coords": coords_c128.detach().cpu()})
    _atomic_json(output_dir / "shape" / "upsample_c128" / "summary.json", coords_c128_stats)
    shape_c128, shape_c128_summary = _run_native_shape_flow(
        pipeline=pipeline,
        image_1024=canonical["image_1024"],
        camera=camera,
        coords=coords_c128,
        grid_resolution=128,
        output_dir=output_dir,
        stage_name="c128_flow",
        seed=int(args.shape_seed),
        args=args,
        device=device,
    )
    del coords_c128, shape_c64
    _empty_cuda_cache()

    print("[exp-c] decoder upsample(1): C128 → C256 support", flush=True)
    coords_c256 = _decoder_upsample_once(pipeline, shape_c128, device)
    if int(coords_c256[:, 1:].amax().item()) >= 256:
        raise RuntimeError("C256 upsample coordinates exceed the 256 grid")
    coords_c256_stats = _coord_stats(coords_c256, 256, "upsample_c128_to_c256")
    _atomic_save(output_dir / "shape" / "upsample_c256" / "coords.pt", {"coords": coords_c256.detach().cpu()})
    _atomic_json(output_dir / "shape" / "upsample_c256" / "summary.json", coords_c256_stats)
    shape_c256, shape_c256_summary = _run_native_shape_flow(
        pipeline=pipeline,
        image_1024=canonical["image_1024"],
        camera=camera,
        coords=coords_c256,
        grid_resolution=256,
        output_dir=output_dir,
        stage_name="c256_flow",
        seed=int(args.shape_seed) + 1,
        args=args,
        device=device,
    )
    del coords_c256
    _empty_cuda_cache()

    print("[exp-c] native C256 texture flow", flush=True)
    texture_c256, texture_c256_summary = _run_native_texture_flow(
        pipeline=pipeline,
        image_1024=canonical["image_1024"],
        camera=camera,
        shape_slat=shape_c256,
        grid_resolution=256,
        output_dir=output_dir,
        args=args,
        seed=int(args.texture_seed),
        device=device,
    )
    print("[exp-c] native C256 decode → 4096", flush=True)
    decode_started = time.perf_counter()
    decoded = pipeline.decode_latent(shape_c256, texture_c256, 4096)
    if len(decoded) != 1:
        raise RuntimeError(f"native 4096 decoder returned {len(decoded)} meshes")
    native_mesh = decoded[0]
    _atomic_save(
        output_dir / "final" / "final_material_mesh.pt",
        {"format": FORMAT, "mesh": native_mesh.cpu()},
    )
    vertex_mesh, face_mesh = _native_mesh_to_pbr(native_mesh, device)
    _atomic_save(
        output_dir / "final" / "final_per_vertex_pbr_mesh.pt",
        {"format": FORMAT, "mesh": vertex_mesh},
    )
    _atomic_save(
        output_dir / "final" / "final_per_face_pbr_mesh.pt",
        {"format": FORMAT, "mesh": face_mesh},
    )
    decode_summary = {
        "resolution": 4096,
        "shape_grid_resolution": 256,
        "texture_grid_resolution": 256,
        "vertices": int(vertex_mesh.vertices.shape[0]),
        "faces": int(vertex_mesh.faces.shape[0]),
        "seconds": float(time.perf_counter() - decode_started),
        "mode": "pipeline.decode_latent(native_c256, native_c256, 4096)",
    }
    _atomic_json(output_dir / "decode" / "summary.json", decode_summary)
    del shape_c256, texture_c256, native_mesh
    _empty_cuda_cache()

    render_meta: Dict[str, Any] = {}
    if bool(args.render):
        render_meta = _render_final_and_metrics(
            output_dir=output_dir,
            camera_path=output_dir / "global_camera.json",
            mesh_path=output_dir / "final" / "final_per_vertex_pbr_mesh.pt",
            canonical=canonical,
            device=device,
            args=args,
        )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "experiment": "baseline4096_from1024",
        "input": str(Path(args.image).expanduser().resolve()),
        "input_resolution": 1024,
        "output_resolution": 4096,
        "baseline_dir": str(Path(args.baseline_dir).expanduser().resolve()),
        "camera": camera,
        "runtime": runtime,
        "cuda_policy": "physical CUDA 4 only; logical cuda:0",
        "seed": {
            "baseline": int(args.seed),
            "shape_c128": int(args.shape_seed),
            "shape_c256": int(args.shape_seed) + 1,
            "texture_c256": int(args.texture_seed),
        },
        "preprocess": preprocess_meta,
        "baseline": {
            "pipeline_type": "1024_cascade",
            "input_image": "canonical image_1024",
            "resolution": 1024,
            "shape": baseline_shape_stats,
            "texture": baseline_texture_stats,
            "mesh": str(output_dir / "baseline" / "baseline_1024_mesh.pt"),
            "latents": str(output_dir / "baseline" / "baseline_c64_latents.pt"),
        },
        "cascade": {
            "requested_path": "C64 shape SLat → decoder.upsample(1) → C128 shape flow → decoder.upsample(1) → C256 shape flow → C256 texture flow → native 4096 decode",
            "upsample_c64_to_c128": coords_c128_stats,
            "shape_c128_flow": shape_c128_summary,
            "upsample_c128_to_c256": coords_c256_stats,
            "shape_c256_flow": shape_c256_summary,
            "texture_c256_flow": texture_c256_summary,
            "decode_4096": decode_summary,
        },
        "support": {
            "c64_tokens": baseline_shape_stats["tokens"],
            "c128_tokens": coords_c128_stats["tokens"],
            "c256_tokens": coords_c256_stats["tokens"],
            "coordinate_source": "shape_slat_decoder.upsample(upsample_times=1) at both transitions",
        },
        "routing": {
            "global": "canonical image_1024",
            "proj": "canonical image_1024 at native stage grid",
            "front_proj_rows": int(coords_c256_stats["tokens"]),
            "back_proj_rows": 0,
            "front_proj_fraction": 1.0,
            "back_proj_fraction": 0.0,
            "uses_4096_gt_tiles": False,
            "uses_back_image": False,
            "single_texture_state": True,
        },
        "shape_flow": {
            "c128": shape_c128_summary,
            "c256": shape_c256_summary,
        },
        "texture_flow": texture_c256_summary,
        "decode": decode_summary,
        "render": render_meta,
        "timing": {"total_seconds": float(time.perf_counter() - started)},
        "artifacts": {
            "baseline_latents": str(output_dir / "baseline" / "baseline_c64_latents.pt"),
            "c128_support": str(output_dir / "shape" / "upsample_c128" / "coords.pt"),
            "c128_shape_flow": str(output_dir / "shape" / "c128_flow" / "final_state.pt"),
            "c256_support": str(output_dir / "shape" / "upsample_c256" / "coords.pt"),
            "c256_shape_flow": str(output_dir / "shape" / "c256_flow" / "final_state.pt"),
            "c256_texture_flow": str(output_dir / "texture" / "c256_flow" / "final_state.pt"),
            "native_material_mesh": str(output_dir / "final" / "final_material_mesh.pt"),
            "final_vertex_mesh": str(output_dir / "final" / "final_per_vertex_pbr_mesh.pt"),
            "final_face_mesh": str(output_dir / "final" / "final_per_face_pbr_mesh.pt"),
            "metrics_4096": str(output_dir / "metrics_4096.json") if args.render else None,
            "multiview": str(output_dir / "multiview") if args.render else None,
        },
    }
    _atomic_json(output_dir / "config.json", {"format": FORMAT, "args": vars(args), "runtime": runtime})
    _atomic_json(output_dir / "summary.json", summary)
    report = write_comparison_report(Path(args.output_root).expanduser().resolve())
    if report is not None:
        summary["comparison_report"] = str(report)
        _atomic_json(output_dir / "summary.json", summary)
    report_lines = [
        "# Exp-C: baseline4096_from1024",
        "",
        "- Official baseline: `pipeline.run(..., pipeline_type=\"1024_cascade\")`, producing the C64 baseline SLat.",
        "- Shape cascade: decoder `upsample(upsample_times=1)` C64→C128, native shape flow at C128, then decoder `upsample(upsample_times=1)` C128→C256 and native shape flow at C256.",
        "- Texture: native C256 texture flow conditioned by the final C256 shape SLat.",
        "- Decode: native `pipeline.decode_latent(..., resolution=4096)`.",
        "- Conditions: canonical 1024 image only; no 4096 GT tiles and no back image.",
        f"- C64/C128/C256 tokens: {baseline_shape_stats['tokens']:,} / {coords_c128_stats['tokens']:,} / {coords_c256_stats['tokens']:,}.",
    ]
    (output_dir / "RESULT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[done] baseline4096_from1024 output={output_dir}", flush=True)
    return summary


def write_comparison_report(root: Path) -> Optional[Path]:
    entries = {
        "Exp-A front-only": root / "exp_a_front_only" / "summary.json",
        "Exp-B front-global_back-proj": root / "exp_b_front_global_back_proj" / "summary.json",
        "Exp-C baseline4096_from1024": root / "exp_c_baseline4096_from1024" / "summary.json",
    }
    loaded: Dict[str, Mapping[str, Any]] = {}
    for name, path in entries.items():
        if path.is_file():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded:
        return None
    table_metrics: Dict[str, Dict[str, Any]] = {}
    table_routing: Dict[str, Dict[str, Any]] = {}
    lines = [
        "# Pixal3D single-view shared-SLat comparison",
        "",
        "All rows use one canonical/shared support. Exp-B changes only texture projection rows: global remains front, while front-invisible rows use the baseline-material back render.",
        "",
        "| experiment | aligned front PSNR | aligned front SSIM | back proj fraction | notes |",
        "|---|---:|---:|---:|---|",
    ]
    for name, summary in loaded.items():
        metric = {}
        metrics_path = summary.get("artifacts", {}).get("metrics_4096") or summary.get("artifacts", {}).get("metrics")
        if metrics_path and Path(metrics_path).is_file():
            metrics_payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            rows = metrics_payload.get("rows", [])
            if rows:
                metric = rows[-1].get("metrics", rows[-1])
        if not metric:
            # Keep the archive fallback for old summaries; current Exp-C
            # writes its native 4096 metrics directly under artifacts.
            evaluation = summary.get("source_summary", {}).get("evaluation", {})
            if not isinstance(evaluation, Mapping):
                evaluation = summary.get("evaluation", {})
            if isinstance(evaluation, Mapping):
                candidate = evaluation.get("experiment") or evaluation.get("final")
                if isinstance(candidate, Mapping):
                    metric = candidate
        routing = summary.get("routing", {})
        if isinstance(routing, Mapping):
            routing = routing.get("stats", routing)
        if not isinstance(routing, Mapping) or "back_proj_fraction" not in routing:
            condition_routing = summary.get("conditions", {}).get("routing", {})
            if isinstance(condition_routing, Mapping):
                routing = condition_routing
        psnr = metric.get("psnr_db", metric.get("full_psnr_db", "n/a"))
        ssim = metric.get("ssim", "n/a")
        back_fraction = routing.get("back_proj_fraction", 0.0) if isinstance(routing, Mapping) else 0.0
        table_metrics[name] = dict(metric)
        table_routing[name] = dict(routing) if isinstance(routing, Mapping) else {}
        lines.append(f"| {name} | {psnr} | {ssim} | {back_fraction} | {summary.get('status', 'unknown')} |")
    a_name = "Exp-A front-only"
    b_name = "Exp-B front-global_back-proj"
    c_name = "Exp-C baseline4096_from1024"
    a_metric = table_metrics.get(a_name, {})
    b_metric = table_metrics.get(b_name, {})
    c_metric = table_metrics.get(c_name, {})

    def _delta(key: str) -> str:
        if key not in a_metric or key not in b_metric:
            return "n/a"
        return f"{float(b_metric[key]) - float(a_metric[key]):+.4f}"

    b_route = table_routing.get(b_name, {})
    front_fraction = b_route.get("front_proj_fraction", "n/a")
    back_fraction = b_route.get("back_proj_fraction", "n/a")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Exp-B routed texture proj rows as front={front_fraction} and back={back_fraction}; global stayed front and the run kept one texture state (no second global token/latent).",
            f"- On the aligned front view, Exp-B changed PSNR by {_delta('psnr_db')} dB and SSIM by {_delta('ssim')} relative to Exp-A. The measured front result therefore regressed in this seed; the implementation does not claim front non-degradation.",
            "- There is no held-out back-view ground truth in this single-view task. The 3-view contact sheets are the back-side diagnostic; visual inspection of the saved 120°/240° sheets does not show a clear stability gain for Exp-B, and the 240° view contains a stronger local color outlier. Back-proj routing should therefore be treated as implemented but not validated as an improvement by this run.",
            "- Exp-C is the native cascade control: C64 baseline SLat → one decoder upsample and C128 shape flow → one decoder upsample and C256 shape flow → C256 texture flow → native 4096 decode. Its metric is reported for comparison only; it does not replace the routed-proj ablation.",
            "- Review `conditions/texture/routing_stats.json`, `metrics_4096.json`, and the `multiview/multiview_rgb_contact_sheet.png` in each experiment directory for the exact diagnostics.",
        ]
    )
    path = root / "comparison_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_routes()
    if args.experiment == "baseline4096_from1024":
        output_dir = _resolve_output_dir(args)
        return _run_native_cascade_exp_c(args, output_dir)

    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    device = _logical_cuda_device(int(args.cuda_device))
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_summary = output_dir / "summary.json"
    if existing_summary.is_file() and not bool(args.resume):
        raise RuntimeError(
            f"refusing to reuse completed/nonempty output {existing_summary}; choose a new --output-dir or pass --resume"
        )
    started = time.perf_counter()
    _seed(int(args.seed))
    runtime = _runtime(device, int(args.cuda_device))
    print(
        f"[run] experiment={args.experiment} physical_cuda={args.cuda_device} logical={device} "
        f"output={output_dir}",
        flush=True,
    )
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=bool(args.low_vram))
    source_image = Image.open(args.image)
    canonical = pipeline.preprocess_canonical_images(source_image)
    preprocess_meta = legacy._save_canonical_images(canonical, output_dir)
    camera = legacy._load_camera(Path(args.baseline_dir).expanduser().resolve())
    _atomic_json(output_dir / "global_camera.json", camera)
    baseline, baseline_meta = legacy._load_or_make_baseline(
        args,
        pipeline,
        canonical["image_1024"],
        camera,
        output_dir,
    )
    support, transforms, support_meta = _prepare_support(
        args=args,
        baseline=baseline,
        camera=camera,
        output_dir=output_dir,
        device=device,
    )
    visibility_payload = _build_visibility(
        args=args,
        baseline=baseline,
        camera=camera,
        support=support,
        output_dir=output_dir,
        device=device,
    )
    front_tiles = _tile_images(canonical["image_4096"], output_dir, "tiles_front_4096")
    back_tiles: Optional[Dict[int, Image.Image]] = None
    back_render_meta: Optional[Dict[str, Any]] = None
    if args.experiment == "front_global_back_proj":
        baseline_vertex = legacy._baseline_vertex_mesh(
            baseline,
            device,
            output_dir / "baseline" / "baseline_per_vertex_pbr_mesh.pt",
        )
        back_render_meta = singleview_support.render_back_material_observation(
            baseline_vertex,
            camera,
            output_dir / "inputs" / "back_render",
            device,
            resolution=CANONICAL_SIZE,
        )
        with Image.open(back_render_meta["rgb_path"]) as back_image:
            back_tiles = _tile_images(back_image, output_dir, "tiles_back_4096")

    route_mode = (
        "front_global_back_proj"
        if args.experiment == "front_global_back_proj"
        else "front_only"
    )
    # Shape condition remains front-only for both experiments.  The texture
    # helper performs the only row-wise front/back substitution.
    texture_conditions, condition_meta = _build_conditions(
        pipeline=pipeline,
        support=support,
        front_tiles=front_tiles,
        back_tiles=back_tiles,
        visibility_payload=visibility_payload,
        camera=camera,
        output_dir=output_dir,
        device=device,
        mode=route_mode,
        condition_batch_size=int(args.condition_batch_size),
    )
    shape_conditions = legacy._build_batched_image_conditions(
        pipeline,
        pipeline.image_cond_model_shape_1024,
        support.tile_views,
        front_tiles,
        output_dir,
        "shape",
        camera,
        device,
        int(args.condition_batch_size),
    )
    shape_final, texture_final, shape_summary, texture_summary = _run_flows(
        pipeline=pipeline,
        support=support,
        shape_conditions=shape_conditions,
        texture_conditions=texture_conditions,
        output_dir=output_dir,
        device=device,
        args=args,
    )
    final = legacy._decode_and_merge(
        pipeline=pipeline,
        support=support,
        shape_features=shape_final,
        texture_features=texture_final,
        camera=camera,
        output_dir=output_dir,
        device=device,
        decode_batch_size=int(args.decode_batch_size),
    )
    render_meta: Dict[str, Any] = {}
    if bool(args.render):
        render_meta = _render_final_and_metrics(
            output_dir=output_dir,
            camera_path=output_dir / "global_camera.json",
            mesh_path=output_dir / "final" / "final_per_vertex_pbr_mesh.pt",
            canonical=canonical,
            device=device,
            args=args,
        )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "experiment": args.experiment,
        "input": str(Path(args.image).expanduser().resolve()),
        "baseline_dir": str(Path(args.baseline_dir).expanduser().resolve()),
        "camera": camera,
        "runtime": runtime,
        "seed": {
            "global": int(args.seed),
            "shape": int(args.shape_seed),
            "texture": int(args.texture_seed),
        },
        "preprocess": preprocess_meta,
        "baseline": baseline_meta,
        "layout": {
            "canonical": CANONICAL_SIZE,
            "tile_size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "tile_count": TILE_COUNT,
            "tile_order": "row-major",
            "rectangle": "half-open",
        },
        "support": support_meta,
        "visibility": {
            "source": visibility_payload.get("source"),
            "frozen": True,
            "front_visible_rows": int(visibility_payload["visible"].sum()),
            "front_invisible_mapped_rows": int(
                (visibility_payload["mapping_valid"] & ~visibility_payload["visible"]).sum()
            ),
            "tile_stats": visibility_payload.get("tile_stats", {}),
            "nearest_triangle": "baseline mesh triangle, frozen before shape flow",
        },
        "conditions": condition_meta,
        "routing": {
            "global": "front_tile_image",
            "proj": route_mode,
            "stats": condition_meta.get("routing", {}),
            "back_render": back_render_meta,
            "single_texture_state": True,
            "second_global_token": False,
            "second_texture_latent": False,
            "front_back_endpoint_average": False,
        },
        "shape_flow": shape_summary,
        "texture_flow": texture_summary,
        "decode": {key: value for key, value in final.items() if key not in {"vertex_mesh", "face_mesh"}},
        "render": render_meta,
        "timing": {"total_seconds": float(time.perf_counter() - started)},
        "artifacts": {
            "support": str(output_dir / "support" / "master_support.pt"),
            "visibility": str(output_dir / "support" / "frozen_visibility.pt"),
            "shape_flow": str(output_dir / "shape" / "flow_summary.json"),
            "texture_flow": str(output_dir / "texture" / "flow_summary.json"),
            "final_vertex_mesh": str(output_dir / "final" / "final_per_vertex_pbr_mesh.pt"),
            "final_face_mesh": str(output_dir / "final" / "final_per_face_pbr_mesh.pt"),
            "metrics_4096": str(output_dir / "metrics_4096.json") if args.render else None,
            "multiview": str(output_dir / "multiview"),
        },
    }
    _atomic_json(output_dir / "config.json", {"format": FORMAT, "args": vars(args), "runtime": runtime})
    _atomic_json(output_dir / "summary.json", summary)
    report = write_comparison_report(Path(args.output_root).expanduser().resolve())
    if report is not None:
        summary["comparison_report"] = str(report)
        _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] {args.experiment} output={output_dir}", flush=True)
    return summary


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    names = {
        "front_only": "exp_a_front_only",
        "front_global_back_proj": "exp_b_front_global_back_proj",
        "baseline4096_from1024": "exp_c_baseline4096_from1024",
    }
    return Path(args.output_root).expanduser().resolve() / names[args.experiment]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("front_only", "front_global_back_proj", "baseline4096_from1024"),
        default="front_only",
        help="Exp-A, Exp-B or the Global-1024-derived Exp-C control",
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--support-source", type=Path, default=DEFAULT_SUPPORT_SOURCE)
    parser.add_argument("--visibility-source", type=Path, default=None)
    parser.add_argument("--c256-source", type=Path, default=DEFAULT_C256_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--shape-seed", type=int, default=20260824)
    parser.add_argument("--texture-seed", type=int, default=20260825)
    parser.add_argument("--condition-batch-size", type=int, default=CONDITION_BATCH_SIZE)
    parser.add_argument("--native-geometry-encode-batch-size", type=int, default=1)
    parser.add_argument("--flow-batch-size", type=int, default=FLOW_BATCH_SIZE)
    parser.add_argument("--decode-batch-size", type=int, default=DECODE_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--visibility-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-step-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.condition_batch_size <= 0 or args.flow_batch_size <= 0 or args.decode_batch_size <= 0:
        raise ValueError("condition/flow/decode batch sizes must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.cuda_device != 4:
        raise ValueError("Codex.md execution is fixed to physical CUDA 4")
    if args.render_resolution != CANONICAL_SIZE:
        raise ValueError("the native single-view artifact must be rendered at 4096")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
