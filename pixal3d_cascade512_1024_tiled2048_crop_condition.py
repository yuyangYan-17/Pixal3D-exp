#!/usr/bin/env python3
"""512 -> 1024 -> tiled 2048 Pixal3D cascade with per-cube crop conditions.

The native 512/C32 and 1024/C64 shape stages are preserved.  The C64 shape
SLat is then upsampled with the native decoder, quantized to C128, partitioned
into eight non-overlapping C64 cubes, and sampled as one real sparse batch.
Every cube owns an independently encoded, square, 16-pixel-aligned image crop;
both its global token and its pointwise DINO/NAF projection features come from
that crop.  Shape flow precedes texture flow and the assembled global C128
latents are decoded once at resolution 2048.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

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

from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_cascade512_1024_tiled2048_square_crop_condition_v1"
GLOBAL_GRID = 128
CUBE_SIZE = 64
CUBE_STARTS = (0, 64)
CANONICAL_SIZE = 4096
PROJECTION_SIZE = 1024
DINO_PATCH = 16
DEFAULT_MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def square_aligned_box(
    projected_xy: torch.Tensor,
    image_size: int = CANONICAL_SIZE,
    multiple: int = DINO_PATCH,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Smallest centered in-bounds square containing the clipped projection."""
    points = torch.as_tensor(projected_xy, dtype=torch.float64).cpu()
    if points.ndim != 2 or points.shape[1] != 2 or not points.shape[0]:
        raise ValueError("projected_xy must be non-empty [N,2]")
    if not torch.isfinite(points).all():
        raise ValueError("projected_xy contains NaN/Inf")
    raw_lo, raw_hi = points.amin(0), points.amax(0)
    lo = raw_lo.clamp(0.0, float(image_size))
    hi = raw_hi.clamp(0.0, float(image_size))
    if bool((hi <= lo).any()):
        raise RuntimeError("projected cube does not intersect the canonical image")
    needed = max(float((hi - lo).max()), 1.0)
    side = min(image_size, int(math.ceil(needed / multiple)) * multiple)
    center = (lo + hi) * 0.5
    starts: list[int] = []
    for axis in range(2):
        lower = max(0, int(math.ceil(float(hi[axis]))) - side)
        upper = min(int(math.floor(float(lo[axis]))), image_size - side)
        if lower > upper:
            raise RuntimeError("cannot fit aligned square around projected bounds")
        preferred = int(round(float(center[axis]) - side / 2.0))
        starts.append(min(max(preferred, lower), upper))
    x0, y0 = starts
    box = (x0, y0, x0 + side, y0 + side)
    return box, {
        "raw_bbox_pixel_edges_4096": [*raw_lo.tolist(), *raw_hi.tolist()],
        "clipped_bbox_pixel_edges_4096": [*lo.tolist(), *hi.tolist()],
        "crop_box_4096": list(box),
        "crop_size": [side, side],
        "square": True,
        "side_multiple": multiple,
    }


def cube_crop(start: Sequence[int], camera: Mapping[str, float]) -> dict[str, Any]:
    import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as camera_core

    start_t = torch.as_tensor(start, dtype=torch.float64)
    bits = torch.tensor(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=torch.float64,
    )
    boundaries = start_t[None] + bits * CUBE_SIZE
    corners_q = 2.0 * boundaries / float(GLOBAL_GRID) - 1.0
    uv, depth, finite = camera_core._project_global_q_to_image(
        corners_q,
        global_camera=camera,
        image_width=PROJECTION_SIZE,
        image_height=PROJECTION_SIZE,
    )
    if not bool(finite.all()):
        raise RuntimeError(f"cube {tuple(start)} crosses the camera plane")
    pixel_edges = (uv.double() + 0.5) * (CANONICAL_SIZE / PROJECTION_SIZE)
    box, stats = square_aligned_box(pixel_edges)
    x0, y0, x1, y1 = box
    return {
        **stats,
        "projection_crop_box": (
            x0 / CANONICAL_SIZE,
            y0 / CANONICAL_SIZE,
            x1 / CANONICAL_SIZE,
            y1 / CANONICAL_SIZE,
        ),
        "cube_corners_q": corners_q.tolist(),
        "cube_corner_depth": depth.detach().cpu().tolist(),
    }


def build_records(coords: torch.Tensor, camera: Mapping[str, float]) -> list[dict[str, Any]]:
    if coords.ndim != 2 or coords.shape[1] != 4 or bool((coords[:, 0] != 0).any()):
        raise ValueError("global C128 coords must be single-batch [N,4]")
    xyz = coords[:, 1:].cpu().int()
    writes = torch.zeros(coords.shape[0], dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for sx in CUBE_STARTS:
        for sy in CUBE_STARTS:
            for sz in CUBE_STARTS:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                mask = ((xyz >= start) & (xyz < start + CUBE_SIZE)).all(1)
                rows = torch.where(mask)[0].long()
                writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
                local_xyz = xyz.index_select(0, rows) - start
                local_coords = torch.cat(
                    (torch.zeros((rows.numel(), 1), dtype=torch.int32), local_xyz), dim=1
                )
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": tuple(int(v) for v in start.tolist()),
                        "global_row_ids": rows,
                        "local_coords": local_coords,
                        "crop": cube_crop(start.tolist(), camera),
                    }
                )
                cube_id += 1
    if not torch.all(writes == 1):
        raise RuntimeError("stride-64 cubes must partition every C128 token exactly once")
    return records


def normalization_tensors(spec: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(spec["mean"], dtype=torch.float32, device=device)[None]
    std = torch.as_tensor(spec["std"], dtype=torch.float32, device=device)[None]
    return mean, std


def upsample_and_quantize(
    pipeline: Any,
    slat: SparseTensor,
    source_resolution: int,
    target_grid: int,
    baseline_rounding: bool,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    candidates = decoder.upsample(slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    scaled = (candidates[:, 1:].float() + 0.5) / float(source_resolution)
    if baseline_rounding:
        xyz = (scaled * (target_grid - 1)).round().int()
    else:
        xyz = (scaled * target_grid).int()
    coords = torch.cat((candidates[:, :1].int(), xyz), dim=1).unique(dim=0)
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= target_grid)).any()):
        raise RuntimeError(f"quantized support escapes C{target_grid}")
    return coords


def extract_crop_conditions(
    pipeline: Any,
    image: Image.Image,
    camera: Mapping[str, float],
    global_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    stage: str,
    output_dir: Path,
) -> dict[int, dict[str, torch.Tensor]]:
    model = (
        pipeline.image_cond_model_shape_1024
        if stage == "shape"
        else pipeline.image_cond_model_tex_1024
    )
    root = output_dir / "conditions" / stage
    result: dict[int, dict[str, torch.Tensor]] = {}
    low_vram = bool(pipeline.low_vram)
    if low_vram:
        model.to(pipeline.device)
        pipeline.low_vram = False
    try:
        for rec in records:
            cube_id = int(rec["cube_id"])
            rows = rec["global_row_ids"].long()
            if not rows.numel():
                continue
            box = tuple(int(v) for v in rec["crop"]["crop_box_4096"])
            crop = image.crop(box).convert("RGB")
            if crop.width != crop.height or crop.width % DINO_PATCH:
                raise RuntimeError(f"cube {cube_id} condition crop is not a 16-aligned square")
            crop_path = root / "crops" / f"cube_{cube_id:02d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            subset = global_coords.index_select(
                0, rows.to(device=global_coords.device)
            ).to(pipeline.device)
            cond = pipeline.get_proj_cond_shape(
                model,
                [crop],
                subset,
                camera_angle_x=float(camera["camera_angle_x"]),
                distance=float(camera["distance"]),
                mesh_scale=float(camera.get("mesh_scale", 1.0)),
                grid_resolution_override=GLOBAL_GRID,
                projection_crop_box=rec["crop"]["projection_crop_box"],
                preserve_image_resolution=True,
            )
            glob = cond["cond"]["global"].detach().cpu().contiguous()
            proj = cond["cond"]["proj"].feats.detach().cpu().contiguous()
            if glob.shape[0] != 1 or proj.shape[0] != rows.numel():
                raise RuntimeError(f"cube {cube_id} condition shape mismatch")
            payload = {
                "format": FORMAT,
                "stage": stage,
                "cube_id": cube_id,
                "start": rec["start"],
                "global_row_ids": rows,
                "local_coords": rec["local_coords"],
                "crop": rec["crop"],
                "global": glob,
                "proj": proj,
                "global_token_source": "this_cube_square_crop",
                "projected_token_source": "this_cube_square_crop_dino_naf_uv",
            }
            atomic_save(root / f"cube_{cube_id:02d}.pt", payload)
            result[cube_id] = payload
            print(
                f"[condition-{stage}] cube={cube_id} crop={crop.width}x{crop.height} "
                f"tokens={rows.numel():,}",
                flush=True,
            )
            del cond, subset, glob, proj, crop
            empty_cuda()
    finally:
        if low_vram:
            pipeline.low_vram = True
            model.cpu()
        empty_cuda()
    return result


def pack_sparse(records: Sequence[Mapping[str, Any]], features: torch.Tensor) -> SparseTensor:
    feats: list[torch.Tensor] = []
    coords: list[torch.Tensor] = []
    for batch_id, rec in enumerate(rec for rec in records if rec["global_row_ids"].numel()):
        feats.append(features.index_select(0, rec["global_row_ids"].long()))
        local = rec["local_coords"].clone()
        local[:, 0] = batch_id
        coords.append(local)
    if not feats:
        raise RuntimeError("all C64 cubes are empty")
    packed = SparseTensor(torch.cat(feats, 0), torch.cat(coords, 0))
    if len(packed) != len(feats):
        raise RuntimeError("packed sparse tensor batch size mismatch")
    return packed


def pack_conditions(
    active: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, torch.Tensor]],
    packed_coords: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    glob = torch.cat([conditions[int(rec["cube_id"])]["global"] for rec in active], 0).to(device)
    proj = torch.cat([conditions[int(rec["cube_id"])]["proj"] for rec in active], 0).to(device)
    if proj.shape[0] != packed_coords.shape[0]:
        raise RuntimeError("packed projected condition lost sparse row alignment")
    return {
        "cond": {"global": glob, "proj": SparseTensor(proj, packed_coords)},
        "neg_cond": {
            "global": torch.zeros_like(glob),
            "proj": SparseTensor(torch.zeros_like(proj), packed_coords),
        },
    }


def unpack_global(
    sampled: SparseTensor,
    active: Sequence[Mapping[str, Any]],
    global_rows: int,
) -> torch.Tensor:
    output = torch.empty((global_rows, sampled.feats.shape[1]), dtype=sampled.feats.dtype)
    writes = torch.zeros(global_rows, dtype=torch.int16)
    for batch_id, rec in enumerate(active):
        mask = sampled.coords[:, 0] == batch_id
        if not bool(mask.any()):
            raise RuntimeError(f"tiled flow result lacks batch {batch_id}")
        part_coords = sampled.coords[mask].detach().cpu().clone()
        part_coords[:, 0] = 0
        if not torch.equal(part_coords, rec["local_coords"]):
            raise RuntimeError(f"cube {rec['cube_id']} support/order changed during flow")
        rows = rec["global_row_ids"].long()
        output.index_copy_(0, rows, sampled.feats[mask].detach().cpu())
        writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
    if not torch.all(writes == 1):
        raise RuntimeError("assembled global flow result must have exactly one write per row")
    return output


@torch.no_grad()
def run_tiled_flow(
    pipeline: Any,
    stage: str,
    global_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, torch.Tensor]],
    seed: int,
    shape_normalized: torch.Tensor | None,
    output_dir: Path,
) -> torch.Tensor:
    active = [rec for rec in records if rec["global_row_ids"].numel()]
    model = pipeline.models[
        "shape_slat_flow_model_1024" if stage == "shape" else "tex_slat_flow_model_1024"
    ]
    shape_channels = 0 if shape_normalized is None else int(shape_normalized.shape[1])
    noise_channels = int(model.in_channels) - shape_channels
    noise = torch.randn(
        (global_coords.shape[0], noise_channels),
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    packed_noise = pack_sparse(active, noise).to(pipeline.device)
    packed_cond = pack_conditions(active, conditions, packed_noise.coords, pipeline.device)
    concat = pack_sparse(active, shape_normalized).to(pipeline.device) if shape_normalized is not None else None
    params = dict(
        pipeline.shape_slat_sampler_params if stage == "shape" else pipeline.tex_slat_sampler_params
    )
    if pipeline.low_vram:
        model.to(pipeline.device)
    sampler = pipeline.shape_slat_sampler if stage == "shape" else pipeline.tex_slat_sampler
    started = time.perf_counter()
    sampled = sampler.sample(
        model,
        packed_noise,
        concat_cond=concat,
        **packed_cond,
        **params,
        verbose=True,
        tqdm_desc=f"Sampling tiled C128 {stage} as B={len(active)} C64 cubes",
    ).samples
    seconds = time.perf_counter() - started
    if pipeline.low_vram:
        model.cpu()
    global_features = unpack_global(sampled, active, global_coords.shape[0])
    atomic_save(
        output_dir / stage / "final_state_normalized.pt",
        {"coords": global_coords.cpu(), "features": global_features, "seed": seed},
    )
    atomic_json(
        output_dir / stage / "flow_summary.json",
        {
            "stage": stage,
            "mode": "one_real_sparse_batch_of_nonoverlapping_C64_cubes",
            "batch_size": len(active),
            "global_grid": GLOBAL_GRID,
            "context": CUBE_SIZE,
            "stride": CUBE_SIZE,
            "tokens": int(global_coords.shape[0]),
            "seconds": seconds,
            "sampler_params": params,
        },
    )
    del packed_noise, packed_cond, concat, sampled, noise
    empty_cuda()
    return global_features


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    from inference import init_pipeline

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [v.strip() for v in visible.split(",")] != [str(args.cuda_device)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected only CUDA {args.cuda_device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    source = Image.open(args.image).convert("RGB")
    canonical = pipeline.preprocess_canonical_images(source)
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for key in ("image_512", "image_1024", "image_4096", "foreground_mask_4096"):
        if key in canonical and isinstance(canonical[key], Image.Image):
            canonical[key].save(inputs / f"{key}.png")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    if "camera" in camera:
        camera = camera["camera"]
    atomic_json(out / "global_camera.json", camera)
    atomic_json(
        out / "config.json",
        {
            "format": FORMAT,
            "status": "running",
            "image": args.image,
            "camera": args.camera,
            "cuda_device": args.cuda_device,
            "seed": args.seed,
            "shape_seed": args.shape_seed,
            "texture_seed": args.texture_seed,
            "path": "512/C32 shape -> 1024/C64 shape -> tiled 2048/C128 shape -> tiled C128 texture -> global 2048 decode",
            "crop_condition": "per-C64 square 16-aligned crop; crop-local global token and crop-local DINO/NAF UV features",
        },
    )

    print("[stage 1] native sparse structure from image_512 -> C32", flush=True)
    cond_ss = pipeline.get_proj_cond_ss(
        [canonical["image_512"]],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
    )
    coords_c32 = pipeline.sample_sparse_structure(cond_ss, 32)
    del cond_ss
    empty_cuda()

    print("[stage 2] native 512/C32 shape flow", flush=True)
    cond_lr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [canonical["image_512"]],
        coords_c32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
    )
    shape_c32 = pipeline.sample_shape_slat(
        cond_lr, pipeline.models["shape_slat_flow_model_512"], coords_c32
    )
    del cond_lr, coords_c32
    empty_cuda()

    print("[stage 3] decoder upsample(4), native baseline quantization -> C64", flush=True)
    coords_c64 = upsample_and_quantize(pipeline, shape_c32, 512, 64, baseline_rounding=True)
    cond_c64 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [canonical["image_1024"]],
        coords_c64,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=64,
    )
    shape_c64 = pipeline.sample_shape_slat(
        cond_c64, pipeline.models["shape_slat_flow_model_1024"], coords_c64
    )
    atomic_save(
        out / "baseline" / "shape_c64.pt",
        {"coords": shape_c64.coords.cpu(), "features": shape_c64.feats.cpu()},
    )
    del cond_c64, coords_c64, shape_c32
    empty_cuda()

    print("[stage 4] decoder upsample(4), native floor down-quantization -> C128", flush=True)
    coords_c128 = upsample_and_quantize(pipeline, shape_c64, 1024, 128, baseline_rounding=False)
    atomic_save(out / "support" / "coords_c128.pt", {"coords": coords_c128.cpu()})
    records = build_records(coords_c128.cpu(), camera)
    atomic_json(
        out / "support" / "cube_layout.json",
        {
            "global_grid": 128,
            "context": 64,
            "stride": 64,
            "cubes": [
                {
                    "cube_id": rec["cube_id"],
                    "start": rec["start"],
                    "tokens": int(rec["global_row_ids"].numel()),
                    "crop": rec["crop"],
                }
                for rec in records
            ],
        },
    )
    del shape_c64
    empty_cuda()

    shape_conditions = extract_crop_conditions(
        pipeline, canonical["image_4096"], camera, coords_c128, records, "shape", out
    )
    shape_norm = run_tiled_flow(
        pipeline, "shape", coords_c128.cpu(), records, shape_conditions,
        args.shape_seed, None, out,
    )
    shape_mean, shape_std = normalization_tensors(pipeline.shape_slat_normalization, torch.device("cpu"))
    shape_denorm = shape_norm * shape_std + shape_mean
    atomic_save(
        out / "shape" / "final_state_denormalized.pt",
        {"coords": coords_c128.cpu(), "features": shape_denorm},
    )
    del shape_conditions
    empty_cuda()

    texture_conditions = extract_crop_conditions(
        pipeline, canonical["image_4096"], camera, coords_c128, records, "texture", out
    )
    texture_norm = run_tiled_flow(
        pipeline, "texture", coords_c128.cpu(), records, texture_conditions,
        args.texture_seed, shape_norm, out,
    )
    tex_mean, tex_std = normalization_tensors(pipeline.tex_slat_normalization, torch.device("cpu"))
    texture_denorm = texture_norm * tex_std + tex_mean
    atomic_save(
        out / "texture" / "final_state_denormalized.pt",
        {"coords": coords_c128.cpu(), "features": texture_denorm},
    )
    del texture_conditions
    empty_cuda()

    print("[decode] assembled global C128 shape+texture -> 2048", flush=True)
    shape_st = SparseTensor(shape_denorm.to(device), coords_c128.to(device))
    texture_st = SparseTensor(texture_denorm.to(device), coords_c128.to(device))
    decoded = pipeline.decode_latent(shape_st, texture_st, 2048)
    if len(decoded) != 1:
        raise RuntimeError(f"global decoder returned {len(decoded)} meshes")
    native = decoded[0]
    atomic_save(out / "final" / "final_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
    import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc

    vertex_mesh, face_mesh = expc._native_mesh_to_pbr(native, device)
    vertex_path = out / "final" / "final_per_vertex_pbr_mesh.pt"
    atomic_save(vertex_path, {"format": FORMAT, "mesh": vertex_mesh})
    atomic_save(out / "final" / "final_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": face_mesh})
    atomic_json(
        out / "decode" / "summary.json",
        {
            "resolution": 2048,
            "grid_resolution": 128,
            "vertices": int(vertex_mesh.vertices.shape[0]),
            "faces": int(vertex_mesh.faces.shape[0]),
        },
    )
    del decoded, native, shape_st, texture_st, pipeline
    empty_cuda()

    print("[render] native PBR multiview at 2048", flush=True)
    import pixal3d_render_global4096_multiview as multiview

    manifest = multiview.render(
        SimpleNamespace(
            mesh=vertex_path,
            camera=out / "global_camera.json",
            output_dir=out / "multiview_2048",
            angles="0,60,120,180,240,300",
            resolution=2048,
            face_chunk_size=args.render_face_chunk_size,
            device=str(device),
            force=False,
        )
    )
    atomic_json(
        out / "summary.json",
        {
            "format": FORMAT,
            "status": "complete",
            "tokens_c128": int(coords_c128.shape[0]),
            "active_cubes": sum(bool(rec["global_row_ids"].numel()) for rec in records),
            "mesh": vertex_path,
            "multiview": out / "multiview_2048" / "multiview_rgb_contact_sheet.png",
            "render_manifest": manifest,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("assets/choose/0_img.png"))
    parser.add_argument(
        "--camera", type=Path,
        default=Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json"),
    )
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/cascade512_1024_tiled2048_square_crop_cuda4"),
    )
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-seed", type=int, default=43)
    parser.add_argument("--texture-seed", type=int, default=44)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
