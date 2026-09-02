#!/usr/bin/env python3
"""Build Global C256 from block-local native cascades, then run synchronous C64 flows.

Experiment definition:
1. Run the full-image native 1024 cascade only through its pre-HR C64 support.
2. Partition that C64 space into non-overlapping C16 blocks and retain nonempty blocks.
3. For every retained block, independently regenerate support through native
   Structure C16 -> occupancy C32 -> LR Shape C32 -> learned upsample C64.
4. Map every local C64 support into its exact non-overlapping Global C256 block.
5. Extract full-image features once per conditioning model.  Sample projected
   features for each block at its correct global spatial position (no image crops).
6. Use one global noise/state, synchronously evaluate block-local C64 Shape and
   Texture velocities, then decode Global C256 once at resolution 4096.

There is intentionally no halo, overlap, owner fusion, or cross-block context.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global_c256_restructured_blocks_singleview_v1"
LOCAL_BLOCKS = 4
BASE_GRID = 64
BASE_BLOCK = 16
GLOBAL_GRID = 256
FLOW_BLOCK = 64


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_save(path: Path, value: object) -> None:
    cube_flow.atomic_save(path, value)


def atomic_json(path: Path, value: object) -> None:
    cube_flow.atomic_json(path, value)


def block_id(index: tuple[int, int, int]) -> int:
    x, y, z = index
    return x * 16 + y * 4 + z


def block_index(value: int) -> tuple[int, int, int]:
    return value // 16, (value // 4) % 4, value % 4


def base_camera_to_world(model: Any, distance: float, device: torch.device) -> torch.Tensor:
    result = model.proj_grid.front_view_transform_matrix.to(device=device, dtype=torch.float32).clone()
    result[1, 3] = -float(distance)
    return result


def local_to_global_camera_transform(
    model: Any, *, block_start: tuple[int, int, int], global_resolution: int,
    global_extent: int, distance: float, device: torch.device,
) -> torch.Tensor:
    """Camera-to-world matrix making a normalized local cube occupy one global block."""
    if global_resolution <= 1 or global_extent <= 1:
        raise ValueError("invalid grid mapping")
    scale = float(global_extent - 1) / float(global_resolution - 1)
    center_unrotated = torch.tensor(
        [float(v) + (global_extent - 1) / 2.0 for v in block_start],
        dtype=torch.float32, device=device,
    ) / float(global_resolution - 1) - 0.5
    # ProjGrid applies (x,y,z)->(x,-z,y) before the camera transform.
    translation = torch.stack((center_unrotated[0], -center_unrotated[2], center_unrotated[1]))
    affine = torch.eye(4, dtype=torch.float32, device=device)
    affine[:3, :3] *= scale
    affine[:3, 3] = translation
    camera = base_camera_to_world(model, distance, device)
    # project() uses inverse(camera_to_world).  This makes it evaluate
    # inv(camera) @ (affine @ p_local), exactly the desired global position.
    return torch.linalg.inv(affine) @ camera


def validate_projection_transform(
    model: Any, transform: torch.Tensor, *, local_resolution: int,
    global_resolution: int, block_start: tuple[int, int, int],
    distance: float, fov: float, device: torch.device,
) -> float:
    corners = torch.tensor(
        [[x, y, z] for x in (0, local_resolution - 1)
         for y in (0, local_resolution - 1) for z in (0, local_resolution - 1)],
        dtype=torch.long, device=device,
    )
    extent = local_resolution
    global_indices = corners + torch.tensor(block_start, dtype=torch.long, device=device)
    fov_t = torch.tensor([fov], device=device); dist_t = torch.tensor([distance], device=device)
    scale_t = torch.ones(1, device=device)
    local_px = model.proj_grid.project_grid_indices(
        fov_t, dist_t, scale_t, transform.unsqueeze(0), corners,
        grid_resolution=local_resolution,
    )[0]
    global_px = model.proj_grid.project_grid_indices(
        fov_t, dist_t, scale_t, None, global_indices,
        grid_resolution=global_resolution,
    )[0]
    error = float((local_px - global_px).abs().max().item())
    if error > 2e-3:
        raise RuntimeError(f"local/global projection transform mismatch: max pixel error {error}")
    return error


@torch.no_grad()
def extract_full_image_features(model: Any, image: Image.Image, device: torch.device) -> dict[str, torch.Tensor | None]:
    """Mirror DinoV3ProjFeatureExtractor.forward while retaining feature maps."""
    model.to(device).eval()
    resized = image.resize((model.image_size, model.image_size), Image.Resampling.LANCZOS).convert("RGB")
    array = np.asarray(resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    image_for_naf = image_tensor.clone()
    normalized = model.transform(image_tensor)
    z = model.extract_features(normalized)
    num_register = int(getattr(model.model.config, "num_register_tokens", 4))
    global_features = torch.cat((z[:, :1], z[:, 1:1 + num_register]), 1)
    patch = z[:, 1 + num_register:]
    patch_h = int(model.image_size // model.patch_size)
    patch_w = patch_h
    if patch.shape[1] != patch_h * patch_w:
        raise RuntimeError("DINO patch layout mismatch")
    lr_map = patch.reshape(1, patch_h, patch_w, -1)
    hr_map = None
    if bool(getattr(model, "use_naf_upsample", False)):
        model._load_naf()
        hr_map = model.naf_model(
            image_for_naf, lr_map.permute(0, 3, 1, 2), model.naf_target_size,
        )
    return {"global": global_features, "lr": lr_map, "hr": hr_map}


@torch.no_grad()
def project_cached_features(
    model: Any, cached: Mapping[str, torch.Tensor | None], *, transform: torch.Tensor,
    fov: float, distance: float, coords: torch.Tensor | None,
    grid_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = transform.device
    fov_t = torch.tensor([fov], device=device); dist_t = torch.tensor([distance], device=device)
    scale_t = torch.ones(1, device=device)
    indices = None if coords is None else coords[:, 1:4].to(device=device, dtype=torch.long)
    lr = model.proj_grid(
        cached["lr"], fov_t, dist_t, scale_t, transform.unsqueeze(0),
        grid_indices=indices, grid_resolution=grid_resolution,
    )
    if cached["hr"] is not None:
        hr = model.proj_grid(
            cached["hr"], fov_t, dist_t, scale_t, transform.unsqueeze(0), BHWC=False,
            grid_indices=indices, grid_resolution=grid_resolution,
        )
        proj = torch.cat((lr, hr), -1)
    else:
        proj = lr
    return cached["global"], proj


def condition_from_dense(global_features: torch.Tensor, proj: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "cond": {"global": global_features, "proj": proj},
        "neg_cond": {"global": torch.zeros_like(global_features), "proj": torch.zeros_like(proj)},
    }


def condition_from_sparse(
    global_features: torch.Tensor, proj: torch.Tensor, coords: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    features = proj[0]
    sparse = SparseTensor(features, coords)
    return {
        "cond": {"global": global_features, "proj": sparse},
        "neg_cond": {"global": torch.zeros_like(global_features),
                     "proj": SparseTensor(torch.zeros_like(features), coords)},
    }


@torch.no_grad()
def learned_c64_support(pipeline: Any, lr_slat: SparseTensor, device: torch.device) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(device); decoder.low_vram = True
    candidates = decoder.upsample(lr_slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu(); decoder.low_vram = False
    result = torch.cat((
        candidates[:, :1],
        (((candidates[:, 1:] + 0.5) / 512.0) * 63.0).round().int(),
    ), 1).unique(dim=0)
    if torch.any(result[:, 1:] < 0) or torch.any(result[:, 1:] >= 64):
        raise RuntimeError("learned C64 support is outside [0,63]")
    return result.int().contiguous()


@torch.no_grad()
def build_native_baseline_c64(
    pipeline: Any, image512: Image.Image, camera: Mapping[str, float],
    output: Path, seed: int, steps: int, device: torch.device,
) -> torch.Tensor:
    path = output / "baseline_pre_hr" / "baseline_c64_support.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)["coords"].int()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    ss_cond = pipeline.get_proj_cond_ss(
        [image512], camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]), mesh_scale=1.0,
    )
    coords32 = pipeline.sample_sparse_structure(ss_cond, 32, 1, {"steps": steps})
    shape_cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512, [image512], coords32,
        camera_angle_x=float(camera["camera_angle_x"]), distance=float(camera["distance"]),
        mesh_scale=1.0, grid_resolution_override=32,
    )
    lr_slat = pipeline.sample_shape_slat(
        shape_cond, pipeline.models["shape_slat_flow_model_512"], coords32, {"steps": steps},
    )
    coords64 = learned_c64_support(pipeline, lr_slat, device).cpu()
    atomic_save(output / "baseline_pre_hr" / "baseline_c32_support.pt", {"coords": coords32.cpu()})
    atomic_save(path, {"format": FORMAT, "coords": coords64, "before_hr_shape_flow": True})
    atomic_json(output / "baseline_pre_hr" / "summary.json", {
        "c32_tokens": int(coords32.shape[0]), "c64_tokens": int(coords64.shape[0]),
        "seed": seed, "steps": steps,
    })
    del ss_cond, shape_cond, lr_slat, coords32
    empty_cuda()
    return coords64


def active_blocks(coords64: torch.Tensor) -> list[dict[str, Any]]:
    xyz = coords64[:, 1:4].cpu().int()
    records = []
    for bx in range(4):
        for by in range(4):
            for bz in range(4):
                index = (bx, by, bz); start = torch.tensor(index, dtype=torch.int32) * 16
                inside = ((xyz >= start) & (xyz < start + 16)).all(1)
                if inside.any():
                    records.append({"cube_id": block_id(index), "block_index": index,
                                    "baseline_tokens": int(inside.sum())})
    return records


@torch.no_grad()
def generate_local_structures(
    pipeline: Any, image512: Image.Image, camera: Mapping[str, float], records: list[dict[str, Any]],
    output: Path, seed: int, steps: int, device: torch.device,
) -> None:
    root = output / "local_structure"
    pending = [r for r in records if not (root / f"block_{r['cube_id']:02d}" / "c32_support.pt").is_file()]
    if not pending:
        return
    model = pipeline.image_cond_model_ss
    cached = extract_full_image_features(model, image512, device)
    fov, distance = float(camera["camera_angle_x"]), float(camera["distance"])
    flow = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    was_low_vram = bool(pipeline.low_vram)
    if was_low_vram:
        flow.to(device); decoder.to(device); pipeline.low_vram = False
    try:
        for order, rec in enumerate(pending, 1):
            bid = int(rec["cube_id"]); idx = rec["block_index"]
            out = root / f"block_{bid:02d}"; out.mkdir(parents=True, exist_ok=True)
            transform = local_to_global_camera_transform(
                model, block_start=tuple(v * 16 for v in idx), global_resolution=64,
                global_extent=16, distance=distance, device=device,
            )
            error = validate_projection_transform(
                model, transform, local_resolution=16, global_resolution=64,
                block_start=tuple(v * 16 for v in idx), distance=distance, fov=fov, device=device,
            )
            glob, proj = project_cached_features(
                model, cached, transform=transform, fov=fov, distance=distance,
                coords=None, grid_resolution=16,
            )
            torch.manual_seed(seed + 10_000 + bid); torch.cuda.manual_seed_all(seed + 10_000 + bid)
            coords32 = pipeline.sample_sparse_structure(
                condition_from_dense(glob, proj), 32, 1, {"steps": steps},
            ).cpu().int()
            atomic_save(out / "c32_support.pt", {"format": FORMAT, "coords": coords32,
                        "block_index": idx, "projection_max_error_pixels": error})
            print(f"[local-ss] block={bid:02d} {order}/{len(pending)} C32={len(coords32):,}", flush=True)
            del glob, proj, coords32
            empty_cuda()
    finally:
        if was_low_vram:
            pipeline.low_vram = True; flow.cpu(); decoder.cpu()
    del cached
    if pipeline.low_vram: model.cpu()
    empty_cuda()


@torch.no_grad()
def generate_local_c64_supports(
    pipeline: Any, image512: Image.Image, camera: Mapping[str, float], records: list[dict[str, Any]],
    output: Path, seed: int, steps: int, device: torch.device,
) -> None:
    root = output / "local_structure"
    pending = [r for r in records if not (root / f"block_{r['cube_id']:02d}" / "c64_support.pt").is_file()]
    if not pending:
        return
    model = pipeline.image_cond_model_shape_512
    cached = extract_full_image_features(model, image512, device)
    fov, distance = float(camera["camera_angle_x"]), float(camera["distance"])
    flow = pipeline.models["shape_slat_flow_model_512"]
    decoder = pipeline.models["shape_slat_decoder"]
    was_low_vram = bool(pipeline.low_vram)
    if was_low_vram:
        flow.to(device); decoder.to(device); pipeline.low_vram = False
    try:
        for order, rec in enumerate(pending, 1):
            bid = int(rec["cube_id"]); idx = rec["block_index"]
            out = root / f"block_{bid:02d}"
            coords32 = torch.load(out / "c32_support.pt", map_location="cpu", weights_only=False)["coords"].int().to(device)
            transform = local_to_global_camera_transform(
                model, block_start=tuple(v * 16 for v in idx), global_resolution=64,
                global_extent=16, distance=distance, device=device,
            )
            glob, proj = project_cached_features(
                model, cached, transform=transform, fov=fov, distance=distance,
                coords=coords32, grid_resolution=32,
            )
            torch.manual_seed(seed + 20_000 + bid); torch.cuda.manual_seed_all(seed + 20_000 + bid)
            lr = pipeline.sample_shape_slat(
                condition_from_sparse(glob, proj, coords32), flow, coords32, {"steps": steps},
            )
            coords64 = learned_c64_support(pipeline, lr, device).cpu()
            atomic_save(out / "c64_support.pt", {"format": FORMAT, "coords": coords64,
                        "block_index": idx, "c32_tokens": int(coords32.shape[0])})
            print(f"[local-lr] block={bid:02d} {order}/{len(pending)} C64={len(coords64):,}", flush=True)
            del glob, proj, lr, coords32, coords64
            empty_cuda()
    finally:
        if was_low_vram:
            pipeline.low_vram = True; flow.cpu(); decoder.cpu()
    del cached
    if pipeline.low_vram: model.cpu()
    empty_cuda()


def assemble_global_support(records: list[dict[str, Any]], output: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    parts = []
    for rec in records:
        bid, idx = int(rec["cube_id"]), rec["block_index"]
        local = torch.load(output / "local_structure" / f"block_{bid:02d}" / "c64_support.pt",
                           map_location="cpu", weights_only=False)["coords"].int()
        xyz = local[:, 1:4] + torch.tensor(idx, dtype=torch.int32) * 64
        parts.append(torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz), 1))
        rec["generated_c64_tokens"] = int(len(local))
    coords = torch.cat(parts, 0).unique(dim=0).int().contiguous()
    xyz = coords[:, 1:4]
    if torch.any(xyz < 0) or torch.any(xyz >= 256):
        raise RuntimeError("assembled support outside Global C256")
    flow_records = []
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    for rec in records:
        bid, idx = int(rec["cube_id"]), rec["block_index"]
        start = torch.tensor(idx, dtype=torch.int32) * 64
        mask = ((xyz >= start) & (xyz < start + 64)).all(1)
        rows = torch.where(mask)[0].long(); local_xyz = xyz[mask] - start
        coverage[rows] += 1
        flow_records.append({"cube_id": bid, "block_index": idx, "start": tuple(start.tolist()),
                             "global_row_ids": rows, "owned_row_ids": rows, "local_xyz": local_xyz})
    if not torch.all(coverage == 1):
        raise RuntimeError("non-overlap block support coverage is not exactly one")
    atomic_save(output / "global_support" / "global_c256_support.pt", {"format": FORMAT, "coords": coords})
    atomic_json(output / "global_support" / "summary.json", {
        "tokens": int(len(coords)), "active_blocks": len(records), "unique_removed": int(sum(len(p) for p in parts)-len(coords)),
        "records": [{k: r[k] for k in ("cube_id", "block_index", "baseline_tokens", "generated_c64_tokens")} for r in records],
    })
    return coords, flow_records


@torch.no_grad()
def build_global_conditions(
    pipeline: Any, image: Image.Image, camera: Mapping[str, float], coords: torch.Tensor,
    records: list[dict[str, Any]], stage: str, output: Path, device: torch.device,
) -> dict[str, Any]:
    root = output / "conditions" / stage
    model = pipeline.image_cond_model_shape_1024 if stage == "shape" else pipeline.image_cond_model_tex_1024
    pending = [r for r in records if not (root / f"block_{int(r['cube_id']):02d}.pt").is_file()]
    if pending:
        cached = extract_full_image_features(model, image, device)
        fov, distance = float(camera["camera_angle_x"]), float(camera["distance"])
        for order, rec in enumerate(pending, 1):
            bid, idx = int(rec["cube_id"]), rec["block_index"]
            local_coords = torch.cat((torch.zeros((len(rec["local_xyz"]), 1), dtype=torch.int32), rec["local_xyz"]), 1).to(device)
            transform = local_to_global_camera_transform(
                model, block_start=tuple(v * 64 for v in idx), global_resolution=256,
                global_extent=64, distance=distance, device=device,
            )
            error = validate_projection_transform(
                model, transform, local_resolution=64, global_resolution=256,
                block_start=tuple(v * 64 for v in idx), distance=distance, fov=fov, device=device,
            )
            glob, proj = project_cached_features(
                model, cached, transform=transform, fov=fov, distance=distance,
                coords=local_coords, grid_resolution=64,
            )
            atomic_save(root / f"block_{bid:02d}.pt", {
                "format": FORMAT, "stage": stage, "cube_id": bid, "block_index": idx,
                "global_row_ids": rec["global_row_ids"], "global": glob.cpu(),
                "proj": proj[0].cpu(), "projection_max_error_pixels": error,
                "image_condition": "full global image; block-specific global projection; no crop",
            })
            print(f"[condition-{stage}] block={bid:02d} {order}/{len(pending)} tokens={len(local_coords):,}", flush=True)
            del glob, proj, local_coords
            empty_cuda()
        del cached
        if pipeline.low_vram: model.cpu()
        empty_cuda()
    cubes = {}
    for rec in records:
        bid = int(rec["cube_id"])
        payload = torch.load(root / f"block_{bid:02d}.pt", map_location="cpu", weights_only=False)
        if not torch.equal(payload["global_row_ids"].long(), rec["global_row_ids"].long()):
            raise RuntimeError(f"cached {stage} condition row mismatch for block {bid}")
        cubes[bid] = payload
    return {"cubes": cubes, "fingerprint_sha256": FORMAT}


@torch.no_grad()
def synchronous_flow(
    *, stage: str, pipeline: Any, records: list[dict[str, Any]], condition: Mapping[str, Any],
    output: Path, device: torch.device, seed: int, steps: int,
    concat: torch.Tensor | None,
) -> torch.Tensor:
    final_path = output / stage / "final_normalized.pt"
    if final_path.is_file():
        return torch.load(final_path, map_location="cpu", weights_only=False)["features"].float()
    model = pipeline.models["shape_slat_flow_model_1024" if stage == "shape" else "tex_slat_flow_model_1024"]
    sampler = pipeline.shape_slat_sampler if stage == "shape" else pipeline.tex_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params if stage == "shape" else pipeline.tex_slat_sampler_params)
    params["steps"] = steps
    channels = int(model.in_channels) if concat is None else int(model.in_channels) - int(concat.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((sum(len(r["global_row_ids"]) for r in records), channels), generator=generator)
    # Records partition global rows exactly; row count equals support size.
    schedule = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    model.to(device).eval(); history = []
    for step, (t, t_next) in enumerate(zip(schedule[:-1], schedule[1:])):
        velocity = torch.empty_like(state); elapsed = 0.0
        for order, rec in enumerate(records, 1):
            values, timing = cube_flow._one_prediction(
                [rec], state, condition, sampler, model, params, float(t), float(t_next), device, concat,
            )
            velocity.index_copy_(0, rec["global_row_ids"], values[0]); elapsed += timing["seconds"]
            print(f"[{stage}] step={step+1}/{steps} block={int(rec['cube_id']):02d} {order}/{len(records)}", flush=True)
        state = cube_flow.jacobi_update(state, velocity, float(t), float(t_next))
        history.append({"step": step, "t": float(t), "t_next": float(t_next), "seconds": elapsed})
        atomic_save(output / stage / f"step_{step+1:02d}.pt", {"features": state})
    model.cpu(); empty_cuda()
    atomic_save(final_path, {"format": FORMAT, "features": state})
    atomic_json(output / stage / "summary.json", {"tokens": len(state), "channels": channels,
                "steps": steps, "seed": seed, "records": history})
    return state


@torch.no_grad()
def decode_global(
    pipeline: Any, coords: torch.Tensor, shape: torch.Tensor, texture: torch.Tensor,
    output: Path, device: torch.device,
) -> dict[str, Any]:
    result_path = output / "decode" / "summary.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    shape_raw = cube_flow.denormalize(shape, pipeline.shape_slat_normalization)
    tex_raw = cube_flow.denormalize(texture, pipeline.tex_slat_normalization)
    atomic_save(output / "decode" / "latents_denormalized.pt", {"coords": coords, "shape": shape_raw, "texture": tex_raw})
    decoded = pipeline.decode_latent(
        SparseTensor(shape_raw.to(device), coords.to(device)),
        SparseTensor(tex_raw.to(device), coords.to(device)), 4096,
    )
    if len(decoded) != 1:
        raise RuntimeError("global decoder did not return one mesh")
    native = decoded[0]
    atomic_save(output / "decode" / "global_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
    vertex, face = expc._native_mesh_to_pbr(native, device)
    atomic_save(output / "decode" / "global_per_vertex_pbr_mesh.pt", {"format": FORMAT, "mesh": vertex})
    atomic_save(output / "decode" / "global_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": face})
    result = {"status": "complete", "tokens": len(coords), "vertices": int(native.vertices.shape[0]),
              "faces": int(native.faces.shape[0]),
              "per_vertex_mesh": str((output / "decode" / "global_per_vertex_pbr_mesh.pt").resolve())}
    atomic_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=base / "global_camera.json")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--output", type=Path, default=Path("outputs/global_c256_restructured_blocks_cuda5"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=43001)
    p.add_argument("--texture-seed", type=int, default=44001)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-after-support", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args(); started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    camera = json.loads(args.camera.read_text()); camera["mesh_scale"] = 1.0
    atomic_json(output / "config.json", {"format": FORMAT, "args": vars(args), "camera": camera,
                "no_halo": True, "block_layout": "C64 split C16 stride16; generated C64 maps to disjoint Global C256 C64 blocks",
                "image_condition": "full image global token + correct global-position projected features; no bbox crop"})
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image512, image1024 = canonical["image_512"], canonical["image_1024"]
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    image512.save(output / "inputs" / "global_512.png")
    image1024.save(output / "inputs" / "global_1024.png")

    baseline64 = build_native_baseline_c64(pipeline, image512, camera, output, args.seed, args.steps, device)
    records = active_blocks(baseline64)
    atomic_json(output / "active_blocks.json", {"count": len(records), "records": records})
    print(f"[active] baseline C64={len(baseline64):,} nonempty blocks={len(records)}", flush=True)
    generate_local_structures(pipeline, image512, camera, records, output, args.seed, args.steps, device)
    generate_local_c64_supports(pipeline, image512, camera, records, output, args.seed, args.steps, device)
    coords, flow_records = assemble_global_support(records, output)
    print(f"[global-support] C256 tokens={len(coords):,}", flush=True)
    if args.stop_after_support:
        print(f"[done-support] {output}", flush=True); return

    shape_condition = build_global_conditions(pipeline, image1024, camera, coords, flow_records, "shape", output, device)
    shape = synchronous_flow(stage="shape", pipeline=pipeline, records=flow_records,
                             condition=shape_condition, output=output, device=device,
                             seed=args.shape_seed, steps=args.steps, concat=None)
    del shape_condition; empty_cuda()
    texture_condition = build_global_conditions(pipeline, image1024, camera, coords, flow_records, "texture", output, device)
    texture = synchronous_flow(stage="texture", pipeline=pipeline, records=flow_records,
                               condition=texture_condition, output=output, device=device,
                               seed=args.texture_seed, steps=args.steps, concat=shape)
    del texture_condition; empty_cuda()
    decoded = decode_global(pipeline, coords, shape, texture, output, device)
    atomic_json(output / "summary.json", {"format": FORMAT, "status": "complete", "active_blocks": len(records),
                "global_c256_tokens": len(coords), "decode": decoded, "seconds": time.perf_counter()-started})
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
