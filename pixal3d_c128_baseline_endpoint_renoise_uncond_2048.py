#!/usr/bin/env python3
"""C2048/C128 texture endpoint re-noise and tiled-unconditional sweep.

The native 1024 baseline supplies twelve predicted texture x0 endpoints and a
shared O-Voxel geometry/material field.  Its geometry is voxelized on a C2048
flexible dual-grid and encoded by the official shape encoder to one fixed C128
shape SLat.  Baseline materials are queried on the same C2048 cells and encoded
by the official texture encoder into twelve aligned C128 texture endpoints.

Variant n skips n model steps: the SAME final endpoint E12 is forward-noised at
the remaining uniform time t[n], then steps n+1..12 are evaluated with zero
image condition in disjoint local-C64 contexts.  The C128 shape latent never
changes.  Every final global C128 texture SLat is decoded once at resolution
2048 and rendered from the front and back.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import o_voxel
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

import pixal3d.models as pixal3d_models
import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_c256_uniform_endpoint_prefix_uncond_sweep as c256_reference
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d.representations import MeshWithVoxel
from pixal3d.utils import render_utils
from render_pixal3d_raw_ovoxel import load_envmap
from used.root_experiments.pixal3d_guided_endpoint_sr import _support_subdivisions


FORMAT = "pixal3d_c128_final_baseline_renoise_uncond_2048_v2"
STEPS = 12
INPUT_GRID = 2048
LATENT_GRID = 128
CONTEXT = 64
STRIDE = 64
FLOW_BATCH_SIZE = 44
DEFAULT_ENDPOINTS = Path("outputs/baseline1024_uniform_texture_endpoints_c1282048_cuda4")
DEFAULT_BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
DEFAULT_OUTPUT = Path("outputs/c128_final_baseline_renoise_uncond_2048_cuda4")
DEFAULT_ENCODER_ROOT = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts"
)
DEFAULT_SHAPE_ENCODER = DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"
DEFAULT_TEXTURE_ENCODER = DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def linear_keys(coords: torch.Tensor, grid: int = LATENT_GRID) -> torch.Tensor:
    xyz = coords.to(torch.int64)
    return xyz[:, 0] * grid * grid + xyz[:, 1] * grid + xyz[:, 2]


def build_records(coords: torch.Tensor) -> tuple[list[dict[str, Any]], torch.Tensor]:
    xyz = coords[:, 1:].cpu().to(torch.int32)
    records: list[dict[str, Any]] = []
    owner = torch.full((coords.shape[0],), -1, dtype=torch.int16)
    cube_id = 0
    for sx in (0, 64):
        for sy in (0, 64):
            for sz in (0, 64):
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                rows = torch.where(((xyz >= start) & (xyz < start + CONTEXT)).all(1))[0].long()
                local = xyz.index_select(0, rows) - start
                owner[rows] = cube_id
                records.append({
                    "cube_id": cube_id,
                    "start": (sx, sy, sz),
                    "global_row_ids": rows,
                    "owned_row_ids": rows,
                    "local_xyz": local,
                })
                cube_id += 1
    if len(records) != 8 or bool((owner < 0).any()):
        raise RuntimeError("C128 must be covered once by eight disjoint C64 contexts")
    return records, owner


@torch.no_grad()
def prepare_c2048(
    geometry_path: Path,
    cache: Path,
    resume: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if resume and cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[C2048] cache hit rows={saved['coords'].shape[0]:,}", flush=True)
        return saved["coords"], saved["dual_vertices"], saved["intersected"]
    vertices, faces, _ = cube_flow._load_mesh_geometry(geometry_path)
    print("[C2048] baseline geometry -> flexible dual-grid", flush=True)
    coords, dual_world, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=INPUT_GRID,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )
    coords = coords.to(torch.int32).cpu().contiguous()
    dual = (
        dual_world.float().cpu() * INPUT_GRID - coords.float()
    ).clamp(0, 1).contiguous()
    intersected = intersected.cpu().contiguous()
    atomic_save(cache, {
        "format": FORMAT,
        "grid": INPUT_GRID,
        "coords": coords,
        "dual_vertices": dual,
        "intersected": intersected,
    })
    print(f"[C2048] saved rows={coords.shape[0]:,}", flush=True)
    return coords, dual, intersected


@torch.no_grad()
def encode_fixed_shape(
    coords2048: torch.Tensor,
    dual2048: torch.Tensor,
    intersected2048: torch.Tensor,
    encoder_path: Path,
    cache: Path,
    device: torch.device,
    resume: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if resume and cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[shape encoder] cache hit C128 rows={saved['coords'].shape[0]:,}", flush=True)
        return saved["coords"].int(), saved["raw_features"].float()
    coords4 = torch.cat((torch.zeros_like(coords2048[:, :1]), coords2048), 1).to(device)
    sparse = SparseTensor(dual2048.to(device), coords4)
    intersected = sparse.replace(intersected2048.to(device))
    encoder = pixal3d_models.from_pretrained(str(encoder_path)).eval().to(device)
    print("[shape encoder] official C2048 -> fixed C128 shape SLat", flush=True)
    encoded = encoder(sparse, intersected, sample_posterior=False)
    coords = encoded.coords.detach().int().cpu().contiguous()
    features = encoded.feats.detach().float().cpu().contiguous()
    order = torch.argsort(linear_keys(coords[:, 1:]), stable=True)
    coords = coords.index_select(0, order)
    features = features.index_select(0, order)
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= LATENT_GRID)).any()):
        raise RuntimeError("shape encoder output is not a C128 support")
    if torch.unique(linear_keys(coords[:, 1:])).numel() != coords.shape[0]:
        raise RuntimeError("shape encoder returned duplicate C128 coordinates")
    atomic_save(cache, {
        "format": FORMAT,
        "coords": coords,
        "raw_features": features,
        "source": "C2048 flexible dual-grid -> official shape encoder",
    })
    print(f"[shape encoder] C128 rows={coords.shape[0]:,}", flush=True)
    encoder.cpu()
    del encoder, encoded, sparse, intersected, coords4
    empty_cuda()
    return coords, features


def sort_encoded(
    encoded: SparseTensor,
    target_coords: torch.Tensor,
) -> torch.Tensor:
    keys = linear_keys(encoded.coords[:, 1:].detach().cpu())
    target_keys = linear_keys(target_coords[:, 1:].cpu())
    order = torch.argsort(keys, stable=True)
    if not torch.equal(keys.index_select(0, order), target_keys):
        raise RuntimeError("texture encoder C128 support differs from fixed shape support")
    return encoded.feats.index_select(0, order.to(encoded.device)).detach().float().cpu()


@torch.no_grad()
def query_baseline_field(
    field_path: Path,
    coords2048: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    payload = torch.load(field_path, map_location="cpu", weights_only=False)
    field = MeshWithVoxel(
        vertices=torch.empty((1, 3), device=device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=device),
        origin=torch.as_tensor(payload["origin"]).tolist(),
        voxel_size=float(payload["voxel_size"]),
        coords=payload["coords"].to(device=device, dtype=torch.int32),
        attrs=payload["attrs"].to(device=device, dtype=torch.float32),
        voxel_shape=torch.Size(payload["voxel_shape"]),
        layout=dict(payload["layout"]),
    )
    parts: list[torch.Tensor] = []
    for start in range(0, coords2048.shape[0], chunk_size):
        points = (
            coords2048[start : start + chunk_size].to(device).float() + 0.5
        ) / INPUT_GRID - 0.5
        parts.append(field.query_attrs(points).detach().float().cpu())
    result = torch.cat(parts, 0).contiguous()
    del field, payload, parts
    empty_cuda()
    return result


@torch.no_grad()
def encode_texture_endpoints(
    pipeline: Any,
    endpoint_root: Path,
    output: Path,
    coords2048: torch.Tensor,
    target_coords: torch.Tensor,
    encoder_path: Path,
    device: torch.device,
    chunk_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    coords4 = torch.cat((torch.zeros_like(coords2048[:, :1]), coords2048), 1).to(device)
    print("[texture support] building exact C2048 -> C128 guide hierarchy", flush=True)
    guide_subs = _support_subdivisions(coords4, levels=4)
    guide_keys = torch.sort(linear_keys(guide_subs[0].coords[:, 1:].cpu())).values
    if not torch.equal(guide_keys, linear_keys(target_coords[:, 1:])):
        raise RuntimeError("C2048 guide hierarchy does not reproduce fixed C128 support")
    encoder = pixal3d_models.from_pretrained(str(encoder_path)).eval().to(device)
    mean = torch.as_tensor(pipeline.tex_slat_normalization["mean"], device=device)[None]
    std = torch.as_tensor(pipeline.tex_slat_normalization["std"], device=device)[None]
    records: list[dict[str, Any]] = []
    for step in range(1, STEPS + 1):
        path = output / "encoded_c128_endpoints" / f"step_{step:02d}.pt"
        if resume and path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            records.append({k: saved[k] for k in ("step", "t", "t_next", "path", "normalized_sha256")})
            print(f"[texture encoder {step:02d}/12] cache hit", flush=True)
            continue
        source = endpoint_root / "ovoxel1024" / f"texture_endpoint_step_{step:02d}.pt"
        meta = torch.load(source, map_location="cpu", weights_only=False)
        print(f"[material query {step:02d}/12] C2048 rows={coords2048.shape[0]:,}", flush=True)
        attrs = query_baseline_field(source, coords2048, device, chunk_size)
        sparse = SparseTensor(attrs.to(device).mul_(2.0).sub_(1.0), coords4)
        encoded = encoder(sparse, sample_posterior=False, guide_subs=guide_subs)
        raw = sort_encoded(SparseTensor(encoded.feats, encoded.coords), target_coords)
        normalized = (raw.to(device) - mean) / std
        normalized_cpu = normalized.detach().float().cpu().contiguous()
        record = {
            "step": step,
            "t": float(meta["t"]),
            "t_next": float(meta["t_next"]),
            "path": str(path.resolve()),
            "normalized_sha256": tensor_hash(normalized_cpu),
        }
        atomic_save(path, {
            "format": FORMAT,
            **record,
            "coords": target_coords,
            "raw_features": raw,
            "normalized_features": normalized_cpu,
            "source_ovoxel1024": str(source.resolve()),
            "material_query": "baseline 1024 O-Voxel queried at C2048 cell centers",
        })
        records.append(record)
        del meta, attrs, sparse, encoded, raw, normalized, normalized_cpu
        empty_cuda()
    encoder.cpu()
    del encoder, guide_subs, coords4
    empty_cuda()
    return records


def validate_final_endpoint_cache(saved, expected_format, coords, endpoint_hash, noise_hash):
    record = saved["record"]
    if (saved.get("format") != expected_format
            or not torch.equal(saved["coords"].int(), coords.int())
            or record.get("endpoint_step") != STEPS
            or record.get("final_endpoint_sha256") != endpoint_hash
            or record.get("noise_sha256") != noise_hash):
        raise RuntimeError("Incompatible flow cache: use a fresh output directory for fixed-E12 runs")


@torch.no_grad()
def run_variants(
    pipeline: Any,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    owner: torch.Tensor,
    shape_normalized: torch.Tensor,
    endpoint_root: Path,
    output: Path,
    device: torch.device,
    noise_seed: int,
    resume: bool,
) -> list[dict[str, Any]]:
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"].eval().to(device)
    schedule = sampler.timestep_schedule(STEPS, 1.0)
    sigma_min = float(sampler.sigma_min)
    endpoint_payloads = [
        torch.load(endpoint_root / f"step_{step:02d}.pt", map_location="cpu", weights_only=False)
        for step in range(1, STEPS + 1)
    ]
    final_payload = endpoint_payloads[-1]
    if not torch.equal(final_payload["coords"].int(), coords.int()):
        raise RuntimeError("final endpoint support mismatch")
    final_endpoint = final_payload["normalized_features"].float().to(device)
    final_endpoint_hash = tensor_hash(final_endpoint)
    channels = int(final_endpoint.shape[1])
    epsilon = torch.randn(
        (coords.shape[0], channels),
        generator=torch.Generator(device="cpu").manual_seed(int(noise_seed)),
    ).float()
    fixed_noise = epsilon.to(device)
    noise_hash = tensor_hash(epsilon)
    groups = cube_flow.pack_groups(
        records,
        FLOW_BATCH_SIZE,
        10_000_000,
        require_owned=True,
    )
    if len(groups) != 1 or len(groups[0]) != 8:
        raise RuntimeError(
            f"expected one physical batch with 8 active C64 contexts, got {[len(g) for g in groups]}"
        )
    results: list[dict[str, Any]] = []
    for n in range(1, STEPS + 1):
        final_path = output / "flow" / f"prefix_{n:02d}" / "final_texture_normalized.pt"
        if resume and final_path.is_file():
            saved = torch.load(final_path, map_location="cpu", weights_only=False)
            validate_final_endpoint_cache(saved, FORMAT, coords, final_endpoint_hash, noise_hash)
            results.append(saved["record"])
            print(f"[C128 variant {n:02d}/12] cache hit", flush=True)
            continue
        start_t = float(schedule[n])
        epsilon_weight = sigma_min + (1.0 - sigma_min) * start_t
        state = ((1.0 - start_t) * final_endpoint + epsilon_weight * fixed_noise).cpu()
        atomic_save(output / "noised_states" / f"n_{n:02d}.pt", {
            "coords": coords, "normalized_features": state, "t": start_t,
            "final_endpoint_sha256": final_endpoint_hash, "noise_sha256": noise_hash,
        })
        step_rows: list[dict[str, Any]] = []
        print(
            f"[C128 variant {n:02d}/12] E12 re-noise, skipped steps={n}, "
            f"start_t={start_t:.6f}, uncond_steps={STEPS-n}",
            flush=True,
        )
        for step_index in range(n, STEPS):
            t_now = float(schedule[step_index])
            t_next = float(schedule[step_index + 1])
            step_started = time.perf_counter()
            proposals: list[tuple[int, torch.Tensor, torch.Tensor]] = []
            for group in groups:
                proposals.extend(
                    c256_reference.unconditional_velocity(
                        group,
                        state,
                        shape_normalized,
                        sampler,
                        model,
                        t_now,
                        t_next,
                        device,
                    )
                )
            velocity = cube_flow.validate_owner_scatter(owner, proposals, channels)
            state = cube_flow.jacobi_update(state, velocity, t_now, t_next)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"variant {n} step {step_index+1} produced non-finite state")
            step_rows.append({
                "step": step_index + 1,
                "t": t_now,
                "t_next": t_next,
                "route": "image_unconditional_local_C64",
                "seconds": time.perf_counter() - step_started,
            })
        record = {
            "n": n,
            "skipped_prefix_steps": n,
            "endpoint_step": STEPS,
            "final_endpoint_sha256": final_endpoint_hash,
            "noise_sha256": noise_hash,
            "unconditional_suffix_steps": STEPS - n,
            "start_t": start_t,
            "clean_endpoint_weight": 1.0 - start_t,
            "epsilon_weight": epsilon_weight,
            "shape_c128_fixed": True,
            "context": CONTEXT,
            "stride": STRIDE,
            "configured_flow_batch_size": FLOW_BATCH_SIZE,
            "effective_batch_size": 8,
            "steps": step_rows,
            "state_sha256": tensor_hash(state),
            "path": str(final_path.resolve()),
        }
        atomic_save(final_path, {
            "format": FORMAT,
            "coords": coords,
            "normalized_features": state,
            "record": record,
        })
        results.append(record)
        print(f"[C128 variant {n:02d}/12] flow complete", flush=True)
    atomic_save(output / "flow" / "fixed_noise.pt", {
        "format": FORMAT,
        "seed": noise_seed,
        "coords": coords,
        "epsilon": epsilon,
        "sha256": tensor_hash(epsilon),
    })
    model.cpu()
    del model, endpoint_payloads, epsilon, final_endpoint, fixed_noise
    empty_cuda()
    return results


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    if value.ndim == 3:
        value = value.permute(1, 2, 0)
    elif value.ndim != 2:
        raise ValueError(f"unexpected render tensor shape {tuple(value.shape)}")
    return np.nan_to_num(value.numpy(), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def save_image(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(value, 0.0, 1.0)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[..., 0]
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_render(result: Mapping[str, torch.Tensor], output: Path) -> dict[str, np.ndarray]:
    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, np.ndarray] = {}
    for mode in ("shaded", "base_color", "metallic", "roughness", "alpha", "normal", "mask"):
        if mode not in result:
            raise KeyError(f"renderer did not return {mode}")
        saved[mode] = tensor_to_numpy(result[mode])
        save_image(saved[mode], output / f"{mode}.png")
    save_image(saved["shaded"], output / "render.png")
    # Evaluate saved PNG values on both fresh runs and resume, matching the gated run.
    return load_render(output)


def load_render(output: Path) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(Image.open(output / filename), dtype=np.float32) / 255.0
        for key, filename in (("shaded", "render.png"), ("mask", "mask.png"))
    }


def compare_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    foreground: np.ndarray,
) -> dict[str, float]:
    delta = prediction - reference
    full_mse = float(np.mean(delta**2))
    fg_mse = float(np.mean(delta[foreground] ** 2))
    _, score = structural_similarity(
        reference,
        prediction,
        data_range=1.0,
        channel_axis=2,
        full=True,
    )
    return {
        "psnr_db": 10.0 * math.log10(1.0 / max(full_mse, 1e-12)),
        "foreground_psnr_db": 10.0 * math.log10(1.0 / max(fg_mse, 1e-12)),
        "ssim": float(np.mean(score)),
        "foreground_ssim": float(np.mean(score[foreground])),
        "foreground_mae": float(np.mean(np.abs(delta[foreground]))),
        "foreground_pixels": int(foreground.sum()),
    }


def error_map(
    reference: np.ndarray,
    prediction: np.ndarray,
    foreground: np.ndarray,
    scale: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.mean(np.abs(prediction - reference), axis=2)
    value = np.clip(error / scale, 0.0, 1.0)
    heat = np.stack(
        (
            np.clip(3.0 * value, 0.0, 1.0),
            np.clip(3.0 * value - 1.0, 0.0, 1.0),
            np.clip(3.0 * value - 2.0, 0.0, 1.0),
        ),
        axis=2,
    )
    heat[~foreground] = 0
    return error.astype(np.float32), heat.astype(np.float32)


def load_baseline_mesh(endpoint_root: Path, device: torch.device) -> MeshWithVoxel:
    geometry = torch.load(
        endpoint_root / "ovoxel1024" / "shared_geometry.pt",
        map_location="cpu",
        weights_only=False,
    )
    field = torch.load(
        endpoint_root / "ovoxel1024" / "texture_endpoint_step_12.pt",
        map_location="cpu",
        weights_only=False,
    )
    return MeshWithVoxel(
        geometry["vertices"].to(device),
        geometry["faces"].to(device),
        origin=torch.as_tensor(field["origin"]).tolist(),
        voxel_size=float(field["voxel_size"]),
        coords=field["coords"].to(device),
        attrs=field["attrs"].to(device),
        voxel_shape=torch.Size(field["voxel_shape"]),
        layout=dict(field["layout"]),
    )


def camera_views(camera: Mapping[str, float]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    front, intrinsics = render_utils.proj_camera_to_render_params(
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
    )
    turn = torch.tensor(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=front.dtype,
        device=front.device,
    )
    return {"front": front, "back": front @ turn}, intrinsics


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def make_grid(
    records: Sequence[Mapping[str, Any]],
    output: Path,
    image_key: str,
    metric_key: str,
    title: str,
) -> None:
    columns, rows = 4, 3
    thumb, label_h, gap = 1024, 224, 28
    canvas = Image.new(
        "RGB",
        (
            columns * thumb + (columns + 1) * gap,
            rows * (thumb + label_h) + (rows + 1) * gap,
        ),
        (18, 20, 25),
    )
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = gap + column * (thumb + gap)
        y = gap + row * (thumb + label_h + gap)
        scores = record[metric_key]
        draw.text(
            (x + 10, y + 8),
            f"n={record['n']:02d}/12 · {title}",
            fill="white",
            font=font(48),
        )
        suffix_steps = int(
            record["suffix_steps"]
            if "suffix_steps" in record
            else record["unconditional_suffix_steps"]
        )
        suffix_label = str(record.get("suffix_label", "uncond"))
        draw.text(
            (x + 10, y + 76),
            f"E12 + noise · {suffix_label} {suffix_steps:02d}",
            fill=(255, 204, 102),
            font=font(39),
        )
        draw.text(
            (x + 10, y + 136),
            f"FG PSNR {scores['foreground_psnr_db']:.2f} · SSIM {scores['foreground_ssim']:.4f}",
            fill=(102, 204, 255),
            font=font(35),
        )
        with Image.open(record[image_key]) as source:
            tile = source.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y + label_h))
    canvas.save(output)


def make_metric_plot(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    steps = np.asarray([int(row["n"]) for row in records])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    for column, (key, label) in enumerate(
        (("front_vs_gt", "Front vs GT"), ("back_vs_baseline", "Back vs baseline"))
    ):
        psnr = [float(row[key]["foreground_psnr_db"]) for row in records]
        ssim = [float(row[key]["foreground_ssim"]) for row in records]
        axes[0, column].plot(steps, psnr, "o-", linewidth=2.2)
        axes[1, column].plot(steps, ssim, "o-", linewidth=2.2)
        axes[0, column].set_title(f"{label}: foreground PSNR")
        axes[1, column].set_title(f"{label}: foreground SSIM")
    for row, ylabel in enumerate(("PSNR (dB)", "SSIM")):
        for axis in axes[row]:
            axis.set_xlabel("Skipped steps n / 12 (fixed E12 re-noise)")
            axis.set_ylabel(ylabel)
            axis.set_xticks(steps)
            axis.grid(alpha=0.3)
    fig.suptitle("C2048 decode from tiled C128 texture flow", fontsize=17)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


@torch.no_grad()
def decode_render_evaluate(
    pipeline: Any,
    shape_raw: torch.Tensor,
    coords: torch.Tensor,
    variants: list[dict[str, Any]],
    endpoint_root: Path,
    baseline_root: Path,
    camera: Mapping[str, float],
    output: Path,
    device: torch.device,
    render_resolution: int,
    face_chunk_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    views, intrinsics = camera_views(camera)
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": render_resolution,
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "peel_layers": 8,
            "face_chunk_size": face_chunk_size,
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)

    def render_views(mesh: MeshWithVoxel, root: Path) -> dict[str, dict[str, np.ndarray]]:
        outputs: dict[str, dict[str, np.ndarray]] = {}
        for name, extrinsics in views.items():
            view_root = root / name
            if resume and (view_root / "render.png").is_file() and (view_root / "mask.png").is_file():
                outputs[name] = load_render(view_root)
                continue
            torch.cuda.manual_seed_all(100_000 + (0 if name == "front" else 180))
            result = renderer.render(
                mesh,
                extrinsics.to(device),
                intrinsics.to(device),
                envmap=envmap,
                use_envmap_bg=False,
            )
            outputs[name] = save_render(result, view_root)
            del result
            empty_cuda()
        return outputs

    print("[baseline render] native 1024 endpoint, front and back", flush=True)
    baseline_mesh = load_baseline_mesh(endpoint_root, device)
    baseline_render = render_views(baseline_mesh, output / "baseline1024_reference")
    del baseline_mesh
    empty_cuda()

    gt = np.asarray(
        Image.open(baseline_root / "canonical_1024.png")
        .convert("RGB")
        .resize((render_resolution, render_resolution), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    gt_foreground = np.asarray(
        Image.open(baseline_root / "raw_ovoxel_render" / "alpha.png")
        .convert("L")
        .resize((render_resolution, render_resolution), Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 127
    back_mask = baseline_render["back"]["mask"]
    if back_mask.ndim == 3:
        back_mask = back_mask[..., 0]
    back_foreground = back_mask > 0.5

    print("[decode] fixed C128 shape -> shared 2048 geometry", flush=True)
    shape = SparseTensor(shape_raw.to(device), coords.to(device))
    shape_meshes, subs = pipeline.decode_shape_slat(shape, 2048)
    geometry = shape_meshes[0]
    tex_mean, tex_std = cube_flow._norm_tensors(pipeline.tex_slat_normalization, 32)
    evaluated: list[dict[str, Any]] = []
    for variant in variants:
        n = int(variant["n"])
        root = output / "variants" / f"n_{n:02d}"
        print(f"[decode/render {n:02d}/12] global C128 -> 2048, front/back", flush=True)
        saved = torch.load(variant["path"], map_location="cpu", weights_only=False)
        texture_raw = saved["normalized_features"].float() * tex_std + tex_mean
        texture = SparseTensor(texture_raw.to(device), coords.to(device))
        tex_voxel = pipeline.decode_tex_slat(texture, subs)[0]
        mesh = MeshWithVoxel(
            geometry.vertices,
            geometry.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1.0 / 2048,
            coords=tex_voxel.coords[:, 1:],
            attrs=tex_voxel.feats,
            voxel_shape=torch.Size([*tex_voxel.shape, *tex_voxel.spatial_shape]),
            layout=dict(pipeline.pbr_attr_layout),
        )
        renders = render_views(mesh, root / "renders")
        front_scores = compare_metrics(gt, renders["front"]["shaded"], gt_foreground)
        back_scores = compare_metrics(
            baseline_render["back"]["shaded"],
            renders["back"]["shaded"],
            back_foreground,
        )
        front_error, front_heat = error_map(gt, renders["front"]["shaded"], gt_foreground)
        back_error, back_heat = error_map(
            baseline_render["back"]["shaded"],
            renders["back"]["shaded"],
            back_foreground,
        )
        np.save(root / "front_vs_gt_absolute_error.npy", front_error)
        np.save(root / "back_vs_baseline_absolute_error.npy", back_error)
        front_error_path = root / "front_vs_gt_error_map.png"
        back_error_path = root / "back_vs_baseline_error_map.png"
        save_image(front_heat, front_error_path)
        save_image(back_heat, back_error_path)
        record = {
            **variant,
            "front_render": str((root / "renders" / "front" / "render.png").resolve()),
            "back_render": str((root / "renders" / "back" / "render.png").resolve()),
            "front_error_map": str(front_error_path.resolve()),
            "back_error_map": str(back_error_path.resolve()),
            "front_vs_gt": front_scores,
            "back_vs_baseline": back_scores,
        }
        atomic_json(root / "metrics.json", record)
        evaluated.append(record)
        del saved, texture_raw, texture, tex_voxel, mesh, renders
        empty_cuda()
    return evaluated


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--texture-encoder", type=Path, default=DEFAULT_TEXTURE_ENCODER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=44)
    parser.add_argument("--query-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [value.strip() for value in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.physical_cuda}"
        )
    if args.render_resolution != 1024:
        raise ValueError("this experiment keeps every rendered subfigure at 1024 pixels")
    endpoint_summary_path = args.endpoint_dir / "summary.json"
    if not endpoint_summary_path.is_file():
        raise FileNotFoundError(f"missing completed baseline endpoint summary: {endpoint_summary_path}")
    endpoint_summary = json.loads(endpoint_summary_path.read_text(encoding="utf-8"))
    if endpoint_summary.get("status") != "complete":
        raise RuntimeError("baseline endpoint run is not complete")
    if endpoint_summary.get("texture_steps") != STEPS or endpoint_summary.get("texture_rescale_t") != 1.0:
        raise RuntimeError("baseline endpoints must use 12 uniform texture steps")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)

    coords2048, dual2048, intersected2048 = prepare_c2048(
        args.endpoint_dir / "ovoxel1024" / "shared_geometry.pt",
        output / "c2048" / "flexible_dual_grid.pt",
        args.resume,
    )
    coords128, shape_raw = encode_fixed_shape(
        coords2048,
        dual2048,
        intersected2048,
        args.shape_encoder,
        output / "support" / "fixed_c128_shape_slat.pt",
        device,
        args.resume,
    )
    shape_normalized = cube_flow.normalize(shape_raw, pipeline.shape_slat_normalization)
    atomic_save(output / "support" / "fixed_c128_shape_normalized.pt", {
        "format": FORMAT,
        "coords": coords128,
        "normalized_features": shape_normalized,
        "sha256": tensor_hash(shape_normalized),
    })
    cube_records, owner = build_records(coords128)
    atomic_json(output / "support" / "c128_c64_layout.json", {
        "global_tokens": int(coords128.shape[0]),
        "context": CONTEXT,
        "stride": STRIDE,
        "active_contexts": len(cube_records),
        "configured_batch_size": FLOW_BATCH_SIZE,
        "effective_batch_size": len(cube_records),
        "contexts": [
            {"cube_id": int(row["cube_id"]), "start": list(row["start"]),
             "tokens": int(row["global_row_ids"].numel())}
            for row in cube_records
        ],
    })
    endpoint_records = encode_texture_endpoints(
        pipeline,
        args.endpoint_dir.resolve(),
        output,
        coords2048,
        coords128,
        args.texture_encoder,
        device,
        args.query_chunk_size,
        args.resume,
    )
    c2048_tokens = int(coords2048.shape[0])
    del coords2048, dual2048, intersected2048
    empty_cuda()

    variants = run_variants(
        pipeline,
        coords128,
        cube_records,
        owner,
        shape_normalized,
        output / "encoded_c128_endpoints",
        output,
        device,
        args.noise_seed,
        args.resume,
    )
    camera = json.loads((args.baseline_dir / "global_camera.json").read_text(encoding="utf-8"))
    evaluated = decode_render_evaluate(
        pipeline,
        shape_raw,
        coords128,
        variants,
        args.endpoint_dir.resolve(),
        args.baseline_dir.resolve(),
        camera,
        output,
        device,
        args.render_resolution,
        args.face_chunk_size,
        args.resume,
    )

    visuals = {
        "front_renders_3x4": output / "front_renders_3x4.png",
        "front_error_maps_3x4": output / "front_error_maps_3x4.png",
        "back_renders_3x4": output / "back_renders_3x4.png",
        "back_error_maps_3x4": output / "back_error_maps_3x4.png",
        "foreground_metrics": output / "foreground_metrics.png",
    }
    make_grid(evaluated, visuals["front_renders_3x4"], "front_render", "front_vs_gt", "front vs GT")
    make_grid(evaluated, visuals["front_error_maps_3x4"], "front_error_map", "front_vs_gt", "front error")
    make_grid(evaluated, visuals["back_renders_3x4"], "back_render", "back_vs_baseline", "back vs baseline")
    make_grid(evaluated, visuals["back_error_maps_3x4"], "back_error_map", "back_vs_baseline", "back error")
    make_metric_plot(evaluated, visuals["foreground_metrics"])

    summary = {
        "format": FORMAT,
        "status": "complete",
        "semantics": (
            "variant n forward-noises the SAME encoded final baseline endpoint E12 at uniform t[n], then runs "
            "image-unconditional local-C64 texture flow for human steps n+1..12; fixed C128 shape concat"
        ),
        "physical_cuda": args.physical_cuda,
        "input_grid": INPUT_GRID,
        "latent_grid": LATENT_GRID,
        "decode_resolution": INPUT_GRID,
        "render_resolution": args.render_resolution,
        "schedule": pipeline.tex_slat_sampler.timestep_schedule(STEPS, 1.0),
        "context": CONTEXT,
        "stride": STRIDE,
        "configured_flow_batch_size": FLOW_BATCH_SIZE,
        "effective_flow_batch_size": len(cube_records),
        "c2048_tokens": c2048_tokens,
        "c128_tokens": int(coords128.shape[0]),
        "fixed_shape_raw_sha256": tensor_hash(shape_raw),
        "fixed_shape_normalized_sha256": tensor_hash(shape_normalized),
        "baseline_endpoint_summary": str(endpoint_summary_path.resolve()),
        "endpoint_records": endpoint_records,
        "results": evaluated,
        "visuals": {key: str(path.resolve()) for key, path in visuals.items()},
    }
    atomic_json(output / "summary.json", summary)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
