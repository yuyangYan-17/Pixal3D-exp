#!/usr/bin/env python3
"""C256 texture sweep: baseline endpoint-guided prefix, image-unconditional suffix."""
from __future__ import annotations

import argparse
import gc
import hashlib
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

import numpy as np
import o_voxel
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

import pixal3d.models as pixal3d_models
import pixal3d_global_c256_cube_owner_flow_singleview as base
import pixal3d_global_c256_encdown_context1024_flow_singleview as encdown
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel
from render_pixal3d_raw_ovoxel import load_envmap, render_static_ovoxel, save_render_outputs
from used.root_experiments.pixal3d_guided_endpoint_sr import _support_subdivisions


FORMAT = "pixal3d_c256_uniform_endpoint_prefix_uncond_sweep_v1"
STEPS = 12
DEFAULT_ENDPOINTS = Path("outputs/baseline1024_uniform_texture_endpoints_cuda4")
DEFAULT_BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
DEFAULT_OUTPUT = Path("outputs/c256_uniform_endpoint_prefix_uncond_sweep_strict_cuda4")
DEFAULT_ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
)
DEFAULT_SHAPE_ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
)


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


def sparse_from_payload(path: Path, feature_key: str, device: torch.device) -> SparseTensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SparseTensor(
        payload[feature_key].to(device=device, dtype=torch.float32),
        payload["coords"].to(device=device, dtype=torch.int32),
    )


def sort_to_reference(value: SparseTensor, reference: torch.Tensor) -> SparseTensor:
    value_keys = base.linear_keys(value.coords[:, 1:].detach().cpu(), encdown.GRID)
    ref_keys = base.linear_keys(reference[:, 1:].cpu(), encdown.GRID)
    order = torch.argsort(value_keys, stable=True)
    sorted_keys = value_keys.index_select(0, order)
    if not torch.equal(sorted_keys, ref_keys):
        raise RuntimeError("texture encoder C256 support differs from fixed shape support")
    order_device = order.to(value.feats.device)
    return SparseTensor(
        value.feats.index_select(0, order_device),
        value.coords.index_select(0, order_device),
    )


@torch.no_grad()
def prepare_c4096(
    baseline_mesh: Path, cache: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[C4096] cache hit rows={payload['coords'].shape[0]:,}", flush=True)
        return payload["coords"], payload["dual_vertices"], payload["intersected"]
    vertices, faces, _ = base._load_mesh_geometry(baseline_mesh)
    print("[C4096] baseline mesh -> flexible dual-grid", flush=True)
    coords, dual_world, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=4096,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )
    coords = coords.to(torch.int32).cpu().contiguous()
    dual = (dual_world.float().cpu() * 4096.0 - coords.float()).clamp(0, 1).contiguous()
    intersected = intersected.cpu().contiguous()
    atomic_save(cache, {"format": FORMAT, "coords": coords, "dual_vertices": dual, "intersected": intersected})
    print(f"[C4096] saved rows={coords.shape[0]:,}", flush=True)
    return coords, dual, intersected


@torch.no_grad()
def prepare_shape_support(
    coords4096: torch.Tensor,
    dual4096: torch.Tensor,
    intersected4096: torch.Tensor,
    encoder_path: Path,
    cache: Path,
    device: torch.device,
) -> torch.Tensor:
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[shape support] cache hit C256 rows={payload['coords'].shape[0]:,}", flush=True)
        return payload["coords"].to(torch.int32).contiguous()
    coords4 = torch.cat((torch.zeros_like(coords4096[:, :1]), coords4096), 1).to(device)
    vertex_sparse = SparseTensor(dual4096.to(device), coords4)
    intersected_sparse = vertex_sparse.replace(intersected4096.to(device))
    encoder = pixal3d_models.from_pretrained(str(encoder_path)).eval().to(device)
    print("[shape support] official shape encoder C4096 -> C256", flush=True)
    latent = encoder(vertex_sparse, intersected_sparse, sample_posterior=False)
    coords = latent.coords.detach().to(torch.int32).cpu().contiguous()
    order = torch.argsort(base.linear_keys(coords[:, 1:], encdown.GRID), stable=True)
    coords = coords.index_select(0, order)
    if torch.unique(base.linear_keys(coords[:, 1:], encdown.GRID)).numel() != coords.shape[0]:
        raise RuntimeError("official shape encoder returned duplicate C256 support")
    atomic_save(cache, {
        "format": FORMAT,
        "coords": coords,
        "source": "fresh baseline shared geometry -> C4096 flexible dual-grid -> official shape encoder",
    })
    print(f"[shape support] saved C256 rows={coords.shape[0]:,}", flush=True)
    encoder.cpu()
    del encoder, latent, vertex_sparse, intersected_sparse, coords4
    empty_cuda()
    return coords


@torch.no_grad()
def prepare_shape_flow(
    pipeline: Any,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    owner: torch.Tensor,
    image_path: Path,
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    resume: bool,
) -> torch.Tensor:
    final_path = output_dir / "shape" / "final_state_normalized.pt"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if not torch.equal(payload["coords"].int(), coords):
            raise RuntimeError("cached shape flow support mismatch")
        print("[shape flow] cache hit", flush=True)
        return payload["features"].float().contiguous()
    condition_args = SimpleNamespace(
        condition_image_4096=str(image_path),
        velocity_fusion="owner",
    )
    condition = encdown.build_condition(
        pipeline, condition_args, coords, records, camera, "shape", output_dir
    )
    model = pipeline.models["shape_slat_flow_model_1024"]
    channels = int(model.in_channels)
    noise = torch.randn(
        (coords.shape[0], channels), generator=torch.Generator().manual_seed(43)
    ).float()
    atomic_save(output_dir / "shape" / "initial_noise.pt", {
        "format": FORMAT, "coords": coords, "features": noise, "seed": 43,
    })
    params = dict(pipeline.shape_slat_sampler_params)
    params["steps"] = STEPS
    shape_norm, _ = base.run_flow(
        "shape", noise, coords, records, owner, condition,
        pipeline.shape_slat_sampler, model, params, output_dir, device,
        encdown.FLOW_BATCH_SIZE, 10_000_000, None, resume, "owner", 32.0,
    )
    atomic_save(final_path, {"format": FORMAT, "coords": coords, "features": shape_norm})
    return shape_norm


@torch.no_grad()
def query_endpoint_field(
    field_path: Path,
    coords4096: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    payload = torch.load(field_path, map_location="cpu", weights_only=False)
    query_mesh = MeshWithVoxel(
        vertices=torch.empty((1, 3), device=device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=device),
        origin=torch.as_tensor(payload["origin"]).tolist(),
        voxel_size=float(payload["voxel_size"]),
        coords=payload["coords"].to(device=device, dtype=torch.int32),
        attrs=payload["attrs"].to(device=device, dtype=torch.float32),
        voxel_shape=torch.Size(payload["voxel_shape"]),
        layout=dict(payload["layout"]),
    )
    rows: list[torch.Tensor] = []
    for start in range(0, coords4096.shape[0], chunk_size):
        points = (coords4096[start : start + chunk_size].to(device).float() + 0.5) / 4096.0 - 0.5
        rows.append(query_mesh.query_attrs(points).detach().float().cpu())
    result = torch.cat(rows, 0).contiguous()
    del query_mesh, payload, rows
    empty_cuda()
    return result


@torch.no_grad()
def encode_all_endpoints(
    pipeline: Any,
    endpoint_dir: Path,
    output_dir: Path,
    coords4096: torch.Tensor,
    dual4096: torch.Tensor,
    intersected4096: torch.Tensor,
    target_coords: torch.Tensor,
    texture_encoder_path: Path,
    device: torch.device,
    query_chunk_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    coords4 = torch.cat((torch.zeros_like(coords4096[:, :1]), coords4096), 1).to(device)
    print("[guide support] build exact C4096 -> C256 hierarchy", flush=True)
    guide_subs = _support_subdivisions(coords4, levels=4)
    guide_keys = base.linear_keys(guide_subs[0].coords[:, 1:].cpu(), encdown.GRID)
    target_keys = base.linear_keys(target_coords[:, 1:].cpu(), encdown.GRID)
    if not torch.equal(torch.sort(guide_keys).values, target_keys):
        raise RuntimeError("C4096 hierarchy leaf does not reproduce fixed C256 support")
    encoder = pixal3d_models.from_pretrained(str(texture_encoder_path)).eval().to(device)
    mean = torch.as_tensor(pipeline.tex_slat_normalization["mean"], device=device)[None]
    std = torch.as_tensor(pipeline.tex_slat_normalization["std"], device=device)[None]
    records: list[dict[str, Any]] = []
    for step in range(1, STEPS + 1):
        encoded_path = output_dir / "encoded_c256_endpoints" / f"step_{step:02d}.pt"
        if resume and encoded_path.is_file():
            saved = torch.load(encoded_path, map_location="cpu", weights_only=False)
            records.append({k: saved[k] for k in ("step", "t", "t_next", "path", "raw_sha256", "normalized_sha256")})
            print(f"[encode endpoint] step={step:02d} cache hit", flush=True)
            continue
        field_path = endpoint_dir / "ovoxel1024" / f"texture_endpoint_step_{step:02d}.pt"
        field_meta = torch.load(field_path, map_location="cpu", weights_only=False)
        print(f"[query endpoint] step={step:02d} C4096 rows={coords4096.shape[0]:,}", flush=True)
        attrs = query_endpoint_field(field_path, coords4096, device, query_chunk_size)
        encoder_input = attrs.to(device).mul_(2.0).sub_(1.0)
        sparse = SparseTensor(encoder_input, coords4)
        encoded = encoder(
            sparse,
            sample_posterior=False,
            guide_subs=guide_subs,
        )
        encoded = SparseTensor(encoded.feats, encoded.coords)
        encoded = sort_to_reference(encoded, target_coords)
        normalized = (encoded.feats.float() - mean) / std
        raw_cpu = encoded.feats.detach().float().cpu().contiguous()
        norm_cpu = normalized.detach().float().cpu().contiguous()
        record = {
            "step": step,
            "t": float(field_meta["t"]),
            "t_next": float(field_meta["t_next"]),
            "path": str(encoded_path.resolve()),
            "raw_sha256": tensor_hash(raw_cpu),
            "normalized_sha256": tensor_hash(norm_cpu),
        }
        atomic_save(encoded_path, {
            "format": FORMAT, **record, "coords": target_coords,
            "raw_feats": raw_cpu, "normalized_feats": norm_cpu,
            "source_ovoxel1024": str(field_path.resolve()),
            "material_query": "C4096 cell centers into native MeshWithVoxel.query_attrs",
        })
        records.append(record)
        print(f"[encode endpoint] step={step:02d} C256 rows={target_coords.shape[0]:,}", flush=True)
        del field_meta, attrs, encoder_input, sparse, encoded, normalized, raw_cpu, norm_cpu
        empty_cuda()
    encoder.cpu()
    del encoder, guide_subs, coords4
    empty_cuda()
    return records


@torch.no_grad()
def unconditional_velocity(
    group: Sequence[Mapping[str, Any]],
    state: torch.Tensor,
    shape_concat: torch.Tensor,
    sampler: Any,
    model: Any,
    t: float,
    t_next: float,
    device: torch.device,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    packed = base._pack_state(group, state, device)
    concat = base._pack_concat(group, shape_concat, device)
    batch = len(group)
    zero_proj = SparseTensor(torch.zeros((packed.feats.shape[0], 2048), device=device), packed.coords)
    zero_cond = {"global": torch.zeros((batch, 5, 1024), device=device), "proj": zero_proj}
    result = sampler.sample_once(
        model, packed, float(t), float(t_next),
        cond=zero_cond, neg_cond=zero_cond, concat_cond=concat,
        guidance_strength=0.0, guidance_rescale=0.0, guidance_interval=(0.0, 1.0),
    )
    parts = base.legacy._split_sparse_batch(result.pred_v, batch, "unconditional texture velocity")
    proposals: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for rec, part in zip(group, parts):
        if not torch.equal(part.coords.cpu(), base._local_coords(rec)):
            raise RuntimeError("texture model changed local sparse row order")
        proposals.append((int(rec["cube_id"]), rec["global_row_ids"], part.feats.detach().float().cpu()))
    return proposals


@torch.no_grad()
def run_sweep(
    pipeline: Any,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    owner: torch.Tensor,
    shape_norm: torch.Tensor,
    endpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
    resume: bool,
) -> list[dict[str, Any]]:
    model = pipeline.models["tex_slat_flow_model_1024"].eval().to(device)
    sampler = pipeline.tex_slat_sampler
    schedule = sampler.timestep_schedule(STEPS, 1.0)
    channels = int(model.in_channels) - int(shape_norm.shape[1])
    noise = torch.randn((coords.shape[0], channels), generator=torch.Generator().manual_seed(seed)).float()
    groups = base.pack_groups(records, encdown.FLOW_BATCH_SIZE, 10_000_000, require_owned=True)
    endpoints = [
        torch.load(endpoint_dir / f"step_{step:02d}.pt", map_location="cpu", weights_only=False)["normalized_feats"].float()
        for step in range(1, STEPS + 1)
    ]
    for value in endpoints:
        if value.shape != noise.shape:
            raise RuntimeError("encoded endpoint feature shape mismatch")
    results: list[dict[str, Any]] = []
    for prefix in range(1, STEPS + 1):
        final_path = output_dir / "flow" / f"guided_prefix_{prefix:02d}" / "final_state_normalized.pt"
        if resume and final_path.is_file():
            payload = torch.load(final_path, map_location="cpu", weights_only=False)
            results.append(payload["record"])
            print(f"[C256 sweep] prefix={prefix:02d} cache hit", flush=True)
            continue
        state = noise.clone()
        step_rows: list[dict[str, Any]] = []
        for step, (t, t_next) in enumerate(zip(schedule[:-1], schedule[1:])):
            started = time.perf_counter()
            if step < prefix:
                velocity = sampler._xstart_to_pred(state, float(t), endpoints[step])
                route = "encoded_baseline_endpoint"
            else:
                proposals: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                for group in groups:
                    proposals.extend(unconditional_velocity(
                        group, state, shape_norm, sampler, model, t, t_next, device
                    ))
                velocity = base.validate_owner_scatter(owner, proposals, channels)
                route = "image_unconditional_model"
            state = base.jacobi_update(state, velocity, t, t_next)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"non-finite texture state prefix={prefix} step={step+1}")
            step_rows.append({
                "step": step + 1, "t": float(t), "t_next": float(t_next), "route": route,
                "seconds": time.perf_counter() - started,
            })
        record = {
            "guided_prefix_steps": prefix,
            "unconditional_suffix_steps": STEPS - prefix,
            "schedule": schedule,
            "shape_concat_retained": True,
            "image_condition_in_suffix": "zero DINO global/proj tokens",
            "steps": step_rows,
            "state_sha256": tensor_hash(state),
            "path": str(final_path.resolve()),
        }
        atomic_save(final_path, {"format": FORMAT, "coords": coords, "features": state, "record": record})
        results.append(record)
        print(f"[C256 sweep] prefix={prefix:02d}/12 complete", flush=True)
    model.cpu()
    del model, endpoints, noise
    empty_cuda()
    return results


def metrics(reference: np.ndarray, prediction: np.ndarray, foreground: np.ndarray) -> dict[str, float]:
    delta = prediction - reference
    mse = float(np.mean(delta * delta))
    fg_mse = float(np.mean(delta[foreground] ** 2))
    _, ssim_map = structural_similarity(reference, prediction, data_range=1.0, channel_axis=2, full=True)
    return {
        "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
        "foreground_psnr_db": 10.0 * math.log10(1.0 / max(fg_mse, 1e-12)),
        "ssim": float(np.mean(ssim_map)),
        "foreground_ssim": float(np.mean(ssim_map[foreground])),
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


@torch.no_grad()
def decode_render_grid(
    pipeline: Any,
    shape_norm: torch.Tensor,
    coords: torch.Tensor,
    records: list[dict[str, Any]],
    camera: Mapping[str, float],
    reference_image: Path,
    foreground_mask: Path,
    output_dir: Path,
    device: torch.device,
    resolution: int,
    resume: bool,
) -> Path:
    shape_raw = base.denormalize(shape_norm, pipeline.shape_slat_normalization)
    tex_mean, tex_std = base._norm_tensors(pipeline.tex_slat_normalization, 32)
    image = Image.open(reference_image).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    reference = np.asarray(image, dtype=np.float32) / 255.0
    foreground = np.asarray(
        Image.open(foreground_mask).convert("L").resize((resolution, resolution), Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 127
    envmap = load_envmap("studio", device=device)
    rendered_paths: list[Path] = []
    for record in records:
        prefix = int(record["guided_prefix_steps"])
        render_dir = output_dir / "renders" / f"guided_prefix_{prefix:02d}"
        render_path = render_dir / "render.png"
        if not (resume and render_path.is_file()):
            saved = torch.load(record["path"], map_location="cpu", weights_only=False)
            tex_raw = saved["features"].float() * tex_std + tex_mean
            shape = SparseTensor(shape_raw.to(device), coords.to(device))
            texture = SparseTensor(tex_raw.to(device), coords.to(device))
            print(f"[decode4096] guided prefix={prefix:02d}", flush=True)
            mesh = pipeline.decode_latent(shape, texture, 4096)[0]
            rendered = render_static_ovoxel(
                mesh,
                camera_angle_x=float(camera["camera_angle_x"]),
                distance=float(camera["distance"]),
                resolution=resolution,
                envmap=envmap,
                ssaa=1,
                peel_layers=8,
                face_chunk_size=4_000_000,
                use_envmap_bg=False,
                verbose=False,
            )
            save_render_outputs(rendered, render_dir)
            del saved, tex_raw, shape, texture, mesh, rendered
            empty_cuda()
        prediction = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
        record.update(metrics(reference, prediction, foreground))
        rendered_paths.append(render_path)

    thumb, label_h, margin = 512, 84, 12
    columns, rows = 3, 4
    sheet = Image.new("RGB", (columns * thumb + 4 * margin, rows * (thumb + label_h) + 5 * margin), (18, 21, 27))
    draw = ImageDraw.Draw(sheet)
    for index, (record, path) in enumerate(zip(records, rendered_paths)):
        row, col = divmod(index, columns)
        x = margin + col * (thumb + margin)
        y = margin + row * (thumb + label_h + margin)
        prefix = int(record["guided_prefix_steps"])
        switch_t = 1.0 - prefix / STEPS
        draw.text((x + 6, y + 4), f"GUIDED PREFIX {prefix}/12", fill="white", font=font(21))
        draw.text((x + 6, y + 33), f"then UNCOND from t={switch_t:.4f}", fill=(255, 208, 112), font=font(17))
        draw.text((x + 6, y + 58), f"PSNR {record['psnr_db']:.2f} · SSIM {record['ssim']:.3f}", fill=(112, 205, 255), font=font(17))
        with Image.open(path) as rendered:
            tile = rendered.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + label_h))
    grid = output_dir / "guided_prefix_unconditional_suffix_grid_3x4.png"
    sheet.save(grid)
    return grid


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--texture-encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--texture-seed", type=int, default=44)
    parser.add_argument("--query-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [x.strip() for x in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.physical_cuda}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    endpoint_summary = json.loads((args.endpoint_dir / "summary.json").read_text(encoding="utf-8"))
    if endpoint_summary.get("texture_rescale_t") != 1.0:
        raise RuntimeError("baseline endpoints are not on the uniform texture schedule")
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)

    coords4096, dual4096, intersected4096 = prepare_c4096(
        args.endpoint_dir / "ovoxel1024/shared_geometry.pt",
        out / "c4096" / "flexible_dual_grid.pt",
    )
    coords = prepare_shape_support(
        coords4096, dual4096, intersected4096, args.shape_encoder,
        out / "support" / "global_c256_encoder_support.pt", device,
    )
    records, _ = encdown.build_cube_records(coords)
    owner, _ = base.build_owner_map(coords, records)
    camera = json.loads((args.baseline_dir / "global_camera.json").read_text(encoding="utf-8"))
    shape_norm = prepare_shape_flow(
        pipeline, coords, records, owner,
        args.baseline_dir / "canonical_1024.png", camera, out, device, args.resume,
    )
    endpoint_records = encode_all_endpoints(
        pipeline, args.endpoint_dir.resolve(), out, coords4096, dual4096, intersected4096,
        coords, args.texture_encoder, device, args.query_chunk_size, args.resume,
    )
    c4096_tokens = int(coords4096.shape[0])
    del coords4096, dual4096, intersected4096
    empty_cuda()
    sweep_records = run_sweep(
        pipeline, coords, records, owner, shape_norm,
        out / "encoded_c256_endpoints", out, device, args.texture_seed, args.resume,
    )
    grid = decode_render_grid(
        pipeline, shape_norm, coords, sweep_records, camera,
        args.baseline_dir / "canonical_1024.png",
        args.baseline_dir / "raw_ovoxel_render/alpha.png",
        out, device, args.render_resolution, args.resume,
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "semantics": "first n uniform C256 Euler steps use matching encoded baseline C64 x0 endpoints; remaining 12-n steps use zero image condition; C256 shape concat retained",
        "texture_rescale_t_baseline": 1.0,
        "texture_rescale_t_c256": 1.0,
        "schedule": [1.0 - i / STEPS for i in range(STEPS + 1)],
        "flow_batch_size": encdown.FLOW_BATCH_SIZE,
        "baseline_shared_geometry": str((args.endpoint_dir / "ovoxel1024/shared_geometry.pt").resolve()),
        "c4096_tokens": c4096_tokens,
        "c256_tokens": int(coords.shape[0]),
        "endpoint_records": endpoint_records,
        "results": sweep_records,
        "grid": str(grid.resolve()),
    }
    atomic_json(out / "summary.json", summary)
    print(f"[done] {grid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
