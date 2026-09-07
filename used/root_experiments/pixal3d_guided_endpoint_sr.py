#!/usr/bin/env python3
"""Guided 1024-endpoint -> C4096 voxel -> C256 SLat SR experiment.

The input is a paired shape/texture x0 endpoint saved by
``pixal3d_baseline1024_texture_endpoints.py``.  The endpoint is decoded at its
native 1024 resolution, re-voxelized at 4096, encoded on one explicitly saved
four-level support hierarchy, used as the analytical endpoint of the first
C256 Euler step, and then sampled normally for the remaining flow steps.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import o_voxel
import torch
from PIL import Image

import pixal3d.models as pixal3d_models
import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as native_sr
import pixal3d_render_global4096_multiview as multiview
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel


FORMAT = "pixal3d_guided_endpoint_sr_v1"
DEFAULT_ENDPOINT_DIR = Path("outputs/baseline1024_shape_tex_endpoints")
DEFAULT_OUTPUT = Path("outputs/guided_endpoint_sr")
DEFAULT_ENCODER_ROOT = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS.2-4B/ckpts"
)


def _atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _normalization(normalization: Mapping[str, Any], value: SparseTensor) -> tuple[torch.Tensor, torch.Tensor]:
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.dtype)[None]
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.dtype)[None]
    return std, mean


def _denormalize(normalization: Mapping[str, Any], value: SparseTensor) -> SparseTensor:
    std, mean = _normalization(normalization, value)
    return value.replace(value.feats * std + mean)


def _normalize(normalization: Mapping[str, Any], value: SparseTensor) -> SparseTensor:
    std, mean = _normalization(normalization, value)
    return value.replace((value.feats - mean) / std)


def _load_endpoint(path: Path, device: torch.device) -> SparseTensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SparseTensor(
        payload["endpoint_normalized_feats"].to(device=device, dtype=torch.float32),
        payload["coords"].to(device=device, dtype=torch.int32),
    )


def _support_subdivisions(leaf_coords: torch.Tensor, levels: int = 4) -> list[SparseTensor]:
    """Create explicit coarse-to-fine subs from a C4096 leaf support."""
    if leaf_coords.ndim != 2 or leaf_coords.shape[1] != 4:
        raise ValueError("leaf_coords must have shape [N, 4]")
    current = leaf_coords
    fine_to_coarse: list[SparseTensor] = []
    for _ in range(levels):
        parent_xyz = current[:, 1:] // 2
        parent4 = torch.cat([current[:, :1], parent_xyz], dim=1)
        parents, inverse = torch.unique(parent4, dim=0, sorted=True, return_inverse=True)
        child_xyz = current[:, 1:] % 2
        child_index = child_xyz[:, 0] + 2 * child_xyz[:, 1] + 4 * child_xyz[:, 2]
        selected = torch.zeros((parents.shape[0], 8), device=current.device, dtype=torch.bool)
        selected[inverse, child_index.long()] = True
        logits = torch.where(selected, torch.ones_like(selected, dtype=torch.float32), -torch.ones_like(selected, dtype=torch.float32))
        fine_to_coarse.append(SparseTensor(logits, parents.int()))
        current = parents.int()
    return list(reversed(fine_to_coarse))


def _voxelize_shape(mesh: Any, resolution: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords, dual_world, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=mesh.vertices.detach().cpu().float(),
        faces=mesh.faces.detach().cpu().int(),
        grid_size=int(resolution),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )
    coords = coords.int().cpu()
    dual = (dual_world.float().cpu() * resolution - coords.float()).clamp(0, 1)
    return coords, dual, intersected.cpu()


@torch.no_grad()
def _voxelize_texture(
    decoded: MeshWithVoxel,
    coords: torch.Tensor,
    resolution: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    live = decoded.to(device)
    rows = []
    for start in range(0, coords.shape[0], chunk_size):
        centers = (coords[start : start + chunk_size].to(device).float() + 0.5) / resolution - 0.5
        rows.append(live.query_attrs(centers).detach().cpu().float())
    return torch.cat(rows, dim=0)


def _slat_payload(value: SparseTensor) -> dict[str, torch.Tensor]:
    return {"coords": value.coords.detach().cpu(), "feats": value.feats.detach().cpu()}


@torch.no_grad()
def _run_flows(
    pipeline: Any,
    image: Image.Image,
    camera: Mapping[str, float],
    shape_guide_raw: SparseTensor,
    texture_guide_raw: SparseTensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SparseTensor, SparseTensor, dict[str, Any]]:
    coords = shape_guide_raw.coords
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image],
        coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=256,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1000)
    shape_noise = SparseTensor(
        torch.randn((coords.shape[0], shape_model.in_channels), generator=generator).to(device),
        coords,
    )
    shape_endpoint = _normalize(pipeline.shape_slat_normalization, shape_guide_raw)
    shape_params = dict(pipeline.shape_slat_sampler_params)
    shape_params.update({"steps": args.steps, "return_model_history": False})
    if pipeline.low_vram:
        shape_model.to(device)
    shape_result = pipeline.shape_slat_sampler.sample(
        shape_model,
        shape_noise,
        first_step_endpoint=shape_endpoint,
        **shape_cond,
        **shape_params,
        verbose=True,
        tqdm_desc="Guided C256 shape flow",
    ).samples
    if pipeline.low_vram:
        shape_model.cpu()
    shape_raw = _denormalize(pipeline.shape_slat_normalization, shape_result)
    del shape_cond, shape_noise, shape_endpoint, shape_result
    _empty_cache()

    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=256,
    )
    shape_normalized = _normalize(pipeline.shape_slat_normalization, shape_raw)
    texture_channels = int(texture_model.in_channels) - shape_normalized.feats.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 2000)
    texture_noise = shape_normalized.replace(
        torch.randn((coords.shape[0], texture_channels), generator=generator).to(device)
    )
    texture_endpoint = _normalize(pipeline.tex_slat_normalization, texture_guide_raw)
    texture_params = dict(pipeline.tex_slat_sampler_params)
    texture_params.update({"steps": args.steps, "return_model_history": False})
    if pipeline.low_vram:
        texture_model.to(device)
    texture_result = pipeline.tex_slat_sampler.sample(
        texture_model,
        texture_noise,
        concat_cond=shape_normalized,
        first_step_endpoint=texture_endpoint,
        **texture_cond,
        **texture_params,
        verbose=True,
        tqdm_desc="Guided C256 texture flow",
    ).samples
    if pipeline.low_vram:
        texture_model.cpu()
    texture_raw = _denormalize(pipeline.tex_slat_normalization, texture_result)
    meta = {
        "steps": int(args.steps),
        "first_step": "official _xstart_to_pred(noise,t=1,encoded_baseline_x0) + Euler",
        "remaining_steps": "ordinary conditioned model Euler flow",
        "shape_tokens": int(shape_raw.coords.shape[0]),
        "texture_tokens": int(texture_raw.coords.shape[0]),
    }
    return shape_raw, texture_raw, meta


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    endpoint_dir = args.endpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = json.loads((endpoint_dir / "global_camera.json").read_text())
    image = Image.open(endpoint_dir / "canonical_1024.png").convert("RGB")
    pipeline = init_pipeline(str(args.model_path), device="cuda", low_vram=args.low_vram)

    shape_normalized = _load_endpoint(
        endpoint_dir / "shape_endpoint_latents" / f"step_{args.endpoint_step:02d}.pt", device
    )
    texture_normalized = _load_endpoint(
        endpoint_dir / "texture_endpoint_latents" / f"step_{args.endpoint_step:02d}.pt", device
    )
    shape_endpoint = _denormalize(pipeline.shape_slat_normalization, shape_normalized)
    texture_endpoint = _denormalize(pipeline.tex_slat_normalization, texture_normalized)
    print(f"[decode] paired baseline endpoint step={args.endpoint_step}", flush=True)
    meshes, baseline_subs = pipeline.decode_shape_slat(shape_endpoint, 1024)
    texture_voxels = pipeline.decode_tex_slat(texture_endpoint, baseline_subs)
    decoded_texture = MeshWithVoxel(
        meshes[0].vertices,
        meshes[0].faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 1024,
        coords=texture_voxels[0].coords[:, 1:],
        attrs=texture_voxels[0].feats,
        voxel_shape=torch.Size([*texture_voxels[0].shape, *texture_voxels[0].spatial_shape]),
        layout=pipeline.pbr_attr_layout,
    )

    voxel_cache = output_dir / "voxel4096" / "paired_endpoint_voxels.pt"
    if voxel_cache.is_file() and args.resume:
        voxel_payload = torch.load(voxel_cache, map_location="cpu", weights_only=False)
        coords4096 = voxel_payload["coords"]
        dual4096 = voxel_payload["dual_vertices"]
        intersected4096 = voxel_payload["intersected"]
        attrs4096 = voxel_payload["attrs"]
    else:
        print("[voxelize] shape endpoint -> flexible dual grid C4096", flush=True)
        coords4096, dual4096, intersected4096 = _voxelize_shape(meshes[0], 4096)
        print(f"[voxelize] texture endpoint -> C4096 rows={coords4096.shape[0]:,}", flush=True)
        attrs4096 = _voxelize_texture(
            decoded_texture, coords4096, 4096, device, args.query_chunk_size
        )
        _atomic_save(voxel_cache, {
            "coords": coords4096, "dual_vertices": dual4096,
            "intersected": intersected4096, "attrs": attrs4096,
        })

    del meshes, baseline_subs, texture_voxels, decoded_texture
    del shape_normalized, texture_normalized, shape_endpoint, texture_endpoint
    _empty_cache()

    coords4 = torch.cat([torch.zeros_like(coords4096[:, :1]), coords4096], dim=1).to(device)
    guide_subs = _support_subdivisions(coords4, levels=4)
    _atomic_save(output_dir / "guided_encoder" / "guide_subs.pt", [
        {"coords": sub.coords.cpu(), "feats": sub.feats.cpu()} for sub in guide_subs
    ])
    print(
        "[guided enc] support " + " -> ".join(
            [str(coords4096.shape[0])] + [str(sub.coords.shape[0]) for sub in reversed(guide_subs)]
        ), flush=True,
    )
    encoded_path = output_dir / "guided_encoder" / "encoded_slats.pt"
    feasibility_path = output_dir / "guided_encoder" / "feasibility.json"
    if args.resume and encoded_path.is_file() and feasibility_path.is_file():
        encoded_payload = torch.load(encoded_path, map_location="cpu", weights_only=False)
        shape_encoded = SparseTensor(
            encoded_payload["shape"]["feats"].to(device),
            encoded_payload["shape"]["coords"].to(device),
        )
        texture_encoded = SparseTensor(
            encoded_payload["texture"]["feats"].to(device),
            encoded_payload["texture"]["coords"].to(device),
        )
        feasibility = json.loads(feasibility_path.read_text())
        print(f"[guided enc] cache hit C256 tokens={shape_encoded.coords.shape[0]:,}", flush=True)
    else:
        # Encode sequentially. At C4096 the leaf activations dominate memory.
        shape_encoder = pixal3d_models.from_pretrained(str(args.shape_encoder)).eval().to(device)
        vertex_sparse = SparseTensor(dual4096.to(device), coords4.int())
        intersected_sparse = vertex_sparse.replace(intersected4096.to(device))
        shape_encoded, shape_diag = shape_encoder(
            vertex_sparse, intersected_sparse, sample_posterior=False,
            guide_subs=guide_subs, return_guide_diagnostics=True,
        )
        # Drop the four levels of inherited spatial maps before texture encode.
        shape_encoded = SparseTensor(shape_encoded.feats, shape_encoded.coords)
        shape_encoder.cpu()
        del shape_encoder, vertex_sparse, intersected_sparse
        _empty_cache()

        texture_encoder = pixal3d_models.from_pretrained(str(args.texture_encoder)).eval().to(device)
        pbr_sparse = SparseTensor(attrs4096.to(device) * 2 - 1, coords4.int())
        texture_encoded, texture_diag = texture_encoder(
            pbr_sparse, sample_posterior=False,
            guide_subs=guide_subs, return_guide_diagnostics=True,
        )
        texture_encoded = SparseTensor(texture_encoded.feats, texture_encoded.coords)
        if not torch.equal(shape_encoded.coords, texture_encoded.coords):
            raise RuntimeError("guided shape and texture encoders produced different C256 supports")
        if not torch.equal(shape_encoded.coords, guide_subs[0].coords):
            raise RuntimeError("guided C256 support does not equal the requested support")
        feasibility = {
            "status": "passed",
            "leaf_c4096_tokens": int(coords4096.shape[0]),
            "slat_c256_tokens": int(shape_encoded.coords.shape[0]),
            "shape_texture_coords_exactly_equal": True,
            "requested_coords_exactly_equal": True,
            "shape": shape_diag,
            "texture": texture_diag,
        }
        _atomic_save(encoded_path, {
            "shape": _slat_payload(shape_encoded), "texture": _slat_payload(texture_encoded)
        })
        _atomic_json(feasibility_path, feasibility)
        print(f"[guided enc] PASSED C256 tokens={shape_encoded.coords.shape[0]:,}", flush=True)
        texture_encoder.cpu()
        del texture_encoder, pbr_sparse
        _empty_cache()

    if args.guided_enc_only:
        summary = {"format": FORMAT, "status": "guided_encoder_passed", "feasibility": feasibility}
        _atomic_json(output_dir / "summary.json", summary)
        return summary

    shape_final, texture_final, flow_meta = _run_flows(
        pipeline, image, camera, shape_encoded, texture_encoded, args, device
    )
    _atomic_save(output_dir / "flow" / "final_slats.pt", {
        "shape": _slat_payload(shape_final), "texture": _slat_payload(texture_final)
    })
    print("[decode] final C256 SLat -> 4096", flush=True)
    final_mesh = pipeline.decode_latent(shape_final, texture_final, 4096)[0]
    _atomic_save(output_dir / "final" / "material_mesh.pt", {"format": FORMAT, "mesh": final_mesh.cpu()})
    vertex_mesh, face_mesh = native_sr._native_mesh_to_pbr(final_mesh, device)
    vertex_path = output_dir / "final" / "per_vertex_pbr_mesh.pt"
    _atomic_save(vertex_path, {"format": FORMAT, "mesh": vertex_mesh})
    _atomic_save(output_dir / "final" / "per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": face_mesh})
    camera_path = output_dir / "global_camera.json"
    _atomic_json(camera_path, {key: float(value) for key, value in camera.items()})
    render_meta = {}
    if args.render:
        render_meta = multiview.render(SimpleNamespace(
            mesh=vertex_path, camera=camera_path, output_dir=output_dir / "multiview",
            angles=args.angles, resolution=args.render_resolution,
            face_chunk_size=args.face_chunk_size, device=str(device), force=args.force_render,
        ))
    summary = {
        "format": FORMAT,
        "status": "complete",
        "endpoint_step": int(args.endpoint_step),
        "feasibility": feasibility,
        "flow": flow_meta,
        "decode_resolution": 4096,
        "render": render_meta,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16")
    parser.add_argument("--texture-encoder", type=Path, default=DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16")
    parser.add_argument("--endpoint-step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--query-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guided-enc-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--angles", default="0,120,240")
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--force-render", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.endpoint_step < 12:
        raise ValueError("--endpoint-step must be in [0, 11]")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 for guided-first + normal flow")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
