#!/usr/bin/env python3
"""Transplant an independently flowed head into a stable global C128 support."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from scipy.spatial import cKDTree

from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import Mesh
from pixal3d_c128_block_local_cascade_shape_renoise_2048 import empty_cuda
from pixal3d_c128_local_decode_merge_experiment import load_decoder_pipeline


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL = ROOT / "outputs/c128_head_crop_native_c2048_baseline_cuda4/latents/shape_c128_denormalized.pt"
DEFAULT_BASE = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/conditional/n_08/shape_c128_decode_input.pt"
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json"
DEFAULT_REFERENCE = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/inputs/canonical_1024.png"
DEFAULT_MASK = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/raw_ovoxel_render/alpha.png"
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_TRANSFORM = ROOT / "outputs/head_similarity_fresh_vs_inherited_cuda4/phase_a/metrics.json"
DEFAULT_OUT = ROOT / "outputs/c128_head_feature_transplant_cuda4"


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_transform(path: Path) -> tuple[float, torch.Tensor, torch.Tensor]:
    sim = json.loads(path.read_text(encoding="utf-8"))["similarity"]
    return float(sim["scale"]), torch.tensor(sim["rotation"], dtype=torch.float32), torch.tensor(sim["translation"], dtype=torch.float32)


def load_base(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coords = payload["coords"].int()
    raw = payload.get("raw_features", payload.get("features")).float()
    if coords.ndim != 2 or coords.shape[1] != 4 or raw.ndim != 2 or raw.shape[0] != coords.shape[0]:
        raise ValueError("invalid global C128 decode input")
    if torch.unique(coords, dim=0).shape[0] != len(coords):
        raise ValueError("global C128 decode input has duplicate coordinates")
    return coords, raw


def load_local(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coords = payload["coords"].int()
    features = payload.get("features", payload.get("raw_features"))
    if isinstance(features, SparseTensor):
        features = features.feats
    features = features.float().cpu()
    if coords.ndim != 2 or coords.shape[1] != 4 or features.shape[0] != coords.shape[0]:
        raise ValueError("invalid local C128 state")
    return coords, features


def map_nearest(
    global_coords: torch.Tensor,
    local_coords: torch.Tensor,
    local_features: torch.Tensor,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Map global C128 cell centers into local C128 coordinates and query NN."""
    g = (global_coords[:, 1:].float() + 0.5) / 128.0 - 0.5
    local_q = ((g - translation[None]) @ rotation) / float(scale)
    local_cell = (local_q + 0.5) * 128.0 - 0.5
    tree = cKDTree(local_coords[:, 1:].numpy().astype(np.float32))
    distances, indices = tree.query(local_cell.numpy().astype(np.float32), k=1, workers=-1)
    distances_t = torch.from_numpy(distances.astype(np.float32))
    indices_t = torch.from_numpy(indices.astype(np.int64))
    mapped_global = scale * (((local_coords[:, 1:].float() + 0.5) / 128.0 - 0.5) @ rotation.T) + translation[None]
    return local_cell, distances_t, indices_t, {
        "mapped_local_q_bbox": [local_q.min(0).values.tolist(), local_q.max(0).values.tolist()],
        "mapped_local_c128_bbox": [local_cell.min(0).values.tolist(), local_cell.max(0).values.tolist()],
        "mapped_local_support_global_q_bbox": [mapped_global.min(0).values.tolist(), mapped_global.max(0).values.tolist()],
    }


def render_mesh(mesh: Mesh, output: Path, camera: dict[str, Any], reference: Path, mask: Path, angles: tuple[int, ...], device: torch.device) -> dict[str, Any]:
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    live = mesh.to(device)
    try:
        return render_geometry(live, camera, reference, mask, output, 1024, angles, 200_000)
    finally:
        del live
        empty_cuda()


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-state", type=Path, default=DEFAULT_LOCAL)
    p.add_argument("--base-state", type=Path, default=DEFAULT_BASE)
    p.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-mask", type=Path, default=DEFAULT_MASK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--radii", default="1.0,2.0,3.0,4.0")
    p.add_argument("--blend-alpha", type=float, default=0.65)
    p.add_argument("--angles", default="0,60,180,240")
    args = p.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    camera_payload = json.loads(args.camera.resolve().read_text(encoding="utf-8"))
    camera = camera_payload.get("camera", camera_payload)
    base_coords, base_raw = load_base(args.base_state.resolve())
    local_coords, local_raw = load_local(args.local_state.resolve())
    scale, rotation, translation = load_transform(args.transform.resolve())
    local_cell, distances, indices, map_meta = map_nearest(base_coords, local_coords, local_raw, scale, rotation, translation)
    radii = [float(x.strip()) for x in args.radii.split(",") if x.strip()]
    if not radii or any(x <= 0 for x in radii):
        raise ValueError("radii must be positive")
    atomic_json(
        out / "manifest.json",
        {
            "format": "pixal3d_c128_local_feature_transplant_v1",
            "local_state": str(args.local_state.resolve()),
            "base_state": str(args.base_state.resolve()),
            "base_tokens": int(len(base_coords)),
            "local_tokens": int(len(local_coords)),
            "similarity": {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()},
            "mapping": map_meta,
            "radii_in_local_c128_cells": radii,
            "blend_alpha": float(args.blend_alpha),
            "note": "global support is kept fixed; only raw C128 features are replaced/interpolated from the independently flowed head",
        },
    )
    variants: list[tuple[str, torch.Tensor, dict[str, Any]]] = []
    for radius in radii:
        selected = distances <= radius
        replacement = base_raw.clone()
        replacement[selected] = local_raw.index_select(0, indices[selected])
        variants.append(
            (
                f"replace_r{radius:g}",
                replacement,
                {"radius": radius, "selected_tokens": int(selected.sum()), "distance_mean_selected": float(distances[selected].mean()) if selected.any() else None},
            )
        )
        if radius == radii[-1]:
            blend = base_raw.clone()
            weights = torch.clamp(1.0 - distances / radius, 0.0, 1.0) * float(args.blend_alpha)
            selected_blend = weights > 0
            local_selected = local_raw.index_select(0, indices[selected_blend])
            blend[selected_blend] = blend[selected_blend] * (1.0 - weights[selected_blend, None]) + local_selected * weights[selected_blend, None]
            variants.append(
                (
                    f"blend_r{radius:g}_a{args.blend_alpha:g}",
                    blend,
                    {"radius": radius, "selected_tokens": int(selected_blend.sum()), "blend_alpha": float(args.blend_alpha)},
                )
            )
    pipeline = load_decoder_pipeline(args.model_path.resolve(), device)
    angles = tuple(int(x.strip()) % 360 for x in args.angles.split(",") if x.strip())
    rows = []
    for name, raw, meta in variants:
        variant = out / name
        print(f"[decode] {name} tokens={len(base_coords):,} mapped={meta['selected_tokens']:,}", flush=True)
        slat = SparseTensor(raw.to(device), base_coords.to(device))
        meshes, _ = pipeline.decode_shape_slat(slat, 2048)
        if len(meshes) != 1:
            raise RuntimeError(f"decoder returned {len(meshes)} meshes")
        mesh = meshes[0]
        atomic_save(variant / "geometry_mesh.pt", {"format": "pixal3d_c128_local_feature_transplant_v1", "mesh": mesh.cpu()})
        del slat, meshes
        empty_cuda()
        print(f"[render] {name} vertices={len(mesh.vertices):,} faces={len(mesh.faces):,}", flush=True)
        render = render_mesh(mesh.cpu(), variant, camera, args.reference.resolve(), args.reference_mask.resolve(), angles, device)
        row = {"name": name, **meta, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "render": render}
        atomic_json(variant / "summary.json", row)
        rows.append(row)
        del mesh
        gc.collect()
        empty_cuda()
    atomic_json(out / "summary.json", {"format": "pixal3d_c128_local_feature_transplant_v1", "status": "complete", "variants": rows})
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
