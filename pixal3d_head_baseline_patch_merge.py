#!/usr/bin/env python3
"""Patch a full C2048 mesh with the decoded geometry of an independent head run.

This is an ablation for the question: can the detail of a native 1024 head
crop survive if it is transferred after the local run, instead of being
represented by a global C128 latent?  The crop mesh is aligned to the full
object with the previously measured similarity transform, then its spatial
box is used to remove overlapping base faces before concatenation.

It is deliberately a mesh-level experiment.  It does not claim that the
result is a valid global SLat state; it tests the strongest direct baseline
teacher signal before attempting latent transplantation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch

from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import Mesh

from pixal3d_c128_block_local_cascade_shape_renoise_2048 import empty_cuda
from pixal3d_c128_local_decode_merge_experiment import load_decoder_pipeline


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_STATE = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4/latents/shape_c64_denormalized.pt"
DEFAULT_LOCAL_MESH = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4/final/geometry_mesh.pt"
DEFAULT_BASE_MESH = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/conditional/n_08/geometry_mesh.pt"
DEFAULT_GLOBAL_CAMERA = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json"
DEFAULT_TRANSFORM = ROOT / "outputs/head_similarity_fresh_vs_inherited_cuda4/phase_a/metrics.json"
DEFAULT_REFERENCE = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/inputs/canonical_1024.png"
DEFAULT_MASK = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/raw_ovoxel_render/alpha.png"
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_OUTPUT = ROOT / "outputs/head_baseline_patch_merge_cuda4"


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_mesh(path: Path) -> Mesh:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload["mesh"]
    return Mesh(mesh.vertices.detach().cpu().float(), mesh.faces.detach().cpu().int())


def load_transform(path: Path) -> tuple[float, torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sim = payload["similarity"]
    return (
        float(sim["scale"]),
        torch.tensor(sim["rotation"], dtype=torch.float32),
        torch.tensor(sim["translation"], dtype=torch.float32),
    )


def transform_mesh(mesh: Mesh, scale: float, rotation: torch.Tensor, translation: torch.Tensor) -> Mesh:
    vertices = scale * (mesh.vertices.double() @ rotation.double().T) + translation.double()
    return Mesh(vertices.float(), mesh.faces.clone())


def shift_mesh(mesh: Mesh, shift: torch.Tensor) -> Mesh:
    if torch.count_nonzero(shift).item() == 0:
        return mesh
    return Mesh(mesh.vertices + shift[None].to(mesh.vertices), mesh.faces)


@torch.no_grad()
def decode_local_state(
    path: Path,
    model_path: Path,
    resolution: int,
    device: torch.device,
    promote_c64_to_c128: bool,
) -> Mesh:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coords = payload["coords"].int()
    features = payload["features"].float()
    if promote_c64_to_c128:
        if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= 64)).any()):
            raise ValueError("--promote-c64-to-c128 expects a C64 state")
        # Preserve the physical C64 cell centers while presenting the state
        # on the C128 grid expected by a 2048 decoder.  The old direct C64 ->
        # 2048 call treated [0,63] as the lower half of C128, which clipped
        # the independent head to one side of the object.
        xyz128 = ((coords[:, 1:].float() + 0.5) * 2.0).floor().int()
        coords = torch.cat((coords[:, :1], xyz128), dim=1)
        if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
            raise RuntimeError("C64 -> C128 promotion created duplicate coordinates")
    slat = SparseTensor(features.to(device), coords.to(device))
    pipeline = load_decoder_pipeline(model_path.resolve(), device)
    try:
        print(f"[decode] independent head C64 -> {resolution}", flush=True)
        meshes, _ = pipeline.decode_shape_slat(slat, int(resolution))
        if len(meshes) != 1:
            raise RuntimeError(f"decoder returned {len(meshes)} local meshes")
        mesh = meshes[0]
        return Mesh(mesh.vertices.detach().cpu().float(), mesh.faces.detach().cpu().int())
    finally:
        del slat, pipeline
        empty_cuda()


def crop_base_faces(base: Mesh, lo: torch.Tensor, hi: torch.Tensor, margin: float) -> tuple[Mesh, int]:
    """Remove base faces whose centroid lies in the aligned patch box."""
    vertices = base.vertices
    faces = base.faces.long()
    lower = lo - float(margin)
    upper = hi + float(margin)
    kept: list[torch.Tensor] = []
    removed = 0
    for part in faces.split(1_000_000):
        fv = vertices.index_select(0, part.reshape(-1)).reshape(part.shape[0], 3, 3)
        center = fv.mean(dim=1)
        inside = ((center >= lower) & (center <= upper)).all(dim=1)
        kept.append(part[~inside])
        removed += int(inside.sum())
    kept_faces = torch.cat(kept, 0)
    used = torch.unique(kept_faces)
    remap = torch.full((vertices.shape[0],), -1, dtype=torch.int64)
    remap[used] = torch.arange(used.shape[0], dtype=torch.int64)
    cropped = Mesh(vertices.index_select(0, used), remap[kept_faces].int())
    return cropped, removed


def concatenate(base: Mesh, patch: Mesh) -> Mesh:
    offset = int(base.vertices.shape[0])
    faces = torch.cat((base.faces.int(), patch.faces.int() + offset), 0)
    vertices = torch.cat((base.vertices.float(), patch.vertices.float()), 0)
    return Mesh(vertices, faces)


def render_variant(
    mesh: Mesh,
    out: Path,
    camera: dict[str, Any],
    reference: Path,
    mask: Path,
    angles: Sequence[int],
    device: torch.device,
    resolution: int,
    chunk_size: int,
) -> dict[str, Any]:
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    live = mesh.to(device)
    try:
        return render_geometry(
            live,
            camera,
            reference,
            mask,
            out,
            int(resolution),
            tuple(int(v) % 360 for v in angles),
            int(chunk_size),
        )
    finally:
        del live
        empty_cuda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-state", type=Path, default=DEFAULT_LOCAL_STATE)
    parser.add_argument("--local-mesh", type=Path, default=DEFAULT_LOCAL_MESH)
    parser.add_argument("--base-mesh", type=Path, default=DEFAULT_BASE_MESH)
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=Path, default=DEFAULT_GLOBAL_CAMERA)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--local-resolution", type=int, choices=(1024, 2048), default=2048)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200_000)
    parser.add_argument("--angles", default="0,60,180,240")
    parser.add_argument("--margins", default="0.01,0.02,0.04,0.06")
    parser.add_argument(
        "--patch-shift",
        default="0,0,0",
        help="global xyz translation applied after similarity alignment; useful for the input-view depth ablation",
    )
    parser.add_argument("--use-saved-local-mesh", action="store_true")
    parser.add_argument(
        "--promote-c64-to-c128",
        action="store_true",
        help="double C64 cell coordinates before a 2048 decode; required for a native C64 state",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    margins = tuple(float(v.strip()) for v in args.margins.split(",") if v.strip())
    if any(v < 0 for v in margins):
        raise ValueError("margins must be non-negative")
    angles = tuple(int(v.strip()) for v in args.angles.split(",") if v.strip())
    camera_payload = json.loads(args.camera.read_text(encoding="utf-8"))
    camera = camera_payload.get("camera", camera_payload)
    scale, rotation, translation = load_transform(args.transform.resolve())
    patch_shift = torch.tensor(
        [float(v.strip()) for v in args.patch_shift.split(",")], dtype=torch.float32
    )
    if patch_shift.shape != (3,):
        raise ValueError("--patch-shift must contain three comma-separated values")

    print("[load] full block-local SR mesh", flush=True)
    base = load_mesh(args.base_mesh.resolve())
    if args.use_saved_local_mesh:
        print("[load] saved independent head mesh", flush=True)
        local = load_mesh(args.local_mesh.resolve())
    else:
        local = decode_local_state(
            args.local_state.resolve(),
            args.model_path.resolve(),
            args.local_resolution,
            device,
            args.promote_c64_to_c128,
        )
    patch = shift_mesh(transform_mesh(local, scale, rotation, translation), patch_shift)
    lo = patch.vertices.min(0).values
    hi = patch.vertices.max(0).values
    atomic_json(
        output / "manifest.json",
        {
            "format": "pixal3d_head_baseline_patch_merge_v1",
            "base_mesh": str(args.base_mesh.resolve()),
            "local_source": str((args.local_mesh if args.use_saved_local_mesh else args.local_state).resolve()),
            "local_decode_resolution": 1024 if args.use_saved_local_mesh else int(args.local_resolution),
            "promote_c64_to_c128": bool(args.promote_c64_to_c128),
            "alignment_transform": {
                "source": str(args.transform.resolve()),
                "scale": scale,
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            },
            "patch_shift_after_alignment": patch_shift.tolist(),
            "aligned_patch_bbox_global": [lo.tolist(), hi.tolist()],
            "base_vertices": int(base.vertices.shape[0]),
            "base_faces": int(base.faces.shape[0]),
            "patch_vertices": int(patch.vertices.shape[0]),
            "patch_faces": int(patch.faces.shape[0]),
            "cuda_device": int(args.cuda_device),
        },
    )

    variants: list[tuple[str, Mesh, int]] = []
    variants.append(("overlay", concatenate(base, patch), 0))
    for margin in margins:
        print(f"[merge] replacing base faces in aligned patch box margin={margin:.3f}", flush=True)
        trimmed, removed = crop_base_faces(base, lo, hi, margin)
        variants.append((f"replace_m{margin:.3f}", concatenate(trimmed, patch), removed))
        del trimmed

    rows = []
    for name, mesh, removed in variants:
        variant = output / name
        print(
            f"[render] {name} vertices={mesh.vertices.shape[0]:,} faces={mesh.faces.shape[0]:,} "
            f"removed_base_faces={removed:,}",
            flush=True,
        )
        render = render_variant(
            mesh,
            variant,
            camera,
            args.reference.resolve(),
            args.reference_mask.resolve(),
            angles,
            device,
            args.render_resolution,
            args.render_chunk_size,
        )
        atomic_save(variant / "geometry_mesh.pt", {"format": "pixal3d_head_baseline_patch_merge_v1", "mesh": mesh})
        row = {
            "name": name,
            "removed_base_faces": int(removed),
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "render": render,
        }
        atomic_json(variant / "summary.json", row)
        rows.append(row)
        del mesh
        empty_cuda()
    atomic_json(output / "summary.json", {"format": "pixal3d_head_baseline_patch_merge_v1", "status": "complete", "variants": rows})
    del variants, patch, local, base
    gc.collect()
    empty_cuda()
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
