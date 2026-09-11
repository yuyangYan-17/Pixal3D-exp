#!/usr/bin/env python3
"""Replace only the input-view visible base faces under an aligned head patch."""
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

from pixal3d.representations import Mesh
from run_head_similarity_experiment import render_buffers
from pixal3d_c128_block_local_cascade_shape_renoise_2048 import empty_cuda


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/conditional/n_08/geometry_mesh.pt"
DEFAULT_OVERLAY = ROOT / "outputs/head_baseline_patch_merge_shiftz0025_cuda4/overlay/geometry_mesh.pt"
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json"
DEFAULT_REFERENCE = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/inputs/canonical_1024.png"
DEFAULT_MASK = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/raw_ovoxel_render/alpha.png"
DEFAULT_OUT = ROOT / "outputs/head_baseline_projected_cut_cuda4"


def load_mesh(path: Path) -> Mesh:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload["mesh"]
    return Mesh(mesh.vertices.detach().cpu().float(), mesh.faces.detach().cpu().int())


def split_overlay(path: Path, base_vertices: int) -> Mesh:
    mesh = load_mesh(path)
    faces = mesh.faces.long()
    patch_faces = faces[(faces >= base_vertices).all(dim=1)] - base_vertices
    used = torch.unique(patch_faces)
    remap = torch.full((mesh.vertices.shape[0] - base_vertices,), -1, dtype=torch.long)
    remap[used] = torch.arange(len(used), dtype=torch.long)
    return Mesh(mesh.vertices[base_vertices:].index_select(0, used), remap[patch_faces].int())


def concatenate(base: Mesh, patch: Mesh) -> Mesh:
    offset = int(base.vertices.shape[0])
    return Mesh(
        torch.cat((base.vertices, patch.vertices), 0),
        torch.cat((base.faces.int(), patch.faces.int() + offset), 0),
    )


def project_centers(vertices: torch.Tensor, faces: torch.Tensor, camera: dict[str, Any], resolution: int):
    """Project triangle centers for the canonical camera (look-at origin, +z camera)."""
    f = float(resolution) / (2.0 * np.tan(float(camera["camera_angle_x"]) / 2.0))
    d = float(camera["distance"])
    centers = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3).mean(1)
    z = d - centers[:, 2]
    u = f * centers[:, 0] / z + resolution / 2.0
    v = resolution / 2.0 - f * centers[:, 1] / z
    return torch.stack((u, v), 1), z


def cut_visible_faces(
    base: Mesh,
    patch_mask: np.ndarray,
    base_depth: np.ndarray,
    camera: dict[str, Any],
    resolution: int,
    tolerance: float,
) -> tuple[Mesh, int, int]:
    """Cut only faces whose projected centroid is visible in the base depth map."""
    faces = base.faces.long()
    kept_parts: list[torch.Tensor] = []
    removed = 0
    candidates = 0
    h, w = patch_mask.shape
    for part in faces.split(1_000_000):
        uv, depth = project_centers(base.vertices, part, camera, resolution)
        ix = torch.round(uv[:, 0]).long()
        iy = torch.round(uv[:, 1]).long()
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        local_patch = np.zeros(len(part), dtype=bool)
        valid = inside.numpy()
        if valid.any():
            local_patch[valid] = patch_mask[iy.numpy()[valid], ix.numpy()[valid]]
        local_patch_t = torch.from_numpy(local_patch)
        candidates += int(local_patch_t.sum())
        # A centroid belongs to the visible base layer when it agrees with the
        # renderer depth at its pixel. This avoids deleting the back side of
        # the model that happens to project into the same patch silhouette.
        base_d = np.full(len(part), np.inf, np.float32)
        if valid.any():
            base_d[valid] = base_depth[iy.numpy()[valid], ix.numpy()[valid]]
        visible = local_patch & np.isfinite(base_d) & (np.abs(depth.numpy() - base_d) <= tolerance)
        kept_parts.append(part[~torch.from_numpy(visible)])
        removed += int(visible.sum())
    kept = torch.cat(kept_parts, 0).int()
    used = torch.unique(kept.long())
    remap = torch.full((len(base.vertices),), -1, dtype=torch.long)
    remap[used] = torch.arange(len(used), dtype=torch.long)
    return Mesh(base.vertices.index_select(0, used), remap[kept.long()].int()), removed, candidates


def render_variant(mesh: Mesh, out: Path, camera: dict[str, Any], reference: Path, mask: Path, angles: Sequence[int], device: torch.device):
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    live = mesh.to(device)
    try:
        return render_geometry(live, camera, reference, mask, out, 1024, tuple(angles), 200_000)
    finally:
        del live
        empty_cuda()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-mask", type=Path, default=DEFAULT_MASK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--tolerances", default="0.002,0.005,0.01")
    p.add_argument("--angles", default="0,60,180,240")
    args = p.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    camera_payload = json.loads(args.camera.resolve().read_text())
    camera = camera_payload.get("camera", camera_payload)
    base = load_mesh(args.base.resolve())
    patch = split_overlay(args.overlay.resolve(), len(base.vertices))
    print(f"[load] base V={len(base.vertices):,} patch V={len(patch.vertices):,}", flush=True)
    print("[render] base and patch depth buffers", flush=True)
    base_r = render_buffers(base, camera, device, 1024)
    empty_cuda()
    patch_r = render_buffers(patch, camera, device, 1024)
    empty_cuda()
    patch_mask = patch_r["mask"].numpy() > .5
    base_depth = base_r["depth"].numpy()
    atomic = {
        "format": "pixal3d_head_projected_cut_merge_v1",
        "base": str(args.base.resolve()),
        "patch_overlay": str(args.overlay.resolve()),
        "patch_pixels": int(patch_mask.sum()),
    }
    (out / "manifest.json").write_text(json.dumps(atomic, indent=2) + "\n")
    tolerances = [float(x.strip()) for x in args.tolerances.split(",") if x.strip()]
    angles = tuple(int(x.strip()) % 360 for x in args.angles.split(",") if x.strip())
    rows = []
    for tolerance in tolerances:
        print(f"[cut] visible input-view faces tolerance={tolerance:.4f}", flush=True)
        trimmed, removed, candidates = cut_visible_faces(base, patch_mask, base_depth, camera, 1024, tolerance)
        mesh = concatenate(trimmed, patch)
        name = f"cut_t{tolerance:.3f}"
        variant = out / name
        print(f"[render] {name} removed={removed:,}/{candidates:,} V={len(mesh.vertices):,} F={len(mesh.faces):,}", flush=True)
        render = render_variant(mesh, variant, camera, args.reference.resolve(), args.reference_mask.resolve(), angles, device)
        torch.save({"format": "pixal3d_head_projected_cut_merge_v1", "mesh": mesh}, variant / "geometry_mesh.pt")
        row = {"name": name, "tolerance": tolerance, "removed_faces": removed, "candidate_faces": candidates, "vertices": len(mesh.vertices), "faces": len(mesh.faces), "render": render}
        (variant / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
        rows.append(row)
        del trimmed, mesh
        gc.collect()
        empty_cuda()
    atomic["status"] = "complete"
    atomic["variants"] = rows
    (out / "summary.json").write_text(json.dumps(atomic, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
