#!/usr/bin/env python3
"""Measure whether the aligned independent-head mesh is visible over the SR mesh."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image

from pixal3d.representations import Mesh
from run_head_similarity_experiment import render_buffers


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8_anchor16_cuda4/conditional/n_08/geometry_mesh.pt"
DEFAULT_OVERLAY = ROOT / "outputs/head_baseline_patch_merge_cuda4/overlay/geometry_mesh.pt"
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json"
DEFAULT_OUT = ROOT / "outputs/head_baseline_patch_merge_cuda4/visibility_diagnostic"


def load_mesh(path: Path) -> Mesh:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload["mesh"]
    return Mesh(mesh.vertices.detach().cpu().float(), mesh.faces.detach().cpu().int())


def split_overlay(path: Path, base_vertices: int) -> Mesh:
    mesh = load_mesh(path)
    faces = mesh.faces.long()
    patch_faces = faces[(faces >= base_vertices).all(dim=1)] - base_vertices
    if len(patch_faces) == 0:
        raise RuntimeError("overlay has no faces using appended patch vertices")
    used = torch.unique(patch_faces)
    remap = torch.full((mesh.vertices.shape[0] - base_vertices,), -1, dtype=torch.long)
    remap[used] = torch.arange(len(used), dtype=torch.long)
    return Mesh(mesh.vertices[base_vertices:].index_select(0, used), remap[patch_faces].int())


def save_normal(path: Path, normal: np.ndarray, mask: np.ndarray) -> None:
    out = np.uint8(np.clip(np.moveaxis(normal, 0, -1), 0, 1) * 255)
    out[~mask] = 128
    Image.fromarray(out).save(path)


def save_gray(path: Path, normal: np.ndarray, mask: np.ndarray) -> None:
    n = np.moveaxis(normal, 0, -1)
    gray = .18 + .72 * np.clip(n[..., 2], 0, 1)
    gray[~mask] = 1
    Image.fromarray(np.uint8(np.clip(gray, 0, 1) * 255)).convert("RGB").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.uint8(mask) * 255).save(path)


def save_delta(path: Path, delta: np.ndarray, valid: np.ndarray) -> None:
    finite = valid & np.isfinite(delta)
    canvas = np.full((*delta.shape, 3), 128, np.uint8)
    if finite.any():
        lo, hi = np.quantile(delta[finite], [.02, .98])
        t = np.clip((delta - lo) / (hi - lo + 1e-8), 0, 1)
        # blue = patch closer to camera, red = patch farther away
        canvas[..., 0] = np.uint8(255 * t)
        canvas[..., 1] = np.uint8(255 * (1 - np.abs(2 * t - 1)))
        canvas[..., 2] = np.uint8(255 * (1 - t))
        canvas[~finite] = 128
    Image.fromarray(canvas).save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--resolution", type=int, default=1024)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    base = load_mesh(args.base.resolve())
    overlay = load_mesh(args.overlay.resolve())
    patch = split_overlay(args.overlay.resolve(), int(base.vertices.shape[0]))
    camera_payload = json.loads(args.camera.resolve().read_text())
    camera = camera_payload.get("camera", camera_payload)
    print(f"[render] base V={len(base.vertices):,} patch V={len(patch.vertices):,}", flush=True)
    base_r = render_buffers(base, camera, device, args.resolution)
    torch.cuda.empty_cache()
    patch_r = render_buffers(patch, camera, device, args.resolution)
    torch.cuda.empty_cache()
    # The overlay renderer is useful to establish that this diagnostic uses
    # exactly the same rasterizer/depth convention as the saved mesh result.
    overlay_r = render_buffers(overlay, camera, device, args.resolution)
    torch.cuda.empty_cache()

    bm = base_r["mask"].numpy() > .5
    pm = patch_r["mask"].numpy() > .5
    om = overlay_r["mask"].numpy() > .5
    bd = base_r["depth"].numpy()
    pd = patch_r["depth"].numpy()
    od = overlay_r["depth"].numpy()
    valid = pm & bm & np.isfinite(pd) & np.isfinite(bd)
    delta = pd - bd
    # MeshRenderer depth is compared directly; lower means closer in this
    # camera setup. Keep both signs in the report in case the renderer changes.
    patch_front = pm & (~bm | (pd < bd))
    patch_front_eps = pm & (~bm | (pd <= bd + .002))
    changed = om != bm
    report = {
        "base_vertices": int(len(base.vertices)),
        "patch_vertices": int(len(patch.vertices)),
        "base_foreground_pixels": int(bm.sum()),
        "patch_foreground_pixels": int(pm.sum()),
        "overlay_foreground_pixels": int(om.sum()),
        "intersection_pixels": int(valid.sum()),
        "patch_front_of_base_pixels": int(patch_front.sum()),
        "patch_front_or_within_0.002_pixels": int(patch_front_eps.sum()),
        "overlay_changed_pixels_vs_base": int(changed.sum()),
        "depth_delta_patch_minus_base_intersection": {
            "mean": float(delta[valid].mean()) if valid.any() else None,
            "median": float(np.median(delta[valid])) if valid.any() else None,
            "p10": float(np.quantile(delta[valid], .1)) if valid.any() else None,
            "p90": float(np.quantile(delta[valid], .9)) if valid.any() else None,
        },
        "renderer_note": "lower depth is treated as closer for patch_front_of_base",
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, r in (("base", base_r), ("patch", patch_r), ("overlay", overlay_r)):
        mask = r["mask"].numpy() > .5
        save_normal(args.out / f"{name}_normal.png", r["normal"].numpy(), mask)
        save_gray(args.out / f"{name}_gray.png", r["normal"].numpy(), mask)
        save_mask(args.out / f"{name}_mask.png", mask)
    save_mask(args.out / "patch_front_mask.png", patch_front)
    save_mask(args.out / "patch_front_eps_mask.png", patch_front_eps)
    save_mask(args.out / "overlay_changed_mask.png", changed)
    save_delta(args.out / "patch_minus_base_depth.png", delta, valid)
    # A pure input-view teacher composite shows the best possible result if
    # occlusion/stitching is the only failure mode.
    pnormal = np.moveaxis(patch_r["normal"].numpy(), 0, -1)
    bnormal = np.moveaxis(base_r["normal"].numpy(), 0, -1)
    for label, take in (("all_patch", pm), ("front_patch", patch_front_eps)):
        n = np.where(take[..., None], pnormal, bnormal)
        m = bm | take
        save_normal(args.out / f"teacher_{label}_normal.png", np.moveaxis(n, -1, 0), m)
        save_gray(args.out / f"teacher_{label}_gray.png", np.moveaxis(n, -1, 0), m)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
