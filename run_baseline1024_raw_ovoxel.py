#!/usr/bin/env python3
"""Run one native Pixal3D 1024 baseline and render its raw O-Voxel mesh.

This deliberately does not build per-face or per-vertex PBR approximations.
The decoded ``MeshWithVoxel`` is passed directly to Pixal3D's nvdiffrast
renderer, which queries the decoded sparse O-Voxel PBR field at surface
positions while shading.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from pixal3d.representations import MeshWithVoxel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--ssaa", type=int, default=2)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/nvme04/yyyan/download/model/Pixal3D"),
    )
    parser.add_argument(
        "--moge-model",
        type=Path,
        default=Path("/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.cuda_device < 0 or args.cuda_device >= torch.cuda.device_count():
        raise ValueError(
            f"invalid CUDA device {args.cuda_device}; device_count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the native preprocessing, MoGe camera estimation, sampler defaults,
    # and exact 1024_cascade invocation without entering its comparison branch.
    from pixal3d_baseline1024_pbr_mesh_compare import (
        _atomic_json,
        _atomic_torch_save,
        _layout_to_json,
        _native_camera_and_mesh,
        _raw_checkpoint_payload,
        _stats,
        _validate_layout,
    )
    from render_pixal3d_raw_ovoxel import (
        load_envmap,
        render_static_ovoxel,
        save_render_outputs,
    )

    print(f"[GPU] cuda:{args.cuda_device} {torch.cuda.get_device_name(args.cuda_device)}")
    print(f"[Output] {output_dir}")
    raw_mesh, canonical_image, camera = _native_camera_and_mesh(args, output_dir)
    if not isinstance(raw_mesh, MeshWithVoxel):
        raise TypeError(f"decoder returned {type(raw_mesh)!r}, expected MeshWithVoxel")
    if raw_mesh.device != device:
        raw_mesh = raw_mesh.to(device)
    _validate_layout(raw_mesh.layout)

    canonical_path = output_dir / "canonical_1024.png"
    checkpoint_path = output_dir / "raw_ovoxel_mesh.pt"
    camera_path = output_dir / "global_camera.json"
    canonical_image.save(canonical_path)
    _atomic_torch_save(checkpoint_path, _raw_checkpoint_payload(raw_mesh))
    _atomic_json(
        camera_path,
        {
            "camera_angle_x": float(camera["camera_angle_x"]),
            "distance": float(camera["distance"]),
            "mesh_scale": float(camera.get("mesh_scale", 1.0)),
            "canonical_1024_intrinsics": {
                "fx": 512.0 / math.tan(float(camera["camera_angle_x"]) / 2.0),
                "fy": 512.0 / math.tan(float(camera["camera_angle_x"]) / 2.0),
                "cx": 512.0,
                "cy": 512.0,
            },
        },
    )

    envmap = load_envmap(args.envmap, device=device)
    torch.cuda.manual_seed_all(int(args.seed) + 100_000)
    renders = render_static_ovoxel(
        raw_mesh,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        resolution=int(args.render_resolution),
        envmap=envmap,
        ssaa=int(args.ssaa),
        peel_layers=int(args.peel_layers),
        face_chunk_size=int(args.face_chunk_size),
        use_envmap_bg=False,
    )
    render_paths = save_render_outputs(renders, output_dir / "raw_ovoxel_render")

    summary = {
        "pipeline_type": "1024_cascade",
        "decoder_resolution": 1024,
        "seed": int(args.seed),
        "cuda_device": int(args.cuda_device),
        "gpu": torch.cuda.get_device_name(args.cuda_device),
        "representation": "MeshWithVoxel (raw decoded O-Voxel)",
        "renderer": "pixal3d.utils.render_utils.render_frames -> PbrMeshRenderer -> nvdiffrast",
        "material_sampling": "raw sparse O-Voxel surface-position lookup via grid_sample_3d trilinear",
        "camera_angle_x": float(camera["camera_angle_x"]),
        "distance": float(camera["distance"]),
        "render_resolution": int(args.render_resolution),
        "ssaa": int(args.ssaa),
        "peel_layers": int(args.peel_layers),
        "vertices": int(raw_mesh.vertices.shape[0]),
        "faces": int(raw_mesh.faces.shape[0]),
        "active_ovoxels": int(raw_mesh.coords.shape[0]),
        "pbr_layout": _layout_to_json(raw_mesh.layout),
        "pbr_ranges": _stats(raw_mesh.attrs, raw_mesh.layout),
        "artifacts": {
            "canonical_1024": str(canonical_path),
            "camera": str(camera_path),
            "raw_ovoxel_mesh": str(checkpoint_path),
            "renders": render_paths,
        },
        "excluded_representations": ["per_face_pbr", "per_vertex_pbr"],
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, summary)
    print(f"[Done] render={render_paths['render']}")
    print(f"[Done] raw_mesh={checkpoint_path}")
    print(f"[Done] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
