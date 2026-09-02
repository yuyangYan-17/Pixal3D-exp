#!/usr/bin/env python3
"""Decode and render the C256 endpoint used by the first guided Euler step."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch

import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as native_sr
import pixal3d_render_global4096_multiview as multiview
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel


DEFAULT_RUN = Path("outputs/guided_endpoint_sr_step0_cuda5")


def _atomic_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--angles", default="0,120,240")
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument(
        "--ordinary-shape-slat",
        type=Path,
        default=None,
        help="Decode this {coords,feats} shape SLat without forced subdivisions.",
    )
    parser.add_argument(
        "--ordinary-texture-slat",
        type=Path,
        default=None,
        help="Use this ordinary {coords,feats} texture SLat for the control.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    encoded_path = run_dir / "guided_encoder" / "encoded_slats.pt"
    camera_path = run_dir / "global_camera.json"
    if not encoded_path.is_file():
        raise FileNotFoundError(encoded_path)
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)

    payload = torch.load(encoded_path, map_location="cpu", weights_only=False)
    if args.ordinary_shape_slat is None:
        shape_payload = payload["shape"]
    else:
        shape_payload = torch.load(
            args.ordinary_shape_slat.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
    shape = SparseTensor(
        shape_payload["feats"].to(device),
        shape_payload["coords"].to(device),
    )
    texture_payload = (
        payload["texture"]
        if args.ordinary_texture_slat is None
        else torch.load(
            args.ordinary_texture_slat.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
    )
    texture = SparseTensor(
        texture_payload["feats"].to(device),
        texture_payload["coords"].to(device),
    )
    if not torch.equal(shape.coords, texture.coords):
        raise RuntimeError("guided shape/texture endpoint supports differ")
    print(f"[guide endpoint] C256 tokens={shape.coords.shape[0]:,}", flush=True)

    pipeline = init_pipeline(str(args.model_path), device="cuda", low_vram=True)
    if args.ordinary_shape_slat is None:
        print("[guide endpoint] forced-subs C256 decode -> 4096", flush=True)
        saved_subs = torch.load(
            run_dir / "guided_encoder" / "guide_subs.pt",
            map_location="cpu",
            weights_only=False,
        )
        guide_subs = [
            SparseTensor(item["feats"].to(device), item["coords"].to(device))
            for item in saved_subs
        ]
        meshes, used_subs = pipeline.decode_shape_slat(
            shape, 4096, guide_subs=guide_subs
        )
        output_root = run_dir / "guided_endpoint_forced_subs_visualization"
    else:
        print("[ordinary control] unforced C256 decode -> 4096", flush=True)
        meshes, used_subs = pipeline.decode_shape_slat(shape, 4096)
        output_root = run_dir / (
            "normal_shape_texture_encoder_ordinary_decoder_visualization"
            if args.ordinary_texture_slat is not None
            else "normal_encoder_ordinary_decoder_visualization"
        )
    tex_voxels = pipeline.decode_tex_slat(texture, used_subs)
    decoded = MeshWithVoxel(
        meshes[0].vertices,
        meshes[0].faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 4096,
        coords=tex_voxels[0].coords[:, 1:],
        attrs=tex_voxels[0].feats,
        voxel_shape=torch.Size(
            [*tex_voxels[0].shape, *tex_voxels[0].spatial_shape]
        ),
        layout=pipeline.pbr_attr_layout,
    )
    if args.ordinary_shape_slat is None:
        _atomic_save(
            output_root / "material_mesh.pt",
            {"format": "pixal3d_guided_first_endpoint_v1", "mesh": decoded.cpu()},
        )
    vertex_mesh, face_mesh = native_sr._native_mesh_to_pbr(decoded, device)
    vertex_path = output_root / "per_vertex_pbr_mesh.pt"
    output_format = (
        "pixal3d_guided_first_endpoint_v1"
        if args.ordinary_shape_slat is None
        else "pixal3d_normal_encoder_ordinary_decoder_control_v1"
    )
    if args.ordinary_shape_slat is not None:
        finite_vertices = torch.isfinite(vertex_mesh.vertices).all(dim=1)
        if not bool(finite_vertices.all().item()):
            finite_faces = finite_vertices[vertex_mesh.faces.long()].all(dim=1)
            invalid_vertices = int((~finite_vertices).sum().item())
            removed_faces = int((~finite_faces).sum().item())
            vertex_mesh.vertices[~finite_vertices] = 0
            vertex_mesh.faces = vertex_mesh.faces[finite_faces]
            print(
                f"[finite mesh] replaced vertices={invalid_vertices:,}; "
                f"removed faces={removed_faces:,}",
                flush=True,
            )
    _atomic_save(vertex_path, {"format": output_format, "mesh": vertex_mesh})
    if args.ordinary_shape_slat is None:
        _atomic_save(
            output_root / "per_face_pbr_mesh.pt",
            {"format": output_format, "mesh": face_mesh},
        )
    del decoded, shape, texture, pipeline, vertex_mesh, face_mesh
    del meshes, used_subs, tex_voxels
    torch.cuda.empty_cache()

    multiview.render(
        SimpleNamespace(
            mesh=vertex_path,
            camera=camera_path,
            output_dir=output_root / "multiview",
            angles=args.angles,
            resolution=args.resolution,
            face_chunk_size=args.face_chunk_size,
            device=str(device),
            force=args.force_render,
        )
    )
    print(f"[done] {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
