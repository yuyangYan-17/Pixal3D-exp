#!/usr/bin/env python3
"""Decode a block-local C64 state independently and merge it in object space.

The local Shape512/Shape1024 experiment can produce a valid C64 state in each
of eight 64^3 object-space blocks, but decoding the concatenated state with the
global C128 decoder creates a hard discontinuity at the block planes.  This
driver tests the corresponding local decoder route: each block is decoded as a
native C64 shape, its normalized mesh is scaled into the block's half-width,
and the CPU meshes are concatenated only after every decoder invocation has
finished.  Keeping completed meshes on CPU is deliberate; otherwise the
decoder's output tensors accumulate on GPU and make later blocks OOM.

This is geometry-only.  It consumes a saved denormalized C64 state and does
not rerun either Flow stage or any texture model.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image

import pixal3d.models as models
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.representations import Mesh
from pixal3d.modules.sparse import SparseTensor

from pixal3d_c128_block_local_cascade_shape_renoise_2048 import (
    DEFAULT_BASELINE,
    DEFAULT_MODEL_PATH,
    empty_cuda,
    normal_metrics,
)


ROOT = Path(__file__).resolve().parent
GLOBAL_GRID = 128
BLOCK_GRID = 64
BLOCK_STARTS = (0, 64)
DEFAULT_STATE = (
    ROOT
    / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8"
    / "conditional/n_08/shape_c64_final_denormalized.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/c128_block_local_decode_merge_fresh_n8_cuda4"


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    path_payload = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(path_payload, encoding="utf-8")
    os.replace(temporary, path)


def sort_xyz(coords: torch.Tensor) -> torch.Tensor:
    if not coords.numel():
        return coords.reshape(0, 3).int()
    order = np.lexsort(
        (
            coords[:, 2].cpu().numpy(),
            coords[:, 1].cpu().numpy(),
            coords[:, 0].cpu().numpy(),
        )
    )
    return coords.index_select(0, torch.from_numpy(order).long()).int()


def load_decoder_pipeline(model_path: Path, device: torch.device) -> Any:
    config = json.loads((model_path / "pipeline.json").read_text(encoding="utf-8"))["args"]
    decoder_name = "shape_slat_decoder"
    print(f"[model] loading {decoder_name}", flush=True)
    decoder = models.from_pretrained(
        str(model_path / config["models"][decoder_name])
    ).eval()
    pipeline = Pixal3DImageTo3DPipeline(
        models={decoder_name: decoder},
        low_vram=True,
    )
    pipeline._device = device
    return pipeline


def load_state(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coords = payload["coords"].int()
    features = payload["features"].float()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"expected [N,4] C128 coords, got {tuple(coords.shape)}")
    if features.ndim != 2 or features.shape[0] != coords.shape[0]:
        raise ValueError("state features are not row-aligned with coords")
    xyz = coords[:, 1:]
    if bool(((xyz < 0) | (xyz >= GLOBAL_GRID)).any()):
        raise ValueError("state contains coordinates outside C128")
    if torch.unique(xyz, dim=0).shape[0] != xyz.shape[0]:
        raise ValueError("state contains duplicate global coordinates")
    return coords, features


def block_records(coords: torch.Tensor) -> list[dict[str, Any]]:
    xyz = coords[:, 1:]
    records: list[dict[str, Any]] = []
    cube_id = 0
    for sx in BLOCK_STARTS:
        for sy in BLOCK_STARTS:
            for sz in BLOCK_STARTS:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                inside = ((xyz >= start) & (xyz < start + BLOCK_GRID)).all(1)
                rows = torch.where(inside)[0].long()
                if rows.numel():
                    local = sort_xyz(xyz.index_select(0, rows) - start)
                    local_keys = (
                        local[:, 0].long() * BLOCK_GRID + local[:, 1].long()
                    ) * BLOCK_GRID + local[:, 2].long()
                    local_order = torch.argsort(local_keys)
                    rows = rows.index_select(0, local_order)
                    local = local.index_select(0, local_order)
                else:
                    local = torch.empty((0, 3), dtype=torch.int32)
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": (sx, sy, sz),
                        "rows": rows,
                        "local_xyz": local,
                    }
                )
                cube_id += 1
    coverage = torch.zeros(coords.shape[0], dtype=torch.int16)
    for record in records:
        coverage.index_add_(
            0, record["rows"], torch.ones(record["rows"].shape[0], dtype=torch.int16)
        )
    if not torch.all(coverage == 1):
        raise RuntimeError("C128 state is not partitioned exactly once by eight blocks")
    return records


@torch.no_grad()
def decode_blocks(
    pipeline: Any,
    coords: torch.Tensor,
    raw_features: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    resolution: int,
    device: torch.device,
    output: Path,
) -> tuple[Mesh, list[dict[str, Any]]]:
    """Decode blocks one by one, moving all completed outputs to CPU."""
    vertex_parts: list[torch.Tensor] = []
    face_parts: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    vertex_offset = 0
    factor = float(BLOCK_GRID) / float(GLOBAL_GRID)

    for index, record in enumerate(records):
        rows = record["rows"].long()
        if not rows.numel():
            metadata.append(
                {"cube_id": int(record["cube_id"]), "tokens": 0, "vertices": 0, "faces": 0}
            )
            continue
        local_xyz = record["local_xyz"].int()
        local_coords = torch.cat(
            (torch.zeros((local_xyz.shape[0], 1), dtype=torch.int32), local_xyz), dim=1
        )
        local_features = raw_features.index_select(0, rows).contiguous()
        slat = SparseTensor(local_features.to(device), local_coords.to(device))
        print(
            f"[decode {resolution}] cube={int(record['cube_id']):02d} "
            f"tokens={rows.numel():,}",
            flush=True,
        )
        started = time.perf_counter()
        meshes = None
        substructures = None
        decoded = None
        try:
            meshes, substructures = pipeline.decode_shape_slat(slat, int(resolution))
            if len(meshes) != 1:
                raise RuntimeError(f"decoder returned {len(meshes)} meshes for one block")
            decoded = meshes[0]
            vertices = decoded.vertices.detach().cpu().float()
            faces = decoded.faces.detach().cpu().long()
            if vertices.numel() == 0 or faces.numel() == 0:
                raise RuntimeError("decoder returned an empty block mesh")
            start = torch.tensor(record["start"], dtype=vertices.dtype)
            # Decoder vertices use [-0.5, 0.5] for a complete C64 object.
            # A global C128 block occupies one half of that interval.
            center = (start + float(BLOCK_GRID) / 2.0) / float(GLOBAL_GRID) - 0.5
            global_vertices = vertices * factor + center[None]
            global_faces = faces.to(torch.int32) + int(vertex_offset)
            vertex_parts.append(global_vertices.contiguous())
            face_parts.append(global_faces.contiguous())
            block_meta = {
                "cube_id": int(record["cube_id"]),
                "start": list(record["start"]),
                "tokens": int(rows.numel()),
                "vertices": int(vertices.shape[0]),
                "faces": int(faces.shape[0]),
                "decode_seconds": float(time.perf_counter() - started),
                "mapping": {
                    "factor": factor,
                    "center": [float(v) for v in center],
                    "local_decoder_resolution": int(resolution),
                },
            }
            metadata.append(block_meta)
            vertex_offset += int(vertices.shape[0])
            print(
                f"[decode {resolution}] cube={int(record['cube_id']):02d} done "
                f"vertices={vertices.shape[0]:,} faces={faces.shape[0]:,} "
                f"seconds={block_meta['decode_seconds']:.1f}",
                flush=True,
            )
        finally:
            # ``decoded`` and ``meshes`` can retain CUDA tensors even after the
            # CPU copies above.  Delete them before the next block.
            del decoded, meshes, substructures, slat
            empty_cuda()

    if not vertex_parts or not face_parts:
        raise RuntimeError("all local block decodes were empty")
    merged = Mesh(torch.cat(vertex_parts, 0), torch.cat(face_parts, 0))
    atomic_json(output / f"local_decode_{resolution}_blocks.json", {"blocks": metadata})
    del vertex_parts, face_parts
    empty_cuda()
    return merged, metadata


def render_mesh(
    mesh: Mesh,
    camera: Mapping[str, Any],
    reference_path: Path,
    mask_path: Path,
    output: Path,
    resolution: int,
    angles: Sequence[int],
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    live_mesh = mesh.to(device)
    try:
        result = render_geometry(
            live_mesh,
            camera,
            reference_path,
            mask_path,
            output,
            int(resolution),
            tuple(int(v) % 360 for v in angles),
            int(chunk_size),
        )
    finally:
        del live_mesh
        empty_cuda()
    front = output / f"multiview_{resolution}/view_000_camera_normal.png"
    raw_reference = DEFAULT_BASELINE / "raw_ovoxel_render/normal.png"
    raw_mask = DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png"
    return {
        **result,
        "baseline_normal_metrics": normal_metrics(front, raw_reference, raw_mask),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", type=Path, default=DEFAULT_BASELINE / "global_camera.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_BASELINE / "canonical_1024.png")
    parser.add_argument("--reference-mask", type=Path, default=DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--resolution", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--angles", default="0,60,180,240")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200_000)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    output = args.output_dir.resolve() / f"resolution_{args.resolution:04d}"
    output.mkdir(parents=True, exist_ok=True)
    coords, raw_features = load_state(args.state.resolve())
    camera_payload = json.loads(args.camera.read_text(encoding="utf-8"))
    camera = camera_payload.get("camera", camera_payload)
    records = block_records(coords)
    atomic_json(
        output / "manifest.json",
        {
            "format": "pixal3d_c128_local_decode_merge_v1",
            "state": str(args.state.resolve()),
            "state_tokens": int(coords.shape[0]),
            "decode_resolution": int(args.resolution),
            "cuda_device": int(args.cuda_device),
            "mapping": "local decoder output * 0.5 + ((start+32)/128 - 0.5); CPU concatenate after each block",
            "blocks": [
                {
                    "cube_id": int(record["cube_id"]),
                    "start": list(record["start"]),
                    "tokens": int(record["rows"].numel()),
                }
                for record in records
            ],
        },
    )
    pipeline = load_decoder_pipeline(args.model_path.resolve(), device)
    mesh, metadata = decode_blocks(
        pipeline,
        coords,
        raw_features,
        records,
        int(args.resolution),
        device,
        output,
    )
    mesh_path = output / "geometry_mesh.pt"
    atomic_save(mesh_path, {"format": "pixal3d_c128_local_decode_merge_v1", "mesh": mesh})
    try:
        import trimesh

        trimesh.Trimesh(
            vertices=mesh.vertices.numpy(),
            faces=mesh.faces.numpy(),
            process=False,
        ).export(output / "geometry_mesh.glb")
    except Exception as exc:
        atomic_json(output / "glb_export_error.json", {"error": repr(exc)})
    render = render_mesh(
        mesh,
        camera,
        args.reference.resolve(),
        args.reference_mask.resolve(),
        output,
        int(args.render_resolution),
        tuple(int(v.strip()) for v in args.angles.split(",") if v.strip()),
        int(args.render_chunk_size),
        device,
    )
    summary = {
        "format": "pixal3d_c128_local_decode_merge_v1",
        "status": "complete",
        "state": str(args.state.resolve()),
        "decode_resolution": int(args.resolution),
        "tokens": int(coords.shape[0]),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "mesh": str(mesh_path.resolve()),
        "blocks": metadata,
        "render": render,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
