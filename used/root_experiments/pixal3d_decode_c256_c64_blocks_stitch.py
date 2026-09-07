#!/usr/bin/env python3
"""Decode disjoint Global-C256 C64 blocks at local 1024 and stitch PBR meshes."""
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

import torch

import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global_c256_restructured_blocks_singleview as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr


FORMAT = "pixal3d_decode_c256_c64_blocks_stitch_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def place_local_mesh(mesh: MeshWithVertexPbr, start: tuple[int, int, int]) -> MeshWithVertexPbr:
    # Decoder vertices span [-.5,.5].  A C64 block is one quarter of C256.
    centre_q = 2.0 * (torch.tensor(start, dtype=torch.float32) + 32.0) / 256.0 - 1.0
    vertices = centre_q[None] / 2.0 + mesh.vertices.float() / 4.0
    return MeshWithVertexPbr(
        vertices, mesh.faces.int(), mesh.vertex_attrs.float(), layout=dict(mesh.layout))


def sanitize(mesh: MeshWithVertexPbr) -> tuple[MeshWithVertexPbr, int, int]:
    bad = ~torch.isfinite(mesh.vertices).all(1)
    bad_attrs = ~torch.isfinite(mesh.vertex_attrs).all(1)
    if bad_attrs.any():
        attrs = mesh.vertex_attrs.clone()
        attrs[bad_attrs] = 0
    else:
        attrs = mesh.vertex_attrs
    keep = ~bad[mesh.faces.long()].any(1)
    vertices = mesh.vertices.clone() if bad.any() else mesh.vertices
    if bad.any():
        vertices[bad] = 0
    return MeshWithVertexPbr(
        vertices, mesh.faces[keep].int(), attrs, layout=dict(mesh.layout)), int(bad.sum()), int((~keep).sum())


def block_records(coords: torch.Tensor) -> list[dict[str, Any]]:
    xyz = coords[:, 1:4].cpu().int()
    records = []
    block_id = 0
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    for bx in range(4):
        for by in range(4):
            for bz in range(4):
                start = torch.tensor((bx, by, bz), dtype=torch.int32) * 64
                mask = ((xyz >= start) & (xyz < start + 64)).all(1)
                if not mask.any():
                    continue
                rows = torch.where(mask)[0].long()
                coverage.index_add_(0, rows, torch.ones(len(rows), dtype=torch.int16))
                records.append({
                    "block_id": block_id, "block_index": (bx, by, bz),
                    "start": tuple(int(v) for v in start.tolist()), "rows": rows,
                    "local_xyz": xyz.index_select(0, rows) - start,
                })
                block_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError("disjoint C64 block coverage is not exactly one")
    return records


@torch.no_grad()
def decode_blocks(
    pipeline: Any, coords: torch.Tensor, shape: torch.Tensor, texture: torch.Tensor,
    records: list[dict[str, Any]], output: Path, device: torch.device,
) -> None:
    root = output / "blocks"
    shape_raw = cube_flow.denormalize(shape, pipeline.shape_slat_normalization)
    texture_raw = cube_flow.denormalize(texture, pipeline.tex_slat_normalization)
    for order, rec in enumerate(records, 1):
        block_id = int(rec["block_id"])
        block_dir = root / f"block_{block_id:02d}"
        mesh_path = block_dir / "global_placed_per_vertex_pbr_mesh.pt"
        if mesh_path.is_file():
            continue
        rows = rec["rows"]
        local_coords = torch.cat((
            torch.zeros((len(rows), 1), dtype=torch.int32), rec["local_xyz"]), 1)
        decoded = pipeline.decode_latent(
            SparseTensor(shape_raw.index_select(0, rows).to(device), local_coords.to(device)),
            SparseTensor(texture_raw.index_select(0, rows).to(device), local_coords.to(device)),
            1024)
        if len(decoded) != 1:
            raise RuntimeError(f"block {block_id} decode returned {len(decoded)} meshes")
        vertex, _ = expc._native_mesh_to_pbr(decoded[0], device)
        placed = place_local_mesh(vertex.cpu(), rec["start"])
        placed, bad_vertices, removed_faces = sanitize(placed)
        block_dir.mkdir(parents=True, exist_ok=True)
        cube_flow.atomic_save(mesh_path, {
            "format": FORMAT, "mesh": placed, "block_id": block_id,
            "block_index": rec["block_index"], "start_c256": rec["start"],
            "tokens": len(rows), "local_decode_resolution": 1024,
            "nonfinite_vertices_replaced": bad_vertices,
            "incident_faces_removed": removed_faces,
        })
        core.atomic_json(block_dir / "summary.json", {
            "format": FORMAT, "block_id": block_id, "block_index": rec["block_index"],
            "start_c256": rec["start"], "tokens": len(rows),
            "vertices": len(placed.vertices), "faces": len(placed.faces),
            "nonfinite_vertices_replaced": bad_vertices,
            "incident_faces_removed": removed_faces,
            "mesh": str(mesh_path.resolve()),
        })
        print(f"[decode-block] {order}/{len(records)} id={block_id:02d} "
              f"tokens={len(rows):,} vertices={len(placed.vertices):,} "
              f"faces={len(placed.faces):,}", flush=True)
        del decoded, vertex, placed
        empty_cuda()


def stitch(records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    final_path = output / "stitched" / "global_stitched_per_vertex_pbr_mesh.pt"
    if final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        mesh = payload["mesh"]
        return {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "mesh": str(final_path.resolve())}
    vertices, faces, attrs = [], [], []
    offset = 0
    block_stats = []
    for rec in records:
        path = output / "blocks" / f"block_{int(rec['block_id']):02d}" / "global_placed_per_vertex_pbr_mesh.pt"
        mesh = torch.load(path, map_location="cpu", weights_only=False)["mesh"]
        vertices.append(mesh.vertices.float())
        attrs.append(mesh.vertex_attrs.float())
        faces.append(mesh.faces.long() + offset)
        block_stats.append({"block_id": int(rec["block_id"]), "vertices": len(mesh.vertices), "faces": len(mesh.faces)})
        offset += len(mesh.vertices)
    merged = MeshWithVertexPbr(
        torch.cat(vertices), torch.cat(faces).int(), torch.cat(attrs),
        layout=dict(torch.load(
            output / "blocks" / "block_00" / "global_placed_per_vertex_pbr_mesh.pt",
            map_location="cpu", weights_only=False)["mesh"].layout))
    cube_flow.atomic_save(final_path, {
        "format": FORMAT, "mesh": merged, "welding": False,
        "deduplication": False, "source_blocks": len(records)})
    result = {"vertices": len(merged.vertices), "faces": len(merged.faces),
              "mesh": str(final_path.resolve()), "blocks": block_stats}
    core.atomic_json(output / "stitched" / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    root = Path("outputs/global_c256_c64_stride32_owner_flow_cuda5")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--support", type=Path, default=Path(
        "outputs/global_c256_c32_context_owner_dec1_all_blocks_cuda5/global_support/global_c256_support.pt"))
    p.add_argument("--shape", type=Path, default=root / "shape/final_normalized.pt")
    p.add_argument("--texture", type=Path, default=root / "texture/final_normalized.pt")
    p.add_argument("--output", type=Path, default=root / "local_c64_decode_stitched")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coords = torch.load(args.support, map_location="cpu", weights_only=False)["coords"].int()
    shape = torch.load(args.shape, map_location="cpu", weights_only=False)["features"].float()
    texture = torch.load(args.texture, map_location="cpu", weights_only=False)["features"].float()
    if len(coords) != len(shape) or len(coords) != len(texture):
        raise RuntimeError("support/latent row mismatch")
    records = block_records(coords)
    print(f"[input] tokens={len(coords):,} disjoint_blocks={len(records)}", flush=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    decode_blocks(pipeline, coords, shape, texture, records, output, device)
    result = stitch(records, output)
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "tokens": len(coords),
        "blocks": len(records), "stitch": result, "seconds": time.perf_counter() - started})
    print(f"[done] vertices={result['vertices']:,} faces={result['faces']:,} output={output}", flush=True)


if __name__ == "__main__":
    main()
