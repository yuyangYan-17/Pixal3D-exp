#!/usr/bin/env python3
"""Geometry experiment with the requested C64 -> C128 two-stage route.

The support/resolution bookkeeping is:

    complete baseline C64 support
      -> context=32/stride=32 blocks in global C64
      -> local C32 Shape512 Flow (local indices 0..31)
      -> local C32 decoder upsample to local C64 support
      -> map each block into global C128 (start * 2 + local C64)
      -> context=64/stride=64 blocks in global C128
      -> local C64 Shape1024 Flow (local indices 0..63)
      -> assemble one global C128 SparseTensor
      -> one global Shape decoder call at 2048

Both Flow stages use fresh noise after the support changes.  Conditions are
geometry-only and are extracted from a native 1024x1024 crop directly cut
from canonical_4096.png.  The important distinction from the earlier driver
is that the first context32 partition is made on the baseline C64 support;
the baseline is not first promoted to C128 and then split into 32-cubes.
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
from PIL import Image, ImageDraw

import pixal3d_c128_context32_hr_crop_local_geometry as previous
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import Mesh


ROOT = Path(__file__).resolve().parent
FORMAT = "pixal3d_c64_context32_global_c128_geometry_v1"
BASELINE_GRID = 64
GLOBAL_GRID = 128
BLOCK32 = 32
BLOCK64 = 64
C512_GRID = 512
CANONICAL_SIZE = 4096
CROP_SIZE = 1024
STEPS = 12
DECODE_RESOLUTION = 2048

DEFAULT_MODEL = previous.DEFAULT_MODEL
DEFAULT_BASELINE = previous.DEFAULT_BASELINE
DEFAULT_CANONICAL = previous.DEFAULT_CANONICAL
DEFAULT_CAMERA = previous.DEFAULT_CAMERA
DEFAULT_OUTPUT = ROOT / "outputs/c64_context32_global_c128_geometry_cuda4"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    previous.atomic_json(path, payload)


def atomic_save(path: Path, payload: Any) -> None:
    previous.atomic_save(path, payload)


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def jsonable(value: Any) -> Any:
    return previous.jsonable(value)


def validate_coords(coords: torch.Tensor, grid: int, name: str) -> torch.Tensor:
    return previous.validate_coords(coords, grid, name)


def sort_coords(coords: torch.Tensor, grid: int) -> torch.Tensor:
    return previous.sort_coords(coords, grid)


def denormalize(features: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    return previous.denormalize(features, spec)


def parse_camera(path: Path) -> dict[str, Any]:
    return previous.parse_camera(path)


def parse_angles(value: str) -> tuple[int, ...]:
    return previous.parse_angles(value)


def seed_noise(shape: Sequence[int], seed: int) -> torch.Tensor:
    return previous.seed_noise(shape, seed)


def make_c64_context32_blocks(coords_c64: torch.Tensor) -> list[dict[str, Any]]:
    """Partition baseline C64 into disjoint local C32 blocks."""
    coords_c64 = validate_coords(coords_c64, BASELINE_GRID, "baseline C64")
    xyz = coords_c64[:, 1:]
    starts = tuple(range(0, BASELINE_GRID, BLOCK32))
    coverage = torch.zeros(len(coords_c64), dtype=torch.int16)
    records: list[dict[str, Any]] = []
    offset = 0
    cube_id = 0
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + BLOCK32)).all(dim=1)
                )[0].long()
                if not len(rows):
                    cube_id += 1
                    continue
                local_xyz = xyz.index_select(0, rows) - start
                local_coords = torch.cat(
                    (
                        torch.zeros((len(rows), 1), dtype=torch.int32),
                        local_xyz,
                    ),
                    dim=1,
                )
                local_rows = torch.arange(
                    offset, offset + len(rows), dtype=torch.long
                )
                records.append(
                    {
                        "cube_id": int(cube_id),
                        "start": (sx, sy, sz),
                        "global_row_ids": local_rows,
                        "source_c64_rows": rows,
                        "local_xyz": local_xyz.int(),
                        "local_coords": local_coords.int(),
                        "projection_coords": coords_c64.index_select(0, rows).int(),
                        "owned_row_ids": local_rows,
                        "tokens": int(len(rows)),
                    }
                )
                coverage.index_add_(
                    0, rows, torch.ones(len(rows), dtype=torch.int16)
                )
                offset += len(rows)
                cube_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError("baseline C64 support is not partitioned exactly once")
    if offset != len(coords_c64):
        raise RuntimeError("context32 records do not cover baseline C64")
    return records


def make_global_c128_context64_blocks(
    coords_c128: torch.Tensor,
) -> list[dict[str, Any]]:
    """Partition the promoted support into local C64 blocks."""
    coords_c128 = validate_coords(coords_c128, GLOBAL_GRID, "promoted C128")
    xyz = coords_c128[:, 1:]
    starts = tuple(range(0, GLOBAL_GRID, BLOCK64))
    coverage = torch.zeros(len(coords_c128), dtype=torch.int16)
    records: list[dict[str, Any]] = []
    offset = 0
    cube_id = 0
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + BLOCK64)).all(dim=1)
                )[0].long()
                if not len(rows):
                    cube_id += 1
                    continue
                local_xyz = xyz.index_select(0, rows) - start
                local_coords = torch.cat(
                    (
                        torch.zeros((len(rows), 1), dtype=torch.int32),
                        local_xyz,
                    ),
                    dim=1,
                )
                local_rows = torch.arange(
                    offset, offset + len(rows), dtype=torch.long
                )
                records.append(
                    {
                        "cube_id": int(cube_id),
                        "start": (sx, sy, sz),
                        "global_row_ids": local_rows,
                        "local_xyz": local_xyz.int(),
                        "local_coords": local_coords.int(),
                        "projection_coords": coords_c128.index_select(0, rows).int(),
                        "owned_row_ids": local_rows,
                        "tokens": int(len(rows)),
                    }
                )
                coverage.index_add_(
                    0, rows, torch.ones(len(rows), dtype=torch.int16)
                )
                offset += len(rows)
                cube_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError("promoted C128 support is not partitioned exactly once")
    if offset != len(coords_c128):
        raise RuntimeError("context64 records do not cover promoted C128")
    return records


@torch.no_grad()
def decode_c32_to_global_c128(
    pipeline: Any,
    records32: Sequence[Mapping[str, Any]],
    state32: torch.Tensor,
    device: torch.device,
    output: Path,
    max_tokens: int,
    max_blocks: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Decode local C32 support to local C64, then promote it to global C128."""
    groups = previous.make_groups(records32, max_tokens, max_blocks)
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    local_results: list[dict[str, Any]] = []
    try:
        for group_index, group in enumerate(groups):
            values: list[SparseTensor] = []
            for record in group:
                rows = record["global_row_ids"].long()
                raw = denormalize(
                    state32.index_select(0, rows).to(device),
                    pipeline.shape_slat_normalization,
                )
                values.append(SparseTensor(raw, record["local_coords"].int()))
            packed = previous.pack_sparse_values(
                values, device, "local C32 decoder input"
            )
            print(
                f"[decode support] group={group_index + 1}/{len(groups)} "
                f"blocks={len(group)} tokens={len(packed.coords):,} "
                "local C32->local C64",
                flush=True,
            )
            candidates = decoder.upsample(packed, upsample_times=4)
            for batch_id, record in enumerate(group):
                part = candidates[candidates[:, 0] == batch_id].detach().cpu()
                if not len(part):
                    raise RuntimeError(
                        f"cube {record['cube_id']}: empty local C64 support"
                    )
                local_xyz = torch.div(
                    (part[:, 1:].float() + 0.5) * BLOCK64,
                    C512_GRID,
                    rounding_mode="floor",
                ).int()
                local_coords = torch.cat(
                    (
                        torch.zeros((len(local_xyz), 1), dtype=torch.int32),
                        local_xyz,
                    ),
                    dim=1,
                )
                local_coords = sort_coords(
                    validate_coords(
                        local_coords.unique(dim=0),
                        BLOCK64,
                        f"cube {record['cube_id']} local C64",
                    ),
                    BLOCK64,
                )
                start = torch.tensor(record["start"], dtype=torch.int32)
                # The input block is 32 cells in global C64.  Its decoded
                # local C64 occupies the corresponding 64 cells in global
                # C128, hence the factor 2 on the block start.
                global_xyz = start[None] * 2 + local_coords[:, 1:]
                global_coords = torch.cat(
                    (
                        torch.zeros((len(global_xyz), 1), dtype=torch.int32),
                        global_xyz,
                    ),
                    dim=1,
                )
                global_coords = validate_coords(
                    global_coords,
                    GLOBAL_GRID,
                    f"cube {record['cube_id']} global C128",
                )
                local_results.append(
                    {
                        "cube_id": int(record["cube_id"]),
                        "start_c64": tuple(int(x) for x in record["start"]),
                        "local_xyz_c64": local_coords[:, 1:].int(),
                        "local_coords_c64": local_coords.int(),
                        "global_coords_c128": global_coords.int(),
                        "tokens": int(len(global_coords)),
                    }
                )
                atomic_save(
                    output / "support/stage1_local_c64"
                    / f"cube_{int(record['cube_id']):02d}.pt",
                    {
                        "format": FORMAT,
                        "cube_id": int(record["cube_id"]),
                        "input_local_c32": record["local_coords"],
                        "output_local_c64": local_coords,
                        "output_global_c128": global_coords,
                        "mapping": "global_c128 = start_c64 * 2 + local_c64",
                        "features_carried": False,
                    },
                )
            del candidates, packed, values
            empty_cuda()
    finally:
        decoder.cpu()
        decoder.low_vram = False
    local_results.sort(key=lambda item: int(item["cube_id"]))
    global_parts = [item["global_coords_c128"] for item in local_results]
    global_coords = sort_coords(
        validate_coords(
            torch.cat(global_parts, dim=0), GLOBAL_GRID, "assembled global C128"
        ).unique(dim=0),
        GLOBAL_GRID,
    )
    if len(global_coords) != sum(int(item["tokens"]) for item in local_results):
        raise RuntimeError(
            "stage1 C32 blocks produced overlapping global C128 coordinates"
        )
    records64 = make_global_c128_context64_blocks(global_coords)
    atomic_save(
        output / "support/stage1_global_c128_support.pt",
        {
            "format": FORMAT,
            "coords": global_coords,
            "source": "local C32 Flow -> decoder local C64 -> block start * 2",
            "features_carried": False,
        },
    )
    atomic_json(
        output / "support/stage1_promotion_summary.json",
        {
            "format": FORMAT,
            "input_baseline_c64_tokens": int(sum(
                int(r["tokens"]) for r in records32
            )),
            "local_c64_tokens": int(sum(
                int(item["tokens"]) for item in local_results
            )),
            "global_c128_tokens": int(len(global_coords)),
            "active_context32_blocks": len(records32),
            "active_context64_blocks": len(records64),
            "mapping": "global_c128 = start_c64 * 2 + local_c64",
            "features_carried": False,
        },
    )
    return global_coords, records64


def assemble_global_c128(
    records64: Sequence[Mapping[str, Any]],
    state64: torch.Tensor,
    shape_spec: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild one global C128 sparse latent from disjoint local blocks."""
    coords_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    for record in records64:
        rows = record["global_row_ids"].long()
        local = record["local_xyz"].int()
        start = torch.tensor(record["start"], dtype=torch.int32)
        global_xyz = local + start
        coords_parts.append(
            torch.cat(
                (torch.zeros((len(local), 1), dtype=torch.int32), global_xyz),
                dim=1,
            )
        )
        raw_parts.append(
            denormalize(state64.index_select(0, rows), shape_spec).float()
        )
    coords = torch.cat(coords_parts, dim=0)
    raw = torch.cat(raw_parts, dim=0)
    coords = validate_coords(coords, GLOBAL_GRID, "global C128 decode coords")
    xyz = coords[:, 1:].long()
    key = (xyz[:, 0] * GLOBAL_GRID + xyz[:, 1]) * GLOBAL_GRID + xyz[:, 2]
    order = torch.argsort(key, stable=True)
    coords = coords.index_select(0, order).contiguous()
    raw = raw.index_select(0, order).contiguous()
    if len(coords) != len(coords.unique(dim=0)):
        raise RuntimeError("global C128 decode support contains duplicates")
    return coords, raw


def make_contact_sheet(
    paths: Sequence[tuple[str, Path]], output: Path
) -> None:
    if not paths:
        return
    images = [(label, Image.open(path).convert("RGB")) for label, path in paths]
    panel = images[0][1].width
    header = 32
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new(
        "RGB", (cols * panel, rows * (panel + header)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % cols) * panel
        y = (index // cols) * (panel + header)
        sheet.paste(image, (x, y + header))
        draw.text((x + 8, y + 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def render_normals(
    mesh: Mesh,
    camera: Mapping[str, Any],
    output: Path,
    resolution: int,
    angles: Sequence[int],
    chunk_size: int,
) -> dict[str, Any]:
    from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views
    from pixal3d.utils import render_utils

    extrinsics, intrinsics, _ = _make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), angles
    )
    rendered = render_utils.render_frames(
        mesh,
        [extrinsics[a].to(mesh.device) for a in angles],
        [intrinsics.to(mesh.device) for _ in angles],
        options={
            "resolution": int(resolution),
            "near": 0.01,
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "chunk_size": int(chunk_size),
        },
        return_types=["normal", "mask"],
        verbose=True,
    )
    render_dir = output / f"multiview_{resolution}"
    render_dir.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[str, Path]] = []
    masks: list[str] = []
    views: list[str] = []
    for index, angle in enumerate(angles):
        normal = np.asarray(rendered["normal"][index])
        mask = np.asarray(rendered["mask"][index])
        normal_path = render_dir / f"view_{int(angle):03d}_camera_normal.png"
        mask_path = render_dir / f"view_{int(angle):03d}_mask.png"
        Image.fromarray(normal).convert("RGB").save(normal_path)
        Image.fromarray(mask).convert("L").save(mask_path)
        paths.append((f"yaw {angle} camera normal", normal_path))
        views.append(str(normal_path.resolve()))
        masks.append(str(mask_path.resolve()))
    sheet = render_dir / "camera_normal_contact_sheet.png"
    make_contact_sheet(paths, sheet)
    return {
        "camera_normal_contact_sheet": str(sheet.resolve()),
        "views": views,
        "masks": masks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline-c64", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--canonical-4096", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed-c32", type=int, default=72201)
    parser.add_argument("--seed-c64", type=int, default=72202)
    parser.add_argument("--max-flow-tokens", type=int, default=20000)
    parser.add_argument("--max-flow-blocks", type=int, default=8)
    parser.add_argument("--max-decode-tokens", type=int, default=20000)
    parser.add_argument("--max-decode-blocks", type=int, default=8)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200000)
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.steps != STEPS:
        raise ValueError("this experiment uses the native 12-step schedule")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = parse_camera(args.camera)
    image_4096 = Image.open(args.canonical_4096).convert("RGB")
    if image_4096.size != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(f"canonical image must be 4096x4096, got {image_4096.size}")
    angles = parse_angles(args.angles)
    started = time.perf_counter()

    config = {
        "format": FORMAT,
        "status": "running",
        "cuda_device": int(args.cuda_device),
        "route": (
            "baseline C64 -> context32 local C32 Shape512 -> local C64 "
            "-> global C128 -> context64 local C64 Shape1024 "
            "-> global C128 decode2048"
        ),
        "baseline_c64": args.baseline_c64,
        "canonical_4096": args.canonical_4096,
        "actual_feature_input": "native 1024x1024 crop directly from canonical_4096",
        "condition_projection_grids": {
            "shape512": BASELINE_GRID,
            "shape1024": GLOBAL_GRID,
        },
        "flow_coordinate_ranges": {
            "shape512": [0, 31],
            "shape1024": [0, 63],
        },
        "changed_support_latent_policy": "fresh noise after each support change",
        "texture": False,
        "steps": int(args.steps),
        "seed_c32": int(args.seed_c32),
        "seed_c64": int(args.seed_c64),
    }
    atomic_json(output / "config.json", config)

    print("[model] loading geometry-only pipeline", flush=True)
    pipeline = previous.cascade.init_shape_pipeline(args.model_path, device)
    coords64, features64 = previous.load_baseline(args.baseline_c64, pipeline)
    atomic_save(
        output / "support/baseline_c64_endpoint.pt",
        {
            "format": FORMAT,
            "coords": coords64,
            "features": features64,
            "complete_endpoint": True,
            "features_used": "support only; no latent copied after repartition",
        },
    )
    records32 = make_c64_context32_blocks(coords64)
    baseline_tokens = int(len(coords64))
    context32_blocks = int(len(records32))
    atomic_json(
        output / "support/context32_block_layout.json",
        {
            "format": FORMAT,
            "input_grid": BASELINE_GRID,
            "context": BLOCK32,
            "stride": BLOCK32,
            "tokens": int(len(coords64)),
            "active_blocks": len(records32),
            "all_blocks": (BASELINE_GRID // BLOCK32) ** 3,
            "blocks": [
                {
                    "cube_id": int(r["cube_id"]),
                    "start_c64": list(r["start"]),
                    "tokens": int(r["tokens"]),
                    "local_indices": [0, 31],
                }
                for r in records32
            ],
        },
    )
    print(
        f"[support] baseline C64 tokens={len(coords64):,}, "
        f"active context32 blocks={len(records32)}",
        flush=True,
    )

    cond32 = previous.extract_conditions(
        pipeline,
        image_4096,
        camera,
        records32,
        BASELINE_GRID,
        "shape512",
        output,
    )
    state32 = previous.run_complete_local_flow(
        pipeline,
        records32,
        cond32,
        "shape_slat_flow_model_512",
        device,
        args.seed_c32,
        "shape512",
        output,
        args.max_flow_tokens,
        args.max_flow_blocks,
    )
    global_coords128, records64 = decode_c32_to_global_c128(
        pipeline,
        records32,
        state32,
        device,
        output,
        args.max_decode_tokens,
        args.max_decode_blocks,
    )
    stage1_tokens = int(len(global_coords128))
    del cond32, state32, records32, coords64, features64
    empty_cuda()

    print(
        f"[support] stage1 promoted global C128 tokens={stage1_tokens:,}, "
        f"active context64 blocks={len(records64)}",
        flush=True,
    )
    cond64 = previous.extract_conditions(
        pipeline,
        image_4096,
        camera,
        records64,
        GLOBAL_GRID,
        "shape1024",
        output,
    )
    state64 = previous.run_complete_local_flow(
        pipeline,
        records64,
        cond64,
        "shape_slat_flow_model_1024",
        device,
        args.seed_c64,
        "shape1024",
        output,
        args.max_flow_tokens,
        args.max_flow_blocks,
    )
    decode_coords, decode_raw = assemble_global_c128(
        records64, state64, pipeline.shape_slat_normalization
    )
    atomic_save(
        output / "support/global_c128_decode_input.pt",
        {
            "format": FORMAT,
            "coords": decode_coords,
            "raw_features": decode_raw,
            "source": "all context64 local Shape1024 outputs assembled before decode",
            "features_copied_after_support_change": False,
        },
    )
    print(
        f"[decode mesh] global C128 together, tokens={len(decode_coords):,}",
        flush=True,
    )
    slat = SparseTensor(decode_raw.to(device), decode_coords.to(device))
    meshes, _ = pipeline.decode_shape_slat(slat, DECODE_RESOLUTION)
    if len(meshes) != 1:
        raise RuntimeError(f"expected one global decoded mesh, got {len(meshes)}")
    mesh = meshes[0]
    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = final_dir / "geometry_mesh.pt"
    atomic_save(
        mesh_path,
        {
            "format": FORMAT,
            "mesh": mesh.cpu(),
            "mesh_coordinate_system": "global normalized object q [-0.5,0.5]",
        },
    )
    glb_path = final_dir / "geometry_mesh.glb"
    glb_error: str | None = None
    try:
        import trimesh

        trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        ).export(glb_path)
    except Exception as exc:
        glb_error = repr(exc)
        atomic_json(final_dir / "glb_export_error.json", {"error": glb_error})
    render = render_normals(
        mesh.to(device),
        camera,
        final_dir,
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "texture_executed": False,
        "cuda_device": int(args.cuda_device),
        "seconds": time.perf_counter() - started,
        "baseline_c64_tokens": baseline_tokens,
        "stage1_global_c128_tokens": stage1_tokens,
        "context32_active_blocks": context32_blocks,
        "context64_active_blocks": len(records64),
        "shape512_flow_tokens": int(
            json.loads(
                (output / "flow/shape512/summary.json").read_text()
            )["tokens"]
        ),
        "shape1024_flow_tokens": int(state64.shape[0]),
        "global_decode_c128_tokens": int(len(decode_coords)),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_pt": str(mesh_path.resolve()),
        "mesh_glb": str(glb_path.resolve()) if glb_error is None else None,
        "glb_error": glb_error,
        "camera_normal_render": render,
        "condition_source": "direct native 1024x1024 crop from canonical_4096",
        "condition_crop_size": [CROP_SIZE, CROP_SIZE],
        "projection_grids": {"shape512": BASELINE_GRID, "shape1024": GLOBAL_GRID},
        "local_coordinate_ranges": {
            "shape512": [0, 31],
            "shape1024": [0, 63],
        },
        "baseline_features_copied_after_support_change": False,
    }
    atomic_json(output / "summary.json", summary)
    config["status"] = "complete"
    atomic_json(output / "config.json", config)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
