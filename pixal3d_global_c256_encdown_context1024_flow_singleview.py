#!/usr/bin/env python3
"""C4096 shape-encoder downsample -> global C256 -> disjoint C64 flow.

This is the encoder-downsample/full-1024-image-condition variant of
``pixal3d_global_c256_cube_owner_flow_singleview``.  It deliberately reuses
that experiment's flow fusion, checkpointing, normalization and single global
C256-to-4096 decode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Sparse CUDA allocation policy must be selected before importing torch.  The
# global C4096 encoder has a high transient activation peak and otherwise the
# caching allocator can strand enough reserved segments to fail the final MLP.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import o_voxel
import torch
from PIL import Image

import pixal3d.models as pixal3d_models
import pixal3d_global_c256_cube_owner_flow_singleview as base
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global_c256_encdown_context1024_flow_singleview_v1"
GRID = 256
CUBE = 64                 # physical C4096 context width: 1024
STRIDE = 64               # physical C4096 stride: 1024
STARTS = (0, 64, 128, 192)
FLOW_BATCH_SIZE = 44
INPUT_GRID = 4096
IMAGE_SIZE = 1024
DEFAULT_ROOT = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
DEFAULT_ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
)


def cube_layout(
    grid: int = GRID, cube: int = CUBE, stride: int = STRIDE
) -> list[dict[str, Any]]:
    starts = tuple(range(0, grid - cube + 1, stride))
    if starts != STARTS:
        raise ValueError(f"expected disjoint C64 layout {STARTS}, got {starts}")
    return [
        {"cube_id": i, "start": (sx, sy, sz)}
        for i, (sx, sy, sz) in enumerate(
            (x, y, z) for x in starts for y in starts for z in starts
        )
    ]


def build_cube_records(
    global_coords: torch.Tensor,
    grid: int = GRID,
    cube: int = CUBE,
    stride: int = STRIDE,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if global_coords.ndim != 2 or global_coords.shape[1] != 4:
        raise ValueError("global_coords must be [N,4]")
    xyz = global_coords[:, 1:].cpu().to(torch.int32)
    records: list[dict[str, Any]] = []
    coverage = torch.zeros(xyz.shape[0], dtype=torch.int16)
    for item in cube_layout(grid, cube, stride):
        start = torch.tensor(item["start"], dtype=torch.int32)
        inside = ((xyz >= start) & (xyz < start + cube)).all(1)
        ids = torch.where(inside)[0].to(torch.int64)
        local = xyz.index_select(0, ids) - start
        if ids.numel() and torch.unique(base.linear_keys(local, cube)).numel() != ids.numel():
            raise RuntimeError("local C64 coordinates are not unique")
        coverage.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int16))
        records.append(
            {
                **item,
                "start4096": tuple(int(v) * 16 for v in item["start"]),
                "global_row_ids": ids,
                "local_xyz": local,
                "coordinate_transform": "global C256 minus disjoint C64 context start",
            }
        )
    if len(records) != 64:
        raise RuntimeError(f"expected 64 contexts, got {len(records)}")
    if coverage.numel() and not bool((coverage == 1).all()):
        raise RuntimeError("disjoint context layout must cover every global row exactly once")
    return records, coverage


def _encoder_fingerprint(args: argparse.Namespace, hashes: Mapping[str, str]) -> dict[str, Any]:
    encoder = Path(args.shape_encoder).expanduser().resolve()
    return {
        **hashes,
        "schema": FORMAT,
        "input_grid": INPUT_GRID,
        "encoder": str(encoder),
        "encoder_json_sha256": base.sha256_file(encoder.with_suffix(".json")),
        "encoder_weights_sha256": base.sha256_file(encoder.with_suffix(".safetensors")),
        "voxelizer": {
            "function": "o_voxel.convert.mesh_to_flexible_dual_grid",
            "grid_size": INPUT_GRID,
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "face_weight": 1.0,
            "boundary_weight": 0.2,
            "regularization_weight": 1e-2,
        },
    }


@torch.no_grad()
def prepare_support(
    args: argparse.Namespace, out: Path
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor, dict[str, Any]]:
    vertices, faces, hashes = base._load_mesh_geometry(Path(args.baseline_mesh))
    fingerprint = _encoder_fingerprint(args, hashes)
    fp_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    cache = out / "support" / "global_c256_encoder_support.pt"
    reused = False
    voxel_seconds = encoder_seconds = 0.0
    input_tokens = None

    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("fingerprint_sha256") != fp_hash:
            raise RuntimeError("encoder-downsample support cache fingerprint mismatch")
        coords = payload["coords"].to(torch.int32).cpu().contiguous()
        input_tokens = int(payload["input_c4096_tokens"])
        reused = True
    else:
        started = time.perf_counter()
        raw_coords, dual_world, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=vertices,
            faces=faces,
            grid_size=INPUT_GRID,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
        voxel_seconds = time.perf_counter() - started
        raw_coords = raw_coords.to(torch.int32).cpu().contiguous()
        input_tokens = int(raw_coords.shape[0])
        dual = (dual_world.to(torch.float32).cpu() * INPUT_GRID - raw_coords.float()).clamp(0, 1)
        coords4 = torch.cat((torch.zeros_like(raw_coords[:, :1]), raw_coords), 1).to(args.device)
        vertex_sparse = SparseTensor(dual.to(args.device), coords4)
        intersected_sparse = vertex_sparse.replace(intersected.to(args.device))

        encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
        encoder.to(args.device)
        started = time.perf_counter()
        latent = encoder(vertex_sparse, intersected_sparse, sample_posterior=False)
        torch.cuda.synchronize(torch.device(args.device))
        encoder_seconds = time.perf_counter() - started
        if not isinstance(latent, SparseTensor):
            raise TypeError("shape encoder did not return SparseTensor")
        coords = latent.coords.detach().cpu().to(torch.int32).contiguous()
        if bool((coords[:, 0] != 0).any()):
            raise RuntimeError("encoder output has a nonzero batch index")
        if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= GRID)).any()):
            raise RuntimeError("encoder output does not lie on global C256")
        order = torch.argsort(base.linear_keys(coords[:, 1:]), stable=True)
        coords = coords.index_select(0, order)
        keys = base.linear_keys(coords[:, 1:])
        if torch.unique(keys).numel() != coords.shape[0]:
            raise RuntimeError("encoder C256 support contains duplicate coordinates")
        base.atomic_save(
            cache,
            {
                "format": FORMAT,
                "fingerprint": fingerprint,
                "fingerprint_sha256": fp_hash,
                "input_c4096_tokens": input_tokens,
                "coords": coords,
                "encoder_features_intentionally_discarded": True,
            },
        )
        del latent, encoder, vertex_sparse, intersected_sparse, raw_coords, dual_world, dual, intersected
        base._empty_cuda()

    records, coverage = build_cube_records(coords)
    owner, owner_stats = base.build_owner_map(coords, records)
    support_hash = base.tensor_sha256(coords)
    owner_hash = base.tensor_sha256(owner)
    base.atomic_save(
        out / "support" / "global_c256_support.pt",
        {
            "format": FORMAT,
            "coords": coords,
            "global_row_id": torch.arange(coords.shape[0]),
            "support_sha256": support_hash,
            "fingerprint": fingerprint,
            "source": "official shape encoder output support; encoder features discarded",
        },
    )
    base.atomic_save(
        out / "cubes" / "owner_map.pt",
        {"owner_cube_id": owner, "owner_sha256": owner_hash, "support_sha256": support_hash},
    )
    token_rows = []
    for rec in records:
        ids = rec["global_row_ids"]
        base.atomic_save(
            out / "cubes" / f"cube_{int(rec['cube_id']):03d}.pt",
            {
                "cube_id": rec["cube_id"],
                "start": rec["start"],
                "global_row_ids": ids,
                "local_coords": torch.cat((torch.zeros((ids.numel(), 1), dtype=torch.int32), rec["local_xyz"]), 1),
                "owned_row_ids": rec["owned_row_ids"],
            },
        )
        token_rows.append({"cube_id": rec["cube_id"], "start": rec["start"], "tokens": int(ids.numel())})
    c4096_stats = {
        "unique_c4096_token_count": input_tokens,
        "input_c4096_token_count": input_tokens,
        "voxelize_seconds": voxel_seconds,
        "encoder_seconds": encoder_seconds,
        "cache_reused": reused,
        "downsample": "official shape encoder C4096 -> C256",
    }
    c256_stats = {
        "unique_c256_token_count": int(coords.shape[0]),
        "coord_min": coords[:, 1:].min(0).values,
        "coord_max": coords[:, 1:].max(0).values,
        "support_sha256": support_hash,
        "encoder_features_intentionally_discarded": True,
    }
    cube_stats = {
        **owner_stats,
        "owner_sha256": owner_hash,
        "coverage_histogram": base.histogram(coverage.tolist()),
        "cube_count": 64,
        "empty_cubes": sum(not r["global_row_ids"].numel() for r in records),
        "nonempty_cubes": sum(bool(r["global_row_ids"].numel()) for r in records),
        "flow_active_cubes": sum(bool(r["owned_row_ids"].numel()) for r in records),
    }
    base.atomic_json(out / "support" / "c4096_stats.json", c4096_stats)
    base.atomic_json(out / "support" / "global_c256_stats.json", c256_stats)
    base.atomic_json(out / "cubes" / "coverage_owner_stats.json", cube_stats)
    base.atomic_json(out / "cubes" / "layout.json", {"starts": STARTS, "cube_count": 64, "records": token_rows})
    base.atomic_json(out / "inputs" / "fingerprints.json", fingerprint)
    return coords, records, owner, {"support": c256_stats, "cubes": cube_stats, "fingerprint_sha256": fp_hash}


def attach_cube_projection_crops(
    records: Sequence[Mapping[str, Any]], camera: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "source_grid": [IMAGE_SIZE, IMAGE_SIZE],
        "projection_grid": [IMAGE_SIZE, IMAGE_SIZE],
        "cube_count": len(records),
        "mode": "one full-image projection shared by all C256 rows",
    }


@torch.no_grad()
def build_condition(
    pipeline: Any,
    args: argparse.Namespace,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    camera: Mapping[str, Any],
    stage: str,
    out: Path,
) -> dict[str, Any]:
    image_path = Path(args.condition_image_4096)
    fingerprint = {
        "format": FORMAT,
        "stage": stage,
        "image_1024": str(image_path.resolve()),
        "image_1024_sha256": base.sha256_file(image_path),
        "support_sha256": base.tensor_sha256(coords),
        "camera": {k: float(camera[k]) for k in ("camera_angle_x", "distance", "mesh_scale")},
        "projection_grid": GRID,
        "condition": "single full 1024 DINO+NAF pass, then gather projected tokens by global row",
    }
    fp_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    cache = out / "conditions" / f"{stage}_global1024.pt"
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("fingerprint_sha256") != fp_hash:
            raise RuntimeError(f"{stage} global-1024 condition cache mismatch")
        glob = payload["global"]
        proj = payload["proj"]
    else:
        image = Image.open(image_path).convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"condition image must be 1024x1024, got {image.size}")
        model = pipeline.image_cond_model_shape_1024 if stage == "shape" else pipeline.image_cond_model_tex_1024
        condition = pipeline.get_proj_cond_shape(
            model,
            [image],
            coords.to(pipeline.device),
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=GRID,
            projection_crop_box=None,
            transform_matrix=None,
            preserve_image_resolution=False,
        )
        glob = condition["cond"]["global"].detach().cpu().contiguous()
        proj = condition["cond"]["proj"].feats.detach().cpu().contiguous()
        base.atomic_save(cache, {"format": FORMAT, "fingerprint": fingerprint, "fingerprint_sha256": fp_hash, "global": glob, "proj": proj})
        del condition, model, image
        base._empty_cuda()
    if proj.shape[0] != coords.shape[0]:
        raise RuntimeError("global projected image tokens are not aligned to C256 support")
    cubes = {
        int(rec["cube_id"]): {
            "global_row_ids": rec["global_row_ids"].cpu().long(),
            "global": glob,
            "proj": proj.index_select(0, rec["global_row_ids"].long()),
        }
        for rec in base.flow_condition_records(records, str(args.velocity_fusion))
    }
    manifest = {
        "format": FORMAT,
        "stage": stage,
        "fingerprint_sha256": fp_hash,
        "dino_passes": 1,
        "naf_passes": 1,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "global_c256_tokens": int(coords.shape[0]),
        "active_contexts": len(cubes),
        "token_routing": "project all global C256 points into full 1024 image, then gather by context",
    }
    base.atomic_json(out / "conditions" / f"{stage}_global1024_manifest.json", manifest)
    return {"cubes": cubes, "fingerprint_sha256": fp_hash, "manifest": manifest}


def fixed_flow_batch_probe(*args: Any, **kwargs: Any) -> int:
    requested = int(args[-1] if args else kwargs.get("requested", FLOW_BATCH_SIZE))
    if requested != FLOW_BATCH_SIZE:
        raise ValueError(f"this variant requires --flow-batch-size={FLOW_BATCH_SIZE}")
    return FLOW_BATCH_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=4)
    p.add_argument("--input-image", default=str(DEFAULT_ROOT / "canonical_1024.png"))
    p.add_argument("--condition-image-4096", dest="condition_image_4096", default=str(DEFAULT_ROOT / "canonical_1024.png"))
    p.add_argument("--canonical-gt", default=str(DEFAULT_ROOT / "canonical_1024.png"))
    p.add_argument("--foreground-mask", default=str(DEFAULT_ROOT / "canonical_1024.png"))
    p.add_argument("--baseline-mesh", default=str(DEFAULT_ROOT / "raw_ovoxel_mesh.pt"))
    p.add_argument("--camera-json", default=str(DEFAULT_ROOT / "global_camera.json"))
    p.add_argument("--output-dir", default="outputs/global_c256_encdown_context1024_flow_singleview_cuda4")
    p.add_argument("--model-path", default=base.MODEL_PATH)
    p.add_argument("--shape-encoder", default=str(DEFAULT_ENCODER))
    p.add_argument("--grid-resolution", type=int, default=GRID)
    p.add_argument("--cube-size", type=int, default=CUBE)
    p.add_argument("--cube-stride", type=int, default=STRIDE)
    p.add_argument("--shape-steps", type=int, default=12)
    p.add_argument("--texture-steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=43)
    p.add_argument("--texture-seed", type=int, default=44)
    p.add_argument("--flow-batch-size", type=int, default=FLOW_BATCH_SIZE)
    p.add_argument("--max-batch-tokens", type=int, default=10_000_000)
    p.add_argument("--velocity-fusion", choices=("owner", "gaussian"), default="owner")
    p.add_argument("--gaussian-sigma", type=float, default=32.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    # The reused main resolves globals from the base module, so install this
    # variant's contracts explicitly before entering it.
    base.FORMAT = FORMAT
    base.CUBE = CUBE
    base.STRIDE = STRIDE
    base.STARTS = STARTS
    base.CANONICAL_IMAGE_SIZE = IMAGE_SIZE
    base.prepare_support = prepare_support
    base.build_cube_records = build_cube_records
    base.cube_layout = cube_layout
    base.attach_cube_projection_crops = attach_cube_projection_crops
    base.build_condition = build_condition
    base.probe_max_batch = fixed_flow_batch_probe
    base.parse_args = parse_args
    base.main()
    # A successful --resume supersedes an earlier failed attempt in the same
    # output directory; do not leave a stale failure marker beside status=complete.
    failure = Path(parse_args().output_dir).resolve() / "failure.json"
    if failure.is_file():
        failure.unlink()


if __name__ == "__main__":
    main()
