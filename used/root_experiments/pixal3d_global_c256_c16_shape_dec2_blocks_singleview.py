#!/usr/bin/env python3
"""Global C256 assembled from baseline-C16 local Shape512/decoder-x2 cascades.

For every nonempty C16 block of the native baseline C64 support:
  C16 Shape512 flow -> shape decoder x2 -> native local C64 support.
The disjoint local supports are mapped into one Global C256 support.  HR Shape
and Texture use one global noise/state each; all local block velocities are
evaluated before the global state is synchronously updated.  Decode happens
once at Global C256 -> 4096.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
from PIL import Image

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global_c256_restructured_blocks_singleview as core
from inference import MODEL_PATH, init_pipeline


FORMAT = "pixal3d_global_c256_c16_shape_dec2_blocks_singleview_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def generate_local_c64_supports(
    pipeline: Any,
    image512: Image.Image,
    camera: Mapping[str, float],
    baseline64: torch.Tensor,
    records: list[dict[str, Any]],
    output: Path,
    seed: int,
    steps: int,
    device: torch.device,
) -> None:
    root = output / "local_c16_shape_dec2"
    pending = [
        record for record in records
        if not (root / f"block_{int(record['cube_id']):02d}" / "c64_support.pt").is_file()
    ]
    if not pending:
        return
    model = pipeline.image_cond_model_shape_512
    cached = core.extract_full_image_features(model, image512, device)
    decoder = pipeline.models["shape_slat_decoder"]
    fov = float(camera["camera_angle_x"])
    distance = float(camera["distance"])
    xyz = baseline64[:, 1:4].cpu().int()
    for order, record in enumerate(pending, 1):
        block_id = int(record["cube_id"])
        index = tuple(int(v) for v in record["block_index"])
        base_start = torch.tensor(index, dtype=torch.int32) * 16
        inside = ((xyz >= base_start) & (xyz < base_start + 16)).all(1)
        local16_xyz = xyz[inside] - base_start
        if int(inside.sum()) != int(record["baseline_tokens"]):
            raise RuntimeError(f"baseline row mismatch for block {block_id}")
        coords16 = torch.cat((
            torch.zeros((len(local16_xyz), 1), dtype=torch.int32), local16_xyz), 1)

        transform = core.local_to_global_camera_transform(
            model, block_start=tuple(int(v) for v in base_start.tolist()),
            global_resolution=64, global_extent=16, distance=distance, device=device)
        projection_error = core.validate_projection_transform(
            model, transform, local_resolution=16, global_resolution=64,
            block_start=tuple(int(v) for v in base_start.tolist()),
            distance=distance, fov=fov, device=device)
        glob, proj = core.project_cached_features(
            model, cached, transform=transform, fov=fov, distance=distance,
            coords=coords16.to(device), grid_resolution=16)
        torch.manual_seed(seed + 20_000 + block_id)
        torch.cuda.manual_seed_all(seed + 20_000 + block_id)
        lr16 = pipeline.sample_shape_slat(
            core.condition_from_sparse(glob, proj, coords16.to(device)),
            pipeline.models["shape_slat_flow_model_512"], coords16.to(device),
            {"steps": steps})
        if pipeline.low_vram:
            decoder.to(device); decoder.low_vram = True
        candidates64 = decoder.upsample(lr16, upsample_times=2)
        if pipeline.low_vram:
            decoder.cpu(); decoder.low_vram = False
        coords64 = candidates64.int().unique(dim=0).cpu().contiguous()
        if len(coords64) == 0:
            raise RuntimeError(f"decoder x2 returned empty C64 support for block {block_id}")
        if torch.any(coords64[:, 1:] < 0) or torch.any(coords64[:, 1:] >= 64):
            raise RuntimeError(f"decoder x2 returned coordinates outside C64 for block {block_id}")
        block_dir = root / f"block_{block_id:02d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        cube_flow.atomic_save(block_dir / "c64_support.pt", {
            "format": FORMAT, "block_id": block_id, "block_index": index,
            "baseline_local_c16": coords16, "shape512_endpoint": lr16.cpu(),
            "coords": coords64, "decoder_upsample_times": 2,
            "candidate_resolution": 64, "quantizer": None,
            "projection_max_error_pixels": projection_error,
        })
        record["generated_c64_tokens"] = int(len(coords64))
        print(
            f"[local-dec2] block={block_id:02d} {order}/{len(pending)} "
            f"C16={len(coords16):,} C64={len(coords64):,}", flush=True)
        del glob, proj, lr16, candidates64, coords64
        empty_cuda()
    del cached
    if pipeline.low_vram:
        model.cpu()
    empty_cuda()


def assemble_global_support(
    records: list[dict[str, Any]], output: Path,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    source = output / "local_c16_shape_dec2"
    parts: list[torch.Tensor] = []
    raw_count = 0
    for record in records:
        block_id = int(record["cube_id"])
        index = tuple(int(v) for v in record["block_index"])
        payload = torch.load(
            source / f"block_{block_id:02d}" / "c64_support.pt",
            map_location="cpu", weights_only=False)
        local = payload["coords"].int()
        global_xyz = local[:, 1:4] + torch.tensor(index, dtype=torch.int32) * 64
        part = torch.cat((torch.zeros((len(local), 1), dtype=torch.int32), global_xyz), 1)
        parts.append(part)
        raw_count += len(part)
        record["generated_c64_tokens"] = int(len(local))
    concatenated = torch.cat(parts, 0).int()
    coords = concatenated.unique(dim=0).int().contiguous()
    duplicate_count = int(raw_count - len(coords))
    if duplicate_count != 0:
        raise RuntimeError(f"disjoint block assembly unexpectedly has {duplicate_count} duplicate points")
    xyz = coords[:, 1:4]
    if torch.any(xyz < 0) or torch.any(xyz >= 256):
        raise RuntimeError("assembled support outside Global C256")

    coverage = torch.zeros(len(coords), dtype=torch.int16)
    flow_records: list[dict[str, Any]] = []
    for record in records:
        block_id = int(record["cube_id"])
        index = tuple(int(v) for v in record["block_index"])
        start = torch.tensor(index, dtype=torch.int32) * 64
        mask = ((xyz >= start) & (xyz < start + 64)).all(1)
        rows = torch.where(mask)[0].long()
        local_xyz = xyz[mask] - start
        coverage[rows] += 1
        flow_records.append({
            "cube_id": block_id, "block_index": index,
            "start": tuple(int(v) for v in start.tolist()),
            "global_row_ids": rows, "owned_row_ids": rows,
            "local_xyz": local_xyz,
        })
    if not torch.all(coverage == 1):
        raise RuntimeError("Global C256 support ownership is not exactly one")
    cube_flow.atomic_save(output / "global_support" / "global_c256_support.pt", {
        "format": FORMAT, "coords": coords})
    core.atomic_json(output / "global_support" / "summary.json", {
        "format": FORMAT, "tokens": int(len(coords)),
        "active_blocks": len(records), "concatenated_tokens": raw_count,
        "duplicate_points": duplicate_count,
        "coverage_min": int(coverage.min()), "coverage_max": int(coverage.max()),
        "records": [{
            "cube_id": int(record["cube_id"]),
            "block_index": record["block_index"],
            "baseline_tokens": int(record["baseline_tokens"]),
            "generated_c64_tokens": int(record["generated_c64_tokens"]),
        } for record in records],
    })
    return coords, flow_records


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=base / "global_camera.json")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--baseline-c64", type=Path, default=Path(
        "outputs/global_c256_restructured_blocks_cuda5/baseline_pre_hr/baseline_c64_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_c16_shape512_dec2_all_blocks_cuda5"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=43001)
    p.add_argument("--texture-seed", type=int, default=44001)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-after-support", action="store_true")
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
    camera = json.loads(args.camera.read_text())
    camera["mesh_scale"] = 1.0
    core.atomic_json(output / "config.json", {
        "format": FORMAT, "args": vars(args), "camera": camera,
        "local_support_path": "baseline C16 -> Shape512 flow -> shape decoder x2 -> native C64",
        "global_flow": "one global noise/state; per-step all local velocities then one synchronous update",
        "block_overlap": "disjoint Global C256 C64 blocks; duplicates must equal zero",
        "decode": "one Global C256 Shape+Texture decode at 4096",
    })

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image512, image1024 = canonical["image_512"], canonical["image_1024"]
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    image512.save(input_dir / "global_512.png")
    image1024.save(input_dir / "global_1024.png")

    baseline64 = torch.load(args.baseline_c64, map_location="cpu", weights_only=False)["coords"].int()
    records = core.active_blocks(baseline64)
    core.atomic_json(output / "active_blocks.json", {
        "count": len(records), "baseline_c64_tokens": int(len(baseline64)),
        "records": records})
    print(f"[active] baseline C64={len(baseline64):,} blocks={len(records)}", flush=True)

    generate_local_c64_supports(
        pipeline, image512, camera, baseline64, records, output,
        args.seed, args.steps, device)
    coords, flow_records = assemble_global_support(records, output)
    print(f"[global-support] tokens={len(coords):,} duplicate_points=0", flush=True)
    if args.stop_after_support:
        print(f"[done-support] {output}", flush=True)
        return

    shape_condition = core.build_global_conditions(
        pipeline, image1024, camera, coords, flow_records, "shape", output, device)
    shape = core.synchronous_flow(
        stage="shape", pipeline=pipeline, records=flow_records,
        condition=shape_condition, output=output, device=device,
        seed=args.shape_seed, steps=args.steps, concat=None)
    del shape_condition
    empty_cuda()

    texture_condition = core.build_global_conditions(
        pipeline, image1024, camera, coords, flow_records, "texture", output, device)
    texture = core.synchronous_flow(
        stage="texture", pipeline=pipeline, records=flow_records,
        condition=texture_condition, output=output, device=device,
        seed=args.texture_seed, steps=args.steps, concat=shape)
    del texture_condition
    empty_cuda()

    decoded = core.decode_global(pipeline, coords, shape, texture, output, device)
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete",
        "baseline_c64_tokens": int(len(baseline64)),
        "active_blocks": len(records), "global_c256_tokens": int(len(coords)),
        "duplicate_points": 0, "decode": decoded,
        "seconds": time.perf_counter() - started,
    })
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
