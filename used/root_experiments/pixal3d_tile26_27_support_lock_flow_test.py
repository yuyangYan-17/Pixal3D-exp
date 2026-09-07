#!/usr/bin/env python3
"""Measure the effect of inheriting tile-26 C64 support in tile 27.

This is a diagnostic after the documented baseline-mesh -> local voxelize ->
shape-encode path.  It keeps tile-27 image conditioning and shape features,
replaces only sparse support coordinates, and runs the native texture SLat
flow with the sampler defaults saved by the collector.  No guidance or flow
weight is introduced.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


RESOLUTION = 64
CHANNELS = 32
OVERLAP_TILE_IDS = (26, 27)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_payload(root: Path, tile_id: int) -> Dict[str, Any]:
    path = root / "tiles" / f"tile_{tile_id:02d}" / "tile_latents.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("tile_id") != tile_id:
        raise RuntimeError(f"{path}: tile id mismatch")
    for key in ("shape_coords", "shape_norm", "global_centers_object"):
        if key not in payload:
            raise RuntimeError(f"{path}: missing {key}")
    return payload


def _box_mask(
    uv: torch.Tensor,
    valid: torch.Tensor,
    box: Sequence[int],
) -> torch.Tensor:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (
        valid
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )


def _q_to_c64_coords(q_local: torch.Tensor) -> torch.Tensor:
    indices = torch.round((q_local + 1.0) * ((RESOLUTION - 1) / 2.0))
    indices = indices.to(torch.int32)
    if bool(((indices < 0) | (indices >= RESOLUTION)).any().item()):
        raise RuntimeError("mapped support quantized outside C64")
    return torch.cat(
        [torch.zeros((indices.shape[0], 1), dtype=torch.int32), indices], dim=1
    )


def _nearest_rows(
    query: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []
    for start in range(0, int(query.shape[0]), int(chunk_size)):
        distance = torch.cdist(query[start : start + chunk_size], reference)
        values, indices = distance.min(dim=1)
        rows.append(indices)
        distances.append(values)
    if not rows:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )
    return torch.cat(rows), torch.cat(distances)


def _build_variant(
    *,
    native: Mapping[str, Any],
    native_global: torch.Tensor,
    forced_global: torch.Tensor,
    forced_local_q: torch.Tensor,
    name: str,
    replace_native_overlap: torch.Tensor,
) -> Dict[str, Any]:
    native_coords = native["shape_coords"].to(torch.int32)
    native_shape = native["shape_norm"].to(torch.float32)
    keep = ~replace_native_overlap
    base_coords = native_coords[keep]
    base_shape = native_shape[keep]
    base_global = native_global[keep]
    nearest, nearest_distance = _nearest_rows(forced_global, native_global)
    forced_shape = native_shape[nearest]
    forced_coords = forced_global.new_zeros((forced_global.shape[0], 4), dtype=torch.int32)
    forced_coords[:, 1:] = torch.round(
        (forced_local_q + 1.0) * ((RESOLUTION - 1) / 2.0)
    ).to(torch.int32)

    entries: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, bool]] = {}
    for row in range(int(base_coords.shape[0])):
        key = tuple(int(v) for v in base_coords[row, 1:].tolist())
        entries[key] = (base_global[row], base_shape[row], False)
    for row in range(int(forced_coords.shape[0])):
        key = tuple(int(v) for v in forced_coords[row, 1:].tolist())
        entries[key] = (forced_global[row], forced_shape[row], True)

    ordered = list(entries.items())
    coords = torch.tensor(
        [[0, key[0], key[1], key[2]] for key, _ in ordered], dtype=torch.int32
    )
    centers = torch.stack([value[0] for _, value in ordered], dim=0).to(torch.float32)
    shape = torch.stack([value[1] for _, value in ordered], dim=0).to(torch.float32)
    forced_mask = torch.tensor([value[2] for _, value in ordered], dtype=torch.bool)
    return {
        "name": name,
        "coords": coords,
        "shape": shape,
        "global_centers_object": centers,
        "forced_mask": forced_mask,
        "source_nearest_distance": nearest_distance,
        "support_tokens": int(coords.shape[0]),
        "forced_tokens_before_dedup": int(forced_coords.shape[0]),
        "forced_tokens_after_dedup": int(forced_mask.sum().item()),
    }


def _spatial_noise(
    center_sets: Iterable[torch.Tensor],
    *,
    seed: int,
    channels: int,
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    keys: List[Tuple[int, int, int]] = []
    for centers in center_sets:
        quantized = torch.round((centers + 0.5) * 4096.0).to(torch.int64)
        keys.extend(tuple(int(v) for v in row.tolist()) for row in quantized)
    unique = sorted(set(keys))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randn((len(unique), channels), generator=generator)
    return {key: values[index] for index, key in enumerate(unique)}


def _make_noise(
    centers: torch.Tensor,
    lookup: Mapping[Tuple[int, int, int], torch.Tensor],
) -> torch.Tensor:
    quantized = torch.round((centers + 0.5) * 4096.0).to(torch.int64)
    return torch.stack(
        [lookup[tuple(int(v) for v in row.tolist())] for row in quantized], dim=0
    ).to(torch.float32)


def _prediction_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "steps",
        "rescale_t",
        "verbose",
        "tqdm_desc",
        "record_trajectory",
        "trajectory_device",
        "return_model_history",
    }
    return {key: value for key, value in params.items() if key not in excluded}


@torch.no_grad()
def _run_texture_flow(
    *,
    pipeline: Any,
    variant: Mapping[str, Any],
    image: Image.Image,
    transform: Any,
    params: Mapping[str, Any],
    noise_lookup: Mapping[Tuple[int, int, int], torch.Tensor],
) -> Dict[str, Any]:
    device = torch.device("cuda")
    coords = variant["coords"].to(device=device, dtype=torch.int32).contiguous()
    shape = variant["shape"].to(device=device, dtype=torch.float32)
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image.convert("RGB")],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=RESOLUTION,
    )
    model = pipeline.models["tex_slat_flow_model_1024"]
    model.to(device)
    sampler = pipeline.tex_slat_sampler
    steps = int(params["steps"])
    rescale_t = float(params.get("rescale_t", 3.0))
    times = tuple(float(value) for value in sampler.timestep_schedule(steps, rescale_t))
    current = _make_noise(variant["global_centers_object"], noise_lookup)
    velocity_records: List[torch.Tensor] = []
    start_state = current.clone()
    try:
        for step in range(steps):
            t = times[step]
            t_next = times[step + 1]
            state = SparseTensor(current.to(device), coords)
            concat = SparseTensor(shape, coords)
            output = sampler.sample_once(
                model,
                state,
                t,
                t_next,
                **condition,
                **_prediction_kwargs(params),
                concat_cond=concat,
            )
            velocity = output.pred_v
            if not isinstance(velocity, SparseTensor):
                raise RuntimeError(f"{variant['name']}: sampler returned non-sparse velocity")
            velocity_cpu = velocity.feats.detach().cpu().to(torch.float32)
            if not torch.isfinite(velocity_cpu).all():
                raise RuntimeError(f"{variant['name']}: non-finite velocity at step {step}")
            velocity_records.append(velocity_cpu)
            current = current - float(t - t_next) * velocity_cpu
            del state, concat, output, velocity
    finally:
        model.cpu()
        del condition, shape, coords
        _empty_cuda_cache()
    return {
        "name": variant["name"],
        "initial": start_state,
        "endpoint": current,
        "velocities": torch.stack(velocity_records, dim=0),
        "times": times,
        "steps": steps,
    }


def _summary_stats(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().float().reshape(-1)
    if value.numel() == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(value.numel()),
        "mean": float(value.mean().item()),
        "median": float(value.median().item()),
        "p95": float(torch.quantile(value, 0.95).item()),
        "max": float(value.max().item()),
    }


def _compare_to_native(native_result: Mapping[str, Any], variant: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    indices, distances = _nearest_rows(
        variant["global_centers_object"], native_result["global_centers_object"]
    )
    native_endpoint = native_result["endpoint"].index_select(0, indices)
    endpoint_delta = (result["endpoint"] - native_endpoint).abs()
    cosine = F.cosine_similarity(result["endpoint"], native_endpoint, dim=1)
    velocity_rows: List[Dict[str, Any]] = []
    native_velocities = native_result["velocities"]
    variant_velocities = result["velocities"]
    for step in range(int(variant_velocities.shape[0])):
        native_velocity = native_velocities[step].index_select(0, indices)
        delta = (variant_velocities[step] - native_velocity).abs()
        velocity_rows.append(
            {
                "step": step,
                "mean_abs_delta": float(delta.mean().item()),
                "p95_abs_delta": float(torch.quantile(delta, 0.95).item()),
                "max_abs_delta": float(delta.max().item()),
                "mean_nearest_center_distance": float(distances.mean().item()),
            }
        )
    forced = variant["forced_mask"]
    nonforced = ~forced
    def part(mask: torch.Tensor) -> Dict[str, Any]:
        return {
            "rows": int(mask.sum().item()),
            "nearest_center_distance": _summary_stats(distances[mask]),
            "endpoint_abs_delta": _summary_stats(endpoint_delta[mask]),
            "endpoint_cosine": _summary_stats(1.0 - cosine[mask]),
        }
    return {
        "matched_rows": int(indices.shape[0]),
        "nearest_center_distance": _summary_stats(distances),
        "endpoint_abs_delta": _summary_stats(endpoint_delta),
        "endpoint_cosine_one_minus": _summary_stats(1.0 - cosine),
        "forced_part": part(forced),
        "nonforced_part": part(nonforced),
        "velocity_by_step": velocity_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = json.loads((root / "global_camera.json").read_text(encoding="utf-8"))
    camera = {key: float(camera[key]) for key in ("camera_angle_x", "distance", "mesh_scale")}
    payload26 = _load_payload(root, 26)
    payload27 = _load_payload(root, 27)
    boxes = {tile_id: tuple(int(v) for v in payload["box"]) for tile_id, payload in ((26, payload26), (27, payload27))}
    transform27 = core._derive_tile_camera(tile_id=27, box=boxes[27], global_camera=camera, extend_pixel=0)
    centers26 = payload26["global_centers_object"].to(torch.float32)
    centers27 = payload27["global_centers_object"].to(torch.float32)
    q26 = centers26 * (2.0 * camera["mesh_scale"])
    q27 = centers27 * (2.0 * camera["mesh_scale"])
    uv26, _, valid26 = core._project_global_q_to_4096(q26, global_camera=camera)
    uv27, _, valid27 = core._project_global_q_to_4096(q27, global_camera=camera)
    overlap_box = (
        max(boxes[26][0], boxes[27][0]),
        max(boxes[26][1], boxes[27][1]),
        min(boxes[26][2], boxes[27][2]),
        min(boxes[26][3], boxes[27][3]),
    )
    native27_overlap = _box_mask(uv27, valid27, overlap_box)
    inside27_from26 = _box_mask(uv26, valid26, boxes[27])
    q26_overlap = q26[_box_mask(uv26, valid26, overlap_box)]
    q26_inside27 = q26[inside27_from26]
    mapped_q27_overlap, _ = core._global_q_to_local_q(q26_overlap, global_camera=camera, transform=transform27)
    mapped_q27_all, _ = core._global_q_to_local_q(q26_inside27, global_camera=camera, transform=transform27)
    forced_overlap_global = q26_overlap / (2.0 * camera["mesh_scale"])
    forced_all_global = q26_inside27 / (2.0 * camera["mesh_scale"])

    variant_overlap = _build_variant(
        native=payload27,
        native_global=centers27,
        forced_global=forced_overlap_global,
        forced_local_q=mapped_q27_overlap,
        name="locked_overlap",
        replace_native_overlap=native27_overlap,
    )
    variant_all = _build_variant(
        native=payload27,
        native_global=centers27,
        forced_global=forced_all_global,
        forced_local_q=mapped_q27_all,
        name="locked_all_26_inside_27",
        replace_native_overlap=native27_overlap,
    )
    native_variant = {
        "name": "native_27",
        "coords": payload27["shape_coords"].to(torch.int32),
        "shape": payload27["shape_norm"].to(torch.float32),
        "global_centers_object": centers27,
        "forced_mask": torch.zeros((centers27.shape[0],), dtype=torch.bool),
        "support_tokens": int(centers27.shape[0]),
    }
    lookup = _spatial_noise(
        (centers27, variant_overlap["global_centers_object"], variant_all["global_centers_object"]),
        seed=int(args.seed),
        channels=CHANNELS,
    )
    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=True)
    image = Image.open(root / "tiles" / "tile_27" / "tile_reference.png").convert("RGB")
    params = dict(payload27["record"]["texture_flow"]["sampler"])
    params["steps"] = int(params.get("steps", 12))
    native_result = _run_texture_flow(
        pipeline=pipeline, variant=native_variant, image=image, transform=transform27,
        params=params, noise_lookup=lookup,
    )
    native_result["global_centers_object"] = centers27
    overlap_result = _run_texture_flow(
        pipeline=pipeline, variant=variant_overlap, image=image, transform=transform27,
        params=params, noise_lookup=lookup,
    )
    overlap_result["global_centers_object"] = variant_overlap["global_centers_object"]
    all_result = _run_texture_flow(
        pipeline=pipeline, variant=variant_all, image=image, transform=transform27,
        params=params, noise_lookup=lookup,
    )
    all_result["global_centers_object"] = variant_all["global_centers_object"]
    comparisons = {
        variant_overlap["name"]: _compare_to_native(native_result, variant_overlap, overlap_result),
        variant_all["name"]: _compare_to_native(native_result, variant_all, all_result),
    }
    support = {
        "tile_26_native_tokens": int(centers26.shape[0]),
        "tile_27_native_tokens": int(centers27.shape[0]),
        "tile_27_native_overlap_tokens": int(native27_overlap.sum().item()),
        "overlap_box_4096": list(overlap_box),
        "overlap_pixels": int(max(0, overlap_box[2] - overlap_box[0]) * max(0, overlap_box[3] - overlap_box[1])),
        "tile_26_centers_inside_27": int(inside27_from26.sum().item()),
        "mapped_26_overlap_local_q_rows": int(mapped_q27_overlap.shape[0]),
        "mapped_26_inside_27_local_q_rows": int(mapped_q27_all.shape[0]),
        "locked_overlap_variant": {key: value for key, value in variant_overlap.items() if key.endswith("tokens") or key == "support_tokens"},
        "locked_all_variant": {key: value for key, value in variant_all.items() if key.endswith("tokens") or key == "support_tokens"},
        "native_overlap_fraction": float(native27_overlap.float().mean().item()),
    }
    summary = {
        "format": "pixal3d_tile26_27_support_lock_flow_test_v1",
        "cuda_device": int(args.cuda_device),
        "camera": camera,
        "tile_layout": {"tile_26": list(boxes[26]), "tile_27": list(boxes[27]), "overlap": list(overlap_box)},
        "flow": {"stage": "texture1024", "params": params, "noise": "deterministic hash of continuous global center quantized at 4096"},
        "support": support,
        "comparisons_to_native_27": comparisons,
    }
    _atomic_json(output_dir / "support_lock_flow_summary.json", summary)
    torch.save(
        {
            "native_27": native_result,
            "locked_overlap": overlap_result,
            "locked_all_26_inside_27": all_result,
            "native_27_overlap_mask": native27_overlap,
            "variant_overlap": {key: value for key, value in variant_overlap.items() if isinstance(value, (torch.Tensor, str, int, bool))},
            "variant_all": {key: value for key, value in variant_all.items() if isinstance(value, (torch.Tensor, str, int, bool))},
        },
        output_dir / "support_lock_flow_states.pt",
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
