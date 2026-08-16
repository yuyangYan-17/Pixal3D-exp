#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 0: materialize missing official PureHR endpoints.

This command is deliberately separate from ``pixal3d_shared_coarse_oracle``.
It may call the repository's original ``_run_pure_hr_flow`` route, but it does
not run any MRA, donor consensus, guidance, or oracle calculation.  A
deterministic reproduction gate is mandatory before new endpoints are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import pixal3d_cross_tile_pbr_perstep as base
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline


PHASE_A_TILE_IDS = (18, 19, 20, 25, 26, 27, 32, 33, 34)
CANONICAL_TILE_COUNT = 49


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_sparse(path: Path) -> Any:
    return base._load_sparse_payload(path)


def _select_cuda(requested: int) -> Tuple[int, Optional[int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PureHR materialization")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    values = [int(v.strip()) for v in visible.split(",") if v.strip().lstrip("-").isdigit()] if visible else []
    if values and int(requested) in values:
        logical = values.index(int(requested))
        physical: Optional[int] = int(requested)
    else:
        logical = int(requested)
        physical = int(requested) if not values else None
    if logical < 0 or logical >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {requested} unavailable: visible={visible!r}, count={torch.cuda.device_count()}")
    torch.cuda.set_device(logical)
    return logical, physical


def _cross_args(args: argparse.Namespace, tile_ids: Sequence[int], *, resume: bool) -> SimpleNamespace:
    """Build only the fields consumed by the official context/flow helpers."""
    return SimpleNamespace(
        tile_ids=",".join(str(int(v)) for v in tile_ids),
        resume=bool(resume),
        low_vram=bool(args.low_vram),
        seed=int(args.seed),
        extend_pixel=0,
        face_projection_chunk_size=int(args.face_projection_chunk_size),
        material_query_chunk_size=65_536,
        material_face_chunk_size=16_384,
        query_chunk_size=int(args.query_chunk_size),
        roundtrip_tolerance=2e-5,
        noise_timestep=float(args.noise_timestep),
        noise_strength=float(args.noise_strength),
        max_num_tokens=1_000_000,
        ss_steps=12,
        ss_guidance_strength=7.5,
        ss_guidance_rescale=0.7,
        ss_rescale_t=5.0,
        shape_steps=12,
        shape_guidance_strength=7.5,
        shape_guidance_rescale=0.5,
        shape_rescale_t=3.0,
        texture_steps=int(args.texture_steps),
        texture_guidance_strength=float(args.texture_guidance_strength),
        texture_guidance_rescale=float(args.texture_guidance_rescale),
        texture_rescale_t=float(args.texture_rescale_t),
    )


def _load_cached_contexts(
    *,
    root: Path,
    pipeline: Any,
    tile_ids: Sequence[int],
    args: argparse.Namespace,
) -> Sequence[Any]:
    global_camera = json.loads((root / "global_camera.json").read_text(encoding="utf-8"))
    baseline = base._load_mesh(root / "global_baseline_mesh.pt")
    image_path = root / "canonical_4096.png"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        image_4096 = image.convert("RGB")
    cross_args = _cross_args(args, tile_ids, resume=True)
    boxes = core._tile_layout(canonical_size=4096, tile_size=1024, stride=512)
    # Resume mode only reads the fixed caches.  These three arguments are not
    # touched on that path and remain None intentionally.
    contexts = base._prepare_tile_contexts(
        args=cross_args,
        pipeline=pipeline,
        baseline_mesh=baseline,
        global_camera=global_camera,
        image_4096=image_4096,
        output_dir=root,
        global_attr_field=None,
        shape_encoder=None,
        pbr_encoder=None,
        boxes=boxes,
    )
    by_id = {int(context.tile_id): context for context in contexts}
    missing = [int(tile_id) for tile_id in tile_ids if int(tile_id) not in by_id]
    if missing:
        raise RuntimeError(f"fixed context cache is missing requested tiles: {missing}")
    for context in contexts:
        context.source_context_dir = str(root.resolve())
    return [by_id[int(tile_id)] for tile_id in tile_ids]


def _copy_gate_context(context: Any, gate_dir: Path) -> Any:
    gate_dir.mkdir(parents=True, exist_ok=True)
    context.tile_dir = gate_dir
    camera = gate_dir / "tile_camera.json"
    camera.write_text(json.dumps(dict(context.transform.__dict__), indent=2) + "\n", encoding="utf-8")
    return context


def _compare_sparse(old: Any, new: Any) -> Dict[str, Any]:
    if tuple(old.feats.shape) != tuple(new.feats.shape) or not torch.equal(old.coords, new.coords):
        return {
            "coords_exact": False,
            "feature_shape_exact": tuple(old.feats.shape) == tuple(new.feats.shape),
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
        }
    difference = (new.feats.to(torch.float64) - old.feats.to(torch.float64)).abs()
    return {
        "coords_exact": True,
        "feature_shape_exact": True,
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(difference).item() / (torch.linalg.vector_norm(old.feats.to(torch.float64)).item() + 1e-12)),
    }


@torch.no_grad()
def _decode_field(context: Any, endpoint: Any, pipeline: Any, chunk_size: int) -> torch.Tensor:
    shape_denorm = base._denormalize_slat(context.shape_norm, pipeline.shape_slat_normalization)
    shape_denorm = base._sparse_to_device(shape_denorm, torch.device("cuda"))
    endpoint = base._sparse_to_device(endpoint, torch.device("cuda"))
    _, field, _ = base._decode_endpoint(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_norm=endpoint,
        query_points=context.target_points.to("cuda"),
        query_chunk_size=int(chunk_size),
        label=f"PureHR reproduction tile {context.tile_id}",
    )
    return field.detach().cpu().to(torch.float32)


def _run_gate(
    *,
    reference_root: Path,
    gate_tile_id: int,
    gate_output: Path,
    pipeline: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    old_path = reference_root / "tiles" / f"tile_{gate_tile_id:02d}" / "pure_HR_endpoint.pt"
    if not old_path.is_file():
        raise FileNotFoundError(f"deterministic gate reference is missing: {old_path}")
    context = _load_cached_contexts(root=reference_root, pipeline=pipeline, tile_ids=[gate_tile_id], args=args)[0]
    old_endpoint = _load_sparse(old_path)
    gate_context = _copy_gate_context(context, gate_output / "gate" / f"tile_{gate_tile_id:02d}")
    flow_args = _cross_args(args, [gate_tile_id], resume=False)
    texture_params = core._sampler_overrides(flow_args)[2]
    flow_stats = base._run_pure_hr_flow(
        contexts=[gate_context],
        pipeline=pipeline,
        texture_params=texture_params,
        args=flow_args,
    )
    new_path = gate_context.tile_dir / "pure_HR_endpoint.pt"
    new_endpoint = _load_sparse(new_path)
    endpoint_comparison = _compare_sparse(old_endpoint, new_endpoint)
    old_field = _decode_field(context, old_endpoint, pipeline, args.query_chunk_size)
    new_field = _decode_field(context, new_endpoint, pipeline, args.query_chunk_size)
    field_diff = _compare_sparse(
        type("Field", (), {"coords": context.target_coords, "feats": old_field})(),
        type("Field", (), {"coords": context.target_coords, "feats": new_field})(),
    )
    result = {
        "status": "passed" if endpoint_comparison["coords_exact"] and endpoint_comparison["relative_l2"] is not None and endpoint_comparison["relative_l2"] < 1e-5 and field_diff["relative_l2"] is not None and field_diff["relative_l2"] < 1e-4 else "failed",
        "gate_tile_id": int(gate_tile_id),
        "reference_root": str(reference_root.resolve()),
        "reference_endpoint": str(old_path.resolve()),
        "reproduced_endpoint": str(new_path.resolve()),
        "flow": _jsonable(flow_stats),
        "endpoint": endpoint_comparison,
        "pbr_field": field_diff,
        "acceptance": {"endpoint_relative_l2_lt": 1e-5, "pbr_field_relative_l2_lt": 1e-4, "coords_exact": True},
    }
    _atomic_json(gate_output / "purehr_reproduction_gate.json", result)
    if result["status"] != "passed":
        raise RuntimeError(f"STOP: PureHR deterministic reproduction gate failed: {json.dumps(result, ensure_ascii=False)}")
    return result


def _write_tile_provenance(context: Any, endpoint_path: Path, output_dir: Path, args: argparse.Namespace, flow_stats: Mapping[str, Any]) -> Dict[str, Any]:
    transform = context.transform
    record = {
        "tile_id": int(context.tile_id),
        "canonical_box": [int(v) for v in context.box],
        "source_context_dir": str(getattr(context, "source_context_dir", context.tile_dir.resolve())),
        "initial_state_coord_digest": _digest_tensor(context.initial_state.coords),
        "initial_state_feature_digest": _digest_tensor(context.initial_state.feats),
        "fixed_shape_coord_digest": _digest_tensor(context.shape_norm.coords),
        "fixed_shape_feature_digest": _digest_tensor(context.shape_norm.feats),
        "model_path": str(args.model_path),
        "native_schedule": flow_stats.get("native_schedule"),
        "sampler_params": {
            "texture_steps": int(args.texture_steps),
            "texture_guidance_strength": float(args.texture_guidance_strength),
            "texture_guidance_rescale": float(args.texture_guidance_rescale),
            "texture_rescale_t": float(args.texture_rescale_t),
        },
        "noise_timestep": float(args.noise_timestep),
        "flow_steps": int(flow_stats.get("flow_steps", 0)),
        "route": "official pure HR",
        "pure_HR": {"route": "official pure HR", "flow_route": flow_stats.get("route")},
        "guidance_used": False,
        "MRA_used": False,
        "cross_tile_used": False,
        "endpoint": str(endpoint_path.resolve()),
        "tile_camera": dict(transform.__dict__),
    }
    _atomic_json(output_dir / "tiles" / f"tile_{context.tile_id:02d}" / "provenance.json", record)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, default=Path("outputs/cross_tile_pbr_perstep_guided_cuda4_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/purehr_completion_phase_a_cuda4"))
    parser.add_argument("--reference-root", type=Path, default=Path("outputs/cross_tile_pbr_perstep_guided_cuda4_pair_smoke"))
    parser.add_argument("--gate-tile-id", type=int, default=24)
    parser.add_argument("--tile-ids", type=str, default=",".join(str(v) for v in PHASE_A_TILE_IDS))
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument("--gate-texture-steps", type=int, default=1)
    parser.add_argument("--gate-texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--gate-texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--gate-texture-rescale-t", type=float, default=3.0)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--query-chunk-size", type=int, default=250_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    logical, physical = _select_cuda(int(args.cuda_device))
    print(f"[cuda] requested_physical={args.cuda_device} logical={logical} name={torch.cuda.get_device_name(logical)}")
    output_dir = args.output_dir.expanduser().resolve()
    context_dir = args.context_dir.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use one official pipeline for both the gate and missing endpoint runs;
    # the route and sampler parameters are still recorded separately.
    pipeline = init_pipeline(str(args.model_path), device="cuda", low_vram=bool(args.low_vram))
    gate_args = argparse.Namespace(**vars(args))
    gate_args.texture_steps = int(args.gate_texture_steps)
    gate_args.texture_guidance_strength = float(args.gate_texture_guidance_strength)
    gate_args.texture_guidance_rescale = float(args.gate_texture_guidance_rescale)
    gate_args.texture_rescale_t = float(args.gate_texture_rescale_t)
    gate = _run_gate(
        reference_root=reference_root,
        gate_tile_id=int(args.gate_tile_id),
        gate_output=output_dir,
        pipeline=pipeline,
        args=gate_args,
    )
    if bool(args.gate_only):
        return 0
    tile_ids = [int(v.strip()) for v in str(args.tile_ids).split(",") if v.strip()]
    contexts = _load_cached_contexts(root=context_dir, pipeline=pipeline, tile_ids=tile_ids, args=args)
    missing_contexts = []
    for context in contexts:
        endpoint = output_dir / "tiles" / f"tile_{context.tile_id:02d}" / "pure_HR_endpoint.pt"
        if bool(args.force) or not endpoint.is_file():
            missing_contexts.append(context)
    flow_stats: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    if missing_contexts:
        flow_args = _cross_args(args, [context.tile_id for context in missing_contexts], resume=False)
        for context in missing_contexts:
            out_tile = output_dir / "tiles" / f"tile_{context.tile_id:02d}"
            out_tile.mkdir(parents=True, exist_ok=True)
            context.tile_dir = out_tile
            (out_tile / "tile_camera.json").write_text(json.dumps(dict(context.transform.__dict__), indent=2) + "\n", encoding="utf-8")
        texture_params = core._sampler_overrides(flow_args)[2]
        flow_stats = base._run_pure_hr_flow(
            contexts=missing_contexts,
            pipeline=pipeline,
            texture_params=texture_params,
            args=flow_args,
        )
        for context in missing_contexts:
            endpoint = context.tile_dir / "pure_HR_endpoint.pt"
            provenance[str(context.tile_id)] = _write_tile_provenance(context, endpoint, output_dir, args, flow_stats)
    else:
        print("[purehr] all requested endpoints already materialized; no flow run")
    existing = sorted(int(path.parent.name.split("_")[-1]) for path in (output_dir / "tiles").glob("tile_*/pure_HR_endpoint.pt")) if (output_dir / "tiles").is_dir() else []
    _atomic_json(output_dir / "tile_preparation_summary.json", {
        "prepared_tile_ids": existing,
        "skipped_tile_ids": sorted(set(PHASE_A_TILE_IDS) - set(existing)),
        "active_tile_count": len(existing),
        "layout_tile_count": CANONICAL_TILE_COUNT,
        "source_context_dir": str(context_dir),
        "gate": str((output_dir / "purehr_reproduction_gate.json").resolve()),
    })
    _atomic_json(output_dir / "purehr_provenance.json", {
        "format": "pixal3d_purehr_materialization_v1",
        "cuda": {"requested_physical": int(args.cuda_device), "logical": int(logical), "physical": physical, "name": torch.cuda.get_device_name(logical)},
        "gate": gate,
        "tiles": provenance,
        "route": "official pure HR",
    })
    print(f"[purehr] materialized={len(missing_contexts)} existing={len(existing)} output={output_dir}")
    return 0


def main() -> None:
    raise SystemExit(run(_build_parser().parse_args()))


if __name__ == "__main__":
    main()
