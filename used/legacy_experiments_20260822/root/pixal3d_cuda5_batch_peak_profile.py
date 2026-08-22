#!/usr/bin/env python3
"""Profile real multi-view/tile sparse batches on a selected CUDA device.

This is a measurement harness for the fixed-geometry multi-view experiment.
It deliberately reuses the experiment's native flow, decoder, and direct PBR
encoder helpers.  It does not change the Jacobi flow or perform any output
fusion; the multi-context encode uses each decoded field as its own endpoint
only to isolate the encoder batch cost.  Flow, decoder, and encoder failures
are recorded independently so every barrier can use its own physical batch.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch

import pixal3d_cross_tile_pbr_perstep as cross_tile
import pixal3d_global_c4096_visible_local_flow as global_c4096
import pixal3d_multiview_fixed_geometry_pbr_gaussian_sr as experiment
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d.models as pixal3d_models
from inference import init_pipeline


GIB = float(2**30)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _memory() -> Dict[str, float]:
    torch.cuda.synchronize()
    return {
        "allocated_gib": float(torch.cuda.memory_allocated() / GIB),
        "reserved_gib": float(torch.cuda.memory_reserved() / GIB),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / GIB),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / GIB),
    }


def _measure(label: str, function: Any) -> tuple[Any, Dict[str, Any]]:
    torch.cuda.synchronize()
    before = _memory()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        value = function()
        torch.cuda.synchronize()
        after = _memory()
        record = {
            "label": label,
            "status": "ok",
            "seconds": float(time.perf_counter() - started),
            "before": before,
            "after": after,
            "peak": {
                "allocated_gib": after["peak_allocated_gib"],
                "reserved_gib": after["peak_reserved_gib"],
            },
            "peak_delta_allocated_gib": after["peak_allocated_gib"] - before["allocated_gib"],
            "peak_delta_reserved_gib": after["peak_reserved_gib"] - before["reserved_gib"],
        }
        return value, record
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize()
        after = _memory()
        record = {
            "label": label,
            "status": "oom",
            "seconds": float(time.perf_counter() - started),
            "before": before,
            "after": after,
            "peak": {
                "allocated_gib": after["peak_allocated_gib"],
                "reserved_gib": after["peak_reserved_gib"],
            },
            "peak_delta_allocated_gib": after["peak_allocated_gib"] - before["allocated_gib"],
            "peak_delta_reserved_gib": after["peak_reserved_gib"] - before["reserved_gib"],
            "error": str(exc),
        }
        _clear_cuda()
        return None, record
    except Exception as exc:
        # Numerical/decoder validity failures also make a batch unusable, but
        # must not suppress the independent encoder/flow capacity probes.
        try:
            torch.cuda.synchronize()
            after = _memory()
        except Exception:
            after = {
                "allocated_gib": float(torch.cuda.memory_allocated() / GIB),
                "reserved_gib": float(torch.cuda.memory_reserved() / GIB),
                "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / GIB),
                "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / GIB),
            }
        record = {
            "label": label,
            "status": "error",
            "seconds": float(time.perf_counter() - started),
            "before": before,
            "after": after,
            "peak": {
                "allocated_gib": after["peak_allocated_gib"],
                "reserved_gib": after["peak_reserved_gib"],
            },
            "peak_delta_allocated_gib": after["peak_allocated_gib"] - before["allocated_gib"],
            "peak_delta_reserved_gib": after["peak_reserved_gib"] - before["reserved_gib"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        _clear_cuda()
        return None, record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiview-image", default="test_pic/mask_compare_output/image2_resized.png")
    parser.add_argument("--baseline-dir", default="outputs/baseline1024_pbr_mesh_compare")
    parser.add_argument("--model-path", default=experiment.MODEL_PATH)
    parser.add_argument("--shape-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--pbr-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--selected-views", nargs="+", type=int, default=[0, 120, 240])
    parser.add_argument("--tile-ids", nargs="+", type=int, default=[24, 25, 26, 32, 33, 39])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 12, 16, 24, 32])
    parser.add_argument(
        "--context-build-batch-size",
        type=int,
        default=1,
        help="Safe batch used only to construct reusable contexts; phase batch sizes are swept independently.",
    )
    parser.add_argument(
        "--repeat-heaviest",
        type=int,
        default=0,
        help="Repeat the heaviest built context N times for a conservative worst-tile physical batch sweep.",
    )
    parser.add_argument(
        "--flow-only",
        action="store_true",
        help="Profile only the flow barrier; useful above the encoder/decoder OOM boundary.",
    )
    parser.add_argument("--output-dir", default="outputs/multiview_fixed_geometry_pbr_gaussian_cuda5_batch_profile")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _experiment_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    values = vars(experiment._parser().parse_args([])).copy()
    values.update({
        "multiview_image": args.multiview_image,
        "baseline_dir": args.baseline_dir,
        "model_path": args.model_path,
        "shape_encoder": args.shape_encoder,
        "pbr_encoder": args.pbr_encoder,
        "cuda_device": int(args.cuda_device),
        "selected_views": [int(value) for value in args.selected_views],
        "tile_ids": [int(value) for value in args.tile_ids],
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "flow_only": bool(args.flow_only),
        "debug": False,
        "render": False,
        "num_steps": 1,
        "low_vram": True,
    })
    return argparse.Namespace(**values)


def _groups(contexts: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    return experiment._flow_groups(contexts, int(batch_size))


def _profile_batch(contexts: Sequence[Any], pipeline: Any, pbr_encoder: Any, args: argparse.Namespace,
                   batch_size: int) -> Dict[str, Any]:
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {
        **pipeline.tex_slat_sampler_params,
        "steps": 1,
        "rescale_t": float(args.texture_rescale_t),
        "guidance_strength": float(args.texture_guidance_strength),
        "guidance_rescale": float(args.texture_guidance_rescale),
    }
    schedule = cross_tile._native_schedule(sampler, merged)
    start = cross_tile._schedule_start(schedule, float(args.noise_timestep))
    t = float(schedule[start])
    step_kwargs = cross_tile._sampler_step_kwargs(merged)
    states = {context.context_id: experiment._sparse_cpu(context.initial_state) for context in contexts}
    groups = list(_groups(contexts, int(batch_size)))
    result: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "contexts": len(contexts),
        "group_sizes": [len(group) for group in groups],
        "group_token_counts": [sum(int(context.texture_norm.feats.shape[0]) for context in group) for group in groups],
        "status": "ok",
    }
    model.to("cuda")
    try:
        def run_predictions() -> Dict[int, Dict[str, Any]]:
            result: Dict[int, Dict[str, Any]] = {}
            for group in groups:
                result.update(experiment._predict_flow_batch(group, states, model, sampler, t, step_kwargs))
            return result

        predictions, flow_record = _measure(f"batch_{batch_size}_flow", run_predictions)
        result["flow"] = flow_record
        if bool(args.flow_only):
            result["phase_status"] = {"flow": flow_record["status"]}
            result["status"] = "ok" if flow_record["status"] == "ok" else "flow_oom"
            result["total_measured_seconds"] = float(flow_record.get("seconds", 0.0))
            del predictions
            _clear_cuda()
            return result
        # The three barriers have different memory scaling and therefore must
        # be profiled independently.  A flow OOM must not suppress the decoder
        # or encoder measurement at the same requested batch size.
        endpoints = (
            {key: value["x0"] for key, value in predictions.items()}
            if predictions is not None
            else states
        )

        model.cpu()
        _clear_cuda()
        snapshots, decode_record = _measure(
            f"batch_{batch_size}_decode",
            lambda: experiment._decode_snapshots_batched(
                contexts,
                endpoints,
                pipeline,
                args,
                int(batch_size),
            ),
        )
        result["decode"] = decode_record
        # PBR re-encode memory depends on sparse support, not the field value.
        # Use an independent physical field so decoder failure cannot hide the
        # encoder limit and no decoded snapshot has to remain GPU-resident.
        fused_fields = {
            context.context_id: torch.full(
                (int(context.geometry.coords.shape[0]), 6),
                0.5,
                dtype=torch.float32,
            )
            for context in contexts
        }
        encode_predictions = (
            predictions
            if predictions is not None
            else {context.context_id: {"x0": states[context.context_id]} for context in contexts}
        )
        _, encode_record = _measure(
            f"batch_{batch_size}_pbr_encode",
            lambda: {
                context_id: endpoint
                for group in groups
                for endpoint_map in [experiment._encode_fused_batch(group, fused_fields, encode_predictions, pbr_encoder, pipeline)]
                for context_id, endpoint in endpoint_map.items()
            },
        )
        result["pbr_encode"] = encode_record
        result["phase_status"] = {
            "flow": flow_record["status"],
            "decode": decode_record["status"],
            "pbr_encode": encode_record["status"],
        }
        failed = [(name, status) for name, status in result["phase_status"].items() if status != "ok"]
        result["status"] = "ok" if not failed else "+".join(f"{name}_{status}" for name, status in failed)
        result["total_measured_seconds"] = sum(
            float(result[name].get("seconds", 0.0)) for name in ("flow", "decode", "pbr_encode") if name in result
        )
        del predictions, snapshots, fused_fields, endpoints, encode_predictions
        _clear_cuda()
        return result
    finally:
        model.cpu()
        pbr_encoder.cpu()
        _clear_cuda()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available() or int(args.cuda_device) >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device {args.cuda_device} is unavailable")
    if not args.selected_views or not args.tile_ids or not args.batch_sizes:
        raise ValueError("selected views, tile IDs, and batch sizes must be non-empty")
    if any(int(value) <= 0 for value in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if int(args.context_build_batch_size) <= 0:
        raise ValueError("context build batch size must be positive")
    if int(args.repeat_heaviest) < 0:
        raise ValueError("repeat heaviest must be non-negative")
    torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_args = _experiment_args(args, output_dir)
    source_path = Path(args.multiview_image).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    camera_path = baseline_dir / "global_camera.json"
    mesh_path = baseline_dir / "global_baseline_mesh.pt"
    current_camera_path = baseline_dir / "summary.json"
    current_mesh_path = baseline_dir / "raw_ovoxel_mesh.pt"
    legacy_baseline = camera_path.is_file() and mesh_path.is_file()
    current_baseline = current_camera_path.is_file() and current_mesh_path.is_file()
    if not source_path.is_file() or not (legacy_baseline or current_baseline):
        raise FileNotFoundError(
            f"missing profile input: source={source_path}, "
            f"legacy=({camera_path}, {mesh_path}), current=({current_camera_path}, {current_mesh_path})"
        )
    camera = (
        json.loads(camera_path.read_text(encoding="utf-8"))
        if legacy_baseline
        else global_c4096._load_camera(baseline_dir)
    )
    views_all = experiment._load_views(source_path, output_dir, experiment.ANGLES_DEFAULT)
    selected_views = {angle: views_all[angle] for angle in exp_args.selected_views}
    experiment._seed(int(args.seed))
    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=True)
    baseline = (
        cross_tile._load_mesh(mesh_path).to("cpu")
        if legacy_baseline
        else global_c4096._load_baseline(current_mesh_path).to("cpu")
    )
    baseline_attr = core._make_attribute_query_mesh(baseline, torch.device("cuda"))
    initial_profile_records: List[Dict[str, Any]] = []
    # Context construction is not the per-step encoder benchmark.  Build it
    # once with an explicitly safe batch so a large decoder/flow probe cannot
    # force repeated geometry voxelization after an unrelated initial OOM.
    actual_build_batch_size = int(args.context_build_batch_size)
    exp_args.flow_batch_size = actual_build_batch_size
    exp_args._initial_encode_profile_records = initial_profile_records
    contexts, _, _ = experiment._build_contexts(
        exp_args, pipeline, baseline, camera, selected_views, output_dir, baseline_attr
    )
    if not contexts:
        raise RuntimeError("profile tile selection produced no active contexts")
    # Put the heaviest supports first.  Every requested physical batch then
    # includes the worst available combination instead of an easy spatial
    # prefix, making the OOM boundary conservative for the complete run.
    contexts = sorted(
        contexts,
        key=lambda context: (
            int(context.geometry.coords.shape[0]),
            int(context.texture_norm.feats.shape[0]),
        ),
        reverse=True,
    )
    measured_source_contexts = len(contexts)
    if int(args.repeat_heaviest) > 0:
        contexts = [contexts[0]] * int(args.repeat_heaviest)
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder))).eval()

    context_rows = [{
        "context_id": int(context.context_id),
        "angle": int(context.angle),
        "tile_id": int(context.tile_id),
        "texture_tokens": int(context.texture_norm.feats.shape[0]),
        "local_ovoxels": int(context.geometry.coords.shape[0]),
    } for context in contexts]
    batch_runs: List[Dict[str, Any]] = []
    for batch_size in sorted({int(value) for value in args.batch_sizes}):
        # A repeated-worst sweep needs exactly one physical group.  Replaying
        # the same group until ``repeat_heaviest`` contexts are exhausted does
        # not change the peak and only multiplies runtime/autotuning noise.
        profile_contexts = (
            contexts[: int(batch_size)]
            if int(args.repeat_heaviest) > 0
            else contexts
        )
        run_record = _profile_batch(
            profile_contexts, pipeline, pbr_encoder, exp_args, batch_size
        )
        if run_record.get("total_measured_seconds") is not None and run_record["total_measured_seconds"] > 0:
            run_record["contexts_per_second"] = len(profile_contexts) / float(run_record["total_measured_seconds"])
            run_record["texture_tokens_per_second"] = sum(int(context.texture_norm.feats.shape[0]) for context in profile_contexts) / float(run_record["total_measured_seconds"])
        batch_runs.append(run_record)
        _write_json(output_dir / "batch_peak_profile_progress.json", {
            "format": "pixal3d_cuda5_batch_peak_profile_progress_v2",
            "completed_batch_sizes": [row["batch_size"] for row in batch_runs],
            "batch_runs": batch_runs,
        })

    pbr_encoder.cpu()
    del pbr_encoder, baseline_attr, baseline, pipeline
    _clear_cuda()
    summary = {
        "format": "pixal3d_cuda5_batch_peak_profile_v2",
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(int(args.cuda_device)),
        "input": str(source_path),
        "baseline_dir": str(baseline_dir),
        "selected_views": [int(value) for value in args.selected_views],
        "tile_ids": [int(value) for value in args.tile_ids],
        "contexts": context_rows,
        "measured_source_contexts": measured_source_contexts,
        "repeat_heaviest": int(args.repeat_heaviest),
        "initial_encode": {
            "records": initial_profile_records,
            "requested_build_batch_size": int(args.context_build_batch_size),
            "actual_build_batch_size": actual_build_batch_size,
        },
        "batch_runs": batch_runs,
        "algorithm": {
            "fixed_geometry": True,
            "source_tile": "256 crop -> 1024 resize",
            "real_sparse_batch": True,
            "single_batch_consistency_test": False,
            "serial_fallback": False,
            "flow_decode_encode_only": True,
            "initial_enc_profiled": True,
            "worst_context_repeated": bool(int(args.repeat_heaviest) > 0),
            "flow_only": bool(args.flow_only),
            "fusion_or_state_update": False,
        },
    }
    _write_json(output_dir / "batch_peak_profile.json", summary)
    return summary


def main() -> None:
    args = _parser().parse_args()
    summary = run(args)
    output_path = Path(args.output_dir).expanduser().resolve() / "batch_peak_profile.json"
    print(json.dumps({
        "output": str(output_path),
        "batch_status": [{"batch_size": row["batch_size"], "status": row["status"]} for row in summary["batch_runs"]],
    }, indent=2))


if __name__ == "__main__":
    main()
