#!/usr/bin/env python3
"""Corrected Global-1024 -> C1024 -> C256 HR-tile velocity experiment.

The support route is fixed to:

    official Global-1024 baseline (Shape1024 on C64)
    -> learned shape-decoder subdivision on C1024
    -> fixed floor quantization to global C256

Fresh shape noise is generated once for the complete C256 support.  Absolute
global C256 rows are grouped only by their projection into the canonical 4096
image's 1024 crops (stride 512).  Each crop predicts a velocity on its gathered
rows; velocities for the same global row are averaged before the one global
Euler update.  Texture repeats this with fresh global texture noise and the
completed C256 shape state as its row-aligned concat condition.  There is no
local coordinate transform, local quantization, encoding, or latent transport.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from PIL import Image

import pixal3d_global_c64_hr_tile_condition_ablation as ablation
import pixal3d_projective_tile_generation_eval_projected_c64_only as base
import pixal3d_projective_tile_wavelet_fusion_c256 as c256_route
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_tile_encoded_query_noise_flow_overlap_render as render_helpers
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT_VERSION = "pixal3d_global1024_to_c256_hr_tile_velocity_average_v1"
GRID_GLOBAL = 256
DECODE_RESOLUTION = 4096


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fresh_global_noise(
    *,
    coords: torch.Tensor,
    channels: int,
    seed: int,
    device: torch.device,
) -> SparseTensor:
    if int(channels) <= 0:
        raise ValueError("noise channel count must be positive")
    noise = SparseTensor(
        feats=base._randn(
            int(coords.shape[0]),
            int(channels),
            device=device,
            seed=int(seed),
        ),
        coords=coords,
    )
    if not torch.equal(noise.coords, coords):
        raise RuntimeError("fresh global noise changed C256 support/order")
    return noise


@torch.no_grad()
def _prepare_global_condition_cpu(
    *,
    pipeline: Any,
    image_model: Any,
    image_1024: Image.Image,
    coords256: torch.Tensor,
    camera: Mapping[str, float],
    stage_name: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    condition = pipeline.get_proj_cond_shape(
        image_model,
        [image_1024],
        coords256,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_GLOBAL,
    )
    packed = pipeline._pack_proj_condition_cpu(
        condition,
        expected_coords=coords256,
        name=f"corrected_global_{stage_name}_C256_fallback",
    )
    del condition
    _empty_cuda_cache()
    return packed


def _mesh_stats(mesh: Any) -> Dict[str, Any]:
    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "ovoxels": int(mesh.coords.shape[0]),
        "decoded_coord_min": mesh.coords.amin(dim=0).detach().cpu().tolist(),
        "decoded_coord_max": mesh.coords.amax(dim=0).detach().cpu().tolist(),
        "vertex_bbox_min": mesh.vertices.amin(dim=0).detach().cpu().tolist(),
        "vertex_bbox_max": mesh.vertices.amax(dim=0).detach().cpu().tolist(),
        "voxel_size": float(torch.as_tensor(mesh.voxel_size).item()),
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    official = summary.get("official_global1024", {})
    support = summary.get("c256_support", {})
    evaluation = summary.get("evaluation", {})
    lines = [
        "# Corrected Global-1024 to C256 HR-tile velocity experiment",
        "",
        "- Support route: official Global-1024 Shape1024 C64 -> decoder C1024 -> fixed C256.",
        "- C256 noise: generated once globally and then gathered by projected tile rows.",
        "- Tile fusion: arithmetic mean of velocities for identical absolute global C256 rows.",
        "- Local coordinate transform / quantization / encoding / transport: none.",
        f"- Official Global-1024 C64 tokens: `{official.get('c64_tokens')}`.",
        f"- Subdivided C1024 rows: `{support.get('source_c1024_points')}`.",
        f"- Fixed C256 tokens: `{support.get('global_c256_tokens')}`.",
    ]
    decode = summary.get("c256_decode", {})
    if decode:
        lines.extend(
            [
                "",
                "## C256 native decode",
                "",
                f"- Resolution: `{decode.get('resolution')}`.",
                f"- Status: `{decode.get('status')}`.",
            ]
        )
        if decode.get("error"):
            lines.append(f"- Error: `{decode['error']}`.")
    baseline_metric = evaluation.get("baseline")
    experiment_metric = evaluation.get("experiment")
    if baseline_metric:
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                f"- Official Global-1024 PSNR: `{baseline_metric['psnr_db']:.6f} dB`.",
                f"- Official Global-1024 SSIM: `{baseline_metric['ssim']:.6f}`.",
            ]
        )
    if experiment_metric:
        lines.extend(
            [
                f"- Corrected C256 HR-tile PSNR: `{experiment_metric['psnr_db']:.6f} dB`.",
                f"- Delta: `{evaluation['psnr_delta_db']:+.6f} dB`.",
                f"- Corrected C256 HR-tile SSIM: `{experiment_metric['ssim']:.6f}`.",
                f"- SSIM delta: `{evaluation['ssim_delta']:+.6f}`.",
            ]
        )
    visual = summary.get("visual_analysis")
    if visual:
        lines.extend(
            [
                "",
                "## Visual analysis",
                "",
                f"- Input view: {visual['input_view']}",
                f"- Back view: {visual['back_view']}",
                f"- Back-view ground truth: `{visual['back_view_ground_truth']}`.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    if torch.cuda.current_device() != 4:
        raise RuntimeError("this corrected experiment must run on physical CUDA 4")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        f"[cuda] current={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    ablation._atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    camera = core._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        moge_model_path=args.moge_model_path,
    )
    ablation._atomic_json(output_dir / "global_camera.json", camera)
    ss_params, shape_params, texture_params = ablation._sampler_params(args)

    print("[official-baseline] unmodified pipeline.run(..., 1024_cascade)")
    baseline_started = time.perf_counter()
    torch.manual_seed(int(args.seed))
    baseline_output, baseline_latents = pipeline.run(
        image_1024,
        camera_params=dict(camera),
        seed=int(args.seed),
        sparse_structure_sampler_params=dict(ss_params),
        shape_slat_sampler_params=dict(shape_params),
        tex_slat_sampler_params=dict(texture_params),
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    if len(baseline_output) != 1:
        raise RuntimeError("official Global-1024 did not return exactly one mesh")
    baseline_mesh = core._validate_mesh(
        baseline_output[0], "corrected route official Global-1024 baseline"
    ).to("cpu")
    baseline_shape, baseline_texture, baseline_resolution = baseline_latents
    if int(baseline_resolution) != 1024:
        raise RuntimeError(f"official baseline resolution is {baseline_resolution}, not 1024")
    if not torch.equal(baseline_shape.coords, baseline_texture.coords):
        raise RuntimeError("official Global-1024 shape/texture supports differ")
    baseline_seconds = float(time.perf_counter() - baseline_started)
    print(
        f"[official-baseline] C64={baseline_shape.coords.shape[0]:,} "
        f"seconds={baseline_seconds:.3f}"
    )

    # This is the corrected support construction: subdivision happens only
    # after the complete Global-1024 Shape1024 latent exists.
    coords1024, subdivision_stats = base._learned_subdivide_shape1024_to_c1024(
        pipeline, baseline_shape
    )
    coords256, source_to_c256, quantization_stats = (
        c256_route._quantize_global_c1024_to_c256(coords1024)
    )
    if int(coords256.shape[0]) > int(args.max_num_tokens):
        raise RuntimeError(
            f"corrected C256 support has {coords256.shape[0]:,} tokens, exceeding "
            f"--max-num-tokens={int(args.max_num_tokens):,}"
        )
    print(
        f"[corrected-support] Global1024_C64={baseline_shape.coords.shape[0]:,} "
        f"decoder_C1024={coords1024.shape[0]:,} C256={coords256.shape[0]:,}"
    )
    ablation._atomic_torch_save(
        output_dir / "global1024_c64_to_c1024_to_c256_support.pt",
        {
            "format": FORMAT_VERSION,
            "official_global1024_coords_c64": baseline_shape.coords.detach().cpu(),
            "decoder_subdivided_coords_c1024": coords1024.detach().cpu(),
            "fixed_coords_c256": coords256.detach().cpu(),
            "c1024_source_to_c256": source_to_c256.detach().cpu(),
            "subdivision": subdivision_stats,
            "quantization": quantization_stats,
        },
    )
    del coords1024, source_to_c256, baseline_output, baseline_latents
    _empty_cuda_cache()

    projected_norm, projected_depth, projected_valid = (
        pipeline._project_sparse_coords_to_image_norm(
            image_cond_model=pipeline.image_cond_model_shape_1024,
            coords=coords256,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution=GRID_GLOBAL,
        )
    )
    tiles, tile_summary_live = ablation._build_global_row_tiles(
        image_4096=image_4096,
        projected_full_norm=projected_norm,
        projection_valid=projected_valid,
    )
    ablation._save_projection_overlay(
        image=image_4096,
        tile_summary=tile_summary_live,
        output_path=output_dir / "corrected_global_c256_projection_and_tiles.png",
    )
    tile_summary = {
        key: value
        for key, value in tile_summary_live.items()
        if key not in {"eligible", "coverage", "pixel_x", "pixel_y"}
    }
    print(
        f"[tile-rows] C256={coords256.shape[0]:,} "
        f"active={tile_summary['active_tile_count']}/49 "
        f"covered={tile_summary['covered_row_count']:,} "
        f"overlap={tile_summary['overlap_row_count']:,} "
        f"uncovered={tile_summary['uncovered_row_count']:,}"
    )

    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_noise_seed = int(args.seed) + int(args.shape_noise_seed_offset)
    shape_noise = _fresh_global_noise(
        coords=coords256,
        channels=int(shape_model.in_channels),
        seed=shape_noise_seed,
        device=device,
    )
    global_shape_condition_cpu = _prepare_global_condition_cpu(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_shape_1024,
        image_1024=image_1024,
        coords256=coords256,
        camera=camera,
        stage_name="shape",
    )
    if bool(pipeline.low_vram):
        shape_model.cpu()
    shape_condition_stats = ablation._prepare_tile_conditions(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_shape_1024,
        image_4096=image_4096,
        global_coords=coords256,
        tiles=tiles,
        camera=camera,
        stage_name="shape",
        grid_resolution=GRID_GLOBAL,
    )
    merged_shape_params = {**pipeline.shape_slat_sampler_params, **dict(shape_params)}
    shape_norm, shape_flow_stats = ablation._run_tiled_global_flow(
        pipeline=pipeline,
        model=shape_model,
        sampler=pipeline.shape_slat_sampler,
        initial_noise=shape_noise,
        global_condition_cpu=global_shape_condition_cpu,
        tiles=tiles,
        stage_name="shape",
        sampler_params=merged_shape_params,
        concat_cond=None,
        coordinate_label="C256",
    )
    shape_raw = ablation._denormalize(shape_norm, pipeline.shape_slat_normalization)
    del global_shape_condition_cpu
    _empty_cuda_cache()

    texture_projection, _, texture_projection_valid = (
        pipeline._project_sparse_coords_to_image_norm(
            image_cond_model=pipeline.image_cond_model_tex_1024,
            coords=coords256,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution=GRID_GLOBAL,
        )
    )
    projection_error = ablation._max_abs(projected_norm, texture_projection)
    projection_valid_equal = torch.equal(
        projected_valid.detach().cpu(), texture_projection_valid.detach().cpu()
    )
    if projection_error > 1e-7 or not projection_valid_equal:
        raise RuntimeError("shape/texture global C256 projections disagree")

    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(shape_norm.feats.shape[1])
    texture_noise_seed = int(args.seed) + int(args.texture_noise_seed_offset)
    texture_noise = _fresh_global_noise(
        coords=coords256,
        channels=texture_channels,
        seed=texture_noise_seed,
        device=device,
    )
    global_texture_condition_cpu = _prepare_global_condition_cpu(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_tex_1024,
        image_1024=image_1024,
        coords256=coords256,
        camera=camera,
        stage_name="texture",
    )
    if bool(pipeline.low_vram):
        texture_model.cpu()
    texture_condition_stats = ablation._prepare_tile_conditions(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_tex_1024,
        image_4096=image_4096,
        global_coords=coords256,
        tiles=tiles,
        camera=camera,
        stage_name="texture",
        grid_resolution=GRID_GLOBAL,
    )
    merged_texture_params = {
        **pipeline.tex_slat_sampler_params,
        **dict(texture_params),
    }
    texture_norm, texture_flow_stats = ablation._run_tiled_global_flow(
        pipeline=pipeline,
        model=texture_model,
        sampler=pipeline.tex_slat_sampler,
        initial_noise=texture_noise,
        global_condition_cpu=global_texture_condition_cpu,
        tiles=tiles,
        stage_name="texture",
        sampler_params=merged_texture_params,
        concat_cond=shape_norm,
        coordinate_label="C256",
    )
    texture_raw = ablation._denormalize(
        texture_norm, pipeline.tex_slat_normalization
    )
    del global_texture_condition_cpu
    _empty_cuda_cache()

    for name, value in {
        "shape_noise": shape_noise,
        "texture_noise": texture_noise,
        "shape": shape_raw,
        "texture": texture_raw,
    }.items():
        if not torch.equal(value.coords, coords256):
            raise RuntimeError(f"{name} changed corrected global C256 support/order")

    ablation._atomic_torch_save(
        output_dir / "corrected_global_c256_noise_and_latents.pt",
        {
            "format": FORMAT_VERSION,
            "grid_resolution": GRID_GLOBAL,
            "native_decode_resolution": DECODE_RESOLUTION,
            "coords256": coords256.detach().cpu(),
            "shape_noise_seed": shape_noise_seed,
            "shape_initial_noise": shape_noise.feats.detach().cpu(),
            "shape_norm": shape_norm.feats.detach().cpu(),
            "shape_raw": shape_raw.feats.detach().cpu(),
            "texture_noise_seed": texture_noise_seed,
            "texture_initial_noise": texture_noise.feats.detach().cpu(),
            "texture_norm": texture_norm.feats.detach().cpu(),
            "texture_raw": texture_raw.feats.detach().cpu(),
        },
    )
    ablation._atomic_torch_save(
        output_dir / "corrected_global_c256_tile_rows.pt",
        {
            "format": FORMAT_VERSION,
            "coords256": coords256.detach().cpu(),
            "projected_full_norm": projected_norm.detach().cpu(),
            "projected_depth": projected_depth.detach().cpu(),
            "projection_valid": projected_valid.detach().cpu(),
            "tiles": [
                {
                    "tile_id": tile["tile_id"],
                    "box": tile["box"],
                    "projection_crop_box": tile["projection_crop_box"],
                    "global_rows": tile["global_rows"],
                }
                for tile in tiles
            ],
        },
    )

    summary: Dict[str, Any] = {
        "format": FORMAT_VERSION,
        "status": "flows_complete_decode_pending",
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "cuda_device": 4,
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "seed": int(args.seed),
        "camera": camera,
        "official_global1024": {
            "pipeline_call": "unmodified pipeline.run(..., pipeline_type='1024_cascade')",
            "resolution": 1024,
            "c64_tokens": int(baseline_shape.coords.shape[0]),
            "seconds": baseline_seconds,
            "mesh": _mesh_stats(baseline_mesh),
        },
        "c256_support": {
            "route": "Global1024 Shape1024 C64 -> decoder subdivision C1024 -> fixed C256",
            **dict(subdivision_stats),
            **dict(quantization_stats),
        },
        "noise": {
            "shape": {
                "scope": "one complete global C256 tensor before any tile gather",
                "seed": shape_noise_seed,
                "sha256": ablation._tensor_sha256(shape_noise.feats),
            },
            "texture": {
                "scope": "one complete global C256 tensor before any tile gather",
                "seed": texture_noise_seed,
                "sha256": ablation._tensor_sha256(texture_noise.feats),
            },
        },
        "invariants": {
            "coordinates": "immutable absolute global C256 rows",
            "tile_operation": "projection membership then row gather only",
            "local_coordinate_transform": False,
            "requantization_after_c256_fix": False,
            "reencoding": False,
            "latent_transport": False,
            "shape_velocity_overlap": "arithmetic mean on identical global rows",
            "texture_velocity_overlap": "arithmetic mean on identical global rows",
            "global_state_updates_per_step": 1,
        },
        "tile_assignment": tile_summary,
        "conditions": {
            "shape": shape_condition_stats,
            "texture": texture_condition_stats,
            "shape_texture_projection_max_abs_error": projection_error,
            "shape_texture_projection_valid_equal": projection_valid_equal,
        },
        "shape_flow": shape_flow_stats,
        "texture_flow": texture_flow_stats,
        "seconds_before_decode": float(time.perf_counter() - started),
    }
    ablation._atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "EXPERIMENT_REPORT.md", summary)

    experimental_mesh: Optional[Any] = None
    decode_started = time.perf_counter()
    try:
        print("[decode] corrected global C256 -> native 4096 decoder")
        decoded = pipeline.decode_latent(shape_raw, texture_raw, DECODE_RESOLUTION)
        if len(decoded) != 1:
            raise RuntimeError(f"C256 decoder returned {len(decoded)} meshes")
        experimental_mesh = core._validate_mesh(
            decoded[0], "corrected global C256 HR-tile output"
        ).to("cpu")
        del decoded
        torch.save(
            experimental_mesh,
            output_dir / "corrected_global_c256_native4096_mesh.pt",
        )
        summary["c256_decode"] = {
            "status": "completed",
            "resolution": DECODE_RESOLUTION,
            "seconds": float(time.perf_counter() - decode_started),
            "mesh": _mesh_stats(experimental_mesh),
        }
    except Exception as error:
        summary["c256_decode"] = {
            "status": "failed",
            "resolution": DECODE_RESOLUTION,
            "seconds": float(time.perf_counter() - decode_started),
            "type": type(error).__name__,
            "error": str(error),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "latents_preserved": str(
                output_dir / "corrected_global_c256_noise_and_latents.pt"
            ),
        }
        for key in ("shape_slat_decoder", "tex_slat_decoder"):
            model = pipeline.models.get(key)
            if model is not None:
                model.cpu()
        _empty_cuda_cache()
        print(
            f"[decode-failed] {type(error).__name__}: {error}; "
            "C256 noise/latents remain saved"
        )

    torch.save(baseline_mesh, output_dir / "official_global1024_mesh.pt")
    envmap = load_envmap(str(args.envmap), device="cuda")
    reference = output_dir / "canonical_4096.png"
    baseline_metric = core._render(
        baseline_mesh,
        output_dir=output_dir / "official_global1024" / "aligned_eval",
        camera=camera,
        reference_image=reference,
        args=args,
        envmap=envmap,
    )
    baseline_multiview = (
        render_helpers._render_merged_mesh_multiview(
            baseline_mesh,
            output_dir=output_dir / "official_global1024" / "multiview",
            camera=camera,
            args=args,
            envmap=envmap,
        )
        if bool(args.render_multiview)
        else {"enabled": False}
    )
    evaluation: Dict[str, Any] = {
        "baseline": render_helpers._metric_subset(baseline_metric),
        "baseline_multiview": baseline_multiview,
        "experiment": None,
        "experiment_multiview": {"enabled": False},
        "psnr_delta_db": None,
    }

    if experimental_mesh is not None:
        experiment_metric = core._render(
            experimental_mesh,
            output_dir=output_dir / "corrected_c256_hr_tile" / "aligned_eval",
            camera=camera,
            reference_image=reference,
            args=args,
            envmap=envmap,
        )
        experiment_multiview = (
            render_helpers._render_merged_mesh_multiview(
                experimental_mesh,
                output_dir=output_dir / "corrected_c256_hr_tile" / "multiview",
                camera=camera,
                args=args,
                envmap=envmap,
            )
            if bool(args.render_multiview)
            else {"enabled": False}
        )
        input_comparison = output_dir / "input_view_textured_comparison.png"
        ablation._save_input_comparison(
            reference_path=reference,
            baseline_render_path=Path(baseline_metric["render_png"]),
            experiment_render_path=Path(experiment_metric["render_png"]),
            baseline_psnr=float(baseline_metric["psnr_db"]),
            experiment_psnr=float(experiment_metric["psnr_db"]),
            output_path=input_comparison,
            baseline_label="Official Global-1024",
            experiment_label="Global1024-derived C256 + HR tile velocity",
        )
        if bool(args.render_multiview):
            multiview_comparison = output_dir / "multiview_comparison.png"
            ablation._save_multiview_comparison(
                baseline=baseline_multiview,
                experiment=experiment_multiview,
                output_path=multiview_comparison,
                baseline_label="Official Global-1024",
                experiment_label="Global1024-derived C256 + HR tile velocity",
            )
        else:
            multiview_comparison = None
        evaluation.update(
            {
                "experiment": render_helpers._metric_subset(experiment_metric),
                "experiment_multiview": experiment_multiview,
                "psnr_delta_db": float(experiment_metric["psnr_db"])
                - float(baseline_metric["psnr_db"]),
                "ssim_delta": float(experiment_metric["ssim"])
                - float(baseline_metric["ssim"]),
                "input_comparison_png": str(input_comparison),
                "multiview_comparison_png": (
                    None if multiview_comparison is None else str(multiview_comparison)
                ),
            }
        )

    summary["evaluation"] = evaluation
    summary["status"] = (
        "completed" if experimental_mesh is not None else "flows_complete_decode_failed"
    )
    summary["seconds"] = float(time.perf_counter() - started)
    ablation._atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "EXPERIMENT_REPORT.md", summary)
    print(
        f"[done] status={summary['status']} "
        f"summary={output_dir / 'summary.json'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", default=str(Path(__file__).parent / "assets" / "choose" / "0_img.png")
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/global1024_to_c256_hr_tile_velocity_average/seed_42",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-noise-seed-offset", type=int, default=401)
    parser.add_argument("--texture-noise-seed-offset", type=int, default=501)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)

    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=4096)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--render-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=2)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument("--multiview-yaws-degrees", default="0,-45,45,-90,90,180")
    parser.add_argument("--multiview-pitches-degrees", default="0,0,0,0,0,0")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    if int(args.cuda_device) != 4:
        raise ValueError("this corrected experiment is fixed to physical CUDA 4")
    if int(args.shape_steps) != 12 or int(args.texture_steps) != 12:
        raise ValueError("shape and texture flows must remain 12 steps")
    if int(args.max_num_tokens) < 1:
        raise ValueError("--max-num-tokens must be positive")
    if int(args.shape_noise_seed_offset) == int(args.texture_noise_seed_offset):
        raise ValueError("shape and texture noise namespaces must be distinct")
    for value in (
        args.render_resolution,
        args.metric_resolution,
        args.render_ssaa,
        args.render_peel_layers,
        args.multiview_resolution,
    ):
        if int(value) < 1:
            raise ValueError("render settings must be positive")
    if not math.isfinite(float(args.multiview_radius_scale)):
        raise ValueError("multiview radius scale must be finite")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
