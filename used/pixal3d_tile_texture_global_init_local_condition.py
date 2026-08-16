#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct-global-initialized local texture flow experiment.

This is a small follow-up to the fixed-shape local texture experiment.  The
local geometry and shape SLat support are kept unchanged.  The global
baseline PBR field is resampled onto each local support and encoded into a
local texture SLat (the support-compatible ``G_tex`` endpoint).  Its stage
mean/std are removed with the native Pixal3D texture-SLat normalization, and
that normalized result is passed directly to the native texture FlowEuler
sampler as the state at t=1.  No random texture noise or native endpoint
bridge is applied.

The texture condition remains the current LR tile image condition.  Thus the
experiment asks whether starting local flow from the global-derived local
texture endpoint improves the aligned input view and multiview result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as base
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


GROUP_NAME = "local_tile_condition_global_init"


def _clone_sparse(value: SparseTensor) -> SparseTensor:
    """Clone features while retaining native sparse spatial caches."""
    return value.replace(value.feats.detach().clone())


def _latent_stats(value: SparseTensor) -> Dict[str, Any]:
    features = value.feats.detach().to(torch.float32)
    flat = features.reshape(-1)
    return {
        "tokens": int(features.shape[0]),
        "channels": int(features.shape[1]),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "rms": float(torch.sqrt(torch.mean(flat.square())).item()),
    }


def _run_direct_initialized_texture_flow(
    *,
    pipeline: Any,
    fixed_shape_norm: SparseTensor,
    global_texture_norm: SparseTensor,
    condition: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    tile_id: int,
) -> tuple[SparseTensor, Dict[str, Any]]:
    """Flow from normalized global-derived local texture SLat at native t=1."""
    model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    merged_params = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    if not torch.equal(global_texture_norm.coords, fixed_shape_norm.coords):
        raise RuntimeError("global-derived texture and fixed shape supports differ")
    expected_channels = int(model.in_channels) - int(fixed_shape_norm.feats.shape[1])
    if int(global_texture_norm.feats.shape[1]) != expected_channels:
        raise RuntimeError(
            "global-derived texture channels do not match texture flow input: "
            f"flow={expected_channels} latent={global_texture_norm.feats.shape[1]}"
        )

    schedule = [
        float(v)
        for v in sampler.timestep_schedule(
            int(merged_params["steps"]), float(merged_params["rescale_t"])
        )
    ]
    if not schedule or abs(schedule[0] - 1.0) > 1e-6:
        raise RuntimeError(f"native texture schedule does not start at t=1: {schedule}")

    if pipeline.low_vram:
        model.to(torch.device(pipeline.device))
    started = time.perf_counter()
    try:
        # The input is already the normalized G_tex endpoint.  Passing it
        # directly to sample() makes the sampler treat it as x_t at t=1;
        # deliberately do not create or bridge through random noise.
        result = sampler.sample(
            model,
            _clone_sparse(global_texture_norm),
            cond=condition["cond"],
            neg_cond=condition["neg_cond"],
            concat_cond=fixed_shape_norm,
            **merged_params,
            verbose=True,
            tqdm_desc=f"Tile {tile_id:02d} direct-global-init local texture flow",
            record_trajectory=False,
            return_model_history=False,
        )
    finally:
        if pipeline.low_vram:
            model.cpu()
    base._sync_cuda()
    output = getattr(result, "samples", result)
    if not isinstance(output, SparseTensor):
        raise RuntimeError(f"texture flow returned {type(output)!r}, expected SparseTensor")
    if not torch.equal(output.coords, fixed_shape_norm.coords):
        raise RuntimeError("direct-initialized texture flow changed local support")
    return output, {
        "flow_seconds": float(time.perf_counter() - started),
        "flow_steps": int(merged_params["steps"]),
        "sampler": {
            "steps": int(merged_params["steps"]),
            "rescale_t": float(merged_params["rescale_t"]),
            "guidance_strength": float(merged_params.get("guidance_strength", 0.0)),
            "guidance_rescale": float(merged_params.get("guidance_rescale", 0.0)),
        },
        "native_timestep_schedule": schedule,
        "initial_timestep": 1.0,
        "initial_state": "global-baseline PBR field -> local PBR encoder -> normalized local texture SLat",
        "initial_state_is_global_derived": True,
        "initial_state_stats": _latent_stats(global_texture_norm),
        "stage_mean_removed": True,
        "stage_std_removed": True,
        "normalization": "pipeline.tex_slat_normalization: (x - mean) / std",
        "random_texture_noise": False,
        "native_noised_endpoint_bridge": False,
        "execution": "native FlowEulerSampler.sample directly from the normalized global-derived local state",
        "support_preserved": True,
    }


@torch.no_grad()
def _decode_to_patch(
    *,
    pipeline: Any,
    fixed_shape_denorm: SparseTensor,
    texture_denorm: SparseTensor,
    tile_id: int,
    box: Sequence[int],
    global_camera: Mapping[str, float],
    transform: base.TileCameraTransform,
    query_chunk_size: int,
) -> tuple[base.ReturnedTilePatch, MeshWithVoxel]:
    started = time.perf_counter()
    decoded = pipeline.decode_latent(
        fixed_shape_denorm,
        texture_denorm,
        base.OVOXEL_RESOLUTION,
    )
    base._sync_cuda()
    if len(decoded) != 1:
        raise RuntimeError("texture decoder returned more than one mesh")
    mesh = base._validate_mesh(decoded[0], f"tile {tile_id:02d} direct-init decode")
    patch = base._local_mesh_to_global_patch(
        tile_id=tile_id,
        box=box,
        local_mesh=mesh,
        global_camera=global_camera,
        transform=transform,
        query_chunk_size=query_chunk_size,
    )
    stats = {
        "decode_seconds": float(time.perf_counter() - started),
        "local_vertices": int(mesh.vertices.shape[0]),
        "local_faces": int(mesh.faces.shape[0]),
        "local_active_ovoxels": int(mesh.coords.shape[0]),
        "local_pbr_range": base._tensor_range(mesh.attrs),
        "global_patch": patch.stats,
    }
    return patch, mesh


def _save_input_comparison(
    *,
    canonical_path: Path,
    baseline_path: Path,
    group_path: Path,
    baseline_metrics: Mapping[str, Any],
    group_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    entries = [
        (canonical_path, "canonical input", None),
        (baseline_path, "ordinary global baseline", baseline_metrics),
        (group_path, GROUP_NAME, group_metrics),
    ]
    panel = 520
    header = 70
    canvas = Image.new("RGB", (panel * len(entries), panel + header), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (path, title, metrics) in enumerate(entries):
        if path.is_file():
            with Image.open(path) as image:
                image = ImageOps.contain(image.convert("RGB"), (panel - 8, panel - 8))
            canvas.paste(
                image,
                (
                    index * panel + (panel - image.width) // 2,
                    header + (panel - image.height) // 2,
                ),
            )
        draw.text((index * panel + 8, 8), title, fill=(255, 255, 255))
        if metrics:
            draw.text(
                (index * panel + 8, 34),
                f"PSNR {metrics.get('psnr_db')} SSIM {metrics.get('ssim')} LPIPS {metrics.get('lpips')}",
                fill=(220, 220, 220),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_multiview_overview(
    *,
    baseline_paths: Sequence[Path],
    group_paths: Sequence[Path],
    output_path: Path,
) -> None:
    count = min(len(baseline_paths), len(group_paths))
    panel = 360
    header = 44
    canvas = Image.new("RGB", (panel * 2, (panel + header) * count), "black")
    draw = ImageDraw.Draw(canvas)
    for row in range(count):
        for column, (title, paths) in enumerate(
            (("ordinary global baseline", baseline_paths), (GROUP_NAME, group_paths))
        ):
            path = paths[row]
            if path.is_file():
                with Image.open(path) as image:
                    image = ImageOps.contain(image.convert("RGB"), (panel - 4, panel - 4))
                canvas.paste(
                    image,
                    (
                        column * panel + (panel - image.width) // 2,
                        row * (panel + header) + header,
                    ),
                )
            draw.text(
                (column * panel + 5, row * (panel + header) + 10),
                title,
                fill=(255, 255, 255),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _mean_multiview_metrics(multiview: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(multiview.get("pair_metrics", []))
    if not rows:
        return {"views": 0, "psnr_db": None, "ssim": None}
    return {
        "views": int(len(rows)),
        "psnr_db": float(np.mean([row["baseline_vs_stitched_psnr_db"] for row in rows])),
        "ssim": float(np.mean([row["baseline_vs_stitched_ssim"] for row in rows])),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base_path = Path(encoder_path).expanduser()
        if not Path(f"{base_path}.json").is_file() or not Path(f"{base_path}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for base path {base_path}")
    if not args.skip_lpips and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips package unavailable; continuing without LPIPS")
        args.skip_lpips = True
    if args.render_multiview and int(args.multiview_turntable_frames) != 24:
        raise ValueError("this experiment expects 24 turntable frames for 30 views")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] requested/current index={int(args.cuda_device)} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.image).expanduser().resolve()
    with Image.open(source_path) as source:
        source_image = source.convert("RGB")
    source_image.save(output_dir / "input_original.png")

    pipeline = base.init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    canonical = pipeline.preprocess_canonical_images(source_image)
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    canonical["foreground_mask_4096"].save(output_dir / "canonical_foreground_mask_4096.png")
    base._atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    global_camera = base._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        moge_model_path=args.moge_model_path,
    )
    base._atomic_json(output_dir / "global_camera.json", global_camera)
    ss_params, shape_params, texture_params = base._sampler_overrides(args)

    print("[global-baseline] running ordinary Pixal3D 1024_cascade")
    base._seed_everything(int(args.seed))
    baseline_started = time.perf_counter()
    baseline_output, baseline_latents = pipeline.run(
        image_1024,
        camera_params=global_camera,
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    baseline_seconds = time.perf_counter() - baseline_started
    if len(baseline_output) != 1:
        raise RuntimeError(f"global baseline returned {len(baseline_output)} meshes")
    baseline_live = base._validate_mesh(baseline_output[0], "ordinary global baseline")
    baseline_shape_slat, baseline_texture_slat, decoded_resolution = baseline_latents
    if int(decoded_resolution) != int(base.OVOXEL_RESOLUTION):
        raise RuntimeError(f"baseline decoder resolution is {decoded_resolution}")

    envmap = (
        base.load_envmap(str(args.envmap), device="cuda")
        if (args.render or args.render_multiview)
        else None
    )
    baseline_dir = output_dir / "global_baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_render = (
        base._render(
            baseline_live,
            output_dir=baseline_dir / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        if args.render
        else None
    )
    baseline_mesh = baseline_live.to("cpu")
    baseline_summary = {
        "route": "ordinary pipeline.run with pipeline_type=1024_cascade",
        "generation_seconds": float(baseline_seconds),
        "decoder_resolution": int(decoded_resolution),
        "vertices": int(baseline_mesh.vertices.shape[0]),
        "faces": int(baseline_mesh.faces.shape[0]),
        "active_ovoxels": int(baseline_mesh.coords.shape[0]),
        "shape_slat_tokens": int(baseline_shape_slat.feats.shape[0]),
        "texture_slat_tokens": int(baseline_texture_slat.feats.shape[0]),
        "render": baseline_render,
    }
    base._atomic_json(baseline_dir / "summary.json", baseline_summary)
    del baseline_output, baseline_live, baseline_latents
    del baseline_shape_slat, baseline_texture_slat
    base._empty_cuda_cache()

    print("[global-analysis] building global PBR field and local encoders")
    face_min, face_max, face_finite = base._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    global_attr_field = base._make_attribute_query_mesh(baseline_mesh, device)
    shape_encoder = base.pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()
    pbr_encoder = base.pixal3d_models.from_pretrained(
        str(Path(args.pbr_encoder).expanduser())
    ).eval()
    if not args.low_vram:
        shape_encoder.to(device)
        pbr_encoder.to(device)

    boxes = base._tile_layout(stride=base.TILE_SIZE)
    requested_ids = base._parse_tile_ids(args.tile_ids)
    patches: List[base.ReturnedTilePatch] = []
    tile_records: List[Dict[str, Any]] = []
    attempted_tiles = 0

    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        if args.max_tiles is not None and attempted_tiles >= int(args.max_tiles):
            break
        attempted_tiles += 1
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image_lr = base._make_lr_tile_image(image_1024, box)
        tile_image_lr.save(tile_dir / "tile_lr_condition_reference.png")
        transform = base._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        base._atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
        selected_face_ids = base._tile_face_ids_from_bbox(
            face_min, face_max, face_finite, box
        )
        selected_face_count = int(selected_face_ids.shape[0])
        print(f"[tile {tile_id:02d}] bbox_faces={selected_face_count:,} box={box}")
        if selected_face_count == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "no triangle projection bbox intersects tile",
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            continue

        started = time.perf_counter()
        try:
            geometry = base._prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_min=face_min,
                global_face_max=face_max,
                global_face_finite=face_finite,
                global_camera=global_camera,
                transform=transform,
            )
            if geometry.stats["global_local_global_q_max_abs_error"] > float(args.roundtrip_tolerance):
                raise RuntimeError(
                    "global/local camera round-trip exceeded tolerance: "
                    f"{geometry.stats['global_local_global_q_max_abs_error']}"
                )
            local_attrs, material_stats = base._resample_local_attrs_from_global(
                geometry=geometry,
                global_attr_field=global_attr_field,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
                face_chunk_size=int(args.material_face_chunk_size),
            )
            shape_reference, shape_stats = base._encode_local_shape(
                encoder=shape_encoder,
                local_coords=geometry.coords,
                local_dual_vertices=geometry.dual_vertices,
                local_intersected=geometry.intersected,
                device=device,
                low_vram=bool(args.low_vram),
            )
            texture_reference, pbr_stats = base._encode_local_pbr(
                encoder=pbr_encoder,
                coords=geometry.coords,
                attrs=local_attrs,
                device=device,
                low_vram=bool(args.low_vram),
            )
            alignment_stats = base._latent_support_diagnostics(
                shape_reference, texture_reference
            )
            if not alignment_stats["coordinates_exactly_equal"]:
                raise RuntimeError(
                    "fixed shape/global-derived texture supports differ: "
                    + json.dumps(alignment_stats, ensure_ascii=False)
                )
            if int(shape_reference.feats.shape[0]) > int(args.max_num_tokens):
                raise RuntimeError(
                    f"local latent has {shape_reference.feats.shape[0]:,} tokens, "
                    f"exceeding --max-num-tokens={int(args.max_num_tokens):,}"
                )

            fixed_shape_norm = base._normalize_slat(
                shape_reference, pipeline.shape_slat_normalization
            )
            fixed_shape_denorm = base._denormalize_slat(
                fixed_shape_norm, pipeline.shape_slat_normalization
            )
            global_texture_norm = base._normalize_slat(
                texture_reference, pipeline.tex_slat_normalization
            )
            global_texture_denorm = base._denormalize_slat(
                global_texture_norm, pipeline.tex_slat_normalization
            )
            condition = pipeline.get_proj_cond_shape(
                pipeline.image_cond_model_tex_1024,
                [tile_image_lr.convert("RGB")],
                fixed_shape_norm.coords,
                camera_angle_x=float(transform.camera_angle_x),
                distance=float(transform.distance),
                mesh_scale=float(transform.mesh_scale),
                grid_resolution_override=int(base.LATENT_RESOLUTION),
            )
            if not torch.equal(
                condition["cond"]["proj"].coords.to(torch.int32),
                fixed_shape_norm.coords.to(torch.int32),
            ):
                raise RuntimeError("local tile condition changed fixed shape support order")

            texture_norm, flow_stats = _run_direct_initialized_texture_flow(
                pipeline=pipeline,
                fixed_shape_norm=fixed_shape_norm,
                global_texture_norm=global_texture_norm,
                condition=condition,
                texture_params=texture_params,
                tile_id=tile_id,
            )
            texture_denorm = base._denormalize_slat(
                texture_norm, pipeline.tex_slat_normalization
            )
            patch, decoded_mesh = _decode_to_patch(
                pipeline=pipeline,
                fixed_shape_denorm=fixed_shape_denorm,
                texture_denorm=texture_denorm,
                tile_id=tile_id,
                box=box,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
            )
            patches.append(patch)
            tile_record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "tile_seconds": float(time.perf_counter() - started),
                "projected_bbox_faces": int(selected_face_count),
                "geometry": geometry.stats,
                "material_resampling": material_stats,
                "shape_encoder": shape_stats,
                "pbr_encoder": pbr_stats,
                "latent_support": alignment_stats,
                "fixed_shape": {
                    "source": "global baseline mesh -> local dual grid -> shape encoder",
                    "tokens": int(fixed_shape_norm.feats.shape[0]),
                    "channels": int(fixed_shape_norm.feats.shape[1]),
                    "normalization": "shape_slat_normalization",
                    "used_as_texture_concat_cond": True,
                    "stats": _latent_stats(fixed_shape_norm),
                },
                "global_derived_texture_start": {
                    "source": "global baseline MeshWithVoxel PBR field -> local support resampling -> local PBR encoder",
                    "raw_stats": _latent_stats(texture_reference),
                    "normalized_stats": _latent_stats(global_texture_norm),
                    "denormalized_roundtrip_max_abs_error": float(
                        (global_texture_denorm.feats - texture_reference.feats)
                        .abs()
                        .max()
                        .item()
                    ),
                    "normalization": "tex_slat_normalization: (x - mean) / std",
                    "support_equal_to_fixed_shape": True,
                },
                "condition": {
                    "source": "current LR tile image condition",
                    "image": str(tile_dir / "tile_lr_condition_reference.png"),
                    "support_preserved": True,
                },
                "texture_flow": flow_stats,
                "texture_decode": {
                    "decoded_vertices": int(decoded_mesh.vertices.shape[0]),
                    "decoded_faces": int(decoded_mesh.faces.shape[0]),
                    "returned_patch": patch.stats,
                },
                "fixed_shape_unchanged": True,
                "random_texture_noise": False,
                "condition_is_local_tile_image": True,
            }
            tile_records.append(tile_record)
            base._write_tile_summary(tile_dir, tile_record)
            print(
                f"[tile {tile_id:02d}] success direct-init "
                f"tokens={fixed_shape_norm.feats.shape[0]:,} "
                f"faces={patch.faces.shape[0]:,} "
                f"seconds={tile_record['tile_seconds']:.2f}"
            )
            del (
                geometry,
                local_attrs,
                shape_reference,
                texture_reference,
                fixed_shape_norm,
                fixed_shape_denorm,
                global_texture_norm,
                global_texture_denorm,
                condition,
                texture_norm,
                texture_denorm,
                decoded_mesh,
                patch,
            )
            base._empty_cuda_cache()
        except Exception as exc:
            traceback.print_exc()
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "projected_bbox_faces": int(selected_face_count),
                "error": f"{type(exc).__name__}: {exc}",
                "tile_seconds": float(time.perf_counter() - started),
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            print(f"[tile {tile_id:02d}] failed: {type(exc).__name__}: {exc}")
            base._empty_cuda_cache()

    successful_tiles = [row for row in tile_records if row["status"] == "success"]
    failed_tiles = [row for row in tile_records if row["status"] == "failed"]
    skipped_tiles = [row for row in tile_records if row["status"] == "skipped"]
    if not patches:
        raise RuntimeError("no successful direct-initialized local texture tiles")

    group_dir = output_dir / GROUP_NAME
    group_dir.mkdir(parents=True, exist_ok=True)
    if len(boxes) == 16 and base.TILE_STRIDE == base.TILE_SIZE:
        stitched_mesh, stitch_stats = base._stitch_tile_patches(
            patches,
            layout=baseline_mesh.layout,
        )
        stitch_stats["layout_policy"] = "4x4 disjoint tiles; direct concat without overlap owner/weld"
    else:
        stitched_mesh, stitch_stats = base._stitch_tile_patches_nearest(
            patches,
            layout=baseline_mesh.layout,
            global_camera=global_camera,
            face_chunk_size=int(args.face_projection_chunk_size),
            weld_tolerance=float(args.stitch_tolerance),
        )
    stitched_patch = base.ReturnedTilePatch(
        tile_id=-1,
        box=(0, 0, int(base.CANONICAL_IMAGE_SIZE), int(base.CANONICAL_IMAGE_SIZE)),
        vertices=stitched_mesh.vertices,
        faces=stitched_mesh.faces,
        vertex_attrs=stitched_mesh.vertex_attrs,
        stats=stitch_stats,
    )
    glb_stats = (
        base._export_tiled_glb(
            [stitched_patch],
            group_dir / f"{GROUP_NAME}.glb",
        )
        if args.export_glb
        else {"enabled": False}
    )
    overlap_stats = base._save_tile_overlap_visualization(
        image_4096=image_4096,
        boxes=boxes,
        successful_ids=[patch.tile_id for patch in patches],
        output_path=group_dir / "tile_overlap_coverage.png",
    )

    render_stats: Dict[str, Any] = {
        "enabled": False,
        "overlap_visualization": overlap_stats,
    }
    group_render_path: Optional[Path] = None
    if args.render:
        aligned = base._render(
            stitched_mesh,
            output_dir=group_dir / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        against_baseline = (
            base._render(
                stitched_mesh,
                output_dir=group_dir / "against_global_baseline",
                camera=global_camera,
                reference_image=Path(str(baseline_render["render_png"])),
                args=args,
                envmap=envmap,
            )
            if baseline_render is not None
            else None
        )
        render_stats.update(
            {
                "aligned": aligned,
                "against_global_baseline": against_baseline,
                "input_metrics": base._metric_subset(aligned),
                "baseline_metrics": base._metric_subset(against_baseline),
            }
        )
        group_render_path = Path(str(aligned["render_png"]))

    multiview = (
        base._render_multiview_comparison(
            baseline_mesh,
            stitched_mesh,
            output_dir=group_dir / "multiview",
            camera=global_camera,
            args=args,
            envmap=envmap,
        )
        if args.render_multiview
        else {"enabled": False}
    )
    group_summary = {
        "status": "success" if not failed_tiles else "success_with_tile_failures",
        "successful_tiles": int(len(successful_tiles)),
        "failed_tiles": int(len(failed_tiles)),
        "skipped_tiles": int(len(skipped_tiles)),
        "stitch": stitch_stats,
        "glb": glb_stats,
        "render": render_stats,
        "multiview": multiview,
        "mean_multiview_vs_global_baseline": _mean_multiview_metrics(multiview),
        "condition": "current LR tile image projected condition",
        "initialization": {
            "global_result": "global baseline PBR field resampled to local support and re-encoded as local texture SLat",
            "normalization": "native tex_slat_normalization removes channel mean/std",
            "flow_start": "normalized global-derived local texture SLat passed directly at native t=1",
            "random_texture_noise": False,
            "native_noised_endpoint_bridge": False,
            "shape_flow_used": False,
            "fixed_shape": True,
        },
    }
    base._atomic_json(group_dir / "summary.json", group_summary)

    if args.render and baseline_render is not None and group_render_path is not None:
        _save_input_comparison(
            canonical_path=output_dir / "canonical_1024.png",
            baseline_path=Path(str(baseline_render["render_png"])),
            group_path=group_render_path,
            baseline_metrics=base._metric_subset(baseline_render),
            group_metrics=render_stats["input_metrics"],
            output_path=output_dir / "comparison_input_baseline_global_init.png",
        )
    if args.render_multiview and multiview.get("enabled"):
        _save_multiview_overview(
            baseline_paths=[Path(v) for v in multiview["baseline_frame_pngs"]],
            group_paths=[Path(v) for v in multiview["stitched_local_frame_pngs"]],
            output_path=output_dir / "comparison_multiview_baseline_global_init.png",
        )

    summary = {
        "format": "pixal3d_local_texture_global_derived_direct_init_v1",
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "global_camera": global_camera,
        "protocol": {
            "fixed_shape": True,
            "shape_source": "global baseline mesh -> local dual grid -> shape encoder",
            "shape_flow_used": False,
            "texture_condition": "local LR tile image condition",
            "texture_start": "global baseline PBR field -> local PBR encoder -> tex normalization -> direct t=1 state",
            "global_result_reencoded_to_local_support": True,
            "mean_removed": True,
            "std_removed": True,
            "random_texture_noise": False,
            "native_noised_endpoint_bridge": False,
            "same_sampler": True,
            "same_decoder": True,
            "same_local_to_global": True,
            "same_stitcher": "existing fixed-shape direct-concat policy for disjoint 4x4 tiles",
            "multiview_policy": "six fixed views plus 24 turntable frames, total 30",
        },
        "global_baseline_1024": baseline_summary,
        "successful_tiles": int(len(successful_tiles)),
        "failed_tiles": int(len(failed_tiles)),
        "skipped_tiles": int(len(skipped_tiles)),
        "groups": {GROUP_NAME: group_summary},
        "tiles": tile_records,
    }
    base._atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] success={len(successful_tiles)} failed={len(failed_tiles)} "
        f"skipped={len(skipped_tiles)} summary={output_dir / 'summary.json'}"
    )


def main() -> None:
    args = base.build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
