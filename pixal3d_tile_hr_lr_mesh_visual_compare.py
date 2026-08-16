#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare complete HR/LR local tile flow meshes with identical support/noise.

This is a follow-up visual experiment for
``pixal3d_tile_c1024_local_slat_and_local_decode_return_global.py``.  It keeps
the normal HR route intact and adds a second, complete LR route per tile:

    global baseline mesh
      -> local projected C1024 dual grid
      -> shape/PBR encoders on the exact same support
      -> identical shape/texture initial noise
      -> HR condition flow and LR condition flow
      -> local decode
      -> exact local-to-global face-corner conversion
      -> nearest tile-center ownership (half of each overlap)
      -> global rendering and image metrics

The HR and LR texture flows use their branch's generated shape SLat as
``concat_cond``; this is an end-to-end route comparison, so shape-condition
differences are intentionally propagated into texture.  The shape-model
condition difference is only the image condition: HR is the 1024 crop from
canonical 4096, while LR is the matching 256 crop from canonical 1024 resized
to 1024.  No fusion, mask, CCA, or new generation method is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as base
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


@contextmanager
def _fork_rng(device: torch.device):
    """Keep the extra LR branch from changing the normal HR RNG trajectory."""
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield


def _clone_sparse(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().clone(), value.coords)


def _sample_flow(
    *,
    pipeline: Any,
    sampler: Any,
    model: torch.nn.Module,
    noise: SparseTensor,
    condition: Mapping[str, Any],
    sampler_params: Mapping[str, Any],
    concat_cond: Optional[SparseTensor],
    description: str,
) -> SparseTensor:
    """Run one unchanged native Pixal3D Euler/CFG sampling call."""
    device = torch.device(pipeline.device)
    if pipeline.low_vram:
        model.to(device)
    kwargs = dict(sampler_params)
    if concat_cond is not None:
        kwargs["concat_cond"] = concat_cond
    try:
        result = sampler.sample(
            model,
            noise,
            cond=condition["cond"],
            neg_cond=condition["neg_cond"],
            **kwargs,
            verbose=True,
            tqdm_desc=description,
            record_trajectory=False,
            return_model_history=False,
        )
    finally:
        if pipeline.low_vram:
            model.cpu()
    samples = getattr(result, "samples", result)
    if not isinstance(samples, SparseTensor):
        raise RuntimeError(f"{description} returned {type(samples)!r}, expected SparseTensor")
    return samples


def _condition(
    pipeline: Any,
    model: torch.nn.Module,
    image: Image.Image,
    coords: torch.Tensor,
    transform: base.TileCameraTransform,
) -> Dict[str, Any]:
    return pipeline.get_proj_cond_shape(
        model,
        [image.convert("RGB")],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=base.LATENT_RESOLUTION,
    )


def _run_pair_flow(
    *,
    pipeline: Any,
    tile_image_hr: Image.Image,
    tile_image_lr: Image.Image,
    tile_box_4096: Sequence[int],
    transform: base.TileCameraTransform,
    shape_reference: SparseTensor,
    texture_reference: SparseTensor,
    shape_params: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    seed: int,
    tile_id: int,
) -> Dict[str, Any]:
    """Run HR/LR shape+texture flows with exactly shared support and noise."""
    device = torch.device(pipeline.device)
    shape_reference = shape_reference.to(device)
    texture_reference = texture_reference.to(device)
    if not torch.equal(shape_reference.coords, texture_reference.coords):
        raise RuntimeError("shape/PBR reference support differs before paired flow")
    coords = shape_reference.coords.to(torch.int32)

    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    merged_shape_params = {**pipeline.shape_slat_sampler_params, **dict(shape_params)}
    merged_texture_params = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    shape_steps = int(merged_shape_params["steps"])
    texture_steps = int(merged_texture_params["steps"])
    shape_times = pipeline.shape_slat_sampler.timestep_schedule(
        shape_steps, float(merged_shape_params["rescale_t"])
    )
    texture_times = pipeline.tex_slat_sampler.timestep_schedule(
        texture_steps, float(merged_texture_params["rescale_t"])
    )

    # Conditions are computed before noise creation, matching the existing HR
    # route.  LR condition creation is isolated so it cannot perturb HR RNG.
    shape_condition_hr = _condition(
        pipeline, pipeline.image_cond_model_shape_1024, tile_image_hr, coords, transform
    )
    with _fork_rng(device):
        shape_condition_lr = _condition(
            pipeline, pipeline.image_cond_model_shape_1024, tile_image_lr, coords, transform
        )

    base._seed_everything(int(seed))
    shape_clean = base._normalize_slat(shape_reference, pipeline.shape_slat_normalization)
    if int(shape_model.in_channels) != int(shape_clean.feats.shape[1]):
        raise RuntimeError(
            "shape flow channels do not match reference: "
            f"flow={shape_model.in_channels} ref={shape_clean.feats.shape[1]}"
        )
    shape_noise = SparseTensor(
        torch.randn(
            coords.shape[0], int(shape_model.in_channels), device=device, dtype=torch.float32
        ),
        coords,
    )
    shape_noised = base._native_noised_endpoint(
        shape_clean,
        shape_noise,
        pipeline.shape_slat_sampler,
        shape_times[0],
    )
    shape_noised_hr = _clone_sparse(shape_noised)
    shape_noised_lr = _clone_sparse(shape_noised)
    shape_noise_max_abs_diff = float(
        (shape_noised_hr.feats - shape_noised_lr.feats).abs().max().item()
    )

    hr_shape_started = time.perf_counter()
    hr_shape_norm = _sample_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=shape_model,
        noise=shape_noised_hr,
        condition=shape_condition_hr,
        sampler_params=merged_shape_params,
        concat_cond=None,
        description=f"Tile {tile_id:02d} HR shape SLat flow",
    )
    base._sync_cuda()
    hr_shape_seconds = time.perf_counter() - hr_shape_started
    if not torch.equal(hr_shape_norm.coords, coords):
        raise RuntimeError("HR shape flow changed local support")

    # The LR call sees byte-for-byte cloned x_t at the first step.  Its RNG
    # state is forked so this extra branch cannot affect subsequent HR noise.
    with _fork_rng(device):
        lr_shape_started = time.perf_counter()
        lr_shape_norm = _sample_flow(
            pipeline=pipeline,
            sampler=pipeline.shape_slat_sampler,
            model=shape_model,
            noise=shape_noised_lr,
            condition=shape_condition_lr,
            sampler_params=merged_shape_params,
            concat_cond=None,
            description=f"Tile {tile_id:02d} LR shape SLat flow",
        )
        base._sync_cuda()
        lr_shape_seconds = time.perf_counter() - lr_shape_started
    if not torch.equal(lr_shape_norm.coords, coords):
        raise RuntimeError("LR shape flow changed local support")

    shape_cond_hr = base._normalize_slat(
        base._denormalize_slat(hr_shape_norm, pipeline.shape_slat_normalization),
        pipeline.shape_slat_normalization,
    )
    shape_cond_lr = base._normalize_slat(
        base._denormalize_slat(lr_shape_norm, pipeline.shape_slat_normalization),
        pipeline.shape_slat_normalization,
    )
    if not torch.equal(shape_cond_hr.coords, shape_cond_lr.coords):
        raise RuntimeError("HR/LR generated shape support differs")

    texture_condition_hr = _condition(
        pipeline, pipeline.image_cond_model_tex_1024, tile_image_hr, coords, transform
    )
    with _fork_rng(device):
        texture_condition_lr = _condition(
            pipeline, pipeline.image_cond_model_tex_1024, tile_image_lr, coords, transform
        )

    texture_channels = int(texture_model.in_channels) - int(shape_cond_hr.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture channel count {texture_channels}")
    texture_clean = base._normalize_slat(texture_reference, pipeline.tex_slat_normalization)
    if int(texture_clean.feats.shape[1]) != texture_channels:
        raise RuntimeError(
            "texture flow channels do not match reference: "
            f"flow_noise={texture_channels} ref={texture_clean.feats.shape[1]}"
        )
    texture_noise = SparseTensor(
        torch.randn(coords.shape[0], texture_channels, device=device, dtype=torch.float32),
        coords,
    )
    texture_noised = base._native_noised_endpoint(
        texture_clean,
        texture_noise,
        pipeline.tex_slat_sampler,
        texture_times[0],
    )
    texture_noised_hr = _clone_sparse(texture_noised)
    texture_noised_lr = _clone_sparse(texture_noised)
    texture_noise_max_abs_diff = float(
        (texture_noised_hr.feats - texture_noised_lr.feats).abs().max().item()
    )

    hr_texture_started = time.perf_counter()
    hr_texture_norm = _sample_flow(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        noise=texture_noised_hr,
        condition=texture_condition_hr,
        sampler_params=merged_texture_params,
        concat_cond=shape_cond_hr,
        description=f"Tile {tile_id:02d} HR texture SLat flow",
    )
    base._sync_cuda()
    hr_texture_seconds = time.perf_counter() - hr_texture_started
    if not torch.equal(hr_texture_norm.coords, coords):
        raise RuntimeError("HR texture flow changed local support")

    with _fork_rng(device):
        lr_texture_started = time.perf_counter()
        lr_texture_norm = _sample_flow(
            pipeline=pipeline,
            sampler=pipeline.tex_slat_sampler,
            model=texture_model,
            noise=texture_noised_lr,
            condition=texture_condition_lr,
            sampler_params=merged_texture_params,
            concat_cond=shape_cond_lr,
            description=f"Tile {tile_id:02d} LR texture SLat flow",
        )
        base._sync_cuda()
        lr_texture_seconds = time.perf_counter() - lr_texture_started
    if not torch.equal(lr_texture_norm.coords, coords):
        raise RuntimeError("LR texture flow changed local support")

    result = {
        "hr_shape_denorm": base._denormalize_slat(
            hr_shape_norm, pipeline.shape_slat_normalization
        ),
        "lr_shape_denorm": base._denormalize_slat(
            lr_shape_norm, pipeline.shape_slat_normalization
        ),
        "hr_texture_denorm": base._denormalize_slat(
            hr_texture_norm, pipeline.tex_slat_normalization
        ),
        "lr_texture_denorm": base._denormalize_slat(
            lr_texture_norm, pipeline.tex_slat_normalization
        ),
        "stats": {
            "tile_id": int(tile_id),
            "seed": int(seed),
            "canonical_4096_tile_box": [int(value) for value in tile_box_4096],
            "canonical_1024_lr_crop_box": [int(value // 4) for value in tile_box_4096],
            "support": {
                "shape_reference_tokens": int(shape_reference.feats.shape[0]),
                "texture_reference_tokens": int(texture_reference.feats.shape[0]),
                "shape_texture_reference_coords_equal": bool(
                    torch.equal(shape_reference.coords, texture_reference.coords)
                ),
                "hr_lr_shape_coords_equal": bool(
                    torch.equal(hr_shape_norm.coords, lr_shape_norm.coords)
                ),
                "hr_lr_texture_coords_equal": bool(
                    torch.equal(hr_texture_norm.coords, lr_texture_norm.coords)
                ),
            },
            "initial_noise": {
                "shape_hr_lr_max_abs_diff": shape_noise_max_abs_diff,
                "texture_hr_lr_max_abs_diff": texture_noise_max_abs_diff,
                "shape_noise_range": base._tensor_range(shape_noise.feats),
                "texture_noise_range": base._tensor_range(texture_noise.feats),
                "noise_timestep": {
                    "shape": float(shape_times[0]),
                    "texture": float(texture_times[0]),
                },
            },
            "flow_seconds": {
                "hr_shape": float(hr_shape_seconds),
                "lr_shape": float(lr_shape_seconds),
                "hr_texture": float(hr_texture_seconds),
                "lr_texture": float(lr_texture_seconds),
            },
            "sampler": {
                "shape": dict(merged_shape_params),
                "texture": dict(merged_texture_params),
                "shape_timestep_schedule": [float(value) for value in shape_times],
                "texture_timestep_schedule": [float(value) for value in texture_times],
            },
            "condition": {
                "hr": "canonical 4096 crop 1024x1024",
                "lr": "canonical 1024 matching 256x256 crop resized to 1024x1024",
                "shape_model_only_image_condition_differs": True,
                "texture_model_image_condition_differs": True,
                "texture_hr_concat_cond": "HR generated normalized shape SLat",
                "texture_lr_concat_cond": "LR generated normalized shape SLat",
                "texture_concat_cond_is_branch_specific": True,
                "local_camera": {
                    "camera_angle_x": float(transform.camera_angle_x),
                    "distance": float(transform.distance),
                    "mesh_scale": float(transform.mesh_scale),
                },
            },
        },
    }
    del (
        shape_condition_hr,
        shape_condition_lr,
        texture_condition_hr,
        texture_condition_lr,
        shape_clean,
        shape_noise,
        shape_noised,
        texture_clean,
        texture_noise,
        texture_noised,
        texture_noised_hr,
        texture_noised_lr,
        shape_cond_hr,
        shape_cond_lr,
        hr_shape_norm,
        lr_shape_norm,
        hr_texture_norm,
        lr_texture_norm,
    )
    base._empty_cuda_cache()
    return result


def _save_four_way_comparison(
    *,
    input_path: Path,
    baseline_render: Mapping[str, Any],
    hr_render: Mapping[str, Any],
    lr_render: Mapping[str, Any],
    output_path: Path,
) -> None:
    panel_size = 512
    header = 72
    entries = [
        (input_path, "input/canonical", None),
        (Path(str(baseline_render["render_png"])), "baseline 1024", base._metric_subset(baseline_render)),
        (Path(str(hr_render["render_png"])), "HR local", base._metric_subset(hr_render)),
        (Path(str(lr_render["render_png"])), "LR local", base._metric_subset(lr_render)),
    ]
    canvas = Image.new("RGB", (panel_size * len(entries), panel_size + header), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (path, title, metrics) in enumerate(entries):
        with Image.open(path) as image:
            panel = ImageOps.contain(image.convert("RGB"), (panel_size, panel_size))
        x = index * panel_size + (panel_size - panel.width) // 2
        canvas.paste(panel, (x, header + (panel_size - panel.height) // 2))
        draw.text((index * panel_size + 8, 8), title, fill=(255, 255, 255))
        if metrics is not None:
            draw.text(
                (index * panel_size + 8, 31),
                f"PSNR {metrics.get('psnr_db')} SSIM {metrics.get('ssim')}",
                fill=(220, 220, 220),
            )
            draw.text(
                (index * panel_size + 8, 50),
                f"LPIPS {metrics.get('lpips')}",
                fill=(180, 180, 180),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if not args.render:
        raise ValueError("this visual comparison requires --render")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.max_num_tokens) < 1:
        raise ValueError("--max-num-tokens must be positive")
    if int(args.surface_samples) < 1 or int(args.nearest_chunk_size) < 1:
        raise ValueError("surface/chunk sizes must be positive")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base_path = Path(encoder_path).expanduser()
        if not Path(f"{base_path}.json").is_file() or not Path(f"{base_path}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for base path {base_path}")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = Image.open(args.image).convert("RGB")
    source_image.save(output_dir / "input_original.png")

    pipeline = base.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096 = canonical["image_4096"]
    image_1024 = canonical["image_1024"]
    image_512 = canonical["image_512"]
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
    baseline_live = base._validate_mesh(baseline_output[0], "global ordinary Pixal3D-1024 baseline")
    baseline_shape_slat, baseline_texture_slat, decoded_resolution = baseline_latents
    if int(decoded_resolution) != base.OVOXEL_RESOLUTION:
        raise RuntimeError(f"baseline decoder resolution is {decoded_resolution}")
    baseline_mesh = baseline_live.to("cpu")
    baseline_dir = output_dir / "global_baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    envmap = base.load_envmap(str(args.envmap), device="cuda")
    baseline_render = base._render(
        baseline_live,
        output_dir=baseline_dir / "aligned_eval",
        camera=global_camera,
        reference_image=output_dir / "canonical_1024.png",
        args=args,
        envmap=envmap,
    )
    baseline_summary = {
        "route": "ordinary pipeline.run(..., pipeline_type='1024_cascade')",
        "generation_seconds": float(baseline_seconds),
        "decoder_resolution": int(decoded_resolution),
        "vertices": int(baseline_mesh.vertices.shape[0]),
        "faces": int(baseline_mesh.faces.shape[0]),
        "active_ovoxels": int(baseline_mesh.coords.shape[0]),
        "shape_slat_tokens": int(baseline_shape_slat.feats.shape[0]),
        "texture_slat_tokens": int(baseline_texture_slat.feats.shape[0]),
        "render": base._metric_subset(baseline_render),
        "render_detail": baseline_render,
    }
    base._atomic_json(baseline_dir / "summary.json", baseline_summary)
    del baseline_output, baseline_live, baseline_latents
    del baseline_shape_slat, baseline_texture_slat
    base._empty_cuda_cache()

    print("[global-analysis] projecting baseline mesh for tile selection")
    face_min, face_max, face_finite = base._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    global_attr_field = base._make_attribute_query_mesh(baseline_mesh, device)
    print(f"[encoder] loading shape encoder: {args.shape_encoder}")
    shape_encoder = base.pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
    print(f"[encoder] loading PBR encoder: {args.pbr_encoder}")
    pbr_encoder = base.pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
    if not args.low_vram:
        shape_encoder.to(device)
        pbr_encoder.to(device)

    boxes = base._tile_layout()
    requested_ids = base._parse_tile_ids(args.tile_ids)
    hr_patches: List[base.ReturnedTilePatch] = []
    lr_patches: List[base.ReturnedTilePatch] = []
    tile_records: List[Dict[str, Any]] = []
    attempted_tiles = 0
    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image_hr = image_4096.crop(box).convert("RGB")
        tile_image_lr = base._make_lr_tile_image(image_1024, box)
        tile_image_hr.save(tile_dir / "tile_hr_reference.png")
        tile_image_lr.save(tile_dir / "tile_lr_reference.png")
        transform = base._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        base._atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
        selected_face_ids = base._tile_face_ids_from_bbox(face_min, face_max, face_finite, box)
        selected_face_count = int(selected_face_ids.shape[0])
        print(f"[tile {tile_id:02d}] bbox_faces={selected_face_count:,} box={box}")
        if selected_face_count == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "projected_bbox_faces": 0,
                "reason": "no triangle projection bbox intersects tile",
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            continue
        if args.max_tiles is not None and attempted_tiles >= int(args.max_tiles):
            break
        attempted_tiles += 1
        started = time.perf_counter()
        geometry = None
        local_attrs = None
        shape_reference = None
        texture_reference = None
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
                    f"{geometry.stats['global_local_global_q_max_abs_error']:.3e}"
                )
            local_attrs, material_stats = base._resample_local_attrs_from_global(
                geometry=geometry,
                global_attr_field=global_attr_field,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
                face_chunk_size=int(args.material_face_chunk_size),
            )
            shape_reference, shape_encoder_stats = base._encode_local_shape(
                encoder=shape_encoder,
                local_coords=geometry.coords,
                local_dual_vertices=geometry.dual_vertices,
                local_intersected=geometry.intersected,
                device=device,
                low_vram=bool(args.low_vram),
            )
            texture_reference, pbr_encoder_stats = base._encode_local_pbr(
                encoder=pbr_encoder,
                coords=geometry.coords,
                attrs=local_attrs,
                device=device,
                low_vram=bool(args.low_vram),
            )
            alignment_stats = base._latent_support_diagnostics(shape_reference, texture_reference)
            if not alignment_stats["coordinates_exactly_equal"]:
                raise RuntimeError("shape/PBR encoder output coordinates differ")
            if int(shape_reference.feats.shape[0]) > int(args.max_num_tokens):
                raise RuntimeError(
                    f"local latent has {shape_reference.feats.shape[0]:,} tokens, "
                    f"exceeding --max-num-tokens={int(args.max_num_tokens):,}"
                )

            pair = _run_pair_flow(
                pipeline=pipeline,
                tile_image_hr=tile_image_hr,
                tile_image_lr=tile_image_lr,
                tile_box_4096=box,
                transform=transform,
                shape_reference=shape_reference,
                texture_reference=texture_reference,
                shape_params=shape_params,
                texture_params=texture_params,
                seed=int(args.seed),
                tile_id=tile_id,
            )
            with torch.no_grad():
                hr_decoded = pipeline.decode_latent(
                    pair["hr_shape_denorm"], pair["hr_texture_denorm"], base.OVOXEL_RESOLUTION
                )
                lr_decoded = pipeline.decode_latent(
                    pair["lr_shape_denorm"], pair["lr_texture_denorm"], base.OVOXEL_RESOLUTION
                )
            base._sync_cuda()
            if len(hr_decoded) != 1 or len(lr_decoded) != 1:
                raise RuntimeError("paired decoder returned more than one mesh")
            hr_mesh = base._validate_mesh(hr_decoded[0], f"tile {tile_id:02d} HR decode")
            lr_mesh = base._validate_mesh(lr_decoded[0], f"tile {tile_id:02d} LR decode")
            hr_patch = base._local_mesh_to_global_patch(
                tile_id=tile_id,
                box=box,
                local_mesh=hr_mesh,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
            )
            lr_patch = base._local_mesh_to_global_patch(
                tile_id=tile_id,
                box=box,
                local_mesh=lr_mesh,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
            )
            hr_patches.append(hr_patch)
            lr_patches.append(lr_patch)
            record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "tile_seconds": float(time.perf_counter() - started),
                "projected_bbox_faces": selected_face_count,
                "geometry": geometry.stats,
                "material_resampling": material_stats,
                "shape_encoder": shape_encoder_stats,
                "pbr_encoder": pbr_encoder_stats,
                "latent_support": alignment_stats,
                "pair_flow": pair["stats"],
                "hr_mesh": {
                    "vertices": int(hr_mesh.vertices.shape[0]),
                    "faces": int(hr_mesh.faces.shape[0]),
                    "active_ovoxels": int(hr_mesh.coords.shape[0]),
                    "global_patch": hr_patch.stats,
                },
                "lr_mesh": {
                    "vertices": int(lr_mesh.vertices.shape[0]),
                    "faces": int(lr_mesh.faces.shape[0]),
                    "active_ovoxels": int(lr_mesh.coords.shape[0]),
                    "global_patch": lr_patch.stats,
                },
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            print(
                f"[tile {tile_id:02d}] success HR_faces={hr_patch.faces.shape[0]:,} "
                f"LR_faces={lr_patch.faces.shape[0]:,} seconds={record['tile_seconds']:.2f}"
            )
            del hr_decoded, lr_decoded, hr_mesh, lr_mesh, pair
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "projected_bbox_faces": selected_face_count,
                "tile_seconds": float(time.perf_counter() - started),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
        finally:
            geometry = None
            local_attrs = None
            shape_reference = None
            texture_reference = None
            base._empty_cuda_cache()

    del shape_encoder, pbr_encoder, global_attr_field
    base._empty_cuda_cache()
    successful_rows = [row for row in tile_records if row["status"] == "success"]
    failed_rows = [row for row in tile_records if row["status"] == "failed"]
    skipped_rows = [row for row in tile_records if row["status"] == "skipped"]
    if not hr_patches or not lr_patches:
        raise RuntimeError("no successful HR/LR tile patches to stitch")

    print("[stitch] HR nearest-owner half-overlap stitch")
    hr_stitched, hr_stitch_stats = base._stitch_tile_patches_nearest(
        hr_patches,
        layout=baseline_mesh.layout,
        global_camera=global_camera,
        face_chunk_size=int(args.face_projection_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    print("[stitch] LR nearest-owner half-overlap stitch")
    lr_stitched, lr_stitch_stats = base._stitch_tile_patches_nearest(
        lr_patches,
        layout=baseline_mesh.layout,
        global_camera=global_camera,
        face_chunk_size=int(args.face_projection_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    hr_stitch_dir = output_dir / "stitched_hr_global_mesh"
    lr_stitch_dir = output_dir / "stitched_lr_global_mesh"
    hr_stitch_dir.mkdir(parents=True, exist_ok=True)
    lr_stitch_dir.mkdir(parents=True, exist_ok=True)
    hr_overlap = base._save_tile_overlap_visualization(
        image_4096=image_4096,
        boxes=boxes,
        successful_ids=[patch.tile_id for patch in hr_patches],
        output_path=hr_stitch_dir / "tile_overlap_coverage.png",
    )
    lr_overlap = base._save_tile_overlap_visualization(
        image_4096=image_4096,
        boxes=boxes,
        successful_ids=[patch.tile_id for patch in lr_patches],
        output_path=lr_stitch_dir / "tile_overlap_coverage.png",
    )

    print("[render] baseline / HR / LR input-view renders and metrics")
    hr_render = base._render(
        hr_stitched,
        output_dir=hr_stitch_dir / "aligned_eval",
        camera=global_camera,
        reference_image=output_dir / "canonical_1024.png",
        args=args,
        envmap=envmap,
    )
    lr_render = base._render(
        lr_stitched,
        output_dir=lr_stitch_dir / "aligned_eval",
        camera=global_camera,
        reference_image=output_dir / "canonical_1024.png",
        args=args,
        envmap=envmap,
    )
    comparison_path = output_dir / "comparison_input_baseline_hr_lr.png"
    _save_four_way_comparison(
        input_path=output_dir / "canonical_1024.png",
        baseline_render=baseline_render,
        hr_render=hr_render,
        lr_render=lr_render,
        output_path=comparison_path,
    )

    multiview: Dict[str, Any] = {}
    if args.render_multiview:
        print("[render] HR multi-view")
        multiview["baseline_vs_hr"] = base._render_multiview_comparison(
            baseline_mesh,
            hr_stitched,
            output_dir=hr_stitch_dir / "multiview_baseline_vs_hr",
            camera=global_camera,
            args=args,
            envmap=envmap,
        )
        print("[render] LR multi-view")
        multiview["baseline_vs_lr"] = base._render_multiview_comparison(
            baseline_mesh,
            lr_stitched,
            output_dir=lr_stitch_dir / "multiview_baseline_vs_lr",
            camera=global_camera,
            args=args,
            envmap=envmap,
        )

    visual_metrics = {
        "reference_image": str(output_dir / "canonical_1024.png"),
        "camera": global_camera,
        "baseline": base._metric_subset(baseline_render),
        "hr": base._metric_subset(hr_render),
        "lr": base._metric_subset(lr_render),
        "baseline_minus_hr": {
            key: float(baseline_render[key] - hr_render[key])
            for key in ("psnr_db", "ssim", "lpips")
            if baseline_render.get(key) is not None and hr_render.get(key) is not None
        },
        "baseline_minus_lr": {
            key: float(baseline_render[key] - lr_render[key])
            for key in ("psnr_db", "ssim", "lpips")
            if baseline_render.get(key) is not None and lr_render.get(key) is not None
        },
        "hr_minus_lr": {
            key: float(hr_render[key] - lr_render[key])
            for key in ("psnr_db", "ssim", "lpips")
            if hr_render.get(key) is not None and lr_render.get(key) is not None
        },
        "baseline_render_detail": baseline_render,
        "hr_render_detail": hr_render,
        "lr_render_detail": lr_render,
        "comparison_png": str(comparison_path),
        "multiview": multiview,
    }
    base._atomic_json(output_dir / "visual_metrics.json", visual_metrics)
    base._atomic_json(
        hr_stitch_dir / "summary.json",
        {"stitch": hr_stitch_stats, "overlap": hr_overlap, "render": hr_render},
    )
    base._atomic_json(
        lr_stitch_dir / "summary.json",
        {"stitch": lr_stitch_stats, "overlap": lr_overlap, "render": lr_render},
    )
    summary = {
        "format": "pixal3d_hr_lr_same_support_same_noise_mesh_visual_compare_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "cuda_device": int(args.cuda_device),
        "route": {
            "baseline": "ordinary global 1024_cascade",
            "hr": "canonical 4096 1024x1024 tile condition",
            "lr": "canonical 1024 matching 256x256 crop resized to 1024x1024",
            "support": "same local C64 encoder support per tile",
            "initial_noise": "same cloned shape and texture noise per tile",
            "stitch": "projected tile-center nearest-owner; each overlap half retained by one tile; spatial weld only",
            "no_new_generation_method": True,
        },
        "global_baseline": baseline_summary,
        "successful_tiles": int(len(successful_rows)),
        "failed_tiles": int(len(failed_rows)),
        "skipped_tiles": int(len(skipped_rows)),
        "stitch": {"hr": hr_stitch_stats, "lr": lr_stitch_stats},
        "visual_metrics_json": str(output_dir / "visual_metrics.json"),
        "comparison_png": str(comparison_path),
        "hr_multiview_dir": str(hr_stitch_dir / "multiview_baseline_vs_hr"),
        "lr_multiview_dir": str(lr_stitch_dir / "multiview_baseline_vs_lr"),
        "tiles": tile_records,
    }
    base._atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] success={len(successful_rows)} failed={len(failed_rows)} "
        f"skipped={len(skipped_rows)} summary={output_dir / 'summary.json'}"
    )


def main() -> None:
    parser = base.build_parser()
    parser.description = __doc__
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
