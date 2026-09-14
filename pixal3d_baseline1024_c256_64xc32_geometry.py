#!/usr/bin/env python3
"""双级几何起点扫描 → 同模式 C64 texture flow → 4096 材质与输入视角前景 PSNR。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pixal3d_cascade512_1024_tiled2048_crop_condition as geometry_ops
from pixal3d_sweep_texture import prepare_texture_initial, run_texture, save_texture_sweep


def run(args: argparse.Namespace) -> None:
    from dataclasses import replace
    import time

    path = ("image->native baseline mesh1024->voxel2048->Encoder S128"
            "->partial-time noise->343 C32 synchronized Shape512"
            "->global C128 latent->decode2048->voxel4096->Encoder S256"
            "->partial-time noise (same start)->343 C64 synchronized Shape1024"
            "->global C256 latent->decode4096->8 normal views"
            " + baseline Shape C64->Tex1024->material1024"
            "->trilinear query at bridge voxel4096->Texture Encoder C256"
            "->same start_t noise->12 C64 window texture flow steps (final normalized Shape concat)"
            "->single global Texture Decoder->material4096->input-view foreground PSNR; no CFG in tiled stages")
    if args.geometry_source:
        import json
        previous = json.loads((args.geometry_source / "config.json").read_text())["args"]
        for key in ("image", "camera", "model_path", "encoder_path"):
            if Path(previous[key]).resolve() != Path(getattr(args, key)).resolve():
                raise ValueError(f"geometry source {key} does not match this run")
        for key in ("seed", "shape_seed", "shape_steps", "ss_steps"):
            if previous[key] != getattr(args, key):
                raise ValueError(f"geometry source {key} does not match this run")
    init_args = argparse.Namespace(**vars(args))
    if args.smoke_test:
        init_args.output_dir = args.output_dir / "texture_smoke"
    state = geometry_ops.initialize_run(init_args, path_description=path)
    root = state.out
    if args.smoke_test:
        # Exercise both condition branches, all windows, full 4096 decoding and rendering.
        results = []
        for mode in ("conditional", "unconditional"):
            source = args.geometry_source / mode / "start_11"
            experiment = replace(state, out=root / mode)
            results.append(run_texture(experiment, args, mode, 11, source=source))
        geometry_ops.atomic_json(root / "smoke.json", dict(status="complete", results=results))
        geometry_ops.atomic_json(root / "config.json", dict(status="complete", path=path, args=vars(args)))
        return
    if args.geometry_source is None:
        coords_c128, initial_c128 = geometry_ops.prepare_baseline_encoded_s128(state, args)
    times = state.pipeline.shape_slat_sampler.timestep_schedule(
        12, state.pipeline.shape_slat_sampler_params.get("rescale_t", 1.0))
    completed = []
    geometry_ops.atomic_json(root / "progress.json", dict(
        completed=completed, total=24, status="running"))
    geometry_ops.atomic_json(root / "sweep.json", dict(
        status="running", starts=list(range(12)), times=times,
        decode_resolution=4096, tiled_stages=2, paired_start_steps=True, cfg=False))
    for mode in ("conditional", "unconditional"):
        for start in range(12):
            experiment = replace(state, out=root / mode / f"start_{start:02d}",
                                 started=time.perf_counter())
            marker = experiment.out / "experiment_complete.json"
            source = (args.geometry_source / mode / f"start_{start:02d}") if args.geometry_source else experiment.out
            source_marker = source / "experiment_complete.json"
            if args.geometry_source or (args.resume and source_marker.is_file()):
                import json
                saved = json.loads(source_marker.read_text())
                if (saved.get("tiled_stages") != 2 or saved.get("decode_resolution") != 4096
                        or saved.get("start_step") != start or saved.get("stage2_start_step") != start
                        or saved.get("mode") != mode):
                    raise RuntimeError(f"incompatible completed experiment: {marker}")
                print(f"[sweep] reuse geometry: {mode} start={start}; texture follows", flush=True)
                texture = run_texture(experiment, args, mode, start, source=source)
                saved.update(texture_executed=True, texture=texture)
                geometry_ops.atomic_json(marker, saved)
                geometry_ops.atomic_json(experiment.out / "summary.json", saved)
                geometry_ops.atomic_json(experiment.out / "config.json", dict(
                    status="complete", path=path, mode=mode, start_step=start,
                    stage2_start_step=start, args=vars(args), texture_executed=True))
                completed.append(dict(mode=mode, start=start))
                save_texture_sweep(root)
                geometry_ops.atomic_json(root / "progress.json", dict(completed=completed, total=24, status="running"))
                if args.geometry_source is None:
                    geometry_ops.save_sweep_contact_sheet(root, mode, times)
                continue
            geometry_ops.atomic_json(experiment.out / "config.json", dict(
                status="running", path=path, mode=mode, start_step=start,
                start_t=times[start], stage2_start_step=start,
                decode_resolution=4096, args=vars(args)))
            print(f"[sweep] {mode} start={start} t={times[start]:.6f}", flush=True)
            normalized_c128, records_c32 = geometry_ops.run_synchronized_overlap_stage(
                experiment, args, coords_c128, 128, 32, 16,
                "shape_slat_flow_model_512", "overlap343_c32_shape512", args.shape_seed + 1,
                initial_normalized=initial_c128, start_step=start, condition_mode=mode)
            coords_c256, initial_c256 = geometry_ops.decode_c128_and_encode_s256(
                experiment, args, coords_c128, normalized_c128)
            del normalized_c128
            # Query precisely the bridge voxel4096 before the second Shape flow.
            initial_texture = prepare_texture_initial(experiment, args, mode, coords_c256)
            del initial_texture
            normalized_c256, records_c64 = geometry_ops.run_synchronized_overlap_stage(
                experiment, args, coords_c256, 256, 64, 32,
                "shape_slat_flow_model_1024", "overlap343_c64_shape1024", args.shape_seed + 2,
                initial_normalized=initial_c256, start_step=start, condition_mode=mode)
            del initial_c256
            summary = geometry_ops.decode_global_geometry(
                experiment, args, coords_global=coords_c256,
                final_normalized=normalized_c256, decode_resolution=4096, global_grid=256,
                token_counts=dict(encoded_s128=len(coords_c128), encoded_s256=len(coords_c256)),
                active_cube_counts={
                    "343_local_c32": sum(bool(r["global_row_ids"].numel()) for r in records_c32),
                    "343_local_c64": sum(bool(r["global_row_ids"].numel()) for r in records_c64)})
            del normalized_c256, records_c32, records_c64
            geometry_ops.empty_cuda()
            geometry_ops.render_saved_normal_multiview(experiment, args)
            texture = run_texture(experiment, args, mode, start)
            summary.update(mode=mode, start_step=start, start_t=times[start],
                           stage2_start_step=start, tiled_stages=2, cfg=False,
                           texture_executed=True, texture=texture)
            geometry_ops.atomic_json(experiment.out / "summary.json", summary)
            geometry_ops.atomic_json(marker, summary)
            save_texture_sweep(root)
            geometry_ops.save_sweep_contact_sheet(root, mode, times)
            completed.append(dict(mode=mode, start=start))
            geometry_ops.atomic_json(root / "progress.json", dict(
                completed=completed, total=24, status="running"))
    geometry_ops.atomic_json(root / "sweep.json", dict(
        status="complete", completed=completed, times=times, tiled_stages=2,
        paired_start_steps=True, decode_resolution=4096, cfg=False,
        texture_executed=True, texture_steps=args.tex_steps, texture_psnr=str(root / "texture_psnr.json"),
        sheets=[str(root / f"{mode}_texture_3x4.png") for mode in ("conditional", "unconditional")]))
    geometry_ops.atomic_json(root / "progress.json", dict(
        completed=completed, total=24, status="complete"))
    geometry_ops.atomic_json(root / "config.json", dict(
        status="complete", path=path, args=vars(args)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(geometry_ops.DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/baseline_encoded_s128_s256_material_cascade_v2_cuda4"))
    parser.add_argument("--geometry-source", type=Path, help="Read-only reuse of a completed two-stage geometry sweep")
    parser.add_argument("--encoder-path", type=Path, default=Path(
        "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/microsoft/TRELLIS___2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-seed", type=int, default=43)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, choices=[12], default=12)
    parser.add_argument("--flow-batch-size", type=int, default=8)
    parser.add_argument("--normal-resolution", type=int, default=1024)
    parser.add_argument("--normal-face-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tex-steps", type=int, default=12)
    parser.add_argument("--tex-seed", type=int, default=46)
    parser.add_argument("--texture-encoder-path", type=Path, default=Path(
        "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/microsoft/TRELLIS___2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--material-query-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--texture-resolution", type=int, default=1024, help="Input-view render/PSNR resolution")
    parser.add_argument("--smoke-test", action="store_true", help="Full material cascade for both cached start_11 geometries, isolated outputs")
    args = parser.parse_args()
    if min(args.flow_batch_size, args.shape_steps, args.ss_steps,
           args.normal_resolution, args.normal_face_chunk_size, args.tex_steps, args.texture_resolution,
           args.material_query_chunk_size) <= 0:
        parser.error("batch size, steps and rendering sizes must be positive")
    if args.smoke_test and args.geometry_source is None:
        parser.error("--smoke-test requires --geometry-source")
    if args.geometry_source and args.geometry_source.resolve() == args.output_dir.resolve():
        parser.error("--geometry-source and --output-dir must differ")
    return args


if __name__ == "__main__":
    run(parse_args())
