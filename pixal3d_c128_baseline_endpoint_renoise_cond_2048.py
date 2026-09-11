#!/usr/bin/env python3
"""Conditional-suffix sweep re-noising the SAME final baseline endpoint E12."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import torch
from PIL import Image

import pixal3d_c128_baseline_endpoint_renoise_uncond_2048 as base
import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
from inference import MODEL_PATH, init_pipeline


FORMAT = "pixal3d_c128_final_baseline_renoise_cond_2048_v2"
DEFAULT_SHARED = Path("outputs/c128_final_baseline_renoise_uncond_2048_cuda4")
DEFAULT_OUTPUT = Path("outputs/c128_final_baseline_renoise_cond_2048_cuda4")


@torch.no_grad()
def full_image_texture_condition(
    pipeline: Any,
    image_path: Path,
    camera: Mapping[str, float],
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    cache = output / "conditions" / "texture_global_c128_full_image.pt"
    if resume and cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if not torch.equal(payload["coords"].int(), coords.int()):
            raise RuntimeError("cached image condition support mismatch")
        glob, proj = payload["global"], payload["proj"]
        print("[condition] cache hit", flush=True)
    else:
        image = Image.open(image_path).convert("RGB")
        if image.size != (1024, 1024):
            raise RuntimeError(f"expected canonical 1024 condition, got {image.size}")
        print("[condition] project complete 1024 image onto global C128 support", flush=True)
        condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [image],
            coords.to(pipeline.device),
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
            grid_resolution_override=base.LATENT_GRID,
        )
        glob = condition["cond"]["global"].detach().float().cpu().contiguous()
        proj = condition["cond"]["proj"].feats.detach().float().cpu().contiguous()
        base.atomic_save(cache, {
            "format": FORMAT,
            "coords": coords.int(),
            "global": glob,
            "proj": proj,
            "source_image": str(image_path.resolve()),
            "source": "one complete-image texture projection on global C128 support",
        })
        image.close()
        del condition
        base.empty_cuda()
    if glob.shape != (1, 5, 1024) or proj.shape[0] != coords.shape[0]:
        raise RuntimeError("global C128 texture condition is not sparse-row aligned")
    cubes: dict[int, dict[str, torch.Tensor]] = {}
    for record in records:
        rows = record["global_row_ids"].cpu().long()
        cubes[int(record["cube_id"])] = {
            "global_row_ids": rows,
            "global": glob,
            "proj": proj.index_select(0, rows).contiguous(),
        }
    return {"cubes": cubes, "fingerprint_sha256": base.tensor_hash(proj)}


@torch.no_grad()
def run_conditional_variants(
    pipeline: Any,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    owner: torch.Tensor,
    shape_normalized: torch.Tensor,
    endpoints: Path,
    fixed_noise_path: Path,
    condition: Mapping[str, Any],
    output: Path,
    device: torch.device,
    resume: bool,
) -> list[dict[str, Any]]:
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"].eval().to(device)
    schedule = sampler.timestep_schedule(base.STEPS, 1.0)
    sigma_min = float(sampler.sigma_min)
    epsilon_payload = torch.load(fixed_noise_path, map_location="cpu", weights_only=False)
    epsilon = epsilon_payload["epsilon"].float()
    endpoint_payloads = [
        torch.load(endpoints / f"step_{step:02d}.pt", map_location="cpu", weights_only=False)
        for step in range(1, base.STEPS + 1)
    ]
    if not torch.equal(endpoint_payloads[-1]["coords"].int(), coords.int()):
        raise RuntimeError("final endpoint support mismatch")
    if not torch.equal(epsilon_payload["coords"].int(), coords.int()):
        raise RuntimeError("fixed noise support mismatch")
    final_endpoint = endpoint_payloads[-1]["normalized_features"].float().to(device)
    fixed_noise = epsilon.to(device)
    final_endpoint_hash = base.tensor_hash(final_endpoint)
    noise_hash = base.tensor_hash(epsilon)
    channels = int(epsilon.shape[1])
    groups = cube_flow.pack_groups(records, base.FLOW_BATCH_SIZE, 10_000_000, require_owned=True)
    if len(groups) != 1 or len(groups[0]) != 8:
        raise RuntimeError("expected one physical batch of eight C64 contexts")
    params = {
        "steps": base.STEPS,
        "rescale_t": 1.0,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.0, 1.0),
    }
    results: list[dict[str, Any]] = []
    for n in range(1, base.STEPS + 1):
        final_path = output / "flow" / f"prefix_{n:02d}" / "final_texture_normalized.pt"
        if resume and final_path.is_file():
            saved = torch.load(final_path, map_location="cpu", weights_only=False)
            base.validate_final_endpoint_cache(saved, FORMAT, coords, final_endpoint_hash, noise_hash)
            results.append(saved["record"])
            print(f"[conditional variant {n:02d}/12] cache hit", flush=True)
            continue
        start_t = float(schedule[n])
        epsilon_weight = sigma_min + (1.0 - sigma_min) * start_t
        state = ((1.0 - start_t) * final_endpoint + epsilon_weight * fixed_noise).cpu()
        base.atomic_save(output / "noised_states" / f"n_{n:02d}.pt", {
            "coords": coords, "normalized_features": state, "t": start_t,
            "final_endpoint_sha256": final_endpoint_hash, "noise_sha256": noise_hash,
        })
        step_rows: list[dict[str, Any]] = []
        print(
            f"[conditional variant {n:02d}/12] start_t={start_t:.6f}, "
            f"conditional_steps={base.STEPS-n}", flush=True,
        )
        for step_index in range(n, base.STEPS):
            t_now = float(schedule[step_index])
            t_next = float(schedule[step_index + 1])
            started = time.perf_counter()
            proposals: list[tuple[int, torch.Tensor, torch.Tensor]] = []
            for group in groups:
                values, timing = cube_flow._one_prediction(
                    group, state, condition, sampler, model, params,
                    t_now, t_next, device, shape_normalized,
                )
                proposals.extend(
                    (int(record["cube_id"]), record["global_row_ids"], velocity)
                    for record, velocity in zip(group, values)
                )
            velocity = cube_flow.validate_owner_scatter(owner, proposals, channels)
            state = cube_flow.jacobi_update(state, velocity, t_now, t_next)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"variant {n}, step {step_index+1} is non-finite")
            step_rows.append({
                "step": step_index + 1,
                "t": t_now,
                "t_next": t_next,
                "route": "full_image_conditional_local_C64",
                "seconds": time.perf_counter() - started,
                "model_seconds": timing["seconds"],
            })
        record = {
            "n": n,
            "skipped_prefix_steps": n,
            "endpoint_step": base.STEPS,
            "final_endpoint_sha256": final_endpoint_hash,
            "noise_sha256": noise_hash,
            "conditional_suffix_steps": base.STEPS - n,
            "suffix_steps": base.STEPS - n,
            "suffix_label": "cond",
            "start_t": start_t,
            "clean_endpoint_weight": 1.0 - start_t,
            "epsilon_weight": epsilon_weight,
            "shape_c128_fixed": True,
            "condition": "complete canonical 1024 image projected once to global C128",
            "guidance_strength": 1.0,
            "context": base.CONTEXT,
            "stride": base.STRIDE,
            "configured_flow_batch_size": base.FLOW_BATCH_SIZE,
            "effective_batch_size": 8,
            "steps": step_rows,
            "state_sha256": base.tensor_hash(state),
            "path": str(final_path.resolve()),
        }
        base.atomic_save(final_path, {
            "format": FORMAT,
            "coords": coords,
            "normalized_features": state,
            "record": record,
        })
        results.append(record)
        print(f"[conditional variant {n:02d}/12] flow complete", flush=True)
    model.cpu()
    del model, endpoint_payloads, epsilon, epsilon_payload, final_endpoint, fixed_noise
    base.empty_cuda()
    return results


def comparison_plot(
    unconditional: Sequence[Mapping[str, Any]],
    conditional: Sequence[Mapping[str, Any]],
    output: Path,
    visibility_gated: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    steps = [int(row["n"]) for row in conditional]
    series = [(unconditional, "unconditional", "tab:blue"),
              (conditional, "conditional", "tab:orange")]
    if visibility_gated is not None:
        series.append((visibility_gated, "visibility-gated (experiment 3)", "tab:green"))
    for rows, label, color in series:
        if [int(row["n"]) for row in rows] != steps:
            raise ValueError(f"Mismatched n values for {label}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    for column, (group, title) in enumerate(
        (("front_vs_gt", "Front vs GT"), ("back_vs_baseline", "Back vs baseline"))
    ):
        for rows, label, color in series:
            axes[0, column].plot(
                steps, [row[group]["foreground_psnr_db"] for row in rows], "o-", label=label, color=color,
            )
            axes[1, column].plot(
                steps, [row[group]["foreground_ssim"] for row in rows], "o-", label=label, color=color,
            )
        axes[0, column].set_title(f"{title}: foreground PSNR")
        axes[1, column].set_title(f"{title}: foreground SSIM")
    for row_index, ylabel in enumerate(("PSNR (dB)", "SSIM")):
        for axis in axes[row_index]:
            axis.set_xlabel("Skipped steps n / 12 (fixed E12 re-noise)")
            axis.set_ylabel(ylabel)
            axis.set_xticks(steps)
            axis.grid(alpha=0.3)
            axis.legend()
    fig.suptitle("Conditional / unconditional / visibility-gated C128 texture suffix"
                 if visibility_gated is not None else
                 "Image-conditional vs image-unconditional C128 texture suffix", fontsize=17)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--endpoint-dir", type=Path, default=base.DEFAULT_ENDPOINTS)
    parser.add_argument("--baseline-dir", type=Path, default=base.DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [value.strip() for value in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.physical_cuda}")
    if args.render_resolution != 1024:
        raise ValueError("rendered subfigures must remain 1024 pixels")
    uncond_summary = json.loads((args.shared_dir / "summary.json").read_text(encoding="utf-8"))
    if uncond_summary.get("status") != "complete":
        raise RuntimeError("unconditional reference experiment is incomplete")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)

    shape_payload = torch.load(
        args.shared_dir / "support" / "fixed_c128_shape_slat.pt",
        map_location="cpu", weights_only=False,
    )
    shape_norm_payload = torch.load(
        args.shared_dir / "support" / "fixed_c128_shape_normalized.pt",
        map_location="cpu", weights_only=False,
    )
    coords = shape_payload["coords"].int()
    shape_raw = shape_payload["raw_features"].float()
    shape_normalized = shape_norm_payload["normalized_features"].float()
    records, owner = base.build_records(coords)
    camera = json.loads((args.baseline_dir / "global_camera.json").read_text(encoding="utf-8"))
    condition = full_image_texture_condition(
        pipeline,
        args.baseline_dir / "canonical_1024.png",
        camera,
        coords,
        records,
        output,
        args.resume,
    )
    variants = run_conditional_variants(
        pipeline, coords, records, owner, shape_normalized,
        args.shared_dir / "encoded_c128_endpoints",
        args.shared_dir / "flow" / "fixed_noise.pt",
        condition, output, device, args.resume,
    )
    evaluated = base.decode_render_evaluate(
        pipeline, shape_raw, coords, variants,
        args.endpoint_dir.resolve(), args.baseline_dir.resolve(), camera,
        output, device, args.render_resolution, args.face_chunk_size, args.resume,
    )
    visuals = {
        "front_renders_3x4": output / "front_renders_3x4.png",
        "front_error_maps_3x4": output / "front_error_maps_3x4.png",
        "back_renders_3x4": output / "back_renders_3x4.png",
        "back_error_maps_3x4": output / "back_error_maps_3x4.png",
        "foreground_metrics": output / "foreground_metrics.png",
        "conditional_vs_unconditional": output / "conditional_vs_unconditional_metrics.png",
    }
    base.make_grid(evaluated, visuals["front_renders_3x4"], "front_render", "front_vs_gt", "front vs GT")
    base.make_grid(evaluated, visuals["front_error_maps_3x4"], "front_error_map", "front_vs_gt", "front error")
    base.make_grid(evaluated, visuals["back_renders_3x4"], "back_render", "back_vs_baseline", "back vs baseline")
    base.make_grid(evaluated, visuals["back_error_maps_3x4"], "back_error_map", "back_vs_baseline", "back error")
    base.make_metric_plot(evaluated, visuals["foreground_metrics"])
    gated_summary_path = Path(__file__).resolve().parent / "outputs/c128_final_baseline_renoise_visibility_gated_2048_cuda4/summary.json"
    gated_results = None
    if gated_summary_path.is_file():
        gated_summary = json.loads(gated_summary_path.read_text(encoding="utf-8"))
        if gated_summary.get("status") == "complete":
            gated_results = gated_summary["results"]
    comparison_plot(uncond_summary["results"], evaluated, visuals["conditional_vs_unconditional"], gated_results)
    summary = {
        "format": FORMAT,
        "status": "complete",
        "semantics": (
            "variant n forward-noises the SAME final baseline endpoint E12 with the same fixed epsilon, "
            "then runs complete-image-conditional local-C64 flow for steps n+1..12"
        ),
        "comparison_reference": str((args.shared_dir / "summary.json").resolve()),
        "physical_cuda": args.physical_cuda,
        "input_grid": base.INPUT_GRID,
        "latent_grid": base.LATENT_GRID,
        "c2048_tokens": int(uncond_summary["c2048_tokens"]),
        "c128_tokens": int(coords.shape[0]),
        "context": base.CONTEXT,
        "stride": base.STRIDE,
        "configured_flow_batch_size": base.FLOW_BATCH_SIZE,
        "effective_flow_batch_size": 8,
        "condition_fingerprint_sha256": condition["fingerprint_sha256"],
        "results": evaluated,
        "visuals": {key: str(path.resolve()) for key, path in visuals.items()},
    }
    base.atomic_json(output / "summary.json", summary)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
