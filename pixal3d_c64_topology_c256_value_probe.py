#!/usr/bin/env python3
"""One-step/one-block C64-topology -> tiled-C256-value probe on CUDA 4.

This is the deliberately minimal, falsifiable phase requested in codex.md.  It
runs both branches to the same sampler step, captures post-RMSNorm/post-RoPE
Q/K, synchronizes every non-empty C256 cube at block 20, and evaluates the two
head-space message variants.  It does not decode or claim an end-to-end image
quality result; the saved report labels that remaining experiment explicitly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

import pixal3d.models as models
from inference import IMAGE_COND_CONFIGS, build_image_cond_model
from pixal3d.experiments.global_attention_routing import (
    ParentCorrespondence,
    attention_for_queries,
    gather_packed_rows,
    global_replace_and_residual,
    outside_group_attention_ratio,
    scatter_unique_rows,
    tensor_rms,
)
from pixal3d.modules.sparse import SparseTensor


MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16")
BASELINE = Path("outputs/baseline1024_uniform_texture_endpoints_cuda4")
FINE = Path("outputs/global_c256_encdown_context1024_flow_singleview_cuda4")
IMAGE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img/canonical_1024.png")
CAMERA = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json")
DEFAULT_OUT = Path("outputs/c64_topology_c256_value_probe_cuda4")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_tensor(path: Path, key: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)[key]


def nearest_canonical_correspondence(
    coarse_coords: torch.Tensor, fine_coords: torch.Tensor
) -> tuple[ParentCorrespondence, dict[str, Any]]:
    coarse = coarse_coords[:, 1:].cpu().numpy().astype(np.int64)
    fine = fine_coords[:, 1:].cpu().numpy().astype(np.int64)
    exact_parent = fine // 4
    coarse_set = {tuple(row) for row in coarse}
    exact_hits = np.fromiter(
        (tuple(row) in coarse_set for row in exact_parent), dtype=np.bool_, count=len(fine)
    )
    # Compare voxel centres in one canonical C64 frame.  This is the actual
    # nearest-support rule; /4 is reported only as an audit statistic.
    fine_centres_c64 = (fine.astype(np.float64) + 0.5) / 4.0 - 0.5
    distance, parent = cKDTree(coarse.astype(np.float64)).query(fine_centres_c64, k=1)
    parent_t = torch.from_numpy(parent.astype(np.int64))
    counts = torch.bincount(parent_t, minlength=len(coarse))
    nonempty = counts[counts > 0].float()
    stats = {
        "c256_total_points": int(len(fine)),
        "c64_total_points": int(len(coarse)),
        "each_fine_has_one_parent": True,
        "parent_rule": "nearest baseline C64 support point in canonical voxel-centre coordinates",
        "fine_div4_exact_support_hits": int(exact_hits.sum()),
        "fine_div4_exact_support_misses": int((~exact_hits).sum()),
        "c64_parents_without_child": int((counts == 0).sum()),
        "children_nonempty_min": int(nonempty.min()),
        "children_nonempty_median": float(nonempty.median()),
        "children_all_mean": float(counts.float().mean()),
        "children_max": int(counts.max()),
        "nearest_distance_c64_units": {
            "min": float(distance.min()), "median": float(np.median(distance)),
            "mean": float(distance.mean()), "max": float(distance.max()),
        },
        "duplicate_fine_points": int(len(fine) - len(np.unique(fine, axis=0))),
    }
    return ParentCorrespondence(parent_t.long(), len(coarse)), stats


def pack_fine(
    global_features: torch.Tensor,
    global_coords: torch.Tensor,
    projected: torch.Tensor,
    global_tokens: torch.Tensor,
    cube_dir: Path,
    device: torch.device,
) -> tuple[SparseTensor, SparseTensor, dict[str, Any], torch.Tensor, list[int]]:
    feat_parts, coord_parts, proj_parts, row_parts, cube_ids = [], [], [], [], []
    for path in sorted(cube_dir.glob("cube_*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        rows = record["global_row_ids"].long()
        if not rows.numel():
            continue
        batch_id = len(cube_ids)
        coords = record["local_coords"].clone().int()
        coords[:, 0] = batch_id
        feat_parts.append(global_features.index_select(0, rows))
        coord_parts.append(coords)
        proj_parts.append(projected.index_select(0, rows))
        row_parts.append(rows)
        cube_ids.append(int(record["cube_id"]))
    packed_rows = torch.cat(row_parts)
    packed = SparseTensor(torch.cat(feat_parts).to(device), torch.cat(coord_parts).to(device))
    packed_proj = SparseTensor(torch.cat(proj_parts).to(device), packed.coords)
    condition = {"global": global_tokens.repeat(len(cube_ids), 1, 1).to(device), "proj": packed_proj}
    return packed, packed_proj, condition, packed_rows, cube_ids


def split_global(value: SparseTensor, packed_rows: torch.Tensor, count: int) -> torch.Tensor:
    return scatter_unique_rows(value.feats, packed_rows, count)


@torch.no_grad()
def baseline_condition(
    coords: torch.Tensor, device: torch.device, out: Path
) -> dict[str, Any]:
    cache = out / "baseline_c64_condition.pt"
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        return {
            "global": payload["global"].to(device),
            "proj": SparseTensor(payload["proj"].to(device), coords.to(device)),
        }
    model = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])
    model.to(device).eval()
    if getattr(model, "use_naf_upsample", False):
        model._load_naf()
    camera = json.loads(CAMERA.read_text())
    image = Image.open(IMAGE).convert("RGB")
    global_token, projected = model(
        [image],
        camera_angle_x=torch.tensor([camera["camera_angle_x"]], device=device),
        distance=torch.tensor([camera["distance"]], device=device),
        mesh_scale=torch.tensor([camera["mesh_scale"]], device=device),
        transform_matrix=None,
        grid_indices=coords[:, 1:].to(device),
        grid_resolution=64,
        projection_crop_box=None,
    )
    payload = {
        "global": global_token.detach().float().cpu(),
        "proj": projected[0].detach().float().cpu(),
    }
    torch.save(payload, cache)
    model.cpu(); del model, projected, global_token; empty_cuda()
    return {
        "global": payload["global"].to(device),
        "proj": SparseTensor(payload["proj"].to(device), coords.to(device)),
    }


def normalized_shape(raw: torch.Tensor) -> torch.Tensor:
    config = json.loads(Path("/home/nvme04/yyyan/download/model/Pixal3D/pipeline.json").read_text())["args"]
    mean = torch.tensor(config["shape_slat_normalization"]["mean"])[None]
    std = torch.tensor(config["shape_slat_normalization"]["std"])[None]
    return (raw.float() - mean) / std


def local_tile_id_from_c64(coords: torch.Tensor) -> torch.Tensor:
    xyz = coords[:, 1:].long()
    cube = torch.div(xyz, 16, rounding_mode="floor").clamp(0, 3)
    return cube[:, 0] * 16 + cube[:, 1] * 4 + cube[:, 2]


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-step", type=int, default=6, help="0-based model call")
    parser.add_argument("--block", type=int, default=20, help="0-based block id")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.split(",") != [str(args.physical_cuda)]:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={args.physical_cuda}, got {visible}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    c64_coords = load_tensor(BASELINE / "final_shape_slat.pt", "coords").int()
    c64_shape = normalized_shape(load_tensor(BASELINE / "final_shape_slat.pt", "raw_feats"))
    c256_coords = load_tensor(FINE / "support/global_c256_support.pt", "coords").int()
    c256_shape = load_tensor(FINE / "shape/final_state_normalized.pt", "features").float()
    c256_noise = load_tensor(FINE / "texture/initial_noise.pt", "features").float()
    cond256_payload = torch.load(FINE / "conditions/texture_global1024.pt", map_location="cpu", weights_only=False)
    correspondence, mapping_stats = nearest_canonical_correspondence(c64_coords, c256_coords)

    cube_coverage = json.loads((FINE / "cubes/coverage_owner_stats.json").read_text())
    coverage_histogram = cube_coverage["coverage_histogram"]
    mapping_stats.update({
        "tile_overlap_unique_points": int(sum(v for k, v in coverage_histogram.items() if int(k) > 1)),
        "tile_overlap_extra_occurrences": int(sum((int(k) - 1) * v for k, v in coverage_histogram.items() if int(k) > 1)),
        "tile_coverage_histogram": coverage_histogram,
        "active_tiles": int(cube_coverage["flow_active_cubes"]),
    })
    save_json(out / "correspondence_stats.json", mapping_stats)

    cond64 = baseline_condition(c64_coords, device, out)
    flow = models.from_pretrained(str(MODEL)).to(device).eval()
    if not (flow.model_channels == 1536 and flow.num_heads == 12 and flow.num_blocks == 30):
        raise RuntimeError("unexpected runtime texture flow architecture")
    schedule = [float(x) for x in __import__("pixal3d.pipelines.samplers.flow_euler", fromlist=["FlowEulerSampler"]).FlowEulerSampler.timestep_schedule(12, 3.0)]

    # Both branches start from the same explicit seed.  Point counts differ, so
    # this is reproducible RNG equality rather than pointwise noise equality.
    generator = torch.Generator().manual_seed(args.seed)
    c64_state = torch.randn((len(c64_coords), 32), generator=generator)
    captured: dict[str, Any] = {}
    attn64 = flow.blocks[args.block].self_attn

    def capture64(_module: Any, qkv: Any, head_output: Any) -> None:
        q, k, v = qkv.feats.unbind(1)
        captured["q64"] = q.detach().clone()
        captured["k64"] = k.detach().clone()
        captured["v64_base"] = v.detach().clone()
        captured["y64_base"] = head_output.feats.detach().clone()

    for step in range(args.target_step + 1):
        t, t_next = schedule[step], schedule[step + 1]
        if step == args.target_step:
            attn64.runtime_attention_processor = capture64
        sparse = SparseTensor(c64_state.to(device), c64_coords.to(device))
        concat = SparseTensor(c64_shape.to(device), sparse.coords)
        velocity = flow(sparse, torch.tensor([1000 * t], device=device), cond64, concat_cond=concat)
        attn64.runtime_attention_processor = None
        c64_state = (c64_state - (t - t_next) * velocity.feats.float().cpu())
        del sparse, concat, velocity
    if "q64" not in captured:
        raise RuntimeError("baseline attention capture did not fire")

    routed: dict[str, Any] = {}
    attn256 = flow.blocks[args.block].self_attn
    global_count = len(c256_coords)
    packed_rows_reference: torch.Tensor | None = None

    def capture256(_module: Any, qkv: Any, head_output: Any) -> None:
        q_local, k_local, v_local_packed = qkv.feats.unbind(1)
        assert packed_rows_reference is not None
        v256 = scatter_unique_rows(v_local_packed, packed_rows_reference, global_count)
        local256 = scatter_unique_rows(head_output.feats, packed_rows_reference, global_count)
        # R and the residual algebra use float32 accumulation.  The routed V
        # is cast back only for the pretrained bf16 attention kernel.
        v64_from_fine_fp32, child_counts = correspondence.reduce_mean(v256.float())
        v64_from_fine = v64_from_fine_fp32.to(captured["q64"].dtype)
        # flash-attn uses exactly the same scale and routing semantics as the
        # baseline kernel but substitutes fine-derived V.
        import flash_attn
        y64 = flash_attn.flash_attn_func(
            captured["q64"].unsqueeze(0), captured["k64"].unsqueeze(0),
            v64_from_fine.unsqueeze(0),
        )[0]
        replace, combined, residual = global_replace_and_residual(
            local256.float(), y64.float(), correspondence
        )

        coarse_groups = local_tile_id_from_c64(c64_coords).to(device)
        valid = torch.where(child_counts > 0)[0]
        query_rows = valid[torch.linspace(0, valid.numel() - 1, 512, device=device).long()]
        weights = attention_for_queries(captured["q64"], captured["k64"], query_rows)
        ratios = outside_group_attention_ratio(
            weights, coarse_groups.index_select(0, query_rows), coarse_groups
        )
        z = c64_coords[:, 3].to(device)
        qz = z.index_select(0, query_rows)
        low, high = torch.quantile(z.float(), torch.tensor([0.25, 0.75], device=device))
        routed.update({
            "q_local_shape": list(q_local.shape), "k_local_shape": list(k_local.shape),
            "v256_shape": list(v256.shape), "v64_from_fine_shape": list(v64_from_fine.shape),
            "y64_global_shape": list(y64.shape), "y_local_shape": list(local256.shape),
            "y_local_residual_shape": list(residual.shape), "y_final_shape": list(combined.shape),
            "local_message_rms": tensor_rms(local256), "global_message_rms": tensor_rms(replace),
            "local_residual_rms": tensor_rms(residual),
            "global_correction_rms": tensor_rms(combined - local256),
            "replace_minus_local_rms": tensor_rms(replace - local256),
            "outside_tile_attention_ratio_mean": float(ratios.float().mean()),
            "outside_tile_attention_ratio_median": float(ratios.float().median()),
            "outside_tile_attention_ratio_z_low_quartile": float(ratios[qz <= low].float().mean()),
            "outside_tile_attention_ratio_z_high_quartile": float(ratios[qz >= high].float().mean()),
            "diagnostic_query_count": int(query_rows.numel()),
            "within_parent_residual_reduction_rms": tensor_rms(correspondence.reduce_mean(residual)[0]),
        })
        torch.save({
            "query_rows": query_rows.cpu(), "outside_ratio_per_query_head": ratios.float().cpu(),
            "q64_sample": captured["q64"].index_select(0, query_rows[:32]).cpu(),
            "k64_sample": captured["k64"].index_select(0, query_rows[:32]).cpu(),
            "v256_sample": v256[:32].cpu(), "y64_global_sample": y64[:32].cpu(),
            "replace_sample": replace[:32].cpu(), "combined_sample": combined[:32].cpu(),
            "residual_sample": residual[:32].cpu(),
        }, out / "diagnostic_tensors.pt")
        ratio_by_query = ratios.float().mean(1).cpu().numpy()
        np.savetxt(
            out / "outside_tile_attention_ratio.csv",
            np.column_stack((query_rows.cpu().numpy(), ratio_by_query)),
            delimiter=",", header="c64_query_row,outside_tile_attention_ratio", comments="",
        )
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(ratio_by_query, bins=32)
        axis.axvline(float(ratio_by_query.mean()), color="red", linestyle="--", label="mean")
        axis.set(xlabel="outside-tile attention ratio", ylabel="C64 query count",
                 title=f"Baseline C64 routing, step {args.target_step}, block {args.block}")
        axis.legend(); figure.tight_layout()
        figure.savefig(out / "outside_tile_attention_ratio.png", dpi=160)
        plt.close(figure)

    c256_state = c256_noise
    for step in range(args.target_step + 1):
        t, t_next = schedule[step], schedule[step + 1]
        packed, _, condition, packed_rows, _ = pack_fine(
            c256_state, c256_coords, cond256_payload["proj"], cond256_payload["global"],
            FINE / "cubes", device,
        )
        packed_rows_reference = packed_rows
        concat_packed, _, _, concat_rows, cube_ids = pack_fine(
            c256_shape, c256_coords, cond256_payload["proj"], cond256_payload["global"],
            FINE / "cubes", device,
        )
        if not torch.equal(packed_rows, concat_rows):
            raise RuntimeError("texture/shape packed row mismatch")
        if step == args.target_step:
            attn256.runtime_attention_processor = capture256
        velocity = flow(
            packed, torch.full((len(cube_ids),), 1000 * t, device=device),
            condition, concat_cond=concat_packed,
        )
        attn256.runtime_attention_processor = None
        velocity_global = split_global(velocity, packed_rows, global_count).float().cpu()
        c256_state = c256_state - (t - t_next) * velocity_global
        del packed, concat_packed, condition, velocity, velocity_global
        empty_cuda()
    if not routed:
        raise RuntimeError("fine attention capture did not fire")

    runtime = {
        "physical_cuda": args.physical_cuda,
        "logical_device": str(device), "gpu": torch.cuda.get_device_name(device),
        "timestep_step_index": args.target_step, "t": schedule[args.target_step],
        "t_next": schedule[args.target_step + 1], "block_id_zero_based": args.block,
        "hidden_shape_c64": [len(c64_coords), flow.model_channels],
        "q64_shape": list(captured["q64"].shape), "k64_shape": list(captured["k64"].shape),
        "v64_base_shape": list(captured["v64_base"].shape),
        "attention_output_c64_head_shape": list(captured["y64_base"].shape),
        "c64_coords_min": c64_coords[:, 1:].min(0).values.tolist(),
        "c64_coords_max": c64_coords[:, 1:].max(0).values.tolist(),
        "c256_coords_min": c256_coords[:, 1:].min(0).values.tolist(),
        "c256_coords_max": c256_coords[:, 1:].max(0).values.tolist(),
        "model_channels": flow.model_channels, "heads": flow.num_heads,
        "head_dim": flow.model_channels // flow.num_heads,
        "blocks": flow.num_blocks, "pe_mode": flow.pe_mode,
        **routed,
    }
    save_json(out / "runtime_diagnostics.json", runtime)

    report = f"""# C64 topology → C256 value minimal probe

## Scope and status

Training-free, one object, texture only, step `{args.target_step}` (`t={schedule[args.target_step]:.6f}`), block `{args.block}` (zero-based). Shape support/latent, image, scheduler and weights are fixed. This phase executes the codex.md-authorized tensor/offline minimum; it does **not** yet roll the two interventions through the remaining sampler steps or decode renders, so image-quality improvement is not claimed.

## Implementation

- `pixal3d/modules/sparse/attention/modules.py`: optional runtime processor after final Q/K normalization+RoPE and attention kernel, before the original `to_out`.
- `pixal3d/experiments/global_attention_routing.py`: verified unique scatter, mean R, P, both variants, and sampled explicit-attention diagnostics.
- `pixal3d_c64_topology_c256_value_probe.py`: synchronized real-weight CUDA4 probe.

Actual block order is AdaLN/modulation → self-attention → gate/residual → projected image attention → MLP. Texture shape concatenation occurs before `input_layer`. Runtime model: 30 blocks, hidden 1536, 12 heads × 128, bfloat16, sparse RoPE, QK RMSNorm.

## Runtime tensors

```json
{json.dumps(runtime, indent=2)}
```

## Correspondence

```json
{json.dumps(mapping_stats, indent=2)}
```

`fine // 4` was not assumed: it misses {mapping_stats['fine_div4_exact_support_misses']} fine rows because that coarse voxel is absent from the independently generated baseline support. The total mapping uses nearest support in canonical voxel-centre coordinates. The current hard C64-cube implementation has coverage exactly once, so overlap never reweights a global point.

## Attention/message diagnosis

For 512 deterministic coarse queries, baseline routing places mean `{runtime['outside_tile_attention_ratio_mean']:.6f}` of attention mass outside the corresponding C256 spatial cube. The intervention correction RMS is `{runtime['global_correction_rms']:.6f}` versus local-message RMS `{runtime['local_message_rms']:.6f}`. The within-parent residual reduces back to `{runtime['within_parent_residual_reduction_rms']:.6g}` RMS, numerically checking `(I-PR)`.

The z-low/z-high figures are coordinate-stratified diagnostics, not visibility labels; this minimal pass does not introduce a front/back mask.

## Quantitative/visual results

No PSNR/SSIM/LPIPS or variant render is reported in this offline phase. Existing baseline and original tiled renders are not sufficient to score Replace/residual variants. The next conclusive phase must inject each variant at this block, finish identical Euler trajectories, decode with the unchanged decoder, and render fixed views.

## Conclusion

**Hypothesis:** Does restoring baseline-derived global self-attention routing improve C256 tiled texture flow?

**Result: Partially supported (mechanism only, image-quality claim unresolved).** Baseline routing demonstrably carries nonzero mass across tile boundaries and produces a nontrivial fine-value correction while preserving a zero-mean within-parent residual. This establishes the missing-communication mechanism but is not evidence yet that final texture renders improve.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(out), "runtime": runtime, "mapping": mapping_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
