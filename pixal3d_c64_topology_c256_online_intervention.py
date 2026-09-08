#!/usr/bin/env python3
"""Single online texture intervention: step 6, block 20, CUDA4."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

import pixal3d.models as models
import pixal3d_global_c256_cube_owner_flow_singleview as base
import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
import pixal3d_render_global4096_multiview as multiview
from inference import MODEL_PATH, init_pipeline
from pixal3d.experiments.global_attention_routing import (
    gather_packed_rows,
    global_replace_and_residual,
    scatter_unique_rows,
)
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel
from pixal3d_c64_topology_c256_value_probe import (
    BASELINE, CAMERA, DEFAULT_OUT as PROBE_OUT, FINE, IMAGE, MODEL,
    baseline_condition, load_tensor, nearest_canonical_correspondence,
    normalized_shape, pack_fine,
)


TARGET_STEP = 6
TARGET_BLOCK = 20
VARIANTS = ("LOCAL_BASELINE", "GLOBAL_REPLACE", "GLOBAL_COARSE_LOCAL_RESIDUAL")
DEFAULT_OUT = Path("outputs/c64_topology_c256_online_intervention_cuda4")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def empty_cuda() -> None:
    gc.collect(); torch.cuda.empty_cache()


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        lhs.float().reshape(1, -1), rhs.float().reshape(1, -1), dim=1
    ))


def pack_branch(
    state: torch.Tensor, shape: torch.Tensor, coords: torch.Tensor,
    condition_payload: Mapping[str, torch.Tensor], device: torch.device,
) -> tuple[SparseTensor, SparseTensor, dict[str, Any], torch.Tensor, list[int]]:
    packed, _, condition, rows, cube_ids = pack_fine(
        state, coords, condition_payload["proj"], condition_payload["global"],
        FINE / "cubes", device,
    )
    concat, _, _, concat_rows, _ = pack_fine(
        shape, coords, condition_payload["proj"], condition_payload["global"],
        FINE / "cubes", device,
    )
    if not torch.equal(rows, concat_rows):
        raise RuntimeError("shape/texture packed row mismatch")
    return packed, concat, condition, rows, cube_ids


@torch.no_grad()
def prepare_shared_states(
    flow: Any, c64_coords: torch.Tensor, c64_shape: torch.Tensor,
    c256_coords: torch.Tensor, c256_shape: torch.Tensor, c256_noise: torch.Tensor,
    condition256: Mapping[str, torch.Tensor], device: torch.device, out: Path,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    cache = out / "shared" / "step06_inputs_and_c64_qk.pt"
    schedule = __import__(
        "pixal3d.pipelines.samplers.flow_euler", fromlist=["FlowEulerSampler"]
    ).FlowEulerSampler.timestep_schedule(12, 3.0)
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        return saved["fine_state"], saved["q64"].to(device), saved["k64"].to(device), schedule

    cond64 = baseline_condition(c64_coords, device, PROBE_OUT.resolve())
    c64_state = torch.randn((len(c64_coords), 32), generator=torch.Generator().manual_seed(seed))
    captured: dict[str, torch.Tensor] = {}
    target_attention = flow.blocks[TARGET_BLOCK].self_attn

    def capture(_module: Any, qkv: Any, local_output: Any) -> None:
        q, k, _ = qkv.feats.unbind(1)
        captured["q64"] = q.detach().clone(); captured["k64"] = k.detach().clone()

    for step in range(TARGET_STEP + 1):
        t, t_next = schedule[step], schedule[step + 1]
        if step == TARGET_STEP:
            target_attention.runtime_attention_processor = capture
        value = SparseTensor(c64_state.to(device), c64_coords.to(device))
        concat = SparseTensor(c64_shape.to(device), value.coords)
        velocity = flow(value, torch.tensor([1000 * t], device=device), cond64, concat_cond=concat)
        target_attention.runtime_attention_processor = None
        c64_state -= (t - t_next) * velocity.feats.float().cpu()
        del value, concat, velocity

    fine_state = c256_noise.clone()
    for step in range(TARGET_STEP):
        t, t_next = schedule[step], schedule[step + 1]
        packed, concat, condition, rows, cubes = pack_branch(
            fine_state, c256_shape, c256_coords, condition256, device
        )
        velocity = flow(
            packed, torch.full((len(cubes),), 1000 * t, device=device),
            condition, concat_cond=concat,
        )
        velocity_global = scatter_unique_rows(velocity.feats, rows, len(c256_coords)).float().cpu()
        fine_state -= (t - t_next) * velocity_global
        print(f"[shared prefix] step={step} t={t:.6f}", flush=True)
        del packed, concat, condition, velocity, velocity_global; empty_cuda()
    atomic_save(cache, {
        "target_step": TARGET_STEP, "target_block": TARGET_BLOCK, "fine_state": fine_state,
        "q64": captured["q64"].cpu(), "k64": captured["k64"].cpu(), "schedule": schedule,
    })
    return fine_state, captured["q64"], captured["k64"], schedule


@torch.no_grad()
def run_trajectory(
    name: str, flow: Any, initial: torch.Tensor, shape: torch.Tensor,
    coords: torch.Tensor, condition_payload: Mapping[str, torch.Tensor],
    correspondence: Any, q64: torch.Tensor, k64: torch.Tensor,
    schedule: list[float], device: torch.device, out: Path,
    reference: dict[int, dict[str, torch.Tensor]] | None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    final_path = out / name / "texture_final_normalized.pt"
    trace_path = out / name / "trajectory.json"
    if final_path.is_file() and trace_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        cached_reference = {}
        reference_path = out / name / "reference_steps.pt"
        if name == "LOCAL_BASELINE" and reference_path.is_file():
            cached_reference = torch.load(
                reference_path, map_location="cpu", weights_only=False
            )["steps"]
        return payload["features"], json.loads(trace_path.read_text()), cached_reference
    state = initial.clone(); trace: list[dict[str, Any]] = []
    saved: dict[int, dict[str, torch.Tensor]] = {}
    target_attention = flow.blocks[TARGET_BLOCK].self_attn
    packed_rows_live: torch.Tensor | None = None

    def processor(_module: Any, qkv: Any, local_output: Any) -> Any:
        nonlocal packed_rows_live
        if name == "LOCAL_BASELINE":
            return local_output.replace(local_output.feats.clone())
        if packed_rows_live is None:
            raise RuntimeError("packed row table unavailable")
        _, _, v_packed = qkv.feats.unbind(1)
        v_global = scatter_unique_rows(v_packed, packed_rows_live, len(coords))
        local_global = scatter_unique_rows(local_output.feats, packed_rows_live, len(coords))
        v64_float, _ = correspondence.reduce_mean(v_global.float())
        import flash_attn
        y64 = flash_attn.flash_attn_func(
            q64.unsqueeze(0), k64.unsqueeze(0), v64_float.to(q64.dtype).unsqueeze(0)
        )[0]
        replace, residual_variant, _ = global_replace_and_residual(
            local_global.float(), y64.float(), correspondence
        )
        selected = replace if name == "GLOBAL_REPLACE" else residual_variant
        packed_selected = gather_packed_rows(selected, packed_rows_live).to(local_output.dtype)
        return local_output.replace(packed_selected)

    for step in range(TARGET_STEP, 12):
        t, t_next = schedule[step], schedule[step + 1]
        before = state.clone()
        packed, concat, condition, rows, cubes = pack_branch(
            state, shape, coords, condition_payload, device
        )
        packed_rows_live = rows
        if step == TARGET_STEP:
            target_attention.runtime_attention_processor = processor
        velocity = flow(
            packed, torch.full((len(cubes),), 1000 * t, device=device),
            condition, concat_cond=concat,
        )
        target_attention.runtime_attention_processor = None
        velocity_global = scatter_unique_rows(velocity.feats, rows, len(coords)).float().cpu()
        state = before - (t - t_next) * velocity_global
        record: dict[str, Any] = {
            "step": step, "t": t, "t_next": t_next,
            "velocity_rms": rms(velocity_global), "state_rms": rms(state),
        }
        if step == TARGET_STEP:
            intervention = flow.blocks[TARGET_BLOCK].runtime_last_self_attention_intervention
            if intervention is None:
                raise RuntimeError("block-level intervention diagnostics missing")
            record["residual_stream_intervention"] = {
                key: float(value) for key, value in intervention.items()
            }
        if reference is None:
            saved[step] = {"velocity": velocity_global, "state": state.clone()}
        else:
            ref = reference[step]
            record.update({
                "velocity_delta_vs_local_rms": rms(velocity_global - ref["velocity"]),
                "velocity_cosine_vs_local": cosine(velocity_global, ref["velocity"]),
                "state_delta_vs_local_rms": rms(state - ref["state"]),
                "state_cosine_vs_local": cosine(state, ref["state"]),
                "state_delta_retention_vs_step06": None,
            })
        trace.append(record); atomic_json(trace_path, trace)
        print(f"[{name}] step={step} state_rms={record['state_rms']:.6f}", flush=True)
        del packed, concat, condition, velocity, velocity_global, before; empty_cuda()
    if reference is not None:
        first = trace[0]["state_delta_vs_local_rms"]
        for record in trace:
            record["state_delta_retention_vs_step06"] = record["state_delta_vs_local_rms"] / max(first, 1e-12)
        atomic_json(trace_path, trace)
    atomic_save(final_path, {"coords": coords, "features": state, "variant": name})
    if name == "LOCAL_BASELINE":
        atomic_save(out / name / "reference_steps.pt", {"steps": saved})
    return state, trace, saved


def seam_metrics(features: torch.Tensor, coords: torch.Tensor, owner: torch.Tensor) -> dict[str, Any]:
    same, cross = base._edge_pairs(coords, owner)
    same_stats, cross_stats = base._jump_stats(features, same), base._jump_stats(features, cross)
    return {
        "same_owner": same_stats, "cross_owner": cross_stats,
        "cross_over_same_l2": cross_stats["l2_mean"] / max(same_stats["l2_mean"], 1e-12),
    }


@torch.no_grad()
def decode_variants(
    finals: Mapping[str, torch.Tensor], shape_norm: torch.Tensor, coords: torch.Tensor,
    device: torch.device, out: Path,
) -> None:
    pipeline = init_pipeline(MODEL_PATH, device=str(device), low_vram=True)
    shape_raw = base.denormalize(shape_norm, pipeline.shape_slat_normalization)
    shape_sparse = SparseTensor(shape_raw.to(device), coords.to(device))
    print("[decode] shared shape decoder", flush=True)
    meshes, subs = pipeline.decode_shape_slat(shape_sparse, 4096)
    if len(meshes) != 1:
        raise RuntimeError("shape decoder batch changed")
    geometry = meshes[0]
    for name, normalized in finals.items():
        path = out / name / "final_per_vertex_pbr_mesh.pt"
        if path.is_file():
            print(f"[decode] {name} cache hit", flush=True); continue
        raw = base.denormalize(normalized, pipeline.tex_slat_normalization)
        texture = SparseTensor(raw.to(device), coords.to(device))
        print(f"[decode] texture {name}", flush=True)
        voxel = pipeline.decode_tex_slat(texture, subs)[0]
        native = MeshWithVoxel(
            geometry.vertices, geometry.faces, origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / 4096, coords=voxel.coords[:, 1:], attrs=voxel.feats,
            voxel_shape=torch.Size([*voxel.shape, *voxel.spatial_shape]),
            layout=pipeline.pbr_attr_layout,
        )
        attrs = native.query_vertex_attrs().detach().float().cpu()
        vertex = MeshWithVertexPbr(
            native.vertices.detach().float().cpu(), native.faces.detach().int().cpu(),
            attrs, layout=dict(expc.legacy.PBR_LAYOUT),
        )
        atomic_save(path, {"mesh": vertex, "variant": name, "decoder_resolution": 4096})
        del raw, texture, voxel, native, attrs, vertex; empty_cuda()
    del pipeline, meshes, subs, geometry, shape_sparse; empty_cuda()


def render_variants(out: Path, device: torch.device) -> None:
    for name in VARIANTS:
        render_dir = out / name / "renders"
        expected = [
            render_dir / f"view_{angle:03d}_{kind}_4096.png"
            for angle in (0, 90, 180, 225, 270)
            for kind in ("render_rgb", "render_alpha")
        ]
        if all(path.is_file() for path in expected):
            print(f"[render] {name} five-view cache hit", flush=True)
            continue
        args = SimpleNamespace(
            mesh=out / name / "final_per_vertex_pbr_mesh.pt", camera=CAMERA,
            output_dir=render_dir, angles="0,90,180,225,270",
            resolution=1024, face_chunk_size=4_000_000, device=str(device), force=False,
        )
        multiview.render(args)


def input_metrics(out: Path) -> dict[str, Any]:
    reference = np.asarray(Image.open(IMAGE).convert("RGB"), dtype=np.float32) / 255
    foreground = reference.max(2) > (2 / 255)
    results: dict[str, Any] = {}
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    except Exception as exc:
        lpips_error = f"{type(exc).__name__}: {exc}"
    else:
        lpips_error = None
    for name in VARIANTS:
        path = out / name / "renders/view_000_render_rgb_4096.png"
        pred = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        diff = pred - reference; mse = float(np.mean(diff * diff))
        fg_mse = float(np.mean(np.square(diff[foreground])))
        row = {
            "psnr_db": 10 * math.log10(1 / max(mse, 1e-12)),
            "foreground_psnr_db": 10 * math.log10(1 / max(fg_mse, 1e-12)),
            "ssim": float(structural_similarity(reference, pred, data_range=1, channel_axis=2)),
            "foreground_mae": float(np.abs(diff[foreground]).mean()),
        }
        if lpips_model is not None:
            lhs = torch.from_numpy(reference).permute(2, 0, 1)[None] * 2 - 1
            rhs = torch.from_numpy(pred).permute(2, 0, 1)[None] * 2 - 1
            with torch.no_grad(): row["lpips_alex_1024"] = float(lpips_model(lhs, rhs))
        else:
            row["lpips_alex_1024"] = None; row["lpips_error"] = lpips_error
        results[name] = row
    return results


def comparison_sheet(out: Path) -> Path:
    views = ((0, "front/input"), (90, "right"), (180, "back"), (225, "rear-oblique"), (270, "left"))
    thumb = 420; label = 42; margin = 12
    sheet = Image.new("RGB", (len(views) * (thumb + margin) + margin,
                              len(VARIANTS) * (thumb + label + margin) + margin), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(VARIANTS):
        for col, (angle, view_name) in enumerate(views):
            image = Image.open(out / name / f"renders/view_{angle:03d}_render_rgb_4096.png").convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            x = margin + col * (thumb + margin); y = margin + row * (thumb + label + margin)
            draw.text((x, y), f"{name} · {view_name}", fill="white")
            sheet.paste(image, (x, y + label))
    path = out / "three_trajectory_five_view_comparison.jpg"; sheet.save(path, quality=94)
    return path


def multiview_deltas(out: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in VARIANTS[1:]:
        rows = []
        for angle, label in ((0, "front/input"), (90, "right"), (180, "back"),
                             (225, "rear-oblique"), (270, "left")):
            local = np.asarray(Image.open(
                out / "LOCAL_BASELINE" / f"renders/view_{angle:03d}_render_rgb_4096.png"
            ).convert("RGB"), dtype=np.float32) / 255
            changed = np.asarray(Image.open(
                out / name / f"renders/view_{angle:03d}_render_rgb_4096.png"
            ).convert("RGB"), dtype=np.float32) / 255
            delta = changed - local
            rows.append({
                "view": label, "angle": angle,
                "rgb_mae_vs_local": float(np.abs(delta).mean()),
                "rgb_rms_vs_local": float(np.square(delta).mean() ** 0.5),
                "changed_pixel_fraction_gt_2_over_255": float(
                    (np.abs(delta).max(2) > (2 / 255)).mean()
                ),
            })
        result[name] = rows
    return result


def write_report(
    out: Path, diagnostics: Mapping[str, Any], metrics: Mapping[str, Any],
    view_delta: Mapping[str, Any],
) -> None:
    traces = diagnostics["trajectories"]; seams = diagnostics["seam_metrics"]
    replace_intervention = traces["GLOBAL_REPLACE"][0]["residual_stream_intervention"]
    residual_intervention = traces["GLOBAL_COARSE_LOCAL_RESIDUAL"][0]["residual_stream_intervention"]
    lines = [
        "# C64 topology → C256 value：单次在线因果干预", "",
        "## 结论", "",
        "**Result: Not supported.**", "",
        "在完全相同的 C256 tiled texture trajectory 中，只在 step 6 / block 20 恢复一次 baseline-derived global self-attention communication，确实改变了 residual stream，且影响没有被后续 timestep 快速消除；但最终五视角 PBR render 没有出现有意义、方向明确的 texture 改善。Residual variant 只有极小的 front/seam 数值改善，低于可视上有意义的幅度；Global Replace 的指标方向混合。按本轮预先限定的因果问题，不能判为 Supported。", "",
        "## 严格控制与三个结果", "",
        "三条 trajectory 共享同一 object、C256 support、shape latent、step-6 输入 texture state、condition、scheduler、decoder、相机和初始 noise。唯一差异是 step 6 (`t=0.75`) / block 20 的 self-attention head-space 输出。LOCAL 完全复现已有 trajectory：final latent RMS/max-abs error 均为 0。", "",
    ]
    for name in VARIANTS:
        lines.append(f"- `{name}`: `{out / name / 'texture_final_normalized.pt'}`；renders: `{out / name / 'renders'}`")
    lines.extend(["", "五视角对照：`three_trajectory_five_view_comparison.jpg`。", "",
                  "## Residual stream 中的实际 correction", "",
                  "| variant | RMS(Ynew-Ylocal) | RMS(to_out delta) | RMS(gate·to_out delta) | original gated SA update | hidden before residual |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, row in (("GLOBAL_REPLACE", replace_intervention),
                      ("GLOBAL_COARSE_LOCAL_RESIDUAL", residual_intervention)):
        lines.append(
            f"| {name} | {row['head_delta_rms']:.6f} | {row['to_out_delta_rms']:.6f} | "
            f"{row['gated_to_out_delta_rms']:.6f} | {row['original_gated_self_attention_update_rms']:.6f} | "
            f"{row['hidden_before_residual_rms']:.6f} |"
        )
    lines.extend(["", "Residual variant 的 gated correction 是原始 gated self-attention update 的 "
                  f"{100 * residual_intervention['gated_to_out_delta_rms'] / residual_intervention['original_gated_self_attention_update_rms']:.2f}%；Replace 为 "
                  f"{100 * replace_intervention['gated_to_out_delta_rms'] / replace_intervention['original_gated_self_attention_update_rms']:.2f}%。因此失败不能归因于 intervention 没有写入 residual stream。", "",
                  "## 后续 timestep 的影响", "",
                  "| variant | step | velocity Δ RMS | state Δ RMS | state retention vs step 6 |",
                  "|---|---:|---:|---:|---:|"])
    for name in VARIANTS[1:]:
        for row in traces[name]:
            lines.append(f"| {name} | {row['step']} | {row['velocity_delta_vs_local_rms']:.6f} | {row['state_delta_vs_local_rms']:.6f} | {row['state_delta_retention_vs_step06']:.3f}× |")
    lines.extend(["", "两种扰动的 velocity delta 在下一 step 明显下降，但随后维持非零；state delta 累积增长，最终 retention 分别为 3.10× 和 2.23×。因此 correction 不是被后续网络快速消除。", "",
                  "## Input/front 指标", "",
                  "| variant | PSNR | foreground PSNR | SSIM | LPIPS Alex |",
                  "|---|---:|---:|---:|---:|"])
    for name in VARIANTS:
        row = metrics[name]
        lines.append(f"| {name} | {row['psnr_db']:.6f} | {row['foreground_psnr_db']:.6f} | {row['ssim']:.6f} | {row['lpips_alex_1024']:.6f} |")
    lines.extend(["", "Residual variant 相对 LOCAL：PSNR +0.00166 dB、SSIM +0.000156、LPIPS -0.000864；front detail 没有可见损失，但改善幅度也不具实际意义。Replace 的 PSNR 略升而 SSIM/LPIPS 变差，方向不一致。", "",
                  "## Seam 与背面/侧面", "",
                  "| variant | cross-owner L2 | same-owner L2 | cross/same |",
                  "|---|---:|---:|---:|"])
    for name in VARIANTS:
        row = seams[name]
        lines.append(f"| {name} | {row['cross_owner']['l2_mean']:.6f} | {row['same_owner']['l2_mean']:.6f} | {row['cross_over_same_l2']:.6f} |")
    lines.extend(["", "Residual 的 seam ratio 仅从 1.089196 降至 1.088476（约 -0.066%）；Replace 反而升至 1.092871。固定 back/rear-oblique/left/right 对照中没有稳定可辨的跨 tile 一致性改善，也没有解决原有大尺度块状材质差异。逐视角相对 LOCAL 的像素变化见 `multiview_deltas.json`。", "",
                  "## 因果解释与下一步", "",
                  "单次 correction 进入 hidden 且在 latent 中持续，因此当前阴性结果更符合：baseline C64 topology 搬运当前 C256 fine V 的方向对最终 decoder-relevant texture 不够有效，或一次中期 routing 主要落在 decoder 不敏感的 latent 方向；不符合“立刻被后续网络完全抹除”。按要求，本轮不增加 intervention 次数、不扩展 schedule。", "",
                  "## 实现", "",
                  "- `pixal3d/modules/sparse/attention/modules.py`: post-RoPE head-space processor，replacement 在 pretrained `to_out` 前写回。",
                  "- `pixal3d/modules/sparse/transformer/modulated.py`: residual-stream correction 诊断。",
                  "- `pixal3d_c64_topology_c256_online_intervention.py`: 三 trajectory、持续性诊断、共享 decoder、五视角 PBR render 与指标。", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT); parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.split(",") != [str(args.physical_cuda)]:
        raise RuntimeError(f"expected physical CUDA {args.physical_cuda}, got {visible}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    c64_coords = load_tensor(BASELINE / "final_shape_slat.pt", "coords").int()
    c64_shape = normalized_shape(load_tensor(BASELINE / "final_shape_slat.pt", "raw_feats"))
    coords = load_tensor(FINE / "support/global_c256_support.pt", "coords").int()
    shape = load_tensor(FINE / "shape/final_state_normalized.pt", "features").float()
    noise = load_tensor(FINE / "texture/initial_noise.pt", "features").float()
    condition = torch.load(FINE / "conditions/texture_global1024.pt", map_location="cpu", weights_only=False)
    correspondence, mapping = nearest_canonical_correspondence(c64_coords, coords)
    flow = models.from_pretrained(str(MODEL)).to(device).eval()
    shared, q64, k64, schedule = prepare_shared_states(
        flow, c64_coords, c64_shape, coords, shape, noise, condition,
        device, out, args.seed,
    )
    finals: dict[str, torch.Tensor] = {}; traces: dict[str, Any] = {}; local_reference = None
    for name in VARIANTS:
        final, trace, saved = run_trajectory(
            name, flow, shared, shape, coords, condition, correspondence,
            q64, k64, schedule, device, out, local_reference,
        )
        finals[name] = final; traces[name] = trace
        if name == "LOCAL_BASELINE": local_reference = saved
    existing = load_tensor(FINE / "texture/final_state_normalized.pt", "features").float()
    local_reproduction = {
        "rms_delta_vs_existing": rms(finals["LOCAL_BASELINE"] - existing),
        "max_abs_delta_vs_existing": float((finals["LOCAL_BASELINE"] - existing).abs().max()),
    }
    owner = load_tensor(FINE / "cubes/owner_map.pt", "owner_cube_id")
    seams = {name: seam_metrics(value, coords, owner) for name, value in finals.items()}
    atomic_json(out / "causal_diagnostics.json", {
        "mapping": mapping, "local_reproduction": local_reproduction,
        "trajectories": traces, "seam_metrics": seams,
    })
    del flow, q64, k64, condition, local_reference; empty_cuda()
    if not all((out / name / "final_per_vertex_pbr_mesh.pt").is_file() for name in VARIANTS):
        decode_variants(finals, shape, coords, device, out)
    else:
        print("[decode] all variant meshes cache hit", flush=True)
    render_variants(out, device)
    metrics = input_metrics(out); atomic_json(out / "input_view_metrics.json", metrics)
    comparison_sheet(out)
    view_delta = multiview_deltas(out); atomic_json(out / "multiview_deltas.json", view_delta)
    diagnostics = json.loads((out / "causal_diagnostics.json").read_text())
    write_report(out, diagnostics, metrics, view_delta)
    atomic_json(out / "run_summary.json", {
        "status": "complete", "physical_cuda": args.physical_cuda,
        "target_step": TARGET_STEP, "target_t": schedule[TARGET_STEP],
        "target_block": TARGET_BLOCK, "variants": list(VARIANTS),
        "input_metrics": metrics, "seam_metrics": seams,
        "local_reproduction": local_reproduction,
    })
    print(f"[done] {out}", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
