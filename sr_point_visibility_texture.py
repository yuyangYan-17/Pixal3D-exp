#!/usr/bin/env python3
"""Point-routed visible/hidden texture flow experiments.

Every forward retains the full local context and homogeneous image conditions.
Above the visibility threshold the whole block uses conditional flow; otherwise:

* visible owner points use image-conditional texture flow;
* hidden owner points use a clean baseline endpoint for the first ``n`` steps;
* the same hidden points use image-unconditional texture flow afterwards.

The two model branches are evaluated separately for mixed blocks, then owner
velocities are reduced into one synchronized global update.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/sr_0_img_trilinear_face_n3_20260919/fixed"),
        help="prepared fixed-geometry visibility root",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sr_point_visibility_20260919"),
    )
    p.add_argument("--gpu", default="1")
    p.add_argument("--seeds", default="46")
    p.add_argument("--n-values", default="0,1,2,3,4")
    p.add_argument("--visibility-threshold", type=float, default=0.30)
    p.add_argument("--all-conditional", action="store_true",
                   help="explicit all-conditional control; n=0 alone still routes unseen points")
    p.add_argument("--render-resolution", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=58)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--only-flow", action="store_true")
    p.add_argument(
        "--hidden-mode",
        choices=("unconditional", "global_only"),
        default="unconditional",
        help="hidden-point branch after the endpoint-guide prefix",
    )
    p.add_argument(
        "--anchor-visible",
        action="store_true",
        help="keep visible-point velocity on the matching all-conditional trajectory",
    )
    p.add_argument(
        "--anchor-dilation",
        type=int,
        default=0,
        help="C256 Manhattan-radius dilation of the visible anchor mask",
    )
    p.add_argument(
        "--visible-context-only",
        action="store_true",
        help=(
            "conditional branch keeps full sparse context but zeros local image "
            "projection features on hidden context rows"
        ),
    )
    p.add_argument(
        "--visible-context-start",
        type=int,
        default=-1,
        help="start step for visible-context-only masking; -1 means every step",
    )
    p.add_argument(
        "--visible-context-hidden-weight",
        type=float,
        default=None,
        help="retain this fraction of hidden local projection features when vctx is active",
    )
    p.add_argument(
        "--trim-conditional-context",
        action="store_true",
        help="conditional branch drops hidden context rows instead of only masking proj",
    )
    p.add_argument(
        "--trim-conditional-context-start",
        type=int,
        default=-1,
        help="start step for conditional-context trimming; -1 means every step",
    )
    p.add_argument(
        "--reuse-prefix",
        type=Path,
        default=None,
        help="reuse a compatible saved texture-flow prefix instead of recomputing it",
    )
    p.add_argument(
        "--reuse-prefix-step",
        type=int,
        default=3,
        help="last zero-based checkpoint step included in --reuse-prefix",
    )
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = (
    f"/tmp/flex_gemm_point_visibility_20260919_gpu{ARGS.gpu}.json"
)
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "0"

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from sr_tools import common, rendering, shape as sync, texture


VERSION = "block_threshold_point_selection_v2"


def parse_ints(value: str):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def dilate_visible_mask(visible, coords, radius):
    """Dilate visible C256 support with a sparse Manhattan neighborhood."""
    radius = int(radius)
    if radius <= 0:
        return visible.clone()
    xyz = coords[:, 1:].long().cpu()
    grid = 256
    keys = ((xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]).tolist()
    active = set(k for k, flag in zip(keys, visible.tolist()) if flag)
    for _ in range(radius):
        expanded = set(active)
        for key in active:
            x = key // (grid * grid)
            rem = key % (grid * grid)
            y = rem // grid
            z = rem % grid
            if x > 0:
                expanded.add(key - grid * grid)
            if x + 1 < grid:
                expanded.add(key + grid * grid)
            if y > 0:
                expanded.add(key - grid)
            if y + 1 < grid:
                expanded.add(key + grid)
            if z > 0:
                expanded.add(key - 1)
            if z + 1 < grid:
                expanded.add(key + 1)
        active = expanded
    return torch.tensor([key in active for key in keys], dtype=torch.bool)


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, payload):
    common.js(path, payload)


def load_sparse(path: Path):
    return common.load_payload(path)


def branch_view(
    block,
    owner_mask,
    branch,
    condition_mask=None,
    condition_weight=0.0,
    trim_context=False,
):
    """Keep all context rows, route only owner points in ``owner_mask``."""
    view = dict(block)
    if trim_context:
        view["global_ids"] = block["global_ids"][owner_mask]
        view["coords"] = block["coords"][owner_mask]
        view["rows"] = block["rows"][owner_mask]
        view["owned"] = block["owned"][owner_mask]
    else:
        view["owned"] = block["owned"] & owner_mask
    view["branch"] = branch
    view["source_block_id"] = int(block["block_id"])
    if condition_mask is not None:
        view["condition_mask"] = condition_mask[block["global_ids"]].clone()
        if trim_context:
            view["condition_mask"] = view["condition_mask"][owner_mask]
        view["condition_weight"] = float(condition_weight)
    return view


def routed_blocks(
    data, visible, coverage, n, step, guide_threshold=1.0,
    visible_context_only=False, visible_context_hidden_weight=0.0,
    trim_conditional_context=False,
):
    """Use separate full-context forwards, then select owner velocities."""
    from train_method.texture_block_routing import route_blocks
    if visible_context_only or trim_conditional_context:
        raise ValueError("block routing requires full, unmasked conditional context")
    routes, stats = route_blocks(
        data, visible, coverage, step if step < n else None,
        threshold=ARGS.visibility_threshold, guide_threshold=guide_threshold,
        all_conditional=ARGS.all_conditional,
    )
    routes.update(stats)
    routes["visible_owner_points"] = sum(
        int((visible[b["global_ids"]] & b["owned"]).sum()) for b in data["blocks"]
    )
    routes["hidden_owner_points"] = sum(
        int((~visible[b["global_ids"]] & b["owned"]).sum()) for b in data["blocks"]
    )
    routes["guided_owner_points"] = sum(int(b["owned"].sum()) for b in routes["guided"])
    return routes

def branch_groups(branches, batch_size, max_tokens):
    return {
        name: list(sync.groups(branches[name], batch_size, max_tokens))
        for name in ("conditional", "unconditional", "guided")
    }


@torch.no_grad()
def guided_values(sampler, group, x, guides, ids, t, guide_step):
    guide = guides[guide_step]
    return [
        sampler._xstart_to_pred(x[b["global_ids"]], t, guide[b["global_ids"]]).float()
        for b in group
    ]


@torch.no_grad()
def run_flow(
    pipe,
    data,
    shape,
    bank,
    guides,
    visible,
    coverage,
    out: Path,
    seed: int,
    n: int,
    batch_size: int,
    max_tokens: int,
    hidden_mode: str,
    visible_context_only: bool,
    visible_context_start: int,
    visible_context_hidden_weight: float,
    trim_conditional_context: bool,
    trim_conditional_context_start: int,
    reuse_prefix: Path | None,
    reuse_prefix_step: int,
):
    out.mkdir(parents=True, exist_ok=True)
    endpoint = out / "endpoint.pt"
    model = pipe.models["tex_slat_flow_model_1024"]
    sampler = pipe.tex_slat_sampler
    params = dict(pipe.tex_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(12, params.pop("rescale_t", 1.0))
    channels = model.in_channels - shape.shape[1]
    if channels <= 0:
        raise RuntimeError("texture flow has no texture-only channels")
    start_step = 0
    if reuse_prefix is not None:
        reuse_prefix = reuse_prefix.resolve()
        prefix_initial = common.load_payload(reuse_prefix / "initial.pt")
        prefix_checkpoint = common.load_payload(
            reuse_prefix / f"step_{reuse_prefix_step:02d}.pt"
        )
        if (
            not torch.equal(prefix_initial["coords"], data["coords"])
            or not torch.equal(prefix_checkpoint["coords"], data["coords"])
            or prefix_initial.get("seed") != int(seed)
            or prefix_checkpoint.get("seed") != int(seed)
            or prefix_checkpoint.get("n") != int(n)
        ):
            raise RuntimeError("reuse prefix is incompatible with this flow variant")
        x = prefix_checkpoint["features"].clone()
        noise = prefix_initial["noise"]
        start_step = int(reuse_prefix_step) + 1
    else:
        noise = torch.randn(
            (len(data["coords"]), channels),
            generator=torch.Generator().manual_seed(int(seed)),
        )
        x = noise.clone()
    shape_hash = sync.tensor_hash(shape)
    identity = dict(
        version=VERSION,
        visibility_threshold=ARGS.visibility_threshold,
        all_conditional=ARGS.all_conditional,
        seed=int(seed),
        n=int(n),
        steps=12,
        times=times,
        shape_hash=shape_hash,
        batch_size=int(batch_size),
        max_tokens=int(max_tokens),
        routing=(
            "high visibility block -> conditional; low mixed block -> separate full-context "
            "forwards and point selection; hidden -> baseline prefix then unconditional"
        ),
        hidden_mode=hidden_mode,
        visible_context_only=bool(visible_context_only),
        visible_context_start=int(visible_context_start),
        visible_context_hidden_weight=float(visible_context_hidden_weight),
        trim_conditional_context=bool(trim_conditional_context),
        trim_conditional_context_start=int(trim_conditional_context_start),
        reuse_prefix=str(reuse_prefix) if reuse_prefix is not None else None,
        reuse_prefix_step=int(reuse_prefix_step),
    )
    manifest = out / "sampler.json"
    if manifest.exists():
        cached = load_json(manifest)
        if cached != identity:
            raise RuntimeError(f"variant manifest mismatch: {manifest}")
    else:
        save_json(manifest, identity)
    common.save(
        out / "initial.pt",
        coords=data["coords"],
        noise=noise,
        state=x,
        seed=int(seed),
        resume_prefix=str(reuse_prefix) if reuse_prefix is not None else None,
        resume_prefix_step=(
            int(reuse_prefix_step) if reuse_prefix is not None else None
        ),
    )

    if endpoint.exists():
        cached = common.load_payload(endpoint)
        if not torch.equal(cached["coords"], data["coords"]):
            raise RuntimeError("cached endpoint support mismatch")
        return dict(
            coords=cached["coords"],
            features=cached["features"],
            normalized=True,
            elapsed=float(load_json(out / "timing.json")["texture_flow_seconds"]),
            resumed=True,
        )

    started = time.perf_counter()
    model.to(pipe.device)
    try:
        for step, (t, tn) in enumerate(zip(times, times[1:])):
            if step < start_step:
                continue
            checkpoint = out / f"step_{step:02d}.pt"
            input_hash = sync.tensor_hash(x)
            if checkpoint.exists():
                cached = common.load_payload(checkpoint)
                if (
                    not torch.equal(cached["coords"], data["coords"])
                    or cached["input_hash"] != input_hash
                    or cached["seed"] != int(seed)
                    or cached["n"] != int(n)
                ):
                    raise RuntimeError(f"checkpoint mismatch: {checkpoint}")
                x = cached["features"]
                continue

            routes = routed_blocks(
                data,
                visible,
                coverage,
                n,
                step,
                visible_context_only=(
                    visible_context_only
                    and (
                        visible_context_start < 0
                        or step >= visible_context_start
                    )
                ),
                visible_context_hidden_weight=visible_context_hidden_weight,
                trim_conditional_context=(
                    trim_conditional_context
                    and (
                        trim_conditional_context_start < 0
                        or step >= trim_conditional_context_start
                    )
                ),
            )
            grouped = branch_groups(routes, batch_size, max_tokens)
            summed = torch.zeros_like(x)
            counts = torch.zeros(len(x), dtype=torch.int32)
            calls = {k: 0 for k in ("conditional", "unconditional", "guided")}

            for branch in ("conditional", "unconditional"):
                for group in grouped[branch]:
                    values = texture.predict_safe(
                        pipe,
                        model,
                        group,
                        x,
                        shape,
                        bank,
                        t,
                        params,
                        unconditional=(
                            branch == "unconditional"
                            and hidden_mode == "unconditional"
                        ),
                        global_only=(
                            branch == "unconditional"
                            and hidden_mode == "global_only"
                        ),
                    )
                    sync.reduce_predictions(summed, counts, group, values)
                    calls[branch] += 1
            for group in grouped["guided"]:
                values = guided_values(sampler, group, x, guides, data["coords"], t, step)
                sync.reduce_predictions(summed, counts, group, values)
                calls["guided"] += 1

            if not torch.equal(counts, data["counts"]):
                raise RuntimeError(
                    f"owner coverage mismatch at step {step}: "
                    f"got {int(counts.min())}/{int(counts.max())}, "
                    f"expected {int(data['counts'].min())}/{int(data['counts'].max())}"
                )
            if sync.tensor_hash(x) != input_hash:
                raise RuntimeError("flow predictor mutated global state")
            velocity = summed / counts[:, None]
            x = x - (t - tn) * velocity
            if not torch.isfinite(x).all():
                raise RuntimeError("non-finite texture flow state")
            common.save(
                checkpoint,
                coords=data["coords"],
                features=x,
                velocity=velocity,
                input_hash=input_hash,
                seed=int(seed),
                n=int(n),
                t=t,
                t_next=tn,
            )
            save_json(
                out / f"step_{step:02d}.json",
                dict(
                    step=step,
                    t=t,
                    t_next=tn,
                    calls=calls,
                    mixed_blocks=routes["mixed_blocks"],
                    high_blocks=routes["high_blocks"],
                    hidden_blocks=routes["hidden_blocks"],
                    hidden_conditional_owner_occurrences=routes["hidden_conditional_owner_occurrences"],
                    visible_owner_points=routes["visible_owner_points"],
                    hidden_owner_points=routes["hidden_owner_points"],
                    guided_owner_points=routes["guided_owner_points"],
                    output_hash=sync.tensor_hash(x),
                ),
            )
            print(
                f"FLOW seed={seed} n={n} step={step + 1}/12 "
                f"calls={calls} mixed={routes['mixed_blocks']}",
                flush=True,
            )
    finally:
        model.cpu()
        common.empty_cuda()
    elapsed = time.perf_counter() - started
    common.save(endpoint, coords=data["coords"], features=x, normalized=True)
    save_json(out / "timing.json", dict(texture_flow_seconds=elapsed, steps=12))
    return dict(coords=data["coords"], features=x, normalized=True, elapsed=elapsed, resumed=False)


@torch.no_grad()
def run_anchor_visible_flow(
    pipe,
    data,
    shape,
    bank,
    guides,
    visible,
    coverage,
    out: Path,
    reference_out: Path,
    seed: int,
    n: int,
    batch_size: int,
    max_tokens: int,
    hidden_mode: str,
    visible_context_only: bool,
    visible_context_start: int,
    visible_context_hidden_weight: float,
    trim_conditional_context: bool,
    trim_conditional_context_start: int,
    reuse_prefix: Path | None = None,
    reuse_prefix_step: int = 3,
):
    """Route hidden points while anchoring visible points to all-conditional v.

    The all-conditional control has already saved its per-point synchronized
    velocity at every step.  Reusing that velocity for visible owners removes
    the unwanted feedback from hidden-point routing into the observed image
    trajectory.  Hidden owners still use the requested endpoint/unconditional
    schedule and are updated in the same global state.
    """
    if n <= 0:
        raise ValueError("anchor-visible requires n > 0")
    out.mkdir(parents=True, exist_ok=True)
    reference_out = reference_out.resolve()
    if not (reference_out / "initial.pt").exists():
        raise FileNotFoundError(reference_out / "initial.pt")
    model = pipe.models["tex_slat_flow_model_1024"]
    sampler = pipe.tex_slat_sampler
    params = dict(pipe.tex_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(12, params.pop("rescale_t", 1.0))
    ref_initial = common.load_payload(reference_out / "initial.pt")
    if not torch.equal(ref_initial["coords"], data["coords"]):
        raise RuntimeError("anchor reference support mismatch")
    x = ref_initial["state"].clone()
    identity = dict(
        version=VERSION,
        seed=int(seed),
        n=int(n),
        steps=12,
        times=times,
        shape_hash=sync.tensor_hash(shape),
        anchor_visible=True,
        reference=str(reference_out),
        routing=(
            "visible owners use cached all-conditional velocity from the same seed; "
            "hidden owners use endpoint guide for first n steps then image-unconditional"
        ),
        hidden_mode=hidden_mode,
        visible_context_only=bool(visible_context_only),
        visible_context_start=int(visible_context_start),
        visible_context_hidden_weight=float(visible_context_hidden_weight),
        trim_conditional_context=bool(trim_conditional_context),
        trim_conditional_context_start=int(trim_conditional_context_start),
        batch_size=int(batch_size),
        max_tokens=int(max_tokens),
    )
    manifest = out / "sampler.json"
    if manifest.exists():
        cached = load_json(manifest)
        if cached != identity:
            raise RuntimeError(f"variant manifest mismatch: {manifest}")
    else:
        save_json(manifest, identity)
    common.save(out / "initial.pt", coords=data["coords"], noise=x, state=x, seed=int(seed))
    hidden = ~visible
    expected_hidden = data["counts"].clone()
    expected_hidden[visible] = 0
    started = time.perf_counter()
    model.to(pipe.device)
    try:
        for step, (t, tn) in enumerate(zip(times, times[1:])):
            checkpoint = out / f"step_{step:02d}.pt"
            input_hash = sync.tensor_hash(x)
            if checkpoint.exists():
                cached = common.load_payload(checkpoint)
                if (
                    not torch.equal(cached["coords"], data["coords"])
                    or cached["input_hash"] != input_hash
                    or cached["seed"] != int(seed)
                    or cached["n"] != int(n)
                ):
                    raise RuntimeError(f"checkpoint mismatch: {checkpoint}")
                x = cached["features"]
                continue

            routes = routed_blocks(
                data,
                visible,
                coverage,
                n,
                step,
                visible_context_only=(
                    visible_context_only
                    and (
                        visible_context_start < 0
                        or step >= visible_context_start
                    )
                ),
                visible_context_hidden_weight=visible_context_hidden_weight,
                trim_conditional_context=(
                    trim_conditional_context
                    and (
                        trim_conditional_context_start < 0
                        or step >= trim_conditional_context_start
                    )
                ),
            )
            grouped = branch_groups(routes, batch_size, max_tokens)
            summed = torch.zeros_like(x)
            counts = torch.zeros(len(x), dtype=torch.int32)
            calls = {k: 0 for k in ("unconditional", "guided")}
            for group in grouped["unconditional"]:
                values = texture.predict_safe(
                    pipe,
                    model,
                    group,
                    x,
                    shape,
                    bank,
                    t,
                    params,
                    unconditional=hidden_mode == "unconditional",
                    global_only=hidden_mode == "global_only",
                )
                sync.reduce_predictions(summed, counts, group, values)
                calls["unconditional"] += 1
            for group in grouped["guided"]:
                values = guided_values(sampler, group, x, guides, data["coords"], t, step)
                sync.reduce_predictions(summed, counts, group, values)
                calls["guided"] += 1

            if not torch.equal(counts, expected_hidden):
                raise RuntimeError(
                    f"hidden owner coverage mismatch at step {step}: "
                    f"got {int(counts.min())}/{int(counts.max())}, "
                    f"expected hidden {int(expected_hidden.min())}/{int(expected_hidden.max())}"
                )
            ref_step = common.load_payload(reference_out / f"step_{step:02d}.pt")
            if not torch.equal(ref_step["coords"], data["coords"]):
                raise RuntimeError("anchor step support mismatch")
            reference_velocity = ref_step["velocity"]
            velocity = reference_velocity.clone()
            velocity[hidden] = summed[hidden] / counts[hidden, None]
            if sync.tensor_hash(x) != input_hash:
                raise RuntimeError("flow predictor mutated global state")
            x = x - (t - tn) * velocity
            if not torch.isfinite(x).all():
                raise RuntimeError("non-finite anchored texture flow state")
            common.save(
                checkpoint,
                coords=data["coords"],
                features=x,
                velocity=velocity,
                input_hash=input_hash,
                seed=int(seed),
                n=int(n),
                t=t,
                t_next=tn,
            )
            save_json(
                out / f"step_{step:02d}.json",
                dict(
                    step=step,
                    t=t,
                    t_next=tn,
                    calls=calls,
                    mixed_blocks=routes["mixed_blocks"],
                    visible_owner_points=routes["visible_owner_points"],
                    hidden_owner_points=routes["hidden_owner_points"],
                    guided_owner_points=routes["guided_owner_points"],
                    visible_velocity_source="all_conditional_reference",
                    output_hash=sync.tensor_hash(x),
                ),
            )
            print(
                f"ANCHOR FLOW seed={seed} n={n} step={step + 1}/12 "
                f"calls={calls} mixed={routes['mixed_blocks']}",
                flush=True,
            )
    finally:
        model.cpu()
        common.empty_cuda()
    elapsed = time.perf_counter() - started
    common.save(out / "endpoint.pt", coords=data["coords"], features=x, normalized=True)
    save_json(out / "timing.json", dict(texture_flow_seconds=elapsed, steps=12, anchor_visible=True))
    return dict(coords=data["coords"], features=x, normalized=True, elapsed=elapsed, resumed=False)


@torch.no_grad()
def decode_material_memory(pipe, root: Path, payload, texture_features):
    """Decode final material without persisting a multi-GB textured mesh."""
    from pixal3d.representations.mesh import MeshWithVoxel

    topology = common.load_payload(root / "shared/final_mesh/topology.pt")
    geometry = common.load_payload(root / "shared/final_mesh/geometry_mesh.pt")["mesh"]
    tex = common.SparseTensor(
        rendering.norm(pipe, "tex", texture_features.to(pipe.device), inverse=True),
        payload["coords"].to(pipe.device),
    )
    field = pipe.decode_tex_slat(tex, [sub.to(pipe.device) for sub in topology["subs"]])
    if not torch.isfinite(field.feats).all():
        raise RuntimeError("decoded material has non-finite values")
    mesh = MeshWithVoxel(
        geometry.vertices,
        geometry.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 4096,
        coords=field.coords[:, 1:].cpu(),
        attrs=field.feats.float().cpu(),
        voxel_shape=torch.Size([*field.shape, *field.spatial_shape]),
        layout=pipe.pbr_attr_layout,
    )
    del topology, geometry, tex, field
    common.empty_cuda()
    return mesh


def image_metrics(reference, prediction, mask, lpips_model, device):
    ref = np.asarray(reference.convert("RGB"), dtype=np.float64) / 255.0
    pred = np.asarray(prediction.convert("RGB"), dtype=np.float64) / 255.0
    mask = np.asarray(mask) > 127
    mse = float(((ref[mask] - pred[mask]) ** 2).mean())
    _, ssim_map = structural_similarity(
        ref,
        pred,
        data_range=1.0,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        win_size=11,
        use_sample_covariance=False,
        full=True,
    )

    def tensor(image):
        return (
            torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
            .permute(2, 0, 1)[None]
            .to(device)
            / 127.5
            - 1
        )

    distance = lpips_model(tensor(reference), tensor(prediction))[0, 0]
    return dict(
        psnr_db=-10.0 * np.log10(max(mse, 1e-30)),
        ssim=float(ssim_map[mask].mean()),
        lpips=float(distance[mask].mean().detach().cpu()),
    )


def pair_metrics(first, second, lpips_model, device):
    """Back-view agreement with baseline; not a ground-truth quality score."""
    ref = np.asarray(first.convert("RGB"), dtype=np.float64) / 255.0
    pred = np.asarray(second.convert("RGB"), dtype=np.float64) / 255.0
    mse = float(((ref - pred) ** 2).mean())
    _, ssim_map = structural_similarity(
        ref,
        pred,
        data_range=1.0,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        win_size=11,
        use_sample_covariance=False,
        full=True,
    )

    def tensor(image):
        return (
            torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
            .permute(2, 0, 1)[None]
            .to(device)
            / 127.5
            - 1
        )

    lp = lpips_model(tensor(first), tensor(second))[0, 0].mean().detach().cpu()
    return dict(
        psnr_db=-10.0 * np.log10(max(mse, 1e-30)),
        ssim=float(ssim_map.mean()),
        lpips=float(lp),
        note="agreement with baseline back view, not ground-truth quality",
    )


@torch.no_grad()
def render_front_back(pipe, mesh, camera, render_out, resolution):
    """Render only the two views used by the visibility experiment."""
    import math
    import utils3d
    from pixal3d.renderers import PbrMeshRenderer
    from pixal3d.utils.render_utils import proj_camera_to_render_params

    render_out.mkdir(parents=True, exist_ok=True)
    views = render_out / f"views_{resolution}"
    views.mkdir(parents=True, exist_ok=True)
    _, intr = proj_camera_to_render_params(
        camera["camera_angle_x"], camera["distance"]
    )
    renderer = PbrMeshRenderer(
        dict(
            resolution=resolution,
            near=0.01,
            far=100,
            ssaa=1,
            peel_layers=1,
            face_chunk_size=2_000_000,
        ),
        device=pipe.device,
    )
    gpu_mesh = mesh.to(pipe.device)
    started = time.perf_counter()
    for yaw in (0, 180):
        a = math.radians(yaw)
        eye = (
            torch.tensor([math.sin(a), 0.0, math.cos(a)], device=pipe.device)
            * camera["distance"]
        )
        ext = utils3d.torch.extrinsics_look_at(
            eye,
            torch.zeros(3, device=pipe.device),
            torch.tensor([0.0, 1.0, 0.0], device=pipe.device),
        )
        rendered = renderer.render(
            gpu_mesh, ext, intr, envmap=rendering.UnlitEnvironment()
        )
        for kind in ("base_color", "normal"):
            image = (
                (rendered[kind].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            Image.fromarray(image).save(views / f"{kind}_{yaw:03d}.png")
        del rendered
        common.empty_cuda()
    elapsed = time.perf_counter() - started
    save_json(
        render_out / f"render_{resolution}.json",
        dict(
            status="COMPLETE",
            resolution=resolution,
            views=[0, 180],
            seconds=elapsed,
        ),
    )
    del gpu_mesh, renderer
    common.empty_cuda()
    return elapsed


def render_and_measure(pipe, mesh, root, out, baseline_views, canonical, lpips_model):
    out.mkdir(parents=True, exist_ok=True)
    camera = load_json(root / "baseline/camera.json")
    render_out = out / "render"
    render_out.mkdir(parents=True, exist_ok=True)
    render_seconds = render_front_back(
        pipe, mesh, camera, render_out, ARGS.render_resolution
    )
    front = Image.open(render_out / f"views_{ARGS.render_resolution}/base_color_000.png").convert("RGB")
    back = Image.open(render_out / f"views_{ARGS.render_resolution}/base_color_180.png").convert("RGB")
    report = dict(
        front=image_metrics(
            canonical["reference"],
            front,
            canonical["mask"],
            lpips_model,
            pipe.device,
        ),
        back_vs_baseline=pair_metrics(
            baseline_views["back"], back, lpips_model, pipe.device
        ),
        front_vs_baseline=pair_metrics(
            baseline_views["front"], front, lpips_model, pipe.device
        ),
        render_resolution=ARGS.render_resolution,
        render_seconds=render_seconds,
        note="front metrics use the input image; back metrics only measure agreement with baseline because no back ground truth exists",
    )
    save_json(out / "metrics.json", report)
    return report


def prepare_context(pipe, root):
    data = common.load_payload(root / "shared/mapping.pt")
    shape_payload = common.load_payload(root / "shared/shape1024/endpoint.pt")
    if not torch.equal(data["coords"], shape_payload["coords"]):
        raise RuntimeError("shape endpoint and mapping support differ")
    encoded = common.load_payload(root / "shared/bridge/encoded.pt")
    if not torch.equal(data["coords"], encoded["coords"]):
        raise RuntimeError("visibility and shape support differ")
    visible = encoded["visible"].bool()
    coverage_payload = common.load_payload(root / "guides/coverage_0.pt")
    valid = coverage_payload["valid"].bool()
    first_guide = common.load_payload(root / "guides/guide_0.pt")
    if first_guide.get("query_method") == "native_c64_endpoint_nearest3_lift_v1":
        if not torch.equal(first_guide["coords"], data["coords"]) or len(valid) != len(data["coords"]):
            raise RuntimeError("lifted guide coverage/support mismatch")
        coverage = valid.float()
    else:
        ancestry = common.load_payload(root / "shared/bridge/ancestry.pt")
        parent = ancestry["voxel_to_c256"].long()
        if len(parent) != len(valid):
            raise RuntimeError("voxel guide coverage/ancestry mismatch")
        coverage = torch.zeros(len(data["coords"]), dtype=torch.float32)
        counts = torch.zeros(len(data["coords"]), dtype=torch.float32)
        counts.index_add_(0, parent, torch.ones(len(parent)))
        coverage.index_add_(0, parent, valid.float())
        coverage = coverage / counts.clamp_min(1.0)
    guides = [
        common.load_payload(root / f"guides/guide_{k}.pt")["features"]
        for k in range(4)
    ]
    for guide in guides:
        if guide.shape[0] != data["coords"].shape[0]:
            raise RuntimeError("guide support length mismatch")
    bank = {}
    for tile in data["tiles"]:
        if not len(tile["rows"]):
            continue
        path = root / "shared/texture_conditions" / "tiles" / f"{tile['tile_id']:02d}" / "conditions.pt"
        p = common.load_payload(path)
        if not torch.equal(p["coords"], tile["coords"]):
            raise RuntimeError(f"texture condition support mismatch tile={tile['tile_id']}")
        bank[tile["tile_id"]] = {k: p[k] for k in ("global_", "proj")}
    return data, shape_payload, bank, guides, visible, coverage


def ensure_baseline_render(pipe, root, out):
    from pixal3d.representations.mesh import MeshWithVoxel

    render_dir = out / "baseline_render_1024"
    target = render_dir / "views_1024/base_color_180.png"
    if target.exists():
        return {
            "front": Image.open(render_dir / "views_1024/base_color_000.png").convert("RGB"),
            "back": Image.open(target).convert("RGB"),
        }
    render_dir.mkdir(parents=True, exist_ok=True)
    mesh = common.load_payload(root / "baseline/textured_mesh.pt")["mesh"]
    camera = load_json(root / "baseline/camera.json")
    rendering.render(pipe, mesh, render_dir, resolution=1024, camera=camera)
    del mesh
    common.empty_cuda()
    return {
        "front": Image.open(render_dir / "views_1024/base_color_000.png").convert("RGB"),
        "back": Image.open(target).convert("RGB"),
    }


def main():
    if ARGS.hidden_mode != "unconditional" or ARGS.anchor_visible or ARGS.reuse_prefix:
        raise ValueError("strict block routing requires unconditional hidden mode, no cached anchor/prefix")
    if ARGS.visible_context_only or ARGS.trim_conditional_context:
        raise ValueError("strict block routing requires unchanged full conditional context")
    if not 0 <= ARGS.visibility_threshold <= 1:
        raise ValueError("visibility threshold must lie in [0,1]")
    torch.set_num_threads(8)
    root = ARGS.root.resolve()
    out = ARGS.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(ARGS.seeds)
    n_values = parse_ints(ARGS.n_values)
    if any(n < 0 or n > 4 for n in n_values):
        raise ValueError("n-values must lie in [0,4]")

    pipe = common.setup(out)
    data, shape_payload, bank, guides, visible, coverage = prepare_context(pipe, root)
    raw_visible = visible
    if ARGS.anchor_visible:
        visible = dilate_visible_mask(visible, data["coords"], ARGS.anchor_dilation)
        if ARGS.anchor_dilation:
            print(
                f"[anchor] raw_visible={int(raw_visible.sum()):,} "
                f"dilated_visible={int(visible.sum()):,} "
                f"radius={ARGS.anchor_dilation}",
                flush=True,
            )
    shape = shape_payload["features"]
    batch_size = int(ARGS.batch_size)
    max_block_points = max(len(b["rows"]) for b in data["blocks"])
    max_tokens = int(ARGS.max_tokens or batch_size * max_block_points)
    print(
        f"[context] C256={len(data['coords']):,} visible={int(visible.sum()):,} "
        f"blocks={len(data['blocks'])} batch={batch_size} max_tokens={max_tokens:,}",
        flush=True,
    )

    if ARGS.only_flow:
        lpips_model = None
        baseline_views = None
        canonical = None
    else:
        baseline_views = ensure_baseline_render(pipe, root, out)
        baseline_image = Image.open(root / "baseline/image_4096.png").convert("RGB").resize(
            (ARGS.render_resolution, ARGS.render_resolution), Image.Resampling.LANCZOS
        )
        mask = Image.open(root / "baseline/foreground_mask_4096.png").resize(
            (ARGS.render_resolution, ARGS.render_resolution), Image.Resampling.LANCZOS
        )
        canonical = {"reference": baseline_image, "mask": mask}
        import lpips

        # Use the repository's established LPIPS initialization.  The package
        # alex.pth contains only the learned linear calibration layers; the
        # AlexNet trunk is loaded separately by pretrained=True.
        lpips_model = lpips.LPIPS(
            net="alex", version="0.1", pretrained=True, pnet_rand=False, spatial=True
        ).eval().to(pipe.device)

    summary_path = out / "summary.json"
    if summary_path.exists():
        previous = load_json(summary_path)
        summary = list(previous.get("results", []))
    else:
        summary = []
    for seed in seeds:
        for n in n_values:
            name = (
                f"seed_{seed}_anchor"
                f"_d{ARGS.anchor_dilation}_n_{n}"
                if ARGS.anchor_visible
                else f"seed_{seed}_n_{n}"
            )
            if ARGS.all_conditional:
                name += "_all_conditional"
            if ARGS.hidden_mode != "unconditional":
                name += f"_h{ARGS.hidden_mode}"
            if ARGS.visible_context_only:
                name += (
                    f"_vctx{ARGS.visible_context_start}"
                    if ARGS.visible_context_start >= 0
                    else "_vctx"
                )
                if ARGS.visible_context_hidden_weight is not None:
                    name += f"_w{ARGS.visible_context_hidden_weight:g}"
            if ARGS.trim_conditional_context:
                name += (
                    f"_trim{ARGS.trim_conditional_context_start}"
                    if ARGS.trim_conditional_context_start >= 0
                    else "_trim"
                )
            variant = out / name
            print(f"[variant] {name}", flush=True)
            if ARGS.anchor_visible:
                result = run_anchor_visible_flow(
                    pipe,
                    data,
                    shape,
                    bank,
                    guides,
                    visible,
                    coverage,
                    variant / "texture",
                    out / f"seed_{seed}_n_0/texture",
                    seed,
                    n,
                    batch_size,
                    max_tokens,
                    ARGS.hidden_mode,
                    ARGS.visible_context_only,
                    ARGS.visible_context_start,
                    float(
                        ARGS.visible_context_hidden_weight
                        if ARGS.visible_context_hidden_weight is not None
                        else 0.0
                    ),
                    ARGS.trim_conditional_context,
                    ARGS.trim_conditional_context_start,
                    ARGS.reuse_prefix,
                    ARGS.reuse_prefix_step,
                )
            else:
                result = run_flow(
                    pipe,
                    data,
                    shape,
                    bank,
                    guides,
                    visible,
                    coverage,
                    variant / "texture",
                    seed,
                    n,
                    batch_size,
                    max_tokens,
                    ARGS.hidden_mode,
                    ARGS.visible_context_only,
                    ARGS.visible_context_start,
                    float(
                        ARGS.visible_context_hidden_weight
                        if ARGS.visible_context_hidden_weight is not None
                        else 0.0
                    ),
                    ARGS.trim_conditional_context,
                    ARGS.trim_conditional_context_start,
                    ARGS.reuse_prefix,
                    ARGS.reuse_prefix_step,
                )
            record = dict(
                variant=name,
                seed=seed,
                n=n,
                anchor_visible=bool(ARGS.anchor_visible),
                anchor_dilation=int(ARGS.anchor_dilation),
                hidden_mode=ARGS.hidden_mode,
                visible_context_only=bool(ARGS.visible_context_only),
                visible_context_start=int(ARGS.visible_context_start),
                visible_context_hidden_weight=(
                    float(ARGS.visible_context_hidden_weight)
                    if ARGS.visible_context_hidden_weight is not None
                    else None
                ),
                trim_conditional_context=bool(ARGS.trim_conditional_context),
                trim_conditional_context_start=int(ARGS.trim_conditional_context_start),
                reuse_prefix=(
                    str(ARGS.reuse_prefix.resolve())
                    if ARGS.reuse_prefix is not None
                    else None
                ),
                reuse_prefix_step=int(ARGS.reuse_prefix_step),
                flow_seconds=result["elapsed"],
                resumed=result["resumed"],
                visible_points=int(visible.sum()),
                hidden_points=int((~visible).sum()),
            )
            if not ARGS.only_flow:
                mesh = decode_material_memory(pipe, root, shape_payload, result["features"])
                metrics = render_and_measure(
                    pipe,
                    mesh,
                    root,
                    variant,
                    baseline_views,
                    canonical,
                    lpips_model,
                )
                record["metrics"] = metrics
                del mesh
                common.empty_cuda()
            save_json(variant / "result.json", record)
            summary = [
                old
                for old in summary
                if old.get("variant") != name
            ] + [record]
            save_json(out / "summary.json", dict(version=VERSION, results=summary))

    save_json(
        out / "experiment.json",
        dict(
            status="COMPLETE",
            version=VERSION,
            root=str(root),
            seeds=seeds,
            n_values=n_values,
            C256=len(data["coords"]),
            raw_visible_points=int(raw_visible.sum()),
            visible_points=int(visible.sum()),
            hidden_points=int((~visible).sum()),
            anchor_visible=bool(ARGS.anchor_visible),
            anchor_dilation=int(ARGS.anchor_dilation),
            visible_context_only=bool(ARGS.visible_context_only),
            visible_context_start=int(ARGS.visible_context_start),
            visible_context_hidden_weight=(
                float(ARGS.visible_context_hidden_weight)
                if ARGS.visible_context_hidden_weight is not None
                else None
            ),
            trim_conditional_context=bool(ARGS.trim_conditional_context),
            trim_conditional_context_start=int(ARGS.trim_conditional_context_start),
            reuse_prefix=(
                str(ARGS.reuse_prefix.resolve())
                if ARGS.reuse_prefix is not None
                else None
            ),
            reuse_prefix_step=int(ARGS.reuse_prefix_step),
            batch_size=batch_size,
            max_tokens=max_tokens,
            results=len(summary),
        ),
    )
    print("[done]", out, flush=True)


if __name__ == "__main__":
    main()
