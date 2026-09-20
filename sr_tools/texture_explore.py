"""Texture-only experiments on a previously decoded, fixed geometry.

High-visibility blocks use conditional flow. Low-visibility mixed blocks
run separate full-context conditional/unconditional forwards and select
velocities by global point visibility. Hidden points use baseline endpoints
for the guide prefix. Each individual forward has homogeneous conditioning.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from . import common, rendering, texture, shape as sync
from .capacity import statistics
from .visibility import lookup_rows


def _copy_branch(block, mask, branch, keep_context=False):
    """Create a branch-homogeneous view of one original depth block.

    ``keep_context`` retains every row in the sparse block while limiting
    ownership updates to ``mask``.  The conditional and unconditional calls
    remain separate, but each network sees the same spatial context as the
    original block.  This avoids changing the sparse-convolution receptive
    field merely because visibility differs.
    """
    if not bool(mask.any()):
        return None
    out = dict(block)
    if keep_context:
        out["owned"] = block["owned"] & mask
    else:
        out["global_ids"] = block["global_ids"][mask]
        out["coords"] = block["coords"][mask]
        out["rows"] = block["rows"][mask]
        out["owned"] = block["owned"][mask]
        out["depth_weights"] = block["depth_weights"][mask]
    out["branch"] = branch
    out["source_block_id"] = int(block["block_id"])
    out["block_id"] = f"{block['block_id']}:{branch}"
    return out


def _coverage_per_latent(root: Path, n_points: int) -> torch.Tensor:
    """Aggregate the baseline field support probe to C256 points."""
    ancestry = common.load_payload(root / "shared/bridge/ancestry.pt")
    parent = ancestry["voxel_to_c256"].long()
    valid = common.load_payload(root / "guides/coverage_0.pt")["valid"]
    if len(parent) != len(valid):
        raise RuntimeError("guide coverage and voxel ancestry have different lengths")
    count = torch.zeros(n_points, dtype=torch.float32)
    hits = torch.zeros(n_points, dtype=torch.float32)
    count.index_add_(0, parent, torch.ones(len(parent)))
    hits.index_add_(0, parent, valid.float())
    return hits / count.clamp_min(1.0)


def _load_guides(root: Path, n_points: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    guides = []
    for k in range(4):
        p = common.load_payload(root / f"guides/guide_{k}.pt")
        if len(p["coords"]) != n_points:
            raise RuntimeError("guide C256 support does not match fixed texture support")
        guides.append(p["features"])
    coverage = _coverage_per_latent(root, n_points)
    return guides, coverage


def _load_bank(root: Path, data):
    bank = {}
    base = root / "shared/texture_conditions/tiles"
    for tile in data["tiles"]:
        if not len(tile["rows"]):
            continue
        p = base / f"{tile['tile_id']:02d}/conditions.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        c = common.load_payload(p)
        if not torch.equal(c["coords"], tile["coords"]):
            raise RuntimeError(f"condition support mismatch for tile {tile['tile_id']}")
        bank[tile["tile_id"]] = {k: c[k] for k in ("global_", "proj")}
    return bank


def _make_blocks(
    data, visible, coverage, scope_tile, guide_step, guide_threshold,
    keep_context=True, hidden_mode="unconditional",
):
    from train_method.texture_block_routing import route_blocks
    if not keep_context or hidden_mode != "unconditional":
        raise ValueError("block routing requires full context and image-unconditional hidden flow")
    selected = dict(data, blocks=[
        b for b in data["blocks"] if scope_tile is None or b["tile_id"] == scope_tile
    ])
    result, _ = route_blocks(selected, visible, coverage, guide_step,
                             guide_threshold=guide_threshold)
    if scope_tile is not None:
        for b in data["blocks"]:
            if b["tile_id"] != scope_tile and len(b["global_ids"]):
                result["conditional"].append(dict(b, branch="conditional"))
    return result


def _groups(branch_blocks, batch_size, max_tokens):
    return {
        k: list(sync.groups(v, batch_size, max_tokens))
        for k, v in branch_blocks.items()
    }


@torch.no_grad()
def _guided_values(pipe, sampler, group, x, guides, ids, t, guide_step):
    guide = guides[guide_step]
    return [
        sampler._xstart_to_pred(
            x[b["global_ids"]], t, guide[b["global_ids"]]
        ).float()
        for b in group
    ]


def _predict_branch(pipe, model, groups, x, shape, bank, t, params, branch):
    out = []
    for group in groups:
        if branch == "conditional":
            vals = texture.predict_safe(pipe, model, group, x, shape, bank, t, params)
        elif branch == "unconditional":
            vals = texture.predict_safe(
                pipe, model, group, x, shape, bank, t, params, unconditional=True
            )
        elif branch == "global":
            vals = texture.predict_safe(
                pipe, model, group, x, shape, bank, t, params, global_only=True
            )
        elif branch == "conditional_hidden":
            vals = texture.predict_safe(pipe, model, group, x, shape, bank, t, params)
        else:
            raise ValueError(branch)
        out.append((group, vals))
    return out


@torch.no_grad()
def flow(
    pipe,
    root: Path,
    out: Path,
    *,
    seed: int = 46,
    scope_tile: int | None = None,
    guide_steps: int = 1,
    guide_threshold: float = 1.0,
    keep_context: bool = True,
    hidden_mode: str = "unconditional",
    batch_size: int | None = None,
    max_tokens: int | None = None,
    steps: int = 12,
):
    """Run texture flow on fixed geometry and return the normalized C256 latent."""
    shared = root / "shared"
    data = common.load_payload(shared / "mapping.pt")
    shape_payload = common.load_payload(shared / "shape1024/endpoint.pt")
    shape = shape_payload["features"]
    coords = shape_payload["coords"]
    if not torch.equal(coords, data["coords"]):
        raise RuntimeError("fixed shape endpoint and mapping supports differ")
    encoded = common.load_payload(shared / "bridge/encoded.pt")
    visible = encoded["visible"]
    guides, coverage = _load_guides(root, len(coords))
    bank = _load_bank(root, data)
    model = pipe.models["tex_slat_flow_model_1024"]
    sampler = pipe.tex_slat_sampler
    params = dict(pipe.tex_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(steps, params.pop("rescale_t", 1.0))
    if guide_steps < 0 or guide_steps > 4:
        raise ValueError("guide_steps must be in [0,4]")
    if hidden_mode not in ("unconditional", "global", "conditional_hidden"):
        raise ValueError(
            "hidden_mode must be 'unconditional', 'global', or 'conditional_hidden'"
        )
    if batch_size is None:
        batch_size = 58
    if max_tokens is None:
        max_tokens = batch_size * statistics(data)["max_block_points"]
    x = torch.randn(
        (len(coords), model.in_channels - shape.shape[1]),
        generator=torch.Generator().manual_seed(int(seed)),
    )
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        version="block_threshold_point_selection_v2",
        seed=int(seed),
        scope_tile=scope_tile,
        guide_steps=int(guide_steps),
        guide_threshold=float(guide_threshold),
        keep_context=bool(keep_context),
        hidden_mode=hidden_mode,
        times=times,
        fixed_geometry=str(root.resolve()),
        routing=(
            "visible point conditional; trusted hidden point endpoint-guided; "
            f"remaining hidden point {hidden_mode}; one condition type per forward"
        ),
        batch_size=int(batch_size),
        max_tokens=int(max_tokens),
    )
    manifest = out / "sampler.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("routing configuration changed; use a fresh output directory")
    common.js(manifest, identity)
    common.save(out / "initial.pt", coords=coords, noise=x, state=x, seed=int(seed))
    shape_hash = sync.tensor_hash(shape)
    started = time.perf_counter()
    model.cuda()
    try:
        for step, (t, tn) in enumerate(zip(times, times[1:])):
            if step >= steps:
                break
            before = sync.tensor_hash(x)
            active_guide_step = step if step < guide_steps else None
            branches = _make_blocks(
                data,
                visible,
                coverage,
                scope_tile,
                active_guide_step,
                guide_threshold,
                keep_context=keep_context,
                hidden_mode=hidden_mode,
            )
            grouped = _groups(branches, batch_size, max_tokens)
            total = torch.zeros_like(x)
            counts = torch.zeros(len(x), dtype=torch.int32)
            hidden_branch = hidden_mode
            calls = {"conditional": 0, "guided": 0, hidden_branch: 0}
            for branch in ("conditional", hidden_branch):
                outputs = _predict_branch(
                    pipe,
                    model,
                    grouped[branch],
                    x,
                    shape,
                    bank,
                    t,
                    params,
                    branch,
                )
                calls[branch] = len(outputs)
                for group, values in outputs:
                    sync.reduce_predictions(total, counts, group, values)
            if active_guide_step is not None:
                for group in grouped["guided"]:
                    values = _guided_values(
                        pipe, sampler, group, x, guides, coords, t, active_guide_step
                    )
                    calls["guided"] += 1
                    for b, v in zip(group, values):
                        sync.reduce_predictions(total, counts, [b], [v])
            expected = data["counts"]
            if not torch.equal(counts, expected):
                raise RuntimeError(
                    f"routing left uncovered/duplicated owner points at step {step}: "
                    f"min={int(counts.min())}, max={int(counts.max())}, "
                    f"expected={int(expected.min())}/{int(expected.max())}"
                )
            if before != sync.tensor_hash(x):
                raise RuntimeError("flow predictor mutated the shared state")
            velocity = total / counts[:, None]
            x = x - (t - tn) * velocity
            if not torch.isfinite(x).all():
                raise RuntimeError("texture flow produced nonfinite features")
            common.save(
                out / f"step_{step:02d}.pt",
                coords=coords,
                features=x,
                velocity=velocity,
                t=t,
                t_next=tn,
                input_hash=before,
                shape_hash=shape_hash,
            )
            common.js(
                out / f"step_{step:02d}.json",
                dict(
                    step=step,
                    t=t,
                    t_next=tn,
                    guide_step=active_guide_step,
                    calls=calls,
                    branch_blocks={k: sum(len(g) for g in grouped[k]) for k in grouped},
                    trusted_hidden_points=int(
                        sum(
                            int(b["global_ids"].numel())
                            for b in branches["guided"]
                        )
                    ),
                    hidden_branch_points=int(
                        sum(
                            int(b["global_ids"].numel())
                            for b in branches[hidden_branch]
                        )
                    ),
                    output_hash=sync.tensor_hash(x),
                ),
            )
            print(
                "TEXTURE EXPLORE",
                step + 1,
                "/",
                steps,
                "branch calls",
                calls,
                flush=True,
            )
    finally:
        model.cpu()
        common.empty_cuda()
    elapsed = time.perf_counter() - started
    common.save(out / "endpoint.pt", coords=coords, features=x, normalized=True)
    common.js(out / "timing.json", dict(texture_flow_seconds=elapsed, steps=steps))
    return dict(coords=coords, features=x, normalized=True)


@torch.no_grad()
def decode_evaluate_render(pipe, root: Path, out: Path, payload, *, render=True):
    """Decode the fixed geometry, calculate 1024 foreground metrics and render views."""
    from .metrics import evaluate

    out.mkdir(parents=True, exist_ok=True)
    baseline = root / "baseline"
    canonical = {
        k: common.Image.open(baseline / f"{k}.png").copy()
        for k in ("image_4096", "foreground_mask_4096")
    }
    camera = json.loads((baseline / "camera.json").read_text())
    base = common.load_payload(baseline / "textured_mesh.pt")["mesh"]
    started = time.perf_counter()
    mesh = rendering.decode_material(pipe, root / "shared", out / "texture", payload, payload["features"])
    decode_seconds = time.perf_counter() - started
    report = evaluate(
        {"baseline1024": base, "sr": mesh},
        canonical,
        camera,
        out / "evaluation_1024",
    )
    render_seconds = 0.0
    if render:
        started = time.perf_counter()
        rendering.render(pipe, mesh, out / "texture", resolution=2048, camera=camera)
        render_seconds = time.perf_counter() - started
    common.js(
        out / "result.json",
        dict(
            status="COMPLETE",
            metrics=report["results"],
            decode_seconds=decode_seconds,
            render_seconds=render_seconds,
        ),
    )
    return mesh, report
