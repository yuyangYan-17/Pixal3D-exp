#!/usr/bin/env python3
"""Real-checkpoint equivalence tests for visibility-routed conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

import pixal3d_visibility_routed_conditioning as experiment
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


def _slice_condition(
    condition: Mapping[str, Any],
    rows: torch.Tensor,
    coords: torch.Tensor,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for branch_name in ("cond", "neg_cond"):
        branch = condition[branch_name]
        projection = branch["proj"]
        output[branch_name] = {
            "global": branch["global"],
            "proj": SparseTensor(
                projection.feats.index_select(0, rows),
                coords,
            ),
        }
    return output


def _routed(
    *,
    front: Mapping[str, Any],
    global_back: torch.Tensor,
    visibility: torch.Tensor,
    coords: torch.Tensor,
) -> Dict[str, Any]:
    projection = front["proj"]
    return {
        "mode": "visibility_routed",
        "global_front": front["global"],
        "proj_front": projection,
        "global_back": global_back,
        "proj_back": projection.replace(
            torch.zeros_like(projection.feats)
        ),
        "token_visibility": visibility,
        "mask_coords": coords,
        "routing_kind": "hard",
        "record_diagnostics": False,
        "record_self_attention_intervention": False,
    }


def _maximum_error(left: SparseTensor, right: SparseTensor) -> float:
    if not torch.equal(left.coords, right.coords):
        raise RuntimeError("equivalence output coordinates differ")
    return float((left.feats.float() - right.feats.float()).abs().max().item())


@torch.no_grad()
def _check_model(
    *,
    model: Any,
    condition: Mapping[str, Any],
    rows: int,
    texture: bool,
    seed: int,
) -> Dict[str, Any]:
    full_projection = condition["hr"]["cond"]["proj"]
    count = min(int(rows), int(full_projection.feats.shape[0]))
    selected = torch.linspace(
        0,
        full_projection.feats.shape[0] - 1,
        count,
        dtype=torch.float64,
    ).round().to(torch.long)
    coords = full_projection.coords.index_select(0, selected).to("cuda")
    hr = experiment.posterior._move_nested(
        _slice_condition(condition["hr"], selected, coords.cpu()),
        torch.device("cuda"),
    )
    back_source = experiment.posterior._move_nested(
        _slice_condition(condition["back"], selected, coords.cpu()),
        torch.device("cuda"),
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    state = SparseTensor(
        torch.randn(
            (count, 32),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ),
        coords,
    )
    concat: Optional[SparseTensor] = None
    if texture:
        concat = SparseTensor(
            torch.randn(
                (count, 32),
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            ),
            coords,
        )
    kwargs = {"concat_cond": concat} if concat is not None else {}
    timestep = torch.tensor([731.0], device="cuda")
    local = model(state, timestep, hr["cond"], **kwargs)
    local_repeat = model(state, timestep, hr["cond"], **kwargs)
    all_front = model(
        state,
        timestep,
        _routed(
            front=hr["cond"],
            global_back=back_source["cond"]["global"],
            visibility=torch.ones(count, device="cuda", dtype=torch.bool),
            coords=coords,
        ),
        **kwargs,
    )
    zero_projection = hr["cond"]["proj"].replace(
        torch.zeros_like(hr["cond"]["proj"].feats)
    )
    global_reference = model(
        state,
        timestep,
        {
            "global": back_source["cond"]["global"],
            "proj": zero_projection,
        },
        **kwargs,
    )
    all_back = model(
        state,
        timestep,
        _routed(
            front=hr["cond"],
            global_back=back_source["cond"]["global"],
            visibility=torch.zeros(count, device="cuda", dtype=torch.bool),
            coords=coords,
        ),
        **kwargs,
    )
    zero_reference = model(
        state,
        timestep,
        hr["neg_cond"],
        **kwargs,
    )
    all_zero = model(
        state,
        timestep,
        _routed(
            front=hr["cond"],
            global_back=hr["neg_cond"]["global"],
            visibility=torch.zeros(count, device="cuda", dtype=torch.bool),
            coords=coords,
        ),
        **kwargs,
    )
    mixed = model(
        state,
        timestep,
        _routed(
            front=hr["cond"],
            global_back=back_source["cond"]["global"],
            visibility=(torch.arange(count, device="cuda") % 2 == 0),
            coords=coords,
        ),
        **kwargs,
    )
    errors = {
        "default_path_repeat_max_abs": _maximum_error(
            local, local_repeat
        ),
        "all_front_vs_local_max_abs": _maximum_error(local, all_front),
        "all_back_global_vs_reference_max_abs": _maximum_error(
            global_reference, all_back
        ),
        "all_back_zero_vs_reference_max_abs": _maximum_error(
            zero_reference, all_zero
        ),
    }
    tolerance = 1e-5
    if any(value >= tolerance for value in errors.values()):
        raise AssertionError(f"real model equivalence failed: {errors}")
    return {
        "tokens": count,
        "texture_concat_condition": texture,
        "input_shapes": {
            "state": list(state.feats.shape),
            "global": list(hr["cond"]["global"].shape),
            "projected": list(hr["cond"]["proj"].feats.shape),
            "coords": list(coords.shape),
        },
        "errors": errors,
        "tolerance": tolerance,
        "mixed_output_finite": bool(torch.isfinite(mixed.feats).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--source-cache-dir",
        default="outputs/joint_online_canonical_posterior/source_cache",
    )
    parser.add_argument("--tile-id", type=int, default=24)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument(
        "--output",
        default=(
            "outputs/visibility_routed_conditioning/"
            "REAL_MODEL_EQUIVALENCE.json"
        ),
    )
    args = parser.parse_args()
    if int(args.cuda_device) != 4:
        raise ValueError("Codex.md requires CUDA 4")
    torch.cuda.set_device(4)
    pipeline = init_pipeline(
        args.model_path, device="cuda", low_vram=True
    )
    cache = Path(args.source_cache_dir)
    tile = f"tile_{int(args.tile_id):02d}"
    conditions: Dict[str, Dict[str, Any]] = {}
    for latent in ("shape", "texture"):
        saved = torch.load(
            cache / "per_tile" / tile / f"{latent}_conditions.pt",
            map_location="cpu",
            weights_only=False,
        )
        conditions[latent] = {
            "hr": saved["hr"],
            # The cached matched LR branch supplies a distinct real global
            # context. Projected features are explicitly zeroed in the test.
            "back": saved["lr"],
        }
    results: Dict[str, Any] = {}
    for latent, texture in (("shape", False), ("texture", True)):
        key = (
            "tex_slat_flow_model_1024"
            if texture
            else "shape_slat_flow_model_1024"
        )
        model = pipeline.models[key].to("cuda").eval()
        results[latent] = _check_model(
            model=model,
            condition=conditions[latent],
            rows=int(args.tokens),
            texture=texture,
            seed=20260731 + int(texture),
        )
        model.cpu()
        torch.cuda.empty_cache()
    payload = {
        "status": "passed",
        "cuda_device": 4,
        "tile_id": int(args.tile_id),
        "models": results,
    }
    output = Path(args.output)
    experiment._atomic_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
