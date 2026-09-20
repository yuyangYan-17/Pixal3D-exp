#!/usr/bin/env python3
"""Explore texture routing while reusing a completed fixed-geometry run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


GPU4 = "GPU-5a01b63c-14ed-235f-7936-8043e91e88a5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("outputs/sr_0_img_visibility_guidance"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "subset", "full"), default="subset")
    p.add_argument("--tile", type=int, default=10)
    p.add_argument("--guide-steps", type=int, default=1)
    p.add_argument("--guide-threshold", type=float, default=1.0)
    p.add_argument(
        "--hidden-mode",
        choices=("unconditional", "global", "conditional_hidden"),
        default="unconditional",
        help="hidden-point branch after endpoint guidance",
    )
    p.add_argument(
        "--split-context",
        action="store_true",
        help="remove the other visibility class from each branch context (diagnostic)",
    )
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument(
        "--guide-query",
        choices=("reuse", "inference_sparse_trilinear_v1"),
        default="reuse",
        help="reuse prepared guides or rebuild them with inference.py trilinear + mesh-face fallback",
    )
    p.add_argument("--gpu", default=GPU4)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


def _ensure_link(target: Path, link: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        raise RuntimeError(f"refusing to replace existing link {link}")
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(Path(os.path.relpath(target, link.parent)), target_is_directory=target.is_dir())


def _prepare_query_root(pipe, common, base_source: Path, output: Path, query_method: str):
    """Build a private fixed-root view with freshly encoded guide tensors."""
    from sr_tools import texture_guidance

    if not (base_source / "baseline_fields").exists():
        raise RuntimeError(
            "rebuilding guides requires baseline_fields in the source run: "
            f"{base_source / 'baseline_fields'}"
        )
    encoded = common.load_payload(base_source / "shared/bridge/encoded.pt")
    texture_guidance.encode_guides(
        pipe,
        base_source / "shared/bridge/voxels.pt",
        encoded["coords"],
        base_source / "baseline_fields",
        output / "guides",
        query_method=query_method,
        fallback_mesh=base_source / "baseline/textured_mesh.pt",
    )
    root = output / "fixed"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("baseline", "shared", "prepared.json"):
        _ensure_link(base_source / name, root / name)
    _ensure_link(output / "guides", root / "guides")
    return root


def main():
    a = parse_args()
    a.source = a.source.resolve()
    a.output_dir = a.output_dir.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["PYTHONUNBUFFERED"] = "1"
    import torch

    torch.set_num_threads(8)
    from sr_tools import common, texture_explore

    if not (a.source / "prepared.json").exists():
        raise RuntimeError(f"fixed source is not prepared: {a.source}")
    scope = None if a.mode == "full" else a.tile
    guide_steps = 0 if a.mode == "smoke" else a.guide_steps
    steps = 1 if a.mode == "smoke" else a.steps
    started = time.perf_counter()
    pipe = common.setup(a.output_dir)
    query_root = a.source
    if a.guide_query != "reuse":
        query_root = _prepare_query_root(
            pipe, common, a.source, a.output_dir, a.guide_query
        )
    common.js(
        a.output_dir / "experiment.json",
        dict(
            source=str(query_root),
            base_source=str(a.source),
            mode=a.mode,
            scope_tile=scope,
            guide_steps=guide_steps,
            guide_threshold=a.guide_threshold,
            hidden_mode=a.hidden_mode,
            keep_context=not a.split_context,
            seed=a.seed,
            steps=steps,
            guide_query=a.guide_query,
            fixed_geometry=True,
            routing="one homogeneous branch per sparse forward",
        ),
    )
    payload = texture_explore.flow(
        pipe,
        query_root,
        a.output_dir / "texture",
        seed=a.seed,
        scope_tile=scope,
        guide_steps=guide_steps,
        guide_threshold=a.guide_threshold,
        hidden_mode=a.hidden_mode,
        keep_context=not a.split_context,
        steps=steps,
    )
    if a.mode != "smoke":
        texture_explore.decode_evaluate_render(
            pipe, query_root, a.output_dir, payload, render=not a.no_render
        )
    common.js(
        a.output_dir / "status.json",
        dict(
            status="SMOKE_PASS" if a.mode == "smoke" else "COMPLETE",
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    print("COMPLETE", a.output_dir, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
