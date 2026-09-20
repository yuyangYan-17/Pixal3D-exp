#!/usr/bin/env python3
"""Decode a hidden-point interpolation between two saved texture-flow endpoints.

This is a diagnostic only: visible points stay on the all-conditional endpoint,
while hidden points use x0 + alpha * (x_n4 - x_n0).  It tests whether the
front/back trade-off is caused by an overly hard binary route.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="4")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/sr_0_img_trilinear_face_n3_20260919/fixed"),
    )
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/sr_point_visibility_20260919"),
    )
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--alphas", default="0.25,0.5,0.75")
    p.add_argument("--base-variant", default=None)
    p.add_argument("--candidate-variant", default=None)
    p.add_argument("--blend-mode", choices=("hidden", "all"), default="hidden")
    return p.parse_args()


def main():
    args = parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    # The experiment module owns the repository's decoder, renderer, and
    # metric setup.  Hide this utility's arguments from its parser on import.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.argv = ["sr_point_visibility_texture.py", "--gpu", str(args.gpu)]
    import sr_point_visibility_texture as exp

    exp.ARGS.render_resolution = args.resolution
    root = args.root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    pipe = exp.common.setup(experiment_dir)
    data = exp.common.load_payload(root / "shared/mapping.pt")
    encoded = exp.common.load_payload(root / "shared/bridge/encoded.pt")
    visible = encoded["visible"].bool()
    shape_payload = exp.common.load_payload(root / "shared/shape1024/endpoint.pt")
    base_variant = args.base_variant or f"seed_{args.seed}_n_0"
    candidate_variant = args.candidate_variant or f"seed_{args.seed}_n_4"
    n0 = exp.common.load_payload(
        experiment_dir / base_variant / "texture/endpoint.pt"
    )
    n4 = exp.common.load_payload(
        experiment_dir / candidate_variant / "texture/endpoint.pt"
    )
    if not exp.torch.equal(n0["coords"], data["coords"]):
        raise RuntimeError("n=0 endpoint support mismatch")
    if not exp.torch.equal(n4["coords"], data["coords"]):
        raise RuntimeError("n=4 endpoint support mismatch")
    if not exp.torch.equal(shape_payload["coords"], data["coords"]):
        raise RuntimeError("shape endpoint support mismatch")

    baseline_views = exp.ensure_baseline_render(pipe, root, experiment_dir)
    from PIL import Image

    reference = Image.open(root / "baseline/image_4096.png").convert("RGB").resize(
        (args.resolution, args.resolution), Image.Resampling.LANCZOS
    )
    mask = Image.open(root / "baseline/foreground_mask_4096.png").resize(
        (args.resolution, args.resolution), Image.Resampling.LANCZOS
    )
    import lpips

    lpips_model = lpips.LPIPS(
        net="alex", version="0.1", pretrained=True, pnet_rand=False, spatial=True
    ).eval().to(pipe.device)
    canonical = {"reference": reference, "mask": mask}
    summary = []
    for alpha in alphas:
        features = n0["features"] + alpha * (n4["features"] - n0["features"])
        if args.blend_mode == "hidden":
            hidden = ~visible
            features[visible] = n0["features"][visible]
        label = (
            f"seed_{args.seed}_{candidate_variant.replace('/', '_')}_"
            f"{args.blend_mode}_blend_{alpha:g}"
        )
        out = experiment_dir / label
        mesh = exp.decode_material_memory(pipe, root, shape_payload, features)
        metrics = exp.render_and_measure(
            pipe, mesh, root, out, baseline_views, canonical, lpips_model
        )
        record = {
            "variant": label,
            "seed": args.seed,
            "alpha": alpha,
            "base_variant": base_variant,
            "candidate_variant": candidate_variant,
            "blend_mode": args.blend_mode,
            "metrics": metrics,
        }
        exp.save_json(out / "result.json", record)
        summary.append(record)
        del mesh, features
        exp.common.empty_cuda()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    exp.save_json(experiment_dir / "blend_summary.json", summary)


if __name__ == "__main__":
    main()
