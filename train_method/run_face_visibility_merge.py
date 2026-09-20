#!/usr/bin/env python3
"""Merge two decoded materials at final-mesh face level.

This is a diagnostic for visibility-aware texture decoding.  Faces visible in
the input camera keep the full-conditional material; all other faces use the
hidden/unconditional candidate.  It avoids the cross-surface mixing caused by
merging sparse 4096 attributes before the renderer's trilinear query.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    p.add_argument("--base-variant", default="seed_48_n_0")
    p.add_argument("--candidate-variant", default="seed_48_n_4_vctx4")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--face-chunk", type=int, default=2_000_000)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.argv = ["sr_point_visibility_texture.py", "--gpu", str(args.gpu)]
    import sr_point_visibility_texture as exp

    from pixal3d.representations.mesh import MeshWithFacePbr
    from flex_gemm.ops.grid_sample import grid_sample_3d

    exp.ARGS.render_resolution = args.resolution
    root = args.root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / (
        f"face_merge_{args.base_variant}_with_{args.candidate_variant}"
    )
    output.mkdir(parents=True, exist_ok=True)
    pipe = exp.common.setup(experiment_dir)
    shape_payload = exp.common.load_payload(root / "shared/shape1024/endpoint.pt")
    base = exp.common.load_payload(
        experiment_dir / args.base_variant / "texture/endpoint.pt"
    )
    candidate = exp.common.load_payload(
        experiment_dir / args.candidate_variant / "texture/endpoint.pt"
    )
    if not exp.torch.equal(base["coords"], candidate["coords"]):
        raise RuntimeError("base and candidate C256 supports differ")

    topology = exp.common.load_payload(root / "shared/final_mesh/topology.pt")
    geometry = exp.common.load_payload(
        root / "shared/final_mesh/geometry_mesh.pt"
    )["mesh"]

    @exp.torch.no_grad()
    def decode_field(features):
        tex = exp.common.SparseTensor(
            exp.rendering.norm(
                pipe,
                "tex",
                features.to(pipe.device),
                inverse=True,
            ),
            base["coords"].to(pipe.device),
        )
        started = time.perf_counter()
        field = pipe.decode_tex_slat(
            tex,
            [sub.to(pipe.device) for sub in topology["subs"]],
        )
        elapsed = time.perf_counter() - started
        coords = field.coords.detach().cpu().int()
        attrs = field.feats.detach().cpu().float()
        voxel_shape = exp.torch.Size([*field.shape, *field.spatial_shape])
        del field, tex
        exp.common.empty_cuda()
        return coords, attrs, voxel_shape, elapsed

    base_coords, base_attrs, voxel_shape, base_decode_seconds = decode_field(
        base["features"]
    )
    cand_coords, cand_attrs, cand_shape, cand_decode_seconds = decode_field(
        candidate["features"]
    )
    if not exp.torch.equal(base_coords, cand_coords):
        raise RuntimeError("decoded material supports differ")
    if voxel_shape != cand_shape or base_attrs.shape != cand_attrs.shape:
        raise RuntimeError("decoded material layouts differ")

    visible_payload = exp.common.load_payload(
        root / "shared/final_mesh/final_visibility/visible_vertices.pt"
    )
    face_ids = visible_payload["face_ids"].long()
    visible_faces = exp.torch.zeros(len(geometry.faces), dtype=exp.torch.bool)
    visible_faces[face_ids] = True

    coords_gpu = base_coords[:, 1:].to(pipe.device)
    base_gpu = base_attrs.to(pipe.device)
    cand_gpu = cand_attrs.to(pipe.device)
    sparse_coords = exp.torch.cat(
        [exp.torch.zeros_like(coords_gpu[:, :1]), coords_gpu], dim=-1
    )
    voxel_shape = exp.torch.Size(voxel_shape)
    face_attrs = exp.torch.empty(
        (len(geometry.faces), base_attrs.shape[1]), dtype=exp.torch.float16
    )
    started = time.perf_counter()
    for start in range(0, len(geometry.faces), args.face_chunk):
        end = min(start + args.face_chunk, len(geometry.faces))
        faces = geometry.faces[start:end].long()
        centers = geometry.vertices[faces].mean(dim=1).to(pipe.device)
        grid = ((centers + 0.5) * 4096).reshape(1, -1, 3)
        base_sample = grid_sample_3d(
            base_gpu, sparse_coords, voxel_shape, grid, mode="trilinear"
        ).reshape(end - start, -1)
        cand_sample = grid_sample_3d(
            cand_gpu, sparse_coords, voxel_shape, grid, mode="trilinear"
        ).reshape(end - start, -1)
        use_base = visible_faces[start:end].to(pipe.device).unsqueeze(1)
        selected = exp.torch.where(use_base, base_sample, cand_sample)
        face_attrs[start:end] = selected.float().cpu().to(exp.torch.float16)
        if start == 0 or end == len(geometry.faces):
            print(f"FACE SAMPLE {end:,}/{len(geometry.faces):,}", flush=True)
        del faces, centers, grid, base_sample, cand_sample, use_base, selected
    sample_seconds = time.perf_counter() - started

    mesh = MeshWithFacePbr(
        geometry.vertices,
        geometry.faces,
        face_attrs,
        layout=pipe.pbr_attr_layout,
    )
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
    report = exp.render_and_measure(
        pipe,
        mesh,
        root,
        output,
        baseline_views,
        {"reference": reference, "mask": mask},
        lpips_model,
    )
    exp.save_json(
        output / "result.json",
        dict(
            variant=output.name,
            base_variant=args.base_variant,
            candidate_variant=args.candidate_variant,
            visible_faces=int(visible_faces.sum()),
            total_faces=len(visible_faces),
            visible_face_fraction=float(visible_faces.float().mean()),
            base_decode_seconds=base_decode_seconds,
            candidate_decode_seconds=cand_decode_seconds,
            face_sampling_seconds=sample_seconds,
            metrics=report,
        ),
    )
    print(exp.json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
