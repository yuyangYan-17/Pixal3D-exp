#!/usr/bin/env python3
"""A/B test ordinary versus support-guided C4096 shape encoding.

The input is the exact C4096 flexible-dual-grid cache used by the guided
endpoint experiment.  This script deliberately calls the stock encoder
without ``guide_subs`` and compares its C256 result with the cached guided
encoding.  Equal coordinates and features prove that a subsequent ordinary,
deterministic decoder must produce the same mesh as the already-rendered
ordinary-decoder control.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch

import pixal3d.models as pixal3d_models
from pixal3d.modules.sparse import SparseTensor


DEFAULT_RUN = Path("outputs/guided_endpoint_sr_step0_cuda5")
DEFAULT_ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
)


def _atomic_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    voxel_path = run_dir / "voxel4096" / "paired_endpoint_voxels.pt"
    guided_path = run_dir / "guided_encoder" / "encoded_slats.pt"
    output_dir = run_dir / "normal_encoder_roundtrip_control"
    normal_path = output_dir / "normal_shape_slat.pt"
    report_path = output_dir / "comparison.json"
    for required in (voxel_path, guided_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if report_path.is_file() and normal_path.is_file() and not args.force:
        print(report_path.read_text(), flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)

    print(f"[load] {voxel_path}", flush=True)
    voxels = torch.load(voxel_path, map_location="cpu", weights_only=False)
    coords = voxels["coords"].to(device=device, dtype=torch.int32)
    dual = voxels["dual_vertices"].to(device=device)
    intersected = voxels["intersected"].to(device=device)
    del voxels
    coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    leaf_token_count = int(coords4.shape[0])
    vertices = SparseTensor(dual, coords4)
    intersected_sparse = vertices.replace(intersected)
    print(f"[ordinary enc] C4096 tokens={coords4.shape[0]:,}; guide_subs=None", flush=True)

    started = time.time()
    encoder = pixal3d_models.from_pretrained(
        str(args.shape_encoder.expanduser().resolve())
    ).eval().to(device)
    normal = encoder(
        vertices,
        intersected_sparse,
        sample_posterior=False,
        guide_subs=None,
    )
    normal = SparseTensor(normal.feats, normal.coords)
    elapsed = time.time() - started
    print(f"[ordinary enc] C256 tokens={normal.coords.shape[0]:,} in {elapsed:.1f}s", flush=True)

    del encoder, vertices, intersected_sparse, dual, intersected, coords, coords4
    gc.collect()
    torch.cuda.empty_cache()

    guided_payload = torch.load(guided_path, map_location="cpu", weights_only=False)["shape"]
    guided_coords = guided_payload["coords"].to(device=device, dtype=normal.coords.dtype)
    guided_feats = guided_payload["feats"].to(device=device, dtype=normal.feats.dtype)
    coords_equal = torch.equal(normal.coords, guided_coords)
    if coords_equal and normal.feats.shape == guided_feats.shape:
        delta = (normal.feats - guided_feats).abs()
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        mean_abs = float(delta.float().mean().item()) if delta.numel() else 0.0
        feats_exact = torch.equal(normal.feats, guided_feats)
    else:
        max_abs = None
        mean_abs = None
        feats_exact = False

    report = {
        "experiment": "baseline_step0_decode_voxelize4096_ordinary_shape_enc",
        "encoder_guide_subs": None,
        "leaf_c4096_tokens": leaf_token_count,
        "ordinary_c256_tokens": int(normal.coords.shape[0]),
        "guided_c256_tokens": int(guided_coords.shape[0]),
        "coords_exactly_equal_to_guided": coords_equal,
        "features_exactly_equal_to_guided": feats_exact,
        "feature_max_abs_difference": max_abs,
        "feature_mean_abs_difference": mean_abs,
        "encoder_seconds": elapsed,
        "consequence": (
            "ordinary encoder output is bit-identical to guided encoder output; "
            "ordinary deterministic decode is therefore exactly the existing "
            "guided_endpoint_visualization control"
            if coords_equal and feats_exact
            else "ordinary and guided encoder outputs differ; decode/render is required"
        ),
        "existing_ordinary_decoder_render": str(
            (run_dir / "guided_endpoint_visualization" / "multiview" /
             "multiview_rgb_contact_sheet.png").resolve()
        ),
    }
    _atomic_save(normal_path, {
        "coords": normal.coords.cpu(),
        "feats": normal.feats.cpu(),
    })
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
