#!/usr/bin/env python3
"""Encode the cached C4096 endpoint texture without guided support alignment."""
from __future__ import annotations

import argparse
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
    "microsoft/TRELLIS___2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
)


def _atomic_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--texture-encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    voxel_path = run_dir / "voxel4096" / "paired_endpoint_voxels.pt"
    guided_path = run_dir / "guided_encoder" / "encoded_slats.pt"
    output_path = run_dir / "normal_encoder_roundtrip_control" / "normal_texture_slat.pt"
    report_path = run_dir / "normal_encoder_roundtrip_control" / "texture_comparison.json"
    if output_path.is_file() and report_path.is_file() and not args.force:
        print(report_path.read_text(), flush=True)
        return 0

    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    voxels = torch.load(voxel_path, map_location="cpu", weights_only=False)
    coords = voxels["coords"].to(device=device, dtype=torch.int32)
    attrs = voxels["attrs"].to(device=device)
    del voxels
    coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    pbr = SparseTensor(attrs * 2 - 1, coords4)
    print(f"[ordinary texture enc] C4096 tokens={coords4.shape[0]:,}; guide_subs=None", flush=True)

    started = time.time()
    encoder = pixal3d_models.from_pretrained(
        str(args.texture_encoder.expanduser().resolve())
    ).eval().to(device)
    normal = encoder(pbr, sample_posterior=False, guide_subs=None)
    normal = SparseTensor(normal.feats, normal.coords)
    elapsed = time.time() - started

    guided = torch.load(guided_path, map_location="cpu", weights_only=False)["texture"]
    guided_coords = guided["coords"].to(device=device, dtype=normal.coords.dtype)
    guided_feats = guided["feats"].to(device=device, dtype=normal.feats.dtype)
    coords_equal = torch.equal(normal.coords, guided_coords)
    if coords_equal:
        delta = (normal.feats - guided_feats).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.float().mean().item())
        changed_rows = int((delta.amax(dim=1) > 0).sum().item())
    else:
        max_abs = mean_abs = None
        changed_rows = None
    report = {
        "experiment": "baseline_step0_texture_voxel4096_ordinary_encoder",
        "encoder_guide_subs": None,
        "leaf_c4096_tokens": int(coords4.shape[0]),
        "ordinary_c256_tokens": int(normal.coords.shape[0]),
        "guided_c256_tokens": int(guided_coords.shape[0]),
        "coords_exactly_equal_to_guided": coords_equal,
        "features_exactly_equal_to_guided": (
            bool(torch.equal(normal.feats, guided_feats)) if coords_equal else False
        ),
        "changed_feature_rows": changed_rows,
        "feature_max_abs_difference": max_abs,
        "feature_mean_abs_difference": mean_abs,
        "encoder_seconds": elapsed,
    }
    _atomic_save(output_path, {"coords": normal.coords.cpu(), "feats": normal.feats.cpu()})
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
