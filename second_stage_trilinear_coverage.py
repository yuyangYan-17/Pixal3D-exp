#!/usr/bin/env python3
"""Measure second-stage sparse-trilinear coverage after first-stage texture follow.

The second-stage bridge queries a decoded sparse material field at the dual
points of a 4096 voxelization.  This script reproduces that query with the
same ``grid_sample_3d(..., mode='trilinear')`` implementation and reports
both point-level and C256-parent coverage.  It intentionally does not use a
nearest-face fallback, so the raw missing rate is visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bridge-voxels",
        type=Path,
        default=Path("outputs/sr_0_img_visibility_guidance/shared/bridge/voxels.pt"),
    )
    p.add_argument(
        "--bridge-ancestry",
        type=Path,
        default=Path("outputs/sr_0_img_visibility_guidance/shared/bridge/ancestry.pt"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/first_stage_texture_follow_20260919/second_stage_trilinear"),
    )
    p.add_argument("--gpu", default="1")
    p.add_argument("--chunk", type=int, default=262144)
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
# Use a fresh per-test autotune cache.  Reusing the multi-GPU project cache can
# select a stale Triton binary/configuration after a different physical card
# has populated it, which is unrelated to sparse support coverage.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = (
    "/tmp/flex_gemm_second_stage_coverage_20260919_c.json"
)
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "0"

import torch

from sr_tools import common
from sr_tools.texture_guidance import _InferenceSparseTrilinearIndex


def save_json(path: Path, value):
    common.js(path, value)


def field_from_mesh(path: Path):
    payload = common.load_payload(path)
    mesh = payload["mesh"]
    coords = torch.cat(
        (torch.zeros_like(mesh.coords[:, :1]), mesh.coords.int()), dim=1
    ).contiguous()
    attrs = mesh.attrs.float().contiguous()
    resolution = int(round(1.0 / float(mesh.voxel_size)))
    record = dict(
        source=str(path.resolve()),
        resolution=resolution,
        field_points=len(coords),
        channels=int(attrs.shape[1]),
        mesh_vertices=len(mesh.vertices),
        mesh_faces=len(mesh.faces),
    )
    del mesh, payload
    return coords, attrs, record


@torch.no_grad()
def measure(name, mesh_path: Path, dual: torch.Tensor, parents: torch.Tensor, n_parents: int, out: Path):
    coords, attrs, record = field_from_mesh(mesh_path)
    index = _InferenceSparseTrilinearIndex(
        coords,
        resolution=record["resolution"],
        device="cuda:0",
        chunk=ARGS.chunk,
    )
    augmented = torch.cat((attrs.to(index.device), index.ones_device), dim=1).contiguous()
    n = len(dual)
    native_count = 0
    support_sum = 0.0
    support_min = float("inf")
    support_max = 0.0
    support_samples = []
    parent_hits = torch.zeros(n_parents, dtype=torch.float64)
    parent_counts = torch.zeros(n_parents, dtype=torch.float64)
    print(
        f"[{name}] field={record['field_points']:,} res={record['resolution']} "
        f"query={n:,} C256={n_parents:,}",
        flush=True,
    )
    for start in range(0, n, ARGS.chunk):
        end = min(start + ARGS.chunk, n)
        # _sample_augmented is the low-level path; unlike query(), it expects
        # the query grid on the same CUDA device as the sparse field.
        points = (dual[start:end].float() - 0.5).to(index.device)
        _, support = index._sample_augmented(augmented, points)
        support = support.float()
        native = support > 0
        native_count += int(native.sum().item())
        support_sum += float(support.sum().item())
        support_min = min(support_min, float(support.min().item()))
        support_max = max(support_max, float(support.max().item()))
        stride = max(1, len(support) // 2048)
        support_samples.append(support[::stride].cpu())
        p = parents[start:end]
        parent_counts.index_add_(0, p, torch.ones(len(p), dtype=torch.float64))
        parent_hits.index_add_(0, p, native.double().cpu())
        if (start // ARGS.chunk) % 32 == 0 or end == n:
            print(
                f"[{name}] {end:,}/{n:,} native={native_count / end:.6f}",
                flush=True,
            )
        del points, support, native, p
    samples = torch.cat(support_samples)
    parent_fraction = parent_hits / parent_counts.clamp_min(1.0)
    result = dict(
        **record,
        query_points=n,
        native_trilinear_points=native_count,
        native_trilinear_fraction=native_count / n,
        missing_points=n - native_count,
        missing_fraction=1.0 - native_count / n,
        support_weight_mean=support_sum / n,
        support_weight_min=support_min,
        support_weight_max=support_max,
        support_weight_p01=float(torch.quantile(samples, 0.01)),
        support_weight_p50=float(torch.quantile(samples, 0.50)),
        support_weight_p99=float(torch.quantile(samples, 0.99)),
        c256_parents=n_parents,
        c256_all_points_supported=int((parent_fraction >= 1.0).sum()),
        c256_all_points_supported_fraction=float((parent_fraction >= 1.0).float().mean()),
        c256_partial_or_supported_fraction=float((parent_fraction > 0).float().mean()),
        c256_zero_support=int((parent_fraction == 0).sum()),
        query_method="same _InferenceSparseTrilinearIndex as inference/grid_sample_3d",
        fallback_used=False,
    )
    save_json(out / f"{name}.json", result)
    del augmented, index, coords, attrs, samples, support_samples
    del parent_hits, parent_counts, parent_fraction
    common.empty_cuda()
    gc.collect()
    return result


def main():
    torch.set_num_threads(8)
    out = ARGS.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    vox = common.load_payload(ARGS.bridge_voxels.resolve())
    dual = vox["dual"].float().contiguous()
    ancestry = common.load_payload(ARGS.bridge_ancestry.resolve())
    parents = ancestry["voxel_to_c256"].long().contiguous()
    if len(parents) != len(dual):
        raise RuntimeError("bridge ancestry length does not match 4096 dual points")
    n_parents = int(parents.max().item()) + 1

    fields = {
        "native_1024_control": Path("outputs/sr_0_img_z_tent/baseline/textured_mesh.pt"),
        "first_stage_baseline_2048": Path(
            "outputs/first_stage_texture_follow_20260919/baseline_geometry/textured_mesh.pt"
        ),
        "first_stage_flow_2048": Path(
            "outputs/first_stage_texture_follow_20260919/shape512_flow_geometry/textured_mesh.pt"
        ),
    }
    records = {}
    for name, path in fields.items():
        if not path.exists():
            raise FileNotFoundError(path)
        records[name] = measure(name, path.resolve(), dual, parents, n_parents, out)
    summary = dict(
        status="COMPLETE",
        gpu=str(ARGS.gpu),
        query_resolution=4096,
        query_points=len(dual),
        c256_parents=n_parents,
        bridge_voxels=str(ARGS.bridge_voxels.resolve()),
        bridge_geometry_note="the bridge was generated from the same first-stage Shape512 flow 2048 geometry; vertices/faces were checked equal",
        records=records,
        interpretation="native_trilinear_fraction is the raw second-stage query hit rate before any nearest-face fallback",
    )
    save_json(out / "result.json", summary)
    print("[done]", out, flush=True)


if __name__ == "__main__":
    main()
