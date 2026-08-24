#!/usr/bin/env python3
"""Local-only overlapping-tile O-Voxel geometry test.

The local shape flow may use a cropped C4096 support to obtain the native
local SLat coordinate set, but the final global O-Voxel contains no baseline
cells, dual vertices, intersected flags, or Hermite observations.  It is
constructed only from transformed local-mesh Hermite observations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import pixal3d_ovoxel_hermite_qef_sr as base
from pixal3d.representations import MeshWithVertexPbr


DEFAULT_OUTPUT = Path("outputs/geometry_ovoxel_local_only_stride512")
DEFAULT_GLOBAL_CACHE = Path(
    "outputs/geometry_ovoxel_hermite_qef_sr/baseline_c4096/raw_ovoxel_hermite.pt"
)


def _parse_ids(value: str) -> Optional[set[int]]:
    if not value.strip():
        return None
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(token))
    return result


def _json_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _torch_observations(observations: Mapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value) for key, value in observations.items()}


def _load_global_support(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"[support] loading C4096 support only for local encoder crops: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _run_local_tiles(
    args: argparse.Namespace,
    output_dir: Path,
    global_support: Mapping[str, Any],
    layout: Sequence[Tuple[int, int, int]],
    tile_ids: Sequence[int],
    device: torch.device,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    image = Image.open(args.image).convert("RGB")
    successful: List[int] = []
    diagnostics: Dict[str, Any] = {}
    observation_items: List[Dict[str, np.ndarray]] = []

    cache_ready = bool(tile_ids) and not args.force_tiles and all(
        (output_dir / "local_tiles" / f"tile_{tile_id:03d}" / "global_hermite.pt").is_file()
        for tile_id in tile_ids
    )
    pipeline = None
    shape_encoder = None
    if not args.skip_local_flow and not cache_ready:
        print("[pipeline] loading native local shape flow components")
        pipeline = base._init_local_shape_pipeline(args.model_path, device, bool(args.low_vram))
        canonical = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        canonical.save(output_dir / "canonical_1024.png")
        import pixal3d.models as pixal3d_models
        shape_encoder = pixal3d_models.from_pretrained(str(args.shape_encoder)).eval()
        if not args.low_vram:
            shape_encoder.to(device)
        projector = pipeline.image_cond_model_shape_1024
        feature_cache = base._build_image_feature_cache(projector, canonical, device)
        base._atomic_torch_save(
            output_dir / "image_condition_global_token.pt",
            {"global": feature_cache["global"].cpu(), "source": "single original image feature extraction"},
        )
    elif cache_ready:
        print("[pipeline] all selected overlap tiles are cached; skipping local model reload")

    try:
        for tile_id in tile_ids:
            start = layout[tile_id]
            tile_dir = output_dir / "local_tiles" / f"tile_{tile_id:03d}"
            tile_dir.mkdir(parents=True, exist_ok=True)
            hermite_path = tile_dir / "global_hermite.pt"
            flow_path = tile_dir / "shape_flow_and_raw_ovoxel.pt"
            if hermite_path.is_file() and not args.force_tiles:
                h, tile_diag = base._load_tile_hermite(hermite_path)
                observation_items.append(h)
                successful.append(tile_id)
                diagnostics[str(tile_id)] = tile_diag
                print(f"[tile {tile_id:03d}] cache hit")
                continue
            if args.skip_local_flow:
                raise RuntimeError(f"tile {tile_id} is not cached and --skip-local-flow was set")
            crop = base._crop_global_ovoxel(
                global_support, start, int(args.tile_size), int(args.global_resolution)
            )
            transform_local = base._build_local_camera_transform(
                projector,
                float(args.camera_angle_x),
                float(args.camera_distance),
                start,
                int(args.tile_size),
                int(args.global_resolution),
                device,
            )
            tile_flow = base._run_local_shape_flow(
                pipeline=pipeline,
                shape_encoder=shape_encoder,
                crop=crop,
                condition=None,
                condition_model=projector,
                feature_cache=feature_cache,
                transform_local=transform_local,
                camera_angle_x=float(args.camera_angle_x),
                camera_distance=float(args.camera_distance),
                args=args,
                tile_id=tile_id,
            )
            base._atomic_torch_save(flow_path, tile_flow)
            tile_result, tile_diag = base._local_mesh_to_global_hermite(
                tile_flow,
                start,
                int(args.tile_size),
                int(args.global_resolution),
                args,
            )
            h = tile_result["hermite"]
            base._atomic_torch_save(
                hermite_path,
                {
                    "resolution": int(args.global_resolution),
                    "hermite": {
                        key: torch.from_numpy(value)
                        for key, value in h.items()
                        if key != "key"
                    },
                    "diagnostics": tile_diag,
                    "vertices": tile_result["vertices"],
                    "faces": tile_result["faces"],
                    "start": list(start),
                },
            )
            observation_items.append(h)
            successful.append(tile_id)
            diagnostics[str(tile_id)] = tile_diag
            print(
                f"[tile {tile_id:03d}] start={tuple(start)} "
                f"cells={crop['coords'].shape[0]:,} H={h['q'].shape[0]:,} "
                f"fallback={tile_diag['provenance_fallback_fraction']:.4f}"
            )
    finally:
        if shape_encoder is not None:
            del shape_encoder
        if pipeline is not None:
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    return observation_items, {
        "successful_tile_ids": successful,
        "tile_diagnostics": diagnostics,
    }


def _empty_global_qef(
    args: argparse.Namespace,
    output_dir: Path,
    local: Mapping[str, np.ndarray],
    selected: Mapping[str, np.ndarray],
    mode_stats: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    if local["q"].size == 0:
        raise RuntimeError("no local Hermite observations were produced")

    # ``selected`` was computed once by main(): choose the highest
    # confidence-weighted tau cluster per edge, with no baseline tau anchor.
    # Do not repeat this O(N log N) operation for the 20--100M-row local table.
    weighted = not args.unweighted
    observations = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in selected.items()
    }
    observations["weight"] = (
        observations["confidence"] * observations["tile_weight"]
        if weighted
        else np.ones(observations["q"].shape[0], dtype=np.float32)
    ).astype(np.float32, copy=False)
    active_keys = np.unique(observations["key"].astype(np.int64, copy=False))
    cells = np.unique(
        base._edge_cells(observations["edge_coord"], observations["edge_axis"], int(args.global_resolution)),
        axis=0,
    ).astype(np.int32, copy=False)
    if cells.size == 0:
        raise RuntimeError("local Hermite observations did not create any global cells")

    started = time.perf_counter()
    aggregate = base._aggregate_qef(
        observations,
        cells,
        int(args.global_resolution),
        float(args.regularization_weight),
        int(args.qef_observation_chunk),
        device=device,
    )
    solved = base._solve_qef_batches(
        aggregate,
        int(args.global_resolution),
        int(args.qef_batch_size),
        device=device,
    )
    residual = base._observation_residuals(
        observations,
        solved["vertices"],
        cells,
        int(args.global_resolution),
        int(args.qef_observation_chunk),
        device=device,
    )
    intersected = base._final_intersected(
        cells, active_keys, int(args.global_resolution)
    )
    dual_cell = (
        solved["vertices"] + 0.5
    ) * float(args.global_resolution) - cells.astype(np.float32)
    dual_cell = np.nan_to_num(dual_cell, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    edge_keys, edge_counts = np.unique(observations["key"], return_counts=True)
    overlap_stats = {
        "edge_count": int(edge_keys.size),
        "mean_observations_per_edge": float(edge_counts.mean()),
        "p95_observations_per_edge": float(np.percentile(edge_counts, 95)),
        "max_observations_per_edge": int(edge_counts.max(initial=0)),
        "edges_with_multiple_observations": int((edge_counts > 1).sum()),
        "multiple_observation_fraction": float((edge_counts > 1).mean()) if edge_counts.size else 0.0,
    }
    stats = {
        "variant": "local_only_unweighted" if args.unweighted else "local_only_weighted",
        "baseline_content_in_final_global_ovoxel": False,
        "baseline_observation_count": 0,
        "baseline_active_edge_count": 0,
        "global_resolution": int(args.global_resolution),
        "tile_size": int(args.tile_size),
        "tile_stride": int(args.tile_stride),
        "cell_count": int(cells.shape[0]),
        "active_edge_count": int(active_keys.shape[0]),
        "observation_count": int(observations["q"].shape[0]),
        "selected_observation_count": int(selected["q"].shape[0]),
        "local_only_total_weight": float(observations["weight"].sum()),
        "qef_residual_mean": float(residual["mean"]),
        "qef_residual_p95": float(residual["p95"]),
        "qef_residual_max": float(residual["max"]),
        "qef_rank_histogram": solved["rank_histogram"],
        "qef_clamped_count": int(solved["clamped_count"]),
        "qef_seconds": float(time.perf_counter() - started),
        "tau_mode_selection": dict(mode_stats),
        **overlap_stats,
    }
    payload = {
        "resolution": int(args.global_resolution),
        "coords": torch.from_numpy(cells),
        "dual_vertices_cell": torch.from_numpy(dual_cell.astype(np.float32)),
        "dual_vertices_object": torch.from_numpy(solved["vertices"].astype(np.float32)),
        "intersected": torch.from_numpy(intersected),
        "active_edge_keys": torch.from_numpy(active_keys),
        "observations": _torch_observations(observations),
        "qef": {
            "A": torch.from_numpy(aggregate["A"]),
            "b": torch.from_numpy(aggregate["b"]),
            "mean": torch.from_numpy(aggregate["mean"]),
            "weight_sum": torch.from_numpy(aggregate["weight_sum"]),
            "residual_per_cell": torch.from_numpy(residual["per_cell"]),
            "rank_histogram": solved["rank_histogram"],
            "clamped_count": int(solved["clamped_count"]),
        },
        "stats": stats,
    }
    base._atomic_torch_save(output_dir / "local_only_qef.pt", payload)
    base._atomic_json(output_dir / "geometry_diagnostics.json", stats)
    print(
        f"[local-only] cells={cells.shape[0]:,} edges={active_keys.shape[0]:,} "
        f"selected_obs={selected['q'].shape[0]:,} "
        f"multi_edge_fraction={overlap_stats['multiple_observation_fraction']:.4f} "
        f"residual_p95={residual['p95']:.4e}"
    )
    return payload


def _render_normal(
    args: argparse.Namespace,
    output_dir: Path,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
) -> None:
    from render_pixal3d_raw_ovoxel import load_envmap

    attrs = torch.zeros((vertices.shape[0], 6), device=device, dtype=torch.float32)
    attrs[:, :3] = 0.7
    attrs[:, 3] = 0.0
    attrs[:, 4] = 0.5
    attrs[:, 5] = 1.0
    mesh = MeshWithVertexPbr(vertices, faces, attrs, base.PBR_LAYOUT)
    camera = {
        "camera_angle_x": float(args.camera_angle_x),
        "distance": float(args.camera_distance),
    }
    extrinsics, intrinsics = base._make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), device
    )
    renderer = __import__("pixal3d.renderers", fromlist=["PbrMeshRenderer"]).PbrMeshRenderer(
        rendering_options={
            "resolution": int(args.render_resolution),
            "near": max(0.01, float(args.camera_distance) - 2.0),
            "far": float(args.camera_distance) + 10.0,
            "ssaa": 1,
            "peel_layers": 6,
            "face_chunk_size": 4_000_000,
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)
    for angle in base.YAW_ANGLES:
        print(f"[normal-render] yaw={angle}")
        rendered = renderer.render(
            mesh,
            extrinsics[angle],
            intrinsics,
            envmap=envmap,
            use_envmap_bg=False,
        )
        normal = rendered["normal"].detach().float().cpu()
        normal = normal.permute(1, 2, 0).numpy()
        mask = rendered["mask"].detach().float().cpu()
        if mask.ndim == 3:
            mask = mask.permute(1, 2, 0).numpy()
        else:
            mask = mask.numpy()
        base._save_image(
            normal * mask[..., None],
            output_dir / "normal_renders" / f"yaw{angle:03d}.png",
        )
        del rendered
        torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=base.DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--global-support", type=Path, default=DEFAULT_GLOBAL_CACHE)
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--shape-encoder", type=Path, default=base.DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-resolution", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--tile-ids", type=str, default="")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-active-cells", type=int, default=1)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-local-flow", action="store_true")
    parser.add_argument("--force-tiles", action="store_true")
    parser.add_argument("--tau-cluster-threshold", type=float, default=0.08)
    parser.add_argument("--edge-temperature", type=float, default=1.0)
    parser.add_argument("--face-weight", type=float, default=1.0)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--tile-boundary-band", type=float, default=0.15)
    parser.add_argument("--qef-observation-chunk", type=int, default=5_000_000)
    parser.add_argument("--qef-batch-size", type=int, default=262_144)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--render-normal", action="store_true")
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--camera-angle-x", type=float, default=0.517371749106554)
    parser.add_argument("--camera-distance", type=float, default=1.889538288116455)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.global_resolution != 4096 or args.tile_size != 1024 or args.tile_stride != 512:
        raise ValueError("this test requires global_resolution=4096, tile_size=1024, tile_stride=512")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.cuda_device < 0 or args.cuda_device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GPU] cuda:{args.cuda_device} {torch.cuda.get_device_name(args.cuda_device)}")
    base._atomic_json(output_dir / "config.json", _json_args(args))
    global_support = _load_global_support(args.global_support)
    layout, active_ids, active_counts = base._active_tile_ids(
        global_support,
        int(args.global_resolution),
        int(args.tile_size),
        int(args.tile_stride),
        int(args.min_active_cells),
    )
    requested = _parse_ids(args.tile_ids)
    if requested is not None:
        active_ids = [tile_id for tile_id in active_ids if tile_id in requested]
    if args.max_tiles is not None:
        active_ids = active_ids[: int(args.max_tiles)]
    print(
        f"[tiles] candidates={len(layout)} global-active={sum(c >= args.min_active_cells for c in active_counts)} "
        f"selected={len(active_ids)} stride=512"
    )
    base._atomic_json(output_dir / "tile_layout.json", {
        "candidate_count": len(layout),
        "global_active_count": int(sum(c >= args.min_active_cells for c in active_counts)),
        "selected_tile_ids": active_ids,
        "selected_starts": [list(layout[tile_id]) for tile_id in active_ids],
        "active_cell_counts": [int(active_counts[tile_id]) for tile_id in active_ids],
    })

    items, tile_summary = _run_local_tiles(
        args, output_dir, global_support, layout, active_ids, device
    )
    local = base._concat_observations(items)
    base._atomic_torch_save(
        output_dir / "local_hermite_observations.pt",
        _torch_observations(local),
    )
    empty = base._concat_observations([])
    selected, mode_stats = base._select_tau_modes(
        empty, local, int(args.global_resolution), float(args.tau_cluster_threshold)
    )
    base._atomic_json(output_dir / "mode_selection.json", mode_stats)
    payload = _empty_global_qef(args, output_dir, local, selected, mode_stats, device)
    vertices, faces = base._decode_final_mesh(payload, device)
    base._atomic_torch_save(
        output_dir / "local_only_final_mesh.pt",
        {"vertices": vertices.cpu(), "faces": faces.cpu(), "representation": "local-only-empty-global-qef"},
    )
    base._atomic_json(output_dir / "local_only_mesh_stats.json", {
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "baseline_content_in_final_global_ovoxel": False,
    })
    if args.render_normal:
        _render_normal(args, output_dir, vertices, faces, device)
    summary = {
        "format": "pixal3d_local_only_empty_global_overlap_v1",
        "cuda_device": int(args.cuda_device),
        "gpu": torch.cuda.get_device_name(args.cuda_device),
        "global_resolution": int(args.global_resolution),
        "tile_size": int(args.tile_size),
        "tile_stride": int(args.tile_stride),
        "candidate_tile_count": len(layout),
        "global_active_tile_count": int(sum(c >= args.min_active_cells for c in active_counts)),
        "selected_tile_count": len(active_ids),
        "successful_tile_count": len(tile_summary["successful_tile_ids"]),
        "selected_tile_ids": active_ids,
        "local_observation_count": int(local["q"].shape[0]),
        "baseline_content_in_final_global_ovoxel": False,
        "baseline_used_only_for_local_encoder_support": True,
        "qef": payload["stats"],
        "mesh": {"vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])},
        "normal_render": bool(args.render_normal),
    }
    base._atomic_json(output_dir / "summary.json", summary)
    print(f"[done] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
