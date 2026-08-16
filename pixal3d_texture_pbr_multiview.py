"""Render saved texture-only PBR variants from several fixed views.

The experiment stores the sparse endpoints and sampled fields, not the full
decoded meshes. This utility therefore reuses the official decoder to
reconstruct each requested tile, sends the resulting MeshWithVertexPbr
objects through the official PbrMeshRenderer, and saves per-variant view
images plus contact sheets.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import utils3d
from PIL import Image, ImageDraw, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_texture_pbr_degradation_experiment as experiment
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr
from pixal3d.utils import render_utils


VARIANT_ORDER = (
    "G",
    "G_low",
    "G_low_HR_high",
    "HR",
    "HR_low",
    "HR_low_G_high",
    "latent_interp_25",
    "latent_interp_50",
    "latent_interp_75",
)

FIXED_VIEWS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", -90.0, 0.0),
    ("top", 0.0, 75.0),
    ("bottom", 0.0, -75.0),
)


def _empty_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        render_face_chunk_size=int(args.face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
    )


def _fixed_cameras(camera: Mapping[str, float], args: argparse.Namespace):
    device = torch.device("cuda")
    radius = float(camera["distance"]) * float(args.radius_scale)
    fov = torch.tensor(float(camera["camera_angle_x"]), device=device)
    intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    extrinsics = []
    intrinsics = []
    labels = []
    for label, yaw_degrees, pitch_degrees in FIXED_VIEWS:
        yaw = math.radians(yaw_degrees)
        pitch = math.radians(pitch_degrees)
        direction = torch.tensor(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            device=device,
            dtype=torch.float32,
        )
        position = target + direction * radius
        extrinsics.append(utils3d.torch.extrinsics_look_at(position, target, up))
        intrinsics.append(intrinsic)
        labels.append(label)
    options = {
        "resolution": int(args.resolution),
        "near": max(0.01, radius - 2.0),
        "far": radius + 10.0,
        "ssaa": int(args.ssaa),
        "peel_layers": int(args.peel_layers),
        "face_chunk_size": int(args.face_chunk_size),
    }
    return extrinsics, intrinsics, labels, options


@torch.no_grad()
def _render_variant_frames(
    sample: MeshWithVertexPbr,
    *,
    renderer: Any,
    extrinsics: Sequence[torch.Tensor],
    intrinsics: Sequence[torch.Tensor],
    options: Mapping[str, Any],
    envmap: Any,
    args: argparse.Namespace,
) -> List[np.ndarray]:
    live = sample.to("cuda")
    rendered = render_utils.render_frames(
        live,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        options=dict(options),
        verbose=True,
        renderer=renderer,
        envmap=envmap,
        use_envmap_bg=bool(args.use_envmap_bg),
    )
    del live
    _empty_cuda_cache()
    frames = rendered.get("shaded")
    if frames is None or len(frames) != len(extrinsics):
        raise RuntimeError("official renderer returned incomplete multi-view frames")
    return [np.asarray(frame).astype(np.uint8) for frame in frames]


def _save_variant_sheet(
    *,
    output_path: Path,
    frame_paths: Mapping[str, Sequence[Path]],
    variants: Sequence[str],
    view_labels: Sequence[str],
    panel: int,
) -> None:
    header = 42
    row_header = 126
    width = row_header + len(view_labels) * panel
    height = header + len(variants) * (panel + 30)
    sheet = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(view_labels):
        x = row_header + col * panel
        draw.text((x + 6, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(variants):
        y = header + row_index * (panel + 30)
        draw.text((8, y + panel // 2 - 8), variant, fill=(255, 255, 255))
        for col, view_label in enumerate(view_labels):
            path = frame_paths[variant][col]
            with Image.open(path) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            sheet.paste(image, (x + (panel - image.width) // 2, y + (panel - image.height) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _save_tile_overview(
    *,
    output_path: Path,
    tile_paths: Mapping[int, Mapping[str, Sequence[Path]]],
    variants: Sequence[str],
    view_label: str,
    panel: int,
) -> None:
    successful = sorted(tile_paths)
    header = 46
    row_header = 86
    width = row_header + len(variants) * panel
    height = header + len(successful) * (panel + 24)
    sheet = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(sheet)
    for col, variant in enumerate(variants):
        x = row_header + col * panel
        draw.text((x + 4, 15), variant, fill=(255, 255, 255))
    draw.text((5, 15), view_label, fill=(255, 255, 255))
    for row_index, tile_id in enumerate(successful):
        y = header + row_index * (panel + 24)
        draw.text((8, y + panel // 2 - 8), f"tile {tile_id:02d}", fill=(255, 255, 255))
        for col, variant in enumerate(variants):
            paths = tile_paths[tile_id][variant]
            view_index = 0 if view_label == "front" else 2
            with Image.open(paths[view_index]) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            sheet.paste(image, (x + (panel - image.width) // 2, y + (panel - image.height) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _decode_variant(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_norm: torch.Tensor,
    coords: torch.Tensor,
    label: str,
    query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    device = torch.device("cuda")
    latent = SparseTensor(texture_norm.to(device=device), coords.to(device=device, dtype=torch.int32))
    empty = torch.empty((0, 3), device=device, dtype=torch.float32)
    mesh, _, stats = experiment._decode_and_query(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_latent_norm=latent,
        normalization=pipeline.tex_slat_normalization,
        query_points_device=empty,
        resolution=experiment.OVOXEL_RESOLUTION,
        query_chunk_size=int(query_chunk_size),
        label=label,
    )
    vertices = mesh.vertices.detach().cpu().float()
    faces = mesh.faces.detach().cpu().int()
    attrs = experiment._query_common_fields(mesh, mesh.vertices, int(query_chunk_size)).detach().cpu().float()
    del latent, mesh
    _empty_cuda_cache()
    return vertices, faces, attrs, stats


def _run_tile(
    *,
    pipeline: Any,
    row: Mapping[str, Any],
    output_root: Path,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    tile_id = int(row["tile_id"])
    tile_dir = output_root / f"tile_{tile_id:02d}"
    endpoints_path = tile_dir.parent.parent / "tiles" / f"tile_{tile_id:02d}" / "endpoints.pt"
    endpoints = torch.load(str(endpoints_path), map_location="cpu")
    device = torch.device("cuda")
    shape_coords = endpoints["shape_coords"].to(device=device, dtype=torch.int32)
    shape_norm = SparseTensor(
        endpoints["shape_norm"].to(device=device, dtype=torch.float32),
        shape_coords,
    )
    shape_denorm = experiment._denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
    texture_coords = endpoints["g_tex_coords"].to(device=device, dtype=torch.int32)
    g_norm = endpoints["g_tex_norm"].float()
    hr_norm = endpoints["hr_tex_norm"].float()
    transform = core.TileCameraTransform(**endpoints["transform"])
    started = time.perf_counter()

    g_vertices, g_faces, g_attrs, g_stats = _decode_variant(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_norm=g_norm,
        coords=texture_coords,
        label=f"tile_{tile_id:02d} G multiview",
        query_chunk_size=int(args.query_chunk_size),
    )
    hr_vertices, hr_faces, hr_attrs, hr_stats = _decode_variant(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_norm=hr_norm,
        coords=texture_coords,
        label=f"tile_{tile_id:02d} HR multiview",
        query_chunk_size=int(args.query_chunk_size),
    )
    if g_vertices.shape != hr_vertices.shape or not torch.equal(g_vertices, hr_vertices):
        raise RuntimeError(f"tile {tile_id}: G/HR decoded geometry mismatch")
    if not torch.equal(g_faces, hr_faces):
        raise RuntimeError(f"tile {tile_id}: G/HR decoded face topology mismatch")

    global_positions = experiment._map_local_to_global_chunked(
        g_vertices,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    ).detach().cpu().float()
    ids = experiment._coarse_cell_ids(global_positions, 1)
    g_low, _ = experiment._project_cell_mean(g_attrs, ids)
    hr_low, _ = experiment._project_cell_mean(hr_attrs, ids)
    fields: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
        "G": (g_vertices, g_faces, g_attrs),
        "G_low": (g_vertices, g_faces, g_low),
        "G_low_HR_high": (g_vertices, g_faces, g_low + (hr_attrs - hr_low)),
        "HR": (hr_vertices, hr_faces, hr_attrs),
        "HR_low": (g_vertices, g_faces, hr_low),
        "HR_low_G_high": (g_vertices, g_faces, hr_low + (g_attrs - g_low)),
    }

    interp_stats: Dict[str, Any] = {}
    for alpha, name in ((0.25, "latent_interp_25"), (0.50, "latent_interp_50"), (0.75, "latent_interp_75")):
        vertices, faces, attrs, stats = _decode_variant(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=g_norm + float(alpha) * (hr_norm - g_norm),
            coords=texture_coords,
            label=f"tile_{tile_id:02d} {name} multiview",
            query_chunk_size=int(args.query_chunk_size),
        )
        fields[name] = (vertices, faces, attrs)
        interp_stats[name] = stats

    extrinsics, intrinsics, view_labels, options = _fixed_cameras(
        {
            "camera_angle_x": float(transform.camera_angle_x),
            "distance": float(transform.distance),
        },
        args,
    )
    reference_sample = MeshWithVertexPbr(
        g_vertices,
        g_faces,
        g_attrs,
        layout=dict(experiment.PBR_LAYOUT),
    )
    renderer = render_utils.get_renderer(reference_sample, **options)
    tile_paths: Dict[str, Sequence[Path]] = {}
    for variant in VARIANT_ORDER:
        vertices, faces, attrs = fields[variant]
        sample = MeshWithVertexPbr(vertices, faces, attrs, layout=dict(experiment.PBR_LAYOUT))
        frames = _render_variant_frames(
            sample,
            renderer=renderer,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            options=options,
            envmap=envmap,
            args=args,
        )
        variant_dir = tile_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for frame, view_label in zip(frames, view_labels):
            path = variant_dir / f"{view_label}.png"
            Image.fromarray(frame).convert("RGB").save(path)
            paths.append(path)
        tile_paths[variant] = paths
        del sample, frames
        _empty_cuda_cache()

    sheet_path = tile_dir / f"tile_{tile_id:02d}_multiview_sheet.png"
    _save_variant_sheet(
        output_path=sheet_path,
        frame_paths=tile_paths,
        variants=VARIANT_ORDER,
        view_labels=view_labels,
        panel=int(args.sheet_panel),
    )
    result = {
        "status": "success",
        "tile_id": tile_id,
        "views": list(view_labels),
        "resolution": int(args.resolution),
        "ssaa": int(args.ssaa),
        "peel_layers": int(args.peel_layers),
        "renderer": "official pixal3d.utils.render_utils -> PbrMeshRenderer -> nvdiffrast",
        "variants": {
            name: {
                "frames": [str(path) for path in paths],
                "sheet": str(sheet_path),
            }
            for name, paths in tile_paths.items()
        },
        "sheet": str(sheet_path),
        "decode_stats": {
            "G": g_stats,
            "HR": hr_stats,
            **interp_stats,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(tile_dir / "multiview_summary.json", result)
    del shape_norm, shape_denorm, g_vertices, g_faces, g_attrs, hr_vertices, hr_faces, hr_attrs
    del global_positions, ids, g_low, hr_low, fields, renderer, reference_sample
    _empty_cuda_cache()
    return result


def run(args: argparse.Namespace) -> None:
    torch.cuda.set_device(int(args.cuda_device))
    source_dir = Path(args.experiment_dir).expanduser().resolve()
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    selected = _parse_ids(args.tile_ids)
    output_root = source_dir / "multiview_quality"
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    all_paths: Dict[int, Dict[str, Sequence[Path]]] = {}
    tile_summaries = []
    for row in summary.get("tiles", []):
        tile_id = int(row["tile_id"])
        if row.get("status") != "success":
            continue
        if selected is not None and tile_id not in selected:
            continue
        tile_output = output_root / f"tile_{tile_id:02d}"
        cached = tile_output / "multiview_summary.json"
        if bool(args.resume) and cached.is_file():
            result = json.loads(cached.read_text(encoding="utf-8"))
            print(f"[multiview tile {tile_id:02d}] reused")
        else:
            print(f"[multiview tile {tile_id:02d}] rendering {len(VARIANT_ORDER)} variants")
            try:
                result = _run_tile(
                    pipeline=pipeline,
                    row=row,
                    output_root=output_root,
                    global_camera=global_camera,
                    args=args,
                    envmap=envmap,
                )
            except Exception as exc:
                result = {"status": "failed", "tile_id": tile_id, "reason": f"{type(exc).__name__}: {exc}"}
                _write_json(tile_output / "multiview_summary.json", result)
                print(f"[multiview tile {tile_id:02d}] FAILED: {result['reason']}")
        if result.get("status") == "success":
            all_paths[tile_id] = {
                variant: [Path(path) for path in result["variants"][variant]["frames"]]
                for variant in VARIANT_ORDER
            }
        tile_summaries.append(result)

    if all_paths:
        _save_tile_overview(
            output_path=output_root / "all_tiles_front_overview.png",
            tile_paths=all_paths,
            variants=VARIANT_ORDER,
            view_label="front",
            panel=int(args.overview_panel),
        )
        _save_tile_overview(
            output_path=output_root / "all_tiles_back_overview.png",
            tile_paths=all_paths,
            variants=VARIANT_ORDER,
            view_label="back",
            panel=int(args.overview_panel),
        )
    output = {
        "format": "pixal3d_texture_pbr_multiview_v1",
        "source_experiment": str(source_dir),
        "cuda_device": int(args.cuda_device),
        "views": [label for label, _, _ in FIXED_VIEWS],
        "variant_order": list(VARIANT_ORDER),
        "resolution": int(args.resolution),
        "ssaa": int(args.ssaa),
        "peel_layers": int(args.peel_layers),
        "successful_tiles": sorted(all_paths),
        "failed_tiles": [row.get("tile_id") for row in tile_summaries if row.get("status") == "failed"],
        "tile_summaries": tile_summaries,
        "front_overview": str(output_root / "all_tiles_front_overview.png") if all_paths else None,
        "back_overview": str(output_root / "all_tiles_back_overview.png") if all_paths else None,
    }
    _write_json(output_root / "multiview_summary.json", output)
    print(f"[done] multiview output={output_root}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-chunk-size", type=int, default=32_768)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=1)
    parser.add_argument("--peel-layers", type=int, default=4)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--sheet-panel", type=int, default=256)
    parser.add_argument("--overview-panel", type=int, default=160)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    return parser


if __name__ == "__main__":
    run(_build_parser().parse_args())
