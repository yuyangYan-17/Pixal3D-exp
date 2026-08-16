#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-render saved global merged PBR meshes at 4096 and recompute metrics.

This is deliberately a render-only pass.  It loads the already stitched
``MeshWithVertexPbr`` checkpoints, uses Pixal3D's official PBR renderer for
the aligned camera and the six fixed global views, and compares the aligned
render with ``canonical_4096.png`` at the full 4096x4096 metric resolution.
No latent decode, tile transform, overlap ownership, or mesh stitching is
performed here.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_texture_pbr_global_stitch as stitch
import pixal3d_texture_pbr_multiview as multiview
from inference import MODEL_PATH
from pixal3d.representations import MeshWithVertexPbr
from pixal3d.utils import render_utils


VARIANT_ORDER = tuple(multiview.VARIANT_ORDER)
VIEW_LABELS = tuple(label for label, _, _ in multiview.FIXED_VIEWS)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_mesh(path: Path) -> MeshWithVertexPbr:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    mesh = payload.get("mesh") if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVertexPbr):
        raise TypeError(f"{path} does not contain MeshWithVertexPbr: {type(mesh)!r}")
    return mesh.to("cpu")


def _parse_variants(value: Optional[str]) -> List[str]:
    if value is None or not value.strip():
        return list(VARIANT_ORDER)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in VARIANT_ORDER]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choices={list(VARIANT_ORDER)}")
    return [item for item in VARIANT_ORDER if item in set(requested)]


def _render_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        render_resolution=int(args.resolution),
        metric_resolution=int(args.metric_resolution),
        render_ssaa=int(args.ssaa),
        render_peel_layers=int(args.peel_layers),
        render_face_chunk_size=int(args.face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        envmap=str(args.envmap),
        lpips_net="vgg",
        skip_lpips=bool(args.skip_lpips),
    )


def _save_frame(path: Path, frame: torch.Tensor) -> None:
    if frame.dim() == 2:
        frame = frame[None].repeat(3, 1, 1)
    array = np.clip(
        frame.detach().cpu().numpy().transpose(1, 2, 0) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).convert("RGB").save(path)


@torch.no_grad()
def _render_multiview_4096(
    mesh: MeshWithVertexPbr,
    *,
    global_camera: Mapping[str, float],
    envmap: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, str]:
    mv_args = SimpleNamespace(
        resolution=int(args.resolution),
        ssaa=int(args.ssaa),
        peel_layers=int(args.peel_layers),
        face_chunk_size=int(args.face_chunk_size),
        radius_scale=float(args.radius_scale),
        use_envmap_bg=bool(args.use_envmap_bg),
    )
    extrinsics, intrinsics, labels, options = multiview._fixed_cameras(
        global_camera, mv_args
    )
    renderer = render_utils.get_renderer(mesh, **options)
    live = mesh.to("cuda")
    paths: Dict[str, str] = {}
    try:
        for extrinsic, intrinsic, label in zip(extrinsics, intrinsics, labels):
            rendered = renderer.render(
                live,
                extrinsic,
                intrinsic,
                envmap=envmap,
                use_envmap_bg=bool(args.use_envmap_bg),
            )
            if "shaded" not in rendered:
                raise RuntimeError(
                    f"official renderer did not return shaded for {label}: "
                    f"{sorted(rendered.keys())}"
                )
            path = output_dir / f"{label}.png"
            _save_frame(path, rendered["shaded"])
            paths[label] = str(path)
            del rendered
            _empty_cuda_cache()
    finally:
        del live, renderer
        _empty_cuda_cache()
    return paths


def _save_variant_sheet(
    *,
    path: Path,
    reference_path: Path,
    aligned_paths: Mapping[str, Path],
    multiview_paths: Mapping[str, Mapping[str, Path]],
    variants: Sequence[str],
    panel: int,
) -> None:
    columns = ["input_canonical", "aligned_front", *VIEW_LABELS]
    row_header = 128
    header = 44
    cell_h = panel + 28
    sheet = Image.new(
        "RGB",
        (row_header + len(columns) * panel, header + len(variants) * cell_h),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(variants):
        y = header + row_index * cell_h
        draw.text((7, y + panel // 2 - 8), variant, fill=(255, 255, 255))
        paths = [reference_path, aligned_paths[variant]] + [
            multiview_paths[variant][label] for label in VIEW_LABELS
        ]
        for col, image_path in enumerate(paths):
            with Image.open(image_path) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            sheet.paste(
                image,
                (x + (panel - image.width) // 2, y + (panel - image.height) // 2),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_front_overview(
    *,
    path: Path,
    reference_path: Path,
    aligned_paths: Mapping[str, Path],
    variants: Sequence[str],
    panel: int,
) -> None:
    columns = ["input_canonical", *variants]
    header = 44
    row_header = 128
    sheet = Image.new("RGB", (row_header + len(columns) * panel, header + panel), "black")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    paths = [reference_path] + [aligned_paths[variant] for variant in variants]
    for col, image_path in enumerate(paths):
        with Image.open(image_path) as source:
            image = ImageOps.contain(source.convert("RGB"), (panel, panel))
        x = row_header + col * panel
        sheet.paste(image, (x + (panel - image.width) // 2, header + (panel - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_multiview_sheet(
    *,
    path: Path,
    multiview_paths: Mapping[str, Mapping[str, Path]],
    variants: Sequence[str],
    panel: int,
) -> None:
    header = 44
    row_header = 128
    sheet = Image.new(
        "RGB",
        (row_header + len(VIEW_LABELS) * panel, header + len(variants) * (panel + 28)),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(VIEW_LABELS):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(variants):
        y = header + row_index * (panel + 28)
        draw.text((7, y + panel // 2 - 8), variant, fill=(255, 255, 255))
        for col, label in enumerate(VIEW_LABELS):
            with Image.open(multiview_paths[variant][label]) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            sheet.paste(image, (x + (panel - image.width) // 2, y + (panel - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official Pixal3D rendering")
    torch.cuda.set_device(int(args.cuda_device))
    source_dir = Path(args.experiment_dir).expanduser().resolve()
    merged_root = source_dir / "global_stitched_quality"
    output_root = source_dir / "global_stitched_quality_4096"
    reference_path = source_dir / "canonical_4096.png"
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    variants = _parse_variants(args.variants)
    output_root.mkdir(parents=True, exist_ok=True)
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    render_args = _render_args(args)
    aligned_paths: Dict[str, Path] = {}
    multiview_paths: Dict[str, Dict[str, Path]] = {}
    records: Dict[str, Any] = {}

    for variant in variants:
        variant_root = output_root / variant
        aligned_dir = variant_root / "aligned_view"
        multiview_dir = variant_root / "multiview"
        checkpoint = merged_root / variant / "global_merged_mesh.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metrics_path = aligned_dir / "metrics.json"
        expected_view_paths = [multiview_dir / f"{label}.png" for label in VIEW_LABELS]
        if bool(args.resume) and metrics_path.is_file() and all(path.is_file() for path in expected_view_paths):
            print(f"[4096 {variant}] reused rendered outputs")
            aligned_paths[variant] = aligned_dir / "render.png"
            multiview_paths[variant] = {
                label: multiview_dir / f"{label}.png" for label in VIEW_LABELS
            }
            records[variant] = json.loads((variant_root / "render_4096_summary.json").read_text(encoding="utf-8"))
            continue

        started = time.perf_counter()
        print(f"[4096 {variant}] loading {checkpoint}")
        mesh = _load_mesh(checkpoint)
        print(
            f"[4096 {variant}] aligned render vertices={mesh.vertices.shape[0]:,} "
            f"faces={mesh.faces.shape[0]:,}"
        )
        render_result = core._render(
            mesh,
            output_dir=aligned_dir,
            camera=global_camera,
            reference_image=reference_path,
            args=render_args,
            envmap=envmap,
        )
        aligned_path = Path(str(render_result["render_png"]))
        print(f"[4096 {variant}] six-view render")
        view_paths = _render_multiview_4096(
            mesh,
            global_camera=global_camera,
            envmap=envmap,
            args=args,
            output_dir=multiview_dir,
        )
        aligned_paths[variant] = aligned_path
        multiview_paths[variant] = {label: Path(path) for label, path in view_paths.items()}
        record = {
            "variant": variant,
            "checkpoint": str(checkpoint),
            "aligned_render": str(aligned_path),
            "aligned_metrics": str(aligned_dir / "metrics.json"),
            "multiview": view_paths,
            "mesh_vertices": int(mesh.vertices.shape[0]),
            "mesh_faces": int(mesh.faces.shape[0]),
            "render_resolution": int(args.resolution),
            "metric_resolution": int(args.metric_resolution),
            "ssaa": int(args.ssaa),
            "peel_layers": int(args.peel_layers),
            "face_chunk_size": int(args.face_chunk_size),
            "renderer": "official core._render/render_utils -> PbrMeshRenderer -> nvdiffrast",
            "camera": global_camera,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(variant_root / "render_4096_summary.json", record)
        records[variant] = record
        del mesh
        _empty_cuda_cache()

    comparison_sheet = output_root / "global_aligned_view_and_multiview_sheet.png"
    _save_variant_sheet(
        path=comparison_sheet,
        reference_path=reference_path,
        aligned_paths=aligned_paths,
        multiview_paths=multiview_paths,
        variants=variants,
        panel=int(args.sheet_panel),
    )
    front_overview = output_root / "global_aligned_front_overview.png"
    _save_front_overview(
        path=front_overview,
        reference_path=reference_path,
        aligned_paths=aligned_paths,
        variants=variants,
        panel=int(args.overview_panel),
    )
    multiview_sheet = output_root / "global_multiview_sheet.png"
    _save_multiview_sheet(
        path=multiview_sheet,
        multiview_paths=multiview_paths,
        variants=variants,
        panel=int(args.sheet_panel),
    )
    output = {
        "format": "pixal3d_texture_pbr_global_render_4096_v1",
        "source_experiment": str(source_dir),
        "source_merged_root": str(merged_root),
        "cuda_device": int(args.cuda_device),
        "reference_view": str(reference_path),
        "camera": global_camera,
        "variants": variants,
        "views": list(VIEW_LABELS),
        "render_resolution": int(args.resolution),
        "metric_resolution": int(args.metric_resolution),
        "ssaa": int(args.ssaa),
        "peel_layers": int(args.peel_layers),
        "face_chunk_size": int(args.face_chunk_size),
        "lpips": "skipped" if bool(args.skip_lpips) else "computed",
        "records": records,
        "comparison_sheet": str(comparison_sheet),
        "front_overview": str(front_overview),
        "multiview_sheet": str(multiview_sheet),
    }
    _write_json(output_root / "global_render_4096_summary.json", output)
    print(f"[done] 4096 render output={output_root}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="outputs/codex_texture_pbr_degradation_cuda4_all_tiles",
    )
    parser.add_argument("--model-path", default=MODEL_PATH, help="kept for CLI compatibility")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--variants", default=None, help="comma-separated subset; default all nine")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=4096)
    parser.add_argument("--ssaa", type=int, default=1)
    parser.add_argument("--peel-layers", type=int, default=4)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sheet-panel", type=int, default=512)
    parser.add_argument("--overview-panel", type=int, default=512)
    return parser


if __name__ == "__main__":
    run(_build_parser().parse_args())
