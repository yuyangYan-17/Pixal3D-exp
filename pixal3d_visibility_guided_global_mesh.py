#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Return the CUDA-4 visibility-guided tile meshes to global space.

The visibility-guided experiment intentionally stores sparse texture
endpoints instead of decoded meshes.  This post-process decodes those saved
endpoints with the official Pixal3D decoder, maps the local decoded geometry
through the exact local->global camera transform, removes projected tile
overlap, welds the global vertices, and renders the complete object.

The aligned reference is ``canonical_1024.png``: it is the image and camera
used by the Pixal3D global baseline.  The original input and canonical image
are both retained in the experiment directory for visual comparison.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
from PIL import Image, ImageDraw, ImageFont, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_texture_pbr_degradation_experiment as experiment
import pixal3d_texture_pbr_global_stitch as stitch
import pixal3d_texture_pbr_multiview as multiview
import pixal3d_texture_visibility_guided_pbr_flow as visibility_flow
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr
from pixal3d.utils import render_utils


VARIANT_ORDER = (
    "G",
    "pure_HR",
    "endpoint_pbr_fusion",
    "endpoint_reencode",
    "perstep_guided",
)
ENDPOINT_FILES = {
    "G": "G_endpoint.pt",
    "pure_HR": "pure_HR_endpoint.pt",
    "endpoint_reencode": "endpoint_reencode_endpoint.pt",
    "perstep_guided": "guided_endpoint.pt",
}
VIEW_LABELS = tuple(label for label, _, _ in multiview.FIXED_VIEWS)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _parse_variants(value: Optional[str]) -> List[str]:
    if value is None or not value.strip():
        return list(VARIANT_ORDER)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in requested if name not in VARIANT_ORDER]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choices={VARIANT_ORDER}")
    return requested


def _lift_support_weights_to_vertices(
    vertices: torch.Tensor,
    local_coords: torch.Tensor,
    weights: torch.Tensor,
    resolution: int = 1024,
) -> torch.Tensor:
    """Vectorized local C1024 support-to-vertex lookup for direct fusion."""
    coords = local_coords.detach().cpu().to(torch.long)
    values = weights.detach().cpu().to(torch.float32)
    cells = (
        torch.floor((vertices.detach().cpu().to(torch.float32) + 0.5) * int(resolution))
        .to(torch.long)
        .clamp(0, int(resolution) - 1)
    )
    support_keys = (coords[:, 0] * int(resolution) + coords[:, 1]) * int(resolution) + coords[:, 2]
    cell_keys = (cells[:, 0] * int(resolution) + cells[:, 1]) * int(resolution) + cells[:, 2]
    sorted_keys, order = torch.sort(support_keys)
    positions = torch.searchsorted(sorted_keys, cell_keys)
    valid = positions < sorted_keys.shape[0]
    safe_positions = positions.clamp_max(max(0, sorted_keys.shape[0] - 1))
    valid &= sorted_keys.index_select(0, safe_positions) == cell_keys
    output = torch.zeros((vertices.shape[0],), dtype=torch.float32)
    if bool(valid.any().item()):
        output[valid] = values.index_select(0, order.index_select(0, safe_positions[valid]))
    return output


def _render_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        render_resolution=int(args.aligned_resolution),
        metric_resolution=int(args.metric_resolution),
        render_ssaa=int(args.aligned_ssaa),
        render_peel_layers=int(args.peel_layers),
        render_face_chunk_size=int(args.face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        envmap=str(args.envmap),
        lpips_net=str(args.lpips_net),
        skip_lpips=bool(args.skip_lpips),
    )


def _decode_tile_payload(
    *,
    pipeline: Any,
    source_dir: Path,
    tile_id: int,
    row: Mapping[str, Any],
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    variants: Sequence[str],
) -> Dict[str, Any]:
    tile_dir = source_dir / "tiles" / f"tile_{int(tile_id):02d}"
    fixed = _load_torch(tile_dir / "fixed_shape.pt")
    transform = core.TileCameraTransform(
        **json.loads((tile_dir / "tile_camera.json").read_text(encoding="utf-8"))
    )
    device = torch.device("cuda")
    shape_coords = fixed["coords"].to(device=device, dtype=torch.int32)
    shape_norm = SparseTensor(
        fixed["G_shape_norm"].to(device=device, dtype=torch.float32),
        shape_coords,
    )
    shape_denorm = experiment._denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
    texture_coords = fixed["coords"].to(device=device, dtype=torch.int32)

    decoded: Dict[str, Any] = {}
    decode_stats: Dict[str, Any] = {}
    required_endpoints = set(variants) & {"G", "pure_HR", "endpoint_reencode", "perstep_guided"}
    if "endpoint_pbr_fusion" in variants:
        required_endpoints.update(("G", "pure_HR"))
    endpoint_names = [
        name
        for name in ("G", "pure_HR", "endpoint_reencode", "perstep_guided")
        if name in required_endpoints
    ]
    # endpoint_pbr_fusion is a vertex-space diagnostic built from G and HR.
    for name in endpoint_names:
        path = tile_dir / ENDPOINT_FILES[name]
        payload = _load_torch(path)
        coords = payload["coords"].to(device=device, dtype=torch.int32)
        if not torch.equal(coords, texture_coords):
            raise RuntimeError(f"tile {tile_id}: {name} endpoint support differs from fixed shape")
        vertices, faces, attrs, stats = multiview._decode_variant(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_norm=payload["norm"].to(torch.float32),
            coords=coords,
            label=f"tile_{int(tile_id):02d} {name} global mesh",
            query_chunk_size=int(args.query_chunk_size),
        )
        decoded[name] = (vertices, faces, attrs)
        decode_stats[name] = stats

    if not decoded:
        raise RuntimeError(f"tile {tile_id}: no requested endpoint decoded")
    reference_name = "G" if "G" in decoded else endpoint_names[0]
    reference_vertices, reference_faces, reference_attrs = decoded[reference_name]
    g_attrs = reference_attrs
    for name, (vertices, faces, _) in decoded.items():
        if vertices.shape != reference_vertices.shape or not torch.equal(vertices, reference_vertices):
            raise RuntimeError(f"tile {tile_id}: decoded geometry changed for {name}")
        if not torch.equal(faces, reference_faces):
            raise RuntimeError(f"tile {tile_id}: decoded topology changed for {name}")

    attrs: Dict[str, torch.Tensor] = {}
    if "G" in variants or "endpoint_pbr_fusion" in variants:
        attrs["G"] = g_attrs
    if "pure_HR" in variants or "endpoint_pbr_fusion" in variants:
        attrs["pure_HR"] = decoded["pure_HR"][2]
    for name in ("endpoint_reencode", "perstep_guided"):
        if name in variants:
            attrs[name] = decoded[name][2]
    if "endpoint_pbr_fusion" in variants:
        guidance = _load_torch(tile_dir / "guidance_weights.pt")
        guidance_geometry = _load_torch(tile_dir / "guidance_geometry.pt")
        # guidance_weights.w_final is defined on the local C1024 material
        # support. fixed_shape.pt contains the smaller fixed C64 SLat support,
        # which has different coordinates and a different length. Using the
        # C64 coords here silently selected the wrong entries and made every
        # lifted vertex weight zero, so endpoint_pbr_fusion became exactly G
        # after global stitching.
        local_c1024_coords = guidance_geometry.get("local_c1024_coords")
        final_weights = guidance.get("w_final")
        if local_c1024_coords is None or final_weights is None:
            raise RuntimeError(f"tile {tile_id}: C1024 guidance support is incomplete")
        local_c1024_coords = local_c1024_coords.to(torch.int32)
        final_weights = final_weights.to(torch.float32)
        if local_c1024_coords.ndim != 2 or local_c1024_coords.shape[1] != 3:
            raise RuntimeError(
                f"tile {tile_id}: invalid local C1024 coords shape {tuple(local_c1024_coords.shape)}"
            )
        if final_weights.ndim != 1 or final_weights.shape[0] != local_c1024_coords.shape[0]:
            raise RuntimeError(
                f"tile {tile_id}: C1024 coords/weights mismatch: "
                f"coords={tuple(local_c1024_coords.shape)} weights={tuple(final_weights.shape)}"
            )
        if not bool(((final_weights == 0.0) | (final_weights == 1.0)).all().item()):
            raise RuntimeError(f"tile {tile_id}: endpoint fusion weights are not binary")
        vertex_weights = _lift_support_weights_to_vertices(
            reference_vertices,
            local_c1024_coords,
            final_weights,
        )
        attrs["endpoint_pbr_fusion"] = g_attrs + vertex_weights[:, None] * (
            decoded["pure_HR"][2] - g_attrs
        )

    global_vertices = experiment._map_local_to_global_chunked(
        reference_vertices,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    ).detach().cpu().to(torch.float32)
    q_local = reference_vertices * (2.0 * float(transform.mesh_scale))
    q_global = global_vertices * (2.0 * float(global_camera["mesh_scale"]))
    q_roundtrip, _ = core._global_q_to_local_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
    )
    roundtrip = (q_roundtrip.cpu() - q_local.cpu()).abs()
    if float(roundtrip.max().item()) > 1e-4:
        raise RuntimeError(
            f"tile {tile_id}: local/global roundtrip too large: "
            f"{float(roundtrip.max().item()):.3e}"
        )

    base_patch = core.ReturnedTilePatch(
        tile_id=int(tile_id),
        box=tuple(int(value) for value in row["box"]),
        vertices=global_vertices,
        faces=reference_faces.to(torch.int32),
        vertex_attrs=attrs[variants[0]].detach().cpu().to(torch.float32),
        stats={
            "tile_id": int(tile_id),
            "local_vertices": int(reference_vertices.shape[0]),
            "local_faces": int(reference_faces.shape[0]),
            "global_vertices": int(global_vertices.shape[0]),
            "global_faces": int(reference_faces.shape[0]),
            "local_to_global_roundtrip_max_abs_q": float(roundtrip.max().item()),
            "local_to_global_roundtrip_mean_abs_q": float(roundtrip.mean().item()),
            "coordinate_space": "global normalized object space",
            "transform": "official core._local_q_to_global_q",
        },
    )
    cache_variants = list(variants)
    if "endpoint_pbr_fusion" in variants and "pure_HR" not in cache_variants:
        # The aligned-input visibility promotion needs the stitched pure-HR
        # field even when the caller requested only endpoint_pbr_fusion.
        cache_variants.append("pure_HR")
    patches: Dict[str, core.ReturnedTilePatch] = {}
    for name in cache_variants:
        patches[name] = core.ReturnedTilePatch(
            tile_id=int(tile_id),
            box=base_patch.box,
            vertices=global_vertices,
            faces=base_patch.faces,
            vertex_attrs=attrs[name].detach().cpu().to(torch.float32),
            stats=dict(base_patch.stats),
        )

    payload = {
        "format": "pixal3d_visibility_guided_global_tile_cache_v1",
        "tile_id": int(tile_id),
        "box": list(base_patch.box),
        "global_vertices": global_vertices,
        "faces": base_patch.faces,
        "attrs": {name: patches[name].vertex_attrs for name in cache_variants},
        "decode_stats": decode_stats,
        "transform": transform.__dict__,
        "roundtrip": {
            "max_abs_q": float(roundtrip.max().item()),
            "mean_abs_q": float(roundtrip.mean().item()),
        },
    }
    del fixed, shape_norm, shape_denorm, texture_coords, decoded, attrs
    _empty_cuda_cache()
    return payload


def _patches_from_payloads(
    payloads: Mapping[int, Mapping[str, Any]],
    variant: str,
) -> List[core.ReturnedTilePatch]:
    return [
        core.ReturnedTilePatch(
            tile_id=int(tile_id),
            box=tuple(int(value) for value in payloads[tile_id]["box"]),
            vertices=payloads[tile_id]["global_vertices"].to(torch.float32),
            faces=payloads[tile_id]["faces"].to(torch.int32),
            vertex_attrs=payloads[tile_id]["attrs"][variant].to(torch.float32),
            stats={"source": "visibility-guided local mesh returned to global space"},
        )
        for tile_id in sorted(payloads)
    ]


def _input_visible_vertex_mask(
    geometry: Mapping[str, Any],
    *,
    global_camera: Mapping[str, float],
    resolution: int,
    face_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return vertices incident to faces visible from the aligned input camera."""
    if resolution <= 0:
        raise ValueError("input visibility resolution must be positive")
    mesh = SimpleNamespace(
        vertices=geometry["welded_vertices"].to(torch.float32),
        faces=geometry["welded_faces"].to(torch.int32),
    )
    buffers = visibility_flow._render_global_visibility_buffers(
        mesh,
        global_camera=global_camera,
        resolution=int(resolution),
        face_chunk_size=int(face_chunk_size),
        device=device,
    )
    triangle_id = buffers["triangle_id"].to(torch.long)
    visible_faces = torch.unique(triangle_id[triangle_id >= 0])
    mask = torch.zeros((mesh.vertices.shape[0],), dtype=torch.bool)
    if bool(visible_faces.numel()):
        mask[mesh.faces.index_select(0, visible_faces).reshape(-1).to(torch.long)] = True
    print(
        f"[global visibility] aligned visible pixels={int((triangle_id >= 0).sum().item())} "
        f"faces={int(visible_faces.numel())} vertices={int(mask.sum().item())}"
    )
    del mesh, buffers, triangle_id, visible_faces
    _empty_cuda_cache()
    return mask


def _save_image(path: Path, image: np.ndarray | Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path)
    else:
        Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB").save(path)


def _render_variant(
    *,
    variant: str,
    mesh: MeshWithVertexPbr,
    output_root: Path,
    reference_path: Path,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    variant_root = output_root / variant
    aligned_dir = variant_root / "input_view"
    aligned = core._render(
        mesh,
        output_dir=aligned_dir,
        camera=global_camera,
        reference_image=reference_path,
        args=_render_args(args),
        envmap=envmap,
    )

    mv_args = SimpleNamespace(
        resolution=int(args.multiview_resolution),
        ssaa=int(args.multiview_ssaa),
        peel_layers=int(args.multiview_peel_layers),
        face_chunk_size=int(args.face_chunk_size),
        radius_scale=float(args.radius_scale),
        use_envmap_bg=bool(args.use_envmap_bg),
    )
    extrinsics, intrinsics, labels, options = multiview._fixed_cameras(global_camera, mv_args)
    renderer = render_utils.get_renderer(mesh, **options)
    frames = multiview._render_variant_frames(
        mesh,
        renderer=renderer,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        options=options,
        envmap=envmap,
        args=mv_args,
    )
    multiview_dir = variant_root / "multiview"
    view_paths: Dict[str, str] = {}
    for frame, label in zip(frames, labels):
        path = multiview_dir / f"{label}.png"
        _save_image(path, frame)
        view_paths[label] = str(path)
    del renderer, frames
    _empty_cuda_cache()
    return {
        "variant": variant,
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "input_view": str(aligned["render_png"]),
        "input_view_metrics": core._metric_subset(aligned),
        "input_view_metrics_json": str(aligned_dir / "metrics.json"),
        "multiview": view_paths,
        "renderer": "official core._render + pixal3d.utils.render_utils.PbrMeshRenderer",
        "camera": dict(global_camera),
    }


def _save_input_view_sheet(
    *,
    path: Path,
    reference_path: Path,
    records: Mapping[str, Mapping[str, Any]],
    panel: int,
) -> None:
    columns = ["canonical_1024", *records.keys()]
    header = 46
    row_header = 136
    canvas = Image.new("RGB", (row_header + len(columns) * panel, header + panel), "black")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(columns):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    paths = [reference_path] + [Path(records[name]["input_view"]) for name in records]
    for col, image_path in enumerate(paths):
        with Image.open(image_path) as source:
            image = ImageOps.contain(source.convert("RGB"), (panel, panel))
        x = row_header + col * panel
        canvas.paste(image, (x + (panel - image.width) // 2, header + (panel - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_multiview_sheet(
    *,
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    panel: int,
) -> None:
    header = 46
    row_header = 136
    canvas = Image.new(
        "RGB",
        (row_header + len(VIEW_LABELS) * panel, header + len(records) * (panel + 28)),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(VIEW_LABELS):
        draw.text((row_header + col * panel + 5, 14), label, fill=(255, 255, 255))
    for row_index, (variant, record) in enumerate(records.items()):
        y = header + row_index * (panel + 28)
        draw.text((7, y + panel // 2 - 8), variant, fill=(255, 255, 255))
        for col, label in enumerate(VIEW_LABELS):
            with Image.open(record["multiview"][label]) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = row_header + col * panel
            canvas.paste(image, (x + (panel - image.width) // 2, y + (panel - image.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _load_annotation_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font when available, with a PIL fallback."""
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=int(size))
    return ImageFont.load_default()


def _format_metric(value: Any, *, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _save_annotated_input_view(
    *,
    path: Path,
    source_path: Path,
    variant: str,
    metrics: Mapping[str, Any],
) -> None:
    """Save one variant's input-view render with its measured metrics."""
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    banner = 104
    canvas = Image.new("RGB", (image.width, image.height + banner), "black")
    canvas.paste(image, (0, banner))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_annotation_font(28, bold=True)
    metric_font = _load_annotation_font(24)
    title = f"{variant} | input view"
    metric_line = (
        f"PSNR {_format_metric(metrics.get('psnr_db'))} dB    "
        f"SSIM {_format_metric(metrics.get('ssim'))}    "
        f"LPIPS {_format_metric(metrics.get('lpips'))}"
    )
    draw.text((20, 12), title, fill=(255, 255, 255), font=title_font)
    draw.text((20, 57), metric_line, fill=(220, 235, 255), font=metric_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_variant_multiview_sheet(
    *,
    path: Path,
    variant: str,
    record: Mapping[str, Any],
    panel: int,
) -> None:
    """Arrange one variant's fixed views into a standalone 3-by-2 sheet."""
    columns = 3
    rows = int(math.ceil(len(VIEW_LABELS) / columns))
    title_height = 52
    label_height = 34
    canvas = Image.new(
        "RGB",
        (columns * panel, title_height + rows * (panel + label_height)),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_annotation_font(28, bold=True)
    label_font = _load_annotation_font(22, bold=True)
    draw.text((16, 12), f"{variant} | multiview", fill=(255, 255, 255), font=title_font)
    for index, label in enumerate(VIEW_LABELS):
        row = index // columns
        col = index % columns
        x = col * panel
        y = title_height + row * (panel + label_height)
        draw.text((x + 10, y + 5), label, fill=(220, 235, 255), font=label_font)
        source_path = Path(record["multiview"][label])
        with Image.open(source_path) as source:
            image = ImageOps.contain(source.convert("RGB"), (panel, panel))
        image_x = x + (panel - image.width) // 2
        image_y = y + label_height + (panel - image.height) // 2
        canvas.paste(image, (image_x, image_y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _compose_variant_visuals(
    *,
    output_root: Path,
    variants: Sequence[str],
    panel: int,
) -> Dict[str, Any]:
    """Add per-variant annotated input and standalone multiview images."""
    summary_path = output_root / "global_mesh_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = summary.get("variants", {})
    for variant in variants:
        if variant not in records:
            raise FileNotFoundError(f"variant record not found in {summary_path}: {variant}")
        record = records[variant]
        input_view = Path(record["input_view"])
        multiview_paths = record.get("multiview", {})
        if not input_view.is_file():
            raise FileNotFoundError(input_view)
        missing_views = [label for label in VIEW_LABELS if not Path(multiview_paths[label]).is_file()]
        if missing_views:
            raise FileNotFoundError(f"{variant}: missing multiview images {missing_views}")
        variant_root = output_root / variant
        annotated_path = variant_root / "input_view" / "render_with_metrics.png"
        sheet_path = variant_root / "multiview_comparison.png"
        _save_annotated_input_view(
            path=annotated_path,
            source_path=input_view,
            variant=variant,
            metrics=record.get("input_view_metrics", {}),
        )
        _save_variant_multiview_sheet(
            path=sheet_path,
            variant=variant,
            record=record,
            panel=int(panel),
        )
        record["input_view_annotated"] = str(annotated_path)
        record["multiview_sheet"] = str(sheet_path)
        _write_json(variant_root / "global_variant_summary.json", record)

    _write_json(summary_path, summary)
    _write_report(output_root / "global_mesh_report.md", summary)
    print(f"[done] per-variant visual sheets output={output_root}")
    return summary


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Visibility-guided global mesh and input-view evaluation",
        "",
        f"- CUDA device: `{summary['cuda_device']}`",
        f"- successful tiles: `{summary['successful_tiles']}`",
        f"- skipped tiles: `{summary['skipped_tiles']}`",
        f"- failed tiles: `{summary['failed_tiles']}`",
        f"- reference: `{summary['reference_view']}`",
        "",
        "## Global mesh / camera transform",
        "",
        "All decoded local vertices were returned through the official local-to-global camera transform. Tile overlap was assigned by projected nearest tile center, followed by spatial welding; no remeshing or texture flow was run in this post-process.",
        "",
        f"- stitch tolerance: `{summary['stitch_tolerance']}`",
        f"- coordinate transform: `{summary['coordinate_transform']}`",
        "",
        "## Input-view metrics",
        "",
        "| variant | vertices | faces | PSNR (dB) | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, record in summary["variants"].items():
        metric = record.get("input_view_metrics", {})
        lines.append(
            f"| {variant} | {record['mesh_vertices']:,} | {record['mesh_faces']:,} | "
            f"{metric.get('psnr_db')} | {metric.get('ssim')} | {metric.get('lpips')} |"
        )
    lines.extend(
        [
            "",
            "The metrics are computed against the aligned `canonical_1024.png` input view using the official Pixal3D renderer. `input_original.png` remains available for the original-resolution visual comparison; it is not substituted for the canonical camera reference.",
            "",
            "## Per-variant visual outputs",
            "",
        ]
    )
    for variant, record in summary["variants"].items():
        if record.get("input_view_annotated") or record.get("multiview_sheet"):
            lines.append(
                f"- `{variant}` annotated input: `{record.get('input_view_annotated')}`; "
                f"standalone multiview: `{record.get('multiview_sheet')}`"
            )
    lines.append("")
    override = summary.get("variants", {}).get("endpoint_pbr_fusion", {}).get(
        "endpoint_fusion_input_visibility_override"
    )
    if override and override.get("enabled"):
        lines.extend(
            [
                "## Endpoint fusion aligned-view consistency",
                "",
                f"- input-camera visible vertex promotion: `{override.get('visible_vertex_count')}/{override.get('total_vertex_count')}` ({override.get('visible_vertex_fraction')})",
                f"- rule: `{override.get('rule')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    source_dir = Path(args.experiment_dir).expanduser().resolve()
    output_root = source_dir / "global_mesh_quality"
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(args.compose_only):
        variants = _parse_variants(args.variants)
        if int(args.variant_sheet_panel) <= 0:
            raise ValueError("variant sheet panel must be positive")
        return _compose_variant_visuals(
            output_root=output_root,
            variants=variants,
            panel=int(args.variant_sheet_panel),
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official Pixal3D decoding and rendering")
    torch.cuda.set_device(int(args.cuda_device))
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    reference_path = source_dir / "canonical_1024.png"
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    variants = _parse_variants(args.variants)
    selected = _parse_ids(args.tile_ids)
    rows = {
        int(row["tile_id"]): row
        for row in summary.get("tiles", [])
        if row.get("status") == "success"
        and (selected is None or int(row["tile_id"]) in selected)
    }
    if not rows:
        raise RuntimeError("no successful tiles selected")

    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    cache_root = output_root / "decoded_global_tiles"
    payloads: Dict[int, Dict[str, Any]] = {}
    tile_records: List[Dict[str, Any]] = []
    for tile_id in sorted(rows):
        cache_path = cache_root / f"tile_{tile_id:02d}.pt"
        started = time.perf_counter()
        try:
            payload = None
            if bool(args.resume) and cache_path.is_file():
                candidate = _load_torch(cache_path)
                cache_required_variants = set(variants)
                if "endpoint_pbr_fusion" in variants:
                    cache_required_variants.add("pure_HR")
                if (
                    candidate.get("format") == "pixal3d_visibility_guided_global_tile_cache_v1"
                    and all(name in candidate.get("attrs", {}) for name in cache_required_variants)
                ):
                    payload = candidate
                    print(f"[global tile {tile_id:02d}] reused decoded global cache")
            if payload is None:
                print(f"[global tile {tile_id:02d}] decode local endpoints and return mesh to global")
                payload = _decode_tile_payload(
                    pipeline=pipeline,
                    source_dir=source_dir,
                    tile_id=tile_id,
                    row=rows[tile_id],
                    global_camera=global_camera,
                    args=args,
                    variants=variants,
                )
                _atomic_torch_save(cache_path, payload)
            payloads[tile_id] = payload
            tile_records.append(
                {
                    "tile_id": int(tile_id),
                    "status": "success",
                    "cache": str(cache_path),
                    "global_vertices": int(payload["global_vertices"].shape[0]),
                    "global_faces": int(payload["faces"].shape[0]),
                    "roundtrip": payload.get("roundtrip"),
                    "seconds": float(time.perf_counter() - started),
                }
            )
        except Exception as exc:
            tile_records.append(
                {
                    "tile_id": int(tile_id),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "seconds": float(time.perf_counter() - started),
                }
            )
            print(f"[global tile {tile_id:02d}] FAILED: {type(exc).__name__}: {exc}")
        finally:
            _empty_cuda_cache()
    if not payloads:
        raise RuntimeError("all selected tiles failed before global stitching")

    # The pipeline is no longer needed after endpoint decode; releasing it
    # leaves the A800 memory headroom for the global renderer.
    del pipeline
    _empty_cuda_cache()

    geometry_cache = output_root / "global_stitch_geometry.pt"
    shared_geometry = None
    if bool(args.resume) and geometry_cache.is_file():
        candidate_geometry = _load_torch(geometry_cache)
        if candidate_geometry.get("format") == "pixal3d_visibility_guided_global_stitch_geometry_v1":
            shared_geometry = candidate_geometry["geometry"]
            print("[global stitch] reused shared projected ownership and weld map")
    if shared_geometry is None:
        print("[global stitch] computing shared projected ownership and weld map")
        shared_geometry = stitch._stitch_global_geometry(
            _patches_from_payloads(payloads, variants[0]),
            global_camera=global_camera,
            face_chunk_size=int(args.stitch_face_chunk_size),
            weld_tolerance=float(args.stitch_tolerance),
        )
        _atomic_torch_save(
            geometry_cache,
            {
                "format": "pixal3d_visibility_guided_global_stitch_geometry_v1",
                "geometry": shared_geometry,
                "source_tiles": sorted(payloads),
                "stitch_tolerance": float(args.stitch_tolerance),
            },
        )
    envmap = core.load_envmap(str(args.envmap), device="cuda") if bool(args.render) else None
    input_visible_vertex_mask = None
    if "endpoint_pbr_fusion" in variants:
        input_visible_vertex_mask = _input_visible_vertex_mask(
            shared_geometry,
            global_camera=global_camera,
            resolution=int(args.aligned_resolution),
            face_chunk_size=int(args.face_chunk_size),
            device=torch.device("cuda"),
        )
    variant_records: Dict[str, Any] = {}
    aligned_paths: Dict[str, Path] = {}

    for variant in variants:
        print(f"[global stitch] {variant}: applying stitched PBR attributes")
        merged_attrs = stitch._apply_stitched_attrs(shared_geometry, payloads, variant)
        endpoint_visibility_override: Dict[str, Any] = {"enabled": False}
        if variant == "endpoint_pbr_fusion" and input_visible_vertex_mask is not None:
            pure_attrs = stitch._apply_stitched_attrs(shared_geometry, payloads, "pure_HR")
            merged_attrs = torch.where(
                input_visible_vertex_mask[:, None],
                pure_attrs,
                merged_attrs,
            )
            endpoint_visibility_override = {
                "enabled": True,
                "rule": "vertices incident to faces visible in the aligned input-camera z-buffer use pure_HR; other vertices retain C1024 visibility-guided fusion",
                "visible_vertex_count": int(input_visible_vertex_mask.sum().item()),
                "total_vertex_count": int(input_visible_vertex_mask.numel()),
                "visible_vertex_fraction": float(input_visible_vertex_mask.to(torch.float32).mean().item()),
            }
            del pure_attrs
        merged = MeshWithVertexPbr(
            vertices=shared_geometry["welded_vertices"],
            faces=shared_geometry["welded_faces"],
            vertex_attrs=merged_attrs,
            layout=dict(experiment.PBR_LAYOUT),
        )
        variant_root = output_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        mesh_path = variant_root / "global_merged_mesh.pt"
        _atomic_torch_save(
            mesh_path,
            {
                "format": "pixal3d_visibility_guided_global_merged_mesh_v1",
                "variant": variant,
                "mesh": merged,
                "stitch_stats": dict(shared_geometry["stats"]),
                "global_camera": global_camera,
                "source_tiles": sorted(payloads),
            },
        )
        glb_info: Dict[str, Any] = {"status": "skipped"}
        if bool(args.save_glb):
            glb_patch = core.ReturnedTilePatch(
                tile_id=-1,
                box=(0, 0, 4096, 4096),
                vertices=merged.vertices,
                faces=merged.faces,
                vertex_attrs=merged.vertex_attrs,
                stats={"source": "global welded mesh"},
            )
            glb_info = core._export_tiled_glb([glb_patch], variant_root / "global_merged_mesh.glb")

        record: Dict[str, Any] = {
            "variant": variant,
            "mesh": str(mesh_path),
            "glb": glb_info,
            "mesh_vertices": int(merged.vertices.shape[0]),
            "mesh_faces": int(merged.faces.shape[0]),
            "stitch": dict(shared_geometry["stats"]),
            "endpoint_fusion_input_visibility_override": endpoint_visibility_override,
        }
        if bool(args.render):
            rendered = _render_variant(
                variant=variant,
                mesh=merged,
                output_root=output_root,
                reference_path=reference_path,
                global_camera=global_camera,
                args=args,
                envmap=envmap,
            )
            record.update(rendered)
            aligned_paths[variant] = Path(rendered["input_view"])
        _write_json(variant_root / "global_variant_summary.json", record)
        variant_records[variant] = record
        del merged_attrs, merged
        _empty_cuda_cache()

    del shared_geometry, input_visible_vertex_mask
    if envmap is not None:
        del envmap
    _empty_cuda_cache()

    input_sheet = output_root / "input_view_comparison.png"
    multiview_sheet = output_root / "multiview_comparison.png"
    if bool(args.render):
        _save_input_view_sheet(
            path=input_sheet,
            reference_path=reference_path,
            records=variant_records,
            panel=int(args.sheet_panel),
        )
        _save_multiview_sheet(
            path=multiview_sheet,
            records=variant_records,
            panel=int(args.sheet_panel),
        )

    output = {
        "format": "pixal3d_visibility_guided_global_mesh_v1",
        "source_experiment": str(source_dir),
        "cuda_device": int(args.cuda_device),
        "reference_view": str(reference_path),
        "original_input": str(source_dir / "input_original.png"),
        "canonical_4096": str(source_dir / "canonical_4096.png"),
        "coordinate_transform": "official core._local_q_to_global_q",
        "stitch_policy": "projected nearest-tile owner followed by spatial weld",
        "stitch_tolerance": float(args.stitch_tolerance),
        "successful_tiles": sorted(payloads),
        "skipped_tiles": [int(row["tile_id"]) for row in summary.get("tiles", []) if row.get("status") == "skipped"],
        "failed_tiles": [row["tile_id"] for row in tile_records if row["status"] == "failed"],
        "tile_records": tile_records,
        "variants": variant_records,
        "input_view_comparison": str(input_sheet) if bool(args.render) else None,
        "multiview_comparison": str(multiview_sheet) if bool(args.render) else None,
        "render_resolution": int(args.aligned_resolution),
        "metric_resolution": int(args.metric_resolution),
        "multiview_resolution": int(args.multiview_resolution),
    }
    summary_path = output_root / "global_mesh_summary.json"
    report_path = output_root / "global_mesh_report.md"
    _write_json(summary_path, output)
    _write_report(report_path, output)
    if bool(args.render):
        output = _compose_variant_visuals(
            output_root=output_root,
            variants=variants,
            panel=int(args.variant_sheet_panel),
        )
    print(f"[done] global mesh output={output_root}")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="outputs/visibility_guided_pbr_flow_cuda4")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--variants", default=None, help="comma-separated subset of the five saved variants")
    parser.add_argument(
        "--compose-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compose per-variant annotated input and multiview images from an existing global_mesh_quality run",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-chunk-size", type=int, default=32_768)
    parser.add_argument("--stitch-face-chunk-size", type=int, default=250_000)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / 1024.0)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-glb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aligned-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--aligned-ssaa", type=int, default=1)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=4)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--sheet-panel", type=int, default=256)
    parser.add_argument("--variant-sheet-panel", type=int, default=512)
    return parser


if __name__ == "__main__":
    parsed = _build_parser().parse_args()
    if not parsed.skip_lpips and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips unavailable; continuing without LPIPS")
        parsed.skip_lpips = True
    run(parsed)
