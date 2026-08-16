#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test C: compare native and LS coarse content in C1024 field space.

This is an independent diagnostic following the current ``Codex.md``.  It
reuses the Phase-1 definitions of the active O-voxel supports, sparse
trilinear prolongation ``P``, least-squares restriction ``A``, global field
query, and visibility mask.  It deliberately does not run texture flow,
hidden correction, decoder re-encoding, mesh stitching, or rendering.

The central comparison is made on the same C1024 fine rows::

    C_native = P G256
    C_LS     = P A G1024

The old coefficient-space discrepancy ``A G1024`` versus ``G256`` is kept as
a diagnostic only.  No artificial pass/fail threshold is used.
"""

from __future__ import annotations

import argparse
import math
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor  # noqa: F401  (torch.load compatibility)
from pixal3d.representations import MeshWithVoxel

from pixal3d_sparse_mra_hidden_phase1 import (
    CANONICAL_RESOLUTION,
    CHANNEL_NAMES,
    COARSE_RESOLUTION,
    GLOBAL_RESOLUTION,
    PBR_CHANNELS,
    _apply_operator,
    _build_prolongation,
    _cell_centers,
    _decode_purehr_mesh,
    _empty_cuda_cache,
    _jsonable,
    _local_to_global_normalized,
    _load_torch,
    _load_visibility,
    _make_panel,
    _query_global_at_local_points,
    _sample_visibility,
    _solve_restriction,
    _voxelize_support,
    _write_csv,
    _write_json,
)


FORMAT = "pixal3d_sparse_mra_test_c_v1"
EPS = 1e-8
REGIONS = ("all", "observed", "hidden")
GROUPS = {
    "all_six_joint": slice(0, 6),
    "rgb_joint": slice(0, 3),
    "r": 0,
    "g": 1,
    "b": 2,
    "metallic": 3,
    "roughness": 4,
    "alpha": 5,
}


def _finite_stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())


def _group_tensor(value: torch.Tensor, spec: slice | int) -> torch.Tensor:
    if isinstance(spec, slice):
        return value[:, spec]
    return value[:, int(spec) : int(spec) + 1]


def _scalar_metric(delta: torch.Tensor, reference: torch.Tensor) -> Dict[str, Any]:
    """Return absolute and relative metrics for one channel group."""
    delta64 = delta.detach().cpu().to(torch.float64)
    reference64 = reference.detach().cpu().to(torch.float64)
    if delta64.numel() == 0:
        return {
            "rows": int(delta64.shape[0]),
            "scalar_count": 0,
            "delta_l2": None,
            "reference_l2": None,
            "relative_l2": None,
            "mean_abs": None,
            "max_abs": None,
            "p95_abs": None,
            "p99_abs": None,
        }
    absolute = delta64.abs().reshape(-1)
    delta_l2 = float(torch.linalg.vector_norm(delta64).item())
    reference_l2 = float(torch.linalg.vector_norm(reference64).item())
    # torch.quantile rejects very large flattened tensors on some builds;
    # NumPy's partition-based quantile handles the largest C1024 tile here.
    absolute_np = absolute.numpy()
    p95, p99 = np.quantile(absolute_np, (0.95, 0.99))
    return {
        "rows": int(delta64.shape[0]),
        "scalar_count": int(absolute.numel()),
        "delta_l2": delta_l2,
        "reference_l2": reference_l2,
        "relative_l2": delta_l2 / (reference_l2 + EPS),
        "mean_abs": float(absolute.mean().item()),
        "max_abs": float(absolute.max().item()),
        "p95_abs": float(p95),
        "p99_abs": float(p99),
    }


def _metric_bundle(delta: torch.Tensor, reference: torch.Tensor) -> Dict[str, Any]:
    if delta.shape != reference.shape:
        raise ValueError(f"metric field shape mismatch: {tuple(delta.shape)} vs {tuple(reference.shape)}")
    return {
        name: _scalar_metric(_group_tensor(delta, spec), _group_tensor(reference, spec))
        for name, spec in GROUPS.items()
    }


def _relation_metrics(
    delta: torch.Tensor,
    reference: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for region, mask in masks.items():
        mask_cpu = mask.detach().cpu().bool()
        result[region] = _metric_bundle(delta[mask_cpu], reference[mask_cpu])
    return result


def _metric_value(
    record: Mapping[str, Any],
    relation: str,
    region: str = "all",
    group: str = "all_six_joint",
    field: str = "relative_l2",
) -> Optional[float]:
    try:
        value = record["metrics"][relation][region][group][field]
        return None if value is None else float(value)
    except (KeyError, TypeError, ValueError):
        return None


def _ratio_metric(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    return float(numerator / (denominator + EPS))


def _make_agreement_visual(
    path: Path,
    *,
    uv_fine: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    box: Sequence[int],
    hidden: torch.Tensor,
    size: int = 192,
) -> None:
    columns = (
        "G1024",
        "PG256",
        "PAG1024",
        "PG256-PAG1024",
        "abs diff",
        "hidden",
    )
    rows = ("rgb", "metallic", "roughness")
    row_to_spec: Dict[str, slice | int] = {
        "rgb": slice(0, 3),
        "metallic": 3,
        "roughness": 4,
    }
    header = 36
    label_width = 110
    sheet = Image.new(
        "RGB",
        (label_width + len(columns) * size, header + len(rows) * (size + 22)),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(columns):
        draw.text((label_width + column * size + 4, 10), label, fill=(255, 255, 255))
    for row_index, row_name in enumerate(rows):
        y = header + row_index * (size + 22)
        draw.text((7, y + size // 2 - 7), row_name, fill=(255, 255, 255))
        spec = row_to_spec[row_name]
        for column, label in enumerate(columns):
            if label == "G1024":
                panel_values = fields["g_fine"][:, spec]
                signed = False
            elif label == "PG256":
                panel_values = fields["p_g256"][:, spec]
                signed = False
            elif label == "PAG1024":
                panel_values = fields["pa_g"][:, spec]
                signed = False
            elif label == "PG256-PAG1024":
                panel_values = fields["delta"][:, spec]
                signed = True
            elif label == "abs diff":
                panel_values = fields["delta"].abs()[:, spec]
                signed = False
            else:
                panel_values = hidden.to(torch.float32)
                signed = False
            panel = _make_panel(
                uv_fine,
                panel_values.mean(dim=1) if panel_values.ndim == 2 and panel_values.shape[1] > 1 else panel_values.reshape(-1),
                box=box,
                size=size,
                signed=signed,
            )
            sheet.paste(panel, (label_width + column * size, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_scatter(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    points = []
    for record in records:
        x = _metric_value(record, "coefficient_discrepancy")
        y = _metric_value(record, "coarse_agreement")
        if x is not None and y is not None:
            points.append((int(record["tile_id"]), x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.5, 5.5), dpi=160)
        if points:
            xs = [point[1] for point in points]
            ys = [point[2] for point in points]
            axis.scatter(xs, ys, s=44, color="#2f6f9f")
            for tile_id, x, y in points:
                axis.annotate(f"{tile_id:02d}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("coefficient discrepancy: ||AG1024-G256|| / ||G256||")
        axis.set_ylabel("lifted field agreement: ||PAG1024-PG256|| / ||PAG1024||")
        axis.set_title("Coefficient-space vs fine field-space discrepancy")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
    except Exception:
        # Keep the required artifact even in environments without matplotlib.
        image = Image.new("RGB", (900, 650), "white")
        draw = ImageDraw.Draw(image)
        draw.text((30, 25), "coefficient vs field discrepancy", fill="black")
        if points:
            xs = [point[1] for point in points]
            ys = [point[2] for point in points]
            x_max = max(max(xs), 1e-8)
            y_max = max(max(ys), 1e-8)
            for tile_id, x, y in points:
                px = 80 + int(760 * x / x_max)
                py = 570 - int(500 * y / y_max)
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(47, 111, 159))
                draw.text((px + 6, py - 8), f"{tile_id:02d}", fill="black")
        image.save(path)


def _load_local_mesh_from_decoder(
    source_dir: Path,
    tile_id: int,
    global_camera: Mapping[str, Any],
    pipeline: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any, Mapping[str, Any], Mapping[str, Any], float]:
    endpoint = _load_torch(source_dir / "tiles" / f"tile_{tile_id:02d}" / "endpoints.pt")
    transform = core.TileCameraTransform(**endpoint["transform"])
    mesh, decode_stats = _decode_purehr_mesh(
        pipeline,
        endpoint,
        label=f"test_c tile {tile_id:02d} PureHR",
    )
    cache = _load_torch(
        source_dir / "global_stitched_quality" / "decoded_global_tiles" / f"tile_{tile_id:02d}.pt"
    )
    local_vertices = mesh.vertices.detach().cpu().to(torch.float32)
    faces = mesh.faces.detach().cpu().to(torch.int32)
    reconstructed_global = _local_to_global_normalized(
        local_vertices,
        transform=transform,
        global_camera=global_camera,
        chunk_size=65_536,
    )
    cached_global = cache["global_vertices"].detach().cpu().to(torch.float32)
    if reconstructed_global.shape != cached_global.shape:
        raise RuntimeError(
            f"tile {tile_id}: decoder/global cache vertex shape mismatch "
            f"{tuple(reconstructed_global.shape)} vs {tuple(cached_global.shape)}"
        )
    roundtrip = (reconstructed_global - cached_global).abs()
    roundtrip_max = float(roundtrip.max().item()) if roundtrip.numel() else 0.0
    del mesh, endpoint, reconstructed_global, cached_global, roundtrip
    return local_vertices, faces, transform, cache, decode_stats, roundtrip_max


def _expected_phase1_support(phase1_dir: Path, tile_id: int) -> Dict[str, Optional[int]]:
    path = phase1_dir / "tiles" / f"tile_{tile_id:02d}" / "operator_metrics.json"
    if not path.is_file():
        return {"C256": None, "C1024": None}
    try:
        payload = _load_torch(path) if path.suffix == ".pt" else __import__("json").loads(path.read_text())
        geometry = payload.get("geometry", {})
        return {
            "C256": int(geometry["C256_active_support"]),
            "C1024": int(geometry["C1024_active_support"]),
        }
    except Exception:
        return {"C256": None, "C1024": None}


def _process_tile(
    *,
    args: argparse.Namespace,
    source_dir: Path,
    phase1_dir: Path,
    output_dir: Path,
    tile_id: int,
    row: Mapping[str, Any],
    pipeline: Any,
    global_field: MeshWithVoxel,
    global_camera: Mapping[str, Any],
    visibility: Mapping[str, Any],
) -> Dict[str, Any]:
    tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tile_dir / "coarse_agreement_metrics.json"
    visual_path = output_dir / "visualizations" / f"tile_{tile_id:02d}_coarse_agreement.png"
    if bool(args.resume) and metrics_path.is_file() and visual_path.is_file():
        payload = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[tile {tile_id:02d}] reused Test C cache", flush=True)
        return payload

    (
        local_vertices,
        local_faces,
        transform,
        cache,
        decode_stats,
        geometry_roundtrip_max,
    ) = _load_local_mesh_from_decoder(
        source_dir, tile_id, global_camera, pipeline
    )
    coarse_coords, _, _ = _voxelize_support(local_vertices, local_faces, COARSE_RESOLUTION)
    fine_coords, _, _ = _voxelize_support(local_vertices, local_faces, GLOBAL_RESOLUTION)
    coarse_points = _cell_centers(coarse_coords, COARSE_RESOLUTION)
    fine_points = _cell_centers(fine_coords, GLOBAL_RESOLUTION)
    expected = _expected_phase1_support(phase1_dir, tile_id)
    support_match = {
        "phase1_C256": expected["C256"],
        "phase1_C1024": expected["C1024"],
        "C256_match": expected["C256"] is None or expected["C256"] == int(coarse_coords.shape[0]),
        "C1024_match": expected["C1024"] is None or expected["C1024"] == int(fine_coords.shape[0]),
    }
    if not support_match["C256_match"] or not support_match["C1024_match"]:
        raise RuntimeError(
            f"tile {tile_id}: reconstructed support differs from Phase1: {support_match}"
        )

    print(
        f"[tile {tile_id:02d}] mesh={local_vertices.shape[0]:,} "
        f"C256={coarse_coords.shape[0]:,} C1024={fine_coords.shape[0]:,}",
        flush=True,
    )
    g_coarse, _, _ = _query_global_at_local_points(
        global_field,
        coarse_points,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    )
    g_fine, global_fine_points, uv_fine = _query_global_at_local_points(
        global_field,
        fine_points,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    )
    q_global_fine = global_fine_points * (2.0 * float(global_camera["mesh_scale"]))
    _, depth_fine, finite_fine = core._project_global_q_to_4096(
        q_global_fine, global_camera=global_camera
    )
    observed = _sample_visibility(
        uv_fine,
        depth_fine,
        finite_fine,
        visibility,
        depth_tolerance_pixels=float(args.depth_tolerance_pixels),
        focal_pixels=float(core._focal_pixels(float(global_camera["camera_angle_x"]), CANONICAL_RESOLUTION)),
    )
    hidden = ~observed

    operator, operator_info = _build_prolongation(coarse_coords, fine_points)
    a_ls_np, solve_info = _solve_restriction(
        operator, g_fine, label=f"tile_{tile_id:02d}_TestC_G1024"
    )
    a_ls = torch.from_numpy(a_ls_np)
    p_g256 = _apply_operator(operator, g_coarse)
    pa_g = _apply_operator(operator, a_ls)

    masks = {"all": torch.ones(fine_points.shape[0], dtype=torch.bool), "observed": observed, "hidden": hidden}
    relation_deltas = {
        "coarse_agreement": (p_g256 - pa_g, pa_g),
        "ls_vs_g1024": (pa_g - g_fine, g_fine),
        "native_vs_g1024": (p_g256 - g_fine, g_fine),
        "coefficient_discrepancy": (a_ls - g_coarse, g_coarse),
    }
    metrics = {}
    for relation, (delta, reference) in relation_deltas.items():
        # E_coeff lives on C256 coefficient rows, so the C1024 observed/
        # hidden mask is not applicable.  Region-wise diagnostics are only
        # defined for the three fine-space field comparisons.
        relation_masks = (
            {"all": torch.ones(delta.shape[0], dtype=torch.bool)}
            if relation == "coefficient_discrepancy"
            else masks
        )
        metrics[relation] = _relation_metrics(delta, reference, relation_masks)
    metrics["lift_ratio"] = {}
    for region in REGIONS:
        metrics["lift_ratio"][region] = {}
        for group in GROUPS:
            coefficient = metrics["coefficient_discrepancy"].get("all", {}).get(group, {})
            metrics["lift_ratio"][region][group] = _ratio_metric(
                metrics["coarse_agreement"][region][group]["relative_l2"],
                coefficient.get("relative_l2"),
            )

    _make_agreement_visual(
        visual_path,
        uv_fine=uv_fine,
        fields={"g_fine": g_fine, "p_g256": p_g256, "pa_g": pa_g, "delta": p_g256 - pa_g},
        box=row["box"],
        hidden=hidden,
    )
    result = {
        "format": f"{FORMAT}_tile",
        "tile_id": int(tile_id),
        "box": [int(value) for value in row["box"]],
        "status": "success",
        "geometry": {
            "local_vertices": int(local_vertices.shape[0]),
            "local_faces": int(local_faces.shape[0]),
            "cache_geometry_roundtrip_max_abs": geometry_roundtrip_max,
            "C256_active_support": int(coarse_coords.shape[0]),
            "C1024_active_support": int(fine_coords.shape[0]),
            "query_definition": "same active O-voxel cell centers as Phase1",
            "support_match": support_match,
        },
        "mask": {
            "observed_count": int(observed.sum()),
            "hidden_count": int(hidden.sum()),
            "observed_ratio": float(observed.float().mean().item()),
            "rule": visibility.get("rule"),
            "depth_tolerance_pixels": float(args.depth_tolerance_pixels),
        },
        "P": operator_info,
        "A": solve_info,
        "decode": decode_stats,
        "metrics": metrics,
        "diagnostic_only": {
            "coefficient_discrepancy": True,
            "coarse_agreement_is_field_space": True,
        },
        "artifacts": {"visualization": str(visual_path)},
    }
    _write_json(metrics_path, result)
    del local_vertices, local_faces, cache, coarse_coords, fine_coords, coarse_points, fine_points
    del g_coarse, g_fine, a_ls, p_g256, pa_g, operator
    _empty_cuda_cache()
    print(
        f"[tile {tile_id:02d}] done agreement="
        f"{_metric_value(result, 'coarse_agreement'):.6g} "
        f"native={_metric_value(result, 'native_vs_g1024'):.6g} "
        f"LS={_metric_value(result, 'ls_vs_g1024'):.6g}",
        flush=True,
    )
    return result


def _aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    relations = ("coarse_agreement", "ls_vs_g1024", "native_vs_g1024", "coefficient_discrepancy", "lift_ratio")
    fields = ("relative_l2", "mean_abs", "p95_abs", "max_abs", "delta_l2", "reference_l2")
    aggregate: Dict[str, Any] = {}
    for region in REGIONS:
        aggregate[region] = {}
        for relation in relations:
            aggregate[region][relation] = {}
            for group in GROUPS:
                aggregate[region][relation][group] = {}
                for field in fields:
                    if relation == "lift_ratio":
                        values = [
                            record.get("metrics", {}).get("lift_ratio", {}).get(region, {}).get(group)
                            for record in successful
                        ]
                    else:
                        values = [
                            record.get("metrics", {}).get(relation, {})
                            .get(region, {})
                            .get(group, {})
                            .get(field)
                            for record in successful
                        ]
                    aggregate[region][relation][group][field] = _finite_stats(values)
    return aggregate


def _tile_csv_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        tile_id = record.get("tile_id")
        if record.get("status") != "success":
            rows.append({"tile_id": tile_id, "status": record.get("status"), "reason": record.get("reason")})
            continue
        row: Dict[str, Any] = {
            "tile_id": tile_id,
            "status": "success",
            "C256_active": record.get("geometry", {}).get("C256_active_support"),
            "C1024_active": record.get("geometry", {}).get("C1024_active_support"),
            "observed_ratio": record.get("mask", {}).get("observed_ratio"),
        }
        for region in REGIONS:
            for relation in ("coarse_agreement", "ls_vs_g1024", "native_vs_g1024", "coefficient_discrepancy"):
                prefix = f"{relation}_{region}"
                metric = record["metrics"].get(relation, {}).get(region, {}).get("all_six_joint", {})
                for field in ("relative_l2", "mean_abs", "p95_abs", "p99_abs", "max_abs", "delta_l2"):
                    row[f"{prefix}_{field}"] = metric.get(field) if metric else None
            row[f"lift_ratio_{region}"] = record["metrics"]["lift_ratio"][region]["all_six_joint"]
        for group in GROUPS:
            row[f"agreement_all_{group}_relative_l2"] = record["metrics"]["coarse_agreement"]["all"][group]["relative_l2"]
            row[f"agreement_all_{group}_mean_abs"] = record["metrics"]["coarse_agreement"]["all"][group]["mean_abs"]
            row[f"agreement_all_{group}_p95_abs"] = record["metrics"]["coarse_agreement"]["all"][group]["p95_abs"]
        rows.append(row)
    return rows


def _get_aggregate_value(aggregate: Mapping[str, Any], relation: str, group: str = "all_six_joint", region: str = "all", field: str = "mean") -> Optional[float]:
    value = aggregate.get(region, {}).get(relation, {}).get(group, {}).get("relative_l2", {}).get(field)
    return None if value is None else float(value)


def _write_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_dir: Path,
    phase1_dir: Path,
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> Dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") != "success"]
    group = "all_six_joint"
    agreement = _get_aggregate_value(aggregate, "coarse_agreement", group)
    ls_error = _get_aggregate_value(aggregate, "ls_vs_g1024", group)
    native_error = _get_aggregate_value(aggregate, "native_vs_g1024", group)
    coeff_error = _get_aggregate_value(aggregate, "coefficient_discrepancy", group)
    hidden_agreement = _get_aggregate_value(aggregate, "coarse_agreement", group, "hidden")
    observed_agreement = _get_aggregate_value(aggregate, "coarse_agreement", group, "observed")
    lift_ratio = _get_aggregate_value(aggregate, "lift_ratio", group)
    native_ls_gap = None if native_error is None or ls_error is None else native_error - ls_error
    channel_means = {
        name: _get_aggregate_value(aggregate, "coarse_agreement", name, "all", "mean")
        for name in GROUPS
    }
    largest_channel = max(
        ((name, value) for name, value in channel_means.items() if value is not None),
        key=lambda item: item[1],
        default=("unknown", None),
    )
    if agreement is not None and coeff_error is not None and agreement < coeff_error:
        coefficient_shrinks = "是：lift 后的 field-space agreement 小于 coefficient-space discrepancy"
    else:
        coefficient_shrinks = "否：lift 后没有小于 coefficient-space discrepancy"
    if agreement is not None and native_error is not None and ls_error is not None:
        if agreement < min(native_error, ls_error) and coeff_error is not None and agreement < coeff_error:
            case = "Case A（高度一致的 coarse content）"
        elif coeff_error is not None and agreement < coeff_error:
            case = "Case B（部分一致）"
        else:
            case = "Case C（明显不一致）"
    else:
        case = "无法分类（指标缺失）"
    if case.startswith("Case A"):
        next_stage = "支持继续评估 PG256 + (I-PA)H 的 hidden-only field reconstruction，但仍需单独验证 detail 的语义。"
        range_conclusion = "支持把 range(P) 作为 shared coarse field space；PA/I-PA 的数学定义沿用 Phase1 的 AP 验证。"
    elif case.startswith("Case B"):
        next_stage = "暂缓直接进入 per-step guidance；可进入 PG256 + (I-PA)H 的诊断性重建，但先分析通道/region mismatch。"
        range_conclusion = "数学上的 range(P)、PA、I-PA 定义仍成立，但 native coarse 与 LS coarse 的语义一致性只得到部分支持。"
    else:
        next_stage = "不建议进入下一轮 hidden-only reconstruction 或 per-step guidance，先研究 native restriction / support mismatch。"
        range_conclusion = "数学分解仍可定义，但 Test C 不支持把 native C256 当作与 LS coarse 相同的 shared content。"

    report_lines = [
        "# Test C：Native C256 与 LS coarse 的 fine-space coarse-content agreement",
        "",
        "## 实验范围",
        "",
        f"- source cache：`{source_dir}`",
        f"- Phase1 reference cache：`{phase1_dir}`",
        f"- CUDA device：`{args.cuda_device}`",
        f"- successful tiles：`{len(successful)}`；failed tiles：`{len(failed)}`",
        "- 复用 Phase1 的 15 个有效 tile、local mesh、C256/C1024 active O-voxel query rows、显式 sparse P、least-squares A 和 observed/hidden mask。",
        "- 本实验只做 field-space diagnostic：不跑 texture flow、不做 hidden correction、不 encode 回 SLat、不 stitching、不渲染。",
        "- 没有预设人工 pass/fail threshold；结论依据三类误差的相对关系。",
        "",
        f"## 核心结果（all-six joint，{len(successful)} tile aggregate mean）",
        "",
        f"- `E_agree = ||PAG1024-PG256|| / ||PAG1024||`：`{agreement}`",
        f"- `E_LS = ||PAG1024-G1024|| / ||G1024||`：`{ls_error}`",
        f"- `E_native = ||PG256-G1024|| / ||G1024||`：`{native_error}`",
        f"- 旧 coefficient discrepancy `E_coeff = ||AG1024-G256|| / ||G256||`：`{coeff_error}`（diagnostic_only=true）",
        f"- `R_lift = E_agree / E_coeff`：`{lift_ratio}`",
        f"- observed `E_agree`：`{observed_agreement}`；hidden `E_agree`：`{hidden_agreement}`",
        f"- `E_native - E_LS`：`{native_ls_gap}`",
        "",
        "## 通道分析",
        "",
        f"- all-six agreement relative L2 最大的 channel/group：`{largest_channel[0]}`，mean=`{largest_channel[1]}`。",
        "- 每个 group 同时记录 relative L2、absolute L2、mean abs、max abs、p95 abs、p99 abs；metallic 不仅依赖 relative L2。",
        "- 完整的 RGB joint、R/G/B、metallic、roughness、alpha、all-six joint aggregate 位于 `coarse_agreement_metrics.json`。",
        "",
        "## 必须问题的结论",
        "",
        f"1. `PG256` 与 `PAG1024` 在同一 fine field space 的平均差异：all-six relative L2=`{agreement}`，mean abs=`{aggregate.get('all', {}).get('coarse_agreement', {}).get(group, {}).get('mean_abs', {}).get('mean')}`，p95 abs=`{aggregate.get('all', {}).get('coarse_agreement', {}).get(group, {}).get('p95_abs', {}).get('mean')}`。",
        f"2. 差异最大的 group（按 mean relative L2）：`{largest_channel[0]}`；请结合其 absolute L2/mean abs 判断低 norm channel 的相对值。",
        f"3. hidden coarse agreement=`{hidden_agreement}`，observed=`{observed_agreement}`，整体=`{agreement}`；hidden/overall ratio=`{None if agreement is None or hidden_agreement is None else hidden_agreement / (agreement + EPS)}`。",
        f"4. coefficient discrepancy 在经过 P 后显著缩小？**{coefficient_shrinks}**；`R_lift={lift_ratio}`。",
        f"5. `E_native` 与 `E_LS` 的差异：`E_native-E_LS={native_ls_gap}`；两者分别为 `{native_error}` 与 `{ls_error}`。",
        f"6. native C256 与 LS coarse 的 coarse content 判断：**{case}**。",
        f"7. 是否支持继续使用 `Vc=range(P)`、`Pi_c=PA`、`Pi_d=I-PA`：**{range_conclusion}**",
        f"8. 是否值得进入 `PG256+(I-PA)H` hidden-only reconstruction：**{next_stage}**",
        "",
        "## Interpretation",
        "",
        f"本次数据按相对关系属于 **{case}**。旧的 coefficient-space discrepancy 只作为诊断项，不作为 coarse-space failure 判据。",
        "Test C 的 coarse agreement 比较严格使用同一个 C1024 fine query support，因此 PG256 与 PAG1024 没有 support 或可视化坐标差异造成的混淆。",
        "",
        "## 产物",
        "",
        "- `coarse_agreement_metrics.json`：逐 tile、逐 region、逐 PBR channel/group 的完整指标。",
        "- `tile_stats.csv`：逐 tile 的主指标。",
        "- `visualizations/tile_XX_coarse_agreement.png`：G1024、PG256、PAG1024、signed/absolute difference、hidden mask。",
        "- `coefficient_vs_field_discrepancy.png`：15 tile 的 coefficient-space 与 field-space discrepancy 关系图。",
        "",
    ]
    report_path = output_dir / "TEST_C_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    summary = {
        "format": f"{FORMAT}_summary",
        "codex": "Test C coarse-content agreement",
        "cuda_device": int(args.cuda_device),
        "source_dir": str(source_dir),
        "phase1_dir": str(phase1_dir),
        "successful_tiles": [int(record["tile_id"]) for record in successful],
        "failed_tiles": [int(record["tile_id"]) for record in failed],
        "aggregate": aggregate,
        "conclusion": {
            "case": case,
            "coefficient_discrepancy_is_diagnostic_only": True,
            "range_P_supported": case.startswith("Case A") or case.startswith("Case B"),
            "recommend_next_hidden_reconstruction": not case.startswith("Case C"),
            "recommend_per_step_guidance": False,
        },
        "artifacts": {
            "report": str(report_path),
            "metrics": str(output_dir / "coarse_agreement_metrics.json"),
            "tile_stats": str(output_dir / "tile_stats.csv"),
            "scatter": str(output_dir / "coefficient_vs_field_discrepancy.png"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="outputs/codex_texture_pbr_degradation_cuda4_all_tiles")
    parser.add_argument("--phase1-dir", default="outputs/sparse_mra_hidden_phase1")
    parser.add_argument("--visibility-dir", default="outputs/visibility_guided_pbr_flow_cuda4_mesh_ovoxel_slat/visibility_4096")
    parser.add_argument("--output-dir", default="outputs/sparse_mra_test_c")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-visualization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--depth-tolerance-pixels", type=float, default=4.0)
    return parser


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for global PBR query")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    torch.cuda.set_device(int(args.cuda_device))
    print(
        f"[cuda] requested/current={int(args.cuda_device)}/{torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )
    source_dir = Path(args.source_dir).expanduser().resolve()
    phase1_dir = Path(args.phase1_dir).expanduser().resolve()
    visibility_dir = Path(args.visibility_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = __import__("json").loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    global_camera = __import__("json").loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    selected = _parse_ids(args.tile_ids)
    rows = [
        dict(row)
        for row in source_summary.get("tiles", [])
        if row.get("status") == "success"
        and (selected is None or int(row["tile_id"]) in selected)
    ]
    rows.sort(key=lambda row: int(row["tile_id"]))
    if len(rows) != 15 and selected is None:
        raise RuntimeError(f"current source does not expose the required 15 successful tiles: {len(rows)}")
    if not rows:
        raise RuntimeError("no successful tiles selected")

    visibility = _load_visibility(visibility_dir)
    print(f"[visibility] {visibility.get('rule')} source={visibility_dir}", flush=True)
    baseline_payload = _load_torch(source_dir / "global_baseline_mesh.pt")
    baseline_mesh = baseline_payload["mesh"] if isinstance(baseline_payload, Mapping) else baseline_payload
    if not isinstance(baseline_mesh, MeshWithVoxel):
        raise RuntimeError(f"expected MeshWithVoxel global baseline, got {type(baseline_mesh)!r}")
    global_field = core._make_attribute_query_mesh(baseline_mesh, torch.device("cuda"))
    del baseline_payload, baseline_mesh
    _empty_cuda_cache()

    need_processing = []
    for row in rows:
        tile_id = int(row["tile_id"])
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        visual_path = output_dir / "visualizations" / f"tile_{tile_id:02d}_coarse_agreement.png"
        if not (bool(args.resume) and (tile_dir / "coarse_agreement_metrics.json").is_file() and visual_path.is_file()):
            need_processing.append(row)
    pipeline = None
    if need_processing:
        print("[Pipeline] loading official decoder for exact Phase1 local mesh", flush=True)
        pipeline = init_pipeline(args.model_path, device="cuda", low_vram=True)

    records: List[Dict[str, Any]] = []
    for row in rows:
        tile_id = int(row["tile_id"])
        try:
            records.append(
                _process_tile(
                    args=args,
                    source_dir=source_dir,
                    phase1_dir=phase1_dir,
                    output_dir=output_dir,
                    tile_id=tile_id,
                    row=row,
                    pipeline=pipeline,
                    global_field=global_field,
                    global_camera=global_camera,
                    visibility=visibility,
                )
            )
        except Exception as exc:
            failure = {
                "format": f"{FORMAT}_tile",
                "tile_id": tile_id,
                "box": row.get("box"),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            records.append(failure)
            _write_json(output_dir / "tiles" / f"tile_{tile_id:02d}" / "coarse_agreement_metrics.json", failure)
            print(f"[tile {tile_id:02d}] FAILED: {failure['reason']}", flush=True)
            traceback.print_exc()
        finally:
            _empty_cuda_cache()
    del pipeline, global_field
    _empty_cuda_cache()

    successful = [record for record in records if record.get("status") == "success"]
    aggregate = _aggregate(records)
    _write_json(
        output_dir / "coarse_agreement_metrics.json",
        {
            "format": f"{FORMAT}_metrics",
            "cuda_device": int(args.cuda_device),
            "source_dir": str(source_dir),
            "phase1_dir": str(phase1_dir),
            "successful_tiles": [int(record["tile_id"]) for record in successful],
            "failed_tiles": [int(record["tile_id"]) for record in records if record.get("status") != "success"],
            "aggregate": aggregate,
            "tiles": records,
        },
    )
    _write_csv(output_dir / "tile_stats.csv", _tile_csv_rows(records))
    _save_scatter(output_dir / "coefficient_vs_field_discrepancy.png", successful)
    summary = _write_report(
        args=args,
        output_dir=output_dir,
        source_dir=source_dir,
        phase1_dir=phase1_dir,
        records=records,
        aggregate=aggregate,
    )
    print(f"[done] report={summary['artifacts']['report']}", flush=True)
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
