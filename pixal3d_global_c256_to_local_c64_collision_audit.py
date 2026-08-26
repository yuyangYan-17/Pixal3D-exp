#!/usr/bin/env python3
"""Audit fresh global C256 support mapped into 49 projective local C64 lattices.

This is intentionally a geometry-only program.  It loads only vertices/faces
from the baseline mesh and never imports latent features from that artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import o_voxel
import torch
from PIL import Image, ImageDraw, ImageFont

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core


FORMAT = "pixal3d_global_c256_to_local_c64_collision_audit_v1"
SUPPORT_KEYS = {
    "format", "resolution", "coords", "global_row_ids", "source_mesh_sha256",
    "voxelizer_config", "downsample_config",
}
FORBIDDEN_SUPPORT_KEYS = {
    "attrs", "features", "dual_vertices", "intersected", "pbr", "PBR",
    "baseline_attrs", "surface_points",
}
VOXELIZER_CONFIG = {
    "function": "o_voxel.convert.mesh_to_flexible_dual_grid",
    "grid_size": 4096,
    "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    "face_weight": 1.0,
    "boundary_weight": 0.2,
    "regularization_weight": 1e-2,
}
DOWNSAMPLE_CONFIG = {
    "source_resolution": 4096,
    "target_resolution": 256,
    "integer_floor_divisor": 16,
    "stable_order": "linear_key_(x*256+y)*256+z",
}
STARTS = tuple(range(0, 3073, 512))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = coords.to(torch.int64)
    return (xyz[:, 0] * resolution + xyz[:, 1]) * resolution + xyz[:, 2]


def stable_unique_coords(coords: torch.Tensor, resolution: int) -> tuple[torch.Tensor, int]:
    keys = linear_keys(coords, resolution)
    keys, order = torch.sort(keys, stable=True)
    sorted_coords = coords.index_select(0, order)
    keep = torch.ones(keys.numel(), dtype=torch.bool, device=keys.device)
    if keys.numel() > 1:
        keep[1:] = keys[1:] != keys[:-1]
    return sorted_coords[keep].contiguous(), int(keys.numel() - keep.sum().item())


def downsample_c4096_to_c256(coords4096: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords_all = torch.div(coords4096.to(torch.int64), 16, rounding_mode="floor")
    coords, inverse, counts = torch.unique(
        coords_all, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    order = torch.argsort(linear_keys(coords, 256), stable=True)
    old_to_new = torch.empty_like(order)
    old_to_new[order] = torch.arange(order.numel(), dtype=order.dtype, device=order.device)
    return coords[order].to(torch.int32).contiguous(), old_to_new[inverse], counts[order]


def endpoint_q(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    return 2.0 * coords.to(torch.float64) / float(resolution - 1) - 1.0


def cell_center_q(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    return 2.0 * (coords.to(torch.float64) + 0.5) / float(resolution) - 1.0


def c64_coords_from_q(q_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous = (q_local + 1.0) * 63.0 / 2.0
    coords = torch.round(continuous).to(torch.int32)
    valid = torch.isfinite(continuous).all(1) & (coords >= 0).all(1) & (coords < 64).all(1)
    return coords, valid, continuous - coords.to(continuous.dtype)


def tile_layout() -> list[dict[str, Any]]:
    result = []
    for y0 in STARTS:
        for x0 in STARTS:
            result.append({"tile_id": len(result), "box": (x0, y0, x0 + 1024, y0 + 1024)})
    return result


def half_open_membership(uv: torch.Tensor, box: Sequence[int], finite: torch.Tensor | None = None) -> torch.Tensor:
    x0, y0, x1, y1 = box
    mask = torch.isfinite(uv).all(1) if finite is None else finite.clone()
    return mask & (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)


def collision_groups(coords64: torch.Tensor, global_row_ids: torch.Tensor) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Return collision membership without dropping original row identities."""
    if coords64.numel() == 0:
        return [], torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    keys = linear_keys(coords64, 64)
    unique_keys, inverse, counts = torch.unique(keys, sorted=True, return_inverse=True, return_counts=True)
    groups = []
    for local_index in torch.nonzero(counts > 1, as_tuple=False).flatten().tolist():
        positions = torch.nonzero(inverse == local_index, as_tuple=False).flatten()
        groups.append({
            "local_c64_coord": coords64[int(positions[0])].tolist(),
            "positions": positions.tolist(),
            "global_row_ids": global_row_ids.index_select(0, positions).to(torch.int64).tolist(),
            "multiplicity": int(positions.numel()),
            "linear_key": int(unique_keys[local_index]),
        })
    return groups, inverse, counts


def collision_stats(coords64: torch.Tensor, global_row_ids: torch.Tensor) -> dict[str, Any]:
    groups, inverse, counts = collision_groups(coords64, global_row_ids)
    valid = int(coords64.shape[0])
    unique = int(counts.numel())
    collided_rows = int(counts[counts > 1].sum().item())
    hist = Counter(int(x) for x in counts.tolist())
    return {
        "valid_local_rows": valid,
        "unique_local_c64_cells": unique,
        "collision_cell_count": len(groups),
        "collided_row_count": collided_rows,
        "collision_excess_rows": valid - unique,
        "collision_cell_fraction": len(groups) / unique if unique else 0.0,
        "collided_row_fraction": collided_rows / valid if valid else 0.0,
        "local_multiplicity_histogram": {str(k): hist[k] for k in sorted(hist)},
        "max_collision_multiplicity": int(counts.max().item()) if counts.numel() else 0,
        "inverse": inverse,
        "counts": counts,
        "groups": groups,
    }


def support_schema_is_clean(payload: Mapping[str, Any]) -> bool:
    keys = set(payload)
    return keys == SUPPORT_KEYS and not (keys & FORBIDDEN_SUPPORT_KEYS)


def cache_matches(payload: Mapping[str, Any], mesh_sha256: str) -> bool:
    return (
        payload.get("format") == f"{FORMAT}_c4096_cache"
        and payload.get("source_mesh_sha256") == mesh_sha256
        and payload.get("voxelizer_config") == VOXELIZER_CONFIG
        and isinstance(payload.get("coords"), torch.Tensor)
    )


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    x = values.to(torch.float64).flatten()
    if not x.numel():
        return {"count": 0}
    return {
        "count": int(x.numel()), "min": float(x.min()), "mean": float(x.mean()),
        "p50": float(torch.quantile(x, 0.50)), "p95": float(torch.quantile(x, 0.95)),
        "p99": float(torch.quantile(x, 0.99)), "max": float(x.max()),
    }


def _hist(values: Iterable[int]) -> dict[str, int]:
    counter = Counter(int(x) for x in values)
    return {str(k): counter[k] for k in sorted(counter)}


def _project_chunked(q: torch.Tensor, camera: Mapping[str, float], chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    uv, depth, finite = [], [], []
    for start in range(0, q.shape[0], chunk_size):
        a, b, c = core._project_global_q_to_image(
            q[start:start + chunk_size], global_camera=camera, image_width=4096, image_height=4096
        )
        uv.append(a.cpu()); depth.append(b.cpu()); finite.append(c.cpu())
    return torch.cat(uv), torch.cat(depth), torch.cat(finite)


def _quantized_metrics(original_coords: torch.Tensor, reconstructed_q: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    rec = torch.round((reconstructed_q + 1.0) * 255.0 / 2.0).to(torch.int64)
    delta = rec - original_coords.to(torch.int64)
    abs_delta = delta.abs()
    l1 = abs_delta.sum(1)
    l2 = torch.linalg.vector_norm(delta.to(torch.float64), dim=1)
    linf = abs_delta.max(1).values
    q_original = endpoint_q(original_coords, 256)
    qerr = torch.linalg.vector_norm(reconstructed_q.to(torch.float64) - q_original, dim=1)
    metrics = {
        "rows": int(rec.shape[0]),
        "identity_rows": int((delta == 0).all(1).sum()),
        "identity_fraction": float((delta == 0).all(1).to(torch.float64).mean()) if rec.shape[0] else 0.0,
        "index_l1_histogram": _hist(l1.tolist()),
        "index_l2_histogram": _hist(torch.round(l2 * 1000).to(torch.int64).tolist()),
        "index_l2_histogram_scale": 1000,
        "index_linf_histogram": _hist(linf.tolist()),
        "q_global_reconstruction_error": _distribution(qerr),
    }
    return metrics, (delta == 0).all(1)


def _enrich_group(
    group: dict[str, Any], tile_id: int, box: Sequence[int], global_coords: torch.Tensor,
    q_global: torch.Tensor, q_local: torch.Tensor, uv_global: torch.Tensor,
    uv_tile: torch.Tensor, residual: torch.Tensor, depth: torch.Tensor,
) -> dict[str, Any]:
    pos = torch.tensor(group["positions"], dtype=torch.int64)
    rows = torch.tensor(group["global_row_ids"], dtype=torch.int64)
    gc = global_coords.index_select(0, rows).to(torch.float64)
    qg = q_global.index_select(0, pos).to(torch.float64)
    ql = q_local.index_select(0, pos).to(torch.float64)
    ug = uv_global.index_select(0, pos).to(torch.float64)
    ut = uv_tile.index_select(0, pos).to(torch.float64)
    rr = residual.index_select(0, pos).to(torch.float64)
    dd = depth.index_select(0, pos).to(torch.float64)
    span_c = gc.max(0).values - gc.min(0).values
    span_q = qg.max(0).values - qg.min(0).values
    uv_dist = torch.cdist(ug, ug).max() if ug.shape[0] > 1 else torch.tensor(0.0)
    out = {
        "tile_id": tile_id, "box": list(box), "local_c64_coord": group["local_c64_coord"],
        "linear_key": group["linear_key"], "multiplicity": group["multiplicity"],
        "global_row_ids": rows.tolist(), "global_c256_coords": gc.to(torch.int64).tolist(),
        "q_global": qg.tolist(), "q_local_continuous": ql.tolist(),
        "uv_4096": ug.tolist(), "uv_tile": ut.tolist(), "quantization_residual": rr.tolist(),
        "global_c256_coord_diameter": {
            "l1": float(span_c.sum()), "l2": float(torch.linalg.vector_norm(span_c)),
            "linf": float(span_c.max()),
        },
        "global_q_diameter": {
            "l1": float(span_q.sum()), "l2": float(torch.linalg.vector_norm(span_q)),
            "linf": float(span_q.max()),
        },
        "uv_diameter_pixels": float(uv_dist),
        "depth_range": float(dd.max() - dd.min()),
    }
    return out


def _split_quantized(metrics_mask: torch.Tensor, collision_mask: torch.Tensor, original: torch.Tensor, reconstructed_q: torch.Tensor) -> dict[str, Any]:
    result = {}
    for name, mask in (("collision_rows", collision_mask), ("non_collision_rows", ~collision_mask)):
        if mask.any():
            result[name] = _quantized_metrics(original[mask], reconstructed_q[mask])[0]
        else:
            result[name] = {"rows": 0, "identity_rows": 0, "identity_fraction": 0.0}
    return result


def _audit_convention(
    name: str, q_all: torch.Tensor, coords256: torch.Tensor, uv_all: torch.Tensor,
    depth_all: torch.Tensor, finite_all: torch.Tensor, camera: Mapping[str, float], chunk_size: int,
    output_dir: Path, write_mappings: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]], dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []
    memberships: list[list[int]] = [[] for _ in range(coords256.shape[0])]
    quant_parts: list[dict[str, Any]] = []
    max_roundtrip = 0.0
    for tile in tile_layout():
        tile_id, box = tile["tile_id"], tile["box"]
        transform = core._derive_tile_camera(
            tile_id=tile_id, box=box, global_camera=camera, source_width=4096,
            source_height=4096, model_width=1024, model_height=1024, extend_pixel=0,
        )
        selected = torch.nonzero(half_open_membership(uv_all, box, finite_all), as_tuple=False).flatten()
        for row in selected.tolist():
            memberships[row].append(tile_id)
        qg = q_all.index_select(0, selected)
        ql, uv_tile = core._global_q_to_local_q(qg, global_camera=camera, transform=transform)
        back, _ = core._local_q_to_global_q(ql, global_camera=camera, transform=transform)
        abs_err = (back - qg).abs().flatten()
        rt = _distribution(abs_err)
        max_roundtrip = max(max_roundtrip, float(rt.get("max", 0.0)))
        c64, valid, residual = c64_coords_from_q(ql)
        valid_pos = torch.nonzero(valid, as_tuple=False).flatten()
        valid_rows = selected.index_select(0, valid_pos)
        valid_c64 = c64.index_select(0, valid_pos)
        base = collision_stats(valid_c64, valid_rows)
        groups = base.pop("groups")
        inverse = base.pop("inverse")
        counts = base.pop("counts")
        collision_row_mask = counts.index_select(0, inverse) > 1 if counts.numel() else torch.empty(0, dtype=torch.bool)
        ql_quant = endpoint_q(valid_c64, 64)
        qg_reconstructed, _ = core._local_q_to_global_q(ql_quant, global_camera=camera, transform=transform)
        qmetrics, identity = _quantized_metrics(coords256.index_select(0, valid_rows), qg_reconstructed)
        qmetrics.update(_split_quantized(identity, collision_row_mask, coords256.index_select(0, valid_rows), qg_reconstructed))
        quant_parts.append({
            "tile_id": tile_id, "global_row_ids": valid_rows, "identity": identity,
            "collision_mask": collision_row_mask, "q_reconstructed": qg_reconstructed,
        })
        local_examples = []
        valid_qg = qg.index_select(0, valid_pos)
        valid_ql = ql.index_select(0, valid_pos)
        valid_uvg = uv_all.index_select(0, valid_rows)
        valid_uvt = uv_tile.index_select(0, valid_pos)
        valid_res = residual.index_select(0, valid_pos)
        valid_depth = depth_all.index_select(0, valid_rows)
        for group in groups:
            local_examples.append(_enrich_group(
                group, tile_id, box, coords256, valid_qg, valid_ql, valid_uvg,
                valid_uvt, valid_res, valid_depth,
            ))
        all_examples.extend(local_examples)
        row = {
            "convention": name, "tile_id": tile_id, "box": list(box),
            "candidate_global_rows": int(selected.numel()),
            "out_of_local_range_rows": int((~valid).sum()), **base,
            "continuous_roundtrip": rt, "quantized_roundtrip": qmetrics,
        }
        stats_rows.append(row)
        if write_mappings:
            atomic_torch_save(output_dir / "tile_mappings" / f"tile_{tile_id:02d}.pt", {
                "format": FORMAT, "convention": name, "tile_id": tile_id, "box": box,
                "global_row_ids": selected.to(torch.int64), "valid_mask": valid,
                "valid_global_row_ids": valid_rows.to(torch.int64), "coords64": valid_c64,
                "collision_inverse": inverse.to(torch.int64), "collision_counts": counts.to(torch.int64),
                "q_local_continuous": valid_ql.to(torch.float32),
                "uv_4096": valid_uvg.to(torch.float32), "uv_tile": valid_uvt.to(torch.float32),
                "quantization_residual": valid_res.to(torch.float32),
            })
        print(f"[{name}] tile {tile_id:02d}/48 candidates={selected.numel()} valid={valid.sum().item()} collisions={base['collision_cell_count']}", flush=True)
    if max_roundtrip >= 2e-5:
        raise RuntimeError(f"continuous camera roundtrip failed for {name}: {max_roundtrip}")

    # Global quantized aggregation is over tile memberships, as required by the audit.
    originals, reconstructed, collided = [], [], []
    for part in quant_parts:
        rows = part["global_row_ids"]
        originals.append(coords256.index_select(0, rows)); reconstructed.append(part["q_reconstructed"])
        collided.append(part["collision_mask"])
    original_cat = torch.cat(originals) if originals else torch.empty((0, 3), dtype=torch.int32)
    reconstructed_cat = torch.cat(reconstructed) if reconstructed else torch.empty((0, 3), dtype=torch.float64)
    collision_cat = torch.cat(collided) if collided else torch.empty(0, dtype=torch.bool)
    overall, identity = _quantized_metrics(original_cat, reconstructed_cat)
    overall.update(_split_quantized(identity, collision_cat, original_cat, reconstructed_cat))
    overall["continuous_roundtrip_max_abs_error"] = max_roundtrip
    return stats_rows, all_examples, memberships, overall


def _select_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    per_tile = {}
    for tile_id in range(49):
        rows = [e for e in examples if e["tile_id"] == tile_id]
        per_tile[str(tile_id)] = sorted(rows, key=lambda e: (e["linear_key"], e["global_row_ids"]))[:100]
    return {
        "max_multiplicity_top100": sorted(
            examples, key=lambda e: (-e["multiplicity"], e["tile_id"], e["linear_key"])
        )[:100],
        "max_global_coordinate_distance_top100": sorted(
            examples, key=lambda e: (-e["global_c256_coord_diameter"]["l2"], e["tile_id"], e["linear_key"])
        )[:100],
        "stable_first100_per_tile": per_tile,
    }


def _cross_tile(memberships: list[list[int]], uv: torch.Tensor) -> dict[str, Any]:
    counts = torch.tensor([len(x) for x in memberships], dtype=torch.int64)
    pairs = []
    sets = [set(i for i, tiles in enumerate(memberships) if t in tiles) for t in range(49)]
    for a in range(49):
        for b in range(a + 1, 49):
            shared = len(sets[a] & sets[b]); union = len(sets[a] | sets[b])
            if shared:
                pairs.append({"tile_a": a, "tile_b": b, "shared_global_row_count": shared, "jaccard": shared / union})
    zero = torch.nonzero(counts == 0, as_tuple=False).flatten()
    zero_uv = uv.index_select(0, zero) if zero.numel() else torch.empty((0, 2))
    return {
        "membership_count_histogram": _hist(counts.tolist()),
        "zero_membership_rows": int(zero.numel()),
        "zero_membership_global_row_ids": zero.tolist(),
        "zero_membership_uv_min": zero_uv.min(0).values.tolist() if zero.numel() else None,
        "zero_membership_uv_max": zero_uv.max(0).values.tolist() if zero.numel() else None,
        "rows_with_membership_gt4": int((counts > 4).sum()),
        "membership_by_global_row_id": memberships,
        "tile_pair_shared_counts_and_jaccard": pairs,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(r["valid_local_rows"] for r in rows)
    unique = sum(r["unique_local_c64_cells"] for r in rows)
    collided = sum(r["collided_row_count"] for r in rows)
    cells = sum(r["collision_cell_count"] for r in rows)
    excess = sum(r["collision_excess_rows"] for r in rows)
    return {
        "tile_count": 49, "tiles_with_within_tile_c64_collision": sum(r["collision_excess_rows"] > 0 for r in rows),
        "all_tile_valid_local_rows": valid, "all_tile_unique_local_c64_cells": unique,
        "all_tile_collision_cell_count": cells, "all_tile_collided_row_count": collided,
        "all_tile_collision_excess_rows": excess,
        "collision_cell_fraction": cells / unique if unique else 0.0,
        "collided_row_fraction": collided / valid if valid else 0.0,
        "max_collision_multiplicity": max(r["max_collision_multiplicity"] for r in rows),
        "worst_by_collided_row_fraction": [r["tile_id"] for r in sorted(rows, key=lambda x: (-x["collided_row_fraction"], x["tile_id"]))[:5]],
        "worst_by_excess_rows": [r["tile_id"] for r in sorted(rows, key=lambda x: (-x["collision_excess_rows"], x["tile_id"]))[:5]],
        "worst_by_max_multiplicity": [r["tile_id"] for r in sorted(rows, key=lambda x: (-x["max_collision_multiplicity"], x["tile_id"]))[:5]],
    }


def _write_csv(path: Path, endpoint_rows: list[dict[str, Any]], center_rows: list[dict[str, Any]]) -> None:
    fields = [
        "convention", "tile_id", "box", "candidate_global_rows", "valid_local_rows",
        "out_of_local_range_rows", "unique_local_c64_cells", "collision_cell_count",
        "collided_row_count", "collision_excess_rows", "collision_cell_fraction",
        "collided_row_fraction", "max_collision_multiplicity", "local_multiplicity_histogram",
        "continuous_roundtrip_max_abs_error", "quantized_c256_identity_fraction",
    ]
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row in endpoint_rows + center_rows:
                writer.writerow({
                    **{k: row.get(k) for k in fields},
                    "box": json.dumps(row["box"]),
                    "local_multiplicity_histogram": json.dumps(row["local_multiplicity_histogram"], sort_keys=True),
                    "continuous_roundtrip_max_abs_error": row["continuous_roundtrip"].get("max", 0.0),
                    "quantized_c256_identity_fraction": row["quantized_roundtrip"]["identity_fraction"],
                })
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _draw_visualizations(output_dir: Path, endpoint_rows: list[dict[str, Any]], memberships: list[list[int]], uv: torch.Tensor, examples: list[dict[str, Any]], canonical: Path) -> None:
    vis = output_dir / "visualizations"; vis.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (7 * 250, 7 * 150), "white"); draw = ImageDraw.Draw(sheet)
    for row in endpoint_rows:
        x = (row["tile_id"] % 7) * 250; y = (row["tile_id"] // 7) * 150
        frac = min(1.0, row["collided_row_fraction"] * 20)
        color = (255, int(240 * (1-frac)), int(240 * (1-frac)))
        draw.rectangle((x, y, x+249, y+149), fill=color, outline="black")
        lines = [f"tile {row['tile_id']:02d}", f"candidate {row['candidate_global_rows']}",
                 f"unique {row['unique_local_c64_cells']}", f"collision cells {row['collision_cell_count']}",
                 f"collided rows {row['collided_row_count']}", f"max mult {row['max_collision_multiplicity']}"]
        draw.multiline_text((x+8, y+8), "\n".join(lines), fill="black", font=font, spacing=4)
    sheet.save(vis / "collision_contact_sheet.png")

    coverage = Image.open(canonical).convert("RGB").resize((1024, 1024))
    d = ImageDraw.Draw(coverage, "RGBA")
    palette = [(100,100,100,100), (30,120,255,150), (30,200,80,150), (255,180,20,150), (230,30,60,170)]
    for i, tiles in enumerate(memberships):
        u, v = uv[i].tolist()
        if math.isfinite(u) and math.isfinite(v) and 0 <= u < 4096 and 0 <= v < 4096:
            c = palette[min(len(tiles), 4)]; x, y = int(u/4), int(v/4)
            d.ellipse((x-1, y-1, x+1, y+1), fill=c)
    coverage.save(vis / "cross_tile_coverage_4096.png")

    worst = max(endpoint_rows, key=lambda r: (r["collided_row_fraction"], r["collision_excess_rows"], -r["tile_id"]))
    wid = worst["tile_id"]; groups = [e for e in examples if e["tile_id"] == wid]
    canvas = Image.new("RGB", (1800, 900), "white")
    gt = Image.open(canonical).convert("RGB").crop(tuple(worst["box"])).resize((900, 900))
    gd = ImageDraw.Draw(gt)
    colors = [(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),(145,30,180)]
    for gi, group in enumerate(groups):
        color = colors[gi % len(colors)]
        for p in group["uv_4096"]:
            x, y = p[0]-worst["box"][0], p[1]-worst["box"][1]
            gd.ellipse((x-3,y-3,x+3,y+3), fill=color)
        if group["uv_4096"]:
            x, y = group["uv_4096"][0]; gd.text((x-worst["box"][0]+4,y-worst["box"][1]+4), str(group["local_c64_coord"]), fill=color, font=font)
    canvas.paste(gt, (0,0)); cd = ImageDraw.Draw(canvas)
    views = [(0,1,"XY"),(0,2,"XZ"),(1,2,"YZ")]
    for vi,(a,b,label) in enumerate(views):
        ox, oy = 930 + (vi%2)*420, 50 + (vi//2)*420
        cd.rectangle((ox,oy,ox+350,oy+350), outline="black"); cd.text((ox,oy-20), label, fill="black", font=font)
        for gi,g in enumerate(groups):
            c=colors[gi%len(colors)]; coord=g["local_c64_coord"]
            x=ox+int(coord[a]/63*350); y=oy+350-int(coord[b]/63*350); rad=min(12,2+g["multiplicity"])
            cd.ellipse((x-rad,y-rad,x+rad,y+rad), fill=c)
    canvas.save(vis / f"worst_tile_{wid:02d}_local_c64_collision.png")


def _preflight(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    paths = {name: Path(value).resolve() for name, value in {
        "input_image": args.input_image, "canonical_image": args.canonical_image,
        "baseline_mesh": args.baseline_mesh, "camera_json": args.camera_json,
    }.items()}
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError(path)
    summary_path = paths["baseline_mesh"].parent.parent / "summary.json"
    if not summary_path.is_file(): raise FileNotFoundError(summary_path)
    paths["summary"] = summary_path
    fps = {name: fingerprint(path) for name, path in paths.items()}
    config = {
        "format": FORMAT, "status": "preflight", "files": fps,
        "voxelizer_config": VOXELIZER_CONFIG, "downsample_config": DOWNSAMPLE_CONFIG,
        "tile_config": {"source_size": 4096, "tile_size": 1024, "stride": 512, "starts": STARTS},
        "chunk_size": args.chunk_size, "cache_policy": args.cache_policy,
        "runtime": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    atomic_json(output_dir / "config.json", config)
    summary = json.loads(summary_path.read_text()); camera = json.loads(paths["camera_json"].read_text())
    input_size = Image.open(paths["input_image"]).size; canonical_size = Image.open(paths["canonical_image"]).size
    checks = {
        "summary_status_complete": summary.get("status") == "complete",
        "summary_input_path_matches": Path(summary.get("input", "")).resolve() == paths["input_image"],
        "summary_source_size_matches_input": tuple(summary.get("preprocess", {}).get("source_size", ())) == input_size,
        "canonical_size_is_4096": canonical_size == (4096, 4096),
        "summary_output_resolution_is_4096": summary.get("output_resolution") == 4096,
        "summary_mesh_path_matches": Path(summary.get("baseline", {}).get("mesh", "")).resolve() == paths["baseline_mesh"],
        "summary_camera_path_matches": Path(summary.get("render", {}).get("manifest", {}).get("camera", "")).resolve() == paths["camera_json"],
        "summary_camera_values_match": all(abs(float(summary.get("camera", {}).get(k, math.inf))-float(camera[k])) < 1e-9 for k in ("camera_angle_x","distance","mesh_scale")),
    }
    preflight = {"format": FORMAT, "status": "passed" if all(checks.values()) else "failed", "checks": checks,
                 "input_image_size": input_size, "canonical_image_size": canonical_size}
    atomic_json(output_dir / "preflight.json", preflight)
    if not all(checks.values()): raise RuntimeError(f"input consistency failed: {checks}")
    return config, {k: float(camera[k]) for k in ("camera_angle_x","distance","mesh_scale")}, fps


def _voxelize(args: argparse.Namespace, output_dir: Path, mesh_sha: str) -> tuple[torch.Tensor, dict[str, Any]]:
    cache = output_dir / "c4096_occupancy_cache.pt"
    if cache.is_file() and args.cache_policy != "rebuild":
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if cache_matches(payload, mesh_sha):
            coords = payload["coords"].to(torch.int32).contiguous()
            if coords.ndim == 2 and coords.shape[1] == 3 and not bool(((coords < 0)|(coords >= 4096)).any()):
                stats = dict(payload["stats"]); stats["cache_reused"] = True
                return coords, stats
        if args.cache_policy == "require": raise RuntimeError("C4096 cache fingerprint/config mismatch")
        print("[cache] fingerprint/config mismatch; refusing reuse and rebuilding", flush=True)
    elif args.cache_policy == "require":
        raise RuntimeError("C4096 cache required but absent")
    print("[voxelizer] loading mesh vertices/faces only", flush=True)
    artifact = torch.load(args.baseline_mesh, map_location="cpu", weights_only=False)
    mesh = artifact["mesh"] if isinstance(artifact, dict) else artifact
    vertices = mesh.vertices.to(torch.float32).cpu().contiguous()
    faces = mesh.faces.to(torch.int32).cpu().contiguous()
    del artifact, mesh
    start = time.perf_counter()
    result = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces, grid_size=4096,
        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], face_weight=1.0,
        boundary_weight=0.2, regularization_weight=1e-2,
    )
    raw = result[0].to(torch.int32).cpu().contiguous()
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0: raise RuntimeError(f"bad voxelizer output {raw.shape}")
    if bool(((raw < 0)|(raw >= 4096)).any()): raise RuntimeError("C4096 coordinate out of range")
    coords, duplicates = stable_unique_coords(raw, 4096)
    stats = {"raw_voxelizer_token_count": int(raw.shape[0]), "unique_c4096_token_count": int(coords.shape[0]),
             "voxelizer_duplicate_count": duplicates, "coord_min": coords.min(0).values.tolist(),
             "coord_max": coords.max(0).values.tolist(), "seconds": time.perf_counter()-start, "cache_reused": False}
    atomic_torch_save(cache, {"format": f"{FORMAT}_c4096_cache", "source_mesh_sha256": mesh_sha,
                              "voxelizer_config": VOXELIZER_CONFIG, "coords": coords, "stats": stats})
    return coords, stats


def _report(output_dir: Path, c4096: int, c256: int, endpoint_summary: dict[str, Any], center_summary: dict[str, Any], quant: dict[str, Any], cross: dict[str, Any], max_rt: float, classifications: dict[str, int], verdict: str) -> None:
    hist = cross["membership_count_histogram"]
    text = f"""# Global C256 → local C64 单视图冲突审计报告

本实验只使用 GT 正面和 baseline mesh 的 `vertices/faces`，没有读取旧 support、attrs、PBR 或任何 flow/encoder/decoder。

## 直接结论

- fresh C4096 active coords：**{c4096}**；fresh C256 active coords：**{c256}**。
- endpoint q 下，49 tile 中 **{endpoint_summary['tiles_with_within_tile_c64_collision']}** 个出现 **Within-tile C64 collision**。
- collision cells / collided rows / excess rows：**{endpoint_summary['all_tile_collision_cell_count']} / {endpoint_summary['all_tile_collided_row_count']} / {endpoint_summary['all_tile_collision_excess_rows']}**；比例分别为 **{endpoint_summary['collision_cell_fraction']:.8%} / {endpoint_summary['collided_row_fraction']:.8%}**；最大 multiplicity **{endpoint_summary['max_collision_multiplicity']}**。
- collision 反例分类（分类可重叠）：相邻 global C256 cell {classifications.get('adjacent',0)} 组，不同深度表面 {classifications.get('different_depth',0)} 组，接近 C64 量化边界 {classifications.get('quantization_boundary',0)} 组。细节见 `collision_examples.json`。
- cell-center 下 collision tile 数为 **{center_summary['tiles_with_within_tile_c64_collision']}**，最终 one-to-one 结论与 endpoint **{'一致' if bool(center_summary['tiles_with_within_tile_c64_collision']) == bool(endpoint_summary['tiles_with_within_tile_c64_collision']) else '不一致'}**。
- continuous camera roundtrip 最大绝对误差 **{max_rt:.10g}**，**{'通过' if max_rt < 2e-5 else '未通过'}** `<2e-5`。
- endpoint quantized local C64 反投影到原 global C256 index 的比例：**{quant['identity_fraction']:.8%}**。
- **Cross-tile membership overlap** 分布：0={hist.get('0',0)}，1={hist.get('1',0)}，2={hist.get('2',0)}，4={hist.get('4',0)}；其他计数见 JSON；>4 异常行 **{cross['rows_with_membership_gt4']}**。

## Verdict

`{verdict}`

`direct_global_row_to_local_sparse_row_safe={str(verdict == 'one_to_one_for_all_tiles').lower()}`。因此，{'可以' if verdict == 'one_to_one_for_all_tiles' else '不可以'}直接把 global C256 row gather 成 local SparseTensor row。本轮按要求停止，不进入 restriction/prolongation 或 flow。

“Within-tile C64 collision”与“Cross-tile membership overlap”在全部统计中分开处理；后者是重叠 tile 的预期覆盖，不参与同 tile local key 去重。
"""
    path = output_dir / "REPORT.md"
    fd, name = tempfile.mkstemp(prefix=".REPORT.md.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--canonical-image", required=True)
    parser.add_argument("--baseline-mesh", required=True)
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--cache-policy", choices=("auto","rebuild","require"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if args.chunk_size <= 0: raise ValueError("chunk-size must be positive")
    try:
        config, camera, fps = _preflight(args, output_dir)
    except Exception:
        if not (output_dir / "preflight.json").exists(): atomic_json(output_dir / "preflight.json", {"format":FORMAT,"status":"failed"})
        raise
    coords4096, c4096_stats = _voxelize(args, output_dir, fps["baseline_mesh"]["sha256"])
    atomic_json(output_dir / "c4096_occupancy_stats.json", c4096_stats)
    coords256, inverse, child_counts = downsample_c4096_to_c256(coords4096)
    valid_support = (coords256.shape[0] > 0 and not bool(((coords256<0)|(coords256>=256)).any())
                     and bool((linear_keys(coords256,256)[1:] > linear_keys(coords256,256)[:-1]).all()))
    if not valid_support: raise RuntimeError("invalid fresh C256 support")
    support = {"format": FORMAT, "resolution": 256, "coords": coords256, "global_row_ids": torch.arange(coords256.shape[0],dtype=torch.int64),
               "source_mesh_sha256": fps["baseline_mesh"]["sha256"], "voxelizer_config": VOXELIZER_CONFIG,
               "downsample_config": DOWNSAMPLE_CONFIG}
    if not support_schema_is_clean(support): raise RuntimeError("support schema contains forbidden/unknown fields")
    atomic_torch_save(output_dir / "c256_support.pt", support)
    child_hist = _hist(child_counts.tolist())
    boundary = ((coords256 == 0)|(coords256 == 255)).sum(0)
    c256_stats = {"unique_c256_token_count": int(coords256.shape[0]), "coord_min": coords256.min(0).values.tolist(),
                  "coord_max": coords256.max(0).values.tolist(), "boundary_hits_per_axis": boundary.tolist(),
                  "c4096_children_per_c256": _distribution(child_counts), "c4096_children_per_c256_histogram": child_hist,
                  "inverse4096_rows": int(inverse.numel())}
    atomic_json(output_dir / "c256_support_stats.json", c256_stats)

    q_endpoint = endpoint_q(coords256,256); q_center = cell_center_q(coords256,256)
    uv_endpoint, depth_endpoint, finite_endpoint = _project_chunked(q_endpoint,camera,args.chunk_size)
    uv_center, depth_center, finite_center = _project_chunked(q_center,camera,args.chunk_size)
    endpoint_rows, endpoint_examples, memberships, endpoint_quant = _audit_convention(
        "endpoint",q_endpoint,coords256,uv_endpoint,depth_endpoint,finite_endpoint,camera,args.chunk_size,output_dir,True)
    center_rows, center_examples, _, center_quant = _audit_convention(
        "cell_center",q_center,coords256,uv_center,depth_center,finite_center,camera,args.chunk_size,output_dir,False)
    _write_csv(output_dir / "per_tile_collision_stats.csv", endpoint_rows, center_rows)
    selected = {"endpoint":_select_examples(endpoint_examples),"cell_center":_select_examples(center_examples)}
    atomic_json(output_dir / "collision_examples.json", selected)
    cross = _cross_tile(memberships,uv_endpoint)
    if cross["rows_with_membership_gt4"]: raise RuntimeError("membership >4 indicates tile membership implementation error")
    atomic_json(output_dir / "cross_tile_membership.json",cross)
    atomic_json(output_dir / "quantized_roundtrip.json",{"endpoint":endpoint_quant,"cell_center":center_quant})
    endpoint_summary = _summary(endpoint_rows); center_summary = _summary(center_rows)
    classifications = {"adjacent":0,"different_depth":0,"quantization_boundary":0}
    for e in endpoint_examples:
        if e["global_c256_coord_diameter"]["linf"] <= 1: classifications["adjacent"] += 1
        if max(c[2] for c in e["global_c256_coords"]) - min(c[2] for c in e["global_c256_coords"]) > 1: classifications["different_depth"] += 1
        if any(abs(abs(v)-0.5) < 0.05 for row in e["quantization_residual"] for v in row): classifications["quantization_boundary"] += 1
    verdict = "within_tile_c64_collision_found" if endpoint_summary["all_tile_collision_excess_rows"] else "one_to_one_for_all_tiles"
    collision_summary = {"format":FORMAT,"status":"complete","verdict":verdict,
                         "direct_global_row_to_local_sparse_row_safe":verdict=="one_to_one_for_all_tiles",
                         "endpoint":endpoint_summary,"cell_center":center_summary,"collision_origin_classification":classifications,
                         "continuous_camera_gate_passed":endpoint_quant["continuous_roundtrip_max_abs_error"]<2e-5}
    atomic_json(output_dir / "collision_summary.json",collision_summary)
    _draw_visualizations(output_dir,endpoint_rows,memberships,uv_endpoint,endpoint_examples,Path(args.canonical_image))
    _report(output_dir,int(coords4096.shape[0]),int(coords256.shape[0]),endpoint_summary,center_summary,endpoint_quant,cross,
            endpoint_quant["continuous_roundtrip_max_abs_error"],classifications,verdict)
    config["status"]="complete"; config["verdict"]=verdict; atomic_json(output_dir / "config.json",config)
    print(f"[complete] verdict={verdict} output={output_dir}",flush=True)


if __name__ == "__main__":
    main()
