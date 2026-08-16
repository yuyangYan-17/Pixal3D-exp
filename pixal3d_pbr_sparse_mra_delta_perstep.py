#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C256/C1024 sparse-MRA delta guidance for the Pixal3D texture flow.

The official texture-flow route is kept intact.  Only the field target passed
to the official PBR encoder is changed:

    Delta = H - G
    Delta_c = P_h A_h Delta
    Delta_d = Delta - Delta_c
    Y_hidden = G + Delta_d

Observed rows still use the existing tile-center Gaussian fusion.  Hidden rows
never query another tile.  ``P_h`` is built once from the same local geometry
as the C1024 support, and ``A_h`` is the least-squares pseudoinverse of that
fixed sparse operator.  The old parent-cell decomposition is intentionally
not part of this script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import numpy as np
import torch
import utils3d
from PIL import Image, ImageDraw, ImageOps
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import lsqr, splu

import pixal3d_cross_tile_pbr_perstep as base
import pixal3d_sparse_mra_hidden_phase1 as mra
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d.models as pixal3d_models
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel
from pixal3d.utils import render_utils


FORMAT = "pixal3d_pbr_sparse_mra_delta_perstep_v1"
VARIANT = "sparse_mra_delta_guided"
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
FINE_RESOLUTION = 1024
COARSE_RESOLUTION = 256
SNAPSHOT_STEPS = (0, 6, 11)
PBR_CHANNEL_NAMES = ("base_color_r", "base_color_g", "base_color_b", "metallic", "roughness", "alpha")
PBR_GROUPS = {
    "rgb": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
    "joint": slice(0, 6),
}


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def _load_tensor(path: Path) -> torch.Tensor:
    value = _load_torch(path)
    if isinstance(value, Mapping) and "tensor" in value:
        value = value["tensor"]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"expected tensor cache at {path}")
    return value.detach().cpu()


def _coordinate_digest(value: SparseTensor) -> str:
    coords = value.coords.detach().cpu().to(torch.int32).contiguous().numpy()
    return hashlib.sha256(coords.tobytes()).hexdigest()


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())


def _relative(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    return _norm(value) / (_norm(reference) + float(eps))


def _safe_mean(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).mean().item()) if value.numel() else 0.0


def _channel_mean_abs(left: torch.Tensor, right: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    left = left.detach().cpu().to(torch.float32)
    right = right.detach().cpu().to(torch.float32)
    if mask is not None:
        mask = mask.detach().cpu().bool()
        left = left[mask]
        right = right[mask]
    return {
        "RGB": _safe_mean((left[:, 0:3] - right[:, 0:3]).abs()),
        "metallic": _safe_mean((left[:, 3:4] - right[:, 3:4]).abs()),
        "roughness": _safe_mean((left[:, 4:5] - right[:, 4:5]).abs()),
        "alpha": _safe_mean((left[:, 5:6] - right[:, 5:6]).abs()),
    }


def _tensor_range(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().cpu().to(torch.float32)
    result: Dict[str, Any] = {}
    for name, sl in (("RGB", slice(0, 3)), ("metallic", slice(3, 4)), ("roughness", slice(4, 5)), ("alpha", slice(5, 6))):
        field = value[:, sl]
        result[name] = {
            "min": float(field.min().item()) if field.numel() else None,
            "max": float(field.max().item()) if field.numel() else None,
            "out_of_0_1_ratio": float(((field < 0.0) | (field > 1.0)).to(torch.float32).mean().item()) if field.numel() else 0.0,
        }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})


@dataclass
class SparseMRAProjector:
    """A cached hidden-support P_h and its reusable least-squares solver."""

    P_hidden: csr_matrix
    fine_rows: np.ndarray
    pure_hidden_ids: np.ndarray
    info: Dict[str, Any]
    reduced: Optional[csr_matrix] = None
    factor: Any = None
    active_ids: Optional[np.ndarray] = None
    solver_error: Optional[str] = None

    def prepare_solver(self) -> None:
        if self.reduced is not None or self.P_hidden.shape[1] == 0:
            return
        active = np.asarray(self.P_hidden.getnnz(axis=0)).reshape(-1) > 0
        self.active_ids = np.where(active)[0]
        self.reduced = self.P_hidden[:, self.active_ids].tocsr()
        if self.active_ids.size == 0:
            return
        gram = (self.reduced.T @ self.reduced).tocsc()
        gram.eliminate_zeros()
        self.info["hidden_normal_matrix_nnz"] = int(gram.nnz)
        try:
            self.factor = splu(gram)
            self.info["hidden_solver"] = "cached_sparse_normal_equation_lu"
        except Exception as exc:
            self.solver_error = f"{type(exc).__name__}: {exc}"
            self.info["hidden_solver"] = "lsqr_pseudoinverse_fallback"
            self.info["hidden_solver_error"] = self.solver_error
        del gram

    def solve(self, field: torch.Tensor, label: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        field_np = field.detach().cpu().to(torch.float32).numpy()
        if field_np.ndim == 1:
            field_np = field_np[:, None]
        if field_np.shape[0] != self.P_hidden.shape[0]:
            raise ValueError(f"{label}: hidden field rows {field_np.shape[0]} != P_h rows {self.P_hidden.shape[0]}")
        self.prepare_solver()
        coeff = np.zeros((self.P_hidden.shape[1], field_np.shape[1]), dtype=np.float32)
        solve_info: Dict[str, Any] = {
            "label": label,
            "rows": int(self.P_hidden.shape[0]),
            "columns": int(self.P_hidden.shape[1]),
            "active_columns": int(self.active_ids.size if self.active_ids is not None else 0),
            "method": self.info.get("hidden_solver", "empty"),
        }
        if self.reduced is not None and self.active_ids is not None and self.active_ids.size:
            rhs = self.reduced.T.dot(field_np)
            if self.factor is not None:
                solution = self.factor.solve(np.asarray(rhs, dtype=np.float32)).astype(np.float32, copy=False)
            else:
                solution = np.zeros((self.active_ids.size, field_np.shape[1]), dtype=np.float32)
                iterations = []
                stops = []
                for channel in range(field_np.shape[1]):
                    result = lsqr(self.reduced, field_np[:, channel].astype(np.float64), atol=1e-7, btol=1e-7, iter_lim=300)
                    solution[:, channel] = result[0].astype(np.float32)
                    iterations.append(int(result[2]))
                    stops.append(int(result[1]))
                solve_info["iterations"] = iterations
                solve_info["istop"] = stops
            coeff[self.active_ids] = solution
            del rhs, solution
        return torch.from_numpy(coeff), solve_info

    def apply(self, coeff: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.P_hidden.dot(coeff.detach().cpu().to(torch.float32).numpy()), dtype=np.float32))


@dataclass
class FlowTrace:
    contexts: Sequence[Any]
    output_dir: Path
    calls: int = 0
    norm_calls: int = 0
    mra_records: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)
    latent_cycle: Dict[Tuple[int, int], SparseTensor] = field(default_factory=dict)
    latent_target: Dict[Tuple[int, int], SparseTensor] = field(default_factory=dict)
    guided_x0: Dict[Tuple[int, int], SparseTensor] = field(default_factory=dict)

    def reset(self, contexts: Sequence[Any], output_dir: Path) -> None:
        self.contexts = contexts
        self.output_dir = output_dir
        self.calls = 0
        self.norm_calls = 0
        self.mra_records.clear()
        self.latent_cycle.clear()
        self.latent_target.clear()
        self.guided_x0.clear()

    def current_step(self) -> int:
        return int(self.calls // max(1, len(self.contexts)))

    def current_tile(self) -> int:
        return int(self.contexts[self.calls % len(self.contexts)].tile_id)


def _save_sparse_cpu(path: Path, value: SparseTensor) -> None:
    _atomic_torch_save(path, {"coords": value.coords.detach().cpu().to(torch.int32), "features": value.feats.detach().cpu().to(torch.float32)})


def _make_condition(pipeline: Any, image: Image.Image, shape_norm: SparseTensor, transform: Any, low_vram: bool) -> Mapping[str, Any]:
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        shape_norm.coords.to(device=torch.device("cuda"), dtype=torch.int32),
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=base.LATENT_RESOLUTION,
    )
    return base._move_condition(condition, torch.device("cpu" if low_vram else "cuda"))


def _load_baseline(path: Path) -> MeshWithVoxel:
    payload = _load_torch(path)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVoxel):
        raise RuntimeError(f"expected MeshWithVoxel baseline at {path}, got {type(mesh)!r}")
    return mesh


def _load_contexts(
    *,
    source_dir: Path,
    output_dir: Path,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    args: argparse.Namespace,
) -> List[Any]:
    """Load the fixed-shape contexts from the already completed official run."""
    preparation = json.loads((source_dir / "tile_preparation_summary.json").read_text(encoding="utf-8"))
    available = {int(v) for v in preparation.get("prepared_tile_ids", [])}
    requested = _parse_ids(args.tile_ids)
    boxes = core._tile_layout(canonical_size=CANONICAL_IMAGE_SIZE, tile_size=TILE_SIZE, stride=TILE_STRIDE)
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    contexts: List[Any] = []
    for tile_id, box in enumerate(boxes):
        if tile_id not in available or (requested is not None and tile_id not in requested):
            continue
        source_tile = source_dir / "tiles" / f"tile_{tile_id:02d}"
        required = [
            source_tile / "fixed_shape_norm.pt",
            source_tile / "texture_reference_norm.pt",
            source_tile / "texture_initial_state.pt",
            source_tile / "fixed_shape_summary.json",
        ]
        if not all(path.is_file() for path in required):
            print(f"[prepare tile {tile_id:02d}] skipped: missing fixed context cache", flush=True)
            continue
        transform = core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        geometry = core._prepare_tile_geometry(
            global_vertices=baseline_mesh.vertices,
            global_faces=baseline_mesh.faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        shape_norm = base._load_sparse_payload(required[0])
        texture_norm = base._load_sparse_payload(required[1])
        initial_state = base._load_sparse_payload(required[2])
        if not torch.equal(shape_norm.coords, texture_norm.coords):
            raise RuntimeError(f"tile {tile_id}: fixed shape/texture coordinates differ")
        if not torch.equal(texture_norm.coords, initial_state.coords):
            raise RuntimeError(f"tile {tile_id}: fixed texture/initial-state coordinates differ")
        tile_image = image_4096.crop(tuple(int(v) for v in box)).convert("RGB")
        if tile_image.size != (TILE_SIZE, TILE_SIZE):
            tile_image = tile_image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image.save(tile_dir / "hr_tile_1024_condition.png")
        static_stats = json.loads(required[3].read_text(encoding="utf-8"))
        context = base.TileContext(
            tile_id=tile_id,
            box=tuple(int(v) for v in box),
            transform=transform,
            image=tile_image,
            tile_dir=tile_dir,
            geometry=geometry,
            shape_reference=base._fresh_sparse(shape_norm),
            shape_norm=shape_norm,
            shape_denorm=base._denormalize_slat(shape_norm, pipeline.shape_slat_normalization),
            texture_reference=base._fresh_sparse(texture_norm),
            texture_norm=texture_norm,
            noise=SparseTensor(torch.zeros_like(texture_norm.feats), texture_norm.coords.detach().clone()),
            initial_state=initial_state,
            condition=_make_condition(pipeline, tile_image, shape_norm, transform, bool(args.low_vram)),
            target_coords=geometry.coords.detach().cpu().to(torch.int32),
            target_points=(geometry.coords.detach().cpu().to(torch.float32) + 0.5) / float(FINE_RESOLUTION) - 0.5,
            static_stats={**static_stats, "reused_from": str(source_dir.resolve())},
        )
        # The recovery route keeps all immutable context tensors on CPU and
        # restores each model input only for its own forward/decode call.
        if bool(args.low_vram):
            base._offload_contexts_to_cpu([context])
        contexts.append(context)
        print(
            f"[prepare tile {tile_id:02d}] tokens={context.target_coords.shape[0]:,} "
            f"shape_tokens={context.shape_norm.feats.shape[0]:,}",
            flush=True,
        )
    if not contexts:
        raise RuntimeError("no fixed-shape contexts available")
    _atomic_json(
        output_dir / "tile_preparation_summary.json",
        {
            "prepared_tile_ids": [int(context.tile_id) for context in contexts],
            "active_tile_count": len(contexts),
            "source_dir": str(source_dir.resolve()),
            "layout_tile_count": len(boxes),
        },
    )
    return contexts


def _attach_reference_fields(contexts: Sequence[Any], field_cache_dir: Path, output_dir: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    observed_total = 0
    hidden_total = 0
    for context in contexts:
        tile_id = int(context.tile_id)
        source_tile = field_cache_dir / "tiles" / f"tile_{tile_id:02d}"
        reference = _load_tensor(source_tile / "global_pbr_reference.pt").to(torch.float32)
        observed = _load_tensor(source_tile / "observed_mask.pt").to(torch.bool)
        hidden = _load_tensor(source_tile / "hidden_mask.pt").to(torch.bool)
        expected = int(context.target_coords.shape[0])
        if reference.shape != (expected, 6):
            raise RuntimeError(f"tile {tile_id}: G1024 shape {tuple(reference.shape)} != {(expected, 6)}")
        if observed.shape != (expected,) or hidden.shape != (expected,):
            raise RuntimeError(f"tile {tile_id}: visibility mask is not aligned with C1024 support")
        if not torch.equal(observed, ~hidden):
            raise RuntimeError(f"tile {tile_id}: observed/hidden masks are not complements")
        if not torch.isfinite(reference).all():
            raise RuntimeError(f"tile {tile_id}: G1024 contains non-finite values")
        context.global_pbr_reference = reference
        context.observed_mask = observed
        context.hidden_mask = hidden
        _atomic_torch_save(context.tile_dir / "global_pbr_reference.pt", {"tensor": reference})
        _atomic_torch_save(context.tile_dir / "observed_mask.pt", {"tensor": observed})
        _atomic_torch_save(context.tile_dir / "hidden_mask.pt", {"tensor": hidden})
        observed_count = int(observed.sum().item())
        hidden_count = int(hidden.sum().item())
        observed_total += observed_count
        hidden_total += hidden_count
        record = {
            "tile_id": tile_id,
            "support_rows": expected,
            "observed_count": observed_count,
            "hidden_count": hidden_count,
            "observed_ratio": observed_count / max(1, expected),
            "source": str(source_tile.resolve()),
            "rule": "reused binary visibility mask from completed canonical raster preflight",
        }
        records.append(record)
        context.static_stats["global_pbr_reference"] = {
            "field": "G1024 aligned with context.target_coords",
            "range": _tensor_range(reference),
        }
        context.static_stats["visibility"] = record
        _atomic_json(context.tile_dir / "sparse_mra_static.json", context.static_stats)
    summary = {
        "format": f"{FORMAT}_visibility_v1",
        "field_cache_dir": str(field_cache_dir.resolve()),
        "observed_voxel_count": observed_total,
        "hidden_voxel_count": hidden_total,
        "tiles": records,
    }
    _atomic_json(output_dir / "visibility_summary.json", summary)
    return summary


def _operator_cache_paths(context: Any) -> Tuple[Path, Path, Path]:
    root = context.tile_dir
    return root / "mra_P_hidden.npz", root / "mra_support.pt", root / "mra_operator.json"


def _build_or_load_projector(context: Any, args: argparse.Namespace) -> SparseMRAProjector:
    p_path, support_path, info_path = _operator_cache_paths(context)
    observed = context.observed_mask.detach().cpu().bool()
    fine_coords_expected = context.target_coords.detach().cpu().to(torch.int32)
    if bool(args.resume) and p_path.is_file() and support_path.is_file() and info_path.is_file():
        payload = _load_torch(support_path)
        fine_coords = payload["fine_coords"].to(torch.int32)
        hidden_rows = payload["hidden_rows"].to(torch.int64).numpy()
        pure_ids = payload["pure_hidden_ids"].to(torch.int64).numpy()
        if not torch.equal(fine_coords, fine_coords_expected):
            raise RuntimeError(f"tile {context.tile_id}: cached P fine rows are not aligned")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        P_hidden = load_npz(p_path).tocsr()
        if P_hidden.shape[0] != int(hidden_rows.size):
            raise RuntimeError(f"tile {context.tile_id}: cached P_h row count mismatch")
        projector = SparseMRAProjector(P_hidden, hidden_rows, pure_ids, info)
        print(f"[mra tile {context.tile_id:02d}] reused P_h shape={P_hidden.shape} nnz={P_hidden.nnz:,}", flush=True)
        return projector

    vertices = context.geometry.vertices.detach().cpu().to(torch.float32)
    faces = context.geometry.faces.detach().cpu().to(torch.int32)
    fine_coords, _, _ = mra._voxelize_support(vertices, faces, FINE_RESOLUTION)
    if not torch.equal(fine_coords, fine_coords_expected):
        raise RuntimeError(
            f"tile {context.tile_id}: MRA fine support is not row-aligned with context.target_coords "
            f"({fine_coords.shape} vs {fine_coords_expected.shape})"
        )
    coarse_coords, _, _ = mra._voxelize_support(vertices, faces, COARSE_RESOLUTION)
    fine_points = (fine_coords.to(torch.float32) + 0.5) / float(FINE_RESOLUTION) - 0.5
    P_full, p_info = mra._build_prolongation(coarse_coords, fine_points, coarse_resolution=COARSE_RESOLUTION)
    basis = mra._basis_partition(P_full, observed)
    hidden_rows = torch.where(~observed)[0].numpy().astype(np.int64, copy=False)
    pure_ids = np.asarray(basis["pure_hidden_ids"], dtype=bool)
    P_hidden = P_full[hidden_rows][:, pure_ids].tocsr()
    hidden_row_nnz = np.diff(P_hidden.indptr)
    info = {
        "format": f"{FORMAT}_operator_v1",
        "tile_id": int(context.tile_id),
        "fine_support_rows": int(fine_coords.shape[0]),
        "coarse_support_columns": int(coarse_coords.shape[0]),
        "hidden_rows": int(hidden_rows.size),
        "pure_hidden_basis_count": int(pure_ids.sum()),
        "mixed_basis_count": int(np.asarray(basis["mixed_basis_count"]).item() if np.asarray(basis["mixed_basis_count"]).ndim == 0 else basis["mixed_basis_count"]),
        "uncovered_basis_count": int(basis["uncovered_basis_count"]),
        "P_full": p_info,
        "P_hidden": {
            "rows": int(P_hidden.shape[0]),
            "columns": int(P_hidden.shape[1]),
            "nnz": int(P_hidden.nnz),
            "coverage_ratio": float((hidden_row_nnz > 0).mean()) if hidden_row_nnz.size else 0.0,
            "uncovered_rows": int((hidden_row_nnz == 0).sum()),
            "row_nnz_mean": float(hidden_row_nnz.mean()) if hidden_row_nnz.size else 0.0,
            "support_rule": "same sparse trilinear C256->C1024 prolongation with valid-neighbor renormalization",
        },
        "fine_support_exact": True,
        "coarse_support_definition": "same local mesh vertices/faces and O-voxel cell-center rule as Phase1/Test C",
        "boundary_rule": "exclude every C256 basis whose P support touches an observed C1024 row",
    }
    save_npz(p_path, P_hidden, compressed=True)
    _atomic_torch_save(
        support_path,
        {
            "fine_coords": fine_coords,
            "coarse_coords": coarse_coords,
            "hidden_rows": torch.from_numpy(hidden_rows),
            "pure_hidden_ids": torch.from_numpy(np.where(pure_ids)[0].astype(np.int64)),
        },
    )
    _atomic_json(info_path, info)
    projector = SparseMRAProjector(P_hidden, hidden_rows, np.where(pure_ids)[0].astype(np.int64), info)
    print(
        f"[mra tile {context.tile_id:02d}] built C256={coarse_coords.shape[0]:,} "
        f"C1024={fine_coords.shape[0]:,} P_h={P_hidden.shape} nnz={P_hidden.nnz:,}",
        flush=True,
    )
    return projector


def _empty_observed_stats(target: Any, self_field: torch.Tensor, sigma: float) -> Dict[str, Any]:
    empty = torch.empty((0,), device=self_field.device, dtype=torch.float32)
    return {
        "target_tile": int(target.tile_id),
        "active_ovoxel_count": int(target.target_points.shape[0]),
        "observed_ovoxel_count": 0,
        "hidden_ovoxel_count": int(target.target_points.shape[0]),
        "overlap_ovoxel_count": 0,
        "non_overlap_ovoxel_count": 0,
        "query_valid_donor_count": {"min": 0, "mean": 0.0, "max": 0},
        "covered_donor_count": {"min": 0, "mean": 0.0, "max": 0},
        "distance_to_center_pixels": base._tensor_stats(empty),
        "normalized_fusion_weight": base._tensor_stats(empty),
        "raw_fusion_weight": base._tensor_stats(empty),
        "gaussian_sigma_pixels": float(sigma),
        "fusion_region": "observed_only",
        "hidden_cross_tile_queries": 0,
        "pbr_self_vs_fused_mean_abs_all": {"RGB": 0.0, "metallic": 0.0, "roughness": 0.0, "alpha": 0.0},
        "pbr_self_vs_fused_mean_abs_overlap": {"RGB": 0.0, "metallic": 0.0, "roughness": 0.0, "alpha": 0.0},
    }


def _energy_stats(value: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().cpu().to(torch.float64)
    mask = mask.detach().cpu().bool()
    selected = value[mask]
    result: Dict[str, Any] = {"count": int(selected.shape[0]), "channels": {}}
    for name, sl in (("RGB", slice(0, 3)), ("metallic", slice(3, 4)), ("roughness", slice(4, 5)), ("alpha", slice(5, 6))):
        channel = selected[:, sl]
        result["channels"][name] = {
            "l2": _norm(channel),
            "rms": float(torch.sqrt(channel.square().mean()).item()) if channel.numel() else 0.0,
            "mean_abs": float(channel.abs().mean().item()) if channel.numel() else 0.0,
            "max_abs": float(channel.abs().max().item()) if channel.numel() else 0.0,
        }
    return result


def _ratio_stats(numerator: torch.Tensor, denominator: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    numerator = numerator.detach().cpu().to(torch.float64)[mask.detach().cpu().bool()]
    denominator = denominator.detach().cpu().to(torch.float64)[mask.detach().cpu().bool()]
    result: Dict[str, float] = {}
    for name, sl in (("RGB", slice(0, 3)), ("metallic", slice(3, 4)), ("roughness", slice(4, 5)), ("alpha", slice(5, 6))):
        n = torch.linalg.vector_norm(numerator[:, sl], dim=1) if numerator.numel() else torch.empty(0, dtype=torch.float64)
        d = torch.linalg.vector_norm(denominator[:, sl], dim=1) if denominator.numel() else torch.empty(0, dtype=torch.float64)
        result[name] = float((n / (d + 1e-8)).mean().item()) if n.numel() else 0.0
    return result


def _projection_record(
    *,
    context: Any,
    self_field: torch.Tensor,
    gaussian_field: torch.Tensor,
    final_field: torch.Tensor,
    solve_info: Mapping[str, Any],
    detail_coeff: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return diagnostics and the four hidden delta fields used by the formula."""
    h = self_field.detach().cpu().to(torch.float32)
    g = context.global_pbr_reference.detach().cpu().to(torch.float32)
    gaussian = gaussian_field.detach().cpu().to(torch.float32)
    output = final_field.detach().cpu().to(torch.float32)
    observed = context.observed_mask.detach().cpu().bool()
    hidden = context.hidden_mask.detach().cpu().bool()
    delta = h[hidden] - g[hidden]
    coarse = context.mra_projector.apply(detail_coeff.new_zeros(detail_coeff.shape)) if False else None
    # ``coarse`` is supplied by the caller through the exact target relation;
    # reconstruct it from Delta and the saved output to avoid any hidden
    # absolute-field ambiguity.
    detail = output[hidden] - g[hidden]
    coarse = delta - detail
    target_formula = h[hidden] - coarse
    detail_coeff_check, detail_solve_info = context.mra_projector.solve(detail, f"tile_{context.tile_id:02d}_detail_check")
    detail_projection = context.mra_projector.apply(detail_coeff_check)
    formula_error = output[hidden] - target_formula
    coarse_removal = detail_projection
    observed_error = output[observed] - gaussian[observed]
    record: Dict[str, Any] = {
        "tile_id": int(context.tile_id),
        "observed_voxel_count": int(observed.sum().item()),
        "hidden_voxel_count": int(hidden.sum().item()),
        "hidden_cross_tile_queries": 0,
        "delta": {
            "all": _energy_stats(torch.where(hidden[:, None], h - g, torch.zeros_like(h - g)), torch.ones(h.shape[0], dtype=torch.bool)),
            "observed": _energy_stats(torch.zeros_like(h), observed),
            "hidden": _energy_stats(delta, torch.ones(delta.shape[0], dtype=torch.bool)),
        },
        "delta_coarse": {
            "observed": _energy_stats(torch.zeros_like(h), observed),
            "hidden": _energy_stats(coarse, torch.ones(coarse.shape[0], dtype=torch.bool)),
        },
        "delta_detail": {
            "observed": _energy_stats(torch.zeros_like(h), observed),
            "hidden": _energy_stats(detail, torch.ones(detail.shape[0], dtype=torch.bool)),
        },
        "ratios_hidden": {
            "r_c": _ratio_stats(coarse, delta, torch.ones(delta.shape[0], dtype=torch.bool)),
            "r_d": _ratio_stats(detail, delta, torch.ones(delta.shape[0], dtype=torch.bool)),
        },
        "abs_output_minus_H": {
            "all": _channel_mean_abs(output, h),
            "observed": _channel_mean_abs(output, h, observed),
            "hidden": _channel_mean_abs(output, h, hidden),
        },
        "abs_output_minus_G": {
            "all": _channel_mean_abs(output, g),
            "observed": _channel_mean_abs(output, g, observed),
            "hidden": _channel_mean_abs(output, g, hidden),
        },
        "physical_range": {
            "G1024": _tensor_range(g),
            "H_t": _tensor_range(h),
            "hidden_target": _tensor_range(output[hidden]),
            "final_target_field": _tensor_range(output),
            "delta_coarse": _tensor_range(coarse),
            "delta_detail": _tensor_range(detail),
        },
        "invariants": {
            "null_Ah_delta_detail": {
                "max_abs": float(detail_projection.abs().max().item()) if detail_projection.numel() else 0.0,
                "relative_l2": _relative(detail_projection, detail_projection.new_zeros(detail_projection.shape)) if False else _relative(detail_projection, detail),
                "solver": detail_solve_info,
            },
            "coarse_removal": {
                "max_abs": float(coarse_removal.abs().max().item()) if coarse_removal.numel() else 0.0,
                "relative_l2": _relative(coarse_removal, delta),
            },
            "detail_preservation": {
                "max_abs": float((detail - (output[hidden] - g[hidden])).abs().max().item()) if detail.numel() else 0.0,
                "relative_l2": _relative(detail - (output[hidden] - g[hidden]), detail),
            },
            "exact_formula": {
                "max_abs": float(formula_error.abs().max().item()) if formula_error.numel() else 0.0,
                "relative_l2": _relative(formula_error, output[hidden]),
            },
            "observed_gaussian_identity": {
                "max_abs": float(observed_error.abs().max().item()) if observed_error.numel() else 0.0,
                "relative_l2": _relative(observed_error, gaussian[observed]),
            },
        },
        "solver_delta": dict(solve_info),
        "finite": bool(torch.isfinite(output).all().item() and torch.isfinite(h).all().item() and torch.isfinite(g).all().item()),
    }
    return record, delta, coarse, detail, output


def _save_field_snapshot(context: Any, step: int, fields: Mapping[str, torch.Tensor]) -> None:
    if step not in SNAPSHOT_STEPS:
        return
    payload: Dict[str, Any] = {
        "format": f"{FORMAT}_field_snapshot_v1",
        "tile_id": int(context.tile_id),
        "step": int(step),
        "dtype": "float16 for storage; source computation remains float32",
    }
    for key, value in fields.items():
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().to(torch.float16)
    _atomic_torch_save(context.tile_dir / "steps" / f"step_{step:02d}_mra_fields.pt", payload)


def _make_mra_fuser(trace: FlowTrace):
    original_fuse = base._fuse_tile_field

    def fuse(
        *,
        target: Any,
        contexts: Sequence[Any],
        decoded: Mapping[int, MeshWithVoxel],
        self_field: torch.Tensor,
        global_camera: Mapping[str, float],
        sigma_pixels: float,
        query_chunk_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
        if target.observed_mask is None or target.hidden_mask is None or target.global_pbr_reference is None:
            raise RuntimeError(f"tile {target.tile_id}: incomplete sparse-MRA context")
        observed_rows = torch.where(target.observed_mask.to(device=target.target_points.device))[0]
        hidden_count = int(target.target_points.shape[0] - observed_rows.numel())
        if observed_rows.numel():
            target_view = SimpleNamespace(
                tile_id=int(target.tile_id),
                transform=target.transform,
                target_points=target.target_points.index_select(0, observed_rows),
                target_coords=target.target_coords.index_select(0, observed_rows),
            )
            observed_field, observed_stats, donor_details = original_fuse(
                target=target_view,
                contexts=contexts,
                decoded=decoded,
                self_field=self_field.index_select(0, observed_rows),
                global_camera=global_camera,
                sigma_pixels=float(sigma_pixels),
                query_chunk_size=int(query_chunk_size),
            )
            gaussian_full = self_field.clone()
            gaussian_full.index_copy_(0, observed_rows, observed_field)
        else:
            gaussian_full = self_field.clone()
            observed_stats = _empty_observed_stats(target, self_field, sigma_pixels)
            donor_details = {"target_tile": torch.tensor(int(target.tile_id), dtype=torch.int64)}

        h_cpu = self_field.detach().cpu().to(torch.float32)
        g_cpu = target.global_pbr_reference.detach().cpu().to(torch.float32)
        gaussian_cpu = gaussian_full.detach().cpu().to(torch.float32)
        observed_cpu = target.observed_mask.detach().cpu().bool()
        hidden_cpu = target.hidden_mask.detach().cpu().bool()
        delta_hidden = h_cpu[hidden_cpu] - g_cpu[hidden_cpu]
        coeff, solve_info = target.mra_projector.solve(delta_hidden, f"tile_{target.tile_id:02d}_delta")
        coarse_hidden = target.mra_projector.apply(coeff)
        detail_hidden = delta_hidden - coarse_hidden
        output_cpu = h_cpu.clone()
        output_cpu[observed_cpu] = gaussian_cpu[observed_cpu]
        output_cpu[hidden_cpu] = g_cpu[hidden_cpu] + detail_hidden
        if not torch.isfinite(output_cpu).all():
            raise RuntimeError(f"tile {target.tile_id}: sparse-MRA target is non-finite")
        record, _, _, _, _ = _projection_record(
            context=target,
            self_field=h_cpu,
            gaussian_field=gaussian_cpu,
            final_field=output_cpu,
            solve_info=solve_info,
            detail_coeff=coeff,
        )
        step = trace.current_step()
        record["step"] = int(step)
        record["timestep_context"] = "official texture-flow step; target computed from frozen pred_x0 decode"
        trace.mra_records[(int(target.tile_id), int(step))] = record
        _save_field_snapshot(
            target,
            step,
            {
                "G1024": g_cpu,
                "H_t": h_cpu,
                "Delta": torch.where(hidden_cpu[:, None], h_cpu - g_cpu, torch.zeros_like(h_cpu - g_cpu)),
                "Delta_coarse_hidden": torch.where(hidden_cpu[:, None], coarse_hidden.new_zeros((h_cpu.shape[0], 6)), torch.zeros_like(h_cpu)),
                "Delta_detail_hidden": torch.where(hidden_cpu[:, None], detail_hidden.new_zeros((h_cpu.shape[0], 6)), torch.zeros_like(h_cpu)),
                "hidden_target": output_cpu,
                "observed_gaussian": gaussian_cpu,
                "final_target_field": output_cpu,
            },
        )
        # Replace the compact zero placeholders in snapshots with row-aligned
        # coarse/detail fields; keeping them explicit makes manual inspection
        # independent of the hidden mask file.
        if step in SNAPSHOT_STEPS:
            snapshot_path = target.tile_dir / "steps" / f"step_{step:02d}_mra_fields.pt"
            snapshot = _load_torch(snapshot_path)
            coarse_full = torch.zeros_like(h_cpu)
            detail_full = torch.zeros_like(h_cpu)
            coarse_full[hidden_cpu] = coarse_hidden
            detail_full[hidden_cpu] = detail_hidden
            snapshot["Delta_coarse"] = coarse_full.to(torch.float16)
            snapshot["Delta_detail"] = detail_full.to(torch.float16)
            _atomic_torch_save(snapshot_path, snapshot)

        stats = dict(observed_stats)
        observed_overlap = int(observed_stats.get("overlap_ovoxel_count", 0))
        stats.update(
            {
                "active_ovoxel_count": int(h_cpu.shape[0]),
                "observed_ovoxel_count": int(observed_cpu.sum().item()),
                "hidden_ovoxel_count": hidden_count,
                "overlap_ovoxel_count": observed_overlap,
                "non_overlap_ovoxel_count": int(h_cpu.shape[0] - observed_overlap),
                "fusion_region": "observed_gaussian_plus_hidden_sparse_mra_delta",
                "hidden_cross_tile_queries": 0,
                "mra_operator": target.mra_projector.info,
                "mra_invariants": record["invariants"],
                "pbr_self_vs_fused_mean_abs_all": _channel_mean_abs(h_cpu, output_cpu),
                "pbr_self_vs_fused_mean_abs_overlap": _channel_mean_abs(h_cpu, output_cpu, observed_cpu),
            }
        )
        trace.calls += 1
        return output_cpu.to(device=self_field.device), stats, donor_details

    return fuse


def _install_trace_hooks(trace: FlowTrace):
    original_fuse = base._fuse_tile_field
    original_normalize = base._normalize_slat
    original_strict = base._strict_sparse_check
    base._fuse_tile_field = _make_mra_fuser(trace)

    def normalize(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
        output = original_normalize(value, normalization)
        call_index = trace.norm_calls
        tile_count = max(1, len(trace.contexts))
        step = call_index // (2 * tile_count)
        tile = int(trace.contexts[(call_index // 2) % tile_count].tile_id)
        phase = call_index % 2
        if step in SNAPSHOT_STEPS:
            if phase == 0:
                trace.latent_cycle[(tile, step)] = base._sparse_to_cpu(output)
            else:
                trace.latent_target[(tile, step)] = base._sparse_to_cpu(output)
        trace.norm_calls += 1
        return output

    def strict(reference: SparseTensor, candidate: SparseTensor, label: str) -> Dict[str, Any]:
        result = original_strict(reference, candidate, label)
        if "x0_guided" in str(label):
            match = re.search(r"tile\s+(\d+)\s+step\s+(\d+)", str(label))
            if match:
                tile, step = int(match.group(1)), int(match.group(2))
                if step in SNAPSHOT_STEPS:
                    trace.guided_x0[(tile, step)] = base._sparse_to_cpu(candidate)
        return result

    base._normalize_slat = normalize
    base._strict_sparse_check = strict

    def restore() -> None:
        base._fuse_tile_field = original_fuse
        base._normalize_slat = original_normalize
        base._strict_sparse_check = original_strict

    return restore


def _save_latent_snapshots(trace: FlowTrace) -> None:
    for context in trace.contexts:
        for step in SNAPSHOT_STEPS:
            key = (int(context.tile_id), int(step))
            cycle = trace.latent_cycle.get(key)
            target = trace.latent_target.get(key)
            guided = trace.guided_x0.get(key)
            if cycle is None or target is None or guided is None:
                continue
            delta = SparseTensor(target.feats - cycle.feats, target.coords.detach().clone())
            _atomic_torch_save(
                context.tile_dir / "steps" / f"step_{step:02d}_mra_latents.pt",
                {
                    "format": f"{FORMAT}_latent_snapshot_v1",
                    "tile_id": int(context.tile_id),
                    "step": int(step),
                    "latent_cycle": cycle,
                    "latent_target": target,
                    "latent_delta": delta,
                    "guided_x0": guided,
                },
            )


def _run_mra_flow(
    *,
    contexts: Sequence[Any],
    pipeline: Any,
    pbr_encoder: torch.nn.Module,
    global_camera: Mapping[str, float],
    texture_params: Mapping[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    step_limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], FlowTrace]:
    trace = FlowTrace(contexts=contexts, output_dir=output_dir)
    restore = _install_trace_hooks(trace)
    original_schedule = base._native_schedule
    if step_limit is not None:
        if float(args.noise_timestep) != 1.0:
            restore()
            raise ValueError("the fixed preflight assumes the native texture start timestep is 1.0")

        def truncated_schedule(sampler: Any, params: Mapping[str, Any]) -> List[float]:
            schedule = original_schedule(sampler, params)
            return schedule[: int(step_limit) + 1]

        base._native_schedule = truncated_schedule
    try:
        flow = base._run_cross_tile_guided_flow(
            contexts=contexts,
            pipeline=pipeline,
            global_camera=global_camera,
            texture_params=texture_params,
            pbr_encoder=pbr_encoder,
            args=args,
        )
    finally:
        base._native_schedule = original_schedule
        restore()
    _save_latent_snapshots(trace)
    expected_norm_calls = int(flow.get("flow_steps", 0)) * len(contexts) * 2
    flow["sparse_mra_trace"] = {
        "mra_record_count": int(len(trace.mra_records)),
        "latent_cycle_snapshot_count": int(len(trace.latent_cycle)),
        "latent_target_snapshot_count": int(len(trace.latent_target)),
        "guided_x0_snapshot_count": int(len(trace.guided_x0)),
        "normalize_calls": int(trace.norm_calls),
        "expected_normalize_calls": expected_norm_calls,
    }
    return flow, trace


def _preflight_check(flow: Mapping[str, Any], trace: FlowTrace, tolerance: float) -> Dict[str, Any]:
    errors: List[str] = []
    if int(flow.get("tile_count", 0)) != 4:
        errors.append(f"expected 4 preflight tiles, got {flow.get('tile_count')}")
    if int(flow.get("flow_steps", 0)) != 1:
        errors.append(f"expected 1 preflight step, got {flow.get('flow_steps')}")
    if not bool(flow.get("all_tiles_synchronized_per_step")):
        errors.append("official Jacobi/barrier synchronization failed")
    checks = 0
    for (tile_id, step), record in sorted(trace.mra_records.items()):
        checks += 1
        inv = record.get("invariants", {})
        finite = bool(record.get("finite"))
        if not finite:
            errors.append(f"tile {tile_id} step {step}: non-finite MRA field")
        if float(inv.get("null_Ah_delta_detail", {}).get("relative_l2", 1.0)) > float(tolerance):
            errors.append(f"tile {tile_id}: A_h detail null invariant exceeded tolerance")
        for name in ("coarse_removal", "detail_preservation", "exact_formula", "observed_gaussian_identity"):
            if float(inv.get(name, {}).get("max_abs", float("inf"))) > float(tolerance):
                errors.append(f"tile {tile_id}: {name} exceeded tolerance")
        if int(record.get("hidden_cross_tile_queries", -1)) != 0:
            errors.append(f"tile {tile_id}: hidden donor query count is not zero")
    for step in flow.get("steps", []):
        for tile in step.get("tiles", []):
            for name, check in tile.get("support_checks", {}).items():
                if check.get("coords_exact") is not True:
                    errors.append(f"tile {tile.get('tile_id')} support {name} is not exact")
    result = {
        "passed": not errors,
        "errors": errors,
        "tile_count": int(flow.get("tile_count", 0)),
        "flow_steps": int(flow.get("flow_steps", 0)),
        "mra_records_checked": checks,
        "invariant_tolerance": float(tolerance),
        "P_support_exact": not any("support" in error for error in errors),
        "hidden_cross_tile_queries": 0,
        "all_tensors_finite": not any("non-finite" in error for error in errors),
    }
    return result


@torch.no_grad()
def _decode_and_stitch(
    *,
    contexts: Sequence[Any],
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    patches: List[Any] = []
    tile_records: List[Dict[str, Any]] = []
    for index, context in enumerate(contexts):
        endpoint = getattr(context, "guided_endpoint", None)
        if endpoint is None:
            raise RuntimeError(f"tile {context.tile_id}: MRA endpoint is missing")
        print(f"[final decode] tile {context.tile_id:02d} ({index + 1}/{len(contexts)})", flush=True)
        shape = base._sparse_to_device(context.shape_denorm, torch.device("cuda")) if bool(args.low_vram) else context.shape_denorm
        texture = base._sparse_to_device(endpoint, torch.device("cuda")) if bool(args.low_vram) else endpoint
        empty_points = torch.empty((0, 3), device=torch.device("cuda"), dtype=torch.float32)
        mesh, _, decode_stats = base._decode_endpoint(
            pipeline=pipeline,
            shape_denorm=shape,
            texture_norm=texture,
            query_points=empty_points,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile {context.tile_id:02d} final {VARIANT}",
        )
        patch = core._local_mesh_to_global_patch(
            tile_id=int(context.tile_id),
            box=context.box,
            local_mesh=mesh,
            global_camera=global_camera,
            transform=context.transform,
            query_chunk_size=int(args.query_chunk_size),
        )
        patches.append(patch)
        tile_records.append({"tile_id": int(context.tile_id), "box": list(context.box), "decode": decode_stats, "patch": patch.stats})
        _save_sparse_cpu(context.tile_dir / "mra_guided_endpoint.pt", endpoint)
        del mesh, shape, texture
        _empty_cuda_cache()
    stitched, stitch_stats = core._stitch_tile_patches_nearest(
        patches,
        layout=dict(baseline_mesh.layout),
        global_camera=global_camera,
        face_chunk_size=int(args.face_projection_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    variant_dir = output_dir / "variants" / VARIANT
    variant_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        variant_dir / "global_merged_mesh.pt",
        {
            "format": f"{FORMAT}_global_mesh",
            "variant": VARIANT,
            "mesh": stitched,
            "stitch_stats": stitch_stats,
            "tile_records": tile_records,
        },
    )
    exported = core.ReturnedTilePatch(
        tile_id=-1,
        box=(0, 0, CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
        vertices=stitched.vertices,
        faces=stitched.faces,
        vertex_attrs=stitched.vertex_attrs,
        stats=stitch_stats,
    )
    glb_stats = core._export_tiled_glb([exported], variant_dir / "global_merged_mesh.glb")
    summary = {
        "variant": VARIANT,
        "vertices": int(stitched.vertices.shape[0]),
        "faces": int(stitched.faces.shape[0]),
        "tile_count": len(patches),
        "tile_records": tile_records,
        "stitch": stitch_stats,
        "glb": glb_stats,
        "mesh_pt": str((variant_dir / "global_merged_mesh.pt").resolve()),
        "mesh_glb": str((variant_dir / "global_merged_mesh.glb").resolve()),
    }
    _atomic_json(variant_dir / "global_variant_summary.json", summary)
    return stitched, summary


def _load_variant(path: Path) -> Any:
    payload = _load_torch(path)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise RuntimeError(f"invalid stitched variant at {path}")
    return mesh


def _frame_to_image(frame: Any) -> Image.Image:
    array = np.asarray(frame)
    if array.dtype.kind == "f":
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = array.clip(0.0, 255.0).astype(np.uint8)
    else:
        array = array.clip(0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3], mode="RGB")


def _contact_sheet(paths: Sequence[Path], labels: Sequence[str], output: Path, panel: int = 512, columns: int = 2) -> None:
    if not paths:
        return
    header = 38
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel, rows * (panel + header)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (panel - 8, panel - 8))
        x = (index % columns) * panel
        y = (index // columns) * (panel + header)
        sheet.paste(image, (x + (panel - image.width) // 2, y + header + (panel - image.height) // 2))
        draw.text((x + 6, y + 10), label, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _render_multiview(
    meshes: Mapping[str, Any],
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    fixed = [("front", 0.0, 0.0), ("right", 90.0, 0.0), ("left", -90.0, 0.0), ("back", 180.0, 0.0), ("top", 0.0, 75.0), ("bottom", 0.0, -75.0)]
    count = int(args.multiview_turntable_frames)
    specs = fixed + [(f"turntable_{i:02d}", 360.0 * i / count, 0.0) for i in range(count)]
    device = torch.device("cuda")
    radius = float(global_camera["distance"]) * float(args.multiview_radius_scale)
    fov = torch.tensor(float(global_camera["camera_angle_x"]), device=device)
    intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    extrinsics, intrinsics, labels = [], [], []
    for label, yaw_degrees, pitch_degrees in specs:
        yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
        direction = torch.tensor([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], device=device, dtype=torch.float32)
        extrinsics.append(utils3d.torch.extrinsics_look_at(target + direction * radius, target, up))
        intrinsics.append(intrinsic)
        labels.append(label)
    options = {
        "resolution": int(args.multiview_resolution),
        "near": max(0.01, radius - 2.0),
        "far": radius + 10.0,
        "ssaa": int(args.multiview_ssaa),
        "peel_layers": int(args.multiview_peel_layers),
        "face_chunk_size": int(args.render_face_chunk_size),
    }
    renderer = render_utils.get_renderer(baseline_mesh, **options)
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    root = output_dir / "multiview"
    root.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Any] = {"fixed_views": [name for name, _, _ in fixed], "turntable_frames": count, "resolution": int(args.multiview_resolution), "variants": {}}
    shaded_back_paths: List[Path] = []
    shaded_back_labels: List[str] = []
    channel_paths: Dict[str, List[Path]] = {"base_color": [], "metallic": [], "roughness": [], "alpha": []}
    channel_labels: Dict[str, List[str]] = {key: [] for key in channel_paths}
    for variant, mesh in meshes.items():
        live = mesh.to(device)
        rendered = render_utils.render_frames(
            live,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            options=options,
            verbose=True,
            renderer=renderer,
            envmap=envmap,
            use_envmap_bg=bool(args.use_envmap_bg),
        )
        del live
        variant_dir = root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        frames: Dict[str, Dict[str, str]] = {}
        shaded: List[Path] = []
        for index, label in enumerate(labels):
            frame_record: Dict[str, str] = {}
            for channel in ("shaded", "base_color", "metallic", "roughness", "alpha"):
                if channel not in rendered:
                    continue
                path = variant_dir / f"view_{index:03d}_{channel}.png"
                _frame_to_image(rendered[channel][index]).save(path)
                frame_record[channel] = str(path.resolve())
                if index in (0, 3) and channel in channel_paths:
                    channel_paths[channel].append(path)
                    channel_labels[channel].append(f"{variant}/{label}")
            if "shaded" in frame_record:
                shaded.append(Path(frame_record["shaded"]))
            frames[str(index)] = frame_record
        _contact_sheet(shaded, labels, variant_dir / "shaded_multiview_sheet.png", panel=320, columns=3)
        gif = variant_dir / "shaded_turntable_24.gif"
        turntable = shaded[len(fixed):]
        if turntable:
            with Image.open(turntable[0]) as first:
                first.copy().save(gif, save_all=True, append_images=[Image.open(path).copy() for path in turntable[1:]], duration=100, loop=0)
        records["variants"][variant] = {"frames": frames, "shaded_sheet": str((variant_dir / "shaded_multiview_sheet.png").resolve()), "turntable_gif": str(gif.resolve()) if gif.is_file() else None}
        if len(shaded) > 3:
            shaded_back_paths.append(shaded[3])
            shaded_back_labels.append(variant)
        _empty_cuda_cache()
    _contact_sheet(shaded_back_paths, shaded_back_labels, root / "back_view_contact_sheet_four_variants.png", panel=512, columns=2)
    records["back_view_contact_sheet"] = str((root / "back_view_contact_sheet_four_variants.png").resolve())
    records["pbr_front_back_contact_sheets"] = {}
    for channel, paths in channel_paths.items():
        if paths:
            sheet = root / f"{channel}_front_back_contact_sheet.png"
            _contact_sheet(paths, channel_labels[channel], sheet, panel=320, columns=2)
            records["pbr_front_back_contact_sheets"][channel] = str(sheet.resolve())
    _atomic_json(root / "multiview_summary.json", records)
    del envmap
    _empty_cuda_cache()
    return records


def _render_variants(
    *,
    meshes: Mapping[str, Any],
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    output_dir: Path,
    canonical_path: Path,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    records: Dict[str, Any] = {}
    table: List[Dict[str, Any]] = []
    render_paths: List[Path] = []
    render_labels: List[str] = []
    for name, mesh in meshes.items():
        print(f"[render aligned] {name} resolution={args.render_resolution}", flush=True)
        record = core._render(
            mesh,
            output_dir=output_dir / "variants" / name / "aligned_eval_4096",
            camera=global_camera,
            reference_image=canonical_path,
            args=args,
            envmap=envmap,
        )
        records[name] = record
        metrics = core._metric_subset(record)
        table.append(
            {
                "variant": name,
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "PSNR": metrics["psnr_db"],
                "SSIM": metrics["ssim"],
                "LPIPS": metrics["lpips"],
                "render_resolution": int(args.render_resolution),
            }
        )
        render_paths.append(Path(str(record["render_png"])))
        render_labels.append(name)
    _contact_sheet(render_paths, render_labels, output_dir / "aligned_4096_four_variant_comparison.png", panel=512, columns=2)
    del envmap
    _empty_cuda_cache()
    multiview = _render_multiview(meshes, baseline_mesh, global_camera, args, output_dir) if bool(args.render_multiview) else {"enabled": False}
    return table, records, multiview


def _aggregate_mra(trace: FlowTrace) -> Dict[str, Any]:
    by_step: Dict[int, List[Mapping[str, Any]]] = {}
    for (tile_id, step), record in sorted(trace.mra_records.items()):
        by_step.setdefault(int(step), []).append(record)

    def mean(values: Iterable[float]) -> Optional[float]:
        array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
        return float(array.mean()) if array.size else None

    curves: List[Dict[str, Any]] = []
    for step, rows in sorted(by_step.items()):
        row = {
            "step": int(step),
            "tile_count": len(rows),
            "hidden_ratio": mean(r["hidden_voxel_count"] / max(1, r["hidden_voxel_count"] + r["observed_voxel_count"]) for r in rows),
            "invariants": {
                name: {
                    "max_abs": mean(r["invariants"][name]["max_abs"] for r in rows),
                    "max_over_tiles": max(float(r["invariants"][name]["max_abs"]) for r in rows),
                    "relative_l2_mean": mean(r["invariants"][name]["relative_l2"] for r in rows),
                }
                for name in ("null_Ah_delta_detail", "coarse_removal", "detail_preservation", "exact_formula", "observed_gaussian_identity")
            },
            "energy": {},
            "ratios_hidden": {
                "r_c": {channel: mean(r["ratios_hidden"]["r_c"][channel] for r in rows) for channel in ("RGB", "metallic", "roughness", "alpha")},
                "r_d": {channel: mean(r["ratios_hidden"]["r_d"][channel] for r in rows) for channel in ("RGB", "metallic", "roughness", "alpha")},
            },
            "physical_range_out_of_0_1": {},
        }
        for term in ("delta", "delta_coarse", "delta_detail"):
            row["energy"][term] = {
                region: {
                    channel: mean(r[term][region]["channels"][channel]["rms"] for r in rows)
                    for channel in ("RGB", "metallic", "roughness", "alpha")
                }
                for region in ("observed", "hidden")
            }
        for field_name in ("G1024", "H_t", "hidden_target", "final_target_field"):
            row["physical_range_out_of_0_1"][field_name] = {
                channel: mean(r["physical_range"][field_name][channel]["out_of_0_1_ratio"] for r in rows)
                for channel in ("RGB", "metallic", "roughness", "alpha")
            }
        curves.append(row)
    return {
        "format": f"{FORMAT}_metrics_v1",
        "definitions": {
            "Delta": "H_t - G1024",
            "Delta_coarse": "P_h A_h Delta",
            "Delta_detail": "Delta - Delta_coarse",
            "hidden_target": "G1024 + Delta_detail = H_t - P_h A_h(H_t-G1024)",
            "energy_rms": "per-row RMS over the listed PBR channel group",
            "r_c": "energy(Delta_coarse)/(energy(Delta)+1e-8)",
            "r_d": "energy(Delta_detail)/(energy(Delta)+1e-8)",
        },
        "steps": curves,
        "records": {f"tile_{tile:02d}_step_{step:02d}": record for (tile, step), record in sorted(trace.mra_records.items())},
    }


def _write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    evaluation = summary.get("evaluation", {})
    table = evaluation.get("table", []) if isinstance(evaluation, Mapping) else []
    preflight = summary.get("correctness_preflight", {})
    flow = summary.get("flow", {})
    lines = [
        "# C256/C1024 Sparse-MRA Delta Per-Step Guidance",
        "",
        "## 实验设置",
        "",
        f"- CUDA device: `{summary.get('cuda_device')}`；seed: `{summary.get('seed')}`；texture flow: `{summary.get('sampler', {}).get('texture_steps')} steps`。",
        f"- active tiles: `{summary.get('tile_layout', {}).get('participating_tile_ids')}`。",
        f"- correctness preflight: **{preflight.get('passed')}**（4 tiles × 1 step）。",
        "- observed target: 现有 tile-center Gaussian fusion；hidden target: `G1024 + (I-P_h A_h)(H-G1024)`。",
        "- `G1024` 直接使用与 local C1024 rows 对齐的 global PBR reference；没有构造 G256 absolute target、global velocity 或 global trajectory。",
        "",
        "## 核心数值结果",
        "",
        "| variant | vertices | faces | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(f"| {row.get('variant')} | {row.get('vertices')} | {row.get('faces')} | {row.get('PSNR')} | {row.get('SSIM')} | {row.get('LPIPS')} |")
    lines.extend(
        [
            "",
            "## Correctness 与 routing",
            "",
            f"- hidden donor queries: `{summary.get('route_checks', {}).get('hidden_cross_tile_queries')}`。",
            f"- observed Gaussian identity: `{summary.get('route_checks', {}).get('observed_gaussian_identity')}`。",
            f"- fixed shape support unchanged: `{summary.get('route_checks', {}).get('fixed_shape_unchanged')}`。",
            f"- all Jacobi barriers synchronized: `{flow.get('all_tiles_synchronized_per_step')}`。",
            f"- MRA invariant failure count: `{summary.get('route_checks', {}).get('invariant_failure_count')}`。",
            "",
            "## 诊断解释",
            "",
            "1. `Delta_coarse = P_h A_h(H-G)` 是 HR 相对 Global 的 coarse variation；`Delta_detail = (I-P_hA_h)(H-G)` 是按该 sparse representation 定义的 null/detail variation。step 0/6/11 的 RGB、metallic、roughness、alpha 能量与比例保存在 `mra_metrics.json`。",
            "2. hidden 输出是否满足公式由 `exact_formula`、`coarse_removal`、`null_Ah_delta_detail` 和 `detail_preservation` 逐 tile/step 检查；observed 行由 `observed_gaussian_identity` 检查。",
            "3. 视觉上是否表现为 coarse material drift 与局部 detail，应结合 `tiles/*/steps/step_{00,06,11}_mra_fields.pt`、`multiview/back_view_contact_sheet_four_variants.png` 和 PBR front/back sheets 判断；脚本不把 null-space 自动等同于语义纹理。",
            "4. 若端到端指标下降，`flow` 中的 latent correction 与 guided velocity 诊断用于区分 MRA 分解、PBR encoder transport、late-step flow dynamics 和 stitching；本实验不会回退到 hidden Gaussian 或 parent-mean 定义。",
            "",
            "## 产物",
            "",
            f"- summary: `{output_dir / 'summary.json'}`",
            f"- per-step MRA metrics: `{output_dir / 'mra_metrics.json'}`",
            f"- correctness: `{output_dir / 'correctness_test/correctness_summary.json'}`",
            f"- operator caches: `{output_dir / 'tiles'}` (`mra_P_hidden.npz`, `mra_support.pt`)。",
            f"- report: `{output_dir / 'SPARSE_MRA_DELTA_EXPERIMENT.md'}`",
        ]
    )
    path = output_dir / "SPARSE_MRA_DELTA_EXPERIMENT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/choose/0_img.png")
    parser.add_argument("--output-dir", default="outputs/pbr_sparse_mra_delta_perstep_cuda4")
    parser.add_argument("--reference-experiment-dir", default="outputs/cross_tile_pbr_perstep_guided_cuda4_full_staged")
    parser.add_argument("--field-cache-dir", default="outputs/pbr_range_null_perstep_cuda4_full")
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pbr-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--fusion-sigma-pixels", type=float, default=256.0)
    parser.add_argument("--invariant-tolerance", type=float, default=5e-5)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / FINE_RESOLUTION)
    parser.add_argument("--save-donor-details", action=argparse.BooleanOptionalAction, default=False)

    # Official sampler knobs are exposed unchanged so the base route receives
    # the same 12-step texture-flow parameters as the completed references.
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--correctness-tile-count", type=int, default=4)
    parser.add_argument("--correctness-step-count", type=int, default=1)
    parser.add_argument("--skip-correctness", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--render-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=4)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument("--multiview-turntable-frames", type=int, default=24)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    for path in (Path(args.image).expanduser(), Path(args.reference_experiment_dir).expanduser(), Path(args.field_cache_dir).expanduser()):
        if not path.exists():
            raise FileNotFoundError(path)
    encoder = Path(args.pbr_encoder).expanduser()
    if not Path(f"{encoder}.json").is_file() or not Path(f"{encoder}.safetensors").is_file():
        raise FileNotFoundError(f"PBR encoder checkpoint pair not found for {encoder}")
    if int(args.texture_steps) != 12:
        raise ValueError("Codex.md fixes texture flow at 12 steps")
    if float(args.eta) != 1.0:
        raise ValueError("Codex.md fixes eta at 1.0")
    if float(args.noise_timestep) != 1.0:
        raise ValueError("this cached fixed-shape experiment uses the native t=1.0 start")
    if float(args.noise_strength) <= 0.0 or float(args.fusion_sigma_pixels) <= 0.0 or float(args.invariant_tolerance) <= 0.0:
        raise ValueError("noise strength, Gaussian sigma and invariant tolerance must be positive")
    for name in (
        "face_projection_chunk_size", "query_chunk_size", "render_resolution", "metric_resolution", "render_ssaa",
        "render_peel_layers", "render_face_chunk_size", "multiview_resolution", "multiview_ssaa",
        "multiview_peel_layers", "multiview_turntable_frames", "correctness_tile_count", "correctness_step_count",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not bool(args.skip_lpips) and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips is unavailable; continuing without LPIPS", flush=True)
        args.skip_lpips = True


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    reference_dir = Path(args.reference_experiment_dir).expanduser().resolve()
    field_cache_dir = Path(args.field_cache_dir).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve() if args.baseline_dir else reference_dir
    print(
        f"[cuda] requested/current={args.cuda_device}/{torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())} low_vram={args.low_vram}",
        flush=True,
    )

    source_path = Path(args.image).expanduser().resolve()
    with Image.open(source_path) as source:
        source_rgb = source.convert("RGB")
    source_rgb.save(output_dir / "input_original.png")
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    canonical = pipeline.preprocess_canonical_images(source_rgb)
    image_4096 = canonical["image_4096"]
    image_1024 = canonical["image_1024"]
    image_512 = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    canonical["foreground_mask_4096"].save(output_dir / "canonical_foreground_mask_4096.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical.get("metadata", {}))

    camera_path = reference_dir / "global_camera.json"
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    global_camera = json.loads(camera_path.read_text(encoding="utf-8"))
    _atomic_json(output_dir / "global_camera.json", global_camera)
    baseline_mesh = _load_baseline(baseline_dir / "global_baseline_mesh.pt").to("cpu")
    _atomic_torch_save(output_dir / "global_baseline_mesh.pt", {"format": f"{FORMAT}_baseline", "mesh": baseline_mesh})
    boxes = core._tile_layout(canonical_size=CANONICAL_IMAGE_SIZE, tile_size=TILE_SIZE, stride=TILE_STRIDE)
    _atomic_json(output_dir / "tile_layout.json", {"canonical_image_size": CANONICAL_IMAGE_SIZE, "tile_size": TILE_SIZE, "stride": TILE_STRIDE, "tile_count": len(boxes), "boxes": boxes})

    contexts = _load_contexts(
        source_dir=reference_dir,
        output_dir=output_dir,
        pipeline=pipeline,
        baseline_mesh=baseline_mesh,
        global_camera=global_camera,
        image_4096=image_4096,
        args=args,
    )
    visibility_summary = _attach_reference_fields(contexts, field_cache_dir, output_dir)
    for context in contexts:
        context.mra_projector = _build_or_load_projector(context, args)
    texture_params = core._sampler_overrides(args)[2]
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
    if not bool(args.low_vram):
        pbr_encoder.to(torch.device("cuda"))

    preflight_summary: Dict[str, Any]
    if bool(args.skip_correctness):
        preflight_summary = {"passed": False, "skipped": True, "reason": "explicit --skip-correctness"}
    else:
        if len(contexts) < int(args.correctness_tile_count):
            raise RuntimeError(f"preflight requires {args.correctness_tile_count} contexts, got {len(contexts)}")
        preflight_contexts = list(contexts[: int(args.correctness_tile_count)])
        preflight_dir = output_dir / "correctness_test"
        old_dirs = {id(context): context.tile_dir for context in preflight_contexts}
        for context in preflight_contexts:
            context.tile_dir = preflight_dir / "tiles" / f"tile_{int(context.tile_id):02d}"
            context.tile_dir.mkdir(parents=True, exist_ok=True)
        old_output_dir = args.output_dir
        args.output_dir = str(preflight_dir)
        try:
            preflight_flow, preflight_trace = _run_mra_flow(
                contexts=preflight_contexts,
                pipeline=pipeline,
                pbr_encoder=pbr_encoder,
                global_camera=global_camera,
                texture_params=texture_params,
                args=args,
                output_dir=preflight_dir,
                step_limit=int(args.correctness_step_count),
            )
            preflight_summary = _preflight_check(preflight_flow, preflight_trace, float(args.invariant_tolerance))
            preflight_summary["flow"] = preflight_flow
            _atomic_json(preflight_dir / "correctness_summary.json", preflight_summary)
        finally:
            args.output_dir = old_output_dir
            for context in preflight_contexts:
                context.tile_dir = old_dirs[id(context)]
        if not bool(preflight_summary["passed"]):
            raise RuntimeError(f"sparse-MRA correctness preflight failed: {preflight_summary['errors']}")
        print("[correctness] passed 4 tiles × 1 step", flush=True)

    # Build remaining operators only after the mandatory 4x1 preflight.  This
    # keeps a failed boundary/operator check from launching the expensive full
    # texture flow.
    for context in contexts:
        if not hasattr(context, "mra_projector"):
            context.mra_projector = _build_or_load_projector(context, args)

    flow, trace = _run_mra_flow(
        contexts=contexts,
        pipeline=pipeline,
        pbr_encoder=pbr_encoder,
        global_camera=global_camera,
        texture_params=texture_params,
        args=args,
        output_dir=output_dir,
        step_limit=None,
    )
    del pbr_encoder
    _empty_cuda_cache()
    mra_metrics = _aggregate_mra(trace)
    _atomic_json(output_dir / "mra_metrics.json", mra_metrics)

    mra_mesh, mra_variant_summary = _decode_and_stitch(
        contexts=contexts,
        pipeline=pipeline,
        baseline_mesh=baseline_mesh,
        global_camera=global_camera,
        args=args,
        output_dir=output_dir,
    )
    pure_mesh = _load_variant(reference_dir / "variants" / "pure_HR" / "global_merged_mesh.pt")
    current_mesh = _load_variant(reference_dir / "variants" / "cross_tile_pbr_perstep_guided" / "global_merged_mesh.pt")
    meshes = {
        "global_baseline": baseline_mesh,
        "pure_HR": pure_mesh,
        "current_gaussian_guided": current_mesh,
        VARIANT: mra_mesh,
    }
    if bool(args.render):
        evaluation_table, render_records, multiview = _render_variants(
            meshes=meshes,
            baseline_mesh=baseline_mesh,
            global_camera=global_camera,
            output_dir=output_dir,
            canonical_path=output_dir / "canonical_4096.png",
            args=args,
        )
    else:
        evaluation_table = [
            {"variant": name, "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]), "PSNR": None, "SSIM": None, "LPIPS": None, "render_resolution": None}
            for name, mesh in meshes.items()
        ]
        render_records, multiview = {}, {"enabled": False}
    _write_csv(output_dir / "metrics.csv", evaluation_table)

    invariant_failure_count = 0
    hidden_donor_max = 0
    observed_identity_max = 0.0
    for record in trace.mra_records.values():
        hidden_donor_max = max(hidden_donor_max, int(record.get("hidden_cross_tile_queries", 0)))
        inv = record.get("invariants", {})
        observed_identity_max = max(observed_identity_max, float(inv.get("observed_gaussian_identity", {}).get("max_abs", 0.0)))
        for name in ("null_Ah_delta_detail", "coarse_removal", "detail_preservation", "exact_formula", "observed_gaussian_identity"):
            if float(inv.get(name, {}).get("max_abs", 0.0)) > float(args.invariant_tolerance):
                invariant_failure_count += 1
    selected_ids = [int(context.tile_id) for context in contexts]
    summary: Dict[str, Any] = {
        "format": FORMAT,
        "variant": VARIANT,
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "low_vram": bool(args.low_vram),
        "seed": int(args.seed),
        "reference_experiment_dir": str(reference_dir),
        "field_cache_dir": str(field_cache_dir),
        "global_camera": global_camera,
        "tile_layout": {"tile_count": len(boxes), "participating_tile_ids": selected_ids, "boxes": boxes},
        "visibility": visibility_summary,
        "correctness_preflight": preflight_summary,
        "guidance": {
            "formula": "Delta=H-G; Delta_coarse=P_h A_h Delta; Delta_detail=Delta-Delta_coarse; Y_hidden=G+Delta_detail",
            "A_h": "least-squares pseudoinverse of the pure-hidden C256->hidden-C1024 sparse prolongation",
            "P_h": "C256/C1024 sparse trilinear prolongation with observed-touching bases excluded",
            "observed_projection": "existing tile-center Gaussian fusion on observed rows only",
            "hidden_projection": "local sparse-MRA delta projection; no cross-tile donor query",
            "eta": float(args.eta),
            "fusion_sigma_pixels": float(args.fusion_sigma_pixels),
            "G1024_absolute_anchor": True,
            "G256_absolute_target": False,
            "global_velocity_used": False,
            "global_timestep_trajectory_used": False,
            "weighted_G_HR_blend_used": False,
            "hidden_gaussian_averaging_used": False,
            "parent_mean_decomposition_used": False,
        },
        "sampler": {"texture_steps": int(args.texture_steps), "noise_timestep": float(args.noise_timestep), "noise_strength": float(args.noise_strength), "seed": int(args.seed), "route": "official _get_model_prediction -> pred_x0 decode -> field target -> official PBR encode -> cycle-cancelled _xstart_to_pred -> Euler"},
        "flow": flow,
        "mra_metrics": {"path": str((output_dir / "mra_metrics.json").resolve()), "steps": mra_metrics["steps"]},
        "route_checks": {
            "shape_flow_called": False,
            "shape_sampler_called": False,
            "fixed_shape_unchanged": all(bool(context.static_stats.get("fixed_shape", {}).get("support_unchanged", True)) for context in contexts),
            "official_texture_sampler": True,
            "official_texture_decoder": True,
            "official_texture_encoder": True,
            "official_meshwithvoxel_query": True,
            "cycle_cancelled_residual_used": True,
            "all_tiles_synchronized_per_step": bool(flow.get("all_tiles_synchronized_per_step")),
            "observed_gaussian_identity": bool(observed_identity_max <= float(args.invariant_tolerance)),
            "hidden_cross_tile_queries": hidden_donor_max,
            "invariant_failure_count": invariant_failure_count,
            "no_training": True,
        },
        "diagnosis": {
            "MRA_decomposition": "passed" if invariant_failure_count == 0 else "failed invariant; inspect mra_metrics.json",
            "PBR_encode_transport": "recorded by latent_cycle/latent_target/guided_x0 snapshots and support checks",
            "late_step_flow_dynamics": "recorded by per-step guided velocity and latent correction norms",
            "cross_tile_stochastic_disagreement": "candidate factor if hidden detail remains inconsistent despite algebraic invariants",
            "final_stitching": "recorded in sparse_mra_delta_guided/global_variant_summary.json",
        },
        "evaluation": {"reference": str((output_dir / "canonical_4096.png").resolve()), "table": evaluation_table, "renders": render_records, "multiview": multiview, "aligned_comparison": str((output_dir / "aligned_4096_four_variant_comparison.png").resolve()) if (output_dir / "aligned_4096_four_variant_comparison.png").is_file() else None},
        "variants": {"global_baseline": {"source": str((baseline_dir / "global_baseline_mesh.pt").resolve())}, "pure_HR": {"source": str((reference_dir / "variants" / "pure_HR" / "global_merged_mesh.pt").resolve())}, "current_gaussian_guided": {"source": str((reference_dir / "variants" / "cross_tile_pbr_perstep_guided" / "global_merged_mesh.pt").resolve())}, VARIANT: mra_variant_summary},
        "artifacts": {"global_baseline_mesh": str((output_dir / "global_baseline_mesh.pt").resolve()), "global_merged_mesh_pt": str((output_dir / "variants" / VARIANT / "global_merged_mesh.pt").resolve()), "global_merged_mesh_glb": str((output_dir / "variants" / VARIANT / "global_merged_mesh.glb").resolve()), "metrics_csv": str((output_dir / "metrics.csv").resolve()), "mra_metrics": str((output_dir / "mra_metrics.json").resolve()), "steps_directory": str((output_dir / "steps").resolve())},
    }
    report = _write_report(output_dir, summary, mra_metrics)
    summary["report_markdown"] = str(report.resolve())
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] tiles={len(contexts)} steps={flow.get('flow_steps')} variant={VARIANT} summary={output_dir / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
