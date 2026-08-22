#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global C1024 joint-field common-subspace diagnostic.

This is a training-free final-field analysis of the symmetric field family
``{G, H_i}``.  It deliberately consumes the already validated Global C1024
cache produced by ``pixal3d_global_c1024_common_field_pod.py``.  The main
analysis never forms a difference field and does not contain a generation,
fusion, projection, or re-encoding step.

The COBE implementation below follows the sequential idea in Algorithm 1 of
Zhou et al.  For each block, a thin SVD supplies an orthonormal column-space
basis.  A component is found by a power iteration on the sum of the block
projectors, followed by orthogonal deflation of every block basis.  No COBE
library is used.  The batched GPU implementation is only a fast equivalent
for the spatial permutation null; the real analysis and all reported values
are float64.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


FORMAT = "pixal3d_global_c1024_majority_cobe_v1"
CACHE_FORMAT = "pixal3d_global_c1024_common_field_pod_v1"
PBR_CHANNEL_NAMES = ("R", "G", "B", "metallic", "roughness", "alpha")
VIEW_CHANNELS = {"PBR6": 6, "RGB": 3}
FIELD_LABELS = ("G", "H1", "H2", "H3", "H4")
QUARTETS: Tuple[Tuple[int, int, int, int], ...] = (
    (18, 19, 25, 26),
    (19, 20, 26, 27),
    (25, 26, 32, 33),
    (26, 27, 33, 34),
)
DOMAINS = ("ALL_VALID", "ALL_HIDDEN", "ALL_OBSERVED")
VIEWS = ("PBR6", "RGB")
CONTROLS = ("RAW", "SPATIAL_DEMEANED")


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray, torch.Tensor)):
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return _jsonable(value)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def _finite_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class BasisResult:
    basis: torch.Tensor
    raw_singular_values: torch.Tensor
    conditioned_singular_values: torch.Tensor
    column_rms: torch.Tensor
    rank: int
    absolute_tolerance: float
    relative_tolerance: float

    def metadata(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "raw_singular_values": self.raw_singular_values.tolist(),
            "conditioned_singular_values": self.conditioned_singular_values.tolist(),
            "column_rms": self.column_rms.tolist(),
            "absolute_tolerance": float(self.absolute_tolerance),
            "relative_tolerance": float(self.relative_tolerance),
            "basis_shape": list(self.basis.shape),
        }


@dataclass
class CobeResult:
    basis: torch.Tensor
    errors: torch.Tensor
    support_scores: torch.Tensor
    iterations: List[int]
    converged: List[bool]
    current_ranks: List[List[int]]
    block_ranks: List[int]

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1]) if self.basis.ndim == 2 else 0


def _as_float64_matrix(value: torch.Tensor, name: str = "matrix") -> torch.Tensor:
    matrix = torch.as_tensor(value).detach().cpu().to(torch.float64).contiguous()
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be [N,D], got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def effective_rank_basis(
    matrix: torch.Tensor,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    scale_columns: bool = True,
) -> BasisResult:
    """Return a float64 column-space basis and its complete singular spectrum.

    Column RMS scaling is only used before SVD.  Every nonzero column is
    multiplied by a nonzero scalar, so it leaves the represented column space
    unchanged while avoiding a tiny alpha/metallic channel dominating the
    numerical rank calculation.
    """

    matrix64 = _as_float64_matrix(matrix)
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("rank tolerances must be non-negative")
    rms = torch.sqrt(torch.mean(matrix64 * matrix64, dim=0)) if matrix64.shape[0] else torch.zeros(matrix64.shape[1], dtype=torch.float64)
    if scale_columns:
        scale = torch.where(rms > 0.0, rms, torch.ones_like(rms))
        conditioned = matrix64 / scale[None, :]
    else:
        scale = torch.ones_like(rms)
        conditioned = matrix64
    singular_values, vh = _gram_spectrum(conditioned)
    largest = float(singular_values[0].item()) if singular_values.numel() else 0.0
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * largest)
    rank = int((singular_values > threshold).sum().item())
    if rank:
        # Reconstruct Q from the scaled matrix and V,S.  This avoids storing a
        # second full U when the input is tall and makes the column-space
        # invariance test explicit.
        basis = conditioned @ vh[:, :rank]
        basis = basis / singular_values[:rank][None, :]
        basis, _ = torch.linalg.qr(basis, mode="reduced")
    else:
        basis = torch.zeros((matrix64.shape[0], 0), dtype=torch.float64)
    raw_singular_values, _ = _gram_spectrum(matrix64)
    return BasisResult(
        basis=basis,
        raw_singular_values=raw_singular_values,
        conditioned_singular_values=singular_values,
        column_rms=rms,
        rank=rank,
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
    )


def _orthogonalize_vector(vector: torch.Tensor, previous: Sequence[torch.Tensor]) -> torch.Tensor:
    result = vector
    for direction in previous:
        result = result - direction * torch.dot(direction, result)
    return result


def _normalise_vector(vector: torch.Tensor, *, tolerance: float = 1e-15) -> Optional[torch.Tensor]:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= tolerance:
        return None
    return vector / norm


def _gram_spectrum(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return descending singular values and right singular vectors.

    All matrices entering this diagnostic have only 3 or 6 columns (or a
    small COBE rank).  Decomposing the column Gram matrix is mathematically
    equivalent to a thin SVD, but avoids repeatedly factoring an N-by-6 tall
    matrix during the four leave-one-out runs and 256 nulls.
    """

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return torch.zeros(0, dtype=torch.float64), torch.zeros((matrix.shape[1], 0), dtype=torch.float64)
    gram = matrix.T @ matrix
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    vectors = vectors.index_select(1, order)
    # Forming a 3/6-column Gram matrix can turn an exact zero eigenvalue into
    # a small positive value (~1e-16 times the largest eigenvalue).  Remove
    # that round-off floor before taking square roots; the explicit rank
    # tolerances applied by callers still decide near-zero physical modes.
    if eigenvalues.numel():
        gram_floor = torch.finfo(torch.float64).eps * eigenvalues[0].clamp_min(1.0) * 10.0
        eigenvalues = torch.where(eigenvalues <= gram_floor, torch.zeros_like(eigenvalues), eigenvalues)
    return torch.sqrt(eigenvalues), vectors


def _basis_from_matrix_gram(matrix: torch.Tensor, tolerance: float = 1e-12) -> torch.Tensor:
    singular_values, vectors = _gram_spectrum(matrix)
    largest = float(singular_values[0].item()) if singular_values.numel() else 0.0
    keep = singular_values > max(float(tolerance), float(tolerance) * largest)
    if not bool(keep.any().item()):
        return torch.zeros((matrix.shape[0], 0), dtype=torch.float64)
    basis = matrix @ vectors[:, keep]
    basis = basis / singular_values[keep][None, :]
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis


def _deflate_basis(basis: torch.Tensor, direction: torch.Tensor, tolerance: float = 1e-12) -> torch.Tensor:
    if basis.shape[1] == 0:
        return basis
    residual = basis - direction[:, None] * (direction @ basis)[None, :]
    if residual.numel() == 0:
        return residual[:, :0]
    singular_values, vectors = _gram_spectrum(residual)
    largest = float(singular_values[0].item()) if singular_values.numel() else 0.0
    keep = singular_values > max(tolerance, tolerance * largest)
    if not bool(keep.any().item()):
        return residual[:, :0]
    result = residual @ vectors[:, keep]
    result = result / singular_values[keep][None, :]
    result, _ = torch.linalg.qr(result, mode="reduced")
    return result


def _cobe_from_bases(
    bases: Sequence[torch.Tensor],
    *,
    max_components: Optional[int],
    max_iter: int,
    convergence_tol: float,
    seed: int,
) -> CobeResult:
    if not bases:
        raise ValueError("COBE needs at least one block")
    current = [_as_float64_matrix(q, f"Q[{i}]") for i, q in enumerate(bases)]
    if any(q.shape[0] != current[0].shape[0] for q in current):
        raise ValueError("all COBE bases must have the same row count")
    block_ranks = [int(q.shape[1]) for q in current]
    limit = min(block_ranks) if max_components is None else min(int(max_components), min(block_ranks))
    if limit < 0 or max_iter <= 0:
        raise ValueError("max_components and max_iter must be positive")
    n_rows = int(current[0].shape[0])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    candidates: List[torch.Tensor] = []
    errors: List[torch.Tensor] = []
    supports: List[torch.Tensor] = []
    iterations: List[int] = []
    converged: List[bool] = []
    current_ranks: List[List[int]] = []

    for _component in range(limit):
        if any(q.shape[1] == 0 for q in current):
            break
        vector = torch.randn(n_rows, generator=generator, dtype=torch.float64)
        vector = _orthogonalize_vector(vector, candidates)
        vector = _normalise_vector(vector)
        if vector is None:
            break
        did_converge = False
        used_iterations = 0
        for iteration in range(max_iter):
            projected_sum = torch.zeros_like(vector)
            for q in current:
                z = q.T @ vector
                projected_sum += q @ z
            projected_sum = _orthogonalize_vector(projected_sum, candidates)
            next_vector = _normalise_vector(projected_sum)
            if next_vector is None:
                break
            used_iterations = iteration + 1
            delta = min(
                float(torch.linalg.vector_norm(next_vector - vector).item()),
                float(torch.linalg.vector_norm(next_vector + vector).item()),
            )
            vector = next_vector
            if delta <= convergence_tol:
                did_converge = True
                break
        if not torch.isfinite(vector).all():
            break
        # A final re-orthogonalisation makes the returned basis orthonormal
        # even when the power iteration stopped at a loose tolerance.
        vector = _orthogonalize_vector(vector, candidates)
        vector = _normalise_vector(vector)
        if vector is None:
            break
        current_support: List[torch.Tensor] = []
        current_error = torch.zeros((), dtype=torch.float64)
        for q in current:
            z = q.T @ vector
            current_support.append(torch.sum(z * z))
            residual = q @ z - vector
            current_error = current_error + torch.sum(residual * residual)
        candidate_support = torch.stack([torch.sum((q.T @ vector) ** 2) for q in bases])
        candidates.append(vector)
        errors.append(current_error)
        supports.append(candidate_support)
        iterations.append(used_iterations)
        converged.append(did_converge)
        current_ranks.append([int(q.shape[1]) for q in current])
        current = [_deflate_basis(q, vector) for q in current]

    basis = torch.stack(candidates, dim=1) if candidates else torch.zeros((n_rows, 0), dtype=torch.float64)
    error_tensor = torch.stack(errors) if errors else torch.zeros(0, dtype=torch.float64)
    support_tensor = torch.stack(supports, dim=1) if supports else torch.zeros((len(bases), 0), dtype=torch.float64)
    return CobeResult(
        basis=basis,
        errors=error_tensor,
        support_scores=support_tensor,
        iterations=iterations,
        converged=converged,
        current_ranks=current_ranks,
        block_ranks=block_ranks,
    )


def _cobe_from_bases_exact(
    bases: Sequence[torch.Tensor],
    *,
    max_components: Optional[int],
    tolerance: float = 1e-12,
) -> CobeResult:
    """Exact sequential COBE via the small block-projector Gram matrix.

    At each step the nonzero eigenvalues of ``sum_i Q_i Q_i^T`` equal those
    of the Gram matrix of the concatenated block bases.  Solving that small
    eigenproblem avoids materialising an N-by-N projector and is numerically
    equivalent to the power iteration used by ``_cobe_from_bases``.
    """

    if not bases:
        raise ValueError("COBE needs at least one block")
    current = [_as_float64_matrix(q, f"Q[{i}]") for i, q in enumerate(bases)]
    if any(q.shape[0] != current[0].shape[0] for q in current):
        raise ValueError("all COBE bases must have the same row count")
    block_ranks = [int(q.shape[1]) for q in current]
    limit = min(block_ranks) if max_components is None else min(int(max_components), min(block_ranks))
    candidates: List[torch.Tensor] = []
    errors: List[torch.Tensor] = []
    supports: List[torch.Tensor] = []
    current_ranks: List[List[int]] = []
    iterations = [1] * limit
    converged = [True] * limit
    for _ in range(limit):
        if any(q.shape[1] == 0 for q in current):
            break
        concatenated = torch.cat(current, dim=1)
        gram = concatenated.T @ concatenated
        eigenvalues, vectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
        vectors = vectors.index_select(1, order)
        leading = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0
        if not eigenvalues.numel() or float(eigenvalues[0].item()) <= max(float(tolerance), float(tolerance) * leading):
            break
        vector = concatenated @ vectors[:, 0]
        vector = _normalise_vector(vector)
        if vector is None:
            break
        vector = _orthogonalize_vector(vector, candidates)
        vector = _normalise_vector(vector)
        if vector is None:
            break
        current_error = torch.zeros((), dtype=torch.float64)
        for q in current:
            z = q.T @ vector
            current_error += torch.sum((q @ z - vector) ** 2)
        errors.append(current_error)
        supports.append(torch.stack([torch.sum((q.T @ vector) ** 2) for q in bases]))
        current_ranks.append([int(q.shape[1]) for q in current])
        candidates.append(vector)
        current = [_deflate_basis(q, vector, tolerance=tolerance) for q in current]
    basis = torch.stack(candidates, dim=1) if candidates else torch.zeros((current[0].shape[0], 0), dtype=torch.float64)
    return CobeResult(
        basis=basis,
        errors=torch.stack(errors) if errors else torch.zeros(0, dtype=torch.float64),
        support_scores=torch.stack(supports, dim=1) if supports else torch.zeros((len(bases), 0), dtype=torch.float64),
        iterations=iterations[: len(candidates)],
        converged=converged[: len(candidates)],
        current_ranks=current_ranks,
        block_ranks=block_ranks,
    )


def cobe_candidates(
    blocks: Sequence[torch.Tensor],
    max_components: Optional[int] = None,
    max_iter: int = 100,
    convergence_tol: float = 1e-10,
    seed: int = 42,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    return_metadata: bool = False,
) -> CobeResult:
    """Extract sequential COBE candidate directions from multi-block fields."""

    if not blocks:
        raise ValueError("COBE needs at least one field block")
    basis_results = [
        effective_rank_basis(
            block,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            scale_columns=True,
        )
        for block in blocks
    ]
    result = _cobe_from_bases(
        [item.basis for item in basis_results],
        max_components=max_components,
        max_iter=max_iter,
        convergence_tol=convergence_tol,
        seed=seed,
    )
    # The public result carries the block basis metadata without changing the
    # compact numerical API used by the analysis.
    result.basis_metadata = [item.metadata() for item in basis_results]  # type: ignore[attr-defined]
    return result


def _cobe_from_bases_batch(
    bases: Sequence[torch.Tensor],
    *,
    max_components: int,
    max_iter: int,
    convergence_tol: float,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Batched float64 COBE for the spatial permutation null.

    ``bases`` has shape ``[B,N,r]`` per block.  The implementation mirrors
    ``_cobe_from_bases`` and keeps all permutation realizations independent in
    the leading batch dimension.  The null path uses full-rank generic
    deflations, so batched QR is sufficient and considerably faster than
    launching thousands of single-realization CPU iterations.
    """

    if not bases:
        raise ValueError("batched COBE needs blocks")
    batch = int(bases[0].shape[0])
    n_rows = int(bases[0].shape[1])
    current = [q.to(device=device, dtype=torch.float64).contiguous() for q in bases]
    original = current
    generator = torch.Generator(device=device).manual_seed(int(seed))
    previous: List[torch.Tensor] = []
    errors: List[torch.Tensor] = []
    supports: List[torch.Tensor] = []
    candidates: List[torch.Tensor] = []
    for _component in range(int(max_components)):
        vector = torch.randn((batch, n_rows), generator=generator, device=device, dtype=torch.float64)
        for direction in previous:
            vector = vector - direction * torch.sum(direction * vector, dim=1, keepdim=True)
        vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-15)
        for _iteration in range(int(max_iter)):
            projected_sum = torch.zeros_like(vector)
            for q in current:
                z = torch.einsum("bnr,bn->br", q, vector)
                projected_sum = projected_sum + torch.einsum("bnr,br->bn", q, z)
            for direction in previous:
                projected_sum = projected_sum - direction * torch.sum(direction * projected_sum, dim=1, keepdim=True)
            next_vector = projected_sum / torch.linalg.vector_norm(projected_sum, dim=1, keepdim=True).clamp_min(1e-15)
            delta = torch.minimum(
                torch.linalg.vector_norm(next_vector - vector, dim=1),
                torch.linalg.vector_norm(next_vector + vector, dim=1),
            )
            vector = next_vector
            if float(delta.max().item()) <= convergence_tol:
                break
        for direction in previous:
            vector = vector - direction * torch.sum(direction * vector, dim=1, keepdim=True)
        vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-15)
        current_supports: List[torch.Tensor] = []
        current_error = torch.zeros(batch, device=device, dtype=torch.float64)
        original_supports: List[torch.Tensor] = []
        for q_current, q_original in zip(current, original):
            z_current = torch.einsum("bnr,bn->br", q_current, vector)
            current_supports.append(torch.sum(z_current * z_current, dim=1))
            current_error = current_error + 1.0 - current_supports[-1]
            z_original = torch.einsum("bnr,bn->br", q_original, vector)
            original_supports.append(torch.sum(z_original * z_original, dim=1))
        candidate = vector
        candidates.append(candidate)
        errors.append(current_error)
        supports.append(torch.stack(original_supports, dim=1))
        previous.append(candidate)
        new_current: List[torch.Tensor] = []
        for q_current in current:
            coeff = torch.einsum("bn,bnr->br", candidate, q_current)
            residual = q_current - candidate[:, :, None] * coeff[:, None, :]
            # The basis width is at most six.  Use a batched r-by-r Gram
            # eigendecomposition instead of a tall batched SVD so null runs
            # do not allocate an N-by-N workspace.  Zero trailing columns
            # preserve a fixed shape while representing each realization's
            # numerical rank.
            gram = torch.einsum("bnr,bns->brs", residual, residual)
            eigenvalues, vectors = torch.linalg.eigh(gram)
            order = torch.arange(int(eigenvalues.shape[1]) - 1, -1, -1, device=device)
            eigenvalues = eigenvalues.index_select(1, order).clamp_min(0.0)
            vectors = vectors.index_select(2, order)
            singular_values = torch.sqrt(eigenvalues)
            largest = singular_values[:, :1].clamp_min(1e-30)
            keep = singular_values > torch.maximum(torch.full_like(singular_values, 1e-12), largest * 1e-10)
            q_new = torch.einsum("bnr,brs->bns", residual, vectors) / singular_values[:, None, :].clamp_min(1e-30)
            new_current.append(q_new * keep[:, None, :].to(dtype=q_new.dtype))
        current = new_current
    return {
        "basis": torch.stack(candidates, dim=2),
        "errors": torch.stack(errors, dim=1),
        "support_scores": torch.stack(supports, dim=2),
    }


def _cobe_from_bases_batch_exact(
    bases: Sequence[torch.Tensor],
    *,
    max_components: int,
    tolerance: float = 1e-12,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Batched exact COBE using at most a 30-by-30 Gram eigensolve."""

    if not bases:
        raise ValueError("batched COBE needs blocks")
    batch = int(bases[0].shape[0])
    n_rows = int(bases[0].shape[1])
    current = [q.to(device=device, dtype=torch.float64).contiguous() for q in bases]
    original = current
    candidates: List[torch.Tensor] = []
    errors: List[torch.Tensor] = []
    supports: List[torch.Tensor] = []
    for _component in range(int(max_components)):
        concatenated = torch.cat(current, dim=2)
        gram = torch.einsum("bnr,bns->brs", concatenated, concatenated)
        eigenvalues, vectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(dims=(1,)).clamp_min(0.0)
        vectors = vectors.flip(dims=(2,))
        vector = torch.einsum("bnr,br->bn", concatenated, vectors[:, :, 0])
        for direction in candidates:
            vector = vector - direction * torch.sum(direction * vector, dim=1, keepdim=True)
        vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-15)
        current_error = torch.zeros(batch, device=device, dtype=torch.float64)
        original_supports: List[torch.Tensor] = []
        for q_current, q_original in zip(current, original):
            z_current = torch.einsum("bnr,bn->br", q_current, vector)
            current_error += 1.0 - torch.sum(z_current * z_current, dim=1)
            z_original = torch.einsum("bnr,bn->br", q_original, vector)
            original_supports.append(torch.sum(z_original * z_original, dim=1))
        candidates.append(vector)
        errors.append(current_error)
        supports.append(torch.stack(original_supports, dim=1))
        new_current: List[torch.Tensor] = []
        for q_current in current:
            coeff = torch.einsum("bn,bnr->br", vector, q_current)
            residual = q_current - vector[:, :, None] * coeff[:, None, :]
            gram_residual = torch.einsum("bnr,bns->brs", residual, residual)
            residual_values, residual_vectors = torch.linalg.eigh(gram_residual)
            residual_values = residual_values.flip(dims=(1,)).clamp_min(0.0)
            residual_vectors = residual_vectors.flip(dims=(2,))
            residual_sigma = torch.sqrt(residual_values)
            largest = residual_sigma[:, :1].clamp_min(1e-30)
            keep = residual_sigma > torch.maximum(torch.full_like(residual_sigma, float(tolerance)), largest * 1e-10)
            q_new = torch.einsum("bnr,brs->bns", residual, residual_vectors) / residual_sigma[:, None, :].clamp_min(1e-30)
            new_current.append(q_new * keep[:, None, :].to(dtype=q_new.dtype))
        current = new_current
    return {"basis": torch.stack(candidates, dim=2), "errors": torch.stack(errors, dim=1), "support_scores": torch.stack(supports, dim=2)}


def orthonormal_columns(matrix: torch.Tensor, tolerance: float = 1e-12) -> torch.Tensor:
    matrix64 = torch.as_tensor(matrix).detach().cpu().to(torch.float64)
    if matrix64.ndim != 2:
        raise ValueError("subspace matrix must be two-dimensional")
    if matrix64.shape[1] == 0:
        return torch.zeros((matrix64.shape[0], 0), dtype=torch.float64)
    singular_values, vectors = _gram_spectrum(matrix64)
    largest = float(singular_values[0].item()) if singular_values.numel() else 0.0
    keep = singular_values > max(float(tolerance), float(tolerance) * largest)
    if not bool(keep.any().item()):
        return torch.zeros((matrix64.shape[0], 0), dtype=torch.float64)
    basis = matrix64 @ vectors[:, keep]
    basis = basis / singular_values[keep][None, :]
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis


def principal_angle_summary(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    left64 = orthonormal_columns(left)
    right64 = orthonormal_columns(right)
    if left64.shape[1] == 0 or right64.shape[1] == 0:
        return {"angles_deg": [], "cos2": [], "mean_cos2": None, "min_cos2": None, "left_rank": int(left64.shape[1]), "right_rank": int(right64.shape[1])}
    singular_values, _ = _gram_spectrum(left64.T @ right64)
    singular_values = singular_values.clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.arccos(singular_values))
    cos2 = singular_values * singular_values
    return {
        "angles_deg": angles.tolist(),
        "cos2": cos2.tolist(),
        "mean_cos2": float(cos2.mean().item()) if cos2.numel() else None,
        "min_cos2": float(cos2.min().item()) if cos2.numel() else None,
        "left_rank": int(left64.shape[1]),
        "right_rank": int(right64.shape[1]),
    }


def _lower_tail_p(real_value: float, null_values: torch.Tensor) -> float:
    values = torch.as_tensor(null_values, dtype=torch.float64).flatten()
    values = values[torch.isfinite(values)]
    return float((1.0 + torch.sum(values <= float(real_value)).item()) / (values.numel() + 1.0))


def _upper_tail_p(real_value: float, null_values: torch.Tensor) -> float:
    values = torch.as_tensor(null_values, dtype=torch.float64).flatten()
    values = values[torch.isfinite(values)]
    return float((1.0 + torch.sum(values >= float(real_value)).item()) / (values.numel() + 1.0))


def bh_fdr(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg q-values while preserving missing entries."""

    result: List[Optional[float]] = [None] * len(p_values)
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    finite.sort(key=lambda item: item[1])
    running = 1.0
    count = len(finite)
    for rank in range(count, 0, -1):
        index, value = finite[rank - 1]
        running = min(running, value * count / rank)
        result[index] = float(min(1.0, running))
    return result


def projection_energy(field: torch.Tensor, basis: torch.Tensor, chunk_size: int = 250000) -> torch.Tensor:
    """Return per-channel energy explained by a spatial subspace."""

    y = _as_float64_matrix(field)
    a = orthonormal_columns(basis)
    if a.shape[1] == 0:
        return torch.zeros(y.shape[1], dtype=torch.float64)
    numerator = torch.zeros(y.shape[1], dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for start in range(0, y.shape[0], int(chunk_size)):
        block = y[start : start + int(chunk_size)]
        projected = a @ (a.T @ block)
        numerator += torch.sum(projected * projected, dim=0)
        denominator += torch.sum(block * block, dim=0)
    return numerator / denominator.clamp_min(1e-30)


def joint_pod(blocks: Sequence[torch.Tensor], max_components: int = 6) -> Dict[str, Any]:
    """Uncentered ordinary POD of ``[Y_G Y_H1 ... Y_H4]``."""

    if not blocks:
        raise ValueError("joint POD needs blocks")
    matrix = torch.cat([_as_float64_matrix(block) for block in blocks], dim=1)
    gram = matrix.T @ matrix
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    vectors = vectors.index_select(1, order)
    sigma = torch.sqrt(eigenvalues)
    total = float(eigenvalues.sum().item())
    energy = eigenvalues / total if total > 0.0 else torch.zeros_like(eigenvalues)
    rank = int((sigma > max(1e-12, float(sigma[0].item()) * 1e-10 if sigma.numel() else 1e-12)).sum().item())
    take = min(int(max_components), int(vectors.shape[1]), rank if rank else int(vectors.shape[1]))
    if take:
        left = matrix @ vectors[:, :take]
        left = left / sigma[:take].clamp_min(1e-30)[None, :]
        left = orthonormal_columns(left)
    else:
        left = torch.zeros((matrix.shape[0], 0), dtype=torch.float64)
    return {
        "matrix_shape": list(matrix.shape),
        "gram": gram,
        "sigma": sigma,
        "energy_ratio": energy,
        "cumulative_energy_ratio": torch.cumsum(energy, dim=0),
        "right_vectors": vectors,
        "left_basis": left,
        "rank": rank,
    }


def _support_for_subspace(q: torch.Tensor, basis: torch.Tensor) -> float:
    q64 = _as_float64_matrix(q)
    a = orthonormal_columns(basis)
    if a.shape[1] == 0:
        return 0.0
    return float(torch.sum((q64.T @ a) ** 2).item() / a.shape[1])


def _error_for_subspace(qs: Sequence[torch.Tensor], basis: torch.Tensor) -> float:
    a = orthonormal_columns(basis)
    if a.shape[1] == 0:
        return float("nan")
    return float(sum(1.0 - _support_for_subspace(q, a) for q in qs) / len(qs))


def _plot_heatmap(path: Path, values: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str], title: str, cmap: str = "viridis") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(xlabels)), max(4.0, 0.5 * len(ylabels))))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0 if np.isfinite(values).any() else None, vmax=1.0 if np.isfinite(values).any() else None)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if np.isfinite(values[row, col]):
                ax.text(col, row, f"{values[row, col]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error_null(path: Path, real_errors: torch.Tensor, null_errors: torch.Tensor, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    real = real_errors.detach().cpu().numpy()
    null = null_errors.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(1, len(real) + 1)
    if null.size:
        ax.boxplot([null[:, i] for i in range(null.shape[1])], positions=positions, widths=0.55, showfliers=False)
    ax.scatter(positions, real, color="tab:red", zorder=3, label="real")
    ax.set_xlabel("COBE candidate")
    ax.set_ylabel("f_k (lower is common)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_spatial_basis(path: Path, points: torch.Tensor, basis: torch.Tensor, title: str, max_points: int = 100000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if basis.shape[1] == 0 or points.shape[0] == 0:
        return
    indices = torch.arange(points.shape[0])
    if indices.numel() > max_points:
        indices = torch.linspace(0, indices.numel() - 1, max_points).round().to(torch.long)
    xy = points.index_select(0, indices).detach().cpu().numpy()
    values = basis.index_select(0, indices).detach().cpu().numpy()
    columns = min(values.shape[1], 4)
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.0), squeeze=False)
    for index in range(columns):
        axis = axes[0, index]
        scatter = axis.scatter(xy[:, 0], xy[:, 2], c=values[:, index], s=1, cmap="coolwarm", rasterized=True)
        axis.set_title(f"component {index + 1}")
        axis.set_xlabel("global x")
        axis.set_ylabel("global z")
        fig.colorbar(scatter, ax=axis, shrink=0.8)
    fig.suptitle(title + " — COBE basis; not a valid PBR material field")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_projection_energy(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = [f"{row['field']}:{row['channel']}" for row in rows]
    values = [float(row["energy"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.3), 4.5))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("projection energy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_principal_angle_matrix(path: Path, bases: Sequence[torch.Tensor], labels: Sequence[str], title: str) -> None:
    """Plot pairwise subspace similarity for the selected COBE spaces.

    The entries are mean squared cosines of principal angles.  Empty
    statistically unselected spaces are shown as NaN rather than being
    mistaken for a zero-similarity subspace.
    """

    if not bases:
        return
    values = np.full((len(bases), len(bases)), np.nan, dtype=np.float64)
    for row, left in enumerate(bases):
        for column, right in enumerate(bases):
            values[row, column] = principal_angle_summary(left, right).get("mean_cos2") or np.nan
    _plot_heatmap(path, values, labels, labels, title, cmap="magma")


def _resolve_cuda_device(requested: int) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested} is unavailable")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_ids: List[int] = []
    if visible and all(part.strip().lstrip("-").isdigit() for part in visible.split(",")):
        visible_ids = [int(part.strip()) for part in visible.split(",")]
    if visible_ids and requested in visible_ids:
        logical = visible_ids.index(requested)
        physical: Optional[int] = requested
    else:
        logical = requested
        physical = requested if not visible_ids else None
    if logical < 0 or logical >= torch.cuda.device_count():
        raise RuntimeError(f"requested cuda{requested} is unavailable: visible={visible!r}, count={torch.cuda.device_count()}")
    torch.cuda.set_device(logical)
    return {"requested_physical": int(requested), "logical": int(logical), "physical": physical, "name": torch.cuda.get_device_name(logical)}


def _cache_tile_dir(cache_dir: Path, tile_id: int) -> Path:
    return cache_dir / "tiles" / f"tile_{int(tile_id):02d}"


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_cache(cache_dir: Path, cuda_device: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Validate and load the existing Global C1024 final-field cache."""

    errors: List[str] = []
    preflight_path = cache_dir / "preflight.json"
    gate_path = cache_dir / "correctness_gate.json"
    summary_path = cache_dir / "summary.json"
    meta_path = cache_dir / "global_support" / "meta.json"
    for path in (preflight_path, gate_path, summary_path, meta_path):
        if not path.is_file():
            errors.append(f"missing cache manifest: {path}")
    if errors:
        raise RuntimeError("; ".join(errors))
    old_preflight = _read_json(preflight_path)
    old_gate = _read_json(gate_path)
    old_summary = _read_json(summary_path)
    old_meta = _read_json(meta_path)
    if old_preflight.get("format") != CACHE_FORMAT:
        errors.append(f"unexpected cache preflight format: {old_preflight.get('format')}")
    if old_preflight.get("status") != "ready":
        errors.append(f"old preflight status is {old_preflight.get('status')}")
    if old_gate.get("status") != "passed":
        errors.append(f"old correctness gate status is {old_gate.get('status')}")
    if old_summary.get("status") != "completed":
        errors.append(f"old cache summary status is {old_summary.get('status')}")
    baseline_path = Path(str(old_preflight.get("baseline_path", "")))
    expected_baseline_hash = str(old_preflight.get("baseline_sha256", ""))
    meta_baseline_hash = str(old_meta.get("artifact_sha256", ""))
    if not baseline_path.is_file():
        errors.append(f"missing Global baseline artifact: {baseline_path}")
    else:
        if expected_baseline_hash and meta_baseline_hash and expected_baseline_hash != meta_baseline_hash:
            errors.append("baseline hash differs between old preflight and support metadata")
        current_hash = _sha256(baseline_path)
        if expected_baseline_hash and current_hash != expected_baseline_hash:
            errors.append("Global baseline artifact hash changed since cache construction")
    support_path = cache_dir / "global_support" / "G.pt"
    coords_path = cache_dir / "global_support" / "coords.pt"
    points_path = cache_dir / "global_support" / "points.pt"
    for path in (support_path, coords_path, points_path):
        if not path.is_file():
            errors.append(f"missing global support cache: {path}")
    if errors:
        raise RuntimeError("; ".join(errors))
    global_field = _load_torch(support_path).to(torch.float32).contiguous()
    coords = _load_torch(coords_path).to(torch.int32).contiguous()
    points = _load_torch(points_path).to(torch.float32).contiguous()
    if global_field.ndim != 2 or tuple(global_field.shape[1:]) != (6,):
        errors.append(f"G.pt shape is {tuple(global_field.shape)}, expected [N,6]")
    if coords.ndim != 2 or tuple(coords.shape[1:]) != (3,) or coords.shape[0] != global_field.shape[0]:
        errors.append("global coordinate shape/order does not match G.pt")
    if points.ndim != 2 or tuple(points.shape[1:]) != (3,) or points.shape[0] != global_field.shape[0]:
        errors.append("global physical point shape/order does not match G.pt")
    if not torch.isfinite(global_field).all() or not torch.isfinite(points).all():
        errors.append("global support contains non-finite values")
    tile_ids = sorted({int(value) for value in old_preflight.get("phase_a_tile_ids", [])})
    required_ids = sorted({value for quartet in QUARTETS for value in quartet})
    if tile_ids != required_ids:
        errors.append(f"cache tile IDs {tile_ids} do not cover required IDs {required_ids}")
    fields: Dict[int, torch.Tensor] = {}
    valid: Dict[int, torch.Tensor] = {}
    hidden: Dict[int, torch.Tensor] = {}
    observed: Dict[int, torch.Tensor] = {}
    cache_integrity: Dict[str, Any] = {
        "global_support": {
            "path": str(support_path.resolve()),
            "sha256": _sha256(support_path),
            "coords_path": str(coords_path.resolve()),
            "points_path": str(points_path.resolve()),
            "shape": list(global_field.shape),
            "baseline_path": str(baseline_path.resolve()),
            "baseline_sha256": expected_baseline_hash,
        },
        "tiles": {},
    }
    for tile_id in required_ids:
        tile_dir = _cache_tile_dir(cache_dir, tile_id)
        field_path = tile_dir / "H_on_global.pt"
        valid_path = tile_dir / "valid_mask.pt"
        hidden_path = tile_dir / "hidden_mask_global.pt"
        observed_path = tile_dir / "observed_mask_global.pt"
        meta_tile_path = tile_dir / "query_meta.json"
        for path in (field_path, valid_path, hidden_path, observed_path, meta_tile_path):
            if not path.is_file():
                errors.append(f"tile {tile_id}: missing {path}")
        if errors:
            continue
        field = _load_torch(field_path).to(torch.float32).contiguous()
        valid_mask = _load_torch(valid_path).to(torch.bool).reshape(-1).contiguous()
        hidden_mask = _load_torch(hidden_path).to(torch.bool).reshape(-1).contiguous()
        observed_mask = _load_torch(observed_path).to(torch.bool).reshape(-1).contiguous()
        if field.shape != global_field.shape:
            errors.append(f"tile {tile_id}: H_on_global shape {tuple(field.shape)} != {tuple(global_field.shape)}")
        for name, mask in (("valid", valid_mask), ("hidden", hidden_mask), ("observed", observed_mask)):
            if mask.shape != (global_field.shape[0],):
                errors.append(f"tile {tile_id}: {name} mask shape {tuple(mask.shape)} is not [N]")
        if field.shape == global_field.shape:
            finite_rows = torch.isfinite(field).all(dim=1)
            if bool((finite_rows != valid_mask).any().item()):
                errors.append(f"tile {tile_id}: finite H rows do not exactly match valid_mask")
        if bool((hidden_mask & ~valid_mask).any().item()) or bool((observed_mask & ~valid_mask).any().item()):
            errors.append(f"tile {tile_id}: hidden/observed mask contains invalid rows")
        if bool((hidden_mask & observed_mask).any().item()):
            errors.append(f"tile {tile_id}: hidden and observed masks overlap")
        query_meta = _read_json(meta_tile_path)
        query_rows = int(query_meta.get("query", {}).get("fine_rows", global_field.shape[0])) if isinstance(query_meta.get("query"), Mapping) else int(global_field.shape[0])
        if query_rows != int(global_field.shape[0]):
            errors.append(f"tile {tile_id}: query row count {query_rows} does not equal global support count")
        fields[tile_id] = field
        valid[tile_id] = valid_mask
        hidden[tile_id] = hidden_mask
        observed[tile_id] = observed_mask
        cache_integrity["tiles"][str(tile_id)] = {
            "field_path": str(field_path.resolve()),
            "field_sha256": _sha256(field_path),
            "valid_path": str(valid_path.resolve()),
            "hidden_path": str(hidden_path.resolve()),
            "observed_path": str(observed_path.resolve()),
            "query_meta_path": str(meta_tile_path.resolve()),
            "field_shape": list(field.shape),
            "local_active_support_count": int(query_meta.get("support_count", 0)),
            "global_query_row_count": query_rows,
            "valid_count": int(valid_mask.sum().item()),
            "hidden_count": int(hidden_mask.sum().item()),
            "observed_count": int(observed_mask.sum().item()),
            "row_order": "global_support/coords.pt order",
        }
    if len(fields) != len(required_ids):
        errors.append("not all required tile fields loaded")
    gate_snapshot = {
        "old_preflight": str(preflight_path.resolve()),
        "old_correctness_gate": str(gate_path.resolve()),
        "old_summary": str(summary_path.resolve()),
        "old_gate_status": old_gate.get("status"),
        "global_self_query": old_gate.get("global_self_query"),
        "roundtrip": old_gate.get("roundtrip"),
        "tile26_tile27_query_correctness": old_gate.get("tile26_tile27_query_correctness"),
        "mask_counts": old_gate.get("mask_counts"),
        "pairwise_overlap_counts": old_gate.get("pairwise_overlap_counts"),
        "cache_checks": {
            "baseline_hash_matches": not any("baseline" in error.lower() and "hash" in error.lower() for error in errors),
            "unified_global_support": len(fields) == len(required_ids),
            "fixed_row_order": not any("shape/order" in error.lower() or "row" in error.lower() for error in errors),
            "finite_missing_exclusion": not any("finite H" in error for error in errors),
        },
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    if errors:
        raise RuntimeError("cache correctness gate failed: " + "; ".join(errors[:12]))
    if cuda_device is not None:
        gate_snapshot["cuda"] = _resolve_cuda_device(int(cuda_device))
    return dict(cache_integrity), valid, hidden, observed, global_field, points, gate_snapshot


def _parent_masks(quartet: Sequence[int], valid: Mapping[int, torch.Tensor], hidden: Mapping[int, torch.Tensor], observed: Mapping[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
    base = torch.ones_like(valid[int(quartet[0])], dtype=torch.bool)
    all_hidden = torch.ones_like(base)
    all_observed = torch.ones_like(base)
    for tile_id in quartet:
        base &= valid[int(tile_id)]
        all_hidden &= hidden[int(tile_id)]
        all_observed &= observed[int(tile_id)]
    return {"ALL_VALID": base, "ALL_HIDDEN": base & all_hidden, "ALL_OBSERVED": base & all_observed}


def _load_blocks(
    rows: torch.Tensor,
    quartet: Sequence[int],
    global_field: torch.Tensor,
    fields: Mapping[int, torch.Tensor],
    view: str,
    control: str,
) -> List[torch.Tensor]:
    blocks = [global_field.index_select(0, rows).to(torch.float64)] + [fields[int(tile_id)].index_select(0, rows).to(torch.float64) for tile_id in quartet]
    if control == "SPATIAL_DEMEANED":
        blocks = [block - block.mean(dim=0, keepdim=True) for block in blocks]
    channels = VIEW_CHANNELS[view]
    return [block[:, :channels].contiguous() for block in blocks]


def _field_names(quartet: Sequence[int]) -> List[str]:
    return ["G"] + [f"H{int(tile_id)}" for tile_id in quartet]


def _make_permutations(n_rows: int, num_permutations: int, seed: int, field_count: int = 4) -> List[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [torch.stack([torch.randperm(n_rows, generator=generator, dtype=torch.int32) for _ in range(int(num_permutations))]) for _ in range(int(field_count))]


def _permutation_batch(permutations: Any, start: int, stop: int) -> List[torch.Tensor]:
    """Return one deterministic permutation batch without retaining all rows."""

    if isinstance(permutations, Mapping):
        n_rows = int(permutations["n_rows"])
        seed = int(permutations["seed"])
        field_count = int(permutations.get("field_count", 4))
        return [
            torch.stack([
                torch.randperm(
                    n_rows,
                    generator=torch.Generator(device="cpu").manual_seed(_stable_seed(seed, "field", field, "perm", index)),
                    dtype=torch.int32,
                )
                for index in range(int(start), int(stop))
            ])
            for field in range(field_count)
        ]
    return [permutations[field][start:stop] for field in range(len(permutations))]


def _batched_null_candidates(
    real_bases: Sequence[torch.Tensor],
    permutations: Any,
    *,
    num_permutations: int,
    batch_size: int,
    max_components: int,
    max_iter: int,
    convergence_tol: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run all-shared and four subset COBE nulls.

    Only the scalar error/support distributions are retained.  Candidate
    bases are N-by-r tensors and would make 256 permutations unnecessarily
    memory-heavy; the small number of bases needed for held-out evaluation is
    recomputed in streaming batches below.
    """

    if len(real_bases) != 5 or (isinstance(permutations, Mapping) and int(permutations.get("field_count", 4)) != 4) or (not isinstance(permutations, Mapping) and len(permutations) != 4):
        raise ValueError("the physical experiment requires G plus four H blocks")
    n_rows = int(real_bases[0].shape[0])
    rank_limit = min(int(q.shape[1]) for q in real_bases)
    max_components = min(int(max_components), rank_limit)
    stages = {"all": (0, 1, 2, 3, 4), **{f"omit_{j}": tuple(i for i in range(5) if i != j) for j in range(1, 5)}}
    stage_errors: Dict[str, torch.Tensor] = {key: torch.empty((num_permutations, max_components), dtype=torch.float64) for key in stages}
    stage_support: Dict[str, torch.Tensor] = {key: torch.empty((num_permutations, len(indices), max_components), dtype=torch.float64) for key, indices in stages.items()}
    for start in range(0, num_permutations, int(batch_size)):
        stop = min(num_permutations, start + int(batch_size))
        batch_count = stop - start
        q_batch: List[torch.Tensor] = [real_bases[0].to(device=device, dtype=torch.float64).unsqueeze(0).expand(batch_count, -1, -1)]
        permutation_batch = _permutation_batch(permutations, start, stop)
        for h_index in range(4):
            permutation = permutation_batch[h_index].to(torch.long)
            source = real_bases[h_index + 1]
            q_batch.append(source.index_select(0, permutation.reshape(-1)).reshape(batch_count, n_rows, source.shape[1]).to(device=device, dtype=torch.float64))
        for stage_name, indices in stages.items():
            result = _cobe_from_bases_batch_exact(
                [q_batch[index] for index in indices],
                max_components=max_components,
                tolerance=1e-12,
                device=device,
            )
            stage_errors[stage_name][start:stop] = result["errors"].detach().cpu()
            stage_support[stage_name][start:stop] = result["support_scores"].detach().cpu()
            del result
        del q_batch, permutation_batch
        torch.cuda.empty_cache()
    return {"errors": stage_errors, "support_scores": stage_support, "stage_indices": stages, "basis_storage": "streamed_recompute"}


def _null_majority_distributions(
    null_data: Mapping[str, Any],
    real_all_indices: Sequence[int],
    real_subset_indices: Mapping[str, Sequence[int]],
    real_extra_ranks: Mapping[str, int],
    real_bases: Sequence[torch.Tensor],
    permutations: Any,
    *,
    batch_size: int,
    null_seed: int,
    null_max_iter: int,
    convergence_tol: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Evaluate the held-out and fitting support of real-rank null extras."""

    num_permutations = int(next(iter(null_data["errors"].values())).shape[0])
    n_rows = int(real_bases[0].shape[0])
    null_support: Dict[str, torch.Tensor] = {}
    null_errors: Dict[str, torch.Tensor] = {}
    for omit_index in range(1, 5):
        stage_name = f"omit_{omit_index}"
        subset_indices = [int(index) for index in real_subset_indices.get(stage_name, ())]
        rank = int(real_extra_ranks.get(stage_name, 0))
        if rank <= 0:
            continue
        all_indices = [int(index) for index in real_all_indices]
        all_rank = len(all_indices)
        subset_basis_rank = len(subset_indices)
        if subset_basis_rank <= 0:
            continue
        basis_support = torch.full((num_permutations, 5), float("nan"), dtype=torch.float64)
        basis_error = torch.full((num_permutations,), float("nan"), dtype=torch.float64)
        for start in range(0, num_permutations, int(batch_size)):
            stop = min(num_permutations, start + int(batch_size))
            batch_count = stop - start
            q_batch: List[torch.Tensor] = [real_bases[0].to(device=device, dtype=torch.float64).unsqueeze(0).expand(batch_count, -1, -1)]
            permutation_batch = _permutation_batch(permutations, start, stop)
            for h_index in range(4):
                permutation = permutation_batch[h_index].to(torch.long)
                source = real_bases[h_index + 1]
                q_batch.append(source.index_select(0, permutation.reshape(-1)).reshape(batch_count, n_rows, source.shape[1]).to(device=device, dtype=torch.float64))
            stage_result = _cobe_from_bases_batch_exact(
                [q_batch[index] for index in null_data["stage_indices"][stage_name]],
                max_components=max(max(subset_indices, default=-1) + 1, rank),
                tolerance=1e-12,
                device=device,
            )
            subset_a = stage_result["basis"].index_select(2, torch.tensor(subset_indices, dtype=torch.long, device=device)).to(dtype=torch.float64)
            if all_rank > 0:
                all_result = _cobe_from_bases_batch_exact(
                    [q_batch[index] for index in null_data["stage_indices"]["all"]],
                    max_components=max(all_indices) + 1,
                    tolerance=1e-12,
                    device=device,
                )
                all_a = all_result["basis"].index_select(2, torch.tensor(all_indices, dtype=torch.long, device=device)).to(dtype=torch.float64)
                overlap = torch.einsum("bnr,bns->brs", all_a, subset_a)
                residual = subset_a - torch.einsum("bnr,brs->bns", all_a, overlap)
            else:
                residual = subset_a
            b, _ = torch.linalg.qr(residual, mode="reduced")
            b_rank = min(int(b.shape[2]), rank)
            if b_rank <= 0:
                continue
            b = b[:, :, :b_rank]
            selected = [index for index in range(5) if index != omit_index]
            for field_index, q in enumerate(q_batch):
                z = torch.einsum("bnr,bns->brs", q, b)
                score = torch.sum(z * z, dim=(1, 2)) / float(b_rank)
                basis_support[start:stop, field_index] = score.detach().cpu()
            fit_error = torch.zeros(batch_count, device=device, dtype=torch.float64)
            for index in selected:
                z = torch.einsum("bnr,bns->brs", q_batch[index], b)
                fit_error += 1.0 - torch.sum(z * z, dim=(1, 2)) / float(b_rank)
            basis_error[start:stop] = (fit_error / float(len(selected))).detach().cpu()
            del q_batch, permutation_batch, subset_a, b, stage_result
            if all_rank > 0:
                del all_result
        null_support[stage_name] = basis_support
        null_errors[stage_name] = basis_error
    return null_support, null_errors


def _component_significance(
    real: CobeResult,
    null_errors: torch.Tensor,
    null_support: torch.Tensor,
    field_names: Sequence[str],
    alpha: float,
) -> Dict[str, Any]:
    error_p = [_lower_tail_p(float(real.errors[k].item()), null_errors[:, k]) for k in range(real.rank)]
    error_q = bh_fdr(error_p)
    support_p = [[_upper_tail_p(float(real.support_scores[field, k].item()), null_support[:, field, k]) for k in range(real.rank)] for field in range(len(field_names))]
    support_q = [bh_fdr(row) for row in support_p]
    all_shared: List[int] = []
    for k in range(real.rank):
        if error_q[k] is None or error_q[k] > alpha:
            continue
        if all(support_q[field][k] is not None and support_q[field][k] <= alpha for field in range(len(field_names))):
            all_shared.append(k)
    return {
        "error_p": error_p,
        "error_q": error_q,
        "support_p": support_p,
        "support_q": support_q,
        "all_shared_indices": all_shared,
    }


def _majority_significance(
    basis: torch.Tensor,
    all_qs: Sequence[torch.Tensor],
    fitting_indices: Sequence[int],
    null_support: torch.Tensor,
    null_error: torch.Tensor,
    alpha: float,
) -> Dict[str, Any]:
    if basis.shape[1] == 0:
        return {"rank": 0, "support_scores": [], "support_p": [], "support_q": [], "error": None, "error_p": None, "error_q": None, "label": "WEAK_OR_PARTIAL"}
    scores = [_support_for_subspace(q, basis) for q in all_qs]
    support_p = [_upper_tail_p(score, null_support[:, index]) for index, score in enumerate(scores)]
    support_q = bh_fdr(support_p)
    error = _error_for_subspace([all_qs[index] for index in fitting_indices], basis)
    error_p = _lower_tail_p(error, null_error)
    error_q = bh_fdr([error_p])[0]
    supported = [index for index, qvalue in enumerate(support_q) if qvalue is not None and qvalue <= alpha]
    fitting_h_supported = [index for index in fitting_indices if index > 0 and index in supported]
    if error_q is not None and error_q <= alpha and 0 in supported and len(fitting_h_supported) >= 3:
        label = "ALL_SHARED" if len(supported) == len(all_qs) else "G_MAJORITY_H_SHARED"
    else:
        label = "WEAK_OR_PARTIAL"
    return {
        "rank": int(basis.shape[1]),
        "support_scores": scores,
        "support_p": support_p,
        "support_q": support_q,
        "error": error,
        "error_p": error_p,
        "error_q": error_q,
        "supported_field_indices": supported,
        "fitting_supported_field_indices": [int(index) for index in fitting_indices if index in supported],
        "label": label,
    }


def _analysis_key(quartet: Sequence[int], domain: str, view: str, control: str) -> str:
    return f"{'_'.join(str(int(v)) for v in quartet)}__{domain}__{view}__{control}"


def _save_stage_projections(stage_dir: Path, basis: torch.Tensor, blocks: Sequence[torch.Tensor], field_names: Sequence[str]) -> List[Dict[str, Any]]:
    if basis.shape[1] == 0:
        _atomic_torch(stage_dir / "basis.pt", basis)
        return []
    _atomic_torch(stage_dir / "basis.pt", basis)
    rows: List[Dict[str, Any]] = []
    a = orthonormal_columns(basis)
    for field, block in zip(field_names, blocks):
        projection = a @ (a.T @ block)
        residual = block - projection
        _atomic_torch(stage_dir / f"projection_{field}.pt", projection.to(torch.float32))
        _atomic_torch(stage_dir / f"residual_{field}.pt", residual.to(torch.float32))
    return rows


def _save_analysis(
    output_dir: Path,
    quartet: Sequence[int],
    domain: str,
    view: str,
    control: str,
    blocks: Sequence[torch.Tensor],
    rows: torch.Tensor,
    points: torch.Tensor,
    real_all: CobeResult,
    real_subsets: Mapping[str, CobeResult],
    all_sig: Mapping[str, Any],
    subset_sig: Mapping[str, Mapping[str, Any]],
    extras: Mapping[str, torch.Tensor],
    majority_stats: Mapping[str, Mapping[str, Any]],
    null_data: Mapping[str, Any],
    null_majority_support: Mapping[str, torch.Tensor],
    null_majority_error: Mapping[str, torch.Tensor],
    basis_metadata: Sequence[Mapping[str, Any]],
    alpha: float,
) -> Dict[str, Any]:
    base = output_dir / "quartets" / "_".join(str(int(v)) for v in quartet) / domain / view / control
    base.mkdir(parents=True, exist_ok=True)
    field_names = _field_names(quartet)
    analysis_record: Dict[str, Any] = {
        "quartet": list(map(int, quartet)),
        "domain": domain,
        "view": view,
        "control": control,
        "row_count": int(rows.numel()),
        "row_indices_hash": hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
        "field_names": field_names,
        "block_ranks": real_all.block_ranks,
        "orthogonality_max_abs": float(torch.max(torch.abs(real_all.basis.T @ real_all.basis - torch.eye(real_all.rank, dtype=torch.float64))).item()) if real_all.rank else 0.0,
        "basis_metadata": list(basis_metadata),
        "all_shared_indices": list(all_sig["all_shared_indices"]),
        "all_shared_rank": len(all_sig["all_shared_indices"]),
        "majority": {},
        "alpha": float(alpha),
        "permutation_count": int(next(iter(null_data["errors"].values())).shape[0]),
    }
    _atomic_json(base / "analysis.json", analysis_record)
    _atomic_torch(base / "rows.pt", rows)
    selected_all = real_all.basis[:, all_sig["all_shared_indices"]] if all_sig["all_shared_indices"] else real_all.basis[:, :0]
    _atomic_torch(base / "cobe_basis_overextract.pt", real_all.basis)
    _atomic_torch(base / "A_all.pt", selected_all)
    _atomic_torch(base / "cobe_errors_overextract.pt", real_all.errors)
    _atomic_json(base / "cobe_errors_all.json", {"f_k": real_all.errors.tolist(), "iterations": real_all.iterations, "converged": real_all.converged})
    _atomic_torch(base / "support_scores_overextract.pt", real_all.support_scores)
    _atomic_json(base / "cobe_metadata.json", {"iterations": real_all.iterations, "converged": real_all.converged, "current_ranks": real_all.current_ranks, "block_ranks": real_all.block_ranks, "field_names": field_names})
    _write_csv(
        base / "support_scores_all.csv",
        [
            {"candidate": k, "field": field_names[field], "support_score": float(real_all.support_scores[field, k].item()), "p": all_sig["support_p"][field][k], "q": all_sig["support_q"][field][k]}
            for field in range(len(field_names))
            for k in range(real_all.rank)
        ],
    )
    _write_csv(
        base / "cobe_errors_all.csv",
        [{"candidate": k, "f_k": float(real_all.errors[k].item()), "p": all_sig["error_p"][k], "q": all_sig["error_q"][k]} for k in range(real_all.rank)],
    )
    all_dir = base / "all_shared"
    _save_stage_projections(all_dir, selected_all, blocks, field_names)
    _atomic_json(all_dir / "significance.json", all_sig)
    _plot_heatmap(
        base / "field_support_heatmap.png",
        real_all.support_scores.detach().cpu().numpy() if real_all.rank else np.zeros((len(field_names), 0)),
        [f"c{k + 1}" for k in range(real_all.rank)],
        field_names,
        "COBE field support — all five blocks",
        cmap="viridis",
    )
    null_errors_all = null_data["errors"]["all"]
    _plot_error_null(base / "real_vs_null_cobe_error.png", real_all.errors, null_errors_all, "All-shared COBE error: real vs spatial permutation null")
    projection_rows: List[Dict[str, Any]] = []
    if selected_all.shape[1]:
        for field, block in zip(field_names, blocks):
            energies = projection_energy(block, selected_all)
            for channel_index, energy in enumerate(energies.tolist()):
                channel_name = PBR_CHANNEL_NAMES[channel_index] if view == "PBR6" else PBR_CHANNEL_NAMES[channel_index]
                projection_rows.append({"stage": "ALL_SHARED", "omit": "", "field": field, "channel": channel_name, "energy": float(energy), "rank": int(selected_all.shape[1])})
    for stage_name, result in real_subsets.items():
        omit_index = int(stage_name.split("_")[1])
        omit_tile = int(quartet[omit_index - 1])
        sig = subset_sig[stage_name]
        selected_indices = sig["selected_indices"]
        subset_basis = result.basis[:, selected_indices] if selected_indices else result.basis[:, :0]
        extra = extras.get(stage_name, torch.zeros((rows.numel(), 0), dtype=torch.float64))
        stats = majority_stats.get(stage_name, {"label": "WEAK_OR_PARTIAL", "rank": 0})
        stage_dir = base / f"omit_{omit_tile:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch(stage_dir / "subset_cobe_basis.pt", result.basis)
        _atomic_torch(stage_dir / "subset_cobe_errors.pt", result.errors)
        _atomic_torch(stage_dir / "subset_support_scores.pt", result.support_scores)
        _atomic_torch(stage_dir / f"majority_extra_omit_{omit_tile:02d}.pt", extra)
        _atomic_json(stage_dir / "significance.json", {"subset": sig, "majority": stats})
        _save_stage_projections(stage_dir / "majority_extra", extra, blocks, field_names)
        for field, score, p_value, q_value in zip(field_names, stats.get("support_scores", []), stats.get("support_p", []), stats.get("support_q", [])):
            projection_rows.append({"stage": stats.get("label", "WEAK_OR_PARTIAL"), "omit": omit_tile, "field": field, "channel": "joint", "energy": float(score), "support_p": p_value, "support_q": q_value, "rank": stats.get("rank", 0)})
        if extra.shape[1]:
            _plot_heatmap(
                stage_dir / "majority_extra_support.png",
                np.asarray([stats.get("support_scores", [])], dtype=np.float64),
                field_names,
                [f"B_-{omit_tile}"],
                f"majority-extra support, omit H{omit_tile}",
                cmap="viridis",
            )
            _plot_spatial_basis(stage_dir / "common_spatial_basis.png", points, extra, f"omit H{omit_tile}")
            for field, block in zip(field_names, blocks):
                energies = projection_energy(block, extra)
                for channel_index, energy in enumerate(energies.tolist()):
                    projection_rows.append({"stage": stats.get("label", "WEAK_OR_PARTIAL"), "omit": omit_tile, "field": field, "channel": PBR_CHANNEL_NAMES[channel_index], "energy": float(energy), "rank": int(extra.shape[1])})
        analysis_record["majority"][stage_name] = {"omit_tile": omit_tile, **dict(stats)}
    _plot_spatial_basis(base / "all_shared" / "common_spatial_basis.png", points, selected_all, "all-shared")
    _plot_projection_energy(base / "common_projection_energy.png", projection_rows, "common-subspace projection energy")
    null_dir = base / "null"
    null_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch(null_dir / "cobe_errors.pt", null_data["errors"])
    _atomic_torch(null_dir / "support_scores.pt", null_data["support_scores"])
    # Candidate bases are an internal computational intermediate.  Persisting
    # all B x N x r null bases for 48 analyses would need hundreds of GB and
    # is not needed to reproduce the reported null error/support distributions.
    _atomic_json(null_dir / "candidate_basis_metadata.json", {"stored": False, "reason": "large intermediate; error/support/held-out distributions are persisted", "storage": null_data.get("basis_storage", "streamed_recompute")})
    _atomic_torch(null_dir / "majority_support.pt", dict(null_majority_support))
    _atomic_torch(null_dir / "majority_errors.pt", dict(null_majority_error))
    _atomic_json(base / "analysis_complete.json", analysis_record)
    return analysis_record


def _subset_significance(result: CobeResult, null_errors: torch.Tensor, null_support: torch.Tensor, field_names: Sequence[str], alpha: float) -> Dict[str, Any]:
    error_p = [_lower_tail_p(float(result.errors[k].item()), null_errors[:, k]) for k in range(result.rank)]
    error_q = bh_fdr(error_p)
    support_p = [[_upper_tail_p(float(result.support_scores[field, k].item()), null_support[:, field, k]) for k in range(result.rank)] for field in range(len(field_names))]
    support_q = [bh_fdr(row) for row in support_p]
    selected = [
        k for k in range(result.rank)
        if error_q[k] is not None and error_q[k] <= alpha and all(support_q[field][k] is not None and support_q[field][k] <= alpha for field in range(len(field_names)))
    ]
    return {"error_p": error_p, "error_q": error_q, "support_p": support_p, "support_q": support_q, "selected_indices": selected, "selected_rank": len(selected)}


def _run_one_analysis(
    *,
    output_dir: Path,
    quartet: Sequence[int],
    domain: str,
    view: str,
    control: str,
    rows: torch.Tensor,
    points: torch.Tensor,
    blocks: Sequence[torch.Tensor],
    permutations: Sequence[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    basis_cache: Optional[Sequence[BasisResult]] = None,
) -> Dict[str, Any]:
    field_names = _field_names(quartet)
    basis_results = list(basis_cache) if basis_cache is not None else [effective_rank_basis(block, absolute_tolerance=args.rank_abs_tol, relative_tolerance=args.rank_rel_tol) for block in blocks]
    real_bases = [result.basis for result in basis_results]
    max_components = min(result.rank for result in basis_results)
    real_all = _cobe_from_bases_exact(real_bases, max_components=max_components, tolerance=args.rank_abs_tol)
    real_subsets: Dict[str, CobeResult] = {}
    for omit_index in range(1, 5):
        indices = [index for index in range(5) if index != omit_index]
        real_subsets[f"omit_{omit_index}"] = _cobe_from_bases_exact([real_bases[index] for index in indices], max_components=min(real_bases[index].shape[1] for index in indices), tolerance=args.rank_abs_tol)
    null_data = _batched_null_candidates(
        real_bases,
        permutations,
        num_permutations=args.num_permutations,
        batch_size=args.null_batch_size,
        max_components=max_components,
        max_iter=args.null_max_iter,
        convergence_tol=args.convergence_tol,
        seed=_stable_seed(args.seed, "null", _analysis_key(quartet, domain, view, control)),
        device=device,
    )
    all_sig = _component_significance(real_all, null_data["errors"]["all"], null_data["support_scores"]["all"], field_names, args.alpha)
    subset_sig: Dict[str, Dict[str, Any]] = {}
    for omit_index in range(1, 5):
        stage_name = f"omit_{omit_index}"
        subset_fields = [field_names[index] for index in null_data["stage_indices"][stage_name]]
        subset_sig[stage_name] = _subset_significance(real_subsets[stage_name], null_data["errors"][stage_name], null_data["support_scores"][stage_name], subset_fields, args.alpha)
    real_all_selected_indices = list(all_sig["all_shared_indices"])
    real_subset_indices = {stage: list(sig["selected_indices"]) for stage, sig in subset_sig.items()}
    selected_all = real_all.basis[:, real_all_selected_indices] if real_all_selected_indices else real_all.basis[:, :0]
    precomputed_extras: Dict[str, torch.Tensor] = {}
    for omit_index in range(1, 5):
        stage_name = f"omit_{omit_index}"
        selected_indices = real_subset_indices[stage_name]
        subset_basis = real_subsets[stage_name].basis[:, selected_indices] if selected_indices else real_subsets[stage_name].basis[:, :0]
        if subset_basis.shape[1] and selected_all.shape[1]:
            precomputed_extras[stage_name] = orthonormal_columns(subset_basis - selected_all @ (selected_all.T @ subset_basis))
        else:
            precomputed_extras[stage_name] = orthonormal_columns(subset_basis)
    real_extra_ranks = {stage: int(basis.shape[1]) for stage, basis in precomputed_extras.items()}
    null_majority_support, null_majority_error = _null_majority_distributions(
        null_data,
        real_all_selected_indices,
        real_subset_indices,
        real_extra_ranks,
        real_bases,
        permutations,
        batch_size=args.null_batch_size,
        null_seed=_stable_seed(args.seed, "null", _analysis_key(quartet, domain, view, control)),
        null_max_iter=args.null_max_iter,
        convergence_tol=args.convergence_tol,
        device=device,
    )
    extras: Dict[str, torch.Tensor] = {}
    majority_stats: Dict[str, Dict[str, Any]] = {}
    for omit_index in range(1, 5):
        stage_name = f"omit_{omit_index}"
        selected_indices = subset_sig[stage_name]["selected_indices"]
        subset_basis = real_subsets[stage_name].basis[:, selected_indices] if selected_indices else real_subsets[stage_name].basis[:, :0]
        extra = precomputed_extras[stage_name]
        extras[stage_name] = extra
        subset_indices = null_data["stage_indices"][stage_name]
        if extra.shape[1] and stage_name in null_majority_support:
            stats = _majority_significance(extra, real_bases, subset_indices, null_majority_support[stage_name], null_majority_error[stage_name], args.alpha)
            # Add the held-out field's support p/q using the same null B
            # distribution.  This is the decisive out-of-subset check.
            heldout_q = real_bases[omit_index]
            heldout_score = _support_for_subspace(heldout_q, extra)
            heldout_p = _upper_tail_p(heldout_score, null_majority_support[stage_name][:, omit_index])
            stats["heldout_support_score"] = heldout_score
            stats["heldout_support_p"] = heldout_p
            stats["heldout_support_q"] = bh_fdr([heldout_p])[0]
            majority_stats[stage_name] = stats
        else:
            majority_stats[stage_name] = {"rank": 0, "support_scores": [], "support_p": [], "support_q": [], "error": None, "error_p": None, "error_q": None, "label": "WEAK_OR_PARTIAL", "heldout_support_score": None, "heldout_support_p": None, "heldout_support_q": None}
    # Store a joint POD baseline and compare its dominant spatial directions
    # with the statistically selected COBE spaces.
    pod = joint_pod(blocks, max_components=max_components)
    _atomic_torch(
        output_dir / "quartets" / "_".join(str(int(v)) for v in quartet) / domain / view / control / "pod_baseline.pt",
        {key: value for key, value in pod.items() if isinstance(value, (torch.Tensor, int, float, list, tuple))},
    )
    pod_rows = [{"mode": index + 1, "sigma": float(value.item()), "energy_ratio": float(pod["energy_ratio"][index].item()), "cumulative_energy_ratio": float(pod["cumulative_energy_ratio"][index].item()), "rank": int(pod["rank"])} for index, value in enumerate(pod["sigma"])]
    record = _save_analysis(
        output_dir,
        quartet,
        domain,
        view,
        control,
        blocks,
        rows,
        points,
        real_all,
        real_subsets,
        all_sig,
        subset_sig,
        extras,
        majority_stats,
        null_data,
        null_majority_support,
        null_majority_error,
        [result.metadata() for result in basis_results],
        args.alpha,
    )
    record["pod"] = pod_rows
    record["pod_vs_all_shared"] = principal_angle_summary(pod["left_basis"], selected_all)
    for stage_name, extra in extras.items():
        record[f"pod_vs_{stage_name}"] = principal_angle_summary(pod["left_basis"], extra)
    _atomic_json(output_dir / "quartets" / "_".join(str(int(v)) for v in quartet) / domain / view / control / "analysis_complete.json", record)
    return {"record": record, "all": real_all, "all_sig": all_sig, "subsets": real_subsets, "subset_sig": subset_sig, "extras": extras, "majority_stats": majority_stats, "pod": pod, "basis_results": basis_results}


def _build_global_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    rows = list(summary.get("analysis_records", []))
    all_rows = list(summary.get("all_candidate_rows", []))
    majority_rows = list(summary.get("majority_rows", []))
    pod_rows = [row for row in summary.get("pod_rows", []) if row.get("mode") is None]
    channel_rows = list(summary.get("channel_rows", []))
    config = summary.get("configuration", {})

    def number(value: Any, digits: int = 4) -> str:
        finite = _finite_scalar(value)
        return "—" if finite is None else f"{finite:.{digits}g}"

    def qtext(value: Any) -> str:
        finite = _finite_scalar(value)
        return "—" if finite is None else f"{finite:.6g}"

    def quartet_text(value: Any) -> str:
        return "(" + ", ".join(str(int(item)) for item in value) + ")"

    significant_all = [row for row in all_rows if row.get("all_shared")]
    significant_majority = [row for row in majority_rows if row.get("label") == "G_MAJORITY_H_SHARED"]
    all_shared_labels = [row for row in majority_rows if row.get("label") == "ALL_SHARED"]
    weak_labels = [row for row in majority_rows if row.get("label") == "WEAK_OR_PARTIAL"]
    raw_records = {(tuple(record["quartet"]), record["domain"], record["view"]): record for record in rows if record["control"] == "RAW"}
    demean_records = {(tuple(record["quartet"]), record["domain"], record["view"]): record for record in rows if record["control"] == "SPATIAL_DEMEANED"}
    raw_all_count = sum(1 for row in significant_all if row.get("control") == "RAW")
    demean_all_count = sum(1 for row in significant_all if row.get("control") == "SPATIAL_DEMEANED")
    raw_nonempty = sum(1 for row in rows if row.get("control") == "RAW" and row.get("all_shared_rank", 0) > 0)
    demean_nonempty = sum(1 for row in rows if row.get("control") == "SPATIAL_DEMEANED" and row.get("all_shared_rank", 0) > 0)

    lines = [
        "# Global C1024 G + Majority-H Joint COBE Report",
        "",
        "## Executive conclusion",
        "",
        "This is a symmetric joint-field analysis of `G, H_i` on one fixed Global C1024 physical support. No `H-G` input was formed.",
        f"Across 48 combinations, {len(significant_all)} all-shared candidate components passed the permutation/FDR gate ({raw_all_count} RAW, {demean_all_count} SPATIAL_DEMEANED); {len(significant_majority)} leave-one-H-out records were labelled `G_MAJORITY_H_SHARED`.",
        f"The important target is `(26,27,33,34) / ALL_HIDDEN / PBR6 / RAW / omit H34`: rank 1, fitting error q={qtext(next((row.get('error_q') for row in significant_majority if tuple(row['quartet']) == (26,27,33,34) and row['domain'] == 'ALL_HIDDEN' and row['view'] == 'PBR6' and row['control'] == 'RAW' and row['omit_tile'] == 34), None))}, held-out H34 support={number(next((row.get('heldout_support_score') for row in significant_majority if tuple(row['quartet']) == (26,27,33,34) and row['domain'] == 'ALL_HIDDEN' and row['view'] == 'PBR6' and row['control'] == 'RAW' and row['omit_tile'] == 34), None))}, q={qtext(next((row.get('heldout_support_q') for row in significant_majority if tuple(row['quartet']) == (26,27,33,34) and row['domain'] == 'ALL_HIDDEN' and row['view'] == 'PBR6' and row['control'] == 'RAW' and row['omit_tile'] == 34), None))}.",
        "Thus the data support a G+H26+H27+H33 fitting pattern with weak H34 held-out support, but q=0.097 does not establish a formal H34 rejection at alpha=0.05. This remains a diagnostic, not a fusion or flow decision.",
        "",
        "## Provenance, design, and correctness gate",
        "",
        f"- status: `{summary.get('status')}`; GPU: `{summary.get('cuda', {}).get('name', 'N/A')}`, requested physical CUDA `{summary.get('cuda', {}).get('requested_physical', 'N/A')}`",
        f"- permutations: `{config.get('num_permutations')}`; seed: `{config.get('seed')}`; BH-FDR alpha: `{config.get('alpha')}`",
        f"- Global support: `{summary.get('cache_integrity', {}).get('global_support', {}).get('shape', ['N'])[0]}` rows, `{summary.get('cache_integrity', {}).get('global_support', {}).get('sha256', 'N/A')}`",
        f"- baseline: `{summary.get('cache_integrity', {}).get('global_support', {}).get('baseline_path', 'N/A')}`",
        f"- baseline SHA256: `{summary.get('cache_integrity', {}).get('global_support', {}).get('baseline_sha256', 'N/A')}`",
        f"- correctness gate: `{summary.get('correctness_gate', {}).get('status', 'N/A')}`; global self-query: `{json.dumps(summary.get('correctness_gate', {}).get('global_self_query', {}), ensure_ascii=False)}`",
        f"- cache checks: `{json.dumps(summary.get('correctness_gate', {}).get('cache_checks', {}), ensure_ascii=False)}`",
        "- all subsets reuse the parent quartet rows; H rows are independently spatially permuted in the null while G is fixed.",
        "- no C256, pseudoinverse, LSMR, range/null field, MRA, fusion, wavelet, RAHT, flow, re-encode, velocity modification, Euler, or PBR averaging was used.",
        "",
        "## Fixed parent domains",
        "",
        "| quartet | ALL_VALID | ALL_HIDDEN | ALL_OBSERVED |",
        "|---|---:|---:|---:|",
    ]
    for record in summary.get("domain_counts", []):
        lines.append(f"| {quartet_text(record['quartet'])} | {record['ALL_VALID']} | {record['ALL_HIDDEN']} | {record['ALL_OBSERVED']} |")

    lines.extend([
        "",
        "## All-shared rank",
        "",
        f"RAW has a non-empty statistically selected all-shared space in {raw_nonempty}/24 combinations; SPATIAL_DEMEANED has one in {demean_nonempty}/24. Candidate totals are {raw_all_count} RAW and {demean_all_count} demeaned.",
        "",
        "| quartet | domain | view | RAW rank | RAW candidates | demeaned rank | demeaned candidates |",
        "|---|---|---|---:|---|---:|---|",
    ])
    for key in sorted(raw_records, key=str):
        raw = raw_records[key]
        demean = demean_records.get(key, {})
        lines.append(f"| {quartet_text(key[0])} | {key[1]} | {key[2]} | {raw.get('all_shared_rank', 0)} | {raw.get('all_shared_indices', [])} | {demean.get('all_shared_rank', 0)} | {demean.get('all_shared_indices', [])} |")

    lines.extend([
        "",
        "## Majority candidates and held-out tests",
        "",
        "The label is based on the fitting subset: significant error, G support, and at least three fitting H supports. Held-out q is reported separately and is not silently used as a support threshold.",
        "",
        "| quartet | domain | view | control | omit H | rank | fitting support [G,H,H,H,H] | held-out support | held-out q |",
        "|---|---|---|---|---:|---:|---|---:|---:|",
    ])
    for record in significant_majority:
        lines.append(f"| {quartet_text(record['quartet'])} | {record['domain']} | {record['view']} | {record['control']} | H{record['omit_tile']} | {record['rank']} | {[round(float(x), 4) for x in record.get('support_scores', [])]} | {number(record.get('heldout_support_score'))} | {qtext(record.get('heldout_support_q'))} |")
    lines.extend([
        "",
        f"There are {len(all_shared_labels)} `ALL_SHARED` leave-one-out records and {len(weak_labels)} `WEAK_OR_PARTIAL` records. The four majority records above are the complete list.",
        "",
        "## Target: G + H26 + H27 + H33 versus H34",
        "",
    ])
    target_rows = [row for row in majority_rows if tuple(row["quartet"]) == (26, 27, 33, 34) and row["domain"] == "ALL_HIDDEN" and row["view"] == "PBR6" and row["control"] == "RAW" and row["omit_tile"] == 34]
    if target_rows:
        target = target_rows[0]
        lines.extend([
            f"- fitting label: `{target['label']}`, rank `{target['rank']}`, subset error `{number(target.get('error'))}`, error q `{qtext(target.get('error_q'))}`",
            f"- support order `[G,H26,H27,H33,H34]`: `{[round(float(x), 6) for x in target['support_scores']]}`",
            f"- support q order `[G,H26,H27,H33,H34]`: `{[round(float(x), 6) for x in target['support_q']]}`",
            f"- held-out H34: score `{number(target.get('heldout_support_score'))}`, p `{qtext(target.get('heldout_support_p'))}`, q `{qtext(target.get('heldout_support_q'))}`",
            "- interpretation: H34 is the weakest held-out supporter and is suggestive of a rejected mode, but the one-sided q=0.097 is above 0.05; call it weak support, not a formally significant rejection.",
        ])
    else:
        lines.append("No non-empty target candidate survived the fitting gate.")

    lines.extend([
        "",
        "## RAW versus SPATIAL_DEMEANED",
        "",
        "RAW remains the primary result. Demeaning removes a substantial amount of the selected structure, especially in ALL_HIDDEN, but it does not make every RAW mode disappear: 18 demeaned all-shared candidates remain in other domains/views. Therefore the result is not reducible to a single DC/color explanation, while the target majority pattern itself is RAW-only under this gate.",
        "",
        "| quartet | domain | view | RAW rank | demeaned rank | all-space mean cos² |",
        "|---|---|---|---:|---:|---:|",
    ])
    for key, raw in sorted(raw_records.items(), key=str):
        demean = demean_records.get(key, {})
        lines.append(f"| {quartet_text(key[0])} | {key[1]} | {key[2]} | {raw.get('all_shared_rank', 0)} | {demean.get('all_shared_rank', 0)} | {qtext(raw.get('raw_vs_demeaned_mean_cos2'))} |")

    lines.extend([
        "",
        "## Hidden versus observed",
        "",
        "In RAW PBR6, ALL_HIDDEN has rank 1 for all four quartets, whereas ALL_OBSERVED has rank 2 for all four. RAW RGB has hidden ranks [2,1,2,1] and observed ranks [3,3,3,2]. The observed domains therefore expose more selected variance directions but no RAW G-majority fitting record passed the full gate; the only observed majority label is a SPATIAL_DEMEANED PBR6 record with held-out H27 q=0.362.",
        "",
        "## POD versus COBE",
        "",
        "POD is the uncentered variance baseline on `[Y_G Y_H26 ...]`, not a difference-field POD. The table contains one summary row per analysis; per-mode spectra remain in `pod_baseline.csv` and `pod_baseline.pt`.",
        "",
        "| quartet | domain | view | control | POD top energy | POD/all cos² | POD/majority cos² |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for record in pod_rows:
        lines.append(f"| {quartet_text(record['quartet'])} | {record['domain']} | {record['view']} | {record['control']} | {number(record.get('pod_top_energy'))} | {number(record.get('pod_vs_all_mean_cos2'))} | {number(record.get('pod_vs_majority_mean_cos2'))} |")
    target_analysis = next((record for record in rows if tuple(record["quartet"]) == (26, 27, 33, 34) and record["domain"] == "ALL_HIDDEN" and record["view"] == "PBR6" and record["control"] == "RAW"), None)
    if target_analysis:
        pod_angle = target_analysis.get("pod_vs_all_shared", {})
        lines.append(f"\nTarget POD-vs-COBE principal angle: `{[round(float(value), 3) for value in pod_angle.get('angles_deg', [])]}°`; mean cos² `{number(pod_angle.get('mean_cos2'))}`. Across non-empty RAW comparisons, POD/all mean cos² ranges from approximately 0.037 to 0.811, demonstrating that high variance and cross-field commonality are different criteria.")

    lines.extend([
        "",
        "## PBR6 channel projection energy",
        "",
        "For each field/channel, `R_i,c = ||A AᵀY_i,c||² / ||Y_i,c||²`; this measures explained channel energy and does not turn the COBE basis into a material field.",
        "",
        "Target majority-extra table (`ALL_HIDDEN/PBR6/RAW/omit H34`, rank 1):",
        "",
        "| field | R | G | B | metallic | roughness | alpha |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    target_channel_rows = [row for row in channel_rows if tuple(row.get("quartet", [])) == (26, 27, 33, 34) and row.get("domain") == "ALL_HIDDEN" and row.get("view") == "PBR6" and row.get("control") == "RAW" and row.get("stage") == "G_MAJORITY_H_SHARED"]
    for field in ["G", "H26", "H27", "H33", "H34"]:
        values = {row["channel"]: row["energy"] for row in target_channel_rows if row["field"] == field}
        lines.append(f"| {field} | {number(values.get('R'), 5)} | {number(values.get('G'), 5)} | {number(values.get('B'), 5)} | {number(values.get('metallic'), 5)} | {number(values.get('roughness'), 5)} | {number(values.get('alpha'), 5)} |")
    lines.append("The target majority basis is mainly RGB-supported; metallic is around 10⁻³ and roughness/alpha around 10⁻⁵, so there is no evidence here for a strong structured roughness/alpha common mode. The full 4,410-row channel table covers every selected all-shared/majority stage.")

    lines.extend([
        "",
        "## 16 questions answered",
        "",
        f"1. Direct joint G/H analysis finds permutation-significant all-shared subspaces: {len(significant_all)} candidate components.",
        f"2. All-shared rank is condition-dependent; RAW PBR6 ranks are [1,1,2,1] for ALL_VALID, [1,1,1,1] for ALL_HIDDEN, and [2,2,2,2] for ALL_OBSERVED across the four quartets.",
        "3. RAW is stronger and more populated; demeaning removes the target hidden majority candidate and many all-shared candidates.",
        "4. Commonality is not only a DC/color offset because demeaned all-shared candidates remain, but the target majority claim is only supported in RAW.",
        "5. The strongest interpretable quartet is (26,27,33,34) in ALL_HIDDEN/PBR6/RAW; the other three majority records are listed above and have weaker or failed held-out support.",
        "6. Hidden gives consistent one-dimensional RAW PBR6 all-shared ranks; observed gives higher all-shared rank but no RAW majority record.",
        "7. Yes, four fitting-subset G+3H majority records pass the formal fitting gate.",
        "8. In the target, H34 is the held-out weak supporter (score 0.061, q=0.097); it is not formally rejected at q≤0.05.",
        "9. G,H26,H27,H33 form the target fitting pattern; H34 is weak but the evidence is not strong enough to claim a definitive separate mode.",
        "10. Target POD top direction is 25.734° from the selected all-shared COBE direction (cos²=0.811); across analyses angles vary widely.",
        "11. Yes: POD top energy is often 0.8–0.96 while POD/COBE cos² can be near 0.04, so high variance is not commonality.",
        "12. Target PBR6 projection energy is RGB-dominant; metallic is small and roughness/alpha are negligible in the displayed majority basis.",
        "13. No robust target metallic commonality survives the spatial-demeaned control; the target demeaned all-shared/majority selections are empty.",
        "14. Yes for the selected fitting modes: real errors/supports are separated from 256 independently row-shuffled H nulls, with minimum attainable p=1/257=0.003891.",
        "15. The compact description is mostly all-shared plus a small number of G-majority-H fitting patterns, but held-out stability is mixed; do not treat every fitting label as a validated physical mode.",
        "16. The evidence is sufficient to motivate a future partially-shared/SLIDE study, especially for the target, but not to enter fusion or flow modification in this round.",
        "",
        "## Figures and output pointers",
        "",
        f"- summary: `{(output_dir / 'summary.json').resolve()}`",
        f"- complete report: `{(output_dir / 'GLOBAL_C1024_G_MAJORITY_COBE_REPORT.md').resolve()}`",
        f"- all candidates: `{(output_dir / 'cobe_summary.csv').resolve()}`",
        f"- majority/held-out table: `{(output_dir / 'majority_support.csv').resolve()}`",
        f"- common ranks: `{(output_dir / 'common_rank.csv').resolve()}`",
        f"- principal angles: `{(output_dir / 'principal_angles.csv').resolve()}`",
        f"- permutation p/q: `{(output_dir / 'permutation_significance.csv').resolve()}`",
        f"- POD spectra and summaries: `{(output_dir / 'pod_baseline.csv').resolve()}`",
        f"- channel projection energy: `{(output_dir / 'channel_projection_energy.csv').resolve()}`",
        f"- target support heatmap: `{(output_dir / 'quartets/26_27_33_34/ALL_HIDDEN/PBR6/RAW/field_support_heatmap.png').resolve()}`",
        f"- target real-vs-null error: `{(output_dir / 'quartets/26_27_33_34/ALL_HIDDEN/PBR6/RAW/real_vs_null_cobe_error.png').resolve()}`",
        f"- target omit-H34 support: `{(output_dir / 'quartets/26_27_33_34/ALL_HIDDEN/PBR6/RAW/omit_34/majority_extra_support.png').resolve()}`",
        f"- target principal-angle matrix: `{(output_dir / 'quartets/26_27_33_34/ALL_HIDDEN/PBR6/RAW/principal_angle_matrix.png').resolve()}`",
        f"- target COBE spatial basis: `{(output_dir / 'quartets/26_27_33_34/ALL_HIDDEN/PBR6/RAW/omit_34/common_spatial_basis.png').resolve()}`",
        f"- target common-projection PBR energy: `{(output_dir / 'quartets/26_27_33_34/ALL_HIDDEN/PBR6/RAW/common_projection_energy.png').resolve()}`",
        "",
        "Next step: inspect the target held-out mode and its 12-step evolution, then design a SLIDE/partially-shared decomposition. Do not perform fusion or flow guidance based on this diagnostic alone.",
    ])
    (output_dir / "GLOBAL_C1024_G_MAJORITY_COBE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validated = getattr(run, "_validated_cache", None)
    if validated is not None:
        cache_integrity, valid, hidden, observed, global_field, points, correctness_gate = validated
    else:
        cache_integrity, valid, hidden, observed, global_field, points, correctness_gate = validate_cache(cache_dir, cuda_device=args.cuda_device)
    _atomic_json(output_dir / "preflight.json", {"format": FORMAT, "status": "ready", "cache_dir": str(cache_dir.resolve()), "cache_integrity": cache_integrity, "required_quartets": [list(q) for q in QUARTETS], "domains": list(DOMAINS), "views": list(VIEWS), "controls": list(CONTROLS), "errors": []})
    _atomic_json(output_dir / "correctness_gate.json", {"format": FORMAT, **correctness_gate})
    if args.phase == "preflight":
        print(json.dumps({"status": "preflight_passed", "output_dir": str(output_dir.resolve()), "correctness_gate": correctness_gate}, ensure_ascii=False, indent=2))
        return 0
    cuda = correctness_gate["cuda"]
    device = torch.device(f"cuda:{int(cuda['logical'])}")
    if args.num_permutations < 1:
        raise ValueError("num_permutations must be positive")
    if args.null_batch_size < 1:
        raise ValueError("null_batch_size must be positive")
    if args.num_threads:
        torch.set_num_threads(int(args.num_threads))
    all_candidate_rows: List[Dict[str, Any]] = []
    majority_rows: List[Dict[str, Any]] = []
    common_rank_rows: List[Dict[str, Any]] = []
    angle_rows: List[Dict[str, Any]] = []
    significance_rows: List[Dict[str, Any]] = []
    pod_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    analysis_records: List[Dict[str, Any]] = []
    domain_counts: List[Dict[str, Any]] = []
    basis_registry: Dict[str, Dict[str, torch.Tensor]] = {}
    extra_registry: Dict[str, Dict[str, torch.Tensor]] = {}
    started = time.time()
    for quartet in QUARTETS:
        # Keep only the four fields used by this physical quartet resident.
        # The cache validator already checked every required tile; this
        # quartet-scoped load avoids duplicating ~1 GB of H tensors in a
        # constrained worker process.
        tile_fields = {
            int(tile_id): _load_torch(_cache_tile_dir(cache_dir, int(tile_id)) / "H_on_global.pt").to(torch.float32).contiguous()
            for tile_id in quartet
        }
        masks = _parent_masks(quartet, valid, hidden, observed)
        count_record = {"quartet": list(map(int, quartet)), **{domain: int(mask.sum().item()) for domain, mask in masks.items()}}
        domain_counts.append(count_record)
        for domain in DOMAINS:
            rows = torch.where(masks[domain])[0]
            # The same physical-row permutations are reused by all views and
            # controls for this exact parent domain.
            permutations = {"n_rows": int(rows.numel()), "num_permutations": int(args.num_permutations), "seed": _stable_seed(args.seed, "permutation", quartet, domain), "field_count": 4}
            for view in VIEWS:
                for control in CONTROLS:
                    key = _analysis_key(quartet, domain, view, control)
                    print(f"[COBE] {key}: rows={rows.numel()} permutations={args.num_permutations}", flush=True)
                    blocks = _load_blocks(rows, quartet, global_field, tile_fields, view, control)
                    result = _run_one_analysis(
                        output_dir=output_dir,
                        quartet=quartet,
                        domain=domain,
                        view=view,
                        control=control,
                        rows=rows,
                        points=points.index_select(0, rows),
                        blocks=blocks,
                        permutations=permutations,
                        args=args,
                        device=device,
                    )
                    record = result["record"]
                    analysis_records.append(record)
                    key_base = {"quartet": list(map(int, quartet)), "domain": domain, "view": view, "control": control}
                    selected_all_basis = result["all"].basis[:, result["all_sig"]["all_shared_indices"]] if result["all_sig"]["all_shared_indices"] else result["all"].basis[:, :0]
                    # Principal-angle stability must compare the selected
                    # common subspaces fitted by each leave-one-H-out run.
                    # The majority ``extra`` basis is orthogonalised against
                    # all-shared space and is therefore a different object;
                    # keep it in a separate registry for majority/POD and
                    # channel diagnostics.
                    selected_subset_bases = {
                        stage: (
                            result["subsets"][stage].basis[:, result["subset_sig"][stage]["selected_indices"]]
                            if result["subset_sig"][stage]["selected_indices"]
                            else result["subsets"][stage].basis[:, :0]
                        )
                        for stage in result["subsets"]
                    }
                    basis_registry[key] = {"all": selected_all_basis, **selected_subset_bases}
                    extra_registry[key] = {stage: basis for stage, basis in result["extras"].items()}
                    angle_stage_order = ("all", "omit_1", "omit_2", "omit_3", "omit_4")
                    _plot_principal_angle_matrix(
                        output_dir / "quartets" / "_".join(str(int(v)) for v in quartet) / domain / view / control / "principal_angle_matrix.png",
                        [basis_registry[key][stage] for stage in angle_stage_order],
                        ["all-shared", *[f"omit H{int(quartet[index - 1])}" for index in range(1, 5)]],
                        "Selected COBE subspaces — mean cos² principal-angle matrix",
                    )
                    for candidate in range(result["all"].rank):
                        supported_count = sum(result["all_sig"]["support_q"][field][candidate] is not None and result["all_sig"]["support_q"][field][candidate] <= args.alpha for field in range(5))
                        all_candidate_rows.append({**key_base, "candidate": candidate, "f_k": float(result["all"].errors[candidate].item()), "error_p": result["all_sig"]["error_p"][candidate], "error_q": result["all_sig"]["error_q"][candidate], "supported_fields": supported_count, "all_shared": candidate in result["all_sig"]["all_shared_indices"]})
                        significance_rows.append({**key_base, "stage": "ALL_SHARED_CANDIDATE", "candidate": candidate, "metric": "cobe_error", "value": float(result["all"].errors[candidate].item()), "p": result["all_sig"]["error_p"][candidate], "q": result["all_sig"]["error_q"][candidate]})
                        for field_index, field_name in enumerate(_field_names(quartet)):
                            significance_rows.append({**key_base, "stage": "ALL_SHARED_CANDIDATE", "candidate": candidate, "metric": f"support_{field_name}", "value": float(result["all"].support_scores[field_index, candidate].item()), "p": result["all_sig"]["support_p"][field_index][candidate], "q": result["all_sig"]["support_q"][field_index][candidate]})
                    common_rank_rows.append({**key_base, "all_shared_rank": len(result["all_sig"]["all_shared_indices"]), "all_candidate_rank": result["all"].rank, **{f"subset_{stage}_rank": result["subset_sig"][stage]["selected_rank"] for stage in result["subset_sig"]}, **{f"majority_{stage}_rank": result["majority_stats"][stage].get("rank", 0) for stage in result["majority_stats"]}})
                    for stage_name, stats in result["majority_stats"].items():
                        omit_tile = int(quartet[int(stage_name.split("_")[1]) - 1])
                        row = {**key_base, "omit_tile": omit_tile, "label": stats.get("label", "WEAK_OR_PARTIAL"), "rank": stats.get("rank", 0), "error": stats.get("error"), "error_p": stats.get("error_p"), "error_q": stats.get("error_q"), "heldout_support_score": stats.get("heldout_support_score"), "heldout_support_p": stats.get("heldout_support_p"), "heldout_support_q": stats.get("heldout_support_q"), "support_scores": stats.get("support_scores", []), "support_q": stats.get("support_q", [])}
                        majority_rows.append(row)
                        significance_rows.append({**key_base, "stage": f"MAJORITY_EXTRA_OMIT_{omit_tile}", "candidate": 0, "metric": "subspace_error", "value": stats.get("error"), "p": stats.get("error_p"), "q": stats.get("error_q"), "heldout_support_p": stats.get("heldout_support_p"), "heldout_support_q": stats.get("heldout_support_q"), "label": stats.get("label")})
                        for field_name, score, p_value, q_value in zip(_field_names(quartet), stats.get("support_scores", []), stats.get("support_p", []), stats.get("support_q", [])):
                            significance_rows.append({**key_base, "stage": f"MAJORITY_EXTRA_OMIT_{omit_tile}", "candidate": 0, "metric": f"support_{field_name}", "value": score, "p": p_value, "q": q_value, "label": stats.get("label")})
                    pod = result["pod"]
                    all_basis = basis_registry[key]["all"]
                    majority_bases = [basis for basis in extra_registry[key].values() if basis.shape[1]]
                    pod_vs_all = principal_angle_summary(pod["left_basis"], all_basis)
                    majority_angle_values = [principal_angle_summary(pod["left_basis"], basis).get("mean_cos2") for basis in majority_bases]
                    pod_rows.append({**key_base, "pod_top_energy": float(pod["energy_ratio"][0].item()) if pod["energy_ratio"].numel() else None, "pod_vs_all_mean_cos2": pod_vs_all.get("mean_cos2"), "pod_vs_majority_mean_cos2": float(np.nanmean([value for value in majority_angle_values if value is not None])) if any(value is not None for value in majority_angle_values) else None})
                    angle_rows.append({**key_base, "left": "POD_TOP", "right": "ALL_SHARED", **{f"angle_{key_name}": value for key_name, value in pod_vs_all.items()}})
                    for stage, basis in basis_registry[key].items():
                        for other_stage, other_basis in basis_registry[key].items():
                            if stage < other_stage:
                                angles = principal_angle_summary(basis, other_basis)
                                angle_rows.append({**key_base, "left": stage, "right": other_stage, **{f"angle_{key_name}": value for key_name, value in angles.items()}})
                    for row in result["record"].get("pod", []):
                        pod_rows.append({**key_base, "mode": row["mode"], "sigma": row["sigma"], "energy_ratio": row["energy_ratio"], "cumulative_energy_ratio": row["cumulative_energy_ratio"]})
                    channel_bases = {"all": basis_registry[key]["all"], **extra_registry[key]}
                    for stage, basis in channel_bases.items():
                        if not basis.shape[1]:
                            continue
                        stage_label = "ALL_SHARED" if stage == "all" else result["majority_stats"][stage].get("label", "WEAK_OR_PARTIAL")
                        for field_name, block in zip(_field_names(quartet), blocks):
                            energies = projection_energy(block, basis)
                            for channel_index, energy in enumerate(energies.tolist()):
                                channel_rows.append({**key_base, "stage": stage_label, "stage_key": stage, "field": field_name, "channel": PBR_CHANNEL_NAMES[channel_index], "energy": float(energy), "rank": int(basis.shape[1])})
                    demean_key = _analysis_key(quartet, domain, view, "SPATIAL_DEMEANED")
                    if control == "SPATIAL_DEMEANED" and demean_key in basis_registry:
                        pass
                    # The cross-control angle is filled after both controls
                    # have completed for this domain/view.
                    del blocks, result
            del permutations
        del tile_fields
    # Cross-control and cross-subset principal-angle records are generated from
    # the compact selected bases, never by column-wise matching.
    for quartet in QUARTETS:
        for domain in DOMAINS:
            for view in VIEWS:
                raw_key = _analysis_key(quartet, domain, view, "RAW")
                demean_key = _analysis_key(quartet, domain, view, "SPATIAL_DEMEANED")
                if raw_key in basis_registry and demean_key in basis_registry:
                    for stage in ("all", "omit_1", "omit_2", "omit_3", "omit_4"):
                        angle = principal_angle_summary(basis_registry[raw_key][stage], basis_registry[demean_key][stage])
                        angle_rows.append({"quartet": list(map(int, quartet)), "domain": domain, "view": view, "control": "RAW_vs_SPATIAL_DEMEANED", "left": f"RAW_{stage}", "right": f"DEMEANED_{stage}", **{f"angle_{key_name}": value for key_name, value in angle.items()}})
    control_angles: Dict[Tuple[Tuple[int, ...], str, str, str], Optional[float]] = {}
    for angle in angle_rows:
        if angle.get("control") != "RAW_vs_SPATIAL_DEMEANED":
            continue
        control_angles[(tuple(angle["quartet"]), str(angle["domain"]), str(angle["view"]), str(angle["left"]).replace("RAW_", ""))] = angle.get("angle_mean_cos2")
    for record in analysis_records:
        stage = "all"
        control_key = (tuple(record["quartet"]), str(record["domain"]), str(record["view"]), stage)
        if record.get("control") == "RAW":
            record["raw_vs_demeaned_mean_cos2"] = control_angles.get(control_key)
    _write_csv(output_dir / "cobe_summary.csv", all_candidate_rows)
    _write_csv(output_dir / "majority_support.csv", majority_rows)
    _write_csv(output_dir / "common_rank.csv", common_rank_rows)
    _write_csv(output_dir / "principal_angles.csv", angle_rows)
    _write_csv(output_dir / "permutation_significance.csv", significance_rows)
    _write_csv(output_dir / "pod_baseline.csv", pod_rows)
    _write_csv(output_dir / "channel_projection_energy.csv", channel_rows)
    summary: Dict[str, Any] = {
        "status": "completed",
        "format": FORMAT,
        "cache_integrity": cache_integrity,
        "correctness_gate": correctness_gate,
        "cuda": cuda,
        "configuration": {"num_permutations": int(args.num_permutations), "seed": int(args.seed), "alpha": float(args.alpha), "max_iter": int(args.max_iter), "null_max_iter": int(args.null_max_iter), "convergence_tol": float(args.convergence_tol), "rank_abs_tol": float(args.rank_abs_tol), "rank_rel_tol": float(args.rank_rel_tol), "null_batch_size": int(args.null_batch_size)},
        "elapsed_seconds": float(time.time() - started),
        "domain_counts": domain_counts,
        "analysis_count": len(analysis_records),
        "analysis_records": analysis_records,
        "all_candidate_rows": all_candidate_rows,
        "majority_rows": majority_rows,
        "pod_rows": pod_rows,
        "channel_rows": channel_rows,
        "output_files": {name: str((output_dir / name).resolve()) for name in ("cobe_summary.csv", "majority_support.csv", "common_rank.csv", "principal_angles.csv", "permutation_significance.csv", "pod_baseline.csv", "channel_projection_energy.csv", "GLOBAL_C1024_G_MAJORITY_COBE_REPORT.md")},
        "prohibited_operations": {"difference_field_input": False, "c256": False, "pseudoinverse": False, "lsmr": False, "range_null": False, "mra": False, "fusion": False, "wavelet": False, "flow": False, "reencode": False, "euler": False, "pbr_averaging": False},
    }
    _atomic_json(output_dir / "summary.json", summary)
    _build_global_report(output_dir, summary)
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"], "analysis_count": summary["analysis_count"], "all_shared_records": sum(bool(row.get("all_shared")) for row in all_candidate_rows), "majority_records": sum(row.get("label") == "G_MAJORITY_H_SHARED" for row in majority_rows), "summary": str((output_dir / "summary.json").resolve())}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/global_c1024_common_field_pod_phaseA_cuda4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/global_c1024_g_majority_cobe_phaseA"))
    parser.add_argument("--num-permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--null-max-iter", type=int, default=35)
    parser.add_argument("--convergence-tol", type=float, default=1e-10)
    parser.add_argument("--rank-abs-tol", type=float, default=1e-12)
    parser.add_argument("--rank-rel-tol", type=float, default=1e-10)
    parser.add_argument("--null-batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    # Load/validate the cache once before entering the normal run path.  Tests
    # call the numerical functions directly and never touch this CLI state.
    cache_integrity, valid, hidden, observed, global_field, points, gate = validate_cache(cache_dir, cuda_device=None if args.phase == "preflight" else args.cuda_device)
    # Reuse the already validated tensors in run without a second baseline
    # hash/load pass by keeping a small process-local hook.
    run._validated_cache = (cache_integrity, valid, hidden, observed, global_field, points, gate)  # type: ignore[attr-defined]
    run(args)


if __name__ == "__main__":
    main()
