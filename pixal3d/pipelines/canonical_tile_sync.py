"""Canonical-space operators for projective Pixal3D tile synchronization.

The operators in this module deliberately do not average model velocities.
They implement the four pieces needed by the training-free 2048 pipeline:

* sparse common atoms shared by global and projective local token cells;
* stateless spatial white noise restricted to those token cells;
* robust local-minus-global clean-endpoint fusion;
* a coverage-aware high-pass whose restriction to every global parent is zero.

The common atom lattice is sparse.  A target cell is materialized only when it
is covered by an active global token or by an exact projective local footprint.
Local footprints are rasterized by transforming their corners for a candidate
bounding box and then applying the exact inverse transform to every candidate
atom center.  No fixed ``/ 4`` camera approximation, coordinate clamp, dense
``2048**3`` allocation, or sparse-row correspondence assumption is used.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def endpoint_indices_to_q(indices: torch.Tensor, resolution: int) -> torch.Tensor:
    """Convert endpoint-lattice indices to normalized canonical coordinates."""
    if int(resolution) < 2:
        raise ValueError("endpoint resolution must be at least two")
    return (
        indices.to(torch.float64) * (2.0 / float(int(resolution) - 1)) - 1.0
    )


def q_to_endpoint_indices(q: torch.Tensor, resolution: int) -> torch.Tensor:
    """Quantize normalized coordinates without clipping them to the cube."""
    if int(resolution) < 2:
        raise ValueError("endpoint resolution must be at least two")
    return torch.round(
        (q.to(torch.float64) + 1.0) * (float(int(resolution) - 1) / 2.0)
    ).to(torch.int64)


def target_cell_centers(indices: torch.Tensor, resolution: int) -> torch.Tensor:
    """Return center coordinates for a regular target-cell lattice."""
    if int(resolution) < 1:
        raise ValueError("target resolution must be positive")
    return (
        (indices.to(torch.float64) + 0.5)
        * (2.0 / float(int(resolution)))
        - 1.0
    )


def endpoint_cell_bounds(
    indices: torch.Tensor,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Voronoi cell bounds of endpoint-lattice coordinates inside [-1, 1]."""
    indices = indices.to(torch.int64)
    if bool(((indices < 0) | (indices >= int(resolution))).any().item()):
        raise ValueError("endpoint indices lie outside the lattice")
    centers = endpoint_indices_to_q(indices, int(resolution))
    half = 1.0 / float(int(resolution) - 1)
    lower = torch.where(
        indices == 0,
        torch.full_like(centers, -1.0),
        centers - half,
    )
    upper = torch.where(
        indices == int(resolution) - 1,
        torch.full_like(centers, 1.0),
        centers + half,
    )
    return lower, upper


def _validate_coords(coords: torch.Tensor, resolution: int, label: str) -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{label} coordinates must have shape [N,4]")
    if coords.shape[0] < 1:
        raise ValueError(f"{label} support is empty")
    xyz = coords[:, 1:4].to(torch.int64)
    valid = (
        (coords[:, 0].to(torch.int64) == 0)
        & (xyz >= 0).all(dim=1)
        & (xyz < int(resolution)).all(dim=1)
    )
    if not bool(valid.all().item()):
        raise ValueError(
            f"{label} contains {int((~valid).sum().item())} invalid rows"
        )
    if torch.unique(coords.to(torch.int64), dim=0).shape[0] != coords.shape[0]:
        raise ValueError(f"{label} support contains duplicate coordinates")


def _linearize_xyz(xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = xyz.to(torch.int64)
    return (
        (xyz[:, 0] * int(resolution) + xyz[:, 1]) * int(resolution)
        + xyz[:, 2]
    )


def _delinearize_xyz(codes: torch.Tensor, resolution: int) -> torch.Tensor:
    codes = codes.to(torch.int64)
    z = torch.remainder(codes, int(resolution))
    quotient = torch.div(codes, int(resolution), rounding_mode="floor")
    y = torch.remainder(quotient, int(resolution))
    x = torch.div(quotient, int(resolution), rounding_mode="floor")
    return torch.stack([x, y, z], dim=1)


def _target_axis_parent_table(
    endpoint_resolution: int,
    target_resolution: int,
) -> List[torch.Tensor]:
    axis = torch.arange(int(target_resolution), dtype=torch.int64)
    centers = target_cell_centers(axis, int(target_resolution))
    parents = q_to_endpoint_indices(centers, int(endpoint_resolution))
    return [
        torch.where(parents == index)[0]
        for index in range(int(endpoint_resolution))
    ]


def _enumerate_global_memberships(
    coords: torch.Tensor,
    endpoint_resolution: int,
    target_resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enumerate target atoms in every active global endpoint cell."""
    tables = _target_axis_parent_table(
        int(endpoint_resolution), int(target_resolution)
    )
    atom_codes: List[torch.Tensor] = []
    parent_rows: List[torch.Tensor] = []
    coords_cpu = coords.detach().to(device="cpu", dtype=torch.int64)
    for row, xyz in enumerate(coords_cpu[:, 1:4].tolist()):
        axes = [tables[int(value)] for value in xyz]
        if any(axis.numel() == 0 for axis in axes):
            continue
        grid = torch.cartesian_prod(*axes)
        if grid.ndim == 1:
            grid = grid[None]
        atom_codes.append(_linearize_xyz(grid, int(target_resolution)))
        parent_rows.append(
            torch.full((grid.shape[0],), row, dtype=torch.int64)
        )
    if not atom_codes:
        raise RuntimeError("global support covers no target atoms")
    codes = torch.cat(atom_codes)
    rows = torch.cat(parent_rows)
    if torch.unique(codes).shape[0] != codes.shape[0]:
        raise RuntimeError("global endpoint cells produced overlapping atoms")
    return codes, rows


def _corner_offsets(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=device,
        dtype=torch.bool,
    )


def _rasterize_local_memberships(
    coords: torch.Tensor,
    *,
    endpoint_resolution: int,
    target_resolution: int,
    local_to_global: TensorTransform,
    global_to_local: TensorTransform,
    chunk_size: int,
    boundary_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Rasterize exact mapped local footprints on a sparse target lattice."""
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    coords_cpu = coords.detach().to(device="cpu", dtype=torch.int64)
    local_xyz = coords_cpu[:, 1:4]
    lower, upper = endpoint_cell_bounds(
        local_xyz, int(endpoint_resolution)
    )
    offsets = _corner_offsets(torch.device("cpu"))
    all_codes: List[torch.Tensor] = []
    all_rows: List[torch.Tensor] = []
    all_fractions: List[torch.Tensor] = []
    candidate_count = 0
    outside_candidate_count = 0
    max_roundtrip = 0.0

    for begin in range(0, coords_cpu.shape[0], int(chunk_size)):
        end = min(begin + int(chunk_size), coords_cpu.shape[0])
        lo = lower[begin:end]
        hi = upper[begin:end]
        corners_local = torch.where(
            offsets[None],
            hi[:, None, :],
            lo[:, None, :],
        ).reshape(-1, 3)
        corners_global = local_to_global(corners_local)
        if corners_global.shape != corners_local.shape:
            raise RuntimeError("local_to_global returned an invalid shape")
        corners_global = corners_global.to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(corners_global).all().item()):
            raise RuntimeError("local_to_global produced non-finite corners")
        corners_global = corners_global.reshape(end - begin, 8, 3)
        minimum = corners_global.amin(dim=1)
        maximum = corners_global.amax(dim=1)

        # Target cell center q(j) = -1 + 2(j+.5)/T.  The one-cell margin
        # protects against a curved z-dependent footprint whose extrema lie
        # between corner planes; exact inverse membership below is decisive.
        j_min = torch.floor(
            (minimum + 1.0) * (float(target_resolution) / 2.0) - 0.5
        ).to(torch.int64) - 1
        j_max = torch.ceil(
            (maximum + 1.0) * (float(target_resolution) / 2.0) - 0.5
        ).to(torch.int64) + 1
        j_min = j_min.clamp(0, int(target_resolution) - 1)
        j_max = j_max.clamp(0, int(target_resolution) - 1)
        widths = (j_max - j_min + 1).clamp_min(0)
        max_width = widths.amax(dim=0)
        candidate_offsets = torch.cartesian_prod(
            torch.arange(int(max_width[0].item()), dtype=torch.int64),
            torch.arange(int(max_width[1].item()), dtype=torch.int64),
            torch.arange(int(max_width[2].item()), dtype=torch.int64),
        )
        if candidate_offsets.ndim == 1:
            candidate_offsets = candidate_offsets[None]
        candidate_xyz = j_min[:, None, :] + candidate_offsets[None]
        in_bbox = (candidate_xyz <= j_max[:, None, :]).all(dim=2)
        source_rows = (
            torch.arange(begin, end, dtype=torch.int64)[:, None]
            .expand(-1, candidate_offsets.shape[0])
        )
        candidate_xyz = candidate_xyz[in_bbox]
        source_rows = source_rows[in_bbox]
        candidate_count += int(candidate_xyz.shape[0])
        if candidate_xyz.numel() == 0:
            continue

        cell_width = 2.0 / float(target_resolution)
        cell_lo = (
            -1.0
            + candidate_xyz.to(torch.float64) * cell_width
        )
        cell_hi = cell_lo + cell_width
        source_minimum = minimum.index_select(0, source_rows - begin)
        source_maximum = maximum.index_select(0, source_rows - begin)
        intersection_width = (
            torch.minimum(cell_hi, source_maximum)
            - torch.maximum(cell_lo, source_minimum)
        ).clamp_min(0.0)
        fractions = (intersection_width / cell_width).prod(dim=1)
        intersects = fractions > float(boundary_epsilon)
        outside_candidate_count += int((~intersects).sum().item())
        if not bool(intersects.any().item()):
            continue
        candidate_xyz = candidate_xyz[intersects]
        source_rows = source_rows[intersects]
        fractions = fractions[intersects].clamp(0.0, 1.0)

        # The exact center round-trip catches sign/camera-convention mistakes.
        source_center_local = endpoint_indices_to_q(
            coords_cpu[source_rows, 1:4], int(endpoint_resolution)
        )
        source_center_global = local_to_global(source_center_local).to(
            torch.float64
        )
        center_roundtrip = global_to_local(source_center_global).to(
            torch.float64
        )
        error = (center_roundtrip - source_center_local).abs()
        max_roundtrip = max(max_roundtrip, float(error.max().item()))
        all_codes.append(
            _linearize_xyz(candidate_xyz, int(target_resolution))
        )
        all_rows.append(source_rows)
        all_fractions.append(fractions)

    if not all_codes:
        raise RuntimeError("mapped local support covers no target atoms")
    codes = torch.cat(all_codes)
    rows = torch.cat(all_rows)
    fractions = torch.cat(all_fractions).to(torch.float32)
    pair_base = int(coords_cpu.shape[0])
    pair_codes = codes * pair_base + rows
    order = torch.argsort(pair_codes)
    pair_codes = pair_codes[order]
    fractions = fractions[order]
    if bool((pair_codes[1:] == pair_codes[:-1]).any().item()):
        raise RuntimeError("local rasterizer produced duplicate token/atom pairs")
    codes = torch.div(pair_codes, pair_base, rounding_mode="floor")
    rows = torch.remainder(pair_codes, pair_base)
    counts = torch.bincount(rows, minlength=coords_cpu.shape[0])
    mass = torch.zeros(coords_cpu.shape[0], dtype=torch.float32)
    mass.index_add_(0, rows, fractions)
    if bool((counts == 0).any().item()):
        empty = torch.where(counts == 0)[0]
        raise RuntimeError(
            "mapped local cells lost all atoms; first rows "
            f"{empty[:16].tolist()}"
        )
    return codes, rows, fractions, {
        "candidate_atom_memberships": float(candidate_count),
        "rejected_bbox_candidates": float(outside_candidate_count),
        "accepted_atom_memberships": float(codes.shape[0]),
        "max_q_roundtrip_error": float(max_roundtrip),
        "min_atoms_per_token": float(counts.min().item()),
        "max_atoms_per_token": float(counts.max().item()),
        "mean_atoms_per_token": float(counts.float().mean().item()),
        "min_atom_mass_per_token": float(mass.min().item()),
        "max_atom_mass_per_token": float(mass.max().item()),
        "mean_atom_mass_per_token": float(mass.mean().item()),
        "overlap_estimator": (
            "exact-transform corner AABB intersected with target cells"
        ),
    }


def _lookup_sorted_codes(
    sorted_codes: torch.Tensor,
    query_codes: torch.Tensor,
) -> torch.Tensor:
    positions = torch.searchsorted(sorted_codes, query_codes)
    valid = positions < sorted_codes.shape[0]
    safe = positions.clamp_max(sorted_codes.shape[0] - 1)
    valid &= sorted_codes.index_select(0, safe) == query_codes
    return torch.where(valid, positions, torch.full_like(positions, -1))


@dataclass(frozen=True)
class LocalAtomMapping:
    """Sparse incidence table between one tile's tokens and common atoms."""

    tile_id: int
    coords: torch.Tensor
    atom_indices: torch.Tensor
    token_indices: torch.Tensor
    overlap_fractions: torch.Tensor
    token_atom_counts: torch.Tensor
    token_atom_mass: torch.Tensor
    diagnostics: Mapping[str, float]

    def to(self, device: torch.device) -> "LocalAtomMapping":
        return LocalAtomMapping(
            tile_id=int(self.tile_id),
            coords=self.coords.to(device),
            atom_indices=self.atom_indices.to(device),
            token_indices=self.token_indices.to(device),
            overlap_fractions=self.overlap_fractions.to(device),
            token_atom_counts=self.token_atom_counts.to(device),
            token_atom_mass=self.token_atom_mass.to(device),
            diagnostics=dict(self.diagnostics),
        )


@dataclass(frozen=True)
class CommonAtomSpace:
    """Sparse common-cell complex for one flow stage."""

    stage: str
    global_resolution: int
    local_resolution: int
    target_resolution: int
    atom_coords: torch.Tensor
    atom_ids: torch.Tensor
    atom_volume: float
    global_coords: torch.Tensor
    global_atom_indices: torch.Tensor
    global_parent_rows: torch.Tensor
    global_token_atom_counts: torch.Tensor
    atom_reference_parent: torch.Tensor
    local_mappings: Tuple[LocalAtomMapping, ...]
    diagnostics: Mapping[str, object]

    @property
    def atom_count(self) -> int:
        return int(self.atom_coords.shape[0])

    def to(self, device: torch.device) -> "CommonAtomSpace":
        return CommonAtomSpace(
            stage=self.stage,
            global_resolution=int(self.global_resolution),
            local_resolution=int(self.local_resolution),
            target_resolution=int(self.target_resolution),
            atom_coords=self.atom_coords.to(device),
            atom_ids=self.atom_ids.to(device),
            atom_volume=float(self.atom_volume),
            global_coords=self.global_coords.to(device),
            global_atom_indices=self.global_atom_indices.to(device),
            global_parent_rows=self.global_parent_rows.to(device),
            global_token_atom_counts=self.global_token_atom_counts.to(device),
            atom_reference_parent=self.atom_reference_parent.to(device),
            local_mappings=tuple(
                mapping.to(device) for mapping in self.local_mappings
            ),
            diagnostics=dict(self.diagnostics),
        )

    def lift_global(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[0] != self.global_coords.shape[0]:
            raise ValueError("global values are not aligned with global support")
        return values.index_select(0, self.atom_reference_parent)

    def restrict_global(self, atom_values: torch.Tensor) -> torch.Tensor:
        """Volume restriction over the complete augmented parent partition."""
        if atom_values.ndim != 2 or atom_values.shape[0] != self.atom_count:
            raise ValueError("atom values are not aligned with atom space")
        parents = self.atom_reference_parent
        result = torch.zeros(
            (self.global_coords.shape[0], atom_values.shape[1]),
            dtype=atom_values.dtype,
            device=atom_values.device,
        )
        result.index_add_(0, parents, atom_values)
        counts = torch.bincount(
            parents, minlength=self.global_coords.shape[0]
        ).to(device=atom_values.device, dtype=atom_values.dtype)
        return result / counts[:, None].clamp_min(1.0)

    def restrict_local(
        self,
        atom_values: torch.Tensor,
        local_index: int,
    ) -> torch.Tensor:
        if atom_values.ndim != 2 or atom_values.shape[0] != self.atom_count:
            raise ValueError("atom values are not aligned with atom space")
        mapping = self.local_mappings[int(local_index)]
        result = torch.zeros(
            (mapping.coords.shape[0], atom_values.shape[1]),
            dtype=atom_values.dtype,
            device=atom_values.device,
        )
        result.index_add_(
            0,
            mapping.token_indices,
            atom_values.index_select(0, mapping.atom_indices)
            * mapping.overlap_fractions[:, None].to(atom_values.dtype),
        )
        mass = mapping.token_atom_mass.to(
            device=atom_values.device, dtype=atom_values.dtype
        )
        return result / mass[:, None].clamp_min(1e-12)


@dataclass(frozen=True)
class C128MasterAtomSpace:
    """C128 master cells with exact local-C64 footprint incidences.

    Unlike :class:`CommonAtomSpace`, the global endpoint already has one
    independently generated feature per target atom.  ``coarse_parent`` is
    therefore used only by the low-frequency projector; it is never used to
    broadcast a C64 feature onto the C128 master.
    """

    stage: str
    target_resolution: int
    local_resolution: int
    atom_coords: torch.Tensor
    atom_ids: torch.Tensor
    coarse_parent: torch.Tensor
    coarse_parent_count: int
    local_mappings: Tuple[LocalAtomMapping, ...]
    diagnostics: Mapping[str, object]

    @property
    def atom_count(self) -> int:
        return int(self.atom_coords.shape[0])

    def to(self, device: torch.device) -> "C128MasterAtomSpace":
        return C128MasterAtomSpace(
            stage=str(self.stage),
            target_resolution=int(self.target_resolution),
            local_resolution=int(self.local_resolution),
            atom_coords=self.atom_coords.to(device),
            atom_ids=self.atom_ids.to(device),
            coarse_parent=self.coarse_parent.to(device),
            coarse_parent_count=int(self.coarse_parent_count),
            local_mappings=tuple(
                mapping.to(device) for mapping in self.local_mappings
            ),
            diagnostics=dict(self.diagnostics),
        )

    def restrict_local(
        self,
        atom_values: torch.Tensor,
        local_index: int,
    ) -> torch.Tensor:
        if atom_values.ndim != 2 or atom_values.shape[0] != self.atom_count:
            raise ValueError("atom values are not aligned with C128 master")
        mapping = self.local_mappings[int(local_index)]
        result = torch.zeros(
            (mapping.coords.shape[0], atom_values.shape[1]),
            device=atom_values.device,
            dtype=atom_values.dtype,
        )
        result.index_add_(
            0,
            mapping.token_indices,
            atom_values.index_select(0, mapping.atom_indices)
            * mapping.overlap_fractions[:, None].to(atom_values.dtype),
        )
        mass = mapping.token_atom_mass.to(
            device=atom_values.device, dtype=atom_values.dtype
        )
        return result / mass[:, None].clamp_min(1e-12)

    def project_coarse(self, atom_values: torch.Tensor) -> torch.Tensor:
        """Uniform restriction from C128 atoms to represented C64 parents."""
        if atom_values.ndim != 2 or atom_values.shape[0] != self.atom_count:
            raise ValueError("atom values are not aligned with C128 master")
        numerator = torch.zeros(
            (self.coarse_parent_count, atom_values.shape[1]),
            device=atom_values.device,
            dtype=atom_values.dtype,
        )
        numerator.index_add_(0, self.coarse_parent, atom_values)
        counts = torch.bincount(
            self.coarse_parent, minlength=self.coarse_parent_count
        ).to(device=atom_values.device, dtype=atom_values.dtype)
        return numerator / counts[:, None].clamp_min(1.0)


@dataclass(frozen=True)
class C128NativeWindow:
    """One native C64 model window over a sparse C128 canvas."""

    window_index: int
    token_indices: torch.Tensor
    local_coords: torch.Tensor
    weights: torch.Tensor
    start: Tuple[int, int, int]
    end: Tuple[int, int, int]

    def to(self, device: torch.device) -> "C128NativeWindow":
        return C128NativeWindow(
            window_index=int(self.window_index),
            token_indices=self.token_indices.to(device),
            local_coords=self.local_coords.to(device),
            weights=self.weights.to(device),
            start=tuple(int(value) for value in self.start),
            end=tuple(int(value) for value in self.end),
        )


def _assign_reference_parents(
    *,
    atom_xyz: torch.Tensor,
    atom_codes: torch.Tensor,
    global_codes: torch.Tensor,
    global_parent_rows: torch.Tensor,
    global_coords: torch.Tensor,
    global_resolution: int,
    target_resolution: int,
) -> torch.Tensor:
    reference = torch.full(
        (atom_codes.shape[0],), -1, dtype=torch.int64
    )
    global_positions = _lookup_sorted_codes(atom_codes, global_codes)
    if bool((global_positions < 0).any().item()):
        raise RuntimeError("global atom was lost from the common atom union")
    reference[global_positions] = global_parent_rows
    missing = reference < 0
    if not bool(missing.any().item()):
        return reference

    # Orphan atoms are topology-birth/detail atoms outside active global
    # endpoint cells.  The task specification requires a nearby global surface
    # latent as their coarse reference.  cKDTree keeps this sparse and avoids
    # an O(N_atom*N_global) matrix.
    atom_q = target_cell_centers(
        atom_xyz[missing], int(target_resolution)
    ).numpy()
    global_q = endpoint_indices_to_q(
        global_coords[:, 1:4].to(torch.int64), int(global_resolution)
    ).numpy()
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(global_q)
        _, nearest = tree.query(atom_q, k=1, workers=-1)
        nearest_rows = torch.from_numpy(
            np.asarray(nearest, dtype=np.int64)
        )
    except Exception:
        query = torch.from_numpy(atom_q)
        source = torch.from_numpy(global_q)
        chunks: List[torch.Tensor] = []
        for begin in range(0, query.shape[0], 4096):
            distance = torch.cdist(query[begin : begin + 4096], source)
            chunks.append(distance.argmin(dim=1))
        nearest_rows = torch.cat(chunks)
    reference[missing] = nearest_rows
    return reference


def build_common_atom_space(
    *,
    stage: str,
    global_coords: torch.Tensor,
    global_resolution: int,
    local_coords: Sequence[torch.Tensor],
    local_resolution: int,
    local_to_global: Sequence[TensorTransform],
    global_to_local: Sequence[TensorTransform],
    tile_ids: Optional[Sequence[int]] = None,
    target_resolution: Optional[int] = None,
    chunk_size: int = 2048,
    boundary_epsilon: float = 1e-9,
    max_roundtrip_error: float = 2e-5,
) -> CommonAtomSpace:
    """Build one sparse exact-transform common atom space.

    ``target_resolution`` defaults to four times the endpoint resolution.  It
    is a logical isotropic lattice: a projective local cell usually occupies
    roughly one target cell in x/y and four cells in z, preserving the stated
    anisotropic bandwidth without manufacturing local z detail.
    """
    if len(local_coords) != len(local_to_global) or len(local_coords) != len(
        global_to_local
    ):
        raise ValueError("local supports and transforms must have equal length")
    if tile_ids is None:
        tile_ids = list(range(len(local_coords)))
    if len(tile_ids) != len(local_coords):
        raise ValueError("tile_ids are not aligned with local supports")
    target = (
        int(target_resolution)
        if target_resolution is not None
        else 4 * int(global_resolution)
    )
    if target < int(global_resolution) or target < int(local_resolution):
        raise ValueError("target resolution is too small")
    _validate_coords(global_coords, int(global_resolution), "global")
    for index, coords in enumerate(local_coords):
        _validate_coords(coords, int(local_resolution), f"local[{index}]")

    global_codes, global_rows = _enumerate_global_memberships(
        global_coords,
        int(global_resolution),
        target,
    )
    local_code_rows: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    local_diagnostics: List[Mapping[str, float]] = []
    all_codes: List[torch.Tensor] = [global_codes]
    for coords, forward, inverse in zip(
        local_coords, local_to_global, global_to_local
    ):
        codes, rows, fractions, diagnostics = _rasterize_local_memberships(
            coords,
            endpoint_resolution=int(local_resolution),
            target_resolution=target,
            local_to_global=forward,
            global_to_local=inverse,
            chunk_size=int(chunk_size),
            boundary_epsilon=float(boundary_epsilon),
        )
        if diagnostics["max_q_roundtrip_error"] >= float(max_roundtrip_error):
            raise RuntimeError(
                f"{stage}: transform round-trip error "
                f"{diagnostics['max_q_roundtrip_error']:.6g} exceeds "
                f"{float(max_roundtrip_error):.6g}"
            )
        local_code_rows.append((codes, rows, fractions))
        local_diagnostics.append(diagnostics)
        all_codes.append(codes)

    atom_ids = torch.unique(torch.cat(all_codes), sorted=True)
    atom_xyz = _delinearize_xyz(atom_ids, target)
    global_atom_indices = _lookup_sorted_codes(atom_ids, global_codes)
    if bool((global_atom_indices < 0).any().item()):
        raise RuntimeError("global membership lookup failed")
    global_counts = torch.bincount(
        global_rows, minlength=global_coords.shape[0]
    )
    mappings: List[LocalAtomMapping] = []
    for tile_id, coords, code_rows, diagnostics in zip(
        tile_ids,
        local_coords,
        local_code_rows,
        local_diagnostics,
    ):
        codes, rows, fractions = code_rows
        atom_indices = _lookup_sorted_codes(atom_ids, codes)
        if bool((atom_indices < 0).any().item()):
            raise RuntimeError(f"tile {tile_id}: atom lookup failed")
        token_counts = torch.bincount(rows, minlength=coords.shape[0])
        token_mass = torch.zeros(coords.shape[0], dtype=torch.float32)
        token_mass.index_add_(0, rows, fractions)
        mappings.append(
            LocalAtomMapping(
                tile_id=int(tile_id),
                coords=coords.detach().to(device="cpu", dtype=torch.int32),
                atom_indices=atom_indices,
                token_indices=rows,
                overlap_fractions=fractions,
                token_atom_counts=token_counts,
                token_atom_mass=token_mass,
                diagnostics=dict(diagnostics),
            )
        )

    reference_parent = _assign_reference_parents(
        atom_xyz=atom_xyz,
        atom_codes=atom_ids,
        global_codes=global_codes,
        global_parent_rows=global_rows,
        global_coords=global_coords.detach().cpu(),
        global_resolution=int(global_resolution),
        target_resolution=target,
    )
    reference_counts = torch.bincount(
        reference_parent, minlength=global_coords.shape[0]
    )
    if bool((reference_counts == 0).any().item()):
        raise RuntimeError("a global parent has no atom in the augmented partition")
    atom_volume = (2.0 / float(target)) ** 3
    diagnostics: Dict[str, object] = {
        "stage": str(stage),
        "global_resolution": int(global_resolution),
        "local_resolution": int(local_resolution),
        "target_resolution": int(target),
        "atom_count": int(atom_ids.shape[0]),
        "atom_volume": float(atom_volume),
        "global_tokens": int(global_coords.shape[0]),
        "global_memberships": int(global_codes.shape[0]),
        "orphan_atoms_with_nearest_global_reference": int(
            (reference_parent.index_select(0, global_atom_indices.new_zeros(0))).numel()
        ),
        "local": [
            {
                "tile_id": int(tile_id),
                "tokens": int(coords.shape[0]),
                **dict(item),
            }
            for tile_id, coords, item in zip(
                tile_ids, local_coords, local_diagnostics
            )
        ],
    }
    # Count atoms not covered by a real global footprint, not merely atoms
    # whose nearest-reference row happens to equal a global row.
    real_global_mask = torch.zeros(atom_ids.shape[0], dtype=torch.bool)
    real_global_mask[global_atom_indices] = True
    diagnostics["orphan_atoms_with_nearest_global_reference"] = int(
        (~real_global_mask).sum().item()
    )
    return CommonAtomSpace(
        stage=str(stage),
        global_resolution=int(global_resolution),
        local_resolution=int(local_resolution),
        target_resolution=int(target),
        atom_coords=torch.cat(
            [
                torch.zeros((atom_xyz.shape[0], 1), dtype=torch.int32),
                atom_xyz.to(torch.int32),
            ],
            dim=1,
        ),
        atom_ids=atom_ids,
        atom_volume=float(atom_volume),
        global_coords=global_coords.detach().to(
            device="cpu", dtype=torch.int32
        ),
        global_atom_indices=global_atom_indices,
        global_parent_rows=global_rows,
        global_token_atom_counts=global_counts,
        atom_reference_parent=reference_parent,
        local_mappings=tuple(mappings),
        diagnostics=diagnostics,
    )


def build_c128_master_atom_space(
    *,
    stage: str,
    master_coords: torch.Tensor,
    coarse_parent: torch.Tensor,
    coarse_parent_count: int,
    local_coords: Sequence[torch.Tensor],
    local_to_global: Sequence[TensorTransform],
    global_to_local: Sequence[TensorTransform],
    tile_ids: Optional[Sequence[int]] = None,
    target_resolution: int = 128,
    local_resolution: int = 64,
    chunk_size: int = 2048,
    boundary_epsilon: float = 1e-9,
    max_roundtrip_error: float = 2e-5,
) -> C128MasterAtomSpace:
    """Build an asymmetric C128-master/local-C64 exact-footprint space.

    The master support is fixed before any shape or texture flow.  Local
    footprints are intersected with that support, so no later coordinate
    averaging, post-generation support reduction, or a finer intermediate can occur.
    """
    target = int(target_resolution)
    local_res = int(local_resolution)
    if target != 128 or local_res != 64:
        raise ValueError(
            "the native 2048 coupled flow requires target C128/local C64"
        )
    _validate_coords(master_coords, target, "C128 master")
    if coarse_parent.ndim != 1 or coarse_parent.shape[0] != master_coords.shape[0]:
        raise ValueError("coarse_parent must align with C128 master rows")
    if int(coarse_parent_count) < 1:
        raise ValueError("coarse_parent_count must be positive")
    if bool(
        (
            (coarse_parent.to(torch.int64) < 0)
            | (coarse_parent.to(torch.int64) >= int(coarse_parent_count))
        ).any().item()
    ):
        raise ValueError("coarse_parent contains an out-of-range row")
    if len(local_coords) != len(local_to_global) or len(local_coords) != len(
        global_to_local
    ):
        raise ValueError("local supports and transforms must have equal length")
    if tile_ids is None:
        tile_ids = list(range(len(local_coords)))
    if len(tile_ids) != len(local_coords):
        raise ValueError("tile_ids are not aligned with local supports")

    master_codes_unsorted = _linearize_xyz(
        master_coords[:, 1:4].detach().cpu(), target
    )
    order = torch.argsort(master_codes_unsorted)
    atom_ids = master_codes_unsorted.index_select(0, order)
    atom_coords = master_coords.detach().to(
        device="cpu", dtype=torch.int32
    ).index_select(0, order)
    parents = coarse_parent.detach().to(
        device="cpu", dtype=torch.int64
    ).index_select(0, order)
    mappings: List[LocalAtomMapping] = []
    local_diagnostics: List[Mapping[str, float]] = []

    for tile_id, coords, forward, inverse in zip(
        tile_ids,
        local_coords,
        local_to_global,
        global_to_local,
    ):
        _validate_coords(coords, local_res, f"local[{int(tile_id)}]")
        codes, rows, fractions, diagnostics = _rasterize_local_memberships(
            coords,
            endpoint_resolution=local_res,
            target_resolution=target,
            local_to_global=forward,
            global_to_local=inverse,
            chunk_size=int(chunk_size),
            boundary_epsilon=float(boundary_epsilon),
        )
        if diagnostics["max_q_roundtrip_error"] >= float(max_roundtrip_error):
            raise RuntimeError(
                f"{stage}: transform round-trip error "
                f"{diagnostics['max_q_roundtrip_error']:.6g} exceeds "
                f"{float(max_roundtrip_error):.6g}"
            )
        positions = _lookup_sorted_codes(atom_ids, codes)
        on_master = positions >= 0
        positions = positions[on_master]
        rows = rows[on_master]
        fractions = fractions[on_master]
        counts_all = torch.bincount(rows, minlength=coords.shape[0])
        keep_tokens = counts_all > 0
        dropped_tokens = int((~keep_tokens).sum().item())
        if not bool(keep_tokens.any().item()):
            raise RuntimeError(
                f"{stage}: tile {int(tile_id)} has no local C64 token "
                "intersecting the fixed C128 master"
            )
        if dropped_tokens:
            row_remap = torch.full(
                (coords.shape[0],), -1, dtype=torch.int64
            )
            row_remap[keep_tokens] = torch.arange(
                int(keep_tokens.sum().item()), dtype=torch.int64
            )
            rows = row_remap.index_select(0, rows)
        filtered_coords = coords.detach().to(
            device="cpu", dtype=torch.int32
        )[keep_tokens]
        counts = torch.bincount(rows, minlength=filtered_coords.shape[0])
        mass = torch.zeros(filtered_coords.shape[0], dtype=torch.float32)
        mass.index_add_(0, rows, fractions)
        filtered_diagnostics = {
            **dict(diagnostics),
            "master_intersection_edges": float(positions.shape[0]),
            "edges_outside_master_dropped": float((~on_master).sum().item()),
            "local_tokens_outside_master_dropped": float(dropped_tokens),
            "min_master_mass_per_token": float(mass.min().item()),
            "max_master_mass_per_token": float(mass.max().item()),
            "mean_master_mass_per_token": float(mass.mean().item()),
        }
        mappings.append(
            LocalAtomMapping(
                tile_id=int(tile_id),
                coords=filtered_coords,
                atom_indices=positions,
                token_indices=rows,
                overlap_fractions=fractions.to(torch.float32),
                token_atom_counts=counts,
                token_atom_mass=mass,
                diagnostics=filtered_diagnostics,
            )
        )
        local_diagnostics.append(filtered_diagnostics)

    represented = torch.bincount(
        parents, minlength=int(coarse_parent_count)
    )
    diagnostics: Dict[str, object] = {
        "stage": str(stage),
        "target_resolution": target,
        "local_resolution": local_res,
        "atom_count": int(atom_coords.shape[0]),
        "coarse_parent_count": int(coarse_parent_count),
        "represented_coarse_parents": int((represented > 0).sum().item()),
        "unrepresented_coarse_parents": int((represented == 0).sum().item()),
        "global_master_mapping": "one independently generated C128 state per atom",
        "coarse_projector": "uniform C128-to-nearest-active-C64 parent",
        "local": [
            {
                "tile_id": int(tile_id),
                "tokens": int(coords.shape[0]),
                **dict(item),
            }
            for tile_id, coords, item in zip(
                tile_ids, local_coords, local_diagnostics
            )
        ],
    }
    return C128MasterAtomSpace(
        stage=str(stage),
        target_resolution=target,
        local_resolution=local_res,
        atom_coords=atom_coords,
        atom_ids=atom_ids,
        coarse_parent=parents,
        coarse_parent_count=int(coarse_parent_count),
        local_mappings=tuple(mappings),
        diagnostics=diagnostics,
    )


def _namespace_key(seed: int, namespace: str, resolution: int) -> np.uint64:
    digest = hashlib.blake2b(
        f"{int(seed)}|{namespace}|{int(resolution)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return np.frombuffer(digest, dtype="<u8")[0]


def _splitmix64(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.uint64, copy=False)
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    value = (value ^ (value >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return value ^ (value >> np.uint64(31))


def stateless_atom_noise(
    atom_ids: torch.Tensor,
    channels: int,
    *,
    seed: int,
    namespace: str,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 262144,
) -> torch.Tensor:
    """Generate N(0,1) by a stateless atom/channel counter key."""
    if int(channels) < 1:
        raise ValueError("channels must be positive")
    ids = atom_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
    namespace_key = _namespace_key(int(seed), str(namespace), int(resolution))
    channel_ids = np.arange(int(channels), dtype=np.uint64)[None]
    pieces: List[torch.Tensor] = []
    for begin in range(0, ids.shape[0], int(chunk_size)):
        atom = ids[begin : begin + int(chunk_size)].astype(np.uint64)[:, None]
        counter = (
            atom * np.uint64(0xD2B74407B1CE6E93)
            + channel_ids * np.uint64(0xCA5A826395121157)
            + namespace_key
        )
        bits1 = _splitmix64(counter)
        bits2 = _splitmix64(counter + np.uint64(0x9E3779B97F4A7C15))
        uniform1 = (
            (bits1 >> np.uint64(11)).astype(np.float64) + 0.5
        ) * (1.0 / float(1 << 53))
        uniform2 = (
            (bits2 >> np.uint64(11)).astype(np.float64) + 0.5
        ) * (1.0 / float(1 << 53))
        normal = np.sqrt(-2.0 * np.log(uniform1)) * np.cos(
            2.0 * math.pi * uniform2
        )
        pieces.append(torch.from_numpy(normal.astype(np.float32)))
    return torch.cat(pieces, dim=0).to(device=device, dtype=dtype)


def _restrict_white_noise(
    atom_noise: torch.Tensor,
    atom_indices: torch.Tensor,
    token_indices: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros(
        (token_counts.shape[0], atom_noise.shape[1]),
        device=atom_noise.device,
        dtype=torch.float32,
    )
    result.index_add_(
        0,
        token_indices,
        atom_noise.index_select(0, atom_indices).to(torch.float32),
    )
    denominator = token_counts.to(
        device=atom_noise.device, dtype=torch.float32
    ).sqrt()
    return (result / denominator[:, None].clamp_min(1.0)).to(atom_noise.dtype)


def _restrict_partial_white_noise(
    atom_noise: torch.Tensor,
    atom_ids: torch.Tensor,
    mapping: LocalAtomMapping,
    *,
    seed: int,
    namespace: str,
    resolution: int,
) -> torch.Tensor:
    """Restrict partial target-cell overlaps with the correct edge variance.

    The common part ``f*xi_atom`` gives covariance ``f`` with a full global
    atom.  ``sqrt(f-f²)*eta_edge`` restores the intersection variance to ``f``.
    This is the overlap-matrix equivalent of explicitly splitting a target
    cell along one local boundary, without materializing a dense adaptive
    octree node for every tiny sliver.
    """
    fractions = mapping.overlap_fractions.to(
        device=atom_noise.device, dtype=torch.float32
    ).clamp(0.0, 1.0)
    edge_ids = (
        atom_ids.index_select(0, mapping.atom_indices).to(torch.int64)
        * int(mapping.coords.shape[0])
        + mapping.token_indices.to(torch.int64)
    )
    independent = stateless_atom_noise(
        edge_ids,
        atom_noise.shape[1],
        seed=int(seed),
        namespace=f"{namespace}/tile-{int(mapping.tile_id)}/partial",
        resolution=int(resolution),
        device=atom_noise.device,
        dtype=atom_noise.dtype,
    )
    shared = atom_noise.index_select(0, mapping.atom_indices).to(torch.float32)
    contribution = (
        fractions[:, None] * shared
        + torch.sqrt(
            (fractions - fractions.square()).clamp_min(0.0)
        )[:, None]
        * independent.to(torch.float32)
    )
    result = torch.zeros(
        (mapping.coords.shape[0], atom_noise.shape[1]),
        device=atom_noise.device,
        dtype=torch.float32,
    )
    result.index_add_(0, mapping.token_indices, contribution)
    denominator = mapping.token_atom_mass.to(
        device=atom_noise.device, dtype=torch.float32
    ).sqrt()
    return (result / denominator[:, None].clamp_min(1e-12)).to(
        atom_noise.dtype
    )


def shared_spatial_noise(
    atom_space: CommonAtomSpace,
    channels: int,
    *,
    seed: int,
    namespace: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Return global/local token noise from one canonical atom realization."""
    target_device = device or torch.device("cpu")
    space = atom_space.to(target_device)
    atom_noise = stateless_atom_noise(
        space.atom_ids,
        int(channels),
        seed=int(seed),
        namespace=namespace or f"noise/{space.stage}",
        resolution=int(space.target_resolution),
        device=target_device,
        dtype=dtype,
    )
    global_noise = _restrict_white_noise(
        atom_noise,
        space.global_atom_indices,
        space.global_parent_rows,
        space.global_token_atom_counts,
    )
    local_noise = [
        _restrict_partial_white_noise(
            atom_noise,
            space.atom_ids,
            mapping,
            seed=int(seed),
            namespace=namespace or f"noise/{space.stage}",
            resolution=int(space.target_resolution),
        )
        for mapping in space.local_mappings
    ]
    return global_noise, local_noise, atom_noise


def shared_c128_master_local_noise(
    atom_space: C128MasterAtomSpace,
    channels: int,
    *,
    seed: int,
    namespace: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Generate direct C128 master noise and exact-overlap local C64 noise."""
    target_device = device or torch.device("cpu")
    space = atom_space.to(target_device)
    atom_noise = stateless_atom_noise(
        space.atom_ids,
        int(channels),
        seed=int(seed),
        namespace=namespace or f"noise/{space.stage}",
        resolution=int(space.target_resolution),
        device=target_device,
        dtype=dtype,
    )
    local_noise = [
        _restrict_partial_white_noise(
            atom_noise,
            space.atom_ids,
            mapping,
            seed=int(seed),
            namespace=namespace or f"noise/{space.stage}",
            resolution=int(space.target_resolution),
        )
        for mapping in space.local_mappings
    ]
    return atom_noise, local_noise, atom_noise


def raised_cosine_tile_weights(
    uv: torch.Tensor,
    width: int,
    height: int,
    *,
    minimum: float = 0.0,
) -> torch.Tensor:
    """Separable sin² image window used only as residual confidence."""
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape [N,2]")
    u = (uv[:, 0].to(torch.float32) / float(width)).clamp(0.0, 1.0)
    v = (uv[:, 1].to(torch.float32) / float(height)).clamp(0.0, 1.0)
    weight = torch.sin(math.pi * u).square() * torch.sin(math.pi * v).square()
    return weight.clamp_min(float(minimum))


def _scatter_weighted_mean(
    values: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    output_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    numerator = torch.zeros(
        (int(output_rows), values.shape[1]),
        dtype=torch.float32,
        device=values.device,
    )
    denominator = torch.zeros(
        (int(output_rows), 1),
        dtype=torch.float32,
        device=values.device,
    )
    numerator.index_add_(0, indices, values.to(torch.float32) * weights[:, None])
    denominator.index_add_(0, indices, weights[:, None])
    return numerator / denominator.clamp_min(1e-12), denominator[:, 0]


@dataclass(frozen=True)
class EndpointFusionResult:
    unified_atoms: torch.Tensor
    global_atoms: torch.Tensor
    merged_residual: torch.Tensor
    high_residual: torch.Tensor
    coverage: torch.Tensor
    local_synced_endpoints: Tuple[torch.Tensor, ...]
    diagnostics: Mapping[str, float]


def restrict_fusion_to_atom_subset(
    atom_space: CommonAtomSpace,
    fusion: EndpointFusionResult,
    global_endpoint: torch.Tensor,
    atom_indices: torch.Tensor,
) -> Tuple[torch.Tensor, Mapping[str, float]]:
    """Re-high-pass a published fusion on the final sparse decode support.

    Selecting learned surface atoms after a flow stage changes the measure on
    which the parent mean is defined.  Recomputing the covered mean on that
    selected support prevents support selection itself from reintroducing a
    global low-frequency residual.
    """
    if atom_indices.ndim != 1 or atom_indices.numel() < 1:
        raise ValueError("atom_indices must be a nonempty vector")
    device = global_endpoint.device
    space = atom_space.to(device)
    selected = atom_indices.to(device=device, dtype=torch.int64)
    if bool(
        ((selected < 0) | (selected >= int(space.atom_count))).any().item()
    ):
        raise ValueError("selected atom index lies outside atom space")
    if torch.unique(selected).shape[0] != selected.shape[0]:
        raise ValueError("selected atom indices contain duplicates")
    parents = space.atom_reference_parent.index_select(0, selected)
    residual = fusion.merged_residual.index_select(0, selected).to(torch.float32)
    coverage = fusion.coverage.index_select(0, selected).to(torch.float32)
    numerator = torch.zeros(
        (space.global_coords.shape[0], residual.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.zeros(
        space.global_coords.shape[0], dtype=torch.float32, device=device
    )
    numerator.index_add_(0, parents, residual * coverage[:, None])
    denominator.index_add_(0, parents, coverage)
    mean = numerator / denominator[:, None].clamp_min(1e-12)
    high = coverage[:, None] * (
        residual - mean.index_select(0, parents)
    )
    unified = (
        global_endpoint.index_select(0, parents).to(torch.float32) + high
    )
    check_numerator = torch.zeros_like(numerator)
    check_count = torch.zeros_like(denominator)
    check_numerator.index_add_(0, parents, unified)
    check_count.index_add_(0, parents, torch.ones_like(coverage))
    represented = check_count > 0
    restricted = (
        check_numerator[represented]
        / check_count[represented, None]
    )
    expected = global_endpoint[represented].to(torch.float32)
    error = (restricted - expected).abs()
    return unified.to(global_endpoint.dtype), {
        "selected_atoms": float(selected.shape[0]),
        "represented_global_parents": float(represented.sum().item()),
        "max_abs_selected_restriction_error": float(error.max().item()),
        "mean_abs_selected_restriction_error": float(error.mean().item()),
    }


def fuse_clean_endpoints(
    atom_space: CommonAtomSpace,
    global_endpoint: torch.Tensor,
    local_endpoints: Sequence[torch.Tensor],
    *,
    local_token_weights: Optional[Sequence[torch.Tensor]] = None,
    visibility: Optional[Sequence[torch.Tensor]] = None,
    support_confidence: Optional[Sequence[torch.Tensor]] = None,
    huber_delta: float = 1.0,
    robust_iterations: int = 3,
    reject_opposite: bool = True,
) -> EndpointFusionResult:
    """Fuse local-minus-global endpoints and remove every global-parent mean."""
    if len(local_endpoints) != len(atom_space.local_mappings):
        raise ValueError("local endpoints are not aligned with atom mappings")
    device = global_endpoint.device
    space = atom_space.to(device)
    if local_token_weights is None:
        local_token_weights = [
            torch.ones(mapping.coords.shape[0], device=device)
            for mapping in space.local_mappings
        ]
    if visibility is None:
        visibility = [
            torch.ones(mapping.coords.shape[0], device=device)
            for mapping in space.local_mappings
        ]
    if support_confidence is None:
        support_confidence = [
            torch.ones(mapping.coords.shape[0], device=device)
            for mapping in space.local_mappings
        ]
    if not (
        len(local_token_weights)
        == len(visibility)
        == len(support_confidence)
        == len(local_endpoints)
    ):
        raise ValueError("local confidence inputs are not aligned")

    global_atoms = space.lift_global(global_endpoint)
    edge_residuals: List[torch.Tensor] = []
    edge_atoms: List[torch.Tensor] = []
    edge_weights: List[torch.Tensor] = []
    for endpoint, mapping, image_weight, visible, confidence in zip(
        local_endpoints,
        space.local_mappings,
        local_token_weights,
        visibility,
        support_confidence,
    ):
        if endpoint.ndim != 2 or endpoint.shape[0] != mapping.coords.shape[0]:
            raise ValueError(
                f"tile {mapping.tile_id}: endpoint/support row mismatch"
            )
        token_weight = (
            image_weight.to(device=device, dtype=torch.float32)
            * visible.to(device=device, dtype=torch.float32)
            * confidence.to(device=device, dtype=torch.float32)
        ).clamp_min(0.0)
        residual = endpoint.index_select(
            0, mapping.token_indices
        ) - global_atoms.index_select(0, mapping.atom_indices)
        edge_residuals.append(residual)
        edge_atoms.append(mapping.atom_indices)
        edge_weights.append(
            token_weight.index_select(0, mapping.token_indices)
            * mapping.overlap_fractions.to(
                device=device, dtype=torch.float32
            )
        )
    residuals = torch.cat(edge_residuals, dim=0)
    atoms = torch.cat(edge_atoms, dim=0)
    base_weights = torch.cat(edge_weights, dim=0)
    merged, weight_sum = _scatter_weighted_mean(
        residuals, atoms, base_weights, space.atom_count
    )

    robust_weights = base_weights
    rejected_opposite = torch.zeros_like(base_weights, dtype=torch.bool)
    for iteration in range(max(0, int(robust_iterations))):
        center = merged.index_select(0, atoms)
        distance = torch.linalg.vector_norm(
            residuals.to(torch.float32) - center, dim=1
        )
        huber = torch.clamp(
            float(huber_delta) / distance.clamp_min(1e-12),
            max=1.0,
        )
        if bool(reject_opposite) and iteration > 0:
            dot = (residuals.to(torch.float32) * center).sum(dim=1)
            valid_direction = (
                (dot >= 0.0)
                | (torch.linalg.vector_norm(center, dim=1) < 1e-8)
            )
            rejected_opposite |= ~valid_direction
            huber = huber * valid_direction.to(huber.dtype)
        robust_weights = base_weights * huber
        merged, weight_sum = _scatter_weighted_mean(
            residuals, atoms, robust_weights, space.atom_count
        )

    # Confidence is a coverage mask, not a noise variance.  Multiple tiles
    # saturate at full coverage.
    coverage_sum = torch.zeros(
        space.atom_count, device=device, dtype=torch.float32
    )
    coverage_sum.index_add_(0, atoms, base_weights.clamp(0.0, 1.0))
    coverage = coverage_sum.clamp(0.0, 1.0)
    merged = torch.where(
        (weight_sum > 0)[:, None],
        merged,
        torch.zeros_like(merged),
    )

    parents = space.atom_reference_parent
    covered_mass = coverage
    parent_numerator = torch.zeros(
        (space.global_coords.shape[0], merged.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    parent_denominator = torch.zeros(
        space.global_coords.shape[0],
        device=device,
        dtype=torch.float32,
    )
    parent_numerator.index_add_(
        0, parents, merged.to(torch.float32) * covered_mass[:, None]
    )
    parent_denominator.index_add_(0, parents, covered_mass)
    covered_mean = (
        parent_numerator / parent_denominator[:, None].clamp_min(1e-12)
    )
    high = coverage[:, None] * (
        merged.to(torch.float32) - covered_mean.index_select(0, parents)
    )
    unified = global_atoms.to(torch.float32) + high
    restricted_high = space.restrict_global(high)
    restricted_unified = space.restrict_global(unified)
    low_frequency_error = (
        restricted_unified - global_endpoint.to(torch.float32)
    ).abs()
    high_error = restricted_high.abs()
    local_synced = tuple(
        space.restrict_local(unified, index).to(global_endpoint.dtype)
        for index in range(len(space.local_mappings))
    )
    diagnostics = {
        "atom_count": float(space.atom_count),
        "residual_edges": float(residuals.shape[0]),
        "covered_atoms": float((coverage > 0).sum().item()),
        "fully_covered_atoms": float((coverage >= 1.0).sum().item()),
        "opposite_edges_rejected": float(rejected_opposite.sum().item()),
        "max_abs_Rg_high": float(high_error.max().item()),
        "max_abs_Rg_unified_minus_global": float(
            low_frequency_error.max().item()
        ),
        "mean_abs_Rg_unified_minus_global": float(
            low_frequency_error.mean().item()
        ),
    }
    return EndpointFusionResult(
        unified_atoms=unified.to(global_endpoint.dtype),
        global_atoms=global_atoms,
        merged_residual=merged.to(global_endpoint.dtype),
        high_residual=high.to(global_endpoint.dtype),
        coverage=coverage,
        local_synced_endpoints=local_synced,
        diagnostics=diagnostics,
    )


def fuse_c128_master_clean_endpoints(
    atom_space: C128MasterAtomSpace,
    global_endpoint: torch.Tensor,
    local_endpoints: Sequence[torch.Tensor],
    *,
    local_token_weights: Optional[Sequence[torch.Tensor]] = None,
    visibility: Optional[Sequence[torch.Tensor]] = None,
    support_confidence: Optional[Sequence[torch.Tensor]] = None,
    huber_delta: float = 1.0,
    robust_iterations: int = 3,
    reject_opposite: bool = True,
) -> EndpointFusionResult:
    """Fuse local C64 endpoints against a native C128 global master.

    Local endpoint values are lifted only through their exact footprint
    incidences.  The master is already defined independently at every C128
    atom.  High-pass removal uses the explicit C128-to-C64 coarse projector,
    never a broadcast master feature.
    """
    if len(local_endpoints) != len(atom_space.local_mappings):
        raise ValueError("local endpoints are not aligned with C128 mappings")
    device = global_endpoint.device
    space = atom_space.to(device)
    if global_endpoint.ndim != 2 or global_endpoint.shape[0] != space.atom_count:
        raise ValueError("global endpoint is not aligned with C128 master")
    if local_token_weights is None:
        local_token_weights = [
            torch.ones(mapping.coords.shape[0], device=device)
            for mapping in space.local_mappings
        ]
    if visibility is None:
        visibility = [
            torch.ones(mapping.coords.shape[0], device=device)
            for mapping in space.local_mappings
        ]
    if support_confidence is None:
        support_confidence = [
            torch.ones(mapping.coords.shape[0], device=device)
            for mapping in space.local_mappings
        ]
    if not (
        len(local_token_weights)
        == len(visibility)
        == len(support_confidence)
        == len(local_endpoints)
    ):
        raise ValueError("local confidence inputs are not aligned")

    edge_residuals: List[torch.Tensor] = []
    edge_atoms: List[torch.Tensor] = []
    edge_weights: List[torch.Tensor] = []
    for endpoint, mapping, image_weight, visible, confidence in zip(
        local_endpoints,
        space.local_mappings,
        local_token_weights,
        visibility,
        support_confidence,
    ):
        if endpoint.ndim != 2 or endpoint.shape[0] != mapping.coords.shape[0]:
            raise ValueError(
                f"tile {mapping.tile_id}: endpoint/support row mismatch"
            )
        token_weight = (
            image_weight.to(device=device, dtype=torch.float32)
            * visible.to(device=device, dtype=torch.float32)
            * confidence.to(device=device, dtype=torch.float32)
        ).clamp_min(0.0)
        edge_residuals.append(
            endpoint.index_select(0, mapping.token_indices)
            - global_endpoint.index_select(0, mapping.atom_indices)
        )
        edge_atoms.append(mapping.atom_indices)
        edge_weights.append(
            token_weight.index_select(0, mapping.token_indices)
            * mapping.overlap_fractions.to(
                device=device, dtype=torch.float32
            )
        )
    residuals = torch.cat(edge_residuals, dim=0)
    atoms = torch.cat(edge_atoms, dim=0)
    base_weights = torch.cat(edge_weights, dim=0)
    merged, weight_sum = _scatter_weighted_mean(
        residuals, atoms, base_weights, space.atom_count
    )
    rejected_opposite = torch.zeros_like(base_weights, dtype=torch.bool)
    robust_weights = base_weights
    for iteration in range(max(0, int(robust_iterations))):
        center = merged.index_select(0, atoms)
        distance = torch.linalg.vector_norm(
            residuals.to(torch.float32) - center, dim=1
        )
        huber = torch.clamp(
            float(huber_delta) / distance.clamp_min(1e-12),
            max=1.0,
        )
        if bool(reject_opposite) and iteration > 0:
            dot = (residuals.to(torch.float32) * center).sum(dim=1)
            valid_direction = (
                (dot >= 0.0)
                | (torch.linalg.vector_norm(center, dim=1) < 1e-8)
            )
            rejected_opposite |= ~valid_direction
            huber = huber * valid_direction.to(huber.dtype)
        robust_weights = base_weights * huber
        merged, weight_sum = _scatter_weighted_mean(
            residuals, atoms, robust_weights, space.atom_count
        )

    coverage_sum = torch.zeros(
        space.atom_count, device=device, dtype=torch.float32
    )
    coverage_sum.index_add_(0, atoms, base_weights.clamp(0.0, 1.0))
    coverage = coverage_sum.clamp(0.0, 1.0)
    merged = torch.where(
        (weight_sum > 0)[:, None],
        merged,
        torch.zeros_like(merged),
    )
    parents = space.coarse_parent
    numerator = torch.zeros(
        (space.coarse_parent_count, merged.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(
        space.coarse_parent_count,
        device=device,
        dtype=torch.float32,
    )
    numerator.index_add_(
        0, parents, merged.to(torch.float32) * coverage[:, None]
    )
    denominator.index_add_(0, parents, coverage)
    covered_mean = numerator / denominator[:, None].clamp_min(1e-12)
    high = coverage[:, None] * (
        merged.to(torch.float32) - covered_mean.index_select(0, parents)
    )
    unified = global_endpoint.to(torch.float32) + high
    projected_high = space.project_coarse(high)
    projected_unified = space.project_coarse(unified)
    projected_global = space.project_coarse(global_endpoint.to(torch.float32))
    represented = torch.bincount(
        parents, minlength=space.coarse_parent_count
    ) > 0
    high_error = projected_high[represented].abs()
    low_error = (
        projected_unified[represented] - projected_global[represented]
    ).abs()
    local_synced = tuple(
        space.restrict_local(unified, index).to(global_endpoint.dtype)
        for index in range(len(space.local_mappings))
    )
    diagnostics = {
        "atom_count": float(space.atom_count),
        "residual_edges": float(residuals.shape[0]),
        "covered_atoms": float((coverage > 0).sum().item()),
        "fully_covered_atoms": float((coverage >= 1.0).sum().item()),
        "opposite_edges_rejected": float(rejected_opposite.sum().item()),
        "represented_coarse_parents": float(represented.sum().item()),
        "max_abs_Pcoarse_high": float(high_error.max().item()),
        "max_abs_Pcoarse_unified_minus_global": float(low_error.max().item()),
        "mean_abs_Pcoarse_unified_minus_global": float(low_error.mean().item()),
        # Stable aliases keep progress/report consumers generic.
        "max_abs_Rg_high": float(high_error.max().item()),
        "max_abs_Rg_unified_minus_global": float(low_error.max().item()),
        "mean_abs_Rg_unified_minus_global": float(low_error.mean().item()),
    }
    return EndpointFusionResult(
        unified_atoms=unified.to(global_endpoint.dtype),
        global_atoms=global_endpoint,
        merged_residual=merged.to(global_endpoint.dtype),
        high_residual=high.to(global_endpoint.dtype),
        coverage=coverage,
        local_synced_endpoints=local_synced,
        diagnostics=diagnostics,
    )


def sampler_clean_endpoint_to_velocity(
    sampler: object,
    state: object,
    t: float,
    clean_endpoint: torch.Tensor,
) -> torch.Tensor:
    """Use Pixal3D's actual probability path to recover synchronized velocity.

    Pixal3D samples from t=1 (noise) to t=0 (data) and names the clean endpoint
    ``x_0``.  Calling the sampler's own conversion avoids guessing a reversed
    x0/x1 convention from variable names in a design document.
    """
    state_features = state.feats if hasattr(state, "feats") else state
    endpoint = (
        state.replace(clean_endpoint)
        if hasattr(state, "replace")
        else clean_endpoint
    )
    prediction = sampler._xstart_to_pred(state, float(t), endpoint)
    return prediction.feats if hasattr(prediction, "feats") else prediction


@dataclass(frozen=True)
class CoupledFlowResult:
    global_samples: Any
    local_samples: Tuple[Any, ...]
    final_fusion: EndpointFusionResult
    times: Tuple[float, ...]
    step_records: Tuple[Mapping[str, float], ...]
    elapsed_seconds: float


def _sparse_features(value: Any) -> torch.Tensor:
    return value.feats if hasattr(value, "feats") else value


def _slice_c128_window_tree(
    value: Any,
    token_indices: torch.Tensor,
    local_coords: torch.Tensor,
    master_coords: torch.Tensor,
) -> Any:
    """Slice only token-aligned sparse values and preserve global conditions."""
    if hasattr(value, "feats") and hasattr(value, "coords"):
        if value.feats.shape[0] != master_coords.shape[0]:
            raise ValueError(
                "window sparse condition is not aligned with C128 support"
            )
        if not torch.equal(value.coords, master_coords):
            raise RuntimeError(
                "window sparse condition coordinates/order differ from C128 support"
            )
        return value.__class__(
            feats=value.feats.index_select(0, token_indices),
            coords=local_coords,
        )
    if isinstance(value, Mapping):
        return {
            key: _slice_c128_window_tree(
                item, token_indices, local_coords, master_coords
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _slice_c128_window_tree(
                item, token_indices, local_coords, master_coords
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _slice_c128_window_tree(
                item, token_indices, local_coords, master_coords
            )
            for item in value
        ]
    return value


@torch.no_grad()
def run_c128_2048_coupled_endpoint_flow(
    *,
    sampler: object,
    model: torch.nn.Module,
    atom_space: C128MasterAtomSpace,
    global_windows: Sequence[C128NativeWindow],
    global_state: Any,
    local_states: Sequence[Any],
    global_condition: Mapping[str, Any],
    local_conditions: Sequence[Mapping[str, Any]],
    steps: int,
    rescale_t: float,
    sampler_step_kwargs: Optional[Mapping[str, Any]] = None,
    global_concat_cond: Optional[Any] = None,
    local_concat_cond: Optional[Sequence[Any]] = None,
    local_token_weights: Optional[Sequence[torch.Tensor]] = None,
    visibility: Optional[Sequence[torch.Tensor]] = None,
    support_confidence: Optional[Sequence[torch.Tensor]] = None,
    huber_delta: float = 1.0,
    robust_iterations: int = 3,
    invariant_tolerance: float = 2e-5,
    progress_callback: Optional[
        Callable[[int, int, Mapping[str, float]], None]
    ] = None,
) -> CoupledFlowResult:
    """Run a native-window C128 master coupled to projective local C64 flows.

    The pretrained model only sees native C64 window coordinates.  Window
    clean endpoints are merged on the fixed C128 canvas to form the unique
    global master endpoint.  Local tile endpoints are then fused relative to
    that master through exact footprint incidences and projected to zero mean
    over the designated C64 coarse parents.
    """
    if int(atom_space.target_resolution) != 128:
        raise ValueError("global master must use C128")
    if int(atom_space.local_resolution) != 64:
        raise ValueError("local tile model must use C64")
    if len(local_states) != len(atom_space.local_mappings):
        raise ValueError("local states are not aligned with atom space")
    if len(local_conditions) != len(local_states):
        raise ValueError("local conditions are not aligned with states")
    if local_concat_cond is not None and len(local_concat_cond) != len(
        local_states
    ):
        raise ValueError("local concat conditions are not aligned")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if not global_windows:
        raise ValueError("native C128 master has no active C64 windows")

    master_coords = global_state.coords
    if not torch.equal(master_coords, atom_space.atom_coords.to(master_coords.device)):
        raise RuntimeError("global state does not use the fixed C128 atom support")
    windows = tuple(window.to(master_coords.device) for window in global_windows)
    coverage_check = torch.zeros(
        master_coords.shape[0], device=master_coords.device, dtype=torch.float32
    )
    for window in windows:
        if window.local_coords.shape[0] != window.token_indices.shape[0]:
            raise ValueError("window local coordinates/token indices differ")
        if window.weights.shape[0] != window.token_indices.shape[0]:
            raise ValueError("window weights/token indices differ")
        xyz = window.local_coords[:, 1:4]
        if bool(((xyz < 0) | (xyz >= 64)).any().item()):
            raise ValueError("native model window coordinates must be in [0,63]")
        coverage_check.index_add_(
            0, window.token_indices, window.weights.to(torch.float32)
        )
    if bool((coverage_check <= 0).any().item()):
        raise RuntimeError("native C64 windows leave C128 master tokens uncovered")
    del coverage_check

    times = tuple(
        float(value)
        for value in sampler.timestep_schedule(int(steps), float(rescale_t))
    )
    if len(times) != int(steps) + 1:
        raise RuntimeError("sampler returned an invalid timestep schedule")
    kwargs = dict(sampler_step_kwargs or {})
    current_global = global_state
    current_local = list(local_states)
    records: List[Mapping[str, float]] = []
    started = time.perf_counter()

    for step in range(int(steps)):
        t = times[step]
        t_next = times[step + 1]
        dt = float(t - t_next)
        endpoint_sum = torch.zeros_like(
            current_global.feats, dtype=torch.float32
        )
        endpoint_weight = torch.zeros(
            (current_global.feats.shape[0], 1),
            device=current_global.device,
            dtype=torch.float32,
        )
        for window in windows:
            indices = window.token_indices
            patch_state = current_global.__class__(
                feats=current_global.feats.index_select(0, indices),
                coords=window.local_coords,
            )
            patch_call = _slice_c128_window_tree(
                global_condition,
                indices,
                window.local_coords,
                current_global.coords,
            )
            patch_call = dict(patch_call)
            patch_call.update(kwargs)
            if global_concat_cond is not None:
                patch_call["concat_cond"] = _slice_c128_window_tree(
                    global_concat_cond,
                    indices,
                    window.local_coords,
                    current_global.coords,
                )
            patch_output = sampler.sample_once(
                model, patch_state, t, t_next, **patch_call
            )
            patch_endpoint = _sparse_features(patch_output.pred_x_0)
            weights = window.weights.to(torch.float32)[:, None]
            endpoint_sum.index_add_(
                0, indices, patch_endpoint.to(torch.float32) * weights
            )
            endpoint_weight.index_add_(0, indices, weights)
            del (
                patch_state,
                patch_call,
                patch_output,
                patch_endpoint,
                weights,
            )
        if bool((endpoint_weight <= 0).any().item()):
            raise RuntimeError("C128 master endpoint merge has uncovered tokens")
        global_endpoint = (
            endpoint_sum / endpoint_weight.clamp_min(1e-12)
        ).to(current_global.dtype)
        del endpoint_sum, endpoint_weight

        local_endpoints: List[torch.Tensor] = []
        for index, (state, condition) in enumerate(
            zip(current_local, local_conditions)
        ):
            local_call = dict(condition)
            local_call.update(kwargs)
            if local_concat_cond is not None:
                local_call["concat_cond"] = local_concat_cond[index]
            output = sampler.sample_once(
                model, state, t, t_next, **local_call
            )
            local_endpoints.append(_sparse_features(output.pred_x_0))
            del output, local_call

        fusion = fuse_c128_master_clean_endpoints(
            atom_space,
            global_endpoint,
            local_endpoints,
            local_token_weights=local_token_weights,
            visibility=visibility,
            support_confidence=support_confidence,
            huber_delta=float(huber_delta),
            robust_iterations=int(robust_iterations),
        )
        invariant = float(
            fusion.diagnostics["max_abs_Pcoarse_unified_minus_global"]
        )
        if invariant >= float(invariant_tolerance):
            raise RuntimeError(
                f"C128 clean-endpoint coarse invariant failed at step {step}: "
                f"{invariant:.6g} >= {float(invariant_tolerance):.6g}"
            )

        global_endpoint_value = current_global.replace(global_endpoint)
        global_velocity = sampler._xstart_to_pred(
            current_global, float(t), global_endpoint_value
        )
        next_global = current_global - dt * global_velocity
        next_local: List[Any] = []
        for state, synced_endpoint in zip(
            current_local, fusion.local_synced_endpoints
        ):
            endpoint_value = state.replace(synced_endpoint)
            synchronized_velocity = sampler._xstart_to_pred(
                state, float(t), endpoint_value
            )
            next_local.append(state - dt * synchronized_velocity)
            del endpoint_value, synchronized_velocity
        current_global = next_global
        current_local = next_local
        record = {
            "step": float(step),
            "t": float(t),
            "t_next": float(t_next),
            "dt": float(dt),
            "native_window_count": float(len(windows)),
            **{
                str(key): float(value)
                for key, value in fusion.diagnostics.items()
            },
        }
        records.append(record)
        if progress_callback is not None:
            progress_callback(step + 1, int(steps), record)
        del (
            global_endpoint,
            global_endpoint_value,
            global_velocity,
            next_global,
            local_endpoints,
            fusion,
        )

    final_fusion = fuse_c128_master_clean_endpoints(
        atom_space,
        _sparse_features(current_global),
        [_sparse_features(value) for value in current_local],
        local_token_weights=local_token_weights,
        visibility=visibility,
        support_confidence=support_confidence,
        huber_delta=float(huber_delta),
        robust_iterations=int(robust_iterations),
    )
    final_invariant = float(
        final_fusion.diagnostics["max_abs_Pcoarse_unified_minus_global"]
    )
    if final_invariant >= float(invariant_tolerance):
        raise RuntimeError(
            "final C128 clean-endpoint coarse invariant failed: "
            f"{final_invariant:.6g}"
        )
    return CoupledFlowResult(
        global_samples=current_global,
        local_samples=tuple(current_local),
        final_fusion=final_fusion,
        times=times,
        step_records=tuple(records),
        elapsed_seconds=float(time.perf_counter() - started),
    )


@torch.no_grad()
def run_coupled_endpoint_flow(
    *,
    sampler: object,
    model: torch.nn.Module,
    atom_space: CommonAtomSpace,
    global_state: Any,
    local_states: Sequence[Any],
    global_condition: Mapping[str, Any],
    local_conditions: Sequence[Mapping[str, Any]],
    steps: int,
    rescale_t: float,
    sampler_step_kwargs: Optional[Mapping[str, Any]] = None,
    global_concat_cond: Optional[Any] = None,
    local_concat_cond: Optional[Sequence[Any]] = None,
    local_token_weights: Optional[Sequence[torch.Tensor]] = None,
    visibility: Optional[Sequence[torch.Tensor]] = None,
    support_confidence: Optional[Sequence[torch.Tensor]] = None,
    huber_delta: float = 1.0,
    robust_iterations: int = 3,
    invariant_tolerance: float = 2e-5,
    progress_callback: Optional[
        Callable[[int, int, Mapping[str, float]], None]
    ] = None,
) -> CoupledFlowResult:
    """Run global normally while synchronizing only local clean endpoints."""
    if len(local_states) != len(atom_space.local_mappings):
        raise ValueError("local states are not aligned with atom space")
    if len(local_conditions) != len(local_states):
        raise ValueError("local conditions are not aligned with states")
    if local_concat_cond is not None and len(local_concat_cond) != len(
        local_states
    ):
        raise ValueError("local concat conditions are not aligned")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    times = tuple(
        float(value)
        for value in sampler.timestep_schedule(int(steps), float(rescale_t))
    )
    if len(times) != int(steps) + 1:
        raise RuntimeError("sampler returned an invalid timestep schedule")
    kwargs = dict(sampler_step_kwargs or {})
    current_global = global_state
    current_local = list(local_states)
    records: List[Mapping[str, float]] = []
    started = time.perf_counter()

    for step in range(int(steps)):
        t = times[step]
        t_next = times[step + 1]
        dt = float(t - t_next)
        global_call = dict(global_condition)
        global_call.update(kwargs)
        if global_concat_cond is not None:
            global_call["concat_cond"] = global_concat_cond
        global_out = sampler.sample_once(
            model, current_global, t, t_next, **global_call
        )
        global_endpoint = _sparse_features(global_out.pred_x_0)

        local_outs: List[Any] = []
        local_endpoints: List[torch.Tensor] = []
        for index, (state, condition) in enumerate(
            zip(current_local, local_conditions)
        ):
            local_call = dict(condition)
            local_call.update(kwargs)
            if local_concat_cond is not None:
                local_call["concat_cond"] = local_concat_cond[index]
            output = sampler.sample_once(
                model, state, t, t_next, **local_call
            )
            local_outs.append(output)
            local_endpoints.append(_sparse_features(output.pred_x_0))

        fusion = fuse_clean_endpoints(
            atom_space,
            global_endpoint,
            local_endpoints,
            local_token_weights=local_token_weights,
            visibility=visibility,
            support_confidence=support_confidence,
            huber_delta=float(huber_delta),
            robust_iterations=int(robust_iterations),
        )
        invariant = float(
            fusion.diagnostics["max_abs_Rg_unified_minus_global"]
        )
        if invariant >= float(invariant_tolerance):
            raise RuntimeError(
                f"clean-endpoint low-frequency invariant failed at step {step}: "
                f"{invariant:.6g} >= {float(invariant_tolerance):.6g}"
            )

        next_local: List[Any] = []
        for state, synced_endpoint in zip(
            current_local, fusion.local_synced_endpoints
        ):
            endpoint_value = (
                state.replace(synced_endpoint)
                if hasattr(state, "replace")
                else synced_endpoint
            )
            synchronized_velocity = sampler._xstart_to_pred(
                state, float(t), endpoint_value
            )
            next_local.append(state - dt * synchronized_velocity)
        current_global = global_out.pred_x_prev
        current_local = next_local
        record = {
            "step": float(step),
            "t": float(t),
            "t_next": float(t_next),
            "dt": float(dt),
            **{
                str(key): float(value)
                for key, value in fusion.diagnostics.items()
            },
        }
        records.append(record)
        if progress_callback is not None:
            progress_callback(step + 1, int(steps), record)
        del local_outs

    # At t=0 the state itself is the clean endpoint.  Recompute once so the
    # published unified field corresponds exactly to the saved final samples.
    final_fusion = fuse_clean_endpoints(
        atom_space,
        _sparse_features(current_global),
        [_sparse_features(value) for value in current_local],
        local_token_weights=local_token_weights,
        visibility=visibility,
        support_confidence=support_confidence,
        huber_delta=float(huber_delta),
        robust_iterations=int(robust_iterations),
    )
    final_invariant = float(
        final_fusion.diagnostics["max_abs_Rg_unified_minus_global"]
    )
    if final_invariant >= float(invariant_tolerance):
        raise RuntimeError(
            "final clean-endpoint low-frequency invariant failed: "
            f"{final_invariant:.6g}"
        )
    return CoupledFlowResult(
        global_samples=current_global,
        local_samples=tuple(current_local),
        final_fusion=final_fusion,
        times=times,
        step_records=tuple(records),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def atom_overlap_count(
    atom_space: CommonAtomSpace,
    local_index: int,
) -> torch.Tensor:
    """Count global/local shared atoms for each local token."""
    mapping = atom_space.local_mappings[int(local_index)]
    global_mask = torch.zeros(atom_space.atom_count, dtype=torch.bool)
    global_mask[atom_space.global_atom_indices] = True
    shared = global_mask.index_select(0, mapping.atom_indices)
    counts = torch.zeros(mapping.coords.shape[0], dtype=torch.int64)
    counts.index_add_(
        0, mapping.token_indices, shared.to(torch.int64)
    )
    return counts


def atom_overlap_mass(
    atom_space: CommonAtomSpace,
    local_index: int,
) -> torch.Tensor:
    """Fractional target-cell mass shared with a real global footprint."""
    mapping = atom_space.local_mappings[int(local_index)]
    global_mask = torch.zeros(atom_space.atom_count, dtype=torch.bool)
    global_mask[atom_space.global_atom_indices] = True
    shared = global_mask.index_select(0, mapping.atom_indices)
    mass = torch.zeros(mapping.coords.shape[0], dtype=torch.float32)
    mass.index_add_(
        0,
        mapping.token_indices,
        mapping.overlap_fractions * shared.to(torch.float32),
    )
    return mass
