"""Small deterministic tests for the isolated global mesh revoxelizer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "root"))

from pixal3d_ovoxel_global_mesh_revoxelize_merge import (
    EDGE_CELL_OFFSETS,
    _decode_edge_keys,
    _edge_cells,
    _edge_keys,
    global_to_local,
    local_to_global,
)


def test_uniform_3d_placement_roundtrip() -> None:
    points = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [-0.1, 0.2, 0.4],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    origin = (1536, 512, 1024)
    placed = local_to_global(points, origin)
    restored = global_to_local(placed, origin)
    assert float(np.abs(restored - points).max()) < 1e-5


def test_canonical_edge_key_and_four_cell_closure() -> None:
    coord = np.asarray([[37, 41, 43]], dtype=np.int32)
    axis = np.asarray([2], dtype=np.int8)
    key = _edge_keys(coord, axis)
    decoded_coord, decoded_axis = _decode_edge_keys(key)
    assert np.array_equal(decoded_coord, coord)
    assert np.array_equal(decoded_axis, axis)
    assert np.array_equal(_edge_cells(coord, axis)[0], coord[0] + EDGE_CELL_OFFSETS[2])


def test_local_voxel_convention_is_not_double_scaled() -> None:
    points = np.asarray([[0.0, 512.0, 1024.0]], dtype=np.float32)
    origin = (512, 1024, 0)
    placed = local_to_global(points, origin, "local_voxel")
    expected = -0.5 + (np.asarray(origin, dtype=np.float32) + points) / 4096.0
    assert np.allclose(placed, expected, atol=1e-7)
