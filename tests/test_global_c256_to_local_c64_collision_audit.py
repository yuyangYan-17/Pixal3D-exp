from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pixal3d_global_c256_to_local_c64_collision_audit as audit


CAMERA = {
    "camera_angle_x": 0.517371749106554,
    "distance": 1.889538288116455,
    "mesh_scale": 1.0,
}


def test_4096_floor_downsample_boundary_examples():
    source = torch.tensor([[0, 15, 16], [4095, 4095, 4095]], dtype=torch.int32)
    result = torch.div(source, 16, rounding_mode="floor")
    assert result.tolist() == [[0, 0, 1], [255, 255, 255]]


def test_endpoint_and_cell_center_conventions():
    coords = torch.tensor([[0, 127, 255]], dtype=torch.int32)
    endpoint = audit.endpoint_q(coords, 256)
    center = audit.cell_center_q(coords, 256)
    assert endpoint[0, 0] == -1 and endpoint[0, 2] == 1
    assert torch.allclose(center[0, [0, 2]], torch.tensor([-255 / 256, 255 / 256], dtype=torch.float64))
    c64 = torch.tensor([[0, 31, 63]])
    assert torch.equal(torch.round((audit.endpoint_q(c64, 64) + 1) * 63 / 2).to(torch.int64), c64)


def test_7x7_layout_and_half_open_membership():
    layout = audit.tile_layout()
    assert len(layout) == 49
    assert layout[0]["box"] == (0, 0, 1024, 1024)
    assert layout[48]["box"] == (3072, 3072, 4096, 4096)
    uv = torch.tensor([[0.0, 0.0], [1023.999, 2.0], [1024.0, 2.0]])
    assert audit.half_open_membership(uv, layout[0]["box"]).tolist() == [True, True, False]


def test_exact_camera_continuous_roundtrip():
    transform = audit.core._derive_tile_camera(
        tile_id=24, box=(1536, 1536, 2560, 2560), global_camera=CAMERA,
        source_width=4096, source_height=4096, model_width=1024, model_height=1024,
        extend_pixel=0,
    )
    q = torch.tensor([[-0.1, 0.2, -0.4], [0.3, -0.2, 0.7]], dtype=torch.float64)
    local, _ = audit.core._global_q_to_local_q(q, global_camera=CAMERA, transform=transform)
    back, _ = audit.core._local_q_to_global_q(local, global_camera=CAMERA, transform=transform)
    assert (q - back).abs().max().item() < 2e-5


def test_synthetic_one_to_one_has_zero_collision():
    coords = torch.tensor([[1, 2, 3], [2, 2, 3], [3, 2, 3]], dtype=torch.int32)
    result = audit.collision_stats(coords, torch.tensor([9, 10, 11]))
    assert result["collision_cell_count"] == 0
    assert result["collision_excess_rows"] == 0
    assert result["max_collision_multiplicity"] == 1


def test_synthetic_many_to_one_collision_counts():
    coords = torch.tensor([[1, 2, 3], [1, 2, 3], [8, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=torch.int32)
    result = audit.collision_stats(coords, torch.arange(5))
    assert result["collision_cell_count"] == 2
    assert result["collided_row_count"] == 5
    assert result["collision_excess_rows"] == 3
    assert result["max_collision_multiplicity"] == 3


def test_collision_unique_keeps_original_global_row_ids():
    coords = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=torch.int32)
    groups, _, _ = audit.collision_groups(coords, torch.tensor([41, 7, 99]))
    assert len(groups) == 1
    assert groups[0]["global_row_ids"] == [41, 7]


def test_cross_tile_membership_is_not_within_tile_collision():
    uv = torch.tensor([[750.0, 200.0]])
    first = audit.half_open_membership(uv, (0, 0, 1024, 1024))
    second = audit.half_open_membership(uv, (512, 0, 1536, 1024))
    assert first.item() and second.item()
    for _ in range(2):
        result = audit.collision_stats(torch.tensor([[4, 5, 6]], dtype=torch.int32), torch.tensor([123]))
        assert result["collision_excess_rows"] == 0


def test_quantized_roundtrip_histogram_and_identity_fraction():
    coords = torch.tensor([[0, 0, 0], [10, 20, 30]], dtype=torch.int32)
    reconstructed = audit.endpoint_q(torch.tensor([[0, 0, 0], [11, 20, 28]]), 256)
    metrics, identity = audit._quantized_metrics(coords, reconstructed)
    assert identity.tolist() == [True, False]
    assert metrics["identity_fraction"] == 0.5
    assert metrics["index_l1_histogram"] == {"0": 1, "3": 1}
    assert metrics["index_linf_histogram"] == {"0": 1, "2": 1}


def test_cache_fingerprint_mismatch_rejected():
    payload = {
        "format": f"{audit.FORMAT}_c4096_cache",
        "source_mesh_sha256": "right",
        "voxelizer_config": audit.VOXELIZER_CONFIG,
        "coords": torch.zeros((1, 3), dtype=torch.int32),
    }
    assert audit.cache_matches(payload, "right")
    assert not audit.cache_matches(payload, "wrong")
    changed = dict(payload); changed["voxelizer_config"] = {**audit.VOXELIZER_CONFIG, "grid_size": 12}
    assert not audit.cache_matches(changed, "right")


def test_global_support_schema_excludes_ovoxel_pbr_and_features():
    payload = {
        "format": audit.FORMAT, "resolution": 256,
        "coords": torch.zeros((1, 3), dtype=torch.int32),
        "global_row_ids": torch.arange(1), "source_mesh_sha256": "hash",
        "voxelizer_config": audit.VOXELIZER_CONFIG,
        "downsample_config": audit.DOWNSAMPLE_CONFIG,
    }
    assert audit.support_schema_is_clean(payload)
    for forbidden in ("attrs", "features", "dual_vertices", "intersected", "pbr"):
        assert not audit.support_schema_is_clean({**payload, forbidden: torch.zeros(1)})
