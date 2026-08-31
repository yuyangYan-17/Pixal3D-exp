import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d_c64_c256_flow_consistency import (
    build_c256_tiles,
    build_parent_mapping,
    metric_row,
    per_parent_variance,
    segment_mean,
)


def test_parent_mapping_is_exact_floor_and_total():
    coords = torch.tensor([[0, 0, 0, 0], [0, 3, 2, 1], [0, 4, 0, 0], [0, 7, 3, 3], [0, 255, 255, 255]], dtype=torch.int32)
    coarse, parent, offsets = build_parent_mapping(coords)
    assert coarse.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 63, 63, 63]]
    assert parent.tolist() == [0, 0, 1, 1, 2]
    assert offsets.tolist() == [0, 2, 4, 5]
    assert torch.equal(coarse[parent, 1:], coords[:, 1:] // 4)


def test_c256_nonoverlap_tiles_apply_exact_local_c64_transform():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 63, 63, 63], [0, 64, 64, 64], [0, 127, 0, 255], [0, 255, 255, 255]],
        dtype=torch.int32,
    )
    records, stats = build_c256_tiles(coords)
    assert len(records) == 64
    assert stats["tile_count"] == 64
    seen = torch.zeros(len(coords), dtype=torch.int32)
    for record in records:
        ids = record["global_row_ids"]
        if ids.numel():
            start = torch.tensor(record["start_c256"], dtype=torch.int32)
            assert torch.equal(record["local_coords"][:, 1:] + start, coords[ids, 1:])
            assert int(record["local_coords"][:, 1:].min()) >= 0
            assert int(record["local_coords"][:, 1:].max()) < 64
        seen.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    assert torch.equal(seen, torch.ones_like(seen))


def test_segment_mean_and_population_variance():
    values = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 2.0]])
    parent = torch.tensor([0, 0, 1])
    assert torch.equal(segment_mean(values, parent, 2), torch.tensor([[2.0, 0.0], [5.0, 2.0]]))
    assert torch.equal(per_parent_variance(values, parent, 2), torch.tensor([1.0, 0.0]))


def test_identical_parent_broadcast_has_perfect_metrics_and_zero_fine_energy():
    coarse = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    parent = torch.tensor([0, 0, 1])
    fine = coarse[parent]
    row, detail = metric_row(1, 1.0, 0.8, coarse, coarse, coarse, fine, fine, fine, parent, 2)
    assert abs(row["endpoint_cosine_mean"] - 1.0) < 1e-6
    assert row["endpoint_relative_l2"] == 0.0
    assert abs(row["velocity_cosine_mean"] - 1.0) < 1e-6
    assert row["velocity_relative_l2"] == 0.0
    assert row["children_variance_mean"] == 0.0
    assert row["fine_coarse_energy_ratio"] == 0.0
    assert torch.all(detail["endpoint_cosine"] > 0.99999)
