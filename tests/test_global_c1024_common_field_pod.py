from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d_global_c1024_common_field_pod import (
    build_sparse_query_matrix,
    official_layout,
    pairwise_cosine_from_gram,
    uncentered_pod,
)


def _apply(q, values):
    return torch.from_numpy(q.dot(values.numpy())).to(torch.float32)


def test_common_support_query_row_order_is_identical():
    points = torch.tensor([[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]], dtype=torch.float32)
    support_a = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.int32)
    support_b = support_a[[2, 0, 1]]
    q_a, valid_a, _ = build_sparse_query_matrix(support_a, points, resolution=4)
    q_b, valid_b, _ = build_sparse_query_matrix(support_b, points, resolution=4)
    values_a = torch.tensor([[10.0], [20.0], [30.0]])
    values_b = values_a[[2, 0, 1]]
    assert torch.equal(valid_a, valid_b)
    assert torch.allclose(_apply(q_a, values_a), _apply(q_b, values_b), atol=1e-7)


def test_missing_is_not_zero_and_is_excluded():
    support = torch.tensor([[1, 1, 1]], dtype=torch.int32)
    points = torch.tensor([[-0.125, -0.125, -0.125], [0.375, 0.375, 0.375]], dtype=torch.float32)
    q, valid, meta = build_sparse_query_matrix(support, points, resolution=4)
    assert valid.tolist() == [True, False]
    assert meta["invalid_rows"] == 1
    result = _apply(q, torch.tensor([[7.0]]))
    assert result[1].item() == 0.0
    assert not bool(valid[1])


def test_query_matrix_matches_manual_sparse_trilinear_renormalization():
    resolution = 4
    support = torch.tensor([[1, 1, 1], [2, 1, 1], [1, 2, 1]], dtype=torch.int32)
    point = torch.tensor([[0.0, 0.0, -0.125]], dtype=torch.float32)
    q, valid, _ = build_sparse_query_matrix(support, point, resolution=resolution)
    assert bool(valid[0])
    # The point is halfway between the centers of cells 1 and 2 in x/y, with
    # only three of the eight neighbours present.  Each present neighbour has
    # the same raw weight and is renormalized to one third.
    values = torch.tensor([[3.0], [9.0], [15.0]])
    assert torch.allclose(_apply(q, values), torch.tensor([[9.0]]), atol=1e-7)


def test_own_center_query_is_one_hot_for_active_voxel():
    support = torch.tensor([[0, 0, 0], [2, 1, 3]], dtype=torch.int32)
    points = -0.5 + (support.to(torch.float32) + 0.5) / 8.0
    q, valid, _ = build_sparse_query_matrix(support, points, resolution=8)
    values = torch.tensor([[2.0, 4.0], [8.0, 16.0]])
    assert valid.all()
    assert torch.allclose(_apply(q, values), values, atol=1e-6)


def test_uncentered_pod_shared_amplitude_direction():
    u = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    columns = torch.cat([0.4 * u, 0.7 * u, 0.5 * u], dim=1)
    pod = uncentered_pod(columns)
    assert pod["energy_ratio"][0].item() == pytest.approx(1.0, abs=1e-10)
    assert pod["directional_energy_ratio"][0].item() == pytest.approx(1.0, abs=1e-10)
    cosine = pairwise_cosine_from_gram(pod["gram"], pod["norms"])
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-10)


def test_uncentered_pod_private_direction_is_not_rank_one():
    u = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    v = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    columns = torch.cat([u, u, v], dim=1)
    pod = uncentered_pod(columns)
    assert pod["energy_ratio"][0].item() < 0.999
    assert pod["energy_ratio"][:2].sum().item() == pytest.approx(1.0, abs=1e-10)
    cosine = pairwise_cosine_from_gram(pod["gram"], pod["norms"])
    assert cosine[0, 1].item() == pytest.approx(1.0, abs=1e-10)
    assert abs(cosine[0, 2].item()) < 1e-10


def test_pod_does_not_mean_center_columns():
    columns = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    pod = uncentered_pod(columns)
    expected = columns.to(torch.float64).T @ columns.to(torch.float64)
    assert torch.allclose(pod["gram"], expected)
    assert pod["energy_ratio"][0].item() == pytest.approx(1.0, abs=1e-10)


def test_official_phase_layout_contains_required_tile_ids():
    boxes = official_layout()
    assert len(boxes) == 49
    assert [boxes[i] for i in sorted({18, 19, 20, 25, 26, 27, 32, 33, 34})]
