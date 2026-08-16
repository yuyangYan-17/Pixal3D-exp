from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d_shared_coarse_oracle import (  # noqa: E402
    FORMAL_VALID_TILE_IDS,
    PHASE_A_TILE_IDS,
    build_prolongation,
    compute_l2_consensus,
    construct_oracles,
    discover_candidates,
    expected_layout,
    _donor_query_field,
    _load_operator_cache,
    _pairwise_metrics,
    partition_hidden_operator,
    phase_a_ids,
    solve_direct_lsmr,
)


def test_official_layout_and_phase_a_ids_are_exact():
    boxes = expected_layout()
    assert len(boxes) == 49
    assert boxes[0] == (0, 0, 1024, 1024)
    assert boxes[26] == (2560, 1536, 3584, 2560)
    assert boxes[-1] == (3072, 3072, 4096, 4096)
    assert phase_a_ids() == set(PHASE_A_TILE_IDS)
    assert len(PHASE_A_TILE_IDS) == 9
    assert len(FORMAL_VALID_TILE_IDS) == 48
    assert 6 not in FORMAL_VALID_TILE_IDS


def test_sparse_trilinear_prolongation_renormalizes_missing_support():
    coarse = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    points = torch.tensor(
        [
            [-0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0],
            [-0.5 + 1.0 / 256.0, -0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0],
        ],
        dtype=torch.float32,
    )
    matrix, info = build_prolongation(coarse, points)
    assert matrix.dtype == np.float64
    assert info["uncovered_rows"] == 0
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    assert matrix[0, 0] == 1.0


def test_direct_float64_lsmr_projection_has_transpose_residual():
    coarse = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.int32)
    points = torch.tensor(
        [
            [-0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0],
            [-0.5 + 1.5 / 256.0, -0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0],
            [-0.5 + 2.5 / 256.0, -0.5 + 0.5 / 256.0, -0.5 + 0.5 / 256.0],
        ],
        dtype=torch.float32,
    )
    operator, _ = build_prolongation(coarse, points)
    field = torch.tensor([[1.0, 0.0], [0.5, 0.25], [0.0, 1.0]], dtype=torch.float64)
    coeff, info = solve_direct_lsmr(operator, field, label="test", maxiter=100)
    projected = torch.from_numpy(operator.dot(coeff.numpy()))
    residual = torch.from_numpy(operator.T.dot((field - projected).numpy()))
    assert info["normal_equation_used"] is False
    assert max(channel["relative_residual"] for channel in info["channels"]) < 1e-7
    assert float(residual.abs().max()) < 1e-7


def test_endpoint_discovery_uses_canonical_box_across_legacy_tile_ids(tmp_path):
    root = tmp_path / "cross_tile_pbr_perstep_guided_cuda4"
    tile = root / "tiles" / "tile_06"
    tile.mkdir(parents=True)
    json_path = tile / "tile_camera.json"
    json_path.write_text(json.dumps({"box": list(expected_layout()[18])}), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps({"pure_HR": {"route": "native FlowEulerSampler.sample_once suffix; no endpoint guidance"}}),
        encoding="utf-8",
    )
    torch.save({"coords": torch.zeros((1, 4), dtype=torch.int32), "features": torch.zeros((1, 2))}, tile / "pure_HR_endpoint.pt")

    _, endpoints = discover_candidates(18, None, [root])
    assert len(endpoints) == 1
    assert endpoints[0].match_mode == "tile_camera_box"
    assert endpoints[0].rejection is None


def test_lsmr_accepts_stationary_out_of_range_residual_without_normal_equations():
    operator = csr_matrix(np.asarray([[1.0], [0.0]], dtype=np.float64))
    field = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    _, info = solve_direct_lsmr(operator, field, label="out-of-range", maxiter=1)
    channel = info["channels"][0]
    assert channel["relative_residual"] > 0.5
    assert channel["scaled_transpose_residual"] <= channel["scaled_transpose_tolerance"]
    assert info["normal_equation_used"] is False


def test_hidden_operator_partition_excludes_basis_touching_observed_row():
    full = csr_matrix(np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float64))
    hidden_operator, pure_ids = partition_hidden_operator(
        full,
        torch.tensor([True, False]),
    )
    np.testing.assert_array_equal(pure_ids, np.asarray([1], dtype=np.int64))
    np.testing.assert_allclose(hidden_operator.toarray(), np.asarray([[1.0]]))


def test_donor_query_field_is_C_not_Delta():
    donor = SimpleNamespace(
        C=torch.full((3, 6), 2.0),
        Delta=torch.full((3, 6), 99.0),
    )
    assert torch.equal(_donor_query_field(donor), donor.C)
    assert not torch.equal(_donor_query_field(donor), donor.Delta)


def test_observed_identity_and_shared_algebra_are_exact():
    H = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)
    G = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    C = torch.tensor([[0.0, 0.0], [0.2, 0.1]], dtype=torch.float32)
    C_shared = torch.tensor([[0.0, 0.0], [0.05, 0.02]], dtype=torch.float32)
    Y_null, Y_shared, C_private = construct_oracles(
        H, G, C, C_shared, torch.tensor([False, True])
    )
    assert torch.equal(Y_null[0], H[0])
    assert torch.equal(Y_shared[0], H[0])
    assert torch.equal(Y_shared[1], H[1] - C_private[1])
    torch.testing.assert_close(Y_shared[1], G[1] + C_shared[1] + (H - G - C)[1])


def test_operator_cache_rejects_fine_coordinate_permutation(tmp_path):
    root = tmp_path / "operators" / "tiles" / "tile_00"
    root.mkdir(parents=True)
    save_npz(root / "mra_P_hidden.npz", csr_matrix(np.asarray([[1.0]], dtype=np.float32)))
    torch.save(
        {
            "fine_coords": torch.tensor([[0, 0, 0]], dtype=torch.int32),
            "coarse_coords": torch.tensor([[0, 0, 0]], dtype=torch.int32),
            "hidden_rows": torch.tensor([0], dtype=torch.int64),
            "pure_hidden_ids": torch.tensor([0], dtype=torch.int64),
        },
        root / "mra_support.pt",
    )
    (root / "mra_operator.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(operator_data_tolerance=1e-12)
    with pytest.raises(ValueError, match="C1024 coords"):
        _load_operator_cache(
            operator_cache_dir=tmp_path / "operators",
            tile_id=0,
            fine_coords=torch.tensor([[0, 1, 0]], dtype=torch.int32),
            hidden_mask=torch.tensor([True]),
            rebuilt_coarse_coords=torch.tensor([[0, 0, 0]], dtype=torch.int32),
            args=args,
        )


def test_pairwise_disagreement_uses_mean_norm_denominator():
    a = torch.tensor([[3.0, 0, 0, 0, 0, 0]])
    b = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    metrics = _pairwise_metrics([(1, a, torch.tensor([True])), (2, b, torch.tensor([True]))], 10)
    assert metrics["relative_disagreement"]["RGB"]["mean"] == pytest.approx(1.0)
    assert metrics["cosine"]["RGB"]["mean"] == pytest.approx(1.0)


def test_zero_norm_cosine_is_excluded_from_statistics():
    zero = torch.zeros((1, 6))
    nonzero = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    metrics = _pairwise_metrics([(1, zero, torch.tensor([True])), (2, nonzero, torch.tensor([True]))], 10)
    assert metrics["valid_cosine_pair_count"]["RGB"] == 0
    assert metrics["zero_norm_pair_count"]["RGB"] == 1
    assert metrics["cosine"]["RGB"]["count"] == 0


def test_consensus_is_invariant_to_input_donor_order():
    first = [(9, torch.full((2, 1), 2.0), torch.tensor([True, False])), (3, torch.full((2, 1), 4.0), torch.tensor([True, True]))]
    second = list(reversed(first))
    raw_a, count_a, ids_a = compute_l2_consensus(first)
    raw_b, count_b, ids_b = compute_l2_consensus(second)
    assert ids_a == ids_b == [3, 9]
    assert torch.equal(count_a, count_b)
    assert torch.equal(raw_a, raw_b)


def test_uncovered_hidden_row_has_zero_C_and_null_preserves_H():
    P = csr_matrix(np.asarray([[1.0], [0.0]], dtype=np.float64))
    field = torch.tensor([[2.0], [7.0]], dtype=torch.float64)
    coeff, _ = solve_direct_lsmr(P, field, label="uncovered")
    C = torch.from_numpy(P.dot(coeff.numpy()))
    H = field.to(torch.float32)
    G = torch.zeros_like(H)
    Y_null, _, _ = construct_oracles(H, G, C.to(torch.float32), torch.zeros_like(C, dtype=torch.float32), torch.tensor([True, True]))
    assert C[1].item() == 0.0
    assert torch.equal(Y_null[1], H[1])


def test_stationarity_for_detail_and_shared_residual():
    P = csr_matrix(np.asarray([[1.0], [0.0]], dtype=np.float64))
    detail = torch.tensor([[0.0], [3.0]], dtype=torch.float64)
    shared_residual = torch.tensor([[0.0], [-5.0]], dtype=torch.float64)
    assert float(np.abs(P.T.dot(detail.numpy())).max()) == 0.0
    assert float(np.abs(P.T.dot(shared_residual.numpy())).max()) == 0.0


def test_phase_a_route_metadata_has_no_flow_or_encoder_operations():
    # This is the contract consumed by preflight/run: Stage 1 is field-only;
    # the only allowed model operation lives in the separate materializer.
    source = Path(__file__).resolve().parents[1] / "pixal3d_shared_coarse_oracle.py"
    text = source.read_text(encoding="utf-8")
    phase_a_body = text.split("def build_consensus(", 1)[1].split("def _order_invariance", 1)[0]
    assert "_run_pure_hr_flow" not in phase_a_body
    assert "_xstart_to_pred" not in phase_a_body
    assert "torch.cuda" not in phase_a_body
