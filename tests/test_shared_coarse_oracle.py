from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d_shared_coarse_oracle import (  # noqa: E402
    FORMAL_VALID_TILE_IDS,
    PHASE_A_TILE_IDS,
    build_prolongation,
    discover_candidates,
    expected_layout,
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
