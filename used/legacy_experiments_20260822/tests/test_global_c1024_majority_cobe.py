from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d_global_c1024_majority_cobe import (  # noqa: E402
    _cobe_from_bases,
    _error_for_subspace,
    bh_fdr,
    cobe_candidates,
    effective_rank_basis,
    orthonormal_columns,
    principal_angle_summary,
    projection_energy,
)


def _angles(left: torch.Tensor, right: torch.Tensor):
    return principal_angle_summary(left, right)["angles_deg"]


def _synthetic_shared(
    *,
    rows: int = 80,
    shared_rank: int = 2,
    block_count: int = 5,
    private_rank: int = 2,
    seed: int = 7,
):
    generator = torch.Generator().manual_seed(seed)
    shared, _ = torch.linalg.qr(torch.randn(rows, shared_rank, generator=generator, dtype=torch.float64))
    blocks = []
    for block_index in range(block_count):
        private, _ = torch.linalg.qr(torch.randn(rows, private_rank, generator=generator, dtype=torch.float64))
        # Orthogonalise private directions against the exact common subspace.
        private = orthonormal_columns(private - shared @ (shared.T @ private))
        coefficients = torch.randn(shared_rank, 6, generator=generator, dtype=torch.float64)
        private_coefficients = torch.randn(private.shape[1], 6, generator=generator, dtype=torch.float64)
        blocks.append(shared @ coefficients + private @ private_coefficients)
    return shared, blocks


def test_exact_all_shared_recovers_common_subspace():
    shared, blocks = _synthetic_shared(shared_rank=2, private_rank=2)
    result = cobe_candidates(blocks, max_components=2, max_iter=200, convergence_tol=1e-12, seed=3)
    assert result.rank == 2
    assert max(_angles(shared, result.basis)) < 1e-5
    assert torch.max(torch.abs(result.basis.T @ result.basis - torch.eye(2, dtype=torch.float64))) < 1e-10


def test_different_amplitude_does_not_change_common_space():
    shared, blocks = _synthetic_shared(shared_rank=1, private_rank=2)
    amplitudes = [0.01, 0.2, 3.0, 17.0, 0.07]
    scaled = [block.clone() for block in blocks]
    for index, block in enumerate(scaled):
        scaled[index] = block + (amplitudes[index] - 1.0) * shared @ torch.randn(1, 6, generator=torch.Generator().manual_seed(100 + index), dtype=torch.float64)
    result = cobe_candidates(scaled, max_components=1, max_iter=200, convergence_tol=1e-12, seed=5)
    assert max(_angles(shared, result.basis)) < 1e-5


def test_high_energy_private_direction_does_not_replace_shared_candidate():
    generator = torch.Generator().manual_seed(11)
    shared, _ = torch.linalg.qr(torch.randn(100, 1, generator=generator, dtype=torch.float64))
    private_directions = []
    for _ in range(5):
        private, _ = torch.linalg.qr(torch.randn(100, 1, generator=generator, dtype=torch.float64))
        private_directions.append(private)
    blocks = []
    for index in range(5):
        private = private_directions[index] - shared @ (shared.T @ private_directions[index])
        private = private / torch.linalg.vector_norm(private)
        private_amp = 100.0 if index == 4 else 1.0
        blocks.append(torch.cat([shared, private], dim=1) @ torch.tensor([[0.01, 0.0, 0.0, 0.0], [0.0, private_amp, 0.0, 0.0]], dtype=torch.float64))
    result = cobe_candidates(blocks, max_components=1, max_iter=200, convergence_tol=1e-12, seed=9)
    assert max(_angles(shared, result.basis)) < 1e-5


def test_majority_shared_omit_held_out_has_low_support():
    generator = torch.Generator().manual_seed(13)
    shared, _ = torch.linalg.qr(torch.randn(90, 1, generator=generator, dtype=torch.float64))
    blocks = [shared @ torch.randn(1, 4, generator=generator, dtype=torch.float64) for _ in range(4)]
    private, _ = torch.linalg.qr(torch.randn(90, 2, generator=generator, dtype=torch.float64))
    blocks.append(private @ torch.randn(2, 4, generator=generator, dtype=torch.float64))
    bases = [effective_rank_basis(block).basis for block in blocks]
    subset = _cobe_from_bases([bases[0], bases[1], bases[2], bases[3]], max_components=1, max_iter=200, convergence_tol=1e-12, seed=4)
    assert max(_angles(shared, subset.basis)) < 1e-5
    scores = [float(torch.sum((basis.T @ subset.basis) ** 2).item()) for basis in bases]
    assert min(scores[:4]) > 0.99
    assert scores[4] < 0.1


def test_all_shared_leave_one_out_is_stable():
    shared, blocks = _synthetic_shared(shared_rank=1, private_rank=2)
    bases = [effective_rank_basis(block).basis for block in blocks]
    results = [_cobe_from_bases([bases[index] for index in range(5) if index != omit], max_components=1, max_iter=200, convergence_tol=1e-12, seed=2) for omit in range(5)]
    for result in results:
        assert max(_angles(shared, result.basis)) < 1e-5
    for left in results:
        for right in results:
            assert principal_angle_summary(left.basis, right.basis)["mean_cos2"] == pytest.approx(1.0, abs=1e-8)


def test_fixed_parent_rows_are_reused_for_subset():
    rows = torch.arange(17)
    parent = rows[torch.tensor([True, False, True, True, False, True, True, False, True, True, False, True, True, True, False, True, True])]
    subset = parent.clone()
    assert subset.shape == parent.shape
    assert torch.equal(subset, parent)


def test_permutation_null_breaks_aligned_row_support():
    generator = torch.Generator().manual_seed(21)
    shared, blocks = _synthetic_shared(rows=100, shared_rank=1, private_rank=1, seed=21)
    real = cobe_candidates(blocks, max_components=1, max_iter=200, convergence_tol=1e-12, seed=1)
    shuffled = [blocks[0]] + [block[torch.randperm(block.shape[0], generator=torch.Generator().manual_seed(90 + index))] for index, block in enumerate(blocks[1:])]
    null = cobe_candidates(shuffled, max_components=1, max_iter=200, convergence_tol=1e-12, seed=1)
    assert real.errors[0].item() < null.errors[0].item()
    assert real.support_scores[:, 0].mean().item() > null.support_scores[:, 0].mean().item()


def test_rank_deficient_zero_and_constant_columns_are_safe():
    matrix = torch.tensor([[1.0, 0.5, 0.0, 7.0], [2.0, 0.5, 0.0, 7.0], [3.0, 0.5, 0.0, 7.0]], dtype=torch.float64)
    result = effective_rank_basis(matrix)
    assert result.rank == 2
    assert result.basis.shape == (3, 2)
    assert torch.isfinite(result.basis).all()


def test_common_space_is_invariant_to_nonzero_column_rescaling():
    shared, blocks = _synthetic_shared(shared_rank=2, private_rank=0)
    scales = [torch.tensor([0.1, 3.0, 7.0, 0.5, 2.0, 11.0], dtype=torch.float64) for _ in blocks]
    scaled = [block * scale[None, :] for block, scale in zip(blocks, scales)]
    original = cobe_candidates(blocks, max_components=2, max_iter=200, convergence_tol=1e-12, seed=8)
    transformed = cobe_candidates(scaled, max_components=2, max_iter=200, convergence_tol=1e-12, seed=8)
    assert principal_angle_summary(original.basis, transformed.basis)["mean_cos2"] == pytest.approx(1.0, abs=1e-8)
    assert max(_angles(shared, transformed.basis)) < 1e-5


def test_pipeline_has_no_delta_input_and_projection_energy_is_channelwise():
    source = Path(__file__).resolve().parents[1] / "pixal3d_global_c1024_majority_cobe.py"
    text = source.read_text(encoding="utf-8")
    assert "Delta = H - G" not in text
    assert "H - G" not in text
    basis = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    field = torch.tensor([[2.0, 1.0], [0.0, 3.0], [0.0, 0.0]], dtype=torch.float64)
    energy = projection_energy(field, basis)
    assert energy[0].item() == pytest.approx(1.0)
    assert energy[1].item() == pytest.approx(0.1)


def test_bh_fdr_preserves_order_and_missing_values():
    q = bh_fdr([0.01, None, 0.04, 0.2])
    assert q[1] is None
    assert q[0] <= q[2] <= q[3]
