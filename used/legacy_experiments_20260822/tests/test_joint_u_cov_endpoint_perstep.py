from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d_joint_u_cov_endpoint_perstep import (  # noqa: E402
    GLOBAL_U_RESOLUTION,
    _make_correction,
    _normalize_slat,
    _denormalize_slat,
    build_sparse_c4096_operator,
    covariance_vector_product,
    differentiable_sparse_trilinear_query,
    _operator_apply_unweighted,
    _solve_joint_u,
)
from pixal3d.modules.sparse import SparseTensor  # noqa: E402
from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler  # noqa: E402


def _manual_sparse_query(support, values, points, resolution):
    support = {tuple(row.tolist()): values[i] for i, row in enumerate(support)}
    result = []
    for point in points:
        grid = (point + 0.5) * resolution
        base = torch.floor(grid - 0.5).to(torch.int64)
        frac = grid - (base.to(torch.float32) + 0.5)
        out = torch.zeros(values.shape[1], dtype=values.dtype)
        total = torch.tensor(0.0, dtype=values.dtype)
        for bits in range(8):
            bit = torch.tensor([(bits >> axis) & 1 for axis in range(3)])
            coord = tuple((base + bit).tolist())
            weight = torch.where(bit.bool(), frac, 1.0 - frac).prod()
            if coord in support and bool(((base + bit) >= 0).all()) and bool(((base + bit) < resolution).all()):
                out = out + support[coord] * weight
                total = total + weight
        result.append(out / total)
    return torch.stack(result)


def test_normalization_roundtrip_preserves_features_and_coords():
    coords = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=torch.int32)
    value = SparseTensor(torch.randn(2, 32), coords)
    normalization = {"mean": [0.25] * 32, "std": [2.5] * 32}
    recovered = _denormalize_slat(_normalize_slat(value, normalization), normalization)
    assert torch.equal(value.coords, recovered.coords)
    assert torch.allclose(value.feats, recovered.feats, atol=1e-6, rtol=1e-6)


def test_covariance_matrix_free_matches_dense_product():
    generator = torch.Generator().manual_seed(4)
    centered = torch.randn(8, 5, 3, generator=generator)
    gradient = torch.randn(5, 3, generator=generator)
    sigma2 = 0.031
    result, low_rank, isotropic = covariance_vector_product(centered, gradient, sigma2)
    b = centered.reshape(8, -1) / (7.0 ** 0.5)
    dense = (b.T @ b + sigma2 * torch.eye(15)) @ gradient.reshape(-1)
    assert torch.allclose(result.reshape(-1), dense, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result, low_rank + isotropic, atol=1e-7)


def test_covariance_support_mismatch_is_hard_failure():
    centered = torch.randn(8, 4, 2)
    with pytest.raises(ValueError, match="supports differ"):
        covariance_vector_product(centered, torch.randn(5, 2), 0.01)


def test_rms_endpoint_correction_is_normalized_trust_region():
    coords = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32)
    endpoint = SparseTensor(torch.zeros(2, 3), coords)
    direction = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
    corrected, delta_rms = _make_correction(endpoint, direction, rho=0.17)
    assert torch.equal(corrected.coords, endpoint.coords)
    assert delta_rms == pytest.approx(0.17, abs=1e-6)
    assert torch.sqrt(torch.mean((corrected.feats - endpoint.feats).square())).item() == pytest.approx(0.17, abs=1e-6)


def test_a_operator_matches_manual_sparse_trilinear_convention():
    resolution = 16
    support = torch.tensor(
        [[2, 2, 2], [3, 2, 2], [2, 3, 2], [3, 3, 2], [2, 2, 3]],
        dtype=torch.int32,
    )
    points = torch.tensor(
        [[-0.5 + 2.5 / resolution, -0.5 + 2.5 / resolution, -0.5 + 2.5 / resolution],
         [0.0, 0.0, 0.0],
         [-0.5 + 2.75 / resolution, -0.5 + 2.25 / resolution, -0.5 + 2.5 / resolution]],
        dtype=torch.float32,
    )
    # Include the exact support in an operator cache by using points whose
    # stencil is precisely contained in the supplied support.  For the
    # sparse-query convention, missing neighbors are renormalized.
    operator = build_sparse_c4096_operator([("global", points, 1.0)], resolution=resolution, chunk_size=32)
    values = torch.arange(operator.support_coords.shape[0], dtype=torch.float32)[:, None]
    got = _operator_apply_unweighted(operator, 0, values)
    expected = _manual_sparse_query(operator.support_coords, values, points, resolution)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
    assert operator.metadata["dense_4096_cube_allocated"] is False
    assert operator.matrix.nnz <= points.shape[0] * 8


def test_u_synthetic_recovery_has_zero_observation_residual():
    resolution = 16
    points_g = torch.tensor([[-0.25, -0.25, -0.25], [0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    points_t = torch.tensor([[-0.2, -0.1, -0.15], [0.1, 0.2, 0.15], [0.2, -0.1, 0.05]])
    operator = build_sparse_c4096_operator(
        [("global", points_g, 1.0), ("tile_7", points_t, 0.75)],
        resolution=resolution,
        chunk_size=32,
    )
    true_u = torch.randn(operator.support_coords.shape[0], 6, generator=torch.Generator().manual_seed(8))
    global_field = _operator_apply_unweighted(operator, 0, true_u)
    tile_field = _operator_apply_unweighted(operator, 1, true_u)
    solved, stats = _solve_joint_u(operator, global_field, {7: tile_field}, maxiter=200)
    assert torch.isfinite(solved).all()
    assert stats["iterations_mean"] > 0
    assert torch.allclose(_operator_apply_unweighted(operator, 0, solved.to(torch.float32)), global_field, atol=1e-5, rtol=1e-5)
    assert torch.allclose(_operator_apply_unweighted(operator, 1, solved.to(torch.float32)), tile_field, atol=1e-5, rtol=1e-5)


def test_differentiable_query_has_finite_input_gradient():
    coords = torch.tensor([[0, 1, 1, 1], [0, 2, 1, 1], [0, 1, 2, 1], [0, 1, 1, 2]], dtype=torch.int32)
    attrs = torch.randn(4, 6, requires_grad=True)
    points = torch.tensor([[-0.5 + 1.5 / 8, -0.5 + 1.5 / 8, -0.5 + 1.5 / 8]])
    result = differentiable_sparse_trilinear_query(attrs, coords, points, resolution=8)
    loss = result.square().sum()
    loss.backward()
    assert attrs.grad is not None
    assert attrs.grad.shape == attrs.shape
    assert torch.isfinite(attrs.grad).all()


def test_rho_zero_official_euler_equivalence():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    state = torch.randn(7, 4, generator=torch.Generator().manual_seed(9))
    velocity = torch.randn(7, 4, generator=torch.Generator().manual_seed(10))
    t, t_next = 0.375, 0.21428571428571436
    pure_next = state - (t - t_next) * velocity
    corrected_endpoint = sampler._pred_to_xstart(state, t, velocity)
    corrected_velocity = sampler._xstart_to_pred(state, t, corrected_endpoint)
    guided_next = state - (t - t_next) * corrected_velocity
    assert torch.allclose(pure_next, guided_next, atol=1e-6, rtol=1e-6)


def test_official_x0_velocity_roundtrip():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    state = torch.randn(11, 6, generator=torch.Generator().manual_seed(41))
    velocity = torch.randn(11, 6, generator=torch.Generator().manual_seed(42))
    endpoint = sampler._pred_to_xstart(state, 0.625, velocity)
    recovered = sampler._xstart_to_pred(state, 0.625, endpoint)
    assert torch.allclose(recovered, velocity, atol=1e-6, rtol=1e-6)


def test_gradient_sign_is_consistency_descent_direction():
    matrix = torch.tensor([[1.5, -0.25], [0.4, 0.75], [-0.2, 1.1]])
    target = torch.tensor([0.3, -0.8, 0.5])
    z = torch.tensor([1.2, -0.7], requires_grad=True)
    loss = 0.5 * (matrix @ z - target).square().sum()
    grad_e = torch.autograd.grad(loss, z)[0]
    consistency_direction = -grad_e
    epsilon = 1e-3
    before = loss.detach()
    after = 0.5 * (matrix @ (z.detach() + epsilon * consistency_direction) - target).square().sum()
    assert torch.isfinite(consistency_direction).all()
    assert after < before
