from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.canonical_tile_sync import (
    C128NativeWindow,
    atom_overlap_count,
    atom_overlap_mass,
    build_c128_master_atom_space,
    build_common_atom_space,
    endpoint_indices_to_q,
    fuse_c128_master_clean_endpoints,
    fuse_clean_endpoints,
    run_c128_2048_coupled_endpoint_flow,
    shared_c128_master_local_noise,
    shared_spatial_noise,
    stateless_atom_noise,
)


def _coords(rows: list[tuple[int, int, int]]) -> torch.Tensor:
    return torch.tensor([[0, *row] for row in rows], dtype=torch.int32)


def _forward(q_local: torch.Tensor) -> torch.Tensor:
    q_local = q_local.to(torch.float64)
    z = q_local[:, 2]
    ax = 0.25 * (1.0 + 0.08 * z)
    ay = 0.25 * (1.0 - 0.05 * z)
    bx = 0.12 + 0.03 * z
    by = -0.08 + 0.02 * z
    return torch.stack(
        [
            ax * q_local[:, 0] + bx,
            ay * q_local[:, 1] + by,
            z,
        ],
        dim=1,
    )


def _inverse(q_global: torch.Tensor) -> torch.Tensor:
    q_global = q_global.to(torch.float64)
    z = q_global[:, 2]
    ax = 0.25 * (1.0 + 0.08 * z)
    ay = 0.25 * (1.0 - 0.05 * z)
    bx = 0.12 + 0.03 * z
    by = -0.08 + 0.02 * z
    return torch.stack(
        [
            (q_global[:, 0] - bx) / ax,
            (q_global[:, 1] - by) / ay,
            z,
        ],
        dim=1,
    )


def _space():
    # A small active global band surrounding the projective local cells.
    global_rows = [
        (x, y, z)
        for x in range(3, 6)
        for y in range(2, 5)
        for z in range(2, 6)
    ]
    local_rows = [
        (x, y, z)
        for x in range(2, 6)
        for y in range(2, 6)
        for z in range(2, 6)
    ]
    return build_common_atom_space(
        stage="shape512",
        global_coords=_coords(global_rows),
        global_resolution=8,
        local_coords=[_coords(local_rows)],
        local_resolution=8,
        local_to_global=[_forward],
        global_to_local=[_inverse],
        tile_ids=[17],
        target_resolution=32,
        chunk_size=16,
    )


def test_exact_z_dependent_transform_roundtrip() -> None:
    generator = torch.Generator().manual_seed(7)
    q_local = torch.rand((10000, 3), generator=generator) * 2.0 - 1.0
    q_roundtrip = _inverse(_forward(q_local))
    assert (_forward(q_roundtrip) - _forward(q_local)).abs().max() < 2e-12
    assert (q_roundtrip - q_local).abs().max() < 2e-12


def test_atom_space_is_sparse_and_preserves_cell_membership() -> None:
    space = _space()
    assert space.atom_count < 32**3
    assert space.atom_count > space.global_coords.shape[0]
    assert space.local_mappings[0].tile_id == 17
    assert space.local_mappings[0].token_atom_counts.min() > 0
    assert space.diagnostics["local"][0]["max_q_roundtrip_error"] < 2e-5

    shared_counts = atom_overlap_count(space, 0)
    assert shared_counts.shape[0] == space.local_mappings[0].coords.shape[0]
    assert (shared_counts > 0).any()


def test_field_operators_and_coverage_highpass_are_exact() -> None:
    space = _space()
    channels = 5
    generator = torch.Generator().manual_seed(12)
    global_endpoint = torch.randn(
        (space.global_coords.shape[0], channels), generator=generator
    )
    lifted = space.lift_global(global_endpoint)
    assert torch.allclose(
        space.restrict_global(lifted),
        global_endpoint,
        atol=1e-6,
        rtol=1e-6,
    )
    ones = torch.ones_like(global_endpoint)
    assert torch.equal(space.lift_global(ones), torch.ones_like(lifted))

    mapping = space.local_mappings[0]
    local_endpoint = torch.randn(
        (mapping.coords.shape[0], channels), generator=generator
    )
    # Partial coverage deliberately masks most rows and gives nonuniform image
    # confidence.  The high-pass invariant must still be exact.
    weights = torch.linspace(0.0, 1.0, mapping.coords.shape[0])
    visibility = (
        torch.arange(mapping.coords.shape[0]) % 3 != 0
    ).to(torch.float32)
    result = fuse_clean_endpoints(
        space,
        global_endpoint,
        [local_endpoint],
        local_token_weights=[weights],
        visibility=[visibility],
        huber_delta=0.5,
    )
    assert result.diagnostics["max_abs_Rg_high"] < 1e-5
    assert (
        result.diagnostics["max_abs_Rg_unified_minus_global"] < 1e-5
    )
    assert torch.allclose(
        space.restrict_global(result.unified_atoms),
        global_endpoint,
        atol=1e-5,
        rtol=1e-5,
    )


def test_shared_noise_marginals_and_overlap_covariance() -> None:
    space = _space()
    global_noise, local_noise, _ = shared_spatial_noise(
        space,
        8192,
        seed=91,
        namespace="noise/shape512",
        dtype=torch.float64,
    )
    assert abs(float(global_noise.mean().item())) < 0.03
    assert abs(float(local_noise[0].mean().item())) < 0.03
    assert abs(float(global_noise.std().item()) - 1.0) < 0.03
    assert abs(float(local_noise[0].std().item()) - 1.0) < 0.03

    mapping = space.local_mappings[0]
    shared = atom_overlap_mass(space, 0)
    local_row = int(torch.argmax(shared).item())
    local_atoms = mapping.atom_indices[
        mapping.token_indices == local_row
    ]
    global_atom_to_parent = torch.full(
        (space.atom_count,), -1, dtype=torch.int64
    )
    global_atom_to_parent[space.global_atom_indices] = space.global_parent_rows
    edge_mask = mapping.token_indices == local_row
    parents = global_atom_to_parent.index_select(
        0, mapping.atom_indices[edge_mask]
    )
    fractions = mapping.overlap_fractions[edge_mask]
    valid = parents >= 0
    parents = parents[valid]
    fractions = fractions[valid]
    parent = int(torch.mode(parents).values.item())
    overlap = float(fractions[parents == parent].sum().item())
    expected = overlap / math.sqrt(
        float(mapping.token_atom_mass[local_row].item())
        * int(space.global_token_atom_counts[parent].item())
    )
    observed = torch.mean(
        local_noise[0][local_row] * global_noise[parent]
    ).item()
    assert abs(observed - expected) < 0.04


def test_stateless_noise_does_not_depend_on_atom_order_or_subset() -> None:
    ids = torch.tensor([2, 7, 19, 31, 80], dtype=torch.int64)
    full = stateless_atom_noise(
        ids,
        11,
        seed=4,
        namespace="noise/texture1024",
        resolution=64,
        device=torch.device("cpu"),
    )
    permutation = torch.tensor([4, 1, 3], dtype=torch.int64)
    subset_ids = ids.index_select(0, permutation)
    subset = stateless_atom_noise(
        subset_ids,
        11,
        seed=4,
        namespace="noise/texture1024",
        resolution=64,
        device=torch.device("cpu"),
    )
    assert torch.equal(subset, full.index_select(0, permutation))
    other_stage = stateless_atom_noise(
        subset_ids,
        11,
        seed=4,
        namespace="noise/shape1024",
        resolution=64,
        device=torch.device("cpu"),
    )
    assert not torch.equal(subset, other_stage)


def test_endpoint_grid_is_not_cell_center_grid() -> None:
    endpoint = endpoint_indices_to_q(
        torch.tensor([[0, 63]], dtype=torch.int64), 64
    )
    assert torch.equal(endpoint, torch.tensor([[-1.0, 1.0]], dtype=torch.float64))


def _identity(q: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float64)


def _c128_master_space():
    master = _coords(
        [
            (x, y, z)
            for x in range(61, 68)
            for y in range(61, 68)
            for z in range(61, 68)
        ]
    )
    coarse_parent = (master[:, 1] >= 64).to(torch.int64)
    local = _coords(
        [
            (x, y, z)
            for x in range(31, 34)
            for y in range(31, 34)
            for z in range(31, 34)
        ]
    )
    return build_c128_master_atom_space(
        stage="shape1024_c128_master",
        master_coords=master,
        coarse_parent=coarse_parent,
        coarse_parent_count=2,
        local_coords=[local],
        local_to_global=[_identity],
        global_to_local=[_identity],
        tile_ids=[24],
        target_resolution=128,
        local_resolution=64,
    )


def test_c128_master_is_native_and_coarse_projector_highpass_is_exact() -> None:
    space = _c128_master_space()
    assert space.target_resolution == 128
    assert space.local_resolution == 64
    assert torch.equal(space.atom_coords[:, 1:4].amin(0), torch.tensor([61, 61, 61]))
    assert torch.equal(space.atom_coords[:, 1:4].amax(0), torch.tensor([67, 67, 67]))
    generator = torch.Generator().manual_seed(102)
    global_endpoint = torch.randn((space.atom_count, 7), generator=generator)
    local_endpoint = torch.randn(
        (space.local_mappings[0].coords.shape[0], 7), generator=generator
    )
    result = fuse_c128_master_clean_endpoints(
        space,
        global_endpoint,
        [local_endpoint],
        huber_delta=0.75,
        robust_iterations=3,
    )
    assert result.diagnostics["max_abs_Pcoarse_high"] < 1e-5
    assert result.diagnostics["max_abs_Pcoarse_unified_minus_global"] < 1e-5
    assert torch.allclose(
        space.project_coarse(result.unified_atoms),
        space.project_coarse(global_endpoint),
        atol=1e-5,
        rtol=1e-5,
    )


def test_c128_master_noise_is_direct_and_local_noise_uses_real_overlap() -> None:
    space = _c128_master_space()
    global_noise, local_noise, atom_noise = shared_c128_master_local_noise(
        space,
        4096,
        seed=51,
        namespace="noise/shape1024_c128_master",
        dtype=torch.float64,
    )
    assert torch.equal(global_noise, atom_noise)
    assert abs(float(global_noise.mean())) < 0.03
    assert abs(float(global_noise.std()) - 1.0) < 0.03
    assert abs(float(local_noise[0].mean())) < 0.04
    assert abs(float(local_noise[0].std()) - 1.0) < 0.04


class _OneStepSampler:
    def timestep_schedule(self, steps: int, rescale_t: float):
        assert steps == 1
        return (1.0, 0.0)

    def sample_once(self, model, state, t, t_next, **kwargs):
        endpoint = state.replace(torch.zeros_like(state.feats))
        return SimpleNamespace(pred_x_0=endpoint)

    def _xstart_to_pred(self, state, t: float, endpoint):
        return state - endpoint


def test_native_c128_window_runner_never_changes_support() -> None:
    space = _c128_master_space()
    channels = 3
    global_state = SparseTensor(
        feats=torch.randn(space.atom_count, channels),
        coords=space.atom_coords,
    )
    mapping = space.local_mappings[0]
    local_state = SparseTensor(
        feats=torch.randn(mapping.coords.shape[0], channels),
        coords=mapping.coords,
    )
    window_coords = space.atom_coords.clone()
    window_coords[:, 1:4] -= 61
    window = C128NativeWindow(
        window_index=0,
        token_indices=torch.arange(space.atom_count),
        local_coords=window_coords,
        weights=torch.ones(space.atom_count),
        start=(0, 0, 0),
        end=(64, 64, 64),
    )
    result = run_c128_2048_coupled_endpoint_flow(
        sampler=_OneStepSampler(),
        model=torch.nn.Identity(),
        atom_space=space,
        global_windows=[window],
        global_state=global_state,
        local_states=[local_state],
        global_condition={},
        local_conditions=[{}],
        steps=1,
        rescale_t=1.0,
    )
    assert torch.equal(result.global_samples.coords, space.atom_coords)
    assert torch.equal(result.local_samples[0].coords, mapping.coords)
    assert result.step_records[0]["max_abs_Pcoarse_high"] < 1e-5
    assert (
        result.final_fusion.diagnostics[
            "max_abs_Pcoarse_unified_minus_global"
        ]
        < 1e-5
    )


def test_main_2048_script_has_no_post_generation_resolution_reduction() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "pixal3d_canonical_tile_superresolution.py"
    ).read_text(encoding="utf-8")
    assert "_coarsen_latent_pair_for_decode" not in source
    assert "decode_resolution_ablation" not in source
    assert "coords256" not in source
    assert "GRID_UNIFIED = 128" in source
    assert "choices=(2048,)" in source
    assert "pipeline.decode_latent(\n        unified_shape_denorm" in source
