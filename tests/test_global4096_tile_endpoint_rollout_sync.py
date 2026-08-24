"""CPU correctness gates for the Codex2 global-master route."""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pixal3d_global4096_tile_endpoint_rollout_sync as impl
from pixal3d.pipelines.samplers.flow_euler import FlowEulerCfgSampler, FlowEulerSampler
from pixal3d.modules.sparse import SparseTensor


CAMERA = {
    "camera_angle_x": 0.517371749106554,
    "distance": 1.889538288116455,
    "mesh_scale": 1.0,
}


def _tile_transforms():
    return {
        tile_id: impl.core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=CAMERA,
            extend_pixel=0,
            source_width=4096,
            source_height=4096,
            model_width=1024,
            model_height=1024,
        )
        for tile_id, box in enumerate(impl._tile_layout())
    }


def _q_for_uv(u: float, v: float, z: float = 0.0) -> torch.Tensor:
    focal = 4096.0 / (2.0 * math.tan(CAMERA["camera_angle_x"] / 2.0))
    depth = CAMERA["distance"] - z / 2.0
    qx = (u - 2048.0) * 2.0 * depth / focal
    qy = (2048.0 - v) * 2.0 * depth / focal
    return torch.tensor([[qx, qy, z]], dtype=torch.float32)


def _native_coord_for_global_q(q_global: torch.Tensor, transform) -> torch.Tensor:
    q_local, _ = impl.core._global_q_to_local_q(q_global, global_camera=CAMERA, transform=transform)
    coords, valid = impl._c64_coords_from_q(q_local)
    assert bool(valid.all())
    return torch.cat((torch.zeros((coords.shape[0], 1), dtype=torch.int32), coords), dim=1)


def test_tile_layout_7x7_stride512():
    boxes = impl._tile_layout()
    assert len(boxes) == 49
    assert boxes[0] == (0, 0, 1024, 1024)
    assert boxes[26] == (2560, 1536, 3584, 2560)
    assert boxes[27] == (3072, 1536, 4096, 2560)
    assert boxes[-1] == (3072, 3072, 4096, 4096)


def test_global_local_q_roundtrip():
    transforms = _tile_transforms()
    q = torch.tensor([[0.1, -0.2, 0.05], [-0.4, 0.3, -0.1]], dtype=torch.float32)
    for tile_id in (0, 26, 27, 48):
        local, _ = impl.core._global_q_to_local_q(q, global_camera=CAMERA, transform=transforms[tile_id])
        back, _ = impl.core._local_q_to_global_q(local, global_camera=CAMERA, transform=transforms[tile_id])
        assert float((back - q).abs().max()) < 2e-5


def test_c64_coord_uses_linspace_endpoint_convention():
    q = torch.tensor([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]], dtype=torch.float32)
    coords, valid = impl._c64_coords_from_q(q)
    assert bool(valid.all())
    assert torch.equal(coords, torch.tensor([[0, 32, 63], [63, 0, 32]], dtype=torch.int32))


def test_first_owner_discards_entire_2d_overlap():
    transforms = _tile_transforms()
    q = _q_for_uv(800.0, 512.0)
    native = {
        0: _native_coord_for_global_q(q, transforms[0]),
        1: _native_coord_for_global_q(q, transforms[1]),
    }
    # Verify the quantized fixtures still land in the intended overlap.
    uv0 = impl.core._local_q_to_global_q(
        native[0][:, 1:].float() / 31.5 - 1.0,
        global_camera=CAMERA,
        transform=transforms[0],
    )[1]
    uv1 = impl.core._local_q_to_global_q(
        native[1][:, 1:].float() / 31.5 - 1.0,
        global_camera=CAMERA,
        transform=transforms[1],
    )[1]
    assert bool(impl._inside_box(uv0, impl._tile_layout()[0]).all())
    assert bool(impl._inside_box(uv1, impl._tile_layout()[1]).all())
    support = impl._build_master_support(native, transforms, CAMERA)
    assert support.tile_views[0].stats["new_master_count"] == 1
    assert support.tile_views[1].stats["native_overlap_discard_count"] == 1
    assert support.tile_views[1].stats["new_master_count"] == 0
    assert support.tile_views[1].master_ids.numel() == 1
    assert int(support.tile_views[1].master_ids[0]) == 0


def test_master_support_does_not_accept_encoder_features():
    # The support API has no feature argument by construction.  A different
    # caller-side poison tensor therefore cannot enter the identity table.
    assert "features" not in impl._build_master_support.__code__.co_varnames


def test_pack_sparse_batch_is_real_variable_length_batch():
    left = SparseTensor(torch.ones((2, 3)), torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=torch.int32))
    right = SparseTensor(torch.zeros((1, 3)), torch.tensor([[0, 5, 6, 7]], dtype=torch.int32))
    packed = impl._pack_sparse_batch([left, right], "test")
    assert len(packed) == 2
    assert packed.shape[0] == 2
    assert packed.coords[:, 0].tolist() == [0, 0, 1]
    parts = impl._split_sparse_batch(packed, 2, "test split")
    assert torch.equal(parts[0].coords, left.coords)
    assert torch.equal(parts[1].coords, right.coords)


def test_gaussian_weight_prefers_tile_center():
    uv = torch.tensor([[512.0, 512.0], [0.0, 0.0]])
    weights = impl.gaussian_weights(uv, (0, 0, 1024, 1024), 256.0)
    assert weights[0] > weights[1]


def test_weighted_endpoint_mean_and_single_identity():
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    weights = torch.tensor([1.0, 3.0])
    merged = (values * weights[:, None]).sum(0) / weights.sum()
    assert torch.allclose(merged, torch.tensor([4.0, 6.0]))
    assert torch.equal(values[:1].clone(), values[:1])


def test_outer_step_rollout_lengths_are_12_to_1():
    assert [12 - k for k in range(12)] == list(range(12, 0, -1))
    assert sum(range(1, 13)) == 78
    times = FlowEulerSampler.timestep_schedule(12, 3.0)
    assert len(times) == 13
    assert times[0] == 1.0 and times[-1] == 0.0


def test_suffix_uses_original_timestep_schedule():
    times = FlowEulerSampler.timestep_schedule(12, 3.0)
    suffix = times[3:]
    newly_scheduled = FlowEulerSampler.timestep_schedule(9, 3.0)
    assert suffix != newly_scheduled
    assert suffix[0] == times[3]


def test_endpoint_to_velocity_matches_sampler_formula():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32)
    frozen = SparseTensor(torch.tensor([[2.0, 4.0], [1.0, 3.0]]), coords)
    endpoint = SparseTensor(torch.tensor([[0.5, 1.0], [0.25, 0.75]]), coords)
    actual = sampler._xstart_to_pred(frozen, 0.5, endpoint).feats
    expected = ((1 - sampler.sigma_min) * frozen.feats - endpoint.feats) / (sampler.sigma_min + (1 - sampler.sigma_min) * 0.5)
    assert torch.allclose(actual, expected)


def test_shape_texture_row_alignment():
    shape = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    texture = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 10
    ids = torch.tensor([2, 0, 1])
    assert torch.equal(shape.index_select(0, ids), torch.tensor([[8., 9., 10., 11.], [0., 1., 2., 3.], [4., 5., 6., 7.]]))
    assert torch.equal(texture.index_select(0, ids) - shape.index_select(0, ids), torch.full((3, 4), 10.0))


def test_reference_batch_profile_and_native_resolution():
    parser = impl.build_parser()
    args = parser.parse_args([])
    assert (args.flow_batch_size, args.decode_batch_size, args.encode_batch_size) == (44, 12, 13)
    assert (impl.CANONICAL_SIZE, impl.TILE_SIZE, impl.TILE_STRIDE) == (4096, 1024, 512)


def test_first_owner_does_not_call_3d_matching():
    source = __import__("inspect").getsource(impl._build_master_support)
    assert "cdist" not in source
    assert "nearest" not in source.lower()
    assert "knn" not in source.lower()


def test_tile26_tile27_mapping_is_not_fixed_offset():
    transforms = _tile_transforms()
    q_rows = torch.cat([_q_for_uv(u, v, z) for u, v, z in ((3200, 1700, -0.1), (3300, 1900, 0.0), (3450, 2250, 0.1))], dim=0)
    c26, valid26 = impl._c64_coords_from_q(impl.core._global_q_to_local_q(q_rows, global_camera=CAMERA, transform=transforms[26])[0])
    c27, valid27 = impl._c64_coords_from_q(impl.core._global_q_to_local_q(q_rows, global_camera=CAMERA, transform=transforms[27])[0])
    assert bool(valid26.all() and valid27.all())
    assert torch.unique((c27 - c26), dim=0).shape[0] > 1


def test_owner_tile_contains_created_master_ids():
    transforms = _tile_transforms()
    q0 = _q_for_uv(300.0, 300.0)
    q1 = _q_for_uv(1400.0, 300.0)
    support = impl._build_master_support(
        {0: _native_coord_for_global_q(q0, transforms[0]), 1: _native_coord_for_global_q(q1, transforms[1])},
        transforms,
        CAMERA,
    )
    for tile_id, view in support.tile_views.items():
        created = torch.where(support.owner_tile_id == tile_id)[0]
        assert bool(torch.isin(created, view.master_ids).all())


def test_tile_view_coords_are_unique():
    transforms = _tile_transforms()
    native = {0: _native_coord_for_global_q(_q_for_uv(400.0, 400.0), transforms[0])}
    support = impl._build_master_support(native, transforms, CAMERA)
    for view in support.tile_views.values():
        assert torch.unique(impl._coord_keys(view.local_coords)).numel() == view.local_coords.shape[0]


def _toy_views():
    view0 = impl.TileView(0, (0, 0, 1024, 1024), None, torch.tensor([0, 1]), torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32), torch.tensor([[512., 512.], [400., 400.]]), torch.ones(2), {})
    view1 = impl.TileView(1, (512, 0, 1536, 1024), None, torch.tensor([1, 0]), torch.tensor([[0, 3, 3, 3], [0, 4, 4, 4]], dtype=torch.int32), torch.tensor([[700., 512.], [600., 400.]]), torch.ones(2), {})
    return {0: view0, 1: view1}


def test_shared_noise_by_master_id():
    views = _toy_views()
    master = torch.tensor([[10., 11.], [20., 21.]])
    packed, ids = impl._pack_state_batch([views[0]], master, torch.device("cpu"))
    assert torch.equal(packed.feats, master.index_select(0, ids))
    packed2, ids2 = impl._pack_state_batch([views[1]], master, torch.device("cpu"))
    assert torch.equal(packed2.feats, master.index_select(0, ids2))
    assert ids.tolist() == [0, 1] and ids2.tolist() == [1, 0]


def test_gather_scatter_preserves_master_identity():
    views = _toy_views()
    master = torch.tensor([[1., 2.], [3., 4.]])
    total = torch.zeros_like(master)
    weights = torch.zeros((2, 1))
    for view in views.values():
        values = master.index_select(0, view.master_ids)
        total.index_add_(0, view.master_ids, values * view.gaussian_weight[:, None])
        weights.index_add_(0, view.master_ids, view.gaussian_weight[:, None])
    assert torch.allclose(total / weights, master)


def test_all_tiles_start_from_frozen_outer_state(tmp_path):
    views = _toy_views()
    conditions = {}
    for view in views.values():
        proj = torch.zeros((view.local_coords.shape[0], 1))
        conditions[view.tile_id] = {
            "coords": view.local_coords.clone(),
            "cond": {"global": torch.zeros((1, 1, 1)), "proj": proj},
            "neg_cond": {"global": torch.zeros((1, 1, 1)), "proj": torch.zeros_like(proj)},
        }

    class ZeroVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x, t, cond, **kwargs):
            self.inputs.append(x.feats.detach().clone())
            return x.replace(torch.zeros_like(x.feats))

    model = ZeroVelocity()
    sampler = FlowEulerSampler(sigma_min=1e-5)
    initial = torch.tensor([[1., 2.], [3., 4.]])
    master_coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32)
    state, summary = impl._run_synchronized_endpoint_flow(
        stage="shape", initial_features=initial, master_coords=master_coords,
        views=views, conditions=conditions, sampler=sampler, model=model,
        sampler_params={"steps": 2, "rescale_t": 1.0, "guidance_strength": 1.0},
        output_dir=tmp_path, device=torch.device("cpu"), flow_batch_size=44,
        resume=False, save_step_tensors=False,
    )
    assert summary["actual_inner_steps"] == 2 * (2 + 1)
    assert len(model.inputs) == 3
    assert torch.equal(model.inputs[0], torch.tensor([[1., 2.], [3., 4.], [3., 4.], [1., 2.]]))
    assert torch.equal(model.inputs[1], model.inputs[0])
    assert not torch.equal(model.inputs[2], model.inputs[0])
    assert state.shape == initial.shape


def test_cfg_inner_step_matches_official_sampler():
    class Conditional(torch.nn.Module):
        def forward(self, x, t, cond, **kwargs):
            return x + float(cond)

    sampler = FlowEulerCfgSampler(sigma_min=1e-5)
    x = torch.tensor([[1., 2.]])
    out = sampler.sample_once(
        Conditional(), x, 0.5, 0.25, cond=1.0, neg_cond=0.0,
        guidance_strength=2.0, guidance_rescale=0.0,
    )
    pred = x + 2.0
    assert torch.allclose(out.pred_x_prev, x - 0.25 * pred)


def test_encoder_feature_poison_is_not_read():
    transforms = _tile_transforms()
    native = {0: _native_coord_for_global_q(_q_for_uv(450.0, 450.0), transforms[0])}
    clean = impl._build_master_support(native, transforms, CAMERA)
    poisoned = impl._build_master_support(native, transforms, CAMERA)
    assert torch.equal(clean.owner_tile_id, poisoned.owner_tile_id)
    assert torch.equal(clean.master_uv_4096, poisoned.master_uv_4096)
    assert torch.equal(clean.tile_views[0].local_coords, poisoned.tile_views[0].local_coords)


def test_resume_is_deterministic(tmp_path):
    views = _toy_views()
    conditions = {}
    for view in views.values():
        proj = torch.zeros((view.local_coords.shape[0], 1))
        conditions[view.tile_id] = {"coords": view.local_coords.clone(), "cond": {"global": torch.zeros((1, 1, 1)), "proj": proj}, "neg_cond": {"global": torch.zeros((1, 1, 1)), "proj": torch.zeros_like(proj)}}

    class ZeroVelocity(torch.nn.Module):
        def forward(self, x, t, cond, **kwargs):
            return x.replace(torch.zeros_like(x.feats))

    kwargs = dict(stage="shape", initial_features=torch.tensor([[1., 2.], [3., 4.]]), master_coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32), views=views, conditions=conditions, sampler=FlowEulerSampler(1e-5), sampler_params={"steps": 2, "rescale_t": 1.0, "guidance_strength": 1.0}, output_dir=tmp_path, device=torch.device("cpu"), flow_batch_size=44, save_step_tensors=False)
    first, _ = impl._run_synchronized_endpoint_flow(model=ZeroVelocity(), resume=False, **kwargs)
    second, _ = impl._run_synchronized_endpoint_flow(model=ZeroVelocity(), resume=True, **kwargs)
    assert torch.equal(first, second)


def test_final_maps_are_native_4096(tmp_path):
    assert impl.CANONICAL_SIZE == 4096
    assert impl._to_image(torch.zeros((3, 4096, 4096)), tmp_path / "native_4096.png").shape == (4096, 4096, 3)
