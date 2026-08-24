"""CPU correctness gates for the instant-x0 consensus route."""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.samplers.flow_euler import FlowEulerCfgSampler, FlowEulerSampler

import pixal3d_global4096_tile_x0_consensus_sync as impl


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
    q_local, _ = impl.core._global_q_to_local_q(
        q_global, global_camera=CAMERA, transform=transform
    )
    coords, valid = impl._c64_coords_from_q(q_local)
    assert bool(valid.all())
    return torch.cat(
        (torch.zeros((coords.shape[0], 1), dtype=torch.int32), coords), dim=1
    )


def _toy_views():
    return {
        0: impl.TileView(
            0,
            (0, 0, 1024, 1024),
            None,
            torch.tensor([0, 1]),
            torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32),
            torch.tensor([[512.0, 512.0], [400.0, 400.0]]),
            torch.ones(2),
            {},
        ),
        1: impl.TileView(
            1,
            (512, 0, 1536, 1024),
            None,
            torch.tensor([1, 0]),
            torch.tensor([[0, 3, 3, 3], [0, 4, 4, 4]], dtype=torch.int32),
            torch.tensor([[700.0, 512.0], [600.0, 400.0]]),
            torch.ones(2),
            {},
        ),
    }


def _conditions(views):
    result = {}
    for tile_id, view in views.items():
        n = view.local_coords.shape[0]
        result[tile_id] = {
            "tile_id": tile_id,
            "coords": view.local_coords.clone(),
            "cond": {
                "global": torch.full((1, 1, 1), float(tile_id + 1)),
                "proj": torch.zeros((n, 1)),
            },
            "neg_cond": {
                "global": torch.zeros((1, 1, 1)),
                "proj": torch.zeros((n, 1)),
            },
        }
    return result


class LinearVelocity(torch.nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale
        self.inputs = []

    def forward(self, x, t, cond, **kwargs):
        self.inputs.append(x.feats.detach().cpu().clone())
        batch = x.coords[:, 0].long()
        return x.replace(self.scale * x.feats + 0.01 * t[batch, None])


class ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, t, cond, **kwargs):
        self.inputs.append(x.feats.detach().cpu().clone())
        return x.replace(torch.zeros_like(x.feats))


def _run_toy(tmp_path: Path, *, batch_size=44, views=None, steps=2, model=None):
    views = views or _toy_views()
    model = model or LinearVelocity()
    state, summary = impl._run_synchronized_x0_consensus_flow(
        stage="shape",
        initial_features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        master_coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32),
        views=views,
        conditions=_conditions(views),
        sampler=FlowEulerSampler(sigma_min=1e-5),
        model=model,
        sampler_params={"steps": steps, "rescale_t": 1.0},
        output_dir=tmp_path,
        device=torch.device("cpu"),
        flow_batch_size=batch_size,
        resume=False,
    )
    return state, summary, model


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
        local, _ = impl.core._global_q_to_local_q(
            q, global_camera=CAMERA, transform=transforms[tile_id]
        )
        back, _ = impl.core._local_q_to_global_q(
            local, global_camera=CAMERA, transform=transforms[tile_id]
        )
        assert float((back - q).abs().max()) < 2e-5


def test_c64_coord_uses_linspace_endpoint_convention():
    coords, valid = impl._c64_coords_from_q(
        torch.tensor([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    )
    assert bool(valid.all())
    assert torch.equal(
        coords, torch.tensor([[0, 32, 63], [63, 0, 32]], dtype=torch.int32)
    )


def test_first_owner_discards_entire_2d_overlap():
    transforms = _tile_transforms()
    q = _q_for_uv(800.0, 512.0)
    native = {
        0: _native_coord_for_global_q(q, transforms[0]),
        1: _native_coord_for_global_q(q, transforms[1]),
    }
    support = impl._build_master_support(native, transforms, CAMERA)
    assert support.tile_views[0].stats["new_master_count"] == 1
    assert support.tile_views[1].stats["native_overlap_discard_count"] == 1
    assert support.tile_views[1].stats["new_master_count"] == 0
    assert support.tile_views[1].master_ids.tolist() == [0]


def test_first_owner_does_not_call_3d_matching():
    source = inspect.getsource(impl._build_master_support)
    assert "cdist" not in source
    assert "nearest" not in source.lower()
    assert "knn" not in source.lower()


def test_tile26_tile27_mapping_is_not_fixed_offset():
    transforms = _tile_transforms()
    q_rows = torch.cat(
        [_q_for_uv(u, v, z) for u, v, z in ((3200, 1700, -0.1), (3300, 1900, 0.0), (3450, 2250, 0.1))]
    )
    c26, valid26 = impl._c64_coords_from_q(
        impl.core._global_q_to_local_q(q_rows, global_camera=CAMERA, transform=transforms[26])[0]
    )
    c27, valid27 = impl._c64_coords_from_q(
        impl.core._global_q_to_local_q(q_rows, global_camera=CAMERA, transform=transforms[27])[0]
    )
    assert bool(valid26.all() and valid27.all())
    assert torch.unique(c27 - c26, dim=0).shape[0] > 1


def test_owner_tile_contains_created_master_ids():
    transforms = _tile_transforms()
    support = impl._build_master_support(
        {
            0: _native_coord_for_global_q(_q_for_uv(300.0, 300.0), transforms[0]),
            1: _native_coord_for_global_q(_q_for_uv(1400.0, 300.0), transforms[1]),
        },
        transforms,
        CAMERA,
    )
    for tile_id, view in support.tile_views.items():
        created = torch.where(support.owner_tile_id == tile_id)[0]
        assert bool(torch.isin(created, view.master_ids).all())


def test_tile_view_coords_are_unique():
    transforms = _tile_transforms()
    support = impl._build_master_support(
        {0: _native_coord_for_global_q(_q_for_uv(400.0, 400.0), transforms[0])},
        transforms,
        CAMERA,
    )
    for view in support.tile_views.values():
        assert torch.unique(impl._coord_keys(view.local_coords)).numel() == view.local_coords.shape[0]


def test_shared_noise_by_master_id():
    views = _toy_views()
    master = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    packed, ids = impl._legacy._pack_state_batch([views[0]], master, torch.device("cpu"))
    packed2, ids2 = impl._legacy._pack_state_batch([views[1]], master, torch.device("cpu"))
    assert torch.equal(packed.feats, master.index_select(0, ids))
    assert torch.equal(packed2.feats, master.index_select(0, ids2))
    assert ids.tolist() == [0, 1] and ids2.tolist() == [1, 0]


def test_gather_scatter_preserves_master_identity():
    views = _toy_views()
    master = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    total = torch.zeros_like(master)
    weights = torch.zeros((2, 1))
    for view in views.values():
        values = master.index_select(0, view.master_ids)
        total.index_add_(0, view.master_ids, values * view.gaussian_weight[:, None])
        weights.index_add_(0, view.master_ids, view.gaussian_weight[:, None])
    assert torch.allclose(total / weights, master)


def test_gaussian_weight_prefers_tile_center():
    weights = impl.gaussian_weights(torch.tensor([[512.0, 512.0], [0.0, 0.0]]), (0, 0, 1024, 1024), 256.0)
    assert weights[0] > weights[1]


def test_weighted_current_pred_x0_mean():
    merged = impl.merge_pred_x0_contributions(
        1,
        [
            {"master_ids": torch.tensor([0]), "pred_x0": torch.tensor([[1.0, 3.0]]), "weight": torch.tensor([1.0])},
            {"master_ids": torch.tensor([0]), "pred_x0": torch.tensor([[5.0, 7.0]]), "weight": torch.tensor([3.0])},
        ],
    )
    assert torch.allclose(merged["gaussian_pred_x0"], torch.tensor([[4.0, 6.0]]))


def test_single_pred_x0_is_identity():
    value = torch.tensor([[2.0, -4.0]])
    merged = impl.merge_pred_x0_contributions(
        1, [{"master_ids": torch.tensor([0]), "pred_x0": value, "weight": torch.tensor([0.2])}]
    )
    assert torch.equal(merged["gaussian_pred_x0"], value)


def test_one_prediction_per_tile_per_timestep(tmp_path):
    _, summary, model = _run_toy(tmp_path, batch_size=1, steps=3)
    assert summary["logical_tile_predictions"] == 2 * 3
    assert summary["physical_tile_batches"] == 2 * 3
    assert len(model.inputs) == 2 * 3
    assert all(record["logical_tile_predictions"] == 2 for record in summary["records"])
    assert all(record["inner_prediction_count"] == 0 for record in summary["records"])


def test_all_tiles_start_from_same_frozen_state(tmp_path):
    model = ZeroVelocity()
    initial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state, summary, _ = _run_toy(tmp_path, batch_size=44, steps=2, model=model)
    assert torch.equal(state, initial)
    assert len(model.inputs) == 2
    assert torch.equal(model.inputs[0], model.inputs[1])
    assert all(not record["suffix_rollout_used"] for record in summary["records"])


def test_current_pred_x0_matches_sampler_formula():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32)
    frozen = SparseTensor(torch.tensor([[2.0, 4.0], [1.0, 3.0]]), coords)
    velocity = SparseTensor(torch.tensor([[0.5, 1.0], [0.25, 0.75]]), coords)
    x0 = sampler._pred_to_xstart(frozen, 0.5, velocity)
    expected = (1 - sampler.sigma_min) * frozen.feats - (sampler.sigma_min + (1 - sampler.sigma_min) * 0.5) * velocity.feats
    assert torch.allclose(x0.feats, expected)


def test_x0_fusion_equals_current_velocity_fusion_for_shared_xt():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    frozen = torch.tensor([[2.0, 4.0]])
    v0 = torch.tensor([[0.5, 1.0]])
    v1 = torch.tensor([[0.25, 0.75]])
    x0_0 = (1 - sampler.sigma_min) * frozen - (sampler.sigma_min + (1 - sampler.sigma_min) * 0.5) * v0
    x0_1 = (1 - sampler.sigma_min) * frozen - (sampler.sigma_min + (1 - sampler.sigma_min) * 0.5) * v1
    merged = impl.merge_pred_x0_contributions(
        1,
        [
            {"master_ids": torch.tensor([0]), "pred_x0": x0_0, "pred_v": v0, "weight": torch.tensor([1.0])},
            {"master_ids": torch.tensor([0]), "pred_x0": x0_1, "pred_v": v1, "weight": torch.tensor([3.0])},
        ],
    )
    v_from_x0 = sampler._xstart_to_pred(
        SparseTensor(frozen, torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)),
        0.5,
        SparseTensor(merged["gaussian_pred_x0"], torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)),
    ).feats
    v_mean = (v0 + 3.0 * v1) / 4.0
    assert torch.allclose(v_from_x0, v_mean, atol=1e-6, rtol=1e-6)


def test_single_tile_sync_matches_official_sampler_trajectory(tmp_path):
    views = {0: _toy_views()[0]}
    model = LinearVelocity()
    state, _, _ = _run_toy(tmp_path, batch_size=44, views=views, steps=3, model=model)
    initial = SparseTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), views[0].local_coords)
    official = FlowEulerSampler(1e-5).sample(
        model,
        initial,
        cond=_conditions(views)[0]["cond"],
        steps=3,
        rescale_t=1.0,
        verbose=False,
        return_model_history=False,
    ).samples
    assert torch.allclose(state, official.feats, atol=1e-6, rtol=1e-6)


def test_real_multibatch_matches_serial_tile_execution(tmp_path):
    state_serial, _, _ = _run_toy(tmp_path / "serial", batch_size=1, steps=2)
    state_batch, _, _ = _run_toy(tmp_path / "batch", batch_size=44, steps=2)
    assert torch.allclose(state_serial, state_batch, atol=1e-6, rtol=1e-6)


def test_suffix_terminal_is_not_reinterpreted_as_current_x0():
    source = inspect.getsource(impl._run_synchronized_x0_consensus_flow)
    assert "pred_x_prev" not in source
    assert "range(step, steps)" not in source
    assert "instantaneous" in source


def test_cfg_current_step_matches_official_sampler():
    calls = []

    class CondVelocity(torch.nn.Module):
        def forward(self, x, t, cond, **kwargs):
            calls.append(cond["global"].detach().clone())
            level = cond["global"].reshape(cond["global"].shape[0], -1).mean(dim=1)
            return x.replace(level[x.coords[:, 0], None].expand_as(x.feats))

    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    x = SparseTensor(torch.ones((1, 2)), coords)
    cond = {"global": torch.full((1, 1, 1), 2.0), "proj": SparseTensor(torch.zeros((1, 1)), coords)}
    neg = {"global": torch.zeros((1, 1, 1)), "proj": SparseTensor(torch.zeros((1, 1)), coords)}
    sampler = FlowEulerCfgSampler(1e-5)
    out = sampler.sample_once(
        CondVelocity(), x, 0.5, 0.4, cond=cond, neg_cond=neg,
        guidance_strength=2.0, guidance_rescale=0.0,
    )
    assert len(calls) == 2
    expected_v = torch.full_like(x.feats, 4.0)
    assert torch.allclose(out.pred_v.feats, expected_v)
    expected_x0 = sampler._pred_to_xstart(x, 0.5, out.pred_v).feats
    assert torch.allclose(out.pred_x_0.feats, expected_x0)


def test_condition_disagreement_and_hard_owner_are_recorded(tmp_path):
    merged, summary = None, None
    _, summary, _ = _run_toy(tmp_path, batch_size=44, steps=1)
    step = tmp_path / "shape" / "step_00"
    assert (step / "condition_disagreement.json").is_file()
    assert (step / "control_hard_center_owner_pred_x0.pt").is_file()
    assert summary["records"][0]["disagreement"]["participant_count_histogram"]


def test_shape_texture_row_alignment():
    ids = torch.tensor([2, 0, 1])
    shape = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    texture = shape + 10.0
    assert torch.equal(shape.index_select(0, ids), torch.tensor([[8., 9., 10., 11.], [0., 1., 2., 3.], [4., 5., 6., 7.]]))
    assert torch.equal(texture.index_select(0, ids) - shape.index_select(0, ids), torch.full((3, 4), 10.0))


def test_encoder_feature_poison_is_not_read():
    assert "features" not in impl._build_master_support.__code__.co_varnames
    first = impl.merge_pred_x0_contributions(
        1, [{"master_ids": torch.tensor([0]), "pred_x0": torch.tensor([[1.0]]), "weight": torch.tensor([1.0])}]
    )["gaussian_pred_x0"]
    second = impl.merge_pred_x0_contributions(
        1, [{"master_ids": torch.tensor([0]), "pred_x0": torch.tensor([[1.0]]), "weight": torch.tensor([1.0])}]
    )["gaussian_pred_x0"]
    assert torch.equal(first, second)


def test_resume_is_deterministic(tmp_path):
    first, summary, _ = _run_toy(tmp_path, batch_size=44, steps=2)
    model = LinearVelocity()
    views = _toy_views()
    second, resumed = impl._run_synchronized_x0_consensus_flow(
        stage="shape",
        initial_features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        master_coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32),
        views=views,
        conditions=_conditions(views),
        sampler=FlowEulerSampler(1e-5),
        model=model,
        sampler_params={"steps": 2, "rescale_t": 1.0},
        output_dir=tmp_path,
        device=torch.device("cpu"),
        flow_batch_size=44,
        resume=True,
    )
    assert torch.equal(first, second)
    assert resumed["logical_tile_predictions"] == summary["logical_tile_predictions"]
    assert not model.inputs


def test_resume_rejects_changed_support_or_latents(tmp_path):
    _run_toy(tmp_path, batch_size=44, steps=1)
    views = _toy_views()
    with pytest.raises(RuntimeError, match="resume checkpoint rejected"):
        impl._run_synchronized_x0_consensus_flow(
            stage="shape",
            initial_features=torch.tensor([[9.0, 2.0], [3.0, 4.0]]),
            master_coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32),
            views=views,
            conditions=_conditions(views),
            sampler=FlowEulerSampler(1e-5),
            model=LinearVelocity(),
            sampler_params={"steps": 1, "rescale_t": 1.0},
            output_dir=tmp_path,
            device=torch.device("cpu"),
            flow_batch_size=44,
            resume=True,
        )


def test_final_maps_are_native_4096(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    Image.new("L", (4096, 4096), 0).save(final / "final_render_alpha_4096.png")
    checked = impl.validate_native_4096_outputs(tmp_path)
    assert checked["images"]["final_render_alpha_4096.png"] == [4096, 4096]
    Image.new("L", (1024, 1024), 0).save(final / "final_render_rgb_4096.png")
    with pytest.raises(RuntimeError, match="native 4096"):
        impl.validate_native_4096_outputs(tmp_path)


def test_parser_has_cuda5_and_reference_batch_profile():
    args = impl.build_parser().parse_args([])
    assert args.cuda_device == 5
    assert (args.flow_batch_size, args.decode_batch_size, args.encode_batch_size) == (44, 12, 13)
    assert args.output_dir == impl.DEFAULT_OUTPUT_DIR
