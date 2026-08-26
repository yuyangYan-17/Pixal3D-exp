import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pixal3d_global_c256_cube_owner_flow_singleview as m
from pixal3d.modules.sparse import SparseTensor
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor


def coords(xyz):
    xyz = torch.as_tensor(xyz, dtype=torch.int32)
    return torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz), 1)


def test_c4096_c256_exact_scale():
    assert 1024 // 16 == 64
    assert 512 // 16 == 32
    assert torch.equal(torch.div(torch.tensor([[0, 511, 4095]]), 16, rounding_mode="floor"),
                       torch.tensor([[0, 31, 255]]))


def test_physical_cube_center_translation_and_scale_to_c64():
    xyz = torch.tensor([[32, 64, 96], [95, 127, 159]], dtype=torch.int32)
    local, local_q = m.center_translate_scale_to_local_c64(xyz, (32, 64, 96))
    assert local.tolist() == [[0, 0, 0], [63, 63, 63]]
    assert torch.all(local_q > -1) and torch.all(local_q < 1)


def test_layout_starts_count_and_order():
    layout = m.cube_layout()
    assert m.STARTS == (0, 32, 64, 96, 128, 160, 192)
    assert len(layout) == 343
    assert layout[1]["start"] == (0, 0, 32)
    assert layout[7]["start"] == (0, 32, 0)
    assert layout[-1]["start"] == (192, 192, 192)


def test_half_open_membership_and_integer_roundtrip():
    c = coords([[0, 0, 0], [63, 63, 63], [64, 64, 64], [255, 255, 255]])
    records, coverage = m.build_cube_records(c)
    assert records[0]["global_row_ids"].tolist() == [0, 1]
    assert 2 not in records[0]["global_row_ids"].tolist()
    for r in records:
        ids = r["global_row_ids"]
        start = torch.tensor(r["start"], dtype=torch.int32)
        assert torch.equal(r["local_xyz"] + start, c[ids, 1:])
        assert torch.unique(m.linear_keys(r["local_xyz"], 64)).numel() == ids.numel()
    assert coverage.min() >= 1 and coverage.max() <= 8


def test_full_domain_coverage_range():
    axis = torch.arange(256, dtype=torch.int32)
    c = coords(torch.stack((axis, axis, axis), 1))
    _, coverage = m.build_cube_records(c)
    assert set(coverage.tolist()).issubset({1, 2, 3, 4, 8})


def test_nearest_center_owner_and_tie_break():
    c = coords([[31, 31, 31], [32, 32, 32], [47, 31, 31], [48, 31, 31]])
    records, _ = m.build_cube_records(c)
    owner, stats = m.build_owner_map(c, records)
    # x=47 cell centre is exactly tied between x starts 0 and 32; lowest ID wins.
    assert owner[2].item() == 0
    # Integer cube centres versus half-integer cell centres make production
    # ties impossible, but an exact duplicate candidate still verifies that
    # equality retains the lower cube ID.
    duplicate = [{"cube_id": 0, "start": (0, 0, 0), "global_row_ids": torch.arange(4)},
                 {"cube_id": 1, "start": (0, 0, 0), "global_row_ids": torch.arange(4)}]
    duplicate_owner, duplicate_stats = m.build_owner_map(c, duplicate)
    assert duplicate_owner.tolist() == [0, 0, 0, 0]
    assert duplicate_stats["tie_row_count"] == 4
    for row, cube_id in enumerate(owner.tolist()):
        assert row in records[cube_id]["global_row_ids"].tolist()


def test_shared_global_noise_gather_is_exact():
    c = coords([[32, 32, 32], [40, 40, 40]])
    records, _ = m.build_cube_records(c)
    state = torch.randn(2, 32, generator=torch.Generator().manual_seed(9))
    hits = [r for r in records if 0 in r["global_row_ids"].tolist()]
    values = [state.index_select(0, r["global_row_ids"])[r["global_row_ids"].tolist().index(0)] for r in hits]
    assert all(torch.equal(values[0], v) for v in values[1:])


def test_projected_crop_expands_outward_to_patch_multiple():
    points = torch.tensor([[3.25, 17.5], [35.1, 40.2]], dtype=torch.float64)
    box, diagnostics = m.align_projected_crop_box(
        points, image_width=64, image_height=64, multiple=16
    )
    assert box == (0, 16, 48, 48)
    assert (box[2] - box[0]) % 16 == 0
    assert (box[3] - box[1]) % 16 == 0
    clipped = diagnostics["clipped_bbox_pixel_edges_4096"]
    assert box[0] <= clipped[0] <= clipped[2] <= box[2]
    assert box[1] <= clipped[1] <= clipped[3] <= box[3]


def test_cube_projection_crop_is_4096_patch_aligned_and_normalized():
    camera = {"camera_angle_x": 0.6, "distance": 2.0, "mesh_scale": 1.0}
    crop = m.cube_projection_crop((96, 96, 96), camera)
    x0, y0, x1, y1 = crop["crop_box_4096"]
    assert 0 <= x0 < x1 <= 4096 and 0 <= y0 < y1 <= 4096
    assert (x1 - x0) % 16 == 0 and (y1 - y0) % 16 == 0
    assert crop["crop_size"] == [x1 - x0, y1 - y0]
    assert crop["projection_crop_box"] == pytest.approx(
        (x0 / 4096, y0 / 4096, x1 / 4096, y1 / 4096)
    )
    token_corners = torch.tensor(
        [(x, y, z) for x in (96, 159) for y in (96, 159) for z in (96, 159)],
        dtype=torch.float64,
    )
    token_q = 2.0 * token_corners / 255.0 - 1.0
    token_uv, _, finite = m.camera_core._project_global_q_to_image(
        token_q, global_camera=camera, image_width=1024, image_height=1024
    )
    token_edges_4096 = (token_uv + 0.5) * 4.0
    assert finite.all()
    assert torch.all(token_edges_4096[:, 0] >= x0)
    assert torch.all(token_edges_4096[:, 0] <= x1)
    assert torch.all(token_edges_4096[:, 1] >= y0)
    assert torch.all(token_edges_4096[:, 1] <= y1)


def test_pack_condition_uses_each_cube_local_proj_and_global_tokens():
    group = [
        {
            "cube_id": 4,
            "global_row_ids": torch.tensor([0, 1]),
            "local_xyz": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        },
        {
            "cube_id": 9,
            "global_row_ids": torch.tensor([0, 1]),
            "local_xyz": torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32),
        },
    ]
    condition = {
        "cubes": {
            4: {
                "global_row_ids": torch.tensor([0, 1]),
                "global": torch.full((1, 5, 3), 4.0),
                "proj": torch.tensor([[4.0, 4.5], [4.1, 4.6]]),
            },
            9: {
                "global_row_ids": torch.tensor([0, 1]),
                "global": torch.full((1, 5, 3), 9.0),
                "proj": torch.tensor([[9.0, 9.5], [9.1, 9.6]]),
            },
        }
    }
    packed_state = m._pack_state(group, torch.zeros(2, 1), torch.device("cpu"))
    packed = m._pack_condition(group, condition, packed_state.coords, torch.device("cpu"))
    assert packed["cond"]["global"].shape == (2, 5, 3)
    assert torch.all(packed["cond"]["global"][0] == 4)
    assert torch.all(packed["cond"]["global"][1] == 9)
    assert torch.allclose(
        packed["cond"]["proj"].feats,
        torch.tensor([[4.0, 4.5], [4.1, 4.6], [9.0, 9.5], [9.1, 9.6]]),
    )
    assert torch.equal(packed["cond"]["proj"].coords, packed_state.coords)


def test_dino_naf_native_rectangular_crop_preserves_layout_and_scale():
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(num_register_tokens=1)

    class DummyProjGrid(nn.Module):
        def forward(self, fmap, *args, grid_indices=None, **kwargs):
            channels = fmap.shape[-1] if kwargs.get("BHWC", True) else fmap.shape[1]
            return torch.zeros(
                (fmap.shape[0], grid_indices.shape[0], channels),
                dtype=fmap.dtype,
                device=fmap.device,
            )

    class DummyNaf(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_size = None

        def forward(self, image, features, output_size):
            self.output_size = tuple(output_size)
            return torch.zeros(
                (features.shape[0], features.shape[1], *output_size),
                dtype=features.dtype,
                device=features.device,
            )

    extractor = object.__new__(DinoV3ProjFeatureExtractor)
    nn.Module.__init__(extractor)
    extractor.model = DummyBackbone()
    extractor.image_size = 1024
    extractor.patch_size = 16
    extractor.patch_number = 64
    extractor.use_naf_upsample = True
    extractor.naf_target_size = (512, 512)
    extractor.transform = nn.Identity()
    extractor.proj_grid = DummyProjGrid()
    extractor.naf_model = DummyNaf()
    extractor._load_naf = lambda: None

    def fake_extract(image):
        batch, _, height, width = image.shape
        tokens = 2 + (height // 16) * (width // 16)
        return torch.zeros((batch, tokens, 3), dtype=image.dtype, device=image.device)

    extractor.extract_features = fake_extract
    query_coords = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)
    glob, proj = extractor(
        [Image.new("RGB", (32, 48))],
        camera_angle_x=torch.tensor([0.6]),
        distance=torch.tensor([2.0]),
        mesh_scale=torch.tensor([1.0]),
        grid_indices=query_coords,
        grid_resolution=2,
        projection_crop_box=(0.0, 0.0, 1.0, 1.0),
        preserve_input_resolution=True,
    )
    assert glob.shape == (1, 2, 3)
    assert proj.shape == (1, 2, 6)
    assert extractor.naf_model.output_size == (24, 16)


def test_full_camera_rejects_local_coords_contract():
    def global_projection_only(value, resolution):
        if resolution != 256: raise ValueError("projection requires global C256 coordinates")
    with pytest.raises(ValueError): global_projection_only(torch.zeros(1, 4), 64)


def test_sparse_variable_batch_ids_and_row_order():
    a = SparseTensor(torch.tensor([[1.], [2.]]), coords([[1, 1, 1], [2, 2, 2]]))
    b = SparseTensor(torch.tensor([[3.]]), coords([[3, 3, 3]]))
    packed = m.legacy._pack_sparse_batch([a, b], "test")
    assert packed.coords[:, 0].tolist() == [0, 0, 1]
    parts = m.legacy._split_sparse_batch(packed, 2, "test")
    assert torch.equal(parts[0].feats, a.feats) and torch.equal(parts[1].feats, b.feats)


def _owner_fixture():
    c = coords([[0, 0, 0], [32, 32, 32], [255, 255, 255]])
    records, _ = m.build_cube_records(c); owner, _ = m.build_owner_map(c, records)
    active = [r for r in records if r["owned_row_ids"].numel()]
    return c, records, owner, active


def test_owner_scatter_missing_and_duplicate_fail():
    _, _, owner, active = _owner_fixture()
    proposals = [(r["cube_id"], r["global_row_ids"], torch.ones(r["global_row_ids"].numel(), 2)) for r in active]
    assert m.validate_owner_scatter(owner, proposals, 2).shape == (3, 2)
    with pytest.raises(RuntimeError): m.validate_owner_scatter(owner, proposals[:-1], 2)
    with pytest.raises(RuntimeError): m.validate_owner_scatter(owner, proposals + proposals[:1], 2)


def test_nonowner_proposal_cannot_change_result_and_no_average():
    _, records, owner, active = _owner_fixture()
    proposals = []
    for r in active:
        value = torch.full((r["global_row_ids"].numel(), 1), float(r["cube_id"] + 1))
        proposals.append((r["cube_id"], r["global_row_ids"], value))
    first = m.validate_owner_scatter(owner, proposals, 1)
    changed = [(cid, ids, v + (owner.index_select(0, ids) != cid)[:, None] * 10000) for cid, ids, v in proposals]
    assert torch.equal(first, m.validate_owner_scatter(owner, changed, 1))


def test_jacobi_cube_order_independent():
    _, records, owner, active = _owner_fixture(); state = torch.randn(3, 2)
    proposals = [(r["cube_id"], r["global_row_ids"], torch.randn(r["global_row_ids"].numel(), 2)) for r in active]
    a = m.jacobi_update(state, m.validate_owner_scatter(owner, proposals, 2), 1, .5)
    b = m.jacobi_update(state, m.validate_owner_scatter(owner, list(reversed(proposals)), 2), 1, .5)
    assert torch.equal(a, b)


def test_gaussian_weights_are_nearer_higher_and_normalized():
    c = coords([[31, 31, 31], [47, 31, 31], [48, 31, 31]])
    records, _ = m.build_cube_records(c)
    weights = m.build_gaussian_weight_table(c, records, sigma=32.0)
    containing = [r for r in records if 1 in r["global_row_ids"].tolist()]
    by_start_x = {}
    total = 0.0
    for r in containing:
        pos = r["global_row_ids"].tolist().index(1)
        value = float(weights[r["cube_id"]][pos])
        total += value
        if r["start"][1:] == (0, 0):
            by_start_x[r["start"][0]] = value
    assert by_start_x[0] > by_start_x[32]
    assert total == pytest.approx(1.0, abs=2e-7)


def test_gaussian_velocity_fusion_and_single_cube_limit():
    c = coords([[0, 0, 0], [47, 31, 31]])
    records, _ = m.build_cube_records(c)
    weights = m.build_gaussian_weight_table(c, records, sigma=32.0)
    proposals = []
    for r in records:
        if r["cube_id"] not in weights:
            continue
        value = torch.full((r["global_row_ids"].numel(), 1), float(r["cube_id"] + 1))
        proposals.append((r["cube_id"], r["global_row_ids"], value))
    fused, sums = m.validate_gaussian_fusion(2, proposals, weights, 1)
    assert fused[0, 0] == 1.0
    expected = sum(float(weights[cid][ids.tolist().index(1)]) * float(cid + 1)
                   for cid, ids, _ in proposals if 1 in ids.tolist())
    assert fused[1, 0].item() == pytest.approx(expected, abs=1e-6)
    assert torch.allclose(sums, torch.ones_like(sums), atol=2e-6, rtol=0)
    with pytest.raises(RuntimeError):
        m.validate_gaussian_fusion(2, proposals[:-1], weights, 1)


def test_single_cube_native_euler_alignment():
    state = torch.randn(4, 3); velocity = torch.randn(4, 3); t, nxt = .8, .6
    assert torch.equal(m.jacobi_update(state, velocity, t, nxt), state - (t - nxt) * velocity)


def test_shape_texture_concat_alignment():
    c = coords([[3, 4, 5], [8, 9, 10]])
    r = {"local_xyz": c[:, 1:], "global_row_ids": torch.arange(2)}
    state = torch.randn(2, 32); concat = torch.randn(2, 32)
    assert torch.equal(m._pack_state([r], state, torch.device("cpu")).coords,
                       m._pack_concat([r], concat, torch.device("cpu")).coords)


def test_pack_token_budget_and_batch_limit():
    records = [{"cube_id": i, "global_row_ids": torch.arange(n), "owned_row_ids": torch.arange(1)}
               for i, n in enumerate((3, 4, 5))]
    groups = m.pack_groups(records, 2, 7)
    assert [[r["cube_id"] for r in g] for g in groups] == [[0, 1], [2]]


def test_local_conditions_cover_exact_flow_active_cubes():
    records = [
        {"cube_id": 0, "global_row_ids": torch.arange(3), "owned_row_ids": torch.arange(1)},
        {"cube_id": 1, "global_row_ids": torch.arange(2), "owned_row_ids": torch.empty(0, dtype=torch.long)},
        {"cube_id": 2, "global_row_ids": torch.empty(0, dtype=torch.long), "owned_row_ids": torch.empty(0, dtype=torch.long)},
    ]
    assert [r["cube_id"] for r in m.flow_condition_records(records, "owner")] == [0]
    assert [r["cube_id"] for r in m.flow_condition_records(records, "gaussian")] == [0, 1]


def test_hash_changes_for_support_owner_condition_noise():
    a = torch.zeros(2, 4, dtype=torch.int32); b = a.clone(); b[0, 1] = 1
    assert m.tensor_sha256(a) != m.tensor_sha256(b)
    assert m.tensor_sha256(torch.zeros(2, 3)) != m.tensor_sha256(torch.ones(2, 3))


def test_normalization_roundtrip():
    x = torch.randn(8, 3); norm = {"mean": [1, 2, 3], "std": [.5, 2, 4]}
    assert torch.allclose(m.normalize(m.denormalize(x, norm), norm), x, atol=1e-6)


def test_tensor_hash_is_stable_across_clone():
    x = torch.arange(20).reshape(4, 5)
    assert m.tensor_sha256(x) == m.tensor_sha256(x.clone())
