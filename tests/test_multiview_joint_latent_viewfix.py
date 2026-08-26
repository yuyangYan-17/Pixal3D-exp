from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pixal3d_global4096_multiview_joint_shape_tex_sr as route


def test_mapping_preserves_local_rows_and_marks_one_donor_per_master(monkeypatch):
    master = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(route, "_yaw_matrix", lambda angle: torch.eye(3))
    monkeypatch.setattr(route, "_world_to_view_q", lambda q, rotation: q)
    monkeypatch.setattr(
        route.core,
        "_local_q_to_global_q",
        lambda q, **kwargs: (q.clone(), torch.full((q.shape[0], 2), 100.0)),
    )
    monkeypatch.setattr(
        route.core,
        "_project_global_q_to_image",
        lambda q, **kwargs: (
            torch.full((q.shape[0], 2), 100.0),
            torch.ones(q.shape[0]),
            torch.ones(q.shape[0], dtype=torch.bool),
        ),
    )

    native = torch.tensor(
        [[0, 31, 31, 31], [0, 32, 32, 32]], dtype=torch.int32
    )
    transform = SimpleNamespace(
        mesh_scale=1.0,
        box=(0, 0, 256, 256),
        source_width=1024,
        source_height=1024,
    )
    ids, coords, uv, _, representative, stats = route._map_master_to_context(
        master_q_world=master,
        native_coords=native,
        angle=120,
        transform=transform,
        camera={"camera_angle_x": 0.5, "distance": 2.0, "mesh_scale": 1.0},
        virtual_box=(0, 0, 1024, 1024),
    )

    assert ids.tolist() == [0, 0]
    assert torch.equal(coords, native)
    assert uv.tolist() == [[400.0, 400.0], [400.0, 400.0]]
    assert representative.sum().item() == 1
    assert stats["selected_local_rows"] == 2
    assert stats["unique_master_rows"] == 1
    assert stats["duplicate_local_receipts"] == 1
    assert stats["mapping_direction"] == "local_c64_to_world_to_nearest_global_master"


def test_texture_flow_is_direct_latent_endpoint_fusion():
    names = set(route._run_texture_flow.__code__.co_names)
    assert "_fuse_endpoint" in names
    assert "_fuse_pbr_at_master" not in names
    assert "_decode_texture_batches" not in names
    assert "_encode_pbr_fields_batch" not in names


def test_fusion_uses_representative_visible_rows_and_global_fallback():
    context = SimpleNamespace(
        context_id=7,
        master_ids=torch.tensor([0, 0, 1]),
        visible=torch.tensor([True, True, False]),
        donor_representative=torch.tensor([True, False, True]),
        gaussian_weight=torch.tensor([0.5, 1.0, 1.0]),
    )
    prediction = route.SparseTensor(
        torch.tensor([[2.0], [100.0], [9.0]]),
        torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]], dtype=torch.int32),
    )
    merged, count, fallback_mask, _ = route._fuse_endpoint(
        contexts=[context],
        predictions={7: prediction},
        fallback=torch.tensor([[11.0], [22.0]]),
        master_count=2,
        channel_count=1,
        stage="test",
    )
    assert merged[:, 0].tolist() == [2.0, 22.0]
    assert count.tolist() == [1, 0]
    assert fallback_mask.tolist() == [False, True]
