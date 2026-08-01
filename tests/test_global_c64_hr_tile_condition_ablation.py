from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pixal3d_global_c64_hr_tile_condition_ablation as experiment
from pixal3d.modules.sparse import SparseTensor


def test_complete_tile_layout_is_exactly_7_by_7() -> None:
    boxes = experiment._tile_layout()
    assert len(boxes) == 49
    assert boxes[0] == (0, 0, 1024, 1024)
    assert boxes[6] == (3072, 0, 4096, 1024)
    assert boxes[-1] == (3072, 3072, 4096, 4096)
    assert all(x1 - x0 == 1024 and y1 - y0 == 1024 for x0, y0, x1, y1 in boxes)


def test_global_rows_are_assigned_without_coordinate_changes() -> None:
    projected = torch.tensor(
        [
            [0.125, 0.125],
            [0.500, 0.500],
            [0.875, 0.875],
            [-0.010, 0.500],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones(4, dtype=torch.bool)
    tiles, summary = experiment._build_global_row_tiles(
        image_4096=Image.new("RGB", (4096, 4096)),
        projected_full_norm=projected,
        projection_valid=valid,
    )

    assert len(tiles) == 49
    assert summary["eligible_row_count"] == 3
    assert summary["covered_row_count"] == 3
    assert summary["uncovered_row_count"] == 1
    assert int(summary["coverage"][0]) == 4
    assert int(summary["coverage"][1]) == 4
    assert int(summary["coverage"][2]) == 1
    assert int(summary["coverage"][3]) == 0
    for tile in tiles:
        rows = tile["global_rows"]
        assert rows.dtype == torch.long
        assert torch.equal(rows, torch.sort(rows).values)


class _FakePipeline:
    low_vram = False
    device = torch.device("cpu")

    @staticmethod
    def _materialize_proj_condition(packed, coords, device):
        marker = packed["marker"]
        projected = SparseTensor(
            feats=torch.zeros(coords.shape[0], 1, device=device),
            coords=coords,
        )
        return {
            "cond": {
                "global": torch.tensor([marker], device=device),
                "proj": projected,
            },
            "neg_cond": {
                "global": torch.zeros(1, device=device),
                "proj": projected.replace(torch.zeros_like(projected.feats)),
            },
        }


class _FakeSampler:
    @staticmethod
    def timestep_schedule(steps, rescale_t):
        assert steps == 1
        return [1.0, 0.0]

    @staticmethod
    def _get_model_prediction(
        model,
        state,
        timestep,
        condition,
        **kwargs,
    ):
        del model, timestep, kwargs
        marker = float(condition["global"].reshape(-1)[0].item())
        velocity = state.replace(torch.full_like(state.feats, marker))
        return state, state, velocity


def test_velocity_overlap_is_arithmetic_mean_before_one_global_update() -> None:
    coords = torch.tensor(
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int32
    )
    noise = SparseTensor(feats=torch.zeros(3, 2), coords=coords)
    tiles = [
        {
            "tile_id": 0,
            "enabled": True,
            "global_rows": torch.tensor([0, 1]),
            "shape_condition_cpu": {"marker": 2.0},
        },
        {
            "tile_id": 1,
            "enabled": True,
            "global_rows": torch.tensor([1, 2]),
            "shape_condition_cpu": {"marker": 4.0},
        },
    ]
    final, trace = experiment._run_tiled_global_flow(
        pipeline=_FakePipeline(),
        model=torch.nn.Identity(),
        sampler=_FakeSampler(),
        initial_noise=noise,
        global_condition_cpu={"marker": 9.0},
        tiles=tiles,
        stage_name="shape",
        sampler_params={"steps": 1, "rescale_t": 1.0},
        concat_cond=None,
    )

    expected = torch.tensor([[-2.0, -2.0], [-3.0, -3.0], [-4.0, -4.0]])
    assert torch.equal(final.coords, coords)
    assert torch.equal(final.feats, expected)
    assert trace["global_update_count_per_step"] == 1
    assert trace["steps_detail"][0]["overlap_rows"] == 1
    assert trace["steps_detail"][0]["uncovered_global_fallback_rows"] == 0
