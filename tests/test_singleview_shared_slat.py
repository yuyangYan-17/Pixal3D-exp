"""CPU contracts for the Codex.md single-view shared-SLat route."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as impl
import pixal3d_singleview_shared_slat_support as support


def test_singleview_tile_layout_is_7_by_7_with_half_overlap():
    boxes = support.tile_boxes()
    assert len(boxes) == 49
    assert boxes[0] == (0, 0, 1024, 1024)
    assert boxes[1] == (512, 0, 1536, 1024)
    assert boxes[26] == (2560, 1536, 3584, 2560)
    assert boxes[-1] == (3072, 3072, 4096, 4096)


def test_route_proj_rows_only_changes_projection_rows():
    front = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    back = front + 100.0
    visible = torch.tensor([True, False, True, False])
    routed = support.route_proj_rows(front, back, visible)
    assert torch.equal(routed[visible], front[visible])
    assert torch.equal(routed[~visible], back[~visible])


def test_route_does_not_replace_front_global_condition():
    front_global = torch.tensor([[[1.0, 2.0]]])
    back_global = torch.tensor([[[9.0, 9.0]]])
    front_proj = torch.zeros((2, 4))
    back_proj = torch.ones((2, 4))
    routed = support.route_proj_rows(
        front_proj,
        back_proj,
        torch.tensor([True, False]),
    )
    assert torch.equal(front_global, torch.tensor([[[1.0, 2.0]]]))
    assert not torch.equal(front_global, back_global)
    assert torch.equal(routed[0], front_proj[0])
    assert torch.equal(routed[1], back_proj[1])


def test_parser_uses_fresh_singleview_output_and_cuda4():
    args = impl.build_parser().parse_args([])
    assert args.cuda_device == 4
    assert args.resume is False
    assert args.experiment == "front_only"
    assert "singleview_shared_slat_shape_tex_sr_cuda4" in str(args.output_root)


def test_experiment_output_names_are_separate():
    args = impl.build_parser().parse_args(["--experiment", "front_global_back_proj"])
    assert impl._resolve_output_dir(args).name == "exp_b_front_global_back_proj"
    args = impl.build_parser().parse_args(["--experiment", "baseline4096_from1024"])
    assert impl._resolve_output_dir(args).name == "exp_c_baseline4096_from1024"

