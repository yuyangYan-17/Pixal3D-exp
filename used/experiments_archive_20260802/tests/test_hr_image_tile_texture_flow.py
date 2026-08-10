import inspect
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pixal3d_directory_texture_eval as directory_eval
from pixal3d.pipelines.pixal3d_image_to_3d import (
    Pixal3DImageTo3DPipeline,
)
from pixal3d.modules.sparse import SparseTensor
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid,
    project_points_to_image_batch,
    sample_features,
)


def _pipeline_without_models() -> Pixal3DImageTo3DPipeline:
    pipeline = object.__new__(Pixal3DImageTo3DPipeline)
    pipeline.low_vram = False
    return pipeline


def test_shared_preprocess_preserves_original_global_alpha_path():
    width, height = 1600, 1200
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
    rgba[..., 1] = 90
    rgba[..., 2] = 170
    rgba[170:1040, 320:1280, 3] = 255
    source = Image.fromarray(rgba, mode="RGBA")
    pipeline = _pipeline_without_models()

    original_global = pipeline.preprocess_image(source)
    bundle = pipeline.preprocess_image_with_hr(source)

    assert bundle["global_image"].size == original_global.size
    np.testing.assert_array_equal(
        np.asarray(bundle["global_image"]),
        np.asarray(original_global),
    )
    assert bundle["hr_image"].width == bundle["hr_image"].height
    assert bundle["hr_image"].width > bundle["global_image"].width
    assert bundle["foreground_mask_hr"].size == bundle["hr_image"].size

    forward = np.asarray(
        bundle["global_to_hr_transform"]["global_to_hr_matrix"],
        dtype=np.float64,
    )
    inverse = np.asarray(
        bundle["global_to_hr_transform"]["hr_to_global_matrix"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        forward @ inverse,
        np.eye(3),
        rtol=0.0,
        atol=1e-12,
    )


def test_shared_preprocess_calls_rembg_once_for_global_and_hr():
    class FakeRemBg:
        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            array = np.asarray(image.convert("RGB"))
            alpha = np.zeros(array.shape[:2], dtype=np.uint8)
            alpha[
                array.shape[0] // 5 : array.shape[0] * 4 // 5,
                array.shape[1] // 4 : array.shape[1] * 3 // 4,
            ] = 255
            return Image.fromarray(
                np.dstack((array, alpha)),
                mode="RGBA",
            )

    pipeline = _pipeline_without_models()
    pipeline.rembg_model = FakeRemBg()
    source = Image.new("RGB", (1800, 1200), (40, 120, 210))
    bundle = pipeline.preprocess_image_with_hr(source)

    assert pipeline.rembg_model.calls == 1
    assert bundle["global_image"].width <= 1024
    assert bundle["hr_image"].width > bundle["global_image"].width
    assert bundle["foreground_mask_hr"].size == bundle["hr_image"].size


def test_synthetic_projection_and_token_assignment_use_global_order():
    image = Image.new("RGB", (64, 64), (120, 80, 40))
    mask_array = np.zeros((64, 64), dtype=np.uint8)
    mask_array[:, :] = 255
    mask = Image.fromarray(mask_array, mode="L")
    projected = torch.tensor(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones(4, dtype=torch.bool)

    tiles, summary = Pixal3DImageTo3DPipeline._build_hr_image_tiles(
        hr_image=image,
        foreground_mask_hr=mask,
        projected_full_norm=projected,
        projection_valid=valid,
        tile_size=32,
        tile_stride=32,
        min_foreground_ratio=0.0,
        weight_mode="uniform",
    )

    active = [tile for tile in tiles if tile["enabled"]]
    assert len(active) == 4
    assert [tile["token_indices"].item() for tile in active] == [0, 1, 2, 3]
    assert summary["covered_token_count"] == 4
    assert summary["overlap_token_count"] == 0


def test_projection_debug_artifacts_include_every_tile_image():
    image = Image.new("RGB", (32, 32), (100, 120, 140))
    mask = Image.new("L", (32, 32), 255)
    projected = torch.tensor(
        [[0.25, 0.25], [0.75, 0.75]],
        dtype=torch.float32,
    )
    valid = torch.ones(2, dtype=torch.bool)
    tiles, _ = Pixal3DImageTo3DPipeline._build_hr_image_tiles(
        hr_image=image,
        foreground_mask_hr=mask,
        projected_full_norm=projected,
        projection_valid=valid,
        tile_size=16,
        tile_stride=16,
        min_foreground_ratio=0.0,
        weight_mode="tent",
    )
    transform = {
        "global_size": [32, 32],
        "hr_size": [32, 32],
        "global_to_hr_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "hr_to_global_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        artifacts = Pixal3DImageTo3DPipeline._save_hr_image_tile_debug(
            debug_dir=temporary_directory,
            global_image=image,
            hr_image=image,
            foreground_mask_hr=mask,
            projected_full_norm=projected,
            projection_valid=valid,
            tiles=tiles,
            global_to_hr_transform=transform,
        )
        assert Path(artifacts["projection_global"]).is_file()
        assert Path(artifacts["projection_hr_tiles"]).is_file()
        assert Path(artifacts["foreground_mask_hr"]).is_file()
        assert Path(artifacts["tile_metadata"]).is_file()
        assert len(list(Path(artifacts["tile_directory"]).glob("tile_*.png"))) == 4


def test_disabled_trace_schema_remains_v3_while_tile_trace_is_v4():
    trajectory = SimpleNamespace(
        times=torch.linspace(1.0, 0.0, 13).tolist(),
        time_intervals=[1.0 / 12.0] * 12,
        states=[torch.zeros(2, 1) for _ in range(13)],
        velocities=[torch.zeros(2, 1) for _ in range(12)],
    )
    global_flow = SimpleNamespace(trajectory=trajectory)
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1]],
        dtype=torch.int32,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        disabled_path = Path(temporary_directory) / "disabled.pt"
        Pixal3DImageTo3DPipeline._save_2048_flow_trace(
            output_path=disabled_path,
            stage_name="texture",
            coords=coords,
            sampler_params={"steps": 12, "rescale_t": 3.0},
            global_flow=global_flow,
            patch_trace={"enabled": False, "status": "global_only_complete"},
        )
        disabled = torch.load(disabled_path, weights_only=False)
        assert disabled["format"] == "pixal3d_2048_texture_flow_trace_v3"
        assert set(disabled["global_flow"]) == {
            "times",
            "time_intervals",
            "states",
            "velocities",
            "base_latent_state_index",
        }

        tile_path = Path(temporary_directory) / "tile.pt"
        Pixal3DImageTo3DPipeline._save_2048_flow_trace(
            output_path=tile_path,
            stage_name="texture",
            coords=coords,
            sampler_params={"steps": 12, "rescale_t": 3.0},
            global_flow=global_flow,
            patch_trace={
                "enabled": True,
                "mode": "hr_image_tile_velocity_flow",
            },
        )
        tile = torch.load(tile_path, weights_only=False)
        assert tile["format"] == "pixal3d_2048_texture_flow_trace_v4"
        assert "raw_times" in tile["global_flow"]
        assert "mapped_times" in tile["global_flow"]


def test_global_hr_tile_coordinate_round_trip():
    global_norm = torch.tensor(
        [[0.3, 0.4], [0.49, 0.72]],
        dtype=torch.float64,
    )
    crop_box = (0.25, 0.25, 0.75, 0.75)
    tile_norm = Pixal3DImageTo3DPipeline._global_norm_to_tile_norm(
        global_norm,
        crop_box,
    )
    recovered = Pixal3DImageTo3DPipeline._tile_norm_to_global_norm(
        tile_norm,
        crop_box,
    )
    torch.testing.assert_close(recovered, global_norm, rtol=0.0, atol=1e-12)

    global_size = 1024.0
    hr_size = 4096.0
    global_pixels = global_norm * global_size
    hr_pixels = global_pixels * (hr_size / global_size)
    recovered_global_pixels = hr_pixels * (global_size / hr_size)
    torch.testing.assert_close(
        recovered_global_pixels,
        global_pixels,
        rtol=0.0,
        atol=1e-12,
    )


def test_scatter_merge_preserves_global_token_order_and_fallback():
    velocity_sum = torch.zeros((4, 2), dtype=torch.float32)
    weight_sum = torch.zeros((4, 1), dtype=torch.float32)
    coverage_count = torch.zeros(4, dtype=torch.int32)
    indices = torch.tensor([2, 0], dtype=torch.long)
    tile_velocity = torch.tensor(
        [[20.0, 21.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    Pixal3DImageTo3DPipeline._scatter_add_tile_velocity(
        velocity_sum,
        weight_sum,
        coverage_count,
        indices,
        tile_velocity,
        torch.ones(2),
    )
    fallback = torch.tensor(
        [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0], [130.0, 131.0]]
    )
    merged, covered = Pixal3DImageTo3DPipeline._finalize_tile_velocity(
        velocity_sum,
        weight_sum,
        fallback,
    )

    torch.testing.assert_close(merged[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(merged[2], torch.tensor([20.0, 21.0]))
    torch.testing.assert_close(merged[1], fallback[1])
    torch.testing.assert_close(merged[3], fallback[3])
    assert covered.tolist() == [True, False, True, False]


def test_overlap_velocity_is_weighted_before_single_global_update():
    velocity_sum = torch.zeros((3, 1), dtype=torch.float32)
    weight_sum = torch.zeros((3, 1), dtype=torch.float32)
    coverage_count = torch.zeros(3, dtype=torch.int32)
    shared_index = torch.tensor([1], dtype=torch.long)
    Pixal3DImageTo3DPipeline._scatter_add_tile_velocity(
        velocity_sum,
        weight_sum,
        coverage_count,
        shared_index,
        torch.tensor([[2.0]]),
        torch.tensor([0.25]),
    )
    Pixal3DImageTo3DPipeline._scatter_add_tile_velocity(
        velocity_sum,
        weight_sum,
        coverage_count,
        shared_index,
        torch.tensor([[6.0]]),
        torch.tensor([0.75]),
    )
    merged, covered = Pixal3DImageTo3DPipeline._finalize_tile_velocity(
        velocity_sum,
        weight_sum,
        torch.tensor([[9.0], [9.0], [9.0]]),
    )
    torch.testing.assert_close(merged[:, 0], torch.tensor([9.0, 5.0, 9.0]))
    assert covered.tolist() == [False, True, False]
    assert coverage_count.tolist() == [0, 2, 0]


def test_synthetic_tile_flow_merges_each_step_then_updates_global_once():
    class FakeSampler:
        def _get_model_prediction(
            self,
            model,
            x_t,
            t,
            cond,
            **kwargs,
        ):
            value = float(cond["global"].reshape(-1)[0].item())
            velocity = x_t.replace(torch.full_like(x_t.feats, value))
            return x_t, x_t, velocity

    def packed_condition(token_count, value):
        return {
            "cond": {
                "global": torch.tensor([[[float(value)]]]),
                "proj": torch.zeros(token_count, 1),
            },
            "neg_cond": {
                "global": torch.zeros(1, 1, 1),
                "proj": torch.zeros(token_count, 1),
            },
        }

    def tile(tile_index, indices, value):
        token_indices = torch.tensor(indices, dtype=torch.long)
        return {
            "tile_index": tile_index,
            "box_hr": (0, 0, 32, 32),
            "box_hr_actual": (0, 0, 32, 32),
            "projection_crop_box": (0.0, 0.0, 0.5, 0.5),
            "foreground_pixels": 1024,
            "foreground_ratio": 1.0,
            "foreground_enabled": True,
            "enabled": True,
            "skipped_reason": None,
            "token_indices": token_indices,
            "token_count": len(indices),
            "weights": torch.ones(len(indices)),
            "tile_image_path": None,
            "condition_cpu": packed_condition(len(indices), value),
        }

    pipeline = _pipeline_without_models()
    pipeline.image_cond_model_tex_1024 = SimpleNamespace(
        use_naf_upsample=True
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [0, 3, 3, 3],
        ],
        dtype=torch.int32,
    )
    noise = SparseTensor(feats=torch.zeros(4, 1), coords=coords)
    shape_cond = SparseTensor(feats=torch.ones(4, 2), coords=coords)
    mapped_times = torch.linspace(1.0, 0.0, 13).tolist()
    intervals = [
        mapped_times[index] - mapped_times[index + 1]
        for index in range(12)
    ]
    trajectory = SimpleNamespace(
        times=mapped_times,
        time_intervals=intervals,
        states=[torch.zeros(4, 1) for _ in range(13)],
        velocities=[torch.full((4, 1), 9.0) for _ in range(12)],
    )
    global_flow = SimpleNamespace(trajectory=trajectory)

    with patch.object(torch.cuda, "is_available", return_value=False):
        result, trace, diagnostics = (
            pipeline._run_hr_image_tile_texture_flow(
                flow_model=object(),
                sampler=FakeSampler(),
                global_noise=noise,
                global_condition_cpu=packed_condition(4, 5.0),
                shape_concat_cond=shape_cond,
                sampler_params={
                    "steps": 12,
                    "rescale_t": 1.0,
                    "guidance_strength": 1.0,
                    "guidance_interval": (0.6, 0.9),
                    "guidance_rescale": 0.0,
                },
                global_flow=global_flow,
                tiles=[
                    tile(0, [0, 1], 2.0),
                    tile(1, [1, 2], 6.0),
                ],
                start_step=6,
                fallback_mode="saved_global",
                weight_mode="uniform",
                condition_extraction={
                    "active_tile_count": 2,
                    "dino_per_active_tile": True,
                    "naf_per_active_tile": True,
                    "features_premerged": False,
                },
            )
        )

    remaining_time = sum(intervals[6:])
    expected = -remaining_time * torch.tensor([[2.0], [4.0], [6.0], [9.0]])
    torch.testing.assert_close(result.feats, expected)
    assert torch.equal(result.coords, coords)
    assert len(trace["steps"]) == 6
    assert all(
        record["active_tile_count"] == 2 for record in trace["steps"]
    )
    assert all(
        record["overlap_token_count"] == 1 for record in trace["steps"]
    )
    assert diagnostics["feature_fusion"] is False
    assert diagnostics["velocity_fusion"] is True

    with patch.object(torch.cuda, "is_available", return_value=False):
        current_fallback_result, current_trace, _ = (
            pipeline._run_hr_image_tile_texture_flow(
                flow_model=object(),
                sampler=FakeSampler(),
                global_noise=noise,
                global_condition_cpu=packed_condition(4, 5.0),
                shape_concat_cond=shape_cond,
                sampler_params={
                    "steps": 12,
                    "rescale_t": 1.0,
                    "guidance_strength": 1.0,
                    "guidance_interval": (0.6, 0.9),
                    "guidance_rescale": 0.0,
                },
                global_flow=global_flow,
                tiles=[
                    tile(0, [0, 1], 2.0),
                    tile(1, [1, 2], 6.0),
                ],
                start_step=6,
                fallback_mode="current_global",
                weight_mode="uniform",
                condition_extraction={
                    "active_tile_count": 2,
                    "dino_per_active_tile": True,
                    "naf_per_active_tile": True,
                    "features_premerged": False,
                },
            )
        )
    expected_current = -remaining_time * torch.tensor(
        [[2.0], [4.0], [6.0], [5.0]]
    )
    torch.testing.assert_close(
        current_fallback_result.feats,
        expected_current,
    )
    assert all(
        record["current_global_fallback_evaluated"]
        for record in current_trace["steps"]
    )


def test_proj_grid_sparse_projection_shapes():
    grid = ProjGrid(grid_resolution=4, image_resolution=64)
    image_points, depth, valid = grid.project_grid_indices(
        camera_angle_x=torch.tensor([0.85]),
        distance=torch.tensor([2.0]),
        mesh_scale=torch.tensor([1.0]),
        grid_indices=torch.tensor([[0, 0, 0], [3, 3, 3]]),
        grid_resolution=4,
    )
    assert image_points.shape == (1, 2, 2)
    assert depth.shape == (1, 2)
    assert valid.shape == (1, 2)
    assert torch.isfinite(image_points).all()
    assert torch.isfinite(depth).all()


def test_disabled_projection_path_matches_prechange_equations_bitwise():
    torch.manual_seed(7)
    grid = ProjGrid(grid_resolution=4, image_resolution=64)
    feature_map = torch.randn(1, 4, 4, 3)
    camera_angle_x = torch.tensor([0.85])
    distance = torch.tensor([2.0])
    mesh_scale = torch.tensor([1.0])
    grid_indices = torch.tensor([[0, 1, 2], [3, 2, 1]])

    actual = grid(
        feature_map,
        camera_angle_x,
        distance,
        mesh_scale,
        grid_indices=grid_indices,
        grid_resolution=4,
    )

    one_dim = torch.linspace(-1, 1, 4, dtype=grid.grid_points.dtype)
    unrotated = one_dim[grid_indices].unsqueeze(0)
    points = torch.stack(
        (
            unrotated[..., 0],
            -unrotated[..., 2],
            unrotated[..., 1],
        ),
        dim=-1,
    )
    points = points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
    transform = grid.front_view_transform_matrix.expand(1, -1, -1).clone()
    transform[:, 1, 3] = -distance
    image_points, _, _ = project_points_to_image_batch(
        points,
        transform,
        camera_angle_x,
        64,
    )
    query = (image_points + 0.5) / 64 * 2 - 1
    expected = sample_features(
        feature_map.permute(0, 3, 1, 2),
        query,
    ).permute(0, 2, 1)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_disabled_cli_and_pipeline_defaults_regress_to_existing_path():
    original_argv = sys.argv
    try:
        sys.argv = [
            "pixal3d_directory_texture_eval.py",
            "--image-dir",
            "images",
            "--output-dir",
            "outputs",
        ]
        args = directory_eval.parse_args()
    finally:
        sys.argv = original_argv
    assert args.hr_image_tile_texture_flow is False
    assert "__hrtile" not in args.experiment_tag
    assert args.experiment_tag == (
        "s-original-saved-step6-g1-original_interval-r0-whaar"
        "__t-global-saved-step6-g1-original_interval-r0-whaar"
    )
    assert args.texture_mode == "global_original"

    default = inspect.signature(
        Pixal3DImageTo3DPipeline.run
    ).parameters["hr_image_tile_texture_flow"].default
    assert default is False


def test_enabled_cli_encodes_tile_configuration_in_tag():
    original_argv = sys.argv
    try:
        sys.argv = [
            "pixal3d_directory_texture_eval.py",
            "--image-dir",
            "images",
            "--output-dir",
            "outputs",
            "--hr-image-tile-texture-flow",
            "--hr-image-tile-size",
            "1024",
            "--hr-image-tile-stride",
            "512",
            "--hr-image-tile-start-step",
            "6",
            "--hr-image-tile-min-foreground-ratio",
            "0.01",
            "--hr-image-tile-fallback",
            "current_global",
            "--hr-image-tile-weight",
            "uniform",
        ]
        args = directory_eval.parse_args()
    finally:
        sys.argv = original_argv
    assert args.hr_image_tile_texture_flow is True
    assert (
        "__hrtile-size1024-stride512-step6-minfg0p01"
        "-fallbackcurrent_global-weightuniform"
    ) in args.experiment_tag


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
