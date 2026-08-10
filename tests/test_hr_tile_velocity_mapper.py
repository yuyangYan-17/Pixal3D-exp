import numpy as np
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pixal3d_hr_tile_velocity_mapper as experiment


def test_mapper_has_exact_identity_initialization_and_zero_residual_invariant():
    mapper = experiment.LowRankVelocityMapper(channels=32)
    g = torch.randn(13, 32)
    d = torch.randn(13, 32)
    assert torch.count_nonzero(mapper(g, d)) == 0
    with torch.no_grad():
        mapper.base_map.weight.normal_()
        mapper.modulator[-1].weight.normal_()
        mapper.modulator[-1].bias.normal_()
    assert torch.count_nonzero(mapper(g, torch.zeros_like(d))) == 0


def test_owner_assignment_is_unique_and_uses_low_tile_id_for_ties():
    uv = np.array([[512.0, 512.0], [768.0, 512.0], [1100.0, 600.0]])
    valid = np.ones(3, dtype=np.bool_)
    indices = [np.empty(0, dtype=np.int64) for _ in range(49)]
    # Tile 0 center=(512,512), tile 1 center=(1024,512). Point 1 ties.
    indices[0] = np.array([0, 1])
    indices[1] = np.array([1, 2])
    owner, counts, eligible = experiment.owner_assignments(
        uv, valid, indices, min_tokens=1
    )
    assert eligible == [0, 1]
    assert owner.tolist() == [0, 0, 1]
    assert counts.sum() == np.count_nonzero(owner >= 0)


def test_shared_global_state_is_gathered_bit_exactly():
    x0 = torch.randn(31, 32)
    epsilon = torch.randn_like(x0)
    t = 0.37
    xt = (1.0 - t) * x0 + (
        experiment.SIGMA_MIN + (1.0 - experiment.SIGMA_MIN) * t
    ) * epsilon
    rows = torch.tensor([2, 7, 11, 29])
    assert torch.equal(xt[rows], xt.index_select(0, rows))


def test_nested_source_crop_matches_direct_canonical_tile_coordinates():
    camera = {
        "canonical_preprocess": {
            "source_size": [4096, 4096],
            "square_extent_source": [600, 1051, 3495, 3946],
        }
    }
    box = experiment.tile_boxes()[17]
    crop = experiment.source_crop_box(camera, box)
    left, top, right, _ = camera["canonical_preprocess"]["square_extent_source"]
    side = right - left
    canonical_uv = np.array([[box[0] + 12.5, box[1] + 300.25]])
    raw_uv = np.array([
        [left + canonical_uv[0, 0] * side / 4096, top + canonical_uv[0, 1] * side / 4096]
    ])
    crop_pixels = (raw_uv / 4096 - np.array(crop[:2])) / (
        np.array(crop[2:]) - np.array(crop[:2])
    ) * 1024
    direct = canonical_uv - np.array(box[:2])
    assert np.allclose(crop_pixels, direct, atol=1e-10)


def test_object_split_is_disjoint_and_deterministic():
    objects = [f"object-{index}" for index in range(10)]
    first = experiment.split_objects(objects, seed=7, test_fraction=0.2)
    second = experiment.split_objects(objects, seed=7, test_fraction=0.2)
    assert first == second
    train, test = first
    assert len(train) == 8 and len(test) == 2
    assert set(train).isdisjoint(test)


def test_loss_comparison_reports_gain_and_degradation_with_consistent_signs():
    improved = experiment.compare_loss(2.0, 1.5)
    assert improved["status"] == "improved"
    assert improved["delta"] == -0.5
    assert improved["gain_percent"] == 25.0

    degraded = experiment.compare_loss(2.0, 2.5)
    assert degraded["status"] == "degraded"
    assert degraded["delta"] == 0.5
    assert degraded["gain_percent"] == -25.0


def test_auto_batch_plan_uses_free_vram_and_model_token_counts():
    plan = experiment.design_batch_plan(
        model_token_counts=[10_000, 20_000, 30_000],
        owner_token_counts=[100, 200, 300],
        free_memory_bytes=16 * 2**30,
        total_memory_bytes=24 * 2**30,
        requested_batch_size=0,
        requested_model_token_budget=0,
        max_auto_batch_size=8,
        memory_fraction=0.5,
    )
    assert plan.model_token_budget == 32_768
    assert plan.max_examples == 1
    assert plan.item_model_tokens_median == 20_000
    assert plan.item_owner_tokens_median == 200

    manual = experiment.design_batch_plan(
        model_token_counts=[10_000, 20_000, 30_000],
        owner_token_counts=[100, 200, 300],
        free_memory_bytes=16 * 2**30,
        total_memory_bytes=24 * 2**30,
        requested_batch_size=4,
        requested_model_token_budget=50_000,
        max_auto_batch_size=8,
        memory_fraction=0.5,
    )
    assert manual.max_examples == 4
    assert manual.model_token_budget == 50_000


def test_training_batch_respects_example_and_model_token_limits():
    def make_item(tile_id):
        return experiment.TileItem(
            split="train",
            object_id="object",
            view_name="view_000",
            tile_id=tile_id,
            object_dir="/tmp/object",
            view_dir="/tmp/object/view_000",
            tile_rows=(0, 1),
            owner_positions=(0,),
        )

    items = [make_item(index) for index in range(4)]
    # The function pops from the end: item 3 + item 2 fit, item 1 would exceed.
    counts = {item.key: 100 + item.tile_id * 10 for item in items}
    selected = experiment._take_training_batch(
        items.copy(), counts, max_examples=3, model_token_budget=250
    )
    assert [item.tile_id for item in selected] == [3, 2]
