import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from sr_tools.visibility import latent_visibility, classify_blocks
from sr_tools.texture_guidance import query_field, routed_predictions
from sr_tools.texture_explore import _make_blocks
from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler


class VisibilityGuidance(unittest.TestCase):
    def test_any_visible_descendant_and_exact_threshold(self):
        voxels = torch.tensor(
            [[1, 2, 3], [15, 2, 3], [16, 2, 3], [17, 2, 3]], dtype=torch.int32
        )
        coarse = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.int32)
        visible, voxel_visible, parents = latent_visibility(voxels, voxels[[0]], coarse)
        self.assertEqual(visible.tolist(), [False, True])
        self.assertEqual(parents.tolist(), [1, 1, 0, 0])
        self.assertEqual(voxel_visible.tolist(), [True, False, False, False])
        blocks = [
            dict(block_id=0, global_ids=torch.arange(10)),
            dict(block_id=1, global_ids=torch.arange(9)),
        ]
        with tempfile.TemporaryDirectory() as d:
            classify_blocks(
                dict(blocks=blocks), torch.tensor([True] * 3 + [False] * 7), Path(d)
            )
        self.assertFalse(blocks[0]["conditional"])  # exactly 30%, not >30%
        self.assertTrue(blocks[1]["conditional"])

    def test_endpoints_and_step_routing(self):
        sampler = FlowEulerSampler(1e-5)
        pipe = SimpleNamespace(tex_slat_sampler=sampler)
        x = torch.randn(6, 3)
        guide = torch.randn(6, 3)
        group = [
            dict(block_id=0, conditional=True, global_ids=torch.arange(3), visible_mask=torch.ones(3, dtype=torch.bool)),
            dict(block_id=1, conditional=False, global_ids=torch.arange(3, 6), visible_mask=torch.zeros(3, dtype=torch.bool)),
        ]

        def predict(
            pipe, model, blocks, x, shape, bank, t, params, unconditional=False
        ):
            return [
                torch.full((len(b["global_ids"]), 3), -2.0 if unconditional else 2.0)
                for b in blocks
            ]

        with patch(
            "sr_tools.texture_guidance.texture.predict_safe", side_effect=predict
        ):
            for n in range(1, 5):
                for step in range(12):
                    t = sampler.timestep_schedule(12, 3)[step]
                    out = routed_predictions(
                        pipe, None, group, x, None, None, t, {}, step, n, guide
                    )
                    torch.testing.assert_close(out[0], torch.full((3, 3), 2.0))
                    if step < n:
                        recovered = sampler._pred_to_xstart(x[3:], t, out[1])
                        torch.testing.assert_close(
                            recovered, guide[3:], atol=2e-6, rtol=2e-6
                        )
                    else:
                        torch.testing.assert_close(out[1], torch.full((3, 3), -2.0))
            out = routed_predictions(
                pipe, None, group, x, None, None, 0.8, {}, 0, 0, None
            )
            self.assertTrue((out[0] == 2).all())
            self.assertTrue((out[1] == -2).all())

    def test_nearest_three_inverse_distance_query(self):
        field = dict(
            coords=torch.tensor(
                [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]], dtype=torch.int32
            ),
            attrs=torch.tensor([[1.0], [3.0], [5.0]]),
            resolution=4,
        )
        # The first point is an exact sparse-field centre.  The second point is
        # between the other two and must be a finite distance-weighted blend.
        xyz = torch.tensor(
            [[0.5, 0.5, 0.5], [1.5, 1.5, 0.5]], dtype=torch.float32
        ) / 4 - 0.5
        values, valid, distance = query_field(field, xyz, return_distance=True)
        self.assertEqual(valid.tolist(), [True, True])
        torch.testing.assert_close(values[0], torch.tensor([1.0]))
        self.assertTrue(float(values[1]) > 1.0 and float(values[1]) < 5.0)
        self.assertTrue(torch.isfinite(distance).all())

    def test_mixed_block_separate_full_context_forwards(self):
        block = dict(
            block_id=7,
            tile_id=0,
            rows=torch.arange(4),
            coords=torch.cat((torch.zeros(4, 1, dtype=torch.int32), torch.arange(4).view(-1, 1),
                              torch.zeros(4, 2, dtype=torch.int32)), dim=1),
            global_ids=torch.arange(4),
            owned=torch.ones(4, dtype=torch.bool),
            depth_weights=torch.ones(4),
            conditional=False,
        )
        branches = _make_blocks(
            dict(blocks=[block]),
            visible=torch.tensor([True, False, False, False]),
            coverage=torch.ones(4),
            scope_tile=None,
            guide_step=0,
            guide_threshold=1.0,
            keep_context=True,
            hidden_mode="unconditional",
        )
        self.assertEqual(len(branches["conditional"]), 1)
        self.assertEqual(len(branches["guided"]), 1)
        self.assertEqual(len(branches["unconditional"]), 1)
        self.assertEqual(branches["conditional"][0]["owned"].tolist(), [True, False, False, False])
        self.assertEqual(branches["guided"][0]["owned"].tolist(), [False, True, True, True])
        self.assertFalse(branches["unconditional"][0]["owned"].any())
        self.assertTrue(torch.equal(branches["guided"][0]["global_ids"], torch.arange(4)))


if __name__ == "__main__":
    unittest.main()
