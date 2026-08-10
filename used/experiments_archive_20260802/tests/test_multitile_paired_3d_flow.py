import types
import unittest

import numpy as np
import torch
from PIL import Image

from pixal3d.modules.sparse import SparseTensor
from pixal3d.modules.sparse.attention.proj_attention import (
    SparseProjectAttention,
)
from pixal3d.pipelines.pixal3d_image_to_3d import (
    Pixal3DImageTo3DPipeline,
)


class FakeRembg:
    def __init__(self):
        self.calls = 0
        self.last_rgb = None

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def __call__(self, rgb):
        self.calls += 1
        self.last_rgb = rgb.copy()
        array = np.asarray(rgb)
        alpha = np.zeros(array.shape[:2], np.uint8)
        alpha[1:-1, 1:-1] = 255
        rgba = np.concatenate([array, alpha[..., None]], axis=-1)
        # Deliberately corrupt proxy RGB: canonical output must use source RGB.
        rgba[..., :3] = 255 - rgba[..., :3]
        return Image.fromarray(rgba, "RGBA")


class FakeCross(torch.nn.Module):
    def forward(self, x, context):
        # Nonlinear in context, so averaging raw globals would fail.
        value = context.float().square().mean(dim=(1, 2))[0]
        return x.replace(x.feats * 0.25 + value.to(x.dtype))


class CanonicalAndLayoutTests(unittest.TestCase):
    def pipeline(self, rembg):
        pipeline = Pixal3DImageTo3DPipeline()
        pipeline.rembg_model = rembg
        pipeline.low_vram = False
        pipeline._device = "cpu"
        return pipeline

    def test_single_preprocessing_alpha_and_rembg(self):
        rembg = FakeRembg()
        pipeline = self.pipeline(rembg)
        rgba = np.zeros((20, 30, 4), np.uint8)
        rgba[..., :3] = (20, 40, 60)
        rgba[3:17, 8:22, 3] = 255
        result = pipeline.preprocess_canonical_images(
            Image.fromarray(rgba, "RGBA")
        )
        self.assertEqual(rembg.calls, 0)
        self.assertEqual(result["image_4096"].size, (4096, 4096))
        self.assertEqual(result["image_1024"].size, (1024, 1024))
        self.assertEqual(result["image_512"].size, (512, 512))
        self.assertTrue(
            np.array_equal(
                np.asarray(result["image_1024"]),
                np.asarray(
                    result["image_4096"].resize(
                        (1024, 1024), Image.Resampling.LANCZOS
                    )
                ),
            )
        )
        self.assertEqual(np.asarray(result["image_4096"])[0, 0].tolist(), [0, 0, 0])

        rgb = Image.new("RGB", (40, 20), (11, 22, 33))
        result = pipeline.preprocess_canonical_images(rgb)
        self.assertEqual(rembg.calls, 1)
        center = np.asarray(result["image_4096"])[2048, 2048]
        self.assertTrue(np.allclose(center, (11, 22, 33), atol=2))

    def test_tile_layout_and_membership(self):
        boxes = Pixal3DImageTo3DPipeline.build_texture_image_tile_layout()
        self.assertEqual(len(boxes), 49)
        self.assertEqual(boxes[0], (0, 0, 1024, 1024))
        self.assertEqual(boxes[-1], (3072, 3072, 4096, 4096))
        self.assertTrue(all(0 <= a < c <= 4096 and 0 <= b < d <= 4096
                            for a, b, c, d in boxes))
        uv = torch.tensor([
            [0.0, 0.0], [4096.0, 4096.0], [512.0, 512.0],
            [768.0, 768.0], [-20.0, 2000.0], [5000.0, 2000.0],
            [1536.0, 1700.0],
        ])
        ids, weights, assignment = (
            Pixal3DImageTo3DPipeline.assign_texture_tiles(uv, boxes)
        )
        counts = (ids >= 0).sum(1)
        self.assertTrue(torch.all((counts >= 1) & (counts <= 4)))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(uv.shape[0])))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all((assignment >= 0) & (assignment <= 4096)))

    def test_global_proj_pairing_uses_the_same_crop(self):
        pipeline = self.pipeline(FakeRembg())
        pipeline.image_cond_model_tex_1024 = types.SimpleNamespace(
            use_naf_upsample=False
        )
        starts = list(range(0, 4096 - 1024 + 1, 512))

        def fake_condition(**kwargs):
            x0, y0, _, _ = kwargs["projection_crop_box"]
            col = starts.index(round(x0 * 4096))
            row = starts.index(round(y0 * 4096))
            layout_id = row * 7 + col
            coords = kwargs["coords"]
            global_value = torch.full((1, 5, 1024), float(layout_id))
            proj_value = torch.full(
                (len(coords), 2048), float(layout_id)
            )
            projection = SparseTensor(proj_value, coords)
            return {
                "cond": {"global": global_value, "proj": projection},
                "neg_cond": {
                    "global": torch.zeros_like(global_value),
                    "proj": projection.replace(torch.zeros_like(proj_value)),
                },
            }

        pipeline.get_proj_cond_shape = fake_condition
        uv_pixels = torch.tensor([
            [768.0, 768.0], [1536.0, 1700.0], [4096.0, 4096.0],
        ])
        coords = torch.tensor([
            [0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9],
        ], dtype=torch.int32)
        condition, summary = pipeline._prepare_multitile_paired_condition(
            image_4096=Image.new("RGB", (4096, 4096)),
            foreground_mask_4096=Image.new("L", (4096, 4096), 255),
            global_coords=coords,
            projected_full_norm=uv_pixels / 4096.0,
            camera_angle_x=0.8,
            distance=2.0,
            mesh_scale=1.0,
            save_slot_proj=True,
        )
        raw_slots = summary["slot_proj"]
        for token_row in range(len(coords)):
            for slot in range(4):
                bank_id = int(condition["tile_ids"][token_row, slot])
                if bank_id < 0:
                    self.assertEqual(raw_slots[token_row, slot].count_nonzero(), 0)
                    continue
                global_marker = condition["global_bank"][bank_id, 0, 0]
                proj_marker = raw_slots[token_row, slot, 0]
                self.assertEqual(global_marker.item(), proj_marker.item())


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.module = SparseProjectAttention(FakeCross(), 6, 8)
        self.coords = torch.tensor(
            [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int32
        )
        self.x = SparseTensor(torch.randn(3, 6), self.coords)
        self.proj = SparseTensor(torch.randn(3, 8), self.coords)

    def test_legacy_shared_global_identity(self):
        global_context = torch.randn(1, 5, 4)
        legacy = self.module(
            self.x, {"global": global_context, "proj": self.proj}
        )
        paired = self.module(
            self.x,
            {
                "mode": "multi_tile_paired",
                "global_bank": global_context,
                "proj": self.proj,
                "tile_ids": torch.zeros(3, 1, dtype=torch.long),
                "tile_weights": torch.ones(3, 1),
            },
        )
        self.assertTrue(torch.allclose(legacy.feats, paired.feats, atol=1e-6))

    def test_duplicate_context_and_grouped_reference(self):
        one = torch.randn(1, 5, 4)
        bank = one.repeat(2, 1, 1)
        ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
        weights = torch.tensor([[0.3, 0.7]]).repeat(3, 1)
        grouped = self.module.multi_tile_global_grouped(
            self.x, bank, ids, weights
        )
        reference = self.module.multi_tile_global_reference(
            self.x, bank, ids, weights
        )
        single = self.module.cross_attn_block(self.x, one)
        self.assertTrue(torch.allclose(grouped.feats, reference.feats, atol=1e-6))
        self.assertTrue(torch.allclose(grouped.feats, single.feats, atol=1e-6))

    def test_float32_weights_with_half_hidden(self):
        x = SparseTensor(self.x.feats.half(), self.coords)
        bank = torch.randn(2, 5, 4).half()
        ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
        weights = torch.tensor(
            [[0.33333334, 0.66666669]], dtype=torch.float32
        ).repeat(3, 1)
        out = self.module.multi_tile_global_grouped(x, bank, ids, weights)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.isfinite(out.feats).all())

    def test_proj_linear_fusion_including_bias(self):
        slots = torch.randn(7, 4, 8)
        weights = torch.rand(7, 4)
        weights /= weights.sum(1, keepdim=True)
        fused = (slots * weights[..., None]).sum(1)
        left = self.module.proj_linear(fused)
        right = (
            self.module.proj_linear(slots) * weights[..., None]
        ).sum(1)
        self.assertTrue(torch.allclose(left, right, atol=1e-6))

    def test_negative_cfg_preserves_memberships(self):
        ids = torch.tensor([[0, -1], [0, -1], [0, -1]])
        weights = torch.tensor([[1.0, 0.0]]).repeat(3, 1)
        positive = {
            "mode": "multi_tile_paired",
            "global_bank": torch.randn(1, 5, 4),
            "proj": self.proj,
            "tile_ids": ids,
            "tile_weights": weights,
        }
        negative = (
            Pixal3DImageTo3DPipeline.make_multitile_negative_condition(positive)
        )
        self.assertEqual(negative["global_bank"].count_nonzero().item(), 0)
        self.assertEqual(negative["proj"].feats.count_nonzero().item(), 0)
        self.assertIs(negative["tile_ids"], ids)
        self.assertIs(negative["tile_weights"], weights)


class PatchTests(unittest.TestCase):
    def test_patch_coverage_alignment_and_merge_order(self):
        xyz = torch.tensor([
            [0, 0, 0], [31, 31, 31], [32, 32, 32], [63, 63, 63],
            [64, 64, 64], [96, 96, 96], [127, 127, 127],
            [20, 70, 110],
        ], dtype=torch.int32)
        coords = torch.cat([torch.zeros(len(xyz), 1, dtype=torch.int32), xyz], 1)
        patches, coverage = Pixal3DImageTo3DPipeline.build_texture_3d_patches(
            coords
        )
        self.assertEqual(len(patches), 27)
        self.assertTrue(torch.all((coverage >= 1) & (coverage <= 8)))
        results = []
        for patch in patches:
            indices = patch["global_indices"]
            local = patch["local_coords"]
            if not len(indices):
                continue
            self.assertTrue(torch.equal(coords[indices, 0], local[:, 0]))
            self.assertGreaterEqual(int(local[:, 1:].min()), 0)
            self.assertLessEqual(int(local[:, 1:].max()), 63)
            velocity = indices.float()[:, None].repeat(1, 2)
            weights = Pixal3DImageTo3DPipeline.texture_3d_patch_weights(local)
            results.append((indices, velocity, weights))
        merged, merged_coverage = (
            Pixal3DImageTo3DPipeline.merge_texture_3d_patch_velocities(
                len(coords), results, 2, torch.device("cpu")
            )
        )
        reverse, _ = Pixal3DImageTo3DPipeline.merge_texture_3d_patch_velocities(
            len(coords), list(reversed(results)), 2, torch.device("cpu")
        )
        expected = torch.arange(len(coords)).float()[:, None].repeat(1, 2)
        self.assertTrue(torch.allclose(merged, expected, atol=1e-6))
        self.assertTrue(torch.allclose(reverse, expected, atol=1e-6))
        self.assertTrue(torch.equal(merged_coverage, coverage))
        with self.assertRaises(RuntimeError):
            Pixal3DImageTo3DPipeline.merge_texture_3d_patch_velocities(
                len(coords) + 1, results, 2, torch.device("cpu")
            )

    def test_start_step_12_identity(self):
        pipeline = Pixal3DImageTo3DPipeline()
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        noise = SparseTensor(torch.randn(1, 2), coords)
        final = torch.randn(1, 2)
        trajectory = types.SimpleNamespace(
            states=[torch.zeros_like(final) for _ in range(12)] + [final],
            velocities=[torch.zeros_like(final) for _ in range(12)],
            times=list(np.linspace(1, 0, 13)),
            time_intervals=[1 / 12] * 12,
        )
        output, trace, diagnostics = (
            pipeline._run_multitile_3d_patch_texture_flow(
                flow_model=None,
                sampler=None,
                global_noise=noise,
                shape_concat_cond=noise,
                sampler_params={},
                global_flow=types.SimpleNamespace(trajectory=trajectory),
                condition={},
                start_step=12,
            )
        )
        self.assertTrue(torch.equal(output.feats, final))
        self.assertEqual(trace["steps"], [])
        self.assertEqual(diagnostics["patch_flow_calls"], 0)


if __name__ == "__main__":
    unittest.main()
