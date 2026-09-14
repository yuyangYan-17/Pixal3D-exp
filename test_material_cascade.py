"""Small numerical checks; end-to-end GPU smoke is the entrypoint's --smoke-test."""
import unittest
import tempfile
from pathlib import Path

import torch

from pixal3d_sweep_texture import load_texture_payload, noise_encoded_texture, texture_schedule
from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler


class MaterialCascadeTests(unittest.TestCase):
    def test_material_resume_accepts_v2_and_rejects_old_pure_noise_cache(self):
        with tempfile.TemporaryDirectory(prefix="material_cascade_test_") as tmp:
            path = Path(tmp) / "latent.pt"
            torch.save(dict(features=torch.ones(2, 32), experiment=dict(version=2)), path)
            self.assertEqual(load_texture_payload(path)["features"].shape, (2, 32))
            torch.save(dict(features=torch.ones(2, 32), experiment=dict(version=1)), path)
            with self.assertRaises(RuntimeError):
                load_texture_payload(path)

    def test_all_geometry_start_times_have_twelve_texture_updates(self):
        sampler = FlowEulerSampler(1e-5)
        for start_t in sampler.timestep_schedule(12, 3)[:-1]:
            times = texture_schedule(sampler, start_t, 12, 3)
            self.assertEqual(len(times), 13)
            self.assertEqual(times[0], start_t)
            self.assertEqual(times[-1], 0)
            self.assertTrue(all(t > n for t, n in zip(times, times[1:])))

    def test_material_initialization_survives_partial_noise(self):
        material = torch.tensor([[1., 2.]])
        noise = torch.tensor([[3., 4.]])
        self.assertTrue(torch.equal(noise_encoded_texture(material, noise, 1., 1e-5), noise))
        actual = noise_encoded_texture(material, noise, .25, 1e-5)
        torch.testing.assert_close(actual, .75 * material + (.00001 + .99999 * .25) * noise)
        self.assertFalse(torch.equal(actual, noise_encoded_texture(material * 2, noise, .25, 1e-5)))

    @unittest.skipUnless(torch.cuda.is_available(), "real sparse trilinear query requires CUDA")
    def test_baseline_field_query_uses_world_coordinates_and_trilinear_weights(self):
        from pixal3d.representations.mesh import MeshWithVoxel
        coords = torch.cartesian_prod(*(torch.arange(3, device="cuda") for _ in range(3))).int()
        centers = coords.float() + .5
        attrs = torch.cat([centers / 3, torch.ones_like(centers)], 1)
        field = MeshWithVoxel(torch.zeros((1, 3), device="cuda"), torch.zeros((0, 3), device="cuda", dtype=torch.int),
                              [-.5] * 3, 1 / 1024, coords, attrs, torch.Size([1, 6, 3, 3, 3]))
        query_grid1024 = torch.tensor([[.75, 1.25, 1.5], [1.5, 2., .875]], device="cuda")
        dual4096 = query_grid1024 / 1024
        expected = torch.cat([query_grid1024 / 3, torch.ones_like(query_grid1024)], 1)
        torch.testing.assert_close(field.query_attrs(dual4096 - .5), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
