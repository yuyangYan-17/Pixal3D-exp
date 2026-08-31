import torch

from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler


class ConstantVelocity(torch.nn.Module):
    def forward(self, x_t, timestep, cond=None):
        del timestep, cond
        return torch.full_like(x_t, 0.25)


def test_endpoint_callback_observes_current_state_velocity_and_time():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    noise = torch.tensor([[1.0, -2.0]])
    seen = []

    result = sampler.sample(
        ConstantVelocity(),
        noise,
        steps=2,
        verbose=False,
        return_model_history=False,
        endpoint_callback=lambda **payload: seen.append(payload),
    )

    assert len(seen) == 2
    assert seen[0]["step_index"] == 0
    assert seen[0]["t"] == 1.0
    assert seen[0]["t_next"] == 0.5
    assert torch.equal(seen[0]["x_t"], noise)
    for payload in seen:
        expected = sampler._pred_to_xstart(
            payload["x_t"], payload["t"], payload["v"]
        )
        assert torch.equal(payload["endpoint"], expected)

    # Observing endpoints must not perturb the ordinary Euler trajectory.
    reference = sampler.sample(
        ConstantVelocity(),
        noise,
        steps=2,
        verbose=False,
        return_model_history=False,
    )
    assert torch.equal(result.samples, reference.samples)
