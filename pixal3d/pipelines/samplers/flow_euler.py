from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @staticmethod
    def timestep_schedule(steps: int, rescale_t: float = 1.0) -> List[float]:
        """Return the exact nonlinear time schedule used by Euler sampling."""
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        return [float(t) for t in t_seq]

    @staticmethod
    def _snapshot_features(value, device: Union[str, torch.device]):
        """Detach one dense/sparse prediction as a plain feature tensor."""
        features = value.feats if hasattr(value, "feats") else value
        return features.detach().to(device=device, copy=True)

    @staticmethod
    def invert_euler_trajectory(
        final_sample,
        velocities: Sequence[torch.Tensor],
        time_intervals: Sequence[float],
        target_step: int,
    ):
        """Algebraically undo saved Euler updates back to ``target_step``.

        Forward sampling uses ``x_next = x - (t - t_next) * velocity``.
        Consequently no model call (and no DDIM inversion) is needed here.
        ``target_step=0`` reconstructs the initial noise and
        ``target_step=len(velocities)`` returns the final sample.
        """
        if len(velocities) != len(time_intervals):
            raise ValueError(
                "velocities and time_intervals must have identical lengths"
            )
        if not 0 <= target_step <= len(velocities):
            raise ValueError(
                f"target_step must be in [0, {len(velocities)}], "
                f"got {target_step}"
            )

        is_sparse = hasattr(final_sample, "feats")
        current = (
            final_sample.feats.detach().clone()
            if is_sparse
            else final_sample.detach().clone()
        )
        for step_index in range(len(velocities) - 1, target_step - 1, -1):
            velocity = velocities[step_index].to(
                device=current.device,
                dtype=current.dtype,
            )
            current = current + float(time_intervals[step_index]) * velocity
        return final_sample.replace(current) if is_sparse else current

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({
            "pred_x_prev": pred_x_prev,
            "pred_x_0": pred_x_0,
            "pred_v": pred_v,
            "time_interval": float(t - t_prev),
        })

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        record_trajectory: bool = False,
        trajectory_device: Union[str, torch.device] = "cpu",
        return_model_history: bool = True,
        endpoint_callback: Optional[Callable[..., None]] = None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = self.timestep_schedule(steps, rescale_t)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({
            "samples": None,
            "pred_x_t": [],
            "pred_x_0": [],
            "trajectory": None,
        })
        if record_trajectory:
            ret.trajectory = edict({
                "times": list(t_seq),
                "time_intervals": [
                    float(t - t_prev) for t, t_prev in t_pairs
                ],
                "states": [
                    self._snapshot_features(sample, trajectory_device)
                ],
                "velocities": [],
            })
        for step_index, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc=tqdm_desc, disable=not verbose)
        ):
            # Keep this reference before the Euler update.  A predicted
            # endpoint is defined by the model velocity at the *current*
            # state/time, not by pred_x_prev.
            current_sample = sample
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            if endpoint_callback is not None:
                endpoint_callback(
                    step_index=step_index,
                    x_t=current_sample,
                    v=out.pred_v,
                    t=float(t),
                    t_next=float(t_prev),
                    endpoint=out.pred_x_0,
                )
            sample = out.pred_x_prev
            if return_model_history:
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
            if record_trajectory:
                ret.trajectory.velocities.append(
                    self._snapshot_features(out.pred_v, trajectory_device)
                )
                ret.trajectory.states.append(
                    self._snapshot_features(sample, trajectory_device)
                )
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)
