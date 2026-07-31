#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free online canonical-posterior fusion for Pixal3D tiles.

The script keeps the global-to-local encoder support, local FDG decode,
local-to-global mapping, triangle ownership, vertex welding, PBR sampling, and
rendering route of ``pixal3d_tile_encoded_query_noise_flow_overlap_render.py``.
Only the normalized shape/texture SLat trajectories are changed.

For every Euler step all tiles are first evaluated at their current state with
matched LR/HR image conditions.  A regularized CCA is then estimated from the
current object's pooled sufficient statistics and the canonical posterior
correction is applied to every tile before any tile advances to the next step.
No fitted map is loaded from disk and no gradient update is performed.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import numpy as np
import torch
from PIL import Image
from scipy import sparse as scipy_sparse
from scipy.sparse import csgraph
from tqdm import tqdm

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_tile_encoded_query_noise_flow_overlap_render as baseline
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT_VERSION = "pixal3d_online_canonical_posterior_shape_texture_v1"
MODE_COUNTS = (1, 2, 4, 8, 16, 24, 32)
FUSION_MODES = (
    "local",
    "old_anchor",
    "shape_posterior",
    "texture_posterior",
    "joint_posterior",
)
TEXTURE_EVIDENCE_MODES = (
    "texture",
    "texture_global_shape",
    "texture_fused_shape",
)


@dataclass
class CCAFit:
    mean_x: torch.Tensor
    mean_y: torch.Tensor
    weight_x: torch.Tensor
    weight_y: torch.Tensor
    rho: torch.Tensor
    diagnostics: Dict[str, Any]


@dataclass
class TileRuntime:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: core.TileCameraTransform
    coords_cpu: torch.Tensor
    global_shape_norm_cpu: torch.Tensor
    global_texture_norm_cpu: torch.Tensor
    source_stats: Dict[str, Any]
    shape_final_norm_cpu: Optional[torch.Tensor] = None
    texture_final_norm_cpu: Optional[torch.Tensor] = None
    shape_trace: List[Dict[str, Any]] = field(default_factory=list)
    texture_trace: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return int(self.coords_cpu.shape[0])


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            _json_value(dict(payload)),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rms(value: torch.Tensor) -> float:
    return float(value.double().square().mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.double().reshape(-1)
    right64 = right.double().reshape(-1)
    denominator = (
        torch.linalg.vector_norm(left64)
        * torch.linalg.vector_norm(right64)
    )
    if float(denominator.item()) <= 1e-24:
        return 1.0 if bool((left64 == right64).all().item()) else 0.0
    return float(((left64 @ right64) / denominator).item())


def _condition_number(matrix: torch.Tensor) -> Optional[float]:
    singular = torch.linalg.svdvals(matrix)
    if singular.numel() == 0:
        return None
    minimum = float(singular[-1].item())
    maximum = float(singular[0].item())
    if minimum <= 0.0:
        return None
    return maximum / minimum


def _symmetric_inverse_sqrt(
    covariance_regularized: torch.Tensor,
) -> torch.Tensor:
    symmetric = 0.5 * (
        covariance_regularized + covariance_regularized.transpose(0, 1)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    numerical_floor = torch.finfo(torch.float64).eps * max(
        1.0, float(eigenvalues.abs().max().item())
    )
    eigenvalues = eigenvalues.clamp_min(numerical_floor)
    inverse_sqrt = (
        eigenvectors * eigenvalues.rsqrt().reshape(1, -1)
    ) @ eigenvectors.transpose(0, 1)
    assert torch.isfinite(inverse_sqrt).all()
    return inverse_sqrt


def _weighted_moments(
    x_tiles: Sequence[torch.Tensor],
    y_tiles: Sequence[torch.Tensor],
    *,
    weighting: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ML means/covariances in float64 without concatenating tokens."""
    if not x_tiles or len(x_tiles) != len(y_tiles):
        raise ValueError("CCA requires equal non-empty tile lists")
    x64 = [value.detach().to(device="cpu", dtype=torch.float64) for value in x_tiles]
    y64 = [value.detach().to(device="cpu", dtype=torch.float64) for value in y_tiles]
    for left, right in zip(x64, y64):
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[0] != right.shape[0]
            or left.shape[0] == 0
        ):
            raise ValueError("CCA tile tensors must be paired non-empty [N,C]")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("CCA input contains non-finite values")
    if weighting == "token":
        count = float(sum(value.shape[0] for value in x64))
        mean_x = sum((value.sum(dim=0) for value in x64)) / count
        mean_y = sum((value.sum(dim=0) for value in y64)) / count
        covariance_x = sum(
            (value - mean_x).transpose(0, 1) @ (value - mean_x)
            for value in x64
        ) / count
        covariance_y = sum(
            (value - mean_y).transpose(0, 1) @ (value - mean_y)
            for value in y64
        ) / count
        covariance_xy = sum(
            (left - mean_x).transpose(0, 1) @ (right - mean_y)
            for left, right in zip(x64, y64)
        ) / count
    elif weighting == "tile":
        tile_count = float(len(x64))
        mean_x = sum(value.mean(dim=0) for value in x64) / tile_count
        mean_y = sum(value.mean(dim=0) for value in y64) / tile_count
        covariance_x = sum(
            (value - mean_x).transpose(0, 1) @ (value - mean_x)
            / float(value.shape[0])
            for value in x64
        ) / tile_count
        covariance_y = sum(
            (value - mean_y).transpose(0, 1) @ (value - mean_y)
            / float(value.shape[0])
            for value in y64
        ) / tile_count
        covariance_xy = sum(
            (left - mean_x).transpose(0, 1) @ (right - mean_y)
            / float(left.shape[0])
            for left, right in zip(x64, y64)
        ) / tile_count
    else:
        raise ValueError(f"unsupported CCA weighting {weighting}")
    for value in (mean_x, mean_y, covariance_x, covariance_y, covariance_xy):
        assert value.dtype == torch.float64
        assert torch.isfinite(value).all()
    return mean_x, mean_y, covariance_x, covariance_y, covariance_xy


def _fit_regularized_cca(
    x_tiles: Sequence[torch.Tensor],
    y_tiles: Sequence[torch.Tensor],
    *,
    regularization: float,
    weighting: str,
) -> CCAFit:
    """Fit square or rectangular regularized CCA in float64."""
    (
        mean_x,
        mean_y,
        covariance_x,
        covariance_y,
        covariance_xy,
    ) = _weighted_moments(x_tiles, y_tiles, weighting=weighting)
    x_channels = int(covariance_x.shape[0])
    y_channels = int(covariance_y.shape[0])
    identity_x = torch.eye(x_channels, dtype=torch.float64)
    identity_y = torch.eye(y_channels, dtype=torch.float64)
    scale_x = float(torch.trace(covariance_x).item()) / float(x_channels)
    scale_y = float(torch.trace(covariance_y).item()) / float(y_channels)
    regularizer_x = float(regularization) * scale_x
    regularizer_y = float(regularization) * scale_y
    # Degenerate synthetic inputs can have exactly zero covariance.  The
    # machine-epsilon floor is solely for the eigensolver and never becomes a
    # fusion coefficient; zero cross-covariance still yields rho == 0.
    floor = torch.finfo(torch.float64).eps
    covariance_x_regularized = covariance_x + max(
        regularizer_x, floor
    ) * identity_x
    covariance_y_regularized = covariance_y + max(
        regularizer_y, floor
    ) * identity_y
    assert torch.isfinite(covariance_x_regularized).all()
    assert torch.isfinite(covariance_y_regularized).all()
    inverse_sqrt_x = _symmetric_inverse_sqrt(covariance_x_regularized)
    inverse_sqrt_y = _symmetric_inverse_sqrt(covariance_y_regularized)
    whitened_cross = (
        inverse_sqrt_x @ covariance_xy @ inverse_sqrt_y
    )
    left, rho, right_transpose = torch.linalg.svd(
        whitened_cross, full_matrices=False
    )
    right = right_transpose.transpose(0, 1)
    rho = rho.clamp(0.0, 1.0)
    weight_x = inverse_sqrt_x @ left
    weight_y = inverse_sqrt_y @ right
    assert weight_y.shape == (y_channels, y_channels)
    assert torch.isfinite(rho).all()
    diagnostics = {
        "tokens": int(sum(value.shape[0] for value in x_tiles)),
        "tiles": int(len(x_tiles)),
        "weighting": weighting,
        "regularization_epsilon": float(regularization),
        "x_channels": x_channels,
        "y_channels": y_channels,
        "covariance_x_condition": _condition_number(covariance_x),
        "covariance_y_condition": _condition_number(covariance_y),
        "covariance_x_regularized_condition": _condition_number(
            covariance_x_regularized
        ),
        "covariance_y_regularized_condition": _condition_number(
            covariance_y_regularized
        ),
        "regularizer_x": regularizer_x,
        "regularizer_y": regularizer_y,
        "canonical_correlations": rho.tolist(),
        "rho_squared_sum": float(rho.square().sum().item()),
    }
    return CCAFit(
        mean_x=mean_x,
        mean_y=mean_y,
        weight_x=weight_x,
        weight_y=weight_y,
        rho=rho,
        diagnostics=diagnostics,
    )


def _posterior_clean_prediction(
    *,
    predictor: torch.Tensor,
    local_reference: torch.Tensor,
    local_hr: torch.Tensor,
    fit: CCAFit,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply h* = h + rho*g - rho^2*r without an explicit matrix inverse."""
    predictor64 = predictor.detach().to(device="cpu", dtype=torch.float64)
    reference64 = local_reference.detach().to(device="cpu", dtype=torch.float64)
    hr64 = local_hr.detach().to(device="cpu", dtype=torch.float64)
    if (
        predictor64.shape[0] != reference64.shape[0]
        or reference64.shape != hr64.shape
    ):
        raise ValueError("posterior tensors do not share token support")
    canonical_predictor = (
        predictor64 - fit.mean_x
    ) @ fit.weight_x
    canonical_reference = (
        reference64 - fit.mean_y
    ) @ fit.weight_y
    delta_canonical = (
        fit.rho * canonical_predictor
        - fit.rho.square() * canonical_reference
    )
    delta_local = torch.linalg.solve(
        fit.weight_y.transpose(0, 1),
        delta_canonical.transpose(0, 1),
    ).transpose(0, 1)
    z0_star = hr64 + delta_local
    canonical_roundtrip_error = float(
        (
            delta_local @ fit.weight_y - delta_canonical
        ).abs().max().item()
    )
    assert canonical_roundtrip_error < 1e-4
    assert torch.isfinite(z0_star).all()
    return z0_star.to(torch.float32), {
        "canonical_back_transform_max_error": canonical_roundtrip_error,
        "posterior_correction_rms": _rms(delta_local),
        "local_hr_clean_prediction_rms": _rms(hr64),
        "correction_over_hr_norm": float(
            torch.linalg.vector_norm(delta_local).item()
            / max(1e-12, float(torch.linalg.vector_norm(hr64).item()))
        ),
        "correction_hr_residual_cosine": _cosine(
            delta_local, hr64 - reference64
        ),
    }


def _principal_angle_diagnostics(
    previous: Optional[CCAFit],
    current: Optional[CCAFit],
) -> Dict[str, Any]:
    if previous is None or current is None:
        return {"available": False}
    output: Dict[str, Any] = {"available": True, "degrees_by_mode_count": {}}
    maximum = min(
        previous.weight_y.shape[1], current.weight_y.shape[1]
    )
    for count in MODE_COUNTS:
        modes = min(int(count), maximum)
        previous_q = torch.linalg.qr(
            previous.weight_y[:, :modes], mode="reduced"
        ).Q
        current_q = torch.linalg.qr(
            current.weight_y[:, :modes], mode="reduced"
        ).Q
        singular = torch.linalg.svdvals(
            previous_q.transpose(0, 1) @ current_q
        ).clamp(0.0, 1.0)
        angles = torch.rad2deg(torch.acos(singular))
        output["degrees_by_mode_count"][str(count)] = {
            "mean": float(angles.mean().item()),
            "max": float(angles.max().item()),
            "all": angles.tolist(),
        }
    return output


def _synthetic_tests() -> Dict[str, Any]:
    """CPU-only mathematical assertions required by the experiment spec."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260730)
    tiles_x = [
        torch.randn((73, 32), generator=generator),
        torch.randn((41, 32), generator=generator),
    ]
    mapping = torch.randn((32, 32), generator=generator) / math.sqrt(32.0)
    tiles_y = [
        value @ mapping + 0.1 * torch.randn(value.shape, generator=generator)
        for value in tiles_x
    ]
    square = _fit_regularized_cca(
        tiles_x, tiles_y, regularization=1e-4, weighting="token"
    )
    assert square.weight_x.shape == (32, 32)
    assert square.weight_y.shape == (32, 32)
    rectangular_x = [
        torch.randn((value.shape[0], 64), generator=generator)
        for value in tiles_x
    ]
    rectangular = _fit_regularized_cca(
        rectangular_x,
        tiles_y,
        regularization=1e-4,
        weighting="tile",
    )
    assert rectangular.weight_x.shape == (64, 32)
    assert rectangular.weight_y.shape == (32, 32)

    n = 37
    predictor = torch.randn((n, 32), generator=generator)
    reference = torch.randn((n, 32), generator=generator)
    hr = torch.randn((n, 32), generator=generator)
    identity_weight = torch.eye(32, dtype=torch.float64)
    base_fit = CCAFit(
        mean_x=torch.zeros(32, dtype=torch.float64),
        mean_y=torch.zeros(32, dtype=torch.float64),
        weight_x=identity_weight,
        weight_y=identity_weight,
        rho=torch.zeros(32, dtype=torch.float64),
        diagnostics={},
    )
    zero_result, _ = _posterior_clean_prediction(
        predictor=predictor,
        local_reference=reference,
        local_hr=hr,
        fit=base_fit,
    )
    assert torch.equal(zero_result, hr), "rho=0 must return ordinary HR"
    unit_fit = CCAFit(
        mean_x=base_fit.mean_x,
        mean_y=base_fit.mean_y,
        weight_x=identity_weight,
        weight_y=identity_weight,
        rho=torch.ones(32, dtype=torch.float64),
        diagnostics={},
    )
    unit_result, _ = _posterior_clean_prediction(
        predictor=predictor,
        local_reference=reference,
        local_hr=hr,
        fit=unit_fit,
    )
    assert torch.allclose(
        unit_result, predictor + (hr - reference), atol=1e-6, rtol=0.0
    ), "rho=1 limit differs from g+(h-r)"
    equal_result, _ = _posterior_clean_prediction(
        predictor=reference,
        local_reference=reference,
        local_hr=hr,
        fit=unit_fit,
    )
    assert torch.allclose(equal_result, hr, atol=1e-6, rtol=0.0)

    independent_x = [torch.eye(32), -torch.eye(32)]
    zero_y_base = torch.eye(32)
    zero_y = [zero_y_base, zero_y_base]
    zero_cross_fit = _fit_regularized_cca(
        independent_x,
        zero_y,
        regularization=1e-4,
        weighting="token",
    )
    assert float(zero_cross_fit.rho.abs().max().item()) < 1e-10
    zero_cross_result, _ = _posterior_clean_prediction(
        predictor=independent_x[0],
        local_reference=zero_y[0],
        local_hr=zero_y[0] + 0.5,
        fit=zero_cross_fit,
    )
    assert torch.equal(zero_cross_result, zero_y[0] + 0.5)

    # Equal LR/HR conditions imply H-R == 0.  A nonzero correction is then
    # permitted only through the global shared canonical evidence.
    equal_condition_result, _ = _posterior_clean_prediction(
        predictor=predictor,
        local_reference=reference,
        local_hr=reference,
        fit=square,
    )
    assert torch.isfinite(equal_condition_result).all()
    return {
        "status": "passed",
        "square_cca_shape": [32, 32],
        "rectangular_cca_shapes": {
            "weight_x": [64, 32],
            "weight_y": [32, 32],
        },
        "limits": ["rho=0", "rho=1", "g=r", "zero-cross-covariance"],
        "equal_lr_hr_condition": "finite global-shared correction only",
    }


def _normalization_roundtrip(
    value_raw: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
    *,
    label: str,
) -> Tuple[SparseTensor, float]:
    value_norm = baseline._normalize(value_raw, normalization)
    reconstructed_raw = baseline._denormalize(value_norm, normalization)
    error = float(
        (
            reconstructed_raw.feats.float() - value_raw.feats.float()
        ).abs().max().item()
    )
    if error >= 1e-5:
        raise AssertionError(f"{label}: normalization roundtrip error {error}")
    return value_norm, error


def _sampler_params(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return (
        {
            "steps": int(args.ss_steps),
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        {
            "steps": int(args.shape_steps),
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        {
            "steps": int(args.texture_steps),
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        },
    )


def _load_cached_global_mesh(path: Path) -> Any:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(saved, Mapping) and "mesh" in saved:
        saved = saved["mesh"]
    return core._validate_mesh(saved, "cached global Pixal3D mesh")


def _prepare_global_source(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    output_dir: Path,
    source_cache_dir: Path,
) -> Tuple[
    Image.Image,
    Image.Image,
    Mapping[str, float],
    Any,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    """Create/reuse only global object inputs; never load a fitted latent map."""
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    canonical["image_512"].save(output_dir / "canonical_512.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    camera_path = source_cache_dir / "global_camera.json"
    if camera_path.is_file():
        global_camera = json.loads(camera_path.read_text("utf-8"))
    else:
        global_camera = core._estimate_camera(
            image_1024=image_1024,
            output_dir=source_cache_dir,
            manual_fov=float(args.fov),
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            moge_model_path=args.moge_model_path,
        )
        _atomic_json(camera_path, global_camera)
    _atomic_json(output_dir / "global_camera.json", global_camera)

    mesh_cache_path = source_cache_dir / "global_baseline_mesh.pt"
    global_generation_seconds = 0.0
    if mesh_cache_path.is_file():
        baseline_mesh = _load_cached_global_mesh(mesh_cache_path)
        global_source = "experiment source cache"
        print(f"[global] reused {mesh_cache_path}")
    elif args.global_mesh_cache is not None:
        supplied = Path(args.global_mesh_cache).expanduser().resolve()
        baseline_mesh = _load_cached_global_mesh(supplied)
        _atomic_torch_save(
            mesh_cache_path,
            {
                "format": "online_posterior_global_mesh_cache_v1",
                "mesh": baseline_mesh,
                "global_seed": int(args.global_seed),
                "source_path": str(supplied),
            },
        )
        global_source = f"supplied global mesh cache: {supplied}"
        print(f"[global] imported {supplied}")
    else:
        ss_params, shape_params, texture_params = _sampler_params(args)
        _seed_everything(int(args.global_seed))
        started = time.perf_counter()
        output, latents = pipeline.run(
            image_1024,
            camera_params=global_camera,
            seed=int(args.global_seed),
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=texture_params,
            preprocess_image=False,
            return_latent=True,
            pipeline_type="1024_cascade",
            max_num_tokens=int(args.max_num_tokens),
        )
        global_generation_seconds = time.perf_counter() - started
        if len(output) != 1 or int(latents[2]) != core.OVOXEL_RESOLUTION:
            raise RuntimeError("global Pixal3D route returned an invalid mesh")
        baseline_mesh = core._validate_mesh(
            output[0], "ordinary global Pixal3D 1024"
        ).to("cpu")
        _atomic_torch_save(
            mesh_cache_path,
            {
                "format": "online_posterior_global_mesh_cache_v1",
                "mesh": baseline_mesh,
                "global_seed": int(args.global_seed),
            },
        )
        del output, latents
        _empty_cuda_cache()
        global_source = "generated by this experiment"

    global_ovoxel_object = core._ovoxel_indices_to_object(
        baseline_mesh.coords,
        origin=baseline_mesh.origin,
        voxel_size=baseline_mesh.voxel_size,
    )
    global_ovoxel_q = global_ovoxel_object * (
        2.0 * float(global_camera["mesh_scale"])
    )
    global_ovoxel_uv, _, global_ovoxel_finite = (
        core._project_global_q_to_4096(
            global_ovoxel_q, global_camera=global_camera
        )
    )
    global_face_uv, global_face_finite = core._project_face_centers(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    return (
        image_4096,
        image_1024,
        global_camera,
        baseline_mesh,
        global_ovoxel_q,
        global_ovoxel_uv,
        global_ovoxel_finite,
        global_face_uv,
        global_face_finite,
        {
            "source": global_source,
            "generation_seconds": float(global_generation_seconds),
            "cache_path": str(mesh_cache_path),
            "seed": int(args.global_seed),
        },
    )


def _load_or_create_tile_anchor(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    device: torch.device,
    source_cache_dir: Path,
    image_4096: Image.Image,
    image_1024: Image.Image,
    global_camera: Mapping[str, float],
    baseline_mesh: Any,
    global_ovoxel_q: torch.Tensor,
    global_ovoxel_uv: torch.Tensor,
    global_ovoxel_finite: torch.Tensor,
    global_face_uv: torch.Tensor,
    global_face_finite: torch.Tensor,
    shape_encoder: torch.nn.Module,
    pbr_encoder: torch.nn.Module,
    tile_id: int,
    box: Sequence[int],
) -> Optional[TileRuntime]:
    projected_count = int(
        core._inside_tile(
            global_ovoxel_uv, global_ovoxel_finite, box
        ).sum().item()
    )
    tile_cache_dir = source_cache_dir / "per_tile" / f"tile_{tile_id:02d}"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = tile_cache_dir / "global_anchor_only.pt"
    transform = core._derive_tile_camera(
        tile_id=tile_id,
        box=box,
        global_camera=global_camera,
        extend_pixel=int(args.extend_pixel),
    )
    if projected_count < int(args.min_tile_ovoxels):
        _atomic_json(
            tile_cache_dir / "source_summary.json",
            {
                "status": "skipped",
                "tile_id": tile_id,
                "box": list(box),
                "projected_global_ovoxels": projected_count,
                "reason": "projected global O-Voxels below source threshold",
            },
        )
        return None

    tile_hr = image_4096.crop(tuple(box)).convert("RGB")
    tile_lr = baseline._make_lr_reference_tile(image_1024, box)
    tile_hr.save(tile_cache_dir / "tile_reference_hr.png")
    tile_lr.save(tile_cache_dir / "tile_reference_lr_from_global_1024.png")
    if anchor_path.is_file():
        cached = torch.load(anchor_path, map_location="cpu", weights_only=False)
        if cached.get("format") != "global_anchor_only_v1":
            raise RuntimeError(f"unexpected anchor cache format in {anchor_path}")
        coords_cpu = cached["coords"].to(torch.int32).contiguous()
        global_shape_norm_cpu = cached["global_shape_norm"].float()
        global_texture_norm_cpu = cached["global_texture_norm"].float()
        source_stats = dict(cached["source_stats"])
    else:
        mapping = core._map_global_ovoxels_to_local(
            global_mesh=baseline_mesh,
            global_q=global_ovoxel_q,
            global_uv_4096=global_ovoxel_uv,
            finite_projection=global_ovoxel_finite,
            global_camera=global_camera,
            transform=transform,
        )
        local_vertices, local_faces, _, _, geometry_stats = (
            core._prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_uv=global_face_uv,
                global_face_finite=global_face_finite,
                global_camera=global_camera,
                transform=transform,
            )
        )
        global_shape_raw, shape_encoder_stats = core._encode_local_shape(
            encoder=shape_encoder,
            vertices=local_vertices,
            faces=local_faces,
            device=device,
            low_vram=bool(args.low_vram),
        )
        global_texture_raw, texture_encoder_stats = core._encode_local_pbr(
            encoder=pbr_encoder,
            coords=mapping.local_coords,
            attrs=mapping.local_attrs,
            device=device,
            low_vram=bool(args.low_vram),
        )
        global_shape_raw, global_texture_raw, alignment_stats = (
            core._align_latent_supports(
                global_shape_raw, global_texture_raw
            )
        )
        if not torch.equal(
            global_shape_raw.coords, global_texture_raw.coords
        ):
            raise AssertionError("shape/texture anchor coordinates differ")
        global_shape_norm, shape_roundtrip = _normalization_roundtrip(
            global_shape_raw,
            pipeline.shape_slat_normalization,
            label=f"tile {tile_id:02d} shape",
        )
        global_texture_norm, texture_roundtrip = _normalization_roundtrip(
            global_texture_raw,
            pipeline.tex_slat_normalization,
            label=f"tile {tile_id:02d} texture",
        )
        coords_cpu = global_shape_norm.coords.detach().cpu().to(torch.int32)
        global_shape_norm_cpu = global_shape_norm.feats.detach().cpu().float()
        global_texture_norm_cpu = (
            global_texture_norm.feats.detach().cpu().float()
        )
        source_stats = {
            "tile_id": int(tile_id),
            "box": [int(value) for value in box],
            "projected_global_ovoxels": projected_count,
            "common_tokens": int(coords_cpu.shape[0]),
            "mapping": mapping.stats,
            "geometry_encoder_input": geometry_stats,
            "shape_encoder": shape_encoder_stats,
            "texture_encoder": texture_encoder_stats,
            "alignment": alignment_stats,
            "shape_normalization_roundtrip_max_error": shape_roundtrip,
            "texture_normalization_roundtrip_max_error": texture_roundtrip,
            "cache_contains_local_flow_endpoint": False,
            "cache_contains_fitted_cca_or_ridge": False,
        }
        _atomic_torch_save(
            anchor_path,
            {
                "format": "global_anchor_only_v1",
                "coords": coords_cpu,
                "global_shape_norm": global_shape_norm_cpu,
                "global_texture_norm": global_texture_norm_cpu,
                "source_stats": source_stats,
            },
        )
        del mapping, global_shape_raw, global_texture_raw
        _empty_cuda_cache()
    if coords_cpu.ndim != 2 or coords_cpu.shape[1] != 4:
        raise RuntimeError("cached C64 coordinates have invalid shape")
    if bool((coords_cpu[:, 0] != 0).any().item()):
        raise RuntimeError("cached C64 coordinates do not use batch zero")
    if int(torch.unique(coords_cpu, dim=0).shape[0]) != int(
        coords_cpu.shape[0]
    ):
        raise RuntimeError("cached C64 support contains duplicate coordinates")
    if (
        global_shape_norm_cpu.shape != global_texture_norm_cpu.shape
        or global_shape_norm_cpu.shape[0] != coords_cpu.shape[0]
    ):
        raise RuntimeError("cached shape/texture anchors are not aligned")
    return TileRuntime(
        tile_id=int(tile_id),
        box=tuple(int(value) for value in box),
        transform=transform,
        coords_cpu=coords_cpu,
        global_shape_norm_cpu=global_shape_norm_cpu,
        global_texture_norm_cpu=global_texture_norm_cpu,
        source_stats=source_stats,
    )


def _prepare_tiles(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    device: torch.device,
    source_cache_dir: Path,
    image_4096: Image.Image,
    image_1024: Image.Image,
    global_camera: Mapping[str, float],
    baseline_mesh: Any,
    global_ovoxel_q: torch.Tensor,
    global_ovoxel_uv: torch.Tensor,
    global_ovoxel_finite: torch.Tensor,
    global_face_uv: torch.Tensor,
    global_face_finite: torch.Tensor,
) -> Tuple[List[TileRuntime], List[Dict[str, Any]]]:
    shape_encoder = pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()
    pbr_encoder = pixal3d_models.from_pretrained(
        str(Path(args.pbr_encoder).expanduser())
    ).eval()
    if not bool(args.low_vram):
        shape_encoder.to(device)
        pbr_encoder.to(device)
    requested = core._parse_tile_ids(args.tile_ids)
    boxes = core._tile_layout()
    tiles: List[TileRuntime] = []
    records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if requested is not None and tile_id not in requested:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        try:
            tile = _load_or_create_tile_anchor(
                args=args,
                pipeline=pipeline,
                device=device,
                source_cache_dir=source_cache_dir,
                image_4096=image_4096,
                image_1024=image_1024,
                global_camera=global_camera,
                baseline_mesh=baseline_mesh,
                global_ovoxel_q=global_ovoxel_q,
                global_ovoxel_uv=global_ovoxel_uv,
                global_ovoxel_finite=global_ovoxel_finite,
                global_face_uv=global_face_uv,
                global_face_finite=global_face_finite,
                shape_encoder=shape_encoder,
                pbr_encoder=pbr_encoder,
                tile_id=tile_id,
                box=box,
            )
            if tile is None:
                records.append(
                    {
                        "tile_id": tile_id,
                        "status": "skipped",
                        "reason": "projected global O-Voxels below threshold",
                    }
                )
            else:
                tiles.append(tile)
                records.append(
                    {
                        "tile_id": tile_id,
                        "status": "success",
                        "tokens": tile.tokens,
                    }
                )
                print(
                    f"[source tile {tile_id:02d}] tokens={tile.tokens:,}"
                )
        except Exception as error:
            records.append(
                {
                    "tile_id": tile_id,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[source tile {tile_id:02d}] FAILED: {error}")
    del shape_encoder, pbr_encoder
    _empty_cuda_cache()
    if not tiles:
        raise RuntimeError("no valid tile anchors")
    return tiles, records


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, SparseTensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    return value


def _make_condition(
    *,
    pipeline: Any,
    image_model: Any,
    image: Image.Image,
    coords_cpu: torch.Tensor,
    transform: core.TileCameraTransform,
) -> Mapping[str, Any]:
    coords = coords_cpu.to(device=pipeline.device, dtype=torch.int32)
    result = pipeline.get_proj_cond_shape(
        image_model,
        [image.convert("RGB")],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=64,
    )
    return _move_nested(result, torch.device("cpu"))


def _prepare_conditions(
    *,
    pipeline: Any,
    tiles: Sequence[TileRuntime],
    source_cache_dir: Path,
    latent_name: str,
) -> Dict[int, Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if latent_name == "shape":
        image_model = pipeline.image_cond_model_shape_1024
    elif latent_name == "texture":
        image_model = pipeline.image_cond_model_tex_1024
    else:
        raise ValueError(latent_name)
    output: Dict[int, Tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for tile in tqdm(tiles, desc=f"prepare {latent_name} conditions"):
        tile_cache_dir = (
            source_cache_dir / "per_tile" / f"tile_{tile.tile_id:02d}"
        )
        condition_path = tile_cache_dir / f"{latent_name}_conditions.pt"
        if condition_path.is_file():
            saved = torch.load(
                condition_path, map_location="cpu", weights_only=False
            )
            if saved.get("format") != "matched_lr_hr_condition_v1":
                raise RuntimeError(
                    f"unexpected condition cache {condition_path}"
                )
            if not torch.equal(saved["coords"], tile.coords_cpu):
                raise RuntimeError(
                    f"tile {tile.tile_id}: cached condition support differs"
                )
            condition_lr = saved["lr"]
            condition_hr = saved["hr"]
        else:
            image_hr = Image.open(
                tile_cache_dir / "tile_reference_hr.png"
            ).convert("RGB")
            image_lr = Image.open(
                tile_cache_dir / "tile_reference_lr_from_global_1024.png"
            ).convert("RGB")
            condition_hr = _make_condition(
                pipeline=pipeline,
                image_model=image_model,
                image=image_hr,
                coords_cpu=tile.coords_cpu,
                transform=tile.transform,
            )
            condition_lr = _make_condition(
                pipeline=pipeline,
                image_model=image_model,
                image=image_lr,
                coords_cpu=tile.coords_cpu,
                transform=tile.transform,
            )
            _atomic_torch_save(
                condition_path,
                {
                    "format": "matched_lr_hr_condition_v1",
                    "latent": latent_name,
                    "coords": tile.coords_cpu,
                    "lr": condition_lr,
                    "hr": condition_hr,
                },
            )
        output[tile.tile_id] = (condition_lr, condition_hr)
    _empty_cuda_cache()
    return output


def _call_parameters(
    sampler: Any,
    params: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    call_params = dict(params)
    steps = int(call_params.pop("steps"))
    rescale_t = float(call_params.pop("rescale_t", 1.0))
    call_params.pop("verbose", None)
    call_params.pop("tqdm_desc", None)
    inference_parameters = inspect.signature(
        sampler._inference_model
    ).parameters
    if "guidance_interval" in inference_parameters:
        call_params.setdefault("guidance_interval", (0.0, 1.0))
    sequence = sampler.timestep_schedule(steps, rescale_t)
    pairs = [
        (float(sequence[index]), float(sequence[index + 1]))
        for index in range(steps)
    ]
    return call_params, pairs


def _sparse_from_cpu(
    features_cpu: torch.Tensor,
    coords_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> SparseTensor:
    return SparseTensor(
        features_cpu.to(device=device, dtype=torch.float32),
        coords_cpu.to(device=device, dtype=torch.int32),
    )


def _random_sparse_state(
    tile: TileRuntime,
    channels: int,
    *,
    device: torch.device,
    seed: int,
) -> SparseTensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    features = torch.randn(
        (tile.tokens, int(channels)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return SparseTensor(
        features,
        tile.coords_cpu.to(device=device, dtype=torch.int32),
    )


def _predict_velocity(
    *,
    sampler: Any,
    model: torch.nn.Module,
    state: SparseTensor,
    timestep: float,
    condition_cpu: Mapping[str, Any],
    call_params: Mapping[str, Any],
    concat_cond: Optional[SparseTensor],
    label: str,
) -> SparseTensor:
    parameters = dict(call_params)
    if concat_cond is not None:
        assert torch.equal(state.coords, concat_cond.coords)
        parameters["concat_cond"] = concat_cond
    condition = _move_nested(condition_cpu, state.device)
    _, _, prediction = sampler._get_model_prediction(
        model,
        state,
        float(timestep),
        **dict(condition),
        **parameters,
    )
    if not isinstance(prediction, SparseTensor):
        raise TypeError(f"{label}: flow prediction is not SparseTensor")
    assert torch.equal(state.coords, prediction.coords)
    assert torch.isfinite(prediction.feats).all()
    del condition
    return prediction


def _texture_predictor(
    tile: TileRuntime,
    *,
    evidence_mode: str,
) -> torch.Tensor:
    if evidence_mode == "texture":
        return tile.global_texture_norm_cpu
    if evidence_mode == "texture_global_shape":
        return torch.cat(
            (
                tile.global_texture_norm_cpu,
                tile.global_shape_norm_cpu,
            ),
            dim=1,
        )
    if evidence_mode == "texture_fused_shape":
        if tile.shape_final_norm_cpu is None:
            raise RuntimeError("final shape is unavailable for texture evidence")
        return torch.cat(
            (
                tile.global_texture_norm_cpu,
                tile.shape_final_norm_cpu,
            ),
            dim=1,
        )
    raise ValueError(evidence_mode)


def _self_consistency(
    fit: CCAFit,
    x_tiles: Sequence[torch.Tensor],
    y_tiles: Sequence[torch.Tensor],
) -> Dict[str, float]:
    squared_error = 0.0
    baseline_error = 0.0
    values = 0
    for predictor, target in zip(x_tiles, y_tiles):
        x64 = predictor.double()
        y64 = target.double()
        canonical_prediction = (
            (x64 - fit.mean_x) @ fit.weight_x
        ) * fit.rho
        centered_prediction = torch.linalg.solve(
            fit.weight_y.transpose(0, 1),
            canonical_prediction.transpose(0, 1),
        ).transpose(0, 1)
        prediction = fit.mean_y + centered_prediction
        squared_error += float((prediction - y64).square().sum().item())
        baseline_error += float((y64 - fit.mean_y).square().sum().item())
        values += int(y64.numel())
    return {
        "rmse": math.sqrt(squared_error / max(1, values)),
        "mean_baseline_rmse": math.sqrt(baseline_error / max(1, values)),
        "r2_against_mean": (
            1.0 - squared_error / baseline_error
            if baseline_error > 1e-24
            else 0.0
        ),
    }


def _aggregate_step_statistics(
    *,
    latent_name: str,
    step_index: int,
    timestep: float,
    t_previous: float,
    tile_rows: Sequence[Mapping[str, Any]],
    fit: Optional[CCAFit],
    fitted_this_step: bool,
    previous_fit: Optional[CCAFit],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    def weighted(key: str) -> Optional[float]:
        rows = [
            row for row in tile_rows
            if row.get(key) is not None and int(row["tokens"]) > 0
        ]
        total = sum(int(row["tokens"]) for row in rows)
        if total == 0:
            return None
        return float(
            sum(float(row[key]) * int(row["tokens"]) for row in rows)
            / total
        )

    result: Dict[str, Any] = {
        "latent": latent_name,
        "step": int(step_index),
        "timestep": float(timestep),
        "t_previous": float(t_previous),
        "tiles": len(tile_rows),
        "tokens": sum(int(row["tokens"]) for row in tile_rows),
        "cca_fitted_this_step": bool(fitted_this_step),
        "posterior_correction_rms_token_weighted": weighted(
            "posterior_correction_rms"
        ),
        "local_hr_clean_prediction_rms_token_weighted": weighted(
            "local_hr_clean_prediction_rms"
        ),
        "correction_over_hr_norm_token_weighted": weighted(
            "correction_over_hr_norm"
        ),
        "hr_minus_lr_clean_rms_token_weighted": weighted(
            "hr_minus_lr_clean_rms"
        ),
        "correction_hr_residual_cosine_token_weighted": weighted(
            "correction_hr_residual_cosine"
        ),
        "applied_velocity_rms_token_weighted": weighted(
            "applied_velocity_rms"
        ),
        "hr_posterior_velocity_cosine_token_weighted": weighted(
            "hr_posterior_velocity_cosine"
        ),
        "matched_clean_residual_max_error": max(
            (
                float(row.get("matched_clean_residual_max_error", 0.0))
                for row in tile_rows
            ),
            default=0.0,
        ),
        "canonical_back_transform_max_error": max(
            (
                float(row.get("canonical_back_transform_max_error", 0.0))
                for row in tile_rows
            ),
            default=0.0,
        ),
    }
    if fit is not None:
        result["cca"] = fit.diagnostics
        result["basis_change_from_previous"] = _principal_angle_diagnostics(
            previous_fit, fit
        )
    if extra:
        result.update(dict(extra))
    return result


@torch.no_grad()
def _run_object_flow(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    device: torch.device,
    output_dir: Path,
    tiles: Sequence[TileRuntime],
    latent_name: str,
    conditions: Mapping[
        int, Tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    fusion_kind: str,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Advance all tile states synchronously with object-level online CCA."""
    if latent_name == "shape":
        sampler = pipeline.shape_slat_sampler
        model = pipeline.models["shape_slat_flow_model_1024"]
        channels = int(model.in_channels)
        seed_offset = 201
    elif latent_name == "texture":
        sampler = pipeline.tex_slat_sampler
        model = pipeline.models["tex_slat_flow_model_1024"]
        channels = int(model.in_channels) - 32
        seed_offset = 301
    else:
        raise ValueError(latent_name)
    if channels != 32:
        raise RuntimeError(
            f"{latent_name}: expected 32 flow channels, got {channels}"
        )
    if bool(args.low_vram):
        model.to(device)
    call_params, time_pairs = _call_parameters(sampler, params)
    states: Dict[int, SparseTensor] = {
        tile.tile_id: _random_sparse_state(
            tile,
            channels,
            device=device,
            seed=int(args.seed) + tile.tile_id * 1000 + seed_offset,
        )
        for tile in tiles
    }
    concat_conditions: Dict[int, Optional[SparseTensor]] = {}
    for tile in tiles:
        if latent_name == "texture":
            if tile.shape_final_norm_cpu is None:
                raise RuntimeError("texture flow requires final normalized shape")
            concat_conditions[tile.tile_id] = _sparse_from_cpu(
                tile.shape_final_norm_cpu,
                tile.coords_cpu,
                device=device,
            )
            assert torch.equal(
                states[tile.tile_id].coords,
                concat_conditions[tile.tile_id].coords,
            )
        else:
            concat_conditions[tile.tile_id] = None

    requires_lr = fusion_kind in ("posterior", "old_anchor")
    fixed_fit: Optional[CCAFit] = None
    previous_fit: Optional[CCAFit] = None
    step_summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step_index, (timestep, t_previous) in enumerate(time_pairs):
        predictions: Dict[int, Dict[str, torch.Tensor]] = {}
        for tile in tqdm(
            tiles,
            desc=f"{latent_name} {fusion_kind} step {step_index:02d}",
            leave=False,
        ):
            state = states[tile.tile_id]
            condition_lr, condition_hr = conditions[tile.tile_id]
            velocity_hr = _predict_velocity(
                sampler=sampler,
                model=model,
                state=state,
                timestep=timestep,
                condition_cpu=condition_hr,
                call_params=call_params,
                concat_cond=concat_conditions[tile.tile_id],
                label=f"tile {tile.tile_id:02d} {latent_name} HR",
            )
            clean_hr = sampler._pred_to_xstart(
                state, timestep, velocity_hr
            )
            assert torch.equal(state.coords, clean_hr.coords)
            row = {
                "velocity_hr": velocity_hr.feats.detach().cpu().float(),
                "clean_hr": clean_hr.feats.detach().cpu().float(),
            }
            if requires_lr:
                velocity_lr = _predict_velocity(
                    sampler=sampler,
                    model=model,
                    state=state,
                    timestep=timestep,
                    condition_cpu=condition_lr,
                    call_params=call_params,
                    concat_cond=concat_conditions[tile.tile_id],
                    label=f"tile {tile.tile_id:02d} {latent_name} LR",
                )
                clean_lr = sampler._pred_to_xstart(
                    state, timestep, velocity_lr
                )
                assert torch.equal(state.coords, clean_lr.coords)
                sigma_t = float(
                    sampler.sigma_min
                    + (1.0 - sampler.sigma_min) * timestep
                )
                residual_direct = clean_hr.feats.float() - clean_lr.feats.float()
                residual_formula = -sigma_t * (
                    velocity_hr.feats.float() - velocity_lr.feats.float()
                )
                residual_error = float(
                    (residual_direct - residual_formula).abs().max().item()
                )
                if residual_error >= 1e-5:
                    raise AssertionError(
                        f"tile {tile.tile_id:02d} {latent_name} step "
                        f"{step_index}: matched clean residual error "
                        f"{residual_error}"
                    )
                row.update(
                    {
                        "velocity_lr": (
                            velocity_lr.feats.detach().cpu().float()
                        ),
                        "clean_lr": clean_lr.feats.detach().cpu().float(),
                        "matched_clean_residual_max_error": residual_error,
                    }
                )
            predictions[tile.tile_id] = row

        fit: Optional[CCAFit] = None
        fitted_this_step = False
        predictors: List[torch.Tensor] = []
        references: List[torch.Tensor] = []
        if fusion_kind == "posterior":
            if latent_name == "shape":
                predictors = [
                    tile.global_shape_norm_cpu for tile in tiles
                ]
            else:
                predictors = [
                    _texture_predictor(
                        tile,
                        evidence_mode=args.texture_global_evidence,
                    )
                    for tile in tiles
                ]
            references = [
                predictions[tile.tile_id]["clean_lr"] for tile in tiles
            ]
            if (
                args.cca_time_mode == "fixed_first"
                and fixed_fit is not None
            ):
                fit = fixed_fit
            else:
                fit = _fit_regularized_cca(
                    predictors,
                    references,
                    regularization=float(args.cca_regularization),
                    weighting=args.cca_weighting,
                )
                fitted_this_step = True
                if (
                    args.cca_time_mode == "fixed_first"
                    and fixed_fit is None
                ):
                    fixed_fit = fit

        tile_step_rows: List[Dict[str, Any]] = []
        for tile in tiles:
            state = states[tile.tile_id]
            prediction = predictions[tile.tile_id]
            velocity_hr_cpu = prediction["velocity_hr"]
            clean_hr_cpu = prediction["clean_hr"]
            diagnostic: Dict[str, Any] = {
                "step": int(step_index),
                "timestep": float(timestep),
                "t_previous": float(t_previous),
                "tokens": tile.tokens,
                "local_hr_clean_prediction_rms": _rms(clean_hr_cpu),
            }
            if requires_lr:
                clean_lr_cpu = prediction["clean_lr"]
                diagnostic.update(
                    {
                        "hr_minus_lr_clean_rms": _rms(
                            clean_hr_cpu - clean_lr_cpu
                        ),
                        "matched_clean_residual_max_error": prediction[
                            "matched_clean_residual_max_error"
                        ],
                    }
                )
            if fusion_kind == "local":
                applied_velocity = _sparse_from_cpu(
                    velocity_hr_cpu,
                    tile.coords_cpu,
                    device=device,
                )
                diagnostic.update(
                    {
                        "posterior_correction_rms": 0.0,
                        "correction_over_hr_norm": 0.0,
                        "correction_hr_residual_cosine": 0.0,
                        "canonical_back_transform_max_error": 0.0,
                    }
                )
            elif fusion_kind == "old_anchor":
                anchor_cpu = (
                    tile.global_shape_norm_cpu
                    if latent_name == "shape"
                    else tile.global_texture_norm_cpu
                )
                clean_star_cpu = (
                    anchor_cpu + clean_hr_cpu - clean_lr_cpu
                )
                correction = clean_star_cpu - clean_hr_cpu
                clean_star = _sparse_from_cpu(
                    clean_star_cpu,
                    tile.coords_cpu,
                    device=device,
                )
                applied_velocity = sampler._xstart_to_pred(
                    state, timestep, clean_star
                )
                diagnostic.update(
                    {
                        "negative_control_formula": "G+(H-R)",
                        "posterior_correction_rms": _rms(correction),
                        "correction_over_hr_norm": float(
                            torch.linalg.vector_norm(correction.double()).item()
                            / max(
                                1e-12,
                                float(
                                    torch.linalg.vector_norm(
                                        clean_hr_cpu.double()
                                    ).item()
                                ),
                            )
                        ),
                        "correction_hr_residual_cosine": _cosine(
                            correction, clean_hr_cpu - clean_lr_cpu
                        ),
                        "canonical_back_transform_max_error": 0.0,
                    }
                )
            elif fusion_kind == "posterior":
                assert fit is not None
                predictor = (
                    tile.global_shape_norm_cpu
                    if latent_name == "shape"
                    else _texture_predictor(
                        tile,
                        evidence_mode=args.texture_global_evidence,
                    )
                )
                clean_star_cpu, posterior_stats = (
                    _posterior_clean_prediction(
                        predictor=predictor,
                        local_reference=clean_lr_cpu,
                        local_hr=clean_hr_cpu,
                        fit=fit,
                    )
                )
                clean_star = _sparse_from_cpu(
                    clean_star_cpu,
                    tile.coords_cpu,
                    device=device,
                )
                applied_velocity = sampler._xstart_to_pred(
                    state, timestep, clean_star
                )
                diagnostic.update(posterior_stats)
            else:
                raise ValueError(fusion_kind)
            assert torch.equal(state.coords, applied_velocity.coords)
            assert torch.isfinite(applied_velocity.feats).all()
            diagnostic["applied_velocity_rms"] = _rms(
                applied_velocity.feats
            )
            diagnostic["hr_posterior_velocity_cosine"] = _cosine(
                velocity_hr_cpu, applied_velocity.feats.detach().cpu()
            )
            states[tile.tile_id] = (
                state
                - float(timestep - t_previous) * applied_velocity
            )
            assert torch.equal(
                states[tile.tile_id].coords,
                tile.coords_cpu.to(device=device),
            )
            assert torch.isfinite(states[tile.tile_id].feats).all()
            tile_step_rows.append(diagnostic)
            if latent_name == "shape":
                tile.shape_trace.append(diagnostic)
            else:
                tile.texture_trace.append(diagnostic)

        extra: Dict[str, Any] = {}
        if latent_name == "texture" and fusion_kind == "posterior":
            texture_only_predictors = [
                tile.global_texture_norm_cpu for tile in tiles
            ]
            texture_only_fit = _fit_regularized_cca(
                texture_only_predictors,
                references,
                regularization=float(args.cca_regularization),
                weighting=args.cca_weighting,
            )
            selected_consistency = _self_consistency(
                fit, predictors, references
            )
            texture_consistency = _self_consistency(
                texture_only_fit,
                texture_only_predictors,
                references,
            )
            extra = {
                "texture_evidence_variant": args.texture_global_evidence,
                "shape_concat_condition_rms_token_weighted": float(
                    sum(
                        _rms(tile.shape_final_norm_cpu) * tile.tokens
                        for tile in tiles
                    )
                    / sum(tile.tokens for tile in tiles)
                ),
                "texture_only_canonical_correlations": (
                    texture_only_fit.rho.tolist()
                ),
                "selected_predictor_canonical_correlations": (
                    fit.rho.tolist()
                ),
                "self_consistency": {
                    "texture_only": texture_consistency,
                    "selected_predictor": selected_consistency,
                    "selected_minus_texture_only_r2": float(
                        selected_consistency["r2_against_mean"]
                        - texture_consistency["r2_against_mean"]
                    ),
                },
            }
        step_summary = _aggregate_step_statistics(
            latent_name=latent_name,
            step_index=step_index,
            timestep=timestep,
            t_previous=t_previous,
            tile_rows=tile_step_rows,
            fit=fit,
            fitted_this_step=fitted_this_step,
            previous_fit=previous_fit,
            extra=extra,
        )
        step_summaries.append(step_summary)
        _atomic_json(
            output_dir
            / "statistics"
            / f"{latent_name}_step_{step_index:02d}.json",
            step_summary,
        )
        if fit is not None:
            previous_fit = fit

    _sync_cuda()
    for tile in tiles:
        final_cpu = states[tile.tile_id].feats.detach().cpu().float()
        if latent_name == "shape":
            tile.shape_final_norm_cpu = final_cpu
        else:
            tile.texture_final_norm_cpu = final_cpu
        trace = tile.shape_trace if latent_name == "shape" else tile.texture_trace
        _atomic_torch_save(
            output_dir
            / "per_tile"
            / f"tile_{tile.tile_id:02d}"
            / f"{latent_name}_trace.pt",
            {
                "format": FORMAT_VERSION,
                "tile_id": tile.tile_id,
                "latent": latent_name,
                "coords": tile.coords_cpu,
                "trace": trace,
                "final_norm": final_cpu,
            },
        )
    elapsed = float(time.perf_counter() - started)
    if bool(args.low_vram):
        model.cpu()
    del states, concat_conditions
    _empty_cuda_cache()
    return {
        "latent": latent_name,
        "fusion_kind": fusion_kind,
        "time_mode": args.cca_time_mode,
        "steps": len(time_pairs),
        "tiles": len(tiles),
        "tokens": sum(tile.tokens for tile in tiles),
        "elapsed_seconds": elapsed,
        "step_statistics": step_summaries,
    }


def _vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    faces64 = faces.to(torch.int64)
    triangles = vertices.index_select(0, faces64.reshape(-1)).reshape(-1, 3, 3)
    face_normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces64[:, corner], face_normals)
    return torch.nn.functional.normalize(normals, dim=1, eps=1e-12)


def _deterministic_rows(count: int, maximum: int) -> torch.Tensor:
    if count <= maximum:
        return torch.arange(count, dtype=torch.int64)
    return torch.round(
        torch.linspace(0, count - 1, maximum, dtype=torch.float64)
    ).to(torch.int64)


def _layout_slice(layout: Mapping[str, Any], name: str) -> slice:
    value = layout[name]
    if isinstance(value, slice):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return slice(int(value[0]), int(value[1]))
    raise ValueError(f"unsupported PBR layout entry {name}: {value!r}")


def _overlap_consistency(
    tiles: Sequence[baseline.ReturnedTileMesh],
    *,
    device: torch.device,
    maximum_samples: int,
    nearest_chunk_size: int,
) -> Dict[str, Any]:
    """Evaluate adjacent decoded tiles before ownership/welding."""
    by_id = {int(tile.tile_id): tile for tile in tiles}
    normal_cache: Dict[int, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    for tile_id, left in sorted(by_id.items()):
        for neighbor_id in (tile_id + 1, tile_id + 7):
            if neighbor_id not in by_id:
                continue
            if neighbor_id == tile_id + 1 and tile_id // 7 != neighbor_id // 7:
                continue
            right = by_id[neighbor_id]
            left_box = core._tile_layout()[tile_id]
            right_box = core._tile_layout()[neighbor_id]
            x0 = max(left_box[0], right_box[0])
            y0 = max(left_box[1], right_box[1])
            x1 = min(left_box[2], right_box[2])
            y1 = min(left_box[3], right_box[3])
            if x0 >= x1 or y0 >= y1:
                continue

            def select(tile: baseline.ReturnedTileMesh) -> torch.Tensor:
                uv = tile.vertex_uv_4096
                mask = (
                    (uv[:, 0] >= float(x0))
                    & (uv[:, 0] <= float(x1))
                    & (uv[:, 1] >= float(y0))
                    & (uv[:, 1] <= float(y1))
                )
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                subset = _deterministic_rows(
                    int(indices.shape[0]), int(maximum_samples)
                )
                return indices.index_select(0, subset)

            left_rows = select(left)
            right_rows = select(right)
            if left_rows.numel() == 0 or right_rows.numel() == 0:
                rows.append(
                    {
                        "tile_pair": [tile_id, neighbor_id],
                        "status": "skipped",
                        "reason": "no decoded vertices in image-space overlap",
                    }
                )
                continue
            left_points = left.vertices.index_select(0, left_rows).to(device)
            right_points = right.vertices.index_select(0, right_rows).to(device)
            left_distance, left_nn = core._nearest_distances(
                left_points,
                right_points,
                chunk_size=int(nearest_chunk_size),
            )
            right_distance, right_nn = core._nearest_distances(
                right_points,
                left_points,
                chunk_size=int(nearest_chunk_size),
            )
            if tile_id not in normal_cache:
                normal_cache[tile_id] = _vertex_normals(
                    left.vertices, left.faces
                )
            if neighbor_id not in normal_cache:
                normal_cache[neighbor_id] = _vertex_normals(
                    right.vertices, right.faces
                )
            left_normals = normal_cache[tile_id].index_select(
                0, left_rows
            ).to(device)
            right_normals = normal_cache[neighbor_id].index_select(
                0, right_rows
            ).to(device)
            left_attrs = left.vertex_attrs.index_select(
                0, left_rows
            ).to(device)
            right_attrs = right.vertex_attrs.index_select(
                0, right_rows
            ).to(device)
            left_attr_delta = (
                left_attrs
                - right_attrs.index_select(0, left_nn)
            )
            right_attr_delta = (
                right_attrs
                - left_attrs.index_select(0, right_nn)
            )
            left_normal_cos = (
                left_normals
                * right_normals.index_select(0, left_nn)
            ).sum(dim=1)
            right_normal_cos = (
                right_normals
                * left_normals.index_select(0, right_nn)
            ).sum(dim=1)
            base_slice = _layout_slice(left.layout, "base_color")
            roughness_slice = _layout_slice(left.layout, "roughness")
            metallic_slice = _layout_slice(left.layout, "metallic")

            def symmetric_attr_difference(channel_slice: slice) -> float:
                return float(
                    0.5
                    * (
                        left_attr_delta[:, channel_slice].abs().mean()
                        + right_attr_delta[:, channel_slice].abs().mean()
                    ).item()
                )

            rows.append(
                {
                    "tile_pair": [tile_id, neighbor_id],
                    "status": "success",
                    "left_samples": int(left_rows.shape[0]),
                    "right_samples": int(right_rows.shape[0]),
                    "overlap_chamfer_l1_object": float(
                        0.5
                        * (
                            left_distance.mean()
                            + right_distance.mean()
                        ).item()
                    ),
                    "overlap_normal_consistency_absolute": float(
                        0.5
                        * (
                            left_normal_cos.abs().mean()
                            + right_normal_cos.abs().mean()
                        ).item()
                    ),
                    "overlap_pbr_latent_rmse": float(
                        torch.cat(
                            (
                                left_attr_delta.reshape(-1),
                                right_attr_delta.reshape(-1),
                            )
                        ).square().mean().sqrt().item()
                    ),
                    "overlap_base_color_mae": symmetric_attr_difference(
                        base_slice
                    ),
                    "overlap_roughness_mae": symmetric_attr_difference(
                        roughness_slice
                    ),
                    "overlap_metallic_mae": symmetric_attr_difference(
                        metallic_slice
                    ),
                }
            )
            del (
                left_points,
                right_points,
                left_distance,
                right_distance,
                left_nn,
                right_nn,
            )
            _empty_cuda_cache()
    successful = [row for row in rows if row["status"] == "success"]
    keys = (
        "overlap_chamfer_l1_object",
        "overlap_normal_consistency_absolute",
        "overlap_pbr_latent_rmse",
        "overlap_base_color_mae",
        "overlap_roughness_mae",
        "overlap_metallic_mae",
    )
    return {
        "adjacent_pairs": len(rows),
        "successful_pairs": len(successful),
        "pair_mean": {
            key: (
                float(np.mean([float(row[key]) for row in successful]))
                if successful
                else None
            )
            for key in keys
        },
        "pairs": rows,
    }


def _mesh_topology_diagnostics(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    duplicate_tolerance: float,
    weld_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_left = np.concatenate(
        (faces_np[:, 0], faces_np[:, 1], faces_np[:, 2])
    )
    edge_right = np.concatenate(
        (faces_np[:, 1], faces_np[:, 2], faces_np[:, 0])
    )
    adjacency = scipy_sparse.coo_matrix(
        (
            np.ones(edge_left.shape[0] * 2, dtype=np.uint8),
            (
                np.concatenate((edge_left, edge_right)),
                np.concatenate((edge_right, edge_left)),
            ),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    component_count, labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_sizes = np.bincount(labels, minlength=component_count)
    vertices_np = (
        vertices.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    object_center_z = float(np.median(vertices_np[:, 2]))
    component_z_sums = np.bincount(
        labels,
        weights=vertices_np[:, 2],
        minlength=component_count,
    )
    component_mean_z = np.divide(
        component_z_sums,
        np.maximum(component_sizes, 1),
    )
    backside_components = int((component_mean_z < object_center_z).sum())
    if weld_stats is not None and bool(weld_stats.get("enabled", False)):
        weld_input = int(weld_stats["input_vertices"])
        duplicate_ratio = float(weld_stats["vertices_welded"]) / max(
            1, weld_input
        )
        duplicate_source = "exact baseline welding quantization statistics"
    else:
        duplicate_ratio = 0.0
        duplicate_source = "welding disabled; ratio not recomputed"
    return {
        "vertices": vertex_count,
        "faces": face_count,
        "connected_components": int(component_count),
        "largest_component_vertices": int(component_sizes.max(initial=0)),
        "backside_component_count": int(backside_components),
        "near_coincident_vertex_ratio": duplicate_ratio,
        "near_coincident_tolerance_object": float(duplicate_tolerance),
        "near_coincident_source": duplicate_source,
    }


def _multiview_flicker_proxy(frame_paths: Sequence[str]) -> Dict[str, Any]:
    """Report a render-domain proxy; view rotation prevents true correspondence."""
    means: List[np.ndarray] = []
    for path in frame_paths:
        image = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64)
        alpha = image[..., 3] > 0
        if not alpha.any():
            continue
        means.append(image[..., :3][alpha].mean(axis=0) / 255.0)
    if len(means) < 2:
        return {"available": False, "reason": "fewer than two nonempty views"}
    values = np.stack(means)
    return {
        "available": True,
        "definition": (
            "standard deviation of foreground mean RGB across rendered yaws; "
            "a coarse material-flicker proxy, not point-corresponded flicker"
        ),
        "foreground_mean_rgb_by_view": values.tolist(),
        "rgb_standard_deviation": values.std(axis=0).tolist(),
        "scalar_rms": float(np.sqrt(np.mean(np.square(values.std(axis=0))))),
    }


def _decode_and_evaluate(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    device: torch.device,
    output_dir: Path,
    tiles: Sequence[TileRuntime],
    global_camera: Mapping[str, float],
    baseline_mesh: Any,
) -> Dict[str, Any]:
    returned_tiles: List[baseline.ReturnedTileMesh] = []
    decode_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for tile in tqdm(tiles, desc="decode fused tiles"):
        tile_started = time.perf_counter()
        try:
            if (
                tile.shape_final_norm_cpu is None
                or tile.texture_final_norm_cpu is None
            ):
                raise RuntimeError("final shape/texture latent is unavailable")
            shape_norm = _sparse_from_cpu(
                tile.shape_final_norm_cpu,
                tile.coords_cpu,
                device=device,
            )
            texture_norm = _sparse_from_cpu(
                tile.texture_final_norm_cpu,
                tile.coords_cpu,
                device=device,
            )
            assert torch.equal(shape_norm.coords, texture_norm.coords)
            shape_raw = baseline._denormalize(
                shape_norm, pipeline.shape_slat_normalization
            )
            texture_raw = baseline._denormalize(
                texture_norm, pipeline.tex_slat_normalization
            )
            with torch.no_grad():
                decoded = pipeline.decode_latent(
                    shape_raw, texture_raw, core.OVOXEL_RESOLUTION
                )
            if len(decoded) != 1:
                raise RuntimeError("local decoder returned != 1 mesh")
            local_mesh = core._validate_mesh(
                decoded[0], f"tile {tile.tile_id:02d} fused local mesh"
            )
            returned = baseline._return_local_mesh_to_global(
                tile_id=tile.tile_id,
                local_mesh=local_mesh,
                global_camera=global_camera,
                transform=tile.transform,
            )
            returned_tiles.append(returned)
            if bool(args.save_mesh_checkpoints):
                _atomic_torch_save(
                    output_dir
                    / "per_tile"
                    / f"tile_{tile.tile_id:02d}"
                    / "final_latents.pt",
                    {
                        "coords": tile.coords_cpu,
                        "shape_norm": tile.shape_final_norm_cpu,
                        "texture_norm": tile.texture_final_norm_cpu,
                    },
                )
            record = {
                "tile_id": tile.tile_id,
                "status": "success",
                "tokens": tile.tokens,
                "decode_seconds": float(
                    time.perf_counter() - tile_started
                ),
                "returned_global_mesh": returned.stats,
            }
            del shape_norm, texture_norm, shape_raw, texture_raw, decoded
            _empty_cuda_cache()
        except Exception as error:
            record = {
                "tile_id": tile.tile_id,
                "status": "failed",
                "tokens": tile.tokens,
                "decode_seconds": float(
                    time.perf_counter() - tile_started
                ),
                "reason": f"{type(error).__name__}: {error}",
            }
            print(f"[decode tile {tile.tile_id:02d}] FAILED: {error}")
        decode_records.append(record)
    if len(returned_tiles) != len(tiles):
        raise RuntimeError(
            f"decode failures are fatal: {len(returned_tiles)}/"
            f"{len(tiles)} tiles succeeded"
        )

    overlap = _overlap_consistency(
        returned_tiles,
        device=device,
        maximum_samples=int(args.overlap_samples),
        nearest_chunk_size=int(args.nearest_chunk_size),
    )
    (
        merged_vertices,
        merged_faces,
        merged_attrs,
        merged_layout,
        ownership,
        merge_stats,
    ) = baseline._merge_tile_meshes_by_nearest_center(
        tiles=returned_tiles,
        face_projection_chunk_size=int(args.face_projection_chunk_size),
        vertex_weld_tolerance=float(args.vertex_weld_tolerance),
        device=device,
    )
    merged_mesh = baseline._direct_mesh_with_local_vertex_pbr(
        vertices=merged_vertices,
        faces=merged_faces,
        vertex_attrs=merged_attrs,
        layout=merged_layout,
    )
    if bool(args.save_mesh_checkpoints):
        torch.save(merged_mesh, output_dir / "fused_mesh.pt")

    envmap = load_envmap(str(args.envmap), device="cuda")
    reference_path = output_dir / "canonical_1024.png"
    global_metric = core._render(
        baseline_mesh,
        output_dir=output_dir / "global_baseline" / "aligned_eval",
        camera=global_camera,
        reference_image=reference_path,
        args=args,
        envmap=envmap,
    )
    fused_metric = core._render(
        merged_mesh,
        output_dir=output_dir / "aligned_eval",
        camera=global_camera,
        reference_image=reference_path,
        args=args,
        envmap=envmap,
    )
    if bool(args.render_multiview):
        multiview = baseline._render_merged_mesh_multiview(
            merged_mesh,
            output_dir=output_dir / "multiview",
            camera=global_camera,
            args=args,
            envmap=envmap,
        )
        flicker = _multiview_flicker_proxy(multiview["frame_pngs"])
    else:
        multiview = {"enabled": False}
        flicker = {"available": False, "reason": "multiview disabled"}
    topology = _mesh_topology_diagnostics(
        merged_vertices,
        merged_faces,
        duplicate_tolerance=max(
            float(args.vertex_weld_tolerance), 1e-6
        ),
        weld_stats=merge_stats["vertex_welding"],
    )
    low_frequency_similarity = core._mesh_surface_similarity(
        baseline_mesh,
        merged_mesh,
        samples=int(args.surface_samples),
        chunk_size=int(args.nearest_chunk_size),
        seed=int(args.seed) + 901,
        device=device,
    )
    return {
        "decode_seconds": float(time.perf_counter() - started),
        "successful_tiles": len(returned_tiles),
        "failed_tiles": sum(
            row["status"] == "failed" for row in decode_records
        ),
        "tiles": decode_records,
        "geometry_merge": merge_stats,
        "triangle_ownership_by_tile": ownership,
        "render_metrics": baseline._metric_subset(fused_metric),
        "global_render_metrics": baseline._metric_subset(global_metric),
        "multiview": multiview,
        "geometry_diagnostics": {
            **topology,
            "low_frequency_chamfer_to_global": low_frequency_similarity,
            "overlap": {
                "chamfer_l1_object": overlap["pair_mean"][
                    "overlap_chamfer_l1_object"
                ],
                "normal_consistency_absolute": overlap["pair_mean"][
                    "overlap_normal_consistency_absolute"
                ],
            },
        },
        "material_diagnostics": {
            "overlap_pbr_latent_consistency_rmse": overlap["pair_mean"][
                "overlap_pbr_latent_rmse"
            ],
            "overlap_base_color_difference": overlap["pair_mean"][
                "overlap_base_color_mae"
            ],
            "overlap_roughness_difference": overlap["pair_mean"][
                "overlap_roughness_mae"
            ],
            "overlap_metallic_difference": overlap["pair_mean"][
                "overlap_metallic_mae"
            ],
            "multiview_material_flicker_proxy": flicker,
            "input_view_texture_metrics": baseline._metric_subset(
                fused_metric
            ),
        },
        "overlap_pair_details": overlap,
    }


def _fusion_kinds(mode: str) -> Tuple[str, str]:
    if mode == "local":
        return "local", "local"
    if mode == "old_anchor":
        return "old_anchor", "old_anchor"
    if mode == "shape_posterior":
        return "posterior", "local"
    if mode == "texture_posterior":
        return "local", "posterior"
    if mode == "joint_posterior":
        return "posterior", "posterior"
    raise ValueError(mode)


def _load_flow_checkpoint(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    tiles: Sequence[TileRuntime],
    latent_name: str,
    fusion_kind: str,
    checkpoint_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    checkpoint_root = checkpoint_dir or output_dir
    summary_path = checkpoint_root / f"{latent_name}_flow_summary.json"
    if not bool(args.resume_flow) or not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text("utf-8"))
    expected = {
        "latent": latent_name,
        "fusion_kind": fusion_kind,
        "time_mode": args.cca_time_mode,
        "seed": int(args.seed),
        "cca_weighting": args.cca_weighting,
    }
    if latent_name == "texture":
        expected["texture_global_evidence"] = args.texture_global_evidence
    if any(summary.get(key) != value for key, value in expected.items()):
        print(f"[resume] ignored incompatible {summary_path}")
        return None
    loaded: Dict[int, Tuple[torch.Tensor, List[Dict[str, Any]]]] = {}
    for tile in tiles:
        path = (
            checkpoint_root
            / "per_tile"
            / f"tile_{tile.tile_id:02d}"
            / f"{latent_name}_trace.pt"
        )
        if not path.is_file():
            return None
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(saved["coords"], tile.coords_cpu):
            return None
        loaded[tile.tile_id] = (
            saved["final_norm"].float(),
            list(saved["trace"]),
        )
    for tile in tiles:
        final, trace = loaded[tile.tile_id]
        if latent_name == "shape":
            tile.shape_final_norm_cpu = final
            tile.shape_trace = trace
        else:
            tile.texture_final_norm_cpu = final
            tile.texture_trace = trace
        if checkpoint_root != output_dir:
            _atomic_torch_save(
                output_dir
                / "per_tile"
                / f"tile_{tile.tile_id:02d}"
                / f"{latent_name}_trace.pt",
                {
                    "format": FORMAT_VERSION,
                    "tile_id": tile.tile_id,
                    "latent": latent_name,
                    "coords": tile.coords_cpu,
                    "trace": trace,
                    "final_norm": final,
                },
            )
    summary["resumed_from_checkpoint"] = True
    summary["checkpoint_source_dir"] = str(checkpoint_root)
    if checkpoint_root != output_dir:
        _save_flow_checkpoint_summary(
            args=args, output_dir=output_dir, flow=summary
        )
    print(f"[resume] loaded complete {latent_name} flow from {summary_path}")
    return summary


def _save_flow_checkpoint_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    flow: Mapping[str, Any],
) -> None:
    latent_name = str(flow["latent"])
    _atomic_json(
        output_dir / f"{latent_name}_flow_summary.json",
        {
            **dict(flow),
            "seed": int(args.seed),
            "texture_global_evidence": args.texture_global_evidence,
            "cca_weighting": args.cca_weighting,
        },
    )


def _configuration_directory_name(
    fusion_mode: str,
    cca_time_mode: str,
) -> str:
    if fusion_mode == "local":
        return "local_hr_baseline"
    if fusion_mode == "joint_posterior":
        return (
            "joint_posterior_fixed"
            if cca_time_mode == "fixed_first"
            else "joint_posterior_per_step"
        )
    return fusion_mode


def _metric_deltas(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for key in ("psnr_db", "ssim", "lpips"):
        left = current.get(key)
        right = reference.get(key)
        output[key] = (
            None
            if left is None or right is None
            else float(left) - float(right)
        )
    return output


def _update_suite_summary(suite_root: Path) -> None:
    configurations: Dict[str, Any] = {}
    for name in (
        "local_hr_baseline",
        "old_anchor",
        "shape_posterior",
        "texture_posterior",
        "joint_posterior_fixed",
        "joint_posterior_per_step",
    ):
        path = suite_root / name / "summary.json"
        if path.is_file():
            configurations[name] = json.loads(path.read_text("utf-8"))
    if not configurations:
        return
    references: Dict[str, Mapping[str, Any]] = {}
    for name in ("local_hr_baseline", "old_anchor"):
        row = configurations.get(name)
        if row and isinstance(row.get("evaluation"), Mapping):
            references[name] = row["evaluation"].get("render_metrics", {})
    any_evaluation = next(
        (
            row["evaluation"]
            for row in configurations.values()
            if isinstance(row.get("evaluation"), Mapping)
        ),
        None,
    )
    global_metrics = (
        any_evaluation.get("global_render_metrics", {})
        if any_evaluation
        else {}
    )
    aggregate_rows: Dict[str, Any] = {}
    for name, row in configurations.items():
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, Mapping):
            aggregate_rows[name] = {
                "status": "flow_only",
                "summary": str(suite_root / name / "summary.json"),
            }
            continue
        metrics = evaluation.get("render_metrics", {})
        aggregate_rows[name] = {
            "status": "complete",
            "summary": str(suite_root / name / "summary.json"),
            "generation_seconds": row.get("generation_seconds"),
            "render_metrics": metrics,
            "deltas": {
                "vs_global": _metric_deltas(metrics, global_metrics),
                **{
                    f"vs_{reference_name}": _metric_deltas(
                        metrics, reference_metrics
                    )
                    for reference_name, reference_metrics in references.items()
                },
            },
            "geometry_diagnostics": evaluation.get(
                "geometry_diagnostics"
            ),
            "material_diagnostics": evaluation.get(
                "material_diagnostics"
            ),
            "multiview_paths": evaluation.get("multiview", {}).get(
                "frame_pngs", []
            ),
        }
    payload = {
        "format": FORMAT_VERSION,
        "suite_root": str(suite_root),
        "configurations_present": sorted(configurations),
        "configurations": aggregate_rows,
        "claims": {
            "method_success_declared": False,
            "reason": (
                "Set only after complete shape+texture renders, multiview "
                "geometry/material comparisons, and multi-seed aggregation."
            ),
        },
    }
    _atomic_json(suite_root / "summary.json", payload)


def run(args: argparse.Namespace) -> None:
    synthetic = _synthetic_tests()
    if bool(args.synthetic_test_only):
        print(json.dumps(synthetic, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] physical_index={int(args.cuda_device)} "
        f"current={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_root = (
        Path(args.suite_root).expanduser().resolve()
        if args.suite_root is not None
        else output_dir.parent
    )
    source_cache_dir = (
        Path(args.source_cache_dir).expanduser().resolve()
        if args.source_cache_dir is not None
        else suite_root / "source_cache"
    )
    _atomic_json(output_dir / "synthetic_tests.json", synthetic)
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    run_started = time.perf_counter()
    (
        image_4096,
        image_1024,
        global_camera,
        baseline_mesh,
        global_ovoxel_q,
        global_ovoxel_uv,
        global_ovoxel_finite,
        global_face_uv,
        global_face_finite,
        global_source,
    ) = _prepare_global_source(
        args=args,
        pipeline=pipeline,
        output_dir=output_dir,
        source_cache_dir=source_cache_dir,
    )
    tiles, source_records = _prepare_tiles(
        args=args,
        pipeline=pipeline,
        device=device,
        source_cache_dir=source_cache_dir,
        image_4096=image_4096,
        image_1024=image_1024,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        global_ovoxel_q=global_ovoxel_q,
        global_ovoxel_uv=global_ovoxel_uv,
        global_ovoxel_finite=global_ovoxel_finite,
        global_face_uv=global_face_uv,
        global_face_finite=global_face_finite,
    )
    if any(row["status"] == "failed" for row in source_records):
        raise RuntimeError("one or more tile anchors failed; see source records")
    _, shape_params, texture_params = _sampler_params(args)
    shape_kind, texture_kind = _fusion_kinds(args.fusion_mode)

    shape_flow = _load_flow_checkpoint(
        args=args,
        output_dir=output_dir,
        tiles=tiles,
        latent_name="shape",
        fusion_kind=shape_kind,
        checkpoint_dir=(
            Path(args.shape_flow_checkpoint_dir).expanduser().resolve()
            if args.shape_flow_checkpoint_dir is not None
            else None
        ),
    )
    if shape_flow is None:
        shape_conditions = _prepare_conditions(
            pipeline=pipeline,
            tiles=tiles,
            source_cache_dir=source_cache_dir,
            latent_name="shape",
        )
        shape_flow = _run_object_flow(
            args=args,
            pipeline=pipeline,
            device=device,
            output_dir=output_dir,
            tiles=tiles,
            latent_name="shape",
            conditions=shape_conditions,
            fusion_kind=shape_kind,
            params=shape_params,
        )
        _save_flow_checkpoint_summary(
            args=args, output_dir=output_dir, flow=shape_flow
        )
        del shape_conditions
    _empty_cuda_cache()

    texture_flow = _load_flow_checkpoint(
        args=args,
        output_dir=output_dir,
        tiles=tiles,
        latent_name="texture",
        fusion_kind=texture_kind,
    )
    if texture_flow is None:
        texture_conditions = _prepare_conditions(
            pipeline=pipeline,
            tiles=tiles,
            source_cache_dir=source_cache_dir,
            latent_name="texture",
        )
        texture_flow = _run_object_flow(
            args=args,
            pipeline=pipeline,
            device=device,
            output_dir=output_dir,
            tiles=tiles,
            latent_name="texture",
            conditions=texture_conditions,
            fusion_kind=texture_kind,
            params=texture_params,
        )
        _save_flow_checkpoint_summary(
            args=args, output_dir=output_dir, flow=texture_flow
        )
        del texture_conditions
    _empty_cuda_cache()

    if not bool(args.decode):
        evaluation: Optional[Dict[str, Any]] = None
    else:
        evaluation = _decode_and_evaluate(
            args=args,
            pipeline=pipeline,
            device=device,
            output_dir=output_dir,
            tiles=tiles,
            global_camera=global_camera,
            baseline_mesh=baseline_mesh,
        )
    generation_seconds = float(time.perf_counter() - run_started)
    summary = {
        "format": FORMAT_VERSION,
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "cuda_device": int(args.cuda_device),
        "seed": int(args.seed),
        "global_seed": int(args.global_seed),
        "configuration": {
            "fusion_mode": args.fusion_mode,
            "cca_time_mode": args.cca_time_mode,
            "texture_global_evidence": args.texture_global_evidence,
            "cca_weighting": args.cca_weighting,
            "cca_regularization": float(args.cca_regularization),
            "training_free": True,
            "gradient_updates": 0,
            "loaded_fitted_cca_or_ridge": False,
            "future_local_endpoint_used": False,
        },
        "global_source": global_source,
        "generation_seconds": generation_seconds,
        "successful_tiles": len(tiles),
        "skipped_tiles": sum(
            row["status"] == "skipped" for row in source_records
        ),
        "failed_tiles": sum(
            row["status"] == "failed" for row in source_records
        ),
        "source_tiles": source_records,
        "shape_flow": shape_flow,
        "texture_flow": texture_flow,
        "evaluation": evaluation,
        "multiview_paths": (
            evaluation.get("multiview", {}).get("frame_pngs", [])
            if evaluation
            else []
        ),
        "interpretation": {
            "data_provenance": (
                "global mesh and global-derived encoder anchors are object "
                "inputs; no diagnostic CCA/ridge or local endpoint is loaded"
            ),
            "method": (
                "full local HR clean prediction plus online regularized "
                "canonical posterior correction in normalized flow space"
            ),
            "proof_of_concept_scope": (
                "single object" if len(tiles) < 48 else "complete 48-tile object"
            ),
            "success_claim": (
                "not evaluated before complete shape+texture decode/render"
                if evaluation is None
                else "deferred to cross-configuration and multi-seed summary"
            ),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    _update_suite_summary(suite_root)
    print(f"[done] summary={output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(Path(__file__).parent / "assets" / "choose" / "0_img.png"),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/joint_online_canonical_posterior/"
            "joint_posterior_per_step"
        ),
    )
    parser.add_argument(
        "--suite-root",
        default="outputs/joint_online_canonical_posterior",
    )
    parser.add_argument("--source-cache-dir", default=None)
    parser.add_argument("--shape-flow-checkpoint-dir", default=None)
    parser.add_argument("--global-mesh-cache", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument(
        "--fusion-mode", choices=FUSION_MODES, default="joint_posterior"
    )
    parser.add_argument(
        "--cca-time-mode",
        choices=("fixed_first", "per_step"),
        default="per_step",
    )
    parser.add_argument(
        "--texture-global-evidence",
        choices=TEXTURE_EVIDENCE_MODES,
        default="texture_fused_shape",
    )
    parser.add_argument(
        "--cca-weighting", choices=("token", "tile"), default="token"
    )
    parser.add_argument("--cca-regularization", type=float, default=1e-4)
    parser.add_argument(
        "--synthetic-test-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shape-encoder",
        default=str(
            core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"
        ),
    )
    parser.add_argument(
        "--pbr-encoder",
        default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-tile-ovoxels", type=int, default=1001)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--vertex-weld-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--save-mesh-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument(
        "--render-multiview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=2)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--multiview-yaws-degrees", default="0,-45,45,-90,90,180"
    )
    parser.add_argument(
        "--multiview-pitches-degrees", default="0,0,0,10,10,0"
    )
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
    )
    parser.add_argument("--surface-samples", type=int, default=10_000)
    parser.add_argument("--overlap-samples", type=int, default=2_048)
    parser.add_argument("--nearest-chunk-size", type=int, default=1_024)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if bool(args.synthetic_test_only):
        return
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    for name in (
        "ss_steps",
        "shape_steps",
        "texture_steps",
        "min_tile_ovoxels",
        "max_num_tokens",
        "face_projection_chunk_size",
        "render_resolution",
        "metric_resolution",
        "render_ssaa",
        "render_peel_layers",
        "surface_samples",
        "overlap_samples",
        "nearest_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_tiles is not None and int(args.max_tiles) <= 0:
        raise ValueError("--max-tiles must be positive")
    if (
        not math.isfinite(float(args.cca_regularization))
        or float(args.cca_regularization) <= 0.0
    ):
        raise ValueError("--cca-regularization must be finite and positive")
    if not math.isfinite(float(args.vertex_weld_tolerance)):
        raise ValueError("--vertex-weld-tolerance must be finite")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base = Path(encoder_path).expanduser()
        if not Path(f"{base}.json").is_file() or not Path(
            f"{base}.safetensors"
        ).is_file():
            raise FileNotFoundError(
                f"encoder checkpoint pair not found for {base}"
            )
    requested = core._parse_tile_ids(args.tile_ids)
    if requested is not None:
        invalid = sorted(value for value in requested if value not in range(49))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; expected 0..48")
    if bool(args.render_multiview):
        yaws = baseline._parse_angle_csv(
            args.multiview_yaws_degrees,
            label="--multiview-yaws-degrees",
        )
        pitches = baseline._parse_angle_csv(
            args.multiview_pitches_degrees,
            label="--multiview-pitches-degrees",
        )
        if len(pitches) not in (1, len(yaws)):
            raise ValueError("multiview pitch count must be one or match yaws")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
