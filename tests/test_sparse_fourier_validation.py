import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparse_fourier_validation import (
    canonical_frequencies,
    fft_low,
    frequencies_for_cutoff,
    grid,
    project,
    rel,
)
from sparse_fourier_gaussian_alpha import carrier_preserving_gaussian


def test_dense_projection_matches_radial_fft():
    points, _ = grid(8, torch.device("cpu"))
    k1 = torch.tensor([1.0, 1.0, 0.0], dtype=points.dtype)
    k2 = torch.tensor([3.0, 0.0, 0.0], dtype=points.dtype)
    values = (torch.sin(2 * math.pi * (points @ k1)) + .2 * torch.sin(2 * math.pi * (points @ k2)))[:, None]
    frequencies = frequencies_for_cutoff(canonical_frequencies(3), 2.0)
    low = project(points, values, frequencies).low
    expected = fft_low(values.reshape(8, 8, 8, 1), 2.0).reshape_as(values)
    assert rel(low, expected) < 1e-12


def test_projection_recomposes_is_orthogonal_and_idempotent():
    points, _ = grid(7, torch.device("cpu"))
    torch.manual_seed(4)
    features = torch.randn(len(points), 5, dtype=torch.float64)
    frequencies = frequencies_for_cutoff(canonical_frequencies(3), 2.0)
    first = project(points, features, frequencies)
    second = project(points, first.low, frequencies)
    assert rel(first.low + first.high, features) < 1e-15
    assert rel(second.low, first.low) < 1e-12
    normalized_inner = (first.low * first.high).sum().abs() / (
        torch.linalg.vector_norm(first.low) * torch.linalg.vector_norm(first.high)
    )
    assert normalized_inner < 1e-12


def test_translation_and_channel_rotation_invariance():
    points, _ = grid(6, torch.device("cpu"))
    torch.manual_seed(9)
    features = torch.randn(len(points), 4, dtype=torch.float64)
    frequencies = frequencies_for_cutoff(canonical_frequencies(2), 2.0)
    base = project(points, features, frequencies).low
    translated = project(points + torch.tensor([.17, -.23, .31]), features, frequencies).low
    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    channel_rotated = project(points, features @ rotation, frequencies).low
    assert rel(translated, base) < 1e-12
    assert rel(channel_rotated, base @ rotation) < 1e-12


def test_carrier_preserving_gaussian_restores_channel_dc():
    points, _ = grid(6, torch.device("cpu"))
    torch.manual_seed(12)
    features = torch.randn(len(points), 3, dtype=torch.float32)
    low = carrier_preserving_gaussian(points, features, sigma_vox=1.5, resolution=6, chunk=64)
    assert torch.allclose(low.mean(0), features.mean(0), atol=2e-6, rtol=0)
