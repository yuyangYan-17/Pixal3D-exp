"""Preserve per-sample GEMM dimensions while batching sparse attention."""

from contextlib import contextmanager
import torch
from pixal3d.modules.sparse.linear import SparseLinear


@contextmanager
def batch_stable_linear(model):
    originals = []
    sparse_layout = [None, 0]
    for layer in model.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        original = layer.forward
        originals.append((layer, original))
        if isinstance(layer, SparseLinear):

            def forward(x, layer=layer, original=original):
                sparse_layout[:] = [x.layout, len(x.feats)]
                if x.shape[0] == 1:
                    return original(x)
                return x.replace(
                    torch.cat(
                        [
                            torch.nn.functional.linear(
                                x.feats[sl], layer.weight, layer.bias
                            )
                            for sl in x.layout
                        ]
                    )
                )
        else:

            def forward(x, original=original):
                if x.ndim < 2 or x.shape[0] <= 1:
                    return original(x)
                if x.ndim == 2 and x.shape[0] == sparse_layout[1]:
                    return torch.cat(
                        [original(x[sl]) for sl in sparse_layout[0]], dim=0
                    )
                return torch.cat(
                    [original(x[i : i + 1]) for i in range(x.shape[0])], dim=0
                )

        layer.forward = forward
    try:
        yield
    finally:
        for layer, original in originals:
            layer.forward = original
