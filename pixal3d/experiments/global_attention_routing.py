"""C64-topology / C256-value routing primitives.

All fusion in this module is in attention head space, before the pretrained
``to_out`` projection.  Coordinates used by local RoPE never enter the global
QK product: global routing receives already-rotated baseline C64 Q/K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ParentCorrespondence:
    """A total map from unique fine rows to baseline coarse rows."""

    fine_to_coarse: torch.Tensor
    coarse_count: int

    def __post_init__(self) -> None:
        ids = self.fine_to_coarse
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise TypeError("fine_to_coarse must be a 1-D torch.long tensor")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.coarse_count):
            raise ValueError("fine_to_coarse contains an out-of-range parent")

    def reduce_mean(self, fine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """R: mean fine values by parent; empty parents are returned as zero."""
        if fine.shape[0] != self.fine_to_coarse.numel():
            raise ValueError("fine row count does not match correspondence")
        parent = self.fine_to_coarse.to(fine.device)
        shape = (self.coarse_count, *fine.shape[1:])
        reduced = torch.zeros(shape, dtype=fine.dtype, device=fine.device)
        reduced.index_add_(0, parent, fine)
        counts = torch.bincount(parent, minlength=self.coarse_count)
        denominator = counts.clamp_min(1).to(fine.dtype)
        denominator = denominator.reshape(-1, *([1] * (fine.ndim - 1)))
        return reduced / denominator, counts

    def broadcast(self, coarse: torch.Tensor) -> torch.Tensor:
        """P: copy each coarse row to all of its fine children."""
        if coarse.shape[0] != self.coarse_count:
            raise ValueError("coarse row count does not match correspondence")
        return coarse.index_select(0, self.fine_to_coarse.to(coarse.device))


def scatter_unique_rows(
    packed: torch.Tensor,
    packed_to_global: torch.Tensor,
    global_count: int,
) -> torch.Tensor:
    """Restore packed tile rows to unique global order, rejecting duplicates."""
    ids = packed_to_global.to(device=packed.device, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() != packed.shape[0]:
        raise ValueError("packed_to_global must have one id per packed row")
    counts = torch.bincount(ids, minlength=global_count)
    if counts.numel() != global_count or not bool(torch.all(counts == 1)):
        duplicate = int((counts > 1).sum())
        missing = int((counts == 0).sum())
        raise ValueError(
            f"packed rows are not a unique global cover: duplicates={duplicate}, "
            f"missing={missing}"
        )
    result = torch.empty(
        (global_count, *packed.shape[1:]), dtype=packed.dtype, device=packed.device
    )
    result.index_copy_(0, ids, packed)
    return result


def gather_packed_rows(global_value: torch.Tensor, packed_to_global: torch.Tensor) -> torch.Tensor:
    return global_value.index_select(0, packed_to_global.to(global_value.device))


def global_replace_and_residual(
    local_global_order: torch.Tensor,
    global_message: torch.Tensor,
    correspondence: ParentCorrespondence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Replace, coarse+residual, and within-parent residual."""
    local_coarse, _ = correspondence.reduce_mean(local_global_order)
    residual = local_global_order - correspondence.broadcast(local_coarse)
    replace = correspondence.broadcast(global_message)
    return replace, replace + residual, residual


def attention_for_queries(
    q: torch.Tensor,
    k: torch.Tensor,
    query_rows: torch.Tensor,
) -> torch.Tensor:
    """Explicit baseline attention for a small diagnostic query subset.

    Returns [queries, heads, keys] in float32.
    """
    selected = q.index_select(0, query_rows.to(q.device)).float()
    logits = torch.einsum("qhd,khd->qhk", selected, k.float())
    logits.mul_(q.shape[-1] ** -0.5)
    return torch.softmax(logits, dim=-1)


def outside_group_attention_ratio(
    attention: torch.Tensor,
    query_groups: torch.Tensor,
    key_groups: torch.Tensor,
) -> torch.Tensor:
    """Per-query/head mass assigned outside the query's spatial tile."""
    outside = key_groups.to(attention.device)[None, None, :] != query_groups.to(
        attention.device
    )[:, None, None]
    return (attention * outside).sum(dim=-1)


def tensor_rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())
