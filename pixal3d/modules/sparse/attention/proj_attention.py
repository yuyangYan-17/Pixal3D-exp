"""
Sparse View-Aligned Projection Attention Module for Pixal3D

Sparse versions of ProjectAttention and GatedProjectAttention.

Supports two modes:
- "proj": Standard projection (DINOv3 only)
- "gated_proj": Gated fusion of DINOv3 (semantic) + VAE (color) features
"""

from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor, VarLenTensor


class SparseProjectAttention(nn.Module):
    """
    Sparse Projection-based Attention Module with per-block proj_linear.
    """
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels, bias=True)

    @staticmethod
    def _validate_multi_tile_context(
        x: SparseTensor,
        context: Dict[str, Any],
    ) -> Tuple[torch.Tensor, SparseTensor, torch.Tensor, torch.Tensor]:
        required = {"global_bank", "proj", "tile_ids", "tile_weights"}
        missing = required.difference(context)
        if missing:
            raise KeyError(
                "multi_tile_paired context is missing: "
                + ", ".join(sorted(missing))
            )
        global_bank = context["global_bank"]
        proj_context = context["proj"]
        tile_ids = context["tile_ids"]
        tile_weights = context["tile_weights"]
        if not isinstance(global_bank, torch.Tensor) or global_bank.ndim != 3:
            raise ValueError("global_bank must have shape [T, L, C]")
        if not isinstance(proj_context, SparseTensor):
            raise TypeError("multi-tile proj must be a SparseTensor")
        if tile_ids.ndim != 2 or tile_weights.shape != tile_ids.shape:
            raise ValueError("tile_ids/tile_weights must have equal [K, M] shapes")
        token_count = x.feats.shape[0]
        if (
            proj_context.feats.shape[0] != token_count
            or tile_ids.shape[0] != token_count
        ):
            raise ValueError("multi-tile condition is not aligned with hidden rows")
        if not torch.equal(proj_context.coords, x.coords):
            raise ValueError("multi-tile projected feature coordinates are misaligned")
        if tile_ids.device != x.device or tile_weights.device != x.device:
            raise ValueError("multi-tile membership tensors must be on the hidden device")
        valid = tile_ids >= 0
        counts = valid.sum(dim=1)
        if torch.any(counts < 1) or torch.any(counts > tile_ids.shape[1]):
            raise ValueError("every token must have at least one tile membership")
        if torch.any(tile_ids[valid] >= global_bank.shape[0]):
            raise ValueError("tile_ids contains an out-of-range tile")
        if torch.any(tile_weights[~valid] != 0):
            raise ValueError("invalid membership slots must have zero weight")
        if not torch.isfinite(tile_weights).all() or torch.any(tile_weights < 0):
            raise ValueError("tile weights must be finite and non-negative")
        weight_sum = tile_weights.sum(dim=1)
        if not torch.allclose(
            weight_sum,
            torch.ones_like(weight_sum),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("tile weights for every token must sum to one")
        sorted_ids = tile_ids.masked_fill(~valid, global_bank.shape[0]).sort(1).values
        if sorted_ids.shape[1] > 1 and torch.any(
            (sorted_ids[:, 1:] == sorted_ids[:, :-1])
            & (sorted_ids[:, 1:] < global_bank.shape[0])
        ):
            raise ValueError("a token cannot contain duplicate tile IDs")
        return global_bank, proj_context, tile_ids, tile_weights

    def multi_tile_global_reference(
        self,
        x: SparseTensor,
        global_bank: torch.Tensor,
        tile_ids: torch.Tensor,
        tile_weights: torch.Tensor,
    ) -> SparseTensor:
        """Slow membership-by-membership implementation used for validation."""
        result = torch.zeros_like(x.feats)
        accumulated = torch.zeros(
            x.feats.shape[0], device=x.device, dtype=tile_weights.dtype
        )
        for row in range(x.feats.shape[0]):
            for slot in range(tile_ids.shape[1]):
                tile_id = int(tile_ids[row, slot].item())
                if tile_id < 0:
                    continue
                subset = SparseTensor(
                    feats=x.feats[row : row + 1],
                    coords=x.coords[row : row + 1],
                )
                out = self.cross_attn_block(
                    subset, global_bank[tile_id : tile_id + 1]
                )
                weight = tile_weights[row, slot]
                result[row] += out.feats[0] * weight.to(out.feats.dtype)
                accumulated[row] += weight
        if not torch.allclose(
            accumulated,
            torch.ones_like(accumulated),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("reference multi-tile attention missed memberships")
        return x.replace(result)

    def multi_tile_global_grouped(
        self,
        x: SparseTensor,
        global_bank: torch.Tensor,
        tile_ids: torch.Tensor,
        tile_weights: torch.Tensor,
    ) -> SparseTensor:
        """Apply cross-attention once per used tile and scatter queries back."""
        result = torch.zeros_like(x.feats)
        accumulated = torch.zeros(
            x.feats.shape[0], device=x.device, dtype=tile_weights.dtype
        )
        processed = 0
        valid = tile_ids >= 0
        for tile_id_tensor in torch.unique(tile_ids[valid], sorted=True):
            tile_id = int(tile_id_tensor.item())
            rows, slots = torch.where(tile_ids == tile_id)
            if rows.numel() == 0:
                continue
            subset = SparseTensor(feats=x.feats[rows], coords=x.coords[rows])
            out = self.cross_attn_block(
                subset, global_bank[tile_id : tile_id + 1]
            )
            weights = tile_weights[rows, slots]
            result.index_add_(
                0,
                rows,
                out.feats * weights[:, None].to(out.feats.dtype),
            )
            accumulated.index_add_(0, rows, weights)
            processed += int(rows.numel())
        if processed != int(valid.sum().item()):
            raise RuntimeError("grouped multi-tile attention missed memberships")
        if not torch.allclose(
            accumulated,
            torch.ones_like(accumulated),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("grouped multi-tile attention accumulated invalid weights")
        if result.shape != x.feats.shape or not torch.isfinite(result).all():
            raise RuntimeError("grouped multi-tile attention produced invalid output")
        return x.replace(result)
        
    def forward(
        self, 
        x: SparseTensor, 
        context: Union[Dict[str, Union[torch.Tensor, VarLenTensor, SparseTensor]], 
                       Tuple[Union[torch.Tensor, VarLenTensor], SparseTensor]]
    ) -> SparseTensor:
        multi_tile = isinstance(context, dict) and (
            context.get("mode") == "multi_tile_paired"
            or "global_bank" in context
        )
        if multi_tile:
            global_bank, proj_context, tile_ids, tile_weights = (
                self._validate_multi_tile_context(x, context)
            )
            global_out = self.multi_tile_global_grouped(
                x, global_bank, tile_ids, tile_weights
            )
        elif isinstance(context, dict):
            global_context = context['global']
            proj_context = context['proj']
            global_out = self.cross_attn_block(x, global_context)
        else:
            global_context, proj_context = context
            global_out = self.cross_attn_block(x, global_context)
        
        if isinstance(proj_context, SparseTensor):
            proj_feats = self.proj_linear(proj_context.feats)
            combined_feats = proj_feats + global_out.feats
        else:
            proj_feats = self.proj_linear(proj_context)
            combined_feats = proj_feats + global_out.feats
        
        return global_out.replace(combined_feats)


class SparseGatedProjectAttention(nn.Module):
    """
    Sparse Concat-Projection Attention Module for DINOv3 + VAE features.
    
    Concatenates DINOv3 and VAE projected features and applies a single linear
    projection to model_channels. Zero-initialized for stable training.
    
    Context dict must contain:
    - 'global': Global image features for cross-attention
    - 'proj_semantic': DINOv3 projected features (SparseTensor or Tensor)
    - 'proj_color': VAE projected features (SparseTensor or Tensor)
    """
    def __init__(
        self,
        cross_attn_block: nn.Module,
        channels: int,
        dino_in_channels: int,
        vae_in_channels: int,
    ):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(dino_in_channels + vae_in_channels, channels, bias=True)
        # Zero-init: at start, fused=0, only global cross-attn contributes
        nn.init.zeros_(self.proj_linear.weight)
        nn.init.zeros_(self.proj_linear.bias)

    def _get_feats(self, t):
        return t.feats if isinstance(t, SparseTensor) else t

    def forward(
        self,
        x: SparseTensor,
        context: Union[Dict[str, Union[torch.Tensor, VarLenTensor, SparseTensor]], Tuple],
    ) -> SparseTensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_semantic = context['proj_semantic']
            proj_color = context['proj_color']
        else:
            global_context, proj_semantic, proj_color = context

        global_out = self.cross_attn_block(x, global_context)

        fused = self.proj_linear(torch.cat([
            self._get_feats(proj_semantic),
            self._get_feats(proj_color),
        ], dim=-1))
        combined_feats = fused + global_out.feats

        return global_out.replace(combined_feats)
