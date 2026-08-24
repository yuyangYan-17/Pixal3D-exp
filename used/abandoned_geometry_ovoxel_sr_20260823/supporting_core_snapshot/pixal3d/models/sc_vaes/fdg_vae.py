from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        return_raw_ovoxel = bool(kwargs.pop("return_raw_ovoxel", False))
        # Optional cache/materialization hook.  It deliberately leaves the
        # default decoder API unchanged while allowing a caller to offload the
        # sparse O-Voxel fields before the very large native mesh/provenance
        # extraction allocates its temporary buffers.
        return_ovoxel_fields = bool(kwargs.pop("return_ovoxel_fields", False))
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            intersected = h.replace(h.feats[..., 3:6] > 0)
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            if return_ovoxel_fields:
                return {
                    "ovoxel_fields": [{
                        "coords": v.coords[:, 1:],
                        "dual_vertices": v.feats,
                        "intersected": i.feats,
                        "intersected_logits": logits.feats,
                        "quad_lerp": q.feats,
                    } for v, i, logits, q in zip(
                        vertices, intersected, intersected_logits, quad_lerp
                    )]
                }
            mesh = []
            raw_ovoxel = []
            for v, i, logits, q in zip(vertices, intersected, intersected_logits, quad_lerp):
                if return_raw_ovoxel:
                    mesh_vertices, mesh_faces, provenance = flexible_dual_grid_to_mesh(
                        v.coords[:, 1:], v.feats, i.feats, q.feats,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        grid_size=self.resolution,
                        train=False,
                        return_provenance=True,
                    )
                else:
                    mesh_vertices, mesh_faces = flexible_dual_grid_to_mesh(
                        v.coords[:, 1:], v.feats, i.feats, q.feats,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        grid_size=self.resolution,
                        train=False,
                    )
                mesh.append(Mesh(mesh_vertices, mesh_faces))
                if return_raw_ovoxel:
                    raw_ovoxel.append({
                        "coords": v.coords[:, 1:],
                        "dual_vertices": v.feats,
                        "intersected": i.feats,
                        "intersected_logits": logits.feats,
                        "quad_lerp": q.feats,
                        "mesh_vertices": mesh_vertices,
                        "mesh_faces": mesh_faces,
                        "provenance": provenance,
                    })
            if return_raw_ovoxel:
                return {
                    "meshes": mesh,
                    "raw_ovoxel": raw_ovoxel,
                    "subdivisions": out_list[1:],
                }
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
