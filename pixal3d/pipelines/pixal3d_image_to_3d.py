from typing import *
from pathlib import Path
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel
import trimesh

class Pixal3DImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Pixal3D (proj mode) image-to-3D models.

    Based on Trellis2 pipeline, using proj mode for inference.
    Each stage (SS, Shape 512, Shape 1024, Tex 1024) has its own image_cond_model (DinoV3ProjFeatureExtractor).
    Condition building uses camera-aware projection (requires camera_angle_x, distance, mesh_scale parameters).

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model_ss (nn.Module): Proj image cond model for sparse structure stage.
        image_cond_model_shape_512 (nn.Module): Proj image cond model for shape LR (512) stage.
        image_cond_model_shape_1024 (nn.Module): Proj image cond model for shape HR (1024) stage.
        image_cond_model_tex_1024 (nn.Module): Proj image cond model for texture (1024) stage.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model_ss: nn.Module = None,
        image_cond_model_shape_512: nn.Module = None,
        image_cond_model_shape_1024: nn.Module = None,
        image_cond_model_tex_1024: nn.Module = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model_ss = image_cond_model_ss
        self.image_cond_model_shape_512 = image_cond_model_shape_512
        self.image_cond_model_shape_1024 = image_cond_model_shape_1024
        self.image_cond_model_tex_1024 = image_cond_model_tex_1024
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pixal3DImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        # Proj mode: image_cond_models need to be loaded externally, set to None here
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None

        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_canonical_images(
        self,
        input: Image.Image,
        bg_color: tuple = (0, 0, 0),
    ) -> Dict[str, Any]:
        """Run segmentation/cropping once and create the canonical pyramid."""
        source = input.copy()
        source_size = (int(source.width), int(source.height))
        source_alpha = (
            np.asarray(source.getchannel("A"))
            if source.mode == "RGBA"
            else None
        )
        has_alpha = source_alpha is not None and not np.all(source_alpha == 255)
        max_size = max(source.size)
        proxy_scale = min(1.0, 1024.0 / float(max_size))
        proxy_size = (
            max(1, int(round(source.width * proxy_scale))),
            max(1, int(round(source.height * proxy_scale))),
        )
        rembg_calls = 0
        if has_alpha:
            alpha_source = source.getchannel("A")
            alpha_proxy = alpha_source.resize(
                proxy_size, Image.Resampling.LANCZOS
            )
            alpha_kind = "rgba"
        else:
            proxy_rgb = source.convert("RGB").resize(
                proxy_size, Image.Resampling.LANCZOS
            )
            if self.low_vram:
                self.rembg_model.to(self.device)
            segmented = self.rembg_model(proxy_rgb).convert("RGBA")
            rembg_calls = 1
            if self.low_vram:
                self.rembg_model.cpu()
            alpha_proxy = segmented.getchannel("A")
            alpha_source = alpha_proxy.resize(
                source_size, Image.Resampling.LANCZOS
            )
            alpha_kind = "rembg"

        alpha_np = np.asarray(alpha_source)
        foreground_pixels = np.argwhere(alpha_np > 0.8 * 255)
        if foreground_pixels.size == 0:
            raise ValueError("Foreground preprocessing produced an empty alpha mask")
        # Right/bottom are exclusive pixel edges.
        foreground_bbox_source = (
            int(np.min(foreground_pixels[:, 1])),
            int(np.min(foreground_pixels[:, 0])),
            int(np.max(foreground_pixels[:, 1])) + 1,
            int(np.max(foreground_pixels[:, 0])) + 1,
        )
        center = (
            (foreground_bbox_source[0] + foreground_bbox_source[2]) / 2.0,
            (foreground_bbox_source[1] + foreground_bbox_source[3]) / 2.0,
        )
        side = max(
            foreground_bbox_source[2] - foreground_bbox_source[0],
            foreground_bbox_source[3] - foreground_bbox_source[1],
        )
        side = max(1, int(math.ceil(side * 1.1)))
        left = int(math.floor(center[0] - side / 2.0))
        top = int(math.floor(center[1] - side / 2.0))
        square_extent = (left, top, left + side, top + side)
        padding = (
            max(0, -left),
            max(0, left + side - source.width),
            max(0, -top),
            max(0, top + side - source.height),
        )
        source_rgba = source.convert("RGBA")
        source_rgba.putalpha(alpha_source)
        square_rgba = source_rgba.crop(square_extent)
        if square_rgba.size != (side, side):
            raise RuntimeError("canonical source crop is not square")
        square_array = np.asarray(square_rgba).astype(np.float32) / 255.0
        background = np.asarray(bg_color, dtype=np.float32) / 255.0
        composited = (
            square_array[:, :, :3] * square_array[:, :, 3:4]
            + background * (1.0 - square_array[:, :, 3:4])
        )
        source_square = Image.fromarray(
            (np.clip(composited, 0, 1) * 255).astype(np.uint8), mode="RGB"
        )
        image_4096 = source_square.resize(
            (4096, 4096), Image.Resampling.LANCZOS
        )
        image_1024 = image_4096.resize(
            (1024, 1024), Image.Resampling.LANCZOS
        )
        image_512 = image_4096.resize(
            (512, 512), Image.Resampling.LANCZOS
        )
        foreground_mask_4096 = square_rgba.getchannel("A").resize(
            (4096, 4096), Image.Resampling.LANCZOS
        )
        metadata = {
            "version": "canonical_v1",
            "source_size": list(source_size),
            "alpha_source": alpha_kind,
            "rembg_calls": rembg_calls,
            "rembg_input": list(proxy_size) if rembg_calls else None,
            "foreground_bbox_source": list(foreground_bbox_source),
            "square_extent_source": list(square_extent),
            "padding": {
                "left": padding[0], "right": padding[1],
                "top": padding[2], "bottom": padding[3],
            },
            "source_square_size": [side, side],
        }
        print(
            "[canonical-preprocess] "
            f"alpha_source={alpha_kind} rembg_calls={rembg_calls} "
            f"rembg_input={metadata['rembg_input']} "
            f"foreground_bbox_source={foreground_bbox_source} "
            f"square_extent_source={square_extent} padding={padding} "
            "image_4096=4096x4096 image_1024=1024x1024 image_512=512x512"
        )
        canonical = {
            "image_4096": image_4096,
            "image_1024": image_1024,
            "image_512": image_512,
            "foreground_mask_4096": foreground_mask_4096,
            "source_square_rgba": square_rgba,
            "source_square_black_rgb": source_square,
            "metadata": metadata,
        }
        # Read-only aliases keep the explicitly legacy 2D tile experiment
        # runnable without another preprocessing operation.
        canonical.update(
            {
                "global_image": image_1024,
                "hr_image": image_4096,
                "foreground_mask_hr": foreground_mask_4096,
                "global_to_hr_transform": {
                    "convention": "canonical pixel-edge coordinates",
                    "global_size": [1024, 1024],
                    "hr_size": [4096, 4096],
                    "global_to_hr_matrix": [
                        [4.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "hr_to_global_matrix": [
                        [0.25, 0.0, 0.0],
                        [0.0, 0.25, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        )
        return canonical

    def preprocess_image(
        self, input: Image.Image, bg_color: tuple = (0, 0, 0)
    ) -> Image.Image:
        """Compatibility wrapper returning the canonical 1024 image."""
        return self.preprocess_canonical_images(input, bg_color)["image_1024"]

    def preprocess_image_with_hr(
        self, input: Image.Image, bg_color: tuple = (0, 0, 0)
    ) -> Dict[str, Any]:
        """Compatibility wrapper for the former shared HR bundle."""
        canonical = self.preprocess_canonical_images(input, bg_color)
        return {
            **canonical,
            "global_image": canonical["image_1024"],
            "hr_image": canonical["image_4096"],
            "foreground_mask_hr": canonical["foreground_mask_4096"],
            "global_to_hr_transform": {
                "convention": "canonical normalized image coordinates",
                "global_size": [1024, 1024],
                "hr_size": [4096, 4096],
                "global_to_hr_matrix": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
                "hr_to_global_matrix": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]],
            },
        }

    @staticmethod
    def build_texture_image_tile_layout(
        canonical_size: int = 4096,
        tile_size: int = 1024,
        tile_stride: int = 512,
    ) -> List[Tuple[int, int, int, int]]:
        """Return only complete, in-bounds image tiles in row-major order."""
        if canonical_size <= 0 or tile_size <= 0 or tile_stride <= 0:
            raise ValueError("canonical/tile sizes and stride must be positive")
        if tile_size > canonical_size:
            raise ValueError("tile_size cannot exceed canonical_size")
        starts = list(range(0, canonical_size - tile_size + 1, tile_stride))
        if not starts or starts[-1] != canonical_size - tile_size:
            raise ValueError(
                "tile layout must land exactly on the final canonical edge"
            )
        return [
            (x0, y0, x0 + tile_size, y0 + tile_size)
            for y0 in starts
            for x0 in starts
        ]

    @staticmethod
    def assign_texture_tiles(
        raw_uv: torch.Tensor,
        boxes: Sequence[Sequence[int]],
        canonical_size: int = 4096,
        max_memberships: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign finite pixel-space projections to overlapping image tiles."""
        if raw_uv.ndim != 2 or raw_uv.shape[1] != 2:
            raise ValueError("raw_uv must have shape [N, 2]")
        if not torch.isfinite(raw_uv).all():
            raise RuntimeError("non-finite projected UV cannot be assigned")
        if max_memberships != 4:
            raise ValueError("the first paired condition format requires M=4")
        assignment_uv = raw_uv.clamp(0.0, float(canonical_size))
        if not torch.isfinite(assignment_uv).all():
            raise RuntimeError("clamped assignment UV is non-finite")
        n = raw_uv.shape[0]
        tile_ids = torch.full(
            (n, max_memberships), -1, dtype=torch.long, device=raw_uv.device
        )
        tile_weights = torch.zeros(
            (n, max_memberships), dtype=torch.float32, device=raw_uv.device
        )
        counts = torch.zeros(n, dtype=torch.long, device=raw_uv.device)
        for tile_id, box in enumerate(boxes):
            x0, y0, x1, y1 = (float(value) for value in box)
            # Half-open membership, except that the canonical maximum belongs
            # to the last row/column.
            in_x = (assignment_uv[:, 0] >= x0) & (
                (assignment_uv[:, 0] < x1)
                | ((x1 == canonical_size) & (assignment_uv[:, 0] == x1))
            )
            in_y = (assignment_uv[:, 1] >= y0) & (
                (assignment_uv[:, 1] < y1)
                | ((y1 == canonical_size) & (assignment_uv[:, 1] == y1))
            )
            rows = torch.where(in_x & in_y)[0]
            if rows.numel() == 0:
                continue
            slots = counts[rows]
            if torch.any(slots >= max_memberships):
                raise RuntimeError("a token belongs to more than four image tiles")
            tile_ids[rows, slots] = tile_id
            local_x = (assignment_uv[rows, 0] - x0) / (x1 - x0)
            local_y = (assignment_uv[rows, 1] - y0) / (y1 - y0)
            weight = (
                (1.0 - (2.0 * local_x - 1.0).abs())
                * (1.0 - (2.0 * local_y - 1.0).abs())
            ).clamp_min(1e-3)
            tile_weights[rows, slots] = weight.float()
            counts[rows] += 1
        if torch.any(counts < 1) or torch.any(counts > 4):
            bad = torch.where((counts < 1) | (counts > 4))[0].tolist()
            raise RuntimeError(f"invalid image tile coverage for token rows {bad[:16]}")
        tile_weights /= tile_weights.sum(dim=1, keepdim=True)
        if (
            not torch.isfinite(tile_weights).all()
            or torch.any(tile_weights < 0)
            or not torch.allclose(
                tile_weights.sum(1),
                torch.ones(n, device=raw_uv.device),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise RuntimeError("image tile weights failed normalization")
        sorted_ids = tile_ids.masked_fill(tile_ids < 0, len(boxes)).sort(1).values
        if torch.any(
            (sorted_ids[:, 1:] == sorted_ids[:, :-1])
            & (sorted_ids[:, 1:] < len(boxes))
        ):
            raise RuntimeError("duplicate image tile membership")
        return tile_ids, tile_weights, assignment_uv

    @staticmethod
    def fuse_texture_slot_proj(
        slot_proj: torch.Tensor,
        tile_weights: torch.Tensor,
        tile_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if slot_proj.ndim != 3 or tile_weights.shape != slot_proj.shape[:2]:
            raise ValueError("slot_proj and tile_weights are not aligned")
        if tile_ids is not None:
            if tile_ids.shape != tile_weights.shape:
                raise ValueError("tile_ids and weights are not aligned")
            if torch.any(slot_proj[tile_ids < 0] != 0):
                raise ValueError("invalid projected-feature slots must be zero")
        fused = (slot_proj * tile_weights[..., None].to(slot_proj.dtype)).sum(1)
        if not torch.isfinite(fused).all():
            raise RuntimeError("fused projected features are non-finite")
        return fused

    @staticmethod
    def make_multitile_negative_condition(
        condition: Mapping[str, Any],
    ) -> Dict[str, Any]:
        projection = condition["proj"]
        if not isinstance(projection, SparseTensor):
            raise TypeError("paired condition proj must be a SparseTensor")
        return {
            "mode": "multi_tile_paired",
            "global_bank": torch.zeros_like(condition["global_bank"]),
            "proj": projection.replace(torch.zeros_like(projection.feats)),
            "tile_ids": condition["tile_ids"],
            "tile_weights": condition["tile_weights"],
        }

    @staticmethod
    def build_texture_3d_patches(
        global_coords: torch.Tensor,
        grid_size: int = 128,
        patch_size: int = 64,
        patch_stride: int = 32,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        if global_coords.ndim != 2 or global_coords.shape[1] != 4:
            raise ValueError("global_coords must have shape [N, 4]")
        if torch.any(global_coords[:, 1:] < 0) or torch.any(
            global_coords[:, 1:] >= grid_size
        ):
            raise ValueError("global coordinates lie outside the requested grid")
        starts = list(range(0, grid_size - patch_size + 1, patch_stride))
        if not starts or starts[-1] != grid_size - patch_size:
            raise ValueError("3D patch layout must land on the final grid edge")
        patches: List[Dict[str, Any]] = []
        coverage = torch.zeros(
            global_coords.shape[0], dtype=torch.int32, device=global_coords.device
        )
        xyz = global_coords[:, 1:]
        for sx in starts:
            for sy in starts:
                for sz in starts:
                    mask = (
                        (xyz[:, 0] >= sx) & (xyz[:, 0] < sx + patch_size)
                        & (xyz[:, 1] >= sy) & (xyz[:, 1] < sy + patch_size)
                        & (xyz[:, 2] >= sz) & (xyz[:, 2] < sz + patch_size)
                    )
                    indices = torch.where(mask)[0]
                    local_coords = global_coords[indices].clone()
                    local_coords[:, 1:] -= torch.tensor(
                        [sx, sy, sz],
                        device=local_coords.device,
                        dtype=local_coords.dtype,
                    )
                    if indices.numel() and (
                        local_coords[:, 1:].amin() < 0
                        or local_coords[:, 1:].amax() >= patch_size
                    ):
                        raise RuntimeError("3D patch local coordinates are invalid")
                    coverage.index_add_(
                        0, indices, torch.ones_like(indices, dtype=torch.int32)
                    )
                    patches.append(
                        {
                            "start": (sx, sy, sz),
                            "global_indices": indices,
                            "local_coords": local_coords,
                        }
                    )
        if torch.any(coverage < 1) or torch.any(coverage > 8):
            raise RuntimeError("strict 3D patch coverage failed")
        return patches, coverage

    @staticmethod
    def texture_3d_patch_weights(
        local_coords: torch.Tensor,
        patch_size: int = 64,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        xyz = local_coords[:, 1:].to(torch.float32)
        normalized = xyz / float(patch_size - 1)
        axes = (1.0 - (2.0 * normalized - 1.0).abs()).clamp_min(eps)
        return axes.prod(dim=1)

    @staticmethod
    def merge_texture_3d_patch_velocities(
        token_count: int,
        patch_results: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        channels: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        velocity_sum = torch.zeros(token_count, channels, device=device)
        weight_sum = torch.zeros(token_count, 1, device=device)
        coverage = torch.zeros(token_count, dtype=torch.int32, device=device)
        for indices, velocity, weights in patch_results:
            if velocity.shape != (indices.numel(), channels):
                raise ValueError("patch velocity is not aligned with indices")
            velocity_sum.index_add_(
                0, indices, velocity.float() * weights[:, None].float()
            )
            weight_sum.index_add_(0, indices, weights[:, None].float())
            coverage.index_add_(
                0, indices, torch.ones_like(indices, dtype=torch.int32)
            )
        if (
            torch.any(coverage < 1)
            or torch.any(weight_sum <= 0)
            or not torch.isfinite(velocity_sum).all()
            or not torch.isfinite(weight_sum).all()
        ):
            bad = torch.where((coverage < 1) | (weight_sum[:, 0] <= 0))[0]
            raise RuntimeError(
                "strict patch velocity coverage failed for rows "
                f"{bad[:16].tolist()}"
            )
        merged = velocity_sum / weight_sum
        if not torch.isfinite(merged).all():
            raise RuntimeError("merged patch velocity is non-finite")
        return merged, coverage

    # =========================================================================
    # Proj mode condition building
    # =========================================================================

    @torch.no_grad()
    def get_proj_cond_ss(
        self,
        image: list,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
    ) -> dict:
        """
        Get proj conditioning for sparse structure stage.

        Args:
            image: List of PIL images.
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': ...}
        """
        device = self.device
        image_cond_model = self.image_cond_model_ss
        if self.low_vram:
            image_cond_model.to(device)
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = image_cond_model(
            image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
        )
        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
        projection_crop_box: Optional[Sequence[float]] = None,
        transform_matrix: Optional[torch.Tensor] = None,
        preserve_image_resolution: bool = False,
    ) -> dict:
        """
        Get proj conditioning for shape/texture stages (sparse-token aligned).

        Args:
            image_cond_model: The proj image cond model for this stage.
            image: List of PIL images.
            coords: Sparse structure coordinates [N, 4] (batch_idx, x, y, z).
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.
            grid_resolution_override: Override the grid resolution if not None.
            projection_crop_box: Optional normalized crop in the complete
                camera image. The image model projects globally before mapping
                points into crop-local feature coordinates.
            transform_matrix: Optional camera-to-world matrix, shaped [4, 4]
                or [1, 4, 4]. When omitted, the standard centered front-view
                camera is used. Sparse tensor identity still comes from
                ``coords``; this matrix only controls image-feature sampling.
            preserve_image_resolution: Forward a patch-aligned native-size
                crop to DINO/NAF instead of resizing it to the model's nominal
                square input size.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': SparseTensor}
        """
        device = self.device
        if self.low_vram:
            image_cond_model.to(device)

        B = 1
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"coords must have shape [N, 4], got {tuple(coords.shape)}")
        if torch.any(coords[:, 0] != 0):
            raise ValueError("get_proj_cond_shape currently supports batch size 1 only")

        grid_res = int(grid_resolution_override or image_cond_model.grid_resolution)
        print(
            f"[proj-sparse] grid={grid_res} tokens={int(coords.shape[0]):,} "
            f"dense_tokens={grid_res ** 3:,}"
        )
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        if transform_matrix is not None:
            transform_matrix = torch.as_tensor(
                transform_matrix,
                dtype=torch.float32,
                device=device,
            )
            if transform_matrix.shape == (4, 4):
                transform_matrix = transform_matrix.unsqueeze(0)
            if transform_matrix.shape != (B, 4, 4):
                raise ValueError(
                    "transform_matrix must have shape [4, 4] or [1, 4, 4], "
                    f"got {tuple(transform_matrix.shape)}"
                )
        image_model_kwargs = {
            "camera_angle_x": cam_angle,
            "distance": dist_tensor,
            "mesh_scale": scale_tensor,
            "transform_matrix": transform_matrix,
            "grid_indices": coords[:, 1:4],
            "grid_resolution": grid_res,
            "projection_crop_box": projection_crop_box,
        }
        if preserve_image_resolution:
            image_model_kwargs["preserve_input_resolution"] = True
        z_global, z_proj = image_cond_model(image, **image_model_kwargs)
        if z_proj.shape[0] != B or z_proj.shape[1] != coords.shape[0]:
            raise RuntimeError(
                "Sparse projection output is not aligned with coords: "
                f"proj={tuple(z_proj.shape)} coords={tuple(coords.shape)}"
            )
        z_proj_sparse = z_proj[0]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    # =========================================================================
    # Sampling methods (consistent with Trellis2)
    # =========================================================================

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            noise (torch.Tensor, optional): Explicit dense flow noise. This is
                used by canonical global/local synchronization so overlapping
                camera cells can read the same spatial white-noise realization.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        if noise is None:
            noise = torch.randn(
                num_samples, in_channels, reso, reso, reso
            ).to(self.device)
        else:
            expected = (num_samples, in_channels, reso, reso, reso)
            if tuple(noise.shape) != expected:
                raise ValueError(
                    f"explicit sparse-structure noise has shape "
                    f"{tuple(noise.shape)}, expected {expected}"
                )
            noise = noise.to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with cascade (LR → HR).
        
        Args:
            lr_cond (dict): The conditioning information for LR stage.
            cond (dict): The conditioning information for HR stage.
            flow_model_lr: LR flow model.
            flow_model: HR flow model.
            lr_resolution (int): LR resolution.
            resolution (int): Target HR resolution.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            max_num_tokens (int): Maximum number of tokens.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling LR shape SLat (proj, 512)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent (HR)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample texture structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    def export_tex_voxel_point_cloud(
        self,
        tex_voxel,
        out_path: str,
        max_points: int = 100000000,
        alpha_threshold: float = -1.0,
    ):
        """
        将 tex_voxel 导出为带 base-color 的 PLY 点云。
        tex_voxel.coords: [N, 4]，第一列通常是 batch idx，后三列是 xyz voxel coords
        tex_voxel.feats:  [N, 6]，前 3 维通常是 base color，后面是 metallic/roughness/alpha
        """
        coords = tex_voxel.coords.detach().cpu()
        feats = tex_voxel.feats.detach().cpu().float()

        # 若包含 batch 维，取 batch=0
        if coords.shape[1] == 4:
            batch_mask = coords[:, 0] == 0
            coords = coords[batch_mask, 1:]
            feats = feats[batch_mask]
        else:
            coords = coords[:, :3]

        # 去掉非有限值
        finite_mask = torch.isfinite(feats).all(dim=1)
        coords = coords[finite_mask]
        feats = feats[finite_mask]

        # 可选：按 alpha 过滤
        if alpha_threshold >= 0 and feats.shape[1] >= 6:
            alpha = feats[:, 5]
            keep = alpha > alpha_threshold
            coords = coords[keep]
            feats = feats[keep]

        if coords.shape[0] == 0:
            print(f"[Debug] No tex voxels left after filtering, skip export: {out_path}")
            return

        # 若点太多，随机下采样，避免 PLY 太大
        if coords.shape[0] > max_points:
            idx = torch.randperm(coords.shape[0])[:max_points]
            coords = coords[idx]
            feats = feats[idx]

        # 用 tex_voxel 自己的 spatial_shape 做归一化，更稳妥
        spatial_shape = tex_voxel.spatial_shape
        if isinstance(spatial_shape, torch.Size):
            spatial_shape = list(spatial_shape)
        spatial_shape = np.array(spatial_shape, dtype=np.float32)

        # voxel index -> world coord, 对应 aabb 大致 [-0.5, 0.5]
        points = (coords.numpy().astype(np.float32) + 0.5) / spatial_shape[None, :] - 0.5

        # 前 3 通道作为 base color
        colors = feats[:, :3].numpy()
        colors = np.clip(colors, 0.0, 1.0)
        colors = (colors * 255.0).astype(np.uint8)

        cloud = trimesh.PointCloud(vertices=points, colors=colors)
        cloud.export(out_path)

        print(f"[Debug] tex_voxel point cloud saved to: {out_path}")
        print(f"[Debug] num_points={len(points):,}, spatial_shape={tuple(spatial_shape.tolist())}")
        print(f"[Debug] base_color mean={feats[:, :3].mean(dim=0).tolist()}")
        print(f"[Debug] base_color std ={feats[:, :3].std(dim=0).tolist()}")
        print(f"[Debug] base_color min ={feats[:, :3].amin(dim=0).tolist()}")
        print(f"[Debug] base_color max ={feats[:, :3].amax(dim=0).tolist()}")

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
        debug_point_cloud_path: Optional[Union[str, Path]] = None,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
            debug_point_cloud_path (str or Path, optional): Export the decoded
                texture voxels as a PLY only when an explicit path is supplied.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        if debug_point_cloud_path is not None:
            self.export_tex_voxel_point_cloud(
                tex_voxels[0],
                out_path=str(debug_point_cloud_path),
            )
        out_mesh = []
        torch.cuda.synchronize()
        for m, v in zip(meshes, tex_voxels):
            # m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh

    @staticmethod
    def _build_2048_overlap_patches(
        coords: torch.Tensor,
        grid_resolution: int = 128,
        patch_size: int = 64,
        patch_stride: int = 32,
    ) -> List[Dict[str, Any]]:
        """Build the fixed 3 x 3 x 3 overlapping sparse patch layout."""
        if grid_resolution != 128 or patch_size != 64 or patch_stride != 32:
            raise ValueError(
                "The current 2048 experiment requires grid=128, "
                "patch_size=64 and patch_stride=32"
            )
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"coords must be [N, 4], got {tuple(coords.shape)}")
        xyz = coords[:, 1:4]
        if torch.any(xyz < 0) or torch.any(xyz >= grid_resolution):
            raise ValueError("2048 sparse coordinates must lie in [0, 127]")

        starts = list(range(0, grid_resolution - patch_size + 1, patch_stride))
        if starts != [0, 32, 64]:
            raise RuntimeError(f"Unexpected patch starts: {starts}")
        overlap = patch_size - patch_stride
        patches: List[Dict[str, Any]] = []
        patch_index = 0
        for start_x in starts:
            for start_y in starts:
                for start_z in starts:
                    start = torch.tensor(
                        [start_x, start_y, start_z],
                        device=xyz.device,
                        dtype=xyz.dtype,
                    )
                    end = start + patch_size
                    mask = ((xyz >= start) & (xyz < end)).all(dim=1)
                    token_indices = mask.nonzero(as_tuple=False).flatten()
                    weights = None
                    local_coords = None
                    if token_indices.numel() > 0:
                        global_patch_coords = coords[token_indices]
                        local_coords = global_patch_coords.clone()
                        local_coords[:, 1:4] -= start
                        local_xyz = local_coords[:, 1:4]
                        if torch.any(local_xyz < 0) or torch.any(
                            local_xyz >= patch_size
                        ):
                            raise RuntimeError(
                                f"Patch {patch_index} local coordinates fall "
                                f"outside [0, {patch_size - 1}]"
                            )
                        if not torch.equal(local_xyz + start, xyz[token_indices]):
                            raise RuntimeError(
                                f"Patch {patch_index} local/global coordinate "
                                "round-trip check failed"
                            )
                        patch_xyz = xyz[token_indices].to(torch.float32)
                        weight_axes = []
                        for axis in range(3):
                            axis_weight = torch.ones(
                                token_indices.shape[0],
                                device=coords.device,
                                dtype=torch.float32,
                            )
                            axis_start = int(start[axis].item())
                            axis_end = axis_start + patch_size
                            if axis_start > 0:
                                left_weight = (
                                    patch_xyz[:, axis] - axis_start + 1.0
                                ) / float(overlap + 1)
                                axis_weight = torch.minimum(
                                    axis_weight, left_weight
                                )
                            if axis_end < grid_resolution:
                                right_weight = (
                                    axis_end - patch_xyz[:, axis]
                                ) / float(overlap + 1)
                                axis_weight = torch.minimum(
                                    axis_weight, right_weight
                                )
                            weight_axes.append(axis_weight)
                        weights = (
                            weight_axes[0] * weight_axes[1] * weight_axes[2]
                        ).clamp_min(torch.finfo(torch.float32).eps)
                    patches.append(
                        {
                            "patch_index": patch_index,
                            "start": (start_x, start_y, start_z),
                            "end": (
                                start_x + patch_size,
                                start_y + patch_size,
                                start_z + patch_size,
                            ),
                            "token_indices": token_indices,
                            "weights": weights,
                            "local_coords": local_coords,
                        }
                    )
                    patch_index += 1
        if len(patches) != 27:
            raise RuntimeError(f"Expected 27 patches, got {len(patches)}")
        return patches

    @staticmethod
    def _slice_sparse_condition(
        branch: Mapping[str, Any],
        token_indices: torch.Tensor,
        patch_local_coords: torch.Tensor,
        global_token_count: int,
        global_coords: torch.Tensor,
    ) -> Dict[str, Any]:
        """Slice projected features and attach the patch-local coordinates."""
        patch_branch: Dict[str, Any] = {}
        for key, value in branch.items():
            if isinstance(value, SparseTensor):
                if value.feats.shape[0] != global_token_count:
                    raise ValueError(
                        f"Sparse condition {key!r} has {value.feats.shape[0]} "
                        f"tokens; expected {global_token_count}"
                    )
                if not torch.equal(value.coords, global_coords):
                    raise RuntimeError(
                        f"Sparse condition {key!r} coordinates/order are not "
                        "aligned with the global latent"
                    )
                patch_branch[key] = SparseTensor(
                    feats=value.feats[token_indices],
                    coords=patch_local_coords,
                )
            else:
                # In particular, cond['global'] / neg_cond['global'] retain
                # the exact same tensor/value used by the global flow.
                patch_branch[key] = value
        return patch_branch

    @staticmethod
    def _slice_aligned_sparse_tensor(
        value: Optional[SparseTensor],
        token_indices: torch.Tensor,
        patch_local_coords: torch.Tensor,
        global_coords: torch.Tensor,
        name: str,
    ) -> Optional[SparseTensor]:
        """Slice a token-aligned sparse tensor and switch to local coordinates."""
        if value is None:
            return None
        if not isinstance(value, SparseTensor):
            raise TypeError(f"{name} must be a SparseTensor")
        if value.feats.shape[0] != global_coords.shape[0]:
            raise ValueError(
                f"{name} has {value.feats.shape[0]} tokens; expected "
                f"{global_coords.shape[0]}"
            )
        if not torch.equal(value.coords, global_coords):
            raise RuntimeError(
                f"{name} coordinates/order are not aligned with the global latent"
            )
        return SparseTensor(
            feats=value.feats[token_indices],
            coords=patch_local_coords,
        )

    @staticmethod
    def _velocity_similarity(
        merged_velocity: torch.Tensor,
        global_velocity: torch.Tensor,
    ) -> Dict[str, float]:
        merged = merged_velocity.detach().to(device="cpu", dtype=torch.float32)
        reference = global_velocity.detach().to(
            device="cpu", dtype=torch.float32
        )
        if merged.shape != reference.shape:
            raise ValueError(
                "Velocity shapes differ: "
                f"merged={tuple(merged.shape)} global={tuple(reference.shape)}"
            )
        merged_flat = merged.reshape(-1)
        reference_flat = reference.reshape(-1)
        eps = torch.finfo(torch.float32).eps
        difference = merged_flat - reference_flat
        return {
            "cosine_similarity": float(
                F.cosine_similarity(
                    merged_flat.unsqueeze(0),
                    reference_flat.unsqueeze(0),
                    dim=1,
                    eps=eps,
                ).item()
            ),
            "mean_token_cosine_similarity": float(
                F.cosine_similarity(merged, reference, dim=1, eps=eps)
                .mean()
                .item()
            ),
            "mse": float(difference.square().mean().item()),
            "relative_l2": float(
                (difference.norm() / reference_flat.norm().clamp_min(eps)).item()
            ),
            "norm_ratio": float(
                (merged_flat.norm() / reference_flat.norm().clamp_min(eps)).item()
            ),
        }

    @staticmethod
    def _project_sparse_coords_to_image_norm(
        image_cond_model: nn.Module,
        coords: torch.Tensor,
        camera_angle_x: float,
        distance: float,
        mesh_scale: float,
        grid_resolution: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project sparse coordinates through the condition model's camera."""
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"coords must be [N, 4], got {tuple(coords.shape)}")
        if torch.any(coords[:, 0] != 0):
            raise ValueError("Image-tile projection supports batch size 1 only")
        proj_grid = getattr(image_cond_model, "proj_grid", None)
        if proj_grid is None or not hasattr(proj_grid, "project_grid_indices"):
            raise TypeError(
                "Texture image condition model does not expose the canonical "
                "ProjGrid.project_grid_indices path"
            )
        device = coords.device
        image_points, depth, valid_mask = proj_grid.project_grid_indices(
            camera_angle_x=torch.tensor(
                [camera_angle_x], device=device, dtype=torch.float32
            ),
            distance=torch.tensor(
                [distance], device=device, dtype=torch.float32
            ),
            mesh_scale=torch.tensor(
                [mesh_scale], device=device, dtype=torch.float32
            ),
            grid_indices=coords[:, 1:4],
            grid_resolution=int(grid_resolution),
        )
        full_image_norm = (
            image_points + 0.5
        ) / float(proj_grid.image_resolution)
        return (
            full_image_norm[0],
            depth[0],
            valid_mask[0],
        )

    @staticmethod
    def _image_tile_starts(
        image_extent: int,
        tile_size: int,
        tile_stride: int,
    ) -> List[int]:
        if image_extent <= 0:
            raise ValueError("image_extent must be positive")
        if tile_size <= 0 or tile_stride <= 0:
            raise ValueError("tile_size and tile_stride must be positive")
        if tile_stride > tile_size:
            raise ValueError(
                "tile_stride may not exceed tile_size because that would "
                "leave image-space gaps"
            )
        return list(range(0, image_extent, tile_stride))

    @staticmethod
    def _global_norm_to_tile_norm(
        global_norm: torch.Tensor,
        projection_crop_box: Sequence[float],
    ) -> torch.Tensor:
        crop = torch.as_tensor(
            projection_crop_box,
            dtype=global_norm.dtype,
            device=global_norm.device,
        )
        if crop.shape != (4,):
            raise ValueError("projection_crop_box must contain four values")
        extent = crop[2:4] - crop[0:2]
        if torch.any(extent <= 0):
            raise ValueError("projection_crop_box has non-positive extent")
        return (global_norm - crop[0:2]) / extent

    @staticmethod
    def _tile_norm_to_global_norm(
        tile_norm: torch.Tensor,
        projection_crop_box: Sequence[float],
    ) -> torch.Tensor:
        crop = torch.as_tensor(
            projection_crop_box,
            dtype=tile_norm.dtype,
            device=tile_norm.device,
        )
        if crop.shape != (4,):
            raise ValueError("projection_crop_box must contain four values")
        extent = crop[2:4] - crop[0:2]
        if torch.any(extent <= 0):
            raise ValueError("projection_crop_box has non-positive extent")
        return tile_norm * extent + crop[0:2]

    @staticmethod
    def _image_tile_token_weights(
        projected_full_norm: torch.Tensor,
        token_indices: torch.Tensor,
        tile_box: Sequence[int],
        hr_size: Sequence[int],
        weight_mode: str,
    ) -> torch.Tensor:
        """Return uniform or separable tent weights in tile image space."""
        if weight_mode not in {"tent", "uniform"}:
            raise ValueError("weight_mode must be 'tent' or 'uniform'")
        if token_indices.ndim != 1:
            raise ValueError("token_indices must be one-dimensional")
        if token_indices.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        if weight_mode == "uniform":
            return torch.ones(token_indices.numel(), dtype=torch.float32)

        x0, y0, x1, y1 = (float(value) for value in tile_box)
        tile_width = x1 - x0
        tile_height = y1 - y0
        if tile_width <= 0 or tile_height <= 0:
            raise ValueError("tile_box must have positive extent")
        hr_width, hr_height = (float(value) for value in hr_size)
        selected = projected_full_norm[token_indices].to(torch.float32)
        # Normalized full-image coordinates are pixel-edge coordinates after
        # the projection's +0.5 convention.
        local_x = (selected[:, 0] * hr_width - x0) / tile_width
        local_y = (selected[:, 1] * hr_height - y0) / tile_height
        weight_x = (1.0 - (2.0 * local_x - 1.0).abs()).clamp_min(1e-3)
        weight_y = (1.0 - (2.0 * local_y - 1.0).abs()).clamp_min(1e-3)
        return (weight_x * weight_y).to(torch.float32)

    @classmethod
    def _build_hr_image_tiles(
        cls,
        hr_image: Image.Image,
        foreground_mask_hr: Image.Image,
        projected_full_norm: torch.Tensor,
        projection_valid: torch.Tensor,
        tile_size: int,
        tile_stride: int,
        min_foreground_ratio: float,
        weight_mode: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build mask-filtered image tiles and global sparse-token indices."""
        if hr_image.width != hr_image.height:
            raise ValueError(f"hr_image must be square, got {hr_image.size}")
        if foreground_mask_hr.size != hr_image.size:
            raise ValueError(
                "foreground_mask_hr must align exactly with hr_image: "
                f"{foreground_mask_hr.size} vs {hr_image.size}"
            )
        if projected_full_norm.ndim != 2 or projected_full_norm.shape[1] != 2:
            raise ValueError(
                "projected_full_norm must have shape [N, 2], got "
                f"{tuple(projected_full_norm.shape)}"
            )
        if projection_valid.shape != (projected_full_norm.shape[0],):
            raise ValueError("projection_valid shape does not match projections")
        if (
            not math.isfinite(min_foreground_ratio)
            or not 0.0 <= min_foreground_ratio <= 1.0
        ):
            raise ValueError("min_foreground_ratio must lie in [0, 1]")
        if weight_mode not in {"tent", "uniform"}:
            raise ValueError("weight_mode must be tent or uniform")

        hr_width, hr_height = hr_image.size
        projection_cpu = projected_full_norm.detach().to(
            device="cpu", dtype=torch.float32
        )
        valid_cpu = projection_valid.detach().to(
            device="cpu", dtype=torch.bool
        )
        finite = torch.isfinite(projection_cpu).all(dim=1)
        pixel_x = torch.floor(projection_cpu[:, 0] * hr_width).to(torch.long)
        pixel_y = torch.floor(projection_cpu[:, 1] * hr_height).to(torch.long)
        in_bounds = (
            (pixel_x >= 0)
            & (pixel_x < hr_width)
            & (pixel_y >= 0)
            & (pixel_y < hr_height)
        )
        mask_array = np.asarray(
            foreground_mask_hr.convert("L"),
            dtype=np.uint8,
        )
        mask_tensor = torch.from_numpy(mask_array > 0)
        mask_valid = torch.zeros_like(in_bounds)
        bounded_indices = in_bounds.nonzero(as_tuple=False).flatten()
        if bounded_indices.numel() > 0:
            mask_valid[bounded_indices] = mask_tensor[
                pixel_y[bounded_indices],
                pixel_x[bounded_indices],
            ]
        eligible = valid_cpu & finite & in_bounds & mask_valid

        starts_x = cls._image_tile_starts(
            hr_width, int(tile_size), int(tile_stride)
        )
        starts_y = cls._image_tile_starts(
            hr_height, int(tile_size), int(tile_stride)
        )
        tiles: List[Dict[str, Any]] = []
        coverage_count = torch.zeros(
            projected_full_norm.shape[0], dtype=torch.int32
        )
        tile_index = 0
        for y0 in starts_y:
            for x0 in starts_x:
                x1 = x0 + int(tile_size)
                y1 = y0 + int(tile_size)
                actual_x1 = min(x1, hr_width)
                actual_y1 = min(y1, hr_height)
                mask_crop = mask_tensor[y0:actual_y1, x0:actual_x1]
                foreground_pixels = int(mask_crop.sum().item())
                foreground_ratio = foreground_pixels / float(tile_size ** 2)
                foreground_enabled = bool(
                    foreground_pixels > 0
                    and foreground_ratio >= min_foreground_ratio
                )
                token_mask = (
                    eligible
                    & (pixel_x >= x0)
                    & (pixel_x < actual_x1)
                    & (pixel_y >= y0)
                    & (pixel_y < actual_y1)
                )
                token_indices = token_mask.nonzero(
                    as_tuple=False
                ).flatten()
                enabled = bool(
                    foreground_enabled and token_indices.numel() > 0
                )
                if enabled:
                    coverage_count.index_add_(
                        0,
                        token_indices,
                        torch.ones_like(token_indices, dtype=torch.int32),
                    )
                    tile_image = Image.new(
                        "RGB",
                        (int(tile_size), int(tile_size)),
                        (0, 0, 0),
                    )
                    tile_image.paste(
                        hr_image.crop((x0, y0, actual_x1, actual_y1)),
                        (0, 0),
                    )
                else:
                    tile_image = None
                normalized_box = (
                    x0 / float(hr_width),
                    y0 / float(hr_height),
                    x1 / float(hr_width),
                    y1 / float(hr_height),
                )
                weights = cls._image_tile_token_weights(
                    projection_cpu,
                    token_indices,
                    (x0, y0, x1, y1),
                    (hr_width, hr_height),
                    weight_mode,
                )
                if token_indices.numel() != weights.numel():
                    raise RuntimeError(
                        f"Tile {tile_index} token/weight count mismatch"
                    )
                tiles.append(
                    {
                        "tile_index": int(tile_index),
                        "box_hr": (int(x0), int(y0), int(x1), int(y1)),
                        "box_hr_actual": (
                            int(x0),
                            int(y0),
                            int(actual_x1),
                            int(actual_y1),
                        ),
                        "projection_crop_box": tuple(
                            float(value) for value in normalized_box
                        ),
                        "foreground_pixels": foreground_pixels,
                        "foreground_ratio": float(foreground_ratio),
                        "foreground_enabled": foreground_enabled,
                        "enabled": enabled,
                        "skipped_reason": (
                            None
                            if enabled
                            else (
                                "foreground_ratio"
                                if not foreground_enabled
                                else "no_sparse_tokens"
                            )
                        ),
                        "token_indices": token_indices,
                        "token_count": int(token_indices.numel()),
                        "weights": weights,
                        "image": tile_image,
                    }
                )
                tile_index += 1

        if min_foreground_ratio == 0.0:
            missed_eligible = eligible & (coverage_count == 0)
            if torch.any(missed_eligible):
                raise RuntimeError(
                    "Image-tile layout failed to assign "
                    f"{int(missed_eligible.sum().item())} projected foreground "
                    "tokens; this indicates a coordinate-mapping error"
                )
        summary = {
            "hr_size": [int(hr_width), int(hr_height)],
            "tile_count": int(len(tiles)),
            "active_tile_count": int(
                sum(bool(tile["enabled"]) for tile in tiles)
            ),
            "eligible_token_count": int(eligible.sum().item()),
            "covered_token_count": int((coverage_count > 0).sum().item()),
            "overlap_token_count": int((coverage_count > 1).sum().item()),
            "eligible_mask": eligible,
            "coverage_count": coverage_count,
            "projected_pixel_x": pixel_x,
            "projected_pixel_y": pixel_y,
        }
        return tiles, summary

    @staticmethod
    def _pack_proj_condition_cpu(
        condition: Mapping[str, Mapping[str, Any]],
        expected_coords: torch.Tensor,
        name: str,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Offload one projected condition without retaining SparseTensor state."""
        packed: Dict[str, Dict[str, torch.Tensor]] = {}
        for branch_name in ("cond", "neg_cond"):
            if branch_name not in condition:
                raise KeyError(f"{name} is missing {branch_name!r}")
            branch = condition[branch_name]
            global_value = branch.get("global")
            proj_value = branch.get("proj")
            if not isinstance(global_value, torch.Tensor):
                raise TypeError(
                    f"{name}.{branch_name}.global must be a tensor"
                )
            if not isinstance(proj_value, SparseTensor):
                raise TypeError(
                    f"{name}.{branch_name}.proj must be a SparseTensor"
                )
            if proj_value.feats.shape[0] != expected_coords.shape[0]:
                raise RuntimeError(
                    f"{name}.{branch_name}.proj token count is not aligned"
                )
            if not torch.equal(proj_value.coords, expected_coords):
                raise RuntimeError(
                    f"{name}.{branch_name}.proj coordinate order is not aligned"
                )
            packed[branch_name] = {
                "global": global_value.detach().to(
                    device="cpu", copy=True
                ),
                "proj": proj_value.feats.detach().to(
                    device="cpu", copy=True
                ),
            }
        if not torch.count_nonzero(packed["neg_cond"]["global"]) == 0:
            raise RuntimeError(f"{name} negative global condition is not zero")
        if not torch.count_nonzero(packed["neg_cond"]["proj"]) == 0:
            raise RuntimeError(f"{name} negative proj condition is not zero")
        return packed

    @staticmethod
    def _materialize_proj_condition(
        packed: Mapping[str, Mapping[str, torch.Tensor]],
        coords: torch.Tensor,
        device: Union[str, torch.device],
    ) -> Dict[str, Dict[str, Any]]:
        """Move a packed condition to a device in the exact supplied order."""
        output: Dict[str, Dict[str, Any]] = {}
        for branch_name in ("cond", "neg_cond"):
            branch = packed[branch_name]
            proj_features = branch["proj"].to(device=device)
            if proj_features.shape[0] != coords.shape[0]:
                raise RuntimeError(
                    f"{branch_name} projected condition token mismatch"
                )
            output[branch_name] = {
                "global": branch["global"].to(device=device),
                "proj": SparseTensor(
                    feats=proj_features,
                    coords=coords,
                ),
            }
        return output

    @staticmethod
    def _scatter_add_tile_velocity(
        velocity_sum: torch.Tensor,
        weight_sum: torch.Tensor,
        coverage_count: torch.Tensor,
        token_indices: torch.Tensor,
        tile_velocity: torch.Tensor,
        token_weights: torch.Tensor,
    ) -> None:
        """Accumulate one tile without changing global token order."""
        if token_indices.ndim != 1:
            raise ValueError("token_indices must be one-dimensional")
        if tile_velocity.shape[0] != token_indices.shape[0]:
            raise ValueError("tile velocity/token count mismatch")
        if token_weights.shape != (token_indices.shape[0],):
            raise ValueError("tile weight/token count mismatch")
        if tile_velocity.shape[1:] != velocity_sum.shape[1:]:
            raise ValueError("tile/global velocity channel mismatch")
        if torch.any(token_indices < 0) or torch.any(
            token_indices >= velocity_sum.shape[0]
        ):
            raise IndexError("tile token index lies outside the global latent")
        weights = token_weights.to(
            device=velocity_sum.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        velocity_sum.index_add_(
            0,
            token_indices,
            tile_velocity.to(torch.float32) * weights,
        )
        weight_sum.index_add_(0, token_indices, weights)
        coverage_count.index_add_(
            0,
            token_indices,
            torch.ones_like(token_indices, dtype=coverage_count.dtype),
        )

    @staticmethod
    def _finalize_tile_velocity(
        velocity_sum: torch.Tensor,
        weight_sum: torch.Tensor,
        fallback_velocity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize overlaps and fill uncovered tokens from global velocity."""
        if velocity_sum.shape != fallback_velocity.shape:
            raise ValueError("fallback velocity shape does not match global sum")
        if weight_sum.shape != (velocity_sum.shape[0], 1):
            raise ValueError("weight_sum has the wrong shape")
        covered = weight_sum[:, 0] > 0
        merged = velocity_sum / weight_sum.clamp_min(
            torch.finfo(torch.float32).eps
        )
        merged[~covered] = fallback_velocity[~covered].to(torch.float32)
        return merged, covered

    @staticmethod
    def _tile_trace_metadata(
        tile: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "tile_index": int(tile["tile_index"]),
            "box_hr": list(tile["box_hr"]),
            "box_hr_actual": list(tile["box_hr_actual"]),
            "projection_crop_box": list(tile["projection_crop_box"]),
            "foreground_pixels": int(tile["foreground_pixels"]),
            "foreground_ratio": float(tile["foreground_ratio"]),
            "foreground_enabled": bool(tile["foreground_enabled"]),
            "enabled": bool(tile["enabled"]),
            "skipped_reason": tile["skipped_reason"],
            "token_count": int(tile["token_count"]),
            "tile_image_path": tile.get("tile_image_path"),
        }

    @classmethod
    def _save_hr_image_tile_debug(
        cls,
        debug_dir: Union[str, Path],
        global_image: Image.Image,
        hr_image: Image.Image,
        foreground_mask_hr: Image.Image,
        projected_full_norm: torch.Tensor,
        projection_valid: torch.Tensor,
        tiles: List[Dict[str, Any]],
        global_to_hr_transform: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Save projection overlays, tile crops, mask, and JSON metadata."""
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        global_vis = global_image.convert("RGB").copy()
        global_draw = ImageDraw.Draw(global_vis)
        projection_cpu = projected_full_norm.detach().to(
            device="cpu", dtype=torch.float32
        )
        valid_cpu = projection_valid.detach().to(
            device="cpu", dtype=torch.bool
        )
        for point, valid in zip(projection_cpu.tolist(), valid_cpu.tolist()):
            x = point[0] * global_vis.width - 0.5
            y = point[1] * global_vis.height - 0.5
            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or x < 0
                or x >= global_vis.width
                or y < 0
                or y >= global_vis.height
            ):
                continue
            global_draw.ellipse(
                (x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                fill=((255, 220, 0) if valid else (140, 140, 140)),
            )
        global_projection_path = debug_path / "projection_global.png"
        global_vis.save(global_projection_path)

        hr_vis = hr_image.convert("RGB").copy()
        hr_draw = ImageDraw.Draw(hr_vis)
        palette = (
            (255, 80, 80),
            (80, 220, 120),
            (80, 140, 255),
            (255, 190, 60),
            (190, 90, 255),
            (50, 220, 220),
            (255, 100, 190),
            (170, 220, 50),
        )
        tile_dir = debug_path / "tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)
        for point in projection_cpu.tolist():
            x = point[0] * hr_vis.width - 0.5
            y = point[1] * hr_vis.height - 0.5
            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or x < 0
                or x >= hr_vis.width
                or y < 0
                or y >= hr_vis.height
            ):
                continue
            hr_draw.ellipse(
                (x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                fill=(135, 135, 135),
            )
        for tile in tiles:
            color = palette[int(tile["tile_index"]) % len(palette)]
            x0, y0, x1, y1 = tile["box_hr"]
            hr_draw.rectangle(
                (x0, y0, min(x1, hr_vis.width) - 1, min(y1, hr_vis.height) - 1),
                outline=color,
                width=max(1, hr_vis.width // 1024),
            )
            hr_draw.text(
                (x0 + 4, y0 + 4),
                f"{tile['tile_index']}:{tile['token_count']}",
                fill=color,
            )
            indices = tile["token_indices"]
            for point in projection_cpu[indices].tolist():
                x = point[0] * hr_vis.width - 0.5
                y = point[1] * hr_vis.height - 0.5
                hr_draw.ellipse(
                    (x - 2, y - 2, x + 2, y + 2),
                    fill=color,
                )

            tile_canvas = Image.new(
                "RGB",
                (
                    int(tile["box_hr"][2] - tile["box_hr"][0]),
                    int(tile["box_hr"][3] - tile["box_hr"][1]),
                ),
                (0, 0, 0),
            )
            ax0, ay0, ax1, ay1 = tile["box_hr_actual"]
            tile_canvas.paste(
                hr_image.crop((ax0, ay0, ax1, ay1)),
                (0, 0),
            )
            tile_path = tile_dir / f"tile_{int(tile['tile_index']):04d}.png"
            tile_canvas.save(tile_path)
            tile["tile_image_path"] = str(tile_path.resolve())

        hr_projection_path = debug_path / "projection_hr_tiles.png"
        hr_vis.save(hr_projection_path)
        mask_path = debug_path / "foreground_mask_hr.png"
        foreground_mask_hr.save(mask_path)
        metadata_path = debug_path / "tile_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "global_to_hr_transform": dict(
                        global_to_hr_transform
                    ),
                    "tiles": [
                        cls._tile_trace_metadata(tile) for tile in tiles
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "debug_dir": str(debug_path.resolve()),
            "projection_global": str(global_projection_path.resolve()),
            "projection_hr_tiles": str(hr_projection_path.resolve()),
            "foreground_mask_hr": str(mask_path.resolve()),
            "tile_metadata": str(metadata_path.resolve()),
            "tile_directory": str(tile_dir.resolve()),
        }

    _HAAR_3D_BAND_NAMES = (
        "LLL",
        "LLH",
        "LHL",
        "LHH",
        "HLL",
        "HLH",
        "HHL",
        "HHH",
    )

    @staticmethod
    def _haar_analysis_axis(
        value: torch.Tensor,
        axis: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        moved = value.movedim(axis, -1)
        if moved.shape[-1] % 2 != 0:
            raise ValueError(
                f"Haar DWT axis length must be even, got {moved.shape[-1]}"
            )
        even = moved[..., 0::2]
        odd = moved[..., 1::2]
        scale = math.sqrt(2.0)
        low = (even + odd) / scale
        high = (even - odd) / scale
        return low.movedim(-1, axis), high.movedim(-1, axis)

    @staticmethod
    def _haar_synthesis_axis(
        low: torch.Tensor,
        high: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        if low.shape != high.shape:
            raise ValueError(
                f"Haar low/high shapes differ: {low.shape} vs {high.shape}"
            )
        low_moved = low.movedim(axis, -1)
        high_moved = high.movedim(axis, -1)
        scale = math.sqrt(2.0)
        even = (low_moved + high_moved) / scale
        odd = (low_moved - high_moved) / scale
        output = torch.empty(
            (*low_moved.shape[:-1], low_moved.shape[-1] * 2),
            dtype=low.dtype,
            device=low.device,
        )
        output[..., 0::2] = even
        output[..., 1::2] = odd
        return output.movedim(-1, axis)

    @classmethod
    def _haar_dwt3d(cls, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One-level orthonormal 3D Haar DWT for [B,C,X,Y,Z]."""
        if value.ndim != 5:
            raise ValueError(f"3D DWT expects [B,C,X,Y,Z], got {value.shape}")
        x_low, x_high = cls._haar_analysis_axis(value, 2)
        ll, lh = cls._haar_analysis_axis(x_low, 3)
        hl, hh = cls._haar_analysis_axis(x_high, 3)
        lll, llh = cls._haar_analysis_axis(ll, 4)
        lhl, lhh = cls._haar_analysis_axis(lh, 4)
        hll, hlh = cls._haar_analysis_axis(hl, 4)
        hhl, hhh = cls._haar_analysis_axis(hh, 4)
        return {
            "LLL": lll,
            "LLH": llh,
            "LHL": lhl,
            "LHH": lhh,
            "HLL": hll,
            "HLH": hlh,
            "HHL": hhl,
            "HHH": hhh,
        }

    @classmethod
    def _haar_idwt3d(cls, bands: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Inverse of :meth:`_haar_dwt3d`."""
        missing = set(cls._HAAR_3D_BAND_NAMES) - set(bands)
        if missing:
            raise ValueError(f"Missing Haar bands: {sorted(missing)}")
        ll = cls._haar_synthesis_axis(bands["LLL"], bands["LLH"], 4)
        lh = cls._haar_synthesis_axis(bands["LHL"], bands["LHH"], 4)
        hl = cls._haar_synthesis_axis(bands["HLL"], bands["HLH"], 4)
        hh = cls._haar_synthesis_axis(bands["HHL"], bands["HHH"], 4)
        x_low = cls._haar_synthesis_axis(ll, lh, 3)
        x_high = cls._haar_synthesis_axis(hl, hh, 3)
        return cls._haar_synthesis_axis(x_low, x_high, 2)

    @classmethod
    def _dwt3d(
        cls,
        value: torch.Tensor,
        wavelet_family: str,
    ) -> Dict[str, torch.Tensor]:
        if wavelet_family == "haar":
            return cls._haar_dwt3d(value)
        if wavelet_family == "sym4":
            raise NotImplementedError(
                "sym4 is reserved for a later implementation; use haar"
            )
        raise ValueError(f"Unsupported wavelet family: {wavelet_family}")

    @classmethod
    def _idwt3d(
        cls,
        bands: Mapping[str, torch.Tensor],
        wavelet_family: str,
    ) -> torch.Tensor:
        if wavelet_family == "haar":
            return cls._haar_idwt3d(bands)
        if wavelet_family == "sym4":
            raise NotImplementedError(
                "sym4 is reserved for a later implementation; use haar"
            )
        raise ValueError(f"Unsupported wavelet family: {wavelet_family}")

    @staticmethod
    def _sparse_patch_velocity_to_dense(
        velocity: SparseTensor,
        local_coords: torch.Tensor,
        patch_size: int = 64,
    ) -> torch.Tensor:
        """Scatter [N,C] sparse velocity into an FP32 dense patch."""
        if velocity.feats.shape[0] != local_coords.shape[0]:
            raise ValueError("Velocity and coordinate token counts differ")
        xyz = local_coords[:, 1:4].long()
        if torch.any(xyz < 0) or torch.any(xyz >= patch_size):
            raise ValueError("Patch-local coordinates must lie in [0, 63]")
        channels = int(velocity.feats.shape[1])
        # Model predictions may be FP16/BF16.  Haar analysis, frequency CFG,
        # synthesis, and sparse gathering deliberately run in FP32 so their
        # numerical error does not perturb the subsequent Euler trajectory.
        dense_flat = torch.zeros(
            (channels, patch_size ** 3),
            device=velocity.feats.device,
            dtype=torch.float32,
        )
        linear = (xyz[:, 0] * patch_size + xyz[:, 1]) * patch_size + xyz[:, 2]
        dense_flat[:, linear] = velocity.feats.float().transpose(0, 1)
        return dense_flat.reshape(
            1, channels, patch_size, patch_size, patch_size
        )

    @staticmethod
    def _gather_dense_patch_velocity(
        dense: torch.Tensor,
        local_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Gather [1,C,64,64,64] back into local sparse token order."""
        if dense.ndim != 5 or dense.shape[0] != 1:
            raise ValueError(f"Expected dense [1,C,X,Y,Z], got {dense.shape}")
        patch_size = int(dense.shape[2])
        if tuple(dense.shape[2:]) != (patch_size, patch_size, patch_size):
            raise ValueError("Dense patch must be cubic")
        xyz = local_coords[:, 1:4].long()
        linear = (xyz[:, 0] * patch_size + xyz[:, 1]) * patch_size + xyz[:, 2]
        dense_flat = dense.reshape(dense.shape[1], -1)
        return dense_flat[:, linear].transpose(0, 1).contiguous()

    @classmethod
    def _check_3d_wavelet_implementation(
        cls,
        device: Union[str, torch.device],
        wavelet_family: str,
    ) -> Dict[str, Any]:
        """Check DWT round-trip and uniform-band CFG equivalence."""
        if wavelet_family != "haar":
            raise NotImplementedError(
                "Only haar is implemented; sym4 remains a reserved interface"
            )
        values = torch.linspace(
            -1.0,
            1.0,
            steps=2 * 8 * 8 * 8,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 2, 8, 8, 8)
        conditional = values
        unconditional = values.flip(dims=(2, 3, 4)) * 0.75
        cond_bands = cls._dwt3d(conditional, wavelet_family)
        reconstruction = cls._idwt3d(cond_bands, wavelet_family)
        round_trip_error = float(
            (reconstruction - conditional).abs().max().item()
        )

        strength = 2.75
        uncond_bands = cls._dwt3d(unconditional, wavelet_family)
        uniform_bands = {
            name: uncond_bands[name]
            + strength * (cond_bands[name] - uncond_bands[name])
            for name in cls._HAAR_3D_BAND_NAMES
        }
        uniform_reconstruction = cls._idwt3d(
            uniform_bands, wavelet_family
        )
        spatial_cfg = unconditional + strength * (
            conditional - unconditional
        )
        uniform_cfg_error = float(
            (uniform_reconstruction - spatial_cfg).abs().max().item()
        )
        tolerance = 2e-6
        if round_trip_error > tolerance or uniform_cfg_error > tolerance:
            raise RuntimeError(
                "3D wavelet self-check failed: "
                f"round_trip={round_trip_error:.8e}, "
                f"uniform_cfg={uniform_cfg_error:.8e}, "
                f"tolerance={tolerance:.8e}"
            )
        return {
            "round_trip_max_abs_error": round_trip_error,
            "uniform_band_cfg_max_abs_error": uniform_cfg_error,
            "tolerance": tolerance,
        }

    @torch.no_grad()
    def _run_2048_patch_flow(
        self,
        flow_model: nn.Module,
        sampler: Any,
        stage_name: str,
        global_noise: SparseTensor,
        cond: Dict[str, Any],
        concat_cond: Optional[SparseTensor],
        sampler_params: Dict[str, Any],
        global_flow: Any,
        start_step: int,
        start_source: str,
        patch_guidance_mode: str = "original_cfg",
        guidance_strength: float = 7.5,
        guidance_interval: str = "original_interval",
        guidance_rescale: float = 0.0,
        wavelet_family: str = "haar",
        skip_residual_mode: str = "off",
    ) -> Tuple[SparseTensor, Dict[str, Any], Dict[str, Any]]:
        """Resume a sparse shape/texture flow with local spatial predictions.

        Texture ``concat_cond`` is sliced with the latent but is deliberately
        shared by the positive and negative image branches. Skip residuals and
        sym4 remain disabled interfaces.
        """
        if stage_name not in {"shape", "texture"}:
            raise ValueError("stage_name must be shape or texture")
        trajectory = global_flow.trajectory
        if trajectory is None:
            raise RuntimeError(
                f"The global {stage_name} flow did not record a trajectory"
            )
        num_steps = len(trajectory.velocities)
        if num_steps != 12:
            raise ValueError(
                f"The 2048 experiment requires exactly 12 steps, got {num_steps}"
            )
        if not 0 <= start_step <= num_steps:
            raise ValueError(
                f"start_step must be in [0, {num_steps}], got {start_step}"
            )
        if start_source not in {"saved_state", "algebraic_inverse"}:
            raise ValueError(
                "start_source must be 'saved_state' or 'algebraic_inverse'"
            )
        if patch_guidance_mode not in {
            "original_cfg",
            "conditional_only",
            "uniform_cfg",
            "wavelet_cfg",
        }:
            raise ValueError(
                "patch_guidance_mode must be original_cfg, conditional_only, "
                "uniform_cfg, or wavelet_cfg"
            )
        if guidance_interval not in {
            "original_interval",
            "all_remaining",
        }:
            raise ValueError(
                "guidance_interval must be original_interval or "
                "all_remaining"
            )
        if not math.isfinite(guidance_strength):
            raise ValueError("guidance_strength must be finite")
        if not math.isfinite(guidance_rescale) or not 0.0 <= guidance_rescale <= 1.0:
            raise ValueError("guidance_rescale must be finite and in [0, 1]")
        if skip_residual_mode != "off":
            raise NotImplementedError(
                "Skip residual is reserved but not implemented in this pass"
            )
        wavelet_checks = None
        if patch_guidance_mode == "wavelet_cfg":
            wavelet_checks = self._check_3d_wavelet_implementation(
                global_noise.device,
                wavelet_family,
            )
        real_patch_wavelet_check: Optional[Dict[str, Any]] = None

        inverse_state = sampler.invert_euler_trajectory(
            trajectory.states[-1],
            trajectory.velocities,
            trajectory.time_intervals,
            start_step,
        )
        saved_state = trajectory.states[start_step]
        inverse_difference = (
            inverse_state.to(torch.float32) - saved_state.to(torch.float32)
        ).abs()
        inverse_check = {
            "max_abs_error": float(inverse_difference.max().item()),
            "mean_abs_error": float(inverse_difference.mean().item()),
        }
        selected_state = (
            saved_state if start_source == "saved_state" else inverse_state
        )
        patch_state = global_noise.replace(
            selected_state.to(
                device=global_noise.device,
                dtype=global_noise.dtype,
                copy=True,
            )
        )

        patches = self._build_2048_overlap_patches(patch_state.coords)
        active_patches = [
            patch for patch in patches if patch["token_indices"].numel() > 0
        ]
        if not active_patches:
            raise RuntimeError("All 27 spatial patches are empty")
        patch_records = [
            {
                "patch_index": int(patch["patch_index"]),
                "start": list(patch["start"]),
                "end": list(patch["end"]),
                "token_count": int(patch["token_indices"].numel()),
                "skipped_empty": patch["token_indices"].numel() == 0,
                "coordinate_mode": "local",
                "local_coord_min": (
                    patch["local_coords"][:, 1:4]
                    .amin(dim=0)
                    .detach()
                    .cpu()
                    .tolist()
                    if patch["local_coords"] is not None
                    else None
                ),
                "local_coord_max": (
                    patch["local_coords"][:, 1:4]
                    .amax(dim=0)
                    .detach()
                    .cpu()
                    .tolist()
                    if patch["local_coords"] is not None
                    else None
                ),
            }
            for patch in patches
        ]
        print(
            f"[{stage_name}-patch-flow] start_step={start_step}/{num_steps} "
            f"source={start_source} active_patches={len(active_patches)}/27 "
            f"tokens={patch_state.feats.shape[0]:,}"
        )
        print(
            f"[{stage_name}-patch-flow-coordinates] patch_coordinate_mode=local "
            f"validated_nonempty_patches={len(active_patches)}/27 "
            "model_local_range=[0,63] global_latent_coords=preserved"
        )
        print(
            f"[{stage_name}-patch-flow-guidance] mode={patch_guidance_mode} "
            f"wavelet_family={wavelet_family} "
            f"strength={guidance_strength:.6g} "
            f"interval={guidance_interval} "
            f"rescale={guidance_rescale:.6g} "
            f"skip_residual={skip_residual_mode}"
        )
        if wavelet_checks is not None:
            print(
                f"[{stage_name}-patch-flow-wavelet-check] "
                f"round_trip_max_abs="
                f"{wavelet_checks['round_trip_max_abs_error']:.8e} "
                f"uniform_cfg_max_abs="
                f"{wavelet_checks['uniform_band_cfg_max_abs_error']:.8e}"
            )
        print(
            f"[{stage_name}-patch-flow] algebraic inverse check: "
            f"max_abs={inverse_check['max_abs_error']:.8e} "
            f"mean_abs={inverse_check['mean_abs_error']:.8e}"
        )

        prediction_kwargs = {
            key: value
            for key, value in sampler_params.items()
            if key
            not in {
                "steps",
                "rescale_t",
                "verbose",
                "tqdm_desc",
                "record_trajectory",
                "trajectory_device",
                "return_model_history",
            }
        }
        raw_prediction_kwargs = {
            key: value
            for key, value in prediction_kwargs.items()
            if key
            not in {
                "guidance_strength",
                "guidance_rescale",
                "guidance_interval",
            }
        }
        original_guidance_strength = float(
            prediction_kwargs.get("guidance_strength", 1.0)
        )
        original_guidance_rescale = float(
            prediction_kwargs.get("guidance_rescale", 0.0)
        )
        original_guidance_interval = tuple(
            prediction_kwargs.get("guidance_interval", (0.0, 1.0))
        )
        similarities: List[Dict[str, Any]] = []
        merged_velocities: List[torch.Tensor] = []
        token_count = int(patch_state.feats.shape[0])
        for step_index in range(start_step, num_steps):
            t = float(trajectory.times[step_index])
            t_prev = float(trajectory.times[step_index + 1])
            time_interval = float(trajectory.time_intervals[step_index])
            original_interval_active = (
                original_guidance_interval[0]
                <= t
                <= original_guidance_interval[1]
            )
            guidance_step_active = (
                patch_guidance_mode in {"uniform_cfg", "wavelet_cfg"}
                and (
                    guidance_interval == "all_remaining"
                    or original_interval_active
                )
            )
            velocity_sum = torch.zeros(
                patch_state.feats.shape,
                device=patch_state.device,
                dtype=torch.float32,
            )
            plain_velocity_sum = (
                torch.zeros_like(velocity_sum)
                if patch_guidance_mode in {"uniform_cfg", "wavelet_cfg"}
                else None
            )
            weight_sum = torch.zeros(
                (token_count, 1),
                device=patch_state.device,
                dtype=torch.float32,
            )
            conditional_band_energy_sum = (
                torch.zeros(
                    len(self._HAAR_3D_BAND_NAMES),
                    device=patch_state.device,
                    dtype=torch.float64,
                )
                if patch_guidance_mode == "wavelet_cfg"
                else None
            )
            unconditional_band_energy_sum = (
                torch.zeros_like(conditional_band_energy_sum)
                if conditional_band_energy_sum is not None
                else None
            )
            guided_band_energy_sum = (
                torch.zeros_like(conditional_band_energy_sum)
                if conditional_band_energy_sum is not None
                else None
            )

            for patch in active_patches:
                token_indices = patch["token_indices"]
                patch_local_coords = patch["local_coords"]
                if patch_local_coords is None:
                    raise RuntimeError(
                        f"Active patch {patch['patch_index']} has no local coords"
                    )
                patch_input = SparseTensor(
                    feats=patch_state.feats[token_indices],
                    coords=patch_local_coords,
                )
                patch_cond = self._slice_sparse_condition(
                    cond["cond"],
                    token_indices,
                    patch_local_coords,
                    token_count,
                    patch_state.coords,
                )
                patch_neg_cond = self._slice_sparse_condition(
                    cond["neg_cond"],
                    token_indices,
                    patch_local_coords,
                    token_count,
                    patch_state.coords,
                )
                patch_concat_cond = self._slice_aligned_sparse_tensor(
                    concat_cond,
                    token_indices,
                    patch_local_coords,
                    patch_state.coords,
                    name=f"{stage_name} concat_cond",
                )
                patch_raw_kwargs = dict(raw_prediction_kwargs)
                patch_prediction_kwargs = dict(prediction_kwargs)
                if patch_concat_cond is not None:
                    # The identical object is supplied to both raw CFG calls;
                    # only patch_cond versus patch_neg_cond changes.
                    patch_raw_kwargs["concat_cond"] = patch_concat_cond
                    patch_prediction_kwargs["concat_cond"] = patch_concat_cond
                if patch_guidance_mode == "original_cfg":
                    # Keep the baseline path byte-for-byte equivalent: the
                    # sampler still owns interval CFG and guidance rescale.
                    _, _, patch_velocity = (
                        sampler._get_model_prediction(
                            flow_model,
                            patch_input,
                            t,
                            patch_cond,
                            neg_cond=patch_neg_cond,
                            **patch_prediction_kwargs,
                        )
                    )
                    plain_patch_velocity = patch_velocity
                elif patch_guidance_mode == "conditional_only":
                    conditional_kwargs = dict(patch_raw_kwargs)
                    conditional_kwargs.update(
                        {
                            "guidance_strength": 1.0,
                            "guidance_rescale": 0.0,
                            "guidance_interval": (0.0, 1.0),
                        }
                    )
                    _, _, patch_velocity = sampler._get_model_prediction(
                        flow_model,
                        patch_input,
                        t,
                        patch_cond,
                        neg_cond=patch_neg_cond,
                        **conditional_kwargs,
                    )
                    plain_patch_velocity = patch_velocity
                else:
                    pred_cond, pred_uncond = (
                        sampler._inference_model_cfg_pair(
                            flow_model,
                            patch_input,
                            t,
                            patch_cond,
                            patch_neg_cond,
                            **patch_raw_kwargs,
                        )
                    )
                    plain_strength = (
                        original_guidance_strength
                        if original_interval_active
                        else 1.0
                    )
                    plain_patch_velocity = (
                        sampler._combine_cfg_predictions(
                            patch_input,
                            t,
                            pred_cond,
                            pred_uncond,
                            guidance_strength=plain_strength,
                            guidance_rescale=(
                                original_guidance_rescale
                                if original_interval_active
                                else 0.0
                            ),
                        )
                    )
                    if patch_guidance_mode == "uniform_cfg":
                        patch_velocity = sampler._combine_cfg_predictions(
                            patch_input,
                            t,
                            pred_cond,
                            pred_uncond,
                            guidance_strength=(
                                float(guidance_strength)
                                if guidance_step_active
                                else 1.0
                            ),
                            guidance_rescale=(
                                float(guidance_rescale)
                                if guidance_step_active
                                else 0.0
                            ),
                        )
                    else:
                        cond_dense = self._sparse_patch_velocity_to_dense(
                            pred_cond,
                            patch_local_coords,
                        )
                        uncond_dense = self._sparse_patch_velocity_to_dense(
                            pred_uncond,
                            patch_local_coords,
                        )
                        cond_bands = self._dwt3d(cond_dense, wavelet_family)
                        uncond_bands = self._dwt3d(
                            uncond_dense, wavelet_family
                        )
                        if real_patch_wavelet_check is None:
                            if cond_dense.dtype != torch.float32:
                                raise RuntimeError(
                                    "Wavelet dense velocity must be FP32, got "
                                    f"{cond_dense.dtype}"
                                )
                            roundtrip_dense = self._idwt3d(
                                cond_bands, wavelet_family
                            )
                            roundtrip_sparse = (
                                self._gather_dense_patch_velocity(
                                    roundtrip_dense,
                                    patch_local_coords,
                                )
                            )
                            roundtrip_max_abs = float(
                                (
                                    roundtrip_sparse
                                    - pred_cond.feats.float()
                                )
                                .abs()
                                .max()
                                .item()
                            )
                            real_patch_tolerance = 1e-5
                            real_patch_wavelet_check = {
                                "step_index": int(step_index),
                                "patch_index": int(patch["patch_index"]),
                                "token_count": int(
                                    patch_local_coords.shape[0]
                                ),
                                "model_prediction_dtype": str(
                                    pred_cond.feats.dtype
                                ).replace("torch.", ""),
                                "wavelet_compute_dtype": str(
                                    cond_dense.dtype
                                ).replace("torch.", ""),
                                "round_trip_max_abs_error": roundtrip_max_abs,
                                "tolerance_exclusive": real_patch_tolerance,
                            }
                            if (
                                not math.isfinite(roundtrip_max_abs)
                                or roundtrip_max_abs >= real_patch_tolerance
                            ):
                                raise RuntimeError(
                                    "Real-patch FP32 wavelet round-trip failed: "
                                    f"max_abs={roundtrip_max_abs:.8e}, "
                                    f"required<{real_patch_tolerance:.8e}"
                                )
                            if wavelet_checks is None:
                                raise RuntimeError(
                                    "Wavelet checks were not initialized"
                                )
                            wavelet_checks["real_patch_sparse_round_trip"] = (
                                real_patch_wavelet_check
                            )
                            print(
                                f"[{stage_name}-patch-flow-wavelet-real-patch-check] "
                                f"step={step_index:02d} "
                                f"patch={patch['patch_index']} "
                                f"model_dtype="
                                f"{real_patch_wavelet_check['model_prediction_dtype']} "
                                "wavelet_dtype=float32 "
                                f"max_abs={roundtrip_max_abs:.8e} "
                                f"required<{real_patch_tolerance:.8e}"
                            )
                            del roundtrip_dense, roundtrip_sparse
                        if guidance_step_active:
                            guided_bands = {"LLL": cond_bands["LLL"]}
                            for band_name in self._HAAR_3D_BAND_NAMES[1:]:
                                guided_bands[band_name] = (
                                    uncond_bands[band_name]
                                    + float(guidance_strength)
                                    * (
                                        cond_bands[band_name]
                                        - uncond_bands[band_name]
                                    )
                                )
                            guided_dense = self._idwt3d(
                                guided_bands, wavelet_family
                            )
                            patch_velocity = patch_input.replace(
                                self._gather_dense_patch_velocity(
                                    guided_dense,
                                    patch_local_coords,
                                ).to(patch_input.dtype)
                            )
                            if guidance_rescale > 0.0:
                                patch_velocity = sampler._rescale_cfg_prediction(
                                    patch_input,
                                    t,
                                    pred_cond,
                                    patch_velocity,
                                    guidance_rescale=float(guidance_rescale),
                                )
                        else:
                            guided_bands = cond_bands
                            guided_dense = None
                            patch_velocity = pred_cond

                        conditional_band_energy_sum += torch.stack(
                            [
                                cond_bands[name]
                                .to(torch.float32)
                                .square()
                                .mean()
                                .to(torch.float64)
                                for name in self._HAAR_3D_BAND_NAMES
                            ]
                        )
                        unconditional_band_energy_sum += torch.stack(
                            [
                                uncond_bands[name]
                                .to(torch.float32)
                                .square()
                                .mean()
                                .to(torch.float64)
                                for name in self._HAAR_3D_BAND_NAMES
                            ]
                        )
                        guided_band_energy_sum += torch.stack(
                            [
                                guided_bands[name]
                                .to(torch.float32)
                                .square()
                                .mean()
                                .to(torch.float64)
                                for name in self._HAAR_3D_BAND_NAMES
                            ]
                        )
                weights = patch["weights"].unsqueeze(1)
                velocity_sum.index_add_(
                    0,
                    token_indices,
                    patch_velocity.feats.to(torch.float32) * weights,
                )
                if plain_velocity_sum is not None:
                    plain_velocity_sum.index_add_(
                        0,
                        token_indices,
                        plain_patch_velocity.feats.to(torch.float32) * weights,
                    )
                weight_sum.index_add_(0, token_indices, weights)
                del (
                    patch_input,
                    patch_cond,
                    patch_neg_cond,
                    patch_concat_cond,
                    patch_raw_kwargs,
                    patch_prediction_kwargs,
                    patch_velocity,
                )
                if patch_guidance_mode == "wavelet_cfg":
                    del (
                        pred_cond,
                        pred_uncond,
                        plain_patch_velocity,
                        cond_dense,
                        uncond_dense,
                        cond_bands,
                        uncond_bands,
                        guided_bands,
                        guided_dense,
                    )

            if torch.any(weight_sum <= 0):
                uncovered = int((weight_sum <= 0).sum().item())
                raise RuntimeError(
                    f"Patch layout left {uncovered} sparse tokens uncovered"
                )
            merged_velocity = velocity_sum / weight_sum
            merged_velocity_cpu = merged_velocity.detach().cpu()
            if plain_velocity_sum is None:
                plain_merged_velocity = merged_velocity
            else:
                plain_merged_velocity = plain_velocity_sum / weight_sum
            plain_merged_velocity_cpu = plain_merged_velocity.detach().cpu()
            global_velocity_cpu = trajectory.velocities[step_index]
            metrics = self._velocity_similarity(
                merged_velocity_cpu,
                global_velocity_cpu,
            )
            similarity_to_plain = self._velocity_similarity(
                merged_velocity_cpu,
                plain_merged_velocity_cpu,
            )
            plain_similarity_to_global = self._velocity_similarity(
                plain_merged_velocity_cpu,
                global_velocity_cpu,
            )
            if patch_guidance_mode == "wavelet_cfg":
                divisor = float(len(active_patches))
                conditional_energy_values = (
                    conditional_band_energy_sum / divisor
                ).detach().cpu().tolist()
                unconditional_energy_values = (
                    unconditional_band_energy_sum / divisor
                ).detach().cpu().tolist()
                guided_energy_values = (
                    guided_band_energy_sum / divisor
                ).detach().cpu().tolist()
                conditional_energy = dict(
                    zip(self._HAAR_3D_BAND_NAMES, conditional_energy_values)
                )
                unconditional_energy = dict(
                    zip(self._HAAR_3D_BAND_NAMES, unconditional_energy_values)
                )
                guided_energy = dict(
                    zip(self._HAAR_3D_BAND_NAMES, guided_energy_values)
                )
                conditional_high_energy = sum(
                    conditional_energy[name]
                    for name in self._HAAR_3D_BAND_NAMES[1:]
                )
                guided_high_energy = sum(
                    guided_energy[name]
                    for name in self._HAAR_3D_BAND_NAMES[1:]
                )
                high_energy_ratio = guided_high_energy / max(
                    conditional_high_energy,
                    torch.finfo(torch.float64).eps,
                )
                frequency_metrics = {
                    "enabled": True,
                    "active_this_step": bool(guidance_step_active),
                    "low_frequency_source": "conditional",
                    "low_frequency_strict_after_rescale": bool(
                        not guidance_step_active or guidance_rescale == 0.0
                    ),
                    "guidance_rescale_applied": bool(
                        guidance_step_active and guidance_rescale > 0.0
                    ),
                    "band_energy_definition": (
                        "mean_squared_coefficient_averaged_over_active_patches"
                    ),
                    "conditional_band_energy": conditional_energy,
                    "unconditional_band_energy": unconditional_energy,
                    "guided_band_energy_pre_rescale": guided_energy,
                    "high_frequency_energy_ratio": float(high_energy_ratio),
                    "high_frequency_rms_amplification": float(
                        math.sqrt(max(high_energy_ratio, 0.0))
                    ),
                }
            else:
                frequency_metrics = {
                    "enabled": False,
                    "active_this_step": False,
                }
            metrics.update(
                {
                    "step_index": int(step_index),
                    "t": t,
                    "t_prev": t_prev,
                    "time_interval": time_interval,
                    "guidance_active_this_step": bool(guidance_step_active),
                    "similarity_to_plain_patch_velocity": (
                        similarity_to_plain
                    ),
                    "plain_patch_similarity_to_global_velocity": (
                        plain_similarity_to_global
                    ),
                    "velocity_norm": {
                        "merged": float(merged_velocity_cpu.float().norm().item()),
                        "plain_patch": float(
                            plain_merged_velocity_cpu.float().norm().item()
                        ),
                        "global": float(
                            global_velocity_cpu.float().norm().item()
                        ),
                    },
                    "frequency_guidance": frequency_metrics,
                }
            )
            similarities.append(metrics)
            merged_velocities.append(merged_velocity_cpu)
            print(
                f"[{stage_name}-patch-flow-similarity] step={step_index:02d} "
                f"t={t:.8f}->{t_prev:.8f} "
                f"cos={metrics['cosine_similarity']:.8f} "
                f"token_cos={metrics['mean_token_cosine_similarity']:.8f} "
                f"rel_l2={metrics['relative_l2']:.8f} "
                f"plain_cos={similarity_to_plain['cosine_similarity']:.8f} "
                f"hf_amp={frequency_metrics.get('high_frequency_rms_amplification', 1.0):.8f}"
            )
            if frequency_metrics["enabled"]:
                guided_energy_log = ",".join(
                    f"{name}="
                    f"{frequency_metrics['guided_band_energy_pre_rescale'][name]:.6e}"
                    for name in self._HAAR_3D_BAND_NAMES
                )
                print(
                    f"[{stage_name}-patch-flow-frequency] step={step_index:02d} "
                    f"active={frequency_metrics['active_this_step']} "
                    f"merged_v_norm="
                    f"{metrics['velocity_norm']['merged']:.8e} "
                    f"plain_v_norm="
                    f"{metrics['velocity_norm']['plain_patch']:.8e} "
                    f"global_v_norm="
                    f"{metrics['velocity_norm']['global']:.8e} "
                    f"guided_band_energy[{guided_energy_log}]"
                )
            patch_state = patch_state.replace(
                patch_state.feats
                - time_interval * merged_velocity.to(patch_state.dtype)
            )
            del (
                velocity_sum,
                weight_sum,
                merged_velocity,
                plain_merged_velocity,
            )
            if plain_velocity_sum is not None:
                del plain_velocity_sum

        if not torch.equal(patch_state.coords, global_noise.coords):
            raise RuntimeError(
                "Patch flow changed the global 128-grid latent coordinates"
            )

        trace_patch = {
            "enabled": True,
            "status": "complete",
            "stage": stage_name,
            "patch_coordinate_mode": "local",
            "global_latent_coordinate_mode": "global",
            "grid_resolution": 128,
            "patch_size": 64,
            "patch_stride": 32,
            "patch_count": 27,
            "active_patch_count": len(active_patches),
            "start_step": int(start_step),
            "start_source": start_source,
            "start_time": float(trajectory.times[start_step]),
            "guidance": {
                "mode": patch_guidance_mode,
                "strength": float(guidance_strength),
                "interval": guidance_interval,
                "rescale": float(guidance_rescale),
                "wavelet_family": wavelet_family,
                "skip_residual_mode": skip_residual_mode,
                "wavelet_checks": wavelet_checks,
                "wavelet_compute_dtype": (
                    "float32" if patch_guidance_mode == "wavelet_cfg" else None
                ),
                "dense_wavelet_bands_saved": False,
            },
            "algebraic_inverse_check": inverse_check,
            "patches": patch_records,
            "similarities": similarities,
            "merged_velocities": merged_velocities,
            "final_state": patch_state.feats.detach().cpu(),
        }
        diagnostics = {
            "patch_coordinate_mode": "local",
            "global_latent_coordinate_mode": "global",
            "patch_start_step": int(start_step),
            "patch_start_source": start_source,
            "patch_guidance_mode": patch_guidance_mode,
            "guidance_strength": float(guidance_strength),
            "guidance_interval": guidance_interval,
            "guidance_rescale": float(guidance_rescale),
            "wavelet_family": wavelet_family,
            "skip_residual_mode": skip_residual_mode,
            "wavelet_checks": wavelet_checks,
            "wavelet_compute_dtype": (
                "float32" if patch_guidance_mode == "wavelet_cfg" else None
            ),
            "dense_wavelet_bands_saved": False,
            "patch_count": 27,
            "active_patch_count": len(active_patches),
            "concat_cond_present": bool(concat_cond is not None),
            "conditional_unconditional_share_concat_cond": bool(
                concat_cond is not None
            ),
            "algebraic_inverse_check": inverse_check,
            "velocity_similarities": similarities,
        }
        return patch_state, trace_patch, diagnostics

    @torch.no_grad()
    def _prepare_multitile_paired_condition(
        self,
        image_4096: Image.Image,
        foreground_mask_4096: Image.Image,
        global_coords: torch.Tensor,
        projected_full_norm: torch.Tensor,
        camera_angle_x: float,
        distance: float,
        mesh_scale: float,
        base_condition: Mapping[str, Any] = None,
        grid_resolution: int = 128,
        tile_size: int = 1024,
        tile_stride: int = 512,
        save_slot_proj: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract local tile conditions plus one hard owner tile per token.

        ``proj`` and ``tile_ids/tile_weights`` preserve the previous paired
        block-fusion representation for comparison.  The new target/context
        path additionally stores:

        * ``base_global`` / ``base_proj``: canonical full-image condition.
        * ``owner_tile_ids``: one center-preferred local tile per token.
        * ``owner_proj``: the projected feature from that same owner tile.

        This keeps every local target pair strictly matched while allowing all
        non-target rows in a 64^3 expert forward to use a valid full-image
        context condition.
        """
        if image_4096.size != (4096, 4096):
            raise ValueError("multi-tile extraction requires canonical 4096")

        base_condition_available = base_condition is not None
        base_global = None
        base_proj = None
        if base_condition_available:
            base_global = base_condition.get("global")
            base_proj = base_condition.get("proj")
            if not isinstance(base_global, torch.Tensor):
                raise TypeError(
                    "base_condition['global'] must be a tensor"
                )
            if not isinstance(base_proj, SparseTensor):
                raise TypeError(
                    "base_condition['proj'] must be a SparseTensor"
                )
            if base_global.ndim != 3 or base_global.shape[0] != 1:
                raise ValueError(
                    "base full-image global must have shape [1, L, C]"
                )
            if base_proj.feats.shape[0] != global_coords.shape[0]:
                raise RuntimeError(
                    "base projected feature is not token aligned"
                )
            if not torch.equal(base_proj.coords, global_coords):
                raise RuntimeError(
                    "base projected coordinates changed token order"
                )
            if not torch.isfinite(base_global).all():
                raise RuntimeError("base global contains NaN/Inf")
            if not torch.isfinite(base_proj.feats).all():
                raise RuntimeError(
                    "base projected feature contains NaN/Inf"
                )

        boxes = self.build_texture_image_tile_layout(
            4096, tile_size, tile_stride
        )
        raw_uv = projected_full_norm.to(global_coords.device) * 4096.0
        tile_ids_layout, tile_weights, assignment_uv = (
            self.assign_texture_tiles(
                raw_uv,
                boxes,
                canonical_size=4096,
            )
        )

        # Hard owner = covering crop in which the projected point has the
        # largest 2D tent weight.  Overlap is therefore used to keep points
        # away from crop borders, but each target row has one unambiguous
        # local (global, proj) pair.
        owner_scores = tile_weights.masked_fill(
            tile_ids_layout < 0,
            -torch.inf,
        )
        owner_slots = owner_scores.argmax(dim=1)
        owner_layout_ids = tile_ids_layout.gather(
            1, owner_slots[:, None]
        )[:, 0]
        owner_raw_weights = tile_weights.gather(
            1, owner_slots[:, None]
        )[:, 0]
        if torch.any(owner_layout_ids < 0):
            bad = torch.where(owner_layout_ids < 0)[0]
            raise RuntimeError(
                "hard image-tile routing left tokens without owners: "
                f"{bad[:16].tolist()}"
            )

        fused_proj = None
        owner_proj = None

        # Runtime local projected features for all memberships.
        # Shape is initialized after the first tile reveals channel count C.
        slot_proj_runtime = None

        owner_write_count = torch.zeros(
            global_coords.shape[0],
            dtype=torch.int32,
            device=global_coords.device,
        )

        # Optional CPU debug copy.
        slot_proj = None
        
        global_bank_parts: List[torch.Tensor] = []
        layout_to_bank: Dict[int, int] = {}
        records: List[Dict[str, Any]] = []
        extraction_started = time.perf_counter()

        used_layout_ids = torch.unique(
            tile_ids_layout[tile_ids_layout >= 0],
            sorted=True,
        ).tolist()
        for layout_tile_id in used_layout_ids:
            layout_tile_id = int(layout_tile_id)
            bank_id = len(global_bank_parts)
            layout_to_bank[layout_tile_id] = bank_id
            rows, slots = torch.where(
                tile_ids_layout == layout_tile_id
            )
            x0, y0, x1, y1 = boxes[layout_tile_id]
            tile_image = image_4096.crop(
                (x0, y0, x1, y1)
            ).convert("RGB")
            if tile_image.size != (tile_size, tile_size):
                raise RuntimeError(
                    "canonical image tile has an invalid size"
                )

            started = time.perf_counter()
            tile_condition = self.get_proj_cond_shape(
                image_cond_model=self.image_cond_model_tex_1024,
                image=[tile_image],
                coords=global_coords[rows],
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                grid_resolution_override=grid_resolution,
                projection_crop_box=(
                    x0 / 4096.0,
                    y0 / 4096.0,
                    x1 / 4096.0,
                    y1 / 4096.0,
                ),
            )["cond"]
            tile_global = tile_condition["global"]
            tile_proj = tile_condition["proj"].feats
            if (
                tile_global.ndim != 3
                or tile_global.shape[0] != 1
                or tile_proj.shape[0] != rows.numel()
            ):
                raise RuntimeError(
                    "tile global/proj extraction is not row aligned"
                )
            if (
                base_proj is not None
                and tile_proj.shape[1] != base_proj.feats.shape[1]
            ):
                raise RuntimeError(
                    "local and base projected channel counts differ"
                )

            if fused_proj is None:
                fused_proj = torch.zeros(
                    global_coords.shape[0],
                    tile_proj.shape[1],
                    device=tile_proj.device,
                    dtype=tile_proj.dtype,
                )
                owner_proj = torch.zeros_like(fused_proj)

                slot_proj_runtime = torch.zeros(
                    global_coords.shape[0],
                    4,
                    tile_proj.shape[1],
                    device=tile_proj.device,
                    dtype=tile_proj.dtype,
                )

            if not torch.isfinite(tile_global).all():
                raise RuntimeError(
                    f"tile {layout_tile_id} global contains NaN/Inf"
                )
            if not torch.isfinite(tile_proj).all():
                raise RuntimeError(
                    f"tile {layout_tile_id} proj contains NaN/Inf"
                )

            weights = tile_weights[rows, slots].to(
                device=tile_proj.device,
                dtype=tile_proj.dtype,
            )
            
            if slot_proj_runtime is None:
                raise RuntimeError("slot_proj_runtime was not initialized")

            # rows[j], slots[j] precisely identifies this point/tile membership.
            slot_proj_runtime[rows, slots] = tile_proj

            fused_proj.index_add_(
                0,
                rows,
                tile_proj * weights[:, None],
            )

            owner_mask = owner_layout_ids[rows] == layout_tile_id
            owner_rows = rows[owner_mask]
            if owner_rows.numel() > 0:
                owner_proj[owner_rows] = tile_proj[owner_mask]
                owner_write_count.index_add_(
                    0,
                    owner_rows,
                    torch.ones_like(owner_rows, dtype=torch.int32),
                )

            if save_slot_proj:
                if slot_proj is None:
                    slot_proj = torch.zeros(
                        global_coords.shape[0],
                        4,
                        tile_proj.shape[1],
                        device="cpu",
                        dtype=tile_proj.dtype,
                    )
                slot_proj[rows.cpu(), slots.cpu()] = (
                    tile_proj.detach().cpu()
                )

            global_bank_parts.append(tile_global[0])
            foreground_ratio = float(
                (
                    np.asarray(
                        foreground_mask_4096.crop(
                            (x0, y0, x1, y1)
                        )
                    )
                    > 0
                ).mean()
            )
            seconds = time.perf_counter() - started
            records.append(
                {
                    "layout_tile_id": layout_tile_id,
                    "bank_tile_id": bank_id,
                    "box": [x0, y0, x1, y1],
                    "token_count": int(rows.numel()),
                    "owner_token_count": int(owner_rows.numel()),
                    "foreground_ratio": foreground_ratio,
                    "global_shape": list(tile_global.shape),
                    "proj_shape": list(tile_proj.shape),
                    "seconds": seconds,
                }
            )
            print(
                f"[texture-tile-cond] tile={layout_tile_id:03d} "
                f"box={(x0, y0, x1, y1)} "
                f"tokens={rows.numel():,} "
                f"owners={owner_rows.numel():,} "
                f"foreground_ratio={foreground_ratio:.6f} "
                f"global_shape={tuple(tile_global.shape)} "
                f"proj_shape={tuple(tile_proj.shape)} "
                f"seconds={seconds:.3f}"
            )

        if not global_bank_parts:
            raise RuntimeError("no image tile condition was extracted")

        if (
            fused_proj is None
            or owner_proj is None
            or slot_proj_runtime is None
        ):
            raise RuntimeError(
                "tile extraction did not initialize projected features"
            )

        if not torch.isfinite(slot_proj_runtime).all():
            raise RuntimeError(
                "membership projected features contain NaN/Inf"
            )

        invalid_slots = tile_ids_layout < 0
        if torch.any(slot_proj_runtime[invalid_slots] != 0):
            raise RuntimeError(
                "invalid membership projected-feature slots are not zero"
            )

        if not base_condition_available:
            # Legacy paired-fusion tests historically called this internal
            # helper without a full-image condition. These placeholders keep
            # that old preparation path numerically unchanged. The
            # target_context_hard runner rejects them explicitly below.
            base_global = torch.zeros_like(
                global_bank_parts[0]
            ).unsqueeze(0)
            base_proj = SparseTensor(
                feats=torch.zeros_like(fused_proj),
                coords=global_coords,
            )
        if torch.any(owner_write_count != 1):
            bad = torch.where(owner_write_count != 1)[0]
            raise RuntimeError(
                "owner projected features were not written exactly once: "
                f"{bad[:16].tolist()}"
            )
        if not torch.isfinite(fused_proj).all():
            raise RuntimeError("fused projected features contain NaN/Inf")
        if not torch.isfinite(owner_proj).all():
            raise RuntimeError("owner projected features contain NaN/Inf")

        tile_ids = torch.full_like(tile_ids_layout, -1)
        owner_tile_ids = torch.full_like(owner_layout_ids, -1)
        for layout_id, bank_id in layout_to_bank.items():
            tile_ids[tile_ids_layout == layout_id] = bank_id
            owner_tile_ids[owner_layout_ids == layout_id] = bank_id
        if torch.any(owner_tile_ids < 0):
            bad = torch.where(owner_tile_ids < 0)[0]
            raise RuntimeError(
                "failed to remap owner tile IDs: "
                f"{bad[:16].tolist()}"
            )

        global_bank = torch.stack(global_bank_parts, dim=0)
        condition = {
            # Previous representation retained for paired_block_fusion.
            "mode": "multi_tile_paired",
            "global_bank": global_bank,
            "proj": SparseTensor(
                feats=fused_proj,
                coords=global_coords,
            ),
            "tile_ids": tile_ids,
            "tile_weights": tile_weights.to(fused_proj.device),
            # New target/context representation.
            "base_condition_available": base_condition_available,
            "base_global": base_global,
            "base_proj": base_proj,
             # All local projected features, without fusion.
            # [N, 4, C], aligned with tile_ids/tile_weights slots.
            "slot_proj": slot_proj_runtime,
            "owner_tile_ids": owner_tile_ids,
            "owner_slots": owner_slots,
            "owner_weights": owner_raw_weights,
            "owner_proj": SparseTensor(
                feats=owner_proj,
                coords=global_coords,
            ),
        }

        counts = (tile_ids >= 0).sum(1)
        histogram = {
            int(value): int((counts == value).sum().item())
            for value in torch.unique(counts).tolist()
        }
        owner_histogram = {
            int(value): int((owner_tile_ids == value).sum().item())
            for value in torch.unique(owner_tile_ids).tolist()
        }
        summary = {
            "canonical_size": 4096,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "tile_count": len(boxes),
            "active_tile_count": len(global_bank_parts),
            "token_count": int(global_coords.shape[0]),
            "membership_min": int(counts.min().item()),
            "membership_max": int(counts.max().item()),
            "membership_histogram": histogram,
            "owner_histogram": owner_histogram,
            "owner_weight_min": float(owner_raw_weights.min().item()),
            "owner_weight_max": float(owner_raw_weights.max().item()),
            "weight_sum_min": float(tile_weights.sum(1).min().item()),
            "weight_sum_max": float(tile_weights.sum(1).max().item()),
            "raw_uv": raw_uv.detach().cpu(),
            "assignment_uv": assignment_uv.detach().cpu(),
            "boxes": boxes,
            "tiles": records,
            "slot_proj": slot_proj,
            "seconds": time.perf_counter() - extraction_started,
        }
        print(
            "[texture-image-tiles] canonical=4096x4096 "
            f"tile={tile_size} stride={tile_stride} "
            f"count={len(boxes)} "
            f"active_tiles={len(global_bank_parts)} "
            f"tokens={global_coords.shape[0]:,} "
            f"membership_min={summary['membership_min']} "
            f"membership_max={summary['membership_max']} "
            f"membership_histogram={histogram} "
            f"owner_weight_min={summary['owner_weight_min']:.8f} "
            f"owner_weight_max={summary['owner_weight_max']:.8f} "
            f"weight_sum_min={summary['weight_sum_min']:.8f} "
            f"weight_sum_max={summary['weight_sum_max']:.8f}"
        )
        return condition, summary

    @torch.no_grad()
    def _run_multitile_3d_patch_texture_flow(
        self,
        flow_model: nn.Module,
        sampler: Any,
        global_noise: SparseTensor,
        shape_concat_cond: SparseTensor,
        sampler_params: Mapping[str, Any],
        global_flow: Any,
        condition: Mapping[str, Any],
        start_step: int,
        patch_size: int = 64,
        patch_stride: int = 32,
        fusion_mode: str = "paired_block_fusion",
    ) -> Tuple[SparseTensor, Dict[str, Any], Dict[str, Any]]:
        """Resume texture flow with either old paired fusion or target/context experts.

        ``target_context_hard`` runs one complete 64^3 Flow forward per owner
        tile present in a patch.  Target rows use that tile's matched local
        global/proj; every other row remains in the complete patch as valid
        three-dimensional context and uses the canonical full-image
        global/proj.  Only target-row velocities are collected.  No velocity
        fallback is permitted.
        """
        allowed_modes = {
            "paired_block_fusion",
            "target_context_hard",
            "membership_velocity_fusion",
        }
        if fusion_mode not in allowed_modes:
            raise ValueError(
                f"unsupported multi-tile fusion mode {fusion_mode!r}; "
                f"expected one of {sorted(allowed_modes)}"
            )

        trajectory = global_flow.trajectory
        if trajectory is None:
            raise RuntimeError("global texture trajectory was not recorded")
        num_steps = len(trajectory.velocities)
        if len(trajectory.states) != num_steps + 1:
            raise RuntimeError("global texture trajectory is incomplete")
        if not 0 <= start_step <= num_steps:
            raise ValueError(
                "multi-tile start step is outside the trajectory"
            )
        if global_noise.feats.shape[0] != shape_concat_cond.feats.shape[0]:
            raise RuntimeError(
                "texture noise and shape condition token counts differ"
            )
        if not torch.equal(
            global_noise.coords,
            shape_concat_cond.coords,
        ):
            raise RuntimeError(
                "texture noise and shape condition coordinates differ"
            )

        mode_name = {
            "paired_block_fusion":
                "multi_tile_paired_3d_patch_flow",
            "target_context_hard":
                "multi_tile_target_context_hard_3d_patch_flow",
            "membership_velocity_fusion":
                "multi_tile_membership_velocity_3d_patch_flow",
        }[fusion_mode]

        condition_version = {
            "paired_block_fusion":
                "multi_tile_paired_v1",
            "target_context_hard":
                "multi_tile_target_context_hard_v1",
            "membership_velocity_fusion":
                "multi_tile_membership_velocity_v1",
        }[fusion_mode]

        selected_state = trajectory.states[start_step].to(
            device=global_noise.device,
            dtype=global_noise.dtype,
            copy=True,
        )
        x_global = global_noise.replace(selected_state)
        if start_step == num_steps:
            return (
                x_global,
                {
                    "enabled": True,
                    "status": "identity_start_step_final",
                    "mode": mode_name,
                    "condition_format_version": condition_version,
                    "start_step": start_step,
                    "steps": [],
                    "patch_flow_calls": 0,
                    "expert_flow_calls": 0,
                    "final_state": selected_state.detach().cpu(),
                },
                {
                    "identity": True,
                    "patch_flow_calls": 0,
                    "expert_flow_calls": 0,
                    "condition_mode": fusion_mode,
                },
            )

        if fusion_mode == "target_context_hard":
            if not condition.get("base_condition_available", False):
                raise RuntimeError(
                    "target_context_hard requires the canonical "
                    "full-image base condition"
                )
            required = {
                "global_bank",
                "base_global",
                "base_proj",
                "owner_tile_ids",
                "owner_proj",
            }
            missing = required - set(condition)
            if missing:
                raise KeyError(
                    "target/context condition is missing: "
                    f"{sorted(missing)}"
                )
            if not isinstance(condition["base_proj"], SparseTensor):
                raise TypeError("base_proj must be a SparseTensor")
            if not isinstance(condition["owner_proj"], SparseTensor):
                raise TypeError("owner_proj must be a SparseTensor")
            for key in ("base_proj", "owner_proj"):
                projection = condition[key]
                if projection.feats.shape[0] != global_noise.feats.shape[0]:
                    raise RuntimeError(f"{key} is not token aligned")
                if not torch.equal(
                    projection.coords,
                    global_noise.coords,
                ):
                    raise RuntimeError(f"{key} coordinates are misaligned")
            if condition["owner_tile_ids"].shape != (
                global_noise.feats.shape[0],
            ):
                raise RuntimeError("owner tile IDs are not token aligned")
            if torch.any(condition["owner_tile_ids"] < 0):
                raise RuntimeError("owner tile IDs contain invalid values")
            if torch.any(
                condition["owner_tile_ids"]
                >= condition["global_bank"].shape[0]
            ):
                raise RuntimeError("owner tile IDs exceed global bank")

        if fusion_mode == "membership_velocity_fusion":
            if not condition.get("base_condition_available", False):
                raise RuntimeError(
                    "membership_velocity_fusion requires the canonical "
                    "full-image base condition"
                )

            required = {
                "global_bank",
                "base_global",
                "base_proj",
                "tile_ids",
                "tile_weights",
                "slot_proj",
            }
            missing = required - set(condition)
            if missing:
                raise KeyError(
                    "membership velocity condition is missing: "
                    f"{sorted(missing)}"
                )

            base_proj = condition["base_proj"]
            tile_ids = condition["tile_ids"]
            tile_weights = condition["tile_weights"]
            slot_proj_runtime = condition["slot_proj"]

            if not isinstance(base_proj, SparseTensor):
                raise TypeError("base_proj must be a SparseTensor")

            token_count = global_noise.feats.shape[0]

            if base_proj.feats.shape[0] != token_count:
                raise RuntimeError("base_proj is not token aligned")

            if not torch.equal(base_proj.coords, global_noise.coords):
                raise RuntimeError("base_proj coordinates are misaligned")

            if tile_ids.shape != (token_count, 4):
                raise RuntimeError(
                    f"tile_ids must be [{token_count}, 4], "
                    f"got {tuple(tile_ids.shape)}"
                )

            if tile_weights.shape != tile_ids.shape:
                raise RuntimeError(
                    "tile_weights and tile_ids are not aligned"
                )

            if (
                slot_proj_runtime.ndim != 3
                or slot_proj_runtime.shape[:2] != tile_ids.shape
            ):
                raise RuntimeError(
                    "slot_proj must have shape [N, 4, C]"
                )

            valid = tile_ids >= 0

            if torch.any(valid.sum(dim=1) < 1):
                raise RuntimeError(
                    "some tokens have no local image-tile membership"
                )

            if torch.any(tile_weights[~valid] != 0):
                raise RuntimeError(
                    "invalid tile slots have nonzero weights"
                )

            if torch.any(slot_proj_runtime[~valid] != 0):
                raise RuntimeError(
                    "invalid tile slots have nonzero projected features"
                )

            if torch.any(tile_weights[valid] <= 0):
                raise RuntimeError(
                    "valid tile memberships must have positive weights"
                )

            row_weight_sum = tile_weights.sum(dim=1)
            if not torch.allclose(
                row_weight_sum,
                torch.ones_like(row_weight_sum),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise RuntimeError(
                    "tile membership weights are not normalized"
                )

            if torch.any(tile_ids[valid] >= condition["global_bank"].shape[0]):
                raise RuntimeError("tile IDs exceed global bank")

        patches, static_coverage = self.build_texture_3d_patches(
            global_noise.coords,
            128,
            patch_size,
            patch_stride,
        )
        coverage_histogram = {
            int(value): int((static_coverage == value).sum().item())
            for value in torch.unique(static_coverage).tolist()
        }
        print(
            "[texture-3d-patches] grid=128 "
            f"patch={patch_size} stride={patch_stride} "
            f"count={len(patches)} "
            f"tokens={global_noise.feats.shape[0]:,} "
            f"coverage_min={static_coverage.min().item()} "
            f"coverage_max={static_coverage.max().item()} "
            f"coverage_histogram={coverage_histogram} "
            f"fusion_mode={fusion_mode}"
        )

        prediction_kwargs = {
            key: value
            for key, value in sampler_params.items()
            if key
            not in {
                "steps",
                "rescale_t",
                "verbose",
                "tqdm_desc",
                "record_trajectory",
                "trajectory_device",
                "return_model_history",
            }
        }
        raw_times = np.linspace(
            1.0, 0.0, num_steps + 1
        ).tolist()
        steps: List[Dict[str, Any]] = []
        total_patch_flow_calls = 0
        total_expert_flow_calls = 0

        for step_index in range(start_step, num_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_started = time.perf_counter()
            x_step_start = x_global
            mapped_t = float(trajectory.times[step_index])
            mapped_t_next = float(
                trajectory.times[step_index + 1]
            )
            dt = float(
                trajectory.time_intervals[step_index]
            )
            if not math.isclose(
                mapped_t - mapped_t_next,
                dt,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "global trajectory timestep mismatch"
                )

            patch_results = []
            step_patch_flow_calls = 0
            step_expert_flow_calls = 0
            expert_count_per_patch: List[int] = []

            for patch_index, patch in enumerate(patches):
                indices = patch["global_indices"]
                if indices.numel() == 0:
                    raise RuntimeError(
                        f"3D patch {patch_index} is empty"
                    )
                local_coords = patch["local_coords"]
                x_patch = SparseTensor(
                    feats=x_step_start.feats[indices],
                    coords=local_coords,
                )
                shape_patch = SparseTensor(
                    feats=shape_concat_cond.feats[indices],
                    coords=local_coords,
                )
                if not torch.equal(
                    x_patch.coords,
                    shape_patch.coords,
                ):
                    raise RuntimeError(
                        f"shape/texture coordinate mismatch "
                        f"at patch {patch_index}"
                    )

                if fusion_mode == "paired_block_fusion":
                    proj_patch = SparseTensor(
                        feats=condition["proj"].feats[indices],
                        coords=local_coords,
                    )
                    patch_condition = {
                        "mode": "multi_tile_paired",
                        "global_bank": condition["global_bank"],
                        "proj": proj_patch,
                        "tile_ids": condition["tile_ids"][indices],
                        "tile_weights": condition[
                            "tile_weights"
                        ][indices],
                    }
                    patch_negative = (
                        self.make_multitile_negative_condition(
                            patch_condition
                        )
                    )
                    _, _, patch_velocity = (
                        sampler._get_model_prediction(
                            flow_model,
                            x_patch,
                            mapped_t,
                            patch_condition,
                            neg_cond=patch_negative,
                            concat_cond=shape_patch,
                            **prediction_kwargs,
                        )
                    )
                    if (
                        patch_velocity.feats.shape
                        != x_patch.feats.shape
                        or not torch.equal(
                            patch_velocity.coords,
                            local_coords,
                        )
                        or not torch.isfinite(
                            patch_velocity.feats
                        ).all()
                    ):
                        raise RuntimeError(
                            f"invalid paired patch velocity "
                            f"at patch {patch_index}"
                        )
                    patch_velocity_feats = (
                        patch_velocity.feats
                    )
                    step_patch_flow_calls += 1
                    total_patch_flow_calls += 1
                    expert_count_per_patch.append(1)
                elif fusion_mode == "target_context_hard":
                    # Complete 64^3 context is retained for every local
                    # expert.  Each row has exactly one condition:
                    # target -> local owner tile; context -> full image.
                    patch_owner_ids = condition[
                        "owner_tile_ids"
                    ][indices]
                    active_owner_ids = torch.unique(
                        patch_owner_ids,
                        sorted=True,
                    )
                    if active_owner_ids.numel() == 0:
                        raise RuntimeError(
                            f"patch {patch_index} has no owner tiles"
                        )

                    base_proj_feats = condition[
                        "base_proj"
                    ].feats[indices]
                    owner_proj_feats = condition[
                        "owner_proj"
                    ].feats[indices]
                    patch_velocity_sum = torch.zeros(
                        x_patch.feats.shape,
                        device=x_patch.device,
                        dtype=torch.float32,
                    )
                    target_coverage = torch.zeros(
                        x_patch.feats.shape[0],
                        device=x_patch.device,
                        dtype=torch.int32,
                    )

                    for owner_id_tensor in active_owner_ids:
                        owner_id = int(
                            owner_id_tensor.item()
                        )
                        target_rows = torch.where(
                            patch_owner_ids == owner_id
                        )[0]
                        if target_rows.numel() == 0:
                            raise RuntimeError(
                                "active owner tile has no target rows"
                            )

                        expert_proj_feats = (
                            base_proj_feats.clone()
                        )
                        expert_proj_feats[target_rows] = (
                            owner_proj_feats[target_rows]
                        )
                        expert_proj = SparseTensor(
                            feats=expert_proj_feats,
                            coords=local_coords,
                        )

                        base_global = condition[
                            "base_global"
                        ]
                        local_global = condition[
                            "global_bank"
                        ][owner_id : owner_id + 1]
                        expert_global_bank = torch.cat(
                            [base_global, local_global],
                            dim=0,
                        )

                        # Four slots are retained to satisfy the current
                        # multi-tile attention interface, but each row has
                        # exactly one valid slot with weight one.
                        expert_tile_ids = torch.full(
                            (
                                x_patch.feats.shape[0],
                                4,
                            ),
                            -1,
                            device=x_patch.device,
                            dtype=torch.long,
                        )
                        expert_tile_weights = torch.zeros(
                            (
                                x_patch.feats.shape[0],
                                4,
                            ),
                            device=x_patch.device,
                            dtype=torch.float32,
                        )
                        expert_tile_ids[:, 0] = 0
                        expert_tile_ids[target_rows, 0] = 1
                        expert_tile_weights[:, 0] = 1.0

                        expert_condition = {
                            "mode": "multi_tile_paired",
                            "global_bank": expert_global_bank,
                            "proj": expert_proj,
                            "tile_ids": expert_tile_ids,
                            "tile_weights": expert_tile_weights,
                        }
                        expert_negative = (
                            self.make_multitile_negative_condition(
                                expert_condition
                            )
                        )
                        _, _, expert_velocity = (
                            sampler._get_model_prediction(
                                flow_model,
                                x_patch,
                                mapped_t,
                                expert_condition,
                                neg_cond=expert_negative,
                                concat_cond=shape_patch,
                                **prediction_kwargs,
                            )
                        )
                        if (
                            expert_velocity.feats.shape
                            != x_patch.feats.shape
                            or not torch.equal(
                                expert_velocity.coords,
                                local_coords,
                            )
                            or not torch.isfinite(
                                expert_velocity.feats
                            ).all()
                        ):
                            raise RuntimeError(
                                "invalid target/context expert "
                                f"velocity at patch={patch_index}, "
                                f"owner={owner_id}"
                            )

                        patch_velocity_sum.index_add_(
                            0,
                            target_rows,
                            expert_velocity.feats[
                                target_rows
                            ].float(),
                        )
                        target_coverage.index_add_(
                            0,
                            target_rows,
                            torch.ones_like(
                                target_rows,
                                dtype=torch.int32,
                            ),
                        )
                        step_expert_flow_calls += 1
                        total_expert_flow_calls += 1

                        del (
                            expert_proj_feats,
                            expert_proj,
                            expert_global_bank,
                            expert_tile_ids,
                            expert_tile_weights,
                            expert_condition,
                            expert_negative,
                            expert_velocity,
                        )

                    if torch.any(target_coverage != 1):
                        bad = torch.where(
                            target_coverage != 1
                        )[0]
                        raise RuntimeError(
                            "target/context experts did not "
                            "produce exactly one velocity for every "
                            f"patch row; patch={patch_index}, "
                            f"bad_rows={bad[:16].tolist()}"
                        )
                    patch_velocity_feats = (
                        patch_velocity_sum.to(
                            dtype=x_patch.dtype
                        )
                    )
                    expert_count_per_patch.append(
                        int(active_owner_ids.numel())
                    )
                else:
                    # ============================================================
                    # membership_velocity_fusion
                    #
                    # One full Flow forward per active local image tile.
                    #
                    # In tile-k expert:
                    #   rows covered by k -> local global k + local proj k
                    #   all other rows     -> canonical base global + base proj
                    #
                    # No multi-condition fusion occurs inside Flow.
                    # Velocities are fused afterwards with original tent weights.
                    # ============================================================

                    patch_tile_ids = condition["tile_ids"][indices]          # [P, 4]
                    patch_tile_weights = condition["tile_weights"][indices]  # [P, 4]
                    patch_slot_proj = condition["slot_proj"][indices]        # [P, 4, C]

                    valid_memberships = patch_tile_ids >= 0

                    active_tile_ids = torch.unique(
                        patch_tile_ids[valid_memberships],
                        sorted=True,
                    )

                    if active_tile_ids.numel() == 0:
                        raise RuntimeError(
                            f"patch {patch_index} has no active image tiles"
                        )

                    base_proj_feats = condition["base_proj"].feats[indices]

                    if base_proj_feats.shape[0] != x_patch.feats.shape[0]:
                        raise RuntimeError(
                            f"base projected feature is not aligned at "
                            f"patch {patch_index}"
                        )

                    patch_velocity_sum = torch.zeros(
                        x_patch.feats.shape,
                        device=x_patch.device,
                        dtype=torch.float32,
                    )

                    patch_velocity_weight_sum = torch.zeros(
                        x_patch.feats.shape[0],
                        1,
                        device=x_patch.device,
                        dtype=torch.float32,
                    )

                    membership_coverage = torch.zeros(
                        x_patch.feats.shape[0],
                        device=x_patch.device,
                        dtype=torch.int32,
                    )

                    expected_membership_coverage = (
                        valid_memberships.sum(dim=1).to(torch.int32)
                    )

                    for tile_id_tensor in active_tile_ids:
                        tile_id = int(tile_id_tensor.item())

                        # member_rows[j] has tile_id in member_slots[j].
                        member_rows, member_slots = torch.where(
                            patch_tile_ids == tile_id
                        )

                        if member_rows.numel() == 0:
                            raise RuntimeError(
                                f"active tile {tile_id} has no member rows"
                            )

                        # The same tile must not occur twice in one point's slots.
                        if torch.unique(member_rows).numel() != member_rows.numel():
                            raise RuntimeError(
                                f"duplicate tile membership in patch={patch_index}, "
                                f"tile={tile_id}"
                            )

                        member_weights = patch_tile_weights[
                            member_rows,
                            member_slots,
                        ].to(
                            device=x_patch.device,
                            dtype=torch.float32,
                        )

                        if (
                            not torch.isfinite(member_weights).all()
                            or torch.any(member_weights <= 0)
                        ):
                            raise RuntimeError(
                                f"invalid membership weights at patch={patch_index}, "
                                f"tile={tile_id}"
                            )

                        local_proj_feats = patch_slot_proj[
                            member_rows,
                            member_slots,
                        ]

                        if not torch.isfinite(local_proj_feats).all():
                            raise RuntimeError(
                                f"local proj contains NaN/Inf at "
                                f"patch={patch_index}, tile={tile_id}"
                            )

                        # --------------------------------------------------------
                        # Build one-condition-per-row projected feature.
                        # --------------------------------------------------------
                        expert_proj_feats = base_proj_feats.clone()
                        expert_proj_feats[member_rows] = local_proj_feats

                        expert_proj = SparseTensor(
                            feats=expert_proj_feats,
                            coords=local_coords,
                        )

                        # Bank index 0 = canonical resized full image.
                        # Bank index 1 = the currently active local tile.
                        base_global = condition["base_global"]
                        local_global = condition["global_bank"][
                            tile_id : tile_id + 1
                        ]

                        expert_global_bank = torch.cat(
                            [base_global, local_global],
                            dim=0,
                        )

                        # Preserve the current multi_tile_paired interface, but
                        # every row has exactly one valid slot of weight 1.
                        #
                        # Non-members -> bank 0, base image.
                        # Members     -> bank 1, current local tile.
                        expert_tile_ids = torch.full(
                            (x_patch.feats.shape[0], 4),
                            -1,
                            device=x_patch.device,
                            dtype=torch.long,
                        )

                        expert_tile_weights = torch.zeros(
                            (x_patch.feats.shape[0], 4),
                            device=x_patch.device,
                            dtype=torch.float32,
                        )

                        expert_tile_ids[:, 0] = 0
                        expert_tile_ids[member_rows, 0] = 1
                        expert_tile_weights[:, 0] = 1.0

                        # Strictly verify: no internal weighted fusion.
                        expert_valid_count = (expert_tile_ids >= 0).sum(dim=1)
                        if torch.any(expert_valid_count != 1):
                            raise RuntimeError(
                                "expert condition has more than one valid "
                                "global condition per row"
                            )

                        if not torch.allclose(
                            expert_tile_weights.sum(dim=1),
                            torch.ones(
                                x_patch.feats.shape[0],
                                device=x_patch.device,
                            ),
                            atol=0,
                            rtol=0,
                        ):
                            raise RuntimeError(
                                "expert condition weights are not exactly one"
                            )

                        expert_condition = {
                            "mode": "multi_tile_paired",
                            "global_bank": expert_global_bank,
                            "proj": expert_proj,
                            "tile_ids": expert_tile_ids,
                            "tile_weights": expert_tile_weights,
                        }

                        expert_negative = self.make_multitile_negative_condition(
                            expert_condition
                        )

                        _, _, expert_velocity = sampler._get_model_prediction(
                            flow_model,
                            x_patch,
                            mapped_t,
                            expert_condition,
                            neg_cond=expert_negative,
                            concat_cond=shape_patch,
                            **prediction_kwargs,
                        )

                        if (
                            expert_velocity.feats.shape != x_patch.feats.shape
                            or not torch.equal(
                                expert_velocity.coords,
                                local_coords,
                            )
                            or not torch.isfinite(
                                expert_velocity.feats
                            ).all()
                        ):
                            raise RuntimeError(
                                "invalid membership expert velocity at "
                                f"patch={patch_index}, tile={tile_id}"
                            )

                        # --------------------------------------------------------
                        # Velocity-level weighted accumulation.
                        #
                        # Only rows using current local tile contribute.
                        # Base-filled context rows are intentionally discarded.
                        # --------------------------------------------------------
                        weighted_member_velocity = (
                            expert_velocity.feats[member_rows].float()
                            * member_weights[:, None]
                        )

                        patch_velocity_sum.index_add_(
                            0,
                            member_rows,
                            weighted_member_velocity,
                        )

                        patch_velocity_weight_sum.index_add_(
                            0,
                            member_rows,
                            member_weights[:, None],
                        )

                        membership_coverage.index_add_(
                            0,
                            member_rows,
                            torch.ones_like(
                                member_rows,
                                dtype=torch.int32,
                            ),
                        )

                        step_expert_flow_calls += 1
                        total_expert_flow_calls += 1

                        del (
                            local_proj_feats,
                            expert_proj_feats,
                            expert_proj,
                            expert_global_bank,
                            expert_tile_ids,
                            expert_tile_weights,
                            expert_condition,
                            expert_negative,
                            expert_velocity,
                            weighted_member_velocity,
                        )

                    # Every point must receive exactly one velocity from each of
                    # its image-tile memberships.
                    if torch.any(
                        membership_coverage != expected_membership_coverage
                    ):
                        bad = torch.where(
                            membership_coverage != expected_membership_coverage
                        )[0]
                        raise RuntimeError(
                            "membership experts did not produce one velocity "
                            "per valid membership; "
                            f"patch={patch_index}, "
                            f"bad_rows={bad[:16].tolist()}"
                        )

                    expected_weight_sum = (
                        patch_tile_weights
                        .masked_fill(~valid_memberships, 0)
                        .sum(dim=1, keepdim=True)
                        .float()
                    )

                    if not torch.allclose(
                        patch_velocity_weight_sum,
                        expected_weight_sum,
                        atol=1e-5,
                        rtol=1e-5,
                    ):
                        max_error = (
                            patch_velocity_weight_sum - expected_weight_sum
                        ).abs().max().item()
                        raise RuntimeError(
                            "velocity fusion weights do not match image-tile "
                            f"weights; patch={patch_index}, "
                            f"max_error={max_error:.8e}"
                        )

                    if torch.any(patch_velocity_weight_sum <= 0):
                        bad = torch.where(
                            patch_velocity_weight_sum[:, 0] <= 0
                        )[0]
                        raise RuntimeError(
                            "some patch rows received no local velocity; "
                            f"patch={patch_index}, "
                            f"bad_rows={bad[:16].tolist()}"
                        )

                    # Weighted mean velocity for every point.
                    patch_velocity_feats = (
                        patch_velocity_sum
                        / patch_velocity_weight_sum.clamp_min(1e-12)
                    ).to(dtype=x_patch.dtype)

                    if not torch.isfinite(patch_velocity_feats).all():
                        raise RuntimeError(
                            f"weighted patch velocity is non-finite at "
                            f"patch={patch_index}"
                        )

                    expert_count_per_patch.append(
                        int(active_tile_ids.numel())
                    )

                weights = self.texture_3d_patch_weights(
                    local_coords,
                    patch_size,
                ).to(
                    device=x_patch.device,
                    dtype=torch.float32,
                )
                patch_results.append(
                    (
                        indices,
                        patch_velocity_feats,
                        weights,
                    )
                )

            merged, coverage = (
                self.merge_texture_3d_patch_velocities(
                    token_count=x_step_start.feats.shape[0],
                    patch_results=patch_results,
                    channels=x_step_start.feats.shape[1],
                    device=x_step_start.device,
                )
            )
            metrics = self._velocity_similarity(
                merged,
                trajectory.velocities[step_index],
            )

            # The only global update in this timestep.
            x_global = x_step_start.replace(
                x_step_start.feats
                - dt * merged.to(x_step_start.dtype)
            )
            if not torch.equal(
                x_global.coords,
                global_noise.coords,
            ):
                raise RuntimeError(
                    "target/context flow changed global "
                    "coordinates or token order"
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record = {
                **metrics,
                "step_index": step_index,
                "raw_t": raw_times[step_index],
                "raw_t_next": raw_times[
                    step_index + 1
                ],
                "mapped_t": mapped_t,
                "mapped_t_next": mapped_t_next,
                "time_interval": dt,
                "patch_count": len(patches),
                "patch_flow_calls": step_patch_flow_calls,
                "expert_flow_calls": step_expert_flow_calls,
                "expert_count_per_patch": (
                    expert_count_per_patch
                ),
                "expert_count_min": int(
                    min(expert_count_per_patch)
                ),
                "expert_count_max": int(
                    max(expert_count_per_patch)
                ),
                "expert_count_mean": float(
                    np.mean(expert_count_per_patch)
                ),
                "velocity_coverage": 1.0,
                "velocity_coverage_min": int(
                    coverage.min().item()
                ),
                "velocity_coverage_max": int(
                    coverage.max().item()
                ),
                "seconds": (
                    time.perf_counter() - step_started
                ),
            }
            steps.append(record)
            print(
                "[texture-multitile-3d-flow] "
                f"mode={fusion_mode} "
                f"step={step_index:02d} "
                f"raw_t={record['raw_t']:.8f}->"
                f"{record['raw_t_next']:.8f} "
                f"mapped_t={mapped_t:.8f}->"
                f"{mapped_t_next:.8f} "
                f"patches={len(patches)} "
                f"experts={step_expert_flow_calls} "
                f"experts_per_patch="
                f"{record['expert_count_min']}-"
                f"{record['expert_count_max']} "
                f"(mean={record['expert_count_mean']:.2f}) "
                f"tokens={x_global.feats.shape[0]:,} "
                "velocity_coverage=1.000000 "
                f"velocity_coverage_min="
                f"{coverage.min().item()} "
                f"velocity_coverage_max="
                f"{coverage.max().item()} "
                f"cos_vs_global="
                f"{metrics['cosine_similarity']:.8f} "
                f"rel_l2_vs_global="
                f"{metrics['relative_l2']:.8f} "
                f"mse_vs_global={metrics['mse']:.8e} "
                f"seconds={record['seconds']:.3f}"
            )

        trace = {
            "enabled": True,
            "status": "complete",
            "stage": "texture",
            "mode": mode_name,
            "condition_format_version": condition_version,
            "fusion_mode": fusion_mode,
            "routing": (
                "all_membership_velocity_fusion"
                if fusion_mode == "membership_velocity_fusion"
                else (
                    "hard_center_owner"
                    if fusion_mode == "target_context_hard"
                    else "soft_paired_block"
                )
            ),
            "context_condition": (
                "canonical_full_image"
                if fusion_mode in {
                    "target_context_hard",
                    "membership_velocity_fusion",
                }
                else None
            ),
            "target_condition": (
                "all_covering_local_tiles"
                if fusion_mode == "membership_velocity_fusion"
                else (
                    "hard_owner_local_tile"
                    if fusion_mode == "target_context_hard"
                    else None
                )
            ),
            "velocity_fusion": (
                "normalized_2d_tent_weighted_mean"
                if fusion_mode == "membership_velocity_fusion"
                else None
            ),
            "grid_resolution": 128,
            "patch_size": patch_size,
            "patch_stride": patch_stride,
            "patch_count": len(patches),
            "coverage_histogram": coverage_histogram,
            "start_step": start_step,
            "global_update_count_per_step": 1,
            "velocity_fallback": None,
            "patch_flow_calls": total_patch_flow_calls,
            "expert_flow_calls": total_expert_flow_calls,
            "steps": steps,
            "baseline_final_state": trajectory.states[-1],
            "final_state": x_global.feats.detach().cpu(),
        }
        diagnostics = {
            "patch_coordinate_mode": "local",
            "condition_mode": fusion_mode,
            "routing": (
                "hard_center_owner"
                if fusion_mode == "target_context_hard"
                else "soft_paired_block"
            ),
            "context_condition": (
                "canonical_full_image"
                if fusion_mode == "target_context_hard"
                else None
            ),
            "target_condition": (
                "hard_owner_local_tile"
                if fusion_mode == "target_context_hard"
                else None
            ),
            "velocity_fallback": None,
            "coverage_histogram": coverage_histogram,
            "patch_flow_calls": total_patch_flow_calls,
            "expert_flow_calls": total_expert_flow_calls,
        }
        return x_global, trace, diagnostics

    def _prepare_hr_image_tile_conditions(
        self,
        tiles: List[Dict[str, Any]],
        global_coords: torch.Tensor,
        camera_angle_x: float,
        distance: float,
        mesh_scale: float,
        grid_resolution: int,
    ) -> Dict[str, Any]:
        """Run DINOv3 and NAF independently for every active image tile."""
        active_tiles = [tile for tile in tiles if tile["enabled"]]
        if not active_tiles:
            raise RuntimeError(
                "No image tile has both foreground and sparse texture tokens"
            )
        extraction_started = time.perf_counter()
        extraction_records: List[Dict[str, Any]] = []
        for tile in active_tiles:
            tile_started = time.perf_counter()
            token_indices = tile["token_indices"].to(
                device=global_coords.device,
                dtype=torch.long,
            )
            tile_coords = global_coords[token_indices]
            if tile_coords.shape[0] != int(tile["token_count"]):
                raise RuntimeError(
                    f"Tile {tile['tile_index']} coordinate/token mismatch"
                )
            if not torch.equal(
                tile_coords,
                global_coords.index_select(0, token_indices),
            ):
                raise RuntimeError(
                    f"Tile {tile['tile_index']} changed global token order"
                )
            tile_image = tile.get("image")
            if not isinstance(tile_image, Image.Image):
                raise RuntimeError(
                    f"Tile {tile['tile_index']} is missing its image crop"
                )
            tile_condition = self.get_proj_cond_shape(
                image_cond_model=self.image_cond_model_tex_1024,
                image=[tile_image],
                coords=tile_coords,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                grid_resolution_override=int(grid_resolution),
                projection_crop_box=tile["projection_crop_box"],
            )
            tile["condition_cpu"] = self._pack_proj_condition_cpu(
                tile_condition,
                expected_coords=tile_coords,
                name=f"tile[{tile['tile_index']}]",
            )
            condition_seconds = time.perf_counter() - tile_started
            tile["condition_seconds"] = float(condition_seconds)
            extraction_records.append(
                {
                    "tile_index": int(tile["tile_index"]),
                    "token_count": int(tile["token_count"]),
                    "condition_seconds": float(condition_seconds),
                    "dino_rerun": True,
                    "naf_rerun": bool(
                        getattr(
                            self.image_cond_model_tex_1024,
                            "use_naf_upsample",
                            False,
                        )
                    ),
                    "projection_mode": "global_then_crop_local",
                }
            )
            del tile_condition
            tile["image"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[hr-image-tile-cond] tile={tile['tile_index']:04d} "
                f"tokens={tile['token_count']:,} "
                f"foreground={tile['foreground_ratio']:.6f} "
                f"seconds={condition_seconds:.3f}"
            )
        return {
            "active_tile_count": int(len(active_tiles)),
            "total_seconds": float(time.perf_counter() - extraction_started),
            "tiles": extraction_records,
            "dino_per_active_tile": True,
            "naf_per_active_tile": bool(
                getattr(
                    self.image_cond_model_tex_1024,
                    "use_naf_upsample",
                    False,
                )
            ),
            "features_premerged": False,
        }

    @torch.no_grad()
    def _run_hr_image_tile_texture_flow(
        self,
        flow_model: nn.Module,
        sampler: Any,
        global_noise: SparseTensor,
        global_condition_cpu: Mapping[
            str, Mapping[str, torch.Tensor]
        ],
        shape_concat_cond: SparseTensor,
        sampler_params: Mapping[str, Any],
        global_flow: Any,
        tiles: List[Dict[str, Any]],
        start_step: int,
        fallback_mode: str,
        weight_mode: str,
        condition_extraction: Mapping[str, Any],
    ) -> Tuple[SparseTensor, Dict[str, Any], Dict[str, Any]]:
        """Resume texture sampling with per-tile predictions at each step."""
        trajectory = global_flow.trajectory
        if trajectory is None:
            raise RuntimeError(
                "Global texture flow did not record its trajectory"
            )
        num_steps = len(trajectory.velocities)
        if num_steps != 12 or len(trajectory.states) != 13:
            raise RuntimeError(
                "HR image-tile texture flow requires 13 states and 12 "
                f"velocities, got {len(trajectory.states)} and {num_steps}"
            )
        if not 0 <= int(start_step) <= num_steps:
            raise ValueError(
                f"start_step must lie in [0, {num_steps}], got {start_step}"
            )
        if fallback_mode not in {"saved_global", "current_global"}:
            raise ValueError(
                "fallback_mode must be saved_global or current_global"
            )
        if weight_mode not in {"tent", "uniform"}:
            raise ValueError("weight_mode must be tent or uniform")
        if global_noise.feats.shape[0] != shape_concat_cond.feats.shape[0]:
            raise RuntimeError(
                "Texture noise and shape concat condition token counts differ"
            )
        if not torch.equal(global_noise.coords, shape_concat_cond.coords):
            raise RuntimeError(
                "Texture noise and shape concat condition coordinates differ"
            )

        active_tiles = [tile for tile in tiles if tile["enabled"]]
        if not active_tiles:
            raise RuntimeError("All high-resolution image tiles are disabled")
        for tile in active_tiles:
            if "condition_cpu" not in tile:
                raise RuntimeError(
                    f"Tile {tile['tile_index']} condition was not extracted"
                )

        selected_state = trajectory.states[int(start_step)].to(
            device=global_noise.device,
            dtype=global_noise.dtype,
            copy=True,
        )
        x_global = global_noise.replace(selected_state)
        if not torch.equal(x_global.coords, global_noise.coords):
            raise RuntimeError("Restoring the saved state changed coordinates")

        prediction_kwargs = {
            key: value
            for key, value in sampler_params.items()
            if key
            not in {
                "steps",
                "rescale_t",
                "verbose",
                "tqdm_desc",
                "record_trajectory",
                "trajectory_device",
                "return_model_history",
            }
        }
        raw_times = np.linspace(1.0, 0.0, num_steps + 1).tolist()
        token_count = int(x_global.feats.shape[0])
        merged_velocities: List[torch.Tensor] = []
        step_records: List[Dict[str, Any]] = []
        run_started = time.perf_counter()
        for step_index in range(int(start_step), num_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_started = time.perf_counter()
            mapped_t = float(trajectory.times[step_index])
            mapped_t_next = float(trajectory.times[step_index + 1])
            raw_t = float(raw_times[step_index])
            raw_t_next = float(raw_times[step_index + 1])
            time_interval = float(
                trajectory.time_intervals[step_index]
            )
            if not math.isclose(
                mapped_t - mapped_t_next,
                time_interval,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Trajectory dt mismatch at step {step_index}"
                )

            velocity_sum = torch.zeros(
                x_global.feats.shape,
                device=x_global.device,
                dtype=torch.float32,
            )
            weight_sum = torch.zeros(
                (token_count, 1),
                device=x_global.device,
                dtype=torch.float32,
            )
            coverage_count = torch.zeros(
                token_count,
                device=x_global.device,
                dtype=torch.int32,
            )
            tile_velocity_norms: List[Dict[str, Any]] = []
            for tile in active_tiles:
                token_indices = tile["token_indices"].to(
                    device=x_global.device,
                    dtype=torch.long,
                )
                tile_coords = x_global.coords[token_indices]
                if tile_coords.shape[0] != int(tile["token_count"]):
                    raise RuntimeError(
                        f"Tile {tile['tile_index']} latent gather is misaligned"
                    )
                x_tile = SparseTensor(
                    feats=x_global.feats[token_indices],
                    coords=tile_coords,
                )
                tile_shape_cond = SparseTensor(
                    feats=shape_concat_cond.feats[token_indices],
                    coords=tile_coords,
                )
                if not torch.equal(x_tile.coords, tile_shape_cond.coords):
                    raise RuntimeError(
                        f"Tile {tile['tile_index']} shape/texture coords differ"
                    )
                tile_condition = self._materialize_proj_condition(
                    tile["condition_cpu"],
                    coords=tile_coords,
                    device=x_global.device,
                )
                for branch_name in ("cond", "neg_cond"):
                    projection = tile_condition[branch_name]["proj"]
                    if projection.feats.shape[0] != x_tile.feats.shape[0]:
                        raise RuntimeError(
                            f"Tile {tile['tile_index']} {branch_name} token "
                            "order is not aligned"
                        )
                    if not torch.equal(projection.coords, x_tile.coords):
                        raise RuntimeError(
                            f"Tile {tile['tile_index']} {branch_name} coords "
                            "are not aligned"
                        )
                _, _, tile_velocity = sampler._get_model_prediction(
                    flow_model,
                    x_tile,
                    mapped_t,
                    tile_condition["cond"],
                    neg_cond=tile_condition["neg_cond"],
                    concat_cond=tile_shape_cond,
                    **prediction_kwargs,
                )
                if tile_velocity.feats.shape != x_tile.feats.shape:
                    raise RuntimeError(
                        f"Tile {tile['tile_index']} velocity shape mismatch"
                    )
                if not torch.equal(tile_velocity.coords, tile_coords):
                    raise RuntimeError(
                        f"Tile {tile['tile_index']} velocity coords changed"
                    )
                self._scatter_add_tile_velocity(
                    velocity_sum=velocity_sum,
                    weight_sum=weight_sum,
                    coverage_count=coverage_count,
                    token_indices=token_indices,
                    tile_velocity=tile_velocity.feats,
                    token_weights=tile["weights"],
                )
                tile_velocity_norms.append(
                    {
                        "tile_index": int(tile["tile_index"]),
                        "token_count": int(tile["token_count"]),
                        "velocity_norm": float(
                            tile_velocity.feats.float().norm().item()
                        ),
                    }
                )
                del (
                    x_tile,
                    tile_shape_cond,
                    tile_condition,
                    tile_velocity,
                )

            uncovered_mask = weight_sum[:, 0] <= 0
            uncovered_count = int(uncovered_mask.sum().item())
            saved_global_velocity = trajectory.velocities[step_index].to(
                device=x_global.device,
                dtype=torch.float32,
            )
            current_global_evaluated = False
            if fallback_mode == "current_global" and uncovered_count > 0:
                current_condition = self._materialize_proj_condition(
                    global_condition_cpu,
                    coords=x_global.coords,
                    device=x_global.device,
                )
                _, _, current_global_velocity = (
                    sampler._get_model_prediction(
                        flow_model,
                        x_global,
                        mapped_t,
                        current_condition["cond"],
                        neg_cond=current_condition["neg_cond"],
                        concat_cond=shape_concat_cond,
                        **prediction_kwargs,
                    )
                )
                fallback_velocity = current_global_velocity.feats.to(
                    torch.float32
                )
                current_global_evaluated = True
                del current_condition, current_global_velocity
            else:
                fallback_velocity = saved_global_velocity

            merged_velocity, covered_mask = self._finalize_tile_velocity(
                velocity_sum=velocity_sum,
                weight_sum=weight_sum,
                fallback_velocity=fallback_velocity,
            )
            if not torch.equal(covered_mask, ~uncovered_mask):
                raise RuntimeError("Covered/uncovered mask consistency failed")
            metrics = self._velocity_similarity(
                merged_velocity,
                trajectory.velocities[step_index],
            )
            covered_ratio = float(
                covered_mask.to(torch.float32).mean().item()
            )
            overlap_ratio = float(
                (coverage_count > 1).to(torch.float32).mean().item()
            )
            x_global = x_global.replace(
                x_global.feats
                - time_interval * merged_velocity.to(x_global.dtype)
            )
            if not torch.equal(x_global.coords, global_noise.coords):
                raise RuntimeError(
                    "Image-tile flow changed global coordinates or token order"
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
                allocated = int(torch.cuda.memory_allocated())
                reserved = int(torch.cuda.memory_reserved())
            else:
                peak_allocated = None
                peak_reserved = None
                allocated = None
                reserved = None
            step_seconds = time.perf_counter() - step_started
            step_record = {
                **metrics,
                "step_index": int(step_index),
                "raw_t": raw_t,
                "raw_t_next": raw_t_next,
                "mapped_t": mapped_t,
                "mapped_t_next": mapped_t_next,
                "time_interval": time_interval,
                "active_tile_count": int(len(active_tiles)),
                "tile_token_counts": [
                    {
                        "tile_index": int(tile["tile_index"]),
                        "token_count": int(tile["token_count"]),
                    }
                    for tile in active_tiles
                ],
                "covered_token_count": int(covered_mask.sum().item()),
                "covered_token_ratio": covered_ratio,
                "overlap_token_count": int(
                    (coverage_count > 1).sum().item()
                ),
                "overlap_token_ratio": overlap_ratio,
                "uncovered_token_count": uncovered_count,
                "fallback_mode": fallback_mode,
                "current_global_fallback_evaluated": bool(
                    current_global_evaluated
                ),
                "tile_velocity_norms": tile_velocity_norms,
                "velocity_norm": {
                    "merged": float(merged_velocity.norm().item()),
                    "saved_global": float(
                        saved_global_velocity.norm().item()
                    ),
                    "fallback": float(fallback_velocity.norm().item()),
                },
                "cuda": {
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                },
                "step_seconds": float(step_seconds),
            }
            step_records.append(step_record)
            merged_velocities.append(merged_velocity.detach().cpu())
            print(
                f"[hr-image-tile-flow] step={step_index:02d} "
                f"raw_t={raw_t:.8f}->{raw_t_next:.8f} "
                f"mapped_t={mapped_t:.8f}->{mapped_t_next:.8f} "
                f"tiles={len(active_tiles)} "
                f"covered={covered_ratio:.6f} "
                f"overlap={overlap_ratio:.6f} "
                f"uncovered={uncovered_count:,} "
                f"cos={metrics['cosine_similarity']:.8f} "
                f"rel_l2={metrics['relative_l2']:.8f} "
                f"seconds={step_seconds:.3f}"
            )
            del (
                velocity_sum,
                weight_sum,
                coverage_count,
                saved_global_velocity,
                fallback_velocity,
                merged_velocity,
                covered_mask,
                uncovered_mask,
            )

        trace = {
            "enabled": True,
            "status": "complete",
            "stage": "texture",
            "mode": "hr_image_tile_velocity_flow",
            "global_latent_coordinate_mode": "global",
            "tile_latent_coordinate_mode": "global_subset",
            "token_order": "shape_slat.coords_global_order",
            "start_step": int(start_step),
            "start_step_semantics": (
                f"states[{int(start_step)}] is the starting state; "
                f"velocity steps {int(start_step)}..{num_steps - 1} execute"
            ),
            "start_time": float(trajectory.times[int(start_step)]),
            "fallback_mode": fallback_mode,
            "weight_mode": weight_mode,
            "feature_fusion": False,
            "velocity_fusion": True,
            "global_update_count_per_step": 1,
            "condition_extraction": dict(condition_extraction),
            "tiles": [
                {
                    **self._tile_trace_metadata(tile),
                    "global_indices": tile["token_indices"].detach().cpu(),
                    "condition_seconds": tile.get("condition_seconds"),
                    "condition_models": (
                        {
                            "dino_rerun": True,
                            "naf_rerun": bool(
                                getattr(
                                    self.image_cond_model_tex_1024,
                                    "use_naf_upsample",
                                    False,
                                )
                            ),
                            "projection": "global_then_crop_local",
                        }
                        if tile["enabled"]
                        else None
                    ),
                }
                for tile in tiles
            ],
            "steps": step_records,
            "merged_velocities": merged_velocities,
            "baseline_final_state": trajectory.states[-1],
            "final_state": x_global.feats.detach().cpu(),
            "elapsed_seconds": float(time.perf_counter() - run_started),
        }
        diagnostics = {
            "patch_coordinate_mode": "image_tile_global_subset",
            "global_latent_coordinate_mode": "global",
            "patch_start_step": int(start_step),
            "patch_start_source": "saved_state",
            "patch_guidance_mode": "hr_image_tile_velocity_flow",
            "guidance_strength": float(
                prediction_kwargs.get("guidance_strength", 1.0)
            ),
            "guidance_interval": list(
                prediction_kwargs.get("guidance_interval", (0.0, 1.0))
            ),
            "guidance_rescale": float(
                prediction_kwargs.get("guidance_rescale", 0.0)
            ),
            "wavelet_family": None,
            "skip_residual_mode": "off",
            "concat_cond_present": True,
            "conditional_unconditional_share_concat_cond": True,
            "active_tile_count": int(len(active_tiles)),
            "tile_count": int(len(tiles)),
            "tile_token_counts": [
                {
                    "tile_index": int(tile["tile_index"]),
                    "token_count": int(tile["token_count"]),
                    "enabled": bool(tile["enabled"]),
                }
                for tile in tiles
            ],
            "fallback_mode": fallback_mode,
            "weight_mode": weight_mode,
            "dino_per_active_tile": True,
            "naf_per_active_tile": bool(
                getattr(
                    self.image_cond_model_tex_1024,
                    "use_naf_upsample",
                    False,
                )
            ),
            "feature_fusion": False,
            "velocity_fusion": True,
            "velocity_similarities": step_records,
            "elapsed_seconds": trace["elapsed_seconds"],
        }
        return x_global, trace, diagnostics

    @staticmethod
    def _save_2048_flow_trace(
        output_path: Union[str, Path],
        stage_name: str,
        coords: torch.Tensor,
        sampler_params: Dict[str, Any],
        global_flow: Any,
        patch_trace: Dict[str, Any],
        experiment_tag: Optional[str] = None,
    ) -> int:
        """Atomically persist every global state/velocity and patch results."""
        if stage_name not in {"shape", "texture"}:
            raise ValueError("stage_name must be shape or texture")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        trajectory = global_flow.trajectory
        image_tile_mode = (
            patch_trace.get("mode") == "hr_image_tile_velocity_flow"
        )
        if image_tile_mode:
            num_steps = len(trajectory.velocities)
            payload = {
                "format": f"pixal3d_2048_{stage_name}_flow_trace_v4",
                "stage": stage_name,
                "experiment_tag": experiment_tag,
                "coordinate_system": {
                    "source_candidate_resolution": 512,
                    "target_resolution": 2048,
                    "target_grid_resolution": 128,
                    "token_order": "global_hr_coords_unique_order",
                    "global_latent_coordinate_mode": "global",
                    "patch_model_coordinate_mode": "global_subset",
                    "image_projection_mode": "global_then_crop_local",
                },
                "coords": coords.detach().cpu(),
                "sampler": dict(sampler_params),
                "global_flow": {
                    "raw_times": torch.linspace(
                        1.0,
                        0.0,
                        num_steps + 1,
                        dtype=torch.float64,
                    ),
                    "mapped_times": torch.tensor(
                        trajectory.times, dtype=torch.float64
                    ),
                    "times": torch.tensor(
                        trajectory.times, dtype=torch.float64
                    ),
                    "time_intervals": torch.tensor(
                        trajectory.time_intervals, dtype=torch.float64
                    ),
                    "states": trajectory.states,
                    "velocities": trajectory.velocities,
                    "velocity_semantics": (
                        "final Euler velocity after conditional/unconditional "
                        "CFG, guidance interval, and guidance rescale"
                    ),
                    "guidance": {
                        "strength": sampler_params.get(
                            "guidance_strength"
                        ),
                        "interval": sampler_params.get(
                            "guidance_interval"
                        ),
                        "rescale": sampler_params.get(
                            "guidance_rescale"
                        ),
                        "time_rescale": sampler_params.get("rescale_t"),
                    },
                    "base_latent_state_index": (
                        len(trajectory.states) - 1
                    ),
                    "baseline_final_state": trajectory.states[-1],
                },
                "patch_flow": patch_trace,
            }
        else:
            # Preserve the pre-existing trace schema byte-for-byte in the
            # disabled/spatial-patch paths.
            payload = {
                "format": f"pixal3d_2048_{stage_name}_flow_trace_v3",
                "stage": stage_name,
                "experiment_tag": experiment_tag,
                "coordinate_system": {
                    "source_candidate_resolution": 512,
                    "target_resolution": 2048,
                    "target_grid_resolution": 128,
                    "token_order": "global_hr_coords_unique_order",
                    "global_latent_coordinate_mode": "global",
                    "patch_model_coordinate_mode": "local",
                },
                "coords": coords.detach().cpu(),
                "sampler": dict(sampler_params),
                "global_flow": {
                    "times": torch.tensor(
                        trajectory.times, dtype=torch.float64
                    ),
                    "time_intervals": torch.tensor(
                        trajectory.time_intervals, dtype=torch.float64
                    ),
                    "states": trajectory.states,
                    "velocities": trajectory.velocities,
                    "base_latent_state_index": (
                        len(trajectory.states) - 1
                    ),
                },
                "patch_flow": patch_trace,
            }
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
        return int(output_path.stat().st_size)
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        camera_params: dict,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        shape_flow_trace_path: Optional[Union[str, Path]] = None,
        texture_flow_trace_path: Optional[Union[str, Path]] = None,
        shape_patch_start_step: int = 6,
        shape_patch_start_source: str = "saved_state",
        shape_guidance_mode: str = "original_cfg",
        shape_guidance_strength: float = 7.5,
        shape_guidance_interval: str = "original_interval",
        shape_guidance_rescale: float = 0.0,
        shape_wavelet_family: str = "haar",
        shape_skip_residual_mode: str = "off",
        texture_patch_start_step: int = 6,
        texture_patch_start_source: str = "saved_state",
        texture_guidance_mode: str = "global_original",
        texture_guidance_strength: float = 1.0,
        texture_guidance_interval: str = "original_interval",
        texture_guidance_rescale: float = 0.0,
        texture_wavelet_family: str = "haar",
        texture_skip_residual_mode: str = "off",
        flow_experiment_tag: Optional[str] = None,
        hr_texture_context: Optional[Mapping[str, Any]] = None,
        hr_image_tile_texture_flow: bool = False,
        hr_image_tile_size: int = 1024,
        hr_image_tile_stride: int = 1024,
        hr_image_tile_start_step: int = 6,
        hr_image_tile_min_foreground_ratio: float = 0.0,
        hr_image_tile_fallback: str = "saved_global",
        hr_image_tile_weight: str = "tent",
        hr_image_tile_save_debug: bool = False,
        hr_image_tile_debug_dir: Optional[Union[str, Path]] = None,
        texture_multitile_3d_patch_flow: bool = False,
        texture_multitile_start_step: int = 6,
        texture_canonical_image_size: int = 4096,
        texture_image_tile_size: int = 1024,
        texture_image_tile_stride: int = 512,
        texture_3d_patch_size: int = 64,
        texture_3d_patch_stride: int = 32,
        texture_multitile_global_mode: str = "paired_block_fusion",
        texture_multitile_save_debug: bool = False,
        texture_multitile_debug_dir: Optional[Union[str, Path]] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the Pixal3D pipeline (proj mode, cascade).

        Args:
            image (Image.Image): The image prompt.
            camera_params (dict): Camera parameters with keys:
                - camera_angle_x (float): Horizontal FOV in radians.
                - distance (float): Camera distance.
                - mesh_scale (float): Mesh scale factor.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): Cascade type, including experimental '2048_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            shape_flow_trace_path: Required output artifact for 2048 shape flow.
            texture_flow_trace_path: Required output artifact for 2048 texture flow.
            shape_*: Independent local guidance controls for shape flow.
            texture_*: Independent global/local guidance controls for texture flow.
            flow_experiment_tag: Unique output/cache identity for this flow.
            hr_texture_context: Shared-preprocessing bundle containing the HR
                image, aligned foreground mask, and global-to-HR transform.
            hr_image_tile_texture_flow: Replace the configured latter texture
                steps with synchronized high-resolution image-tile velocities.
            hr_image_tile_start_step: ``states[start_step]`` is restored and
                velocity steps ``start_step..11`` are executed.
        """
        # Check pipeline type
        self.last_shape_flow_diagnostics = None
        self.last_texture_flow_diagnostics = None
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1024
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1536
        elif pipeline_type == '2048_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            if shape_flow_trace_path is None:
                raise ValueError(
                    "shape_flow_trace_path is required for the 2048 experiment"
                )
            if texture_flow_trace_path is None:
                raise ValueError(
                    "texture_flow_trace_path is required for the 2048 experiment"
                )
            hr_resolution = 2048
        else:
            raise ValueError(f"Invalid pipeline type for Pixal3D proj mode: {pipeline_type}. "
                             f"Supported: '1024_cascade', '1536_cascade', '2048_cascade'.")

        # Validate image_cond_models are set
        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set."
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set."
        assert self.image_cond_model_tex_1024 is not None, "image_cond_model_tex_1024 not set."

        # Extract camera params
        camera_angle_x = camera_params['camera_angle_x']
        distance = camera_params['distance']
        mesh_scale = camera_params.get('mesh_scale', 1.0)
        
        if preprocess_image:
            hr_texture_context = self.preprocess_canonical_images(image)
        if hr_texture_context is not None and "image_4096" in hr_texture_context:
            image_4096 = hr_texture_context["image_4096"]
            image_1024 = hr_texture_context["image_1024"]
            image_512 = hr_texture_context["image_512"]
            image = image_1024
        else:
            image_4096 = image
            image_1024 = image
            image_512 = image
        if texture_multitile_3d_patch_flow:
            if pipeline_type != "2048_cascade":
                raise ValueError("multi-tile 3D patch flow requires 2048_cascade")
            if hr_image_tile_texture_flow:
                raise ValueError("legacy and paired image tile flows are exclusive")
            if texture_guidance_mode != "global_original":
                raise ValueError("paired multi-tile flow requires global_original")
            expected = (4096, 1024, 512, 64, 32)
            actual = (
                texture_canonical_image_size,
                texture_image_tile_size,
                texture_image_tile_stride,
                texture_3d_patch_size,
                texture_3d_patch_stride,
            )
            if actual != expected:
                raise ValueError(
                    "paired v1 requires canonical/tile/stride/patch/stride "
                    f"{expected}, got {actual}"
                )
            if texture_multitile_global_mode not in {
                "paired_block_fusion",
                "target_context_hard",
                "membership_velocity_fusion",
            }:
                raise ValueError(
                    "texture_multitile_global_mode must be one of "
                    "paired_block_fusion, target_context_hard, "
                    "membership_velocity_fusion"
                )

            if not 0 <= texture_multitile_start_step <= 12:
                raise ValueError("texture_multitile_start_step must be in [0,12]")
            if hr_texture_context is None or "foreground_mask_4096" not in hr_texture_context:
                raise ValueError("canonical preprocessing context is required")
        if hr_image_tile_texture_flow:
            if pipeline_type != "2048_cascade":
                raise ValueError(
                    "HR image-tile texture flow currently requires "
                    "pipeline_type='2048_cascade'"
                )
            if texture_guidance_mode != "global_original":
                raise ValueError(
                    "HR image-tile texture flow uses the original texture CFG "
                    "semantics and cannot be combined with --texture-mode "
                    "local patch guidance"
                )
            if hr_texture_context is None:
                raise ValueError(
                    "hr_texture_context is required when preprocessing was "
                    "performed outside pipeline.run"
                )
            required_context_keys = {
                "global_image",
                "hr_image",
                "foreground_mask_hr",
                "global_to_hr_transform",
            }
            missing_context_keys = required_context_keys - set(
                hr_texture_context
            )
            if missing_context_keys:
                raise ValueError(
                    "hr_texture_context is missing: "
                    f"{sorted(missing_context_keys)}"
                )
            context_global_image = hr_texture_context["global_image"]
            if not isinstance(context_global_image, Image.Image):
                raise TypeError(
                    "hr_texture_context['global_image'] must be a PIL image"
                )
            if image.size != context_global_image.size or not np.array_equal(
                np.asarray(image.convert("RGB")),
                np.asarray(context_global_image.convert("RGB")),
            ):
                raise RuntimeError(
                    "pipeline image is not the global image from the shared "
                    "HR preprocessing operation"
                )
            if int(hr_image_tile_size) <= 0:
                raise ValueError("hr_image_tile_size must be positive")
            if (
                int(hr_image_tile_stride) <= 0
                or int(hr_image_tile_stride) > int(hr_image_tile_size)
            ):
                raise ValueError(
                    "hr_image_tile_stride must be positive and no greater "
                    "than hr_image_tile_size"
                )
            if not 0 <= int(hr_image_tile_start_step) <= 12:
                raise ValueError(
                    "hr_image_tile_start_step must lie in [0, 12]"
                )
            if (
                not math.isfinite(hr_image_tile_min_foreground_ratio)
                or not 0.0
                <= float(hr_image_tile_min_foreground_ratio)
                <= 1.0
            ):
                raise ValueError(
                    "hr_image_tile_min_foreground_ratio must lie in [0, 1]"
                )
            if hr_image_tile_fallback not in {
                "saved_global",
                "current_global",
            }:
                raise ValueError(
                    "hr_image_tile_fallback must be saved_global or "
                    "current_global"
                )
            if hr_image_tile_weight not in {"tent", "uniform"}:
                raise ValueError(
                    "hr_image_tile_weight must be tent or uniform"
                )
            if (
                hr_image_tile_save_debug
                and hr_image_tile_debug_dir is None
            ):
                raise ValueError(
                    "hr_image_tile_debug_dir is required when saving debug "
                    "artifacts"
                )
        torch.manual_seed(seed)

        # ---- Stage 1: Sparse Structure (proj) ----
        cond_ss = self.get_proj_cond_ss(
            [image_512],
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        ss_res = 32
        coords = self.sample_sparse_structure(
            cond_ss, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        del cond_ss
        torch.cuda.empty_cache()

        # ---- Stage 2: Shape LR 512 (proj) ----
        cond_shape_lr = self.get_proj_cond_shape(
            self.image_cond_model_shape_512, [image_512], coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        lr_slat = self.sample_shape_slat(
            cond_shape_lr, self.models['shape_slat_flow_model_512'],
            coords, shape_slat_sampler_params
        )
        del cond_shape_lr
        torch.cuda.empty_cache()

        # ---- Stage 3a: Upsample LR → HR ----
        print("Stage 3a: Upsample LR → HR")
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(lr_slat, upsample_times=4) # 32 * 16 = 512
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        if hr_resolution == 2048:
            grid_res = 128
            # Preserve the original candidate order up to the same unique()
            # operation used by Pixal3D, but requantize explicitly onto the
            # fixed 128-grid: floor((coord + 0.5) / 512 * 128).
            quant_coords = torch.cat([
                hr_coords[:, :1],
                (
                    (hr_coords[:, 1:] + 0.5)
                    / lr_resolution
                    * grid_res
                ).int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            num_tokens = hr_coords_unique.shape[0]
            if num_tokens > max_num_tokens:
                raise RuntimeError(
                    "The fixed 2048 coordinate set exceeds max_num_tokens: "
                    f"tokens={num_tokens}, max_num_tokens={max_num_tokens}. "
                    "Increase --max-num-tokens; 2048 mode will not silently "
                    "reduce the grid resolution."
                )
            print(
                f"[shape-2048] fixed_grid=128 tokens={num_tokens:,} "
                "source_candidates=512-grid"
            )
        else:
            while True:
                grid_res = actual_hr_resolution // 16
                quant_coords = torch.cat([
                    hr_coords[:, :1],
                    ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
                ], dim=1)
                hr_coords_unique = quant_coords.unique(dim=0)
                num_tokens = hr_coords_unique.shape[0]
                if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
                    break
                actual_hr_resolution -= 128

        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        # ---- Stage 3b: Shape HR (proj) ----
        print("Stage 3b: Shape HR (proj)")
        cond_shape_hr = self.get_proj_cond_shape(
            self.image_cond_model_shape_1024, [image_1024], hr_coords_unique,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res,
        )
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], self.models['shape_slat_flow_model_1024'].in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        flow_model_hr = self.models['shape_slat_flow_model_1024']
        if self.low_vram:
            flow_model_hr.to(self.device)
        # 分辨率是actual_grid_res = actual_hr_resolution // 16

        if actual_hr_resolution == 2048:
            if int(flow_model_hr.in_channels) != 32:
                raise RuntimeError(
                    "The 2048 shape experiment requires N x 32 noise, "
                    f"but the flow model reports {flow_model_hr.in_channels} channels"
                )
            global_flow = self.shape_slat_sampler.sample(
                flow_model_hr,
                noise_hr,
                **cond_shape_hr,
                **sampler_params_hr,
                verbose=True,
                tqdm_desc="Sampling global HR shape SLat (proj, 2048)",
                record_trajectory=True,
                trajectory_device="cpu",
                return_model_history=False,
            )
            # Publish the expensive global baseline before patch inference so
            # its 13 states and 12 final CFG velocities survive a later patch
            # OOM or interruption. The completed patch result atomically
            # replaces this artifact below.
            baseline_trace_size = self._save_2048_flow_trace(
                output_path=shape_flow_trace_path,
                stage_name="shape",
                coords=hr_coords_unique,
                sampler_params=sampler_params_hr,
                global_flow=global_flow,
                patch_trace={
                    "enabled": True,
                    "status": "global_complete_patch_pending",
                    "stage": "shape",
                    "patch_coordinate_mode": "local",
                    "global_latent_coordinate_mode": "global",
                    "grid_resolution": 128,
                    "patch_size": 64,
                    "patch_stride": 32,
                    "patch_count": 27,
                    "start_step": int(shape_patch_start_step),
                    "start_source": str(shape_patch_start_source),
                    "guidance": {
                        "mode": str(shape_guidance_mode),
                        "strength": float(shape_guidance_strength),
                        "interval": str(shape_guidance_interval),
                        "rescale": float(shape_guidance_rescale),
                        "wavelet_family": str(shape_wavelet_family),
                        "skip_residual_mode": str(
                            shape_skip_residual_mode
                        ),
                        "wavelet_compute_dtype": (
                            "float32"
                            if shape_guidance_mode == "wavelet_cfg"
                            else None
                        ),
                        "dense_wavelet_bands_saved": False,
                    },
                },
                experiment_tag=flow_experiment_tag,
            )
            print(
                f"[shape-2048-trace] global baseline saved="
                f"{shape_flow_trace_path} bytes={baseline_trace_size:,}"
            )
            hr_slat, patch_trace, flow_diagnostics = self._run_2048_patch_flow(
                flow_model=flow_model_hr,
                sampler=self.shape_slat_sampler,
                stage_name="shape",
                global_noise=noise_hr,
                cond=cond_shape_hr,
                concat_cond=None,
                sampler_params=sampler_params_hr,
                global_flow=global_flow,
                start_step=int(shape_patch_start_step),
                start_source=str(shape_patch_start_source),
                patch_guidance_mode=str(shape_guidance_mode),
                guidance_strength=float(shape_guidance_strength),
                guidance_interval=str(shape_guidance_interval),
                guidance_rescale=float(shape_guidance_rescale),
                wavelet_family=str(shape_wavelet_family),
                skip_residual_mode=str(shape_skip_residual_mode),
            )
            trace_size = self._save_2048_flow_trace(
                output_path=shape_flow_trace_path,
                stage_name="shape",
                coords=hr_coords_unique,
                sampler_params=sampler_params_hr,
                global_flow=global_flow,
                patch_trace=patch_trace,
                experiment_tag=flow_experiment_tag,
            )
            flow_diagnostics.update(
                {
                    "trace_path": str(Path(shape_flow_trace_path).resolve()),
                    "trace_bytes": int(trace_size),
                    "global_state_count": len(global_flow.trajectory.states),
                    "global_velocity_count": len(
                        global_flow.trajectory.velocities
                    ),
                    "experiment_tag": flow_experiment_tag,
                }
            )
            self.last_shape_flow_diagnostics = flow_diagnostics
            print(
                f"[shape-2048-trace] saved={shape_flow_trace_path} "
                f"bytes={trace_size:,}"
            )
            del patch_trace, global_flow
        else:
            hr_slat = self.shape_slat_sampler.sample(
                flow_model_hr,
                noise_hr,
                **cond_shape_hr,
                **sampler_params_hr,
                verbose=True,
                tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
            ).samples

        # hr_slat = self.shape_slat_sampler.sample(
        #         flow_model_hr,
        #         noise_hr,
        #         **cond_shape_hr,
        #         **sampler_params_hr,
        #         verbose=True,
        #         tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
        #     ).samples

        if self.low_vram:
            flow_model_hr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(hr_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        # ---- Stage 4: Texture (proj) ----
        tex_grid_res = actual_hr_resolution // 16
        cond_tex = self.get_proj_cond_shape(
            self.image_cond_model_tex_1024, [image_1024], shape_slat.coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=tex_grid_res,
        )
        if actual_hr_resolution == 2048:
            shape_std = torch.tensor(
                self.shape_slat_normalization['std']
            )[None].to(shape_slat.device)
            shape_mean = torch.tensor(
                self.shape_slat_normalization['mean']
            )[None].to(shape_slat.device)
            shape_concat_cond = (shape_slat - shape_mean) / shape_std
            tex_flow_model = self.models['tex_slat_flow_model_1024']
            tex_noise_channels = (
                int(tex_flow_model.in_channels)
                - int(shape_concat_cond.feats.shape[1])
            )
            if tex_noise_channels <= 0:
                raise RuntimeError(
                    "Texture flow input channels must exceed shape condition "
                    f"channels, got {tex_flow_model.in_channels} and "
                    f"{shape_concat_cond.feats.shape[1]}"
                )
            tex_noise = shape_concat_cond.replace(
                feats=torch.randn(
                    shape_concat_cond.coords.shape[0],
                    tex_noise_channels,
                    device=self.device,
                )
            )
            if not torch.equal(tex_noise.coords, shape_concat_cond.coords):
                raise RuntimeError("Texture noise and shape condition misaligned")
            tex_sampler_params = {
                **self.tex_slat_sampler_params,
                **tex_slat_sampler_params,
            }
            if self.low_vram:
                tex_flow_model.to(self.device)
            global_tex_flow = self.tex_slat_sampler.sample(
                tex_flow_model,
                tex_noise,
                concat_cond=shape_concat_cond,
                **cond_tex,
                **tex_sampler_params,
                verbose=True,
                tqdm_desc="Sampling global texture SLat (proj, 2048)",
                record_trajectory=True,
                trajectory_device="cpu",
                return_model_history=False,
            )
            texture_patch_enabled = (
                texture_guidance_mode != "global_original"
            )
            texture_image_tile_enabled = bool(
                hr_image_tile_texture_flow
            )
            texture_multitile_enabled = bool(
                texture_multitile_3d_patch_flow
            )
            texture_second_pass_enabled = bool(
                texture_patch_enabled
                or texture_image_tile_enabled
                or texture_multitile_enabled
            )
            pending_texture_trace = {
                "enabled": texture_second_pass_enabled,
                "status": (
                    "global_complete_multitile_3d_pending"
                    if texture_multitile_enabled
                    else (
                        "global_complete_image_tile_pending"
                        if texture_image_tile_enabled
                        else (
                            "global_complete_patch_pending"
                            if texture_patch_enabled
                            else "global_only_complete"
                        )
                    )
                ),
                "stage": "texture",
                "mode": (
                    "multi_tile_paired_3d_patch_flow"
                    if texture_multitile_enabled
                    else (
                        "hr_image_tile_velocity_flow"
                        if texture_image_tile_enabled
                        else (
                            "spatial_patch_flow"
                            if texture_patch_enabled
                            else "global_original"
                        )
                    )
                ),
                "patch_coordinate_mode": (
                    (
                        "global_subset"
                        if texture_image_tile_enabled
                        else "local"
                    )
                    if texture_second_pass_enabled
                    else None
                ),
                "global_latent_coordinate_mode": "global",
                "grid_resolution": 128,
                "patch_size": (
                    None if texture_image_tile_enabled else int(texture_3d_patch_size)
                ),
                "patch_stride": (
                    None if texture_image_tile_enabled else int(texture_3d_patch_stride)
                ),
                "patch_count": (
                    None if texture_image_tile_enabled else 27
                ),
                "start_step": int(
                    texture_multitile_start_step
                    if texture_multitile_enabled
                    else (
                        hr_image_tile_start_step
                        if texture_image_tile_enabled
                        else texture_patch_start_step
                    )
                ),
                "start_source": (
                    "saved_state"
                    if texture_image_tile_enabled
                    else str(texture_patch_start_source)
                ),
                "concat_cond": {
                    "present": True,
                    "token_aligned": True,
                    "same_for_conditional_and_unconditional": True,
                    "channels": int(shape_concat_cond.feats.shape[1]),
                },
                "guidance": {
                    "mode": str(texture_guidance_mode),
                    "strength": float(texture_guidance_strength),
                    "interval": str(texture_guidance_interval),
                    "rescale": float(texture_guidance_rescale),
                    "wavelet_family": str(texture_wavelet_family),
                    "skip_residual_mode": str(
                        texture_skip_residual_mode
                    ),
                    "wavelet_compute_dtype": (
                        "float32"
                        if texture_guidance_mode == "wavelet_cfg"
                        else None
                    ),
                    "dense_wavelet_bands_saved": False,
                },
                "hr_image_tiles": (
                    {
                        "tile_size": int(hr_image_tile_size),
                        "tile_stride": int(hr_image_tile_stride),
                        "start_step": int(hr_image_tile_start_step),
                        "fallback": str(hr_image_tile_fallback),
                        "weight": str(hr_image_tile_weight),
                        "min_foreground_ratio": float(
                            hr_image_tile_min_foreground_ratio
                        ),
                        "save_debug": bool(hr_image_tile_save_debug),
                        "condition_semantics": (
                            "DINOv3+NAF rerun per active tile; global camera "
                            "projection transformed to tile-local coordinates"
                        ),
                        "merge_semantics": (
                            "per-step CFG velocity scatter/weighted merge; "
                            "one global Euler update"
                        ),
                    }
                    if texture_image_tile_enabled
                    else None
                ),
            }
            if not texture_image_tile_enabled and not texture_multitile_enabled:
                # Keep the existing disabled/spatial-patch trace payload
                # exactly unchanged.
                pending_texture_trace = {
                    "enabled": bool(texture_patch_enabled),
                    "status": (
                        "global_complete_patch_pending"
                        if texture_patch_enabled
                        else "global_only_complete"
                    ),
                    "stage": "texture",
                    "patch_coordinate_mode": (
                        "local" if texture_patch_enabled else None
                    ),
                    "global_latent_coordinate_mode": "global",
                    "grid_resolution": 128,
                    "patch_size": 64,
                    "patch_stride": 32,
                    "patch_count": 27,
                    "start_step": int(texture_patch_start_step),
                    "start_source": str(texture_patch_start_source),
                    "concat_cond": {
                        "present": True,
                        "token_aligned": True,
                        "same_for_conditional_and_unconditional": True,
                        "channels": int(shape_concat_cond.feats.shape[1]),
                    },
                    "guidance": {
                        "mode": str(texture_guidance_mode),
                        "strength": float(texture_guidance_strength),
                        "interval": str(texture_guidance_interval),
                        "rescale": float(texture_guidance_rescale),
                        "wavelet_family": str(texture_wavelet_family),
                        "skip_residual_mode": str(
                            texture_skip_residual_mode
                        ),
                        "wavelet_compute_dtype": (
                            "float32"
                            if texture_guidance_mode == "wavelet_cfg"
                            else None
                        ),
                        "dense_wavelet_bands_saved": False,
                    },
                }
            baseline_tex_trace_size = self._save_2048_flow_trace(
                output_path=texture_flow_trace_path,
                stage_name="texture",
                coords=shape_slat.coords,
                sampler_params=tex_sampler_params,
                global_flow=global_tex_flow,
                patch_trace=pending_texture_trace,
                experiment_tag=flow_experiment_tag,
            )
            print(
                f"[texture-2048-trace] global baseline saved="
                f"{texture_flow_trace_path} bytes={baseline_tex_trace_size:,}"
            )
            if texture_multitile_enabled:
                if texture_multitile_start_step == 12:
                    # Strict identity: no projection, tile DINO/NAF, or patch
                    # model call is allowed at the final trajectory state.
                    final_state = global_tex_flow.trajectory.states[12].to(
                        device=tex_noise.device,
                        dtype=tex_noise.dtype,
                        copy=True,
                    )
                    tex_slat_normalized = tex_noise.replace(final_state)
                    texture_patch_trace = {
                        "enabled": True,
                        "status": "identity_start_step_final",
                        "stage": "texture",
                        "mode": "multi_tile_paired_3d_patch_flow",
                        "condition_format_version": "multi_tile_paired_v1",
                        "start_step": 12,
                        "tile_condition_calls": 0,
                        "patch_flow_calls": 0,
                        "final_state": final_state.detach().cpu(),
                    }
                    texture_diagnostics = {
                        "patch_coordinate_mode": "local",
                        "global_latent_coordinate_mode": "global",
                        "patch_guidance_mode": "multi_tile_paired_3d_patch_flow",
                        "concat_cond_present": True,
                        "conditional_unconditional_share_concat_cond": True,
                        "identity": True,
                        "tile_condition_calls": 0,
                        "patch_flow_calls": 0,
                        "velocity_fallback": None,
                    }
                else:
                    projected_full_norm, projected_depth, projection_valid = (
                        self._project_sparse_coords_to_image_norm(
                            image_cond_model=self.image_cond_model_tex_1024,
                            coords=shape_slat.coords,
                            camera_angle_x=camera_angle_x,
                            distance=distance,
                            mesh_scale=mesh_scale,
                            grid_resolution=tex_grid_res,
                        )
                    )
                    if not torch.isfinite(projected_full_norm).all():
                        raise RuntimeError("texture projection contains NaN/Inf")
                    # Projection-valid and foreground masks are diagnostics,
                    # never membership filters.
                    paired_condition, tile_summary = (
                        self._prepare_multitile_paired_condition(
                            image_4096=image_4096,
                            foreground_mask_4096=hr_texture_context[
                                "foreground_mask_4096"
                            ],
                            global_coords=shape_slat.coords,
                            projected_full_norm=projected_full_norm,
                            camera_angle_x=camera_angle_x,
                            distance=distance,
                            mesh_scale=mesh_scale,                            base_condition=cond_tex["cond"],

                            grid_resolution=tex_grid_res,
                            tile_size=texture_image_tile_size,
                            tile_stride=texture_image_tile_stride,
                            save_slot_proj=texture_multitile_save_debug,
                        )
                    )
                    tex_slat_normalized, texture_patch_trace, texture_diagnostics = (
                        self._run_multitile_3d_patch_texture_flow(
                            flow_model=tex_flow_model,
                            sampler=self.tex_slat_sampler,
                            global_noise=tex_noise,
                            shape_concat_cond=shape_concat_cond,
                            sampler_params=tex_sampler_params,
                            global_flow=global_tex_flow,
                            condition=paired_condition,
                            start_step=texture_multitile_start_step,                            fusion_mode=texture_multitile_global_mode,

                            patch_size=texture_3d_patch_size,
                            patch_stride=texture_3d_patch_stride,
                        )
                    )
                    texture_patch_trace.update(
                        {
                            "canonical_preprocessing_version": "canonical_v1",
                            "canonical_image_size": texture_canonical_image_size,
                            "image_tile_size": texture_image_tile_size,
                            "image_tile_stride": texture_image_tile_stride,
                            "tile_count": tile_summary["tile_count"],
                            "multi_tile_fusion_mode": texture_multitile_global_mode,
                            "condition_summary": {
                                key: value for key, value in tile_summary.items()
                                if key not in {
                                    "raw_uv", "assignment_uv", "slot_proj"
                                }
                            },
                            "projection_valid_count_diagnostic_only": int(
                                projection_valid.sum().item()
                            ),
                        }
                    )
                    texture_diagnostics.update(
                        {
                            "patch_guidance_mode": "multi_tile_paired_3d_patch_flow",
                            "global_latent_coordinate_mode": "global",
                            "concat_cond_present": True,
                            "conditional_unconditional_share_concat_cond": True,
                            "dino_per_active_tile": True,
                            "naf_per_active_tile": bool(
                                getattr(
                                    self.image_cond_model_tex_1024,
                                    "use_naf_upsample",
                                    False,
                                )
                            ),
                        }
                    )
                    if texture_multitile_save_debug:
                        if texture_multitile_debug_dir is None:
                            raise ValueError(
                                "texture_multitile_debug_dir is required"
                            )
                        debug_dir = Path(texture_multitile_debug_dir)
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            paired_condition["global_bank"].detach().cpu(),
                            debug_dir / "tile_global_bank.pt",
                        )
                        torch.save(
                            paired_condition["tile_ids"].detach().cpu(),
                            debug_dir / "token_tile_ids.pt",
                        )
                        torch.save(
                            paired_condition["tile_weights"].detach().cpu(),
                            debug_dir / "token_tile_weights.pt",
                        )
                        torch.save(
                            paired_condition["proj"].feats.detach().cpu(),
                            debug_dir / "fused_proj.pt",
                        )
                        if tile_summary["slot_proj"] is not None:
                            torch.save(
                                tile_summary["slot_proj"],
                                debug_dir / "optional_slot_proj.pt",
                            )
                    del (
                        paired_condition,
                        tile_summary,
                        projected_full_norm,
                        projected_depth,
                        projection_valid,
                    )
            elif texture_image_tile_enabled:
                hr_image = hr_texture_context["hr_image"]
                foreground_mask_hr = hr_texture_context[
                    "foreground_mask_hr"
                ]
                global_to_hr_transform = hr_texture_context[
                    "global_to_hr_transform"
                ]
                if not isinstance(hr_image, Image.Image):
                    raise TypeError(
                        "hr_texture_context['hr_image'] must be a PIL image"
                    )
                if not isinstance(foreground_mask_hr, Image.Image):
                    raise TypeError(
                        "hr_texture_context['foreground_mask_hr'] must be "
                        "a PIL image"
                    )
                expected_global_size = list(image.size)
                expected_hr_size = list(hr_image.size)
                if list(
                    global_to_hr_transform.get("global_size", [])
                ) != expected_global_size:
                    raise RuntimeError(
                        "global_to_hr_transform global size is inconsistent"
                    )
                if list(
                    global_to_hr_transform.get("hr_size", [])
                ) != expected_hr_size:
                    raise RuntimeError(
                        "global_to_hr_transform HR size is inconsistent"
                    )
                global_to_hr_matrix = np.asarray(
                    global_to_hr_transform["global_to_hr_matrix"],
                    dtype=np.float64,
                )
                hr_to_global_matrix = np.asarray(
                    global_to_hr_transform["hr_to_global_matrix"],
                    dtype=np.float64,
                )
                if (
                    global_to_hr_matrix.shape != (3, 3)
                    or hr_to_global_matrix.shape != (3, 3)
                    or not np.allclose(
                        global_to_hr_matrix @ hr_to_global_matrix,
                        np.eye(3),
                        rtol=0.0,
                        atol=1e-10,
                    )
                ):
                    raise RuntimeError(
                        "Global-to-HR transform round-trip check failed"
                    )

                projected_full_norm, projected_depth, projection_valid = (
                    self._project_sparse_coords_to_image_norm(
                        image_cond_model=self.image_cond_model_tex_1024,
                        coords=shape_slat.coords,
                        camera_angle_x=camera_angle_x,
                        distance=distance,
                        mesh_scale=mesh_scale,
                        grid_resolution=tex_grid_res,
                    )
                )
                tiles, tile_summary = self._build_hr_image_tiles(
                    hr_image=hr_image,
                    foreground_mask_hr=foreground_mask_hr,
                    projected_full_norm=projected_full_norm,
                    projection_valid=projection_valid,
                    tile_size=int(hr_image_tile_size),
                    tile_stride=int(hr_image_tile_stride),
                    min_foreground_ratio=float(
                        hr_image_tile_min_foreground_ratio
                    ),
                    weight_mode=str(hr_image_tile_weight),
                )
                print(
                    f"[hr-image-tiles] hr={hr_image.width}x{hr_image.height} "
                    f"tile={int(hr_image_tile_size)} "
                    f"stride={int(hr_image_tile_stride)} "
                    f"active={tile_summary['active_tile_count']}/"
                    f"{tile_summary['tile_count']} "
                    f"eligible_tokens="
                    f"{tile_summary['eligible_token_count']:,} "
                    f"covered_tokens="
                    f"{tile_summary['covered_token_count']:,} "
                    f"overlap_tokens="
                    f"{tile_summary['overlap_token_count']:,}"
                )
                debug_artifacts: Dict[str, Any] = {}
                if hr_image_tile_save_debug:
                    debug_artifacts = self._save_hr_image_tile_debug(
                        debug_dir=hr_image_tile_debug_dir,
                        global_image=image,
                        hr_image=hr_image,
                        foreground_mask_hr=foreground_mask_hr,
                        projected_full_norm=projected_full_norm,
                        projection_valid=projection_valid,
                        tiles=tiles,
                        global_to_hr_transform=global_to_hr_transform,
                    )

                global_condition_cpu = self._pack_proj_condition_cpu(
                    cond_tex,
                    expected_coords=shape_slat.coords,
                    name="global_texture_condition",
                )
                # The global condition is now represented by plain CPU tensors;
                # release its large GPU projection before DINO/NAF tile passes.
                del cond_tex
                cond_tex = None
                if self.low_vram:
                    tex_flow_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                condition_extraction = (
                    self._prepare_hr_image_tile_conditions(
                        tiles=tiles,
                        global_coords=shape_slat.coords,
                        camera_angle_x=camera_angle_x,
                        distance=distance,
                        mesh_scale=mesh_scale,
                        grid_resolution=tex_grid_res,
                    )
                )
                if self.low_vram:
                    tex_flow_model.to(self.device)
                tex_slat_normalized, texture_patch_trace, texture_diagnostics = (
                    self._run_hr_image_tile_texture_flow(
                        flow_model=tex_flow_model,
                        sampler=self.tex_slat_sampler,
                        global_noise=tex_noise,
                        global_condition_cpu=global_condition_cpu,
                        shape_concat_cond=shape_concat_cond,
                        sampler_params=tex_sampler_params,
                        global_flow=global_tex_flow,
                        tiles=tiles,
                        start_step=int(hr_image_tile_start_step),
                        fallback_mode=str(hr_image_tile_fallback),
                        weight_mode=str(hr_image_tile_weight),
                        condition_extraction=condition_extraction,
                    )
                )
                texture_patch_trace.update(
                    {
                        "config": {
                            "tile_size": int(hr_image_tile_size),
                            "tile_stride": int(hr_image_tile_stride),
                            "start_step": int(hr_image_tile_start_step),
                            "min_foreground_ratio": float(
                                hr_image_tile_min_foreground_ratio
                            ),
                            "fallback": str(hr_image_tile_fallback),
                            "weight": str(hr_image_tile_weight),
                            "save_debug": bool(
                                hr_image_tile_save_debug
                            ),
                        },
                        "global_to_hr_transform": dict(
                            global_to_hr_transform
                        ),
                        "projection": {
                            "complete_image_normalized_xy": (
                                projected_full_norm.detach().cpu()
                            ),
                            "depth": projected_depth.detach().cpu(),
                            "valid": projection_valid.detach().cpu(),
                            "global_image_size": list(image.size),
                            "hr_image_size": list(hr_image.size),
                            "camera_angle_x": float(camera_angle_x),
                            "distance": float(distance),
                            "mesh_scale": float(mesh_scale),
                            "grid_resolution": int(tex_grid_res),
                        },
                        "tile_summary": {
                            "tile_count": int(
                                tile_summary["tile_count"]
                            ),
                            "active_tile_count": int(
                                tile_summary["active_tile_count"]
                            ),
                            "eligible_token_count": int(
                                tile_summary["eligible_token_count"]
                            ),
                            "covered_token_count": int(
                                tile_summary["covered_token_count"]
                            ),
                            "overlap_token_count": int(
                                tile_summary["overlap_token_count"]
                            ),
                        },
                        "debug_artifacts": dict(debug_artifacts),
                    }
                )
                texture_diagnostics.update(
                    {
                        "hr_image_tile_texture_flow": True,
                        "tile_size": int(hr_image_tile_size),
                        "tile_stride": int(hr_image_tile_stride),
                        "min_foreground_ratio": float(
                            hr_image_tile_min_foreground_ratio
                        ),
                        "eligible_token_count": int(
                            tile_summary["eligible_token_count"]
                        ),
                        "initial_covered_token_count": int(
                            tile_summary["covered_token_count"]
                        ),
                        "initial_overlap_token_count": int(
                            tile_summary["overlap_token_count"]
                        ),
                        "global_to_hr_transform": dict(
                            global_to_hr_transform
                        ),
                        "debug_artifacts": dict(debug_artifacts),
                    }
                )
                del (
                    tiles,
                    tile_summary,
                    global_condition_cpu,
                    condition_extraction,
                    projected_full_norm,
                    projected_depth,
                    projection_valid,
                )
            elif texture_patch_enabled:
                tex_slat_normalized, texture_patch_trace, texture_diagnostics = (
                    self._run_2048_patch_flow(
                        flow_model=tex_flow_model,
                        sampler=self.tex_slat_sampler,
                        stage_name="texture",
                        global_noise=tex_noise,
                        cond=cond_tex,
                        concat_cond=shape_concat_cond,
                        sampler_params=tex_sampler_params,
                        global_flow=global_tex_flow,
                        start_step=int(texture_patch_start_step),
                        start_source=str(texture_patch_start_source),
                        patch_guidance_mode=str(texture_guidance_mode),
                        guidance_strength=float(texture_guidance_strength),
                        guidance_interval=str(texture_guidance_interval),
                        guidance_rescale=float(texture_guidance_rescale),
                        wavelet_family=str(texture_wavelet_family),
                        skip_residual_mode=str(
                            texture_skip_residual_mode
                        ),
                    )
                )
            else:
                tex_slat_normalized = global_tex_flow.samples
                texture_patch_trace = pending_texture_trace
                texture_diagnostics = {
                    "patch_coordinate_mode": None,
                    "global_latent_coordinate_mode": "global",
                    "patch_start_step": int(texture_patch_start_step),
                    "patch_start_source": str(texture_patch_start_source),
                    "patch_guidance_mode": "global_original",
                    "guidance_strength": float(texture_guidance_strength),
                    "guidance_interval": str(texture_guidance_interval),
                    "guidance_rescale": float(texture_guidance_rescale),
                    "wavelet_family": str(texture_wavelet_family),
                    "skip_residual_mode": str(
                        texture_skip_residual_mode
                    ),
                    "concat_cond_present": True,
                    "conditional_unconditional_share_concat_cond": True,
                    "velocity_similarities": [],
                }
            tex_trace_size = self._save_2048_flow_trace(
                output_path=texture_flow_trace_path,
                stage_name="texture",
                coords=shape_slat.coords,
                sampler_params=tex_sampler_params,
                global_flow=global_tex_flow,
                patch_trace=texture_patch_trace,
                experiment_tag=flow_experiment_tag,
            )
            texture_diagnostics.update(
                {
                    "trace_path": str(
                        Path(texture_flow_trace_path).resolve()
                    ),
                    "trace_bytes": int(tex_trace_size),
                    "global_state_count": len(
                        global_tex_flow.trajectory.states
                    ),
                    "global_velocity_count": len(
                        global_tex_flow.trajectory.velocities
                    ),
                    "experiment_tag": flow_experiment_tag,
                }
            )
            self.last_texture_flow_diagnostics = texture_diagnostics
            print(
                f"[texture-2048-trace] saved={texture_flow_trace_path} "
                f"bytes={tex_trace_size:,}"
            )
            tex_std = torch.tensor(
                self.tex_slat_normalization['std']
            )[None].to(tex_slat_normalized.device)
            tex_mean = torch.tensor(
                self.tex_slat_normalization['mean']
            )[None].to(tex_slat_normalized.device)
            tex_slat = tex_slat_normalized * tex_std + tex_mean
            if self.low_vram:
                tex_flow_model.cpu()
            del (
                shape_std,
                shape_mean,
                shape_concat_cond,
                tex_noise,
                tex_sampler_params,
                global_tex_flow,
                pending_texture_trace,
                texture_patch_trace,
                texture_diagnostics,
                tex_slat_normalized,
                tex_std,
                tex_mean,
            )
        else:
            tex_slat = self.sample_tex_slat(
                cond_tex, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        del cond_tex
        torch.cuda.empty_cache()

        # ---- Stage 5: Decode ----
        res = actual_hr_resolution
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
