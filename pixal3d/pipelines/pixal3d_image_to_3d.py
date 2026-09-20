from typing import Any, Dict, Callable, Optional, Sequence, Tuple, List
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..representations import Mesh, MeshWithVoxel


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
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
        "tex_slat_flow_model_1024",
        "tex_slat_decoder",
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
        default_pipeline_type: str = "1024_cascade",
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
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
        self._device = "cpu"

    @classmethod
    def from_pretrained(
        cls, path: str, config_file: str = "pipeline.json"
    ) -> "Pixal3DImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        pipeline.shape_slat_sampler = getattr(
            samplers, args["shape_slat_sampler"]["name"]
        )(**args["shape_slat_sampler"]["args"])
        pipeline.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]

        pipeline.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(
            **args["tex_slat_sampler"]["args"]
        )
        pipeline.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]

        pipeline.shape_slat_normalization = args["shape_slat_normalization"]
        pipeline.tex_slat_normalization = args["tex_slat_normalization"]

        # Proj mode: image_cond_models need to be loaded externally, set to None here
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None

        pipeline.rembg_model = getattr(rembg, args["rembg_model"]["name"])(
            **args["rembg_model"]["args"]
        )

        pipeline.low_vram = args.get("low_vram", True)
        pipeline.default_pipeline_type = args.get(
            "default_pipeline_type", "1024_cascade"
        )
        pipeline.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
        pipeline._device = "cpu"

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
            np.asarray(source.getchannel("A")) if source.mode == "RGBA" else None
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
            alpha_proxy = alpha_source.resize(proxy_size, Image.Resampling.LANCZOS)
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
            alpha_source = alpha_proxy.resize(source_size, Image.Resampling.LANCZOS)
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
        composited = square_array[:, :, :3] * square_array[:, :, 3:4] + background * (
            1.0 - square_array[:, :, 3:4]
        )
        source_square = Image.fromarray(
            (np.clip(composited, 0, 1) * 255).astype(np.uint8), mode="RGB"
        )
        image_4096 = source_square.resize((4096, 4096), Image.Resampling.LANCZOS)
        image_1024 = image_4096.resize((1024, 1024), Image.Resampling.LANCZOS)
        image_512 = image_4096.resize((512, 512), Image.Resampling.LANCZOS)
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
                "left": padding[0],
                "right": padding[1],
                "top": padding[2],
                "bottom": padding[3],
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
            image,
            camera_angle_x=cam_angle,
            distance=dist_tensor,
            mesh_scale=scale_tensor,
        )
        if self.low_vram:
            image_cond_model.cpu()
        return {
            "cond": {"global": z_global, "proj": z_proj},
            "neg_cond": {
                "global": torch.zeros_like(z_global),
                "proj": torch.zeros_like(z_proj),
            },
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
            raise ValueError(
                f"coords must have shape [N, 4], got {tuple(coords.shape)}"
            )
        if torch.any(coords[:, 0] != 0):
            raise ValueError("get_proj_cond_shape currently supports batch size 1 only")

        grid_res = int(grid_resolution_override or image_cond_model.grid_resolution)
        print(
            f"[proj-sparse] grid={grid_res} tokens={int(coords.shape[0]):,} "
            f"dense_tokens={grid_res**3:,}"
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
            "cond": {"global": z_global, "proj": z_proj_st},
            "neg_cond": {
                "global": torch.zeros_like(z_global),
                "proj": SparseTensor(
                    feats=torch.zeros_like(z_proj_sparse), coords=coords
                ),
            },
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
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        if noise is None:
            noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(
                self.device
            )
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
        decoder = self.models["sparse_structure_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s) > 0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = (
                torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            )
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

        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
        guide_subs: Optional[List[SparseTensor]] = None,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models["shape_slat_decoder"].set_resolution(resolution)
        if self.low_vram:
            self.models["shape_slat_decoder"].to(self.device)
            self.models["shape_slat_decoder"].low_vram = True
        ret = self.models["shape_slat_decoder"](
            slat, guide_subs=guide_subs, return_subs=True
        )
        if self.low_vram:
            self.models["shape_slat_decoder"].cpu()
            self.models["shape_slat_decoder"].low_vram = False
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
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(
            shape_slat.device
        )
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(
            shape_slat.device
        )
        shape_slat = (shape_slat - mean) / std

        in_channels = (
            flow_model.in_channels
            if isinstance(flow_model, nn.Module)
            else flow_model[0].in_channels
        )
        noise = shape_slat.replace(
            feats=torch.randn(
                shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]
            ).to(self.device)
        )
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

        std = torch.tensor(self.tex_slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization["mean"])[None].to(slat.device)
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
            self.models["tex_slat_decoder"].to(self.device)
        ret = self.models["tex_slat_decoder"](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models["tex_slat_decoder"].cpu()
        return ret

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        torch.cuda.synchronize()
        for m, v in zip(meshes, tex_voxels):
            # m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices,
                    m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=v.coords[:, 1:],
                    attrs=v.feats,
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout,
                )
            )
        return out_mesh
