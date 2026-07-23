from typing import *

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..representations import Mesh


class Pixal3DImageTo3DPipeline(Pipeline):
    """
    Geometry-only Pixal3D image-to-3D pipeline for resolution-grid testing.

    Differences from upstream Pixal3D main:
      - texture/PBR SLat models are not loaded or sampled;
      - decode_latent returns Mesh objects directly from shape_slat_decoder;
      - supports 512, 1024, 1024_cascade, 1536_cascade;
      - keeps Pixal3D proj/pixel-aligned conditioning and camera-aware condition building.
    """

    model_names_to_load = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ]

    def __init__(
        self,
        models: Optional[dict[str, nn.Module]] = None,
        sparse_structure_sampler: Optional[samplers.Sampler] = None,
        shape_slat_sampler: Optional[samplers.Sampler] = None,
        tex_slat_sampler: Optional[samplers.Sampler] = None,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: Optional[dict] = None,
        tex_slat_sampler_params: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        tex_slat_normalization: Optional[dict] = None,
        image_cond_model_ss: Optional[nn.Module] = None,
        image_cond_model_shape_512: Optional[nn.Module] = None,
        image_cond_model_shape_1024: Optional[nn.Module] = None,
        image_cond_model_tex_1024: Optional[nn.Module] = None,
        rembg_model: Optional[Callable] = None,
        low_vram: bool = True,
        default_pipeline_type: str = "1024_cascade",
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = None
        self.sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        self.shape_slat_sampler_params = shape_slat_sampler_params or {}
        self.tex_slat_sampler_params = {}
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = None
        self.image_cond_model_ss = image_cond_model_ss
        self.image_cond_model_shape_512 = image_cond_model_shape_512
        self.image_cond_model_shape_1024 = image_cond_model_shape_1024
        self.image_cond_model_tex_1024 = None
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
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pixal3DImageTo3DPipeline":
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(
            **args["sparse_structure_sampler"]["args"]
        )
        pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]

        pipeline.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(
            **args["shape_slat_sampler"]["args"]
        )
        pipeline.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]

        # Geometry-only: do not construct texture sampler/normalization and do not load texture models.
        pipeline.tex_slat_sampler = None
        pipeline.tex_slat_sampler_params = {}
        pipeline.shape_slat_normalization = args["shape_slat_normalization"]
        pipeline.tex_slat_normalization = None

        # Proj mode image condition models are built externally by the batch script / app.
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None

        pipeline.rembg_model = getattr(rembg, args["rembg_model"]["name"])(**args["rembg_model"]["args"])
        pipeline.low_vram = args.get("low_vram", True)
        pipeline.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")
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

    def preprocess_image(self, input: Image.Image, bg_color: tuple = (0, 0, 0)) -> Image.Image:
        """Remove background, crop to object, and composite on a solid background."""
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize(
                (int(input.width * scale), int(input.height * scale)),
                Image.Resampling.LANCZOS,
            )

        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox_pixels = np.argwhere(alpha > 0.8 * 255)
        if bbox_pixels.size == 0:
            return input.convert("RGB")

        bbox = (
            np.min(bbox_pixels[:, 1]),
            np.min(bbox_pixels[:, 0]),
            np.max(bbox_pixels[:, 1]),
            np.max(bbox_pixels[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.1)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(bbox)  # type: ignore[arg-type]
        output = np.array(output).astype(np.float32) / 255.0
        rgb = output[:, :, :3]
        a = output[:, :, 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        output = rgb * a + bg * (1.0 - a)
        output = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
        return output

    # ---------------------------------------------------------------------
    # Pixal3D projection-conditioned feature extraction
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def get_proj_cond_ss(
        self,
        image: list[Image.Image],
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
    ) -> dict:
        device = self.device
        image_cond_model = self.image_cond_model_ss
        if image_cond_model is None:
            raise RuntimeError("image_cond_model_ss is not set. Build it before running Pixal3D.")

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
            "neg_cond": {"global": torch.zeros_like(z_global), "proj": torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list[Image.Image],
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: Optional[int] = None,
    ) -> dict:
        device = self.device
        if image_cond_model is None:
            raise RuntimeError("Pixal3D shape image condition model is not set.")

        if self.low_vram:
            image_cond_model.to(device)

        orig_grid_res = image_cond_model.grid_resolution
        replaced_grid = False
        if grid_resolution_override is not None and int(grid_resolution_override) != int(orig_grid_res):
            image_cond_model.grid_resolution = int(grid_resolution_override)
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=int(grid_resolution_override),
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)
            replaced_grid = True

        try:
            bsz = 1
            cam_angle = torch.tensor([camera_angle_x], device=device)
            dist_tensor = torch.tensor([distance], device=device)
            scale_tensor = torch.tensor([mesh_scale], device=device)
            z_global, z_proj = image_cond_model(
                image,
                camera_angle_x=cam_angle,
                distance=dist_tensor,
                mesh_scale=scale_tensor,
            )

            grid_res = int(image_cond_model.grid_resolution)
            z_proj_grid = z_proj.reshape(bsz, grid_res, grid_res, grid_res, -1)
            batch_indices = coords[:, 0].long()
            x_coords = coords[:, 1].long()
            y_coords = coords[:, 2].long()
            z_coords = coords[:, 3].long()
            z_proj_sparse = z_proj_grid[batch_indices, x_coords, y_coords, z_coords]
            z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

            return {
                "cond": {"global": z_global, "proj": z_proj_st},
                "neg_cond": {
                    "global": torch.zeros_like(z_global),
                    "proj": SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords),
                },
            }
        finally:
            if replaced_grid:
                image_cond_model.grid_resolution = orig_grid_res
                image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                    grid_resolution=orig_grid_res,
                    image_resolution=image_cond_model.proj_grid.image_resolution,
                ).to(device)
            if self.low_vram:
                image_cond_model.cpu()

    # ---------------------------------------------------------------------
    # Sampling and decoding
    # ---------------------------------------------------------------------
    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
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

        decoder = self.models["sparse_structure_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s) > 0
        if self.low_vram:
            decoder.cpu()

        if int(resolution) != int(decoded.shape[2]):
            if int(resolution) > int(decoded.shape[2]):
                raise ValueError(
                    f"requested sparse resolution={resolution} exceeds decoder output "
                    f"resolution={decoded.shape[2]}"
                )
            ratio = int(decoded.shape[2]) // int(resolution)
            if ratio < 1:
                raise ValueError(
                    f"invalid sparse downsample ratio={ratio}; decoded={decoded.shape[2]}, requested={resolution}"
                )
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        tqdm_desc: str = "Sampling shape SLat (proj)",
    ) -> SparseTensor:
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
            tqdm_desc=tqdm_desc,
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
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        decoder = self.models["shape_slat_decoder"]
        decoder.set_resolution(int(resolution))
        if self.low_vram:
            decoder.to(self.device)
            decoder.low_vram = True
        try:
            return decoder(slat, return_subs=True)
        finally:
            if self.low_vram:
                try:
                    decoder.cpu()
                finally:
                    decoder.low_vram = False

    @torch.no_grad()
    def decode_latent(self, shape_slat: SparseTensor, resolution: int) -> List[Mesh]:
        meshes, _ = self.decode_shape_slat(shape_slat, int(resolution))
        return meshes

    @staticmethod
    def _coords_debug(coords: torch.Tensor) -> tuple[Optional[int], Optional[int]]:
        if coords.numel() == 0:
            return None, None
        return int(coords[:, 1:].min().item()), int(coords[:, 1:].max().item())

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
    ) -> Union[List[Mesh], Tuple[List[Mesh], Tuple[SparseTensor, None, int]]]:
        """
        Run Pixal3D in geometry-only proj mode.

        Supported pipeline_type:
          - 512: direct shape_slat_flow_model_512 on sparse grid 32, decode at 512
          - 1024: direct shape_slat_flow_model_1024 on sparse grid 64, decode at 1024
          - 1024_cascade: 512 LR shape, decoder.upsample support, 1024 HR shape
          - 1536_cascade: 512 LR shape, decoder.upsample support, 1536 HR shape with token cap fallback
        """
        del tex_slat_sampler_params  # texture is intentionally disabled in this file

        pipeline_type = pipeline_type or self.default_pipeline_type
        valid_types = {"512", "1024", "1024_cascade", "1536_cascade"}
        if pipeline_type not in valid_types:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}. Valid types: {sorted(valid_types)}")

        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set."
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set."

        camera_angle_x = float(camera_params["camera_angle_x"])
        distance = float(camera_params["distance"])
        mesh_scale = float(camera_params.get("mesh_scale", 1.0))

        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(int(seed))

        # Sparse structure. This follows TRELLIS-style direct 1024 support for the plain 1024 mode;
        # cascades deliberately start from 32^3 sparse support and refine through 512 -> HR.
        ss_res_map = {
            "512": 32,
            "1024": 64,
            "1024_cascade": 32,
            "1536_cascade": 32,
        }
        ss_res = ss_res_map[pipeline_type]
        cond_ss = self.get_proj_cond_ss(
            [image],
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        coords = self.sample_sparse_structure(cond_ss, ss_res, num_samples, sparse_structure_sampler_params)
        del cond_ss
        torch.cuda.empty_cache()

        coord_min, coord_max = self._coords_debug(coords)
        print(
            f"[DEBUG][ss/proj] pipeline_type={pipeline_type}, ss_res={ss_res}, "
            f"num_coords={coords.shape[0]}, coord_min={coord_min}, coord_max={coord_max}"
        )

        if pipeline_type == "512":
            cond_shape = self.get_proj_cond_shape(
                self.image_cond_model_shape_512,
                [image],
                coords,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape,
                self.models["shape_slat_flow_model_512"],
                coords,
                shape_slat_sampler_params,
                tqdm_desc="Sampling shape SLat (proj, 512)",
            )
            del cond_shape
            res = 512

        elif pipeline_type == "1024":
            cond_shape = self.get_proj_cond_shape(
                self.image_cond_model_shape_1024,
                [image],
                coords,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                grid_resolution_override=64,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape,
                self.models["shape_slat_flow_model_1024"],
                coords,
                shape_slat_sampler_params,
                tqdm_desc="Sampling shape SLat (proj, 1024)",
            )
            del cond_shape
            res = 1024

        else:
            target_res = int(pipeline_type.split("_")[0])

            # LR 512 shape stage.
            cond_shape_lr = self.get_proj_cond_shape(
                self.image_cond_model_shape_512,
                [image],
                coords,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
            )
            lr_slat = self.sample_shape_slat(
                cond_shape_lr,
                self.models["shape_slat_flow_model_512"],
                coords,
                shape_slat_sampler_params,
                tqdm_desc="Sampling LR shape SLat (proj, 512)",
            )
            del cond_shape_lr
            torch.cuda.empty_cache()

            # Use the trained decoder upsampler to propose HR support.
            decoder = self.models["shape_slat_decoder"]
            if self.low_vram:
                decoder.to(self.device)
                decoder.low_vram = True
            hr_coords = decoder.upsample(lr_slat, upsample_times=4)
            if self.low_vram:
                decoder.cpu()
                decoder.low_vram = False

            lr_resolution = 512
            actual_hr_resolution = target_res
            while True:
                grid_res = actual_hr_resolution // 16
                quant_coords = torch.cat(
                    [
                        hr_coords[:, :1],
                        ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
                    ],
                    dim=1,
                )
                valid = (quant_coords[:, 1:] >= 0).all(dim=1) & (quant_coords[:, 1:] < grid_res).all(dim=1)
                hr_coords_unique = quant_coords[valid].unique(dim=0)
                num_tokens = int(hr_coords_unique.shape[0])
                print(
                    f"[DEBUG][cascade/proj] requested_res={target_res}, try_res={actual_hr_resolution}, "
                    f"latent_grid={grid_res}, num_tokens={num_tokens}, max_num_tokens={max_num_tokens}"
                )
                if max_num_tokens is None or int(max_num_tokens) <= 0:
                    break
                if num_tokens < int(max_num_tokens) or actual_hr_resolution == 1024:
                    if actual_hr_resolution != target_res:
                        print(f"Due to the limited number of tokens, the resolution is reduced to {actual_hr_resolution}.")
                    break
                actual_hr_resolution -= 128

            actual_grid_res = actual_hr_resolution // 16
            del lr_slat, hr_coords, quant_coords
            torch.cuda.empty_cache()

            # HR shape stage with grid override for 1536 -> 96^3 projected features.
            cond_shape_hr = self.get_proj_cond_shape(
                self.image_cond_model_shape_1024,
                [image],
                hr_coords_unique,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                grid_resolution_override=actual_grid_res,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape_hr,
                self.models["shape_slat_flow_model_1024"],
                hr_coords_unique,
                shape_slat_sampler_params,
                tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
            )
            del cond_shape_hr, hr_coords_unique
            res = actual_hr_resolution

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, None, int(res))
        return out_mesh
