import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native baseline1024: SS512 -> Shape512 -> Shape1024 -> Texture1024"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="GPU-5a01b63c-14ed-235f-7936-8043e91e88a5")
    args = parser.parse_args()
    return args


# Bind the selected physical device before importing any CUDA-dependent package.
if __name__ == "__main__":
    cli_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu

import math
import torch
import numpy as np
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")

from pixal3d.pipelines import Pixal3DImageTo3DPipeline

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"
MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================


def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel

    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False):
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_512"]
    )
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_1024"]
    )
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["tex_1024"]
    )

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in [
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ]:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, "use_naf_upsample", False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in [
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ]:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, "use_naf_upsample", False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline


# ============================================================================
# Camera Estimation
# ============================================================================


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(
    camera_angle_x, grid_point, target_point, mesh_scale, image_resolution
):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw = gp[0].item(), gp[1].item()
    xt = float(target_point[0].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(
    image_path,
    moge_model,
    device="cuda",
    mesh_scale=1.0,
    extend_pixel=0,
    image_resolution=512,
):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x,
        grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )["distance_from_x"]
    return {
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": mesh_scale,
    }


def main(args):
    """Run the same native 1024 baseline used by sr.py, without any SR stages."""
    from sr_tools import common, rendering
    from sr_tools.baseline import run_baseline1024
    from sr_tools.metrics import evaluate

    out = args.output_dir.resolve()
    with torch.no_grad():
        pipe = common.setup(out)
        canonical, camera, mesh = run_baseline1024(pipe, args.image, out)
        rendering.render(pipe, mesh, out, resolution=2048, camera=camera)
        evaluate({"baseline1024": mesh}, canonical, camera, out / "evaluation_1024")
    print("COMPLETE", out)


if __name__ == "__main__":
    main(cli_args)
