"""Shared canonical-foreground metrics on native input-camera 1024 renders."""

import math
import gc
from . import common
import torch as T
import numpy as np
from PIL import Image


@T.no_grad()
def evaluate(meshes, canonical, cam, out):
    from pixal3d.renderers import PbrMeshRenderer
    from pixal3d.utils.render_utils import proj_camera_to_render_params
    from .rendering import UnlitEnvironment
    from skimage.metrics import structural_similarity
    import lpips

    out.mkdir(parents=True, exist_ok=True)
    reference = canonical["image_4096"].resize((1024, 1024), Image.Resampling.LANCZOS)
    mask = (
        np.asarray(
            canonical["foreground_mask_4096"].resize(
                (1024, 1024), Image.Resampling.LANCZOS
            )
        )
        > 127
    )
    assert mask.any()
    reference.save(out / "reference.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / "foreground.png")
    ext, intr = proj_camera_to_render_params(cam["camera_angle_x"], cam["distance"])
    for name, mesh in meshes.items():
        renderer = PbrMeshRenderer(
            dict(
                resolution=1024,
                near=0.01,
                far=100,
                ssaa=1,
                peel_layers=1,
                face_chunk_size=2_000_000,
            ),
            device="cuda:0",
        )
        gpu = mesh.cuda()
        rendered = renderer.render(gpu, ext, intr, envmap=UnlitEnvironment())
        pixels = (
            (rendered["base_color"].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        Image.fromarray(pixels).save(out / f"{name}.png")
        del gpu, rendered, renderer
        gc.collect()
        common.empty_cuda()
    model = (
        lpips.LPIPS(
            net="alex", version="0.1", pretrained=True, pnet_rand=False, spatial=True
        )
        .eval()
        .cuda()
    )

    def tensor(im):
        return (
            T.from_numpy(np.asarray(im, dtype=np.float32).copy())
            .permute(2, 0, 1)[None]
            .cuda()
            / 127.5
            - 1
        )

    ref = np.asarray(reference, dtype=np.float64) / 255
    ref_tensor = tensor(reference)
    results = {}
    sheet = Image.new("RGB", (1024 * (len(meshes) + 1), 1024))
    sheet.paste(reference, (0, 0))
    for i, name in enumerate(meshes, 1):
        im = Image.open(out / f"{name}.png").convert("RGB")
        pred = np.asarray(im, dtype=np.float64) / 255
        mse = float(((ref[mask] - pred[mask]) ** 2).mean())
        _, ssim_map = structural_similarity(
            ref,
            pred,
            data_range=1.0,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            win_size=11,
            use_sample_covariance=False,
            full=True,
        )
        distance = model(ref_tensor, tensor(im))[0, 0].cpu().numpy()
        results[name] = dict(
            psnr_db=-10 * math.log10(max(mse, 1e-30)),
            ssim=float(ssim_map[mask].mean()),
            lpips=float(distance[mask].mean()),
        )
        assert all(math.isfinite(v) for v in results[name].values())
        sheet.paste(im, (i * 1024, 0))
    sheet.save(out / "input_baseline_sr.png")
    report = dict(
        resolution=1024,
        foreground_pixels=int(mask.sum()),
        results=results,
        extrinsics=ext.cpu().tolist(),
        intrinsics=intr.cpu().tolist(),
        protocol="native unlit base_color; shared canonical input alpha>127; no registration; RGB foreground MSE; Gaussian11 sigma1.5 SSIM foreground mean; AlexNet LPIPS v0.1 spatial foreground mean",
    )
    common.js(out / "metrics.json", report)
    del model, ref_tensor
    gc.collect()
    common.empty_cuda()
    return report
