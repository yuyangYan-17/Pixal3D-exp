"""Native SS512 -> Shape512 -> Shape1024 -> Texture1024 comparison baseline."""

import gc
import hashlib
import json
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch as T
from . import common

ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/microsoft/TRELLIS___2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
)


@T.no_grad()
def run_baseline1024(pipe, image, out):
    out.mkdir(parents=True, exist_ok=True)
    signature = {
        "image_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest(),
        "version": 2,
    }
    manifest = out / "input.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError(
            "Baseline output belongs to another input or code version; use a new directory"
        )
    if not manifest.exists() and (out / "textured_mesh.pt").exists():
        raise ValueError("Unversioned baseline cache; use a new directory")
    common.js(manifest, signature)
    if (out / "foreground_mask_4096.png").exists():
        canonical = {
            k: common.Image.open(out / f"{k}.png").copy()
            for k in ("image_512", "image_1024", "image_4096", "foreground_mask_4096")
        }
    else:
        canonical = pipe.preprocess_canonical_images(common.Image.open(image))
        common.save_canonical_images(canonical, out)
        canonical["foreground_mask_4096"].save(out / "foreground_mask_4096.png")
        common.js(out / "preprocess.json", canonical["metadata"])
    if (out / "camera.json").exists():
        cam = json.loads((out / "camera.json").read_text())
    else:
        from inference import load_moge_model, get_camera_params_wild_moge

        moge = load_moge_model(device="cuda:0")
        cam = get_camera_params_wild_moge(
            str(out / "image_512.png"), moge, device="cuda:0", image_resolution=512
        )
        del moge
        gc.collect()
        common.empty_cuda()
        common.js(out / "camera.json", cam)
    state = common.RunState(pipe, canonical, cam, pipe.device, out, time.perf_counter())
    args = SimpleNamespace(
        resume=True, ss_steps=12, shape_steps=12, encoder_path=ENCODER
    )
    if not (out / "textured_mesh.pt").exists():
        T.manual_seed(42)
        np.random.seed(42)
        coords = common.run_native_prefix(state, args)
        shape = common.run_native_shape1024(state, args, coords)
        if (out / "texture_latent.pt").exists():
            payload = common.load_payload(out / "texture_latent.pt")
            assert T.equal(payload["coords"], shape.coords.cpu())
            texture = common.SparseTensor(
                payload["features"].to(pipe.device), payload["coords"].to(pipe.device)
            )
        else:
            cond = pipe.get_proj_cond_shape(
                pipe.image_cond_model_tex_1024,
                [canonical["image_1024"]],
                shape.coords,
                camera_angle_x=cam["camera_angle_x"],
                distance=cam["distance"],
                mesh_scale=1.0,
                grid_resolution_override=64,
            )
            T.manual_seed(46)
            texture = pipe.sample_tex_slat(
                cond, pipe.models["tex_slat_flow_model_1024"], shape, {"steps": 12}
            )
            common.save(
                out / "texture_latent.pt",
                coords=texture.coords.cpu(),
                features=texture.feats.cpu(),
                normalized=False,
            )
            del cond
        mesh = pipe.decode_latent(shape, texture, 1024)[0].cpu()
        common.save(out / "textured_mesh.pt", mesh=mesh)
        del shape, texture
        common.empty_cuda()
    else:
        mesh = common.load_payload(out / "textured_mesh.pt")["mesh"]
    common.js(
        out / "result.json",
        dict(
            status="COMPLETE",
            resolution=1024,
            steps=12,
            seeds=[42, 46],
            pipeline="SS512 -> Shape512 -> Shape1024 -> Texture1024 -> mesh1024",
        ),
    )
    return canonical, cam, mesh


@T.no_grad()
def encode_baseline(pipe, canonical, camera, mesh, out):
    """Voxelize the baseline at 2048 and encode its normalized global C128 support."""
    state = common.RunState(
        pipe, canonical, camera, pipe.device, out, time.perf_counter()
    )
    args = SimpleNamespace(resume=True, encoder_path=ENCODER)
    return common.encode_mesh_latent(
        state, args, mesh, 2048, out / "baseline/encoded_s128.pt"
    )
