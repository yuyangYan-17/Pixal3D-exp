#!/usr/bin/env python3
"""Single-image shape/material SR, baseline comparison, metrics and visualization."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path


def status(out, stage):
    from sr_tools import common

    common.js(out / "status.json", dict(status="RUNNING", stage=stage, pid=os.getpid()))
    print("STAGE", stage, flush=True)


def mapping(coords, stage, cam, out, image):
    from sr_tools import shape as s, common, capacity

    data = s.construct(
        coords,
        stage,
        cam,
        out,
        image,
        padding=32,
        image_stride=1024,
        depth_stride=16 if stage == 512 else 32,
    )
    assert len(data["tiles"]) == (9 if stage == 512 else 16)
    assert len(data["blocks"]) == 7 * len(data["tiles"])
    assert (data["counts"] > 0).all() and (data["counts"] <= 2).all()
    common.js(out / "block_statistics.json", capacity.statistics(data))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--geometry-only",
        action="store_true",
        help="run baseline and both geometry stages, then stop before texture flow",
    )
    ap.add_argument(
        "--gpu",
        default="GPU-5a01b63c-14ed-235f-7936-8043e91e88a5",
        help="Physical GPU index or UUID; defaults to physical GPU4",
    )
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    return run(a)


def run(a):
    import torch

    torch.set_num_threads(8)
    try:
        with torch.no_grad():
            return run_pipeline(a)
    except Exception as exc:
        from sr_tools.common import js

        js(a.output_dir / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise


def run_pipeline(a):
    import torch
    from sr_tools import (
        shape as s,
        texture as tex,
        capacity,
        rendering as material,
        common,
    )
    from sr_tools.metrics import evaluate
    from sr_tools.baseline import run_baseline1024, encode_baseline

    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(
        version=3,
        shape_depth_blend="voxel_center_triangle_v1",
        texture_depth_blend="uniform",
        decoder_output="chunked_layernorm_v1",
        image=str(a.image.resolve()),
        sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
        image_contexts=[2048, 1024],
        image_strides=[1024, 1024],
        image_padding=32,
        depth_contexts=[32, 64],
        depth_strides=[16, 32],
        steps=12,
        seeds=[42, 43, 44, 46],
        metrics_resolution=1024,
        visualization_resolution=2048,
        foreground_metrics=True,
        geometry_only=bool(a.geometry_only),
    )
    if (out / "experiment.json").exists():
        assert json.loads((out / "experiment.json").read_text()) == signature
    common.js(out / "experiment.json", signature)
    target = out / ("smoke" if a.smoke else "sr")
    target.mkdir(exist_ok=True)
    status(out, "setup")
    pipe = common.setup(out)
    status(out, "baseline1024_and_encode_C128")
    canonical, cam, base = run_baseline1024(pipe, a.image, out / "baseline")
    c128, enc = encode_baseline(pipe, canonical, cam, base, out / "baseline")
    common.js(target / "camera.json", cam)
    status(out, "shape512_conditions")
    data = mapping(c128, 512, cam, target / "shape512", canonical["image_4096"])
    bank = s.conditions(pipe, data, 512, target / "shape512", cam)
    a.batch_size = capacity.calibrate(pipe, data, bank, 512, target / "shape512")
    a.max_tokens = a.batch_size * capacity.statistics(data)["max_block_points"]
    status(out, "shape512_flow")
    x = s.flow(
        pipe, data, bank, 512, target / "shape512", a, initial=enc, smoke=a.smoke
    )
    status(out, "global_upsample")
    # Smoke uses a valid encoded shape for the full-support decoder interface.
    c256 = s.upsample(
        pipe,
        c128,
        enc if a.smoke else x,
        target / "upsample",
        input_description="encoded C128 for smoke; synchronized C128 for full run",
    )
    del data, bank, x
    gc.collect()
    common.empty_cuda()
    status(out, "shape1024_conditions")
    data = mapping(c256, 1024, cam, target / "shape1024", canonical["image_4096"])
    bank = s.conditions(pipe, data, 1024, target / "shape1024", cam)
    a.batch_size = capacity.calibrate(pipe, data, bank, 1024, target / "shape1024")
    a.max_tokens = a.batch_size * capacity.statistics(data)["max_block_points"]
    status(out, "shape1024_flow")
    shape = s.flow(pipe, data, bank, 1024, target / "shape1024", a, smoke=a.smoke)
    del bank
    gc.collect()
    common.empty_cuda()
    payload = dict(coords=c256, features=shape, normalized=True)
    if a.geometry_only and not a.smoke:
        status(out, "geometry_only_decode")
        material.decode_geometry(
            pipe, target / "final_mesh", payload["coords"], payload["features"], 4096
        )
        common.js(
            out / "geometry_only.json",
            dict(
                status="COMPLETE",
                C128=len(c128),
                C256=len(c256),
                geometry_mesh=str(target / "final_mesh/geometry_mesh.pt"),
            ),
        )
        common.js(out / "status.json", dict(status="GEOMETRY_ONLY_COMPLETE", pid=os.getpid()))
        return
    tex.validate_geometry(payload, data)
    material_out = target / "texture"
    material_out.mkdir(exist_ok=True)
    status(out, "texture_conditions")
    bank = tex.conditions(pipe, target, material_out, data, cam)
    channels = pipe.models["tex_slat_flow_model_1024"].in_channels - shape.shape[1]
    a.batch_size = capacity.calibrate(
        pipe,
        data,
        bank,
        1024,
        material_out,
        model_key="tex_slat_flow_model_1024",
        sampler_params=pipe.tex_slat_sampler_params,
        noise_channels=channels,
        predict_fn=lambda pipe, model, group, x, bank, t, params: tex.predict(
            pipe, model, group, x, shape, bank, t, params
        ),
    )
    a.max_tokens = a.batch_size * capacity.statistics(data)["max_block_points"]
    status(out, "texture_flow")
    texture = tex.flow(pipe, data, shape, bank, material_out, a)
    del bank
    gc.collect()
    common.empty_cuda()
    if a.smoke:
        candidates = [b for b in data["blocks"] if 64 <= len(b["rows"]) <= 1024]
        if candidates:
            ids = max(candidates, key=lambda b: len(b["rows"]))["global_ids"]
        else:
            ids = min(
                (b for b in data["blocks"] if len(b["rows"])),
                key=lambda b: len(b["rows"]),
            )["global_ids"][:1024]
        payload = dict(coords=c256[ids], features=shape[ids], normalized=True)
        texture = texture[ids]
    status(
        out, "smoke_bounded_geometry_decode" if a.smoke else "global_geometry_decode"
    )
    geometry = material.decode_geometry(
        pipe, target / "final_mesh", payload["coords"], payload["features"], 4096
    )
    del geometry, data
    common.empty_cuda()
    status(
        out, "smoke_bounded_material_decode" if a.smoke else "global_material_decode"
    )
    mesh = material.decode_material(pipe, target, material_out, payload, texture)
    assert len(mesh.vertices) > 0 and torch.isfinite(mesh.attrs).all()
    del texture
    common.empty_cuda()
    status(out, "metrics1024")
    report = evaluate(
        {"baseline1024": base, "sr": mesh}, canonical, cam, target / "evaluation_1024"
    )
    status(out, "baseline_multiview2048")
    material.render(pipe, base, out / "baseline", resolution=2048, camera=cam)
    status(out, "sr_multiview2048")
    material.render(pipe, mesh, material_out, resolution=2048, camera=cam)
    ca = json.loads((out / "baseline/views_2048/cameras.json").read_text())
    cb = json.loads((material_out / "views_2048/cameras.json").read_text())
    assert ca["views"] == cb["views"]
    result = dict(
        status="PASS" if a.smoke else "COMPLETE",
        image=str(a.image),
        C128=len(c128),
        C256=len(c256),
        metrics=report["results"],
        metrics_resolution=1024,
        views_resolution=2048,
        camera_alignment_exact=True,
        smoke_scope="one global step each shape stage and texture, complete support; bounded material decode; rendered metrics are smoke only"
        if a.smoke
        else None,
    )
    common.js(out / ("smoke.json" if a.smoke else "result.json"), result)
    common.js(
        out / "status.json",
        dict(status="SMOKE_PASS" if a.smoke else "COMPLETE", pid=os.getpid()),
    )
    print("SMOKE PASS" if a.smoke else "COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
