#!/usr/bin/env python3
"""Visibility-routed material guidance: shared geometry, conditional control, n=1..4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("outputs/sr_0_img_z_tent"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", default="GPU-5a01b63c-14ed-235f-7936-8043e91e88a5")
    p.add_argument(
        "--phase", choices=["batch", "prepare", "variant", "smoke"], default="batch"
    )
    p.add_argument("--n", type=int, choices=range(5), default=0)
    return p.parse_args()


def stage(out, name):
    from sr_tools import common

    common.js(out / "status.json", dict(status="RUNNING", stage=name, pid=os.getpid()))
    print("STAGE", name, flush=True)


def aggregate(out):
    rows = {}
    for n in range(5):
        p = out / f"n{n}/result.json"
        if p.exists():
            rows[str(n)] = json.loads(p.read_text())
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def batch(a):
    from sr_tools import common

    a.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for phase, n in [("smoke", 0), ("prepare", 0)] + [("variant", k) for k in range(5)]:
        name = phase if phase != "variant" else f"n{n}"
        stage(a.output_dir, name)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(a.source.resolve()),
            "--output-dir",
            str(a.output_dir.resolve()),
            "--gpu",
            a.gpu,
            "--phase",
            phase,
            "--n",
            str(n),
        ]
        with (a.output_dir / f"{name}.log").open("a") as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if ret.returncode:
            common.js(
                a.output_dir / f"{name}_failure.json",
                dict(
                    status="FAILED",
                    returncode=ret.returncode,
                    log=str(a.output_dir / f"{name}.log"),
                ),
            )
            failures.append(name)
            if phase in ("smoke", "prepare"):
                break
        aggregate(a.output_dir)
    common.js(
        a.output_dir / "status.json",
        dict(status="FAILED" if failures else "COMPLETE", failures=failures),
    )
    return 1 if failures else 0


def prepare(a):
    from sr_tools import (
        common,
        rendering,
        visibility,
        texture_guidance as guidance,
        shape,
        capacity,
    )
    from types import SimpleNamespace
    from sr import mapping

    out = a.output_dir
    source = a.source.resolve()
    source_signature = json.loads((source / "experiment.json").read_text())
    assert source_signature.get("shape_depth_blend") == "voxel_center_triangle_v1"
    assert (source / "result.json").exists(), "Source run must be complete"
    identity = dict(
        version=1,
        source=str(source),
        source_signature=source_signature,
        c128_hash=common.tensor_hash(
            common.load_payload(source / "sr/shape512/endpoint.pt")["features"]
        ),
        visibility_threshold=0.30,
        visibility_resolution=4096,
        geometry_bridge="mesh2048 -> voxel4096 -> encoder C256",
        texture_weights="uniform",
        shape_weights="triangle",
        n=[0, 1, 2, 3, 4],
    )
    manifest = out / "experiment.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity
    common.js(manifest, identity)
    baseline = source / "baseline"
    if not (out / "baseline").exists():
        (out / "baseline").symlink_to(baseline, target_is_directory=True)
    camera = json.loads((baseline / "camera.json").read_text())
    canonical = {
        k: common.Image.open(baseline / f"{k}.png").copy()
        for k in ["image_512", "image_1024", "image_4096", "foreground_mask_4096"]
    }
    pipe = common.setup(out)
    stage(out, "baseline_first_four_fields")
    guidance.baseline_fields(pipe, baseline, canonical, camera, out / "baseline_fields")
    shared = out / "shared"
    shared.mkdir(exist_ok=True)
    stage(out, "C128_decode_mesh2048")
    c128 = common.load_payload(source / "sr/shape512/endpoint.pt")
    mesh = rendering.decode_geometry(
        pipe, shared / "intermediate2048", c128["coords"], c128["features"], 2048
    )
    stage(out, "voxel4096_encode_and_visibility")
    encoded = visibility.encode_geometry(pipe, mesh, camera, shared / "bridge")
    del mesh, c128
    common.empty_cuda()
    stage(out, "encode_four_texture_guides")
    guidance.encode_guides(
        pipe,
        shared / "bridge/voxels.pt",
        encoded["coords"],
        out / "baseline_fields",
        out / "guides",
    )
    stage(out, "shape1024")
    data = mapping(
        encoded["coords"], 1024, camera, shared / "shape1024", canonical["image_4096"]
    )
    visibility.classify_blocks(data, encoded["visible"], shared)
    common.save(shared / "mapping.pt", **data)
    bank = shape.conditions(pipe, data, 1024, shared / "shape1024", camera)
    batch_size = capacity.calibrate(pipe, data, bank, 1024, shared / "shape1024")
    args = SimpleNamespace(
        batch_size=batch_size,
        max_tokens=batch_size * capacity.statistics(data)["max_block_points"],
    )
    features = shape.flow(pipe, data, bank, 1024, shared / "shape1024", args)
    del bank, data
    common.empty_cuda()
    stage(out, "final_geometry4096")
    rendering.decode_geometry(
        pipe, shared / "final_mesh", encoded["coords"], features, 4096
    )
    common.js(
        out / "prepared.json",
        dict(status="COMPLETE", points=len(encoded["coords"]), camera=camera),
    )


def variant(a):
    from types import SimpleNamespace
    from sr_tools import (
        common,
        texture,
        texture_guidance as guidance,
        rendering,
        capacity,
        metrics,
    )

    out = a.output_dir
    shared = out / "shared"
    dest = out / f"n{a.n}"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").exists():
        return
    assert (out / "prepared.json").exists()
    pipe = common.setup(dest)
    camera = json.loads((out / "prepared.json").read_text())["camera"]
    data = common.load_payload(shared / "mapping.pt")
    payload = common.load_payload(shared / "shape1024/endpoint.pt")
    shape = payload["features"]
    texture.validate_geometry(payload, data)
    conditions_dir = shared / "texture_conditions"
    conditions_dir.mkdir(exist_ok=True)
    bank = texture.conditions(pipe, shared, conditions_dir, data, camera)
    size = capacity.calibrate(
        pipe,
        data,
        bank,
        1024,
        dest,
        model_key="tex_slat_flow_model_1024",
        sampler_params=pipe.tex_slat_sampler_params,
        noise_channels=pipe.models["tex_slat_flow_model_1024"].in_channels
        - shape.shape[1],
        predict_fn=lambda pipe, model, group, x, bank, t, params: texture.predict(
            pipe, model, group, x, shape, bank, t, params
        ),
    )
    args = SimpleNamespace(
        batch_size=size, max_tokens=size * capacity.statistics(data)["max_block_points"]
    )
    stage(dest, "texture_flow")
    feats = guidance.flow(
        pipe, data, shape, bank, out / "guides", dest / "texture", args, a.n
    )
    del data, bank
    common.empty_cuda()
    stage(dest, "texture_decode")
    mesh = rendering.decode_material(pipe, shared, dest / "texture", payload, feats)
    baseline = a.source / "baseline"
    canonical = {
        k: common.Image.open(baseline / f"{k}.png").copy()
        for k in ["image_4096", "foreground_mask_4096"]
    }
    base = common.load_payload(baseline / "textured_mesh.pt")["mesh"]
    stage(dest, "metrics1024")
    report = metrics.evaluate(
        {"baseline1024": base, "sr": mesh}, canonical, camera, dest / "evaluation_1024"
    )
    stage(dest, "views2048")
    rendering.render(pipe, mesh, dest / "texture", resolution=2048, camera=camera)
    common.js(
        dest / "result.json",
        dict(
            status="COMPLETE",
            n=a.n,
            metrics=report["results"],
            protocol="n=0 all-conditional control; n>0 low-visibility blocks guided for n updates then image-unconditional",
        ),
    )
    common.js(dest / "status.json", dict(status="COMPLETE"))


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["PYTHONUNBUFFERED"] = "1"
    import torch

    torch.set_num_threads(8)
    a.output_dir = a.output_dir.resolve()
    a.source = a.source.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.phase == "batch":
        return batch(a)
    with torch.no_grad():
        if a.phase == "prepare":
            prepare(a)
        elif a.phase == "variant":
            variant(a)
        else:
            from sr_tools.guidance_smoke import run

            run(a.output_dir / "smoke", a.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
