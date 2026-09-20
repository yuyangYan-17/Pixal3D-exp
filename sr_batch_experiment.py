#!/usr/bin/env python3
"""Batch baseline/geometry/texture experiment with fixed geometry routing.

The geometry stage is executed once per image.  The texture stage reuses its
geometry and evaluates several random seeds using the visibility-partitioned
flow in :mod:`sr_tools.texture_explore`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


GPU4 = "GPU-5a01b63c-14ed-235f-7936-8043e91e88a5"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--assets",
        type=Path,
        default=Path("/home/nvme04/yyyan/Pixal3D/assets/images"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sr_texture_visibility_final_20260919"),
    )
    p.add_argument(
        "--phase",
        choices=("smoke", "prepare", "batch", "summary"),
        default="batch",
        help="prepare stops after baseline/geometry/visibility caches; it does not run texture flow",
    )
    p.add_argument("--seeds", default="46,47,48")
    p.add_argument("--guide-steps", type=int, default=4)
    p.add_argument("--guide-threshold", type=float, default=1.0)
    p.add_argument(
        "--hidden-mode",
        choices=("conditional_hidden", "global", "unconditional"),
        default="unconditional",
    )
    p.add_argument("--gpu", default=GPU4)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def images(args):
    values = sorted(
        p for p in args.assets.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if args.limit:
        values = values[: args.limit]
    if not values:
        raise RuntimeError(f"no images under {args.assets}")
    return values


def run_subprocess(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        ret = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - started
    if ret.returncode:
        raise RuntimeError(f"command failed ({ret.returncode}); see {log_path}")
    return elapsed


def image_record(root, image):
    return root / image.stem


def baseline(args, image, out):
    result = out / "result.json"
    timing_path = out / "baseline/timing.json"
    if result.exists() and timing_path.exists():
        return json.loads(timing_path.read_text())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PIXAL3D_LOW_MEMORY_DECODER"] = "1"
    elapsed = run_subprocess(
        [
            sys.executable,
            str(Path(__file__).with_name("inference.py")),
            "--image",
            str(image.resolve()),
            "--output-dir",
            str(out.resolve()),
            "--gpu",
            args.gpu,
        ],
        out / "baseline.log",
        env,
    )
    payload = dict(baseline_seconds=elapsed, pipeline="native baseline1024")
    (out / "baseline/timing.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def geometry(args, image, out):
    marker = out / "geometry_only.json"
    timing_path = out / "geometry_timing.json"
    if marker.exists() and timing_path.exists():
        return json.loads(timing_path.read_text())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PIXAL3D_LOW_MEMORY_DECODER"] = "1"
    elapsed = run_subprocess(
        [
            sys.executable,
            str(Path(__file__).with_name("sr.py")),
            "--image",
            str(image.resolve()),
            "--output-dir",
            str(out.resolve()),
            "--geometry-only",
            "--gpu",
            args.gpu,
        ],
        out / "geometry.log",
        env,
    )
    payload = dict(geometry_seconds=elapsed, pipeline="shape512 -> shape1024")
    (out / "geometry_timing.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _symlink(target, link):
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise RuntimeError(f"refusing to replace existing path {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def _remove_flow_checkpoints(directory):
    """Keep final endpoints and JSON manifests, remove resumable tensor steps."""
    directory = Path(directory)
    for path in (directory / "initial.pt", *directory.glob("step_*.pt")):
        if path.exists():
            path.unlink()


def _trim_geometry_cache(out):
    """Drop geometry inputs that texture preparation has already consumed."""
    for stage in ("shape512", "shape1024"):
        stage_dir = out / "sr" / stage
        _remove_flow_checkpoints(stage_dir)
        tiles = stage_dir / "tiles"
        if tiles.exists():
            shutil.rmtree(tiles)
    bridge = out / "texture_explore" / "shared" / "bridge"
    # Flow only needs encoded C256 visibility and voxel->C256 ancestry after
    # guide encoding. The raw 4096 dual grid and visible-face voxelization are
    # reproducible preparation artifacts and dominate disk usage.
    for name in ("voxels.pt", "visible_voxels.pt", "visible_faces.pt"):
        path = bridge / name
        if path.exists():
            path.unlink()
    intermediate = out / "texture_explore" / "shared" / "intermediate2048"
    if intermediate.exists() and not intermediate.is_symlink():
        shutil.rmtree(intermediate)


def _link_selected_texture(out, seed):
    """Expose the first completed seed through the conventional SR path."""
    target = out / "texture_explore" / f"seed_{seed}" / "texture"
    link = out / "sr" / "texture"
    if not target.exists() or link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(Path(os.path.relpath(target, link.parent)), target_is_directory=True)


def prepare(args, image, out):
    """Build visibility, endpoint and texture-condition caches once."""
    from sr_tools import common, rendering, texture, texture_guidance, visibility
    from sr import mapping

    root = out / "texture_explore"
    marker = root / "prepared.json"
    if marker.exists():
        _trim_geometry_cache(out)
        return json.loads(marker.read_text())
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    _symlink(out / "baseline", root / "baseline")
    shared = root / "shared"
    shared.mkdir(exist_ok=True)
    _symlink(out / "sr/final_mesh", shared / "final_mesh")
    pipe = common.setup(root)
    baseline_root = out / "baseline"
    canonical = {
        k: common.Image.open(baseline_root / f"{k}.png").copy()
        for k in ("image_4096", "image_1024", "image_512", "foreground_mask_4096")
    }
    camera = json.loads((baseline_root / "camera.json").read_text())
    c128 = common.load_payload(out / "sr/shape512/endpoint.pt")
    rendering.decode_geometry(
        pipe,
        shared / "intermediate2048",
        c128["coords"],
        c128["features"],
        2048,
    )
    mesh = common.load_payload(shared / "intermediate2048/geometry_mesh.pt")["mesh"]
    shape = common.load_payload(out / "sr/shape1024/endpoint.pt")
    # The fixed shape endpoint already contains the C256 support/features.
    # Re-encoding a dense 2048 mesh through the dual-grid VAE can exceed 80 GiB
    # on large assets; visibility only needs voxel ancestry and can be derived
    # from the fixed support without changing the mesh or flow inputs.
    encoded = visibility.derive_visibility_from_shape_support(
        mesh, camera, shared / "bridge", shape
    )
    if not torch.equal(encoded["coords"], shape["coords"]):
        # The shape flow may retain a few support points that the bridge
        # encoder drops after voxelization.  Map the bridge visibility labels
        # to the fixed shape support by nearest C256 lattice coordinate; no
        # shape feature is changed and no geometry is regenerated.
        from scipy.spatial import cKDTree

        source_xyz = encoded["coords"][:, 1:].numpy()
        target_xyz = shape["coords"][:, 1:].numpy()
        source_to_target = cKDTree(target_xyz).query(source_xyz, k=1)[1]
        target_to_source = cKDTree(source_xyz).query(target_xyz, k=1)[1]
        old_ancestry = common.load_payload(shared / "bridge/ancestry.pt")
        mapped_parent = torch.as_tensor(
            source_to_target[old_ancestry["voxel_to_c256"].numpy()], dtype=torch.long
        )
        visible = encoded["visible"][torch.as_tensor(target_to_source).long()]
        common.save(
            shared / "bridge/ancestry.pt",
            voxel_visible=old_ancestry["voxel_visible"],
            voxel_to_c256=mapped_parent,
            visible_faces_file=old_ancestry["visible_faces_file"],
            rule="nearest C256 lattice remap from bridge encoder to fixed shape support",
        )
        encoded = dict(
            coords=shape["coords"],
            features=shape["features"],
            normalized=True,
            visible=visible,
        )
        common.save(shared / "bridge/encoded.pt", **encoded)
        common.js(
            shared / "bridge/visibility.json",
            dict(
                total_faces=len(mesh.faces),
                bridge_latent_points=len(source_xyz),
                latent_points=len(target_xyz),
                visible_latent_points=int(visible.sum()),
                support_remap="nearest C256 lattice coordinate",
            ),
        )
    common.save(
        shared / "shape1024/endpoint.pt",
        coords=shape["coords"],
        features=shape["features"],
        normalized=True,
    )
    texture_guidance.baseline_fields(
        pipe,
        out,
        canonical,
        camera,
        root / "baseline_fields",
        decode_fields=False,
    )
    texture_guidance.lift_guides_from_c64_endpoints(
        root / "baseline_fields",
        encoded["coords"],
        root / "guides",
    )
    data = mapping(
        encoded["coords"],
        1024,
        camera,
        shared / "shape1024",
        canonical["image_4096"],
    )
    visibility.classify_blocks(data, encoded["visible"], shared)
    common.save(shared / "mapping.pt", **data)
    texture.conditions(
        pipe,
        out / "sr",
        shared / "texture_conditions",
        data,
        camera,
    )
    payload = dict(
        status="COMPLETE",
        image=str(image.resolve()),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        camera=camera,
        C128=len(c128["coords"]),
        C256=len(encoded["coords"]),
        prepare_seconds=time.perf_counter() - started,
        fixed_geometry=str((out / "sr/final_mesh").resolve()),
    )
    common.js(marker, payload)
    _trim_geometry_cache(out)
    return payload


def texture_seeds(args, image, out, seed_values):
    from sr_tools import common, texture_explore

    root = out / "texture_explore"
    started = time.perf_counter()
    pipe = common.setup(root)
    records = []
    for seed in seed_values:
        seed_out = root / f"seed_{seed}"
        result_path = seed_out / "result.json"
        if result_path.exists():
            records.append(json.loads(result_path.read_text()))
            _link_selected_texture(out, seed_values[0] if seed_values else seed)
            continue
        flow_started = time.perf_counter()
        payload = texture_explore.flow(
            pipe,
            root,
            seed_out / "texture",
            seed=seed,
            scope_tile=None,
            guide_steps=args.guide_steps,
            guide_threshold=args.guide_threshold,
            hidden_mode=args.hidden_mode,
        )
        flow_seconds = time.perf_counter() - flow_started
        _, report = texture_explore.decode_evaluate_render(
            pipe,
            root,
            seed_out,
            payload,
            render=not args.no_render,
        )
        record = dict(
            status="COMPLETE",
            image=str(image.resolve()),
            seed=seed,
            metrics=report["results"],
            texture_flow_seconds=flow_seconds,
            decode_seconds=json.loads(
                (seed_out / "result.json").read_text()
            )["decode_seconds"],
            render_seconds=json.loads(
                (seed_out / "result.json").read_text()
            )["render_seconds"],
            routing=(
                "visible conditional; trusted hidden endpoint then "
                f"{args.hidden_mode}; separate forward per condition type"
            ),
        )
        common.js(result_path, record)
        _remove_flow_checkpoints(seed_out / "texture")
        _link_selected_texture(out, seed_values[0] if seed_values else seed)
        records.append(record)
    common.js(
        root / "seed_summary.json",
        dict(
            image=str(image.resolve()),
            seeds=list(seed_values),
            records=records,
            texture_stage_seconds=time.perf_counter() - started,
        ),
    )
    return records


def summary(args, image_paths):
    import statistics

    rows = []
    for image in image_paths:
        out = image_record(args.output_root, image)
        p = out / "texture_explore/seed_summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        recs = d["records"]
        if not recs:
            continue
        row = {"image": image.name, "seeds": [r["seed"] for r in recs]}
        for name in ("psnr_db", "ssim", "lpips"):
            row[name] = statistics.mean(r["metrics"]["sr"][name] for r in recs)
        row["baseline_psnr_db"] = recs[0]["metrics"]["baseline1024"]["psnr_db"]
        row["baseline_ssim"] = recs[0]["metrics"]["baseline1024"]["ssim"]
        row["baseline_lpips"] = recs[0]["metrics"]["baseline1024"]["lpips"]
        timing_path = out / "full_timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            row["baseline_seconds"] = timing.get("baseline_seconds")
            row["full_pipeline_seconds"] = timing.get("full_pipeline_seconds")
            row["geometry_seconds"] = timing.get("geometry_seconds")
            row["prepare_seconds"] = timing.get("prepare_seconds")
        row["texture_flow_seconds_mean"] = statistics.mean(
            r["texture_flow_seconds"] for r in recs
        )
        row["decode_seconds_mean"] = statistics.mean(
            r["decode_seconds"] for r in recs
        )
        row["render_seconds_mean"] = statistics.mean(
            r["render_seconds"] for r in recs
        )
        rows.append(row)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    if rows:
        for key in (
            "psnr_db",
            "ssim",
            "lpips",
            "baseline_psnr_db",
            "baseline_ssim",
            "baseline_lpips",
            "baseline_seconds",
            "full_pipeline_seconds",
            "geometry_seconds",
            "prepare_seconds",
            "texture_flow_seconds_mean",
            "decode_seconds_mean",
            "render_seconds_mean",
        ):
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                all_metrics[key] = statistics.mean(values)
    payload = dict(status="COMPLETE", images=len(rows), rows=rows, mean=all_metrics)
    (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def smoke(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_visibility_guidance.py",
            "tests/test_sync.py",
            "tests/test_decoder_output_chunks.py",
        ],
        env=env,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("unit smoke tests failed")
    return {"status": "SMOKE_PASS"}


def main():
    args = args_parser()
    args.assets = args.assets.resolve()
    args.output_root = args.output_root.resolve()
    # Bind the physical GPU before any pipeline helper touches CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTHONUNBUFFERED"] = "1"
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_values = [int(v) for v in args.seeds.split(",") if v.strip()]
    image_paths = images(args)
    if args.phase == "smoke":
        print(json.dumps(smoke(args)), flush=True)
        return 0
    if args.phase == "summary":
        print(json.dumps(summary(args, image_paths), indent=2), flush=True)
        return 0
    if args.phase == "prepare":
        prepare_started = time.perf_counter()
        rows = []
        for index, image in enumerate(image_paths, 1):
            out = image_record(args.output_root, image)
            out.mkdir(parents=True, exist_ok=True)
            marker = out / "visibility_prepare_timing.json"
            print(f"PREPARE {index}/{len(image_paths)} {image.name}", flush=True)
            started = time.perf_counter()
            b = baseline(args, image, out)
            g = geometry(args, image, out)
            p = prepare(args, image, out)
            timing = dict(
                image=image.name,
                baseline_seconds=b["baseline_seconds"],
                geometry_seconds=g["geometry_seconds"],
                prepare_seconds=p["prepare_seconds"],
                prepare_total_seconds=time.perf_counter() - started,
                status="COMPLETE",
            )
            marker.write_text(json.dumps(timing, indent=2) + "\n")
            rows.append(timing)
            print(f"PREPARED {image.name} {timing}", flush=True)
        payload = dict(
            status="COMPLETE",
            phase="prepare",
            images=len(rows),
            rows=rows,
            batch_seconds=time.perf_counter() - prepare_started,
        )
        (args.output_root / "prepare_batch.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    batch_started = time.perf_counter()
    rows = []
    for index, image in enumerate(image_paths, 1):
        out = image_record(args.output_root, image)
        out.mkdir(parents=True, exist_ok=True)
        print(f"IMAGE {index}/{len(image_paths)} {image.name}", flush=True)
        started = time.perf_counter()
        b = baseline(args, image, out)
        g = geometry(args, image, out)
        p = prepare(args, image, out)
        records = texture_seeds(args, image, out, seed_values)
        timing = dict(
            baseline_seconds=b["baseline_seconds"],
            geometry_seconds=g["geometry_seconds"],
            prepare_seconds=p["prepare_seconds"],
            texture_seed_seconds={str(r["seed"]): r["texture_flow_seconds"] for r in records},
            full_pipeline_seconds=time.perf_counter() - started,
            seed_count=len(records),
        )
        (out / "full_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        rows.append(dict(image=image.name, **timing))
        print(f"DONE {image.name} {timing}", flush=True)
    common = dict(
        status="COMPLETE",
        images=len(rows),
        seeds=seed_values,
        guide_steps=args.guide_steps,
        guide_threshold=args.guide_threshold,
        hidden_mode=args.hidden_mode,
        rows=rows,
        batch_seconds=time.perf_counter() - batch_started,
    )
    (args.output_root / "batch.json").write_text(json.dumps(common, indent=2) + "\n")
    print(json.dumps(summary(args, image_paths), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
