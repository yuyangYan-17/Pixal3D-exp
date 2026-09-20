#!/usr/bin/env python3
"""First-stage geometry flow + encoded 2048 texture SLat experiment.

This is deliberately an isolated experiment.  It does not run the second
shape stage or texture flow.  It builds one 2048 flexible dual grid from the
native 1024 textured mesh, transfers the native sparse material field onto
that grid, encodes shape and texture to S128, and then decodes the same
texture SLat with two different geometry SLat endpoints:

  1. baseline encoded geometry S128;
  2. the already-computed first-stage (shape512) flow endpoint.

The two outputs answer whether texture can follow geometry through the shape
decoder using only ``tex_slat + geometry_subs``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/sr_0_img_z_tent/baseline"),
    )
    parser.add_argument(
        "--shape-endpoint",
        type=Path,
        default=Path("outputs/sr_0_img_z_tent/sr/shape512/endpoint.pt"),
        help="first-stage shape512 flow endpoint; must be normalized S128",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/first_stage_texture_follow_20260919"),
    )
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--query-chunk", type=int, default=262144)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    return parser.parse_args()


# CUDA-dependent imports happen only after the physical GPU has been selected.
ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)

import numpy as np
import torch

from pixal3d import models
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations.mesh import MeshWithVoxel
from sr_tools import common, rendering
from sr_tools.baseline import ENCODER
from sr_tools.texture_guidance import TEX_ENCODER
from sr_tools.visibility import lookup_rows


RESOLUTION = 2048
FORMAT = common.FORMAT


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value):
    common.js(path, value)


def load_mesh(path: Path):
    payload = common.load_payload(path)
    if "mesh" not in payload:
        raise RuntimeError(f"mesh payload missing 'mesh': {path}")
    return payload["mesh"]


@torch.no_grad()
def voxelize_2048(mesh, out: Path):
    target = out / "voxelization_2048.pt"
    if target.exists():
        payload = common.load_payload(target)
        assert payload["grid_size"] == RESOLUTION
        return payload

    import o_voxel

    print("[voxel] baseline textured mesh -> flexible dual grid 2048", flush=True)
    indices, dual, intersections = o_voxel.convert.mesh_to_flexible_dual_grid(
        mesh.vertices.detach().float().cpu(),
        mesh.faces.detach().long().cpu(),
        grid_size=RESOLUTION,
        aabb=[[-0.5] * 3, [0.5] * 3],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )
    common.save(
        target,
        coords=indices.cpu().int(),
        dual=dual.cpu().float(),
        intersections=intersections.cpu(),
        grid_size=RESOLUTION,
        aabb=[[-0.5] * 3, [0.5] * 3],
        source="baseline 1024 textured MeshWithVoxel",
    )
    del indices, dual, intersections
    common.empty_cuda()
    return common.load_payload(target)


@torch.no_grad()
def transfer_material(mesh, voxel, pipe, out: Path, chunk: int):
    """Transfer the native 1024 sparse material field to the 2048 dual points.

    This calls the same MeshWithVoxel sparse trilinear query used by the
    renderer.  ``dual`` is in [0,1]^3; MeshWithVoxel uses world coordinates
    [-.5,.5]^3, hence the explicit ``- .5``.
    """
    target = out / "material_2048.pt"
    if target.exists():
        payload = common.load_payload(target)
        assert torch.equal(payload["coords"], voxel["coords"])
        return payload

    mesh_gpu = mesh.to(pipe.device)
    dual = voxel["dual"].float()
    attrs = torch.empty(
        (len(dual), mesh.attrs.shape[1]), dtype=torch.float32, device="cpu"
    )
    zero_rows = 0
    for start in range(0, len(dual), chunk):
        end = min(start + chunk, len(dual))
        values = mesh_gpu.query_attrs(dual[start:end].to(pipe.device) - 0.5)
        values = values.float().cpu()
        attrs[start:end] = values
        zero_rows += int((values.abs().sum(dim=1) <= 1e-8).sum())
        if (start // chunk) % 10 == 0 or end == len(dual):
            print(
                f"[material transfer] {end:,}/{len(dual):,} "
                f"zero_rows={zero_rows:,}",
                flush=True,
            )
        del values
    del mesh_gpu
    common.empty_cuda()
    if not torch.isfinite(attrs).all():
        raise RuntimeError("non-finite material after 1024 -> 2048 transfer")
    common.save(
        target,
        coords=voxel["coords"].int(),
        dual=voxel["dual"].float(),
        attrs=attrs,
        resolution=RESOLUTION,
        query="MeshWithVoxel.query_attrs / flex_gemm sparse trilinear",
        source_voxel_size=float(mesh.voxel_size),
        zero_rows=zero_rows,
    )
    return common.load_payload(target)


@torch.no_grad()
def encode_shape_s128(voxel, pipe, out: Path):
    target = out / "shape_s128.pt"
    if target.exists():
        return common.load_payload(target)

    coords = torch.cat(
        (torch.zeros_like(voxel["coords"][:, :1]), voxel["coords"]), dim=1
    ).int()
    vertices = SparseTensor(
        voxel["dual"].float() * RESOLUTION - voxel["coords"].float(), coords
    ).to(pipe.device)
    intersections = vertices.replace(voxel["intersections"].to(pipe.device))
    print(f"[shape encoder] 2048 dual points={len(coords):,} -> S128", flush=True)
    encoder = models.from_pretrained(str(ENCODER)).eval().to(pipe.device)
    try:
        encoded = encoder(vertices, intersections, sample_posterior=False)
        raw = encoded.feats.float().cpu()
        mean, std = common.normalization_tensors(
            pipe.shape_slat_normalization, raw.device
        )
        normalized = (raw - mean) / std
        result = dict(
            coords=encoded.coords.int().cpu(),
            features=normalized,
            normalized=True,
            resolution=RESOLUTION,
            encoder=str(ENCODER),
            voxel_tokens=len(coords),
        )
    finally:
        encoder.cpu()
    if not torch.isfinite(result["features"]).all():
        raise RuntimeError("shape encoder returned non-finite S128")
    common.save(target, **result)
    del encoder, encoded, vertices, intersections
    common.empty_cuda()
    return common.load_payload(target)


@torch.no_grad()
def encode_texture_s128(material, shape, pipe, out: Path):
    target = out / "tex_s128.pt"
    if target.exists():
        return common.load_payload(target)

    source_coords = torch.cat(
        (torch.zeros_like(material["coords"][:, :1]), material["coords"]), dim=1
    ).int()
    source = SparseTensor(
        (material["attrs"].float() * 2.0 - 1.0).to(pipe.device),
        source_coords.to(pipe.device),
    )
    print(f"[texture encoder] 2048 material points={len(source_coords):,} -> S128", flush=True)
    encoder = models.from_pretrained(str(TEX_ENCODER)).eval().to(pipe.device)
    try:
        encoded = encoder(source, sample_posterior=False)
        encoded_coords = encoded.coords.int().cpu()
        raw = encoded.feats.float().cpu()
    finally:
        encoder.cpu()
    shape_coords = shape["coords"].int().cpu()
    if torch.equal(encoded_coords, shape_coords):
        aligned = raw
        found_fraction = 1.0
        order_equal = True
    else:
        rows, found = lookup_rows(encoded_coords, shape_coords, RESOLUTION // 16)
        if not bool(found.all()):
            missing = int((~found).sum())
            raise RuntimeError(
                "texture encoder support cannot be aligned to shape S128: "
                f"missing={missing}/{len(found)}"
            )
        aligned = raw[rows]
        found_fraction = float(found.float().mean())
        order_equal = False
    normalized = rendering.norm(pipe, "tex", aligned)
    if not torch.isfinite(normalized).all():
        raise RuntimeError("texture encoder returned non-finite S128")
    result = dict(
        coords=shape_coords,
        features=normalized,
        normalized=True,
        resolution=RESOLUTION,
        encoder=str(TEX_ENCODER),
        source_encoder_coords=encoded_coords,
        source_support_fraction=found_fraction,
        source_order_equal=order_equal,
        channels=int(normalized.shape[1]),
    )
    common.save(target, **result)
    save_json(
        out / "support_alignment.json",
        dict(
            shape_points=len(shape_coords),
            texture_encoder_points=len(encoded_coords),
            source_support_fraction=found_fraction,
            source_order_equal=order_equal,
            aligned_to="shape_s128 coords",
        ),
    )
    del encoder, encoded, source, aligned, normalized
    common.empty_cuda()
    return common.load_payload(target)


def save_mesh_variant(mesh, field, coords, pipe, variant_dir: Path, topology):
    variant_dir.mkdir(parents=True, exist_ok=True)
    geometry = mesh.cpu()
    field = field.cpu()
    cpu_subs = rendering.cpu_topology(topology)
    common.save(variant_dir / "geometry_mesh.pt", mesh=geometry)
    common.save(variant_dir / "topology.pt", coords=coords.cpu(), subs=cpu_subs)
    textured = MeshWithVoxel(
        geometry.vertices,
        geometry.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / RESOLUTION,
        coords=field.coords[:, 1:].cpu(),
        attrs=field.feats.float(),
        voxel_shape=torch.Size([*field.shape, *field.spatial_shape]),
        layout=pipe.pbr_attr_layout,
    )
    common.save(variant_dir / "textured_mesh.pt", mesh=textured)
    save_json(
        variant_dir / "decode.json",
        dict(
            resolution=RESOLUTION,
            latent_points=len(coords),
            material_voxels=len(field.coords),
            vertices=len(geometry.vertices),
            faces=len(geometry.faces),
            texture_decoder_inputs=["tex_s128", "geometry_decoder_subs"],
            direct_geometry_latent_to_texture_decoder=False,
        ),
    )
    return textured


@torch.no_grad()
def decode_variant(pipe, shape_payload, tex_payload, out: Path, name: str):
    target = out / name / "textured_mesh.pt"
    if target.exists():
        return load_mesh(target)

    shape = SparseTensor(
        common.denormalize_shape(pipe, shape_payload["features"]).to(pipe.device),
        shape_payload["coords"].to(pipe.device),
    )
    print(f"[decode {name}] shape S128 -> 2048 geometry + subs", flush=True)
    meshes, subs = pipe.decode_shape_slat(shape, RESOLUTION)
    geometry = meshes[0].cpu()
    tex_features = rendering.norm(
        pipe, "tex", tex_payload["features"].to(pipe.device), inverse=True
    )
    tex = SparseTensor(tex_features, tex_payload["coords"].to(pipe.device))
    print(f"[decode {name}] tex S128 + geometry subs -> 2048 material", flush=True)
    field = pipe.decode_tex_slat(tex, subs)
    if not torch.isfinite(field.feats).all():
        raise RuntimeError(f"non-finite decoded material field: {name}")
    result = save_mesh_variant(geometry, field, shape.coords, pipe, out / name, subs)
    del shape, meshes, subs, tex, field, geometry
    common.empty_cuda()
    return result


def load_canonical(baseline_dir: Path):
    return {
        key: common.Image.open(baseline_dir / f"{key}.png").copy()
        for key in ("image_512", "image_1024", "image_4096", "foreground_mask_4096")
    }


def main():
    out = ARGS.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline_dir = ARGS.baseline_dir.resolve()
    shape_endpoint_path = ARGS.shape_endpoint.resolve()
    if not (baseline_dir / "textured_mesh.pt").exists():
        raise FileNotFoundError(baseline_dir / "textured_mesh.pt")
    if not shape_endpoint_path.exists():
        raise FileNotFoundError(shape_endpoint_path)

    torch.set_num_threads(8)
    print(f"[setup] physical GPU={ARGS.gpu}; visible device=cuda:0", flush=True)
    pipe = common.setup(out)
    mesh = load_mesh(baseline_dir / "textured_mesh.pt")
    camera = load_json(baseline_dir / "camera.json")
    canonical = load_canonical(baseline_dir)

    save_json(
        out / "experiment.json",
        dict(
            status="RUNNING",
            resolution=RESOLUTION,
            baseline_mesh=str(baseline_dir / "textured_mesh.pt"),
            shape_first_stage_endpoint=str(shape_endpoint_path),
            gpu=str(ARGS.gpu),
            material_transfer="baseline MeshWithVoxel.query_attrs sparse trilinear",
            texture_decoder_contract="tex_slat + geometry_decoder_subs",
            render_resolution=ARGS.render_resolution,
        ),
    )
    voxel = voxelize_2048(mesh, out)
    material = transfer_material(mesh, voxel, pipe, out, ARGS.query_chunk)
    shape = encode_shape_s128(voxel, pipe, out)
    tex = encode_texture_s128(material, shape, pipe, out)

    endpoint = common.load_payload(shape_endpoint_path)
    if not endpoint.get("normalized", False):
        raise RuntimeError("shape endpoint is not normalized")
    if not torch.equal(endpoint["coords"].int(), shape["coords"].int()):
        raise RuntimeError("first-stage endpoint support differs from encoded S128")
    common.save(
        out / "shape512_flow_endpoint.pt",
        coords=endpoint["coords"].int(),
        features=endpoint["features"].float(),
        normalized=True,
        source=str(shape_endpoint_path),
    )

    baseline_decoded = decode_variant(pipe, shape, tex, out, "baseline_geometry")
    flow_decoded = decode_variant(pipe, endpoint, tex, out, "shape512_flow_geometry")

    if not ARGS.skip_render:
        for name, value in (
            ("baseline_geometry", baseline_decoded),
            ("shape512_flow_geometry", flow_decoded),
        ):
            render_dir = out / name / "render"
            render_dir.mkdir(parents=True, exist_ok=True)
            print(f"[render {name}] {ARGS.render_resolution} multi-view", flush=True)
            rendering.render(
                pipe,
                value,
                render_dir,
                resolution=ARGS.render_resolution,
                camera=camera,
            )

    metrics = None
    if not ARGS.skip_metrics:
        try:
            from sr_tools.metrics import evaluate

            print("[metrics] input-camera 1024 foreground metrics", flush=True)
            metrics = evaluate(
                {
                    "native_baseline1024": mesh,
                    "baseline_geometry": baseline_decoded,
                    "shape512_flow_geometry": flow_decoded,
                },
                canonical,
                camera,
                out / "evaluation_1024",
            )
        except Exception as exc:
            save_json(out / "metrics_failure.json", {"error": repr(exc)})
            print(f"[metrics] skipped after failure: {exc!r}", flush=True)

    result = dict(
        status="COMPLETE",
        resolution=RESOLUTION,
        baseline_mesh_vertices=len(mesh.vertices),
        baseline_mesh_faces=len(mesh.faces),
        voxel_points=len(voxel["coords"]),
        shape_s128_points=len(shape["coords"]),
        shape_s128_channels=int(shape["features"].shape[1]),
        tex_s128_points=len(tex["coords"]),
        tex_s128_channels=int(tex["features"].shape[1]),
        shape_support_equal=True,
        texture_support_aligned=True,
        material_transfer_zero_rows=int(material.get("zero_rows", -1)),
        decoder_contract="texture decoder consumes tex SLat and geometry decoder subs; no direct geometry latent input",
        baseline_geometry_mesh=str(out / "baseline_geometry/textured_mesh.pt"),
        shape512_flow_geometry_mesh=str(out / "shape512_flow_geometry/textured_mesh.pt"),
        metrics=metrics["results"] if metrics is not None else None,
    )
    save_json(out / "result.json", result)
    save_json(out / "experiment.json", {**load_json(out / "experiment.json"), **result})
    print("[done]", out, flush=True)


if __name__ == "__main__":
    main()
