#!/usr/bin/env python3
"""Render the final C2048 PBR mesh with all eight C128/C64 cube borders."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import utils3d
from PIL import Image, ImageDraw, ImageFont

import pixal3d_c128_baseline_endpoint_renoise_uncond_2048 as experiment
import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import MeshRenderer, PbrMeshRenderer
from pixal3d.renderers.pbr_mesh_renderer import intrinsics_to_projection
from pixal3d.representations import Mesh, MeshWithVoxel
from pixal3d.utils import render_utils
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid
from render_pixal3d_raw_ovoxel import load_envmap


DEFAULT_EXPERIMENT = Path("outputs/c128_baseline_endpoint_renoise_cond_2048_cuda4")
DEFAULT_SHARED = Path("outputs/c128_baseline_endpoint_renoise_uncond_2048_cuda4")
OUTER_COLOR = (255, 174, 66)
INNER_COLOR = (61, 196, 255)


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def project(points: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, resolution: int) -> torch.Tensor:
    near, far = 0.01, 20.0
    projection = intrinsics_to_projection(intrinsics, near, far)
    full = projection @ extrinsics
    homo = torch.cat((points.to(full.device), torch.ones((points.shape[0], 1), device=full.device)), 1)
    clip = homo @ full.T
    ndc = clip[:, :2] / clip[:, 3:4]
    pixels = torch.empty_like(ndc)
    pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * resolution
    pixels[:, 1] = (0.5 + ndc[:, 1] * 0.5) * resolution
    return pixels.detach().cpu()


def append_cylinder(
    vertices: list[torch.Tensor],
    faces: list[torch.Tensor],
    attrs: list[torch.Tensor],
    start: torch.Tensor,
    end: torch.Tensor,
    radius: float,
    color: tuple[int, int, int],
    sides: int = 10,
) -> None:
    direction = end - start
    direction = direction / torch.linalg.norm(direction)
    helper = torch.tensor([0.0, 0.0, 1.0])
    if abs(float(torch.dot(direction, helper))) > 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0])
    axis_a = torch.linalg.cross(direction, helper)
    axis_a = axis_a / torch.linalg.norm(axis_a)
    axis_b = torch.linalg.cross(direction, axis_a)
    base = sum(value.shape[0] for value in vertices)
    rings = []
    for center in (start, end):
        ring = []
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            ring.append(center + radius * (math.cos(angle) * axis_a + math.sin(angle) * axis_b))
        rings.append(torch.stack(ring))
    vertices.append(torch.cat(rings, 0))
    attrs.append(torch.tensor(color, dtype=torch.float32).div(255).repeat(2 * sides, 1))
    cylinder_faces = []
    for index in range(sides):
        nxt = (index + 1) % sides
        cylinder_faces.extend(
            ((base + index, base + nxt, base + sides + nxt),
             (base + index, base + sides + nxt, base + sides + index))
        )
    faces.append(torch.tensor(cylinder_faces, dtype=torch.int32))


def partition_wire_mesh(radius: float, device: torch.device) -> Mesh:
    """Return the exact union of all borders of the 2x2x2 C64 partition."""
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    attrs: list[torch.Tensor] = []
    levels = (-0.5, 0.0, 0.5)
    for varying_axis in range(3):
        fixed_axes = [axis for axis in range(3) if axis != varying_axis]
        for first in levels:
            for second in levels:
                p0 = torch.zeros(3)
                p1 = torch.zeros(3)
                p0[varying_axis], p1[varying_axis] = -0.5, 0.5
                p0[fixed_axes[0]] = p1[fixed_axes[0]] = first
                p0[fixed_axes[1]] = p1[fixed_axes[1]] = second
                outer = abs(first) == 0.5 and abs(second) == 0.5
                append_cylinder(
                    vertices, faces, attrs, p0, p1, radius,
                    OUTER_COLOR if outer else INNER_COLOR,
                )
    return Mesh(torch.cat(vertices).to(device), torch.cat(faces).to(device), torch.cat(attrs).to(device))


def depth_composite(
    pbr: torch.Tensor,
    geometry_depth: torch.Tensor,
    geometry_mask: torch.Tensor,
    wire_attr: torch.Tensor,
    wire_depth: torch.Tensor,
    wire_mask: torch.Tensor,
    title: str,
) -> Image.Image:
    pbr_hwc = pbr.detach().float().permute(1, 2, 0).cpu().clamp(0, 1)
    wire_hwc = wire_attr.detach().float().permute(1, 2, 0).cpu().clamp(0, 1)
    geom_depth = geometry_depth.detach().float().cpu()
    geom_mask = geometry_mask.detach().float().cpu() > 0.5
    line_depth = wire_depth.detach().float().cpu()
    line_mask = wire_mask.detach().float().cpu().clamp(0, 1)
    visible = (~geom_mask) | (line_depth <= geom_depth + 0.004)
    alpha = line_mask * torch.where(visible, 0.96, 0.13)
    composed = pbr_hwc * (1.0 - alpha[..., None]) + wire_hwc * alpha[..., None]
    image = Image.fromarray((composed.numpy() * 255.0 + 0.5).astype("uint8")).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 18, 624, 116), radius=14, fill=(8, 10, 16, 220))
    draw.text((38, 30), title, font=font(27), fill="white")
    draw.line((40, 82, 92, 82), fill=(*OUTER_COLOR, 255), width=6)
    draw.text((106, 70), "C128 outer boundary", font=font(18), fill="white")
    draw.line((335, 82, 387, 82), fill=(*INNER_COLOR, 255), width=6)
    draw.text((401, 70), "C64 split boundary", font=font(18), fill="white")
    return image.convert("RGB")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--baseline-dir", type=Path, default=experiment.DEFAULT_BASELINE)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [value.strip() for value in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.physical_cuda}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output = args.output or args.experiment_dir / "final_n12_pbr_with_c64_cube_wireframes.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    camera = json.loads((args.baseline_dir / "global_camera.json").read_text(encoding="utf-8"))
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    shape_payload = torch.load(
        args.shared_dir / "support" / "fixed_c128_shape_slat.pt",
        map_location="cpu", weights_only=False,
    )
    texture_payload = torch.load(
        args.experiment_dir / "flow" / "prefix_12" / "final_texture_normalized.pt",
        map_location="cpu", weights_only=False,
    )
    coords = shape_payload["coords"].int().to(device)
    shape = SparseTensor(shape_payload["raw_features"].float().to(device), coords)
    shape_meshes, subdivisions = pipeline.decode_shape_slat(shape, 2048)
    tex_mean, tex_std = cube_flow._norm_tensors(pipeline.tex_slat_normalization, 32)
    texture_raw = texture_payload["normalized_features"].float() * tex_std + tex_mean
    texture = SparseTensor(texture_raw.to(device), coords)
    tex_voxel = pipeline.decode_tex_slat(texture, subdivisions)[0]
    geometry = shape_meshes[0]
    mesh = MeshWithVoxel(
        geometry.vertices, geometry.faces,
        origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / 2048,
        coords=tex_voxel.coords[:, 1:], attrs=tex_voxel.feats,
        voxel_shape=torch.Size([*tex_voxel.shape, *tex_voxel.spatial_shape]),
        layout=dict(pipeline.pbr_attr_layout),
    )

    # The canonical condition and every C64 block share this fixed camera.
    # Decoder coordinates g map to CV as (g.x, -g.y, s*d - g.z).
    # ProjGrid uses endpoint samples i/(R-1)-0.5; decoder cells use i/R-0.5.
    scale = float(camera.get("mesh_scale", 1.0))
    distance = float(camera["distance"]) * scale
    origin = torch.tensor([0.0, 0.0, distance], device=device)
    extrinsics, canonical_intrinsics = render_utils.proj_camera_to_render_params(
        float(camera["camera_angle_x"]), distance,
    )
    frame_points = torch.tensor(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=torch.float32, device=device,
    )
    frame_pixels = project(frame_points, extrinsics, canonical_intrinsics, args.resolution)
    # Extend the sensor canvas, preserving pixel focal length and camera pose.
    overflow = max(0.0, float((-frame_pixels).max()),
                   float((frame_pixels - args.resolution).max()))
    padding = math.ceil(overflow + 32)
    render_resolution = args.resolution + 2 * padding
    intrinsics = canonical_intrinsics.clone()
    intrinsics[0, 0] *= args.resolution / render_resolution
    intrinsics[1, 1] *= args.resolution / render_resolution

    # Verify against the actual condition projector, including its Blender rotation.
    indices = torch.tensor([[x,y,z] for x in (0,32,64,127)
                            for y in (0,32,64,127) for z in (0,32,64,127)], device=device)
    projector = ProjGrid(128, args.resolution).to(device)
    uv, _, _ = projector.project_grid_indices(
        camera_angle_x=torch.tensor([camera["camera_angle_x"]], device=device),
        distance=torch.tensor([camera["distance"]], device=device),
        mesh_scale=torch.tensor([scale], device=device),
        grid_indices=indices, grid_resolution=128,
    )
    render_uv = project(indices.float()/127 - 0.5, extrinsics,
                        canonical_intrinsics, args.resolution)
    projection_error = float((render_uv - uv[0].cpu()).abs().max())
    if projection_error > 0.001:
        raise RuntimeError(f"Condition/render projection mismatch: {projection_error} px")

    mesh_bounds_min = geometry.vertices.amin(dim=0).detach().float().cpu()
    mesh_bounds_max = geometry.vertices.amax(dim=0).detach().float().cpu()
    if bool((mesh_bounds_min < -0.501).any() or (mesh_bounds_max > 0.501).any()):
        raise RuntimeError(
            "Decoded mesh lies outside the expected C128 world frame: "
            f"min={mesh_bounds_min.tolist()} max={mesh_bounds_max.tolist()}"
        )
    near = max(0.01, distance - 2.0)
    far = distance + 10.0
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": render_resolution, "near": near,
            "far": far, "ssaa": 1, "peel_layers": 8,
            "face_chunk_size": 4_000_000,
        },
        device=str(device),
    )
    result = renderer.render(
        mesh, extrinsics, intrinsics, envmap=load_envmap("studio", device=device),
        use_envmap_bg=False,
    )
    geometry_renderer = MeshRenderer(
        rendering_options={
            "resolution": render_resolution, "near": near, "far": far, "ssaa": 1,
            "chunk_size": 4_000_000, "antialias": False,
        },
        device=str(device),
    )
    geometry_buffers = geometry_renderer.render(
        geometry, extrinsics, intrinsics, return_types=["depth", "mask"],
    )
    wire = partition_wire_mesh(radius=0.0035, device=device)
    wire_renderer = MeshRenderer(
        rendering_options={
            "resolution": render_resolution, "near": near, "far": far, "ssaa": 1,
            "chunk_size": None, "antialias": True,
        },
        device=str(device),
    )
    wire_buffers = wire_renderer.render(
        wire, extrinsics, intrinsics, return_types=["attr", "depth", "mask"],
    )
    composed = depth_composite(
        result["shaded"], geometry_buffers["depth"], geometry_buffers["mask"],
        wire_buffers["attr"], wire_buffers["depth"], wire_buffers["mask"],
        "Fixed generation camera / extended canvas",
    )
    full_output = output.with_name(output.stem + "_full_frame.png")
    composed.save(full_output)
    crop_box = (padding, padding, padding + args.resolution, padding + args.resolution)
    composed.crop(crop_box).save(output)
    reference = Image.open(args.baseline_dir / "canonical_1024.png").convert("RGB")
    reference = reference.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
    raw_pbr = Image.fromarray((result["shaded"].float().permute(1,2,0).cpu().clamp(0,1).numpy()*255).astype("uint8"))
    comparison = Image.new("RGB", (3*args.resolution, args.resolution+64), (20,22,28))
    for index, (panel, label) in enumerate([
        (reference, "Canonical reference"),
        (raw_pbr.crop(crop_box), "PBR / same generation camera"),
        (composed.crop(crop_box), "C128 / 8 shared-camera C64 blocks"),
    ]):
        comparison.paste(panel, (index*args.resolution,64))
        ImageDraw.Draw(comparison).text((index*args.resolution+20,18),label,font=font(25),fill="white")
    comparison_output = output.with_name(output.stem + "_reference_comparison.png")
    comparison.save(comparison_output)
    metadata = {
        "output": str(output.resolve()), "resolution": args.resolution,
        "texture_variant": "conditional n=12", "decode_resolution": 2048,
        "latent_grid": 128, "cube_context": 64, "stride": 64,
        "cube_starts": [[x, y, z] for x in (0, 64) for y in (0, 64) for z in (0, 64)],
        "yaw_deg": 0.0, "pitch_deg": 0.0,
        "source_camera": str(args.baseline_dir / "global_camera.json"),
        "reference_image": str(args.baseline_dir / "canonical_1024.png"),
        "mesh_scale": scale,
        "condition_projection_max_error_px": projection_error,
        "extrinsics_world_to_cv": extrinsics.cpu().tolist(),
        "canonical_intrinsics_normalized": canonical_intrinsics.cpu().tolist(),
        "extended_intrinsics_normalized": intrinsics.cpu().tolist(),
        "extended_canvas_padding": padding,
        "full_frame_output": str(full_output),
        "comparison_output": str(comparison_output),
        "camera_distance": distance,
        "camera_origin": origin.detach().float().cpu().tolist(),
        "camera_angle_x": float(camera["camera_angle_x"]),
        "camera_fit": "disabled; fixed generation camera with sensor canvas padding",
        "projected_frame_bbox_pixels": [
            frame_pixels.amin(dim=0).tolist(), frame_pixels.amax(dim=0).tolist(),
        ],
        "mesh_bounds_min": mesh_bounds_min.tolist(),
        "mesh_bounds_max": mesh_bounds_max.tolist(),
        "partition_world_levels_xyz": [-0.5, 0.0, 0.5],
        "coordinate_mapping": "C128 index / 128 - 0.5, independently on renderer world x/y/z",
        "depth_tested": True,
        "visible_wire_alpha": 0.96,
        "occluded_wire_alpha": 0.13,
    }
    experiment.atomic_json(output.with_suffix(".json"), metadata)
    print(f"[done] {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
