#!/usr/bin/env python3
"""Refine one baseline-C64 block through the sparse-structure flow, then decode it."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
import torch.nn.functional as F
from PIL import Image

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global_c256_restructured_blocks_singleview as blocks
import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
import pixal3d_render_global_slat_points_with_local_cubes as composite
from inference import MODEL_PATH, init_pipeline
from pixal3d import models
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT = "pixal3d_baseline_c64_guided_single_head_block_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    baseline = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=baseline / "global_camera.json")
    p.add_argument("--baseline-c64", type=Path, default=Path(
        "outputs/global_c256_restructured_blocks_cuda5/baseline_pre_hr/baseline_c64_support.pt"))
    p.add_argument("--placeholder-support", type=Path, default=Path(
        "outputs/global_c256_cube_owner_flow_singleview_cuda4/support/global_c256_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_baseline_guided_head_block38_cuda5"))
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--ss-encoder", default=
        "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    p.add_argument("--block-id", type=int, default=38)
    p.add_argument("--block-index", default="2,1,2")
    p.add_argument("--direct-c16-upsample", action="store_true",
                   help="Legacy nearest-support C16-to-C32 experiment")
    p.add_argument("--c16-shape-cascade", action="store_true",
                   help="C16 Shape512 flow -> decoder x4 C256 candidates -> baseline quantize C64")
    p.add_argument("--decoder-upsample-times", type=int, choices=(2, 4), default=4,
                   help="C16 cascade decoder subdivisions; x2 lands directly on C64")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shape-seed", type=int, default=43001)
    p.add_argument("--texture-seed", type=int, default=44001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--point-radius", type=int, default=3)
    p.add_argument("--face-chunk-size", type=int, default=1_000_000)
    return p.parse_args()


def decode_ss(decoder, z: torch.Tensor, resolution: int) -> torch.Tensor:
    decoded = decoder(z) > 0
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > .5
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = tuple(int(v) for v in args.block_index.split(","))
    if len(index) != 3 or args.block_id != index[0] * 16 + index[1] * 4 + index[2]:
        raise ValueError("block id/index mismatch")
    camera = json.loads(args.camera.read_text())
    camera["mesh_scale"] = 1.0

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image512, image1024 = canonical["image_512"], canonical["image_1024"]

    baseline = torch.load(args.baseline_c64, map_location="cpu", weights_only=False)["coords"].int()
    base_start = torch.tensor(index, dtype=torch.int32) * 16
    inside = ((baseline[:, 1:4] >= base_start) & (baseline[:, 1:4] < base_start + 16)).all(1)
    local16 = baseline[inside, 1:4] - base_start
    if not len(local16):
        raise RuntimeError("selected baseline C64 block is empty")

    coords64 = None
    if args.c16_shape_cascade:
        # Local analogue of the native baseline cascade.  The decoder receives
        # a real denormalized Shape512 endpoint, never bare support coordinates.
        coords16 = torch.cat((
            torch.zeros((len(local16), 1), dtype=torch.int32), local16.int()), 1)
        shape512_model = pipeline.image_cond_model_shape_512
        cached = blocks.extract_full_image_features(shape512_model, image512, device)
        transform = blocks.local_to_global_camera_transform(
            shape512_model, block_start=tuple(int(v) for v in base_start.tolist()),
            global_resolution=64, global_extent=16,
            distance=float(camera["distance"]), device=device)
        projection_error = blocks.validate_projection_transform(
            shape512_model, transform, local_resolution=16, global_resolution=64,
            block_start=tuple(int(v) for v in base_start.tolist()),
            distance=float(camera["distance"]), fov=float(camera["camera_angle_x"]),
            device=device)
        glob, proj = blocks.project_cached_features(
            shape512_model, cached, transform=transform,
            fov=float(camera["camera_angle_x"]), distance=float(camera["distance"]),
            coords=coords16.to(device), grid_resolution=16)
        torch.manual_seed(args.seed + 20_000 + args.block_id)
        torch.cuda.manual_seed_all(args.seed + 20_000 + args.block_id)
        lr16 = pipeline.sample_shape_slat(
            blocks.condition_from_sparse(glob, proj, coords16.to(device)),
            pipeline.models["shape_slat_flow_model_512"], coords16.to(device),
            {"steps": args.steps})
        decoder = pipeline.models["shape_slat_decoder"]
        if pipeline.low_vram:
            decoder.to(device); decoder.low_vram = True
        upsample_times = int(args.decoder_upsample_times)
        candidates = decoder.upsample(lr16, upsample_times=upsample_times)
        if pipeline.low_vram:
            decoder.cpu(); decoder.low_vram = False
        if upsample_times == 2:
            # C16 * 2^2 = C64 exactly: retain the decoder's native support and
            # avoid the C256-to-C64 requantization used by the x4 experiment.
            coords64 = candidates.int().unique(dim=0).cpu().contiguous()
            candidate_resolution = 64
            quantizer = "none; native decoder C64 support"
        else:
            # Exact 1024-cascade quantizer with the local source resolution changed
            # from C512 to C256: round((candidate + .5) / 256 * (64 - 1)).
            coords64 = torch.cat((
                candidates[:, :1],
                (((candidates[:, 1:] + .5) / 256.0) * 63.0).round().int(),
            ), 1).unique(dim=0).int().cpu().contiguous()
            candidate_resolution = 256
            quantizer = "round((coord + 0.5) / 256 * 63)"
        if torch.any(coords64[:, 1:] < 0) or torch.any(coords64[:, 1:] >= 64):
            raise RuntimeError("C16 Shape cascade produced C64 support outside [0,63]")
        coords32 = torch.empty((0, 4), dtype=torch.int32)
        guide64_xyz = torch.empty((0, 3), dtype=torch.int32)
        guide_roundtrip_c32 = torch.empty((0, 4), dtype=torch.int32)
        cube_flow.atomic_save(output / f"c16_shape512_decoder{upsample_times}_c64_support.pt", {
            "format": FORMAT, "block_id": args.block_id, "block_index": index,
            "baseline_local_c16": local16, "coords16": coords16,
            "shape512_endpoint": lr16.cpu(), "candidates": candidates.cpu(),
            "coords64": coords64,
            "decoder_upsample_times": upsample_times,
            "candidate_resolution": candidate_resolution,
            "quantizer": quantizer,
            "path": ("C16 noise -> Shape512 flow -> shape decoder upsample x2 -> native C64"
                     if upsample_times == 2 else
                     "C16 noise -> Shape512 flow -> shape decoder upsample x4 -> "
                     "C256 candidates -> baseline centre quantizer C64"),
            "projection_max_error_pixels": projection_error,
            "skipped_models": ["sparse_structure_encoder", "sparse_structure_flow_model",
                               "sparse_structure_decoder"],
        })
        del cached, glob, proj, lr16, candidates
        empty_cuda()
    elif args.direct_c16_upsample:
        # Nearest-support 2x upsample: every active C16 voxel owns all eight C32
        # children.  Shape512 will create Gaussian feature noise on this fixed C32
        # support; no SS encoder, SS flow, or SS decoder is invoked.
        offsets = torch.tensor(
            [[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
            dtype=torch.int32,
        )
        xyz32 = (local16[:, None, :] * 2 + offsets[None]).reshape(-1, 3).unique(dim=0)
        coords32 = torch.cat((torch.zeros((len(xyz32), 1), dtype=torch.int32), xyz32), 1)
        guide64_xyz = torch.empty((0, 3), dtype=torch.int32)
        guide_roundtrip_c32 = coords32
        projection_error = None
        cube_flow.atomic_save(output / "direct_c16_to_c32_support.pt", {
            "format": FORMAT, "block_id": args.block_id, "block_index": index,
            "baseline_local_c16": local16, "coords32": coords32,
            "upsample": "nearest support 2x; each C16 voxel expands to 2x2x2 C32 children",
            "skipped_models": ["sparse_structure_encoder", "sparse_structure_flow_model",
                               "sparse_structure_decoder"],
        })
    else:
        # A C16 crop represents a zoomed local object.  Rasterize its voxel centres
        # across the canonical C64 VAE input, then encode it to the flow model's C16x8
        # latent.
        guide64_xyz = torch.round(local16.float() * (63.0 / 15.0)).long().unique(dim=0)
        occupancy64 = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
        occupancy64[0, 0, guide64_xyz[:, 0], guide64_xyz[:, 1], guide64_xyz[:, 2]] = 1.0
        encoder = models.from_pretrained(args.ss_encoder).eval().to(device)
        guide_z = encoder(occupancy64, sample_posterior=False).float()
        encoder.cpu()
        del encoder, occupancy64
        empty_cuda()
        if tuple(guide_z.shape[1:]) != (8, 16, 16, 16):
            raise RuntimeError(f"unexpected encoded guide shape {tuple(guide_z.shape)}")

        ss_model = pipeline.image_cond_model_ss
        cached = blocks.extract_full_image_features(ss_model, image512, device)
        transform = blocks.local_to_global_camera_transform(
            ss_model, block_start=tuple(int(v) for v in base_start.tolist()),
            global_resolution=64, global_extent=16, distance=float(camera["distance"]), device=device)
        projection_error = blocks.validate_projection_transform(
            ss_model, transform, local_resolution=16, global_resolution=64,
            block_start=tuple(int(v) for v in base_start.tolist()), distance=float(camera["distance"]),
            fov=float(camera["camera_angle_x"]), device=device)
        glob, proj = blocks.project_cached_features(
            ss_model, cached, transform=transform, fov=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]), coords=None, grid_resolution=16)
        cond = blocks.condition_from_dense(glob, proj)
        flow = pipeline.models["sparse_structure_flow_model"]
        decoder = pipeline.models["sparse_structure_decoder"]
        flow.to(device).eval()
        decoder.to(device).eval()
        ss_params = {**pipeline.sparse_structure_sampler_params, "steps": args.steps}
        sampled_z = pipeline.sparse_structure_sampler.sample(
            flow, guide_z, **cond, **ss_params, verbose=True,
            tqdm_desc="Baseline-guided sparse structure").samples
        guide_roundtrip_c32 = decode_ss(decoder, guide_z, 32).cpu()
        coords32 = decode_ss(decoder, sampled_z, 32).cpu()
        cube_flow.atomic_save(output / "sparse_structure_guidance.pt", {
            "format": FORMAT, "block_id": args.block_id, "block_index": index,
            "baseline_local_c16": local16, "guide_c64_voxels": guide64_xyz,
            "encoded_guide_z": guide_z.cpu(), "sampled_z": sampled_z.cpu(),
            "guide_roundtrip_c32": guide_roundtrip_c32, "flow_output_c32": coords32,
            "initial_state": "encoded baseline occupancy latent; no random SS noise",
            "projection_max_error_pixels": projection_error,
        })
        flow.cpu()
        del sampled_z, guide_z, cached, glob, proj, cond
        empty_cuda()

    # Original SS-guided and legacy nearest-support branches still need their
    # C32 Shape512 stage.  The C16 cascade above has already produced C64.
    if coords64 is None:
        shape512_model = pipeline.image_cond_model_shape_512
        cached = blocks.extract_full_image_features(shape512_model, image512, device)
        transform = blocks.local_to_global_camera_transform(
            shape512_model, block_start=tuple(int(v) for v in base_start.tolist()),
            global_resolution=64, global_extent=16, distance=float(camera["distance"]), device=device)
        glob, proj = blocks.project_cached_features(
            shape512_model, cached, transform=transform, fov=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]), coords=coords32.to(device), grid_resolution=32)
        torch.manual_seed(args.seed + 20_000 + args.block_id)
        torch.cuda.manual_seed_all(args.seed + 20_000 + args.block_id)
        lr = pipeline.sample_shape_slat(
            blocks.condition_from_sparse(glob, proj, coords32.to(device)),
            pipeline.models["shape_slat_flow_model_512"], coords32.to(device), {"steps": args.steps})
        coords64 = blocks.learned_c64_support(pipeline, lr, device).cpu()
        del cached, glob, proj, lr
    cube_flow.atomic_save(output / "guided_c64_support.pt", {
        "format": FORMAT, "coords": coords64, "source": "baseline-guided sparse structure"})
    empty_cuda()

    # One local block through the HR shape/texture flows.
    row_ids = torch.arange(len(coords64), dtype=torch.long)
    record = {"cube_id": args.block_id, "block_index": index,
              "start": tuple(v * 64 for v in index), "global_row_ids": row_ids,
              "owned_row_ids": row_ids, "local_xyz": coords64[:, 1:4].int()}
    global_coords = coords64.clone().int()
    global_coords[:, 1:4] += torch.tensor(index, dtype=torch.int32) * 64
    shape_cond = blocks.build_global_conditions(
        pipeline, image1024, camera, global_coords, [record], "shape", output, device)
    shape = blocks.synchronous_flow(
        stage="shape", pipeline=pipeline, records=[record], condition=shape_cond,
        output=output, device=device, seed=args.shape_seed, steps=args.steps, concat=None)
    del shape_cond
    empty_cuda()
    tex_cond = blocks.build_global_conditions(
        pipeline, image1024, camera, global_coords, [record], "texture", output, device)
    texture = blocks.synchronous_flow(
        stage="texture", pipeline=pipeline, records=[record], condition=tex_cond,
        output=output, device=device, seed=args.texture_seed, steps=args.steps, concat=shape)
    del tex_cond
    empty_cuda()

    shape_raw = cube_flow.denormalize(shape, pipeline.shape_slat_normalization)
    texture_raw = cube_flow.denormalize(texture, pipeline.tex_slat_normalization)
    decoded = pipeline.decode_latent(
        SparseTensor(shape_raw.to(device), coords64.to(device)),
        SparseTensor(texture_raw.to(device), coords64.to(device)), 1024)
    native = decoded[0]
    local_vertex, local_face = expc._native_mesh_to_pbr(native, device)
    cube_flow.atomic_save(output / "local_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
    cube_flow.atomic_save(output / "local_per_vertex_pbr_mesh.pt", {"format": FORMAT, "mesh": local_vertex})
    cube_flow.atomic_save(output / "local_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": local_face})

    global_start = tuple(v * 64 for v in index)
    placed = composite.place_local_mesh(local_vertex, global_start, float(camera["mesh_scale"]))
    cube_flow.atomic_save(output / "head_block_in_global_coordinates.pt", {"format": FORMAT, "mesh": placed})

    # Use the coherent pre-existing Global C256 support for placeholders; do not
    # reuse the erroneous repeated-object support from the previous experiment.
    placeholder = torch.load(args.placeholder_support, map_location="cpu", weights_only=False)["coords"].int()
    pxyz = placeholder[:, 1:4]
    pstart = torch.tensor(global_start, dtype=torch.int32)
    keep = ~(((pxyz >= pstart) & (pxyz < pstart + 64)).all(1))
    point_q = 2.0 * (pxyz[keep].float() + .5) / 256.0 - 1.0
    points = point_q / (2.0 * float(camera["mesh_scale"]))
    angles = tuple(int(v) for v in args.angles.split(","))
    extrinsics, intrinsics, _ = _make_camera_views(camera["camera_angle_x"], camera["distance"], angles)
    render_dir = output / "global_slat_plus_guided_head_block_multiview_4096"
    render_dir.mkdir(parents=True, exist_ok=True)
    renderer = PbrMeshRenderer({"resolution": args.resolution, "near": .01,
        "far": camera["distance"] + 10, "ssaa": 1, "peel_layers": 8,
        "face_chunk_size": args.face_chunk_size}, device=str(device))
    envmap = load_envmap("studio", device=device)
    live = placed.to(device)
    rgb_paths, normal_paths = [], []
    for angle in angles:
        cloud = composite.point_image(points, extrinsics[angle], intrinsics, args.resolution, args.point_radius)
        rendered = renderer.render(live, extrinsics[angle].to(device), intrinsics.to(device),
                                   envmap=envmap, use_envmap_bg=False)
        alpha = composite.tensor_image(rendered["mask"], "L")
        rgb = Image.composite(composite.tensor_image(rendered["shaded"]), cloud, alpha)
        normal = Image.composite(composite.tensor_image(rendered["normal"]), cloud, alpha)
        rgb_path = render_dir / f"view_{angle:03d}_guided_pbr.png"
        normal_path = render_dir / f"view_{angle:03d}_guided_camera_normal.png"
        rgb.save(rgb_path); normal.save(normal_path)
        rgb_paths.append((angle, rgb_path)); normal_paths.append((angle, normal_path))
    rgb_sheet = render_dir / "guided_head_block_pbr_contact_sheet.png"
    normal_sheet = render_dir / "guided_head_block_camera_normal_contact_sheet.png"
    composite.contact_sheet(rgb_paths, rgb_sheet, "Global C256 SLAT + baseline-guided head block PBR")
    composite.contact_sheet(normal_paths, normal_sheet, "Global C256 SLAT + baseline-guided head block camera normal")
    cube_flow.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "block_id": args.block_id,
        "block_index": index, "baseline_block_tokens": int(len(local16)),
        "guide_c64_voxels": int(len(guide64_xyz)),
        "guide_roundtrip_c32_tokens": int(len(guide_roundtrip_c32)),
        "flow_output_c32_tokens": int(len(coords32)), "guided_c64_tokens": int(len(coords64)),
        "placeholder_points": int(keep.sum()), "vertices": int(local_vertex.vertices.shape[0]),
        "faces": int(local_vertex.faces.shape[0]), "pbr_contact_sheet": str(rgb_sheet.resolve()),
        "camera_normal_contact_sheet": str(normal_sheet.resolve()),
        "support_initialization": (
            ("baseline C16 support -> Shape512 native noise/flow -> decoder upsample x2 "
             "directly to native C64 support" if args.decoder_upsample_times == 2 else
             "baseline C16 support -> Shape512 native noise/flow -> decoder upsample x4 "
             "to C256 candidates -> baseline centre quantizer C64")
            if args.c16_shape_cascade else
            ("direct baseline C16 support 2x nearest upsample to C32; "
             "Shape512 native Gaussian feature noise" if args.direct_c16_upsample else
             "deterministic encoded baseline block latent, directly used as x(t=1)")),
        "skipped_sparse_structure_models": bool(
            args.direct_c16_upsample or args.c16_shape_cascade),
        "seconds": time.perf_counter() - started,
    })
    print(f"[done] {rgb_sheet}", flush=True)


if __name__ == "__main__":
    main()
