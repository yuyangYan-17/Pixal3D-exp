#!/usr/bin/env python3
"""Prepare a small, fully cached Pixal3D dataset from material-bearing GLBs.

The expensive geometry, encoding, rendering and canonical image operations are
delegated to the existing Pixal3D/data_toolkit implementations.  This file is
intentionally an orchestration layer plus the dataset-specific projection and
selection logic requested in ``Codex.md``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any

import numpy as np
from PIL import Image


VOXEL_RESOLUTION = 1024
SLAT_RESOLUTION = 64
RENDER_RESOLUTION = 4096
IMAGE_RESOLUTION = 1024
TILE_SIZE = 1024
TILE_STRIDE = 512
CAMERA_RADIUS = 2.0
CAMERA_FOV_DEG = 40.0
FACE_CHUNK_SIZE = 4_000_000
SHAPE_ENCODER = "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TEXTURE_ENCODER = "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def glb_has_material(path: Path) -> bool:
    """Read only the GLB JSON chunk and require at least one material."""
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            return False
        magic, version, _ = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            return False
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            return False
        length, kind = struct.unpack("<I4s", chunk_header)
        if kind != b"JSON":
            return False
        document = json.loads(stream.read(length).decode("utf-8").rstrip("\x00 \t\r\n"))
    return bool(document.get("materials"))


def discover_glbs(input_root: Path, output_root: Path, workers: int) -> list[dict[str, Any]]:
    output_resolved = output_root.resolve()
    paths = [
        path for path in input_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".glb"
        and output_resolved not in path.resolve().parents
    ]
    # Staged GLBs are scanned as well.  This makes --resume work after the
    # default move operation even when a previous run stopped mid-object.
    if output_root.is_dir():
        paths.extend(
            path for path in output_root.glob("*/*.glb")
            if path.is_file() and path.stem == path.parent.name
        )
    paths = sorted(set(paths), key=lambda value: str(value))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(sha256_file, paths))
    records: dict[str, dict[str, Any]] = {}
    for path, full_hash in zip(paths, hashes):
        record = records.setdefault(full_hash, {"sha256": full_hash, "sources": []})
        record["sources"].append(path)
    return list(records.values())


def choose_object_ids(records: list[dict[str, Any]], min_length: int = 16) -> None:
    """Use short content hashes, extending only in the unlikely collision case."""
    for record in records:
        length = min_length
        while any(
            other is not record
            and other.get("object_id") == record["sha256"][:length]
            and other["sha256"] != record["sha256"]
            for other in records
        ):
            length += 4
        record["object_id"] = record["sha256"][:length]


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _npz_valid(path: Path) -> bool:
    try:
        with np.load(path) as data:
            return "coords" in data and "feats" in data and len(data["coords"]) == len(data["feats"])
    except Exception:
        return False


def object_is_complete(object_dir: Path, expected_views: int, full_hash: str) -> bool:
    meta_path = object_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("status") != "complete" or meta.get("sha256") != full_hash:
        return False
    if meta.get("num_selected_views") != expected_views:
        return False
    object_id = object_dir.name
    required = [
        object_dir / f"{object_id}.glb",
        object_dir / "slat" / "shape_gt_c64.npz",
        object_dir / "slat" / "texture_gt_c64.npz",
        object_dir / "views" / "views.json",
    ]
    for index in range(expected_views):
        view = object_dir / "views" / f"view_{index:03d}"
        required.extend([view / "image_4096.png", view / "image_1024.png", view / "camera.json", view / "tile_indices.npz"])
    return all(path.is_file() and path.stat().st_size > 0 for path in required) and all(
        _npz_valid(path) for path in required if path.suffix == ".npz" and "tile_indices" not in path.name
    )


def object_is_skipped(object_dir: Path, full_hash: str) -> bool:
    try:
        meta = json.loads((object_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("status") == "skipped" and meta.get("sha256") == full_hash


def mark_object_skipped(
    object_dir: Path,
    object_id: str,
    full_hash: str,
    stage: str,
    error: BaseException | str,
) -> dict[str, str]:
    """Keep the staged GLB and remove every partial training artifact."""
    for name in (".work", ".candidate_tmp", "slat", "views"):
        path = object_dir / name
        if path.exists():
            shutil.rmtree(path)
    record = {
        "object_id": object_id,
        "stage": stage,
        "error": str(error),
    }
    _json_dump(object_dir / "meta.json", {
        "status": "skipped",
        "object_id": object_id,
        "sha256": full_hash,
        "failed_stage": stage,
        "error": str(error),
    })
    return record


def stage_glb(record: dict[str, Any], output_root: Path, copy_inputs: bool) -> Path:
    object_id = record["object_id"]
    object_dir = output_root / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    destination = object_dir / f"{object_id}.glb"
    if destination.exists():
        if sha256_file(destination) != record["sha256"]:
            raise RuntimeError(f"hash mismatch in existing staged GLB: {destination}")
        return destination
    source = record["sources"][0]
    if copy_inputs:
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))
    return destination


def resolve_blender() -> str:
    override = os.environ.get("PIXAL3D_BLENDER_PATH")
    candidates = [override, shutil.which("blender"), "/tmp/blender-4.5.1-linux-x64/blender"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            blender = str(Path(candidate).resolve())
            break
    else:
        from data_toolkit import dump_mesh
        dump_mesh._install_blender()
        if not Path(dump_mesh.BLENDER_PATH).is_file():
            raise RuntimeError("Blender installation failed; set PIXAL3D_BLENDER_PATH")
        blender = dump_mesh.BLENDER_PATH
    # dump_pbr.py imports Pillow inside Blender.  Reuse the toolkit's installer
    # instead of assuming Blender shares packages with the host interpreter.
    installer = Path(__file__).resolve().parent / "data_toolkit" / "blender_script" / "install_pillow.py"
    subprocess.run([blender, "-b", "--python", str(installer)], check=True)
    return blender


def build_ovoxels(object_dir_text: str, object_id: str, blender_path: str) -> str:
    """CPU-heavy reusable dump -> 1024 O-Voxel stages (worker-safe)."""
    from easydict import EasyDict as edict
    from data_toolkit import dump_mesh, dump_pbr, dual_grid, voxelize_pbr

    object_dir = Path(object_dir_text)
    glb = object_dir / f"{object_id}.glb"
    work = object_dir / ".work"
    for name in ("mesh_dumps", "pbr_dumps", "dual_grid_1024", "pbr_voxels_1024"):
        (work / name).mkdir(parents=True, exist_ok=True)
    dump_mesh.BLENDER_PATH = blender_path
    dump_pbr.BLENDER_PATH = blender_path
    mesh_dump = work / "mesh_dumps" / f"{object_id}.pickle"
    pbr_dump = work / "pbr_dumps" / f"{object_id}.pickle"
    if not mesh_dump.exists():
        dump_mesh._dump_mesh(str(glb), object_id, str(work))
    if not pbr_dump.exists():
        dump_pbr._dump_pbr(str(glb), object_id, str(work))
    dual_grid.opt = edict(resolution=[VOXEL_RESOLUTION])
    voxelize_pbr.opt = edict(resolution=[VOXEL_RESOLUTION])
    dual_path = work / "dual_grid_1024" / f"{object_id}.vxz"
    pbr_path = work / "pbr_voxels_1024" / f"{object_id}.vxz"
    if not dual_path.exists():
        result = dual_grid._dual_grid_mesh(str(glb), {"sha256": object_id}, str(work), str(work))
        if result.get("error"):
            raise RuntimeError(result["error"])
    if not pbr_path.exists():
        result = voxelize_pbr._pbr_voxelize(str(glb), {"sha256": object_id}, str(work), str(work))
        if result.get("error"):
            raise RuntimeError(result["error"])
    return object_dir_text


def load_encoders() -> tuple[Any, Any]:
    import pixal3d.models as models
    print(f"[models] loading {SHAPE_ENCODER}", flush=True)
    shape = models.from_pretrained(SHAPE_ENCODER).eval().cuda()
    print(f"[models] loading {TEXTURE_ENCODER}", flush=True)
    texture = models.from_pretrained(TEXTURE_ENCODER).eval().cuda()
    return shape, texture


def encode_global_slats(object_dir: Path, object_id: str, shape_encoder: Any, texture_encoder: Any) -> np.ndarray:
    import o_voxel
    import torch
    import pixal3d.modules.sparse as sp

    slat_dir = object_dir / "slat"
    slat_dir.mkdir(exist_ok=True)
    shape_path = slat_dir / "shape_gt_c64.npz"
    texture_path = slat_dir / "texture_gt_c64.npz"
    work = object_dir / ".work"
    if not _npz_valid(shape_path):
        coords, attr = o_voxel.io.read_vxz(str(work / "dual_grid_1024" / f"{object_id}.vxz"), num_threads=4)
        vertices = sp.SparseTensor(
            (attr["vertices"] / 255.0).float(),
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        )
        intersected = vertices.replace(torch.cat([
            attr["intersected"] % 2,
            attr["intersected"] // 2 % 2,
            attr["intersected"] // 4 % 2,
        ], dim=-1).bool())
        with torch.no_grad():
            latent = shape_encoder(vertices.cuda(), intersected.cuda())
        np.savez_compressed(
            shape_path,
            coords=latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
            feats=latent.feats.cpu().numpy().astype(np.float32),
        )
    if not _npz_valid(texture_path):
        coords, attr = o_voxel.io.read_vxz(str(work / "pbr_voxels_1024" / f"{object_id}.vxz"), num_threads=4)
        feats = torch.cat([attr[key] for key in ("base_color", "metallic", "roughness", "alpha")], dim=-1) / 255.0 * 2 - 1
        voxels = sp.SparseTensor(feats.float(), torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1))
        with torch.no_grad():
            latent = texture_encoder(voxels.cuda())
        np.savez_compressed(
            texture_path,
            coords=latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
            feats=latent.feats.cpu().numpy().astype(np.float32),
        )
    with np.load(shape_path) as shape_data, np.load(texture_path) as texture_data:
        shape_coords = shape_data["coords"].copy()
        texture_coords = texture_data["coords"]
        if not np.array_equal(shape_coords, texture_coords):
            raise RuntimeError("shape and texture C64 encoders produced different global support")
    return shape_coords


def tile_boxes() -> list[tuple[int, int, int, int]]:
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    return Pixal3DImageTo3DPipeline.build_texture_image_tile_layout(
        RENDER_RESOLUTION, TILE_SIZE, TILE_STRIDE
    )


def project_global_support(
    coords: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    crop: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    points = (coords.astype(np.float64) + 0.5) / SLAT_RESOLUTION - 0.5
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = homogeneous @ extrinsics.T
    depth = camera[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.0)
    source_uv = np.stack([
        intrinsics[0, 0] * camera[:, 0] / safe_depth + intrinsics[0, 2],
        intrinsics[1, 1] * camera[:, 1] / safe_depth + intrinsics[1, 2],
    ], axis=1) * RENDER_RESOLUTION
    left, top, right, bottom = crop["square_extent_source"]
    side = float(right - left)
    canonical_uv = (source_uv - np.array([left, top], dtype=np.float64)) * (RENDER_RESOLUTION / side)
    valid = (
        (depth > 0)
        & np.isfinite(canonical_uv).all(axis=1)
        & (canonical_uv[:, 0] >= 0) & (canonical_uv[:, 0] <= RENDER_RESOLUTION)
        & (canonical_uv[:, 1] >= 0) & (canonical_uv[:, 1] <= RENDER_RESOLUTION)
    )
    indices: list[np.ndarray] = []
    counts = []
    for x0, y0, x1, y1 in tile_boxes():
        membership = valid & (canonical_uv[:, 0] >= x0) & (canonical_uv[:, 1] >= y0)
        membership &= ((canonical_uv[:, 0] < x1) | ((x1 == RENDER_RESOLUTION) & (canonical_uv[:, 0] == x1)))
        membership &= ((canonical_uv[:, 1] < y1) | ((y1 == RENDER_RESOLUTION) & (canonical_uv[:, 1] == y1)))
        item = np.flatnonzero(membership).astype(np.int32)
        indices.append(item)
        counts.append(len(item))
    return canonical_uv.astype(np.float32), valid, indices, np.asarray(counts, dtype=np.int64)


def _load_render_mesh(work_dir: Path, object_id: str) -> Any:
    """Load normalized dump geometry with its already-materialized O-Voxel PBR.

    MeshWithVoxel selects the renderer's exact face-chunk path.  Its attributes
    are produced by ``blender_dump_to_volumetric_attr``, so GLB material and
    texture parsing still comes from the project's standard Blender/PBR path.
    """
    import o_voxel
    import torch
    import pixal3d.modules.sparse as sp
    from pixal3d.datasets.sparse_voxel_pbr import SparseVoxelPbrDataset
    from pixal3d.representations import MeshWithVoxel

    loader = object.__new__(SparseVoxelPbrDataset)
    material_mesh = loader.read_mesh_with_texture(str(work_dir / "pbr_dumps"), object_id)["mesh"][0]
    coords, attr = o_voxel.io.read_vxz(str(work_dir / "pbr_voxels_1024" / f"{object_id}.vxz"), num_threads=4)
    feats = torch.cat([attr[key] for key in ("base_color", "metallic", "roughness", "alpha")], dim=-1) / 255.0
    sparse = sp.SparseTensor(feats.float(), torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1))
    return MeshWithVoxel(
        material_mesh.vertices,
        material_mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / VOXEL_RESOLUTION,
        coords=coords,
        attrs=feats.float(),
        voxel_shape=torch.Size([*sparse.shape, *sparse.spatial_shape]),
        layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
    )


def _load_envmap() -> Any:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    import torch
    from pixal3d.renderers import EnvMap
    path = Path(__file__).resolve().parent / "assets" / "hdri" / "studio.exr"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        try:
            import OpenEXR
            channels = OpenEXR.File(str(path)).channels()
            if "RGB" in channels:
                image = np.asarray(channels["RGB"].pixels, dtype=np.float32)
            else:
                image = np.stack(
                    [np.asarray(channels[key].pixels, dtype=np.float32) for key in "RGB"],
                    axis=-1,
                )
        except Exception as error:
            raise RuntimeError(f"failed to load HDR environment map: {path}") from error
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.as_tensor(image, dtype=torch.float32, device="cuda"))


def render_and_select_views(object_dir: Path, object_id: str, coords: np.ndarray, num_views: int) -> None:
    import torch
    from pixal3d.renderers import PbrMeshRenderer
    from pixal3d.utils.random_utils import sphere_hammersley_sequence
    from pixal3d.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

    mesh = _load_render_mesh(object_dir / ".work", object_id).cuda()
    envmap = _load_envmap()
    renderer = PbrMeshRenderer()
    renderer.rendering_options.resolution = RENDER_RESOLUTION
    renderer.rendering_options.ssaa = 1
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.face_chunk_size = FACE_CHUNK_SIZE
    cameras = [sphere_hammersley_sequence(index, num_views) for index in range(num_views)]
    yaws = [camera[0] for camera in cameras]
    pitches = [camera[1] for camera in cameras]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, CAMERA_RADIUS, CAMERA_FOV_DEG
    )
    selected_count = max(1, math.ceil(num_views / 3))
    views_dir = object_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    views_dir.mkdir()
    candidates: list[dict[str, Any]] = []
    temp_parent = object_dir / ".candidate_tmp"
    if temp_parent.exists():
        shutil.rmtree(temp_parent)
    temp_parent.mkdir()
    try:
        for index, (extrinsic, intrinsic) in enumerate(zip(extrinsics, intrinsics)):
            print(f"[{object_id}] rendering candidate {index + 1}/{num_views}", flush=True)
            with torch.no_grad():
                result = renderer.render(mesh, extrinsic, intrinsic, envmap=envmap, use_envmap_bg=False)
            rgb = np.clip(result["shaded"].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            alpha = np.clip(result["alpha"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            rgba = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
            canonical = Pixal3DImageTo3DPipeline.preprocess_canonical_images(None, rgba)
            uv, valid, indices, counts = project_global_support(
                coords,
                extrinsic.detach().cpu().numpy(),
                intrinsic.detach().cpu().numpy(),
                canonical["metadata"],
            )
            candidate_dir = temp_parent / f"candidate_{index:03d}"
            candidate_dir.mkdir()
            canonical["image_4096"].save(candidate_dir / "image_4096.png")
            canonical["image_1024"].save(candidate_dir / "image_1024.png")
            candidates.append({
                "candidate_index": index,
                "variance": float(np.var(counts.astype(np.float64))),
                "counts": counts,
                "indices": indices,
                "uv": uv,
                "valid": valid,
                "extrinsics": extrinsic.detach().cpu().numpy(),
                "intrinsics": intrinsic.detach().cpu().numpy(),
                "yaw": float(yaws[index]),
                "pitch": float(pitches[index]),
                "preprocess": canonical["metadata"],
                "temp_dir": candidate_dir,
            })
            del result, canonical
            torch.cuda.empty_cache()
        candidates.sort(key=lambda value: (value["variance"], value["candidate_index"]))
        manifest_views = []
        for rank, candidate in enumerate(candidates[:selected_count]):
            view_dir = views_dir / f"view_{rank:03d}"
            view_dir.mkdir()
            shutil.move(str(candidate["temp_dir"] / "image_4096.png"), view_dir / "image_4096.png")
            shutil.move(str(candidate["temp_dir"] / "image_1024.png"), view_dir / "image_1024.png")
            camera = {
                "candidate_index": candidate["candidate_index"],
                "yaw_radians": candidate["yaw"],
                "pitch_radians": candidate["pitch"],
                "radius": CAMERA_RADIUS,
                "fov_degrees": CAMERA_FOV_DEG,
                "extrinsics_world_to_camera_opencv": candidate["extrinsics"].tolist(),
                "intrinsics_normalized_opencv": candidate["intrinsics"].tolist(),
                "canonical_preprocess": candidate["preprocess"],
            }
            _json_dump(view_dir / "camera.json", camera)
            tile_pack = {
                "global_coords_c64": coords.astype(np.uint8),
                "projected_uv_4096": candidate["uv"],
                "valid_mask": candidate["valid"].astype(np.bool_),
                "counts": candidate["counts"],
            }
            tile_pack.update({f"tile_{index:03d}": item for index, item in enumerate(candidate["indices"])})
            np.savez_compressed(view_dir / "tile_indices.npz", **tile_pack)
            manifest_views.append({
                "rank": rank,
                "path": f"view_{rank:03d}",
                "image_4096": f"view_{rank:03d}/image_4096.png",
                "image_1024": f"view_{rank:03d}/image_1024.png",
                "camera": f"view_{rank:03d}/camera.json",
                "camera_parameter_index": rank,
                "tile_indices": f"view_{rank:03d}/tile_indices.npz",
            })
        _json_dump(views_dir / "views.json", {"views": manifest_views})
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)
        del renderer, mesh, envmap
        torch.cuda.empty_cache()


def audit_object(object_dir: Path, selected_count: int) -> None:
    forbidden_parts = ("candidate", "raw_render", "before_padding", "score")
    offenders = [
        path for path in object_dir.rglob("*")
        if path.is_file() and any(part in path.name.lower() for part in forbidden_parts)
    ]
    view_dirs = sorted(path for path in (object_dir / "views").glob("view_*") if path.is_dir())
    if offenders:
        raise RuntimeError(f"forbidden intermediate outputs remain: {offenders}")
    if len(view_dirs) != selected_count:
        raise RuntimeError(f"expected {selected_count} final views, found {len(view_dirs)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", "--input-root", type=Path, required=True)
    parser.add_argument("--output_root", "--output-root", type=Path, required=True)
    parser.add_argument("--num_views", "--num-views", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=1)
    parser.add_argument("--copy_inputs", "--copy-inputs", action="store_true", help="copy instead of moving source GLBs")
    parser.add_argument("--limit", type=int, default=None, help="optional cap for a very small smoke dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_views <= 0 or args.num_workers <= 0:
        raise SystemExit("--num_views and --num_workers must be positive")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = discover_glbs(input_root, output_root, args.num_workers)
    choose_object_ids(records)
    material_records = []
    for record in records:
        if glb_has_material(record["sources"][0]):
            material_records.append(record)
        else:
            print(f"[skip] no GLB material: {record['sources'][0]}")
    records = sorted(material_records, key=lambda value: value["object_id"])
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        records = records[:args.limit]
    selected_count = max(1, math.ceil(args.num_views / 3))
    pending = []
    for record in records:
        object_dir = output_root / record["object_id"]
        if args.resume and object_is_complete(object_dir, selected_count, record["sha256"]):
            print(f"[resume] complete, skipping {record['object_id']}")
            continue
        if args.resume and object_is_skipped(object_dir, record["sha256"]):
            print(f"[resume] previously unsupported, skipping {record['object_id']}")
            continue
        if object_dir.exists() and not args.resume and any(object_dir.iterdir()):
            raise RuntimeError(f"output exists; pass --resume to continue: {object_dir}")
        stage_glb(record, output_root, args.copy_inputs)
        pending.append(record)
    if not pending:
        print(f"Nothing to process. {len(records)} unique material GLBs are complete.")
        return
    blender = resolve_blender()
    failures: list[dict[str, str]] = []
    ready: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(build_ovoxels, str(output_root / item["object_id"]), item["object_id"], blender): item
            for item in pending
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            object_id = item["object_id"]
            try:
                future.result()
                ready.append(item)
                print(f"[{object_id}] 1024 O-Voxels ready", flush=True)
            except Exception as error:
                failure = mark_object_skipped(
                    output_root / object_id,
                    object_id,
                    item["sha256"],
                    "glb_dump_or_ovoxel",
                    error,
                )
                failures.append(failure)
                print(f"[{object_id}] SKIPPED during GLB/PBR/O-Voxel preprocessing: {error}", flush=True)
    encoded: list[tuple[dict[str, Any], np.ndarray]] = []
    if ready:
        shape_encoder, texture_encoder = load_encoders()
        try:
            for record in ready:
                object_id = record["object_id"]
                object_dir = output_root / object_id
                try:
                    coords = encode_global_slats(object_dir, object_id, shape_encoder, texture_encoder)
                    encoded.append((record, coords))
                except Exception as error:
                    failures.append(mark_object_skipped(
                        object_dir, object_id, record["sha256"], "slat_encoding", error
                    ))
                    print(f"[{object_id}] SKIPPED during SLat encoding: {error}", flush=True)
        finally:
            import torch
            del shape_encoder, texture_encoder
            torch.cuda.empty_cache()
    for record, coords in encoded:
        object_id = record["object_id"]
        object_dir = output_root / object_id
        try:
            render_and_select_views(object_dir, object_id, coords, args.num_views)
            shutil.rmtree(object_dir / ".work", ignore_errors=True)
            audit_object(object_dir, selected_count)
            staged_glb = object_dir / f"{object_id}.glb"
            _json_dump(object_dir / "meta.json", {
                "status": "complete",
                "object_id": object_id,
                "sha256": record["sha256"],
                "duplicate_sources": [
                    str(path) for path in record["sources"]
                    if path.resolve() != staged_glb.resolve()
                ][1:],
                "voxel_resolution": VOXEL_RESOLUTION,
                "slat_resolution": SLAT_RESOLUTION,
                "num_candidate_views": args.num_views,
                "num_selected_views": selected_count,
                "selection_metric": "population_variance_of_49_overlapping_tile_point_counts",
                "canonical_resolution": RENDER_RESOLUTION,
                "tile_size": TILE_SIZE,
                "tile_stride": TILE_STRIDE,
            })
            print(f"[{object_id}] complete", flush=True)
        except Exception as error:
            failures.append(mark_object_skipped(
                object_dir, object_id, record["sha256"], "render_or_view_selection", error
            ))
            print(f"[{object_id}] SKIPPED during rendering/view selection: {error}", flush=True)
    complete_objects = [
        item["object_id"] for item in records
        if object_is_complete(output_root / item["object_id"], selected_count, item["sha256"])
    ]
    skipped_objects = []
    for item in records:
        meta_path = output_root / item["object_id"] / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") == "skipped" and meta.get("sha256") == item["sha256"]:
            skipped_objects.append({
                "object_id": item["object_id"],
                "stage": str(meta.get("failed_stage", "unknown")),
                "error": str(meta.get("error", "unknown")),
            })
    _json_dump(output_root / "dataset.json", {
        "objects": complete_objects,
        "skipped": skipped_objects,
        "num_views": args.num_views,
        "num_selected_views": selected_count,
    })
    if failures:
        print(
            f"Completed with {len(failures)} skipped object(s); "
            f"details are recorded in {output_root / 'dataset.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
