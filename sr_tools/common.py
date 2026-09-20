"""Checkpoint IO, native baseline stages and mesh-to-latent encoding."""

from __future__ import annotations
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from PIL import Image
from pixal3d.modules.sparse import SparseTensor

FORMAT = "pixal3d_local_c32_to_c64_c128_geometry_helpers_v2"
LOCAL_C32_SIZE = 32
NATIVE_C64_GRID = 64
YAWS = (0, -45, 45, -90, 90, 180)


@dataclass
class RunState:
    """一次实验运行所需的共享状态；只由 helper 创建和消费。"""

    pipeline: Any
    canonical: Mapping[str, Image.Image]
    camera: Mapping[str, float]
    device: torch.device
    out: Path
    started: float


def jsonable(value: Any) -> Any:
    """把 Tensor、NumPy 类型和 Path 转成 JSON 可写的普通对象。"""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """先写临时文件，再原子替换，避免中断时留下半个 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(path: Path, payload: Any) -> None:
    """以临时文件原子保存 torch checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def empty_cuda() -> None:
    """回收 Python 引用并清空 CUDA allocator 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_sparse(path: Path, value: SparseTensor, normalized: bool) -> None:
    """把 SparseTensor 的坐标和特征保存到 CPU checkpoint。"""
    # 只保存当前实验真正需要的张量，不把 pipeline 或模型对象序列化进去。
    atomic_save(
        path,
        {
            "format": FORMAT,
            "coords": value.coords.detach().cpu().int(),
            "features": value.feats.detach().cpu(),
            "normalized": bool(normalized),
        },
    )


def load_payload(path: Path) -> dict[str, Any]:
    """读取本实验格式的 checkpoint，拒绝其它实验留下的缓存。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise RuntimeError(f"refusing checkpoint from another format: {path}")
    return payload


def denormalize_shape(pipeline: Any, features: torch.Tensor) -> torch.Tensor:
    """把 shape flow 的 normalized latent 恢复为 decoder 使用的原始分布。"""
    mean, std = normalization_tensors(
        pipeline.shape_slat_normalization,
        torch.device("cpu"),
    )
    return features.float() * std + mean


def save_canonical_images(
    canonical: Mapping[str, Image.Image], output_dir: Path
) -> None:
    """保存前景预处理后供各阶段使用的 512/1024/4096 三个图像尺度。"""
    for key in ("image_512", "image_1024", "image_4096"):
        if key not in canonical:
            raise KeyError(f"canonical preprocessing did not return {key}")
        canonical[key].save(output_dir / f"{key}.png")


def save_geometry_mesh(mesh: Any, output_dir: Path) -> tuple[Path, Path]:
    """保存无 UV 的 geometry mesh PT 和 GLB。"""
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_pt = output_dir / "geometry_mesh.pt"
    mesh_glb = output_dir / "geometry_mesh.glb"
    # PT 保存 Pixal3D 原生 Mesh；GLB 只写 vertices/faces，不创建 UV 或材质。
    atomic_save(mesh_pt, {"format": FORMAT, "mesh": mesh.cpu()})
    tri = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    tri.export(mesh_glb)
    return mesh_pt, mesh_glb


def _check_coords(coords: torch.Tensor, resolution: int, name: str) -> torch.Tensor:
    coords = torch.as_tensor(coords).int().cpu()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N,4], got {tuple(coords.shape)}")
    if bool((coords[:, 0] != 0).any()):
        raise ValueError(f"{name} must contain batch index zero only")
    xyz = coords[:, 1:]
    if bool((xyz < 0).any()) or bool((xyz >= resolution).any()):
        raise ValueError(f"{name} coordinates must lie in [0,{resolution - 1}]")
    return coords


@torch.no_grad()
def encode_mesh_latent(state, args, mesh, resolution, cache_path):
    """在固定世界坐标 AABB voxelize，再以 encoder posterior mean 编码并归一化。"""
    import o_voxel
    from pixal3d import models

    if args.resume and cache_path.is_file():
        payload = load_payload(cache_path)
        if payload["resolution"] != resolution:
            raise RuntimeError("encoded latent cache resolution mismatch")
        return payload["coords"], payload["features"]
    print(f"[encode] mesh -> voxelize {resolution} -> C{resolution // 16}", flush=True)
    indices, dual, intersections = o_voxel.convert.mesh_to_flexible_dual_grid(
        mesh.vertices.detach().float().cpu(),
        mesh.faces.detach().long().cpu(),
        grid_size=resolution,
        aabb=[[-0.5] * 3, [0.5] * 3],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )
    vertices = SparseTensor(
        dual * resolution - indices,
        torch.cat([torch.zeros_like(indices[:, :1]), indices], 1),
    ).to(state.device)
    intersected = vertices.replace(intersections.to(state.device))
    encoder = models.from_pretrained(str(args.encoder_path)).eval().to(state.device)
    try:
        encoded = encoder(vertices, intersected, sample_posterior=False)
        coords = encoded.coords.int().cpu()
        raw = encoded.feats.float().cpu()
        mean, std = normalization_tensors(
            state.pipeline.shape_slat_normalization, raw.device
        )
        normalized = (raw - mean) / std
        if not torch.isfinite(normalized).all():
            raise RuntimeError("encoder returned nonfinite latent")
        _check_coords(coords, resolution // 16, "encoded latent")
        atomic_save(
            cache_path,
            dict(
                format=FORMAT,
                coords=coords,
                features=normalized,
                resolution=resolution,
                normalized=True,
                encoder=str(args.encoder_path),
                voxel_tokens=len(indices),
            ),
        )
    finally:
        encoder.cpu()
    del encoder, encoded, vertices, intersected
    empty_cuda()
    return coords, normalized


def normalization_tensors(
    spec: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 shape flow 的训练归一化参数，并整理成可广播形状。"""
    mean = torch.as_tensor(spec["mean"], dtype=torch.float32, device=device)[None]
    std = torch.as_tensor(spec["std"], dtype=torch.float32, device=device)[None]
    return mean, std


def upsample_c32_to_c64(pipeline: Any, slat: SparseTensor) -> torch.Tensor:
    """把 C32 shape latent 解码成 C64 support，保留 batch 列。

    decoder 首先把 C32 latent support 上采样到 C512 坐标，再严格使用 baseline
    的端点量化规则 ``round((x+0.5)/512*63)`` 得到 C64 坐标。这个函数不负责
    生成 latent 特征，只生成下一阶段需要的稀疏 support。
    """
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    candidates = decoder.upsample(slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    if candidates.shape[0] == 0:
        raise RuntimeError("shape decoder upsample produced no support candidates")

    scaled = (candidates[:, 1:].float() + 0.5) / 512.0
    xyz = (scaled * 63).round().int()
    coords = torch.cat((candidates[:, :1].int(), xyz), dim=1).unique(dim=0)
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= 64)).any()):
        raise RuntimeError("C32 decoder support escapes the C64 grid")
    return coords


@torch.no_grad()
def run_native_prefix(state: RunState, args: Any) -> torch.Tensor:
    """从 image_512 重新生成 SS C32、Shape512 和 native C64 support。"""
    pipeline = state.pipeline
    prefix_coords_path = state.out / "prefix" / "native_c64_support_coords.pt"
    prefix_shape_path = state.out / "prefix" / "native_shape_c32_denormalized.pt"
    if args.resume and prefix_coords_path.is_file():
        payload = load_payload(prefix_coords_path)
        coords_c64 = payload["coords"].int()
        if not (state.out / "baseline/shape_c64_shape1024_denormalized.pt").exists():
            # Resume at exactly the same RNG position as a continuous baseline run.
            torch.set_rng_state(payload["cpu_rng_state"])
            torch.cuda.set_rng_state(payload["cuda_rng_state"], state.device)
        print(
            f"[prefix] reuse current-run native C64 support ({coords_c64.shape[0]:,} tokens)",
            flush=True,
        )
        return coords_c64

    camera = state.camera
    canonical = state.canonical
    print("[prefix 1/3] image_512 -> Sparse Structure C32", flush=True)
    cond_ss = pipeline.get_proj_cond_ss(
        [canonical["image_512"]],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
    )
    coords_c32 = pipeline.sample_sparse_structure(
        cond_ss,
        LOCAL_C32_SIZE,
        sampler_params={"steps": int(args.ss_steps)},
    )
    del cond_ss
    empty_cuda()

    print("[prefix 2/3] C32 -> Shape512 flow", flush=True)
    cond_shape_c32 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [canonical["image_512"]],
        coords_c32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
    )
    shape_c32 = pipeline.sample_shape_slat(
        cond_shape_c32,
        pipeline.models["shape_slat_flow_model_512"],
        coords_c32,
        sampler_params={"steps": int(args.shape_steps)},
    )
    save_sparse(prefix_shape_path, shape_c32, normalized=False)
    del cond_shape_c32, coords_c32
    empty_cuda()

    print("[prefix 3/3] decoder upsample -> native global C64 support", flush=True)
    coords_c64 = upsample_c32_to_c64(pipeline, shape_c32).cpu()
    atomic_save(
        prefix_coords_path,
        {
            "format": FORMAT,
            "coords": coords_c64,
            "source": "fresh image_512 SS/Shape512 prefix",
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(state.device),
        },
    )
    del shape_c32
    empty_cuda()
    return coords_c64


@torch.no_grad()
def run_native_shape1024(
    state: RunState,
    args: Any,
    coords_c64: torch.Tensor,
) -> SparseTensor:
    """按原生 baseline 的 Shape1024 运行 C64，返回反归一化 C64 latent。"""
    pipeline = state.pipeline
    camera = state.camera
    image_1024 = state.canonical["image_1024"]
    denormalized_path = state.out / "baseline" / "shape_c64_shape1024_denormalized.pt"

    if args.resume and denormalized_path.is_file():
        payload = load_payload(denormalized_path)
        cached_coords = payload["coords"].int()
        if not torch.equal(cached_coords, coords_c64.int()):
            raise RuntimeError("cached baseline Shape1024 C64 coordinates do not match")
        print(
            f"[baseline Shape1024] reuse current-run C64 latent "
            f"({cached_coords.shape[0]:,} tokens)",
            flush=True,
        )
        return SparseTensor(
            payload["features"].to(pipeline.device),
            cached_coords.to(pipeline.device),
        )

    print("[baseline Shape1024] project native C64 points from image_1024", flush=True)
    cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        coords_c64.to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=NATIVE_C64_GRID,
    )
    print("[baseline Shape1024] native global C64 -> Shape1024 flow", flush=True)
    shape_c64 = pipeline.sample_shape_slat(
        cond,
        pipeline.models["shape_slat_flow_model_1024"],
        coords_c64.to(pipeline.device),
        sampler_params={"steps": int(args.shape_steps)},
    )
    save_sparse(denormalized_path, shape_c64, normalized=False)
    del cond
    empty_cuda()
    return shape_c64


def save(path, **data):
    atomic_save(path, dict(format=FORMAT, **data))


def js(path, data):
    atomic_json(path, data)


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def setup(out):
    from inference import init_pipeline

    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    out.mkdir(parents=True, exist_ok=True)
    return init_pipeline(device="cuda:0", low_vram=True)
