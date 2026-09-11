#!/usr/bin/env python3
"""C2048 geometry experiment: local C32 -> C512 -> C64 re-flow per block.

The input is the fixed C128 shape endpoint produced by the C2048 experiment.
For each disjoint global C64 block this driver:

1. pools the endpoint to a local C32 support and re-noises its features at
   timestep ``n``;
2. runs the native Shape512 flow on local coordinates, then uses the native
   decoder upsampler and round-to-(grid-1) quantization to create local C64;
3. samples the endpoint features onto that new C64 support, re-noises again,
   and runs the native Shape1024 flow locally;
4. maps the local C64 coordinates back to global C128 and decodes one global
   2048 shape mesh.

The image condition is intentionally crop-local at both shape stages.  The
crop is a fixed 1024 square cut from the 2048 resize of the canonical 4096
image.  Its box is centered on the projected endpoint C128 points belonging
to the block.  The condition extractor receives global C128 coordinates so
projection lands at the correct image location; only the sparse Flow state
uses 0..31 or 0..63 local coordinates.

This is geometry-only.  Texture is neither sampled nor decoded.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_CUDA", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_EXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import pixal3d_global_c256_cube_owner_flow_singleview as cubes
import pixal3d_global4096_tile_endpoint_rollout_sync as legacy
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as camera_core
import pixal3d.models as models
from inference import IMAGE_COND_CONFIGS, build_image_cond_model
from pixal3d.pipelines import Pixal3DImageTo3DPipeline, samplers
from pixal3d.modules.sparse import SparseTensor


ROOT = Path(__file__).resolve().parent
FORMAT = "pixal3d_c128_block_local_cascade_shape_renoise_2048_v1"
GLOBAL_GRID = 128
BLOCK_GRID = 64
LOCAL_C32 = 32
LOCAL_C64 = 64
BLOCK_STARTS = (0, 64)
CANONICAL_4096 = 4096
CONDITION_IMAGE_2048 = 2048
CROP_SIZE = 1024
DINO_PATCH = 16
STEPS = 12
MODES = ("conditional", "unconditional", "visibility_gated")
DEFAULT_MODEL_PATH = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_BASELINE = ROOT / "outputs/baseline1024_raw_ovoxel_cuda4_0_img"
DEFAULT_ENDPOINT = ROOT / "outputs/c128_baseline_endpoint_renoise_uncond_2048_cuda4"
DEFAULT_VISIBILITY = ROOT / "outputs/c128_baseline_visibility/visibility.pt"
DEFAULT_OUTPUT = ROOT / "outputs/c128_block_local_cascade_shape_renoise_2048_cuda4"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def denormalize(features: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(spec["mean"], dtype=features.dtype)[None]
    std = torch.as_tensor(spec["std"], dtype=features.dtype)[None]
    return features * std + mean


def normalize(features: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(spec["mean"], dtype=features.dtype)[None]
    std = torch.as_tensor(spec["std"], dtype=features.dtype)[None]
    return (features - mean) / std


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or any(item < 1 or item > STEPS for item in result):
        raise ValueError(f"n-values must be a comma-separated subset of 1..{STEPS}")
    return result


def parse_window_starts(value: str) -> tuple[int, ...]:
    starts = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not starts:
        raise ValueError("window-starts must contain at least one start")
    if starts[0] != 0 or starts[-1] != GLOBAL_GRID - BLOCK_GRID:
        raise ValueError(
            f"window-starts must cover both global edges 0 and {GLOBAL_GRID - BLOCK_GRID}"
        )
    if any(start < 0 or start > GLOBAL_GRID - BLOCK_GRID for start in starts):
        raise ValueError("window-starts contains a start outside C128")
    return starts


def ownership_bounds(starts: Sequence[int]) -> tuple[int, ...]:
    """Build disjoint global ownership intervals inside overlapping windows."""
    starts = tuple(int(value) for value in starts)
    if not starts or starts[0] != 0 or starts[-1] != GLOBAL_GRID - BLOCK_GRID:
        raise ValueError("window starts must include both global edges")
    bounds = [0]
    for left, right in zip(starts[:-1], starts[1:]):
        overlap_lo = max(left, right)
        overlap_hi = min(left + BLOCK_GRID, right + BLOCK_GRID)
        boundary = right if overlap_hi <= overlap_lo else int(round((overlap_lo + overlap_hi) * 0.5))
        if boundary <= bounds[-1] or boundary >= GLOBAL_GRID:
            raise ValueError(f"window starts do not form valid ownership intervals: {starts}")
        bounds.append(boundary)
    bounds.append(GLOBAL_GRID)
    return tuple(bounds)


def cube_records(
    coords: torch.Tensor,
    starts: Sequence[int] = BLOCK_STARTS,
    require_partition: bool = True,
) -> list[dict[str, Any]]:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("C128 coords must have shape [N,4]")
    xyz = coords[:, 1:].int()
    records: list[dict[str, Any]] = []
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    window_starts = tuple(int(value) for value in starts)
    if not window_starts:
        raise ValueError("cube layout has no window starts")
    cube_id = 0
    for sx in window_starts:
        for sy in window_starts:
            for sz in window_starts:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                inside = ((xyz >= start) & (xyz < start + BLOCK_GRID)).all(1)
                rows = torch.where(inside)[0].long()
                if not rows.numel():
                    cube_id += 1
                    continue
                local_xyz = xyz.index_select(0, rows) - start
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": (sx, sy, sz),
                        "global_row_ids": rows,
                        "local_xyz": local_xyz,
                    }
                )
                coverage.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
                cube_id += 1
    if require_partition and not torch.all(coverage == 1):
        raise RuntimeError("C128 support is not partitioned exactly once by the disjoint C64 layout")
    return records


def projected_crop(
    global_rows: torch.Tensor,
    global_coords: torch.Tensor,
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a fixed 1024 crop around projected endpoint support points."""
    xyz = global_coords.index_select(0, global_rows)[:, 1:].float()
    q = 2.0 * (xyz + 0.5) / float(GLOBAL_GRID) - 1.0
    uv, depth, finite = camera_core._project_global_q_to_image(
        q,
        global_camera=camera,
        image_width=CONDITION_IMAGE_2048,
        image_height=CONDITION_IMAGE_2048,
    )
    finite = finite & torch.isfinite(uv).all(1)
    if not bool(finite.any()):
        raise RuntimeError("a C64 block has no finite projected support points")
    points = uv[finite].double()
    raw_lo = points.amin(0)
    raw_hi = points.amax(0)
    lo = raw_lo.clamp(0.0, float(CONDITION_IMAGE_2048))
    hi = raw_hi.clamp(0.0, float(CONDITION_IMAGE_2048))
    if bool((hi <= lo).any()):
        raise RuntimeError("projected endpoint support has no image intersection")
    extent = hi - lo
    if float(extent.max()) > CROP_SIZE + 1e-4:
        raise RuntimeError(
            f"projected support extent {extent.tolist()} exceeds fixed crop {CROP_SIZE}"
        )
    center = (lo + hi) * 0.5
    starts: list[int] = []
    for axis in range(2):
        lower = max(0, int(math.ceil(float(hi[axis]))) - CROP_SIZE)
        upper = min(
            int(math.floor(float(lo[axis]))),
            CONDITION_IMAGE_2048 - CROP_SIZE,
        )
        if lower > upper:
            raise RuntimeError("cannot place fixed crop around projected support")
        preferred = int(round(float(center[axis]) - CROP_SIZE / 2.0))
        starts.append(min(max(preferred, lower), upper))
    x0, y0 = starts
    box = (x0, y0, x0 + CROP_SIZE, y0 + CROP_SIZE)
    return {
        "crop_box_2048": list(box),
        "projection_crop_box": [
            x0 / CONDITION_IMAGE_2048,
            y0 / CONDITION_IMAGE_2048,
            (x0 + CROP_SIZE) / CONDITION_IMAGE_2048,
            (y0 + CROP_SIZE) / CONDITION_IMAGE_2048,
        ],
        "raw_bbox_pixel_edges_2048": [*raw_lo.tolist(), *raw_hi.tolist()],
        "clipped_bbox_pixel_edges_2048": [*lo.tolist(), *hi.tolist()],
        "projected_extent": extent.tolist(),
        "finite_projected_tokens": int(finite.sum()),
        "support_tokens": int(global_rows.numel()),
        "depth_range": [float(depth[finite].min()), float(depth[finite].max())],
        "size": [CROP_SIZE, CROP_SIZE],
        "alignment": DINO_PATCH,
    }


def save_crops(
    image_2048: Image.Image,
    records: Sequence[Mapping[str, Any]],
    global_coords: torch.Tensor,
    camera: Mapping[str, Any],
    output: Path,
) -> None:
    crop_dir = output / "inputs" / "crops_1024_from_2048"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        crop = projected_crop(record["global_row_ids"], global_coords, camera)
        record["crop"] = crop  # type: ignore[index]
        box = tuple(int(item) for item in crop["crop_box_2048"])
        image_2048.crop(box).convert("RGB").save(
            crop_dir / f"cube_{int(record['cube_id']):02d}.png"
        )
    atomic_json(
        output / "support" / "crop_layout.json",
        {
            "format": FORMAT,
            "source": "canonical_4096 resized to 2048 before cropping",
            "source_size": [CONDITION_IMAGE_2048, CONDITION_IMAGE_2048],
            "crop_size": [CROP_SIZE, CROP_SIZE],
            "cubes": [
                {
                    "cube_id": int(record["cube_id"]),
                    "start": list(record["start"]),
                    "tokens": int(record["global_row_ids"].numel()),
                    "crop": record["crop"],
                }
                for record in records
            ],
        },
    )


def pool_c128_to_c32(
    records: Sequence[Mapping[str, Any]],
    global_coords: torch.Tensor,
    clean_norm: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Create one fixed local C32 support and pooled E12 features per block."""
    stage_records: list[dict[str, Any]] = []
    clean_parts: list[torch.Tensor] = []
    offset = 0
    for record in records:
        rows = record["global_row_ids"].long()
        local64 = global_coords.index_select(0, rows)[:, 1:].int() - torch.tensor(
            record["start"], dtype=torch.int32
        )
        local32, inverse = torch.unique(
            torch.div(local64, 2, rounding_mode="floor"),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        pooled = torch.zeros(
            (local32.shape[0], clean_norm.shape[1]), dtype=clean_norm.dtype
        )
        pooled.index_add_(0, inverse, clean_norm.index_select(0, rows))
        counts = torch.bincount(inverse, minlength=local32.shape[0]).to(pooled.dtype)
        pooled /= counts[:, None].clamp_min(1.0)

        # For projection, use an actual endpoint C128 token nearest to each
        # pooled cell centre.  This keeps the image location tied to E12 while
        # the Flow state itself uses local 0..31 coordinates.
        representatives: list[int] = []
        for cell in local32:
            member = torch.where(inverse == int(len(representatives)))[0]
            centre = cell.float() * 2.0 + 0.5
            distances = (local64.index_select(0, member).float() - centre).square().sum(1)
            representatives.append(int(member[int(distances.argmin())]))
        representative_rows = rows[torch.as_tensor(representatives, dtype=torch.long)]
        stage_rows = torch.arange(offset, offset + local32.shape[0], dtype=torch.long)
        stage_records.append(
            {
                "cube_id": int(record["cube_id"]),
                "start": tuple(int(item) for item in record["start"]),
                "global_row_ids": stage_rows,
                "local_xyz": local32.int(),
                "projection_coords": global_coords.index_select(0, representative_rows).int(),
                "source_c128_rows": representative_rows,
                "owned_row_ids": stage_rows,
            }
        )
        clean_parts.append(pooled)
        offset += int(local32.shape[0])
    return stage_records, torch.cat(clean_parts, 0)


def sort_xyz(coords: torch.Tensor) -> torch.Tensor:
    if not coords.numel():
        return coords.reshape(0, 3).int()
    order = np.lexsort(
        (
            coords[:, 2].cpu().numpy(),
            coords[:, 1].cpu().numpy(),
            coords[:, 0].cpu().numpy(),
        )
    )
    return coords.index_select(
        0, torch.from_numpy(order).to(device=coords.device).long()
    ).int()


@torch.no_grad()
def local_c32_to_c64(
    pipeline: Any,
    stage32_records: Sequence[Mapping[str, Any]],
    state32_norm: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Decode batched local C32 supports and quantize them to local C64."""
    values: list[SparseTensor] = []
    active = [record for record in stage32_records if record["global_row_ids"].numel()]
    if not active:
        raise RuntimeError("all local C32 blocks are empty")
    mean_std = pipeline.shape_slat_normalization
    for record in active:
        rows = record["global_row_ids"].long()
        coords = torch.cat(
            (
                torch.zeros((rows.numel(), 1), dtype=torch.int32),
                record["local_xyz"].int(),
            ),
            dim=1,
        )
        raw = denormalize(state32_norm.index_select(0, rows), mean_std)
        values.append(SparseTensor(raw, coords))
    packed = legacy._pack_sparse_batch(values, "local Shape512 decoder input").to(device)
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    candidates = decoder.upsample(packed, upsample_times=4)
    decoder.cpu()
    decoder.low_vram = False
    parts: list[torch.Tensor] = []
    for batch_id in range(len(active)):
        part = candidates[candidates[:, 0] == batch_id]
        if not part.numel():
            raise RuntimeError(f"local C32 decoder returned empty block {batch_id}")
        local = ((part[:, 1:].float() + 0.5) / 512.0 * (LOCAL_C64 - 1)).round().int()
        local = sort_xyz(local.unique(dim=0)).cpu()
        if bool(((local < 0) | (local >= LOCAL_C64)).any()):
            raise RuntimeError("local Shape512 decoder support escaped C64")
        parts.append(local)
    del packed, candidates, values
    empty_cuda()

    result: list[dict[str, Any]] = []
    offset = 0
    for record, local in zip(active, parts):
        rows = torch.arange(offset, offset + local.shape[0], dtype=torch.long)
        start = torch.tensor(record["start"], dtype=torch.int32)
        global_xyz = local + start
        global_coords = torch.cat(
            (torch.zeros((local.shape[0], 1), dtype=torch.int32), global_xyz), dim=1
        )
        result.append(
            {
                "cube_id": int(record["cube_id"]),
                "start": tuple(int(item) for item in record["start"]),
                "global_row_ids": rows,
                "local_xyz": local,
                "projection_coords": global_coords,
                "owned_row_ids": rows,
            }
        )
        offset += int(local.shape[0])
    return result


def nearest_endpoint_features(
    stage64_records: Sequence[Mapping[str, Any]],
    original_records: Sequence[Mapping[str, Any]],
    global_coords: torch.Tensor,
    clean_norm: torch.Tensor,
) -> torch.Tensor:
    """Transfer E12 features to newly generated local C64 support by 3-D NN."""
    original_by_id = {int(record["cube_id"]): record for record in original_records}
    parts: list[torch.Tensor] = []
    for record in stage64_records:
        source = original_by_id[int(record["cube_id"])]
        source_rows = source["global_row_ids"].long()
        source_xyz = global_coords.index_select(0, source_rows)[:, 1:].numpy()
        target_xyz = (
            record["local_xyz"].int()
            + torch.tensor(record["start"], dtype=torch.int32)
        ).numpy()
        tree = cKDTree(source_xyz.astype(np.float32, copy=False))
        _, nearest = tree.query(target_xyz.astype(np.float32, copy=False), k=1)
        nearest_t = torch.from_numpy(np.asarray(nearest, dtype=np.int64))
        parts.append(clean_norm.index_select(0, source_rows[nearest_t]))
    return torch.cat(parts, 0)


def zero_condition(
    model: Any,
    record: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    image_model = model
    embed_dim = int(getattr(image_model, "embed_dim", 1024))
    registers = int(getattr(getattr(image_model, "model", None), "config", None).num_register_tokens)
    proj_channels = int(getattr(image_model, "proj_channels", embed_dim * 2))
    count = int(record["global_row_ids"].numel())
    return {
        "global_row_ids": record["global_row_ids"].clone(),
        "global": torch.zeros((1, 1 + registers, embed_dim), dtype=torch.float32),
        "proj": torch.zeros((count, proj_channels), dtype=torch.float32),
    }


@torch.no_grad()
def extract_conditions(
    pipeline: Any,
    image_2048: Image.Image,
    camera: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    stage: str,
    enabled: Mapping[int, bool],
    output: Path,
) -> dict[str, Any]:
    """Extract crop global/projection features and create zeros for gated rows."""
    if stage == "shape512":
        model = pipeline.image_cond_model_shape_512
    elif stage == "shape1024":
        model = pipeline.image_cond_model_shape_1024
    else:
        raise ValueError(stage)
    cache_dir = output / "conditions" / stage
    result: dict[int, dict[str, torch.Tensor]] = {}
    low_vram = bool(pipeline.low_vram)
    if low_vram:
        model.to(pipeline.device)
        pipeline.low_vram = False
    try:
        for record in records:
            cube_id = int(record["cube_id"])
            if not record["global_row_ids"].numel():
                continue
            cache = cache_dir / f"cube_{cube_id:02d}.pt"
            if bool(enabled.get(cube_id, False)):
                projection_coords = record["projection_coords"].int()
                crop_info = record.get("crop")
                if crop_info is None:
                    raise RuntimeError(f"cube {cube_id} is missing fixed crop metadata")
                box = tuple(int(item) for item in crop_info["crop_box_2048"])
                crop = image_2048.crop(box).convert("RGB")
                if crop.size != (CROP_SIZE, CROP_SIZE):
                    raise RuntimeError(f"cube {cube_id} crop has size {crop.size}")
                condition = pipeline.get_proj_cond_shape(
                    model,
                    [crop],
                    projection_coords.to(pipeline.device),
                    camera_angle_x=float(camera["camera_angle_x"]),
                    distance=float(camera["distance"]),
                    mesh_scale=float(camera.get("mesh_scale", 1.0)),
                    grid_resolution_override=GLOBAL_GRID,
                    projection_crop_box=crop_info["projection_crop_box"],
                    preserve_image_resolution=True,
                )["cond"]
                payload = {
                    "global_row_ids": record["global_row_ids"].clone(),
                    "global": condition["global"].detach().cpu().contiguous(),
                    "proj": condition["proj"].feats.detach().cpu().contiguous(),
                    "source": "cube support projected to 2048 resized-image 1024 crop",
                    "stage": stage,
                    "cube_id": cube_id,
                }
                atomic_save(cache, payload)
                result[cube_id] = payload
                del condition, crop
                print(
                    f"[condition {stage}] cube={cube_id:02d} "
                    f"tokens={len(record['global_row_ids']):,} crop={CROP_SIZE}",
                    flush=True,
                )
            else:
                payload = zero_condition(model, record)
                result[cube_id] = payload
            empty_cuda()
    finally:
        if low_vram:
            pipeline.low_vram = True
            model.cpu()
        empty_cuda()
    return {"cubes": result, "stage": stage}


def route_condition(
    conditions: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    enabled: Mapping[int, bool],
) -> dict[str, Any]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    by_id = conditions["cubes"]
    for record in records:
        cube_id = int(record["cube_id"])
        payload = by_id[cube_id]
        if bool(enabled.get(cube_id, False)):
            result[cube_id] = payload
        else:
            result[cube_id] = {
                "global_row_ids": payload["global_row_ids"],
                "global": torch.zeros_like(payload["global"]),
                "proj": torch.zeros_like(payload["proj"]),
            }
    return {"cubes": result}


def re_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    t: float,
    sigma_min: float,
) -> torch.Tensor:
    weight = float(sigma_min) + (1.0 - float(sigma_min)) * float(t)
    return (1.0 - float(t)) * clean + weight * noise


def seeded_noise(shape: Sequence[int], seed: int) -> torch.Tensor:
    return torch.randn(
        tuple(int(item) for item in shape),
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        dtype=torch.float32,
    )


def init_shape_pipeline(model_path: Path, device: torch.device) -> Any:
    """Load only the models needed by this geometry-only experiment."""
    config = json.loads((model_path / "pipeline.json").read_text(encoding="utf-8"))["args"]
    loaded: dict[str, Any] = {}
    for name in (
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ):
        print(f"[model] loading {name}", flush=True)
        loaded[name] = models.from_pretrained(
            str(model_path / config["models"][name])
        ).eval()
    sampler_config = config["shape_slat_sampler"]
    pipeline = Pixal3DImageTo3DPipeline(
        models=loaded,
        shape_slat_sampler=getattr(samplers, sampler_config["name"])(
            **sampler_config["args"]
        ),
        shape_slat_sampler_params=dict(sampler_config["params"]),
        shape_slat_normalization=config["shape_slat_normalization"],
        low_vram=True,
    )
    pipeline._device = device
    print("[model] loading shape512 image condition", flush=True)
    pipeline.image_cond_model_shape_512 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_512"]
    )
    print("[model] loading shape1024 image condition", flush=True)
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_1024"]
    )
    for model in (
        pipeline.image_cond_model_shape_512,
        pipeline.image_cond_model_shape_1024,
    ):
        if getattr(model, "use_naf_upsample", False):
            model._load_naf()
    return pipeline


@torch.no_grad()
def run_suffix(
    pipeline: Any,
    records: Sequence[Mapping[str, Any]],
    state: torch.Tensor,
    conditions: Mapping[str, Any],
    model_key: str,
    n: int,
    schedule: Sequence[float],
    mode: str,
    device: torch.device,
    max_batch_tokens: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run only steps n..11 using independent local sparse batches."""
    if n >= STEPS:
        return state.clone(), []
    model = pipeline.models[model_key].to(device).eval()
    sampler = pipeline.shape_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params)
    params["guidance_strength"] = 0.0 if mode == "unconditional" else float(
        params.get("guidance_strength", 7.5)
    )
    # The condition is already selected cube-by-cube.  The native sampler's
    # interval/rescale controls remain active for conditional and gated paths.
    groups = cubes.pack_groups(
        records,
        flow_batch_size=8,
        max_batch_tokens=int(max_batch_tokens),
        require_owned=True,
    )
    if not groups:
        raise RuntimeError(f"{model_key}: no non-empty local groups")
    state = state.clone()
    timings: list[dict[str, Any]] = []
    for step in range(n, STEPS):
        t, t_next = float(schedule[step]), float(schedule[step + 1])
        started = time.perf_counter()
        proposals: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for group in groups:
            values, timing = cubes._one_prediction(
                group,
                state,
                conditions,
                sampler,
                model,
                params,
                t,
                t_next,
                device,
            )
            proposals.extend(
                (int(record["cube_id"]), record["global_row_ids"], value)
                for record, value in zip(group, values)
            )
        # Rows are disjoint within every local stage; update from the same
        # pre-step state after all block predictions have been computed.
        for _, rows, velocity in proposals:
            state.index_copy_(
                0,
                rows,
                state.index_select(0, rows) - float(t - t_next) * velocity,
            )
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"non-finite {model_key} state at step {step}")
        timings.append(
            {
                "step": step + 1,
                "t": t,
                "t_next": t_next,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[flow {mode} {model_key} n={n:02d}] step={step + 1:02d}/12 "
            f"tokens={len(state):,}",
            flush=True,
        )
    model.cpu()
    empty_cuda()
    return state, timings


def attach_crops(records: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]]) -> None:
    by_id = {int(record["cube_id"]): record for record in source}
    for record in records:
        record["crop"] = by_id[int(record["cube_id"])] ["crop"]  # type: ignore[index]


def assemble_global(
    records: Sequence[Mapping[str, Any]],
    state_norm: torch.Tensor,
    shape_spec: Mapping[str, Any],
    window_starts: Sequence[int] = BLOCK_STARTS,
    owner_bounds: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one global support, optionally retaining only a halo-safe owner core."""
    starts = tuple(int(value) for value in window_starts)
    bounds = tuple(int(value) for value in (owner_bounds or ownership_bounds(starts)))
    if len(bounds) != len(starts) + 1:
        raise ValueError("owner bounds must have one interval per window start")
    start_to_index = {start: index for index, start in enumerate(starts)}
    coords: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    for record in records:
        local = record["local_xyz"].int()
        start = torch.tensor(record["start"], dtype=torch.int32)
        global_xyz = local + start
        start_indices = [start_to_index[int(value)] for value in record["start"]]
        owned = torch.ones(local.shape[0], dtype=torch.bool)
        for axis, start_index in enumerate(start_indices):
            owned &= (
                (global_xyz[:, axis] >= bounds[start_index])
                & (global_xyz[:, axis] < bounds[start_index + 1])
            )
        if not bool(owned.any()):
            continue
        local = global_xyz[owned]
        rows = record["global_row_ids"].long()[owned]
        coords.append(torch.cat((torch.zeros((local.shape[0], 1), dtype=torch.int32), local), 1))
        features.append(state_norm.index_select(0, rows))
    if not coords:
        raise RuntimeError("ownership intervals removed every local C64 token")
    coords_cat = torch.cat(coords, 0)
    features_cat = denormalize(torch.cat(features, 0), shape_spec)
    keys = (coords_cat[:, 1:].long()[:, 0] * GLOBAL_GRID + coords_cat[:, 1:].long()[:, 1]) * GLOBAL_GRID + coords_cat[:, 1:].long()[:, 2]
    if torch.unique(keys).numel() != keys.numel():
        raise RuntimeError("local block assembly produced duplicate global C128 coordinates")
    return coords_cat, features_cat


def boundary_shell_mask(
    coords: torch.Tensor,
    width: int,
    planes: Sequence[int] = (64,),
) -> torch.Tensor:
    """Select a shell around the decoder ownership planes."""
    if int(width) <= 0:
        return torch.zeros(coords.shape[0], dtype=torch.bool)
    xyz = coords[:, 1:].int()
    mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    for plane in planes:
        mask |= (
            ((xyz[:, 0] - int(plane)).abs() < int(width))
            | ((xyz[:, 1] - int(plane)).abs() < int(width))
            | ((xyz[:, 2] - int(plane)).abs() < int(width))
        )
    return mask


def add_boundary_anchors(
    local_coords: torch.Tensor,
    local_raw: torch.Tensor,
    baseline_coords: torch.Tensor,
    baseline_raw: torch.Tensor,
    width: int,
    feature_policy: str,
    planes: Sequence[int] = (64,),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Add baseline E12 support at split planes while retaining local interiors.

    The local Flow state is allowed to change inside each block.  The global
    decoder still needs a continuous latent neighborhood at x/y/z=64.  We
    therefore use the baseline endpoint as a boundary scaffold only.  On
    common shell coordinates, ``baseline`` selects the coherent baseline
    feature; ``fresh`` leaves the locally flowed feature untouched.
    """
    if feature_policy not in ("fresh", "baseline"):
        raise ValueError(f"unsupported boundary feature policy: {feature_policy}")
    if int(width) <= 0:
        return local_coords, local_raw, {
            "enabled": False,
            "width": int(width),
            "planes": [int(plane) for plane in planes],
            "feature_policy": feature_policy,
            "fresh_tokens": int(local_coords.shape[0]),
            "baseline_tokens": int(baseline_coords.shape[0]),
            "decoder_tokens": int(local_coords.shape[0]),
            "new_baseline_tokens": 0,
            "common_shell_features_from_baseline": 0,
        }
    local_keys = (
        (local_coords[:, 1:].long()[:, 0] * GLOBAL_GRID + local_coords[:, 1:].long()[:, 1])
        * GLOBAL_GRID
        + local_coords[:, 1:].long()[:, 2]
    )
    baseline_keys = (
        (baseline_coords[:, 1:].long()[:, 0] * GLOBAL_GRID + baseline_coords[:, 1:].long()[:, 1])
        * GLOBAL_GRID
        + baseline_coords[:, 1:].long()[:, 2]
    )
    local_rows = {int(value): index for index, value in enumerate(local_keys.tolist())}
    baseline_rows = {int(value): index for index, value in enumerate(baseline_keys.tolist())}
    selected = boundary_shell_mask(baseline_coords, int(width), planes)
    selected_keys = {
        int(value)
        for value, enabled in zip(baseline_keys.tolist(), selected.tolist())
        if enabled
    }
    union_keys = sorted(set(local_rows) | selected_keys)
    coords_parts: list[torch.Tensor] = []
    feature_parts: list[torch.Tensor] = []
    new_baseline = 0
    common_baseline = 0
    for value in union_keys:
        local_row = local_rows.get(value)
        baseline_row = baseline_rows.get(value)
        if local_row is not None:
            coords_parts.append(local_coords[local_row])
            if (
                feature_policy == "baseline"
                and baseline_row is not None
                and value in selected_keys
            ):
                feature_parts.append(baseline_raw[baseline_row])
                common_baseline += 1
            else:
                feature_parts.append(local_raw[local_row])
        else:
            if baseline_row is None:
                raise RuntimeError("boundary anchor key is missing from baseline support")
            coords_parts.append(baseline_coords[baseline_row])
            feature_parts.append(baseline_raw[baseline_row])
            new_baseline += 1
    coords = torch.stack(coords_parts, 0).int()
    raw = torch.stack(feature_parts, 0).float()
    if torch.unique(
        (coords[:, 1:].long()[:, 0] * GLOBAL_GRID + coords[:, 1:].long()[:, 1])
        * GLOBAL_GRID
        + coords[:, 1:].long()[:, 2]
    ).numel() != coords.shape[0]:
        raise RuntimeError("boundary-anchored decoder support contains duplicates")
    return coords, raw, {
        "enabled": True,
        "width": int(width),
        "planes": [int(plane) for plane in planes],
        "feature_policy": feature_policy,
        "fresh_tokens": int(local_coords.shape[0]),
        "baseline_tokens": int(baseline_coords.shape[0]),
        "decoder_tokens": int(coords.shape[0]),
        "new_baseline_tokens": int(new_baseline),
        "common_shell_features_from_baseline": int(common_baseline),
    }


def make_sheet(rows: Sequence[Mapping[str, Any]], path: Path, title: str) -> None:
    if not rows:
        return
    panel = 512
    header = 46
    gap = 12
    cols = min(4, len(rows))
    rows_count = (len(rows) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (cols * panel + (cols + 1) * gap, rows_count * (panel + header) + (rows_count + 1) * gap),
        (22, 24, 29),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for index, row in enumerate(rows):
        image_path = Path(row["front_normal"])
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
        x = gap + (index % cols) * (panel + gap)
        y = gap + (index // cols) * (panel + header + gap)
        canvas.paste(image, (x, y + header))
        draw.text((x + 8, y + 10), f"n={row['n']:02d} | {row['mode']}", fill="white", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def normal_metrics(prediction: Path, reference: Path, mask_path: Path) -> dict[str, float]:
    """Compare two encoded camera-normal renders pixel by pixel.

    The caller must record what ``reference`` represents.  In this experiment
    it is a raw baseline render, so these values are regression/proxy metrics,
    never ground-truth geometry metrics.
    """
    from skimage.metrics import structural_similarity

    pred = np.asarray(Image.open(prediction).convert("RGB").resize((1024, 1024)), dtype=np.float32) / 255.0
    ref = np.asarray(Image.open(reference).convert("RGB").resize((1024, 1024)), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L").resize((1024, 1024), Image.Resampling.NEAREST)) > 127
    diff = pred - ref
    _, ssim_map = structural_similarity(ref, pred, channel_axis=2, data_range=1.0, full=True)
    fg = diff[mask]
    mse = float(np.mean(diff * diff))
    fg_mse = float(np.mean(fg * fg)) if fg.size else mse
    return {
        "normal_psnr_db": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
        "normal_foreground_psnr_db": float(10.0 * math.log10(1.0 / max(fg_mse, 1e-12))),
        "normal_ssim": float(np.mean(ssim_map)),
        "normal_foreground_ssim": float(np.mean(ssim_map[mask])) if mask.any() else float(np.mean(ssim_map)),
        "normal_mae": float(np.mean(np.abs(diff))),
        "normal_foreground_mae": float(np.mean(np.abs(fg))) if fg.size else float(np.mean(np.abs(diff))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("assets/choose/0_img.png"))
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=ROOT / "outputs/cascade512_1024_tiled2048_square_crop_cuda4/inputs",
        help="reuse image_4096/image_1024/foreground_mask_4096 when present",
    )
    parser.add_argument("--camera", type=Path, default=DEFAULT_BASELINE / "global_camera.json")
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--visibility", type=Path, default=DEFAULT_VISIBILITY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=44)
    parser.add_argument("--n-values", default="1,4,8,12")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument(
        "--feature-init",
        choices=("endpoint", "fresh"),
        default="endpoint",
        help="endpoint=re-noise E12 and run suffix; fresh=complete native local flow from noise",
    )
    parser.add_argument(
        "--shape1024-feature-init",
        choices=("same", "fresh"),
        default="same",
        help="use the feature-init policy at Shape1024, or restart Shape1024 from fresh noise",
    )
    parser.add_argument("--max-batch-tokens", type=int, default=60_000)
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200_000)
    parser.add_argument(
        "--window-starts",
        default="0,64",
        help="C64 window starts per axis; use 0,48,64 for a halo-overlap layout",
    )
    parser.add_argument(
        "--condition-crop-size",
        type=int,
        default=CROP_SIZE,
        help="square crop size in the 2048 condition image; must be DINO-patch aligned",
    )
    parser.add_argument(
        "--boundary-anchor-width",
        type=int,
        default=0,
        help="add baseline E12 support within this many C128 cells of x/y/z=64",
    )
    parser.add_argument(
        "--boundary-anchor-feature-policy",
        choices=("fresh", "baseline"),
        default="baseline",
        help="feature source on common boundary-shell coordinates",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    n_values = parse_int_list(args.n_values)
    window_starts = parse_window_starts(args.window_starts)
    owner_bounds = ownership_bounds(window_starts)
    if args.condition_crop_size <= 0 or args.condition_crop_size > CONDITION_IMAGE_2048:
        raise ValueError("condition-crop-size must be in 1..2048")
    if args.condition_crop_size % DINO_PATCH:
        raise ValueError(f"condition-crop-size must be divisible by {DINO_PATCH}")
    global CROP_SIZE
    CROP_SIZE = int(args.condition_crop_size)
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError(f"modes must be drawn from {MODES}")
    if args.boundary_anchor_width < 0 or args.boundary_anchor_width > GLOBAL_GRID // 4:
        raise ValueError("boundary-anchor-width must be in 0..32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    endpoint = args.endpoint_dir.resolve()
    shape_payload = load(endpoint / "support/fixed_c128_shape_slat.pt")
    coords = shape_payload["coords"].int()
    raw = shape_payload["raw_features"].float()
    shape_norm_payload = load(endpoint / "support/fixed_c128_shape_normalized.pt")
    clean_norm = shape_norm_payload["normalized_features"].float()
    if not torch.equal(coords, shape_norm_payload["coords"].int()):
        raise RuntimeError("endpoint normalized coordinates differ from raw endpoint")
    if not torch.equal(normalize(raw, json.loads((args.model_path / "pipeline.json").read_text())["args"]["shape_slat_normalization"]), clean_norm):
        raise RuntimeError("endpoint normalized features do not match pipeline normalization")
    if not args.camera.is_file():
        raise FileNotFoundError(args.camera)
    camera = load(args.camera) if args.camera.suffix == ".pt" else json.loads(args.camera.read_text(encoding="utf-8"))
    if "camera" in camera:
        camera = camera["camera"]
    visibility_payload = load(args.visibility)
    visible_tokens = visibility_payload["visible"].bool()
    if not torch.equal(coords[:, 1:].int(), visibility_payload["coords"].int()):
        raise RuntimeError("visibility coordinates do not match fixed C128 endpoint")

    print("[model] loading geometry-only Pixal3D pipeline", flush=True)
    pipeline = init_shape_pipeline(args.model_path, device)
    cached_canonical = {
        "image_4096": args.canonical_dir / "image_4096.png",
        "image_1024": args.canonical_dir / "image_1024.png",
        "foreground_mask_4096": args.canonical_dir / "foreground_mask_4096.png",
    }
    if all(path.is_file() for path in cached_canonical.values()):
        print(f"[canonical] reuse cached pyramid from {args.canonical_dir}", flush=True)
        canonical = {
            key: Image.open(path).convert("L" if "mask" in key else "RGB")
            for key, path in cached_canonical.items()
        }
    else:
        source = Image.open(args.image).convert("RGB")
        canonical = pipeline.preprocess_canonical_images(source)
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    canonical["image_4096"].save(input_dir / "canonical_4096.png")
    canonical["image_4096"].resize(
        (CONDITION_IMAGE_2048, CONDITION_IMAGE_2048), Image.Resampling.LANCZOS
    ).save(input_dir / "canonical_2048.png")
    canonical["image_1024"].save(input_dir / "canonical_1024.png")
    canonical["foreground_mask_4096"].resize(
        (1024, 1024), Image.Resampling.NEAREST
    ).save(input_dir / "foreground_mask_1024.png")
    image_2048 = canonical["image_4096"].resize(
        (CONDITION_IMAGE_2048, CONDITION_IMAGE_2048), Image.Resampling.LANCZOS
    )

    disjoint_layout = window_starts == BLOCK_STARTS
    records = cube_records(coords, window_starts, require_partition=disjoint_layout)
    save_crops(image_2048, records, coords, camera, output)
    routing: list[dict[str, Any]] = []
    for record in records:
        rows = record["global_row_ids"].long()
        fraction = float(visible_tokens.index_select(0, rows).float().mean())
        routing.append(
            {
                "cube_id": int(record["cube_id"]),
                "start": list(record["start"]),
                "tokens": int(rows.numel()),
                "visible_tokens": int(visible_tokens.index_select(0, rows).sum()),
                "visible_fraction": fraction,
                "conditional": bool(fraction > 0.1),
            }
        )
    atomic_json(output / "routing.json", {"threshold": 0.1, "cubes": routing})

    stage32_records, clean32 = pool_c128_to_c32(records, coords, clean_norm)
    attach_crops(stage32_records, records)
    stage32_by_id = {int(record["cube_id"]): record for record in stage32_records}
    atomic_save(
        output / "support/local_c32_endpoint.pt",
        {
            "format": FORMAT,
            "coords": torch.cat(
                [
                    torch.cat((torch.zeros((len(record["local_xyz"]), 1), dtype=torch.int32), record["local_xyz"]), 1)
                    for record in stage32_records
                ],
                0,
            ),
            "features_normalized": clean32,
        },
    )
    atomic_json(
        output / "support/local_support_summary.json",
        {
            "format": FORMAT,
            "c128_tokens": int(coords.shape[0]),
            "c32_tokens": int(clean32.shape[0]),
            "c128_per_cube": {str(row["cube_id"]): row["tokens"] for row in routing},
            "c32_per_cube": {str(record["cube_id"]): int(record["local_xyz"].shape[0]) for record in stage32_records},
        },
    )

    pipeline.shape_slat_sampler_params["steps"] = STEPS
    schedule = pipeline.shape_slat_sampler.timestep_schedule(
        STEPS, float(pipeline.shape_slat_sampler_params.get("rescale_t", 3.0))
    )
    sigma_min = float(pipeline.shape_slat_sampler.sigma_min)
    route_flags = {
        "conditional": {int(row["cube_id"]): True for row in routing},
        "unconditional": {int(row["cube_id"]): False for row in routing},
        "visibility_gated": {int(row["cube_id"]): bool(row["conditional"]) for row in routing},
    }
    manifest = {
        "format": FORMAT,
        "status": "running",
        "path": "fixed C128 E12 -> local C32 -> Shape512 -> decoder C64 -> Shape1024 -> optional baseline boundary anchors -> global C128 decode2048",
        "geometry_only": True,
        "texture": False,
        "endpoint": str(endpoint / "support/fixed_c128_shape_slat.pt"),
        "endpoint_feature_hash": tensor_hash(clean_norm),
        "endpoint_support_tokens": int(coords.shape[0]),
        "block_grid": BLOCK_GRID,
        "block_stride": BLOCK_GRID if disjoint_layout else None,
        "window_starts": list(window_starts),
        "ownership_bounds": list(owner_bounds),
        "ownership_planes": list(owner_bounds[1:-1]),
        "local_coordinate_ranges": {"shape512": [0, 31], "shape1024": [0, 63]},
        "condition_source": "canonical 4096 resized to 2048, patch-aligned square crop centered on projected endpoint C128 points",
        "condition_projection_coordinate_system": "global C128 coords; attached to local Flow rows after projection",
        "condition_image_size": [CONDITION_IMAGE_2048, CONDITION_IMAGE_2048],
        "condition_crop_size": [CROP_SIZE, CROP_SIZE],
        "shape512_endpoint": "mean-pool normalized E12 C128 features by local C64//2 cells",
        "shape1024_endpoint": "nearest E12 normalized C128 feature at each newly generated local C64 coordinate",
        "schedule": schedule,
        "sigma_min": sigma_min,
        "n_values": list(n_values),
        "modes": list(modes),
        "noise_seed": int(args.noise_seed),
        "routing": routing,
        "sampler_params": dict(pipeline.shape_slat_sampler_params),
        "support_can_change_at_c32_to_c64": True,
        "feature_init": args.feature_init,
        "shape1024_feature_init": args.shape1024_feature_init,
        "boundary_anchor": {
            "width": int(args.boundary_anchor_width),
            "planes": list(owner_bounds[1:-1]),
            "feature_policy": args.boundary_anchor_feature_policy,
            "source": str(endpoint / "support/fixed_c128_shape_slat.pt"),
        },
        "max_batch_tokens": int(args.max_batch_tokens),
        "cuda_device": int(args.cuda_device),
    }
    atomic_json(output / "run_manifest.json", manifest)

    noise32 = seeded_noise(clean32.shape, args.noise_seed)
    atomic_save(
        output / "support/local_c32_noise.pt",
        {"seed": args.noise_seed, "noise": noise32, "hash": tensor_hash(noise32)},
    )
    shape_norm_spec = pipeline.shape_slat_normalization
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        all_rows[mode] = []
        for n in n_values:
            variant = output / mode / f"n_{n:02d}"
            state_path = variant / "shape_c64_final_normalized.pt"
            record_path = variant / "record.json"
            if args.resume and state_path.is_file() and record_path.is_file():
                row = json.loads(record_path.read_text(encoding="utf-8"))
                all_rows[mode].append(row)
                print(f"[resume] {mode} n={n:02d}", flush=True)
                continue

            print(f"[variant {mode} n={n:02d}] local Shape512 suffix", flush=True)
            enabled32 = route_flags[mode]
            flow_start32 = 0 if args.feature_init == "fresh" else n
            need32 = any(enabled32.values()) and flow_start32 < STEPS
            if need32:
                conditions32 = extract_conditions(
                    pipeline, image_2048, camera, stage32_records, "shape512", enabled32, output / mode / f"n_{n:02d}"
                )
                routed32 = route_condition(conditions32, stage32_records, enabled32)
            else:
                conditions32 = extract_conditions(
                    pipeline, image_2048, camera, stage32_records, "shape512", {k: False for k in enabled32}, output / mode / f"n_{n:02d}"
                )
                routed32 = route_condition(conditions32, stage32_records, {k: False for k in enabled32})
            state32 = (
                noise32.clone()
                if args.feature_init == "fresh"
                else re_noise(clean32, noise32, schedule[n], sigma_min)
            )
            state32, shape512_timings = run_suffix(
                pipeline,
                stage32_records,
                state32,
                routed32,
                "shape_slat_flow_model_512",
                flow_start32,
                schedule,
                mode,
                device,
                args.max_batch_tokens,
            )
            atomic_save(
                variant / "shape_c32_final_normalized.pt",
                {"coords": torch.cat([r["local_xyz"] for r in stage32_records], 0), "features": state32, "n": n},
            )
            del conditions32, routed32
            empty_cuda()

            print(f"[variant {mode} n={n:02d}] decoder C512 -> local C64 support", flush=True)
            stage64_records = local_c32_to_c64(pipeline, stage32_records, state32, device)
            attach_crops(stage64_records, records)
            clean64 = nearest_endpoint_features(stage64_records, records, coords, clean_norm)
            noise64 = seeded_noise(clean64.shape, args.noise_seed + 100003)
            shape1024_fresh = (
                args.feature_init == "fresh"
                or args.shape1024_feature_init == "fresh"
            )
            flow_start64 = 0 if shape1024_fresh else n
            state64 = (
                noise64.clone()
                if shape1024_fresh
                else re_noise(clean64, noise64, schedule[n], sigma_min)
            )
            need64 = any(enabled32.values()) and flow_start64 < STEPS
            if need64:
                conditions64 = extract_conditions(
                    pipeline, image_2048, camera, stage64_records, "shape1024", enabled32, variant
                )
                routed64 = route_condition(conditions64, stage64_records, enabled32)
            else:
                conditions64 = extract_conditions(
                    pipeline, image_2048, camera, stage64_records, "shape1024", {k: False for k in enabled32}, variant
                )
                routed64 = route_condition(conditions64, stage64_records, {k: False for k in enabled32})
            print(f"[variant {mode} n={n:02d}] local Shape1024 suffix", flush=True)
            state64, shape1024_timings = run_suffix(
                pipeline,
                stage64_records,
                state64,
                routed64,
                "shape_slat_flow_model_1024",
                flow_start64,
                schedule,
                mode,
                device,
                args.max_batch_tokens,
            )
            coords_global, raw_global = assemble_global(
                stage64_records,
                state64,
                shape_norm_spec,
                window_starts,
                owner_bounds,
            )
            decode_coords, decode_raw, anchor_meta = add_boundary_anchors(
                coords_global,
                raw_global,
                coords,
                raw,
                int(args.boundary_anchor_width),
                args.boundary_anchor_feature_policy,
                owner_bounds[1:-1],
            )
            atomic_save(
                variant / "shape_c64_final_normalized.pt",
                {"coords": coords_global, "features": normalize(raw_global, shape_norm_spec), "n": n},
            )
            atomic_save(
                variant / "shape_c64_final_denormalized.pt",
                {"coords": coords_global, "features": raw_global, "n": n},
            )
            atomic_save(
                variant / "support_c64_local.pt",
                {
                    "format": FORMAT,
                    "cubes": [
                        {"cube_id": int(record["cube_id"]), "start": record["start"], "local_xyz": record["local_xyz"]}
                        for record in stage64_records
                    ],
                },
            )
            atomic_save(
                variant / "shape_c128_decode_input.pt",
                {
                    "format": FORMAT,
                    "coords": decode_coords,
                    "raw_features": decode_raw,
                    "boundary_anchor": anchor_meta,
                },
            )
            del conditions64, routed64, state32, clean64, noise64, state64
            empty_cuda()

            print(
                f"[variant {mode} n={n:02d}] global C128 decode 2048 "
                f"tokens={decode_coords.shape[0]:,} anchors={anchor_meta['new_baseline_tokens']:,}",
                flush=True,
            )
            slat = SparseTensor(decode_raw.to(device), decode_coords.to(device))
            meshes, _ = pipeline.decode_shape_slat(slat, 2048)
            if len(meshes) != 1:
                raise RuntimeError(f"expected one decoded mesh, got {len(meshes)}")
            mesh = meshes[0]
            variant.mkdir(parents=True, exist_ok=True)
            mesh_path = variant / "geometry_mesh.pt"
            atomic_save(mesh_path, {"format": FORMAT, "mesh": mesh.cpu()})
            import trimesh

            trimesh.Trimesh(
                vertices=mesh.vertices.detach().cpu().numpy(),
                faces=mesh.faces.detach().cpu().numpy(),
                process=False,
            ).export(variant / "geometry_mesh.glb")
            del slat, meshes, mesh
            empty_cuda()

            from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

            mesh_payload = load(mesh_path)
            mesh_for_render = mesh_payload["mesh"].to(device)
            render = render_geometry(
                mesh_for_render,
                camera,
                output / "inputs/canonical_1024.png",
                DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png",
                variant,
                int(args.render_resolution),
                tuple(int(item) % 360 for item in args.angles.split(",")),
                int(args.render_chunk_size),
            )
            front_normal = variant / f"multiview_{args.render_resolution}/view_000_camera_normal.png"
            normal_ref = DEFAULT_BASELINE / "raw_ovoxel_render/normal.png"
            normal_mask = DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png"
            normal_metrics_row = normal_metrics(
                front_normal, normal_ref, normal_mask
            )
            normal_reference = {
                "kind": "raw_ovoxel_baseline_render_proxy",
                "is_ground_truth": False,
                "image": str(normal_ref.resolve()),
                "mask": str(normal_mask.resolve()),
                "comparison": "RGB encoded normal, resized to 1024, pixel MSE/PSNR/SSIM/MAE",
            }
            row = {
                "format": FORMAT,
                "mode": mode,
                "n": n,
                "feature_init": args.feature_init,
                "shape1024_feature_init": args.shape1024_feature_init,
                "start_t": schedule[0] if args.feature_init == "fresh" else schedule[n],
                "start_t_shape512": schedule[0] if args.feature_init == "fresh" else schedule[n],
                "start_t_shape1024": schedule[0] if shape1024_fresh else schedule[n],
                "flow_start_step": flow_start64 if flow_start32 == flow_start64 else None,
                "flow_start_step_shape512": flow_start32,
                "flow_start_step_shape1024": flow_start64,
                "suffix_steps": STEPS - flow_start64,
                "c32_tokens": int(sum(len(record["local_xyz"]) for record in stage32_records)),
                "c64_tokens": int(coords_global.shape[0]),
                "window_starts": list(window_starts),
                "ownership_bounds": list(owner_bounds),
                "decoder_c128_tokens": int(decode_coords.shape[0]),
                "vertices": int(mesh_payload["mesh"].vertices.shape[0]),
                "faces": int(mesh_payload["mesh"].faces.shape[0]),
                "boundary_anchor": anchor_meta,
                "shape512_timings": shape512_timings,
                "shape1024_timings": shape1024_timings,
                "geometry_mesh": str(mesh_path.resolve()),
                "front_normal": str(front_normal.resolve()),
                "render": render,
                "normal_reference": normal_reference,
                "normal_reference_metrics": normal_metrics_row,
                # Kept for readers of earlier experiment records.
                "baseline_normal_metrics": normal_metrics_row,
                "state_hash": tensor_hash(load(variant / "shape_c64_final_normalized.pt")["features"]),
            }
            atomic_json(record_path, row)
            all_rows[mode].append(row)
            print(
                json.dumps(
                    {"mode": mode, "n": n, "c64_tokens": row["c64_tokens"], "vertices": row["vertices"], "faces": row["faces"], **normal_metrics_row},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del mesh_payload, mesh_for_render
            empty_cuda()
        make_sheet(all_rows[mode], output / mode / "front_normals_sheet.png", mode)
        atomic_json(output / mode / "summary.json", {"format": FORMAT, "mode": mode, "results": all_rows[mode]})

    manifest.update({"status": "complete", "seconds": time.perf_counter() - started, "results": all_rows})
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "summary.json", manifest)
    shutil.copy2(__file__, output / Path(__file__).name)
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
