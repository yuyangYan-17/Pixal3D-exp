#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Pixal3D DINOv3 global features from several image constructions.

Compared sources:
  1. hr_native:
       preprocess_image_with_hr()["hr_image"] at native aligned resolution.
  2. hr_resize_1024:
       the same HR image resized to 1024x1024.
  3. pipeline_global_1024:
       preprocess_image_with_hr()["global_image"], then resized to 1024x1024
       exactly as DinoV3ProjFeatureExtractor.forward() does for PIL inputs.
  4. tiles_stride1024_mean:
       1024x1024 black-padded HR tiles, stride 1024, raw global tensors
       averaged position-wise over tiles.
  5. tiles_stride512_mean:
       1024x1024 black-padded HR tiles, stride 512, raw global tensors
       averaged position-wise over tiles.

The script reuses:
  - Pixal3DImageTo3DPipeline.preprocess_image_with_hr()
  - DinoV3ProjFeatureExtractor.transform
  - DinoV3ProjFeatureExtractor.extract_features()

Outputs:
  output_dir/
    processed_global.png
    processed_hr.png
    hr_resize_1024.png
    pipeline_global_1024.png
    dino_global_features.pt
    dino_global_comparison.json
    tile_records_stride1024.json
    tile_records_stride512.json
    tiles_stride1024/   (optional)
    tiles_stride512/    (optional)

Example:
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
  python compare_dino_global_sources.py \
    --image assets/choose/0_img.png \
    --output-dir outputs/dino_global_compare
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Set environment defaults before importing Pixal3D.
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"
DEFAULT_DINO_MODEL = (
    "/home/nvme04/yyyan/download/model/"
    "dinov3-vitl16-pretrain-lvd1689m/facebook/"
    "dinov3-vitl16-pretrain-lvd1689m"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help=(
            "DINO weight/input dtype. bfloat16 is recommended for the native "
            "HR pass. All compared branches use the same dtype."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=1,
        help="Number of equal-sized 1024 tiles processed in one DINO batch.",
    )
    parser.add_argument(
        "--save-tiles",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-native-hr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the expensive native-resolution full-frame DINO pass.",
    )
    parser.add_argument(
        "--require-native-hr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of continuing if the native HR DINO pass OOMs.",
    )
    parser.add_argument(
        "--foreground-epsilon",
        type=float,
        default=0.0,
        help=(
            "A tile is included in the optional foreground-only mean when its "
            "foreground ratio is greater than this value. The requested all-"
            "tile means always include every tile."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype != torch.float32:
        print(
            f"[warning] dtype={name} on CPU is not recommended; using float32",
            file=sys.stderr,
        )
        dtype = torch.float32
    return dtype


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def pil_to_tensor_batch(
    images: Sequence[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    if not images:
        raise ValueError("images must not be empty")
    expected_size = images[0].size
    tensors: List[torch.Tensor] = []
    for index, image in enumerate(images):
        rgb = image.convert("RGB")
        if rgb.size != expected_size:
            raise ValueError(
                "All images in a DINO batch must have the same size: "
                f"image[0]={expected_size}, image[{index}]={rgb.size}"
            )
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensors.append(tensor)
    return torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)


def load_and_preprocess(
    image_path: Path,
    model_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Reuse the current Pixal3D HR preprocessing implementation exactly.

    The full pipeline is loaded only to obtain its configured rembg model and
    call preprocess_image_with_hr(). It is deleted before loading DINO.
    """
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    print(f"[pipeline] loading preprocessing pipeline from {model_path}")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
    pipeline._device = device
    pipeline.low_vram = True

    with Image.open(image_path) as opened:
        source = opened.copy()

    started = time.perf_counter()
    bundle = pipeline.preprocess_image_with_hr(source)
    elapsed = time.perf_counter() - started
    print(
        "[preprocess] "
        f"source={source.size[0]}x{source.size[1]} "
        f"global={bundle['global_image'].size[0]}x"
        f"{bundle['global_image'].size[1]} "
        f"hr={bundle['hr_image'].size[0]}x{bundle['hr_image'].size[1]} "
        f"seconds={elapsed:.3f}"
    )

    # Ensure returned PIL objects survive deletion of the pipeline/source.
    result = {
        "global_image": bundle["global_image"].copy(),
        "hr_image": bundle["hr_image"].copy(),
        "foreground_mask_hr": bundle["foreground_mask_hr"].copy(),
        "global_to_hr_transform": dict(bundle["global_to_hr_transform"]),
    }

    del bundle, source, pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_dino_extractor(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    print(f"[dino] loading from {model_name}")
    extractor = DinoV3ProjFeatureExtractor(
        model_name=model_name,
        image_size=1024,
        grid_resolution=64,
        use_naf_upsample=False,
    )
    extractor.eval()
    extractor.requires_grad_(False)
    # The repository overrides extractor.to(device) with a device-only
    # signature, so move the module first and cast the frozen DINO backbone
    # explicitly instead of calling extractor.to(device=..., dtype=...).
    extractor.to(device)
    extractor.model.to(device=device, dtype=dtype)
    return extractor


@torch.inference_mode()
def extract_global_batch(
    extractor,
    images: Sequence[Image.Image],
    device: torch.device,
    label: str,
) -> torch.Tensor:
    """
    Return raw DINO global features as float32 CPU tensor [B, 1+R, D].

    This deliberately calls extractor.extract_features() directly so the
    native-HR branch is not forcibly resized to extractor.image_size.
    """
    if not images:
        raise ValueError("images must not be empty")

    width, height = images[0].size
    x = pil_to_tensor_batch(images, device=device)
    normalized = extractor.transform(x)

    started = time.perf_counter()
    features = extractor.extract_features(normalized)
    num_register_tokens = int(
        getattr(extractor.model.config, "num_register_tokens", 4)
    )
    global_features = torch.cat(
        [
            features[:, 0:1],
            features[:, 1 : 1 + num_register_tokens],
        ],
        dim=1,
    ).contiguous()
    elapsed = time.perf_counter() - started

    result = global_features.float().cpu()
    print(
        f"[dino-global] label={label} batch={len(images)} "
        f"image={width}x{height} output={tuple(result.shape)} "
        f"seconds={elapsed:.3f}"
    )

    del x, normalized, features, global_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def image_tile_starts(
    image_extent: int,
    tile_size: int,
    tile_stride: int,
) -> List[int]:
    """
    Match the current HR tile implementation:
        list(range(0, image_extent, tile_stride))

    Therefore an edge crop extending beyond the image is black padded by
    Pillow, and 4152 with tile=1024 creates 5x5 tiles at stride 1024.
    """
    if image_extent <= 0:
        raise ValueError("image_extent must be positive")
    if tile_size <= 0 or tile_stride <= 0:
        raise ValueError("tile_size and tile_stride must be positive")
    if tile_stride > tile_size:
        raise ValueError("tile_stride may not exceed tile_size")
    return list(range(0, image_extent, tile_stride))


def build_tiles(
    image: Image.Image,
    foreground_mask: Image.Image,
    tile_size: int,
    stride: int,
    save_directory: Optional[Path],
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    if image.size != foreground_mask.size:
        raise ValueError(
            f"image/mask size mismatch: {image.size} vs {foreground_mask.size}"
        )
    if image.width != image.height:
        raise ValueError(f"Expected square HR image, got {image.size}")

    starts_x = image_tile_starts(image.width, tile_size, stride)
    starts_y = image_tile_starts(image.height, tile_size, stride)
    tiles: List[Image.Image] = []
    records: List[Dict[str, Any]] = []

    if save_directory is not None:
        save_directory.mkdir(parents=True, exist_ok=True)

    tile_index = 0
    for y0 in starts_y:
        for x0 in starts_x:
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            # Pillow returns the requested tile_size canvas and fills the
            # out-of-image area with black, matching the current pipeline.
            tile = image.convert("RGB").crop((x0, y0, x1, y1))
            mask_tile = foreground_mask.convert("L").crop((x0, y0, x1, y1))
            if tile.size != (tile_size, tile_size):
                raise RuntimeError(
                    f"Unexpected tile size {tile.size}; expected "
                    f"{(tile_size, tile_size)}"
                )

            mask_array = np.asarray(mask_tile, dtype=np.uint8)
            foreground_ratio = float(np.count_nonzero(mask_array > 0)) / float(
                tile_size * tile_size
            )
            record = {
                "tile_index": tile_index,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "actual_x1": min(x1, image.width),
                "actual_y1": min(y1, image.height),
                "foreground_ratio": foreground_ratio,
                "has_foreground": bool(foreground_ratio > 0.0),
            }
            tiles.append(tile)
            records.append(record)

            if save_directory is not None:
                tile.save(
                    save_directory
                    / f"tile_{tile_index:04d}_x{x0}_y{y0}.png"
                )
            tile_index += 1

    return tiles, records


@torch.inference_mode()
def extract_tile_globals(
    extractor,
    tiles: Sequence[Image.Image],
    records: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    label: str,
    foreground_epsilon: float,
) -> Dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("tile_batch_size must be positive")
    if len(tiles) != len(records):
        raise ValueError("tiles and records must have identical lengths")

    chunks: List[torch.Tensor] = []
    for start in range(0, len(tiles), batch_size):
        end = min(start + batch_size, len(tiles))
        chunk = extract_global_batch(
            extractor,
            tiles[start:end],
            device=device,
            label=f"{label}[{start}:{end}]",
        )
        chunks.append(chunk)

    all_globals = torch.cat(chunks, dim=0)  # [T, G, D]
    all_mean = all_globals.mean(dim=0, keepdim=True)

    ratios = torch.tensor(
        [float(record["foreground_ratio"]) for record in records],
        dtype=torch.float32,
    )
    active_mask = ratios > float(foreground_epsilon)
    result: Dict[str, torch.Tensor] = {
        "all": all_globals,
        "all_mean": all_mean,
        "foreground_ratios": ratios,
        "active_mask": active_mask,
    }

    if bool(active_mask.any()):
        result["foreground_mean"] = all_globals[active_mask].mean(
            dim=0, keepdim=True
        )
        weights = ratios[active_mask].clamp_min(torch.finfo(torch.float32).eps)
        weighted = (
            all_globals[active_mask] * weights[:, None, None]
        ).sum(dim=0, keepdim=True) / weights.sum()
        result["foreground_weighted_mean"] = weighted

    return result


def token_labels(num_tokens: int) -> List[str]:
    if num_tokens <= 0:
        return []
    labels = ["cls"]
    labels.extend(f"reg_{index}" for index in range(1, num_tokens))
    return labels


def tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    x = value.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "l2_norm": float(x.norm().item()),
    }


def compare_features(
    first: torch.Tensor,
    second: torch.Tensor,
) -> Dict[str, Any]:
    a = first.detach().float().cpu()
    b = second.detach().float().cpu()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.ndim != 3 or a.shape[0] != 1:
        raise ValueError(
            "Expected global feature shape [1, G, D], got "
            f"{tuple(a.shape)}"
        )

    eps = torch.finfo(torch.float32).eps
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    difference = a - b

    per_token_cosine = F.cosine_similarity(
        a[0], b[0], dim=-1, eps=eps
    )
    per_token_rel_l2 = (
        difference[0].norm(dim=-1)
        / b[0].norm(dim=-1).clamp_min(eps)
    )

    labels = token_labels(int(a.shape[1]))
    token_metrics = {
        label: {
            "cosine_similarity": float(per_token_cosine[index].item()),
            "relative_l2_to_second": float(per_token_rel_l2[index].item()),
            "first_norm": float(a[0, index].norm().item()),
            "second_norm": float(b[0, index].norm().item()),
        }
        for index, label in enumerate(labels)
    }

    return {
        "cosine_similarity": float(
            F.cosine_similarity(
                a_flat[None], b_flat[None], dim=1, eps=eps
            ).item()
        ),
        "relative_l2_to_second": float(
            (difference.norm() / b.norm().clamp_min(eps)).item()
        ),
        "symmetric_relative_l2": float(
            (
                2.0
                * difference.norm()
                / (a.norm() + b.norm()).clamp_min(eps)
            ).item()
        ),
        "mse": float(difference.square().mean().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "max_abs": float(difference.abs().max().item()),
        "first_norm": float(a.norm().item()),
        "second_norm": float(b.norm().item()),
        "norm_ratio_first_over_second": float(
            (a.norm() / b.norm().clamp_min(eps)).item()
        ),
        "mean_token_cosine_similarity": float(
            per_token_cosine.mean().item()
        ),
        "token_metrics": token_metrics,
    }


def tile_record_metrics(
    tile_globals: torch.Tensor,
    tile_mean: torch.Tensor,
    baseline: Optional[torch.Tensor],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    values = tile_globals.float().cpu()
    mean = tile_mean.float().cpu()
    eps = torch.finfo(torch.float32).eps

    mean_flat = mean.reshape(1, -1).expand(values.shape[0], -1)
    values_flat = values.reshape(values.shape[0], -1)
    cosine_to_mean = F.cosine_similarity(
        values_flat, mean_flat, dim=1, eps=eps
    )

    cosine_to_baseline: Optional[torch.Tensor] = None
    if baseline is not None:
        base_flat = baseline.float().cpu().reshape(1, -1).expand(
            values.shape[0], -1
        )
        cosine_to_baseline = F.cosine_similarity(
            values_flat, base_flat, dim=1, eps=eps
        )

    enriched: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        item = dict(record)
        item["global_l2_norm"] = float(values[index].norm().item())
        item["cosine_to_all_tile_mean"] = float(
            cosine_to_mean[index].item()
        )
        if cosine_to_baseline is not None:
            item["cosine_to_pipeline_global_1024"] = float(
                cosine_to_baseline[index].item()
            )
        enriched.append(item)
    return enriched


def print_reference_table(
    features: Mapping[str, torch.Tensor],
    reference_name: str,
    comparisons: Mapping[str, Any],
) -> None:
    if reference_name not in features:
        return

    print()
    print(f"[comparison-to-reference] reference={reference_name}")
    header = (
        f"{'method':30s} "
        f"{'cosine':>12s} "
        f"{'rel_l2':>12s} "
        f"{'sym_rel_l2':>12s} "
        f"{'mse':>12s}"
    )
    print(header)
    print("-" * len(header))
    for name in features:
        if name == reference_name:
            continue
        key = f"{name}__vs__{reference_name}"
        metric = comparisons.get(key)
        if metric is None:
            continue
        print(
            f"{name:30s} "
            f"{metric['cosine_similarity']:12.8f} "
            f"{metric['relative_l2_to_second']:12.8f} "
            f"{metric['symmetric_relative_l2']:12.8f} "
            f"{metric['mse']:12.8e}"
        )


def main() -> None:
    args = parse_args()
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.tile_batch_size <= 0:
        raise ValueError("--tile-batch-size must be positive")
    if not math.isfinite(args.foreground_epsilon):
        raise ValueError("--foreground-epsilon must be finite")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(
        f"[setup] image={args.image} output={args.output_dir} "
        f"device={device} dtype={dtype}"
    )

    preprocess_bundle = load_and_preprocess(
        image_path=args.image,
        model_path=args.model_path,
        device=device,
    )
    global_image: Image.Image = preprocess_bundle["global_image"]
    hr_image: Image.Image = preprocess_bundle["hr_image"]
    foreground_mask_hr: Image.Image = preprocess_bundle[
        "foreground_mask_hr"
    ]

    global_image.save(args.output_dir / "processed_global.png")
    hr_image.save(args.output_dir / "processed_hr.png")
    foreground_mask_hr.save(
        args.output_dir / "processed_hr_foreground_mask.png"
    )

    hr_resize_1024 = hr_image.resize(
        (1024, 1024), Image.Resampling.LANCZOS
    )
    # This is the exact size presented to the baseline tex/shape DINO
    # extractor when a PIL image is passed through its normal forward().
    pipeline_global_1024 = global_image.resize(
        (1024, 1024), Image.Resampling.LANCZOS
    )
    hr_resize_1024.save(args.output_dir / "hr_resize_1024.png")
    pipeline_global_1024.save(
        args.output_dir / "pipeline_global_1024.png"
    )

    extractor = build_dino_extractor(
        model_name=args.dino_model,
        device=device,
        dtype=dtype,
    )
    patch_size = int(extractor.patch_size)
    num_register_tokens = int(
        getattr(extractor.model.config, "num_register_tokens", 4)
    )
    embed_dim = int(extractor.embed_dim)

    features: Dict[str, torch.Tensor] = {}
    errors: Dict[str, str] = {}

    # Baseline/current-pipeline global.
    features["pipeline_global_1024"] = extract_global_batch(
        extractor,
        [pipeline_global_1024],
        device=device,
        label="pipeline_global_1024",
    )

    # HR image resized to training scale.
    features["hr_resize_1024"] = extract_global_batch(
        extractor,
        [hr_resize_1024],
        device=device,
        label="hr_resize_1024",
    )

    # Native full HR global. This may be computationally expensive.
    if not args.skip_native_hr:
        try:
            features["hr_native"] = extract_global_batch(
                extractor,
                [hr_image],
                device=device,
                label="hr_native",
            )
        except torch.cuda.OutOfMemoryError as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors["hr_native"] = message
            print(f"[warning] native HR DINO OOM: {message}", file=sys.stderr)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.require_native_hr:
                raise
        except RuntimeError as exc:
            # Also retain shape/attention backend failures in the report.
            message = f"{type(exc).__name__}: {exc}"
            errors["hr_native"] = message
            print(
                f"[warning] native HR DINO failed: {message}",
                file=sys.stderr,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.require_native_hr:
                raise

    tile_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
    tile_records_by_stride: Dict[int, List[Dict[str, Any]]] = {}

    for stride in (1024, 512):
        save_dir = (
            args.output_dir / f"tiles_stride{stride}"
            if args.save_tiles
            else None
        )
        tiles, records = build_tiles(
            image=hr_image,
            foreground_mask=foreground_mask_hr,
            tile_size=args.tile_size,
            stride=stride,
            save_directory=save_dir,
        )
        print(
            f"[tiles] stride={stride} tile={args.tile_size} "
            f"count={len(tiles)} foreground_tiles="
            f"{sum(record['has_foreground'] for record in records)}"
        )
        outputs = extract_tile_globals(
            extractor=extractor,
            tiles=tiles,
            records=records,
            device=device,
            batch_size=args.tile_batch_size,
            label=f"tiles_stride{stride}",
            foreground_epsilon=args.foreground_epsilon,
        )
        tile_outputs[f"stride{stride}"] = outputs

        mean_name = f"tiles_stride{stride}_mean"
        features[mean_name] = outputs["all_mean"]

        if "foreground_mean" in outputs:
            features[
                f"tiles_stride{stride}_foreground_mean"
            ] = outputs["foreground_mean"]
        if "foreground_weighted_mean" in outputs:
            features[
                f"tiles_stride{stride}_foreground_weighted_mean"
            ] = outputs["foreground_weighted_mean"]

        enriched_records = tile_record_metrics(
            tile_globals=outputs["all"],
            tile_mean=outputs["all_mean"],
            baseline=features.get("pipeline_global_1024"),
            records=records,
        )
        tile_records_by_stride[stride] = enriched_records
        atomic_json(
            args.output_dir / f"tile_records_stride{stride}.json",
            enriched_records,
        )
        del tiles
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Pairwise comparisons among every available [1, G, D] feature.
    pairwise: Dict[str, Any] = {}
    names = list(features.keys())
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            key_forward = f"{first_name}__vs__{second_name}"
            key_reverse = f"{second_name}__vs__{first_name}"
            pairwise[key_forward] = compare_features(
                features[first_name],
                features[second_name],
            )
            pairwise[key_reverse] = compare_features(
                features[second_name],
                features[first_name],
            )

    feature_statistics = {
        name: tensor_stats(value) for name, value in features.items()
    }

    metadata = {
        "source_image": str(args.image),
        "model_path": str(args.model_path),
        "dino_model": str(args.dino_model),
        "device": str(device),
        "dtype": str(dtype),
        "source_sizes": {
            "global_image": list(global_image.size),
            "hr_image": list(hr_image.size),
            "hr_resize_1024": list(hr_resize_1024.size),
            "pipeline_global_1024": list(pipeline_global_1024.size),
        },
        "dino": {
            "patch_size": patch_size,
            "num_register_tokens": num_register_tokens,
            "num_global_tokens": 1 + num_register_tokens,
            "embed_dim": embed_dim,
        },
        "tiling": {
            "tile_size": args.tile_size,
            "stride1024_count": len(tile_records_by_stride[1024]),
            "stride512_count": len(tile_records_by_stride[512]),
            "tile_start_rule": "range(0, image_extent, stride)",
            "edge_padding": "Pillow crop outside image, black fill",
            "all_tile_mean_includes_background_tiles": True,
            "foreground_epsilon": args.foreground_epsilon,
        },
        "global_to_hr_transform": preprocess_bundle[
            "global_to_hr_transform"
        ],
    }

    report = {
        "metadata": metadata,
        "errors": errors,
        "feature_statistics": feature_statistics,
        "pairwise": pairwise,
    }
    atomic_json(
        args.output_dir / "dino_global_comparison.json",
        report,
    )

    # Preserve exact raw float32 tensors for later direct experiments.
    save_payload: Dict[str, Any] = {
        "metadata": metadata,
        "errors": errors,
        "features": {
            name: value.float().cpu() for name, value in features.items()
        },
        "tile_globals": {
            name: {
                key: value.float().cpu()
                for key, value in output.items()
            }
            for name, output in tile_outputs.items()
        },
    }
    torch.save(
        save_payload,
        args.output_dir / "dino_global_features.pt",
    )

    print_reference_table(
        features=features,
        reference_name="pipeline_global_1024",
        comparisons=pairwise,
    )

    print()
    print(
        "[done] "
        f"features={args.output_dir / 'dino_global_features.pt'} "
        f"report={args.output_dir / 'dino_global_comparison.json'}"
    )


if __name__ == "__main__":
    main()
