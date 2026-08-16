#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-render an existing Pixal3D experiment at 4096 and recompute metrics.

This is intentionally a render-only follow-up: it loads the cached baseline,
pure-HR, and guided meshes and never re-runs shape or texture generation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from render_pixal3d_raw_ovoxel import (
    LPIPSEvaluator,
    image_to_tensor,
    load_mesh_checkpoint,
    psnr_metric,
    ssim_metric,
)


VARIANTS = (
    ("global_baseline", "global_baseline_mesh.pt"),
    ("pure_HR", "variants/pure_HR/global_merged_mesh.pt"),
    (
        "cross_tile_pbr_perstep_guided",
        "variants/cross_tile_pbr_perstep_guided/global_merged_mesh.pt",
    ),
)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_args(args: argparse.Namespace, face_chunk_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        render_resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        render_ssaa=int(args.render_ssaa),
        render_peel_layers=int(args.render_peel_layers),
        render_face_chunk_size=int(face_chunk_size),
        use_envmap_bg=False,
        envmap=str(args.envmap),
        lpips_net=str(args.lpips_net),
        skip_lpips=True,
    )


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _render_one(
    *,
    mesh_path: Path,
    output_dir: Path,
    reference_path: Path,
    camera: Mapping[str, Any],
    envmap: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    mesh = load_mesh_checkpoint(mesh_path, device="cpu")
    candidates = [int(args.render_face_chunk_size)]
    for candidate in (1_000_000, 500_000, 250_000):
        if candidate not in candidates:
            candidates.append(candidate)
    last_error: BaseException | None = None
    result: Dict[str, Any] | None = None
    used_chunk = candidates[-1]
    for face_chunk_size in candidates:
        try:
            print(
                f"[render4096] {mesh_path.name} resolution={args.render_resolution} "
                f"face_chunk_size={face_chunk_size}"
            )
            result = core._render(
                mesh,
                output_dir=output_dir,
                camera=camera,
                reference_image=reference_path,
                args=_render_args(args, face_chunk_size),
                envmap=envmap,
            )
            used_chunk = face_chunk_size
            break
        except Exception as exc:
            last_error = exc
            if not _is_oom(exc) or face_chunk_size == candidates[-1]:
                raise
            print(
                f"[render4096] CUDA OOM at face_chunk_size={face_chunk_size}; "
                "retrying with a smaller render chunk"
            )
            _empty_cuda_cache()
    if result is None:
        raise RuntimeError("4096 render did not return a result") from last_error
    result["render_face_chunk_size_requested"] = int(args.render_face_chunk_size)
    result["render_face_chunk_size_used"] = int(used_chunk)
    del mesh
    _empty_cuda_cache()
    return result


def _recompute_metrics(
    *,
    render_result: Mapping[str, Any],
    reference_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    render_path = Path(str(render_result["render_png"]))
    with Image.open(reference_path) as reference_image:
        reference = reference_image.convert("RGB")
    with Image.open(render_path) as rendered_image:
        rendered = rendered_image.convert("RGB")
    metric_size = (int(args.metric_resolution), int(args.metric_resolution))
    reference_tensor = image_to_tensor(reference, metric_size)
    prediction_tensor = image_to_tensor(rendered, metric_size)
    record = {
        "psnr_db": psnr_metric(reference_tensor, prediction_tensor),
        "ssim": ssim_metric(reference_tensor, prediction_tensor),
        "metric_resolution": int(args.metric_resolution),
        "lpips": None,
        "lpips_resolution": None,
        "lpips_fallback": False,
    }
    if not bool(args.skip_lpips):
        try:
            evaluator = LPIPSEvaluator(str(args.lpips_net), torch.device("cuda"))
            record["lpips"] = evaluator.evaluate(reference_tensor, prediction_tensor)
            record["lpips_resolution"] = int(args.metric_resolution)
            evaluator.model.cpu()
            del evaluator
            _empty_cuda_cache()
        except Exception as exc:
            if not _is_oom(exc):
                raise
            print(
                "[metrics4096] full-resolution LPIPS OOM; retrying LPIPS at "
                f"{args.lpips_fallback_resolution}x{args.lpips_fallback_resolution}"
            )
            _empty_cuda_cache()
            fallback_size = (
                int(args.lpips_fallback_resolution),
                int(args.lpips_fallback_resolution),
            )
            fallback_reference = image_to_tensor(reference, fallback_size)
            fallback_prediction = image_to_tensor(rendered, fallback_size)
            evaluator = LPIPSEvaluator(str(args.lpips_net), torch.device("cuda"))
            record["lpips"] = evaluator.evaluate(fallback_reference, fallback_prediction)
            record["lpips_resolution"] = int(args.lpips_fallback_resolution)
            record["lpips_fallback"] = True
            evaluator.model.cpu()
            del evaluator
            _empty_cuda_cache()
    return record


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _save_comparison(
    *,
    reference_path: Path,
    render_paths: Mapping[str, Path],
    metrics: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    panel_size: int,
) -> None:
    entries = [
        ("input", reference_path),
        ("global_baseline", render_paths["global_baseline"]),
        ("pure_HR", render_paths["pure_HR"]),
        ("cross_tile_pbr_perstep_guided", render_paths["cross_tile_pbr_perstep_guided"]),
    ]
    header = 180
    canvas = Image.new("RGB", (panel_size * len(entries), panel_size + header), "black")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(max(24, panel_size // 80))
    metric_font = _font(max(18, panel_size // 120))
    for index, (label, path) in enumerate(entries):
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != (panel_size, panel_size):
                image = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        x = index * panel_size
        canvas.paste(image, (x, header))
        draw.text((x + 24, 20), label, fill=(255, 255, 255), font=title_font)
        row = metrics.get(label)
        if row is not None:
            draw.text(
                (x + 24, 72),
                f"PSNR {row.get('psnr_db'):.8f}  SSIM {row.get('ssim'):.8f}",
                fill=(220, 220, 220),
                font=metric_font,
            )
            lpips_resolution = row.get("lpips_resolution")
            suffix = f" @ {lpips_resolution}px" if lpips_resolution else ""
            draw.text(
                (x + 24, 112),
                f"LPIPS {row.get('lpips'):.8f}{suffix}" if row.get("lpips") is not None else "LPIPS unavailable",
                fill=(190, 190, 190),
                font=metric_font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, compress_level=6)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "vertices",
        "faces",
        "PSNR",
        "SSIM",
        "LPIPS",
        "metric_resolution",
        "LPIPS_resolution",
        "LPIPS_fallback",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="outputs/cross_tile_pbr_perstep_guided_cuda4_full_staged",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=4096)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--lpips-fallback-resolution", type=int, default=512)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else experiment_dir / "hd4096"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = experiment_dir / "canonical_4096.png"
    camera = _load_json(experiment_dir / "global_camera.json")
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    print(
        f"[cuda] device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    records: Dict[str, Dict[str, Any]] = {}
    render_paths: Dict[str, Path] = {}
    for variant, relative_mesh_path in VARIANTS:
        mesh_path = experiment_dir / relative_mesh_path
        variant_dir = output_dir / variant / "aligned_eval"
        render_result = _render_one(
            mesh_path=mesh_path,
            output_dir=variant_dir,
            reference_path=reference_path,
            camera=camera,
            envmap=envmap,
            args=args,
        )
        metric_result = _recompute_metrics(
            render_result=render_result,
            reference_path=reference_path,
            args=args,
        )
        merged = dict(render_result)
        merged.update(metric_result)
        _save_json(variant_dir / "metrics_4096.json", merged)
        records[variant] = merged
        render_paths[variant] = Path(str(render_result["render_png"]))
        print(
            f"[metrics4096] {variant}: PSNR={merged['psnr_db']:.8f} "
            f"SSIM={merged['ssim']:.8f} LPIPS={merged['lpips']}"
        )
        _empty_cuda_cache()
    del envmap
    _empty_cuda_cache()

    with Image.open(reference_path) as source:
        reference_copy = output_dir / "input_4096.png"
        source.convert("RGB").save(reference_copy)
    comparison_path = output_dir / "input_baseline_pure_HR_guided_comparison_4096.png"
    _save_comparison(
        reference_path=reference_copy,
        render_paths=render_paths,
        metrics=records,
        output_path=comparison_path,
        panel_size=int(args.render_resolution),
    )

    rows = []
    for variant, _ in VARIANTS:
        row = records[variant]
        rows.append(
            {
                "variant": variant,
                "vertices": row.get("decoder_vertices"),
                "faces": row.get("decoder_faces"),
                "PSNR": row.get("psnr_db"),
                "SSIM": row.get("ssim"),
                "LPIPS": row.get("lpips"),
                "metric_resolution": row.get("metric_resolution"),
                "LPIPS_resolution": row.get("lpips_resolution"),
                "LPIPS_fallback": row.get("lpips_fallback"),
            }
        )
    _write_csv(output_dir / "metrics_4096.csv", rows)
    summary = {
        "format": "pixal3d_render_only_4096_metrics_v1",
        "experiment_dir": str(experiment_dir),
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "render_resolution": int(args.render_resolution),
        "metric_resolution": int(args.metric_resolution),
        "render_ssaa": int(args.render_ssaa),
        "render_peel_layers": int(args.render_peel_layers),
        "reference": str(reference_path),
        "comparison_4096": str(comparison_path),
        "metrics_csv": str((output_dir / "metrics_4096.csv").resolve()),
        "variants": records,
        "table": rows,
    }
    _save_json(output_dir / "summary_4096.json", summary)
    print(f"[done] comparison={comparison_path}")
    print(f"[done] metrics={output_dir / 'metrics_4096.csv'}")


if __name__ == "__main__":
    main()
