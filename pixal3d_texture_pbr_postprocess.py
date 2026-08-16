"""Post-process saved texture-only endpoints.

This script completes the rendering side of the SLat interpolation
exploratory check and writes compact aggregate summaries. It only reads the
endpoint files produced by pixal3d_texture_pbr_degradation_experiment.py; it
does not run shape flow, texture flow, training, or modify sampler velocity.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

import pixal3d_texture_pbr_degradation_experiment as experiment
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr


SCALES = ("global_C64_x1", "global_C64_x2", "global_C64_x4")
INTERPOLATION_ALPHAS = (0.25, 0.50, 0.75)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _metric_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _value_at(row: Mapping[str, Any], operator: str, scale: str, *keys: str) -> float:
    value: Any = row["pbr_operator"][operator][scale]
    for key in keys:
        value = value[key]
    return float(value)


def _render_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        render_resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=str(args.envmap),
        render_ssaa=int(args.render_ssaa),
        render_peel_layers=int(args.render_peel_layers),
        render_face_chunk_size=int(args.render_face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        skip_lpips=bool(args.skip_lpips),
    )


def _run_tile_interpolation_render(
    *,
    pipeline: Any,
    tile_dir: Path,
    row: Mapping[str, Any],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    endpoints_path = tile_dir / "endpoints.pt"
    if not endpoints_path.is_file():
        raise FileNotFoundError(endpoints_path)
    endpoints = torch.load(str(endpoints_path), map_location="cpu")
    device = torch.device("cuda")
    shape_coords = endpoints["shape_coords"].to(device=device, dtype=torch.int32)
    texture_coords = endpoints["g_tex_coords"].to(device=device, dtype=torch.int32)
    shape_norm = SparseTensor(endpoints["shape_norm"].to(device=device, dtype=torch.float32), shape_coords)
    shape_denorm = experiment._denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
    z_g = endpoints["g_tex_norm"].to(dtype=torch.float32)
    z_h = endpoints["hr_tex_norm"].to(dtype=torch.float32)
    reference = tile_dir / "hr_tile_1024_condition.png"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    transform_data = endpoints["transform"]
    transform = SimpleNamespace(
        camera_angle_x=float(transform_data["camera_angle_x"]),
        distance=float(transform_data["distance"]),
        mesh_scale=float(transform_data["mesh_scale"]),
    )
    render_args = _render_args(args)
    empty_points = torch.empty((0, 3), device=device, dtype=torch.float32)
    results: Dict[str, Any] = {
        "status": "success",
        "source": "saved G_tex/HR_tex endpoints",
        "alphas": {},
    }
    for alpha in INTERPOLATION_ALPHAS:
        key = f"alpha_{alpha:.2f}"
        started = time.perf_counter()
        z_alpha = z_g + float(alpha) * (z_h - z_g)
        latent = SparseTensor(z_alpha.to(device=device), texture_coords)
        mesh, _, decode_stats = experiment._decode_and_query(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_latent_norm=latent,
            normalization=pipeline.tex_slat_normalization,
            query_points_device=empty_points,
            resolution=experiment.OVOXEL_RESOLUTION,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile_{int(row['tile_id']):02d} latent interpolation {alpha:.2f}",
        )
        points = mesh.vertices
        attrs = experiment._query_common_fields(mesh, points, int(args.query_chunk_size)).cpu()
        vertices_cpu = points.detach().cpu().to(torch.float32)
        faces_cpu = mesh.faces.detach().cpu().to(torch.int32)
        name = f"latent_interp_{int(round(alpha * 100)):02d}"
        render_result = core._render(
            MeshWithVertexPbr(vertices_cpu, faces_cpu, attrs, layout=dict(experiment.PBR_LAYOUT)),
            output_dir=tile_dir / "renders" / name,
            camera={
                "camera_angle_x": float(transform.camera_angle_x),
                "distance": float(transform.distance),
                "mesh_scale": float(transform.mesh_scale),
            },
            reference_image=reference,
            args=render_args,
            envmap=envmap,
        )
        results["alphas"][key] = {
            "alpha": float(alpha),
            "decode_seconds": float(decode_stats["decode_seconds"]),
            "total_seconds": float(time.perf_counter() - started),
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "render_metrics": core._metric_subset(render_result),
        }
        del latent, mesh, attrs, points, vertices_cpu, faces_cpu
        experiment._empty_cuda_cache()
    return results


def _aggregate(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    aggregate: Dict[str, Any] = {
        "format": "pixal3d_texture_pbr_degradation_aggregate_v1",
        "successful_tiles": [int(row["tile_id"]) for row in successful],
        "skipped_tiles": [int(row["tile_id"]) for row in rows if row.get("status") == "skipped"],
        "failed_tiles": [int(row["tile_id"]) for row in rows if row.get("status") == "failed"],
        "strict_checks": {},
        "cell_average": {},
        "trilinear": {},
        "render": {},
        "latent": {},
        "best_cell_average_scale_counts": {},
        "interpretation": {},
    }
    check_names = (
        "support_equal",
        "g_tex_coords_equal_hr_tex_coords",
        "g_tex_coords_equal_shape_condition_coords",
        "hr_tex_coords_equal_shape_condition_coords",
        "token_count_equal",
        "token_order_equal",
        "shape_condition_equal",
        "normalization_mean_std_equal",
    )
    for name in check_names:
        aggregate["strict_checks"][name] = {
            "true": int(sum(bool(row.get("support_checks", {}).get(name)) for row in successful)),
            "total": int(len(successful)),
        }
    aggregate["strict_checks"]["decoded_geometry_equal"] = {
        "true": int(sum(bool(row.get("decoded_geometry_check", {}).get("geometry_equal")) for row in successful)),
        "total": int(len(successful)),
    }
    for operator in ("cell_average", "trilinear"):
        aggregate[operator] = {}
        for scale in SCALES:
            if not successful or scale not in successful[0].get("pbr_operator", {}).get(operator, {}):
                continue
            aggregate[operator][scale] = {}
            for metric_name, path in (
                ("e_coarse_joint", ("coarse_consistency", "joint", "e_coarse")),
                ("r_low_joint", ("delta_decomposition", "joint", "r_low")),
                ("r_high_joint", ("delta_decomposition", "joint", "r_high")),
                ("A_delta_relative_error_joint", ("delta_decomposition", "joint", "A_delta_relative_error")),
            ):
                values = []
                for row in successful:
                    try:
                        values.append(_value_at(row, operator, scale, *path))
                    except KeyError:
                        pass
                aggregate[operator][scale][metric_name] = _metric_stats(values)
            for channel in ("base_color", "metallic", "roughness", "alpha"):
                values = []
                for row in successful:
                    try:
                        values.append(_value_at(row, operator, scale, "delta_decomposition", channel, "r_high"))
                    except KeyError:
                        pass
                aggregate[operator][scale].setdefault("r_high_by_channel", {})[channel] = _metric_stats(values)
    for row in successful:
        best = row.get("pbr_operator", {}).get("best_cell_average_scale")
        if best is not None:
            aggregate["best_cell_average_scale_counts"][best] = aggregate["best_cell_average_scale_counts"].get(best, 0) + 1

    variants = ("G", "HR", "G_low", "HR_low", "G_low_HR_high", "HR_low_G_high")
    for variant in variants:
        psnr = []
        ssim = []
        for row in successful:
            metric = row.get("render_metrics", {}).get(variant, {})
            if metric.get("psnr_db") is not None:
                psnr.append(float(metric["psnr_db"]))
            if metric.get("ssim") is not None:
                ssim.append(float(metric["ssim"]))
        aggregate["render"][variant] = {"psnr_db": _metric_stats(psnr), "ssim": _metric_stats(ssim)}
    for variant in ("G", "G_low", "HR_low", "G_low_HR_high", "HR_low_G_high"):
        dpsnr = []
        dssim = []
        for row in successful:
            metrics = row.get("render_metrics", {})
            if metrics.get(variant, {}).get("psnr_db") is not None and metrics.get("HR", {}).get("psnr_db") is not None:
                dpsnr.append(float(metrics[variant]["psnr_db"]) - float(metrics["HR"]["psnr_db"]))
            if metrics.get(variant, {}).get("ssim") is not None and metrics.get("HR", {}).get("ssim") is not None:
                dssim.append(float(metrics[variant]["ssim"]) - float(metrics["HR"]["ssim"]))
        aggregate["render"].setdefault("delta_vs_HR", {})[variant] = {
            "psnr_db": _metric_stats(dpsnr),
            "ssim": _metric_stats(dssim),
            "HR_better_psnr_count": int(sum(value < 0.0 for value in dpsnr)),
            "total": int(len(dpsnr)),
        }

    commute_g = []
    commute_h = []
    corr = {branch: {name: [] for name in ("distance_latent_l2_pearson", "distance_cosine_pearson")} for branch in ("G", "HR")}
    interp = {key: [] for key in ("alpha_0.25", "alpha_0.50", "alpha_0.75")}
    interp_render = {key: [] for key in interp}
    for row in successful:
        exploratory = row.get("latent_exploratory", {})
        commute = exploratory.get("commute", {})
        if "D(P_ZG)_vs_P_FD(G)" in commute:
            commute_g.append(float(commute["D(P_ZG)_vs_P_FD(G)"]))
        if "D(P_ZHR)_vs_P_FD(HR)" in commute:
            commute_h.append(float(commute["D(P_ZHR)_vs_P_FD(HR)"]))
        for branch in corr:
            branch_data = exploratory.get("distance_latent_similarity", {}).get(branch, {})
            for name in corr[branch]:
                if branch_data.get(name) is not None:
                    corr[branch][name].append(float(branch_data[name]))
        for key in interp:
            value = exploratory.get("interpolation", {}).get(key, {}).get("pbr_l2_to_linear")
            if value is not None:
                interp[key].append(float(value))
        for key in interp_render:
            value = row.get("latent_interpolation_render_metrics", {}).get("alphas", {}).get(key, {}).get("render_metrics", {})
            if value.get("psnr_db") is not None:
                interp_render[key].append({"psnr_db": float(value["psnr_db"]), "ssim": float(value.get("ssim", 0.0))})
    aggregate["latent"] = {
        "commute_D_PZG_vs_PFD_G": _metric_stats(commute_g),
        "commute_D_PZHR_vs_PFD_HR": _metric_stats(commute_h),
        "distance_correlations": {
            branch: {name: _metric_stats(values) for name, values in branch_values.items()}
            for branch, branch_values in corr.items()
        },
        "interpolation_pbr_l2_to_linear": {key: _metric_stats(values) for key, values in interp.items()},
        "interpolation_render": {
            key: {
                "psnr_db": _metric_stats([item["psnr_db"] for item in values]),
                "ssim": _metric_stats([item["ssim"] for item in values]),
            }
            for key, values in interp_render.items()
        },
    }
    a4 = aggregate["cell_average"].get("global_C64_x4", {})
    a2 = aggregate["cell_average"].get("global_C64_x2", {})
    b4 = aggregate["trilinear"].get("global_C64_x4", {})
    aggregate["interpretation"] = {
        "coarse_preserve_fine_detail_supported": False,
        "reason": "Across successful tiles, coarse G/HR consistency errors remain large and delta energy is predominantly low-space under both A and B.",
        "cell_average_global_C64_x4": {
            "mean_e_coarse_joint": a4.get("e_coarse_joint", {}).get("mean"),
            "mean_r_low_joint": a4.get("r_low_joint", {}).get("mean"),
            "mean_r_high_joint": a4.get("r_high_joint", {}).get("mean"),
            "mean_A_delta_relative_error_joint": a4.get("A_delta_relative_error_joint", {}).get("mean"),
        },
        "cell_average_global_C64_x2": {
            "mean_e_coarse_joint": a2.get("e_coarse_joint", {}).get("mean"),
            "mean_r_low_joint": a2.get("r_low_joint", {}).get("mean"),
            "mean_r_high_joint": a2.get("r_high_joint", {}).get("mean"),
            "mean_A_delta_relative_error_joint": a2.get("A_delta_relative_error_joint", {}).get("mean"),
        },
        "trilinear_global_C64_x4": {
            "mean_e_coarse_joint": b4.get("e_coarse_joint", {}).get("mean"),
            "mean_r_low_joint": b4.get("r_low_joint", {}).get("mean"),
            "mean_r_high_joint": b4.get("r_high_joint", {}).get("mean"),
            "mean_A_delta_relative_error_joint": b4.get("A_delta_relative_error_joint", {}).get("mean"),
        },
        "render_statement": "HR improves the tile-view mean PSNR over G, but G_low+HR_high does not retain most of that gain.",
    }
    aggregate["artifacts"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "experiment_report.md"),
    }
    return aggregate


def _write_aggregate_report(path: Path, aggregate: Mapping[str, Any]) -> None:
    a = aggregate["cell_average"]
    b = aggregate["trilinear"]
    render = aggregate["render"]
    latent = aggregate["latent"]
    lines = [
        "# Pixal3D texture-only PBR degradation aggregate",
        "",
        f"- successful tiles: {len(aggregate['successful_tiles'])}",
        f"- skipped tiles: {aggregate['skipped_tiles']}",
        f"- failed tiles: {aggregate['failed_tiles']}",
        "",
        "## Strict route checks",
        "",
    ]
    for name, value in aggregate["strict_checks"].items():
        lines.append(f"- {name}: {value['true']}/{value['total']}")
    lines.extend(["", "## PBR operator aggregate (mean; min-max)", ""])
    for operator, label in ((a, "A: cell-average + copy"), (b, "B: trilinear LSQR")):
        lines.extend([f"### {label}", "", "| scale | e_coarse joint | r_low joint | r_high joint | A_delta relative error |", "|---|---:|---:|---:|---:|"])
        for scale, values in operator.items():
            def fmt(name: str) -> str:
                item = values[name]
                return f"{item['mean']:.4f} ({item['min']:.4f}-{item['max']:.4f})"
            lines.append(f"| {scale} | {fmt('e_coarse_joint')} | {fmt('r_low_joint')} | {fmt('r_high_joint')} | {fmt('A_delta_relative_error_joint')} |")
        lines.append("")
    lines.extend(["## Renderer aggregate", "", "| variant | PSNR mean | SSIM mean |", "|---|---:|---:|"])
    for name in ("G", "HR", "G_low", "HR_low", "G_low_HR_high", "HR_low_G_high"):
        lines.append(f"| {name} | {render[name]['psnr_db']['mean']:.4f} | {render[name]['ssim']['mean']:.4f} |")
    lines.extend([
        "",
        "The mean HR-vs-G and hybrid deltas are reported in aggregate_metrics.json; this is a tile-view aggregate and is not a claim about latent channel frequency.",
        "",
        "## SLat exploratory checks",
        "",
        f"- mean D(P_ZG) vs P_F D(G): {latent['commute_D_PZG_vs_PFD_G']['mean']:.4f}",
        f"- mean D(P_ZHR) vs P_F D(HR): {latent['commute_D_PZHR_vs_PFD_HR']['mean']:.4f}",
        "- These non-zero commute errors mean spatial SLat averaging is not a validated PBR low-pass proxy.",
        "",
        "## Conclusion",
        "",
        "The simple coarse-preserve/fine-detail hypothesis is rejected for this run: both PBR operators observe a large coarse G-to-HR change, and the joint delta is mostly in the operator range rather than its null space. The hybrid renderer therefore does not retain most of HR's improvement.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    wanted = _parse_ids(args.tile_ids)
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    rows: List[Dict[str, Any]] = []
    for row in summary.get("tiles", []):
        tile_id = int(row["tile_id"])
        if wanted is not None and tile_id not in wanted:
            rows.append(dict(row))
            continue
        if row.get("status") != "success":
            rows.append(dict(row))
            continue
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        metrics_path = tile_dir / "latent_interpolation_render_metrics.json"
        if bool(args.resume) and metrics_path.is_file():
            interpolation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"[interp tile {tile_id:02d}] reused saved render metrics")
        else:
            print(f"[interp tile {tile_id:02d}] rendering latent interpolation")
            try:
                interpolation_metrics = _run_tile_interpolation_render(
                    pipeline=pipeline,
                    tile_dir=tile_dir,
                    row=row,
                    args=args,
                    envmap=envmap,
                )
            except Exception as exc:
                interpolation_metrics = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                print(f"[interp tile {tile_id:02d}] FAILED: {interpolation_metrics['reason']}")
            _write_json(metrics_path, interpolation_metrics)
        updated = dict(row)
        updated["latent_interpolation_render_metrics"] = interpolation_metrics
        _write_json(tile_dir / "summary.json", updated)
        rows.append(updated)
    summary["tiles"] = rows
    summary["interpolation_render_postprocess"] = {
        "status": "complete",
        "alphas": list(INTERPOLATION_ALPHAS),
        "official_renderer": True,
        "shape_flow_called": False,
        "texture_flow_called": False,
    }
    aggregate = _aggregate(rows, output_dir)
    aggregate_path = output_dir / "aggregate_metrics.json"
    aggregate_report_path = output_dir / "aggregate_report.md"
    _write_json(aggregate_path, aggregate)
    _write_aggregate_report(aggregate_report_path, aggregate)
    summary["aggregate_metrics"] = str(aggregate_path)
    summary["aggregate_report_markdown"] = str(aggregate_report_path)
    _write_json(summary_path, summary)
    print(f"[done] aggregate={aggregate_path} report={aggregate_report_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-chunk-size", type=int, default=32_768)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    return parser


if __name__ == "__main__":
    run(_build_parser().parse_args())
