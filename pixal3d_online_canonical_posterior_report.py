#!/usr/bin/env python3
"""Aggregate the online canonical-posterior experiment and make plots.

This script is deliberately evaluation-only.  It reads completed run summaries,
never model latents, and writes a deterministic cross-configuration report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIG_NAMES = (
    "local_hr_baseline",
    "old_anchor",
    "shape_posterior",
    "texture_posterior",
    "joint_posterior_fixed",
    "joint_posterior_per_step",
)
CONFIG_LABELS = {
    "local_hr_baseline": "Local HR",
    "old_anchor": "Old anchor",
    "shape_posterior": "Shape post.",
    "texture_posterior": "Texture post.",
    "joint_posterior_fixed": "Joint fixed",
    "joint_posterior_per_step": "Joint/step",
}
EVIDENCE_DIRS = {
    "texture": "ablations/texture_evidence_texture",
    "texture_global_shape": "ablations/texture_evidence_global_shape",
    "texture_fused_shape": "joint_posterior_per_step",
}
EVIDENCE_LABELS = {
    "texture": "Texture",
    "texture_global_shape": "Texture + global shape",
    "texture_fused_shape": "Texture + fused shape",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _flow_digest(flow: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(flow, Mapping):
        return None
    steps = flow.get("step_statistics", [])
    if not isinstance(steps, Sequence) or not steps:
        return {
            "latent": flow.get("latent"),
            "fusion_kind": flow.get("fusion_kind"),
            "steps": flow.get("steps"),
        }
    spectra = [
        _nested(step, "cca", "canonical_correlations")
        for step in steps
        if isinstance(step, Mapping)
        and isinstance(_nested(step, "cca", "canonical_correlations"), list)
    ]
    top_rho = [
        float(spectrum[0]) for spectrum in spectra if len(spectrum) > 0
    ]
    correction = [
        _finite(step.get("posterior_correction_rms_token_weighted"))
        for step in steps
        if isinstance(step, Mapping)
    ]
    correction_ratio = [
        _finite(step.get("correction_over_hr_norm_token_weighted"))
        for step in steps
        if isinstance(step, Mapping)
    ]
    residual_errors = [
        float(step.get("matched_clean_residual_max_error", 0.0))
        for step in steps
        if isinstance(step, Mapping)
    ]
    back_errors = [
        float(step.get("canonical_back_transform_max_error", 0.0))
        for step in steps
        if isinstance(step, Mapping)
    ]
    return {
        "latent": flow.get("latent"),
        "fusion_kind": flow.get("fusion_kind"),
        "time_mode": flow.get("time_mode"),
        "steps": flow.get("steps"),
        "tiles": flow.get("tiles"),
        "tokens": flow.get("tokens"),
        "elapsed_seconds": flow.get("elapsed_seconds"),
        "cca_spectra": spectra,
        "top_canonical_correlation_by_step": top_rho,
        "posterior_correction_rms_by_step": correction,
        "correction_over_hr_norm_by_step": correction_ratio,
        "matched_clean_residual_max_error": max(residual_errors, default=0.0),
        "canonical_back_transform_max_error": max(back_errors, default=0.0),
    }


def _render_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(row, "evaluation", "render_metrics")
    return value if isinstance(value, Mapping) else {}


def _geometry_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(row, "evaluation", "geometry_diagnostics")
    return value if isinstance(value, Mapping) else {}


def _material_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(row, "evaluation", "material_diagnostics")
    return value if isinstance(value, Mapping) else {}


def _delta(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
    keys: Iterable[str],
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for key in keys:
        left = _finite(current.get(key))
        right = _finite(reference.get(key))
        result[key] = (
            None if left is None or right is None else float(left - right)
        )
    return result


def _configuration_digest(
    row: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    global_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    render = dict(_render_metrics(row))
    geometry = dict(_geometry_metrics(row))
    material = dict(_material_metrics(row))
    return {
        "status": "complete" if row.get("evaluation") else "flow_only",
        "summary": row.get("output_dir"),
        "seed": row.get("seed"),
        "cuda_device": row.get("cuda_device"),
        "configuration": row.get("configuration"),
        "generation_seconds": row.get("generation_seconds"),
        "successful_tiles": row.get("successful_tiles"),
        "skipped_tiles": row.get("skipped_tiles"),
        "failed_tiles": row.get("failed_tiles"),
        "shape_metrics": _flow_digest(row.get("shape_flow")),
        "texture_metrics": _flow_digest(row.get("texture_flow")),
        "render_metrics": render,
        "multiview_paths": row.get("multiview_paths", []),
        "geometry_metrics": geometry,
        "material_metrics": material,
        "render_deltas": {
            "vs_global": _delta(
                render, global_metrics, ("psnr_db", "ssim", "lpips")
            ),
            **{
                f"vs_{name}": _delta(
                    render,
                    _render_metrics(reference),
                    ("psnr_db", "ssim", "lpips"),
                )
                for name, reference in references.items()
            },
        },
    }


def _scalar_diagnostics(row: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    render = _render_metrics(row)
    geometry = _geometry_metrics(row)
    material = _material_metrics(row)
    return {
        "psnr_db": _finite(render.get("psnr_db")),
        "ssim": _finite(render.get("ssim")),
        "lpips": _finite(render.get("lpips")),
        "connected_components": _finite(
            geometry.get("connected_components")
        ),
        "backside_component_count": _finite(
            geometry.get("backside_component_count")
        ),
        "overlap_chamfer_l1_object": _finite(
            _nested(geometry, "overlap", "chamfer_l1_object")
        ),
        "overlap_normal_consistency": _finite(
            _nested(geometry, "overlap", "normal_consistency_absolute")
        ),
        "low_frequency_chamfer_to_global": _finite(
            _nested(
                geometry,
                "low_frequency_chamfer_to_global",
                "chamfer_l1_object",
            )
        ),
        "overlap_pbr_latent_rmse": _finite(
            material.get("overlap_pbr_latent_consistency_rmse")
        ),
        "overlap_base_color_difference": _finite(
            material.get("overlap_base_color_difference")
        ),
        "overlap_roughness_difference": _finite(
            material.get("overlap_roughness_difference")
        ),
        "overlap_metallic_difference": _finite(
            material.get("overlap_metallic_difference")
        ),
        "multiview_material_flicker": _finite(
            _nested(
                material,
                "multiview_material_flicker_proxy",
                "scalar_rms",
            )
        ),
    }


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    array = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "std_population": float(array.std()) if array.size else None,
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "values": [float(value) for value in array],
    }


def _multi_seed_digest(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scalar_rows = [_scalar_diagnostics(row) for row in rows]
    keys = scalar_rows[0].keys() if scalar_rows else ()
    return {
        "seeds": [int(row["seed"]) for row in rows],
        "runs": [
            {
                "seed": int(row["seed"]),
                "summary": row.get("output_dir"),
                **scalars,
            }
            for row, scalars in zip(rows, scalar_rows)
        ],
        "aggregate": {
            key: _mean_std([row[key] for row in scalar_rows])
            for key in keys
        },
    }


def _evidence_digest(
    evidence_rows: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    variants: Dict[str, Any] = {}
    for name, row in evidence_rows.items():
        flow = row.get("texture_flow")
        steps = (
            flow.get("step_statistics", [])
            if isinstance(flow, Mapping)
            else []
        )
        selected_r2 = [
            _finite(
                _nested(step, "self_consistency", "selected_predictor",
                        "r2_against_mean")
            )
            for step in steps
        ]
        texture_r2 = [
            _finite(
                _nested(step, "self_consistency", "texture_only",
                        "r2_against_mean")
            )
            for step in steps
        ]
        r2_gain = [
            _finite(
                _nested(
                    step,
                    "self_consistency",
                    "selected_minus_texture_only_r2",
                )
            )
            for step in steps
        ]
        selected_rho = [
            _finite(
                (
                    step.get("selected_predictor_canonical_correlations")
                    or [None]
                )[0]
            )
            for step in steps
        ]
        variants[name] = {
            "summary": row.get("output_dir"),
            "steps": len(steps),
            "selected_predictor_r2_by_step": selected_r2,
            "texture_only_r2_by_step": texture_r2,
            "selected_minus_texture_only_r2_by_step": r2_gain,
            "top_canonical_correlation_by_step": selected_rho,
            "selected_predictor_r2": _mean_std(selected_r2),
            "texture_only_r2": _mean_std(texture_r2),
            "selected_minus_texture_only_r2": _mean_std(r2_gain),
        }
    fused = _nested(
        variants,
        "texture_fused_shape",
        "selected_predictor_r2",
        "mean",
    )
    texture = _nested(
        variants, "texture", "selected_predictor_r2", "mean"
    )
    global_shape = _nested(
        variants,
        "texture_global_shape",
        "selected_predictor_r2",
        "mean",
    )
    return {
        "variants": variants,
        "fused_shape_minus_texture_only_mean_r2": (
            None
            if fused is None or texture is None
            else float(fused - texture)
        ),
        "fused_shape_minus_global_shape_mean_r2": (
            None
            if fused is None or global_shape is None
            else float(fused - global_shape)
        ),
        "scope": (
            "flow-level online self-consistency; only the designated main "
            "variant texture_fused_shape was decoded and rendered"
        ),
    }


def _residual_energy_digest(
    diagnostic_root: Path,
) -> Optional[Dict[str, Any]]:
    latent_rows: Dict[str, Any] = {}
    for latent in ("shape", "texture"):
        path = diagnostic_root / f"{latent}_summary.json"
        row = _read_json(path)
        if row is None:
            return None
        by_seed = _nested(row, "time_stability", "by_seed")
        if not isinstance(by_seed, Mapping):
            return None
        per_step: Dict[str, Any] = {}
        step_names = sorted(
            {
                str(step)
                for seed_rows in by_seed.values()
                if isinstance(seed_rows, Mapping)
                for step in seed_rows
            },
            key=int,
        )
        for step_name in step_names:
            per_mode: Dict[str, Any] = {}
            for mode in ("1", "2", "4", "8", "16", "24", "32"):
                values = [
                    _finite(
                        _nested(
                            seed_rows.get(step_name, {}),
                            "matched_hr_residual_shared_energy",
                            mode,
                            "mean",
                        )
                    )
                    for seed_rows in by_seed.values()
                    if isinstance(seed_rows, Mapping)
                ]
                per_mode[mode] = _mean_std(values)
            per_step[step_name] = per_mode
        latent_rows[latent] = {
            "summary": str(path.resolve()),
            "seeds": sorted(int(seed) for seed in by_seed),
            "by_step_across_seed": per_step,
        }
    return {
        "status": "complete",
        "projection": "delta_centered @ W_R",
        "right_weight_definition": "W_R = C_RR^{-1/2} V",
        "mode_counts": [1, 2, 4, 8, 16, 24, 32],
        "shape": latent_rows["shape"],
        "texture": latent_rows["texture"],
        "plot": str(
            (
                diagnostic_root
                / "plots"
                / "hr_residual_shared_energy_by_step.png"
            ).resolve()
        ),
    }


def _weighting_digest(
    token_row: Mapping[str, Any],
    tile_row: Mapping[str, Any],
) -> Dict[str, Any]:
    variants: Dict[str, Any] = {}
    for weighting, row in (("token", token_row), ("tile", tile_row)):
        latent_rows: Dict[str, Any] = {}
        for latent in ("shape", "texture"):
            flow = row.get(f"{latent}_flow", {})
            steps = (
                flow.get("step_statistics", [])
                if isinstance(flow, Mapping)
                else []
            )
            top_rho = [
                _finite(
                    (
                        _nested(step, "cca", "canonical_correlations")
                        or [None]
                    )[0]
                )
                for step in steps
            ]
            correction_ratio = [
                _finite(step.get("correction_over_hr_norm_token_weighted"))
                for step in steps
            ]
            residual_errors = [
                _finite(step.get("matched_clean_residual_max_error"))
                for step in steps
            ]
            back_errors = [
                _finite(step.get("canonical_back_transform_max_error"))
                for step in steps
            ]
            latent_rows[latent] = {
                "top_canonical_correlation_by_step": top_rho,
                "correction_over_hr_norm_by_step": correction_ratio,
                "top_canonical_correlation": _mean_std(top_rho),
                "correction_over_hr_norm": _mean_std(correction_ratio),
                "matched_clean_residual_max_error": max(
                    value
                    for value in residual_errors
                    if value is not None
                ),
                "canonical_back_transform_max_error": max(
                    value for value in back_errors if value is not None
                ),
            }
        variants[weighting] = {
            "summary": row.get("output_dir"),
            "shape": latent_rows["shape"],
            "texture": latent_rows["texture"],
        }
    return {
        "variants": variants,
        "tile_minus_token_mean_top_rho": {
            latent: float(
                _nested(
                    variants,
                    "tile",
                    latent,
                    "top_canonical_correlation",
                    "mean",
                )
                - _nested(
                    variants,
                    "token",
                    latent,
                    "top_canonical_correlation",
                    "mean",
                )
            )
            for latent in ("shape", "texture")
        },
        "scope": (
            "flow-level covariance-weighting ablation; main experiments use "
            "pooled token maximum-likelihood weighting"
        ),
    }


def _style_axes(axes: Iterable[Any]) -> None:
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)


def _plot_configuration_metrics(
    plot_dir: Path, configs: Mapping[str, Mapping[str, Any]]
) -> None:
    names = [name for name in CONFIG_NAMES if name in configs]
    labels = [CONFIG_LABELS[name] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, key, title in zip(
        axes,
        ("psnr_db", "ssim", "lpips"),
        ("Input-view PSNR ↑", "Input-view SSIM ↑", "Input-view LPIPS ↓"),
    ):
        values = [_render_metrics(configs[name]).get(key) for name in names]
        axis.bar(labels, values, color="#4C78A8")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
    _style_axes(axes)
    fig.tight_layout()
    fig.savefig(plot_dir / "configuration_render_metrics.png", dpi=180)
    plt.close(fig)


def _plot_consistency_metrics(
    plot_dir: Path, configs: Mapping[str, Mapping[str, Any]]
) -> None:
    names = [name for name in CONFIG_NAMES if name in configs]
    labels = [CONFIG_LABELS[name] for name in names]
    scalars = [_scalar_diagnostics(configs[name]) for name in names]
    keys = (
        "connected_components",
        "backside_component_count",
        "overlap_normal_consistency",
        "overlap_pbr_latent_rmse",
    )
    titles = (
        "Connected components ↓",
        "Backside components ↓",
        "Overlap normal consistency ↑",
        "Overlap PBR latent RMSE ↓",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, key, title in zip(axes.flat, keys, titles):
        values = [row[key] for row in scalars]
        axis.bar(labels, values, color="#59A14F")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        if key in ("connected_components", "backside_component_count"):
            axis.set_yscale("symlog", linthresh=1.0)
    _style_axes(axes.flat)
    fig.tight_layout()
    fig.savefig(plot_dir / "configuration_consistency_metrics.png", dpi=180)
    plt.close(fig)


def _plot_cca_dynamics(plot_dir: Path, main: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, latent in enumerate(("shape", "texture")):
        flow = main.get(f"{latent}_flow", {})
        steps = flow.get("step_statistics", [])
        x = [int(step["step"]) for step in steps]
        rho = [
            float(step["cca"]["canonical_correlations"][0])
            for step in steps
        ]
        correction = [
            float(step["correction_over_hr_norm_token_weighted"])
            for step in steps
        ]
        axes[0, column].plot(x, rho, marker="o", color="#4C78A8")
        axes[0, column].set_title(f"{latent.title()} top canonical ρ")
        axes[1, column].plot(x, correction, marker="o", color="#E45756")
        axes[1, column].set_title(f"{latent.title()} correction / HR norm")
        axes[1, column].set_xlabel("Flow step")
    _style_axes(axes.flat)
    fig.tight_layout()
    fig.savefig(plot_dir / "cca_dynamics.png", dpi=180)
    plt.close(fig)


def _plot_multi_seed(plot_dir: Path, multi_seed: Mapping[str, Any]) -> None:
    aggregate = multi_seed["aggregate"]
    keys = (
        "psnr_db",
        "ssim",
        "lpips",
        "overlap_normal_consistency",
        "overlap_pbr_latent_rmse",
        "backside_component_count",
    )
    titles = (
        "PSNR ↑",
        "SSIM ↑",
        "LPIPS ↓",
        "Overlap normals ↑",
        "PBR latent RMSE ↓",
        "Backside components ↓",
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, key, title in zip(axes.flat, keys, titles):
        stats = aggregate[key]
        axis.errorbar(
            [0],
            [stats["mean"]],
            yerr=[stats["std_population"]],
            marker="o",
            capsize=6,
            color="#B279A2",
        )
        axis.scatter(
            np.linspace(-0.08, 0.08, len(stats["values"])),
            stats["values"],
            s=24,
            color="#4C78A8",
            zorder=3,
        )
        axis.set_xlim(-0.3, 0.3)
        axis.set_xticks([])
        axis.set_title(title)
    _style_axes(axes.flat)
    fig.suptitle("Joint posterior per-step, seeds 42–45 (mean ± population SD)")
    fig.tight_layout()
    fig.savefig(plot_dir / "multiseed_joint_metrics.png", dpi=180)
    plt.close(fig)


def _plot_evidence_ablation(
    plot_dir: Path, ablation: Mapping[str, Any]
) -> None:
    variants = ablation["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in EVIDENCE_DIRS:
        row = variants.get(name)
        if not row:
            continue
        label = EVIDENCE_LABELS[name]
        x = np.arange(len(row["top_canonical_correlation_by_step"]))
        axes[0].plot(
            x,
            row["top_canonical_correlation_by_step"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            x,
            row["selected_predictor_r2_by_step"],
            marker="o",
            label=label,
        )
    axes[0].set_title("Texture predictor top canonical ρ ↑")
    axes[1].set_title("Online self-consistency R² ↑")
    for axis in axes:
        axis.set_xlabel("Texture flow step")
        axis.legend(fontsize=8)
    _style_axes(axes)
    fig.tight_layout()
    fig.savefig(plot_dir / "texture_evidence_ablation.png", dpi=180)
    plt.close(fig)


def _plot_weighting_ablation(
    plot_dir: Path, ablation: Mapping[str, Any]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"token": "#4C78A8", "tile": "#F28E2B"}
    for weighting, row in ablation["variants"].items():
        for column, latent in enumerate(("shape", "texture")):
            latent_row = row[latent]
            rho = latent_row["top_canonical_correlation_by_step"]
            correction = latent_row["correction_over_hr_norm_by_step"]
            x = np.arange(len(rho))
            axes[0, column].plot(
                x,
                rho,
                marker="o",
                label=weighting,
                color=colors[weighting],
            )
            axes[1, column].plot(
                x,
                correction,
                marker="o",
                label=weighting,
                color=colors[weighting],
            )
    for column, latent in enumerate(("Shape", "Texture")):
        axes[0, column].set_title(f"{latent} top canonical ρ")
        axes[1, column].set_title(f"{latent} correction / HR norm")
        axes[1, column].set_xlabel("Flow step")
    for axis in axes.flat:
        axis.legend()
    _style_axes(axes.flat)
    fig.tight_layout()
    fig.savefig(plot_dir / "cca_weighting_ablation.png", dpi=180)
    plt.close(fig)


def _pct_delta(current: float, reference: float) -> float:
    return 100.0 * (current - reference) / reference


def _report_markdown(
    suite_root: Path,
    configs: Mapping[str, Mapping[str, Any]],
    multi_seed: Mapping[str, Any],
    ablation: Mapping[str, Any],
    weighting_ablation: Mapping[str, Any],
    residual_energy: Optional[Mapping[str, Any]],
    claims: Mapping[str, Any],
) -> str:
    local = _scalar_diagnostics(configs["local_hr_baseline"])
    old = _scalar_diagnostics(configs["old_anchor"])
    joint = _scalar_diagnostics(configs["joint_posterior_per_step"])
    aggregate = multi_seed["aggregate"]
    lines = [
        "# Online Test-Time Canonical Posterior Fusion 实验报告",
        "",
        "## 范围与合规性",
        "",
        (
            "本实验在单对象的 48 个有效 tile 上完成六个配置，并对主方法运行 "
            "seeds 42、43、44、45。所有正式生成均使用 CUDA 4；方法为纯测试时、"
            "逐步在线拟合，没有梯度更新，也未读取诊断实验的 CCA/ridge map 或"
            "未来 local endpoint。tile 6 因 global O-Voxel 投影不足按基线规则显式"
            "跳过，其余 48 个 tile 成功，0 个静默失败。"
        ),
        "",
        "## 单对象 seed 42 结果",
        "",
        "| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Components ↓ | Backside ↓ | Normal ↑ | PBR RMSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIG_NAMES:
        row = _scalar_diagnostics(configs[name])
        lines.append(
            f"| {CONFIG_LABELS[name]} | {row['psnr_db']:.3f} | "
            f"{row['ssim']:.4f} | {row['lpips']:.4f} | "
            f"{int(row['connected_components'])} | "
            f"{int(row['backside_component_count'])} | "
            f"{row['overlap_normal_consistency']:.4f} | "
            f"{row['overlap_pbr_latent_rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 数据支持的结论",
            "",
            (
                f"- 相对 local HR，joint per-step 的输入视图 PSNR "
                f"{joint['psnr_db'] - local['psnr_db']:+.3f} dB、SSIM "
                f"{joint['ssim'] - local['ssim']:+.4f}、LPIPS "
                f"{joint['lpips'] - local['lpips']:+.4f}。因此输入视图质量"
                "发生明显退化，不能判定总体成功。"
            ),
            (
                f"- 几何一致性明显改善：connected components 从 "
                f"{int(local['connected_components'])} 降到 "
                f"{int(joint['connected_components'])}，backside components 从 "
                f"{int(local['backside_component_count'])} 降到 "
                f"{int(joint['backside_component_count'])}，overlap normal 从 "
                f"{local['overlap_normal_consistency']:.4f} 升到 "
                f"{joint['overlap_normal_consistency']:.4f}。"
            ),
            (
                f"- 材质缝一致性部分改善：PBR latent RMSE "
                f"{_pct_delta(joint['overlap_pbr_latent_rmse'], local['overlap_pbr_latent_rmse']):+.1f}%，"
                f"base-color difference "
                f"{_pct_delta(joint['overlap_base_color_difference'], local['overlap_base_color_difference']):+.1f}%，"
                f"metallic difference "
                f"{_pct_delta(joint['overlap_metallic_difference'], local['overlap_metallic_difference']):+.1f}%；"
                f"但 roughness difference "
                f"{_pct_delta(joint['overlap_roughness_difference'], local['overlap_roughness_difference']):+.1f}%。"
            ),
            (
                f"- 相对 old anchor，joint per-step 输入视图 PSNR "
                f"{joint['psnr_db'] - old['psnr_db']:+.3f} dB，LPIPS "
                f"{joint['lpips'] - old['lpips']:+.4f}；但 old anchor 的 components、"
                "overlap normal 和 PBR RMSE 更好。因此新方法只在保真度上明显优于"
                "旧公式，不能声称全面优于旧公式。"
            ),
            "",
            "## 多随机种子",
            "",
            (
                f"Joint per-step 的四 seed 平均为 PSNR "
                f"{aggregate['psnr_db']['mean']:.3f}±"
                f"{aggregate['psnr_db']['std_population']:.3f} dB，SSIM "
                f"{aggregate['ssim']['mean']:.4f}±"
                f"{aggregate['ssim']['std_population']:.4f}，LPIPS "
                f"{aggregate['lpips']['mean']:.4f}±"
                f"{aggregate['lpips']['std_population']:.4f}。结果稳定，但稳定地"
                "低于 local HR 的输入视图质量。"
            ),
            "",
            "## Texture predictor 消融",
            "",
            (
                "三种接口均已实际运行流级在线消融；主方案 "
                "`texture_fused_shape` 才执行完整 decode/render。其平均 online "
                f"self-consistency R² 相对 texture-only 为 "
                f"{ablation['fused_shape_minus_texture_only_mean_r2']:+.4f}，"
                "相对 texture+global-shape 为 "
                f"{ablation['fused_shape_minus_global_shape_mean_r2']:+.4f}。"
            ),
            "",
            "## Covariance weighting 消融",
            "",
            (
                "主实验使用 pooled token maximum-likelihood covariance；额外的 "
                "tile-equal 流级消融也已实际运行。tile-equal 相对 token 的全步骤"
                "平均 top canonical ρ 变化为：shape "
                f"{weighting_ablation['tile_minus_token_mean_top_rho']['shape']:+.4f}，"
                "texture "
                f"{weighting_ablation['tile_minus_token_mean_top_rho']['texture']:+.4f}。"
                "该消融未参与主方案选择，也未使用最终渲染指标。"
            ),
        ]
    )
    if residual_energy is not None:
        shape_steps = residual_energy["shape"]["by_step_across_seed"]
        texture_steps = residual_energy["texture"]["by_step_across_seed"]
        last_shape = str(max(int(step) for step in shape_steps))
        last_texture = str(max(int(step) for step in texture_steps))
        lines.extend(
            [
                "",
                "## Corrected residual-energy 分析",
                "",
                (
                    "已按 `delta_centered @ W_R`（"
                    "`W_R = C_RR^{-1/2} V`）重算 k=1/2/4/8/16/24/32。"
                    "以 k=8 为例，shape 跨 seed shared-energy 均值从第 0 步 "
                    f"{shape_steps['0']['8']['mean']:.4f} 变为末步 "
                    f"{shape_steps[last_shape]['8']['mean']:.4f}；texture 从 "
                    f"{texture_steps['0']['8']['mean']:.4f} 变为 "
                    f"{texture_steps[last_texture]['8']['mean']:.4f}。"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            (
                "**本单对象 proof-of-concept 未达到完整成功标准。** 在线 canonical "
                "posterior 在几何连通性、背面异常和多数材质缝指标上提供了强证据，"
                "但输入视图保真度明显下降，且不在所有几何/材质指标上优于 old "
                "anchor。该结果支持方法机制有效，不支持宣称整体生成质量已提升。"
            ),
            "",
            "## 产物",
            "",
            "- `summary.json`：六配置、四 seed、三证据接口的机器可读汇总。",
            "- `plots/`：输入视图、几何/材质、多 seed、CCA 动态及证据消融图。",
            "- `statistics/`：主方法逐步 shape/texture CCA 统计。",
            "- 各配置 `multiview/`：固定 yaw 的六视角渲染。",
            "",
            (
                f"成功判定字段：`method_success_declared = "
                f"{str(claims['method_success_declared']).lower()}`。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run(suite_root: Path, diagnostic_root: Path) -> None:
    suite_root = suite_root.expanduser().resolve()
    diagnostic_root = diagnostic_root.expanduser().resolve()
    configs = {
        name: row
        for name in CONFIG_NAMES
        if (row := _read_json(suite_root / name / "summary.json")) is not None
    }
    missing = [name for name in CONFIG_NAMES if name not in configs]
    if missing:
        raise RuntimeError(f"missing completed configurations: {missing}")

    local = configs["local_hr_baseline"]
    global_metrics = _nested(local, "evaluation", "global_render_metrics")
    if not isinstance(global_metrics, Mapping):
        raise RuntimeError("global render metrics are missing")
    references = {
        "local_hr_baseline": configs["local_hr_baseline"],
        "old_anchor": configs["old_anchor"],
    }
    config_digest = {
        name: _configuration_digest(
            row, references=references, global_metrics=global_metrics
        )
        for name, row in configs.items()
    }

    seed_rows = [configs["joint_posterior_per_step"]]
    for seed in (43, 44, 45):
        path = (
            suite_root
            / "seeds"
            / f"seed_{seed}"
            / "joint_posterior_per_step"
            / "summary.json"
        )
        row = _read_json(path)
        if row is None:
            raise RuntimeError(f"missing multi-seed result: {path}")
        seed_rows.append(row)
    seed_rows.sort(key=lambda row: int(row["seed"]))
    multi_seed = _multi_seed_digest(seed_rows)

    evidence_rows: Dict[str, Mapping[str, Any]] = {}
    for name, relative in EVIDENCE_DIRS.items():
        row = _read_json(suite_root / relative / "summary.json")
        if row is None:
            raise RuntimeError(f"missing texture evidence ablation: {relative}")
        evidence_rows[name] = row
    evidence = _evidence_digest(evidence_rows)
    tile_weighting_row = _read_json(
        suite_root / "ablations" / "cca_weighting_tile" / "summary.json"
    )
    if tile_weighting_row is None:
        raise RuntimeError("missing tile-equal covariance ablation")
    weighting_ablation = _weighting_digest(
        configs["joint_posterior_per_step"], tile_weighting_row
    )
    residual_energy = _residual_energy_digest(diagnostic_root)

    joint_scalar = _scalar_diagnostics(
        configs["joint_posterior_per_step"]
    )
    local_scalar = _scalar_diagnostics(configs["local_hr_baseline"])
    claims = {
        "method_success_declared": False,
        "proof_of_concept_scope": "one object, 48 successful tiles, seeds 42-45",
        "input_view_improved_vs_local": False,
        "geometry_consistency_improved_vs_local": bool(
            joint_scalar["connected_components"]
            < local_scalar["connected_components"]
            and joint_scalar["backside_component_count"]
            < local_scalar["backside_component_count"]
            and joint_scalar["overlap_normal_consistency"]
            > local_scalar["overlap_normal_consistency"]
        ),
        "material_consistency_partially_improved_vs_local": bool(
            joint_scalar["overlap_pbr_latent_rmse"]
            < local_scalar["overlap_pbr_latent_rmse"]
        ),
        "texture_fused_shape_self_consistency_superior_to_texture_only": bool(
            evidence["fused_shape_minus_texture_only_mean_r2"] > 0
        ),
        "reason": (
            "Geometry and most seam diagnostics improve, but input-view "
            "PSNR/SSIM/LPIPS degrade materially versus independent local HR; "
            "the method also does not dominate old anchor on all consistency "
            "metrics."
        ),
    }

    main = configs["joint_posterior_per_step"]
    statistics_dir = suite_root / "statistics"
    for latent in ("shape", "texture"):
        flow = main.get(f"{latent}_flow", {})
        for step in flow.get("step_statistics", []):
            _write_json(
                statistics_dir
                / f"{latent}_step_{int(step['step']):02d}.json",
                step,
            )

    global_dir = suite_root / "global_baseline"
    global_dir.mkdir(parents=True, exist_ok=True)
    source_eval = Path(str(global_metrics["render_png"])).parent
    aligned_link = global_dir / "aligned_eval"
    if not aligned_link.exists():
        aligned_link.symlink_to(source_eval, target_is_directory=True)
    _write_json(
        global_dir / "summary.json",
        {
            "format": "pixal3d_global_baseline_reference_v1",
            "seed": local.get("global_seed"),
            "source": local.get("global_source"),
            "render_metrics": dict(global_metrics),
            "note": (
                "The global mesh is a fixed object input/reference.  Its "
                "aligned render was generated once by the local baseline "
                "evaluation path and is linked here without regeneration."
            ),
        },
    )

    plot_dir = suite_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_configuration_metrics(plot_dir, configs)
    _plot_consistency_metrics(plot_dir, configs)
    _plot_cca_dynamics(plot_dir, main)
    _plot_multi_seed(plot_dir, multi_seed)
    _plot_evidence_ablation(plot_dir, evidence)
    _plot_weighting_ablation(plot_dir, weighting_ablation)

    payload = {
        "format": "pixal3d_online_canonical_posterior_experiment_v2",
        "suite_root": str(suite_root),
        "training_free": True,
        "gradient_updates": 0,
        "cuda_device": 4,
        "configurations_present": list(CONFIG_NAMES),
        "global_baseline": {
            "summary": str(global_dir / "summary.json"),
            "render_metrics": dict(global_metrics),
        },
        "configurations": config_digest,
        "multi_seed_joint_posterior_per_step": multi_seed,
        "texture_evidence_ablation": evidence,
        "cca_weighting_ablation": weighting_ablation,
        "corrected_residual_energy_analysis": residual_energy,
        "claims": claims,
        "plots": sorted(str(path) for path in plot_dir.glob("*.png")),
        "report": str(suite_root / "EXPERIMENT_REPORT.md"),
    }
    _write_json(suite_root / "summary.json", payload)
    report = _report_markdown(
        suite_root,
        configs,
        multi_seed,
        evidence,
        weighting_ablation,
        residual_energy,
        claims,
    )
    (suite_root / "EXPERIMENT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(f"[done] {suite_root / 'summary.json'}")
    print(f"[done] {suite_root / 'EXPERIMENT_REPORT.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root",
        default="outputs/joint_online_canonical_posterior",
    )
    parser.add_argument(
        "--diagnostic-root",
        default="outputs/global_local_latent_relationship",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(Path(arguments.suite_root), Path(arguments.diagnostic_root))
