#!/usr/bin/env python3
"""Run the validated n=0/n=4 visibility gate over the remaining assets.

The expensive baseline/geometry preparation is intentionally separate.  This
driver is resumable: it keeps the two point-routed flow endpoints, direct
renders/metrics, the pixel-query appearance gate, and one timing/summary JSON
per image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "sr_point_visibility_texture.py"
GATE = REPO / "train_method/run_pixel_query_visibility_merge.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--assets",
        type=Path,
        default=REPO / "assets/images",
    )
    p.add_argument(
        "--prepared-root",
        type=Path,
        default=REPO / "outputs/sr_texture_visibility_final_20260919",
    )
    p.add_argument(
        "--experiment-root",
        type=Path,
        default=REPO / "outputs/sr_point_visibility_batch_20260920",
    )
    p.add_argument("--gpu", default="4")
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=58)
    p.add_argument("--face-chunk", type=int, default=2_000_000)
    p.add_argument("--include", default="", help="comma-separated filenames; empty means all")
    p.add_argument("--exclude", default="0_img.png")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def image_list(args):
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    values = sorted(
        p for p in args.assets.resolve().iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )
    include = {v.strip() for v in args.include.split(",") if v.strip()}
    exclude = {v.strip() for v in args.exclude.split(",") if v.strip()}
    if include:
        values = [p for p in values if p.name in include]
    values = [p for p in values if p.name not in exclude]
    if args.limit:
        values = values[: args.limit]
    if not values:
        raise RuntimeError("no images selected")
    return values


def run(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        ret = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - started
    if ret.returncode:
        raise RuntimeError(f"return code {ret.returncode}; see {log_path}")
    return elapsed


def child_env(gpu):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PIXAL3D_LOW_MEMORY_DECODER"] = "1"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def load_json(path):
    return json.loads(Path(path).read_text())


def run_one(args, image):
    stem = image.stem
    prepared = args.prepared_root.resolve() / stem / "texture_explore"
    if not (prepared / "prepared.json").exists():
        raise FileNotFoundError(f"prepared cache is missing: {prepared / 'prepared.json'}")
    experiment = args.experiment_root.resolve() / stem
    logs = experiment / "logs"
    experiment.mkdir(parents=True, exist_ok=True)
    env = child_env(args.gpu)
    timing = {
        "image": image.name,
        "prepared_root": str(prepared),
        "experiment_dir": str(experiment),
        "seed": args.seed,
        "gpu": str(args.gpu),
    }
    started = time.perf_counter()

    core_command = [
        sys.executable,
        str(CORE),
        "--gpu", str(args.gpu),
        "--root", str(prepared),
        "--output-dir", str(experiment),
        "--seeds", str(args.seed),
        "--n-values", "4",
        "--render-resolution", "1024",
        "--batch-size", str(args.batch_size),
    ]
    # Let the core validate its versioned sampler manifest before resuming.
    timing["core_wall_seconds"] = run(core_command, logs / "core.log", env)

    control_command = list(core_command)
    control_command[control_command.index("--n-values") + 1] = "0"
    control_command.append("--all-conditional")
    timing["control_wall_seconds"] = run(control_command, logs / "control.log", env)
    base_variant = f"seed_{args.seed}_n_0_all_conditional"
    candidate_variant = f"seed_{args.seed}_n_4"
    gate_dir = experiment / f"pixel_gate_merge_{base_variant}_with_{candidate_variant}"
    gate_result = gate_dir / "result.json"
    gate_command = [
        sys.executable,
        str(GATE),
        "--gpu", str(args.gpu),
        "--root", str(prepared),
        "--experiment-dir", str(experiment),
        "--base-variant", base_variant,
        "--candidate-variant", candidate_variant,
        "--face-chunk", str(args.face_chunk),
        "--appearance-gate",
    ]
    if not gate_result.exists():
        timing["gate_wall_seconds"] = run(gate_command, logs / "appearance_gate.log", env)
    else:
        timing["gate_wall_seconds"] = 0.0
        timing["gate_resumed"] = True

    result = load_json(gate_result)
    base_metrics_path = prepared / "baseline" / "evaluation_1024" / "metrics.json"
    if not base_metrics_path.exists():
        base_metrics_path = args.prepared_root.resolve() / stem / "evaluation_1024" / "metrics.json"
    baseline = load_json(base_metrics_path)["results"]["baseline1024"]
    front = result["metrics"]["front"]
    timing.update(
        {
            "baseline_metrics": baseline,
            "front_metrics": front,
            "front_delta_vs_baseline": {
                "psnr_db": front["psnr_db"] - baseline["psnr_db"],
                "ssim": front["ssim"] - baseline["ssim"],
                "lpips": front["lpips"] - baseline["lpips"],
            },
            "back_vs_baseline_render": result["metrics"]["back_vs_baseline"],
            "base_flow_seconds": load_json(experiment / base_variant / "texture/timing.json")["texture_flow_seconds"],
            "candidate_flow_seconds": load_json(experiment / candidate_variant / "texture/timing.json")["texture_flow_seconds"],
            "base_direct_render_seconds": load_json(experiment / base_variant / "result.json")["metrics"]["render_seconds"],
            "candidate_direct_render_seconds": load_json(experiment / candidate_variant / "result.json")["metrics"]["render_seconds"],
            "gate_base_decode_seconds": result["base_decode_seconds"],
            "gate_candidate_decode_seconds": result["candidate_decode_seconds"],
            "gate_render_seconds": result["metrics"]["render_seconds"],
            "gate_candidate_rows": int(result.get("use_candidate_rows", 0)),
            "touched_field_rows": result["touched_field_rows"],
            "total_field_rows": result["total_field_rows"],
            "total_wall_seconds": time.perf_counter() - started,
            "status": "COMPLETE",
        }
    )
    # Keep the selected-row count even though older gate manifests only stored
    # it in query_support.pt.
    support = gate_dir / "query_support.pt"
    if support.exists():
        import torch

        payload = torch.load(support, map_location="cpu", weights_only=False)
        timing["candidate_rows_from_support"] = int(payload["use_candidate"].sum())
        del payload
    (experiment / "visibility_gate_timing.json").write_text(
        json.dumps(timing, indent=2) + "\n"
    )
    return timing


def main():
    args = parse_args()
    images = image_list(args)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    rows = []
    batch_started = time.perf_counter()
    for index, image in enumerate(images, 1):
        print(f"VISIBILITY {index}/{len(images)} {image.name}", flush=True)
        try:
            row = run_one(args, image)
        except Exception as exc:
            row = {"image": image.name, "status": "FAILED", "error": repr(exc)}
            (args.experiment_root / image.stem).mkdir(parents=True, exist_ok=True)
            (args.experiment_root / image.stem / "visibility_gate_timing.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            rows.append(row)
            print(f"FAILED {image.name}: {exc}", flush=True)
            continue
        rows.append(row)
        print(
            f"DONE {image.name} front_psnr={row['front_metrics']['psnr_db']:.4f} "
            f"delta={row['front_delta_vs_baseline']['psnr_db']:.4f} "
            f"wall={row['total_wall_seconds']:.1f}s",
            flush=True,
        )
    payload = {
        "status": "COMPLETE" if all(r.get("status") == "COMPLETE" for r in rows) else "PARTIAL",
        "seed": args.seed,
        "gpu": str(args.gpu),
        "images": len(rows),
        "batch_wall_seconds": time.perf_counter() - batch_started,
        "rows": rows,
    }
    (args.experiment_root / "visibility_gate_batch.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
