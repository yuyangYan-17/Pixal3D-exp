#!/usr/bin/env bash
# Unattended, gate-controlled CUDA-5 execution for the fixed-geometry study.
set -Eeuo pipefail

repo_dir="/home/nvme04/yyyan/Pixal3D"
python_bin="/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python"
queue_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda5_queue"
status_path="$queue_dir/status.json"
mkdir -p "$queue_dir"

if [[ ! -x "$python_bin" ]]; then
  echo "missing executable: $python_bin" >&2
  exit 1
fi

current_stage="queued"
write_status() {
  local state="$1"
  local stage="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{\n  "state": "%s",\n  "stage": "%s",\n  "updated_utc": "%s"\n}\n' "$state" "$stage" "$timestamp" > "$status_path"
}
trap 'write_status failed "$current_stage"' ERR

run_stage() {
  local stage="$1"
  local output_dir="$2"
  shift 2
  current_stage="$stage"
  if [[ -e "$output_dir/summary.json" ]]; then
    echo "refusing to overwrite completed or partial output: $output_dir" >&2
    return 1
  fi
  write_status running "$stage"
  (
    cd "$repo_dir"
    env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
      "$python_bin" pixal3d_multiview_fixed_geometry_pbr_gaussian_sr.py \
      --cuda-device 5 --flow-batch-size 8 --output-dir "$output_dir" "$@"
  ) > "$queue_dir/${stage}.log" 2>&1
}

verify_stage() {
  local stage="$1"
  local output_dir="$2"
  local expected_contexts="$3"
  local expected_steps="$4"
  local expected_views="$5"
  local expected_tile_ids="$6"
  local require_render="$7"
  "$python_bin" - "$output_dir" "$stage" "$expected_contexts" "$expected_steps" "$expected_views" "$expected_tile_ids" "$require_render" <<'PY' > "$output_dir/gate.json"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
expected_contexts = sys.argv[3]
expected_steps = int(sys.argv[4])
expected_views = [int(value) for value in sys.argv[5].split(",")]
expected_tile_ids = None if sys.argv[6] == "all" else [int(value) for value in sys.argv[6].split(",")]
require_render = bool(int(sys.argv[7]))

def load(name):
    path = root / name
    assert path.is_file(), f"missing required artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))

summary = load("summary.json")
args = summary["args"]
assert args["cuda_device"] == 5
assert args["selected_views"] == expected_views
assert args["num_steps"] == expected_steps
assert args["tile_ids"] == expected_tile_ids
assert summary["correctness"]["all_passed"] is True
assert summary["fixed_geometry"] is True
assert summary["full_view_4096_upsample"] is False
assert summary["condition_fusion"] is False
assert summary["slat_fusion"] is False
assert summary["soft_visibility"] is False
assert summary["round_trip_residual_cancellation"] is False
assert summary["final_gaussian_blend"] is False
contexts = int(summary["active_contexts"])
if expected_contexts == "all":
    assert 0 < contexts <= 147
else:
    assert contexts == int(expected_contexts)

dino = load("dino_cache_diagnostics.json")
assert dino["steps_recomputed"] == 0
assert dino["local_cached_count"] == contexts
assert len(load("visibility_stats.json")["views"]) == len(expected_views)

flow = load("flow_summary.json")
assert len(flow["steps"]) == expected_steps
assert flow["round_trip_residual_cancellation"] is False
assert flow["soft_visibility"] is False
assert flow["slat_fusion"] is False
for step in flow["steps"]:
    assert step["contexts"] == contexts
    assert step["fixed_texture_support"] is True
    assert step["finite_features"] is True
    assert all(step["barriers"].values())

coverage = load("cross_view_coverage_stats.json")["per_step_tile_coverage"]
assert len(coverage) == contexts * expected_steps
cross_view_receipts = sum(row["cross_view_receipts"] for row in coverage)
assert cross_view_receipts > 0, "cross-view consensus received no valid donor"
final = load("final_assignment.json")
assert final["geometry_source"] == "unchanged baseline 1024 mesh"
assert final["final_gaussian_blend"] is False
for name in ("final_per_vertex_pbr_mesh.pt", "final_per_face_pbr_mesh.pt", "MULTIVIEW_GAUSSIAN_PBR_SR_REPORT.md"):
    assert (root / name).is_file(), f"missing final artifact: {name}"
if require_render:
    assert summary["render"]["enabled"] is True
    assert (root / "renders" / "six_view_sheet.png").is_file()
    assert (root / "turntable" / "turntable.gif").is_file()

print(json.dumps({"stage": stage, "passed": True, "active_contexts": contexts, "steps": expected_steps,
                  "cross_view_receipts": cross_view_receipts, "render_checked": require_render}, indent=2))
PY
}

full_output="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda5_rework_b8"

write_status queued "$current_stage"
run_stage full "$full_output" --selected-views 0 120 240 --num-steps 12 --no-render
verify_stage full "$full_output" all 12 "0,120,240" all 0

# Rendering consumes the completed final mesh only; it does not rerun flow.
current_stage="render"
write_status running "$current_stage"
(
  cd "$repo_dir"
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    "$python_bin" pixal3d_multiview_fixed_geometry_pbr_gaussian_sr.py \
    --cuda-device 5 --render-only --output-dir "$full_output"
) > "$queue_dir/render.log" 2>&1

current_stage="complete"
write_status complete "$current_stage"
