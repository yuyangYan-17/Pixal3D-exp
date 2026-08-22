#!/usr/bin/env bash
# Unattended CUDA-4 execution for the corrected fixed-geometry multi-view run.
# The full flow and final render run serially; no process polling is used.
set -Eeuo pipefail

repo_dir="/home/nvme04/yyyan/Pixal3D"
python_bin="/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python"
queue_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda4_fix_queue"
status_path="$queue_dir/status.json"
full_output="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda4_fix_full_b8"
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

if [[ -e "$full_output/summary.json" ]]; then
  echo "refusing to overwrite completed output: $full_output" >&2
  exit 1
fi

write_status running full
(
  cd "$repo_dir"
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    "$python_bin" pixal3d_multiview_fixed_geometry_pbr_gaussian_sr.py \
    --cuda-device 4 \
    --flow-batch-size 8 \
    --selected-views 0 120 240 \
    --num-steps 12 \
    --no-render \
    --output-dir "$full_output"
) > "$queue_dir/full.log" 2>&1

write_status running render
(
  cd "$repo_dir"
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    "$python_bin" pixal3d_multiview_fixed_geometry_pbr_gaussian_sr.py \
    --cuda-device 4 \
    --render-only \
    --output-dir "$full_output"
) > "$queue_dir/render.log" 2>&1

write_status complete complete
