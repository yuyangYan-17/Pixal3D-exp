#!/usr/bin/env bash
# Queue CUDA-5 peak-tile and heavy-support batch profiles after the full run.
# The queued work never competes with the full process and never serializes an
# OOMed batch internally; the profile harness records failed candidates.
set -Eeuo pipefail

repo_dir="/home/nvme04/yyyan/Pixal3D"
python_bin="/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python"
full_pid="${1:?usage: $0 FULL_RUN_PID}"
full_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda5_rework_b8"
peak_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda5_peak_tile_240_32"
peak_log="$peak_dir/profile.log"

cd "$repo_dir"
while kill -0 "$full_pid" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$full_dir/summary.json" ]]; then
  echo "full run exited without summary: $full_dir/summary.json" >&2
  exit 2
fi

mkdir -p "$peak_dir"
if [[ ! -f "$peak_dir/batch_peak_profile.json" ]]; then
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    "$python_bin" pixal3d_cuda5_batch_peak_profile.py \
    --cuda-device 5 \
    --selected-views 240 \
    --tile-ids 32 \
    --batch-sizes 1 \
    --output-dir "$peak_dir" \
    > "$peak_log" 2>&1
fi

./run_cuda5_batch_profile_after_full.sh
