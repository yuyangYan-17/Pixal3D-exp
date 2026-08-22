#!/usr/bin/env bash
# Wait for the CUDA-4 full run, then measure the peak tile and heavy batches.
set -Eeuo pipefail

repo_dir="/home/nvme04/yyyan/Pixal3D"
python_bin="/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python"
full_pid="${1:?usage: $0 FULL_RUN_PID}"
full_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda4_rework_b8"
peak_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda4_peak_tile_240_32"
batch_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda4_batch_profile"

cd "$repo_dir"
while kill -0 "$full_pid" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$full_dir/summary.json" ]]; then
  echo "CUDA-4 full run exited without summary: $full_dir/summary.json" >&2
  exit 2
fi

mkdir -p "$peak_dir" "$batch_dir"
if [[ ! -f "$peak_dir/batch_peak_profile.json" ]]; then
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    "$python_bin" pixal3d_cuda5_batch_peak_profile.py \
    --cuda-device 4 \
    --selected-views 240 \
    --tile-ids 32 \
    --batch-sizes 1 \
    --output-dir "$peak_dir" \
    > "$peak_dir/profile.log" 2>&1
fi

if [[ ! -f "$batch_dir/batch_peak_profile.json" ]]; then
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    "$python_bin" pixal3d_cuda5_batch_peak_profile.py \
    --cuda-device 4 \
    --selected-views 0 120 240 \
    --tile-ids 32 33 30 31 29 26 39 25 10 22 38 9 11 40 16 18 \
    --batch-sizes 8 12 16 24 32 \
    --output-dir "$batch_dir" \
    > "$batch_dir/profile.log" 2>&1
fi
