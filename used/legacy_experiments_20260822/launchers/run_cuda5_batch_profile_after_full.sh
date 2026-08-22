#!/usr/bin/env bash
# Run the CUDA-5 multi-tile batch profile in the foreground of this launcher.
# Start this script when CUDA 5 is available; it deliberately does not poll
# another process or fall back to serial execution.
set -Eeuo pipefail

repo_dir="/home/nvme04/yyyan/Pixal3D"
python_bin="/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python"
profile_dir="$repo_dir/outputs/multiview_fixed_geometry_pbr_gaussian_cuda5_batch_profile"
log_path="$profile_dir/profile.log"
mkdir -p "$profile_dir"

if [[ -f "$profile_dir/batch_peak_profile.json" ]]; then
  exit 0
fi

cd "$repo_dir"
# Use a 16-tile heavy-support mix per view. It includes the global peak
# view-240/tile-32 and gives the B=32 probe a genuinely full group instead of
# silently testing only the 24-context subset.
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
  "$python_bin" pixal3d_cuda5_batch_peak_profile.py \
  --cuda-device 5 \
  --selected-views 0 120 240 \
  --tile-ids 32 33 30 31 29 26 39 25 10 22 38 9 11 40 16 18 \
  --batch-sizes 8 12 16 24 32 \
  --output-dir "$profile_dir" \
  > "$log_path" 2>&1
