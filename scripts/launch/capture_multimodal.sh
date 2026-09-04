#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 image-path" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

image_path="$1"
model="${TYR_LAUNCH_MODEL:-google/gemma-4-E2B-it}"
device="${TYR_LAUNCH_DEVICE:-auto}"
output_dir="${TYR_LAUNCH_OUTPUT_DIR:-launch/generated/multimodal}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$output_dir/$timestamp"
mkdir -p "$run_dir"

{
  echo "timestamp_utc=$timestamp"
  echo "commit=$(git rev-parse HEAD)"
  echo "model=$model"
  echo "device_requested=$device"
  echo "image=$image_path"
} | tee "$run_dir/provenance.txt"

set +e
lake exe Gemma4RunHF \
  --source "$model" \
  --device "$device" \
  --image "$image_path" \
  --prompt "Describe this image precisely and concisely." \
  --max-new-tokens 96 \
  --stream 2>&1 | tee "$run_dir/transcript.txt"
status="${PIPESTATUS[0]}"
set -e
echo "$status" > "$run_dir/exit-code.txt"

if [[ "$status" -ne 0 ]]; then
  echo "Multimodal inference failed; logs preserved in $run_dir" >&2
  exit "$status"
fi

echo "PASS: multimodal record saved to $run_dir"
