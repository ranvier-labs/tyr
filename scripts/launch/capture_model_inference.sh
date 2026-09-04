#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

source_model="${TYR_LAUNCH_MODEL:-Qwen/Qwen3.5-0.8B}"
device="${TYR_LAUNCH_DEVICE:-auto}"
prompt="${TYR_LAUNCH_PROMPT:-Why do tensor shapes belong in types?}"
max_tokens="${TYR_LAUNCH_MAX_TOKENS:-64}"
output_dir="${TYR_LAUNCH_OUTPUT_DIR:-launch/generated/model-inference}"
mkdir -p "$output_dir"

commit="$(git rev-parse HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$output_dir/$timestamp"
mkdir -p "$run_dir"

{
  echo "timestamp_utc=$timestamp"
  echo "commit=$commit"
  echo "model=$source_model"
  echo "device_requested=$device"
  echo "max_new_tokens=$max_tokens"
  echo "prompt=$prompt"
  uname -a
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  fi
} | tee "$run_dir/provenance.txt"

echo
echo "Running official-checkpoint inference from Lean..."
set +e
lake exe Qwen35RunHF \
  --source "$source_model" \
  --device "$device" \
  --prompt "$prompt" \
  --max-new-tokens "$max_tokens" \
  --stream 2>&1 | tee "$run_dir/transcript.txt"
status="${PIPESTATUS[0]}"
set -e

echo "$status" > "$run_dir/exit-code.txt"
if [[ "$status" -ne 0 ]]; then
  echo "Inference failed; preserved provenance and transcript in $run_dir" >&2
  exit "$status"
fi

echo
echo "PASS: launch-quality inference record saved to $run_dir"
