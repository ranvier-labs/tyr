#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

data_root="${TYR_QM9_DATA_ROOT:-$repo_root/data/qm9-launch}"
archive="$data_root/gdb9.tar.gz"
xyz_dir="$data_root/xyz"
jsonl="$data_root/qm9-branching.jsonl"
max_molecules="${TYR_QM9_MAX_MOLECULES:-5000}"
url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

mkdir -p "$data_root" "$xyz_dir"

if [[ ! -f "$archive" ]]; then
  echo "Downloading QM9 coordinate archive..."
  curl --fail --location --retry 5 --output "$archive" "$url"
fi

if [[ ! -f "$xyz_dir/gdb9.sdf" ]]; then
  echo "Extracting QM9 coordinates..."
  tar -xzf "$archive" -C "$xyz_dir"
fi

echo "Preparing $max_molecules molecule records for Tyr..."
python3 scripts/qm9_sdf_to_branching_jsonl.py \
  "$xyz_dir/gdb9.sdf" \
  --out "$jsonl" \
  --max-molecules "$max_molecules"

echo "$jsonl"
