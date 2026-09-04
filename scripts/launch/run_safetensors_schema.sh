#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

echo "SafeTensors schema → generated Lean API"
echo
lake env lean launch/demos/safetensors_schema.lean
echo
echo "PASS: checkpoint structure became typed Lean declarations during elaboration."
