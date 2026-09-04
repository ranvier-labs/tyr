#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

echo '$ lake env lean launch/demos/shape_safety.lean'
lake env lean launch/demos/shape_safety.lean

echo
echo '$ lake env lean launch/demos/shape_mismatch.lean'
log_file="$(mktemp -t tyr-shape-mismatch.XXXXXX)"
trap 'rm -f "$log_file"' EXIT
if lake env lean launch/demos/shape_mismatch.lean >"$log_file" 2>&1; then
  echo "error: the intentionally invalid program unexpectedly compiled" >&2
  exit 1
fi

cat "$log_file"
if ! grep -qi 'application type mismatch' "$log_file"; then
  echo "error: Lean failed, but not with the expected type mismatch" >&2
  exit 1
fi

echo
echo 'PASS: Tyr accepted the valid projection and rejected the invalid shape before execution.'
