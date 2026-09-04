#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

binary=".lake/build/bin/BranchingFlowsMoleculeGenerate"
if [[ ! -x "$binary" ]]; then
  echo "Missing $binary. Build the molecule generation example first." >&2
  exit 1
fi

output_dir="launch/generated/molecule"
mkdir -p "$output_dir"
prefix="$output_dir/water"

echo "Molecule-shaped branching flow"
"$binary" "$prefix" | tee "$output_dir/trajectory.txt"

for stage in target bridge generated; do
  ./scripts/launch/render_xyz.py \
    "${prefix}_${stage}.xyz" \
    "${prefix}_${stage}.svg"
done

# Newer builds export every generator trajectory state. Older already-built
# binaries still provide bridge and final states; preserve a useful two-frame
# branching view until the executable is rebuilt.
if ! compgen -G "${prefix}_step_*.xyz" >/dev/null; then
  cp "${prefix}_bridge.xyz" "${prefix}_step_0.xyz"
  cp "${prefix}_generated.xyz" "${prefix}_step_1.xyz"
fi

for xyz in "${prefix}"_step_*.xyz; do
  ./scripts/launch/render_xyz.py "$xyz" "${xyz%.xyz}.svg"
done

./scripts/launch/render_branching_storyboard.py \
  "$prefix" \
  "$output_dir/branching-storyboard.svg"
./scripts/launch/render_branching_trajectory.py \
  "${prefix}_trajectory.jsonl" \
  "$output_dir/branching-trajectory.svg"
./scripts/launch/render_molecule_graphical_abstract.py

if command -v rsvg-convert >/dev/null 2>&1; then
  rsvg-convert -w 1600 "$output_dir/branching-storyboard.svg" \
    -o "$output_dir/branching-storyboard.png"
  rsvg-convert -w 1600 "$output_dir/branching-trajectory.svg" \
    -o "$output_dir/branching-trajectory.png"
  rsvg-convert -w 1600 "$output_dir/graphical-abstract.svg" \
    -o "$output_dir/graphical-abstract.png"
fi

echo
echo "PASS: generated XYZ states, lineage trajectory, and branching figures in $output_dir."
