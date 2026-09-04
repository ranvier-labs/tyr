#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

data_path="${1:--}"
steps="${TYR_MOLECULE_STEPS:-1200}"
batch_size="${TYR_MOLECULE_BATCH_SIZE:-32}"
seed="${TYR_MOLECULE_SEED:-20260709}"
output_dir="${TYR_LAUNCH_OUTPUT_DIR:-launch/generated/learned-molecule}"
mkdir -p "$output_dir"
prefix="$output_dir/sample"

echo "Learned molecule branching showcase"
echo "steps=$steps batch_size=$batch_size seed=$seed data=$data_path"

lake exe BranchingFlowsMoleculeTrainGenerate \
  --data "$data_path" \
  --out-prefix "$prefix" \
  --steps "$steps" \
  --batch-size "$batch_size" \
  --seed "$seed" | tee "$output_dir/training-and-generation.txt"

for xyz in "${prefix}"_target.xyz "${prefix}"_source.xyz "${prefix}"_generated.xyz "${prefix}"_step_*.xyz; do
  [[ -f "$xyz" ]] || continue
  ./scripts/launch/render_xyz.py "$xyz" "${xyz%.xyz}.svg"
done

./scripts/launch/render_branching_storyboard.py "$prefix" "$output_dir/branching-storyboard.svg"

echo
echo "PASS: learned training metrics, generation events, trajectory frames, and storyboard are in $output_dir."
