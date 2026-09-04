#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() {
  echo "[qm9-paper] $*" >&2
}

die() {
  echo "[qm9-paper] error: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  TYR_QM9_XYZ_DIR=/path/to/qm9_xyz scripts/branchingflows/run_qm9_paper.sh
  TYR_QM9_JSONL=/path/to/qm9_branching.jsonl scripts/branchingflows/run_qm9_paper.sh

Required input:
  TYR_QM9_JSONL      Preprocessed Tyr BranchingFlows JSONL dataset.
  TYR_QM9_XYZ_DIR    Directory of QM9 .xyz coordinate files; converted to JSONL.

Useful overrides:
  TYR_QM9_PROFILE=paper-qm9-main|paper-qm9-appendix|smoke
  TYR_QM9_RUN_ROOT=/path/to/run
  TYR_QM9_RESUME_CHECKPOINT=/path/to/checkpoint
  TYR_QM9_STEPS=1000
  TYR_QM9_BATCH_SIZE=128
  TYR_QM9_ARCHITECTURE=full|compact
  TYR_QM9_HIDDEN_DIM=384
  TYR_QM9_HEADS=12
  TYR_QM9_HEAD_DIM=64
  TYR_QM9_RFF_DIM=64
  TYR_QM9_LAYERS=12
  TYR_QM9_COORD_UPDATE_LAYERS=6
  TYR_QM9_COORD_WEIGHT=10
  TYR_QM9_LABEL_WEIGHT=0.3333333333333333
  TYR_QM9_SPLITS_WEIGHT=1
  TYR_QM9_DEL_WEIGHT=1
  TYR_QM9_BRANCHING_TIME_PROB=0.5
  TYR_QM9_SAMPLE_COUNT=10000
  TYR_QM9_SAMPLE_STEPS=1000
  TYR_QM9_MAX_MOLECULES=1000       Cap preprocessing for small dry runs.
  TYR_QM9_NO_GENERATE=1            Train/eval only.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_cmd python3
require_cmd git
require_cmd lake

profile="${TYR_QM9_PROFILE:-paper-qm9-main}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${TYR_QM9_RUN_ROOT:-${REPO_ROOT}/output/branchingflows/qm9-paper-${timestamp}}"
if [[ "${run_root}" != /* ]]; then
  run_root="${REPO_ROOT}/${run_root}"
fi
jsonl="${TYR_QM9_JSONL:-}"
xyz_dir="${TYR_QM9_XYZ_DIR:-}"
checkpoint_dir="${TYR_QM9_CHECKPOINT_DIR:-${run_root}/checkpoints/latest}"
out_prefix="${TYR_QM9_OUT_PREFIX:-${run_root}/samples/qm9}"

mkdir -p "${run_root}/data" "${run_root}/logs" "${run_root}/samples" "${run_root}/checkpoints"

git_rev="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git_branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"

if [[ -z "${jsonl}" ]]; then
  [[ -n "${xyz_dir}" ]] || die "set TYR_QM9_JSONL or TYR_QM9_XYZ_DIR"
  [[ -d "${xyz_dir}" ]] || die "TYR_QM9_XYZ_DIR is not a directory: ${xyz_dir}"
  jsonl="${run_root}/data/qm9_branching.jsonl"
  convert_cmd=(python3 "${REPO_ROOT}/scripts/qm9_xyz_to_branching_jsonl.py" "${xyz_dir}" --out "${jsonl}")
  if [[ -n "${TYR_QM9_MAX_MOLECULES:-}" ]]; then
    convert_cmd+=(--max-molecules "${TYR_QM9_MAX_MOLECULES}")
  fi
  if [[ "${TYR_QM9_KEEP_ORIGINAL_ORDER:-0}" == "1" ]]; then
    convert_cmd+=(--keep-original-order)
  fi
  log "preprocessing QM9 xyz files into ${jsonl}"
  (cd "${REPO_ROOT}" && "${convert_cmd[@]}") 2>&1 | tee "${run_root}/logs/preprocess.log"
else
  [[ -f "${jsonl}" ]] || die "TYR_QM9_JSONL does not exist: ${jsonl}"
fi

run_cmd=(
  lake exe BranchingFlowsMoleculeTrainGenerate
  --profile "${profile}"
  --data "${jsonl}"
  --out-prefix "${out_prefix}"
  --checkpoint-dir "${checkpoint_dir}"
)

if [[ -n "${TYR_QM9_RESUME_CHECKPOINT:-}" ]]; then
  run_cmd+=(--resume-checkpoint "${TYR_QM9_RESUME_CHECKPOINT}")
fi
if [[ -n "${TYR_QM9_STEPS:-}" ]]; then
  run_cmd+=(--steps "${TYR_QM9_STEPS}")
fi
if [[ -n "${TYR_QM9_TOTAL_STEPS:-}" ]]; then
  run_cmd+=(--total-steps "${TYR_QM9_TOTAL_STEPS}")
fi
if [[ -n "${TYR_QM9_WARMUP_STEPS:-}" ]]; then
  run_cmd+=(--warmup-steps "${TYR_QM9_WARMUP_STEPS}")
fi
if [[ -n "${TYR_QM9_COOLDOWN_STEPS:-}" ]]; then
  run_cmd+=(--cooldown-steps "${TYR_QM9_COOLDOWN_STEPS}")
fi
if [[ -n "${TYR_QM9_BATCH_SIZE:-}" ]]; then
  run_cmd+=(--batch-size "${TYR_QM9_BATCH_SIZE}")
fi
if [[ -n "${TYR_QM9_MAX_LEN:-}" ]]; then
  run_cmd+=(--max-len "${TYR_QM9_MAX_LEN}")
fi
if [[ -n "${TYR_QM9_ARCHITECTURE:-}" ]]; then
  run_cmd+=(--architecture "${TYR_QM9_ARCHITECTURE}")
fi
if [[ -n "${TYR_QM9_HIDDEN_DIM:-}" ]]; then
  run_cmd+=(--hidden-dim "${TYR_QM9_HIDDEN_DIM}")
fi
if [[ -n "${TYR_QM9_HEADS:-}" ]]; then
  run_cmd+=(--heads "${TYR_QM9_HEADS}")
fi
if [[ -n "${TYR_QM9_HEAD_DIM:-}" ]]; then
  run_cmd+=(--head-dim "${TYR_QM9_HEAD_DIM}")
fi
if [[ -n "${TYR_QM9_MLP:-}" ]]; then
  run_cmd+=(--mlp "${TYR_QM9_MLP}")
fi
if [[ -n "${TYR_QM9_RFF_DIM:-}" ]]; then
  run_cmd+=(--rff-dim "${TYR_QM9_RFF_DIM}")
fi
if [[ -n "${TYR_QM9_LAYERS:-}" ]]; then
  run_cmd+=(--layers "${TYR_QM9_LAYERS}")
fi
if [[ -n "${TYR_QM9_COORD_UPDATE_LAYERS:-}" ]]; then
  run_cmd+=(--coord-update-layers "${TYR_QM9_COORD_UPDATE_LAYERS}")
fi
if [[ -n "${TYR_QM9_LR:-}" ]]; then
  run_cmd+=(--lr "${TYR_QM9_LR}")
fi
if [[ -n "${TYR_QM9_LR_END:-}" ]]; then
  run_cmd+=(--lr-end "${TYR_QM9_LR_END}")
fi
if [[ -n "${TYR_QM9_WEIGHT_DECAY:-}" ]]; then
  run_cmd+=(--weight-decay "${TYR_QM9_WEIGHT_DECAY}")
fi
if [[ -n "${TYR_QM9_COORD_WEIGHT:-}" ]]; then
  run_cmd+=(--coord-weight "${TYR_QM9_COORD_WEIGHT}")
fi
if [[ -n "${TYR_QM9_LABEL_WEIGHT:-}" ]]; then
  run_cmd+=(--label-weight "${TYR_QM9_LABEL_WEIGHT}")
fi
if [[ -n "${TYR_QM9_SPLITS_WEIGHT:-}" ]]; then
  run_cmd+=(--splits-weight "${TYR_QM9_SPLITS_WEIGHT}")
fi
if [[ -n "${TYR_QM9_DEL_WEIGHT:-}" ]]; then
  run_cmd+=(--del-weight "${TYR_QM9_DEL_WEIGHT}")
fi
if [[ -n "${TYR_QM9_BRANCHING_TIME_PROB:-}" ]]; then
  run_cmd+=(--branching-time-prob "${TYR_QM9_BRANCHING_TIME_PROB}")
fi
if [[ -n "${TYR_QM9_SPLIT_LOGIT_CAP:-}" ]]; then
  run_cmd+=(--split-logit-cap "${TYR_QM9_SPLIT_LOGIT_CAP}")
fi
if [[ -n "${TYR_QM9_SAMPLE_COUNT:-}" ]]; then
  run_cmd+=(--sample-count "${TYR_QM9_SAMPLE_COUNT}")
fi
if [[ -n "${TYR_QM9_SAMPLE_STEPS:-}" ]]; then
  run_cmd+=(--sample-steps "${TYR_QM9_SAMPLE_STEPS}")
fi
if [[ -n "${TYR_QM9_SEED:-}" ]]; then
  run_cmd+=(--seed "${TYR_QM9_SEED}")
fi
if [[ "${TYR_QM9_NO_GENERATE:-0}" == "1" ]]; then
  run_cmd+=(--no-generate)
fi

manifest="${run_root}/run.env"
{
  echo "TYR_QM9_PROFILE=${profile}"
  echo "TYR_QM9_RUN_ROOT=${run_root}"
  echo "TYR_QM9_JSONL=${jsonl}"
  echo "TYR_QM9_XYZ_DIR=${xyz_dir}"
  echo "TYR_QM9_CHECKPOINT_DIR=${checkpoint_dir}"
  echo "TYR_QM9_OUT_PREFIX=${out_prefix}"
  echo "TYR_QM9_ARCHITECTURE=${TYR_QM9_ARCHITECTURE:-profile-default}"
  echo "TYR_QM9_HIDDEN_DIM=${TYR_QM9_HIDDEN_DIM:-profile-default}"
  echo "TYR_QM9_HEADS=${TYR_QM9_HEADS:-profile-default}"
  echo "TYR_QM9_HEAD_DIM=${TYR_QM9_HEAD_DIM:-profile-default}"
  echo "TYR_QM9_RFF_DIM=${TYR_QM9_RFF_DIM:-profile-default}"
  echo "TYR_QM9_LAYERS=${TYR_QM9_LAYERS:-profile-default}"
  echo "TYR_QM9_COORD_UPDATE_LAYERS=${TYR_QM9_COORD_UPDATE_LAYERS:-profile-default}"
  echo "TYR_QM9_WEIGHT_DECAY=${TYR_QM9_WEIGHT_DECAY:-profile-default}"
  echo "TYR_QM9_COORD_WEIGHT=${TYR_QM9_COORD_WEIGHT:-profile-default}"
  echo "TYR_QM9_LABEL_WEIGHT=${TYR_QM9_LABEL_WEIGHT:-profile-default}"
  echo "TYR_QM9_SPLITS_WEIGHT=${TYR_QM9_SPLITS_WEIGHT:-profile-default}"
  echo "TYR_QM9_DEL_WEIGHT=${TYR_QM9_DEL_WEIGHT:-profile-default}"
  echo "TYR_QM9_BRANCHING_TIME_PROB=${TYR_QM9_BRANCHING_TIME_PROB:-profile-default}"
  echo "TYR_QM9_SPLIT_LOGIT_CAP=${TYR_QM9_SPLIT_LOGIT_CAP:-none}"
  echo "TYR_GIT_BRANCH=${git_branch}"
  echo "TYR_GIT_REV=${git_rev}"
  printf 'TYR_QM9_COMMAND='
  printf '%q ' "${run_cmd[@]}"
  printf '\n'
} > "${manifest}"

log "run root: ${run_root}"
log "git: ${git_branch} ${git_rev}"
log "training command written to ${manifest}"

(cd "${REPO_ROOT}" && "${run_cmd[@]}") 2>&1 | tee "${run_root}/logs/train.log"
