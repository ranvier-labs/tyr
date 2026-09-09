#!/usr/bin/env bash

# Print the directory containing one committed NanoChat checkpoint. Old flat
# directories remain supported when they do not contain a CURRENT pointer.
nanochat_checkpoint_dir() {
  local checkpoint_dir="$1"
  local snapshot_name
  if [[ -e "${checkpoint_dir}/CURRENT" || -L "${checkpoint_dir}/CURRENT" ]]; then
    [[ -f "${checkpoint_dir}/CURRENT" ]] || return 1
    snapshot_name="$(cat "${checkpoint_dir}/CURRENT")" || return 1
    case "${snapshot_name}" in
      snapshot-*) ;;
      *) return 1 ;;
    esac
    case "${snapshot_name}" in
      */*|*\\*|*.*) return 1 ;;
    esac
    checkpoint_dir="${checkpoint_dir}/.snapshots/${snapshot_name}"
  fi
  [[ -f "${checkpoint_dir}/step.pt" ]] || return 1
  printf '%s\n' "${checkpoint_dir}"
}
