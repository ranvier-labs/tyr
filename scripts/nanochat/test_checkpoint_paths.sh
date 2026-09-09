#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/checkpoint_paths.sh"
checkpoint_test_dir="$(mktemp -d "${TMPDIR:-/tmp}/nanochat-checkpoints.XXXXXX")"
trap 'rm -rf "${checkpoint_test_dir}"' EXIT

mkdir -p "${checkpoint_test_dir}/flat" "${checkpoint_test_dir}/versioned/.snapshots/snapshot-1-2"
touch "${checkpoint_test_dir}/flat/step.pt"
touch "${checkpoint_test_dir}/versioned/.snapshots/snapshot-1-2/step.pt"
printf 'snapshot-1-2' > "${checkpoint_test_dir}/versioned/CURRENT"
[[ "$(nanochat_checkpoint_dir "${checkpoint_test_dir}/flat")" == "${checkpoint_test_dir}/flat" ]]
[[ "$(nanochat_checkpoint_dir "${checkpoint_test_dir}/versioned")" == \
   "${checkpoint_test_dir}/versioned/.snapshots/snapshot-1-2" ]]

# A broken published pointer must not fall back to stale flat files.
touch "${checkpoint_test_dir}/versioned/step.pt"
for invalid in 'snapshot-missing' '../flat' 'snapshot-../flat' 'snapshot-bad\name' ''; do
  printf '%s' "${invalid}" > "${checkpoint_test_dir}/versioned/CURRENT"
  if nanochat_checkpoint_dir "${checkpoint_test_dir}/versioned" >/dev/null; then
    printf 'accepted invalid snapshot pointer: %s\n' "${invalid}" >&2
    exit 1
  fi
done
rm "${checkpoint_test_dir}/versioned/CURRENT"
mkdir "${checkpoint_test_dir}/versioned/CURRENT"
if nanochat_checkpoint_dir "${checkpoint_test_dir}/versioned" >/dev/null; then
  printf 'accepted directory snapshot pointer\n' >&2
  exit 1
fi
rmdir "${checkpoint_test_dir}/versioned/CURRENT"
ln -s missing "${checkpoint_test_dir}/versioned/CURRENT"
if nanochat_checkpoint_dir "${checkpoint_test_dir}/versioned" >/dev/null; then
  printf 'accepted dangling snapshot pointer\n' >&2
  exit 1
fi
if nanochat_checkpoint_dir "${checkpoint_test_dir}/missing" >/dev/null; then
  printf 'accepted missing checkpoint\n' >&2
  exit 1
fi
printf 'NanoChat checkpoint path checks passed\n'
