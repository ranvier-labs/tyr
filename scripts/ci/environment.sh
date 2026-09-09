#!/usr/bin/env bash
# Source from the repository root before builds, key generation and test runs.
lean_prefix="$(lean --print-prefix)"
if [[ "$(uname -s)" == Darwin ]]; then
  omp_prefix="$(brew --prefix libomp)"
  export DYLD_LIBRARY_PATH="$PWD/external/libtorch/lib:${omp_prefix}/lib:${lean_prefix}/lib/lean:${DYLD_LIBRARY_PATH:-}"
else
  multiarch_dir="/usr/lib/$(gcc -print-multiarch)"
  gomp_dir="$(dirname "$(gcc -print-file-name=libgomp.so)")"
  export LEAN_CC="$PWD/scripts/lean_cc_wrapper.sh"
  export LEAN_CC_GCC=/usr/bin/gcc
  export LIBRARY_PATH="${multiarch_dir}:${gomp_dir}:/usr/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${lean_prefix}/lib:${lean_prefix}/lib/glibc:${multiarch_dir}:${gomp_dir}:/usr/lib:${lean_prefix}/lib/lean:${LD_LIBRARY_PATH:-}"
fi
