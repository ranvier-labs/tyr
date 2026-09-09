#!/usr/bin/env bash
# Invoke under the host-wide GPU flock (see cuda-smoke.yml).
set -euo pipefail
qualification_root=${TYR_QUALIFICATION_ROOT:-"$HOME/tyr-qualification"}
python_bin="$qualification_root/venv/bin/python"
reports=(gpu)
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then reports+=(models); fi
mkdir -p output/qualification
# A persistent runner must never publish a previous run's successful report.
rm -f output/qualification/{gpu,models,summary}.json
finish() {
  local status=$?
  if ! python3 scripts/qualification/summarize.py --directory output/qualification "${reports[@]}"; then status=1; fi
  exit "$status"
}
trap finish EXIT

# The lock coordinates Tyr qualification jobs across both repository runners.
# Refuse to overlap an unrelated CUDA workload that does not use that lock.
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active" ]]; then
  echo "GPU is occupied by an unrelated workload; retry after it finishes." >&2
  exit 1
fi
export PATH="$(dirname "$python_bin"):$HOME/.elan/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
# Preparation shares the GPU lock too: another repository job cannot replace
# the venv or partially download a fixture while this run uses those inputs.
bash scripts/qualification/setup_python.sh
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then
  python3 scripts/qualification/prepare.py --cache "$qualification_root/fixtures" \
    --download --paths-output output/qualification/fixture-paths.json
fi
export TYR_QUALIFICATION_PYTHON="$python_bin"
export GPU=${GPU:-GB10}
export TYR_GPU_TARGET="$GPU"
case "$GPU" in
  GB10|B200|B300)
    export TYR_GPU_FAMILY=BLACKWELL
    modules=(Tyr.GPU.Kernels.MhaGB10 Tyr.GPU.Kernels.FusedRMSNorm Tyr.GPU.Kernels.FusedLayerNorm Tyr.GPU.Kernels.MhaH100Decode Tyr.GPU.Kernels.RKCombine Tyr.GPU.Kernels.BrownianSample)
    gpu_runner=TestGPUGB10E2E ;;
  H100)
    export TYR_GPU_FAMILY=HOPPER
    modules=(Tyr.GPU.Kernels.Copy Tyr.GPU.Kernels.Rotary Tyr.GPU.Kernels.FusedLayerNorm Tyr.GPU.Kernels.FusedRMSNorm Tyr.GPU.Kernels.MhaH100 Tyr.GPU.Kernels.MhaH100Decode)
    gpu_runner=TestGPUE2E ;;
  *) echo "No strict suite configured for GPU=$GPU" >&2; exit 1 ;;
esac
export TYR_GPU_CODEGEN_MODULE="${modules[*]}"
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
source scripts/ci/environment.sh

python3 scripts/qualification/run.py --kind gpu --python "$python_bin" \
  --check-runtime --report output/qualification/runtime.json
# Keep the known codegen bootstrap explicit on a fresh checkout.
TYR_SKIP_GPU_CODEGEN=1 lake -R build "${modules[@]}"
targets=("$gpu_runner" RunMhaH100Decode LagunaModelTest)
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then
  targets+=(Qwen3TTSEndToEnd Qwen3ASRTranscribe)
fi
lake -R build "${targets[@]}"
python3 scripts/check_ffi_abi.py
python3 scripts/check_ffi_abi.py --header cc/src/tyr_owned_kv_abi.h --generated .lake/build/ir/Tyr/Inference/OwnedKV.c
python3 scripts/qualification/run.py --kind gpu --python "$python_bin" \
  --report output/qualification/gpu.json
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then
  python3 scripts/qualification/run.py --kind models --python "$python_bin" \
    --cache "$qualification_root/fixtures" --report output/qualification/models.json
fi
