/-
  Fused cross-entropy training kernels for the Qwen3-TTS talker vocabulary.
-/

import Tyr.GPU.Kernels.Prelude

/-! # Loss kernels

Stable per-token cross entropy and logits VJP for BF16 logits (vocabulary 3072)
and a deterministic mean reduction of per-token FP32 losses.
-/

namespace Tyr.GPU.Kernels.Loss

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Stable per-token cross entropy and logits VJP for BF16 logits with a fixed
    vocabulary of 3072. One 256-thread CTA owns one token row and each thread
    retains twelve logits. The output gradient is scaled by grad_scale, normally
    1 / rows for a mean-reduced loss. -/
@[gpu_kernel .SM90]
def crossEntropyRowsVocab3072Bf16Train
    (logits_ptr : GPtr GpuFloat.BFloat16)
    (targets_ptr : GPtr GpuFloat.Int64)
    (losses_ptr : GPtr GpuFloat.Float32)
    (grad_logits_ptr : GPtr GpuFloat.BFloat16)
    (rows : KVal UInt64)
    (grad_scale : KVal Float32) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float warp_reduce[8];"
  emitRaw "__shared__ float row_max;"
  emitRaw "__shared__ float row_sum;"
  emitRaw s!"const __nv_bfloat16* logits = reinterpret_cast<const __nv_bfloat16*>({logits_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const int64_t* targets = reinterpret_cast<const int64_t*>({targets_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* losses = reinterpret_cast<float*>({losses_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* grad_logits = reinterpret_cast<__nv_bfloat16*>({grad_logits_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const uint32_t row = static_cast<uint32_t>(blockIdx.x);"
  emitRaw s!"if (row >= static_cast<uint64_t>({rows.id.toIdent})) return;"
  emitRaw "const uint32_t row_base = row * 3072u;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float values[12];"
  emitRaw "float local_max = -3.402823466e+38F;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 12; ++item) {"
  emitRaw "  const uint32_t col = static_cast<uint32_t>(threadIdx.x) + static_cast<uint32_t>(item) * 256u;"
  emitRaw "  const float value = __bfloat162float(logits[row_base + col]);"
  emitRaw "  values[item] = value;"
  emitRaw "  local_max = fmaxf(local_max, value);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));"
  emitRaw "if (lane == 0) warp_reduce[warp_id] = local_max;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  local_max = lane < 8 ? warp_reduce[lane] : -3.402823466e+38F;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));"
  emitRaw "  if (lane == 0) row_max = local_max;"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "float local_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 12; ++item) local_sum += __expf(values[item] - row_max);"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);"
  emitRaw "if (lane == 0) warp_reduce[warp_id] = local_sum;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  local_sum = lane < 8 ? warp_reduce[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);"
  emitRaw "  if (lane == 0) row_sum = local_sum;"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "const int target = static_cast<int>(targets[row]);"
  emitRaw "if (threadIdx.x == 0) {"
  emitRaw "  const float target_logit = __bfloat162float(logits[row_base + static_cast<uint32_t>(target)]);"
  emitRaw "  losses[row] = logf(row_sum) + row_max - target_logit;"
  emitRaw "}"
  emitRaw "const float inv_sum = 1.0f / row_sum;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 12; ++item) {"
  emitRaw "  const uint32_t col = static_cast<uint32_t>(threadIdx.x) + static_cast<uint32_t>(item) * 256u;"
  emitRaw "  const float probability = __expf(values[item] - row_max) * inv_sum;"
  emitRaw s!"  const float grad = (probability - (static_cast<int>(col) == target ? 1.0f : 0.0f)) * static_cast<float>({grad_scale.id.toIdent});"
  emitRaw "  grad_logits[row_base + col] = __float2bfloat16_rn(grad);"
  emitRaw "}"

/-- Deterministically reduce per-token FP32 losses to one mean loss scalar. -/
@[gpu_kernel .SM90]
def reduceMeanLossRowsF32
    (losses_ptr : GPtr GpuFloat.Float32)
    (mean_loss_ptr : GPtr GpuFloat.Float32)
    (rows : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float warp_sum[8];"
  emitRaw s!"const float* losses = reinterpret_cast<const float*>({losses_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* mean_loss = reinterpret_cast<float*>({mean_loss_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float sum = 0.0f;"
  emitRaw s!"const uint32_t row_count = static_cast<uint32_t>({rows.id.toIdent});"
  emitRaw "for (uint32_t row = static_cast<uint32_t>(threadIdx.x); row < row_count; row += static_cast<uint32_t>(blockDim.x)) sum += losses[row];"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);"
  emitRaw "if (lane == 0) warp_sum[warp_id] = sum;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  sum = lane < 8 ? warp_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);"
  emitRaw "  if (lane == 0) mean_loss[0] = sum / static_cast<float>(row_count);"
  emitRaw "}"

end Tyr.GPU.Kernels.Loss
