/-
  Mixed-precision optimizer kernels used by native training paths.
-/

import Tyr.GPU.Kernels.Prelude

/-! # Optimizer kernels

`Tyr.GPU.Kernels.Optimizer.adamWMasterBf16`: fused AdamW update with FP32 master
parameters and moments and a BF16 working copy.
-/

namespace Tyr.GPU.Kernels.Optimizer

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Fused AdamW update with an FP32 master parameter and moments, a BF16
    gradient, and a rounded BF16 model parameter. Bias-correction reciprocals
    are supplied by the host so the elementwise kernel contains no step-wide
    transcendental work. -/
@[gpu_kernel .SM90]
def adamWMasterBf16
    (master_ptr : GPtr GpuFloat.Float32)
    (model_ptr : GPtr GpuFloat.BFloat16)
    (grad_ptr : GPtr GpuFloat.BFloat16)
    (moment1_ptr : GPtr GpuFloat.Float32)
    (moment2_ptr : GPtr GpuFloat.Float32)
    (elements : KVal UInt64)
    (learning_rate beta1 beta2 epsilon weight_decay
      inv_bias_correction1 inv_bias_correction2 : KVal Float32) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;"
  emitRaw s!"if (idx >= static_cast<uint64_t>({elements.id.toIdent})) return;"
  emitRaw s!"float* master = reinterpret_cast<float*>({master_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* model = reinterpret_cast<__nv_bfloat16*>({model_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* grad = reinterpret_cast<const __nv_bfloat16*>({grad_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* moment1 = reinterpret_cast<float*>({moment1_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* moment2 = reinterpret_cast<float*>({moment2_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const float p = master[idx];"
  emitRaw "const float g = __bfloat162float(grad[idx]);"
  emitRaw s!"const float m = fmaf(static_cast<float>({beta1.id.toIdent}), moment1[idx], (1.0f - static_cast<float>({beta1.id.toIdent})) * g);"
  emitRaw s!"const float v = fmaf(static_cast<float>({beta2.id.toIdent}), moment2[idx], (1.0f - static_cast<float>({beta2.id.toIdent})) * g * g);"
  emitRaw s!"const float v_hat = v * static_cast<float>({inv_bias_correction2.id.toIdent});"
  emitRaw s!"const float denom = sqrtf(v_hat) + static_cast<float>({epsilon.id.toIdent});"
  emitRaw s!"const float normalized = (m * static_cast<float>({inv_bias_correction1.id.toIdent})) / denom;"
  emitRaw s!"const float update = normalized + static_cast<float>({weight_decay.id.toIdent}) * p;"
  emitRaw s!"const float next = fmaf(-static_cast<float>({learning_rate.id.toIdent}), update, p);"
  emitRaw "moment1[idx] = m;"
  emitRaw "moment2[idx] = v;"
  emitRaw "master[idx] = next;"
  emitRaw "model[idx] = __float2bfloat16_rn(next);"

end Tyr.GPU.Kernels.Optimizer
