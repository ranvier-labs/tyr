/-
  Tyr/GPU/Kernels/FusedRMSNorm.lean

  Fused residual + RMSNorm kernels.

  This is the RMSNorm sibling of `Tyr.GPU.Kernels.FusedLayerNorm`: same
  64x1024 tile shape, same residual output surface, but RMS normalization with
  a learnable weight and no bias or mean subtraction.
-/

import Tyr.GPU.Kernels.Prelude

/-! # Fused residual RMSNorm

Fused residual + RMSNorm for $d_{model} = 1024$ (16 × 64 hidden tiles) in BF16
and FP32, with Blackwell-family variants.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev tileDim : Nat := 64
private abbrev rowTileDim : Nat := 16
private abbrev hiddenTileCount : Nat := 16
private abbrev hiddenDim : Nat := tileDim * hiddenTileCount
private def invHiddenDim : Float := 0.0009765625 -- 1 / 1024
private def rmsNormEps : Float := 1.0e-6

private def fusedRMSNormResidual1024Bf16Body
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  comment "Fused residual + RMSNorm on 64x1024 tiles"

  let coord ← blockCoord2D

  let x ← allocRT .BFloat16 rowTileDim tileDim
  let residual ← allocRT .BFloat16 rowTileDim tileDim
  let outResid ← allocRT .BFloat16 rowTileDim tileDim
  let outResidF ← allocRT .Float32 rowTileDim tileDim
  let sq ← allocRT .Float32 rowTileDim tileDim

  let tileShared ← allocST .BFloat16 rowTileDim tileDim

  let sqSum ← zeroRV .Float32 rowTileDim
  let tmpSqSum ← allocRV .Float32 rowTileDim
  let meanSquare ← allocRV .Float32 rowTileDim
  let invRms ← allocRV .Float32 rowTileDim

  let weightShared ← allocSV .BFloat16 tileDim
  let weightRV ← allocRV .BFloat16 tileDim
  let weightF ← allocRV .Float32 tileDim

  comment s!"Pass 1: out_resid and RMS statistics across full hidden dimension ({hiddenDim})"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    convert outResidF outResid
    mul sq outResidF outResidF
    rowSum tmpSqSum sq
    addVec sqSum sqSum tmpSqSum

  scalarMulVec meanSquare sqSum invHiddenDim
  scalarAddVec meanSquare meanSquare rmsNormEps
  rsqrtVec invRms meanSquare

  comment "Pass 2: normalize and apply RMSNorm weight"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared
    convert outResidF outResid

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared
    convertVec weightF weightRV

    mulCol outResidF outResidF invRms
    mulRow outResidF outResidF weightF

    convert outResid outResidF
    store tileShared outResid
    sync
    storeGlobal out_ptr tileShared tileCoord

private def fusedRMSNormResidual1024F32Body
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "Float32 fused residual + RMSNorm on 64x1024 tiles"

  let coord ← blockCoord2D

  let x ← allocRT .Float32 rowTileDim tileDim
  let residual ← allocRT .Float32 rowTileDim tileDim
  let outResid ← allocRT .Float32 rowTileDim tileDim
  let sq ← allocRT .Float32 rowTileDim tileDim

  let tileShared ← allocST .Float32 rowTileDim tileDim

  let sqSum ← zeroRV .Float32 rowTileDim
  let tmpSqSum ← allocRV .Float32 rowTileDim
  let meanSquare ← allocRV .Float32 rowTileDim
  let invRms ← allocRV .Float32 rowTileDim

  let weightShared ← allocSV .Float32 tileDim
  let weightRV ← allocRV .Float32 tileDim

  comment s!"Pass 1: out_resid and RMS statistics across full hidden dimension ({hiddenDim})"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    mul sq outResid outResid
    rowSum tmpSqSum sq
    addVec sqSum sqSum tmpSqSum

  scalarMulVec meanSquare sqSum invHiddenDim
  scalarAddVec meanSquare meanSquare rmsNormEps
  rsqrtVec invRms meanSquare

  comment "Pass 2: normalize and apply RMSNorm weight"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared

    mulCol outResid outResid invRms
    mulRow outResid outResid weightRV

    store tileShared outResid
    sync
    storeGlobal out_ptr tileShared tileCoord

/-- Fused residual + RMSNorm kernel for `d_model = 1024` (`16 x 64` hidden tiles). -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  fusedRMSNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the canonical bf16 fused RMSNorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024Blackwell
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  setFamily .Blackwell
  fusedRMSNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Float32 variant of `tkFusedRMSNormResidual1024`. -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024F32
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  fusedRMSNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the f32 fused RMSNorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024F32Blackwell
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  fusedRMSNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Row-parallel fixed-shape FP32 RMSNorm route for `[64, 1024]`. -/
@[gpu_kernel .SM90]
def fusedRMSNormResidual64x1024F32Direct
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float row_sq_sum[8];"
  emitRaw "__shared__ float row_inv_rms;"
  emitRaw s!"const float* x_raw = reinterpret_cast<const float*>({x_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* residual_raw = reinterpret_cast<const float*>({residual_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* weight_raw = reinterpret_cast<const float*>({weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* out_raw = reinterpret_cast<float*>({out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* out_residual_raw = reinterpret_cast<float*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const int row_base = static_cast<int>(blockIdx.x) * 1024;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float values[4];"
  emitRaw "float square_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  const int idx = row_base + col;"
  emitRaw "  const float value = x_raw[idx] + residual_raw[idx];"
  emitRaw "  values[item] = value;"
  emitRaw "  out_residual_raw[idx] = value;"
  emitRaw "  square_sum = fmaf(value, value, square_sum);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "if (lane == 0) row_sq_sum[warp_id] = square_sum;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  square_sum = lane < 8 ? row_sq_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "  if (lane == 0) row_inv_rms = rsqrtf(square_sum * 0.0009765625f + 1.0e-6f);"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  out_raw[row_base + col] = values[item] * row_inv_rms * weight_raw[col];"
  emitRaw "}"

/-- BF16 counterpart of the row-parallel fixed-shape RMSNorm route. -/
@[gpu_kernel .SM90]
def fusedRMSNormResidual64x1024Bf16Direct
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float row_sq_sum[8];"
  emitRaw "__shared__ float row_inv_rms;"
  emitRaw s!"const __nv_bfloat16* x_raw = reinterpret_cast<const __nv_bfloat16*>({x_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* residual_raw = reinterpret_cast<const __nv_bfloat16*>({residual_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* weight_raw = reinterpret_cast<const __nv_bfloat16*>({weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* out_raw = reinterpret_cast<__nv_bfloat16*>({out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* out_residual_raw = reinterpret_cast<__nv_bfloat16*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const int row_base = static_cast<int>(blockIdx.x) * 1024;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float values[4];"
  emitRaw "float square_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  const int idx = row_base + col;"
  emitRaw "  const float value = __bfloat162float(x_raw[idx]) + __bfloat162float(residual_raw[idx]);"
  emitRaw "  values[item] = value;"
  emitRaw "  out_residual_raw[idx] = __float2bfloat16_rn(value);"
  emitRaw "  square_sum = fmaf(value, value, square_sum);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "if (lane == 0) row_sq_sum[warp_id] = square_sum;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  square_sum = lane < 8 ? row_sq_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "  if (lane == 0) row_inv_rms = rsqrtf(square_sum * 0.0009765625f + 1.0e-6f);"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const int col = static_cast<int>(threadIdx.x) + item * 256;"
  emitRaw "  const float normalized = values[item] * row_inv_rms * __bfloat162float(weight_raw[col]);"
  emitRaw "  out_raw[row_base + col] = __float2bfloat16_rn(normalized);"
  emitRaw "}"

/-- Training RMSNorm forward for arbitrary row counts at hidden size 1024.
    Saves the rounded BF16 residual and FP32 inverse RMS for backward. -/
@[gpu_kernel .SM90]
def fusedRMSNormResidualRows1024Bf16TrainFwd
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16)
    (inv_rms_ptr : GPtr GpuFloat.Float32)
    (rows : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float row_sq_sum[8];"
  emitRaw "__shared__ float row_inv_rms;"
  emitRaw s!"const __nv_bfloat16* x_raw = reinterpret_cast<const __nv_bfloat16*>({x_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* residual_raw = reinterpret_cast<const __nv_bfloat16*>({residual_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* weight_raw = reinterpret_cast<const __nv_bfloat16*>({weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* out_raw = reinterpret_cast<__nv_bfloat16*>({out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* out_residual_raw = reinterpret_cast<__nv_bfloat16*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* inv_rms_raw = reinterpret_cast<float*>({inv_rms_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const uint32_t row = static_cast<uint32_t>(blockIdx.x);"
  emitRaw s!"if (row >= static_cast<uint64_t>({rows.id.toIdent})) return;"
  emitRaw "const uint32_t row_base = row * 1024u;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float values[4];"
  emitRaw "float square_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const uint32_t col = static_cast<uint32_t>(threadIdx.x) + static_cast<uint32_t>(item) * 256u;"
  emitRaw "  const uint32_t idx = row_base + col;"
  emitRaw "  const float sum = __bfloat162float(x_raw[idx]) + __bfloat162float(residual_raw[idx]);"
  emitRaw "  const __nv_bfloat16 rounded = __float2bfloat16_rn(sum);"
  emitRaw "  const float value = __bfloat162float(rounded);"
  emitRaw "  values[item] = value;"
  emitRaw "  out_residual_raw[idx] = rounded;"
  emitRaw "  square_sum = fmaf(value, value, square_sum);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "if (lane == 0) row_sq_sum[warp_id] = square_sum;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  square_sum = lane < 8 ? row_sq_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);"
  emitRaw "  if (lane == 0) {"
  emitRaw "    row_inv_rms = rsqrtf(square_sum * 0.0009765625f + 1.0e-6f);"
  emitRaw "    inv_rms_raw[row] = row_inv_rms;"
  emitRaw "  }"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const uint32_t col = static_cast<uint32_t>(threadIdx.x) + static_cast<uint32_t>(item) * 256u;"
  emitRaw "  const float normalized = values[item] * row_inv_rms * __bfloat162float(weight_raw[col]);"
  emitRaw "  out_raw[row_base + col] = __float2bfloat16_rn(normalized);"
  emitRaw "}"

/-- RMSNorm input/residual VJP at hidden size 1024. The direct saved-residual
    gradient is added to the normalization VJP. This result is both dX and
    dResidual because forward saved x + residual. -/
@[gpu_kernel .SM90]
def fusedRMSNormResidualRows1024Bf16BwdInput
    (grad_out_ptr : GPtr GpuFloat.BFloat16)
    (grad_out_resid_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (inv_rms_ptr : GPtr GpuFloat.Float32)
    (grad_input_ptr : GPtr GpuFloat.BFloat16)
    (rows : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "__shared__ float row_dot_sum[8];"
  emitRaw "__shared__ float row_correction;"
  emitRaw s!"const __nv_bfloat16* grad_out_raw = reinterpret_cast<const __nv_bfloat16*>({grad_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* grad_out_resid_raw = reinterpret_cast<const __nv_bfloat16*>({grad_out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* out_residual_raw = reinterpret_cast<const __nv_bfloat16*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* weight_raw = reinterpret_cast<const __nv_bfloat16*>({weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* inv_rms_raw = reinterpret_cast<const float*>({inv_rms_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* grad_input_raw = reinterpret_cast<__nv_bfloat16*>({grad_input_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const uint32_t row = static_cast<uint32_t>(blockIdx.x);"
  emitRaw s!"if (row >= static_cast<uint64_t>({rows.id.toIdent})) return;"
  emitRaw "const uint32_t row_base = row * 1024u;"
  emitRaw "const int lane = static_cast<int>(threadIdx.x) & 31;"
  emitRaw "const int warp_id = static_cast<int>(threadIdx.x) >> 5;"
  emitRaw "float residual_values[4];"
  emitRaw "float weighted_grads[4];"
  emitRaw "float direct_grads[4];"
  emitRaw "float dot_sum = 0.0f;"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const uint32_t col = static_cast<uint32_t>(threadIdx.x) + static_cast<uint32_t>(item) * 256u;"
  emitRaw "  const uint32_t idx = row_base + col;"
  emitRaw "  const float value = __bfloat162float(out_residual_raw[idx]);"
  emitRaw "  const float weighted_grad = __bfloat162float(grad_out_raw[idx]) * __bfloat162float(weight_raw[col]);"
  emitRaw "  residual_values[item] = value;"
  emitRaw "  weighted_grads[item] = weighted_grad;"
  emitRaw "  direct_grads[item] = __bfloat162float(grad_out_resid_raw[idx]);"
  emitRaw "  dot_sum = fmaf(weighted_grad, value, dot_sum);"
  emitRaw "}"
  emitRaw "#pragma unroll"
  emitRaw "for (int offset = 16; offset > 0; offset >>= 1) dot_sum += __shfl_down_sync(0xffffffffu, dot_sum, offset);"
  emitRaw "if (lane == 0) row_dot_sum[warp_id] = dot_sum;"
  emitRaw "__syncthreads();"
  emitRaw "if (warp_id == 0) {"
  emitRaw "  dot_sum = lane < 8 ? row_dot_sum[lane] : 0.0f;"
  emitRaw "  #pragma unroll"
  emitRaw "  for (int offset = 16; offset > 0; offset >>= 1) dot_sum += __shfl_down_sync(0xffffffffu, dot_sum, offset);"
  emitRaw "  if (lane == 0) {"
  emitRaw "    const float inv = inv_rms_raw[row];"
  emitRaw "    row_correction = dot_sum * inv * inv * inv * 0.0009765625f;"
  emitRaw "  }"
  emitRaw "}"
  emitRaw "__syncthreads();"
  emitRaw "const float inv = inv_rms_raw[row];"
  emitRaw "#pragma unroll"
  emitRaw "for (int item = 0; item < 4; ++item) {"
  emitRaw "  const uint32_t col = static_cast<uint32_t>(threadIdx.x) + static_cast<uint32_t>(item) * 256u;"
  emitRaw "  const uint32_t idx = row_base + col;"
  emitRaw "  const float norm_grad = fmaf(-residual_values[item], row_correction, weighted_grads[item] * inv);"
  emitRaw "  grad_input_raw[idx] = __float2bfloat16_rn(norm_grad + direct_grads[item]);"
  emitRaw "}"

/-- Deterministic FP32 RMSNorm weight-gradient reduction. One thread owns one
    weight column and reduces every row, avoiding atomics and a zeroing pass. -/
@[gpu_kernel .SM90]
def fusedRMSNormResidualRows1024Bf16BwdWeight
    (grad_out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16)
    (inv_rms_ptr : GPtr GpuFloat.Float32)
    (grad_weight_ptr : GPtr GpuFloat.Float32)
    (rows : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw s!"const __nv_bfloat16* grad_out_raw = reinterpret_cast<const __nv_bfloat16*>({grad_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* out_residual_raw = reinterpret_cast<const __nv_bfloat16*>({out_resid_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* inv_rms_raw = reinterpret_cast<const float*>({inv_rms_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"float* grad_weight_raw = reinterpret_cast<float*>({grad_weight_ptr.id.toIdent}.raw_ptr);"
  emitRaw "const uint32_t col = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;"
  emitRaw "if (col >= 1024u) return;"
  emitRaw "float sum = 0.0f;"
  emitRaw s!"const uint32_t row_count = static_cast<uint32_t>({rows.id.toIdent});"
  emitRaw "for (uint32_t row = 0; row < row_count; ++row) {"
  emitRaw "  const uint32_t idx = row * 1024u + col;"
  emitRaw "  const float normalized = __bfloat162float(out_residual_raw[idx]) * inv_rms_raw[row];"
  emitRaw "  sum = fmaf(__bfloat162float(grad_out_raw[idx]), normalized, sum);"
  emitRaw "}"
  emitRaw "grad_weight_raw[col] = sum;"

end Tyr.GPU.Kernels
