/- Rotary embedding forward kernel (ThunderKittens-style tile program). -/

import Tyr.GPU.Kernels.Prelude

/-! # Rotary embedding kernels

RoPE forward and backward kernels on 64×64 tiles and BF16 head-dim-64 Q/K
launches, including the Qwen3-TTS talker variant.
-/

namespace Tyr.GPU.Kernels.Rotary

open Tyr.GPU
open Tyr.GPU.Codegen

/-- RoPE forward on one 64x64 tile.
    Input is split as [x1 | x2] over columns, rotated with sin/cos, and concatenated. -/
@[gpu_kernel .SM90]
def rotaryFwd (x_ptr : GPtr GpuFloat.Float32) (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32) (out_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64) (_head_dim : KVal UInt64) : KernelM Unit := do
  let coord ← blockCoord2D

  let x ← allocRT .Float32 64 64
  let x1 ← allocRT .Float32 64 32
  let x2 ← allocRT .Float32 64 32
  let sinT ← allocRT .Float32 64 32
  let cosT ← allocRT .Float32 64 32
  let y1 ← allocRT .Float32 64 32
  let y2 ← allocRT .Float32 64 32
  let tmp ← allocRT .Float32 64 32
  let negX2 ← allocRT .Float32 64 32
  let out ← allocRT .Float32 64 64

  let xShared ← allocST .Float32 64 64
  let sinShared ← allocST .Float32 64 32
  let cosShared ← allocST .Float32 64 32
  let outShared ← allocST .Float32 64 64

  loadGlobal xShared x_ptr coord
  loadGlobal sinShared sin_ptr coord
  loadGlobal cosShared cos_ptr coord
  sync

  load x xShared
  load sinT sinShared
  load cosT cosShared

  sliceCols x1 x 0 32
  sliceCols x2 x 32 32

  -- y1 = x1*cos - x2*sin
  mul y1 x1 cosT
  neg negX2 x2
  mul negX2 negX2 sinT
  add y1 y1 negX2

  -- y2 = x2*cos + x1*sin
  mul y2 x2 cosT
  mul tmp x1 sinT
  add y2 y2 tmp

  concatCols out y1 y2

  store outShared out
  sync
  storeGlobal out_ptr outShared coord

/-- Direct fixed-shape RoPE route for one contiguous 64x64 FP32 tile. Each
    thread walks a coalesced grid-stride sequence and reads only the input pair
    and its corresponding sine/cosine entry; no shared-memory round trip or
    materialized register tiles are required. -/
@[gpu_kernel .SM90]
def rotaryFwd64x64Direct (x_ptr : GPtr GpuFloat.Float32)
    (sin_ptr : GPtr GpuFloat.Float32) (cos_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw "for (int idx = static_cast<int>(threadIdx.x); idx < 4096; idx += static_cast<int>(blockDim.x)) {"
  emitRaw "  const int col = idx & 63;"
  emitRaw "  const int trig_idx = (idx >> 6) * 32 + (col & 31);"
  emitRaw "  const int pair_idx = col < 32 ? idx + 32 : idx - 32;"
  emitRaw s!"  const float x = reinterpret_cast<const float*>({x_ptr.id.toIdent}.raw_ptr)[idx];"
  emitRaw s!"  const float pair = reinterpret_cast<const float*>({x_ptr.id.toIdent}.raw_ptr)[pair_idx];"
  emitRaw s!"  const float sine = reinterpret_cast<const float*>({sin_ptr.id.toIdent}.raw_ptr)[trig_idx];"
  emitRaw s!"  const float cosine = reinterpret_cast<const float*>({cos_ptr.id.toIdent}.raw_ptr)[trig_idx];"
  emitRaw s!"  reinterpret_cast<float*>({out_ptr.id.toIdent}.raw_ptr)[idx] = col < 32 ? fmaf(x, cosine, -pair * sine) : fmaf(x, cosine, pair * sine);"
  emitRaw "}"

/-- BF16 D64 RoPE forward for Q and K in one launch. Q and K may have
    different row counts (for GQA); rows are flattened `[batch, seq, heads]`.
    One CUDA thread owns one 32-column pair and writes both rotated halves. -/
@[gpu_kernel .SM90]
def rotaryFwdQKD64Bf16Direct
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32)
    (q_out_ptr : GPtr GpuFloat.BFloat16)
    (k_out_ptr : GPtr GpuFloat.BFloat16)
    (q_rows : KVal UInt64)
    (k_rows : KVal UInt64)
    (q_heads : KVal UInt64)
    (k_heads : KVal UInt64)
    (seq_len : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw s!"const __nv_bfloat16* q_raw = reinterpret_cast<const __nv_bfloat16*>({q_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* k_raw = reinterpret_cast<const __nv_bfloat16*>({k_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* sin_raw = reinterpret_cast<const float*>({sin_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* cos_raw = reinterpret_cast<const float*>({cos_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* q_out_raw = reinterpret_cast<__nv_bfloat16*>({q_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* k_out_raw = reinterpret_cast<__nv_bfloat16*>({k_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const uint64_t q_pairs = static_cast<uint64_t>({q_rows.id.toIdent}) * 32ull;"
  emitRaw s!"const uint64_t k_pairs = static_cast<uint64_t>({k_rows.id.toIdent}) * 32ull;"
  emitRaw s!"const uint64_t q_head_count = static_cast<uint64_t>({q_heads.id.toIdent});"
  emitRaw s!"const uint64_t k_head_count = static_cast<uint64_t>({k_heads.id.toIdent});"
  emitRaw s!"const uint64_t sequence = static_cast<uint64_t>({seq_len.id.toIdent});"
  emitRaw "const bool aligned_heads = q_head_count == k_head_count;"
  emitRaw "const uint64_t total_pairs = q_pairs > k_pairs ? q_pairs : k_pairs;"
  emitRaw "for (uint64_t pair_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair_idx < total_pairs; pair_idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {"
  emitRaw "  const uint64_t row = pair_idx >> 5;"
  emitRaw "  const uint64_t col = pair_idx & 31ull;"
  emitRaw "  const uint64_t first = row * 64ull + col;"
  emitRaw "  const uint64_t second = first + 32ull;"
  emitRaw "  float q_sine = 0.0f;"
  emitRaw "  float q_cosine = 0.0f;"
  emitRaw "  if (pair_idx < q_pairs) {"
  emitRaw "    const uint64_t q_position = (row / q_head_count) % sequence;"
  emitRaw "    const uint64_t q_trig_idx = q_position * 32ull + col;"
  emitRaw "    q_sine = sin_raw[q_trig_idx];"
  emitRaw "    q_cosine = cos_raw[q_trig_idx];"
  emitRaw "    const float x1 = __bfloat162float(q_raw[first]);"
  emitRaw "    const float x2 = __bfloat162float(q_raw[second]);"
  emitRaw "    q_out_raw[first] = __float2bfloat16_rn(fmaf(x1, q_cosine, -x2 * q_sine));"
  emitRaw "    q_out_raw[second] = __float2bfloat16_rn(fmaf(x2, q_cosine, x1 * q_sine));"
  emitRaw "  }"
  emitRaw "  if (pair_idx < k_pairs) {"
  emitRaw "    float sine;"
  emitRaw "    float cosine;"
  emitRaw "    if (aligned_heads && pair_idx < q_pairs) {"
  emitRaw "      sine = q_sine;"
  emitRaw "      cosine = q_cosine;"
  emitRaw "    } else {"
  emitRaw "      const uint64_t k_position = (row / k_head_count) % sequence;"
  emitRaw "      const uint64_t k_trig_idx = k_position * 32ull + col;"
  emitRaw "      sine = sin_raw[k_trig_idx];"
  emitRaw "      cosine = cos_raw[k_trig_idx];"
  emitRaw "    }"
  emitRaw "    const float x1 = __bfloat162float(k_raw[first]);"
  emitRaw "    const float x2 = __bfloat162float(k_raw[second]);"
  emitRaw "    k_out_raw[first] = __float2bfloat16_rn(fmaf(x1, cosine, -x2 * sine));"
  emitRaw "    k_out_raw[second] = __float2bfloat16_rn(fmaf(x2, cosine, x1 * sine));"
  emitRaw "  }"
  emitRaw "}"

/-- BF16 D64 RoPE backward for dQ and dK in one launch. This applies the
    inverse rotation and mirrors `rotaryFwdQKD64Bf16Direct` row ownership. -/
@[gpu_kernel .SM90]
def rotaryBwdQKD64Bf16Direct
    (dQ_ptr : GPtr GpuFloat.BFloat16)
    (dK_ptr : GPtr GpuFloat.BFloat16)
    (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32)
    (dQ_out_ptr : GPtr GpuFloat.BFloat16)
    (dK_out_ptr : GPtr GpuFloat.BFloat16)
    (q_rows : KVal UInt64)
    (k_rows : KVal UInt64)
    (q_heads : KVal UInt64)
    (k_heads : KVal UInt64)
    (seq_len : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw s!"const __nv_bfloat16* dq_raw = reinterpret_cast<const __nv_bfloat16*>({dQ_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* dk_raw = reinterpret_cast<const __nv_bfloat16*>({dK_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* sin_raw = reinterpret_cast<const float*>({sin_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* cos_raw = reinterpret_cast<const float*>({cos_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* dq_out_raw = reinterpret_cast<__nv_bfloat16*>({dQ_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* dk_out_raw = reinterpret_cast<__nv_bfloat16*>({dK_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const uint64_t q_pairs = static_cast<uint64_t>({q_rows.id.toIdent}) * 32ull;"
  emitRaw s!"const uint64_t k_pairs = static_cast<uint64_t>({k_rows.id.toIdent}) * 32ull;"
  emitRaw s!"const uint64_t q_head_count = static_cast<uint64_t>({q_heads.id.toIdent});"
  emitRaw s!"const uint64_t k_head_count = static_cast<uint64_t>({k_heads.id.toIdent});"
  emitRaw s!"const uint64_t sequence = static_cast<uint64_t>({seq_len.id.toIdent});"
  emitRaw "const bool aligned_heads = q_head_count == k_head_count;"
  emitRaw "const uint64_t total_pairs = q_pairs > k_pairs ? q_pairs : k_pairs;"
  emitRaw "for (uint64_t pair_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair_idx < total_pairs; pair_idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {"
  emitRaw "  const uint64_t row = pair_idx >> 5;"
  emitRaw "  const uint64_t col = pair_idx & 31ull;"
  emitRaw "  const uint64_t first = row * 64ull + col;"
  emitRaw "  const uint64_t second = first + 32ull;"
  emitRaw "  float q_sine = 0.0f;"
  emitRaw "  float q_cosine = 0.0f;"
  emitRaw "  if (pair_idx < q_pairs) {"
  emitRaw "    const uint64_t q_position = (row / q_head_count) % sequence;"
  emitRaw "    const uint64_t q_trig_idx = q_position * 32ull + col;"
  emitRaw "    q_sine = sin_raw[q_trig_idx];"
  emitRaw "    q_cosine = cos_raw[q_trig_idx];"
  emitRaw "    const float dy1 = __bfloat162float(dq_raw[first]);"
  emitRaw "    const float dy2 = __bfloat162float(dq_raw[second]);"
  emitRaw "    dq_out_raw[first] = __float2bfloat16_rn(fmaf(dy1, q_cosine, dy2 * q_sine));"
  emitRaw "    dq_out_raw[second] = __float2bfloat16_rn(fmaf(dy2, q_cosine, -dy1 * q_sine));"
  emitRaw "  }"
  emitRaw "  if (pair_idx < k_pairs) {"
  emitRaw "    float sine;"
  emitRaw "    float cosine;"
  emitRaw "    if (aligned_heads && pair_idx < q_pairs) {"
  emitRaw "      sine = q_sine;"
  emitRaw "      cosine = q_cosine;"
  emitRaw "    } else {"
  emitRaw "      const uint64_t k_position = (row / k_head_count) % sequence;"
  emitRaw "      const uint64_t k_trig_idx = k_position * 32ull + col;"
  emitRaw "      sine = sin_raw[k_trig_idx];"
  emitRaw "      cosine = cos_raw[k_trig_idx];"
  emitRaw "    }"
  emitRaw "    const float dy1 = __bfloat162float(dk_raw[first]);"
  emitRaw "    const float dy2 = __bfloat162float(dk_raw[second]);"
  emitRaw "    dk_out_raw[first] = __float2bfloat16_rn(fmaf(dy1, cosine, dy2 * sine));"
  emitRaw "    dk_out_raw[second] = __float2bfloat16_rn(fmaf(dy2, cosine, -dy1 * sine));"
  emitRaw "  }"
  emitRaw "}"

/-- Shape-specialized Qwen3-TTS talker RoPE body. The model fixes
    QH=16, KVH=2, S=768, and D=64, so the generated CUDA can strength-reduce
    all head/position arithmetic and keep tensor indices in 32 bits. -/
private def rotaryQwen3TtsTalkerD64Bf16DirectBody
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32)
    (q_out_ptr : GPtr GpuFloat.BFloat16)
    (k_out_ptr : GPtr GpuFloat.BFloat16)
    (q_rows : KVal UInt64)
    (k_rows : KVal UInt64)
    (inverse : Bool) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw s!"const __nv_bfloat16* q_raw = reinterpret_cast<const __nv_bfloat16*>({q_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const __nv_bfloat16* k_raw = reinterpret_cast<const __nv_bfloat16*>({k_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* sin_raw = reinterpret_cast<const float*>({sin_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const float* cos_raw = reinterpret_cast<const float*>({cos_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* q_out_raw = reinterpret_cast<__nv_bfloat16*>({q_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"__nv_bfloat16* k_out_raw = reinterpret_cast<__nv_bfloat16*>({k_out_ptr.id.toIdent}.raw_ptr);"
  emitRaw s!"const uint32_t q_pairs = static_cast<uint32_t>({q_rows.id.toIdent}) * 32u;"
  emitRaw s!"const uint32_t k_pairs = static_cast<uint32_t>({k_rows.id.toIdent}) * 32u;"
  emitRaw "for (uint32_t pair_idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair_idx < q_pairs; pair_idx += static_cast<uint32_t>(gridDim.x) * blockDim.x) {"
  emitRaw "  const uint32_t row = pair_idx >> 5;"
  emitRaw "  const uint32_t col = pair_idx & 31u;"
  emitRaw "  const uint32_t first = (row << 6) + col;"
  emitRaw "  const uint32_t second = first + 32u;"
  emitRaw "  const uint32_t q_position = (row >> 4) % 768u;"
  emitRaw "  const uint32_t q_trig_idx = q_position * 32u + col;"
  emitRaw "  const float q_sine = sin_raw[q_trig_idx];"
  emitRaw "  const float q_cosine = cos_raw[q_trig_idx];"
  emitRaw "  const float q1 = __bfloat162float(q_raw[first]);"
  emitRaw "  const float q2 = __bfloat162float(q_raw[second]);"
  if inverse then
    emitRaw "  q_out_raw[first] = __float2bfloat16_rn(fmaf(q1, q_cosine, q2 * q_sine));"
    emitRaw "  q_out_raw[second] = __float2bfloat16_rn(fmaf(q2, q_cosine, -q1 * q_sine));"
  else
    emitRaw "  q_out_raw[first] = __float2bfloat16_rn(fmaf(q1, q_cosine, -q2 * q_sine));"
    emitRaw "  q_out_raw[second] = __float2bfloat16_rn(fmaf(q2, q_cosine, q1 * q_sine));"
  emitRaw "  if (pair_idx < k_pairs) {"
  emitRaw "    const uint32_t k_position = (row >> 1) % 768u;"
  emitRaw "    const uint32_t k_trig_idx = k_position * 32u + col;"
  emitRaw "    const float k_sine = sin_raw[k_trig_idx];"
  emitRaw "    const float k_cosine = cos_raw[k_trig_idx];"
  emitRaw "    const float k1 = __bfloat162float(k_raw[first]);"
  emitRaw "    const float k2 = __bfloat162float(k_raw[second]);"
  if inverse then
    emitRaw "    k_out_raw[first] = __float2bfloat16_rn(fmaf(k1, k_cosine, k2 * k_sine));"
    emitRaw "    k_out_raw[second] = __float2bfloat16_rn(fmaf(k2, k_cosine, -k1 * k_sine));"
  else
    emitRaw "    k_out_raw[first] = __float2bfloat16_rn(fmaf(k1, k_cosine, -k2 * k_sine));"
    emitRaw "    k_out_raw[second] = __float2bfloat16_rn(fmaf(k2, k_cosine, k1 * k_sine));"
  emitRaw "  }"
  emitRaw "}"

/-- Qwen3-TTS talker BF16 RoPE forward specialization (QH16/KVH2/S768/D64). -/
@[gpu_kernel .SM90]
def rotaryFwdQwen3TtsTalkerD64Bf16Direct
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32)
    (q_out_ptr : GPtr GpuFloat.BFloat16)
    (k_out_ptr : GPtr GpuFloat.BFloat16)
    (q_rows : KVal UInt64)
    (k_rows : KVal UInt64) : KernelM Unit := do
  rotaryQwen3TtsTalkerD64Bf16DirectBody q_ptr k_ptr sin_ptr cos_ptr
    q_out_ptr k_out_ptr q_rows k_rows false

/-- Qwen3-TTS talker BF16 RoPE backward specialization (QH16/KVH2/S768/D64). -/
@[gpu_kernel .SM90]
def rotaryBwdQwen3TtsTalkerD64Bf16Direct
    (dQ_ptr : GPtr GpuFloat.BFloat16)
    (dK_ptr : GPtr GpuFloat.BFloat16)
    (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32)
    (dQ_out_ptr : GPtr GpuFloat.BFloat16)
    (dK_out_ptr : GPtr GpuFloat.BFloat16)
    (q_rows : KVal UInt64)
    (k_rows : KVal UInt64) : KernelM Unit := do
  rotaryQwen3TtsTalkerD64Bf16DirectBody dQ_ptr dK_ptr sin_ptr cos_ptr
    dQ_out_ptr dK_out_ptr q_rows k_rows true

end Tyr.GPU.Kernels.Rotary
