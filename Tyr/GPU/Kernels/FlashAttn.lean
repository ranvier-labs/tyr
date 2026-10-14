/-
  Tyr/GPU/Kernels/FlashAttn.lean

  FlashAttention forward kernels using the native Lean4 GPU DSL.

  Backward and training-companion kernels live in `Tyr.GPU.Kernels.FlashAttnBwd`.
-/
import Tyr.GPU.Codegen.Macros

import Tyr.GPU.Kernels.Prelude

/-! # FlashAttention example kernel

FlashAttention forward with global-memory I/O
(`Tyr.GPU.Kernels.flashAttnFwdNew`) and a simple GEMM test kernel.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- FlashAttention forward kernel with proper global memory I/O

Demonstrates:
- GlobalLayout parameters for type-safe memory access
- Runtime coordinate computation from blockIdx
- Standard Lean4 `for` loop syntax with index access
- Complete load/store from global memory
-/
@[gpu_kernel .SM90]
def flashAttnFwdNew (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64) : KernelM Unit := do
  let _ := (seq_len, head_dim)
  comment "=== FlashAttention Forward ==="
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 4

  comment "Compute tile coordinates from block index"
  let coord ← blockCoord2D  -- batch=0, depth=0, row=blockIdx.y, col=blockIdx.x

  comment "Declare register tiles"
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  comment "Online softmax state"
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  comment "Declare shared tiles"
  let qShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let kShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let vShared : ST GpuFloat.BFloat16 tileSize tileSize .Col ← allocST .BFloat16 tileSize tileSize .Col
  let oShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize

  comment "Load Q from global memory to shared (long-resident)"
  loadGlobal qShared Q_ptr coord
  sync
  load q qShared

  comment "Main loop over K, V blocks"
  for kvIdx in krange 0 numKvBlocks do
    comment "Load K, V from global memory"
    loadGlobal kShared K_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared V_ptr (coord.withRow kvIdx.id)
    sync

    comment "Load K, V to registers"
    load k kShared
    load v vShared

    comment "S = Q × K^T (bf16 inputs, f32 accumulator)"
    mmaT s q k s

    comment "Apply causal mask"
    makeCausal s s (some (-1e10))

    comment "Online softmax"
    onlineSoftmax s o softmaxState

    comment "Convert to bf16 for V multiply"
    convert p s

    comment "Accumulate O = O + P × V"
    mma o p v o

    sync

  comment "Final normalization: O = O / row_sum"
  finalizeSoftmax o softmaxState

  comment "Store output to global memory"
  let oBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal O_ptr oShared coord

/-- Simple GEMM kernel for testing -/
@[gpu_kernel .SM90]
def simpleGemmNew (A_ptr : GPtr GpuFloat.BFloat16) (B_ptr : GPtr GpuFloat.BFloat16)
    (C_ptr : GPtr GpuFloat.Float32) (M : KVal UInt64) (N : KVal UInt64) (K : KVal UInt64)
    : KernelM Unit := do
  let _ := (M, N, K)
  comment "=== Simple GEMM ==="

  let numKBlocks : Nat := 8

  comment "Compute tile coordinates from block index"
  let coord ← blockCoord2D

  comment "Declare tiles"
  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let cShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64

  comment "Main GEMM loop over K dimension"
  for kIdx in krange 0 numKBlocks do
    comment "Load A and B tiles from global memory"
    loadGlobal aShared A_ptr (coord.withCol kIdx.id)
    loadGlobal bShared B_ptr (coord.withRow kIdx.id)
    sync
    load a aShared
    load b bShared
    mma c a b c
    sync

  comment "Store result to global memory"
  store cShared c
  storeGlobal C_ptr cShared coord

end Tyr.GPU.Kernels
