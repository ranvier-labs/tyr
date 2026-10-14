/- Experimental causal FlashAttention kernels for single-tile (`seq=64`, `head_dim=64`) setups.
   This path is exact for one 64x64 attention tile; larger sequence variants must fall back
   until shifted/block-aware causal masking is available in the DSL. -/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Primitives
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Codegen.Attribute

/-! # Causal FlashAttention (64)

Causal FlashAttention forward for one KV block with sequence 64 and head dim 64,
with and without log-sum-exp output.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Causal FlashAttention forward for one KV block (`seq=64`, `d=64`). -/
@[gpu_kernel .SM90]
def tkFlashAttnCausalFwd1Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 1
  let scale : Float := 0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    makeCausal s s (some (-1.0e10))
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- Causal FlashAttention forward with LSE output for one KV block (`seq=64`, `d=64`). -/
@[gpu_kernel .SM90]
def tkFlashAttnCausalFwd1BlockLse
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 1
  let scale : Float := 0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lseShared ← allocSV .Float32 tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    makeCausal s s (some (-1.0e10))
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let lse ← computeLSE softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord
  storeVec lseShared lse
  storeVecGlobalRow lse_ptr lseShared coord

end Tyr.GPU.Kernels
