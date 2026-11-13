/-
  Tyr/GPU/Kernels/RingAttnBwd.lean

  This module is explicitly a speculative backward scaffold. The forward path is
  now split into partial/comm/reduction kernels; backward mirrors that split in
  structure, but the exact global causal indexing and multi-stage accumulation
  from the vendored ThunderKittens kernel are still not faithfully represented.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

/-! # Ring attention backward

Speculative backward shell for ring attention: partial and communication steps
and finalization (`Tyr.GPU.Kernels.RingAttn.ringAttnBwd`).
-/

namespace Tyr.GPU.Kernels.RingAttn

open Tyr.GPU
open Tyr.GPU.Codegen

private def ringBwdPartialStep
    (q : RT GpuFloat.BFloat16 64 64) (dO : RT GpuFloat.BFloat16 64 64)
    (k : RT GpuFloat.BFloat16 64 64) (v : RT GpuFloat.BFloat16 64 64 .Col)
    (lseVec : RV GpuFloat.Float32 64) (dVec : RV GpuFloat.Float32 64)
    (dQ : RT GpuFloat.Float32 64 64) (dK : RT GpuFloat.Float32 64 64)
    (dV : RT GpuFloat.Float32 64 64) : KernelM Unit := do
  let s : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let p : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let dP : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let dS : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let dSBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  comment "Speculative partial backward step for the current ring shard"
  mmaT s q k s
  comment "Causal masking is still local-only here; global ring offsets are not modeled yet"
  makeCausal s s (some (-1.0e10))
  subCol s s lseVec
  exp p s

  let vRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  swapLayout vRow v
  mmaT dP dO vRow dP
  subCol dP dP dVec
  mul dS p dP
  makeCausal dS dS (some 0.0)
  convert dSBf dS

  let kCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  swapLayout kCol k
  mma dQ dSBf kCol dQ

  let dST : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  transpose dST dSBf
  let qCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  swapLayout qCol q
  mma dK dST qCol dK

  let pBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let pT : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  convert pBf p
  transpose pT pBf
  let dOCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  swapLayout dOCol dO
  mma dV pT dOCol dV

private def ringBwdCommStep
    (k : RT GpuFloat.BFloat16 64 64) (v : RT GpuFloat.BFloat16 64 64 .Col)
    (dK : RT GpuFloat.Float32 64 64) (dV : RT GpuFloat.Float32 64 64)
    (kShared : ST GpuFloat.BFloat16 64 64) (vShared : ST GpuFloat.BFloat16 64 64 .Col)
    (dKShared : ST GpuFloat.Float32 64 64) (dVShared : ST GpuFloat.Float32 64 64)
    : KernelM Unit := do
  comment "Speculative ring communication phase: publish K/V and accumulated dK/dV"
  store kShared k
  store vShared v
  store dKShared dK
  store dVShared dV
  multimemStore kShared k
  multimemStore vShared v
  multimemStore dKShared dK
  multimemStore dVShared dV
  Support.barrierAllDevices "ring backward communication complete" 0
  load k kShared
  load v vShared
  load dK dKShared
  load dV dVShared

private def ringBwdFinalize
    (dQ : RT GpuFloat.Float32 64 64) (dK : RT GpuFloat.Float32 64 64)
    (dV : RT GpuFloat.Float32 64 64) (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32) (dV_ptr : GPtr GpuFloat.Float32)
    (coord : RTileCoord) : KernelM Unit := do
  let dQShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let dKShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let dVShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  store dQShared dQ
  store dKShared dK
  store dVShared dV
  storeGlobal dQ_ptr dQShared coord
  storeGlobalAdd dK_ptr dKShared coord
  storeGlobalAdd dV_ptr dVShared coord

/-- Speculative ring backward shell.
It preserves the intended phase split:
1. partial derivative computation for the current K/V shard,
2. communication of the circulating state,
3. final writeback of stationary dQ and accumulated dK/dV. -/
@[gpu_kernel .SM90]
def ringAttnBwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32) (D_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32) (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (rank : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (O_ptr, rank, world_size)
  let coord ← blockCoord2D

  comment "=== Ring Attention Backward (speculative phased scaffold) ==="

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let dO : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let dQ : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let dK : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let dV : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let dOShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let dKShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let dVShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let lseShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let dVecShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  let lseVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let dVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  Support.asyncTileLoad qShared Q_ptr coord (64 * 64 * 2)
  Support.asyncTileLoad dOShared dO_ptr coord (64 * 64 * 2)
  Support.asyncTileLoad kShared K_ptr coord (64 * 64 * 2)
  Support.asyncTileLoad vShared V_ptr coord (64 * 64 * 2)
  loadVecGlobalRow lseShared L_ptr coord
  loadVecGlobalRow dVecShared D_ptr coord
  sync
  load q qShared
  load dO dOShared
  load k kShared
  load v vShared
  loadVec lseVec lseShared
  loadVec dVec dVecShared

  ringBwdPartialStep q dO k v lseVec dVec dQ dK dV
  ringBwdCommStep k v dK dV kShared vShared dKShared dVShared
  ringBwdFinalize dQ dK dV dQ_ptr dK_ptr dV_ptr coord
  Support.barrierAllDevices "ring backward finalize" 1

end Tyr.GPU.Kernels.RingAttn
