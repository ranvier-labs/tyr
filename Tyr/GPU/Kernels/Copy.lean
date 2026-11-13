/-
  Tyr/GPU/Kernels/Copy.lean

  Minimal ThunderKittens-style copy kernel:
  copy one 64x64 tile from input to output using global -> shared -> register flow.
-/

import Tyr.GPU.Kernels.Prelude

/-! # Copy kernels

Reference 64×64 FP32 tile copies: tile-based, direct coalesced and
`float4`-vectorized variants (`Tyr.GPU.Kernels.copy64x64Direct`,
`Tyr.GPU.Kernels.copy64x64Float4`).
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

@[gpu_kernel .SM90]
def copy64x64 (input : GPtr GpuFloat.Float32) (output : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "ThunderKittens-style minimal copy kernel"
  let coord ← blockCoord2D
  let reg ← allocRT .Float32 64 64
  let smem ← allocST .Float32 64 64
  loadGlobal smem input coord
  load reg smem
  store smem reg
  storeGlobal output smem coord
  sync

/-- Direct coalesced 64x64 FP32 copy. A 256-thread CTA copies 16 contiguous
    warp-wide stripes without shared memory or register-tile round trips. -/
@[gpu_kernel .SM90]
def copy64x64Direct (input : GPtr GpuFloat.Float32)
    (output : GPtr GpuFloat.Float32) : KernelM Unit := do
  let tid ← getThreadIdx 0 "copy_tid"
  for stripe in List.range 16 do
    let base ← constIntVal (stripe * 256) s!"copy_base_{stripe}"
    let offset ← scalarAddVal tid base s!"copy_offset_{stripe}"
    let value ← loadFloat32Scalar input offset s!"copy_value_{stripe}"
    storeFloat32Scalar output offset value

/-- Vectorized direct copy for the fixed contiguous 64x64 FP32 contract.
    A 512-thread CTA issues two coalesced 16-byte transfers per thread. Torch
    CUDA allocations satisfy `float4` alignment. -/
@[gpu_kernel .SM90]
def copy64x64Float4 (input : GPtr GpuFloat.Float32)
    (output : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  for stripe in List.range 2 do
    emitRaw s!"reinterpret_cast<float4*>({output.id.toIdent}.raw_ptr)[threadIdx.x + {stripe * 512}] = reinterpret_cast<const float4*>({input.id.toIdent}.raw_ptr)[threadIdx.x + {stripe * 512}];"

abbrev tkCopy := copy64x64Float4

end Tyr.GPU.Kernels
