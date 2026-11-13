/-
  Tyr/GPU/Kernels/UlyssesAttn.lean

  Ulysses is modeled here as transport/orchestration around all-to-all, matching
  the vendored ThunderKittens `ulysses_attn.cu`. The actual local attention math
  belongs to a separate FlashAttention launch; this module owns the movement
  between head-parallel and sequence-parallel layouts.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

/-! # Ulysses attention

All-to-all transport for Ulysses sequence parallelism: redistributing Q/K/V
between head-parallel and sequence-parallel shards and returning outputs.
-/

namespace Tyr.GPU.Kernels.UlyssesAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Standalone ThunderKittens-shaped all-to-all transport kernel.
The caller supplies already-partitioned views and chooses the axis mapping by
which input/output pointers it passes in. -/
@[gpu_kernel .SM90]
def allToAllFwd (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    (scatter_axis : KVal UInt32) (gather_axis : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size, scatter_axis, gather_axis)
  comment "Ulysses all_to_all transport surface; axis selection is an orchestration concern"
  Support.allToAllTile "generic shard" output_ptr input_ptr dev_idx world_size scatter_axis gather_axis
  Support.barrierAllDevices "all_to_all epilogue" 1

/-- Phase 1/2 of Ulysses forward: redistribute Q/K/V from head-parallel shards
into the sequence-parallel views consumed by a local FlashAttention launch. -/
@[gpu_kernel .SM90]
def ulyssesQkvAllToAll (q_seq_ptr : GPtr GpuFloat.BFloat16) (k_seq_ptr : GPtr GpuFloat.BFloat16)
    (v_seq_ptr : GPtr GpuFloat.BFloat16)
    (q_heads_ptr : GPtr GpuFloat.BFloat16) (k_heads_ptr : GPtr GpuFloat.BFloat16)
    (v_heads_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let scatter ← constIntVal 1 "scatter_depth"
  let gather ← constIntVal 2 "gather_rows"

  comment "Ulysses QKV transport: head-parallel -> sequence-parallel"
  Support.allToAllTile "Q heads->seq" q_seq_ptr q_heads_ptr dev_idx world_size scatter gather
  Support.allToAllTile "K heads->seq" k_seq_ptr k_heads_ptr dev_idx world_size scatter gather
  Support.allToAllTile "V heads->seq" v_seq_ptr v_heads_ptr dev_idx world_size scatter gather
  Support.barrierAllDevices "QKV transport complete" 1

/-- Phase 4 of Ulysses forward: return the local attention output from the
sequence-parallel layout back to the head-parallel view. -/
@[gpu_kernel .SM90]
def ulyssesAttnFwd (o_heads_ptr : GPtr GpuFloat.BFloat16) (o_seq_ptr : GPtr GpuFloat.BFloat16)
    (q_seq_ptr : GPtr GpuFloat.BFloat16) (k_seq_ptr : GPtr GpuFloat.BFloat16)
    (v_seq_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (q_seq_ptr, k_seq_ptr, v_seq_ptr, dev_idx, world_size)
  let scatter ← constIntVal 2 "scatter_rows"
  let gather ← constIntVal 1 "gather_depth"

  comment "Ulysses orchestration shell:"
  comment "1. `ulyssesQkvAllToAll` runs before local attention."
  comment "2. A separate FlashAttention launch consumes q_seq/k_seq/v_seq."
  comment "3. This kernel returns the local attention output back to head shards."
  Support.allToAllTile "O seq->heads" o_heads_ptr o_seq_ptr dev_idx world_size scatter gather
  Support.barrierAllDevices "output return complete" 1

end Tyr.GPU.Kernels.UlyssesAttn
