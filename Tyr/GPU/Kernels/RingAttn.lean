/-
  Tyr/GPU/Kernels/RingAttn.lean

  Ring attention surfaces split into the same coarse phases the vendored
  ThunderKittens kernel exposes:

  1. partial attention against the current KV ring buffer,
  2. communication to advance K/V into the next ping-pong buffer,
  3. numerically stable reduction of running O/L statistics.

  The Lean DSL still cannot encode the exact multi-launch schedule or peer index
  arithmetic from `ring_attn_h100.cu`, so the kernels below treat the caller's
  pointers as already-partitioned current/next views.
-/

import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

/-! # Ring attention

Three-phase ring attention: partial attention on the local K/V shard, shard
rotation, and merging the running $(O, L)$ state.
-/

namespace Tyr.GPU.Kernels.RingAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Ring phase 1: compute attention for the current K/V shard and emit a partial
output tile plus its per-row log-sum-exp summary. -/
@[gpu_kernel .SM90]
def ringAttnPartial (Q_ptr : GPtr GpuFloat.BFloat16) (K_curr_ptr : GPtr GpuFloat.BFloat16)
    (V_curr_ptr : GPtr GpuFloat.BFloat16) (O_partial_ptr : GPtr GpuFloat.Float32)
    (L_partial_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (seq_len, head_dim, ring_stage, dev_idx, world_size)
  comment "Ring partial phase: stationary Q, streamed current-ring K/V"
  comment "This is the typed single-query-tile shell over the current shard; comm/reduce remain separate kernels."

  let tileRows ← constIntVal 128 "ring_partial_tile_rows"
  let zero ← constIntVal 0 "ring_partial_zero"
  let rowIdx : KVal UInt32 := ⟨← getBlockIdxX, "ring_partial_row_idx"⟩
  let headIdx : KVal UInt32 := ⟨← getBlockIdxY, "ring_partial_head_idx"⟩
  let batchIdx : KVal UInt32 := ⟨← getBlockIdxZ, "ring_partial_batch_idx"⟩
  let qCoord := makeRTileCoord batchIdx.id headIdx.id rowIdx.id zero.id
  let kvBlocksTotal ← layoutRows K_curr_ptr "ring_partial_total_rows"
  let numKvBlocks ← scalarDivVal kvBlocksTotal tileRows "ring_partial_num_kv_blocks"

  let qShared : ST GpuFloat.BFloat16 64 128 ← allocST .BFloat16 64 128
  let kShared : ST GpuFloat.BFloat16 128 128 ← allocST .BFloat16 128 128
  let vShared : ST GpuFloat.BFloat16 128 128 .Col ← allocST .BFloat16 128 128 .Col
  let oShared : ST GpuFloat.Float32 64 128 ← allocST .Float32 64 128
  let lShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  let q : RT GpuFloat.BFloat16 64 128 ← allocRT .BFloat16 64 128
  let k : RT GpuFloat.BFloat16 128 128 ← allocRT .BFloat16 128 128
  let v : RT GpuFloat.BFloat16 128 128 .Col ← allocRT .BFloat16 128 128 .Col
  let o : RT GpuFloat.Float32 64 128 ← zeroRT .Float32 64 128
  let softmaxState ← allocSoftmaxState .Float32 64

  Support.asyncTileLoad qShared Q_ptr qCoord (64 * 128 * GpuFloat.bytes .BFloat16)
  sync
  load q qShared

  for kvIdx in kvrange 0 numKvBlocks do
    let kvCoord := makeRTileCoord batchIdx.id headIdx.id kvIdx.id zero.id
    let scores : RT GpuFloat.Float32 64 128 ← zeroRT .Float32 64 128
    let probs : RT GpuFloat.BFloat16 64 128 ← allocRT .BFloat16 64 128
    Support.asyncTileLoad kShared K_curr_ptr kvCoord (128 * 128 * GpuFloat.bytes .BFloat16)
    Support.asyncTileLoad vShared V_curr_ptr kvCoord (128 * 128 * GpuFloat.bytes .BFloat16)
    sync
    load k kShared
    load v vShared
    mmaT scores q k scores
    scalarMul scores scores 0.08838834764
    onlineSoftmax scores o softmaxState
    convert probs scores
    mma o probs v o
    sync

  finalizeSoftmax o softmaxState
  let lse ← computeLSE softmaxState
  store oShared o
  storeVec lShared lse
  storeGlobal O_partial_ptr oShared qCoord
  storeVecGlobal L_partial_ptr lShared qCoord

/-- Ring phase 2: move the current K/V shard into the caller-provided "next"
buffers. This is the explicit communication stage the old monolithic kernel hid. -/
@[gpu_kernel .SM90]
def ringAttnComm (K_next_ptr : GPtr GpuFloat.BFloat16) (V_next_ptr : GPtr GpuFloat.BFloat16)
    (K_curr_ptr : GPtr GpuFloat.BFloat16) (V_curr_ptr : GPtr GpuFloat.BFloat16)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (ring_stage, dev_idx, world_size)
  comment "Ring communication phase: current K/V -> next ping-pong buffer"
  let tileRows ← constIntVal 128 "ring_comm_tile_rows"
  let zero ← constIntVal 0 "ring_comm_zero"
  let taskIdx : KVal UInt32 := ⟨← getBlockIdxX, "ring_comm_task"⟩
  let _batchCount ← layoutBatch K_curr_ptr "ring_comm_batch_count"
  let headCount ← layoutDepth K_curr_ptr "ring_comm_head_count"
  let totalRows ← layoutRows K_curr_ptr "ring_comm_total_rows"
  let rowBlocks ← scalarDivVal totalRows tileRows "ring_comm_row_blocks"
  let blocksPerBatch ← scalarMulVal headCount rowBlocks "ring_comm_blocks_per_batch"
  let batchIdx ← scalarDivVal taskIdx blocksPerBatch "ring_comm_batch_idx"
  let afterBatch ← scalarMod taskIdx blocksPerBatch "ring_comm_after_batch"
  let headIdx ← scalarDivVal afterBatch rowBlocks "ring_comm_head_idx"
  let rowIdx ← scalarMod afterBatch rowBlocks "ring_comm_row_idx"
  let coord := makeRTileCoord batchIdx.id headIdx.id rowIdx.id zero.id

  let kShared : ST GpuFloat.BFloat16 128 128 ← allocST .BFloat16 128 128
  let vShared : ST GpuFloat.BFloat16 128 128 ← allocST .BFloat16 128 128
  Support.asyncTileLoad kShared K_curr_ptr coord (128 * 128 * GpuFloat.bytes .BFloat16)
  Support.asyncTileLoad vShared V_curr_ptr coord (128 * 128 * GpuFloat.bytes .BFloat16)
  storeGlobalAsync K_next_ptr kShared coord
  storeGlobalAsync V_next_ptr vShared coord
  Support.barrierAllDevices "ring K/V epilogue" 1

/-- Ring phase 3: merge the running `(O, L)` state with the current partial
contribution using the standard log-sum-exp reduction. -/
@[gpu_kernel .SM90]
def ringAttnReduce (O_running_ptr : GPtr GpuFloat.Float32) (O_block_ptr : GPtr GpuFloat.Float32)
    (L_running_ptr : GPtr GpuFloat.Float32) (L_block_ptr : GPtr GpuFloat.Float32)
    (O_out_ptr : GPtr GpuFloat.BFloat16) (L_out_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  let _ := (seq_len, head_dim)
  comment "Ring reduction phase: stable merge of running and current partial tiles"
  let rowIdx : KVal UInt32 := ⟨← getBlockIdxX, "ring_reduce_row_idx"⟩
  let headIdx : KVal UInt32 := ⟨← getBlockIdxY, "ring_reduce_head_idx"⟩
  let batchIdx : KVal UInt32 := ⟨← getBlockIdxZ, "ring_reduce_batch_idx"⟩
  let zero ← constIntVal 0 "ring_reduce_zero"
  let coord := makeRTileCoord batchIdx.id headIdx.id rowIdx.id zero.id

  let oRunningShared : ST GpuFloat.Float32 128 128 ← allocST .Float32 128 128
  let oBlockShared : ST GpuFloat.Float32 128 128 ← allocST .Float32 128 128
  let oOutShared : ST GpuFloat.BFloat16 128 128 ← allocST .BFloat16 128 128
  let lRunningShared : SV GpuFloat.Float32 128 ← allocSV .Float32 128
  let lBlockShared : SV GpuFloat.Float32 128 ← allocSV .Float32 128
  let lOutShared : SV GpuFloat.Float32 128 ← allocSV .Float32 128

  let oRunning : RT GpuFloat.Float32 128 128 ← allocRT .Float32 128 128
  let oBlock : RT GpuFloat.Float32 128 128 ← allocRT .Float32 128 128
  let oRunningScaled : RT GpuFloat.Float32 128 128 ← allocRT .Float32 128 128
  let oBlockScaled : RT GpuFloat.Float32 128 128 ← allocRT .Float32 128 128
  let oMerged : RT GpuFloat.Float32 128 128 ← allocRT .Float32 128 128
  let oOut : RT GpuFloat.BFloat16 128 128 ← allocRT .BFloat16 128 128

  let lRunning : RV GpuFloat.Float32 128 ← allocRV .Float32 128
  let lBlock : RV GpuFloat.Float32 128 ← allocRV .Float32 128
  let lMerged : RV GpuFloat.Float32 128 ← allocRV .Float32 128
  let lRunningScale : RV GpuFloat.Float32 128 ← allocRV .Float32 128
  let lBlockScale : RV GpuFloat.Float32 128 ← allocRV .Float32 128

  Support.asyncTileLoad oRunningShared O_running_ptr coord (128 * 128 * GpuFloat.bytes .Float32)
  Support.asyncTileLoad oBlockShared O_block_ptr coord (128 * 128 * GpuFloat.bytes .Float32)
  loadVecGlobal lRunningShared L_running_ptr coord
  loadVecGlobal lBlockShared L_block_ptr coord
  sync

  load oRunning oRunningShared
  load oBlock oBlockShared
  loadVec lRunning lRunningShared
  loadVec lBlock lBlockShared

  subVec lMerged lBlock lRunning
  expVec lMerged lMerged
  scalarAddVec lMerged lMerged 1.0
  logVec lMerged lMerged
  addVec lMerged lMerged lRunning

  subVec lRunningScale lRunning lMerged
  expVec lRunningScale lRunningScale
  subVec lBlockScale lBlock lMerged
  expVec lBlockScale lBlockScale

  copy oRunningScaled oRunning
  mulCol oRunningScaled oRunningScaled lRunningScale
  copy oBlockScaled oBlock
  mulCol oBlockScaled oBlockScaled lBlockScale
  add oMerged oRunningScaled oBlockScaled
  convert oOut oMerged

  store oOutShared oOut
  storeVec lOutShared lMerged
  storeGlobal O_out_ptr oOutShared coord
  storeVecGlobal L_out_ptr lOutShared coord

end Tyr.GPU.Kernels.RingAttn
