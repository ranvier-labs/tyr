/- Reduced GB10/Blackwell MHA kernels for the 2-block 128x64 path. -/
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Kernels.Prelude

/-! # GB10 multi-head attention

FlashAttention forward and backward kernels tuned for GB10/Blackwell (sequence
128, head dim 64), including pipelined and long-sequence schedules.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- GB10/Blackwell FlashAttention forward for two KV blocks (seq=128, d=64). -/
@[gpu_kernel .SM90]
def tkFlashAttnGb10Fwd2Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
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
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- GB10/Blackwell FlashAttention forward for two KV blocks with LSE output. -/
@[gpu_kernel .SM90]
def tkFlashAttnGb10Fwd2BlockLse
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
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

/-- GB10/Blackwell reduced MHA forward runtime-sequence short schedule. -/
@[gpu_kernel .SM90]
def tkMhaGb10Fwd2Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rowTileSize : Nat := 16
  let blockN64 ← constUInt64Val (tileSize : Int) "block_n"
  let blockNMinusOne ← constUInt64Val ((tileSize - 1 : Nat) : Int) "block_n_minus_one"
  let seqPlusPad ← scalarAddVal seq_len blockNMinusOne "seq_plus_pad"
  let numKvBlocks ← scalarDivVal seqPlusPad blockN64 "num_kv_blocks"
  let numKvBlocks32 : KVal UInt32 ← castScalar .UInt32 numKvBlocks "num_kv_blocks_u32"
  let seqLen32 : KVal UInt32 ← castScalar .UInt32 seq_len "seq_len_u32"
  let blockN32 : KVal UInt32 ← constIntVal (tileSize : Int) "block_n_u32"
  let scale : Float := 0.125
  let lScale : Float := -8.0

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 rowTileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 rowTileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 rowTileSize

  let qShared ← allocST .BFloat16 rowTileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 rowTileSize tileSize
  let lShared ← allocSV .Float32 rowTileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in kvrange 0 numKvBlocks32 do
    let s ← zeroRT .Float32 rowTileSize tileSize
    let p ← allocRT .BFloat16 rowTileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    let kvIdxVal : KVal UInt32 := ⟨kvIdx.id, "kv_idx"⟩
    let kvBase ← scalarMulVal kvIdxVal blockN32 "kv_base"
    let cutoff ← scalarSub seqLen32 kvBase "kv_cutoff"
    let needsMask ← scalarLt cutoff blockN32 "needs_tail_mask"
    emitIf needsMask.id do
      rightFillVal s s cutoff (some (-3.4028234663852886e38))
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let l ← computeLSE softmaxState
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 rowTileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lShared l
  storeVecGlobalRow l_ptr lShared coord

/-- GB10/Blackwell specialized MHA forward for S768 D64 with two-stage KV pipeline. -/
@[gpu_kernel .SM90]
def tkMhaGb10FwdS768D64Pipelined
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64)
    (_head_dim : KVal UInt64)
    (q_heads : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rowTileSize : Nat := 16
  let fixedKvBlocks : Nat := 12
  let kvBytes : Nat := tileSize * tileSize * GpuFloat.bytes .BFloat16
  let blockN64 ← constUInt64Val (tileSize : Int) "block_n"
  let blockNMinusOne ← constUInt64Val ((tileSize - 1 : Nat) : Int) "block_n_minus_one"
  let seqPlusPad ← scalarAddVal seq_len blockNMinusOne "seq_plus_pad"
  let numKvBlocks ← scalarDivVal seqPlusPad blockN64 "num_kv_blocks"
  let numKvBlocks32 : KVal UInt32 ← castScalar .UInt32 numKvBlocks "num_kv_blocks_u32"
  let seqLen32 : KVal UInt32 ← castScalar .UInt32 seq_len "seq_len_u32"
  let blockN32 : KVal UInt32 ← constIntVal (tileSize : Int) "block_n_u32"
  let scale : Float := 0.125
  let lScale : Float := -8.0

  let zero ← constIntVal 0 "s768_zero"
  let queryBlock ← getBlockIdx 0 "s768_query_block"
  let headIdx ← getBlockIdx 1 "s768_head"
  let batchIdx ← getBlockIdx 2 "s768_batch"
  let coord := makeRTileCoord batchIdx.id headIdx.id queryBlock.id zero.id
  let qHeads32 : KVal UInt32 ← castScalar .UInt32 q_heads "s768_q_heads"
  let queryBlocks ← constIntVal 48 "s768_query_blocks"
  let batchHeadBase ← scalarMulVal batchIdx qHeads32 "s768_batch_head_base"
  let flatHead ← scalarAddVal batchHeadBase headIdx "s768_flat_head"
  let flatLBase ← scalarMulVal flatHead queryBlocks "s768_flat_l_base"
  let flatLRow ← scalarAddVal flatLBase queryBlock "s768_flat_l_row"
  let lCoord := coord.withRow flatLRow.id

  let q ← allocRT .BFloat16 rowTileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 rowTileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 rowTileSize

  let qShared ← allocST .BFloat16 rowTileSize tileSize
  let kShared0 ← allocST .BFloat16 tileSize tileSize
  let kShared1 ← allocST .BFloat16 tileSize tileSize
  let vShared0 ← allocST .BFloat16 tileSize tileSize .Col
  let vShared1 ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 rowTileSize tileSize
  let lShared ← allocSV .Float32 rowTileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  let kvSem0 ← allocSemaphore
  let kvSem1 ← allocSemaphore
  initSemaphore kvSem0 1
  initSemaphore kvSem1 1
  let issueKv (kShared : ST GpuFloat.BFloat16 tileSize tileSize)
      (vShared : ST GpuFloat.BFloat16 tileSize tileSize .Col)
      (sem : Semaphore) (kvIdx : Nat) : KernelM Unit := do
    let kvIdxVal ← constIntVal kvIdx s!"s768_kv_{kvIdx}"
    let kvCoord := coord.withRow kvIdxVal.id
    expectBytes sem (2 * kvBytes)
    loadGlobalAsync kShared k_ptr kvCoord sem.id
    loadGlobalAsync vShared v_ptr kvCoord sem.id
  issueKv kShared0 vShared0 kvSem0 0
  issueKv kShared1 vShared1 kvSem1 1

  for kvIdx in List.range fixedKvBlocks do
    let s ← zeroRT .Float32 rowTileSize tileSize
    let p ← allocRT .BFloat16 rowTileSize tileSize
    let phase ← constIntVal (kvIdx / 2) s!"s768_phase_{kvIdx}"
    if kvIdx % 2 == 0 then
      waitSemaphorePhaseVal kvSem0 phase
      load k kShared0
      load v vShared0
      if kvIdx + 2 < fixedKvBlocks then
        issueKv kShared0 vShared0 kvSem0 (kvIdx + 2)
    else
      waitSemaphorePhaseVal kvSem1 phase
      load k kShared1
      load v vShared1
      if kvIdx + 2 < fixedKvBlocks then
        issueKv kShared1 vShared1 kvSem1 (kvIdx + 2)

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o

  finalizeSoftmax o softmaxState
  let l ← computeLSE softmaxState
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 rowTileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lShared l
  storeVecGlobalRow l_ptr lShared lCoord

/-- GB10/SM121 S768 forward using four independent warp MMAs per CTA.
    The warps own 32 query rows each and reuse each staged K/V tile across all
    128 query rows. K and V use separate single-buffer TMA pipelines so their
    full register tiles do not overlap. This uses ordinary warp MMA on GB10;
    Hopper
    WGMMA is a separate architecture-specific lowering. -/
@[gpu_kernel .SM90]
def tkMhaGb10FwdS768D64Warp4
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64)
    (q_heads : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let warpRows : Nat := 32
  let fixedKvBlocks : Nat := 12
  let tileBytes : Nat := tileSize * tileSize * GpuFloat.bytes .BFloat16
  let scale : Float := 0.125
  let scoreScaleLog2e : Float := 0.18033688011112042
  let lScale : Float := -8.0

  let zero ← constIntVal 0 "warp4_zero"
  let queryBlock ← getBlockIdx 0 "warp4_query_block"
  let headIdx ← getBlockIdx 1 "warp4_head"
  let batchIdx ← getBlockIdx 2 "warp4_batch"
  let coord := makeRTileCoord batchIdx.id headIdx.id queryBlock.id zero.id
  let warpId ← getWarpId "warp4_warp"
  let warpZero ← scalarEq warpId zero "warp4_is_warp_zero"
  let qHeads32 : KVal UInt32 ← castScalar .UInt32 q_heads "warp4_q_heads"
  let queryWarpBlocks ← constIntVal 24 "warp4_query_warp_blocks"
  let four ← constIntVal 4 "warp4_four"
  let batchHeadBase ← scalarMulVal batchIdx qHeads32 "warp4_batch_head_base"
  let flatHead ← scalarAddVal batchHeadBase headIdx "warp4_flat_head"
  let flatLBase ← scalarMulVal flatHead queryWarpBlocks "warp4_flat_l_base"
  let queryWarpBase ← scalarMulVal queryBlock four "warp4_query_warp_base"
  let queryWarpRow ← scalarAddVal queryWarpBase warpId "warp4_query_warp_row"
  let flatLRow ← scalarAddVal flatLBase queryWarpRow "warp4_flat_l_row"
  let lCoord := coord.withRow flatLRow.id
  let oCoord := coord.withRow queryWarpBase.id

  let q ← allocRT .BFloat16 warpRows tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 warpRows tileSize
  let softmaxState ← allocSoftmaxState .Float32 warpRows

  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col

  warpgroupLoadGlobal q q_ptr oCoord

  let kSem ← allocSemaphore
  let vSem ← allocSemaphore
  initSemaphore kSem 1
  initSemaphore vSem 1
  let issueK (kvIdx : Nat) : KernelM Unit := do
    let kvIdxVal ← constIntVal kvIdx s!"warp4_k_{kvIdx}"
    let kvCoord := coord.withRow kvIdxVal.id
    expectBytes kSem tileBytes
    loadGlobalAsync kShared k_ptr kvCoord kSem.id
  let issueV (kvIdx : Nat) : KernelM Unit := do
    let kvIdxVal ← constIntVal kvIdx s!"warp4_v_{kvIdx}"
    let kvCoord := coord.withRow kvIdxVal.id
    expectBytes vSem tileBytes
    loadGlobalAsync vShared v_ptr kvCoord vSem.id
  issueK 0
  issueV 0

  for kvIdx in List.range fixedKvBlocks do
    let s ← zeroRT .Float32 warpRows tileSize
    let p ← allocRT .BFloat16 warpRows tileSize
    let phase ← constIntVal (kvIdx % 2) s!"warp4_phase_{kvIdx}"
    emitIf warpZero.id do
      waitSemaphorePhaseVal kSem phase
      waitSemaphorePhaseVal vSem phase
    blockSync
    load k kShared
    blockSync
    if kvIdx + 1 < fixedKvBlocks then issueK (kvIdx + 1)

    mmaT s q k s
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s

    load v vShared
    blockSync
    if kvIdx + 1 < fixedKvBlocks then issueV (kvIdx + 1)
    mma o p v o

  finalizeSoftmax o softmaxState
  let l ← computeLSEScaled softmaxState scale
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 warpRows tileSize
  convert oBf16 o
  warpgroupStoreGlobal o_ptr oBf16 oCoord

  storeVecGlobalRowRV l_ptr l lCoord

/-- GB10/Blackwell reduced MHA forward long-context schedule. -/
@[gpu_kernel .SM90]
def tkMhaGb10FwdLong
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rowTileSize : Nat := 32
  let blockN64 ← constUInt64Val (tileSize : Int) "block_n"
  let blockNMinusOne ← constUInt64Val ((tileSize - 1 : Nat) : Int) "block_n_minus_one"
  let seqPlusPad ← scalarAddVal seq_len blockNMinusOne "seq_plus_pad"
  let numKvBlocks ← scalarDivVal seqPlusPad blockN64 "num_kv_blocks"
  let numKvBlocks32 : KVal UInt32 ← castScalar .UInt32 numKvBlocks "num_kv_blocks_u32"
  let seqLen32 : KVal UInt32 ← castScalar .UInt32 seq_len "seq_len_u32"
  let blockN32 : KVal UInt32 ← constIntVal (tileSize : Int) "block_n_u32"
  let scale : Float := 0.125
  let lScale : Float := -8.0

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 rowTileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 rowTileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 rowTileSize

  let qShared ← allocST .BFloat16 rowTileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 rowTileSize tileSize
  let lShared ← allocSV .Float32 rowTileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in kvrange 0 numKvBlocks32 do
    let s ← zeroRT .Float32 rowTileSize tileSize
    let p ← allocRT .BFloat16 rowTileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    let kvIdxVal : KVal UInt32 := ⟨kvIdx.id, "kv_idx"⟩
    let kvBase ← scalarMulVal kvIdxVal blockN32 "kv_base"
    let cutoff ← scalarSub seqLen32 kvBase "kv_cutoff"
    let needsMask ← scalarLt cutoff blockN32 "needs_tail_mask"
    emitIf needsMask.id do
      rightFillVal s s cutoff (some (-3.4028234663852886e38))
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let l ← computeLSE softmaxState
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 rowTileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lShared l
  storeVecGlobalRow l_ptr lShared coord

/-- GB10/Blackwell reduced MHA backward prep. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdPrep2Block
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (d_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rowTileSize : Nat := 16
  let coord ← blockCoord2D

  let dO ← allocRT .BFloat16 rowTileSize tileSize
  let o ← allocRT .BFloat16 rowTileSize tileSize
  let dOf ← allocRT .Float32 rowTileSize tileSize
  let of ← allocRT .Float32 rowTileSize tileSize
  let prod ← allocRT .Float32 rowTileSize tileSize
  let dVec ← allocRV .Float32 rowTileSize

  let dOShared ← allocST .BFloat16 rowTileSize tileSize
  let oShared ← allocST .BFloat16 rowTileSize tileSize
  let dShared ← allocSV .Float32 rowTileSize

  loadGlobal dOShared dO_ptr coord
  loadGlobal oShared o_ptr coord
  sync
  load dO dOShared
  load o oShared

  convert dOf dO
  convert of o
  mul prod dOf of
  rowSum dVec prod

  storeVec dShared dVec
  storeVecGlobalRow d_ptr dShared coord

/-- Query-owned dQ kernel without materialized dK/dV accumulators. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdDQDirect
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rows : Nat := 16
  let scale : Float := 0.125
  let invLScale : Float := -0.125
  let coord ← blockCoord2D
  let q ← allocRT .BFloat16 rows tileSize
  let dO ← allocRT .BFloat16 rows tileSize
  let dQ ← zeroRT .Float32 rows tileSize
  let lTk ← allocRV .Float32 rows
  let lse ← allocRV .Float32 rows
  let dVec ← allocRV .Float32 rows
  let rowShared ← allocST .BFloat16 rows tileSize
  let kvShared ← allocST .BFloat16 tileSize tileSize
  let colShared ← allocST .BFloat16 tileSize tileSize .Col
  let outShared ← allocST .Float32 rows tileSize
  let vecShared ← allocSV .Float32 rows
  loadGlobal rowShared q_ptr coord
  sync
  load q rowShared
  loadGlobal rowShared dO_ptr coord
  sync
  load dO rowShared
  loadVecGlobalRow vecShared l_ptr coord
  sync
  loadVec lTk vecShared
  loadVecGlobalRow vecShared d_ptr coord
  sync
  loadVec dVec vecShared
  scalarMulVec lse lTk invLScale
  for kvIdx in krange 0 2 do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize rows
    let pT ← allocRT .Float32 tileSize rows
    let dPT ← zeroRT .Float32 tileSize rows
    let dST ← allocRT .Float32 tileSize rows
    loadGlobal kvShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal colShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kvShared
    load v colShared
    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec
    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize rows
    scalarMul dSTScaled dST scale
    let dSTBf16 ← allocRT .BFloat16 tileSize rows
    convert dSTBf16 dSTScaled
    let dSRow ← allocRT .BFloat16 rows tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ
  store outShared dQ
  sync
  storeGlobal dQ_ptr outShared coord

/-- Key-owned direct dK/dV kernel. Each warp owns 16 key rows and reduces all
    query tiles locally, eliminating global partial tensors and reduction. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdDKDVDirect
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let rows : Nat := 16
  let dim : Nat := 64
  let scale : Float := 0.125
  let invLScale : Float := -0.125
  let coord ← blockCoord2D
  let k ← allocRT .BFloat16 rows dim
  let v ← allocRT .BFloat16 rows dim .Col
  let dK ← zeroRT .Float32 rows dim
  let dV ← zeroRT .Float32 rows dim
  let rowShared ← allocST .BFloat16 rows dim
  let colShared ← allocST .BFloat16 rows dim .Col
  let outShared ← allocST .Float32 rows dim
  let vecShared ← allocSV .Float32 rows
  loadGlobal rowShared k_ptr coord
  loadGlobal colShared v_ptr coord
  sync
  load k rowShared
  load v colShared
  for qIdx in krange 0 8 do
    let qCoord := coord.withRow qIdx.id
    let q ← allocRT .BFloat16 rows dim
    let dO ← allocRT .BFloat16 rows dim
    let lTk ← allocRV .Float32 rows
    let lse ← allocRV .Float32 rows
    let dVec ← allocRV .Float32 rows
    loadGlobal rowShared q_ptr qCoord
    sync
    load q rowShared
    loadGlobal rowShared dO_ptr qCoord
    sync
    load dO rowShared
    loadVecGlobalRow vecShared l_ptr qCoord
    sync
    loadVec lTk vecShared
    loadVecGlobalRow vecShared d_ptr qCoord
    sync
    loadVec dVec vecShared
    scalarMulVec lse lTk invLScale
    let sT ← zeroRT .Float32 rows rows
    let pT ← allocRT .Float32 rows rows
    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT
    let vRow ← allocRT .BFloat16 rows dim
    swapLayout vRow v
    let dPT ← zeroRT .Float32 rows rows
    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec
    let dST ← allocRT .Float32 rows rows
    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 rows rows
    scalarMul dSTScaled dST scale
    let pTBf16 ← allocRT .BFloat16 rows rows
    let dSTBf16 ← allocRT .BFloat16 rows rows
    convert pTBf16 pT
    convert dSTBf16 dSTScaled
    let qCol ← allocRT .BFloat16 rows dim .Col
    let dOCol ← allocRT .BFloat16 rows dim .Col
    swapLayout qCol q
    swapLayout dOCol dO
    mma dK dSTBf16 qCol dK
    mma dV pTBf16 dOCol dV
  store outShared dK
  sync
  storeGlobal dK_ptr outShared coord
  store outShared dV
  sync
  storeGlobal dV_ptr outShared coord

/-- Shape-specialized B/H/S768 backward prep with flattened row-vector output. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdPrepS768D64
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (d_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64)
    (q_heads : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let zero ← constIntVal 0 "bwd768_prep_zero"
  let queryBlock ← getBlockIdx 0 "bwd768_prep_query"
  let headIdx ← getBlockIdx 1 "bwd768_prep_head"
  let batchIdx ← getBlockIdx 2 "bwd768_prep_batch"
  let coord := makeRTileCoord batchIdx.id headIdx.id queryBlock.id zero.id
  let qHeads32 : KVal UInt32 ← castScalar .UInt32 q_heads "bwd768_prep_heads"
  let queryBlocks ← constIntVal 48 "bwd768_prep_blocks"
  let flatHeadBase ← scalarMulVal batchIdx qHeads32 "bwd768_prep_batch_head"
  let flatHead ← scalarAddVal flatHeadBase headIdx "bwd768_prep_flat_head"
  let flatBase ← scalarMulVal flatHead queryBlocks "bwd768_prep_flat_base"
  let flatRow ← scalarAddVal flatBase queryBlock "bwd768_prep_flat_row"
  let dCoord := coord.withRow flatRow.id
  let dO ← allocRT .BFloat16 16 64
  let o ← allocRT .BFloat16 16 64
  let dOf ← allocRT .Float32 16 64
  let of ← allocRT .Float32 16 64
  let prod ← allocRT .Float32 16 64
  let dVec ← allocRV .Float32 16
  let dOShared ← allocST .BFloat16 16 64
  let oShared ← allocST .BFloat16 16 64
  let dShared ← allocSV .Float32 16
  loadGlobal dOShared dO_ptr coord
  loadGlobal oShared o_ptr coord
  sync
  load dO dOShared
  load o oShared
  convert dOf dO
  convert of o
  mul prod dOf of
  rowSum dVec prod
  storeVec dShared dVec
  storeVecGlobalRow d_ptr dShared dCoord

/-- Four-warp query-owned B/H/S768 dQ specialization with shared TMA K/V staging. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdDQS768D64
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64)
    (q_heads : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileBytes : Nat := 32 * 64 * GpuFloat.bytes .BFloat16
  let zero ← constIntVal 0 "bwd768_dq_zero"
  let queryBlock ← getBlockIdx 0 "bwd768_dq_query"
  let headIdx ← getBlockIdx 1 "bwd768_dq_head"
  let batchIdx ← getBlockIdx 2 "bwd768_dq_batch"
  let coord := makeRTileCoord batchIdx.id headIdx.id queryBlock.id zero.id
  let qHeads32 : KVal UInt32 ← castScalar .UInt32 q_heads "bwd768_dq_heads"
  let queryBlocks ← constIntVal 48 "bwd768_dq_blocks"
  let four ← constIntVal 4 "bwd768_dq_four"
  let warpId ← getWarpId "bwd768_dq_warp"
  let warpZero ← scalarEq warpId zero "bwd768_dq_is_warp_zero"
  let flatHeadBase ← scalarMulVal batchIdx qHeads32 "bwd768_dq_batch_head"
  let flatHead ← scalarAddVal flatHeadBase headIdx "bwd768_dq_flat_head"
  let flatBase ← scalarMulVal flatHead queryBlocks "bwd768_dq_flat_base"
  let queryWarpBase ← scalarMulVal queryBlock four "bwd768_dq_query_warp_base"
  let queryWarpRow ← scalarAddVal queryWarpBase warpId "bwd768_dq_query_warp_row"
  let flatRow ← scalarAddVal flatBase queryWarpRow "bwd768_dq_flat_row"
  let vecCoord := coord.withRow flatRow.id
  let rowCoord := coord.withRow queryWarpBase.id
  let q ← allocRT .BFloat16 16 64
  let dO ← allocRT .BFloat16 16 64
  let o ← allocRT .BFloat16 16 64
  let dQ ← zeroRT .Float32 16 64
  let lTk ← allocRV .Float32 16
  let lse ← allocRV .Float32 16
  let dReduced ← allocRV .Float32 16
  let dVec ← allocRV .Float32 16
  let kShared0 ← allocST .BFloat16 32 64
  let kShared1 ← allocST .BFloat16 32 64
  let vShared0 ← allocST .BFloat16 32 64
  let vShared1 ← allocST .BFloat16 32 64
  warpgroupLoadGlobal q q_ptr rowCoord
  warpgroupLoadGlobal dO dO_ptr rowCoord
  warpgroupLoadGlobal o o_ptr rowCoord
  loadVecGlobalRowRV lTk l_ptr vecCoord
  let dOf ← allocRT .Float32 16 64
  let of ← allocRT .Float32 16 64
  convert dOf dO
  convert of o
  mul dOf dOf of
  rowSum dReduced dOf
  storeVecGlobalRowRV d_ptr dReduced vecCoord
  convertVecLayout dVec dReduced
  scalarMulVec lse lTk (-0.125)

  let kvSem0 ← allocSemaphore
  let kvSem1 ← allocSemaphore
  initSemaphore kvSem0 1
  initSemaphore kvSem1 1
  let issueKv (kShared vShared : ST GpuFloat.BFloat16 32 64)
      (sem : Semaphore) (kvIdx : KVal UInt32) : KernelM Unit := do
    let kvCoord := coord.withRow kvIdx.id
    expectBytes sem (2 * tileBytes)
    loadGlobalAsync kShared k_ptr kvCoord sem.id
    loadGlobalAsync vShared v_ptr kvCoord sem.id
  let firstKv ← constIntVal 0 "bwd768_dq_first_kv"
  let secondKv ← constIntVal 1 "bwd768_dq_second_kv"
  let one ← constIntVal 1 "bwd768_dq_one"
  let two ← constIntVal 2 "bwd768_dq_two"
  let kvBlockCount ← constIntVal 24 "bwd768_dq_kv_count"
  let zeroKv ← constIntVal 0 "bwd768_dq_kv_zero"
  issueKv kShared0 vShared0 kvSem0 firstKv
  issueKv kShared1 vShared1 kvSem1 secondKv

  for kvIdx in krange 0 24 do
    let kvIdxVal : KVal UInt32 := ⟨kvIdx.id, "bwd768_dq_kvidx"⟩
    let kvStage ← scalarMod kvIdxVal two "bwd768_dq_kv_stage"
    let kvHalf ← scalarDivVal kvIdxVal two "bwd768_dq_kv_half"
    let kvPhase ← scalarMod kvHalf two "bwd768_dq_kv_phase"
    let isStage0 ← scalarEq kvStage zeroKv "bwd768_dq_is_stage0"
    emitIf warpZero.id do
      ifThenElse isStage0
        (waitSemaphorePhaseVal kvSem0 kvPhase)
        (waitSemaphorePhaseVal kvSem1 kvPhase)
    blockSync
    let isAfterFirst ← scalarGt kvIdxVal zeroKv "bwd768_dq_after_first"
    let nextKvIdx ← scalarAddVal kvIdxVal one "bwd768_dq_next_kv"
    let hasNext ← scalarLt nextKvIdx kvBlockCount "bwd768_dq_has_next"
    let nextStage ← scalarMod nextKvIdx two "bwd768_dq_next_stage"
    let nextIsStage0 ← scalarEq nextStage zeroKv "bwd768_dq_next_is_stage0"
    emitIf isAfterFirst.id do
      emitIf hasNext.id do
        ifThenElse nextIsStage0
          (issueKv kShared0 vShared0 kvSem0 nextKvIdx)
          (issueKv kShared1 vShared1 kvSem1 nextKvIdx)
    for fragIdx in krange 0 2 do
      let fragVal : KVal UInt32 := ⟨fragIdx.id, "bwd768_dq_frag"⟩
      let k ← allocRT .BFloat16 16 64
      let v ← allocRT .BFloat16 16 64
      let sT ← zeroRT .Float32 16 16
      let pT ← allocRT .Float32 16 16
      let dPT ← zeroRT .Float32 16 16
      let dST ← allocRT .Float32 16 16
      ifThenElse isStage0
        (do loadSubtileVal k kShared0 fragVal zeroKv; loadSubtileVal v vShared0 fragVal zeroKv)
        (do loadSubtileVal k kShared1 fragVal zeroKv; loadSubtileVal v vShared1 fragVal zeroKv)
      mmaT sT k q sT
      scalarMul sT sT 0.125
      subRow sT sT lse
      exp pT sT
      mmaT dPT v dO dPT
      subRow dPT dPT dVec
      mul dST pT dPT
      scalarMul dST dST 0.125
      let dSTBf16 ← allocRT .BFloat16 16 16
      convert dSTBf16 dST
      let dSRow ← allocRT .BFloat16 16 16
      transpose dSRow dSTBf16
      let kCol ← allocRT .BFloat16 16 64 .Col
      swapLayout kCol k
      mma dQ dSRow kCol dQ
  let dQBf16 ← allocRT .BFloat16 16 64
  convert dQBf16 dQ
  warpgroupStoreGlobal dQ_ptr dQBf16 rowCoord

/-- Four-warp key-owned B/H/S768 dK/dV specialization with shared Q/dO staging. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdDKDVS768D64
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.BFloat16)
    (dV_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64)
    (q_heads : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let queryTileBytes : Nat := 48 * 64 * GpuFloat.bytes .BFloat16
  let zero ← constIntVal 0 "bwd768_dkdv_zero"
  let keyBlock ← getBlockIdx 0 "bwd768_dkdv_key"
  let headIdx ← getBlockIdx 1 "bwd768_dkdv_head"
  let batchIdx ← getBlockIdx 2 "bwd768_dkdv_batch"
  let coord := makeRTileCoord batchIdx.id headIdx.id keyBlock.id zero.id
  let qHeads32 : KVal UInt32 ← castScalar .UInt32 q_heads "bwd768_dkdv_heads"
  let queryBlocks ← constIntVal 48 "bwd768_dkdv_blocks"
  let four ← constIntVal 4 "bwd768_dkdv_four"
  let warpId ← getWarpId "bwd768_dkdv_warp"
  let warpZero ← scalarEq warpId zero "bwd768_dkdv_is_warp_zero"
  let flatHeadBase ← scalarMulVal batchIdx qHeads32 "bwd768_dkdv_batch_head"
  let flatHead ← scalarAddVal flatHeadBase headIdx "bwd768_dkdv_flat_head"
  let flatBase ← scalarMulVal flatHead queryBlocks "bwd768_dkdv_flat_base"
  let keyWarpBase ← scalarMulVal keyBlock four "bwd768_dkdv_key_warp_base"
  let keyCoord := coord.withRow keyWarpBase.id
  let k ← allocRT .BFloat16 16 64
  -- V is invariant across the 12 staged query groups and is only consumed by
  -- `mmaT V dO`. Load it in that operation's row layout directly instead of
  -- swapping the same register tile inside every loop iteration.
  let v ← allocRT .BFloat16 16 64
  let dK ← zeroRT .Float32 16 64
  let dV ← zeroRT .Float32 16 64
  let qShared0 ← allocST .BFloat16 48 64
  let qShared1 ← allocST .BFloat16 48 64
  let dOShared0 ← allocST .BFloat16 48 64
  let dOShared1 ← allocST .BFloat16 48 64
  let lShared0 ← allocSV .Float32 16
  let lShared1 ← allocSV .Float32 16
  let dShared0 ← allocSV .Float32 16
  let dShared1 ← allocSV .Float32 16
  let lShared2 ← allocSV .Float32 16
  let dShared2 ← allocSV .Float32 16
  warpgroupLoadGlobal k k_ptr keyCoord
  warpgroupLoadGlobal v v_ptr keyCoord

  let qSem0 ← allocSemaphore
  let qSem1 ← allocSemaphore
  initSemaphore qSem0 1
  initSemaphore qSem1 1
  let issueQ (qShared : ST GpuFloat.BFloat16 48 64)
      (dOShared : ST GpuFloat.BFloat16 48 64)
      (sem : Semaphore) (qIdx : KVal UInt32) : KernelM Unit := do
    let qCoord := coord.withRow qIdx.id
    expectBytes sem (2 * queryTileBytes)
    loadGlobalAsync qShared q_ptr qCoord sem.id
    loadGlobalAsync dOShared dO_ptr qCoord sem.id
  let firstQ ← constIntVal 0 "bwd768_dkdv_first_q"
  let secondQ ← constIntVal 1 "bwd768_dkdv_second_q"
  issueQ qShared0 dOShared0 qSem0 firstQ
  issueQ qShared1 dOShared1 qSem1 secondQ
  let one ← constIntVal 1 "bwd768_dkdv_one"
  let two ← constIntVal 2 "bwd768_dkdv_two"
  let three ← constIntVal 3 "bwd768_dkdv_three"
  let queryBlockCount ← constIntVal 16 "bwd768_dkdv_q_count"
  let zeroQ ← constIntVal 0 "bwd768_dkdv_q_zero"

  for qIdx in krange 0 16 do
    let qIdxVal : KVal UInt32 := ⟨qIdx.id, "bwd768_dkdv_qidx"⟩
    let qStage ← scalarMod qIdxVal two "bwd768_dkdv_q_stage"
    let qHalf ← scalarDivVal qIdxVal two "bwd768_dkdv_q_half"
    let qPhase ← scalarMod qHalf two "bwd768_dkdv_q_phase"
    let isStage0 ← scalarEq qStage zeroQ "bwd768_dkdv_is_stage0"
    emitIf warpZero.id do
      ifThenElse isStage0
        (waitSemaphorePhaseVal qSem0 qPhase)
        (waitSemaphorePhaseVal qSem1 qPhase)
    blockSync
    -- Once every warp has retired the previous iteration, refill that retired
    -- stage with the tile after the current one. The other stage remains stable
    -- for the loads below, so TMA overlaps this iteration's MMA safely.
    let isAfterFirst ← scalarGt qIdxVal zeroQ "bwd768_dkdv_after_first"
    let nextQIdx ← scalarAddVal qIdxVal one "bwd768_dkdv_next_q"
    let hasNext ← scalarLt nextQIdx queryBlockCount "bwd768_dkdv_has_next"
    let nextStage ← scalarMod nextQIdx two "bwd768_dkdv_next_stage"
    let nextIsStage0 ← scalarEq nextStage zeroQ "bwd768_dkdv_next_is_stage0"
    emitIf isAfterFirst.id do
      emitIf hasNext.id do
        ifThenElse nextIsStage0
          (issueQ qShared0 dOShared0 qSem0 nextQIdx)
          (issueQ qShared1 dOShared1 qSem1 nextQIdx)
    let groupRow ← scalarMulVal qIdxVal three "bwd768_dkdv_group_row"
    let flatRow0 ← scalarAddVal flatBase groupRow "bwd768_dkdv_flat_row0"
    let flatRow1 ← scalarAddVal flatRow0 one "bwd768_dkdv_flat_row1"
    let flatRow2 ← scalarAddVal flatRow1 one "bwd768_dkdv_flat_row2"
    let vecCoord0 := coord.withRow flatRow0.id
    let vecCoord1 := coord.withRow flatRow1.id
    let vecCoord2 := coord.withRow flatRow2.id
    emitIf warpZero.id do
      loadVecGlobalRow lShared0 l_ptr vecCoord0
      loadVecGlobalRow lShared1 l_ptr vecCoord1
      loadVecGlobalRow lShared2 l_ptr vecCoord2
      loadVecGlobalRow dShared0 d_ptr vecCoord0
      loadVecGlobalRow dShared1 d_ptr vecCoord1
      loadVecGlobalRow dShared2 d_ptr vecCoord2
    -- Publish three 16-row L/D fragments once per 48-row staged query group.
    blockSync
    for fragIdx in krange 0 3 do
      let fragVal : KVal UInt32 := ⟨fragIdx.id, "bwd768_dkdv_frag"⟩
      let q ← allocRT .BFloat16 16 64
      let dO ← allocRT .BFloat16 16 64
      let lTk ← allocRV .Float32 16
      let lse ← allocRV .Float32 16
      let dVec ← allocRV .Float32 16
      ifThenElse isStage0
        (do loadSubtileVal q qShared0 fragVal zeroQ; loadSubtileVal dO dOShared0 fragVal zeroQ)
        (do loadSubtileVal q qShared1 fragVal zeroQ; loadSubtileVal dO dOShared1 fragVal zeroQ)
      let isFrag0 ← scalarEq fragVal zeroQ "bwd768_dkdv_is_frag0"
      let isFrag1 ← scalarEq fragVal one "bwd768_dkdv_is_frag1"
      ifThenElse isFrag0
        (do loadVec lTk lShared0; loadVec dVec dShared0)
        (ifThenElse isFrag1
          (do loadVec lTk lShared1; loadVec dVec dShared1)
          (do loadVec lTk lShared2; loadVec dVec dShared2))
      scalarMulVec lse lTk (-0.125)
      let sT ← zeroRT .Float32 16 16
      let pT ← allocRT .Float32 16 16
      mmaT sT k q sT
      scalarMul sT sT 0.125
      subRow sT sT lse
      exp pT sT
      let dPT ← zeroRT .Float32 16 16
      mmaT dPT v dO dPT
      subRow dPT dPT dVec
      let dST ← allocRT .Float32 16 16
      mul dST pT dPT
      let dSTScaled ← allocRT .Float32 16 16
      scalarMul dSTScaled dST 0.125
      let pTBf16 ← allocRT .BFloat16 16 16
      let dSTBf16 ← allocRT .BFloat16 16 16
      convert pTBf16 pT
      convert dSTBf16 dSTScaled
      let qCol ← allocRT .BFloat16 16 64 .Col
      let dOCol ← allocRT .BFloat16 16 64 .Col
      swapLayout qCol q
      swapLayout dOCol dO
      mma dK dSTBf16 qCol dK
      mma dV pTBf16 dOCol dV
  let dKBf16 ← allocRT .BFloat16 16 64
  convert dKBf16 dK
  warpgroupStoreGlobal dK_ptr dKBf16 keyCoord
  let dVBf16 ← allocRT .BFloat16 16 64
  convert dVBf16 dV
  warpgroupStoreGlobal dV_ptr dVBf16 keyCoord

/-- GB10/Blackwell reduced MHA backward with partial dK/dV outputs. -/
@[gpu_kernel .SM90]
def tkMhaGb10Bwd2BlockPartials
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (_dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rowTileSize : Nat := 16
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 rowTileSize tileSize
  let dO ← allocRT .BFloat16 rowTileSize tileSize
  let dQ ← zeroRT .Float32 rowTileSize tileSize
  let lTk ← allocRV .Float32 rowTileSize
  let lse ← allocRV .Float32 rowTileSize
  let dVec ← allocRV .Float32 rowTileSize

  let qRowShared ← allocST .BFloat16 rowTileSize tileSize
  let kvRowShared ← allocST .BFloat16 tileSize tileSize
  let colShared ← allocST .BFloat16 tileSize tileSize .Col
  let outShared ← allocST .Float32 tileSize tileSize
  let dQShared ← allocST .Float32 rowTileSize tileSize
  let vecShared ← allocSV .Float32 rowTileSize

  loadGlobal qRowShared q_ptr coord
  sync
  load q qRowShared
  loadGlobal qRowShared dO_ptr coord
  sync
  load dO qRowShared

  loadVecGlobalRow vecShared l_ptr coord
  sync
  loadVec lTk vecShared
  loadVecGlobalRow vecShared d_ptr coord
  sync
  loadVec dVec vecShared
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize rowTileSize
    let pT ← allocRT .Float32 tileSize rowTileSize
    let dPT ← zeroRT .Float32 tileSize rowTileSize
    let dST ← allocRT .Float32 tileSize rowTileSize
    let dKPart ← zeroRT .Float32 tileSize tileSize

    loadGlobal kvRowShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal colShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kvRowShared
    load v colShared

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize rowTileSize
    scalarMul dSTScaled dST scale

    let dSTBf16 ← allocRT .BFloat16 tileSize rowTileSize
    convert dSTBf16 dSTScaled

    let qCol ← allocRT .BFloat16 rowTileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart

    let dSRow ← allocRT .BFloat16 rowTileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := (coord.withRow kvIdx.id).withCol coord.r
    store outShared dKPart
    sync
    storeGlobal dK_part_ptr outShared dkvCoord
    sync

  store dQShared dQ
  sync
  storeGlobal dQ_ptr dQShared coord

/-- GB10 dV partials split from dQ/dK to avoid the 255-register spill cliff. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdDVPartials
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let tileSize : Nat := 64
  let rowTileSize : Nat := 16
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let invLScale : Float := -0.125
  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 rowTileSize tileSize
  let dO ← allocRT .BFloat16 rowTileSize tileSize
  let lTk ← allocRV .Float32 rowTileSize
  let lse ← allocRV .Float32 rowTileSize
  let qShared ← allocST .BFloat16 rowTileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let outShared ← allocST .Float32 tileSize tileSize
  let vecShared ← allocSV .Float32 rowTileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared
  loadGlobal qShared dO_ptr coord
  sync
  load dO qShared
  loadVecGlobalRow vecShared l_ptr coord
  sync
  loadVec lTk vecShared
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let sT ← zeroRT .Float32 tileSize rowTileSize
    let pT ← allocRT .Float32 tileSize rowTileSize
    let pTBf16 ← allocRT .BFloat16 tileSize rowTileSize
    let dOCol ← allocRT .BFloat16 rowTileSize tileSize .Col
    let dVPart ← zeroRT .Float32 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT
    convert pTBf16 pT
    swapLayout dOCol dO
    mma dVPart pTBf16 dOCol dVPart

    let dkvCoord := (coord.withRow kvIdx.id).withCol coord.r
    store outShared dVPart
    sync
    storeGlobal dV_part_ptr outShared dkvCoord
    sync

/-- Reduce the eight query-tile dK/dV partials into final 16x64 row tiles.
    Each CTA is one warp and owns disjoint output rows, so no atomics are
    required. -/
@[gpu_kernel .SM90]
def tkMhaGb10BwdReducePartials
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  let rows : Nat := 16
  let cols : Nat := 64
  let queryBlocks : Nat := 8
  let coord ← blockCoord2D

  let dK ← zeroRT .Float32 rows cols
  let dV ← zeroRT .Float32 rows cols
  let dKPart ← allocRT .Float32 rows cols
  let dVPart ← allocRT .Float32 rows cols

  for qIdx in List.range queryBlocks do
    let qIdxVal ← constIntVal qIdx s!"bwd_reduce_q_{qIdx}"
    let partialCoord := coord.withCol qIdxVal.id
    loadRegisterGlobal dKPart dK_part_ptr partialCoord
    loadRegisterGlobal dVPart dV_part_ptr partialCoord
    add dK dK dKPart
    add dV dV dVPart

  storeRegisterGlobal dK_ptr dK coord
  storeRegisterGlobal dV_ptr dV coord

end Tyr.GPU.Kernels
