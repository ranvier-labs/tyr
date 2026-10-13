/- ThunderKittens-style FlashAttention forward kernels for 128x64 (2 KV blocks). -/
import Tyr.GPU.Codegen.Macros

import Tyr.GPU.Kernels.Prelude

/-! # H100 multi-head attention

`mha_h100`-style FlashAttention for H100 using async tile loads and an online
softmax in base 2, with forward (optional LSE) and backward kernels.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def asyncLoadGlobalTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : RTileCoord)
    (sem : Semaphore) : KernelM Unit := do
  initSemaphore sem 0 1
  blockSync
  expectBytes sem (rows * cols * dtype.bytes)
  loadGlobalAsync dst src coord sem.id
  waitSemaphore sem

private def asyncLoadGlobalPair {dtype : GpuFloat} {rows cols : Nat}
    {layoutA layoutB : TileLayout}
    (dstA : ST dtype rows cols layoutA)
    (srcA : GPtr dtype)
    (coordA : RTileCoord)
    (dstB : ST dtype rows cols layoutB)
    (srcB : GPtr dtype)
    (coordB : RTileCoord)
    (sem : Semaphore) : KernelM Unit := do
  initSemaphore sem 0 1
  blockSync
  expectBytes sem (2 * rows * cols * dtype.bytes)
  loadGlobalAsync dstA srcA coordA sem.id
  loadGlobalAsync dstB srcB coordB sem.id
  waitSemaphore sem

private def vectorCoordFromTileRow (coord : RTileCoord) : RTileCoord :=
  { b := coord.b, d := coord.d, r := coord.c, c := coord.r }

private def onlineSoftmaxLog2TkH100 {cols : Nat}
    (scores : RT GpuFloat.Float32 16 cols .Row)
    (rowMax rowSum lastScaled maxScaled : RV GpuFloat.Float32 16)
    (scoreScaleLog2e : Float)
    : KernelM Unit := do
  copyVec lastScaled rowMax
  scalarMulVec lastScaled lastScaled scoreScaleLog2e
  mmaAsyncWait
  rowMaxAccum rowMax scores rowMax
  scalarMul scores scores scoreScaleLog2e
  scalarMulVec maxScaled rowMax scoreScaleLog2e
  subCol scores scores maxScaled
  exp2 scores scores
  subVec lastScaled lastScaled maxScaled
  exp2Vec lastScaled lastScaled
  mulVec rowSum rowSum lastScaled
  rowSumAccum rowSum scores rowSum
  scalarAdd scores scores 0.0

private def computeLSEScaledTkH100
    (rowSum maxScaled : RV GpuFloat.Float32 16)
    : KernelM (RV GpuFloat.Float32 16) := do
  scalarMulVec maxScaled maxScaled 0.69314718056
  logVec rowSum rowSum
  addVec rowSum rowSum maxScaled
  pure rowSum

/-- FlashAttention forward for two KV blocks (seq=128, head_dim=64).
    This kernel is currently non-causal because dynamic block-offset masking is
    not yet represented in the IR. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd2Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125 -- 1 / sqrt(64)
  let scoreScaleLog2e : Float := 0.18033688011125
  setLaunchBounds 128 1

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- FlashAttention forward (2 KV blocks) with LSE output.
    `lse_ptr` stores one 64-element vector per query tile row. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd2BlockLse
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125 -- 1 / sqrt(64)
  let scoreScaleLog2e : Float := 0.18033688011125
  setLaunchBounds 128 1

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lseShared ← allocSTColVec .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
  finalizeSoftmax o softmaxState
  let lse ← computeLSEScaled softmaxState scale

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeColVec lseShared lse
  storeColVecGlobalAsync lse_ptr lseShared (vectorCoordFromTileRow coord)
  tmaStoreAsyncWait

/-- `mha_h100`-style forward:
    - output `o_ptr` (bf16)
    - output `l_ptr` where `l = -8 * lse` for `head_dim=64` (ThunderKittens convention) -/
@[gpu_kernel .SM90]
def tkMhaH100Fwd2Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let scoreScaleLog2e : Float := 0.18033688011125
  let lScale : Float := -8.0
  setLaunchBounds 128 1

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let o ← zeroRT .Float32 tileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lShared ← allocSTColVec .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
  finalizeSoftmax o softmaxState
  let l ← computeLSEScaled softmaxState scale
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeColVec lShared l
  storeColVecGlobalAsync l_ptr lShared (vectorCoordFromTileRow coord)
  tmaStoreAsyncWait

/-- `mha_h100` backward prep:
    `d_ptr[row] = sum_j(dO[row,j] * O[row,j])`. -/
@[gpu_kernel .SM90]
def tkMhaH100BwdPrep2Block
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (d_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let coord ← blockCoord2D

  let dO ← allocRT .BFloat16 tileSize tileSize
  let o ← allocRT .BFloat16 tileSize tileSize
  let dOf ← allocRT .Float32 tileSize tileSize
  let of ← allocRT .Float32 tileSize tileSize
  let prod ← allocRT .Float32 tileSize tileSize
  let dVec ← allocRV .Float32 tileSize

  let dOShared ← allocST .BFloat16 tileSize tileSize
  let oShared ← allocST .BFloat16 tileSize tileSize
  let dShared ← allocSTColVec .Float32 tileSize tileSize
  let doSem ← allocSemaphore

  asyncLoadGlobalPair dOShared dO_ptr coord oShared o_ptr coord doSem
  load dO dOShared
  load o oShared

  convert dOf dO
  convert of o
  mul prod dOf of
  rowSum dVec prod

  storeColVec dShared dVec
  storeColVecGlobalAsync d_ptr dShared (vectorCoordFromTileRow coord)
  tmaStoreAsyncWait

/-- `mha_h100` backward (2 blocks, non-causal) with in-kernel `dK`/`dV`
    accumulation. The launcher name keeps the older `...Partials` suffix for
    ABI stability, but `dK_part_ptr` and `dV_part_ptr` now point at final
    zero-initialized `[1, 1, seq, 64]` buffers. Each query tile contributes to
    the KV tile with TMA store-add, matching ThunderKittens' `kv_store`
    accumulation contract more closely than the earlier external partial stack. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockPartials
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let dO ← allocRT .BFloat16 tileSize tileSize
  let dQ ← zeroRT .Float32 tileSize tileSize
  let lTk ← allocRV .Float32 tileSize
  let lse ← allocRV .Float32 tileSize
  let dVec ← allocRV .Float32 tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  -- Reuse one FP32 staging tile; wait before overwriting it with the next TMA store.
  let dKVShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize
    let dKPart ← zeroRT .Float32 tileSize tileSize
    let dVPart ← zeroRT .Float32 tileSize tileSize

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    -- Reference-style orientation: keep score/probability blocks in KxQ order.
    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let pTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert pTBf16 pT
    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dOCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dVPart pTBf16 dOCol dVPart

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := coord.withRow kvIdx.id
    store dKVShared dKPart
    groupSync 4 4
    storeGlobalAdd dK_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4
    store dKVShared dVPart
    groupSync 4 4
    storeGlobalAdd dV_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4

  store dKVShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dKVShared coord

/-- Q-centric `dQ` pass for 2-block H100 MHA backward.
    This keeps the stable direct-store `dQ` accumulation while the K/V gradients
    are produced by the KV-centric sweep below. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockDq
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let dO ← allocRT .BFloat16 tileSize tileSize
  let dQ ← zeroRT .Float32 tileSize tileSize
  let lTk ← allocRV .Float32 tileSize
  let lse ← allocRV .Float32 tileSize
  let dVec ← allocRV .Float32 tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let dQShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

  store dQShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dQShared coord

/-- Reuse phase bit for double-buffered semaphores, matching TK's
    `((qo_idx - q_start) / 2) % 2` for the non-causal `q_start = 0` path. -/
private def stagePhase (iter : Nat) : Nat :=
  (iter / 2) % 2

set_option maxRecDepth 8192 in
private def tkMhaH100BwdKvSweepPc
    (numQBlocks : Nat)
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    : KernelM Unit := do
  setLaunchBounds 384 1
  let tileSize : Nat := 64
  let scale : Float := 0.125
  let scoreScaleLog2e : Float := 0.18033688011125

  let zero ← constIntVal 0 "mha_bwd_zero"
  let one ← constIntVal 1 "mha_bwd_one"
  let two ← constIntVal 2 "mha_bwd_two"
  let three ← constIntVal 3 "mha_bwd_three"
  let four ← constIntVal 4 "mha_bwd_four"
  let numQBlocksVal ← constIntVal numQBlocks "mha_bwd_num_qblocks"

  let kvGroupIdx ← getBlockIdx 0 "mha_bwd_kv_group"
  let kvBaseRow ← scalarMulVal kvGroupIdx two "mha_bwd_kv_base_row"
  let kvRow1 ← scalarAddVal kvBaseRow one "mha_bwd_kv_row1"

  let mkCoord (row : KVal UInt32) : RTileCoord :=
    { b := zero.id, d := zero.id, r := row.id, c := zero.id }
  let mkVecCoord (col : KVal UInt32) : RTileCoord :=
    { b := zero.id, d := zero.id, r := zero.id, c := col.id }

  let kvCoord0 := mkCoord kvBaseRow
  let kvCoord1 := mkCoord kvRow1
  let qStartCoord := mkCoord zero
  let qStartVecCoord := mkVecCoord zero

  let kShared0 ← allocST .BFloat16 tileSize tileSize
  let kShared1 ← allocST .BFloat16 tileSize tileSize
  let vShared0 ← allocST .BFloat16 tileSize tileSize .Col
  let vShared1 ← allocST .BFloat16 tileSize tileSize .Col
  let qStage0 ← allocST .BFloat16 tileSize tileSize
  let qStage1 ← allocST .BFloat16 tileSize tileSize
  let doStage0 ← allocST .BFloat16 tileSize tileSize
  let doStage1 ← allocST .BFloat16 tileSize tileSize
  let dsShared0 ← allocST .BFloat16 tileSize tileSize
  let dsShared1 ← allocST .BFloat16 tileSize tileSize
  let qgShared ← allocST .Float32 tileSize tileSize
  let dKShared0 ← aliasST kShared0 .Float32 tileSize tileSize .Row
    "reuse kShared0/kShared1 storage for dK tile 0"
  let dKShared1 ← aliasST vShared0 .Float32 tileSize tileSize .Row
    "reuse vShared0/vShared1 storage for dK tile 1"
  let dVShared0 ← aliasST qStage0 .Float32 tileSize tileSize .Row
    "reuse qStage0/qStage1 storage for dV tile 0"
  let dVShared1 ← aliasST doStage0 .Float32 tileSize tileSize .Row
    "reuse doStage0/doStage1 storage for dV tile 1"
  let lStage0 ← allocSTRowVec .Float32 tileSize tileSize
  let lStage1 ← allocSTRowVec .Float32 tileSize tileSize
  let dStage0 ← allocSTRowVec .Float32 tileSize tileSize
  let dStage1 ← allocSTRowVec .Float32 tileSize tileSize

  let kvSem ← allocSemaphore
  let qSem0 ← allocSemaphore
  let qSem1 ← allocSemaphore
  let doSem0 ← allocSemaphore
  let doSem1 ← allocSemaphore
  let vecSem0 ← allocSemaphore
  let vecSem1 ← allocSemaphore
  let computeDone0 ← allocSemaphore
  let computeDone1 ← allocSemaphore
  let qgReady ← allocSemaphore

  initSemaphore kvSem 0 1
  initSemaphore qSem0 0 1
  initSemaphore qSem1 0 1
  initSemaphore doSem0 0 1
  initSemaphore doSem1 0 1
  initSemaphore vecSem0 0 1
  initSemaphore vecSem1 0 1
  initSemaphore computeDone0 1
  initSemaphore computeDone1 1
  initSemaphore qgReady 1

  let warpGroupIdx ← getWarpGroupIdx
  let warpId ← getWarpId "mha_bwd_warp_id"
  let laneId ← getLaneId "mha_bwd_lane_id"
  let isWarpGroup0 ← scalarEq warpGroupIdx zero "mha_bwd_is_wg0"
  let isWarpGroup2 ← scalarEq warpGroupIdx two "mha_bwd_is_wg2"
  let localWarp ← scalarMod warpId four "mha_bwd_local_warp"
  let isLocalWarp0 ← scalarEq localWarp zero "mha_bwd_is_local_warp0"
  let isLocalWarp1 ← scalarEq localWarp one "mha_bwd_is_local_warp1"
  let isLane0 ← scalarEq laneId zero "mha_bwd_is_lane0"

  ifThen isWarpGroup2 do
    ifThen isLocalWarp0 do
      expectBytesWarp kvSem (4 * tileSize * tileSize * GpuFloat.bytes .BFloat16)
      loadGlobalAsyncWarp kShared0 k_ptr kvCoord0 kvSem.id
      loadGlobalAsyncWarp vShared0 v_ptr kvCoord0 kvSem.id
      loadGlobalAsyncWarp kShared1 k_ptr kvCoord1 kvSem.id
      loadGlobalAsyncWarp vShared1 v_ptr kvCoord1 kvSem.id

      expectBytesWarp qSem0 (tileSize * tileSize * GpuFloat.bytes .BFloat16)
      loadGlobalAsyncWarp qStage0 q_ptr qStartCoord qSem0.id
      expectBytesWarp doSem0 (tileSize * tileSize * GpuFloat.bytes .BFloat16)
      loadGlobalAsyncWarp doStage0 dO_ptr qStartCoord doSem0.id
      expectBytesWarp vecSem0 (2 * tileSize * GpuFloat.bytes .Float32)
      loadRowVecGlobalAsyncWarp lStage0 l_ptr qStartVecCoord vecSem0.id
      loadRowVecGlobalAsyncWarp dStage0 d_ptr qStartVecCoord vecSem0.id

  blockSync

  ifThenElse isWarpGroup2
    (do
      warpgroupDecreaseRegisters 24
      ifThen isLocalWarp0 do
        for qIdx in krange 0 numQBlocks do
          let qIdxVal : KVal UInt32 := ⟨qIdx.id, "mha_bwd_qidx_prefetch"⟩
          let ticVal ← scalarMod qIdxVal two "mha_bwd_tic_prefetch"
          let qHalf ← scalarDivVal qIdxVal two "mha_bwd_qidx_half_prefetch"
          let qPhase ← scalarMod qHalf two "mha_bwd_qphase_prefetch"
          let nextQIdxVal ← scalarAddVal qIdxVal one "mha_bwd_qidx_next"
          let tocVal ← scalarMod nextQIdxVal two "mha_bwd_toc_prefetch"
          let hasNext ← scalarLt nextQIdxVal numQBlocksVal "mha_bwd_has_next"
          let isTic0 ← scalarEq ticVal zero "mha_bwd_is_tic0_prefetch"
          let isToc0 ← scalarEq tocVal zero "mha_bwd_is_toc0_prefetch"
          ifThen hasNext do
            let nextQCoord := mkCoord nextQIdxVal
            let nextQVecCoord := mkVecCoord nextQIdxVal
            ifThenElse isToc0
              (do
                expectBytesWarp qSem0 (tileSize * tileSize * GpuFloat.bytes .BFloat16)
                loadGlobalAsyncWarp qStage0 q_ptr nextQCoord qSem0.id
                expectBytesWarp doSem0 (tileSize * tileSize * GpuFloat.bytes .BFloat16)
                loadGlobalAsyncWarp doStage0 dO_ptr nextQCoord doSem0.id
                expectBytesWarp vecSem0 (2 * tileSize * GpuFloat.bytes .Float32)
                loadRowVecGlobalAsyncWarp lStage0 l_ptr nextQVecCoord vecSem0.id
                loadRowVecGlobalAsyncWarp dStage0 d_ptr nextQVecCoord vecSem0.id)
              (do
                expectBytesWarp qSem1 (tileSize * tileSize * GpuFloat.bytes .BFloat16)
                loadGlobalAsyncWarp qStage1 q_ptr nextQCoord qSem1.id
                expectBytesWarp doSem1 (tileSize * tileSize * GpuFloat.bytes .BFloat16)
                loadGlobalAsyncWarp doStage1 dO_ptr nextQCoord doSem1.id
                expectBytesWarp vecSem1 (2 * tileSize * GpuFloat.bytes .Float32)
                loadRowVecGlobalAsyncWarp lStage1 l_ptr nextQVecCoord vecSem1.id
                loadRowVecGlobalAsyncWarp dStage1 d_ptr nextQVecCoord vecSem1.id)
          ifThenElse isTic0
            (waitSemaphorePhaseVal computeDone0 qPhase)
            (waitSemaphorePhaseVal computeDone1 qPhase)

      ifThen isLocalWarp1 do
        for qIdx in krange 0 numQBlocks do
          let qIdxVal : KVal UInt32 := ⟨qIdx.id, "mha_bwd_qidx_store"⟩
          let ticVal ← scalarMod qIdxVal two "mha_bwd_tic_store"
          let qHalf ← scalarDivVal qIdxVal two "mha_bwd_qidx_half_store"
          let qPhase ← scalarMod qHalf two "mha_bwd_qphase_store"
          let isTic0 ← scalarEq ticVal zero "mha_bwd_is_tic0_store"
          let qCoord := mkCoord qIdxVal
          ifThenElse isTic0
            (waitSemaphorePhaseVal computeDone0 qPhase)
            (waitSemaphorePhaseVal computeDone1 qPhase)
          storeGlobalAddWarp dQ_ptr qgShared qCoord
          tmaStoreAsyncWait
          ifThen isLane0 do
            arriveSemaphoreWarp qgReady 1)
    (do
      ifThenElse isWarpGroup0
        (do
          warpgroupIncreaseRegisters 256
          let dKAccum ← zeroRT .Float32 16 tileSize
          let dVAccum ← zeroRT .Float32 16 tileSize
          waitSemaphorePhase kvSem 0

          for qIdx in krange 0 numQBlocks do
            let qIdxVal : KVal UInt32 := ⟨qIdx.id, "mha_bwd_qidx_wg0"⟩
            let stageVal ← scalarMod qIdxVal two "mha_bwd_stage_wg0"
            let qHalf ← scalarDivVal qIdxVal two "mha_bwd_qidx_half_wg0"
            let qPhase ← scalarMod qHalf two "mha_bwd_qphase_wg0"
            let nextQIdxVal ← scalarAddVal qIdxVal one "mha_bwd_qidx_next_wg0"
            let tocVal ← scalarMod nextQIdxVal two "mha_bwd_toc_wg0"
            let isStage0 ← scalarEq stageVal zero "mha_bwd_is_stage0_wg0"
            let isAfterFirst ← scalarGt qIdxVal zero "mha_bwd_after_first_wg0"
            let sT ← allocRT .Float32 16 tileSize
            let pT ← allocRT .Float32 16 tileSize
            let dPT ← allocRT .Float32 16 tileSize
            let dST ← allocRT .Float32 16 tileSize
            let pTBf16 ← allocRT .BFloat16 16 tileSize
            let dSTBf16 ← allocRT .BFloat16 16 tileSize
            let qgReg ← allocRT .Float32 16 tileSize
            ifThenElse isStage0
              (do
                waitSemaphorePhaseVal vecSem0 qPhase
                streamRowVecTile16x64 sT lStage0
                waitSemaphorePhaseVal qSem0 qPhase
                warpgroupMmaSharedT64x16 sT kShared0 qStage0
                mmaCommitGroup
                waitSemaphorePhaseVal doSem0 qPhase
                warpgroupMmSharedT64x16 dPT vShared0 doStage0
                mmaCommitGroup
                mmaAsyncWait)
              (do
                waitSemaphorePhaseVal vecSem1 qPhase
                streamRowVecTile16x64 sT lStage1
                waitSemaphorePhaseVal qSem1 qPhase
                warpgroupMmaSharedT64x16 sT kShared0 qStage1
                mmaCommitGroup
                waitSemaphorePhaseVal doSem1 qPhase
                warpgroupMmSharedT64x16 dPT vShared0 doStage1
                mmaCommitGroup
                mmaAsyncWait)

            scalarMul sT sT scoreScaleLog2e
            exp2 pT sT

            ifThenElse isStage0
              (streamSubRowVecTile16x64 dPT dStage0)
              (streamSubRowVecTile16x64 dPT dStage1)
            mul dST pT dPT
            scalarMul dST dST scale

            convert pTBf16 pT
            convert dSTBf16 dST

            ifThenElse isStage0
              (do
                warpgroupMma dVAccum pTBf16 doStage0
                warpgroupMma dKAccum dSTBf16 qStage0
                mmaAsyncWait
                warpgroupStore dsShared0 dSTBf16)
              (do
                warpgroupMma dVAccum pTBf16 doStage1
                warpgroupMma dKAccum dSTBf16 qStage1
                mmaAsyncWait
                warpgroupStore dsShared0 dSTBf16)
            groupSync 8 10
            warpgroupMmSharedAtB64x16 qgReg dsShared0 kShared0
            warpgroupMmaSharedAtB64x16 qgReg dsShared1 kShared1
            mmaCommitGroup
            waitSemaphorePhaseVal qgReady tocVal
            ifThen isAfterFirst do
              tmaStoreAsyncWait
            mmaAsyncWait
            warpgroupStore qgShared qgReg
            groupSync 4 4
            ifThen isLocalWarp0 do
              ifThen isLane0 do
                ifThenElse isStage0
                  (arriveSemaphoreWarp computeDone0 1)
                  (arriveSemaphoreWarp computeDone1 1)

          groupSync 8 10
          warpgroupStore dKShared0 dKAccum
          groupSync 4 4
          ifThen isLocalWarp0 do
            storeGlobalAddWarp dK_ptr dKShared0 kvCoord0
            tmaStoreCommitGroup
          waitSemaphorePhase qgReady (if numQBlocks % 2 == 0 then 1 else 0)
          warpgroupStore dVShared0 dVAccum
          groupSync 4 4
          ifThen isLocalWarp0 do
            storeGlobalAddWarp dV_ptr dVShared0 kvCoord0
            tmaStoreCommitGroup
            tmaStoreAsyncWait)
        (do
          warpgroupIncreaseRegisters 224
          let dKAccum ← zeroRT .Float32 16 tileSize
          let dVAccum ← zeroRT .Float32 16 tileSize
          waitSemaphorePhase kvSem 0

          for qIdx in krange 0 numQBlocks do
            let qIdxVal : KVal UInt32 := ⟨qIdx.id, "mha_bwd_qidx_wg1"⟩
            let stageVal ← scalarMod qIdxVal two "mha_bwd_stage_wg1"
            let qHalf ← scalarDivVal qIdxVal two "mha_bwd_qidx_half_wg1"
            let qPhase ← scalarMod qHalf two "mha_bwd_qphase_wg1"
            let isStage0 ← scalarEq stageVal zero "mha_bwd_is_stage0_wg1"
            let sT ← allocRT .Float32 16 tileSize
            let pT ← allocRT .Float32 16 tileSize
            let dPT ← allocRT .Float32 16 tileSize
            let dST ← allocRT .Float32 16 tileSize
            let pTBf16 ← allocRT .BFloat16 16 tileSize
            let dSTBf16 ← allocRT .BFloat16 16 tileSize

            ifThenElse isStage0
              (do
                waitSemaphorePhaseVal vecSem0 qPhase
                streamRowVecTile16x64 sT lStage0
                waitSemaphorePhaseVal qSem0 qPhase
                warpgroupMmaSharedT64x16 sT kShared1 qStage0
                mmaCommitGroup
                waitSemaphorePhaseVal doSem0 qPhase
                warpgroupMmSharedT64x16 dPT vShared1 doStage0
                mmaCommitGroup
                mmaAsyncWait)
              (do
                waitSemaphorePhaseVal vecSem1 qPhase
                streamRowVecTile16x64 sT lStage1
                waitSemaphorePhaseVal qSem1 qPhase
                warpgroupMmaSharedT64x16 sT kShared1 qStage1
                mmaCommitGroup
                waitSemaphorePhaseVal doSem1 qPhase
                warpgroupMmSharedT64x16 dPT vShared1 doStage1
                mmaCommitGroup
                mmaAsyncWait)

            scalarMul sT sT scoreScaleLog2e
            exp2 pT sT

            ifThenElse isStage0
              (streamSubRowVecTile16x64 dPT dStage0)
              (streamSubRowVecTile16x64 dPT dStage1)
            mul dST pT dPT
            scalarMul dST dST scale

            convert pTBf16 pT
            convert dSTBf16 dST

            ifThenElse isStage0
              (do
                warpgroupMma dVAccum pTBf16 doStage0
                warpgroupMma dKAccum dSTBf16 qStage0
                mmaAsyncWait
                warpgroupStore dsShared1 dSTBf16)
              (do
                warpgroupMma dVAccum pTBf16 doStage1
                warpgroupMma dKAccum dSTBf16 qStage1
                mmaAsyncWait
                warpgroupStore dsShared1 dSTBf16)
            groupSync 8 10

          groupSync 8 10
          warpgroupStore dKShared1 dKAccum
          groupSync 4 5
          ifThen isLocalWarp0 do
            storeGlobalAddWarp dK_ptr dKShared1 kvCoord1
            tmaStoreCommitGroup
          waitSemaphorePhase qgReady (if numQBlocks % 2 == 0 then 1 else 0)
          warpgroupStore dVShared1 dVAccum
          groupSync 4 5
          ifThen isLocalWarp0 do
            storeGlobalAddWarp dV_ptr dVShared1 kvCoord1
            tmaStoreCommitGroup
            tmaStoreAsyncWait))

@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockKvSweep
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  tkMhaH100BwdKvSweepPc 2 q_ptr k_ptr v_ptr dO_ptr l_ptr d_ptr dQ_ptr dK_ptr dV_ptr

/-- FlashAttention forward for 12 KV blocks (seq=768, head_dim=64). -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd12Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125 -- 1 / sqrt(64)
  let scoreScaleLog2e : Float := 0.18033688011125
  setLaunchBounds 128 1

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- FlashAttention forward (12 KV blocks) with LSE output. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd12BlockLse
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125 -- 1 / sqrt(64)
  let scoreScaleLog2e : Float := 0.18033688011125
  setLaunchBounds 128 1

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lseShared ← allocSTColVec .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
  finalizeSoftmax o softmaxState
  let lse ← computeLSEScaled softmaxState scale

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeColVec lseShared lse
  storeColVecGlobalAsync lse_ptr lseShared (vectorCoordFromTileRow coord)
  tmaStoreAsyncWait

/-- `mha_h100`-style forward for 12 KV blocks (`seq=768`, `d=64`).
    Output convention matches ThunderKittens: `l = -8 * lse`. -/
@[gpu_kernel .SM90]
def tkMhaH100Fwd12Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let kvTileRows : Nat := 128
  let kvBlocks : Nat := 6
  let kvStages : Nat := 4
  let scoreScaleLog2e : Float := 0.18033688011125
  let lScale : Float := -8.0
  let qBytes : Nat := tileSize * tileSize * GpuFloat.bytes .BFloat16
  let kvBytes : Nat := kvTileRows * tileSize * GpuFloat.bytes .BFloat16
  setLaunchBounds 512 1

  let zero ← constIntVal 0 "mha_fwd_zero"
  let one ← constIntVal 1 "mha_fwd_one"
  let two ← constIntVal 2 "mha_fwd_two"
  let three ← constIntVal 3 "mha_fwd_three"
  let four ← constIntVal 4 "mha_fwd_four"

  let blockX ← getBlockIdx 0 "mha_fwd_block_x"
  let blockY ← getBlockIdx 1 "mha_fwd_head"
  let blockZ ← getBlockIdx 2 "mha_fwd_batch"
  let seqIdx ← scalarMulVal blockX three "mha_fwd_seq_idx"
  let qRow0 := seqIdx
  let qRow1 ← scalarAddVal seqIdx one "mha_fwd_qrow1"
  let qRow2 ← scalarAddVal seqIdx two "mha_fwd_qrow2"

  let mkTileCoord (row : KVal UInt32) : RTileCoord :=
    { b := blockZ.id, d := blockY.id, r := row.id, c := zero.id }
  let mkLCoord (row : KVal UInt32) : RTileCoord :=
    { b := blockZ.id, d := blockY.id, r := zero.id, c := row.id }

  let qShared ← allocSTArray .BFloat16 tileSize tileSize .Row 3
  let kShared ← allocSTArray .BFloat16 kvTileRows tileSize .Row kvStages
  let vShared ← allocSTArray .BFloat16 kvTileRows tileSize .Row kvStages
  let lShared ← allocSTColVecArray .Float32 tileSize tileSize 3

  let qSem ← allocSemaphore
  let kSem ← allocSemaphoreArray kvStages
  let vSem ← allocSemaphoreArray kvStages
  let computeDone ← allocSemaphoreArray kvStages

  initSemaphore qSem 0 1
  for stage in List.range kvStages do
    let stageVal ← constIntVal stage s!"mha_fwd_stage_init{stage}"
    initSemaphoreArray kSem stageVal 0 1
    initSemaphoreArray vSem stageVal 0 1
    initSemaphoreArray computeDone stageVal 3
  let loadKvStageInitial (stage kvIdx : Nat) : KernelM Unit := do
    let stageVal ← constIntVal stage s!"mha_fwd_stage{stage}"
    let kvIdxVal ← constIntVal kvIdx s!"mha_fwd_kv{kvIdx}"
    let kvCoord := mkTileCoord kvIdxVal
    expectBytesArray kSem stageVal kvBytes
    loadGlobalAsyncArraySemArray kShared stageVal k_ptr kvCoord kSem stageVal
    expectBytesArray vSem stageVal kvBytes
    loadGlobalAsyncArraySemArray vShared stageVal v_ptr kvCoord vSem stageVal
  let loadKvStageWarp (stage kvIdx : Nat) : KernelM Unit := do
    let stageVal ← constIntVal stage s!"mha_fwd_stage{stage}"
    let kvIdxVal ← constIntVal kvIdx s!"mha_fwd_kv{kvIdx}"
    let kvCoord := mkTileCoord kvIdxVal
    expectBytesArrayWarp kSem stageVal kvBytes
    loadGlobalAsyncArrayWarp kShared stageVal k_ptr kvCoord kSem stageVal
    expectBytesArrayWarp vSem stageVal kvBytes
    loadGlobalAsyncArrayWarp vShared stageVal v_ptr kvCoord vSem stageVal

  expectBytes qSem (3 * qBytes)
  loadGlobalAsyncArray qShared zero q_ptr (mkTileCoord qRow0) qSem.id
  loadGlobalAsyncArray qShared one q_ptr (mkTileCoord qRow1) qSem.id
  loadGlobalAsyncArray qShared two q_ptr (mkTileCoord qRow2) qSem.id
  for stage in List.range (kvStages - 1) do
    loadKvStageInitial stage stage
  blockSync

  let warpGroupIdx ← getWarpGroupIdx
  let warpGroupLaneId ← getWarpGroupLaneId "mha_fwd_wg_lane_id"
  let warpId ← getWarpId "mha_fwd_warp_id"
  let isWarpGroup3 ← scalarEq warpGroupIdx three "mha_fwd_is_wg3"
  let isWarpGroupLane0 ← scalarEq warpGroupLaneId zero "mha_fwd_is_wg_lane0"
  let localWarp ← scalarMod warpId four "mha_fwd_local_warp"
  let isLocalWarp0 ← scalarEq localWarp zero "mha_fwd_is_local_warp0"
  let qRowDyn ← scalarAddVal seqIdx warpGroupIdx "mha_fwd_qrow_dyn"
  let consumerBarrier ← scalarAddVal warpGroupIdx four "mha_fwd_consumer_barrier"

  let consumerBody : KernelM Unit := do
    warpgroupIncreaseRegisters 160
    let att ← allocRT .Float32 16 kvTileRows
    let attBf16 ← allocRT .BFloat16 16 kvTileRows
    let oReg ← zeroRT .Float32 16 tileSize
    let rowMax ← negInftyRV .Float32 16
    let rowSum ← zeroRV .Float32 16
    let lastScaled ← allocRV .Float32 16
    let maxScaled ← allocRV .Float32 16

    waitSemaphore qSem
    for kvIdx in krange 0 kvBlocks do
      let kvIdxVal : KVal UInt32 := ⟨kvIdx.id, "mha_fwd_kv_loop"⟩
      let stageVal ← scalarMod kvIdxVal four "mha_fwd_stage"
      let phaseVal ← scalarDivVal kvIdxVal four "mha_fwd_phase"
      waitSemaphoreArrayPhaseVal kSem stageVal phaseVal
      warpgroupMmSharedArrayT64x16 att qShared warpGroupIdx kShared stageVal
      onlineSoftmaxLog2TkH100 att rowMax rowSum lastScaled maxScaled scoreScaleLog2e
      convert attBf16 att
      mulCol oReg oReg lastScaled
      waitSemaphoreArrayPhaseVal vSem stageVal phaseVal
      warpgroupMmaRhsArray oReg attBf16 vShared stageVal
      mmaAsyncWait
      ifThen isWarpGroupLane0 do
        arriveSemaphoreArrayWarp computeDone stageVal 1

    divCol oReg oReg rowSum
    warpgroupStoreArray qShared warpGroupIdx oReg
    groupSyncVal 4 consumerBarrier
    ifThen isLocalWarp0 do
      storeGlobalAsyncArray o_ptr qShared warpGroupIdx (mkTileCoord qRowDyn)

    let l ← computeLSEScaledTkH100 rowSum maxScaled
    scalarMulVec l l lScale
    warpgroupStoreColVecArray lShared warpGroupIdx l
    groupSyncVal 4 consumerBarrier
    ifThen isLocalWarp0 do
      storeColVecGlobalAsyncArray l_ptr lShared warpGroupIdx (mkLCoord qRowDyn)
    tmaStoreAsyncWait

  ifThenElse isWarpGroup3
    (do
      warpgroupDecreaseRegisters 32
      ifThen isLocalWarp0 do
        loadKvStageWarp 3 3
        waitSemaphoreArrayPhaseVal computeDone two zero
        loadKvStageWarp 0 4
        waitSemaphoreArrayPhaseVal computeDone three zero
        loadKvStageWarp 1 5
        waitSemaphoreArrayPhaseVal computeDone zero one)
    (do
      consumerBody)

/-- `mha_h100` backward for 12 KV blocks (`seq=768`, non-causal) with in-kernel
    `dK`/`dV` accumulation. As above, the function name keeps the older
    `...Partials` suffix for ABI stability while callers pass final
    zero-initialized `[1, 1, seq, 64]` gradient buffers. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockPartials
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let dO ← allocRT .BFloat16 tileSize tileSize
  let dQ ← zeroRT .Float32 tileSize tileSize
  let lTk ← allocRV .Float32 tileSize
  let lse ← allocRV .Float32 tileSize
  let dVec ← allocRV .Float32 tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  -- Reuse one FP32 staging tile; wait before overwriting it with the next TMA store.
  let dKVShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize
    let dKPart ← zeroRT .Float32 tileSize tileSize
    let dVPart ← zeroRT .Float32 tileSize tileSize

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let pTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert pTBf16 pT
    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dOCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dVPart pTBf16 dOCol dVPart

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := coord.withRow kvIdx.id
    store dKVShared dKPart
    groupSync 4 4
    storeGlobalAdd dK_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4
    store dKVShared dVPart
    groupSync 4 4
    storeGlobalAdd dV_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4

  store dKVShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dKVShared coord

/-- Q-centric `dQ` pass for 12-block H100 MHA backward (`seq=768`, `d=64`). -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockDq
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let dO ← allocRT .BFloat16 tileSize tileSize
  let dQ ← zeroRT .Float32 tileSize tileSize
  let lTk ← allocRV .Float32 tileSize
  let lse ← allocRV .Float32 tileSize
  let dVec ← allocRV .Float32 tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let dQShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

  store dQShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dQShared coord

/-- KV-centric `mha_h100` backward sweep for 12 blocks (`seq=768`, `d=64`).
    Each CTA owns one KV tile and sweeps all query tiles, matching the
    ThunderKittens K/V reduction direction. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockKvSweep
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  tkMhaH100BwdKvSweepPc 12 q_ptr k_ptr v_ptr dO_ptr l_ptr d_ptr dQ_ptr dK_ptr dV_ptr

end Tyr.GPU.Kernels
