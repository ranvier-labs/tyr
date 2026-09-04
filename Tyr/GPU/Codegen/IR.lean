import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.AST

/-!
# Tyr.GPU.Codegen.IR

`Tyr.GPU.Codegen.IR` is the central intermediate representation for Tyr GPU kernels.
It models kernels as `KStmt` programs over `VarId`s and operation tags from
`Codegen.AST`.

Key pieces:

- `KScalarType`: scalar parameter kinds used at kernel boundaries.
- `KStmt`: declarative instruction set for allocation, memory, mma, control flow.
- `KParam`: typed kernel signature parameters.
- `Kernel`: complete lowered kernel record consumed by emitters.

Higher-level modules (`Ops`, `Notation`, `Arch`) construct this IR; emitters
(`EmitNew`) serialize it to target-specific CUDA/C++.
-/

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Scalar parameter type for `KVal` kernel parameters. -/
inductive KScalarType where
  /-- 8-bit unsigned integer (`uint8_t`). -/
  | UInt8
  /-- 16-bit unsigned integer (`uint16_t`). -/
  | UInt16
  /-- 32-bit unsigned integer (`uint32_t`); the default scalar size. -/
  | UInt32
  /-- 64-bit unsigned integer (`uint64_t`); used for flat pointer offsets. -/
  | UInt64
  /-- Host `size_t`. -/
  | USize
  /-- Lean `Float` mapped to C++ `double`. -/
  | Float
  /-- Single-precision IEEE 754 (`float`). -/
  | Float32
  /-- Boolean predicate, lowered as `uint8_t`. -/
  | Bool
  deriving Repr, Inhabited, BEq

/-- Convert scalar type to C++ type name used in extern/kernel signatures. -/
def KScalarType.toCpp : KScalarType → String
  | .UInt8 => "uint8_t"
  | .UInt16 => "uint16_t"
  | .UInt32 => "uint32_t"
  | .UInt64 => "uint64_t"
  | .USize => "size_t"
  | .Float => "double"
  | .Float32 => "float"
  | .Bool => "uint8_t"

/-- Concrete ThunderKittens tile descriptor shape required on a global pointer.

Most generated kernels infer these from typed TMA load/store statements. Some
kernel templates need to stage through C++ helper code or raw TK idioms while
still using generated launch wrappers, so they need a declarative way to attach
the same descriptor to the generated `gl<...>` parameter type. -/
inductive GlobalTileDescriptor where
  /-- Plain shared tile descriptor `st<dtype, rows, cols>` on the global pointer. -/
  | st (dtype : GpuFloat) (rows cols : Nat)
  /-- Row-vector tile descriptor `row_vec<st<dtype, rows, cols>>`. -/
  | rowVecSt (dtype : GpuFloat) (rows cols : Nat)
  /-- Column-vector tile descriptor `col_vec<st<dtype, rows, cols>>`. -/
  | colVecSt (dtype : GpuFloat) (rows cols : Nat)
  deriving Repr, Inhabited, BEq, Hashable

/-- GPU kernel statement using VarId -/
inductive KStmt where
  -- Tile declarations
  /-- Declare a register tile `rt<dtype, rows, cols, layout>` (per-warp registers). -/
  | declRT (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  /-- Declare a shared-memory tile `st<dtype, rows, cols, layout>` (CTA-shared SMEM). -/
  | declST (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  /-- Declare a length-`len` array of shared tiles in dynamic shared memory. -/
  | declSTArray (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout) (len : Nat)
  /-- Declare a typed shared-tile alias view over an existing SMEM allocation
      (no additional shared-memory bytes are reserved). -/
  | declSTAlias (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
      (src : VarId) (comment : String)
  /-- Declare a register vector `rv<dtype, len>` distributed across the warp. -/
  | declRV (v : VarId) (dtype : GpuFloat) (len : Nat)
  /-- Declare a shared-memory vector `sv<dtype, len>`. -/
  | declSV (v : VarId) (dtype : GpuFloat) (len : Nat)
  /-- Declare a shared `row_vec<st<dtype, rows, cols>>` view (storage is `sv<dtype, cols>`). -/
  | declSTRowVec (v : VarId) (dtype : GpuFloat) (rows cols : Nat)
  /-- Declare a shared `col_vec<st<dtype, rows, cols>>` view (storage is `sv<dtype, rows>`). -/
  | declSTColVec (v : VarId) (dtype : GpuFloat) (rows cols : Nat)
  /-- Declare an array of shared `col_vec<st<dtype, rows, cols>>` views. -/
  | declSTColVecArray (v : VarId) (dtype : GpuFloat) (rows cols len : Nat)
  /-- Declare a single shared semaphore/barrier (`__shared__ semaphore`). -/
  | declSemaphore (v : VarId)
  /-- Declare a length-`len` shared-memory array of semaphores. -/
  | declSemaphoreArray (v : VarId) (len : Nat)

  -- Tensor memory tile declarations (Blackwell SM100)
  /-- Declare a Blackwell tensor-memory tile `tt<dtype, rows, cols>` (SM100 TMEM). -/
  | declTT (v : VarId) (dtype : GpuFloat) (rows cols : Nat)
  /-- Declare a Blackwell tensor-memory allocator. -/
  | declTMEMPool (v : VarId) (numCtas clusterSize : Nat)

  -- Kernel parameter declarations (used by attribute-generated code)
  /-- Declare a global-memory pointer kernel parameter (named in C++). -/
  | declGPtr (v : VarId) (dtype : GpuFloat) (name : String)
  /-- Declare a scalar-typed runtime kernel parameter (named in C++). -/
  | declKVal (v : VarId) (dtype : GpuFloat) (name : String)
  /-- Declare a mutable local scalar variable, optionally initialized from another scalar. -/
  | declScalar (v : VarId) (ty : KScalarType) (init : Option VarId)

  -- Memory operations
  /-- SMEM→register tile copy: `kittens::load(rt, st)`; shapes/layout must match. -/
  | load (dst src : VarId)
  /-- Load a compile-time tile fragment from shared memory into a register tile. -/
  | loadSubtile (dst src : VarId) (rows cols rowIdx colIdx : Nat)
  /-- Load a runtime-selected tile fragment from shared memory into registers. -/
  | loadSubtileVal (dst src : VarId) (rows cols : Nat) (rowIdx colIdx : VarId)
  /-- Four-warp cooperative shared-to-register load. Each warp receives one
      row fragment of the shared tile. -/
  | warpgroupLoad (dst src : VarId)
  /-- Register→SMEM tile copy: `kittens::store(st, rt)`; shapes/layout must match. -/
  | store (dst src : VarId)
  /-- Async SMEM→SMEM copy via TMA: `kittens::load_async`. -/
  | loadAsync (dst src : VarId)
  /-- Async register→SMEM store via TMA: `kittens::store_async`. -/
  | storeAsync (dst src : VarId)
  /-- Atomic store-add for gradient accumulation (e.g. dQ in attention backward). -/
  | storeAdd (dst src : VarId)
  /-- Async atomic store-add (`tma::store_add_async`) for gradient accumulation. -/
  | storeAddAsync (dst src : VarId)
  /-- Async atomic store-min (`tma::store_min_async`). -/
  | storeMinAsync (dst src : VarId)
  /-- Warpgroup-wide store of a register fragment into a shared tile (`warpgroup::store`). -/
  | warpgroupStore (dst src : VarId)
  /-- Warpgroup store into element `dstIdx` of a shared-tile array. -/
  | warpgroupStoreIdx (dst dstIdx src : VarId)
  /-- Commit the current async TMA store group (`tma::store_commit_group`). -/
  | tmaStoreCommitGroup
  /-- Wait for outstanding async TMA stores (`tma::store_async_wait`). -/
  | tmaStoreAsyncWait
  /-- TMA prefetch hint for a shared tile descriptor. -/
  | prefetch (src : VarId)
  /-- Set a transaction-byte expectation on a barrier prior to TMA load. -/
  | tmaExpect (barrier : VarId) (bytes : Nat)
  /-- CTA-wide rendezvous: lowers to `__syncthreads()`. -/
  | blockSync
  /-- ThunderKittens `group<warps>::sync(barrierId)` synchronization. -/
  | groupSync (warps barrierId : Nat)
  /-- `group<warps>::sync` with a runtime barrier id. -/
  | groupSyncVal (warps : Nat) (barrierId : VarId)

  -- TMA operations with global pointers (legacy single coord)
  /-- Legacy single-coordinate TMA load: `shared ← global[coord]`. -/
  | tmaLoad (dst src : VarId) (coord : VarId)
  /-- Legacy single-coordinate TMA store: `global[coord] ← shared`. -/
  | tmaStore (dst src : VarId) (coord : VarId)

  -- Global memory operations with 4D coordinates (ThunderKittens style)
  /-- Synchronous GMEM→SMEM tile load at 4D coord `{b,d,r,c}` (`kittens::load`). -/
  | loadGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Four-warp cooperative GMEM→register tile load at 4D coord. -/
  | warpgroupLoadGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Warp-level GMEM→register tile load at 4D coord. -/
  | loadRegisterGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Synchronous SMEM→GMEM tile store at 4D coord `{b,d,r,c}` (`kittens::store`). -/
  | storeGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Four-warp cooperative register→GMEM tile store at 4D coord. -/
  | warpgroupStoreGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Warp-level register→GMEM tile store at 4D coord. -/
  | storeRegisterGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Async TMA load at 4D coord, signaling completion on `sem` (`tma::load_async`). -/
  | loadGlobalAsync (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  /-- Async TMA load at 4D coord issued by the currently active warp. -/
  | loadGlobalAsyncWarp (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  /-- Async TMA load into element `dstIdx` of a shared-tile array. -/
  | loadGlobalAsyncIdx (dst dstIdx src : VarId) (coordB coordD coordR coordC sem : VarId)
  /-- Async TMA load into shared-tile-array element `dstIdx` with semaphore-array index `semIdx`. -/
  | loadGlobalAsyncIdxSemIdx (dst dstIdx src : VarId) (coordB coordD coordR coordC sem semIdx : VarId)
  /-- Warp-issued async TMA load into shared-tile-array element with semaphore-array index. -/
  | loadGlobalAsyncWarpIdx (dst dstIdx src : VarId) (coordB coordD coordR coordC sem semIdx : VarId)
  /-- Async TMA store at 4D coord (`tma::store_async`). -/
  | storeGlobalAsync (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Async TMA store from element `srcIdx` of a shared-tile array. -/
  | storeGlobalAsyncIdx (dst src srcIdx : VarId) (coordB coordD coordR coordC : VarId)
  /-- Atomic add store at 4D coord for gradient accumulation (`tma::store_add_async`). -/
  | storeGlobalAdd (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Warp-issued atomic add store at 4D coord. -/
  | storeGlobalAddWarp (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Attach a TK tile descriptor to a `gl<...>` parameter without emitting a load. -/
  | requireGlobalTma (ptr : VarId) (descriptor : GlobalTileDescriptor)
  /-- Read a dynamic dimension (batch/depth/rows/cols) from a global-layout parameter. -/
  | layoutDim (dst src : VarId) (axis : LayoutDimAxis)

  -- Vector global memory operations
  /-- Vector GMEM→SMEM load at flat element offset (`kittens::load`). -/
  | loadVecGlobal (dst src : VarId) (offset : VarId)
  /-- Vector SMEM→GMEM store at flat element offset (`kittens::store`). -/
  | storeVecGlobal (dst src : VarId) (offset : VarId)
  /-- Atomic add vector store at flat element offset. -/
  | storeVecGlobalAdd (dst src : VarId) (offset : VarId)
  /-- Vector GMEM→SMEM load using a 4D runtime coordinate. -/
  | loadVecGlobalCoord (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Vector SMEM→GMEM store using a 4D runtime coordinate. -/
  | storeVecGlobalCoord (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Atomic add vector store using a 4D runtime coordinate. -/
  | storeVecGlobalAddCoord (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Load a single scalar from global memory at a runtime element offset. -/
  | loadScalarGlobal (dst src offset : VarId)
  /-- Store a single scalar to global memory at a runtime element offset. -/
  | storeScalarGlobal (dst src offset : VarId)
  /-- Load one global element with conversion (e.g. BF16 → Float32). -/
  | loadFlatGlobal (dst src offset : VarId) (srcDtype : GpuFloat) (dstTy : KScalarType)
  /-- Store one scalar to a typed global element with conversion (e.g. Float32 → BF16). -/
  | storeFlatGlobal (dst offset src : VarId) (dstDtype : GpuFloat) (srcTy : KScalarType)
  /-- Declare a typed view into dynamic shared memory at an element offset. -/
  | declSharedLinear (v offsetElems : VarId) (ty : KScalarType)
  /-- Load a scalar from a dynamic shared-memory linear view. -/
  | loadSharedLinear (dst buf offset : VarId) (ty : KScalarType)
  /-- Store a scalar into a dynamic shared-memory linear view. -/
  | storeSharedLinear (buf offset src : VarId) (ty : KScalarType)

  -- Distributed / Multimem operations
  /-- Multimem load-reduce: register tile = reduce(SMEM tile) across the cluster. -/
  | multimemLoadReduce (op : ReduceOp) (dst src : VarId)
  /-- Multimem store: SMEM tile broadcast across the cluster. -/
  | multimemStore (dst src : VarId)
  /-- Multimem reduce: SMEM tile op= register tile across the cluster. -/
  | multimemRed (op : ReduceOp) (dst src : VarId)

  -- MMA operations
  /-- Tensor-core MMA `dst = a · b + c` with transpose tag; lowers to `kittens::mma`. -/
  | mma (trans : MMATranspose) (dst a b c : VarId)
  /-- Tensor-core MM (no accumulate) `dst = a · b`; lowers to `kittens::mm`. -/
  | mm (trans : MMATranspose) (dst a b : VarId)
  /-- Hopper WGMMA accumulate `dst += a · b` (`warpgroup::mma_*`); async, requires later wait. -/
  | warpgroupMma (trans : MMATranspose) (dst a b : VarId)
  /-- Hopper WGMMA overwrite `dst = a · b` (`warpgroup::mm_*`); async, requires later wait. -/
  | warpgroupMm (trans : MMATranspose) (dst a b : VarId)
  /-- WGMMA accumulate with both A/B sourced from indexed shared-tile arrays. -/
  | warpgroupMmaIdx (trans : MMATranspose) (dst a aIdx b bIdx : VarId)
  /-- WGMMA overwrite with both A/B sourced from indexed shared-tile arrays. -/
  | warpgroupMmIdx (trans : MMATranspose) (dst a aIdx b bIdx : VarId)
  /-- WGMMA accumulate with B sourced from element `bIdx` of a shared-tile array. -/
  | warpgroupMmaRhsIdx (trans : MMATranspose) (dst a b bIdx : VarId)
  /-- WGMMA fence on a register accumulator (`warpgroup::mma_fence`). -/
  | mmaFence (dst : VarId)
  /-- Commit pending WGMMA operations into a group (`warpgroup::mma_commit_group`). -/
  | mmaCommitGroup
  /-- Wait until at most `n` WGMMA groups are still in flight (`warpgroup::mma_async_wait`). -/
  | mmaAsyncWait (n : Nat)

  -- Blackwell-specific MMA (tcgen05 / 2-CTA MMA)
  /-- Blackwell SM100 MMA with zero-init into TMEM destination (`mm2_*`). -/
  | tcgen05Mm (trans : MMATranspose) (dst a b : VarId)
  /-- Blackwell SM100 accumulating MMA into TMEM destination (`mma2_*`). -/
  | tcgen05Mma (trans : MMATranspose) (dst a b c : VarId)
  /-- Blackwell SM100 scaled MMA (MXFP8/NVFP4) with E8M0 scale tiles in TMEM. -/
  | tcgen05MmaScaled (trans : MMATranspose) (dst a b c scaleA scaleB : VarId)
  /-- Commit tcgen05 results to a cluster-aware semaphore. -/
  | tcgen05Commit (sem : VarId) (clusterSize : Nat)
  /-- Cooperatively load a tensor-memory tile into per-warp register fragments. -/
  | tensorLoadAsync (dst src : VarId)
  /-- Wait for prior tensor-memory loads to become visible in registers. -/
  | tensorLoadWait

  -- Tensor memory operations (Blackwell SM100)
  /-- Allocate a TT from a TMEM pool at a given byte offset. -/
  | tmemAllocate (dst pool : VarId) (dtype : GpuFloat) (rows cols superlane offset : Nat)
  /-- Provision a TMEM pool across `clusterSize` CTAs (must be inside `electOneSync`). -/
  | tmemProvision (pool : VarId) (clusterSize : Nat)
  /-- Release a previously provisioned TMEM pool. -/
  | tmemDeprovision (pool : VarId)
  /-- Load an E8M0 scale tile into TMEM at the given pipeline `stage`. -/
  | loadScaleTmem (dst src : VarId) (stage : Nat)
  /-- Extract a subtile from a pipelined TMEM allocation at byte `offset`. -/
  | tmemSubtile (dst src : VarId) (offset : Nat)

  -- Cluster operations (SM90+ clusters, SM100 2-CTA)
  /-- Read this CTA's rank within the cluster along `axis` (0=x, 1=y, 2=z). -/
  | clusterIdx (dst : VarId) (axis : Nat)
  /-- Cluster-scoped TMA load visible to all CTAs in the cluster. -/
  | clusterTmaLoad (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  /-- Cluster-scoped TMA store from cluster scope. -/
  | clusterTmaStore (dst src : VarId) (coordB coordD coordR coordC : VarId)
  /-- Arrive on a cluster-scope barrier semaphore. -/
  | clusterArrive (sem : VarId)
  /-- Wait on a cluster-scope barrier semaphore. -/
  | clusterWait (sem : VarId)

  -- Architecture-specific load variants (for explicit control)
  /-- Explicit SM80 `cp.async` GMEM→SMEM load at 4D coord (no TMA). -/
  | cpAsyncLoad (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  /-- Explicit Hopper/Blackwell TMA load at 4D coord (forces TMA path). -/
  | tmaLoadAsync (dst src : VarId) (coordB coordD coordR coordC sem : VarId)

  -- Element-wise unary
  /-- Element-wise unary op on a tile or vector (`kittens::<op>`). -/
  | unary (op : UnaryOp) (dst src : VarId)

  -- Element-wise binary
  /-- Element-wise binary op on matching-shape tiles/vectors. -/
  | binary (op : BinaryOp) (dst a b : VarId)

  -- Element-wise ternary (FMA)
  /-- Element-wise ternary op (e.g. fused multiply-add `dst = a*b + c`). -/
  | ternary (op : TernaryOp) (dst a b c : VarId)

  -- Element-wise comparison mask
  /-- Element-wise equality mask: `dst = (a == b) ? 1 : 0`. -/
  | eqMask (dst a b : VarId)

  -- Scalar operations
  /-- Multiply a tile/vector element-wise by a compile-time `Float` scalar. -/
  | scalarMul (dst src : VarId) (scalar : Float)
  /-- Add a compile-time `Float` scalar element-wise to a tile/vector. -/
  | scalarAdd (dst src : VarId) (scalar : Float)

  -- Broadcasting
  /-- Broadcast a row/col vector across a tile (`kittens::broadcast_{row,col}`). -/
  | broadcast (axis : BroadcastAxis) (dst vec : VarId)
  /-- Broadcasted element-wise binary: `dst = tile op broadcast(vec)`. -/
  | binaryBroadcast (op : BinaryOp) (axis : BroadcastAxis) (dst tile vec : VarId)

  -- Reductions
  /-- Reduce a tile along an axis into a vector (`kittens::row_/col_<op>`). -/
  | reduce (op : ReduceOp) (axis : ReduceAxis) (dst src : VarId)
  /-- Reduce along an axis combined with an existing accumulator vector. -/
  | reduceAccum (op : ReduceOp) (axis : ReduceAxis) (dst src accum : VarId)

  -- Scan/prefix operations
  /-- Cumulative sum along an axis (`kittens::cumsum_*`). -/
  | cumsum (axis : ReduceAxis) (dst src : VarId)
  /-- Cumulative product along an axis (used for decay in Mamba/SSM kernels). -/
  | cumprod (axis : ReduceAxis) (dst src : VarId)

  -- Outer product
  /-- Outer product `dst[i,j] = a[i] * b[j]`. -/
  | outer (dst a b : VarId)

  -- Layout/type conversions
  /-- Swap a register tile's layout between Row and Col (`kittens::swap_layout`). -/
  | swapLayout (dst src : VarId)
  /-- Transpose a register tile (`kittens::transpose`). -/
  | transpose (dst src : VarId)
  /-- Element-wise dtype conversion (e.g. BF16↔Float32) on matching-shape tiles. -/
  | convert (dst src : VarId)
  /-- Copy a register vector while converting between TK align/ortho layouts. -/
  | convertVecLayout (dst src : VarId)

  -- Masking
  /-- Apply a compile-time mask op (causal/tri/fill) optionally with a fill value. -/
  | mask (op : MaskOp) (dst src : VarId) (fillVal : Option Float)
  /-- Runtime-cutoff right-fill: mask cols ≥ `colVar` with `fillVal`. The compile-time
      `MaskOp.RightFill` variant takes a `Nat`; this variant accepts a runtime
      `KVal UInt32` (the underlying VarId). Used by decode-style attention where
      the valid KV length within the last block is determined at runtime. -/
  | maskRightFillVal (dst src colVar : VarId) (fillVal : Option Float)

  -- Tile slicing
  /-- Take a row-slice subtile starting at `startRow` of `numRows` rows. -/
  | sliceRows (dst src : VarId) (startRow numRows : Nat)
  /-- Take a column-slice subtile starting at `startCol` of `numCols` columns. -/
  | sliceCols (dst src : VarId) (startCol numCols : Nat)
  /-- Concatenate two tiles along the column axis (`dst = [left | right]`). -/
  | concatCols (dst left right : VarId)

  -- Synchronization
  /-- Wait at a numbered barrier (`bar.sync barrierId`). -/
  | sync (barrierId : Nat)
  /-- Signal arrival on a numbered barrier without waiting. -/
  | arrive (barrierId : Nat)
  /-- Arrive and wait on a numbered barrier. -/
  | arriveAndWait (barrierId : Nat)

  -- Named barriers (for warp specialization in FA3)
  /-- PTX named-barrier sync with explicit thread count (FA3 warp specialization). -/
  | namedBarrierSync (id : Nat) (numThreads : Nat)
  /-- PTX named-barrier arrive with explicit thread count (FA3 warp specialization). -/
  | namedBarrierArrive (id : Nat) (numThreads : Nat)

  -- Warp group operations (for warp specialization)
  /-- Read warpgroup index within the CTA (`kittens::warpgroup::groupid()`). -/
  | warpGroupIdx (dst : VarId)
  /-- Read lane id within the current warpgroup. -/
  | warpGroupLaneId (dst : VarId)
  /-- Read warp id within the CTA (`kittens::warpid()`). -/
  | warpId (dst : VarId)
  /-- Read lane id within the warp (`kittens::laneid()`). -/
  | laneId (dst : VarId)
  /-- Elect one thread per warp; result is true on the elected lane. -/
  | electOneSync (dst : VarId)
  /-- Decrease warpgroup register budget for producer/loader specialization. -/
  | warpgroupDecreaseRegisters (n : Nat)
  /-- Increase warpgroup register budget for consumer specialization. -/
  | warpgroupIncreaseRegisters (n : Nat)

  -- Fence operations (for WGMMA pipelining)
  /-- Fence ensuring SMEM writes are visible to subsequent WGMMA (`fence_view_async_shared`). -/
  | fenceViewAsyncShared
  /-- Async-proxy fence used in WGMMA pipelining (`fence_proxy_async`). -/
  | fenceProxyAsync

  -- Semaphore operations
  /-- Apply a semaphore op (init/expect/arrive/wait/invalidate) from warp 0. -/
  | semaphore (op : SemaphoreOp) (sem : VarId)
  /-- Apply a semaphore op from the currently active warp. -/
  | semaphoreWarp (op : SemaphoreOp) (sem : VarId)
  /-- Wait on a semaphore using a runtime phase value (`wait(sem, phaseVar)`). -/
  | semaphoreWaitVal (sem phase : VarId)
  /-- Apply a semaphore op to one element of a semaphore array from warp 0. -/
  | semaphoreArray (op : SemaphoreOp) (sem idx : VarId)
  /-- Apply a semaphore op to one element of a semaphore array from the active warp. -/
  | semaphoreArrayWarp (op : SemaphoreOp) (sem idx : VarId)
  /-- Wait on a semaphore-array element using a runtime phase value. -/
  | semaphoreArrayWaitVal (sem idx phase : VarId)
  /-- Arrive on a semaphore-array element with a static `count`. -/
  | semaphoreArrayArrive (sem idx : VarId) (count : Nat)

  -- Control flow
  /-- Compile-time-bounded for loop `for v in lo..hi` over a sub-program. -/
  | forLoop (v : VarId) (lo hi : Nat) (body : Array KStmt)
  /-- For loop with runtime upper bound `hi : KVal`. -/
  | forLoopVal (v : VarId) (lo : Nat) (hi : VarId) (body : Array KStmt)
  /-- Reverse for loop counting `hi-1` down to `lo`. -/
  | forLoopRev (v : VarId) (lo hi : Nat) (body : Array KStmt)
  /-- Reverse for loop with runtime upper bound. -/
  | forLoopValRev (v : VarId) (lo : Nat) (hi : VarId) (body : Array KStmt)
  /-- Strided for loop `for v in start..hi by step` with runtime bounds/step. -/
  | forLoopStrideVal (v start hi step : VarId) (body : Array KStmt)
  /-- Runtime conditional with then/else sub-programs. -/
  | ifStmt (cond : VarId) (thenBody elseBody : Array KStmt)
  /-- Execute a sub-program only on threads belonging to warpgroup `wgIdx`. -/
  | ifWarpGroup (wgIdx : Nat) (body : Array KStmt)
  /-- Emit a comment line in the generated CUDA source. -/
  | comment (text : String)

  -- Block/thread index accessors
  /-- Read `blockIdx.{x,y,z}` (axis: 0=x, 1=y, 2=z). -/
  | getBlockIdx (dst : VarId) (axis : Nat)
  /-- Read `threadIdx.{x,y,z}` (axis: 0=x, 1=y, 2=z). -/
  | getThreadIdx (dst : VarId) (axis : Nat)
  /-- Read `gridDim.{x,y,z}` (axis: 0=x, 1=y, 2=z). -/
  | getGridDim (dst : VarId) (axis : Nat)
  /-- Read `blockDim.{x,y,z}` (axis: 0=x, 1=y, 2=z). -/
  | getBlockDim (dst : VarId) (axis : Nat)

  -- Constants
  /-- Materialize a compile-time integer constant into a runtime scalar. -/
  | constInt (dst : VarId) (value : Int)
  | constUInt64 (dst : VarId) (value : Nat)
  /-- Materialize a compile-time Float constant into a runtime scalar. -/
  | constFloat (dst : VarId) (value : Float)
  /-- Apply a unary op to a runtime scalar (neg/exp/log/rsqrt/...). -/
  | scalarUnary (op : ScalarUnaryOp) (dst src : VarId)
  /-- Compare two runtime scalars; result is a Bool scalar. -/
  | scalarCompare (op : ScalarCompareOp) (dst a b : VarId)
  /-- Apply a binary op to two runtime scalars (add/sub/mul/...). -/
  | scalarBinary (op : ScalarBinaryOp) (dst a b : VarId)
  /-- Select between two runtime scalars by a Bool condition. -/
  | scalarSelect (dst cond ifTrue ifFalse : VarId)
  /-- Assign a runtime scalar from another scalar of the same type. -/
  | scalarAssign (dst src : VarId)
  /-- Cast a runtime scalar to a different scalar type. -/
  | scalarCast (dst src : VarId) (dstTy : KScalarType)
  /-- CTA-wide scalar reduction (`kittens::block_reduce`); all threads receive the result. -/
  | blockReduce (op : ReduceOp) (dst src : VarId) (ty : KScalarType)
  /-- Fill a vector with an arithmetic progression `dst[i] = start + i*step`. -/
  | vecIota (dst : VarId) (start step : Float)
  /-- Broadcast-fill a vector with a single runtime scalar value. -/
  | vecFillScalar (dst scalar : VarId)
  /-- Escape hatch: emit the given C++/CUDA code verbatim into the kernel body. -/
  | raw (code : String)

  deriving Repr, Inhabited, BEq

/-- Kernel parameter -/
structure KParam where
  /-- C++ identifier used for this parameter in the generated kernel signature. -/
  name : String
  /-- Element dtype (used for both global pointers and typed scalars). -/
  dtype : GpuFloat
  /-- True if the parameter is a global-memory pointer; false for scalars. -/
  isPointer : Bool := false
  /-- Scalar C++ type when `isPointer = false`. -/
  scalarTy : KScalarType := .UInt64
  deriving Repr, Inhabited, BEq

/-- Complete kernel definition -/
structure Kernel where
  /-- C++ identifier of the generated kernel function. -/
  name : String
  /-- Target GPU architecture (SM80/SM90/SM100/...). -/
  arch : GpuArch
  /-- Target GPU family (Ampere, Hopper, or Blackwell). -/
  family : GpuFamily
  /-- Ordered kernel signature parameters (pointers and scalars). -/
  params : Array KParam
  /-- Lowered kernel body as a sequence of `KStmt` instructions. -/
  body : Array KStmt
  /-- Total dynamic shared-memory bytes the launch wrapper must reserve. -/
  sharedMemBytes : Nat := 0
  /-- Optional `__launch_bounds__(maxThreads, minBlocks)` annotation. -/
  launchBounds : Option (Nat × Nat) := none
  deriving Repr, Inhabited, BEq

end Tyr.GPU.Codegen
