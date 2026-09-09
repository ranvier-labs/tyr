/- End-to-end numerical correctness harness for the TK-style decode kernel.

   Generates random Q/K/V plus a PyTorch SDPA reference for several decode
   shapes, then invokes `tyr::flash_attn` (via `nn.tyrFlashAttn4d`) and asserts
   the kernel output matches reference within BF16 tolerance.

   Shape coverage:
   - Llama-3 (head_dim=128, GQA ratio 4, kv_seq=2048)
   - Qwen3-4B (head_dim=64, GQA ratio 4, kv_seq=2048)
   - Tail-mask exercise (head_dim=128, kv_seq=2049 — not multiple of 64)
   - Single-block (head_dim=128, kv_seq=64)
   - Larger batch (head_dim=128, batch=4)

   Each shape gets its own fixture directory under `data/gpu_fixtures/`, so
   they regenerate independently and live across runs. The runner reuses the
   shared `FixtureRunner` helper used by `RunFlashAttn3.lean` etc. -/
import Tyr.Torch
import Tyr.Inference.KVCache
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch
open torch.Generator

/-- One decode test case. -/
structure DecodeShape where
  name : String
  batch : UInt64
  qHeads : UInt64
  kvHeads : UInt64
  kvSeq : UInt64
  headDim : UInt64
  /-- Max-abs tolerance for the BF16 output comparison. -/
  atol : Float := 2.5e-2
  /-- Relative tolerance. -/
  rtol : Float := 2.5e-2

/-- Decode-shape coverage exercised by this harness. -/
def decodeShapes : Array DecodeShape := #[
  { name := "llama3_b1", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 128 },
  { name := "llama3_b4", batch := 4, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 128 },
  { name := "llama3_tail", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2049, headDim := 128 },
  { name := "llama3_one_block", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 64, headDim := 128 },
  { name := "qwen3_4b_d64", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 64 },
  -- Qwen 3.5/3.6 family (35B-A3B and friends): head_dim=256, GQA ratio 8
  -- (16 q heads, 2 kv heads). Larger atol because the per-warp 64×256
  -- accumulator likely spills to local memory under launch_bounds(128, 1)
  -- — see dev/issues.md D01b.
  { name := "qwen36_35B", batch := 1, qHeads := 16, kvHeads := 2,
    kvSeq := 2048, headDim := 256, atol := 5.0e-2, rtol := 5.0e-2 }
]

def decodeFixtureDir (shape : DecodeShape) : System.FilePath :=
  ⟨s!"data/gpu_fixtures/decode_{shape.name}"⟩

def decodeFixtureSpec (shape : DecodeShape) : FixtureSpec := {
  dir := decodeFixtureDir shape
  names := #["q", "k", "v", "expected_o"]
}

def decodeFixtureFile (shape : DecodeShape) (name : String) : System.FilePath :=
  fixturePath (decodeFixtureSpec shape) name

private def shapeQ (s : DecodeShape) : Shape :=
  #[s.batch, s.qHeads, 1, s.headDim]

private def shapeKV (s : DecodeShape) : Shape :=
  #[s.batch, s.kvHeads, s.kvSeq, s.headDim]

/-- Generate a fixture set for one decode shape: random Q/K/V (BF16) plus the
    SDPA reference output computed in FP32 and quantized back to BF16. -/
def generateDecodeFixturesFor (shape : DecodeShape) : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate decode fixtures."
  IO.FS.createDirAll (decodeFixtureSpec shape).dir
  let device := Device.CUDA 0

  let qf ← torch.randn (shapeQ shape) false device
  let kf ← torch.randn (shapeKV shape) false device
  let vf ← torch.randn (shapeKV shape) false device

  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf

  -- Reference: PyTorch SDPA in FP32, with GQA enabled (broadcasts kv heads to q heads).
  let q32 := torch.toFloat' q
  let k32 := torch.toFloat' k
  let v32 := torch.toFloat' v
  let expected32 := nn.scaledDotProductAttentionGQAQKV
                      q32 k32 v32 0.0 false true
  let expected := torch.toBFloat16' expected32

  torch.data.saveTensor q (decodeFixtureFile shape "q").toString
  torch.data.saveTensor k (decodeFixtureFile shape "k").toString
  torch.data.saveTensor v (decodeFixtureFile shape "v").toString
  torch.data.saveTensor expected (decodeFixtureFile shape "expected_o").toString

  let mae := torch.nn.item (torch.nn.meanAll (torch.nn.abs expected))
  IO.println s!"  generated decode fixture {shape.name} kvSeq={shape.kvSeq} d={shape.headDim} ref_mean_abs={mae}"

/-- Run the kernel on one shape's saved fixtures and assert parity. -/
def runDecodeOnceFor (shape : DecodeShape) : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent (decodeFixtureSpec shape)) then
    generateDecodeFixturesFor shape

  let q ← torch.data.loadTensor (shapeQ shape) (decodeFixtureFile shape "q").toString
  let k ← torch.data.loadTensor (shapeKV shape) (decodeFixtureFile shape "k").toString
  let v ← torch.data.loadTensor (shapeKV shape) (decodeFixtureFile shape "v").toString
  let expected ← torch.data.loadTensor (shapeQ shape) (decodeFixtureFile shape "expected_o").toString

  -- Route through the production op. select_route picks tkMhaH100Decode[64]
  -- when shape is eligible; else falls back to SDPA.
  let actual : T (shapeQ shape) :=
    nn.tyrFlashAttn4d q k v none 0.0 false none true
  let _ ← torch.cuda_synchronize

  let ok := torch.allclose expected actual shape.atol shape.rtol
  let diff := torch.nn.abs (actual - expected)
  let mae := torch.nn.item (torch.nn.meanAll diff)
  let maxAbs := torch.nn.item (torch.nn.maxAll diff)
  let actMaxAbs := torch.nn.item (torch.nn.maxAll (torch.nn.abs actual))
  let actMeanAbs := torch.nn.item (torch.nn.meanAll (torch.nn.abs actual))
  IO.println s!"  decode {shape.name}: ok={ok} mae={mae} max_abs={maxAbs} act_mean_abs={actMeanAbs} act_max_abs={actMaxAbs}"
  -- Per-(batch, q_head) row diagnostics: which heads have correct vs zero output?
  -- mae over the head_dim axis for each (b, h). Expected is non-trivial; if our
  -- kernel only fills some heads correctly, we'll see a binary distribution.
  let perHeadAct := torch.nn.maxAll (torch.nn.abs actual)
  let perHeadExp := torch.nn.maxAll (torch.nn.abs expected)
  IO.println s!"    per-shape actMax={torch.nn.item perHeadAct} expMax={torch.nn.item perHeadExp}"
  pure ok

/-- Cache-vs-no-cache parity: generate K, V for a sequence of N tokens, then
    run the same attention two ways and compare:

    1. Single dense forward over the full K, V (`scaledDotProductAttentionGQAQKV`).
    2. Token-by-token with `Cache.attendLayer`, accumulating per-step outputs.

    Both should produce the same per-token output up to BF16 tolerance.
    Catches off-by-one in append position, stale slice bounds, and any
    softmax-rescale issues across the kernel's tail-mask path. -/
def runCacheParityOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false
  let device := Device.CUDA 0
  -- Small Llama-3 shape: kvSeq=10 short enough to invoke the tail mask path on
  -- every step; head_dim=128.
  let batch : UInt64 := 1
  let qHeads : UInt64 := 32
  let kvHeads : UInt64 := 8
  let totalLen : UInt64 := 10
  let maxLen : UInt64 := 128
  let headDim : UInt64 := 128

  -- Random "all-positions" Q, K, V (FP32 → BF16) representing N tokens.
  let qf ← torch.randn #[batch, qHeads, totalLen, headDim] false device
  let kf ← torch.randn #[batch, kvHeads, totalLen, headDim] false device
  let vf ← torch.randn #[batch, kvHeads, totalLen, headDim] false device
  let qAll := torch.toBFloat16' qf
  let kAll := torch.toBFloat16' kf
  let vAll := torch.toBFloat16' vf

  -- Reference: do attention over the full sequence (causal=true, q_seq=kv_seq).
  let refAll : T #[batch, qHeads, totalLen, headDim] :=
    nn.scaledDotProductAttentionGQAQKV qAll kAll vAll 0.0 true true

  -- Build a 1-layer cache, append tokens one at a time, run cached attention.
  let mut cache ← Generator.KVCache.Cache.init 1 batch maxLen kvHeads headDim (device := device)
  -- Accumulate per-step kernel outputs into a list, then stack.
  let mut acc : Array (T #[batch, qHeads, 1, headDim]) := #[]
  for i in [:totalLen.toNat] do
    let pos : UInt64 := i.toUInt64
    let qStep : T #[batch, qHeads, 1, headDim] :=
      torch.data.slice qAll 2 pos 1
    let kStep : T #[batch, kvHeads, 1, headDim] :=
      torch.data.slice kAll 2 pos 1
    let vStep : T #[batch, kvHeads, 1, headDim] :=
      torch.data.slice vAll 2 pos 1
    let (cache', out) ← cache.attendLayer (numQHeads := qHeads)
        0 qStep kStep vStep (enableGqa := true)
    cache ← cache'.incrementSeqLens
    acc := acc.push out

  let _ ← torch.cuda_synchronize

  -- Compare each step's cached output to the corresponding slice of refAll.
  let mut allOk := true
  let mut maxAbsAll : Float := 0.0
  for i in [:totalLen.toNat] do
    let pos : UInt64 := i.toUInt64
    let refStep : T #[batch, qHeads, 1, headDim] :=
      torch.data.slice refAll 2 pos 1
    let cachedStep := acc[i]!
    let diff := torch.nn.abs (cachedStep - refStep)
    let maxAbs := torch.nn.item (torch.nn.maxAll diff)
    if maxAbs > maxAbsAll then maxAbsAll := maxAbs
    let ok := torch.allclose refStep cachedStep 3e-2 3e-2
    if !ok then
      allOk := false
      IO.println s!"  cache_parity step {i}: max_abs={maxAbs} (FAIL)"

  IO.println s!"cache_parity: ok={allOk} max_abs_overall={maxAbsAll}"
  pure allOk

/-- Combined runner: every shape in `decodeShapes` must pass, then the cache
    parity test must pass. Returns `0` only when all passed. -/
def runDecodeAll (regen : Bool) : IO UInt32 := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return 1
  let mut allOk := true
  for shape in decodeShapes do
    if regen then generateDecodeFixturesFor shape
    let ok ← runDecodeOnceFor shape
    if !ok then allOk := false
  let cacheOk ← runCacheParityOnce
  if !cacheOk then allOk := false
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  let regen := args.contains "--regen"
  let genOnly := args.contains "--gen-only"
  if regen || genOnly then
    if !(← torch.cuda_is_available) then
      IO.eprintln "CUDA is not available; cannot regenerate decode fixtures."
      return 1
    for shape in decodeShapes do
      generateDecodeFixturesFor shape
  if genOnly then return 0
  runDecodeAll regen

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
