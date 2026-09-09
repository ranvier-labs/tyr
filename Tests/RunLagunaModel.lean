/-
  Tests/RunLagunaModel.lean

  Validates the full Laguna model (Tyr.Model.Laguna.Model): gated attention
  with interleaved full/sliding-window layers, dense + MoE FFN, KV cache, and
  cached generation.

  Tiny config `lagunaTiny`: hidden 256, head_dim 128, heads 4 full / 6 sliding,
  kv 2, 4 layers [full, sliding, sliding, sliding], vocab 1024, 8 experts
  top-2, moe_intermediate 64, shared_expert_intermediate 64, dense layer-0
  intermediate 512, sliding_window 8, mlp_only_layers #[0], YaRN defaults.

  DTYPE NOTE: the NVFP4 MoE block is a BF16-only pipeline (dequantization
  hardcodes BF16 expert weights; libtorch `linear` requires matching dtypes),
  so every test that touches MoE layers casts the whole model to bf16.
  `denseTiny` (num_experts = 0 → all layers dense SwiGLU) is the fp32 variant
  used for the tight-tolerance parity checks.

  Tests (self-consistency, deterministic seeds, randomly initialized weights):
  (a) KV-cache parity: logits from a single full-sequence forward == logits
      from prefill + step-by-step decode (seq 20 > window 8, so sliding-layer
      decode-time KV truncation is exercised). CPU fp32 dense (tol 2e-4),
      CUDA fp32 dense (tol 1e-2), CPU + CUDA bf16 MoE (tol 0.1 — bf16 has
      ~3 decimal digits; prefill (causal seq SDPA) and decode (q_seq=1 SDPA)
      kernels round differently; random-weight logits are O(1)). The cache
      fixture gives MoE selection scores a guaranteed margin, so rounding
      cannot switch experts. The same dequantized weights also run through
      a tight FP32 MoE cache oracle. Unconstrained routing remains covered
      by LagunaMoeTest and the generation checks below.
  (b) Sliding-window invariance: single-layer sliding model (window 8, fp32,
      dense MLP) — changing input token 0 leaves logits at positions 8..19
      EXACTLY unchanged (window edge: position 7 sees token 0, position 8
      does not), while a full-attention model does change. Changing token 15
      changes position 19 (sanity against degenerate passes). CPU + CUDA.
  (c) Greedy generate (bf16 MoE model): budget termination, EOS termination
      (forced EOS), stream callback count, output shapes. CPU + CUDA.

  Build: lake build -R Tests.RunLagunaModel
  Run:   lake -R exe LagunaModelTest
-/
import Tyr.Torch
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.Rope
import Tyr.Model.Laguna.MoE
import Tyr.Model.Laguna.Model

open torch
open torch.Model
open torch.laguna

private def check (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"PASS: {msg}"
  else
    throw (IO.userError s!"FAIL: {msg}")

/-- Tiny test config with MoE layers 1-3 (see module doc). -/
private def lagunaTiny : Config := { LagunaConfig.laguna_s_2_1 with
  vocab_size := 1024
  hidden_size := 256
  intermediate_size := 512
  num_hidden_layers := 4
  num_attention_heads := 4
  num_attention_heads_sliding := 6
  num_key_value_heads := 2
  head_dim := 128
  max_position_embeddings := 256
  sliding_window := 8
  num_experts := 8
  num_experts_per_tok := 2
  moe_intermediate_size := 64
  shared_expert_intermediate_size := 64
  mlp_only_layers := #[0]
  layer_types := #[.fullAttention, .slidingAttention, .slidingAttention, .slidingAttention] }

/-- Same shape as `lagunaTiny` but all-dense FFN (num_experts = 0) — the
    fp32-friendly variant for tight-tolerance parity checks. -/
private def denseTiny : Config := { lagunaTiny with
  num_experts := 0 }

/-- Single sliding-window layer, dense MLP (no MoE → fully deterministic). -/
private def cfgSlidingOnly : Config := { lagunaTiny with
  num_hidden_layers := 1
  layer_types := #[.slidingAttention]
  mlp_only_layers := #[0] }

/-- Single full-attention layer, dense MLP. -/
private def cfgFullOnly : Config := { lagunaTiny with
  num_hidden_layers := 1
  layer_types := #[.fullAttention]
  mlp_only_layers := #[0] }

private def deviceLabel : Device → String
  | .CPU => "cpu"
  | .CUDA i => s!"cuda:{i}"
  | .MPS => "mps"

/-- Move all tensors of a model to `device`, optionally casting to bf16. -/
private def moveModel {α : Type} [TensorStruct α] (m : α) (device : Device) (bf16 : Bool) : α :=
  TensorStruct.map (fun t => (if bf16 then toBFloat16' t else t).to device) m

/-- Validate before aggregating: NaN must not disappear in `if d > max`. -/
private def maxAbsDiff (a b : T #[]) : IO Float := do
  if a.runtimeShape != b.runtimeShape then
    throw <| IO.userError "Tensor comparison runtime shape mismatch"
  let diff := nn.item (nn.maxAll (nn.abs (sub (toFloat' a) (toFloat' b))))
  if !diff.isFinite then
    throw <| IO.userError "Non-finite tensor comparison"
  pure diff

private def checkComparisonFailures : IO Unit := do
  for invalid in #[Float.ofBits 0x7ff8000000000000, Float.ofBits 0x7ff0000000000000] do
    let rejected ← try
      let _ ← maxAbsDiff (full #[1] invalid) (zeros #[1])
      pure false
    catch _ => pure true
    check rejected "comparison rejects NaN/infinity before max aggregation"
  let rejected ← try
    let _ ← maxAbsDiff (zeros #[1]) (zeros #[2])
    pure false
  catch _ => pure true
  check rejected "comparison rejects broadcastable shape mismatch"

/-- Deterministic int64 token-id tensor `[1, seq]` on `device`. -/
private def mkIds (vals : Array Int64) (device : Device) : T #[1, vals.size.toUInt64] :=
  reshape ((data.fromInt64Array vals).to device) #[1, vals.size.toUInt64]

/-- Base 20-token prompt (deterministic). -/
private def baseIds : Array Int64 :=
  Array.ofFn (n := 20) fun i => Int64.ofNat ((i.val * 37 + 11) % 1024)

/-! ## (a) KV-cache parity -/

/-- Sigmoid router scores lie in [0,1]. Biases spaced by 2 guarantee the
selected set even when differently sized BF16 matmuls perturb the scores.
Only selection is stabilized: the distinct expert weights and input-dependent
routing weights still participate in every cache comparison. -/
private def withStableRoutes (cfg : Config) (m : LagunaForCausalLM cfg) : LagunaForCausalLM cfg :=
  { m with model := { m.model with layers := m.model.layers.map fun layer =>
      { layer with sparseMoe := layer.sparseMoe.map fun moe =>
          let bias : T #[cfg.num_experts] := reshape
            (mul_scalar (toFloat' ((arange 0 cfg.num_experts).to moe.router.weight.device)) 2.0)
            #[cfg.num_experts]
          { moe with router := { moe.router with eScoreCorrectionBias := some bias } } } } }

/-- Materialize the original NVFP4 weights once, preserving their BF16
rounding, then upcast the entire model. The oracle exercises the same MoE
weights and routing, including the single-token expert dispatch path. -/
private def floatReference (cfg : Config) (m : LagunaForCausalLM cfg) : IO (LagunaForCausalLM cfg) := do
  let layers ← m.model.layers.mapM fun layer => do
    let sparseMoe ← layer.sparseMoe.mapM fun moe => do
      let p := moe.experts
      let gateProj ← nvfp4.dequantBank p.gatePacked p.gateScale p.gateGlobal
        cfg.num_experts cfg.moe_intermediate_size cfg.hidden_size
      let upProj ← nvfp4.dequantBank p.upPacked p.upScale p.upGlobal
        cfg.num_experts cfg.moe_intermediate_size cfg.hidden_size
      let downProj ← nvfp4.dequantBank p.downPacked p.downScale p.downGlobal
        cfg.num_experts cfg.hidden_size cfg.moe_intermediate_size
      pure { moe with denseExperts := some { gateProj, upProj, downProj } }
    pure { layer with sparseMoe }
  pure (TensorStruct.map (fun t => toFloat' t) { m with model := { m.model with layers } })

private def runCacheParityModel (cfg : Config) (device : Device) (lbl : String)
    (model : LagunaForCausalLM cfg) (tol : Float) : IO Unit := do
  let seq : UInt64 := 20
  let ids : T #[1, seq] := mkIds baseIds device

  -- Single full-sequence forward (uncached path).
  let logitsAll ← model.forward cfg ids

  let tables ← precomputeRotaryTables cfg seq device

  -- Cached prefill over the SAME 20 tokens: all positions must match.
  let cache0 := LagunaModel.initCache (batch := 1) cfg model.model seq device
  let (hiddenPre, _) ← model.model.forwardWithCache cfg (model.embedTokens ids) tables cache0
  let logitsPre : T #[1, seq, 1024] := linear3d hiddenPre model.lmHead
  let dPrefill ← maxAbsDiff (nn.eraseShape logitsPre) (nn.eraseShape logitsAll)

  -- Prefill 12 tokens, then decode positions 12..19 one token at a time.
  let prefillLen : UInt64 := 12
  let idsPre : T #[1, prefillLen] := data.slice ids 1 0 prefillLen
  let cacheP0 := LagunaModel.initCache (batch := 1) cfg model.model seq device
  let (hiddenP, cacheP1) ←
    model.model.forwardWithCache cfg (model.embedTokens idsPre) tables cacheP0
  let logitsP : T #[1, prefillLen, 1024] := linear3d hiddenP model.lmHead
  let logitsPRef : T #[1, prefillLen, 1024] := data.slice logitsAll 1 0 prefillLen
  let dPrefillPart ← maxAbsDiff (nn.eraseShape logitsP) (nn.eraseShape logitsPRef)
  -- This isolates cache writes from batch-size-dependent arithmetic: both
  -- forwards consume exactly the same prefix with the same operation shapes.
  let logitsUncachedPrefix ← model.forward cfg idsPre
  let dSamePrefix ← maxAbsDiff (nn.eraseShape logitsP) (nn.eraseShape logitsUncachedPrefix)
  check (dSamePrefix == 0.0) s!"(a) [{lbl}] same-prefix prefill is exactly equal to uncached forward"

  let mut cache := cacheP1
  let mut dStep : Float := 0.0
  for pos in [prefillLen.toNat : seq.toNat] do
    let tok : T #[1, 1] := data.slice ids 1 pos.toUInt64 1
    let (hiddenStep, cache') ←
      model.model.forwardStep cfg (model.embedTokens tok) tables pos.toUInt64 cache
    cache := cache'
    let logitsStep : T #[1, 1, 1024] := linear3d hiddenStep model.lmHead
    let logitsRef : T #[1, 1, 1024] := data.slice logitsAll 1 pos.toUInt64 1
    let d ← maxAbsDiff (nn.eraseShape logitsStep) (nn.eraseShape logitsRef)
    if d > dStep then dStep := d

  IO.println s!"  [{lbl}] prefill-20 maxAbs={dPrefill}  prefill-12 maxAbs={dPrefillPart}  decode maxAbs={dStep}  (tol={tol})"
  check (dPrefill ≤ tol) s!"(a) [{lbl}] prefill-20 logits match full forward (maxAbs={dPrefill})"
  check (dPrefillPart ≤ tol) s!"(a) [{lbl}] prefill-12 logits match full forward (maxAbs={dPrefillPart})"
  check (dStep ≤ tol) s!"(a) [{lbl}] decode-step logits match full forward (maxAbs={dStep})"

private def runCacheParity (cfg : Config) (device : Device) (bf16 : Bool) (tol : Float) : IO Unit := do
  let lbl := s!"{deviceLabel device}{(if bf16 then "/bf16" else "/fp32")}"
  torch.manualSeed 1234
  let model := withStableRoutes cfg (moveModel (← LagunaForCausalLM.init cfg) device bf16)
  if bf16 then
    let reference ← floatReference cfg model
    let referenceTol := match device with | .CPU => 2e-4 | _ => 1e-2
    runCacheParityModel cfg device s!"{deviceLabel device}/fp32-moe-oracle" reference referenceTol
  runCacheParityModel cfg device lbl model tol

/-! ## (b) Sliding-window invariance -/

private def runSlidingWindowCheck (device : Device) : IO Unit := do
  let lbl := deviceLabel device
  torch.manualSeed 99
  let mSlide : LagunaForCausalLM cfgSlidingOnly := moveModel (← LagunaForCausalLM.init cfgSlidingOnly) device false
  torch.manualSeed 199
  let mFull : LagunaForCausalLM cfgFullOnly := moveModel (← LagunaForCausalLM.init cfgFullOnly) device false

  -- idsB: token 0 changed; idsC: token 15 changed.
  let idsB : Array Int64 := baseIds.set 0 (Int64.ofNat ((11 + 500) % 1024))
  let idsC : Array Int64 := baseIds.set 15 (Int64.ofNat (((15 * 37 + 11) + 500) % 1024))
  let idsA' : T #[1, 20] := mkIds baseIds device
  let idsB' : T #[1, 20] := mkIds idsB device
  let idsC' : T #[1, 20] := mkIds idsC device

  let logitsA ← mSlide.forward cfgSlidingOnly idsA'
  let logitsB ← mSlide.forward cfgSlidingOnly idsB'
  let logitsC ← mSlide.forward cfgSlidingOnly idsC'

  -- Window edge: position 7 attends to token 0 (7-0 < 8); position 8 does not.
  let dAt7 ← maxAbsDiff (nn.eraseShape (data.slice logitsA 1 7 1)) (nn.eraseShape (data.slice logitsB 1 7 1))
  let dAt8 ← maxAbsDiff (nn.eraseShape (data.slice logitsA 1 8 1)) (nn.eraseShape (data.slice logitsB 1 8 1))
  let dAt19 ← maxAbsDiff (nn.eraseShape (data.slice logitsA 1 19 1)) (nn.eraseShape (data.slice logitsB 1 19 1))
  let dInside ← maxAbsDiff (nn.eraseShape (data.slice logitsA 1 19 1)) (nn.eraseShape (data.slice logitsC 1 19 1))
  IO.println s!"  [{lbl}] sliding 1L: Δ@7(tok0)={dAt7}  Δ@8(tok0)={dAt8}  Δ@19(tok0)={dAt19}  Δ@19(tok15)={dInside}"
  check (dAt7 > 1e-3) s!"(b) [{lbl}] token 0 affects position 7 (inside window, Δ={dAt7})"
  check (dAt8 ≤ 1e-6) s!"(b) [{lbl}] token 0 does NOT affect position 8 (window edge, Δ={dAt8})"
  check (dAt19 ≤ 1e-6) s!"(b) [{lbl}] token 0 does NOT affect position 19 (outside window, Δ={dAt19})"
  check (dInside > 1e-3) s!"(b) [{lbl}] token 15 affects position 19 (inside window, Δ={dInside})"

  -- Full-attention control: token 0 DOES change position 19.
  let logitsFA ← mFull.forward cfgFullOnly idsA'
  let logitsFB ← mFull.forward cfgFullOnly idsB'
  let dFull ← maxAbsDiff (nn.eraseShape (data.slice logitsFA 1 19 1)) (nn.eraseShape (data.slice logitsFB 1 19 1))
  IO.println s!"  [{lbl}] full 1L: Δ@19(tok0)={dFull}"
  check (dFull > 1e-3) s!"(b) [{lbl}] full-attention layer: token 0 affects position 19 (Δ={dFull})"

/-! ## (c) Generation -/

private def runGenerateCheck (device : Device) : IO Unit := do
  let lbl := deviceLabel device
  torch.manualSeed 77
  let model : LagunaForCausalLM lagunaTiny := moveModel (← LagunaForCausalLM.init lagunaTiny) device true
  let prompt : Array Int64 := #[100, 200, 300, 400, 500]
  let ids : T #[1, 5] := mkIds prompt device

  -- 1) Budget termination with EOS disabled: exactly 5 + 6 tokens.
  let r1 ← model.generate lagunaTiny ids 6 .greedy #[]
  check (r1.1 == 11) s!"(c) [{lbl}] budget termination: outSeq={r1.1} == 11"

  -- 2) Forced EOS: first greedy token of run 1 as EOS → stops after 1 token.
  let firstCol : T #[1, 1] := data.slice r1.2 1 5 1
  let toks ← data.tensorToUInt64Array' (nn.eraseShape firstCol)
  let t0 := toks.getD 0 0
  let r2 ← model.generate lagunaTiny ids 6 .greedy #[t0]
  check (r2.1 == 6) s!"(c) [{lbl}] EOS termination (eos={t0}): outSeq={r2.1} == 6"

  -- 3) Streaming callback fires once per generated token.
  let counter ← IO.mkRef 0
  let onStep : StreamCallback 1 := fun _ _ => counter.modify (· + 1)
  let r3 ← model.generateStream lagunaTiny ids onStep 6 .greedy #[]
  let ncb ← counter.get
  check (r3.1 == 11) s!"(c) [{lbl}] stream run outSeq={r3.1} == 11"
  check (ncb == 6) s!"(c) [{lbl}] stream callback fired {ncb} == 6 times"

private def expectRejected (action : IO α) (label : String) : IO Unit := do
  let rejected ← try
    let _ ← action
    pure false
  catch _ => pure true
  check rejected label

/-- Covers partial and over-window prefill, repeated ring wrap, independent
forks, preserved functional snapshots, capacity checks, and inference graphs.
BF16 MoE uses stable expert selection and compares identical decode positions
against the functional cache. The 0.1 bound permits BF16 reduction rounding
when ring order permutes paired keys/values; FP32 retains the tight bound. -/
private def runOwnedCacheCheck (device : Device) (bf16 : Bool := false) : IO Unit := do
  let cfg := if bf16 then lagunaTiny else denseTiny
  let lbl := s!"{deviceLabel device}/{if bf16 then "bf16-moe" else "fp32-dense"}"
  let tol := if bf16 then 0.1 else if device == .CPU then 2e-4 else 1e-2
  torch.manualSeed 987
  let model := withStableRoutes cfg (moveModel (← LagunaForCausalLM.init cfg) device bf16)
  let ids : T #[1, 20] := mkIds baseIds device
  let tables ← precomputeRotaryTables cfg 20 device
  for promptLen in #[5, 12] do
    let session ← model.model.initCacheSession (batch := 1) cfg 20 device
    let capacities ← session.capacities
    check (capacities == #[20, 8, 8, 8]) s!"(d) [{lbl}] sliding caches allocate only one window"
    let promptLenIds : T #[1, promptLen] := data.slice ids 1 0 promptLen
    let embeds := model.embedTokens promptLenIds
    let shortTables ← precomputeRotaryTables cfg 1 device
    expectRejected (model.prefillSession cfg shortTables embeds session)
      "session rejects short rotary tables without damaging state"
    let functional0 := model.model.initCache cfg 20 device
    let (expected, functional) ← model.prefill cfg tables embeds functional0
    let actual ← model.prefillSession cfg tables embeds session
    let prefillDiff ← maxAbsDiff (nn.eraseShape actual) (nn.eraseShape expected)
    check (prefillDiff == 0.0) s!"(d) [{lbl}] owned prefill {promptLen} matches functional exactly"
    check (!autograd.has_grad_fn actual) s!"(d) [{lbl}] inference output has no autograd graph"
    expectRejected (model.prefillSession cfg tables embeds session) "session rejects repeated prefill"
    let tok : T #[1, 1] := data.slice ids 1 promptLen 1
    expectRejected (model.decodeSession cfg tables (model.embedTokens tok) (promptLen + 1) session)
      "session rejects skipped positions without damaging state"
    let forked ← session.fork
    let mut functional := functional
    let mut firstLogits : Option (T #[1, cfg.vocab_size]) := none
    let mut maxDiff := 0.0
    -- Keep an actual old cache tensor alive across subsequent functional calls.
    let some oldCache := functional.kvCaches[1]? | throw <| IO.userError "missing sliding cache"
    let oldK := oldCache.kStoreDyn
    let oldKCopy := autograd.clone oldK
    for pos in [promptLen.toNat:20] do
      let tok : T #[1, 1] := data.slice ids 1 pos.toUInt64 1
      let emb := model.embedTokens tok
      let (expected, next) ← model.decodeStep cfg tables emb pos.toUInt64 functional
      functional := next
      let actual ← model.decodeSession cfg tables emb pos.toUInt64 session
      let diff ← maxAbsDiff (nn.eraseShape actual) (nn.eraseShape expected)
      maxDiff := max maxDiff diff
      if pos == promptLen.toNat then firstLogits := some actual
    check ((← maxAbsDiff oldK oldKCopy) == 0.0) "functional decode preserves earlier cache snapshots"
    -- Original ring has wrapped repeatedly. A shallow fork would now read
    -- overwritten prompt slots and disagree at the fork's first decode step.
    let forkLogits ← model.decodeSession cfg tables (model.embedTokens tok) promptLen forked
    let some first := firstLogits | throw <| IO.userError "missing first decode output"
    check ((← maxAbsDiff (nn.eraseShape forkLogits) (nn.eraseShape first)) == 0.0)
      s!"(d) [{lbl}] fork owns independent cache storage after ring wrap"
    expectRejected (model.decodeSession cfg tables (model.embedTokens tok) 20 session)
      "session rejects capacity overflow"
    IO.println s!"  [{lbl}] owned cache promptLen={promptLen} decode maxAbs={maxDiff} (tol={tol}), capacities={capacities}"
    check (maxDiff ≤ tol) s!"(d) [{lbl}] ring decode matches functional cache"
  expectRejected (model.model.initCacheSession (batch := 1) cfg 0 device)
    "session rejects zero capacity"

  -- The first layer successfully writes K/V before this second-layer IO error.
  -- A failure after mutation must invalidate the session rather than expose
  -- partially advanced buffers, and no-grad must restore the caller's mode.
  let some second := model.model.layers[1]? | throw <| IO.userError "missing second test layer"
  let broken := { model with model := { model.model with
    layers := model.model.layers.set! 1 { second with attnFull := none, attnSliding := none } } }
  let prompt : T #[1, 5] := data.slice ids 1 0 5
  let wasEnabled ← autograd.is_grad_enabled
  try
    for enabled in #[true, false] do
      autograd.set_grad_enabled enabled
      let session ← broken.model.initCacheSession (batch := 1) cfg 20 device
      expectRejected (broken.prefillSession cfg tables (broken.embedTokens prompt) session)
        s!"(d) [{lbl}] malformed second layer fails after an earlier cache write"
      check ((← autograd.is_grad_enabled) == enabled)
        s!"(d) [{lbl}] failed inference restores caller gradient mode {enabled}"
      expectRejected session.capacities "partially updated session cannot expose capacities"
      expectRejected session.fork "partially updated session cannot be forked"
  finally
    autograd.set_grad_enabled wasEnabled

/-- Opt-in timing of identical eight-token decode workloads. Allocation and
prefill are outside the timer; CUDA is synchronized at both boundaries. -/
private def runCacheBenchmark (device : Device) : IO Unit := autograd.no_grad do
  let cfg := denseTiny
  torch.manualSeed 987
  let model := moveModel (← LagunaForCausalLM.init cfg) device false
  let ids : T #[1, 20] := mkIds baseIds device
  let prompt : T #[1, 12] := data.slice ids 1 0 12
  let synchronize : IO Unit := if device == .CPU then pure () else torch.cuda_synchronize
  for capacity in #[(128 : UInt64), 8192] do
    let tables ← precomputeRotaryTables cfg capacity device
    let mut functionalMs : Nat := 0
    let mut ownedMs : Nat := 0
    let mut ownedCapacity : UInt64 := 0
    for _ in [:3] do
      let (_, initial) ← model.prefill cfg tables (model.embedTokens prompt)
        (model.model.initCache cfg capacity device)
      let mut cache := initial
      synchronize
      let start ← IO.monoMsNow
      for pos in [12:20] do
        let tok : T #[1, 1] := data.slice ids 1 pos.toUInt64 1
        let (_, next) ← model.decodeStep cfg tables (model.embedTokens tok) pos.toUInt64 cache
        cache := next
      synchronize
      functionalMs := functionalMs + ((← IO.monoMsNow) - start)
      let session ← model.model.initCacheSession (batch := 1) cfg capacity device
      ownedCapacity := (← session.capacities).foldl (· + ·) 0
      let _ ← model.prefillSession cfg tables (model.embedTokens prompt) session
      synchronize
      let start ← IO.monoMsNow
      for pos in [12:20] do
        let tok : T #[1, 1] := data.slice ids 1 pos.toUInt64 1
        let _ ← model.decodeSession cfg tables (model.embedTokens tok) pos.toUInt64 session
      synchronize
      ownedMs := ownedMs + ((← IO.monoMsNow) - start)
    let bytesPerPosition := 2 * cfg.num_key_value_heads * cfg.head_dim * 4
    IO.println s!"CACHE_BENCH device={deviceLabel device} capacity={capacity} tokens=24 functional_ms={functionalMs} owned_ms={ownedMs} functional_cache_bytes={capacity * cfg.num_hidden_layers * bytesPerPosition} owned_cache_bytes={ownedCapacity * bytesPerPosition}"

def main : IO Unit := do
  for name in #["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TYR_LIBTORCH_DIR"] do
    IO.println s!"runtime {name}={(← IO.getEnv name).getD "<unset>"}"
  checkComparisonFailures
  IO.println "-- (a) KV-cache parity: full forward vs prefill + decode"
  -- fp32 all-dense 4-layer model: tight tolerance on CPU, looser on CUDA
  -- (different SDPA kernels between causal prefill and q_seq=1 decode).
  runCacheParity denseTiny Device.CPU false 2e-4
  -- bf16 MoE model on CPU (production dtype; wider documented tolerance).
  runCacheParity lagunaTiny Device.CPU true 0.1
  if ← torch.cuda_is_available then
    runCacheParity denseTiny (Device.CUDA 0) false 1e-2
    runCacheParity lagunaTiny (Device.CUDA 0) true 0.1
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA (a) cases."

  IO.println "-- (b) sliding-window invariance (single-layer fp32 dense models)"
  runSlidingWindowCheck Device.CPU
  if ← torch.cuda_is_available then
    runSlidingWindowCheck (Device.CUDA 0)
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA (b) cases."

  IO.println "-- (c) cached generation (bf16 MoE model)"
  runGenerateCheck Device.CPU
  if ← torch.cuda_is_available then
    runGenerateCheck (Device.CUDA 0)
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA (c) cases."

  IO.println "-- (d) owned cache sessions and ring storage"
  runOwnedCacheCheck Device.CPU
  runOwnedCacheCheck Device.CPU true
  if ← torch.cuda_is_available then
    runOwnedCacheCheck (Device.CUDA 0)
    runOwnedCacheCheck (Device.CUDA 0) true
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA (d) cases."

  if (← IO.getEnv "TYR_LAGUNA_CACHE_BENCH") == some "1" then
    runCacheBenchmark Device.CPU
    if ← torch.cuda_is_available then runCacheBenchmark (Device.CUDA 0)
  IO.println "All Laguna model tests passed."
