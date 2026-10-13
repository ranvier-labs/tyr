/-
  Tyr/Model/Laguna/MoE.lean

  Laguna sparse MoE block for poolside Laguna-S-2.1 (NVFP4 variant):
  sigmoid top-k router (aux-loss-free bias supported), NVFP4-packed expert
  banks dequantized on the fly, and a dense shared expert.

  Mirrors the HuggingFace reference math in
  `dev/laguna_reference/modeling_laguna.py`
  (`LagunaTopKRouter`, `LagunaExperts`, `LagunaSparseMoeBlock`, `LagunaMLP`):

    logits           = x @ routerWeightᵀ                (FP32; see note below)
    scores           = sigmoid(logits)                  (optional tanh softcap)
    selection        = scores + eScoreCorrectionBias    (bias only for picking)
    idx              = topk(selection, k)
    weights          = gather(scores, idx)              (UNbiased)
    weights          = weights / Σ weights              (norm_topk_prob)
    out              = Σ_k weights_k · expert_{idx_k}(x)   (weights on OUTPUT)
    out              = moe_routed_scaling_factor · out + sharedExpert(x)
    expert(x)        = down(silu(gate x) * up x)        (SwiGLU, BF16 matmuls)

  Deviation from the HF reference, by design: HF computes
  `F.linear(hidden, weight).float()` (BF16 matmul, then cast), while this
  implementation follows the task spec and computes the router matmul in FP32
  (cast first, then matmul). The difference is below BF16 output rounding and
  is covered by the test tolerance.

  Token-dispatch strategy: per-unique-expert gather (HF `LagunaExperts`
  style). For each expert hit by the batch we gather its token rows, dequantize
  that expert's three NVFP4 matrices ONCE per forward (regardless of token
  count), run the SwiGLU in BF16, weight the outputs, and scatter-add them into
  the accumulator. Iteration is in ascending expert index order, matching the
  HF `expert_hit` enumeration, so BF16 accumulation order (and thus rounding)
  matches the reference as closely as the op set allows.

  Dense-BF16 expert banks: BF16 checkpoints (e.g. the tiny parity fixture)
  ship experts as fused dense 3-D banks (`mlp.experts.gate_up_proj` /
  `mlp.experts.down_proj`) instead of per-expert NVFP4-packed tensors. Those
  fit `LagunaDenseExperts` (set on `LagunaSparseMoeBlock.denseExperts` by the
  checkpoint loader); `forward2d` then slices the dense banks in the same
  ascending-expert dispatch order and skips NVFP4 dequantization entirely.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Derive
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.NvFp4
import Tyr.Model.Laguna.Fused

/-! # Laguna mixture of experts

Top-$k$ sigmoid router `torch.laguna.LagunaTopKRouter` (aux-loss-free load
balancing), dense and NVFP4-packed expert banks
(`torch.laguna.LagunaDenseExperts`, `torch.laguna.LagunaPackedExperts`) and the
combined `torch.laguna.LagunaSparseMoeBlock`.
-/

namespace torch.laguna

/-- Top-k sigmoid router for Laguna MoE (aux-loss-free load balancing). -/
structure LagunaTopKRouter (cfg : Config) where
  /-- Router projection `[num_experts, hidden_size]` (BF16 in the checkpoint). -/
  weight : T #[cfg.num_experts, cfg.hidden_size]
  /-- Optional per-expert selection bias (added to selection scores only). -/
  eScoreCorrectionBias : Option (T #[cfg.num_experts]) := none
  deriving TensorStruct

namespace LagunaTopKRouter

/-- Route tokens to experts.
    Returns `(selectedExperts, routingWeights)`, both `[tokens, num_experts_per_tok]`;
    `selectedExperts` is int64, `routingWeights` has `x`'s dtype. -/
def route {tokens : UInt64}
    (cfg : Config)
    (r : LagunaTopKRouter cfg)
    (x : T #[tokens, cfg.hidden_size])
    : IO (T #[tokens, cfg.num_experts_per_tok] × T #[tokens, cfg.num_experts_per_tok]) := do
  -- FP32 router logits: x @ weightᵀ.
  let logits0 : T #[] := torch.einsum2 "eh,th->te" (toFloat' r.weight) (toFloat' x)
  -- Optional logit softcapping (0.0 in Laguna-S-2.1, so normally skipped).
  let logits : T #[] :=
    if cfg.moe_router_logit_softcapping > 0.0 then
      let c := cfg.moe_router_logit_softcapping
      mul_scalar (nn.tanh (div_scalar logits0 c)) c
    else
      logits0
  let scores : T #[] := nn.sigmoid logits
  -- Bias affects expert SELECTION only; routing weights stay unbiased.
  let selectionScores : T #[] :=
    match r.eScoreCorrectionBias with
    | some bias => add scores (toFloat' bias)
    | none => scores
  let (_selVals, topIdx) := torch.topk_2d (reshape selectionScores #[tokens, cfg.num_experts]) cfg.num_experts_per_tok 1
  let weights : T #[] := gather scores 1 topIdx
  let weights : T #[] :=
    if cfg.norm_topk_prob then
      nn.div weights (nn.sumDim weights 1 true)
    else
      weights
  let weightsX : T #[] := castLike x weights
  pure (
    reshape topIdx #[tokens, cfg.num_experts_per_tok],
    reshape weightsX #[tokens, cfg.num_experts_per_tok])

end LagunaTopKRouter

/-- NVFP4-packed expert weight banks for Laguna MoE (shape-erased device tensors).
    Runtime shapes:
    - gate/up: packed U8 `[E, moe_intermediate, hidden/2]`,
      scales F8_E4M3 `[E, moe_intermediate, hidden/16]`
    - down: packed U8 `[E, hidden, moe_intermediate/2]`,
      scales F8_E4M3 `[E, hidden, moe_intermediate/16]`
    - globals: F32 `[E]` -/
structure LagunaPackedExperts (cfg : Config) where
  gatePacked : T #[]
  gateScale : T #[]
  gateGlobal : T #[]
  upPacked : T #[]
  upScale : T #[]
  upGlobal : T #[]
  downPacked : T #[]
  downScale : T #[]
  downGlobal : T #[]
  deriving TensorStruct

/-- Dense BF16 expert weight banks for Laguna MoE (shape-erased device
    tensors). Used for BF16 checkpoints that store experts as fused 3-D banks
    instead of NVFP4-packed per-expert tensors. Runtime shapes:
    - gateProj/upProj: BF16 `[E, moe_intermediate, hidden]`
    - downProj: BF16 `[E, hidden, moe_intermediate]` -/
structure LagunaDenseExperts (cfg : Config) where
  gateProj : T #[]
  upProj : T #[]
  downProj : T #[]
  deriving TensorStruct

/-- Sparse MoE block: sigmoid-routed packed experts + dense shared expert. -/
structure LagunaSparseMoeBlock (cfg : Config) where
  router : LagunaTopKRouter cfg
  experts : LagunaPackedExperts cfg
  /-- Optional dense BF16 expert banks. When set, `forward2d` uses these
      instead of the NVFP4-packed `experts` banks (which may be unused
      placeholders); the per-expert dispatch order is unchanged. -/
  denseExperts : Option (LagunaDenseExperts cfg) := none
  sharedGateProj : T #[cfg.shared_expert_intermediate_size, cfg.hidden_size]
  sharedUpProj : T #[cfg.shared_expert_intermediate_size, cfg.hidden_size]
  sharedDownProj : T #[cfg.hidden_size, cfg.shared_expert_intermediate_size]
  deriving TensorStruct

namespace LagunaSparseMoeBlock

/-- Dequantize one expert's matrix from a packed bank slice.
    `bankPacked`/`bankScales` are the `[E, out, in/*]` banks, `bankGlobal` the `[E]`
    global scales; `e` selects the expert row.

    NOTE: the banks are reshaped to typed 3D/1D shapes before `data.slice`:
    `data.slice`'s result type evaluates `sliceShape` at runtime, which panics
    (`Array.set!` out of bounds) when the input shape is fully erased (`#[]`). -/
private def dequantExpertSlice
    (bankPacked bankScales bankGlobal : T #[])
    (numExperts e : UInt64) (outFeatures inFeatures : UInt64)
    : IO (T #[]) := do
  let packed3d : T #[numExperts, outFeatures, inFeatures / 2] :=
    reshape bankPacked #[numExperts, outFeatures, inFeatures / 2]
  let scales3d : T #[numExperts, outFeatures, inFeatures / 16] :=
    reshape bankScales #[numExperts, outFeatures, inFeatures / 16]
  let globals1d : T #[numExperts] := reshape bankGlobal #[numExperts]
  nvfp4.dequantMatrix
    (reshape (data.slice packed3d 0 e 1) #[outFeatures, inFeatures / 2])
    (reshape (data.slice scales3d 0 e 1) #[outFeatures, inFeatures / 16])
    (reshape (data.slice globals1d 0 e 1) #[1])
    outFeatures inFeatures

/-- Fetch one expert's three SwiGLU matrices in BF16: slices of the dense
    banks when `denseExperts` is set, otherwise NVFP4 dequant of the packed
    bank rows. -/
private def expertWeights
    (cfg : Config)
    (m : LagunaSparseMoeBlock cfg)
    (e : UInt64)
    : IO (T #[cfg.moe_intermediate_size, cfg.hidden_size]
        × T #[cfg.moe_intermediate_size, cfg.hidden_size]
        × T #[cfg.hidden_size, cfg.moe_intermediate_size]) := do
  match m.denseExperts with
  | some de =>
    -- NOTE: reshape the erased banks to typed 3-D shapes before `data.slice`
    -- (see `dequantExpertSlice` for why).
    let gate3 : T #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size] :=
      reshape de.gateProj #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size]
    let up3 : T #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size] :=
      reshape de.upProj #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size]
    let down3 : T #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] :=
      reshape de.downProj #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]
    pure (
      reshape (data.slice gate3 0 e 1) #[cfg.moe_intermediate_size, cfg.hidden_size],
      reshape (data.slice up3 0 e 1) #[cfg.moe_intermediate_size, cfg.hidden_size],
      reshape (data.slice down3 0 e 1) #[cfg.hidden_size, cfg.moe_intermediate_size])
  | none =>
    let gateW : T #[cfg.moe_intermediate_size, cfg.hidden_size] :=
      reshape (← dequantExpertSlice m.experts.gatePacked m.experts.gateScale m.experts.gateGlobal
        cfg.num_experts e cfg.moe_intermediate_size cfg.hidden_size) #[cfg.moe_intermediate_size, cfg.hidden_size]
    let upW : T #[cfg.moe_intermediate_size, cfg.hidden_size] :=
      reshape (← dequantExpertSlice m.experts.upPacked m.experts.upScale m.experts.upGlobal
        cfg.num_experts e cfg.moe_intermediate_size cfg.hidden_size) #[cfg.moe_intermediate_size, cfg.hidden_size]
    let downW : T #[cfg.hidden_size, cfg.moe_intermediate_size] :=
      reshape (← dequantExpertSlice m.experts.downPacked m.experts.downScale m.experts.downGlobal
        cfg.num_experts e cfg.hidden_size cfg.moe_intermediate_size) #[cfg.hidden_size, cfg.moe_intermediate_size]
    pure (gateW, upW, downW)

/-- Batched single-token decode path. For one token the k router-selected
    experts are distinct, so the k expert triples can be gathered and
    dequantized as one bank and the SwiGLU runs as batched GEMVs (`bmm`) —
    ~15 kernel launches per layer instead of ~15 per expert. Same math as the
    general path (weights on expert output, scaling factor, + shared expert);
    only the reduction order over k differs (BF16 tolerance).

    On CUDA with BF16 input and NVFP4-packed banks (`denseExperts = none`),
    the routed sum is computed by the fused kernel in
    `Tyr/Model/Laguna/Fused.lean` (one streaming pass over the packed
    weights, no BF16 materialization) unless `useFused` is false; everything
    else keeps the eager gather+dequant+bmm path below. -/
private def forwardDecode1
    (cfg : Config)
    (m : LagunaSparseMoeBlock cfg)
    (x : T #[1, cfg.hidden_size])
    (flatIdx flatW : T #[])
    (useFused : Bool := true)
    : IO (T #[1, cfg.hidden_size]) := do
  let k := cfg.num_experts_per_tok
  let E := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size
  let useFusedKernel : Bool :=
    useFused && m.denseExperts.isNone && h % 32 == 0 && mi % 32 == 0 &&
    (match x.device with | .CUDA _ => true | _ => false) &&
    x.dtype == .BFloat16 &&
    -- Bank dtype gate: random-weight test models cast every tensor (packed
    -- banks included) to BF16; only genuinely NVFP4-packed banks take the
    -- fused path, everything else keeps the eager fallback.
    m.experts.gatePacked.dtype == .UInt8 && m.experts.upPacked.dtype == .UInt8 &&
    m.experts.downPacked.dtype == .UInt8
  let acc : T #[] ←
    if useFusedKernel then do
      -- Fused NVFP4 kernel: routed[t] = Σ_p w[t,p] · expert_{idx[t,p]}(x[t]).
      let xE : T #[] := x
      let topIdx2d : T #[] := reshape flatIdx #[1, k]
      let topW2d : T #[] := reshape flatW #[1, k]
      lagunaMoeFp4Forward xE topIdx2d topW2d
        m.experts.gatePacked m.experts.gateScale m.experts.gateGlobal
        m.experts.upPacked m.experts.upScale m.experts.upGlobal
        m.experts.downPacked m.experts.downScale m.experts.downGlobal
        E mi h
    else do
      let hitT : T #[k] := reshape flatIdx #[k]
      let (gateW, upW, downW) ← do
        match m.denseExperts with
        | some de =>
          let g : T #[k, mi, h] := reshape (nn.embedding1d hitT (reshape de.gateProj #[E, mi * h])) #[k, mi, h]
          let u : T #[k, mi, h] := reshape (nn.embedding1d hitT (reshape de.upProj #[E, mi * h])) #[k, mi, h]
          let d : T #[k, h, mi] := reshape (nn.embedding1d hitT (reshape de.downProj #[E, h * mi])) #[k, h, mi]
          pure (g, u, d)
        | none => do
          let gp : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.gatePacked #[E, mi * (h / 2)])) #[k, mi, h / 2]
          let gs : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.gateScale #[E, mi * (h / 16)])) #[k, mi, h / 16]
          let gg : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.gateGlobal #[E, 1])) #[k]
          let up : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.upPacked #[E, mi * (h / 2)])) #[k, mi, h / 2]
          let us : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.upScale #[E, mi * (h / 16)])) #[k, mi, h / 16]
          let ug : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.upGlobal #[E, 1])) #[k]
          let dp : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.downPacked #[E, h * (mi / 2)])) #[k, h, mi / 2]
          let ds : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.downScale #[E, h * (mi / 16)])) #[k, h, mi / 16]
          let dg : T #[] := reshape (nn.embedding1d hitT (reshape m.experts.downGlobal #[E, 1])) #[k]
          let g ← nvfp4.dequantBank gp gs gg k mi h
          let u ← nvfp4.dequantBank up us ug k mi h
          let d ← nvfp4.dequantBank dp ds dg k h mi
          pure (reshape g #[k, mi, h], reshape u #[k, mi, h], reshape d #[k, h, mi])
      -- Batched SwiGLU: [k,1,h] @ Wᵀ per projection.
      let x3 : T #[k, 1, h] := reshape (nn.expand' (reshape x #[1, 1, h]) #[k, 1, h]) #[k, 1, h]
      let g : T #[k, 1, mi] := nn.bmm x3 (nn.transpose3d_12 gateW)
      let u : T #[k, 1, mi] := nn.bmm x3 (nn.transpose3d_12 upW)
      let hid : T #[k, 1, mi] := mul (nn.silu g) u
      let o : T #[k, 1, h] := nn.bmm hid (nn.transpose3d_12 downW)
      let ow : T #[k, 1, h] := reshape (mul' (reshape flatW #[k, 1, 1]) o) #[k, 1, h]
      pure (reshape (nn.sumDim ow 0 false) #[1, h])
  -- Dense shared expert (SwiGLU MLP, no gating score).
  let sharedH : T #[1, cfg.shared_expert_intermediate_size] :=
    mul (nn.silu (linear x m.sharedGateProj)) (linear x m.sharedUpProj)
  let sharedOut : T #[1, h] := linear sharedH m.sharedDownProj
  pure (reshape (add (mul_scalar acc cfg.moe_routed_scaling_factor) sharedOut) #[1, h])

/-- MoE forward over a 2D `[tokens, hidden]` batch.
    Returns `moe_routed_scaling_factor · Σ_k w_k · expert_k(x) + sharedExpert(x)`
    in `x`'s dtype. Single-token decode on CUDA uses the fused NVFP4 kernel
    for the packed-bank path unless `useFused` is false. -/
def forward2d {tokens : UInt64}
    (cfg : Config)
    (m : LagunaSparseMoeBlock cfg)
    (x : T #[tokens, cfg.hidden_size])
    (useFused : Bool := true)
    : IO (T #[tokens, cfg.hidden_size]) := do
  let k := cfg.num_experts_per_tok
  let (topIdx, weights) ← m.router.route cfg x
  -- Flattened dispatch views: pair p = token * k + slot.
  let flatIdx : T #[] := reshape topIdx #[tokens * k]
  let flatW : T #[] := reshape weights #[tokens * k]
  let device := x.device
  if tokens == 1 then
    -- Single-token decode: batched path (k distinct experts, no dispatch scan).
    let x1 : T #[1, cfg.hidden_size] := reshape x #[1, cfg.hidden_size]
    pure (reshape (← forwardDecode1 cfg m x1 flatIdx flatW useFused) #[tokens, cfg.hidden_size])
  else do
    let h := cfg.hidden_size
    let mi := cfg.moe_intermediate_size
    let fusedAvailable : Bool :=
      useFused && m.denseExperts.isNone && h % 32 == 0 && mi % 32 == 0 &&
      (match device with | .CUDA _ => true | _ => false) &&
      x.dtype == .BFloat16 &&
      m.experts.gatePacked.dtype == .UInt8 && m.experts.upPacked.dtype == .UInt8 &&
      m.experts.downPacked.dtype == .UInt8
    if fusedAvailable then do
      -- Multi-token fused path: the kernel grids over pairs = tokens*k.
      let xE : T #[] := x
      let topIdx2d : T #[] := reshape flatIdx #[tokens, k]
      let topW2d : T #[] := reshape flatW #[tokens, k]
      let routed ← lagunaMoeFp4Forward xE topIdx2d topW2d
        m.experts.gatePacked m.experts.gateScale m.experts.gateGlobal
        m.experts.upPacked m.experts.upScale m.experts.upGlobal
        m.experts.downPacked m.experts.downScale m.experts.downGlobal
        cfg.num_experts mi h
      -- Dense shared expert (SwiGLU MLP, no gating score).
      let sharedH : T #[tokens, cfg.shared_expert_intermediate_size] :=
        mul (nn.silu (linear x m.sharedGateProj)) (linear x m.sharedUpProj)
      let sharedOut : T #[tokens, cfg.hidden_size] := linear sharedH m.sharedDownProj
      pure (reshape (add (mul_scalar routed cfg.moe_routed_scaling_factor) sharedOut) #[tokens, cfg.hidden_size])
    else do
      -- Per-pair token row ids: [0..tokens) repeated k times each.
      -- NOTE: `arange` always allocates on CPU; move to the model device so
      -- `masked_select`/`embedding1d` see input and mask/index on one device.
      let tokIdsFlat : T #[] :=
        reshape (nn.expand' (reshape ((arange 0 tokens).to device) #[tokens, 1]) #[tokens, k]) #[tokens * k]
      -- Accumulator in the model dtype (HF: torch.zeros_like(hidden_states)).
      let mut acc : T #[] := castLike x (torch.zeros #[tokens, cfg.hidden_size] false device)
      -- Per-unique-expert dispatch: materialize each hit expert once per forward.
      -- One CPU readback of the (small) dispatch list — <= tokens*k entries —
      -- instead of scanning all num_experts with a GPU mask + host sync per
      -- expert (that is ~256 syncs/layer, fatal at decode). Iteration stays in
      -- ascending expert order, matching the HF `expert_hit` accumulation order.
      let flatIdxHost ← data.tensorToUInt64Array (reshape (data.toLong flatIdx) #[tokens * k])
      let hitExperts : Array UInt64 := Id.run do
        let mut seen : Array UInt64 := #[]
        for id in flatIdxHost do
          if !(seen.contains id) then
            seen := seen.push id
        pure (seen.qsort (· < ·))
      for eU in hitExperts do
        let maskE : T #[] := eq_scalar flatIdx eU.toInt64
        let tokEr : T #[] := nn.masked_select tokIdsFlat maskE
        let wE : T #[] := nn.masked_select flatW maskE
        let nE : UInt64 := (T.runtimeShape tokEr).getD 0 0
        let tokE : T #[nE] := reshape tokEr #[nE]
        let (gateW, upW, downW) ← expertWeights cfg m eU
        -- SwiGLU expert MLP in BF16.
        let xE : T #[nE, cfg.hidden_size] := nn.embedding1d tokE x
        let hid : T #[nE, cfg.moe_intermediate_size] :=
          mul (nn.silu (linear xE gateW)) (linear xE upW)
        let outE : T #[nE, cfg.hidden_size] := linear hid downW
        -- Routing weights multiply the expert OUTPUT (moe_apply_router_weight_on_input=false).
        let outEW : T #[nE, cfg.hidden_size] := mul' (reshape wE #[nE, 1]) outE
        let idxRows : T #[nE, cfg.hidden_size] :=
          reshape (nn.expand' (reshape tokE #[nE, 1]) #[nE, cfg.hidden_size]) #[nE, cfg.hidden_size]
        acc := scatter_add acc 0 idxRows outEW
      -- Dense shared expert (SwiGLU MLP, no gating score).
      let sharedH : T #[tokens, cfg.shared_expert_intermediate_size] :=
        mul (nn.silu (linear x m.sharedGateProj)) (linear x m.sharedUpProj)
      let sharedOut : T #[tokens, cfg.hidden_size] := linear sharedH m.sharedDownProj
      -- HF multiplies the routed output by the scaling factor, then adds the shared expert.
      let routed : T #[] := mul_scalar acc cfg.moe_routed_scaling_factor
      pure (reshape (add routed sharedOut) #[tokens, cfg.hidden_size])

end LagunaSparseMoeBlock

end torch.laguna
