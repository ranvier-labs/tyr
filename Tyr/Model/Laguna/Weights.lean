/-
  Tyr/Model/Laguna/Weights.lean

  Pretrained weight loading for poolside Laguna text causal-LM
  (Laguna-S-2.1) from HuggingFace SafeTensors (single-file and sharded).

  Two checkpoint serializations are supported (detected per MoE layer):

  A. BF16 exports (e.g. `Tests/fixtures/laguna/tiny/model.safetensors`):
     everything is BF16 with standard `modeling_laguna.py` state_dict names,
     and the routed experts are FUSED dense 3-D banks
     `mlp.experts.gate_up_proj` `[E, 2*moeInt, hidden]` (dim 1: first half
     gate, second half up — HF `chunk(2, dim=-1)`) and
     `mlp.experts.down_proj` `[E, hidden, moeInt]`. These load into
     `LagunaDenseExperts` (see MoE.lean).

  B. Real NVFP4 (`poolside/Laguna-S-2.1-NVFP4`, 14 shards):
     non-expert tensors are BF16 with the same standard names; routed experts
     are PER-EXPERT NVFP4-packed
     `mlp.experts.{i}.{gate,up,down}_proj.{weight_packed,weight_scale,weight_global_scale}`
     (U8 `[out, in/2]`, F8_E4M3 `[out, in/16]`, F32 scalar). The 256 experts
     are loaded on CPU, stacked into the nine `[E, ...]` banks of
     `LagunaPackedExperts`, and moved to the target device once per layer
     (layer-by-layer, so peak memory stays ~1 per-layer bank set instead of a
     full-model CPU copy). Experts stay packed — dequantization happens
     per-forward in the MoE block.

  Ignored checkpoint tensors: `self_attn.{k,v}_scale` (BF16 scalars used by
  vLLM KV quantization) and `*.input_global_scale` (activation quantization
  scales) — the tyr runtime consumes neither.

  Router bias naming: transformers `modeling_laguna.py` stores the
  aux-loss-free selection bias as `mlp.gate.e_score_correction_bias`, while
  the vLLM-trained NVFP4 checkpoint stores it as
  `mlp.experts.e_score_correction_bias` (HF remaps it on load). Both names
  are tried; the bias is optional (absent ⇒ no bias).

  Sharded-load performance: the C++ loader (`cc/src/tyr.cpp`
  `ShardedSafeTensorsDir`) parses all shard headers ONCE per directory and
  caches them (it never reads `model.safetensors.index.json`), so each
  per-tensor load below is an O(1) hash lookup plus one file read. The full
  NVFP4 checkpoint is ~145k tensors; loading is dominated by reading ~53 GB
  of packed expert bytes from disk, not by lookup overhead.
-/
import Tyr.Torch
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Laguna.Model

/-! # Laguna weights

`torch.laguna.WeightSource` abstracts one SafeTensors file or a sharded
directory; loaders resolve tolerant candidate names and check element counts
against the expected shape.
-/

namespace torch.laguna

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique)

/-- Candidate names for one checkpoint tensor, tolerant of common
    wrapper prefixes (mirrors the Qwen35 loader). -/
private def nameCandidates (name : String) : Array String :=
  let out : Array String := #[]
  let out := pushUnique out name
  let out := pushUnique out s!"language_model.{name}"
  let out := pushUnique out s!"model.language_model.{name}"
  let out :=
    if name.startsWith "model." then
      let suffix := name.drop 6
      let out := pushUnique out s!"model.language_model.{suffix}"
      pushUnique out s!"language_model.{suffix}"
    else
      out
  out

/-- Checkpoint tensor source: one SafeTensors file or a sharded directory.
    Unifies the single-file/sharded loader pairs so each component loader is
    written once (the Qwen35 loader duplicates them by hand). -/
private inductive WeightSource where
  | single (path : String)
  | sharded (dir : String)

namespace WeightSource

/-- Load `name` with expected shape `s` (numel-checked, reshaped in C++). -/
def load (src : WeightSource) (name : String) (s : Shape)
    (device : Device := Device.CPU) : IO (T s) :=
  match src with
  | .single path => torch.safetensors.loadTensorOnDevice path name s device
  | .sharded dir => torch.safetensors.loadTensorShardedOnDevice dir name s device

/-- Try to load `name`; `none` on any failure (missing tensor, shape
    mismatch, unreadable file). Used for layout probes and optional tensors. -/
def tryLoad (src : WeightSource) (name : String) (s : Shape)
    (device : Device := Device.CPU) : IO (Option (T s)) := do
  try
    pure (some (← src.load name s device))
  catch _ =>
    pure none

/-- Try each of `names` in order; `none` if none loads. -/
def tryLoadCandidates (src : WeightSource) (names : Array String) (s : Shape)
    (device : Device := Device.CPU) : IO (Option (T s)) := do
  for n in names do
    if let some t ← src.tryLoad n s device then
      return some t
  pure none

/-- Load the first of `names` that resolves; throws listing all candidates
    if none loads. -/
def loadCandidates (src : WeightSource) (names : Array String) (s : Shape)
    (device : Device := Device.CPU) : IO (T s) := do
  match (← src.tryLoadCandidates names s device) with
  | some t => pure t
  | none => throw <| IO.userError s!"Failed to load tensor: {names}"

end WeightSource

/-- Load one attention block (`numHeads` is the per-layer query head count).
    `self_attn.{k,v}_scale` scalars are intentionally not loaded. -/
private def loadAttention (src : WeightSource) (cfg : Config) (layerIdx numHeads : UInt64)
    (device : Device) : IO (LagunaAttention cfg numHeads) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let q ← src.loadCandidates (nameCandidates s!"{p}.q_proj.weight")
    #[numHeads * cfg.head_dim, cfg.hidden_size] device
  let k ← src.loadCandidates (nameCandidates s!"{p}.k_proj.weight")
    #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] device
  let v ← src.loadCandidates (nameCandidates s!"{p}.v_proj.weight")
    #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] device
  let o ← src.loadCandidates (nameCandidates s!"{p}.o_proj.weight")
    #[cfg.hidden_size, numHeads * cfg.head_dim] device
  let g ← src.loadCandidates (nameCandidates s!"{p}.g_proj.weight")
    #[numHeads, cfg.hidden_size] device
  let qn ← src.loadCandidates (nameCandidates s!"{p}.q_norm.weight") #[cfg.head_dim] device
  let kn ← src.loadCandidates (nameCandidates s!"{p}.k_norm.weight") #[cfg.head_dim] device
  pure {
    q_proj := reqGradFalse q
    k_proj := reqGradFalse k
    v_proj := reqGradFalse v
    o_proj := reqGradFalse o
    g_proj := reqGradFalse g
    q_norm := reqGradFalse qn
    k_norm := reqGradFalse kn
  }

/-- Load the dense SwiGLU MLP of a `mlp_only_layers` layer. -/
private def loadDenseMLP (src : WeightSource) (cfg : Config) (layerIdx : UInt64)
    (device : Device) : IO (LagunaMLP cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← src.loadCandidates (nameCandidates s!"{p}.gate_proj.weight")
    #[cfg.intermediate_size, cfg.hidden_size] device
  let up ← src.loadCandidates (nameCandidates s!"{p}.up_proj.weight")
    #[cfg.intermediate_size, cfg.hidden_size] device
  let down ← src.loadCandidates (nameCandidates s!"{p}.down_proj.weight")
    #[cfg.hidden_size, cfg.intermediate_size] device
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

/-- Unused placeholder packed banks for the dense-BF16 expert layout
    (`forward2d` reads `denseExperts` instead). -/
private def placeholderPackedExperts (cfg : Config) (device : Device) : LagunaPackedExperts cfg :=
  let z : T #[] := nn.eraseShape (torch.zeros #[1] false device)
  { gatePacked := z, gateScale := z, gateGlobal := z,
    upPacked := z, upScale := z, upGlobal := z,
    downPacked := z, downScale := z, downGlobal := z }

/-- Load and stack the per-expert NVFP4-packed banks of one MoE layer
    (layout B). Expert tensors load on CPU, are stacked into `[E, ...]`
    banks with one `cat` per bank, and the nine banks move to `device` in
    one go — peak extra memory is one layer's banks (~1.4 GB for S-2.1). -/
private def loadPackedExpertBanks (src : WeightSource) (p : String) (cfg : Config)
    (device : Device) : IO (LagunaPackedExperts cfg) := do
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size
  let mut gateP : Array (T #[]) := Array.mkEmpty e.toNat
  let mut gateS : Array (T #[]) := Array.mkEmpty e.toNat
  let mut gateG : Array (T #[]) := Array.mkEmpty e.toNat
  let mut upP : Array (T #[]) := Array.mkEmpty e.toNat
  let mut upS : Array (T #[]) := Array.mkEmpty e.toNat
  let mut upG : Array (T #[]) := Array.mkEmpty e.toNat
  let mut downP : Array (T #[]) := Array.mkEmpty e.toNat
  let mut downS : Array (T #[]) := Array.mkEmpty e.toNat
  let mut downG : Array (T #[]) := Array.mkEmpty e.toNat
  for i in [:e.toNat] do
    let ep := s!"{p}.experts.{i}"
    -- gate_proj / up_proj: packed [mi, h/2], scales [mi, h/16], global scalar.
    let gp ← src.load s!"{ep}.gate_proj.weight_packed" #[mi, h / 2] Device.CPU
    let gs ← src.load s!"{ep}.gate_proj.weight_scale" #[mi, h / 16] Device.CPU
    let gg ← src.load s!"{ep}.gate_proj.weight_global_scale" #[1] Device.CPU
    let up ← src.load s!"{ep}.up_proj.weight_packed" #[mi, h / 2] Device.CPU
    let us ← src.load s!"{ep}.up_proj.weight_scale" #[mi, h / 16] Device.CPU
    let ug ← src.load s!"{ep}.up_proj.weight_global_scale" #[1] Device.CPU
    -- down_proj: packed [h, mi/2], scales [h, mi/16], global scalar.
    let dp ← src.load s!"{ep}.down_proj.weight_packed" #[h, mi / 2] Device.CPU
    let ds ← src.load s!"{ep}.down_proj.weight_scale" #[h, mi / 16] Device.CPU
    let dg ← src.load s!"{ep}.down_proj.weight_global_scale" #[1] Device.CPU
    gateP := gateP.push (nn.eraseShape (reshape gp #[1, mi, h / 2]))
    gateS := gateS.push (nn.eraseShape (reshape gs #[1, mi, h / 16]))
    gateG := gateG.push (nn.eraseShape gg)
    upP := upP.push (nn.eraseShape (reshape up #[1, mi, h / 2]))
    upS := upS.push (nn.eraseShape (reshape us #[1, mi, h / 16]))
    upG := upG.push (nn.eraseShape ug)
    downP := downP.push (nn.eraseShape (reshape dp #[1, h, mi / 2]))
    downS := downS.push (nn.eraseShape (reshape ds #[1, h, mi / 16]))
    downG := downG.push (nn.eraseShape dg)
  pure {
    gatePacked := reqGradFalse ((nn.cat_dyn gateP 0).to device)
    gateScale := reqGradFalse ((nn.cat_dyn gateS 0).to device)
    gateGlobal := reqGradFalse ((nn.cat_dyn gateG 0).to device)
    upPacked := reqGradFalse ((nn.cat_dyn upP 0).to device)
    upScale := reqGradFalse ((nn.cat_dyn upS 0).to device)
    upGlobal := reqGradFalse ((nn.cat_dyn upG 0).to device)
    downPacked := reqGradFalse ((nn.cat_dyn downP 0).to device)
    downScale := reqGradFalse ((nn.cat_dyn downS 0).to device)
    downGlobal := reqGradFalse ((nn.cat_dyn downG 0).to device)
  }

/-- Load one sparse MoE block, auto-detecting the expert serialization:
    fused dense BF16 banks (layout A) when `mlp.experts.gate_up_proj`
    resolves, otherwise per-expert NVFP4-packed tensors (layout B). -/
private def loadSparseMoe (src : WeightSource) (cfg : Config) (layerIdx : UInt64)
    (device : Device) : IO (LagunaSparseMoeBlock cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let e := cfg.num_experts
  let mi := cfg.moe_intermediate_size
  let h := cfg.hidden_size

  let routerW ← src.loadCandidates (nameCandidates s!"{p}.gate.weight") #[e, h] device
  -- Optional selection bias; transformers and vLLM-trained checkpoints use
  -- different keys (see module doc).
  let biasOpt ← src.tryLoadCandidates
    (nameCandidates s!"{p}.gate.e_score_correction_bias"
      ++ nameCandidates s!"{p}.experts.e_score_correction_bias")
    #[e] device

  let sharedGate ← src.loadCandidates (nameCandidates s!"{p}.shared_expert.gate_proj.weight")
    #[cfg.shared_expert_intermediate_size, h] device
  let sharedUp ← src.loadCandidates (nameCandidates s!"{p}.shared_expert.up_proj.weight")
    #[cfg.shared_expert_intermediate_size, h] device
  let sharedDown ← src.loadCandidates (nameCandidates s!"{p}.shared_expert.down_proj.weight")
    #[h, cfg.shared_expert_intermediate_size] device

  let guOpt ← src.tryLoadCandidates (nameCandidates s!"{p}.experts.gate_up_proj")
    #[e, 2 * mi, h] device
  let (experts, denseExperts) ←
    match guOpt with
    | some gu => do
      -- Layout A: fused dense banks; dim-1 first half gate, second half up.
      let down ← src.loadCandidates (nameCandidates s!"{p}.experts.down_proj") #[e, h, mi] device
      let gate3 : T #[e, mi, h] := reshape (data.slice gu 1 0 mi) #[e, mi, h]
      let up3 : T #[e, mi, h] := reshape (data.slice gu 1 mi mi) #[e, mi, h]
      let dense : LagunaDenseExperts cfg := {
        gateProj := nn.eraseShape (reqGradFalse gate3)
        upProj := nn.eraseShape (reqGradFalse up3)
        downProj := nn.eraseShape (reqGradFalse down)
      }
      pure (placeholderPackedExperts cfg device, some dense)
    | none => do
      -- Layout B: per-expert NVFP4-packed tensors, stacked into banks.
      let packed ← loadPackedExpertBanks src p cfg device
      pure (packed, none)

  pure {
    router := { weight := reqGradFalse routerW, eScoreCorrectionBias := biasOpt.map reqGradFalse }
    experts := experts
    denseExperts := denseExperts
    sharedGateProj := reqGradFalse sharedGate
    sharedUpProj := reqGradFalse sharedUp
    sharedDownProj := reqGradFalse sharedDown
  }

/-- Load one decoder layer: exactly one attention variant (by
    `Config.layerType`) and one FFN variant (by `Config.isDenseMlpLayer`). -/
private def loadLayer (src : WeightSource) (cfg : Config) (layerIdx : UInt64)
    (device : Device) : IO (LagunaLayer cfg) := do
  let p := s!"model.layers.{layerIdx}"
  let inputNorm ← src.loadCandidates (nameCandidates s!"{p}.input_layernorm.weight")
    #[cfg.hidden_size] device
  let postNorm ← src.loadCandidates (nameCandidates s!"{p}.post_attention_layernorm.weight")
    #[cfg.hidden_size] device

  let attnFull ←
    match cfg.layerType layerIdx with
    | .fullAttention =>
      let a ← loadAttention src cfg layerIdx cfg.num_attention_heads device
      pure (some a)
    | .slidingAttention => pure none
  let attnSliding ←
    match cfg.layerType layerIdx with
    | .slidingAttention =>
      let a ← loadAttention src cfg layerIdx cfg.num_attention_heads_sliding device
      pure (some a)
    | .fullAttention => pure none

  let denseMlp ←
    if cfg.isDenseMlpLayer layerIdx || !cfg.isMoE then
      let m ← loadDenseMLP src cfg layerIdx device
      pure (some m)
    else
      pure none
  let sparseMoe ←
    if cfg.isDenseMlpLayer layerIdx || !cfg.isMoE then
      pure none
    else
      let m ← loadSparseMoe src cfg layerIdx device
      pure (some m)

  pure {
    input_layernorm := reqGradFalse inputNorm
    attnFull := attnFull
    attnSliding := attnSliding
    post_attention_layernorm := reqGradFalse postNorm
    denseMlp := denseMlp
    sparseMoe := sparseMoe
  }

/-- Shared loader core over an abstract weight source. -/
private def loadCore (src : WeightSource) (desc : String) (cfg : Config)
    (device : Device) (log : Handlers) : IO (LagunaForCausalLM cfg) := do
  log.onInfo s!"Loading LagunaForCausalLM from {desc}..."

  let embedTokens ← src.loadCandidates (nameCandidates "model.embed_tokens.weight")
    #[cfg.vocab_size, cfg.hidden_size] device

  let mut layers : Array (LagunaLayer cfg) := Array.mkEmpty cfg.num_hidden_layers.toNat
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayer src cfg i.toUInt64 device
    layers := layers.push layer
    if (i + 1) % 4 == 0 || i + 1 == cfg.num_hidden_layers.toNat then
      log.onInfo s!"  loaded layers {i + 1}/{cfg.num_hidden_layers.toNat}"

  let normW ← src.loadCandidates (nameCandidates "model.norm.weight") #[cfg.hidden_size] device

  let model : LagunaModel cfg := {
    embed_tokens := reqGradFalse embedTokens
    layers := layers
    norm := reqGradFalse normW
  }

  let lmHeadOpt ← src.tryLoadCandidates (nameCandidates "lm_head.weight")
    #[cfg.vocab_size, cfg.hidden_size] device
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      log.onInfo "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  log.onInfo "Loaded LagunaForCausalLM weights."
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

namespace LagunaForCausalLM

/-- Load a Laguna model from a sharded HF SafeTensors directory.
    Routed experts are loaded on CPU and moved to `device` layer by layer. -/
def loadSharded (modelDir : String) (cfg : Config := Config.laguna_s_2_1)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (LagunaForCausalLM cfg) :=
  loadCore (.sharded modelDir) modelDir cfg device log

/-- Load a Laguna model from a single HF SafeTensors file. -/
def load (path : String) (cfg : Config := Config.laguna_s_2_1)
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (LagunaForCausalLM cfg) :=
  loadCore (.single path) path cfg device log

end LagunaForCausalLM

end torch.laguna
