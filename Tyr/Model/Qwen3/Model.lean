/-
  Tyr/Model/Qwen3/Model.lean

  Standalone Qwen3 causal-LM built on shared Qwen transformer blocks.
  Includes:
  - LM head
  - masked/unmasked forward
  - incremental KV-cache decode
  - greedy generation (cached + uncached)
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen
import Tyr.Model.Qwen3.Config
import Tyr.Model.Utils

/-! # Qwen3 causal LM model

`torch.qwen3.Qwen3ForCausalLM` adds an LM head (tied to the token embeddings by
default) on top of the Qwen decoder and returns `torch.qwen3.CausalLMOutput`.
-/

namespace torch.qwen3

open torch
open torch.Model

private def tokenInSetMask {batch : UInt64}
    (tokens : T #[batch])
    (allow : Array UInt64)
    : T #[batch] :=
  Id.run do
    let mut out : T #[batch] := torch.eq_scalar tokens (-1)
    for tok in allow do
      let isTok : T #[batch] := torch.eq_scalar tokens (Int64.ofNat tok.toNat)
      out := logicalOr out isTok
    out

private def eosVectorOnDevice {batch : UInt64}
    (eosTokenIds : Array UInt64)
    (device : Device)
    : Option (T #[batch]) :=
  match eosTokenIds[0]? with
  | none => none
  | some eosTok =>
    let eosCpu : T #[batch] := torch.full_int #[batch] (Int64.ofNat eosTok.toNat)
    let eosDev : T #[batch] := if eosCpu.device == device then eosCpu else eosCpu.to device
    some eosDev

/-- Causal-LM output container. -/
structure CausalLMOutput (cfg : Config) (batch seq : UInt64) where
  logits : T #[batch, seq, cfg.vocab_size]
  hiddenStates : Option (Array (T #[batch, seq, cfg.hidden_size])) := none

/-- Full standalone Qwen3 causal language model. -/
structure Qwen3ForCausalLM (cfg : Config) where
  model : qwen.Qwen3Model cfg
  lmHead : T #[cfg.vocab_size, cfg.hidden_size]
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Qwen3ForCausalLM

/-- Initialize model + LM head. By default, LM head is tied to token embeddings. -/
def init (cfg : Config) (tieWordEmbeddings : Bool := true) : IO (Qwen3ForCausalLM cfg) := do
  let model ← qwen.Qwen3Model.init cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initWeight #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model, lmHead, tieWordEmbeddings }

/-- Token embedding lookup. -/
def embedTokens {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  nn.embedding inputIds m.model.embed_tokens

/-- Forward from pre-computed input embeddings. -/
def forwardEmbeds {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocab_size] :=
  let (cos, sin) := rotary.computeFreqsPure seq cfg.head_dim cfg.rope_theta
  let hidden :=
    match attnMask with
    | some mask =>
      m.model.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.model.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  let hidden := m.model.norm.forward3d hidden
  linear3d hidden m.lmHead

/-- Standard forward pass from token IDs. -/
def forward {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocab_size] :=
  m.forwardEmbeds (m.embedTokens inputIds) attnMask

/-- Forward pass that also returns per-layer hidden states. -/
def forwardWithHiddenStates {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : CausalLMOutput cfg batch seq :=
  let (cos, sin) := rotary.computeFreqsPure seq cfg.head_dim cfg.rope_theta
  let hiddenStates :=
    match attnMask with
    | some mask => m.model.forwardWithHiddenStatesMasked cfg inputIds cos sin mask
    | none => m.model.forwardWithHiddenStates cfg inputIds cos sin
  let finalHidden : T #[batch, seq, cfg.hidden_size] :=
    hiddenStates.getD (hiddenStates.size - 1) (torch.zeros #[batch, seq, cfg.hidden_size])
  let logits := linear3d finalHidden m.lmHead
  { logits, hiddenStates := some hiddenStates }

/-- One-layer KV cache type for cached decoding. -/
abbrev LayerKVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch cfg.num_key_value_heads cfg.head_dim

private def initLayerKVCaches {batch : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache cfg batch) :=
  m.model.layers.map (fun _ =>
    qwen.QwenAttention.initKVCache
      maxLen
      (batch := batch)
      (num_kv_heads := cfg.num_key_value_heads)
      (head_dim := cfg.head_dim)
      device)

private def precomputeDecodeRotary {maxLen : UInt64}
    (_m : Qwen3ForCausalLM cfg)
    (device : Device)
    : T #[maxLen, cfg.head_dim / 2] × T #[maxLen, cfg.head_dim / 2] :=
  rotary.computeFreqsOnDevicePure
    maxLen
    cfg.head_dim
    cfg.rope_theta
    device

private def decodeStepFromEmbedWithCache {batch maxLen : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (cosAll : T #[maxLen, cfg.head_dim / 2])
    (sinAll : T #[maxLen, cfg.head_dim / 2])
    (tokenEmbed : T #[batch, 1, cfg.hidden_size])
    (position : UInt64)
    (caches : Array (LayerKVCache cfg batch))
    : IO (T #[batch, cfg.vocab_size] × Array (LayerKVCache cfg batch)) := do
  let cos : T #[1, cfg.head_dim / 2] := data.slice cosAll 0 position 1
  let sin : T #[1, cfg.head_dim / 2] := data.slice sinAll 0 position 1

  let mut hidden : T #[batch, 1, cfg.hidden_size] := tokenEmbed
  let mut nextCaches := caches

  for i in [:m.model.layers.size] do
    let layer ←
      match m.model.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Qwen3 layer at index {i}"
    let cache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Qwen3 KV cache at index {i}"
    let (hNext, cNext) := layer.forwardStep hidden cos sin cache
    hidden := hNext
    nextCaches := nextCaches.set! i cNext

  let hiddenNorm := m.model.norm.forward3d hidden
  let logits3 : T #[batch, 1, cfg.vocab_size] := linear3d hiddenNorm m.lmHead
  let logits2 : T #[batch, cfg.vocab_size] := reshape logits3 #[batch, cfg.vocab_size]
  pure (logits2, nextCaches)

private partial def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (cosAll : T #[maxLen, cfg.head_dim / 2])
    (sinAll : T #[maxLen, cfg.head_dim / 2])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (caches : Array (LayerKVCache cfg batch))
    (position : Nat)
    (lastLogits : T #[batch, cfg.vocab_size])
    : IO (T #[batch, cfg.vocab_size] × Array (LayerKVCache cfg batch)) := do
  if position >= seq.toNat then
    pure (lastLogits, caches)
  else
    let tok : T #[batch, 1, cfg.hidden_size] := data.slice inputsEmbeds 1 position.toUInt64 1
    let (logits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll tok position.toUInt64 caches
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches' (position + 1) logits

private partial def greedyLoopCached {batch maxLen : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (cosAll : T #[maxLen, cfg.head_dim / 2])
    (sinAll : T #[maxLen, cfg.head_dim / 2])
    (eosTokenIds : Array UInt64)
    (eosVector : Option (T #[batch]))
    (remaining : Nat)
    (caches : Array (LayerKVCache cfg batch))
    (lastLogits : T #[batch, cfg.vocab_size])
    (finished : Option (T #[batch]))
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw : T #[batch] := nn.argmax lastLogits 1
  let nextTok : T #[batch] :=
    match eosVector, finished with
    | some eosTok, some doneMask =>
      let activeMask : T #[batch] := torch.logical_not doneMask
      torch.where_ activeMask nextTokRaw eosTok
    | _, _ => nextTokRaw
  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  match eosVector with
  | none =>
    let nextEmb : T #[batch, 1, cfg.hidden_size] := m.embedTokens nextCol
    let (nextLogits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
    greedyLoopCached m cosAll sinAll eosTokenIds none (remaining - 1) caches' nextLogits none appended
  | some _ =>
    let reachedEos : T #[batch] := tokenInSetMask nextTok eosTokenIds
    let finished' : T #[batch] :=
      match finished with
      | some doneMask => logicalOr doneMask reachedEos
      | none => reachedEos
    let hasActiveRows : Bool := torch.any (torch.logical_not finished')
    if !hasActiveRows then
      return ⟨curSeq + 1, appended⟩
    else
      let nextEmb : T #[batch, 1, cfg.hidden_size] := m.embedTokens nextCol
      let (nextLogits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
      greedyLoopCached m cosAll sinAll eosTokenIds eosVector (remaining - 1) caches' nextLogits (some finished') appended

private partial def greedyLoopUncached {batch : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (eosTokenIds : Array UInt64)
    (eosVector : Option (T #[batch]))
    (remaining : Nat)
    (finished : Option (T #[batch]))
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"

  let logits := m.forward curIds none
  let lastPos := curSeq - 1
  let last3 : T #[batch, 1, cfg.vocab_size] :=
    reshape (data.slice logits 1 lastPos 1) #[batch, 1, cfg.vocab_size]
  let last2 : T #[batch, cfg.vocab_size] := reshape last3 #[batch, cfg.vocab_size]
  let nextTokRaw : T #[batch] := nn.argmax last2 1
  let nextTok : T #[batch] :=
    match eosVector, finished with
    | some eosTok, some doneMask =>
      let activeMask : T #[batch] := torch.logical_not doneMask
      torch.where_ activeMask nextTokRaw eosTok
    | _, _ => nextTokRaw
  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  match eosVector with
  | none =>
    greedyLoopUncached m eosTokenIds none (remaining - 1) none appended
  | some _ =>
    let reachedEos : T #[batch] := tokenInSetMask nextTok eosTokenIds
    let finished' : T #[batch] :=
      match finished with
      | some doneMask => logicalOr doneMask reachedEos
      | none => reachedEos
    let hasActiveRows : Bool := torch.any (torch.logical_not finished')
    if !hasActiveRows then
      return ⟨curSeq + 1, appended⟩
    else
      greedyLoopUncached m eosTokenIds eosVector (remaining - 1) (some finished') appended

/-- Greedy generation with static per-layer KV caches. -/
def generateGreedy {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let inputsEmbeds := m.embedTokens inputIds
  let cacheDevice := inputsEmbeds.device
  let cacheMaxLen : UInt64 := seq + maxNewTokens
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds cacheDevice
  let (cosAll, sinAll) := precomputeDecodeRotary (maxLen := cacheMaxLen) m cacheDevice
  let caches0 := initLayerKVCaches m cacheMaxLen cacheDevice
  let tok0 : T #[batch, 1, cfg.hidden_size] := data.slice inputsEmbeds 1 0 1
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m cosAll sinAll tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches1 1 logits0
  greedyLoopCached m cosAll sinAll eosTokenIds eosVector maxNewTokens.toNat cachesPrefill lastLogits none inputIds

/-- Reference greedy generation by full re-forward on each decode step. -/
def generateGreedyUncached {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds inputIds.device
  greedyLoopUncached m eosTokenIds eosVector maxNewTokens.toNat none inputIds

end Qwen3ForCausalLM

end torch.qwen3
