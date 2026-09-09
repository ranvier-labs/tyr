/-
  Tyr/Model/Qwen3ASR/Model.lean

  Lean-native Qwen3-ASR thinker + top-level model:
  - audio tower
  - text decoder
  - LM head (ASR / forced-aligner shape aware)
  - rope-delta helpers + greedy generation utilities
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.AudioEncoder
import Tyr.Model.Qwen3ASR.ForcedAligner
import Tyr.Model.Utils

namespace torch.qwen3asr

open torch.Model

abbrev TextQwenConfig (cfg : ThinkerConfig) : qwen.QwenConfig :=
  TextConfig.toQwenConfig cfg.textConfig

abbrev ThinkerLmVocabSize (cfg : ThinkerConfig) : UInt64 :=
  ThinkerConfig.lmHeadOutDim cfg

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

private def buildLengthMask {batch maxLen : UInt64} (lengths : Array UInt64) : T #[batch, maxLen] :=
  Id.run do
    let mut vals : Array Int64 := #[]
    for i in [:batch.toNat] do
      let len := lengths.getD i 0
      for j in [:maxLen.toNat] do
        let v : Int64 := if j.toUInt64 < len then 1 else 0
        vals := vals.push v
    reshape (data.fromInt64Array vals) #[batch, maxLen]

private def maskRowCounts {batch seq : UInt64} (attentionMask : T #[batch, seq]) : IO (Array UInt64) := do
  let flat : T #[batch * seq] := reshape (data.toLong attentionMask) #[batch * seq]
  let vals ← data.tensorToUInt64Array flat
  let mut counts : Array UInt64 := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let mut c : UInt64 := 0
    for t in [:seq.toNat] do
      if vals.getD (b * seq.toNat + t) 0 != 0 then
        c := c + 1
    counts := counts.push c
  pure counts

private def clipTo (x hi : UInt64) : UInt64 :=
  if hi == 0 then 0 else if x < hi then x else hi - 1

/-- Derive per-batch position rows used by reference `get_rope_index`.
    Padded tokens map to position `1`; valid tokens are `0,1,2,...`. -/
private def positionRowsFromMask {batch seq : UInt64}
    (attentionMask : T #[batch, seq])
    : IO (Array (Array UInt64)) := do
  let flat : T #[batch * seq] := reshape (data.toLong attentionMask) #[batch * seq]
  let vals ← data.tensorToUInt64Array flat
  let mut rows : Array (Array UInt64) := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let mut row : Array UInt64 := Array.mkEmpty seq.toNat
    let mut running : UInt64 := 0
    for t in [:seq.toNat] do
      let v := vals.getD (b * seq.toNat + t) 0
      if v == 0 then
        row := row.push 1
      else
        row := row.push running
        running := running + 1
    rows := rows.push row
  pure rows

/-- Gather `[seq, half]` rotary table rows using explicit per-token positions. -/
private def gatherRowsByPositions {seq half : UInt64}
    (table : T #[seq, half])
    (positions : Array UInt64)
    : T #[seq, half] :=
  let idxVals : Array Int64 :=
    Id.run do
      let mut out : Array Int64 := Array.mkEmpty seq.toNat
      for i in [:seq.toNat] do
        let p := clipTo (positions.getD i 0) seq
        out := out.push (Int64.ofNat p.toNat)
      out
  let idxCpu : T #[seq] := reshape (data.fromInt64Array idxVals) #[seq]
  let idx : T #[seq] := idxCpu.to table.device
  let idx2d : T #[seq, half] := nn.expand (reshape idx #[seq, 1]) #[seq, half]
  reshape (torch.gather table 0 idx2d) #[seq, half]

/-- Output container mirroring `...CausalLMOutputWithPast` fields that are
    currently represented in the Lean port. -/
structure Qwen3ASRThinkerCausalLMOutput (cfg : ThinkerConfig) (batch seq : UInt64) where
  logits : T #[batch, seq, ThinkerLmVocabSize cfg]
  ropeDeltas : Option (T #[batch, 1]) := none

/-- Small Lean equivalent of generation input preparation payload. -/
structure ThinkerGenerationInputs (cfg : ThinkerConfig) (batch seq frames : UInt64) where
  inputIds : T #[batch, seq]
  inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none
  featureAttentionMask : Option (T #[batch, frames]) := none

/-- Thinker model: audio encoder + text decoder + LM head. -/
structure Qwen3ASRThinkerForConditionalGeneration (cfg : ThinkerConfig) where
  audioTower : AudioEncoder cfg.audioConfig
  textModel : qwen.Qwen3Model (TextQwenConfig cfg)
  audioProjectionWeight : T #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim]
  audioProjectionBias : T #[cfg.textConfig.hiddenSize]
  lmHead : T #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize]
  deriving TensorStruct

namespace Qwen3ASRThinkerForConditionalGeneration

def init (cfg : ThinkerConfig) : IO (Qwen3ASRThinkerForConditionalGeneration cfg) := do
  let audioTower ← AudioEncoder.init cfg.audioConfig
  let textModel ← qwen.Qwen3Model.init (TextQwenConfig cfg)
  let audioProjectionWeight ←
    if h : cfg.audioConfig.outputDim = cfg.textConfig.hiddenSize then
      let eye : T #[cfg.textConfig.hiddenSize, cfg.textConfig.hiddenSize] :=
        autograd.set_requires_grad (torch.eye cfg.textConfig.hiddenSize false) true
      let eye' : T #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] := by
        simpa [h] using eye
      pure eye'
    else
      throw <| IO.userError
        s!"Qwen3-ASR expects audio output dim == text hidden dim, got {cfg.audioConfig.outputDim} and {cfg.textConfig.hiddenSize}"
  let audioProjectionBias := initBias #[cfg.textConfig.hiddenSize]
  let lmHead ← initWeight #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize] cfg.textConfig.hiddenSize
  pure { audioTower, textModel, audioProjectionWeight, audioProjectionBias, lmHead }

def embedText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    : T #[batch, textSeq, cfg.textConfig.hiddenSize] :=
  nn.embedding textIds m.textModel.embed_tokens

def projectAudio {batch audioSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (audioFeatures : T #[batch, audioSeq, cfg.audioConfig.outputDim])
    : T #[batch, audioSeq, cfg.textConfig.hiddenSize] :=
  affine3d audioFeatures m.audioProjectionWeight m.audioProjectionBias

def buildInputsFromText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    : T #[batch, textSeq, cfg.textConfig.hiddenSize] :=
  embedText m textIds

def buildInputsFromAudioAndText {batch audioSeq textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (audioFeatures : T #[batch, audioSeq, cfg.audioConfig.outputDim])
    (textIds : T #[batch, textSeq])
    : T #[batch, audioSeq + textSeq, cfg.textConfig.hiddenSize] :=
  let audioEmb := projectAudio m audioFeatures
  let textEmb := embedText m textIds
  nn.cat audioEmb textEmb 1

def forwardEmbeds {batch seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputsEmbeds : T #[batch, seq, cfg.textConfig.hiddenSize])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, ThinkerLmVocabSize cfg] :=
  let (cos, sin) :=
    rotary.computeFreqsOnDevicePure
      seq
      cfg.textConfig.headDim
      cfg.textConfig.ropeTheta
      inputsEmbeds.device
  let hidden := match attnMask with
    | some mask =>
      m.textModel.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.textModel.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  let hidden := m.textModel.norm.forward3d hidden
  linear3d hidden m.lmHead

private def forwardEmbedsMaskedWithPositionRows {batch seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputsEmbeds : T #[batch, seq, cfg.textConfig.hiddenSize])
    (attnMask : T #[batch, seq])
    : IO (T #[batch, seq, ThinkerLmVocabSize cfg]) := do
  let half := cfg.textConfig.headDim / 2
  let (baseCos, baseSin) :=
    rotary.computeFreqsOnDevicePure
      seq
      cfg.textConfig.headDim
      cfg.textConfig.ropeTheta
      inputsEmbeds.device
  let rows ← positionRowsFromMask attnMask

  if batch == 0 then
    pure (torch.zeros #[batch, seq, ThinkerLmVocabSize cfg])
  else
    let mut perBatchHidden : Array (T #[]) := #[]
    for b in [:batch.toNat] do
      let row := rows.getD b (Array.replicate seq.toNat 0)
      let cosB : T #[seq, half] := gatherRowsByPositions baseCos row
      let sinB : T #[seq, half] := gatherRowsByPositions baseSin row
      let embB : T #[1, seq, cfg.textConfig.hiddenSize] := data.slice inputsEmbeds 0 b.toUInt64 1
      let maskB : T #[1, seq] := data.slice attnMask 0 b.toUInt64 1
      let hiddenB :=
        m.textModel.layers.foldl
          (fun h layer => layer.forwardMasked h cosB sinB maskB true)
          embB
      let hiddenB := m.textModel.norm.forward3d hiddenB
      perBatchHidden := perBatchHidden.push (nn.eraseShape hiddenB)

    let hiddenDyn : T #[] := nn.cat_dyn perBatchHidden 0
    let hidden : T #[batch, seq, cfg.textConfig.hiddenSize] :=
      reshape hiddenDyn #[batch, seq, cfg.textConfig.hiddenSize]
    pure (linear3d hidden m.lmHead)

def forwardText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, textSeq]) := none)
    : T #[batch, textSeq, ThinkerLmVocabSize cfg] :=
  forwardEmbeds m (buildInputsFromText m textIds) attnMask

def forwardAudioText {batch audioSeq textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (audioFeatures : T #[batch, audioSeq, cfg.audioConfig.outputDim])
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, audioSeq + textSeq]) := none)
    : T #[batch, audioSeq + textSeq, ThinkerLmVocabSize cfg] :=
  forwardEmbeds m (buildInputsFromAudioAndText m audioFeatures textIds) attnMask

def encodeAudio {batch frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.audioConfig.numMelBins, frames])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames, cfg.audioConfig.outputDim] :=
  m.audioTower.forward inputFeatures attnMask

def encodeAudioVarLen {batch frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.audioConfig.numMelBins, frames])
    (featureLens : Array UInt64)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames, cfg.audioConfig.outputDim] :=
  m.audioTower.forwardVarLen inputFeatures featureLens

private def featureLengthsFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq : UInt64}
    (featureAttentionMask : T #[batch, featureSeq])
    : IO (Array UInt64) := do
  let _ := cfg
  let featureLensTensor : T #[batch] := nn.sumDim (data.toLong featureAttentionMask) 1 false
  let featureLensRaw ← data.tensorToUInt64Array featureLensTensor
  pure <| featureLensRaw.map (fun l => if l <= featureSeq then l else featureSeq)

private def getPlaceholderMask2d (cfg : ThinkerConfig) {batch seq : UInt64}
    (inputIds : T #[batch, seq])
    : T #[batch, seq] :=
  eq_scalar inputIds (Int64.ofNat cfg.audioTokenId.toNat)

private def getPlaceholderMask (cfg : ThinkerConfig) {batch seq : UInt64}
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.textConfig.hiddenSize] :=
  let mask2d := getPlaceholderMask2d cfg inputIds
  nn.expand (reshape mask2d #[batch, seq, 1]) #[batch, seq, cfg.textConfig.hiddenSize]

private def audioValidMaskFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq audioSeq : UInt64}
    (featureAttentionMask : T #[batch, featureSeq])
    : IO (T #[batch, audioSeq, cfg.textConfig.hiddenSize]) := do
  let featureLens ← featureLengthsFromFeatureMask cfg featureAttentionMask
  let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
  let valid2d : T #[batch, audioSeq] := buildLengthMask audioLens
  let valid2dBool : T #[batch, audioSeq] := eq_scalar valid2d 1
  pure <| nn.expand (reshape valid2dBool #[batch, audioSeq, 1]) #[batch, audioSeq, cfg.textConfig.hiddenSize]

private def audioAttentionMaskFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq audioSeq : UInt64}
    (featureAttentionMask : T #[batch, featureSeq])
    : IO (T #[batch, audioSeq]) := do
  let featureLens ← featureLengthsFromFeatureMask cfg featureAttentionMask
  let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
  pure (buildLengthMask audioLens)

private def buildInputsEmbeds {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    : IO (T #[batch, seq, cfg.textConfig.hiddenSize]) := do
  let inputsEmbeds0 := embedText m inputIds
  match inputFeatures with
  | none => pure inputsEmbeds0
  | some feats =>
    let audioSeq := AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames
    let placeholderMask2d := getPlaceholderMask2d cfg inputIds
    let placeholderLensTensor : T #[batch] := nn.sumDim (data.toLong placeholderMask2d) 1 false
    let placeholderLens ← data.tensorToUInt64Array placeholderLensTensor
    let (audioLensOpt, _audioAttnMask, audioFeaturesRaw) ←
      match featureAttentionMask with
      | some fm =>
        let featureLens ← featureLengthsFromFeatureMask cfg fm
        let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
        let m2d ← audioAttentionMaskFromFeatureMask cfg (audioSeq := audioSeq) fm
        let raw := encodeAudioVarLen m feats featureLens
        pure (some audioLens, some m2d, raw)
      | none =>
        let raw := encodeAudio m feats none
        pure (none, none, raw)

    let expectedAudioLens :=
      match audioLensOpt with
      | some xs => xs.map (fun l => if l <= audioSeq then l else audioSeq)
      | none => Array.replicate batch.toNat audioSeq
    let expectedRows := batch.toNat
    if expectedAudioLens.size != expectedRows then
      throw <| IO.userError
        s!"audio expected-length row mismatch: expected {expectedRows}, got {expectedAudioLens.size}"
    if placeholderLens.size != expectedRows then
      throw <| IO.userError
        s!"audio placeholder row mismatch: expected {expectedRows}, got {placeholderLens.size}"
    for i in [:expectedRows] do
      let got := placeholderLens[i]!
      let expected := expectedAudioLens[i]!
      if got = expected then
        pure ()
      else if got = 0 && expected > 0 then
        throw <| IO.userError
          s!"audio placeholder mismatch at batch {i}: expected {expected}, got 0 (silent audio-drop)"
      else if got < expected then
        throw <| IO.userError
          s!"audio placeholder mismatch at batch {i}: expected {expected}, got {got} (too few placeholders)"
      else
        throw <| IO.userError
          s!"audio placeholder mismatch at batch {i}: expected {expected}, got {got} (too many placeholders)"

    let audioFeatures := projectAudio m audioFeaturesRaw
    let source ←
      match featureAttentionMask with
      | some fm =>
        let validMask ← audioValidMaskFromFeatureMask cfg (audioSeq := audioSeq) fm
        let validMask :=
          if validMask.device == audioFeatures.device then validMask else validMask.to audioFeatures.device
        pure (nn.masked_select audioFeatures validMask)
      | none =>
        pure (reshape audioFeatures #[batch * audioSeq * cfg.textConfig.hiddenSize])
    let placeholderMask := getPlaceholderMask cfg inputIds
    pure (nn.masked_scatter inputsEmbeds0 placeholderMask source)

/-- Port of reference `get_rope_index` calculation from attention masks. -/
def getRopeIndex {batch seq : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (attentionMask : T #[batch, seq])
    : IO (T #[3, batch, seq] × T #[batch, 1]) := do
  let flat : T #[batch * seq] := reshape (data.toLong attentionMask) #[batch * seq]
  let vals ← data.tensorToUInt64Array flat

  let mut posVals : Array Int64 := Array.mkEmpty (3 * batch.toNat * seq.toNat)
  let mut deltaVals : Array Int64 := Array.mkEmpty batch.toNat

  for b in [:batch.toNat] do
    let mut row : Array UInt64 := Array.mkEmpty seq.toNat
    let mut running : UInt64 := 0
    let mut maxPos : UInt64 := 1
    let mut validCount : UInt64 := 0

    for t in [:seq.toNat] do
      let v := vals.getD (b * seq.toNat + t) 0
      if v == 0 then
        row := row.push 1
      else
        let p := running
        row := row.push p
        running := running + 1
        validCount := validCount + 1
        if p > maxPos then
          maxPos := p

    let delta := (maxPos + 1) - validCount
    deltaVals := deltaVals.push (Int64.ofNat delta.toNat)

    for _ in [:3] do
      for p in row do
        posVals := posVals.push (Int64.ofNat p.toNat)

  let positionIds : T #[3, batch, seq] := reshape (data.fromInt64Array posVals) #[3, batch, seq]
  let ropeDeltas : T #[batch, 1] := reshape (data.fromInt64Array deltaVals) #[batch, 1]
  pure (positionIds, ropeDeltas)

abbrev LayerKVCache (cfg : ThinkerConfig) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch cfg.textConfig.numKeyValueHeads cfg.textConfig.headDim

private def initLayerKVCaches {batch : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache cfg batch) :=
  m.textModel.layers.map (fun _ =>
    qwen.QwenAttention.initKVCache
      maxLen
      (batch := batch)
      (num_kv_heads := cfg.textConfig.numKeyValueHeads)
      (head_dim := cfg.textConfig.headDim)
      device)

private def precomputeDecodeRotary {maxLen : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (device : Device)
    : T #[maxLen, cfg.textConfig.headDim / 2] × T #[maxLen, cfg.textConfig.headDim / 2] :=
  rotary.computeFreqsOnDevicePure
    maxLen
    cfg.textConfig.headDim
    cfg.textConfig.ropeTheta
    device

private def decodeStepFromEmbedWithCache {batch maxLen : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (cosAll : T #[maxLen, cfg.textConfig.headDim / 2])
    (sinAll : T #[maxLen, cfg.textConfig.headDim / 2])
    (tokenEmbed : T #[batch, 1, cfg.textConfig.hiddenSize])
    (position : UInt64)
    (caches : Array (LayerKVCache cfg batch))
    : IO (T #[batch, ThinkerLmVocabSize cfg] × Array (LayerKVCache cfg batch)) := do
  let cos : T #[1, cfg.textConfig.headDim / 2] := data.slice cosAll 0 position 1
  let sin : T #[1, cfg.textConfig.headDim / 2] := data.slice sinAll 0 position 1

  let mut hidden : T #[batch, 1, cfg.textConfig.hiddenSize] := tokenEmbed
  let mut nextCaches := caches

  for i in [:m.textModel.layers.size] do
    let layer ←
      match m.textModel.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing thinker text layer at index {i}"
    let cache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing thinker KV cache at index {i}"
    let (hNext, cNext) := layer.forwardStep hidden cos sin cache
    hidden := hNext
    nextCaches := nextCaches.set! i cNext

  let hiddenNorm := m.textModel.norm.forward3d hidden
  let logits3 : T #[batch, 1, ThinkerLmVocabSize cfg] := linear3d hiddenNorm m.lmHead
  let logits2 : T #[batch, ThinkerLmVocabSize cfg] := reshape logits3 #[batch, ThinkerLmVocabSize cfg]
  pure (logits2, nextCaches)

private partial def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (cosAll : T #[maxLen, cfg.textConfig.headDim / 2])
    (sinAll : T #[maxLen, cfg.textConfig.headDim / 2])
    (inputsEmbeds : T #[batch, seq, cfg.textConfig.hiddenSize])
    (caches : Array (LayerKVCache cfg batch))
    (position : Nat)
    (lastLogits : T #[batch, ThinkerLmVocabSize cfg])
    : IO (T #[batch, ThinkerLmVocabSize cfg] × Array (LayerKVCache cfg batch)) := do
  if position >= seq.toNat then
    pure (lastLogits, caches)
  else
    let tok : T #[batch, 1, cfg.textConfig.hiddenSize] := data.slice inputsEmbeds 1 position.toUInt64 1
    let (logits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll tok position.toUInt64 caches
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches' (position + 1) logits

private partial def greedyLoopCached {batch maxLen : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (cosAll : T #[maxLen, cfg.textConfig.headDim / 2])
    (sinAll : T #[maxLen, cfg.textConfig.headDim / 2])
    (eosTokenIds : Array UInt64)
    (eosVector : Option (T #[batch]))
    (remaining : Nat)
    (caches : Array (LayerKVCache cfg batch))
    (lastLogits : T #[batch, ThinkerLmVocabSize cfg])
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
    let nextEmb : T #[batch, 1, cfg.textConfig.hiddenSize] := m.embedText nextCol
    let (nextLogits, caches') ←
      decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
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
      let nextEmb : T #[batch, 1, cfg.textConfig.hiddenSize] := m.embedText nextCol
      let (nextLogits, caches') ←
        decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
      greedyLoopCached m cosAll sinAll eosTokenIds eosVector (remaining - 1) caches' nextLogits (some finished') appended

private def isUInt32Prefix (pref xs : Array UInt32) : Bool :=
  if pref.size > xs.size then
    false
  else
    Id.run do
      let mut ok := true
      for i in [:pref.size] do
        if ok && pref[i]! != xs[i]! then
          ok := false
      ok

/-- Reusable streaming prompt prefill state for batch=1 greedy decode.
    Carries text-model KV/cos/sin state for the current prompt prefix. -/
structure StreamingPromptCache (cfg : ThinkerConfig) where
  seq : UInt64
  maxLen : UInt64
  promptTokenIds : Array UInt32 := #[]
  promptEmbedsDyn : T #[]
  cosAll : T #[maxLen, cfg.textConfig.headDim / 2]
  sinAll : T #[maxLen, cfg.textConfig.headDim / 2]
  kvCaches : Array (LayerKVCache cfg 1)
  lastLogits : T #[1, ThinkerLmVocabSize cfg]

private def buildStreamingPromptCacheFromEmbeds {seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputsEmbeds : T #[1, seq, cfg.textConfig.hiddenSize])
    (promptTokenIds : Array UInt32)
    (maxNewTokens : UInt64)
    : IO (StreamingPromptCache cfg) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedyWithPromptCache requires non-empty prompt sequence"
  let reserve := if maxNewTokens == 0 then 1 else maxNewTokens
  let maxLen : UInt64 := seq + reserve
  let cacheDevice := inputsEmbeds.device
  let (cosAll, sinAll) := precomputeDecodeRotary (maxLen := maxLen) m cacheDevice
  let caches0 := initLayerKVCaches m maxLen cacheDevice
  let tok0 : T #[1, 1, cfg.textConfig.hiddenSize] := data.slice inputsEmbeds 1 0 1
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m cosAll sinAll tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches1 1 logits0
  pure {
    seq := seq
    maxLen := maxLen
    promptTokenIds := promptTokenIds
    promptEmbedsDyn := nn.eraseShape inputsEmbeds
    cosAll := cosAll
    sinAll := sinAll
    kvCaches := cachesPrefill
    lastLogits := lastLogits
  }

private def canReuseStreamingPromptCache {seq : UInt64}
    (cache : StreamingPromptCache cfg)
    (_inputsEmbeds : T #[1, seq, cfg.textConfig.hiddenSize])
    (promptTokenIds : Array UInt32)
    (maxNewTokens : UInt64)
    : IO Bool := do
  if cache.seq > seq then
    pure false
  else if !isUInt32Prefix cache.promptTokenIds promptTokenIds then
    pure false
  else
    let reserve := if maxNewTokens == 0 then 1 else maxNewTokens
    let neededMax := seq + reserve
    if cache.maxLen < neededMax then
      pure false
    else
      pure true

private def extendStreamingPromptCacheFromEmbeds {seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (cache : StreamingPromptCache cfg)
    (inputsEmbeds : T #[1, seq, cfg.textConfig.hiddenSize])
    (promptTokenIds : Array UInt32)
    : IO (StreamingPromptCache cfg) := do
  if cache.seq == seq then
    pure {
      cache with
      promptTokenIds := promptTokenIds
      promptEmbedsDyn := nn.eraseShape inputsEmbeds
    }
  else
    let (lastLogits, kvCaches) ←
      prefillCachesFromEmbeds
        m
        cache.cosAll
        cache.sinAll
        inputsEmbeds
        cache.kvCaches
        cache.seq.toNat
        cache.lastLogits
    pure {
      cache with
      seq := seq
      promptTokenIds := promptTokenIds
      promptEmbedsDyn := nn.eraseShape inputsEmbeds
      kvCaches := kvCaches
      lastLogits := lastLogits
    }

/-- Lean analogue of `prepare_inputs_for_generation` behavior for audio features:
    only feed audio on the first generation step (`cachePosition == 0`). -/
def prepareInputsForGeneration {batch seq frames : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (cachePosition : UInt64 := 0)
    : ThinkerGenerationInputs cfg batch seq frames :=
  if cachePosition == 0 then
    { inputIds, inputFeatures, featureAttentionMask }
  else
    { inputIds, inputFeatures := none, featureAttentionMask }

/-- Faithful thinker forward shape: text embeddings with in-place audio placeholder replacement. -/
def forwardWithAux {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (Qwen3ASRThinkerCausalLMOutput cfg batch seq) := do
  let inputsEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  let logits ←
    match attentionMask with
    | none => pure (forwardEmbeds m inputsEmbeds none)
    | some mask => forwardEmbedsMaskedWithPositionRows m inputsEmbeds mask

  let ropeDeltas ←
    match attentionMask with
    | none => pure none
    | some mask =>
      let (_positionIds, ropeRaw) ← m.getRopeIndex mask
      let rawVals : Array UInt64 ←
        data.tensorToUInt64Array (reshape (data.toLong ropeRaw) #[batch])
      let validCounts ← maskRowCounts mask
      let mut out : Array Int64 := Array.mkEmpty batch.toNat
      for b in [:batch.toNat] do
        let raw := rawVals.getD b 0
        let valid := validCounts.getD b 0
        let pad := seq - valid
        let adj := if raw >= pad then raw - pad else 0
        out := out.push (Int64.ofNat adj.toNat)
      let deltas : T #[batch, 1] := reshape (data.fromInt64Array out) #[batch, 1]
      pure (some deltas)

  pure { logits, ropeDeltas }

def forward {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, ThinkerLmVocabSize cfg]) := do
  return (← m.forwardWithAux inputIds inputFeatures featureAttentionMask attentionMask).logits

private partial def greedyLoopUncached {batch : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    {promptSeq : UInt64}
    (promptEmbeds : T #[batch, promptSeq, cfg.textConfig.hiddenSize])
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
  if curSeq < promptSeq then
    throw <| IO.userError
      s!"generateGreedyUncached prompt boundary mismatch: promptSeq={promptSeq}, curSeq={curSeq}"

  let inputsEmbeds : T #[batch, curSeq, cfg.textConfig.hiddenSize] :=
    if h : curSeq = promptSeq then
      by simpa [h] using promptEmbeds
    else
      let suffixSeq := curSeq - promptSeq
      let suffixIds : T #[batch, suffixSeq] := data.slice curIds 1 promptSeq suffixSeq
      let suffixEmbeds : T #[batch, suffixSeq, cfg.textConfig.hiddenSize] := m.embedText suffixIds
      let inputsEmbedsDyn : T #[] := nn.cat_dyn #[nn.eraseShape promptEmbeds, nn.eraseShape suffixEmbeds] 1
      reshape inputsEmbedsDyn #[batch, curSeq, cfg.textConfig.hiddenSize]
  let logits := m.forwardEmbeds inputsEmbeds none
  let lastPos := curSeq - 1
  let last3 : T #[batch, 1, ThinkerLmVocabSize cfg] :=
    reshape (data.slice logits 1 lastPos 1) #[batch, 1, ThinkerLmVocabSize cfg]
  let last2 : T #[batch, ThinkerLmVocabSize cfg] := reshape last3 #[batch, ThinkerLmVocabSize cfg]
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
    greedyLoopUncached
      m promptEmbeds eosTokenIds none (remaining - 1) none appended
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
      greedyLoopUncached
        m
        promptEmbeds
        eosTokenIds
        eosVector
        (remaining - 1)
        (some finished')
        appended

/-- Compatibility path mirroring the previous full re-forward greedy loop.
    Useful for parity checks against the cached generation path. -/
def generateGreedyUncached {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩
  let promptEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds inputIds.device
  greedyLoopUncached
    m
    promptEmbeds
    eosTokenIds
    eosVector
    maxNewTokens.toNat
    none
    inputIds

/-- Greedy generation utility (Lean-only, no vLLM dependency).
    Returns generated full sequence `[prompt || new_tokens]` with dynamic output length. -/
def generateGreedy {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let inputsEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  let cacheDevice := inputsEmbeds.device
  let cacheMaxLen : UInt64 := seq + maxNewTokens
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds cacheDevice
  let (cosAll, sinAll) := precomputeDecodeRotary (maxLen := cacheMaxLen) m cacheDevice
  let caches0 := initLayerKVCaches m cacheMaxLen cacheDevice
  let tok0 : T #[batch, 1, cfg.textConfig.hiddenSize] := data.slice inputsEmbeds 1 0 1
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m cosAll sinAll tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches1 1 logits0
  greedyLoopCached m cosAll sinAll eosTokenIds eosVector maxNewTokens.toNat cachesPrefill lastLogits none inputIds

/-- Greedy generation with reusable batch=1 prompt-prefix cache.
    Intended for streaming decode where successive prompts typically extend
    previously accepted text prefixes. -/
def generateGreedyFromInputsEmbedsWithPromptCache {seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[1, seq])
    (inputsEmbeds : T #[1, seq, cfg.textConfig.hiddenSize])
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => T #[1, outSeq])) × StreamingPromptCache cfg) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedyWithPromptCache requires non-empty prompt sequence"
  if promptTokenIds.size != seq.toNat then
    throw <| IO.userError
      s!"generateGreedyWithPromptCache prompt-id mismatch: tensor_seq={seq}, prompt_ids={promptTokenIds.size}"
  if maxNewTokens == 0 then
    let cache ←
      match prefixCache with
      | some c =>
        if (← canReuseStreamingPromptCache c inputsEmbeds promptTokenIds maxNewTokens) then
          extendStreamingPromptCacheFromEmbeds m c inputsEmbeds promptTokenIds
        else
          buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
      | none =>
        buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
    pure (⟨seq, inputIds⟩, cache)
  else
    let cache ←
      match prefixCache with
      | some c =>
        if (← canReuseStreamingPromptCache c inputsEmbeds promptTokenIds maxNewTokens) then
          extendStreamingPromptCacheFromEmbeds m c inputsEmbeds promptTokenIds
        else
          buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
      | none =>
        buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
    let generated ←
      greedyLoopCached
        m
        cache.cosAll
        cache.sinAll
        eosTokenIds
        (eosVectorOnDevice (batch := 1) eosTokenIds inputIds.device)
        maxNewTokens.toNat
        cache.kvCaches
        cache.lastLogits
        none
        inputIds
    pure (generated, cache)

/-- Greedy generation with reusable batch=1 prompt-prefix cache.
    Intended for streaming decode where successive prompts typically extend
    previously accepted text prefixes. -/
def generateGreedyWithPromptCache {seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[1, seq])
    (inputFeatures : Option (T #[1, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[1, frames]) := none)
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => T #[1, outSeq])) × StreamingPromptCache cfg) := do
  let inputsEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  m.generateGreedyFromInputsEmbedsWithPromptCache
    inputIds
    inputsEmbeds
    promptTokenIds
    prefixCache
    maxNewTokens
    eosTokenIds

def forwardFromMel {batch frames textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.audioConfig.numMelBins, frames])
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames + textSeq]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames + textSeq, ThinkerLmVocabSize cfg] :=
  let audio := m.encodeAudio inputFeatures
  m.forwardAudioText audio textIds attnMask

end Qwen3ASRThinkerForConditionalGeneration

/-- Top-level Qwen3-ASR model wrapper (thinker-only inference in this Lean port). -/
structure Qwen3ASRForConditionalGeneration (cfg : Qwen3ASRConfig) where
  thinker : Qwen3ASRThinkerForConditionalGeneration cfg.thinkerConfig
  supportLanguages : Array String := cfg.supportLanguages
  deriving TensorStruct

namespace Qwen3ASRForConditionalGeneration

def init (cfg : Qwen3ASRConfig) : IO (Qwen3ASRForConditionalGeneration cfg) := do
  let thinker ← Qwen3ASRThinkerForConditionalGeneration.init cfg.thinkerConfig
  pure { thinker, supportLanguages := cfg.supportLanguages }

def getSupportedLanguages (m : Qwen3ASRForConditionalGeneration cfg) : Array String :=
  m.supportLanguages

def forwardText {batch textSeq : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, textSeq]) := none)
    : T #[batch, textSeq, ThinkerLmVocabSize cfg.thinkerConfig] :=
  m.thinker.forwardText textIds attnMask

def forwardFromMel {batch frames textSeq : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames])
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq, ThinkerLmVocabSize cfg.thinkerConfig] :=
  m.thinker.forwardFromMel inputFeatures textIds attnMask

def forwardWithAux {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (Qwen3ASRThinkerCausalLMOutput cfg.thinkerConfig batch seq) :=
  m.thinker.forwardWithAux inputIds inputFeatures featureAttentionMask attentionMask

def forward {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, ThinkerLmVocabSize cfg.thinkerConfig]) :=
  m.thinker.forward inputIds inputFeatures featureAttentionMask attentionMask

def generateGreedy {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  m.thinker.generateGreedy inputIds inputFeatures featureAttentionMask maxNewTokens eosTokenIds

abbrev StreamingPromptCache (cfg : Qwen3ASRConfig) :=
  Qwen3ASRThinkerForConditionalGeneration.StreamingPromptCache cfg.thinkerConfig

/-- Greedy generation with reusable batch=1 streaming prompt cache. -/
def generateGreedyWithPromptCache {seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[1, seq])
    (inputFeatures : Option (T #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[1, frames]) := none)
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => T #[1, outSeq])) × StreamingPromptCache cfg) :=
  m.thinker.generateGreedyWithPromptCache
    inputIds
    inputFeatures
    featureAttentionMask
    promptTokenIds
    prefixCache
    maxNewTokens
    eosTokenIds

/-- Greedy generation from precomputed input embeddings with reusable
    batch=1 streaming prompt cache. -/
def generateGreedyFromInputsEmbedsWithPromptCache {seq : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[1, seq])
    (inputsEmbeds : T #[1, seq, cfg.thinkerConfig.textConfig.hiddenSize])
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => T #[1, outSeq])) × StreamingPromptCache cfg) :=
  m.thinker.generateGreedyFromInputsEmbedsWithPromptCache
    inputIds
    inputsEmbeds
    promptTokenIds
    prefixCache
    maxNewTokens
    eosTokenIds

def generateGreedyUncached {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  m.thinker.generateGreedyUncached inputIds inputFeatures featureAttentionMask maxNewTokens eosTokenIds

def alignFromOutputIds {batch seq : UInt64}
    (_m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (outputIds : T #[batch, seq])
    (wordLists : Array (Array String))
    (timestampTokenId : UInt64 := cfg.timestampTokenId)
    (timestampSegmentTime : Float := cfg.timestampSegmentTime)
    : IO (Array ForcedAlignResult) :=
  torch.qwen3asr.alignFromOutputIds inputIds outputIds wordLists timestampTokenId timestampSegmentTime

def alignFromLogits {batch seq : UInt64}
    (_m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (logits : T #[batch, seq, ThinkerLmVocabSize cfg.thinkerConfig])
    (wordLists : Array (Array String))
    (timestampTokenId : UInt64 := cfg.timestampTokenId)
    (timestampSegmentTime : Float := cfg.timestampSegmentTime)
    : IO (Array ForcedAlignResult) :=
  torch.qwen3asr.alignFromLogits inputIds logits wordLists timestampTokenId timestampSegmentTime

def alignPrepared {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (wordLists : Array (Array String))
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    (timestampTokenId : UInt64 := cfg.timestampTokenId)
    (timestampSegmentTime : Float := cfg.timestampSegmentTime)
    : IO (Array ForcedAlignResult) := do
  let logits ← m.forward inputIds inputFeatures featureAttentionMask attentionMask
  m.alignFromLogits inputIds logits wordLists timestampTokenId timestampSegmentTime

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
