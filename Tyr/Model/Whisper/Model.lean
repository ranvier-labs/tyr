/-
  Tyr/Model/Whisper/Model.lean

  Native Whisper encoder-decoder model in Tyr.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.LayerNorm
import Tyr.Model.Whisper.Config
import Tyr.Model.Utils

/-! # Whisper model

Encoder-decoder transformer: `torch.whisper.WhisperAttention` with an
incremental `torch.whisper.WhisperAttention.KVCache` and cross-attention over
key/value states projected once from the encoder.
-/

namespace torch.whisper

open torch.Model

private def activate {s : Shape} (name : String) (x : T s) : T s :=
  if name == "relu" then
    nn.relu x
  else
    nn.gelu x

structure WhisperAttention (dModel nHeads : UInt64) where
  qProjWeight : T #[dModel, dModel]
  qProjBias : T #[dModel]
  kProjWeight : T #[dModel, dModel]
  kProjBias : T #[dModel]
  vProjWeight : T #[dModel, dModel]
  vProjBias : T #[dModel]
  outProjWeight : T #[dModel, dModel]
  outProjBias : T #[dModel]
  deriving TensorStruct

namespace WhisperAttention

def init (dModel nHeads : UInt64) : IO (WhisperAttention dModel nHeads) := do
  if nHeads == 0 || dModel % nHeads != 0 then
    throw <| IO.userError s!"WhisperAttention requires d_model divisible by n_heads, got {dModel} and {nHeads}"
  let qProjWeight ← initWeight #[dModel, dModel] dModel
  let qProjBias := initBias #[dModel]
  let kProjWeight ← initWeight #[dModel, dModel] dModel
  let kProjBias := initBias #[dModel]
  let vProjWeight ← initWeight #[dModel, dModel] dModel
  let vProjBias := initBias #[dModel]
  let outProjWeight ← initWeight #[dModel, dModel] dModel
  let outProjBias := initBias #[dModel]
  pure { qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias }

/-- Incremental self-attention KV cache for one decoder layer. -/
structure KVCache (batch nHeads headDim : UInt64) where
  kStoreDyn : T #[]
  vStoreDyn : T #[]
  seq : UInt64 := 0
  maxLen : UInt64 := 0

/-- Initialize an empty static-capacity KV cache on `device`. -/
def initKVCache {batch nHeads headDim : UInt64}
    (maxLen : UInt64)
    (device : Device := Device.CPU) : KVCache batch nHeads headDim :=
  let k0 : T #[batch, nHeads, maxLen, headDim] :=
    torch.zeros #[batch, nHeads, maxLen, headDim] false device
  let v0 : T #[batch, nHeads, maxLen, headDim] :=
    torch.zeros #[batch, nHeads, maxLen, headDim] false device
  { kStoreDyn := nn.eraseShape k0, vStoreDyn := nn.eraseShape v0, seq := 0, maxLen := maxLen }

/-- Project external key/value states once into attention layout.
    Output layout is `[batch, nHeads, kvSeq, headDim]`. -/
def projectKV {batch kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (keyValueStates : T #[batch, kvSeq, dModel])
    : T #[batch, nHeads, kvSeq, dModel / nHeads] × T #[batch, nHeads, kvSeq, dModel / nHeads] :=
  let headDim : UInt64 := dModel / nHeads
  let k : T #[batch, kvSeq, dModel] := affine3d keyValueStates m.kProjWeight m.kProjBias
  let v : T #[batch, kvSeq, dModel] := affine3d keyValueStates m.vProjWeight m.vProjBias
  let k : T #[batch, kvSeq, nHeads, headDim] := reshape k #[batch, kvSeq, nHeads, headDim]
  let v : T #[batch, kvSeq, nHeads, headDim] := reshape v #[batch, kvSeq, nHeads, headDim]
  (nn.transpose_for_attention k, nn.transpose_for_attention v)

/-- Run attention where K/V are already projected and transposed.
    `keyStates` / `valueStates` layout: `[batch, nHeads, kvSeq, headDim]`. -/
def forwardWithProjectedKV {batch qSeq kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (queryStates : T #[batch, qSeq, dModel])
    (keyStates : T #[batch, nHeads, kvSeq, dModel / nHeads])
    (valueStates : T #[batch, nHeads, kvSeq, dModel / nHeads])
    (isCausal : Bool := false)
    : T #[batch, qSeq, dModel] :=
  let headDim : UInt64 := dModel / nHeads
  let q : T #[batch, qSeq, dModel] := affine3d queryStates m.qProjWeight m.qProjBias
  let q : T #[batch, qSeq, nHeads, headDim] := reshape q #[batch, qSeq, nHeads, headDim]
  let qh : T #[batch, nHeads, qSeq, headDim] := nn.transpose_for_attention q
  let attn : T #[batch, nHeads, qSeq, headDim] :=
    nn.scaledDotProductAttentionGQAQKV
      (n_kv_head := nHeads)
      qh
      keyStates
      valueStates
      0.0
      isCausal
      true
  let out : T #[batch, qSeq, nHeads, headDim] := nn.transpose_from_attention attn
  let out : T #[batch, qSeq, dModel] := reshape out #[batch, qSeq, dModel]
  affine3d out m.outProjWeight m.outProjBias

def forwardCross {batch qSeq kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (queryStates : T #[batch, qSeq, dModel])
    (keyValueStates : T #[batch, kvSeq, dModel])
    (isCausal : Bool := false)
    : T #[batch, qSeq, dModel] :=
  let (kh, vh) := projectKV m keyValueStates
  forwardWithProjectedKV m queryStates kh vh isCausal

def forwardSelf {batch seq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (x : T #[batch, seq, dModel])
    (isCausal : Bool := false)
    : T #[batch, seq, dModel] :=
  forwardCross m x x isCausal

/-- Incremental one-token self-attention step with KV cache update. -/
def forwardSelfStep {batch : UInt64}
    (m : WhisperAttention dModel nHeads)
    (x : T #[batch, 1, dModel])
    (cache : KVCache batch nHeads (dModel / nHeads))
    : T #[batch, 1, dModel] × KVCache batch nHeads (dModel / nHeads) :=
  let headDim : UInt64 := dModel / nHeads
  let q0 : T #[batch, 1, dModel] := affine3d x m.qProjWeight m.qProjBias
  let k0 : T #[batch, 1, dModel] := affine3d x m.kProjWeight m.kProjBias
  let v0 : T #[batch, 1, dModel] := affine3d x m.vProjWeight m.vProjBias

  let q : T #[batch, 1, nHeads, headDim] := reshape q0 #[batch, 1, nHeads, headDim]
  let k : T #[batch, 1, nHeads, headDim] := reshape k0 #[batch, 1, nHeads, headDim]
  let v : T #[batch, 1, nHeads, headDim] := reshape v0 #[batch, 1, nHeads, headDim]

  let qh : T #[batch, nHeads, 1, headDim] := nn.transpose_for_attention q
  let kNew : T #[batch, nHeads, 1, headDim] := nn.transpose_for_attention k
  let vNew : T #[batch, nHeads, 1, headDim] := nn.transpose_for_attention v

  let kStore : T #[batch, nHeads, cache.maxLen, headDim] :=
    castLike kNew (reshape cache.kStoreDyn #[batch, nHeads, cache.maxLen, headDim])
  let vStore : T #[batch, nHeads, cache.maxLen, headDim] :=
    castLike vNew (reshape cache.vStoreDyn #[batch, nHeads, cache.maxLen, headDim])

  if cache.seq < cache.maxLen then
    let kStore' : T #[batch, nHeads, cache.maxLen, headDim] :=
      data.sliceScatter kStore 2 cache.seq kNew
    let vStore' : T #[batch, nHeads, cache.maxLen, headDim] :=
      data.sliceScatter vStore 2 cache.seq vNew
    let kvLen : UInt64 := cache.seq + 1
    let kAll : T #[batch, nHeads, kvLen, headDim] := data.slice kStore' 2 0 kvLen
    let vAll : T #[batch, nHeads, kvLen, headDim] := data.slice vStore' 2 0 kvLen
    let attn : T #[batch, nHeads, 1, headDim] :=
      nn.scaledDotProductAttentionGQAQKV
        (n_kv_head := nHeads)
        qh
        kAll
        vAll
        0.0
        false
        true
    let out : T #[batch, 1, nHeads, headDim] := nn.transpose_from_attention attn
    let out : T #[batch, 1, dModel] := reshape out #[batch, 1, dModel]
    let out : T #[batch, 1, dModel] := affine3d out m.outProjWeight m.outProjBias
    let cache' : KVCache batch nHeads headDim := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : UInt64 := if cache.maxLen == 0 then 0 else cache.maxLen - 1
    let kStore' : T #[batch, nHeads, cache.maxLen, headDim] :=
      data.sliceScatter kStore 2 writePos kNew
    let vStore' : T #[batch, nHeads, cache.maxLen, headDim] :=
      data.sliceScatter vStore 2 writePos vNew
    let kvLen : UInt64 := cache.maxLen
    let kAll : T #[batch, nHeads, kvLen, headDim] := data.slice kStore' 2 0 kvLen
    let vAll : T #[batch, nHeads, kvLen, headDim] := data.slice vStore' 2 0 kvLen
    let attn : T #[batch, nHeads, 1, headDim] :=
      nn.scaledDotProductAttentionGQAQKV
        (n_kv_head := nHeads)
        qh
        kAll
        vAll
        0.0
        false
        true
    let out : T #[batch, 1, nHeads, headDim] := nn.transpose_from_attention attn
    let out : T #[batch, 1, dModel] := reshape out #[batch, 1, dModel]
    let out : T #[batch, 1, dModel] := affine3d out m.outProjWeight m.outProjBias
    let cache' : KVCache batch nHeads headDim := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

end WhisperAttention

structure WhisperEncoderLayer (cfg : WhisperConfig) where
  selfAttn : WhisperAttention cfg.dModel cfg.encoderAttentionHeads
  selfAttnLayerNorm : LayerNorm cfg.dModel
  fc1Weight : T #[cfg.encoderFfnDim, cfg.dModel]
  fc1Bias : T #[cfg.encoderFfnDim]
  fc2Weight : T #[cfg.dModel, cfg.encoderFfnDim]
  fc2Bias : T #[cfg.dModel]
  finalLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace WhisperEncoderLayer

def init (cfg : WhisperConfig) : IO (WhisperEncoderLayer cfg) := do
  let selfAttn ← WhisperAttention.init cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeight #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias := initBias #[cfg.encoderFfnDim]
  let fc2Weight ← initWeight #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias := initBias #[cfg.dModel]
  let finalLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

def forward {batch seq : UInt64}
    (m : WhisperEncoderLayer cfg)
    (x : T #[batch, seq, cfg.dModel])
    : T #[batch, seq, cfg.dModel] :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf h0 (isCausal := false)
  let x1 := x + h1
  let h2 := m.finalLayerNorm.forward3d x1
  let h3 : T #[batch, seq, cfg.encoderFfnDim] := affine3d h2 m.fc1Weight m.fc1Bias
  let h4 := activate cfg.activationFunction h3
  let h5 : T #[batch, seq, cfg.dModel] := affine3d h4 m.fc2Weight m.fc2Bias
  x1 + h5

end WhisperEncoderLayer

structure WhisperDecoderLayer (cfg : WhisperConfig) where
  selfAttn : WhisperAttention cfg.dModel cfg.decoderAttentionHeads
  selfAttnLayerNorm : LayerNorm cfg.dModel
  encoderAttn : WhisperAttention cfg.dModel cfg.decoderAttentionHeads
  encoderAttnLayerNorm : LayerNorm cfg.dModel
  fc1Weight : T #[cfg.decoderFfnDim, cfg.dModel]
  fc1Bias : T #[cfg.decoderFfnDim]
  fc2Weight : T #[cfg.dModel, cfg.decoderFfnDim]
  fc2Bias : T #[cfg.dModel]
  finalLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace WhisperDecoderLayer

def init (cfg : WhisperConfig) : IO (WhisperDecoderLayer cfg) := do
  let selfAttn ← WhisperAttention.init cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  let encoderAttn ← WhisperAttention.init cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeight #[cfg.decoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias := initBias #[cfg.decoderFfnDim]
  let fc2Weight ← initWeight #[cfg.dModel, cfg.decoderFfnDim] cfg.decoderFfnDim
  let fc2Bias := initBias #[cfg.dModel]
  let finalLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  pure {
    selfAttn
    selfAttnLayerNorm
    encoderAttn
    encoderAttnLayerNorm
    fc1Weight
    fc1Bias
    fc2Weight
    fc2Bias
    finalLayerNorm
  }

def forward {batch seq encSeq : UInt64}
    (m : WhisperDecoderLayer cfg)
    (x : T #[batch, seq, cfg.dModel])
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : T #[batch, seq, cfg.dModel] :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf h0 (isCausal := true)
  let x1 := x + h1
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardCross h2 encoderHidden (isCausal := false)
  let x2 := x1 + h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : T #[batch, seq, cfg.decoderFfnDim] := affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : T #[batch, seq, cfg.dModel] := affine3d h6 m.fc2Weight m.fc2Bias
  x2 + h7

/-- One-token decoder layer step with incremental self KV cache and precomputed cross KV. -/
def forwardStepCached {batch encSeq : UInt64}
    (m : WhisperDecoderLayer cfg)
    (x : T #[batch, 1, cfg.dModel])
    (crossK : T #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads])
    (crossV : T #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads])
    (selfCache : WhisperAttention.KVCache batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads))
    : T #[batch, 1, cfg.dModel] × WhisperAttention.KVCache batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads) :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let (h1a, selfCache') := m.selfAttn.forwardSelfStep h0 selfCache
  let x1 := x + h1a
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardWithProjectedKV h2 crossK crossV (isCausal := false)
  let x2 := x1 + h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : T #[batch, 1, cfg.decoderFfnDim] := affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : T #[batch, 1, cfg.dModel] := affine3d h6 m.fc2Weight m.fc2Bias
  (x2 + h7, selfCache')

end WhisperDecoderLayer

structure WhisperModel (cfg : WhisperConfig) where
  encoderConv1Weight : T #[cfg.dModel, cfg.numMelBins, 3]
  encoderConv1Bias : T #[cfg.dModel]
  encoderConv2Weight : T #[cfg.dModel, cfg.dModel, 3]
  encoderConv2Bias : T #[cfg.dModel]
  encoderPositionalEmbedding : T #[cfg.maxSourcePositions, cfg.dModel]
  encoderLayers : Array (WhisperEncoderLayer cfg)
  encoderLayerNorm : LayerNorm cfg.dModel

  decoderTokenEmbedding : T #[cfg.vocabSize, cfg.dModel]
  decoderPositionalEmbedding : T #[cfg.maxTargetPositions, cfg.dModel]
  decoderLayers : Array (WhisperDecoderLayer cfg)
  decoderLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace WhisperModel

def init (cfg : WhisperConfig) : IO (WhisperModel cfg) := do
  if !cfg.hasValidHeads then
    throw <| IO.userError
      s!"Invalid Whisper config: d_model={cfg.dModel}, encoder_heads={cfg.encoderAttentionHeads}, decoder_heads={cfg.decoderAttentionHeads}"
  let encoderConv1Weight ← initWeight #[cfg.dModel, cfg.numMelBins, 3] cfg.numMelBins
  let encoderConv1Bias := initBias #[cfg.dModel]
  let encoderConv2Weight ← initWeight #[cfg.dModel, cfg.dModel, 3] cfg.dModel
  let encoderConv2Bias := initBias #[cfg.dModel]
  let encoderPositionalEmbedding ← initWeight #[cfg.maxSourcePositions, cfg.dModel] cfg.dModel

  let mut encoderLayers : Array (WhisperEncoderLayer cfg) := #[]
  for _ in [:cfg.encoderLayers.toNat] do
    encoderLayers := encoderLayers.push (← WhisperEncoderLayer.init cfg)
  let encoderLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ← initWeight #[cfg.vocabSize, cfg.dModel] cfg.dModel
  let decoderPositionalEmbedding ← initWeight #[cfg.maxTargetPositions, cfg.dModel] cfg.dModel
  let mut decoderLayers : Array (WhisperDecoderLayer cfg) := #[]
  for _ in [:cfg.decoderLayers.toNat] do
    decoderLayers := decoderLayers.push (← WhisperDecoderLayer.init cfg)
  let decoderLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps

  pure {
    encoderConv1Weight
    encoderConv1Bias
    encoderConv2Weight
    encoderConv2Bias
    encoderPositionalEmbedding
    encoderLayers
    encoderLayerNorm
    decoderTokenEmbedding
    decoderPositionalEmbedding
    decoderLayers
    decoderLayerNorm
  }

def encode {batch frames : UInt64}
    (m : WhisperModel cfg)
    (inputFeatures : T #[batch, cfg.numMelBins, frames])
    : IO (T #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel]) := do
  let x1 : T #[batch, cfg.dModel, frames] :=
    reshape
      (nn.gelu (nn.conv1d_group_bias inputFeatures m.encoderConv1Weight m.encoderConv1Bias 1 1 1 1))
      #[batch, cfg.dModel, frames]
  let outSeq := WhisperConfig.conv2OutputSeq frames
  if outSeq > cfg.maxSourcePositions then
    throw <| IO.userError
      s!"Whisper encoder sequence {outSeq} exceeds max_source_positions {cfg.maxSourcePositions}"
  let x2 : T #[batch, cfg.dModel, outSeq] :=
    reshape
      (nn.gelu (nn.conv1d_group_bias x1 m.encoderConv2Weight m.encoderConv2Bias 2 1 1 1))
      #[batch, cfg.dModel, outSeq]
  let x2t : T #[batch, outSeq, cfg.dModel] := permute x2 #[0, 2, 1]
  let pos : T #[outSeq, cfg.dModel] := data.slice m.encoderPositionalEmbedding 0 0 outSeq
  let posB : T #[batch, outSeq, cfg.dModel] :=
    nn.expand (reshape pos #[1, outSeq, cfg.dModel]) #[batch, outSeq, cfg.dModel]
  let mut h : T #[batch, outSeq, cfg.dModel] := x2t + posB
  for layer in m.encoderLayers do
    h := layer.forward h
  pure (m.encoderLayerNorm.forward3d h)

def decode {batch seq encSeq : UInt64}
    (m : WhisperModel cfg)
    (inputIds : T #[batch, seq])
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : IO (T #[batch, seq, cfg.dModel]) := do
  if seq > cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder sequence {seq} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : T #[batch, seq, cfg.dModel] := nn.embedding inputIds m.decoderTokenEmbedding
  let pos : T #[seq, cfg.dModel] := data.slice m.decoderPositionalEmbedding 0 0 seq
  let posB : T #[batch, seq, cfg.dModel] :=
    nn.expand (reshape pos #[1, seq, cfg.dModel]) #[batch, seq, cfg.dModel]
  let mut h : T #[batch, seq, cfg.dModel] := tokEmb + posB
  for layer in m.decoderLayers do
    h := layer.forward h encoderHidden
  pure (m.decoderLayerNorm.forward3d h)

end WhisperModel

structure WhisperForConditionalGeneration (cfg : WhisperConfig) where
  model : WhisperModel cfg
  projOut : T #[cfg.vocabSize, cfg.dModel]
  deriving TensorStruct

namespace WhisperForConditionalGeneration

/-- Per-layer precomputed cross-attention K/V for one encoded audio chunk. -/
structure LayerCrossCache (cfg : WhisperConfig) where
  kDyn : T #[]
  vDyn : T #[]

/-- Per-layer self-attention KV cache for decoder incremental decode. -/
abbrev LayerKVCache (cfg : WhisperConfig) (batch : UInt64) :=
  WhisperAttention.KVCache batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads)

def init (cfg : WhisperConfig) : IO (WhisperForConditionalGeneration cfg) := do
  let model ← WhisperModel.init cfg
  let projOut ← initWeight #[cfg.vocabSize, cfg.dModel] cfg.dModel
  pure { model, projOut }

def encode {batch frames : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.numMelBins, frames])
    : IO (T #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel]) :=
  m.model.encode inputFeatures

def decode {batch seq encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : IO (T #[batch, seq, cfg.vocabSize]) := do
  let hidden ← m.model.decode inputIds encoderHidden
  pure (linear3d hidden m.projOut)

/-- Initialize one self-attention KV cache per decoder layer. -/
def initLayerKVCaches {batch : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache cfg batch) :=
  m.model.decoderLayers.map (fun _ =>
    WhisperAttention.initKVCache
      maxLen
      (batch := batch)
      (nHeads := cfg.decoderAttentionHeads)
      (headDim := cfg.dModel / cfg.decoderAttentionHeads)
      device)

/-- Precompute cross-attention K/V once per decoder layer for an encoded chunk. -/
def precomputeCrossCaches {batch encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : IO (Array (LayerCrossCache cfg)) := do
  let mut out : Array (LayerCrossCache cfg) := #[]
  for layer in m.model.decoderLayers do
    let (k, v) := layer.encoderAttn.projectKV encoderHidden
    out := out.push { kDyn := nn.eraseShape k, vDyn := nn.eraseShape v }
  pure out

/-- Decode one token step using static self KV caches and precomputed cross K/V. -/
def decodeStepWithCache {batch encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (tokenIds : T #[batch, 1])
    (position : UInt64)
    (crossCaches : Array (LayerCrossCache cfg))
    (selfCaches : Array (LayerKVCache cfg batch))
    : IO (T #[batch, cfg.vocabSize] × Array (LayerKVCache cfg batch)) := do
  if position >= cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder position {position} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : T #[batch, 1, cfg.dModel] := nn.embedding tokenIds m.model.decoderTokenEmbedding
  let pos : T #[1, cfg.dModel] := data.slice m.model.decoderPositionalEmbedding 0 position 1
  let posB : T #[batch, 1, cfg.dModel] :=
    nn.expand (reshape pos #[1, 1, cfg.dModel]) #[batch, 1, cfg.dModel]
  let mut h : T #[batch, 1, cfg.dModel] := tokEmb + posB
  let mut nextCaches := selfCaches
  for i in [:m.model.decoderLayers.size] do
    let layer ←
      match m.model.decoderLayers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Whisper decoder layer at index {i}"
    let selfCache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Whisper self KV cache at index {i}"
    let crossCache ←
      match crossCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Whisper cross KV cache at index {i}"
    let crossK : T #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] :=
      reshape crossCache.kDyn #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads]
    let crossV : T #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] :=
      reshape crossCache.vDyn #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads]
    let (hNext, cacheNext) := layer.forwardStepCached h crossK crossV selfCache
    h := hNext
    nextCaches := nextCaches.set! i cacheNext
  let hiddenNorm : T #[batch, 1, cfg.dModel] := m.model.decoderLayerNorm.forward3d h
  let logits3 : T #[batch, 1, cfg.vocabSize] := linear3d hiddenNorm m.projOut
  let logits2 : T #[batch, cfg.vocabSize] := reshape logits3 #[batch, cfg.vocabSize]
  pure (logits2, nextCaches)

end WhisperForConditionalGeneration

end torch.whisper
