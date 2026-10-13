/-
  Tyr/Model/Gemma4/Multimodal.lean

  Gemma 4 multimodal path for Tyr:
  - Gemma 4 vision patch embedder + 2D RoPE encoder + spatial pooler
  - Gemma 4 multimodal projection into text hidden size
  - Placeholder-based fusion into prompt embeddings

  This follows the public Hugging Face Gemma 4 reference architecture for the
  image tower and projector. Tyr currently preprocesses images one at a time
  and concatenates their soft tokens, instead of batching padded patch tensors.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Gemma4.Model
import Tyr.Model.Gemma4.VLConfig

/-! # Gemma 4 vision tower

Vision encoder and multimodal glue for Gemma 4: patch grids
(`torch.gemma4.ImagePatchGrid`), the optional clipped linear
`torch.gemma4.Gemma4VisionLinear` used by small towers, the resulting
`torch.gemma4.ImageFeatures`, and per-layer inputs merged into the text model.
-/

namespace torch.gemma4

open torch
open torch.Model

abbrev ImagePatchGrid (cfg : VLConfig) :=
  Sigma (fun patchRows =>
    Sigma (fun patchCols =>
      T #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config]))

abbrev ImageFeatures (cfg : VLConfig) :=
  Sigma (fun n => T #[n, cfg.text_config.hidden_size])

abbrev PerLayerInputs (cfg : VLConfig) (batch seq : UInt64) :=
  Option (T #[batch, seq, cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size_per_layer_input])

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad t false

private def clampMin {s : Shape} (x : T s) (lo : Float) : T s :=
  let loTensor : T s := add_scalar (zeros_like x) lo
  where_ (lt_scalar x lo) loTensor x

private def clampMax {s : Shape} (x : T s) (hi : Float) : T s :=
  let hiTensor : T s := add_scalar (zeros_like x) hi
  where_ (gt x hiTensor) hiTensor x

private def clampRange {s : Shape}
    (x : T s)
    (lo? : Option Float := none)
    (hi? : Option Float := none)
    : T s :=
  let x1 :=
    match lo? with
    | some lo => clampMin x lo
    | none => x
  match hi? with
  | some hi => clampMax x1 hi
  | none => x1

private def linear2d {tokens inDim outDim : UInt64}
    (x : T #[tokens, inDim])
    (w : T #[outDim, inDim])
    : T #[tokens, outDim] :=
  let yDyn : T #[] := torch.einsum2 "oh,th->to" w x
  reshape yDyn #[tokens, outDim]

private def countMask2d {batch seq : UInt64} (mask : T #[batch, seq]) : IO UInt64 := do
  let row : T #[batch] := reshape (nn.sumDim (data.toLong mask) 1 false) #[batch]
  let counts ← data.tensorToUInt64Array row
  pure (counts.foldl (· + ·) 0)

private def positionIds (patchRows patchCols : UInt64) :
    T #[patchRows * patchCols] × T #[patchRows * patchCols] :=
  Id.run do
    let mut xIds : Array Int64 := #[]
    let mut yIds : Array Int64 := #[]
    for y in [:patchRows.toNat] do
      for x in [:patchCols.toNat] do
        xIds := xIds.push x.toUInt64.toInt64
        yIds := yIds.push y.toUInt64.toInt64
    pure
      ( reshape (data.fromInt64Array xIds) #[patchRows * patchCols]
      , reshape (data.fromInt64Array yIds) #[patchRows * patchCols] )

private def rmsNormNoScale2d {n dim : UInt64}
    (x : T #[n, dim])
    (eps : Float := 1e-6)
    : T #[n, dim] :=
  let xf : T #[n, dim] := toFloat' x
  let var : T #[n, 1] := nn.meanDim (xf * xf) 1 true
  let inv : T #[n, 1] := nn.rsqrt (var + eps)
  let inv : T #[n, dim] := nn.expand inv #[n, dim]
  restoreInputDType x (xf * inv)

private def rmsNormNoScale4d {a b c dim : UInt64}
    (x : T #[a, b, c, dim])
    (eps : Float := 1e-6)
    : T #[a, b, c, dim] :=
  let flat : T #[a * b * c, dim] := reshape x #[a * b * c, dim]
  reshape (rmsNormNoScale2d flat eps) #[a, b, c, dim]

private def apply2dRotaryEmb {tokens nHead headDim : UInt64}
    (x : T #[1, tokens, nHead, headDim])
    (patchRows patchCols : UInt64)
    (ropeTheta : Float)
    : T #[1, tokens, nHead, headDim] :=
  let perDim := headDim / 2
  let xPart : T #[1, tokens, nHead, perDim] := data.slice x 3 0 perDim
  let yPart : T #[1, tokens, nHead, perDim] := data.slice x 3 perDim perDim
  let (xIdsCpu, yIdsCpu) := positionIds patchRows patchCols
  let (xCosTable, xSinTable) := rotary.computeFreqsOnDevicePure patchCols perDim ropeTheta x.device
  let (yCosTable, ySinTable) := rotary.computeFreqsOnDevicePure patchRows perDim ropeTheta x.device
  let xIds : T #[tokens] := xIdsCpu.to x.device
  let yIds : T #[tokens] := yIdsCpu.to x.device
  let xCos : T #[tokens, perDim / 2] := nn.embedding1d xIds xCosTable
  let xSin : T #[tokens, perDim / 2] := nn.embedding1d xIds xSinTable
  let yCos : T #[tokens, perDim / 2] := nn.embedding1d yIds yCosTable
  let ySin : T #[tokens, perDim / 2] := nn.embedding1d yIds ySinTable
  let xRot : T #[1, tokens, nHead, perDim] := rotary.applyRotaryEmb xPart xCos xSin
  let yRot : T #[1, tokens, nHead, perDim] := rotary.applyRotaryEmb yPart yCos ySin
  nn.cat xRot yRot 3

private def replaceTokenId {batch seq : UInt64}
    (inputIds : T #[batch, seq])
    (fromId toId : UInt64)
    : T #[batch, seq] :=
  let mask : T #[batch, seq] := eq_scalar inputIds (Int64.ofNat fromId.toNat)
  let repl : T #[batch, seq] := (full_int #[batch, seq] (Int64.ofNat toId.toNat)).to inputIds.device
  where_ mask repl inputIds

/-- Optional clipped linear used by small Gemma 4 vision towers. -/
structure Gemma4VisionLinear (outDim inDim : UInt64) where
  weight : T #[outDim, inDim]
  input_min : Option Float := none
  input_max : Option Float := none
  output_min : Option Float := none
  output_max : Option Float := none
  deriving TensorStruct

namespace Gemma4VisionLinear

def forward2d {tokens outDim inDim : UInt64}
    (m : Gemma4VisionLinear outDim inDim)
    (x : T #[tokens, inDim])
    : T #[tokens, outDim] :=
  let x' := clampRange x m.input_min m.input_max
  let y := linear2d x' m.weight
  clampRange y m.output_min m.output_max

end Gemma4VisionLinear

structure Gemma4VisionPatchEmbedder (cfg : VLConfig) where
  input_proj : Gemma4VisionLinear cfg.vision_config.hidden_size (VisionConfig.patchDim cfg.vision_config)
  position_embedding_table : T #[2, cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  deriving TensorStruct

namespace Gemma4VisionPatchEmbedder

def forward {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4VisionPatchEmbedder cfg)
    (patchGrid : T #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config])
    : T #[patchRows * patchCols, cfg.vision_config.hidden_size] :=
  let tokens := patchRows * patchCols
  let flat : T #[tokens, VisionConfig.patchDim cfg.vision_config] :=
    reshape patchGrid #[tokens, VisionConfig.patchDim cfg.vision_config]
  let flat := mul_scalar (add_scalar flat (-0.5)) 2.0
  let hidden : T #[tokens, cfg.vision_config.hidden_size] := m.input_proj.forward2d flat
  let xTable : T #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] :=
    reshape (data.slice m.position_embedding_table 0 0 1)
      #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  let yTable : T #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] :=
    reshape (data.slice m.position_embedding_table 0 1 1)
      #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  let (xIdsCpu, yIdsCpu) := positionIds patchRows patchCols
  let xIds : T #[tokens] := xIdsCpu.to hidden.device
  let yIds : T #[tokens] := yIdsCpu.to hidden.device
  let xPos : T #[tokens, cfg.vision_config.hidden_size] := nn.embedding1d xIds xTable
  let yPos : T #[tokens, cfg.vision_config.hidden_size] := nn.embedding1d yIds yTable
  hidden + xPos + yPos

end Gemma4VisionPatchEmbedder

structure Gemma4VisionMLP (cfg : VisionConfig) where
  gate_proj : Gemma4VisionLinear cfg.intermediate_size cfg.hidden_size
  up_proj : Gemma4VisionLinear cfg.intermediate_size cfg.hidden_size
  down_proj : Gemma4VisionLinear cfg.hidden_size cfg.intermediate_size
  deriving TensorStruct

namespace Gemma4VisionMLP

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionMLP cfg)
    (x : T #[tokens, cfg.hidden_size])
    : T #[tokens, cfg.hidden_size] :=
  let gate : T #[tokens, cfg.intermediate_size] := m.gate_proj.forward2d x
  let up : T #[tokens, cfg.intermediate_size] := m.up_proj.forward2d x
  m.down_proj.forward2d (nn.gelu gate * up)

end Gemma4VisionMLP

structure Gemma4VisionAttention (cfg : VisionConfig) where
  q_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  k_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  v_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  o_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  q_norm : Gemma4RMSNorm cfg.head_dim
  k_norm : Gemma4RMSNorm cfg.head_dim
  deriving TensorStruct

namespace Gemma4VisionAttention

def forward {patchRows patchCols : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionAttention cfg)
    (x : T #[patchRows * patchCols, cfg.hidden_size])
    : T #[patchRows * patchCols, cfg.hidden_size] :=
  let tokens := patchRows * patchCols
  let q0 : T #[tokens, cfg.hidden_size] := m.q_proj.forward2d x
  let k0 : T #[tokens, cfg.hidden_size] := m.k_proj.forward2d x
  let v0 : T #[tokens, cfg.hidden_size] := m.v_proj.forward2d x

  let q1 : T #[1, tokens, cfg.num_attention_heads, cfg.head_dim] :=
    reshape q0 #[1, tokens, cfg.num_attention_heads, cfg.head_dim]
  let k1 : T #[1, tokens, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape k0 #[1, tokens, cfg.num_key_value_heads, cfg.head_dim]
  let v1 : T #[1, tokens, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape v0 #[1, tokens, cfg.num_key_value_heads, cfg.head_dim]

  let q2 := m.q_norm.forward4d q1
  let k2 := m.k_norm.forward4d k1
  let v2 := rmsNormNoScale4d v1 cfg.rms_norm_eps

  let q3 : T #[1, tokens, cfg.num_attention_heads, cfg.head_dim] :=
    apply2dRotaryEmb q2 patchRows patchCols cfg.rope_theta
  let k3 : T #[1, tokens, cfg.num_key_value_heads, cfg.head_dim] :=
    apply2dRotaryEmb k2 patchRows patchCols cfg.rope_theta

  let qh : T #[1, cfg.num_attention_heads, tokens, cfg.head_dim] :=
    mul_scalar (nn.transpose_for_attention q3) (Float.sqrt cfg.head_dim.toFloat)
  let kh : T #[1, cfg.num_key_value_heads, tokens, cfg.head_dim] := nn.transpose_for_attention k3
  let vh : T #[1, cfg.num_key_value_heads, tokens, cfg.head_dim] := nn.transpose_for_attention v2
  let attn : T #[1, cfg.num_attention_heads, tokens, cfg.head_dim] :=
    nn.scaledDotProductAttentionGQA qh kh vh 0.0 false true
  let out : T #[tokens, cfg.hidden_size] :=
    reshape (nn.transpose_from_attention attn) #[tokens, cfg.hidden_size]
  m.o_proj.forward2d out

end Gemma4VisionAttention

structure Gemma4VisionBlock (cfg : VisionConfig) where
  input_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_attention_layernorm : Gemma4RMSNorm cfg.hidden_size
  pre_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  self_attn : Gemma4VisionAttention cfg
  mlp : Gemma4VisionMLP cfg
  deriving TensorStruct

namespace Gemma4VisionBlock

def forward {patchRows patchCols : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionBlock cfg)
    (x : T #[patchRows * patchCols, cfg.hidden_size])
    : T #[patchRows * patchCols, cfg.hidden_size] :=
  let residual1 := x
  let h1 := m.input_layernorm.forward2d x
  let attn := m.self_attn.forward (patchRows := patchRows) (patchCols := patchCols) cfg h1
  let h2 := residual1 + m.post_attention_layernorm.forward2d attn
  let residual2 := h2
  let h3 := m.pre_feedforward_layernorm.forward2d h2
  let mlp := m.mlp.forward cfg h3
  residual2 + m.post_feedforward_layernorm.forward2d mlp

end Gemma4VisionBlock

structure Gemma4VisionModel (cfg : VLConfig) where
  patch_embedder : Gemma4VisionPatchEmbedder cfg
  blocks : Array (Gemma4VisionBlock cfg.vision_config)
  std_bias : Option (T #[cfg.vision_config.hidden_size]) := none
  std_scale : Option (T #[cfg.vision_config.hidden_size]) := none
  deriving TensorStruct

namespace Gemma4VisionModel

private def poolFeatures {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (x : T #[patchRows * patchCols, cfg.vision_config.hidden_size])
    : T #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.vision_config.hidden_size] :=
  let pool := cfg.vision_config.pooling_kernel_size
  let pooledRows := patchRows / pool
  let pooledCols := patchCols / pool
  let xGrid : T #[1, patchRows, patchCols, cfg.vision_config.hidden_size] :=
    reshape x #[1, patchRows, patchCols, cfg.vision_config.hidden_size]
  let xNchw : T #[1, cfg.vision_config.hidden_size, patchRows, patchCols] := permute xGrid #[0, 3, 1, 2]
  let pooled : T #[1, cfg.vision_config.hidden_size, pooledRows, pooledCols] :=
    nn.avg_pool2d xNchw #[pool, pool] #[pool, pool] #[0, 0]
  let pooledNhwc : T #[1, pooledRows, pooledCols, cfg.vision_config.hidden_size] :=
    permute pooled #[0, 2, 3, 1]
  mul_scalar
    (reshape pooledNhwc #[pooledRows * pooledCols, cfg.vision_config.hidden_size])
    (Float.sqrt cfg.vision_config.hidden_size.toFloat)

def forward {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4VisionModel cfg)
    (patchGrid : T #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config])
    : T #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.vision_config.hidden_size] :=
  Id.run do
    let mut hidden := m.patch_embedder.forward cfg patchGrid
    for i in [:m.blocks.size] do
      match m.blocks[i]? with
      | some blk =>
        hidden := blk.forward (patchRows := patchRows) (patchCols := patchCols) cfg.vision_config hidden
      | none =>
        pure ()
    let pooled := poolFeatures cfg hidden
    match m.std_bias, m.std_scale with
    | some bias, some scale =>
      let nTok := (patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size)
      let bias : T #[nTok, cfg.vision_config.hidden_size] :=
        nn.expand (reshape bias #[1, cfg.vision_config.hidden_size]) #[nTok, cfg.vision_config.hidden_size]
      let scale : T #[nTok, cfg.vision_config.hidden_size] :=
        nn.expand (reshape scale #[1, cfg.vision_config.hidden_size]) #[nTok, cfg.vision_config.hidden_size]
      (pooled - bias) * scale
    | _, _ =>
      pooled

end Gemma4VisionModel

structure Gemma4MultimodalEmbedder (cfg : VLConfig) where
  embedding_projection : T #[cfg.text_config.hidden_size, cfg.vision_config.hidden_size]
  deriving TensorStruct

namespace Gemma4MultimodalEmbedder

def forward {tokens : UInt64}
    (cfg : VLConfig)
    (m : Gemma4MultimodalEmbedder cfg)
    (inputsEmbeds : T #[tokens, cfg.vision_config.hidden_size])
    : T #[tokens, cfg.text_config.hidden_size] :=
  linear2d (rmsNormNoScale2d inputsEmbeds cfg.text_config.rms_norm_eps) m.embedding_projection

end Gemma4MultimodalEmbedder

structure Gemma4ForConditionalGeneration (cfg : VLConfig) where
  vision_tower : Gemma4VisionModel cfg
  embed_vision : Gemma4MultimodalEmbedder cfg
  language_model : Gemma4ForCausalLM cfg.text_config
  deriving TensorStruct

namespace Gemma4ForConditionalGeneration

def getImageFeatures {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (patchGrid : T #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config])
    : T #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.text_config.hidden_size] :=
  let visionFeatures := m.vision_tower.forward cfg patchGrid
  m.embed_vision.forward cfg visionFeatures

def getImageFeaturesMany
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (images : Array (ImagePatchGrid cfg))
    : IO (Option (ImageFeatures cfg)) := do
  let mut acc? : Option (ImageFeatures cfg) := none
  for img in images do
    match img with
    | ⟨patchRows, ⟨patchCols, patchGrid⟩⟩ =>
      let nTok := (patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size)
      let feats : T #[nTok, cfg.text_config.hidden_size] := getImageFeatures cfg m patchGrid
      acc? :=
        match acc? with
        | none =>
          some ⟨nTok, feats⟩
        | some ⟨nPrev, prev⟩ =>
          some ⟨nPrev + nTok, nn.cat prev feats 0⟩
  pure acc?

def repeatFeaturesForBatch {batch tokens : UInt64}
    (cfg : VLConfig)
    (features : T #[tokens, cfg.text_config.hidden_size])
    : ImageFeatures cfg :=
  Id.run do
    if batch <= 1 then
      return ⟨tokens, features⟩
    let mut acc : T #[tokens, cfg.text_config.hidden_size] := features
    let mut total := tokens
    for _ in [1:batch.toNat] do
      acc := nn.cat acc features 0
      total := total + tokens
    return ⟨total, acc⟩

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : UInt64}
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, hidden])
    (tokenId : UInt64)
    (features : T #[featTokens, featDim])
    : IO (T #[batch, seq, hidden]) := do
  if featDim != hidden then
    throw <| IO.userError
      s!"Gemma4 feature hidden size ({featDim}) does not match text hidden size ({hidden})"
  let tokenMask2d : T #[batch, seq] := eq_scalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"Gemma4 placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"
  let tokenMask : T #[batch, seq, hidden] :=
    nn.expand (reshape tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let src : T #[featTokens * featDim] := reshape features #[featTokens * featDim]
  pure (nn.masked_scatter inputsEmbeds tokenMask src)

private def buildInputEmbedsWithFeatures {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (imageFeatures : Option (ImageFeatures cfg) := none)
    : IO (T #[batch, seq, cfg.text_config.hidden_size] × PerLayerInputs cfg batch seq) := do
  let padId := cfg.text_config.pad_token_id.getD 0
  let llmInputIds : T #[batch, seq] := replaceTokenId inputIds cfg.image_token_id padId
  let llmInputsEmbeds : T #[batch, seq, cfg.text_config.hidden_size] :=
    m.language_model.embedTokens cfg.text_config llmInputIds
  let perLayerInputs :=
    Gemma4Model.computePerLayerInputs cfg.text_config m.language_model.model llmInputIds llmInputsEmbeds
  match imageFeatures with
  | some ⟨_n, feats⟩ =>
    let embeds ←
      scatterFeaturesIntoToken
        (inputIds := inputIds)
        (inputsEmbeds := llmInputsEmbeds)
        cfg.image_token_id
        feats
    pure (embeds, perLayerInputs)
  | none =>
    pure (llmInputsEmbeds, perLayerInputs)

private def buildVisionBidirectionalPrefillMasks {batch seq : UInt64}
    (cfg : VLConfig)
    (inputIds : T #[batch, seq])
    : IO (T #[batch, seq, seq] × T #[batch, seq, seq]) := do
  let padId := cfg.text_config.pad_token_id.getD 0
  let slidingWindow := cfg.text_config.sliding_window.toNat
  let mut slidingFlat : Array Int64 := #[]
  let mut fullFlat : Array Int64 := #[]

  for b in [:batch.toNat] do
    let row2 : T #[1, seq] := data.slice inputIds 0 b.toUInt64 1
    let row1 : T #[seq] := reshape (data.toLong row2) #[seq]
    let ids ← data.tensorToUInt64Array row1
    let mut groupIds : Array Int64 := Array.replicate ids.size (-1)
    let mut nextGroup : Int64 := 0

    for i in [:ids.size] do
      let tok := ids[i]!
      if tok == cfg.image_token_id then
        let prevIsImage :=
          i > 0 && ids[i - 1]! == cfg.image_token_id
        if prevIsImage then
          groupIds := groupIds.set! i (nextGroup - 1)
        else
          groupIds := groupIds.set! i nextGroup
          nextGroup := nextGroup + 1

    for q in [:ids.size] do
      let qTok := ids[q]!
      let qValid := qTok != padId
      let qGroup := groupIds[q]!
      for k in [:ids.size] do
        let kTok := ids[k]!
        let kValid := kTok != padId
        let baseFull := qValid && kValid && k <= q
        let baseSliding :=
          baseFull &&
          (slidingWindow == 0 || (q - k) < slidingWindow)
        let sameVisionGroup := qGroup >= 0 && qGroup == groupIds[k]!
        fullFlat := fullFlat.push (if baseFull || sameVisionGroup then 1 else 0)
        slidingFlat := slidingFlat.push (if baseSliding || sameVisionGroup then 1 else 0)

  let slidingMask : T #[batch, seq, seq] :=
    reshape (data.fromInt64Array slidingFlat) #[batch, seq, seq]
  let fullMask : T #[batch, seq, seq] :=
    reshape (data.fromInt64Array fullFlat) #[batch, seq, seq]
  pure (slidingMask.to inputIds.device, fullMask.to inputIds.device)

private def bidirectionalVisionPrefillMasks? {batch seq : UInt64}
    (cfg : VLConfig)
    (inputIds : T #[batch, seq])
    : IO (Option (T #[batch, seq, seq] × T #[batch, seq, seq])) := do
  if cfg.text_config.use_bidirectional_attention == "vision" then
    some <$> buildVisionBidirectionalPrefillMasks cfg inputIds
  else
    pure none

def forwardText {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.text_config.vocab_size] :=
  m.language_model.forward cfg.text_config inputIds

def forwardWithImageFeatures {batch seq featTokens : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (imageFeatures : T #[featTokens, cfg.text_config.hidden_size])
    : IO (T #[batch, seq, cfg.text_config.vocab_size]) := do
  let (inputsEmbeds, perLayerInputs) ←
    buildInputEmbedsWithFeatures cfg m inputIds (some ⟨featTokens, imageFeatures⟩)
  match (← bidirectionalVisionPrefillMasks? cfg inputIds) with
  | some (slidingMask, fullMask) =>
    pure <|
      m.language_model.forwardEmbedsWithPerLayerInputsBidirectionalVision
        cfg.text_config
        inputIds
        inputsEmbeds
        perLayerInputs
        slidingMask
        fullMask
  | none =>
    pure <|
      m.language_model.forwardEmbedsWithPerLayerInputs
        cfg.text_config
        inputIds
        inputsEmbeds
        perLayerInputs

def generate {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (ImageFeatures cfg) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let (inputsEmbeds, perLayerInputs) ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures
  let prefillMasks ← bidirectionalVisionPrefillMasks? cfg inputIds
  m.language_model.generateFromEmbedsWithPerLayerInputs
    cfg.text_config
    inputIds
    inputsEmbeds
    perLayerInputs
    maxNewTokens
    strategy
    eosTokenIds
    prefillMasks

def generateStream {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (ImageFeatures cfg) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let (inputsEmbeds, perLayerInputs) ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures
  let prefillMasks ← bidirectionalVisionPrefillMasks? cfg inputIds
  m.language_model.generateFromEmbedsStreamWithPerLayerInputs
    cfg.text_config
    inputIds
    inputsEmbeds
    perLayerInputs
    onStep
    maxNewTokens
    strategy
    eosTokenIds
    prefillMasks

end Gemma4ForConditionalGeneration

end torch.gemma4
