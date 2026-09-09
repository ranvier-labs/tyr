/- 
  Tyr/Model/Qwen3TTS/SpeechTokenizerEncoder.lean

  Lean-native Qwen3-TTS 12Hz speech-tokenizer encoder (waveform -> codec IDs).
  This ports the upstream 12Hz encoder path built on Mimi:
  - causal SEANet-style conv encoder
  - causal transformer stack with RoPE + sliding-window attention
  - split RVQ encode (semantic + acoustic)

  Scope in this module:
  - pure Lean compute for encoder forward
  - real weight loading from `speech_tokenizer/model.safetensors`
  - first 16 code groups (1 semantic + 15 acoustic), matching
    Qwen3-TTS `encoder_valid_num_quantizers`.

  Input expected by `encode`:
  - mono waveform tensor `[batch, 1, samples]` at 24kHz.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3TTS.SpeechTokenizer

namespace torch.qwen3tts

private def requireTrue (ok : Bool) (msg : String) : IO Unit := do
  unless ok do
    throw (IO.userError msg)

private def freeze {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad t false

private def loadFrozen (path : String) (name : String) (s : Shape) : IO (T s) := do
  let t ← safetensors.loadTensor path name s
  pure (freeze t)

private def ceilDiv (x d : UInt64) : UInt64 :=
  if d == 0 then 0 else (x + d - 1) / d

private def framesAfterDown1 (samples : UInt64) : UInt64 := ceilDiv samples 4
private def framesAfterDown2 (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown1 samples) 5
private def framesAfterDown3 (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown2 samples) 6
def framesBeforeFinalDownsample (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown3 samples) 8
private def framesAfterDown4 (samples : UInt64) : UInt64 := framesBeforeFinalDownsample samples
def encodedFrames (samples : UInt64) : UInt64 := ceilDiv (framesAfterDown4 samples) 2

private def effectiveKernel (kernel dilation : UInt64) : UInt64 :=
  (kernel - 1) * dilation + 1

private def paddingTotal (kernel stride dilation : UInt64) : UInt64 :=
  effectiveKernel kernel dilation - stride

private def extraPaddingForCausalConv (inputLen kernel stride dilation : UInt64) : UInt64 :=
  let outLen := ceilDiv inputLen stride
  if outLen == 0 then
    0
  else
    let neededLen := (outLen - 1) * stride + effectiveKernel kernel dilation
    let baseLen := inputLen + paddingTotal kernel stride dilation
    if neededLen >= baseLen then neededLen - baseLen else 0

private def addBias3d {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (bias : T #[channels]) : T #[batch, channels, frames] :=
  let b : T #[1, channels, 1] := reshape bias #[1, channels, 1]
  x + nn.expand b #[batch, channels, frames]

private def scale3d {batch seq dim : UInt64}
    (x : T #[batch, seq, dim])
    (scale : T #[dim]) : T #[batch, seq, dim] :=
  let s : T #[1, 1, dim] := reshape scale #[1, 1, dim]
  x * nn.expand s #[batch, seq, dim]

private def clampMin1d {n : UInt64} (x : T #[n]) (minVal : Float) : T #[n] :=
  let minT : T #[n] := torch.full #[n] minVal false x.device
  where_ (lt_scalar x minVal) minT x

private def stack2dDyn {batch frames k : UInt64}
    (tensors : Array (T #[batch, frames])) : T #[k, batch, frames] :=
  if tensors.isEmpty then
    reshape (torch.zeros #[0, batch, frames]) #[k, batch, frames]
  else
    let unsq := tensors.map (fun t => nn.unsqueeze t 0)
    reshape (nn.cat_impl unsq 0) #[k, batch, frames]

/-! ## Parameters -/

structure EncoderResBlock (inC hiddenC : UInt64) where
  conv1Weight : T #[hiddenC, inC, 3]
  conv1Bias : T #[hiddenC]
  conv2Weight : T #[inC, hiddenC, 1]
  conv2Bias : T #[inC]
  deriving TensorStruct

structure EncoderTransformerLayer where
  inputLayerNormWeight : T #[512]
  inputLayerNormBias : T #[512]
  qProj : T #[512, 512]
  kProj : T #[512, 512]
  vProj : T #[512, 512]
  oProj : T #[512, 512]
  selfAttnLayerScale : T #[512]
  postAttentionLayerNormWeight : T #[512]
  postAttentionLayerNormBias : T #[512]
  fc1 : T #[2048, 512]
  fc2 : T #[512, 2048]
  mlpLayerScale : T #[512]
  deriving TensorStruct

structure EncoderCodebook where
  clusterUsage : T #[2048]
  embedSum : T #[2048, 256]
  deriving TensorStruct, Inhabited

structure EncodeRVQ where
  inputProj : T #[256, 512, 1]
  codebooks : Array EncoderCodebook
  deriving TensorStruct

structure SpeechTokenizer12HzEncoder where
  conv0Weight : T #[64, 1, 7]
  conv0Bias : T #[64]

  res1 : EncoderResBlock 64 32
  down1Weight : T #[128, 64, 8]
  down1Bias : T #[128]

  res2 : EncoderResBlock 128 64
  down2Weight : T #[256, 128, 10]
  down2Bias : T #[256]

  res3 : EncoderResBlock 256 128
  down3Weight : T #[512, 256, 12]
  down3Bias : T #[512]

  res4 : EncoderResBlock 512 256
  down4Weight : T #[1024, 512, 16]
  down4Bias : T #[1024]

  finalConvWeight : T #[512, 1024, 3]
  finalConvBias : T #[512]

  transformerLayers : Array EncoderTransformerLayer
  downsampleWeight : T #[512, 512, 4]

  semanticRVQ : EncodeRVQ
  acousticRVQ : EncodeRVQ

  inputSampleRate : UInt64 := 24000
  encodeDownsampleRate : UInt64 := 1920
  deriving TensorStruct

namespace SpeechTokenizer12HzEncoder

private def loadResBlock
    (weightsPath : String)
    (namePrefix : String)
    (inC hiddenC : UInt64)
    : IO (EncoderResBlock inC hiddenC) := do
  let conv1Weight ← loadFrozen weightsPath s!"{namePrefix}.1.conv.weight" #[hiddenC, inC, 3]
  let conv1Bias ← loadFrozen weightsPath s!"{namePrefix}.1.conv.bias" #[hiddenC]
  let conv2Weight ← loadFrozen weightsPath s!"{namePrefix}.3.conv.weight" #[inC, hiddenC, 1]
  let conv2Bias ← loadFrozen weightsPath s!"{namePrefix}.3.conv.bias" #[inC]
  pure { conv1Weight, conv1Bias, conv2Weight, conv2Bias }

private def loadTransformerLayer (weightsPath : String) (i : Nat) : IO EncoderTransformerLayer := do
  let p := s!"encoder.encoder_transformer.layers.{i}"
  let inputLayerNormWeight ← loadFrozen weightsPath s!"{p}.input_layernorm.weight" #[512]
  let inputLayerNormBias ← loadFrozen weightsPath s!"{p}.input_layernorm.bias" #[512]
  let qProj ← loadFrozen weightsPath s!"{p}.self_attn.q_proj.weight" #[512, 512]
  let kProj ← loadFrozen weightsPath s!"{p}.self_attn.k_proj.weight" #[512, 512]
  let vProj ← loadFrozen weightsPath s!"{p}.self_attn.v_proj.weight" #[512, 512]
  let oProj ← loadFrozen weightsPath s!"{p}.self_attn.o_proj.weight" #[512, 512]
  let selfAttnLayerScale ← loadFrozen weightsPath s!"{p}.self_attn_layer_scale.scale" #[512]
  let postAttentionLayerNormWeight ← loadFrozen weightsPath s!"{p}.post_attention_layernorm.weight" #[512]
  let postAttentionLayerNormBias ← loadFrozen weightsPath s!"{p}.post_attention_layernorm.bias" #[512]
  let fc1 ← loadFrozen weightsPath s!"{p}.mlp.fc1.weight" #[2048, 512]
  let fc2 ← loadFrozen weightsPath s!"{p}.mlp.fc2.weight" #[512, 2048]
  let mlpLayerScale ← loadFrozen weightsPath s!"{p}.mlp_layer_scale.scale" #[512]
  pure {
    inputLayerNormWeight
    inputLayerNormBias
    qProj
    kProj
    vProj
    oProj
    selfAttnLayerScale
    postAttentionLayerNormWeight
    postAttentionLayerNormBias
    fc1
    fc2
    mlpLayerScale
  }

private def loadEncoderCodebook (weightsPath : String) (namePrefix : String) : IO EncoderCodebook := do
  let clusterUsage ← loadFrozen weightsPath s!"{namePrefix}.cluster_usage" #[2048]
  let embedSum ← loadFrozen weightsPath s!"{namePrefix}.embed_sum" #[2048, 256]
  pure { clusterUsage, embedSum }

private def loadEncodeRVQ
    (weightsPath : String)
    (namePrefix : String)
    (numCodebooks : Nat)
    : IO EncodeRVQ := do
  let inputProj ← loadFrozen weightsPath s!"{namePrefix}.input_proj.weight" #[256, 512, 1]
  let mut codebooks : Array EncoderCodebook := #[]
  for i in [:numCodebooks] do
    let cb ← loadEncoderCodebook weightsPath s!"{namePrefix}.layers.{i}.codebook"
    codebooks := codebooks.push cb
  pure { inputProj, codebooks }

/-- Load 12Hz speech-tokenizer encoder weights from `speech_tokenizer` directory. -/
def loadFromDir (speechTokenizerDir : String) (device : Device := Device.CPU) : IO SpeechTokenizer12HzEncoder := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  let counts ← cfg.encoderCodebookCounts
  requireTrue (counts == (1, 15))
    "Fixed 16-group encoder requires encoder_valid_num_quantizers=16 and one semantic quantizer"

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"

  let conv0Weight ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
  let conv0Bias ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.bias" #[64]
  let res1 ← loadResBlock weightsPath "encoder.encoder.layers.1.block" 64 32
  let down1Weight ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.weight" #[128, 64, 8]
  let down1Bias ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.bias" #[128]

  let res2 ← loadResBlock weightsPath "encoder.encoder.layers.4.block" 128 64
  let down2Weight ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.weight" #[256, 128, 10]
  let down2Bias ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.bias" #[256]

  let res3 ← loadResBlock weightsPath "encoder.encoder.layers.7.block" 256 128
  let down3Weight ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.weight" #[512, 256, 12]
  let down3Bias ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.bias" #[512]

  let res4 ← loadResBlock weightsPath "encoder.encoder.layers.10.block" 512 256
  let down4Weight ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.weight" #[1024, 512, 16]
  let down4Bias ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.bias" #[1024]

  let finalConvWeight ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.weight" #[512, 1024, 3]
  let finalConvBias ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.bias" #[512]

  let mut transformerLayers : Array EncoderTransformerLayer := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    transformerLayers := transformerLayers.push layer

  let downsampleWeight ← loadFrozen weightsPath "encoder.downsample.conv.weight" #[512, 512, 4]
  let semanticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.semantic_residual_vector_quantizer" 1
  -- Match Qwen3-TTS `encoder_valid_num_quantizers=16`: first 15 acoustic groups.
  let acousticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.acoustic_residual_vector_quantizer" 15

  let enc : SpeechTokenizer12HzEncoder := {
    conv0Weight
    conv0Bias
    res1
    down1Weight
    down1Bias
    res2
    down2Weight
    down2Bias
    res3
    down3Weight
    down3Bias
    res4
    down4Weight
    down4Bias
    finalConvWeight
    finalConvBias
    transformerLayers
    downsampleWeight
    semanticRVQ
    acousticRVQ
    inputSampleRate := 24000
    encodeDownsampleRate := 1920
  }
  pure (TensorStruct.map (fun t => t.to device) enc)

/-- Load 12Hz-family speech-tokenizer encoder weights while allowing
    tokenizer variants with different quantizer counts. -/
def loadFromDirFlexible (speechTokenizerDir : String) (device : Device := Device.CPU) : IO SpeechTokenizer12HzEncoder := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  let (semanticLayers, acousticLayers) ← cfg.encoderCodebookCounts

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"

  let conv0Weight ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.weight" #[64, 1, 7]
  let conv0Bias ← loadFrozen weightsPath "encoder.encoder.layers.0.conv.bias" #[64]
  let res1 ← loadResBlock weightsPath "encoder.encoder.layers.1.block" 64 32
  let down1Weight ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.weight" #[128, 64, 8]
  let down1Bias ← loadFrozen weightsPath "encoder.encoder.layers.3.conv.bias" #[128]

  let res2 ← loadResBlock weightsPath "encoder.encoder.layers.4.block" 128 64
  let down2Weight ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.weight" #[256, 128, 10]
  let down2Bias ← loadFrozen weightsPath "encoder.encoder.layers.6.conv.bias" #[256]

  let res3 ← loadResBlock weightsPath "encoder.encoder.layers.7.block" 256 128
  let down3Weight ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.weight" #[512, 256, 12]
  let down3Bias ← loadFrozen weightsPath "encoder.encoder.layers.9.conv.bias" #[512]

  let res4 ← loadResBlock weightsPath "encoder.encoder.layers.10.block" 512 256
  let down4Weight ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.weight" #[1024, 512, 16]
  let down4Bias ← loadFrozen weightsPath "encoder.encoder.layers.12.conv.bias" #[1024]

  let finalConvWeight ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.weight" #[512, 1024, 3]
  let finalConvBias ← loadFrozen weightsPath "encoder.encoder.layers.14.conv.bias" #[512]

  let mut transformerLayers : Array EncoderTransformerLayer := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    transformerLayers := transformerLayers.push layer

  let downsampleWeight ← loadFrozen weightsPath "encoder.downsample.conv.weight" #[512, 512, 4]
  let semanticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.semantic_residual_vector_quantizer" semanticLayers
  let acousticRVQ ← loadEncodeRVQ weightsPath "encoder.quantizer.acoustic_residual_vector_quantizer" acousticLayers

  let enc : SpeechTokenizer12HzEncoder := {
    conv0Weight
    conv0Bias
    res1
    down1Weight
    down1Bias
    res2
    down2Weight
    down2Bias
    res3
    down3Weight
    down3Bias
    res4
    down4Weight
    down4Bias
    finalConvWeight
    finalConvBias
    transformerLayers
    downsampleWeight
    semanticRVQ
    acousticRVQ
    inputSampleRate := cfg.inputSampleRate
    encodeDownsampleRate := cfg.encodeDownsampleRate
  }
  pure (TensorStruct.map (fun t => t.to device) enc)

def numSemanticCodeGroups (m : SpeechTokenizer12HzEncoder) : UInt64 :=
  m.semanticRVQ.codebooks.size.toUInt64

def numAcousticCodeGroups (m : SpeechTokenizer12HzEncoder) : UInt64 :=
  m.acousticRVQ.codebooks.size.toUInt64

def numCodeGroups (m : SpeechTokenizer12HzEncoder) : UInt64 :=
  numSemanticCodeGroups m + numAcousticCodeGroups m

private def causalConvConstant {batch inC outC inLen outLen kernel : UInt64}
    (x : T #[batch, inC, inLen])
    (weight : T #[outC, inC, kernel])
    (bias : Option (T #[outC]) := none)
    (stride : UInt64 := 1)
    (dilation : UInt64 := 1)
    : T #[batch, outC, outLen] :=
  let leftPad : UInt64 := paddingTotal kernel stride dilation
  let extraPad : UInt64 := extraPaddingForCausalConv inLen kernel stride dilation
  let left : T #[batch, inC, leftPad] := torch.zeros #[batch, inC, leftPad] false x.device
  let right : T #[batch, inC, extraPad] := torch.zeros #[batch, inC, extraPad] false x.device
  let xPad0 : T #[batch, inC, inLen + leftPad] := nn.cat left x 2
  let xPad : T #[batch, inC, inLen + leftPad + extraPad] := nn.cat xPad0 right 2
  let y0 : T #[batch, outC, outLen] :=
    reshape (nn.conv1d xPad weight stride 0 dilation) #[batch, outC, outLen]
  match bias with
  | some b => addBias3d y0 b
  | none => y0

private def causalConvReplicate {batch inC outC inLen outLen kernel : UInt64}
    (x : T #[batch, inC, inLen])
    (weight : T #[outC, inC, kernel])
    (bias : Option (T #[outC]) := none)
    (stride : UInt64 := 1)
    (dilation : UInt64 := 1)
    : T #[batch, outC, outLen] :=
  let leftPad : UInt64 := paddingTotal kernel stride dilation
  let extraPad : UInt64 := extraPaddingForCausalConv inLen kernel stride dilation
  let first : T #[batch, inC, 1] := data.slice x 2 0 1
  let last : T #[batch, inC, 1] := data.slice x 2 (inLen - 1) 1
  let left : T #[batch, inC, leftPad] := nn.expand first #[batch, inC, leftPad]
  let right : T #[batch, inC, extraPad] :=
    if extraPad == 0 then
      torch.zeros #[batch, inC, extraPad] false x.device
    else
      nn.expand last #[batch, inC, extraPad]
  let xPad0 : T #[batch, inC, inLen + leftPad] := nn.cat left x 2
  let xPad : T #[batch, inC, inLen + leftPad + extraPad] := nn.cat xPad0 right 2
  let y0 : T #[batch, outC, outLen] :=
    reshape (nn.conv1d xPad weight stride 0 dilation) #[batch, outC, outLen]
  match bias with
  | some b => addBias3d y0 b
  | none => y0

private def forwardResBlock {batch frames inC hiddenC : UInt64}
    (blk : EncoderResBlock inC hiddenC)
    (x : T #[batch, inC, frames]) : T #[batch, inC, frames] :=
  let residual := x
  let h0 : T #[batch, inC, frames] := nn.elu x
  let h1 : T #[batch, hiddenC, frames] :=
    causalConvConstant h0 blk.conv1Weight (some blk.conv1Bias) 1 1
  let h2 : T #[batch, hiddenC, frames] := nn.elu h1
  let h3 : T #[batch, inC, frames] :=
    causalConvConstant h2 blk.conv2Weight (some blk.conv2Bias) 1 1
  residual + h3

private def forwardAttention {batch seq : UInt64}
    (layer : EncoderTransformerLayer)
    (x : T #[batch, seq, 512])
    (cos sin : T #[seq, 32]) : T #[batch, seq, 512] :=
  let q0 : T #[batch, seq, 512] := linear3d x layer.qProj
  let k0 : T #[batch, seq, 512] := linear3d x layer.kProj
  let v0 : T #[batch, seq, 512] := linear3d x layer.vProj

  let q : T #[batch, seq, 8, 64] := reshape q0 #[batch, seq, 8, 64]
  let k : T #[batch, seq, 8, 64] := reshape k0 #[batch, seq, 8, 64]
  let v : T #[batch, seq, 8, 64] := reshape v0 #[batch, seq, 8, 64]

  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  let qh : T #[batch, 8, seq, 64] := nn.transpose_for_attention q
  let kh : T #[batch, 8, seq, 64] := nn.transpose_for_attention k
  let vh : T #[batch, 8, seq, 64] := nn.transpose_for_attention v

  let attn : T #[batch, 8, seq, 64] :=
    nn.scaledDotProductAttentionGQAWindow qh kh vh 0.0 true true 250
  let attn : T #[batch, seq, 8, 64] := nn.transpose_from_attention attn
  let attn : T #[batch, seq, 512] := reshape attn #[batch, seq, 512]
  linear3d attn layer.oProj

private def forwardTransformerLayer {batch seq : UInt64}
    (layer : EncoderTransformerLayer)
    (x : T #[batch, seq, 512])
    (cos sin : T #[seq, 32])
    : T #[batch, seq, 512] :=
  let residual1 := x
  let h1 : T #[batch, seq, 512] :=
    nn.layer_norm x layer.inputLayerNormWeight layer.inputLayerNormBias 1e-5
  let h2 : T #[batch, seq, 512] := forwardAttention layer h1 cos sin
  let h3 : T #[batch, seq, 512] := residual1 + scale3d h2 layer.selfAttnLayerScale

  let residual2 := h3
  let h4 : T #[batch, seq, 512] :=
    nn.layer_norm h3 layer.postAttentionLayerNormWeight layer.postAttentionLayerNormBias 1e-5
  let h5 : T #[batch, seq, 2048] := linear3d h4 layer.fc1
  let h6 : T #[batch, seq, 2048] := nn.gelu h5
  let h7 : T #[batch, seq, 512] := linear3d h6 layer.fc2
  residual2 + scale3d h7 layer.mlpLayerScale

private def codebookEmbedding (cb : EncoderCodebook) : T #[2048, 256] :=
  let usage : T #[2048] := clampMin1d cb.clusterUsage 1e-5
  let usage2d : T #[2048, 256] := nn.expand (reshape usage #[2048, 1]) #[2048, 256]
  nn.div cb.embedSum usage2d

private def quantizeWithCodebook {batch frames : UInt64}
    (cb : EncoderCodebook)
    (x : T #[batch, 256, frames])
    : T #[batch, frames] × T #[batch, 256, frames] :=
  let embed : T #[2048, 256] := codebookEmbedding cb
  let xFlat : T #[batch * frames, 256] :=
    reshape (nn.transpose x 1 2) #[batch * frames, 256]

  let x2 : T #[batch * frames, 1] := nn.sumDim (nn.pow xFlat 2.0) 1 true
  let e2 : T #[2048, 1] := nn.sumDim (nn.pow embed 2.0) 1 true
  let dot : T #[batch * frames, 2048] := nn.mm xFlat (nn.transpose2d embed)
  let x2e : T #[batch * frames, 2048] := nn.expand x2 #[batch * frames, 2048]
  let e2Row : T #[1, 2048] := reshape (nn.transpose2d e2) #[1, 2048]
  let e2e : T #[batch * frames, 2048] := nn.expand e2Row #[batch * frames, 2048]
  let dist : T #[batch * frames, 2048] := x2e + e2e - mul_scalar dot 2.0

  let idxFlat : T #[batch * frames] := nn.argmax (mul_scalar dist (-1.0)) 1
  let quantFlat : T #[batch * frames, 256] := nn.embedding1d idxFlat embed
  let quant : T #[batch, 256, frames] :=
    reshape (nn.transpose (reshape quantFlat #[batch, frames, 256]) 1 2) #[batch, 256, frames]
  let idx : T #[batch, frames] := reshape idxFlat #[batch, frames]
  (idx, quant)

private def rvqEncode {batch frames k : UInt64}
    (rvq : EncodeRVQ)
    (embeddings : T #[batch, 512, frames])
    (numLayers : Nat)
    : T #[k, batch, frames] := Id.run do
  let mut residual : T #[batch, 256, frames] :=
    reshape (nn.conv1d embeddings rvq.inputProj 1 0 1) #[batch, 256, frames]
  let mut allIndices : Array (T #[batch, frames]) := #[]
  for i in [:numLayers] do
    let cb := rvq.codebooks[i]!
    let (idx, quantized) := quantizeWithCodebook cb residual
    residual := residual - quantized
    allIndices := allIndices.push idx
  stack2dDyn (batch := batch) (frames := frames) (k := k) allIndices

/-- Encode waveform `[batch,1,samples]` to codec IDs `[batch,16,encodedFrames(samples)]`.
    `ropeStartPos` is in the pre-downsample transformer timeline (stride 960 samples). -/
def encodeWithRoPEOffset {batch samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : T #[batch, 1, samples])
    (ropeStartPos : UInt64 := 0)
    : T #[batch, 16, encodedFrames samples] :=
  let t1 := framesAfterDown1 samples
  let t2 := framesAfterDown2 samples
  let t3 := framesAfterDown3 samples
  let t4 := framesAfterDown4 samples
  let t5 := encodedFrames samples

  let h0 : T #[batch, 64, samples] := causalConvConstant audio m.conv0Weight (some m.conv0Bias) 1 1
  let h1 : T #[batch, 64, samples] := forwardResBlock m.res1 h0

  let h2In : T #[batch, 64, samples] := nn.elu h1
  let h2 : T #[batch, 128, t1] := causalConvConstant h2In m.down1Weight (some m.down1Bias) 4 1
  let h3 : T #[batch, 128, t1] := forwardResBlock m.res2 h2

  let h4In : T #[batch, 128, t1] := nn.elu h3
  let h4 : T #[batch, 256, t2] := causalConvConstant h4In m.down2Weight (some m.down2Bias) 5 1
  let h5 : T #[batch, 256, t2] := forwardResBlock m.res3 h4

  let h6In : T #[batch, 256, t2] := nn.elu h5
  let h6 : T #[batch, 512, t3] := causalConvConstant h6In m.down3Weight (some m.down3Bias) 6 1
  let h7 : T #[batch, 512, t3] := forwardResBlock m.res4 h6

  let h8In : T #[batch, 512, t3] := nn.elu h7
  let h8 : T #[batch, 1024, t4] := causalConvConstant h8In m.down4Weight (some m.down4Bias) 8 1
  let h9In : T #[batch, 1024, t4] := nn.elu h8
  let h9 : T #[batch, 512, t4] := causalConvConstant h9In m.finalConvWeight (some m.finalConvBias) 1 1

  let x0 : T #[batch, t4, 512] := reshape (nn.transpose h9 1 2) #[batch, t4, 512]
  let ropeTotal : UInt64 := ropeStartPos + t4
  let (cosAll, sinAll) := rotary.computeFreqsPure ropeTotal 64 10000.0
  let cosAll : T #[ropeTotal, 32] :=
    if cosAll.device == x0.device then cosAll else cosAll.to x0.device
  let sinAll : T #[ropeTotal, 32] :=
    if sinAll.device == x0.device then sinAll else sinAll.to x0.device
  let cos : T #[t4, 32] := data.slice cosAll 0 ropeStartPos t4
  let sin : T #[t4, 32] := data.slice sinAll 0 ropeStartPos t4
  let x1 := m.transformerLayers.foldl (fun h layer => forwardTransformerLayer layer h cos sin) x0
  let x2 : T #[batch, 512, t4] := reshape (nn.transpose x1 1 2) #[batch, 512, t4]

  let x3 : T #[batch, 512, t5] := causalConvReplicate x2 m.downsampleWeight none 2 1

  let semanticCodes : T #[1, batch, t5] := rvqEncode (k := 1) m.semanticRVQ x3 1
  let acousticCodes : T #[15, batch, t5] := rvqEncode (k := 15) m.acousticRVQ x3 15
  let codes : T #[16, batch, t5] := nn.cat semanticCodes acousticCodes 0
  reshape (nn.transpose codes 0 1) #[batch, 16, t5]

/-- Encode waveform `[batch,1,samples]` to tokenizer-configured codec IDs
    `[batch,codeGroups,encodedFrames(samples)]`.
    `ropeStartPos` is in the pre-downsample transformer timeline (stride 960 samples). -/
def encodeWithRoPEOffsetDynamic {batch samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : T #[batch, 1, samples])
    (ropeStartPos : UInt64 := 0)
    : Sigma (fun codeGroups => T #[batch, codeGroups, encodedFrames samples]) :=
  let t1 := framesAfterDown1 samples
  let t2 := framesAfterDown2 samples
  let t3 := framesAfterDown3 samples
  let t4 := framesAfterDown4 samples
  let t5 := encodedFrames samples

  let h0 : T #[batch, 64, samples] := causalConvConstant audio m.conv0Weight (some m.conv0Bias) 1 1
  let h1 : T #[batch, 64, samples] := forwardResBlock m.res1 h0

  let h2In : T #[batch, 64, samples] := nn.elu h1
  let h2 : T #[batch, 128, t1] := causalConvConstant h2In m.down1Weight (some m.down1Bias) 4 1
  let h3 : T #[batch, 128, t1] := forwardResBlock m.res2 h2

  let h4In : T #[batch, 128, t1] := nn.elu h3
  let h4 : T #[batch, 256, t2] := causalConvConstant h4In m.down2Weight (some m.down2Bias) 5 1
  let h5 : T #[batch, 256, t2] := forwardResBlock m.res3 h4

  let h6In : T #[batch, 256, t2] := nn.elu h5
  let h6 : T #[batch, 512, t3] := causalConvConstant h6In m.down3Weight (some m.down3Bias) 6 1
  let h7 : T #[batch, 512, t3] := forwardResBlock m.res4 h6

  let h8In : T #[batch, 512, t3] := nn.elu h7
  let h8 : T #[batch, 1024, t4] := causalConvConstant h8In m.down4Weight (some m.down4Bias) 8 1
  let h9In : T #[batch, 1024, t4] := nn.elu h8
  let h9 : T #[batch, 512, t4] := causalConvConstant h9In m.finalConvWeight (some m.finalConvBias) 1 1

  let x0 : T #[batch, t4, 512] := reshape (nn.transpose h9 1 2) #[batch, t4, 512]
  let ropeTotal : UInt64 := ropeStartPos + t4
  let (cosAll, sinAll) := rotary.computeFreqsPure ropeTotal 64 10000.0
  let cosAll : T #[ropeTotal, 32] :=
    if cosAll.device == x0.device then cosAll else cosAll.to x0.device
  let sinAll : T #[ropeTotal, 32] :=
    if sinAll.device == x0.device then sinAll else sinAll.to x0.device
  let cos : T #[t4, 32] := data.slice cosAll 0 ropeStartPos t4
  let sin : T #[t4, 32] := data.slice sinAll 0 ropeStartPos t4
  let x1 := m.transformerLayers.foldl (fun h layer => forwardTransformerLayer layer h cos sin) x0
  let x2 : T #[batch, 512, t4] := reshape (nn.transpose x1 1 2) #[batch, 512, t4]

  let x3 : T #[batch, 512, t5] := causalConvReplicate x2 m.downsampleWeight none 2 1

  let semLayers : Nat := m.semanticRVQ.codebooks.size
  let acLayers : Nat := m.acousticRVQ.codebooks.size
  let semK : UInt64 := semLayers.toUInt64
  let acK : UInt64 := acLayers.toUInt64
  let semanticCodes : T #[semK, batch, t5] := rvqEncode (k := semK) m.semanticRVQ x3 semLayers
  let acousticCodes : T #[acK, batch, t5] := rvqEncode (k := acK) m.acousticRVQ x3 acLayers
  let codes : T #[semK + acK, batch, t5] := nn.cat semanticCodes acousticCodes 0
  let out : T #[batch, semK + acK, t5] := reshape (nn.transpose codes 0 1) #[batch, semK + acK, t5]
  ⟨semK + acK, out⟩

/-- Encode waveform `[batch,1,samples]` with zero RoPE offset. -/
def encode {batch samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : T #[batch, 1, samples]) : T #[batch, 16, encodedFrames samples] :=
  encodeWithRoPEOffset m audio 0

/-- Convenience encode for mono waveform `[samples]` into frame-major `[frames,16]`.
    RoPE offset is in the pre-downsample transformer timeline. -/
def encodeMonoFrameMajorWithRoPEOffset {samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : T #[samples])
    (ropeStartPos : UInt64 := 0)
    : T #[encodedFrames samples, 16] :=
  let x0 : T #[1, 1, samples] := reshape audio #[1, 1, samples]
  let codes : T #[1, 16, encodedFrames samples] := encodeWithRoPEOffset m x0 ropeStartPos
  let c0 : T #[1, encodedFrames samples, 16] := reshape (nn.transpose codes 1 2) #[1, encodedFrames samples, 16]
  reshape c0 #[encodedFrames samples, 16]

/-- Convenience encode for mono waveform `[samples]` into frame-major
    `[frames,codeGroups]` with tokenizer-configured code-group count.
    RoPE offset is in the pre-downsample transformer timeline. -/
def encodeMonoFrameMajorWithRoPEOffsetDynamic {samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : T #[samples])
    (ropeStartPos : UInt64 := 0)
    : Sigma (fun codeGroups => T #[encodedFrames samples, codeGroups]) :=
  let x0 : T #[1, 1, samples] := reshape audio #[1, 1, samples]
  let ⟨codeGroups, codes⟩ := encodeWithRoPEOffsetDynamic m x0 ropeStartPos
  let c0 : T #[1, encodedFrames samples, codeGroups] :=
    reshape (nn.transpose codes 1 2) #[1, encodedFrames samples, codeGroups]
  ⟨codeGroups, reshape c0 #[encodedFrames samples, codeGroups]⟩

/-- Convenience encode for mono waveform `[samples]` into frame-major `[frames,16]`. -/
def encodeMonoFrameMajor {samples : UInt64}
    (m : SpeechTokenizer12HzEncoder)
    (audio : T #[samples]) : T #[encodedFrames samples, 16] :=
  encodeMonoFrameMajorWithRoPEOffset m audio 0

/-- Stateful streaming encode state for incremental waveform -> codec conversion. -/
structure EncodeStreamState where
  chunkSamples : Nat
  leftContextSamples : Nat
  offsetSamples : Nat := 0
  history : Array Float := #[]

/-- Initialize streaming encode state. -/
def initEncodeStreamState
    (chunkSamples : Nat := 240000)
    (leftContextSamples : Nat := 288000)
    : EncodeStreamState :=
  { chunkSamples, leftContextSamples, offsetSamples := 0, history := #[] }

/-- Encode one streaming chunk and emit flat codec IDs (`rows * 16` values). -/
def pushEncodeStream
    (m : SpeechTokenizer12HzEncoder)
    (st : EncodeStreamState)
    (newSamples : Array Float)
    : IO (EncodeStreamState × Array UInt64) := do
  if newSamples.isEmpty then
    return (st, #[])

  let ctxStart : Nat := st.offsetSamples - st.history.size
  let chunk := st.history ++ newSamples
  let chunkSamplesU64 : UInt64 := chunk.size.toUInt64
  let audioChunk0 : T #[chunkSamplesU64] := reshape (data.fromFloatArray chunk) #[chunkSamplesU64]
  let audioChunk : T #[chunkSamplesU64] :=
    if audioChunk0.device == m.conv0Weight.device then audioChunk0 else audioChunk0.to m.conv0Weight.device
  let framesChunk : UInt64 := encodedFrames chunkSamplesU64
  let ropeStartPos : UInt64 := framesBeforeFinalDownsample ctxStart.toUInt64
  let codesChunk : T #[framesChunk, 16] := encodeMonoFrameMajorWithRoPEOffset m audioChunk ropeStartPos
  let flatChunk : T #[framesChunk * 16] := reshape codesChunk #[framesChunk * 16]
  let valsChunk ← data.tensorToUInt64Array flatChunk

  let dropRows : Nat := (encodedFrames st.history.size.toUInt64).toNat
  let keepRows : Nat := (encodedFrames newSamples.size.toUInt64).toNat
  let startIdx : Nat := dropRows * 16
  let endIdx : Nat := Nat.min valsChunk.size ((dropRows + keepRows) * 16)
  let valsKeep := valsChunk.extract startIdx endIdx

  let nextOffset := st.offsetSamples + newSamples.size
  let histStart :=
    if chunk.size > st.leftContextSamples then
      chunk.size - st.leftContextSamples
    else
      0
  let nextHist := chunk.extract histStart chunk.size
  let st' : EncodeStreamState := {
    st with
      offsetSamples := nextOffset
      history := nextHist
  }
  return (st', valsKeep)

/-- Encode one streaming chunk and emit flat codec IDs (`rows * codeGroups` values). -/
def pushEncodeStreamDynamic
    (m : SpeechTokenizer12HzEncoder)
    (st : EncodeStreamState)
    (newSamples : Array Float)
    : IO (EncodeStreamState × UInt64 × Array UInt64) := do
  if newSamples.isEmpty then
    return (st, numCodeGroups m, #[])

  let ctxStart : Nat := st.offsetSamples - st.history.size
  let chunk := st.history ++ newSamples
  let chunkSamplesU64 : UInt64 := chunk.size.toUInt64
  let audioChunk0 : T #[chunkSamplesU64] := reshape (data.fromFloatArray chunk) #[chunkSamplesU64]
  let audioChunk : T #[chunkSamplesU64] :=
    if audioChunk0.device == m.conv0Weight.device then audioChunk0 else audioChunk0.to m.conv0Weight.device
  let framesChunk : UInt64 := encodedFrames chunkSamplesU64
  let ropeStartPos : UInt64 := framesBeforeFinalDownsample ctxStart.toUInt64
  let ⟨codeGroups, codesChunk⟩ := encodeMonoFrameMajorWithRoPEOffsetDynamic m audioChunk ropeStartPos
  let flatChunk : T #[framesChunk * codeGroups] := reshape codesChunk #[framesChunk * codeGroups]
  let valsChunk ← data.tensorToUInt64Array flatChunk

  let groupsNat : Nat := codeGroups.toNat
  let dropRows : Nat := (encodedFrames st.history.size.toUInt64).toNat
  let keepRows : Nat := (encodedFrames newSamples.size.toUInt64).toNat
  let startIdx : Nat := dropRows * groupsNat
  let endIdx : Nat := Nat.min valsChunk.size ((dropRows + keepRows) * groupsNat)
  let valsKeep := valsChunk.extract startIdx endIdx

  let nextOffset := st.offsetSamples + newSamples.size
  let histStart :=
    if chunk.size > st.leftContextSamples then
      chunk.size - st.leftContextSamples
    else
      0
  let nextHist := chunk.extract histStart chunk.size
  let st' : EncodeStreamState := {
    st with
      offsetSamples := nextOffset
      history := nextHist
  }
  return (st', codeGroups, valsKeep)

end SpeechTokenizer12HzEncoder

end torch.qwen3tts
