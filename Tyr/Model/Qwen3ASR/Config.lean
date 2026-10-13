/-
  Tyr/Model/Qwen3ASR/Config.lean

  Configuration types for the Lean4 Qwen3-ASR port.
-/
import Tyr.Basic
import Tyr.Model.Qwen.Config

/-! # Qwen3-ASR configuration

`torch.qwen3asr.AudioEncoderConfig` and the text decoder config, mirroring the
reference classes, plus the frame-length arithmetic after each convolution
(`torch.qwen3asr.AudioEncoderConfig.framesAfterConv3`).
-/

namespace torch.qwen3asr

/-- Qwen3-ASR audio encoder config. Mirrors `Qwen3ASRAudioEncoderConfig`. -/
structure AudioEncoderConfig where
  numMelBins : UInt64 := 128
  encoderLayers : UInt64 := 32
  encoderAttentionHeads : UInt64 := 20
  encoderFfnDim : UInt64 := 5120
  dModel : UInt64 := 1280
  dropout : Float := 0.0
  attentionDropout : Float := 0.0
  activationFunction : String := "gelu"
  activationDropout : Float := 0.0
  scaleEmbedding : Bool := false
  initializerRange : Float := 0.02
  maxSourcePositions : UInt64 := 1500
  nWindow : UInt64 := 100
  outputDim : UInt64 := 3584
  nWindowInfer : UInt64 := 400
  convChunkSize : UInt64 := 500
  downsampleHiddenSize : UInt64 := 480
  deriving Repr, Inhabited

namespace AudioEncoderConfig

def downsampleOnce (n : UInt64) : UInt64 :=
  if n == 0 then 0 else ((n - 1) / 2) + 1

def downsampleTwice (n : UInt64) : UInt64 :=
  downsampleOnce (downsampleOnce n)

def downsampleThrice (n : UInt64) : UInt64 :=
  downsampleOnce (downsampleTwice n)

def melAfterConv1 (cfg : AudioEncoderConfig) : UInt64 :=
  downsampleOnce cfg.numMelBins

def melAfterConv2 (cfg : AudioEncoderConfig) : UInt64 :=
  downsampleTwice cfg.numMelBins

def melAfterConv3 (cfg : AudioEncoderConfig) : UInt64 :=
  downsampleThrice cfg.numMelBins

def framesAfterConv3 (_cfg : AudioEncoderConfig) (frames : UInt64) : UInt64 :=
  downsampleThrice frames

def convOutInDim (cfg : AudioEncoderConfig) : UInt64 :=
  cfg.downsampleHiddenSize * melAfterConv3 cfg

def headDim (cfg : AudioEncoderConfig) : UInt64 :=
  cfg.dModel / cfg.encoderAttentionHeads

/-- Reference `_get_feat_extract_output_lengths` port from Qwen3-ASR processor/modeling. -/
def featExtractOutputLength (inputLen : UInt64) : UInt64 :=
  let leave := inputLen.toNat % 100
  let feat :=
    if leave == 0 then 0
    else ((leave - 1) / 2) + 1
  let outLeave :=
    if feat == 0 then 0
    else ((((feat - 1) / 2) + 1 - 1) / 2) + 1
  let out := outLeave + (inputLen.toNat / 100) * 13
  out.toUInt64

def featExtractOutputLengths (inputLens : Array UInt64) : Array UInt64 :=
  inputLens.map featExtractOutputLength

end AudioEncoderConfig

/-- Qwen3-ASR text decoder config. Mirrors `Qwen3ASRTextConfig`. -/
structure TextConfig where
  vocabSize : UInt64 := 151936
  hiddenSize : UInt64 := 4096
  intermediateSize : UInt64 := 22016
  numHiddenLayers : UInt64 := 32
  numAttentionHeads : UInt64 := 32
  numKeyValueHeads : UInt64 := 32
  headDim : UInt64 := 128
  hiddenAct : String := "silu"
  maxPositionEmbeddings : UInt64 := 128000
  initializerRange : Float := 0.02
  rmsNormEps : Float := 1e-6
  useCache : Bool := true
  tieWordEmbeddings : Bool := false
  ropeTheta : Float := 5000000.0
  attentionBias : Bool := false
  attentionDropout : Float := 0.0
  deriving Repr, Inhabited

namespace TextConfig

def toQwenConfig (cfg : TextConfig) : qwen.QwenConfig := {
  vocab_size := cfg.vocabSize
  hidden_size := cfg.hiddenSize
  intermediate_size := cfg.intermediateSize
  num_hidden_layers := cfg.numHiddenLayers
  num_attention_heads := cfg.numAttentionHeads
  num_key_value_heads := cfg.numKeyValueHeads
  head_dim := cfg.headDim
  rope_theta := cfg.ropeTheta
  rms_norm_eps := cfg.rmsNormEps
  max_position_embeddings := cfg.maxPositionEmbeddings
}

end TextConfig

/-- Composite thinker config: audio encoder + text decoder. -/
structure ThinkerConfig where
  audioConfig : AudioEncoderConfig := {}
  textConfig : TextConfig := {}
  audioTokenId : UInt64 := 151646
  audioStartTokenId : UInt64 := 151647
  userTokenId : UInt64 := 872
  modelType : String := "qwen3_asr_thinker"
  classifyNum : UInt64 := 0
  initializerRange : Float := 0.02
  deriving Repr, Inhabited

namespace ThinkerConfig

private def containsSubstr (s pat : String) : Bool :=
  if pat.isEmpty then
    false
  else
    (s.splitOn pat).length > 1

/-- Forced-aligner checkpoints expose `model_type` containing `"forced_aligner"`. -/
def isForcedAligner (cfg : ThinkerConfig) : Bool :=
  containsSubstr cfg.modelType "forced_aligner"

/-- LM head output dim:
    - normal ASR thinker: text vocab size
    - forced aligner thinker: `classify_num` from config -/
def lmHeadOutDim (cfg : ThinkerConfig) : UInt64 :=
  if isForcedAligner cfg && cfg.classifyNum > 0 then
    cfg.classifyNum
  else
    cfg.textConfig.vocabSize

end ThinkerConfig

def defaultSupportedLanguages : Array String :=
  #[
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese",
    "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish",
    "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"
  ]

/-- Top-level Qwen3-ASR config. -/
structure Qwen3ASRConfig where
  thinkerConfig : ThinkerConfig := {}
  supportLanguages : Array String := defaultSupportedLanguages
  /-- Token ID used by forced-aligner prompt to mark timestamp slots. -/
  timestampTokenId : UInt64 := 0
  /-- Milliseconds represented by one forced-aligner timestamp bin. -/
  timestampSegmentTime : Float := 1.0
  deriving Repr, Inhabited

def defaultEosTokenIds : Array UInt64 := #[151645, 151643]

namespace Qwen3ASRConfig

def getTextConfig (cfg : Qwen3ASRConfig) : TextConfig :=
  cfg.thinkerConfig.textConfig

def getAudioConfig (cfg : Qwen3ASRConfig) : AudioEncoderConfig :=
  cfg.thinkerConfig.audioConfig

end Qwen3ASRConfig

end torch.qwen3asr
