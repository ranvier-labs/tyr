/-
  Tyr/Model/KittenTTS/Config.lean

  Configuration for a Lean4 KittenTTS port.
  This mirrors the upstream model at the level needed for inference:
  - ALBERT-style phoneme encoder
  - duration / prosody predictor
  - AdaIN decoder
  - lightweight generator configuration
 -/
import Tyr.Basic

/-! # KittenTTS configuration

`torch.kittentts.AlbertConfig` for the text encoder and
`torch.kittentts.GeneratorConfig` for the waveform generator, with derived sizes
such as `torch.kittentts.AlbertConfig.headDim` and
`torch.kittentts.GeneratorConfig.totalUpsample`.
-/

namespace torch.kittentts

structure AlbertConfig where
  numHiddenLayers : UInt64 := 12
  numAttentionHeads : UInt64 := 12
  hiddenSize : UInt64 := 768
  intermediateSize : UInt64 := 3072
  maxPositionEmbeddings : UInt64 := 512
  embeddingSize : UInt64 := 128
  innerGroupNum : UInt64 := 1
  numHiddenGroups : UInt64 := 1
  typeVocabSize : UInt64 := 2
  layerNormEps : Float := 1e-12
  hiddenDropoutProb : Float := 0.0
  attentionDropoutProb : Float := 0.0
  deriving Repr, Inhabited

namespace AlbertConfig

def headDim (cfg : AlbertConfig) : UInt64 :=
  if cfg.numAttentionHeads == 0 then 0 else cfg.hiddenSize / cfg.numAttentionHeads

def allHeadSize (cfg : AlbertConfig) : UInt64 :=
  cfg.numAttentionHeads * cfg.headDim

def layersPerGroup (cfg : AlbertConfig) : UInt64 :=
  if cfg.numHiddenGroups == 0 then 0 else cfg.numHiddenLayers / cfg.numHiddenGroups

end AlbertConfig

structure GeneratorConfig where
  resblockKernelSizes : Array UInt64 := #[3, 7, 11]
  upsampleRates : Array UInt64 := #[5, 5, 4, 3]
  upsampleInitialChannel : UInt64 := 512
  resblockDilationSizes : Array (Array UInt64) := #[#[1, 3, 5], #[1, 3, 5], #[1, 3, 5]]
  upsampleKernelSizes : Array UInt64 := #[10, 10, 8, 6]
  genIstftNFft : UInt64 := 20
  genIstftHopSize : UInt64 := 5
  harmonicCount : UInt64 := 8
  deriving Repr, Inhabited

namespace GeneratorConfig

private def pow2 : Nat → UInt64
  | 0 => 1
  | n + 1 => 2 * pow2 n

def totalUpsample (cfg : GeneratorConfig) : UInt64 :=
  cfg.upsampleRates.foldl (fun acc x => acc * x) 1

def sourceUpsample (cfg : GeneratorConfig) : UInt64 :=
  totalUpsample cfg * cfg.genIstftHopSize

def freqBins (cfg : GeneratorConfig) : UInt64 :=
  cfg.genIstftNFft / 2 + 1

def stftFeatureChannels (cfg : GeneratorConfig) : UInt64 :=
  2 * freqBins cfg

def finalChannels (cfg : GeneratorConfig) : UInt64 :=
  if cfg.upsampleRates.isEmpty then
    cfg.upsampleInitialChannel
  else
    cfg.upsampleInitialChannel / pow2 cfg.upsampleRates.size

private def prodPrefix (xs : Array UInt64) (count : Nat) : UInt64 :=
  Id.run do
    let mut acc := 1
    for i in [:min count xs.size] do
      acc := acc * xs.getD i 1
    acc

def prefixUpsample (cfg : GeneratorConfig) (count : Nat) : UInt64 :=
  prodPrefix cfg.upsampleRates count

def suffixUpsample (cfg : GeneratorConfig) (start : Nat) : UInt64 :=
  Id.run do
    let mut acc := 1
    for i in [start:cfg.upsampleRates.size] do
      acc := acc * cfg.upsampleRates.getD i 1
    acc

def stageInChannels (cfg : GeneratorConfig) (idx : Nat) : UInt64 :=
  cfg.upsampleInitialChannel / pow2 idx

def stageOutChannels (cfg : GeneratorConfig) (idx : Nat) : UInt64 :=
  cfg.upsampleInitialChannel / pow2 (idx + 1)

def stageInFrames (cfg : GeneratorConfig) (baseFrames : UInt64) (idx : Nat) : UInt64 :=
  baseFrames * prefixUpsample cfg idx

def stageUpsampledFrames (cfg : GeneratorConfig) (baseFrames : UInt64) (idx : Nat) : UInt64 :=
  baseFrames * prefixUpsample cfg (idx + 1)

def harFrames (cfg : GeneratorConfig) (baseFrames : UInt64) : UInt64 :=
  totalUpsample cfg * baseFrames + 1

def stageOutFrames (cfg : GeneratorConfig) (baseFrames : UInt64) (idx : Nat) : UInt64 :=
  if idx + 1 < cfg.upsampleRates.size then
    stageUpsampledFrames cfg baseFrames idx
  else
    harFrames cfg baseFrames

def waveSamples (cfg : GeneratorConfig) (baseFrames : UInt64) : UInt64 :=
  baseFrames * sourceUpsample cfg

def noiseStride (cfg : GeneratorConfig) (idx : Nat) : UInt64 :=
  if idx + 1 < cfg.upsampleRates.size then
    suffixUpsample cfg (idx + 1)
  else
    1

def noiseKernel (cfg : GeneratorConfig) (idx : Nat) : UInt64 :=
  if idx + 1 < cfg.upsampleRates.size then
    2 * noiseStride cfg idx
  else
    1

def noisePadding (cfg : GeneratorConfig) (idx : Nat) : UInt64 :=
  if idx + 1 < cfg.upsampleRates.size then
    (noiseStride cfg idx + 1) / 2
  else
    0

end GeneratorConfig

structure KittenTTSConfig where
  hiddenDim : UInt64 := 512
  maxConvDim : UInt64 := 512
  maxDur : UInt64 := 50
  nLayer : UInt64 := 3
  nMels : UInt64 := 80
  nToken : UInt64 := 178
  styleDim : UInt64 := 128
  textEncoderKernelSize : UInt64 := 5
  asrResDim : UInt64 := 64
  decoderOutDim : UInt64 := 0
  sampleRate : UInt64 := 24000
  plbert : AlbertConfig := {}
  generator : GeneratorConfig := {}
  deriving Repr, Inhabited

namespace KittenTTSConfig

def fullStyleDim (cfg : KittenTTSConfig) : UInt64 :=
  2 * cfg.styleDim

def decoderChannels (cfg : KittenTTSConfig) : UInt64 :=
  if cfg.decoderOutDim == 0 then cfg.generator.upsampleInitialChannel else cfg.decoderOutDim

end KittenTTSConfig

end torch.kittentts
