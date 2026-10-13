/-
  Tyr/Model/Qwen3TTS/SpeakerEncoder.lean

  Lean4 port of the Qwen3-TTS speaker encoder (ECAPA-TDNN style):
  - TimeDelayNetBlock (Conv1d + ReLU)
  - Res2Net block
  - Squeeze-Excitation block
  - Attentive statistics pooling
  - Final 1x1 projection to speaker embedding
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen3TTS.Config
import Tyr.Model.Utils

/-! # ECAPA-TDNN speaker encoder

Speaker encoder blocks: TDNN (Conv1d + ReLU), squeeze-excitation channel gating,
and Res2Net channel-group blocks, followed by attentive statistics pooling and a
1×1 projection to the speaker embedding.
-/

namespace torch.qwen3tts

open torch.Model

private def c0 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 0 512
private def c1 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 1 512
private def c2 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 2 512
private def c3 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 3 512
private def c4 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encChannels.getD 4 1536

private def k0 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 0 5
private def k1 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 1 3
private def k2 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 2 3
private def k3 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 3 3
private def k4 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encKernelSizes.getD 4 1

private def d0 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 0 1
private def d1 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 1 2
private def d2 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 2 3
private def d3 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 3 4
private def d4 (cfg : SpeakerEncoderConfig) : UInt64 := cfg.encDilations.getD 4 1

private def mfaInChannels (cfg : SpeakerEncoderConfig) : UInt64 := c1 cfg + c2 cfg + c3 cfg

private def samePadding (kernelSize dilation : UInt64) : UInt64 :=
  (dilation * (kernelSize - 1)) / 2

private def addBias3d {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (bias : T #[channels]) : T #[batch, channels, frames] :=
  let b : T #[1, channels, 1] := reshape bias #[1, channels, 1]
  let be : T #[batch, channels, frames] := nn.expand b #[batch, channels, frames]
  x + be

private def getOrFirst! [Inhabited α] (xs : Array α) (i : Nat) : α :=
  match xs[i]? with
  | some x => x
  | none =>
    match xs[0]? with
    | some x => x
    | none => panic! "empty array"

/-- TDNN block used throughout ECAPA speaker encoder: Conv1d + ReLU. -/
structure TimeDelayNetBlock (inChannels outChannels kernelSize dilation : UInt64) where
  weight : T #[outChannels, inChannels, kernelSize]
  bias : T #[outChannels]
  deriving TensorStruct, Inhabited

namespace TimeDelayNetBlock

def init (inChannels outChannels kernelSize dilation : UInt64)
    : IO (TimeDelayNetBlock inChannels outChannels kernelSize dilation) := do
  let weight ← initWeight #[outChannels, inChannels, kernelSize] inChannels
  let bias := initBias #[outChannels]
  pure { weight, bias }

def forward {batch frames inChannels outChannels kernelSize dilation : UInt64}
    (m : TimeDelayNetBlock inChannels outChannels kernelSize dilation)
    (x : T #[batch, inChannels, frames]) : T #[batch, outChannels, frames] :=
  let y : T #[batch, outChannels, frames] :=
    reshape (nn.conv1d x m.weight 1 (samePadding kernelSize dilation) dilation)
      #[batch, outChannels, frames]
  nn.relu (addBias3d y m.bias)

end TimeDelayNetBlock

/-- SE block: channel-wise gating from global temporal summary. -/
structure SqueezeExcitationBlock (channels seChannels : UInt64) where
  conv1Weight : T #[seChannels, channels, 1]
  conv1Bias : T #[seChannels]
  conv2Weight : T #[channels, seChannels, 1]
  conv2Bias : T #[channels]
  deriving TensorStruct

namespace SqueezeExcitationBlock

def init (channels seChannels : UInt64) : IO (SqueezeExcitationBlock channels seChannels) := do
  let conv1Weight ← initWeight #[seChannels, channels, 1] channels
  let conv2Weight ← initWeight #[channels, seChannels, 1] seChannels
  let conv1Bias := initBias #[seChannels]
  let conv2Bias := initBias #[channels]
  pure { conv1Weight, conv1Bias, conv2Weight, conv2Bias }

def forward {batch frames channels seChannels : UInt64}
    (m : SqueezeExcitationBlock channels seChannels)
    (x : T #[batch, channels, frames]) : T #[batch, channels, frames] :=
  let mean : T #[batch, channels, 1] := nn.meanDim x 2 true
  let h1 : T #[batch, seChannels, 1] :=
    reshape (nn.conv1d mean m.conv1Weight 1 0 1) #[batch, seChannels, 1]
  let h1 := nn.relu (addBias3d h1 m.conv1Bias)
  let h2 : T #[batch, channels, 1] :=
    reshape (nn.conv1d h1 m.conv2Weight 1 0 1) #[batch, channels, 1]
  let gate1 := nn.sigmoid (addBias3d h2 m.conv2Bias)
  let gate : T #[batch, channels, frames] := nn.expand gate1 #[batch, channels, frames]
  x * gate

end SqueezeExcitationBlock

/-- Res2Net block split over channel groups. -/
structure Res2NetBlock (channels scale kernelSize dilation : UInt64) where
  blocks : Array (TimeDelayNetBlock (channels / scale) (channels / scale) kernelSize dilation)
  deriving TensorStruct

namespace Res2NetBlock

def init (channels scale kernelSize dilation : UInt64)
    : IO (Res2NetBlock channels scale kernelSize dilation) := do
  let nBlocks := if scale.toNat <= 1 then 0 else scale.toNat - 1
  let mut blocks : Array (TimeDelayNetBlock (channels / scale) (channels / scale) kernelSize dilation) := #[]
  for _ in [:nBlocks] do
    let b ← TimeDelayNetBlock.init (channels / scale) (channels / scale) kernelSize dilation
    blocks := blocks.push b
  pure { blocks }

def forward {batch frames channels scale kernelSize dilation : UInt64}
    (m : Res2NetBlock channels scale kernelSize dilation)
    (x : T #[batch, channels, frames]) : T #[batch, channels, frames] :=
  if scale.toNat <= 1 then
    x
  else
    Id.run do
      let partChannels := channels / scale
      let mut parts : Array (T #[batch, partChannels, frames]) := #[]
      for i in [:scale.toNat] do
        let start := i.toUInt64 * partChannels
        let part : T #[batch, partChannels, frames] := data.slice x 1 start partChannels
        parts := parts.push part

      let mut outputs : Array (T #[batch, partChannels, frames]) := #[]
      let first := getOrFirst! parts 0
      outputs := outputs.push first
      let mut prev := first

      for i in [1:scale.toNat] do
        let part := getOrFirst! parts i
        let block := getOrFirst! m.blocks (i - 1)
        let input := if i == 1 then part else part + prev
        let out := TimeDelayNetBlock.forward block input
        outputs := outputs.push out
        prev := out

      let merged : T #[batch, channels, frames] := reshape (nn.cat_impl outputs 1) #[batch, channels, frames]
      merged

end Res2NetBlock

/-- ECAPA building block: TDNN -> Res2Net -> TDNN -> SE (+ residual). -/
structure SqueezeExcitationRes2NetBlock
    (inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64) where
  tdnn1 : TimeDelayNetBlock inChannels outChannels 1 1
  res2net : Res2NetBlock outChannels res2netScale kernelSize dilation
  tdnn2 : TimeDelayNetBlock outChannels outChannels 1 1
  seBlock : SqueezeExcitationBlock outChannels seChannels
  deriving TensorStruct

namespace SqueezeExcitationRes2NetBlock

def init (inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64)
    : IO (SqueezeExcitationRes2NetBlock inChannels outChannels res2netScale seChannels kernelSize dilation) := do
  let tdnn1 ← TimeDelayNetBlock.init inChannels outChannels 1 1
  let res2net ← Res2NetBlock.init outChannels res2netScale kernelSize dilation
  let tdnn2 ← TimeDelayNetBlock.init outChannels outChannels 1 1
  let seBlock ← SqueezeExcitationBlock.init outChannels seChannels
  pure { tdnn1, res2net, tdnn2, seBlock }

def forward {batch frames inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64}
    (m : SqueezeExcitationRes2NetBlock inChannels outChannels res2netScale seChannels kernelSize dilation)
    (x : T #[batch, inChannels, frames]) : T #[batch, outChannels, frames] :=
  let residual : T #[batch, outChannels, frames] := reshape x #[batch, outChannels, frames]
  let h1 := TimeDelayNetBlock.forward m.tdnn1 x
  let h2 := Res2NetBlock.forward m.res2net h1
  let h3 := TimeDelayNetBlock.forward m.tdnn2 h2
  let h4 := SqueezeExcitationBlock.forward m.seBlock h3
  h4 + residual

end SqueezeExcitationRes2NetBlock

/-- Attentive statistics pooling (mean+std with learned temporal attention). -/
structure AttentiveStatisticsPooling (channels attentionChannels : UInt64) where
  tdnn : TimeDelayNetBlock (channels * 3) attentionChannels 1 1
  convWeight : T #[channels, attentionChannels, 1]
  convBias : T #[channels]
  eps : Float := 1e-12
  deriving TensorStruct

namespace AttentiveStatisticsPooling

def init (channels attentionChannels : UInt64)
    : IO (AttentiveStatisticsPooling channels attentionChannels) := do
  let tdnn ← TimeDelayNetBlock.init (channels * 3) attentionChannels 1 1
  let convWeight ← initWeight #[channels, attentionChannels, 1] attentionChannels
  let convBias := initBias #[channels]
  pure { tdnn, convWeight, convBias }

def forward {batch frames channels attentionChannels : UInt64}
    (m : AttentiveStatisticsPooling channels attentionChannels)
    (x : T #[batch, channels, frames]) : T #[batch, channels * 2, 1] :=
  let mean : T #[batch, channels] := nn.meanDim x 2 false
  let mean1 : T #[batch, channels, 1] := reshape mean #[batch, channels, 1]
  let meanRep : T #[batch, channels, frames] := nn.expand mean1 #[batch, channels, frames]

  let diff := x - meanRep
  let var : T #[batch, channels] := nn.meanDim (diff * diff) 2 false
  let std : T #[batch, channels] := nn.sqrt (var + m.eps)
  let std1 : T #[batch, channels, 1] := reshape std #[batch, channels, 1]
  let stdRep : T #[batch, channels, frames] := nn.expand std1 #[batch, channels, frames]

  let attIn12 : T #[batch, channels + channels, frames] := nn.cat x meanRep 1
  let attIn : T #[batch, channels * 3, frames] := nn.cat attIn12 stdRep 1
  let att0 := TimeDelayNetBlock.forward m.tdnn attIn
  let att1 := nn.tanh att0
  let att2 : T #[batch, channels, frames] :=
    reshape (nn.conv1d att1 m.convWeight 1 0 1) #[batch, channels, frames]
  let att3 := addBias3d att2 m.convBias
  let attention : T #[batch, channels, frames] := nn.softmax_dim att3 2

  let weightedMean : T #[batch, channels] := nn.sumDim (attention * x) 2 false
  let weightedMean1 : T #[batch, channels, 1] := reshape weightedMean #[batch, channels, 1]
  let weightedMeanRep : T #[batch, channels, frames] := nn.expand weightedMean1 #[batch, channels, frames]

  let centered := x - weightedMeanRep
  let weightedVar : T #[batch, channels] := nn.sumDim (attention * (centered * centered)) 2 false
  let weightedStd : T #[batch, channels] := nn.sqrt (weightedVar + m.eps)

  let pooled : T #[batch, channels * 2] := nn.cat weightedMean weightedStd 1
  reshape pooled #[batch, channels * 2, 1]

end AttentiveStatisticsPooling

/-- Qwen3-TTS speaker encoder module. -/
structure SpeakerEncoder (cfg : SpeakerEncoderConfig) where
  tdnn0 : TimeDelayNetBlock cfg.melDim (c0 cfg) (k0 cfg) (d0 cfg)
  block1 : SqueezeExcitationRes2NetBlock
    (c0 cfg) (c1 cfg) cfg.encRes2NetScale cfg.encSeChannels (k1 cfg) (d1 cfg)
  block2 : SqueezeExcitationRes2NetBlock
    (c1 cfg) (c2 cfg) cfg.encRes2NetScale cfg.encSeChannels (k2 cfg) (d2 cfg)
  block3 : SqueezeExcitationRes2NetBlock
    (c2 cfg) (c3 cfg) cfg.encRes2NetScale cfg.encSeChannels (k3 cfg) (d3 cfg)
  mfa : TimeDelayNetBlock (mfaInChannels cfg) (c4 cfg) (k4 cfg) (d4 cfg)
  asp : AttentiveStatisticsPooling (c4 cfg) cfg.encAttentionChannels
  fcWeight : T #[cfg.encDim, (c4 cfg) * 2, 1]
  fcBias : T #[cfg.encDim]
  deriving TensorStruct

namespace SpeakerEncoder

def init (cfg : SpeakerEncoderConfig) : IO (SpeakerEncoder cfg) := do
  if cfg.encChannels.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_channels entries"
  if cfg.encKernelSizes.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_kernel_sizes entries"
  if cfg.encDilations.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_dilations entries"
  if cfg.encRes2NetScale < 2 then
    throw <| IO.userError "speaker encoder requires enc_res2net_scale >= 2"
  if c0 cfg != c1 cfg || c1 cfg != c2 cfg || c2 cfg != c3 cfg then
    throw <| IO.userError "speaker encoder expects enc_channels[0..3] to match for residual SE-Res2Net blocks"
  if c4 cfg != mfaInChannels cfg then
    throw <| IO.userError "speaker encoder expects enc_channels[4] = enc_channels[1] + enc_channels[2] + enc_channels[3]"

  let tdnn0 ← TimeDelayNetBlock.init cfg.melDim (c0 cfg) (k0 cfg) (d0 cfg)
  let block1 ← SqueezeExcitationRes2NetBlock.init
    (c0 cfg) (c1 cfg) cfg.encRes2NetScale cfg.encSeChannels (k1 cfg) (d1 cfg)
  let block2 ← SqueezeExcitationRes2NetBlock.init
    (c1 cfg) (c2 cfg) cfg.encRes2NetScale cfg.encSeChannels (k2 cfg) (d2 cfg)
  let block3 ← SqueezeExcitationRes2NetBlock.init
    (c2 cfg) (c3 cfg) cfg.encRes2NetScale cfg.encSeChannels (k3 cfg) (d3 cfg)
  let mfa ← TimeDelayNetBlock.init (mfaInChannels cfg) (c4 cfg) (k4 cfg) (d4 cfg)
  let asp ← AttentiveStatisticsPooling.init (c4 cfg) cfg.encAttentionChannels
  let fcWeight ← initWeight #[cfg.encDim, (c4 cfg) * 2, 1] ((c4 cfg) * 2)
  let fcBias := initBias #[cfg.encDim]

  pure { tdnn0, block1, block2, block3, mfa, asp, fcWeight, fcBias }

def forward {batch frames : UInt64}
    (m : SpeakerEncoder cfg)
    (mel : T #[batch, frames, cfg.melDim]) : T #[batch, cfg.encDim] :=
  let x0 : T #[batch, cfg.melDim, frames] := reshape (nn.transpose mel 1 2) #[batch, cfg.melDim, frames]
  let x1 := TimeDelayNetBlock.forward m.tdnn0 x0
  let x2 := SqueezeExcitationRes2NetBlock.forward m.block1 x1
  let x3 := SqueezeExcitationRes2NetBlock.forward m.block2 x2
  let x4 := SqueezeExcitationRes2NetBlock.forward m.block3 x3

  let agg12 : T #[batch, c1 cfg + c2 cfg, frames] := nn.cat x2 x3 1
  let agg123 : T #[batch, mfaInChannels cfg, frames] := nn.cat agg12 x4 1
  let mfaOut := TimeDelayNetBlock.forward m.mfa agg123
  let pooled := AttentiveStatisticsPooling.forward m.asp mfaOut

  let y0 : T #[batch, cfg.encDim, 1] :=
    reshape (nn.conv1d pooled m.fcWeight 1 0 1) #[batch, cfg.encDim, 1]
  let y1 := addBias3d y0 m.fcBias
  reshape y1 #[batch, cfg.encDim]

end SpeakerEncoder

instance {cfg : SpeakerEncoderConfig} {batch frames : UInt64} :
    Module (SpeakerEncoder cfg) (T #[batch, frames, cfg.melDim]) (T #[batch, cfg.encDim]) where
  forward := SpeakerEncoder.forward

end torch.qwen3tts
