/-
  Tyr/Model/KittenTTS/Model.lean

  Lean4 KittenTTS port focused on single-utterance inference.

  The text / duration / prosody / decoder / iSTFT vocoder stack mirrors the
  upstream Kokoro / StyleTTS2-style architecture closely enough for single
  utterance inference, including harmonic source generation and inverse STFT.
 -/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.KittenTTS.Config
import Tyr.Model.Utils

/-! # KittenTTS model

Forward pass of KittenTTS: linear/convolution helpers, the ALBERT text encoder,
prosody prediction and the upsampling waveform generator.
-/

namespace torch.kittentts

open torch.Model

private def initUniform (shape : Shape) (scale : Float) : IO (T shape) := do
  let w ← torch.uniform shape (-scale) scale
  pure (autograd.set_requires_grad w true)

private def xavierScale (dim : Nat) : Float :=
  1.0 / Float.sqrt (max 1 dim).toFloat

private def affineShift {s : Shape} (normed g b : T s) : T s :=
  (normed * add_scalar g 1.0) + b

private def addBias2d {batch dim : UInt64}
    (x : T #[batch, dim])
    (b : T #[dim])
    : T #[batch, dim] :=
  x + nn.expand (reshape b #[1, dim]) #[batch, dim]

private def addBias3d {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (b : T #[channels])
    : T #[batch, channels, frames] :=
  let be : T #[batch, channels, frames] := nn.expand (reshape b #[1, channels, 1]) #[batch, channels, frames]
  x + be

private def seqToCF {batch seq channels : UInt64}
    (x : T #[batch, seq, channels])
    : T #[batch, channels, seq] :=
  reshape (nn.transpose x 1 2) #[batch, channels, seq]

private def cfToSeq {batch channels seq : UInt64}
    (x : T #[batch, channels, seq])
    : T #[batch, seq, channels] :=
  reshape (nn.transpose x 1 2) #[batch, seq, channels]

private def repeatNearest1d {batch channels frames scale : UInt64}
    (x : T #[batch, channels, frames])
    : T #[batch, channels, frames * scale] :=
  reshape (torch.interpolate_scale x #[scale.toFloat] "nearest") #[batch, channels, frames * scale]

private def resize1d {batch channels inFrames outFrames : UInt64}
    (x : T #[batch, channels, inFrames])
    (mode : String := "linear")
    : T #[batch, channels, outFrames] :=
  reshape (torch.interpolate x #[outFrames] mode false) #[batch, channels, outFrames]

private def layerNormCF {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (weight bias : T #[channels])
    (eps : Float := 1e-5)
    : T #[batch, channels, frames] :=
  seqToCF (nn.layer_norm (cfToSeq x) weight bias eps)

private def instanceNormCF {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (eps : Float := 1e-5)
    : T #[batch, channels, frames] :=
  let mean : T #[batch, channels, 1] := nn.meanDim x 2 true
  let meanE : T #[batch, channels, frames] := nn.expand mean #[batch, channels, frames]
  let centered : T #[batch, channels, frames] := x - meanE
  let var : T #[batch, channels, 1] := nn.meanDim (centered * centered) 2 true
  let invStd : T #[batch, channels, 1] := torch.rsqrt (var + eps)
  let invStdE : T #[batch, channels, frames] := nn.expand invStd #[batch, channels, frames]
  centered * invStdE

private def styleExpand {batch dim frames : UInt64}
    (style : T #[batch, dim])
    : T #[batch, dim, frames] :=
  nn.expand (reshape style #[batch, dim, 1]) #[batch, dim, frames]

private def snake1d {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (alpha : T #[1, channels, 1])
    : T #[batch, channels, frames] :=
  let a : T #[batch, channels, frames] := nn.expand alpha #[batch, channels, frames]
  let inv : T #[batch, channels, frames] := nn.div (torch.ones #[batch, channels, frames] false x.device) a
  x + inv * nn.pow (nn.sin (x * a)) 2.0

private def geluApprox {s : Shape} (x : T s) : T s :=
  let cubic := nn.pow x 3.0
  let inner := x + mul_scalar cubic 0.044715
  let t := nn.tanh (mul_scalar inner 0.7978846)
  mul_scalar x 0.5 * add_scalar t 1.0

private def safeIndex {α} [Inhabited α] (xs : Array α) (idx : Nat) : α :=
  match xs[idx]? with
  | some x => x
  | none => default

private def durationToFrames (vals : Array Float) (speed : Float) : Array UInt64 :=
  vals.map fun v =>
    let denom := if speed <= 0.0 then 1.0 else speed
    let raw := ((v / denom) + 0.5).toUInt64
    if raw == 0 then 1 else raw

private def buildAlignment {seq frames : UInt64}
    (durations : Array UInt64)
    (device : Device := Device.CPU)
    : T #[1, seq, frames] := Id.run do
  if frames == 0 then
    reshape (torch.zeros #[1, seq, frames] false device) #[1, seq, frames]
  else
    let mut flat : Array Float := Array.replicate (seq * frames).toNat 0.0
    let mut cursor : Nat := 0
    for i in [:seq.toNat] do
      let dur := durations.getD i 1
      for _ in [:dur.toNat] do
        if cursor < frames.toNat then
          let idx := i * frames.toNat + cursor
          flat := flat.set! idx 1.0
          cursor := cursor + 1
    (reshape (data.fromFloatArray flat) #[1, seq, frames]).to device

/-- Linear layer with explicit weight / bias tensors. -/
structure LinearNorm (inDim outDim : UInt64) where
  weight : T #[outDim, inDim]
  bias : T #[outDim]
  deriving TensorStruct, Inhabited

namespace LinearNorm

def init (inDim outDim : UInt64) : IO (LinearNorm inDim outDim) := do
  let scale := xavierScale outDim.toNat
  let weight ← initUniform #[outDim, inDim] scale
  let bias := initBias #[outDim]
  pure { weight, bias }

def forward2d {batch inDim outDim : UInt64}
    (m : LinearNorm inDim outDim)
    (x : T #[batch, inDim])
    : T #[batch, outDim] :=
  affine x m.weight m.bias

def forward3d {batch seq inDim outDim : UInt64}
    (m : LinearNorm inDim outDim)
    (x : T #[batch, seq, inDim])
    : T #[batch, seq, outDim] :=
  affine3d x m.weight m.bias

end LinearNorm

structure Conv1dParams
    (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weight : T #[outC, inC, kernel]
  bias : T #[outC]
  deriving TensorStruct, Inhabited

namespace Conv1dParams

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (Conv1dParams inC outC kernel stride padding dilation) := do
  let weight ← initWeight #[outC, inC, kernel] (inC * kernel)
  let bias := initBias #[outC]
  pure { weight, bias }

def forward {batch frames inC outC kernel stride padding dilation : UInt64}
    (m : Conv1dParams inC outC kernel stride padding dilation)
    (x : T #[batch, inC, frames])
    : T #[batch, outC, convOutputSize frames kernel stride padding dilation] :=
  let y0 : T #[batch, outC, convOutputSize frames kernel stride padding dilation] :=
    reshape (nn.conv1d x m.weight stride padding dilation) #[batch, outC, convOutputSize frames kernel stride padding dilation]
  addBias3d y0 m.bias

end Conv1dParams

structure DepthwiseConv1dParams
    (channels kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weight : T #[channels, 1, kernel]
  bias : T #[channels]
  deriving TensorStruct, Inhabited

namespace DepthwiseConv1dParams

def init (channels kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (DepthwiseConv1dParams channels kernel stride padding dilation) := do
  let weight ← initWeight #[channels, 1, kernel] kernel
  let bias := initBias #[channels]
  pure { weight, bias }

def forward {batch frames channels kernel stride padding dilation : UInt64}
    (m : DepthwiseConv1dParams channels kernel stride padding dilation)
    (x : T #[batch, channels, frames])
    : T #[batch, channels, convOutputSize frames kernel stride padding dilation] :=
  reshape
    (nn.conv1d_group_bias x m.weight m.bias stride padding dilation channels)
    #[batch, channels, convOutputSize frames kernel stride padding dilation]

end DepthwiseConv1dParams

structure DepthwiseTransConv1dParams
    (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1) where
  weight : T #[channels, 1, kernel]
  bias : T #[channels]
  deriving TensorStruct, Inhabited

namespace DepthwiseTransConv1dParams

def init (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1)
    : IO (DepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation) := do
  let weight ← initWeight #[channels, 1, kernel] kernel
  let bias := initBias #[channels]
  pure { weight, bias }

def forward {batch frames channels kernel stride padding outputPadding dilation : UInt64}
    (m : DepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation)
    (x : T #[batch, channels, frames])
    : T #[batch, channels, convTransposeOutputSize frames kernel stride padding outputPadding dilation] := Id.run do
  let outFrames := convTransposeOutputSize frames kernel stride padding outputPadding dilation
  let mut parts : Array (T #[]) := #[]
  for ci in [:channels.toNat] do
    let xc : T #[batch, 1, frames] := data.slice x 1 ci.toUInt64 1
    let wc : T #[1, 1, kernel] := data.slice m.weight 0 ci.toUInt64 1
    let bc : T #[1] := data.slice m.bias 0 ci.toUInt64 1
    let yc : T #[batch, 1, outFrames] :=
      reshape (nn.conv_transpose1d_bias xc wc bc stride padding outputPadding dilation) #[batch, 1, outFrames]
    parts := parts.push (nn.eraseShape yc)
  reshape (nn.cat_dyn parts 1) #[batch, channels, outFrames]

end DepthwiseTransConv1dParams

private def normDim0_3d {d0 d1 d2 : UInt64}
    (x : T #[d0, d1, d2])
    : T #[d0] :=
  let sq : T #[d0, d1, d2] := x * x
  let s2 : T #[d0, d1, 1] := nn.sumDim sq 2 true
  let s1 : T #[d0, 1, 1] := nn.sumDim s2 1 true
  reshape (nn.sqrt (s1 + (1e-12 : Float))) #[d0]

private def applyWeightNormConv1d {outC inC kernel : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG : T #[outC])
    : T #[outC, inC, kernel] :=
  let denom : T #[outC, 1, 1] := reshape (normDim0_3d weightV) #[outC, 1, 1]
  let denomE : T #[outC, inC, kernel] := nn.expand denom #[outC, inC, kernel]
  let gain : T #[outC, 1, 1] := reshape weightG #[outC, 1, 1]
  let gainE : T #[outC, inC, kernel] := nn.expand gain #[outC, inC, kernel]
  nn.div (weightV * gainE) denomE

private def applyWeightNormTransConv1d {inC outC kernel : UInt64}
    (weightV : T #[inC, outC, kernel])
    (weightG : T #[inC])
    : T #[inC, outC, kernel] :=
  let denom : T #[inC, 1, 1] := reshape (normDim0_3d weightV) #[inC, 1, 1]
  let denomE : T #[inC, outC, kernel] := nn.expand denom #[inC, outC, kernel]
  let gain : T #[inC, 1, 1] := reshape weightG #[inC, 1, 1]
  let gainE : T #[inC, outC, kernel] := nn.expand gain #[inC, outC, kernel]
  nn.div (weightV * gainE) denomE

structure WNConv1dParams
    (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weightV : T #[outC, inC, kernel]
  weightG : T #[outC]
  bias : T #[outC]
  deriving TensorStruct, Inhabited

namespace WNConv1dParams

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (WNConv1dParams inC outC kernel stride padding dilation) := do
  let weightV ← initWeight #[outC, inC, kernel] (inC * kernel)
  let weightG : T #[outC] := autograd.set_requires_grad (normDim0_3d weightV) true
  let bias := initBias #[outC]
  pure { weightV, weightG, bias }

def forward {batch frames inC outC kernel stride padding dilation : UInt64}
    (m : WNConv1dParams inC outC kernel stride padding dilation)
    (x : T #[batch, inC, frames])
    : T #[batch, outC, convOutputSize frames kernel stride padding dilation] :=
  let weight := applyWeightNormConv1d m.weightV m.weightG
  let y0 : T #[batch, outC, convOutputSize frames kernel stride padding dilation] :=
    reshape (nn.conv1d x weight stride padding dilation) #[batch, outC, convOutputSize frames kernel stride padding dilation]
  addBias3d y0 m.bias

end WNConv1dParams

structure WNConvTranspose1dParams
    (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (outputPadding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weightV : T #[inC, outC, kernel]
  weightG : T #[inC]
  bias : T #[outC]
  deriving TensorStruct, Inhabited

namespace WNConvTranspose1dParams

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (outputPadding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation) := do
  let weightV ← initWeight #[inC, outC, kernel] (max 1 (outC * kernel).toNat |>.toUInt64)
  let weightG : T #[inC] := autograd.set_requires_grad (normDim0_3d weightV) true
  let bias := initBias #[outC]
  pure { weightV, weightG, bias }

def forward {batch frames inC outC kernel stride padding outputPadding dilation : UInt64}
    (m : WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation)
    (x : T #[batch, inC, frames])
    : T #[batch, outC, convTransposeOutputSize frames kernel stride padding outputPadding dilation] :=
  let weight := applyWeightNormTransConv1d m.weightV m.weightG
  reshape
    (nn.conv_transpose1d_bias x weight m.bias stride padding outputPadding dilation)
    #[batch, outC, convTransposeOutputSize frames kernel stride padding outputPadding dilation]

end WNConvTranspose1dParams

structure WNDepthwiseTransConv1dParams
    (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1) where
  weightV : T #[channels, 1, kernel]
  weightG : T #[channels]
  bias : T #[channels]
  deriving TensorStruct, Inhabited

namespace WNDepthwiseTransConv1dParams

def init (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1)
    : IO (WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation) := do
  let weightV ← initWeight #[channels, 1, kernel] kernel
  let weightG : T #[channels] := autograd.set_requires_grad (normDim0_3d weightV) true
  let bias := initBias #[channels]
  pure { weightV, weightG, bias }

def forward {batch frames channels kernel stride padding outputPadding dilation : UInt64}
    (m : WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation)
    (x : T #[batch, channels, frames])
    : T #[batch, channels, convTransposeOutputSize frames kernel stride padding outputPadding dilation] := Id.run do
  let outFrames := convTransposeOutputSize frames kernel stride padding outputPadding dilation
  let weight : T #[channels, 1, kernel] := applyWeightNormTransConv1d m.weightV m.weightG
  let mut parts : Array (T #[]) := #[]
  for ci in [:channels.toNat] do
    let xc : T #[batch, 1, frames] := data.slice x 1 ci.toUInt64 1
    let wc : T #[1, 1, kernel] := data.slice weight 0 ci.toUInt64 1
    let bc : T #[1] := data.slice m.bias 0 ci.toUInt64 1
    let yc : T #[batch, 1, outFrames] :=
      reshape (nn.conv_transpose1d_bias xc wc bc stride padding outputPadding dilation) #[batch, 1, outFrames]
    parts := parts.push (nn.eraseShape yc)
  reshape (nn.cat_dyn parts 1) #[batch, channels, outFrames]

end WNDepthwiseTransConv1dParams

structure BiLSTM (inputSize hiddenSize : UInt64) where
  wxForward : T #[4 * hiddenSize, inputSize]
  whForward : T #[4 * hiddenSize, hiddenSize]
  biasIHForward : T #[4 * hiddenSize]
  biasHHForward : T #[4 * hiddenSize]
  wxBackward : T #[4 * hiddenSize, inputSize]
  whBackward : T #[4 * hiddenSize, hiddenSize]
  biasIHBackward : T #[4 * hiddenSize]
  biasHHBackward : T #[4 * hiddenSize]
  deriving TensorStruct, Inhabited

namespace BiLSTM

def init (inputSize hiddenSize : UInt64) : IO (BiLSTM inputSize hiddenSize) := do
  let scale := xavierScale hiddenSize.toNat
  let wxForward ← initUniform #[4 * hiddenSize, inputSize] scale
  let whForward ← initUniform #[4 * hiddenSize, hiddenSize] scale
  let biasIHForward ← initUniform #[4 * hiddenSize] scale
  let biasHHForward ← initUniform #[4 * hiddenSize] scale
  let wxBackward ← initUniform #[4 * hiddenSize, inputSize] scale
  let whBackward ← initUniform #[4 * hiddenSize, hiddenSize] scale
  let biasIHBackward ← initUniform #[4 * hiddenSize] scale
  let biasHHBackward ← initUniform #[4 * hiddenSize] scale
  pure {
    wxForward, whForward, biasIHForward, biasHHForward,
    wxBackward, whBackward, biasIHBackward, biasHHBackward
  }

private def step
    {hidden input : UInt64}
    (x : T #[1, input])
    (h c : T #[1, hidden])
    (wx : T #[4 * hidden, input])
    (wh : T #[4 * hidden, hidden])
    (bih bhh : T #[4 * hidden])
    : T #[1, hidden] × T #[1, hidden] :=
  let gatesIH : T #[1, 4 * hidden] := affine x wx bih
  let gatesHH : T #[1, 4 * hidden] := affine h wh bhh
  let gates : T #[1, 4 * hidden] := gatesIH + gatesHH

  let iRaw : T #[1, hidden] := data.slice gates 1 0 hidden
  let fRaw : T #[1, hidden] := data.slice gates 1 hidden hidden
  let gRaw : T #[1, hidden] := data.slice gates 1 (2 * hidden) hidden
  let oRaw : T #[1, hidden] := data.slice gates 1 (3 * hidden) hidden

  let iGate : T #[1, hidden] := nn.sigmoid iRaw
  let fGate : T #[1, hidden] := nn.sigmoid fRaw
  let gGate : T #[1, hidden] := nn.tanh gRaw
  let oGate : T #[1, hidden] := nn.sigmoid oRaw

  let cNext : T #[1, hidden] := fGate * c + iGate * gGate
  let hNext : T #[1, hidden] := oGate * nn.tanh cNext
  (hNext, cNext)

private def runDirection {seq input hidden : UInt64}
    (x : T #[1, seq, input])
    (wx : T #[4 * hidden, input])
    (wh : T #[4 * hidden, hidden])
    (bih bhh : T #[4 * hidden])
    (reverse : Bool := false)
    : T #[1, seq, hidden] := Id.run do
  let mut h : T #[1, hidden] := torch.zeros #[1, hidden] false x.device
  let mut c : T #[1, hidden] := torch.zeros #[1, hidden] false x.device
  let mut outs : Array (T #[]) := #[]
  let indices :=
    if reverse then
      (List.range seq.toNat).reverse
    else
      List.range seq.toNat
  for idx in indices do
    let xt3 : T #[1, 1, input] := data.slice x 1 idx.toUInt64 1
    let xt : T #[1, input] := reshape xt3 #[1, input]
    let (hNext, cNext) := step xt h c wx wh bih bhh
    h := hNext
    c := cNext
    let pushed : T #[1, 1, hidden] := reshape hNext #[1, 1, hidden]
    outs := outs.push (nn.eraseShape pushed)
  let ordered := if reverse then outs.reverse else outs
  reshape (nn.cat_dyn ordered 1) #[1, seq, hidden]

def forward {seq input hidden : UInt64}
    (m : BiLSTM input hidden)
    (x : T #[1, seq, input])
    : T #[1, seq, 2 * hidden] :=
  let fw := runDirection x m.wxForward m.whForward m.biasIHForward m.biasHHForward false
  let bw := runDirection x m.wxBackward m.whBackward m.biasIHBackward m.biasHHBackward true
  nn.cat fw bw 2

end BiLSTM

structure AlbertEmbeddings (nToken : UInt64) (cfg : AlbertConfig) where
  wordEmbeddings : T #[nToken, cfg.embeddingSize]
  positionEmbeddings : T #[cfg.maxPositionEmbeddings, cfg.embeddingSize]
  tokenTypeEmbeddings : T #[cfg.typeVocabSize, cfg.embeddingSize]
  layerNormWeight : T #[cfg.embeddingSize]
  layerNormBias : T #[cfg.embeddingSize]
  deriving TensorStruct, Inhabited

namespace AlbertEmbeddings

def init (nToken : UInt64) (cfg : AlbertConfig) : IO (AlbertEmbeddings nToken cfg) := do
  let scale := xavierScale cfg.embeddingSize.toNat
  let wordEmbeddings ← initUniform #[nToken, cfg.embeddingSize] scale
  let positionEmbeddings ← initUniform #[cfg.maxPositionEmbeddings, cfg.embeddingSize] scale
  let tokenTypeEmbeddings ← initUniform #[cfg.typeVocabSize, cfg.embeddingSize] scale
  let layerNormWeight ← initUniform #[cfg.embeddingSize] 0.02
  let layerNormBias := initBias #[cfg.embeddingSize]
  pure { wordEmbeddings, positionEmbeddings, tokenTypeEmbeddings, layerNormWeight, layerNormBias }

def forward {seq nToken : UInt64}
    (m : AlbertEmbeddings nToken cfg)
    (inputIds : T #[1, seq])
    : T #[1, seq, cfg.embeddingSize] :=
  let tokTypes : T #[1, seq] := torch.full_int #[1, seq] 0
  let posIds1d : T #[seq] := torch.arange 0 seq 1
  let posIds : T #[1, seq] := reshape posIds1d #[1, seq]
  let word : T #[1, seq, cfg.embeddingSize] := nn.embedding inputIds m.wordEmbeddings
  let pos : T #[1, seq, cfg.embeddingSize] := nn.embedding posIds m.positionEmbeddings
  let typ : T #[1, seq, cfg.embeddingSize] := nn.embedding tokTypes m.tokenTypeEmbeddings
  nn.layer_norm (word + pos + typ) m.layerNormWeight m.layerNormBias cfg.layerNormEps

end AlbertEmbeddings

structure KittenAlbertSelfAttention (cfg : AlbertConfig) where
  query : LinearNorm cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  key : LinearNorm cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  value : LinearNorm cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  dense : LinearNorm (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  layerNormWeight : T #[cfg.hiddenSize]
  layerNormBias : T #[cfg.hiddenSize]
  deriving TensorStruct, Inhabited

namespace KittenAlbertSelfAttention

def init (cfg : AlbertConfig) : IO (KittenAlbertSelfAttention cfg) := do
  let query ← LinearNorm.init cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let key ← LinearNorm.init cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let value ← LinearNorm.init cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let dense ← LinearNorm.init (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  let layerNormWeight ← initUniform #[cfg.hiddenSize] 0.02
  let layerNormBias := initBias #[cfg.hiddenSize]
  pure { query, key, value, dense, layerNormWeight, layerNormBias }

def forward {seq : UInt64}
    (m : KittenAlbertSelfAttention cfg)
    (x : T #[1, seq, cfg.hiddenSize])
    : T #[1, seq, cfg.hiddenSize] :=
  let q0 : T #[1, seq, AlbertConfig.allHeadSize cfg] := m.query.forward3d x
  let k0 : T #[1, seq, AlbertConfig.allHeadSize cfg] := m.key.forward3d x
  let v0 : T #[1, seq, AlbertConfig.allHeadSize cfg] := m.value.forward3d x

  let q : T #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg] :=
    reshape q0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]
  let k : T #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg] :=
    reshape k0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]
  let v : T #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg] :=
    reshape v0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]

  let qh : T #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg] := nn.transpose_for_attention q
  let kh : T #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg] := nn.transpose_for_attention k
  let vh : T #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg] := nn.transpose_for_attention v
  let attn : T #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg] :=
    nn.scaledDotProductAttentionGQA qh kh vh 0.0 false true
  let out4 : T #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg] := nn.transpose_from_attention attn
  let out3 : T #[1, seq, AlbertConfig.allHeadSize cfg] :=
    reshape out4 #[1, seq, AlbertConfig.allHeadSize cfg]
  let denseOut : T #[1, seq, cfg.hiddenSize] := m.dense.forward3d out3
  nn.layer_norm (denseOut + x) m.layerNormWeight m.layerNormBias cfg.layerNormEps

end KittenAlbertSelfAttention

structure KittenAlbertLayer (cfg : AlbertConfig) where
  attention : KittenAlbertSelfAttention cfg
  fullLayerNormWeight : T #[cfg.hiddenSize]
  fullLayerNormBias : T #[cfg.hiddenSize]
  ffn : LinearNorm cfg.hiddenSize cfg.intermediateSize
  ffnOutput : LinearNorm cfg.intermediateSize cfg.hiddenSize
  deriving TensorStruct, Inhabited

namespace KittenAlbertLayer

def init (cfg : AlbertConfig) : IO (KittenAlbertLayer cfg) := do
  let attention ← KittenAlbertSelfAttention.init cfg
  let fullLayerNormWeight ← initUniform #[cfg.hiddenSize] 0.02
  let fullLayerNormBias := initBias #[cfg.hiddenSize]
  let ffn ← LinearNorm.init cfg.hiddenSize cfg.intermediateSize
  let ffnOutput ← LinearNorm.init cfg.intermediateSize cfg.hiddenSize
  pure { attention, fullLayerNormWeight, fullLayerNormBias, ffn, ffnOutput }

def forward {seq : UInt64}
    (m : KittenAlbertLayer cfg)
    (x : T #[1, seq, cfg.hiddenSize])
    : T #[1, seq, cfg.hiddenSize] :=
  let attnOut := m.attention.forward x
  let f0 : T #[1, seq, cfg.intermediateSize] := m.ffn.forward3d attnOut
  let f1 := geluApprox f0
  let f2 : T #[1, seq, cfg.hiddenSize] := m.ffnOutput.forward3d f1
  nn.layer_norm (f2 + attnOut) m.fullLayerNormWeight m.fullLayerNormBias cfg.layerNormEps

end KittenAlbertLayer

structure KittenAlbertLayerGroup (cfg : AlbertConfig) where
  layers : Array (KittenAlbertLayer cfg)
  deriving TensorStruct, Inhabited

namespace KittenAlbertLayerGroup

def init (cfg : AlbertConfig) : IO (KittenAlbertLayerGroup cfg) := do
  let mut layers : Array (KittenAlbertLayer cfg) := #[]
  for _ in [:cfg.innerGroupNum.toNat] do
    layers := layers.push (← KittenAlbertLayer.init cfg)
  pure { layers }

def forward {seq : UInt64}
    (m : KittenAlbertLayerGroup cfg)
    (x : T #[1, seq, cfg.hiddenSize])
    : T #[1, seq, cfg.hiddenSize] :=
  m.layers.foldl (fun h layer => layer.forward h) x

end KittenAlbertLayerGroup

structure KittenAlbertEncoder (cfg : AlbertConfig) where
  embeddingHiddenMappingIn : LinearNorm cfg.embeddingSize cfg.hiddenSize
  groups : Array (KittenAlbertLayerGroup cfg)
  deriving TensorStruct, Inhabited

namespace KittenAlbertEncoder

def init (cfg : AlbertConfig) : IO (KittenAlbertEncoder cfg) := do
  let embeddingHiddenMappingIn ← LinearNorm.init cfg.embeddingSize cfg.hiddenSize
  let mut groups : Array (KittenAlbertLayerGroup cfg) := #[]
  for _ in [:cfg.numHiddenGroups.toNat] do
    groups := groups.push (← KittenAlbertLayerGroup.init cfg)
  pure { embeddingHiddenMappingIn, groups }

def forward {seq : UInt64}
    (m : KittenAlbertEncoder cfg)
    (x : T #[1, seq, cfg.embeddingSize])
    : T #[1, seq, cfg.hiddenSize] := Id.run do
  let mapped : T #[1, seq, cfg.hiddenSize] := m.embeddingHiddenMappingIn.forward3d x
  let layersPerGroup :=
    if cfg.numHiddenGroups == 0 then 1 else max 1 (cfg.numHiddenLayers / cfg.numHiddenGroups)
  let mut h := mapped
  for i in [:cfg.numHiddenLayers.toNat] do
    let groupIdx := if cfg.numHiddenGroups == 0 then 0 else i / layersPerGroup.toNat
    let group := safeIndex m.groups groupIdx
    h := group.forward h
  h

end KittenAlbertEncoder

structure KittenAlbert (nToken : UInt64) (cfg : AlbertConfig) where
  embeddings : AlbertEmbeddings nToken cfg
  encoder : KittenAlbertEncoder cfg
  pooler : LinearNorm cfg.hiddenSize cfg.hiddenSize
  deriving TensorStruct, Inhabited

namespace KittenAlbert

def init (nToken : UInt64) (cfg : AlbertConfig) : IO (KittenAlbert nToken cfg) := do
  let embeddings ← AlbertEmbeddings.init nToken cfg
  let encoder ← KittenAlbertEncoder.init cfg
  let pooler ← LinearNorm.init cfg.hiddenSize cfg.hiddenSize
  pure { embeddings, encoder, pooler }

def forward {seq nToken : UInt64}
    (m : KittenAlbert nToken cfg)
    (inputIds : T #[1, seq])
    : T #[1, seq, cfg.hiddenSize] × T #[1, cfg.hiddenSize] :=
  let emb := m.embeddings.forward inputIds
  let h := m.encoder.forward emb
  let cls3 : T #[1, 1, cfg.hiddenSize] := data.slice h 1 0 1
  let cls2 : T #[1, cfg.hiddenSize] := reshape cls3 #[1, cfg.hiddenSize]
  let pooled : T #[1, cfg.hiddenSize] := nn.tanh (m.pooler.forward2d cls2)
  (h, pooled)

end KittenAlbert

structure TextConvLayer (cfg : KittenTTSConfig) where
  conv : Conv1dParams cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize 1 ((cfg.textEncoderKernelSize - 1) / 2) 1
  lnWeight : T #[cfg.hiddenDim]
  lnBias : T #[cfg.hiddenDim]
  deriving TensorStruct, Inhabited

namespace TextConvLayer

def init (cfg : KittenTTSConfig) : IO (TextConvLayer cfg) := do
  let conv ← Conv1dParams.init cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize 1 ((cfg.textEncoderKernelSize - 1) / 2) 1
  let lnWeight ← initUniform #[cfg.hiddenDim] 0.02
  let lnBias := initBias #[cfg.hiddenDim]
  pure { conv, lnWeight, lnBias }

def forward {seq : UInt64}
    (m : TextConvLayer cfg)
    (x : T #[1, cfg.hiddenDim, seq])
    : T #[1, cfg.hiddenDim, seq] :=
  let y : T #[1, cfg.hiddenDim, seq] := m.conv.forward x
  let y := layerNormCF y m.lnWeight m.lnBias
  nn.leaky_relu y 0.2

end TextConvLayer

structure TextEncoder (cfg : KittenTTSConfig) where
  embedding : T #[cfg.nToken, cfg.hiddenDim]
  convs : Array (TextConvLayer cfg)
  lstm : BiLSTM cfg.hiddenDim (cfg.hiddenDim / 2)
  deriving TensorStruct, Inhabited

namespace TextEncoder

def init (cfg : KittenTTSConfig) : IO (TextEncoder cfg) := do
  let scale := xavierScale cfg.hiddenDim.toNat
  let embedding ← initUniform #[cfg.nToken, cfg.hiddenDim] scale
  let mut convs : Array (TextConvLayer cfg) := #[]
  for _ in [:cfg.nLayer.toNat] do
    convs := convs.push (← TextConvLayer.init cfg)
  let lstm ← BiLSTM.init cfg.hiddenDim (cfg.hiddenDim / 2)
  pure { embedding, convs, lstm }

def forward {seq : UInt64}
    (m : TextEncoder cfg)
    (inputIds : T #[1, seq])
    : T #[1, cfg.hiddenDim, seq] := Id.run do
  let emb : T #[1, seq, cfg.hiddenDim] := nn.embedding inputIds m.embedding
  let mut h : T #[1, cfg.hiddenDim, seq] := seqToCF emb
  for conv in m.convs do
    h := conv.forward h
  let hSeq : T #[1, seq, cfg.hiddenDim] := cfToSeq h
  let outSeq : T #[1, seq, cfg.hiddenDim] := m.lstm.forward hSeq
  seqToCF outSeq

end TextEncoder

structure AdaLayerNorm (styleDim channels : UInt64) where
  fc : LinearNorm styleDim (2 * channels)
  deriving TensorStruct, Inhabited

namespace AdaLayerNorm

def init (styleDim channels : UInt64) : IO (AdaLayerNorm styleDim channels) := do
  let fc ← LinearNorm.init styleDim (2 * channels)
  pure { fc }

def forward {batch frames styleDim channels : UInt64}
    (m : AdaLayerNorm styleDim channels)
    (x : T #[batch, channels, frames])
    (style : T #[batch, styleDim])
    : T #[batch, channels, frames] :=
  let h : T #[batch, 2 * channels] := m.fc.forward2d style
  let gamma : T #[batch, 1, channels] := reshape (data.slice h 1 0 channels) #[batch, 1, channels]
  let beta : T #[batch, 1, channels] := reshape (data.slice h 1 channels channels) #[batch, 1, channels]
  let xSeq : T #[batch, frames, channels] := cfToSeq x
  let normed : T #[batch, frames, channels] := nn.layer_norm' xSeq #[channels] none none 1e-5
  let g : T #[batch, frames, channels] := nn.expand gamma #[batch, frames, channels]
  let b : T #[batch, frames, channels] := nn.expand beta #[batch, frames, channels]
  seqToCF (affineShift normed g b)

end AdaLayerNorm

structure AdaIN1d (styleDim channels : UInt64) where
  fc : LinearNorm styleDim (2 * channels)
  normWeight : T #[channels]
  normBias : T #[channels]
  deriving TensorStruct, Inhabited

namespace AdaIN1d

def init (styleDim channels : UInt64) : IO (AdaIN1d styleDim channels) := do
  let fc ← LinearNorm.init styleDim (2 * channels)
  let normWeight := autograd.set_requires_grad (torch.ones #[channels] false Device.CPU) true
  let normBias := initBias #[channels]
  pure { fc, normWeight, normBias }

def forward {frames styleDim channels : UInt64}
    (m : AdaIN1d styleDim channels)
    (x : T #[1, channels, frames])
    (style : T #[1, styleDim])
    : T #[1, channels, frames] :=
  let h : T #[1, 2 * channels] := m.fc.forward2d style
  let gamma : T #[1, channels, 1] := reshape (data.slice h 1 0 channels) #[1, channels, 1]
  let beta : T #[1, channels, 1] := reshape (data.slice h 1 channels channels) #[1, channels, 1]
  let baseNorm := instanceNormCF x
  let normW : T #[1, channels, frames] := nn.expand (reshape m.normWeight #[1, channels, 1]) #[1, channels, frames]
  let normB : T #[1, channels, frames] := nn.expand (reshape m.normBias #[1, channels, 1]) #[1, channels, frames]
  let normed : T #[1, channels, frames] := baseNorm * normW + normB
  let g : T #[1, channels, frames] := nn.expand gamma #[1, channels, frames]
  let b : T #[1, channels, frames] := nn.expand beta #[1, channels, frames]
  affineShift normed g b

end AdaIN1d

structure DurationEncoder (cfg : KittenTTSConfig) where
  lstms : Array (BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2))
  norms : Array (AdaLayerNorm cfg.styleDim cfg.hiddenDim)
  deriving TensorStruct, Inhabited

namespace DurationEncoder

def init (cfg : KittenTTSConfig) : IO (DurationEncoder cfg) := do
  let mut lstms : Array (BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)) := #[]
  let mut norms : Array (AdaLayerNorm cfg.styleDim cfg.hiddenDim) := #[]
  for _ in [:cfg.nLayer.toNat] do
    lstms := lstms.push (← BiLSTM.init (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2))
    norms := norms.push (← AdaLayerNorm.init cfg.styleDim cfg.hiddenDim)
  pure { lstms, norms }

def forward {seq : UInt64}
    (m : DurationEncoder cfg)
    (x : T #[1, cfg.hiddenDim, seq])
    (style : T #[1, cfg.styleDim])
    : T #[1, cfg.hiddenDim + cfg.styleDim, seq] := Id.run do
  let s : T #[1, cfg.styleDim, seq] := styleExpand style
  let mut h : T #[1, cfg.hiddenDim + cfg.styleDim, seq] := nn.cat x s 1
  for i in [:m.lstms.size] do
    let lstm := safeIndex m.lstms i
    let norm := safeIndex m.norms i
    let ySeq : T #[1, seq, cfg.hiddenDim] := BiLSTM.forward lstm (cfToSeq h)
    let yCF : T #[1, cfg.hiddenDim, seq] := seqToCF ySeq
    let yNorm : T #[1, cfg.hiddenDim, seq] := AdaLayerNorm.forward norm yCF style
    h := nn.cat yNorm s 1
  h

end DurationEncoder

structure AdainResBlk1dSame (inC outC styleDim : UInt64) where
  conv1 : WNConv1dParams inC outC 3 1 1 1
  conv2 : WNConv1dParams outC outC 3 1 1 1
  norm1 : AdaIN1d styleDim inC
  norm2 : AdaIN1d styleDim outC
  shortcut : Option (WNConv1dParams inC outC 1 1 0 1) := none
  deriving TensorStruct, Inhabited

namespace AdainResBlk1dSame

def init (inC outC styleDim : UInt64) : IO (AdainResBlk1dSame inC outC styleDim) := do
  let conv1 ← WNConv1dParams.init inC outC 3 1 1 1
  let conv2 ← WNConv1dParams.init outC outC 3 1 1 1
  let norm1 ← AdaIN1d.init styleDim inC
  let norm2 ← AdaIN1d.init styleDim outC
  let shortcut ←
    if inC == outC then
      pure none
    else
      pure (some (← WNConv1dParams.init inC outC 1 1 0 1))
  pure { conv1, conv2, norm1, norm2, shortcut }

def forward {frames inC outC styleDim : UInt64}
    (m : AdainResBlk1dSame inC outC styleDim)
    (x : T #[1, inC, frames])
    (style : T #[1, styleDim])
    : T #[1, outC, frames] :=
  let shortcut : T #[1, outC, frames] :=
    match m.shortcut with
    | some sc => sc.forward x
    | none => reshape x #[1, outC, frames]
  let h0 : T #[1, inC, frames] := m.norm1.forward x style
  let h1 : T #[1, inC, frames] := nn.leaky_relu h0 0.2
  let h2 : T #[1, outC, frames] := m.conv1.forward h1
  let h3 : T #[1, outC, frames] := m.norm2.forward h2 style
  let h4 : T #[1, outC, frames] := nn.leaky_relu h3 0.2
  let h5 : T #[1, outC, frames] := m.conv2.forward h4
  (h5 + shortcut) / Float.sqrt 2.0

end AdainResBlk1dSame

structure AdainResBlk1dUp (inC outC styleDim : UInt64) where
  pool : WNDepthwiseTransConv1dParams inC 3 2 1 1 1
  conv1 : WNConv1dParams inC outC 3 1 1 1
  conv2 : WNConv1dParams outC outC 3 1 1 1
  norm1 : AdaIN1d styleDim inC
  norm2 : AdaIN1d styleDim outC
  shortcut : Option (WNConv1dParams inC outC 1 1 0 1) := none
  deriving TensorStruct, Inhabited

namespace AdainResBlk1dUp

def init (inC outC styleDim : UInt64) : IO (AdainResBlk1dUp inC outC styleDim) := do
  let pool ← WNDepthwiseTransConv1dParams.init inC 3 2 1 1 1
  let conv1 ← WNConv1dParams.init inC outC 3 1 1 1
  let conv2 ← WNConv1dParams.init outC outC 3 1 1 1
  let norm1 ← AdaIN1d.init styleDim inC
  let norm2 ← AdaIN1d.init styleDim outC
  let shortcut ←
    if inC == outC then
      pure none
    else
      pure (some (← WNConv1dParams.init inC outC 1 1 0 1))
  pure { pool, conv1, conv2, norm1, norm2, shortcut }

def forward {frames inC outC styleDim : UInt64}
    (m : AdainResBlk1dUp inC outC styleDim)
    (x : T #[1, inC, frames])
    (style : T #[1, styleDim])
    : T #[1, outC, 2 * frames] :=
  let shortcut0 : T #[1, inC, 2 * frames] := repeatNearest1d (scale := 2) x
  let shortcut : T #[1, outC, 2 * frames] :=
    match m.shortcut with
    | some sc => sc.forward shortcut0
    | none => reshape shortcut0 #[1, outC, 2 * frames]
  let h0 : T #[1, inC, frames] := m.norm1.forward x style
  let h1 : T #[1, inC, frames] := nn.leaky_relu h0 0.2
  let h2 : T #[1, inC, 2 * frames] := m.pool.forward h1
  let h3 : T #[1, outC, 2 * frames] := m.conv1.forward h2
  let h4 : T #[1, outC, 2 * frames] := m.norm2.forward h3 style
  let h5 : T #[1, outC, 2 * frames] := nn.leaky_relu h4 0.2
  let h6 : T #[1, outC, 2 * frames] := m.conv2.forward h5
  (h6 + shortcut) / Float.sqrt 2.0

end AdainResBlk1dUp

structure AdaINResBlock1 (channels kernel : UInt64) (styleDim : UInt64) where
  convs1 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1)
  convs2 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1)
  norms1 : Array (AdaIN1d styleDim channels)
  norms2 : Array (AdaIN1d styleDim channels)
  alpha1 : Array (T #[1, channels, 1])
  alpha2 : Array (T #[1, channels, 1])
  deriving TensorStruct, Inhabited

namespace AdaINResBlock1

def init (channels kernel styleDim : UInt64) : IO (AdaINResBlock1 channels kernel styleDim) := do
  let mut convs1 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1) := #[]
  let mut convs2 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1) := #[]
  let mut norms1 : Array (AdaIN1d styleDim channels) := #[]
  let mut norms2 : Array (AdaIN1d styleDim channels) := #[]
  let mut alpha1 : Array (T #[1, channels, 1]) := #[]
  let mut alpha2 : Array (T #[1, channels, 1]) := #[]
  for _ in [:3] do
    convs1 := convs1.push (← Conv1dParams.init channels channels kernel 1 ((kernel - 1) / 2) 1)
    convs2 := convs2.push (← Conv1dParams.init channels channels kernel 1 ((kernel - 1) / 2) 1)
    norms1 := norms1.push (← AdaIN1d.init styleDim channels)
    norms2 := norms2.push (← AdaIN1d.init styleDim channels)
    alpha1 := alpha1.push (reshape (torch.ones #[channels] false Device.CPU) #[1, channels, 1])
    alpha2 := alpha2.push (reshape (torch.ones #[channels] false Device.CPU) #[1, channels, 1])
  pure { convs1, convs2, norms1, norms2, alpha1, alpha2 }

def forward {frames channels kernel styleDim : UInt64}
    (m : AdaINResBlock1 channels kernel styleDim)
    (x : T #[1, channels, frames])
    (style : T #[1, styleDim])
    : T #[1, channels, frames] := Id.run do
  let mut h := x
  for i in [:3] do
    let a1 := safeIndex m.alpha1 i
    let a2 := safeIndex m.alpha2 i
    let h1 := AdaIN1d.forward (safeIndex m.norms1 i) h style
    let h2 := snake1d h1 a1
    let h3 : T #[1, channels, frames] := reshape (Conv1dParams.forward (safeIndex m.convs1 i) h2) #[1, channels, frames]
    let h4 := AdaIN1d.forward (safeIndex m.norms2 i) h3 style
    let h5 := snake1d h4 a2
    let h6 : T #[1, channels, frames] := reshape (Conv1dParams.forward (safeIndex m.convs2 i) h5) #[1, channels, frames]
    h := h + h6
  h

end AdaINResBlock1

structure ProsodyPredictor (cfg : KittenTTSConfig) where
  durationEncoder : DurationEncoder cfg
  lstm : BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  durationProj : LinearNorm cfg.hiddenDim cfg.maxDur
  shared : BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  f0Blk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  f0Blk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  f0Blk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  nBlk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  nBlk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  nBlk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  f0Proj : Conv1dParams (cfg.hiddenDim / 2) 1 1 1 0 1
  nProj : Conv1dParams (cfg.hiddenDim / 2) 1 1 1 0 1
  deriving TensorStruct

namespace ProsodyPredictor

def init (cfg : KittenTTSConfig) : IO (ProsodyPredictor cfg) := do
  let durationEncoder ← DurationEncoder.init cfg
  let lstm ← BiLSTM.init (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let durationProj ← LinearNorm.init cfg.hiddenDim cfg.maxDur
  let shared ← BiLSTM.init (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let f0Blk0 ← AdainResBlk1dSame.init cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let f0Blk1 ← AdainResBlk1dUp.init cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let f0Blk2 ← AdainResBlk1dSame.init (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk0 ← AdainResBlk1dSame.init cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let nBlk1 ← AdainResBlk1dUp.init cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk2 ← AdainResBlk1dSame.init (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  let f0Proj ← Conv1dParams.init (cfg.hiddenDim / 2) 1 1 1 0 1
  let nProj ← Conv1dParams.init (cfg.hiddenDim / 2) 1 1 1 0 1
  pure {
    durationEncoder, lstm, durationProj, shared,
    f0Blk0, f0Blk1, f0Blk2, nBlk0, nBlk1, nBlk2, f0Proj, nProj
  }

def durationEncoding {seq : UInt64}
    (m : ProsodyPredictor cfg)
    (dEn : T #[1, cfg.hiddenDim, seq])
    (style : T #[1, cfg.styleDim])
    : T #[1, cfg.hiddenDim + cfg.styleDim, seq] :=
  m.durationEncoder.forward dEn style

def durationLogits {seq : UInt64}
    (m : ProsodyPredictor cfg)
    (enc : T #[1, cfg.hiddenDim + cfg.styleDim, seq])
    : T #[1, seq, cfg.maxDur] :=
  let hSeq : T #[1, seq, cfg.hiddenDim] := m.lstm.forward (cfToSeq enc)
  m.durationProj.forward3d hSeq

def forwardF0N {frames : UInt64}
    (m : ProsodyPredictor cfg)
    (enc : T #[1, cfg.hiddenDim + cfg.styleDim, frames])
    (style : T #[1, cfg.styleDim])
    : T #[1, 1, 2 * frames] × T #[1, 1, 2 * frames] :=
  let sharedSeq : T #[1, frames, cfg.hiddenDim] := m.shared.forward (cfToSeq enc)
  let sharedCF : T #[1, cfg.hiddenDim, frames] := seqToCF sharedSeq

  let f0a : T #[1, cfg.hiddenDim, frames] := m.f0Blk0.forward sharedCF style
  let f0b : T #[1, cfg.hiddenDim / 2, 2 * frames] := m.f0Blk1.forward f0a style
  let f0c : T #[1, cfg.hiddenDim / 2, 2 * frames] := m.f0Blk2.forward f0b style
  let f0 : T #[1, 1, 2 * frames] := m.f0Proj.forward f0c

  let na : T #[1, cfg.hiddenDim, frames] := m.nBlk0.forward sharedCF style
  let nb : T #[1, cfg.hiddenDim / 2, 2 * frames] := m.nBlk1.forward na style
  let nc : T #[1, cfg.hiddenDim / 2, 2 * frames] := m.nBlk2.forward nb style
  let n : T #[1, 1, 2 * frames] := m.nProj.forward nc

  (f0, n)

end ProsodyPredictor

private def reflectPadLeft1 {channels frames : UInt64}
    (x : T #[1, channels, frames])
    : T #[1, channels, frames + 1] :=
  let pad : T #[1, channels, 1] :=
    if frames <= 1 then
      data.slice x 2 0 1
    else
      data.slice x 2 1 1
  reshape (nn.cat_dyn #[nn.eraseShape pad, nn.eraseShape x] 2) #[1, channels, frames + 1]

private def harmonicMultipliers (harmonicCount : UInt64) : T #[harmonicCount + 1] := Id.run do
  let mut vals : Array Float := #[]
  for i in [: (harmonicCount + 1).toNat] do
    vals := vals.push (i.toFloat + 1.0)
  reshape (data.fromFloatArray vals) #[harmonicCount + 1]

structure AdaIN1dDyn (styleDim : UInt64) where
  fcWeight : T #[]
  fcBias : T #[]
  normWeight : T #[]
  normBias : T #[]
  deriving TensorStruct, Inhabited

namespace AdaIN1dDyn

def init (styleDim channels : UInt64) : IO (AdaIN1dDyn styleDim) := do
  let fc ← LinearNorm.init styleDim (2 * channels)
  let normWeight : T #[channels] := autograd.set_requires_grad (torch.ones #[channels] false Device.CPU) true
  let normBias := initBias #[channels]
  pure {
    fcWeight := nn.eraseShape fc.weight
    fcBias := nn.eraseShape fc.bias
    normWeight := nn.eraseShape normWeight
    normBias := nn.eraseShape normBias
  }

def forward {frames styleDim channels : UInt64}
    (m : AdaIN1dDyn styleDim)
    (x : T #[1, channels, frames])
    (style : T #[1, styleDim])
    : T #[1, channels, frames] :=
  let weight : T #[2 * channels, styleDim] := reshape m.fcWeight #[2 * channels, styleDim]
  let bias : T #[2 * channels] := reshape m.fcBias #[2 * channels]
  let h : T #[1, 2 * channels] := affine style weight bias
  let gamma : T #[1, channels, 1] := reshape (data.slice h 1 0 channels) #[1, channels, 1]
  let beta : T #[1, channels, 1] := reshape (data.slice h 1 channels channels) #[1, channels, 1]
  let normWeight : T #[channels] := reshape m.normWeight #[channels]
  let normBias : T #[channels] := reshape m.normBias #[channels]
  let baseNorm := instanceNormCF x
  let normW : T #[1, channels, frames] := nn.expand (reshape normWeight #[1, channels, 1]) #[1, channels, frames]
  let normB : T #[1, channels, frames] := nn.expand (reshape normBias #[1, channels, 1]) #[1, channels, frames]
  let normed : T #[1, channels, frames] := baseNorm * normW + normB
  let g : T #[1, channels, frames] := nn.expand gamma #[1, channels, frames]
  let b : T #[1, channels, frames] := nn.expand beta #[1, channels, frames]
  affineShift normed g b

end AdaIN1dDyn

structure WNConv1dDyn where
  weightV : T #[]
  weightG : T #[]
  bias : T #[]
  kernel : UInt64
  stride : UInt64 := 1
  padding : UInt64 := 0
  dilation : UInt64 := 1
  deriving TensorStruct, Inhabited

namespace WNConv1dDyn

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO WNConv1dDyn := do
  let weightV ← initWeight #[outC, inC, kernel] (inC * kernel)
  let weightG : T #[outC] := autograd.set_requires_grad (normDim0_3d weightV) true
  let bias := initBias #[outC]
  pure {
    weightV := nn.eraseShape weightV
    weightG := nn.eraseShape weightG
    bias := nn.eraseShape bias
    kernel, stride, padding, dilation
  }

def forward {batch frames inC outC : UInt64}
    (m : WNConv1dDyn)
    (x : T #[batch, inC, frames])
    : T #[batch, outC, convOutputSize frames m.kernel m.stride m.padding m.dilation] :=
  let weightV : T #[outC, inC, m.kernel] := reshape m.weightV #[outC, inC, m.kernel]
  let weightG : T #[outC] := reshape m.weightG #[outC]
  let bias : T #[outC] := reshape m.bias #[outC]
  let weight := applyWeightNormConv1d weightV weightG
  let y0 : T #[batch, outC, convOutputSize frames m.kernel m.stride m.padding m.dilation] :=
    reshape (nn.conv1d x weight m.stride m.padding m.dilation) #[batch, outC, convOutputSize frames m.kernel m.stride m.padding m.dilation]
  addBias3d y0 bias

end WNConv1dDyn

structure WNConvTranspose1dDyn where
  weightV : T #[]
  weightG : T #[]
  bias : T #[]
  kernel : UInt64
  stride : UInt64 := 1
  padding : UInt64 := 0
  outputPadding : UInt64 := 0
  dilation : UInt64 := 1
  deriving TensorStruct, Inhabited

namespace WNConvTranspose1dDyn

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (outputPadding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO WNConvTranspose1dDyn := do
  let fanIn := if outC * kernel == 0 then 1 else outC * kernel
  let weightV ← initWeight #[inC, outC, kernel] fanIn
  let weightG : T #[inC] := autograd.set_requires_grad (normDim0_3d weightV) true
  let bias := initBias #[outC]
  pure {
    weightV := nn.eraseShape weightV
    weightG := nn.eraseShape weightG
    bias := nn.eraseShape bias
    kernel, stride, padding, outputPadding, dilation
  }

def forward {batch frames inC outC : UInt64}
    (m : WNConvTranspose1dDyn)
    (x : T #[batch, inC, frames])
    : T #[batch, outC, convTransposeOutputSize frames m.kernel m.stride m.padding m.outputPadding m.dilation] :=
  let weightV : T #[inC, outC, m.kernel] := reshape m.weightV #[inC, outC, m.kernel]
  let weightG : T #[inC] := reshape m.weightG #[inC]
  let bias : T #[outC] := reshape m.bias #[outC]
  let weight := applyWeightNormTransConv1d weightV weightG
  reshape
    (nn.conv_transpose1d_bias x weight bias m.stride m.padding m.outputPadding m.dilation)
    #[batch, outC, convTransposeOutputSize frames m.kernel m.stride m.padding m.outputPadding m.dilation]

end WNConvTranspose1dDyn

structure GeneratorAdaINResBlock (styleDim : UInt64) where
  convs1 : Array WNConv1dDyn
  convs2 : Array WNConv1dDyn
  norms1 : Array (AdaIN1dDyn styleDim)
  norms2 : Array (AdaIN1dDyn styleDim)
  alpha1 : Array (T #[])
  alpha2 : Array (T #[])
  deriving TensorStruct, Inhabited

namespace GeneratorAdaINResBlock

def init (styleDim channels kernel : UInt64) (dilations : Array UInt64) : IO (GeneratorAdaINResBlock styleDim) := do
  let mut convs1 : Array WNConv1dDyn := #[]
  let mut convs2 : Array WNConv1dDyn := #[]
  let mut norms1 : Array (AdaIN1dDyn styleDim) := #[]
  let mut norms2 : Array (AdaIN1dDyn styleDim) := #[]
  let mut alpha1 : Array (T #[]) := #[]
  let mut alpha2 : Array (T #[]) := #[]
  for i in [:3] do
    let dilation := safeIndex dilations i
    let padding := (kernel * dilation - dilation) / 2
    convs1 := convs1.push (← WNConv1dDyn.init channels channels kernel 1 padding dilation)
    convs2 := convs2.push (← WNConv1dDyn.init channels channels kernel 1 ((kernel - 1) / 2) 1)
    norms1 := norms1.push (← AdaIN1dDyn.init styleDim channels)
    norms2 := norms2.push (← AdaIN1dDyn.init styleDim channels)
    alpha1 := alpha1.push (nn.eraseShape (reshape (torch.ones #[channels] false Device.CPU) #[1, channels, 1]))
    alpha2 := alpha2.push (nn.eraseShape (reshape (torch.ones #[channels] false Device.CPU) #[1, channels, 1]))
  pure { convs1, convs2, norms1, norms2, alpha1, alpha2 }

def forward {frames channels styleDim : UInt64}
    (m : GeneratorAdaINResBlock styleDim)
    (x : T #[1, channels, frames])
    (style : T #[1, styleDim])
    : T #[1, channels, frames] := Id.run do
  let mut h := x
  for i in [:3] do
    let a1 : T #[1, channels, 1] := reshape (safeIndex m.alpha1 i) #[1, channels, 1]
    let a2 : T #[1, channels, 1] := reshape (safeIndex m.alpha2 i) #[1, channels, 1]
    let h1 := AdaIN1dDyn.forward (safeIndex m.norms1 i) h style
    let h2 := snake1d h1 a1
    let h3 : T #[1, channels, frames] :=
      reshape (WNConv1dDyn.forward (inC := channels) (outC := channels) (safeIndex m.convs1 i) h2) #[1, channels, frames]
    let h4 := AdaIN1dDyn.forward (safeIndex m.norms2 i) h3 style
    let h5 := snake1d h4 a2
    let h6 : T #[1, channels, frames] :=
      reshape (WNConv1dDyn.forward (inC := channels) (outC := channels) (safeIndex m.convs2 i) h5) #[1, channels, frames]
    h := h + h6
  h

end GeneratorAdaINResBlock

structure SourceModuleHnNSF (cfg : KittenTTSConfig) where
  linear : LinearNorm (cfg.generator.harmonicCount + 1) 1
  deriving TensorStruct

namespace SourceModuleHnNSF

private def sineAmp : Float := 0.1
private def noiseStd : Float := 0.003
private def voicedThreshold : Float := 10.0

def init (cfg : KittenTTSConfig) : IO (SourceModuleHnNSF cfg) := do
  let linear ← LinearNorm.init (cfg.generator.harmonicCount + 1) 1
  pure { linear }

def forward {baseFrames : UInt64}
    (m : SourceModuleHnNSF cfg)
    (f0 : T #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1])
    : IO
      (T #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1] ×
       T #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1] ×
       T #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1]) := do
  let waveFrames := GeneratorConfig.waveSamples cfg.generator baseFrames
  let harmonics := cfg.generator.harmonicCount + 1
  let mult : T #[1, 1, harmonics] := reshape (harmonicMultipliers cfg.generator.harmonicCount) #[1, 1, harmonics]
  let multE : T #[1, waveFrames, harmonics] := nn.expand mult #[1, waveFrames, harmonics]
  let f0E : T #[1, waveFrames, harmonics] := nn.expand f0 #[1, waveFrames, harmonics]
  let rad0 : T #[1, waveFrames, harmonics] := (f0E * multE) / cfg.sampleRate.toFloat
  let radBase : T #[1, waveFrames, harmonics] := rad0 - nn.floor rad0

  let randIni : T #[1, harmonics] ← torch.rand #[1, harmonics] false f0.device
  let randIni : T #[1, harmonics] :=
    data.sliceScatter randIni 1 0 (torch.zeros #[1, 1] false f0.device)
  let phaseNoise0 : T #[1, waveFrames, harmonics] := torch.zeros #[1, waveFrames, harmonics] false f0.device
  let phaseNoise : T #[1, waveFrames, harmonics] :=
    data.sliceScatter phaseNoise0 1 0 (reshape randIni #[1, 1, harmonics])
  let radInit : T #[1, waveFrames, harmonics] := radBase + phaseNoise

  let radCF : T #[1, harmonics, waveFrames] := seqToCF radInit
  let radDownCF : T #[1, harmonics, baseFrames] := resize1d (outFrames := baseFrames) radCF "linear"
  let radDown : T #[1, baseFrames, harmonics] := cfToSeq radDownCF
  let phaseBase : T #[1, baseFrames, harmonics] := mul_scalar (nn.cumsum radDown 1) 6.283185307179586
  let phaseScaledCF : T #[1, harmonics, baseFrames] :=
    seqToCF phaseBase * (GeneratorConfig.sourceUpsample cfg.generator).toFloat
  let phaseUpCF : T #[1, harmonics, waveFrames] := resize1d (outFrames := waveFrames) phaseScaledCF "linear"
  let sines : T #[1, waveFrames, harmonics] := nn.sin (cfToSeq phaseUpCF)
  let sineWaves : T #[1, waveFrames, harmonics] := sines * sineAmp

  let threshold : T #[1, waveFrames, 1] := torch.full #[1, waveFrames, 1] voicedThreshold false f0.device
  let uv : T #[1, waveFrames, 1] := torch.toFloat' (torch.gt f0 threshold)
  let uvE : T #[1, waveFrames, harmonics] := nn.expand uv #[1, waveFrames, harmonics]
  let noiseAmp : T #[1, waveFrames, harmonics] :=
    uvE * noiseStd + (torch.ones_like uvE - uvE) * (sineAmp / 3.0)
  let noise : T #[1, waveFrames, harmonics] ← torch.randn #[1, waveFrames, harmonics] false f0.device
  let sineNoisy : T #[1, waveFrames, harmonics] := sineWaves * uvE + noiseAmp * noise

  let sineMerge : T #[1, waveFrames, 1] := nn.tanh (m.linear.forward3d sineNoisy)
  let noiseOut : T #[1, waveFrames, 1] ← torch.randn #[1, waveFrames, 1] false f0.device
  pure (sineMerge, noiseOut * (sineAmp / 3.0), uv)

end SourceModuleHnNSF

structure GeneratorStage (styleDim : UInt64) where
  up : WNConvTranspose1dDyn
  noiseConv : WNConv1dDyn
  noiseRes : GeneratorAdaINResBlock styleDim
  resBlocks : Array (GeneratorAdaINResBlock styleDim)
  deriving TensorStruct, Inhabited

structure GeneratorStageDebugOutput where
  xSource : T #[]
  xUp : T #[]
  xMix : T #[]
  xOut : T #[]
  deriving TensorStruct, Inhabited

structure GeneratorDebugOutput where
  harSource : T #[]
  har : T #[]
  stages : Array GeneratorStageDebugOutput
  post : T #[]
  specLog : T #[]
  phaseRaw : T #[]
  spec : T #[]
  audio : T #[]
  deriving TensorStruct, Inhabited

structure Generator (cfg : KittenTTSConfig) where
  source : SourceModuleHnNSF cfg
  stages : Array (GeneratorStage cfg.styleDim)
  convPost : WNConv1dDyn
  window : T #[cfg.generator.genIstftNFft]
  deriving TensorStruct

namespace Generator

private def buildHarFeatures {baseFrames : UInt64}
    (m : Generator cfg)
    (harmonic : T #[GeneratorConfig.waveSamples cfg.generator baseFrames])
    : T #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
  let window : T #[cfg.generator.genIstftNFft] := m.window.to harmonic.device
  let stftDyn : T #[] :=
    signal.stft1d
      (n := GeneratorConfig.waveSamples cfg.generator baseFrames)
      harmonic
      cfg.generator.genIstftNFft
      cfg.generator.genIstftHopSize
      cfg.generator.genIstftNFft
      window
      true
      false
  let packed : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2] :=
    reshape stftDyn #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let re3 : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1] :=
    data.slice packed 2 0 1
  let im3 : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1] :=
    data.slice packed 2 1 1
  let re : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape re3 #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let im : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape im3 #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let mag : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    nn.sqrt (re * re + im * im + (1e-12 : Float))
  let phase : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    nn.atan2 im re
  let mag3 : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape mag #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let phase3 : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape phase #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  nn.cat mag3 phase3 1

def init (cfg : KittenTTSConfig) : IO (Generator cfg) := do
  let source ← SourceModuleHnNSF.init cfg
  let mut stages : Array (GeneratorStage cfg.styleDim) := #[]
  for i in [:cfg.generator.upsampleRates.size] do
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let stride := safeIndex cfg.generator.upsampleRates i
    let kernel := safeIndex cfg.generator.upsampleKernelSizes i
    let up ← WNConvTranspose1dDyn.init inC outC kernel stride ((kernel - stride) / 2) 0 1
    let noiseConv ←
      WNConv1dDyn.init
        (GeneratorConfig.stftFeatureChannels cfg.generator)
        outC
        (GeneratorConfig.noiseKernel cfg.generator i)
        (GeneratorConfig.noiseStride cfg.generator i)
        (GeneratorConfig.noisePadding cfg.generator i)
        1
    let noiseKernel := if i + 1 < cfg.generator.upsampleRates.size then 7 else 11
    let noiseRes ← GeneratorAdaINResBlock.init cfg.styleDim outC noiseKernel #[1, 3, 5]
    let mut resBlocks : Array (GeneratorAdaINResBlock cfg.styleDim) := #[]
    for j in [:cfg.generator.resblockKernelSizes.size] do
      let blockKernel := safeIndex cfg.generator.resblockKernelSizes j
      let blockDilations := safeIndex cfg.generator.resblockDilationSizes j
      resBlocks := resBlocks.push (← GeneratorAdaINResBlock.init cfg.styleDim outC blockKernel blockDilations)
    stages := stages.push { up, noiseConv, noiseRes, resBlocks }
  let convPost ←
    WNConv1dDyn.init
      (GeneratorConfig.finalChannels cfg.generator)
      (cfg.generator.genIstftNFft + 2)
      7
      1
      3
      1
  let window := signal.hannWindow cfg.generator.genIstftNFft
  pure { source, stages, convPost, window }

def debugForward {baseFrames : UInt64}
    (m : Generator cfg)
    (x : T #[1, KittenTTSConfig.decoderChannels cfg, baseFrames])
    (style : T #[1, cfg.styleDim])
    (f0Curve : T #[1, 1, baseFrames])
    : IO GeneratorDebugOutput := do
  let f0Up : T #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames] :=
    repeatNearest1d (scale := GeneratorConfig.sourceUpsample cfg.generator) f0Curve
  let f0Seq : T #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1] := cfToSeq f0Up
  let (harSource, _noiseSource, _uv) ← m.source.forward f0Seq
  let har1d : T #[GeneratorConfig.waveSamples cfg.generator baseFrames] :=
    reshape harSource #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  let har : T #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    buildHarFeatures m har1d

  let mut xDyn : T #[] := nn.eraseShape x
  let mut stageOutputs : Array GeneratorStageDebugOutput := #[]
  for i in [:m.stages.size] do
    let stage := safeIndex m.stages i
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let inFrames := GeneratorConfig.stageInFrames cfg.generator baseFrames i
    let outFrames := GeneratorConfig.stageOutFrames cfg.generator baseFrames i
    let xIn : T #[1, inC, inFrames] := reshape xDyn #[1, inC, inFrames]
    let xAct : T #[1, inC, inFrames] := nn.leaky_relu xIn 0.1
    let xSource0 : T #[1, outC, convOutputSize (GeneratorConfig.harFrames cfg.generator baseFrames) stage.noiseConv.kernel stage.noiseConv.stride stage.noiseConv.padding stage.noiseConv.dilation] :=
      stage.noiseConv.forward har
    let xSource1 : T #[1, outC, outFrames] := reshape xSource0 #[1, outC, outFrames]
    let xSource : T #[1, outC, outFrames] := stage.noiseRes.forward xSource1 style
    let xUp0 : T #[1, outC, convTransposeOutputSize inFrames stage.up.kernel stage.up.stride stage.up.padding stage.up.outputPadding stage.up.dilation] :=
      stage.up.forward xAct
    let xUp : T #[1, outC, outFrames] :=
      if i + 1 < m.stages.size then
        reshape xUp0 #[1, outC, outFrames]
      else
        reflectPadLeft1 xUp0
    let xMix : T #[1, outC, outFrames] := xUp + xSource
    let first : T #[1, outC, outFrames] := (safeIndex stage.resBlocks 0).forward xMix style
    let mut acc := first
    for j in [1:stage.resBlocks.size] do
      acc := acc + (safeIndex stage.resBlocks j).forward xMix style
    let xOut : T #[1, outC, outFrames] := acc / stage.resBlocks.size.toFloat
    stageOutputs := stageOutputs.push {
      xSource := nn.eraseShape xSource
      xUp := nn.eraseShape xUp
      xMix := nn.eraseShape xMix
      xOut := nn.eraseShape xOut
    }
    xDyn := nn.eraseShape xOut

  let xFinal : T #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape xDyn #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let postIn : T #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := nn.leaky_relu xFinal 0.01
  let post : T #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape
      (WNConv1dDyn.forward
        (inC := GeneratorConfig.finalChannels cfg.generator)
        (outC := cfg.generator.genIstftNFft + 2)
        m.convPost
        postIn)
      #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames]

  let freqBins := GeneratorConfig.freqBins cfg.generator
  let specLog : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    data.slice post 1 0 freqBins
  let phaseRaw : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    data.slice post 1 freqBins freqBins
  let spec : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := nn.exp specLog
  let phase : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := nn.sin phaseRaw
  let re : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := spec * nn.cos phase
  let im : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := spec * nn.sin phase
  let re3 : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1] :=
    reshape re #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let im3 : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1] :=
    reshape im #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let packed : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2] :=
    reshape
      (nn.cat_dyn #[nn.eraseShape re3, nn.eraseShape im3] 2)
      #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let window : T #[cfg.generator.genIstftNFft] := m.window.to post.device
  let audio1d : T #[GeneratorConfig.waveSamples cfg.generator baseFrames] :=
    reshape
      (signal.istft1d
        (nn.eraseShape packed)
        cfg.generator.genIstftNFft
        cfg.generator.genIstftHopSize
        cfg.generator.genIstftNFft
        window
        true
        false
        (GeneratorConfig.waveSamples cfg.generator baseFrames))
      #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  pure {
    harSource := nn.eraseShape harSource
    har := nn.eraseShape har
    stages := stageOutputs
    post := nn.eraseShape post
    specLog := nn.eraseShape specLog
    phaseRaw := nn.eraseShape phaseRaw
    spec := nn.eraseShape spec
    audio := nn.eraseShape (reshape audio1d #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames])
  }

def forward {baseFrames : UInt64}
    (m : Generator cfg)
    (x : T #[1, KittenTTSConfig.decoderChannels cfg, baseFrames])
    (style : T #[1, cfg.styleDim])
    (f0Curve : T #[1, 1, baseFrames])
    : IO (T #[]) := do
  let f0Up : T #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames] :=
    repeatNearest1d (scale := GeneratorConfig.sourceUpsample cfg.generator) f0Curve
  let f0Seq : T #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1] := cfToSeq f0Up
  let (harSource, _noiseSource, _uv) ← m.source.forward f0Seq
  let har1d : T #[GeneratorConfig.waveSamples cfg.generator baseFrames] :=
    reshape harSource #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  let har : T #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    buildHarFeatures m har1d

  let mut xDyn : T #[] := nn.eraseShape x
  for i in [:m.stages.size] do
    let stage := safeIndex m.stages i
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let inFrames := GeneratorConfig.stageInFrames cfg.generator baseFrames i
    let outFrames := GeneratorConfig.stageOutFrames cfg.generator baseFrames i
    let xIn : T #[1, inC, inFrames] := reshape xDyn #[1, inC, inFrames]
    let xAct : T #[1, inC, inFrames] := nn.leaky_relu xIn 0.1
    let xSource0 : T #[1, outC, convOutputSize (GeneratorConfig.harFrames cfg.generator baseFrames) stage.noiseConv.kernel stage.noiseConv.stride stage.noiseConv.padding stage.noiseConv.dilation] :=
      stage.noiseConv.forward har
    let xSource1 : T #[1, outC, outFrames] := reshape xSource0 #[1, outC, outFrames]
    let xSource : T #[1, outC, outFrames] := stage.noiseRes.forward xSource1 style
    let xUp0 : T #[1, outC, convTransposeOutputSize inFrames stage.up.kernel stage.up.stride stage.up.padding stage.up.outputPadding stage.up.dilation] :=
      stage.up.forward xAct
    let xUp : T #[1, outC, outFrames] :=
      if i + 1 < m.stages.size then
        reshape xUp0 #[1, outC, outFrames]
      else
        reflectPadLeft1 xUp0
    let xMix : T #[1, outC, outFrames] := xUp + xSource
    let first : T #[1, outC, outFrames] := (safeIndex stage.resBlocks 0).forward xMix style
    let mut acc := first
    for j in [1:stage.resBlocks.size] do
      acc := acc + (safeIndex stage.resBlocks j).forward xMix style
    let xOut : T #[1, outC, outFrames] := acc / stage.resBlocks.size.toFloat
    xDyn := nn.eraseShape xOut

  let xFinal : T #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape xDyn #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let postIn : T #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := nn.leaky_relu xFinal 0.01
  let post : T #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    reshape
      (WNConv1dDyn.forward
        (inC := GeneratorConfig.finalChannels cfg.generator)
        (outC := cfg.generator.genIstftNFft + 2)
        m.convPost
        postIn)
      #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames]

  let freqBins := GeneratorConfig.freqBins cfg.generator
  let specLog : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    data.slice post 1 0 freqBins
  let phaseRaw : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] :=
    data.slice post 1 freqBins freqBins
  let spec : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := nn.exp specLog
  let phase : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := nn.sin phaseRaw
  let re : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := spec * nn.cos phase
  let im : T #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames] := spec * nn.sin phase
  let re3 : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1] :=
    reshape re #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let im3 : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1] :=
    reshape im #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let packed : T #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2] :=
    reshape
      (nn.cat_dyn #[nn.eraseShape re3, nn.eraseShape im3] 2)
      #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let window : T #[cfg.generator.genIstftNFft] := m.window.to post.device
  let audio1d : T #[GeneratorConfig.waveSamples cfg.generator baseFrames] :=
    reshape
      (signal.istft1d
        (nn.eraseShape packed)
        cfg.generator.genIstftNFft
        cfg.generator.genIstftHopSize
        cfg.generator.genIstftNFft
        window
        true
        false
        (GeneratorConfig.waveSamples cfg.generator baseFrames))
      #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  pure (nn.eraseShape (reshape audio1d #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames]))

end Generator

structure DecoderDebugOutput where
  f0 : T #[]
  n : T #[]
  encode : T #[]
  asrRes : T #[]
  decode0 : T #[]
  decode1 : T #[]
  decode2 : T #[]
  decode3 : T #[]
  generator : GeneratorDebugOutput
  deriving TensorStruct, Inhabited

structure Decoder (cfg : KittenTTSConfig) where
  f0Conv : WNConv1dParams 1 1 3 2 1 1
  nConv : WNConv1dParams 1 1 3 2 1 1
  encode : AdainResBlk1dSame (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  asrRes : WNConv1dParams cfg.hiddenDim cfg.asrResDim 1 1 0 1
  decode0 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode1 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode2 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode3 : AdainResBlk1dUp (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  generator : Generator cfg
  deriving TensorStruct

namespace Decoder

def init (cfg : KittenTTSConfig) : IO (Decoder cfg) := do
  let f0Conv ← WNConv1dParams.init 1 1 3 2 1 1
  let nConv ← WNConv1dParams.init 1 1 3 2 1 1
  let encode ← AdainResBlk1dSame.init (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  let asrRes ← WNConv1dParams.init cfg.hiddenDim cfg.asrResDim 1 1 0 1
  let decode0 ← AdainResBlk1dSame.init (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode1 ← AdainResBlk1dSame.init (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode2 ← AdainResBlk1dSame.init (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode3 ← AdainResBlk1dUp.init (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  let generator ← Generator.init cfg
  pure { f0Conv, nConv, encode, asrRes, decode0, decode1, decode2, decode3, generator }

def debugForward {frames : UInt64}
    (m : Decoder cfg)
    (asr : T #[1, cfg.hiddenDim, frames])
    (f0Curve : T #[1, 1, 2 * frames])
    (nCurve : T #[1, 1, 2 * frames])
    (style : T #[1, cfg.styleDim])
    : IO DecoderDebugOutput := do
  let f0 : T #[1, 1, frames] := m.f0Conv.forward f0Curve
  let n : T #[1, 1, frames] := m.nConv.forward nCurve
  let x0 : T #[1, cfg.hiddenDim + 2, frames] := nn.cat asr (nn.cat f0 n 1) 1
  let x1 : T #[1, cfg.maxConvDim, frames] := m.encode.forward x0 style
  let asrRes : T #[1, cfg.asrResDim, frames] := m.asrRes.forward asr

  let d0In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat x1 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d0 : T #[1, cfg.maxConvDim, frames] := m.decode0.forward d0In style
  let d1In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat d0 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d1 : T #[1, cfg.maxConvDim, frames] := m.decode1.forward d1In style
  let d2In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat d1 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d2 : T #[1, cfg.maxConvDim, frames] := m.decode2.forward d2In style
  let d3In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat d2 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d3 : T #[1, KittenTTSConfig.decoderChannels cfg, 2 * frames] := m.decode3.forward d3In style
  let generator ← m.generator.debugForward d3 style f0Curve
  pure {
    f0 := nn.eraseShape f0
    n := nn.eraseShape n
    encode := nn.eraseShape x1
    asrRes := nn.eraseShape asrRes
    decode0 := nn.eraseShape d0
    decode1 := nn.eraseShape d1
    decode2 := nn.eraseShape d2
    decode3 := nn.eraseShape d3
    generator
  }

def forward {frames : UInt64}
    (m : Decoder cfg)
    (asr : T #[1, cfg.hiddenDim, frames])
    (f0Curve : T #[1, 1, 2 * frames])
    (nCurve : T #[1, 1, 2 * frames])
    (style : T #[1, cfg.styleDim])
    : IO (T #[]) := do
  let f0 : T #[1, 1, frames] := m.f0Conv.forward f0Curve
  let n : T #[1, 1, frames] := m.nConv.forward nCurve
  let x0 : T #[1, cfg.hiddenDim + 2, frames] := nn.cat asr (nn.cat f0 n 1) 1
  let x1 : T #[1, cfg.maxConvDim, frames] := m.encode.forward x0 style
  let asrRes : T #[1, cfg.asrResDim, frames] := m.asrRes.forward asr
  let d0In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat x1 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d0 : T #[1, cfg.maxConvDim, frames] := m.decode0.forward d0In style
  let d1In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat d0 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d1 : T #[1, cfg.maxConvDim, frames] := m.decode1.forward d1In style
  let d2In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat d1 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d2 : T #[1, cfg.maxConvDim, frames] := m.decode2.forward d2In style
  let d3In : T #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] := nn.cat d2 (nn.cat asrRes (nn.cat f0 n 1) 1) 1
  let d3 : T #[1, KittenTTSConfig.decoderChannels cfg, 2 * frames] := m.decode3.forward d3In style
  m.generator.forward d3 style f0Curve

end Decoder

structure KittenTTSOutput where
  audio : T #[]
  predDurations : Array UInt64
  alignmentFrames : UInt64
  deriving TensorStruct

structure KittenTTSDebugOutput where
  asr : T #[]
  f0Curve : T #[]
  nCurve : T #[]
  audio : T #[]
  predDurations : Array UInt64
  alignmentFrames : UInt64

structure KittenTTSDurationPrediction where
  durVals : Array Float
  predDurations : Array UInt64
  alignmentFrames : UInt64
  deriving Inhabited, Repr

structure Model (cfg : KittenTTSConfig) where
  bert : KittenAlbert cfg.nToken cfg.plbert
  bertEncoder : LinearNorm cfg.plbert.hiddenSize cfg.hiddenDim
  predictor : ProsodyPredictor cfg
  textEncoder : TextEncoder cfg
  decoder : Decoder cfg
  deriving TensorStruct

namespace Model

private def modelDevice (m : Model cfg) : Device :=
  m.bert.embeddings.wordEmbeddings.device

private structure PreparedSynthesis where
  decoderStyle : T #[]
  asr : T #[]
  f0Curve : T #[]
  nCurve : T #[]
  predDurations : Array UInt64
  alignmentFrames : UInt64

private def prepareInputs {seq : UInt64}
    (m : Model cfg)
    (inputIds : T #[1, seq])
    (refStyle : T #[1, KittenTTSConfig.fullStyleDim cfg])
    : T #[1, seq] × T #[1, KittenTTSConfig.fullStyleDim cfg] :=
  let device := modelDevice m
  let inputIds :=
    if inputIds.device == device then inputIds else inputIds.to device
  let refStyle :=
    if refStyle.device == device then refStyle else refStyle.to device
  (inputIds, refStyle)

def init (cfg : KittenTTSConfig) : IO (Model cfg) := do
  if cfg.hiddenDim % 2 != 0 then
    throw <| IO.userError s!"KittenTTS requires hiddenDim divisible by 2, got {cfg.hiddenDim}"
  if cfg.plbert.hiddenSize % cfg.plbert.numAttentionHeads != 0 then
    throw <| IO.userError
      s!"KittenTTS ALBERT hidden size must be divisible by numAttentionHeads, got {cfg.plbert.hiddenSize} and {cfg.plbert.numAttentionHeads}"
  if cfg.generator.upsampleRates.isEmpty then
    throw <| IO.userError "KittenTTS generator requires at least one upsample stage"
  if cfg.generator.upsampleKernelSizes.size != cfg.generator.upsampleRates.size then
    throw <| IO.userError
      s!"KittenTTS generator upsampleKernelSizes/upsampleRates mismatch: {cfg.generator.upsampleKernelSizes.size} vs {cfg.generator.upsampleRates.size}"
  if cfg.generator.resblockKernelSizes.isEmpty then
    throw <| IO.userError "KittenTTS generator requires at least one resblock kernel size"
  if cfg.generator.resblockDilationSizes.size != cfg.generator.resblockKernelSizes.size then
    throw <| IO.userError
      s!"KittenTTS generator resblockDilationSizes/resblockKernelSizes mismatch: {cfg.generator.resblockDilationSizes.size} vs {cfg.generator.resblockKernelSizes.size}"
  if GeneratorConfig.finalChannels cfg.generator == 0 then
    throw <| IO.userError
      s!"KittenTTS generator final channel count collapsed to zero; upsampleInitialChannel={cfg.generator.upsampleInitialChannel}, stages={cfg.generator.upsampleRates.size}"
  if KittenTTSConfig.decoderChannels cfg != cfg.generator.upsampleInitialChannel then
    throw <| IO.userError
      s!"KittenTTS decoder/generator channel mismatch: decoderChannels={KittenTTSConfig.decoderChannels cfg}, upsampleInitialChannel={cfg.generator.upsampleInitialChannel}"
  let bert ← KittenAlbert.init cfg.nToken cfg.plbert
  let bertEncoder ← LinearNorm.init cfg.plbert.hiddenSize cfg.hiddenDim
  let predictor ← ProsodyPredictor.init cfg
  let textEncoder ← TextEncoder.init cfg
  let decoder ← Decoder.init cfg
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

def predictDurations {seq : UInt64}
    (m : Model cfg)
    (inputIds : T #[1, seq])
    (refStyle : T #[1, KittenTTSConfig.fullStyleDim cfg])
    (speed : Float := 1.0)
    : IO KittenTTSDurationPrediction := do
  let (inputIds, refStyle) := prepareInputs m inputIds refStyle
  if seq == 0 then
    pure {
      durVals := #[]
      predDurations := #[]
      alignmentFrames := 0
    }
  else
    let (bertOut, _pooled) := m.bert.forward inputIds
    let dEnSeq : T #[1, seq, cfg.hiddenDim] := m.bertEncoder.forward3d bertOut
    let dEn : T #[1, cfg.hiddenDim, seq] := seqToCF dEnSeq

    let predictorStyle : T #[1, cfg.styleDim] := data.slice refStyle 1 cfg.styleDim cfg.styleDim

    let durEnc : T #[1, cfg.hiddenDim + cfg.styleDim, seq] :=
      m.predictor.durationEncoding dEn predictorStyle
    let durLogits : T #[1, seq, cfg.maxDur] :=
      m.predictor.durationLogits durEnc
    let durSig : T #[1, seq, cfg.maxDur] := nn.sigmoid durLogits
    let durSum : T #[1, seq] := reshape (nn.sumDim durSig 2 false) #[1, seq]
    let durVals ← data.tensorToFloatArray' (reshape durSum #[])
    let predDurations := durationToFrames durVals speed
    let frames := predDurations.foldl (fun acc x => acc + x) 0
    pure {
      durVals
      predDurations
      alignmentFrames := frames
    }

private def prepareSynthesis {seq : UInt64}
    (m : Model cfg)
    (inputIds : T #[1, seq])
    (refStyle : T #[1, KittenTTSConfig.fullStyleDim cfg])
    (speed : Float := 1.0)
    : IO (Option PreparedSynthesis) := do
  let (inputIds, refStyle) := prepareInputs m inputIds refStyle
  let durPred ← predictDurations m inputIds refStyle speed
  if durPred.alignmentFrames == 0 then
    pure none
  else
    let frames := durPred.alignmentFrames
    let decoderStyle : T #[1, cfg.styleDim] := data.slice refStyle 1 0 cfg.styleDim
    let predictorStyle : T #[1, cfg.styleDim] := data.slice refStyle 1 cfg.styleDim cfg.styleDim
    let (bertOut, _pooled) := m.bert.forward inputIds
    let dEnSeq : T #[1, seq, cfg.hiddenDim] := m.bertEncoder.forward3d bertOut
    let dEn : T #[1, cfg.hiddenDim, seq] := seqToCF dEnSeq
    let durEnc : T #[1, cfg.hiddenDim + cfg.styleDim, seq] :=
      m.predictor.durationEncoding dEn predictorStyle
    let align : T #[1, seq, frames] := buildAlignment durPred.predDurations inputIds.device
    let prosodyFrames : T #[1, cfg.hiddenDim + cfg.styleDim, frames] := nn.bmm durEnc align
    let (f0Curve, nCurve) := m.predictor.forwardF0N prosodyFrames predictorStyle
    let textEnc : T #[1, cfg.hiddenDim, seq] := m.textEncoder.forward inputIds
    let asr : T #[1, cfg.hiddenDim, frames] := nn.bmm textEnc align
    pure <| some {
      decoderStyle := nn.eraseShape decoderStyle
      asr := nn.eraseShape asr
      f0Curve := nn.eraseShape f0Curve
      nCurve := nn.eraseShape nCurve
      predDurations := durPred.predDurations
      alignmentFrames := frames
    }

def synthesizeIds {seq : UInt64}
    (m : Model cfg)
    (inputIds : T #[1, seq])
    (refStyle : T #[1, KittenTTSConfig.fullStyleDim cfg])
    (speed : Float := 1.0)
    : IO KittenTTSOutput := do
  match (← prepareSynthesis m inputIds refStyle speed) with
  | none =>
    pure {
      audio := nn.eraseShape (torch.zeros #[1, 1, 0] false (modelDevice m))
      predDurations := #[]
      alignmentFrames := 0
    }
  | some prepared =>
    let frames := prepared.alignmentFrames
    let decoderStyle : T #[1, cfg.styleDim] := reshape prepared.decoderStyle #[1, cfg.styleDim]
    let asr : T #[1, cfg.hiddenDim, frames] := reshape prepared.asr #[1, cfg.hiddenDim, frames]
    let f0Curve : T #[1, 1, 2 * frames] := reshape prepared.f0Curve #[1, 1, 2 * frames]
    let nCurve : T #[1, 1, 2 * frames] := reshape prepared.nCurve #[1, 1, 2 * frames]
    let audio ← m.decoder.forward asr f0Curve nCurve decoderStyle
    pure {
      audio
      predDurations := prepared.predDurations
      alignmentFrames := frames
    }

def debugSynthesizeIds {seq : UInt64}
    (m : Model cfg)
    (inputIds : T #[1, seq])
    (refStyle : T #[1, KittenTTSConfig.fullStyleDim cfg])
    (speed : Float := 1.0)
    : IO KittenTTSDebugOutput := do
  match (← prepareSynthesis m inputIds refStyle speed) with
  | none =>
    let device := modelDevice m
    pure {
      asr := nn.eraseShape (torch.zeros #[1, cfg.hiddenDim, 0] false device)
      f0Curve := nn.eraseShape (torch.zeros #[1, 1, 0] false device)
      nCurve := nn.eraseShape (torch.zeros #[1, 1, 0] false device)
      audio := nn.eraseShape (torch.zeros #[1, 1, 0] false device)
      predDurations := #[]
      alignmentFrames := 0
    }
  | some prepared =>
    let frames := prepared.alignmentFrames
    let decoderStyle : T #[1, cfg.styleDim] := reshape prepared.decoderStyle #[1, cfg.styleDim]
    let asr : T #[1, cfg.hiddenDim, frames] := reshape prepared.asr #[1, cfg.hiddenDim, frames]
    let f0Curve : T #[1, 1, 2 * frames] := reshape prepared.f0Curve #[1, 1, 2 * frames]
    let nCurve : T #[1, 1, 2 * frames] := reshape prepared.nCurve #[1, 1, 2 * frames]
    let audio ← m.decoder.forward asr f0Curve nCurve decoderStyle
    pure {
      asr := prepared.asr
      f0Curve := prepared.f0Curve
      nCurve := prepared.nCurve
      audio := audio
      predDurations := prepared.predDurations
      alignmentFrames := frames
    }

end Model

end torch.kittentts
