/-
  Tyr/Model/Whisper/Config.lean

  Configuration for Whisper encoder-decoder ASR in Tyr.
-/
import Tyr.Basic

/-! # Whisper configuration

`torch.whisper.WhisperConfig` with derived head dimensions and the encoder
sequence length after the second convolution.
-/

namespace torch.whisper

structure WhisperConfig where
  numMelBins : UInt64 := 80
  vocabSize : UInt64 := 51865
  dModel : UInt64 := 384
  encoderLayers : UInt64 := 4
  encoderAttentionHeads : UInt64 := 6
  encoderFfnDim : UInt64 := 1536
  decoderLayers : UInt64 := 4
  decoderAttentionHeads : UInt64 := 6
  decoderFfnDim : UInt64 := 1536
  maxSourcePositions : UInt64 := 1500
  maxTargetPositions : UInt64 := 448
  activationFunction : String := "gelu"
  layerNormEps : Float := 1e-5
  padTokenId : UInt64 := 50257
  bosTokenId : UInt64 := 50257
  eosTokenId : UInt64 := 50257
  decoderStartTokenId : UInt64 := 50258
  suppressTokens : Array UInt64 := #[]
  beginSuppressTokens : Array UInt64 := #[]
  deriving Repr, Inhabited

namespace WhisperConfig

def encoderHeadDim (cfg : WhisperConfig) : UInt64 :=
  cfg.dModel / cfg.encoderAttentionHeads

def decoderHeadDim (cfg : WhisperConfig) : UInt64 :=
  cfg.dModel / cfg.decoderAttentionHeads

def conv2OutputSeq (frames : UInt64) : UInt64 :=
  (frames + 2 - (3 - 1) - 1) / 2 + 1

def hasValidHeads (cfg : WhisperConfig) : Bool :=
  cfg.encoderAttentionHeads > 0 &&
  cfg.decoderAttentionHeads > 0 &&
  cfg.dModel % cfg.encoderAttentionHeads == 0 &&
  cfg.dModel % cfg.decoderAttentionHeads == 0

end WhisperConfig

end torch.whisper
