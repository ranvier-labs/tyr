/-
  Tyr/Model/Qwen/Embedder.lean

  Flux text embedder using Qwen3.
  Extracts hidden states from specific layers and concatenates them.
-/
import Tyr.Torch
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.RoPE

/-! # Qwen text embedder for Flux

`torch.qwen.QwenFluxEmbedder` encodes prompts by concatenating hidden states
from selected Qwen layers (`torch.qwen.FluxEmbedderConfig`); output width is
number of layers × hidden size.
-/

namespace torch.qwen

/-- Configuration for Flux text embedding.
    Specifies which layers to extract hidden states from. -/
structure FluxEmbedderConfig where
  /-- Layer indices to extract (0-indexed) -/
  output_layers : Array UInt64 := #[8, 17, 26]
  /-- Maximum sequence length -/
  max_seq_len : UInt64 := 512
  deriving Repr, Inhabited

/-- Qwen-based text embedder for Flux.
    Extracts hidden states from specified layers and concatenates them. -/
structure QwenFluxEmbedder (cfg : QwenConfig) (max_seq : UInt64) where
  /-- The Qwen3 model -/
  model : Qwen3Model cfg
  /-- RoPE cache for position embeddings -/
  ropeCache : RoPECache max_seq cfg.head_dim
  /-- Which layers to extract -/
  outputLayers : Array UInt64

namespace QwenFluxEmbedder

/-- Compute output dimension: num_layers * hidden_size -/
def outputDim (cfg : QwenConfig) (numLayers : UInt64) : UInt64 :=
  numLayers * cfg.hidden_size

/-- Create a text embedder from a Qwen3 model -/
def fromModel (cfg : QwenConfig) (max_seq : UInt64) (model : Qwen3Model cfg)
    (outputLayers : Array UInt64 := #[8, 17, 26])
    : IO (QwenFluxEmbedder cfg max_seq) := do
  let ropeCache ← RoPECache.init max_seq cfg.head_dim cfg.rope_theta
  pure { model, ropeCache, outputLayers }

/-- Encode text tokens to Flux-compatible embeddings.
    Input: [batch, seq] token IDs
    Output: [batch, seq, output_dim] where output_dim = num_layers * hidden_size

    For Qwen3-4B with layers [8, 17, 26]:
    output_dim = 3 * 2560 = 7680

    Rewritten in functional style to avoid Id.run issues with FFI shape tracking. -/
def encode {batch seq : UInt64} (cfg : QwenConfig) (max_seq : UInt64)
    (embedder : QwenFluxEmbedder cfg max_seq)
    (input_ids : T #[batch, seq])
    : T #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size] :=
  -- Slice RoPE cache to current sequence length
  let cos := data.slice embedder.ropeCache.cos 0 0 seq
  let sin := data.slice embedder.ropeCache.sin 0 0 seq

  -- Get hidden states from all layers
  let allHiddenStates := embedder.model.forwardWithHiddenStates cfg input_ids cos sin

  -- Extract specified layers using filterMap
  let selectedStates : Array (T #[batch, seq, cfg.hidden_size]) :=
    embedder.outputLayers.filterMap fun layerIdx =>
      if h : layerIdx.toNat < allHiddenStates.size then
        some (allHiddenStates[layerIdx.toNat]'h)
      else
        none

  -- Concatenate along the last dimension
  let numLayers := selectedStates.size.toUInt64
  let outputDim := numLayers * cfg.hidden_size

  -- Concatenate selected states along dim 2
  if selectedStates.size > 0 then
    let concatenated := nn.cat_impl selectedStates 2
    reshape concatenated #[batch, seq, outputDim]
  else
    -- Fallback: return zeros if no layers specified
    torch.zeros #[batch, seq, outputDim]

def encodeMasked {batch seq : UInt64} (cfg : QwenConfig) (max_seq : UInt64)
    (embedder : QwenFluxEmbedder cfg max_seq)
    (input_ids : T #[batch, seq])
    (attn_mask : T #[batch, seq])
    : T #[batch, seq, embedder.outputLayers.size.toUInt64 * cfg.hidden_size] :=
  -- Slice RoPE cache to current sequence length
  let cos := data.slice embedder.ropeCache.cos 0 0 seq
  let sin := data.slice embedder.ropeCache.sin 0 0 seq

  -- Get hidden states from all layers
  let allHiddenStates := embedder.model.forwardWithHiddenStatesMasked cfg input_ids cos sin attn_mask

  -- Extract specified layers using filterMap
  let selectedStates : Array (T #[batch, seq, cfg.hidden_size]) :=
    embedder.outputLayers.filterMap fun layerIdx =>
      if h : layerIdx.toNat < allHiddenStates.size then
        some (allHiddenStates[layerIdx.toNat]'h)
      else
        none
  -- Concatenate along the last dimension
  let numLayers := selectedStates.size.toUInt64
  let outputDim := numLayers * cfg.hidden_size

  -- Concatenate selected states along dim 2
  if selectedStates.size > 0 then
    let concatenated := nn.cat_impl selectedStates 2
    reshape concatenated #[batch, seq, outputDim]
  else
    -- Fallback: return zeros if no layers specified
    torch.zeros #[batch, seq, outputDim]

end QwenFluxEmbedder

end torch.qwen
