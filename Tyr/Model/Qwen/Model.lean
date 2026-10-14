/-
  Tyr/Model/Qwen/Model.lean

  Full Qwen3 model for text encoding.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE

/-! # Qwen3 decoder

`torch.qwen.Qwen3Model`: token embedding, the layer stack and final norm, with
forward passes that also return hidden states from every layer.
-/

namespace torch.qwen

/-- Qwen3 model structure.
    Uses type-level parameters for dimensions. -/
structure Qwen3Model (cfg : QwenConfig) where
  /-- Token embeddings: [vocab_size, hidden_size] -/
  embed_tokens : T #[cfg.vocab_size, cfg.hidden_size]
  /-- Transformer layers -/
  layers : Array (QwenLayer cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size)
  /-- Final layer norm -/
  norm : RMSNorm cfg.hidden_size
  deriving TensorStruct

namespace Qwen3Model

/-- Initialize a Qwen3 model with random weights -/
def init (cfg : QwenConfig) : IO (Qwen3Model cfg) := do
  -- Token embeddings
  let emb ← torch.randn #[cfg.vocab_size, cfg.hidden_size]
  let std := 0.02  -- Standard embedding init
  let embed_tokens := autograd.set_requires_grad (mul_scalar emb std) true

  -- Initialize layers
  let mut layers := #[]
  for _ in [:cfg.num_hidden_layers.toNat] do
    let layer ← QwenLayer.init cfg.hidden_size cfg.num_attention_heads
        cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size cfg.rms_norm_eps
    layers := layers.push layer

  -- Final norm
  let norm := RMSNorm.init cfg.hidden_size cfg.rms_norm_eps

  pure { embed_tokens, layers, norm }

/-- Forward pass returning hidden states from all layers.
    Input: [batch, seq] token IDs
    Output: Array of [batch, seq, hidden_size] hidden states (one per layer + final)
    Rewritten in functional style to avoid Id.run issues with FFI shape tracking. -/
def forwardWithHiddenStates {batch seq : UInt64} (cfg : QwenConfig)
    (model : Qwen3Model cfg)
    (input_ids : T #[batch, seq])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    : Array (T #[batch, seq, cfg.hidden_size]) :=
  -- Embed tokens
  let hidden := nn.embedding input_ids model.embed_tokens
  -- Process each layer, accumulating (current_hidden, collected_states)
  let (finalH, hiddenStates) := model.layers.foldl
    (fun (acc : T #[batch, seq, cfg.hidden_size] × Array (T #[batch, seq, cfg.hidden_size])) layer =>
      let (h, states) := acc
      let h' := layer.forward h cos sin true
      (h', states.push h'))
    (hidden, #[])
  -- Add final normalized state
  let finalHidden := model.norm.forward3d finalH
  hiddenStates.push finalHidden

def forwardWithHiddenStatesMasked {batch seq : UInt64} (cfg : QwenConfig)
    (model : Qwen3Model cfg)
    (input_ids : T #[batch, seq])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    (attn_mask : T #[batch, seq])
    : Array (T #[batch, seq, cfg.hidden_size]) :=
  -- Embed tokens
  let hidden := nn.embedding input_ids model.embed_tokens
  -- Process each layer, accumulating (current_hidden, collected_states)
  let (finalH, hiddenStates) := model.layers.foldl
    (fun (acc : T #[batch, seq, cfg.hidden_size] × Array (T #[batch, seq, cfg.hidden_size])) layer =>
      let (h, states) := acc
      let h' := layer.forwardMasked h cos sin attn_mask true
      (h', states.push h'))
    (hidden, #[])
  -- Add final normalized state
  let finalHidden := model.norm.forward3d finalH
  hiddenStates.push finalHidden

/-- Forward pass returning only the final hidden state.
    Input: [batch, seq] token IDs
    Output: [batch, seq, hidden_size]
    Rewritten in functional style to avoid Id.run issues with FFI shape tracking. -/
def forward {batch seq : UInt64} (cfg : QwenConfig)
    (model : Qwen3Model cfg)
    (input_ids : T #[batch, seq])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    : T #[batch, seq, cfg.hidden_size] :=
  -- Embed tokens
  let hidden := nn.embedding input_ids model.embed_tokens
  -- Process each layer using fold
  let finalH := model.layers.foldl
    (fun h layer => layer.forward h cos sin true)
    hidden
  -- Final normalization
  model.norm.forward3d finalH

def forwardMasked {batch seq : UInt64} (cfg : QwenConfig)
    (model : Qwen3Model cfg)
    (input_ids : T #[batch, seq])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    (attn_mask : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  -- Embed tokens
  let hidden := nn.embedding input_ids model.embed_tokens
  -- Process each layer using fold
  let finalH := model.layers.foldl
    (fun h layer => layer.forwardMasked h cos sin attn_mask true)
    hidden
  -- Final normalization
  model.norm.forward3d finalH

end Qwen3Model

end torch.qwen
