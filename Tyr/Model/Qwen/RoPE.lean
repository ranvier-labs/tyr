/-
  Tyr/Model/Qwen/RoPE.lean

  Rotary Position Embeddings for Qwen3.
  Standard 2D rotation (different from Flux's 4-axis RoPE).
-/
import Tyr.Torch

/-! # Qwen RoPE

`torch.qwen.RoPECache` precomputes cos/sin tables for a sequence length;
`torch.qwen.applyRoPE` rotates query/key tensors.
-/

namespace torch.qwen

/-- Precomputed RoPE cos/sin embeddings for a given sequence length.
    These are cached and reused across forward passes. -/
structure RoPECache (max_seq : UInt64) (head_dim : UInt64) where
  /-- Cosine values: [max_seq, head_dim/2] -/
  cos : T #[max_seq, head_dim / 2]
  /-- Sine values: [max_seq, head_dim/2] -/
  sin : T #[max_seq, head_dim / 2]

namespace RoPECache

/-- Initialize RoPE cache with precomputed cos/sin values.
    Uses theta base for frequency computation. -/
def init (max_seq head_dim : UInt64) (theta : Float := 10000.0) : IO (RoPECache max_seq head_dim) := do
  let (cos, sin) ← rotary.computeFreqs max_seq head_dim theta
  pure { cos, sin }

end RoPECache

/-- Apply rotary position embeddings to query/key tensors.
    x: [batch, seq, n_head, head_dim]
    Returns: [batch, seq, n_head, head_dim] with positions encoded -/
def applyRoPE {batch seq n_head head_dim max_seq : UInt64}
    (x : T #[batch, seq, n_head, head_dim])
    (cache : RoPECache max_seq head_dim)
    (start_pos : UInt64 := 0)
    : T #[batch, seq, n_head, head_dim] :=
  -- Slice cos/sin to current sequence length
  let cos := data.slice cache.cos 0 start_pos seq
  let sin := data.slice cache.sin 0 start_pos seq
  -- Apply rotation
  rotary.applyRotaryEmb x cos sin

end torch.qwen
