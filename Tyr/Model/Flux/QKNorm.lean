/-
  Tyr/Model/Flux/QKNorm.lean

  Query-Key normalization for Flux attention.
  Applies RMSNorm to Q and K independently.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm

/-! # Query-key normalization

`torch.flux.QKNorm` applies RMS normalization to queries and keys before
attention, which stabilises the attention logits;
`torch.flux.QKNorm.forwardAttn` works on the attention layout.
-/

namespace torch.flux

/-- Query-Key normalization.
    Applies RMSNorm to queries and keys for stable attention. -/
structure QKNorm (head_dim : UInt64) where
  query_norm : RMSNorm head_dim
  key_norm : RMSNorm head_dim
  deriving TensorStruct

namespace QKNorm

/-- Initialize QK normalization -/
def init (head_dim : UInt64) (eps : Float := 1e-6) : QKNorm head_dim :=
  { query_norm := RMSNorm.init head_dim eps
  , key_norm := RMSNorm.init head_dim eps }

/-- Forward pass: normalize Q and K.
    q, k: [batch, seq, n_head, head_dim] or [batch, n_head, seq, head_dim]
    Returns normalized (q, k) -/
def forward {batch seq n_head head_dim : UInt64}
    (qkn : QKNorm head_dim)
    (q : T #[batch, seq, n_head, head_dim])
    (k : T #[batch, seq, n_head, head_dim])
    : T #[batch, seq, n_head, head_dim] × T #[batch, seq, n_head, head_dim] :=
  let q_norm := qkn.query_norm.forward4d q
  let k_norm := qkn.key_norm.forward4d k
  (q_norm, k_norm)

/-- Forward for attention-transposed format: [batch, n_head, seq, head_dim] -/
def forwardAttn {batch n_head seq head_dim : UInt64}
    (qkn : QKNorm head_dim)
    (q : T #[batch, n_head, seq, head_dim])
    (k : T #[batch, n_head, seq, head_dim])
    : T #[batch, n_head, seq, head_dim] × T #[batch, n_head, seq, head_dim] :=
  let q_norm := qkn.query_norm.forward5d q
  let k_norm := qkn.key_norm.forward5d k
  (q_norm, k_norm)

end QKNorm

end torch.flux
