/-
  Tyr/Model/Qwen/MLP.lean

  SwiGLU MLP for Qwen3.
  SwiGLU: down(silu(gate(x)) * up(x))
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive

/-! # Qwen MLP

`torch.qwen.QwenMLP` computes
$W_{down}(\operatorname{SiLU}(W_{gate} x) \odot W_{up} x)$.
-/

namespace torch.qwen

/-- Qwen MLP with SwiGLU activation.
    SwiGLU is more expressive than standard GELU MLP. -/
structure QwenMLP (hidden_size intermediate_size : UInt64) where
  /-- Gate projection: [intermediate_size, hidden_size] -/
  gate_proj : T #[intermediate_size, hidden_size]
  /-- Up projection: [intermediate_size, hidden_size] -/
  up_proj : T #[intermediate_size, hidden_size]
  /-- Down projection: [hidden_size, intermediate_size] -/
  down_proj : T #[hidden_size, intermediate_size]
  deriving TensorStruct

namespace QwenMLP

/-- Initialize MLP layers with random weights -/
def init (hidden_size intermediate_size : UInt64) : IO (QwenMLP hidden_size intermediate_size) := do
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let gate ← torch.randn #[intermediate_size, hidden_size]
  let up ← torch.randn #[intermediate_size, hidden_size]
  let down ← torch.randn #[hidden_size, intermediate_size]

  pure {
    gate_proj := autograd.set_requires_grad (mul_scalar gate std) true
    up_proj := autograd.set_requires_grad (mul_scalar up std) true
    down_proj := autograd.set_requires_grad (mul_scalar down std) true
  }

/-- Forward pass for MLP.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size]

    SwiGLU: down(silu(gate(x)) * up(x)) -/
def forward {batch seq hidden_size intermediate_size : UInt64}
    (mlp : QwenMLP hidden_size intermediate_size)
    (x : T #[batch, seq, hidden_size])
    : T #[batch, seq, hidden_size] :=
  -- Compute gate and up projections
  let gate := linear3d x mlp.gate_proj  -- [batch, seq, intermediate_size]
  let up := linear3d x mlp.up_proj      -- [batch, seq, intermediate_size]

  -- SwiGLU: silu(gate) * up
  let gate_silu := nn.silu gate
  let hidden := gate_silu * up

  -- Down projection
  linear3d hidden mlp.down_proj

end QwenMLP

end torch.qwen
