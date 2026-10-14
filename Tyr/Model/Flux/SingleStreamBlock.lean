/-
  Tyr/Model/Flux/SingleStreamBlock.lean

  Single-stream transformer block for Flux.
  Processes concatenated image and text tokens together.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.LayerNorm
import Tyr.Model.Flux.Config
import Tyr.Model.Flux.RoPE
import Tyr.Model.Flux.Modulation
import Tyr.Model.Flux.QKNorm

/-! # Flux single-stream block

`torch.flux.SingleStreamBlock` processes the concatenated image+text sequence
with one set of weights, computing attention and MLP in parallel from a fused
projection under timestep modulation.
-/

namespace torch.flux

/-- Single-stream transformer block.
    Processes concatenated image + text tokens with shared attention.
    Uses fused QKV + MLP projection for efficiency. -/
structure SingleStreamBlock (hidden_size num_heads head_dim mlp_hidden : UInt64) where
  /-- Pre-norm (LayerNorm) -/
  pre_norm : LayerNorm hidden_size
  /-- Fused projection for Q, K, V and MLP gate/up
      Q: [hidden, hidden], K: [hidden, hidden], V: [hidden, hidden]
      gate: [mlp_hidden, hidden], up: [mlp_hidden, hidden]
      Total: [hidden * 3 + mlp_hidden * 2, hidden] -/
  linear1 : T #[num_heads * head_dim * 3 + mlp_hidden * 2, hidden_size]
  /-- Fused output projection for attention + MLP down
      attn_out: [hidden, hidden], mlp_down: [hidden, mlp_hidden]
      Total: [hidden, hidden + mlp_hidden] -/
  linear2 : T #[hidden_size, num_heads * head_dim + mlp_hidden]
  /-- QK normalization -/
  norm : QKNorm head_dim
  deriving TensorStruct

namespace SingleStreamBlock

/-- Initialize single-stream block -/
def init (hidden_size num_heads head_dim mlp_hidden : UInt64)
    : IO (SingleStreamBlock hidden_size num_heads head_dim mlp_hidden) := do
  let pre_norm := LayerNorm.init hidden_size
  let qkv_dim := num_heads * head_dim * 3
  let mlp_in := mlp_hidden * 2
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let l1 ← torch.randn #[qkv_dim + mlp_in, hidden_size]
  let l2 ← torch.randn #[hidden_size, num_heads * head_dim + mlp_hidden]

  let norm := QKNorm.init head_dim
  pure {
    pre_norm
    linear1 := autograd.set_requires_grad (mul_scalar l1 std) true
    linear2 := autograd.set_requires_grad (mul_scalar l2 std) true
    norm
  }

/-- Forward pass for single-stream block.
    x: [batch, seq, hidden_size] - concatenated image + text tokens
    pe: RoPE embeddings
    mod: [batch, 3, hidden_size] - shift/scale/gate modulation
    Returns: [batch, seq, hidden_size] -/
def forward {batch seq hidden_size num_heads head_dim mlp_hidden : UInt64}
    (block : SingleStreamBlock hidden_size num_heads head_dim mlp_hidden)
    (x : T #[batch, seq, hidden_size])
    (pe : T #[])  -- Shape depends on RoPE implementation
    (mod : T #[batch, 3, hidden_size])
    : T #[batch, seq, hidden_size] :=
  -- Extract shift, scale, gate from modulation
  let mod_shift := reshape (data.slice mod 1 0 1) #[batch, hidden_size]
  let mod_scale := reshape (data.slice mod 1 1 1) #[batch, hidden_size]
  let mod_gate := reshape (data.slice mod 1 2 1) #[batch, hidden_size]

  -- Pre-norm with modulation
  let x_norm := block.pre_norm.forward3d x
  let x_mod := applyModulation x_norm mod_scale mod_shift

  -- Fused QKV + MLP projection
  let qkv_mlp := linear3d x_mod block.linear1  -- [batch, seq, qkv_dim + mlp_in]

  -- Split into QKV and MLP parts
  let qkv_dim := num_heads * head_dim * 3
  let qkv := data.slice qkv_mlp 2 0 qkv_dim
  let mlp := data.slice qkv_mlp 2 qkv_dim (mlp_hidden * 2)

  -- Reshape QKV
  let qkv := reshape qkv #[batch, seq, 3, num_heads, head_dim]
  let q := reshape (data.slice qkv 2 0 1) #[batch, seq, num_heads, head_dim]
  let k := reshape (data.slice qkv 2 1 1) #[batch, seq, num_heads, head_dim]
  let v := reshape (data.slice qkv 2 2 1) #[batch, seq, num_heads, head_dim]

  -- QK normalization
  let (q, k) := block.norm.forward q k

  -- Apply RoPE
  let q := flux.applyRope q pe
  let k := flux.applyRope k pe

  -- Transpose for attention: [batch, num_heads, seq, head_dim]
  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  -- Scaled dot-product attention (non-causal for diffusion)
  let attn := nn.scaled_dot_product_attention q k v 0.0 false

  -- Transpose back and reshape
  let attn := nn.transpose_from_attention attn
  let attn := reshape attn #[batch, seq, num_heads * head_dim]

  -- MLP: SwiGLU
  let mlp := reshape mlp #[batch, seq, 2, mlp_hidden]
  let gate := reshape (data.slice mlp 2 0 1) #[batch, seq, mlp_hidden]
  let up := reshape (data.slice mlp 2 1 1) #[batch, seq, mlp_hidden]
  let mlp_out := nn.silu gate * up

  -- Concatenate attention and MLP outputs
  let combined := nn.cat attn mlp_out 2  -- [batch, seq, hidden + mlp_hidden]

  -- Final projection
  let out := linear3d combined block.linear2  -- [batch, seq, hidden_size]

  -- Apply gate and residual
  let mod_gate := nn.unsqueeze mod_gate 1
  let mod_gate := nn.expand mod_gate #[batch, seq, hidden_size]
  x + mod_gate * out

end SingleStreamBlock

end torch.flux
