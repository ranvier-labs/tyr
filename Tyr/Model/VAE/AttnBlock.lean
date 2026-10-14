/-
  Tyr/Model/VAE/AttnBlock.lean

  Self-attention block for VAE decoder.
  Single-head spatial attention.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.VAE.ResnetBlock

/-! # VAE attention block

`torch.vae.AttnBlock`: single-head spatial self-attention with GroupNorm and
residual connection.
-/

namespace torch.vae

/-- Self-attention block for VAE.
    Applies spatial attention using 1×1 convolutions. -/
structure AttnBlock (channels : UInt64) where
  /-- GroupNorm before attention -/
  norm : GroupNormParams 32 channels
  /-- Query projection (1×1 conv) -/
  q : Conv2dParams channels channels 1 0
  /-- Key projection (1×1 conv) -/
  k : Conv2dParams channels channels 1 0
  /-- Value projection (1×1 conv) -/
  v : Conv2dParams channels channels 1 0
  /-- Output projection (1×1 conv) -/
  proj_out : Conv2dParams channels channels 1 0
  deriving TensorStruct

namespace AttnBlock

/-- Initialize attention block -/
def init (channels : UInt64) : IO (AttnBlock channels) := do
  let norm := GroupNormParams.init 32 channels
  let q ← Conv2dParams.init channels channels 1 0
  let k ← Conv2dParams.init channels channels 1 0
  let v ← Conv2dParams.init channels channels 1 0
  let proj_out ← Conv2dParams.init channels channels 1 0
  pure { norm, q, k, v, proj_out }

/-- Forward pass for attention block.
    Input: [batch, channels, height, width]
    Output: [batch, channels, height, width]

    Applies single-head spatial self-attention. -/
def forward {batch channels height width : UInt64}
    (attn : AttnBlock channels)
    (x : T #[batch, channels, height, width])
    : T #[batch, channels, height, width] :=
  let h := attn.norm.forward x

  -- Project to Q, K, V
  let q := nn.conv2d_bias h attn.q.weight attn.q.bias #[1, 1] #[0, 0]
  let k := nn.conv2d_bias h attn.k.weight attn.k.bias #[1, 1] #[0, 0]
  let v := nn.conv2d_bias h attn.v.weight attn.v.bias #[1, 1] #[0, 0]

  -- Reshape to [batch, channels, height*width]
  let spatial := height * width
  let q := reshape q #[batch, channels, spatial]
  let k := reshape k #[batch, channels, spatial]
  let v := reshape v #[batch, channels, spatial]

  -- Compute attention: softmax(Q^T K / sqrt(channels)) V
  -- Q: [batch, channels, spatial], K: [batch, channels, spatial]
  -- Q^T K: [batch, spatial, spatial]
  let q_t := nn.transpose q 1 2  -- [batch, spatial, channels]
  let scale := Float.sqrt channels.toFloat
  let attn_weights := nn.bmm q_t k  -- [batch, spatial, spatial]
  let attn_weights := attn_weights / scale
  let attn_weights := nn.softmax attn_weights (-1)

  -- V: [batch, channels, spatial]
  -- attn_weights @ V^T = [batch, spatial, spatial] @ [batch, spatial, channels]
  let v_t := nn.transpose v 1 2  -- [batch, spatial, channels]
  let out := nn.bmm attn_weights v_t  -- [batch, spatial, channels]
  let out := nn.transpose out 1 2  -- [batch, channels, spatial]

  -- Reshape back to spatial
  let out := reshape out #[batch, channels, height, width]

  -- Output projection + residual
  let out := nn.conv2d_bias out attn.proj_out.weight attn.proj_out.bias #[1, 1] #[0, 0]
  let out := reshape out #[batch, channels, height, width]

  x + out

end AttnBlock

end torch.vae
