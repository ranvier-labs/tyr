/-
  Tyr/Model/VAE/ResnetBlock.lean

  Residual block for VAE decoder.
  Uses GroupNorm and Swish activation.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive

/-! # VAE ResNet block

`torch.vae.GroupNormParams`, `torch.vae.Conv2dParams`, the swish activation and
`torch.vae.ResnetBlock`, with optional channel change via a shortcut.
-/

namespace torch.vae

/-- GroupNorm parameters for VAE.
    Weight and bias for affine transformation. -/
structure GroupNormParams (num_groups channels : UInt64) where
  weight : T #[channels]
  bias : T #[channels]
  deriving TensorStruct

namespace GroupNormParams

/-- Initialize GroupNorm parameters (weight=1, bias=0) -/
def init (num_groups channels : UInt64) : GroupNormParams num_groups channels :=
  let weight := autograd.set_requires_grad (torch.ones #[channels]) true
  let bias := autograd.set_requires_grad (torch.zeros #[channels]) true
  { weight, bias }

/-- Apply GroupNorm to input tensor [batch, channels, height, width] -/
def forward {batch channels height width num_groups : UInt64}
    (gn : GroupNormParams num_groups channels)
    (x : T #[batch, channels, height, width])
    (eps : Float := 1e-6)
    : T #[batch, channels, height, width] :=
  nn.group_norm x num_groups (some gn.weight) (some gn.bias) eps

end GroupNormParams

/-- Conv2d parameters for VAE.
    Weight and bias for standard conv2d. -/
structure Conv2dParams (in_ch out_ch kernel_size : UInt64) (padding : UInt64 := 0) where
  weight : T #[out_ch, in_ch, kernel_size, kernel_size]
  bias : T #[out_ch]
  deriving TensorStruct

namespace Conv2dParams

/-- Initialize Conv2d with Kaiming init -/
def init (in_ch out_ch kernel_size : UInt64) (padding : UInt64 := 0) : IO (Conv2dParams in_ch out_ch kernel_size padding) := do
  let std := Float.sqrt (2.0 / (in_ch * kernel_size * kernel_size).toFloat)
  let w ← torch.randn #[out_ch, in_ch, kernel_size, kernel_size]
  let b := torch.zeros #[out_ch]
  pure {
    weight := autograd.set_requires_grad (mul_scalar w std) true
    bias := autograd.set_requires_grad b true
  }

/-- Apply convolution -/
def forward {batch in_ch out_ch kernel_size padding height width : UInt64}
    (conv : Conv2dParams in_ch out_ch kernel_size padding)
    (x : T #[batch, in_ch, height, width])
    : T #[] :=  -- Shape depends on padding, computed at runtime
  nn.conv2d x conv.weight #[1, 1] #[padding.toNat.toUInt64, padding.toNat.toUInt64]

end Conv2dParams

/-- Swish activation: x * sigmoid(x) -/
def swish {s : Shape} (x : T s) : T s :=
  x * nn.sigmoid x

/-- Residual block for VAE decoder.
    norm1 → swish → conv1 → norm2 → swish → conv2 (+ skip connection) -/
structure ResnetBlock (in_ch out_ch : UInt64) where
  /-- First normalization -/
  norm1 : GroupNormParams 32 in_ch
  /-- First convolution 3×3 -/
  conv1 : Conv2dParams in_ch out_ch 3 1
  /-- Second normalization -/
  norm2 : GroupNormParams 32 out_ch
  /-- Second convolution 3×3 -/
  conv2 : Conv2dParams out_ch out_ch 3 1
  /-- Skip connection (1×1 conv if in_ch ≠ out_ch) -/
  nin_shortcut : Option (Conv2dParams in_ch out_ch 1 0)
  deriving TensorStruct

namespace ResnetBlock

/-- Initialize ResnetBlock -/
def init (in_ch out_ch : UInt64) : IO (ResnetBlock in_ch out_ch) := do
  let norm1 := GroupNormParams.init 32 in_ch
  let conv1 ← Conv2dParams.init in_ch out_ch 3 1
  let norm2 := GroupNormParams.init 32 out_ch
  let conv2 ← Conv2dParams.init out_ch out_ch 3 1
  let nin_shortcut ← if in_ch != out_ch then do
    let c ← Conv2dParams.init in_ch out_ch 1 0
    pure (some c)
  else
    pure none
  pure { norm1, conv1, norm2, conv2, nin_shortcut }

/-- Forward pass for ResnetBlock.
    Input: [batch, in_ch, height, width]
    Output: [batch, out_ch, height, width] -/
def forward {batch in_ch out_ch height width : UInt64}
    (block : ResnetBlock in_ch out_ch)
    (x : T #[batch, in_ch, height, width])
    : T #[batch, out_ch, height, width] :=
  -- Main path
  let h := block.norm1.forward x
  let h := swish h
  let h := nn.conv2d_bias h block.conv1.weight block.conv1.bias #[1, 1] #[1, 1]
  let h := reshape h #[batch, out_ch, height, width]

  let h := block.norm2.forward h
  let h := swish h
  let h := nn.conv2d_bias h block.conv2.weight block.conv2.bias #[1, 1] #[1, 1]
  let h := reshape h #[batch, out_ch, height, width]

  -- Skip connection
  let skip := match block.nin_shortcut with
    | some conv => reshape (nn.conv2d_bias x conv.weight conv.bias #[1, 1] #[0, 0]) #[batch, out_ch, height, width]
    | none => reshape x #[batch, out_ch, height, width]

  h + skip

end ResnetBlock

end torch.vae
