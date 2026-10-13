/-
  Tyr/Model/Utils.lean

  Shared model utility functions used across multiple model families.
  These are higher-level than Tyr.Torch primitives and are specific to
  neural network initialization and token-level inference helpers.

  Note: `logicalOr`, `falseMask`, `zerosOn`, `onesOn`, `castLike`,
  and `restoreInputDType` are already provided by `Tyr.Torch`.
-/
import Tyr.Torch

/-! # Model utilities

Shared helpers for model code: Kaiming-like init with std
$\sqrt{2/\text{fanIn}}$ (`torch.Model.initWeight`), `torch.Model.initBias`,
freezing loaded tensors (`torch.Model.reqGradFalse`) and FP8 dequantization
(`torch.Model.dequantizeFP8`).
-/

namespace torch.Model

open torch

/-- Kaiming-like weight init: randn scaled by sqrt(2 / fanIn). -/
def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

/-- Zero bias tensor with requires_grad=true. -/
def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

/-- Mark a loaded pretrained tensor as frozen (requires_grad=false). -/
def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad t false

/-- Dequantize FP8 weights with blockwise inverse scales (128x128 blocks).
    weight: [out, in], scale_inv: [out/128, in/128]. Returns float32 weights. -/
def dequantizeFP8 {outDim inDim : UInt64}
    (weight : T #[outDim, inDim])
    (scaleInv : T #[outDim / 128, inDim / 128])
    : T #[outDim, inDim] :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := toFloat' weight
  let s := toFloat' scaleInv
  let w := reshape w #[outBlocks, 128, inBlocks, 128]
  let s := reshape s #[outBlocks, 1, inBlocks, 1]
  let s := nn.expand s #[outBlocks, 128, inBlocks, 128]
  let w := w * s
  reshape w #[outDim, inDim]

/-- Dequantize FP8 expert weights with blockwise inverse scales per expert.
    weight: [experts, out, in], scale_inv: [experts, out/128, in/128]. -/
def dequantizeFP8Experts {experts outDim inDim : UInt64}
    (weight : T #[experts, outDim, inDim])
    (scaleInv : T #[experts, outDim / 128, inDim / 128])
    : T #[experts, outDim, inDim] :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := toFloat' weight
  let s := toFloat' scaleInv
  let w := reshape w #[experts, outBlocks, 128, inBlocks, 128]
  let s := reshape s #[experts, outBlocks, 1, inBlocks, 1]
  let s := nn.expand s #[experts, outBlocks, 128, inBlocks, 128]
  let w := w * s
  reshape w #[experts, outDim, inDim]

end torch.Model
