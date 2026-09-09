/-
  Tyr/Module/RMSNorm.lean

  RMS Layer Normalization module with TensorStruct and Module instances.
  Used in LLaMA, Qwen, and other modern transformer architectures.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Widget

namespace torch

/-- RMS Layer Normalization: normalizes using root mean square.
    y = x / sqrt(mean(x^2) + eps) * weight
    Unlike LayerNorm, RMSNorm has no bias and doesn't subtract the mean. -/
structure RMSNorm (dim : UInt64) where
  weight : T #[dim]
  eps : Static Float := ⟨1e-6⟩
  deriving TensorStruct, ToModuleDisplay

namespace RMSNorm

/-- Accumulate normalization in Float32, then restore BF16 activations before
    multiplying by the learned weight (the LLaMA/Qwen operation order).
    Matching Float32/BFloat16 inputs and weights retain their model dtype.
    The functional `nn.rmsNormWeighted` API keeps its promoted result contract. -/
private def forwardImpl {dim : UInt64} {s : Shape} (rn : RMSNorm dim)
    (x : T s) : T s :=
  castLike x (nn.rmsNorm x rn.eps.val) * rn.weight

/-- Initialize RMSNorm with ones for weight -/
def init (dim : UInt64) (eps : Float := 1e-6) : RMSNorm dim :=
  let weight := autograd.set_requires_grad (torch.ones #[dim]) true
  { weight, eps := ⟨eps⟩ }

/-- Forward pass for 2D input [seq, dim] -/
def forward2d {dim seq : UInt64} (rn : RMSNorm dim)
    (x : T #[seq, dim]) : T #[seq, dim] :=
  forwardImpl rn x

/-- Forward pass for 3D input [batch, seq, dim] -/
def forward3d {dim batch seq : UInt64} (rn : RMSNorm dim)
    (x : T #[batch, seq, dim]) : T #[batch, seq, dim] :=
  forwardImpl rn x

/-- Forward pass for 4D input [batch, seq, n_head, head_dim]
    Normalizes over the last dimension (head_dim). -/
def forward4d {batch seq n_head head_dim : UInt64} (rn : RMSNorm head_dim)
    (x : T #[batch, seq, n_head, head_dim]) : T #[batch, seq, n_head, head_dim] :=
  forwardImpl rn x

/-- Forward pass for 4D input in attention format [batch, n_head, seq, head_dim]
    Normalizes over the last dimension (head_dim). -/
def forward5d {batch n_head seq head_dim : UInt64} (rn : RMSNorm head_dim)
    (x : T #[batch, n_head, seq, head_dim]) : T #[batch, n_head, seq, head_dim] :=
  forwardImpl rn x

end RMSNorm

/-- Module instance for 2D forward pass -/
instance {dim seq : UInt64} :
    Module (RMSNorm dim) (T #[seq, dim]) (T #[seq, dim]) where
  forward := RMSNorm.forward2d

/-- Module instance for 3D forward pass -/
instance {dim batch seq : UInt64} :
    Module (RMSNorm dim) (T #[batch, seq, dim]) (T #[batch, seq, dim]) where
  forward := RMSNorm.forward3d

end torch
