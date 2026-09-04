import Tyr.Typed

open torch
open torch.Tensor

-- The batch dimension and both feature dimensions are visible in the type.
def project
    (x : Tensor { shape := #[32, 768], dtype := .Float32 })
    (weight : Tensor { shape := #[512, 768], dtype := .Float32 }) :
    Tensor { shape := #[32, 512], dtype := .Float32 } :=
  linear x weight

#check project
