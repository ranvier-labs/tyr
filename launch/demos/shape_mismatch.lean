import Tyr.Typed

open torch
open torch.Tensor

-- This file is intentionally rejected: the weight's input dimension is 512,
-- while the activation's feature dimension is 768.
def brokenProject
    (x : Tensor { shape := #[32, 768], dtype := .Float32 })
    (weight : Tensor { shape := #[768, 512], dtype := .Float32 }) :
    Tensor { shape := #[32, 768], dtype := .Float32 } :=
  linear x weight
