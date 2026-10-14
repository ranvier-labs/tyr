/-
  Tyr/Model/Flux.lean

  Flux diffusion model for image generation.
  Re-exports all Flux components.
-/

import Tyr.Model.Flux.Config
import Tyr.Model.Flux.RoPE
import Tyr.Model.Flux.Modulation
import Tyr.Model.Flux.QKNorm
import Tyr.Model.Flux.SingleStreamBlock
import Tyr.Model.Flux.DoubleStreamBlock
import Tyr.Model.Flux.Model
import Tyr.Model.Flux.Sampling
import Tyr.Model.Flux.Weights

/-! # Flux

Umbrella import for the Flux2 (Klein) image-generation transformer in
`torch.flux`: configuration, 4-axis RoPE, timestep modulation, QK-norm,
single/double-stream blocks, the full model, flow-matching sampling and
checkpoint loading.
-/
