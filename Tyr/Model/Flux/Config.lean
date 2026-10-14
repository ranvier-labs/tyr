/-
  Tyr/Model/Flux/Config.lean

  Configuration for Flux diffusion model.
  Default values are for Klein 4B.
-/
import Tyr.Basic

/-! # Flux configuration

Hyperparameters for the Flux transformer. `torch.flux.FluxConfig` holds the
model dimensions, `torch.flux.FluxConfig.klein4B` is the Flux2 Klein 4B preset
(MLP ratio 3.0), and `torch.flux.FluxConfig.mlpHiddenDim` /
`torch.flux.FluxConfig.ropeDim` derive the MLP width and per-head rotary
dimension.
-/

namespace torch.flux

/-- Flux model configuration.
    Default values are for Klein 4B. -/
structure FluxConfig where
  /-- Hidden dimension -/
  hidden_size : UInt64 := 3072
  /-- Number of attention heads -/
  num_heads : UInt64 := 24
  /-- Head dimension -/
  head_dim : UInt64 := 128
  /-- Number of double-stream transformer blocks -/
  num_double_layers : UInt64 := 5
  /-- Number of single-stream transformer blocks -/
  num_single_layers : UInt64 := 20
  /-- Context (text) input dimension -/
  context_in_dim : UInt64 := 7680
  /-- Image input dimension (latent patch dim) -/
  in_channels : UInt64 := 128
  /-- Timestep embedding dimension -/
  time_dim : UInt64 := 256
  /-- MLP hidden dimension multiplier -/
  mlp_ratio : Float := 3.0
  /-- RoPE theta base -/
  theta : UInt64 := 2000
  /-- RoPE axes dimensions (for 4-axis RoPE) -/
  axes_dims : Array UInt64 := #[32, 32, 32, 32]
  deriving Repr, Inhabited

namespace FluxConfig

/-- Klein 4B configuration (Flux2 Klein uses mlp_ratio=3.0) -/
def klein4B : FluxConfig := {
  mlp_ratio := 3.0
}

/-- Compute MLP hidden dimension -/
def mlpHiddenDim (cfg : FluxConfig) : UInt64 :=
  (cfg.hidden_size.toFloat * cfg.mlp_ratio).toUInt64

/-- Total RoPE dimension (sum of axes) -/
def ropeDim (cfg : FluxConfig) : UInt64 :=
  cfg.axes_dims.foldl (· + ·) 0

end FluxConfig

end torch.flux
