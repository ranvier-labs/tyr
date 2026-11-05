/-
  Tyr/Model/Qwen3.lean

  Standalone Qwen3 causal-LM (config IO, model, weights).
  This is additive and kept separate from the existing Flux-oriented Qwen module.
-/
import Tyr.Model.Qwen3.Config
import Tyr.Model.Qwen3.ConfigIO
import Tyr.Model.Qwen3.Model
import Tyr.Model.Qwen3.Weights

/-! # Qwen3 causal LM

Umbrella import for the standalone Qwen3 causal language model in `torch.qwen3`:
config, JSON loading, model and weights.
-/
