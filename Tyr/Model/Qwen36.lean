/-
  Tyr/Model/Qwen36.lean

  Standalone Qwen3.6 causal-LM / multimodal surface for Tyr.
  This reuses the shared Qwen3.5-MoE implementation while exposing
  Qwen3.6-specific defaults and pretrained helpers.
-/
import Tyr.Model.Qwen35.Model
import Tyr.Model.Qwen35.Multimodal
import Tyr.Model.Qwen36.Media
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.VLWeights
import Tyr.Model.Qwen36.Config
import Tyr.Model.Qwen36.ConfigIO
import Tyr.Model.Qwen36.Pretrained

/-! # Qwen3.6

Umbrella import for Qwen3.6 in `torch.qwen36`, which reuses the Qwen3.5 model,
multimodal and weight modules and adds its own config, media and pretrained
entry points.
-/
