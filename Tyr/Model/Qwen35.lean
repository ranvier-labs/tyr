/-
  Tyr/Model/Qwen35.lean

  Standalone Qwen3.5 text causal-LM (dense + MoE) for Tyr.
-/
import Tyr.Model.Qwen35.Config
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Model
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.VLConfig
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.Multimodal
import Tyr.Model.Qwen35.Media
import Tyr.Model.Qwen35.VLWeights
import Tyr.Model.Qwen35.Pretrained

/-! # Qwen3.5

Umbrella import for Qwen3.5 in `torch.qwen35`: hybrid linear/full-attention text
model, MoE weights, multimodal config, vision tower, media loading and
pretrained loading.
-/
