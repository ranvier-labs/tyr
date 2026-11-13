/-
  Tyr/Model/Gemma4.lean

  Standalone Gemma 4 text causal-LM for Tyr.
-/
import Tyr.Model.Gemma4.Config
import Tyr.Model.Gemma4.ConfigIO
import Tyr.Model.Gemma4.Model
import Tyr.Model.Gemma4.Weights
import Tyr.Model.Gemma4.Pretrained
import Tyr.Model.Gemma4.VLConfig
import Tyr.Model.Gemma4.VLConfigIO
import Tyr.Model.Gemma4.Media
import Tyr.Model.Gemma4.Multimodal
import Tyr.Model.Gemma4.VLWeights

/-! # Gemma 4

Umbrella import for Gemma 4 in `torch.gemma4`: text config and JSON loading, the
causal LM, SafeTensors weights, Hugging Face download helpers, and the
multimodal (vision) config, media loading, vision tower and weights.
-/
