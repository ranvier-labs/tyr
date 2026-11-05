/-
  Tyr/Model/SileroVAD.lean

  Silero-VAD Lean4 port:
  - tiny model
  - pretrained safetensors loading
  - timestamp / iterator utilities
-/
import Tyr.Model.SileroVAD.Model
import Tyr.Model.SileroVAD.Utils

/-! # Silero VAD

Umbrella import for the Silero voice-activity detector in `torch.silerovad`:
model and timestamp utilities.
-/
