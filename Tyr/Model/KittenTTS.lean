/-
  Tyr/Model/KittenTTS.lean

  Lean4 KittenTTS port:
  - config
  - single-utterance inference model
 -/
import Tyr.Model.KittenTTS.Config
import Tyr.Model.KittenTTS.ConfigIO
import Tyr.Model.KittenTTS.Checkpoint
import Tyr.Model.KittenTTS.Model
import Tyr.Model.KittenTTS.Weights
import Tyr.Model.KittenTTS.Pretrained
import Tyr.Model.KittenTTS.ReferenceBridge

/-! # KittenTTS

Umbrella import for the KittenTTS text-to-speech model in `torch.kittentts`:
configuration and JSON loading, checkpoint access, the ALBERT-encoder/generator
model, weight loading, pretrained download, and a Python reference bridge for
parity checks.
-/
