/-
  Tyr/Model/Whisper.lean

  Native Whisper ASR model for Tyr.
-/
import Tyr.Model.Whisper.Config
import Tyr.Model.Whisper.ConfigIO
import Tyr.Model.Whisper.Model
import Tyr.Model.Whisper.Weights
import Tyr.Model.Whisper.Transcribe

/-! # Whisper

Umbrella import for Whisper speech recognition in `torch.whisper`: config, JSON
loading, encoder-decoder model, weights and transcription.
-/
