import Tyr.Text.StreamingConsensus
import Tyr.Text.VADProvider

/-!
# Tyr.Text

`Tyr.Text` is the umbrella import for Tyr's streaming text utilities.

## Major Components

- `Tyr.Text.StreamingConsensus`: streaming transcription consensus state
  (stable/unstable token windows, rollback, and emitted text deltas).
- `Tyr.Text.VADProvider`: Silero-VAD-backed speech/boundary signals used to
  gate streaming text updates.
-/
