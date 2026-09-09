import Tyr.Model.Qwen3TTS.SpeechTokenizer
import LeanTest

open torch.qwen3tts

namespace Tests.SpeechTokenizerConfig

/-- Exact speech_tokenizer/config.json from Qwen/Qwen3-TTS-12Hz-0.6B-Base
revision 5d83992436eae1d760afd27aff78a71d676296fc, matching fixtures.json.
Git blob SHA-1: 06cc8dc4c5ec8a1929086b71b98c313020d9268b. -/
private def pinnedConfigPath := "Tests/fixtures/qwen3tts/tokenizer-12hz-0.6b-config.json"

private def pinnedJson : IO Lean.Json := do
  let .ok root := Lean.Json.parse (← IO.FS.readFile pinnedConfigPath)
    | throw <| IO.userError "Invalid pinned tokenizer JSON"
  return root

private def subConfig (root : Lean.Json) (name : String) : IO Lean.Json := do
  let .ok value := root.getObjVal? name
    | throw <| IO.userError s!"Missing pinned {name}"
  return value

private def rejected (action : IO α) (expected : String) : IO Unit := do
  let error ← try let _ ← action; pure none catch e => pure (some e.toString)
  match error with
  | some message =>
    LeanTest.assertTrue (message.contains expected) s!"Expected '{expected}', got: {message}"
  | none => throw <| IO.userError s!"Expected config rejection containing '{expected}'"

@[test] def testPinnedSpeechTokenizerEncoderAndDecoderConfig : IO Unit := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile pinnedConfigPath
  LeanTest.assertEqual cfg.encoder.numAttentionHeads 8 "Encoder uses eight attention heads"
  LeanTest.assertEqual cfg.decoder.numAttentionHeads 16 "Decoder uses sixteen attention heads"
  LeanTest.assertEqual cfg.encoder.intermediateSize 2048 "Mimi encoder MLP width"
  LeanTest.assertEqual cfg.decoder.intermediateSize 1024 "Waveform decoder MLP width"
  LeanTest.assertEqual cfg.encoder.slidingWindow 250 "Encoder attention window"
  LeanTest.assertEqual cfg.decoder.slidingWindow 72 "Decoder attention window"
  LeanTest.assertEqual cfg.encoder.numQuantizers 32 "Available encoder codebooks"
  LeanTest.assertEqual cfg.encoderValidNumQuantizers 16 "Qwen emits the first sixteen codebooks"
  LeanTest.assertEqual (← cfg.encoderCodebookCounts) (1, 15) "One semantic plus fifteen acoustic codebooks"
  cfg.validateSupported

@[test] def testSpeechTokenizerRejectsDecoderDimensionsInEncoderMetadata : IO Unit := do
  let root ← pinnedJson
  let encoder ← subConfig root "encoder_config"
  for (field, value) in #[("num_attention_heads", 16), ("num_key_value_heads", 16),
      ("intermediate_size", 1024), ("codebook_dim", 512), ("sliding_window", 72)] do
    let changed := encoder.setObjVal! field (Lean.toJson (value : Nat))
    let cfg := SpeechTokenizer12HzConfig.fromJson (root.setObjVal! "encoder_config" changed)
    rejected cfg.encoderCodebookCounts s!"encoder {field}="
    -- The encoder error must not affect the independent decoder validator.
    cfg.validateSupported

@[test] def testSpeechTokenizerDecoderValidationRemainsIndependent : IO Unit := do
  let root ← pinnedJson
  let decoder ← subConfig root "decoder_config"
  let changed := decoder.setObjVal! "num_attention_heads" (Lean.toJson (8 : Nat))
  let cfg := SpeechTokenizer12HzConfig.fromJson (root.setObjVal! "decoder_config" changed)
  LeanTest.assertEqual (← cfg.encoderCodebookCounts) (1, 15) "Decoder metadata cannot change encoder selection"
  rejected cfg.validateSupported "num_attention_heads=8"

@[test] def testSpeechTokenizerFlexibleEncoderQuantizersUseEncoderMetadata : IO Unit := do
  let root ← pinnedJson
  let encoder ← subConfig root "encoder_config"
  let changed := encoder.setObjVal! "num_semantic_quantizers" (Lean.toJson (2 : Nat))
  let root := (root.setObjVal! "encoder_config" changed).setObjVal!
    "encoder_valid_num_quantizers" (Lean.toJson (8 : Nat))
  let cfg := SpeechTokenizer12HzConfig.fromJson root
  LeanTest.assertEqual cfg.decoder.numQuantizers 16 "Decoder quantizer count stays independent"
  LeanTest.assertEqual (← cfg.encoderCodebookCounts) (2, 6) "Flexible selection uses encoder semantic and valid counts"
  for count in #[0, 1, 2, 33] do
    let cfg := SpeechTokenizer12HzConfig.fromJson
      (root.setObjVal! "encoder_valid_num_quantizers" (Lean.toJson (count : Nat)))
    rejected cfg.encoderCodebookCounts "encoder quantizer split"

@[test] def testSpeechTokenizerEncoderRateMetadataIsValidated : IO Unit := do
  let root ← pinnedJson
  for (field, value) in #[("input_sample_rate", 16000), ("encode_downsample_rate", 960)] do
    let cfg := SpeechTokenizer12HzConfig.fromJson (root.setObjVal! field (Lean.toJson (value : Nat)))
    rejected cfg.encoderCodebookCounts "Unsupported encoder input_sample_rate="
  let encoder ← subConfig root "encoder_config"
  let changed := encoder.setObjVal! "upsampling_ratios" (Lean.toJson (#[2, 2] : Array Nat))
  let cfg := SpeechTokenizer12HzConfig.fromJson (root.setObjVal! "encoder_config" changed)
  rejected cfg.encoderCodebookCounts "encoder upsampling_ratios="

end Tests.SpeechTokenizerConfig
