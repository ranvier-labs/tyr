/- 
  Tyr/Model/Qwen3ASR/Streaming.lean

  Lean-native streaming helpers mirroring upstream Qwen3-ASR vLLM flow:
  - init_streaming_state
  - streaming_transcribe
  - finish_streaming_transcribe
-/
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.Model
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.Processor
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

private def asrTextTag : String := "<asr_text>"
private def languagePrefix : String := "language "

/-- Strip special/control tokens that may leak into decoded ASR text. -/
private def stripSpecialTokens (s : String) : String :=
  let s := s.replace "<|im_end|>" ""
  let s := s.replace "<|im_start|>" ""
  let s := s.replace "<|endoftext|>" ""
  let s := s.replace "<asr_text>" ""
  let s := s.replace "<|asr_text|>" ""
  s.trimAscii.toString
private def sampleRate16k : Float := 16000.0

inductive StreamingDecodeMode where
  | rollingWindow
  | fullAccumulation
  deriving Repr, Inhabited, BEq

/-- Streaming ASR state (single stream). -/
structure ASRStreamingState where
  unfixedChunkNum : Nat := 2
  unfixedTokenNum : Nat := 5
  promptMaxTokens : Nat := 96
  chunkSizeSec : Float := 2.0
  chunkSizeSamples : Nat := 32000
  stepSizeSec : Float := 0.5
  stepSizeSamples : Nat := 8000

  chunkId : Nat := 0
  buffer : Array Float := #[]
  audioAccum : Array Float := #[]
  decodeMode : StreamingDecodeMode := .rollingWindow

  promptRaw : String := ""
  context : String := ""
  forceLanguage : Option String := none

  language : String := ""
  text : String := ""
  rawDecoded : String := ""
  deriving Repr, Inhabited

/-- Streaming decode callback: `(prompt, accumulated_audio_16k) -> raw decoded text`. -/
abbrev StreamingDecodeFn := String → Array Float → IO String

/-- Per-stream reusable prompt-cache state for model-backed ASR decode. -/
abbrev StreamingPromptCache (cfg : Qwen3ASRConfig) :=
  Qwen3ASRForConditionalGeneration.StreamingPromptCache cfg

/-- Cached prompt tokenization state for append-only streaming prompt updates. -/
structure StreamingPromptTokenCache where
  basePrompt : String
  basePromptLen : Nat
  audioToken : String
  beforeAudioIds : Array UInt32
  afterAudioIds : Array UInt32
  lastSuffix : String := ""
  lastSuffixIds : Array UInt32 := #[]
  deriving Inhabited

/-- Cached projected-audio embedding prefix for full-accumulation streaming mode. -/
structure StreamingAudioEncoderCache (cfg : Qwen3ASRConfig) where
  validFeatureLen : UInt64
  validAudioLen : UInt64
  audioEmbedsValidDyn : T #[]

/-- Cached frontend tensors for append-only full-accumulation streaming audio. -/
structure StreamingFrontendCache (cfg : Qwen3ASRConfig) where
  audioSamples : Nat
  frames : UInt64
  inputFeaturesDyn : T #[]
  featureAttentionMaskDyn : T #[]
  inputFeaturesDevDyn : Option (T #[]) := none
  featureAttentionMaskDevDyn : Option (T #[]) := none

/-- Cached device-side prompt token segments for streaming input-id assembly. -/
structure StreamingPromptDeviceCache where
  beforeLen : UInt64
  beforeIdsDevDyn : T #[]
  afterLen : UInt64
  afterIdsDevDyn : T #[]
  lastSuffix : String := ""
  suffixLen : UInt64 := 0
  suffixIdsDevDyn : Option (T #[]) := none

/-- Full decode cache state carried across streaming decode hops. -/
structure StreamingDecodeCache (cfg : Qwen3ASRConfig) where
  promptCache : Option (StreamingPromptCache cfg) := none
  promptTokenCache : Option StreamingPromptTokenCache := none
  promptDeviceCache : Option StreamingPromptDeviceCache := none
  audioEncoderCache : Option (StreamingAudioEncoderCache cfg) := none
  frontendCache : Option (StreamingFrontendCache cfg) := none

private def toLower (s : String) : String :=
  String.ofList (s.toList.map Char.toLower)

private def startsWithStr (s pref : String) : Bool :=
  let ss := s.toList
  let pp := pref.toList
  pp.length ≤ ss.length && ss.take pp.length == pp

private def dropChars (s : String) (n : Nat) : String :=
  String.ofList (s.toList.drop n)

private def minU64 (a b : UInt64) : UInt64 :=
  if a <= b then a else b

private def appendRepeatedTokenId (ids : Array UInt32) (tokId : UInt32) (n : UInt64) : Array UInt32 :=
  Id.run do
    let mut out := ids
    for _ in [:n.toNat] do
      out := out.push tokId
    out

private def mkStreamingPromptTokenCache
    (tok : tokenizer.qwen3.QwenTokenizer)
    (processor : Qwen3ASRProcessor)
    (basePrompt : String)
    : Option StreamingPromptTokenCache := do
  let parts := (basePrompt.splitOn processor.audioToken).toArray
  if parts.size != 2 then
    none
  else
    let beforeAudioIds := tokenizer.qwen3.encodeText tok parts[0]!
    let afterAudioIds := tokenizer.qwen3.encodeText tok parts[1]!
    some {
      basePrompt := basePrompt
      basePromptLen := basePrompt.toList.length
      audioToken := processor.audioToken
      beforeAudioIds := beforeAudioIds
      afterAudioIds := afterAudioIds
    }

private def encodePromptIdsWithCache
    (tok : tokenizer.qwen3.QwenTokenizer)
    (processor : Qwen3ASRProcessor)
    (basePrompt : String)
    (prompt : String)
    (audioLen : UInt64)
    (audioTokenId : UInt32)
    (cache : Option StreamingPromptTokenCache)
    : IO (Array UInt32 × Option StreamingPromptTokenCache) := do
  let seedCache :=
    match cache with
    | some c => some c
    | none => mkStreamingPromptTokenCache tok processor basePrompt
  match seedCache with
  | some c =>
    if startsWithStr prompt c.basePrompt then
      let suffix := dropChars prompt c.basePromptLen
      let suffixIds :=
        if startsWithStr suffix c.lastSuffix then
          let suffixTail := dropChars suffix c.lastSuffix.toList.length
          c.lastSuffixIds ++ tokenizer.qwen3.encodeText tok suffixTail
        else
          tokenizer.qwen3.encodeText tok suffix
      let mut ids := c.beforeAudioIds
      ids := appendRepeatedTokenId ids audioTokenId audioLen
      ids := ids ++ c.afterAudioIds
      ids := ids ++ suffixIds
      pure (ids, some { c with lastSuffix := suffix, lastSuffixIds := suffixIds })
    else
      match processor.encodeWithExpandedAudioTokenIds tok prompt #[audioLen] audioTokenId with
      | .ok ids => pure (ids, seedCache)
      | .error e => throw <| IO.userError e
  | none =>
    match processor.encodeWithExpandedAudioTokenIds tok prompt #[audioLen] audioTokenId with
    | .ok ids => pure (ids, none)
    | .error e => throw <| IO.userError e

private def idsToDyn1dOnDevice (ids : Array UInt32) (dev : Device) : IO (UInt64 × T #[]) := do
  let n : UInt64 := ids.size.toUInt64
  let vals : Array Int64 := ids.map (fun id => Int64.ofNat id.toNat)
  let tCpu : T #[n] := reshape (data.fromInt64Array vals) #[n]
  let tDev : T #[n] := tCpu.to dev
  pure (n, nn.eraseShape tDev)

private def repeatedTokenDynOnDevice (tokId : UInt32) (n : UInt64) (dev : Device) : T #[] :=
  if n == 0 then
    nn.eraseShape (reshape (data.fromInt64Array #[]) #[0])
  else
    let tCpu : T #[n] := torch.full_int #[n] (Int64.ofNat tokId.toNat)
    nn.eraseShape (tCpu.to dev)

private def mkPromptDeviceCache
    (tokenCache : StreamingPromptTokenCache)
    (dev : Device)
    : IO StreamingPromptDeviceCache := do
  let (beforeLen, beforeDyn) ← idsToDyn1dOnDevice tokenCache.beforeAudioIds dev
  let (afterLen, afterDyn) ← idsToDyn1dOnDevice tokenCache.afterAudioIds dev
  let (suffixLen, suffixDynOpt) ←
    if tokenCache.lastSuffixIds.isEmpty then
      pure (0, none)
    else
      let (n, dyn) ← idsToDyn1dOnDevice tokenCache.lastSuffixIds dev
      pure (n, some dyn)
  pure {
    beforeLen := beforeLen
    beforeIdsDevDyn := beforeDyn
    afterLen := afterLen
    afterIdsDevDyn := afterDyn
    lastSuffix := tokenCache.lastSuffix
    suffixLen := suffixLen
    suffixIdsDevDyn := suffixDynOpt
  }

private def updatePromptDeviceCacheSuffix
    (cache : StreamingPromptDeviceCache)
    (tokenCache : StreamingPromptTokenCache)
    (dev : Device)
    : IO StreamingPromptDeviceCache := do
  if cache.lastSuffix == tokenCache.lastSuffix then
    pure cache
  else if tokenCache.lastSuffixIds.isEmpty then
    pure { cache with lastSuffix := tokenCache.lastSuffix, suffixLen := 0, suffixIdsDevDyn := none }
  else
    let (n, dyn) ← idsToDyn1dOnDevice tokenCache.lastSuffixIds dev
    pure {
      cache with
        lastSuffix := tokenCache.lastSuffix
        suffixLen := n
        suffixIdsDevDyn := some dyn
    }

private def buildInputIdsWithPromptDeviceCache
    (promptIds : Array UInt32)
    (promptTokenCache : Option StreamingPromptTokenCache)
    (promptDeviceCache : Option StreamingPromptDeviceCache)
    (audioLen : UInt64)
    (audioTokenId : UInt32)
    (dev : Device)
    : IO ((Sigma (fun seq => T #[1, seq])) × Option StreamingPromptDeviceCache) := do
  match promptTokenCache with
  | some ptc =>
    let devCache ←
      match promptDeviceCache with
      | some dc =>
        if dc.beforeLen == ptc.beforeAudioIds.size.toUInt64 &&
           dc.afterLen == ptc.afterAudioIds.size.toUInt64 then
          updatePromptDeviceCacheSuffix dc ptc dev
        else
          mkPromptDeviceCache ptc dev
      | none =>
        mkPromptDeviceCache ptc dev
    let audioDyn := repeatedTokenDynOnDevice audioTokenId audioLen dev
    let mut parts : Array (UInt64 × T #[]) := #[]
    if devCache.beforeLen > 0 then
      parts := parts.push (devCache.beforeLen, devCache.beforeIdsDevDyn)
    if audioLen > 0 then
      parts := parts.push (audioLen, audioDyn)
    if devCache.afterLen > 0 then
      parts := parts.push (devCache.afterLen, devCache.afterIdsDevDyn)
    if devCache.suffixLen > 0 then
      match devCache.suffixIdsDevDyn with
      | some dyn => parts := parts.push (devCache.suffixLen, dyn)
      | none => pure ()
    let seqFromParts := parts.foldl (fun acc p => acc + p.1) 0
    let seqFromPrompt := promptIds.size.toUInt64
    if seqFromParts != seqFromPrompt then
      throw <| IO.userError
        s!"prompt device cache length mismatch: from_parts={seqFromParts}, from_prompt={seqFromPrompt}"
    if seqFromPrompt == 0 then
      throw <| IO.userError "decodeStreamingChunkWithModel produced empty prompt/input sequence"
    let dynParts := parts.map (fun p => p.2)
    let ids1dDyn : T #[] :=
      match dynParts[0]? with
      | some _ => nn.cat_dyn dynParts 0
      | none => nn.eraseShape (reshape (data.fromInt64Array #[]) #[0])
    let ids1d : T #[seqFromPrompt] := reshape ids1dDyn #[seqFromPrompt]
    let inputIds : T #[1, seqFromPrompt] := reshape ids1d #[1, seqFromPrompt]
    pure (⟨seqFromPrompt, inputIds⟩, some devCache)
  | none =>
    let idsVals : Array Int64 := promptIds.map (fun id => Int64.ofNat id.toNat)
    let seq : UInt64 := idsVals.size.toUInt64
    if seq == 0 then
      throw <| IO.userError "decodeStreamingChunkWithModel produced empty prompt/input sequence"
    let inputIdsCpu : T #[1, seq] := reshape (data.fromInt64Array idsVals) #[1, seq]
    let inputIds : T #[1, seq] := inputIdsCpu.to dev
    pure (⟨seq, inputIds⟩, none)

private def findSingleContiguousAudioSpanStart
    (promptIds : Array UInt32)
    (audioTokenId : UInt32)
    (audioLen : UInt64)
    : Option UInt64 :=
  if audioLen == 0 then
    if promptIds.contains audioTokenId then none else some 0
  else
    let targetLen := audioLen.toNat
    Id.run do
      let mut spans : Array (Nat × Nat) := #[]
      let mut i : Nat := 0
      while i < promptIds.size do
        if promptIds.getD i 0 == audioTokenId then
          let start := i
          let mut j := i
          while j < promptIds.size && promptIds.getD j 0 == audioTokenId do
            j := j + 1
          spans := spans.push (start, j - start)
          i := j
        else
          i := i + 1
      match spans[0]? with
      | some (start, spanLen) =>
        if spans.size == 1 && spanLen == targetLen then
          some start.toUInt64
        else
          none
      | none => none

private def ceilDivNat (num den : Nat) : Nat :=
  if den == 0 then 0 else (num + den - 1) / den

private def buildStreamingFrontendPackWithCache
    {cfg : Qwen3ASRConfig}
    (preprocessor : PreprocessorConfig)
    (audio16k : Array Float)
    (decodeMode : StreamingDecodeMode)
    (cache : Option (StreamingFrontendCache cfg))
    (frontendDevice : Device := Device.CPU)
    : IO ((Sigma (fun frames => WhisperFrontendOutput preprocessor.featureSize frames)) ×
      Option (StreamingFrontendCache cfg) × Option UInt64) := do
  let maxSec :=
    (PreprocessorConfig.expectedSampleCount preprocessor).toFloat / sampleRate16k
  let maxSamples := PreprocessorConfig.expectedSampleCount preprocessor
  if audio16k.size.toUInt64 > maxSamples then
    throw <| IO.userError
      s!"Audio duration exceeds maxSeconds={maxSec}: samples={audio16k.size.toUInt64}, limit={maxSamples}"

  let fullFallback : IO ((Sigma (fun frames => WhisperFrontendOutput preprocessor.featureSize frames)) ×
      Option (StreamingFrontendCache cfg) × Option UInt64) := do
    let fullPack ←
      waveformToWhisperFeaturesDynamic
        preprocessor
        audio16k
        (minSeconds := 0.05)
        (maxSeconds := some maxSec)
        (device := frontendDevice)
    let fullCache : Option (StreamingFrontendCache cfg) :=
      if decodeMode == .fullAccumulation then
        some {
          audioSamples := audio16k.size
          frames := fullPack.1
          inputFeaturesDyn := nn.eraseShape fullPack.2.inputFeatures
          featureAttentionMaskDyn := nn.eraseShape fullPack.2.featureAttentionMask
        }
      else
        none
    pure (fullPack, fullCache, none)

  let canTryIncremental :=
    decodeMode == .fullAccumulation &&
    preprocessor.doNormalize == false &&
    preprocessor.dither == 0.0 &&
    (if preprocessor.hopLength == 0 then (160 : UInt64) else preprocessor.hopLength) > 0 &&
    audio16k.size > 0

  if !canTryIncremental then
    fullFallback
  else
    match cache with
    | none => fullFallback
    | some fc =>
      if audio16k.size <= fc.audioSamples then
        fullFallback
      else
        let hopNat := (if preprocessor.hopLength == 0 then (160 : UInt64) else preprocessor.hopLength).toNat
        let nFftNat := (if preprocessor.nFft == 0 then (400 : UInt64) else preprocessor.nFft).toNat
        let halfWindow := nFftNat / 2
        let contextFrames := ceilDivNat halfWindow hopNat + 1
        let prevFramesNat := fc.frames.toNat
        let replaceStartFrameNat :=
          if prevFramesNat > contextFrames then
            prevFramesNat - contextFrames
          else
            0
        let startFrameNat :=
          if replaceStartFrameNat > contextFrames then
            replaceStartFrameNat - contextFrames
          else
            0
        let dropFramesNat := replaceStartFrameNat - startFrameNat
        let startSample := startFrameNat * hopNat
        if startSample >= audio16k.size then
          fullFallback
        else
          let suffixAudio := audio16k.extract startSample audio16k.size
          let suffixPack ←
            waveformToWhisperFeaturesDynamic
              preprocessor
              suffixAudio
              (minSeconds := 0.0)
              (maxSeconds := none)
              (device := frontendDevice)
          let suffixFrames := suffixPack.1
          let dropFrames := dropFramesNat.toUInt64
          if dropFrames > suffixFrames then
            fullFallback
          else
            let prevFeatures : T #[1, preprocessor.featureSize, fc.frames] :=
              reshape fc.inputFeaturesDyn #[1, preprocessor.featureSize, fc.frames]
            let prevMask : T #[1, fc.frames] :=
              reshape fc.featureAttentionMaskDyn #[1, fc.frames]
            let prefixFrames := replaceStartFrameNat.toUInt64
            let prefixFeatures : T #[1, preprocessor.featureSize, prefixFrames] :=
              data.slice prevFeatures 2 0 prefixFrames
            let prefixMask : T #[1, prefixFrames] :=
              data.slice prevMask 1 0 prefixFrames
            let suffixKeepFrames := suffixFrames - dropFrames
            let suffixFeatures : T #[1, preprocessor.featureSize, suffixKeepFrames] :=
              data.slice suffixPack.2.inputFeatures 2 dropFrames suffixKeepFrames
            let suffixMask : T #[1, suffixKeepFrames] :=
              data.slice suffixPack.2.featureAttentionMask 1 dropFrames suffixKeepFrames
            let mergedFrames := prefixFrames + suffixKeepFrames
            let mergedFeaturesDyn : T #[] :=
              nn.cat_dyn #[nn.eraseShape prefixFeatures, nn.eraseShape suffixFeatures] 2
            let mergedMaskDyn : T #[] :=
              nn.cat_dyn #[nn.eraseShape prefixMask, nn.eraseShape suffixMask] 1
            let mergedOut : WhisperFrontendOutput preprocessor.featureSize mergedFrames := {
              inputFeatures := reshape mergedFeaturesDyn #[1, preprocessor.featureSize, mergedFrames]
              featureAttentionMask := reshape mergedMaskDyn #[1, mergedFrames]
            }
            pure (⟨mergedFrames, mergedOut⟩, some {
              audioSamples := audio16k.size
              frames := mergedFrames
              inputFeaturesDyn := nn.eraseShape mergedOut.inputFeatures
              featureAttentionMaskDyn := nn.eraseShape mergedOut.featureAttentionMask
            }, some prefixFrames)

private def joinWith (sep : String) (parts : List String) : String :=
  match parts with
  | [] => ""
  | p :: ps => ps.foldl (fun acc x => acc ++ sep ++ x) p

private def containsSubstr (s pat : String) : Bool :=
  if pat.isEmpty then
    false
  else
    (s.splitOn pat).length > 1

private def splitFirst (s sep : String) : Option (String × String) :=
  match s.splitOn sep with
  | [] => none
  | [_] => none
  | x :: xs => some (x, joinWith sep xs)

/-- Normalize language as `Xxxxx...` (first upper, rest lower). -/
def normalizeLanguageName (language : String) : Except String String := do
  let s := language.trimAscii.toString
  if s.isEmpty then
    throw "language is empty"
  match s.toList with
  | [] => throw "language is empty"
  | c :: cs =>
    pure <| String.ofList (Char.toUpper c :: cs.map Char.toLower)

/-- Validate language against configured support list. -/
def validateLanguage (supported : Array String) (language : String) : Except String Unit := do
  if supported.contains language then
    pure ()
  else
    throw s!"Unsupported language: {language}. Supported: {supported}"

private def segmentEq (chars : Array Char) (a b len : Nat) : Bool := Id.run do
  if a + len > chars.size || b + len > chars.size then
    return false
  let mut ok := true
  for j in [:len] do
    if ok && chars[a + j]! != chars[b + j]! then
      ok := false
  return ok

private def segmentToString (chars : Array Char) (start len : Nat) : String :=
  String.ofList ((chars.extract start (start + len)).toList)

private def fixCharRepeats (s : String) (threshold : Nat) : String :=
  if threshold == 0 then
    s
  else
    Id.run do
      let chars := s.toList.toArray
      let mut out : Array Char := #[]
      let mut i : Nat := 0
      while i < chars.size do
        let mut count : Nat := 1
        while i + count < chars.size && chars[i + count]! == chars[i]! do
          count := count + 1
        if count > threshold then
          out := out.push chars[i]!
        else
          for j in [:count] do
            out := out.push chars[i + j]!
        i := i + count
      String.ofList out.toList

private partial def fixPatternRepeats (s : String) (threshold : Nat) (maxLen : Nat := 20) : String :=
  let chars := s.toList.toArray
  let n := chars.size
  let minRepeatChars := threshold * 2
  if threshold == 0 || n < minRepeatChars then
    s
  else
    Id.run do
      let mut i : Nat := 0
      let mut result := ""
      let mut found := false
      while i + minRepeatChars <= n && !found do
        let mut k : Nat := 1
        let mut matched := false
        while k <= maxLen && !matched do
          if i + k * threshold > n then
            k := maxLen + 1
          else
            let mut rep : Nat := 1
            let mut valid := true
            while rep < threshold && valid do
              let startIdx := i + rep * k
              if !(segmentEq chars i startIdx k) then
                valid := false
              rep := rep + 1
            if valid then
              let mut endIndex := i + threshold * k
              while endIndex + k <= n && segmentEq chars i endIndex k do
                endIndex := endIndex + k
              result := result ++ segmentToString chars i k
              let tail := segmentToString chars endIndex (n - endIndex)
              result := result ++ fixPatternRepeats tail threshold maxLen
              i := n
              found := true
              matched := true
            else
              k := k + 1
        if !found then
          result := result.push chars[i]!
          i := i + 1
      if !found then
        result := result ++ segmentToString chars i (n - i)
      result

/-- Port of upstream repetition cleanup used before parsing metadata. -/
def detectAndFixRepetitions (text : String) (threshold : Nat := 20) : String :=
  fixPatternRepeats (fixCharRepeats text threshold) threshold

/-- Parse raw ASR output into `(language, text)` with upstream semantics. -/
def parseAsrOutput (raw : String) (userLanguage : Option String := none) : String × String :=
  let s0 := raw.trimAscii.toString
  if s0.isEmpty then
    match userLanguage with
    | some forced =>
      let f := forced.trimAscii.toString
      (f, "")
    | none => ("", "")
  else
    let s := detectAndFixRepetitions s0
    match userLanguage with
    | some forced =>
      let f := forced.trimAscii.toString
      if f.isEmpty then
        ("", stripSpecialTokens s)
      else
        (f, stripSpecialTokens s)
    | none =>
      match splitFirst s asrTextTag with
      | none => ("", stripSpecialTokens s)
      | some (metaPart, textPart) =>
        let metaLower := toLower metaPart
        if containsSubstr metaLower "language none" then
          let t := stripSpecialTokens textPart
          if t.isEmpty then
            ("", "")
          else
            ("", t)
        else
          Id.run do
            let mut lang := ""
            for line in metaPart.splitOn "\n" do
              if lang.isEmpty then
                let lineTrim := line.trimAscii.toString
                if !lineTrim.isEmpty then
                  let low := toLower lineTrim
                  if startsWithStr low languagePrefix then
                    let val := (dropChars lineTrim languagePrefix.length).trimAscii.toString
                    if !val.isEmpty then
                      match normalizeLanguageName val with
                      | .ok ln => lang := ln
                      | .error _ => lang := val
            (lang, stripSpecialTokens textPart)

private def joinArrayWith (sep : String) (xs : Array String) : String := Id.run do
  if xs.isEmpty then
    return ""
  let mut out := xs[0]!
  for i in [1:xs.size] do
    out := out ++ sep ++ xs[i]!
  out

/-- Merge per-chunk languages by removing empties and consecutive duplicates. -/
def mergeLanguages (langs : Array String) : String :=
  let merged : Array String := Id.run do
    let mut out : Array String := #[]
    let mut prev : Option String := none
    for l in langs do
      let x := l.trimAscii.toString
      if !x.isEmpty then
        match prev with
        | some p =>
          if x != p then
            out := out.push x
            prev := some x
        | none =>
          out := out.push x
          prev := some x
    out
  joinArrayWith "," merged

/-- Build base text prompt for streaming decode. -/
def buildTextPrompt
    (context : String)
    (forceLanguage : Option String := none)
    (processor : Qwen3ASRProcessor := {})
    : String :=
  processor.buildAsrPrompt context forceLanguage

private def chunkSizeSamplesFromSec (chunkSizeSec : Float) : Nat :=
  let rounded := ((chunkSizeSec * sampleRate16k) + 0.5).toUInt64.toNat
  if rounded == 0 then 1 else rounded

private def tailSlice (xs : Array Float) (n : Nat) : Array Float :=
  if xs.size <= n then xs else xs.extract (xs.size - n) xs.size

/-- Initialize streaming state (single stream, 16k PCM). -/
def initStreamingState
    (supportedLanguages : Array String)
    (context : String := "")
    (language : Option String := none)
    (unfixedChunkNum : Nat := 2)
    (unfixedTokenNum : Nat := 5)
    (promptMaxTokens : Nat := 96)
    (chunkSizeSec : Float := 2.0)
    (stepSizeSec : Float := 0.5)
    (decodeMode : StreamingDecodeMode := .rollingWindow)
    : IO ASRStreamingState := do
  if chunkSizeSec <= 0.0 || stepSizeSec <= 0.0 then
    throw <| IO.userError
      s!"chunk_size_sec and step_size_sec must be > 0, got: {chunkSizeSec}, {stepSizeSec}"

  let forceLanguage ←
    match language with
    | none => pure none
    | some l =>
      let s := l.trimAscii.toString
      if s.isEmpty then
        pure none
      else
        match normalizeLanguageName s with
        | .error e => throw <| IO.userError e
        | .ok ln =>
          match validateLanguage supportedLanguages ln with
          | .error e => throw <| IO.userError e
          | .ok _ => pure (some ln)

  let chunkSizeSamples := chunkSizeSamplesFromSec chunkSizeSec
  let stepSizeSamples := chunkSizeSamplesFromSec stepSizeSec
  let promptRaw := buildTextPrompt context forceLanguage {}
  pure {
    unfixedChunkNum
    unfixedTokenNum
    promptMaxTokens
    chunkSizeSec
    chunkSizeSamples
    stepSizeSec
    stepSizeSamples
    chunkId := 0
    buffer := #[]
    audioAccum := #[]
    decodeMode := decodeMode
    promptRaw
    context := context
    forceLanguage
    language := ""
    text := ""
    rawDecoded := ""
  }

private def containsReplacementChar (s : String) : Bool :=
  s.toList.any (fun c => c == Char.ofNat 0xFFFD)

private def rollbackPrefixForChunk
    (tok : tokenizer.qwen3.QwenTokenizer)
    (rawDecoded : String)
    (unfixedTokenNum : Nat)
    (promptMaxTokens : Nat)
    : String := Id.run do
  let curIds := tokenizer.qwen3.encodeText tok rawDecoded
  let mut k := unfixedTokenNum
  let mut pref := ""
  let mut done := false
  while !done do
    let endIdx := if curIds.size > k then curIds.size - k else 0
    let startIdx :=
      if promptMaxTokens == 0 then
        0
      else if endIdx > promptMaxTokens then
        endIdx - promptMaxTokens
      else
        0
    pref :=
      if endIdx > startIdx then
        tokenizer.qwen3.decodeText tok (curIds.extract startIdx endIdx)
      else
        ""
    if !containsReplacementChar pref then
      done := true
    else if endIdx == 0 then
      pref := ""
      done := true
    else
      k := k + 1
  pref

private def rollbackPrefixForFinish
    (tok : tokenizer.qwen3.QwenTokenizer)
    (rawDecoded : String)
    (unfixedTokenNum : Nat)
    (promptMaxTokens : Nat)
    : String :=
  let curIds := tokenizer.qwen3.encodeText tok rawDecoded
  let endIdxRaw := Nat.max 1 (curIds.size - unfixedTokenNum)
  let endIdx := Nat.min endIdxRaw curIds.size
  let startIdx :=
    if promptMaxTokens == 0 then
      0
    else if endIdx > promptMaxTokens then
      endIdx - promptMaxTokens
    else
      0
  tokenizer.qwen3.decodeText tok (curIds.extract startIdx endIdx)

/-- Streaming decode step:
    append new PCM samples, consume fixed `step` chunks, and decode over a
    rolling `chunk` audio window. -/
def streamingTranscribe
    (tok : tokenizer.qwen3.QwenTokenizer)
    (decodeFn : StreamingDecodeFn)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    : IO ASRStreamingState := do
  if state.chunkSizeSamples == 0 || state.stepSizeSamples == 0 then
    throw <| IO.userError "streaming state has invalid chunk_size_samples=0"

  let mut st := { state with buffer := state.buffer ++ pcm16k }

  while st.buffer.size >= st.stepSizeSamples do
    let stepChunk := st.buffer.extract 0 st.stepSizeSamples
    let rest := st.buffer.extract st.stepSizeSamples st.buffer.size
    let audioAccum :=
      match st.decodeMode with
      | .rollingWindow => tailSlice (st.audioAccum ++ stepChunk) st.chunkSizeSamples
      | .fullAccumulation => st.audioAccum ++ stepChunk
    st := { st with buffer := rest, audioAccum := audioAccum }

    let pref :=
      if st.chunkId < st.unfixedChunkNum then
        ""
      else
        rollbackPrefixForChunk tok st.rawDecoded st.unfixedTokenNum st.promptMaxTokens
    let prompt := st.promptRaw ++ pref
    let genText ← decodeFn prompt st.audioAccum
    let rawDecoded := if pref.isEmpty then genText else pref ++ genText
    let (lang, txt) := parseAsrOutput rawDecoded st.forceLanguage
    st := { st with
      rawDecoded := rawDecoded
      language := lang
      text := txt
      chunkId := st.chunkId + 1
    }

  pure st

/-- Flush tail audio shorter than one chunk and decode once more. -/
def finishStreamingTranscribe
    (tok : tokenizer.qwen3.QwenTokenizer)
    (decodeFn : StreamingDecodeFn)
    (state : ASRStreamingState)
    : IO ASRStreamingState := do
  if state.buffer.isEmpty then
    pure state
  else
    let tail := state.buffer
    let audioAccum :=
      match state.decodeMode with
      | .rollingWindow => tailSlice (state.audioAccum ++ tail) state.chunkSizeSamples
      | .fullAccumulation => state.audioAccum ++ tail
    let mut st := { state with buffer := #[], audioAccum := audioAccum }
    let pref :=
      if st.chunkId < st.unfixedChunkNum then
        ""
      else
        rollbackPrefixForFinish tok st.rawDecoded st.unfixedTokenNum st.promptMaxTokens
    let prompt := st.promptRaw ++ pref
    let genText ← decodeFn prompt st.audioAccum
    let rawDecoded := if pref.isEmpty then genText else pref ++ genText
    let (lang, txt) := parseAsrOutput rawDecoded st.forceLanguage
    st := { st with
      rawDecoded := rawDecoded
      language := lang
      text := txt
      chunkId := st.chunkId + 1
    }
    pure st

/-- Lean decode helper that runs one model generation from `(prompt, audio)` pair
    and returns updated full decode-cache state for the next hop. -/
def decodeStreamingChunkWithModelStateCached
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (basePrompt : String)
    (decodeMode : StreamingDecodeMode)
    (prompt : String)
    (audio16k : Array Float)
    (cache : Option (StreamingDecodeCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (String × Option (StreamingDecodeCache cfg)) := do
  if audio16k.isEmpty then
    pure ("", cache)
  else
    autograd.no_grad do
      let cache0 : StreamingDecodeCache cfg := cache.getD {}
      let dev := model.thinker.textModel.embed_tokens.device
      let (frontendPack, frontendCacheCpu', reusedPrefixFrames) ←
        buildStreamingFrontendPackWithCache
          (cfg := cfg)
          preprocessor
          audio16k
          decodeMode
          cache0.frontendCache
          (frontendDevice := dev)
      let frames := frontendPack.1
      let frontendOut := frontendPack.2
      let validFramesTensor : T #[1] := nn.sumDim (data.toLong frontendOut.featureAttentionMask) 1 false
      let validFrames := (nn.item validFramesTensor).toUInt64
      let audioLenRaw := AudioEncoderConfig.featExtractOutputLength validFrames
      let audioLenCap :=
        AudioEncoderConfig.framesAfterConv3
          cfg.thinkerConfig.audioConfig
          frames
      let audioLen := minU64 audioLenRaw audioLenCap

      let processor : Qwen3ASRProcessor := {}
      let (promptIds, promptTokenCache') ←
        encodePromptIdsWithCache
          tok
          processor
          basePrompt
          prompt
          audioLen
          cfg.thinkerConfig.audioTokenId.toUInt32
          cache0.promptTokenCache

      let (inputIdsPack, promptDeviceCache') ←
        buildInputIdsWithPromptDeviceCache
          promptIds
          promptTokenCache'
          cache0.promptDeviceCache
          audioLen
          cfg.thinkerConfig.audioTokenId.toUInt32
          dev
      let seq := inputIdsPack.1
      let inputIds := inputIdsPack.2
      let mkFrontendCacheWithDev
          (cpuCache : Option (StreamingFrontendCache cfg))
          (featuresDev : T #[1, preprocessor.featureSize, frames])
          (maskDev : T #[1, frames])
          : Option (StreamingFrontendCache cfg) :=
        cpuCache.map (fun fc => {
          fc with
            inputFeaturesDevDyn := some (nn.eraseShape featuresDev)
            featureAttentionMaskDevDyn := some (nn.eraseShape maskDev)
        })
      let mkFrontendFullDev
          : IO (T #[1, preprocessor.featureSize, frames] × T #[1, frames] × Option (StreamingFrontendCache cfg)) := do
        let fDev : T #[1, preprocessor.featureSize, frames] := frontendOut.inputFeatures.to dev
        let mDev : T #[1, frames] := frontendOut.featureAttentionMask.to dev
        pure (fDev, mDev, mkFrontendCacheWithDev frontendCacheCpu' fDev mDev)
      let (inputFeaturesDev, _featureAttentionMaskDev, frontendCache') ←
        if decodeMode == .fullAccumulation then
          match reusedPrefixFrames, cache0.frontendCache, frontendCacheCpu' with
          | some prefixFrames, some prevCache, some _ =>
            match prevCache.inputFeaturesDevDyn, prevCache.featureAttentionMaskDevDyn with
            | some prevFeatDyn, some prevMaskDyn =>
              if prefixFrames <= prevCache.frames && prefixFrames <= frames then
                let prevFeatDev : T #[1, preprocessor.featureSize, prevCache.frames] :=
                  reshape prevFeatDyn #[1, preprocessor.featureSize, prevCache.frames]
                let prevMaskDev : T #[1, prevCache.frames] :=
                  reshape prevMaskDyn #[1, prevCache.frames]
                let prefixFeatDev : T #[1, preprocessor.featureSize, prefixFrames] :=
                  data.slice prevFeatDev 2 0 prefixFrames
                let prefixMaskDev : T #[1, prefixFrames] :=
                  data.slice prevMaskDev 1 0 prefixFrames
                let suffixFrames := frames - prefixFrames
                let suffixFeatCpu : T #[1, preprocessor.featureSize, suffixFrames] :=
                  data.slice frontendOut.inputFeatures 2 prefixFrames suffixFrames
                let suffixMaskCpu : T #[1, suffixFrames] :=
                  data.slice frontendOut.featureAttentionMask 1 prefixFrames suffixFrames
                let suffixFeatDev : T #[1, preprocessor.featureSize, suffixFrames] := suffixFeatCpu.to dev
                let suffixMaskDev : T #[1, suffixFrames] := suffixMaskCpu.to dev
                let mergedFeatDyn : T #[] :=
                  nn.cat_dyn #[nn.eraseShape prefixFeatDev, nn.eraseShape suffixFeatDev] 2
                let mergedMaskDyn : T #[] :=
                  nn.cat_dyn #[nn.eraseShape prefixMaskDev, nn.eraseShape suffixMaskDev] 1
                let mergedFeatDev : T #[1, preprocessor.featureSize, frames] :=
                  reshape mergedFeatDyn #[1, preprocessor.featureSize, frames]
                let mergedMaskDev : T #[1, frames] :=
                  reshape mergedMaskDyn #[1, frames]
                pure (mergedFeatDev, mergedMaskDev, mkFrontendCacheWithDev frontendCacheCpu' mergedFeatDev mergedMaskDev)
              else
                mkFrontendFullDev
            | _, _ =>
              mkFrontendFullDev
          | _, _, _ =>
            mkFrontendFullDev
        else
          mkFrontendFullDev
      let fullStepFeature : UInt64 := cfg.thinkerConfig.audioConfig.nWindow * 2
      let (audioEmbedsValid, audioCache') ←
        match cache0.audioEncoderCache with
        | some ac =>
          let canReuse :=
            decodeMode == .fullAccumulation &&
            fullStepFeature > 0 &&
            ac.validFeatureLen <= validFrames
          if !canReuse then
            let audioRaw : T #[1, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames, cfg.thinkerConfig.audioConfig.outputDim] :=
              model.thinker.encodeAudioVarLen inputFeaturesDev #[validFrames]
            let audioProj : T #[1, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames, cfg.thinkerConfig.textConfig.hiddenSize] :=
              model.thinker.projectAudio audioRaw
            let audioValid : T #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize] := data.slice audioProj 1 0 audioLen
            let nextAudioCache :=
              if decodeMode == .fullAccumulation then
                some {
                  validFeatureLen := validFrames
                  validAudioLen := audioLen
                  audioEmbedsValidDyn := nn.eraseShape audioValid
                }
              else
                none
            pure (audioValid, nextAudioCache)
          else
            let prevFull := (ac.validFeatureLen / fullStepFeature) * fullStepFeature
            let curFull := (validFrames / fullStepFeature) * fullStepFeature
            let reuseBase := minU64 prevFull curFull
            let reuseFeatureLen :=
              if reuseBase > 0 && validFrames % fullStepFeature != 0 then
                reuseBase - fullStepFeature
              else
                reuseBase
            let reuseAudioLenRaw := AudioEncoderConfig.featExtractOutputLength reuseFeatureLen
            let reuseAudioLen := minU64 (minU64 reuseAudioLenRaw ac.validAudioLen) audioLen
            let prevAudio : T #[1, ac.validAudioLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
              reshape ac.audioEmbedsValidDyn #[1, ac.validAudioLen, cfg.thinkerConfig.textConfig.hiddenSize]
            let prefixAudio : T #[1, reuseAudioLen, cfg.thinkerConfig.textConfig.hiddenSize] := data.slice prevAudio 1 0 reuseAudioLen
            let tailFeatureLen := validFrames - reuseFeatureLen
            if tailFeatureLen == 0 then
              let audioValid : T #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                if reuseAudioLen == audioLen then
                  reshape prefixAudio #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize]
                else if reuseAudioLen > audioLen then
                  data.slice prefixAudio 1 0 audioLen
                else
                  let padLen := audioLen - reuseAudioLen
                  let pad : T #[1, padLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                    torch.zeros #[1, padLen, cfg.thinkerConfig.textConfig.hiddenSize] false prefixAudio.device
                  reshape (nn.cat prefixAudio pad 1) #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize]
              let nextAudioCache :=
                if decodeMode == .fullAccumulation then
                  some {
                    validFeatureLen := validFrames
                    validAudioLen := audioLen
                    audioEmbedsValidDyn := nn.eraseShape audioValid
                  }
                else
                  none
              pure (audioValid, nextAudioCache)
            else
              let tailFrames := frames - reuseFeatureLen
              let tailInput : T #[1, preprocessor.featureSize, tailFrames] :=
                data.slice inputFeaturesDev 2 reuseFeatureLen tailFrames
              let tailAudioRaw : T #[1, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig tailFrames, cfg.thinkerConfig.audioConfig.outputDim] :=
                model.thinker.encodeAudioVarLen tailInput #[tailFeatureLen]
              let tailAudioProj : T #[1, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig tailFrames, cfg.thinkerConfig.textConfig.hiddenSize] :=
                model.thinker.projectAudio tailAudioRaw
              let tailAudioLenRaw := AudioEncoderConfig.featExtractOutputLength tailFeatureLen
              let tailAudioCap := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig tailFrames
              let tailAudioLen := minU64 tailAudioLenRaw tailAudioCap
              let tailValid : T #[1, tailAudioLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                data.slice tailAudioProj 1 0 tailAudioLen
              let combinedDyn : T #[] := nn.cat_dyn #[nn.eraseShape prefixAudio, nn.eraseShape tailValid] 1
              let combinedLen := reuseAudioLen + tailAudioLen
              let audioValid : T #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                if combinedLen == audioLen then
                  reshape combinedDyn #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize]
                else if combinedLen > audioLen then
                  let combined : T #[1, combinedLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                    reshape combinedDyn #[1, combinedLen, cfg.thinkerConfig.textConfig.hiddenSize]
                  data.slice combined 1 0 audioLen
                else
                  let combined : T #[1, combinedLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                    reshape combinedDyn #[1, combinedLen, cfg.thinkerConfig.textConfig.hiddenSize]
                  let padLen := audioLen - combinedLen
                  let pad : T #[1, padLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                    torch.zeros #[1, padLen, cfg.thinkerConfig.textConfig.hiddenSize] false combined.device
                  reshape (nn.cat combined pad 1) #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize]
              let nextAudioCache :=
                if decodeMode == .fullAccumulation then
                  some {
                    validFeatureLen := validFrames
                    validAudioLen := audioLen
                    audioEmbedsValidDyn := nn.eraseShape audioValid
                  }
                else
                  none
              pure (audioValid, nextAudioCache)
        | none =>
          let audioRaw : T #[1, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames, cfg.thinkerConfig.audioConfig.outputDim] :=
            model.thinker.encodeAudioVarLen inputFeaturesDev #[validFrames]
          let audioProj : T #[1, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames, cfg.thinkerConfig.textConfig.hiddenSize] :=
            model.thinker.projectAudio audioRaw
          let audioValid : T #[1, audioLen, cfg.thinkerConfig.textConfig.hiddenSize] := data.slice audioProj 1 0 audioLen
          let nextAudioCache :=
            if decodeMode == .fullAccumulation then
              some {
                validFeatureLen := validFrames
                validAudioLen := audioLen
                audioEmbedsValidDyn := nn.eraseShape audioValid
              }
            else
              none
          pure (audioValid, nextAudioCache)

      let inputsEmbeds0 : T #[1, seq, cfg.thinkerConfig.textConfig.hiddenSize] := model.thinker.embedText inputIds
      let inputsEmbeds : T #[1, seq, cfg.thinkerConfig.textConfig.hiddenSize] ←
        if audioLen == 0 then
          pure inputsEmbeds0
        else
          match findSingleContiguousAudioSpanStart promptIds cfg.thinkerConfig.audioTokenId.toUInt32 audioLen with
          | some spanStart =>
            if spanStart + audioLen > seq then
              throw <| IO.userError
                s!"decodeStreamingChunkWithModel placeholder span overflow: start={spanStart}, audio_len={audioLen}, seq={seq}"
            else do
              let prefixEmbeds : T #[1, spanStart, cfg.thinkerConfig.textConfig.hiddenSize] :=
                data.slice inputsEmbeds0 1 0 spanStart
              let suffixStart := spanStart + audioLen
              let suffixLen := seq - suffixStart
              let suffixEmbeds : T #[1, suffixLen, cfg.thinkerConfig.textConfig.hiddenSize] :=
                data.slice inputsEmbeds0 1 suffixStart suffixLen
              let mergedDyn : T #[] :=
                nn.cat_dyn
                  #[nn.eraseShape prefixEmbeds, nn.eraseShape audioEmbedsValid, nn.eraseShape suffixEmbeds]
                  1
              pure (reshape mergedDyn #[1, seq, cfg.thinkerConfig.textConfig.hiddenSize])
          | none =>
            throw <| IO.userError
              s!"decodeStreamingChunkWithModel expected one contiguous audio-token span of length {audioLen}, but prompt ids were not span-compatible"

      let (generated, nextPromptCache) ←
        model.generateGreedyFromInputsEmbedsWithPromptCache
          inputIds
          inputsEmbeds
          (promptTokenIds := promptIds)
          (prefixCache := cache0.promptCache)
          (maxNewTokens := maxNewTokens)
          (eosTokenIds := eosTokenIds)

      let nextCache : StreamingDecodeCache cfg := {
        promptCache := some nextPromptCache
        promptTokenCache := promptTokenCache'
        promptDeviceCache := promptDeviceCache'
        audioEncoderCache := audioCache'
        frontendCache := frontendCache'
      }

      let outSeq := generated.1
      let outIds := generated.2
      if outSeq <= seq then
        pure ("", some nextCache)
      else
        let newSeq := outSeq - seq
        let newOnly : T #[1, newSeq] := data.slice outIds 1 seq newSeq
        let flat : T #[newSeq] := reshape newOnly #[newSeq]
        let idsU64 ← data.tensorToUInt64Array flat
        let ids := idsU64.map (fun x => x.toUInt32)
        pure (tokenizer.qwen3.decodeText tok ids, some nextCache)

/-- Lean decode helper that runs one model generation from `(prompt, audio)` pair
    and returns updated prompt cache for the next hop. -/
def decodeStreamingChunkWithModelCached
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (prompt : String)
    (audio16k : Array Float)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (String × Option (StreamingPromptCache cfg)) := do
  let stateCache : StreamingDecodeCache cfg := { promptCache := prefixCache }
  let (text, cache') ←
    decodeStreamingChunkWithModelStateCached
      model
      tok
      preprocessor
      prompt
      .rollingWindow
      prompt
      audio16k
      (cache := some stateCache)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
  pure (text, cache'.bind (fun c => c.promptCache))

/-- Lean decode helper that runs one model generation from `(prompt, audio)` pair. -/
def decodeStreamingChunkWithModel
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (prompt : String)
    (audio16k : Array Float)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO String := do
  let out ← decodeStreamingChunkWithModelCached
    model
    tok
    preprocessor
    prompt
    audio16k
    (prefixCache := none)
    (maxNewTokens := maxNewTokens)
    (eosTokenIds := eosTokenIds)
  pure out.1

/-- Model-backed streaming decode step with reusable full decode-cache state. -/
def streamingTranscribeWithModelStateCached
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (cache : Option (StreamingDecodeCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingDecodeCache cfg)) := do
  let cacheRef ← IO.mkRef cache
  let basePrompt := state.promptRaw
  let decodeMode := state.decodeMode
  let decodeFn : StreamingDecodeFn := fun prompt audio => do
    let cache0 ← cacheRef.get
    let (text, cache') ←
      decodeStreamingChunkWithModelStateCached
        model
        tok
        preprocessor
        basePrompt
        decodeMode
        prompt
        audio
        (cache := cache0)
        (maxNewTokens := maxNewTokens)
        (eosTokenIds := eosTokenIds)
    cacheRef.set cache'
    pure text
  let next ← streamingTranscribe tok decodeFn pcm16k state
  let cache' ← cacheRef.get
  pure (next, cache')

/-- Model-backed streaming decode step with reusable prompt cache across hops. -/
def streamingTranscribeWithModelCached
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingPromptCache cfg)) := do
  let stateCache : StreamingDecodeCache cfg := { promptCache := prefixCache }
  let (next, cache') ←
    streamingTranscribeWithModelStateCached
      model
      tok
      preprocessor
      pcm16k
      state
      (cache := some stateCache)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
  pure (next, cache'.bind (fun c => c.promptCache))

/-- Model-backed streaming decode step (no Python/vLLM bridge). -/
def streamingTranscribeWithModel
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState := do
  let (next, _cache) ←
    streamingTranscribeWithModelCached
      model
      tok
      preprocessor
      pcm16k
      state
      (prefixCache := none)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
  pure next

/-- Model-backed streaming finish step with reusable full decode-cache state. -/
def finishStreamingTranscribeWithModelStateCached
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (cache : Option (StreamingDecodeCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingDecodeCache cfg)) := do
  let cacheRef ← IO.mkRef cache
  let basePrompt := state.promptRaw
  let decodeMode := state.decodeMode
  let decodeFn : StreamingDecodeFn := fun prompt audio => do
    let cache0 ← cacheRef.get
    let (text, cache') ←
      decodeStreamingChunkWithModelStateCached
        model
        tok
        preprocessor
        basePrompt
        decodeMode
        prompt
        audio
        (cache := cache0)
        (maxNewTokens := maxNewTokens)
        (eosTokenIds := eosTokenIds)
    cacheRef.set cache'
    pure text
  let next ← finishStreamingTranscribe tok decodeFn state
  let cache' ← cacheRef.get
  pure (next, cache')

/-- Model-backed streaming finish step with reusable prompt cache across hops. -/
def finishStreamingTranscribeWithModelCached
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingPromptCache cfg)) := do
  let stateCache : StreamingDecodeCache cfg := { promptCache := prefixCache }
  let (next, cache') ←
    finishStreamingTranscribeWithModelStateCached
      model
      tok
      preprocessor
      state
      (cache := some stateCache)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
  pure (next, cache'.bind (fun c => c.promptCache))

/-- Model-backed streaming finish step (flush tail audio). -/
def finishStreamingTranscribeWithModel
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState := do
  let (next, _cache) ←
    finishStreamingTranscribeWithModelCached
      model
      tok
      preprocessor
      state
      (prefixCache := none)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
  pure next

namespace Qwen3ASRForConditionalGeneration

/-- Method-style init wrapper over `torch.qwen3asr.initStreamingState`. -/
def initStreamingState
    (m : Qwen3ASRForConditionalGeneration cfg)
    (context : String := "")
    (language : Option String := none)
    (unfixedChunkNum : Nat := 2)
    (unfixedTokenNum : Nat := 5)
    (promptMaxTokens : Nat := 96)
    (chunkSizeSec : Float := 2.0)
    (stepSizeSec : Float := 0.5)
    (decodeMode : StreamingDecodeMode := .rollingWindow)
    : IO ASRStreamingState :=
  torch.qwen3asr.initStreamingState
    m.supportLanguages context language unfixedChunkNum unfixedTokenNum promptMaxTokens chunkSizeSec stepSizeSec decodeMode

/-- Method-style streaming step with Lean model backend. -/
def streamingTranscribeStateCached
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (cache : Option (StreamingDecodeCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingDecodeCache cfg)) :=
  streamingTranscribeWithModelStateCached
    m
    tok
    preprocessor
    pcm16k
    state
    cache
    maxNewTokens
    eosTokenIds

/-- Method-style streaming step with Lean model backend. -/
def streamingTranscribeCached
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingPromptCache cfg)) :=
  streamingTranscribeWithModelCached
    m
    tok
    preprocessor
    pcm16k
    state
    prefixCache
    maxNewTokens
    eosTokenIds

/-- Method-style streaming step with Lean model backend. -/
def streamingTranscribe
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState :=
  streamingTranscribeWithModel m tok preprocessor pcm16k state maxNewTokens eosTokenIds

/-- Method-style streaming flush with Lean model backend. -/
def finishStreamingTranscribeStateCached
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (cache : Option (StreamingDecodeCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingDecodeCache cfg)) :=
  finishStreamingTranscribeWithModelStateCached
    m
    tok
    preprocessor
    state
    cache
    maxNewTokens
    eosTokenIds

/-- Method-style streaming flush with Lean model backend. -/
def finishStreamingTranscribeCached
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (prefixCache : Option (StreamingPromptCache cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (ASRStreamingState × Option (StreamingPromptCache cfg)) :=
  finishStreamingTranscribeWithModelCached
    m
    tok
    preprocessor
    state
    prefixCache
    maxNewTokens
    eosTokenIds

/-- Method-style streaming flush with Lean model backend. -/
def finishStreamingTranscribe
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState :=
  finishStreamingTranscribeWithModel m tok preprocessor state maxNewTokens eosTokenIds

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
