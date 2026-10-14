/-
  Tyr/Tokenizer/TokenBytes.lean

  Token byte mapping for bits-per-byte (BPB) evaluation.

  Based on nanochat's token_bytes.pt:
  - Maps each token ID → number of UTF-8 bytes
  - Special tokens → 0 bytes (don't count toward BPB)
  - Enables vocab-size-independent evaluation metric
-/
import Tyr.Tokenizer.Types
import Tyr.Torch

/-! # Token byte counts

`tokenizer.TokenBytes` maps each token to its byte length so bits-per-byte can
be computed, with totals over arrays, tensors and masks.
-/

namespace tokenizer

open torch

/-- Token bytes mapping for BPB calculation.
    Maps each token ID to the number of UTF-8 bytes it represents.
    Special tokens have 0 bytes (don't contribute to BPB). -/
structure TokenBytes where
  /-- Bytes per token ID as Int64 tensor of shape [vocabSize] -/
  bytes : T #[]
  /-- Vocabulary size -/
  vocabSize : UInt32
  deriving Repr

/-- Build token bytes mapping from a BPE tokenizer.
    For each token:
    - Regular tokens: count UTF-8 bytes from idToBytes
    - Special tokens: 0 bytes -/
def TokenBytes.fromTokenizer (tok : BPETokenizer) : TokenBytes := Id.run do
  let mut byteCounts : Array Int64 := Array.mkEmpty tok.vocabSize.toNat

  for id in [:tok.vocabSize.toNat] do
    let tokenId : TokenId := id.toUInt32
    if tok.isSpecialToken tokenId then
      -- Special tokens contribute 0 bytes
      byteCounts := byteCounts.push 0
    else
      match tok.getTokenBytes tokenId with
      | some ba => byteCounts := byteCounts.push ba.size.toInt64
      | none => byteCounts := byteCounts.push 0

  let bytesTensor := data.fromInt64Array byteCounts
  { bytes := bytesTensor, vocabSize := tok.vocabSize }

/-- Compute total bytes from an array of token IDs.
    Uses the token bytes mapping to sum up byte counts. -/
def TokenBytes.totalBytes (tb : TokenBytes) (tokenIds : Array UInt32) : IO UInt64 := do
  if tokenIds.isEmpty then return 0
  -- Convert token IDs to Int64 tensor for indexing
  let idxTensor := data.fromInt64Array (tokenIds.map (·.toNat.toInt64))
  -- Use index_select to get byte counts for each token
  -- We need to reshape bytes to have a known dimension for indexSelect
  let flatBytes := reshape tb.bytes #[tb.vocabSize.toUInt64]
  let bytesPerToken := data.indexSelect flatBytes 0 (reshape idxTensor #[tokenIds.size.toUInt64])
  -- Sum all bytes
  let totalFloat := nn.item (nn.sumAll bytesPerToken)
  return totalFloat.toUInt64

/-- Compute total bytes for a 1D tensor of token IDs.
    More efficient than array-based version for tensor inputs. -/
def TokenBytes.totalBytesFromTensor (tb : TokenBytes) (tokenIds : T #[n]) : IO UInt64 := do
  -- Index into bytes to get byte counts per token
  let flatBytes := reshape tb.bytes #[tb.vocabSize.toUInt64]
  let bytesPerToken := data.indexSelect flatBytes 0 tokenIds
  -- Sum all bytes
  let totalFloat := nn.item (nn.sumAll bytesPerToken)
  return totalFloat.toUInt64

/-- Compute total bytes with a loss mask.
    Only counts bytes for tokens where mask > 0.
    Used in BPB evaluation where we only count trained tokens.
    Takes batch size and seq length as parameters since shape is erased. -/
def TokenBytes.totalBytesWithMask (tb : TokenBytes)
    (tokenIds : T #[b, s]) (mask : T #[b, s])
    : IO UInt64 := do
  -- Flatten for indexing
  let numTokens := b * s

  let flatIds := reshape tokenIds #[numTokens]
  let flatMask := reshape mask #[numTokens]

  -- Get byte counts per token
  let flatBytes := reshape tb.bytes #[tb.vocabSize.toUInt64]
  let bytesPerToken := data.indexSelect flatBytes 0 flatIds

  -- Multiply by mask and sum (mask is float, so result is float)
  let bytesFloat := toFloat' bytesPerToken
  let maskedBytes := bytesFloat * flatMask
  let total := nn.item (nn.sumAll maskedBytes)
  return total.toUInt64

/-- Compute total bytes with a loss mask (shape-erased version).
    Takes batch size and sequence length as explicit parameters. -/
def TokenBytes.totalBytesWithMask' (tb : TokenBytes)
    (tokenIds mask : T #[])
    (batchSize seqLen : UInt64)
    : IO UInt64 := do
  let numTokens := batchSize * seqLen

  let flatIds := reshape tokenIds #[numTokens]
  let flatMask := reshape mask #[numTokens]

  -- Get byte counts per token
  let flatBytes := reshape tb.bytes #[tb.vocabSize.toUInt64]
  let bytesPerToken := data.indexSelect flatBytes 0 flatIds

  -- Multiply by mask and sum
  let bytesFloat := toFloat' bytesPerToken
  let maskedBytes := bytesFloat * flatMask
  let total := nn.item (nn.sumAll maskedBytes)
  return total.toUInt64

/-- Save token bytes mapping to a file -/
def TokenBytes.save (tb : TokenBytes) (path : String) : IO Unit := do
  data.saveTensor tb.bytes path

/-- Load token bytes mapping from a file -/
def TokenBytes.load (path : String) (vocabSize : UInt32) : IO TokenBytes := do
  let bytes ← data.loadTensor #[] path
  return { bytes, vocabSize }

/-- Create a dummy token bytes mapping for testing.
    Assumes each token represents 2 bytes on average (rough approximation). -/
def TokenBytes.dummy (vocabSize : UInt32) : TokenBytes := Id.run do
  let mut byteCounts : Array Int64 := Array.mkEmpty vocabSize.toNat
  for _ in [:vocabSize.toNat] do
    byteCounts := byteCounts.push 2  -- Default 2 bytes per token
  let bytesTensor := data.fromInt64Array byteCounts
  { bytes := bytesTensor, vocabSize }

/-- Create a realistic token bytes mapping for testing.
    Uses a distribution typical of BPE tokenizers:
    - Single bytes (0-255): 1 byte each
    - Common words: 3-6 bytes
    - Special tokens (last 100): 0 bytes -/
def TokenBytes.realistic (vocabSize : UInt32) (numSpecialTokens : UInt32 := 100) : TokenBytes := Id.run do
  let mut byteCounts : Array Int64 := Array.mkEmpty vocabSize.toNat

  for id in [:vocabSize.toNat] do
    let byteCount : Int64 :=
      if id < 256 then
        -- Base bytes: 1 byte each
        1
      else if id >= vocabSize.toNat - numSpecialTokens.toNat then
        -- Special tokens: 0 bytes
        0
      else
        -- Regular tokens: average ~3 bytes (BPE merges)
        3
    byteCounts := byteCounts.push byteCount

  let bytesTensor := data.fromInt64Array byteCounts
  { bytes := bytesTensor, vocabSize }

end tokenizer
