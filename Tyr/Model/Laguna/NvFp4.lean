/-
  Tyr/Model/Laguna/NvFp4.lean

  NVFP4 dequantization for the poolside Laguna-S-2.1-NVFP4 checkpoint
  (compressed-tensors `nvfp4-pack-quantized` format).

  Format (verified against the real checkpoint):
  - `weight_packed`: U8 `[out, in/2]`, two E2M1 values per byte along dim -1;
    LOW nibble = element `2i`, HIGH nibble = element `2i+1`.
  - `weight_scale`: F8_E4M3 `[out, in/16]`, one scale per 16-element group
    along the input dim.
  - `weight_global_scale`: F32 scalar (one per expert for stacked banks).
  - Dequant: `W[i,j] = e2m1(nibble) * (scale[i, j/16].to(float32) / globalScale)`.
    E2M1: bits 0-2 index into magnitudes `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`,
    bit 3 is the sign.

  Implementation notes:
  - Nibble unpacking uses a byte-level lookup table (`e2m1LutLo`/`e2m1LutHi`,
    256 F32 entries each) plus an embedding gather, avoiding per-element bit
    math (`index_select_1d` would be the natural choice but its FFI binding is
    ABI-mismatched and segfaults; see `dequantCore`).
  - F8_E4M3 scales are cast to F32 with the libtorch `.to(kFloat32)` cast
    (`toFloat'`), which handles float8 dtypes in this vendored build.
  - The arithmetic is done in F32 in the same op order as the reference
    (`e2m1 * (scale / global)`), then rounded once to BF16, so results are
    bitwise-identical to a PyTorch F32 dequant followed by `.to(bfloat16)`.
-/
import Tyr.Torch

/-! # NVFP4 dequantization

Decodes NVFP4 (E2M1 4-bit) weights: `torch.laguna.nvfp4.e2m1Value` decodes one
nibble (3 magnitude bits, 1 sign bit), and `torch.laguna.nvfp4.dequantMatrix` /
`torch.laguna.nvfp4.dequantBank` expand packed matrices and expert banks.
-/

namespace torch.laguna.nvfp4

/-- E2M1 magnitude table indexed by the low 3 bits of a nibble. -/
private def e2m1Magnitudes : Array Float :=
  #[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

/-- Decode one E2M1 nibble (bits 0-2 magnitude index, bit 3 sign). -/
private def e2m1Value (nibble : Nat) : Float :=
  let mag := e2m1Magnitudes.getD (nibble % 8) 0.0
  if nibble >= 8 then -mag else mag

/-- F32 lookup table for the LOW nibble (element `2i`) of each byte. -/
private def e2m1LutLo : T #[] :=
  data.fromFloatArray (Array.ofFn (n := 256) fun b => e2m1Value (b.val % 16))

/-- F32 lookup table for the HIGH nibble (element `2i+1`) of each byte. -/
private def e2m1LutHi : T #[] :=
  data.fromFloatArray (Array.ofFn (n := 256) fun b => e2m1Value (b.val / 16))

/-- Shared dequant kernel.

    - `packed`: U8 tensor, runtime shape `[lead, out, in/2]`
    - `scales`: F8_E4M3 tensor, runtime shape `[lead, out, in/16]`
    - `globalB`: F32 tensor broadcastable against `[lead, 1, 1, 1]`
      (i.e. one global scale per leading slice)

    Returns BF16 tensor of runtime shape `[lead, out, in]`. -/
private def dequantCore
    (packed scales globalB : T #[])
    (lead outFeatures inFeatures : UInt64)
    : IO (T #[]) := do
  if inFeatures % 16 != 0 then
    throw <| IO.userError
      s!"nvfp4.dequantCore: inFeatures={inFeatures} not divisible by 16"
  let numGroups := inFeatures / 16
  let numPacked := lead * outFeatures * (inFeatures / 2)
  let device := packed.device
  -- Byte LUTs on the same device as the packed weights.
  -- NOTE: `index_select_1d` would be the natural gather, but its FFI binding
  -- is ABI-mismatched (one C++ shape parameter for two Lean shape implicits)
  -- and segfaults; `nn.embedding1d` performs the same row gather.
  let lutLo : T #[256, 1] := reshape (e2m1LutLo.to device) #[256, 1]
  let lutHi : T #[256, 1] := reshape (e2m1LutHi.to device) #[256, 1]
  -- Unpack nibbles via LUT gathers.
  let idx : T #[numPacked] := data.toLong (reshape packed #[numPacked])
  let lo : T #[] := reshape (nn.embedding1d idx lutLo) #[numPacked]
  let hi : T #[] := reshape (nn.embedding1d idx lutHi) #[numPacked]
  -- Interleave low/high: byte i holds elements 2i (low) and 2i+1 (high).
  let pairs : T #[] := nn.cat_dyn #[reshape lo #[numPacked, 1], reshape hi #[numPacked, 1]] 1
  let vals4 : T #[] := reshape pairs #[lead, outFeatures, numGroups, 16]
  -- Per-group scales in F32, broadcast against the 16 elements of each group.
  let scalesF : T #[] := reshape (toFloat' scales) #[lead, outFeatures, numGroups, 1]
  -- Reference op order: divide scale by global first, then multiply by e2m1.
  let scaleDiv : T #[] := nn.div scalesF globalB
  let prod4 : T #[] := mul' scaleDiv vals4
  pure (toBFloat16' (reshape prod4 #[lead, outFeatures, inFeatures]))

/-- Dequantize a single NVFP4-packed matrix.
    - `packed`: U8 `[outFeatures, inFeatures/2]`
    - `scales`: F8_E4M3 `[outFeatures, inFeatures/16]`
    - `globalScale`: F32 scalar (0- or 1-element tensor)
    Returns BF16 `[outFeatures, inFeatures]`. -/
def dequantMatrix
    (packed scales globalScale : T #[])
    (outFeatures inFeatures : UInt64)
    : IO (T #[]) := do
  let globalB : T #[] := reshape (toFloat' globalScale) #[1, 1, 1, 1]
  let packed3 : T #[] := reshape packed #[1, outFeatures, inFeatures / 2]
  let scales3 : T #[] := reshape scales #[1, outFeatures, inFeatures / 16]
  let out3 ← dequantCore packed3 scales3 globalB 1 outFeatures inFeatures
  pure (reshape out3 #[outFeatures, inFeatures])

/-- Dequantize a stacked NVFP4 expert bank.
    - `packed`: U8 `[numExperts, outFeatures, inFeatures/2]`
    - `scales`: F8_E4M3 `[numExperts, outFeatures, inFeatures/16]`
    - `globalScale`: F32 `[numExperts]`
    Returns BF16 `[numExperts, outFeatures, inFeatures]`. -/
def dequantBank
    (packed scales globalScale : T #[])
    (numExperts outFeatures inFeatures : UInt64)
    : IO (T #[]) := do
  let globalB : T #[] := reshape (toFloat' globalScale) #[numExperts, 1, 1, 1]
  dequantCore packed scales globalB numExperts outFeatures inFeatures

end torch.laguna.nvfp4
