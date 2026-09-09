import Std.Data.HashSet
import Tyr.AD.Sparse.Transform

/-!
# Tyr.AD.Sparse.Validate

Structural and finite-value validation for sparse linear maps.
-/

namespace Tyr.AD.Sparse

/-- Reject nonfinite coefficients before they can enter sparse arithmetic or
    disappear during zero removal. This also applies to uncoalesced entries. -/
def validateFiniteEntries (entries : Array SparseEntry) : Except String Unit := do
  for entry in entries do
    unless entry.weight.isFinite do
      throw s!"Sparse entry has nonfinite weight: src={entry.src}, dst={entry.dst}, weight={entry.weight}."

private def entryInBounds (m : SparseLinearMap) (e : SparseEntry) : Bool :=
  let srcOk :=
    match m.inDim? with
    | some n => e.src < n
    | none => true
  let dstOk :=
    match m.outDim? with
    | some n => e.dst < n
    | none => true
  srcOk && dstOk

private def validateEntriesInBounds (m : SparseLinearMap) : Except String Unit :=
  match m.entries.find? (fun e => !(entryInBounds m e)) with
  | some bad =>
    .error s!"Sparse entry out of bounds: src={bad.src}, dst={bad.dst}, shape=({m.inDim?}, {m.outDim?})."
  | none =>
    .ok ()

private def hasNoDuplicateCoords (entries : Array SparseEntry) : Bool := Id.run do
  let mut seen : Std.HashSet (Nat × Nat) := {}
  for e in entries do
    let key := (e.src, e.dst)
    if seen.contains key then
      return false
    seen := seen.insert key
  return true

private def validateNoDuplicateCoords (m : SparseLinearMap) : Except String Unit :=
  if hasNoDuplicateCoords m.entries then
    .ok ()
  else
    .error "Sparse map has duplicate (src,dst) coordinates; coalescing required before validation."

private def validateDeclaredShape (m : SparseLinearMap) : Except String Unit :=
  match m.shape? with
  | some s =>
    if validShape s then
      .ok ()
    else
      .error s!"Invalid sparse map shape: inDim={s.inDim}, outDim={s.outDim}."
  | none => .ok ()

/-- Validate map dimensions, coordinates, and finite coefficients. -/
def validateMap (m : SparseLinearMap) : Except String Unit := do
  validateDeclaredShape m
  validateEntriesInBounds m
  validateFiniteEntries m.entries
  validateNoDuplicateCoords m

end Tyr.AD.Sparse
