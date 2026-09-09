import Tyr.AD.Sparse.Validate

/-!
# Tyr.AD.Sparse.Add

Sparse additive merge and coalescing.
-/

namespace Tyr.AD.Sparse

private def entryLe (a b : SparseEntry) : Bool :=
  if a.src = b.src then a.dst ≤ b.dst else a.src < b.src

private def sortEntries (entries : Array SparseEntry) : Array SparseEntry :=
  -- A non-strict comparator preserves input order within duplicate coordinates.
  (entries.toList.mergeSort entryLe).toArray

/-- Coalesce duplicate `(src,dst)` entries and drop only exact floating-point
    zeros. Nonzero coefficients are never pruned by a magnitude threshold.
    Reject nonfinite inputs and overflowing intermediate sums. -/
def coalesceEntries (entries : Array SparseEntry) : Except String (Array SparseEntry) := do
  validateFiniteEntries entries
  let sorted := sortEntries entries
  let mut out : Array SparseEntry := #[]
  for e in sorted do
    match out.back? with
    | some last =>
      if last.src = e.src && last.dst = e.dst then
        let weight := last.weight + e.weight
        unless weight.isFinite do
          throw s!"Sparse coalescing produced a nonfinite weight: src={e.src}, dst={e.dst}."
        let merged : SparseEntry := { src := last.src, dst := last.dst, weight := weight }
        out := out.pop.push merged
      else
        out := out.push e
    | none =>
      out := out.push e
  return out.filter (fun e => e.weight != 0.0)

/-- Add two sparse maps with strict shape compatibility checks. -/
def add (lhs rhs : SparseLinearMap) : Except String SparseLinearMap := do
  validateMap lhs
  validateMap rhs
  let inDim? ← mergeDim? "input" lhs.inDim? rhs.inDim?
  let outDim? ← mergeDim? "output" lhs.outDim? rhs.outDim?
  let entries ← coalesceEntries (lhs.entries ++ rhs.entries)
  let result : SparseLinearMap := {
    repr := .add lhs.repr rhs.repr
    inDim? := inDim?
    outDim? := outDim?
    entries := entries
  }
  validateMap result
  return result

end Tyr.AD.Sparse
