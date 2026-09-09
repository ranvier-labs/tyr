import LeanTest
import Tyr.AD.Elim.Eliminate

namespace Tests.ADSparseNumerics

open Tyr.AD.Sparse

private def entry (src dst : Nat) (weight : Float) : SparseEntry :=
  { src, dst, weight }

private def scalar (weight : Float) : SparseLinearMap :=
  { repr := .named `scalar, inDim? := some 1, outDim? := some 1,
    entries := #[entry 0 0 weight] }

private def assertOk (result : Except String α) (label : String) : IO α :=
  match result with
  | .ok value => pure value
  | .error error => throw (IO.userError s!"{label}: {error}")

private def assertNonfiniteRejected (result : Except String α) (label : String) : IO Unit := do
  match result with
  | .ok _ => LeanTest.fail s!"{label}: nonfinite arithmetic was accepted"
  | .error error =>
      LeanTest.assertTrue (error.contains "nonfinite") s!"{label}: unexpected error {error}"

private def assertApprox (actual expected : Float) (label : String) : IO Unit :=
  LeanTest.assertTrue
    (actual.isFinite && expected.isFinite &&
      Float.abs (actual - expected) ≤ 1e-12 * (1.0 + Float.abs expected))
    s!"{label}: expected {expected}, got {actual}"

private def scalarValue (map : SparseLinearMap) : Float :=
  map.entries.foldl (fun value e => value + e.weight) 0.0

private def assertEntries (actual expected : Array SparseEntry) (label : String) : IO Unit :=
  LeanTest.assertTrue (actual == expected)
    s!"{label}: expected {repr expected}, got {repr actual}"

@[test] def testSparsePreservesTinyNonzeroCoefficients : IO Unit := do
  for weight in #[(1e-12 : Float), 1e-16, -1e-20, Float.ofBits 1] do
    let coalesced ← assertOk (coalesceEntries #[entry 0 0 weight]) "Tiny coalescing"
    assertEntries coalesced #[entry 0 0 weight] "Coalescing must preserve every nonzero coefficient"
    let added ← assertOk (add (scalar weight) (zeroMap 1 1)) "Adding zero to tiny coefficient"
    assertEntries added.entries #[entry 0 0 weight] "Adding zero must not prune tiny values"
    let composed ← assertOk (compose (scalar weight) (identityMap 1)) "Identity composition"
    assertEntries composed.entries #[entry 0 0 weight] "Identity composition must preserve tiny values"

@[test] def testSparseCompositionPreservesScaledChains : IO Unit := do
  let a := scalar 1e-8
  let b := scalar 1e-8
  let c := scalar 1e16
  let ab ← assertOk (compose a b) "First two small factors"
  LeanTest.assertEqual ab.entries.size 1 "The intermediate 1e-16 derivative must remain present"
  LeanTest.assertTrue ((scalarValue ab) > 0.0) "Small intermediate derivative must not become zero"
  let firstThenLast ← assertOk (compose ab c) "Compose small intermediate with large factor"
  let bc ← assertOk (compose b c) "Last two factors"
  let lastThenFirst ← assertOk (compose a bc) "Compose in reverse association"
  assertApprox (scalarValue firstThenLast) 1.0 "Forward association"
  assertApprox (scalarValue lastThenFirst) 1.0 "Reverse association"

@[test] def testSparseEliminationOrderPreservesScaledDerivative : IO Unit := do
  let edges : Array Tyr.AD.JaxprLike.LocalJacEdge := #[
    { src := 0, dst := 1, map := scalar 1e-8 },
    { src := 1, dst := 2, map := scalar 1e-8 },
    { src := 2, dst := 3, map := scalar 1e16 }
  ]
  let graph ← assertOk
    (Tyr.AD.Elim.ofLocalJacEdgesWithPartitions edges #[0] #[3] #[1, 2]) "Build scaled chain"
  for order in #[#[1, 2], #[2, 1]] do
    let result ← assertOk (Tyr.AD.Elim.runCompleteElimination graph order) "Eliminate scaled chain"
    let some derivative := Tyr.AD.Elim.findEdge? result.graph 0 3
      | LeanTest.fail "Elimination lost the input-output derivative edge"
    assertApprox (scalarValue derivative) 1.0 s!"Elimination order {order}"

@[test] def testSparseCoalescesDuplicatesAndExactCancellation : IO Unit := do
  let coalesced ← assertOk (coalesceEntries #[
    entry 1 0 2e-20, entry 0 0 1e-20, entry 1 0 (-1e-20),
    entry 0 1 (-4.0), entry 0 0 (-1e-20), entry 2 2 (-0.0)
  ]) "Duplicate accumulation and cancellation"
  assertEntries coalesced #[entry 0 1 (-4.0), entry 1 0 1e-20]
    "Only exact cancellations and signed zeros should be removed"
  let cancelled ← assertOk (add (scalar 1e-20) (scalar (-1e-20))) "Exact sparse cancellation"
  assertEntries cancelled.entries #[] "Exact cancellation produces a sparse zero map"
  LeanTest.assertEqual cancelled.inDim? (some 1) "Cancellation preserves input dimension"
  LeanTest.assertEqual cancelled.outDim? (some 1) "Cancellation preserves output dimension"
  let doubled ← assertOk (add (scalar 1e-20) (scalar 1e-20)) "Tiny duplicate sum"
  assertEntries doubled.entries #[entry 0 0 2e-20] "A nonzero duplicate sum must survive"

@[test] def testSparseRejectsNonfiniteInputsBeforeArithmetic : IO Unit := do
  for weight in #[(0.0 / 0.0 : Float), 1.0 / 0.0, -1.0 / 0.0] do
    let invalid := scalar weight
    assertNonfiniteRejected (validateMap invalid) "Map validation"
    assertNonfiniteRejected (coalesceEntries invalid.entries) "Direct coalescing"
    assertNonfiniteRejected (add invalid (zeroMap 1 1)) "Invalid left addition input"
    assertNonfiniteRejected (add (zeroMap 1 1) invalid) "Invalid right addition input"
    assertNonfiniteRejected (compose invalid (zeroMap 1 1)) "Invalid composition into zero"
    assertNonfiniteRejected (compose (zeroMap 1 1) invalid) "Zero composition into invalid map"
    assertNonfiniteRejected (compose identityLike invalid) "Invalid map through identity shortcut"
    assertNonfiniteRejected (compose invalid identityLike) "Invalid map before identity shortcut"

@[test] def testSparseRejectsOverflowingSumsAndProducts : IO Unit := do
  assertNonfiniteRejected (coalesceEntries #[entry 0 0 1e308, entry 0 0 1e308])
    "Duplicate sum overflow"
  assertNonfiniteRejected (coalesceEntries #[entry 0 0 1e308, entry 0 0 1e308, entry 0 0 (-1e308)])
    "Intermediate overflow must not be hidden by later cancellation"
  assertNonfiniteRejected (add (scalar 1e308) (scalar 1e308)) "Sparse addition overflow"
  assertNonfiniteRejected (compose (scalar 1e308) (scalar 1e308)) "Sparse product overflow"
  let incoming : SparseLinearMap := {
    repr := .named `incoming, inDim? := some 1, outDim? := some 2,
    entries := #[entry 0 0 1e308, entry 0 1 1e308]
  }
  let outgoing : SparseLinearMap := {
    repr := .named `outgoing, inDim? := some 2, outDim? := some 1,
    entries := #[entry 0 0 1.0, entry 1 0 1.0]
  }
  assertNonfiniteRejected (compose incoming outgoing)
    "Finite products whose accumulation overflows"

private def fromDense (matrix : Array (Array Float)) : SparseLinearMap := Id.run do
  let mut entries := #[]
  for dst in [:matrix.size] do
    for src in [:matrix[dst]!.size] do
      let weight := matrix[dst]![src]!
      if weight != 0.0 then entries := entries.push (entry src dst weight)
  return {
    repr := .named `denseReferenceInput,
    inDim? := some (matrix.getD 0 #[]).size, outDim? := some matrix.size,
    entries := entries.reverse
  }

-- Ordinary dense multiplication is independent of sparse coordinate matching
-- and coalescing; matrices use row=output and column=input.
private def denseMultiply (left right : Array (Array Float)) : Array (Array Float) := Id.run do
  let columns := (right.getD 0 #[]).size
  let mut result := #[]
  for row in [:left.size] do
    let mut values := #[]
    for column in [:columns] do
      let mut value := 0.0
      for middle in [:right.size] do
        value := value + left[row]![middle]! * right[middle]![column]!
      values := values.push value
    result := result.push values
  return result

private def assertDense (map : SparseLinearMap) (matrix : Array (Array Float))
    (label : String) : IO Unit := do
  let columns := (matrix.getD 0 #[]).size
  LeanTest.assertEqual map.inDim? (some columns) s!"{label}: input dimension"
  LeanTest.assertEqual map.outDim? (some matrix.size) s!"{label}: output dimension"
  let _ ← assertOk (validateMap map) s!"{label}: result validation"
  for dst in [:matrix.size] do
    for src in [:columns] do
      let actual := map.entries.foldl
        (fun value e => if e.src == src && e.dst == dst then value + e.weight else value) 0.0
      assertApprox actual matrix[dst]![src]! s!"{label}[{dst},{src}]"

@[test] def testSparseAlgebraMatchesDenseReference : IO Unit := do
  let a := #[#[1e-8, 2.0], #[-3.0, 0.0], #[0.5, -1e-8]]
  let b := #[#[1e-8, 0.0, -4.0], #[0.25, 2.0, 0.0]]
  let c := #[#[1e8, -0.5], #[0.0, 3.0], #[-2.0, 1e-8]]
  let ab ← assertOk (compose (fromDense a) (fromDense b)) "Rectangular sparse product"
  assertDense ab (denseMultiply b a) "B*A"
  let abc ← assertOk (compose ab (fromDense c)) "Left-associated sparse product"
  assertDense abc (denseMultiply c (denseMultiply b a)) "C*(B*A)"
  let bc ← assertOk (compose (fromDense b) (fromDense c)) "Right-associated first product"
  let abc' ← assertOk (compose (fromDense a) bc) "Right-associated sparse product"
  assertDense abc' (denseMultiply c (denseMultiply b a)) "(C*B)*A"
  let doubled ← assertOk (add ab ab) "Sparse addition against dense reference"
  assertDense doubled ((denseMultiply b a).map (·.map (· * 2.0))) "2*(B*A)"

@[test] def testSparseValidatesResultAfterDimensionInference : IO Unit := do
  let unknown : SparseLinearMap := {
    repr := .named `unknownShape, inDim? := none, outDim? := some 1,
    entries := #[entry 3 0 1.0]
  }
  let _ ← assertOk (validateMap unknown) "Unknown input dimension permits unresolved coordinates"
  match add unknown (zeroMap 2 1) with
  | .ok _ => LeanTest.fail "Inferred result dimension must not leave an out-of-bounds coefficient"
  | .error error => LeanTest.assertTrue (error.contains "out of bounds") "Check the merged result shape"

def run : IO Unit := do
  testSparsePreservesTinyNonzeroCoefficients
  testSparseCompositionPreservesScaledChains
  testSparseEliminationOrderPreservesScaledDerivative
  testSparseCoalescesDuplicatesAndExactCancellation
  testSparseRejectsNonfiniteInputsBeforeArithmetic
  testSparseRejectsOverflowingSumsAndProducts
  testSparseAlgebraMatchesDenseReference
  testSparseValidatesResultAfterDimensionInference

end Tests.ADSparseNumerics
