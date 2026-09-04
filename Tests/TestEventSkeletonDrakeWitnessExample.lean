import LeanTest
import Tyr.EventSkeleton.Examples.DrakeWitness

namespace Tests.EventSkeletonDrakeWitnessExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.DrakeWitness

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def getResult : IO EndToEndResult := do
  match buildEndToEnd? with
  | .ok result => pure result
  | .error msg => LeanTest.fail s!"Drake witness example failed to build: {msg}"

@[test]
def testDrakeReferencesAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel.cc"))
    "Example should reference Drake's rimless-wheel witness/reset example"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/systems/analysis/simulator.cc"))
    "Example should reference Drake's simulator witness-isolation path"

@[test]
def testEndToEndTraceProjectsConcreteSchedule : IO Unit := do
  let result ← getResult
  match result.trace.validate? with
  | .error msg => LeanTest.fail s!"Trace should validate: {msg}"
  | .ok () => pure ()

  LeanTest.assertTrue acceptedWitnessSegment.localizedByEvent
    "The witness interval should end before the originally attempted step"
  LeanTest.assertTrue acceptedWitnessSegment.crossedJumpFlag
    "The witness interval should record the jump transition"

  LeanTest.assertEqual result.moves.size 6
    "Interval, saltation, mark, and branch entries should project to six moves"
  LeanTest.assertTrue (result.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should eliminate the continuous interval"
  LeanTest.assertTrue (result.moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second move should preserve the interval boundary"
  LeanTest.assertTrue (result.moves[2]!.kind == SkeletonMoveKind.saltationTime)
    "Third move should eliminate witness timing"
  LeanTest.assertTrue (result.moves[3]!.kind == SkeletonMoveKind.resetTranspose)
    "Fourth move should apply the reset transpose"
  LeanTest.assertTrue (result.moves[4]!.kind == SkeletonMoveKind.markMarginalize)
    "Fifth move should eliminate the retained contact-mode marks"
  LeanTest.assertTrue (result.moves[4]!.exactness == MoveExactness.controlledApproximation)
    "Top-k dynamic mark support is a fixed-trace approximation"
  LeanTest.assertTrue (result.moves[5]!.kind == SkeletonMoveKind.branchAggregate)
    "Sixth move should aggregate retained branch children"
  LeanTest.assertTrue (result.moves[5]!.exactness == MoveExactness.controlledApproximation)
    "Threshold dynamic branch support is a fixed-trace approximation"

  let supports := result.trace.supports
  LeanTest.assertEqual supports.size 2
    "The example should expose mark and branch dynamic supports"
  LeanTest.assertTrue (supports[0]!.selectedIds == #[101, 7])
    s!"Expected contact-mode source IDs #[101, 7], got {supports[0]!.selectedIds}"
  LeanTest.assertTrue (supports[1]!.selectedIds == #[3, 8])
    s!"Expected branch source IDs #[3, 8], got {supports[1]!.selectedIds}"

@[test]
def testEndToEndReverseMessages : IO Unit := do
  let result ← getResult
  LeanTest.assertTrue (approx result.saltationAlpha (-6.67875) 1.0e-10)
    s!"Expected saltation alpha -6.67875, got {result.saltationAlpha}"
  assertArrayApprox result.preImpactAdjoint #[-4.67875, 0.25] 1.0e-10
    "Saltation should propagate the post-impact cotangent to the pre-impact state"
  assertArrayApprox result.restitutionGrad #[-1.0] 1.0e-10
    "Reset-theta transpose should produce the restitution gradient"

  LeanTest.assertTrue (approx result.markMessage.value 1.9 1.0e-10)
    s!"Expected retained-mode value 1.9, got {result.markMessage.value}"
  assertArrayApprox result.markMessage.stateAdjoint #[-0.38, 0.41] 1.0e-10
    "Mark elimination should include weighted child adjoints plus the probability message"

  LeanTest.assertTrue (approx result.branchResult.value 3.8 1.0e-10)
    s!"Expected branch value 3.8, got {result.branchResult.value}"
  LeanTest.assertTrue (approx result.branchResult.alpha (-0.3) 1.0e-10)
    s!"Expected branch alpha -0.3, got {result.branchResult.alpha}"
  assertArrayApprox result.branchResult.stateAdjoint #[0.5, -0.2] 1.0e-10
    "Branch aggregation should include weighted child resets plus shared timing"

end Tests.EventSkeletonDrakeWitnessExample
