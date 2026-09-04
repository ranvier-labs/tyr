import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonTrace

open LeanTest
open Tyr.EventSkeleton

private def assertOk (res : Except String Unit) (label : String) : IO Unit := do
  match res with
  | .ok () => pure ()
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

@[test]
def testSupportPolicyExactnessTagsDynamicSupport : IO Unit := do
  LeanTest.assertTrue
    (SupportPolicy.fullSupport.defaultExactness == MoveExactness.exact)
    "Full support should be exact"
  LeanTest.assertTrue
    ((SupportPolicy.sampled 7).defaultExactness == MoveExactness.unbiasedEstimator)
    "Sampled support should be an unbiased estimator"
  LeanTest.assertTrue
    ((SupportPolicy.topK 3).defaultExactness == MoveExactness.controlledApproximation)
    "Top-k support should be a fixed-trace approximation"
  LeanTest.assertTrue
    ((SupportPolicy.learnedTail 4).defaultExactness == MoveExactness.learnedApproximation)
    "Learned-tail support should be a learned approximation"

@[test]
def testTopKMarkTraceRecordsRuntimeSelectedIds : IO Unit := do
  let support := RuntimeSupport.topK #[42, 7] (some 100)
  let data : CategoricalMarkData := {
    probs := #[0.6, 0.4]
    messages := #[
      { value := 3.0, stateAdjoint := #[1.0] },
      { value := 4.0, stateAdjoint := #[2.0] }
    ]
  }
  let entry := EventTraceEntry.categoricalMark 9 support data
  assertOk entry.validate? "top-k categorical mark trace"

  match entry.support? with
  | none => LeanTest.fail "Expected categorical mark support"
  | some recorded =>
      LeanTest.assertEqual recorded.retainedCount 2
        "Trace should preserve retained support count"
      LeanTest.assertEqual recorded.totalCandidates? (some 100)
        "Trace should preserve optional source candidate count"
      LeanTest.assertTrue (recorded.selectedIds[0]! == 42 && recorded.selectedIds[1]! == 7)
        s!"Trace should preserve runtime-selected IDs, got {recorded.selectedIds}"

  let moves := entry.moves
  LeanTest.assertEqual moves.size 1
    "Categorical mark entry should project to one local move"
  LeanTest.assertTrue (moves[0]!.kind == SkeletonMoveKind.markMarginalize)
    "Top-k categorical support still uses the mark-marginalization kernel"
  LeanTest.assertTrue (moves[0]!.exactness == MoveExactness.controlledApproximation)
    "Top-k categorical support should be tagged as a controlled approximation"

@[test]
def testSampledMarkTraceRequiresSingleSelectedId : IO Unit := do
  let badSupport : RuntimeSupport := {
    policy := .sampled 5
    selectedIds := #[5, 8]
  }
  let data : SampledMarkData := {
    message := { value := 10.0, stateAdjoint := #[1.0] }
    baseline := 2.0
    logProbStateGrad := #[0.5]
  }
  let badEntry := EventTraceEntry.sampledMark 11 badSupport data
  match badEntry.validate? with
  | .ok () => LeanTest.fail "Sampled mark with two selected IDs should fail validation"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "exactly one selected id")
        s!"Expected sampled-cardinality diagnostic, got: {msg}"

  let goodEntry := EventTraceEntry.sampledMark 11 (RuntimeSupport.sampled 5) data
  assertOk goodEntry.validate? "sampled mark trace"
  let moves := goodEntry.moves
  LeanTest.assertEqual moves.size 1
    "Sampled mark entry should project to one score move"
  LeanTest.assertTrue (moves[0]!.kind == SkeletonMoveKind.markScoreSample)
    "Sampled mark support should use the score-sample kernel"
  LeanTest.assertTrue (moves[0]!.exactness == MoveExactness.unbiasedEstimator)
    "Sampled mark support should be tagged as an unbiased estimator"

@[test]
def testBranchTraceRecordsRuntimeChildrenAndApproximationTag : IO Unit := do
  let support : RuntimeSupport := {
    policy := .threshold 0.25
    selectedIds := #[2, 4]
    totalCandidates? := some 8
  }
  let child0 : BranchChild := {
    resetJac := #[#[1.0]]
    a := #[0.5]
    message := { value := 1.0, stateAdjoint := #[2.0] }
  }
  let child1 : BranchChild := {
    resetJac := #[#[2.0]]
    a := #[1.0]
    message := { value := 3.0, stateAdjoint := #[4.0] }
  }
  let data : BranchEventData := {
    children := #[child0, child1]
    gamma := 1.0
  }
  let entry := EventTraceEntry.branch 12 support data
  assertOk entry.validate? "threshold branch trace"

  let moves := entry.moves
  LeanTest.assertEqual moves.size 1
    "Branch entry should project to one aggregate move"
  LeanTest.assertTrue (moves[0]!.kind == SkeletonMoveKind.branchAggregate)
    "Branch support should use the branch aggregation kernel"
  LeanTest.assertTrue (moves[0]!.exactness == MoveExactness.controlledApproximation)
    "Threshold-selected branch support should be tagged as fixed-trace approximation"

@[test]
def testDynamicTraceProjectsMovesInRecordedOrder : IO Unit := do
  let seg : AcceptedStepSegment := {
    id := 1
    attemptIndex := 0
    tStart := 0.0
    tAttempt := 0.1
    tAfter := 0.1
  }
  let support := RuntimeSupport.full 1
  let markData : CategoricalMarkData := {
    probs := #[1.0]
    messages := #[{ value := 2.0, stateAdjoint := #[3.0] }]
  }
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval seg)
      |>.push (.categoricalMark 3 support markData)

  assertOk trace.validate? "dynamic event trace"
  let moves := trace.moves
  LeanTest.assertEqual moves.size 3
    "Interval plus mark trace should project to interval, checkpoint, and mark moves"
  LeanTest.assertTrue (moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First move should follow first recorded interval entry"
  LeanTest.assertTrue (moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second move should be the interval checkpoint boundary"
  LeanTest.assertTrue (moves[2]!.kind == SkeletonMoveKind.markMarginalize)
    "Third move should follow the recorded mark entry"

  let supports := trace.supports
  LeanTest.assertEqual supports.size 1
    "Trace should expose one dynamic support-bearing entry"
  LeanTest.assertTrue (supports[0]!.selectedIds[0]! == 0)
    "Full support constructor should record local support ID 0"

end Tests.EventSkeletonTrace
