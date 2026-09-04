import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonMoves

open LeanTest
open Tyr.EventSkeleton

@[test]
def testMoveKindDefaultExactness : IO Unit := do
  LeanTest.assertTrue
    (SkeletonMoveKind.saltationTime.defaultExactness == MoveExactness.exact)
    "Saltation-time elimination is exact under transversality"
  LeanTest.assertTrue
    (SkeletonMoveKind.resetTranspose.defaultExactness == MoveExactness.exact)
    "Reset-transpose elimination should be exact"
  LeanTest.assertTrue
    (SkeletonMoveKind.branchAggregate.defaultExactness == MoveExactness.exact)
    "Branch aggregation is exact for an explicit branch set"
  LeanTest.assertTrue
    (SkeletonMoveKind.markScoreSample.defaultExactness == MoveExactness.unbiasedEstimator)
    "Sampled categorical-score moves should be marked as unbiased estimators"
  LeanTest.assertTrue
    (SkeletonMoveKind.learnedComplement.defaultExactness == MoveExactness.learnedApproximation)
    "Learned complements should be marked as learned approximations"
  LeanTest.assertTrue
    (SkeletonMoveKind.clockedUpdate.defaultExactness == MoveExactness.exact)
    "Clocked deterministic discrete updates should be exact"

@[test]
def testAcceptedSegmentPlanIncludesIntervalAndCheckpointMoves : IO Unit := do
  let seg : AcceptedStepSegment := {
    id := 7
    attemptIndex := 11
    tStart := 0.25
    tAttempt := 0.5
    tAfter := 0.4
    madeJumpBefore := false
    madeJumpAfter := true
  }
  LeanTest.assertTrue seg.localizedByEvent
    "Segment should report event localization when tAfter differs from tAttempt"
  LeanTest.assertTrue seg.crossedJumpFlag
    "Segment should report a jump flag crossing"

  let moves := planForAcceptedSegment seg
  LeanTest.assertTrue
    ((moves.map (fun move => move.kind)) ==
      #[SkeletonMoveKind.intervalAdjoint, SkeletonMoveKind.checkpointBoundary])
    "Default segment plan should include an interval adjoint and checkpoint boundary"

  let noCheckpointMoves := planForAcceptedSegment seg (checkpoint := false)
  LeanTest.assertTrue
    ((noCheckpointMoves.map (fun move => move.kind)) == #[SkeletonMoveKind.intervalAdjoint])
    "Checkpoint-free segment plan should keep only the interval adjoint move"

@[test]
def testGraphFromAcceptedSegmentsKeepsMoveVocabularyVisible : IO Unit := do
  let seg0 : AcceptedStepSegment := {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := 0.1
    tAfter := 0.1
  }
  let seg1 : AcceptedStepSegment := {
    id := 1
    attemptIndex := 2
    tStart := 0.1
    tAttempt := 0.3
    tAfter := 0.2
  }
  let g := graphFromAcceptedSegments #[seg0, seg1]
  LeanTest.assertEqual g.vertices.size 2
    "Two accepted segments should create two interval vertices"
  LeanTest.assertTrue (g.containsMoveKind .intervalAdjoint)
    "Interval adjoint move should remain first-class in the skeleton graph"
  LeanTest.assertTrue (g.containsMoveKind .checkpointBoundary)
    "Checkpoint boundary move should remain first-class in the skeleton graph"
  LeanTest.assertEqual g.totalCost.memory 2.0
    "Default checkpoint plan should charge one memory unit per segment"

end Tests.EventSkeletonMoves
