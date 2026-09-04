import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonBranch

open LeanTest
open Tyr.EventSkeleton

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testBranchAggregateWeightedChildrenAndSharedTimingAdjoint : IO Unit := do
  let child0 : BranchChild := {
    weight := 0.25
    resetJac := #[#[2.0, 0.0], #[0.0, 1.0]]
    resetTheta := #[#[1.0], #[2.0]]
    a := #[3.0, 1.0]
    message := { value := 10.0, stateAdjoint := #[1.0, 2.0] }
  }
  let child1 : BranchChild := {
    weight := 0.75
    resetJac := #[#[1.0, 1.0], #[0.0, 2.0]]
    resetTheta := #[#[0.5], #[1.0]]
    a := #[-1.0, 4.0]
    message := { value := 2.0, stateAdjoint := #[2.0, -1.0] }
  }
  let data : BranchEventData := {
    children := #[child0, child1]
    guardGrad := #[1.0, -2.0]
    guardTheta := #[3.0]
    gamma := 2.0
    beta := -1.25
    costStateGrad := #[0.1, 0.2]
    costThetaGrad := #[0.25]
  }
  match data.aggregate? with
  | .error msg => LeanTest.fail s!"aggregate? failed: {msg}"
  | .ok result =>
      LeanTest.assertTrue (approx result.value 4.0 1.0e-12)
        s!"Expected weighted branch value 4.0, got {result.value}"
      LeanTest.assertTrue (approx result.alpha (-1.0) 1.0e-12)
        s!"Expected alpha=-1.0, got {result.alpha}"
      assertArrayApprox result.stateAdjoint #[1.1, 2.7] 1.0e-12
        "Branch aggregation should sum reset-transpose child adjoints plus timing correction"
      assertArrayApprox result.thetaGrad #[-1.5] 1.0e-12
        "Branch theta gradient should sum child reset-theta terms plus guard timing"

@[test]
def testBranchAggregateRejectsZeroGamma : IO Unit := do
  let data : BranchEventData := {
    gamma := 0.0
    children := #[{
      resetJac := #[#[1.0]]
      a := #[1.0]
      message := { stateAdjoint := #[1.0] }
    }]
  }
  match data.aggregate? with
  | .ok _ => LeanTest.fail "Zero gamma should reject non-transverse branch elimination"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "not transverse")
        s!"Expected transversality diagnostic, got: {msg}"

end Tests.EventSkeletonBranch
