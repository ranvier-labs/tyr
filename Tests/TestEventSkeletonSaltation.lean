import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonSaltation

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
def testSaltationReverseMatchesExplicitMatrix : IO Unit := do
  let data : SaltationData := {
    resetJac := #[#[2.0, 0.0], #[0.0, 3.0]]
    guardGrad := #[1.0, -1.0]
    a := #[4.0, -2.0]
    gamma := 2.0
  }
  let pPlus := #[0.5, -1.5]
  match data.reverseState? pPlus, data.saltationTransposeApply? pPlus with
  | .ok rankOne, .ok dense =>
      assertArrayApprox rankOne dense 1.0e-12
        "rank-one reverse update should equal explicit S^T p"
      assertArrayApprox rankOne #[3.5, -7.0] 1.0e-12
        "rank-one reverse update should match the hand-computed value"
  | .error msg, _ => LeanTest.fail s!"reverseState? failed: {msg}"
  | _, .error msg => LeanTest.fail s!"saltationTransposeApply? failed: {msg}"

@[test]
def testSaltationEventCostAndThetaUpdate : IO Unit := do
  let data : SaltationData := {
    resetJac := #[#[2.0, 0.0], #[0.0, 3.0]]
    guardGrad := #[1.0, -1.0]
    a := #[4.0, -2.0]
    gamma := 2.0
    beta := 1.0
    costStateGrad := #[0.25, 0.5]
    resetTheta := #[#[1.0, 0.0], #[0.0, 2.0]]
    guardTheta := #[2.0, -1.0]
    costThetaGrad := #[0.1, 0.2]
  }
  let pPlus := #[0.5, -1.5]
  match data.timingAdjoint? pPlus with
  | .error msg => LeanTest.fail s!"timingAdjoint? failed: {msg}"
  | .ok alpha =>
      LeanTest.assertTrue (approx alpha 2.0 1.0e-12)
        s!"Expected alpha=2.0, got {alpha}"
  match data.reverseState? pPlus with
  | .error msg => LeanTest.fail s!"reverseState? failed: {msg}"
  | .ok pMinus =>
      assertArrayApprox pMinus #[3.25, -6.0] 1.0e-12
        "event cost should shift the reverse state update"
  match data.reverseTheta? pPlus with
  | .error msg => LeanTest.fail s!"reverseTheta? failed: {msg}"
  | .ok theta =>
      assertArrayApprox theta #[4.6, -4.8] 1.0e-12
        "theta update should include reset transpose, guard timing, and event cost"

@[test]
def testSaltationDataFromFieldsComputesGammaAndA : IO Unit := do
  let data :=
    SaltationData.mkFromFields
      #[#[2.0, 0.0], #[0.0, 3.0]]
      #[1.0, -1.0]
      #[1.0, 2.0]
      #[5.0, 7.0]
      (resetTime := #[0.5, 1.0])
      (guardTime := 3.0)
  LeanTest.assertTrue (approx data.gamma 2.0 1.0e-12)
    s!"Expected gamma=2.0, got {data.gamma}"
  assertArrayApprox data.a #[2.5, 0.0] 1.0e-12
    "a should equal fPlus - R_x fMinus - R_t"

@[test]
def testSaltationRejectsZeroGamma : IO Unit := do
  let data : SaltationData := {
    resetJac := #[#[1.0]]
    guardGrad := #[1.0]
    a := #[1.0]
    gamma := 0.0
  }
  match data.reverseState? #[1.0] with
  | .ok _ => LeanTest.fail "Zero gamma should reject non-transverse event elimination"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "not transverse")
        s!"Expected transversality diagnostic, got: {msg}"

end Tests.EventSkeletonSaltation
