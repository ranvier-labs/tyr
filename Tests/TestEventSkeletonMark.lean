import LeanTest
import Tyr.EventSkeleton

namespace Tests.EventSkeletonMark

open LeanTest
open Tyr.EventSkeleton

private def assertArrayApprox
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testExactCategoricalMarkEliminationIncludesProbabilityMessage : IO Unit := do
  let data : CategoricalMarkData := {
    probs := #[0.25, 0.75]
    messages := #[
      { value := 10.0, stateAdjoint := #[1.0, 0.0], thetaGrad := #[0.5] },
      { value := 2.0, stateAdjoint := #[0.0, 2.0], thetaGrad := #[-1.0] }
    ]
    probStateJac := #[#[1.0, 2.0], #[-1.0, 0.5]]
    probThetaJac := #[#[3.0], #[-2.0]]
  }
  match data.exactEliminate? with
  | .error msg => LeanTest.fail s!"exactEliminate? failed: {msg}"
  | .ok msg =>
      LeanTest.assertEqual msg.value 4.0
        "Expected marginalized value sum_y pi_y Q_y"
      assertArrayApprox msg.stateAdjoint #[8.25, 22.5] 1.0e-12
        "Exact mark elimination should include weighted child state and (D_x pi)^T Q"
      assertArrayApprox msg.thetaGrad #[25.375] 1.0e-12
        "Exact mark elimination should include weighted child theta and (D_theta pi)^T Q"

@[test]
def testSampledCategoricalMarkScoreUpdate : IO Unit := do
  let data : SampledMarkData := {
    message := { value := 10.0, stateAdjoint := #[1.0, 2.0], thetaGrad := #[0.5] }
    baseline := 4.0
    logProbStateGrad := #[0.1, -0.2]
    logProbThetaGrad := #[0.25]
  }
  let msg := data.eliminate
  LeanTest.assertEqual msg.value 10.0
    "Sampled mark keeps the sampled downstream value"
  assertArrayApprox msg.stateAdjoint #[1.6, 0.8] 1.0e-12
    "Sampled mark should add (Q-b) grad_x log pi"
  assertArrayApprox msg.thetaGrad #[2.0] 1.0e-12
    "Sampled mark should add (Q-b) grad_theta log pi"

@[test]
def testSimplexHitCotangentSubtractsVelocityBaseline : IO Unit := do
  match simplexHitCotangent? #[2.0, 1.0] #[5.0, 2.0] with
  | .error msg => LeanTest.fail s!"simplexHitCotangent? failed: {msg}"
  | .ok cot =>
      assertArrayApprox cot #[1.0, -2.0] 1.0e-12
        "Simplex-hit cotangent should subtract the velocity-weighted baseline"

@[test]
def testSimplexHitCotangentRejectsZeroVelocity : IO Unit := do
  match simplexHitCotangent? #[0.0, 0.0] #[1.0, 2.0] with
  | .ok _ => LeanTest.fail "Zero total evidence velocity should be rejected"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "not transverse")
        s!"Expected transversality diagnostic, got: {msg}"

end Tests.EventSkeletonMark
