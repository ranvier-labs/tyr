import LeanTest
import Tyr.DiffEq.RootFinder

namespace Tests.DiffEqRootFinder

open torch.DiffEq

private def methods : Array RootFindMethod := #[
  .fixedPoint { maxIters := 128 },
  .adaptiveFixedPoint { maxIters := 128, stepMin := 0.5, stepMax := 0.5 },
  .normRatioFixedPoint { maxIters := 128, stepMin := 0.5, stepMax := 0.5 }
]

@[test] def testFixedPointMethodsSolveContractiveAffineMaps : IO Unit := do
  for method in methods do
    for slope in #[(0.5 : Float), -0.5] do
      let result := RootFinder.solve method (fun (y : Float) => slope * y + 1.0) 0.0
      let expected := 1.0 / (1.0 - slope)
      LeanTest.assertTrue result.converged "Contractive affine map should converge"
      LeanTest.assertTrue (result.value.isFinite && Float.abs (result.value - expected) < 1.0e-5)
        s!"Expected affine fixed point {expected}, got {result.value}"
    let step := fun (y : Float × Float) => (0.25 * y.2 + 1.0, 0.25 * y.1 - 1.0)
    let coupled := RootFinder.solve method step (0.0, 0.0)
    LeanTest.assertTrue coupled.converged "Coupled contractive map should converge"
    LeanTest.assertTrue
      (Float.abs (coupled.value.1 - 0.8) < 1.0e-5 && Float.abs (coupled.value.2 + 0.8) < 1.0e-5)
      s!"Expected coupled fixed point (0.8, -0.8), got {coupled.value}"

@[test] def testRelaxationDefaultsAndNoncontractiveFailure : IO Unit := do
  let defaults : Array RootFindMethod := #[
    .fixedPoint {}, .adaptiveFixedPoint {}, .normRatioFixedPoint {}
  ]
  for method in defaults do
    for target in #[(2.0 : Float), -2.0] do
      let result := RootFinder.solve method (fun (_ : Float) => target) 0.0
      LeanTest.assertTrue (result.converged && result.value == target)
        "Constant fixed-point maps should converge with default settings"
    -- This affine equation has the root -1, but its fixed-point map expands.
    -- The scalar relaxation methods must report failure, not imply Newton behavior.
    let expanding := RootFinder.solve method (fun (y : Float) => 2.0 * y + 1.0) 0.0
    LeanTest.assertTrue (!expanding.converged && expanding.value.isFinite)
      "An expanding affine map should exhaust the fixed-point iteration budget"
    let nonfinite := RootFinder.solve method (fun (_ : Float) => 0.0 / 0.0) 0.0
    LeanTest.assertTrue (!nonfinite.converged) "Nonfinite residuals must not report convergence"

@[test] def testFixedPointMethodToleranceOverrides : IO Unit := do
  for method in methods do
    let bounded := method.withTolerances (some 1.0e-8) (some 1.0e-8) (some 0)
    let result := RootFinder.solve bounded (fun (y : Float) => 0.5 * y + 1.0) 0.0
    LeanTest.assertTrue (!result.converged && result.iterations == 0 && result.value == 0.0)
      "Tolerance overrides must preserve the selected method and honor its iteration budget"

def run : IO Unit := do
  testFixedPointMethodsSolveContractiveAffineMaps
  testRelaxationDefaultsAndNoncontractiveFailure
  testFixedPointMethodToleranceOverrides

end Tests.DiffEqRootFinder
