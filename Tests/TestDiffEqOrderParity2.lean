import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqOrderParity2

open LeanTest
open torch
open torch.DiffEq

private def finalSaved {Y S C : Type} [Inhabited Y]
    (label : String) (sol : Solution Y S C) : IO Y := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure default
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure default

private def solveEndpoint
    (label : String)
    (solver : AbstractSolver (ODETerm Float Unit) Float Float Time Unit)
    (dt : Float) : IO Float := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
  LeanTest.assertTrue (sol.result == Result.successful)
    s!"{label}: solve should succeed"
  finalSaved label sol

private def endpointError
    (label : String)
    (solver : AbstractSolver (ODETerm Float Unit) Float Float Time Unit)
    (dt : Float) : IO Float := do
  let y1 ← solveEndpoint label solver dt
  let exact := Float.exp (-1.0)
  pure (Float.abs (y1 - exact))

private def assertTrend
    (label : String)
    (errCoarse errMedium errFine : Float)
    (minRatio1 minRatio2 : Float)
    (fineTol : Float) : IO Unit := do
  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"{label}: expected coarse/medium/fine monotone decrease, got {errCoarse}, {errMedium}, {errFine}"
  let ratio1 := if errMedium <= 1.0e-16 then 0.0 else errCoarse / errMedium
  let ratio2 := if errFine <= 1.0e-16 then 0.0 else errMedium / errFine
  LeanTest.assertTrue (ratio1 > minRatio1 && ratio2 > minRatio2)
    s!"{label}: trend ratios too weak, got {ratio1}, {ratio2}"
  LeanTest.assertTrue (errFine < fineTol)
    s!"{label}: fine-grid error too large: {errFine}"

/--
Deterministic order-trend checks inspired by:
- `../diffrax/test/test_integrate.py::test_ode_order` (fixed-step refinement trend)
- `../diffrax/test/test_solver.py` (deterministic solver stepping behavior)
-/
@[test] def testHeunOrderTrendDeterministic : IO Unit := do
  let solver :=
    Heun.solver
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
  let errCoarse ← endpointError "Heun coarse" solver 0.2
  let errMedium ← endpointError "Heun medium" solver 0.1
  let errFine ← endpointError "Heun fine" solver 0.05
  assertTrend "Heun order trend" errCoarse errMedium errFine 3.0 3.0 2.5e-4

@[test] def testMidpointOrderTrendDeterministic : IO Unit := do
  let solver :=
    Midpoint.solver
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let errCoarse ← endpointError "Midpoint coarse" solver 0.2
  let errMedium ← endpointError "Midpoint medium" solver 0.1
  let errFine ← endpointError "Midpoint fine" solver 0.05
  assertTrend "Midpoint order trend" errCoarse errMedium errFine 3.0 3.0 2.5e-4

@[test] def testRalstonOrderTrendDeterministic : IO Unit := do
  let solver :=
    Ralston.solver
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let errCoarse ← endpointError "Ralston coarse" solver 0.2
  let errMedium ← endpointError "Ralston medium" solver 0.1
  let errFine ← endpointError "Ralston fine" solver 0.05
  assertTrend "Ralston order trend" errCoarse errMedium errFine 3.0 3.0 2.5e-4

@[test] def testBosh3OrderTrendDeterministic : IO Unit := do
  let solver :=
    Bosh3.solver
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let errCoarse ← endpointError "Bosh3 coarse" solver 0.25
  let errMedium ← endpointError "Bosh3 medium" solver 0.125
  let errFine ← endpointError "Bosh3 fine" solver 0.0625
  assertTrend "Bosh3 order trend" errCoarse errMedium errFine 4.5 4.5 2.0e-5

def run : IO Unit := do
  testHeunOrderTrendDeterministic
  testMidpointOrderTrendDeterministic
  testRalstonOrderTrendDeterministic
  testBosh3OrderTrendDeterministic

end Tests.DiffEqOrderParity2
