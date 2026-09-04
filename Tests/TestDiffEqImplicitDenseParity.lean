import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqImplicitDenseParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertApprox (label : String) (actual expected tol : Float) : IO Unit :=
  LeanTest.assertTrue (approx actual expected tol)
    s!"{label}: expected {expected}, got {actual}"

private def denseValue {S C : Type}
    (label : String)
    (sol : Solution Float S C)
    (t : Time) : IO Float := do
  match sol.interpolation with
  | some _ =>
      pure (sol.evaluate t)
  | none =>
      LeanTest.fail s!"{label}: expected dense interpolation"
      pure 0.0

private def diffraxHermiteEval
    (t0 t1 y0 y1 k0 k1 t : Float) : Float :=
  if t0 == t1 then
    y0
  else
    let theta := (t - t0) / (t1 - t0)
    let a := k0 + k1 + 2.0 * y0 - 2.0 * y1
    let b := -2.0 * k0 - k1 - 3.0 * y0 + 3.0 * y1
    ((a * theta + b) * theta + k0) * theta + y0

private def diffraxHermiteDeriv
    (t0 t1 y0 y1 k0 k1 t : Float) : Float :=
  if t0 == t1 then
    0.0
  else
    let theta := (t - t0) / (t1 - t0)
    let a := k0 + k1 + 2.0 * y0 - 2.0 * y1
    let b := -2.0 * k0 - k1 - 3.0 * y0 + 3.0 * y1
    let dTheta := 3.0 * a * theta * theta + 2.0 * b * theta + k0
    dTheta / (t1 - t0)

@[test] def testKvaerno3DenseKindCanonicalizesToHermiteParity : IO Unit := do
  let term : ODETerm Float Unit := {
    vectorField := fun t y _ => -2.0 * y + y * y + Float.sin t
  }
  let solverDefault :=
    Kvaerno3.solver
      (cfg := {})
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let solverSplit :=
    Kvaerno3.solver
      (cfg := { denseKind := .splitAtStage 1 "kvaerno3-stage2-split-hermite" })
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let solverHermite :=
    Kvaerno3.solver
      (cfg := { denseKind := .hermite })
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let solveDense := fun (solver : AbstractSolver (ODETerm Float Unit) Float Float Time Unit) =>
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.2) (1.0 : Float) ()
      (saveat := { dense := true, t1 := false })
  let defaultSol := solveDense solverDefault
  let splitSol := solveDense solverSplit
  let hermiteSol := solveDense solverHermite
  LeanTest.assertTrue (defaultSol.result == Result.successful)
    "Kvaerno3 default dense solve should succeed"
  LeanTest.assertTrue (splitSol.result == Result.successful)
    "Kvaerno3 split dense solve should succeed"
  LeanTest.assertTrue (hermiteSol.result == Result.successful)
    "Kvaerno3 Hermite dense solve should succeed"

  for tProbe in #[0.17, 0.37, 0.91] do
    let yDefault ← denseValue "Kvaerno3 default" defaultSol tProbe
    let ySplit ← denseValue "Kvaerno3 split" splitSol tProbe
    let yHermite ← denseValue "Kvaerno3 hermite" hermiteSol tProbe
    assertApprox s!"Kvaerno3 default/hermite parity at t={tProbe}" yDefault yHermite 1e-12
    assertApprox s!"Kvaerno3 split/hermite parity at t={tProbe}" ySplit yHermite 1e-12

  let incDefault := defaultSol.evaluate 0.2 (some 0.8) true
  let incSplit := splitSol.evaluate 0.2 (some 0.8) true
  let incHermite := hermiteSol.evaluate 0.2 (some 0.8) true
  assertApprox "Kvaerno3 default/hermite increment parity" incDefault incHermite 1e-12
  assertApprox "Kvaerno3 split/hermite increment parity" incSplit incHermite 1e-12

  let dDefault := defaultSol.derivative 0.53
  let dSplit := splitSol.derivative 0.53
  let dHermite := hermiteSol.derivative 0.53
  assertApprox "Kvaerno3 default/hermite derivative parity" dDefault dHermite 1e-12
  assertApprox "Kvaerno3 split/hermite derivative parity" dSplit dHermite 1e-12

@[test] def testLocalHermiteMatchesDiffraxThirdOrderFormParity : IO Unit := do
  /-
  Diffrax reference: `../diffrax/diffrax/_local_interpolation.py`
  (`ThirdOrderHermitePolynomialInterpolation`).
  -/
  let t0 : Time := 2.0
  let t1 : Time := 3.9
  let y := fun t => 0.4 + 0.7 * t - 1.1 * t * t + 0.4 * t * t * t
  let dy := fun t => 0.7 - 2.2 * t + 1.2 * t * t
  let k0 := dy t0 * (t1 - t0)
  let k1 := dy t1 * (t1 - t0)
  let interp :=
    LocalHermiteDenseInfo.toInterpolation
      ({ t0 := t0, t1 := t1, y0 := y t0, y1 := y t1, m0 := k0, m1 := k1 } :
        LocalHermiteDenseInfo Float)

  for tProbe in #[2.0, 2.6, 3.1, 3.9] do
    let expected := diffraxHermiteEval t0 t1 (y t0) (y t1) k0 k1 tProbe
    let actual := interp.evaluate tProbe none true
    assertApprox s!"LocalHermite value parity at t={tProbe}" actual expected 1e-12
    let expectedDeriv := diffraxHermiteDeriv t0 t1 (y t0) (y t1) k0 k1 tProbe
    let actualDeriv := interp.derivative tProbe true
    assertApprox s!"LocalHermite derivative parity at t={tProbe}" actualDeriv expectedDeriv 1e-12

  let incActual := interp.evaluate 2.3 (some 3.2) true
  let incExpected :=
    diffraxHermiteEval t0 t1 (y t0) (y t1) k0 k1 3.2 -
    diffraxHermiteEval t0 t1 (y t0) (y t1) k0 k1 2.3
  assertApprox "LocalHermite increment parity with Diffrax form" incActual incExpected 1e-12

  let interpZero :=
    LocalHermiteDenseInfo.toInterpolation
      ({ t0 := 1.5, t1 := 1.5, y0 := (2.3 : Float), y1 := (9.4 : Float),
         m0 := (1.2 : Float), m1 := (-3.4 : Float) } : LocalHermiteDenseInfo Float)
  assertApprox "LocalHermite zero-length value parity" (interpZero.evaluate 1.5 none true) 2.3 1e-12
  assertApprox "LocalHermite zero-length increment parity"
    (interpZero.evaluate 1.5 (some 1.5) true) 0.0 1e-12
  assertApprox "LocalHermite zero-length derivative parity" (interpZero.derivative 1.5 true) 0.0 1e-12

@[test] def testKencarp3DefaultDenseImprovesOverHermiteFallback : IO Unit := do
  let tProbe : Time := 0.54
  let explicit : ODETerm Float Unit := { vectorField := fun _t y _ => -1.0 * y }
  let implicit : ODETerm Float Unit := { vectorField := fun _t y _ => -0.9 * y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) := {
    term1 := explicit
    term2 := implicit
  }
  let solverDefault :=
    Kencarp3.solver
      (cfg := {})
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solverHermite :=
    Kencarp3.solver
      (cfg := { denseKind := .hermite })
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solveDense := fun (solver :
      AbstractSolver
        (MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
        Float
        (Float × Float)
        (Time × Time)
        Unit) =>
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.5) (1.0 : Float) ()
      (saveat := { dense := true, t1 := false })
  let solDefault := solveDense solverDefault
  let solHermite := solveDense solverHermite
  LeanTest.assertTrue (solDefault.result == Result.successful)
    "KenCarp3 default dense solve should succeed"
  LeanTest.assertTrue (solHermite.result == Result.successful)
    "KenCarp3 Hermite fallback solve should succeed"
  let yDefault ← denseValue "KenCarp3 default" solDefault tProbe
  let yHermite ← denseValue "KenCarp3 hermite" solHermite tProbe
  let exact := Float.exp (-1.9 * tProbe)
  let errDefault := Float.abs (yDefault - exact)
  let errHermite := Float.abs (yHermite - exact)
  LeanTest.assertTrue (errDefault < errHermite)
    s!"KenCarp3 default split dense should improve over Hermite fallback: {errDefault} vs {errHermite}"

end Tests.DiffEqImplicitDenseParity
