import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqKencarpInterpolationParity

open LeanTest
open torch
open torch.DiffEq

private abbrev SplitTerm :=
  MultiTerm (ODETerm Float Unit) (ODETerm Float Unit)

private abbrev KenCarp3Solver :=
  AbstractSolver
    SplitTerm
    Float
    (Float × Float)
    (Time × Time)
    Unit

private def denseValue {S C : Type}
    (label : String) (sol : Solution Float S C) (t : Time) : IO Float := do
  match sol.interpolation with
  | some _ =>
      pure (sol.evaluate t)
  | none =>
      LeanTest.fail s!"{label}: expected dense interpolation"
      pure 0.0

@[test] def testKencarp3PolyDenseImprovesOverHermiteAndSplitFallbacks : IO Unit := do
  let tProbe : Time := 0.37
  let explicit : ODETerm Float Unit := {
    vectorField := fun _t y _ => -0.3 * y
  }
  let implicit : ODETerm Float Unit := {
    vectorField := fun _t y _ => -1.6 * y
  }
  let terms : SplitTerm := {
    term1 := explicit
    term2 := implicit
  }
  let solverPoly : KenCarp3Solver :=
    Kencarp3.solver
      (cfg := { denseKind := .kencarp3Poly2 })
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solverSplit : KenCarp3Solver :=
    Kencarp3.solver
      (cfg := { denseKind := .splitAtStage 1 "kencarp3-stage2-split-hermite" })
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solverHermite : KenCarp3Solver :=
    Kencarp3.solver
      (cfg := { denseKind := .hermite })
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solveDense := fun (solver : KenCarp3Solver) =>
    diffeqsolve
      (Term := SplitTerm)
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.5) (1.0 : Float) ()
      (saveat := { dense := true, t1 := false })

  let solPoly := solveDense solverPoly
  let solSplit := solveDense solverSplit
  let solHermite := solveDense solverHermite
  LeanTest.assertTrue (solPoly.result == Result.successful)
    "KenCarp3 poly dense solve should succeed"
  LeanTest.assertTrue (solSplit.result == Result.successful)
    "KenCarp3 split dense solve should succeed"
  LeanTest.assertTrue (solHermite.result == Result.successful)
    "KenCarp3 Hermite dense solve should succeed"

  let yPoly ← denseValue "KenCarp3 poly" solPoly tProbe
  let ySplit ← denseValue "KenCarp3 split" solSplit tProbe
  let yHermite ← denseValue "KenCarp3 hermite" solHermite tProbe
  let yExact := Float.exp (-1.9 * tProbe)
  let errPoly := Float.abs (yPoly - yExact)
  let errSplit := Float.abs (ySplit - yExact)
  let errHermite := Float.abs (yHermite - yExact)

  LeanTest.assertTrue (errPoly < errHermite)
    s!"KenCarp3 poly dense should improve over Hermite fallback: {errPoly} vs {errHermite}"
  LeanTest.assertTrue (errPoly < 0.8 * errHermite)
    s!"KenCarp3 poly dense improvement over Hermite should be meaningful: {errPoly} vs {errHermite}"
  LeanTest.assertTrue (errPoly < errSplit)
    s!"KenCarp3 poly dense should improve over split-stage Hermite: {errPoly} vs {errSplit}"

end Tests.DiffEqKencarpInterpolationParity
