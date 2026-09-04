import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqInterpolationParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertApprox (label : String) (actual expected tol : Float) : IO Unit :=
  LeanTest.assertTrue (approx actual expected tol)
    s!"{label}: expected {expected}, got {actual}"

private def finalSavedValue {S C : Type}
    (label : String) (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure 0.0
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure 0.0

@[test] def testLinearPathEndpointIncrementAndDerivativeParity : IO Unit := do
  let path := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let y0 := path.evaluate path.t0 none true
  let y1 := path.evaluate path.t1 none true
  assertApprox "Linear path endpoint t0" y0 1.0 1e-12
  assertApprox "Linear path endpoint t1" y1 5.0 1e-12

  let inc01 := path.evaluate path.t0 (some path.t1) true
  assertApprox "Linear path full increment equals endpoint difference" inc01 (y1 - y0) 1e-12

  let tA : Time := 0.35
  let tB : Time := 1.45
  let incAB := path.evaluate tA (some tB) true
  let diffAB := path.evaluate tB none true - path.evaluate tA none true
  assertApprox "Linear path increment consistency" incAB diffAB 1e-12

  match path.derivative 0.73 true with
  | some d =>
      let eps : Time := 1e-4
      let fd := (path.evaluate (0.73 + eps) none true - path.evaluate (0.73 - eps) none true) / (2.0 * eps)
      assertApprox "Linear path derivative finite-difference sanity" d fd 1e-9
  | none =>
      LeanTest.fail "Linear path should expose derivative"

@[test] def testCubicPathEndpointIncrementAndDerivativeParity : IO Unit := do
  let path :=
    AbstractPath.cubicHermiteInterpolation 0.0 1.0
      (0.0 : Float) (1.0 : Float) (0.0 : Float) (0.0 : Float)
  let y0 := path.evaluate path.t0 none true
  let y1 := path.evaluate path.t1 none true
  assertApprox "Cubic path endpoint t0" y0 0.0 1e-12
  assertApprox "Cubic path endpoint t1" y1 1.0 1e-12

  let tA : Time := 0.25
  let tB : Time := 0.75
  let incAB := path.evaluate tA (some tB) true
  let diffAB := path.evaluate tB none true - path.evaluate tA none true
  assertApprox "Cubic path increment consistency" incAB diffAB 1e-12

  match path.derivative 0.4 true with
  | some d =>
      let eps : Time := 1e-4
      let fd := (path.evaluate (0.4 + eps) none true - path.evaluate (0.4 - eps) none true) / (2.0 * eps)
      assertApprox "Cubic path derivative finite-difference sanity" d fd 1e-5
  | none =>
      LeanTest.fail "Cubic Hermite path should expose derivative"

@[test] def testDenseSolutionEndpointIncrementAndDerivativeParity : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Midpoint.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) ()
      (saveat := { dense := true, t1 := true })
  LeanTest.assertTrue (sol.result == Result.successful)
    "Dense Midpoint solve should succeed"

  let yStart := sol.evaluate 0.0
  assertApprox "Dense solution endpoint t0" yStart 1.0 1e-12

  let yEndDense := sol.evaluate 1.0
  let yEndSaved ← finalSavedValue "Dense Midpoint" sol
  assertApprox "Dense solution endpoint t1 equals saved y(t1)" yEndDense yEndSaved 1e-12

  let tA : Time := 0.35
  let tB : Time := 0.85
  let incAB := sol.evaluate tA (some tB) true
  let diffAB := sol.evaluate tB - sol.evaluate tA
  assertApprox "Dense solution increment consistency" incAB diffAB 1e-10

  let tProbe : Time := 0.37
  let eps : Time := 1e-4
  let d := sol.derivative tProbe
  let fd := (sol.evaluate (tProbe + eps) - sol.evaluate (tProbe - eps)) / (2.0 * eps)
  assertApprox "Dense solution derivative finite-difference sanity" d fd 1e-7

@[test] def testPiecewiseDenseBoundaryLeftRightParity : IO Unit := do
  let segLeft :=
    LocalLinearDenseInfo.toInterpolation
      ({ t0 := 0.0, t1 := 1.0, y0 := (0.0 : Float), y1 := (1.0 : Float) } :
        LocalLinearDenseInfo Float)
  let segRight :=
    LocalLinearDenseInfo.toInterpolation
      ({ t0 := 1.0, t1 := 2.0, y0 := (10.0 : Float), y1 := (12.0 : Float) } :
        LocalLinearDenseInfo Float)
  let interp :=
    PiecewiseDenseInterpolation.toDense
      ({ ts := #[0.0, 1.0, 2.0], segments := #[segLeft, segRight] } :
        PiecewiseDenseInterpolation Float)

  let vLeft := interp.evaluate 1.0 none true
  let vRight := interp.evaluate 1.0 none false
  assertApprox "Piecewise dense left value at knot" vLeft 1.0 1e-12
  assertApprox "Piecewise dense right value at knot" vRight 10.0 1e-12

  let dLeft := interp.derivative 1.0 true
  let dRight := interp.derivative 1.0 false
  assertApprox "Piecewise dense left derivative at knot" dLeft 1.0 1e-12
  assertApprox "Piecewise dense right derivative at knot" dRight 2.0 1e-12

@[test] def testLocalLinearDenseZeroLengthParity : IO Unit := do
  /-
  Diffrax reference: `../diffrax/test/test_local_interpolation.py` (zero-length local interval).
  -/
  let interp :=
    LocalLinearDenseInfo.toInterpolation
      ({ t0 := 2.0, t1 := 2.0, y0 := (2.1 : Float), y1 := (2.2 : Float) } :
        LocalLinearDenseInfo Float)
  let value := interp.evaluate 2.0 none true
  let inc := interp.evaluate 2.0 (some 2.0) true
  let deriv := interp.derivative 2.0 true
  assertApprox "Local linear zero-length value" value 2.1 1e-12
  assertApprox "Local linear zero-length increment" inc 0.0 1e-12
  assertApprox "Local linear zero-length derivative" deriv 0.0 1e-12

@[test] def testLocalLinearDenseIncrementShiftAndAntisymmetryParity : IO Unit := do
  /-
  Diffrax reference: `../diffrax/test/test_local_interpolation.py` (increment semantics).
  -/
  let interp :=
    LocalLinearDenseInfo.toInterpolation
      ({ t0 := 2.0, t1 := 3.3, y0 := (2.1 : Float), y1 := (2.2 : Float) } :
        LocalLinearDenseInfo Float)

  let incA := interp.evaluate 2.6 (some 2.8) true
  let incB := interp.evaluate 2.7 (some 2.9) true
  assertApprox "Local linear increment shift invariance" incA incB 1e-12

  let fwd := interp.evaluate 2.8 (some 2.9) true
  let bwd := interp.evaluate 2.9 (some 2.8) true
  assertApprox "Local linear increment antisymmetry" bwd (-fwd) 1e-12

  let incAC := interp.evaluate 2.6 (some 2.9) true
  let incAB := interp.evaluate 2.6 (some 2.75) true
  let incBC := interp.evaluate 2.75 (some 2.9) true
  assertApprox "Local linear increment additivity" incAC (incAB + incBC) 1e-12

@[test] def testLinearInterpolationKnotAndSlopeParity : IO Unit := do
  /-
  Diffrax reference: `../diffrax/test/test_global_interpolation.py`
  (`LinearInterpolation` knot recovery + per-segment slope behavior).
  -/
  let interp :=
    LinearInterpolation.toDense
      ({ ts := #[0.0, 2.0, 3.0, 5.0], ys := #[1.0, 5.0, 2.0, 10.0] } :
        LinearInterpolation Float)

  assertApprox "LinearInterpolation knot value t=0" (interp.evaluate 0.0 none true) 1.0 1e-12
  assertApprox "LinearInterpolation knot value t=2" (interp.evaluate 2.0 none true) 5.0 1e-12
  assertApprox "LinearInterpolation knot value t=3" (interp.evaluate 3.0 none true) 2.0 1e-12
  assertApprox "LinearInterpolation knot value t=5" (interp.evaluate 5.0 none true) 10.0 1e-12

  assertApprox "LinearInterpolation slope in [0,2]" (interp.derivative 1.0 true) 2.0 1e-12
  assertApprox "LinearInterpolation slope in [2,3]" (interp.derivative 2.5 true) (-3.0) 1e-12
  assertApprox "LinearInterpolation slope in [3,5]" (interp.derivative 4.0 true) 4.0 1e-12
  assertApprox "LinearInterpolation left derivative at knot t=2" (interp.derivative 2.0 true) 2.0 1e-12
  assertApprox "LinearInterpolation right derivative at knot t=2" (interp.derivative 2.0 false) (-3.0) 1e-12
  assertApprox "LinearInterpolation left derivative at knot t=3" (interp.derivative 3.0 true) (-3.0) 1e-12
  assertApprox "LinearInterpolation right derivative at knot t=3" (interp.derivative 3.0 false) 4.0 1e-12

  let inc := interp.evaluate 1.5 (some 4.5) true
  let diff := interp.evaluate 4.5 none true - interp.evaluate 1.5 none true
  assertApprox "LinearInterpolation increment consistency" inc diff 1e-12

def run : IO Unit := do
  testLinearPathEndpointIncrementAndDerivativeParity
  testCubicPathEndpointIncrementAndDerivativeParity
  testDenseSolutionEndpointIncrementAndDerivativeParity
  testPiecewiseDenseBoundaryLeftRightParity
  testLocalLinearDenseZeroLengthParity
  testLocalLinearDenseIncrementShiftAndAntisymmetryParity
  testLinearInterpolationKnotAndSlopeParity

end Tests.DiffEqInterpolationParity
