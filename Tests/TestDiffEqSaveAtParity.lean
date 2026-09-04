import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqSaveAtParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def getStat (name : String) (stats : List (String × Nat)) : Nat :=
  match stats.find? (fun kv => kv.fst == name) with
  | some (_, v) => v
  | none => 0

private def assertSavedTsYsEq {S C : Type}
    (label : String)
    (sol : Solution Float S C)
    (expected : Array Time) : IO Unit := do
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == expected.size)
        s!"{label}: expected {expected.size} saved times, got {ts.size}"
      LeanTest.assertTrue (ys.size == expected.size)
        s!"{label}: expected {expected.size} saved values, got {ys.size}"
      for i in [:expected.size] do
        let tExpected := expected[i]!
        LeanTest.assertTrue (approx ts[i]! tExpected 1e-12)
          s!"{label}: ts[{i}] expected {tExpected}, got {ts[i]!}"
        LeanTest.assertTrue (approx ys[i]! tExpected 1e-12)
          s!"{label}: ys[{i}] expected {tExpected}, got {ys[i]!}"
  | _, _ =>
      LeanTest.fail s!"{label}: expected ts/ys output"

private def assertSavedTsEq {S C : Type}
    (label : String)
    (sol : Solution Float S C)
    (expected : Array Time) : IO Unit := do
  match sol.ts with
  | some ts =>
      LeanTest.assertTrue (ts.size == expected.size)
        s!"{label}: expected {expected.size} saved times, got {ts.size}"
      for i in [:expected.size] do
        let want := expected[i]!
        LeanTest.assertTrue (approx ts[i]! want 1e-12)
          s!"{label}: ts[{i}] expected {want}, got {ts[i]!}"
  | none =>
      LeanTest.fail s!"{label}: expected ts output"

private def solveStepTo (ts : Array Time) (saveat : SaveAt) :
    Solution Float Unit StepToState :=
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -0.5 * y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ctrl : StepTo := { ts := ts }
  diffeqsolve
    (Term := ODETerm Float Unit)
    (Y := Float)
    (VF := Float)
    (Control := Time)
    (Args := Unit)
    (Controller := StepTo)
    term solver ts[0]! ts[ts.size - 1]! none (1.0 : Float) ()
    (saveat := saveat)
    (controller := ctrl)
    (maxSteps := 32)

@[test] def testSaveAtSubsIgnoresSyntheticRootPayload : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let leaf : SubSaveAt := { ts := some #[0.25] }
  let saveat : SaveAt := {
    -- `SaveAt.t1` defaults to true, but with `subs` this should act like a tree root
    -- container and not emit an extra payload entry.
    subs := #[leaf]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "SaveAt(subs=...) solve should succeed"
  assertSavedTsYsEq "SaveAt(subs) root payload suppression" sol #[0.25]

@[test] def testNestedContainerSubSaveAtPayloadIgnored : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let leaf : SubSaveAt := { ts := some #[0.6] }
  let container : SubSaveAt := {
    -- Diffrax-style tree semantics: payload is leaf-defined; non-leaf nodes are
    -- structural containers.
    t1 := true
    subs := #[leaf]
  }
  let saveat : SaveAt := {
    t1 := false
    subs := #[container]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Nested SubSaveAt container solve should succeed"
  assertSavedTsYsEq "Nested SubSaveAt container payload suppression" sol #[0.6]

@[test] def testReverseTimeNestedSubSaveAtUsesPerLeafTsMonotonicity : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let earlyLeaf : SubSaveAt := { ts := some #[0.4, 0.2] }
  let lateLeaf : SubSaveAt := { ts := some #[0.8, 0.6] }
  let saveat : SaveAt := {
    t1 := false
    -- Each leaf is monotone in reverse-time solve direction; sibling concatenation is
    -- intentionally non-monotone and should still be accepted.
    subs := #[earlyLeaf, lateLeaf]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 1.0 0.0 (some 0.1) (1.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Reverse-time nested SubSaveAt solve should succeed with per-leaf monotone ts"
  assertSavedTsYsEq "Reverse-time nested SubSaveAt per-leaf ts monotonicity" sol #[0.4, 0.2, 0.8, 0.6]

@[test] def testSaveAtStepsSkipStepToParity : IO Unit := do
  -- Diffrax reference: `test_saveat_solution_skip_steps`.
  let tsWith7 : Array Time := #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
  let tsWith6 : Array Time := #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  assertSavedTsEq "steps=2 with7"
    (solveStepTo tsWith7 { steps := (2 : Nat), t1 := false }) #[2.0, 4.0, 6.0]
  assertSavedTsEq "steps=2 with6"
    (solveStepTo tsWith6 { steps := (2 : Nat), t1 := false }) #[2.0, 4.0, 6.0]
  assertSavedTsEq "steps=2,t1=true with7"
    (solveStepTo tsWith7 { steps := (2 : Nat), t1 := true }) #[2.0, 4.0, 6.0, 7.0]
  assertSavedTsEq "steps=2,t1=true with6"
    (solveStepTo tsWith6 { steps := (2 : Nat), t1 := true }) #[2.0, 4.0, 6.0]
  assertSavedTsEq "steps=2,t1=true,t0=true with7"
    (solveStepTo tsWith7 { steps := (2 : Nat), t1 := true, t0 := true }) #[0.0, 2.0, 4.0, 6.0, 7.0]
  assertSavedTsEq "steps=2,t1=true,t0=true with6"
    (solveStepTo tsWith6 { steps := (2 : Nat), t1 := true, t0 := true }) #[0.0, 2.0, 4.0, 6.0]

  assertSavedTsEq "steps=3 with7"
    (solveStepTo tsWith7 { steps := (3 : Nat), t1 := false }) #[3.0, 6.0]
  assertSavedTsEq "steps=3 with6"
    (solveStepTo tsWith6 { steps := (3 : Nat), t1 := false }) #[3.0, 6.0]
  assertSavedTsEq "steps=3,t1=true with7"
    (solveStepTo tsWith7 { steps := (3 : Nat), t1 := true }) #[3.0, 6.0, 7.0]
  assertSavedTsEq "steps=3,t1=true with6"
    (solveStepTo tsWith6 { steps := (3 : Nat), t1 := true }) #[3.0, 6.0]
  assertSavedTsEq "steps=3,t1=true,t0=true with7"
    (solveStepTo tsWith7 { steps := (3 : Nat), t1 := true, t0 := true }) #[0.0, 3.0, 6.0, 7.0]
  assertSavedTsEq "steps=3,t1=true,t0=true with6"
    (solveStepTo tsWith6 { steps := (3 : Nat), t1 := true, t0 := true }) #[0.0, 3.0, 6.0]

@[test] def testSaveAtStepsSkipVsTsParity : IO Unit := do
  -- Diffrax reference: `test_saveat_solution_skip_vs_saveat`.
  let ts : Array Time := #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let stride : Nat := 2
  let solSkip := solveStepTo ts { steps := stride, t0 := true, t1 := false }
  let solTs := solveStepTo ts { ts := some #[0.0, 2.0, 4.0, 6.0], t1 := false }
  LeanTest.assertTrue (solSkip.result == Result.successful && solTs.result == Result.successful)
    "Step-skip and explicit-ts solves should both succeed"
  match solSkip.ts, solSkip.ys, solTs.ts, solTs.ys with
  | some tsSkip, some ysSkip, some tsExplicit, some ysExplicit =>
      LeanTest.assertTrue (tsSkip.size == tsExplicit.size)
        s!"skip-vs-ts parity: ts size mismatch {tsSkip.size}/{tsExplicit.size}"
      LeanTest.assertTrue (ysSkip.size == ysExplicit.size)
        s!"skip-vs-ts parity: ys size mismatch {ysSkip.size}/{ysExplicit.size}"
      for i in [:tsSkip.size] do
        LeanTest.assertTrue (approx tsSkip[i]! tsExplicit[i]! 1e-12)
          s!"skip-vs-ts parity: ts[{i}] mismatch {tsSkip[i]!} vs {tsExplicit[i]!}"
        LeanTest.assertTrue (approx ysSkip[i]! ysExplicit[i]! 1e-12)
          s!"skip-vs-ts parity: ys[{i}] mismatch {ysSkip[i]!} vs {ysExplicit[i]!}"
  | _, _, _, _ =>
      LeanTest.fail "skip-vs-ts parity: expected ts/ys outputs from both solves"

@[test] def testDegenerateIntervalRootSaveAtParity : IO Unit := do
  -- Diffrax reference: `test_saveat_solution.py::test_t0_eq_t1` (root SaveAt case).
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let saveat : SaveAt := {
    t0 := true
    t1 := true
    ts := some #[1.0, 1.0, 1.0]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 1.0 1.0 (some 0.1) (2.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Degenerate-interval root SaveAt solve should succeed"
  LeanTest.assertTrue (getStat "num_steps" sol.stats == 0)
    s!"Expected num_steps=0 when t0==t1, got {getStat "num_steps" sol.stats}"
  LeanTest.assertTrue (getStat "num_accepted_steps" sol.stats == 0)
    s!"Expected num_accepted_steps=0 when t0==t1, got {getStat "num_accepted_steps" sol.stats}"
  LeanTest.assertTrue (getStat "num_rejected_steps" sol.stats == 0)
    s!"Expected num_rejected_steps=0 when t0==t1, got {getStat "num_rejected_steps" sol.stats}"
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 5 && ys.size == 5)
        s!"Expected 5 degenerate-interval saves, got ts/ys sizes {ts.size}/{ys.size}"
      for i in [:ts.size] do
        LeanTest.assertTrue (approx ts[i]! 1.0 1e-12)
          s!"Expected ts[{i}] = 1.0 in degenerate solve, got {ts[i]!}"
        LeanTest.assertTrue (approx ys[i]! 2.0 1e-12)
          s!"Expected ys[{i}] = 2.0 in degenerate solve, got {ys[i]!}"
  | _, _ =>
      LeanTest.fail "Expected ts/ys output for degenerate-interval root SaveAt"

@[test] def testDegenerateIntervalSubSaveAtParity : IO Unit := do
  -- Diffrax reference: `test_saveat_solution.py::test_t0_eq_t1` (SubSaveAt case).
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let subA : SubSaveAt := { ts := some #[1.0, 1.0, 1.0], t1 := true }
  let subB : SubSaveAt := { t0 := true, ts := some #[1.0, 1.0, 1.0] }
  let subC : SubSaveAt := { t0 := true, ts := some #[1.0, 1.0, 1.0], steps := (1 : Nat) }
  let saveat : SaveAt := {
    t1 := false
    subs := #[subA, subB, subC]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 1.0 1.0 (some 0.1) (2.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Degenerate-interval SubSaveAt solve should succeed"
  LeanTest.assertTrue (getStat "num_steps" sol.stats == 0)
    s!"Expected num_steps=0 when t0==t1, got {getStat "num_steps" sol.stats}"
  LeanTest.assertTrue (getStat "num_accepted_steps" sol.stats == 0)
    s!"Expected num_accepted_steps=0 when t0==t1, got {getStat "num_accepted_steps" sol.stats}"
  LeanTest.assertTrue (getStat "num_rejected_steps" sol.stats == 0)
    s!"Expected num_rejected_steps=0 when t0==t1, got {getStat "num_rejected_steps" sol.stats}"
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 12 && ys.size == 12)
        s!"Expected 12 SubSaveAt degenerate saves, got ts/ys sizes {ts.size}/{ys.size}"
      for i in [:ts.size] do
        LeanTest.assertTrue (approx ts[i]! 1.0 1e-12)
          s!"Expected ts[{i}] = 1.0 in SubSaveAt degenerate solve, got {ts[i]!}"
        LeanTest.assertTrue (approx ys[i]! 2.0 1e-12)
          s!"Expected ys[{i}] = 2.0 in SubSaveAt degenerate solve, got {ys[i]!}"
  | _, _ =>
      LeanTest.fail "Expected ts/ys output for degenerate-interval SubSaveAt"

@[test] def testEmptyLeafSubSaveAtRejected : IO Unit := do
  -- Diffrax parity: constructing an empty SubSaveAt leaf is invalid.
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let emptyLeaf : SubSaveAt := {}
  let validLeaf : SubSaveAt := { ts := some #[0.25] }
  let saveat : SaveAt := {
    t1 := false
    subs := #[emptyLeaf, validLeaf]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.invalidInput)
    "Empty SubSaveAt leaf should be rejected with invalidInput"
  LeanTest.assertTrue (sol.ts.isNone && sol.ys.isNone)
    "Empty SubSaveAt leaf rejection should not produce saved ts/ys"

def run : IO Unit := do
  testSaveAtSubsIgnoresSyntheticRootPayload
  testNestedContainerSubSaveAtPayloadIgnored
  testReverseTimeNestedSubSaveAtUsesPerLeafTsMonotonicity
  testSaveAtStepsSkipStepToParity
  testSaveAtStepsSkipVsTsParity
  testDegenerateIntervalRootSaveAtParity
  testDegenerateIntervalSubSaveAtParity
  testEmptyLeafSubSaveAtRejected

end Tests.DiffEqSaveAtParity
