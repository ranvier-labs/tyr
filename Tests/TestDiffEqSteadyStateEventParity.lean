import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqSteadyStateEventParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

@[test] def testSteadyStateEventTerminatesEarlyAndSetsMask : IO Unit := do
  let term : ODETerm Float Unit := {
    vectorField := fun _t y _ => -y
  }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let steady : EventSpec Float Unit :=
    EventSpec.steadyState term solver (rtol := 1.0e-3) (atol := 1.0e-6)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 20.0 (some 0.05) (1.0 : Float) () (saveat := { t1 := true })
      (event := some steady)
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Steady-state event should terminate early."
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1 && ys.size == 1)
        s!"Expected one saved endpoint at event time, got ts={ts.size}, ys={ys.size}"
      let tHit := ts[0]!
      let yHit := ys[0]!
      LeanTest.assertTrue (tHit > 0.0 && tHit < 20.0)
        s!"Expected steady-state termination strictly inside interval, got t={tHit}"
      LeanTest.assertTrue (Float.abs yHit < 2.0e-3)
        s!"Expected state near steady-state threshold, got y={yHit}"
  | _, _ =>
      LeanTest.fail "Expected saved output at steady-state event time"
  match sol.eventMask, sol.eventMaskLast with
  | some mask, some lastMask =>
      LeanTest.assertTrue (mask.size == 1 && lastMask.size == 1)
        s!"Expected single-event masks, got sizes {mask.size} and {lastMask.size}"
      LeanTest.assertTrue (mask[0]! && lastMask[0]!)
        "Steady-state event should be recorded in eventMask and eventMaskLast"
  | _, _ =>
      LeanTest.fail "Expected event masks for triggered steady-state event"

@[test] def testSteadyStateEventNonTriggeringSolveSuccessful : IO Unit := do
  let term : ODETerm Float Unit := {
    vectorField := fun _t _y _ => 1.0
  }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let steady : EventSpec Float Unit :=
    EventSpec.steadyState term solver (rtol := 1.0e-3) (atol := 1.0e-6)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 2.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true })
      (event := some steady)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Non-steady dynamics should complete without triggering steady-state event."
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1 && ys.size == 1)
        s!"Expected one saved endpoint at final time, got ts={ts.size}, ys={ys.size}"
      LeanTest.assertTrue (approx ts[0]! 2.0 1e-12)
        s!"Expected final time 2.0, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 2.0 1e-12)
        s!"Expected final state 2.0, got {ys[0]!}"
  | _, _ =>
      LeanTest.fail "Expected saved final output for non-triggering steady-state solve"
  match sol.eventMask with
  | some mask =>
      LeanTest.assertTrue (mask.size == 1) s!"Expected single event-mask entry, got {mask.size}"
      LeanTest.assertTrue (!mask[0]!)
        "Steady-state mask should remain false when event never triggers"
  | none =>
      LeanTest.fail "Event mask should be present when an event is configured"
  LeanTest.assertTrue sol.eventMaskLast.isNone
    "Last-event mask should remain none when steady-state event never triggers"

def run : IO Unit := do
  testSteadyStateEventTerminatesEarlyAndSetsMask
  testSteadyStateEventNonTriggeringSolveSuccessful

end Tests.DiffEqSteadyStateEventParity
