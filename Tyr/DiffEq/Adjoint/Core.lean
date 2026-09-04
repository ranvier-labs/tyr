import Tyr.DiffEq.Integrate

namespace torch
namespace DiffEq

/-! ## Adjoint Support (Core)

Backend-agnostic adjoint utilities for ODEs.
Provide an `AdjointBackend` instance to supply VJPs.

Backsolve adjoints currently mirror Diffrax constraints:
- Only ODE terms and single-term solvers.
- No `SaveAt(steps := true)` or `SaveAt(dense := true)`.
- No events. SDE backsolve adjoints are supported only for single-noise
  Stratonovich SDEs, via the dedicated `sdeBacksolveAdjointStratonovich`
  path (the generic ODE backsolve still rejects multi-term drift+diffusion
  structures); the diffusion control must be replayable.

Direct/forward adjoints now differentiate the shared solver loop for
supported explicit-RK solvers by replaying accepted primal steps from the
loop trace. Unsupported solver/controller combinations now report explicit
unsupported errors instead of falling back to a whole-solve wrapper VJP.
-/

class AdjointBackend (Y Args : Type) where
  vjp : (Time → Y → Args → Y) → Time → Y → Args → Y → (Y × Y × Args)

class AdjointFnBackend (Y Args : Type) where
  vjpFn : (Y → Args → Y) → Y → Args → Y → (Y × Args)

structure AdjointState (Y Args : Type) where
  y : Y
  adjY : Y
  adjArgs : Args

instance [Inhabited Y] [Inhabited Args] : Inhabited (AdjointState Y Args) :=
  ⟨{ y := default, adjY := default, adjArgs := default }⟩

instance [DiffEqSpace Y] [DiffEqSpace Args] : DiffEqSpace (AdjointState Y Args) where
  add a b := {
    y := DiffEqSpace.add a.y b.y
    adjY := DiffEqSpace.add a.adjY b.adjY
    adjArgs := DiffEqSpace.add a.adjArgs b.adjArgs
  }
  sub a b := {
    y := DiffEqSpace.sub a.y b.y
    adjY := DiffEqSpace.sub a.adjY b.adjY
    adjArgs := DiffEqSpace.sub a.adjArgs b.adjArgs
  }
  scale s a := {
    y := DiffEqSpace.scale s a.y
    adjY := DiffEqSpace.scale s a.adjY
    adjArgs := DiffEqSpace.scale s a.adjArgs
  }

instance [DiffEqElem Y] [DiffEqElem Args] : DiffEqElem (AdjointState Y Args) where
  abs a := {
    y := DiffEqElem.abs a.y
    adjY := DiffEqElem.abs a.adjY
    adjArgs := DiffEqElem.abs a.adjArgs
  }
  max a b := {
    y := DiffEqElem.max a.y b.y
    adjY := DiffEqElem.max a.adjY b.adjY
    adjArgs := DiffEqElem.max a.adjArgs b.adjArgs
  }
  addScalar s a := {
    y := DiffEqElem.addScalar s a.y
    adjY := DiffEqElem.addScalar s a.adjY
    adjArgs := DiffEqElem.addScalar s a.adjArgs
  }
  div a b := {
    y := DiffEqElem.div a.y b.y
    adjY := DiffEqElem.div a.adjY b.adjY
    adjArgs := DiffEqElem.div a.adjArgs b.adjArgs
  }

instance [DiffEqSeminorm Y] [DiffEqSeminorm Args] : DiffEqSeminorm (AdjointState Y Args) where
  rms a :=
    let r1 := DiffEqSeminorm.rms a.y
    let r2 := DiffEqSeminorm.rms a.adjY
    let r3 := DiffEqSeminorm.rms a.adjArgs
    max r1 (max r2 r3)

structure AdjointResult (Y Args : Type) where
  adjY0 : Y
  adjArgs : Args

def backsolveAdjointSupported {Y Args : Type}
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (saveat : SaveAt) : Bool :=
  solver.termStructure == TermStructure.single &&
    !saveat.steps &&
    !saveat.dense &&
    !saveat.solverState &&
    !saveat.controllerState &&
    !saveat.madeJump &&
    saveat.subs.size == 0

private def saveAtTsOutOfRange (t0 t1 : Time) (saveat : SaveAt) : Bool :=
  match saveat.ts with
  | none => false
  | some ts =>
      if ts.size == 0 then
        false
      else
        let tmin := if t0 <= t1 then t0 else t1
        let tmax := if t0 <= t1 then t1 else t0
        ts.foldl (init := false) (fun acc t => acc || t < tmin || t > tmax)

def backsolveAdjointUnsupportedReason {Y Args : Type}
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (saveat : SaveAt) : Option String :=
  if saveAtTsOutOfRange t0 t1 saveat then
    some "Adjoint solve failed: `saveat.ts` contains values outside [t0, t1]."
  else if solver.termStructure != TermStructure.single then
    some "Adjoint solve failed: only single-term ODE solvers are supported; multi-term structures (for example SDE drift+diffusion) are not yet supported."
  else if saveat.steps then
    some "Adjoint solve failed: backsolve-style adjoints do not support `saveat.steps := true`."
  else if saveat.dense then
    some "Adjoint solve failed: backsolve-style adjoints do not support `saveat.dense := true`."
  else if saveat.solverState || saveat.controllerState || saveat.madeJump then
    some "Adjoint solve failed: backsolve-style adjoints do not support solver/controller/jump state payload saving."
  else if saveat.subs.size != 0 then
    some "Adjoint solve failed: backsolve-style adjoints do not yet support nested `SaveAt.subs` payload trees."
  else
    none

private def directLikeDt0CompatibilityUnsupportedReason
    (modeName : String) : String :=
  s!"Adjoint solve failed: `{modeName}.requireDt0 := true` preserves the older constant-step contract and therefore requires `dt0`."

private def directLikeDiscreteLoopRuntimeUnsupportedReason
    (modeName : String) : String :=
  s!"Adjoint solve failed: `{modeName}` could not construct a discrete loop adjoint for this solve. Rejected steps, event-localized restarts, or other unsupported runtime control-flow may have occurred."

private def mkAdjointInternalErrorSolution {Y SolverState ControllerState : Type}
    (t0 t1 : Time)
    (stats : List (String × Nat) := []) :
    Solution Y SolverState ControllerState := {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := stats
      result := Result.internalError
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := none
    }

private def mkAdjointFailure
    {Y SolverState ControllerState Args : Type}
    (t0 t1 : Time)
    (tag : String)
    (msg : String) :
    (Solution Y SolverState ControllerState × Option (AdjointResult Y Args) × Option String) :=
  let sol :=
    mkAdjointInternalErrorSolution (Y := Y) (SolverState := SolverState)
      (ControllerState := ControllerState) t0 t1
      [("adjoint_error", 1), (tag, 1)]
  (sol, none, some msg)

abbrev AdjointSolveWithReport (Y SolverState ControllerState Args : Type) :=
  (Solution Y SolverState ControllerState × Option (AdjointResult Y Args) × Option String)

class AcceptedStepReplayController (C : Type) where
  supportsAcceptedStepReplay : Bool

instance (priority := 1000) : AcceptedStepReplayController C where
  supportsAcceptedStepReplay := false

instance : AcceptedStepReplayController ConstantStepSize where
  supportsAcceptedStepReplay := true

instance : AcceptedStepReplayController StepTo where
  supportsAcceptedStepReplay := true

private def directLikeDiscreteLoopUnsupportedReason
    {Y Args Controller : Type}
    [AcceptedStepReplayController Controller]
    (modeName : String)
    (t0 t1 : Time)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args) : Option String :=
  if t0 == t1 then
    none
  else if !(inferInstance : AcceptedStepReplayController Controller).supportsAcceptedStepReplay then
    some
      s!"Adjoint solve failed: `{modeName}` currently requires a controller with a discrete loop adjoint implementation; this controller is not supported yet."
  else
    match solver.odeStepAdjoint? with
    | none =>
        some
          s!"Adjoint solve failed: `{modeName}` currently requires solver step-adjoint support; this solver does not provide it yet."
    | some _ => none

/-! ## Adjoint Method Modes -/

structure DirectAdjoint where
  /--
  Compatibility shim for the older constant-step-only direct-adjoint contract.
  Diffrax does not impose this extra mode-level `dt0` requirement.
  -/
  requireDt0 : Bool := false
  deriving Inhabited

structure RecursiveCheckpointAdjoint where
  /-- Number of primal steps per checkpoint chunk (`0` is normalized to `1`). -/
  checkpointEvery : Nat := 64
  /--
  Recompute each checkpoint chunk before backpropagating through it.
  When `false`, this mode reduces to regular full-trajectory backsolve.
  -/
  recomputeSegments : Bool := true
  deriving Inhabited

structure ForwardMode where
  /--
  Compatibility shim for the older constant-step-only forward-mode contract.
  Diffrax does not impose this extra mode-level `dt0` requirement.
  -/
  requireDt0 : Bool := false
  deriving Inhabited

structure ImplicitAdjoint where
  /--
  Use the current backsolve-based fallback for implicit adjoints.
  If `false`, this mode returns an explicit unsupported message.
  -/
  useBacksolveFallback : Bool := true
  /--
  Optional recursive-checkpoint configuration for implicit adjoints.
  This strategy still requires `useBacksolveFallback := true`.
  -/
  recursiveCheckpoint : Option RecursiveCheckpointAdjoint := none
  deriving Inhabited

/-! ## Backsolve Adjoint Wrapper -/

structure BacksolveAdjoint (Y Args : Type) where
  adjSolver :
    AbstractSolver (ODETerm (AdjointState Y Args) Args)
      (AdjointState Y Args) (AdjointState Y Args) Time Args

def adjointTerm
    [DiffEqSpace Y] [DiffEqSpace Args]
    [AdjointBackend Y Args]
    (term : ODETerm Y Args) : ODETerm (AdjointState Y Args) Args :=
  { vectorField := fun t state args =>
      let (f, vjpY, vjpArgs) := (AdjointBackend.vjp (Y := Y) (Args := Args))
        term.vectorField t state.y args state.adjY
      {
        y := f
        adjY := DiffEqSpace.scale (-1.0) vjpY
        adjArgs := DiffEqSpace.scale (-1.0) vjpArgs
      } }

def backsolveAdjoint
    {Y Args SolverState ControllerState : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (term : ODETerm Y Args)
    (adjSolver :
      AbstractSolver (ODETerm (AdjointState Y Args) Args)
        (AdjointState Y Args) (AdjointState Y Args) Time Args)
    (sol : Solution Y SolverState ControllerState)
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  if sol.result != Result.successful then
    none
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          none
        else if ts.size != ys.size then
          none
        else if Float.abs (ts[ts.size - 1]! - sol.t1) != 0.0 then
          none
        else
          let adjTerm := adjointTerm term
          let runSegment :=
            fun (tStart tStop : Time) (yStart : Y) (adjYStart : Y) (adjArgsStart : Args) =>
              let dtAbs := Float.abs (tStart - tStop)
              if dtAbs == 0.0 then
                some ({ y := yStart, adjY := adjYStart, adjArgs := adjArgsStart } : AdjointState Y Args)
              else
                let state0 : AdjointState Y Args :=
                  { y := yStart, adjY := adjYStart, adjArgs := adjArgsStart }
                let adjSol :=
                  diffeqsolve
                    (Term := ODETerm (AdjointState Y Args) Args)
                    (Y := AdjointState Y Args)
                    (VF := AdjointState Y Args)
                    (Control := Time)
                    (Args := Args)
                    (Controller := ConstantStepSize)
                    adjTerm adjSolver tStart tStop (some dtAbs) state0 args
                    (saveat := { t1 := true }) (maxSteps := 1)
                match adjSol.ys with
                | some ysAdj =>
                    if ysAdj.size == 0 then
                      none
                    else
                      some ysAdj[ysAdj.size - 1]!
                | none => none
          Id.run do
            let numSegs := ts.size - 1
            let mut adjY := adjY1
            let mut adjArgs := DiffEqSpace.scale 0.0 args
            let mut ok := true
            for offset in [:numSegs] do
              if ok then
                let i := ts.size - 1 - offset
                let t1 := ts[i]!
                let t0 := ts[i - 1]!
                let y1 := ys[i]!
                match runSegment t1 t0 y1 adjY adjArgs with
                | some state1 =>
                    adjY := state1.adjY
                    adjArgs := state1.adjArgs
                | none =>
                    ok := false
            if ok && Float.abs (ts[0]! - sol.t0) != 0.0 then
              match runSegment ts[0]! sol.t0 ys[0]! adjY adjArgs with
              | some state0 =>
                  adjY := state0.adjY
                  adjArgs := state0.adjArgs
              | none =>
                  ok := false
            if ok then
              return some { adjY0 := adjY, adjArgs := adjArgs }
            else
              return none
    | _, _ => none

private def explicitRKStepAdjoint
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (tableau : ExplicitRKAdjointTableau)
    (term : ODETerm Y Args)
    (t0 t1 : Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  let s := tableau.b.size
  if tableau.a.size != s || tableau.c.size != s then
    none
  else
    let dt := t1 - t0
    let zeroY := zeroLike y0
    let zeroArgs := zeroLike args
    let (stageYs, _stageKs) := Id.run do
      let mut ys : Array Y := #[]
      let mut ks : Array Y := #[]
      for i in [:s] do
        let row := tableau.a.getD i #[]
        let mut yi := y0
        for j in [:i] do
          let aij := row.getD j 0.0
          let kj := ks.getD j zeroY
          yi := DiffEqSpace.add yi (DiffEqSpace.scale aij kj)
        let ti := t0 + tableau.c.getD i 0.0 * dt
        let ki := DiffEqSpace.scale dt (term.vectorField ti yi args)
        ys := ys.push yi
        ks := ks.push ki
      return (ys, ks)
    Id.run do
      let mut adjY0 := adjY1
      let mut adjArgs := zeroArgs
      let mut adjKs : Array Y := Array.replicate s zeroY
      for i in [:s] do
        let bi := tableau.b.getD i 0.0
        adjKs := adjKs.set! i (DiffEqSpace.add adjKs[i]! (DiffEqSpace.scale bi adjY1))
      for offset in [:s] do
        let i := s - 1 - offset
        let ti := t0 + tableau.c[i]! * dt
        let yi := stageYs[i]!
        let adjKi := adjKs[i]!
        let adjF := DiffEqSpace.scale dt adjKi
        let (_, vjpY, vjpArgs) :=
          (AdjointBackend.vjp (Y := Y) (Args := Args))
            term.vectorField ti yi args adjF
        adjY0 := DiffEqSpace.add adjY0 vjpY
        adjArgs := DiffEqSpace.add adjArgs vjpArgs
        let row := tableau.a[i]!
        for j in [:i] do
          let aij := row.getD j 0.0
          adjKs := adjKs.set! j (DiffEqSpace.add adjKs[j]! (DiffEqSpace.scale aij vjpY))
      return some { adjY0 := adjY0, adjArgs := adjArgs }

private def explicitRKAdjointOverAcceptedAttempts
    {Y Args ControllerState : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (tableau : ExplicitRKAdjointTableau)
    (term : ODETerm Y Args)
    (attempts : Array (SolveLoopAttempt Y ControllerState))
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  let zeroArgs := zeroLike args
  Id.run do
    let mut adjY := adjY1
    let mut adjArgs := zeroArgs
    let mut failure := false
    for offset in [:attempts.size] do
      if !failure then
        let i := attempts.size - 1 - offset
        match attempts[i]? with
        | none =>
            failure := true
        | some attempt =>
            if attempt.accepted then
              match explicitRKStepAdjoint tableau term attempt.tStart attempt.tAfter attempt.yStart args adjY with
              | some stepAdj =>
                  adjY := stepAdj.adjY0
                  adjArgs := DiffEqSpace.add adjArgs stepAdj.adjArgs
              | none =>
                  failure := true
    if failure then
      return none
    else
      return some { adjY0 := adjY, adjArgs := adjArgs }

/-- Fixed-point iteration with an RMS-based stopping rule (same contraction
    style as the forward implicit stage solves). -/
private def dirkFixedPoint {Y : Type} [DiffEqSpace Y] [DiffEqSeminorm Y]
    (iterate : Y → Y) (init : Y) (maxIters : Nat) (rtol atol : Float) : Y := Id.run do
  let mut x := init
  for _ in [:maxIters] do
    let next := iterate x
    let delta := DiffEqSeminorm.rms (DiffEqSpace.sub next x)
    let scale := atol + rtol * DiffEqSeminorm.rms next
    x := next
    if delta <= scale then
      break
  return x

/-- Discrete adjoint of one diagonally-implicit RK step.

Forward stages satisfy `zᵢ = y₀ + Σ_{j≤i} aᵢⱼ kⱼ` with `kᵢ = h·f(tᵢ, zᵢ)`;
the reverse sweep solves the transposed stage equations
`μᵢ = bᵢ·λ + Σ_{j>i} aⱼᵢ·vⱼ + aᵢᵢ·vᵢ` where `vᵢ = h·Jᵢᵀ·μᵢ`, each by the
same fixed-point contraction as the forward solve, using only the term's
VJP (no explicit Jacobians or linear solves). With a strictly
lower-triangular tableau this degenerates exactly to the explicit RK
adjoint. -/
private def dirkStepAdjoint
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args] [DiffEqSeminorm Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (spec : DIRKAdjointSpec)
    (term : ODETerm Y Args)
    (t0 t1 : Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  let tableau := spec.tableau
  let s := tableau.b.size
  if tableau.a.size != s || tableau.c.size != s then
    none
  else
    let dt := t1 - t0
    let zeroY := zeroLike y0
    let zeroArgs := zeroLike args
    -- replay the primal stage states, solving diagonal stages by fixed point
    let stageYs : Array Y := Id.run do
      let mut ys : Array Y := #[]
      let mut ks : Array Y := #[]
      for i in [:s] do
        let row := tableau.a.getD i #[]
        let mut base := y0
        for j in [:i] do
          base := DiffEqSpace.add base
            (DiffEqSpace.scale (row.getD j 0.0) (ks.getD j zeroY))
        let ti := t0 + tableau.c.getD i 0.0 * dt
        let aii := row.getD i 0.0
        let yi :=
          if aii == 0.0 then
            base
          else
            dirkFixedPoint
              (fun y => DiffEqSpace.add base
                (DiffEqSpace.scale (aii * dt) (term.vectorField ti y args)))
              base spec.maxIters spec.rtol spec.atol
        let ki := DiffEqSpace.scale dt (term.vectorField ti yi args)
        ys := ys.push yi
        ks := ks.push ki
      return ys
    Id.run do
      let mut adjY0 := adjY1
      let mut adjArgs := zeroArgs
      let mut adjKs : Array Y := Array.replicate s zeroY
      for i in [:s] do
        adjKs := adjKs.set! i (DiffEqSpace.scale (tableau.b.getD i 0.0) adjY1)
      for offset in [:s] do
        let i := s - 1 - offset
        let ti := t0 + tableau.c[i]! * dt
        let yi := stageYs[i]!
        let row := tableau.a[i]!
        let aii := row.getD i 0.0
        let vjpAt := fun (mu : Y) =>
          (AdjointBackend.vjp (Y := Y) (Args := Args))
            term.vectorField ti yi args (DiffEqSpace.scale dt mu)
        -- solve μ = c + aᵢᵢ·(h Jᵀ μ) for the converged stage cotangent
        let ci := adjKs[i]!
        let mui :=
          if aii == 0.0 then
            ci
          else
            dirkFixedPoint
              (fun mu =>
                let (_, vY, _) := vjpAt mu
                DiffEqSpace.add ci (DiffEqSpace.scale aii vY))
              ci spec.maxIters spec.rtol spec.atol
        let (_, vY, vArgs) := vjpAt mui
        adjY0 := DiffEqSpace.add adjY0 vY
        adjArgs := DiffEqSpace.add adjArgs vArgs
        for j in [:i] do
          let aij := row.getD j 0.0
          adjKs := adjKs.set! j (DiffEqSpace.add adjKs[j]! (DiffEqSpace.scale aij vY))
      return some { adjY0 := adjY0, adjArgs := adjArgs }

private def dirkAdjointOverAcceptedAttempts
    {Y Args ControllerState : Type}
    [DiffEqSpace Y] [DiffEqSpace Args] [DiffEqSeminorm Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (spec : DIRKAdjointSpec)
    (term : ODETerm Y Args)
    (attempts : Array (SolveLoopAttempt Y ControllerState))
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  let zeroArgs := zeroLike args
  Id.run do
    let mut adjY := adjY1
    let mut adjArgs := zeroArgs
    let mut failure := false
    for offset in [:attempts.size] do
      if !failure then
        let i := attempts.size - 1 - offset
        match attempts[i]? with
        | none =>
            failure := true
        | some attempt =>
            if attempt.accepted then
              match dirkStepAdjoint spec term attempt.tStart attempt.tAfter attempt.yStart args adjY with
              | some stepAdj =>
                  adjY := stepAdj.adjY0
                  adjArgs := DiffEqSpace.add adjArgs stepAdj.adjArgs
              | none =>
                  failure := true
    if failure then
      return none
    else
      return some { adjY0 := adjY, adjArgs := adjArgs }

/-- Discrete adjoint of one IMEX additive-RK step (paired explicit and
diagonally-implicit tableaux sharing stage times).

Forward stages satisfy `zᵢ = y₀ + Σ_{j<i} aᴱᵢⱼ kᴱⱼ + Σ_{j≤i} aᴵᵢⱼ kᴵⱼ` with
`kᴱᵢ = h·fᴱ(tᵢ, zᵢ)` and `kᴵᵢ = h·fᴵ(tᵢ, zᵢ)`. The reverse sweep keeps
separate stage cotangents `μᴱᵢ = bᴱᵢλ + Σ_{j>i} aᴱⱼᵢ vⱼ` and
`μᴵᵢ = bᴵᵢλ + Σ_{j>i} aᴵⱼᵢ vⱼ + aᴵᵢᵢ vᵢ`, and solves
`vᵢ = h·Jᴱᵢᵀ μᴱᵢ + h·Jᴵᵢᵀ μᴵᵢ` by fixed-point iteration in `vᵢ` (only the
implicit cotangent depends on it). With `fᴱ = 0` this is the DIRK adjoint. -/
private def imexStepAdjoint
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args] [DiffEqSeminorm Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (spec : IMEXAdjointSpec)
    (explicitTerm implicitTerm : ODETerm Y Args)
    (t0 t1 : Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  let tE := spec.explicit
  let tI := spec.implicit
  let s := tE.b.size
  if tE.a.size != s || tE.c.size != s || tI.a.size != s || tI.b.size != s then
    none
  else
    let dt := t1 - t0
    let zeroY := zeroLike y0
    let zeroArgs := zeroLike args
    -- replay the primal stage states (diagonal stages by fixed point)
    let stageYs : Array Y := Id.run do
      let mut ys : Array Y := #[]
      let mut ksE : Array Y := #[]
      let mut ksI : Array Y := #[]
      for i in [:s] do
        let rowE := tE.a.getD i #[]
        let rowI := tI.a.getD i #[]
        let mut base := y0
        for j in [:i] do
          base := DiffEqSpace.add base
            (DiffEqSpace.scale (rowE.getD j 0.0) (ksE.getD j zeroY))
          base := DiffEqSpace.add base
            (DiffEqSpace.scale (rowI.getD j 0.0) (ksI.getD j zeroY))
        let ti := t0 + tI.c.getD i 0.0 * dt
        let aii := rowI.getD i 0.0
        let yi :=
          if aii == 0.0 then
            base
          else
            dirkFixedPoint
              (fun y => DiffEqSpace.add base
                (DiffEqSpace.scale (aii * dt) (implicitTerm.vectorField ti y args)))
              base spec.maxIters spec.rtol spec.atol
        ys := ys.push yi
        ksE := ksE.push (DiffEqSpace.scale dt (explicitTerm.vectorField ti yi args))
        ksI := ksI.push (DiffEqSpace.scale dt (implicitTerm.vectorField ti yi args))
      return ys
    Id.run do
      let mut adjY0 := adjY1
      let mut adjArgs := zeroArgs
      let mut cE : Array Y := Array.replicate s zeroY
      let mut cI : Array Y := Array.replicate s zeroY
      for i in [:s] do
        cE := cE.set! i (DiffEqSpace.scale (tE.b.getD i 0.0) adjY1)
        cI := cI.set! i (DiffEqSpace.scale (tI.b.getD i 0.0) adjY1)
      for offset in [:s] do
        let i := s - 1 - offset
        let ti := t0 + tI.c[i]! * dt
        let yi := stageYs[i]!
        let rowE := tE.a[i]!
        let rowI := tI.a[i]!
        let aii := rowI.getD i 0.0
        let vjpE := fun (mu : Y) =>
          (AdjointBackend.vjp (Y := Y) (Args := Args))
            explicitTerm.vectorField ti yi args (DiffEqSpace.scale dt mu)
        let vjpI := fun (mu : Y) =>
          (AdjointBackend.vjp (Y := Y) (Args := Args))
            implicitTerm.vectorField ti yi args (DiffEqSpace.scale dt mu)
        -- explicit contribution is constant in vᵢ; compute it once
        let (_, wE, gE) := vjpE cE[i]!
        let ciI := cI[i]!
        let vi :=
          if aii == 0.0 then
            let (_, wI, _) := vjpI ciI
            DiffEqSpace.add wE wI
          else
            dirkFixedPoint
              (fun v =>
                let (_, wI, _) := vjpI (DiffEqSpace.add ciI (DiffEqSpace.scale aii v))
                DiffEqSpace.add wE wI)
              (DiffEqSpace.add wE (let (_, wI, _) := vjpI ciI; wI))
              spec.maxIters spec.rtol spec.atol
        let muI := DiffEqSpace.add ciI (DiffEqSpace.scale aii vi)
        let (_, _, gI) := vjpI muI
        adjY0 := DiffEqSpace.add adjY0 vi
        adjArgs := DiffEqSpace.add adjArgs (DiffEqSpace.add gE gI)
        for j in [:i] do
          cE := cE.set! j (DiffEqSpace.add cE[j]! (DiffEqSpace.scale (rowE.getD j 0.0) vi))
          cI := cI.set! j (DiffEqSpace.add cI[j]! (DiffEqSpace.scale (rowI.getD j 0.0) vi))
      return some { adjY0 := adjY0, adjArgs := adjArgs }

private def imexAdjointOverAcceptedAttempts
    {Y Args ControllerState : Type}
    [DiffEqSpace Y] [DiffEqSpace Args] [DiffEqSeminorm Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (spec : IMEXAdjointSpec)
    (explicitTerm implicitTerm : ODETerm Y Args)
    (attempts : Array (SolveLoopAttempt Y ControllerState))
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  let zeroArgs := zeroLike args
  Id.run do
    let mut adjY := adjY1
    let mut adjArgs := zeroArgs
    let mut failure := false
    for offset in [:attempts.size] do
      if !failure then
        let i := attempts.size - 1 - offset
        match attempts[i]? with
        | none =>
            failure := true
        | some attempt =>
            if attempt.accepted then
              match imexStepAdjoint spec explicitTerm implicitTerm
                  attempt.tStart attempt.tAfter attempt.yStart args adjY with
              | some stepAdj =>
                  adjY := stepAdj.adjY0
                  adjArgs := DiffEqSpace.add adjArgs stepAdj.adjArgs
              | none =>
                  failure := true
    if failure then
      return none
    else
      return some { adjY0 := adjY, adjArgs := adjArgs }

private def runPrimalLoopTrace
    {Y Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    [StepSizeControllerValidation Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (maxSteps : Nat)
    (controller : Controller) :
    Option (SolveLoopState Y solver.SolverState (StepSizeController.State (C := Controller))) :=
  let controllerInvalid :=
    (inferInstance : StepSizeControllerValidation Controller).validate controller t0 t1 dt0
  if controllerInvalid.isSome || dt0 == some 0.0 || t0 == t1 then
    none
  else
    let errorOrder := solver.errorOrder term
    let initCtrlBase :=
      StepSizeController.init controller term t0 t1 y0 args dt0 solver.func errorOrder
    let dtAbs := if initCtrlBase.dt < 0.0 then -initCtrlBase.dt else initCtrlBase.dt
    let dt := if t1 >= t0 then dtAbs else -dtAbs
    let initCtrl : StepSizeState (StepSizeController.State (C := Controller)) :=
      { initCtrlBase with dt := dt }
    let initState := solver.init term t0 t1 y0 args
    let loopInit :=
      initialSolveLoopState t0 t1 y0 initState initCtrl false #[]
    some <|
      runSolveLoop term solver controller t1 args (some maxSteps) #[] errorOrder loopInit

private def directLikeDiscreteLoopAdjointWithController
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller]
    [StepSizeControllerValidation Controller]
    [AcceptedStepReplayController Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (maxSteps : Nat)
    (controller : Controller) :
    Option (AdjointResult Y Args) :=
  if t0 == t1 then
    some { adjY0 := adjY1, adjArgs := zeroLike args }
  else if !(inferInstance : AcceptedStepReplayController Controller).supportsAcceptedStepReplay then
    none
  else
    match solver.odeStepAdjoint? with
    | some (.explicitRK tableau) =>
        match runPrimalLoopTrace (Controller := Controller) term solver t0 t1 dt0 y0 args maxSteps controller with
        | some loopResult =>
            let hasRejected := loopResult.attempts.any (fun attempt => !attempt.accepted)
            if loopResult.result != Result.successful || hasRejected then
              none
            else
              explicitRKAdjointOverAcceptedAttempts tableau term loopResult.attempts args adjY1
        | none => none
    | some (.dirk spec) =>
        match runPrimalLoopTrace (Controller := Controller) term solver t0 t1 dt0 y0 args maxSteps controller with
        | some loopResult =>
            let hasRejected := loopResult.attempts.any (fun attempt => !attempt.accepted)
            if loopResult.result != Result.successful || hasRejected then
              none
            else
              dirkAdjointOverAcceptedAttempts spec term loopResult.attempts args adjY1
        | none => none
    -- IMEX specs only apply to MultiTerm solvers (`diffeqsolveDirectAdjointIMEX`)
    | some (.imexDirk _) => none
    | none => none

def diffeqsolveAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjSolver :
      AbstractSolver (ODETerm (AdjointState Y Args) Args)
        (AdjointState Y Args) (AdjointState Y Args) Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := Controller)) ×
      Option (AdjointResult Y Args)) := by
  let saveatTs :=
    match saveat.ts with
    | some ts => if ts.size == 0 then none else some ts
    | none => none
  let outOfRange :=
    match saveatTs with
    | none => false
    | some ts =>
        let tmin := if t0 <= t1 then t0 else t1
        let tmax := if t0 <= t1 then t1 else t0
        ts.foldl (init := false) (fun acc t => acc || t < tmin || t > tmax)
  if outOfRange || !backsolveAdjointSupported solver saveat then
    let sol : Solution Y solver.SolverState (StepSizeController.State (C := Controller)) := {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := []
      result := Result.internalError
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := none
    }
    exact (sol, none)
  else
    let saveatInternal := { saveat with steps := true, ts := none, dense := false }
    let solSteps :=
      diffeqsolve
        (Term := ODETerm Y Args)
        (Y := Y)
        (VF := Y)
        (Control := Time)
        (Args := Args)
        (Controller := Controller)
        term solver t0 t1 dt0 y0 args (saveat := saveatInternal) (maxSteps := maxSteps)
        (controller := controller)
    let adj := backsolveAdjoint term adjSolver solSteps args adjY1
    let sol :=
      match solSteps.ts, solSteps.ys with
      | some denseTs, some denseYs =>
          if denseTs.size == 0 || denseYs.size == 0 then
            { solSteps with ts := none, ys := none, interpolation := none }
          else
            let tf := solSteps.t1
            let yf := denseYs[denseYs.size - 1]!
            let denseInterp :=
              if saveat.dense || saveatTs.isSome then
                some (LinearInterpolation.toDense { ts := denseTs, ys := denseYs })
              else
                none
            let computeOutputs :=
              fun (tf : Time) (yf : Y) (denseTs : Array Time) (denseYs : Array Y)
                  (denseInterp : Option (DenseInterpolation Y)) =>
                match saveatTs with
                | some ts =>
                    let tsOut :=
                      if saveat.t0 || saveat.t1 then
                        let pre := if saveat.t0 then #[t0] else #[]
                        let post := if saveat.t1 then #[tf] else #[]
                        pre ++ ts ++ post
                      else
                        ts
                    let ysOut :=
                      match denseInterp with
                      | none => #[]
                      | some interp => tsOut.map (fun t => interp.evaluate t none true)
                    (some tsOut, some ysOut)
                | none =>
                    if saveat.steps then
                      (some denseTs, some denseYs)
                    else if saveat.t0 || saveat.t1 then
                      let tsOut :=
                        if saveat.t0 && saveat.t1 then #[t0, tf]
                        else if saveat.t0 then #[t0] else #[tf]
                      let ysOut :=
                        if saveat.t0 && saveat.t1 then #[y0, yf]
                        else if saveat.t0 then #[y0] else #[yf]
                      (some tsOut, some ysOut)
                    else
                      (none, none)
            let (tsOut, ysOut) := computeOutputs tf yf denseTs denseYs denseInterp
            let interpOut := if saveat.dense || saveatTs.isSome then denseInterp else none
            { solSteps with ts := tsOut, ys := ysOut, interpolation := interpOut }
      | _, _ => solSteps
    exact (sol, adj)

def diffeqsolveDirectAdjoint
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : ConstantStepSize := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := ConstantStepSize)) ×
      Option (AdjointResult Y Args)) := by
  let sol :=
    diffeqsolve
      (Term := ODETerm Y Args)
      (Y := Y)
      (VF := Y)
      (Control := Time)
      (Args := Args)
      (Controller := ConstantStepSize)
      term solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
      (controller := controller)
  if sol.result != Result.successful then
    exact (sol, none)
  else
    exact
      (sol,
        directLikeDiscreteLoopAdjointWithController
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 args adjY1 maxSteps controller)

/-! ## SDE backsolve adjoint (Stratonovich)

For `dy = f(t,y,θ) dt + g(t,y,θ) ∘ dW` and a loss on the terminal state,
the continuous adjoint satisfies (backward in time, with the SAME Brownian
path)

    da = −(∂f/∂y)ᵀ a dt − (∂g/∂y)ᵀ a ∘ dW
    dγ = −(∂f/∂θ)ᵀ a dt − (∂g/∂θ)ᵀ a ∘ dW

Stratonovich calculus is time-reversal symmetric, which is what makes the
backward solve meaningful (this is why backsolve SDE adjoints are
Stratonovich-only). The augmented (y, a, γ) system is integrated backward
with a Heun (Stratonovich) scheme; Brownian increments are re-queried from
the diffusion term's control, which must be backed by a replayable path
(e.g. a `VirtualBrownianTree`). Per step, the diffusion map with the
increment frozen, `y ↦ prod (g t y θ) c`, is an ordinary `Y → Y` map, so
both adjoint contractions come from the same `AdjointBackend.vjp` used by
the ODE adjoints. -/

/-- One backward Stratonovich-Heun step of the augmented adjoint system,
    from `tHi` down to `tLo` (so `h = tLo − tHi < 0`); `c` is the control
    increment queried over `(tHi, tLo)`, already orientation-correct. -/
private def sdeAdjointHeunStep
    {Y Args Control : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [AdjointBackend Y Args]
    (drift : ODETerm Y Args)
    (diffusion : ControlTerm Y Y Control Args)
    (tHi tLo : Time)
    (c : Control)
    (y a : Y) (γ : Args) (args : Args) :
    (Y × Y × Args) :=
  let h := tLo - tHi
  let diffMap := fun (t : Time) (y' : Y) (θ : Args) =>
    diffusion.prod (diffusion.vectorField t y' θ) c
  -- augmented vector field at (t, y, a): returns (dy-drift, da, dγ) pieces
  let eval := fun (t : Time) (y' a' : Y) =>
    let f := drift.vectorField t y' args
    let (_, jfA, gfA) :=
      (AdjointBackend.vjp (Y := Y) (Args := Args)) drift.vectorField t y' args a'
    let gInc := diffMap t y' args
    let (_, jgA, ggA) :=
      (AdjointBackend.vjp (Y := Y) (Args := Args)) diffMap t y' args a'
    -- (Δy, Δa, Δγ) over this step for drift·h and diffusion·c respectively
    ( DiffEqSpace.add (DiffEqSpace.scale h f) gInc
    , DiffEqSpace.scale (-1.0) (DiffEqSpace.add (DiffEqSpace.scale h jfA) jgA)
    , DiffEqSpace.scale (-1.0) (DiffEqSpace.add (DiffEqSpace.scale h gfA) ggA) )
  let (dy1, da1, dγ1) := eval tHi y a
  -- predictor
  let yP := DiffEqSpace.add y dy1
  let aP := DiffEqSpace.add a da1
  let (dy2, da2, dγ2) := eval tLo yP aP
  -- trapezoidal corrector
  ( DiffEqSpace.add y (DiffEqSpace.scale 0.5 (DiffEqSpace.add dy1 dy2))
  , DiffEqSpace.add a (DiffEqSpace.scale 0.5 (DiffEqSpace.add da1 da2))
  , DiffEqSpace.add γ (DiffEqSpace.scale 0.5 (DiffEqSpace.add dγ1 dγ2)) )

/-- Backsolve (continuous, optimise-then-discretise) adjoint for a
    single-noise Stratonovich SDE given as drift `ODETerm` plus diffusion
    `ControlTerm` (the standard `MultiTerm` SDE splitting, with `VF = Y`).

    Integrates the augmented `(y, a, γ)` system backward from `(yT, adjY1,
    0)` over `steps` uniform steps, re-querying the diffusion control over
    each backward subinterval — the control must be replayable. Gradients
    are with respect to the initial state and `args` (which may feed the
    drift, the diffusion, or both). -/
def sdeBacksolveAdjointStratonovich
    {Y Args Control : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (drift : ODETerm Y Args)
    (diffusion : ControlTerm Y Y Control Args)
    (t0 t1 : Time)
    (yT : Y)
    (args : Args)
    (adjY1 : Y)
    (steps : Nat := 256) :
    AdjointResult Y Args := Id.run do
  let n := Nat.max steps 1
  let dt := (t1 - t0) / Float.ofNat n
  let mut y := yT
  let mut a := adjY1
  let mut γ := zeroLike args
  for k in [:n] do
    let tHi := t1 - Float.ofNat k * dt
    let tLo := if k + 1 == n then t0 else t1 - Float.ofNat (k + 1) * dt
    let c := diffusion.control tHi tLo
    let (y', a', γ') := sdeAdjointHeunStep drift diffusion tHi tLo c y a γ args
    y := y'
    a := a'
    γ := γ'
  return { adjY0 := a, adjArgs := γ }

/-- IMEX twin of `runPrimalLoopTrace` for `MultiTerm` drift splittings. -/
private def runPrimalLoopTraceIMEX
    {Y Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    [StepSizeControllerValidation Controller]
    (terms : MultiTerm (ODETerm Y Args) (ODETerm Y Args))
    (solver : AbstractSolver (MultiTerm (ODETerm Y Args) (ODETerm Y Args)) Y (Y × Y) (Time × Time) Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (maxSteps : Nat)
    (controller : Controller) :
    Option (SolveLoopState Y solver.SolverState (StepSizeController.State (C := Controller))) :=
  let controllerInvalid :=
    (inferInstance : StepSizeControllerValidation Controller).validate controller t0 t1 dt0
  if controllerInvalid.isSome || dt0 == some 0.0 || t0 == t1 then
    none
  else
    let errorOrder := solver.errorOrder terms
    let initCtrlBase :=
      StepSizeController.init controller terms t0 t1 y0 args dt0 solver.func errorOrder
    let dtAbs := if initCtrlBase.dt < 0.0 then -initCtrlBase.dt else initCtrlBase.dt
    let dt := if t1 >= t0 then dtAbs else -dtAbs
    let initCtrl : StepSizeState (StepSizeController.State (C := Controller)) :=
      { initCtrlBase with dt := dt }
    let initState := solver.init terms t0 t1 y0 args
    let loopInit :=
      initialSolveLoopState t0 t1 y0 initState initCtrl false #[]
    some <|
      runSolveLoop terms solver controller t1 args (some maxSteps) #[] errorOrder loopInit

/-- Direct (discretise-then-optimise) adjoint for IMEX additive-RK solvers
    over a `MultiTerm` explicit/implicit drift splitting. Mirrors
    `diffeqsolveDirectAdjoint`: solves the primal, then replays accepted
    steps through the paired-tableau step adjoint. -/
def diffeqsolveDirectAdjointIMEX
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (terms : MultiTerm (ODETerm Y Args) (ODETerm Y Args))
    (solver : AbstractSolver (MultiTerm (ODETerm Y Args) (ODETerm Y Args)) Y (Y × Y) (Time × Time) Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : ConstantStepSize := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := ConstantStepSize)) ×
      Option (AdjointResult Y Args)) :=
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Y Args) (ODETerm Y Args))
      (Y := Y)
      (VF := (Y × Y))
      (Control := (Time × Time))
      (Args := Args)
      (Controller := ConstantStepSize)
      terms solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
      (controller := controller)
  if sol.result != Result.successful then
    (sol, none)
  else
    let adj :=
      if t0 == t1 then
        some { adjY0 := adjY1, adjArgs := zeroLike args }
      else
        match solver.odeStepAdjoint? with
        | some (.imexDirk spec) =>
            match runPrimalLoopTraceIMEX (Controller := ConstantStepSize)
                terms solver t0 t1 dt0 y0 args maxSteps controller with
            | some loopResult =>
                let hasRejected := loopResult.attempts.any (fun attempt => !attempt.accepted)
                if loopResult.result != Result.successful || hasRejected then
                  none
                else
                  imexAdjointOverAcceptedAttempts spec terms.term1 terms.term2
                    loopResult.attempts args adjY1
            | none => none
        | _ => none
    (sol, adj)

private def terminalSolveValue
    {Y Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (maxSteps : Nat)
    (controller : Controller) : Y :=
  let sol :=
    diffeqsolve
      (Term := ODETerm Y Args)
      (Y := Y)
      (VF := Y)
      (Control := Time)
      (Args := Args)
      (Controller := Controller)
      term solver t0 t1 dt0 y0 args (saveat := { t1 := true }) (maxSteps := maxSteps)
      (controller := controller)
  match sol.ys with
  | some ys =>
      if ys.size == 0 then y0 else ys[ys.size - 1]!
  | none => y0

private def diffeqsolveSolveFnAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [AdjointFnBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [AcceptedStepReplayController Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := Controller)) ×
      Option (AdjointResult Y Args)) := by
  let sol :=
    diffeqsolve
      (Term := ODETerm Y Args)
      (Y := Y)
      (VF := Y)
      (Control := Time)
      (Args := Args)
      (Controller := Controller)
      term solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
      (controller := controller)
  if sol.result != Result.successful then
    exact (sol, none)
  else
    match directLikeDiscreteLoopAdjointWithController
        (Controller := Controller)
        term solver t0 t1 dt0 y0 args adjY1 maxSteps controller with
    | some adj =>
        exact (sol, some adj)
    | none =>
        let solveFn := fun (y : Y) (args : Args) =>
          terminalSolveValue
            (Controller := Controller)
            term solver t0 t1 dt0 y args maxSteps controller
        let (adjY0, adjArgs) :=
          (AdjointFnBackend.vjpFn (Y := Y) (Args := Args)) solveFn y0 args adjY1
        exact (sol, some { adjY0 := adjY0, adjArgs := adjArgs })

def diffeqsolveBacksolveAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjoint : BacksolveAdjoint Y Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := Controller)) ×
      Option (AdjointResult Y Args)) :=
  diffeqsolveAdjoint term solver adjoint.adjSolver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller

def recursiveCheckpointChunkSize (mode : RecursiveCheckpointAdjoint) : Nat :=
  if mode.checkpointEvery == 0 then 1 else mode.checkpointEvery

def recursiveCheckpointAdjointUnsupportedReason
    (mode : RecursiveCheckpointAdjoint) : Option String :=
  let _ := mode
  none

private def recursiveCheckpointGrid
    {Y : Type}
    [Inhabited Y]
    (ts : Array Time)
    (ys : Array Y)
    (chunkSize : Nat) : Option (Array Time × Array Y) :=
  if ts.size == 0 || ys.size == 0 || ts.size != ys.size then
    none
  else
    let stride := if chunkSize == 0 then 1 else chunkSize
    let last := ts.size - 1
    Id.run do
      let mut chkTs : Array Time := #[ts[0]!]
      let mut chkYs : Array Y := #[ys[0]!]
      for i in [:ts.size] do
        if i != 0 then
          if i % stride == 0 || i == last then
            chkTs := chkTs.push ts[i]!
            chkYs := chkYs.push ys[i]!
      return some (chkTs, chkYs)

private def recursiveCheckpointBackprop
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjSolver :
      AbstractSolver (ODETerm (AdjointState Y Args) Args)
        (AdjointState Y Args) (AdjointState Y Args) Time Args)
    (checkTs : Array Time)
    (checkYs : Array Y)
    (dt0 : Option Time)
    (args : Args)
    (adjY1 : Y)
    (maxSteps : Nat)
    (controller : Controller) :
    (Option (AdjointResult Y Args) × Option String) :=
  if checkTs.size == 0 || checkYs.size == 0 || checkTs.size != checkYs.size then
    (none, some "Adjoint solve failed: recursive checkpoint grid is empty or malformed.")
  else
    Id.run do
      let mut adjY := adjY1
      let mut adjArgs := DiffEqSpace.scale 0.0 args
      let mut failure : Option String := none
      let numSegs := checkTs.size - 1
      for offset in [:numSegs] do
        if failure.isNone then
          let i := checkTs.size - 1 - offset
          let segT0 := checkTs[i - 1]!
          let segT1 := checkTs[i]!
          let segY0 := checkYs[i - 1]!
          let segSol :=
            diffeqsolve
              (Term := ODETerm Y Args)
              (Y := Y)
              (VF := Y)
              (Control := Time)
              (Args := Args)
              (Controller := Controller)
              term solver segT0 segT1 dt0 segY0 args (saveat := { t0 := true, steps := true })
              (maxSteps := maxSteps)
              (controller := controller)
          if segSol.result != Result.successful then
            failure :=
              some
                "Adjoint solve failed: recursive checkpoint segment recomputation did not finish successfully."
          else
            match backsolveAdjoint term adjSolver segSol args adjY with
            | none =>
                failure :=
                  some
                    "Adjoint solve failed: recursive checkpoint segment backsolve did not produce an adjoint state."
            | some segAdj =>
                adjY := segAdj.adjY0
                adjArgs := DiffEqSpace.add adjArgs segAdj.adjArgs
      match failure with
      | some msg => return (none, some msg)
      | none => return (some { adjY0 := adjY, adjArgs := adjArgs }, none)

private def directLikeModeUnsupportedReason
    (modeName : String)
    (requireDt0 : Bool)
    (dt0 : Option Time) : Option String :=
  if requireDt0 && dt0.isNone then
    some (directLikeDt0CompatibilityUnsupportedReason modeName)
  else
    none

def implicitAdjointUnsupportedReason
    (mode : ImplicitAdjoint) : Option String :=
  if mode.useBacksolveFallback then
    none
  else
    match mode.recursiveCheckpoint with
    | some _ =>
        some
          "Adjoint solve failed: `ImplicitAdjoint.recursiveCheckpoint` requires `useBacksolveFallback := true`."
    | none =>
        none

private def implicitAdjointSaveAtContractHolds (saveat : SaveAt) : Bool :=
  let hasTs :=
    match saveat.ts with
    | some ts => ts.size != 0
    | none => false
  saveat.t1 &&
    !saveat.t0 &&
    !hasTs &&
    !saveat.steps.enabled &&
    !saveat.dense &&
    !saveat.solverState &&
    !saveat.controllerState &&
    !saveat.madeJump &&
    saveat.subs.size == 0

def implicitAdjointSaveAtUnsupportedReason
    (mode : ImplicitAdjoint) (saveat : SaveAt) : Option String :=
  if !mode.useBacksolveFallback then
    none
  else
    let hasTs :=
      match saveat.ts with
      | some ts => ts.size != 0
      | none => false
    if implicitAdjointSaveAtContractHolds saveat then
      none
    else
      some
        s!"Adjoint solve failed: can only use `ImplicitAdjoint` with `saveat.t1 := true` and all other save fields disabled (Diffrax parity: `saveat=SaveAt(t1=True)`). Got `t0={saveat.t0}`, `t1={saveat.t1}`, `ts?={hasTs}`, `steps?={saveat.steps.enabled}`, `dense={saveat.dense}`, `solverState={saveat.solverState}`, `controllerState={saveat.controllerState}`, `madeJump={saveat.madeJump}`, `subs={saveat.subs.size}`."

private def diffeqsolveDirectAdjointWithReportCore
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [AcceptedStepReplayController Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default)
    (modeName : String)
    (unsupportedTag : String)
    (unsupportedReason : Option String) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  let unsupportedReason :=
    match unsupportedReason with
    | some msg => some msg
    | none =>
        directLikeDiscreteLoopUnsupportedReason
          (Controller := Controller) modeName t0 t1 solver
  match unsupportedReason with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 unsupportedTag msg
  | none =>
      let sol :=
        diffeqsolve
          (Controller := Controller)
          (Term := ODETerm Y Args)
          (Y := Y)
          (VF := Y)
          (Control := Time)
          (Args := Args)
          term solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
          (controller := controller)
      if sol.result != Result.successful then
        exact (sol, none, none)
      else
        match directLikeDiscreteLoopAdjointWithController
            (Controller := Controller)
            term solver t0 t1 dt0 y0 args adjY1 maxSteps controller with
        | some adj =>
            exact (sol, some adj, none)
        | none =>
            exact
              mkAdjointFailure
                (Y := Y)
                (SolverState := solver.SolverState)
                (ControllerState := StepSizeController.State (C := Controller))
                (Args := Args)
                t0 t1 unsupportedTag
                (directLikeDiscreteLoopRuntimeUnsupportedReason modeName)

private def diffeqsolveDirectAdjointConstantWithReportCore
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : ConstantStepSize := default)
    (modeName : String)
    (unsupportedTag : String)
    (unsupportedReason : Option String) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) Args := by
  let unsupportedReason :=
    match unsupportedReason with
    | some msg => some msg
    | none =>
        directLikeDiscreteLoopUnsupportedReason
          (Controller := ConstantStepSize) modeName t0 t1 solver
  match unsupportedReason with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := ConstantStepSize))
          (Args := Args)
          t0 t1 unsupportedTag msg
  | none =>
      let (sol, adj) :=
        diffeqsolveDirectAdjoint
          term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
      if sol.result != Result.successful then
        exact (sol, none, none)
      else
        match adj with
        | some adj =>
            exact (sol, some adj, none)
        | none =>
            exact
              mkAdjointFailure
                (Y := Y)
                (SolverState := solver.SolverState)
                (ControllerState := StepSizeController.State (C := ConstantStepSize))
                (Args := Args)
                t0 t1 unsupportedTag
                (directLikeDiscreteLoopRuntimeUnsupportedReason modeName)

private def diffeqsolveBacksolveAdjointWithReportCore
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjoint : BacksolveAdjoint Y Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default)
    (unsupportedTag : String)
    (unsupportedReason : Option String) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  match unsupportedReason with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 unsupportedTag msg
  | none =>
      match backsolveAdjointUnsupportedReason solver t0 t1 saveat with
      | some msg =>
          exact
            mkAdjointFailure
              (Y := Y)
              (SolverState := solver.SolverState)
              (ControllerState := StepSizeController.State (C := Controller))
              (Args := Args)
              t0 t1 "unsupported_backsolve_adjoint" msg
      | none =>
          let (sol, adj) :=
            diffeqsolveBacksolveAdjoint
              (Controller := Controller)
              term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
          exact (sol, adj, none)

def diffeqsolveDirectAdjointModeWithController
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [AcceptedStepReplayController Controller]
    [Inhabited Controller]
    (mode : DirectAdjoint := {})
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args :=
  let unsupportedReason :=
    directLikeModeUnsupportedReason "DirectAdjoint" mode.requireDt0 dt0
  diffeqsolveDirectAdjointWithReportCore
    (Controller := Controller)
    term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
    "DirectAdjoint"
    "unsupported_direct_adjoint_mode" unsupportedReason

def diffeqsolveDirectAdjointMode
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (mode : DirectAdjoint := {})
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : ConstantStepSize := default) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) Args :=
  let unsupportedReason :=
    directLikeModeUnsupportedReason "DirectAdjoint" mode.requireDt0 dt0
  diffeqsolveDirectAdjointConstantWithReportCore
    term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
    "DirectAdjoint"
    "unsupported_direct_adjoint_mode" unsupportedReason

def diffeqsolveForwardModeWithController
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [AcceptedStepReplayController Controller]
    [Inhabited Controller]
    (mode : ForwardMode := {})
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args :=
  let unsupportedReason :=
    directLikeModeUnsupportedReason "ForwardMode" mode.requireDt0 dt0
  diffeqsolveDirectAdjointWithReportCore
    (Controller := Controller)
    term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
    "ForwardMode"
    "unsupported_forward_mode" unsupportedReason

def diffeqsolveForwardMode
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (mode : ForwardMode := {})
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : ConstantStepSize := default) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) Args :=
  let unsupportedReason :=
    directLikeModeUnsupportedReason "ForwardMode" mode.requireDt0 dt0
  diffeqsolveDirectAdjointConstantWithReportCore
    term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
    "ForwardMode"
    "unsupported_forward_mode" unsupportedReason

def diffeqsolveRecursiveCheckpointAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (mode : RecursiveCheckpointAdjoint := {})
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjoint : BacksolveAdjoint Y Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  match recursiveCheckpointAdjointUnsupportedReason mode with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 "unsupported_recursive_checkpoint_adjoint" msg
  | none =>
      match backsolveAdjointUnsupportedReason solver t0 t1 saveat with
      | some msg =>
          exact
            mkAdjointFailure
              (Y := Y)
              (SolverState := solver.SolverState)
              (ControllerState := StepSizeController.State (C := Controller))
              (Args := Args)
              t0 t1 "unsupported_recursive_checkpoint_backsolve_contract" msg
      | none =>
          if !mode.recomputeSegments then
            let (sol, adjRes) :=
              diffeqsolveBacksolveAdjoint
                (Controller := Controller)
                term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
            exact (sol, adjRes, none)
          else
            let solPrimal :=
              diffeqsolve
                (Term := ODETerm Y Args)
                (Y := Y)
                (VF := Y)
                (Control := Time)
                (Args := Args)
                (Controller := Controller)
                term solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
                (controller := controller)
            if solPrimal.result != Result.successful then
              exact
                (solPrimal, none,
                  some
                    "Adjoint solve failed: primal solve did not finish successfully before recursive checkpoint backpropagation.")
            else
              let saveatInternal := { saveat with t0 := true, steps := true, ts := none, dense := false }
              let solSteps :=
                diffeqsolve
                  (Term := ODETerm Y Args)
                  (Y := Y)
                  (VF := Y)
                  (Control := Time)
                  (Args := Args)
                  (Controller := Controller)
                  term solver t0 t1 dt0 y0 args (saveat := saveatInternal)
                  (maxSteps := maxSteps)
                  (controller := controller)
              if solSteps.result != Result.successful then
                exact
                  (solPrimal, none,
                    some
                      "Adjoint solve failed: could not build recursive checkpoint trajectory because the internal step solve did not finish successfully.")
              else
                match solSteps.ts, solSteps.ys with
                | some ts, some ys =>
                    match recursiveCheckpointGrid ts ys (recursiveCheckpointChunkSize mode) with
                    | none =>
                        exact
                          (solPrimal, none,
                            some
                              "Adjoint solve failed: recursive checkpoint grid construction failed.")
                    | some (chkTs, chkYs) =>
                        let (adjRes, backpropFailure) :=
                          recursiveCheckpointBackprop
                            (Controller := Controller)
                            term solver adjoint.adjSolver chkTs chkYs dt0 args adjY1 maxSteps
                            controller
                        match backpropFailure with
                        | some msg => exact (solPrimal, none, some msg)
                        | none => exact (solPrimal, adjRes, none)
                | _, _ =>
                    exact
                      (solPrimal, none,
                        some
                          "Adjoint solve failed: internal recursive checkpoint step solve did not return `(ts, ys)`.")

def diffeqsolveImplicitAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [AdjointFnBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (mode : ImplicitAdjoint := {})
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjoint : BacksolveAdjoint Y Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  match implicitAdjointUnsupportedReason mode with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 "unsupported_implicit_adjoint" msg
  | none =>
      match implicitAdjointSaveAtUnsupportedReason mode saveat with
      | some msg =>
          exact
            mkAdjointFailure
              (Y := Y)
              (SolverState := solver.SolverState)
              (ControllerState := StepSizeController.State (C := Controller))
              (Args := Args)
              t0 t1 "unsupported_implicit_adjoint_saveat" msg
      | none =>
          if !mode.useBacksolveFallback then
            let (sol, adjRes) :=
              diffeqsolveSolveFnAdjoint
                (Controller := Controller)
                term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
            exact (sol, adjRes, none)
          else
            match mode.recursiveCheckpoint with
            | some recursiveMode =>
                exact
                  diffeqsolveRecursiveCheckpointAdjoint
                    (Controller := Controller)
                    recursiveMode term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps
                    controller
            | none =>
                exact
                  diffeqsolveBacksolveAdjointWithReportCore
                    (Controller := Controller)
                    term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
                    "unsupported_implicit_adjoint"
                    none

end DiffEq
end torch
