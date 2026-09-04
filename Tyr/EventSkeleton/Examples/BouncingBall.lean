import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Trace

/-!
# Drake Bouncing Ball Event-Skeleton Example

This is a solver-backed port of `../drake/examples/bouncing_ball`.
The continuous dynamics and witness event follow Drake's example:

* state `(q, v)` is height and vertical velocity,
* `qdot = v`, `vdot = -9.81`,
* the witness guard is `q = 0` with positive-to-nonpositive direction,
* the unrestricted update is the Newtonian reset `v+ = -e v-`.

The EventSkeleton layer records the localized ODE intervals and impact
saltation events around the executable physics.
-/

namespace Tyr.EventSkeleton.Examples.BouncingBall

open Tyr.EventSkeleton
open torch.DiffEq

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/bouncing_ball/bouncing_ball.h"
      concept := "declares continuous state, positive-to-nonpositive signed-distance witness, and restitution reset"
    },
    {
      path := "../drake/examples/bouncing_ball/bouncing_ball.cc"
      concept := "instantiates BouncingBall on Drake's default scalar set"
    },
    {
      path := "../drake/examples/bouncing_ball/test/bouncing_ball_test.cc"
      concept := "checks closed-form motion over repeated elastic impacts"
    },
    {
      path := "../drake/systems/analysis/simulator.cc"
      concept := "isolates witness events before applying unrestricted updates"
    }
  ]

def stateCoordinateNames : Array String :=
  #["q", "v"]

structure BouncingBallSystemBoundary where
  systemName : String := "BouncingBall"
  headerPath : String := "../drake/examples/bouncing_ball/bouncing_ball.h"
  implementationPath : String := "../drake/examples/bouncing_ball/bouncing_ball.cc"
  outputPortName : String := "y0"
  continuousStateCount : Nat := 2
  positionCount : Nat := 1
  velocityCount : Nat := 1
  miscCount : Nat := 0
  coordinateNames : Array String := stateCoordinateNames
  witnessName : String := "Signed distance"
  witnessDirection : String := "kPositiveThenNonPositive"
  triggerType : String := "kWitness"
  unrestrictedUpdateEvent : Bool := true
  defaultState : Array Float := #[10.0, 0.0]
  gravitationalAcceleration : Float := -9.81
  restitutionCoef : Float := 1.0
  scalarConversionOnDefaultScalars : Bool := true
  deriving Repr, Inhabited

namespace BouncingBallSystemBoundary

def validate? (boundary : BouncingBallSystemBoundary) : Except String Unit := do
  if boundary.systemName != "BouncingBall" then
    .error s!"BouncingBall system name mismatch: {boundary.systemName}"
  if boundary.headerPath != "../drake/examples/bouncing_ball/bouncing_ball.h" then
    .error s!"BouncingBall header path mismatch: {boundary.headerPath}"
  if boundary.implementationPath != "../drake/examples/bouncing_ball/bouncing_ball.cc" then
    .error s!"BouncingBall implementation path mismatch: {boundary.implementationPath}"
  if boundary.outputPortName != "y0" then
    .error s!"BouncingBall output port should be y0, got {boundary.outputPortName}"
  if boundary.continuousStateCount != 2 ||
      boundary.positionCount != 1 ||
      boundary.velocityCount != 1 ||
      boundary.miscCount != 0 then
    .error s!"BouncingBall continuous state split should be q=1, v=1, z=0 over two states, got total={boundary.continuousStateCount}, q={boundary.positionCount}, v={boundary.velocityCount}, z={boundary.miscCount}"
  if boundary.positionCount + boundary.velocityCount + boundary.miscCount !=
      boundary.continuousStateCount then
    .error "BouncingBall state split does not sum to the continuous state count"
  if boundary.coordinateNames != stateCoordinateNames then
    .error s!"BouncingBall coordinate names should be {stateCoordinateNames}, got {boundary.coordinateNames}"
  if boundary.witnessName != "Signed distance" then
    .error s!"BouncingBall witness name mismatch: {boundary.witnessName}"
  if boundary.witnessDirection != "kPositiveThenNonPositive" then
    .error s!"BouncingBall witness direction mismatch: {boundary.witnessDirection}"
  if boundary.triggerType != "kWitness" then
    .error s!"BouncingBall witness trigger should be kWitness, got {boundary.triggerType}"
  if !boundary.unrestrictedUpdateEvent then
    .error "BouncingBall witness should be attached to an unrestricted update event"
  if boundary.defaultState != #[10.0, 0.0] then
    .error s!"BouncingBall default state should be #[10.0, 0.0], got {boundary.defaultState}"
  if boundary.gravitationalAcceleration != -9.81 then
    .error s!"BouncingBall gravitational acceleration should be -9.81, got {boundary.gravitationalAcceleration}"
  if boundary.restitutionCoef != 1.0 then
    .error s!"BouncingBall restitution coefficient should be 1.0, got {boundary.restitutionCoef}"
  if !boundary.scalarConversionOnDefaultScalars then
    .error "BouncingBall implementation should instantiate Drake's default scalar set"

end BouncingBallSystemBoundary

def systemBoundary : BouncingBallSystemBoundary := {}

structure BallParams where
  gravity : Float := -9.81
  restitution : Float := 1.0
  initialHeight : Float := 10.0
  initialVelocity : Float := 0.0
  rootTol : Float := 1.0e-7
  stepSize : Float := 1.0e-2
  deriving Repr, Inhabited

def params : BallParams := {}

namespace BallParams

def validate? (p : BallParams) : Except String Unit := do
  if !p.gravity.isFinite then
    .error s!"BouncingBall gravity must be finite, got {p.gravity}"
  if !p.restitution.isFinite || p.restitution < 0.0 then
    .error s!"BouncingBall restitution must be finite and nonnegative, got {p.restitution}"
  if !p.initialHeight.isFinite || p.initialHeight < 0.0 then
    .error s!"BouncingBall initial height must be finite and nonnegative, got {p.initialHeight}"
  if !p.initialVelocity.isFinite then
    .error s!"BouncingBall initial velocity must be finite, got {p.initialVelocity}"
  if !p.rootTol.isFinite || p.rootTol <= 0.0 then
    .error s!"BouncingBall root tolerance must be positive and finite, got {p.rootTol}"
  if !p.stepSize.isFinite || p.stepSize <= 0.0 then
    .error s!"BouncingBall step size must be positive and finite, got {p.stepSize}"

end BallParams

structure BallState where
  height : Float
  velocity : Float
  deriving Repr, Inhabited

namespace BallState

def isFinite (x : BallState) : Bool :=
  x.height.isFinite && x.velocity.isFinite

end BallState

instance : DiffEqSpace BallState where
  add a b := { height := a.height + b.height, velocity := a.velocity + b.velocity }
  sub a b := { height := a.height - b.height, velocity := a.velocity - b.velocity }
  scale s x := { height := s * x.height, velocity := s * x.velocity }

instance : DiffEqSeminorm BallState where
  rms x := max (Float.abs x.height) (Float.abs x.velocity)

instance : DiffEqElem BallState where
  abs x := { height := Float.abs x.height, velocity := Float.abs x.velocity }
  max a b := { height := max a.height b.height, velocity := max a.velocity b.velocity }
  addScalar s x := { height := x.height + s, velocity := x.velocity + s }
  div a b := { height := a.height / b.height, velocity := a.velocity / b.velocity }

def initialState (p : BallParams := params) : BallState :=
  { height := p.initialHeight, velocity := p.initialVelocity }

def stateAsArray (x : BallState) : Array Float :=
  #[x.height, x.velocity]

def stateFromArray? (xs : Array Float) : Except String BallState := do
  if xs.size != 2 then
    .error s!"BouncingBall state vector must have two entries, got {xs.size}"
  if !(xs[0]!).isFinite || !(xs[1]!).isFinite then
    .error s!"BouncingBall state entries must be finite, got {xs}"
  pure { height := xs[0]!, velocity := xs[1]! }

def derivative (p : BallParams) (x : BallState) : BallState :=
  { height := x.velocity, velocity := p.gravity }

def vectorFieldArray (p : BallParams) (x : BallState) : Array Float :=
  stateAsArray (derivative p x)

def freeFlightFullPhysicsPrimitives
    (p : BallParams := params)
    (x : BallState := initialState p)
    (label : String := "bouncing-ball free-flight primitive physics") :
    FullPhysicsPrimitives :=
  {
    massMatrix := #[#[1.0]]
    qdot := #[x.velocity]
    actuationForces := #[0.0]
    biasForces := #[-p.gravity]
    label := label
  }

def validateFullPhysicsInputs?
    (p : BallParams) (x : BallState) : Except String Unit := do
  p.validate?
  if !x.isFinite then
    .error "BouncingBall full physics state must have finite height and velocity"

def fullPhysicsPrimitiveProvider
    (p : BallParams := params)
    (label : String := "bouncing-ball free-flight full physics provider") :
    FullPhysicsPrimitiveProvider BallState :=
  {
    label := label
    primitivesAt? := fun x => do
      validateFullPhysicsInputs? p x
      pure (freeFlightFullPhysicsPrimitives p x label)
  }

def solveFreeFlightPrimitivePhysics?
    (p : BallParams := params)
    (x : BallState := initialState p)
    (intervalVertex : VertexId := 0)
    (label : String := "bouncing-ball free-flight primitive physics") :
    Except String FullPhysicsResult := do
  validateFullPhysicsInputs? p x
  (freeFlightFullPhysicsPrimitives p x label).solve? intervalVertex

def odeTerm (p : BallParams) : ODETerm BallState Unit :=
  { vectorField := fun _t x _ => derivative p x }

def signedDistance (_p : BallParams) (x : BallState) : Float :=
  x.height

def impactEvent (p : BallParams) : EventSpec BallState Unit :=
  {
    condition := .real (fun _t x _ => signedDistance p x)
    direction := some false
    terminate := true
    rootTol := p.rootTol
  }

def resetState (p : BallParams) (x : BallState) : BallState :=
  { height := 0.0, velocity := -p.restitution * x.velocity }

def resetJacobian (p : BallParams) : Array (Array Float) :=
  #[#[1.0, 0.0], #[0.0, -p.restitution]]

def resetTheta (x : BallState) : Array (Array Float) :=
  #[#[0.0], #[-x.velocity]]

def guardGrad : Array Float :=
  #[1.0, 0.0]

def impactSaltationData (p : BallParams) (pre : BallState) : SaltationData :=
  SaltationData.mkFromFields
    (resetJacobian p)
    guardGrad
    (vectorFieldArray p pre)
    (vectorFieldArray p (resetState p pre))
    (resetTheta := resetTheta pre)

def dropTimeFromRest (p : BallParams) (height : Float) : Float :=
  Float.sqrt (-2.0 * height / p.gravity)

def closedFormUnitRestitutionFromRest
    (p : BallParams)
    (height : Float)
    (time : Float) : BallState := Id.run do
  let phaseTime := dropTimeFromRest p height
  let impactVelocity := p.gravity * phaseTime
  let mut localTime := time
  let mut falling := true
  for _ in [:1000] do
    if localTime >= phaseTime then
      localTime := localTime - phaseTime
      falling := !falling
  if falling then
    return {
      height := height + 0.5 * p.gravity * localTime * localTime
      velocity := p.gravity * localTime
    }
  else
    return {
      height := 0.5 * p.gravity * localTime * localTime - impactVelocity * localTime
      velocity := p.gravity * localTime - impactVelocity
    }

structure IntervalSolve where
  tStart : Float
  tAttempt : Float
  tAfter : Float
  stateAfter : BallState
  result : Result
  deriving Repr, Inhabited

def ballSolver :=
  RK4.solver
    (Term := ODETerm BallState Unit)
    (Y := BallState)
    (VF := BallState)
    (Args := Unit)

def solveInterval? (p : BallParams) (tStart tAttempt : Float) (x0 : BallState) :
    Except String IntervalSolve := do
  let sol :=
    diffeqsolve
      (Term := ODETerm BallState Unit)
      (Y := BallState)
      (VF := BallState)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      (odeTerm p) ballSolver tStart tAttempt (some p.stepSize) x0 ()
      (saveat := { t1 := true })
      (event := some (impactEvent p))
  if !sol.result.isOkay then
    .error s!"bouncing-ball solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "bouncing-ball solve did not save endpoint"
        else
          pure {
            tStart := tStart
            tAttempt := tAttempt
            tAfter := ts[ts.size - 1]!
            stateAfter := ys[ys.size - 1]!
            result := sol.result
          }
    | _, _ => .error "bouncing-ball solve did not save endpoint arrays"

structure ImpactRecord where
  eventIndex : Nat
  time : Float
  preState : BallState
  postState : BallState
  saltation : SaltationData
  deriving Repr, Inhabited

structure SimulationResult where
  references : Array DrakeReference
  finalTime : Float
  finalState : BallState
  impacts : Array ImpactRecord
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def systemDeclarationVertex : VertexId := 4900

def fullPhysicsVertex : VertexId := 4901

def closedFormRegressionVertex : VertexId := 4902

def simulatorBoundaryVertex : VertexId := 4903

private def localMove (vertex : VertexId) (label : String) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    reads := #[vertex]
    writes := #[vertex]
    exactness := .exact
    cost := { work := 1.0, memory := 1.0 }
    label := label
  }

def systemDeclarationMove
    (boundary : BouncingBallSystemBoundary := systemBoundary) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[systemDeclarationVertex]
    reads := #[systemDeclarationVertex]
    writes := #[simulatorBoundaryVertex]
    exactness := .exact
    cost := { work := 1.0, memory := 1.0 }
    label :=
      s!"{boundary.systemName} LeafSystem declaration, y0 output, and {boundary.witnessName} witness reset"
  }

def closedFormRegressionMove : SkeletonMove :=
  localMove closedFormRegressionVertex
    "Drake bouncing_ball_test closed-form elastic repeated-impact regression"

structure BouncingBallResult where
  references : Array DrakeReference
  systemBoundary : BouncingBallSystemBoundary
  params : BallParams
  initialState : BallState
  simulation : SimulationResult
  freeFlightFullPhysics : FullPhysicsResult
  closedFormFinal : BallState
  closedFormHeightError : Float
  closedFormVelocityError : Float
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def intervalSegment
    (idx : Nat)
    (solve : IntervalSolve)
    (madeJumpAfter : Bool) : AcceptedStepSegment :=
  {
    id := idx
    attemptIndex := idx
    tStart := solve.tStart
    tAttempt := solve.tAttempt
    tAfter := solve.tAfter
    madeJumpAfter := madeJumpAfter
    label :=
      if madeJumpAfter then
        s!"bouncing-ball localized flight interval {idx}"
      else
        s!"bouncing-ball terminal flight interval {idx}"
  }

def impactVertex (idx : Nat) : VertexId :=
  400 + idx

def simulateLoop?
    (p : BallParams)
    (tFinal : Float)
    (fuel : Nat)
    (idx : Nat)
    (t : Float)
    (x : BallState)
    (trace : DynamicEventTrace)
    (impacts : Array ImpactRecord) :
    Except String SimulationResult :=
  match fuel with
  | 0 => .error s!"bouncing-ball simulation exceeded impact budget at t={t}"
  | fuel' + 1 => do
      if t >= tFinal then
        trace.validate?
        pure {
          references := drakeReferences
          finalTime := t
          finalState := x
          impacts := impacts
          trace := trace
          moves := trace.moves
        }
      else
        let solve ← solveInterval? p t tFinal x
        match solve.result with
        | Result.eventOccurred =>
            let pre := solve.stateAfter
            if pre.velocity > 0.0 then
              .error s!"impact witness localized with upward velocity {pre.velocity}"
            else
              let post := resetState p pre
              let saltation := impactSaltationData p pre
              saltation.validateGamma
              let segment := intervalSegment idx solve true
              let trace' :=
                trace
                  |>.push (.interval segment)
                  |>.push (.saltation (impactVertex idx) saltation)
              let impacts' := impacts.push {
                eventIndex := idx
                time := solve.tAfter
                preState := pre
                postState := post
                saltation := saltation
              }
              simulateLoop? p tFinal fuel' (idx + 1) solve.tAfter post trace' impacts'
        | Result.successful =>
            let segment := intervalSegment idx solve false
            let trace' := trace.push (.interval segment)
            trace'.validate?
            pure {
              references := drakeReferences
              finalTime := solve.tAfter
              finalState := solve.stateAfter
              impacts := impacts
              trace := trace'
              moves := trace'.moves
            }
        | other =>
            .error s!"unexpected okay result from bouncing-ball solve: {reprStr other}"

def simulate? (p : BallParams := params) (tFinal : Float := 10.0)
    (x0 : BallState := initialState p) (maxImpacts : Nat := 128) :
    Except String SimulationResult :=
  simulateLoop? p tFinal maxImpacts 0 0.0 x0 DynamicEventTrace.empty #[]

def buildEndToEnd?
    (p : BallParams := params)
    (tFinal : Float := 10.0)
    (maxImpacts : Nat := 128) :
    Except String BouncingBallResult := do
  systemBoundary.validate?
  p.validate?
  if Float.abs (p.restitution - 1.0) > 1.0e-12 then
    .error "BouncingBall closed-form regression boundary currently requires unit restitution"
  if Float.abs p.initialVelocity > 1.0e-12 then
    .error "BouncingBall closed-form regression boundary currently requires release from rest"
  let x0 := initialState p
  let simulation ← simulate? p tFinal x0 maxImpacts
  simulation.trace.validate?
  let provider := fullPhysicsPrimitiveProvider p
    "bouncing-ball end-to-end free-flight full physics provider"
  let fullPhysics ← provider.solveAt? x0 fullPhysicsVertex
  let closedFormFinal := closedFormUnitRestitutionFromRest p p.initialHeight tFinal
  let moves :=
    #[
      systemDeclarationMove systemBoundary,
      closedFormRegressionMove
    ] ++ #[fullPhysics.supportMove, fullPhysics.move] ++ simulation.moves
  pure {
    references := drakeReferences
    systemBoundary := systemBoundary
    params := p
    initialState := x0
    simulation := simulation
    freeFlightFullPhysics := fullPhysics
    closedFormFinal := closedFormFinal
    closedFormHeightError :=
      Float.abs (simulation.finalState.height - closedFormFinal.height)
    closedFormVelocityError :=
      Float.abs (simulation.finalState.velocity - closedFormFinal.velocity)
    moves := moves
  }

end Tyr.EventSkeleton.Examples.BouncingBall
