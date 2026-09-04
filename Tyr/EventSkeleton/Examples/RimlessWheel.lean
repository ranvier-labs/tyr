import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.NamedVector
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Trace

/-!
# Drake Rimless Wheel Event-Skeleton Example

This is a solver-backed port of `../drake/examples/rimless_wheel`.
The continuous physics follows Drake's `RimlessWheel` plant:

* continuous state is `(theta, thetadot)`,
* stance dynamics are `thetadot` and `sin(theta) * gravity / length`,
* forward and backward step witnesses are positive-to-nonpositive guards,
* reset changes stance leg, scales angular velocity by `cos(2 * alpha)`,
  updates the discrete toe position, and may enter double support.

The EventSkeleton layer records localized ODE intervals and impact saltation
events without replacing the underlying physics path.
-/

namespace Tyr.EventSkeleton.Examples.RimlessWheel

open Tyr.EventSkeleton
open torch.DiffEq

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/rimless_wheel/simulate.cc"
      concept := "builds the RimlessWheel + SceneGraph diagram, sets command-line initial state and simulator accuracy, advances to t=10, and checks theta remains inside one spoke interval"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel.h"
      concept := "declares RimlessWheel ports, continuous/discrete state access, witnesses, and reset events"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel.cc"
      concept := "defines stance dynamics, forward/backward witness guards, impact reset, toe update, and double-support stop"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel_continuous_state.h"
      concept := "defines RimlessWheelContinuousState coordinate order and BasicVector accessors"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel_continuous_state.cc"
      concept := "defines RimlessWheelContinuousStateIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel_geometry.h"
      concept := "connects the floating-base state output to SceneGraph geometry for Drake visualizer playback"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel_geometry.cc"
      concept := "registers the ramp, center frame, hub, spokes, and floating-base pose output for SceneGraph"
    },
    {
      path := "../drake/examples/rimless_wheel/test/rimless_wheel_geometry_test.cc"
      concept := "acceptance test for adding RimlessWheelGeometry to a DiagramBuilder with RimlessWheel and SceneGraph"
    },
    {
      path := "../drake/examples/rimless_wheel/test/rimless_wheel_test.cc"
      concept := "checks step direction, energy bookkeeping, dense discontinuity, and double-support fixed point"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel_params.h"
      concept := "defines mass, length, gravity, spoke count, and ramp slope defaults"
    },
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel_params.cc"
      concept := "defines RimlessWheelParamsIndices::GetCoordinateNames"
    }
  ]

structure WheelParams where
  mass : Float := 1.0
  length : Float := 1.0
  gravity : Float := 9.81
  numberOfSpokes : Float := 8.0
  slope : Float := 0.08
  rootTol : Float := 1.0e-8
  stepSize : Float := 1.0e-3
  resetClearance : Float := 1.0e-12
  deriving Repr, Inhabited

def params : WheelParams := {}

namespace WheelParams

def lowerBounds : Array (Option Float) :=
  #[some 0.0, some 0.0, some 0.0, some 4.0, none]

def upperBounds : Array (Option Float) :=
  #[none, none, none, none, none]

def asArray (p : WheelParams) : Array Float :=
  #[p.mass, p.length, p.gravity, p.numberOfSpokes, p.slope]

def isValid (p : WheelParams) : Bool :=
  (asArray p).all (fun x => x.isFinite) &&
    p.mass >= 0.0 &&
    p.length >= 0.0 &&
    p.gravity >= 0.0 &&
    p.numberOfSpokes >= 4.0

def fromArray? (xs : Array Float) : Except String WheelParams := do
  if xs.size != 5 then
    .error s!"RimlessWheelParams expects 5 coordinates, got {xs.size}"
  let p : WheelParams := {
    mass := xs[0]!
    length := xs[1]!
    gravity := xs[2]!
    numberOfSpokes := xs[3]!
    slope := xs[4]!
  }
  if !p.isValid then
    .error s!"RimlessWheelParams values are outside Drake's BasicVector domain: {reprStr xs}"
  pure p

end WheelParams

def rimlessWheelContinuousStateVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "RimlessWheelContinuousState"
    headerPath := "../drake/examples/rimless_wheel/rimless_wheel_continuous_state.h"
    implementationPath? := some "../drake/examples/rimless_wheel/rimless_wheel_continuous_state.cc"
    coordinateNames := #["theta", "thetadot"]
    defaults := #[0.0, 0.0]
    lowerBounds := #[none, none]
    upperBounds := #[none, none]
  }

def rimlessWheelParamsVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "RimlessWheelParams"
    headerPath := "../drake/examples/rimless_wheel/rimless_wheel_params.h"
    implementationPath? := some "../drake/examples/rimless_wheel/rimless_wheel_params.cc"
    coordinateNames := #["mass", "length", "gravity", "number_of_spokes", "slope"]
    defaults := WheelParams.asArray params
    lowerBounds := WheelParams.lowerBounds
    upperBounds := WheelParams.upperBounds
  }

def alpha (p : WheelParams := params) : Float :=
  pi / p.numberOfSpokes

def stepLength (p : WheelParams := params) : Float :=
  2.0 * p.length * Float.sin (alpha p)

def doubleSupportThreshold (p : WheelParams := params) : Float :=
  0.01 * Float.sqrt (p.gravity / p.length)

structure WheelState where
  theta : Float
  thetaDot : Float
  deriving Repr, Inhabited

instance : DiffEqSpace WheelState where
  add a b := { theta := a.theta + b.theta, thetaDot := a.thetaDot + b.thetaDot }
  sub a b := { theta := a.theta - b.theta, thetaDot := a.thetaDot - b.thetaDot }
  scale s x := { theta := s * x.theta, thetaDot := s * x.thetaDot }

instance : DiffEqSeminorm WheelState where
  rms x := max (Float.abs x.theta) (Float.abs x.thetaDot)

instance : DiffEqElem WheelState where
  abs x := { theta := Float.abs x.theta, thetaDot := Float.abs x.thetaDot }
  max a b := { theta := max a.theta b.theta, thetaDot := max a.thetaDot b.thetaDot }
  addScalar s x := { theta := x.theta + s, thetaDot := x.thetaDot + s }
  div a b := { theta := a.theta / b.theta, thetaDot := a.thetaDot / b.thetaDot }

structure WheelHybridState where
  cont : WheelState
  toe : Float := 0.0
  doubleSupport : Bool := false
  deriving Repr, Inhabited

inductive StepDirection where
  | forward
  | backward
  deriving Repr, BEq, Inhabited

namespace StepDirection

def name : StepDirection → String
  | .forward => "forward"
  | .backward => "backward"

end StepDirection

def initialState (_p : WheelParams := params) : WheelHybridState :=
  {
    cont := { theta := 0.0, thetaDot := 5.0 }
    toe := 0.0
    doubleSupport := false
  }

def stateAsArray (x : WheelState) : Array Float :=
  #[x.theta, x.thetaDot]

def wheelStateFinite (x : WheelState) : Bool :=
  x.theta.isFinite && x.thetaDot.isFinite

def stateFromArray? (xs : Array Float) : Except String WheelState := do
  if xs.size != 2 then
    .error s!"RimlessWheelContinuousState expects 2 coordinates, got {xs.size}"
  let x : WheelState := { theta := xs[0]!, thetaDot := xs[1]! }
  if !wheelStateFinite x then
    .error s!"RimlessWheelContinuousState values must be finite, got {reprStr xs}"
  pure x

def derivative (p : WheelParams) (doubleSupport : Bool) (x : WheelState) : WheelState :=
  if doubleSupport then
    { theta := 0.0, thetaDot := 0.0 }
  else
    { theta := x.thetaDot, thetaDot := Float.sin x.theta * p.gravity / p.length }

def vectorFieldArray (p : WheelParams) (doubleSupport : Bool) (x : WheelState) : Array Float :=
  stateAsArray (derivative p doubleSupport x)

def odeTerm (p : WheelParams) : ODETerm WheelState Unit :=
  { vectorField := fun _t x _ => derivative p false x }

def forwardGuard (p : WheelParams) (x : WheelState) : Float :=
  p.slope + alpha p - x.theta

def backwardGuard (p : WheelParams) (x : WheelState) : Float :=
  x.theta - p.slope + alpha p

def guard (p : WheelParams) : StepDirection → WheelState → Float
  | .forward, x => forwardGuard p x
  | .backward, x => backwardGuard p x

def guardGrad : StepDirection → Array Float
  | .forward => #[-1.0, 0.0]
  | .backward => #[1.0, 0.0]

def stepEvent (p : WheelParams) (direction : StepDirection) : EventSpec WheelState Unit :=
  {
    condition := .real (fun _t x _ => guard p direction x)
    direction := some false
    terminate := true
    rootTol := p.rootTol
  }

def eventTree (p : WheelParams) : EventTree WheelState Unit :=
  .branch #[
    .leaf (stepEvent p .forward),
    .leaf (stepEvent p .backward)
  ]

def totalEnergy (p : WheelParams) (x : WheelState) : Float :=
  let kinetic := 0.5 * p.mass * Float.pow (p.length * x.thetaDot) 2.0
  let potential := p.mass * p.gravity * p.length * Float.cos x.theta
  kinetic + potential

def thetaDotForEnergy (p : WheelParams) (desiredEnergy theta : Float) : Float :=
  let potential := p.mass * p.gravity * p.length * Float.cos theta
  Float.sqrt (2.0 * (desiredEnergy - potential) / (p.mass * p.length * p.length))

def fixedPointState (p : WheelParams := params) : WheelState :=
  let a := alpha p
  {
    theta := p.slope - a
    thetaDot :=
      (Float.cos (2.0 * a) / Float.sin (2.0 * a)) *
        Float.sqrt (4.0 * p.gravity / p.length * Float.sin a * Float.sin p.slope)
  }

def fixedPointEnergy (p : WheelParams := params) : Float :=
  totalEnergy p (fixedPointState p)

def limitCyclePreForwardState (p : WheelParams := params) : WheelState :=
  let theta := p.slope + alpha p / 2.0
  { theta := theta, thetaDot := thetaDotForEnergy p (fixedPointEnergy p) theta }

structure ResetResult where
  state : WheelHybridState
  enteredDoubleSupport : Bool
  deriving Repr, Inhabited

def resetJacobian (p : WheelParams) (doubleSupport : Bool) : Array (Array Float) :=
  let c := Float.cos (2.0 * alpha p)
  #[#[1.0, 0.0], #[0.0, if doubleSupport then 0.0 else c]]

def applyReset
    (p : WheelParams)
    (direction : StepDirection)
    (x : WheelHybridState) : ResetResult :=
  let a := alpha p
  let c := Float.cos (2.0 * a)
  let rawThetaDot := x.cont.thetaDot * c
  let threshold := doubleSupportThreshold p
  let entersDoubleSupport :=
    match direction with
    | .forward => rawThetaDot < threshold
    | .backward => rawThetaDot > -threshold
  let thetaDot := if entersDoubleSupport then 0.0 else rawThetaDot
  let theta :=
    match direction with
    | .forward => x.cont.theta - 2.0 * a + p.resetClearance
    | .backward => x.cont.theta + 2.0 * a - p.resetClearance
  let toe :=
    match direction with
    | .forward => x.toe + stepLength p
    | .backward => x.toe - stepLength p
  {
    state := {
      cont := { theta := theta, thetaDot := thetaDot }
      toe := toe
      doubleSupport := entersDoubleSupport
    }
    enteredDoubleSupport := entersDoubleSupport
  }

def stepSaltationData
    (p : WheelParams)
    (direction : StepDirection)
    (pre : WheelHybridState)
    (reset : ResetResult) : SaltationData :=
  SaltationData.mkFromFields
    (resetJacobian p reset.state.doubleSupport)
    (guardGrad direction)
    (vectorFieldArray p false pre.cont)
    (vectorFieldArray p reset.state.doubleSupport reset.state.cont)

def floatingBaseState (p : WheelParams) (x : WheelHybridState) : Array Float :=
  let a := alpha p
  let thetaUnrolled := x.cont.theta + a * x.toe / (p.length * Float.sin a)
  #[
    x.toe * Float.cos p.slope + p.length * Float.sin x.cont.theta,
    0.0,
    -x.toe * Float.sin p.slope + p.length * Float.cos x.cont.theta,
    0.0,
    thetaUnrolled,
    0.0,
    -x.cont.thetaDot * p.length * Float.cos x.cont.theta,
    0.0,
    x.cont.thetaDot * p.length * Float.sin x.cont.theta,
    0.0,
    x.cont.thetaDot,
    0.0
  ]

/-! ## RimlessWheelGeometry SceneGraph provider -/

def rimlessWheelGeometrySourceId : Nat := 7610
def rimlessWheelCenterFrameId : Nat := 7611
def rimlessWheelRampGeometryId : Nat := 7612
def rimlessWheelHubGeometryId : Nat := 7613
def rimlessWheelSpokeGeometryBaseId : Nat := 7620

def rimlessWheelGeometryStateInputVertex : VertexId := 7640
def rimlessWheelGeometryProviderVertex : VertexId := 7641
def rimlessWheelGeometryPoseOutputVertex : VertexId := 7642

private def rimlessWheelGeometryProperties (rgba : SceneRgba) : SceneGeometryProperties :=
  {
    roles := #[.illustration, .perception]
    diffuseRgba? := some rgba
    renderLabel? := none
  }

def rimlessWheelSpokeCount? (p : WheelParams := params) : Except String Nat := do
  if !p.numberOfSpokes.isFinite || p.numberOfSpokes < 1.0 then
    .error s!"RimlessWheelGeometry requires at least one finite spoke, got {p.numberOfSpokes}"
  let n := p.numberOfSpokes.toUInt64.toNat
  if n == 0 then
    .error s!"RimlessWheelGeometry spoke count rounded to zero from {p.numberOfSpokes}"
  if Float.abs (n.toFloat - p.numberOfSpokes) > 1.0e-12 then
    .error s!"RimlessWheelGeometry requires an integral spoke count, got {p.numberOfSpokes}"
  pure n

def rimlessWheelSpokeGeometry (p : WheelParams) (i : Nat) : SceneGeometry :=
  {
    id := rimlessWheelSpokeGeometryBaseId + i
    sourceId := rimlessWheelGeometrySourceId
    frameId? := some rimlessWheelCenterFrameId
    X_FG := {
      translation := { x := 0.0, y := 0.0, z := -p.length / 2.0 }
      rotationAxis := SceneVec3.unitY
      rotationAngle := 2.0 * pi * i.toFloat / p.numberOfSpokes
    }
    shape := .cylinder 0.0075 p.length
    name := s!"spoke{i}"
    properties := rimlessWheelGeometryProperties { r := 0.0, g := 0.0, b := 0.0, a := 1.0 }
  }

def rimlessWheelSpokeGeometries? (p : WheelParams := params) : Except String (Array SceneGeometry) := do
  let count ← rimlessWheelSpokeCount? p
  let mut out : Array SceneGeometry := #[]
  for i in [:count] do
    out := out.push (rimlessWheelSpokeGeometry p i)
  pure out

def rimlessWheelBaseGeometries (p : WheelParams := params) : Array SceneGeometry :=
  #[
    {
      id := rimlessWheelRampGeometryId
      sourceId := rimlessWheelGeometrySourceId
      frameId? := none
      X_FG := {
        translation := { x := 0.0, y := 0.0, z := -5.0 }
        rotationAxis := SceneVec3.unitY
        rotationAngle := p.slope
      }
      shape := .box 100.0 1.0 10.0
      name := "ramp"
      properties :=
        rimlessWheelGeometryProperties { r := 0.9297, g := 0.7930, b := 0.6758, a := 1.0 }
    },
    {
      id := rimlessWheelHubGeometryId
      sourceId := rimlessWheelGeometrySourceId
      frameId? := some rimlessWheelCenterFrameId
      X_FG := {
        translation := {}
        rotationAxis := SceneVec3.unitX
        rotationAngle := pi / 2.0
      }
      shape := .cylinder 0.2 0.2
      name := "hub"
      properties := rimlessWheelGeometryProperties { r := 0.6, g := 0.2, b := 0.2, a := 1.0 }
    }
  ]

def rimlessWheelGeometryProvider? (p : WheelParams := params) : Except String SceneGraphProvider := do
  let spokes ← rimlessWheelSpokeGeometries? p
  pure {
    sources := #[
      { id := rimlessWheelGeometrySourceId, name := "RimlessWheelGeometry" }
    ]
    frames := #[
      {
        id := rimlessWheelCenterFrameId
        sourceId := rimlessWheelGeometrySourceId
        name := "center"
      }
    ]
    geometries := rimlessWheelBaseGeometries p ++ spokes
    label := "RimlessWheelGeometry SceneGraph provider"
  }

private def floatingBaseStateFinite (x : Array Float) : Bool :=
  x.all (fun xi => xi.isFinite)

def rimlessWheelGeometryPoseOutput
    (floatingBaseState : Array Float) : SceneFramePoseVector :=
  {
    poses := #[
      {
        frameId := rimlessWheelCenterFrameId
        X_WF := ScenePose3.fromRollPitchYaw
          {
            x := floatingBaseState.getD 0 0.0
            y := floatingBaseState.getD 1 0.0
            z := floatingBaseState.getD 2 0.0
          }
          (floatingBaseState.getD 3 0.0)
          (floatingBaseState.getD 4 0.0)
          (floatingBaseState.getD 5 0.0)
      }
    ]
  }

private def rimlessWheelGeometryMove
    (target : VertexId) (label : String) (reads : Array VertexId := #[])
    (writes : Array VertexId := #[]) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target]
    reads := reads
    writes := writes
    exactness := .exact
    label := label
  }

def rimlessWheelGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := rimlessWheelGeometryStateInputVertex
      kind := .state .boundary
      label := "RimlessWheelGeometry floating_base_state input"
    }
    |>.addVertex {
      id := rimlessWheelGeometryProviderVertex
      kind := .state .boundary
      label := "RimlessWheelGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := rimlessWheelGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "RimlessWheelGeometry geometry_pose output"
    }
    |>.addMove (rimlessWheelGeometryMove rimlessWheelGeometryProviderVertex
      "Register ramp, center frame, hub, and spoke geometry"
      #[] #[rimlessWheelGeometryProviderVertex])
    |>.addMove (rimlessWheelGeometryMove rimlessWheelGeometryPoseOutputVertex
      "OutputGeometryPose: floating_base_state -> center FramePoseVector"
      #[rimlessWheelGeometryStateInputVertex, rimlessWheelGeometryProviderVertex]
      #[rimlessWheelGeometryPoseOutputVertex])

structure RimlessWheelGeometryResult where
  references : Array DrakeReference
  params : WheelParams
  inputPortName : String := "floating_base_state"
  inputPortSize : Nat := 12
  outputPortName : String := "geometry_pose"
  provider : SceneGraphProvider
  sampleFloatingBaseState : Array Float
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildRimlessWheelGeometry?
    (p : WheelParams := params)
    (floatingState : Array Float := floatingBaseState p (initialState p)) :
    Except String RimlessWheelGeometryResult := do
  if !p.mass.isFinite || p.mass <= 0.0 then
    .error "RimlessWheelGeometry requires positive finite mass"
  if !p.length.isFinite || p.length <= 0.0 then
    .error "RimlessWheelGeometry requires positive finite spoke length"
  if !p.slope.isFinite then
    .error "RimlessWheelGeometry requires finite ramp slope"
  if floatingState.size != 12 then
    .error s!"RimlessWheelGeometry floating_base_state must have size 12, got {floatingState.size}"
  if !floatingBaseStateFinite floatingState then
    .error "RimlessWheelGeometry floating_base_state must be finite"
  let provider ← rimlessWheelGeometryProvider? p
  provider.validate?
  let poses := rimlessWheelGeometryPoseOutput floatingState
  poses.validate? provider
  pure {
    references := drakeReferences
    params := p
    provider := provider
    sampleFloatingBaseState := floatingState
    poses := poses
    graph := rimlessWheelGeometryGraph
    moves := rimlessWheelGeometryGraph.moves
  }

structure IntervalSolve where
  tStart : Float
  tAttempt : Float
  tAfter : Float
  stateAfter : WheelState
  result : Result
  direction? : Option StepDirection
  eventMaskLast : Option (Array Bool)
  deriving Repr, Inhabited

def wheelSolver :=
  RK4.solver
    (Term := ODETerm WheelState Unit)
    (Y := WheelState)
    (VF := WheelState)
    (Args := Unit)

def directionFromMask? (mask? : Option (Array Bool)) : Except String StepDirection := do
  match mask? with
  | none => .error "rimless-wheel event occurred without an event mask"
  | some mask =>
      if mask.getD 0 false then
        pure .forward
      else if mask.getD 1 false then
        pure .backward
      else
        .error s!"rimless-wheel event mask did not select a witness: {reprStr mask}"

def solveInterval?
    (p : WheelParams)
    (tStart tAttempt : Float)
    (x0 : WheelState) :
    Except String IntervalSolve := do
  let treeSol :=
    diffeqsolveEventTree
      (Term := ODETerm WheelState Unit)
      (Y := WheelState)
      (VF := WheelState)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      (odeTerm p) wheelSolver tStart tAttempt (some p.stepSize) x0 ()
      (eventTree p)
      (saveat := { t1 := true })
  let sol := treeSol.base
  if !sol.result.isOkay then
    .error s!"rimless-wheel solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "rimless-wheel solve did not save endpoint"
        else
          let direction? ←
            match sol.result with
            | Result.eventOccurred => do
                let direction ← directionFromMask? sol.eventMaskLast
                pure (some direction)
            | Result.successful => pure none
            | other => .error s!"unexpected okay result from rimless-wheel solve: {reprStr other}"
          pure {
            tStart := tStart
            tAttempt := tAttempt
            tAfter := ts[ts.size - 1]!
            stateAfter := ys[ys.size - 1]!
            result := sol.result
            direction? := direction?
            eventMaskLast := sol.eventMaskLast
          }
    | _, _ => .error "rimless-wheel solve did not save endpoint arrays"

structure StepRecord where
  eventIndex : Nat
  direction : StepDirection
  time : Float
  preState : WheelHybridState
  postState : WheelHybridState
  saltation : SaltationData
  enteredDoubleSupport : Bool
  deriving Repr, Inhabited

structure SimulationResult where
  references : Array DrakeReference
  finalTime : Float
  finalState : WheelHybridState
  steps : Array StepRecord
  trace : DynamicEventTrace
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
        s!"rimless-wheel localized {solve.direction?.getD .forward |>.name} stance interval {idx}"
      else
        s!"rimless-wheel terminal stance interval {idx}"
  }

def frozenSegment (idx : Nat) (tStart tFinal : Float) : AcceptedStepSegment :=
  {
    id := idx
    attemptIndex := idx
    tStart := tStart
    tAttempt := tFinal
    tAfter := tFinal
    madeJumpAfter := false
    label := s!"rimless-wheel double-support frozen interval {idx}"
  }

def stepEventVertex (idx : Nat) : VertexId :=
  600 + idx

def simulateLoop?
    (p : WheelParams)
    (tFinal : Float)
    (fuel : Nat)
    (idx : Nat)
    (t : Float)
    (x : WheelHybridState)
    (trace : DynamicEventTrace)
    (steps : Array StepRecord) :
    Except String SimulationResult :=
  match fuel with
  | 0 => .error s!"rimless-wheel simulation exceeded step budget at t={t}"
  | fuel' + 1 => do
      if t >= tFinal then
        trace.validate?
        pure {
          references := drakeReferences
          finalTime := t
          finalState := x
          steps := steps
          trace := trace
          moves := trace.moves
        }
      else if x.doubleSupport then
        let trace' := trace.push (.interval (frozenSegment idx t tFinal))
        trace'.validate?
        pure {
          references := drakeReferences
          finalTime := tFinal
          finalState := x
          steps := steps
          trace := trace'
          moves := trace'.moves
        }
      else
        let solve ← solveInterval? p t tFinal x.cont
        match solve.result, solve.direction? with
        | Result.eventOccurred, some direction =>
            let pre : WheelHybridState :=
              { cont := solve.stateAfter, toe := x.toe, doubleSupport := false }
            let reset := applyReset p direction pre
            let saltation := stepSaltationData p direction pre reset
            saltation.validateGamma
            let segment := intervalSegment idx solve true
            let trace' :=
              trace
                |>.push (.interval segment)
                |>.push (.saltation (stepEventVertex idx) saltation)
            let steps' := steps.push {
              eventIndex := idx
              direction := direction
              time := solve.tAfter
              preState := pre
              postState := reset.state
              saltation := saltation
              enteredDoubleSupport := reset.enteredDoubleSupport
            }
            simulateLoop? p tFinal fuel' (idx + 1) solve.tAfter reset.state trace' steps'
        | Result.successful, none =>
            let segment := intervalSegment idx solve false
            let trace' := trace.push (.interval segment)
            trace'.validate?
            pure {
              references := drakeReferences
              finalTime := solve.tAfter
              finalState := { cont := solve.stateAfter, toe := x.toe, doubleSupport := false }
              steps := steps
              trace := trace'
              moves := trace'.moves
            }
        | _, _ =>
            .error s!"rimless-wheel solve returned inconsistent result {reprStr solve.result} and direction {reprStr solve.direction?}"

def simulate? (p : WheelParams := params) (tFinal : Float := 10.0)
    (x0 : WheelHybridState := initialState p) (maxSteps : Nat := 256) :
    Except String SimulationResult :=
  simulateLoop? p tFinal maxSteps 0 0.0 x0 DynamicEventTrace.empty #[]

structure SimulateExecutableConfig where
  accuracy : Float := 1.0e-4
  initialAngle : Float := 0.0
  initialAngularVelocity : Float := 5.0
  targetRealtimeRate : Float := 1.0
  advanceTo : Float := 10.0
  maxSteps : Nat := 256
  plantName : String := "rimless_wheel"
  includeSceneGraph : Bool := true
  includeDrakeVisualizer : Bool := true
  floatingBaseOutputPort : String := "floating_base_state"
  deriving Repr, Inhabited

namespace SimulateExecutableConfig

def initialHybridState (config : SimulateExecutableConfig) : WheelHybridState :=
  {
    cont := {
      theta := config.initialAngle
      thetaDot := config.initialAngularVelocity
    }
    toe := 0.0
    doubleSupport := false
  }

def validate? (config : SimulateExecutableConfig) (p : WheelParams) :
    Except String Unit := do
  if config.plantName == "" then
    .error "rimless-wheel simulate plant name must be nonempty"
  if config.floatingBaseOutputPort == "" then
    .error "rimless-wheel geometry output port name must be nonempty"
  if !config.accuracy.isFinite || config.accuracy <= 0.0 then
    .error s!"rimless-wheel simulate accuracy must be positive and finite, got {config.accuracy}"
  if !config.targetRealtimeRate.isFinite || config.targetRealtimeRate < 0.0 then
    .error s!"rimless-wheel simulate target realtime rate must be nonnegative and finite, got {config.targetRealtimeRate}"
  if !config.advanceTo.isFinite || config.advanceTo <= 0.0 then
    .error s!"rimless-wheel simulate AdvanceTo time must be positive and finite, got {config.advanceTo}"
  if config.maxSteps == 0 then
    .error "rimless-wheel simulate max step budget must be positive"
  let x0 := config.initialHybridState
  if !wheelStateFinite x0.cont then
    .error "rimless-wheel simulate initial state must be finite"
  let a := alpha p
  if !(config.initialAngle > p.slope - a && config.initialAngle < p.slope + a) then
    .error s!"rimless-wheel initial angle {config.initialAngle} must lie in ({p.slope - a}, {p.slope + a})"
  if !config.includeSceneGraph then
    .error "rimless-wheel simulate.cc boundary should include SceneGraph geometry"
  if !config.includeDrakeVisualizer then
    .error "rimless-wheel simulate.cc boundary should include DrakeVisualizerd"

def finalThetaInsideSpokeInterval
    (_config : SimulateExecutableConfig)
    (p : WheelParams)
    (x : WheelHybridState) : Bool :=
  let a := alpha p
  x.cont.theta >= p.slope - a && x.cont.theta <= p.slope + a

end SimulateExecutableConfig

def simulateExecutableConfig : SimulateExecutableConfig := {}

def simulateExecutableGraph
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 7600, kind := .state .boundary, label := "../drake/examples/rimless_wheel/simulate.cc flags" }
    |>.addVertex { id := 7601, kind := .state .interior, label := config.plantName }
    |>.addVertex { id := 7602, kind := .state .interior, label := "RimlessWheelGeometry + SceneGraph" }
    |>.addVertex { id := 7603, kind := .state .interior, label := "DrakeVisualizerd" }
    |>.addVertex { id := 7604, kind := .state .checkpoint, label := "initial RimlessWheel context" }
    |>.addVertex { id := 7605, kind := .interval, label := s!"Simulator.AdvanceTo({config.advanceTo})" }
    |>.addVertex { id := 7606, kind := .state .checkpoint, label := "postcondition theta in spoke interval" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7601]
      reads := #[7600]
      writes := #[7601]
      label := "DiagramBuilder.AddSystem<RimlessWheel>"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7602]
      reads := #[7601]
      writes := #[7602]
      label := s!"RimlessWheelGeometry::AddToBuilder from {config.floatingBaseOutputPort}"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[7603]
      reads := #[7602]
      writes := #[7603]
      label := "DrakeVisualizerd::AddToBuilder"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[7604]
      reads := #[7600, 7601]
      writes := #[7604]
      label := s!"set theta={config.initialAngle}, thetadot={config.initialAngularVelocity}, accuracy={config.accuracy}"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[7605]
      reads := #[7601, 7604]
      writes := #[7606]
      cost := { work := config.advanceTo }
      label := "Simulator.AdvanceTo via rimless-wheel ODE, event-tree witnesses, and impact resets"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[7606]
      reads := #[7605]
      writes := #[7606]
      label := "DRAKE_DEMAND final theta inside [slope - alpha, slope + alpha]"
    }

structure SimulateExecutableBoundary where
  config : SimulateExecutableConfig
  initialState : WheelHybridState
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildSimulateExecutableBoundary?
    (p : WheelParams := params)
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    Except String SimulateExecutableBoundary := do
  config.validate? p
  let graph := simulateExecutableGraph config
  pure {
    config := config
    initialState := config.initialHybridState
    graph := graph
    moves := graph.moves
  }

def executeSimulateExecutable?
    (p : WheelParams := params)
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    Except String SimulationResult := do
  config.validate? p
  let result ← simulate? p config.advanceTo config.initialHybridState config.maxSteps
  if !config.finalThetaInsideSpokeInterval p result.finalState then
    .error s!"rimless-wheel simulate postcondition failed, theta={result.finalState.cont.theta}"
  pure result

def continuousStateBoundaryVertex : VertexId := 7630

def paramsBoundaryVertex : VertexId := 7631

private def generatedVectorBoundaryMove
    (vertex : VertexId) (boundary : NamedVectorBoundary) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    reads := #[vertex]
    writes := #[vertex]
    exactness := .exact
    cost := { work := boundary.dimension.toFloat, memory := boundary.dimension.toFloat }
    label := s!"generated BasicVector boundary: {boundary.typeName}"
  }

structure RimlessWheelResult where
  references : Array DrakeReference
  params : WheelParams
  continuousStateBoundary : NamedVectorBoundary
  paramsBoundary : NamedVectorBoundary
  geometry : RimlessWheelGeometryResult
  executableBoundary : SimulateExecutableBoundary
  executableRun : SimulationResult
  finalThetaInsideSpokeInterval : Bool
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd?
    (p : WheelParams := params)
    (config : SimulateExecutableConfig := simulateExecutableConfig) :
    Except String RimlessWheelResult := do
  if !p.isValid then
    .error s!"RimlessWheel params are outside Drake's BasicVector domain: {reprStr (WheelParams.asArray p)}"
  rimlessWheelContinuousStateVectorBoundary.validate?
  rimlessWheelParamsVectorBoundary.validate?
  config.validate? p
  let executableBoundary ← buildSimulateExecutableBoundary? p config
  let geometry ← buildRimlessWheelGeometry? p
    (floatingBaseState p executableBoundary.initialState)
  let executableRun ← executeSimulateExecutable? p config
  executableRun.trace.validate?
  let moves :=
    #[
      generatedVectorBoundaryMove continuousStateBoundaryVertex
        rimlessWheelContinuousStateVectorBoundary,
      generatedVectorBoundaryMove paramsBoundaryVertex
        rimlessWheelParamsVectorBoundary
    ] ++ geometry.moves ++ executableBoundary.moves ++ executableRun.moves
  pure {
    references := drakeReferences
    params := p
    continuousStateBoundary := rimlessWheelContinuousStateVectorBoundary
    paramsBoundary := rimlessWheelParamsVectorBoundary
    geometry := geometry
    executableBoundary := executableBoundary
    executableRun := executableRun
    finalThetaInsideSpokeInterval :=
      config.finalThetaInsideSpokeInterval p executableRun.finalState
    moves := moves
  }

end Tyr.EventSkeleton.Examples.RimlessWheel
