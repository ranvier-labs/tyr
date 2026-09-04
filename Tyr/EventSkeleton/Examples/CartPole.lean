import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Trace

/-!
# Drake Cart-Pole Event-Skeleton Example

This ports the core multibody dynamics from
`../drake/examples/multibody/cart_pole`.

Drake models the cart-pole with a prismatic cart joint and an unactuated pole
pin joint.  The state is `(x, theta, xdot, thetadot)` and the manipulator form is

`M(q) vdot = tau_g(q) - C(q, v) v + B f`.

The local primitive uses `M(q) vdot = generalizedForces - biasForces`, so the
bias vector below is `C(q, v) v - tau_g(q)`.
-/

namespace Tyr.EventSkeleton.Examples.CartPole

open Tyr.EventSkeleton
open torch.DiffEq

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/cart_pole/BUILD.bazel"
      concept := "declares the cart-pole passive simulation binary and regression test"
    },
    {
      path := "../drake/examples/multibody/cart_pole/cart_pole.sdf"
      concept := "declares CartSlider prismatic joint, PolePin revolute joint, and default masses"
    },
    {
      path := "../drake/examples/multibody/cart_pole/cart_pole_params.h"
      concept := "defines CartPoleParams coordinate order mc, mp, l, gravity"
    },
    {
      path := "../drake/examples/multibody/cart_pole/cart_pole_params.cc"
      concept := "defines CartPoleParamsIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/multibody/cart_pole/test/cart_pole_test.cc"
      concept := "checks hand-written mass matrix, dynamics, and implicit residual"
    },
    {
      path := "../drake/examples/multibody/cart_pole/cart_pole_passive_simulation.cc"
      concept := "runs passive continuous cart-pole simulation from x = 0, theta = 2"
    }
  ]

def stateCoordinateNames : Array String :=
  #["x", "theta", "xdot", "thetadot"]

def inputCoordinateNames : Array String :=
  #["cart_force"]

def parameterCoordinateNames : Array String :=
  #["mc", "mp", "l", "gravity"]

private def hasDuplicateString (xs : Array String) : Bool := Id.run do
  let mut duplicate := false
  for i in [:xs.size] do
    for j in [:(xs.size - i - 1)] do
      let k := i + j + 1
      if xs[i]! == xs[k]! then
        duplicate := true
  return duplicate

structure CartPoleParamsVectorBoundary where
  typeName : String := "CartPoleParams"
  headerPath : String := "../drake/examples/multibody/cart_pole/cart_pole_params.h"
  implementationPath : String := "../drake/examples/multibody/cart_pole/cart_pole_params.cc"
  coordinateNames : Array String := parameterCoordinateNames
  defaults : Array Float
  isValidLowerBounds : Array (Option Float)
  isValidUpperBounds : Array (Option Float)
  elementLowerBounds : Array (Option Float)
  elementUpperBounds : Array (Option Float)
  movedFromAccessThrows : Bool := true
  supportsNamedVariables : Bool := true
  deriving Repr, Inhabited

namespace CartPoleParamsVectorBoundary

def dimension (boundary : CartPoleParamsVectorBoundary) : Nat :=
  boundary.coordinateNames.size

def indexOf? (boundary : CartPoleParamsVectorBoundary) (name : String) : Option Nat :=
  boundary.coordinateNames.findIdx? (fun candidate => candidate == name)

private def validateBoundsSize? (boundary : CartPoleParamsVectorBoundary)
    (xs : Array (Option Float)) (label : String) : Except String Unit := do
  if xs.size != boundary.dimension then
    .error s!"CartPoleParams {label} bounds size {xs.size} != dimension {boundary.dimension}"

def validate? (boundary : CartPoleParamsVectorBoundary) : Except String Unit := do
  if boundary.typeName != "CartPoleParams" then
    .error s!"CartPole named-vector type mismatch: {boundary.typeName}"
  if boundary.headerPath != "../drake/examples/multibody/cart_pole/cart_pole_params.h" then
    .error s!"CartPoleParams header path mismatch: {boundary.headerPath}"
  if boundary.implementationPath != "../drake/examples/multibody/cart_pole/cart_pole_params.cc" then
    .error s!"CartPoleParams implementation path mismatch: {boundary.implementationPath}"
  if boundary.coordinateNames != parameterCoordinateNames then
    .error s!"CartPoleParams coordinate names mismatch: {boundary.coordinateNames}"
  if hasDuplicateString boundary.coordinateNames then
    .error "CartPoleParams coordinate names must be unique"
  if boundary.defaults.size != boundary.dimension then
    .error s!"CartPoleParams defaults size {boundary.defaults.size} != dimension {boundary.dimension}"
  validateBoundsSize? boundary boundary.isValidLowerBounds "IsValid lower"
  validateBoundsSize? boundary boundary.isValidUpperBounds "IsValid upper"
  validateBoundsSize? boundary boundary.elementLowerBounds "element lower"
  validateBoundsSize? boundary boundary.elementUpperBounds "element upper"
  for i in [:boundary.dimension] do
    let value := boundary.defaults[i]!
    if !value.isFinite then
      .error s!"CartPoleParams.{boundary.coordinateNames[i]!} default is not finite: {value}"
    match boundary.isValidLowerBounds[i]!, boundary.isValidUpperBounds[i]! with
    | some lo, some hi =>
        if lo > hi then
          .error s!"CartPoleParams.{boundary.coordinateNames[i]!} has inverted IsValid bounds [{lo}, {hi}]"
        if value < lo || value > hi then
          .error s!"CartPoleParams.{boundary.coordinateNames[i]!} default {value} violates IsValid bounds [{lo}, {hi}]"
    | some lo, none =>
        if value < lo then
          .error s!"CartPoleParams.{boundary.coordinateNames[i]!} default {value} violates IsValid lower bound {lo}"
    | none, some hi =>
        if value > hi then
          .error s!"CartPoleParams.{boundary.coordinateNames[i]!} default {value} violates IsValid upper bound {hi}"
    | none, none => pure ()

end CartPoleParamsVectorBoundary

private def finiteNonnegative (x : Float) : Bool :=
  Float.isFinite x && x >= 0.0

structure CartPoleParams where
  mc : Float := 10.0
  mp : Float := 1.0
  l : Float := 0.5
  gravity : Float := 9.81
  stepSize : Float := 1.0e-3
  deriving Repr, Inhabited

namespace CartPoleParams

def isValid (p : CartPoleParams) : Bool :=
  finiteNonnegative p.mc &&
  finiteNonnegative p.mp &&
  finiteNonnegative p.l &&
  finiteNonnegative p.gravity

def asArray (p : CartPoleParams) : Array Float :=
  #[p.mc, p.mp, p.l, p.gravity]

def isValidLowerBounds : Array (Option Float) :=
  #[some 0.0, some 0.0, some 0.0, some 0.0]

def isValidUpperBounds : Array (Option Float) :=
  #[none, none, none, none]

def elementLowerBounds : Array (Option Float) :=
  #[none, none, none, none]

def elementUpperBounds : Array (Option Float) :=
  #[none, none, none, none]

def fromArray? (xs : Array Float) : Except String CartPoleParams := do
  if xs.size != 4 then
    .error s!"CartPoleParams expects 4 coordinates, got {xs.size}"
  let p : CartPoleParams :=
    {
      mc := xs[0]!
      mp := xs[1]!
      l := xs[2]!
      gravity := xs[3]!
    }
  if !p.isValid then
    .error s!"CartPoleParams values are outside Drake's IsValid domain: {reprStr xs}"
  pure p

end CartPoleParams

def params : CartPoleParams := {}

def cartPoleParamsVectorBoundary : CartPoleParamsVectorBoundary :=
  {
    defaults := params.asArray
    isValidLowerBounds := CartPoleParams.isValidLowerBounds
    isValidUpperBounds := CartPoleParams.isValidUpperBounds
    elementLowerBounds := CartPoleParams.elementLowerBounds
    elementUpperBounds := CartPoleParams.elementUpperBounds
  }

structure CartPoleInput where
  cartForce : Float := 0.0
  deriving Repr, Inhabited

namespace CartPoleInput

def isValid (u : CartPoleInput) : Bool :=
  Float.isFinite u.cartForce

def asArray (u : CartPoleInput) : Array Float :=
  #[u.cartForce]

def fromArray? (xs : Array Float) : Except String CartPoleInput := do
  if xs.size != 1 then
    .error s!"CartPoleInput expects 1 coordinate, got {xs.size}"
  let u : CartPoleInput := { cartForce := xs[0]! }
  if !u.isValid then
    .error s!"CartPoleInput values are not finite: {reprStr xs}"
  pure u

end CartPoleInput

structure CartPoleState where
  x : Float := 0.0
  theta : Float := 0.0
  xdot : Float := 0.0
  thetadot : Float := 0.0
  deriving Repr, Inhabited

namespace CartPoleState

def isValid (x : CartPoleState) : Bool :=
  Float.isFinite x.x &&
  Float.isFinite x.theta &&
  Float.isFinite x.xdot &&
  Float.isFinite x.thetadot

def asArray (x : CartPoleState) : Array Float :=
  #[x.x, x.theta, x.xdot, x.thetadot]

def fromArray? (xs : Array Float) : Except String CartPoleState := do
  if xs.size != 4 then
    .error s!"CartPoleState expects 4 coordinates, got {xs.size}"
  let x : CartPoleState :=
    {
      x := xs[0]!
      theta := xs[1]!
      xdot := xs[2]!
      thetadot := xs[3]!
    }
  if !x.isValid then
    .error s!"CartPoleState values are not finite: {reprStr xs}"
  pure x

end CartPoleState

instance : DiffEqSpace CartPoleState where
  add a b := {
    x := a.x + b.x
    theta := a.theta + b.theta
    xdot := a.xdot + b.xdot
    thetadot := a.thetadot + b.thetadot
  }
  sub a b := {
    x := a.x - b.x
    theta := a.theta - b.theta
    xdot := a.xdot - b.xdot
    thetadot := a.thetadot - b.thetadot
  }
  scale s x := {
    x := s * x.x
    theta := s * x.theta
    xdot := s * x.xdot
    thetadot := s * x.thetadot
  }

instance : DiffEqSeminorm CartPoleState where
  rms x :=
    max (max (Float.abs x.x) (Float.abs x.theta))
      (max (Float.abs x.xdot) (Float.abs x.thetadot))

instance : DiffEqElem CartPoleState where
  abs x := {
    x := Float.abs x.x
    theta := Float.abs x.theta
    xdot := Float.abs x.xdot
    thetadot := Float.abs x.thetadot
  }
  max a b := {
    x := max a.x b.x
    theta := max a.theta b.theta
    xdot := max a.xdot b.xdot
    thetadot := max a.thetadot b.thetadot
  }
  addScalar s x := {
    x := x.x + s
    theta := x.theta + s
    xdot := x.xdot + s
    thetadot := x.thetadot + s
  }
  div a b := {
    x := a.x / b.x
    theta := a.theta / b.theta
    xdot := a.xdot / b.xdot
    thetadot := a.thetadot / b.thetadot
  }

def defaultState : CartPoleState := {}

def defaultInput : CartPoleInput := {}

structure CartPolePhysicsState where
  state : CartPoleState := defaultState
  input : CartPoleInput := defaultInput
  deriving Repr, Inhabited

def physicsState
    (state : CartPoleState := defaultState)
    (input : CartPoleInput := defaultInput) : CartPolePhysicsState :=
  { state := state, input := input }

def stateAsArray (x : CartPoleState) : Array Float :=
  #[x.x, x.theta, x.xdot, x.thetadot]

def inputAsArray (u : CartPoleInput) : Array Float :=
  u.asArray

def cartPoleModelUri : String :=
  "package://drake/examples/multibody/cart_pole/cart_pole.sdf"

structure CartPoleModelAssetBoundary where
  modelName : String := "CartPole"
  sdfPath : String := "../drake/examples/multibody/cart_pole/cart_pole.sdf"
  packageUri : String := cartPoleModelUri
  linkNames : Array String := #["Cart", "Pole"]
  jointNames : Array String := #["CartSlider", "PolePin"]
  jointTypes : Array String := #["prismatic", "revolute"]
  actuatorNames : Array String := #["CartSlider"]
  jointAxes : Array (Array Float) := #[#[1.0, 0.0, 0.0], #[0.0, -1.0, 0.0]]
  cartMass : Float := params.mc
  polePointMass : Float := params.mp
  poleLength : Float := params.l
  cartBoxSize : Array Float := #[0.24, 0.12, 0.12]
  polePointMassRadius : Float := 0.05
  poleRodRadius : Float := 0.025
  poleRodLength : Float := params.l
  deriving Repr, Inhabited

namespace CartPoleModelAssetBoundary

private def finiteArray (xs : Array Float) : Bool :=
  xs.all Float.isFinite

def validate? (boundary : CartPoleModelAssetBoundary) : Except String Unit := do
  if boundary.modelName != "CartPole" then
    .error s!"CartPole model name mismatch: {boundary.modelName}"
  if boundary.sdfPath != "../drake/examples/multibody/cart_pole/cart_pole.sdf" then
    .error s!"CartPole SDF path mismatch: {boundary.sdfPath}"
  if boundary.packageUri != cartPoleModelUri then
    .error s!"CartPole package URI mismatch: {boundary.packageUri}"
  if boundary.linkNames != #["Cart", "Pole"] then
    .error s!"CartPole link names mismatch: {boundary.linkNames}"
  if boundary.jointNames != #["CartSlider", "PolePin"] then
    .error s!"CartPole joint names mismatch: {boundary.jointNames}"
  if boundary.jointTypes != #["prismatic", "revolute"] then
    .error s!"CartPole joint types mismatch: {boundary.jointTypes}"
  if boundary.actuatorNames != #["CartSlider"] then
    .error s!"CartPole actuator names mismatch: {boundary.actuatorNames}"
  if boundary.jointAxes != #[#[1.0, 0.0, 0.0], #[0.0, -1.0, 0.0]] then
    .error s!"CartPole joint axes mismatch: {boundary.jointAxes}"
  if !boundary.cartMass.isFinite || boundary.cartMass <= 0.0 then
    .error s!"CartPole cart mass must be positive and finite, got {boundary.cartMass}"
  if !boundary.polePointMass.isFinite || boundary.polePointMass <= 0.0 then
    .error s!"CartPole pole point mass must be positive and finite, got {boundary.polePointMass}"
  if !boundary.poleLength.isFinite || boundary.poleLength <= 0.0 then
    .error s!"CartPole pole length must be positive and finite, got {boundary.poleLength}"
  if boundary.cartBoxSize.size != 3 || !finiteArray boundary.cartBoxSize then
    .error s!"CartPole cart box size must have three finite entries, got {boundary.cartBoxSize}"
  if !boundary.polePointMassRadius.isFinite || boundary.polePointMassRadius <= 0.0 then
    .error s!"CartPole point mass radius must be positive and finite, got {boundary.polePointMassRadius}"
  if !boundary.poleRodRadius.isFinite || boundary.poleRodRadius <= 0.0 then
    .error s!"CartPole pole rod radius must be positive and finite, got {boundary.poleRodRadius}"
  if !boundary.poleRodLength.isFinite || boundary.poleRodLength <= 0.0 then
    .error s!"CartPole pole rod length must be positive and finite, got {boundary.poleRodLength}"

end CartPoleModelAssetBoundary

def cartPoleModelAssetBoundary : CartPoleModelAssetBoundary := {}

def qdotAsArray (x : CartPoleState) : Array Float :=
  #[x.xdot, x.thetadot]

def massMatrix (p : CartPoleParams) (x : CartPoleState) : Array (Array Float) :=
  let c := Float.cos x.theta
  let offdiag := p.mp * p.l * c
  #[
    #[p.mc + p.mp, offdiag],
    #[offdiag, p.mp * p.l * p.l]
  ]

def coriolisTimesV (p : CartPoleParams) (x : CartPoleState) : Array Float :=
  let s := Float.sin x.theta
  #[-p.mp * p.l * x.thetadot * x.thetadot * s, 0.0]

def gravityGeneralizedForces (p : CartPoleParams) (x : CartPoleState) : Array Float :=
  let s := Float.sin x.theta
  #[0.0, -p.mp * p.gravity * p.l * s]

def dynamicsBiasTerm (p : CartPoleParams) (x : CartPoleState) : Array Float :=
  FloatArray.sub (coriolisTimesV p x) (gravityGeneralizedForces p x)

def inputGeneralizedForces (u : CartPoleInput) : Array Float :=
  #[u.cartForce, 0.0]

def manipulatorEquation
    (p : CartPoleParams)
    (u : CartPoleInput)
    (x : CartPoleState) : ManipulatorEquation :=
  {
    massMatrix := massMatrix p x
    qdot := qdotAsArray x
    generalizedForces := inputGeneralizedForces u
    biasForces := dynamicsBiasTerm p x
    label := "cart-pole"
  }

def validateFullPhysicsInputs?
    (p : CartPoleParams) (u : CartPoleInput) (x : CartPoleState) :
    Except String Unit := do
  if !p.isValid then
    .error "cart-pole params are invalid"
  if !u.isValid then
    .error "cart-pole input must have one finite cart force coordinate"
  if !x.isValid then
    .error "cart-pole state must have four finite coordinates"

def fullPhysicsPrimitives
    (p : CartPoleParams)
    (u : CartPoleInput)
    (x : CartPoleState)
    (label : String := "cart-pole") : FullPhysicsPrimitives :=
  {
    massMatrix := massMatrix p x
    qdot := qdotAsArray x
    actuationForces := inputGeneralizedForces u
    biasForces := dynamicsBiasTerm p x
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := label
  }

def fullPhysicsPrimitiveProvider
    (p : CartPoleParams := params)
    (label : String := "cart-pole full physics provider") :
    FullPhysicsPrimitiveProvider CartPolePhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      validateFullPhysicsInputs? p snapshot.input snapshot.state
      pure (fullPhysicsPrimitives p snapshot.input snapshot.state label)
  }

def solveFullPhysics?
    (p : CartPoleParams)
    (u : CartPoleInput)
    (x : CartPoleState)
    (intervalVertex : VertexId := 5352)
    (label : String := "cart-pole") :
    Except String FullPhysicsResult := do
  validateFullPhysicsInputs? p u x
  (fullPhysicsPrimitives p u x label).solve? intervalVertex

def derivative? (p : CartPoleParams) (u : CartPoleInput) (x : CartPoleState) :
    Except String CartPoleState := do
  let d ← (manipulatorEquation p u x).solve?
  pure {
    x := d.qdot.getD 0 0.0
    theta := d.qdot.getD 1 0.0
    xdot := d.vdot.getD 0 0.0
    thetadot := d.vdot.getD 1 0.0
  }

def derivative (p : CartPoleParams) (u : CartPoleInput := defaultInput)
    (x : CartPoleState) : CartPoleState :=
  match derivative? p u x with
  | .ok dx => dx
  | .error _ => {}

def proposedDerivativeArrays (dx : CartPoleState) : Array Float × Array Float :=
  (#[dx.x, dx.theta], #[dx.xdot, dx.thetadot])

def implicitResidual
    (p : CartPoleParams)
    (u : CartPoleInput)
    (x : CartPoleState)
    (proposed : CartPoleState) : Array Float :=
  let (proposedQdot, proposedVdot) := proposedDerivativeArrays proposed
  let qResidual := FloatArray.sub proposedQdot (qdotAsArray x)
  let forceResidual :=
    FloatArray.sub
      (FloatMatrix.matVec (massMatrix p x) proposedVdot)
      (FloatArray.sub (inputGeneralizedForces u) (dynamicsBiasTerm p x))
  qResidual ++ forceResidual

def kineticEnergy (p : CartPoleParams) (x : CartPoleState) : Float :=
  let v := qdotAsArray x
  0.5 * FloatArray.dot v (FloatMatrix.matVec (massMatrix p x) v)

def potentialEnergy (p : CartPoleParams) (x : CartPoleState) : Float :=
  -p.mp * p.gravity * p.l * Float.cos x.theta

def totalEnergy (p : CartPoleParams) (x : CartPoleState) : Float :=
  kineticEnergy p x + potentialEnergy p x

def odeTerm (p : CartPoleParams) : ODETerm CartPoleState CartPoleInput :=
  { vectorField := fun _t x u => derivative p u x }

def cartPoleSolver :=
  RK4.solver
    (Term := ODETerm CartPoleState CartPoleInput)
    (Y := CartPoleState)
    (VF := CartPoleState)
    (Args := CartPoleInput)

structure MultibodyCartPoleConfig where
  targetRealtimeRate : Float := 1.0
  simulationTime : Float := 10.0
  timeStep : Float := 0.0
  visualizationEnabled : Bool := true
  deriving Repr, Inhabited

namespace MultibodyCartPoleConfig

def validate? (cfg : MultibodyCartPoleConfig) : Except String Unit := do
  if !cfg.targetRealtimeRate.isFinite || cfg.targetRealtimeRate < 0.0 then
    .error s!"cart-pole target_realtime_rate must be nonnegative and finite, got {cfg.targetRealtimeRate}"
  if !cfg.simulationTime.isFinite || cfg.simulationTime <= 0.0 then
    .error s!"cart-pole simulation_time must be positive and finite, got {cfg.simulationTime}"
  if !cfg.timeStep.isFinite || cfg.timeStep < 0.0 then
    .error s!"cart-pole time_step must be nonnegative and finite, got {cfg.timeStep}"

end MultibodyCartPoleConfig

def multibodyCartPoleConfig : MultibodyCartPoleConfig := {}

def multibodyCartPoleModel : FullMultibodyPlantModel :=
  {
    modelName := "CartPole"
    modelUri := cartPoleModelUri
    numPositions := 2
    numVelocities := 2
    numActuatedDofs := 1
    finalized := true
    label := "parsed cart-pole SDF model"
  }

def multibodyCartPolePlantConfig (cfg : MultibodyCartPoleConfig) :
    MultibodyPlantConfigPrimitive :=
  {
    timeStep := cfg.timeStep
    penetrationAllowance := 0.0
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def multibodyCartPolePassiveStep
    (cfg : MultibodyCartPoleConfig := multibodyCartPoleConfig) :
    FullMultibodyPlantStep :=
  {
    model := multibodyCartPoleModel
    config := multibodyCartPolePlantConfig cfg
    q0 := #[0.0, 2.0]
    v0 := #[0.0, 0.0]
    actuation := #[0.0]
    t0 := 0.0
    t1 := cfg.simulationTime
    label := "multibody-cart-pole-passive-full-plant"
  }

private def cartPoleLocalMove (vertex : VertexId) (label : String) :
    SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := .exact
    label := label
  }

def multibodyCartPoleMoves (cfg : MultibodyCartPoleConfig) :
    Array SkeletonMove :=
  #[
    cartPoleLocalMove 5350 "Parser.AddModelsFromUrl cart_pole.sdf + MultibodyPlant.Finalize",
    cartPoleLocalMove 5351
      s!"AddMultibodyPlantSceneGraph time_step={cfg.timeStep}, visualization={cfg.visualizationEnabled}"
  ]

structure MultibodyCartPoleResult where
  references : Array DrakeReference
  asset : CartPoleModelAssetBoundary
  config : MultibodyCartPoleConfig
  step : FullMultibodyPlantStep
  fullPhysics : FullPhysicsResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildMultibodyCartPole?
    (cfg : MultibodyCartPoleConfig := multibodyCartPoleConfig)
    (asset : CartPoleModelAssetBoundary := cartPoleModelAssetBoundary) :
    Except String MultibodyCartPoleResult := do
  cfg.validate?
  asset.validate?
  let step := multibodyCartPolePassiveStep cfg
  step.validate?
  let x0 : CartPoleState := {
    x := step.q0.getD 0 0.0
    theta := step.q0.getD 1 0.0
    xdot := step.v0.getD 0 0.0
    thetadot := step.v0.getD 1 0.0
  }
  let u0 : CartPoleInput := { cartForce := step.actuation.getD 0 0.0 }
  let fullPhysics ← solveFullPhysics? params u0 x0 5352
    "multibody cart-pole passive benchmark plant"
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval {
        id := 5352
        attemptIndex := 0
        tStart := 0.0
        tAttempt := cfg.simulationTime
        tAfter := cfg.simulationTime
        label := "multibody cart-pole passive Simulator.AdvanceTo"
      })
  trace.validate?
  pure {
    references := drakeReferences
    asset := asset
    config := cfg
    step := step
    fullPhysics := fullPhysics
    trace := trace
    moves := multibodyCartPoleMoves cfg ++ #[fullPhysics.supportMove, fullPhysics.move] ++ trace.moves
  }

structure SimulationResult where
  references : Array DrakeReference
  t0 : Float
  t1 : Float
  input : CartPoleInput
  initialState : CartPoleState
  finalState : CartPoleState
  initialEnergy : Float
  finalEnergy : Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def acceptedSegment (t0 t1 : Float) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := t0
    tAttempt := t1
    tAfter := t1
    label := "cart-pole-continuous-interval"
  }

def solvePassive? (p : CartPoleParams := params)
    (x0 : CartPoleState := { x := 0.0, theta := 2.0, xdot := 0.0, thetadot := 0.0 })
    (t0 : Float := 0.0)
    (t1 : Float := 0.1)
    (u : CartPoleInput := defaultInput) :
    Except String SimulationResult := do
  if !p.isValid then
    .error "cart-pole params are invalid"
  if !u.isValid then
    .error "cart-pole input is invalid"
  if !x0.isValid then
    .error "cart-pole initial state is invalid"
  let sol :=
    diffeqsolve
      (Term := ODETerm CartPoleState CartPoleInput)
      (Y := CartPoleState)
      (VF := CartPoleState)
      (Control := Time)
      (Args := CartPoleInput)
      (Controller := ConstantStepSize)
      (odeTerm p) cartPoleSolver t0 t1 (some p.stepSize) x0 u
      (saveat := { t1 := true })
  if !sol.result.isOkay then
    .error s!"cart-pole solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "cart-pole solve did not save endpoint"
        else
          let final := ys[ys.size - 1]!
          let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment t0 ts[ts.size - 1]!))
          trace.validate?
          pure {
            references := drakeReferences
            t0 := t0
            t1 := ts[ts.size - 1]!
            input := u
            initialState := x0
            finalState := final
            initialEnergy := totalEnergy p x0
            finalEnergy := totalEnergy p final
            trace := trace
            moves := trace.moves
          }
    | _, _ => .error "cart-pole solve did not save endpoint arrays"

def buildEndToEnd? : Except String SimulationResult :=
  solvePassive?

end Tyr.EventSkeleton.Examples.CartPole
