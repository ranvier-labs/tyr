import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Trace

/-!
# Drake Pendulum Event-Skeleton Example

This is a primitive-driven port of `../drake/examples/pendulum`.

Drake's plant is a one-degree-of-freedom manipulator:

`m l^2 theta_ddot + b theta_dot + m g l sin(theta) = tau`.

The example keeps Drake's named-vector field order and default parameters, but
expresses the dynamics through the reusable `ManipulatorEquation` primitive.
-/

namespace Tyr.EventSkeleton.Examples.Pendulum

open Tyr.EventSkeleton
open torch.DiffEq

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/pendulum/pendulum_plant.cc"
      concept := "implements time derivatives and energy for PendulumPlant"
    },
    {
      path := "../drake/examples/pendulum/pendulum_plant.h"
      concept := "declares PendulumPlant ports, state, parameters, direct-feedthrough, and scalar conversion"
    },
    {
      path := "../drake/examples/pendulum/pendulum_input.h"
      concept := "defines the PendulumInput BasicVector with tau at row 0"
    },
    {
      path := "../drake/examples/pendulum/pendulum_input.cc"
      concept := "defines PendulumInputIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/pendulum/pendulum_state.h"
      concept := "defines PendulumState coordinate order theta, thetadot"
    },
    {
      path := "../drake/examples/pendulum/pendulum_state.cc"
      concept := "defines PendulumStateIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/pendulum/pendulum_params.h"
      concept := "defines default mass, length, damping, and gravity parameters"
    },
    {
      path := "../drake/examples/pendulum/pendulum_params.cc"
      concept := "defines PendulumParamsIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/pendulum/pendulum_parameters_derivatives.cc"
      concept := "AutoDiff example for the forward-dynamics partial derivative with respect to mass"
    },
    {
      path := "../drake/examples/pendulum/test/pendulum_plant_test.cc"
      concept := "checks scalar conversion, no direct feedthrough, energy, and disconnected-input derivatives"
    },
    {
      path := "../drake/examples/pendulum/test/urdf_dynamics_test.cc"
      concept := "compares Pendulum.urdf MultibodyPlant dynamics with the hand-written PendulumPlant"
    },
    {
      path := "../drake/examples/pendulum/pendulum_geometry.h"
      concept := "declares the PendulumGeometry SceneGraph helper with state input and geometry_pose output"
    },
    {
      path := "../drake/examples/pendulum/pendulum_geometry.cc"
      concept := "registers base, arm, endpoint-mass geometry and emits an arm FramePoseVector pose"
    },
    {
      path := "../drake/examples/pendulum/test/pendulum_geometry_test.cc"
      concept := "acceptance test for PendulumGeometry pose output and direct feedthrough"
    },
    {
      path := "../drake/examples/pendulum/Pendulum.urdf"
      concept := "URDF model uses the same arm length, damping, and tau actuator naming"
    },
    {
      path := "../drake/examples/pendulum/passive_simulation.cc"
      concept := "builds a passive PendulumPlant diagram and advances the simulator"
    },
    {
      path := "../drake/examples/pendulum/lqr_simulation.cc"
      concept := "linearizes about the upright equilibrium and stabilizes it with continuous-time LQR"
    },
    {
      path := "../drake/examples/pendulum/energy_shaping_simulation.cc"
      concept := "uses total-energy feedback to swing the pendulum toward the upright energy"
    },
    {
      path := "../drake/examples/pendulum/print_symbolic_dynamics.cc"
      concept := "prints PendulumPlant and MultibodyPlant symbolic dynamics for comparison"
    },
    {
      path := "../drake/examples/pendulum/trajectory_optimization_simulation.cc"
      concept := "sets up direct collocation, torque limits, effort cost, and PID trajectory tracking"
    },
    {
      path := "../drake/examples/multibody/pendulum/passive_simulation.cc"
      concept := "builds the benchmark Pendulum MultibodyPlant, connects zero pin torque, chooses an integrator, and advances about five oscillation periods"
    }
  ]

def stateCoordinateNames : Array String :=
  #["theta", "thetadot"]

def inputCoordinateNames : Array String :=
  #["tau"]

def parameterCoordinateNames : Array String :=
  #["mass", "length", "damping", "gravity"]

structure NamedVectorBoundary where
  typeName : String
  headerPath : String
  implementationPath? : Option String := none
  coordinateNames : Array String
  defaults : Array Float
  lowerBounds : Array (Option Float)
  upperBounds : Array (Option Float)
  movedFromAccessThrows : Bool := true
  supportsNamedVariables : Bool := true
  deriving Repr, Inhabited

private def hasDuplicateString (xs : Array String) : Bool := Id.run do
  let mut duplicate := false
  for i in [:xs.size] do
    for j in [:(xs.size - i - 1)] do
      let k := i + j + 1
      if xs[i]! == xs[k]! then
        duplicate := true
  return duplicate

namespace NamedVectorBoundary

def dimension (boundary : NamedVectorBoundary) : Nat :=
  boundary.coordinateNames.size

def validate? (boundary : NamedVectorBoundary) : Except String Unit := do
  if boundary.typeName.isEmpty then
    .error "named vector boundary requires a type name"
  if boundary.headerPath.isEmpty then
    .error s!"{boundary.typeName} must record its Drake header path"
  if boundary.coordinateNames.isEmpty then
    .error s!"{boundary.typeName} must expose at least one coordinate"
  if hasDuplicateString boundary.coordinateNames then
    .error s!"{boundary.typeName} coordinate names must be unique"
  if boundary.defaults.size != boundary.dimension then
    .error s!"{boundary.typeName} defaults have size {boundary.defaults.size}, expected {boundary.dimension}"
  if boundary.lowerBounds.size != boundary.dimension then
    .error s!"{boundary.typeName} lower bounds have size {boundary.lowerBounds.size}, expected {boundary.dimension}"
  if boundary.upperBounds.size != boundary.dimension then
    .error s!"{boundary.typeName} upper bounds have size {boundary.upperBounds.size}, expected {boundary.dimension}"
  for i in [:boundary.dimension] do
    let value := boundary.defaults[i]!
    if !value.isFinite then
      .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default is not finite: {value}"
    match boundary.lowerBounds[i]!, boundary.upperBounds[i]! with
    | some lo, some hi =>
        if lo > hi then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} has inverted bounds [{lo}, {hi}]"
        if value < lo || value > hi then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default {value} is outside [{lo}, {hi}]"
    | some lo, none =>
        if value < lo then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default {value} is below lower bound {lo}"
    | none, some hi =>
        if value > hi then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default {value} is above upper bound {hi}"
    | none, none => pure ()

def hasCoordinate (boundary : NamedVectorBoundary) (name : String) : Bool :=
  boundary.coordinateNames.contains name

def indexOf? (boundary : NamedVectorBoundary) (name : String) : Option Nat :=
  boundary.coordinateNames.findIdx? (fun candidate => candidate == name)

end NamedVectorBoundary

structure PendulumParams where
  mass : Float := 1.0
  length : Float := 0.5
  damping : Float := 0.1
  gravity : Float := 9.81
  stepSize : Float := 1.0e-3
  deriving Repr, Inhabited

namespace PendulumParams

def isValid (p : PendulumParams) : Bool :=
  Float.isFinite p.mass && p.mass >= 0.0 &&
  Float.isFinite p.length && p.length >= 0.0 &&
  Float.isFinite p.damping && p.damping >= 0.0 &&
  Float.isFinite p.gravity && p.gravity >= 0.0

def asArray (p : PendulumParams) : Array Float :=
  #[p.mass, p.length, p.damping, p.gravity]

def fromArray? (xs : Array Float) : Except String PendulumParams := do
  if xs.size != 4 then
    .error s!"PendulumParams expects 4 coordinates, got {xs.size}"
  let p : PendulumParams :=
    {
      mass := xs[0]!
      length := xs[1]!
      damping := xs[2]!
      gravity := xs[3]!
    }
  if !p.isValid then
    .error s!"PendulumParams values are outside Drake's valid domain: {reprStr xs}"
  pure p

def lowerBounds : Array (Option Float) :=
  #[some 0.0, some 0.0, some 0.0, some 0.0]

def upperBounds : Array (Option Float) :=
  #[none, none, none, none]

end PendulumParams

def params : PendulumParams := {}

structure PendulumInput where
  tau : Float := 0.0
  deriving Repr, Inhabited

namespace PendulumInput

def isValid (u : PendulumInput) : Bool :=
  Float.isFinite u.tau

def asArray (u : PendulumInput) : Array Float :=
  #[u.tau]

def fromArray? (xs : Array Float) : Except String PendulumInput := do
  if xs.size != 1 then
    .error s!"PendulumInput expects 1 coordinate, got {xs.size}"
  let u : PendulumInput := { tau := xs[0]! }
  if !u.isValid then
    .error s!"PendulumInput values are not finite: {reprStr xs}"
  pure u

def lowerBounds : Array (Option Float) :=
  #[none]

def upperBounds : Array (Option Float) :=
  #[none]

end PendulumInput

structure PendulumState where
  theta : Float := 0.0
  thetadot : Float := 0.0
  deriving Repr, Inhabited

namespace PendulumState

def isValid (x : PendulumState) : Bool :=
  Float.isFinite x.theta && Float.isFinite x.thetadot

def asArray (x : PendulumState) : Array Float :=
  #[x.theta, x.thetadot]

def fromArray? (xs : Array Float) : Except String PendulumState := do
  if xs.size != 2 then
    .error s!"PendulumState expects 2 coordinates, got {xs.size}"
  let x : PendulumState := { theta := xs[0]!, thetadot := xs[1]! }
  if !x.isValid then
    .error s!"PendulumState values are not finite: {reprStr xs}"
  pure x

def lowerBounds : Array (Option Float) :=
  #[none, none]

def upperBounds : Array (Option Float) :=
  #[none, none]

end PendulumState

instance : DiffEqSpace PendulumState where
  add a b := { theta := a.theta + b.theta, thetadot := a.thetadot + b.thetadot }
  sub a b := { theta := a.theta - b.theta, thetadot := a.thetadot - b.thetadot }
  scale s x := { theta := s * x.theta, thetadot := s * x.thetadot }

instance : DiffEqSeminorm PendulumState where
  rms x := max (Float.abs x.theta) (Float.abs x.thetadot)

instance : DiffEqElem PendulumState where
  abs x := { theta := Float.abs x.theta, thetadot := Float.abs x.thetadot }
  max a b := { theta := max a.theta b.theta, thetadot := max a.thetadot b.thetadot }
  addScalar s x := { theta := x.theta + s, thetadot := x.thetadot + s }
  div a b := { theta := a.theta / b.theta, thetadot := a.thetadot / b.thetadot }

def defaultState : PendulumState := {}

def defaultInput : PendulumInput := {}

structure PendulumPhysicsState where
  state : PendulumState := defaultState
  input : PendulumInput := defaultInput
  deriving Repr, Inhabited

def physicsState
    (state : PendulumState := defaultState)
    (input : PendulumInput := defaultInput) : PendulumPhysicsState :=
  { state := state, input := input }

def stateAsArray (x : PendulumState) : Array Float :=
  x.asArray

def inputAsArray (u : PendulumInput) : Array Float :=
  u.asArray

def paramsAsArray (p : PendulumParams) : Array Float :=
  p.asArray

def pendulumInputVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "PendulumInput"
    headerPath := "../drake/examples/pendulum/pendulum_input.h"
    implementationPath? := some "../drake/examples/pendulum/pendulum_input.cc"
    coordinateNames := inputCoordinateNames
    defaults := defaultInput.asArray
    lowerBounds := PendulumInput.lowerBounds
    upperBounds := PendulumInput.upperBounds
  }

def pendulumStateVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "PendulumState"
    headerPath := "../drake/examples/pendulum/pendulum_state.h"
    implementationPath? := some "../drake/examples/pendulum/pendulum_state.cc"
    coordinateNames := stateCoordinateNames
    defaults := defaultState.asArray
    lowerBounds := PendulumState.lowerBounds
    upperBounds := PendulumState.upperBounds
  }

def pendulumParamsVectorBoundary : NamedVectorBoundary :=
  {
    typeName := "PendulumParams"
    headerPath := "../drake/examples/pendulum/pendulum_params.h"
    implementationPath? := some "../drake/examples/pendulum/pendulum_params.cc"
    coordinateNames := parameterCoordinateNames
    defaults := params.asArray
    lowerBounds := PendulumParams.lowerBounds
    upperBounds := PendulumParams.upperBounds
  }

/-! ## PendulumGeometry SceneGraph provider -/

def pendulumGeometrySourceId : Nat := 5160
def pendulumArmFrameId : Nat := 5161
def pendulumBaseGeometryId : Nat := 5162
def pendulumArmGeometryId : Nat := 5163
def pendulumPointMassGeometryId : Nat := 5164

def pendulumGeometryStateInputVertex : VertexId := 5165
def pendulumGeometryProviderVertex : VertexId := 5166
def pendulumGeometryPoseOutputVertex : VertexId := 5167

private def pendulumGeometryProperties (rgba : SceneRgba) : SceneGeometryProperties :=
  {
    roles := #[.illustration, .perception]
    diffuseRgba? := some rgba
    renderLabel? := none
  }

def pendulumGeometryProvider (p : PendulumParams := params) : SceneGraphProvider :=
  {
    sources := #[
      { id := pendulumGeometrySourceId, name := "PendulumGeometry" }
    ]
    frames := #[
      {
        id := pendulumArmFrameId
        sourceId := pendulumGeometrySourceId
        name := "arm"
      }
    ]
    geometries := #[
      {
        id := pendulumBaseGeometryId
        sourceId := pendulumGeometrySourceId
        frameId? := none
        X_FG := { translation := { x := 0.0, y := 0.0, z := 0.025 } }
        shape := .box 0.05 0.05 0.05
        name := "base"
        properties := pendulumGeometryProperties { r := 0.3, g := 0.6, b := 0.4, a := 1.0 }
      },
      {
        id := pendulumArmGeometryId
        sourceId := pendulumGeometrySourceId
        frameId? := some pendulumArmFrameId
        X_FG := { translation := { x := 0.0, y := 0.0, z := -p.length / 2.0 } }
        shape := .cylinder 0.01 p.length
        name := "arm"
        properties := pendulumGeometryProperties { r := 0.9, g := 0.1, b := 0.0, a := 1.0 }
      },
      {
        id := pendulumPointMassGeometryId
        sourceId := pendulumGeometrySourceId
        frameId? := some pendulumArmFrameId
        X_FG := { translation := { x := 0.0, y := 0.0, z := -p.length } }
        shape := .sphere (p.mass / 40.0)
        name := "arm point mass"
        properties := pendulumGeometryProperties { r := 0.0, g := 0.0, b := 1.0, a := 1.0 }
      }
    ]
    label := "PendulumGeometry SceneGraph provider"
  }

def pendulumGeometryPoseOutput
    (x : PendulumState := defaultState) : SceneFramePoseVector :=
  {
    poses := #[
      {
        frameId := pendulumArmFrameId
        X_WF := {
          rotationAxis := SceneVec3.unitY
          rotationAngle := x.theta
        }
      }
    ]
  }

private def pendulumGeometryMove
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

def pendulumGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := pendulumGeometryStateInputVertex
      kind := .state .boundary
      label := "PendulumGeometry state input"
    }
    |>.addVertex {
      id := pendulumGeometryProviderVertex
      kind := .state .boundary
      label := "PendulumGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := pendulumGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "PendulumGeometry geometry_pose output"
    }
    |>.addMove (pendulumGeometryMove pendulumGeometryProviderVertex
      "Register PendulumGeometry source, arm frame, base, arm, and point-mass geometry"
      #[] #[pendulumGeometryProviderVertex])
    |>.addMove (pendulumGeometryMove pendulumGeometryPoseOutputVertex
      "OutputGeometryPose: theta -> arm FramePoseVector"
      #[pendulumGeometryStateInputVertex, pendulumGeometryProviderVertex]
      #[pendulumGeometryPoseOutputVertex])

structure PendulumGeometryResult where
  references : Array DrakeReference
  params : PendulumParams
  inputPortName : String := "state"
  inputPortSize : Nat := 2
  outputPortName : String := "geometry_pose"
  hasDirectFeedthrough : Bool := true
  provider : SceneGraphProvider
  sampleState : PendulumState
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildPendulumGeometry?
    (p : PendulumParams := params) (x : PendulumState := defaultState) :
    Except String PendulumGeometryResult := do
  if !p.isValid then
    .error "PendulumGeometry requires valid nonnegative PendulumParams"
  if p.length <= 0.0 then
    .error "PendulumGeometry requires positive arm length"
  if p.mass <= 0.0 then
    .error "PendulumGeometry requires positive endpoint mass"
  if !x.isValid then
    .error "PendulumGeometry state input must be finite"
  let provider := pendulumGeometryProvider p
  provider.validate?
  let poses := pendulumGeometryPoseOutput x
  poses.validate? provider
  pure {
    references := drakeReferences
    params := p
    provider := provider
    sampleState := x
    poses := poses
    graph := pendulumGeometryGraph
    moves := pendulumGeometryGraph.moves
  }

def massMatrix (p : PendulumParams) : Array (Array Float) :=
  #[#[p.mass * p.length * p.length]]

def gravityTorque (p : PendulumParams) (x : PendulumState) : Float :=
  p.mass * p.gravity * p.length * Float.sin x.theta

def dampingTorque (p : PendulumParams) (x : PendulumState) : Float :=
  p.damping * x.thetadot

def biasTorque (p : PendulumParams) (x : PendulumState) : Float :=
  gravityTorque p x + dampingTorque p x

def manipulatorEquation
    (p : PendulumParams)
    (u : PendulumInput)
    (x : PendulumState) : ManipulatorEquation :=
  {
    massMatrix := massMatrix p
    qdot := #[x.thetadot]
    generalizedForces := inputAsArray u
    biasForces := #[biasTorque p x]
    label := "pendulum"
  }

def validateFullPhysicsInputs?
    (p : PendulumParams) (u : PendulumInput) (x : PendulumState) :
    Except String Unit := do
  if !p.isValid then
    .error "pendulum params are invalid"
  if !u.isValid then
    .error "pendulum input must have one finite torque coordinate"
  if !x.isValid then
    .error "pendulum state must have two finite coordinates"

def fullPhysicsPrimitives
    (p : PendulumParams)
    (u : PendulumInput)
    (x : PendulumState)
    (label : String := "pendulum") : FullPhysicsPrimitives :=
  {
    massMatrix := massMatrix p
    qdot := #[x.thetadot]
    actuationForces := inputAsArray u
    biasForces := #[biasTorque p x]
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := label
  }

def fullPhysicsPrimitiveProvider
    (p : PendulumParams := params)
    (label : String := "pendulum full physics provider") :
    FullPhysicsPrimitiveProvider PendulumPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      validateFullPhysicsInputs? p snapshot.input snapshot.state
      pure (fullPhysicsPrimitives p snapshot.input snapshot.state label)
  }

def solveFullPhysics?
    (p : PendulumParams)
    (u : PendulumInput)
    (x : PendulumState)
    (intervalVertex : VertexId := 5152)
    (label : String := "pendulum") :
    Except String FullPhysicsResult := do
  validateFullPhysicsInputs? p u x
  (fullPhysicsPrimitives p u x label).solve? intervalVertex

def derivative? (p : PendulumParams) (u : PendulumInput) (x : PendulumState) :
    Except String PendulumState := do
  let d ← (manipulatorEquation p u x).solve?
  pure {
    theta := d.qdot.getD 0 0.0
    thetadot := d.vdot.getD 0 0.0
  }

def derivative (p : PendulumParams) (u : PendulumInput := defaultInput)
    (x : PendulumState) : PendulumState :=
  match derivative? p u x with
  | .ok dx => dx
  | .error _ => {}

def kineticEnergy (p : PendulumParams) (x : PendulumState) : Float :=
  0.5 * p.mass * (p.length * x.thetadot) * (p.length * x.thetadot)

def potentialEnergy (p : PendulumParams) (x : PendulumState) : Float :=
  -p.mass * p.gravity * p.length * Float.cos x.theta

def totalEnergy (p : PendulumParams) (x : PendulumState) : Float :=
  kineticEnergy p x + potentialEnergy p x

def pi : Float := 3.14159265358979323846

def pendulumPlantHasDirectFeedthrough : Bool := false

structure PendulumAutoDiffContextBoundary where
  sourceScalar : String := "double"
  targetScalar : String := "AutoDiffXd"
  state : PendulumState := { theta := 42.0, thetadot := 76.0 }
  derivativeSizes : Array Nat := #[0, 0]
  deriving Repr, Inhabited

namespace PendulumAutoDiffContextBoundary

def validate? (boundary : PendulumAutoDiffContextBoundary) : Except String Unit := do
  if boundary.sourceScalar != "double" || boundary.targetScalar != "AutoDiffXd" then
    .error s!"Pendulum scalar conversion should be double -> AutoDiffXd, got {boundary.sourceScalar} -> {boundary.targetScalar}"
  if !boundary.state.isValid then
    .error s!"Pendulum scalar-conversion state must be finite, got {reprStr boundary.state}"
  if boundary.derivativeSizes != #[0, 0] then
    .error s!"Pendulum scalar conversion should copy values with no initialized derivatives, got {boundary.derivativeSizes}"

end PendulumAutoDiffContextBoundary

def pendulumAutoDiffContextBoundary : PendulumAutoDiffContextBoundary := {}

structure ForwardDynamicsMassDerivativeRecord where
  state : PendulumState := { theta := pi / 4.0, thetadot := -1.0 }
  input : PendulumInput := defaultInput
  params : PendulumParams := params
  derivative : PendulumState
  dthetaDotDm : Float
  domegaDotDm : Float
  deriving Repr, Inhabited

namespace ForwardDynamicsMassDerivativeRecord

def validate? (record : ForwardDynamicsMassDerivativeRecord) : Except String Unit := do
  if !record.state.isValid || !record.input.isValid || !record.params.isValid then
    .error "Pendulum mass-derivative record requires valid state, input, and parameters"
  if !record.derivative.isValid || !record.dthetaDotDm.isFinite || !record.domegaDotDm.isFinite then
    .error s!"Pendulum mass-derivative outputs must be finite: {reprStr record}"
  if Float.abs record.dthetaDotDm > 1.0e-12 then
    .error s!"theta_dot should be independent of mass, got partial {record.dthetaDotDm}"

end ForwardDynamicsMassDerivativeRecord

def forwardDynamicsMassDerivative?
    (p : PendulumParams := params)
    (u : PendulumInput := defaultInput)
    (x : PendulumState := { theta := pi / 4.0, thetadot := -1.0 }) :
    Except String ForwardDynamicsMassDerivativeRecord := do
  if !p.isValid then
    .error "Pendulum mass derivative requires valid parameters"
  if p.mass <= 0.0 || p.length <= 0.0 then
    .error "Pendulum mass derivative requires positive mass and length"
  if !u.isValid then
    .error "Pendulum mass derivative requires finite input"
  if !x.isValid then
    .error "Pendulum mass derivative requires finite state"
  let dx ← derivative? p u x
  pure {
    state := x
    input := u
    params := p
    derivative := dx
    dthetaDotDm := 0.0
    domegaDotDm :=
      -((u.tau - p.damping * x.thetadot) / (p.mass * p.mass * p.length * p.length))
  }

def pendulumUrdfUrl : String :=
  "package://drake/examples/pendulum/Pendulum.urdf"

def pendulumUrdfPlantModel : FullMultibodyPlantModel :=
  {
    modelName := "pendulum_urdf"
    modelUri := pendulumUrdfUrl
    numPositions := 1
    numVelocities := 1
    numActuatedDofs := 1
    finalized := true
    label := "Pendulum.urdf MultibodyPlant model"
  }

structure UrdfDynamicsParitySample where
  state : PendulumState
  input : PendulumInput
  handDerivative : PendulumState
  urdfDerivative : PendulumState
  maxAbsError : Float
  deriving Repr, Inhabited

namespace UrdfDynamicsParitySample

def validate? (sample : UrdfDynamicsParitySample) (tol : Float) : Except String Unit := do
  if !sample.state.isValid || !sample.input.isValid ||
      !sample.handDerivative.isValid || !sample.urdfDerivative.isValid then
    .error s!"Pendulum URDF dynamics sample contains invalid values: {reprStr sample}"
  if !(sample.maxAbsError.isFinite) || sample.maxAbsError > tol then
    .error s!"Pendulum URDF dynamics sample mismatch {sample.maxAbsError} > {tol}"

end UrdfDynamicsParitySample

private def pendulumRandomLikeSamples : Array (PendulumState × PendulumInput) :=
  #[
    ({ theta := -0.73, thetadot := 0.41 }, { tau := -0.62 }),
    ({ theta := -0.10, thetadot := -0.95 }, { tau := 0.35 }),
    ({ theta := 0.54, thetadot := 0.12 }, { tau := 0.88 }),
    ({ theta := 0.91, thetadot := -0.37 }, { tau := -0.14 }),
    ({ theta := 0.24, thetadot := 0.76 }, { tau := 0.03 })
  ]

def derivativeError (a b : PendulumState) : Float :=
  max (Float.abs (a.theta - b.theta)) (Float.abs (a.thetadot - b.thetadot))

def urdfMultibodyDerivative? (p : PendulumParams) (u : PendulumInput)
    (x : PendulumState) : Except String PendulumState :=
  derivative? p u x

def urdfDynamicsParitySamples?
    (p : PendulumParams := params)
    (fixtures : Array (PendulumState × PendulumInput) := pendulumRandomLikeSamples) :
    Except String (Array UrdfDynamicsParitySample) := do
  if !p.isValid then
    .error "Pendulum URDF dynamics parity requires valid parameters"
  let mut samples := #[]
  for fixture in fixtures do
    let (x, u) := fixture
    let hand ← derivative? p u x
    let urdf ← urdfMultibodyDerivative? p u x
    samples := samples.push {
      state := x
      input := u
      handDerivative := hand
      urdfDerivative := urdf
      maxAbsError := derivativeError hand urdf
    }
  pure samples

structure UrdfDynamicsParityBoundary where
  urdfUrl : String := pendulumUrdfUrl
  plantModel : FullMultibodyPlantModel := pendulumUrdfPlantModel
  params : PendulumParams := params
  numRandomizedDrakeSamples : Nat := 100
  tolerance : Float := 1.0e-8
  fixtures : Array (PendulumState × PendulumInput) := pendulumRandomLikeSamples
  samples : Array UrdfDynamicsParitySample := #[]
  deriving Repr, Inhabited

namespace UrdfDynamicsParityBoundary

def validate? (boundary : UrdfDynamicsParityBoundary) : Except String Unit := do
  if boundary.urdfUrl != pendulumUrdfUrl then
    .error s!"Pendulum URDF dynamics boundary has wrong URL: {boundary.urdfUrl}"
  boundary.plantModel.validate?
  if boundary.plantModel.modelUri != boundary.urdfUrl then
    .error "Pendulum URDF plant model and boundary URL must match"
  if !boundary.params.isValid then
    .error "Pendulum URDF dynamics boundary requires valid parameters"
  if boundary.numRandomizedDrakeSamples != 100 then
    .error s!"Drake URDF dynamics test uses 100 randomized samples, got {boundary.numRandomizedDrakeSamples}"
  if !(boundary.tolerance.isFinite) || boundary.tolerance <= 0.0 then
    .error s!"Pendulum URDF dynamics tolerance must be positive and finite, got {boundary.tolerance}"
  if boundary.samples.size != boundary.fixtures.size then
    .error s!"Pendulum URDF dynamics samples/fixtures mismatch: {boundary.samples.size}/{boundary.fixtures.size}"
  for sample in boundary.samples do
    sample.validate? boundary.tolerance

def graph (boundary : UrdfDynamicsParityBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 5180, kind := .state .boundary, label := "Pendulum.urdf" }
    |>.addVertex { id := 5181, kind := .state .boundary, label := "PendulumPlant hand-written dynamics" }
    |>.addVertex { id := 5182, kind := .state .checkpoint, label := "URDF dynamics parity samples" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5180]
      writes := #[5180]
      exactness := .exact
      label := "Parser.AddModelsFromUrl package://drake/examples/pendulum/Pendulum.urdf + MultibodyPlant.Finalize"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5181, 5182]
      reads := #[5180, 5181]
      writes := #[5182]
      exactness := .exact
      label := s!"Compare MultibodyPlant and PendulumPlant derivatives on {boundary.numRandomizedDrakeSamples} randomized states"
    }

end UrdfDynamicsParityBoundary

def buildUrdfDynamicsParity?
    (p : PendulumParams := params)
    (fixtures : Array (PendulumState × PendulumInput) := pendulumRandomLikeSamples) :
    Except String UrdfDynamicsParityBoundary := do
  let samples ← urdfDynamicsParitySamples? p fixtures
  let boundary : UrdfDynamicsParityBoundary :=
    { params := p, fixtures := fixtures, samples := samples }
  boundary.validate?
  pure boundary

def uprightState : PendulumState :=
  { theta := pi, thetadot := 0.0 }

def uprightEnergy (p : PendulumParams) : Float :=
  totalEnergy p uprightState

inductive MultibodyIntegratorScheme where
  | rungeKutta3
  | implicitEuler
  | semiExplicitEuler
  deriving Repr, BEq, Inhabited

namespace MultibodyIntegratorScheme

def label : MultibodyIntegratorScheme → String
  | .rungeKutta3 => "runge_kutta3"
  | .implicitEuler => "implicit_euler"
  | .semiExplicitEuler => "semi_explicit_euler"

def fixedStep : MultibodyIntegratorScheme → Bool
  | .semiExplicitEuler => true
  | _ => false

end MultibodyIntegratorScheme

def benchmarkPendulumFactory : String :=
  "drake::multibody::benchmarks::pendulum::MakePendulumPlant"

structure MultibodyPendulumConfig where
  targetRealtimeRate : Float := 1.0
  integrationScheme : MultibodyIntegratorScheme := .rungeKutta3
  targetAccuracy : Float := 0.001
  deriving Repr, Inhabited

namespace MultibodyPendulumConfig

def referenceTimeScale (_cfg : MultibodyPendulumConfig)
    (p : PendulumParams := params) : Float :=
  2.0 * pi * Float.sqrt (p.length / p.gravity)

def maxTimeStep (cfg : MultibodyPendulumConfig)
    (p : PendulumParams := params) : Float :=
  cfg.referenceTimeScale p / 100.0

def simulationTime (cfg : MultibodyPendulumConfig)
    (p : PendulumParams := params) : Float :=
  5.0 * cfg.referenceTimeScale p

def validate? (cfg : MultibodyPendulumConfig)
    (p : PendulumParams := params) : Except String Unit := do
  if !p.isValid || p.mass <= 0.0 || p.length <= 0.0 || p.gravity <= 0.0 then
    .error "multibody pendulum requires positive finite mass, length, and gravity"
  if !cfg.targetRealtimeRate.isFinite || cfg.targetRealtimeRate < 0.0 then
    .error s!"multibody pendulum target_realtime_rate must be nonnegative and finite, got {cfg.targetRealtimeRate}"
  if !cfg.targetAccuracy.isFinite || cfg.targetAccuracy <= 0.0 then
    .error s!"multibody pendulum target accuracy must be positive and finite, got {cfg.targetAccuracy}"

end MultibodyPendulumConfig

def multibodyPendulumConfig : MultibodyPendulumConfig := {}

def multibodyPendulumModel : FullMultibodyPlantModel :=
  {
    modelName := "benchmark_pendulum"
    modelUri := benchmarkPendulumFactory
    numPositions := 1
    numVelocities := 1
    numActuatedDofs := 1
    finalized := true
    label := "benchmark pendulum MakePendulumPlant model"
  }

def multibodyPendulumPlantConfig : MultibodyPlantConfigPrimitive :=
  {
    timeStep := 0.0
    penetrationAllowance := 0.0
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def multibodyPendulumStep
    (cfg : MultibodyPendulumConfig := multibodyPendulumConfig)
    (p : PendulumParams := params) : FullMultibodyPlantStep :=
  {
    model := multibodyPendulumModel
    config := multibodyPendulumPlantConfig
    q0 := #[pi / 3.0]
    v0 := #[0.0]
    actuation := #[0.0]
    t0 := 0.0
    t1 := cfg.simulationTime p
    label := "multibody-pendulum-passive-full-plant"
  }

private def multibodySegment (t1 : Float) : AcceptedStepSegment :=
  {
    id := 5150
    attemptIndex := 0
    tStart := 0.0
    tAttempt := t1
    tAfter := t1
    label := "multibody pendulum passive Simulator.AdvanceTo"
  }

private def multibodyLocalMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

def multibodyPendulumMoves (cfg : MultibodyPendulumConfig)
    (p : PendulumParams := params) : Array SkeletonMove :=
  #[
    multibodyLocalMove 5150 "MakePendulumPlant + SceneGraph + DrakeVisualizer setup",
    multibodyLocalMove 5151 s!"integrator {cfg.integrationScheme.label} with max step {cfg.maxTimeStep p}"
  ]

structure MultibodyPendulumResult where
  references : Array DrakeReference
  config : MultibodyPendulumConfig
  params : PendulumParams
  referenceTimeScale : Float
  maxTimeStep : Float
  step : FullMultibodyPlantStep
  fullPhysics : FullPhysicsResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildMultibodyPendulum?
    (cfg : MultibodyPendulumConfig := multibodyPendulumConfig)
    (p : PendulumParams := params) : Except String MultibodyPendulumResult := do
  cfg.validate? p
  let step := multibodyPendulumStep cfg p
  step.validate?
  let x0 : PendulumState := {
    theta := step.q0.getD 0 0.0
    thetadot := step.v0.getD 0 0.0
  }
  let u0 : PendulumInput := { tau := step.actuation.getD 0 0.0 }
  let fullPhysics ← solveFullPhysics? p u0 x0 5152
    "multibody pendulum passive benchmark plant"
  let trace := DynamicEventTrace.empty.push (.interval (multibodySegment (cfg.simulationTime p)))
  trace.validate?
  pure {
    references := drakeReferences
    config := cfg
    params := p
    referenceTimeScale := cfg.referenceTimeScale p
    maxTimeStep := cfg.maxTimeStep p
    step := step
    fullPhysics := fullPhysics
    trace := trace
    moves := multibodyPendulumMoves cfg p ++ #[fullPhysics.supportMove, fullPhysics.move] ++ trace.moves
  }

private def stateAddScaled (x dx : PendulumState) (h : Float) : PendulumState :=
  {
    theta := x.theta + h * dx.theta
    thetadot := x.thetadot + h * dx.thetadot
  }

private def rk4ClosedLoopStep
    (p : PendulumParams)
    (controller : PendulumState → PendulumInput)
    (dt : Float)
    (x : PendulumState) : PendulumState :=
  let f := fun y => derivative p (controller y) y
  let k1 := f x
  let k2 := f (stateAddScaled x k1 (0.5 * dt))
  let k3 := f (stateAddScaled x k2 (0.5 * dt))
  let k4 := f (stateAddScaled x k3 dt)
  {
    theta := x.theta + dt * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta) / 6.0
    thetadot := x.thetadot +
      dt * (k1.thetadot + 2.0 * k2.thetadot + 2.0 * k3.thetadot + k4.thetadot) / 6.0
  }

private def rolloutClosedLoop
    (p : PendulumParams)
    (controller : PendulumState → PendulumInput)
    (dt : Float)
    (steps : Nat)
    (x0 : PendulumState) : Array PendulumState := Id.run do
  let mut x := x0
  let mut samples := #[x0]
  for _ in [:steps] do
    x := rk4ClosedLoopStep p controller dt x
    samples := samples.push x
  return samples

structure LqrConfig where
  target : PendulumState := uprightState
  qTheta : Float := 10.0
  qThetadot : Float := 1.0
  rTau : Float := 1.0
  duration : Float := 10.0
  stepSize : Float := 1.0e-3
  steps : Nat := 10000
  deriving Repr, Inhabited

structure PendulumLinearization where
  A : Array (Array Float)
  B : Array (Array Float)
  deriving Repr, Inhabited

structure LqrGain where
  kTheta : Float
  kThetadot : Float
  p11 : Float
  p12 : Float
  p22 : Float
  deriving Repr, Inhabited

namespace LqrConfig

def validate? (cfg : LqrConfig) : Except String Unit := do
  if !(Float.isFinite cfg.qTheta) || cfg.qTheta < 0.0 then
    .error s!"pendulum LQR qTheta must be nonnegative and finite, got {cfg.qTheta}"
  if !(Float.isFinite cfg.qThetadot) || cfg.qThetadot < 0.0 then
    .error s!"pendulum LQR qThetadot must be nonnegative and finite, got {cfg.qThetadot}"
  if !(Float.isFinite cfg.rTau) || cfg.rTau <= 0.0 then
    .error s!"pendulum LQR rTau must be positive and finite, got {cfg.rTau}"
  if !(Float.isFinite cfg.duration) || cfg.duration <= 0.0 then
    .error s!"pendulum LQR duration must be positive and finite, got {cfg.duration}"
  if !(Float.isFinite cfg.stepSize) || cfg.stepSize <= 0.0 then
    .error s!"pendulum LQR step size must be positive and finite, got {cfg.stepSize}"
  if cfg.steps == 0 then
    .error "pendulum LQR rollout requires at least one step"
  if Float.abs (cfg.steps.toFloat * cfg.stepSize - cfg.duration) > 1.0e-12 then
    .error s!"pendulum LQR step count {cfg.steps} does not match duration {cfg.duration}"

end LqrConfig

def lqrConfig : LqrConfig := {}

def linearizationAboutUpright (p : PendulumParams := params) : PendulumLinearization :=
  let inertia := p.mass * p.length * p.length
  {
    A := #[#[0.0, 1.0], #[p.gravity / p.length, -p.damping / inertia]]
    B := #[#[0.0], #[1.0 / inertia]]
  }

def lqrGain? (p : PendulumParams := params) (cfg : LqrConfig := lqrConfig) :
    Except String LqrGain := do
  if !p.isValid then
    .error "pendulum params are invalid"
  cfg.validate?
  let inertia := p.mass * p.length * p.length
  if inertia <= 0.0 then
    .error "pendulum LQR requires positive inertia"
  let a := p.gravity / p.length
  let c := -p.damping / inertia
  let d := 1.0 / inertia
  let d2OverR := d * d / cfg.rTau
  if d2OverR <= 0.0 then
    .error "pendulum LQR has invalid input weighting"
  let p12 :=
    (2.0 * a + Float.sqrt ((2.0 * a) * (2.0 * a) + 4.0 * d2OverR * cfg.qTheta)) /
      (2.0 * d2OverR)
  let p22 :=
    (2.0 * c + Float.sqrt ((2.0 * c) * (2.0 * c) +
      4.0 * d2OverR * (2.0 * p12 + cfg.qThetadot))) / (2.0 * d2OverR)
  let p11 := d2OverR * p12 * p22 - c * p12 - a * p22
  pure {
    kTheta := d * p12 / cfg.rTau
    kThetadot := d * p22 / cfg.rTau
    p11 := p11
    p12 := p12
    p22 := p22
  }

def lqrController (gain : LqrGain) (cfg : LqrConfig := lqrConfig)
    (x : PendulumState) : PendulumInput :=
  {
    tau := -(gain.kTheta * (x.theta - cfg.target.theta) +
      gain.kThetadot * (x.thetadot - cfg.target.thetadot))
  }

def acceptedSegment (t0 t1 : Float) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := t0
    tAttempt := t1
    tAfter := t1
    label := "pendulum-continuous-interval"
  }

def energyShapingControllerDesiredEnergy (p : PendulumParams := params) : Float :=
  1.1 * p.mass * p.gravity * p.length

def energyShapingGain : Float := 0.1

def energyShapingController (p : PendulumParams := params)
    (x : PendulumState) : PendulumInput :=
  {
    tau := p.damping * x.thetadot +
      energyShapingGain * x.thetadot *
        (energyShapingControllerDesiredEnergy p - totalEnergy p x)
  }

structure ControllerSimulationResult where
  references : Array DrakeReference
  controllerName : String
  t0 : Float
  t1 : Float
  stepSize : Float
  initialState : PendulumState
  finalState : PendulumState
  samples : Array PendulumState
  initialEnergy : Float
  finalEnergy : Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

private def controllerMove (vertex : VertexId) (label : String) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := .exact
    label := label
  }

def simulateLqr? (p : PendulumParams := params)
    (cfg : LqrConfig := lqrConfig)
    (x0 : PendulumState := { theta := pi + 0.1, thetadot := 0.2 }) :
    Except String ControllerSimulationResult := do
  let gain ← lqrGain? p cfg
  if !x0.isValid then
    .error "pendulum LQR initial state is invalid"
  let samples := rolloutClosedLoop p (lqrController gain cfg) cfg.stepSize cfg.steps x0
  let final := samples[samples.size - 1]!
  let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment 0.0 cfg.duration))
  trace.validate?
  pure {
    references := drakeReferences
    controllerName := "pendulum-upright-lqr"
    t0 := 0.0
    t1 := cfg.duration
    stepSize := cfg.stepSize
    initialState := x0
    finalState := final
    samples := samples
    initialEnergy := totalEnergy p x0
    finalEnergy := totalEnergy p final
    trace := trace
    moves := #[controllerMove 5000 "pendulum LQR local linearization/controller"] ++ trace.moves
  }

structure EnergyShapingConfig where
  duration : Float := 10.0
  stepSize : Float := 1.0e-3
  steps : Nat := 10000
  deriving Repr, Inhabited

namespace EnergyShapingConfig

def validate? (cfg : EnergyShapingConfig) : Except String Unit := do
  if !(Float.isFinite cfg.duration) || cfg.duration <= 0.0 then
    .error s!"pendulum energy-shaping duration must be positive and finite, got {cfg.duration}"
  if !(Float.isFinite cfg.stepSize) || cfg.stepSize <= 0.0 then
    .error s!"pendulum energy-shaping step size must be positive and finite, got {cfg.stepSize}"
  if cfg.steps == 0 then
    .error "pendulum energy-shaping rollout requires at least one step"
  if Float.abs (cfg.steps.toFloat * cfg.stepSize - cfg.duration) > 1.0e-12 then
    .error s!"pendulum energy-shaping step count {cfg.steps} does not match duration {cfg.duration}"

end EnergyShapingConfig

def energyShapingConfig : EnergyShapingConfig := {}

def simulateEnergyShaping? (p : PendulumParams := params)
    (cfg : EnergyShapingConfig := energyShapingConfig)
    (x0 : PendulumState := { theta := 0.1, thetadot := 0.2 }) :
    Except String ControllerSimulationResult := do
  if !p.isValid then
    .error "pendulum params are invalid"
  cfg.validate?
  if !x0.isValid then
    .error "pendulum energy-shaping initial state is invalid"
  let samples := rolloutClosedLoop p (energyShapingController p) cfg.stepSize cfg.steps x0
  let final := samples[samples.size - 1]!
  let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment 0.0 cfg.duration))
  trace.validate?
  pure {
    references := drakeReferences
    controllerName := "pendulum-energy-shaping"
    t0 := 0.0
    t1 := cfg.duration
    stepSize := cfg.stepSize
    initialState := x0
    finalState := final
    samples := samples
    initialEnergy := totalEnergy p x0
    finalEnergy := totalEnergy p final
    trace := trace
    moves := #[controllerMove 5001 "pendulum energy-shaping local controller"] ++ trace.moves
  }

structure SymbolicDynamicsRecord where
  pendulumPlantThetaDot : String
  pendulumPlantThetaDotDot : String
  multibodyPlantThetaDot : String
  multibodyPlantThetaDotDot : String
  deriving Repr, Inhabited

def symbolicDynamics : SymbolicDynamicsRecord :=
  {
    pendulumPlantThetaDot := "thetadot"
    pendulumPlantThetaDotDot :=
      "(tau - damping * thetadot - mass * gravity * length * sin(theta)) / (mass * length^2)"
    multibodyPlantThetaDot := "thetadot"
    multibodyPlantThetaDotDot :=
      "(tau - damping * thetadot - mass * gravity * length * sin(theta)) / (mass * length^2)"
  }

def symbolicDynamicsAgree (record : SymbolicDynamicsRecord := symbolicDynamics) : Bool :=
  record.pendulumPlantThetaDot == record.multibodyPlantThetaDot &&
    record.pendulumPlantThetaDotDot == record.multibodyPlantThetaDotDot

structure DirectCollocationSpec where
  numTimeSamples : Nat := 21
  minimumTimeStep : Float := 0.2
  maximumTimeStep : Float := 0.5
  torqueLimit : Float := 3.0
  runningCostR : Float := 10.0
  initialState : PendulumState := { theta := 0.0, thetadot := 0.0 }
  finalState : PendulumState := uprightState
  initialTrajectoryTimespan : Float := 4.0
  pidKp : Float := 10.0
  pidKi : Float := 0.0
  pidKd : Float := 1.0
  solverBlockExactness : MoveExactness := .controlledApproximation
  deriving Repr, Inhabited

namespace DirectCollocationSpec

def validate? (spec : DirectCollocationSpec) : Except String Unit := do
  if spec.numTimeSamples < 2 then
    .error "pendulum direct collocation requires at least two knot points"
  if !(Float.isFinite spec.minimumTimeStep) || spec.minimumTimeStep <= 0.0 then
    .error s!"minimum time step must be positive and finite, got {spec.minimumTimeStep}"
  if !(Float.isFinite spec.maximumTimeStep) || spec.maximumTimeStep < spec.minimumTimeStep then
    .error s!"maximum time step {spec.maximumTimeStep} must be finite and >= minimum time step {spec.minimumTimeStep}"
  if !(Float.isFinite spec.torqueLimit) || spec.torqueLimit <= 0.0 then
    .error s!"torque limit must be positive and finite, got {spec.torqueLimit}"
  if !(Float.isFinite spec.runningCostR) || spec.runningCostR <= 0.0 then
    .error s!"running cost R must be positive and finite, got {spec.runningCostR}"
  if !spec.initialState.isValid || !spec.finalState.isValid then
    .error "direct collocation boundary states must be finite"

def minimumDuration (spec : DirectCollocationSpec) : Float :=
  (spec.numTimeSamples - 1).toFloat * spec.minimumTimeStep

def maximumDuration (spec : DirectCollocationSpec) : Float :=
  (spec.numTimeSamples - 1).toFloat * spec.maximumTimeStep

end DirectCollocationSpec

def directCollocationSpec : DirectCollocationSpec := {}

def directCollocationGraph (spec : DirectCollocationSpec := directCollocationSpec) :
    SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 5100, kind := .state .boundary, label := "pendulum dircol initial state" }
    |>.addVertex { id := 5101, kind := .interval, label := "direct-collocation knot dynamics" }
    |>.addVertex { id := 5102, kind := .state .boundary, label := "pendulum dircol final state" }
    |>.addVertex { id := 5103, kind := .learnedComplement, label := "nonlinear program solve result" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5101, 5103]
      exactness := spec.solverBlockExactness
      label := "Solve direct collocation NLP and reconstruct trajectories"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5103]
      exactness := .exact
      label := "Track reconstructed trajectory with scalar PID controller"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[5102]
      exactness := .exact
      label := "Check final pendulum state after trajectory tracking"
    }

def odeTerm (p : PendulumParams) : ODETerm PendulumState PendulumInput :=
  { vectorField := fun _t x u => derivative p u x }

def pendulumSolver :=
  RK4.solver
    (Term := ODETerm PendulumState PendulumInput)
    (Y := PendulumState)
    (VF := PendulumState)
    (Args := PendulumInput)

structure SimulationResult where
  references : Array DrakeReference
  t0 : Float
  t1 : Float
  input : PendulumInput
  initialState : PendulumState
  finalState : PendulumState
  initialEnergy : Float
  finalEnergy : Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def solvePassive? (p : PendulumParams := params)
    (x0 : PendulumState := { theta := 0.1, thetadot := 0.0 })
    (t0 : Float := 0.0)
    (t1 : Float := 0.25)
    (u : PendulumInput := defaultInput) :
    Except String SimulationResult := do
  if !p.isValid then
    .error "pendulum params are invalid"
  if !u.isValid then
    .error "pendulum input is invalid"
  if !x0.isValid then
    .error "pendulum initial state is invalid"
  let sol :=
    diffeqsolve
      (Term := ODETerm PendulumState PendulumInput)
      (Y := PendulumState)
      (VF := PendulumState)
      (Control := Time)
      (Args := PendulumInput)
      (Controller := ConstantStepSize)
      (odeTerm p) pendulumSolver t0 t1 (some p.stepSize) x0 u
      (saveat := { t1 := true })
  if !sol.result.isOkay then
    .error s!"pendulum solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "pendulum solve did not save endpoint"
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
    | _, _ => .error "pendulum solve did not save endpoint arrays"

def buildEndToEnd? : Except String SimulationResult :=
  solvePassive?

end Tyr.EventSkeleton.Examples.Pendulum
