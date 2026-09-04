import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Trace

/-!
# Drake Acrobot Event-Skeleton Example

This ports the core plant dynamics from `../drake/examples/acrobot`.

The Acrobot plant is a two-degree-of-freedom manipulator:

`M(q) qddot + bias(q, qdot) = B tau`, with `B = [0, 1]^T`.

The example keeps Drake's named-vector coordinate order, default parameters,
MIT hardware parameter setter, explicit mass/bias methods, implicit residual,
and a solver-backed continuous interval trace.
-/

namespace Tyr.EventSkeleton.Examples.Acrobot

open Tyr.EventSkeleton
open torch.DiffEq

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/acrobot/BUILD.bazel"
      concept := "declares the root Acrobot example libraries, binaries, tests, and LCM dependencies"
    },
    {
      path := "../drake/examples/acrobot/README.md"
      concept := "documents the Acrobot example, simulations, controllers, and optimization demos"
    },
    {
      path := "../drake/examples/acrobot/Acrobot.urdf"
      concept := "URDF model with base, upper/lower links, shoulder/elbow joints, elbow transmission, and hand frame"
    },
    {
      path := "../drake/examples/acrobot/Acrobot.sdf"
      concept := "SDF model with equivalent Acrobot links, joints, inertias, visuals, and hand frame"
    },
    {
      path := "../drake/examples/acrobot/Acrobot_no_collision.urdf"
      concept := "URDF variant without collision geometry for parser and visualization scenarios"
    },
    {
      path := "../drake/examples/acrobot/acrobot_plant.cc"
      concept := "implements mass matrix, dynamics bias, implicit residual, and energies"
    },
    {
      path := "../drake/examples/acrobot/acrobot_plant.h"
      concept := "declares AcrobotPlant ports, input/state/parameter vector contracts, energy outputs, and scalar conversion"
    },
    {
      path := "../drake/examples/acrobot/acrobot_geometry.h"
      concept := "declares the AcrobotGeometry SceneGraph helper with state input and geometry_pose output"
    },
    {
      path := "../drake/examples/acrobot/acrobot_geometry.cc"
      concept := "registers base, upper-link, and lower-link illustration geometry and emits FramePoseVector poses"
    },
    {
      path := "../drake/examples/acrobot/test/acrobot_geometry_test.cc"
      concept := "acceptance test for adding AcrobotGeometry to a DiagramBuilder with AcrobotPlant and SceneGraph"
    },
    {
      path := "../drake/examples/acrobot/acrobot_state.h"
      concept := "defines AcrobotState coordinate order theta1, theta2, theta1dot, theta2dot"
    },
    {
      path := "../drake/examples/acrobot/acrobot_state.cc"
      concept := "defines AcrobotStateIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/acrobot/acrobot_input.h"
      concept := "defines elbow_torque input coordinate tau"
    },
    {
      path := "../drake/examples/acrobot/acrobot_input.cc"
      concept := "defines AcrobotInputIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/acrobot/acrobot_params.h"
      concept := "defines default AcrobotParams and named parameter order"
    },
    {
      path := "../drake/examples/acrobot/acrobot_params.cc"
      concept := "defines AcrobotParamsIndices::GetCoordinateNames"
    },
    {
      path := "../drake/examples/acrobot/acrobot_io.py"
      concept := "loads and saves deterministic or stochastic Spong simulation scenarios and output tapes"
    },
    {
      path := "../drake/examples/acrobot/acrobot_lcm.cc"
      concept := "implements Acrobot LCM state/command sender and receiver systems"
    },
    {
      path := "../drake/examples/acrobot/acrobot_lcm.h"
      concept := "declares AcrobotStateReceiver, AcrobotCommandSender, AcrobotCommandReceiver, and AcrobotStateSender systems"
    },
    {
      path := "../drake/examples/acrobot/metrics.py"
      concept := "defines ensemble_cost and success_rate metrics for optimizer_demo rollouts"
    },
    {
      path := "../drake/examples/acrobot/test/acrobot_plant_test.cc"
      concept := "checks implicit residual, disconnected zero input, and MIT parameter setter"
    },
    {
      path := "../drake/examples/acrobot/test/acrobot_io_test.py"
      concept := "checks scenario YAML and output tape load/save boundaries"
    },
    {
      path := "../drake/examples/acrobot/test/acrobot_lcm_msg_generator.cc"
      concept := "generates Acrobot LCM message examples for lcmt_acrobot_x and lcmt_acrobot_u"
    },
    {
      path := "../drake/examples/acrobot/test/metrics_test.py"
      concept := "checks ensemble_cost and success_rate metrics for optimizer_demo"
    },
    {
      path := "../drake/examples/acrobot/test/multibody_dynamics_test.cc"
      concept := "compares hand-written AcrobotPlant dynamics with parsed MultibodyPlant dynamics"
    },
    {
      path := "../drake/examples/acrobot/run_passive.cc"
      concept := "runs the passive AcrobotPlant from theta1=1, theta2=1 for the command-line horizon"
    },
    {
      path := "../drake/examples/acrobot/run_lqr.cc"
      concept := "builds BalancingLQRController and simulates near the upright fixed point"
    },
    {
      path := "../drake/examples/acrobot/run_lqr_w_estimator.cc"
      concept := "closes the LQR loop through rotary encoders and a steady-state Kalman observer"
    },
    {
      path := "../drake/examples/acrobot/run_plant_w_lcm.cc"
      concept := "runs the Acrobot plant behind LCM command/state sender-receiver systems"
    },
    {
      path := "../drake/examples/acrobot/run_swing_up.cc"
      concept := "runs AcrobotSpongController from the near-downward swing-up initial state"
    },
    {
      path := "../drake/examples/acrobot/spong_controller.cc"
      concept := "implements the Spong controller output calculation and parameter handling"
    },
    {
      path := "../drake/examples/acrobot/spong_controller.h"
      concept := "defines the Spong swing-up law, LQR balancing switch, and torque saturation"
    },
    {
      path := "../drake/examples/acrobot/spong_controller_w_lcm.cc"
      concept := "runs the Spong controller behind LCM state receiver and command sender systems"
    },
    {
      path := "../drake/examples/acrobot/spong_controller_params.h"
      concept := "defines Spong controller parameter names and defaults"
    },
    {
      path := "../drake/examples/acrobot/spong_controller_params.cc"
      concept := "materializes Spong controller parameter indices and coordinate names"
    },
    {
      path := "../drake/examples/acrobot/spong_sim.cc"
      concept := "loads a scenario, samples stochastic fields, runs a Spong-controlled plant, and writes a state tape"
    },
    {
      path := "../drake/examples/acrobot/spong_sim.py"
      concept := "Python equivalent of the Spong scenario simulation boundary"
    },
    {
      path := "../drake/examples/acrobot/optimizer_demo.py"
      concept := "optimizes Spong controller parameters by repeated stochastic scenario rollouts"
    },
    {
      path := "../drake/examples/acrobot/test/run_swing_up_traj_optimization.cc"
      concept := "solves direct collocation swing-up with SNOPT, stabilizes the trajectory with finite-horizon LQR, and checks the terminal state"
    },
    {
      path := "../drake/examples/acrobot/test/spong_sim_lib_py_test.py"
      concept := "checks the Python Spong simulate() library boundary and output tape shape"
    },
    {
      path := "../drake/examples/acrobot/test/spong_sim_main_test.py"
      concept := "checks Python and C++ spong_sim_main subprocess help, deterministic scenario output, and stochastic dump shape"
    },
    {
      path := "../drake/examples/acrobot/test/example_scenario.yaml"
      concept := "deterministic Spong simulation scenario used by the example runner"
    },
    {
      path := "../drake/examples/acrobot/test/example_stochastic_scenario.yaml"
      concept := "UniformVector stochastic scenario used by optimizer_demo"
    },
    {
      path := "../drake/examples/multibody/acrobot/passive_simulation.cc"
      concept := "builds the benchmark Acrobot MultibodyPlant, connects zero elbow torque, chooses an integrator, and advances the simulator"
    },
    {
      path := "../drake/examples/multibody/acrobot/run_lqr.cc"
      concept := "parses the benchmark acrobot SDF, builds a time-stepping MultibodyPlant, connects a balancing LQR controller, and runs five randomized rollouts"
    }
  ]

def stateCoordinateNames : Array String :=
  #["theta1", "theta2", "theta1dot", "theta2dot"]

def inputCoordinateNames : Array String :=
  #["tau"]

def parameterCoordinateNames : Array String :=
  #["m1", "m2", "l1", "l2", "lc1", "lc2", "Ic1", "Ic2", "b1", "b2", "gravity"]

def spongParameterCoordinateNames : Array String :=
  #["k_e", "k_p", "k_d", "balancing_threshold"]

structure IndexedCoordinateBoundary where
  names : Array String
  sourcePath : String
  deriving Repr, Inhabited

namespace IndexedCoordinateBoundary

def validate? (boundary : IndexedCoordinateBoundary)
    (expected : Array String) (label : String) : Except String Unit := do
  if boundary.sourcePath == "" then
    .error s!"{label} source path must be nonempty"
  if boundary.names != expected then
    .error s!"{label} coordinate names mismatch: expected {expected}, got {boundary.names}"

end IndexedCoordinateBoundary

private def hasDuplicateString (xs : Array String) : Bool := Id.run do
  let mut duplicate := false
  for i in [:xs.size] do
    for j in [:(xs.size - i - 1)] do
      let k := i + j + 1
      if xs[i]! == xs[k]! then
        duplicate := true
  return duplicate

structure AcrobotNamedVectorBoundary where
  typeName : String
  headerPath : String
  implementationPath : String
  coordinateNames : Array String
  defaults : Array Float
  lowerBounds : Array (Option Float)
  upperBounds : Array (Option Float)
  movedFromAccessThrows : Bool := true
  supportsNamedVariables : Bool := true
  deriving Repr, Inhabited

namespace AcrobotNamedVectorBoundary

def dimension (boundary : AcrobotNamedVectorBoundary) : Nat :=
  boundary.coordinateNames.size

def indexOf? (boundary : AcrobotNamedVectorBoundary) (name : String) : Option Nat :=
  boundary.coordinateNames.findIdx? (fun candidate => candidate == name)

def validate? (boundary : AcrobotNamedVectorBoundary) : Except String Unit := do
  if boundary.typeName.isEmpty then
    .error "Acrobot named vector boundary requires a type name"
  if boundary.headerPath.isEmpty then
    .error s!"{boundary.typeName} must record its Drake header path"
  if boundary.implementationPath.isEmpty then
    .error s!"{boundary.typeName} must record its Drake implementation path"
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

end AcrobotNamedVectorBoundary

def spongControllerParamsIndicesBoundary : IndexedCoordinateBoundary :=
  {
    names := spongParameterCoordinateNames
    sourcePath := "../drake/examples/acrobot/spong_controller_params.cc"
  }

private def finiteNonnegative (x : Float) : Bool :=
  Float.isFinite x && x >= 0.0

structure AcrobotParams where
  m1 : Float := 1.0
  m2 : Float := 1.0
  l1 : Float := 1.0
  l2 : Float := 2.0
  lc1 : Float := 0.5
  lc2 : Float := 1.0
  Ic1 : Float := 0.083
  Ic2 : Float := 0.33
  b1 : Float := 0.1
  b2 : Float := 0.1
  gravity : Float := 9.81
  stepSize : Float := 1.0e-3
  deriving Repr, Inhabited

namespace AcrobotParams

def isValid (p : AcrobotParams) : Bool :=
  finiteNonnegative p.m1 &&
  finiteNonnegative p.m2 &&
  finiteNonnegative p.l1 &&
  finiteNonnegative p.l2 &&
  finiteNonnegative p.lc1 &&
  finiteNonnegative p.lc2 &&
  finiteNonnegative p.Ic1 &&
  finiteNonnegative p.Ic2 &&
  finiteNonnegative p.b1 &&
  finiteNonnegative p.b2 &&
  finiteNonnegative p.gravity

def isFiniteForDynamics (p : AcrobotParams) : Bool :=
  Float.isFinite p.m1 &&
  Float.isFinite p.m2 &&
  Float.isFinite p.l1 &&
  Float.isFinite p.l2 &&
  Float.isFinite p.lc1 &&
  Float.isFinite p.lc2 &&
  Float.isFinite p.Ic1 &&
  Float.isFinite p.Ic2 &&
  Float.isFinite p.b1 &&
  Float.isFinite p.b2 &&
  Float.isFinite p.gravity

def asArray (p : AcrobotParams) : Array Float :=
  #[p.m1, p.m2, p.l1, p.l2, p.lc1, p.lc2, p.Ic1, p.Ic2, p.b1, p.b2, p.gravity]

def lowerBounds : Array (Option Float) :=
  #[some 0.0, some 0.0, some 0.0, some 0.0, some 0.0, some 0.0,
    some 0.0, some 0.0, some 0.0, some 0.0, some 0.0]

def upperBounds : Array (Option Float) :=
  #[none, none, none, none, none, none, none, none, none, none, none]

def fromArray? (xs : Array Float) : Except String AcrobotParams := do
  if xs.size != 11 then
    .error s!"AcrobotParams expects 11 coordinates, got {xs.size}"
  let p : AcrobotParams :=
    {
      m1 := xs[0]!
      m2 := xs[1]!
      l1 := xs[2]!
      l2 := xs[3]!
      lc1 := xs[4]!
      lc2 := xs[5]!
      Ic1 := xs[6]!
      Ic2 := xs[7]!
      b1 := xs[8]!
      b2 := xs[9]!
      gravity := xs[10]!
    }
  if !p.isValid then
    .error s!"AcrobotParams values are outside Drake's BasicVector domain: {reprStr xs}"
  pure p

end AcrobotParams

def params : AcrobotParams := {}

def acrobotParamsVectorBoundary : AcrobotNamedVectorBoundary :=
  {
    typeName := "AcrobotParams"
    headerPath := "../drake/examples/acrobot/acrobot_params.h"
    implementationPath := "../drake/examples/acrobot/acrobot_params.cc"
    coordinateNames := parameterCoordinateNames
    defaults := params.asArray
    lowerBounds := AcrobotParams.lowerBounds
    upperBounds := AcrobotParams.upperBounds
  }

/--
Parameter values from Drake's `SetMitAcrobotParameters`.

Drake's comment notes that the identified inertias are negative and the torque
units are effectively motor current.  We preserve those values exactly, and use
`isFiniteForDynamics` rather than the nonnegative BasicVector domain when
running this identified model.
-/
def mitParams : AcrobotParams :=
  {
    m1 := 2.4367
    m2 := 0.6178
    l1 := 0.2563
    l2 := params.l2
    lc1 := 1.6738
    lc2 := 1.5651
    Ic1 := -4.7443
    Ic2 := -1.0068
    b1 := 0.0320
    b2 := 0.0413
    gravity := params.gravity
    stepSize := params.stepSize
  }

structure AcrobotInput where
  tau : Float := 0.0
  deriving Repr, Inhabited

namespace AcrobotInput

def isValid (u : AcrobotInput) : Bool :=
  Float.isFinite u.tau

def asArray (u : AcrobotInput) : Array Float :=
  #[u.tau]

def lowerBounds : Array (Option Float) :=
  #[none]

def upperBounds : Array (Option Float) :=
  #[none]

def fromArray? (xs : Array Float) : Except String AcrobotInput := do
  if xs.size != 1 then
    .error s!"AcrobotInput expects 1 coordinate, got {xs.size}"
  let u : AcrobotInput := { tau := xs[0]! }
  if !u.isValid then
    .error s!"AcrobotInput values are not finite: {reprStr xs}"
  pure u

end AcrobotInput

structure AcrobotState where
  theta1 : Float := 0.0
  theta2 : Float := 0.0
  theta1dot : Float := 0.0
  theta2dot : Float := 0.0
  deriving Repr, Inhabited

namespace AcrobotState

def isValid (x : AcrobotState) : Bool :=
  Float.isFinite x.theta1 &&
  Float.isFinite x.theta2 &&
  Float.isFinite x.theta1dot &&
  Float.isFinite x.theta2dot

def asArray (x : AcrobotState) : Array Float :=
  #[x.theta1, x.theta2, x.theta1dot, x.theta2dot]

def lowerBounds : Array (Option Float) :=
  #[none, none, none, none]

def upperBounds : Array (Option Float) :=
  #[none, none, none, none]

def fromArray? (xs : Array Float) : Except String AcrobotState := do
  if xs.size != 4 then
    .error s!"AcrobotState expects 4 coordinates, got {xs.size}"
  let x : AcrobotState :=
    {
      theta1 := xs[0]!
      theta2 := xs[1]!
      theta1dot := xs[2]!
      theta2dot := xs[3]!
    }
  if !x.isValid then
    .error s!"AcrobotState values are not finite: {reprStr xs}"
  pure x

end AcrobotState

instance : DiffEqSpace AcrobotState where
  add a b := {
    theta1 := a.theta1 + b.theta1
    theta2 := a.theta2 + b.theta2
    theta1dot := a.theta1dot + b.theta1dot
    theta2dot := a.theta2dot + b.theta2dot
  }
  sub a b := {
    theta1 := a.theta1 - b.theta1
    theta2 := a.theta2 - b.theta2
    theta1dot := a.theta1dot - b.theta1dot
    theta2dot := a.theta2dot - b.theta2dot
  }
  scale s x := {
    theta1 := s * x.theta1
    theta2 := s * x.theta2
    theta1dot := s * x.theta1dot
    theta2dot := s * x.theta2dot
  }

instance : DiffEqSeminorm AcrobotState where
  rms x :=
    max (max (Float.abs x.theta1) (Float.abs x.theta2))
      (max (Float.abs x.theta1dot) (Float.abs x.theta2dot))

instance : DiffEqElem AcrobotState where
  abs x := {
    theta1 := Float.abs x.theta1
    theta2 := Float.abs x.theta2
    theta1dot := Float.abs x.theta1dot
    theta2dot := Float.abs x.theta2dot
  }
  max a b := {
    theta1 := max a.theta1 b.theta1
    theta2 := max a.theta2 b.theta2
    theta1dot := max a.theta1dot b.theta1dot
    theta2dot := max a.theta2dot b.theta2dot
  }
  addScalar s x := {
    theta1 := x.theta1 + s
    theta2 := x.theta2 + s
    theta1dot := x.theta1dot + s
    theta2dot := x.theta2dot + s
  }
  div a b := {
    theta1 := a.theta1 / b.theta1
    theta2 := a.theta2 / b.theta2
    theta1dot := a.theta1dot / b.theta1dot
    theta2dot := a.theta2dot / b.theta2dot
  }

def defaultState : AcrobotState := {}

def defaultInput : AcrobotInput := {}

structure AcrobotPhysicsState where
  state : AcrobotState := defaultState
  input : AcrobotInput := defaultInput
  deriving Repr, Inhabited

def physicsState
    (state : AcrobotState := defaultState)
    (input : AcrobotInput := defaultInput) : AcrobotPhysicsState :=
  { state := state, input := input }

def acrobotStateVectorBoundary : AcrobotNamedVectorBoundary :=
  {
    typeName := "AcrobotState"
    headerPath := "../drake/examples/acrobot/acrobot_state.h"
    implementationPath := "../drake/examples/acrobot/acrobot_state.cc"
    coordinateNames := stateCoordinateNames
    defaults := defaultState.asArray
    lowerBounds := AcrobotState.lowerBounds
    upperBounds := AcrobotState.upperBounds
  }

def acrobotInputVectorBoundary : AcrobotNamedVectorBoundary :=
  {
    typeName := "AcrobotInput"
    headerPath := "../drake/examples/acrobot/acrobot_input.h"
    implementationPath := "../drake/examples/acrobot/acrobot_input.cc"
    coordinateNames := inputCoordinateNames
    defaults := defaultInput.asArray
    lowerBounds := AcrobotInput.lowerBounds
    upperBounds := AcrobotInput.upperBounds
  }

def stateAsArray (x : AcrobotState) : Array Float :=
  x.asArray

def inputAsArray (u : AcrobotInput) : Array Float :=
  u.asArray

def qdotAsArray (x : AcrobotState) : Array Float :=
  #[x.theta1dot, x.theta2dot]

structure AcrobotModelAssetBoundary where
  modelName : String := "Acrobot"
  urdfPath : String := "../drake/examples/acrobot/Acrobot.urdf"
  sdfPath : String := "../drake/examples/acrobot/Acrobot.sdf"
  noCollisionUrdfPath : String := "../drake/examples/acrobot/Acrobot_no_collision.urdf"
  linkNames : Array String := #["base_link", "upper_link", "lower_link"]
  jointNames : Array String := #["base_weld", "shoulder", "elbow"]
  actuatedJointNames : Array String := #["elbow"]
  transmissionNames : Array String := #["elbow_trans"]
  frameNames : Array String := #["hand"]
  jointAxes : Array (Array Float) := #[#[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0]]
  baseBoxSize : Array Float := #[0.2, 0.2, 0.2]
  visualCylinderRadius : Float := 0.05
  urdfUpperVisualLength : Float := 1.1
  urdfLowerVisualLength : Float := 2.1
  handFrameOffset : Array Float := #[0.0, 0.0, -2.1]
  deriving Repr, Inhabited

namespace AcrobotModelAssetBoundary

private def finiteArray (xs : Array Float) : Bool :=
  xs.all Float.isFinite

def validate? (boundary : AcrobotModelAssetBoundary) : Except String Unit := do
  if boundary.modelName != "Acrobot" then
    .error s!"Acrobot model name mismatch: {boundary.modelName}"
  if boundary.urdfPath != "../drake/examples/acrobot/Acrobot.urdf" then
    .error s!"Acrobot URDF path mismatch: {boundary.urdfPath}"
  if boundary.sdfPath != "../drake/examples/acrobot/Acrobot.sdf" then
    .error s!"Acrobot SDF path mismatch: {boundary.sdfPath}"
  if boundary.noCollisionUrdfPath != "../drake/examples/acrobot/Acrobot_no_collision.urdf" then
    .error s!"Acrobot no-collision URDF path mismatch: {boundary.noCollisionUrdfPath}"
  if boundary.linkNames != #["base_link", "upper_link", "lower_link"] then
    .error s!"Acrobot link names mismatch: {boundary.linkNames}"
  if boundary.jointNames != #["base_weld", "shoulder", "elbow"] then
    .error s!"Acrobot joint names mismatch: {boundary.jointNames}"
  if boundary.actuatedJointNames != #["elbow"] then
    .error s!"Acrobot actuated joint names mismatch: {boundary.actuatedJointNames}"
  if boundary.transmissionNames != #["elbow_trans"] then
    .error s!"Acrobot transmission names mismatch: {boundary.transmissionNames}"
  if boundary.frameNames != #["hand"] then
    .error s!"Acrobot frame names mismatch: {boundary.frameNames}"
  if boundary.jointAxes.size != 2 ||
      !boundary.jointAxes.all (fun axis => axis == #[0.0, 1.0, 0.0]) then
    .error s!"Acrobot shoulder/elbow axes should both be +Y, got {boundary.jointAxes}"
  if boundary.baseBoxSize != #[0.2, 0.2, 0.2] then
    .error s!"Acrobot base box size mismatch: {boundary.baseBoxSize}"
  if !boundary.visualCylinderRadius.isFinite || boundary.visualCylinderRadius <= 0.0 then
    .error s!"Acrobot visual cylinder radius must be positive and finite, got {boundary.visualCylinderRadius}"
  if !boundary.urdfUpperVisualLength.isFinite || !boundary.urdfLowerVisualLength.isFinite ||
      boundary.urdfUpperVisualLength <= 0.0 || boundary.urdfLowerVisualLength <= 0.0 then
    .error "Acrobot URDF visual cylinder lengths must be positive and finite"
  if boundary.handFrameOffset.size != 3 || !finiteArray boundary.handFrameOffset then
    .error s!"Acrobot hand frame offset must have three finite entries, got {boundary.handFrameOffset}"

end AcrobotModelAssetBoundary

def acrobotModelAssetBoundary : AcrobotModelAssetBoundary := {}

/-! ## AcrobotGeometry SceneGraph provider -/

def acrobotGeometrySourceId : Nat := 5260
def acrobotUpperLinkFrameId : Nat := 5261
def acrobotLowerLinkFrameId : Nat := 5262
def acrobotBaseGeometryId : Nat := 5263
def acrobotUpperLinkGeometryId : Nat := 5264
def acrobotLowerLinkGeometryId : Nat := 5265

def acrobotGeometryStateInputVertex : VertexId := 5266
def acrobotGeometryProviderVertex : VertexId := 5267
def acrobotGeometryPoseOutputVertex : VertexId := 5268

private def illustrationProperties (rgba : SceneRgba) : SceneGeometryProperties :=
  { roles := #[.illustration], diffuseRgba? := some rgba }

def acrobotGeometryProvider (p : AcrobotParams := params) :
    SceneGraphProvider :=
  {
    sources := #[
      { id := acrobotGeometrySourceId, name := "AcrobotGeometry" }
    ]
    frames := #[
      {
        id := acrobotUpperLinkFrameId
        sourceId := acrobotGeometrySourceId
        name := "upper_link"
      },
      {
        id := acrobotLowerLinkFrameId
        sourceId := acrobotGeometrySourceId
        name := "lower_link"
      }
    ]
    geometries := #[
      {
        id := acrobotBaseGeometryId
        sourceId := acrobotGeometrySourceId
        frameId? := none
        X_FG := ScenePose3.identity
        shape := .box 0.2 0.2 0.2
        name := "base"
        properties := illustrationProperties { r := 0.0, g := 1.0, b := 0.0, a := 1.0 }
      },
      {
        id := acrobotUpperLinkGeometryId
        sourceId := acrobotGeometrySourceId
        frameId? := some acrobotUpperLinkFrameId
        X_FG := {
          translation := { x := 0.0, y := 0.15, z := -p.l1 / 2.0 }
        }
        shape := .cylinder 0.05 p.l1
        name := "upper_link"
        properties := illustrationProperties { r := 1.0, g := 0.0, b := 0.0, a := 1.0 }
      },
      {
        id := acrobotLowerLinkGeometryId
        sourceId := acrobotGeometrySourceId
        frameId? := some acrobotLowerLinkFrameId
        X_FG := {
          translation := { x := 0.0, y := 0.25, z := -p.l2 / 2.0 }
        }
        shape := .cylinder 0.05 p.l2
        name := "lower_link"
        properties := illustrationProperties { r := 0.0, g := 0.0, b := 1.0, a := 1.0 }
      }
    ]
    label := "AcrobotGeometry SceneGraph provider"
  }

def acrobotGeometryPoseOutput
    (p : AcrobotParams := params) (x : AcrobotState := defaultState) :
    SceneFramePoseVector :=
  {
    poses := #[
      {
        frameId := acrobotUpperLinkFrameId
        X_WF := {
          rotationAxis := SceneVec3.unitY
          rotationAngle := x.theta1
        }
      },
      {
        frameId := acrobotLowerLinkFrameId
        X_WF := {
          translation := {
            x := -p.l1 * Float.sin x.theta1
            y := 0.0
            z := -p.l1 * Float.cos x.theta1
          }
          rotationAxis := SceneVec3.unitY
          rotationAngle := x.theta1 + x.theta2
        }
      }
    ]
  }

private def acrobotGeometryMove
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

def acrobotGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := acrobotGeometryStateInputVertex
      kind := .state .boundary
      label := "AcrobotGeometry state input"
    }
    |>.addVertex {
      id := acrobotGeometryProviderVertex
      kind := .state .boundary
      label := "AcrobotGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := acrobotGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "AcrobotGeometry geometry_pose output"
    }
    |>.addMove (acrobotGeometryMove acrobotGeometryProviderVertex
      "Register AcrobotGeometry source, frames, and illustration geometries"
      #[] #[acrobotGeometryProviderVertex])
    |>.addMove (acrobotGeometryMove acrobotGeometryPoseOutputVertex
      "OutputGeometryPose: q -> upper/lower FramePoseVector"
      #[acrobotGeometryStateInputVertex, acrobotGeometryProviderVertex]
      #[acrobotGeometryPoseOutputVertex])

structure AcrobotGeometryResult where
  references : Array DrakeReference
  params : AcrobotParams
  inputPortName : String := "state"
  inputPortSize : Nat := 4
  outputPortName : String := "geometry_pose"
  provider : SceneGraphProvider
  sampleState : AcrobotState
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildAcrobotGeometry?
    (p : AcrobotParams := params) (x : AcrobotState := defaultState) :
    Except String AcrobotGeometryResult := do
  if !p.isValid then
    .error "AcrobotGeometry requires valid nonnegative AcrobotParams"
  if !x.isValid then
    .error "AcrobotGeometry state input must be finite"
  let provider := acrobotGeometryProvider p
  provider.validate?
  let poses := acrobotGeometryPoseOutput p x
  poses.validate? provider
  pure {
    references := drakeReferences
    params := p
    provider := provider
    sampleState := x
    poses := poses
    graph := acrobotGeometryGraph
    moves := acrobotGeometryGraph.moves
  }

def massMatrix (p : AcrobotParams) (x : AcrobotState) : Array (Array Float) :=
  let c2 := Float.cos x.theta2
  let i1 := p.Ic1 + p.m1 * p.lc1 * p.lc1
  let i2 := p.Ic2 + p.m2 * p.lc2 * p.lc2
  let m2l1lc2 := p.m2 * p.l1 * p.lc2
  let m12 := i2 + m2l1lc2 * c2
  #[
    #[i1 + i2 + p.m2 * p.l1 * p.l1 + 2.0 * m2l1lc2 * c2, m12],
    #[m12, i2]
  ]

def dynamicsBiasTerm (p : AcrobotParams) (x : AcrobotState) : Array Float :=
  let s1 := Float.sin x.theta1
  let s2 := Float.sin x.theta2
  let s12 := Float.sin (x.theta1 + x.theta2)
  let m2l1lc2 := p.m2 * p.l1 * p.lc2
  let c0 :=
    -2.0 * m2l1lc2 * s2 * x.theta2dot * x.theta1dot +
      -m2l1lc2 * s2 * x.theta2dot * x.theta2dot
  let c1 := m2l1lc2 * s2 * x.theta1dot * x.theta1dot
  let g0 :=
    p.gravity * p.m1 * p.lc1 * s1 +
      p.gravity * p.m2 * (p.l1 * s1 + p.lc2 * s12)
  let g1 := p.gravity * p.m2 * p.lc2 * s12
  #[c0 + g0 + p.b1 * x.theta1dot, c1 + g1 + p.b2 * x.theta2dot]

def inputGeneralizedForces (u : AcrobotInput) : Array Float :=
  #[0.0, u.tau]

def manipulatorEquation
    (p : AcrobotParams)
    (u : AcrobotInput)
    (x : AcrobotState) : ManipulatorEquation :=
  {
    massMatrix := massMatrix p x
    qdot := qdotAsArray x
    generalizedForces := inputGeneralizedForces u
    biasForces := dynamicsBiasTerm p x
    label := "acrobot"
  }

def validateFullPhysicsInputs?
    (p : AcrobotParams) (u : AcrobotInput) (x : AcrobotState) :
    Except String Unit := do
  if !p.isValid then
    .error "acrobot params are invalid"
  if !u.isValid then
    .error "acrobot input must have one finite elbow torque coordinate"
  if !x.isValid then
    .error "acrobot state must have four finite coordinates"

def fullPhysicsPrimitives
    (p : AcrobotParams)
    (u : AcrobotInput)
    (x : AcrobotState)
    (label : String := "acrobot") : FullPhysicsPrimitives :=
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
    (p : AcrobotParams := params)
    (label : String := "acrobot full physics provider") :
    FullPhysicsPrimitiveProvider AcrobotPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      validateFullPhysicsInputs? p snapshot.input snapshot.state
      pure (fullPhysicsPrimitives p snapshot.input snapshot.state label)
  }

def solveFullPhysics?
    (p : AcrobotParams)
    (u : AcrobotInput)
    (x : AcrobotState)
    (intervalVertex : VertexId := 5252)
    (label : String := "acrobot") :
    Except String FullPhysicsResult := do
  validateFullPhysicsInputs? p u x
  (fullPhysicsPrimitives p u x label).solve? intervalVertex

def derivative? (p : AcrobotParams) (u : AcrobotInput) (x : AcrobotState) :
    Except String AcrobotState := do
  let d ← (manipulatorEquation p u x).solve?
  pure {
    theta1 := d.qdot.getD 0 0.0
    theta2 := d.qdot.getD 1 0.0
    theta1dot := d.vdot.getD 0 0.0
    theta2dot := d.vdot.getD 1 0.0
  }

def derivative (p : AcrobotParams) (u : AcrobotInput := defaultInput)
    (x : AcrobotState) : AcrobotState :=
  match derivative? p u x with
  | .ok dx => dx
  | .error _ => {}

def proposedDerivativeArrays (dx : AcrobotState) : Array Float × Array Float :=
  (#[dx.theta1, dx.theta2], #[dx.theta1dot, dx.theta2dot])

def implicitResidual
    (p : AcrobotParams)
    (u : AcrobotInput)
    (x : AcrobotState)
    (proposed : AcrobotState) : Array Float :=
  let (proposedQdot, proposedVdot) := proposedDerivativeArrays proposed
  let qResidual := FloatArray.sub proposedQdot (qdotAsArray x)
  let forceResidual :=
    FloatArray.sub
      (FloatMatrix.matVec (massMatrix p x) proposedVdot)
      (FloatArray.sub (inputGeneralizedForces u) (dynamicsBiasTerm p x))
  qResidual ++ forceResidual

def kineticEnergy (p : AcrobotParams) (x : AcrobotState) : Float :=
  let v := qdotAsArray x
  0.5 * FloatArray.dot v (FloatMatrix.matVec (massMatrix p x) v)

def potentialEnergy (p : AcrobotParams) (x : AcrobotState) : Float :=
  let c1 := Float.cos x.theta1
  let c12 := Float.cos (x.theta1 + x.theta2)
  let link1 := p.m1 * p.gravity * p.lc1 * c1
  let link2 := p.m2 * p.gravity * (p.l1 * c1 + p.lc2 * c12)
  0.0 - (link1 + link2)

def totalEnergy (p : AcrobotParams) (x : AcrobotState) : Float :=
  kineticEnergy p x + potentialEnergy p x

private def pi : Float := 3.14159265358979323846

private def twoPi : Float := 2.0 * pi

private def wrapTo (x low high : Float) : Float := Id.run do
  let width := high - low
  if width <= 0.0 then
    return x
  let mut y := x
  for _ in [:32] do
    if y < low then
      y := y + width
    else if y >= high then
      y := y - width
  return y

def uprightState : AcrobotState :=
  { theta1 := pi, theta2 := 0.0, theta1dot := 0.0, theta2dot := 0.0 }

def runPassiveInitialState : AcrobotState :=
  { theta1 := 1.0, theta2 := 1.0, theta1dot := 0.0, theta2dot := 0.0 }

def runLqrInitialState : AcrobotState :=
  { theta1 := pi + 0.1, theta2 := -0.1, theta1dot := 0.0, theta2dot := 0.0 }

def runSwingUpInitialState : AcrobotState :=
  { theta1 := 0.1, theta2 := -0.1, theta1dot := 0.0, theta2dot := 0.02 }

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

def benchmarkAcrobotFactory : String :=
  "drake::multibody::benchmarks::acrobot::MakeAcrobotPlant"

def benchmarkAcrobotSdfUrl : String :=
  "package://drake/multibody/benchmarks/acrobot/acrobot.sdf"

structure MultibodyAcrobotConfig where
  targetRealtimeRate : Float := 1.0
  passiveSimulationTime : Float := 10.0
  passiveIntegrationScheme : MultibodyIntegratorScheme := .rungeKutta3
  targetAccuracy : Float := 0.001
  lqrSimulationTime : Float := 3.0
  timeStepping : Bool := true
  discreteTimeStep : Float := 1.0e-3
  randomizedRollouts : Nat := 5
  deriving Repr, Inhabited

namespace MultibodyAcrobotConfig

def passiveMaxTimeStep (cfg : MultibodyAcrobotConfig) : Float :=
  cfg.passiveSimulationTime / 1000.0

def lqrPlantTimeStep (cfg : MultibodyAcrobotConfig) : Float :=
  if cfg.timeStepping then cfg.discreteTimeStep else 0.0

def validate? (cfg : MultibodyAcrobotConfig) : Except String Unit := do
  if !cfg.targetRealtimeRate.isFinite || cfg.targetRealtimeRate < 0.0 then
    .error s!"multibody acrobot target_realtime_rate must be nonnegative and finite, got {cfg.targetRealtimeRate}"
  if !cfg.passiveSimulationTime.isFinite || cfg.passiveSimulationTime <= 0.0 then
    .error s!"multibody acrobot passive simulation_time must be positive and finite, got {cfg.passiveSimulationTime}"
  if !cfg.targetAccuracy.isFinite || cfg.targetAccuracy <= 0.0 then
    .error s!"multibody acrobot target accuracy must be positive and finite, got {cfg.targetAccuracy}"
  if !cfg.lqrSimulationTime.isFinite || cfg.lqrSimulationTime <= 0.0 then
    .error s!"multibody acrobot LQR simulation_time must be positive and finite, got {cfg.lqrSimulationTime}"
  if !cfg.discreteTimeStep.isFinite || cfg.discreteTimeStep <= 0.0 then
    .error s!"multibody acrobot discrete time step must be positive and finite, got {cfg.discreteTimeStep}"
  if cfg.randomizedRollouts == 0 then
    .error "multibody acrobot LQR should run at least one randomized rollout"

end MultibodyAcrobotConfig

def multibodyAcrobotConfig : MultibodyAcrobotConfig := {}

structure MultibodyAcrobotRandomInitialConditions where
  shoulderMean : Float := pi
  shoulderStddev : Float := 0.02
  elbowMean : Float := 0.0
  elbowStddev : Float := 0.05
  rollouts : Nat := multibodyAcrobotConfig.randomizedRollouts
  deriving Repr, Inhabited

namespace MultibodyAcrobotRandomInitialConditions

def validate? (spec : MultibodyAcrobotRandomInitialConditions) : Except String Unit := do
  if !spec.shoulderMean.isFinite || !spec.elbowMean.isFinite then
    .error "multibody acrobot random initial means must be finite"
  if !spec.shoulderStddev.isFinite || spec.shoulderStddev < 0.0 then
    .error s!"multibody acrobot shoulder stddev must be nonnegative and finite, got {spec.shoulderStddev}"
  if !spec.elbowStddev.isFinite || spec.elbowStddev < 0.0 then
    .error s!"multibody acrobot elbow stddev must be nonnegative and finite, got {spec.elbowStddev}"
  if spec.rollouts == 0 then
    .error "multibody acrobot random initial condition spec requires at least one rollout"

end MultibodyAcrobotRandomInitialConditions

def multibodyAcrobotRandomInitialConditions :
    MultibodyAcrobotRandomInitialConditions := {}

def multibodyAcrobotModel (modelUri : String) (label : String) :
    FullMultibodyPlantModel :=
  {
    modelName := "benchmark_acrobot"
    modelUri := modelUri
    numPositions := 2
    numVelocities := 2
    numActuatedDofs := 1
    finalized := true
    label := label
  }

def multibodyAcrobotPlantConfig (timeStep : Float) :
    MultibodyPlantConfigPrimitive :=
  {
    timeStep := timeStep
    penetrationAllowance := 0.0
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def multibodyAcrobotPassiveStep
    (cfg : MultibodyAcrobotConfig := multibodyAcrobotConfig) :
    FullMultibodyPlantStep :=
  {
    model := multibodyAcrobotModel benchmarkAcrobotFactory
      "benchmark acrobot MakeAcrobotPlant passive model"
    config := multibodyAcrobotPlantConfig 0.0
    q0 := #[1.0, 1.0]
    v0 := #[0.0, 0.0]
    actuation := #[0.0]
    t0 := 0.0
    t1 := cfg.passiveSimulationTime
    label := "multibody-acrobot-passive-full-plant"
  }

def multibodyAcrobotLqrStep
    (cfg : MultibodyAcrobotConfig := multibodyAcrobotConfig) :
    FullMultibodyPlantStep :=
  {
    model := multibodyAcrobotModel benchmarkAcrobotSdfUrl
      "benchmark acrobot parsed SDF LQR model"
    config := multibodyAcrobotPlantConfig cfg.lqrPlantTimeStep
    q0 := #[pi, 0.0]
    v0 := #[0.0, 0.0]
    actuation := #[0.0]
    t0 := 0.0
    t1 := cfg.lqrSimulationTime
    label := "multibody-acrobot-lqr-full-plant"
  }

def multibodyAcrobotLqrQ : Array Float :=
  #[10.0, 10.0, 1.0, 1.0]

def multibodyAcrobotLqrR : Array Float :=
  #[1.0]

private def multibodySegment (id : SegmentId) (t1 : Float) (label : String) :
    AcceptedStepSegment :=
  {
    id := id
    attemptIndex := 0
    tStart := 0.0
    tAttempt := t1
    tAfter := t1
    label := label
  }

private def multibodyLocalMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

def multibodyAcrobotMoves (cfg : MultibodyAcrobotConfig) :
    Array SkeletonMove :=
  #[
    multibodyLocalMove 5250 "MakeAcrobotPlant + SceneGraph + DrakeVisualizer setup",
    multibodyLocalMove 5251 s!"passive integrator {cfg.passiveIntegrationScheme.label} with max step {cfg.passiveMaxTimeStep}",
    multibodyLocalMove 5253 "Parse benchmark acrobot SDF and build balancing LQR controller",
    multibodyLocalMove 5254 "randomized initial contexts for five LQR rollouts"
  ]

structure MultibodyAcrobotResult where
  references : Array DrakeReference
  config : MultibodyAcrobotConfig
  passiveStep : FullMultibodyPlantStep
  lqrStep : FullMultibodyPlantStep
  passiveFullPhysics : FullPhysicsResult
  lqrFullPhysics : FullPhysicsResult
  randomInitialConditions : MultibodyAcrobotRandomInitialConditions
  lqrQ : Array Float
  lqrR : Array Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildMultibodyAcrobot? (cfg : MultibodyAcrobotConfig := multibodyAcrobotConfig) :
    Except String MultibodyAcrobotResult := do
  cfg.validate?
  let randomSpec :=
    { multibodyAcrobotRandomInitialConditions with rollouts := cfg.randomizedRollouts }
  randomSpec.validate?
  let passiveStep := multibodyAcrobotPassiveStep cfg
  let lqrStep := multibodyAcrobotLqrStep cfg
  passiveStep.validate?
  lqrStep.validate?
  let passiveX0 : AcrobotState := {
    theta1 := passiveStep.q0.getD 0 0.0
    theta2 := passiveStep.q0.getD 1 0.0
    theta1dot := passiveStep.v0.getD 0 0.0
    theta2dot := passiveStep.v0.getD 1 0.0
  }
  let passiveU0 : AcrobotInput := { tau := passiveStep.actuation.getD 0 0.0 }
  let passiveFullPhysics ← solveFullPhysics? params passiveU0 passiveX0 5252
    "multibody acrobot passive benchmark plant"
  let lqrX0 : AcrobotState := {
    theta1 := lqrStep.q0.getD 0 0.0
    theta2 := lqrStep.q0.getD 1 0.0
    theta1dot := lqrStep.v0.getD 0 0.0
    theta2dot := lqrStep.v0.getD 1 0.0
  }
  let lqrU0 : AcrobotInput := { tau := lqrStep.actuation.getD 0 0.0 }
  let lqrFullPhysics ← solveFullPhysics? params lqrU0 lqrX0 5255
    "multibody acrobot LQR benchmark plant"
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (multibodySegment 5252 cfg.passiveSimulationTime
        "multibody acrobot passive Simulator.AdvanceTo"))
      |>.push (.interval (multibodySegment 5255 cfg.lqrSimulationTime
        "multibody acrobot LQR Simulator.AdvanceTo"))
  trace.validate?
  pure {
    references := drakeReferences
    config := cfg
    passiveStep := passiveStep
    lqrStep := lqrStep
    passiveFullPhysics := passiveFullPhysics
    lqrFullPhysics := lqrFullPhysics
    randomInitialConditions := randomSpec
    lqrQ := multibodyAcrobotLqrQ
    lqrR := multibodyAcrobotLqrR
    trace := trace
    moves := multibodyAcrobotMoves cfg ++
      #[passiveFullPhysics.supportMove, passiveFullPhysics.move,
        lqrFullPhysics.supportMove, lqrFullPhysics.move] ++
      trace.moves
  }

def stateAddScaled (x dx : AcrobotState) (h : Float) : AcrobotState :=
  {
    theta1 := x.theta1 + h * dx.theta1
    theta2 := x.theta2 + h * dx.theta2
    theta1dot := x.theta1dot + h * dx.theta1dot
    theta2dot := x.theta2dot + h * dx.theta2dot
  }

structure SpongControllerParams where
  kE : Float := 5.0
  kP : Float := 50.0
  kD : Float := 5.0
  balancingThreshold : Float := 1.0e3
  deriving Repr, Inhabited

namespace SpongControllerParams

def isValid (p : SpongControllerParams) : Bool :=
  Float.isFinite p.kE && p.kE >= 0.0 &&
  Float.isFinite p.kP && p.kP >= 0.0 &&
  Float.isFinite p.kD && p.kD >= 0.0 &&
  Float.isFinite p.balancingThreshold && p.balancingThreshold >= 0.0

end SpongControllerParams

def spongControllerParams : SpongControllerParams := {}

structure AcrobotLinearization where
  A : Array (Array Float)
  B : Array (Array Float)
  deriving Repr, Inhabited

def linearizationAboutUpright (p : AcrobotParams := params) :
    AcrobotLinearization :=
  let eq := manipulatorEquation p defaultInput uprightState
  let minvB :=
    match DenseLinearAlgebra.solveLinear? eq.massMatrix #[0.0, 1.0] with
    | .ok col => col
    | .error _ => #[0.0, 0.0]
  let g11 := -p.gravity * (p.m1 * p.lc1 + p.m2 * (p.l1 + p.lc2))
  let g12 := -p.gravity * p.m2 * p.lc2
  let g22 := -p.gravity * p.m2 * p.lc2
  let accelTheta1 :=
    match DenseLinearAlgebra.solveLinear? eq.massMatrix #[g11, g12] with
    | .ok col => col.map (fun x => -x)
    | .error _ => #[0.0, 0.0]
  let accelTheta2 :=
    match DenseLinearAlgebra.solveLinear? eq.massMatrix #[g12, g22] with
    | .ok col => col.map (fun x => -x)
    | .error _ => #[0.0, 0.0]
  let accelVelocity1 :=
    match DenseLinearAlgebra.solveLinear? eq.massMatrix #[p.b1, 0.0] with
    | .ok col => col.map (fun x => -x)
    | .error _ => #[0.0, 0.0]
  let accelVelocity2 :=
    match DenseLinearAlgebra.solveLinear? eq.massMatrix #[0.0, p.b2] with
    | .ok col => col.map (fun x => -x)
    | .error _ => #[0.0, 0.0]
  {
    A := #[
      #[0.0, 0.0, 1.0, 0.0],
      #[0.0, 0.0, 0.0, 1.0],
      #[accelTheta1.getD 0 0.0, accelTheta2.getD 0 0.0,
        accelVelocity1.getD 0 0.0, accelVelocity2.getD 0 0.0],
      #[accelTheta1.getD 1 0.0, accelTheta2.getD 1 0.0,
        accelVelocity1.getD 1 0.0, accelVelocity2.getD 1 0.0]
    ]
    B := #[#[0.0], #[0.0], #[minvB.getD 0 0.0], #[minvB.getD 1 0.0]]
  }

/--
Steady-state LQR data for Drake's default `BalancingLQRController`.

The values are the stabilizing continuous-time Riccati solution for the
linearization above, using Drake's `Q = diag(10, 10, 1, 1)` and `R = 1`.
They are kept as an explicit local Schur block because Drake obtains them by
calling its dense LQR solver.
-/
structure BalancingLqrData where
  S : Array (Array Float)
  K : Array Float
  deriving Repr, Inhabited

def balancingLqrData : BalancingLqrData :=
  {
    S := #[
      #[16620.60660509, 7470.18733996, 7240.12368132, 3571.58116264],
      #[7470.18733996, 3374.43640788, 3256.40272515, 1608.54161651],
      #[7240.12368132, 3256.40272515, 3154.73036208, 1556.50607288],
      #[3571.58116264, 1608.54161651, 1556.50607288, 768.33308412]
    ]
    K := #[-278.44223125, -112.29125984, -119.72457376, -56.82824017]
  }

inductive SpongMode where
  | balancing
  | swingUp
  deriving Repr, BEq, Inhabited

namespace SpongMode

def id : SpongMode → Nat
  | .balancing => 0
  | .swingUp => 1

def label : SpongMode → String
  | .balancing => "balancing-lqr"
  | .swingUp => "spong-swing-up"

end SpongMode

structure SpongControlOutput where
  tau : Float
  mode : SpongMode
  cost : Float
  energyError : Float
  partialFeedbackTorque : Float
  energyTorque : Float
  saturated : Bool
  deriving Repr, Inhabited

private def controllerStateVector (x : AcrobotState) : Array Float :=
  #[
    wrapTo x.theta1 0.0 twoPi,
    wrapTo x.theta2 (-pi) pi,
    x.theta1dot,
    x.theta2dot
  ]

private def controllerError (x : AcrobotState) : Array Float :=
  FloatArray.sub (controllerStateVector x) (stateAsArray uprightState)

def balancingCost (lqr : BalancingLqrData := balancingLqrData)
    (x : AcrobotState) : Float :=
  let e := controllerError x
  FloatArray.dot e (FloatMatrix.matVec lqr.S e)

private def clampTorque (u : Float) : Float × Bool :=
  if u >= 20.0 then
    (20.0, true)
  else if u <= -20.0 then
    (-20.0, true)
  else
    (u, false)

def balancingLqrController (lqr : BalancingLqrData := balancingLqrData)
    (x : AcrobotState) : AcrobotInput :=
  let errToTarget := FloatArray.sub (stateAsArray uprightState) (controllerStateVector x)
  let (tau, _) := clampTorque (FloatArray.dot lqr.K errToTarget)
  { tau := tau }

def desiredSwingUpEnergy (p : AcrobotParams := params) : Float :=
  (p.m1 * p.lc1 + p.m2 * (p.l1 + p.lc2)) * p.gravity

def spongController? (p : AcrobotParams := params)
    (controllerParams : SpongControllerParams := spongControllerParams)
    (lqr : BalancingLqrData := balancingLqrData)
    (x : AcrobotState) : Except String SpongControlOutput := do
  if !p.isFiniteForDynamics then
    .error "acrobot params are not finite"
  if !controllerParams.isValid then
    .error "Spong controller params are invalid"
  if !x.isValid then
    .error "Spong controller state is invalid"
  let cost := balancingCost lqr x
  if cost < controllerParams.balancingThreshold then
    let raw := (balancingLqrController lqr x).tau
    let (tau, saturated) := clampTorque raw
    pure {
      tau := tau
      mode := .balancing
      cost := cost
      energyError := totalEnergy p x - desiredSwingUpEnergy p
      partialFeedbackTorque := tau
      energyTorque := 0.0
      saturated := saturated
    }
  else
    let minv0 ← DenseLinearAlgebra.solveLinear? (massMatrix p x) #[1.0, 0.0]
    let minv1 ← DenseLinearAlgebra.solveLinear? (massMatrix p x) #[0.0, 1.0]
    let a2 := minv0.getD 1 0.0
    let a3 := minv1.getD 1 0.0
    if Float.abs a3 < 1.0e-12 then
      .error "Spong controller encountered singular partial-feedback denominator"
    let bias := dynamicsBiasTerm p x
    let energyError := totalEnergy p x - desiredSwingUpEnergy p
    let energyTorque := -controllerParams.kE * energyError * x.theta2dot
    let y := -controllerParams.kP * x.theta2 - controllerParams.kD * x.theta2dot
    let partialFeedbackTorque := (a2 * bias.getD 0 0.0 + y) / a3 + bias.getD 1 0.0
    let (tau, saturated) := clampTorque (energyTorque + partialFeedbackTorque)
    pure {
      tau := tau
      mode := .swingUp
      cost := cost
      energyError := energyError
      partialFeedbackTorque := partialFeedbackTorque
      energyTorque := energyTorque
      saturated := saturated
    }

def spongController (p : AcrobotParams := params)
    (controllerParams : SpongControllerParams := spongControllerParams)
    (lqr : BalancingLqrData := balancingLqrData)
    (x : AcrobotState) : AcrobotInput :=
  match spongController? p controllerParams lqr x with
  | .ok out => { tau := out.tau }
  | .error _ => defaultInput

private def rk4ClosedLoopStep
    (p : AcrobotParams)
    (controller : AcrobotState → AcrobotInput)
    (dt : Float)
    (x : AcrobotState) : AcrobotState :=
  let f := fun y => derivative p (controller y) y
  let k1 := f x
  let k2 := f (stateAddScaled x k1 (0.5 * dt))
  let k3 := f (stateAddScaled x k2 (0.5 * dt))
  let k4 := f (stateAddScaled x k3 dt)
  {
    theta1 := x.theta1 + dt * (k1.theta1 + 2.0 * k2.theta1 + 2.0 * k3.theta1 + k4.theta1) / 6.0
    theta2 := x.theta2 + dt * (k1.theta2 + 2.0 * k2.theta2 + 2.0 * k3.theta2 + k4.theta2) / 6.0
    theta1dot := x.theta1dot +
      dt * (k1.theta1dot + 2.0 * k2.theta1dot + 2.0 * k3.theta1dot + k4.theta1dot) / 6.0
    theta2dot := x.theta2dot +
      dt * (k1.theta2dot + 2.0 * k2.theta2dot + 2.0 * k3.theta2dot + k4.theta2dot) / 6.0
  }

def odeTerm (p : AcrobotParams) : ODETerm AcrobotState AcrobotInput :=
  { vectorField := fun _t x u => derivative p u x }

def acrobotSolver :=
  RK4.solver
    (Term := ODETerm AcrobotState AcrobotInput)
    (Y := AcrobotState)
    (VF := AcrobotState)
    (Args := AcrobotInput)

structure SimulationResult where
  references : Array DrakeReference
  t0 : Float
  t1 : Float
  input : AcrobotInput
  initialState : AcrobotState
  finalState : AcrobotState
  initialEnergy : Float
  finalEnergy : Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

structure ControllerSimulationConfig where
  duration : Float := 10.0
  stepSize : Float := 1.0e-3
  steps : Nat := 10000
  deriving Repr, Inhabited

namespace ControllerSimulationConfig

def validate? (cfg : ControllerSimulationConfig) (label : String) :
    Except String Unit := do
  if !cfg.duration.isFinite || cfg.duration <= 0.0 then
    .error s!"{label} duration must be positive and finite, got {cfg.duration}"
  if !cfg.stepSize.isFinite || cfg.stepSize <= 0.0 then
    .error s!"{label} step size must be positive and finite, got {cfg.stepSize}"
  if cfg.steps == 0 then
    .error s!"{label} rollout requires at least one step"
  if Float.abs (cfg.steps.toFloat * cfg.stepSize - cfg.duration) > 1.0e-12 then
    .error s!"{label} step count {cfg.steps} does not match duration {cfg.duration}"

end ControllerSimulationConfig

def controllerSimulationConfig : ControllerSimulationConfig := {}

structure SpongModeSummary where
  balancingSteps : Nat := 0
  swingUpSteps : Nat := 0
  support : RuntimeSupport := { policy := .fullSupport }
  deriving Repr, Inhabited

namespace SpongModeSummary

def observe (summary : SpongModeSummary) (mode : SpongMode) : SpongModeSummary :=
  match mode with
  | .balancing => { summary with balancingSteps := summary.balancingSteps + 1 }
  | .swingUp => { summary with swingUpSteps := summary.swingUpSteps + 1 }

def fromCounts (balancingSteps swingUpSteps : Nat) : SpongModeSummary :=
  let ids :=
    (if balancingSteps > 0 then #[SpongMode.balancing.id] else #[]) ++
    (if swingUpSteps > 0 then #[SpongMode.swingUp.id] else #[])
  let support :=
    if ids.size == 1 then
      {
        policy := .deterministicPick ids[0]!
        selectedIds := ids
        totalCandidates? := some 2
        label := "acrobot-spong-controller-mode"
      }
    else
      {
        policy := .threshold spongControllerParams.balancingThreshold
        selectedIds := ids
        totalCandidates? := some 2
        label := "acrobot-spong-controller-mode"
      }
  {
    balancingSteps := balancingSteps
    swingUpSteps := swingUpSteps
    support := support
  }

def totalSteps (summary : SpongModeSummary) : Nat :=
  summary.balancingSteps + summary.swingUpSteps

def sawBothModes (summary : SpongModeSummary) : Bool :=
  summary.balancingSteps > 0 && summary.swingUpSteps > 0

end SpongModeSummary

structure ControllerSimulationResult where
  references : Array DrakeReference
  controllerName : String
  t0 : Float
  t1 : Float
  stepSize : Float
  initialState : AcrobotState
  finalState : AcrobotState
  samples : Array AcrobotState
  initialEnergy : Float
  finalEnergy : Float
  modeSummary : SpongModeSummary
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
    label := "acrobot-continuous-interval"
  }

private def controllerMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

structure AcrobotLcmChannels where
  stateEstimateChannel : String := "acrobot_xhat"
  commandChannel : String := "acrobot_u"
  deriving Repr, Inhabited

namespace AcrobotLcmChannels

def validate? (channels : AcrobotLcmChannels) : Except String Unit := do
  if channels.stateEstimateChannel == "" then
    .error "acrobot LCM state-estimate channel must be nonempty"
  if channels.commandChannel == "" then
    .error "acrobot LCM command channel must be nonempty"
  if channels.stateEstimateChannel == channels.commandChannel then
    .error "acrobot LCM state and command channels must be distinct"

end AcrobotLcmChannels

def acrobotLcmChannels : AcrobotLcmChannels := {}

structure LcmtAcrobotX where
  theta1 : Float := 0.0
  theta2 : Float := 0.0
  theta1Dot : Float := 0.0
  theta2Dot : Float := 0.0
  deriving Repr, Inhabited

namespace LcmtAcrobotX

def isValid (msg : LcmtAcrobotX) : Bool :=
  Float.isFinite msg.theta1 &&
  Float.isFinite msg.theta2 &&
  Float.isFinite msg.theta1Dot &&
  Float.isFinite msg.theta2Dot

def fromState (x : AcrobotState) : LcmtAcrobotX :=
  {
    theta1 := x.theta1
    theta2 := x.theta2
    theta1Dot := x.theta1dot
    theta2Dot := x.theta2dot
  }

def toState (msg : LcmtAcrobotX) : AcrobotState :=
  {
    theta1 := msg.theta1
    theta2 := msg.theta2
    theta1dot := msg.theta1Dot
    theta2dot := msg.theta2Dot
  }

end LcmtAcrobotX

structure LcmtAcrobotU where
  tau : Float := 0.0
  deriving Repr, Inhabited

namespace LcmtAcrobotU

def isValid (msg : LcmtAcrobotU) : Bool :=
  Float.isFinite msg.tau

def fromInput (u : AcrobotInput) : LcmtAcrobotU :=
  { tau := u.tau }

def toInput (msg : LcmtAcrobotU) : AcrobotInput :=
  { tau := msg.tau }

end LcmtAcrobotU

structure AcrobotLcmIoBoundary where
  headerPath : String := "../drake/examples/acrobot/acrobot_lcm.h"
  implementationPath : String := "../drake/examples/acrobot/acrobot_lcm.cc"
  stateMessageType : String := "lcmt_acrobot_x"
  commandMessageType : String := "lcmt_acrobot_u"
  stateReceiverInputPort : String := "lcmt_acrobot_x"
  stateReceiverOutputPort : String := "acrobot_state"
  commandSenderInputPort : String := "elbow_torque"
  commandSenderOutputPort : String := "lcmt_acrobot_u"
  commandReceiverInputPort : String := "lcmt_acrobot_u"
  commandReceiverOutputPort : String := "elbow_torque"
  stateSenderInputPort : String := "acrobot_state"
  stateSenderOutputPort : String := "lcmt_acrobot_x"
  stateVectorSize : Nat := 4
  commandVectorSize : Nat := 1
  deriving Repr, Inhabited

namespace AcrobotLcmIoBoundary

def validate? (boundary : AcrobotLcmIoBoundary) : Except String Unit := do
  if boundary.headerPath != "../drake/examples/acrobot/acrobot_lcm.h" then
    .error s!"Acrobot LCM header path mismatch: {boundary.headerPath}"
  if boundary.implementationPath != "../drake/examples/acrobot/acrobot_lcm.cc" then
    .error s!"Acrobot LCM implementation path mismatch: {boundary.implementationPath}"
  if boundary.stateMessageType != "lcmt_acrobot_x" ||
      boundary.commandMessageType != "lcmt_acrobot_u" then
    .error s!"Acrobot LCM message types mismatch: {boundary.stateMessageType}, {boundary.commandMessageType}"
  if boundary.stateReceiverInputPort != "lcmt_acrobot_x" ||
      boundary.stateReceiverOutputPort != "acrobot_state" then
    .error "AcrobotStateReceiver port names should be lcmt_acrobot_x -> acrobot_state"
  if boundary.commandSenderInputPort != "elbow_torque" ||
      boundary.commandSenderOutputPort != "lcmt_acrobot_u" then
    .error "AcrobotCommandSender port names should be elbow_torque -> lcmt_acrobot_u"
  if boundary.commandReceiverInputPort != "lcmt_acrobot_u" ||
      boundary.commandReceiverOutputPort != "elbow_torque" then
    .error "AcrobotCommandReceiver port names should be lcmt_acrobot_u -> elbow_torque"
  if boundary.stateSenderInputPort != "acrobot_state" ||
      boundary.stateSenderOutputPort != "lcmt_acrobot_x" then
    .error "AcrobotStateSender port names should be acrobot_state -> lcmt_acrobot_x"
  if boundary.stateVectorSize != 4 then
    .error s!"Acrobot LCM state vector size should be 4, got {boundary.stateVectorSize}"
  if boundary.commandVectorSize != 1 then
    .error s!"Acrobot LCM command vector size should be 1, got {boundary.commandVectorSize}"

end AcrobotLcmIoBoundary

def acrobotLcmIoBoundary : AcrobotLcmIoBoundary := {}

structure AcrobotLcmIoSample where
  boundary : AcrobotLcmIoBoundary
  stateIn : AcrobotState
  stateMessage : LcmtAcrobotX
  stateOut : AcrobotState
  commandIn : AcrobotInput
  commandMessage : LcmtAcrobotU
  commandOut : AcrobotInput
  move : SkeletonMove
  deriving Repr, Inhabited

def buildAcrobotLcmIoSample?
    (boundary : AcrobotLcmIoBoundary := acrobotLcmIoBoundary)
    (x : AcrobotState := { theta1 := 1.0, theta2 := 2.0, theta1dot := 3.0, theta2dot := 4.0 })
    (u : AcrobotInput := { tau := -5.0 }) :
    Except String AcrobotLcmIoSample := do
  boundary.validate?
  if !x.isValid then
    .error "Acrobot LCM state sample is invalid"
  if !u.isValid then
    .error "Acrobot LCM command sample is invalid"
  let xMsg := LcmtAcrobotX.fromState x
  let uMsg := LcmtAcrobotU.fromInput u
  if !xMsg.isValid || !uMsg.isValid then
    .error "Acrobot LCM message sample is invalid"
  pure {
    boundary := boundary
    stateIn := x
    stateMessage := xMsg
    stateOut := xMsg.toState
    commandIn := u
    commandMessage := uMsg
    commandOut := uMsg.toInput
    move := controllerMove 5219 "Acrobot LCM message conversion systems"
  }

structure AcrobotLcmPlantBoundary where
  channels : AcrobotLcmChannels := acrobotLcmChannels
  simulationTime? : Option Float := none
  targetRealtimeRate : Float := 1.0
  initialState : AcrobotState :=
    { theta1 := 0.1, theta2 := 0.1, theta1dot := 0.0, theta2dot := 0.0 }
  visualizerEnabled : Bool := true
  deriving Repr, Inhabited

namespace AcrobotLcmPlantBoundary

def validate? (boundary : AcrobotLcmPlantBoundary) : Except String Unit := do
  boundary.channels.validate?
  match boundary.simulationTime? with
  | some t =>
      if !t.isFinite || t <= 0.0 then
        .error s!"acrobot LCM plant finite simulation time must be positive, got {t}"
  | none => pure ()
  if !boundary.targetRealtimeRate.isFinite || boundary.targetRealtimeRate < 0.0 then
    .error s!"acrobot LCM plant realtime factor must be nonnegative and finite, got {boundary.targetRealtimeRate}"
  if !boundary.initialState.isValid then
    .error "acrobot LCM plant initial state is invalid"

end AcrobotLcmPlantBoundary

def acrobotLcmPlantBoundary : AcrobotLcmPlantBoundary := {}

structure AcrobotLcmControllerBoundary where
  channels : AcrobotLcmChannels := acrobotLcmChannels
  timeLimit? : Option Float := none
  targetRealtimeRate : Float := 1.0
  commandPublishPeriod : Float := 1.0e-3
  controllerName : String := "AcrobotSpongController"
  deriving Repr, Inhabited

namespace AcrobotLcmControllerBoundary

def validate? (boundary : AcrobotLcmControllerBoundary) : Except String Unit := do
  boundary.channels.validate?
  match boundary.timeLimit? with
  | some t =>
      if !t.isFinite || t <= 0.0 then
        .error s!"acrobot LCM controller time limit must be positive, got {t}"
  | none => pure ()
  if !boundary.targetRealtimeRate.isFinite || boundary.targetRealtimeRate < 0.0 then
    .error s!"acrobot LCM controller realtime factor must be nonnegative and finite, got {boundary.targetRealtimeRate}"
  if !boundary.commandPublishPeriod.isFinite || boundary.commandPublishPeriod <= 0.0 then
    .error s!"acrobot LCM controller publish period must be positive and finite, got {boundary.commandPublishPeriod}"
  if boundary.controllerName == "" then
    .error "acrobot LCM controller name must be nonempty"

end AcrobotLcmControllerBoundary

def acrobotLcmControllerBoundary : AcrobotLcmControllerBoundary := {}

structure AcrobotEstimatorLqrBoundary where
  simulationTime : Float := 5.0
  targetRealtimeRate : Float := 1.0
  maximumStepSize : Float := 0.01
  fixedStepMode : Bool := true
  plantInitialState : AcrobotState := runLqrInitialState
  observerInitialState : AcrobotState := uprightState
  encoderCount : Nat := 2
  processNoiseDiagonal : Array Float := #[1.0, 1.0, 1.0, 1.0]
  measurementNoiseDiagonal : Array Float := #[0.1, 0.1]
  logsTrueAndEstimatedState : Bool := true
  checksObservability : Bool := true
  deriving Repr, Inhabited

namespace AcrobotEstimatorLqrBoundary

private def allFiniteNonnegative (xs : Array Float) : Bool :=
  xs.all (fun x => Float.isFinite x && x >= 0.0)

def validate? (boundary : AcrobotEstimatorLqrBoundary) : Except String Unit := do
  if !boundary.simulationTime.isFinite || boundary.simulationTime <= 0.0 then
    .error s!"acrobot estimator LQR simulation time must be positive and finite, got {boundary.simulationTime}"
  if !boundary.targetRealtimeRate.isFinite || boundary.targetRealtimeRate < 0.0 then
    .error s!"acrobot estimator LQR realtime factor must be nonnegative and finite, got {boundary.targetRealtimeRate}"
  if !boundary.maximumStepSize.isFinite || boundary.maximumStepSize <= 0.0 then
    .error s!"acrobot estimator LQR max step must be positive and finite, got {boundary.maximumStepSize}"
  if !boundary.plantInitialState.isValid then
    .error "acrobot estimator LQR plant initial state is invalid"
  if !boundary.observerInitialState.isValid then
    .error "acrobot estimator LQR observer initial state is invalid"
  if boundary.encoderCount != 2 then
    .error s!"acrobot estimator LQR should expose two rotary encoder outputs, got {boundary.encoderCount}"
  if boundary.processNoiseDiagonal.size != 4 || !allFiniteNonnegative boundary.processNoiseDiagonal then
    .error "acrobot estimator LQR process noise diagonal must have four finite nonnegative entries"
  if boundary.measurementNoiseDiagonal.size != 2 || !allFiniteNonnegative boundary.measurementNoiseDiagonal then
    .error "acrobot estimator LQR measurement noise diagonal must have two finite nonnegative entries"

end AcrobotEstimatorLqrBoundary

def acrobotEstimatorLqrBoundary : AcrobotEstimatorLqrBoundary := {}

inductive AcrobotScenarioVector where
  | deterministic (value : Array Float)
  | uniform (min max : Array Float)
  deriving Repr, Inhabited

namespace AcrobotScenarioVector

def dimension : AcrobotScenarioVector → Nat
  | .deterministic value => value.size
  | .uniform min _ => min.size

def isStochastic : AcrobotScenarioVector → Bool
  | .deterministic _ => false
  | .uniform _ _ => true

def center : AcrobotScenarioVector → Array Float
  | .deterministic value => value
  | .uniform min max => Id.run do
      let mut out : Array Float := #[]
      let n := Nat.min min.size max.size
      for i in [:n] do
        out := out.push ((min[i]! + max[i]!) / 2.0)
      return out

private def validateFinite (label : String) (xs : Array Float) :
    Except String Unit := do
  for x in xs do
    if !x.isFinite then
      .error s!"{label} contains non-finite entry {x}"

def validate? (expectedDim : Nat) (label : String) :
    AcrobotScenarioVector → Except String Unit
  | .deterministic value => do
      if value.size != expectedDim then
        .error s!"{label} dimension {value.size} != expected {expectedDim}"
      validateFinite label value
  | .uniform min max => do
      if min.size != expectedDim || max.size != expectedDim then
        .error s!"{label} uniform bounds have dimensions {min.size}/{max.size}, expected {expectedDim}"
      validateFinite (label ++ " uniform min") min
      validateFinite (label ++ " uniform max") max
      for i in [:expectedDim] do
        if min[i]! > max[i]! then
          .error s!"{label} uniform min exceeds max at index {i}"

end AcrobotScenarioVector

structure AcrobotSpongScenario where
  controllerParams : AcrobotScenarioVector :=
    .deterministic #[5.0, 50.0, 5.0, 1.0e3]
  initialState : AcrobotScenarioVector :=
    .deterministic #[1.2, 0.0, 0.0, 0.0]
  tFinal : Float := 30.0
  tapePeriod : Float := 0.05
  deriving Repr, Inhabited

namespace AcrobotSpongScenario

def isStochastic (scenario : AcrobotSpongScenario) : Bool :=
  scenario.controllerParams.isStochastic || scenario.initialState.isStochastic

def support (scenario : AcrobotSpongScenario) : RuntimeSupport :=
  if scenario.isStochastic then
    {
      policy := .sampled 0
      selectedIds := #[0]
      totalCandidates? := some 1
      label := "acrobot-spong-stochastic-scenario-sample"
    }
  else
    {
      policy := .deterministicPick 0
      selectedIds := #[0]
      totalCandidates? := some 1
      label := "acrobot-spong-deterministic-scenario"
    }

def validate? (scenario : AcrobotSpongScenario) : Except String Unit := do
  scenario.controllerParams.validate? 4 "acrobot controller_params"
  scenario.initialState.validate? 4 "acrobot initial_state"
  if !scenario.tFinal.isFinite || scenario.tFinal <= 0.0 then
    .error s!"acrobot scenario t_final must be positive and finite, got {scenario.tFinal}"
  if !scenario.tapePeriod.isFinite || scenario.tapePeriod <= 0.0 then
    .error s!"acrobot scenario tape_period must be positive and finite, got {scenario.tapePeriod}"

end AcrobotSpongScenario

def exampleSpongScenario : AcrobotSpongScenario := {}

def exampleStochasticSpongScenario : AcrobotSpongScenario :=
  {
    controllerParams := .uniform
      #[4.0, 40.0, 4.0, 0.9e3]
      #[6.0, 60.0, 6.0, 1.1e3]
    initialState := .uniform
      #[1.1, -0.1, -0.1, -0.1]
      #[1.3, 0.1, 0.1, 0.1]
    tFinal := 30.0
    tapePeriod := 0.05
  }

inductive AcrobotOptimizerMetric where
  | ensembleCost
  | successRate
  deriving Repr, BEq, Inhabited

namespace AcrobotOptimizerMetric

def label : AcrobotOptimizerMetric → String
  | .ensembleCost => "ensemble_cost"
  | .successRate => "success_rate"

end AcrobotOptimizerMetric

structure AcrobotOptimizerDemoConfig where
  scenarioPath : String :=
    "../drake/examples/acrobot/test/example_stochastic_scenario.yaml"
  metric : AcrobotOptimizerMetric := .ensembleCost
  ensembleSize : Nat := 10
  numEvaluations : Nat := 250
  runnerResource : String := "drake/examples/acrobot/spong_sim_main_cc"
  deriving Repr, Inhabited

namespace AcrobotOptimizerDemoConfig

def totalRollouts (cfg : AcrobotOptimizerDemoConfig) : Nat :=
  cfg.ensembleSize * cfg.numEvaluations

def seedIds (cfg : AcrobotOptimizerDemoConfig) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:cfg.ensembleSize] do
    out := out.push (i + 1)
  return out

def support (cfg : AcrobotOptimizerDemoConfig) : RuntimeSupport :=
  {
    policy := .fullSupport
    selectedIds := cfg.seedIds
    totalCandidates? := some cfg.ensembleSize
    label := "acrobot-optimizer-ensemble-seeds"
  }

def validate? (cfg : AcrobotOptimizerDemoConfig) : Except String Unit := do
  if cfg.scenarioPath == "" then
    .error "acrobot optimizer scenario path must be nonempty"
  if cfg.runnerResource == "" then
    .error "acrobot optimizer runner resource must be nonempty"
  if cfg.ensembleSize == 0 then
    .error "acrobot optimizer ensemble_size must be positive"
  if cfg.numEvaluations == 0 then
    .error "acrobot optimizer num_evaluations must be positive"

end AcrobotOptimizerDemoConfig

def acrobotOptimizerDemoConfig : AcrobotOptimizerDemoConfig := {}

def nominalSpongTapeRows : Nat := 4
def nominalSpongTapeCols : Nat := 601

structure SpongSimLibraryPyTestBoundary where
  testPath : String := "../drake/examples/acrobot/test/spong_sim_lib_py_test.py"
  functionName : String := "examples.acrobot.spong_sim.simulate"
  initialState : Array Float := #[0.01, 0.02, 0.03, 0.04]
  controllerParams : Array Float := #[0.001, 0.002, 0.003, 0.004]
  tFinal : Float := 30.0
  tapePeriod : Float := 0.05
  expectedRows : Nat := nominalSpongTapeRows
  expectedCols : Nat := nominalSpongTapeCols
  deriving Repr, Inhabited

namespace SpongSimLibraryPyTestBoundary

private def allFinite (xs : Array Float) : Bool :=
  xs.all Float.isFinite

def validate? (boundary : SpongSimLibraryPyTestBoundary) :
    Except String Unit := do
  if boundary.testPath != "../drake/examples/acrobot/test/spong_sim_lib_py_test.py" then
    .error s!"Spong sim library Python test path mismatch: {boundary.testPath}"
  if boundary.functionName != "examples.acrobot.spong_sim.simulate" then
    .error s!"Spong sim library boundary should call simulate(), got {boundary.functionName}"
  if boundary.initialState.size != 4 || !allFinite boundary.initialState then
    .error s!"Spong sim library initial_state must have four finite entries, got {boundary.initialState}"
  if boundary.controllerParams.size != 4 || !allFinite boundary.controllerParams then
    .error s!"Spong sim library controller_params must have four finite entries, got {boundary.controllerParams}"
  if !boundary.tFinal.isFinite || boundary.tFinal <= 0.0 then
    .error s!"Spong sim library t_final must be positive and finite, got {boundary.tFinal}"
  if !boundary.tapePeriod.isFinite || boundary.tapePeriod <= 0.0 then
    .error s!"Spong sim library tape_period must be positive and finite, got {boundary.tapePeriod}"
  if boundary.expectedRows != nominalSpongTapeRows ||
      boundary.expectedCols != nominalSpongTapeCols then
    .error s!"Spong sim library expected tape shape should be (4, 601), got ({boundary.expectedRows}, {boundary.expectedCols})"

end SpongSimLibraryPyTestBoundary

def spongSimLibraryPyTestBoundary : SpongSimLibraryPyTestBoundary := {}

inductive SpongSimBackend where
  | py
  | cc
  deriving Repr, BEq, Inhabited

namespace SpongSimBackend

def suffix : SpongSimBackend → String
  | .py => "py"
  | .cc => "cc"

def resource : SpongSimBackend → String
  | .py => "drake/examples/acrobot/spong_sim_main_py"
  | .cc => "drake/examples/acrobot/spong_sim_main_cc"

def helpReturnCode : SpongSimBackend → Int
  | .py => 0
  | .cc => 1

def supportsStochasticScenario : SpongSimBackend → Bool
  | .py => false
  | .cc => true

end SpongSimBackend

structure SpongSimMainTestBoundary where
  backend : SpongSimBackend := .py
  testPath : String := "../drake/examples/acrobot/test/spong_sim_main_test.py"
  executableResource : String := SpongSimBackend.resource .py
  helpCommand : Array String := #["spong_sim_main_py", "--help"]
  helpNeedle : String := "spong-controlled acrobot"
  expectedHelpReturnCode : Int := 0
  expectedHelpStderrEmpty : Bool := true
  scenarioPath : String := "../drake/examples/acrobot/test/example_scenario.yaml"
  stochasticScenarioPath : String :=
    "../drake/examples/acrobot/test/example_stochastic_scenario.yaml"
  outputFileEnv : String := "TEST_TMPDIR/output.yaml"
  dumpScenarioFileEnv : String := "TEST_TMPDIR/scenario_out.yaml"
  deterministicCommand : Array String :=
    #["spong_sim_main_py", "--scenario", "example_scenario.yaml", "--output", "output.yaml"]
  stochasticCommand : Array String :=
    #["spong_sim_main_py", "--scenario", "example_stochastic_scenario.yaml", "--output", "output.yaml", "--dump_scenario", "scenario_out.yaml"]
  supportsStochasticScenario : Bool := false
  expectedRows : Nat := nominalSpongTapeRows
  expectedCols : Nat := nominalSpongTapeCols
  expectedDumpControllerParams : Nat := 4
  expectedDumpInitialState : Nat := 4
  deriving Repr, Inhabited

namespace SpongSimMainTestBoundary

def forBackend (backend : SpongSimBackend) : SpongSimMainTestBoundary :=
  let exe := "spong_sim_main_" ++ backend.suffix
  {
    backend := backend
    executableResource := backend.resource
    helpCommand := #[exe, "--help"]
    expectedHelpReturnCode := backend.helpReturnCode
    deterministicCommand :=
      #[exe, "--scenario", "example_scenario.yaml", "--output", "output.yaml"]
    stochasticCommand :=
      #[exe, "--scenario", "example_stochastic_scenario.yaml", "--output", "output.yaml",
        "--dump_scenario", "scenario_out.yaml"]
    supportsStochasticScenario := backend.supportsStochasticScenario
  }

def validate? (boundary : SpongSimMainTestBoundary) : Except String Unit := do
  if boundary.testPath != "../drake/examples/acrobot/test/spong_sim_main_test.py" then
    .error s!"Spong sim main Python test path mismatch: {boundary.testPath}"
  if boundary.executableResource != boundary.backend.resource then
    .error s!"Spong sim main resource should be {boundary.backend.resource}, got {boundary.executableResource}"
  if boundary.helpCommand != #["spong_sim_main_" ++ boundary.backend.suffix, "--help"] then
    .error s!"Spong sim main help command mismatch: {boundary.helpCommand}"
  if boundary.helpNeedle != "spong-controlled acrobot" then
    .error s!"Spong sim main help output should mention spong-controlled acrobot, got {boundary.helpNeedle}"
  if boundary.expectedHelpReturnCode != boundary.backend.helpReturnCode then
    .error s!"Spong sim main help return code mismatch: {boundary.expectedHelpReturnCode}"
  if !boundary.expectedHelpStderrEmpty then
    .error "Spong sim main help test expects empty stderr"
  if boundary.scenarioPath != "../drake/examples/acrobot/test/example_scenario.yaml" then
    .error s!"Spong sim main deterministic scenario path mismatch: {boundary.scenarioPath}"
  if boundary.stochasticScenarioPath != "../drake/examples/acrobot/test/example_stochastic_scenario.yaml" then
    .error s!"Spong sim main stochastic scenario path mismatch: {boundary.stochasticScenarioPath}"
  if boundary.outputFileEnv != "TEST_TMPDIR/output.yaml" then
    .error s!"Spong sim main output should be under TEST_TMPDIR/output.yaml, got {boundary.outputFileEnv}"
  if boundary.dumpScenarioFileEnv != "TEST_TMPDIR/scenario_out.yaml" then
    .error s!"Spong sim main scenario dump should be under TEST_TMPDIR/scenario_out.yaml, got {boundary.dumpScenarioFileEnv}"
  if boundary.deterministicCommand !=
      #["spong_sim_main_" ++ boundary.backend.suffix, "--scenario", "example_scenario.yaml", "--output", "output.yaml"] then
    .error s!"Spong sim main deterministic command mismatch: {boundary.deterministicCommand}"
  if boundary.stochasticCommand !=
      #["spong_sim_main_" ++ boundary.backend.suffix, "--scenario", "example_stochastic_scenario.yaml", "--output", "output.yaml",
        "--dump_scenario", "scenario_out.yaml"] then
    .error s!"Spong sim main stochastic command mismatch: {boundary.stochasticCommand}"
  if boundary.supportsStochasticScenario != boundary.backend.supportsStochasticScenario then
    .error s!"Spong sim main stochastic support mismatch for backend {reprStr boundary.backend}"
  if boundary.expectedRows != nominalSpongTapeRows ||
      boundary.expectedCols != nominalSpongTapeCols then
    .error s!"Spong sim main expected tape shape should be (4, 601), got ({boundary.expectedRows}, {boundary.expectedCols})"
  if boundary.expectedDumpControllerParams != 4 || boundary.expectedDumpInitialState != 4 then
    .error "Spong sim main stochastic scenario dump should preserve four controller params and four initial states"

end SpongSimMainTestBoundary

def spongSimMainPyTestBoundary : SpongSimMainTestBoundary :=
  SpongSimMainTestBoundary.forBackend .py

def spongSimMainCcTestBoundary : SpongSimMainTestBoundary :=
  SpongSimMainTestBoundary.forBackend .cc

structure SwingUpTrajectoryOptimizationBoundary where
  testPath : String := "../drake/examples/acrobot/test/run_swing_up_traj_optimization.cc"
  requiresSnopt : Bool := true
  snoptUnavailableReturnCode : Int := 0
  solverFailureReturnCode : Int := 1
  realtimeFactor : Float := 1.0
  numTimeSamples : Nat := 21
  minimumTimeStep : Float := 0.05
  maximumTimeStep : Float := 0.2
  equalTimeIntervals : Bool := true
  torqueLimit : Float := 8.0
  initialState : Array Float := #[0.0, 0.0, 0.0, 0.0]
  goalState : Array Float := #[pi, 0.0, 0.0, 0.0]
  runningCostR : Float := 10.0
  initialGuessTimespan : Float := 4.0
  finiteHorizonQ : Array Float := #[10.0, 10.0, 1.0, 1.0]
  finiteHorizonR : Array Float := #[1.0]
  terminalTolerance : Float := 0.1
  visualizerEnabled : Bool := true
  deriving Repr, Inhabited

namespace SwingUpTrajectoryOptimizationBoundary

private def finiteArray (xs : Array Float) : Bool :=
  xs.all Float.isFinite

def validate? (boundary : SwingUpTrajectoryOptimizationBoundary) :
    Except String Unit := do
  if boundary.testPath != "../drake/examples/acrobot/test/run_swing_up_traj_optimization.cc" then
    .error s!"Acrobot swing-up trajectory optimization test path mismatch: {boundary.testPath}"
  if !boundary.requiresSnopt then
    .error "Acrobot swing-up trajectory optimization test requires SNOPT when available"
  if boundary.snoptUnavailableReturnCode != 0 then
    .error s!"Acrobot swing-up test should return 0 when SNOPT is unavailable, got {boundary.snoptUnavailableReturnCode}"
  if boundary.solverFailureReturnCode != 1 then
    .error s!"Acrobot swing-up test should return 1 on solve failure, got {boundary.solverFailureReturnCode}"
  if !boundary.realtimeFactor.isFinite || boundary.realtimeFactor < 0.0 then
    .error s!"Acrobot swing-up realtime factor must be nonnegative and finite, got {boundary.realtimeFactor}"
  if boundary.numTimeSamples != 21 then
    .error s!"Acrobot direct collocation sample count should be 21, got {boundary.numTimeSamples}"
  if !boundary.minimumTimeStep.isFinite || boundary.minimumTimeStep <= 0.0 then
    .error s!"Acrobot direct collocation minimum timestep must be positive, got {boundary.minimumTimeStep}"
  if !boundary.maximumTimeStep.isFinite || boundary.maximumTimeStep < boundary.minimumTimeStep then
    .error s!"Acrobot direct collocation maximum timestep must exceed the minimum, got {boundary.maximumTimeStep}"
  if !boundary.equalTimeIntervals then
    .error "Acrobot direct collocation test adds equal time interval constraints"
  if !boundary.torqueLimit.isFinite || boundary.torqueLimit <= 0.0 then
    .error s!"Acrobot direct collocation torque limit must be positive and finite, got {boundary.torqueLimit}"
  if boundary.initialState.size != 4 || !finiteArray boundary.initialState then
    .error s!"Acrobot direct collocation x0 must have four finite entries, got {boundary.initialState}"
  if boundary.goalState != #[pi, 0.0, 0.0, 0.0] then
    .error s!"Acrobot direct collocation goal should be upright [pi, 0, 0, 0], got {boundary.goalState}"
  if !boundary.runningCostR.isFinite || boundary.runningCostR <= 0.0 then
    .error s!"Acrobot direct collocation running cost R must be positive and finite, got {boundary.runningCostR}"
  if !boundary.initialGuessTimespan.isFinite || boundary.initialGuessTimespan <= 0.0 then
    .error s!"Acrobot initial trajectory timespan must be positive and finite, got {boundary.initialGuessTimespan}"
  if boundary.finiteHorizonQ != #[10.0, 10.0, 1.0, 1.0] then
    .error s!"Acrobot finite-horizon LQR Q mismatch: {boundary.finiteHorizonQ}"
  if boundary.finiteHorizonR != #[1.0] then
    .error s!"Acrobot finite-horizon LQR R mismatch: {boundary.finiteHorizonR}"
  if !boundary.terminalTolerance.isFinite || boundary.terminalTolerance <= 0.0 then
    .error s!"Acrobot terminal tolerance must be positive and finite, got {boundary.terminalTolerance}"
  if !boundary.visualizerEnabled then
    .error "Acrobot swing-up trajectory optimization test adds SceneGraph and DrakeVisualizer"

def graph (boundary : SwingUpTrajectoryOptimizationBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 5230, kind := .state .boundary, label := boundary.testPath }
    |>.addVertex { id := 5231, kind := .checkpoint, label := "SNOPT availability gate" }
    |>.addVertex { id := 5232, kind := .state .interior, label := "DirectCollocation acrobot program" }
    |>.addVertex { id := 5233, kind := .state .interior, label := "SNOPT solution result" }
    |>.addVertex { id := 5234, kind := .state .interior, label := "finite-horizon LQR trajectory stabilizer" }
    |>.addVertex { id := 5235, kind := .interval, label := "Simulator playback over optimized trajectory" }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[5231]
      reads := #[5230]
      writes := #[5231]
      label := "SnoptSolver::is_available && is_enabled gate"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5232]
      reads := #[5230, 5231]
      writes := #[5232]
      label := s!"DirectCollocation N={boundary.numTimeSamples}, dt in [{boundary.minimumTimeStep}, {boundary.maximumTimeStep}], |u| <= {boundary.torqueLimit}"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5233]
      reads := #[5232]
      writes := #[5233]
      label := "SNOPT solve for swing-up direct-collocation program"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[5234]
      reads := #[5233]
      writes := #[5234]
      label := "MakeFiniteHorizonLinearQuadraticRegulator along optimized trajectory"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[5235]
      reads := #[5234]
      writes := #[5235]
      cost := { work := boundary.initialGuessTimespan }
      label := "Simulator.AdvanceTo trajectory end; terminal state within 0.1 of upright"
    }

end SwingUpTrajectoryOptimizationBoundary

def swingUpTrajectoryOptimizationBoundary : SwingUpTrajectoryOptimizationBoundary := {}

structure AcrobotRegressionBoundaryResult where
  spongParams : IndexedCoordinateBoundary
  libraryPy : SpongSimLibraryPyTestBoundary
  mainPy : SpongSimMainTestBoundary
  mainCc : SpongSimMainTestBoundary
  swingUpTrajectoryOptimization : SwingUpTrajectoryOptimizationBoundary
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

private def spongRegressionMove (vertex : VertexId) (label : String)
    (kind : SkeletonMoveKind := .localSchurBlock)
    (exactness : MoveExactness := kind.defaultExactness) : SkeletonMove :=
  {
    kind := kind
    targets := #[vertex]
    exactness := exactness
    label := label
  }

def buildAcrobotRegressionBoundaries? :
    Except String AcrobotRegressionBoundaryResult := do
  spongControllerParamsIndicesBoundary.validate?
    spongParameterCoordinateNames "SpongControllerParamsIndices"
  spongSimLibraryPyTestBoundary.validate?
  spongSimMainPyTestBoundary.validate?
  spongSimMainCcTestBoundary.validate?
  swingUpTrajectoryOptimizationBoundary.validate?
  let swingGraph := swingUpTrajectoryOptimizationBoundary.graph
  let moves :=
    #[
      spongRegressionMove 5226
        "SpongControllerParamsIndices::GetCoordinateNames local named-vector block",
      spongRegressionMove 5227
        "Python spong_sim.simulate library call; expect 4x601 state tape"
        .intervalAdjoint,
      spongRegressionMove 5228
        "spong_sim_main_py subprocess help and deterministic scenario output"
        .intervalAdjoint,
      spongRegressionMove 5229
        "spong_sim_main_cc subprocess deterministic plus stochastic scenario dump"
        .intervalAdjoint
    ] ++ swingGraph.moves
  pure {
    spongParams := spongControllerParamsIndicesBoundary
    libraryPy := spongSimLibraryPyTestBoundary
    mainPy := spongSimMainPyTestBoundary
    mainCc := spongSimMainCcTestBoundary
    swingUpTrajectoryOptimization := swingUpTrajectoryOptimizationBoundary
    graph := { vertices := swingGraph.vertices, moves := moves }
    moves := moves
  }

structure AcrobotExternalBoundaryResult where
  references : Array DrakeReference
  channels : AcrobotLcmChannels
  lcmIo : AcrobotLcmIoSample
  lcmPlant : AcrobotLcmPlantBoundary
  lcmController : AcrobotLcmControllerBoundary
  estimatorLqr : AcrobotEstimatorLqrBoundary
  deterministicScenario : AcrobotSpongScenario
  stochasticScenario : AcrobotSpongScenario
  optimizer : AcrobotOptimizerDemoConfig
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

private def sampledScenarioMove (vertex : VertexId) (label : String) :
    SkeletonMove :=
  {
    kind := .markScoreSample
    targets := #[vertex]
    exactness := .unbiasedEstimator
    cost := { variance := 1.0 }
    label := label
  }

private def optimizerBranchMove (vertex : VertexId)
    (cfg : AcrobotOptimizerDemoConfig) : SkeletonMove :=
  {
    kind := .branchAggregate
    targets := #[vertex]
    exactness := .exact
    cost := { work := cfg.totalRollouts.toFloat }
    label :=
      s!"acrobot optimizer ensemble branch aggregate: {cfg.ensembleSize} seeds x {cfg.numEvaluations} evaluations"
  }

def acrobotExternalBoundaryMoves
    (io : AcrobotLcmIoSample)
    (plant : AcrobotLcmPlantBoundary := acrobotLcmPlantBoundary)
    (controller : AcrobotLcmControllerBoundary := acrobotLcmControllerBoundary)
    (estimator : AcrobotEstimatorLqrBoundary := acrobotEstimatorLqrBoundary)
    (optimizer : AcrobotOptimizerDemoConfig := acrobotOptimizerDemoConfig) :
    Array SkeletonMove :=
  #[
    io.move,
    controllerMove 5220
      s!"LCM plant boundary {plant.channels.commandChannel}->{plant.channels.stateEstimateChannel}",
    controllerMove 5221
      s!"LCM Spong controller boundary {controller.channels.stateEstimateChannel}->{controller.channels.commandChannel}",
    controllerMove 5222
      s!"rotary encoder + steady-state Kalman observer LQR loop, max step {estimator.maximumStepSize}",
    controllerMove 5223 "spong_sim C++/Python scenario runner and state tape output",
    sampledScenarioMove 5224
      "sample UniformVector controller_params and initial_state for stochastic Spong scenario",
    optimizerBranchMove 5225 optimizer
  ]

def buildAcrobotExternalBoundaries?
    (plant : AcrobotLcmPlantBoundary := acrobotLcmPlantBoundary)
    (controller : AcrobotLcmControllerBoundary := acrobotLcmControllerBoundary)
    (estimator : AcrobotEstimatorLqrBoundary := acrobotEstimatorLqrBoundary)
    (deterministicScenario : AcrobotSpongScenario := exampleSpongScenario)
    (stochasticScenario : AcrobotSpongScenario := exampleStochasticSpongScenario)
    (optimizer : AcrobotOptimizerDemoConfig := acrobotOptimizerDemoConfig) :
    Except String AcrobotExternalBoundaryResult := do
  plant.validate?
  controller.validate?
  estimator.validate?
  let io ← buildAcrobotLcmIoSample?
  deterministicScenario.validate?
  stochasticScenario.validate?
  if !stochasticScenario.isStochastic then
    .error "acrobot optimizer scenario should retain stochastic sampled fields"
  optimizer.validate?
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedSegment 0.0 estimator.simulationTime))
      |>.push (.interval (acceptedSegment 0.0 deterministicScenario.tFinal))
      |>.push (.interval (acceptedSegment 0.0 stochasticScenario.tFinal))
  trace.validate?
  pure {
    references := drakeReferences
    channels := plant.channels
    lcmIo := io
    lcmPlant := plant
    lcmController := controller
    estimatorLqr := estimator
    deterministicScenario := deterministicScenario
    stochasticScenario := stochasticScenario
    optimizer := optimizer
    trace := trace
    moves := acrobotExternalBoundaryMoves io plant controller estimator optimizer ++ trace.moves
  }

def solvePassive? (p : AcrobotParams := params)
    (x0 : AcrobotState := runPassiveInitialState)
    (t0 : Float := 0.0)
    (t1 : Float := 0.1)
    (u : AcrobotInput := defaultInput) :
    Except String SimulationResult := do
  if !p.isFiniteForDynamics then
    .error "acrobot params are not finite"
  if !u.isValid then
    .error "acrobot input is invalid"
  if !x0.isValid then
    .error "acrobot initial state is invalid"
  let sol :=
    diffeqsolve
      (Term := ODETerm AcrobotState AcrobotInput)
      (Y := AcrobotState)
      (VF := AcrobotState)
      (Control := Time)
      (Args := AcrobotInput)
      (Controller := ConstantStepSize)
      (odeTerm p) acrobotSolver t0 t1 (some p.stepSize) x0 u
      (saveat := { t1 := true })
  if !sol.result.isOkay then
    .error s!"acrobot solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "acrobot solve did not save endpoint"
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
    | _, _ => .error "acrobot solve did not save endpoint arrays"

private def rolloutBalancingLqr? (p : AcrobotParams)
    (cfg : ControllerSimulationConfig)
    (x0 : AcrobotState)
    (lqr : BalancingLqrData) :
    Except String (Array AcrobotState × SpongModeSummary) := do
  let mut x := x0
  let mut samples := #[x0]
  let mut balancingSteps := 0
  for _ in [:cfg.steps] do
    balancingSteps := balancingSteps + 1
    x := rk4ClosedLoopStep p (fun y => balancingLqrController lqr y) cfg.stepSize x
    samples := samples.push x
  pure (samples, SpongModeSummary.fromCounts balancingSteps 0)

private def rolloutSpong? (p : AcrobotParams)
    (controllerParams : SpongControllerParams)
    (cfg : ControllerSimulationConfig)
    (x0 : AcrobotState)
    (lqr : BalancingLqrData) :
    Except String (Array AcrobotState × SpongModeSummary) := do
  let mut x := x0
  let mut samples := #[x0]
  let mut balancingSteps := 0
  let mut swingUpSteps := 0
  for _ in [:cfg.steps] do
    let control ← spongController? p controllerParams lqr x
    match control.mode with
    | .balancing => balancingSteps := balancingSteps + 1
    | .swingUp => swingUpSteps := swingUpSteps + 1
    x := rk4ClosedLoopStep p (fun y => spongController p controllerParams lqr y) cfg.stepSize x
    samples := samples.push x
  pure (samples, SpongModeSummary.fromCounts balancingSteps swingUpSteps)

private def makeControllerResult?
    (controllerName : String)
    (cfg : ControllerSimulationConfig)
    (x0 : AcrobotState)
    (samples : Array AcrobotState)
    (modeSummary : SpongModeSummary)
    (p : AcrobotParams)
    (controllerBlock : SkeletonMove) :
    Except String ControllerSimulationResult := do
  if samples.size == 0 then
    .error s!"{controllerName} rollout produced no samples"
  let final := samples[samples.size - 1]!
  let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment 0.0 cfg.duration))
  trace.validate?
  pure {
    references := drakeReferences
    controllerName := controllerName
    t0 := 0.0
    t1 := cfg.duration
    stepSize := cfg.stepSize
    initialState := x0
    finalState := final
    samples := samples
    initialEnergy := totalEnergy p x0
    finalEnergy := totalEnergy p final
    modeSummary := modeSummary
    trace := trace
    moves := #[controllerBlock] ++ trace.moves
  }

def simulateLqr? (p : AcrobotParams := params)
    (cfg : ControllerSimulationConfig := controllerSimulationConfig)
    (x0 : AcrobotState := runLqrInitialState)
    (lqr : BalancingLqrData := balancingLqrData) :
    Except String ControllerSimulationResult := do
  if !p.isFiniteForDynamics then
    .error "acrobot params are not finite"
  if !x0.isValid then
    .error "acrobot LQR initial state is invalid"
  cfg.validate? "acrobot LQR"
  let (samples, modeSummary) ← rolloutBalancingLqr? p cfg x0 lqr
  makeControllerResult?
    "acrobot-balancing-lqr"
    cfg
    x0
    samples
    modeSummary
    p
    (controllerMove 5200 "acrobot balancing LQR local solver/controller")

def simulateSwingUp? (p : AcrobotParams := params)
    (controllerParams : SpongControllerParams := spongControllerParams)
    (cfg : ControllerSimulationConfig := controllerSimulationConfig)
    (x0 : AcrobotState := runSwingUpInitialState)
    (lqr : BalancingLqrData := balancingLqrData) :
    Except String ControllerSimulationResult := do
  if !p.isFiniteForDynamics then
    .error "acrobot params are not finite"
  if !x0.isValid then
    .error "acrobot Spong initial state is invalid"
  if !controllerParams.isValid then
    .error "acrobot Spong controller params are invalid"
  cfg.validate? "acrobot Spong"
  let (samples, modeSummary) ← rolloutSpong? p controllerParams cfg x0 lqr
  makeControllerResult?
    "acrobot-spong-swing-up"
    cfg
    x0
    samples
    modeSummary
    p
    (controllerMove 5201 "acrobot Spong dynamic mode controller"
      modeSummary.support.exactness)

def buildEndToEnd? : Except String SimulationResult :=
  solvePassive?

end Tyr.EventSkeleton.Examples.Acrobot
