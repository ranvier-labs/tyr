import Tyr.EventSkeleton.Manipulator

/-!
# Drake Kinova Jaco Arm Controller Example

This ports the system boundary used by `../drake/examples/kinova_jaco_arm`.
The example is controller- and message-facing: Drake loads a 7-DOF arm with
three finger joints, simulates it with an inverse-dynamics controller, and
uses LCM command/status systems around a PID velocity controller.

The reusable primitive is `JointPidInput.evaluate?`.  The Kinova-specific layer
records the arm/finger split, the finger SDK/URDF scale conversion, and the
clocked command/status boundary.  A full URDF-backed multibody provider can
replace the fixture state without changing the controller or event-skeleton
move vocabulary.
-/

namespace Tyr.EventSkeleton.Examples.KinovaJacoArm

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/kinova_jaco_arm/BUILD.bazel"
      concept := "declares jaco_controller, jaco_simulation, and move_jaco_ee binaries with Drake model data and smoke-test arguments"
    },
    {
      path := "../drake/examples/kinova_jaco_arm/jaco_controller.cc"
      concept := "connects status, robot-plan interpolation, PID feedback, desired-velocity feedforward, and command publishing"
    },
    {
      path := "../drake/examples/kinova_jaco_arm/jaco_simulation.cc"
      concept := "loads the Jaco URDF, welds the base, and uses an inverse-dynamics controller in simulation"
    },
    {
      path := "../drake/examples/kinova_jaco_arm/move_jaco_ee.cc"
      concept := "builds a committed robot plan from end-effector pose and current status"
    },
    {
      path := "../drake/examples/kinova_jaco_arm/README.md"
      concept := "documents the example process topology and LCM channels"
    },
    {
      path := "../drake/manipulation/kinova_jaco/jaco_constants.h"
      concept := "defines default joint counts, finger conversion factors, and status period"
    },
    {
      path := "../drake/manipulation/kinova_jaco/jaco_command_sender.cc"
      concept := "splits arm/finger commands and converts finger positions and velocities from URDF to SDK units"
    },
    {
      path := "../drake/manipulation/kinova_jaco/jaco_command_receiver.cc"
      concept := "latches command messages and converts finger positions and velocities from SDK to URDF units"
    },
    {
      path := "../drake/manipulation/kinova_jaco/jaco_status_sender.cc"
      concept := "publishes measured status, including Drake's arm-velocity divide-by-two behavior"
    },
    {
      path := "../drake/manipulation/kinova_jaco/jaco_status_receiver.cc"
      concept := "receives measured status and converts finger positions and velocities from SDK to URDF units"
    },
    {
      path := "package://drake_models/jaco_description/urdf/j2s7s300_sphere_collision.urdf"
      concept := "the Jaco model URL used by Drake's example binaries"
    }
  ]

inductive KinovaBuildTargetKind where
  | ccBinary
  deriving Repr, BEq, Inhabited

structure KinovaBuildTarget where
  kind : KinovaBuildTargetKind := .ccBinary
  name : String
  srcs : Array String := #[]
  data : Array String := #["@drake_models//:jaco_description"]
  deps : Array String := #[]
  addTestRule : Bool := false
  testRuleArgs : Array String := #[]
  deriving Repr, BEq, Inhabited

namespace KinovaBuildTarget

def hasDep (target : KinovaBuildTarget) (dep : String) : Bool :=
  target.deps.any (fun actual => actual == dep)

def hasData (target : KinovaBuildTarget) (datum : String) : Bool :=
  target.data.any (fun actual => actual == datum)

def hasTestRuleArg (target : KinovaBuildTarget) (arg : String) : Bool :=
  target.testRuleArgs.any (fun actual => actual == arg)

def validate? (target : KinovaBuildTarget) : Except String Unit := do
  if target.kind != .ccBinary then
    .error s!"Kinova target {target.name} should be a drake_cc_binary"
  if target.name.isEmpty then
    .error "Kinova BUILD target name cannot be empty"
  if target.srcs != #[s!"{target.name}.cc"] then
    .error s!"Kinova target {target.name} should compile {target.name}.cc, got {target.srcs}"
  if !target.hasData "@drake_models//:jaco_description" then
    .error s!"Kinova target {target.name} should include @drake_models//:jaco_description runfile data"
  match target.name with
  | "jaco_controller" =>
      if !target.addTestRule || !target.hasTestRuleArg "--build_only" then
        .error "jaco_controller should keep the BUILD smoke test --build_only"
      if !target.hasDep "//manipulation/util:robot_plan_interpolator" then
        .error "jaco_controller should depend on RobotPlanInterpolator"
      if !target.hasDep "//systems/controllers:pid_controller" then
        .error "jaco_controller should depend on PidController"
      if !target.hasDep "//systems/lcm:lcm_pubsub_system" then
        .error "jaco_controller should depend on LCM pub/sub systems"
      if !target.hasDep "//systems/primitives:adder" then
        .error "jaco_controller should expose the desired-velocity adder"
  | "jaco_simulation" =>
      if !target.addTestRule || !target.hasTestRuleArg "--simulation_sec=0.1" then
        .error "jaco_simulation should keep the BUILD smoke test --simulation_sec=0.1"
      if !target.hasDep "//multibody/parsing" then
        .error "jaco_simulation should depend on multibody parsing"
      if !target.hasDep "//systems/controllers:inverse_dynamics_controller" then
        .error "jaco_simulation should depend on InverseDynamicsController"
      if !target.hasDep "//visualization:visualization_config_functions" then
        .error "jaco_simulation should depend on default visualization setup"
  | "move_jaco_ee" =>
      if target.addTestRule then
        .error "move_jaco_ee BUILD target should not add a smoke test"
      if !target.hasDep "//manipulation/util:move_ik_demo_base" then
        .error "move_jaco_ee should depend on MoveIkDemoBase"
      if !target.hasDep "//lcm:drake_lcm" then
        .error "move_jaco_ee should depend on DrakeLcm"
      if !target.hasDep "//math:geometric_transform" then
        .error "move_jaco_ee should depend on geometric transform math"
  | other => .error s!"Unexpected Kinova BUILD target {other}"

end KinovaBuildTarget

def buildTargets : Array KinovaBuildTarget :=
  #[
    {
      name := "jaco_controller"
      srcs := #["jaco_controller.cc"]
      deps := #[
        "//common:add_text_logging_gflags",
        "//lcm",
        "//manipulation/kinova_jaco",
        "//manipulation/util:robot_plan_interpolator",
        "//systems/analysis:simulator",
        "//systems/controllers:pid_controller",
        "//systems/lcm:lcm_pubsub_system",
        "//systems/primitives:adder",
        "//systems/primitives:demultiplexer",
        "//systems/primitives:multiplexer",
        "@gflags"
      ]
      addTestRule := true
      testRuleArgs := #["--build_only"]
    },
    {
      name := "jaco_simulation"
      srcs := #["jaco_simulation.cc"]
      deps := #[
        "//common:add_text_logging_gflags",
        "//geometry:drake_visualizer",
        "//geometry:scene_graph",
        "//manipulation/kinova_jaco",
        "//multibody/parsing",
        "//multibody/plant",
        "//systems/analysis:simulator",
        "//systems/controllers:inverse_dynamics_controller",
        "//systems/primitives:demultiplexer",
        "//systems/primitives:multiplexer",
        "//visualization:visualization_config_functions",
        "@gflags"
      ]
      addTestRule := true
      testRuleArgs := #["--simulation_sec=0.1"]
    },
    {
      name := "move_jaco_ee"
      srcs := #["move_jaco_ee.cc"]
      deps := #[
        "//common:add_text_logging_gflags",
        "//lcm:drake_lcm",
        "//lcmtypes:lcmtypes_drake_cc",
        "//manipulation/kinova_jaco:jaco_constants",
        "//manipulation/util:move_ik_demo_base",
        "//math:geometric_transform"
      ]
    }
  ]

def validateBuildTargets? (targets : Array KinovaBuildTarget := buildTargets) :
    Except String Unit := do
  if targets.size != 3 then
    .error s!"Kinova BUILD.bazel should declare three binaries, got {targets.size}"
  if !(targets.any (fun target => target.name == "jaco_controller")) then
    .error "missing jaco_controller BUILD target"
  if !(targets.any (fun target => target.name == "jaco_simulation")) then
    .error "missing jaco_simulation BUILD target"
  if !(targets.any (fun target => target.name == "move_jaco_ee")) then
    .error "missing move_jaco_ee BUILD target"
  for target in targets do
    target.validate?

def defaultArmJoints : Nat := 7
def defaultFingers : Nat := 3
def defaultDof : Nat := defaultArmJoints + defaultFingers

def fingerSdkToUrdf : Float := 1.34 / 118.68
def fingerUrdfToSdk : Float := 1.0 / fingerSdkToUrdf

def statusPeriod : Float := 0.010
def simulationTimeStep : Float := 3.0e-3

def statusChannel : String := "KINOVA_JACO_STATUS"
def commandChannel : String := "KINOVA_JACO_COMMAND"
def planChannel : String := "COMMITTED_ROBOT_PLAN"

def modelUri : String :=
  "package://drake_models/jaco_description/urdf/j2s7s300_sphere_collision.urdf"

structure JacoControllerExecutableBoundary where
  executableName : String := "jaco_controller"
  buildOnlySmokeArg : String := "--build_only"
  urdfFlagDefault : String := ""
  numJointsFlagDefault : Nat := defaultArmJoints
  numFingersFlagDefault : Nat := defaultFingers
  statusChannel : String := KinovaJacoArm.statusChannel
  commandChannel : String := KinovaJacoArm.commandChannel
  planChannel : String := KinovaJacoArm.planChannel
  usesRobotPlanInterpolator : Bool := true
  waitsForFirstStatus : Bool := true
  forcesCommandPublish : Bool := true
  runsUntilInterrupted : Bool := true
  deriving Repr, BEq, Inhabited

namespace JacoControllerExecutableBoundary

def validate? (boundary : JacoControllerExecutableBoundary) : Except String Unit := do
  if boundary.executableName != "jaco_controller" then
    .error s!"Jaco controller executable mismatch: {boundary.executableName}"
  if boundary.buildOnlySmokeArg != "--build_only" then
    .error s!"Jaco controller smoke arg should be --build_only, got {boundary.buildOnlySmokeArg}"
  if boundary.numJointsFlagDefault != defaultArmJoints ||
      boundary.numFingersFlagDefault != defaultFingers then
    .error s!"Jaco controller default joint/finger flags should be {defaultArmJoints}/{defaultFingers}, got {boundary.numJointsFlagDefault}/{boundary.numFingersFlagDefault}"
  if boundary.statusChannel != KinovaJacoArm.statusChannel ||
      boundary.commandChannel != KinovaJacoArm.commandChannel ||
      boundary.planChannel != KinovaJacoArm.planChannel then
    .error "Jaco controller LCM channel defaults should match Drake constants"
  if !boundary.usesRobotPlanInterpolator || !boundary.waitsForFirstStatus ||
      !boundary.forcesCommandPublish || !boundary.runsUntilInterrupted then
    .error "Jaco controller should preserve plan interpolation, first-status wait, forced command publish, and run-forever boundaries"

end JacoControllerExecutableBoundary

def controllerExecutableBoundary : JacoControllerExecutableBoundary := {}

structure JacoSimulationExecutableBoundary where
  executableName : String := "jaco_simulation"
  simulationSecDefault : String := "infinity"
  smokeTestArg : String := "--simulation_sec=0.1"
  realtimeRateDefault : Float := 1.0
  timeStepDefault : Float := simulationTimeStep
  commandChannel : String := KinovaJacoArm.commandChannel
  statusChannel : String := KinovaJacoArm.statusChannel
  weldsBaseFrameToWorld : Bool := true
  usesInverseDynamicsController : Bool := true
  addsDefaultVisualization : Bool := true
  publishesStatusAtPeriod : Float := statusPeriod
  deriving Repr, BEq, Inhabited

namespace JacoSimulationExecutableBoundary

def validate? (boundary : JacoSimulationExecutableBoundary) : Except String Unit := do
  if boundary.executableName != "jaco_simulation" then
    .error s!"Jaco simulation executable mismatch: {boundary.executableName}"
  if boundary.simulationSecDefault != "infinity" then
    .error s!"Jaco simulation default simulation_sec should be infinity, got {boundary.simulationSecDefault}"
  if boundary.smokeTestArg != "--simulation_sec=0.1" then
    .error s!"Jaco simulation smoke arg should be --simulation_sec=0.1, got {boundary.smokeTestArg}"
  if !boundary.realtimeRateDefault.isFinite ||
      Float.abs (boundary.realtimeRateDefault - 1.0) > 1.0e-12 then
    .error s!"Jaco simulation realtime_rate default should be 1, got {boundary.realtimeRateDefault}"
  if !boundary.timeStepDefault.isFinite ||
      Float.abs (boundary.timeStepDefault - simulationTimeStep) > 1.0e-12 then
    .error s!"Jaco simulation time_step default should be {simulationTimeStep}, got {boundary.timeStepDefault}"
  if boundary.commandChannel != KinovaJacoArm.commandChannel ||
      boundary.statusChannel != KinovaJacoArm.statusChannel then
    .error "Jaco simulation LCM channels should match Drake constants"
  if !boundary.weldsBaseFrameToWorld || !boundary.usesInverseDynamicsController ||
      !boundary.addsDefaultVisualization then
    .error "Jaco simulation should weld base, use inverse dynamics, and add visualization"
  if Float.abs (boundary.publishesStatusAtPeriod - statusPeriod) > 1.0e-12 then
    .error s!"Jaco simulation should publish status every {statusPeriod}, got {boundary.publishesStatusAtPeriod}"

end JacoSimulationExecutableBoundary

def simulationExecutableBoundary : JacoSimulationExecutableBoundary := {}

def armJointNames : Array String :=
  #["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3",
    "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6",
    "j2s7s300_joint_7"]

def fingerJointNames : Array String :=
  #["j2s7s300_joint_finger_1", "j2s7s300_joint_finger_2",
    "j2s7s300_joint_finger_3"]

def jointNames : Array String :=
  armJointNames ++ fingerJointNames

def positionCoordinateNames : Array String :=
  (Array.range defaultDof).map (fun i => s!"q{i}")

def velocityCoordinateNames : Array String :=
  (Array.range defaultDof).map (fun i => s!"v{i}")

def stateCoordinateNames : Array String :=
  positionCoordinateNames ++ velocityCoordinateNames

def zeros : Array Float :=
  Array.replicate defaultDof 0.0

def controllerGains : JointPidGains :=
  {
    kp := Array.replicate defaultDof 1.0
    kd := Array.replicate defaultDof 0.0
    ki := Array.replicate defaultDof 0.0
    label := "jaco_controller PID"
  }

def simulationGains : JointPidGains :=
  {
    kp := Array.replicate defaultDof 100.0
    kd := Array.replicate defaultDof 20.0
    ki := Array.replicate defaultDof 0.0
    label := "jaco_simulation inverse-dynamics PID"
  }

structure JacoState where
  q : Array Float
  v : Array Float
  deriving Repr, Inhabited

namespace JacoState

def validate? (x : JacoState) (label : String := "jaco state") :
    Except String Unit := do
  if x.q.size != defaultDof then
    .error s!"{label}: q size {x.q.size} != {defaultDof}"
  if x.v.size != defaultDof then
    .error s!"{label}: v size {x.v.size} != {defaultDof}"
  for i in [:defaultDof] do
    if !(x.q[i]!).isFinite then
      .error s!"{label}: q[{i}] must be finite, got {x.q[i]!}"
    if !(x.v[i]!).isFinite then
      .error s!"{label}: v[{i}] must be finite, got {x.v[i]!}"

def asArray (x : JacoState) : Array Float :=
  x.q ++ x.v

end JacoState

def initialSimulationState : JacoState :=
  {
    q := #[1.80, 3.44, 3.14, 0.76, 4.63, 4.49, 5.03, 0.0, 0.0, 0.0]
    v := zeros
  }

def sampleEstimatedState : JacoState :=
  {
    q := #[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.01, 0.02, 0.03]
    v := #[0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.001, 0.002, 0.003]
  }

def sampleDesiredState : JacoState :=
  {
    q := #[0.20, 0.10, 0.35, 0.30, 0.65, 0.50, 0.80, 0.02, 0.01, 0.04]
    v := #[0.20, 0.10, 0.00, -0.10, 0.05, -0.05, 0.15, 0.004, 0.005, 0.006]
  }

structure EndEffectorPoseTarget where
  x : Float
  y : Float
  z : Float
  roll : Float
  pitch : Float
  yaw : Float
  endEffectorName : String
  deriving Repr, Inhabited

def defaultMoveTarget : EndEffectorPoseTarget :=
  {
    x := 0.3
    y := -0.26
    z := 0.5
    roll := -1.7
    pitch := -1.3
    yaw := -1.8
    endEffectorName := "j2s7s300_end_effector"
  }

structure MoveJacoEeExecutableBoundary where
  executableName : String := "move_jaco_ee"
  statusChannel : String := KinovaJacoArm.statusChannel
  planChannel : String := KinovaJacoArm.planChannel
  baseFrame : String := "base"
  endEffectorName : String := defaultMoveTarget.endEffectorName
  ikSampleCount : Nat := 100
  target : EndEffectorPoseTarget := defaultMoveTarget
  waitsForStatus : Bool := true
  convertsFingerStatusToUrdf : Bool := true
  publishesRobotPlan : Bool := true
  deriving Repr, Inhabited

namespace MoveJacoEeExecutableBoundary

def validate? (boundary : MoveJacoEeExecutableBoundary) : Except String Unit := do
  if boundary.executableName != "move_jaco_ee" then
    .error s!"Jaco move executable mismatch: {boundary.executableName}"
  if boundary.statusChannel != KinovaJacoArm.statusChannel ||
      boundary.planChannel != KinovaJacoArm.planChannel then
    .error "move_jaco_ee LCM channel defaults should match Drake flags"
  if boundary.baseFrame != "base" then
    .error s!"move_jaco_ee should build MoveIkDemoBase from the base frame, got {boundary.baseFrame}"
  if boundary.endEffectorName != defaultMoveTarget.endEffectorName then
    .error s!"move_jaco_ee end-effector mismatch: {boundary.endEffectorName}"
  if boundary.ikSampleCount != 100 then
    .error s!"move_jaco_ee should use 100 IK samples, got {boundary.ikSampleCount}"
  if !boundary.waitsForStatus || !boundary.convertsFingerStatusToUrdf ||
      !boundary.publishesRobotPlan then
    .error "move_jaco_ee should wait for status, convert finger status to URDF units, and publish a robot plan"
  if !boundary.target.x.isFinite || !boundary.target.y.isFinite ||
      !boundary.target.z.isFinite || !boundary.target.roll.isFinite ||
      !boundary.target.pitch.isFinite || !boundary.target.yaw.isFinite then
    .error s!"move_jaco_ee target must be finite, got {reprStr boundary.target}"

end MoveJacoEeExecutableBoundary

def moveEeExecutableBoundary : MoveJacoEeExecutableBoundary := {}

structure JacoCommandMessage where
  utime : Float
  jointPosition : Array Float
  jointVelocity : Array Float
  fingerPosition : Array Float
  fingerVelocity : Array Float
  deriving Repr, Inhabited

structure JacoStatusMessage where
  utime : Float
  jointPosition : Array Float
  jointVelocity : Array Float
  fingerPosition : Array Float
  fingerVelocity : Array Float
  jointTorque : Array Float := #[]
  fingerTorque : Array Float := #[]
  jointCurrent : Array Float := #[]
  fingerCurrent : Array Float := #[]
  deriving Repr, Inhabited

def armPart (xs : Array Float) : Array Float :=
  xs.extract 0 defaultArmJoints

def fingerPart (xs : Array Float) : Array Float :=
  xs.extract defaultArmJoints defaultDof

def splitStateVector? (xs : Array Float) (label : String) :
    Except String (Array Float × Array Float) := do
  if xs.size != defaultDof then
    .error s!"{label}: vector size {xs.size} != Jaco dof {defaultDof}"
  for i in [:xs.size] do
    if !(xs[i]!).isFinite then
      .error s!"{label}: vector[{i}] must be finite, got {xs[i]!}"
  pure (armPart xs, fingerPart xs)

def toCommandMessage? (timeSeconds : Float)
    (position velocity : Array Float) : Except String JacoCommandMessage := do
  let (qArm, qFinger) ← splitStateVector? position "Jaco command position"
  let (vArm, vFinger) ← splitStateVector? velocity "Jaco command velocity"
  if !timeSeconds.isFinite then
    .error s!"Jaco command time must be finite, got {timeSeconds}"
  pure {
    utime := timeSeconds * 1000000.0
    jointPosition := qArm
    jointVelocity := vArm
    fingerPosition := FloatArray.scale fingerUrdfToSdk qFinger
    fingerVelocity := FloatArray.scale fingerUrdfToSdk vFinger
  }

def commandReceiverPosition (msg : JacoCommandMessage) : Array Float :=
  msg.jointPosition ++ FloatArray.scale fingerSdkToUrdf msg.fingerPosition

def commandReceiverVelocity (msg : JacoCommandMessage) : Array Float :=
  msg.jointVelocity ++ FloatArray.scale fingerSdkToUrdf msg.fingerVelocity

def toStatusMessage? (timeSeconds : Float) (state : JacoState) :
    Except String JacoStatusMessage := do
  state.validate? "Jaco status state"
  if !timeSeconds.isFinite then
    .error s!"Jaco status time must be finite, got {timeSeconds}"
  let armVelocity := (armPart state.v).map (fun v => v / 2.0)
  pure {
    utime := timeSeconds * 1000000.0
    jointPosition := armPart state.q
    jointVelocity := armVelocity
    fingerPosition := FloatArray.scale fingerUrdfToSdk (fingerPart state.q)
    fingerVelocity := FloatArray.scale fingerUrdfToSdk (fingerPart state.v)
    jointTorque := Array.replicate defaultArmJoints 0.0
    fingerTorque := Array.replicate defaultFingers 0.0
    jointCurrent := Array.replicate defaultArmJoints 0.0
    fingerCurrent := Array.replicate defaultFingers 0.0
  }

def statusReceiverMeasuredPosition (msg : JacoStatusMessage) : Array Float :=
  msg.jointPosition ++ FloatArray.scale fingerSdkToUrdf msg.fingerPosition

def statusReceiverMeasuredVelocity (msg : JacoStatusMessage) : Array Float :=
  msg.jointVelocity ++ FloatArray.scale fingerSdkToUrdf msg.fingerVelocity

def controllerInput
    (estimated : JacoState := sampleEstimatedState)
    (desired : JacoState := sampleDesiredState) : JointPidInput :=
  {
    estimatedState := estimated.asArray
    desiredState := desired.asArray
    integralError := zeros
    label := "jaco controller PID input"
  }

def evaluateController?
    (estimated : JacoState := sampleEstimatedState)
    (desired : JacoState := sampleDesiredState) :
    Except String JointPidOutput := do
  estimated.validate? "estimated Jaco state"
  desired.validate? "desired Jaco plan state"
  (controllerInput estimated desired).evaluate? controllerGains

def velocityCommand (output : JointPidOutput) : Array Float :=
  FloatArray.add output.feedback output.desiredVelocity

def commandFromController? (timeSeconds : Float)
    (estimated : JacoState := sampleEstimatedState)
    (desired : JacoState := sampleDesiredState) :
    Except String (JointPidOutput × Array Float × JacoCommandMessage) := do
  let output ← evaluateController? estimated desired
  let commandVelocity := velocityCommand output
  let message ← toCommandMessage? timeSeconds output.desiredPosition commandVelocity
  pure (output, commandVelocity, message)

def controllerGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8390, kind := .state .boundary, label := "kinova_jaco_arm BUILD.bazel" }
    |>.addVertex { id := 8391, kind := .state .boundary, label := "@drake_models//:jaco_description runfile data" }
    |>.addVertex { id := 8392, kind := .state .boundary, label := "jaco_simulation executable" }
    |>.addVertex { id := 8393, kind := .state .boundary, label := "move_jaco_ee executable" }
    |>.addVertex { id := 8394, kind := .state .boundary, label := "jaco_controller executable" }
    |>.addVertex { id := 8400, kind := .state .boundary, label := "KINOVA_JACO_STATUS" }
    |>.addVertex { id := 8401, kind := .state .boundary, label := "COMMITTED_ROBOT_PLAN" }
    |>.addVertex { id := 8402, kind := .state .interior, label := "plan_interpolator_desired_state" }
    |>.addVertex { id := 8403, kind := .state .interior, label := "pid_feedback" }
    |>.addVertex { id := 8404, kind := .state .interior, label := "desired_velocity_feedforward" }
    |>.addVertex { id := 8405, kind := .state .boundary, label := "KINOVA_JACO_COMMAND" }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[8390]
      reads := #[8390]
      writes := #[8391]
      cost := { work := 1.0, memory := 1.0 }
      label := "BUILD.bazel resolves @drake_models//:jaco_description runfile data"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8394]
      reads := #[8390, 8391]
      writes := #[8400, 8401, 8405]
      cost := { work := 1.0, memory := 1.0 }
      label := "jaco_controller --build_only diagram boundary with LCM plan/status, RobotPlanInterpolator, PID, and command publisher"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8392]
      reads := #[8391, 8405]
      writes := #[8400]
      cost := { work := 1.0, memory := 1.0 }
      label := "jaco_simulation Parser.AddModelsFromUrl, WeldFrames(base), InverseDynamicsController, AddDefaultVisualization"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8393]
      reads := #[8400]
      writes := #[8401]
      cost := { work := 1.0, memory := 1.0 }
      label := "move_jaco_ee MoveIkDemoBase base->j2s7s300_end_effector publishes COMMITTED_ROBOT_PLAN"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[8400, 8401]
      reads := #[8400, 8401]
      writes := #[8402]
      cost := { work := 1.0, memory := 1.0 }
      label := "jaco-plan-interpolator-at-status-period"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8402, 8403, 8404]
      reads := #[8400, 8402]
      writes := #[8403, 8404, 8405]
      cost := { work := 1.0, memory := 0.25 }
      label := "jaco-pid-feedback-plus-desired-velocity-feedforward"
    }

structure KinovaJacoResult where
  references : Array DrakeReference
  buildTargets : Array KinovaBuildTarget
  controllerExecutable : JacoControllerExecutableBoundary
  simulationExecutable : JacoSimulationExecutableBoundary
  moveEeExecutable : MoveJacoEeExecutableBoundary
  controllerOutput : JointPidOutput
  commandVelocity : Array Float
  commandMessage : JacoCommandMessage
  statusMessage : JacoStatusMessage
  statusRoundTripPosition : Array Float
  statusRoundTripVelocity : Array Float
  moveTarget : EndEffectorPoseTarget
  graph : SkeletonGraph
  deriving Repr, Inhabited

def buildEndToEnd? : Except String KinovaJacoResult := do
  validateBuildTargets?
  controllerExecutableBoundary.validate?
  simulationExecutableBoundary.validate?
  moveEeExecutableBoundary.validate?
  let (output, commandVelocity, commandMessage) ←
    commandFromController? statusPeriod sampleEstimatedState sampleDesiredState
  let statusMessage ← toStatusMessage? statusPeriod sampleEstimatedState
  pure {
    references := drakeReferences
    buildTargets := buildTargets
    controllerExecutable := controllerExecutableBoundary
    simulationExecutable := simulationExecutableBoundary
    moveEeExecutable := moveEeExecutableBoundary
    controllerOutput := output
    commandVelocity := commandVelocity
    commandMessage := commandMessage
    statusMessage := statusMessage
    statusRoundTripPosition := statusReceiverMeasuredPosition statusMessage
    statusRoundTripVelocity := statusReceiverMeasuredVelocity statusMessage
    moveTarget := defaultMoveTarget
    graph := controllerGraph
  }

end Tyr.EventSkeleton.Examples.KinovaJacoArm
