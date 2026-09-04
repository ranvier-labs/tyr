import LeanTest
import Tyr.EventSkeleton.Examples.KinovaJacoArm

namespace Tests.EventSkeletonKinovaJacoArmExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.KinovaJacoArm

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) :
    IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

@[test]
def testDrakeReferencesConstantsAndLayoutAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kinova_jaco_arm/BUILD.bazel"))
    "Example should reference Drake's kinova_jaco_arm BUILD.bazel"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kinova_jaco_arm/jaco_controller.cc"))
    "Example should reference Drake's Jaco controller implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/kinova_jaco_arm/jaco_simulation.cc"))
    "Example should reference Drake's Jaco simulation implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/manipulation/kinova_jaco/jaco_status_sender.cc"))
    "Example should reference Drake's Jaco status sender"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path.contains "j2s7s300_sphere_collision.urdf"))
    "Example should record Drake's Jaco URDF model URL"

  LeanTest.assertEqual defaultArmJoints 7
  LeanTest.assertEqual defaultFingers 3
  LeanTest.assertEqual defaultDof 10
  LeanTest.assertEqual jointNames.size 10
  LeanTest.assertEqual stateCoordinateNames.size 20
  LeanTest.assertEqual statusChannel "KINOVA_JACO_STATUS"
  LeanTest.assertEqual commandChannel "KINOVA_JACO_COMMAND"
  LeanTest.assertEqual planChannel "COMMITTED_ROBOT_PLAN"
  LeanTest.assertTrue (approx statusPeriod 0.010 1.0e-12)
    s!"status period should be 10 ms, got {statusPeriod}"
  LeanTest.assertTrue (approx simulationTimeStep 0.003 1.0e-12)
    s!"simulation time step should be 3 ms, got {simulationTimeStep}"
  LeanTest.assertTrue (approx fingerSdkToUrdf (1.34 / 118.68) 1.0e-12)
    s!"finger SDK-to-URDF scale mismatch: {fingerSdkToUrdf}"

@[test]
def testBuildTargetsAndExecutableBoundariesMatchDrake : IO Unit := do
  let result ← assertOk buildEndToEnd? "Kinova Jaco end-to-end controller"
  let _ ← assertOk (validateBuildTargets? result.buildTargets)
    "Kinova BUILD target metadata"
  let _ ← assertOk result.controllerExecutable.validate?
    "Kinova controller executable boundary"
  let _ ← assertOk result.simulationExecutable.validate?
    "Kinova simulation executable boundary"
  let _ ← assertOk result.moveEeExecutable.validate?
    "Kinova move_jaco_ee executable boundary"

  LeanTest.assertEqual result.buildTargets.size 3
    "Drake BUILD.bazel should expose three Kinova binaries"
  let controller? := result.buildTargets.find? (fun target => target.name == "jaco_controller")
  let simulation? := result.buildTargets.find? (fun target => target.name == "jaco_simulation")
  let move? := result.buildTargets.find? (fun target => target.name == "move_jaco_ee")
  match controller?, simulation?, move? with
  | some controller, some simulation, some move =>
      LeanTest.assertTrue (controller.kind == KinovaBuildTargetKind.ccBinary)
        "jaco_controller should be a drake_cc_binary"
      LeanTest.assertTrue (controller.hasData "@drake_models//:jaco_description")
        "jaco_controller should include Jaco model runfiles"
      LeanTest.assertTrue (controller.hasDep "//manipulation/util:robot_plan_interpolator")
        "jaco_controller should depend on RobotPlanInterpolator"
      LeanTest.assertTrue (controller.hasDep "//systems/controllers:pid_controller")
        "jaco_controller should depend on PidController"
      LeanTest.assertTrue (controller.hasTestRuleArg "--build_only")
        "jaco_controller smoke test should pass --build_only"
      LeanTest.assertTrue (simulation.hasDep "//systems/controllers:inverse_dynamics_controller")
        "jaco_simulation should use InverseDynamicsController"
      LeanTest.assertTrue (simulation.hasDep "//visualization:visualization_config_functions")
        "jaco_simulation should add default visualization"
      LeanTest.assertTrue (simulation.hasTestRuleArg "--simulation_sec=0.1")
        "jaco_simulation smoke test should run for 0.1s"
      LeanTest.assertTrue (move.hasDep "//manipulation/util:move_ik_demo_base")
        "move_jaco_ee should depend on MoveIkDemoBase"
      LeanTest.assertTrue (move.hasDep "//math:geometric_transform")
        "move_jaco_ee should depend on geometric transform math"
      LeanTest.assertFalse move.addTestRule
        "move_jaco_ee should not have a BUILD smoke-test rule"
  | _, _, _ => LeanTest.fail "Expected controller, simulation, and move_jaco_ee BUILD targets"

  LeanTest.assertTrue result.controllerExecutable.usesRobotPlanInterpolator
    "jaco_controller should preserve RobotPlanInterpolator"
  LeanTest.assertTrue result.controllerExecutable.waitsForFirstStatus
    "jaco_controller should wait for an initial lcmt_jaco_status"
  LeanTest.assertTrue result.controllerExecutable.forcesCommandPublish
    "jaco_controller should force-publish lcmt_jaco_command"
  LeanTest.assertEqual result.simulationExecutable.simulationSecDefault "infinity"
    "jaco_simulation default simulation_sec should be infinity"
  LeanTest.assertTrue result.simulationExecutable.weldsBaseFrameToWorld
    "jaco_simulation should weld the Jaco base frame to world"
  LeanTest.assertTrue result.simulationExecutable.addsDefaultVisualization
    "jaco_simulation should call AddDefaultVisualization"
  LeanTest.assertEqual result.moveEeExecutable.baseFrame "base"
    "move_jaco_ee should plan from the base frame"
  LeanTest.assertEqual result.moveEeExecutable.ikSampleCount 100
    "move_jaco_ee should construct MoveIkDemoBase with 100 samples"
  LeanTest.assertTrue result.moveEeExecutable.publishesRobotPlan
    "move_jaco_ee should publish a COMMITTED_ROBOT_PLAN"

@[test]
def testJointPidVelocityCommandMatchesDrakeControllerFormula : IO Unit := do
  let output ← assertOk evaluateController? "Jaco PID controller"
  let expectedFeedback := FloatArray.sub sampleDesiredState.q sampleEstimatedState.q
  assertArrayNear output.positionError expectedFeedback 1.0e-12
    "With kp=1, kd=ki=0, PID feedback should equal q_desired - q"
  assertArrayNear output.feedback expectedFeedback 1.0e-12
    "Jaco controller feedback should match Drake's PID composition"
  let expectedCommandVelocity :=
    FloatArray.add sampleDesiredState.v expectedFeedback
  assertArrayNear (velocityCommand output) expectedCommandVelocity 1.0e-12
    "Jaco command velocity should add PID feedback to desired velocity feedforward"

@[test]
def testCommandAndStatusScalingMatchesDrakeJacoSystems : IO Unit := do
  let (output, commandVelocity, commandMessage) ←
    assertOk (commandFromController? 0.25 sampleEstimatedState sampleDesiredState)
      "Jaco command sender"
  LeanTest.assertTrue (approx commandMessage.utime 250000.0 1.0e-9)
    s!"Command sender should convert seconds to microseconds, got {commandMessage.utime}"
  LeanTest.assertEqual commandMessage.jointPosition.size defaultArmJoints
  LeanTest.assertEqual commandMessage.fingerPosition.size defaultFingers
  assertArrayNear commandMessage.jointPosition (armPart output.desiredPosition) 1.0e-12
    "Command sender should pass arm positions through unchanged"
  assertArrayNear commandMessage.fingerPosition
    (FloatArray.scale fingerUrdfToSdk (fingerPart output.desiredPosition))
    1.0e-12
    "Command sender should convert finger positions from URDF to SDK units"
  assertArrayNear (commandReceiverPosition commandMessage) output.desiredPosition 1.0e-12
    "Command receiver should invert command-sender position scaling"
  assertArrayNear (commandReceiverVelocity commandMessage) commandVelocity 1.0e-12
    "Command receiver should invert command-sender velocity scaling"

  let statusMessage ← assertOk (toStatusMessage? 0.25 sampleEstimatedState)
    "Jaco status sender"
  LeanTest.assertTrue (approx statusMessage.utime 250000.0 1.0e-9)
    s!"Status sender should convert seconds to microseconds, got {statusMessage.utime}"
  assertArrayNear statusMessage.jointPosition (armPart sampleEstimatedState.q) 1.0e-12
    "Status sender should pass arm positions through unchanged"
  assertArrayNear statusMessage.jointVelocity
    ((armPart sampleEstimatedState.v).map (fun v => v / 2.0))
    1.0e-12
    "Status sender should preserve Drake's arm-velocity divide-by-two convention"
  assertArrayNear statusMessage.fingerPosition
    (FloatArray.scale fingerUrdfToSdk (fingerPart sampleEstimatedState.q))
    1.0e-12
    "Status sender should convert finger positions from URDF to SDK units"
  assertArrayNear (statusReceiverMeasuredPosition statusMessage)
    sampleEstimatedState.q 1.0e-12
    "Status receiver should recover measured URDF positions"
  let expectedMeasuredVelocity :=
    ((armPart sampleEstimatedState.v).map (fun v => v / 2.0)) ++
      fingerPart sampleEstimatedState.v
  assertArrayNear (statusReceiverMeasuredVelocity statusMessage)
    expectedMeasuredVelocity 1.0e-12
    "Status receiver should keep half-scaled arm velocities and recover finger velocities"

@[test]
def testSimulationControllerGainsAndMoveTarget : IO Unit := do
  LeanTest.assertTrue
    (simulationGains.kp.all (fun kp => approx kp 100.0 1.0e-12))
    s!"Simulation Kp should be 100 for every Jaco coordinate, got {simulationGains.kp}"
  LeanTest.assertTrue
    (simulationGains.kd.all (fun kd => approx kd 20.0 1.0e-12))
    s!"Simulation Kd should be 2*sqrt(100)=20 for every coordinate, got {simulationGains.kd}"
  LeanTest.assertEqual initialSimulationState.q
    #[1.80, 3.44, 3.14, 0.76, 4.63, 4.49, 5.03, 0.0, 0.0, 0.0]
  LeanTest.assertTrue (approx defaultMoveTarget.x 0.3 1.0e-12)
    s!"move_jaco_ee x target mismatch: {defaultMoveTarget.x}"
  LeanTest.assertTrue (approx defaultMoveTarget.y (-0.26) 1.0e-12)
    s!"move_jaco_ee y target mismatch: {defaultMoveTarget.y}"
  LeanTest.assertTrue (approx defaultMoveTarget.z 0.5 1.0e-12)
    s!"move_jaco_ee z target mismatch: {defaultMoveTarget.z}"
  LeanTest.assertTrue (approx defaultMoveTarget.roll (-1.7) 1.0e-12)
    s!"move_jaco_ee roll target mismatch: {defaultMoveTarget.roll}"
  LeanTest.assertTrue (approx defaultMoveTarget.pitch (-1.3) 1.0e-12)
    s!"move_jaco_ee pitch target mismatch: {defaultMoveTarget.pitch}"
  LeanTest.assertTrue (approx defaultMoveTarget.yaw (-1.8) 1.0e-12)
    s!"move_jaco_ee yaw target mismatch: {defaultMoveTarget.yaw}"
  LeanTest.assertEqual defaultMoveTarget.endEffectorName "j2s7s300_end_effector"

@[test]
def testEndToEndGraphKeepsClockedBoundaryAndLocalPidBlock : IO Unit := do
  let result ← assertOk buildEndToEnd? "Kinova Jaco end-to-end controller"
  LeanTest.assertEqual result.references.size drakeReferences.size
  LeanTest.assertEqual result.buildTargets.size 3
  LeanTest.assertEqual result.controllerOutput.feedback.size defaultDof
  LeanTest.assertEqual result.commandVelocity.size defaultDof
  LeanTest.assertEqual result.commandMessage.jointPosition.size defaultArmJoints
  LeanTest.assertEqual result.commandMessage.fingerPosition.size defaultFingers
  LeanTest.assertEqual result.statusMessage.jointTorque.size defaultArmJoints
  LeanTest.assertEqual result.statusMessage.fingerTorque.size defaultFingers
  LeanTest.assertEqual result.graph.vertices.size 11
  LeanTest.assertEqual result.graph.moves.size 6
  LeanTest.assertTrue (result.graph.containsMoveKind .clockedUpdate)
    "Plan/status boundary should be represented as a clocked update"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "PID algebra should be represented as an exact local block"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "BUILD/runfile resolution should be represented as a checkpoint boundary"
  LeanTest.assertTrue (result.graph.moves.any (fun move =>
      move.kind == .checkpointBoundary &&
      move.label.contains "@drake_models//:jaco_description"))
    "Graph should expose Drake model runfile resolution"
  LeanTest.assertTrue (result.graph.moves.any (fun move =>
      move.kind == .localSchurBlock &&
      move.label.contains "jaco_simulation" &&
      move.label.contains "AddDefaultVisualization"))
    "Graph should expose the jaco_simulation executable setup"
  LeanTest.assertTrue (result.graph.moves.any (fun move =>
      move.kind == .localSchurBlock &&
      move.label.contains "move_jaco_ee" &&
      move.label.contains "MoveIkDemoBase"))
    "Graph should expose the move_jaco_ee planning executable"
  LeanTest.assertTrue (result.graph.moves[4]!.exactness == MoveExactness.exact)
    "Clocked plan interpolation is exact in this fixture"
  LeanTest.assertTrue (result.graph.moves[5]!.exactness == MoveExactness.exact)
    "PID/feedforward algebra is an exact local block"

end Tests.EventSkeletonKinovaJacoArmExample
