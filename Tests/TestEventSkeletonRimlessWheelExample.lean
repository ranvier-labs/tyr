import LeanTest
import Tyr.EventSkeleton.Examples.RimlessWheel

namespace Tests.EventSkeletonRimlessWheelExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.RimlessWheel

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def testPi : Float := 3.14159265358979323846

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def assertSome {α : Type} (x : Option α) (label : String) : IO α := do
  match x with
  | some value => pure value
  | none => LeanTest.fail s!"{label}: expected some"

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

private def hasMoveLabel (moves : Array SkeletonMove) (needle : String) : Bool :=
  moves.any (fun move => move.label.contains needle)

private def hybridFromCont (x : WheelState) : WheelHybridState :=
  { cont := x, toe := 0.0, doubleSupport := false }

private def runStep (x : WheelHybridState) : IO SimulationResult := do
  assertOk (simulate? params 0.2 x 16) "rimless-wheel step run"

@[test]
def testDrakeReferencesAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel.cc"))
    "Example should reference Drake's RimlessWheel implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel.h"))
    "Example should reference Drake's RimlessWheel declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel_continuous_state.h"))
    "Example should reference Drake's RimlessWheel continuous-state BasicVector header"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel_continuous_state.cc"))
    "Example should reference Drake's RimlessWheel continuous-state coordinate names"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/simulate.cc"))
    "Example should reference Drake's RimlessWheel simulate executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel_geometry.h"))
    "Example should reference Drake's RimlessWheel geometry wiring"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel_geometry.cc"))
    "Example should reference Drake's RimlessWheel geometry implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/test/rimless_wheel_geometry_test.cc"))
    "Example should reference Drake's RimlessWheel geometry acceptance test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/test/rimless_wheel_test.cc"))
    "Example should reference Drake's RimlessWheel tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/rimless_wheel/rimless_wheel_params.cc"))
    "Example should reference Drake's RimlessWheel params coordinate names"

@[test]
def testGeneratedVectorBoundariesMatchDrakeBasicVectors : IO Unit := do
  assertOk rimlessWheelContinuousStateVectorBoundary.validate?
    "RimlessWheel continuous-state vector boundary"
  assertOk rimlessWheelParamsVectorBoundary.validate?
    "RimlessWheel params vector boundary"
  LeanTest.assertEqual rimlessWheelContinuousStateVectorBoundary.coordinateNames
    #["theta", "thetadot"]
    "Continuous state coordinate order should match RimlessWheelContinuousStateIndices"
  LeanTest.assertEqual rimlessWheelContinuousStateVectorBoundary.defaults #[0.0, 0.0]
    "Continuous state defaults should match Drake's BasicVector constructor"
  LeanTest.assertEqual (rimlessWheelContinuousStateVectorBoundary.indexOf? "thetadot")
    (some 1)
    "Generated vector lookup should preserve Drake's thetadot row"
  LeanTest.assertEqual rimlessWheelParamsVectorBoundary.coordinateNames
    #["mass", "length", "gravity", "number_of_spokes", "slope"]
    "Parameter coordinate order should match RimlessWheelParamsIndices"
  LeanTest.assertEqual rimlessWheelParamsVectorBoundary.defaults
    #[1.0, 1.0, 9.81, 8.0, 0.08]
    "Parameter defaults should match Drake's RimlessWheelParams constructor"
  LeanTest.assertEqual rimlessWheelParamsVectorBoundary.lowerBounds
    #[some 0.0, some 0.0, some 0.0, some 4.0, none]
    "Parameter lower bounds should match RimlessWheelParams::GetElementBounds"
  LeanTest.assertEqual rimlessWheelParamsVectorBoundary.upperBounds
    #[none, none, none, none, none]
    "Parameter upper bounds should match RimlessWheelParams::GetElementBounds"
  LeanTest.assertEqual (rimlessWheelParamsVectorBoundary.indexOf? "number_of_spokes")
    (some 3)
    "Generated vector lookup should preserve Drake's number_of_spokes row"
  let p ← assertOk (WheelParams.fromArray? rimlessWheelParamsVectorBoundary.defaults)
    "RimlessWheelParams defaults round-trip"
  LeanTest.assertTrue (approx p.numberOfSpokes params.numberOfSpokes 1.0e-12)
    s!"Round-tripped spoke count should match defaults, got {p.numberOfSpokes}"
  let x ← assertOk (stateFromArray? #[0.2, -0.3])
    "RimlessWheelContinuousState array decode"
  LeanTest.assertEqual (stateAsArray x) #[0.2, -0.3]
    "Continuous-state array decode should preserve coordinate order"
  let invalidMsg ← assertError
    (WheelParams.fromArray? #[1.0, 1.0, 9.81, 3.0, 0.08])
    "RimlessWheelParams invalid spoke count"
  LeanTest.assertTrue (invalidMsg.contains "outside Drake")
    s!"Expected Drake-domain diagnostic, got {invalidMsg}"

@[test]
def testRimlessWheelGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let floating := floatingBaseState params (initialState params)
  let result ← assertOk (buildRimlessWheelGeometry? params floating)
    "RimlessWheelGeometry provider"
  assertOk result.provider.validate? "RimlessWheelGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "RimlessWheelGeometry pose output"
  LeanTest.assertEqual result.inputPortName "floating_base_state"
    "RimlessWheelGeometry should declare Drake's floating-base state input"
  LeanTest.assertEqual result.inputPortSize 12
    "RimlessWheelGeometry input should use Drake's 12-vector floating-base state"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "RimlessWheelGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertEqual result.provider.sources.size 1
    "RimlessWheelGeometry should register one SceneGraph source"
  LeanTest.assertEqual result.provider.frames.size 1
    "RimlessWheelGeometry should register the center frame"
  let frame ← assertSome (result.provider.frameById? rimlessWheelCenterFrameId)
    "rimless-wheel center frame lookup"
  LeanTest.assertEqual frame.name "center"
    "RimlessWheelGeometry frame should use Drake's center name"
  LeanTest.assertEqual (result.provider.anchoredGeometries.map (fun g => g.id))
    #[rimlessWheelRampGeometryId]
    "RimlessWheelGeometry should anchor only the ramp"
  let spokeCount ← assertOk (rimlessWheelSpokeCount? params)
    "rimless-wheel spoke count"
  LeanTest.assertEqual result.provider.geometries.size (2 + spokeCount)
    "RimlessWheelGeometry should register ramp, hub, and every spoke"

  let ramp ← assertSome (result.provider.geometryById? rimlessWheelRampGeometryId)
    "rimless-wheel ramp geometry lookup"
  LeanTest.assertEqual ramp.name "ramp"
    "Ramp geometry should preserve Drake's name"
  LeanTest.assertTrue ramp.isAnchored
    "Ramp geometry should be anchored in world"
  LeanTest.assertTrue (ramp.hasRole .illustration && ramp.hasRole .perception)
    "Ramp should carry illustration and perception roles"
  LeanTest.assertTrue
    (ramp.properties.diffuseRgba? ==
      some { r := 0.9297, g := 0.7930, b := 0.6758, a := 1.0 })
    s!"Ramp should carry Drake's desert-sand diffuse color, got {reprStr ramp.properties.diffuseRgba?}"
  LeanTest.assertTrue (ramp.X_FG.rotationAxis == SceneVec3.unitY)
    s!"Ramp should rotate around Y by the slope, got {reprStr ramp.X_FG.rotationAxis}"
  LeanTest.assertTrue (approx ramp.X_FG.rotationAngle params.slope 1.0e-12)
    s!"Ramp pitch should equal slope, got {ramp.X_FG.rotationAngle}"
  LeanTest.assertTrue (approx ramp.X_FG.translation.z (-5.0) 1.0e-12)
    s!"Ramp center should sit half its 10m depth below the origin, got {ramp.X_FG.translation.z}"
  match ramp.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue
        (approx sx 100.0 1.0e-12 && approx sy 1.0 1.0e-12 && approx sz 10.0 1.0e-12)
        s!"Ramp box should be 100 x 1 x 10, got {sx}, {sy}, {sz}"
  | other => LeanTest.fail s!"Rimless-wheel ramp should be a box, got {reprStr other}"

  let hub ← assertSome (result.provider.geometryById? rimlessWheelHubGeometryId)
    "rimless-wheel hub geometry lookup"
  LeanTest.assertEqual hub.name "hub"
    "Hub geometry should preserve Drake's name"
  LeanTest.assertEqual hub.frameId? (some rimlessWheelCenterFrameId)
    "Hub geometry should attach to the center frame"
  LeanTest.assertTrue (hub.X_FG.rotationAxis == SceneVec3.unitX)
    s!"Hub cylinder should be rotated around X, got {reprStr hub.X_FG.rotationAxis}"
  LeanTest.assertTrue (approx hub.X_FG.rotationAngle (testPi / 2.0) 1.0e-12)
    s!"Hub cylinder should use Drake's M_PI_2 X rotation, got {hub.X_FG.rotationAngle}"
  LeanTest.assertTrue
    (hub.properties.diffuseRgba? == some { r := 0.6, g := 0.2, b := 0.2, a := 1.0 })
    s!"Hub should carry Drake's reddish diffuse color, got {reprStr hub.properties.diffuseRgba?}"
  match hub.shape with
  | .cylinder radius length =>
      LeanTest.assertTrue (approx radius 0.2 1.0e-12 && approx length 0.2 1.0e-12)
        s!"Hub cylinder should have radius and length 0.2, got {radius}, {length}"
  | other => LeanTest.fail s!"Rimless-wheel hub should be a cylinder, got {reprStr other}"

  let spoke0 ← assertSome (result.provider.geometryById? rimlessWheelSpokeGeometryBaseId)
    "rimless-wheel spoke0 geometry lookup"
  LeanTest.assertEqual spoke0.name "spoke0"
    "Spoke geometry should preserve Drake's indexed names"
  LeanTest.assertEqual spoke0.frameId? (some rimlessWheelCenterFrameId)
    "Spokes should attach to the center frame"
  LeanTest.assertTrue (spoke0.X_FG.rotationAxis == SceneVec3.unitY)
    s!"Spokes should rotate around Y, got {reprStr spoke0.X_FG.rotationAxis}"
  LeanTest.assertTrue (approx spoke0.X_FG.rotationAngle 0.0 1.0e-12)
    s!"spoke0 should have zero Y rotation, got {spoke0.X_FG.rotationAngle}"
  LeanTest.assertTrue (approx spoke0.X_FG.translation.z (-params.length / 2.0) 1.0e-12)
    s!"Spokes should be centered at -length/2, got {spoke0.X_FG.translation.z}"
  LeanTest.assertTrue
    (spoke0.properties.diffuseRgba? == some { r := 0.0, g := 0.0, b := 0.0, a := 1.0 })
    s!"Spokes should be black, got {reprStr spoke0.properties.diffuseRgba?}"
  match spoke0.shape with
  | .cylinder radius length =>
      LeanTest.assertTrue (approx radius 0.0075 1.0e-12 && approx length params.length 1.0e-12)
        s!"Spoke cylinder should have radius 0.0075 and wheel length, got {radius}, {length}"
  | other => LeanTest.fail s!"Rimless-wheel spoke should be a cylinder, got {reprStr other}"

  let spoke2 ← assertSome (result.provider.geometryById? (rimlessWheelSpokeGeometryBaseId + 2))
    "rimless-wheel spoke2 geometry lookup"
  LeanTest.assertTrue
    (approx spoke2.X_FG.rotationAngle (2.0 * testPi * 2.0 / params.numberOfSpokes) 1.0e-12)
    s!"spoke2 rotation should be 2*pi*2/num_spokes, got {spoke2.X_FG.rotationAngle}"

@[test]
def testRimlessWheelGeometryPoseOutputMatchesDrakeFloatingBaseState : IO Unit := do
  let state : WheelHybridState :=
    { cont := { theta := 0.12, thetaDot := 1.7 }, toe := 0.3, doubleSupport := false }
  let floating := floatingBaseState params state
  let result ← assertOk (buildRimlessWheelGeometry? params floating)
    "RimlessWheelGeometry pose output"
  let pose ← assertSome (result.poses.poseForFrame? rimlessWheelCenterFrameId)
    "rimless-wheel center pose"
  LeanTest.assertTrue (approx pose.translation.x floating[0]! 1.0e-12 &&
      approx pose.translation.y floating[1]! 1.0e-12 &&
      approx pose.translation.z floating[2]! 1.0e-12)
    s!"Pose translation should copy floating_base_state head, got {reprStr pose.translation}"
  LeanTest.assertTrue (approx pose.rotationAxis.x 0.0 1.0e-12 &&
      approx pose.rotationAxis.y 1.0 1.0e-12 &&
      approx pose.rotationAxis.z 0.0 1.0e-12)
    s!"Pure pitch floating-base state should produce a Y-axis rotation, got {reprStr pose.rotationAxis}"
  LeanTest.assertTrue (approx pose.rotationAngle floating[4]! 1.0e-12)
    s!"Pose pitch angle should copy floating_base_state[4], got {pose.rotationAngle}"
  let xAxis_W := pose.rotateVector SceneVec3.unitX
  LeanTest.assertTrue
    (approx xAxis_W.x (Float.cos floating[4]!) 1.0e-12 &&
      approx xAxis_W.z (-(Float.sin floating[4]!)) 1.0e-12)
    s!"Pose rotation should use Drake's roll-pitch-yaw convention, got {reprStr xAxis_W}"

@[test]
def testRimlessWheelGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let result ← assertOk (buildRimlessWheelGeometry? params (floatingBaseState params (initialState params)))
    "RimlessWheelGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "RimlessWheelGeometry should expose registration and pose-output local moves"
  LeanTest.assertTrue (result.moves.all (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.exact))
    "RimlessWheelGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[rimlessWheelGeometryProviderVertex] &&
      move.writes == #[rimlessWheelGeometryProviderVertex] &&
      move.label.contains "Register ramp"))
    "RimlessWheelGeometry graph should record the provider registration move"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[rimlessWheelGeometryPoseOutputVertex] &&
      move.reads == #[rimlessWheelGeometryStateInputVertex, rimlessWheelGeometryProviderVertex] &&
      move.writes == #[rimlessWheelGeometryPoseOutputVertex] &&
      move.label.contains "OutputGeometryPose"))
    "RimlessWheelGeometry graph should record the floating-state-to-FramePoseVector move"

@[test]
def testSimulateExecutableBoundaryMatchesDrakeMain : IO Unit := do
  let boundary ← assertOk (buildSimulateExecutableBoundary? params simulateExecutableConfig)
    "rimless-wheel simulate executable boundary"
  let cfg := boundary.config
  LeanTest.assertTrue (approx cfg.accuracy 1.0e-4 1.0e-12)
    s!"Drake simulate.cc default accuracy should be 1e-4, got {cfg.accuracy}"
  LeanTest.assertTrue (approx cfg.initialAngle 0.0 1.0e-12)
    s!"Drake simulate.cc default initial angle should be 0, got {cfg.initialAngle}"
  LeanTest.assertTrue (approx cfg.initialAngularVelocity 5.0 1.0e-12)
    s!"Drake simulate.cc default angular velocity should be 5, got {cfg.initialAngularVelocity}"
  LeanTest.assertTrue (approx cfg.targetRealtimeRate 1.0 1.0e-12)
    s!"Drake simulate.cc default realtime rate should be 1, got {cfg.targetRealtimeRate}"
  LeanTest.assertTrue (approx cfg.advanceTo 10.0 1.0e-12)
    s!"Drake simulate.cc should advance to 10s, got {cfg.advanceTo}"
  LeanTest.assertEqual cfg.maxSteps 256
    "Boundary should preserve a finite event budget for our executable primitive"
  LeanTest.assertEqual cfg.plantName "rimless_wheel"
    "Boundary should preserve Drake's plant name"
  LeanTest.assertTrue cfg.includeSceneGraph
    "Boundary should include SceneGraph geometry like Drake simulate.cc"
  LeanTest.assertTrue cfg.includeDrakeVisualizer
    "Boundary should include DrakeVisualizerd like Drake simulate.cc"
  LeanTest.assertTrue
    (cfg.initialAngle > params.slope - alpha params &&
      cfg.initialAngle < params.slope + alpha params)
    "Default initial angle should satisfy Drake's above-ground demand"
  LeanTest.assertTrue (approx boundary.initialState.cont.theta 0.0 1.0e-12)
    s!"Boundary initial theta mismatch: {boundary.initialState.cont.theta}"
  LeanTest.assertTrue (approx boundary.initialState.cont.thetaDot 5.0 1.0e-12)
    s!"Boundary initial thetadot mismatch: {boundary.initialState.cont.thetaDot}"

  LeanTest.assertEqual boundary.graph.vertices.size 7
    "Rimless-wheel simulate boundary should expose flags, plant, geometry, visualizer, context, interval, and final checkpoint"
  LeanTest.assertEqual boundary.moves.size 6
    "Rimless-wheel simulate boundary should expose diagram setup, context checkpoint, simulator interval, and final demand"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .localSchurBlock)
    "Diagram and visualization wiring should be local Schur blocks"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .intervalAdjoint)
    "Simulator.AdvanceTo should remain an interval-adjoint primitive"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .checkpointBoundary)
    "Initial and final simulator contexts should be checkpoints"
  LeanTest.assertTrue
    (boundary.moves.any (fun move =>
      move.label.contains "ODE, event-tree witnesses, and impact resets"))
    "Executable boundary should state that it runs through the real hybrid wheel primitive"

  let shortCfg : SimulateExecutableConfig :=
    { simulateExecutableConfig with advanceTo := 0.01, maxSteps := 4 }
  let short ← assertOk (executeSimulateExecutable? params shortCfg)
    "short rimless-wheel simulate executable run"
  LeanTest.assertTrue (approx short.finalTime 0.01 1.0e-12)
    s!"Short executable run should advance to requested time, got {short.finalTime}"
  LeanTest.assertEqual short.steps.size 0
    "Short default run should not hit a spoke-impact event"
  LeanTest.assertTrue (shortCfg.finalThetaInsideSpokeInterval params short.finalState)
    s!"Short executable run should satisfy final theta postcondition, got {short.finalState.cont.theta}"

@[test]
def testDynamicsAndResetMatchDrakeModel : IO Unit := do
  let x : WheelState := { theta := 0.2, thetaDot := 3.0 }
  let dx := derivative params false x
  LeanTest.assertTrue (approx dx.theta 3.0 1.0e-12)
    s!"theta derivative should be thetadot, got {dx.theta}"
  LeanTest.assertTrue
    (approx dx.thetaDot (Float.sin 0.2 * params.gravity / params.length) 1.0e-12)
    s!"thetadot derivative should be sin(theta) * g / l, got {dx.thetaDot}"

  let a := alpha params
  let pre : WheelHybridState :=
    { cont := { theta := params.slope + a, thetaDot := 2.0 }, toe := 0.0, doubleSupport := false }
  let reset := applyReset params .forward pre
  LeanTest.assertTrue
    (approx reset.state.cont.theta (params.slope - a + params.resetClearance) 1.0e-12)
    s!"Forward reset should switch stance leg, got theta={reset.state.cont.theta}"
  LeanTest.assertTrue
    (approx reset.state.cont.thetaDot (2.0 * Float.cos (2.0 * a)) 1.0e-12)
    s!"Forward reset should scale velocity by cos(2 alpha), got {reset.state.cont.thetaDot}"
  LeanTest.assertTrue (approx reset.state.toe (stepLength params) 1.0e-12)
    s!"Forward reset should advance toe by one step length, got {reset.state.toe}"

@[test]
def testForwardLimitCycleStepMatchesDrakeChecks : IO Unit := do
  let steadyEnergy := fixedPointEnergy params
  let result ← runStep (hybridFromCont (limitCyclePreForwardState params))
  LeanTest.assertEqual result.steps.size 1
    "Limit-cycle run to t=0.2 should take one forward step"
  let step := result.steps[0]!
  LeanTest.assertTrue (step.direction == StepDirection.forward)
    s!"Expected forward step, got {reprStr step.direction}"
  LeanTest.assertTrue (result.finalState.cont.theta < 0.0)
    s!"After a forward step theta should be on the other side of zero, got {result.finalState.cont.theta}"
  LeanTest.assertTrue (approx result.finalState.toe (stepLength params) 1.0e-8)
    s!"Toe should advance one step length, got {result.finalState.toe}"
  LeanTest.assertTrue (approx (totalEnergy params result.finalState.cont) steadyEnergy 2.0e-5)
    s!"Limit-cycle energy should be preserved, got {totalEnergy params result.finalState.cont}, expected {steadyEnergy}"
  let floating := floatingBaseState params result.finalState
  LeanTest.assertTrue (floating[4]! > params.slope + alpha params)
    s!"Floating-base pitch should unroll past one step, got {floating[4]!}"

@[test]
def testFastSlowAndBackwardEnergyCasesMatchDrakeChecks : IO Unit := do
  let base := limitCyclePreForwardState params

  let fast0 := hybridFromCont { base with thetaDot := base.thetaDot + 0.2 }
  let fastEnergy := totalEnergy params fast0.cont
  let fast ← runStep fast0
  LeanTest.assertTrue (fast.steps[0]!.direction == StepDirection.forward)
    s!"Fast downhill case should step forward, got {reprStr fast.steps[0]!.direction}"
  LeanTest.assertTrue (totalEnergy params fast.finalState.cont < fastEnergy - 0.01)
    s!"Fast downhill impact should lose energy, got {totalEnergy params fast.finalState.cont} from {fastEnergy}"

  let slow0 := hybridFromCont { base with thetaDot := base.thetaDot - 0.2 }
  let slowEnergy := totalEnergy params slow0.cont
  let slow ← runStep slow0
  LeanTest.assertTrue (slow.steps[0]!.direction == StepDirection.forward)
    s!"Slow downhill case should step forward, got {reprStr slow.steps[0]!.direction}"
  LeanTest.assertTrue (totalEnergy params slow.finalState.cont > slowEnergy + 0.01)
    s!"Slow downhill impact should gain energy, got {totalEnergy params slow.finalState.cont} from {slowEnergy}"

  let back0 := hybridFromCont {
    theta := params.slope - alpha params / 2.0
    thetaDot := -4.0
  }
  let backEnergy := totalEnergy params back0.cont
  let back ← runStep back0
  LeanTest.assertTrue (back.steps[0]!.direction == StepDirection.backward)
    s!"Uphill case should step backward, got {reprStr back.steps[0]!.direction}"
  LeanTest.assertTrue (back.finalState.cont.theta > 0.0)
    s!"After a backward step theta should be on the other side of zero, got {back.finalState.cont.theta}"
  LeanTest.assertTrue (approx back.finalState.toe (-(stepLength params)) 1.0e-8)
    s!"Backward reset should move toe backward one step length, got {back.finalState.toe}"
  LeanTest.assertTrue (totalEnergy params back.finalState.cont < backEnergy + 0.1)
    s!"Backward step should not gain more than Drake's tolerance, got {totalEnergy params back.finalState.cont} from {backEnergy}"

@[test]
def testDoubleSupportFixedPointStops : IO Unit := do
  let a := alpha params
  let angleAboveTouchdown := 1.0e-5

  let front0 := hybridFromCont {
    theta := params.slope + a - angleAboveTouchdown
    thetaDot := 0.0
  }
  let front ← runStep front0
  LeanTest.assertTrue front.finalState.doubleSupport
    "Front-foot near-touchdown run should enter double support"
  LeanTest.assertTrue (approx (Float.abs (front.finalState.cont.theta - params.slope)) a 1.0e-7)
    s!"Double-support front theta should sit at one spoke angle from slope, got {front.finalState.cont.theta}"
  LeanTest.assertTrue (approx front.finalState.cont.thetaDot 0.0 1.0e-12)
    s!"Double support should clamp angular velocity to zero, got {front.finalState.cont.thetaDot}"

  let back0 := hybridFromCont {
    theta := params.slope - a + angleAboveTouchdown
    thetaDot := 0.0
  }
  let back ← runStep back0
  LeanTest.assertTrue back.finalState.doubleSupport
    "Back-foot near-touchdown run should enter double support"
  LeanTest.assertTrue (approx (Float.abs (back.finalState.cont.theta - params.slope)) a 1.0e-7)
    s!"Double-support back theta should sit at one spoke angle from slope, got {back.finalState.cont.theta}"
  LeanTest.assertTrue (approx back.finalState.cont.thetaDot 0.0 1.0e-12)
    s!"Double support should clamp angular velocity to zero, got {back.finalState.cont.thetaDot}"

@[test]
def testSaltationAndTraceRecordExecutableStep : IO Unit := do
  let a := alpha params
  let pre : WheelHybridState :=
    { cont := { theta := params.slope + a, thetaDot := 2.0 }, toe := 0.0, doubleSupport := false }
  let reset := applyReset params .forward pre
  let data := stepSaltationData params .forward pre reset
  LeanTest.assertTrue (approx data.gamma (-2.0) 1.0e-12)
    s!"Forward guard transversality should be -thetadot, got {data.gamma}"
  let matrix ← assertOk data.saltationMatrix? "forward saltation matrix"
  LeanTest.assertEqual matrix.size 2
    "Rimless-wheel saltation should be a 2x2 continuous-state matrix"
  LeanTest.assertEqual matrix[0]!.size 2
    "Rimless-wheel saltation should be a 2x2 continuous-state matrix"

  let result ← runStep (hybridFromCont (limitCyclePreForwardState params))
  match result.trace.validate? with
  | .error msg => LeanTest.fail s!"Rimless-wheel trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertEqual result.trace.entries.size (2 * result.steps.size + 1)
    "Trace should contain one localized interval and one saltation per step, plus the terminal interval"
  LeanTest.assertEqual result.moves.size (4 * result.steps.size + 2)
    "Each step contributes interval/checkpoint and saltation/reset moves; terminal interval contributes interval/checkpoint"
  match result.trace.entries[0]! with
  | .interval segment =>
      LeanTest.assertTrue segment.localizedByEvent
        s!"First interval should be localized by a rimless-wheel witness, got {reprStr segment}"
  | other => LeanTest.fail s!"Expected first trace entry to be an interval, got {reprStr other}"
  match result.trace.entries[1]! with
  | .saltation vertex _ =>
      LeanTest.assertEqual vertex (stepEventVertex 0)
        "Second trace entry should be the first rimless-wheel saltation vertex"
  | other => LeanTest.fail s!"Expected second trace entry to be a saltation event, got {reprStr other}"

@[test]
def testEndToEndResultConnectsDrakeBoundariesToExecutableRun : IO Unit := do
  let result ← assertOk (buildEndToEnd? params simulateExecutableConfig)
    "rimless-wheel end-to-end result"
  assertOk result.continuousStateBoundary.validate?
    "end-to-end continuous-state boundary"
  assertOk result.paramsBoundary.validate?
    "end-to-end params boundary"
  assertOk result.executableRun.trace.validate?
    "end-to-end executable trace"
  LeanTest.assertEqual result.continuousStateBoundary.coordinateNames
    #["theta", "thetadot"]
    "End-to-end result should carry the generated continuous-state BasicVector"
  LeanTest.assertEqual result.paramsBoundary.coordinateNames
    #["mass", "length", "gravity", "number_of_spokes", "slope"]
    "End-to-end result should carry the generated params BasicVector"
  LeanTest.assertEqual result.geometry.inputPortName "floating_base_state"
    "End-to-end geometry should be wired from Drake's floating-base state port"
  LeanTest.assertEqual result.executableRun.finalTime simulateExecutableConfig.advanceTo
    "End-to-end executable run should advance to the Drake simulate.cc horizon"
  LeanTest.assertTrue result.finalThetaInsideSpokeInterval
    s!"End-to-end executable run should satisfy Drake's final theta demand, got {result.executableRun.finalState.cont.theta}"
  LeanTest.assertTrue (result.executableRun.steps.size > 0)
    "End-to-end executable run should include actual rimless-wheel impact events"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "generated BasicVector boundary: RimlessWheelContinuousState")
    "End-to-end graph should include the continuous-state generated-vector boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "generated BasicVector boundary: RimlessWheelParams")
    "End-to-end graph should include the params generated-vector boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "RimlessWheelGeometry::AddToBuilder")
    "End-to-end graph should include the SceneGraph geometry builder boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "Simulator.AdvanceTo via rimless-wheel ODE")
    "End-to-end graph should include the executable simulator boundary"
  LeanTest.assertTrue
    (countMoveKind result.moves .localSchurBlock >= 7)
    "End-to-end graph should retain vector, geometry, and executable local blocks"
  LeanTest.assertTrue
    (countMoveKind result.moves .intervalAdjoint > 0)
    "End-to-end graph should retain interval-adjoint simulation moves"
  LeanTest.assertTrue
    (countMoveKind result.moves .saltationTime > 0)
    "End-to-end graph should retain impact saltation timing moves"

end Tests.EventSkeletonRimlessWheelExample
