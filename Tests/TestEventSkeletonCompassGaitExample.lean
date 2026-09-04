import LeanTest
import Tyr.EventSkeleton.Examples.CompassGait

namespace Tests.EventSkeletonCompassGaitExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.CompassGait

private def pi : Float := 3.14159265358979323846

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def maxAbsDiff (actual expected : Array Float) : Float :=
  FloatArray.maxAbsDiff actual expected

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

private def oneCollisionInitial : CompassHybridState :=
  {
    cont := {
      stance := params.slope + 0.1
      swing := params.slope - 0.15
      stanceDot := 0.6
      swingDot := 0.4
    }
    toe := 0.0
    leftSupport := true
  }

@[test]
def testDrakeReferencesAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait.cc"))
    "Example should reference Drake's CompassGait implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait.h"))
    "Example should reference Drake's CompassGait LeafSystem declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait_continuous_state.h"))
    "Example should reference Drake's CompassGait continuous-state BasicVector header"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait_continuous_state.cc"))
    "Example should reference Drake's CompassGait continuous-state coordinate names"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/simulate.cc"))
    "Example should reference Drake's CompassGait simulate executable"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait_geometry.h"))
    "Example should reference Drake's CompassGait geometry wiring"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait_geometry.cc"))
    "Example should reference Drake's CompassGait geometry implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/test/compass_gait_geometry_test.cc"))
    "Example should reference Drake's CompassGait geometry acceptance test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/test/compass_gait_test.cc"))
    "Example should reference Drake's CompassGait tests"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/compass_gait/compass_gait_params.cc"))
    "Example should reference Drake's CompassGait params coordinate names"

@[test]
def testGeneratedVectorBoundariesMatchDrakeBasicVectors : IO Unit := do
  assertOk compassGaitContinuousStateVectorBoundary.validate?
    "CompassGait continuous-state vector boundary"
  assertOk compassGaitParamsVectorBoundary.validate?
    "CompassGait params vector boundary"
  LeanTest.assertEqual compassGaitContinuousStateVectorBoundary.coordinateNames
    #["stance", "swing", "stancedot", "swingdot"]
    "Continuous state coordinate order should match CompassGaitContinuousStateIndices"
  LeanTest.assertEqual compassGaitContinuousStateVectorBoundary.defaults
    #[0.0, 0.0, 0.0, 0.0]
    "Continuous state defaults should match Drake's BasicVector constructor"
  LeanTest.assertEqual
    (compassGaitContinuousStateVectorBoundary.indexOf? "stancedot")
    (some 2)
    "Generated vector lookup should preserve Drake's stancedot row"
  LeanTest.assertEqual compassGaitParamsVectorBoundary.coordinateNames
    #["mass_hip", "mass_leg", "length_leg", "center_of_mass_leg", "gravity", "slope"]
    "Parameter coordinate order should match CompassGaitParamsIndices"
  LeanTest.assertEqual compassGaitParamsVectorBoundary.defaults
    #[10.0, 5.0, 1.0, 0.5, 9.81, 0.0525]
    "Parameter defaults should match Drake's CompassGaitParams constructor"
  LeanTest.assertEqual compassGaitParamsVectorBoundary.lowerBounds
    #[some 0.0, some 0.0, some 0.0, some 0.0, some 0.0, some 0.0]
    "Parameter lower bounds should match CompassGaitParams::GetElementBounds"
  LeanTest.assertEqual compassGaitParamsVectorBoundary.upperBounds
    #[none, none, none, none, none, some 1.5707]
    "Parameter upper bounds should match CompassGaitParams::GetElementBounds"
  LeanTest.assertEqual (compassGaitParamsVectorBoundary.indexOf? "slope") (some 5)
    "Generated vector lookup should preserve Drake's slope row"
  let p ← assertOk (CompassParams.fromArray? compassGaitParamsVectorBoundary.defaults)
    "CompassGaitParams defaults round-trip"
  LeanTest.assertTrue (approx p.slope params.slope 1.0e-12)
    s!"Round-tripped slope should match defaults, got {p.slope}"
  let x ← assertOk (stateFromArray? #[0.1, -0.2, 0.3, -0.4])
    "CompassGaitContinuousState array decode"
  LeanTest.assertEqual (stateAsArray x) #[0.1, -0.2, 0.3, -0.4]
    "Continuous-state array decode should preserve coordinate order"
  let invalidMsg ← assertError
    (CompassParams.fromArray? #[10.0, 5.0, 1.0, 0.5, 9.81, 2.0])
    "CompassGaitParams invalid slope"
  LeanTest.assertTrue (invalidMsg.contains "outside Drake")
    s!"Expected Drake-domain diagnostic, got {invalidMsg}"

@[test]
def testCompassGaitGeometryProviderMatchesDrakeSceneGraphRegistration : IO Unit := do
  let floating := floatingBaseState params (initialState params)
  let result ← assertOk (buildCompassGaitGeometry? params floating)
    "CompassGaitGeometry provider"
  assertOk result.provider.validate? "CompassGaitGeometry SceneGraph provider"
  assertOk (result.poses.validate? result.provider) "CompassGaitGeometry pose output"
  LeanTest.assertEqual result.inputPortName "floating_base_state"
    "CompassGaitGeometry should declare Drake's floating-base state input"
  LeanTest.assertEqual result.inputPortSize 14
    "CompassGaitGeometry input should use Drake's 14-vector floating-base state"
  LeanTest.assertEqual result.outputPortName "geometry_pose"
    "CompassGaitGeometry should declare an abstract output port named geometry_pose"
  LeanTest.assertEqual result.provider.sources.size 1
    "CompassGaitGeometry should register one SceneGraph source"
  LeanTest.assertEqual result.provider.frames.size 2
    "CompassGaitGeometry should register left and right leg frames"
  let leftFrame ← assertSome (result.provider.frameById? compassGaitLeftLegFrameId)
    "compass-gait left frame lookup"
  let rightFrame ← assertSome (result.provider.frameById? compassGaitRightLegFrameId)
    "compass-gait right frame lookup"
  LeanTest.assertEqual leftFrame.name "left_leg"
    "CompassGaitGeometry left frame should preserve Drake's name"
  LeanTest.assertEqual rightFrame.name "right_leg"
    "CompassGaitGeometry right frame should preserve Drake's name"
  LeanTest.assertEqual rightFrame.parentFrameId? (some compassGaitLeftLegFrameId)
    "CompassGaitGeometry should register the right leg as a child of the left leg frame"
  LeanTest.assertEqual (result.provider.anchoredGeometries.map (fun g => g.id))
    #[compassGaitRampGeometryId]
    "CompassGaitGeometry should anchor only the ramp"
  LeanTest.assertEqual result.provider.shapeNames
    #["box", "sphere", "cylinder", "sphere", "cylinder", "sphere"]
    "CompassGaitGeometry should register ramp, hip, two leg cylinders, and two leg masses"

  let ramp ← assertSome (result.provider.geometryById? compassGaitRampGeometryId)
    "compass-gait ramp geometry lookup"
  LeanTest.assertEqual ramp.name "ramp"
    "Ramp geometry should preserve Drake's name"
  LeanTest.assertTrue ramp.isAnchored
    "Ramp geometry should be anchored in world"
  LeanTest.assertTrue (ramp.X_FG.rotationAxis == SceneVec3.unitY)
    s!"Ramp should rotate around Y by the slope, got {reprStr ramp.X_FG.rotationAxis}"
  LeanTest.assertTrue (approx ramp.X_FG.rotationAngle params.slope 1.0e-12)
    s!"Ramp pitch should equal slope, got {ramp.X_FG.rotationAngle}"
  LeanTest.assertTrue (approx ramp.X_FG.translation.z (-5.0) 1.0e-12)
    s!"Ramp center should sit half its 10m depth below the origin, got {ramp.X_FG.translation.z}"
  LeanTest.assertTrue
    (ramp.properties.diffuseRgba? ==
      some { r := 0.9297, g := 0.7930, b := 0.6758, a := 1.0 })
    s!"Ramp should carry Drake's desert-sand diffuse color, got {reprStr ramp.properties.diffuseRgba?}"
  match ramp.shape with
  | .box sx sy sz =>
      LeanTest.assertTrue
        (approx sx 100.0 1.0e-12 && approx sy 1.0 1.0e-12 && approx sz 10.0 1.0e-12)
        s!"Ramp box should be 100 x 1 x 10, got {sx}, {sy}, {sz}"
  | other => LeanTest.fail s!"Compass-gait ramp should be a box, got {reprStr other}"

  let hip ← assertSome (result.provider.geometryById? compassGaitHipGeometryId)
    "compass-gait hip geometry lookup"
  LeanTest.assertEqual hip.name "hip"
    "Hip geometry should preserve Drake's name"
  LeanTest.assertEqual hip.frameId? (some compassGaitLeftLegFrameId)
    "Hip geometry should attach to the left leg frame"
  LeanTest.assertTrue (hip.X_FG.rotationAxis == SceneVec3.unitX)
    s!"Hip should use Drake's arbitrary X rotation, got {reprStr hip.X_FG.rotationAxis}"
  LeanTest.assertTrue (approx hip.X_FG.rotationAngle (pi / 2.0) 1.0e-12)
    s!"Hip should use Drake's M_PI_2 X rotation, got {hip.X_FG.rotationAngle}"
  LeanTest.assertTrue
    (hip.properties.diffuseRgba? == some { r := 0.0, g := 1.0, b := 0.0, a := 1.0 })
    s!"Hip should carry Drake's green diffuse color, got {reprStr hip.properties.diffuseRgba?}"
  match hip.shape with
  | .sphere radius =>
      LeanTest.assertTrue (approx radius 0.1 1.0e-12)
        s!"Hip sphere should have radius 0.1, got {radius}"
  | other => LeanTest.fail s!"Compass-gait hip should be a sphere, got {reprStr other}"

  let leftLeg ← assertSome (result.provider.geometryById? compassGaitLeftLegGeometryId)
    "compass-gait left leg geometry lookup"
  LeanTest.assertEqual leftLeg.name "left_leg"
    "Left leg geometry should preserve Drake's name"
  LeanTest.assertEqual leftLeg.frameId? (some compassGaitLeftLegFrameId)
    "Left leg geometry should attach to the left leg frame"
  LeanTest.assertTrue (approx leftLeg.X_FG.translation.z (-params.lengthLeg / 2.0) 1.0e-12)
    s!"Left leg cylinder should be centered at -length/2, got {leftLeg.X_FG.translation.z}"
  LeanTest.assertTrue
    (leftLeg.properties.diffuseRgba? == some { r := 1.0, g := 0.0, b := 0.0, a := 1.0 })
    s!"Left leg should be red, got {reprStr leftLeg.properties.diffuseRgba?}"
  match leftLeg.shape with
  | .cylinder radius length =>
      LeanTest.assertTrue (approx radius 0.0075 1.0e-12 && approx length params.lengthLeg 1.0e-12)
        s!"Left leg cylinder should have radius 0.0075 and leg length, got {radius}, {length}"
  | other => LeanTest.fail s!"Compass-gait left leg should be a cylinder, got {reprStr other}"

  let rightLeg ← assertSome (result.provider.geometryById? compassGaitRightLegGeometryId)
    "compass-gait right leg geometry lookup"
  LeanTest.assertEqual rightLeg.name "right_leg"
    "Right leg geometry should preserve Drake's name"
  LeanTest.assertEqual rightLeg.frameId? (some compassGaitRightLegFrameId)
    "Right leg geometry should attach to the right leg frame"
  LeanTest.assertTrue
    (rightLeg.properties.diffuseRgba? == some { r := 0.0, g := 0.0, b := 1.0, a := 1.0 })
    s!"Right leg should be blue, got {reprStr rightLeg.properties.diffuseRgba?}"

  let leftMass ← assertSome (result.provider.geometryById? compassGaitLeftLegMassGeometryId)
    "compass-gait left leg mass geometry lookup"
  let rightMass ← assertSome (result.provider.geometryById? compassGaitRightLegMassGeometryId)
    "compass-gait right leg mass geometry lookup"
  LeanTest.assertEqual leftMass.name "left_leg_mass"
    "Left leg mass geometry should preserve Drake's name"
  LeanTest.assertEqual rightMass.name "right_leg_mass"
    "Right leg mass geometry should preserve Drake's name"
  LeanTest.assertTrue (approx leftMass.X_FG.translation.z (-params.centerOfMassLeg) 1.0e-12 &&
      approx rightMass.X_FG.translation.z (-params.centerOfMassLeg) 1.0e-12)
    s!"Leg mass spheres should sit at -center_of_mass_leg, got {leftMass.X_FG.translation.z}, {rightMass.X_FG.translation.z}"
  match leftMass.shape, rightMass.shape with
  | .sphere leftRadius, .sphere rightRadius =>
      let expected := compassGaitLegMassRadius params
      LeanTest.assertTrue (approx leftRadius expected 1.0e-12 && approx rightRadius expected 1.0e-12)
        s!"Leg mass radius should scale by constant density, got {leftRadius}, {rightRadius}, expected {expected}"
  | leftShape, rightShape =>
      LeanTest.fail s!"Leg masses should be spheres, got {reprStr leftShape}, {reprStr rightShape}"

@[test]
def testCompassGaitGeometryPoseOutputMatchesDrakeFloatingBaseState : IO Unit := do
  let state : CompassHybridState :=
    {
      cont := { stance := 0.2, swing := -0.35, stanceDot := 1.1, swingDot := -0.7 }
      toe := 0.4
      leftSupport := true
    }
  let floating := floatingBaseState params state
  let result ← assertOk (buildCompassGaitGeometry? params floating)
    "CompassGaitGeometry pose output"
  let leftPose ← assertSome (result.poses.poseForFrame? compassGaitLeftLegFrameId)
    "compass-gait left leg pose"
  let rightPose ← assertSome (result.poses.poseForFrame? compassGaitRightLegFrameId)
    "compass-gait right leg pose"
  LeanTest.assertTrue (approx leftPose.translation.x floating[0]! 1.0e-12 &&
      approx leftPose.translation.y floating[1]! 1.0e-12 &&
      approx leftPose.translation.z floating[2]! 1.0e-12)
    s!"Left leg pose translation should copy floating_base_state head, got {reprStr leftPose.translation}"
  LeanTest.assertTrue (approx leftPose.rotationAxis.x 0.0 1.0e-12 &&
      approx leftPose.rotationAxis.y 1.0 1.0e-12 &&
      approx leftPose.rotationAxis.z 0.0 1.0e-12)
    s!"Left leg pure pitch should produce a Y-axis rotation, got {reprStr leftPose.rotationAxis}"
  LeanTest.assertTrue (approx leftPose.rotationAngle floating[4]! 1.0e-12)
    s!"Left leg pose pitch should copy floating_base_state[4], got {leftPose.rotationAngle}"
  LeanTest.assertTrue (rightPose.rotationAxis == SceneVec3.unitY)
    s!"Right leg relative pose should be a Y-axis hip-angle rotation, got {reprStr rightPose.rotationAxis}"
  LeanTest.assertTrue (approx rightPose.rotationAngle floating[6]! 1.0e-12)
    s!"Right leg pose angle should copy floating_base_state[6], got {rightPose.rotationAngle}"
  LeanTest.assertTrue (rightPose.translation == SceneVec3.zero)
    s!"Right leg relative pose should not translate its child frame, got {reprStr rightPose.translation}"

@[test]
def testCompassGaitGeometryGraphRecordsExactSceneGraphBoundary : IO Unit := do
  let result ← assertOk (buildCompassGaitGeometry? params (floatingBaseState params (initialState params)))
    "CompassGaitGeometry graph"
  LeanTest.assertEqual result.moves.size 2
    "CompassGaitGeometry should expose registration and pose-output local moves"
  LeanTest.assertTrue (result.moves.all (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.exact))
    "CompassGaitGeometry moves should be exact local SceneGraph blocks"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[compassGaitGeometryProviderVertex] &&
      move.writes == #[compassGaitGeometryProviderVertex] &&
      move.label.contains "Register ramp"))
    "CompassGaitGeometry graph should record the provider registration move"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.targets == #[compassGaitGeometryPoseOutputVertex] &&
      move.reads == #[compassGaitGeometryStateInputVertex, compassGaitGeometryProviderVertex] &&
      move.writes == #[compassGaitGeometryPoseOutputVertex] &&
      move.label.contains "OutputGeometryPose"))
    "CompassGaitGeometry graph should record the floating-state-to-FramePoseVector move"

@[test]
def testSimulateExecutableBoundaryMatchesDrakeMain : IO Unit := do
  let boundary ← assertOk (buildSimulateExecutableBoundary? params simulateExecutableConfig)
    "compass-gait simulate executable boundary"
  let cfg := boundary.config
  LeanTest.assertTrue (approx cfg.targetRealtimeRate 1.0 1.0e-12)
    s!"Drake simulate.cc default realtime rate should be 1, got {cfg.targetRealtimeRate}"
  LeanTest.assertTrue (approx cfg.accuracy 1.0e-4 1.0e-12)
    s!"Drake simulate.cc default accuracy should be 1e-4, got {cfg.accuracy}"
  LeanTest.assertTrue (approx cfg.advanceTo 10.0 1.0e-12)
    s!"Drake simulate.cc should advance to 10s, got {cfg.advanceTo}"
  LeanTest.assertTrue (approx cfg.inputTorque 0.0 1.0e-12)
    s!"Drake simulate.cc should fix zero hip torque, got {cfg.inputTorque}"
  LeanTest.assertEqual cfg.maxCollisions 256
    "Boundary should preserve a finite collision budget for our executable primitive"
  LeanTest.assertEqual cfg.plantName "compass_gait"
    "Boundary should preserve Drake's plant name"
  LeanTest.assertTrue cfg.includeSceneGraph
    "Boundary should include SceneGraph geometry like Drake simulate.cc"
  LeanTest.assertTrue cfg.includeDrakeVisualizer
    "Boundary should include DrakeVisualizerd like Drake simulate.cc"
  LeanTest.assertTrue (maxAbsDiff (stateAsArray cfg.initialContinuousState)
      #[0.0, 0.0, 0.4, -2.0] < 1.0e-12)
    s!"Default initial state should match Drake simulate.cc, got {reprStr cfg.initialContinuousState}"
  LeanTest.assertTrue (maxAbsDiff (stateAsArray boundary.initialState.cont)
      #[0.0, 0.0, 0.4, -2.0] < 1.0e-12)
    s!"Boundary initial hybrid state mismatch: {reprStr boundary.initialState}"

  LeanTest.assertEqual boundary.graph.vertices.size 8
    "Compass-gait simulate boundary should expose flags, plant, geometry, visualizer, torque, context, interval, and final checkpoint"
  LeanTest.assertEqual boundary.moves.size 7
    "Compass-gait simulate boundary should expose diagram setup, fixed torque, context checkpoint, simulator interval, and final checkpoint"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .localSchurBlock)
    "Diagram and visualization wiring should be local Schur blocks"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .freezeControl)
    "Fixed hip torque should be represented as a frozen input"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .intervalAdjoint)
    "Simulator.AdvanceTo should remain an interval-adjoint primitive"
  LeanTest.assertTrue
    (boundary.moves.any (fun move =>
      move.label.contains "collision witness, impulse projection, and leg-swap reset"))
    "Executable boundary should state that it runs through the real hybrid compass-gait primitive"

  let shortCfg : SimulateExecutableConfig :=
    { simulateExecutableConfig with advanceTo := 0.01, maxCollisions := 4 }
  let short ← assertOk (executeSimulateExecutable? params shortCfg)
    "short compass-gait simulate executable run"
  LeanTest.assertTrue (approx short.finalTime 0.01 1.0e-12)
    s!"Short executable run should advance to requested time, got {short.finalTime}"
  LeanTest.assertEqual short.steps.size 0
    "Short default run should not hit a foot-collision event"
  match short.trace.validate? with
  | .error msg => LeanTest.fail s!"Short executable trace should validate: {msg}"
  | .ok () => pure ()

@[test]
def testHipTorqueFixedPointMatchesDrakeCheck : IO Unit := do
  let p := params
  let torque := p.massLeg * p.gravity * p.centerOfMassLeg
  let stance :=
    Float.asin
      (p.massLeg * p.centerOfMassLeg /
        (p.massLeg * p.centerOfMassLeg + (p.massLeg + p.massHip) * p.lengthLeg))
  let x : CompassState :=
    { stance := stance, swing := pi / 2.0, stanceDot := 0.0, swingDot := 0.0 }
  let dx := derivative p torque x
  LeanTest.assertTrue (maxAbsDiff (stateAsArray dx) #[0.0, 0.0, 0.0, 0.0] < 1.0e-12)
    s!"Hip torque should balance the fixed point, got derivative {reprStr dx}"

@[test]
def testCollisionGuardMatchesDrakeScuffingLogic : IO Unit := do
  let above : CompassState :=
    { stance := 0.1, swing := -0.1, stanceDot := 0.0, swingDot := 0.0 }
  LeanTest.assertTrue (footCollision params above > 0.0)
    s!"Swing foot should be above the ramp, got guard {footCollision params above}"

  let backward : CompassState :=
    { stance := -0.1, swing := 0.1, stanceDot := 0.0, swingDot := 0.0 }
  LeanTest.assertTrue (footCollision params backward > 0.0)
    s!"Backward/scuffing configuration should not trigger collision, got guard {footCollision params backward}"

  let touching : CompassState :=
    { stance := params.slope + 0.1, swing := params.slope - 0.1, stanceDot := 0.0, swingDot := 0.0 }
  LeanTest.assertTrue (approx (footCollision params touching) 0.0 1.0e-14)
    s!"Forward foot-touching configuration should be on the witness surface, got {footCollision params touching}"

@[test]
def testCollisionResetConservesAngularMomentum : IO Unit := do
  let pre : CompassHybridState :=
    {
      cont := {
        stance := params.slope + 0.2
        swing := params.slope - 0.2
        stanceDot := 2.0
        swingDot := -1.0
      }
      toe := 0.0
      leftSupport := true
    }
  let before := angularMomentum params pre false
  let reset ← assertOk (applyReset? params pre) "compass-gait collision reset"
  let after := angularMomentum params reset.state true

  LeanTest.assertTrue (approx before after 1.0e-9)
    s!"Impact should conserve angular momentum about the new stance foot, before={before}, after={after}"
  LeanTest.assertTrue (reset.state.toe > pre.toe)
    s!"Collision reset should move the stance toe downhill, got {reset.state.toe}"
  LeanTest.assertTrue (!reset.state.leftSupport)
    "Collision reset should switch support leg"
  LeanTest.assertTrue (approx reset.state.cont.stance pre.cont.swing 1.0e-12)
    s!"Post-impact stance angle should be old swing angle, got {reset.state.cont.stance}"
  LeanTest.assertTrue (approx reset.state.cont.swing pre.cont.stance 1.0e-12)
    s!"Post-impact swing angle should be old stance angle, got {reset.state.cont.swing}"

@[test]
def testFloatingBaseOutputMatchesDrakeLayout : IO Unit := do
  let leftState : CompassHybridState :=
    {
      cont := { stance := 0.0, swing := 1.0, stanceDot := 2.0, swingDot := 3.0 }
      toe := 0.0
      leftSupport := true
    }
  let leftExpected :=
    #[0.0, 0.0, params.lengthLeg, 0.0, 0.0, 0.0, 1.0,
      2.0 * params.lengthLeg, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0]
  LeanTest.assertTrue (maxAbsDiff (floatingBaseState params leftState) leftExpected < 1.0e-12)
    s!"Left-support floating-base output mismatch: {reprStr (floatingBaseState params leftState)}"

  let rightState := { leftState with leftSupport := false }
  let rightExpected :=
    #[0.0, 0.0, params.lengthLeg, 0.0, 1.0, 0.0, -1.0,
      2.0 * params.lengthLeg, 0.0, 0.0, 0.0, 3.0, 0.0, -1.0]
  LeanTest.assertTrue (maxAbsDiff (floatingBaseState params rightState) rightExpected < 1.0e-12)
    s!"Right-support floating-base output mismatch: {reprStr (floatingBaseState params rightState)}"

@[test]
def testSaltationUsesImpulseResetJacobian : IO Unit := do
  let pre : CompassHybridState :=
    {
      cont := {
        stance := params.slope + 0.2
        swing := params.slope - 0.2
        stanceDot := 2.0
        swingDot := -1.0
      }
      toe := 0.0
      leftSupport := true
    }
  let reset ← assertOk (applyReset? params pre) "compass-gait collision reset"
  let data ← assertOk (stepSaltationData? params 0.0 pre reset) "compass-gait saltation data"
  LeanTest.assertTrue (approx data.gamma (-1.0) 1.0e-12)
    s!"Touchdown transversality should be -stancedot-swingdot, got {data.gamma}"
  LeanTest.assertEqual data.resetJac.size 4
    "Compass-gait reset Jacobian should have four continuous-state rows"
  LeanTest.assertEqual data.resetJac[0]!.size 4
    "Compass-gait reset Jacobian should have four continuous-state columns"
  let reverse ← assertOk (data.reverseState? #[0.0, 0.0, 1.0, 0.0]) "compass-gait saltation reverse state"
  LeanTest.assertEqual reverse.size 4
    "Compass-gait reverse saltation should return a 4D cotangent"

@[test]
def testExecutableSimulationRecordsCollisionSkeleton : IO Unit := do
  let result ← assertOk (simulate? params 0.2 oneCollisionInitial 0.0 8)
    "compass-gait short simulation"
  LeanTest.assertTrue (approx result.finalTime 0.2 1.0e-12)
    s!"Simulation should finish at t=0.2, got {result.finalTime}"
  LeanTest.assertEqual result.steps.size 1
    "Short compass-gait run should contain one localized foot collision"
  let step := result.steps[0]!
  LeanTest.assertTrue (step.time > 0.0 && step.time < 0.2)
    s!"Collision should be localized inside the interval, got t={step.time}"
  LeanTest.assertTrue (approx step.angularMomentumBefore step.angularMomentumAfter 1.0e-8)
    s!"Recorded impact should conserve angular momentum, before={step.angularMomentumBefore}, after={step.angularMomentumAfter}"
  LeanTest.assertTrue (step.postState.toe > step.preState.toe)
    s!"Collision should advance toe position, pre={step.preState.toe}, post={step.postState.toe}"

  match result.trace.validate? with
  | .error msg => LeanTest.fail s!"Compass-gait trace should validate: {msg}"
  | .ok () => pure ()
  LeanTest.assertEqual result.trace.entries.size 3
    "Trace should contain localized interval, collision saltation, and terminal interval"
  LeanTest.assertEqual result.moves.size 6
    "One collision run should project interval/checkpoint/saltation/reset plus terminal interval/checkpoint moves"
  match result.trace.entries[0]! with
  | .interval segment =>
      LeanTest.assertTrue segment.localizedByEvent
        s!"First interval should be localized by foot-collision witness, got {reprStr segment}"
  | other => LeanTest.fail s!"Expected first trace entry to be an interval, got {reprStr other}"
  match result.trace.entries[1]! with
  | .saltation vertex data =>
      LeanTest.assertEqual vertex (collisionEventVertex 0)
        "Second trace entry should be the first compass-gait collision saltation vertex"
      LeanTest.assertTrue (data.gamma < 0.0)
        s!"Foot-collision saltation should have negative crossing speed, got {data.gamma}"
  | other => LeanTest.fail s!"Expected second trace entry to be a saltation event, got {reprStr other}"

@[test]
def testEndToEndResultConnectsDrakeBoundariesToExecutableRun : IO Unit := do
  let result ← assertOk (buildEndToEnd? params simulateExecutableConfig)
    "compass-gait end-to-end result"
  assertOk result.continuousStateBoundary.validate?
    "end-to-end continuous-state boundary"
  assertOk result.paramsBoundary.validate?
    "end-to-end params boundary"
  assertOk result.executableRun.trace.validate?
    "end-to-end executable trace"
  LeanTest.assertEqual result.continuousStateBoundary.coordinateNames
    #["stance", "swing", "stancedot", "swingdot"]
    "End-to-end result should carry the generated continuous-state BasicVector"
  LeanTest.assertEqual result.paramsBoundary.coordinateNames
    #["mass_hip", "mass_leg", "length_leg", "center_of_mass_leg", "gravity", "slope"]
    "End-to-end result should carry the generated params BasicVector"
  LeanTest.assertEqual result.geometry.inputPortName "floating_base_state"
    "End-to-end geometry should be wired from Drake's floating-base state port"
  LeanTest.assertEqual result.executableRun.finalTime simulateExecutableConfig.advanceTo
    "End-to-end executable run should advance to the Drake simulate.cc horizon"
  LeanTest.assertTrue (result.executableRun.steps.size > 0)
    "End-to-end executable run should include actual compass-gait collision events"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "generated BasicVector boundary: CompassGaitContinuousState")
    "End-to-end graph should include the continuous-state generated-vector boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "generated BasicVector boundary: CompassGaitParams")
    "End-to-end graph should include the params generated-vector boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "CompassGaitGeometry::AddToBuilder")
    "End-to-end graph should include the SceneGraph geometry builder boundary"
  LeanTest.assertTrue
    (hasMoveLabel result.moves "Simulator.AdvanceTo via compass-gait ODE")
    "End-to-end graph should include the executable simulator boundary"
  LeanTest.assertTrue
    (countMoveKind result.moves .localSchurBlock >= 7)
    "End-to-end graph should retain vector, geometry, and executable local blocks"
  LeanTest.assertTrue
    (countMoveKind result.moves .freezeControl > 0)
    "End-to-end graph should retain the fixed hip-torque input boundary"
  LeanTest.assertTrue
    (countMoveKind result.moves .saltationTime > 0)
    "End-to-end graph should retain collision saltation timing moves"

end Tests.EventSkeletonCompassGaitExample
