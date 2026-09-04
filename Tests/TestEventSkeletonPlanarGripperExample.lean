import LeanTest
import Tyr.EventSkeleton.Examples.PlanarGripper

namespace Tests.EventSkeletonPlanarGripperExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.PlanarGripper

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) :
    IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO Unit := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error _ => pure ()

private def assertSome {α : Type} (value? : Option α) (label : String) :
    IO α := do
  match value? with
  | some value => pure value
  | none => LeanTest.fail s!"{label}: expected some, got none"

private def assertArrayNear
    (actual expected : Array Float)
    (tol : Float)
    (label : String) : IO Unit := do
  let diff := FloatArray.maxAbsDiff actual expected
  LeanTest.assertTrue (diff < tol)
    s!"{label}: max abs diff {diff}, actual={actual}, expected={expected}"

private def assertReference (path : String) (label : String) : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == path))
    label

@[test]
def testDrakeReferencesHelperGeometryAndPositionLayoutAreRecorded : IO Unit := do
  assertReference "../drake/examples/planar_gripper/BUILD.bazel"
    "Example should reference Drake's PlanarGripper Bazel target graph"
  assertReference "../drake/examples/planar_gripper/README.md"
    "Example should reference Drake's PlanarGripper README boundary"
  assertReference "../drake/examples/planar_gripper/gripper_brick.h"
    "Example should reference Drake's GripperBrick helper header"
  assertReference "../drake/examples/planar_gripper/brick_static_equilibrium_constraint.cc"
    "Example should reference Drake's brick static-equilibrium constraint"
  assertReference "../drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"
    "Example should reference Drake's brick static-equilibrium constraint declaration"
  assertReference "../drake/examples/planar_gripper/test/brick_static_equilibrium_constraint_test.cc"
    "Example should reference Drake's brick static-equilibrium constraint regression"
  assertReference "../drake/examples/planar_gripper/gripper_brick_planning_constraint_helper.h"
    "Example should reference Drake's planning constraint helper declaration"
  assertReference "../drake/examples/planar_gripper/test/gripper_brick_planning_constraint_helper_test.cc"
    "Example should reference Drake's planning constraint helper regression"
  assertReference "../drake/examples/planar_gripper/planar_gripper_common.h"
    "Example should reference Drake's planar gripper common declaration"
  assertReference "../drake/examples/planar_gripper/test/planar_gripper_common_test.cc"
    "Example should reference Drake's planar gripper common regression"
  assertReference "../drake/examples/planar_gripper/test/gripper_brick_test.cc"
    "Example should reference Drake's GripperBrickHelper regression"
  assertReference "../drake/examples/planar_gripper/planar_brick.sdf"
    "Example should reference Drake's planar brick SDF"
  assertReference "../drake/examples/planar_gripper/run_planar_gripper_trajectory_publisher.cc"
    "Example should reference Drake's trajectory publisher executable"
  assertReference "../drake/examples/planar_gripper/planar_gripper_lcm.cc"
    "Example should reference Drake's planar gripper LCM helpers"
  assertReference "../drake/examples/planar_gripper/planar_gripper_lcm.h"
    "Example should reference Drake's planar gripper LCM declarations"
  assertReference "../drake/examples/planar_gripper/planar_gripper_simulation.cc"
    "Example should reference Drake's planar gripper simulation executable"
  assertReference "../drake/examples/planar_gripper/planar_manipuland_lcm.cc"
    "Example should reference Drake's planar manipuland LCM implementation"
  assertReference "../drake/examples/planar_gripper/planar_manipuland_lcm.h"
    "Example should reference Drake's planar manipuland LCM declarations"
  assertReference "../drake/examples/planar_gripper/test/planar_manipuland_lcm_test.cc"
    "Example should reference Drake's planar manipuland LCM passthrough test"
  assertReference "../drake/examples/planar_gripper/test/planar_gripper_lcm_test.cc"
    "Example should reference Drake's planar gripper LCM passthrough test"
  assertReference "../drake/examples/planar_gripper/postures.txt"
    "Example should reference the posture keyframes played by the publisher"
  assertReference "../drake/examples/planar_gripper/planar_gripper.sdf"
    "Example should reference the controlled gripper SDF"
  assertReference "../drake/examples/planar_gripper/planar_gripper.xacro"
    "Example should reference the gripper xacro source"
  LeanTest.assertTrue (planarBrickModelUrl.contains "planar_brick.sdf")
    s!"Expected planar brick model URL to name planar_brick.sdf, got {planarBrickModelUrl}"

  LeanTest.assertEqual params.numPositions 9
  LeanTest.assertEqual params.numFingers 3
  LeanTest.assertEqual params.jointsPerFinger 2
  LeanTest.assertTrue (approx params.gripperOriginToBaseDistance 0.201 1.0e-12)
    s!"Expected Drake base weld radius 0.201, got {params.gripperOriginToBaseDistance}"
  LeanTest.assertTrue (approx params.link1Length 0.085 1.0e-12)
    s!"Expected Drake first-link length 0.085, got {params.link1Length}"
  LeanTest.assertTrue (approx params.pL2Fingertip.z (-0.0713) 1.0e-12)
    s!"Expected Drake fingertip offset z=-0.0713, got {params.pL2Fingertip.z}"
  LeanTest.assertTrue (approx params.fingerTipRadius 0.015 1.0e-12)
    s!"Expected Drake fingertip radius 0.015, got {params.fingerTipRadius}"
  LeanTest.assertTrue (approx params.brickMass 0.028 1.0e-12)
    s!"Expected Drake brick mass 0.028, got {params.brickMass}"
  LeanTest.assertTrue (approx params.brickSize.y 0.1 1.0e-12)
    s!"Expected Drake brick y size 0.1, got {params.brickSize.y}"
  LeanTest.assertTrue (approx params.brickSize.z 0.1 1.0e-12)
    s!"Expected Drake brick z size 0.1, got {params.brickSize.z}"

  LeanTest.assertEqual (fingerBasePositionIndex .finger1) 0
  LeanTest.assertEqual (fingerMidPositionIndex .finger1) 1
  LeanTest.assertEqual (fingerBasePositionIndex .finger2) 2
  LeanTest.assertEqual (fingerMidPositionIndex .finger2) 3
  LeanTest.assertEqual (fingerBasePositionIndex .finger3) 4
  LeanTest.assertEqual (fingerMidPositionIndex .finger3) 5
  LeanTest.assertEqual brickTranslateYPositionIndex 6
  LeanTest.assertEqual brickTranslateZPositionIndex 7
  LeanTest.assertEqual brickRevoluteXPositionIndex 8

  let q ← assertOk (PlanarGripperState.fromArray? drakeTestState.asArray)
    "round-tripping Drake test q"
  LeanTest.assertTrue (q.asArray == drakeTestState.asArray)
    "Planar gripper state should round-trip through the 9-position layout"
  LeanTest.assertTrue
    (approx (fingerLink2Orientation .finger1 drakeTestState)
      (fingerBaseAngle .finger1 + 0.1 + 0.3) 1.0e-12)
    "Finger 1 link-2 orientation should add base weld, base joint, and mid joint"
  LeanTest.assertTrue
    (approx (fingerLink2Orientation .finger2 drakeTestState)
      (fingerBaseAngle .finger2 + 0.3 - 0.4) 1.0e-12)
    "Finger 2 link-2 orientation should add base weld, base joint, and mid joint"
  LeanTest.assertTrue
    (approx (fingerLink2Orientation .finger3 drakeTestState)
      (fingerBaseAngle .finger3 - 1.2 + 0.5) 1.0e-12)
    "Finger 3 link-2 orientation should add base weld, base joint, and mid joint"

@[test]
def testPlanarGripperAssetCatalogRecordsBuildModelsExecutablesAndTests :
    IO Unit := do
  assertOk validatePlanarGripperExampleAssetCatalog?
    "planar gripper asset catalog validation"
  LeanTest.assertEqual planarGripperExampleAssets.size 26
  LeanTest.assertEqual requiredPlanarGripperExampleAssetPaths.size 26
  LeanTest.assertEqual planarGripperModelAssets.size 3
  LeanTest.assertEqual planarGripperTestAssets.size 6
  LeanTest.assertEqual planarGripperSimulationAssets.size 4
  LeanTest.assertEqual planarGripperTrajectoryAssets.size 3

  let build ← assertSome (findPlanarGripperExampleAsset? "BUILD.bazel")
    "BUILD.bazel asset"
  LeanTest.assertTrue (build.kind == PlanarGripperExampleAssetKind.metadata)
    "BUILD.bazel should be recorded as metadata"
  LeanTest.assertTrue
    (build.localDependencies.contains "planar_gripper_simulation.cc")
    "BUILD.bazel should depend on the simulation executable"
  LeanTest.assertTrue
    (build.localDependencies.contains "test/planar_gripper_common_test.cc")
    "BUILD.bazel should depend on the parser/common regression"

  let gripperSdf ← assertSome (findPlanarGripperExampleAsset? "planar_gripper.sdf")
    "planar_gripper.sdf asset"
  LeanTest.assertTrue (gripperSdf.kind == PlanarGripperExampleAssetKind.model)
    "planar_gripper.sdf should be a model asset"
  LeanTest.assertTrue gripperSdf.feedsModelsFilegroup
    "planar_gripper.sdf should feed Drake's models_filegroup"
  LeanTest.assertTrue gripperSdf.feedsSimulation
    "planar_gripper.sdf should feed planar_gripper_simulation"
  LeanTest.assertTrue gripperSdf.feedsTrajectoryPublisher
    "planar_gripper.sdf should feed the trajectory publisher"
  LeanTest.assertTrue
    (gripperSdf.localDependencies.contains "planar_gripper.xacro")
    "planar_gripper.sdf should retain the xacro source dependency"

  let brickSdf ← assertSome (findPlanarGripperExampleAsset? "planar_brick.sdf")
    "planar_brick.sdf asset"
  LeanTest.assertTrue brickSdf.feedsModelsFilegroup
    "planar_brick.sdf should feed Drake's models_filegroup"
  LeanTest.assertTrue brickSdf.feedsSimulation
    "planar_brick.sdf should feed planar_gripper_simulation"

  let simulation ← assertSome
    (findPlanarGripperExampleAsset? "planar_gripper_simulation.cc")
    "planar_gripper_simulation.cc asset"
  LeanTest.assertTrue simulation.feedsSimulation
    "planar_gripper_simulation.cc should feed the simulation boundary"
  LeanTest.assertTrue
    (simulation.localDependencies.contains "planar_gripper.sdf" &&
     simulation.localDependencies.contains "planar_brick.sdf" &&
     simulation.localDependencies.contains "postures.txt")
    "planar_gripper_simulation.cc should retain model and posture dependencies"

  let publisher ← assertSome
    (findPlanarGripperExampleAsset? "run_planar_gripper_trajectory_publisher.cc")
    "run_planar_gripper_trajectory_publisher.cc asset"
  LeanTest.assertTrue publisher.feedsTrajectoryPublisher
    "trajectory publisher source should feed the publisher boundary"
  LeanTest.assertTrue
    (publisher.localDependencies.contains "postures.txt" &&
     publisher.localDependencies.contains "planar_gripper_lcm.h")
    "trajectory publisher source should retain posture and LCM dependencies"

  let eqSource ← assertSome
    (findPlanarGripperExampleAsset? "brick_static_equilibrium_constraint.cc")
    "brick_static_equilibrium_constraint.cc asset"
  LeanTest.assertTrue
    (eqSource.localDependencies.contains
      "gripper_brick_planning_constraint_helper.h")
    "static-equilibrium source should depend on the planning helper declaration"

@[test]
def testPlanarGripperCommonReorderBoundaryMatchesDrakeRegression :
    IO Unit := do
  let reordered ← assertOk
    (reorderKeyframesForControlPlant? drakeCommonTestKeyframes
      drakeCommonTestJointRowIndexMap)
    "Drake planar_gripper_common ReorderKeyframesForPlant regression"

  LeanTest.assertEqual reordered.keyframes.size 6
  for i in [:controlPlantJointOrder.size] do
    let joint := controlPlantJointOrder[i]!
    let oldIndex ← assertOk (rowIndexFor? drakeCommonTestJointRowIndexMap joint)
      s!"original row index for {joint}"
    assertArrayNear reordered.keyframes[i]! drakeCommonTestKeyframes[oldIndex]!
      1.0e-12
      s!"Reordered keyframe row for {joint} should come from original row {oldIndex}"
    LeanTest.assertEqual reordered.rowIndexMap[i]!.jointName joint
    LeanTest.assertEqual reordered.rowIndexMap[i]!.rowIndex i

  assertError
    (reorderKeyframesForControlPlant?
      (constantKeyframeRows 7 4)
      drakeCommonTestJointRowIndexMap)
    "ReorderKeyframesForPlant should reject keyframe row count != row-map size"
  let badRowsAndMap :=
    drakeCommonTestJointRowIndexMap.push
      { jointName := "finger1_ExtraJoint", rowIndex := 6 }
  assertError
    (reorderKeyframesForControlPlant? (constantKeyframeRows 7 4) badRowsAndMap)
    "ReorderKeyframesForPlant should reject row count != planar gripper joint count"
  assertError
    (reorderKeyframesForControlPlant? drakeCommonTestKeyframes
      drakeCommonTestJointRowIndexMap 9)
    "ReorderKeyframesForPlant should reject a plant whose positions include the brick"

@[test]
def testPlanarContactGeometryAndFrictionConeUseFaceSemantics : IO Unit := do
  let tip : PlanarVec2 := { y := 0.2, z := -0.3 }
  assertArrayNear
    (PlanarBoxFace.posZ.contactPointFromFingerTip params.fingerTipRadius tip).toArray
    #[0.2, -0.315]
    1.0e-12
    "Positive-z face should shift the fingertip point inward by one radius"
  assertArrayNear
    (PlanarBoxFace.negZ.contactPointFromFingerTip params.fingerTipRadius tip).toArray
    #[0.2, -0.285]
    1.0e-12
    "Negative-z face should shift the fingertip point inward by one radius"
  assertArrayNear
    (PlanarBoxFace.posY.contactPointFromFingerTip params.fingerTipRadius tip).toArray
    #[0.185, -0.3]
    1.0e-12
    "Positive-y face should shift the fingertip point inward by one radius"
  assertArrayNear
    (PlanarBoxFace.negY.contactPointFromFingerTip params.fingerTipRadius tip).toArray
    #[0.215, -0.3]
    1.0e-12
    "Negative-y face should shift the fingertip point inward by one radius"

  LeanTest.assertTrue
    (PlanarBoxFace.negZ.inFrictionCone params.staticFriction { y := 0.0, z := 1.0 })
    "A positive normal force should satisfy the negative-z face cone"
  LeanTest.assertTrue
    (PlanarBoxFace.posZ.inFrictionCone params.staticFriction { y := 0.0, z := -1.0 })
    "A negative z force should satisfy the positive-z face cone"
  LeanTest.assertTrue
    (PlanarBoxFace.posY.inFrictionCone params.staticFriction { y := -1.0, z := 0.2 })
    "A force into the positive-y face with bounded tangent should satisfy the cone"
  LeanTest.assertFalse
    (PlanarBoxFace.posY.inFrictionCone params.staticFriction { y := 1.0, z := 0.0 })
    "A force pulling away from the positive-y face should violate the cone"
  LeanTest.assertFalse
    (allForcesInFrictionCone params drakeStaticEquilibriumTestContacts)
    "The Drake static-equilibrium regression forces are not a feasible friction-cone witness"

@[test]
def testPlanningConstraintHelpersExposeContactAndNoSlidingPrimitives :
    IO Unit := do
  let depth := 2.0e-3
  let negYTip : PlanarVec2 :=
    {
      y := -params.brickSize.y / 2.0 + depth - params.fingerTipRadius
      z := 0.0
    }
  LeanTest.assertTrue
    (approx (fingerTipContactResidualFromTip params .negY depth negYTip)
      0.0 1.0e-12)
    "Neg-Y fingertip contact residual should match Drake's y + radius = -half_y + depth equation"
  LeanTest.assertTrue
    (fingerTipInShrunkFaceRegionFromTip params .negY 0.8 depth negYTip)
    "A centered fingertip should satisfy the shrunk neg-Y face contact region"
  let outsideNegYTip := { negYTip with z := 0.8 * params.brickSize.z }
  LeanTest.assertFalse
    (fingerTipInShrunkFaceRegionFromTip params .negY 0.8 depth outsideNegYTip)
    "A fingertip outside the shrunk tangent interval should violate the contact region"

  let negZFrom : PlanarVec2 := { y := 0.02, z := -0.05 }
  let rollingTheta := 0.1 * 3.14159265358979323846
  let negZTo : PlanarVec2 :=
    { negZFrom with y := negZFrom.y - params.fingerTipRadius * rollingTheta }
  LeanTest.assertTrue
    (approx (noSlidingResidualNegZ params 0.0 rollingTheta negZFrom negZTo)
      0.0 1.0e-12)
    "Neg-Z no-sliding residual should encode Drake's tangent displacement equals -radius * rolling angle check"

@[test]
def testStaticEquilibriumResidualMatchesDrakeFormula : IO Unit := do
  let residual ← assertOk
    (staticEquilibriumResidual? params drakeTestState drakeStaticEquilibriumTestContacts)
    "Drake test static equilibrium residual"
  let mg := params.brickMass * params.gravity
  let expectedForceY := 2.5 + (-0.2) - mg * Float.sin drakeTestState.brickTheta
  let expectedForceZ := -0.4 + (-0.6) - mg * Float.cos drakeTestState.brickTheta
  let wrench0 := contactWrench params drakeTestState drakeStaticEquilibriumTestContacts[0]!
  let wrench1 := contactWrench params drakeTestState drakeStaticEquilibriumTestContacts[1]!
  let expectedTorque := wrench0.torqueX + wrench1.torqueX
  assertArrayNear residual #[expectedForceY, expectedForceZ, expectedTorque] 1.0e-12
    "Residual should be body-frame gravity plus contact forces and x-axis torque"
  LeanTest.assertTrue (Float.isFinite expectedTorque)
    s!"Expected torque should be finite, got {expectedTorque}"
  LeanTest.assertTrue (Float.abs expectedTorque > 1.0e-9)
    "The Drake test contacts should produce a nontrivial contact moment"

  let symmetricResidual ← assertOk (symmetricSupportResidual? params)
    "symmetric static support residual"
  assertArrayNear symmetricResidual #[0.0, 0.0, 0.0] 1.0e-12
    "Two symmetric bottom supports should exactly balance the brick in body coordinates"

@[test]
def testBrickStaticEquilibriumConstraintBoundaryMatchesDrakeRegression :
    IO Unit := do
  let boundary ← assertOk
    (buildBrickStaticEquilibriumConstraint? params drakeTestState
      drakeStaticEquilibriumTestContacts)
    "brick static equilibrium constraint boundary"
  assertOk boundary.validate? "brick static equilibrium constraint validation"
  LeanTest.assertEqual boundary.numOutputs 3
  LeanTest.assertEqual boundary.numVars 13
  assertArrayNear boundary.lowerBound #[0.0, 0.0, 0.0] 1.0e-12
    "Brick static equilibrium lower bound should be zero"
  assertArrayNear boundary.upperBound #[0.0, 0.0, 0.0] 1.0e-12
    "Brick static equilibrium upper bound should be zero"
  let direct ← assertOk
    (staticEquilibriumResidual? params drakeTestState
      drakeStaticEquilibriumTestContacts)
    "direct static equilibrium residual"
  assertArrayNear boundary.residual direct 1.0e-12
    "Constraint boundary residual should reuse the exact planar static-equilibrium primitive"
  LeanTest.assertTrue (boundary.graph.containsMoveKind .localSchurBlock)
    "Constraint boundary should expose the static-equilibrium local Schur block"

@[test]
def testTrajectoryPublisherBoundaryMatchesDrakeExecutable : IO Unit := do
  let publisher ← assertOk buildTrajectoryPublisher?
    "planar gripper trajectory publisher"
  assertOk publisher.config.validate? "trajectory publisher config"

  LeanTest.assertEqual publisher.config.gripperModelUrl planarGripperModelUrl
  LeanTest.assertEqual publisher.config.keyframePath postureKeyframePath
  LeanTest.assertTrue (approx publisher.config.keyframeDt 0.1 1.0e-12)
    s!"Expected Drake --keyframe_dt default 0.1, got {publisher.config.keyframeDt}"
  LeanTest.assertTrue
    (publisher.config.interpolation == TrajectoryInterpolation.cubicShapePreserving)
    "Publisher should use PiecewisePolynomial::CubicShapePreserving"
  LeanTest.assertEqual publisher.config.derivativeOrder 1
  LeanTest.assertEqual publisher.config.numFingers 3
  LeanTest.assertEqual publisher.config.numJoints 6
  LeanTest.assertEqual publisher.config.stateOutputDim 12
  LeanTest.assertEqual publisher.config.torqueDim 6
  LeanTest.assertEqual publisher.config.statusChannel "PLANAR_GRIPPER_STATUS"
  LeanTest.assertEqual publisher.config.commandChannel "PLANAR_GRIPPER_COMMAND"
  LeanTest.assertTrue (approx publisher.config.statusPeriod 0.010 1.0e-12)
    s!"Expected Drake gripper LCM status period 0.010, got {publisher.config.statusPeriod}"
  LeanTest.assertTrue publisher.config.waitForFirstStatus
    "Publisher should wait for the first lcmt_planar_gripper_status message"
  LeanTest.assertEqual publisher.config.timeSource
    "lcmt_planar_gripper_status.utime * 1e-6"
  LeanTest.assertEqual publisher.config.commandEncoderStateInputPort "state"
  LeanTest.assertEqual publisher.config.commandEncoderTorqueInputPort "torque"
  LeanTest.assertEqual publisher.config.commandEncoderOutputPort "lcmt_gripper_command"
  LeanTest.assertEqual publisher.config.statusDecoderInputPort
    "lcmt_planar_gripper_status"
  LeanTest.assertEqual publisher.assetCatalog.size planarGripperExampleAssets.size
  LeanTest.assertEqual publisher.assetCatalog.size 26

  LeanTest.assertEqual publisher.header.size 9
  LeanTest.assertEqual publisher.postureRows 41
  LeanTest.assertTrue (approx (publisher.config.keyframeTime 0) 0.0 1.0e-12)
    "The first keyframe should start at t=0"
  LeanTest.assertTrue
    (approx (publisher.config.keyframeTime (publisher.postureRows - 1)) 4.0 1.0e-12)
    s!"Expected final keyframe time 4.0, got {publisher.config.keyframeTime (publisher.postureRows - 1)}"
  LeanTest.assertTrue (approx publisher.config.trajectoryDuration 4.0 1.0e-12)
    s!"Expected trajectory duration 4.0, got {publisher.config.trajectoryDuration}"

  assertArrayNear publisher.firstParsedFingerKeyframe
    #[-0.277611, 0.281503, 0.947974, -0.0613293, 0.201025, -1.21373]
    1.0e-12
    "ParseKeyframes should first extract finger joints in Drake's parse order"
  assertArrayNear publisher.firstControlPlantKeyframe
    #[-0.277611, -0.0613293, 0.281503, 0.201025, 0.947974, -1.21373]
    1.0e-12
    "ReorderKeyframesForPlant should arrange rows in control plant velocity order"
  assertArrayNear publisher.firstBrickInitialPose
    #[0.00731802, -0.0197339, -0.454859]
    1.0e-12
    "ParseKeyframes should expose the first brick pose for simulation initialization"
  LeanTest.assertEqual publisher.zeroTorques.size 6
  LeanTest.assertTrue
    (publisher.zeroTorques.all (fun tau => approx tau 0.0 1.0e-12))
    s!"Trajectory publisher should hold commanded torques at zero, got {publisher.zeroTorques}"

  LeanTest.assertEqual publisher.graph.vertices.size 8
  LeanTest.assertEqual publisher.graph.moves.size 5
  LeanTest.assertTrue (publisher.graph.containsMoveKind .localSchurBlock)
    "Publisher graph should expose parse/reorder and trajectory-source local blocks"
  LeanTest.assertTrue (publisher.graph.containsMoveKind .freezeControl)
    "Publisher graph should expose zero torques as a frozen control"
  LeanTest.assertTrue (publisher.graph.containsMoveKind .checkpointBoundary)
    "Publisher graph should expose waiting for the first status message"
  LeanTest.assertTrue (publisher.graph.containsMoveKind .clockedUpdate)
    "Publisher graph should expose status-clocked command publishing"
  LeanTest.assertTrue
    (publisher.graph.moves.any (fun move =>
      move.label.contains "CubicShapePreserving"))
    "Publisher graph should name the cubic shape-preserving trajectory source"
  LeanTest.assertTrue
    (publisher.graph.moves.any (fun move =>
      move.label.contains "force-publish PLANAR_GRIPPER_COMMAND"))
    "Publisher graph should name the forced LCM command publish"

@[test]
def testPlanarManipulandStatusMessageRoundTripsLikeDrakeLcmSystem : IO Unit := do
  let state ← assertOk sampleManipulandStatus.stateVector?
    "sample manipuland status state vector"
  assertArrayNear state #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 1.0e-12
    "PlanarManipulandStatusDecoder should expose position, theta, velocity, and thetadot"

  let roundTrip ← assertOk sampleManipulandStatus.passthrough?
    "sample manipuland status passthrough"
  let roundTripState ← assertOk roundTrip.stateVector?
    "sample manipuland status passthrough state vector"
  assertArrayNear roundTripState state 1.0e-12
    "PlanarManipulandStatusEncoder should preserve the decoded state"
  LeanTest.assertTrue (approx roundTrip.utime sampleManipulandStatus.utime 1.0e-12)
    s!"Expected passthrough utime {sampleManipulandStatus.utime}, got {roundTrip.utime}"

  let timed ← assertOk
    (PlanarManipulandStatusMessage.fromState? 0.25
      #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    "timed manipuland status from state"
  LeanTest.assertTrue (approx timed.utime 250000.0 1.0e-12)
    s!"Expected 0.25 s to encode as 250000 us, got {timed.utime}"

@[test]
def testPlanarGripperSimulationBoundaryUsesFullPlantAndReusablePrimitives :
    IO Unit := do
  let sim ← assertOk buildPlanarGripperSimulation?
    "planar gripper simulation"
  assertOk sim.validate? "planar gripper simulation validation"

  LeanTest.assertTrue (approx sim.config.targetRealtimeRate 1.0 1.0e-12)
    s!"Expected target realtime rate 1.0, got {sim.config.targetRealtimeRate}"
  LeanTest.assertTrue (approx sim.config.simulationTime 4.5 1.0e-12)
    s!"Expected simulation time 4.5, got {sim.config.simulationTime}"
  LeanTest.assertTrue (approx sim.config.timeStep 0.001 1.0e-12)
    s!"Expected time step 0.001, got {sim.config.timeStep}"
  LeanTest.assertTrue (approx sim.config.penetrationAllowance 0.001 1.0e-12)
    s!"Expected penetration allowance 0.001, got {sim.config.penetrationAllowance}"
  LeanTest.assertTrue (approx sim.config.floorStaticFriction 0.5 1.0e-12)
    s!"Expected floor static friction 0.5, got {sim.config.floorStaticFriction}"
  LeanTest.assertTrue (approx sim.config.floorKineticFriction 0.5 1.0e-12)
    s!"Expected floor kinetic friction 0.5, got {sim.config.floorKineticFriction}"
  LeanTest.assertTrue (approx sim.config.brickFloorPenetration 1.0e-5 1.0e-12)
    s!"Expected brick-floor penetration 1e-5, got {sim.config.brickFloorPenetration}"
  LeanTest.assertTrue (sim.config.orientation == .vertical)
    "Default simulation should use Drake's vertical setup"
  LeanTest.assertFalse sim.config.visualizeContacts
    "Default simulation should leave contact visualization disabled"
  LeanTest.assertTrue (sim.config.controlMode == .positionControl)
    "Default simulation should use position control"
  LeanTest.assertTrue sim.config.controlMode.usesInverseDynamicsController
    "Position control should instantiate an inverse-dynamics controller"
  LeanTest.assertFalse sim.config.controlMode.usesDirectTorqueInput
    "Position control should not wire direct commanded torques"
  LeanTest.assertTrue (sim.config.plantConfig.contactApproximation == .sap)
    "Simulation should preserve Drake's SAP contact approximation setting"

  assertArrayNear sim.config.positionControlKp
    #[1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0]
    1.0e-12
    "Inverse-dynamics Kp gains should match planar_gripper_simulation.cc"
  assertArrayNear sim.config.positionControlKd
    #[500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    1.0e-12
    "Inverse-dynamics Kd gains should match planar_gripper_simulation.cc"
  assertArrayNear sim.config.positionControlKi
    #[500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    1.0e-12
    "Inverse-dynamics Ki gains should match planar_gripper_simulation.cc"

  LeanTest.assertEqual sim.parsedPlant.modelUris
    #[planarGripperModelUrl, planarBrickModelUrl]
  LeanTest.assertEqual sim.parsedPlant.numActuators 6
  LeanTest.assertEqual sim.parsedPlant.numJoints 12
  LeanTest.assertEqual sim.parsedPlant.numBodies 17
  LeanTest.assertEqual sim.parsedPlant.modelInstances.size 2
  LeanTest.assertEqual sim.parsedPlant.modelInstances[0]!.name "planar_gripper"
  LeanTest.assertEqual sim.parsedPlant.modelInstances[0]!.numPositions 6
  LeanTest.assertEqual sim.parsedPlant.modelInstances[0]!.numVelocities 6
  LeanTest.assertEqual sim.parsedPlant.modelInstances[1]!.name "brick"
  LeanTest.assertEqual sim.parsedPlant.modelInstances[1]!.numPositions 3
  LeanTest.assertEqual sim.parsedPlant.modelInstances[1]!.numVelocities 3

  LeanTest.assertEqual sim.controlPlant.numPositions 6
  LeanTest.assertEqual sim.controlPlant.numVelocities 6
  LeanTest.assertEqual sim.controlPlant.numActuatedDofs 6
  LeanTest.assertEqual sim.plantStep.model.numPositions 9
  LeanTest.assertEqual sim.plantStep.model.numVelocities 9
  LeanTest.assertEqual sim.plantStep.model.numActuatedDofs 6
  LeanTest.assertEqual sim.plantStep.q0.size 9
  LeanTest.assertEqual sim.plantStep.v0.size 9
  LeanTest.assertEqual sim.plantStep.actuation.size 6
  LeanTest.assertTrue (sim.plantStep.hasContactEnvironment)
    "Full plant advance should keep the floor/contact environment boundary"
  LeanTest.assertTrue (approx sim.plantStep.t1 4.5 1.0e-12)
    s!"Expected simulator AdvanceTo time 4.5, got {sim.plantStep.t1}"
  assertArrayNear sim.plantStep.q0
    (sim.initialGripperPosition ++ sim.initialBrickPose)
    1.0e-12
    "Full plant initial q should concatenate the control-plant keyframe and brick pose"
  assertArrayNear sim.initialState.q sim.plantStep.q0 1.0e-12
    "Simulation initial state q should be the plant step q0 consumed by the primitive provider"
  assertArrayNear sim.initialState.v sim.plantStep.v0 1.0e-12
    "Simulation initial state v should be the plant step v0 consumed by the primitive provider"
  LeanTest.assertEqual sim.assetCatalog.size planarGripperExampleAssets.size
  LeanTest.assertEqual sim.assetCatalog.size 26
  LeanTest.assertTrue
    (sim.plantStep.v0.all (fun v => approx v 0.0 1.0e-12))
    s!"Full plant initial velocities should be zero, got {sim.plantStep.v0}"
  LeanTest.assertTrue
    (sim.plantStep.actuation.all (fun u => approx u 0.0 1.0e-12))
    s!"Full plant initial actuation should be zero, got {sim.plantStep.actuation}"
  assertOk sim.primitivePlant.validate?
    "planar gripper plant primitive physics wrapper"
  LeanTest.assertEqual sim.primitivePlant.intervalVertex 7037
    "Primitive plant wrapper should target the simulator interval vertex"
  assertArrayNear sim.primitivePlant.step.v0 sim.plantStep.v0 1.0e-12
    "Primitive plant wrapper should carry the same plant step velocity"
  assertArrayNear sim.primitivePlant.primitives.qdot sim.plantStep.v0 1.0e-12
    "Primitive plant wrapper should implement the plant step velocity through qdot"
  assertOk sim.fullPhysics.equation.validate?
    "planar gripper primitive full-physics equation"
  LeanTest.assertTrue sim.controllerOutput?.isSome
    "Position control should expose the local PID primitive output"
  match sim.controllerOutput? with
  | some output =>
      assertArrayNear output.feedback (Array.replicate 6 0.0) 1.0e-12
        "At the initial desired posture, the PID feedback should be zero"
  | none =>
      LeanTest.fail "Position control should produce a controller output"
  LeanTest.assertEqual sim.fullPhysics.support.candidates.size 1
    "Primitive simulation should expose the dynamically computed brick-floor candidate"
  LeanTest.assertEqual sim.fullPhysics.support.selectedLocalIndices.size 1
    "Primitive simulation should select the penetrating brick-floor support"
  LeanTest.assertEqual sim.fullPhysics.contactForces.size 1
    "Primitive simulation should produce one scalar floor-contact force bundle"
  LeanTest.assertTrue (sim.fullPhysics.supportMove.exactness == .controlledApproximation)
    "Threshold-selected floor support should remain a fixed-trace approximation"
  LeanTest.assertTrue (sim.fullPhysics.move.exactness == .exact)
    "The selected-support mass-matrix solve should be exact"
  LeanTest.assertEqual sim.fullPhysics.equation.massMatrix.size 9
    "Vertical primitive full physics should use the nine-velocity plant dimension"
  LeanTest.assertTrue
    (approx ((sim.fullPhysics.equation.massMatrix[7]!).getD 7 0.0)
      params.brickMass 1.0e-12)
    "Brick z mass should appear on the support coordinate"
  assertArrayNear sim.fullPhysics.equation.qdot sim.plantStep.v0 1.0e-12
    "Primitive full physics should use the plant initial velocity"
  LeanTest.assertTrue
    (approx (sim.fullPhysics.generalizedContactForce.getD 7 0.0)
      (params.brickMass * params.gravity) 1.0e-12)
    "Floor contact should balance the brick weight on the vertical support coordinate"
  LeanTest.assertTrue
    (approx (sim.fullPhysics.equation.biasForces.getD 7 0.0)
      (params.brickMass * params.gravity) 1.0e-12)
    "Gravity bias should load the same support coordinate"
  assertArrayNear sim.fullPhysics.derivative.vdot
    (Array.replicate 9 0.0) 1.0e-12
    "Supported initial state should have zero primitive acceleration"

  LeanTest.assertEqual sim.floor.collisionName "FloorCollisionGeometry"
  LeanTest.assertEqual sim.floor.visualName "FloorVisualGeometry"
  LeanTest.assertTrue (approx sim.floor.centerX (-0.03559) 1.0e-12)
    s!"Expected Drake AddFloor center x -0.03559, got {sim.floor.centerX}"
  match sim.plantStep.ground? with
  | some ground =>
      LeanTest.assertEqual ground.collisionName "FloorCollisionGeometry"
      LeanTest.assertTrue
        (approx ground.friction.staticFriction 0.5 1.0e-12 &&
         approx ground.friction.dynamicFriction 0.5 1.0e-12)
        s!"Expected 0.5/0.5 floor friction, got {repr ground.friction}"
  | none =>
      LeanTest.fail "Simulation step should include a floor contact environment"

  assertArrayNear sim.config.orientation.gravityVector #[0.0, 0.0, -9.81]
    1.0e-12
    "Vertical setup should keep gravity along world -z"
  LeanTest.assertTrue sim.config.orientation.fixesBrickBaseFrame
    "Vertical setup should fix the brick x-coordinate base frame"
  LeanTest.assertFalse sim.config.orientation.addsBrickTranslateXJoint
    "Vertical setup should not add the horizontal translate-x joint"

  LeanTest.assertTrue (approx sim.manipulandStatusPeriod 0.010 1.0e-12)
    s!"Expected manipuland status period 0.010, got {sim.manipulandStatusPeriod}"
  LeanTest.assertEqual sim.forceSensorJointNames
    #["finger1_sensor_weldjoint", "finger2_sensor_weldjoint", "finger3_sensor_weldjoint"]
  LeanTest.assertEqual sim.forceSensorOutputDim 6
  let sampleState ← assertOk sim.sampleManipulandRoundTrip.stateVector?
    "simulation sample manipuland passthrough state"
  assertArrayNear sampleState #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 1.0e-12
    "Simulation boundary should carry the manipuland LCM passthrough regression"

  LeanTest.assertEqual sim.graph.vertices.size 11
  LeanTest.assertEqual sim.graph.moves.size 10
  LeanTest.assertTrue (sim.graph.containsMoveKind .localSchurBlock)
    "Simulation graph should expose parser, floor, controller, and force sensor blocks"
  LeanTest.assertTrue (sim.graph.containsMoveKind .clockedUpdate)
    "Simulation graph should expose LCM subscription and publishing updates"
  LeanTest.assertTrue (sim.graph.containsMoveKind .markMarginalize)
    "Simulation graph should expose runtime floor support selection"
  LeanTest.assertTrue (sim.graph.containsMoveKind .intervalAdjoint)
    "Simulation graph should expose the primitive full-physics interval solve"
  LeanTest.assertTrue
    (sim.graph.moves.any (fun move => move.label.contains "InverseDynamicsController"))
    "Position-control graph should name the inverse-dynamics controller"
  LeanTest.assertTrue
    (sim.graph.moves.any (fun move => move.label.contains "ForceSensorEvaluator"))
    "Simulation graph should name the force sensor evaluator"
  LeanTest.assertTrue
    (sim.graph.moves.any (fun move =>
      move.label.contains "PlanarManipulandStatusDecoder"))
    "Simulation graph should name the manipuland status decoder boundary"

@[test]
def testPlanarGripperPrimitiveProviderRecomputesHorizontalContactFromState :
    IO Unit := do
  let config : PlanarGripperSimulationConfig :=
    {
      orientation := .horizontal
      controlMode := .torqueControl
    }
  let state ← assertOk (PlanarGripperSimulationState.initial? config)
    "horizontal initial primitive-provider state"
  let provider := planarGripperSimulationPrimitiveProvider config
    "horizontal state-dependent contact provider"

  let initialSupport ← assertOk (provider.supportAt? state)
    "initial horizontal contact support"
  LeanTest.assertEqual initialSupport.candidates.size 1
  LeanTest.assertEqual initialSupport.selectedLocalIndices.size 1
  let initialCandidate := initialSupport.candidates[0]!
  LeanTest.assertTrue
    (approx initialCandidate.signedDistance (-config.floor.penetration) 1.0e-12)
    s!"Initial horizontal signed distance should be -penetration, got {initialCandidate.signedDistance}"
  LeanTest.assertTrue
    (approx (initialCandidate.point_W.getD 0 0.0) config.floor.sphereTipXOffset 1.0e-12)
    s!"Initial horizontal contact point should use the sphere tip x offset, got {initialCandidate.point_W}"
  LeanTest.assertTrue (initialCandidate.mode == .sticking)
    "Stationary penetrating horizontal contact should classify as sticking"

  let separatedQ :=
    state.q.set! planarGripperSimulationHorizontalPositionIndex
      (config.penetrationAllowance + config.floor.penetration + 0.01)
  let separatedState : PlanarGripperSimulationState :=
    { state with q := separatedQ }
  let separatedSupport ← assertOk (provider.supportAt? separatedState)
    "separated horizontal contact support"
  LeanTest.assertEqual separatedSupport.candidates.size 1
  LeanTest.assertEqual separatedSupport.selectedLocalIndices.size 0
  let separatedCandidate := separatedSupport.candidates[0]!
  LeanTest.assertTrue
    (separatedCandidate.signedDistance > config.penetrationAllowance)
    s!"Separated state should move the candidate outside threshold, got {separatedCandidate.signedDistance}"
  LeanTest.assertTrue (separatedCandidate.mode == .separated)
    "Separated candidate should classify as separated"
  let separatedSolve ← assertOk (provider.solveAt? separatedState 7037)
    "separated horizontal primitive solve"
  LeanTest.assertEqual separatedSolve.contactForces.size 0
  LeanTest.assertTrue
    (approx
      (separatedSolve.derivative.vdot.getD planarGripperSimulationHorizontalPositionIndex 0.0)
      (-params.gravity) 1.0e-12)
    s!"Without selected floor support, horizontal brick acceleration should be -g, got {separatedSolve.derivative.vdot}"

  let closingV :=
    state.v.set! planarGripperSimulationHorizontalPositionIndex (-0.2)
  let closingState : PlanarGripperSimulationState :=
    { state with v := closingV }
  let closingSupport ← assertOk (provider.supportAt? closingState)
    "closing horizontal contact support"
  LeanTest.assertEqual closingSupport.selectedLocalIndices.size 1
  LeanTest.assertTrue (closingSupport.candidates[0]!.mode == .impacting)
    "Negative normal velocity should classify the retained floor candidate as impacting"

@[test]
def testPlanarGripperSimulationHorizontalTorqueVariantChangesPlantBoundary :
    IO Unit := do
  let config : PlanarGripperSimulationConfig :=
    {
      orientation := .horizontal
      controlMode := .torqueControl
      visualizeContacts := true
    }
  let sim ← assertOk (buildPlanarGripperSimulation? config)
    "horizontal torque planar gripper simulation"
  assertOk sim.validate? "horizontal torque planar gripper simulation validation"

  LeanTest.assertTrue sim.config.visualizeContacts
    "Variant should preserve the visualize_contacts flag"
  LeanTest.assertTrue (sim.config.orientation == .horizontal)
    "Variant should use the horizontal setup"
  LeanTest.assertTrue (sim.config.controlMode == .torqueControl)
    "Variant should use torque control"
  LeanTest.assertTrue sim.config.controlMode.usesDirectTorqueInput
    "Torque control should wire command torques directly into plant actuation"
  LeanTest.assertFalse sim.config.controlMode.usesInverseDynamicsController
    "Torque control should not instantiate the inverse-dynamics controller"
  assertArrayNear sim.config.orientation.gravityVector #[-9.81, 0.0, 0.0]
    1.0e-12
    "Horizontal setup should rotate gravity onto world -x"
  LeanTest.assertFalse sim.config.orientation.fixesBrickBaseFrame
    "Horizontal setup should not fix the brick x-coordinate base frame"
  LeanTest.assertTrue sim.config.orientation.addsBrickTranslateXJoint
    "Horizontal setup should add the extra brick translate-x joint"

  LeanTest.assertEqual sim.plantStep.model.numPositions 10
  LeanTest.assertEqual sim.plantStep.model.numVelocities 10
  LeanTest.assertEqual sim.plantStep.q0.size 10
  LeanTest.assertEqual sim.plantStep.v0.size 10
  LeanTest.assertTrue (approx sim.plantStep.q0[9]! 0.0 1.0e-12)
    s!"Expected horizontal extra translate-x position 0, got {sim.plantStep.q0[9]!}"
  LeanTest.assertTrue sim.controllerOutput?.isNone
    "Torque-control variant should not instantiate the PID primitive"
  assertOk sim.primitivePlant.validate?
    "horizontal primitive plant wrapper"
  LeanTest.assertEqual sim.primitivePlant.primitives.velocityDim 10
    "Horizontal primitive plant wrapper should expose the full plant velocity dimension"
  assertOk sim.fullPhysics.equation.validate?
    "horizontal primitive full-physics equation"
  LeanTest.assertEqual sim.fullPhysics.equation.massMatrix.size 10
    "Horizontal primitive full physics should use the extra brick x velocity"
  LeanTest.assertTrue
    (approx ((sim.fullPhysics.equation.massMatrix[9]!).getD 9 0.0)
      params.brickMass 1.0e-12)
    "Brick x mass should appear on the horizontal support coordinate"
  LeanTest.assertTrue
    (approx (sim.fullPhysics.generalizedContactForce.getD 9 0.0)
      (params.brickMass * params.gravity) 1.0e-12)
    "Horizontal floor contact should balance gravity on the extra x coordinate"
  LeanTest.assertTrue
    (approx (sim.fullPhysics.equation.biasForces.getD 9 0.0)
      (params.brickMass * params.gravity) 1.0e-12)
    "Horizontal gravity bias should load the extra x coordinate"
  assertArrayNear sim.fullPhysics.derivative.vdot
    (Array.replicate 10 0.0) 1.0e-12
    "Supported horizontal initial state should have zero primitive acceleration"
  LeanTest.assertTrue
    (sim.graph.moves.any (fun move =>
      move.label.contains "direct torque command_decoder"))
    "Torque-control graph should name the direct torque wiring"

@[test]
def testEndToEndConstraintGraphUsesLocalSchurBlock : IO Unit := do
  let result ← assertOk (buildEndToEnd? params)
    "planar gripper end-to-end"
  LeanTest.assertEqual result.references.size drakeReferences.size
  LeanTest.assertEqual result.assetCatalog.size planarGripperExampleAssets.size
  LeanTest.assertEqual result.assetCatalog.size 26
  LeanTest.assertEqual result.drakeTestResidual.size 3
  LeanTest.assertEqual result.symmetricResidual.size 3
  assertArrayNear result.symmetricResidual #[0.0, 0.0, 0.0] 1.0e-12
    "End-to-end result should carry the balanced support residual"

  LeanTest.assertEqual result.graph.vertices.size 3
  LeanTest.assertEqual result.graph.moves.size 1
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "Static equilibrium should be exposed as a local Schur block, not a toy physics callback"
  LeanTest.assertEqual result.graph.moves[0]!.targets #[7002]
  LeanTest.assertEqual result.graph.moves[0]!.reads #[7000, 7001]
  LeanTest.assertEqual result.graph.moves[0]!.writes #[7000]
  LeanTest.assertTrue (result.graph.moves[0]!.exactness == MoveExactness.exact)
    "The planar static-equilibrium block is an exact local elimination primitive"
  LeanTest.assertEqual result.trajectoryPublisher.config.commandChannel
    "PLANAR_GRIPPER_COMMAND"
  LeanTest.assertEqual result.trajectoryPublisher.assetCatalog.size 26
  LeanTest.assertEqual result.trajectoryPublisher.firstControlPlantKeyframe.size 6
  LeanTest.assertTrue (result.trajectoryPublisher.graph.containsMoveKind .clockedUpdate)
    "End-to-end result should retain the trajectory publisher runtime boundary"
  LeanTest.assertEqual result.simulation.parsedPlant.modelUris
    #[planarGripperModelUrl, planarBrickModelUrl]
  LeanTest.assertEqual result.simulation.plantStep.model.numPositions 9
  LeanTest.assertTrue result.simulation.plantStep.hasContactEnvironment
    "End-to-end result should retain the full simulation floor/contact primitive"
  LeanTest.assertEqual result.simulation.fullPhysics.support.selectedLocalIndices.size 1
    "End-to-end result should retain the primitive floor support selection"
  LeanTest.assertEqual result.simulation.assetCatalog.size 26
  LeanTest.assertTrue (result.simulation.graph.containsMoveKind .intervalAdjoint)
    "End-to-end result should retain the primitive full-physics interval"

end Tests.EventSkeletonPlanarGripperExample
