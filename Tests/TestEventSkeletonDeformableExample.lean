import LeanTest
import Tyr.EventSkeleton.Examples.Deformable

namespace Tests.EventSkeletonDeformableExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Deformable

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def maxAbsDiff (xs ys : Array Float) : Float := Id.run do
  let n := Nat.max xs.size ys.size
  let mut acc := 0.0
  for i in [:n] do
    let d := Float.abs (xs.getD i 0.0 - ys.getD i 0.0)
    if d > acc then
      acc := d
  return acc

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertSome {α : Type} (value : Option α) (label : String) : IO α := do
  match value with
  | some x => pure x
  | none => LeanTest.fail s!"{label}: expected some, got none"

@[test]
def testDrakeReferencesAndPrimitivePortsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/BUILD.bazel"))
    "Deformable example should reference Drake's Bazel targets and data deps"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/point_source_force_field.cc"))
    "Deformable example should reference Drake's point-source force field"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/deformable_common.cc"))
    "Deformable example should reference Drake's shared deformable registration helper"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/deformable_common.h"))
    "Deformable example should reference Drake's shared deformable declarations"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/models/deformable_torus.sdf"))
    "Deformable example should reference the torus SDF material parameters"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/models/bubbles.sdf"))
    "Deformable example should reference the bubble SDF material and contact parameters"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/models/deformable_teddy.sdf"))
    "Deformable example should reference the teddy SDF material and contact parameters"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/models/simple_gripper.sdf"))
    "Deformable example should reference the simple gripper SDF joints and hydroelastic pads"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/test/point_source_force_field_test.cc"))
    "Deformable example should reference Drake's force-field test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/parallel_gripper_controller.cc"))
    "Deformable example should reference Drake's parallel gripper controller"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/parallel_gripper_controller.h"))
    "Deformable example should reference Drake's parallel gripper controller declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/suction_cup_controller.cc"))
    "Deformable example should reference Drake's suction controller"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/suction_cup_controller.h"))
    "Deformable example should reference Drake's suction controller declaration"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/multibody/deformable/deformable_subdivision.cc"))
    "Deformable example should reference Drake's subdivision demo"
  LeanTest.assertEqual pointSourceForceField.inputPortName
    "maximum force density magnitude in N/m^3"
    "Point-source field should expose Drake's force-density input port"
  LeanTest.assertEqual pointSourceForceField.cacheEntryName
    "point source of the force field"
    "Point-source field should expose Drake's cached source-position entry"
  LeanTest.assertTrue (approx torusDeformableConfig.youngsModulus 3.0e4 1.0e-12)
    s!"Torus Young's modulus should match deformable_torus.sdf, got {torusDeformableConfig.youngsModulus}"
  LeanTest.assertTrue (approx torusDeformableConfig.poissonsRatio 0.4 1.0e-12)
    s!"Torus Poisson ratio should match deformable_torus.sdf, got {torusDeformableConfig.poissonsRatio}"
  LeanTest.assertTrue (approx torusDeformableConfig.massDensity 1.0e3 1.0e-12)
    s!"Torus mass density should match deformable_torus.sdf, got {torusDeformableConfig.massDensity}"
  LeanTest.assertTrue (torusDeformableConfig.materialModel == DeformableMaterialModel.neohookean)
    "Torus material model should record Drake's neohookean deformable property"

@[test]
def testDeformableAssetCatalogRecordsModelAndMeshClosure : IO Unit := do
  assertOk validateDeformableExampleAssetCatalog?
    "deformable example asset catalog"
  LeanTest.assertEqual deformableExampleAssets.size 21
    "Catalog should cover every file in Drake's deformable example directory"
  LeanTest.assertEqual deformableModelAssets.size 6
    "Catalog should include four SDF models and two local VTK meshes"
  LeanTest.assertEqual deformablePlantAssets.size 12
    "Catalog should identify the files that feed deformable plant construction"
  LeanTest.assertTrue
    (deformableExampleAssetPaths.contains "models/simple_gripper.sdf")
    "Catalog should include the rigid gripper SDF used by deformable_torus.cc"
  LeanTest.assertTrue
    (deformableExampleAssetPaths.contains "test/point_source_force_field_test.cc")
    "Catalog should include Drake's force-field regression test"

  let torus ← assertSome
    (findDeformableExampleAsset? "models/deformable_torus.sdf")
    "deformable torus SDF asset"
  LeanTest.assertTrue (torus.kind == DeformableExampleAssetKind.model)
    "Torus SDF should be a model asset"
  LeanTest.assertTrue (torus.localDependencies.contains "models/torus.vtk")
    "Torus SDF should record its local VTK mesh dependency"
  LeanTest.assertTrue torus.feedsDeformablePlant
    "Torus SDF should feed deformable plant construction"

  let teddy ← assertSome
    (findDeformableExampleAsset? "models/deformable_teddy.sdf")
    "deformable teddy SDF asset"
  LeanTest.assertTrue (teddy.localDependencies.contains "models/teddy.vtk")
    "Teddy SDF should record its local VTK mesh dependency"

  let bubbles ← assertSome
    (findDeformableExampleAsset? "models/bubbles.sdf")
    "deformable bubbles SDF asset"
  LeanTest.assertTrue
    (bubbles.externalDependencies.contains
      "package://drake_models/wsg_50_description/meshes/bubble.vtk")
    "Bubble SDF should record Drake model-package bubble mesh dependency"
  LeanTest.assertTrue
    (bubbles.externalDependencies.contains
      "package://drake_models/wsg_50_description/meshes/textured_bubble.obj")
    "Bubble SDF should record Drake model-package visual mesh dependency"

  let torusDemo ← assertSome
    (findDeformableExampleAsset? "deformable_torus.cc")
    "deformable torus source asset"
  LeanTest.assertTrue
    (torusDemo.localDependencies.contains "models/simple_gripper.sdf")
    "Torus demo should record the simple gripper SDF parser input"
  LeanTest.assertTrue
    (torusDemo.localDependencies.contains "suction_cup_controller.h")
    "Torus demo should record selectable suction controller dependency"

@[test]
def testSdfPhysicsProviderRecordsDeformableAndRigidModelContent : IO Unit := do
  assertOk validateSdfPhysicsProvider?
    "deformable SDF physics provider"
  LeanTest.assertEqual deformableSdfModelSpecs.size 3
    "Provider should expose torus, bubble, and teddy deformable SDF models"

  let torus := torusSdfModelSpec.links[0]!
  LeanTest.assertEqual torus.name "torus"
    "Torus model should expose the Drake torus link"
  LeanTest.assertTrue (approx torus.pose.translation.z 0.02925 1.0e-12)
    s!"Torus pose height should match deformable_torus.sdf, got {torus.pose.translation.z}"
  LeanTest.assertEqual torus.collisionMesh.uri
    "package://drake/examples/multibody/deformable/models/torus.vtk"
    "Torus model should record the local Drake VTK mesh URI"
  LeanTest.assertTrue (maxAbsDiff torus.collisionMesh.scale.asArray #[0.65, 0.65, 0.65] < 1.0e-12)
    s!"Torus mesh scale should match deformable_torus.sdf, got {reprStr torus.collisionMesh.scale}"
  let torusMu ← assertSome torus.proximity.dynamicFriction?
    "torus dynamic friction"
  LeanTest.assertTrue (approx torusMu 1.15 1.0e-12)
    s!"Torus dynamic friction should match deformable_torus.sdf, got {torusMu}"

  LeanTest.assertEqual bubbleSdfModelSpec.links.size 2
    "Bubble SDF should expose left and right deformable membrane links"
  let leftBubble := bubbleSdfModelSpec.links[0]!
  let rightBubble := bubbleSdfModelSpec.links[1]!
  LeanTest.assertEqual leftBubble.name "left"
    "Bubble SDF should preserve the left link name"
  LeanTest.assertTrue (maxAbsDiff leftBubble.pose.translation.asArray #[-0.185, -0.09, 0.06] < 1.0e-12)
    s!"Left bubble pose should match bubbles.sdf, got {reprStr leftBubble.pose.translation}"
  LeanTest.assertTrue (maxAbsDiff rightBubble.pose.translation.asArray #[-0.185, 0.09, 0.06] < 1.0e-12)
    s!"Right bubble pose should match bubbles.sdf, got {reprStr rightBubble.pose.translation}"
  LeanTest.assertEqual leftBubble.collisionMesh.uri
    "package://drake_models/wsg_50_description/meshes/bubble.vtk"
    "Bubble collision mesh should preserve the Drake model-package URI"
  let leftBubbleVisual ← assertSome leftBubble.visualMesh?
    "left bubble visual mesh"
  LeanTest.assertEqual leftBubbleVisual.uri
    "package://drake_models/wsg_50_description/meshes/textured_bubble.obj"
    "Bubble visual mesh should preserve the Drake model-package URI"
  let bubbleMu ← assertSome leftBubble.proximity.dynamicFriction?
    "bubble dynamic friction"
  let bubbleDissipation ← assertSome leftBubble.proximity.huntCrossleyDissipation?
    "bubble Hunt-Crossley dissipation"
  LeanTest.assertTrue (approx bubbleMu 1.0 1.0e-12)
    s!"Bubble dynamic friction should match bubbles.sdf, got {bubbleMu}"
  LeanTest.assertTrue (approx bubbleDissipation 5.0 1.0e-12)
    s!"Bubble contact dissipation should match bubbles.sdf, got {bubbleDissipation}"
  LeanTest.assertTrue (approx leftBubble.config.youngsModulus 1.0e4 1.0e-12)
    s!"Bubble Young's modulus should match bubbles.sdf, got {leftBubble.config.youngsModulus}"
  LeanTest.assertTrue (approx leftBubble.config.massDensity 10.0 1.0e-12)
    s!"Bubble density should match bubbles.sdf, got {leftBubble.config.massDensity}"

  let teddy := teddySdfModelSpec.links[0]!
  LeanTest.assertEqual teddy.name "teddy"
    "Teddy SDF should expose the teddy deformable link"
  LeanTest.assertTrue (maxAbsDiff teddy.collisionMesh.scale.asArray #[0.15, 0.15, 0.15] < 1.0e-12)
    s!"Teddy mesh scale should match deformable_teddy.sdf, got {reprStr teddy.collisionMesh.scale}"
  LeanTest.assertEqual teddy.collisionMesh.uri
    "package://drake/examples/multibody/deformable/models/teddy.vtk"
    "Teddy collision mesh should preserve the local Drake VTK URI"
  LeanTest.assertTrue teddy.visualEmpty
    "Teddy visual geometry should record Drake's empty visual geometry"
  let teddyDiffuse ← assertSome teddy.visualDiffuse?
    "teddy diffuse visual material"
  LeanTest.assertTrue (maxAbsDiff #[teddyDiffuse.r, teddyDiffuse.g, teddyDiffuse.b, teddyDiffuse.a]
      #[0.82, 0.71, 0.55, 1.0] < 1.0e-12)
    s!"Teddy diffuse material should match deformable_teddy.sdf, got {reprStr teddyDiffuse}"
  let teddyMu ← assertSome teddy.proximity.dynamicFriction?
    "teddy dynamic friction"
  LeanTest.assertTrue (approx teddyMu 0.9 1.0e-12)
    s!"Teddy dynamic friction should match deformable_teddy.sdf, got {teddyMu}"
  LeanTest.assertTrue (approx teddy.config.youngsModulus 5.0e4 1.0e-12)
    s!"Teddy Young's modulus should match deformable_teddy.sdf, got {teddy.config.youngsModulus}"

  LeanTest.assertEqual simpleGripperSdfSpec.links.size 3
    "Simple gripper should expose body, left finger, and right finger rigid links"
  LeanTest.assertEqual simpleGripperSdfSpec.joints.size 3
    "Simple gripper should expose translate, left slider, and right slider joints"
  let body := simpleGripperSdfSpec.links[0]!
  LeanTest.assertEqual body.name "body"
    "Simple gripper should preserve the body link"
  LeanTest.assertTrue (approx body.inertia.mass 0.988882 1.0e-12)
    s!"Body mass should match simple_gripper.sdf, got {body.inertia.mass}"
  LeanTest.assertTrue (approx body.inertia.rotational.izz 0.164814 1.0e-12)
    s!"Body rotational inertia should match simple_gripper.sdf, got {body.inertia.rotational.izz}"
  let leftFinger := simpleGripperSdfSpec.links[1]!
  let leftCollision ← assertSome leftFinger.collisionBox?
    "left gripper finger collision box"
  LeanTest.assertTrue (maxAbsDiff leftCollision.size.asArray #[0.007, 0.081, 0.028] < 1.0e-12)
    s!"Finger collision box should match simple_gripper.sdf, got {reprStr leftCollision.size}"
  let padModulus ← assertSome leftFinger.proximity.hydroelasticModulus?
    "left finger hydroelastic modulus"
  let padDissipation ← assertSome leftFinger.proximity.huntCrossleyDissipation?
    "left finger contact dissipation"
  LeanTest.assertTrue (leftFinger.proximity.hydroelastic == HydroelasticRepresentation.compliant)
    "Finger collision pad should record compliant hydroelastic contact"
  LeanTest.assertTrue (approx padModulus 1.0e6 1.0e-12)
    s!"Finger hydroelastic modulus should match simple_gripper.sdf, got {padModulus}"
  LeanTest.assertTrue (approx padDissipation 5.0 1.0e-12)
    s!"Finger contact dissipation should match simple_gripper.sdf, got {padDissipation}"

  let translate := simpleGripperSdfSpec.joints[0]!
  LeanTest.assertEqual translate.name "translate_joint"
    "Simple gripper should preserve the translate joint"
  LeanTest.assertTrue (maxAbsDiff translate.axis.asArray #[0.0, -1.0, 0.0] < 1.0e-12)
    s!"Translate joint axis should match simple_gripper.sdf, got {reprStr translate.axis}"
  LeanTest.assertEqual translate.axisExpressedIn "__model__"
    "Translate joint axis should preserve the SDF expressed_in frame"
  let translateGains ← assertSome translate.controllerGains?
    "translate joint controller gains"
  LeanTest.assertTrue (approx translateGains.p 10000.0 1.0e-12 && approx translateGains.d 1.0 1.0e-12)
    s!"Translate joint gains should match simple_gripper.sdf, got {reprStr translateGains}"

  let rightSlider := simpleGripperSdfSpec.joints[2]!
  let mimic ← assertSome rightSlider.mimic?
    "right slider mimic joint"
  let effortLimit ← assertSome rightSlider.effortLimit?
    "right slider effort limit"
  LeanTest.assertEqual mimic.jointName "left_slider"
    "Right slider should mimic the left slider"
  LeanTest.assertTrue (approx mimic.multiplier (-1.0) 1.0e-12)
    s!"Right slider mimic multiplier should be -1, got {mimic.multiplier}"
  LeanTest.assertTrue (approx mimic.offset 0.0 1.0e-12)
    s!"Right slider mimic offset should be zero, got {mimic.offset}"
  LeanTest.assertTrue (approx effortLimit 0.0 1.0e-12)
    s!"Right slider effort limit should disable the extra actuator, got {effortLimit}"

@[test]
def testPointSourceForceFieldMatchesDrakeEvaluateAtTest : IO Unit := do
  let field : PointSourceForceField :=
    { pointSourceForceField with p_BC := { z := 0.123 }, falloffDistance := 0.2 }
  let pose : RigidPose :=
    { rpy := { roll := 1.0, pitch := 2.0, yaw := 3.0 },
      translation := { x := 3.0, y := 4.0, z := 5.0 } }
  let p_WC := field.sourceWorldPoint pose

  let p_WQ1 := Vec3.add p_WC { z := 0.1 }
  let f1 ← assertOk (field.evaluateAt? pose (some 42.0) p_WQ1)
    "point-source force inside falloff range"
  LeanTest.assertTrue (maxAbsDiff f1.asArray #[0.0, 0.0, -21.0] < 1.0e-12)
    s!"Inside-range force should point toward source with linear falloff, got {reprStr f1}"

  let p_WQ2 := Vec3.add p_WC { z := 0.3 }
  let f2 ← assertOk (field.evaluateAt? pose (some 42.0) p_WQ2)
    "point-source force outside falloff range"
  LeanTest.assertTrue (maxAbsDiff f2.asArray #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Outside-range force should be zero, got {reprStr f2}"

  let fOff ← assertOk (field.evaluateAt? pose (some 0.0) p_WQ1)
    "point-source force with zero input"
  LeanTest.assertTrue (maxAbsDiff fOff.asArray #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Zero maximum force density should turn the field off, got {reprStr fOff}"

  let fUnconnected ← assertOk (field.evaluateAt? pose none p_WQ1)
    "point-source force with unconnected input"
  LeanTest.assertTrue (maxAbsDiff fUnconnected.asArray #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Unconnected input port should evaluate as zero, got {reprStr fUnconnected}"

@[test]
def testParallelGripperControllerMatchesDrakeStateMachine : IO Unit := do
  let cfg : ParallelGripperController :=
    { openWidth := 0.12, closedWidth := 0.04, height := 0.25 }
  let s0 ← assertOk (cfg.desiredState? 0.0) "parallel gripper initial state"
  LeanTest.assertTrue (maxAbsDiff s0 #[0.0, -0.06, 0.0, 0.0] < 1.0e-12)
    s!"Initial state should be open at zero wrist height, got {reprStr s0}"

  let sMidClose ← assertOk (cfg.desiredState? 0.75) "parallel gripper close interpolation"
  LeanTest.assertTrue (maxAbsDiff sMidClose #[0.0, -0.04, 0.0, 0.0] < 1.0e-12)
    s!"Halfway close should interpolate finger position, got {reprStr sMidClose}"

  let sMidLift ← assertOk (cfg.desiredState? 2.25) "parallel gripper lift interpolation"
  LeanTest.assertTrue (maxAbsDiff sMidLift #[0.125, -0.02, 0.0, 0.0] < 1.0e-12)
    s!"Halfway lift should interpolate wrist height, got {reprStr sMidLift}"

  let sHold ← assertOk (cfg.desiredState? 4.0) "parallel gripper hold state"
  LeanTest.assertTrue (maxAbsDiff sHold #[0.25, -0.02, 0.0, 0.0] < 1.0e-12)
    s!"Hold segment should keep lifted closed configuration, got {reprStr sHold}"

  let sOpen ← assertOk (cfg.desiredState? 7.0) "parallel gripper final open state"
  LeanTest.assertTrue (maxAbsDiff sOpen #[0.25, -0.06, 0.0, 0.0] < 1.0e-12)
    s!"Final state should be open at lifted height, got {reprStr sOpen}"

@[test]
def testSuctionCupControllerMatchesDrakeTimingAndForceDensity : IO Unit := do
  let cfg : SuctionCupController :=
    { initialHeight := 0.35, objectHeight := 0.08,
      approachTime := 0.5, startSuctionTime := 1.5,
      retrieveTime := 3.0, releaseSuctionTime := 5.0 }

  let before ← assertOk (cfg.desiredState? 0.0) "suction before approach"
  LeanTest.assertTrue (maxAbsDiff before #[0.35, 0.0] < 1.0e-12)
    s!"Before approach, suction cup should hold initial height, got {reprStr before}"

  let approach ← assertOk (cfg.desiredState? 1.0) "suction approach"
  LeanTest.assertTrue (maxAbsDiff approach #[0.215, -0.27] < 1.0e-12)
    s!"During approach, desired state should descend linearly, got {reprStr approach}"

  let holdObject ← assertOk (cfg.desiredState? 2.0) "suction hold object"
  LeanTest.assertTrue (maxAbsDiff holdObject #[0.08, 0.0] < 1.0e-12)
    s!"During suction hold, desired state should stay at object height, got {reprStr holdObject}"

  let retrieve ← assertOk (cfg.desiredState? 3.5) "suction retrieve"
  LeanTest.assertTrue (maxAbsDiff retrieve #[0.215, 0.27] < 1.0e-12)
    s!"During retrieve, desired state should lift linearly, got {reprStr retrieve}"

  let fBefore ← assertOk (cfg.maxForceDensity? 1.49) "suction force before start"
  let fStart ← assertOk (cfg.maxForceDensity? 1.5) "suction force at start"
  let fRelease ← assertOk (cfg.maxForceDensity? 5.0) "suction force at release"
  let fAfter ← assertOk (cfg.maxForceDensity? 5.01) "suction force after release"
  LeanTest.assertTrue (approx fBefore 0.0 1.0e-12)
    s!"Suction force should be off before start, got {fBefore}"
  LeanTest.assertTrue (approx fStart 2.0e5 1.0e-12)
    s!"Suction force should turn on at start time, got {fStart}"
  LeanTest.assertTrue (approx fRelease 2.0e5 1.0e-12)
    s!"Suction force should stay on through release time, got {fRelease}"
  LeanTest.assertTrue (approx fAfter 0.0 1.0e-12)
    s!"Suction force should turn off after release, got {fAfter}"

@[test]
def testLumpedFemForceDensityPrimitiveUsesConfigMassAndFixedNodes : IO Unit := do
  let fem ← assertOk subdivisionDemoFemForce? "subdivision demo FEM force integration"
  LeanTest.assertEqual fem.samples.size 2
    "The demo FEM primitive should include one forced free sample and one fixed-constraint sample"
  LeanTest.assertTrue (approx fem.totalMass 0.002 1.0e-12)
    s!"Mass density times sample volume should give 2e-3 kg total mass, got {fem.totalMass}"
  LeanTest.assertTrue (approx fem.freeMass 0.001 1.0e-12)
    s!"Only one sample should remain free after fixed-constraint elimination, got {fem.freeMass}"

  let freeNode := fem.nodeForces[0]!
  LeanTest.assertTrue (maxAbsDiff freeNode.forceDensity.asArray #[0.0, 0.0, -3.0e6] < 1.0e-6)
    s!"Point-source force density should point toward the source with Drake's linear falloff, got {reprStr freeNode.forceDensity}"
  LeanTest.assertTrue (maxAbsDiff freeNode.force.asArray #[0.0, 0.0, -3.0] < 1.0e-12)
    s!"Force density times sample volume should produce a nodal force, got {reprStr freeNode.force}"
  LeanTest.assertTrue (approx freeNode.mass 0.001 1.0e-12)
    s!"Sample mass should be density times volume, got {freeNode.mass}"
  LeanTest.assertTrue (maxAbsDiff freeNode.acceleration.asArray #[0.0, 0.0, -3009.81] < 1.0e-9)
    s!"Free sample acceleration should combine gravity with nodal force / mass, got {reprStr freeNode.acceleration}"

  let fixedNode := fem.nodeForces[1]!
  LeanTest.assertTrue fixedNode.fixed
    "Second sample should represent a rigid-deformable fixed constraint"
  LeanTest.assertTrue (maxAbsDiff fixedNode.acceleration.asArray #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Fixed deformable sample should have zero acceleration after constraint elimination, got {reprStr fixedNode.acceleration}"
  LeanTest.assertTrue (fem.maxAccelerationNorm > 3000.0)
    s!"The force-density primitive should expose a nontrivial local acceleration, got {fem.maxAccelerationNorm}"

@[test]
def testSubdivisionSamplingAndEndToEndBoundaryAreVisible : IO Unit := do
  let subdivision ← assertOk subdivisionSampling? "deformable subdivision sampling"
  LeanTest.assertTrue (maxAbsDiff subdivision.coarseForceSum.asArray #[0.0, 0.0, 0.0] < 1.0e-12)
    s!"Coarse quadrature samples outside the field should see zero force, got {reprStr subdivision.coarseForceSum}"
  LeanTest.assertTrue (subdivision.subdividedSamples.size > subdivision.coarseSamples.size)
    "Subdivided sample set should add force-evaluation points"
  LeanTest.assertTrue (subdivision.subdividedForceSum.norm > 1.0)
    s!"Subdivided samples near the source should see nonzero force, got {reprStr subdivision.subdividedForceSum}"

  let result ← assertOk buildEndToEnd? "deformable end-to-end example"
  let _ ← assertOk result.trace.validate? "deformable trace validation"
  LeanTest.assertEqual result.assetCatalog.size 21
    "End-to-end deformable result should carry the validated model/source catalog"
  LeanTest.assertEqual result.deformableModels.size 3
    "End-to-end deformable result should carry the validated deformable SDF provider models"
  LeanTest.assertEqual result.simpleGripperModel.joints.size 3
    "End-to-end deformable result should carry the validated rigid gripper provider model"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.label.contains "force-density"))
    "Point-source force-density evaluation should be visible as a local primitive block"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.label.contains "lumped FEM force-density"))
    "The deformable force-density mass solve should be executable as a local primitive block"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.kind == SkeletonMoveKind.localSchurBlock &&
      move.exactness == MoveExactness.controlledApproximation &&
      move.label.contains "FEM"))
    "Full deformable FEM/SAP solve should remain visible as a controlled solver boundary"
  LeanTest.assertTrue (result.femForce.nodeForces.size == subdivisionDemoFemSamples.size)
    "End-to-end result should carry the evaluated FEM sample forces"
  LeanTest.assertTrue (result.moves.any (fun move => move.kind == SkeletonMoveKind.intervalAdjoint))
    "Deformable rollout should still expose the continuous interval-adjoint primitive"

end Tests.EventSkeletonDeformableExample
