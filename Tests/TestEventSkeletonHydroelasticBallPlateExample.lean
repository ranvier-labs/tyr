import LeanTest
import Tyr.EventSkeleton.Examples.HydroelasticBallPlate

namespace Tests.EventSkeletonHydroelasticBallPlateExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.HydroelasticBallPlate

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

@[test]
def testDrakeReferencesAndDefaultsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/hydroelastic/ball_plate/ball_plate_run_dynamics.cc"))
    "Example should reference Drake's hydroelastic ball-plate runner"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/hydroelastic/ball_plate/make_ball_plate_plant.cc"))
    "Example should reference Drake's ball-plate plant helper"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/hydroelastic/ball_plate/floor.sdf"))
    "Example should reference Drake's compliant floor SDFormat file"

  LeanTest.assertTrue (params.contactModel == ContactModelChoice.hydroelastic)
    "Drake runner default contact model should be hydroelastic"
  LeanTest.assertTrue
    (params.surfaceRepresentation == HydroelasticSurfaceRepresentation.polygon)
    "Drake runner default hydroelastic surface representation should be polygon"
  LeanTest.assertEqual params.expectedPositionCount 14
    "Drake ball-plate plant should have 14 positions"
  LeanTest.assertEqual params.expectedVelocityCount 12
    "Drake ball-plate plant should have 12 velocities"
  LeanTest.assertTrue (approx params.simulationTime 0.4 1.0e-12)
    s!"Default simulation time should be 0.4, got {params.simulationTime}"
  LeanTest.assertTrue (approx params.ballHydroelasticModulus 3.0e4 1.0e-9)
    s!"Default ball modulus should be 3.0e4, got {params.ballHydroelasticModulus}"
  LeanTest.assertTrue (approx params.dissipation 3.0 1.0e-12)
    s!"Default dissipation should be 3.0, got {params.dissipation}"
  LeanTest.assertTrue (approx params.friction.staticFriction 0.3 1.0e-12)
    s!"Default static friction should be 0.3, got {params.friction.staticFriction}"
  LeanTest.assertTrue (approx params.mbpDt 0.001 1.0e-12)
    s!"Default discrete plant step should be 0.001, got {params.mbpDt}"
  LeanTest.assertTrue (approx params.meshTargetEdgeLength 0.015 1.0e-12)
    s!"Target hydroelastic edge length should be radius*0.3 = 0.015, got {params.meshTargetEdgeLength}"
  LeanTest.assertTrue (approx params.floorVolume 0.0045 1.0e-12)
    s!"Floor volume should match the 0.30x0.30x0.05 block, got {params.floorVolume}"

@[test]
def testHydroelasticPatchProviderRecordsSurfacePatches : IO Unit := do
  let defaultSupport := patchSupport params defaultState
  let defaultSelected ← assertOk defaultSupport.selectedPatches?
    "default hydroelastic support selection"
  let defaultRuntime ← assertOk defaultSupport.toRuntimeSupport?
    "default hydroelastic runtime support"
  LeanTest.assertEqual defaultRuntime.selectedIds #[4300]
    "Default Drake pose should retain only the plate-floor support patch"
  LeanTest.assertEqual defaultSelected.size 1
    "Default Drake pose should expose one active hydroelastic patch"
  LeanTest.assertEqual defaultSelected[0]!.id 4300
    "Default active patch should be plate-floor"
  LeanTest.assertTrue (defaultSelected[0]!.pairKind == HydroelasticPairKind.rigidCompliant)
    "Plate-floor contact should be rigid-compliant"
  LeanTest.assertTrue
    (defaultSelected[0]!.representation == HydroelasticSurfaceRepresentation.polygon)
    "Patch should preserve Drake's polygon surface representation"

  let contactSupport := patchSupport params (ballPlateContactState params)
  assertOk (contactSupport.validateGeometry?) "contact support geometry"
  assertOk (contactSupport.validateJacobianWidth? 12) "contact support Jacobian width"
  let selected ← assertOk contactSupport.selectedPatches?
    "contact hydroelastic support selection"
  let runtime ← assertOk contactSupport.toRuntimeSupport?
    "contact hydroelastic runtime support"
  LeanTest.assertEqual runtime.selectedIds #[4100, 4300]
    "Contact state should retain the ball-plate patch and plate-floor patch"
  LeanTest.assertEqual selected.size 2
    "Contact state should expose two active hydroelastic patches"
  LeanTest.assertEqual selected[0]!.id 4100
    "Ball-plate patch should be first and define the branch guard"
  LeanTest.assertTrue (selected[0]!.normalVelocity < 0.0)
    s!"Ball-plate patch should be closing, got vn={selected[0]!.normalVelocity}"
  LeanTest.assertTrue
    (selected.all (fun patch =>
      patch.centroid.size == 3 &&
      patch.normal.size == 3 &&
      patch.normalJacobian.size == 12 &&
      patch.tangentJacobian.size == 12 &&
      patch.tangentJacobian2.size == 12))
    "Hydroelastic patches should expose 3D geometry and 12-wide generalized-velocity rows"

@[test]
def testBallPlatePatchGeneralizedForcePushesBallAndPlateApart : IO Unit := do
  let support := patchSupport params (ballPlateContactState params)
  let forces ← assertOk (patchForces? support) "contact patch forces"
  LeanTest.assertEqual forces.size 2
    "Contact state should produce ball-plate and plate-floor force records"
  let force := forces[0]!
  LeanTest.assertEqual force.patchId 4100
    "First force should come from the ball-plate hydroelastic patch"
  LeanTest.assertTrue (force.pairKind == HydroelasticPairKind.rigidCompliant)
    "Ball-plate contact should be compliant-rigid"
  LeanTest.assertTrue (force.normalForce > 0.0)
    s!"Ball-plate normal force should be positive, got {force.normalForce}"
  LeanTest.assertTrue (force.generalizedForce.getD 2 0.0 > 0.0)
    s!"Ball should receive upward z force, got {force.generalizedForce.getD 2 0.0}"
  LeanTest.assertTrue (force.generalizedForce.getD 8 0.0 < 0.0)
    s!"Plate should receive equal opposing z force, got {force.generalizedForce.getD 8 0.0}"

@[test]
def testPlateFloorPatchSupportsPlateFromCompliantFloor : IO Unit := do
  let support := patchSupport params defaultState
  let forces ← assertOk (patchForces? support) "default patch forces"
  LeanTest.assertEqual forces.size 1
    "Default pose should produce only the plate-floor patch force"
  let force := forces[0]!
  LeanTest.assertEqual force.patchId 4300
    "Default force should come from the plate-floor hydroelastic patch"
  LeanTest.assertTrue (force.pairKind == HydroelasticPairKind.rigidCompliant)
    "Plate-floor support should be rigid-compliant"
  LeanTest.assertTrue (force.normalForce > params.plateMass * params.gravity)
    s!"Floor hydroelastic normal force should support the plate, got {force.normalForce}"
  LeanTest.assertTrue (force.generalizedForce.getD 8 0.0 > 0.0)
    s!"Floor patch should push the plate upward, got {force.generalizedForce.getD 8 0.0}"

@[test]
def testEndToEndTraceAndStepExecute : IO Unit := do
  let result ← assertOk (buildEndToEnd? params) "hydroelastic ball-plate end-to-end"
  let _ ← assertOk result.trace.validate? "hydroelastic trace validation"
  LeanTest.assertEqual result.runtimeSupport.selectedIds #[4100, 4300]
    "Runtime support should preserve stable hydroelastic patch ids"
  LeanTest.assertEqual result.patchForces.size 2
    "End-to-end step should solve retained hydroelastic patch forces"
  LeanTest.assertEqual result.moves.size 3
    "One interval plus one branch aggregate should produce three skeleton moves"
  LeanTest.assertEqual result.branchData.children.size 2
    "Branch data should aggregate both retained hydroelastic patches"
  LeanTest.assertTrue result.derivative.isFinite
    s!"Hydroelastic derivative should remain finite, got {reprStr result.derivative}"
  LeanTest.assertTrue result.oneStepState.isFinite
    s!"One-step hydroelastic state should remain finite, got {reprStr result.oneStepState}"
  LeanTest.assertTrue (result.oneStepState.ball.z < result.contactState.ball.z)
    "Forward Euler step should move the initially closing ball downward over the first millisecond"
  LeanTest.assertTrue (result.branchResult.value > 0.0)
    s!"Branch aggregate should produce a nonzero hydroelastic patch message, got {result.branchResult.value}"
  LeanTest.assertTrue (Float.isFinite result.branchResult.alpha)
    s!"Branch timing adjoint should be finite, got {result.branchResult.alpha}"

end Tests.EventSkeletonHydroelasticBallPlateExample
