import LeanTest
import Tyr.EventSkeleton.Manipulator

namespace Tests.EventSkeletonFullPhysics

open LeanTest
open Tyr.EventSkeleton

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertError {α : Type} (res : Except String α) (label : String) :
    IO String := do
  match res with
  | .ok _ => LeanTest.fail s!"{label}: expected error, got ok"
  | .error msg => pure msg

private def assertSome {α : Type} (value : Option α) (label : String) :
    IO α := do
  match value with
  | some value => pure value
  | none => LeanTest.fail s!"{label}: expected some value"

private def contactA : ContactCandidate :=
  {
    id := 10
    signedDistance := -0.01
    normalVelocity := -0.2
    tangentVelocity := 3.0
    normalJacobian := #[0.0, 1.0]
    tangentJacobian := #[1.0, 0.0]
    label := "penetrating contact"
  }

private def contactB : ContactCandidate :=
  {
    id := 20
    signedDistance := 0.1
    normalVelocity := 0.0
    tangentVelocity := 0.0
    normalJacobian := #[0.0, 1.0]
    tangentJacobian := #[1.0, 0.0]
    label := "separated contact"
  }

private def forceModel : CompliantContactModel :=
  {
    normalStiffness := 1000.0
    normalDamping := 10.0
    tangentDamping := 100.0
    friction := { staticFriction := 0.6, dynamicFriction := 0.5 }
    label := "test compliant model"
  }

private def primitiveBundle (candidates : Array ContactCandidate)
    (source : ContactForceSource := .compliantModel)
    (forces : Array ContactForceScalars := #[])
    (label : String := "primitive bundle") : FullPhysicsPrimitives :=
  {
    massMatrix := #[#[2.0, 0.0], #[0.0, 4.0]]
    qdot := #[1.0, -2.0]
    actuationForces := #[1.0, 0.0]
    biasForces := #[0.5, 1.0]
    contactCandidates := candidates
    supportPolicy := .threshold 0.0
    contactForceSource := source
    contactForces := forces
    compliantContactModel := forceModel
    label := label
  }

private structure ProviderContactState where
  signedDistance : Float
  normalVelocity : Float
  tangentVelocity : Float := 0.0
  deriving Repr, Inhabited

private def providerContactCandidate (state : ProviderContactState) : ContactCandidate :=
  {
    contactA with
      signedDistance := state.signedDistance
      normalVelocity := state.normalVelocity
      tangentVelocity := state.tangentVelocity
  }

private def dynamicProvider : FullPhysicsPrimitiveProvider ProviderContactState :=
  {
    label := "state-dependent contact provider"
    primitivesAt? := fun state =>
      .ok (primitiveBundle #[providerContactCandidate state, contactB]
        .compliantModel #[] "state-dependent primitive bundle")
  }

private def fullPlantStep : FullMultibodyPlantStep :=
  {
    model := {
      modelName := "two-dof-test-plant"
      modelUri := "package://tyr/two_dof_test.urdf"
      numPositions := 2
      numVelocities := 2
      numActuatedDofs := 1
      label := "two dof test plant"
    }
    config := {}
    q0 := #[0.0, 0.0]
    v0 := #[1.0, -2.0]
    actuation := #[1.0]
    t0 := 0.0
    t1 := 0.01
    label := "two-dof full plant primitive step"
  }

@[test]
def testGeneralizedActuationMapExpandsActuatorsIntoVelocityCoordinates :
    IO Unit := do
  let offsetMap :=
    GeneralizedActuationMap.contiguousOffset
      4 2 1 "test floating-base offset actuation map"
  let expanded ← assertOk (offsetMap.generalizedForces? #[2.0, 3.0])
    "offset generalized actuation map"
  LeanTest.assertEqual expanded #[0.0, 2.0, 3.0, 0.0]
    "Actuator inputs should land in their configured velocity coordinates"

  let step : FullMultibodyPlantStep := {
    model := {
      modelName := "four-velocity-test-plant"
      modelUri := "package://tyr/four_velocity_test.urdf"
      numPositions := 4
      numVelocities := 4
      numActuatedDofs := 2
      label := "four velocity test plant"
    }
    config := {}
    q0 := #[0.0, 0.0, 0.0, 0.0]
    v0 := #[0.0, 0.0, 0.0, 0.0]
    actuation := #[5.0, -1.0]
    t0 := 0.0
    t1 := 0.01
    label := "four-velocity full plant primitive step"
  }
  let fromStep ← assertOk (offsetMap.generalizedForcesFromStep? step)
    "step-bound generalized actuation map"
  LeanTest.assertEqual fromStep #[0.0, 5.0, -1.0, 0.0]
    "Plant-step actuation should be expanded through the same map"

  let identity ← assertOk
    ((GeneralizedActuationMap.identity 3 "identity test map").generalizedForces?
      #[1.0, 2.0, 3.0])
    "identity generalized actuation map"
  LeanTest.assertEqual identity #[1.0, 2.0, 3.0]
    "Fully actuated fixed-base plants should use the identity map"

  let duplicateMap : GeneralizedActuationMap := {
    velocityDim := 4
    actuatorVelocityIndices := #[1, 1]
    label := "duplicate test map"
  }
  let duplicateMsg ← assertError
    (duplicateMap.generalizedForces? #[1.0, 2.0])
    "duplicate generalized actuation map"
  LeanTest.assertTrue (duplicateMsg.contains "duplicate")
    s!"Expected duplicate-index diagnostic, got {duplicateMsg}"

  let outOfRangeMap : GeneralizedActuationMap := {
    velocityDim := 4
    actuatorVelocityIndices := #[4]
    label := "out-of-range test map"
  }
  let outOfRangeMsg ← assertError
    (outOfRangeMap.generalizedForces? #[1.0])
    "out-of-range generalized actuation map"
  LeanTest.assertTrue (outOfRangeMsg.contains "outside velocity dimension")
    s!"Expected out-of-range diagnostic, got {outOfRangeMsg}"

@[test]
def testFullPlantPrimitivePhysicsBindsPlantStepToPrimitiveDynamics :
    IO Unit := do
  let physics : FullPlantPrimitivePhysics := {
    step := fullPlantStep
    primitives := primitiveBundle #[contactA, contactB]
      .compliantModel #[] "plant-bound primitive bundle"
    intervalVertex := 123
    label := "plant-bound full physics"
  }
  assertOk physics.validate? "full plant primitive physics validation"
  let result ← assertOk physics.solve? "full plant primitive physics solve"
  LeanTest.assertEqual result.move.targets #[123]
    "Plant-bound primitive solve should target the wrapped interval"
  LeanTest.assertEqual result.derivative.qdot fullPlantStep.v0
    "Plant-bound primitive solve should use the plant step velocity"

  let badPhysics : FullPlantPrimitivePhysics :=
    { physics with step := { fullPlantStep with v0 := #[0.0, 0.0] } }
  let msg ← assertError badPhysics.validate?
    "mismatched plant primitive physics validation"
  LeanTest.assertTrue (msg.contains "qdot[0]")
    s!"Expected velocity mismatch diagnostic, got {msg}"

@[test]
def testDynamicContactSupportFeedsFullMassMatrixPhysics : IO Unit := do
  let equation ← assertOk
    (FullPhysicsEquation.fromDynamicContacts?
      #[#[2.0, 0.0], #[0.0, 4.0]]
      #[1.0, -2.0]
      #[1.0, 0.0]
      #[0.5, 1.0]
      #[contactA, contactB]
      (.threshold 0.0)
      forceModel
      "dynamic-contact-step")
    "full physics equation"
  let result ← assertOk (equation.solve? 42) "full physics solve"

  LeanTest.assertEqual result.support.selectedLocalIndices #[0]
    "Threshold support should select only the penetrating contact"
  LeanTest.assertEqual result.support.totalCandidates 2
    "Support should retain the dynamic source candidate count"
  LeanTest.assertTrue (result.contactForces[0]!.mode == ContactMode.impacting)
    "Runtime contact classification should mark the selected contact as impacting"
  LeanTest.assertTrue (approx result.contactForces[0]!.normalForce 12.0 1.0e-12)
    s!"Normal penalty+damping force should be 12, got {result.contactForces[0]!.normalForce}"
  LeanTest.assertTrue (approx result.contactForces[0]!.tangentForce (-6.0) 1.0e-12)
    s!"Tangential damping should be clipped by the Coulomb cone, got {result.contactForces[0]!.tangentForce}"
  LeanTest.assertEqual result.generalizedContactForce #[-6.0, 12.0]
    "Contact force should map through J^T into generalized coordinates"
  LeanTest.assertEqual result.generalizedForces #[-5.0, 12.0]
    "Actuation and generalized contact force should compose before bias subtraction"
  LeanTest.assertTrue (approx (result.derivative.vdot.getD 0 0.0) (-2.75) 1.0e-12)
    s!"First acceleration should solve M vdot = tau + J^T f - bias, got {result.derivative.vdot}"
  LeanTest.assertTrue (approx (result.derivative.vdot.getD 1 0.0) 2.75 1.0e-12)
    s!"Second acceleration should solve M vdot = tau + J^T f - bias, got {result.derivative.vdot}"
  LeanTest.assertTrue (result.move.kind == SkeletonMoveKind.intervalAdjoint)
    "Full physics solve should expose an interval-adjoint elimination move"
  LeanTest.assertTrue (result.move.exactness == MoveExactness.exact)
    "The assembled mass-matrix dynamics solve is exact for the selected contact support"
  LeanTest.assertTrue (result.supportMove.kind == SkeletonMoveKind.markMarginalize)
    "Dynamic contact support should be represented as a separate mark/support elimination move"
  LeanTest.assertTrue (result.supportMove.exactness == MoveExactness.controlledApproximation)
    "Threshold contact support is a controlled fixed-trace approximation"
  LeanTest.assertEqual result.supportMove.targets #[42]
    "Support-selection diagnostics should target the same interval vertex"
  LeanTest.assertEqual result.move.targets #[42]
    "Full physics move should target the supplied interval vertex"

@[test]
def testBilateralConstraintsSolveThroughFullPhysicsPrimitive : IO Unit := do
  let constraint : BilateralConstraintPrimitive := {
    id := 1
    jacobian := #[#[1.0, 1.0]]
    targetAcceleration := #[0.0]
    label := "sum acceleration closure"
  }
  let primitives : FullPhysicsPrimitives := {
    massMatrix := FloatMatrix.identity 2
    qdot := #[0.0, 0.0]
    actuationForces := #[1.0, 0.0]
    bilateralConstraints := #[constraint]
    label := "bilateral full physics"
  }
  let result ← assertOk (primitives.solve? 91)
    "bilateral constrained full physics solve"
  let solve ← assertSome result.constraintSolve?
    "bilateral constraint solve"
  LeanTest.assertEqual solve.jacobian #[#[1.0, 1.0]]
    "Full physics should assemble the bilateral constraint Jacobian"
  LeanTest.assertEqual solve.delassus #[#[2.0]]
    "The local Schur complement should be J M^-1 J^T"
  LeanTest.assertEqual solve.multiplierRhs #[-1.0]
    "Constraint RHS should cancel the unconstrained closure acceleration"
  LeanTest.assertEqual solve.multipliers #[-0.5]
    "Signed bilateral multiplier should solve the dense Schur block"
  LeanTest.assertEqual result.generalizedConstraintForce #[-0.5, -0.5]
    "Bilateral multipliers should map back through J^T"
  LeanTest.assertEqual result.derivative.vdot #[0.5, -0.5]
    "Constrained acceleration should satisfy J vdot = 0"
  LeanTest.assertEqual solve.constraintAccelerationAfter #[0.0]
    "The solved acceleration should satisfy the target constraint acceleration"
  LeanTest.assertEqual result.generalizedForces #[0.5, -0.5]
    "Full physics generalized forces should include the bilateral constraint force"
  LeanTest.assertEqual result.equation.generalizedForces result.generalizedForces
    "The returned manipulator equation should be the constrained total equation"

@[test]
def testFullPhysicsPrimitiveProviderComputesPrimitivesFromCurrentState :
    IO Unit := do
  let activeState : ProviderContactState := {
    signedDistance := -0.02
    normalVelocity := -0.1
    tangentVelocity := 0.0
  }
  let activeCandidates ← assertOk
    (dynamicProvider.contactCandidateSetAt? activeState)
    "active provider candidate set"
  LeanTest.assertEqual activeCandidates.candidates.size 2
    "The provider should expose the current state's dynamic candidate views"
  LeanTest.assertTrue
    (approx activeCandidates.candidates[0]!.signedDistance (-0.02) 1.0e-12)
    s!"Expected active state signed distance, got {activeCandidates.candidates[0]!.signedDistance}"

  let activeSupport ← assertOk
    (dynamicProvider.supportAt? activeState)
    "active provider support"
  LeanTest.assertEqual activeSupport.selectedLocalIndices #[0]
    "Provider support should be selected from the active state's candidates"

  let activeResult ← assertOk
    (dynamicProvider.solveAt? activeState 515)
    "active provider full physics solve"
  LeanTest.assertEqual activeResult.move.targets #[515]
    "Provider solve should still emit an interval-adjoint move for the target interval"
  LeanTest.assertTrue
    (approx activeResult.contactForces[0]!.normalForce 21.0 1.0e-12)
    s!"Expected state-dependent normal force 21, got {activeResult.contactForces[0]!.normalForce}"
  LeanTest.assertEqual activeResult.generalizedContactForce #[0.0, 21.0]
    "Provider-computed scalar contact force should map through the same J^T f boundary"

  let separatedState : ProviderContactState := {
    signedDistance := 0.2
    normalVelocity := 0.0
    tangentVelocity := 0.0
  }
  let separatedSupport ← assertOk
    (dynamicProvider.supportAt? separatedState)
    "separated provider support"
  LeanTest.assertEqual separatedSupport.selectedLocalIndices #[]
    "Changing state should recompute contact support, not reuse the active branch"
  let separatedResult ← assertOk
    (dynamicProvider.solveAt? separatedState 516)
    "separated provider full physics solve"
  LeanTest.assertEqual separatedResult.contactForces.size 0
    "Separated state should not synthesize contact forces"
  LeanTest.assertEqual separatedResult.generalizedContactForce #[0.0, 0.0]
    "Separated state should produce zero generalized contact force"

@[test]
def testFullPhysicsPrimitivesRecomputeDynamicContactSupport : IO Unit := do
  let activeResult ← assertOk
    ((primitiveBundle #[contactA, contactB] .compliantModel #[] "active primitive state").solve? 77)
    "active full physics primitive solve"
  LeanTest.assertEqual activeResult.support.selectedLocalIndices #[0]
    "The current state's penetrating candidate should be selected"
  LeanTest.assertEqual activeResult.support.totalCandidates 2
    "The primitive bundle should preserve the realized dynamic candidate count"
  LeanTest.assertTrue (approx activeResult.generalizedContactForce[0]! (-6.0) 1.0e-12)
    s!"Expected active contact to contribute tangent force, got {activeResult.generalizedContactForce}"

  let providerCountResult ← assertOk
    ({ primitiveBundle #[contactA, contactB] .compliantModel #[]
        "provider-count primitive state" with
        sourceContactCandidateCount? := some 5 }.solve? 79)
    "provider-count full physics primitive solve"
  LeanTest.assertEqual providerCountResult.support.totalCandidates 5
    "Full physics should preserve the dynamic provider's source candidate count"
  LeanTest.assertEqual providerCountResult.support.candidates.size 2
    "The solver still receives only the retained primitive candidate views"

  let separatedA := {
    contactA with
      signedDistance := 0.25
      normalVelocity := 0.0
      tangentVelocity := 0.0
  }
  let separatedResult ← assertOk
    ((primitiveBundle #[separatedA, contactB] .compliantModel #[] "separated primitive state").solve? 78)
    "separated full physics primitive solve"
  LeanTest.assertEqual separatedResult.support.selectedLocalIndices #[]
    "Support selection must be recomputed from the current candidate array"
  LeanTest.assertEqual separatedResult.generalizedContactForce #[0.0, 0.0]
    "No selected contact means no generalized contact force"
  LeanTest.assertEqual separatedResult.move.targets #[78]
    "The assembled full-physics move still targets the supplied interval"

@[test]
def testFullPhysicsPrimitivesAcceptPrecomputedExactContactForces : IO Unit := do
  let classifiedA := contactA.withClassifiedMode 0.0 1.0e-9
  let exactForces := #[ContactForceScalars.fromCandidate classifiedA 7.0 (-3.0)]
  let result ← assertOk
    ((primitiveBundle #[contactA, contactB] .precomputed exactForces "precomputed exact force").solve? 88)
    "precomputed full physics primitive solve"
  LeanTest.assertEqual result.support.selectedLocalIndices #[0]
    "Precomputed force scalars are still attached to dynamically selected support"
  LeanTest.assertEqual result.contactForces.size 1
    "Model-specific exact force laws should pass through unchanged"
  LeanTest.assertEqual result.contactForces[0]!.candidateId classifiedA.id
    "Precomputed force should remain attached to the selected candidate id"
  LeanTest.assertTrue (approx result.contactForces[0]!.normalForce 7.0 1.0e-12)
    s!"Expected precomputed normal force to pass through, got {result.contactForces[0]!.normalForce}"
  LeanTest.assertTrue (approx result.contactForces[0]!.tangentForce (-3.0) 1.0e-12)
    s!"Expected precomputed tangent force to pass through, got {result.contactForces[0]!.tangentForce}"
  LeanTest.assertEqual result.generalizedContactForce #[-3.0, 7.0]
    "Precomputed scalar forces should map through the same J^T f primitive boundary"
  LeanTest.assertTrue (approx (result.derivative.vdot.getD 0 0.0) (-1.25) 1.0e-12)
    s!"Expected first acceleration from exact scalar force, got {result.derivative.vdot}"
  LeanTest.assertTrue (approx (result.derivative.vdot.getD 1 0.0) 1.5 1.0e-12)
    s!"Expected second acceleration from exact scalar force, got {result.derivative.vdot}"

@[test]
def testFullPhysicsValidationRejectsBadContactBoundaryData : IO Unit := do
  let support := ContactSupport.selectFull #[contactA] "bad support"
  let badForceId : FullPhysicsEquation := {
    massMatrix := #[#[1.0, 0.0], #[0.0, 1.0]]
    qdot := #[0.0, 0.0]
    actuationForces := #[0.0, 0.0]
    contactSupport := support
    contactForces := #[
      { ContactForceScalars.fromCandidate contactA 1.0 0.0 with candidateId := 99 }
    ]
    label := "bad-force-id"
  }
  match badForceId.solve? with
  | .ok _ => LeanTest.fail "Mismatched contact force ids should fail validation"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "candidate id")
        s!"Expected candidate-id diagnostic, got: {msg}"

  let badWidthCandidate := { contactA with normalJacobian := #[1.0] }
  let badWidth : FullPhysicsEquation := {
    massMatrix := #[#[1.0, 0.0], #[0.0, 1.0]]
    qdot := #[0.0, 0.0]
    actuationForces := #[0.0, 0.0]
    contactSupport := ContactSupport.selectFull #[badWidthCandidate] "bad width"
    contactForces := #[ContactForceScalars.fromCandidate badWidthCandidate 1.0 0.0]
    label := "bad-width"
  }
  match badWidth.solve? with
  | .ok _ => LeanTest.fail "Wrong-width contact Jacobians should fail validation"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "normal Jacobian width")
        s!"Expected Jacobian-width diagnostic, got: {msg}"

end Tests.EventSkeletonFullPhysics
