import LeanTest
import Tyr.EventSkeleton.Physics

namespace Tests.EventSkeletonPhysics

open LeanTest
open Tyr.EventSkeleton

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

@[test]
def testCoulombFrictionCombinesLikeDrakeSurfaceProperties : IO Unit := do
  let surface1 : CoulombFriction := { staticFriction := 0.8, dynamicFriction := 0.3 }
  let surface2 : CoulombFriction := { staticFriction := 0.7, dynamicFriction := 0.5 }
  let combined := surface1.combine surface2

  LeanTest.assertTrue
    (approx combined.staticFriction (2.0 * 0.8 * 0.7 / (0.8 + 0.7)) 1.0e-12)
    s!"Static friction should use Drake's harmonic-mean surface rule, got {combined.staticFriction}"
  LeanTest.assertTrue
    (approx combined.dynamicFriction (2.0 * 0.3 * 0.5 / (0.3 + 0.5)) 1.0e-12)
    s!"Dynamic friction should use Drake's harmonic-mean surface rule, got {combined.dynamicFriction}"
  LeanTest.assertTrue ((surface1.combine surface2) == (surface2.combine surface1))
    "Friction surface combination should be commutative"
  LeanTest.assertTrue ((surface1.combine CoulombFriction.frictionless) == CoulombFriction.frictionless)
    "Combining with a frictionless surface should be frictionless"

  match ({ staticFriction := 0.2, dynamicFriction := 0.3 : CoulombFriction }).validate? with
  | .ok _ => LeanTest.fail "Dynamic friction greater than static friction should fail validation"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "exceeds")
        s!"Expected dynamic/static validation diagnostic, got: {msg}"

@[test]
def testVelocityProjectionEnforcesZeroConstraintVelocity : IO Unit := do
  let mass := #[#[2.0, 0.0], #[0.0, 8.0]]
  let jac := #[#[1.0, 0.0]]
  let projection ← assertOk
    (VelocityProjection.project? mass jac #[3.0, 4.0])
    "velocity projection"

  LeanTest.assertTrue (approx (projection.vPost.getD 0 99.0) 0.0 1.0e-12)
    s!"Projected velocity should satisfy the contact constraint, got {reprStr projection.vPost}"
  LeanTest.assertTrue (approx (projection.vPost.getD 1 0.0) 4.0 1.0e-12)
    "Projection should leave unconstrained coordinates unchanged in this diagonal case"
  LeanTest.assertTrue (approx (projection.lambda.getD 0 0.0) 6.0 1.0e-12)
    s!"Expected contact-space impulse multiplier 6, got {reprStr projection.lambda}"
  LeanTest.assertTrue (approx (projection.constraintVelocityAfter.getD 0 99.0) 0.0 1.0e-12)
    s!"Post-projection constraint velocity should be zero, got {reprStr projection.constraintVelocityAfter}"

@[test]
def testVelocityProjectionSupportsMovingConstraintTarget : IO Unit := do
  let mass := #[#[2.0, 0.0], #[0.0, 8.0]]
  let jac := #[#[1.0, 0.0]]
  let projection ← assertOk
    (VelocityProjection.project? mass jac #[3.0, 4.0] (some #[1.0]))
    "moving-target velocity projection"

  LeanTest.assertTrue (approx (projection.vPost.getD 0 99.0) 1.0 1.0e-12)
    s!"Projected velocity should match target constraint velocity, got {reprStr projection.vPost}"
  LeanTest.assertTrue (approx (projection.lambda.getD 0 0.0) 4.0 1.0e-12)
    s!"Expected reduced impulse multiplier 4, got {reprStr projection.lambda}"
  LeanTest.assertTrue (approx (projection.constraintVelocityAfter.getD 0 99.0) 1.0 1.0e-12)
    s!"Post-projection constraint velocity should match target, got {reprStr projection.constraintVelocityAfter}"

@[test]
def testVelocityProjectionRejectsMalformedOrSingularProblems : IO Unit := do
  match VelocityProjection.project? #[#[1.0, 0.0], #[0.0, 1.0]] #[#[1.0]] #[1.0, 2.0] with
  | .ok _ => LeanTest.fail "Wrong-width contact Jacobian should fail"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "constraint Jacobian row")
        s!"Expected Jacobian width diagnostic, got: {msg}"

  match VelocityProjection.project?
      #[#[1.0, 0.0], #[0.0, 1.0]]
      #[#[1.0, 0.0], #[2.0, 0.0]]
      #[1.0, 0.0] with
  | .ok _ => LeanTest.fail "Redundant constraints should produce a singular Delassus system"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "singular")
        s!"Expected singular-system diagnostic, got: {msg}"

@[test]
def testNormalContactLcpBalancesSustainedUnilateralContact : IO Unit := do
  let problem : NormalContactLcpProblem := {
    massMatrix := #[#[2.0, 0.0], #[0.0, 8.0]]
    normalJacobian := #[#[1.0, 0.0]]
    generalizedForces := #[-4.0, 0.0]
    label := "single sustained contact"
  }
  let result ← assertOk (problem.solve? 1.0e-10) "normal contact LCP"

  LeanTest.assertTrue (approx (result.freeAcceleration.getD 0 0.0) (-2.0) 1.0e-12)
    s!"Free acceleration should be closing before contact, got {result.freeAcceleration}"
  LeanTest.assertTrue (approx (result.normalForces.getD 0 0.0) 4.0 1.0e-12)
    s!"Normal force should balance the closing acceleration through Delassus, got {result.normalForces}"
  LeanTest.assertTrue (approx (result.acceleration.getD 0 99.0) 0.0 1.0e-12)
    s!"Post-contact acceleration should satisfy the unilateral constraint, got {result.acceleration}"
  LeanTest.assertTrue (approx (result.normalMotionAfter.getD 0 99.0) 0.0 1.0e-12)
    s!"Normal contact motion should be complementary, got {result.normalMotionAfter}"
  LeanTest.assertTrue (result.solution.maxComplementarity < 1.0e-10)
    s!"Expected tight complementarity, got {result.solution.maxComplementarity}"

@[test]
def testNormalContactLcpFastExitsForSeparatingMotionAndValidatesBias : IO Unit := do
  let separating : NormalContactLcpProblem := {
    massMatrix := #[#[2.0, 0.0], #[0.0, 8.0]]
    normalJacobian := #[#[1.0, 0.0]]
    generalizedForces := #[4.0, 0.0]
    label := "separating contact"
  }
  let result ← assertOk (separating.solve? 1.0e-10) "separating normal LCP"
  LeanTest.assertEqual result.solution.activeSet #[]
    "Separating contact should not activate a normal force"
  LeanTest.assertTrue (approx (result.normalForces.getD 0 99.0) 0.0 1.0e-12)
    s!"Separating contact should have zero normal force, got {result.normalForces}"
  LeanTest.assertTrue (approx (result.normalMotionAfter.getD 0 0.0) 2.0 1.0e-12)
    s!"Separating normal acceleration should pass through, got {result.normalMotionAfter}"

  match ({ separating with normalBias := #[0.0, 1.0] }).solve? with
  | .ok _ => LeanTest.fail "Wrong-sized normal bias should fail validation"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "normal bias size")
        s!"Expected normal bias size diagnostic, got: {msg}"

@[test]
def testLinearBushingRollPitchYawMapsSpringDamperWrenchThroughJacobians : IO Unit := do
  let params : LinearBushingRollPitchYawParams := {
    torqueStiffness := #[10.0, 20.0, 0.0]
    torqueDamping := #[1.0, 2.0, 0.0]
    forceStiffness := #[100.0, 200.0, 300.0]
    forceDamping := #[4.0, 5.0, 6.0]
    label := "test bushing"
  }
  let state : LinearBushingRollPitchYawState := {
    rpyError := #[0.1, -0.2, 3.0]
    angularVelocityError := #[0.5, -0.25, 7.0]
    translationError := #[0.01, -0.02, 0.03]
    translationVelocityError := #[0.4, -0.5, 0.6]
    rpyJacobian := #[#[1.0, 0.0], #[0.0, 1.0], #[1.0, 1.0]]
    translationJacobian := #[#[2.0, 0.0], #[0.0, 3.0], #[1.0, -1.0]]
    label := "test state"
  }
  let result ← assertOk
    (LinearBushingRollPitchYaw.evaluate? 2 params state)
    "linear bushing evaluation"

  LeanTest.assertEqual result.torque #[-1.5, 4.5, -0.0]
    "Torque spring-damper should leave the free yaw axis unforced"
  LeanTest.assertEqual result.force #[-2.6, 6.5, -12.6]
    "Force spring-damper should combine stiffness and damping componentwise"
  LeanTest.assertEqual result.generalizedForce #[-19.3, 36.6]
    "Bushing generalized force should be J^T times torque/force wrench rows"
  LeanTest.assertTrue (result.potentialEnergy > 0.0)
    "Stiffness should contribute stored spring energy"
  LeanTest.assertTrue (result.dissipationPower > 0.0)
    "Damping should report positive dissipated power"

  match LinearBushingRollPitchYaw.evaluate? 2 params { state with translationJacobian := #[#[1.0]] } with
  | .ok _ => LeanTest.fail "Wrong-width bushing Jacobian row should fail"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "translationJacobian")
        s!"Expected translation Jacobian diagnostic, got: {msg}"

@[test]
def testParticleSpringGraphEvaluatesElasticAndDampingForces : IO Unit := do
  let params : ParticleSpringParams := {
    mass := 2.0
    stiffness := 10.0
    damping := 3.0
    gravityZ := 0.0
  }
  let q := #[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
  let v := #[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
  let spring : ParticleSpring := { particle0 := 0, particle1 := 1, restLength := 1.0 }
  let result ← assertOk
    (ParticleSpringSystem.accumulateForces? 2 params #[spring] q v)
    "particle spring force accumulation"

  LeanTest.assertTrue (approx (result.elasticForces.getD 0 0.0) 10.0 1.0e-12)
    s!"Elastic force on particle 0 should pull toward particle 1, got {result.elasticForces}"
  LeanTest.assertTrue (approx (result.elasticForces.getD 3 0.0) (-10.0) 1.0e-12)
    s!"Elastic force on particle 1 should be equal and opposite, got {result.elasticForces}"
  LeanTest.assertTrue (approx (result.dampingForces.getD 0 0.0) 3.0 1.0e-12)
    s!"Damping force on particle 0 should follow Drake's relative-velocity projection, got {result.dampingForces}"
  LeanTest.assertTrue (approx (result.dampingForces.getD 3 0.0) (-3.0) 1.0e-12)
    s!"Damping force on particle 1 should be equal and opposite, got {result.dampingForces}"
  LeanTest.assertTrue (approx (result.forces.getD 0 0.0) 13.0 1.0e-12)
    s!"Total spring force should combine elastic and damping terms, got {result.forces}"
  LeanTest.assertTrue (approx result.elasticEnergy 5.0 1.0e-12)
    s!"Expected 1/2*k*extension^2 = 5, got {result.elasticEnergy}"
  LeanTest.assertTrue (approx result.dampingPower 3.0 1.0e-12)
    s!"Expected d*projected_velocity^2 = 3, got {result.dampingPower}"

  let (accel, _) ← assertOk
    (ParticleSpringSystem.accelerations? 2 params #[spring] q v #[])
    "particle spring acceleration"
  LeanTest.assertTrue (approx (accel.getD 0 0.0) 13.0 1.0e-12)
    s!"Mass per particle is 1 in this setup, so acceleration should equal force, got {accel}"
  LeanTest.assertTrue (approx (accel.getD 3 0.0) (-13.0) 1.0e-12)
    s!"Particle 1 acceleration should be equal and opposite, got {accel}"

end Tests.EventSkeletonPhysics
