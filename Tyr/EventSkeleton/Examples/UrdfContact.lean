import LeanUrdfTypeProvider
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Trace

/-!
# URDF-Backed Contact Hybrid ODE Example

This example consumes a small URDF through `LeanUrdfTypeProvider` and builds a
Drake-style witness/contact skeleton from the generated robot declarations.

The shape mirrors Drake's witness-event pattern in `../drake`: a continuous
ODE segment is integrated until a contact guard reaches zero, then an
unrestricted reset changes velocity.  The URDF contributes the robot topology,
prismatic joint axis, moving-link mass, and collision sphere radius.
-/

namespace Tyr.EventSkeleton.Examples.UrdfContact

open Tyr.EventSkeleton
open LeanUrdfTypeProvider

urdf_type_provider "Tyr/EventSkeleton/Examples/contact_probe.urdf" as ContactProbeUrdf

def robot : Robot := ContactProbeUrdf.model
def probeLink : Link := ContactProbeUrdf.probe_linkLink
def slideJoint : Joint := ContactProbeUrdf.vertical_slideJoint

def probeMass : Float :=
  match probeLink.inertial? with
  | some inertial => inertial.mass
  | none => 0.0

private def collisionSphereRadius? (collision : Collision) : Option Float :=
  match collision.geometry with
  | .sphere radius => some radius
  | _ => none

def contactRadius : Float :=
  (probeLink.collisions.findSome? collisionSphereRadius?).getD 0.0

structure ContactParams where
  gravity : Float := 9.81
  restitution : Float := 0.4
  deriving Repr, Inhabited

structure ContactState where
  position : Float
  velocity : Float
  deriving Repr, Inhabited

def params : ContactParams := {}
def impactTime : Float := 0.2
def initialVelocity : Float := -1.0

/--
Initial position chosen so the analytic free-flight ODE reaches the URDF
collision sphere radius at `impactTime`.
-/
def initialPosition : Float :=
  contactRadius - initialVelocity * impactTime +
    0.5 * params.gravity * impactTime * impactTime

def initialState : ContactState :=
  { position := initialPosition, velocity := initialVelocity }

def freeFlightAt (x : ContactState) (t : Float) : ContactState :=
  {
    position := x.position + x.velocity * t - 0.5 * params.gravity * t * t
    velocity := x.velocity - params.gravity * t
  }

def preImpactState : ContactState :=
  freeFlightAt initialState impactTime

def contactGuard (x : ContactState) : Float :=
  x.position - contactRadius

def contactCandidate (x : ContactState := preImpactState) : ContactCandidate :=
  {
    id := 0
    bodyA := probeLink.name
    bodyB := "world"
    point_W := #[0.0, 0.0, contactRadius]
    normal_W := #[0.0, 0.0, 1.0]
    signedDistance := contactGuard x
    normalVelocity := x.velocity
    tangentVelocity := 0.0
    normalJacobian := #[1.0]
    tangentJacobian := #[0.0]
    label := "urdf probe sphere against world half-space"
  }

def fullPhysicsContactModel : CompliantContactModel :=
  {
    normalStiffness := 2000.0
    normalDamping := 20.0
    tangentDamping := 0.0
    friction := CoulombFriction.frictionless
    label := "urdf probe compliant contact"
  }

def fullPhysicsMassMatrix : Array (Array Float) :=
  #[#[probeMass]]

def fullPhysicsBiasForces : Array Float :=
  #[probeMass * params.gravity]

def contactCandidateSet (x : ContactState := preImpactState) : ContactCandidateSet :=
  {
    candidates := #[contactCandidate x]
    sourceCandidateCount? := some 1
    label := "urdf contact probe collision provider"
  }

def contactCandidateProvider : ContactCandidateProvider ContactState :=
  {
    label := "urdf contact probe collision provider"
    candidatesAt? := fun x => .ok (contactCandidateSet x)
  }

def fullPhysicsPrimitivesAt (x : ContactState) : FullPhysicsPrimitives :=
  {
    massMatrix := fullPhysicsMassMatrix
    qdot := #[x.velocity]
    actuationForces := #[0.0]
    biasForces := fullPhysicsBiasForces
    contactCandidates := (contactCandidateSet x).candidates
    sourceContactCandidateCount? := (contactCandidateSet x).sourceCandidateCount?
    supportPolicy := .threshold 0.0
    contactForceSource := .compliantModel
    compliantContactModel := fullPhysicsContactModel
    label := "urdf-contact-full-physics"
  }

def fullPhysicsPrimitives : FullPhysicsPrimitives :=
  fullPhysicsPrimitivesAt preImpactState

def fullPhysicsProvider : FullPhysicsPrimitiveProvider ContactState :=
  {
    label := "urdf contact full physics provider"
    primitivesAt? := fun x => .ok (fullPhysicsPrimitivesAt x)
  }

def contactEventVertex : VertexId := 200
def contactModeVertex : VertexId := 201
def fullPhysicsIntervalVertex : VertexId := 202

def fullPhysicsEquation? : Except String FullPhysicsEquation :=
  fullPhysicsProvider.equationAt? preImpactState

def fullPhysicsAt?
    (x : ContactState)
    (intervalVertex : VertexId := fullPhysicsIntervalVertex) :
    Except String FullPhysicsResult :=
  fullPhysicsProvider.solveAt? x intervalVertex

def fullPhysicsEulerStep?
    (x : ContactState)
    (dt : Float)
    (intervalVertex : VertexId := fullPhysicsIntervalVertex) :
    Except String ContactState := do
  if !(Float.isFinite dt) || dt < 0.0 then
    .error s!"URDF contact full-physics step dt must be nonnegative and finite, got {dt}"
  let result ← fullPhysicsAt? x intervalVertex
  pure {
    position := x.position + dt * result.derivative.qdot.getD 0 0.0
    velocity := x.velocity + dt * result.derivative.vdot.getD 0 0.0
  }

def resetState (x : ContactState) : ContactState :=
  { position := x.position, velocity := -params.restitution * x.velocity }

def postImpactState : ContactState :=
  resetState preImpactState

structure ContactTrajectorySample where
  time : Float
  phase : String
  state : ContactState
  deriving Repr, Inhabited

/--
Reference trajectory used by the launch example.  Samples are evaluated from
the analytic free-flight solution at 5 ms intervals.  At the localized impact
time both one-sided states are retained, so the velocity reset is represented
as a jump rather than an interpolated segment.
-/
def contactTrajectory : Array ContactTrajectorySample := Id.run do
  let mut samples : Array ContactTrajectorySample := #[]
  for i in [:81] do
    let t := Float.ofNat i / 200.0
    if i < 40 then
      samples := samples.push { time := t, phase := "pre", state := freeFlightAt initialState t }
    else if i == 40 then
      samples := samples.push { time := impactTime, phase := "pre", state := preImpactState }
      samples := samples.push { time := impactTime, phase := "post", state := postImpactState }
    else
      samples := samples.push {
        time := t
        phase := "post"
        state := freeFlightAt postImpactState (t - impactTime)
      }
  return samples

def contactTrajectoryCsv : String :=
  let rows := contactTrajectory.map (fun sample =>
    s!"{sample.time},{sample.phase},{sample.state.position},{sample.state.velocity}")
  String.intercalate "\n" ((#["time,phase,position,velocity"] ++ rows).toList) ++ "\n"

def writeContactTrajectoryCsv (path : String) : IO Unit :=
  IO.FS.writeFile path contactTrajectoryCsv

def freeFlightVectorField (x : ContactState) : Array Float :=
  #[x.velocity, -params.gravity]

def contactResetJac : Array (Array Float) :=
  #[#[1.0, 0.0], #[0.0, -params.restitution]]

def contactResetTheta (x : ContactState) : Array (Array Float) :=
  #[#[0.0], #[-x.velocity]]

def contactGuardGrad : Array Float :=
  #[1.0, 0.0]

def contactSaltationData : SaltationData :=
  SaltationData.mkFromFields
    contactResetJac
    contactGuardGrad
    (freeFlightVectorField preImpactState)
    (freeFlightVectorField postImpactState)
    (resetTheta := contactResetTheta preImpactState)

def terminalPostImpactAdjoint : Array Float :=
  #[0.0, 1.0]

def contactModeSupport : RuntimeSupport :=
  {
    policy := .topK 2
    selectedIds := #[0, 2]
    totalCandidates? := some 3
    label := "runtime-selected contact solver modes"
  }

def contactModeMarkData : CategoricalMarkData :=
  {
    probs := #[0.8, 0.2]
    messages := #[
      { value := 2.0, stateAdjoint := #[0.1, 0.0] },
      { value := 5.0, stateAdjoint := #[0.0, 0.2] }
    ]
    probStateJac := #[#[0.05, 0.0], #[-0.05, 0.0]]
  }

def acceptedContactSegment : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := 0.3
    tAfter := impactTime
    madeJumpAfter := true
    label := "urdf-contact-free-flight"
  }

def contactTrace : DynamicEventTrace :=
  DynamicEventTrace.empty
    |>.push (.interval acceptedContactSegment)
    |>.push (.saltation contactEventVertex contactSaltationData)
    |>.push (.categoricalMark contactModeVertex contactModeSupport contactModeMarkData)

structure UrdfContactResult where
  robot : Robot
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  fullPhysics : FullPhysicsResult
  saltationAlpha : Float
  preImpactAdjoint : Array Float
  restitutionGrad : Array Float
  markMessage : EventMessage
  deriving Repr, Inhabited

def buildEndToEnd? : Except String UrdfContactResult := do
  robot.validateTree
  contactTrace.validate?
  let saltationAlpha ← contactSaltationData.timingAdjoint? terminalPostImpactAdjoint
  let preImpactAdjoint ← contactSaltationData.reverseState? terminalPostImpactAdjoint
  let restitutionGrad ← contactSaltationData.reverseTheta? terminalPostImpactAdjoint
  let markMessage ← contactModeMarkData.exactEliminate?
  let fullPhysicsEquation ← fullPhysicsEquation?
  let fullPhysics ← fullPhysicsEquation.solve? fullPhysicsIntervalVertex
  pure {
    robot := robot
    trace := contactTrace
    moves := contactTrace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
    fullPhysics := fullPhysics
    saltationAlpha := saltationAlpha
    preImpactAdjoint := preImpactAdjoint
    restitutionGrad := restitutionGrad
    markMessage := markMessage
  }

end Tyr.EventSkeleton.Examples.UrdfContact
