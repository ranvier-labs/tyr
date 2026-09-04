import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Physics

/-!
# Drake-Style Rolling Sphere Example

This example mirrors the physics boundary of
`../drake/examples/multibody/rolling_sphere`: a free sphere with spatial
velocity, Coulomb friction, selectable point/hydroelastic/hybrid contact model
metadata, and dynamically generated contact candidates for ground and an
optional wall.

The contact solve is intentionally expressed with existing EventSkeleton
primitives: mass matrix, generalized velocity, contact Jacobian rows, support
selection, generalized force assembly, and velocity projection.
-/

namespace Tyr.EventSkeleton.Examples.RollingSphere

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/rolling_sphere/populate_ball_plant.cc"
      concept := "adds Ball rigid body, sphere collision geometry, half-space ground, contact material, and colored visual spots"
    },
    {
      path := "../drake/examples/multibody/rolling_sphere/populate_ball_plant.h"
      concept := "declares the ball-plant population helper, plant body indices, and colored visual spot geometry boundary"
    },
    {
      path := "../drake/examples/multibody/rolling_sphere/rolling_sphere_run_dynamics.cc"
      concept := "configures point/hydroelastic/hybrid contact, 7 positions, 6 velocities, and the default sliding initial velocity"
    },
    {
      path := "../drake/examples/multibody/rolling_sphere/README.md"
      concept := "documents the supported rigid/compliant contact-model combinations and optional wall"
    }
  ]

structure Vec3 where
  x : Float := 0.0
  y : Float := 0.0
  z : Float := 0.0
  deriving Repr, BEq, Inhabited

structure Rgba where
  r : Float := 0.0
  g : Float := 0.0
  b : Float := 0.0
  a : Float := 1.0
  deriving Repr, BEq, Inhabited

inductive ComplianceKind where
  | rigid
  | compliant
  deriving Repr, BEq, Inhabited

inductive ContactModelChoice where
  | point
  | hydroelastic
  | hybrid
  deriving Repr, BEq, Inhabited

inductive HydroelasticRepresentation where
  | tri
  | poly
  deriving Repr, BEq, Inhabited

inductive ContactPairResolution where
  | pointPair
  | hydroelasticSurface
  | pointFallback
  | unsupported (reason : String)
  deriving Repr, BEq, Inhabited

namespace ContactPairResolution

def usesPoint : ContactPairResolution → Bool
  | .pointPair => true
  | .pointFallback => true
  | _ => false

def usesHydroelastic : ContactPairResolution → Bool
  | .hydroelasticSurface => true
  | _ => false

def isUnsupported : ContactPairResolution → Bool
  | .unsupported _ => true
  | _ => false

end ContactPairResolution

structure CoulombFriction where
  staticFriction : Float := 0.3
  dynamicFriction : Float := 0.3
  deriving Repr, Inhabited

namespace CoulombFriction

def isValid (friction : CoulombFriction) : Bool :=
  friction.staticFriction.isFinite && friction.staticFriction >= 0.0 &&
    friction.dynamicFriction.isFinite && friction.dynamicFriction >= 0.0

end CoulombFriction

structure RollingSphereParams where
  radius : Float := 0.05
  mass : Float := 0.1
  gravity : Float := 9.81
  hydroelasticModulus : Float := 5.0e4
  dissipation : Float := 5.0
  friction : CoulombFriction := {}
  penetrationAllowance : Float := 1.0e-3
  stictionTolerance : Float := 1.0e-4
  normalStiffness : Float := 5.0e4
  contactModel : ContactModelChoice := .point
  hydroRep : HydroelasticRepresentation := .tri
  rigidSphere : Bool := false
  compliantGround : Bool := false
  addWall : Bool := false
  wallCenterX : Float := -0.5
  wallSizeX : Float := 0.2
  wallSizeY : Float := 4.0
  wallSizeZ : Float := 0.4
  wallHydroelasticModulus : Float := 1.0e8
  plantStepSize : Float := 0.0
  simulationTime : Float := 2.0
  deriving Repr, Inhabited

def params : RollingSphereParams := {}

namespace RollingSphereParams

def isValid (p : RollingSphereParams) : Bool :=
  p.radius.isFinite && p.radius > 0.0 &&
    p.mass.isFinite && p.mass > 0.0 &&
    p.gravity.isFinite && p.gravity >= 0.0 &&
    p.hydroelasticModulus.isFinite && p.hydroelasticModulus > 0.0 &&
    p.dissipation.isFinite && p.dissipation >= 0.0 &&
    p.friction.isValid &&
    p.penetrationAllowance.isFinite && p.penetrationAllowance >= 0.0 &&
    p.stictionTolerance.isFinite && p.stictionTolerance >= 0.0 &&
    p.normalStiffness.isFinite && p.normalStiffness >= 0.0 &&
    p.wallCenterX.isFinite &&
    p.wallSizeX.isFinite && p.wallSizeX > 0.0 &&
    p.wallSizeY.isFinite && p.wallSizeY > 0.0 &&
    p.wallSizeZ.isFinite && p.wallSizeZ > 0.0 &&
    p.wallHydroelasticModulus.isFinite && p.wallHydroelasticModulus > 0.0 &&
    p.plantStepSize.isFinite && p.plantStepSize >= 0.0 &&
    p.simulationTime.isFinite && p.simulationTime > 0.0

end RollingSphereParams

structure RollingSphereState where
  x : Float := 0.0
  y : Float := 0.0
  z : Float := 0.0
  roll : Float := 0.0
  pitch : Float := 0.0
  yaw : Float := 0.0
  vx : Float := 0.0
  vy : Float := 0.0
  vz : Float := 0.0
  wx : Float := 0.0
  wy : Float := 0.0
  wz : Float := 0.0
  deriving Repr, Inhabited

namespace RollingSphereState

def isValid (state : RollingSphereState) : Bool :=
  state.x.isFinite && state.y.isFinite && state.z.isFinite &&
    state.roll.isFinite && state.pitch.isFinite && state.yaw.isFinite &&
    state.vx.isFinite && state.vy.isFinite && state.vz.isFinite &&
    state.wx.isFinite && state.wy.isFinite && state.wz.isFinite

def velocityVector (state : RollingSphereState) : Array Float :=
  #[state.vx, state.vy, state.vz, state.wx, state.wy, state.wz]

def withVelocityVector (state : RollingSphereState) (v : Array Float) :
    RollingSphereState :=
  { state with
    vx := v.getD 0 state.vx
    vy := v.getD 1 state.vy
    vz := v.getD 2 state.vz
    wx := v.getD 3 state.wx
    wy := v.getD 4 state.wy
    wz := v.getD 5 state.wz
  }

end RollingSphereState

def positionCoordinateCount : Nat := 7
def velocityCoordinateCount : Nat := 6

def defaultState (p : RollingSphereParams := params) : RollingSphereState :=
  {
    x := 0.0
    y := 0.0
    z := p.radius
    roll := 0.0
    pitch := 0.0
    yaw := 0.0
    vx := 1.5
    vy := 0.0
    vz := 0.0
    wx := 0.0
    wy := -360.0
    wz := 0.0
  }

def degreesToRadians (deg : Float) : Float :=
  pi * deg / 180.0

/-- Drake's default angular-velocity flags are documented as degrees/s, but
the example passes them directly into `SpatialVelocity`.  We preserve that
runtime convention in `defaultState`. -/
def preservesDrakeAngularVelocityFlagUnits : Bool := true

def solidSphereUnitInertiaMoment (radius : Float) : Float :=
  0.4 * radius * radius

def solidSphereRotationalInertia (p : RollingSphereParams) : Float :=
  p.mass * solidSphereUnitInertiaMoment p.radius

def massMatrix (p : RollingSphereParams := params) : Array (Array Float) :=
  let i := solidSphereRotationalInertia p
  #[
    #[p.mass, 0.0, 0.0, 0.0, 0.0, 0.0],
    #[0.0, p.mass, 0.0, 0.0, 0.0, 0.0],
    #[0.0, 0.0, p.mass, 0.0, 0.0, 0.0],
    #[0.0, 0.0, 0.0, i, 0.0, 0.0],
    #[0.0, 0.0, 0.0, 0.0, i, 0.0],
    #[0.0, 0.0, 0.0, 0.0, 0.0, i]
  ]

def visualSpotRadius (p : RollingSphereParams := params) : Float :=
  0.2 * p.radius

def visualSpotRadialOffset (p : RollingSphereParams := params) : Float :=
  p.radius - 0.45 * visualSpotRadius p

structure VisualSpot where
  name : String
  position_B : Vec3
  radius : Float
  color : Rgba
  deriving Repr, Inhabited

private def red : Rgba := { r := 1.0, g := 0.0, b := 0.0, a := 1.0 }
private def green : Rgba := { r := 0.0, g := 1.0, b := 0.0, a := 1.0 }
private def blue : Rgba := { r := 0.0, g := 0.0, b := 1.0, a := 1.0 }

def visualSpots (p : RollingSphereParams := params) : Array VisualSpot :=
  let d := visualSpotRadialOffset p
  let r := visualSpotRadius p
  #[
    { name := "sphere_x+", position_B := { x := d }, radius := r, color := red },
    { name := "sphere_x-", position_B := { x := -d }, radius := r, color := red },
    { name := "sphere_y+", position_B := { y := d }, radius := r, color := green },
    { name := "sphere_y-", position_B := { y := -d }, radius := r, color := green },
    { name := "sphere_z+", position_B := { z := d }, radius := r, color := blue },
    { name := "sphere_z-", position_B := { z := -d }, radius := r, color := blue }
  ]

def sphereCompliance (p : RollingSphereParams) : ComplianceKind :=
  if p.rigidSphere then .rigid else .compliant

def groundCompliance (p : RollingSphereParams) : ComplianceKind :=
  if p.compliantGround then .compliant else .rigid

def wallCompliance (_p : RollingSphereParams) : ComplianceKind :=
  .compliant

private def hydroelasticCompatible (a b : ComplianceKind) : Bool :=
  (a == .rigid && b == .compliant) || (a == .compliant && b == .rigid)

def resolveContactPair
    (model : ContactModelChoice) (a b : ComplianceKind) : ContactPairResolution :=
  match model with
  | .point => .pointPair
  | .hydroelastic =>
      if hydroelasticCompatible a b then
        .hydroelasticSurface
      else
        .unsupported "hydroelastic contact requires one rigid and one compliant geometry"
  | .hybrid =>
      if hydroelasticCompatible a b then
        .hydroelasticSurface
      else
        .pointFallback

def groundPairResolution (p : RollingSphereParams := params) : ContactPairResolution :=
  resolveContactPair p.contactModel (groundCompliance p) (sphereCompliance p)

def wallPairResolution? (p : RollingSphereParams := params) :
    Option ContactPairResolution :=
  if p.addWall then
    some (resolveContactPair p.contactModel (wallCompliance p) (sphereCompliance p))
  else
    none

def contactModelUsesStrictHydro (p : RollingSphereParams) : Bool :=
  p.contactModel == .hydroelastic

def hydroelasticSurfaceRepresentation
    (rep : HydroelasticRepresentation) : HydroelasticSurfaceRepresentation :=
  match rep with
  | .tri => .triangle
  | .poly => .polygon

def hydroelasticComplianceOf : ComplianceKind → HydroelasticCompliance
  | .rigid => .rigid
  | .compliant => .compliant

inductive ContactSurface where
  | ground
  | wall
  deriving Repr, BEq, Inhabited

namespace ContactSurface

def candidateId : ContactSurface → Nat
  | .ground => 100
  | .wall => 200

def label : ContactSurface → String
  | .ground => "sphere_ground"
  | .wall => "sphere_wall"

def supportsWeight : ContactSurface → Bool
  | .ground => true
  | .wall => false

end ContactSurface

def surfaceForCandidateId (id : Nat) : ContactSurface :=
  if id == ContactSurface.wall.candidateId then
    .wall
  else
    .ground

def groundSignedDistance (p : RollingSphereParams) (state : RollingSphereState) :
    Float :=
  state.z - p.radius

def wallRightFaceX (p : RollingSphereParams) : Float :=
  p.wallCenterX + p.wallSizeX / 2.0

def wallSignedDistance (p : RollingSphereParams) (state : RollingSphereState) :
    Float :=
  state.x - wallRightFaceX p - p.radius

def groundNormalJacobian : Array Float :=
  #[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

def groundTangentXJacobian (p : RollingSphereParams) : Array Float :=
  #[1.0, 0.0, 0.0, 0.0, -p.radius, 0.0]

def groundTangentYJacobian (p : RollingSphereParams) : Array Float :=
  #[0.0, 1.0, 0.0, p.radius, 0.0, 0.0]

def wallNormalJacobian : Array Float :=
  #[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def wallTangentYJacobian (p : RollingSphereParams) : Array Float :=
  #[0.0, 1.0, 0.0, 0.0, 0.0, -p.radius]

def wallTangentZJacobian (p : RollingSphereParams) : Array Float :=
  #[0.0, 0.0, 1.0, 0.0, p.radius, 0.0]

def groundContactCandidate (p : RollingSphereParams) (state : RollingSphereState) :
    ContactCandidate :=
  let v := state.velocityVector
  let candidate : ContactCandidate := {
    id := ContactSurface.ground.candidateId
    signedDistance := groundSignedDistance p state
    normalVelocity := FloatArray.dot groundNormalJacobian v
    tangentVelocity := FloatArray.dot (groundTangentXJacobian p) v
    tangentVelocity2 := FloatArray.dot (groundTangentYJacobian p) v
    normalJacobian := groundNormalJacobian
    tangentJacobian := groundTangentXJacobian p
    tangentJacobian2 := groundTangentYJacobian p
    label := ContactSurface.ground.label
  }
  candidate.withClassifiedMode p.penetrationAllowance p.stictionTolerance

def wallContactCandidate (p : RollingSphereParams) (state : RollingSphereState) :
    ContactCandidate :=
  let v := state.velocityVector
  let candidate : ContactCandidate := {
    id := ContactSurface.wall.candidateId
    signedDistance := wallSignedDistance p state
    normalVelocity := FloatArray.dot wallNormalJacobian v
    tangentVelocity := FloatArray.dot (wallTangentYJacobian p) v
    tangentVelocity2 := FloatArray.dot (wallTangentZJacobian p) v
    normalJacobian := wallNormalJacobian
    tangentJacobian := wallTangentYJacobian p
    tangentJacobian2 := wallTangentZJacobian p
    label := ContactSurface.wall.label
  }
  candidate.withClassifiedMode p.penetrationAllowance p.stictionTolerance

def contactCandidates (p : RollingSphereParams := params)
    (state : RollingSphereState := defaultState p) : Array ContactCandidate := Id.run do
  let mut out := #[groundContactCandidate p state]
  if p.addWall then
    out := out.push (wallContactCandidate p state)
  return out

def candidateForSurface (p : RollingSphereParams) (state : RollingSphereState)
    (surface : ContactSurface) : ContactCandidate :=
  match surface with
  | .ground => groundContactCandidate p state
  | .wall => wallContactCandidate p state

def resolutionForSurface (p : RollingSphereParams) (surface : ContactSurface) :
    ContactPairResolution :=
  match surface with
  | .ground => groundPairResolution p
  | .wall =>
      match wallPairResolution? p with
      | some resolution => resolution
      | none => .unsupported "wall contact surface is disabled"

def contactSurfaces (p : RollingSphereParams) : Array ContactSurface :=
  if p.addWall then #[.ground, .wall] else #[.ground]

private def positivePart (x : Float) : Float :=
  if x > 0.0 then x else 0.0

def spherePlanePatchArea (radius penetration : Float) : Float :=
  let h :=
    if penetration <= 0.0 then
      0.0
    else if penetration >= 2.0 * radius then
      2.0 * radius
    else
      penetration
  pi * positivePart (2.0 * radius * h - h * h)

def hydroelasticAveragePressure
    (modulus radius penetration : Float) : Float :=
  modulus * positivePart penetration / radius

def contactPatchCentroid
    (p : RollingSphereParams) (state : RollingSphereState)
    (surface : ContactSurface) : Array Float :=
  match surface with
  | .ground => #[state.x, state.y, 0.0]
  | .wall => #[wallRightFaceX p, state.y, state.z]

def contactPatchNormal (surface : ContactSurface) : Array Float :=
  match surface with
  | .ground => #[0.0, 0.0, 1.0]
  | .wall => #[1.0, 0.0, 0.0]

def contactPatchBodyA (surface : ContactSurface) : String :=
  match surface with
  | .ground => "Ground"
  | .wall => "Wall"

def compliantHydroelasticModulus
    (p : RollingSphereParams) (surface : ContactSurface) : Float :=
  match surface with
  | .wall =>
      if wallCompliance p == .compliant then
        p.wallHydroelasticModulus
      else
        p.hydroelasticModulus
  | .ground =>
      if groundCompliance p == .compliant then
        p.hydroelasticModulus
      else
        p.hydroelasticModulus

def hydroelasticPatchForCandidate?
    (p : RollingSphereParams) (state : RollingSphereState)
    (surface : ContactSurface) (candidate : ContactCandidate) :
    Option HydroelasticContactPatch :=
  if candidate.signedDistance > p.penetrationAllowance then
    none
  else
    let geometricPenetration := positivePart (-candidate.signedDistance)
    let supportPenetration :=
      if geometricPenetration > 0.0 then geometricPenetration else p.penetrationAllowance
    let area := spherePlanePatchArea p.radius supportPenetration
    if area <= 0.0 then
      none
    else
      some {
        id := candidate.id
        bodyA := contactPatchBodyA surface
        bodyB := "Ball"
        complianceA := hydroelasticComplianceOf
          (match surface with
          | .ground => groundCompliance p
          | .wall => wallCompliance p)
        complianceB := hydroelasticComplianceOf (sphereCompliance p)
        representation := hydroelasticSurfaceRepresentation p.hydroRep
        area := area
        centroid := contactPatchCentroid p state surface
        normal := contactPatchNormal surface
        averagePressure :=
          hydroelasticAveragePressure
            (compliantHydroelasticModulus p surface) p.radius supportPenetration
        normalVelocity := candidate.normalVelocity
        tangentVelocity := candidate.tangentVelocity
        tangentVelocity2 := candidate.tangentVelocity2
        normalJacobian := candidate.normalJacobian
        tangentJacobian := candidate.tangentJacobian
        tangentJacobian2 := candidate.tangentJacobian2
        label := s!"{ContactSurface.label surface}_hydroelastic_patch"
      }

structure RollingContactPrimitives where
  hydroelasticPatches : Array HydroelasticContactPatch := #[]
  pointCandidates : Array ContactCandidate := #[]
  fallbackCandidates : Array ContactCandidate := #[]
  sourceSurfaceCount : Nat := 0
  deriving Repr, Inhabited

namespace RollingContactPrimitives

def candidates (primitives : RollingContactPrimitives) : Array ContactCandidate :=
  (primitives.hydroelasticPatches.map (fun patch => patch.equivalentContactCandidate)) ++
    primitives.pointCandidates ++ primitives.fallbackCandidates

def sourceCandidateCount (primitives : RollingContactPrimitives) : Nat :=
  primitives.sourceSurfaceCount

end RollingContactPrimitives

def resolvedContactPrimitives?
    (p : RollingSphereParams) (state : RollingSphereState) :
    Except String RollingContactPrimitives := do
  let mut hydroelasticPatches : Array HydroelasticContactPatch := #[]
  let mut pointCandidates : Array ContactCandidate := #[]
  let mut fallbackCandidates : Array ContactCandidate := #[]
  for surface in contactSurfaces p do
    let candidate := candidateForSurface p state surface
    let active := candidate.signedDistance <= p.penetrationAllowance
    match resolutionForSurface p surface with
    | .pointPair =>
        pointCandidates := pointCandidates.push candidate
    | .pointFallback =>
        fallbackCandidates := fallbackCandidates.push candidate
    | .hydroelasticSurface =>
        match hydroelasticPatchForCandidate? p state surface candidate with
        | some patch => hydroelasticPatches := hydroelasticPatches.push patch
        | none => pure ()
    | .unsupported reason =>
        if active then
          .error s!"rolling sphere {ContactSurface.label surface}: {reason}"
        else
          pure ()
  pure {
    hydroelasticPatches := hydroelasticPatches
    pointCandidates := pointCandidates
    fallbackCandidates := fallbackCandidates
    sourceSurfaceCount := (contactSurfaces p).size
  }

def activeSupport (p : RollingSphereParams := params)
    (state : RollingSphereState := defaultState p) : ContactSupport :=
  ContactSupport.selectByDistance p.penetrationAllowance
    (contactCandidates p state) "rolling sphere active contact support"
    |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

private def signedUnit (x : Float) : Float :=
  if x > 0.0 then 1.0 else if x < 0.0 then -1.0 else 0.0

def supportNormalForce
    (p : RollingSphereParams) (surface : ContactSurface)
    (candidate : ContactCandidate) : Float :=
  if candidate.signedDistance <= p.penetrationAllowance then
    let penetration := positivePart (-candidate.signedDistance)
    let closure := positivePart (-candidate.normalVelocity)
    let supportBias := if surface.supportsWeight then p.mass * p.gravity else 0.0
    positivePart
      (supportBias + p.normalStiffness * penetration + p.dissipation * closure)
  else
    0.0

def frictionForceForSlip
    (p : RollingSphereParams) (normalForce slip : Float) : Float :=
  if normalForce <= 0.0 then
    0.0
  else if Float.abs slip <= p.stictionTolerance then
    0.0
  else
    -p.friction.dynamicFriction * normalForce * signedUnit slip

structure RollingContactForce where
  candidateId : Nat
  surface : ContactSurface
  normalForce : Float
  tangentForce : Float
  tangentForce2 : Float
  generalizedForce : Array Float
  mode : ContactMode
  deriving Repr, Inhabited

def contactForceForCandidateWithNormal
    (p : RollingSphereParams) (surface : ContactSurface)
    (candidate : ContactCandidate) (normalForce : Float) : RollingContactForce :=
  let tangentForce := frictionForceForSlip p normalForce candidate.tangentVelocity
  let tangentForce2 := frictionForceForSlip p normalForce candidate.tangentVelocity2
  {
    candidateId := candidate.id
    surface := surface
    normalForce := normalForce
    tangentForce := tangentForce
    tangentForce2 := tangentForce2
    generalizedForce := candidate.generalizedForce3D normalForce tangentForce tangentForce2
    mode := candidate.mode
  }

def contactForceForCandidate
    (p : RollingSphereParams) (candidate : ContactCandidate) : RollingContactForce :=
  let surface := surfaceForCandidateId candidate.id
  let normalForce := supportNormalForce p surface candidate
  contactForceForCandidateWithNormal p surface candidate normalForce

def contactForces? (p : RollingSphereParams) (support : ContactSupport) :
    Except String (Array RollingContactForce) := do
  let selected ← support.selectedCandidates?
  pure (selected.map (contactForceForCandidate p))

def gravityGeneralizedForce (p : RollingSphereParams) : Array Float :=
  #[0.0, 0.0, -p.mass * p.gravity, 0.0, 0.0, 0.0]

def aggregateContactGeneralizedForce (forces : Array RollingContactForce) :
    Array Float :=
  sumGeneralizedForces (forces.map (fun force => force.generalizedForce))

def contactForceScalars (force : RollingContactForce) : ContactForceScalars :=
  {
    candidateId := force.candidateId
    normalForce := force.normalForce
    tangentForce := force.tangentForce
    tangentForce2 := force.tangentForce2
    mode := force.mode
    label := ContactSurface.label force.surface
  }

structure RollingSolverContacts where
  candidates : Array ContactCandidate := #[]
  forces : Array RollingContactForce := #[]
  deriving Repr, Inhabited

private def hydroelasticSolverCandidate
    (p : RollingSphereParams) (patch : HydroelasticContactPatch) :
    ContactCandidate :=
  patch.equivalentContactCandidate
    |>.withClassifiedMode p.penetrationAllowance p.stictionTolerance

private def hydroelasticContactForce
    (p : RollingSphereParams) (patch : HydroelasticContactPatch) :
    RollingContactForce :=
  let candidate := hydroelasticSolverCandidate p patch
  let surface := surfaceForCandidateId candidate.id
  let normalForce := supportNormalForce p surface candidate
  contactForceForCandidateWithNormal p surface candidate normalForce

def solverContactsForResolved?
    (p : RollingSphereParams) (primitives : RollingContactPrimitives) :
    Except String RollingSolverContacts := do
  let hydroCandidates :=
    primitives.hydroelasticPatches.map (hydroelasticSolverCandidate p)
  let hydroForces :=
    primitives.hydroelasticPatches.map (hydroelasticContactForce p)
  let pointLikeCandidates := primitives.pointCandidates ++ primitives.fallbackCandidates
  let pointLikeSupport :=
    ContactSupport.selectByDistance p.penetrationAllowance pointLikeCandidates
      "rolling sphere point/fallback contact support"
      |>.classifyCandidates p.penetrationAllowance p.stictionTolerance
  let pointLikeForces ← contactForces? p pointLikeSupport
  pure {
    candidates := hydroCandidates ++ pointLikeCandidates
    forces := hydroForces ++ pointLikeForces
  }

def appliedGeneralizedForce
    (p : RollingSphereParams) (forces : Array RollingContactForce) : Array Float :=
  FloatArray.add (gravityGeneralizedForce p) (aggregateContactGeneralizedForce forces)

def derivativeFromAcceleration (state : RollingSphereState) (accel : Array Float) :
    RollingSphereState :=
  {
    x := state.vx
    y := state.vy
    z := state.vz
    roll := state.wx
    pitch := state.wy
    yaw := state.wz
    vx := accel.getD 0 0.0
    vy := accel.getD 1 0.0
    vz := accel.getD 2 0.0
    wx := accel.getD 3 0.0
    wy := accel.getD 4 0.0
    wz := accel.getD 5 0.0
  }

def freeFlightDerivative (p : RollingSphereParams) (state : RollingSphereState) :
    RollingSphereState :=
  derivativeFromAcceleration state #[0.0, 0.0, -p.gravity, 0.0, 0.0, 0.0]

def rollingSphereFullPhysicsIntervalVertex : VertexId := 401

def validateFullPhysicsInputs?
    (p : RollingSphereParams) (state : RollingSphereState) :
    Except String Unit := do
  if !p.isValid then
    .error "rolling sphere params are invalid"
  if !state.isValid then
    .error "rolling sphere state must have twelve finite coordinates"

def fullPhysicsPrimitives?
    (p : RollingSphereParams) (state : RollingSphereState) :
    Except String (FullPhysicsPrimitives × Array RollingContactForce) := do
  validateFullPhysicsInputs? p state
  let resolved ← resolvedContactPrimitives? p state
  let solverContacts ← solverContactsForResolved? p resolved
  pure ({
    massMatrix := massMatrix p
    qdot := state.velocityVector
    actuationForces := Array.replicate velocityCoordinateCount 0.0
    generalizedForceContributions :=
      #[GeneralizedForceContribution.ofForce
          (gravityGeneralizedForce p)
          "rolling sphere gravity generalized force"
          "RollingSphere"]
    contactCandidates := solverContacts.candidates
    sourceContactCandidateCount? := some resolved.sourceCandidateCount
    supportPolicy := .threshold p.penetrationAllowance
    contactForceSource := .precomputed
    contactForces := solverContacts.forces.map contactForceScalars
    distanceTol := p.penetrationAllowance
    tangentVelocityTol := p.stictionTolerance
    label := "rolling sphere full physics"
  }, solverContacts.forces)

def fullPhysicsPrimitiveProvider
    (p : RollingSphereParams := params)
    (label : String := "rolling sphere full physics provider") :
    FullPhysicsPrimitiveProvider RollingSphereState :=
  {
    label := label
    primitivesAt? := fun state => do
      validateFullPhysicsInputs? p state
      let (primitives, _) ← fullPhysicsPrimitives? p state
      pure { primitives with label := label }
  }

def solveFullPhysics?
    (p : RollingSphereParams) (state : RollingSphereState) :
    Except String (FullPhysicsResult × Array RollingContactForce) := do
  let (primitive, forces) ← fullPhysicsPrimitives? p state
  let equation ← primitive.equation?
  let fullPhysics ← equation.solve? rollingSphereFullPhysicsIntervalVertex
  pure (fullPhysics, forces)

def physicsDerivative?
    (p : RollingSphereParams) (state : RollingSphereState) :
    Except String (RollingSphereState × ContactSupport × Array RollingContactForce) := do
  let (fullPhysics, forces) ← solveFullPhysics? p state
  pure (derivativeFromAcceleration state fullPhysics.derivative.vdot,
    fullPhysics.support, forces)

def contactAwareDerivative (p : RollingSphereParams) (state : RollingSphereState) :
    RollingSphereState :=
  let support := activeSupport p state
  if support.selectedLocalIndices.isEmpty then
    freeFlightDerivative p state
  else
    match physicsDerivative? p state with
    | .ok (dx, _, _) => dx
    | .error _ => freeFlightDerivative p state

def addScaledState (state dstate : RollingSphereState) (dt : Float) :
    RollingSphereState :=
  {
    x := state.x + dt * dstate.x
    y := state.y + dt * dstate.y
    z := state.z + dt * dstate.z
    roll := state.roll + dt * dstate.roll
    pitch := state.pitch + dt * dstate.pitch
    yaw := state.yaw + dt * dstate.yaw
    vx := state.vx + dt * dstate.vx
    vy := state.vy + dt * dstate.vy
    vz := state.vz + dt * dstate.vz
    wx := state.wx + dt * dstate.wx
    wy := state.wy + dt * dstate.wy
    wz := state.wz + dt * dstate.wz
  }

def eulerPhysicsStep? (p : RollingSphereParams) (dt : Float)
    (state : RollingSphereState) :
    Except String (RollingSphereState × ContactSupport × Array RollingContactForce × RollingSphereState) := do
  let (dx, support, forces) ← physicsDerivative? p state
  pure (addScaledState state dx dt, support, forces, dx)

def simulatePhysicsSteps? (p : RollingSphereParams) (dt : Float) (steps : Nat)
    (state0 : RollingSphereState) : Except String RollingSphereState := do
  let mut state := state0
  for _ in [:steps] do
    state := addScaledState state (contactAwareDerivative p state) dt
  pure state

def normalProjection? (p : RollingSphereParams) (state : RollingSphereState) :
    Except String VelocityProjection := do
  let support := activeSupport p state
  support.validateJacobianWidth? velocityCoordinateCount
  let rows ← support.constraintJacobianRows? false
  VelocityProjection.project? (massMatrix p) rows state.velocityVector

def stickingProjection? (p : RollingSphereParams) (state : RollingSphereState) :
    Except String VelocityProjection := do
  let support := activeSupport p state
  support.validateJacobianWidth? velocityCoordinateCount
  let rows ← support.constraintJacobianRows? true
  VelocityProjection.project? (massMatrix p) rows state.velocityVector

def kineticEnergy (p : RollingSphereParams) (state : RollingSphereState) : Float :=
  let v2 := state.vx * state.vx + state.vy * state.vy + state.vz * state.vz
  let w2 := state.wx * state.wx + state.wy * state.wy + state.wz * state.wz
  0.5 * p.mass * v2 + 0.5 * solidSphereRotationalInertia p * w2

def potentialEnergy (p : RollingSphereParams) (state : RollingSphereState) : Float :=
  p.mass * p.gravity * state.z

def totalEnergy (p : RollingSphereParams) (state : RollingSphereState) : Float :=
  kineticEnergy p state + potentialEnergy p state

def acceptedContactSegment (dt : Float) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := dt
    tAfter := dt
    label := "rolling sphere contact-support step"
  }

def contactBranchVertex : VertexId := 400

private def nonzeroGamma (candidate : ContactCandidate) : Float :=
  if Float.abs candidate.normalVelocity < 1.0e-9 then
    1.0
  else
    candidate.normalVelocity

def contactMessage (force : RollingContactForce) : EventMessage :=
  {
    value := force.normalForce + Float.abs force.tangentForce + Float.abs force.tangentForce2
    stateAdjoint := force.generalizedForce
    thetaGrad := #[Float.abs force.tangentForce + Float.abs force.tangentForce2]
  }

def branchChildForForce (force : RollingContactForce) : BranchChild :=
  {
    weight := 1.0
    resetJac := FloatMatrix.identity velocityCoordinateCount
    a := force.generalizedForce
    message := contactMessage force
  }

def contactBranchData? (support : ContactSupport) (forces : Array RollingContactForce) :
    Except String BranchEventData := do
  let selected ← support.selectedCandidates?
  if selected.isEmpty then
    .error "rolling-sphere contact branch requires at least one active contact"
  else if selected.size != forces.size then
    .error s!"rolling-sphere selected support size {selected.size} != force count {forces.size}"
  else
    pure {
      children := forces.map branchChildForForce
      guardGrad := selected[0]!.normalJacobian
      gamma := nonzeroGamma selected[0]!
    }

structure RollingSphereResult where
  references : Array DrakeReference
  state : RollingSphereState
  resolvedContacts : RollingContactPrimitives
  solverContacts : RollingSolverContacts
  support : ContactSupport
  runtimeSupport : RuntimeSupport
  contactForces : Array RollingContactForce
  derivative : RollingSphereState
  oneStepState : RollingSphereState
  rolloutState : RollingSphereState
  normalProjection : VelocityProjection
  stickingProjection : VelocityProjection
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  fullPhysics : FullPhysicsResult
  branchData : BranchEventData
  branchResult : BranchAggregateResult
  deriving Repr, Inhabited

def buildEndToEnd? (p : RollingSphereParams := params) :
    Except String RollingSphereResult := do
  let state := defaultState p
  let resolvedContacts ← resolvedContactPrimitives? p state
  let solverContacts ← solverContactsForResolved? p resolvedContacts
  let (fullPhysics, forces) ← solveFullPhysics? p state
  let derivative := derivativeFromAcceleration state fullPhysics.derivative.vdot
  let support := fullPhysics.support
  let runtime ← support.toRuntimeSupport?
  let (oneStepState, _, _, _) ← eulerPhysicsStep? p 1.0e-3 state
  let rolloutState ← simulatePhysicsSteps? p 1.0e-3 10 state
  let normalProjection ← normalProjection? p state
  let stickingProjection ← stickingProjection? p state
  let branchData ← contactBranchData? support forces
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedContactSegment 1.0e-3))
      |>.push (.branch contactBranchVertex runtime branchData)
  trace.validate?
  let branchResult ← branchData.aggregate?
  pure {
    references := drakeReferences
    state := state
    resolvedContacts := resolvedContacts
    solverContacts := solverContacts
    support := support
    runtimeSupport := runtime
    contactForces := forces
    derivative := derivative
    oneStepState := oneStepState
    rolloutState := rolloutState
    normalProjection := normalProjection
    stickingProjection := stickingProjection
    trace := trace
    moves := trace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
    fullPhysics := fullPhysics
    branchData := branchData
    branchResult := branchResult
  }

end Tyr.EventSkeleton.Examples.RollingSphere
