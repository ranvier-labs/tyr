import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Physics

/-!
# Drake-Style Inclined Plane Body Example

This example mirrors the multibody/contact surface of
`../drake/examples/multibody/inclined_plane_with_body`: an inclined plane
registered either as a half-space or finite box, a free rigid body that can be a
sphere, block, or block with four small collision spheres, Coulomb friction
combined by Drake's pair-material rule, and dynamically generated contact
candidates.

The point of this port is not to make a reduced model.  It expresses the
physics through the primitives we already have: signed-distance candidates,
contact velocity/Jacobian rows, Coulomb pair coefficients, `J^T f` generalized
forces, mass/inertia, support selection, and event-trace branch aggregation.
-/

namespace Tyr.EventSkeleton.Examples.InclinedPlaneBody

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

def degreesToRadians (deg : Float) : Float :=
  pi * deg / 180.0

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/inclined_plane_with_body/inclined_plane_with_body.cc"
      concept := "declares the runner flags, default body type, friction, plane angle, and initial free-body state"
    },
    {
      path := "../drake/examples/multibody/inclined_plane_with_body/README.md"
      concept := "documents the sphere, block, and block_with_4Spheres body variants"
    },
    {
      path := "../drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.cc"
      concept := "builds the inclined-plane frame, half-space/box plane geometry, body collision geometry, and gravity"
    },
    {
      path := "../drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
      concept := "defines the model construction API and bodyB_type options"
    },
    {
      path := "../drake/multibody/plant/test/inclined_plane_test.cc"
      concept := "checks the analytic rolling-sphere acceleration, friction force, and energy balance"
    }
  ]

structure Vec3 where
  x : Float := 0.0
  y : Float := 0.0
  z : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace Vec3

def add (a b : Vec3) : Vec3 :=
  { x := a.x + b.x, y := a.y + b.y, z := a.z + b.z }

def sub (a b : Vec3) : Vec3 :=
  { x := a.x - b.x, y := a.y - b.y, z := a.z - b.z }

def scale (s : Float) (v : Vec3) : Vec3 :=
  { x := s * v.x, y := s * v.y, z := s * v.z }

def dot (a b : Vec3) : Float :=
  a.x * b.x + a.y * b.y + a.z * b.z

def cross (a b : Vec3) : Vec3 :=
  {
    x := a.y * b.z - a.z * b.y
    y := a.z * b.x - a.x * b.z
    z := a.x * b.y - a.y * b.x
  }

def norm (v : Vec3) : Float :=
  Float.sqrt (v.dot v)

def toArray (v : Vec3) : Array Float :=
  #[v.x, v.y, v.z]

end Vec3

inductive BodyType where
  | sphere
  | block
  | blockWith4Spheres
  deriving Repr, BEq, Inhabited

namespace BodyType

def label : BodyType → String
  | .sphere => "sphere"
  | .block => "block"
  | .blockWith4Spheres => "block_with_4Spheres"

end BodyType

inductive PlaneShape where
  | halfSpace
  | box
  deriving Repr, BEq, Inhabited

namespace PlaneShape

def label : PlaneShape → String
  | .halfSpace => "half_space"
  | .box => "box"

end PlaneShape

structure InclinedPlaneParams where
  targetRealtimeRate : Float := 1.0
  simulationTime : Float := 2.0
  timeStep : Float := 1.0e-3
  integrationAccuracy : Float := 1.0e-6
  penetrationAllowance : Float := 1.0e-5
  stictionTolerance : Float := 1.0e-5
  angleDegrees : Float := 15.0
  gravity : Float := 9.8
  massB : Float := 0.1
  planeFriction : CoulombFriction := { staticFriction := 0.3, dynamicFriction := 0.3 }
  bodyFriction : CoulombFriction := { staticFriction := 0.3, dynamicFriction := 0.3 }
  bodyType : BodyType := .sphere
  planeShape : PlaneShape := .halfSpace
  contactApproximation : String := "lagged"
  sphereRadius : Float := 0.04
  blockLengthX : Float := 0.4
  blockLengthY : Float := 0.2
  blockLengthZ : Float := 0.04
  supportBudget : Nat := 4
  deriving Repr, Inhabited

def params : InclinedPlaneParams := {}

namespace InclinedPlaneParams

def angleRadians (p : InclinedPlaneParams) : Float :=
  degreesToRadians p.angleDegrees

def planeNormalW (p : InclinedPlaneParams) : Vec3 :=
  let theta := p.angleRadians
  { x := Float.sin theta, y := 0.0, z := Float.cos theta }

def planeTangentXW (p : InclinedPlaneParams) : Vec3 :=
  let theta := p.angleRadians
  { x := Float.cos theta, y := 0.0, z := -Float.sin theta }

def planeTangentYW (_p : InclinedPlaneParams) : Vec3 :=
  { x := 0.0, y := 1.0, z := 0.0 }

def combinedFriction (p : InclinedPlaneParams) : CoulombFriction :=
  p.planeFriction.combine p.bodyFriction

def planeBoxDimensions (p : InclinedPlaneParams) : Vec3 :=
  match p.bodyType with
  | .sphere =>
      { x := 20.0 * p.sphereRadius, y := 10.0 * p.sphereRadius, z := p.sphereRadius }
  | .block | .blockWith4Spheres =>
      { x := 8.0 * p.blockLengthX, y := 8.0 * p.blockLengthY, z := 0.04 }

def traceStepSize (p : InclinedPlaneParams) : Float :=
  if p.timeStep > 0.0 then p.timeStep else 1.0e-3

def validate? (p : InclinedPlaneParams) : Except String Unit := do
  if !(Float.isFinite p.timeStep) || p.timeStep <= 0.0 then
    .error s!"inclined plane time step must be positive and finite, got {p.timeStep}"
  if !(Float.isFinite p.penetrationAllowance) || p.penetrationAllowance < 0.0 then
    .error s!"penetration allowance must be nonnegative and finite, got {p.penetrationAllowance}"
  if !(Float.isFinite p.stictionTolerance) || p.stictionTolerance < 0.0 then
    .error s!"stiction tolerance must be nonnegative and finite, got {p.stictionTolerance}"
  if !(Float.isFinite p.gravity) || p.gravity <= 0.0 then
    .error s!"gravity must be positive and finite, got {p.gravity}"
  if !(Float.isFinite p.massB) || p.massB <= 0.0 then
    .error s!"body mass must be positive and finite, got {p.massB}"
  if !(Float.isFinite p.sphereRadius) || p.sphereRadius <= 0.0 then
    .error s!"sphere radius must be positive and finite, got {p.sphereRadius}"
  if !(Float.isFinite p.blockLengthX) || p.blockLengthX <= 0.0 then
    .error s!"blockLengthX must be positive and finite, got {p.blockLengthX}"
  if !(Float.isFinite p.blockLengthY) || p.blockLengthY <= 0.0 then
    .error s!"blockLengthY must be positive and finite, got {p.blockLengthY}"
  if !(Float.isFinite p.blockLengthZ) || p.blockLengthZ <= 0.0 then
    .error s!"blockLengthZ must be positive and finite, got {p.blockLengthZ}"
  p.planeFriction.validate? "inclined-plane surface friction"
  p.bodyFriction.validate? "body surface friction"

end InclinedPlaneParams

structure BodyState where
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

namespace BodyState

def position (x : BodyState) : Vec3 :=
  { x := x.x, y := x.y, z := x.z }

def linearVelocity (x : BodyState) : Vec3 :=
  { x := x.vx, y := x.vy, z := x.vz }

def angularVelocity (x : BodyState) : Vec3 :=
  { x := x.wx, y := x.wy, z := x.wz }

def velocityVector (x : BodyState) : Array Float :=
  #[x.vx, x.vy, x.vz, x.wx, x.wy, x.wz]

def isFinite (x : BodyState) : Bool :=
  Float.isFinite x.x && Float.isFinite x.y && Float.isFinite x.z &&
  Float.isFinite x.roll && Float.isFinite x.pitch && Float.isFinite x.yaw &&
  Float.isFinite x.vx && Float.isFinite x.vy && Float.isFinite x.vz &&
  Float.isFinite x.wx && Float.isFinite x.wy && Float.isFinite x.wz

end BodyState

def defaultState : BodyState :=
  {
    x := -1.0
    y := 0.0
    z := 1.2
  }

def sphereCenterOnPlane (p : InclinedPlaneParams) : Vec3 :=
  Vec3.scale p.sphereRadius p.planeNormalW

def contactingSphereState (p : InclinedPlaneParams := params) : BodyState :=
  let c := sphereCenterOnPlane p
  { defaultState with x := c.x, y := c.y, z := c.z }

def impactingSphereState (p : InclinedPlaneParams := params) : BodyState :=
  let c := sphereCenterOnPlane p
  let n := p.planeNormalW
  {
    defaultState with
    x := c.x
    y := c.y
    z := c.z
    vx := -n.x
    vy := -n.y
    vz := -n.z
  }

def planeSignedDistance (p : InclinedPlaneParams) (pointW : Vec3) : Float :=
  p.planeNormalW.dot pointW

def pointVelocity (x : BodyState) (r_B : Vec3) : Vec3 :=
  x.linearVelocity.add (x.angularVelocity.cross r_B)

def jacobianRowForDirection (r_B directionW : Vec3) : Array Float :=
  let angular := r_B.cross directionW
  #[directionW.x, directionW.y, directionW.z, angular.x, angular.y, angular.z]

def candidateAtBodyPoint
    (p : InclinedPlaneParams)
    (x : BodyState)
    (id : Nat)
    (r_B : Vec3)
    (label : String) : ContactCandidate :=
  let pointW := x.position.add r_B
  let velocityW := pointVelocity x r_B
  let n := p.planeNormalW
  let tx := p.planeTangentXW
  let ty := p.planeTangentYW
  {
    id := id
    signedDistance := planeSignedDistance p pointW
    normalVelocity := n.dot velocityW
    tangentVelocity := tx.dot velocityW
    tangentVelocity2 := ty.dot velocityW
    normalJacobian := jacobianRowForDirection r_B n
    tangentJacobian := jacobianRowForDirection r_B tx
    tangentJacobian2 := jacobianRowForDirection r_B ty
    label := label
  }

def sphereCollisionCandidate
    (p : InclinedPlaneParams)
    (x : BodyState)
    (id : Nat)
    (center_B : Vec3)
    (radius : Float)
    (label : String) : ContactCandidate :=
  let n := p.planeNormalW
  let contactOffset := center_B.add (Vec3.scale (-radius) n)
  let centerW := x.position.add center_B
  let velocityW := pointVelocity x contactOffset
  let tx := p.planeTangentXW
  let ty := p.planeTangentYW
  {
    id := id
    signedDistance := planeSignedDistance p centerW - radius
    normalVelocity := n.dot velocityW
    tangentVelocity := tx.dot velocityW
    tangentVelocity2 := ty.dot velocityW
    normalJacobian := jacobianRowForDirection contactOffset n
    tangentJacobian := jacobianRowForDirection contactOffset tx
    tangentJacobian2 := jacobianRowForDirection contactOffset ty
    label := label
  }

private def signs : Array Float := #[-1.0, 1.0]

def blockCornerCandidates
    (p : InclinedPlaneParams) (x : BodyState) : Array ContactCandidate := Id.run do
  let hx := p.blockLengthX / 2.0
  let hy := p.blockLengthY / 2.0
  let hz := p.blockLengthZ / 2.0
  let mut out : Array ContactCandidate := #[]
  let mut idx := 0
  for sx in signs do
    for sy in signs do
      let r : Vec3 := { x := sx * hx, y := sy * hy, z := -hz }
      out := out.push (candidateAtBodyPoint p x (2000 + idx) r s!"block_corner_{idx}")
      idx := idx + 1
  return out

def fourSphereCandidates
    (p : InclinedPlaneParams) (x : BodyState) : Array ContactCandidate := Id.run do
  let hx := p.blockLengthX / 2.0
  let hy := p.blockLengthY / 2.0
  let radius := p.blockLengthZ / 2.0
  let mut out : Array ContactCandidate := #[]
  let mut idx := 0
  for sx in signs do
    for sy in signs do
      let center : Vec3 := { x := sx * hx, y := sy * hy, z := -radius }
      out := out.push
        (sphereCollisionCandidate p x (3000 + idx) center radius s!"block_sphere_{idx}")
      idx := idx + 1
  return out

def contactCandidates
    (p : InclinedPlaneParams := params)
    (x : BodyState := defaultState) : Array ContactCandidate :=
  match p.bodyType with
  | .sphere =>
      #[sphereCollisionCandidate p x 1000 {} p.sphereRadius "sphere_plane"]
  | .block =>
      blockCornerCandidates p x
  | .blockWith4Spheres =>
      fourSphereCandidates p x

def selectedSupport
    (p : InclinedPlaneParams := params)
    (x : BodyState := defaultState) : ContactSupport :=
  ContactSupport.selectByDistance p.penetrationAllowance (contactCandidates p x)
    s!"inclined-plane {p.bodyType.label} active contacts"
    |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

def closestSupport
    (p : InclinedPlaneParams := params)
    (x : BodyState := defaultState) : ContactSupport :=
  ContactSupport.selectClosestK p.supportBudget (contactCandidates p x)
    s!"inclined-plane {p.bodyType.label} closest contacts"
    |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

private def positivePart (x : Float) : Float :=
  if x > 0.0 then x else 0.0

private def signedUnit (x : Float) : Float :=
  if x > 0.0 then 1.0 else if x < 0.0 then -1.0 else 0.0

def solidSphereUnitInertiaMoment (radius : Float) : Float :=
  0.4 * radius * radius

def sphereRotationalInertia (p : InclinedPlaneParams) : Float :=
  p.massB * solidSphereUnitInertiaMoment p.sphereRadius

def blockInertiaDiagonal (p : InclinedPlaneParams) : Vec3 :=
  {
    x := p.massB * (p.blockLengthY * p.blockLengthY + p.blockLengthZ * p.blockLengthZ) / 12.0
    y := p.massB * (p.blockLengthX * p.blockLengthX + p.blockLengthZ * p.blockLengthZ) / 12.0
    z := p.massB * (p.blockLengthX * p.blockLengthX + p.blockLengthY * p.blockLengthY) / 12.0
  }

def inertiaDiagonal (p : InclinedPlaneParams) : Vec3 :=
  match p.bodyType with
  | .sphere =>
      let i := sphereRotationalInertia p
      { x := i, y := i, z := i }
  | .block | .blockWith4Spheres =>
      blockInertiaDiagonal p

def normalForceTotal (p : InclinedPlaneParams) : Float :=
  p.massB * p.gravity * Float.cos p.angleRadians

def downhillGravityForce (p : InclinedPlaneParams) : Float :=
  p.massB * p.gravity * Float.sin p.angleRadians

def rollingSphereFrictionMagnitude (p : InclinedPlaneParams) : Float :=
  let gUnit := solidSphereUnitInertiaMoment p.sphereRadius
  p.massB * p.gravity * Float.sin p.angleRadians *
    (gUnit / (gUnit + p.sphereRadius * p.sphereRadius))

def rollingSphereAcceleration (p : InclinedPlaneParams) : Float :=
  let gUnit := solidSphereUnitInertiaMoment p.sphereRadius
  p.gravity * Float.sin p.angleRadians /
    (1.0 + gUnit / (p.sphereRadius * p.sphereRadius))

def rollingSphereSpeedAfterVerticalDrop (p : InclinedPlaneParams) (heightDrop : Float) : Float :=
  let gUnit := solidSphereUnitInertiaMoment p.sphereRadius
  Float.sqrt (2.0 * p.gravity * heightDrop /
    (1.0 + gUnit / (p.sphereRadius * p.sphereRadius)))

def desiredTangentFrictionTotal
    (p : InclinedPlaneParams) (representative : ContactCandidate) : Float :=
  let mu := p.combinedFriction
  let normal := normalForceTotal p
  let dynamicLimit := mu.dynamicFriction * normal
  match p.bodyType with
  | .sphere =>
      let required := rollingSphereFrictionMagnitude p
      if required <= mu.staticFriction * normal then
        -required
      else if Float.abs representative.tangentVelocity > p.stictionTolerance then
        -dynamicLimit * signedUnit representative.tangentVelocity
      else
        -dynamicLimit
  | .block | .blockWith4Spheres =>
      let required := downhillGravityForce p
      if Float.abs representative.tangentVelocity > p.stictionTolerance then
        -dynamicLimit * signedUnit representative.tangentVelocity
      else if required <= mu.staticFriction * normal then
        -required
      else
        -dynamicLimit

structure InclinedPlaneContactForce where
  candidateId : Nat
  normalForce : Float
  tangentForce : Float
  tangentForce2 : Float := 0.0
  generalizedForce : Array Float
  mode : ContactMode
  label : String := ""
  deriving Repr, Inhabited

def contactForces? (p : InclinedPlaneParams) (support : ContactSupport) :
    Except String (Array InclinedPlaneContactForce) := do
  p.validate?
  support.validateJacobianWidth? 6
  let selected ← support.selectedCandidates?
  if selected.isEmpty then
    pure #[]
  else
    let normalEach := normalForceTotal p / selected.size.toFloat
    let tangentEach := desiredTangentFrictionTotal p selected[0]! / selected.size.toFloat
    pure (selected.map (fun candidate =>
      let generalized := candidate.generalizedForce3D normalEach tangentEach 0.0
      {
        candidateId := candidate.id
        normalForce := normalEach
        tangentForce := tangentEach
        tangentForce2 := 0.0
        generalizedForce := generalized
        mode := candidate.mode
        label := candidate.label
      }))

def aggregateGeneralizedForce (forces : Array InclinedPlaneContactForce) : Array Float :=
  sumGeneralizedForces (forces.map (fun force => force.generalizedForce))

def gravityGeneralizedForce (p : InclinedPlaneParams) : Array Float :=
  #[0.0, 0.0, -p.massB * p.gravity, 0.0, 0.0, 0.0]

def fullPhysicsMassMatrix (p : InclinedPlaneParams) : Array (Array Float) :=
  let inertia := inertiaDiagonal p
  FloatMatrix.diagonal #[p.massB, p.massB, p.massB, inertia.x, inertia.y, inertia.z]

def fullPhysicsGravityBias (p : InclinedPlaneParams) : Array Float :=
  #[0.0, 0.0, p.massB * p.gravity, 0.0, 0.0, 0.0]

def zeroActuation : Array Float :=
  Array.replicate 6 0.0

def fullPhysicsContactForceScalars?
    (p : InclinedPlaneParams) (support : ContactSupport) :
    Except String (Array ContactForceScalars) := do
  let forces ← contactForces? p support
  pure (forces.map (fun force =>
    {
      candidateId := force.candidateId
      normalForce := force.normalForce
      tangentForce := force.tangentForce
      tangentForce2 := force.tangentForce2
      mode := force.mode
      label := force.label
    }))

def fullPhysicsPrimitives?
    (p : InclinedPlaneParams) (x : BodyState) :
    Except String FullPhysicsPrimitives := do
  p.validate?
  if !x.isFinite then
    .error "inclined-plane full physics state must be finite"
  let support := selectedSupport p x
  let forces ← fullPhysicsContactForceScalars? p support
  pure {
    massMatrix := fullPhysicsMassMatrix p
    qdot := x.velocityVector
    actuationForces := zeroActuation
    biasForces := fullPhysicsGravityBias p
    contactCandidates := contactCandidates p x
    supportPolicy := .threshold p.penetrationAllowance
    contactForceSource := .precomputed
    contactForces := forces
    distanceTol := p.penetrationAllowance
    tangentVelocityTol := p.stictionTolerance
    label := s!"inclined-plane {p.bodyType.label} full physics"
  }

def fullPhysicsPrimitiveProvider
    (p : InclinedPlaneParams := params)
    (label : String := s!"inclined-plane {p.bodyType.label} full physics provider") :
    FullPhysicsPrimitiveProvider BodyState :=
  {
    label := label
    primitivesAt? := fun x => do
      let primitives ← fullPhysicsPrimitives? p x
      pure { primitives with label := label }
  }

def fullPhysicsIntervalVertex : VertexId := 901

def solveFullPhysics?
    (p : InclinedPlaneParams) (x : BodyState)
    (intervalVertex : VertexId := fullPhysicsIntervalVertex) :
    Except String FullPhysicsResult := do
  let primitives ← fullPhysicsPrimitives? p x
  primitives.solve? intervalVertex

def derivativeFromGeneralizedForce
    (p : InclinedPlaneParams) (x : BodyState) (generalizedForce : Array Float) :
    BodyState :=
  let total := FloatArray.add generalizedForce (gravityGeneralizedForce p)
  let inertia := inertiaDiagonal p
  {
    x := x.vx
    y := x.vy
    z := x.vz
    roll := x.wx
    pitch := x.wy
    yaw := x.wz
    vx := total.getD 0 0.0 / p.massB
    vy := total.getD 1 0.0 / p.massB
    vz := total.getD 2 0.0 / p.massB
    wx := total.getD 3 0.0 / inertia.x
    wy := total.getD 4 0.0 / inertia.y
    wz := total.getD 5 0.0 / inertia.z
  }

def freeFlightDerivative (p : InclinedPlaneParams) (x : BodyState) : BodyState :=
  derivativeFromGeneralizedForce p x #[]

def contactDerivative? (p : InclinedPlaneParams) (support : ContactSupport) (x : BodyState) :
    Except String (BodyState × Array InclinedPlaneContactForce) := do
  let forces ← contactForces? p support
  pure (derivativeFromGeneralizedForce p x (aggregateGeneralizedForce forces), forces)

def contactAwareDerivative? (p : InclinedPlaneParams) (x : BodyState) :
    Except String (BodyState × ContactSupport × Array InclinedPlaneContactForce) := do
  let support := selectedSupport p x
  let selected ← support.selectedCandidates?
  if selected.isEmpty then
    pure (freeFlightDerivative p x, support, #[])
  else
    let (dx, forces) ← contactDerivative? p support x
    pure (dx, support, forces)

def addScaledState (x dx : BodyState) (dt : Float) : BodyState :=
  {
    x := x.x + dt * dx.x
    y := x.y + dt * dx.y
    z := x.z + dt * dx.z
    roll := x.roll + dt * dx.roll
    pitch := x.pitch + dt * dx.pitch
    yaw := x.yaw + dt * dx.yaw
    vx := x.vx + dt * dx.vx
    vy := x.vy + dt * dx.vy
    vz := x.vz + dt * dx.vz
    wx := x.wx + dt * dx.wx
    wy := x.wy + dt * dx.wy
    wz := x.wz + dt * dx.wz
  }

def eulerStep? (p : InclinedPlaneParams) (dt : Float) (x : BodyState) :
    Except String (BodyState × ContactSupport × Array InclinedPlaneContactForce × BodyState) := do
  let (dx, support, forces) ← contactAwareDerivative? p x
  pure (addScaledState x dx dt, support, forces, dx)

private def identityMatrix (n : Nat) : Array (Array Float) := Id.run do
  let mut rows : Array (Array Float) := #[]
  for i in [:n] do
    let mut row : Array Float := #[]
    for j in [:n] do
      row := row.push (if i == j then 1.0 else 0.0)
    rows := rows.push row
  return rows

def candidateMessage (candidate : ContactCandidate) : EventMessage :=
  let closure := positivePart (-candidate.normalVelocity)
  let slip :=
    max (Float.abs candidate.tangentVelocity) (Float.abs candidate.tangentVelocity2)
  {
    value := closure + 0.01 * slip + positivePart (-candidate.signedDistance)
    stateAdjoint :=
      FloatArray.add candidate.normalJacobian
        (FloatArray.scale 0.1 candidate.tangentJacobian)
    thetaGrad := #[closure]
  }

def branchChildForCandidate (weight : Float) (candidate : ContactCandidate) :
    BranchChild :=
  {
    weight := weight
    resetJac := identityMatrix 6
    a := candidate.normalJacobian
    message := candidateMessage candidate
  }

def contactBranchData? (support : ContactSupport) : Except String BranchEventData := do
  let selected ← support.selectedCandidates?
  if selected.isEmpty then
    .error "inclined-plane branch requires at least one retained contact"
  else
    let first := selected[0]!
    let weight := 1.0 / selected.size.toFloat
    pure {
      children := selected.map (branchChildForCandidate weight)
      guardGrad := first.normalJacobian
      gamma := first.normalVelocity
    }

def acceptedContactSegment (p : InclinedPlaneParams := params) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.traceStepSize
    tAfter := p.traceStepSize
    madeJumpAfter := true
    label := "inclined-plane contact interval"
  }

def contactBranchVertex : VertexId := 900

structure InclinedPlaneBodyResult where
  references : Array DrakeReference
  params : InclinedPlaneParams
  airborneState : BodyState
  airborneSupport : ContactSupport
  contactState : BodyState
  contactSupport : ContactSupport
  runtimeSupport : RuntimeSupport
  contactForces : Array InclinedPlaneContactForce
  fullPhysics : FullPhysicsResult
  derivative : BodyState
  oneStepState : BodyState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  branchData : BranchEventData
  branchResult : BranchAggregateResult
  deriving Repr, Inhabited

def buildEndToEnd? (p : InclinedPlaneParams := params) :
    Except String InclinedPlaneBodyResult := do
  p.validate?
  let airborne := defaultState
  let airborneSupport := selectedSupport p airborne
  let contactState := impactingSphereState { p with bodyType := .sphere }
  let contactSupport := selectedSupport { p with bodyType := .sphere } contactState
  contactSupport.validateJacobianWidth? 6
  let runtime ← contactSupport.toRuntimeSupport?
  let (oneStepState, _, contactForces, derivative) ←
    eulerStep? { p with bodyType := .sphere } p.traceStepSize contactState
  let fullPhysics ← solveFullPhysics? { p with bodyType := .sphere } contactState
  let branchData ← contactBranchData? contactSupport
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedContactSegment p))
      |>.push (.branch contactBranchVertex runtime branchData)
  trace.validate?
  let branchResult ← branchData.aggregate?
  pure {
    references := drakeReferences
    params := p
    airborneState := airborne
    airborneSupport := airborneSupport
    contactState := contactState
    contactSupport := contactSupport
    runtimeSupport := runtime
    contactForces := contactForces
    fullPhysics := fullPhysics
    derivative := derivative
    oneStepState := oneStepState
    trace := trace
    moves := trace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
    branchData := branchData
    branchResult := branchResult
  }

end Tyr.EventSkeleton.Examples.InclinedPlaneBody
