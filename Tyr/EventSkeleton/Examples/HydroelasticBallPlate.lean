import Tyr.EventSkeleton.Contact
import Tyr.EventSkeleton.Physics

/-!
# Drake-Style Hydroelastic Ball-Plate Example

This ports the primitive contact surface of
`../drake/examples/hydroelastic/ball_plate`: a compliant hydroelastic ball, a
rigid dinner plate loaded by Drake from SDFormat, and a compliant hydroelastic
floor.  The example records hydroelastic contact patches directly rather than
collapsing them into a single point candidate.

The provider below is analytic and compact, but the boundary is the important
part: a hydroelastic contact backend supplies patch area, centroid, pressure,
surface representation, compliance pairing, relative velocity, and `J` rows.
The dynamics layer consumes the patch by `J^T f` just like point contacts.
-/

namespace Tyr.EventSkeleton.Examples.HydroelasticBallPlate

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/hydroelastic/ball_plate/ball_plate_run_dynamics.cc"
      concept := "declares hydroelastic contact defaults, ball initial pose and velocity, and the 14-position/12-velocity plant"
    },
    {
      path := "../drake/examples/hydroelastic/ball_plate/make_ball_plate_plant.cc"
      concept := "constructs the compliant ball, loads the rigid plate, loads and welds the compliant floor, and adds visual spin markers"
    },
    {
      path := "../drake/examples/hydroelastic/ball_plate/make_ball_plate_plant.h"
      concept := "documents the programmatic ball parameters and hydroelastic material inputs"
    },
    {
      path := "../drake/examples/hydroelastic/ball_plate/floor.sdf"
      concept := "defines the 30x30x5-cm compliant floor block and its hydroelastic modulus, friction, and dissipation"
    },
    {
      path := "../drake/examples/hydroelastic/ball_plate/README.md"
      concept := "explains the non-convex plate, polygon/triangle contact surfaces, and point-contact contrast"
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

inductive ContactModelChoice where
  | point
  | hydroelastic
  | hydroelasticWithFallback
  deriving Repr, BEq, Inhabited

namespace ContactModelChoice

def label : ContactModelChoice → String
  | .point => "point"
  | .hydroelastic => "hydroelastic"
  | .hydroelasticWithFallback => "hydroelastic_with_fallback"

end ContactModelChoice

structure BallPlateParams where
  simulationTime : Float := 0.4
  contactModel : ContactModelChoice := .hydroelastic
  surfaceRepresentation : HydroelasticSurfaceRepresentation := .polygon
  ballHydroelasticModulus : Float := 3.0e4
  floorHydroelasticModulus : Float := 3.0e4
  resolutionHintFactor : Float := 0.3
  dissipation : Float := 3.0
  friction : CoulombFriction := { staticFriction := 0.3, dynamicFriction := 0.3 }
  mbpDt : Float := 0.001
  penetrationAllowance : Float := 0.001
  gravity : Float := 9.81
  ballRadius : Float := 0.05
  ballMass : Float := 0.1
  plateMass : Float := 0.25
  plateRadius : Float := 0.1016
  plateThickness : Float := 0.012
  plateContactZ : Float := 0.055
  plateFloorPenetration : Float := 0.002
  floorSizeX : Float := 0.30
  floorSizeY : Float := 0.30
  floorSizeZ : Float := 0.05
  x0 : Float := 0.10
  z0 : Float := 0.15
  vx0 : Float := 0.0
  vy0 : Float := 0.0
  vz0 : Float := -7.0
  wxDeg0 : Float := 0.0
  wyDeg0 : Float := -10.0
  wzDeg0 : Float := 0.0
  minPatchArea : Float := 1.0e-10
  deriving Repr, Inhabited

def params : BallPlateParams := {}

namespace BallPlateParams

def expectedPositionCount (_p : BallPlateParams) : Nat := 14
def expectedVelocityCount (_p : BallPlateParams) : Nat := 12

def meshTargetEdgeLength (p : BallPlateParams) : Float :=
  p.ballRadius * p.resolutionHintFactor

def floorVolume (p : BallPlateParams) : Float :=
  p.floorSizeX * p.floorSizeY * p.floorSizeZ

def floorTopZ (_p : BallPlateParams) : Float := 0.0

def degPerSecToRadPerSec (deg : Float) : Float :=
  pi * deg / 180.0

def validate? (p : BallPlateParams) : Except String Unit := do
  if !(Float.isFinite p.mbpDt) || p.mbpDt <= 0.0 then
    .error s!"mbp_dt must be positive and finite, got {p.mbpDt}"
  if !(Float.isFinite p.ballRadius) || p.ballRadius <= 0.0 then
    .error s!"ball radius must be positive and finite, got {p.ballRadius}"
  if !(Float.isFinite p.ballMass) || p.ballMass <= 0.0 then
    .error s!"ball mass must be positive and finite, got {p.ballMass}"
  if !(Float.isFinite p.plateMass) || p.plateMass <= 0.0 then
    .error s!"plate mass must be positive and finite, got {p.plateMass}"
  if !(Float.isFinite p.ballHydroelasticModulus) || p.ballHydroelasticModulus <= 0.0 then
    .error s!"ball hydroelastic modulus must be positive and finite, got {p.ballHydroelasticModulus}"
  if !(Float.isFinite p.floorHydroelasticModulus) || p.floorHydroelasticModulus <= 0.0 then
    .error s!"floor hydroelastic modulus must be positive and finite, got {p.floorHydroelasticModulus}"
  p.friction.validate? "ball-plate friction"

end BallPlateParams

structure FreeBodyState where
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

namespace FreeBodyState

def position (x : FreeBodyState) : Vec3 :=
  { x := x.x, y := x.y, z := x.z }

def linearVelocity (x : FreeBodyState) : Vec3 :=
  { x := x.vx, y := x.vy, z := x.vz }

def angularVelocity (x : FreeBodyState) : Vec3 :=
  { x := x.wx, y := x.wy, z := x.wz }

def isFinite (x : FreeBodyState) : Bool :=
  Float.isFinite x.x && Float.isFinite x.y && Float.isFinite x.z &&
  Float.isFinite x.roll && Float.isFinite x.pitch && Float.isFinite x.yaw &&
  Float.isFinite x.vx && Float.isFinite x.vy && Float.isFinite x.vz &&
  Float.isFinite x.wx && Float.isFinite x.wy && Float.isFinite x.wz

end FreeBodyState

structure BallPlateState where
  ball : FreeBodyState
  plate : FreeBodyState
  deriving Repr, Inhabited

namespace BallPlateState

def isFinite (x : BallPlateState) : Bool :=
  x.ball.isFinite && x.plate.isFinite

end BallPlateState

def defaultBallState (p : BallPlateParams := params) : FreeBodyState :=
  {
    x := p.x0
    y := 0.0
    z := p.z0
    vx := p.vx0
    vy := p.vy0
    vz := p.vz0
    wx := BallPlateParams.degPerSecToRadPerSec p.wxDeg0
    wy := BallPlateParams.degPerSecToRadPerSec p.wyDeg0
    wz := BallPlateParams.degPerSecToRadPerSec p.wzDeg0
  }

def defaultPlateState (p : BallPlateParams := params) : FreeBodyState :=
  { z := p.plateContactZ - 0.5 * p.plateThickness }

def defaultState (p : BallPlateParams := params) : BallPlateState :=
  {
    ball := defaultBallState p
    plate := defaultPlateState p
  }

def ballPlateContactState (p : BallPlateParams := params) : BallPlateState :=
  {
    ball := {
      defaultBallState p with
      z := p.plateContactZ + p.ballRadius - 0.005
      vz := -1.0
    }
    plate := defaultPlateState p
  }

private def positivePart (x : Float) : Float :=
  if x > 0.0 then x else 0.0

private def capHeight (radius penetration : Float) : Float :=
  if penetration <= 0.0 then
    0.0
  else if penetration >= 2.0 * radius then
    2.0 * radius
  else
    penetration

def spherePlanePatchArea (radius penetration : Float) : Float :=
  let h := capHeight radius penetration
  pi * positivePart (2.0 * radius * h - h * h)

def hydroelasticAveragePressure
    (modulus radius penetration : Float) : Float :=
  modulus * positivePart penetration / radius

def pointVelocity (body : FreeBodyState) (r : Vec3) : Vec3 :=
  body.linearVelocity.add (body.angularVelocity.cross r)

def jacobianRowForDirection (r direction : Vec3) : Array Float :=
  let angular := r.cross direction
  #[direction.x, direction.y, direction.z, angular.x, angular.y, angular.z]

def relativeJacobianRow
    (rA rB direction : Vec3)
    (includeA includeB : Bool := true) : Array Float :=
  let rowA :=
    if includeA then
      jacobianRowForDirection rA direction
    else
      Array.replicate 6 0.0
  let rowB :=
    if includeB then
      FloatArray.scale (-1.0) (jacobianRowForDirection rB direction)
    else
      Array.replicate 6 0.0
  rowA ++ rowB

def relativeVelocityAlong
    (bodyA : FreeBodyState) (rA : Vec3)
    (bodyB : FreeBodyState) (rB direction : Vec3)
    (includeA includeB : Bool := true) : Float :=
  let vA := if includeA then pointVelocity bodyA rA else {}
  let vB := if includeB then pointVelocity bodyB rB else {}
  direction.dot (vA.sub vB)

def ballPlatePenetration (p : BallPlateParams) (x : BallPlateState) : Float :=
  p.plateContactZ + p.ballRadius - x.ball.z

def ballFloorPenetration (p : BallPlateParams) (x : BallPlateState) : Float :=
  p.floorTopZ + p.ballRadius - x.ball.z

def plateFloorPatchArea (p : BallPlateParams) : Float :=
  pi * p.plateRadius * p.plateRadius

def plateFloorAveragePressure (p : BallPlateParams) : Float :=
  p.floorHydroelasticModulus * p.plateFloorPenetration / p.floorSizeZ

def ballPlatePatch? (p : BallPlateParams) (x : BallPlateState) :
    Option HydroelasticContactPatch :=
  let penetration := ballPlatePenetration p x
  let area := spherePlanePatchArea p.ballRadius penetration
  if area <= 0.0 then
    none
  else
    let n : Vec3 := { z := 1.0 }
    let tx : Vec3 := { x := 1.0 }
    let ty : Vec3 := { y := 1.0 }
    let contactPoint : Vec3 := { x := x.ball.x, y := x.ball.y, z := p.plateContactZ }
    let rBall := contactPoint.sub x.ball.position
    let rPlate := contactPoint.sub x.plate.position
    some {
      id := 4100
      bodyA := "Ball"
      bodyB := "Plate"
      complianceA := .compliant
      complianceB := .rigid
      representation := p.surfaceRepresentation
      area := area
      centroid := contactPoint.toArray
      normal := n.toArray
      averagePressure :=
        hydroelasticAveragePressure p.ballHydroelasticModulus p.ballRadius penetration
      normalVelocity := relativeVelocityAlong x.ball rBall x.plate rPlate n
      tangentVelocity := relativeVelocityAlong x.ball rBall x.plate rPlate tx
      tangentVelocity2 := relativeVelocityAlong x.ball rBall x.plate rPlate ty
      normalJacobian := relativeJacobianRow rBall rPlate n
      tangentJacobian := relativeJacobianRow rBall rPlate tx
      tangentJacobian2 := relativeJacobianRow rBall rPlate ty
      label := "ball_plate_hydroelastic_patch"
    }

def ballFloorPatch? (p : BallPlateParams) (x : BallPlateState) :
    Option HydroelasticContactPatch :=
  let penetration := ballFloorPenetration p x
  let area := spherePlanePatchArea p.ballRadius penetration
  if area <= 0.0 then
    none
  else
    let n : Vec3 := { z := 1.0 }
    let tx : Vec3 := { x := 1.0 }
    let ty : Vec3 := { y := 1.0 }
    let contactPoint : Vec3 := { x := x.ball.x, y := x.ball.y, z := p.floorTopZ }
    let rBall := contactPoint.sub x.ball.position
    some {
      id := 4200
      bodyA := "Ball"
      bodyB := "Floor"
      complianceA := .compliant
      complianceB := .compliant
      representation := p.surfaceRepresentation
      area := area
      centroid := contactPoint.toArray
      normal := n.toArray
      averagePressure :=
        hydroelasticAveragePressure p.floorHydroelasticModulus p.ballRadius penetration
      normalVelocity :=
        relativeVelocityAlong x.ball rBall ({} : FreeBodyState) ({} : Vec3) n true false
      tangentVelocity :=
        relativeVelocityAlong x.ball rBall ({} : FreeBodyState) ({} : Vec3) tx true false
      tangentVelocity2 :=
        relativeVelocityAlong x.ball rBall ({} : FreeBodyState) ({} : Vec3) ty true false
      normalJacobian := relativeJacobianRow rBall ({} : Vec3) n true false
      tangentJacobian := relativeJacobianRow rBall ({} : Vec3) tx true false
      tangentJacobian2 := relativeJacobianRow rBall ({} : Vec3) ty true false
      label := "ball_floor_hydroelastic_patch"
    }

def plateFloorPatch (p : BallPlateParams) (x : BallPlateState) :
    HydroelasticContactPatch :=
  let n : Vec3 := { z := 1.0 }
  let tx : Vec3 := { x := 1.0 }
  let ty : Vec3 := { y := 1.0 }
  let contactPoint : Vec3 := { x := x.plate.x, y := x.plate.y, z := p.floorTopZ }
  let rPlate := contactPoint.sub x.plate.position
  {
    id := 4300
    bodyA := "Plate"
    bodyB := "Floor"
    complianceA := .rigid
    complianceB := .compliant
    representation := p.surfaceRepresentation
    area := plateFloorPatchArea p
    centroid := contactPoint.toArray
    normal := n.toArray
    averagePressure := plateFloorAveragePressure p
    normalVelocity :=
      relativeVelocityAlong x.plate rPlate ({} : FreeBodyState) ({} : Vec3) n true false
    tangentVelocity :=
      relativeVelocityAlong x.plate rPlate ({} : FreeBodyState) ({} : Vec3) tx true false
    tangentVelocity2 :=
      relativeVelocityAlong x.plate rPlate ({} : FreeBodyState) ({} : Vec3) ty true false
    normalJacobian := (Array.replicate 6 0.0) ++ jacobianRowForDirection rPlate n
    tangentJacobian := (Array.replicate 6 0.0) ++ jacobianRowForDirection rPlate tx
    tangentJacobian2 := (Array.replicate 6 0.0) ++ jacobianRowForDirection rPlate ty
    label := "plate_floor_hydroelastic_patch"
  }

def hydroelasticPatches
    (p : BallPlateParams := params)
    (x : BallPlateState := defaultState p) : Array HydroelasticContactPatch := Id.run do
  let mut out : Array HydroelasticContactPatch := #[]
  match ballPlatePatch? p x with
  | some patch => out := out.push patch
  | none => pure ()
  match ballFloorPatch? p x with
  | some patch => out := out.push patch
  | none => pure ()
  out := out.push (plateFloorPatch p x)
  return out

def patchSupport
    (p : BallPlateParams := params)
    (x : BallPlateState := defaultState p) : HydroelasticPatchSupport :=
  HydroelasticPatchSupport.selectByArea p.minPatchArea (hydroelasticPatches p x)
    "hydroelastic ball-plate patch support"

structure PatchForce where
  patchId : Nat
  normalForce : Float
  generalizedForce : Array Float
  pairKind : HydroelasticPairKind
  label : String
  deriving Repr, Inhabited

def patchForces? (support : HydroelasticPatchSupport) :
    Except String (Array PatchForce) := do
  support.validateGeometry?
  support.validateJacobianWidth? 12
  let selected ← support.selectedPatches?
  pure (selected.map (fun patch =>
    {
      patchId := patch.id
      normalForce := patch.normalForce
      generalizedForce := patch.generalizedForce3D
      pairKind := patch.pairKind
      label := patch.label
    }))

def aggregateGeneralizedForce (forces : Array PatchForce) : Array Float :=
  sumGeneralizedForces (forces.map (fun force => force.generalizedForce))

def gravityGeneralizedForce (p : BallPlateParams) : Array Float :=
  #[
    0.0, 0.0, -p.ballMass * p.gravity, 0.0, 0.0, 0.0,
    0.0, 0.0, -p.plateMass * p.gravity, 0.0, 0.0, 0.0
  ]

def solidSphereInertia (p : BallPlateParams) : Float :=
  0.4 * p.ballMass * p.ballRadius * p.ballRadius

def plateIxx (p : BallPlateParams) : Float :=
  p.plateMass * (3.0 * p.plateRadius * p.plateRadius +
    p.plateThickness * p.plateThickness) / 12.0

def plateIzz (p : BallPlateParams) : Float :=
  0.5 * p.plateMass * p.plateRadius * p.plateRadius

def derivativeFromGeneralizedForce
    (p : BallPlateParams) (x : BallPlateState) (generalizedForce : Array Float) :
    BallPlateState :=
  let total := FloatArray.add generalizedForce (gravityGeneralizedForce p)
  let ib := solidSphereInertia p
  let ipx := plateIxx p
  let ipz := plateIzz p
  {
    ball := {
      x := x.ball.vx
      y := x.ball.vy
      z := x.ball.vz
      roll := x.ball.wx
      pitch := x.ball.wy
      yaw := x.ball.wz
      vx := total.getD 0 0.0 / p.ballMass
      vy := total.getD 1 0.0 / p.ballMass
      vz := total.getD 2 0.0 / p.ballMass
      wx := total.getD 3 0.0 / ib
      wy := total.getD 4 0.0 / ib
      wz := total.getD 5 0.0 / ib
    }
    plate := {
      x := x.plate.vx
      y := x.plate.vy
      z := x.plate.vz
      roll := x.plate.wx
      pitch := x.plate.wy
      yaw := x.plate.wz
      vx := total.getD 6 0.0 / p.plateMass
      vy := total.getD 7 0.0 / p.plateMass
      vz := total.getD 8 0.0 / p.plateMass
      wx := total.getD 9 0.0 / ipx
      wy := total.getD 10 0.0 / ipx
      wz := total.getD 11 0.0 / ipz
    }
  }

def contactDerivative? (p : BallPlateParams) (support : HydroelasticPatchSupport)
    (x : BallPlateState) : Except String (BallPlateState × Array PatchForce) := do
  let forces ← patchForces? support
  pure (derivativeFromGeneralizedForce p x (aggregateGeneralizedForce forces), forces)

def addScaledBody (x dx : FreeBodyState) (dt : Float) : FreeBodyState :=
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

def addScaledState (x dx : BallPlateState) (dt : Float) : BallPlateState :=
  {
    ball := addScaledBody x.ball dx.ball dt
    plate := addScaledBody x.plate dx.plate dt
  }

def eulerStep? (p : BallPlateParams) (dt : Float) (x : BallPlateState) :
    Except String (BallPlateState × HydroelasticPatchSupport × Array PatchForce × BallPlateState) := do
  let support := patchSupport p x
  let (dx, forces) ← contactDerivative? p support x
  pure (addScaledState x dx dt, support, forces, dx)

private def identityMatrix (n : Nat) : Array (Array Float) := Id.run do
  let mut rows : Array (Array Float) := #[]
  for i in [:n] do
    let mut row : Array Float := #[]
    for j in [:n] do
      row := row.push (if i == j then 1.0 else 0.0)
    rows := rows.push row
  return rows

def patchMessage (patch : HydroelasticContactPatch) : EventMessage :=
  {
    value := patch.normalForce + 0.01 * patch.area
    stateAdjoint :=
      FloatArray.add patch.normalJacobian
        (FloatArray.scale 0.05 patch.tangentJacobian)
    thetaGrad := #[patch.area, patch.averagePressure]
  }

def branchChildForPatch (weight : Float) (patch : HydroelasticContactPatch) :
    BranchChild :=
  {
    weight := weight
    resetJac := identityMatrix 12
    a := patch.normalJacobian
    message := patchMessage patch
  }

def patchBranchData? (support : HydroelasticPatchSupport) :
    Except String BranchEventData := do
  let selected ← support.selectedPatches?
  if selected.isEmpty then
    .error "hydroelastic ball-plate branch requires at least one retained patch"
  else
    let first := selected[0]!
    let weight := 1.0 / selected.size.toFloat
    pure {
      children := selected.map (branchChildForPatch weight)
      guardGrad := first.normalJacobian
      gamma := first.normalVelocity
    }

def acceptedHydroelasticSegment (p : BallPlateParams := params) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.mbpDt
    tAfter := p.mbpDt
    madeJumpAfter := true
    label := "hydroelastic ball-plate contact interval"
  }

def hydroelasticBranchVertex : VertexId := 1200

structure BallPlateResult where
  references : Array DrakeReference
  params : BallPlateParams
  defaultState : BallPlateState
  defaultSupport : HydroelasticPatchSupport
  contactState : BallPlateState
  contactSupport : HydroelasticPatchSupport
  runtimeSupport : RuntimeSupport
  patchForces : Array PatchForce
  derivative : BallPlateState
  oneStepState : BallPlateState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  branchData : BranchEventData
  branchResult : BranchAggregateResult
  deriving Repr, Inhabited

def buildEndToEnd? (p : BallPlateParams := params) : Except String BallPlateResult := do
  p.validate?
  let x0 := defaultState p
  let defaultSupport := patchSupport p x0
  let contactState := ballPlateContactState p
  let contactSupport := patchSupport p contactState
  contactSupport.validateGeometry?
  contactSupport.validateJacobianWidth? 12
  let runtime ← contactSupport.toRuntimeSupport?
  let (oneStepState, _, patchForces, derivative) ← eulerStep? p p.mbpDt contactState
  let branchData ← patchBranchData? contactSupport
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedHydroelasticSegment p))
      |>.push (.branch hydroelasticBranchVertex runtime branchData)
  trace.validate?
  let branchResult ← branchData.aggregate?
  pure {
    references := drakeReferences
    params := p
    defaultState := x0
    defaultSupport := defaultSupport
    contactState := contactState
    contactSupport := contactSupport
    runtimeSupport := runtime
    patchForces := patchForces
    derivative := derivative
    oneStepState := oneStepState
    trace := trace
    moves := trace.moves
    branchData := branchData
    branchResult := branchResult
  }

end Tyr.EventSkeleton.Examples.HydroelasticBallPlate
