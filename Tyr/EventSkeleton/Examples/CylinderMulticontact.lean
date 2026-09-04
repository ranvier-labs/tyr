import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Manipulator

/-!
# Drake-Style Cylinder Multicontact Example

This example mirrors the contact-support shape of
`../drake/examples/multibody/cylinder_with_multicontact`: a cylinder is
decorated with small collision spheres around each rim, then a runtime support
set selects the sphere contacts that are active at the ground.

The implementation is intentionally a compact event-skeleton model rather than
a monolithic MultibodyPlant port.  The important behavior for the
differentiation surface is dynamic contact-candidate generation, stable source
IDs for retained contacts, branch aggregation over the retained support, and a
coupled contact-space complementarity solve for sustained full-physics contact.
-/

namespace Tyr.EventSkeleton.Examples.CylinderMulticontact

open Tyr.EventSkeleton
open torch.DiffEq

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/cylinder_with_multicontact/populate_cylinder_plant.cc"
      concept := "registers top and bottom rim sphere collision geometries for multicontact"
    },
    {
      path := "../drake/examples/multibody/cylinder_with_multicontact/populate_cylinder_plant.h"
      concept := "declares the cylinder plant population helper and exposes the contact-sphere count/geometry contract used by the dynamic candidate provider"
    },
    {
      path := "../drake/examples/multibody/cylinder_with_multicontact/cylinder_run_dynamics.cc"
      concept := "configures a free cylinder, friction, penetration allowance, and stiction tolerance"
    },
    {
      path := "../drake/examples/multibody/cylinder_with_multicontact/test/populate_cylinder_plant_test.cc"
      concept := "checks plant dimensions and solid-cylinder inertia used by the local mass matrix"
    }
  ]

inductive ContactRim where
  | top
  | bottom
  deriving Repr, BEq, Inhabited

namespace ContactRim

def bodyZ (length : Float) : ContactRim → Float
  | .top => length / 2.0
  | .bottom => -length / 2.0

def idBase : ContactRim → Nat
  | .top => 1000
  | .bottom => 2000

def label : ContactRim → String
  | .top => "top"
  | .bottom => "bottom"

end ContactRim

structure CylinderParams where
  radius : Float := 0.05
  length : Float := 0.2
  mass : Float := 0.1
  gravity : Float := 9.81
  normalStiffness : Float := 5000.0
  normalDamping : Float := 5.0
  contactSphereRadius : Float := 0.0025
  contactsPerRim : Nat := 10
  frictionCoefficient : Float := 0.3
  penetrationAllowance : Float := 1.0e-3
  stictionTolerance : Float := 1.0e-4
  supportBudget : Nat := 4
  deriving Repr, Inhabited

namespace CylinderParams

def validate? (p : CylinderParams) : Except String Unit := do
  if !p.radius.isFinite || p.radius <= 0.0 then
    .error s!"cylinder radius must be positive and finite, got {p.radius}"
  if !p.length.isFinite || p.length <= 0.0 then
    .error s!"cylinder length must be positive and finite, got {p.length}"
  if !p.mass.isFinite || p.mass <= 0.0 then
    .error s!"cylinder mass must be positive and finite, got {p.mass}"
  if !p.gravity.isFinite || p.gravity < 0.0 then
    .error s!"cylinder gravity must be nonnegative and finite, got {p.gravity}"
  if !p.normalStiffness.isFinite || p.normalStiffness < 0.0 then
    .error s!"cylinder normal stiffness must be nonnegative and finite, got {p.normalStiffness}"
  if !p.normalDamping.isFinite || p.normalDamping < 0.0 then
    .error s!"cylinder normal damping must be nonnegative and finite, got {p.normalDamping}"
  if !p.contactSphereRadius.isFinite || p.contactSphereRadius <= 0.0 then
    .error s!"cylinder contact sphere radius must be positive and finite, got {p.contactSphereRadius}"
  if p.contactsPerRim == 0 then
    .error "cylinder contacts per rim must be positive"
  if !p.frictionCoefficient.isFinite || p.frictionCoefficient < 0.0 then
    .error s!"cylinder friction coefficient must be nonnegative and finite, got {p.frictionCoefficient}"
  if !p.penetrationAllowance.isFinite || p.penetrationAllowance < 0.0 then
    .error s!"cylinder penetration allowance must be nonnegative and finite, got {p.penetrationAllowance}"
  if !p.stictionTolerance.isFinite || p.stictionTolerance <= 0.0 then
    .error s!"cylinder stiction tolerance must be positive and finite, got {p.stictionTolerance}"
  if p.supportBudget == 0 then
    .error "cylinder support budget must be positive"

end CylinderParams

structure CylinderState where
  centerX : Float := 0.0
  centerY : Float := 0.0
  centerZ : Float
  vx : Float
  vy : Float := 0.0
  vz : Float
  wx : Float
  wy : Float := 0.0
  wz : Float := 0.0
  deriving Repr, Inhabited

namespace CylinderState

def isFinite (x : CylinderState) : Bool :=
  x.centerX.isFinite && x.centerY.isFinite && x.centerZ.isFinite &&
    x.vx.isFinite && x.vy.isFinite && x.vz.isFinite &&
    x.wx.isFinite && x.wy.isFinite && x.wz.isFinite

end CylinderState

instance : torch.DiffEq.DiffEqSpace CylinderState where
  add a b := {
    centerX := a.centerX + b.centerX
    centerY := a.centerY + b.centerY
    centerZ := a.centerZ + b.centerZ
    vx := a.vx + b.vx
    vy := a.vy + b.vy
    vz := a.vz + b.vz
    wx := a.wx + b.wx
    wy := a.wy + b.wy
    wz := a.wz + b.wz
  }
  sub a b := {
    centerX := a.centerX - b.centerX
    centerY := a.centerY - b.centerY
    centerZ := a.centerZ - b.centerZ
    vx := a.vx - b.vx
    vy := a.vy - b.vy
    vz := a.vz - b.vz
    wx := a.wx - b.wx
    wy := a.wy - b.wy
    wz := a.wz - b.wz
  }
  scale s x := {
    centerX := s * x.centerX
    centerY := s * x.centerY
    centerZ := s * x.centerZ
    vx := s * x.vx
    vy := s * x.vy
    vz := s * x.vz
    wx := s * x.wx
    wy := s * x.wy
    wz := s * x.wz
  }

private def max9
    (a b c d e f g h i : Float) : Float :=
  max a (max b (max c (max d (max e (max f (max g (max h i)))))))

instance : torch.DiffEq.DiffEqSeminorm CylinderState where
  rms x :=
    max9
      (Float.abs x.centerX)
      (Float.abs x.centerY)
      (Float.abs x.centerZ)
      (Float.abs x.vx)
      (Float.abs x.vy)
      (Float.abs x.vz)
      (Float.abs x.wx)
      (Float.abs x.wy)
      (Float.abs x.wz)

instance : torch.DiffEq.DiffEqElem CylinderState where
  abs x := {
    centerX := Float.abs x.centerX
    centerY := Float.abs x.centerY
    centerZ := Float.abs x.centerZ
    vx := Float.abs x.vx
    vy := Float.abs x.vy
    vz := Float.abs x.vz
    wx := Float.abs x.wx
    wy := Float.abs x.wy
    wz := Float.abs x.wz
  }
  max a b := {
    centerX := max a.centerX b.centerX
    centerY := max a.centerY b.centerY
    centerZ := max a.centerZ b.centerZ
    vx := max a.vx b.vx
    vy := max a.vy b.vy
    vz := max a.vz b.vz
    wx := max a.wx b.wx
    wy := max a.wy b.wy
    wz := max a.wz b.wz
  }
  addScalar s x := {
    centerX := x.centerX + s
    centerY := x.centerY + s
    centerZ := x.centerZ + s
    vx := x.vx + s
    vy := x.vy + s
    vz := x.vz + s
    wx := x.wx + s
    wy := x.wy + s
    wz := x.wz + s
  }
  div a b := {
    centerX := a.centerX / b.centerX
    centerY := a.centerY / b.centerY
    centerZ := a.centerZ / b.centerZ
    vx := a.vx / b.vx
    vy := a.vy / b.vy
    vz := a.vz / b.vz
    wx := a.wx / b.wx
    wy := a.wy / b.wy
    wz := a.wz / b.wz
  }

structure ContactPoint where
  rim : ContactRim
  index : Nat
  rx : Float
  ry : Float
  rz : Float
  deriving Repr, Inhabited

def params : CylinderParams := {}

def initialCenterZ : Float := 0.5
def initialVx : Float := 1.0
def initialVz : Float := -1.0
def initialWx : Float := 0.1

def initialState : CylinderState :=
  {
    centerX := 0.0
    centerY := 0.0
    centerZ := initialCenterZ
    vx := initialVx
    vz := initialVz
    wx := initialWx
  }

def bottomSurfaceDistance (p : CylinderParams) (centerZ : Float) : Float :=
  centerZ - p.length / 2.0 - p.contactSphereRadius

def impactTime (p : CylinderParams := params) : Float :=
  let d0 := bottomSurfaceDistance p initialCenterZ
  (initialVz + Float.sqrt (initialVz * initialVz + 2.0 * p.gravity * d0)) / p.gravity

def preContactState (p : CylinderParams := params) : CylinderState :=
  let t := impactTime p
  {
    centerX := initialVx * t
    centerY := 0.0
    centerZ := initialCenterZ + initialVz * t - 0.5 * p.gravity * t * t
    vx := initialVx
    vz := initialVz - p.gravity * t
    wx := initialWx
  }

def thetaForIndex (p : CylinderParams) (i : Nat) : Float :=
  2.0 * pi * i.toFloat / p.contactsPerRim.toFloat

def contactPoint (p : CylinderParams) (rim : ContactRim) (i : Nat) : ContactPoint :=
  let theta := thetaForIndex p i
  {
    rim := rim
    index := i
    rx := Float.cos theta * p.radius
    ry := Float.sin theta * p.radius
    rz := rim.bodyZ p.length
  }

def signedDistance (p : CylinderParams) (x : CylinderState) (pt : ContactPoint) : Float :=
  x.centerZ + pt.rz - p.contactSphereRadius

def normalVelocity (x : CylinderState) (pt : ContactPoint) : Float :=
  x.vz + x.wx * pt.ry - x.wy * pt.rx

def tangentVelocityX (x : CylinderState) (pt : ContactPoint) : Float :=
  x.vx + x.wy * pt.rz - x.wz * pt.ry

def normalJacobian (pt : ContactPoint) : Array Float :=
  #[0.0, 0.0, 1.0, pt.ry, -pt.rx, 0.0]

def tangentJacobianX (pt : ContactPoint) : Array Float :=
  #[1.0, 0.0, 0.0, 0.0, pt.rz, -pt.ry]

def candidateForPoint (p : CylinderParams) (x : CylinderState) (pt : ContactPoint) :
    ContactCandidate :=
  let base := pt.rim.idBase
  {
    id := base + pt.index
    signedDistance := signedDistance p x pt
    normalVelocity := normalVelocity x pt
    tangentVelocity := tangentVelocityX x pt
    normalJacobian := normalJacobian pt
    tangentJacobian := tangentJacobianX pt
    label := s!"collision_{pt.rim.label}_{pt.index}"
  }

def contactCandidateBatch
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) : PackedContactCandidateBatch := Id.run do
  let mut ids : Array Nat := #[]
  let mut signedDistances : Array Float := #[]
  let mut normalVelocities : Array Float := #[]
  let mut tangentVelocities : Array Float := #[]
  let mut normalJacobians : Array (Array Float) := #[]
  let mut tangentJacobians : Array (Array Float) := #[]
  let mut labels : Array String := #[]
  for i in [:p.contactsPerRim] do
    for rim in #[ContactRim.top, ContactRim.bottom] do
      let pt := contactPoint p rim i
      ids := ids.push (rim.idBase + pt.index)
      signedDistances := signedDistances.push (signedDistance p x pt)
      normalVelocities := normalVelocities.push (normalVelocity x pt)
      tangentVelocities := tangentVelocities.push (tangentVelocityX x pt)
      normalJacobians := normalJacobians.push (normalJacobian pt)
      tangentJacobians := tangentJacobians.push (tangentJacobianX pt)
      labels := labels.push s!"collision_{rim.label}_{pt.index}"
  return {
    ids := ids
    signedDistance := signedDistances
    normalVelocity := normalVelocities
    tangentVelocity := tangentVelocities
    normalJacobian := normalJacobians
    tangentJacobian := tangentJacobians
    labels := labels
    sourceCandidateCount? := some ids.size
    label := "packed cylinder rim contact candidates"
  }

def contactCandidateSet?
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) :
    Except String ContactCandidateSet :=
  (contactCandidateBatch p x).toCandidateSet? "cylinder rim contact candidates"

def contactCandidates?
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) :
    Except String (Array ContactCandidate) := do
  let set ← contactCandidateSet? p x
  pure set.candidates

def candidatesForRim (p : CylinderParams) (x : CylinderState) (rim : ContactRim) :
    Array ContactCandidate := Id.run do
  let mut out : Array ContactCandidate := #[]
  for i in [:p.contactsPerRim] do
    out := out.push (candidateForPoint p x (contactPoint p rim i))
  return out

def contactCandidates (p : CylinderParams := params) (x : CylinderState := preContactState p) :
    Array ContactCandidate :=
  match contactCandidates? p x with
  | .ok candidates => candidates
  | .error _ => #[]

def allActiveSupport?
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) : Except String ContactSupport := do
  let set ←
    (contactCandidateBatch p x).retainedByDistance? p.penetrationAllowance
      "packed all penetration-allowance cylinder rim contacts"
  pure <|
    set.selectWithPolicy (.threshold p.penetrationAllowance)
      "all penetration-allowance cylinder rim contacts"
      |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

def allActiveSupport
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) : ContactSupport :=
  match allActiveSupport? p x with
  | .ok support => support
  | .error _ =>
      ContactSupport.selectByDistance p.penetrationAllowance (contactCandidates p x)
        "all penetration-allowance cylinder rim contacts"
        |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

def budgetedSupport?
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) : Except String ContactSupport := do
  let set ←
    (contactCandidateBatch p x).retainedClosestK? p.supportBudget
      "packed budgeted closest cylinder rim contacts"
  pure <|
    set.selectWithPolicy (.topK p.supportBudget)
      "budgeted closest cylinder rim contacts"
      |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

def budgetedSupport
    (p : CylinderParams := params)
    (x : CylinderState := preContactState p) : ContactSupport :=
  match budgetedSupport? p x with
  | .ok support => support
  | .error _ =>
      ContactSupport.selectClosestK p.supportBudget (contactCandidates p x)
        "budgeted closest cylinder rim contacts"
        |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

inductive CylinderSupportMode where
  | allActive
  | budgeted
  deriving Repr, BEq, Inhabited

namespace CylinderSupportMode

def label : CylinderSupportMode → String
  | .allActive => "all-active"
  | .budgeted => "budgeted"

def support?
    (mode : CylinderSupportMode)
    (p : CylinderParams)
    (x : CylinderState) : Except String ContactSupport :=
  match mode with
  | .allActive => allActiveSupport? p x
  | .budgeted => budgetedSupport? p x

end CylinderSupportMode

private def positivePart (x : Float) : Float :=
  if x > 0.0 then x else 0.0

private def signedUnit (x : Float) : Float :=
  if x > 0.0 then 1.0 else if x < 0.0 then -1.0 else 0.0

private def identityMatrix (n : Nat) : Array (Array Float) := Id.run do
  let mut rows : Array (Array Float) := #[]
  for i in [:n] do
    let mut row : Array Float := #[]
    for j in [:n] do
      row := row.push (if i == j then 1.0 else 0.0)
    rows := rows.push row
  return rows

structure ContactForce where
  candidateId : Nat
  normalForce : Float
  tangentForceX : Float
  torqueX : Float
  torqueY : Float
  torqueZ : Float
  mode : ContactMode
  deriving Repr, Inhabited

structure RigidBodyWrench where
  fx : Float := 0.0
  fy : Float := 0.0
  fz : Float := 0.0
  tx : Float := 0.0
  ty : Float := 0.0
  tz : Float := 0.0
  deriving Repr, Inhabited

namespace RigidBodyWrench

def add (a b : RigidBodyWrench) : RigidBodyWrench :=
  {
    fx := a.fx + b.fx
    fy := a.fy + b.fy
    fz := a.fz + b.fz
    tx := a.tx + b.tx
    ty := a.ty + b.ty
    tz := a.tz + b.tz
  }

def fromContact (force : ContactForce) : RigidBodyWrench :=
  {
    fx := force.tangentForceX
    fz := force.normalForce
    tx := force.torqueX
    ty := force.torqueY
    tz := force.torqueZ
  }

def asArray (wrench : RigidBodyWrench) : Array Float :=
  #[wrench.fx, wrench.fy, wrench.fz, wrench.tx, wrench.ty, wrench.tz]

end RigidBodyWrench

def solidCylinderIxx (p : CylinderParams) : Float :=
  p.mass * (3.0 * p.radius * p.radius + p.length * p.length) / 12.0

def solidCylinderIzz (p : CylinderParams) : Float :=
  0.5 * p.mass * p.radius * p.radius

def massMatrix (p : CylinderParams) : Array (Array Float) :=
  FloatMatrix.diagonal
    #[p.mass, p.mass, p.mass, solidCylinderIxx p, solidCylinderIxx p, solidCylinderIzz p]

def velocityVector (x : CylinderState) : Array Float :=
  #[x.vx, x.vy, x.vz, x.wx, x.wy, x.wz]

def gravityGeneralizedForce (p : CylinderParams) : Array Float :=
  #[0.0, 0.0, -p.mass * p.gravity, 0.0, 0.0, 0.0]

def restingState (p : CylinderParams := params) : CylinderState :=
  {
    centerX := 0.0
    centerY := 0.0
    centerZ := p.length / 2.0 + p.contactSphereRadius
    vx := 0.0
    vy := 0.0
    vz := 0.0
    wx := 0.0
    wy := 0.0
    wz := 0.0
  }

def normalBiasForCandidate (_x : CylinderState) (_candidate : ContactCandidate) : Float :=
  0.0

def derivativeFromVelocityAcceleration (x : CylinderState) (vdot : Array Float) :
    CylinderState :=
  {
    centerX := x.vx
    centerY := x.vy
    centerZ := x.vz
    vx := vdot.getD 0 0.0
    vy := vdot.getD 1 0.0
    vz := vdot.getD 2 0.0
    wx := vdot.getD 3 0.0
    wy := vdot.getD 4 0.0
    wz := vdot.getD 5 0.0
  }

def contactForceFromComponents
    (candidate : ContactCandidate) (normalForce tangentForceX : Float) :
    ContactForce :=
  let rx := candidate.normalJacobian.getD 4 0.0 * (-1.0)
  let ry := candidate.normalJacobian.getD 3 0.0
  let rz := candidate.tangentJacobian.getD 4 0.0
  {
    candidateId := candidate.id
    normalForce := normalForce
    tangentForceX := tangentForceX
    torqueX := ry * normalForce
    torqueY := rz * tangentForceX - rx * normalForce
    torqueZ := -ry * tangentForceX
    mode := candidate.mode
  }

def contactForceScalars (force : ContactForce) : ContactForceScalars :=
  {
    candidateId := force.candidateId
    normalForce := force.normalForce
    tangentForce := force.tangentForceX
    mode := force.mode
    label := s!"cylinder contact {force.candidateId}"
  }

def contactForceForCandidate (p : CylinderParams) (candidate : ContactCandidate) :
    ContactForce :=
  let penetration := positivePart (-candidate.signedDistance)
  let closure := positivePart (-candidate.normalVelocity)
  let normalForce := positivePart (p.normalStiffness * penetration + p.normalDamping * closure)
  let frictionLimit := p.frictionCoefficient * normalForce
  let tangentForceX :=
    match candidate.mode with
    | .separated => 0.0
    | .sticking =>
        let desired := -candidate.tangentVelocity * p.mass / p.stictionTolerance
        if Float.abs desired <= frictionLimit then desired else -frictionLimit * signedUnit candidate.tangentVelocity
    | .impacting => -frictionLimit * signedUnit candidate.tangentVelocity
    | .sliding => -frictionLimit * signedUnit candidate.tangentVelocity
  contactForceFromComponents candidate normalForce tangentForceX

def contactForces? (p : CylinderParams) (support : ContactSupport) :
    Except String (Array ContactForce) := do
  let selected ← support.selectedCandidates?
  pure (selected.map (contactForceForCandidate p))

def aggregateContactWrench (forces : Array ContactForce) : RigidBodyWrench :=
  forces.foldl
    (fun acc force => acc.add (RigidBodyWrench.fromContact force))
    {}

def physicsDerivativeFromWrench (p : CylinderParams) (x : CylinderState)
    (wrench : RigidBodyWrench) : CylinderState :=
  let ixx := solidCylinderIxx p
  let izz := solidCylinderIzz p
  {
    centerX := x.vx
    centerY := x.vy
    centerZ := x.vz
    vx := wrench.fx / p.mass
    vy := wrench.fy / p.mass
    vz := wrench.fz / p.mass - p.gravity
    wx := wrench.tx / ixx
    wy := wrench.ty / ixx
    wz := wrench.tz / izz
  }

def cylinderFullPhysicsIntervalVertex : VertexId := 301

def validateFullPhysicsInputs?
    (p : CylinderParams) (x : CylinderState) : Except String Unit := do
  p.validate?
  if !x.isFinite then
    .error "cylinder full physics state must have finite coordinates"

def fullPhysicsPrimitives?
    (p : CylinderParams) (support : ContactSupport) (x : CylinderState) :
    Except String (FullPhysicsPrimitives × Array ContactForce) := do
  validateFullPhysicsInputs? p x
  support.validateJacobianWidth? 6
  let forces ← contactForces? p support
  pure ({
    massMatrix := massMatrix p
    qdot := velocityVector x
    actuationForces := Array.replicate 6 0.0
    generalizedForceContributions :=
      #[GeneralizedForceContribution.ofForce
          (gravityGeneralizedForce p)
          "cylinder gravity generalized force"
          "CylinderMulticontact"]
    contactCandidates := support.candidates
    sourceContactCandidateCount? := some support.totalCandidates
    supportPolicy := support.policy
    contactForceSource := .precomputed
    contactForces := forces.map contactForceScalars
    distanceTol := p.penetrationAllowance
    tangentVelocityTol := p.stictionTolerance
    label := "cylinder multicontact full physics"
  }, forces)

def fullPhysicsPrimitiveProvider
    (p : CylinderParams := params)
    (supportMode : CylinderSupportMode := .budgeted)
    (label : String := "cylinder multicontact full physics provider") :
    FullPhysicsPrimitiveProvider CylinderState :=
  {
    label := label
    primitivesAt? := fun x => do
      validateFullPhysicsInputs? p x
      let support ← supportMode.support? p x
      let (primitives, _) ← fullPhysicsPrimitives? p support x
      pure { primitives with label := s!"{label}:{supportMode.label}" }
  }

def solveFullPhysics?
    (p : CylinderParams) (support : ContactSupport) (x : CylinderState) :
    Except String (FullPhysicsResult × Array ContactForce) := do
  let (primitive, forces) ← fullPhysicsPrimitives? p support x
  let equation ← primitive.equation?
  let fullPhysics ← equation.solve? cylinderFullPhysicsIntervalVertex
  pure (fullPhysics, forces)

def physicsDerivative? (p : CylinderParams) (support : ContactSupport) (x : CylinderState) :
    Except String CylinderState := do
  let (fullPhysics, _) ← solveFullPhysics? p support x
  pure (derivativeFromVelocityAcceleration x fullPhysics.derivative.vdot)

structure CylinderSustainedContactSolve where
  state : CylinderState
  support : ContactSupport
  runtimeSupport : RuntimeSupport
  problem : NormalContactLcpProblem
  lcpResult : NormalContactLcpResult
  contactForces : Array ContactForce
  aggregateWrench : RigidBodyWrench
  derivative : CylinderState
  moves : Array SkeletonMove := #[]
  deriving Repr, Inhabited

private def contactForceFromNormalLcp
    (p : CylinderParams) (candidate : ContactCandidate) (normalForce : Float) :
    ContactForce :=
  let frictionLimit := p.frictionCoefficient * normalForce
  let tangentForceX :=
    match candidate.mode with
    | .separated => 0.0
    | .sticking =>
        let desired := -candidate.tangentVelocity * p.mass / p.stictionTolerance
        if Float.abs desired <= frictionLimit then desired else -frictionLimit * signedUnit candidate.tangentVelocity
    | .impacting => -frictionLimit * signedUnit candidate.tangentVelocity
    | .sliding => -frictionLimit * signedUnit candidate.tangentVelocity
  contactForceFromComponents candidate normalForce tangentForceX

def sustainedContactProblem?
    (p : CylinderParams) (x : CylinderState) (support : ContactSupport)
    (applied : RigidBodyWrench := {}) :
    Except String (RuntimeSupport × NormalContactLcpProblem) := do
  support.validateJacobianWidth? 6
  let runtime ← support.toRuntimeSupport?
  let selected ← support.selectedCandidates?
  let normalRows := selected.map (fun candidate => candidate.normalJacobian)
  let normalBias := selected.map (normalBiasForCandidate x)
  pure
    (runtime, {
      massMatrix := massMatrix p
      normalJacobian := normalRows
      generalizedForces := FloatArray.add (gravityGeneralizedForce p) applied.asArray
      normalBias := normalBias
      label := "cylinder multicontact sustained normal LCP"
    })

def solveSustainedContact?
    (p : CylinderParams := params)
    (x : CylinderState := restingState p)
    (support : ContactSupport := allActiveSupport p x)
    (applied : RigidBodyWrench := {}) :
    Except String CylinderSustainedContactSolve := do
  let (runtime, problem) ← sustainedContactProblem? p x support applied
  let lcp ← problem.solve? 1.0e-7
  let selected ← support.selectedCandidates?
  if selected.size != lcp.normalForces.size then
    .error s!"cylinder sustained contact: selected contact count {selected.size} != LCP force count {lcp.normalForces.size}"
  let mut forces : Array ContactForce := #[]
  for i in [:selected.size] do
    forces := forces.push
      (contactForceFromNormalLcp p selected[i]! (lcp.normalForces.getD i 0.0))
  let wrench := aggregateContactWrench forces
  pure {
    state := x
    support := support
    runtimeSupport := runtime
    problem := problem
    lcpResult := lcp
    contactForces := forces
    aggregateWrench := wrench
    derivative := derivativeFromVelocityAcceleration x lcp.acceleration
    moves := #[
      {
        kind := .branchAggregate
        reads := runtime.selectedIds
        exactness := runtime.exactness
        label := "cylinder dynamic rim contact support"
      },
      {
        kind := .localSchurBlock
        reads := runtime.selectedIds
        exactness := .exact
        label := "cylinder sustained normal-contact LCP solve"
      }
    ]
  }

def freeFlightDerivative (p : CylinderParams) (x : CylinderState) : CylinderState :=
  {
    centerX := x.vx
    centerY := x.vy
    centerZ := x.vz
    vx := 0.0
    vy := 0.0
    vz := -p.gravity
    wx := 0.0
    wy := 0.0
    wz := 0.0
  }

def freeFlightTerm (p : CylinderParams) : ODETerm CylinderState Unit :=
  { vectorField := fun _t x _ => freeFlightDerivative p x }

def minimumCandidateDistance? (p : CylinderParams) (x : CylinderState) : Option Float :=
  match (contactCandidateBatch p x).minimumSignedDistance? with
  | .ok d? => d?
  | .error _ =>
      (contactCandidates p x).foldl
        (fun acc candidate =>
          match acc with
          | none => some candidate.signedDistance
          | some d => some (if candidate.signedDistance < d then candidate.signedDistance else d))
        none

def minimumCandidateDistance (p : CylinderParams) (x : CylinderState) : Float :=
  (minimumCandidateDistance? p x).getD 0.0

def contactAwareDerivative (p : CylinderParams) (x : CylinderState) : CylinderState :=
  if minimumCandidateDistance p x <= 0.0 then
    let support := budgetedSupport p x
    match physicsDerivative? p support x with
    | .ok dx => dx
    | .error _ => freeFlightDerivative p x
  else
    freeFlightDerivative p x

def contactAwareTerm (p : CylinderParams) : ODETerm CylinderState Unit :=
  { vectorField := fun _t x _ => contactAwareDerivative p x }

def firstContactEvent (p : CylinderParams) : EventSpec CylinderState Unit :=
  {
    condition := .real (fun _t x _ => minimumCandidateDistance p x)
    direction := some false
    terminate := true
    rootTol := 1.0e-7
  }

def addScaledState (x dx : CylinderState) (dt : Float) : CylinderState :=
  torch.DiffEq.DiffEqSpace.add x (torch.DiffEq.DiffEqSpace.scale dt dx)

def eulerPhysicsStep? (p : CylinderParams) (dt : Float) (x : CylinderState) :
    Except String (CylinderState × ContactSupport × Array ContactForce × CylinderState) := do
  let support ← budgetedSupport? p x
  let (fullPhysics, forces) ← solveFullPhysics? p support x
  let dx := derivativeFromVelocityAcceleration x fullPhysics.derivative.vdot
  pure (addScaledState x dx dt, fullPhysics.support, forces, dx)

def simulatePhysicsSteps? (p : CylinderParams) (dt : Float) (steps : Nat) (x0 : CylinderState) :
    Except String CylinderState := do
  let mut x := x0
  for _ in [:steps] do
    let (xNext, _, _, _) ← eulerPhysicsStep? p dt x
    x := xNext
  pure x

def candidateMessage (candidate : ContactCandidate) : EventMessage :=
  let closure := positivePart (-candidate.normalVelocity)
  let slip := Float.abs candidate.tangentVelocity
  let modeBonus :=
    match candidate.mode with
    | .impacting => 1.0
    | .sticking => 0.5
    | .sliding => 0.25
    | .separated => 0.0
  {
    value := closure + 0.01 * slip + modeBonus
    stateAdjoint :=
      FloatArray.add candidate.normalJacobian
        (FloatArray.scale 0.1 candidate.tangentJacobian)
    thetaGrad := #[closure * params.frictionCoefficient]
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
    .error "cylinder multicontact branch requires at least one retained contact"
  else
    let first := selected[0]!
    let weight := 1.0 / selected.size.toFloat
    let children := selected.map (branchChildForCandidate weight)
    pure {
      children := children
      guardGrad := first.normalJacobian
      gamma := first.normalVelocity
    }

def acceptedContactSegment (p : CylinderParams := params) : AcceptedStepSegment :=
  let tau := impactTime p
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := tau + 0.05
    tAfter := tau
    madeJumpAfter := true
    label := "cylinder multicontact free-fall interval"
  }

def contactBranchVertex : VertexId := 300

structure DiffEqContactRun where
  eventTime : Float
  eventState : CylinderState
  postStepState : CylinderState
  support : ContactSupport
  runtimeSupport : RuntimeSupport
  contactForces : Array ContactForce
  fullPhysics : FullPhysicsResult
  derivative : CylinderState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def solveToFirstContact? (p : CylinderParams := params) :
    Except String DiffEqContactRun := do
  let term := freeFlightTerm p
  let solver :=
    RK4.solver
      (Term := ODETerm CylinderState Unit)
      (Y := CylinderState)
      (VF := CylinderState)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm CylinderState Unit)
      (Y := CylinderState)
      (VF := CylinderState)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 (impactTime p + 0.1) (some 0.02) initialState ()
      (saveat := { t1 := true })
      (event := some (firstContactEvent p))
  if sol.result != Result.eventOccurred then
    .error s!"expected first contact event, got {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "first contact solve did not save event endpoint"
        else
          let t := ts[ts.size - 1]!
          let x := ys[ys.size - 1]!
          let support ← budgetedSupport? p x
          support.validateJacobianWidth? 6
          let (fullPhysics, contactForces) ← solveFullPhysics? p support x
          let runtime ← fullPhysics.support.toRuntimeSupport?
          let derivative := derivativeFromVelocityAcceleration x fullPhysics.derivative.vdot
          let postStepState := addScaledState x derivative 1.0e-3
          let branchData ← contactBranchData? fullPhysics.support
          let firstContactSegment : AcceptedStepSegment := {
            id := 0
            attemptIndex := 0
            tStart := 0.0
            tAttempt := impactTime p + 0.1
            tAfter := t
            madeJumpAfter := true
            label := "diffeq-localized cylinder first contact"
          }
          let trace :=
            DynamicEventTrace.empty
              |>.push (.interval firstContactSegment)
              |>.push (.branch contactBranchVertex runtime branchData)
          trace.validate?
          pure {
            eventTime := t
            eventState := x
            postStepState := postStepState
            support := fullPhysics.support
            runtimeSupport := runtime
            contactForces := contactForces
            fullPhysics := fullPhysics
            derivative := derivative
            trace := trace
            moves := trace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
          }
    | _, _ => .error "first contact solve did not save endpoint arrays"

structure CylinderMulticontactResult where
  references : Array DrakeReference
  state : CylinderState
  allSupport : ContactSupport
  budgetedSupport : ContactSupport
  runtimeSupport : RuntimeSupport
  contactForces : Array ContactForce
  fullPhysics : FullPhysicsResult
  derivative : CylinderState
  oneStepState : CylinderState
  rolloutState : CylinderState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  branchData : BranchEventData
  branchResult : BranchAggregateResult
  diffEqRun : DiffEqContactRun
  sustainedContact : CylinderSustainedContactSolve
  deriving Repr, Inhabited

def buildEndToEnd? : Except String CylinderMulticontactResult := do
  let x := preContactState params
  let allSupport ← allActiveSupport? params x
  let budgeted ← budgetedSupport? params x
  allSupport.validateJacobianWidth? 6
  budgeted.validateJacobianWidth? 6
  let runtime ← budgeted.toRuntimeSupport?
  let (fullPhysics, contactForces) ← solveFullPhysics? params budgeted x
  let derivative := derivativeFromVelocityAcceleration x fullPhysics.derivative.vdot
  let (oneStepState, _, _, _) ← eulerPhysicsStep? params 1.0e-3 x
  let rolloutState ← simulatePhysicsSteps? params 1.0e-3 10 x
  let branchData ← contactBranchData? budgeted
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedContactSegment params))
      |>.push (.branch contactBranchVertex runtime branchData)
  trace.validate?
  let branchResult ← branchData.aggregate?
  let diffEqRun ← solveToFirstContact? params
  let resting := restingState params
  let restingSupport ← allActiveSupport? params resting
  let sustained ← solveSustainedContact? params resting restingSupport
  pure {
    references := drakeReferences
    state := x
    allSupport := allSupport
    budgetedSupport := budgeted
    runtimeSupport := runtime
    contactForces := contactForces
    fullPhysics := fullPhysics
    derivative := derivative
    oneStepState := oneStepState
    rolloutState := rolloutState
    trace := trace
    moves := trace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
    branchData := branchData
    branchResult := branchResult
    diffEqRun := diffEqRun
    sustainedContact := sustained
  }

end Tyr.EventSkeleton.Examples.CylinderMulticontact
