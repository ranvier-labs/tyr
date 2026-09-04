import Tyr.EventSkeleton.Branch
import Tyr.EventSkeleton.Contact
import Tyr.EventSkeleton.Manipulator

/-!
# Drake-Style Simple Gripper Example

This ports the physics boundary of `../drake/examples/simple_gripper`: an
SDF-defined two-finger gripper, a mug cylinder, generated ring-pad collision
spheres, a coupler constraint between the finger sliders, sine actuation, and
runtime contact support selection.

The provider below is analytic, but the contract is the important part.  A
plant/provider layer supplies joint metadata, generated collision geometry,
coupler constraints, and contact candidates.  The dynamics layer consumes only
the reusable EventSkeleton primitives: `ContactCandidate`, `ContactSupport`,
`CouplerConstraint`, `ManipulatorEquation`, and `FullPhysicsPrimitives`.
-/

namespace Tyr.EventSkeleton.Examples.SimpleGripper

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/simple_gripper/simple_gripper.cc"
      concept := "builds a MultibodyPlant and SceneGraph, loads the gripper and mug SDFs, adds ring-pad collision spheres, adds a coupler constraint, and connects sine actuation"
    },
    {
      path := "../drake/examples/simple_gripper/simple_gripper.sdf"
      concept := "defines the gripper links, prismatic translate joint, left/right finger sliders, link masses, and visual finger geometry"
    },
    {
      path := "../drake/examples/simple_gripper/simple_mug.sdf"
      concept := "defines the free mug cylinder, mass properties, friction, and visual handle"
    },
    {
      path := "../drake/examples/simple_gripper/BUILD.bazel"
      concept := "declares the simple_gripper executable and SDF data dependencies"
    },
    {
      path := "../drake/examples/simple_gripper/README.md"
      concept := "describes the demo as a gripper-contact scenario using SceneGraph and MultibodyPlant"
    }
  ]

def simpleGripperModelUri : String :=
  "package://drake/examples/simple_gripper/simple_gripper.sdf"

def simpleMugModelUri : String :=
  "package://drake/examples/simple_gripper/simple_mug.sdf"

def combinedModelUri : String :=
  simpleGripperModelUri ++ " + " ++ simpleMugModelUri

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

def norm (v : Vec3) : Float :=
  Float.sqrt (v.dot v)

def toArray (v : Vec3) : Array Float :=
  #[v.x, v.y, v.z]

end Vec3

structure Rpy where
  roll : Float := 0.0
  pitch : Float := 0.0
  yaw : Float := 0.0
  deriving Repr, BEq, Inhabited

inductive JointType where
  | fixed
  | prismatic
  deriving Repr, BEq, Inhabited

structure SdfLinkRecord where
  name : String
  mass : Float := 0.0
  visualShape : String := ""
  deriving Repr, BEq, Inhabited

structure SdfJointRecord where
  name : String
  jointType : JointType
  parent : String
  child : String
  axis : Vec3 := {}
  effortLimit : Float := 0.0
  deriving Repr, BEq, Inhabited

structure SdfPlantSummary where
  modelNames : Array String := #[]
  linkRecords : Array SdfLinkRecord := #[]
  jointRecords : Array SdfJointRecord := #[]
  actuatorNames : Array String := #[]
  coupler : CouplerConstraint := {}
  deriving Repr, Inhabited

def linkRecords : Array SdfLinkRecord :=
  #[
    { name := "y_translate_link", mass := 0.0001 },
    { name := "body", mass := 0.988882, visualShape := "box(0.146,0.0725,0.049521)" },
    { name := "left_finger", mass := 0.05, visualShape := "box(0.007,0.081,0.028)" },
    { name := "right_finger", mass := 0.05, visualShape := "box(0.007,0.081,0.028)" },
    { name := "simple_mug", mass := 0.094, visualShape := "cylinder(radius=0.04,length=0.1)" }
  ]

def jointRecords : Array SdfJointRecord :=
  #[
    {
      name := "weld_base"
      jointType := .fixed
      parent := "world"
      child := "y_translate_link"
    },
    {
      name := "translate_joint"
      jointType := .prismatic
      parent := "y_translate_link"
      child := "body"
      axis := { x := 0.0, y := 0.0, z := 1.0 }
      effortLimit := 500.0
    },
    {
      name := "left_slider"
      jointType := .prismatic
      parent := "body"
      child := "left_finger"
      axis := { x := 1.0, y := 0.0, z := 0.0 }
      effortLimit := 500.0
    },
    {
      name := "right_slider"
      jointType := .prismatic
      parent := "body"
      child := "right_finger"
      axis := { x := 1.0, y := 0.0, z := 0.0 }
      effortLimit := 0.0
    }
  ]

structure SimpleGripperParams where
  simulationTime : Float := 10.0
  gripWidth : Float := 0.095
  mbpDiscreteUpdatePeriod : Float := 1.0e-3
  penetrationAllowance : Float := 1.0e-2
  stictionTolerance : Float := 1.0e-2
  ringSamples : Nat := 8
  ringOrientDegrees : Float := 0.0
  ringFriction : CoulombFriction :=
    { staticFriction := 1.0, dynamicFriction := 0.5 }
  mugFriction : CoulombFriction :=
    { staticFriction := 0.9, dynamicFriction := 0.5 }
  rxDegrees : Float := 0.0
  ryDegrees : Float := 0.0
  rzDegrees : Float := 0.0
  gripForce : Float := 10.0
  amplitude : Float := 0.15
  frequency : Float := 2.0
  contactApproximation : String := "sap"
  couplerGearRatio : Float := -1.0
  padMajorRadius : Float := 14.0e-3
  padMinorRadius : Float := 6.0e-3
  padOffset : Float := 0.0046
  padTorusCenterY : Float := 0.0265
  fingerHalfGapAtZero : Float := 0.0105
  fingerWidth : Float := 0.007
  mugRadius : Float := 0.04
  mugMass : Float := 0.094
  gripperMovingMass : Float := 1.0890
  fingerMass : Float := 0.05
  graspPenetration : Float := 2.0e-3
  normalStiffness : Float := 8.0e4
  normalDamping : Float := 20.0
  deriving Repr, Inhabited

def params : SimpleGripperParams := {}

def gripperCoupler (p : SimpleGripperParams := params) : CouplerConstraint :=
  {
    constrainedName := "left_slider"
    referenceName := "right_slider"
    gearRatio := p.couplerGearRatio
    label := "left_slider = rho * right_slider"
  }

def plantSummary (p : SimpleGripperParams := params) : SdfPlantSummary :=
  {
    modelNames := #["simple_gripper", "simple_mug"]
    linkRecords := linkRecords
    jointRecords := jointRecords
    actuatorNames := #["translate_joint", "left_slider"]
    coupler := gripperCoupler p
  }

structure SimpleGripperModelAssetBoundary where
  gripperModelName : String := "simple_gripper"
  mugModelName : String := "simple_mug"
  gripperSdfPath : String := "../drake/examples/simple_gripper/simple_gripper.sdf"
  mugSdfPath : String := "../drake/examples/simple_gripper/simple_mug.sdf"
  gripperPackageUri : String := simpleGripperModelUri
  mugPackageUri : String := simpleMugModelUri
  linkNames : Array String :=
    #["y_translate_link", "body", "left_finger", "right_finger", "simple_mug"]
  jointNames : Array String :=
    #["weld_base", "translate_joint", "left_slider", "right_slider"]
  actuatedJointNames : Array String := #["translate_joint", "left_slider"]
  unactuatedJointNames : Array String := #["right_slider"]
  translateJointAxis : Vec3 := { x := 0.0, y := 0.0, z := 1.0 }
  fingerSliderAxis : Vec3 := { x := 1.0, y := 0.0, z := 0.0 }
  bodyBoxSize : Array Float := #[0.146, 0.0725, 0.049521]
  fingerBoxSize : Array Float := #[0.007, 0.081, 0.028]
  mugCylinderRadius : Float := params.mugRadius
  mugCylinderLength : Float := 0.1
  mugHandleBoxSize : Array Float := #[0.03, 0.02, 0.08]
  padMinorRadius : Float := params.padMinorRadius
  padMajorRadius : Float := params.padMajorRadius
  padTorusCenterY : Float := params.padTorusCenterY
  deriving Repr, Inhabited

namespace SimpleGripperModelAssetBoundary

private def finiteArray (xs : Array Float) : Bool :=
  xs.all Float.isFinite

def validate? (asset : SimpleGripperModelAssetBoundary) :
    Except String Unit := do
  if asset.gripperModelName != "simple_gripper" then
    .error s!"simple gripper model name mismatch: {asset.gripperModelName}"
  if asset.mugModelName != "simple_mug" then
    .error s!"simple mug model name mismatch: {asset.mugModelName}"
  if asset.gripperSdfPath != "../drake/examples/simple_gripper/simple_gripper.sdf" then
    .error s!"simple gripper SDF path mismatch: {asset.gripperSdfPath}"
  if asset.mugSdfPath != "../drake/examples/simple_gripper/simple_mug.sdf" then
    .error s!"simple mug SDF path mismatch: {asset.mugSdfPath}"
  if asset.gripperPackageUri != simpleGripperModelUri then
    .error s!"simple gripper package URI mismatch: {asset.gripperPackageUri}"
  if asset.mugPackageUri != simpleMugModelUri then
    .error s!"simple mug package URI mismatch: {asset.mugPackageUri}"
  if asset.linkNames != #["y_translate_link", "body", "left_finger", "right_finger", "simple_mug"] then
    .error s!"simple gripper link names mismatch: {asset.linkNames}"
  if asset.jointNames != #["weld_base", "translate_joint", "left_slider", "right_slider"] then
    .error s!"simple gripper joint names mismatch: {asset.jointNames}"
  if asset.actuatedJointNames != #["translate_joint", "left_slider"] then
    .error s!"simple gripper actuated joints mismatch: {asset.actuatedJointNames}"
  if asset.unactuatedJointNames != #["right_slider"] then
    .error s!"simple gripper unactuated joints mismatch: {asset.unactuatedJointNames}"
  if asset.translateJointAxis != ({ x := 0.0, y := 0.0, z := 1.0 } : Vec3) then
    .error s!"simple gripper translate axis mismatch: {reprStr asset.translateJointAxis}"
  if asset.fingerSliderAxis != ({ x := 1.0, y := 0.0, z := 0.0 } : Vec3) then
    .error s!"simple gripper finger slider axis mismatch: {reprStr asset.fingerSliderAxis}"
  if asset.bodyBoxSize.size != 3 || !finiteArray asset.bodyBoxSize then
    .error s!"simple gripper body box size must have three finite entries, got {asset.bodyBoxSize}"
  if asset.fingerBoxSize.size != 3 || !finiteArray asset.fingerBoxSize then
    .error s!"simple gripper finger box size must have three finite entries, got {asset.fingerBoxSize}"
  if !asset.mugCylinderRadius.isFinite || asset.mugCylinderRadius <= 0.0 then
    .error s!"simple mug cylinder radius must be positive and finite, got {asset.mugCylinderRadius}"
  if !asset.mugCylinderLength.isFinite || asset.mugCylinderLength <= 0.0 then
    .error s!"simple mug cylinder length must be positive and finite, got {asset.mugCylinderLength}"
  if asset.mugHandleBoxSize.size != 3 || !finiteArray asset.mugHandleBoxSize then
    .error s!"simple mug handle box size must have three finite entries, got {asset.mugHandleBoxSize}"
  if !asset.padMinorRadius.isFinite || asset.padMinorRadius <= 0.0 then
    .error s!"simple gripper pad minor radius must be positive and finite, got {asset.padMinorRadius}"
  if !asset.padMajorRadius.isFinite || asset.padMajorRadius <= 0.0 then
    .error s!"simple gripper pad major radius must be positive and finite, got {asset.padMajorRadius}"
  if !asset.padTorusCenterY.isFinite then
    .error s!"simple gripper pad torus center y must be finite, got {asset.padTorusCenterY}"

end SimpleGripperModelAssetBoundary

def simpleGripperModelAssetBoundary : SimpleGripperModelAssetBoundary := {}

def translateAxis (plant : SdfPlantSummary := plantSummary) : Vec3 :=
  match plant.jointRecords.find? (fun j => j.name == "translate_joint") with
  | some joint => joint.axis
  | none => {}

def gravityVectorForTranslateAxis (axis : Vec3) : Vec3 :=
  if axis == ({ x := 0.0, y := 0.0, z := 1.0 } : Vec3) then
    {}
  else
    { x := 0.0, y := 0.0, z := -9.81 }

structure PadSphere where
  id : Nat
  bodyName : String
  sampleIndex : Nat
  centerInFinger : Vec3
  radius : Float
  friction : CoulombFriction
  label : String := ""
  deriving Repr, Inhabited

def ringOrientRadians (p : SimpleGripperParams := params) : Float :=
  p.ringOrientDegrees * pi / 180.0

def ringPadSphere
    (p : SimpleGripperParams)
    (bodyName : String)
    (baseId : Nat)
    (padOffset : Float)
    (sampleIndex : Nat) : PadSphere :=
  let dtheta := 2.0 * pi / p.ringSamples.toFloat
  let theta := dtheta * sampleIndex.toFloat + ringOrientRadians p
  {
    id := baseId + sampleIndex
    bodyName := bodyName
    sampleIndex := sampleIndex
    centerInFinger := {
      x := padOffset
      y := Float.cos theta * p.padMajorRadius + p.padTorusCenterY
      z := Float.sin theta * p.padMajorRadius
    }
    radius := p.padMinorRadius
    friction := p.ringFriction
    label := s!"{bodyName}:ring-pad:{sampleIndex}"
  }

def ringPadsForBody
    (p : SimpleGripperParams)
    (bodyName : String)
    (baseId : Nat)
    (padOffset : Float) : Array PadSphere := Id.run do
  let mut out : Array PadSphere := #[]
  for i in [:p.ringSamples] do
    out := out.push (ringPadSphere p bodyName baseId padOffset i)
  return out

def generatedPadSpheres (p : SimpleGripperParams := params) : Array PadSphere :=
  if p.gripForce == 0.0 then
    let rightA := ringPadsForBody p "right_finger" 2000 (-p.padOffset)
    let fixedLeftOffset := -(p.gripWidth + p.fingerWidth) + p.padOffset
    rightA ++ ringPadsForBody p "right_finger" 3000 fixedLeftOffset
  else
    ringPadsForBody p "left_finger" 1000 p.padOffset ++
      ringPadsForBody p "right_finger" 2000 (-p.padOffset)

structure SimpleGripperCommand where
  translateForce : Float := 0.0
  leftFingerForce : Float := 0.0
  deriving Repr, Inhabited

def driveOmega (p : SimpleGripperParams := params) : Float :=
  2.0 * pi * p.frequency

def initialTranslateVelocity (p : SimpleGripperParams := params) : Float :=
  -p.amplitude * driveOmega p

def harmonicAccelerationAmplitude (p : SimpleGripperParams := params) : Float :=
  (driveOmega p) * (driveOmega p) * p.amplitude

def harmonicForceAmplitude (p : SimpleGripperParams := params) : Float :=
  p.gripperMovingMass * harmonicAccelerationAmplitude p

def gripActuationForce (p : SimpleGripperParams := params) : Float :=
  (gripperCoupler p).opposingGripActuationForce p.gripForce

def commandAt (p : SimpleGripperParams := params) (t : Float) :
    SimpleGripperCommand :=
  {
    translateForce := harmonicForceAmplitude p * Float.sin ((driveOmega p) * t)
    leftFingerForce := gripActuationForce p
  }

structure SimpleGripperState where
  translateQ : Float := 0.0
  translateV : Float := 0.0
  leftQ : Float := 0.0
  leftV : Float := 0.0
  rightQ : Float := 0.0
  rightV : Float := 0.0
  mugRpy : Rpy := {}
  deriving Repr, Inhabited

namespace SimpleGripperState

def asArray (x : SimpleGripperState) : Array Float :=
  #[x.translateQ, x.leftQ, x.rightQ, x.translateV, x.leftV, x.rightV]

def qdot (x : SimpleGripperState) : Array Float :=
  #[x.translateV, x.leftV, x.rightV]

def generalizedVelocity (x : SimpleGripperState) : Array Float :=
  #[x.translateV, x.leftV, x.rightV]

end SimpleGripperState

def initialState (p : SimpleGripperParams := params) : SimpleGripperState :=
  let fingerOffset := p.gripWidth / 2.0
  {
    translateQ := 0.0
    translateV := initialTranslateVelocity p
    leftQ := -fingerOffset
    leftV := 0.0
    rightQ := fingerOffset
    rightV := 0.0
    mugRpy := {
      roll := p.rxDegrees * pi / 180.0
      pitch := p.ryDegrees * pi / 180.0
      yaw := p.rzDegrees * pi / 180.0 + pi
    }
  }

def rpyQuaternion (rpy : Rpy) : Array Float :=
  let cr := Float.cos (0.5 * rpy.roll)
  let sr := Float.sin (0.5 * rpy.roll)
  let cp := Float.cos (0.5 * rpy.pitch)
  let sp := Float.sin (0.5 * rpy.pitch)
  let cy := Float.cos (0.5 * rpy.yaw)
  let sy := Float.sin (0.5 * rpy.yaw)
  #[
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy
  ]

def initialMugFloatingQ (p : SimpleGripperParams := params) : Array Float :=
  (rpyQuaternion (initialState p).mugRpy) ++ #[0.0, 0.0, 0.0]

def fullPlantInitialQ (p : SimpleGripperParams := params) : Array Float :=
  let x0 := initialState p
  #[x0.translateQ, x0.leftQ, x0.rightQ] ++ initialMugFloatingQ p

def fullPlantInitialV (p : SimpleGripperParams := params) : Array Float :=
  let x0 := initialState p
  #[x0.translateV, x0.leftV, x0.rightV] ++ Array.replicate 6 0.0

def fullPlantInitialActuation (p : SimpleGripperParams := params) : Array Float :=
  #[0.0, gripActuationForce p]

def closingState (p : SimpleGripperParams := params) : SimpleGripperState :=
  let rightQ :=
    p.mugRadius + p.padMinorRadius - p.fingerHalfGapAtZero + p.padOffset -
      p.graspPenetration
  {
    initialState p with
    rightQ := rightQ
    leftQ := (gripperCoupler p).constrainedFromReference rightQ
  }

def fingerBaseX (p : SimpleGripperParams) (bodyName : String) : Float :=
  if bodyName == "left_finger" then
    -p.fingerHalfGapAtZero
  else
    p.fingerHalfGapAtZero

def fingerPositionQ (x : SimpleGripperState) (bodyName : String) : Float :=
  if bodyName == "left_finger" then x.leftQ else x.rightQ

def fingerVelocityQ (x : SimpleGripperState) (bodyName : String) : Float :=
  if bodyName == "left_finger" then x.leftV else x.rightV

def padCenterRelativeToMug
    (p : SimpleGripperParams)
    (x : SimpleGripperState)
    (pad : PadSphere) : Vec3 :=
  {
    x := fingerBaseX p pad.bodyName + fingerPositionQ x pad.bodyName +
      pad.centerInFinger.x
    y := pad.centerInFinger.y - p.padTorusCenterY
    z := x.translateQ + pad.centerInFinger.z
  }

private def safeRadialInv (r : Float) : Float :=
  if r < 1.0e-12 then 0.0 else 1.0 / r

def padContactCandidate
    (p : SimpleGripperParams)
    (x : SimpleGripperState)
    (pad : PadSphere) : ContactCandidate :=
  let center := padCenterRelativeToMug p x pad
  let radial := Float.sqrt (center.x * center.x + center.y * center.y)
  let invRadial := safeRadialInv radial
  let nx := center.x * invRadial
  let ny := center.y * invRadial
  let signedDistance := radial - p.mugRadius - pad.radius
  let qdot := x.generalizedVelocity
  let normalJac :=
    if pad.bodyName == "left_finger" then
      #[0.0, nx, 0.0]
    else
      #[0.0, 0.0, nx]
  let tangentJac :=
    if pad.bodyName == "left_finger" then
      #[0.0, -ny, 0.0]
    else
      #[0.0, 0.0, -ny]
  let tangentJac2 := #[1.0, 0.0, 0.0]
  let candidate : ContactCandidate := {
    id := pad.id
    signedDistance := signedDistance
    normalVelocity := FloatArray.dot normalJac qdot
    tangentVelocity := FloatArray.dot tangentJac qdot
    tangentVelocity2 := FloatArray.dot tangentJac2 qdot
    normalJacobian := normalJac
    tangentJacobian := tangentJac
    tangentJacobian2 := tangentJac2
    mode := .separated
    label := pad.label
  }
  candidate.withClassifiedMode p.penetrationAllowance p.stictionTolerance

def contactCandidates
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p) : Array ContactCandidate :=
  (generatedPadSpheres p).map (padContactCandidate p x)

def contactCandidateSet
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p) : ContactCandidateSet :=
  ContactCandidateSet.ofArray (contactCandidates p x)
    "simple-gripper-ring-pad-candidates"

def contactCandidateProvider
    (p : SimpleGripperParams := params)
    (label : String := "simple gripper ring-pad contact provider") :
    ContactCandidateProvider SimpleGripperState :=
  {
    label := label
    candidatesAt? := fun x => do
      p.ringFriction.validate? "simple gripper ring friction"
      p.mugFriction.validate? "simple mug friction"
      pure { contactCandidateSet p x with label := label }
  }

def activeSupport
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p) : ContactSupport :=
  (contactCandidateSet p x).selectByDistance p.penetrationAllowance
    "simple-gripper-ring-pads"

private def signedUnit (x : Float) : Float :=
  if x > 0.0 then 1.0 else if x < 0.0 then -1.0 else 0.0

def contactForceForCandidate
    (p : SimpleGripperParams)
    (candidate : ContactCandidate) : ContactForceScalars :=
  let penetration := max 0.0 (-candidate.signedDistance)
  let damping := if candidate.normalVelocity < 0.0 then
    -p.normalDamping * candidate.normalVelocity
  else
    0.0
  let normalForce := max 0.0 (p.normalStiffness * penetration + damping)
  let pairFriction := CoulombFriction.combine p.ringFriction p.mugFriction
  let tangentForce := -pairFriction.dynamicFriction * normalForce *
    signedUnit candidate.tangentVelocity
  let tangentForce2 := -pairFriction.dynamicFriction * normalForce *
    signedUnit candidate.tangentVelocity2
  ContactForceScalars.fromCandidate3D candidate normalForce tangentForce tangentForce2

def contactForces?
    (p : SimpleGripperParams)
    (support : ContactSupport) : Except String (Array ContactForceScalars) := do
  let selected ← support.selectedCandidates?
  pure (selected.map (contactForceForCandidate p))

def aggregateGeneralizedForce
    (candidates : Array ContactCandidate)
    (forces : Array ContactForceScalars) : Array Float := Id.run do
  let mut out := #[0.0, 0.0, 0.0]
  for force in forces do
    match candidates.find? (fun candidate => candidate.id == force.candidateId) with
    | some candidate =>
        out := FloatArray.add out (force.generalizedForce candidate)
    | none => pure ()
  return out

structure ReducedFingerSolve where
  equation : ManipulatorEquation
  derivative : ManipulatorDerivative
  liftedVdot : Array Float
  deriving Repr, Inhabited

def reducedFingerEquation
    (p : SimpleGripperParams)
    (x : SimpleGripperState)
    (command : SimpleGripperCommand)
    (contactTau : Array Float) : ManipulatorEquation :=
  let coupler := gripperCoupler p
  let leftTau := command.leftFingerForce + contactTau.getD 1 0.0
  let rightTau := contactTau.getD 2 0.0
  let referenceTau := rightTau + coupler.gearRatio * leftTau
  {
    massMatrix := #[
      #[p.gripperMovingMass, 0.0],
      #[0.0, p.fingerMass + coupler.gearRatio * coupler.gearRatio * p.fingerMass]
    ]
    qdot := #[x.translateV, x.rightV]
    generalizedForces := #[
      command.translateForce + contactTau.getD 0 0.0,
      referenceTau
    ]
    label := "simple-gripper-coupled-reduced-equation"
  }

def solveReduced?
    (p : SimpleGripperParams)
    (x : SimpleGripperState)
    (command : SimpleGripperCommand)
    (contactTau : Array Float) : Except String ReducedFingerSolve := do
  let equation := reducedFingerEquation p x command contactTau
  let derivative ← equation.solve?
  let rightAcc := derivative.vdot.getD 1 0.0
  let leftAcc := (gripperCoupler p).gearRatio * rightAcc
  pure {
    equation := equation
    derivative := derivative
    liftedVdot := #[derivative.vdot.getD 0 0.0, leftAcc, rightAcc]
  }

def stateDerivativeFromVdot
    (x : SimpleGripperState)
    (vdot : Array Float) : SimpleGripperState :=
  {
    translateQ := x.translateV
    translateV := vdot.getD 0 0.0
    leftQ := x.leftV
    leftV := vdot.getD 1 0.0
    rightQ := x.rightV
    rightV := vdot.getD 2 0.0
    mugRpy := {}
  }

structure SimpleGripperStep where
  state : SimpleGripperState
  command : SimpleGripperCommand
  support : ContactSupport
  runtimeSupport : RuntimeSupport
  contactForces : Array ContactForceScalars
  generalizedContactForce : Array Float
  reducedSolve : ReducedFingerSolve
  derivative : SimpleGripperState
  deriving Repr, Inhabited

def physicsStep?
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p)
    (t : Float := 0.0) : Except String SimpleGripperStep := do
  (gripperCoupler p).validate?
  p.ringFriction.validate? "simple gripper ring friction"
  p.mugFriction.validate? "simple mug friction"
  let support := activeSupport p x
  support.validateJacobianWidth? 3
  let runtime ← support.toRuntimeSupport?
  let forces ← contactForces? p support
  let generalizedContactForce :=
    aggregateGeneralizedForce support.candidates forces
  let command := commandAt p t
  let reducedSolve ← solveReduced? p x command generalizedContactForce
  let derivative := stateDerivativeFromVdot x reducedSolve.liftedVdot
  pure {
    state := x
    command := command
    support := support
    runtimeSupport := runtime
    contactForces := forces
    generalizedContactForce := generalizedContactForce
    reducedSolve := reducedSolve
    derivative := derivative
  }

def coupledReducedVelocity
    (_p : SimpleGripperParams)
    (x : SimpleGripperState) : Array Float :=
  #[x.translateV, x.rightV]

def coupledReducedJacobian
    (p : SimpleGripperParams)
    (jac : Array Float) : Array Float :=
  let rho := (gripperCoupler p).gearRatio
  #[jac.getD 0 0.0, rho * jac.getD 1 0.0 + jac.getD 2 0.0]

def coupledReducedContactCandidate
    (p : SimpleGripperParams)
    (x : SimpleGripperState)
    (candidate : ContactCandidate) : ContactCandidate :=
  let normalJac := coupledReducedJacobian p candidate.normalJacobian
  let tangentJac := coupledReducedJacobian p candidate.tangentJacobian
  let tangentJac2 := coupledReducedJacobian p candidate.tangentJacobian2
  let qdot := coupledReducedVelocity p x
  { candidate with
    normalVelocity := FloatArray.dot normalJac qdot
    tangentVelocity := FloatArray.dot tangentJac qdot
    tangentVelocity2 := FloatArray.dot tangentJac2 qdot
    normalJacobian := normalJac
    tangentJacobian := tangentJac
    tangentJacobian2 := tangentJac2
  }

def coupledReducedContactCandidates
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p) : Array ContactCandidate :=
  (contactCandidates p x).map (coupledReducedContactCandidate p x)

def coupledReducedContactCandidateSet
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p) : ContactCandidateSet :=
  ContactCandidateSet.ofArray (coupledReducedContactCandidates p x)
    "simple-gripper-coupled-reduced-contact-candidates"

def coupledReducedContactCandidateProvider
    (p : SimpleGripperParams := params)
    (label : String := "simple gripper coupled-reduced contact provider") :
    ContactCandidateProvider SimpleGripperState :=
  {
    label := label
    candidatesAt? := fun x => do
      p.ringFriction.validate? "simple gripper ring friction"
      p.mugFriction.validate? "simple mug friction"
      (gripperCoupler p).validate?
      pure { coupledReducedContactCandidateSet p x with label := label }
  }

def coupledReducedContactSupport
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p) : ContactSupport :=
  (coupledReducedContactCandidateSet p x).selectWithPolicy
      (.threshold p.penetrationAllowance)
      "simple-gripper-coupled-reduced-contact-support"
    |>.classifyCandidates p.penetrationAllowance p.stictionTolerance

def coupledReducedActuationForces
    (p : SimpleGripperParams)
    (command : SimpleGripperCommand) : Array Float :=
  let rho := (gripperCoupler p).gearRatio
  #[command.translateForce, rho * command.leftFingerForce]

def coupledReducedMassMatrix (p : SimpleGripperParams) : Array (Array Float) :=
  let rho := (gripperCoupler p).gearRatio
  #[
    #[p.gripperMovingMass, 0.0],
    #[0.0, p.fingerMass + rho * rho * p.fingerMass]
  ]

def fullPhysicsPrimitives?
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p)
    (t : Float := 0.0)
    (label : String := "simple_gripper coupled contact primitive") :
    Except String FullPhysicsPrimitives := do
  (gripperCoupler p).validate?
  p.ringFriction.validate? "simple gripper ring friction"
  p.mugFriction.validate? "simple mug friction"
  let support := coupledReducedContactSupport p x
  support.validateJacobianWidth? 2
  let selected ← support.selectedCandidates?
  let contactForces := selected.map (contactForceForCandidate p)
  pure {
    massMatrix := coupledReducedMassMatrix p
    qdot := coupledReducedVelocity p x
    actuationForces := coupledReducedActuationForces p (commandAt p t)
    biasForces := #[]
    contactCandidates := coupledReducedContactCandidates p x
    supportPolicy := .threshold p.penetrationAllowance
    contactForceSource := .precomputed
    contactForces := contactForces
    distanceTol := p.penetrationAllowance
    tangentVelocityTol := p.stictionTolerance
    label := label
  }

structure SimpleGripperTimedState where
  t : Float := 0.0
  state : SimpleGripperState := initialState params
  deriving Repr, Inhabited

def timedState
    (p : SimpleGripperParams := params)
    (state : SimpleGripperState := initialState p)
    (t : Float := 0.0) : SimpleGripperTimedState :=
  { t := t, state := state }

def fullPhysicsPrimitiveProvider
    (p : SimpleGripperParams := params)
    (label : String := "simple gripper timed full physics provider") :
    FullPhysicsPrimitiveProvider SimpleGripperTimedState :=
  {
    label := label
    primitivesAt? := fun timed =>
      fullPhysicsPrimitives? p timed.state timed.t label
  }

def solveFullPhysics?
    (p : SimpleGripperParams := params)
    (x : SimpleGripperState := initialState p)
    (t : Float := 0.0)
    (intervalVertex : VertexId := 5113)
    (label : String := "simple_gripper coupled contact primitive") :
    Except String FullPhysicsResult := do
  let primitive ← fullPhysicsPrimitives? p x t label
  primitive.solve? intervalVertex

def liftedDerivativeFromFullPhysics
    (p : SimpleGripperParams)
    (x : SimpleGripperState)
    (result : FullPhysicsResult) : SimpleGripperState :=
  let rho := (gripperCoupler p).gearRatio
  let translateAcc := result.derivative.vdot.getD 0 0.0
  let rightAcc := result.derivative.vdot.getD 1 0.0
  stateDerivativeFromVdot x #[translateAcc, rho * rightAcc, rightAcc]

def contactBranchData? (support : ContactSupport) :
    Except String BranchEventData := do
  let selected ← support.selectedCandidates?
  let weight := if selected.isEmpty then 0.0 else 1.0 / selected.size.toFloat
  let mut children : Array BranchChild := #[]
  for candidate in selected do
    children := children.push {
      weight := weight
      resetJac := FloatMatrix.identity 6
      a := Array.replicate 6 0.0
      message := {
        value := -candidate.signedDistance
        stateAdjoint := #[0.0, candidate.normalJacobian.getD 1 0.0,
          candidate.normalJacobian.getD 2 0.0, 0.0, 0.0, 0.0]
      }
    }
  pure {
    children := children
    guardGrad := Array.replicate 6 0.0
    gamma := 1.0
  }

def acceptedSegment (p : SimpleGripperParams := params) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.mbpDiscreteUpdatePeriod
    tAfter := p.mbpDiscreteUpdatePeriod
    madeJumpAfter := false
    label := "simple-gripper-discrete-contact-step"
  }

def contactSupportVertex : VertexId := 5100

def traceForStep? (step : SimpleGripperStep) : Except String DynamicEventTrace := do
  let branchData ← contactBranchData? step.support
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval acceptedSegment)
      |>.push (.branch contactSupportVertex step.runtimeSupport branchData)
  trace.validate?
  pure trace

structure MultibodySimpleGripperConfig where
  simulationTime : Float := params.simulationTime
  timeStep : Float := params.mbpDiscreteUpdatePeriod
  penetrationAllowance : Float := params.penetrationAllowance
  stictionTolerance : Float := params.stictionTolerance
  contactApproximation : String := params.contactApproximation
  defaultComplianceType : String := "undefined"
  visualizationEnabled : Bool := true
  deriving Repr, Inhabited

namespace MultibodySimpleGripperConfig

def approximation? (cfg : MultibodySimpleGripperConfig) :
    Except String DiscreteContactApproximation :=
  DiscreteContactApproximation.fromString? cfg.contactApproximation

def validate? (cfg : MultibodySimpleGripperConfig) : Except String Unit := do
  if !cfg.simulationTime.isFinite || cfg.simulationTime <= 0.0 then
    .error s!"simple gripper simulation_time must be positive and finite, got {cfg.simulationTime}"
  if !cfg.timeStep.isFinite || cfg.timeStep < 0.0 then
    .error s!"simple gripper mbp_discrete_update_period must be nonnegative and finite, got {cfg.timeStep}"
  if !cfg.penetrationAllowance.isFinite || cfg.penetrationAllowance < 0.0 then
    .error s!"simple gripper penetration_allowance must be nonnegative and finite, got {cfg.penetrationAllowance}"
  if !cfg.stictionTolerance.isFinite || cfg.stictionTolerance <= 0.0 then
    .error s!"simple gripper stiction tolerance must be positive and finite, got {cfg.stictionTolerance}"
  let _ ← cfg.approximation?

def plantConfig? (cfg : MultibodySimpleGripperConfig) :
    Except String MultibodyPlantConfigPrimitive := do
  let approximation ← cfg.approximation?
  pure {
    timeStep := cfg.timeStep
    penetrationAllowance := cfg.penetrationAllowance
    stictionTolerance := cfg.stictionTolerance
    contactApproximation := approximation
  }

end MultibodySimpleGripperConfig

def multibodySimpleGripperConfig : MultibodySimpleGripperConfig := {}

def parsedSimpleGripperPlant : ParsedMultibodyPlantQuantities :=
  {
    modelUris := #[simpleGripperModelUri, simpleMugModelUri]
    builtInModelInstances := 2
    numModelInstances := 4
    numActuators := 2
    numJoints := 4
    numBodies := 6
    modelInstances := #[
      {
        name := "simple_gripper"
        modelUri := simpleGripperModelUri
        numPositions := 3
        numVelocities := 3
      },
      {
        name := "simple_mug"
        modelUri := simpleMugModelUri
        numPositions := 7
        numVelocities := 6
      }
    ]
    finalized := true
    label := "simple_gripper parser outputs"
  }

def multibodySimpleGripperModel : FullMultibodyPlantModel :=
  {
    modelName := "simple_gripper_with_simple_mug"
    modelUri := combinedModelUri
    numPositions := 10
    numVelocities := 9
    numActuatedDofs := 2
    floatingBases := #[
      {
        bodyName := "simple_mug"
        convention := .quaternion
        floatingPositionsStart := 3
        floatingVelocitiesStartInV := 3
      }
    ]
    finalized := true
    label := "simple_gripper full MultibodyPlant"
  }

def multibodySimpleGripperStep?
    (cfg : MultibodySimpleGripperConfig := multibodySimpleGripperConfig)
    (p : SimpleGripperParams := params) :
    Except String FullMultibodyPlantStep := do
  let plantConfig ← cfg.plantConfig?
  let step : FullMultibodyPlantStep := {
    model := multibodySimpleGripperModel
    config := plantConfig
    q0 := fullPlantInitialQ p
    v0 := fullPlantInitialV p
    actuation := fullPlantInitialActuation p
    t0 := 0.0
    t1 := cfg.simulationTime
    label := "simple_gripper full plant advance"
  }
  step.validate?
  pure step

private def multibodySegment (t1 : Float) : AcceptedStepSegment :=
  {
    id := 5113
    attemptIndex := 0
    tStart := 0.0
    tAttempt := t1
    tAfter := t1
    label := "simple_gripper Simulator.AdvanceTo"
  }

private def multibodyLocalMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

def multibodySimpleGripperMoves
    (cfg : MultibodySimpleGripperConfig)
    (p : SimpleGripperParams := params) : Array SkeletonMove :=
  #[
    ParsedMultibodyPlantQuantities.parserMove 5110
      "Parser.AddModelsFromUrl simple_gripper.sdf + simple_mug.sdf",
    multibodyLocalMove 5111
      s!"Register {2 * p.ringSamples} ring-pad collision spheres, compliance={cfg.defaultComplianceType}",
    multibodyLocalMove 5112
      s!"AddCouplerConstraint rho={p.couplerGearRatio} and Sine actuation"
  ]

structure MultibodySimpleGripperResult where
  references : Array DrakeReference
  asset : SimpleGripperModelAssetBoundary
  parsedPlant : ParsedMultibodyPlantQuantities
  config : MultibodySimpleGripperConfig
  pads : Array PadSphere
  step : FullMultibodyPlantStep
  fullPhysics : FullPhysicsResult
  fullPhysicsDerivative : SimpleGripperState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildMultibodySimpleGripper?
    (cfg : MultibodySimpleGripperConfig := multibodySimpleGripperConfig)
    (p : SimpleGripperParams := params)
    (asset : SimpleGripperModelAssetBoundary := simpleGripperModelAssetBoundary) :
    Except String MultibodySimpleGripperResult := do
  cfg.validate?
  asset.validate?
  parsedSimpleGripperPlant.validate?
  let step ← multibodySimpleGripperStep? cfg p
  let x0 := initialState p
  let fullPhysics ←
    solveFullPhysics? p x0 0.0 5113 "simple_gripper contact benchmark plant"
  let trace := DynamicEventTrace.empty.push (.interval (multibodySegment cfg.simulationTime))
  trace.validate?
  pure {
    references := drakeReferences
    asset := asset
    parsedPlant := parsedSimpleGripperPlant
    config := cfg
    pads := generatedPadSpheres p
    step := step
    fullPhysics := fullPhysics
    fullPhysicsDerivative := liftedDerivativeFromFullPhysics p x0 fullPhysics
    trace := trace
    moves := multibodySimpleGripperMoves cfg p ++
      #[fullPhysics.supportMove, fullPhysics.move] ++ trace.moves
  }

structure SimpleGripperResult where
  references : Array DrakeReference
  plant : SdfPlantSummary
  pads : Array PadSphere
  initialState : SimpleGripperState
  closingState : SimpleGripperState
  initialStep : SimpleGripperStep
  closingStep : SimpleGripperStep
  fullPlant : MultibodySimpleGripperResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd? (p : SimpleGripperParams := params) :
    Except String SimpleGripperResult := do
  let plant := plantSummary p
  plant.coupler.validate?
  let initialStep ← physicsStep? p (initialState p) 0.0
  let closingStep ← physicsStep? p (closingState p) 0.0
  let trace ← traceForStep? closingStep
  let fullPlant ← buildMultibodySimpleGripper? multibodySimpleGripperConfig p
  pure {
    references := drakeReferences
    plant := plant
    pads := generatedPadSpheres p
    initialState := initialState p
    closingState := closingState p
    initialStep := initialStep
    closingStep := closingStep
    fullPlant := fullPlant
    trace := trace
    moves := trace.moves ++ fullPlant.moves
  }

end Tyr.EventSkeleton.Examples.SimpleGripper
