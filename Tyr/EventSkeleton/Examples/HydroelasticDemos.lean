import Tyr.EventSkeleton.Manipulator

/-!
# Drake Hydroelastic Demo Event-Skeleton Examples

This covers the remaining hydroelastic demo family under
`../drake/examples/hydroelastic`: the Python ball-paddle demo, the Python
non-convex pepper/bowl/table demo, and the C++ spatula slip-control demo.

The local pieces below are expressed directly with existing primitives: model
defaults, initial state assembly, dynamic patch providers, patch-to-contact
support adapters, pressure-derived generalized force contributions,
mass-matrix dynamics, and the spatula square-wave gripper controller.  More
sophisticated geometry providers can replace the analytic patch providers
without changing the full-physics boundary.
-/

namespace Tyr.EventSkeleton.Examples.HydroelasticDemos

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/hydroelastic/python_ball_paddle/contact_sim_demo.py"
      concept := "builds a pydrake hydroelastic ball-paddle diagram, logs plant state, and sets the ball initial q/v"
    },
    {
      path := "../drake/examples/hydroelastic/python_ball_paddle/ball.sdf"
      concept := "declares the compliant hydroelastic ball radius, mass, modulus, dissipation, and friction"
    },
    {
      path := "../drake/examples/hydroelastic/python_ball_paddle/paddle.sdf"
      concept := "declares the compliant hydroelastic paddle box and welded top-surface pose"
    },
    {
      path := "../drake/examples/hydroelastic/python_nonconvex_mesh/drop_pepper.py"
      concept := "assembles pepper, bowl, and table free-body state and runs hydroelastic-with-fallback contact"
    },
    {
      path := "../drake/examples/hydroelastic/python_nonconvex_mesh/table.sdf"
      concept := "declares the welded compliant hydroelastic table top"
    },
    {
      path := "../drake/examples/hydroelastic/spatula_slip_control/spatula_slip_control.cc"
      concept := "builds the hydroelastic gripper/spatula plant and square-wave open-loop actuation"
    },
    {
      path := "../drake/examples/hydroelastic/spatula_slip_control/spatula.sdf"
      concept := "declares the compliant hydroelastic spatula handle geometry and material"
    }
  ]

inductive HydroelasticDemoPackage where
  | pythonBallPaddle
  | pythonNonconvexMesh
  | spatulaSlipControl
  deriving Repr, BEq, Inhabited

namespace HydroelasticDemoPackage

def root : HydroelasticDemoPackage → String
  | .pythonBallPaddle => "../drake/examples/hydroelastic/python_ball_paddle"
  | .pythonNonconvexMesh => "../drake/examples/hydroelastic/python_nonconvex_mesh"
  | .spatulaSlipControl => "../drake/examples/hydroelastic/spatula_slip_control"

def label : HydroelasticDemoPackage → String
  | .pythonBallPaddle => "python_ball_paddle"
  | .pythonNonconvexMesh => "python_nonconvex_mesh"
  | .spatulaSlipControl => "spatula_slip_control"

end HydroelasticDemoPackage

structure HydroelasticDocumentationImage where
  package : HydroelasticDemoPackage
  relativePath : String
  includedInModelFilegroup : Bool := false
  concept : String
  deriving Repr, Inhabited

namespace HydroelasticDocumentationImage

def fullPath (asset : HydroelasticDocumentationImage) : String :=
  asset.package.root ++ "/" ++ asset.relativePath

def validate? (asset : HydroelasticDocumentationImage) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "Hydroelastic documentation image path cannot be empty"
  if !(asset.relativePath.endsWith ".jpg") then
    .error s!"Hydroelastic documentation image should be a jpg, got {asset.relativePath}"
  if !(asset.relativePath.contains "images/") then
    .error s!"Hydroelastic documentation image should live under images/, got {asset.relativePath}"
  if asset.concept.isEmpty then
    .error s!"Hydroelastic documentation image {asset.relativePath}: concept cannot be empty"
  if asset.includedInModelFilegroup then
    .error s!"Hydroelastic documentation image {asset.relativePath} should stay out of model filegroups"

end HydroelasticDocumentationImage

def documentationImageAssets : Array HydroelasticDocumentationImage :=
  #[
    {
      package := .pythonBallPaddle
      relativePath := "images/ball_paddle_corner.jpg"
      concept := "README screenshot of the ball contacting the paddle near a corner"
    },
    {
      package := .pythonBallPaddle
      relativePath := "images/ball_paddle_default.jpg"
      concept := "README screenshot of the default ball-paddle hydroelastic setup"
    },
    {
      package := .pythonBallPaddle
      relativePath := "images/ball_paddle_near_edge.jpg"
      concept := "README screenshot of the ball contacting near the paddle edge"
    },
    {
      package := .pythonNonconvexMesh
      relativePath := "images/contact_surface.jpg"
      concept := "README visualization of non-convex mesh hydroelastic contact surface"
    },
    {
      package := .pythonNonconvexMesh
      relativePath := "images/init.jpg"
      concept := "README initial condition screenshot for the pepper/table demo"
    },
    {
      package := .pythonNonconvexMesh
      relativePath := "images/run.jpg"
      concept := "README run screenshot for the pepper/table demo"
    },
    {
      package := .spatulaSlipControl
      relativePath := "images/spatula_1.jpg"
      concept := "README screenshot of the spatula slip-control setup before slip"
    },
    {
      package := .spatulaSlipControl
      relativePath := "images/spatula_2.jpg"
      concept := "README screenshot of the spatula during controlled slip"
    },
    {
      package := .spatulaSlipControl
      relativePath := "images/spatula_3.jpg"
      concept := "README screenshot of the spatula after the slip-control episode"
    }
  ]

def documentationImagePaths : Array String :=
  documentationImageAssets.map (fun asset => asset.fullPath)

private def hasDuplicateDocumentationImagePath
    (assets : Array HydroelasticDocumentationImage) : Bool := Id.run do
  for i in [:assets.size] do
    for j in [:(assets.size - i - 1)] do
      let k := i + j + 1
      if assets[i]!.fullPath == assets[k]!.fullPath then
        return true
  return false

def documentationImagesFor
    (package : HydroelasticDemoPackage) : Array HydroelasticDocumentationImage :=
  documentationImageAssets.filter (fun asset => asset.package == package)

def validateDocumentationImageAssets? : Except String Unit := do
  if documentationImageAssets.size != 9 then
    .error s!"Hydroelastic documentation image catalog should contain 9 images, got {documentationImageAssets.size}"
  if hasDuplicateDocumentationImagePath documentationImageAssets then
    .error "Hydroelastic documentation image catalog contains duplicate paths"
  for asset in documentationImageAssets do
    asset.validate?
  for package in [HydroelasticDemoPackage.pythonBallPaddle,
      HydroelasticDemoPackage.pythonNonconvexMesh,
      HydroelasticDemoPackage.spatulaSlipControl] do
    let images := documentationImagesFor package
    if images.size != 3 then
      .error s!"Hydroelastic package {package.label} should have 3 README images, got {images.size}"

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

def asArray (v : Vec3) : Array Float :=
  #[v.x, v.y, v.z]

def isFinite (v : Vec3) : Bool :=
  Float.isFinite v.x && Float.isFinite v.y && Float.isFinite v.z

end Vec3

private def finiteArray (xs : Array Float) : Bool :=
  xs.all (fun x => x.isFinite)

def averageModulus (a b : Float) : Float :=
  0.5 * (a + b)

def sphereHalfspaceHydroPatch
    (id : Nat)
    (bodyA bodyB : String)
    (radius modulusA modulusB : Float)
    (center : Vec3)
    (surfaceZ : Float)
    (normalVelocity : Float)
    (normalJacobian : Array Float)
    (label : String) : HydroelasticContactPatch :=
  let penetration := max 0.0 (surfaceZ + radius - center.z)
  let contactRadius2 := max 0.0 (radius * radius - (radius - penetration) * (radius - penetration))
  let area := pi * contactRadius2
  let averagePressure :=
    if penetration > 0.0 then
      averageModulus modulusA modulusB * penetration / max radius 1.0e-12
    else
      0.0
  {
    id := id
    bodyA := bodyA
    bodyB := bodyB
    complianceA := .compliant
    complianceB := .compliant
    representation := .polygon
    area := area
    centroid := #[center.x, center.y, surfaceZ]
    normal := #[0.0, 0.0, 1.0]
    averagePressure := averagePressure
    normalVelocity := normalVelocity
    normalJacobian := normalJacobian
    tangentJacobian := Array.replicate normalJacobian.size 0.0
    tangentJacobian2 := Array.replicate normalJacobian.size 0.0
    label := label
  }

def fullPhysicsPrimitivesFromHydroelasticSupport?
    (massMatrix : Array (Array Float))
    (qdot actuationForces biasForces : Array Float)
    (support : HydroelasticPatchSupport)
    (label : String)
    (generalizedForceContributions : Array GeneralizedForceContribution := #[]) :
    Except String FullPhysicsPrimitives := do
  support.validateGeometry?
  let contactSupport := support.equivalentContactSupport
  let contactForces ← support.selectedContactForces?
  pure {
    massMatrix := massMatrix
    qdot := qdot
    actuationForces := actuationForces
    biasForces := biasForces
    generalizedForceContributions := generalizedForceContributions
    contactCandidates := contactSupport.candidates
    sourceContactCandidateCount? := contactSupport.sourceCandidateCount?
    supportPolicy := contactSupport.policy
    contactForceSource := .precomputed
    contactForces := contactForces
    label := label
  }

/-! ## Python ball-paddle demo -/

structure BallPaddleParams where
  simulationTime : Float := 0.5
  contactModel : ContactModelChoice := .hydroelasticWithFallback
  surfaceRepresentation : HydroelasticSurfaceRepresentation := .polygon
  timeStep : Float := 0.001
  targetRealtimeRate : Float := 1.0
  gravity : Float := 9.81
  ballMass : Float := 0.1
  ballRadius : Float := 0.02
  ballHydroelasticModulus : Float := 5.0e6
  ballResolutionHint : Float := 0.005
  ballDissipation : Float := 0.1
  paddleMass : Float := 5.0e3
  paddleSize : Vec3 := { x := 0.2, y := 0.2, z := 0.02 }
  paddleHydroelasticModulus : Float := 2.5e6
  paddleMeshResolutionHint : Float := 100.0
  paddleWeldTranslation : Vec3 := { z := -0.01 }
  ballInitialPosition : Vec3 := { z := 0.1 }
  ballInitialVelocity : Vec3 := {}
  deriving Repr, Inhabited

def ballPaddleParams : BallPaddleParams := {}

namespace BallPaddleParams

def paddleTopZ (p : BallPaddleParams) : Float :=
  p.paddleWeldTranslation.z + 0.5 * p.paddleSize.z

def stateSize (_p : BallPaddleParams) : Nat :=
  13

def initialQ (p : BallPaddleParams) : Array Float :=
  #[1.0, 0.0, 0.0, 0.0] ++ p.ballInitialPosition.asArray

def initialV (p : BallPaddleParams) : Array Float :=
  #[0.0, 0.0, 0.0] ++ p.ballInitialVelocity.asArray

def initialState (p : BallPaddleParams) : Array Float :=
  p.initialQ ++ p.initialV

def ballRotationalInertia (p : BallPaddleParams) : Float :=
  0.4 * p.ballMass * p.ballRadius * p.ballRadius

def massMatrix (p : BallPaddleParams) : Array (Array Float) :=
  let ib := p.ballRotationalInertia
  FloatMatrix.diagonal #[ib, ib, ib, p.ballMass, p.ballMass, p.ballMass]

def gravityBias (p : BallPaddleParams) : Array Float :=
  #[0.0, 0.0, 0.0, 0.0, 0.0, p.ballMass * p.gravity]

def validate? (p : BallPaddleParams) : Except String Unit := do
  if !(Float.isFinite p.simulationTime) || p.simulationTime <= 0.0 then
    .error s!"ball-paddle simulation time must be positive and finite, got {p.simulationTime}"
  if !(Float.isFinite p.timeStep) || p.timeStep < 0.0 then
    .error s!"ball-paddle time step must be nonnegative and finite, got {p.timeStep}"
  if !(Float.isFinite p.gravity) || p.gravity < 0.0 then
    .error s!"ball-paddle gravity must be nonnegative and finite, got {p.gravity}"
  if !(Float.isFinite p.ballMass) || p.ballMass <= 0.0 then
    .error s!"ball mass must be positive and finite, got {p.ballMass}"
  if !(Float.isFinite p.ballRadius) || p.ballRadius <= 0.0 then
    .error s!"ball radius must be positive and finite, got {p.ballRadius}"
  if !(Float.isFinite p.ballHydroelasticModulus) || p.ballHydroelasticModulus <= 0.0 then
    .error s!"ball hydroelastic modulus must be positive and finite, got {p.ballHydroelasticModulus}"
  if !(Float.isFinite p.paddleHydroelasticModulus) || p.paddleHydroelasticModulus <= 0.0 then
    .error s!"paddle hydroelastic modulus must be positive and finite, got {p.paddleHydroelasticModulus}"

end BallPaddleParams

def ballPaddlePatch
    (p : BallPaddleParams := ballPaddleParams)
    (center : Vec3 := { p.ballInitialPosition with z := 0.015 })
    (normalVelocity : Float := p.ballInitialVelocity.z) :
    HydroelasticContactPatch :=
  sphereHalfspaceHydroPatch 2000 "ball" "paddle" p.ballRadius
    p.ballHydroelasticModulus p.paddleHydroelasticModulus center p.paddleTopZ
    normalVelocity #[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    "python_ball_paddle_patch"

def ballPaddleSupport
    (p : BallPaddleParams := ballPaddleParams)
    (center : Vec3 := { p.ballInitialPosition with z := 0.015 })
    (normalVelocity : Float := p.ballInitialVelocity.z) :
    HydroelasticPatchSupport :=
  HydroelasticPatchSupport.selectByArea 1.0e-12
    #[ballPaddlePatch p center normalVelocity]
    "python ball-paddle hydroelastic support"

structure BallPaddlePhysicsState where
  params : BallPaddleParams := ballPaddleParams
  center : Vec3 := { ballPaddleParams.ballInitialPosition with z := 0.015 }
  velocity : Array Float := ballPaddleParams.initialV
  deriving Repr, Inhabited

namespace BallPaddlePhysicsState

def validate? (snapshot : BallPaddlePhysicsState) : Except String Unit := do
  snapshot.params.validate?
  if !snapshot.center.isFinite then
    .error s!"ball-paddle center must be finite, got {reprStr snapshot.center}"
  if snapshot.velocity.size != 6 then
    .error s!"ball-paddle velocity size {snapshot.velocity.size} != 6"
  if !finiteArray snapshot.velocity then
    .error s!"ball-paddle velocity must be finite, got {snapshot.velocity}"

def normalVelocity (snapshot : BallPaddlePhysicsState) : Float :=
  snapshot.velocity.getD 5 0.0

def support (snapshot : BallPaddlePhysicsState) : HydroelasticPatchSupport :=
  ballPaddleSupport snapshot.params snapshot.center snapshot.normalVelocity

def fullPhysicsPrimitives?
    (snapshot : BallPaddlePhysicsState)
    (label : String := "python ball-paddle hydroelastic full physics") :
    Except String FullPhysicsPrimitives := do
  snapshot.validate?
  fullPhysicsPrimitivesFromHydroelasticSupport?
    snapshot.params.massMatrix snapshot.velocity
    (Array.replicate snapshot.velocity.size 0.0)
    snapshot.params.gravityBias snapshot.support label

end BallPaddlePhysicsState

def ballPaddlePhysicsState
    (p : BallPaddleParams := ballPaddleParams)
    (center : Vec3 := { p.ballInitialPosition with z := 0.015 })
    (velocity : Array Float := p.initialV) :
    BallPaddlePhysicsState :=
  { params := p, center := center, velocity := velocity }

def ballPaddleFullPhysicsPrimitiveProvider
    (label : String := "python ball-paddle hydroelastic full physics") :
    FullPhysicsPrimitiveProvider BallPaddlePhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => snapshot.fullPhysicsPrimitives? label
  }

def ballPaddleFullPhysics?
    (p : BallPaddleParams := ballPaddleParams)
    (center : Vec3 := { p.ballInitialPosition with z := 0.015 }) :
    Except String FullPhysicsResult := do
  (ballPaddleFullPhysicsPrimitiveProvider
    "python ball-paddle hydroelastic full physics").solveAt?
      (ballPaddlePhysicsState p center p.initialV) 5504

/-! ## Python non-convex pepper/bowl/table demo -/

structure NonconvexMeshParams where
  simulationTime : Float := 2.0
  contactModel : ContactModelChoice := .hydroelasticWithFallback
  timeStep : Float := 0.01
  targetRealtimeRate : Float := 1.0
  gravity : Float := 9.81
  pepperModelUrl : String := "package://drake_models/veggies/yellow_bell_pepper_no_stem_low.sdf"
  bowlModelUrl : String := "package://drake_models/dishes/evo_bowl.sdf"
  tableModelUrl : String := "package://drake/examples/hydroelastic/python_nonconvex_mesh/table.sdf"
  pepperPosition : Vec3 := { y := -0.15, z := 0.10 }
  pepperWz : Float := 150.0
  pepperMass : Float := 0.15
  pepperPrincipalInertia : Float := 1.0e-4
  bowlPosition : Vec3 := { y := -0.07, z := 0.061 }
  bowlMass : Float := 0.35
  bowlPrincipalInertia : Float := 1.0e-3
  tableWeldTranslation : Vec3 := { z := -0.01 }
  tableSize : Vec3 := { x := 0.2, y := 0.2, z := 0.02 }
  tableHydroelasticModulus : Float := 5.0e6
  pepperHydroelasticModulus : Float := 2.0e5
  deriving Repr, Inhabited

def nonconvexMeshParams : NonconvexMeshParams := {}

namespace NonconvexMeshParams

def tableTopZ (p : NonconvexMeshParams) : Float :=
  p.tableWeldTranslation.z + 0.5 * p.tableSize.z

def initialQ (p : NonconvexMeshParams) : Array Float :=
  #[1.0, 0.0, 0.0, 0.0] ++ p.pepperPosition.asArray ++
    #[1.0, 0.0, 0.0, 0.0] ++ p.bowlPosition.asArray

def initialV (p : NonconvexMeshParams) : Array Float :=
  #[0.0, 0.0, p.pepperWz] ++ #[0.0, 0.0, 0.0] ++ #[0.0, 0.0, 0.0] ++ #[0.0, 0.0, 0.0]

def initialState (p : NonconvexMeshParams) : Array Float :=
  p.initialQ ++ p.initialV

def massMatrix (p : NonconvexMeshParams) : Array (Array Float) :=
  FloatMatrix.diagonal #[
    p.pepperPrincipalInertia, p.pepperPrincipalInertia, p.pepperPrincipalInertia,
    p.pepperMass, p.pepperMass, p.pepperMass,
    p.bowlPrincipalInertia, p.bowlPrincipalInertia, p.bowlPrincipalInertia,
    p.bowlMass, p.bowlMass, p.bowlMass
  ]

def gravityBias (p : NonconvexMeshParams) : Array Float :=
  #[
    0.0, 0.0, 0.0, 0.0, 0.0, p.pepperMass * p.gravity,
    0.0, 0.0, 0.0, 0.0, 0.0, p.bowlMass * p.gravity
  ]

def validate? (p : NonconvexMeshParams) : Except String Unit := do
  if !(Float.isFinite p.simulationTime) || p.simulationTime <= 0.0 then
    .error s!"nonconvex simulation time must be positive and finite, got {p.simulationTime}"
  if !(Float.isFinite p.timeStep) || p.timeStep <= 0.0 then
    .error s!"nonconvex time step must be positive and finite, got {p.timeStep}"
  if !(Float.isFinite p.gravity) || p.gravity < 0.0 then
    .error s!"nonconvex gravity must be nonnegative and finite, got {p.gravity}"
  if !(Float.isFinite p.pepperWz) then
    .error s!"pepper angular velocity must be finite, got {p.pepperWz}"
  if !(Float.isFinite p.pepperMass) || p.pepperMass <= 0.0 then
    .error s!"pepper mass must be positive and finite, got {p.pepperMass}"
  if !(Float.isFinite p.pepperPrincipalInertia) || p.pepperPrincipalInertia <= 0.0 then
    .error s!"pepper inertia must be positive and finite, got {p.pepperPrincipalInertia}"
  if !(Float.isFinite p.bowlMass) || p.bowlMass <= 0.0 then
    .error s!"bowl mass must be positive and finite, got {p.bowlMass}"
  if !(Float.isFinite p.bowlPrincipalInertia) || p.bowlPrincipalInertia <= 0.0 then
    .error s!"bowl inertia must be positive and finite, got {p.bowlPrincipalInertia}"
  if !(Float.isFinite p.tableHydroelasticModulus) || p.tableHydroelasticModulus <= 0.0 then
    .error s!"table hydroelastic modulus must be positive and finite, got {p.tableHydroelasticModulus}"
  if !(Float.isFinite p.pepperHydroelasticModulus) || p.pepperHydroelasticModulus <= 0.0 then
    .error s!"pepper hydroelastic modulus must be positive and finite, got {p.pepperHydroelasticModulus}"

end NonconvexMeshParams

def pepperTablePatch
    (p : NonconvexMeshParams := nonconvexMeshParams)
    (pepperBottom : Vec3 := { p.pepperPosition with z := -0.002 })
    (normalVelocity : Float := 0.0) :
    HydroelasticContactPatch :=
  sphereHalfspaceHydroPatch 2100 "yellow_bell_pepper" "table" 0.04
    p.pepperHydroelasticModulus p.tableHydroelasticModulus pepperBottom p.tableTopZ
    normalVelocity #[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    "pepper_table_nonconvex_hydroelastic_patch"

def pepperTableSupport
    (p : NonconvexMeshParams := nonconvexMeshParams)
    (pepperBottom : Vec3 := { p.pepperPosition with z := -0.002 })
    (normalVelocity : Float := 0.0) :
    HydroelasticPatchSupport :=
  HydroelasticPatchSupport.selectByArea 1.0e-12
    #[pepperTablePatch p pepperBottom normalVelocity]
    "python nonconvex pepper-table hydroelastic support"

structure PepperTablePhysicsState where
  params : NonconvexMeshParams := nonconvexMeshParams
  pepperBottom : Vec3 := { nonconvexMeshParams.pepperPosition with z := -0.002 }
  velocity : Array Float := nonconvexMeshParams.initialV
  deriving Repr, Inhabited

namespace PepperTablePhysicsState

def validate? (snapshot : PepperTablePhysicsState) : Except String Unit := do
  snapshot.params.validate?
  if !snapshot.pepperBottom.isFinite then
    .error s!"pepper-table bottom point must be finite, got {reprStr snapshot.pepperBottom}"
  if snapshot.velocity.size != 12 then
    .error s!"pepper-table velocity size {snapshot.velocity.size} != 12"
  if !finiteArray snapshot.velocity then
    .error s!"pepper-table velocity must be finite, got {snapshot.velocity}"

def normalVelocity (snapshot : PepperTablePhysicsState) : Float :=
  snapshot.velocity.getD 5 0.0

def support (snapshot : PepperTablePhysicsState) : HydroelasticPatchSupport :=
  pepperTableSupport snapshot.params snapshot.pepperBottom snapshot.normalVelocity

def fullPhysicsPrimitives?
    (snapshot : PepperTablePhysicsState)
    (label : String := "python nonconvex pepper-table full physics") :
    Except String FullPhysicsPrimitives := do
  snapshot.validate?
  fullPhysicsPrimitivesFromHydroelasticSupport?
    snapshot.params.massMatrix snapshot.velocity
    (Array.replicate snapshot.velocity.size 0.0)
    snapshot.params.gravityBias snapshot.support label

end PepperTablePhysicsState

def pepperTablePhysicsState
    (p : NonconvexMeshParams := nonconvexMeshParams)
    (pepperBottom : Vec3 := { p.pepperPosition with z := -0.002 })
    (velocity : Array Float := p.initialV) :
    PepperTablePhysicsState :=
  { params := p, pepperBottom := pepperBottom, velocity := velocity }

def pepperTableFullPhysicsPrimitiveProvider
    (label : String := "python nonconvex pepper-table full physics") :
    FullPhysicsPrimitiveProvider PepperTablePhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => snapshot.fullPhysicsPrimitives? label
  }

def pepperTableFullPhysics?
    (p : NonconvexMeshParams := nonconvexMeshParams)
    (pepperBottom : Vec3 := { p.pepperPosition with z := -0.002 }) :
    Except String FullPhysicsResult := do
  (pepperTableFullPhysicsPrimitiveProvider
    "python nonconvex pepper-table full physics").solveAt?
      (pepperTablePhysicsState p pepperBottom p.initialV) 5505

/-! ## Spatula slip-control demo -/

structure SpatulaSlipParams where
  gripperForce : Float := 1.5
  amplitude : Float := 5.0
  dutyCycle : Float := 0.5
  period : Float := 3.0
  stictionTolerance : Float := 1.0e-4
  discreteUpdatePeriod : Float := 4.0e-2
  contactModel : ContactModelChoice := .hydroelastic
  surfaceRepresentation : HydroelasticSurfaceRepresentation := .polygon
  contactApproximation : String := "lagged"
  realtimeRate : Float := 1.0
  simulationSec : Float := 30.0
  accuracy : Float := 1.0e-3
  maxTimeStep : Float := 1.0e-2
  integrationScheme : String := "implicit_euler"
  fingerMass : Float := 0.05
  gripperPoseRpy : Vec3 := { y := -1.57 }
  gripperPoseTranslation : Vec3 := { z := 0.25 }
  spatulaPoseRpy : Vec3 := { x := -0.4, z := 1.57 }
  spatulaPoseTranslation : Vec3 := { x := 0.35, z := 0.25 }
  leftFingerInitial : Float := -0.01
  rightFingerInitial : Float := 0.01
  spatulaYawInertia : Float := 0.18858265237650276
  spatulaSlipRate : Float := 0.0
  torsionalFrictionRadius : Float := 0.015
  torsionalFrictionCoefficient : Float := 1.0
  deriving Repr, Inhabited

def spatulaSlipParams : SpatulaSlipParams := {}

namespace SpatulaSlipParams

def validate? (p : SpatulaSlipParams) : Except String Unit := do
  if !(Float.isFinite p.period) || p.period <= 0.0 then
    .error s!"square-wave period must be positive and finite, got {p.period}"
  if !(Float.isFinite p.dutyCycle) || p.dutyCycle < 0.0 || p.dutyCycle > 1.0 then
    .error s!"duty cycle must be in [0, 1], got {p.dutyCycle}"
  if !(Float.isFinite p.discreteUpdatePeriod) || p.discreteUpdatePeriod < 0.0 then
    .error s!"plant discrete update period must be nonnegative and finite, got {p.discreteUpdatePeriod}"
  if !(Float.isFinite p.simulationSec) || p.simulationSec <= 0.0 then
    .error s!"spatula simulation seconds must be positive and finite, got {p.simulationSec}"
  if !(Float.isFinite p.fingerMass) || p.fingerMass <= 0.0 then
    .error s!"finger mass must be positive and finite, got {p.fingerMass}"
  if !(Float.isFinite p.spatulaYawInertia) || p.spatulaYawInertia <= 0.0 then
    .error s!"spatula yaw inertia must be positive and finite, got {p.spatulaYawInertia}"
  if !(Float.isFinite p.spatulaSlipRate) then
    .error s!"spatula slip rate must be finite, got {p.spatulaSlipRate}"
  if !(Float.isFinite p.torsionalFrictionRadius) || p.torsionalFrictionRadius < 0.0 then
    .error s!"spatula torsional friction radius must be nonnegative and finite, got {p.torsionalFrictionRadius}"
  if !(Float.isFinite p.torsionalFrictionCoefficient) || p.torsionalFrictionCoefficient < 0.0 then
    .error s!"spatula torsional friction coefficient must be nonnegative and finite, got {p.torsionalFrictionCoefficient}"

def squareWave (p : SpatulaSlipParams) (time : Float) : Array Float :=
  let phaseTime := time - Float.floor (time / p.period) * p.period
  let high := phaseTime < p.dutyCycle * p.period
  if high then
    #[p.amplitude, -p.amplitude]
  else
    #[0.0, 0.0]

def constantForce (p : SpatulaSlipParams) : Array Float :=
  #[p.gripperForce, -p.gripperForce]

def actuation (p : SpatulaSlipParams) (time : Float) : Array Float :=
  FloatArray.add (p.squareWave time) p.constantForce

def generalizedActuation (p : SpatulaSlipParams) (time : Float) : Array Float :=
  p.actuation time ++ #[0.0]

def massMatrix (p : SpatulaSlipParams) : Array (Array Float) :=
  FloatMatrix.diagonal #[p.fingerMass, p.fingerMass, p.spatulaYawInertia]

def qdot (p : SpatulaSlipParams) : Array Float :=
  #[0.0, 0.0, p.spatulaSlipRate]

def biasForces (_p : SpatulaSlipParams) : Array Float :=
  #[0.0, 0.0, 0.0]

def torsionalSlipDirection (p : SpatulaSlipParams) : Float :=
  if p.spatulaSlipRate > p.stictionTolerance then
    -1.0
  else if p.spatulaSlipRate < -p.stictionTolerance then
    1.0
  else
    0.0

end SpatulaSlipParams

def spatulaLeftPatch (p : SpatulaSlipParams := spatulaSlipParams) :
    HydroelasticContactPatch :=
  {
    id := 2200
    bodyA := "left_finger_bubble"
    bodyB := "spatula"
    complianceA := .compliant
    complianceB := .compliant
    representation := p.surfaceRepresentation
    area := 3.0e-4
    centroid := #[0.35, -0.012, 0.25]
    normal := #[0.0, 1.0, 0.0]
    averagePressure := 1.0e4
    normalJacobian := #[1.0, 0.0, 0.0]
    tangentJacobian := #[0.0, 0.0, 0.0]
    tangentJacobian2 := #[0.0, 0.0, 0.0]
    label := "left_finger_bubble_spatula_patch"
  }

def spatulaRightPatch (p : SpatulaSlipParams := spatulaSlipParams) :
    HydroelasticContactPatch :=
  {
    id := 2201
    bodyA := "right_finger_bubble"
    bodyB := "spatula"
    complianceA := .compliant
    complianceB := .compliant
    representation := p.surfaceRepresentation
    area := 3.0e-4
    centroid := #[0.35, 0.012, 0.25]
    normal := #[0.0, -1.0, 0.0]
    averagePressure := 1.0e4
    normalJacobian := #[0.0, 1.0, 0.0]
    tangentJacobian := #[0.0, 0.0, 0.0]
    tangentJacobian2 := #[0.0, 0.0, 0.0]
    label := "right_finger_bubble_spatula_patch"
  }

def spatulaPatchSupport (p : SpatulaSlipParams := spatulaSlipParams) :
    HydroelasticPatchSupport :=
  HydroelasticPatchSupport.selectByArea 1.0e-12
    #[spatulaLeftPatch p, spatulaRightPatch p]
    "spatula slip-control finger patch support"

def spatulaSelectedNormalForceSum? (support : HydroelasticPatchSupport) :
    Except String Float := do
  support.validateGeometry?
  let patches ← support.selectedPatches?
  let mut total := 0.0
  for patch in patches do
    total := total + patch.normalForce
  pure total

def spatulaTorsionalFrictionTorque?
    (p : SpatulaSlipParams := spatulaSlipParams)
    (support : HydroelasticPatchSupport := spatulaPatchSupport p) :
    Except String Float := do
  p.validate?
  let normalForce ← spatulaSelectedNormalForceSum? support
  pure (p.torsionalSlipDirection *
    p.torsionalFrictionCoefficient * p.torsionalFrictionRadius * normalForce)

def spatulaTorsionalFrictionContribution?
    (p : SpatulaSlipParams := spatulaSlipParams)
    (support : HydroelasticPatchSupport := spatulaPatchSupport p) :
    Except String GeneralizedForceContribution := do
  let torque ← spatulaTorsionalFrictionTorque? p support
  pure (GeneralizedForceContribution.ofForce #[0.0, 0.0, torque]
    "spatula pressure-dependent torsional friction primitive"
    "hydroelastic spatula slip-control")

structure SpatulaSlipPhysicsState where
  params : SpatulaSlipParams := spatulaSlipParams
  time : Float := 0.0
  deriving Repr, Inhabited

namespace SpatulaSlipPhysicsState

def validate? (snapshot : SpatulaSlipPhysicsState) : Except String Unit := do
  snapshot.params.validate?
  if !snapshot.time.isFinite then
    .error s!"spatula slip-control time must be finite, got {snapshot.time}"

def support (snapshot : SpatulaSlipPhysicsState) : HydroelasticPatchSupport :=
  spatulaPatchSupport snapshot.params

def fullPhysicsPrimitives?
    (snapshot : SpatulaSlipPhysicsState)
    (label : String := "spatula slip-control hydroelastic full physics") :
    Except String FullPhysicsPrimitives := do
  snapshot.validate?
  let support := snapshot.support
  let torsionalFriction ←
    spatulaTorsionalFrictionContribution? snapshot.params support
  fullPhysicsPrimitivesFromHydroelasticSupport?
    snapshot.params.massMatrix snapshot.params.qdot
    (snapshot.params.generalizedActuation snapshot.time)
    snapshot.params.biasForces support label #[torsionalFriction]

end SpatulaSlipPhysicsState

def spatulaSlipPhysicsState
    (p : SpatulaSlipParams := spatulaSlipParams)
    (time : Float := 0.0) :
    SpatulaSlipPhysicsState :=
  { params := p, time := time }

def spatulaSlipFullPhysicsPrimitiveProvider
    (label : String := "spatula slip-control hydroelastic full physics") :
    FullPhysicsPrimitiveProvider SpatulaSlipPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => snapshot.fullPhysicsPrimitives? label
  }

def spatulaFullPhysics?
    (p : SpatulaSlipParams := spatulaSlipParams)
    (time : Float := 0.0) : Except String FullPhysicsResult := do
  (spatulaSlipFullPhysicsPrimitiveProvider
    "spatula slip-control hydroelastic full physics").solveAt?
      (spatulaSlipPhysicsState p time) 5506

def acceptedSegment (dt : Float) (label : String) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := dt
    tAfter := dt
    label := label
  }

private def localMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

structure HydroelasticDemosResult where
  references : Array DrakeReference
  documentationImages : Array HydroelasticDocumentationImage
  ballPaddleInitialState : Array Float
  ballPaddleSupport : HydroelasticPatchSupport
  ballPaddleFullPhysics : FullPhysicsResult
  pepperBowlInitialState : Array Float
  pepperTablePatch : HydroelasticContactPatch
  pepperTableFullPhysics : FullPhysicsResult
  spatulaActuationSamples : Array (Float × Array Float)
  spatulaPatchSupport : HydroelasticPatchSupport
  spatulaFullPhysics : FullPhysicsResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd?
    (bp : BallPaddleParams := ballPaddleParams)
    (nm : NonconvexMeshParams := nonconvexMeshParams)
    (sp : SpatulaSlipParams := spatulaSlipParams) :
    Except String HydroelasticDemosResult := do
  validateDocumentationImageAssets?
  bp.validate?
  nm.validate?
  sp.validate?
  let ballSupport := ballPaddleSupport bp
  ballSupport.validateGeometry?
  let ballFullPhysics ← ballPaddleFullPhysics? bp
  let pepperPatch := pepperTablePatch nm
  pepperPatch.validateGeometry?
  let pepperFullPhysics ← pepperTableFullPhysics? nm
  let spatulaSupport := spatulaPatchSupport sp
  spatulaSupport.validateGeometry?
  let spatulaFullPhysics ← spatulaFullPhysics? sp 0.0
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedSegment bp.timeStep "python ball-paddle plant advance"))
      |>.push (.interval (acceptedSegment nm.timeStep "python nonconvex mesh plant advance"))
      |>.push (.interval (acceptedSegment sp.discreteUpdatePeriod "spatula slip-control plant advance"))
  trace.validate?
  let times := #[0.0, 1.49, 1.5, 2.0, 3.1]
  let mut actuationSamples := #[]
  for t in times do
    actuationSamples := actuationSamples.push (t, sp.actuation t)
  pure {
    references := drakeReferences
    documentationImages := documentationImageAssets
    ballPaddleInitialState := bp.initialState
    ballPaddleSupport := ballSupport
    ballPaddleFullPhysics := ballFullPhysics
    pepperBowlInitialState := nm.initialState
    pepperTablePatch := pepperPatch
    pepperTableFullPhysics := pepperFullPhysics
    spatulaActuationSamples := actuationSamples
    spatulaPatchSupport := spatulaSupport
    spatulaFullPhysics := spatulaFullPhysics
    trace := trace
    moves :=
      #[
        localMove 5500 "python ball-paddle hydroelastic patch provider",
        localMove 5501 "python nonconvex pepper-bowl-table state assembly",
        localMove 5502 "spatula slip-control square-wave gripper controller",
        localMove 5503 "hydroelastic patch-to-contact full-physics adapter",
        ballFullPhysics.supportMove,
        ballFullPhysics.move,
        pepperFullPhysics.supportMove,
        pepperFullPhysics.move,
        spatulaFullPhysics.supportMove,
        spatulaFullPhysics.move,
        localMove 5507 "spatula pressure-dependent torsional friction primitive",
        localMove 5508 "hydroelastic README image documentation boundary"
      ] ++ trace.moves
  }

end Tyr.EventSkeleton.Examples.HydroelasticDemos
