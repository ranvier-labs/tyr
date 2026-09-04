import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Physics
import Tyr.EventSkeleton.Trace

/-!
# Drake-Style Four-Bar Linkage Example

This ports the core mechanics of
`../drake/examples/multibody/four_bar`: a planar four-bar linkage whose
closed loop is approximated by a `LinearBushingRollPitchYaw` force element.

The model is intentionally expressed in the current primitive language:
planar link kinematics produce COM Jacobians and loop-closure Jacobians, the
mass matrix comes from `Jᵀ m J + I`, and the cut B-C joint is closed by the
shared bushing spring-damper primitive.
-/

namespace Tyr.EventSkeleton.Examples.FourBar

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/four_bar/four_bar.sdf"
      concept := "declares links A, B, C, revolute joints WA/AB/WC, and Bc/Cb bushing frames"
    },
    {
      path := "../drake/examples/multibody/four_bar/passive_simulation.cc"
      concept := "loads the SDF, adds LinearBushingRollPitchYaw, sets qA/qB/qC, qdotA, and applied torque"
    },
    {
      path := "../drake/examples/multibody/four_bar/BUILD.bazel"
      concept := "declares the passive_simulation executable and four_bar.sdf data dependency"
    },
    {
      path := "../drake/examples/multibody/four_bar/README.md"
      concept := "derives the initial loop-closing angles and bushing stiffness/damping choices"
    },
    {
      path := "../drake/examples/multibody/four_bar/dev/four_bar_loop.sdf"
      concept := "aspirational direct closed-loop SDF with four revolute joints and no spanning-tree cut"
    },
    {
      path := "../drake/examples/multibody/four_bar/dev/four_bar_weld.sdf"
      concept := "aspirational spanning-tree SDF that splits the coupler and closes the loop with a weld constraint"
    },
    {
      path := "../drake/examples/multibody/four_bar/images/FourBarLinkageGeometry.png"
      concept := "README geometry diagram used to derive the loop-closing initial angles"
    },
    {
      path := "../drake/examples/multibody/four_bar/images/FourBarLinkageSchematic.png"
      concept := "README schematic diagram naming the crank, coupler, rocker, and ground link"
    }
  ]

def fourBarExampleRoot : String :=
  "../drake/examples/multibody/four_bar"

inductive FourBarModelVariant where
  | bushingCut
  | directLoop
  | splitCouplerWeld
  deriving Repr, BEq, Inhabited

namespace FourBarModelVariant

def label : FourBarModelVariant → String
  | .bushingCut => "bushing-cut"
  | .directLoop => "direct-loop"
  | .splitCouplerWeld => "split-coupler-weld"

def loopClosureStrategy : FourBarModelVariant → String
  | .bushingCut => "cut B-C revolute joint closed by LinearBushingRollPitchYaw"
  | .directLoop => "direct SDF kinematic loop with four revolute joints"
  | .splitCouplerWeld => "spanning tree plus weld constraint between split coupler halves"

end FourBarModelVariant

inductive FourBarExampleAssetKind where
  | metadata
  | source
  | model
  | devModel
  | image
  deriving Repr, BEq, Inhabited

inductive FourBarExampleAssetFormat where
  | bazel
  | markdown
  | cpp
  | sdf
  | png
  deriving Repr, BEq, Inhabited

namespace FourBarExampleAssetFormat

def matchesPath (format : FourBarExampleAssetFormat) (path : String) : Bool :=
  match format with
  | .bazel => path == "BUILD.bazel"
  | .markdown => path.endsWith ".md"
  | .cpp => path.endsWith ".cc"
  | .sdf => path.endsWith ".sdf"
  | .png => path.endsWith ".png"

end FourBarExampleAssetFormat

/--
File manifest for Drake's `examples/multibody/four_bar` tree.

The active executable loads `four_bar.sdf` and closes the cut B-C joint with the
shared LinearBushingRollPitchYaw primitive below.  The manifest also records the
two dev SDF variants included by Drake's Bazel `models` filegroup, because they
describe the future direct-loop and constraint-based provider shapes that should
lower into the same primitive physics boundary.
-/
structure FourBarExampleAsset where
  relativePath : String
  format : FourBarExampleAssetFormat
  kind : FourBarExampleAssetKind
  component : String
  inModelFilegroup : Bool := false
  activePassiveModel : Bool := false
  modelVariant? : Option FourBarModelVariant := none
  localDependencies : Array String := #[]
  concept : String := ""
  deriving Repr, Inhabited

namespace FourBarExampleAsset

def fullPath (asset : FourBarExampleAsset) : String :=
  fourBarExampleRoot ++ "/" ++ asset.relativePath

def validate? (asset : FourBarExampleAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "FourBar asset path cannot be empty"
  if !asset.format.matchesPath asset.relativePath then
    .error s!"FourBar asset {asset.relativePath}: format does not match path"
  if asset.component.isEmpty then
    .error s!"FourBar asset {asset.relativePath}: component cannot be empty"
  if asset.concept.isEmpty then
    .error s!"FourBar asset {asset.relativePath}: concept cannot be empty"
  if asset.inModelFilegroup && asset.format != .sdf then
    .error s!"FourBar asset {asset.relativePath}: only SDF assets should be in the Bazel models filegroup"
  if asset.activePassiveModel && asset.modelVariant? != some FourBarModelVariant.bushingCut then
    .error s!"FourBar asset {asset.relativePath}: active passive model should be the bushing-cut variant"
  if asset.modelVariant?.isSome && asset.format != .sdf then
    .error s!"FourBar asset {asset.relativePath}: only SDF assets should declare a model variant"
  for dep in asset.localDependencies do
    if dep.isEmpty then
      .error s!"FourBar asset {asset.relativePath}: local dependency cannot be empty"
    if dep == asset.relativePath then
      .error s!"FourBar asset {asset.relativePath}: cannot depend on itself"

end FourBarExampleAsset

def fourBarExampleAssets : Array FourBarExampleAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      format := .bazel
      kind := .metadata
      component := "build"
      localDependencies := #[
        "passive_simulation.cc",
        "four_bar.sdf",
        "dev/four_bar_loop.sdf",
        "dev/four_bar_weld.sdf"
      ]
      concept := "Bazel passive_simulation target plus models filegroup globbing all SDF variants"
    },
    {
      relativePath := "README.md"
      format := .markdown
      kind := .metadata
      component := "docs"
      localDependencies := #[
        "images/FourBarLinkageSchematic.png",
        "images/FourBarLinkageGeometry.png"
      ]
      concept := "geometry derivation and modeling notes for the bushing-closed four-bar"
    },
    {
      relativePath := "passive_simulation.cc"
      format := .cpp
      kind := .source
      component := "simulation"
      localDependencies := #["four_bar.sdf"]
      concept := "loads four_bar.sdf, adds LinearBushingRollPitchYaw, fixes torque, and advances Simulator"
    },
    {
      relativePath := "four_bar.sdf"
      format := .sdf
      kind := .model
      component := "model"
      inModelFilegroup := true
      activePassiveModel := true
      modelVariant? := some FourBarModelVariant.bushingCut
      concept := "three-link spanning-tree model with Bc/Cb frames for the bushing force element"
    },
    {
      relativePath := "dev/four_bar_loop.sdf"
      format := .sdf
      kind := .devModel
      component := "model"
      inModelFilegroup := true
      modelVariant? := some FourBarModelVariant.directLoop
      concept := "aspirational direct loop model with four revolute joints"
    },
    {
      relativePath := "dev/four_bar_weld.sdf"
      format := .sdf
      kind := .devModel
      component := "model"
      inModelFilegroup := true
      modelVariant? := some FourBarModelVariant.splitCouplerWeld
      localDependencies := #["dev/four_bar_loop.sdf"]
      concept := "aspirational split-coupler tree model closed by a weld constraint"
    },
    {
      relativePath := "images/FourBarLinkageGeometry.png"
      format := .png
      kind := .image
      component := "docs"
      concept := "geometry diagram used by README.md to derive atan2(sqrt(15), 1)"
    },
    {
      relativePath := "images/FourBarLinkageSchematic.png"
      format := .png
      kind := .image
      component := "docs"
      concept := "schematic diagram naming the four-bar links and angle conventions"
    }
  ]

private def hasDuplicateFourBarAssetPath : Bool := Id.run do
  let mut seen : Array String := #[]
  for asset in fourBarExampleAssets do
    if seen.contains asset.relativePath then
      return true
    seen := seen.push asset.relativePath
  return false

def fourBarExampleAssetPaths : Array String :=
  fourBarExampleAssets.map (fun asset => asset.relativePath)

def fourBarModelFilegroupAssets : Array FourBarExampleAsset :=
  fourBarExampleAssets.filter (fun asset => asset.inModelFilegroup)

def fourBarModelVariantAssets : Array FourBarExampleAsset :=
  fourBarExampleAssets.filter (fun asset => asset.modelVariant?.isSome)

def fourBarImageAssets : Array FourBarExampleAsset :=
  fourBarExampleAssets.filter (fun asset => asset.kind == .image)

def findFourBarExampleAsset? (relativePath : String) :
    Option FourBarExampleAsset :=
  fourBarExampleAssets.find? (fun asset => asset.relativePath == relativePath)

def requiredFourBarExampleAssetPaths : Array String :=
  #[
    "BUILD.bazel",
    "README.md",
    "passive_simulation.cc",
    "four_bar.sdf",
    "dev/four_bar_loop.sdf",
    "dev/four_bar_weld.sdf",
    "images/FourBarLinkageGeometry.png",
    "images/FourBarLinkageSchematic.png"
  ]

def validateFourBarExampleAssetCatalog? : Except String Unit := do
  if fourBarExampleAssets.size != 8 then
    .error s!"FourBar asset catalog should contain 8 files, got {fourBarExampleAssets.size}"
  if hasDuplicateFourBarAssetPath then
    .error "FourBar asset catalog contains duplicate paths"
  for asset in fourBarExampleAssets do
    asset.validate?
  for path in requiredFourBarExampleAssetPaths do
    if !(fourBarExampleAssetPaths.contains path) then
      .error s!"FourBar asset catalog is missing {path}"
  for asset in fourBarExampleAssets do
    for dep in asset.localDependencies do
      if !(fourBarExampleAssetPaths.contains dep) then
        .error s!"FourBar asset {asset.relativePath}: missing local dependency {dep}"
  if fourBarModelFilegroupAssets.size != 3 then
    .error s!"FourBar Bazel models filegroup should contain 3 SDF assets, got {fourBarModelFilegroupAssets.size}"
  if fourBarModelVariantAssets.size != 3 then
    .error s!"FourBar model variant count should be 3, got {fourBarModelVariantAssets.size}"
  if fourBarImageAssets.size != 2 then
    .error s!"FourBar README image asset count should be 2, got {fourBarImageAssets.size}"
  if (fourBarExampleAssets.filter (fun asset => asset.activePassiveModel)).size != 1 then
    .error "FourBar should have exactly one active passive_simulation model"

def stateCoordinateNames : Array String :=
  #["qA", "qB", "qC", "qAdot", "qBdot", "qCdot"]

def inputCoordinateNames : Array String :=
  #["applied_torque"]

def jointNames : Array String :=
  #["joint_WA", "joint_AB", "joint_WC"]

def bushingFrameNames : Array String :=
  #["Bc_bushing", "Cb_bushing"]

def fourBarModelUri : String :=
  "package://drake/examples/multibody/four_bar/four_bar.sdf"

structure FourBarModelAssetBoundary where
  modelName : String := "four_bar"
  sdfPath : String := "../drake/examples/multibody/four_bar/four_bar.sdf"
  packageUri : String := fourBarModelUri
  linkNames : Array String := #["A", "B", "C"]
  jointNames : Array String := #["joint_WA", "joint_AB", "joint_WC"]
  jointTypes : Array String := #["revolute", "revolute", "revolute"]
  jointAxes : Array (Array Float) :=
    #[#[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0]]
  actuatedJointNames : Array String := #["joint_WA"]
  unactuatedJointNames : Array String := #["joint_AB", "joint_WC"]
  bushingFrameNames : Array String := #["Bc_bushing", "Cb_bushing"]
  bushingFrameAttachedTo : Array String := #["B", "C"]
  bushingFramePoseInAttached : Array (Array Float) :=
    #[
      #[4.0, 0.0, 0.0, -1.57079632679, 0.0, 0.0],
      #[4.0, 0.0, 0.0, -1.57079632679, 0.0, 0.0]
    ]
  linkBoxSize : Array Float := #[4.2, 0.1, 0.2]
  linkMass : Float := 20.0
  linkIyy : Float := 29.46666666666666
  deriving Repr, Inhabited

namespace FourBarModelAssetBoundary

private def finiteArray (xs : Array Float) : Bool :=
  xs.all Float.isFinite

private def validatePoseArray? (poses : Array (Array Float)) :
    Except String Unit := do
  if poses.size != 2 then
    .error s!"FourBar bushing frame pose array should have two entries, got {poses.size}"
  for i in [:poses.size] do
    let pose := poses[i]!
    if pose.size != 6 || !finiteArray pose then
      .error s!"FourBar bushing frame pose[{i}] must have six finite entries, got {pose}"

def validate? (boundary : FourBarModelAssetBoundary) : Except String Unit := do
  if boundary.modelName != "four_bar" then
    .error s!"FourBar model name mismatch: {boundary.modelName}"
  if boundary.sdfPath != "../drake/examples/multibody/four_bar/four_bar.sdf" then
    .error s!"FourBar SDF path mismatch: {boundary.sdfPath}"
  if boundary.packageUri != fourBarModelUri then
    .error s!"FourBar package URI mismatch: {boundary.packageUri}"
  if boundary.linkNames != #["A", "B", "C"] then
    .error s!"FourBar link names mismatch: {boundary.linkNames}"
  if boundary.jointNames != #["joint_WA", "joint_AB", "joint_WC"] then
    .error s!"FourBar joint names mismatch: {boundary.jointNames}"
  if boundary.jointTypes != #["revolute", "revolute", "revolute"] then
    .error s!"FourBar joint types mismatch: {boundary.jointTypes}"
  if boundary.jointAxes != #[#[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0], #[0.0, 1.0, 0.0]] then
    .error s!"FourBar joint axes mismatch: {boundary.jointAxes}"
  if boundary.actuatedJointNames != #["joint_WA"] then
    .error s!"FourBar actuated joints mismatch: {boundary.actuatedJointNames}"
  if boundary.unactuatedJointNames != #["joint_AB", "joint_WC"] then
    .error s!"FourBar unactuated joints mismatch: {boundary.unactuatedJointNames}"
  if boundary.bushingFrameNames != #["Bc_bushing", "Cb_bushing"] then
    .error s!"FourBar bushing frame names mismatch: {boundary.bushingFrameNames}"
  if boundary.bushingFrameAttachedTo != #["B", "C"] then
    .error s!"FourBar bushing frame attachments mismatch: {boundary.bushingFrameAttachedTo}"
  validatePoseArray? boundary.bushingFramePoseInAttached
  if boundary.linkBoxSize.size != 3 || !finiteArray boundary.linkBoxSize then
    .error s!"FourBar link box size must have three finite entries, got {boundary.linkBoxSize}"
  if !boundary.linkMass.isFinite || boundary.linkMass <= 0.0 then
    .error s!"FourBar link mass must be positive and finite, got {boundary.linkMass}"
  if !boundary.linkIyy.isFinite || boundary.linkIyy <= 0.0 then
    .error s!"FourBar link Iyy must be positive and finite, got {boundary.linkIyy}"

end FourBarModelAssetBoundary

def fourBarModelAssetBoundary : FourBarModelAssetBoundary := {}

structure Vec2 where
  x : Float := 0.0
  z : Float := 0.0
  deriving Repr, Inhabited

namespace Vec2

def add (a b : Vec2) : Vec2 :=
  { x := a.x + b.x, z := a.z + b.z }

def sub (a b : Vec2) : Vec2 :=
  { x := a.x - b.x, z := a.z - b.z }

def scale (s : Float) (v : Vec2) : Vec2 :=
  { x := s * v.x, z := s * v.z }

def dot (a b : Vec2) : Float :=
  a.x * b.x + a.z * b.z

def norm (v : Vec2) : Float :=
  Float.sqrt (v.dot v)

end Vec2

def unit (theta : Float) : Vec2 :=
  { x := Float.cos theta, z := Float.sin theta }

def dUnit (theta : Float) : Vec2 :=
  { x := -Float.sin theta, z := Float.cos theta }

structure FourBarParams where
  linkLength : Float := 4.0
  groundLength : Float := 2.0
  linkMass : Float := 20.0
  linkIyy : Float := 29.46666666666666
  gravity : Float := 9.8
  forceStiffness : Float := 30000.0
  forceDamping : Float := 1500.0
  torqueStiffness : Float := 30000.0
  torqueDamping : Float := 1500.0
  appliedTorque : Float := 0.0
  initialVelocity : Float := 3.0
  stepSize : Float := 1.0e-5
  simulationTime : Float := 10.0
  deriving Repr, Inhabited

def params : FourBarParams := {}

namespace FourBarParams

def bushingParams (p : FourBarParams) : LinearBushingRollPitchYawParams :=
  LinearBushingRollPitchYawParams.fourBarPlanarRevolute
    p.forceStiffness p.forceDamping p.torqueStiffness p.torqueDamping

end FourBarParams

structure FourBarState where
  qA : Float
  qB : Float
  qC : Float
  qAdot : Float := 0.0
  qBdot : Float := 0.0
  qCdot : Float := 0.0
  deriving Repr, Inhabited

namespace FourBarState

def qdot (x : FourBarState) : Array Float :=
  #[x.qAdot, x.qBdot, x.qCdot]

def isFinite (x : FourBarState) : Bool :=
  Float.isFinite x.qA &&
  Float.isFinite x.qB &&
  Float.isFinite x.qC &&
  Float.isFinite x.qAdot &&
  Float.isFinite x.qBdot &&
  Float.isFinite x.qCdot

end FourBarState

/-- `atan2(sqrt(15), 1)` from Drake's README. -/
def initialQA : Float := 1.318116071652818

def initialQB : Float := pi - initialQA

def initialQC : Float := initialQB

def defaultState (p : FourBarParams := params) : FourBarState :=
  {
    qA := initialQA
    qB := initialQB
    qC := initialQC
    qAdot := p.initialVelocity
    qBdot := 0.0
    qCdot := 0.0
  }

def thetaB (x : FourBarState) : Float :=
  x.qA + x.qB

def worldPivotA : Vec2 := {}

def worldPivotC (p : FourBarParams) : Vec2 :=
  { x := -p.groundLength, z := 0.0 }

def jointB0 (p : FourBarParams) (x : FourBarState) : Vec2 :=
  Vec2.scale p.linkLength (unit x.qA)

def endpointBc (p : FourBarParams) (x : FourBarState) : Vec2 :=
  (jointB0 p x).add (Vec2.scale p.linkLength (unit (thetaB x)))

def endpointCb (p : FourBarParams) (x : FourBarState) : Vec2 :=
  (worldPivotC p).add (Vec2.scale p.linkLength (unit x.qC))

def loopClosureError (p : FourBarParams) (x : FourBarState) : Vec2 :=
  (endpointBc p x).sub (endpointCb p x)

def loopClosureErrorNorm (p : FourBarParams) (x : FourBarState) : Float :=
  (loopClosureError p x).norm

def comA (p : FourBarParams) (x : FourBarState) : Vec2 :=
  Vec2.scale (0.5 * p.linkLength) (unit x.qA)

def comB (p : FourBarParams) (x : FourBarState) : Vec2 :=
  (jointB0 p x).add (Vec2.scale (0.5 * p.linkLength) (unit (thetaB x)))

def comC (p : FourBarParams) (x : FourBarState) : Vec2 :=
  (worldPivotC p).add (Vec2.scale (0.5 * p.linkLength) (unit x.qC))

private def zeroVec2 : Vec2 := {}

def comJacobians (p : FourBarParams) (x : FourBarState) :
    Array (Array Vec2) :=
  let l := p.linkLength
  let dA := dUnit x.qA
  let dB := dUnit (thetaB x)
  let dC := dUnit x.qC
  let aRows :=
    #[Vec2.scale (0.5 * l) dA, zeroVec2, zeroVec2]
  let bRows :=
    #[Vec2.add (Vec2.scale l dA) (Vec2.scale (0.5 * l) dB),
      Vec2.scale (0.5 * l) dB, zeroVec2]
  let cRows :=
    #[zeroVec2, zeroVec2, Vec2.scale (0.5 * l) dC]
  #[aRows, bRows, cRows]

def angularVelocityCoefficients : Array (Array Float) :=
  #[
    #[1.0, 0.0, 0.0],
    #[1.0, 1.0, 0.0],
    #[0.0, 0.0, 1.0]
  ]

def massMatrix (p : FourBarParams) (x : FourBarState) :
    Array (Array Float) := Id.run do
  let bodyJacs := comJacobians p x
  let angular := angularVelocityCoefficients
  let mut rows : Array (Array Float) := #[]
  for i in [:3] do
    let mut row : Array Float := #[]
    for j in [:3] do
      let mut mij := 0.0
      for body in [:bodyJacs.size] do
        let jacs := bodyJacs[body]!
        mij := mij + p.linkMass * (jacs[i]!.dot jacs[j]!)
        mij := mij + p.linkIyy *
          ((angular[body]!).getD i 0.0) * ((angular[body]!).getD j 0.0)
      row := row.push mij
    rows := rows.push row
  return rows

def loopClosureJacobianRows (p : FourBarParams) (x : FourBarState) :
    Array (Array Float) :=
  let l := p.linkLength
  let sA := Float.sin x.qA
  let cA := Float.cos x.qA
  let sB := Float.sin (thetaB x)
  let cB := Float.cos (thetaB x)
  let sC := Float.sin x.qC
  let cC := Float.cos x.qC
  #[
    #[-l * sA - l * sB, -l * sB, l * sC],
    #[0.0, 0.0, 0.0],
    #[l * cA + l * cB, l * cB, -l * cC]
  ]

def loopClosureVelocity (p : FourBarParams) (x : FourBarState) : Vec2 :=
  let rows := loopClosureJacobianRows p x
  let v := FloatMatrix.matVec rows x.qdot
  { x := v.getD 0 0.0, z := v.getD 2 0.0 }

def yawError (x : FourBarState) : Float :=
  thetaB x - x.qC

def yawRateError (x : FourBarState) : Float :=
  x.qAdot + x.qBdot - x.qCdot

def bushingState (p : FourBarParams) (x : FourBarState) :
    LinearBushingRollPitchYawState :=
  let err := loopClosureError p x
  let rows := loopClosureJacobianRows p x
  {
    rpyError := #[0.0, 0.0, yawError x]
    angularVelocityError := #[0.0, 0.0, yawRateError x]
    translationError := #[err.x, 0.0, err.z]
    translationVelocityError := FloatMatrix.matVec rows x.qdot
    rpyJacobian := #[#[0.0, 0.0, 0.0], #[0.0, 0.0, 0.0], #[1.0, 1.0, -1.0]]
    translationJacobian := rows
    label := "four-bar Bc/Cb bushing"
  }

def bushingResult? (p : FourBarParams) (x : FourBarState) :
    Except String LinearBushingRollPitchYawResult :=
  LinearBushingRollPitchYaw.evaluate? 3 (p.bushingParams) (bushingState p x)

def gravityGeneralizedForces (p : FourBarParams) (x : FourBarState) :
    Array Float :=
  let l := p.linkLength
  let theta := thetaB x
  let qAForce :=
    -p.linkMass * p.gravity *
      ((0.5 * l * Float.cos x.qA) +
       (l * Float.cos x.qA + 0.5 * l * Float.cos theta))
  let qBForce :=
    -p.linkMass * p.gravity * (0.5 * l * Float.cos theta)
  let qCForce :=
    -p.linkMass * p.gravity * (0.5 * l * Float.cos x.qC)
  #[qAForce, qBForce, qCForce]

def appliedGeneralizedForces (p : FourBarParams) : Array Float :=
  #[p.appliedTorque, 0.0, 0.0]

def totalGeneralizedForces?
    (p : FourBarParams) (x : FourBarState) :
    Except String (Array Float × LinearBushingRollPitchYawResult) := do
  let bushing ← bushingResult? p x
  pure
    (FloatArray.add
      (FloatArray.add (gravityGeneralizedForces p x) (appliedGeneralizedForces p))
      bushing.generalizedForce,
      bushing)

def fullPhysicsPrimitives?
    (p : FourBarParams)
    (x : FourBarState)
    (label : String := "four-bar") :
    Except String (FullPhysicsPrimitives × LinearBushingRollPitchYawResult) := do
  let bushing ← bushingResult? p x
  pure ({
    massMatrix := massMatrix p x
    qdot := x.qdot
    actuationForces := appliedGeneralizedForces p
    generalizedForceContributions := #[
      GeneralizedForceContribution.ofForce
        bushing.generalizedForce
        "four-bar LinearBushingRollPitchYaw generalized force"
        "LinearBushingRollPitchYaw"
    ]
    biasForces := (gravityGeneralizedForces p x).map (fun g => -g)
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := label
  }, bushing)

def bushingFullPhysicsPrimitiveProvider
    (p : FourBarParams := params)
    (label : String := "four-bar bushing full physics provider") :
    FullPhysicsPrimitiveProvider FourBarState :=
  {
    label := label
    primitivesAt? := fun x => do
      let (primitive, _) ← fullPhysicsPrimitives? p x label
      pure primitive
  }

def loopClosureConstraintRows (p : FourBarParams) (x : FourBarState) :
    Array (Array Float) :=
  let rows := loopClosureJacobianRows p x
  #[rows.getD 0 #[], rows.getD 2 #[]]

def loopClosureConstraintVelocity (p : FourBarParams) (x : FourBarState) :
    Array Float :=
  let v := loopClosureVelocity p x
  #[v.x, v.z]

def idealLoopConstraintPrimitive
    (p : FourBarParams)
    (x : FourBarState)
    (id : Nat := 5453)
    (label : String := "four-bar ideal loop-closure bilateral constraint") :
    BilateralConstraintPrimitive :=
  {
    id := id
    jacobian := loopClosureConstraintRows p x
    targetAcceleration := #[]
    label := label
  }

def idealLoopFullPhysicsPrimitives?
    (p : FourBarParams)
    (x : FourBarState)
    (label : String := "four-bar ideal loop-closure full physics") :
    Except String FullPhysicsPrimitives := do
  let constraint := idealLoopConstraintPrimitive p x
  pure {
    massMatrix := massMatrix p x
    qdot := x.qdot
    actuationForces := appliedGeneralizedForces p
    generalizedForceContributions := #[]
    biasForces := (gravityGeneralizedForces p x).map (fun g => -g)
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    bilateralConstraints := #[constraint]
    label := label
  }

def idealLoopFullPhysicsPrimitiveProvider
    (p : FourBarParams := params)
    (label : String := "four-bar ideal loop-closure full physics provider") :
    FullPhysicsPrimitiveProvider FourBarState :=
  {
    label := label
    primitivesAt? := fun x => idealLoopFullPhysicsPrimitives? p x label
  }

def solveFullPhysics?
    (p : FourBarParams)
    (x : FourBarState)
    (intervalVertex : VertexId := 5452)
    (label : String := "four-bar") :
    Except String (FullPhysicsResult × LinearBushingRollPitchYawResult) := do
  let (primitive, bushing) ← fullPhysicsPrimitives? p x label
  let result ← primitive.solve? intervalVertex
  pure (result, bushing)

def solveIdealLoopFullPhysics?
    (p : FourBarParams)
    (x : FourBarState)
    (intervalVertex : VertexId := 5453)
    (label : String := "four-bar ideal loop-closure full physics") :
    Except String FullPhysicsResult := do
  let primitive ← idealLoopFullPhysicsPrimitives? p x label
  primitive.solve? intervalVertex

def derivative? (p : FourBarParams) (x : FourBarState) :
    Except String (FourBarState × LinearBushingRollPitchYawResult) := do
  let (forces, bushing) ← totalGeneralizedForces? p x
  let qdd ← DenseLinearAlgebra.solveLinear? (massMatrix p x) forces
  pure ({
    qA := x.qAdot
    qB := x.qBdot
    qC := x.qCdot
    qAdot := qdd.getD 0 0.0
    qBdot := qdd.getD 1 0.0
    qCdot := qdd.getD 2 0.0
  }, bushing)

def addScaledState (x dx : FourBarState) (dt : Float) : FourBarState :=
  {
    qA := x.qA + dt * dx.qA
    qB := x.qB + dt * dx.qB
    qC := x.qC + dt * dx.qC
    qAdot := x.qAdot + dt * dx.qAdot
    qBdot := x.qBdot + dt * dx.qBdot
    qCdot := x.qCdot + dt * dx.qCdot
  }

def eulerStep? (p : FourBarParams) (dt : Float) (x : FourBarState) :
    Except String (FourBarState × FourBarState × LinearBushingRollPitchYawResult) := do
  let (dx, bushing) ← derivative? p x
  pure (addScaledState x dx dt, dx, bushing)

def simulateSteps? (p : FourBarParams) (steps : Nat)
    (x0 : FourBarState := defaultState p) :
    Except String FourBarState := do
  let mut x := x0
  for _ in [:steps] do
    let (next, _, _) ← eulerStep? p p.stepSize x
    x := next
  pure x

def kineticEnergy (p : FourBarParams) (x : FourBarState) : Float :=
  let v := x.qdot
  0.5 * FloatArray.dot v (FloatMatrix.matVec (massMatrix p x) v)

def potentialEnergy (p : FourBarParams) (x : FourBarState) : Float :=
  p.linkMass * p.gravity * ((comA p x).z + (comB p x).z + (comC p x).z)

def acceptedSegment (dt : Float) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := dt
    tAfter := dt
    label := "four-bar bushing-constrained interval"
  }

structure MultibodyFourBarConfig where
  targetRealtimeRate : Float := 1.0
  simulationTime : Float := params.simulationTime
  timeStep : Float := 0.0
  visualizationEnabled : Bool := true
  deriving Repr, Inhabited

namespace MultibodyFourBarConfig

def validate? (cfg : MultibodyFourBarConfig) : Except String Unit := do
  if !cfg.targetRealtimeRate.isFinite || cfg.targetRealtimeRate < 0.0 then
    .error s!"four-bar target_realtime_rate must be nonnegative and finite, got {cfg.targetRealtimeRate}"
  if !cfg.simulationTime.isFinite || cfg.simulationTime <= 0.0 then
    .error s!"four-bar simulation_time must be positive and finite, got {cfg.simulationTime}"
  if !cfg.timeStep.isFinite || cfg.timeStep < 0.0 then
    .error s!"four-bar time_step must be nonnegative and finite, got {cfg.timeStep}"

end MultibodyFourBarConfig

def multibodyFourBarConfig : MultibodyFourBarConfig := {}

def multibodyFourBarModel : FullMultibodyPlantModel :=
  {
    modelName := "four_bar"
    modelUri := fourBarModelUri
    numPositions := 3
    numVelocities := 3
    numActuatedDofs := 1
    finalized := true
    label := "parsed four-bar SDF model"
  }

def multibodyFourBarPlantConfig (cfg : MultibodyFourBarConfig) :
    MultibodyPlantConfigPrimitive :=
  {
    timeStep := cfg.timeStep
    penetrationAllowance := 0.0
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def multibodyFourBarPassiveStep
    (cfg : MultibodyFourBarConfig := multibodyFourBarConfig)
    (p : FourBarParams := params) :
    FullMultibodyPlantStep :=
  {
    model := multibodyFourBarModel
    config := multibodyFourBarPlantConfig cfg
    q0 := #[initialQA, initialQB, initialQC]
    v0 := #[p.initialVelocity, 0.0, 0.0]
    actuation := #[p.appliedTorque]
    t0 := 0.0
    t1 := cfg.simulationTime
    label := "multibody-four-bar-passive-full-plant"
  }

private def multibodySegment (t1 : Float) : AcceptedStepSegment :=
  {
    id := 5452
    attemptIndex := 0
    tStart := 0.0
    tAttempt := t1
    tAfter := t1
    label := "multibody four-bar passive Simulator.AdvanceTo"
  }

private def multibodyLocalMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

def multibodyFourBarMoves (cfg : MultibodyFourBarConfig)
    (p : FourBarParams := params) : Array SkeletonMove :=
  #[
    multibodyLocalMove 5450
      "Parser.AddModelsFromUrl four_bar.sdf + MultibodyPlant.Finalize",
    multibodyLocalMove 5451
      s!"Add LinearBushingRollPitchYaw k={p.forceStiffness}, d={p.forceDamping}, visualization={cfg.visualizationEnabled}"
  ]

structure MultibodyFourBarResult where
  references : Array DrakeReference
  assetCatalog : Array FourBarExampleAsset
  asset : FourBarModelAssetBoundary
  config : MultibodyFourBarConfig
  step : FullMultibodyPlantStep
  fullPhysics : FullPhysicsResult
  bushing : LinearBushingRollPitchYawResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildMultibodyFourBar?
    (cfg : MultibodyFourBarConfig := multibodyFourBarConfig)
    (p : FourBarParams := params)
    (asset : FourBarModelAssetBoundary := fourBarModelAssetBoundary) :
    Except String MultibodyFourBarResult := do
  validateFourBarExampleAssetCatalog?
  cfg.validate?
  asset.validate?
  let step := multibodyFourBarPassiveStep cfg p
  step.validate?
  let x0 : FourBarState := {
    qA := step.q0.getD 0 0.0
    qB := step.q0.getD 1 0.0
    qC := step.q0.getD 2 0.0
    qAdot := step.v0.getD 0 0.0
    qBdot := step.v0.getD 1 0.0
    qCdot := step.v0.getD 2 0.0
  }
  let (fullPhysics, bushing) ← solveFullPhysics? p x0 5452
    "multibody four-bar passive benchmark plant"
  let trace := DynamicEventTrace.empty.push (.interval (multibodySegment cfg.simulationTime))
  trace.validate?
  pure {
    references := drakeReferences
    assetCatalog := fourBarExampleAssets
    asset := asset
    config := cfg
    step := step
    fullPhysics := fullPhysics
    bushing := bushing
    trace := trace
    moves := multibodyFourBarMoves cfg p ++ #[fullPhysics.supportMove, fullPhysics.move] ++ trace.moves
  }

structure FourBarResult where
  references : Array DrakeReference
  assetCatalog : Array FourBarExampleAsset
  initialState : FourBarState
  derivative : FourBarState
  bushing : LinearBushingRollPitchYawResult
  oneStepState : FourBarState
  rolloutState : FourBarState
  loopError : Vec2
  loopVelocity : Vec2
  initialEnergy : Float
  oneStepEnergy : Float
  fullPlant : MultibodyFourBarResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd? (p : FourBarParams := params) :
    Except String FourBarResult := do
  validateFourBarExampleAssetCatalog?
  let x0 := defaultState p
  if !x0.isFinite then
    .error "four-bar initial state is not finite"
  let (dx, bushing) ← derivative? p x0
  let (x1, _, _) ← eulerStep? p p.stepSize x0
  let rollout ← simulateSteps? p 10 x0
  let fullPlant ← buildMultibodyFourBar? multibodyFourBarConfig p
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedSegment p.stepSize))
  trace.validate?
  pure {
    references := drakeReferences
    assetCatalog := fourBarExampleAssets
    initialState := x0
    derivative := dx
    bushing := bushing
    oneStepState := x1
    rolloutState := rollout
    loopError := loopClosureError p x0
    loopVelocity := loopClosureVelocity p x0
    initialEnergy := kineticEnergy p x0 + potentialEnergy p x0
    oneStepEnergy := kineticEnergy p x1 + potentialEnergy p x1
    fullPlant := fullPlant
    trace := trace
    moves := trace.moves ++ fullPlant.moves
  }

end Tyr.EventSkeleton.Examples.FourBar
