import Tyr.EventSkeleton.Trace

/-!
# Drake Multibody Deformable Event-Skeleton Example

This ports the small, local pieces of
`../drake/examples/multibody/deformable`: the point-source force density field,
the time-programmed parallel gripper controller, the suction-cup controller, and
the subdivision demo's force-sampling boundary.

The full FEM state, deformable contact, and SAP solve remain visible as
controlled solver boundaries.  The implemented primitives are the exact
force/controller computations that Drake exposes in these example files.
-/

namespace Tyr.EventSkeleton.Examples.Deformable

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/deformable/BUILD.bazel"
      concept := "declares deformable demo binaries and model data dependencies"
    },
    {
      path := "../drake/examples/multibody/deformable/point_source_force_field.cc"
      concept := "evaluates a point-source force density that decays linearly with distance"
    },
    {
      path := "../drake/examples/multibody/deformable/point_source_force_field.h"
      concept := "declares the force-density input port and cached source position"
    },
    {
      path := "../drake/examples/multibody/deformable/deformable_common.cc"
      concept := "registers deformable torus material config and rigid-ground contact material"
    },
    {
      path := "../drake/examples/multibody/deformable/deformable_common.h"
      concept := "declares shared deformable setup helpers and material parameters"
    },
    {
      path := "../drake/examples/multibody/deformable/models/deformable_torus.sdf"
      concept := "records torus mesh, material parameters, damping, mass density, and contact dissipation"
    },
    {
      path := "../drake/examples/multibody/deformable/models/bubbles.sdf"
      concept := "records deformable bubble membrane meshes, poses, material parameters, and contact dissipation"
    },
    {
      path := "../drake/examples/multibody/deformable/models/deformable_teddy.sdf"
      concept := "records the deformable teddy mesh, scale, material parameters, and contact friction"
    },
    {
      path := "../drake/examples/multibody/deformable/models/simple_gripper.sdf"
      concept := "records rigid gripper links, inertias, prismatic joints, mimic coupling, and hydroelastic pads"
    },
    {
      path := "../drake/examples/multibody/deformable/test/point_source_force_field_test.cc"
      concept := "checks force direction, falloff, zero outside range, and zero unconnected input"
    },
    {
      path := "../drake/examples/multibody/deformable/parallel_gripper_controller.cc"
      concept := "time-programmed close-lift-hold-open desired state for a parallel jaw gripper"
    },
    {
      path := "../drake/examples/multibody/deformable/parallel_gripper_controller.h"
      concept := "declares the parallel gripper desired-state system"
    },
    {
      path := "../drake/examples/multibody/deformable/suction_cup_controller.cc"
      concept := "time-programmed suction cup desired state and maximum force-density output"
    },
    {
      path := "../drake/examples/multibody/deformable/suction_cup_controller.h"
      concept := "declares the suction cup controller and force-density output"
    },
    {
      path := "../drake/examples/multibody/deformable/deformable_torus.cc"
      concept := "builds the deformable torus demo with selectable parallel or suction gripper"
    },
    {
      path := "../drake/examples/multibody/deformable/deformable_subdivision.cc"
      concept := "demonstrates how element subdivision improves concentrated force integration"
    },
    {
      path := "../drake/examples/multibody/deformable/bubble_gripper.cc"
      concept := "sets up deformable bubble-gripper contact and rendering boundaries"
    },
    {
      path := "../drake/examples/multibody/deformable/deformable_disabling.cc"
      concept := "enables and disables deformable models during a contact simulation"
    },
    {
      path := "../drake/examples/multibody/deformable/README.md"
      concept := "documents the deformable torus, bubble gripper, disabling, and subdivision demos"
    }
  ]

def deformableExampleRoot : String :=
  "../drake/examples/multibody/deformable"

inductive DeformableExampleAssetKind where
  | metadata
  | source
  | model
  | mesh
  | test
  deriving Repr, BEq, Inhabited

inductive DeformableExampleAssetFormat where
  | bazel
  | markdown
  | cpp
  | header
  | sdf
  | vtk
  deriving Repr, BEq, Inhabited

namespace DeformableExampleAssetFormat

def matchesPath (format : DeformableExampleAssetFormat) (path : String) : Bool :=
  match format with
  | .bazel => path == "BUILD.bazel"
  | .markdown => path.endsWith ".md"
  | .cpp => path.endsWith ".cc"
  | .header => path.endsWith ".h"
  | .sdf => path.endsWith ".sdf"
  | .vtk => path.endsWith ".vtk"

end DeformableExampleAssetFormat

/--
File manifest for Drake's `examples/multibody/deformable` tree.

This is a provider catalog, not a FEM backend.  It records the SDF/VTK and
source dependency closure needed before a parser or SceneGraph-backed provider
can lower deformable assets into the primitive force/contact interfaces used
below.
-/
structure DeformableExampleAsset where
  relativePath : String
  format : DeformableExampleAssetFormat
  kind : DeformableExampleAssetKind
  component : String
  feedsDeformablePlant : Bool := false
  localDependencies : Array String := #[]
  externalDependencies : Array String := #[]
  concept : String := ""
  deriving Repr, Inhabited

namespace DeformableExampleAsset

def fullPath (asset : DeformableExampleAsset) : String :=
  deformableExampleRoot ++ "/" ++ asset.relativePath

def validate? (asset : DeformableExampleAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "deformable asset path cannot be empty"
  if !asset.format.matchesPath asset.relativePath then
    .error s!"deformable asset {asset.relativePath}: format does not match path"
  if asset.component.isEmpty then
    .error s!"deformable asset {asset.relativePath}: component cannot be empty"
  if asset.concept.isEmpty then
    .error s!"deformable asset {asset.relativePath}: concept cannot be empty"
  for dep in asset.localDependencies do
    if dep.isEmpty then
      .error s!"deformable asset {asset.relativePath}: local dependency cannot be empty"
    if dep == asset.relativePath then
      .error s!"deformable asset {asset.relativePath}: cannot depend on itself"
  for dep in asset.externalDependencies do
    if dep.isEmpty then
      .error s!"deformable asset {asset.relativePath}: external dependency cannot be empty"

end DeformableExampleAsset

def deformableExampleAssets : Array DeformableExampleAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      format := .bazel
      kind := .metadata
      component := "build"
      localDependencies := #[
        "models/bubbles.sdf",
        "models/deformable_teddy.sdf",
        "models/deformable_torus.sdf",
        "models/simple_gripper.sdf",
        "models/teddy.vtk",
        "models/torus.vtk"
      ]
      externalDependencies := #["@drake_models//:wsg_50_description"]
      concept := "Bazel targets and model data declarations for deformable demos"
    },
    {
      relativePath := "README.md"
      format := .markdown
      kind := .metadata
      component := "docs"
      concept := "overview of torus, bubble gripper, disabling, and subdivision demos"
    },
    {
      relativePath := "bubble_gripper.cc"
      format := .cpp
      kind := .source
      component := "bubble_gripper"
      feedsDeformablePlant := true
      localDependencies := #[
        "parallel_gripper_controller.h",
        "models/bubbles.sdf",
        "models/deformable_teddy.sdf",
        "models/teddy.vtk"
      ]
      externalDependencies := #["package://drake_models/wsg_50_description"]
      concept := "deformable bubble gripper demo with teddy mesh, bubble SDF, camera, and contact"
    },
    {
      relativePath := "deformable_common.cc"
      format := .cpp
      kind := .source
      component := "common"
      feedsDeformablePlant := true
      localDependencies := #["deformable_common.h", "models/torus.vtk"]
      concept := "shared torus deformable body registration and rigid ground setup"
    },
    {
      relativePath := "deformable_common.h"
      format := .header
      kind := .source
      component := "common"
      feedsDeformablePlant := true
      concept := "shared deformable setup declarations"
    },
    {
      relativePath := "deformable_disabling.cc"
      format := .cpp
      kind := .source
      component := "disabling"
      feedsDeformablePlant := true
      localDependencies := #["deformable_common.h", "models/deformable_torus.sdf", "models/torus.vtk"]
      concept := "enabled/disabled deformable torus simulation boundary"
    },
    {
      relativePath := "deformable_subdivision.cc"
      format := .cpp
      kind := .source
      component := "subdivision"
      feedsDeformablePlant := true
      localDependencies := #[
        "deformable_common.h",
        "point_source_force_field.h",
        "models/deformable_torus.sdf",
        "models/torus.vtk"
      ]
      concept := "force-field subdivision demo and volumetric-force integration boundary"
    },
    {
      relativePath := "deformable_torus.cc"
      format := .cpp
      kind := .source
      component := "torus"
      feedsDeformablePlant := true
      localDependencies := #[
        "deformable_common.h",
        "parallel_gripper_controller.h",
        "point_source_force_field.h",
        "suction_cup_controller.h",
        "models/deformable_torus.sdf",
        "models/simple_gripper.sdf",
        "models/torus.vtk"
      ]
      concept := "deformable torus plant with selectable rigid gripper controller"
    },
    {
      relativePath := "models/bubbles.sdf"
      format := .sdf
      kind := .model
      component := "models"
      feedsDeformablePlant := true
      externalDependencies := #[
        "package://drake_models/wsg_50_description/meshes/bubble.vtk",
        "package://drake_models/wsg_50_description/meshes/textured_bubble.obj"
      ]
      concept := "deformable left/right bubble membranes loaded by bubble_gripper"
    },
    {
      relativePath := "models/deformable_teddy.sdf"
      format := .sdf
      kind := .model
      component := "models"
      feedsDeformablePlant := true
      localDependencies := #["models/teddy.vtk"]
      concept := "deformable teddy bear SDF with local VTK mesh dependency"
    },
    {
      relativePath := "models/deformable_torus.sdf"
      format := .sdf
      kind := .model
      component := "models"
      feedsDeformablePlant := true
      localDependencies := #["models/torus.vtk"]
      concept := "deformable torus SDF with local VTK mesh dependency"
    },
    {
      relativePath := "models/simple_gripper.sdf"
      format := .sdf
      kind := .model
      component := "models"
      feedsDeformablePlant := true
      concept := "rigid two-finger gripper with prismatic joints, mimic, and hydroelastic pads"
    },
    {
      relativePath := "models/teddy.vtk"
      format := .vtk
      kind := .mesh
      component := "models"
      feedsDeformablePlant := true
      concept := "local tetrahedral mesh for deformable_teddy.sdf"
    },
    {
      relativePath := "models/torus.vtk"
      format := .vtk
      kind := .mesh
      component := "models"
      feedsDeformablePlant := true
      concept := "local tetrahedral mesh for deformable_torus.sdf and shared torus setup"
    },
    {
      relativePath := "parallel_gripper_controller.cc"
      format := .cpp
      kind := .source
      component := "controllers"
      localDependencies := #["parallel_gripper_controller.h"]
      concept := "time-programmed parallel jaw gripper desired-state implementation"
    },
    {
      relativePath := "parallel_gripper_controller.h"
      format := .header
      kind := .source
      component := "controllers"
      concept := "parallel jaw gripper controller declaration"
    },
    {
      relativePath := "point_source_force_field.cc"
      format := .cpp
      kind := .source
      component := "force_field"
      localDependencies := #["point_source_force_field.h"]
      concept := "point-source force-density field implementation"
    },
    {
      relativePath := "point_source_force_field.h"
      format := .header
      kind := .source
      component := "force_field"
      concept := "point-source force-density field declaration"
    },
    {
      relativePath := "suction_cup_controller.cc"
      format := .cpp
      kind := .source
      component := "controllers"
      localDependencies := #["suction_cup_controller.h"]
      concept := "suction cup trajectory and force-density controller implementation"
    },
    {
      relativePath := "suction_cup_controller.h"
      format := .header
      kind := .source
      component := "controllers"
      concept := "suction cup controller declaration"
    },
    {
      relativePath := "test/point_source_force_field_test.cc"
      format := .cpp
      kind := .test
      component := "force_field"
      localDependencies := #["point_source_force_field.cc", "point_source_force_field.h"]
      concept := "regression coverage for force direction, falloff, and unconnected input"
    }
  ]

private def hasDuplicateDeformableAssetPath : Bool := Id.run do
  let mut seen : Array String := #[]
  for asset in deformableExampleAssets do
    if seen.contains asset.relativePath then
      return true
    seen := seen.push asset.relativePath
  return false

def deformableExampleAssetPaths : Array String :=
  deformableExampleAssets.map (fun asset => asset.relativePath)

def deformableModelAssets : Array DeformableExampleAsset :=
  deformableExampleAssets.filter (fun asset => asset.kind == .model || asset.kind == .mesh)

def deformablePlantAssets : Array DeformableExampleAsset :=
  deformableExampleAssets.filter (fun asset => asset.feedsDeformablePlant)

def findDeformableExampleAsset? (relativePath : String) :
    Option DeformableExampleAsset :=
  deformableExampleAssets.find? (fun asset => asset.relativePath == relativePath)

def requiredDeformableExampleAssetPaths : Array String :=
  #[
    "BUILD.bazel",
    "README.md",
    "bubble_gripper.cc",
    "deformable_common.cc",
    "deformable_common.h",
    "deformable_disabling.cc",
    "deformable_subdivision.cc",
    "deformable_torus.cc",
    "models/bubbles.sdf",
    "models/deformable_teddy.sdf",
    "models/deformable_torus.sdf",
    "models/simple_gripper.sdf",
    "models/teddy.vtk",
    "models/torus.vtk",
    "parallel_gripper_controller.cc",
    "parallel_gripper_controller.h",
    "point_source_force_field.cc",
    "point_source_force_field.h",
    "suction_cup_controller.cc",
    "suction_cup_controller.h",
    "test/point_source_force_field_test.cc"
  ]

def validateDeformableExampleAssetCatalog? : Except String Unit := do
  if deformableExampleAssets.size != 21 then
    .error s!"deformable asset catalog should contain 21 files, got {deformableExampleAssets.size}"
  if hasDuplicateDeformableAssetPath then
    .error "deformable asset catalog contains duplicate paths"
  for asset in deformableExampleAssets do
    asset.validate?
  for path in requiredDeformableExampleAssetPaths do
    if !(deformableExampleAssetPaths.contains path) then
      .error s!"deformable asset catalog is missing {path}"
  for asset in deformableExampleAssets do
    for dep in asset.localDependencies do
      if !(deformableExampleAssetPaths.contains dep) then
        .error s!"deformable asset {asset.relativePath}: missing local dependency {dep}"
  if deformableModelAssets.size != 6 then
    .error s!"deformable model asset count should be 6, got {deformableModelAssets.size}"
  if deformablePlantAssets.size != 12 then
    .error s!"deformable plant-feeding asset count should be 12, got {deformablePlantAssets.size}"

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
  Float.sqrt (dot v v)

def asArray (v : Vec3) : Array Float :=
  #[v.x, v.y, v.z]

def isFinite (v : Vec3) : Bool :=
  Float.isFinite v.x && Float.isFinite v.y && Float.isFinite v.z

end Vec3

structure Rpy where
  roll : Float := 0.0
  pitch : Float := 0.0
  yaw : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace Rpy

def rotate (r : Rpy) (v : Vec3) : Vec3 :=
  let sr := Float.sin r.roll
  let cr := Float.cos r.roll
  let sp := Float.sin r.pitch
  let cp := Float.cos r.pitch
  let sy := Float.sin r.yaw
  let cy := Float.cos r.yaw
  {
    x := cy * cp * v.x + (cy * sp * sr - sy * cr) * v.y +
      (cy * sp * cr + sy * sr) * v.z
    y := sy * cp * v.x + (sy * sp * sr + cy * cr) * v.y +
      (sy * sp * cr - cy * sr) * v.z
    z := -sp * v.x + cp * sr * v.y + cp * cr * v.z
  }

end Rpy

structure RigidPose where
  rpy : Rpy := {}
  translation : Vec3 := {}
  deriving Repr, BEq, Inhabited

namespace RigidPose

def transformPoint (pose : RigidPose) (p_BQ : Vec3) : Vec3 :=
  Vec3.add pose.translation (pose.rpy.rotate p_BQ)

end RigidPose

structure PointSourceForceField where
  bodyName : String := "box"
  p_BC : Vec3 := { z := 0.123 }
  falloffDistance : Float := 0.2
  inputPortName : String := "maximum force density magnitude in N/m^3"
  cacheEntryName : String := "point source of the force field"
  deriving Repr, Inhabited

namespace PointSourceForceField

def validate? (field : PointSourceForceField) : Except String Unit := do
  if field.bodyName.isEmpty then
    .error "point source force field requires a body name"
  if !field.p_BC.isFinite then
    .error "point source offset p_BC must be finite"
  if !(Float.isFinite field.falloffDistance) || field.falloffDistance <= 0.0 then
    .error s!"falloff distance must be positive and finite, got {field.falloffDistance}"

def sourceWorldPoint (field : PointSourceForceField) (pose : RigidPose) : Vec3 :=
  pose.transformPoint field.p_BC

def evaluateAt?
    (field : PointSourceForceField)
    (pose : RigidPose)
    (maxForceDensity? : Option Float)
    (p_WQ : Vec3) : Except String Vec3 := do
  field.validate?
  if !p_WQ.isFinite then
    .error "query point p_WQ must be finite"
  let maxForceDensity := maxForceDensity?.getD 0.0
  if !(Float.isFinite maxForceDensity) then
    .error s!"maximum force density must be finite, got {maxForceDensity}"
  let p_WC := field.sourceWorldPoint pose
  let p_QC_W := Vec3.sub p_WC p_WQ
  let dist := p_QC_W.norm
  if dist == 0.0 || dist > field.falloffDistance || maxForceDensity == 0.0 then
    pure {}
  else
    let magnitude := (field.falloffDistance - dist) * maxForceDensity / field.falloffDistance
    pure (Vec3.scale (magnitude / dist) p_QC_W)

end PointSourceForceField

def pointSourceForceField : PointSourceForceField := {}

inductive DeformableMaterialModel where
  | neohookean
  | linearCorotated
  deriving Repr, BEq, Inhabited

structure DeformableBodyConfig where
  youngsModulus : Float := 3.0e4
  poissonsRatio : Float := 0.4
  massDensity : Float := 1.0e3
  stiffnessDampingCoefficient : Float := 0.01
  contactDamping : Float := 10.0
  elementSubdivisionCount : Nat := 0
  materialModel : DeformableMaterialModel := .neohookean
  label : String := "deformable_torus"
  deriving Repr, Inhabited

namespace DeformableBodyConfig

def validate? (cfg : DeformableBodyConfig) : Except String Unit := do
  if !(Float.isFinite cfg.youngsModulus) || cfg.youngsModulus <= 0.0 then
    .error s!"deformable config {cfg.label}: Young's modulus must be positive and finite, got {cfg.youngsModulus}"
  if !(Float.isFinite cfg.poissonsRatio) || cfg.poissonsRatio <= -1.0 ||
      cfg.poissonsRatio >= 0.5 then
    .error s!"deformable config {cfg.label}: Poisson ratio must lie in (-1, 0.5), got {cfg.poissonsRatio}"
  if !(Float.isFinite cfg.massDensity) || cfg.massDensity <= 0.0 then
    .error s!"deformable config {cfg.label}: mass density must be positive and finite, got {cfg.massDensity}"
  if !(Float.isFinite cfg.stiffnessDampingCoefficient) ||
      cfg.stiffnessDampingCoefficient < 0.0 then
    .error s!"deformable config {cfg.label}: stiffness damping must be nonnegative and finite, got {cfg.stiffnessDampingCoefficient}"
  if !(Float.isFinite cfg.contactDamping) || cfg.contactDamping < 0.0 then
    .error s!"deformable config {cfg.label}: contact damping must be nonnegative and finite, got {cfg.contactDamping}"

def lameMu (cfg : DeformableBodyConfig) : Float :=
  cfg.youngsModulus / (2.0 * (1.0 + cfg.poissonsRatio))

def lameLambda (cfg : DeformableBodyConfig) : Float :=
  cfg.youngsModulus * cfg.poissonsRatio /
    ((1.0 + cfg.poissonsRatio) * (1.0 - 2.0 * cfg.poissonsRatio))

end DeformableBodyConfig

def torusDeformableConfig : DeformableBodyConfig :=
  {
    youngsModulus := 3.0e4
    poissonsRatio := 0.4
    massDensity := 1.0e3
    stiffnessDampingCoefficient := 0.01
    contactDamping := 10.0
    materialModel := .neohookean
    label := "deformable_torus"
  }

def bubbleDeformableConfig : DeformableBodyConfig :=
  {
    youngsModulus := 1.0e4
    poissonsRatio := 0.45
    massDensity := 10.0
    stiffnessDampingCoefficient := 0.05
    contactDamping := 5.0
    materialModel := .neohookean
    label := "bubble"
  }

def teddyDeformableConfig : DeformableBodyConfig :=
  {
    youngsModulus := 5.0e4
    poissonsRatio := 0.45
    massDensity := 1.0e3
    stiffnessDampingCoefficient := 0.05
    contactDamping := 0.0
    materialModel := .neohookean
    label := "deformable_teddy"
  }

private def validateOptionalNonnegative? (label : String) :
    Option Float → Except String Unit
  | none => pure ()
  | some x =>
      if !(Float.isFinite x) || x < 0.0 then
        .error s!"{label} must be nonnegative and finite, got {x}"
      else
        pure ()

private def validatePositiveFinite? (label : String) (x : Float) :
    Except String Unit := do
  if !(Float.isFinite x) || x <= 0.0 then
    .error s!"{label} must be positive and finite, got {x}"

private def validateFinite? (label : String) (x : Float) :
    Except String Unit := do
  if !(Float.isFinite x) then
    .error s!"{label} must be finite, got {x}"

inductive GeometryRole where
  | collision
  | visual
  deriving Repr, BEq, Inhabited

inductive HydroelasticRepresentation where
  | none
  | compliant
  deriving Repr, BEq, Inhabited

structure ProximityProperties where
  staticFriction? : Option Float := none
  dynamicFriction? : Option Float := none
  huntCrossleyDissipation? : Option Float := none
  hydroelasticModulus? : Option Float := none
  hydroelastic : HydroelasticRepresentation := .none
  deriving Repr, Inhabited

namespace ProximityProperties

def validate? (props : ProximityProperties) (label : String) : Except String Unit := do
  validateOptionalNonnegative? s!"{label}: static friction" props.staticFriction?
  validateOptionalNonnegative? s!"{label}: dynamic friction" props.dynamicFriction?
  validateOptionalNonnegative? s!"{label}: Hunt-Crossley dissipation" props.huntCrossleyDissipation?
  validateOptionalNonnegative? s!"{label}: hydroelastic modulus" props.hydroelasticModulus?
  match props.hydroelastic, props.hydroelasticModulus? with
  | .compliant, none =>
      .error s!"{label}: compliant hydroelastic contact requires a modulus"
  | _, _ => pure ()

end ProximityProperties

structure Rgba where
  r : Float
  g : Float
  b : Float
  a : Float
  deriving Repr, BEq, Inhabited

namespace Rgba

def validate? (rgba : Rgba) (label : String) : Except String Unit := do
  validateFinite? s!"{label}: red channel" rgba.r
  validateFinite? s!"{label}: green channel" rgba.g
  validateFinite? s!"{label}: blue channel" rgba.b
  validateFinite? s!"{label}: alpha channel" rgba.a

end Rgba

structure MeshGeometrySpec where
  uri : String
  role : GeometryRole
  scale : Vec3 := { x := 1.0, y := 1.0, z := 1.0 }
  deriving Repr, Inhabited

namespace MeshGeometrySpec

def validate? (mesh : MeshGeometrySpec) (label : String) : Except String Unit := do
  if mesh.uri.isEmpty then
    .error s!"{label}: mesh URI cannot be empty"
  if !mesh.scale.isFinite then
    .error s!"{label}: mesh scale must be finite"
  if mesh.scale.x <= 0.0 || mesh.scale.y <= 0.0 || mesh.scale.z <= 0.0 then
    .error s!"{label}: mesh scale must be positive, got {reprStr mesh.scale}"

end MeshGeometrySpec

structure BoxGeometrySpec where
  size : Vec3
  role : GeometryRole
  deriving Repr, Inhabited

namespace BoxGeometrySpec

def validate? (box : BoxGeometrySpec) (label : String) : Except String Unit := do
  if !box.size.isFinite then
    .error s!"{label}: box size must be finite"
  if box.size.x <= 0.0 || box.size.y <= 0.0 || box.size.z <= 0.0 then
    .error s!"{label}: box size must be positive, got {reprStr box.size}"

end BoxGeometrySpec

structure DeformableLinkSpec where
  name : String
  pose : RigidPose
  collisionMesh : MeshGeometrySpec
  visualMesh? : Option MeshGeometrySpec := none
  visualEmpty : Bool := false
  visualDiffuse? : Option Rgba := none
  proximity : ProximityProperties := {}
  config : DeformableBodyConfig
  deriving Repr, Inhabited

namespace DeformableLinkSpec

def validate? (link : DeformableLinkSpec) : Except String Unit := do
  if link.name.isEmpty then
    .error "deformable link name cannot be empty"
  if !link.pose.translation.isFinite then
    .error s!"deformable link {link.name}: pose translation must be finite"
  link.collisionMesh.validate? s!"deformable link {link.name} collision"
  if link.collisionMesh.role != .collision then
    .error s!"deformable link {link.name}: collision mesh must have collision role"
  match link.visualMesh? with
  | none => pure ()
  | some visualMesh =>
      visualMesh.validate? s!"deformable link {link.name} visual"
      if visualMesh.role != .visual then
        .error s!"deformable link {link.name}: visual mesh must have visual role"
  match link.visualDiffuse? with
  | none => pure ()
  | some rgba => rgba.validate? s!"deformable link {link.name} visual diffuse"
  link.proximity.validate? s!"deformable link {link.name}"
  link.config.validate?

end DeformableLinkSpec

structure DeformableSdfModelSpec where
  assetPath : String
  modelName : String
  links : Array DeformableLinkSpec
  deriving Repr, Inhabited

namespace DeformableSdfModelSpec

def validate? (model : DeformableSdfModelSpec) : Except String Unit := do
  if model.assetPath.isEmpty then
    .error "deformable SDF model asset path cannot be empty"
  if !(deformableExampleAssetPaths.contains model.assetPath) then
    .error s!"deformable SDF model {model.modelName}: asset {model.assetPath} is not in the catalog"
  if model.modelName.isEmpty then
    .error s!"deformable SDF model {model.assetPath}: model name cannot be empty"
  if model.links.isEmpty then
    .error s!"deformable SDF model {model.modelName}: expected at least one link"
  for link in model.links do
    link.validate?

end DeformableSdfModelSpec

structure RotationalInertiaSpec where
  ixx : Float
  ixy : Float
  ixz : Float
  iyy : Float
  iyz : Float
  izz : Float
  deriving Repr, Inhabited

namespace RotationalInertiaSpec

def validate? (inertia : RotationalInertiaSpec) (label : String) :
    Except String Unit := do
  validateOptionalNonnegative? s!"{label}: ixx" (some inertia.ixx)
  validateFinite? s!"{label}: ixy" inertia.ixy
  validateFinite? s!"{label}: ixz" inertia.ixz
  validateOptionalNonnegative? s!"{label}: iyy" (some inertia.iyy)
  validateFinite? s!"{label}: iyz" inertia.iyz
  validateOptionalNonnegative? s!"{label}: izz" (some inertia.izz)

end RotationalInertiaSpec

structure RigidInertiaSpec where
  mass : Float
  rotational : RotationalInertiaSpec
  deriving Repr, Inhabited

namespace RigidInertiaSpec

def validate? (inertia : RigidInertiaSpec) (label : String) :
    Except String Unit := do
  validatePositiveFinite? s!"{label}: mass" inertia.mass
  inertia.rotational.validate? label

end RigidInertiaSpec

structure RigidGripperLinkSpec where
  name : String
  pose : RigidPose
  inertia : RigidInertiaSpec
  visualBox? : Option BoxGeometrySpec := none
  collisionBox? : Option BoxGeometrySpec := none
  visualDiffuse? : Option Rgba := none
  proximity : ProximityProperties := {}
  deriving Repr, Inhabited

namespace RigidGripperLinkSpec

def validate? (link : RigidGripperLinkSpec) : Except String Unit := do
  if link.name.isEmpty then
    .error "rigid gripper link name cannot be empty"
  if !link.pose.translation.isFinite then
    .error s!"rigid gripper link {link.name}: pose translation must be finite"
  link.inertia.validate? s!"rigid gripper link {link.name}"
  match link.visualBox? with
  | none => pure ()
  | some box =>
      box.validate? s!"rigid gripper link {link.name} visual"
      if box.role != .visual then
        .error s!"rigid gripper link {link.name}: visual box must have visual role"
  match link.collisionBox? with
  | none => pure ()
  | some box =>
      box.validate? s!"rigid gripper link {link.name} collision"
      if box.role != .collision then
        .error s!"rigid gripper link {link.name}: collision box must have collision role"
  match link.visualDiffuse? with
  | none => pure ()
  | some rgba => rgba.validate? s!"rigid gripper link {link.name} visual diffuse"
  link.proximity.validate? s!"rigid gripper link {link.name}"

end RigidGripperLinkSpec

inductive SdfJointType where
  | prismatic
  deriving Repr, BEq, Inhabited

structure JointControllerGains where
  p : Float
  d : Float
  deriving Repr, Inhabited

namespace JointControllerGains

def validate? (gains : JointControllerGains) (label : String) :
    Except String Unit := do
  validateOptionalNonnegative? s!"{label}: proportional gain" (some gains.p)
  validateOptionalNonnegative? s!"{label}: derivative gain" (some gains.d)

end JointControllerGains

structure MimicJointSpec where
  jointName : String
  multiplier : Float
  offset : Float
  deriving Repr, Inhabited

namespace MimicJointSpec

def validate? (mimic : MimicJointSpec) (label : String) : Except String Unit := do
  if mimic.jointName.isEmpty then
    .error s!"{label}: mimic joint name cannot be empty"
  validateFinite? s!"{label}: mimic multiplier" mimic.multiplier
  validateFinite? s!"{label}: mimic offset" mimic.offset

end MimicJointSpec

structure RigidGripperJointSpec where
  name : String
  jointType : SdfJointType
  parent : String
  child : String
  axis : Vec3
  axisExpressedIn : String := ""
  controllerGains? : Option JointControllerGains := none
  mimic? : Option MimicJointSpec := none
  effortLimit? : Option Float := none
  deriving Repr, Inhabited

namespace RigidGripperJointSpec

def validate? (joint : RigidGripperJointSpec) : Except String Unit := do
  if joint.name.isEmpty then
    .error "rigid gripper joint name cannot be empty"
  if joint.parent.isEmpty || joint.child.isEmpty then
    .error s!"rigid gripper joint {joint.name}: parent and child cannot be empty"
  if !joint.axis.isFinite || joint.axis.norm == 0.0 then
    .error s!"rigid gripper joint {joint.name}: axis must be finite and nonzero"
  match joint.controllerGains? with
  | none => pure ()
  | some gains => gains.validate? s!"rigid gripper joint {joint.name}"
  match joint.mimic? with
  | none => pure ()
  | some mimic => mimic.validate? s!"rigid gripper joint {joint.name}"
  validateOptionalNonnegative? s!"rigid gripper joint {joint.name}: effort limit" joint.effortLimit?

end RigidGripperJointSpec

structure SimpleGripperSdfSpec where
  assetPath : String
  modelName : String
  pose : RigidPose
  links : Array RigidGripperLinkSpec
  joints : Array RigidGripperJointSpec
  deriving Repr, Inhabited

namespace SimpleGripperSdfSpec

private def hasLinkNamed (model : SimpleGripperSdfSpec) (name : String) : Bool :=
  name == "world" || model.links.any (fun link => link.name == name)

private def hasJointNamed (model : SimpleGripperSdfSpec) (name : String) : Bool :=
  model.joints.any (fun joint => joint.name == name)

def validate? (model : SimpleGripperSdfSpec) : Except String Unit := do
  if model.assetPath.isEmpty then
    .error "simple gripper SDF asset path cannot be empty"
  if !(deformableExampleAssetPaths.contains model.assetPath) then
    .error s!"simple gripper SDF: asset {model.assetPath} is not in the catalog"
  if model.modelName.isEmpty then
    .error "simple gripper SDF model name cannot be empty"
  if !model.pose.translation.isFinite then
    .error "simple gripper model pose translation must be finite"
  if model.links.isEmpty then
    .error "simple gripper SDF requires rigid links"
  if model.joints.isEmpty then
    .error "simple gripper SDF requires joints"
  for link in model.links do
    link.validate?
  for joint in model.joints do
    joint.validate?
    if !(model.hasLinkNamed joint.parent) then
      .error s!"simple gripper joint {joint.name}: unknown parent link {joint.parent}"
    if !(model.hasLinkNamed joint.child) then
      .error s!"simple gripper joint {joint.name}: unknown child link {joint.child}"
    match joint.mimic? with
    | none => pure ()
    | some mimic =>
        if !(model.hasJointNamed mimic.jointName) then
          .error s!"simple gripper joint {joint.name}: unknown mimic target {mimic.jointName}"

end SimpleGripperSdfSpec

def torusSdfModelSpec : DeformableSdfModelSpec :=
  {
    assetPath := "models/deformable_torus.sdf"
    modelName := "deformable"
    links := #[
      {
        name := "torus"
        pose := { translation := { z := 0.02925 } }
        collisionMesh := {
          uri := "package://drake/examples/multibody/deformable/models/torus.vtk"
          role := .collision
          scale := { x := 0.65, y := 0.65, z := 0.65 }
        }
        proximity := {
          dynamicFriction? := some 1.15
          huntCrossleyDissipation? := some 10.0
        }
        config := torusDeformableConfig
      }
    ]
  }

def bubbleSdfModelSpec : DeformableSdfModelSpec :=
  let collisionMesh : MeshGeometrySpec := {
    uri := "package://drake_models/wsg_50_description/meshes/bubble.vtk"
    role := .collision
  }
  let visualMesh : MeshGeometrySpec := {
    uri := "package://drake_models/wsg_50_description/meshes/textured_bubble.obj"
    role := .visual
  }
  let proximity : ProximityProperties := {
    dynamicFriction? := some 1.0
    huntCrossleyDissipation? := some 5.0
  }
  {
    assetPath := "models/bubbles.sdf"
    modelName := "bubble"
    links := #[
      {
        name := "left"
        pose := {
          rpy := { roll := 1.5707, pitch := 3.1416 }
          translation := { x := -0.185, y := -0.09, z := 0.06 }
        }
        collisionMesh := collisionMesh
        visualMesh? := some visualMesh
        proximity := proximity
        config := bubbleDeformableConfig
      },
      {
        name := "right"
        pose := {
          rpy := { roll := -1.5707, pitch := 3.1416 }
          translation := { x := -0.185, y := 0.09, z := 0.06 }
        }
        collisionMesh := collisionMesh
        visualMesh? := some visualMesh
        proximity := proximity
        config := bubbleDeformableConfig
      }
    ]
  }

def teddySdfModelSpec : DeformableSdfModelSpec :=
  {
    assetPath := "models/deformable_teddy.sdf"
    modelName := "deformable"
    links := #[
      {
        name := "teddy"
        pose := {
          rpy := { roll := 1.5707, yaw := -1.5707 }
          translation := { x := -0.17 }
        }
        collisionMesh := {
          uri := "package://drake/examples/multibody/deformable/models/teddy.vtk"
          role := .collision
          scale := { x := 0.15, y := 0.15, z := 0.15 }
        }
        visualEmpty := true
        visualDiffuse? := some { r := 0.82, g := 0.71, b := 0.55, a := 1.0 }
        proximity := { dynamicFriction? := some 0.9 }
        config := teddyDeformableConfig
      }
    ]
  }

def deformableSdfModelSpecs : Array DeformableSdfModelSpec :=
  #[torusSdfModelSpec, bubbleSdfModelSpec, teddySdfModelSpec]

def gripperPadProximity : ProximityProperties :=
  {
    staticFriction? := some 1.5
    dynamicFriction? := some 1.5
    huntCrossleyDissipation? := some 5.0
    hydroelasticModulus? := some 1.0e6
    hydroelastic := .compliant
  }

def simpleGripperSdfSpec : SimpleGripperSdfSpec :=
  let darkGray : Rgba := { r := 0.3, g := 0.3, b := 0.3, a := 0.9 }
  let fingerBox : BoxGeometrySpec := {
    size := { x := 0.007, y := 0.081, z := 0.028 }
    role := .collision
  }
  {
    assetPath := "models/simple_gripper.sdf"
    modelName := "simple_gripper"
    pose := {
      rpy := { roll := -1.57, yaw := 1.57 }
      translation := { y := 0.06, z := 0.08 }
    }
    links := #[
      {
        name := "body"
        pose := { translation := { y := -0.049133 } }
        inertia := {
          mass := 0.988882
          rotational := {
            ixx := 0.162992, ixy := 0.0, ixz := 0.0,
            iyy := 0.162992, iyz := 0.0, izz := 0.164814
          }
        }
        visualBox? := some {
          size := { x := 0.146, y := 0.0725, z := 0.049521 }
          role := .visual
        }
        visualDiffuse? := some darkGray
      },
      {
        name := "left_finger"
        pose := { translation := { x := -0.0105, y := 0.029 } }
        inertia := {
          mass := 0.05
          rotational := {
            ixx := 0.16, ixy := 0.0, ixz := 0.0,
            iyy := 0.16, iyz := 0.0, izz := 0.16
          }
        }
        visualBox? := some { fingerBox with role := .visual }
        collisionBox? := some fingerBox
        visualDiffuse? := some darkGray
        proximity := gripperPadProximity
      },
      {
        name := "right_finger"
        pose := { translation := { x := 0.0105, y := 0.029 } }
        inertia := {
          mass := 0.05
          rotational := {
            ixx := 0.16, ixy := 0.0, ixz := 0.0,
            iyy := 0.16, iyz := 0.0, izz := 0.16
          }
        }
        visualBox? := some { fingerBox with role := .visual }
        collisionBox? := some fingerBox
        visualDiffuse? := some darkGray
        proximity := gripperPadProximity
      }
    ]
    joints := #[
      {
        name := "translate_joint"
        jointType := .prismatic
        parent := "world"
        child := "body"
        axis := { y := -1.0 }
        axisExpressedIn := "__model__"
        controllerGains? := some { p := 10000.0, d := 1.0 }
      },
      {
        name := "left_slider"
        jointType := .prismatic
        parent := "body"
        child := "left_finger"
        axis := { x := 1.0 }
        controllerGains? := some { p := 10000.0, d := 1.0 }
      },
      {
        name := "right_slider"
        jointType := .prismatic
        parent := "body"
        child := "right_finger"
        axis := { x := 1.0 }
        mimic? := some { jointName := "left_slider", multiplier := -1.0, offset := 0.0 }
        effortLimit? := some 0.0
      }
    ]
  }

def validateSdfPhysicsProvider? : Except String Unit := do
  for model in deformableSdfModelSpecs do
    model.validate?
  simpleGripperSdfSpec.validate?

structure DeformableSampleNode where
  id : Nat
  p_WQ : Vec3
  v_WQ : Vec3 := {}
  volumeWeight : Float
  fixed : Bool := false
  label : String := ""
  deriving Repr, Inhabited

namespace DeformableSampleNode

def validate? (node : DeformableSampleNode) : Except String Unit := do
  if !node.p_WQ.isFinite then
    .error s!"deformable sample {node.id}: position must be finite"
  if !node.v_WQ.isFinite then
    .error s!"deformable sample {node.id}: velocity must be finite"
  if !(Float.isFinite node.volumeWeight) || node.volumeWeight <= 0.0 then
    .error s!"deformable sample {node.id}: volume weight must be positive and finite, got {node.volumeWeight}"

def lumpedMass (cfg : DeformableBodyConfig) (node : DeformableSampleNode) : Float :=
  cfg.massDensity * node.volumeWeight

end DeformableSampleNode

structure DeformableSampleForce where
  nodeId : Nat
  forceDensity : Vec3
  force : Vec3
  acceleration : Vec3
  mass : Float
  fixed : Bool
  deriving Repr, Inhabited

structure DeformableFemForceResult where
  config : DeformableBodyConfig
  field : PointSourceForceField
  samples : Array DeformableSampleNode
  nodeForces : Array DeformableSampleForce
  totalForce : Vec3
  totalMass : Float
  freeMass : Float
  gravity : Vec3
  deriving Repr, Inhabited

namespace DeformableFemForceResult

def maxAccelerationNorm (result : DeformableFemForceResult) : Float :=
  result.nodeForces.foldl
    (fun acc node => max acc node.acceleration.norm)
    0.0

end DeformableFemForceResult

def gravityVector : Vec3 :=
  { z := -9.81 }

def integratePointSourceFemForce?
    (cfg : DeformableBodyConfig)
    (field : PointSourceForceField)
    (pose : RigidPose)
    (maxForceDensity : Float)
    (samples : Array DeformableSampleNode)
    (gravity : Vec3 := gravityVector) :
    Except String DeformableFemForceResult := do
  cfg.validate?
  field.validate?
  if !(Float.isFinite maxForceDensity) then
    .error s!"deformable FEM force integration: maximum force density must be finite, got {maxForceDensity}"
  if !gravity.isFinite then
    .error "deformable FEM force integration: gravity must be finite"
  let mut nodeForces : Array DeformableSampleForce := #[]
  let mut totalForce : Vec3 := {}
  let mut totalMass := 0.0
  let mut freeMass := 0.0
  for node in samples do
    node.validate?
    let density ← field.evaluateAt? pose (some maxForceDensity) node.p_WQ
    let force := Vec3.scale node.volumeWeight density
    let mass := node.lumpedMass cfg
    let acceleration :=
      if node.fixed then
        {}
      else
        let forceAccel := Vec3.scale (1.0 / mass) force
        let dampingAccel :=
          Vec3.scale (-(cfg.stiffnessDampingCoefficient)) node.v_WQ
        Vec3.add gravity (Vec3.add forceAccel dampingAccel)
    nodeForces := nodeForces.push {
      nodeId := node.id
      forceDensity := density
      force := force
      acceleration := acceleration
      mass := mass
      fixed := node.fixed
    }
    totalForce := Vec3.add totalForce force
    totalMass := totalMass + mass
    if !node.fixed then
      freeMass := freeMass + mass
  pure {
    config := cfg
    field := field
    samples := samples
    nodeForces := nodeForces
    totalForce := totalForce
    totalMass := totalMass
    freeMass := freeMass
    gravity := gravity
  }

def subdivisionDemoForceField : PointSourceForceField :=
  { pointSourceForceField with
    bodyName := "world"
    p_BC := { x := 0.07, y := 0.07, z := 0.09 }
    falloffDistance := 0.020 }

def subdivisionDemoFemSamples : Array DeformableSampleNode :=
  #[
    {
      id := 0
      p_WQ := { x := 0.07, y := 0.07, z := 0.10 }
      volumeWeight := 1.0e-6
      label := "subdivided torus sample inside point-source support"
    },
    {
      id := 1
      p_WQ := { x := -0.20, y := 0.0, z := 0.04 }
      volumeWeight := 1.0e-6
      fixed := true
      label := "fixed deformable constraint sample"
    }
  ]

def subdivisionDemoFemForce? : Except String DeformableFemForceResult :=
  integratePointSourceFemForce?
    torusDeformableConfig
    subdivisionDemoForceField
    {}
    6.0e6
    subdivisionDemoFemSamples

structure ParallelGripperController where
  openWidth : Float := 0.12
  closedWidth : Float := 0.04
  height : Float := 0.25
  fingersClosedTime : Float := 1.5
  gripperLiftedTime : Float := 3.0
  holdTime : Float := 5.5
  fingersOpenTime : Float := 7.0
  deriving Repr, Inhabited

namespace ParallelGripperController

def validate? (cfg : ParallelGripperController) : Except String Unit := do
  if !(Float.isFinite cfg.openWidth) || cfg.openWidth < 0.0 then
    .error s!"open width must be nonnegative and finite, got {cfg.openWidth}"
  if !(Float.isFinite cfg.closedWidth) || cfg.closedWidth < 0.0 then
    .error s!"closed width must be nonnegative and finite, got {cfg.closedWidth}"
  if !(Float.isFinite cfg.height) then
    .error s!"lift height must be finite, got {cfg.height}"
  if !(cfg.fingersClosedTime > 0.0 &&
      cfg.gripperLiftedTime > cfg.fingersClosedTime &&
      cfg.holdTime > cfg.gripperLiftedTime &&
      cfg.fingersOpenTime > cfg.holdTime) then
    .error "parallel gripper event times must be strictly increasing"

def initialConfiguration (cfg : ParallelGripperController) : Array Float :=
  #[0.0, -cfg.openWidth / 2.0]

def closedConfiguration (cfg : ParallelGripperController) : Array Float :=
  #[0.0, -cfg.closedWidth / 2.0]

def liftedConfiguration (cfg : ParallelGripperController) : Array Float :=
  #[cfg.height, -cfg.closedWidth / 2.0]

def openConfiguration (cfg : ParallelGripperController) : Array Float :=
  #[cfg.height, -cfg.openWidth / 2.0]

private def lerpArray (theta : Float) (a b : Array Float) : Array Float :=
  FloatArray.add (FloatArray.scale (1.0 - theta) a) (FloatArray.scale theta b)

def desiredState? (cfg : ParallelGripperController) (t : Float) :
    Except String (Array Float) := do
  cfg.validate?
  if !(Float.isFinite t) then
    .error s!"parallel gripper time must be finite, got {t}"
  let positions :=
    if t < cfg.fingersClosedTime then
      lerpArray (t / cfg.fingersClosedTime)
        cfg.initialConfiguration cfg.closedConfiguration
    else if t < cfg.gripperLiftedTime then
      lerpArray ((t - cfg.fingersClosedTime) / (cfg.gripperLiftedTime - cfg.fingersClosedTime))
        cfg.closedConfiguration cfg.liftedConfiguration
    else if t < cfg.holdTime then
      cfg.liftedConfiguration
    else if t < cfg.fingersOpenTime then
      lerpArray ((t - cfg.holdTime) / (cfg.fingersOpenTime - cfg.holdTime))
        cfg.liftedConfiguration cfg.openConfiguration
    else
      cfg.openConfiguration
  pure (positions ++ #[0.0, 0.0])

end ParallelGripperController

def parallelGripperController : ParallelGripperController := {}

structure SuctionCupController where
  initialHeight : Float := 0.35
  objectHeight : Float := 0.08
  approachTime : Float := 0.5
  startSuctionTime : Float := 1.5
  retrieveTime : Float := 3.0
  releaseSuctionTime : Float := 5.0
  activeForceDensity : Float := 2.0e5
  deriving Repr, Inhabited

namespace SuctionCupController

def validate? (cfg : SuctionCupController) : Except String Unit := do
  if !(Float.isFinite cfg.initialHeight) || !(Float.isFinite cfg.objectHeight) then
    .error "suction cup heights must be finite"
  if !(cfg.approachTime < cfg.startSuctionTime &&
      cfg.startSuctionTime < cfg.retrieveTime &&
      cfg.retrieveTime < cfg.releaseSuctionTime) then
    .error "suction cup event times must be strictly increasing"
  if !(Float.isFinite cfg.activeForceDensity) || cfg.activeForceDensity < 0.0 then
    .error s!"active force density must be nonnegative and finite, got {cfg.activeForceDensity}"

def travelTime (cfg : SuctionCupController) : Float :=
  cfg.startSuctionTime - cfg.approachTime

def desiredState? (cfg : SuctionCupController) (t : Float) :
    Except String (Array Float) := do
  cfg.validate?
  if !(Float.isFinite t) then
    .error s!"suction cup time must be finite, got {t}"
  let travel := cfg.travelTime
  if t < cfg.approachTime then
    pure #[cfg.initialHeight, 0.0]
  else if t < cfg.startSuctionTime then
    let v := (cfg.objectHeight - cfg.initialHeight) / travel
    let dt := t - cfg.approachTime
    pure #[cfg.initialHeight + dt * v, v]
  else if t < cfg.retrieveTime then
    pure #[cfg.objectHeight, 0.0]
  else if t < cfg.retrieveTime + travel then
    let v := (cfg.initialHeight - cfg.objectHeight) / travel
    let dt := t - cfg.retrieveTime
    pure #[cfg.objectHeight + dt * v, v]
  else
    pure #[cfg.initialHeight, 0.0]

def maxForceDensity? (cfg : SuctionCupController) (t : Float) :
    Except String Float := do
  cfg.validate?
  if !(Float.isFinite t) then
    .error s!"suction cup time must be finite, got {t}"
  pure (if t >= cfg.startSuctionTime && t <= cfg.releaseSuctionTime then
    cfg.activeForceDensity
  else
    0.0)

end SuctionCupController

def suctionCupController : SuctionCupController := {}

structure SubdivisionSamplingResult where
  coarseSamples : Array Vec3
  subdividedSamples : Array Vec3
  coarseForceSum : Vec3
  subdividedForceSum : Vec3
  deriving Repr, Inhabited

private def sumVec3 (xs : Array Vec3) : Vec3 :=
  xs.foldl Vec3.add {}

def evaluateForceSamples?
    (field : PointSourceForceField)
    (pose : RigidPose)
    (maxForceDensity : Float)
    (samples : Array Vec3) : Except String Vec3 := do
  let mut forces := #[]
  for sample in samples do
    forces := forces.push (← field.evaluateAt? pose (some maxForceDensity) sample)
  pure (sumVec3 forces)

def subdivisionSampling? : Except String SubdivisionSamplingResult := do
  let field : PointSourceForceField :=
    { pointSourceForceField with p_BC := {}, falloffDistance := 0.2 }
  let pose : RigidPose := {}
  let coarseSamples := #[
    { x := -0.35, y := -0.35, z := 0.0 },
    { x := 0.35, y := -0.35, z := 0.0 },
    { x := -0.35, y := 0.35, z := 0.0 },
    { x := 0.35, y := 0.35, z := 0.0 }
  ]
  let subdividedSamples := coarseSamples ++ #[
    { x := 0.0, y := 0.1, z := 0.0 },
    { x := 0.1, y := 0.0, z := 0.0 }
  ]
  pure {
    coarseSamples := coarseSamples
    subdividedSamples := subdividedSamples
    coarseForceSum := (← evaluateForceSamples? field pose 42.0 coarseSamples)
    subdividedForceSum := (← evaluateForceSamples? field pose 42.0 subdividedSamples)
  }

def acceptedSegment : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := 0.01
    tAfter := 0.01
    label := "deformable FEM/SAP plant advance boundary"
  }

private def localMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

structure DeformableResult where
  references : Array DrakeReference
  assetCatalog : Array DeformableExampleAsset
  deformableModels : Array DeformableSdfModelSpec
  simpleGripperModel : SimpleGripperSdfSpec
  forceField : PointSourceForceField
  femForce : DeformableFemForceResult
  parallelDesiredSamples : Array (Float × Array Float)
  suctionDesiredSamples : Array (Float × Array Float × Float)
  subdivision : SubdivisionSamplingResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd? : Except String DeformableResult := do
  validateDeformableExampleAssetCatalog?
  validateSdfPhysicsProvider?
  let parallelTimes := #[0.0, 1.5, 3.0, 5.5, 7.0]
  let mut parallelSamples := #[]
  for t in parallelTimes do
    parallelSamples := parallelSamples.push (t, ← parallelGripperController.desiredState? t)
  let suctionTimes := #[0.0, 1.0, 1.5, 3.5, 5.5]
  let mut suctionSamples := #[]
  for t in suctionTimes do
    suctionSamples := suctionSamples.push
      (t, ← suctionCupController.desiredState? t, ← suctionCupController.maxForceDensity? t)
  let femForce ← subdivisionDemoFemForce?
  let trace := DynamicEventTrace.empty.push (.interval acceptedSegment)
  trace.validate?
  pure {
    references := drakeReferences
    assetCatalog := deformableExampleAssets
    deformableModels := deformableSdfModelSpecs
    simpleGripperModel := simpleGripperSdfSpec
    forceField := pointSourceForceField
    femForce := femForce
    parallelDesiredSamples := parallelSamples
    suctionDesiredSamples := suctionSamples
    subdivision := (← subdivisionSampling?)
    trace := trace
    moves :=
      #[
        localMove 5400 "point-source force-density input/cache evaluation",
        localMove 5401 "parallel gripper time-state controller",
        localMove 5402 "suction cup trajectory and force-density controller",
        localMove 5404 "deformable lumped FEM force-density mass solve",
        localMove 5403 "deformable FEM/SAP solve and contact boundary" .controlledApproximation
      ] ++ trace.moves
  }

end Tyr.EventSkeleton.Examples.Deformable
