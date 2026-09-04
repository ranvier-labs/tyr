import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Trace

/-!
# Drake SceneGraph Event-Skeleton Example

This ports the reusable geometry-provider surface exercised by
`../drake/examples/scene_graph`.

The example keeps geometry ownership separate from contact physics:

* the provider records sources, frames, geometries, roles, materials, and poses,
* a query emits point-pair penetration or `ContactCandidate` views,
* dynamics consumes only the primitive contact rows and scalar force law.

That is the intended path toward full physics: providers may compute geometry
however they like, but the executable dynamics remain an assembly of existing
mass-matrix, support-selection, scalar-force, and `J^T f` primitives.
-/

namespace Tyr.EventSkeleton.Examples.SceneGraph

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/scene_graph/BUILD.bazel"
      concept := "declares the scene_graph example binaries and data dependencies"
    },
    {
      path := "../drake/examples/scene_graph/bouncing_ball_plant.cc"
      concept := "registers sphere geometry, assigns illustration/proximity/perception roles, queries point-pair penetration, and applies Hunt-Crossley contact"
    },
    {
      path := "../drake/examples/scene_graph/bouncing_ball_plant.h"
      concept := "declares the bouncing-ball LeafSystem ports, SceneGraph source id, and geometry query input"
    },
    {
      path := "../drake/examples/scene_graph/bouncing_ball_vector.h"
      concept := "defines the two-coordinate BouncingBallVector named state with z and zdot accessors"
    },
    {
      path := "../drake/examples/scene_graph/bouncing_ball_vector.cc"
      concept := "provides BouncingBallVector coordinate names in Drake's generated-vector style"
    },
    {
      path := "../drake/examples/scene_graph/bouncing_ball_run_dynamics.cc"
      concept := "builds two bouncing-ball plants, an anchored half-space ground, camera/render ports, and SceneGraph query connections"
    },
    {
      path := "../drake/examples/scene_graph/solar_system.cc"
      concept := "registers anchored and dynamic geometry across sphere, cylinder, mesh, convex, box, and capsule shapes"
    },
    {
      path := "../drake/examples/scene_graph/solar_system.h"
      concept := "declares SolarSystem pose output and source-id interface used by SceneGraph"
    },
    {
      path := "../drake/examples/scene_graph/solar_system_run_dynamics.cc"
      concept := "builds the SolarSystem + SceneGraph diagram, wires pose output to source poses, adds DrakeVisualizer and Meshcat, and advances the simulator"
    },
    {
      path := "../drake/examples/scene_graph/simple_contact_surface_vis.cc"
      concept := "uses SceneGraph queries to produce hydroelastic contact surfaces or point-pair fallback messages"
    }
  ]

def sceneGraphExampleRoot : String :=
  "../drake/examples/scene_graph"

inductive SceneGraphExampleAssetKind where
  | metadata
  | source
  | mesh
  | material
  | texture
  | binaryBuffer
  | model
  deriving Repr, BEq, Inhabited

inductive SceneGraphExampleAssetFormat where
  | bazel
  | cpp
  | header
  | obj
  | mtl
  | png
  | gltf
  | bin
  | ktx2
  | sdf
  deriving Repr, BEq, Inhabited

namespace SceneGraphExampleAssetFormat

def matchesPath (format : SceneGraphExampleAssetFormat) (path : String) : Bool :=
  match format with
  | .bazel => path == "BUILD.bazel"
  | .cpp => path.endsWith ".cc"
  | .header => path.endsWith ".h"
  | .obj => path.endsWith ".obj"
  | .mtl => path.endsWith ".mtl"
  | .png => path.endsWith ".png"
  | .gltf => path.endsWith ".gltf"
  | .bin => path.endsWith ".bin"
  | .ktx2 => path.endsWith ".ktx2"
  | .sdf => path.endsWith ".sdf"

end SceneGraphExampleAssetFormat

/--
File manifest for Drake's `examples/scene_graph` tree.

The manifest keeps mesh and texture dependency closure explicit.  SceneGraph
providers still expose primitive geometry records; this catalog is only the
asset provider boundary needed to make those records faithful to Drake's data
layout.
-/
structure SceneGraphExampleAsset where
  relativePath : String
  format : SceneGraphExampleAssetFormat
  kind : SceneGraphExampleAssetKind
  component : String
  feedsSceneGraph : Bool := false
  dependencies : Array String := #[]
  concept : String := ""
  deriving Repr, Inhabited

namespace SceneGraphExampleAsset

def fullPath (asset : SceneGraphExampleAsset) : String :=
  sceneGraphExampleRoot ++ "/" ++ asset.relativePath

def validate? (asset : SceneGraphExampleAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "SceneGraph asset path cannot be empty"
  if !asset.format.matchesPath asset.relativePath then
    .error s!"SceneGraph asset {asset.relativePath}: format does not match path"
  if asset.component.isEmpty then
    .error s!"SceneGraph asset {asset.relativePath}: component cannot be empty"
  if asset.concept.isEmpty then
    .error s!"SceneGraph asset {asset.relativePath}: concept cannot be empty"
  for dep in asset.dependencies do
    if dep.isEmpty then
      .error s!"SceneGraph asset {asset.relativePath}: dependency path cannot be empty"
    if dep == asset.relativePath then
      .error s!"SceneGraph asset {asset.relativePath}: cannot depend on itself"

end SceneGraphExampleAsset

def sceneGraphExampleAssets : Array SceneGraphExampleAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      format := .bazel
      kind := .metadata
      component := "build"
      concept := "build targets and data declarations for scene_graph examples"
    },
    {
      relativePath := "bouncing_ball_plant.cc"
      format := .cpp
      kind := .source
      component := "bouncing_ball"
      feedsSceneGraph := true
      dependencies := #["bouncing_ball_plant.h", "bouncing_ball_vector.h"]
      concept := "SceneGraph-aware bouncing-ball plant implementation"
    },
    {
      relativePath := "bouncing_ball_plant.h"
      format := .header
      kind := .source
      component := "bouncing_ball"
      feedsSceneGraph := true
      concept := "BouncingBallPlant LeafSystem declaration"
    },
    {
      relativePath := "bouncing_ball_run_dynamics.cc"
      format := .cpp
      kind := .source
      component := "bouncing_ball"
      feedsSceneGraph := true
      dependencies := #["bouncing_ball_plant.h"]
      concept := "bouncing-ball SceneGraph executable wiring"
    },
    {
      relativePath := "bouncing_ball_vector.cc"
      format := .cpp
      kind := .source
      component := "bouncing_ball"
      dependencies := #["bouncing_ball_vector.h"]
      concept := "BouncingBallVector coordinate implementation"
    },
    {
      relativePath := "bouncing_ball_vector.h"
      format := .header
      kind := .source
      component := "bouncing_ball"
      concept := "BouncingBallVector named-vector declaration"
    },
    {
      relativePath := "cuboctahedron_with_hole.mtl"
      format := .mtl
      kind := .material
      component := "solar_system"
      feedsSceneGraph := true
      dependencies := #["rainbow_checker.png"]
      concept := "OBJ material file that points at the rainbow checker texture"
    },
    {
      relativePath := "cuboctahedron_with_hole.obj"
      format := .obj
      kind := .mesh
      component := "solar_system"
      feedsSceneGraph := true
      dependencies := #["cuboctahedron_with_hole.mtl", "rainbow_checker.png"]
      concept := "convex satellite mesh with a hole and intentionally rejected texture material"
    },
    {
      relativePath := "planet_rings.obj"
      format := .obj
      kind := .mesh
      component := "solar_system"
      feedsSceneGraph := true
      concept := "Mars ring mesh registered as dynamic mesh geometry"
    },
    {
      relativePath := "rainbow_checker.png"
      format := .png
      kind := .texture
      component := "solar_system"
      feedsSceneGraph := true
      concept := "checker texture referenced by cuboctahedron material"
    },
    {
      relativePath := "simple_contact_surface_vis.cc"
      format := .cpp
      kind := .source
      component := "simple_contact_surface_vis"
      feedsSceneGraph := true
      concept := "contact-surface visualization query executable"
    },
    {
      relativePath := "solar_system.cc"
      format := .cpp
      kind := .source
      component := "solar_system"
      feedsSceneGraph := true
      dependencies := #["solar_system.h", "sun.gltf", "cuboctahedron_with_hole.obj", "planet_rings.obj"]
      concept := "SolarSystem geometry registration implementation"
    },
    {
      relativePath := "solar_system.h"
      format := .header
      kind := .source
      component := "solar_system"
      feedsSceneGraph := true
      concept := "SolarSystem LeafSystem declaration"
    },
    {
      relativePath := "solar_system_run_dynamics.cc"
      format := .cpp
      kind := .source
      component := "solar_system"
      feedsSceneGraph := true
      dependencies := #["solar_system.h"]
      concept := "SolarSystem SceneGraph executable wiring"
    },
    {
      relativePath := "sun.bin"
      format := .bin
      kind := .binaryBuffer
      component := "solar_system"
      feedsSceneGraph := true
      concept := "glTF binary buffer for the sun mesh"
    },
    {
      relativePath := "sun.gltf"
      format := .gltf
      kind := .mesh
      component := "solar_system"
      feedsSceneGraph := true
      dependencies := #["sun.bin", "sun.png", "sun.ktx2"]
      concept := "textured glTF sun mesh"
    },
    {
      relativePath := "sun.ktx2"
      format := .ktx2
      kind := .texture
      component := "solar_system"
      feedsSceneGraph := true
      concept := "BasisU compressed sun texture referenced by glTF"
    },
    {
      relativePath := "sun.png"
      format := .png
      kind := .texture
      component := "solar_system"
      feedsSceneGraph := true
      concept := "PNG sun texture referenced by glTF"
    },
    {
      relativePath := "sun.sdf"
      format := .sdf
      kind := .model
      component := "solar_system"
      feedsSceneGraph := true
      dependencies := #["sun.gltf"]
      concept := "SDF wrapper that loads the sun glTF visual mesh"
    }
  ]

private def hasDuplicateSceneGraphAssetPath
    (assets : Array SceneGraphExampleAsset) : Bool := Id.run do
  for i in [:assets.size] do
    for j in [:(assets.size - i - 1)] do
      let k := i + j + 1
      if assets[i]!.relativePath == assets[k]!.relativePath then
        return true
  return false

def sceneGraphExampleAssetPaths : Array String :=
  sceneGraphExampleAssets.map (fun asset => asset.fullPath)

def sceneGraphGeometryAssets : Array SceneGraphExampleAsset :=
  sceneGraphExampleAssets.filter (fun asset => asset.feedsSceneGraph)

def sceneGraphSolarAssets : Array SceneGraphExampleAsset :=
  sceneGraphExampleAssets.filter (fun asset => asset.component == "solar_system")

def findSceneGraphExampleAsset? (path : String) :
    Option SceneGraphExampleAsset :=
  sceneGraphExampleAssets.find? (fun asset =>
    asset.relativePath == path || asset.fullPath == path)

private def requiredSceneGraphExampleAssetPaths : Array String :=
  #[
    "BUILD.bazel",
    "bouncing_ball_plant.cc",
    "bouncing_ball_plant.h",
    "bouncing_ball_run_dynamics.cc",
    "bouncing_ball_vector.cc",
    "bouncing_ball_vector.h",
    "cuboctahedron_with_hole.mtl",
    "cuboctahedron_with_hole.obj",
    "planet_rings.obj",
    "rainbow_checker.png",
    "simple_contact_surface_vis.cc",
    "solar_system.cc",
    "solar_system.h",
    "solar_system_run_dynamics.cc",
    "sun.bin",
    "sun.gltf",
    "sun.ktx2",
    "sun.png",
    "sun.sdf"
  ]

def validateSceneGraphExampleAssetCatalog? : Except String Unit := do
  if sceneGraphExampleAssets.size != requiredSceneGraphExampleAssetPaths.size then
    .error s!"SceneGraph asset catalog size {sceneGraphExampleAssets.size} != expected {requiredSceneGraphExampleAssetPaths.size}"
  if hasDuplicateSceneGraphAssetPath sceneGraphExampleAssets then
    .error "SceneGraph asset catalog contains duplicate paths"
  for asset in sceneGraphExampleAssets do
    asset.validate?
  for path in requiredSceneGraphExampleAssetPaths do
    match findSceneGraphExampleAsset? path with
    | some _ => pure ()
    | none => .error s!"SceneGraph asset catalog missing required path {path}"
  for asset in sceneGraphExampleAssets do
    for dep in asset.dependencies do
      match findSceneGraphExampleAsset? dep with
      | some _ => pure ()
      | none => .error s!"SceneGraph asset {asset.relativePath}: missing dependency {dep}"
  if sceneGraphSolarAssets.size != 12 then
    .error s!"SceneGraph solar asset count {sceneGraphSolarAssets.size} != 12"

/-! ## Bouncing ball SceneGraph -/

def ball1SourceId : Nat := 1
def ball2SourceId : Nat := 2
def anchoredSourceId : Nat := 3

def ball1FrameId : Nat := 101
def ball2FrameId : Nat := 201

def ball1GeometryId : Nat := 1001
def ball2GeometryId : Nat := 2001
def groundGeometryId : Nat := 3001

def statusCameraImageChannel : String := "DRAKE_RGBD_CAMERA_IMAGES"

def bouncingBallVectorNumCoordinates : Nat := 2
def bouncingBallVectorZIndex : Nat := 0
def bouncingBallVectorZdotIndex : Nat := 1

def bouncingBallVectorCoordinateNames : Array String :=
  #["z", "zdot"]

structure BouncingBallVectorSpec where
  vectorName : String := "BouncingBallVector"
  numCoordinates : Nat := bouncingBallVectorNumCoordinates
  qCount : Nat := 1
  vCount : Nat := 1
  zCount : Nat := 0
  coordinateNames : Array String := bouncingBallVectorCoordinateNames
  deriving Repr, BEq, Inhabited

namespace BouncingBallVectorSpec

def validate? (spec : BouncingBallVectorSpec) : Except String Unit := do
  if spec.vectorName.isEmpty then
    .error "BouncingBallVectorSpec vector name cannot be empty"
  if spec.numCoordinates != bouncingBallVectorNumCoordinates then
    .error s!"BouncingBallVectorSpec coordinate count {spec.numCoordinates} != 2"
  if spec.coordinateNames != bouncingBallVectorCoordinateNames then
    .error s!"BouncingBallVectorSpec coordinate names {spec.coordinateNames} != #[\"z\", \"zdot\"]"
  if spec.qCount != 1 || spec.vCount != 1 || spec.zCount != 0 then
    .error s!"BouncingBallVectorSpec q/v/z split should be 1/1/0, got {spec.qCount}/{spec.vCount}/{spec.zCount}"
  if spec.qCount + spec.vCount + spec.zCount != spec.numCoordinates then
    .error "BouncingBallVectorSpec q/v/z split does not sum to coordinate count"

def coordinateName? (spec : BouncingBallVectorSpec) (index : Nat) : Option String :=
  if h : index < spec.coordinateNames.size then
    some spec.coordinateNames[index]
  else
    none

end BouncingBallVectorSpec

def bouncingBallVectorSpec : BouncingBallVectorSpec := {}

structure BouncingBallSceneParams where
  simulationTime : Float := 10.0
  maximumStepSize : Float := 0.002
  renderFps : Float := 10.0
  diameter : Float := 0.1
  mass : Float := 0.1
  gravity : Float := 9.81
  stiffness : Float := 981.0
  dissipation : Float := 0.0
  distanceTol : Float := 0.0
  deriving Repr, Inhabited

namespace BouncingBallSceneParams

def radius (p : BouncingBallSceneParams) : Float :=
  p.diameter / 2.0

def validate? (p : BouncingBallSceneParams) : Except String Unit := do
  if !(Float.isFinite p.simulationTime) || p.simulationTime <= 0.0 then
    .error s!"simulation_time must be positive and finite, got {p.simulationTime}"
  if !(Float.isFinite p.maximumStepSize) || p.maximumStepSize <= 0.0 then
    .error s!"maximum step size must be positive and finite, got {p.maximumStepSize}"
  if !(Float.isFinite p.renderFps) || p.renderFps <= 0.0 then
    .error s!"render_fps must be positive and finite, got {p.renderFps}"
  if !(Float.isFinite p.diameter) || p.diameter <= 0.0 then
    .error s!"diameter must be positive and finite, got {p.diameter}"
  if !(Float.isFinite p.mass) || p.mass <= 0.0 then
    .error s!"mass must be positive and finite, got {p.mass}"
  if !(Float.isFinite p.gravity) || p.gravity < 0.0 then
    .error s!"gravity must be nonnegative and finite, got {p.gravity}"
  if !(Float.isFinite p.stiffness) || p.stiffness < 0.0 then
    .error s!"stiffness must be nonnegative and finite, got {p.stiffness}"
  if !(Float.isFinite p.dissipation) || p.dissipation < 0.0 then
    .error s!"dissipation must be nonnegative and finite, got {p.dissipation}"

def drakeStiffnessFromStaticPenetration (p : BouncingBallSceneParams)
    (penetration : Float := 0.001) : Float :=
  p.mass * p.gravity / penetration

end BouncingBallSceneParams

def bouncingBallParams : BouncingBallSceneParams := {}

structure BouncingBallSceneState where
  z : Float := 0.3
  zdot : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace BouncingBallSceneState

def isFinite (x : BouncingBallSceneState) : Bool :=
  Float.isFinite x.z && Float.isFinite x.zdot

def isValid (x : BouncingBallSceneState) : Bool :=
  x.isFinite

def toArray (x : BouncingBallSceneState) : Array Float :=
  #[x.z, x.zdot]

def ofArray? (xs : Array Float) : Except String BouncingBallSceneState := do
  if xs.size != bouncingBallVectorNumCoordinates then
    .error s!"BouncingBallVector input size {xs.size} != {bouncingBallVectorNumCoordinates}"
  let x : BouncingBallSceneState := {
    z := xs.getD bouncingBallVectorZIndex 0.0
    zdot := xs.getD bouncingBallVectorZdotIndex 0.0
  }
  if !x.isValid then
    .error s!"BouncingBallVector values must be finite, got {xs}"
  pure x

def withZ (x : BouncingBallSceneState) (z : Float) : BouncingBallSceneState :=
  { x with z := z }

def withZdot (x : BouncingBallSceneState) (zdot : Float) : BouncingBallSceneState :=
  { x with zdot := zdot }

def serialize (x : BouncingBallSceneState) : Array (String × Float) :=
  #[("z", x.z), ("zdot", x.zdot)]

end BouncingBallSceneState

structure BouncingBallDerivative where
  zdot : Float
  zddot : Float
  normalForce : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace BouncingBallDerivative

def isFinite (dx : BouncingBallDerivative) : Bool :=
  Float.isFinite dx.zdot && Float.isFinite dx.zddot && Float.isFinite dx.normalForce

end BouncingBallDerivative

def ball1InitialState : BouncingBallSceneState := { z := 0.3, zdot := 0.0 }
def ball2InitialState : BouncingBallSceneState := { z := 0.3, zdot := 0.3 }

def ballProjectedPosition (ballIndex : Nat) : SceneVec3 :=
  if ballIndex == 1 then
    { x := 0.25, y := 0.25 }
  else
    { x := -0.25, y := -0.25 }

private def ballProperties (id : Nat) : SceneGeometryProperties :=
  {
    roles := #[.illustration, .proximity, .perception]
    diffuseRgba? := some { r := 0.8, g := 0.8, b := 0.8, a := 1.0 }
    renderLabel? := some id
  }

private def groundProperties (id : Nat) : SceneGeometryProperties :=
  {
    roles := #[.illustration, .proximity, .perception]
    diffuseRgba? := some { r := 0.8, g := 0.8, b := 0.8, a := 1.0 }
    renderLabel? := some id
  }

private def ballGeometry
    (sourceId frameId geometryId : Nat) (name : String)
    (p : BouncingBallSceneParams := bouncingBallParams) : SceneGeometry :=
  {
    id := geometryId
    sourceId := sourceId
    frameId? := some frameId
    X_FG := .identity
    shape := .sphere p.radius
    name := name
    properties := ballProperties geometryId
  }

def bouncingBallSceneGraph
    (p : BouncingBallSceneParams := bouncingBallParams) : SceneGraphProvider :=
  {
    label := "scene_graph_bouncing_ball"
    sources := #[
      { id := ball1SourceId, name := "ball1" },
      { id := ball2SourceId, name := "ball2" },
      { id := anchoredSourceId, name := "anchored" }
    ]
    frames := #[
      { id := ball1FrameId, sourceId := ball1SourceId, name := "ball_frame" },
      { id := ball2FrameId, sourceId := ball2SourceId, name := "ball_frame" }
    ]
    geometries := #[
      ballGeometry ball1SourceId ball1FrameId ball1GeometryId "ball" p,
      ballGeometry ball2SourceId ball2FrameId ball2GeometryId "ball" p,
      {
        id := groundGeometryId
        sourceId := anchoredSourceId
        frameId? := none
        X_FG := .identity
        shape := .halfSpace SceneVec3.unitZ SceneVec3.zero
        name := "ground"
        properties := groundProperties groundGeometryId
      }
    ]
  }

def ballFramePose (ballIndex : Nat) (state : BouncingBallSceneState) :
    SceneFramePose :=
  let pxy := ballProjectedPosition ballIndex
  {
    frameId := if ballIndex == 1 then ball1FrameId else ball2FrameId
    X_WF := {
      translation := { x := pxy.x, y := pxy.y, z := state.z }
      rotationAxis := SceneVec3.unitZ
      rotationAngle := 0.0
    }
  }

def bouncingBallFramePoses
    (x1 : BouncingBallSceneState := ball1InitialState)
    (x2 : BouncingBallSceneState := ball2InitialState) : SceneFramePoseVector :=
  { poses := #[ballFramePose 1 x1, ballFramePose 2 x2] }

def ballCenter_W (ballIndex : Nat) (state : BouncingBallSceneState) : SceneVec3 :=
  let pxy := ballProjectedPosition ballIndex
  { x := pxy.x, y := pxy.y, z := state.z }

def contactCandidate?
    (provider : SceneGraphProvider)
    (ballIndex : Nat)
    (state : BouncingBallSceneState) : Except String ContactCandidate :=
  let geometryId := if ballIndex == 1 then ball1GeometryId else ball2GeometryId
  sphereHalfSpaceContactCandidate?
    provider geometryId groundGeometryId (ballCenter_W ballIndex state)
    state.zdot #[1.0] #[0.0] #[]
    s!"ball{ballIndex}-ground"

def contactSupport?
    (p : BouncingBallSceneParams := bouncingBallParams)
    (x1 : BouncingBallSceneState := ball1InitialState)
    (x2 : BouncingBallSceneState := ball2InitialState) : Except String ContactSupport := do
  let provider := bouncingBallSceneGraph p
  let c1 ← contactCandidate? provider 1 x1
  let c2 ← contactCandidate? provider 2 x2
  pure (ContactSupport.selectByDistance p.distanceTol #[c1, c2] "scene_graph_bouncing_ball")

def huntCrossleyNormalForce
    (p : BouncingBallSceneParams) (candidate : ContactCandidate) : Float :=
  let depth := max 0.0 (-candidate.signedDistance)
  let penetrationRate := -candidate.normalVelocity
  max 0.0 (p.stiffness * depth * (1.0 + p.dissipation * penetrationRate))

def derivativeFromCandidate
    (p : BouncingBallSceneParams)
    (state : BouncingBallSceneState)
    (candidate : ContactCandidate) : BouncingBallDerivative :=
  let fN := huntCrossleyNormalForce p candidate
  {
    zdot := state.zdot
    zddot := (-p.mass * p.gravity + fN) / p.mass
    normalForce := fN
  }

def derivative?
    (p : BouncingBallSceneParams := bouncingBallParams)
    (ballIndex : Nat := 1)
    (state : BouncingBallSceneState := ball1InitialState) :
    Except String BouncingBallDerivative := do
  let candidate ← contactCandidate? (bouncingBallSceneGraph p) ballIndex state
  pure (derivativeFromCandidate p state candidate)

def pointPairPenetration?
    (p : BouncingBallSceneParams := bouncingBallParams)
    (ballIndex : Nat := 1)
    (state : BouncingBallSceneState := ball1InitialState) :
    Except String (Option ScenePointPairPenetration) :=
  let geometryId := if ballIndex == 1 then ball1GeometryId else ball2GeometryId
  sphereHalfSpacePenetration?
    (bouncingBallSceneGraph p) geometryId groundGeometryId
    (ballCenter_W ballIndex state)
    s!"ball{ballIndex}-ground"

/-! ## SceneGraph query to full-physics primitive adapter -/

def fullPhysicsPrimitivesFromSceneContactQuery?
    (query : SceneContactQueryResult)
    (massMatrix : Array (Array Float))
    (qdot actuationForces : Array Float)
    (biasForces : Array Float := #[])
    (supportPolicy : SupportPolicy := .fullSupport)
    (contactForceSource : ContactForceSource := .precomputed)
    (contactForces : Array ContactForceScalars := #[])
    (compliantContactModel : CompliantContactModel := {})
    (distanceTol : Float := 0.0)
    (tangentVelocityTol : Float := 1.0e-9)
    (label : String := "") :
    Except String FullPhysicsPrimitives := do
  let candidateSet ← query.solverContactCandidateSet?
  candidateSet.validate? (some qdot.size)
  pure {
    massMatrix := massMatrix
    qdot := qdot
    actuationForces := actuationForces
    biasForces := biasForces
    contactCandidates := candidateSet.candidates
    sourceContactCandidateCount? := some candidateSet.totalCandidates
    supportPolicy := supportPolicy
    contactForceSource := contactForceSource
    contactForces := contactForces
    compliantContactModel := compliantContactModel
    distanceTol := distanceTol
    tangentVelocityTol := tangentVelocityTol
    label := if label.isEmpty then query.label else label
  }

structure BouncingBallScenePhysicsState where
  params : BouncingBallSceneParams := bouncingBallParams
  ballIndex : Nat := 1
  state : BouncingBallSceneState := ball1InitialState
  deriving Repr, Inhabited

namespace BouncingBallScenePhysicsState

def validate? (snapshot : BouncingBallScenePhysicsState) : Except String Unit := do
  snapshot.params.validate?
  if snapshot.ballIndex != 1 && snapshot.ballIndex != 2 then
    .error s!"bouncing-ball SceneGraph physics provider expects ballIndex 1 or 2, got {snapshot.ballIndex}"
  if !snapshot.state.isValid then
    .error s!"bouncing-ball SceneGraph physics provider state must be finite, got {snapshot.state.toArray}"

def provider (snapshot : BouncingBallScenePhysicsState) : SceneGraphProvider :=
  bouncingBallSceneGraph snapshot.params

def contactQuery? (snapshot : BouncingBallScenePhysicsState) :
    Except String SceneContactQueryResult := do
  snapshot.validate?
  let sceneProvider := snapshot.provider
  sceneProvider.validate?
  let candidate ← contactCandidate? sceneProvider snapshot.ballIndex snapshot.state
  let candidateSet := ContactCandidateSet.ofArray #[candidate]
    s!"bouncing-ball{snapshot.ballIndex} solver candidates"
  candidateSet.validate? (some 1)
  let query : SceneContactQueryResult := {
    providerLabel := sceneProvider.label
    candidates := candidateSet
    label := s!"bouncing-ball{snapshot.ballIndex} scene contact query"
  }
  query.validate? (some 1)
  pure query

def contactModel (snapshot : BouncingBallScenePhysicsState) :
    CompliantContactModel :=
  {
    normalStiffness := snapshot.params.stiffness
    normalDamping := snapshot.params.stiffness * snapshot.params.dissipation
    tangentDamping := 0.0
    friction := CoulombFriction.frictionless
    label := "bouncing-ball Hunt-Crossley primitive"
  }

def fullPhysicsPrimitives?
    (snapshot : BouncingBallScenePhysicsState)
    (label : String := "bouncing-ball SceneGraph full physics provider") :
    Except String FullPhysicsPrimitives := do
  let query ← snapshot.contactQuery?
  fullPhysicsPrimitivesFromSceneContactQuery?
    query
    #[#[snapshot.params.mass]]
    #[snapshot.state.zdot]
    #[0.0]
    #[snapshot.params.mass * snapshot.params.gravity]
    (.threshold snapshot.params.distanceTol)
    .compliantModel
    #[]
    snapshot.contactModel
    snapshot.params.distanceTol
    1.0e-9
    label

end BouncingBallScenePhysicsState

def bouncingBallScenePhysicsState
    (p : BouncingBallSceneParams := bouncingBallParams)
    (ballIndex : Nat := 1)
    (state : BouncingBallSceneState := ball1InitialState) :
    BouncingBallScenePhysicsState :=
  { params := p, ballIndex := ballIndex, state := state }

def bouncingBallSceneFullPhysicsPrimitiveProvider
    (label : String := "bouncing-ball SceneGraph full physics provider") :
    FullPhysicsPrimitiveProvider BouncingBallScenePhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => snapshot.fullPhysicsPrimitives? label
  }

def bouncingBallSkeletonGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 10, kind := .state .boundary, label := "ball continuous state" }
    |>.addVertex { id := 20, kind := .opaque, label := "SceneGraph provider" }
    |>.addVertex { id := 30, kind := .eventTime, label := "point-pair penetration query" }
    |>.addVertex { id := 40, kind := .interval, label := "Hunt-Crossley dynamics" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[30]
      reads := #[10, 20]
      writes := #[30]
      label := "SceneGraph query object to ContactCandidate"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[40]
      reads := #[10, 30]
      writes := #[10]
      label := "differentiate compliant bouncing-ball dynamics"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[20]
      reads := #[10]
      writes := #[20]
      label := "emit FramePoseVector to SceneGraph"
    }

structure BouncingBallSceneResult where
  references : Array DrakeReference
  vectorSpec : BouncingBallVectorSpec
  ball1StateVector : Array Float
  ball2StateVector : Array Float
  provider : SceneGraphProvider
  poses : SceneFramePoseVector
  support : ContactSupport
  ball1Derivative : BouncingBallDerivative
  ball2Derivative : BouncingBallDerivative
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildBouncingBall? (p : BouncingBallSceneParams := bouncingBallParams) :
    Except String BouncingBallSceneResult := do
  p.validate?
  bouncingBallVectorSpec.validate?
  let _ ← BouncingBallSceneState.ofArray? ball1InitialState.toArray
  let _ ← BouncingBallSceneState.ofArray? ball2InitialState.toArray
  let provider := bouncingBallSceneGraph p
  provider.validate?
  let poses := bouncingBallFramePoses
  poses.validate? provider
  let support ← contactSupport? p
  support.validateJacobianWidth? 1
  let dx1 ← derivative? p 1 ball1InitialState
  let dx2 ← derivative? p 2 ball2InitialState
  pure {
    references := drakeReferences
    vectorSpec := bouncingBallVectorSpec
    ball1StateVector := ball1InitialState.toArray
    ball2StateVector := ball2InitialState.toArray
    provider := provider
    poses := poses
    support := support
    ball1Derivative := dx1
    ball2Derivative := dx2
    graph := bouncingBallSkeletonGraph
    moves := bouncingBallSkeletonGraph.moves
  }

/-! ## Solar-system SceneGraph registration -/

def solarSourceId : Nat := 10
def solarBodyCount : Nat := 7

structure SolarBodySpec where
  frameId : Nat
  name : String
  parentFrameId? : Option Nat := none
  offset : SceneVec3 := {}
  axis : SceneVec3 := SceneVec3.unitZ
  initialAngle : Float := 0.0
  angularRate : Float := 0.0
  deriving Repr, BEq, Inhabited

def earthFrameId : Nat := 1101
def lunaFrameId : Nat := 1102
def convexSatelliteFrameId : Nat := 1103
def boxSatelliteFrameId : Nat := 1104
def capsuleSatelliteFrameId : Nat := 1105
def marsFrameId : Nat := 1106
def phobosFrameId : Nat := 1107

def solarBodySpecs : Array SolarBodySpec :=
  let earthBottom := -1.25
  let marsBottom := -1.5
  #[
    {
      frameId := earthFrameId
      name := "EarthOrbit"
      offset := { z := earthBottom }
      axis := SceneVec3.unitZ
      initialAngle := 0.0
      angularRate := 2.0 * pi / 5.0
    },
    {
      frameId := lunaFrameId
      name := "LunaOrbit"
      parentFrameId? := some earthFrameId
      offset := { x := 3.0, z := -earthBottom }
      axis := { x := 1.0, y := 1.0, z := 1.0 }
      initialAngle := pi / 2.0
      angularRate := 2.0 * pi
    },
    {
      frameId := convexSatelliteFrameId
      name := "ConvexSatelliteOrbit"
      parentFrameId? := some earthFrameId
      offset := { x := 3.0, z := -earthBottom }
      axis := { x := 1.0, y := 1.0, z := 1.0 }
      initialAngle := 7.0 * pi / 6.0
      angularRate := 2.0 * pi
    },
    {
      frameId := boxSatelliteFrameId
      name := "BoxSatelliteOrbit"
      parentFrameId? := some earthFrameId
      offset := { x := 3.0, z := -earthBottom }
      axis := { x := 1.0, y := 1.0, z := 1.0 }
      initialAngle := 11.0 * pi / 6.0
      angularRate := 2.0 * pi
    },
    {
      frameId := capsuleSatelliteFrameId
      name := "CapsuleSatelliteOrbit"
      parentFrameId? := some earthFrameId
      offset := { x := 3.0, z := -earthBottom }
      axis := { x := 1.0, y := 1.0, z := 1.0 }
      initialAngle := pi / 6.0
      angularRate := 2.0 * pi
    },
    {
      frameId := marsFrameId
      name := "MarsOrbit"
      offset := { z := marsBottom }
      axis := { y := 0.1, z := 1.0 }
      initialAngle := pi / 2.0
      angularRate := 2.0 * pi / 6.0
    },
    {
      frameId := phobosFrameId
      name := "PhobosOrbit"
      parentFrameId? := some marsFrameId
      offset := { x := 5.0, z := -marsBottom }
      axis := { z := -1.0 }
      initialAngle := 0.0
      angularRate := 2.0 * pi / 1.1
    }
  ]

private def illustration (rgba : SceneRgba) : SceneGeometryProperties :=
  { roles := #[.illustration], diffuseRgba? := some rgba }

private def solarFrame (spec : SolarBodySpec) : SceneFrame :=
  {
    id := spec.frameId
    sourceId := solarSourceId
    name := spec.name
    parentFrameId? := spec.parentFrameId?
  }

private def anchoredSolarGeometry
    (id : Nat) (shape : SceneGeometryShape) (name : String)
    (pose : ScenePose3) (rgba : SceneRgba) : SceneGeometry :=
  {
    id := id
    sourceId := solarSourceId
    frameId? := none
    X_FG := pose
    shape := shape
    name := name
    properties := illustration rgba
  }

private def dynamicSolarGeometry
    (id frameId : Nat) (shape : SceneGeometryShape) (name : String)
    (pose : ScenePose3) (rgba : SceneRgba) : SceneGeometry :=
  {
    id := id
    sourceId := solarSourceId
    frameId? := some frameId
    X_FG := pose
    shape := shape
    name := name
    properties := illustration rgba
  }

def solarSceneGraph : SceneGraphProvider :=
  let postMaterial : SceneRgba := { r := 0.3, g := 0.15, b := 0.05, a := 1.0 }
  {
    label := "scene_graph_solar_system"
    sources := #[{ id := solarSourceId, name := "solar_system" }]
    frames := solarBodySpecs.map solarFrame
    geometries := #[
      anchoredSolarGeometry 1201
        (.mesh "../drake/examples/scene_graph/sun.gltf" 1.0
          #["sun.bin", "sun.png", "sun.ktx2"])
        "Sun" .identity { r := 1.0, g := 0.8, b := 0.0, a := 1.0 },
      anchoredSolarGeometry 1202 (.cylinder 0.05 1.0) "Post"
        { translation := { z := -1.0 } } postMaterial,
      dynamicSolarGeometry 1210 earthFrameId (.sphere 0.25) "Earth"
        { translation := { x := 3.0, z := 1.25 } }
        { r := 0.0, g := 0.0, b := 1.0, a := 1.0 },
      dynamicSolarGeometry 1211 earthFrameId (.cylinder 0.05 3.0) "EarthHorzArm"
        { translation := { x := 1.5 } } postMaterial,
      dynamicSolarGeometry 1212 earthFrameId (.cylinder 0.05 1.25) "EarthVertArm"
        { translation := { x := 3.0, z := 0.625 } } postMaterial,
      dynamicSolarGeometry 1220 lunaFrameId (.sphere 0.075) "Luna"
        { translation := { x := -0.285773803, y := 0.1428869015, z := 0.1428869015 } }
        { r := 0.5, g := 0.5, b := 0.35, a := 1.0 },
      dynamicSolarGeometry 1230 convexSatelliteFrameId
        (.convex "../drake/examples/scene_graph/cuboctahedron_with_hole.obj" 0.075)
        "ConvexSatellite"
        { translation := { x := -0.285773803, y := 0.1428869015, z := 0.1428869015 } }
        { r := 1.0, g := 1.0, b := 0.0, a := 1.0 },
      dynamicSolarGeometry 1240 boxSatelliteFrameId (.box 0.15 0.15 0.15)
        "BoxSatellite"
        { translation := { x := -0.285773803, y := 0.1428869015, z := 0.1428869015 } }
        { r := 1.0, g := 0.0, b := 1.0, a := 1.0 },
      dynamicSolarGeometry 1250 capsuleSatelliteFrameId (.capsule 0.075 0.2)
        "CapsuleSatellite"
        { translation := { x := -0.285773803, y := 0.1428869015, z := 0.1428869015 } }
        { r := 0.0, g := 1.0, b := 1.0, a := 1.0 },
      dynamicSolarGeometry 1260 marsFrameId (.sphere 0.24) "Mars"
        { translation := { x := 5.0, z := 1.5 } }
        { r := 0.9, g := 0.1, b := 0.0, a := 1.0 },
      dynamicSolarGeometry 1261 marsFrameId
        (.mesh "../drake/examples/scene_graph/planet_rings.obj" 0.24 #[])
        "MarsRings"
        { translation := { x := 5.0, z := 1.5 }, rotationAxis := { x := 1.0, y := 1.0, z := 1.0 }, rotationAngle := pi / 3.0 }
        { r := 0.45, g := 0.9, b := 0.0, a := 1.0 },
      dynamicSolarGeometry 1262 marsFrameId (.cylinder 0.05 5.0) "MarsHorzArm"
        { translation := { x := 2.5 } } postMaterial,
      dynamicSolarGeometry 1263 marsFrameId (.cylinder 0.05 1.5) "MarsVertArm"
        { translation := { x := 5.0, z := 0.75 } } postMaterial,
      dynamicSolarGeometry 1270 phobosFrameId (.sphere 0.06) "Phobos"
        { translation := { x := 0.34 } }
        { r := 0.65, g := 0.6, b := 0.8, a := 1.0 }
    ]
  }

def validateSolarSceneGraphAssetUsage? : Except String Unit := do
  validateSceneGraphExampleAssetCatalog?
  let sun ←
    match solarSceneGraph.geometryById? 1201 with
    | some geometry => pure geometry
    | none => .error "Solar SceneGraph is missing sun geometry 1201"
  match sun.shape with
  | .mesh uri scale supportingFiles =>
      if uri != "../drake/examples/scene_graph/sun.gltf" then
        .error s!"Solar sun geometry should use sun.gltf, got {uri}"
      if scale != 1.0 then
        .error s!"Solar sun geometry scale should be 1.0, got {scale}"
      if supportingFiles != #["sun.bin", "sun.png", "sun.ktx2"] then
        .error s!"Solar sun supporting files mismatch: {supportingFiles}"
      if (findSceneGraphExampleAsset? uri).isNone then
        .error s!"Solar sun mesh URI is not in the SceneGraph asset catalog: {uri}"
      for dep in supportingFiles do
        if (findSceneGraphExampleAsset? dep).isNone then
          .error s!"Solar sun mesh dependency is not in the SceneGraph asset catalog: {dep}"
  | other => .error s!"Solar sun geometry should be a mesh, got {reprStr other}"
  let convexSatellite ←
    match solarSceneGraph.geometryById? 1230 with
    | some geometry => pure geometry
    | none => .error "Solar SceneGraph is missing convex satellite geometry 1230"
  match convexSatellite.shape with
  | .convex uri _ =>
      if uri != "../drake/examples/scene_graph/cuboctahedron_with_hole.obj" then
        .error s!"Solar convex satellite should use cuboctahedron_with_hole.obj, got {uri}"
      if (findSceneGraphExampleAsset? uri).isNone then
        .error s!"Solar convex satellite asset is missing from catalog: {uri}"
  | other => .error s!"Solar convex satellite geometry should be convex, got {reprStr other}"
  let marsRings ←
    match solarSceneGraph.geometryById? 1261 with
    | some geometry => pure geometry
    | none => .error "Solar SceneGraph is missing Mars rings geometry 1261"
  match marsRings.shape with
  | .mesh uri _ supportingFiles =>
      if uri != "../drake/examples/scene_graph/planet_rings.obj" then
        .error s!"Solar Mars rings should use planet_rings.obj, got {uri}"
      if !supportingFiles.isEmpty then
        .error s!"Solar Mars rings should not require supporting files, got {supportingFiles}"
      if (findSceneGraphExampleAsset? uri).isNone then
        .error s!"Solar Mars rings mesh asset is missing from catalog: {uri}"
  | other => .error s!"Solar Mars rings geometry should be a mesh, got {reprStr other}"

def solarDefaultState : Array Float :=
  (solarBodySpecs.map (fun spec => spec.initialAngle)) ++
  (solarBodySpecs.map (fun spec => spec.angularRate))

def solarDerivative? (state : Array Float := solarDefaultState) :
    Except String (Array Float) := do
  if state.size != solarBodyCount * 2 then
    .error s!"solar state size {state.size} != {solarBodyCount * 2}"
  let mut out : Array Float := #[]
  for i in [:solarBodyCount] do
    out := out.push (state.getD (i + solarBodyCount) 0.0)
  for _ in [:solarBodyCount] do
    out := out.push 0.0
  pure out

def solarFramePoseOutput? (state : Array Float := solarDefaultState) :
    Except String SceneFramePoseVector := do
  if state.size != solarBodyCount * 2 then
    .error s!"solar state size {state.size} != {solarBodyCount * 2}"
  let mut poses : Array SceneFramePose := #[]
  for i in [:solarBodySpecs.size] do
    let spec := solarBodySpecs[i]!
    poses := poses.push {
      frameId := spec.frameId
      X_WF := {
        translation := spec.offset
        rotationAxis := spec.axis
        rotationAngle := state.getD i 0.0
      }
    }
  pure { poses := poses }

structure SolarSystemResult where
  assetCatalog : Array SceneGraphExampleAsset
  provider : SceneGraphProvider
  defaultState : Array Float
  framePoses : SceneFramePoseVector
  derivative : Array Float
  deriving Repr, Inhabited

def buildSolarSystem? : Except String SolarSystemResult := do
  validateSolarSceneGraphAssetUsage?
  solarSceneGraph.validate?
  let poses ← solarFramePoseOutput? solarDefaultState
  poses.validate? solarSceneGraph
  let dx ← solarDerivative? solarDefaultState
  pure {
    assetCatalog := sceneGraphExampleAssets
    provider := solarSceneGraph
    defaultState := solarDefaultState
    framePoses := poses
    derivative := dx
  }

/-! ## Solar-system executable boundary -/

structure SolarRunDynamicsParams where
  simulationTime : Float := 13.0
  maximumStepSize : Float := 0.002
  targetRealtimeRate : Float := 1.0
  addDrakeVisualizer : Bool := true
  addMeshcatVisualizer : Bool := true
  deriving Repr, Inhabited

namespace SolarRunDynamicsParams

def validate? (p : SolarRunDynamicsParams) : Except String Unit := do
  if !p.simulationTime.isFinite || p.simulationTime <= 0.0 then
    .error s!"solar_system_run_dynamics simulation_time must be positive and finite, got {p.simulationTime}"
  if !p.maximumStepSize.isFinite || p.maximumStepSize <= 0.0 then
    .error s!"solar_system_run_dynamics maximum step size must be positive and finite, got {p.maximumStepSize}"
  if !p.targetRealtimeRate.isFinite || p.targetRealtimeRate < 0.0 then
    .error s!"solar_system_run_dynamics target realtime rate must be nonnegative and finite, got {p.targetRealtimeRate}"

end SolarRunDynamicsParams

def solarRunDynamicsParams : SolarRunDynamicsParams := {}

def solarRunDynamicsGraph
    (p : SolarRunDynamicsParams := solarRunDynamicsParams) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 6100, kind := .state .boundary, label := "../drake/examples/scene_graph/solar_system_run_dynamics.cc flags" }
    |>.addVertex { id := 6101, kind := .state .interior, label := "DiagramBuilder" }
    |>.addVertex { id := 6102, kind := .opaque, label := "SceneGraph<double> scene_graph" }
    |>.addVertex { id := 6103, kind := .state .interior, label := "SolarSystem leaf system" }
    |>.addVertex { id := 6104, kind := .state .boundary, label := "source pose port for solar_system source_id" }
    |>.addVertex { id := 6105, kind := .state .boundary, label := "DrakeVisualizerd sink" }
    |>.addVertex { id := 6106, kind := .state .boundary, label := "MeshcatVisualizer sink" }
    |>.addVertex { id := 6107, kind := .interval, label := "Simulator.AdvanceTo solar_system_run_dynamics" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[6101, 6102, 6103]
      reads := #[6100]
      writes := #[6101, 6102, 6103]
      label := "DiagramBuilder.AddSystem SceneGraph and SolarSystem(scene_graph)"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[6104]
      reads := #[6103]
      writes := #[6102, 6104]
      label := "Connect SolarSystem.geometry_pose_output to SceneGraph.source_pose_port(source_id)"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[6105, 6106]
      reads := #[6102]
      writes :=
        (if p.addDrakeVisualizer && p.addMeshcatVisualizer then
          #[6105, 6106]
        else if p.addDrakeVisualizer then
          #[6105]
        else if p.addMeshcatVisualizer then
          #[6106]
        else
          #[])
      label := "DrakeVisualizerd::AddToBuilder and MeshcatVisualizer::AddToBuilder"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[6107]
      reads := #[6101, 6102, 6103, 6104]
      writes := #[6107]
      cost := { work := p.simulationTime / p.maximumStepSize }
      label := "Simulator.Initialize; set_maximum_step_size(0.002); AdvanceTo(FLAGS_simulation_time)"
    }

structure SolarRunDynamicsResult where
  assetCatalog : Array SceneGraphExampleAsset
  params : SolarRunDynamicsParams
  provider : SceneGraphProvider
  sourceId : Nat
  defaultState : Array Float
  initialFramePoses : SceneFramePoseVector
  initialDerivative : Array Float
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildSolarRunDynamics?
    (p : SolarRunDynamicsParams := solarRunDynamicsParams) :
    Except String SolarRunDynamicsResult := do
  p.validate?
  validateSolarSceneGraphAssetUsage?
  solarSceneGraph.validate?
  let poses ← solarFramePoseOutput? solarDefaultState
  poses.validate? solarSceneGraph
  let dx ← solarDerivative? solarDefaultState
  let graph := solarRunDynamicsGraph p
  pure {
    assetCatalog := sceneGraphExampleAssets
    params := p
    provider := solarSceneGraph
    sourceId := solarSourceId
    defaultState := solarDefaultState
    initialFramePoses := poses
    initialDerivative := dx
    graph := graph
    moves := graph.moves
  }

/-! ## Simple contact-surface visualization boundary -/

def contactSurfaceBallModelInstance : Nat := 1
def contactSurfaceCylinderModelInstance : Nat := 2

def contactSurfaceMovingBallSourceId : Nat := 4200
def contactSurfaceWorldSourceId : Nat := 4201

def contactSurfaceMovingBallFrameId : Nat := 4210
def contactSurfaceCanFrameId : Nat := 4211

def contactSurfaceBallGeometryId : Nat := 4220
def contactSurfaceBoxGeometryId : Nat := 4221
def contactSurfaceCan1GeometryId : Nat := 4222
def contactSurfaceCan2GeometryId : Nat := 4223

def contactSurfaceBoxPatchId : Nat := 4230
def contactSurfaceCan1PatchId : Nat := 4231
def contactSurfaceCan2PatchId : Nat := 4232

def contactSurfaceLcmChannel : String := "CONTACT_RESULTS"

structure ContactSurfaceVisParams where
  simulationTime : Float := 10.0
  realTime : Bool := true
  length : Float := 1.0
  rigidCylinders : Bool := true
  hybrid : Bool := false
  polygons : Bool := false
  forceFullName : Bool := false
  maximumStepSize : Float := 0.002
  publishPeriod : Float := 1.0 / 64.0
  ballHydroelasticModulus : Float := 1.0e8
  edgeLength : Float := 10.0
  cylinderRadius : Float := 0.5
  cylinderLength : Float := 1.0
  deriving Repr, Inhabited

namespace ContactSurfaceVisParams

def targetRealtimeRate (p : ContactSurfaceVisParams) : Float :=
  if p.realTime then 1.0 else 0.0

def useStrictHydro (p : ContactSurfaceVisParams) : Bool :=
  !p.hybrid

def surfaceRepresentation (p : ContactSurfaceVisParams) :
    HydroelasticSurfaceRepresentation :=
  if p.polygons then .polygon else .triangle

private def validatePositiveFinite
    (value : Float) (field : String) : Except String Unit :=
  if !value.isFinite || value <= 0.0 then
    .error s!"simple_contact_surface_vis {field} must be positive and finite, got {value}"
  else
    .ok ()

def validate? (p : ContactSurfaceVisParams) : Except String Unit := do
  validatePositiveFinite p.simulationTime "simulation_time"
  validatePositiveFinite p.length "length"
  validatePositiveFinite p.maximumStepSize "maximum_step_size"
  validatePositiveFinite p.publishPeriod "publish_period"
  validatePositiveFinite p.ballHydroelasticModulus "ball_hydroelastic_modulus"
  validatePositiveFinite p.edgeLength "edge_length"
  validatePositiveFinite p.cylinderRadius "cylinder_radius"
  validatePositiveFinite p.cylinderLength "cylinder_length"

end ContactSurfaceVisParams

def contactSurfaceVisParams : ContactSurfaceVisParams := {}

structure ContactSurfaceMovingBallState where
  z : Float := 0.0
  zdot : Float := 0.0
  deriving Repr, BEq, Inhabited

def contactSurfaceMovingBallDerivative (time : Float) : Array Float :=
  #[Float.sin time, 0.0]

def contactSurfaceMovingBallPose
    (state : ContactSurfaceMovingBallState := {}) : SceneFramePose :=
  {
    frameId := contactSurfaceMovingBallFrameId
    X_WF := {
      translation := { z := state.z }
      rotationAxis := SceneVec3.unitZ
      rotationAngle := 0.0
    }
  }

private def contactSurfaceMovingBallProperties
    (p : ContactSurfaceVisParams) : SceneGeometryProperties :=
  {
    roles := #[.proximity, .illustration]
    diffuseRgba? := some { r := 0.1, g := 0.8, b := 0.1, a := 0.25 }
    hydroelastic? := some (.compliant p.length p.ballHydroelasticModulus)
  }

private def contactSurfaceRigidBoxProperties
    (p : ContactSurfaceVisParams) : SceneGeometryProperties :=
  {
    roles := #[.proximity, .illustration]
    diffuseRgba? := some { r := 0.5, g := 0.5, b := 0.45, a := 1.0 }
    hydroelastic? := some (.rigid p.edgeLength)
  }

private def contactSurfaceCylinderProperties
    (p : ContactSurfaceVisParams) : SceneGeometryProperties :=
  {
    roles := #[.proximity, .illustration]
    diffuseRgba? := some { r := 0.5, g := 0.5, b := 0.45, a := 0.5 }
    hydroelastic? :=
      if p.rigidCylinders then
        some (.rigid p.cylinderRadius)
      else
        none
  }

def contactSurfaceVisSceneGraph
    (p : ContactSurfaceVisParams := contactSurfaceVisParams) : SceneGraphProvider :=
  let boxZ := -((Float.sqrt 2.0) * p.edgeLength / 2.0)
  {
    label := "scene_graph_simple_contact_surface_vis"
    sources := #[
      { id := contactSurfaceMovingBallSourceId, name := "moving_ball" },
      { id := contactSurfaceWorldSourceId, name := "world" }
    ]
    frames := #[
      {
        id := contactSurfaceMovingBallFrameId
        sourceId := contactSurfaceMovingBallSourceId
        name := "moving_frame"
        frameGroup := 1
      },
      {
        id := contactSurfaceCanFrameId
        sourceId := contactSurfaceWorldSourceId
        name := "double_can"
        frameGroup := 2
      }
    ]
    geometries := #[
      {
        id := contactSurfaceBallGeometryId
        sourceId := contactSurfaceMovingBallSourceId
        frameId? := some contactSurfaceMovingBallFrameId
        X_FG := .identity
        shape := .sphere 1.0
        name := "ball"
        properties := contactSurfaceMovingBallProperties p
      },
      {
        id := contactSurfaceBoxGeometryId
        sourceId := contactSurfaceWorldSourceId
        frameId? := none
        X_FG :=
          ScenePose3.fromAxisAngle { z := boxZ } SceneVec3.unitX (pi / 4.0)
        shape := .box p.edgeLength p.edgeLength p.edgeLength
        name := "box"
        properties := contactSurfaceRigidBoxProperties p
      },
      {
        id := contactSurfaceCan1GeometryId
        sourceId := contactSurfaceWorldSourceId
        frameId? := some contactSurfaceCanFrameId
        X_FG := { translation := { x := -0.5, z := 3.0 } }
        shape := .cylinder p.cylinderRadius p.cylinderLength
        name := "can1"
        properties := contactSurfaceCylinderProperties p
      },
      {
        id := contactSurfaceCan2GeometryId
        sourceId := contactSurfaceWorldSourceId
        frameId? := some contactSurfaceCanFrameId
        X_FG := { translation := { x := 0.5, z := 3.0 } }
        shape := .cylinder p.cylinderRadius p.cylinderLength
        name := "can2"
        properties := contactSurfaceCylinderProperties p
      }
    ]
  }

private def contactSurfaceBodyName
    (p : ContactSurfaceVisParams) (frameName geometryName : String) : String :=
  if p.forceFullName then
    s!"{frameName}::{geometryName}"
  else
    frameName

private def contactSurfacePatch
    (p : ContactSurfaceVisParams)
    (id : Nat)
    (bodyB geometryName label : String)
    (centroid normal : Array Float)
    (forceIndex : Nat) : HydroelasticContactPatch :=
  let normalForce := 1.2 * (Float.ofNat (forceIndex + 1))
  {
    id := id
    bodyA := contactSurfaceBodyName p "MovingBall" "ball"
    bodyB := contactSurfaceBodyName p bodyB geometryName
    complianceA := .compliant
    complianceB := .rigid
    representation := p.surfaceRepresentation
    area := 1.0
    centroid := centroid
    normal := normal
    averagePressure := normalForce
    normalVelocity := 0.0
    tangentVelocity := 0.2 + 0.005 * (Float.ofNat forceIndex)
    tangentVelocity2 := 0.0
    normalJacobian := #[1.0, 0.0]
    tangentJacobian := #[0.0, 1.0]
    tangentJacobian2 := #[0.0, 0.0]
    label := label
  }

private def contactSurfaceHydroelasticPatches?
    (p : ContactSurfaceVisParams) : Except String (Array HydroelasticContactPatch) := do
  if p.useStrictHydro && !p.rigidCylinders then
    .error "simple_contact_surface_vis strict hydroelastic requires rigid cylinder hydroelastic properties"
  let boxPatch :=
    contactSurfacePatch p contactSurfaceBoxPatchId "world" "box" "ball-box"
      #[0.0, 0.0, 1.0] #[0.0, 0.0, 1.0] 0
  let canPatches :=
    if p.rigidCylinders then
      #[
        contactSurfacePatch p contactSurfaceCan1PatchId "FixedCylinders" "can1"
          "ball-can1" #[-0.5, 0.0, 3.0] #[0.0, 0.0, 1.0] 1,
        contactSurfacePatch p contactSurfaceCan2PatchId "FixedCylinders" "can2"
          "ball-can2" #[0.5, 0.0, 3.0] #[0.0, 0.0, 1.0] 2
      ]
    else
      #[]
  pure (#[boxPatch] ++ canPatches)

private def contactSurfaceFallbackPointPairs
    (p : ContactSurfaceVisParams) : Array ScenePointPairPenetration :=
  if !p.useStrictHydro && !p.rigidCylinders then
    #[
      {
        idA := contactSurfaceBallGeometryId
        idB := contactSurfaceCan1GeometryId
        depth := 0.1
        nhatBA_W := SceneVec3.unitZ
        p_WCa := { x := -0.5, z := 3.0 }
        p_WCb := { x := -0.5, z := 3.0 - 0.1 }
        label := "ball-can1-point-pair-fallback"
      },
      {
        idA := contactSurfaceBallGeometryId
        idB := contactSurfaceCan2GeometryId
        depth := 0.1
        nhatBA_W := SceneVec3.unitZ
        p_WCa := { x := 0.5, z := 3.0 }
        p_WCb := { x := 0.5, z := 3.0 - 0.1 }
        label := "ball-can2-point-pair-fallback"
      }
    ]
  else
    #[]

private def contactSurfaceCandidateFromPointPair
    (p : ContactSurfaceVisParams) (pair : ScenePointPairPenetration) :
    ContactCandidate :=
  let bodyB :=
    if pair.idB == contactSurfaceCan1GeometryId then
      contactSurfaceBodyName p "FixedCylinders" "can1"
    else if pair.idB == contactSurfaceCan2GeometryId then
      contactSurfaceBodyName p "FixedCylinders" "can2"
    else
      contactSurfaceBodyName p "world" "box"
  {
    id := pair.idA * 100000 + pair.idB
    bodyA := contactSurfaceBodyName p "MovingBall" "ball"
    bodyB := bodyB
    point_W := pair.p_WCa.toArray
    normal_W := pair.nhatBA_W.toArray
    signedDistance := -pair.depth
    normalVelocity := 0.0
    tangentVelocity := 0.0
    tangentVelocity2 := 0.0
    normalJacobian := #[1.0, 0.0]
    tangentJacobian := #[0.0, 1.0]
    tangentJacobian2 := #[0.0, 0.0]
    mode := .sticking
    label := pair.label
  }

private def contactSurfaceCandidateSet
    (p : ContactSurfaceVisParams)
    (patches : Array HydroelasticContactPatch)
    (pointPairs : Array ScenePointPairPenetration) :
    ContactCandidateSet :=
  ContactCandidateSet.ofArray
    ((patches.map (fun patch => patch.equivalentContactCandidate)) ++
      (pointPairs.map (contactSurfaceCandidateFromPointPair p)))
    "simple_contact_surface_vis"

structure ContactSurfaceVisContactResult where
  timestampMicros : Float
  query : SceneContactQueryResult
  publishChannel : String := contactSurfaceLcmChannel
  publishPeriod : Float
  forceFullName : Bool
  deriving Repr, Inhabited

namespace ContactSurfaceVisContactResult

def hydroelasticPatches (result : ContactSurfaceVisContactResult) :
    Array HydroelasticContactPatch :=
  result.query.hydroelasticPatches

def pointPairs (result : ContactSurfaceVisContactResult) :
    Array ScenePointPairPenetration :=
  result.query.pointPairs

def candidateSet (result : ContactSurfaceVisContactResult) : ContactCandidateSet :=
  result.query.candidates

def validate? (result : ContactSurfaceVisContactResult) : Except String Unit := do
  if !result.timestampMicros.isFinite || result.timestampMicros < 0.0 then
    .error s!"simple_contact_surface_vis timestamp must be nonnegative and finite, got {result.timestampMicros}"
  if result.publishChannel.isEmpty then
    .error "simple_contact_surface_vis publish channel cannot be empty"
  if !result.publishPeriod.isFinite || result.publishPeriod <= 0.0 then
    .error s!"simple_contact_surface_vis publish period must be positive and finite, got {result.publishPeriod}"
  result.query.validate? (some 2)

end ContactSurfaceVisContactResult

def contactSurfaceVisContactResult?
    (p : ContactSurfaceVisParams := contactSurfaceVisParams)
    (time : Float := 0.0) : Except String ContactSurfaceVisContactResult := do
  p.validate?
  let patches ← contactSurfaceHydroelasticPatches? p
  let pointPairs := contactSurfaceFallbackPointPairs p
  let query : SceneContactQueryResult := {
    providerLabel := (contactSurfaceVisSceneGraph p).label
    hydroelasticPatches := patches
    pointPairs := pointPairs
    candidates := contactSurfaceCandidateSet p patches pointPairs
    useStrictHydro := p.useStrictHydro
    representation := p.surfaceRepresentation
    label := "simple_contact_surface_vis query"
  }
  let result : ContactSurfaceVisContactResult := {
    timestampMicros := time * 1.0e6
    query := query
    publishChannel := contactSurfaceLcmChannel
    publishPeriod := p.publishPeriod
    forceFullName := p.forceFullName
  }
  result.validate?
  pure result

def contactSurfaceVisGraph
    (p : ContactSurfaceVisParams := contactSurfaceVisParams) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 6200, kind := .state .boundary, label := "../drake/examples/scene_graph/simple_contact_surface_vis.cc flags" }
    |>.addVertex { id := 6201, kind := .state .interior, label := "DiagramBuilder" }
    |>.addVertex { id := 6202, kind := .opaque, label := "SceneGraph<double> scene_graph" }
    |>.addVertex { id := 6203, kind := .state .interior, label := "MovingBall leaf system" }
    |>.addVertex { id := 6204, kind := .state .boundary, label := "world anchored box and double_can frame poses" }
    |>.addVertex { id := 6205, kind := .eventTime, label := "ContactResultMaker query_object input" }
    |>.addVertex { id := 6206, kind := .state .boundary, label := "DrakeVisualizerd sink" }
    |>.addVertex { id := 6207, kind := .state .boundary, label := "LCM CONTACT_RESULTS publisher" }
    |>.addVertex { id := 6208, kind := .interval, label := "Simulator.AdvanceTo simple_contact_surface_vis" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[6201, 6202, 6203, 6204]
      reads := #[6200]
      writes := #[6201, 6202, 6203, 6204]
      label := "DiagramBuilder.AddSystem SceneGraph, MovingBall, anchored box, and double_can cylinders"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[6202, 6204]
      reads := #[6203, 6204]
      writes := #[6202]
      label := "Connect MovingBall.geometry_pose output and fix world/double_can source pose"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[6205]
      reads := #[6202]
      writes := #[6205]
      exactness :=
        if p.useStrictHydro then
          .exact
        else
          .controlledApproximation
      label := "ContactResultMaker computes hydroelastic surfaces or point-pair fallback query result"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[6206, 6207]
      reads := #[6202, 6205]
      writes := #[6206, 6207]
      label := "DrakeVisualizerd::AddToBuilder and LcmPublisherSystem CONTACT_RESULTS"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[6208]
      reads := #[6201, 6202, 6203, 6205, 6207]
      writes := #[6208]
      cost := { work := p.simulationTime / p.maximumStepSize }
      label := "Simulator.Initialize; set_maximum_step_size(0.002); AdvanceTo(FLAGS_simulation_time)"
    }

structure ContactSurfaceVisResult where
  params : ContactSurfaceVisParams
  provider : SceneGraphProvider
  movingBallState : ContactSurfaceMovingBallState
  movingBallDerivative : Array Float
  movingBallPose : SceneFramePose
  contactResult : ContactSurfaceVisContactResult
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildContactSurfaceVis?
    (p : ContactSurfaceVisParams := contactSurfaceVisParams)
    (state : ContactSurfaceMovingBallState := {})
    (time : Float := 0.0) : Except String ContactSurfaceVisResult := do
  p.validate?
  let provider := contactSurfaceVisSceneGraph p
  provider.validate?
  let pose := contactSurfaceMovingBallPose state
  ({ poses := #[pose] } : SceneFramePoseVector).validate? provider
  let contactResult ← contactSurfaceVisContactResult? p time
  let graph := contactSurfaceVisGraph p
  pure {
    params := p
    provider := provider
    movingBallState := state
    movingBallDerivative := contactSurfaceMovingBallDerivative time
    movingBallPose := pose
    contactResult := contactResult
    graph := graph
    moves := graph.moves
  }

structure SceneGraphExampleResult where
  bouncingBall : BouncingBallSceneResult
  solarSystem : SolarSystemResult
  solarRunDynamics : SolarRunDynamicsResult
  contactSurfaceVis : ContactSurfaceVisResult
  deriving Repr, Inhabited

def buildEndToEnd? : Except String SceneGraphExampleResult := do
  let bouncingBall ← buildBouncingBall?
  let solarSystem ← buildSolarSystem?
  let solarRunDynamics ← buildSolarRunDynamics?
  let contactSurfaceVis ← buildContactSurfaceVis?
  pure {
    bouncingBall := bouncingBall
    solarSystem := solarSystem
    solarRunDynamics := solarRunDynamics
    contactSurfaceVis := contactSurfaceVis
  }

end Tyr.EventSkeleton.Examples.SceneGraph
