import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Contact
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Physics
import Tyr.EventSkeleton.SceneGraph

/-!
# Drake Rod2D Event-Skeleton Example

This is a compact, executable port of the core physics in
`../drake/examples/rod2d`: endpoint geometry, generalized inertia, compliant
half-space contact, impact detection, and velocity-level contact projection.

It also carries the acceleration-level sustained-contact path from Drake's
`constraint_problem_data` and `constraint_solver`: a dynamically selected
contact support is converted into an MLCP/LCP-style contact problem, solved by
the dense complementarity primitive, and mapped back through `J^T` into
generalized acceleration.
-/

namespace Tyr.EventSkeleton.Examples.Rod2D

open Tyr.EventSkeleton
open torch.DiffEq

private def pi : Float := 3.14159265358979323846
private def sqrt2Over2 : Float := 0.70710678118654752440

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/rod2d/rod2d.cc"
      concept := "defines endpoint contact geometry, compliant contact forces, impact tests, and default Painleve state"
    },
    {
      path := "../drake/examples/rod2d/rod2d.h"
      concept := "declares Rod2D system types, state accessors, parameters, contact geometry, and constraint problem data hooks"
    },
    {
      path := "../drake/examples/rod2d/test/rod2d_test.cc"
      concept := "checks state-vector accessors, parameters, contact problem data, impact, discrete, and continuous Rod2D behavior"
    },
    {
      path := "../drake/examples/rod2d/rod2d_geometry.h"
      concept := "declares the Rod2dGeometry SceneGraph helper with state input, geometry_pose output, and configurable visual radius"
    },
    {
      path := "../drake/examples/rod2d/rod2d_geometry.cc"
      concept := "registers the rod2d source, frame, grey cylinder, and maps 2D state to a 3D frame pose"
    },
    {
      path := "../drake/examples/rod2d/rod2d_sim.cc"
      concept := "wires Rod2dGeometry, SceneGraph, DrakeVisualizer, external force input, and simulator/integrator flags"
    },
    {
      path := "../drake/examples/rod2d/constraint_problem_data.h"
      concept := "packages mass-matrix solves, contact Jacobians, and constraint-space problem data"
    },
    {
      path := "../drake/examples/rod2d/constraint_solver.cc"
      concept := "forms sustained-contact LCP blocks and maps packed contact forces back to generalized acceleration"
    },
    {
      path := "../drake/examples/rod2d/constraint_solver.h"
      concept := "declares ConstraintSolver MLCP-to-LCP conversion, packed force recovery, and contact-frame force conversion"
    },
    {
      path := "../drake/examples/rod2d/test/constraint_solver_test.cc"
      concept := "regresses duplicated contacts, duplicated friction directions, sticking, sliding, impact, limits, and bilateral constraints"
    },
    {
      path := "../drake/examples/rod2d/rod2d_state_vector.h"
      concept := "defines the six-element rod state layout"
    },
    {
      path := "../drake/examples/rod2d/rod2d_state_vector.cc"
      concept := "defines Rod2dStateVectorIndices::GetCoordinateNames"
    }
  ]

def rod2dExampleRoot : String := "../drake/examples/rod2d"

inductive Rod2dExampleAssetKind where
  | bazel
  | source
  | header
  | testSource
  | readme
  | documentationImage
  deriving Repr, BEq, Inhabited

namespace Rod2dExampleAssetKind

def matchesPath : Rod2dExampleAssetKind → String → Bool
  | .bazel, path => path == "BUILD.bazel"
  | .source, path => path.endsWith ".cc" && !(path.startsWith "test/")
  | .header, path => path.endsWith ".h" && !(path.startsWith "test/")
  | .testSource, path => path.startsWith "test/" && path.endsWith ".cc"
  | .readme, path => path == "README.md"
  | .documentationImage, path => path.startsWith "images/" && path.endsWith ".png"

end Rod2dExampleAssetKind

structure Rod2dExampleAsset where
  relativePath : String
  kind : Rod2dExampleAssetKind
  physicsBearing : Bool := true
  concept : String
  deriving Repr, Inhabited

namespace Rod2dExampleAsset

def fullPath (asset : Rod2dExampleAsset) : String :=
  rod2dExampleRoot ++ "/" ++ asset.relativePath

def validate? (asset : Rod2dExampleAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "rod2d asset path cannot be empty"
  if !asset.kind.matchesPath asset.relativePath then
    .error s!"rod2d asset {asset.relativePath}: kind does not match path"
  if asset.concept.isEmpty then
    .error s!"rod2d asset {asset.relativePath}: concept cannot be empty"
  if asset.kind == .documentationImage && asset.physicsBearing then
    .error s!"rod2d documentation image {asset.relativePath} should not feed physics"
  if asset.kind == .readme && asset.physicsBearing then
    .error s!"rod2d README {asset.relativePath} should not feed physics"

end Rod2dExampleAsset

def rod2dExampleAssets : Array Rod2dExampleAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      kind := .bazel
      physicsBearing := false
      concept := "declares rod2d plant, geometry, solver, simulator, and tests"
    },
    {
      relativePath := "README.md"
      kind := .readme
      physicsBearing := false
      concept := "documents the rod2d contact solver and simulation example"
    },
    {
      relativePath := "constraint_problem_data.h"
      kind := .header
      concept := "packages mass-matrix solves and acceleration-level contact data"
    },
    {
      relativePath := "constraint_solver.cc"
      kind := .source
      concept := "implements sustained-contact LCP conversion and force recovery"
    },
    {
      relativePath := "constraint_solver.h"
      kind := .header
      concept := "declares sustained-contact solver entry points"
    },
    {
      relativePath := "images/colliding-boxes.png"
      kind := .documentationImage
      physicsBearing := false
      concept := "README image sidecar showing the colliding-boxes contact-solver setup"
    },
    {
      relativePath := "rod2d.cc"
      kind := .source
      concept := "defines endpoint geometry, compliant contact, and impact behavior"
    },
    {
      relativePath := "rod2d.h"
      kind := .header
      concept := "declares Rod2D system types and parameters"
    },
    {
      relativePath := "rod2d_geometry.cc"
      kind := .source
      concept := "registers SceneGraph geometry and pose output"
    },
    {
      relativePath := "rod2d_geometry.h"
      kind := .header
      concept := "declares Rod2dGeometry helper ports"
    },
    {
      relativePath := "rod2d_sim.cc"
      kind := .source
      concept := "wires Rod2D, Rod2dGeometry, SceneGraph, visualizer, and simulator"
    },
    {
      relativePath := "rod2d_state_vector.cc"
      kind := .source
      concept := "defines coordinate names for the six-element state vector"
    },
    {
      relativePath := "rod2d_state_vector.h"
      kind := .header
      concept := "declares the six-element state vector layout"
    },
    {
      relativePath := "test/constraint_solver_test.cc"
      kind := .testSource
      concept := "regresses sustained-contact solver edge cases"
    },
    {
      relativePath := "test/rod2d_test.cc"
      kind := .testSource
      concept := "regresses Rod2D state, parameters, impact, and continuous behavior"
    }
  ]

def rod2dExampleAssetPaths : Array String :=
  rod2dExampleAssets.map (fun asset => asset.relativePath)

def rod2dDocumentationAssets : Array Rod2dExampleAsset :=
  rod2dExampleAssets.filter (fun asset =>
    asset.kind == .readme || asset.kind == .documentationImage)

def findRod2dExampleAsset? (relativePath : String) : Option Rod2dExampleAsset :=
  rod2dExampleAssets.find? (fun asset => asset.relativePath == relativePath)

private def hasDuplicateRod2dAssetPath (assets : Array Rod2dExampleAsset) : Bool :=
  Id.run do
    let mut seen : Array String := #[]
    for asset in assets do
      if seen.contains asset.relativePath then
        return true
      seen := seen.push asset.relativePath
    return false

def validateRod2dExampleAssets? : Except String Unit := do
  if rod2dExampleAssets.size != 15 then
    .error s!"rod2d asset catalog should contain 15 files, got {rod2dExampleAssets.size}"
  if hasDuplicateRod2dAssetPath rod2dExampleAssets then
    .error "rod2d asset catalog contains duplicate paths"
  for asset in rod2dExampleAssets do
    asset.validate?
  for path in #[
      "BUILD.bazel",
      "README.md",
      "images/colliding-boxes.png",
      "rod2d.cc",
      "rod2d_geometry.cc",
      "rod2d_sim.cc",
      "constraint_solver.cc",
      "test/constraint_solver_test.cc",
      "test/rod2d_test.cc"] do
    if !(rod2dExampleAssetPaths.contains path) then
      .error s!"rod2d asset catalog is missing {path}"
  if rod2dDocumentationAssets.size != 2 then
    .error s!"rod2d documentation asset count should be 2, got {rod2dDocumentationAssets.size}"

def stateCoordinateNames : Array String :=
  #["x", "y", "theta", "xdot", "ydot", "thetadot"]

structure Rod2dStateVectorBoundary where
  typeName : String := "Rod2dStateVector"
  headerPath : String := "../drake/examples/rod2d/rod2d_state_vector.h"
  implementationPath : String := "../drake/examples/rod2d/rod2d_state_vector.cc"
  coordinateNames : Array String := stateCoordinateNames
  defaults : Array Float := #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  movedFromAccessThrows : Bool := true
  supportsNamedVariables : Bool := true
  deriving Repr, Inhabited

namespace Rod2dStateVectorBoundary

def dimension (boundary : Rod2dStateVectorBoundary) : Nat :=
  boundary.coordinateNames.size

def indexOf? (boundary : Rod2dStateVectorBoundary) (name : String) : Option Nat :=
  boundary.coordinateNames.findIdx? (fun candidate => candidate == name)

def validate? (boundary : Rod2dStateVectorBoundary) : Except String Unit := do
  if boundary.typeName != "Rod2dStateVector" then
    .error s!"Rod2d state vector type mismatch: {boundary.typeName}"
  if boundary.headerPath != "../drake/examples/rod2d/rod2d_state_vector.h" then
    .error s!"Rod2d state vector header mismatch: {boundary.headerPath}"
  if boundary.implementationPath != "../drake/examples/rod2d/rod2d_state_vector.cc" then
    .error s!"Rod2d state vector implementation mismatch: {boundary.implementationPath}"
  if boundary.coordinateNames != stateCoordinateNames then
    .error s!"Rod2d state coordinate names mismatch: {boundary.coordinateNames}"
  if boundary.defaults.size != boundary.dimension then
    .error s!"Rod2d state defaults have size {boundary.defaults.size}, expected {boundary.dimension}"
  if !boundary.defaults.all (fun x => x.isFinite) then
    .error s!"Rod2d state defaults must be finite, got {boundary.defaults}"

end Rod2dStateVectorBoundary

def rod2dStateVectorBoundary : Rod2dStateVectorBoundary := {}

structure RodParams where
  mass : Float := 1.0
  halfLength : Float := 1.0
  momentInertia : Float := 1.0
  muCoulomb : Float := 1000.0
  gravity : Float := -9.81
  stiffness : Float := 10000.0
  dissipation : Float := 1.0
  muStatic : Float := 1000.0
  stictionSpeedTolerance : Float := 1.0e-5
  contactDistanceTol : Float := 1.0e-6
  rootTol : Float := 1.0e-8
  stepSize : Float := 1.0e-3
  deriving Repr, Inhabited

def params : RodParams := {}

namespace RodParams

def validate? (p : RodParams) : Except String Unit := do
  if !p.mass.isFinite || p.mass <= 0.0 then
    .error s!"Rod2D mass must be positive and finite, got {p.mass}"
  if !p.halfLength.isFinite || p.halfLength <= 0.0 then
    .error s!"Rod2D half length must be positive and finite, got {p.halfLength}"
  if !p.momentInertia.isFinite || p.momentInertia <= 0.0 then
    .error s!"Rod2D moment inertia must be positive and finite, got {p.momentInertia}"
  if !p.muCoulomb.isFinite || p.muCoulomb < 0.0 then
    .error s!"Rod2D Coulomb friction must be finite and nonnegative, got {p.muCoulomb}"
  if !p.gravity.isFinite then
    .error s!"Rod2D gravity must be finite, got {p.gravity}"
  if !p.stiffness.isFinite || p.stiffness < 0.0 then
    .error s!"Rod2D contact stiffness must be finite and nonnegative, got {p.stiffness}"
  if !p.dissipation.isFinite || p.dissipation < 0.0 then
    .error s!"Rod2D contact dissipation must be finite and nonnegative, got {p.dissipation}"
  if !p.muStatic.isFinite || p.muStatic < 0.0 then
    .error s!"Rod2D static friction must be finite and nonnegative, got {p.muStatic}"
  if !p.stictionSpeedTolerance.isFinite || p.stictionSpeedTolerance <= 0.0 then
    .error s!"Rod2D stiction speed tolerance must be positive and finite, got {p.stictionSpeedTolerance}"
  if !p.contactDistanceTol.isFinite || p.contactDistanceTol < 0.0 then
    .error s!"Rod2D contact distance tolerance must be finite and nonnegative, got {p.contactDistanceTol}"
  if !p.rootTol.isFinite || p.rootTol <= 0.0 then
    .error s!"Rod2D root tolerance must be positive and finite, got {p.rootTol}"
  if !p.stepSize.isFinite || p.stepSize <= 0.0 then
    .error s!"Rod2D step size must be positive and finite, got {p.stepSize}"

end RodParams

structure RodState where
  x : Float
  y : Float
  theta : Float
  xdot : Float
  ydot : Float
  thetadot : Float
  deriving Repr, Inhabited

namespace RodState

def isFinite (x : RodState) : Bool :=
  x.x.isFinite && x.y.isFinite && x.theta.isFinite &&
    x.xdot.isFinite && x.ydot.isFinite && x.thetadot.isFinite

end RodState

instance : DiffEqSpace RodState where
  add a b := {
    x := a.x + b.x
    y := a.y + b.y
    theta := a.theta + b.theta
    xdot := a.xdot + b.xdot
    ydot := a.ydot + b.ydot
    thetadot := a.thetadot + b.thetadot
  }
  sub a b := {
    x := a.x - b.x
    y := a.y - b.y
    theta := a.theta - b.theta
    xdot := a.xdot - b.xdot
    ydot := a.ydot - b.ydot
    thetadot := a.thetadot - b.thetadot
  }
  scale s a := {
    x := s * a.x
    y := s * a.y
    theta := s * a.theta
    xdot := s * a.xdot
    ydot := s * a.ydot
    thetadot := s * a.thetadot
  }

private def max6 (a b c d e f : Float) : Float :=
  max a (max b (max c (max d (max e f))))

instance : DiffEqSeminorm RodState where
  rms x :=
    max6
      (Float.abs x.x)
      (Float.abs x.y)
      (Float.abs x.theta)
      (Float.abs x.xdot)
      (Float.abs x.ydot)
      (Float.abs x.thetadot)

instance : DiffEqElem RodState where
  abs x := {
    x := Float.abs x.x
    y := Float.abs x.y
    theta := Float.abs x.theta
    xdot := Float.abs x.xdot
    ydot := Float.abs x.ydot
    thetadot := Float.abs x.thetadot
  }
  max a b := {
    x := max a.x b.x
    y := max a.y b.y
    theta := max a.theta b.theta
    xdot := max a.xdot b.xdot
    ydot := max a.ydot b.ydot
    thetadot := max a.thetadot b.thetadot
  }
  addScalar s x := {
    x := x.x + s
    y := x.y + s
    theta := x.theta + s
    xdot := x.xdot + s
    ydot := x.ydot + s
    thetadot := x.thetadot + s
  }
  div a b := {
    x := a.x / b.x
    y := a.y / b.y
    theta := a.theta / b.theta
    xdot := a.xdot / b.xdot
    ydot := a.ydot / b.ydot
    thetadot := a.thetadot / b.thetadot
  }

def stateAsArray (x : RodState) : Array Float :=
  #[x.x, x.y, x.theta, x.xdot, x.ydot, x.thetadot]

def stateFromArray? (xs : Array Float) : Except String RodState := do
  if xs.size != 6 then
    .error s!"Rod2dStateVector expects 6 coordinates, got {xs.size}"
  if !xs.all (fun x => x.isFinite) then
    .error s!"Rod2dStateVector values must be finite, got {xs}"
  pure {
    x := xs[0]!
    y := xs[1]!
    theta := xs[2]!
    xdot := xs[3]!
    ydot := xs[4]!
    thetadot := xs[5]!
  }

def velocityAsArray (x : RodState) : Array Float :=
  #[x.xdot, x.ydot, x.thetadot]

def stateWithVelocityArray (x : RodState) (v : Array Float) : RodState :=
  {
    x with
    xdot := v.getD 0 0.0
    ydot := v.getD 1 0.0
    thetadot := v.getD 2 0.0
  }

private def arraySlice (xs : Array Float) (start len : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := #[]
    for i in [:len] do
      out := out.push (xs.getD (start + i) 0.0)
    return out

private def nonnegativeAll (xs : Array Float) (tol : Float := 0.0) : Bool :=
  xs.all (fun x => x >= -tol)

def defaultState (p : RodParams := params) : RodState :=
  {
    x := p.halfLength * sqrt2Over2
    y := p.halfLength * sqrt2Over2
    theta := pi / 4.0
    xdot := -1.0
    ydot := 0.0
    thetadot := 0.0
  }

def fallingState : RodState :=
  {
    x := 0.0
    y := 1.0
    theta := pi / 6.0
    xdot := 0.25
    ydot := -1.0
    thetadot := 0.0
  }

def restingVerticalState (p : RodParams := params) : RodState :=
  {
    x := 0.0
    y := p.halfLength
    theta := pi / 2.0
    xdot := 0.0
    ydot := 0.0
    thetadot := 0.0
  }

/-! ## Rod2dGeometry SceneGraph provider -/

def rod2dDefaultVisualRadius : Float := 5.0e-2

def rod2dGeometrySourceId : Nat := 4300
def rod2dGeometryFrameId : Nat := 4301
def rod2dGeometryId : Nat := 4302

def rod2dGeometryStateInputVertex : VertexId := 4310
def rod2dGeometryProviderVertex : VertexId := 4311
def rod2dGeometryPoseOutputVertex : VertexId := 4312

private def rod2dIllustrationProperties : SceneGeometryProperties :=
  { roles := #[.illustration], diffuseRgba? := some { r := 0.7, g := 0.7, b := 0.7, a := 1.0 } }

def rod2dGeometryProvider
    (p : RodParams := params) (radius : Float := rod2dDefaultVisualRadius) :
    SceneGraphProvider :=
  {
    sources := #[
      { id := rod2dGeometrySourceId, name := "rod2d" }
    ]
    frames := #[
      {
        id := rod2dGeometryFrameId
        sourceId := rod2dGeometrySourceId
        name := "rod2d"
      }
    ]
    geometries := #[
      {
        id := rod2dGeometryId
        sourceId := rod2dGeometrySourceId
        frameId? := some rod2dGeometryFrameId
        X_FG := ScenePose3.identity
        shape := .cylinder radius (2.0 * p.halfLength)
        name := "rod2d"
        properties := rod2dIllustrationProperties
      }
    ]
    label := "Rod2dGeometry SceneGraph provider"
  }

private def rod2dStateFinite (x : Array Float) : Bool :=
  x.all (fun xi => xi.isFinite)

def rod2dGeometryPoseOutput (state : Array Float) : SceneFramePoseVector :=
  {
    poses := #[
      {
        frameId := rod2dGeometryFrameId
        X_WF := {
          translation := {
            x := state.getD 0 0.0
            y := 0.0
            z := state.getD 1 0.0
          }
          rotationAxis := SceneVec3.unitY
          rotationAngle := state.getD 2 0.0 + pi / 2.0
        }
      }
    ]
  }

private def rod2dGeometryMove
    (target : VertexId) (label : String) (reads : Array VertexId := #[])
    (writes : Array VertexId := #[]) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target]
    reads := reads
    writes := writes
    exactness := .exact
    label := label
  }

def rod2dGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := rod2dGeometryStateInputVertex
      kind := .state .boundary
      label := "Rod2dGeometry state input"
    }
    |>.addVertex {
      id := rod2dGeometryProviderVertex
      kind := .state .boundary
      label := "Rod2dGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := rod2dGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "Rod2dGeometry geometry_pose output"
    }
    |>.addMove (rod2dGeometryMove rod2dGeometryProviderVertex
      "Register rod2d source, body frame, and grey cylinder geometry"
      #[] #[rod2dGeometryProviderVertex])
    |>.addMove (rod2dGeometryMove rod2dGeometryPoseOutputVertex
      "OutputGeometryPose: 2D state -> rod2d FramePoseVector"
      #[rod2dGeometryStateInputVertex, rod2dGeometryProviderVertex]
      #[rod2dGeometryPoseOutputVertex])

structure Rod2dGeometryResult where
  references : Array DrakeReference
  params : RodParams
  radius : Float
  inputPortName : String := "state"
  inputPortSize : Nat := 6
  outputPortName : String := "geometry_pose"
  provider : SceneGraphProvider
  sampleState : Array Float
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildRod2dGeometry?
    (p : RodParams := params)
    (x : RodState := defaultState p)
    (radius : Float := rod2dDefaultVisualRadius) :
    Except String Rod2dGeometryResult := do
  if !p.halfLength.isFinite || p.halfLength <= 0.0 then
    .error "Rod2dGeometry requires positive finite rod half-length"
  if !radius.isFinite || radius <= 0.0 then
    .error s!"Rod2dGeometry requires positive finite visual radius, got {radius}"
  let state := stateAsArray x
  if state.size != 6 then
    .error s!"Rod2dGeometry state input must have size 6, got {state.size}"
  if !rod2dStateFinite state then
    .error "Rod2dGeometry state input must be finite"
  let provider := rod2dGeometryProvider p radius
  provider.validate?
  let poses := rod2dGeometryPoseOutput state
  poses.validate? provider
  pure {
    references := drakeReferences
    params := p
    radius := radius
    provider := provider
    sampleState := state
    poses := poses
    graph := rod2dGeometryGraph
    moves := rod2dGeometryGraph.moves
  }

inductive Endpoint where
  | left
  | right
  deriving Repr, BEq, Inhabited

namespace Endpoint

def k : Endpoint → Int
  | .left => -1
  | .right => 1

def kFloat (endpoint : Endpoint) : Float :=
  endpoint.k.toNat.toFloat - (if endpoint == .left then 1.0 else 0.0)

def id : Endpoint → Nat
  | .left => 410
  | .right => 411

def label : Endpoint → String
  | .left => "left_endpoint"
  | .right => "right_endpoint"

end Endpoint

def endpointSign : Endpoint → Float
  | .left => -1.0
  | .right => 1.0

structure EndpointKinematics where
  endpoint : Endpoint
  px : Float
  py : Float
  rx : Float
  ry : Float
  vx : Float
  vy : Float
  deriving Repr, Inhabited

def endpointKinematics (p : RodParams) (x : RodState) (endpoint : Endpoint) :
    EndpointKinematics :=
  let k := endpointSign endpoint
  let c := Float.cos x.theta
  let s := Float.sin x.theta
  let rx := k * c * p.halfLength
  let ry := k * s * p.halfLength
  {
    endpoint := endpoint
    px := x.x + rx
    py := x.y + ry
    rx := rx
    ry := ry
    vx := x.xdot - x.thetadot * ry
    vy := x.ydot + x.thetadot * rx
  }

def normalJacobian (kin : EndpointKinematics) : Array Float :=
  #[0.0, 1.0, kin.rx]

def tangentJacobian (kin : EndpointKinematics) : Array Float :=
  #[1.0, 0.0, -kin.ry]

def normalJacobianDotTimesVelocity (x : RodState) (candidate : ContactCandidate) :
    Float :=
  let ry := -(candidate.tangentJacobian.getD 2 0.0)
  (0.0 - ry) * x.thetadot * x.thetadot

def tangentJacobianDotTimesVelocity (x : RodState) (candidate : ContactCandidate) :
    Float :=
  let rx := candidate.normalJacobian.getD 2 0.0
  (0.0 - rx) * x.thetadot * x.thetadot

def slidingDirectionJacobian (candidate : ContactCandidate) : Array Float :=
  let sign := if candidate.tangentVelocity > 0.0 then 1.0 else -1.0
  FloatArray.scale sign candidate.tangentJacobian

def candidateForEndpoint (p : RodParams) (x : RodState) (endpoint : Endpoint) :
    ContactCandidate :=
  let kin := endpointKinematics p x endpoint
  {
    id := endpoint.id
    signedDistance := kin.py
    normalVelocity := kin.vy
    tangentVelocity := kin.vx
    normalJacobian := normalJacobian kin
    tangentJacobian := tangentJacobian kin
    label := endpoint.label
  }

def contactCandidates (p : RodParams := params) (x : RodState := defaultState p) :
    Array ContactCandidate :=
  #[candidateForEndpoint p x .left, candidateForEndpoint p x .right]

def selectedSupport (p : RodParams) (x : RodState) : ContactSupport :=
  ContactSupport.selectByDistance p.contactDistanceTol (contactCandidates p x)
    "rod2d endpoint support"
    |>.classifyCandidates p.contactDistanceTol p.stictionSpeedTolerance

def minimumSignedDistance (p : RodParams) (x : RodState) : Float :=
  let left := (candidateForEndpoint p x .left).signedDistance
  let right := (candidateForEndpoint p x .right).signedDistance
  if left < right then left else right

def lowerEndpoint (x : RodState) : Endpoint :=
  let s := Float.sin x.theta
  if s > 0.0 then .left else .right

def isImpacting (p : RodParams) (x : RodState) : Bool :=
  let candidate := candidateForEndpoint p x (lowerEndpoint x)
  candidate.signedDistance <= 10.0e-15 && candidate.normalVelocity < -10.0e-15

def massMatrix (p : RodParams) : Array (Array Float) :=
  #[
    #[p.mass, 0.0, 0.0],
    #[0.0, p.mass, 0.0],
    #[0.0, 0.0, p.momentInertia]
  ]

private def positivePart (x : Float) : Float :=
  if x > 0.0 then x else 0.0

private def signForDrakeFriction (x : Float) : Float :=
  if x < 0.0 then -1.0 else 1.0

def step5 (x : Float) : Float :=
  let x3 := x * x * x
  x3 * (10.0 + x * (6.0 * x - 15.0))

def muStribeck (muStatic muDynamic s : Float) : Float :=
  if s >= 3.0 then
    muDynamic
  else if s >= 1.0 then
    muStatic - (muStatic - muDynamic) * step5 ((s - 1.0) / 2.0)
  else
    muStatic * step5 s

structure SpatialForce2D where
  fx : Float := 0.0
  fy : Float := 0.0
  tau : Float := 0.0
  deriving Repr, Inhabited

namespace SpatialForce2D

def add (a b : SpatialForce2D) : SpatialForce2D :=
  {
    fx := a.fx + b.fx
    fy := a.fy + b.fy
    tau := a.tau + b.tau
  }

def isFinite (force : SpatialForce2D) : Bool :=
  force.fx.isFinite && force.fy.isFinite && force.tau.isFinite

end SpatialForce2D

structure RodPhysicsState where
  state : RodState := defaultState params
  applied : SpatialForce2D := {}
  deriving Repr, Inhabited

def physicsState
    (state : RodState := defaultState params)
    (applied : SpatialForce2D := {}) : RodPhysicsState :=
  { state := state, applied := applied }

structure ContactForce2D where
  candidateId : Nat
  normalForce : Float
  tangentForce : Float
  torque : Float
  mode : ContactMode
  deriving Repr, Inhabited

namespace ContactForce2D

def toScalars (force : ContactForce2D) : ContactForceScalars :=
  {
    candidateId := force.candidateId
    normalForce := force.normalForce
    tangentForce := force.tangentForce
    tangentForce2 := 0.0
    mode := force.mode
  }

end ContactForce2D

def compliantForceForCandidate (p : RodParams) (candidate : ContactCandidate) :
    ContactForce2D :=
  let h := -candidate.signedDistance
  if h <= 0.0 then
    {
      candidateId := candidate.id
      normalForce := 0.0
      tangentForce := 0.0
      torque := 0.0
      mode := candidate.mode
    }
  else
    let hdot := -candidate.normalVelocity
    let fK := p.stiffness * h
    let fD := fK * p.dissipation * hdot
    let fN := positivePart (fK + fD)
    let mu := muStribeck p.muStatic p.muCoulomb
      (Float.abs candidate.tangentVelocity / p.stictionSpeedTolerance)
    let fF := -mu * fN * signForDrakeFriction candidate.tangentVelocity
    let rx := candidate.normalJacobian.getD 2 0.0
    let ry := -candidate.tangentJacobian.getD 2 0.0
    {
      candidateId := candidate.id
      normalForce := fN
      tangentForce := fF
      torque := rx * fN - ry * fF
      mode := candidate.mode
    }

def compliantForces (p : RodParams) (x : RodState) : Array ContactForce2D :=
  (contactCandidates p x
    |>.map (fun c => c.withClassifiedMode p.contactDistanceTol p.stictionSpeedTolerance))
    |>.map (compliantForceForCandidate p)

def aggregateContactForce (forces : Array ContactForce2D) : SpatialForce2D :=
  forces.foldl
    (fun acc force =>
      acc.add {
        fx := force.tangentForce
        fy := force.normalForce
        tau := force.torque
      })
    {}

def externalForce (p : RodParams) (applied : SpatialForce2D := {}) : SpatialForce2D :=
  {
    fx := applied.fx
    fy := p.mass * p.gravity + applied.fy
    tau := applied.tau
  }

def spatialForceAsArray (force : SpatialForce2D) : Array Float :=
  #[force.fx, force.fy, force.tau]

def rod2dFullPhysicsIntervalVertex : VertexId := 4320

def gravityBiasForFullPhysics (p : RodParams) : Array Float :=
  #[0.0, -(p.mass * p.gravity), 0.0]

def validateFullPhysicsInputs?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {}) :
    Except String Unit := do
  p.validate?
  if !x.isFinite then
    .error "Rod2D full physics state must have finite coordinates"
  if !applied.isFinite then
    .error "Rod2D full physics applied force must be finite"

def fullPhysicsPrimitives?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {})
    (label : String := "rod2d compliant full physics primitive") :
    Except String FullPhysicsPrimitives := do
  validateFullPhysicsInputs? p x applied
  let support := selectedSupport p x
  let selected ← support.selectedCandidates?
  let forces := selected.map (fun candidate =>
    (compliantForceForCandidate p candidate).toScalars)
  pure ({
    massMatrix := massMatrix p
    qdot := velocityAsArray x
    actuationForces := spatialForceAsArray applied
    biasForces := gravityBiasForFullPhysics p
    contactCandidates := support.candidates
    supportPolicy := .threshold p.contactDistanceTol
    contactForceSource := .precomputed
    contactForces := forces
    distanceTol := p.contactDistanceTol
    tangentVelocityTol := p.stictionSpeedTolerance
    label := label
  } : FullPhysicsPrimitives)

def fullPhysicsEquation?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {}) :
    Except String FullPhysicsEquation := do
  let primitives ← fullPhysicsPrimitives? p x applied
  primitives.equation?

def fullPhysicsPrimitiveProvider
    (p : RodParams := params)
    (label : String := "rod2d compliant full physics provider") :
    FullPhysicsPrimitiveProvider RodPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot =>
      fullPhysicsPrimitives? p snapshot.state snapshot.applied label
  }

def solveFullPhysics?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {})
    (intervalVertex : VertexId := rod2dFullPhysicsIntervalVertex) :
    Except String FullPhysicsResult := do
  let equation ← fullPhysicsEquation? p x applied
  equation.solve? intervalVertex

def continuousDerivativeFromFullPhysics?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {}) :
    Except String RodState := do
  let result ← solveFullPhysics? p x applied
  pure {
    x := x.xdot
    y := x.ydot
    theta := x.thetadot
    xdot := result.derivative.vdot.getD 0 0.0
    ydot := result.derivative.vdot.getD 1 0.0
    thetadot := result.derivative.vdot.getD 2 0.0
  }

structure RodConstraintAccelProblemData where
  support : ContactSupport
  runtimeSupport : RuntimeSupport
  massMatrix : Array (Array Float)
  normalJacobian : Array (Array Float)
  tangentJacobian : Array (Array Float)
  normalMinusMuQJacobian : Array (Array Float)
  slidingContacts : Array Nat := #[]
  nonSlidingContacts : Array Nat := #[]
  r : Array Nat := #[]
  muSliding : Array Float := #[]
  muNonSliding : Array Float := #[]
  kN : Array Float := #[]
  kF : Array Float := #[]
  tau : Array Float := #[]
  useComplementarityProblemSolver : Bool := true
  deriving Repr, Inhabited

namespace RodConstraintAccelProblemData

def velocityDim (data : RodConstraintAccelProblemData) : Nat :=
  FloatMatrix.colCount data.massMatrix

def contactCount (data : RodConstraintAccelProblemData) : Nat :=
  data.normalJacobian.size

def frictionDirectionCount (data : RodConstraintAccelProblemData) : Nat :=
  data.tangentJacobian.size

def validate? (data : RodConstraintAccelProblemData) : Except String Unit := do
  DenseLinearAlgebra.validateSquare? data.massMatrix data.tau.size "rod2d mass matrix"
  data.support.validateJacobianWidth? data.tau.size
  if data.normalJacobian.size != data.kN.size then
    .error s!"rod2d sustained contact: N rows {data.normalJacobian.size} != kN size {data.kN.size}"
  if data.normalMinusMuQJacobian.size != data.normalJacobian.size then
    .error s!"rod2d sustained contact: N-muQ rows {data.normalMinusMuQJacobian.size} != N rows {data.normalJacobian.size}"
  if data.tangentJacobian.size != data.kF.size then
    .error s!"rod2d sustained contact: F rows {data.tangentJacobian.size} != kF size {data.kF.size}"
  if data.r.size != data.nonSlidingContacts.size then
    .error s!"rod2d sustained contact: r size {data.r.size} != non-sliding contact count {data.nonSlidingContacts.size}"
  if data.muSliding.size != data.slidingContacts.size then
    .error s!"rod2d sustained contact: sliding mu size {data.muSliding.size} != sliding contact count {data.slidingContacts.size}"
  if data.muNonSliding.size != data.nonSlidingContacts.size then
    .error s!"rod2d sustained contact: non-sliding mu size {data.muNonSliding.size} != non-sliding contact count {data.nonSlidingContacts.size}"

private def solveMass? (data : RodConstraintAccelProblemData) (rhs : Array Float) :
    Except String (Array Float) :=
  DenseLinearAlgebra.solveLinear? data.massMatrix rhs

def freeAcceleration? (data : RodConstraintAccelProblemData) :
    Except String (Array Float) :=
  data.solveMass? data.tau

private def compliance?
    (data : RodConstraintAccelProblemData)
    (lhs rhs : Array (Array Float)) : Except String (Array (Array Float)) := do
  let minvRhsT ← VelocityProjection.massInverseTimesJacobianTranspose?
    data.massMatrix rhs
  pure (FloatMatrix.matMat lhs minvRhsT)

private def frictionConeE (data : RodConstraintAccelProblemData) :
    Array (Array Float) := Id.run do
  let nns := data.nonSlidingContacts.size
  let mut rows : Array (Array Float) := #[]
  for i in [:nns] do
    let reps := data.r.getD i 0
    for _ in [:reps] do
      let mut row := Array.replicate nns 0.0
      row := row.set! i 1.0
      rows := rows.push row
  return rows

private def matGet (m : Array (Array Float)) (i j : Nat) : Float :=
  (m.getD i #[]).getD j 0.0

private def sustainedLcpMatrix? (data : RodConstraintAccelProblemData) :
    Except String (Array (Array Float)) := do
  let nc := data.contactCount
  let nr := data.frictionDirectionCount
  let nns := data.nonSlidingContacts.size
  let nk := nr * 2
  let vars := nc + nk + nns
  let nMinvNmuT ← data.compliance? data.normalJacobian data.normalMinusMuQJacobian
  let nMinvFT ← data.compliance? data.normalJacobian data.tangentJacobian
  let fMinvNmuT ← data.compliance? data.tangentJacobian data.normalMinusMuQJacobian
  let fMinvFT ← data.compliance? data.tangentJacobian data.tangentJacobian
  let e := data.frictionConeE
  let mut rows : Array (Array Float) := #[]
  for i in [:vars] do
    let mut row : Array Float := #[]
    for j in [:vars] do
      let value :=
        if i < nc then
          if j < nc then
            matGet nMinvNmuT i j
          else if j < nc + nr then
            matGet nMinvFT i (j - nc)
          else if j < nc + nk then
            -(matGet nMinvFT i (j - nc - nr))
          else
            0.0
        else if i < nc + nr then
          let fi := i - nc
          if j < nc then
            matGet fMinvNmuT fi j
          else if j < nc + nr then
            matGet fMinvFT fi (j - nc)
          else if j < nc + nk then
            -(matGet fMinvFT fi (j - nc - nr))
          else
            matGet e fi (j - nc - nk)
        else if i < nc + nk then
          let fi := i - nc - nr
          if j < nc then
            -(matGet fMinvNmuT fi j)
          else if j < nc + nr then
            -(matGet fMinvFT fi (j - nc))
          else if j < nc + nk then
            matGet fMinvFT fi (j - nc - nr)
          else
            matGet e fi (j - nc - nk)
        else
          let ci := i - nc - nk
          if j < nc then
            if data.nonSlidingContacts.getD ci (nc + 1) == j then
              data.muNonSliding.getD ci 0.0
            else
              0.0
          else if j < nc + nr then
            -(matGet e (j - nc) ci)
          else if j < nc + nk then
            -(matGet e (j - nc - nr) ci)
          else
            0.0
      row := row.push value
    rows := rows.push row
  pure rows

private def sustainedLcpVector? (data : RodConstraintAccelProblemData) :
    Except String (Array Float) := do
  let free ← data.freeAcceleration?
  let fFree := FloatArray.add (FloatMatrix.matVec data.tangentJacobian free) data.kF
  pure
    (FloatArray.add (FloatMatrix.matVec data.normalJacobian free) data.kN ++
      fFree ++
      FloatArray.scale (-1.0) fFree ++
      Array.replicate data.nonSlidingContacts.size 0.0)

def packedConstraintForceFromLcpZ (data : RodConstraintAccelProblemData)
    (z : Array Float) : Array Float :=
  let nc := data.contactCount
  let nr := data.frictionDirectionCount
  let fN := arraySlice z 0 nc
  let fDPlus := arraySlice z nc nr
  let fDMinus := arraySlice z (nc + nr) nr
  fN ++ FloatArray.sub fDPlus fDMinus

def generalizedConstraintForce (data : RodConstraintAccelProblemData)
    (packedConstraintForce : Array Float) : Array Float :=
  let nc := data.contactCount
  let nr := data.frictionDirectionCount
  let fN := arraySlice packedConstraintForce 0 nc
  let fF := arraySlice packedConstraintForce nc nr
  let force :=
    FloatArray.add
      (FloatMatrix.transposeVec data.normalMinusMuQJacobian fN)
      (FloatMatrix.transposeVec data.tangentJacobian fF)
  if force.isEmpty then
    Array.replicate data.velocityDim 0.0
  else
    force

end RodConstraintAccelProblemData

structure ContactFrameForce2D where
  candidateId : Nat
  normalForce : Float
  tangentForce : Float
  deriving Repr, Inhabited

def contactFrameForces?
    (data : RodConstraintAccelProblemData)
    (packedConstraintForce : Array Float) :
    Except String (Array ContactFrameForce2D) := do
  data.validate?
  let nc := data.contactCount
  let nr := data.frictionDirectionCount
  if packedConstraintForce.size != nc + nr then
    .error s!"rod2d packed contact force has size {packedConstraintForce.size}, expected {nc + nr}"
  let candidates ← data.support.selectedCandidates?
  if candidates.size != nc then
    .error s!"rod2d frame-force conversion has {candidates.size} selected candidates, expected {nc}"
  let mut out : Array ContactFrameForce2D := #[]
  let mut frictionOffset := 0
  let mut nonSlidingOffset := 0
  let mut slidingOffset := 0
  for i in [:nc] do
    let candidate := candidates[i]!
    let normalForce := packedConstraintForce.getD i 0.0
    if candidate.mode == .sliding then
      let mu := data.muSliding.getD slidingOffset 0.0
      let sign := if candidate.tangentVelocity > 0.0 then -1.0 else 1.0
      out := out.push {
        candidateId := candidate.id
        normalForce := normalForce
        tangentForce := sign * mu * normalForce
      }
      slidingOffset := slidingOffset + 1
    else
      let directions := data.r.getD nonSlidingOffset 0
      if directions != 1 then
        .error s!"rod2d contact-frame force conversion requires one friction direction per non-sliding contact, got {directions}"
      out := out.push {
        candidateId := candidate.id
        normalForce := normalForce
        tangentForce := packedConstraintForce.getD (nc + frictionOffset) 0.0
      }
      frictionOffset := frictionOffset + directions
      nonSlidingOffset := nonSlidingOffset + 1
  pure out

structure RodSustainedContactSolve where
  data : RodConstraintAccelProblemData
  lcpMatrix : Array (Array Float)
  lcpVector : Array Float
  lcpSolution : LinearComplementaritySolution
  packedConstraintForce : Array Float
  generalizedConstraintForce : Array Float
  freeAcceleration : Array Float
  acceleration : Array Float
  moves : Array SkeletonMove := #[]
  deriving Repr, Inhabited

private def duplicateCandidate (candidate : ContactCandidate) (copyIndex : Nat) :
    ContactCandidate :=
  if copyIndex == 0 then
    candidate
  else
    { candidate with
      id := candidate.id + 1000 * copyIndex
      label := s!"{candidate.label} duplicate {copyIndex}" }

private def natRangeArray (n : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:n] do
    out := out.push i
  return out

private def selectedSupportWithDuplicates?
    (p : RodParams) (x : RodState)
    (contactPointDuplicateCount : Nat) : Except String ContactSupport := do
  let base := selectedSupport p x
  let selected ← base.selectedCandidates?
  let mut candidates : Array ContactCandidate := #[]
  for candidate in selected do
    for copyIndex in [:(contactPointDuplicateCount + 1)] do
      candidates := candidates.push (duplicateCandidate candidate copyIndex)
  pure {
    policy := base.policy
    candidates := candidates
    selectedLocalIndices := natRangeArray candidates.size
    sourceCandidateCount? := some candidates.size
    label := s!"{base.label} with duplicated solver rows"
  }

def constraintAccelProblemData?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {})
    (useComplementarityProblemSolver : Bool := true)
    (contactPointDuplicateCount : Nat := 0)
    (frictionDirectionDuplicateCount : Nat := 0) :
    Except String RodConstraintAccelProblemData := do
  let support ← selectedSupportWithDuplicates? p x contactPointDuplicateCount
  support.validateJacobianWidth? 3
  let runtime ← support.toRuntimeSupport?
  let selected ← support.selectedCandidates?
  let frictionDirectionCopies := frictionDirectionDuplicateCount + 1
  let mut normalRows : Array (Array Float) := #[]
  let mut tangentRows : Array (Array Float) := #[]
  let mut normalMinusMuQRows : Array (Array Float) := #[]
  let mut slidingContacts : Array Nat := #[]
  let mut nonSlidingContacts : Array Nat := #[]
  let mut r : Array Nat := #[]
  let mut muSliding : Array Float := #[]
  let mut muNonSliding : Array Float := #[]
  let mut kN : Array Float := #[]
  let mut kF : Array Float := #[]
  for i in [:selected.size] do
    let candidate := selected[i]!
    normalRows := normalRows.push candidate.normalJacobian
    kN := kN.push (normalJacobianDotTimesVelocity x candidate)
    if candidate.mode == .sliding then
      slidingContacts := slidingContacts.push i
      muSliding := muSliding.push p.muCoulomb
      let qRow := slidingDirectionJacobian candidate
      normalMinusMuQRows :=
        normalMinusMuQRows.push
          (FloatArray.sub candidate.normalJacobian
            (FloatArray.scale p.muCoulomb qRow))
    else
      nonSlidingContacts := nonSlidingContacts.push i
      r := r.push frictionDirectionCopies
      muNonSliding := muNonSliding.push p.muStatic
      for _ in [:frictionDirectionCopies] do
        tangentRows := tangentRows.push candidate.tangentJacobian
        kF := kF.push (tangentJacobianDotTimesVelocity x candidate)
      normalMinusMuQRows := normalMinusMuQRows.push candidate.normalJacobian
  let data : RodConstraintAccelProblemData := {
    support := support
    runtimeSupport := runtime
    massMatrix := massMatrix p
    normalJacobian := normalRows
    tangentJacobian := tangentRows
    normalMinusMuQJacobian := normalMinusMuQRows
    slidingContacts := slidingContacts
    nonSlidingContacts := nonSlidingContacts
    r := r
    muSliding := muSliding
    muNonSliding := muNonSliding
    kN := kN
    kF := kF
    tau := spatialForceAsArray (externalForce p applied)
    useComplementarityProblemSolver := useComplementarityProblemSolver
  }
  data.validate?
  pure data

def solveSustainedContact? (data : RodConstraintAccelProblemData) :
    Except String RodSustainedContactSolve := do
  data.validate?
  let free ← data.freeAcceleration?
  let normalEval := FloatArray.add (FloatMatrix.matVec data.normalJacobian free) data.kN
  let vars := data.contactCount + data.frictionDirectionCount
  if data.contactCount == 0 || nonnegativeAll normalEval 1.0e-10 then
    let packed := Array.replicate vars 0.0
    let generalized := data.generalizedConstraintForce packed
    pure {
      data := data
      lcpMatrix := #[]
      lcpVector := #[]
      lcpSolution := {
        z := #[]
        w := #[]
        activeSet := #[]
        maxComplementarity := 0.0
      }
      packedConstraintForce := packed
      generalizedConstraintForce := generalized
      freeAcceleration := free
      acceleration := free
      moves := #[
        {
          kind := .localSchurBlock
          reads := data.runtimeSupport.selectedIds
          label := "rod2d sustained contact fast-exit LCP"
        }
      ]
    }
  else
    let lcpMatrix ← data.sustainedLcpMatrix?
    let lcpVector ← data.sustainedLcpVector?
    let solution ← LinearComplementarityProblem.solveByActiveSet?
      lcpMatrix lcpVector 1.0e-7
    let packed := data.packedConstraintForceFromLcpZ solution.z
    let generalized := data.generalizedConstraintForce packed
    let totalForce := FloatArray.add data.tau generalized
    let acceleration ← RodConstraintAccelProblemData.solveMass? data totalForce
    pure {
      data := data
      lcpMatrix := lcpMatrix
      lcpVector := lcpVector
      lcpSolution := solution
      packedConstraintForce := packed
      generalizedConstraintForce := generalized
      freeAcceleration := free
      acceleration := acceleration
      moves := #[
        {
          kind := .branchAggregate
          reads := data.runtimeSupport.selectedIds
          label := "rod2d dynamic contact support"
        },
        {
          kind := .localSchurBlock
          reads := data.runtimeSupport.selectedIds
          label := "rod2d sustained contact MLCP-to-LCP solve"
        }
      ]
    }

def sustainedContactSolve?
    (p : RodParams) (x : RodState) (applied : SpatialForce2D := {}) :
    Except String RodSustainedContactSolve := do
  let data ← constraintAccelProblemData? p x applied
  solveSustainedContact? data

def derivativeFromForce (p : RodParams) (x : RodState) (force : SpatialForce2D) :
    RodState :=
  {
    x := x.xdot
    y := x.ydot
    theta := x.thetadot
    xdot := force.fx / p.mass
    ydot := force.fy / p.mass
    thetadot := force.tau / p.momentInertia
  }

def continuousDerivative (p : RodParams) (x : RodState)
    (applied : SpatialForce2D := {}) : RodState :=
  let contact := aggregateContactForce (compliantForces p x)
  derivativeFromForce p x (contact.add (externalForce p applied))

def freeFlightDerivative (p : RodParams) (x : RodState) : RodState :=
  derivativeFromForce p x (externalForce p {})

def freeFlightTerm (p : RodParams) : ODETerm RodState Unit :=
  { vectorField := fun _t x _ => freeFlightDerivative p x }

def continuousTerm (p : RodParams) : ODETerm RodState Unit :=
  { vectorField := fun _t x _ => continuousDerivative p x }

def firstContactEvent (p : RodParams) : EventSpec RodState Unit :=
  {
    condition := .real (fun _t x _ => minimumSignedDistance p x)
    direction := some false
    terminate := true
    rootTol := p.rootTol
  }

inductive ImpactProjectionKind where
  | normalOnly
  | sticking
  deriving Repr, BEq, Inhabited

def impactJacobianRows (kind : ImpactProjectionKind)
    (contacts : Array ContactCandidate) : Array (Array Float) := Id.run do
  let mut rows : Array (Array Float) := #[]
  for candidate in contacts do
    rows := rows.push candidate.normalJacobian
    if kind == .sticking then
      rows := rows.push candidate.tangentJacobian
  return rows

structure ImpactProjectionResult where
  support : ContactSupport
  runtimeSupport : RuntimeSupport
  projection : VelocityProjection
  postState : RodState
  deriving Repr, Inhabited

def projectImpact? (p : RodParams) (kind : ImpactProjectionKind)
    (x : RodState) : Except String ImpactProjectionResult := do
  let support := selectedSupport p x
  support.validateJacobianWidth? 3
  let contacts ← support.selectedCandidates?
  if contacts.isEmpty then
    .error "rod2d impact projection requires at least one active endpoint"
  let runtime ← support.toRuntimeSupport?
  let jac := impactJacobianRows kind contacts
  let projection ← VelocityProjection.project? (massMatrix p) jac (velocityAsArray x)
  pure {
    support := support
    runtimeSupport := runtime
    projection := projection
    postState := stateWithVelocityArray x projection.vPost
  }

def identity6 : Array (Array Float) :=
  FloatMatrix.identity 6

def liftVelocityJacobianToState (row : Array Float) : Array Float :=
  #[0.0, 0.0, 0.0, row.getD 0 0.0, row.getD 1 0.0, row.getD 2 0.0]

def candidateMessage (candidate : ContactCandidate) : EventMessage :=
  let closure := positivePart (-candidate.normalVelocity)
  let slip := Float.abs candidate.tangentVelocity
  let velocityMsg := FloatArray.add candidate.normalJacobian
    (FloatArray.scale 0.1 candidate.tangentJacobian)
  {
    value := closure + 0.01 * slip
    stateAdjoint := liftVelocityJacobianToState velocityMsg
    thetaGrad := #[closure]
  }

def branchChildForCandidate (weight : Float) (candidate : ContactCandidate) :
    BranchChild :=
  {
    weight := weight
    resetJac := identity6
    a := liftVelocityJacobianToState candidate.normalJacobian
    message := candidateMessage candidate
  }

def contactBranchData? (support : ContactSupport) : Except String BranchEventData := do
  let selected ← support.selectedCandidates?
  if selected.isEmpty then
    .error "rod2d contact branch requires at least one selected endpoint"
  let first := selected[0]!
  let weight := 1.0 / selected.size.toFloat
  pure {
    children := selected.map (branchChildForCandidate weight)
    guardGrad := liftVelocityJacobianToState first.normalJacobian
    gamma := first.normalVelocity
  }

def rodSolver :=
  RK4.solver
    (Term := ODETerm RodState Unit)
    (Y := RodState)
    (VF := RodState)
    (Args := Unit)

def contactBranchVertex : VertexId := 510

structure FirstImpactRun where
  eventTime : Float
  eventState : RodState
  impact : ImpactProjectionResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def solveToFirstImpact? (p : RodParams := params)
    (x0 : RodState := fallingState) (tFinal : Float := 1.0) :
    Except String FirstImpactRun := do
  let sol :=
    diffeqsolve
      (Term := ODETerm RodState Unit)
      (Y := RodState)
      (VF := RodState)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      (freeFlightTerm p) rodSolver 0.0 tFinal (some p.stepSize) x0 ()
      (saveat := { t1 := true })
      (event := some (firstContactEvent p))
  if sol.result != Result.eventOccurred then
    .error s!"expected rod2d first contact event, got {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "rod2d first contact solve did not save event endpoint"
        else
          let t := ts[ts.size - 1]!
          let x := ys[ys.size - 1]!
          let impact ← projectImpact? p .normalOnly x
          let branchData ← contactBranchData? impact.support
          let segment : AcceptedStepSegment := {
            id := 0
            attemptIndex := 0
            tStart := 0.0
            tAttempt := tFinal
            tAfter := t
            madeJumpAfter := true
            label := "rod2d first endpoint contact"
          }
          let trace :=
            DynamicEventTrace.empty
              |>.push (.interval segment)
              |>.push (.branch contactBranchVertex impact.runtimeSupport branchData)
          trace.validate?
          pure {
            eventTime := t
            eventState := x
            impact := impact
            trace := trace
            moves := trace.moves
          }
    | _, _ => .error "rod2d first contact solve did not save endpoint arrays"

structure Rod2DResult where
  references : Array DrakeReference
  assetCatalog : Array Rod2dExampleAsset
  documentationAssets : Array Rod2dExampleAsset
  defaultSupport : ContactSupport
  defaultImpacting : Bool
  penetratingDerivative : RodState
  sustainedContact : RodSustainedContactSolve
  firstImpact : FirstImpactRun
  packageMoves : Array SkeletonMove := #[]
  deriving Repr, Inhabited

private def localMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    reads := #[vertex]
    writes := #[vertex]
    exactness := exactness
    label := label
  }

def buildEndToEnd? : Except String Rod2DResult := do
  validateRod2dExampleAssets?
  let defaultSupport := selectedSupport params (defaultState params)
  defaultSupport.validateJacobianWidth? 3
  let penetrating :=
    { defaultState params with y := (defaultState params).y - 0.01 }
  let sustainedContact ← sustainedContactSolve?
    { params with mass := 2.0, muCoulomb := 15.0, muStatic := 15.0 }
    (restingVerticalState { params with mass := 2.0, muCoulomb := 15.0, muStatic := 15.0 })
    { fx := 100.0, tau := 100.0 }
  let firstImpact ← solveToFirstImpact? params fallingState 1.0
  pure {
    references := drakeReferences
    assetCatalog := rod2dExampleAssets
    documentationAssets := rod2dDocumentationAssets
    defaultSupport := defaultSupport
    defaultImpacting := isImpacting params (defaultState params)
    penetratingDerivative := continuousDerivative params penetrating
    sustainedContact := sustainedContact
    firstImpact := firstImpact
    packageMoves := #[
      localMove 4390 "rod2d package artifact catalog boundary",
      localMove 4391 "rod2d README image documentation boundary"
    ]
  }

end Tyr.EventSkeleton.Examples.Rod2D
