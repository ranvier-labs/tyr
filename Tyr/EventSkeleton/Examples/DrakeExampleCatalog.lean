import Tyr.EventSkeleton.Core

/-!
# Drake Example Package Catalog

This module records the package-level artifacts in `../drake/examples` that do
not define a specific dynamics primitive but still matter for a full example
port: Bazel targets, READMEs, and model package catalogs.  Physics-bearing
sources stay in their owning example modules.
-/

namespace Tyr.EventSkeleton.Examples.DrakeExampleCatalog

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def packageReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/.gitignore"
      concept := "records generated debug YAML outputs ignored by the examples package"
    },
    {
      path := "../drake/examples/BUILD.bazel"
      concept := "declares the top-level examples package groups and visibility"
    },
    {
      path := "../drake/examples/README-MATLAB.md"
      concept := "documents MATLAB-specific example usage and exclusions"
    },
    {
      path := "../drake/examples/allegro_hand/BUILD.bazel"
      concept := "declares Allegro hand common libraries, LCM helpers, parser tests, and constant-load demo targets"
    },
    {
      path := "../drake/examples/allegro_hand/joint_control/BUILD.bazel"
      concept := "declares the Allegro twisting-mug joint-control executable and regression tests"
    },
    {
      path := "../drake/examples/allegro_hand/joint_control/README.md"
      concept := "documents the Allegro joint-control twisting-mug scenario"
    },
    {
      path := "../drake/examples/atlas/BUILD.bazel"
      concept := "declares the Atlas dynamics executable and example package metadata"
    },
    {
      path := "../drake/examples/bouncing_ball/BUILD.bazel"
      concept := "declares the bouncing-ball LeafSystem library and tests"
    },
    {
      path := "../drake/examples/compass_gait/BUILD.bazel"
      concept := "declares CompassGait plant, geometry, simulator, and tests"
    },
    {
      path := "../drake/examples/cubic_polynomial/BUILD.bazel"
      concept := "declares cubic-polynomial reachability and region-of-attraction example executables"
    },
    {
      path := "../drake/examples/fibonacci/BUILD.bazel"
      concept := "declares the Fibonacci difference-equation system, runner, and regression"
    },
    {
      path := "../drake/examples/hydroelastic/ball_plate/BUILD.bazel"
      concept := "declares hydroelastic ball-plate executable and model assets"
    },
    {
      path := "../drake/examples/hydroelastic/python_ball_paddle/BUILD.bazel"
      concept := "declares Python ball-paddle hydroelastic runner and SDF install data"
    },
    {
      path := "../drake/examples/hydroelastic/python_ball_paddle/README.md"
      concept := "documents the Python ball-paddle hydroelastic demo"
    },
    {
      path := "../drake/examples/hydroelastic/python_nonconvex_mesh/BUILD.bazel"
      concept := "declares Python non-convex pepper/table hydroelastic runner and assets"
    },
    {
      path := "../drake/examples/hydroelastic/python_nonconvex_mesh/README.md"
      concept := "documents the non-convex pepper/table hydroelastic demo"
    },
    {
      path := "../drake/examples/hydroelastic/spatula_slip_control/BUILD.bazel"
      concept := "declares spatula slip-control executable and SDF assets"
    },
    {
      path := "../drake/examples/hydroelastic/spatula_slip_control/README.md"
      concept := "documents the spatula slip-control hydroelastic demo"
    },
    {
      path := "../drake/examples/kinova_jaco_arm/BUILD.bazel"
      concept := "declares Kinova Jaco controller, simulation, and move-end-effector targets"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/BUILD.bazel"
      concept := "declares Kuka iiwa controller, plan runner, simulation, LCM, and model install targets"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/README.md"
      concept := "documents the Kuka iiwa example model and controller workflow"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/BUILD.bazel"
      concept := "declares Kuka iiwa model asset package groups"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/README.md"
      concept := "documents Kuka iiwa model assets and package layout"
    },
    {
      path := "../drake/examples/mass_spring_cloth/BUILD.bazel"
      concept := "declares cloth spring model, geometry, runner, params, and tests"
    },
    {
      path := "../drake/examples/multibody/acrobot/BUILD.bazel"
      concept := "declares the multibody acrobot passive and LQR examples"
    },
    {
      path := "../drake/examples/multibody/cylinder_with_multicontact/BUILD.bazel"
      concept := "declares cylinder multicontact plant population, dynamics runner, and tests"
    },
    {
      path := "../drake/examples/multibody/inclined_plane_with_body/BUILD.bazel"
      concept := "declares inclined-plane-with-body executable and package docs"
    },
    {
      path := "../drake/examples/multibody/pendulum/BUILD.bazel"
      concept := "declares the multibody pendulum example executable"
    },
    {
      path := "../drake/examples/multibody/rolling_sphere/BUILD.bazel"
      concept := "declares rolling-sphere plant population, dynamics runner, and model assets"
    },
    {
      path := "../drake/examples/pendulum/BUILD.bazel"
      concept := "declares pendulum plant, named vectors, controllers, simulations, geometry, and tests"
    },
    {
      path := "../drake/examples/quadrotor/BUILD.bazel"
      concept := "declares quadrotor plant, geometry, dynamics runner, LQR runner, and tests"
    },
    {
      path := "../drake/examples/quadrotor/README.md"
      concept := "documents quadrotor example usage and model files"
    },
    {
      path := "../drake/examples/rimless_wheel/BUILD.bazel"
      concept := "declares rimless-wheel plant, geometry, simulator, params, and tests"
    },
    {
      path := "../drake/examples/rod2d/BUILD.bazel"
      concept := "declares rod2d plant, geometry, solver, simulator, and tests"
    },
    {
      path := "../drake/examples/rod2d/README.md"
      concept := "documents rod2d contact-solver and simulation example usage"
    },
    {
      path := "../drake/examples/zmp/BUILD.bazel"
      concept := "declares ZMP example executable and planner dependencies"
    }
  ]

def packageReferencePaths : Array String :=
  packageReferences.map (fun ref => ref.path)

def hasPackageReference (path : String) : Bool :=
  packageReferencePaths.contains path

private def containsString (needle : String) (xs : Array String) : Bool :=
  xs.any (fun x => x == needle)

private def hasDuplicateString (xs : Array String) : Bool := Id.run do
  let mut seen : Array String := #[]
  for x in xs do
    if containsString x seen then
      return true
    seen := seen.push x
  return false

def validatePackageReferences?
    (refs : Array DrakeReference := packageReferences) : Except String Unit := do
  if refs.isEmpty then
    .error "Drake example package catalog cannot be empty"
  for ref in refs do
    if ref.path.isEmpty then
      .error "Drake example package catalog contains an empty path"
    if ref.concept.isEmpty then
      .error s!"Drake example package catalog entry {ref.path} has an empty concept"
  let paths := refs.map (fun ref => ref.path)
  if hasDuplicateString paths then
    .error "Drake example package catalog contains duplicate paths"

def catalogBoundaryVertex : VertexId := 13200
def catalogPathIndexVertex : VertexId := 13201
def catalogPackageGroupsVertex : VertexId := 13202
def catalogDocsVertex : VertexId := 13203

def catalogGraph (refs : Array DrakeReference := packageReferences) :
    SkeletonGraph := Id.run do
  let mut g :=
    SkeletonGraph.empty
      |>.addVertex { id := catalogBoundaryVertex, kind := .state .boundary, label := "../drake/examples package catalog" }
      |>.addVertex { id := catalogPathIndexVertex, kind := .state .checkpoint, label := "unique Drake example artifact paths" }
      |>.addVertex { id := catalogPackageGroupsVertex, kind := .opaque, label := "BUILD/package groups without local physics primitive" }
      |>.addVertex { id := catalogDocsVertex, kind := .checkpoint, label := "README and metadata artifacts" }
  g := g.addMove {
    kind := .localSchurBlock
    targets := #[catalogPathIndexVertex]
    reads := #[catalogBoundaryVertex]
    writes := #[catalogPathIndexVertex]
    exactness := .exact
    cost := { work := refs.size.toFloat, memory := refs.size.toFloat }
    label := "index unique ../drake/examples package artifact paths"
  }
  g := g.addMove {
    kind := .localSchurBlock
    targets := #[catalogPackageGroupsVertex]
    reads := #[catalogBoundaryVertex, catalogPathIndexVertex]
    writes := #[catalogPackageGroupsVertex]
    exactness := .exact
    cost := { work := refs.size.toFloat, memory := 1.0 }
    label := "separate package metadata from physics-bearing example modules"
  }
  g := g.addMove {
    kind := .checkpointBoundary
    targets := #[catalogDocsVertex]
    reads := #[catalogPathIndexVertex]
    writes := #[catalogDocsVertex]
    exactness := .exact
    cost := { work := refs.size.toFloat, memory := 1.0 }
    label := "record README and metadata artifacts as package-level checkpoints"
  }
  return g

structure DrakeExampleCatalogResult where
  references : Array DrakeReference
  referencePaths : Array String
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd?
    (refs : Array DrakeReference := packageReferences) :
    Except String DrakeExampleCatalogResult := do
  validatePackageReferences? refs
  let graph := catalogGraph refs
  pure {
    references := refs
    referencePaths := refs.map (fun ref => ref.path)
    graph := graph
    moves := graph.moves
  }

end Tyr.EventSkeleton.Examples.DrakeExampleCatalog
