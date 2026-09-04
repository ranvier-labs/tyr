import LeanTest
import Tyr.EventSkeleton.Examples.DrakeExampleCatalog

namespace Tests.EventSkeletonDrakeExampleCatalog

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.DrakeExampleCatalog

private def hasNoDuplicatePaths (paths : Array String) : Bool := Id.run do
  for i in [:paths.size] do
    for j in [:(paths.size - i - 1)] do
      let k := i + j + 1
      if paths[i]! == paths[k]! then
        return false
  return true

private def assertPackageReference (path : String) (label : String) : IO Unit := do
  LeanTest.assertTrue (hasPackageReference path) label

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

private def requiredPackageReferencePaths : Array String :=
  #[
    "../drake/examples/.gitignore",
    "../drake/examples/BUILD.bazel",
    "../drake/examples/README-MATLAB.md",
    "../drake/examples/allegro_hand/joint_control/README.md",
    "../drake/examples/hydroelastic/python_ball_paddle/README.md",
    "../drake/examples/kuka_iiwa_arm/models/README.md",
    "../drake/examples/rod2d/README.md"
  ]

@[test]
def testPackageLevelDrakeExampleArtifactsAreRecorded : IO Unit := do
  LeanTest.assertEqual packageReferences.size 36
    "Package catalog should cover every currently non-physics BUILD/README/metadata artifact from ../drake/examples"
  LeanTest.assertTrue (hasNoDuplicatePaths packageReferencePaths)
    "Package catalog should not record duplicate Drake paths"
  LeanTest.assertTrue
    (packageReferences.all (fun ref => !ref.path.isEmpty && !ref.concept.isEmpty))
    "Package catalog entries should include both path and concept text"

  for path in requiredPackageReferencePaths do
    assertPackageReference path s!"Package catalog should record {path}"

@[test]
def testEndToEndCatalogBuildsPackageMetadataGraph : IO Unit := do
  let result ← assertOk buildEndToEnd?
    "Drake example catalog end-to-end build"
  let _ ← assertOk (validatePackageReferences? result.references)
    "Drake example catalog references"
  LeanTest.assertEqual result.referencePaths packageReferencePaths
    "End-to-end catalog should expose the same package path index"
  LeanTest.assertEqual result.graph.vertices.size 4
    "End-to-end catalog graph should keep catalog, path-index, package-group, and docs vertices"
  LeanTest.assertEqual (countMoveKind result.moves .localSchurBlock) 2
    "End-to-end catalog should use exact local moves for catalog indexing and grouping"
  LeanTest.assertEqual (countMoveKind result.moves .checkpointBoundary) 1
    "End-to-end catalog should checkpoint README and metadata artifacts"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.label.contains "separate package metadata"))
    "End-to-end catalog should not pretend package-only artifacts are physics primitives"

end Tests.EventSkeletonDrakeExampleCatalog
