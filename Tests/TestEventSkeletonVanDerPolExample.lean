import LeanTest
import Tyr.EventSkeleton.Examples.VanDerPol

namespace Tests.EventSkeletonVanDerPolExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.VanDerPol

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

@[test]
def testDedicatedDrakeReferencesCoverVanDerPolTargets : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/van_der_pol/BUILD.bazel"))
    "Dedicated Van der Pol module should reference Drake's BUILD.bazel"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/van_der_pol/van_der_pol.h"))
    "Dedicated Van der Pol module should reference Drake's header"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/van_der_pol/van_der_pol.cc"))
    "Dedicated Van der Pol module should reference Drake's implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/van_der_pol/test/van_der_pol_test.cc"))
    "Dedicated Van der Pol module should reference Drake's C++ test"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/van_der_pol/plot_limit_cycle.py"))
    "Dedicated Van der Pol module should reference Drake's plot script"

@[test]
def testBuildTargetsAndLeafSystemDeclarationsMatchDrake : IO Unit := do
  let result ← assertOk (buildEndToEnd? params) "Van der Pol dedicated end-to-end example"
  let _ ← assertOk (validateBuildTargets? result.buildTargets)
    "Van der Pol BUILD target metadata"
  let _ ← assertOk result.systemSpec.validate?
    "Van der Pol LeafSystem declaration metadata"

  LeanTest.assertEqual result.buildTargets.size 3
    "Drake BUILD.bazel should expose library, plot_limit_cycle py_binary, and gtest targets"
  let library? := result.buildTargets.find? (fun target => target.name == "van_der_pol")
  let plot? := result.buildTargets.find? (fun target => target.name == "plot_limit_cycle")
  let test? := result.buildTargets.find? (fun target => target.name == "van_der_pol_test")
  match library?, plot?, test? with
  | some library, some plot, some test =>
      LeanTest.assertTrue (library.kind == VanDerPolBuildTargetKind.ccLibrary)
        "van_der_pol should be a drake_cc_library"
      LeanTest.assertEqual library.srcs #["van_der_pol.cc"]
        "van_der_pol library should compile van_der_pol.cc"
      LeanTest.assertEqual library.hdrs #["van_der_pol.h"]
        "van_der_pol library should expose van_der_pol.h"
      LeanTest.assertTrue library.publicVisibility
        "van_der_pol library should have public visibility"
      LeanTest.assertTrue (library.hasDep "//systems/framework:system_constraint")
        "van_der_pol library should depend on SystemConstraint for mu >= 0"
      LeanTest.assertTrue (library.hasDep "//systems/primitives:vector_log_sink")
        "CalcLimitCycle should keep the VectorLogSink logging dependency visible"
      LeanTest.assertTrue (plot.kind == VanDerPolBuildTargetKind.pyBinary)
        "plot_limit_cycle should be a drake_py_binary"
      LeanTest.assertTrue plot.addTestRule
        "plot_limit_cycle should set add_test_rule = True"
      LeanTest.assertTrue (plot.hasDep "//bindings/pydrake")
        "plot_limit_cycle should depend on pydrake"
      LeanTest.assertTrue (test.kind == VanDerPolBuildTargetKind.ccGoogletest)
        "van_der_pol_test should be a drake_cc_googletest"
      LeanTest.assertTrue (test.hasDep "//systems/framework/test_utilities:scalar_conversion")
        "van_der_pol_test should depend on scalar_conversion utilities"
  | _, _, _ => LeanTest.fail "Expected all three Van der Pol BUILD targets"

  LeanTest.assertEqual result.systemSpec.continuousPositionSize 1
    "VanDerPolOscillator should DeclareContinuousState with one position"
  LeanTest.assertEqual result.systemSpec.continuousVelocitySize 1
    "VanDerPolOscillator should DeclareContinuousState with one velocity"
  LeanTest.assertEqual result.systemSpec.continuousMiscSize 0
    "VanDerPolOscillator should not declare miscellaneous continuous state"
  let position? := result.systemSpec.outputPorts.find? (fun port => port.index == 0)
  let fullState? := result.systemSpec.outputPorts.find? (fun port => port.index == 1)
  match position?, fullState? with
  | some position, some fullState =>
      LeanTest.assertEqual position.name "y0"
        "The position output port should be the documented y0 port"
      LeanTest.assertEqual position.size 1
        "The y0 output port should have size 1"
      LeanTest.assertEqual position.source "CopyPositionToOutput"
        "The y0 output should be produced by CopyPositionToOutput"
      LeanTest.assertEqual fullState.name "y1"
        "The full-state output port should be the documented y1 port"
      LeanTest.assertEqual fullState.size 2
        "The y1 output port should have size 2"
      LeanTest.assertEqual fullState.source "DeclareStateOutputPort"
        "The y1 output should be a state output port"
  | _, _ => LeanTest.fail "Expected Van der Pol y0 and y1 output port specs"
  LeanTest.assertEqual result.systemSpec.numericParameter.name "mu"
    "VanDerPolOscillator should declare numeric parameter mu"
  LeanTest.assertTrue (approx result.systemSpec.numericParameter.defaultValue 1.0 1.0e-12)
    "VanDerPolOscillator should default mu to 1"
  LeanTest.assertEqual result.systemSpec.inequalityConstraint.name "mu >= 0"
    "VanDerPolOscillator should expose the mu >= 0 inequality constraint"
  LeanTest.assertFalse result.systemSpec.inequalityConstraint.hasUpperBound
    "The Drake mu constraint has no upper bound"

@[test]
def testDerivativeAndParameterValidationMatchDrakeSystem : IO Unit := do
  let deriv := derivative params { q := 2.0, qdot := 3.0 }
  LeanTest.assertTrue (approx deriv.q 3.0 1.0e-12)
    s!"q derivative should equal qdot, got {deriv.q}"
  LeanTest.assertTrue (approx deriv.qdot (-11.0) 1.0e-12)
    s!"qdot derivative should match Drake formula, got {deriv.qdot}"
  match ({ params with mu := -1.0 } : VanDerPolParams).validate? with
  | .ok () => LeanTest.fail "Negative mu should fail the Drake mu >= 0 constraint"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "mu")
        s!"Negative mu diagnostic should mention mu, got {msg}"

@[test]
def testLimitCycleClosureMatchesDrakeTolerance : IO Unit := do
  let result ← assertOk (build? params)
    "Van der Pol dedicated example"
  let _ ← assertOk result.rollout.trace.validate? "Van der Pol trace validation"
  LeanTest.assertEqual result.rollout.samples.size (params.steps + 1)
    "CalcLimitCycle rollout should include the initial sample and every RK4 step"
  LeanTest.assertTrue result.closure.start.isFinite
    s!"Limit-cycle start should be finite, got {reprStr result.closure.start}"
  LeanTest.assertTrue result.closure.finish.isFinite
    s!"Limit-cycle finish should be finite, got {reprStr result.closure.finish}"
  LeanTest.assertTrue (result.closure.passes result.limitCycleTest)
    s!"Default rollout should satisfy Drake's closure tolerances, got q error {result.closure.qError}, qdot error {result.closure.qdotError}"
  LeanTest.assertTrue (result.closure.qError <= 1.0e-2)
    s!"Drake q closure tolerance is 1e-2, got {result.closure.qError}"
  LeanTest.assertTrue (result.closure.qdotError <= 5.0e-3)
    s!"Drake qdot closure tolerance is 5e-3, got {result.closure.qdotError}"

@[test]
def testPlotBoundaryRecordsMatplotlibAggExecutableShape : IO Unit := do
  let result ← assertOk (build? params)
    "Van der Pol dedicated example"
  LeanTest.assertEqual result.plotSpec.backendEnv "Agg"
    "Drake plot_limit_cycle.py sets MPLBACKEND=Agg for tests"
  LeanTest.assertEqual result.plotSpec.outputPath "plot_limit_cycle.png"
    "Drake plot script writes plot_limit_cycle.png"
  LeanTest.assertTrue (approx result.plotSpec.xMin (-2.5) 1.0e-12)
    "Plot x lower bound should match Drake script"
  LeanTest.assertTrue (approx result.plotSpec.xMax 2.5 1.0e-12)
    "Plot x upper bound should match Drake script"
  LeanTest.assertTrue (approx result.plotSpec.yMin (-3.0) 1.0e-12)
    "Plot y lower bound should match Drake script"
  LeanTest.assertTrue (approx result.plotSpec.yMax 3.0 1.0e-12)
    "Plot y upper bound should match Drake script"
  LeanTest.assertEqual result.plotSpec.xLabel "q"
    "Plot x label should match Drake script"
  LeanTest.assertEqual result.plotSpec.yLabel "qdot"
    "Plot y label should match Drake script"
  LeanTest.assertEqual result.plotSpec.lineColor "k"
    "Plot line color should match Drake script"
  LeanTest.assertTrue (approx result.plotSpec.lineWidth 2.0 1.0e-12)
    "Plot line width should match Drake script"
  LeanTest.assertEqual result.plotSpec.suppressBrowserEnv "TEST_TMPDIR"
    "Drake suppresses browser opening under TEST_TMPDIR"
  LeanTest.assertEqual result.leanPlotSpec.packagePath "../lean-plot"
    "LeanPlot boundary should point at the sibling plotting package"
  LeanTest.assertEqual result.leanPlotSpec.renderer "LeanPlot.Export.writeSvg"
    "LeanPlot boundary should use dependency-light SVG export"
  LeanTest.assertEqual result.leanPlotSpec.outputPath "plot_limit_cycle.svg"
    "LeanPlot boundary should write an SVG analogue of the Drake PNG"
  LeanTest.assertTrue (!result.leanPlotSpec.usesSkia)
    "LeanPlot EventSkeleton boundary should avoid the Skia backend by default"

@[test]
def testGraphSeparatesExactOdeIntervalFromPlotBoundary : IO Unit := do
  let result ← assertOk (buildEndToEnd? params)
    "Van der Pol dedicated end-to-end example"
  LeanTest.assertEqual (countMoveKind result.moves .intervalAdjoint) 1
    "Van der Pol rollout should have one exact interval-adjoint move"
  LeanTest.assertEqual (countMoveKind result.moves .checkpointBoundary) 1
    "Van der Pol rollout should have one checkpoint boundary move"
  LeanTest.assertEqual (countMoveKind result.moves .localSchurBlock) 3
    "LeafSystem declarations, Drake plot script, and LeanPlot export should be represented as local boundary moves"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.kind == .localSchurBlock &&
      move.targets == #[systemDeclarationVertex] &&
      move.writes == #[positionOutputVertex, fullStateOutputVertex, muConstraintVertex] &&
      move.exactness == .exact &&
      move.label.contains "mu >= 0"))
    "Graph should expose the LeafSystem output ports and mu constraint as a local declaration boundary"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.kind == .localSchurBlock &&
      move.targets == #[plotBoundaryVertex] &&
      move.writes == #[plotOutputVertex] &&
      move.exactness == .exact &&
      move.label.contains "plot_limit_cycle.py"))
    "Graph should expose the Matplotlib plot executable boundary as an exact local Schur block"
  LeanTest.assertTrue (result.moves.any (fun move =>
      move.kind == .localSchurBlock &&
      move.targets == #[leanPlotBoundaryVertex] &&
      move.writes == #[leanPlotOutputVertex] &&
      move.exactness == .exact &&
      move.label.contains "LeanPlot.Export.writeSvg"))
    "Graph should expose the LeanPlot SVG export boundary as an exact local Schur block"

end Tests.EventSkeletonVanDerPolExample
