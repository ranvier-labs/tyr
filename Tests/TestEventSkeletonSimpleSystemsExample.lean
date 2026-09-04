import LeanTest
import Tyr.EventSkeleton.Examples.SimpleSystems

namespace Tests.EventSkeletonSimpleSystemsExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.SimpleSystems

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

@[test]
def testDrakeReferencesAndClockedUpdatePrimitiveAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/BUILD.bazel"))
    "Should reference Drake's root examples BUILD package"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/simple_continuous_time_system.cc"))
    "Should reference Drake's simple continuous system"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/simple_discrete_time_system.cc"))
    "Should reference Drake's simple discrete system"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/simple_mixed_continuous_and_discrete_time_system.cc"))
    "Should reference Drake's mixed continuous/discrete system"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/fibonacci/fibonacci_difference_equation.h"))
    "Should reference Drake's Fibonacci difference equation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/van_der_pol/van_der_pol.cc"))
    "Should reference Drake's van der Pol implementation"

  LeanTest.assertTrue
    (SkeletonMoveKind.clockedUpdate.defaultExactness == MoveExactness.exact)
    "Clocked deterministic updates should be exact update-VJP moves"

  let validData :=
    clockedScalarUpdateData 1.0 1.0 0.99 (cubicUpdate 0.99)
      "valid scalar update"
  assertOk validData.validate? "valid clocked update"

  let badData : ClockedUpdateData := { validData with period := 0.0 }
  match badData.validate? with
  | .ok () => LeanTest.fail "Zero-period clocked update should fail validation"
  | .error msg =>
      LeanTest.assertTrue (msg.contains "period")
        s!"Expected period diagnostic, got {msg}"

@[test]
def testTopLevelExamplesBuildPackageBoundaryMatchesDrake : IO Unit := do
  let result ← assertOk buildExamplesPackageBoundary?
    "root examples package boundary"
  let _ ← assertOk (validateSimpleSystemBuildTargets? result.buildTargets)
    "simple systems build target validation"
  let _ ← assertOk result.modelBoundary.validate?
    "examples model package boundary validation"

  LeanTest.assertEqual result.buildTargets.size 3
    "Root examples BUILD should expose three simple system drake_cc_binary targets"
  LeanTest.assertEqual
    (result.buildTargets.map (fun target => target.name))
    #[
      "simple_continuous_time_system",
      "simple_discrete_time_system",
      "simple_mixed_continuous_and_discrete_time_system"
    ]
    "Root examples BUILD target names should match Drake"

  LeanTest.assertEqual result.modelBoundary.modelPackages expectedInstalledModelPackages
    "Installed model package list should match Drake's root BUILD.bazel"

  LeanTest.assertEqual result.graph.vertices.size 6
    "Package boundary graph should contain BUILD, three binaries, models, and install vertices"
  LeanTest.assertEqual (countMoveKind result.moves .localSchurBlock) 5
    "Package boundary graph should use exact local Schur moves for the three binaries, filegroup, and install target"
  LeanTest.assertTrue
    (result.moves.all (fun move => move.exactness == MoveExactness.exact))
    "Package boundary moves should be exact metadata eliminations"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.kind == .localSchurBlock &&
        move.label.contains "drake_cc_binary(simple_continuous_time_system)"))
    "Graph should record the continuous root binary boundary"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.kind == .localSchurBlock &&
        move.label.contains "filegroup(name=\"models\")"))
    "Graph should record the public models filegroup boundary"
  LeanTest.assertTrue
    (result.moves.any (fun move =>
      move.kind == .localSchurBlock &&
        move.label.contains "install(name=\"install\")"))
    "Graph should record the public install target boundary"

@[test]
def testSimpleContinuousSystemConvergesLikeDrakeExample : IO Unit := do
  let result ← assertOk (simulateContinuous? continuousParams)
    "simple continuous simulation"
  let _ ← assertOk result.trace.validate? "continuous trace validation"
  LeanTest.assertEqual result.rollout.samples.size (continuousParams.steps + 1)
    "Continuous rollout should include the initial sample and every RK4 step"
  LeanTest.assertTrue (result.rollout.final < 1.0e-4)
    s!"Drake demand expects convergence below 1e-4, got {result.rollout.final}"
  LeanTest.assertTrue (approx result.rollout.final result.closedFormFinal 1.0e-10)
    s!"RK4 rollout should match the closed form, got {result.rollout.final} vs {result.closedFormFinal}"
  LeanTest.assertEqual result.moves.size 2
    "Single continuous interval should project interval-adjoint and checkpoint moves"
  LeanTest.assertTrue (result.moves[0]!.kind == SkeletonMoveKind.intervalAdjoint)
    "First continuous move should be interval adjoint"
  LeanTest.assertTrue (result.moves[1]!.kind == SkeletonMoveKind.checkpointBoundary)
    "Second continuous move should be checkpoint boundary"

@[test]
def testSimpleDiscreteSystemRecordsClockedUpdates : IO Unit := do
  let result ← assertOk (simulateDiscrete? discreteParams)
    "simple discrete simulation"
  let _ ← assertOk result.trace.validate? "discrete trace validation"
  LeanTest.assertEqual result.samples.size (discreteParams.steps + 1)
    "Discrete rollout should include the initial state and every update"
  LeanTest.assertTrue (approx result.samples[0]! 0.99 1.0e-12)
    s!"Initial discrete state should match Drake example, got {result.samples[0]!}"
  LeanTest.assertTrue (approx result.samples[1]! 0.970299 1.0e-12)
    s!"First update should cube the state, got {result.samples[1]!}"
  LeanTest.assertTrue (result.samples[result.samples.size - 1]! < 1.0e-4)
    s!"Drake demand expects convergence below 1e-4, got {result.samples[result.samples.size - 1]!}"
  LeanTest.assertEqual result.trace.entries.size discreteParams.steps
    "Pure discrete rollout should record one clocked update entry per update"
  LeanTest.assertEqual result.moves.size discreteParams.steps
    "Pure discrete rollout should project one move per clocked update"
  LeanTest.assertEqual (countMoveKind result.moves .clockedUpdate) discreteParams.steps
    "All pure discrete moves should be clocked updates"

@[test]
def testMixedSystemCombinesIntervalsAndClockedUpdates : IO Unit := do
  let result ← assertOk (simulateMixed? mixedParams)
    "simple mixed simulation"
  let _ ← assertOk result.trace.validate? "mixed trace validation"
  LeanTest.assertEqual result.samples.size (mixedParams.periods + 1)
    "Mixed rollout should include the initial state and one sample per period"
  LeanTest.assertTrue result.finalState.isFinite
    s!"Mixed final state should remain finite, got {reprStr result.finalState}"
  LeanTest.assertTrue (result.finalState.discrete < 1.0e-4)
    s!"Mixed discrete state should converge below 1e-4, got {result.finalState.discrete}"
  LeanTest.assertTrue (result.finalState.continuous < 1.0e-4)
    s!"Mixed continuous state should converge below 1e-4, got {result.finalState.continuous}"
  LeanTest.assertEqual result.trace.entries.size (2 * mixedParams.periods)
    "Each mixed period should record one continuous interval and one clocked update"
  LeanTest.assertEqual (countMoveKind result.moves .intervalAdjoint) mixedParams.periods
    "Mixed trace should project one interval adjoint per continuous period"
  LeanTest.assertEqual (countMoveKind result.moves .checkpointBoundary) mixedParams.periods
    "Mixed trace should project one checkpoint per continuous period"
  LeanTest.assertEqual (countMoveKind result.moves .clockedUpdate) mixedParams.periods
    "Mixed trace should project one clocked update per discrete period"

@[test]
def testFibonacciDifferenceEquationMatchesDrakeSequence : IO Unit := do
  let result ← assertOk (simulateFibonacci? fibonacciParams)
    "Fibonacci simulation"
  let _ ← assertOk result.trace.validate? "Fibonacci trace validation"
  let values := result.samples.map (fun sample => sample.value)
  LeanTest.assertEqual values #[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    "Default ten-step Fibonacci rollout should match Drake's documented sequence"
  LeanTest.assertEqual result.samples[1]!.time fibonacciParams.period
    "Fibonacci samples should be spaced by the Drake period"
  LeanTest.assertTrue (result.finalState == { current := 55, previous := 34 })
    "Final Fibonacci state should carry F_10 and F_9"
  LeanTest.assertEqual result.moves.size fibonacciParams.steps
    "Fibonacci trace should project one clocked update per recurrence update"
  LeanTest.assertEqual (countMoveKind result.moves .clockedUpdate) fibonacciParams.steps
    "Fibonacci moves should all be clocked updates"

@[test]
def testVanDerPolDerivativeOutputsAndLimitCycleRollout : IO Unit := do
  let deriv := vanDerPolDerivative vanDerPolParams { q := 2.0, qdot := 3.0 }
  LeanTest.assertTrue (approx deriv.q 3.0 1.0e-12)
    s!"q derivative should equal qdot, got {deriv.q}"
  LeanTest.assertTrue (approx deriv.qdot (-11.0) 1.0e-12)
    s!"qdot derivative should match Drake formula, got {deriv.qdot}"

  let result ← assertOk (simulateVanDerPol? vanDerPolParams)
    "van der Pol simulation"
  let _ ← assertOk result.trace.validate? "van der Pol trace validation"
  LeanTest.assertEqual result.samples.size (vanDerPolParams.steps + 1)
    "Van der Pol rollout should include the initial sample and every RK4 step"
  LeanTest.assertTrue result.finalState.isFinite
    s!"Van der Pol final state should remain finite, got {reprStr result.finalState}"
  LeanTest.assertEqual result.positionOutput result.finalState.q
    "Position output should expose q"
  LeanTest.assertEqual result.fullStateOutput result.finalState.asArray
    "Full-state output should expose [q, qdot]"
  LeanTest.assertEqual result.moves.size 2
    "Van der Pol continuous interval should project interval-adjoint and checkpoint moves"
  let dq := result.finalState.q - result.initialState.q
  let dv := result.finalState.qdot - result.initialState.qdot
  LeanTest.assertTrue (Float.sqrt (dq * dq + dv * dv) < 0.1)
    s!"Default Drake limit-cycle period should return near the starting state, got final={reprStr result.finalState}"

@[test]
def testEndToEndBuildAggregatesPackageBoundaryAndPrimitiveRuns : IO Unit := do
  let result ← assertOk buildEndToEnd?
    "SimpleSystems end-to-end build"
  let _ ← assertOk result.continuous.trace.validate?
    "end-to-end continuous trace"
  let _ ← assertOk result.discrete.trace.validate?
    "end-to-end discrete trace"
  let _ ← assertOk result.mixed.trace.validate?
    "end-to-end mixed trace"
  let _ ← assertOk result.fibonacci.trace.validate?
    "end-to-end Fibonacci trace"
  let _ ← assertOk result.vanDerPol.trace.validate?
    "end-to-end Van der Pol trace"

  LeanTest.assertEqual result.packageBoundary.buildTargets.size 3
    "End-to-end result should include the root simple-system binaries"
  LeanTest.assertEqual result.continuous.rollout.samples.size
    (continuousParams.steps + 1)
    "End-to-end result should retain the continuous rollout"
  LeanTest.assertEqual result.discrete.samples.size
    (discreteParams.steps + 1)
    "End-to-end result should retain the discrete rollout"
  LeanTest.assertEqual result.mixed.samples.size
    (mixedParams.periods + 1)
    "End-to-end result should retain the mixed rollout"
  LeanTest.assertEqual (result.fibonacci.samples.map (fun sample => sample.value))
    #[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    "End-to-end result should retain the Fibonacci runner sequence"
  LeanTest.assertTrue result.vanDerPol.finalState.isFinite
    "End-to-end result should retain the Van der Pol rollout"

  LeanTest.assertEqual (countMoveKind result.moves .localSchurBlock) 5
    "End-to-end move schedule should include package metadata Schur blocks"
  LeanTest.assertEqual (countMoveKind result.moves .intervalAdjoint) 12
    "End-to-end move schedule should include continuous and mixed interval adjoints"
  LeanTest.assertEqual (countMoveKind result.moves .checkpointBoundary) 12
    "End-to-end move schedule should include interval checkpoint boundaries"
  LeanTest.assertEqual (countMoveKind result.moves .clockedUpdate) 30
    "End-to-end move schedule should include all clocked reset updates"

end Tests.EventSkeletonSimpleSystemsExample
