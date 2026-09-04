import LeanTest
import Tyr.EventSkeleton.Examples.Fibonacci

namespace Tests.EventSkeletonFibonacciExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Fibonacci

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def countMoveKind (moves : Array SkeletonMove) (kind : SkeletonMoveKind) : Nat :=
  (moves.filter (fun move => move.kind == kind)).size

@[test]
def testDrakeReferencesAndDefaultPeriodAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/fibonacci/BUILD.bazel"))
    "Fibonacci module should reference Drake's BUILD.bazel"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/fibonacci/fibonacci_difference_equation.h"))
    "Fibonacci module should reference Drake's difference-equation system"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/fibonacci/test/fibonacci_difference_equation_test.cc"))
    "Fibonacci module should reference Drake's difference-equation regression"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref =>
      ref.path == "../drake/examples/fibonacci/run_fibonacci.cc"))
    "Fibonacci module should reference Drake's runner"
  LeanTest.assertTrue (approx params.period 0.25 1.0e-12)
    s!"Default period should match Drake's kPeriod, got {params.period}"
  LeanTest.assertEqual params.steps 10
    "Default runner length should match Drake's --steps default"
  LeanTest.assertTrue (approx params.finalTime 2.5 1.0e-12)
    s!"Default final time should be steps * period = 2.5, got {params.finalTime}"

@[test]
def testBuildTargetsSystemAndLoggerBoundariesMatchDrake : IO Unit := do
  let result ← assertOk (buildEndToEnd? params) "default Fibonacci end-to-end simulation"
  let _ ← assertOk (validateBuildTargets? result.buildTargets)
    "Fibonacci BUILD target metadata"
  let _ ← assertOk result.systemSpec.validate?
    "Fibonacci system metadata"
  let _ ← assertOk (result.runnerSpec.validate? params)
    "Fibonacci runner metadata"
  let _ ← assertOk (result.gtestSpec.validate? params)
    "Fibonacci gtest metadata"

  LeanTest.assertEqual result.buildTargets.size 3
    "Drake BUILD.bazel should expose library, runner, and gtest targets"
  let library? := result.buildTargets.find?
    (fun target => target.name == "fibonacci_difference_equation")
  let runner? := result.buildTargets.find?
    (fun target => target.name == "run_fibonacci")
  let test? := result.buildTargets.find?
    (fun target => target.name == "fibonacci_difference_equation_test")
  match library?, runner?, test? with
  | some library, some runner, some test =>
      LeanTest.assertTrue (library.kind == FibonacciBuildTargetKind.ccLibrary)
        "Fibonacci difference equation should be a drake_cc_library"
      LeanTest.assertTrue (library.hasDep "//systems/framework:leaf_system")
        "Fibonacci library should depend on LeafSystem"
      LeanTest.assertTrue (runner.kind == FibonacciBuildTargetKind.ccBinary)
        "run_fibonacci should be a drake_cc_binary"
      LeanTest.assertTrue (runner.hasDep "@gflags")
        "run_fibonacci should expose the --steps gflags dependency"
      LeanTest.assertTrue (runner.hasDep "//systems/analysis:simulator")
        "run_fibonacci should use Drake Simulator"
      LeanTest.assertTrue (test.kind == FibonacciBuildTargetKind.ccGoogletest)
        "fibonacci_difference_equation_test should be a drake_cc_googletest"
      LeanTest.assertTrue (test.hasDep "//systems/primitives:vector_log_sink")
        "The gtest should use VectorLogSink"
  | _, _, _ => LeanTest.fail "Expected all three Fibonacci BUILD targets"

  LeanTest.assertEqual result.systemSpec.inputPorts 0
    "Fibonacci LeafSystem should have no inputs"
  LeanTest.assertEqual result.systemSpec.outputPortName "Fn"
    "Fibonacci LeafSystem output port should be named Fn"
  LeanTest.assertFalse result.systemSpec.directFeedthrough
    "Fibonacci LeafSystem should declare no feedthrough"
  LeanTest.assertEqual result.systemSpec.initialDiscreteState #[0.0, 1.0]
    "Fibonacci LeafSystem should DeclareDiscreteState([0,1])"
  LeanTest.assertTrue (approx result.systemSpec.firstUpdateTime 0.0 1.0e-12)
    "Fibonacci periodic update event should be declared with offset 0"
  LeanTest.assertTrue (result.runnerSpec.logger.construction == FibonacciLoggerConstruction.logVectorOutput)
    "run_fibonacci should use LogVectorOutput"
  LeanTest.assertTrue (result.gtestSpec.logger.construction == FibonacciLoggerConstruction.explicitVectorLogSink)
    "The gtest should construct VectorLogSink explicitly"

@[test]
def testFibonacciRecurrenceAndLogMatchDrakeRunner : IO Unit := do
  let result ← assertOk (simulate? params) "default Fibonacci simulation"
  let _ ← assertOk result.trace.validate? "Fibonacci trace validation"
  let values := result.samples.map (fun sample => sample.value)
  LeanTest.assertEqual values #[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    "Default ten-step rollout should match Drake's documented Fibonacci sequence"
  LeanTest.assertEqual result.logData #[0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]
    "Logged Fn data should match VectorLogSink's single row"
  LeanTest.assertEqual result.logSampleTimes[0]! 0.0
    "Log should include the initial sample at t=0"
  LeanTest.assertTrue (approx result.logSampleTimes[result.logSampleTimes.size - 1]! 2.5 1.0e-12)
    "Final default log sample should occur at steps*kPeriod"
  LeanTest.assertEqual result.logLines[0]! "0: 0 (t=0.000000)"
    s!"First log line should match Drake's n/value/time shape, got {result.logLines[0]!}"
  LeanTest.assertEqual result.logLines[10]! "10: 55 (t=2.500000)"
    s!"Last default log line should match Drake runner output shape, got {result.logLines[10]!}"
  LeanTest.assertTrue (result.finalState == { current := 55, previous := 34 })
    s!"Final two-state recurrence should hold F10/F9, got {reprStr result.finalState}"
  LeanTest.assertTrue (result.initialState == { current := 0, previous := 1 })
    "Initial state should match Drake's DeclareDiscreteState([0, 1])"

@[test]
def testClockedUpdateTraceRepresentsPeriodicReset : IO Unit := do
  let result ← assertOk (simulate? params) "default Fibonacci simulation"
  LeanTest.assertEqual result.trace.entries.size params.steps
    "Fibonacci should record one periodic update event per step"
  LeanTest.assertEqual result.trace.moves.size params.steps
    "Each periodic update should project one trace-local clocked-update move"
  LeanTest.assertEqual (countMoveKind result.moves .clockedUpdate) params.steps
    "All Fibonacci elimination moves should be clocked updates"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "The output port evaluation should be exposed as an exact local boundary"
  LeanTest.assertEqual (countMoveKind result.moves .checkpointBoundary) 2
    "The graph should checkpoint the VectorLogSink log and stdout rows"

  match result.trace.entries[0]! with
  | .clockedUpdate vertex data =>
      LeanTest.assertEqual vertex (updateVertex 0)
        "First update should use the stable Fibonacci update vertex id"
      LeanTest.assertTrue (approx data.time 0.0 1.0e-12)
        s!"Drake declares the first periodic update at t=0, got {data.time}"
      LeanTest.assertEqual data.stateBefore #[0.0, 1.0]
        "First update should read [F0, F-1 proxy] = [0, 1]"
      LeanTest.assertEqual data.stateAfter #[1.0, 0.0]
        "First update should write [F1, F0]"
      LeanTest.assertEqual data.updateJac updateJacobian
        "Clocked update should carry the Fibonacci recurrence Jacobian"
  | _ => LeanTest.fail "First trace entry should be a clocked update"

@[test]
def testClockedUpdateVjpMatchesJacobianTranspose : IO Unit := do
  let cotAfter := #[2.0, 3.0]
  LeanTest.assertEqual (updateVjp cotAfter) #[5.0, 2.0]
    "J^T * [2,3] for [[1,1],[1,0]] should be [5,2]"
  let data := updateData 4 params { current := 3, previous := 2 }
    { current := 5, previous := 3 }
  assertOk data.validate? "Fibonacci clocked update data"
  LeanTest.assertEqual data.updateJac #[#[1.0, 1.0], #[1.0, 0.0]]
    "Update data should expose the recurrence Jacobian for AD"

@[test]
def testCustomStepCountMatchesRunnerFinalTime : IO Unit := do
  let p : FibonacciParams := { period := 0.25, steps := 8 }
  let result ← assertOk (simulate? p) "custom Fibonacci simulation"
  let values := result.samples.map (fun sample => sample.value)
  LeanTest.assertEqual values #[0, 1, 1, 2, 3, 5, 8, 13, 21]
    "Eight-step rollout should stop at F8"
  LeanTest.assertTrue (approx result.finalTime 2.0 1.0e-12)
    s!"Runner final time should be steps * period, got {result.finalTime}"
  LeanTest.assertEqual result.samples[result.samples.size - 1]!.n 8
    "Last sample index should match requested steps"

@[test]
def testDrakeGtestSequenceUsesSixStepAdvanceAndInitialSample : IO Unit := do
  let result ← assertOk (simulate? { period := params.period, steps := gtestSpec.advanceSteps })
    "Fibonacci gtest-length simulation"
  LeanTest.assertEqual result.gtestSpec.expectedValues #[0, 1, 1, 2, 3, 5, 8]
    "Drake gtest expected vector should be recorded"
  LeanTest.assertEqual (sampleValues result.samples) result.gtestSpec.expectedValues
    "Advancing to 6*kPeriod should produce the seven logged samples expected by Drake"
  LeanTest.assertEqual result.samples.size 7
    "The gtest log includes the initial t=0 sample plus six updates"
  LeanTest.assertTrue (approx result.finalTime (6.0 * params.period) 1.0e-12)
    s!"Gtest simulation should AdvanceTo 6*kPeriod, got {result.finalTime}"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.label.contains "LogVectorOutput" ||
      move.label.contains "VectorLogSink"))
    "Graph should expose the sampled output logger boundary"

end Tests.EventSkeletonFibonacciExample
