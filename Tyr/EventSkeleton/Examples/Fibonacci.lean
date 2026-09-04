import Tyr.EventSkeleton.Trace

/-!
# Drake Fibonacci Difference Equation Event-Skeleton Example

This ports `../drake/examples/fibonacci` as its own example module.  Drake uses
a pure discrete LeafSystem with a periodic update event; in the event-skeleton
view that is a prescribed clocked reset, so reverse mode applies the update VJP
without a saltation timing scalar.
-/

namespace Tyr.EventSkeleton.Examples.Fibonacci

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/fibonacci/BUILD.bazel"
      concept := "declares the Fibonacci LeafSystem library, runner binary, and VectorLogSink gtest"
    },
    {
      path := "../drake/examples/fibonacci/fibonacci_difference_equation.h"
      concept := "declares the two-state Fibonacci difference equation, period, output port, and periodic update event"
    },
    {
      path := "../drake/examples/fibonacci/test/fibonacci_difference_equation_test.cc"
      concept := "checks the periodic unrestricted update, output port, and Fibonacci state progression"
    },
    {
      path := "../drake/examples/fibonacci/run_fibonacci.cc"
      concept := "builds the diagram, logs the Fn output every period, advances to steps * period, and prints n/value/time rows"
    }
  ]

inductive FibonacciBuildTargetKind where
  | ccLibrary
  | ccBinary
  | ccGoogletest
  deriving Repr, BEq, Inhabited

structure FibonacciBuildTarget where
  kind : FibonacciBuildTargetKind
  name : String
  sources : Array String := #[]
  deps : Array String := #[]
  deriving Repr, BEq, Inhabited

namespace FibonacciBuildTarget

def hasDep (target : FibonacciBuildTarget) (dep : String) : Bool :=
  target.deps.any (fun actual => actual == dep)

def validate? (target : FibonacciBuildTarget) : Except String Unit := do
  if target.name.isEmpty then
    .error "Fibonacci BUILD target name cannot be empty"
  match target.kind with
  | .ccLibrary =>
      if target.sources != #["fibonacci_difference_equation.h"] then
        .error s!"Fibonacci library should expose fibonacci_difference_equation.h, got {target.sources}"
      if !target.hasDep "//systems/framework:leaf_system" then
        .error "Fibonacci library should depend on LeafSystem"
  | .ccBinary =>
      if target.sources != #["run_fibonacci.cc"] then
        .error s!"Fibonacci runner should compile run_fibonacci.cc, got {target.sources}"
      if !target.hasDep ":fibonacci_difference_equation" then
        .error "Fibonacci runner should depend on the local difference-equation library"
      if !target.hasDep "//systems/analysis:simulator" then
        .error "Fibonacci runner should depend on Simulator"
      if !target.hasDep "//systems/primitives:vector_log_sink" then
        .error "Fibonacci runner should depend on VectorLogSink"
      if !target.hasDep "@gflags" then
        .error "Fibonacci runner should expose the --steps gflags option"
  | .ccGoogletest =>
      if target.name != "fibonacci_difference_equation_test" then
        .error s!"Unexpected Fibonacci gtest target {target.name}"
      if !target.hasDep ":fibonacci_difference_equation" then
        .error "Fibonacci gtest should depend on the local difference-equation library"
      if !target.hasDep "//systems/primitives:vector_log_sink" then
        .error "Fibonacci gtest should depend on VectorLogSink"

end FibonacciBuildTarget

def buildTargets : Array FibonacciBuildTarget :=
  #[
    {
      kind := .ccLibrary
      name := "fibonacci_difference_equation"
      sources := #["fibonacci_difference_equation.h"]
      deps := #["//common:essential", "//systems/framework:leaf_system"]
    },
    {
      kind := .ccBinary
      name := "run_fibonacci"
      sources := #["run_fibonacci.cc"]
      deps := #[
        ":fibonacci_difference_equation",
        "//systems/analysis:simulator",
        "//systems/primitives:vector_log_sink",
        "@gflags"
      ]
    },
    {
      kind := .ccGoogletest
      name := "fibonacci_difference_equation_test"
      deps := #[
        ":fibonacci_difference_equation",
        "//systems/analysis:simulator",
        "//systems/primitives:vector_log_sink"
      ]
    }
  ]

def validateBuildTargets? (targets : Array FibonacciBuildTarget := buildTargets) :
    Except String Unit := do
  if targets.size != 3 then
    .error s!"Fibonacci BUILD.bazel should declare library, runner, and gtest targets, got {targets.size}"
  if !(targets.any (fun target => target.name == "fibonacci_difference_equation")) then
    .error "missing fibonacci_difference_equation library target"
  if !(targets.any (fun target => target.name == "run_fibonacci")) then
    .error "missing run_fibonacci binary target"
  if !(targets.any (fun target => target.name == "fibonacci_difference_equation_test")) then
    .error "missing fibonacci_difference_equation_test gtest target"
  for target in targets do
    target.validate?

structure FibonacciSystemSpec where
  systemName : String := "FibonacciDifferenceEquation"
  inputPorts : Nat := 0
  outputPortName : String := "Fn"
  outputPortSize : Nat := 1
  directFeedthrough : Bool := false
  discreteStateSize : Nat := 2
  initialDiscreteState : Array Float := #[0.0, 1.0]
  firstUpdateTime : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace FibonacciSystemSpec

def validate? (spec : FibonacciSystemSpec) : Except String Unit := do
  if spec.systemName != "FibonacciDifferenceEquation" then
    .error s!"Unexpected Fibonacci system name {spec.systemName}"
  if spec.inputPorts != 0 then
    .error s!"Fibonacci system should have zero input ports, got {spec.inputPorts}"
  if spec.outputPortName != "Fn" || spec.outputPortSize != 1 then
    .error s!"Fibonacci output port should be Fn with size 1, got {spec.outputPortName}/{spec.outputPortSize}"
  if spec.directFeedthrough then
    .error "Fibonacci LeafSystem should declare no feedthrough"
  if spec.discreteStateSize != 2 then
    .error s!"Fibonacci discrete state should have size 2, got {spec.discreteStateSize}"
  if spec.initialDiscreteState != #[0.0, 1.0] then
    .error s!"Fibonacci initial discrete state should be [0,1], got {spec.initialDiscreteState}"
  if Float.abs (spec.firstUpdateTime - 0.0) > 1.0e-12 then
    .error s!"Fibonacci first update should be declared at t=0, got {spec.firstUpdateTime}"

end FibonacciSystemSpec

def systemSpec : FibonacciSystemSpec := {}

structure FibonacciParams where
  period : Float := 0.25
  steps : Nat := 10
  deriving Repr, Inhabited

def params : FibonacciParams := {}

namespace FibonacciParams

def validate? (p : FibonacciParams) : Except String Unit := do
  if !(Float.isFinite p.period) || p.period <= 0.0 then
    .error s!"Fibonacci period must be positive and finite, got {p.period}"

def finalTime (p : FibonacciParams) : Float :=
  p.steps.toFloat * p.period

end FibonacciParams

inductive FibonacciLoggerConstruction where
  | logVectorOutput
  | explicitVectorLogSink
  deriving Repr, BEq, Inhabited

structure FibonacciLoggerSpec where
  construction : FibonacciLoggerConstruction
  sourcePort : String := "Fn"
  inputSize : Nat := 1
  samplePeriod : Float := params.period
  deriving Repr, BEq, Inhabited

namespace FibonacciLoggerSpec

def validate? (spec : FibonacciLoggerSpec) (p : FibonacciParams := params) :
    Except String Unit := do
  if spec.sourcePort != "Fn" then
    .error s!"Fibonacci logger should sample output port Fn, got {spec.sourcePort}"
  if spec.inputSize != 1 then
    .error s!"Fibonacci VectorLogSink input size should be 1, got {spec.inputSize}"
  if Float.abs (spec.samplePeriod - p.period) > 1.0e-12 then
    .error s!"Fibonacci logger sample period should match kPeriod={p.period}, got {spec.samplePeriod}"

end FibonacciLoggerSpec

def runnerLoggerSpec : FibonacciLoggerSpec :=
  { construction := .logVectorOutput }

def gtestLoggerSpec : FibonacciLoggerSpec :=
  { construction := .explicitVectorLogSink }

structure FibonacciRunnerSpec where
  executable : String := "run_fibonacci"
  flagName : String := "steps"
  defaultSteps : Nat := 10
  usage : String := "usage: run_fibonacci [--steps=n]"
  logger : FibonacciLoggerSpec := runnerLoggerSpec
  deriving Repr, BEq, Inhabited

namespace FibonacciRunnerSpec

def validate? (spec : FibonacciRunnerSpec) (p : FibonacciParams := params) :
    Except String Unit := do
  if spec.executable != "run_fibonacci" then
    .error s!"Fibonacci runner executable should be run_fibonacci, got {spec.executable}"
  if spec.flagName != "steps" then
    .error s!"Fibonacci runner should expose --steps, got --{spec.flagName}"
  if spec.defaultSteps != 10 then
    .error s!"Fibonacci --steps default should be 10, got {spec.defaultSteps}"
  if spec.usage != "usage: run_fibonacci [--steps=n]" then
    .error s!"Fibonacci runner usage string mismatch: {spec.usage}"
  spec.logger.validate? p
  if spec.logger.construction != .logVectorOutput then
    .error "Fibonacci runner should use LogVectorOutput"

end FibonacciRunnerSpec

def runnerSpec : FibonacciRunnerSpec := {}

structure FibonacciGtestSpec where
  testSuite : String := "Fibonacci"
  testName : String := "CheckSequence"
  advanceSteps : Nat := 6
  expectedValues : Array Nat := #[0, 1, 1, 2, 3, 5, 8]
  logger : FibonacciLoggerSpec := gtestLoggerSpec
  deriving Repr, BEq, Inhabited

namespace FibonacciGtestSpec

def validate? (spec : FibonacciGtestSpec) (p : FibonacciParams := params) :
    Except String Unit := do
  if spec.testSuite != "Fibonacci" || spec.testName != "CheckSequence" then
    .error s!"Unexpected Fibonacci gtest name {spec.testSuite}.{spec.testName}"
  if spec.advanceSteps != 6 then
    .error s!"Drake Fibonacci gtest should AdvanceTo 6*kPeriod, got {spec.advanceSteps}"
  if spec.expectedValues != #[0, 1, 1, 2, 3, 5, 8] then
    .error s!"Drake Fibonacci gtest expected sequence mismatch: {spec.expectedValues}"
  spec.logger.validate? p
  if spec.logger.construction != .explicitVectorLogSink then
    .error "Fibonacci gtest should construct VectorLogSink explicitly"

end FibonacciGtestSpec

def gtestSpec : FibonacciGtestSpec := {}

structure FibonacciState where
  current : Nat := 0
  previous : Nat := 1
  deriving Repr, BEq, Inhabited

namespace FibonacciState

def output (x : FibonacciState) : Nat :=
  x.current

def asFloatArray (x : FibonacciState) : Array Float :=
  #[x.current.toFloat, x.previous.toFloat]

def update (x : FibonacciState) : FibonacciState :=
  { current := x.current + x.previous, previous := x.current }

end FibonacciState

def initialState : FibonacciState := {}

def updateJacobian : Array (Array Float) :=
  #[#[1.0, 1.0], #[1.0, 0.0]]

def updateVjp (cotangentAfter : Array Float) : Array Float :=
  let currentCot := cotangentAfter.getD 0 0.0
  let previousCot := cotangentAfter.getD 1 0.0
  #[currentCot + previousCot, currentCot]

def updateVertex (idx : Nat) : VertexId :=
  12000 + idx

def updateData
    (idx : Nat) (p : FibonacciParams) (before after : FibonacciState) :
    ClockedUpdateData :=
  {
    time := idx.toFloat * p.period
    period := p.period
    stateBefore := before.asFloatArray
    stateAfter := after.asFloatArray
    updateJac := updateJacobian
    label := s!"fibonacci periodic update {idx}"
  }

structure FibonacciSample where
  n : Nat
  time : Float
  value : Nat
  deriving Repr, BEq, Inhabited

def sampleLine (sample : FibonacciSample) : String :=
  s!"{sample.n}: {sample.value} (t={sample.time})"

def sampleValues (samples : Array FibonacciSample) : Array Nat :=
  samples.map (fun sample => sample.value)

def sampleTimes (samples : Array FibonacciSample) : Array Float :=
  samples.map (fun sample => sample.time)

def logDataRow (samples : Array FibonacciSample) : Array Float :=
  samples.map (fun sample => sample.value.toFloat)

structure FibonacciExecutionBoundary where
  systemVertex : VertexId := 12100
  outputPortVertex : VertexId := 12101
  loggerVertex : VertexId := 12102
  simulatorVertex : VertexId := 12103
  stdoutVertex : VertexId := 12104
  sampleOrdering : String := "logger records initial Fn at t=0 before periodic update effects appear in the next sample"
  deriving Repr, BEq, Inhabited

def executionBoundary : FibonacciExecutionBoundary := {}

def executionGraph (p : FibonacciParams) (samples : Array FibonacciSample)
    (trace : DynamicEventTrace) : SkeletonGraph := Id.run do
  let b := executionBoundary
  let mut g :=
    SkeletonGraph.empty
      |>.addVertex { id := b.systemVertex, kind := .state .boundary, label := "FibonacciDifferenceEquation LeafSystem" }
      |>.addVertex { id := b.outputPortVertex, kind := .state .boundary, label := "Fn vector output port" }
      |>.addVertex { id := b.loggerVertex, kind := .checkpoint, label := "VectorLogSink sampled output log" }
      |>.addVertex { id := b.simulatorVertex, kind := .interval, label := s!"Simulator.AdvanceTo({p.finalTime})" }
      |>.addVertex { id := b.stdoutVertex, kind := .state .checkpoint, label := "run_fibonacci stdout rows" }
  g := g.addMove {
    kind := .localSchurBlock
    targets := #[b.systemVertex]
    reads := #[b.systemVertex]
    writes := #[b.outputPortVertex]
    exactness := .exact
    label := "DeclareVectorOutputPort(\"Fn\", 1, Output)"
  }
  for move in trace.moves do
    g := g.addMove move
  g := g.addMove {
    kind := .checkpointBoundary
    targets := #[b.loggerVertex]
    reads := #[b.outputPortVertex, b.simulatorVertex]
    writes := #[b.loggerVertex]
    exactness := .exact
    cost := { work := samples.size.toFloat, memory := samples.size.toFloat }
    label := "LogVectorOutput / VectorLogSink samples Fn at kPeriod"
  }
  g := g.addMove {
    kind := .checkpointBoundary
    targets := #[b.stdoutVertex]
    reads := #[b.loggerVertex]
    writes := #[b.stdoutVertex]
    exactness := .exact
    cost := { work := samples.size.toFloat, memory := 1.0 }
    label := "print n: Fn (t=time) rows"
  }
  return g

structure FibonacciResult where
  references : Array DrakeReference
  buildTargets : Array FibonacciBuildTarget
  systemSpec : FibonacciSystemSpec
  runnerSpec : FibonacciRunnerSpec
  gtestSpec : FibonacciGtestSpec
  params : FibonacciParams
  finalTime : Float
  initialState : FibonacciState
  finalState : FibonacciState
  samples : Array FibonacciSample
  logSampleTimes : Array Float
  logData : Array Float
  logLines : Array String
  trace : DynamicEventTrace
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def simulate? (p : FibonacciParams := params) : Except String FibonacciResult := do
  validateBuildTargets?
  systemSpec.validate?
  runnerSpec.validate? p
  gtestSpec.validate? p
  p.validate?
  let mut state := initialState
  let mut samples : Array FibonacciSample :=
    #[{ n := 0, time := 0.0, value := state.output }]
  let mut trace := DynamicEventTrace.empty
  for i in [:p.steps] do
    let before := state
    let after := before.update
    trace := trace.push
      (.clockedUpdate (updateVertex i) (updateData i p before after))
    state := after
    samples := samples.push
      { n := i + 1, time := (i + 1).toFloat * p.period, value := state.output }
  trace.validate?
  let graph := executionGraph p samples trace
  pure {
    references := drakeReferences
    buildTargets := buildTargets
    systemSpec := systemSpec
    runnerSpec := runnerSpec
    gtestSpec := gtestSpec
    params := p
    finalTime := p.finalTime
    initialState := initialState
    finalState := state
    samples := samples
    logSampleTimes := sampleTimes samples
    logData := logDataRow samples
    logLines := samples.map sampleLine
    trace := trace
    graph := graph
    moves := graph.moves
  }

def buildEndToEnd? (p : FibonacciParams := params) :
    Except String FibonacciResult :=
  simulate? p

end Tyr.EventSkeleton.Examples.Fibonacci
