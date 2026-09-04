import Tyr.EventSkeleton.Trace

/-!
# Drake Simple Systems and Van der Pol Event-Skeleton Examples

This ports Drake's small Systems examples that exercise continuous state,
periodic discrete updates, mixed continuous/discrete state, a pure Fibonacci
difference equation, and the van der Pol oscillator.

The important EventSkeleton addition is the clocked update vertex: a periodic
discrete update is a deterministic reset at a prescribed time, so reverse mode
needs the update VJP but no saltation timing scalar.
-/

namespace Tyr.EventSkeleton.Examples.SimpleSystems

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/BUILD.bazel"
      concept := "root examples package declaring the three simple system binaries plus the release model filegroup and install target"
    },
    {
      path := "../drake/examples/simple_continuous_time_system.cc"
      concept := "hello-world continuous LeafSystem with xdot = -x + x^3 and y = x"
    },
    {
      path := "../drake/examples/simple_discrete_time_system.cc"
      concept := "hello-world periodic discrete LeafSystem with x[n+1] = x[n]^3 and y = x"
    },
    {
      path := "../drake/examples/simple_mixed_continuous_and_discrete_time_system.cc"
      concept := "mixed LeafSystem with xd[n+1] = xd[n]^3 and xcdot = -xc + xc^3"
    },
    {
      path := "../drake/examples/fibonacci/fibonacci_difference_equation.h"
      concept := "pure discrete second-order Fibonacci recurrence encoded as two first-order states"
    },
    {
      path := "../drake/examples/fibonacci/run_fibonacci.cc"
      concept := "logs Fibonacci output at the periodic update time scale"
    },
    {
      path := "../drake/examples/van_der_pol/van_der_pol.h"
      concept := "declares the van der Pol oscillator state, outputs, parameter, and mu >= 0 constraint"
    },
    {
      path := "../drake/examples/van_der_pol/van_der_pol.cc"
      concept := "implements qdot and qddot = -mu(q^2 - 1)qdot - q, plus the default limit-cycle rollout"
    }
  ]

inductive SimpleSystemBuildTargetKind where
  | ccBinary
  deriving Repr, BEq, Inhabited

structure SimpleSystemBuildTarget where
  kind : SimpleSystemBuildTargetKind := .ccBinary
  name : String
  srcs : Array String := #[]
  deps : Array String := #[]
  addTestRule : Bool := true
  deriving Repr, BEq, Inhabited

namespace SimpleSystemBuildTarget

def hasDep (target : SimpleSystemBuildTarget) (dep : String) : Bool :=
  target.deps.any (fun actual => actual == dep)

def validate? (target : SimpleSystemBuildTarget) : Except String Unit := do
  if target.kind != .ccBinary then
    .error s!"SimpleSystems target {target.name} should be a drake_cc_binary"
  if target.name.isEmpty then
    .error "SimpleSystems BUILD target name cannot be empty"
  if !target.addTestRule then
    .error s!"SimpleSystems target {target.name} should preserve Drake's add_test_rule"
  if !target.hasDep "//systems/analysis:simulator" then
    .error s!"SimpleSystems target {target.name} should depend on Simulator"
  match target.name with
  | "simple_continuous_time_system" =>
      if target.srcs != #["simple_continuous_time_system.cc"] then
        .error s!"simple_continuous_time_system should compile its matching cc file, got {target.srcs}"
      if !target.hasDep "//systems/framework:vector_system" then
        .error "simple_continuous_time_system should depend on VectorSystem"
  | "simple_discrete_time_system" =>
      if target.srcs != #["simple_discrete_time_system.cc"] then
        .error s!"simple_discrete_time_system should compile its matching cc file, got {target.srcs}"
      if !target.hasDep "//systems/framework:vector_system" then
        .error "simple_discrete_time_system should depend on VectorSystem"
  | "simple_mixed_continuous_and_discrete_time_system" =>
      if target.srcs != #["simple_mixed_continuous_and_discrete_time_system.cc"] then
        .error s!"simple_mixed_continuous_and_discrete_time_system should compile its matching cc file, got {target.srcs}"
      if !target.hasDep "//systems/framework:leaf_system" then
        .error "simple_mixed_continuous_and_discrete_time_system should depend on LeafSystem"
  | other =>
      .error s!"Unexpected SimpleSystems BUILD target {other}"

end SimpleSystemBuildTarget

def simpleSystemBuildTargets : Array SimpleSystemBuildTarget :=
  #[
    {
      name := "simple_continuous_time_system"
      srcs := #["simple_continuous_time_system.cc"]
      deps := #[
        "//systems/analysis:simulator",
        "//systems/framework:vector_system"
      ]
    },
    {
      name := "simple_discrete_time_system"
      srcs := #["simple_discrete_time_system.cc"]
      deps := #[
        "//systems/analysis:simulator",
        "//systems/framework:vector_system"
      ]
    },
    {
      name := "simple_mixed_continuous_and_discrete_time_system"
      srcs := #["simple_mixed_continuous_and_discrete_time_system.cc"]
      deps := #[
        "//systems/analysis:simulator",
        "//systems/framework:leaf_system"
      ]
    }
  ]

def validateSimpleSystemBuildTargets?
    (targets : Array SimpleSystemBuildTarget := simpleSystemBuildTargets) :
    Except String Unit := do
  if targets.size != 3 then
    .error s!"../drake/examples/BUILD.bazel should declare three simple system binaries, got {targets.size}"
  if !(targets.any (fun target => target.name == "simple_continuous_time_system")) then
    .error "missing simple_continuous_time_system binary"
  if !(targets.any (fun target => target.name == "simple_discrete_time_system")) then
    .error "missing simple_discrete_time_system binary"
  if !(targets.any (fun target => target.name == "simple_mixed_continuous_and_discrete_time_system")) then
    .error "missing simple_mixed_continuous_and_discrete_time_system binary"
  for target in targets do
    target.validate?

def expectedInstalledModelPackages : Array String :=
  #[
    "//examples/acrobot",
    "//examples/kuka_iiwa_arm/models",
    "//examples/multibody/cart_pole",
    "//examples/pendulum",
    "//examples/planar_gripper",
    "//examples/quadrotor"
  ]

structure ExamplesModelPackageBoundary where
  modelPackages : Array String := expectedInstalledModelPackages
  modelsTargetName : String := "models"
  modelsVisibility : String := "//visibility:public"
  installTargetName : String := "install"
  installVisibility : String := "//visibility:public"
  deriving Repr, BEq, Inhabited

namespace ExamplesModelPackageBoundary

def modelDeps (boundary : ExamplesModelPackageBoundary) : Array String :=
  boundary.modelPackages.map (fun package => package ++ ":models")

def installDeps (boundary : ExamplesModelPackageBoundary) : Array String :=
  boundary.modelPackages.map (fun package => package ++ ":install_data")

def validate? (boundary : ExamplesModelPackageBoundary) : Except String Unit := do
  if boundary.modelPackages != expectedInstalledModelPackages then
    .error s!"installed examples model packages should match Drake's root BUILD list, got {boundary.modelPackages}"
  if boundary.modelsTargetName != "models" then
    .error s!"root examples filegroup should be named models, got {boundary.modelsTargetName}"
  if boundary.installTargetName != "install" then
    .error s!"root examples install target should be named install, got {boundary.installTargetName}"
  if boundary.modelsVisibility != "//visibility:public" then
    .error s!"models filegroup should be public, got {boundary.modelsVisibility}"
  if boundary.installVisibility != "//visibility:public" then
    .error s!"install target should be public, got {boundary.installVisibility}"
  if boundary.modelDeps.size != expectedInstalledModelPackages.size then
    .error "models filegroup should contain one :models dep per installed package"
  if boundary.installDeps.size != expectedInstalledModelPackages.size then
    .error "install target should contain one :install_data dep per installed package"

end ExamplesModelPackageBoundary

def examplesModelPackageBoundary : ExamplesModelPackageBoundary := {}

def examplesBuildFileVertex : VertexId := 4500

def simpleContinuousBinaryVertex : VertexId := 4501

def simpleDiscreteBinaryVertex : VertexId := 4502

def simpleMixedBinaryVertex : VertexId := 4503

def examplesModelsFilegroupVertex : VertexId := 4504

def examplesInstallTargetVertex : VertexId := 4505

def SimpleSystemBuildTarget.vertex (target : SimpleSystemBuildTarget) :
    VertexId :=
  match target.name with
  | "simple_continuous_time_system" => simpleContinuousBinaryVertex
  | "simple_discrete_time_system" => simpleDiscreteBinaryVertex
  | "simple_mixed_continuous_and_discrete_time_system" => simpleMixedBinaryVertex
  | _ => examplesBuildFileVertex

def SimpleSystemBuildTarget.buildMove (target : SimpleSystemBuildTarget) :
    SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target.vertex]
    reads := #[examplesBuildFileVertex]
    writes := #[target.vertex]
    exactness := .exact
    cost := { work := (target.srcs.size + target.deps.size).toFloat, memory := 1.0 }
    label := s!"../drake/examples/BUILD.bazel drake_cc_binary({target.name}) add_test_rule={target.addTestRule}"
  }

def examplesModelsFilegroupMove
    (boundary : ExamplesModelPackageBoundary := examplesModelPackageBoundary) :
    SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[examplesModelsFilegroupVertex]
    reads := #[examplesBuildFileVertex]
    writes := #[examplesModelsFilegroupVertex]
    exactness := .exact
    cost := { work := boundary.modelPackages.size.toFloat, memory := 1.0 }
    label := s!"filegroup(name=\"{boundary.modelsTargetName}\") aggregates installed example :models targets"
  }

def examplesInstallTargetMove
    (boundary : ExamplesModelPackageBoundary := examplesModelPackageBoundary) :
    SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[examplesInstallTargetVertex]
    reads := #[examplesBuildFileVertex, examplesModelsFilegroupVertex]
    writes := #[examplesInstallTargetVertex]
    exactness := .exact
    cost := { work := boundary.modelPackages.size.toFloat, memory := 1.0 }
    label := s!"install(name=\"{boundary.installTargetName}\") aggregates installed example :install_data targets"
  }

def examplesPackageGraph
    (targets : Array SimpleSystemBuildTarget := simpleSystemBuildTargets)
    (boundary : ExamplesModelPackageBoundary := examplesModelPackageBoundary) :
    SkeletonGraph := Id.run do
  let mut g :=
    SkeletonGraph.empty
      |>.addVertex { id := examplesBuildFileVertex, kind := .state .boundary, label := "../drake/examples/BUILD.bazel" }
      |>.addVertex { id := examplesModelsFilegroupVertex, kind := .state .boundary, label := s!"filegroup:{boundary.modelsTargetName}" }
      |>.addVertex { id := examplesInstallTargetVertex, kind := .state .boundary, label := s!"install:{boundary.installTargetName}" }
  for target in targets do
    g := g.addVertex { id := target.vertex, kind := .state .boundary, label := target.name }
    g := g.addMove target.buildMove
  g := g.addMove (examplesModelsFilegroupMove boundary)
  g := g.addMove (examplesInstallTargetMove boundary)
  return g

structure ExamplesPackageBoundaryResult where
  references : Array DrakeReference
  buildTargets : Array SimpleSystemBuildTarget
  modelBoundary : ExamplesModelPackageBoundary
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildExamplesPackageBoundary?
    (targets : Array SimpleSystemBuildTarget := simpleSystemBuildTargets)
    (boundary : ExamplesModelPackageBoundary := examplesModelPackageBoundary) :
    Except String ExamplesPackageBoundaryResult := do
  validateSimpleSystemBuildTargets? targets
  boundary.validate?
  let graph := examplesPackageGraph targets boundary
  pure {
    references := drakeReferences
    buildTargets := targets
    modelBoundary := boundary
    graph := graph
    moves := graph.moves
  }

private def cube (x : Float) : Float :=
  x * x * x

private def approxPeriodStepCount (finalTime stepSize : Float) (steps : Nat) : Float :=
  Float.abs (steps.toFloat * stepSize - finalTime)

def cubicDerivative (x : Float) : Float :=
  -x + cube x

def cubicUpdate (x : Float) : Float :=
  cube x

def cubicUpdateJacobian (x : Float) : Array (Array Float) :=
  #[#[3.0 * x * x]]

def rk4ScalarStep (f : Float → Float) (dt x : Float) : Float :=
  let k1 := f x
  let k2 := f (x + 0.5 * dt * k1)
  let k3 := f (x + 0.5 * dt * k2)
  let k4 := f (x + dt * k3)
  x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

structure ScalarRollout where
  initial : Float
  final : Float
  samples : Array Float
  deriving Repr, Inhabited

def rolloutScalarRk4
    (f : Float → Float)
    (stepSize : Float)
    (steps : Nat)
    (initial : Float) : ScalarRollout := Id.run do
  let mut x := initial
  let mut samples := #[initial]
  for _ in [:steps] do
    x := rk4ScalarStep f stepSize x
    samples := samples.push x
  return { initial := initial, final := x, samples := samples }

/-! ## Simple continuous system -/

structure SimpleContinuousParams where
  initial : Float := 0.9
  finalTime : Float := 10.0
  stepSize : Float := 0.01
  steps : Nat := 1000
  deriving Repr, Inhabited

def continuousParams : SimpleContinuousParams := {}

namespace SimpleContinuousParams

def validate? (p : SimpleContinuousParams) : Except String Unit := do
  if !(Float.isFinite p.initial) then
    .error s!"continuous initial state must be finite, got {p.initial}"
  if !(Float.isFinite p.finalTime) || p.finalTime <= 0.0 then
    .error s!"continuous final time must be positive and finite, got {p.finalTime}"
  if !(Float.isFinite p.stepSize) || p.stepSize <= 0.0 then
    .error s!"continuous step size must be positive and finite, got {p.stepSize}"
  if p.steps == 0 then
    .error "continuous rollout requires at least one step"
  if approxPeriodStepCount p.finalTime p.stepSize p.steps > 1.0e-12 then
    .error s!"continuous step count {p.steps} does not match final time {p.finalTime} with step size {p.stepSize}"

end SimpleContinuousParams

def continuousClosedForm (initial time : Float) : Float :=
  if initial == 0.0 then
    0.0
  else
    let z0 := initial * initial
    let denom := 1.0 + ((1.0 / z0) - 1.0) * Float.exp (2.0 * time)
    let mag := Float.sqrt (1.0 / denom)
    if initial < 0.0 then -mag else mag

def continuousSegment (p : SimpleContinuousParams := continuousParams) :
    AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.finalTime
    tAfter := p.finalTime
    label := "simple continuous cubic interval"
  }

structure ContinuousResult where
  references : Array DrakeReference
  params : SimpleContinuousParams
  rollout : ScalarRollout
  closedFormFinal : Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def simulateContinuous? (p : SimpleContinuousParams := continuousParams) :
    Except String ContinuousResult := do
  p.validate?
  let rollout := rolloutScalarRk4 cubicDerivative p.stepSize p.steps p.initial
  let trace := DynamicEventTrace.empty.push (.interval (continuousSegment p))
  trace.validate?
  pure {
    references := drakeReferences
    params := p
    rollout := rollout
    closedFormFinal := continuousClosedForm p.initial p.finalTime
    trace := trace
    moves := trace.moves
  }

/-! ## Simple discrete system -/

structure SimpleDiscreteParams where
  initial : Float := 0.99
  period : Float := 1.0
  steps : Nat := 10
  deriving Repr, Inhabited

def discreteParams : SimpleDiscreteParams := {}

namespace SimpleDiscreteParams

def validate? (p : SimpleDiscreteParams) : Except String Unit := do
  if !(Float.isFinite p.initial) then
    .error s!"discrete initial state must be finite, got {p.initial}"
  if !(Float.isFinite p.period) || p.period <= 0.0 then
    .error s!"discrete period must be positive and finite, got {p.period}"
  if p.steps == 0 then
    .error "discrete rollout requires at least one update"

end SimpleDiscreteParams

def discreteUpdateVertex (idx : Nat) : VertexId :=
  1000 + idx

def clockedScalarUpdateData
    (time period before after : Float)
    (label : String) : ClockedUpdateData :=
  {
    time := time
    period := period
    stateBefore := #[before]
    stateAfter := #[after]
    updateJac := cubicUpdateJacobian before
    label := label
  }

structure DiscreteResult where
  references : Array DrakeReference
  params : SimpleDiscreteParams
  samples : Array Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def simulateDiscrete? (p : SimpleDiscreteParams := discreteParams) :
    Except String DiscreteResult := do
  p.validate?
  let mut x := p.initial
  let mut samples := #[x]
  let mut trace := DynamicEventTrace.empty
  for i in [:p.steps] do
    let before := x
    let after := cubicUpdate before
    let time := (i + 1).toFloat * p.period
    trace := trace.push
      (.clockedUpdate (discreteUpdateVertex i)
        (clockedScalarUpdateData time p.period before after
          s!"simple discrete cubic update {i}"))
    x := after
    samples := samples.push x
  trace.validate?
  pure {
    references := drakeReferences
    params := p
    samples := samples
    trace := trace
    moves := trace.moves
  }

/-! ## Mixed continuous/discrete system -/

structure MixedParams where
  discreteInitial : Float := 0.99
  continuousInitial : Float := 0.9
  period : Float := 1.0
  finalTime : Float := 10.0
  substepsPerPeriod : Nat := 100
  periods : Nat := 10
  deriving Repr, Inhabited

def mixedParams : MixedParams := {}

namespace MixedParams

def continuousStepSize (p : MixedParams) : Float :=
  p.period / p.substepsPerPeriod.toFloat

def validate? (p : MixedParams) : Except String Unit := do
  if !(Float.isFinite p.discreteInitial) then
    .error s!"mixed discrete initial state must be finite, got {p.discreteInitial}"
  if !(Float.isFinite p.continuousInitial) then
    .error s!"mixed continuous initial state must be finite, got {p.continuousInitial}"
  if !(Float.isFinite p.period) || p.period <= 0.0 then
    .error s!"mixed period must be positive and finite, got {p.period}"
  if !(Float.isFinite p.finalTime) || p.finalTime <= 0.0 then
    .error s!"mixed final time must be positive and finite, got {p.finalTime}"
  if p.substepsPerPeriod == 0 then
    .error "mixed rollout requires at least one continuous substep per period"
  if p.periods == 0 then
    .error "mixed rollout requires at least one discrete period"
  if approxPeriodStepCount p.finalTime p.period p.periods > 1.0e-12 then
    .error s!"mixed period count {p.periods} does not match final time {p.finalTime} with period {p.period}"

end MixedParams

structure MixedState where
  discrete : Float
  continuous : Float
  deriving Repr, Inhabited

namespace MixedState

def output (x : MixedState) : Array Float :=
  #[x.discrete, x.continuous]

def isFinite (x : MixedState) : Bool :=
  Float.isFinite x.discrete && Float.isFinite x.continuous

end MixedState

def mixedInitialState (p : MixedParams := mixedParams) : MixedState :=
  { discrete := p.discreteInitial, continuous := p.continuousInitial }

def mixedIntervalSegment (idx : Nat) (p : MixedParams) : AcceptedStepSegment :=
  {
    id := 2000 + idx
    attemptIndex := idx
    tStart := idx.toFloat * p.period
    tAttempt := (idx + 1).toFloat * p.period
    tAfter := (idx + 1).toFloat * p.period
    label := s!"simple mixed continuous interval {idx}"
  }

def mixedUpdateVertex (idx : Nat) : VertexId :=
  3000 + idx

structure MixedResult where
  references : Array DrakeReference
  params : MixedParams
  initialState : MixedState
  finalState : MixedState
  samples : Array MixedState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def simulateMixed? (p : MixedParams := mixedParams) : Except String MixedResult := do
  p.validate?
  let mut state := mixedInitialState p
  let mut samples := #[state]
  let mut trace := DynamicEventTrace.empty
  let dt := p.continuousStepSize
  for i in [:p.periods] do
    trace := trace.push (.interval (mixedIntervalSegment i p))
    for _ in [:p.substepsPerPeriod] do
      state := { state with
        continuous := rk4ScalarStep cubicDerivative dt state.continuous }
    let before := state.discrete
    let after := cubicUpdate before
    trace := trace.push
      (.clockedUpdate (mixedUpdateVertex i)
        (clockedScalarUpdateData ((i + 1).toFloat * p.period) p.period before after
          s!"simple mixed discrete update {i}"))
    state := { state with discrete := after }
    samples := samples.push state
  trace.validate?
  pure {
    references := drakeReferences
    params := p
    initialState := mixedInitialState p
    finalState := state
    samples := samples
    trace := trace
    moves := trace.moves
  }

/-! ## Fibonacci difference equation -/

structure FibonacciParams where
  period : Float := 0.25
  steps : Nat := 10
  deriving Repr, Inhabited

def fibonacciParams : FibonacciParams := {}

namespace FibonacciParams

def validate? (p : FibonacciParams) : Except String Unit := do
  if !(Float.isFinite p.period) || p.period <= 0.0 then
    .error s!"Fibonacci period must be positive and finite, got {p.period}"

end FibonacciParams

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

def fibonacciUpdateVertex (idx : Nat) : VertexId :=
  4000 + idx

def fibonacciUpdateData
    (idx : Nat) (p : FibonacciParams) (before after : FibonacciState) :
    ClockedUpdateData :=
  {
    time := idx.toFloat * p.period
    period := p.period
    stateBefore := before.asFloatArray
    stateAfter := after.asFloatArray
    updateJac := #[#[1.0, 1.0], #[1.0, 0.0]]
    label := s!"fibonacci update {idx}"
  }

structure FibonacciSample where
  n : Nat
  time : Float
  value : Nat
  deriving Repr, BEq, Inhabited

structure FibonacciResult where
  references : Array DrakeReference
  params : FibonacciParams
  samples : Array FibonacciSample
  finalState : FibonacciState
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def simulateFibonacci? (p : FibonacciParams := fibonacciParams) :
    Except String FibonacciResult := do
  p.validate?
  let mut state : FibonacciState := {}
  let mut samples : Array FibonacciSample :=
    #[{ n := 0, time := 0.0, value := state.output }]
  let mut trace := DynamicEventTrace.empty
  for i in [:p.steps] do
    let before := state
    let after := before.update
    trace := trace.push
      (.clockedUpdate (fibonacciUpdateVertex i)
        (fibonacciUpdateData i p before after))
    state := after
    samples := samples.push
      { n := i + 1, time := (i + 1).toFloat * p.period, value := state.output }
  trace.validate?
  pure {
    references := drakeReferences
    params := p
    samples := samples
    finalState := state
    trace := trace
    moves := trace.moves
  }

/-! ## Van der Pol oscillator -/

structure VanDerPolParams where
  mu : Float := 1.0
  initialQ : Float := -0.1144
  initialQdot : Float := 2.0578
  finalTime : Float := 6.667
  stepSize : Float := 0.001
  steps : Nat := 6667
  deriving Repr, Inhabited

def vanDerPolParams : VanDerPolParams := {}

namespace VanDerPolParams

def validate? (p : VanDerPolParams) : Except String Unit := do
  if !(Float.isFinite p.mu) || p.mu < 0.0 then
    .error s!"van der Pol mu must be nonnegative and finite, got {p.mu}"
  if !(Float.isFinite p.initialQ) || !(Float.isFinite p.initialQdot) then
    .error s!"van der Pol initial state must be finite, got q={p.initialQ}, qdot={p.initialQdot}"
  if !(Float.isFinite p.finalTime) || p.finalTime <= 0.0 then
    .error s!"van der Pol final time must be positive and finite, got {p.finalTime}"
  if !(Float.isFinite p.stepSize) || p.stepSize <= 0.0 then
    .error s!"van der Pol step size must be positive and finite, got {p.stepSize}"
  if p.steps == 0 then
    .error "van der Pol rollout requires at least one integration step"
  if approxPeriodStepCount p.finalTime p.stepSize p.steps > 1.0e-12 then
    .error s!"van der Pol step count {p.steps} does not match final time {p.finalTime} with step size {p.stepSize}"

end VanDerPolParams

structure VanDerPolState where
  q : Float := 0.0
  qdot : Float := 0.0
  deriving Repr, Inhabited

namespace VanDerPolState

def asArray (x : VanDerPolState) : Array Float :=
  #[x.q, x.qdot]

def isFinite (x : VanDerPolState) : Bool :=
  Float.isFinite x.q && Float.isFinite x.qdot

end VanDerPolState

def vanDerPolInitialState (p : VanDerPolParams := vanDerPolParams) :
    VanDerPolState :=
  { q := p.initialQ, qdot := p.initialQdot }

def vanDerPolDerivative (p : VanDerPolParams) (x : VanDerPolState) :
    VanDerPolState :=
  {
    q := x.qdot
    qdot := -p.mu * (x.q * x.q - 1.0) * x.qdot - x.q
  }

def rk4VanDerPolStep
    (p : VanDerPolParams) (dt : Float) (x : VanDerPolState) :
    VanDerPolState :=
  let addScaled (x k : VanDerPolState) (s : Float) : VanDerPolState :=
    { q := x.q + s * k.q, qdot := x.qdot + s * k.qdot }
  let combine
      (x k1 k2 k3 k4 : VanDerPolState) (dt : Float) : VanDerPolState :=
    {
      q := x.q + dt * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q) / 6.0
      qdot := x.qdot + dt * (k1.qdot + 2.0 * k2.qdot + 2.0 * k3.qdot + k4.qdot) / 6.0
    }
  let k1 := vanDerPolDerivative p x
  let k2 := vanDerPolDerivative p (addScaled x k1 (0.5 * dt))
  let k3 := vanDerPolDerivative p (addScaled x k2 (0.5 * dt))
  let k4 := vanDerPolDerivative p (addScaled x k3 dt)
  combine x k1 k2 k3 k4 dt

def vanDerPolSegment (p : VanDerPolParams := vanDerPolParams) :
    AcceptedStepSegment :=
  {
    id := 5000
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.finalTime
    tAfter := p.finalTime
    label := "van der Pol limit-cycle interval"
  }

structure VanDerPolResult where
  references : Array DrakeReference
  params : VanDerPolParams
  initialState : VanDerPolState
  finalState : VanDerPolState
  samples : Array VanDerPolState
  positionOutput : Float
  fullStateOutput : Array Float
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def simulateVanDerPol? (p : VanDerPolParams := vanDerPolParams) :
    Except String VanDerPolResult := do
  p.validate?
  let mut state := vanDerPolInitialState p
  let mut samples := #[state]
  for _ in [:p.steps] do
    state := rk4VanDerPolStep p p.stepSize state
    samples := samples.push state
  let trace := DynamicEventTrace.empty.push (.interval (vanDerPolSegment p))
  trace.validate?
  pure {
    references := drakeReferences
    params := p
    initialState := vanDerPolInitialState p
    finalState := state
    samples := samples
    positionOutput := state.q
    fullStateOutput := state.asArray
    trace := trace
    moves := trace.moves
  }

structure SimpleSystemsResult where
  references : Array DrakeReference
  packageBoundary : ExamplesPackageBoundaryResult
  continuous : ContinuousResult
  discrete : DiscreteResult
  mixed : MixedResult
  fibonacci : FibonacciResult
  vanDerPol : VanDerPolResult
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd?
    (pContinuous : SimpleContinuousParams := continuousParams)
    (pDiscrete : SimpleDiscreteParams := discreteParams)
    (pMixed : MixedParams := mixedParams)
    (pFibonacci : FibonacciParams := fibonacciParams)
    (pVanDerPol : VanDerPolParams := vanDerPolParams) :
    Except String SimpleSystemsResult := do
  let packageBoundary ← buildExamplesPackageBoundary?
  let continuous ← simulateContinuous? pContinuous
  let discrete ← simulateDiscrete? pDiscrete
  let mixed ← simulateMixed? pMixed
  let fibonacci ← simulateFibonacci? pFibonacci
  let vanDerPol ← simulateVanDerPol? pVanDerPol
  let moves :=
    packageBoundary.moves ++
    continuous.moves ++
    discrete.moves ++
    mixed.moves ++
    fibonacci.moves ++
    vanDerPol.moves
  pure {
    references := drakeReferences
    packageBoundary := packageBoundary
    continuous := continuous
    discrete := discrete
    mixed := mixed
    fibonacci := fibonacci
    vanDerPol := vanDerPol
    moves := moves
  }

end Tyr.EventSkeleton.Examples.SimpleSystems
