import Tyr.EventSkeleton.Examples.SimpleSystems

/-!
# Drake Van der Pol Event-Skeleton Example

This is the dedicated port of `../drake/examples/van_der_pol`.  The nonlinear
ODE itself is shared with `SimpleSystems`; this module adds the Drake
`CalcLimitCycle` test tolerances and the `plot_limit_cycle.py` executable
boundary as first-class event-skeleton metadata.
-/

namespace Tyr.EventSkeleton.Examples.VanDerPol

open Tyr.EventSkeleton

abbrev DrakeReference := Tyr.EventSkeleton.Examples.SimpleSystems.DrakeReference
abbrev VanDerPolParams := Tyr.EventSkeleton.Examples.SimpleSystems.VanDerPolParams
abbrev VanDerPolState := Tyr.EventSkeleton.Examples.SimpleSystems.VanDerPolState
abbrev VanDerPolResult := Tyr.EventSkeleton.Examples.SimpleSystems.VanDerPolResult

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/van_der_pol/van_der_pol.h"
      concept := "declares the LeafSystem state (q, qdot), y0/y1 output ports, mu parameter, and mu >= 0 constraint"
    },
    {
      path := "../drake/examples/van_der_pol/van_der_pol.cc"
      concept := "implements qdot and qddot = -mu(q^2 - 1)qdot - q plus CalcLimitCycle()"
    },
    {
      path := "../drake/examples/van_der_pol/test/van_der_pol_test.cc"
      concept := "checks scalar conversion and the default limit-cycle closure tolerances"
    },
    {
      path := "../drake/examples/van_der_pol/plot_limit_cycle.py"
      concept := "renders CalcLimitCycle() to plot_limit_cycle.png using the Agg backend"
    },
    {
      path := "../drake/examples/van_der_pol/BUILD.bazel"
      concept := "defines the C++ LeafSystem library, C++ test, and Python plot_limit_cycle binary"
    }
  ]

inductive VanDerPolBuildTargetKind where
  | ccLibrary
  | pyBinary
  | ccGoogletest
  deriving Repr, BEq, Inhabited

structure VanDerPolBuildTarget where
  kind : VanDerPolBuildTargetKind
  name : String
  srcs : Array String := #[]
  hdrs : Array String := #[]
  deps : Array String := #[]
  publicVisibility : Bool := false
  addTestRule : Bool := false
  deriving Repr, BEq, Inhabited

namespace VanDerPolBuildTarget

def hasDep (target : VanDerPolBuildTarget) (dep : String) : Bool :=
  target.deps.any (fun actual => actual == dep)

def validate? (target : VanDerPolBuildTarget) : Except String Unit := do
  if target.name.isEmpty then
    .error "Van der Pol BUILD target name cannot be empty"
  match target.kind with
  | .ccLibrary =>
      if target.name != "van_der_pol" then
        .error s!"Van der Pol library target should be named van_der_pol, got {target.name}"
      if target.srcs != #["van_der_pol.cc"] || target.hdrs != #["van_der_pol.h"] then
        .error s!"Van der Pol library should compile van_der_pol.cc and expose van_der_pol.h, got srcs={target.srcs}, hdrs={target.hdrs}"
      if !target.publicVisibility then
        .error "Van der Pol library should have public visibility"
      if !target.hasDep "//systems/framework:leaf_system" then
        .error "Van der Pol library should depend on LeafSystem"
      if !target.hasDep "//systems/framework:system_constraint" then
        .error "Van der Pol library should depend on SystemConstraint for mu >= 0"
      if !target.hasDep "//systems/primitives:vector_log_sink" then
        .error "Van der Pol library should depend on VectorLogSink for CalcLimitCycle logging"
  | .pyBinary =>
      if target.name != "plot_limit_cycle" then
        .error s!"Van der Pol Python binary should be plot_limit_cycle, got {target.name}"
      if target.srcs != #["plot_limit_cycle.py"] then
        .error s!"Van der Pol Python binary should package plot_limit_cycle.py, got {target.srcs}"
      if !target.hasDep "//bindings/pydrake" then
        .error "Van der Pol plot binary should depend on pydrake"
      if !target.addTestRule then
        .error "Van der Pol plot_limit_cycle py_binary should add a test rule"
  | .ccGoogletest =>
      if target.name != "van_der_pol_test" then
        .error s!"Van der Pol gtest target should be van_der_pol_test, got {target.name}"
      if !target.hasDep ":van_der_pol" then
        .error "Van der Pol gtest should depend on the local van_der_pol library"
      if !target.hasDep "//systems/framework/test_utilities:scalar_conversion" then
        .error "Van der Pol gtest should depend on scalar_conversion test utilities"

end VanDerPolBuildTarget

def buildTargets : Array VanDerPolBuildTarget :=
  #[
    {
      kind := .ccLibrary
      name := "van_der_pol"
      srcs := #["van_der_pol.cc"]
      hdrs := #["van_der_pol.h"]
      deps := #[
        "//systems/analysis:simulator",
        "//systems/framework:diagram_builder",
        "//systems/framework:leaf_system",
        "//systems/framework:system_constraint",
        "//systems/framework:vector",
        "//systems/primitives:vector_log_sink"
      ]
      publicVisibility := true
    },
    {
      kind := .pyBinary
      name := "plot_limit_cycle"
      srcs := #["plot_limit_cycle.py"]
      deps := #["//bindings/pydrake"]
      addTestRule := true
    },
    {
      kind := .ccGoogletest
      name := "van_der_pol_test"
      deps := #[
        ":van_der_pol",
        "//systems/framework/test_utilities:scalar_conversion"
      ]
    }
  ]

def validateBuildTargets? (targets : Array VanDerPolBuildTarget := buildTargets) :
    Except String Unit := do
  if targets.size != 3 then
    .error s!"Van der Pol BUILD.bazel should declare library, plot binary, and gtest targets, got {targets.size}"
  if !(targets.any (fun target => target.name == "van_der_pol")) then
    .error "missing van_der_pol library target"
  if !(targets.any (fun target => target.name == "plot_limit_cycle")) then
    .error "missing plot_limit_cycle Python binary target"
  if !(targets.any (fun target => target.name == "van_der_pol_test")) then
    .error "missing van_der_pol_test gtest target"
  for target in targets do
    target.validate?

structure VanDerPolOutputPortSpec where
  index : Nat
  name : String
  size : Nat
  getter : String
  source : String
  deriving Repr, BEq, Inhabited

structure VanDerPolNumericParameterSpec where
  index : Nat := 0
  name : String := "mu"
  size : Nat := 1
  defaultValue : Float := 1.0
  deriving Repr, BEq, Inhabited

structure VanDerPolInequalityConstraintSpec where
  name : String := "mu >= 0"
  parameterIndex : Nat := 0
  lowerBound : Float := 0.0
  hasUpperBound : Bool := false
  deriving Repr, BEq, Inhabited

structure VanDerPolSystemSpec where
  systemName : String := "VanDerPolOscillator"
  continuousPositionSize : Nat := 1
  continuousVelocitySize : Nat := 1
  continuousMiscSize : Nat := 0
  outputPorts : Array VanDerPolOutputPortSpec := #[
    {
      index := 0
      name := "y0"
      size := 1
      getter := "get_position_output_port"
      source := "CopyPositionToOutput"
    },
    {
      index := 1
      name := "y1"
      size := 2
      getter := "get_full_state_output_port"
      source := "DeclareStateOutputPort"
    }
  ]
  numericParameter : VanDerPolNumericParameterSpec := {}
  inequalityConstraint : VanDerPolInequalityConstraintSpec := {}
  deriving Repr, BEq, Inhabited

namespace VanDerPolSystemSpec

def validate? (spec : VanDerPolSystemSpec) : Except String Unit := do
  if spec.systemName != "VanDerPolOscillator" then
    .error s!"Unexpected Van der Pol system name {spec.systemName}"
  if spec.continuousPositionSize != 1 ||
      spec.continuousVelocitySize != 1 ||
      spec.continuousMiscSize != 0 then
    .error s!"Van der Pol should DeclareContinuousState(1, 1, 0), got ({spec.continuousPositionSize}, {spec.continuousVelocitySize}, {spec.continuousMiscSize})"
  if spec.outputPorts.size != 2 then
    .error s!"Van der Pol should declare y0 and y1 output ports, got {spec.outputPorts.size}"
  let position? := spec.outputPorts.find? (fun port => port.index == 0)
  let fullState? := spec.outputPorts.find? (fun port => port.index == 1)
  match position?, fullState? with
  | some position, some fullState =>
      if position.name != "y0" || position.size != 1 ||
          position.getter != "get_position_output_port" ||
          position.source != "CopyPositionToOutput" then
        .error s!"Van der Pol y0 port should be CopyPositionToOutput size 1, got {reprStr position}"
      if fullState.name != "y1" || fullState.size != 2 ||
          fullState.getter != "get_full_state_output_port" ||
          fullState.source != "DeclareStateOutputPort" then
        .error s!"Van der Pol y1 port should be a size-2 state output port, got {reprStr fullState}"
  | _, _ => .error "Van der Pol output ports should have indices 0 and 1"
  if spec.numericParameter.index != 0 ||
      spec.numericParameter.name != "mu" ||
      spec.numericParameter.size != 1 ||
      Float.abs (spec.numericParameter.defaultValue - 1.0) > 1.0e-12 then
    .error s!"Van der Pol numeric parameter should be mu[0] with default 1, got {reprStr spec.numericParameter}"
  if spec.inequalityConstraint.name != "mu >= 0" ||
      spec.inequalityConstraint.parameterIndex != 0 ||
      Float.abs spec.inequalityConstraint.lowerBound > 1.0e-12 ||
      spec.inequalityConstraint.hasUpperBound then
    .error s!"Van der Pol inequality constraint should be mu >= 0 with no upper bound, got {reprStr spec.inequalityConstraint}"

end VanDerPolSystemSpec

def systemSpec : VanDerPolSystemSpec := {}

def params : VanDerPolParams :=
  Tyr.EventSkeleton.Examples.SimpleSystems.vanDerPolParams

def derivative (p : VanDerPolParams) (x : VanDerPolState) : VanDerPolState :=
  Tyr.EventSkeleton.Examples.SimpleSystems.vanDerPolDerivative p x

def simulate? (p : VanDerPolParams := params) : Except String VanDerPolResult :=
  Tyr.EventSkeleton.Examples.SimpleSystems.simulateVanDerPol? p

structure ScalarConversionSpec where
  testPath : String := "../drake/examples/van_der_pol/test/van_der_pol_test.cc"
  doubleConvertible : Bool := true
  autodiffConvertible : Bool := true
  symbolicConvertible : Bool := true
  deriving Repr, Inhabited

namespace ScalarConversionSpec

def validate? (spec : ScalarConversionSpec) : Except String Unit := do
  if spec.testPath != "../drake/examples/van_der_pol/test/van_der_pol_test.cc" then
    .error s!"Van der Pol scalar-conversion test path mismatch: {spec.testPath}"
  if !spec.doubleConvertible then
    .error "Van der Pol port should preserve the default double scalar instantiation"
  if !spec.autodiffConvertible then
    .error "Van der Pol port should preserve AutoDiffXd scalar conversion"
  if !spec.symbolicConvertible then
    .error "Van der Pol port should preserve symbolic scalar conversion"

end ScalarConversionSpec

def scalarConversionSpec : ScalarConversionSpec := {}

structure LimitCycleTestSpec where
  testPath : String := "../drake/examples/van_der_pol/test/van_der_pol_test.cc"
  qTolerance : Float := 1.0e-2
  qdotTolerance : Float := 5.0e-3
  deriving Repr, Inhabited

namespace LimitCycleTestSpec

def validate? (spec : LimitCycleTestSpec) : Except String Unit := do
  if spec.testPath != "../drake/examples/van_der_pol/test/van_der_pol_test.cc" then
    .error s!"Van der Pol limit-cycle test path mismatch: {spec.testPath}"
  if !(Float.isFinite spec.qTolerance) || spec.qTolerance <= 0.0 then
    .error s!"Van der Pol q tolerance must be positive and finite, got {spec.qTolerance}"
  if !(Float.isFinite spec.qdotTolerance) || spec.qdotTolerance <= 0.0 then
    .error s!"Van der Pol qdot tolerance must be positive and finite, got {spec.qdotTolerance}"

end LimitCycleTestSpec

def limitCycleTestSpec : LimitCycleTestSpec := {}

structure LimitCyclePlotSpec where
  scriptPath : String := "../drake/examples/van_der_pol/plot_limit_cycle.py"
  backendEnv : String := "Agg"
  outputPath : String := "plot_limit_cycle.png"
  xMin : Float := -2.5
  xMax : Float := 2.5
  yMin : Float := -3.0
  yMax : Float := 3.0
  xLabel : String := "q"
  yLabel : String := "qdot"
  lineColor : String := "k"
  lineWidth : Float := 2.0
  suppressBrowserEnv : String := "TEST_TMPDIR"
  deriving Repr, Inhabited

namespace LimitCyclePlotSpec

def validate? (spec : LimitCyclePlotSpec) : Except String Unit := do
  if spec.scriptPath != "../drake/examples/van_der_pol/plot_limit_cycle.py" then
    .error s!"Van der Pol plot script path mismatch: {spec.scriptPath}"
  if spec.backendEnv != "Agg" then
    .error s!"Van der Pol plot script should use Agg for test-safe rendering, got {spec.backendEnv}"
  if spec.outputPath != "plot_limit_cycle.png" then
    .error s!"Van der Pol plot output path mismatch: {spec.outputPath}"
  if !(Float.isFinite spec.xMin) || !(Float.isFinite spec.xMax) || spec.xMin >= spec.xMax then
    .error s!"Van der Pol plot x limits must be finite and ordered, got [{spec.xMin}, {spec.xMax}]"
  if !(Float.isFinite spec.yMin) || !(Float.isFinite spec.yMax) || spec.yMin >= spec.yMax then
    .error s!"Van der Pol plot y limits must be finite and ordered, got [{spec.yMin}, {spec.yMax}]"
  if spec.xLabel != "q" || spec.yLabel != "qdot" then
    .error s!"Van der Pol plot labels should be q/qdot, got {spec.xLabel}/{spec.yLabel}"
  if spec.lineColor != "k" then
    .error s!"Van der Pol plot should use black line color k, got {spec.lineColor}"
  if !(Float.isFinite spec.lineWidth) || spec.lineWidth <= 0.0 then
    .error s!"Van der Pol plot line width must be positive and finite, got {spec.lineWidth}"
  if spec.suppressBrowserEnv != "TEST_TMPDIR" then
    .error s!"Van der Pol plot should suppress browser opening under TEST_TMPDIR, got {spec.suppressBrowserEnv}"

end LimitCyclePlotSpec

def limitCyclePlotSpec : LimitCyclePlotSpec := {}

structure LimitCycleLeanPlotSpec where
  packagePath : String := "../lean-plot"
  renderer : String := "LeanPlot.Export.writeSvg"
  outputPath : String := "plot_limit_cycle.svg"
  xMin : Float := -2.5
  xMax : Float := 2.5
  yMin : Float := -3.0
  yMax : Float := 3.0
  xLabel : String := "q"
  yLabel : String := "qdot"
  lineColor : String := "Color.black"
  lineWidth : Float := 2.0
  usesSkia : Bool := false
  deriving Repr, Inhabited

namespace LimitCycleLeanPlotSpec

def validate? (spec : LimitCycleLeanPlotSpec) : Except String Unit := do
  if spec.packagePath != "../lean-plot" then
    .error s!"LeanPlot sibling package path mismatch: {spec.packagePath}"
  if spec.renderer != "LeanPlot.Export.writeSvg" then
    .error s!"LeanPlot Van der Pol renderer should be LeanPlot.Export.writeSvg, got {spec.renderer}"
  if spec.outputPath != "plot_limit_cycle.svg" then
    .error s!"LeanPlot Van der Pol output path mismatch: {spec.outputPath}"
  if !(Float.isFinite spec.xMin) || !(Float.isFinite spec.xMax) || spec.xMin >= spec.xMax then
    .error s!"LeanPlot Van der Pol x limits must be finite and ordered, got [{spec.xMin}, {spec.xMax}]"
  if !(Float.isFinite spec.yMin) || !(Float.isFinite spec.yMax) || spec.yMin >= spec.yMax then
    .error s!"LeanPlot Van der Pol y limits must be finite and ordered, got [{spec.yMin}, {spec.yMax}]"
  if spec.xLabel != "q" || spec.yLabel != "qdot" then
    .error s!"LeanPlot Van der Pol labels should be q/qdot, got {spec.xLabel}/{spec.yLabel}"
  if spec.lineColor != "Color.black" then
    .error s!"LeanPlot Van der Pol line should preserve Drake's black trace, got {spec.lineColor}"
  if !(Float.isFinite spec.lineWidth) || spec.lineWidth <= 0.0 then
    .error s!"LeanPlot Van der Pol line width must be positive and finite, got {spec.lineWidth}"
  if spec.usesSkia then
    .error "LeanPlot Van der Pol boundary should use dependency-light SVG export, not Skia"

end LimitCycleLeanPlotSpec

def limitCycleLeanPlotSpec : LimitCycleLeanPlotSpec := {}

structure LimitCycleClosure where
  start : VanDerPolState
  finish : VanDerPolState
  qError : Float
  qdotError : Float
  deriving Repr, Inhabited

namespace LimitCycleClosure

def passes (closure : LimitCycleClosure) (spec : LimitCycleTestSpec) : Bool :=
  closure.qError <= spec.qTolerance && closure.qdotError <= spec.qdotTolerance

end LimitCycleClosure

def closureFromRollout (rollout : VanDerPolResult) : LimitCycleClosure :=
  let start := rollout.samples.getD 0 rollout.initialState
  let finish := rollout.samples.getD (rollout.samples.size - 1) rollout.finalState
  {
    start := start
    finish := finish
    qError := Float.abs (finish.q - start.q)
    qdotError := Float.abs (finish.qdot - start.qdot)
  }

def plotBoundaryVertex : VertexId := 5100

def plotOutputVertex : VertexId := 5101

def systemDeclarationVertex : VertexId := 5102

def positionOutputVertex : VertexId := 5103

def fullStateOutputVertex : VertexId := 5104

def muConstraintVertex : VertexId := 5105

def leanPlotBoundaryVertex : VertexId := 5106

def leanPlotOutputVertex : VertexId := 5107

def systemDeclarationMove (spec : VanDerPolSystemSpec := systemSpec) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[systemDeclarationVertex]
    reads := #[]
    writes := #[positionOutputVertex, fullStateOutputVertex, muConstraintVertex]
    exactness := .exact
    cost := { work := 1.0, memory := 1.0 }
    label := s!"DeclareContinuousState({spec.continuousPositionSize}, {spec.continuousVelocitySize}, {spec.continuousMiscSize}); y0/y1 outputs; {spec.inequalityConstraint.name}"
  }

def plotBoundaryMove (spec : LimitCyclePlotSpec := limitCyclePlotSpec) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[plotBoundaryVertex]
    reads := #[5000]
    writes := #[plotOutputVertex]
    exactness := .exact
    cost := { work := 1.0, memory := 1.0 }
    label := s!"render VanDerPol CalcLimitCycle via {spec.scriptPath}"
  }

def leanPlotBoundaryMove (spec : LimitCycleLeanPlotSpec := limitCycleLeanPlotSpec) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[leanPlotBoundaryVertex]
    reads := #[5000]
    writes := #[leanPlotOutputVertex]
    exactness := .exact
    cost := { work := 1.0, memory := 1.0 }
    label := s!"render VanDerPol CalcLimitCycle via {spec.renderer} from {spec.packagePath}"
  }

def graph (rollout : VanDerPolResult)
    (systemSpec : VanDerPolSystemSpec := systemSpec)
    (plotSpec : LimitCyclePlotSpec := limitCyclePlotSpec)
    (leanPlotSpec : LimitCycleLeanPlotSpec := limitCycleLeanPlotSpec) : SkeletonGraph := Id.run do
  let mut g :=
    SkeletonGraph.empty
      |>.addVertex { id := systemDeclarationVertex, kind := .state .boundary, label := systemSpec.systemName }
      |>.addVertex { id := positionOutputVertex, kind := .state .checkpoint, label := "y0 = q output port" }
      |>.addVertex { id := fullStateOutputVertex, kind := .state .checkpoint, label := "y1 = [q, qdot] output port" }
      |>.addVertex { id := muConstraintVertex, kind := .state .boundary, label := systemSpec.inequalityConstraint.name }
      |>.addVertex { id := 5000, kind := .interval, label := "VanDerPolOscillator CalcLimitCycle interval" }
      |>.addVertex { id := plotBoundaryVertex, kind := .state .boundary, label := plotSpec.scriptPath }
      |>.addVertex { id := plotOutputVertex, kind := .state .checkpoint, label := plotSpec.outputPath }
      |>.addVertex { id := leanPlotBoundaryVertex, kind := .state .boundary, label := leanPlotSpec.renderer }
      |>.addVertex { id := leanPlotOutputVertex, kind := .state .checkpoint, label := leanPlotSpec.outputPath }
  for move in rollout.moves do
    g := g.addMove move
  g := g.addMove (systemDeclarationMove systemSpec)
  g := g.addMove (plotBoundaryMove plotSpec)
  g := g.addMove (leanPlotBoundaryMove leanPlotSpec)
  return g

structure ExampleResult where
  references : Array DrakeReference
  buildTargets : Array VanDerPolBuildTarget
  systemSpec : VanDerPolSystemSpec
  scalarConversion : ScalarConversionSpec
  limitCycleTest : LimitCycleTestSpec
  plotSpec : LimitCyclePlotSpec
  leanPlotSpec : LimitCycleLeanPlotSpec
  rollout : VanDerPolResult
  closure : LimitCycleClosure
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def build? (p : VanDerPolParams := params)
    (systemSpec : VanDerPolSystemSpec := systemSpec)
    (scalarSpec : ScalarConversionSpec := scalarConversionSpec)
    (testSpec : LimitCycleTestSpec := limitCycleTestSpec)
    (plotSpec : LimitCyclePlotSpec := limitCyclePlotSpec)
    (leanPlotSpec : LimitCycleLeanPlotSpec := limitCycleLeanPlotSpec) :
    Except String ExampleResult := do
  validateBuildTargets?
  systemSpec.validate?
  scalarSpec.validate?
  testSpec.validate?
  plotSpec.validate?
  leanPlotSpec.validate?
  let rollout ← simulate? p
  let closure := closureFromRollout rollout
  let graph := graph rollout systemSpec plotSpec leanPlotSpec
  pure {
    references := drakeReferences
    buildTargets := buildTargets
    systemSpec := systemSpec
    scalarConversion := scalarSpec
    limitCycleTest := testSpec
    plotSpec := plotSpec
    leanPlotSpec := leanPlotSpec
    rollout := rollout
    closure := closure
    graph := graph
    moves := graph.moves
  }

def buildEndToEnd? (p : VanDerPolParams := params)
    (systemSpec : VanDerPolSystemSpec := systemSpec)
    (scalarSpec : ScalarConversionSpec := scalarConversionSpec)
    (testSpec : LimitCycleTestSpec := limitCycleTestSpec)
    (plotSpec : LimitCyclePlotSpec := limitCyclePlotSpec)
    (leanPlotSpec : LimitCycleLeanPlotSpec := limitCycleLeanPlotSpec) :
    Except String ExampleResult :=
  build? p systemSpec scalarSpec testSpec plotSpec leanPlotSpec

end Tyr.EventSkeleton.Examples.VanDerPol
