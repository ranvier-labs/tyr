import Tyr.EventSkeleton.Trace

/-!
# Drake Cubic Polynomial Event-Skeleton Examples

This ports the executable algebraic boundary of
`../drake/examples/cubic_polynomial`.

Drake's examples build symbolic polynomial systems and then solve SOS/SDP
programs:

* `region_of_attraction.cc` proves the ROA certificate for `xdot = -x + x^3`.
* `backward_reachability.cc` builds the Henrion-Korda occupation-measure
  relaxation for `xdot = 100x^3 - 25x`.

This file keeps the polynomial and optimization vertices explicit.  The local
SDP solver remains a `localSchurBlock`; the algebra around it is executable and
checked here.
-/

namespace Tyr.EventSkeleton.Examples.CubicPolynomial

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/cubic_polynomial/BUILD.bazel"
      concept := "declares region_of_attraction and backward_reachability as drake_cc_binary examples with add_test_rule"
    },
    {
      path := "../drake/examples/cubic_polynomial/region_of_attraction.cc"
      concept := "constructs the symbolic cubic system xdot = -x + x^3 and solves a one-dimensional SOS ROA program"
    },
    {
      path := "../drake/examples/cubic_polynomial/backward_reachability.cc"
      concept := "constructs the symbolic cubic system xdot = 100x^3 - 25x and solves an occupation-measure SOS relaxation"
    }
  ]

structure DrakeBinarySpec where
  name : String
  source : String
  addTestRule : Bool := true
  deps : Array String := #[]
  deriving Repr, BEq, Inhabited

namespace DrakeBinarySpec

def hasDep (spec : DrakeBinarySpec) (dep : String) : Bool :=
  spec.deps.any (fun actual => actual == dep)

def validate? (spec : DrakeBinarySpec) : Except String Unit := do
  if spec.name.isEmpty then
    .error "cubic-polynomial Drake binary name cannot be empty"
  if spec.source.isEmpty then
    .error s!"cubic-polynomial Drake binary {spec.name} requires a source file"
  if !spec.addTestRule then
    .error s!"cubic-polynomial Drake binary {spec.name} should keep BUILD.bazel add_test_rule=true"
  if !spec.hasDep "//solvers:mathematical_program" then
    .error s!"cubic-polynomial Drake binary {spec.name} should depend on MathematicalProgram"
  if !spec.hasDep "//solvers:solve" then
    .error s!"cubic-polynomial Drake binary {spec.name} should depend on Solve"
  if !spec.hasDep "//systems/framework:vector_system" then
    .error s!"cubic-polynomial Drake binary {spec.name} should depend on VectorSystem"

end DrakeBinarySpec

def drakeBinarySpecs : Array DrakeBinarySpec :=
  #[
    {
      name := "region_of_attraction"
      source := "region_of_attraction.cc"
      deps := #[
        "//common:add_text_logging_gflags",
        "//solvers:mathematical_program",
        "//solvers:solve",
        "//systems/framework:vector_system"
      ]
    },
    {
      name := "backward_reachability"
      source := "backward_reachability.cc"
      deps := #[
        "//common/proto:call_python",
        "//solvers:mathematical_program",
        "//solvers:solve",
        "//systems/framework:vector_system"
      ]
    }
  ]

def validateDrakeBinarySpecs? (specs : Array DrakeBinarySpec := drakeBinarySpecs) :
    Except String Unit := do
  if specs.size != 2 then
    .error s!"cubic_polynomial BUILD.bazel should expose 2 binaries, got {specs.size}"
  if !(specs.any (fun spec => spec.name == "region_of_attraction")) then
    .error "missing region_of_attraction Drake binary"
  if !(specs.any (fun spec => spec.name == "backward_reachability")) then
    .error "missing backward_reachability Drake binary"
  for spec in specs do
    spec.validate?

private def powNat (x : Float) (n : Nat) : Float := Id.run do
  let mut out := 1.0
  for _ in [:n] do
    out := out * x
  return out

structure UniPoly where
  coeffs : Array Float := #[]
  deriving Repr, Inhabited

namespace UniPoly

def constant (c : Float) : UniPoly :=
  { coeffs := #[c] }

def monomial (degree : Nat) (c : Float := 1.0) : UniPoly := Id.run do
  let mut coeffs := Array.replicate (degree + 1) 0.0
  coeffs := coeffs.set! degree c
  return { coeffs := coeffs }

def x : UniPoly :=
  monomial 1

def x2 : UniPoly :=
  monomial 2

def eval (p : UniPoly) (x : Float) : Float := Id.run do
  let mut result := 0.0
  let mut xp := 1.0
  for c in p.coeffs do
    result := result + c * xp
    xp := xp * x
  return result

def add (p q : UniPoly) : UniPoly := Id.run do
  let n := Nat.max p.coeffs.size q.coeffs.size
  let mut coeffs : Array Float := #[]
  for i in [:n] do
    coeffs := coeffs.push (p.coeffs.getD i 0.0 + q.coeffs.getD i 0.0)
  return { coeffs := coeffs }

def neg (p : UniPoly) : UniPoly :=
  { coeffs := p.coeffs.map (fun c => -c) }

def sub (p q : UniPoly) : UniPoly :=
  p.add q.neg

def scale (a : Float) (p : UniPoly) : UniPoly :=
  { coeffs := p.coeffs.map (fun c => a * c) }

def mul (p q : UniPoly) : UniPoly := Id.run do
  if p.coeffs.isEmpty || q.coeffs.isEmpty then
    return {}
  let mut coeffs := Array.replicate (p.coeffs.size + q.coeffs.size - 1) 0.0
  for i in [:p.coeffs.size] do
    for j in [:q.coeffs.size] do
      let k := i + j
      coeffs := coeffs.set! k (coeffs[k]! + p.coeffs[i]! * q.coeffs[j]!)
  return { coeffs := coeffs }

def derivative (p : UniPoly) : UniPoly := Id.run do
  if p.coeffs.size <= 1 then
    return { coeffs := #[0.0] }
  let mut coeffs : Array Float := #[]
  for i in [1:p.coeffs.size] do
    coeffs := coeffs.push (i.toFloat * p.coeffs[i]!)
  return { coeffs := coeffs }

def integralOver (p : UniPoly) (lower upper : Float) : Float := Id.run do
  let mut result := 0.0
  for i in [:p.coeffs.size] do
    let exponent := i + 1
    let moment := (powNat upper exponent - powNat lower exponent) / exponent.toFloat
    result := result + p.coeffs[i]! * moment
  return result

def isNearZero (p : UniPoly) (tol : Float := 1.0e-12) : Bool :=
  p.coeffs.all (fun c => Float.abs c < tol)

def lieDerivative (value dynamics : UniPoly) : UniPoly :=
  value.derivative.mul dynamics

end UniPoly

/-! ## Shared cubic dynamics -/

def roaDynamics : UniPoly :=
  { coeffs := #[0.0, -1.0, 0.0, 1.0] }

def backwardReachabilityDynamics : UniPoly :=
  { coeffs := #[0.0, -25.0, 0.0, 100.0] }

def roaDerivative (x : Float) : Float :=
  roaDynamics.eval x

def backwardReachabilityDerivative (x : Float) : Float :=
  backwardReachabilityDynamics.eval x

/-! ## Region of attraction example -/

structure RegionOfAttractionCertificate where
  V : UniPoly
  dynamics : UniPoly
  Vdot : UniPoly
  rho : Float
  lambda : UniPoly
  sosExpression : UniPoly
  solverBlockExactness : MoveExactness := .exact
  deriving Repr, Inhabited

namespace RegionOfAttractionCertificate

def verified (cert : RegionOfAttractionCertificate) : Bool :=
  Float.abs (cert.rho - 1.0) < 1.0e-12 &&
    cert.lambda.coeffs == #[0.5] &&
    cert.sosExpression.isNearZero

end RegionOfAttractionCertificate

def regionOfAttractionCertificate : RegionOfAttractionCertificate :=
  let V := UniPoly.x2
  let Vdot := V.lieDerivative roaDynamics
  let rho := 1.0
  let lambda := UniPoly.constant 0.5
  let sosExpression :=
    ((V.sub (UniPoly.constant rho)).mul UniPoly.x2).sub (lambda.mul Vdot)
  {
    V := V
    dynamics := roaDynamics
    Vdot := Vdot
    rho := rho
    lambda := lambda
    sosExpression := sosExpression
    solverBlockExactness := .exact
  }

/-! ## Backward reachable set example -/

structure BackwardReachabilitySpec where
  dynamics : UniPoly
  T : Float := 1.0
  xBound : Float := 1.0
  terminalRadius : Float := 0.1
  groundTruthRadius : Float := 0.5
  polynomialOrder : Nat := 8
  sosMultiplierOrder : Nat := 6
  freePolynomialCount : Nat := 2
  sosMultiplierCount : Nat := 5
  sosConstraintCount : Nat := 4
  sampleCount : Nat := 1000
  solverBlockExactness : MoveExactness := .controlledApproximation
  deriving Repr, Inhabited

namespace BackwardReachabilitySpec

def domainPolynomial (spec : BackwardReachabilitySpec) : UniPoly :=
  { coeffs := #[spec.xBound * spec.xBound, 0.0, -1.0] }

def terminalPolynomial (spec : BackwardReachabilitySpec) : UniPoly :=
  { coeffs := #[spec.terminalRadius * spec.terminalRadius, 0.0, -1.0] }

def groundTruthPolynomial (spec : BackwardReachabilitySpec) : UniPoly :=
  { coeffs := #[spec.groundTruthRadius * spec.groundTruthRadius, 0.0, -1.0] }

def inGroundTruthReachableSet (spec : BackwardReachabilitySpec) (x : Float) : Bool :=
  spec.groundTruthPolynomial.eval x >= 0.0

def indicator (spec : BackwardReachabilitySpec) (x : Float) : Float :=
  if spec.inGroundTruthReachableSet x then 1.0 else 0.0

def volumeOfConstantOne (spec : BackwardReachabilitySpec) : Float :=
  (UniPoly.constant 1.0).integralOver (-spec.xBound) spec.xBound

def volumeOfGroundTruthInterval (spec : BackwardReachabilitySpec) : Float :=
  2.0 * spec.groundTruthRadius

end BackwardReachabilitySpec

def backwardReachabilitySpec : BackwardReachabilitySpec :=
  { dynamics := backwardReachabilityDynamics }

inductive CubicPlotBackendKind where
  | drakeCallPython
  | leanPlotSvg
  deriving Repr, BEq, Inhabited

structure CubicPlotBackend where
  kind : CubicPlotBackendKind := .leanPlotSvg
  libraryPath : String := "../lean-plot"
  moduleName : String := "LeanPlot"
  exportFunction : String := "LeanPlot.Export.writeSvg"
  outputStem : String := "cubic-backward-reachability"
  deriving Repr, BEq, Inhabited

namespace CubicPlotBackend

def validate? (backend : CubicPlotBackend) : Except String Unit := do
  if backend.kind == .leanPlotSvg then
    if backend.libraryPath != "../lean-plot" then
      .error s!"cubic-polynomial lean-plot backend should point at ../lean-plot, got {backend.libraryPath}"
    if backend.moduleName != "LeanPlot" then
      .error s!"cubic-polynomial lean-plot backend should import LeanPlot, got {backend.moduleName}"
    if backend.exportFunction != "LeanPlot.Export.writeSvg" then
      .error s!"cubic-polynomial lean-plot backend should render SVG through LeanPlot.Export.writeSvg, got {backend.exportFunction}"
    if backend.outputStem.isEmpty then
      .error "cubic-polynomial lean-plot backend requires an output stem"

end CubicPlotBackend

structure PythonCallSpec where
  functionName : String
  args : Array String := #[]
  deriving Repr, BEq, Inhabited

namespace PythonCallSpec

def validate? (call : PythonCallSpec) : Except String Unit := do
  if call.functionName.isEmpty then
    .error "CallPython function name cannot be empty"

end PythonCallSpec

structure BackwardReachabilityPlotSeries where
  xVar : String := "x_val"
  yVar : String
  source : String
  legend : String
  leanPlotMark : String := "Mark.lineSeries"
  deriving Repr, BEq, Inhabited

namespace BackwardReachabilityPlotSeries

def validate? (series : BackwardReachabilityPlotSeries) : Except String Unit := do
  if series.xVar != "x_val" then
    .error s!"backward reachability plot should use x_val as x-channel, got {series.xVar}"
  if series.yVar.isEmpty then
    .error "backward reachability plot y-channel cannot be empty"
  if series.source.isEmpty then
    .error "backward reachability plot source cannot be empty"
  if series.legend.isEmpty then
    .error "backward reachability plot legend cannot be empty"
  if series.leanPlotMark != "Mark.lineSeries" then
    .error s!"backward reachability plot should lower to LeanPlot Mark.lineSeries, got {series.leanPlotMark}"

end BackwardReachabilityPlotSeries

def backwardReachabilityPythonCalls : Array PythonCallSpec :=
  #[
    { functionName := "figure", args := #["1"] },
    { functionName := "clf" },
    { functionName := "plot", args := #["x_val", "w_val"] },
    { functionName := "setvars", args := #["x_val", "x_val", "w_val", "w_val"] },
    { functionName := "plot", args := #["x_val", "ground_val"] },
    { functionName := "setvars", args := #["x_val", "x_val", "ground_val", "ground_val"] },
    { functionName := "plt.xlabel", args := #["x"] },
    { functionName := "plt.ylabel", args := #["w, I_B"] }
  ]

def backwardReachabilityPlotSeries : Array BackwardReachabilityPlotSeries :=
  #[
    {
      yVar := "w_val"
      source := "w_sol.Evaluate({x -> x_val[i]}) from MathematicalProgramResult"
      legend := "polynomial outer approximation"
    },
    {
      yVar := "ground_val"
      source := "indicator(gx0 >= 0)"
      legend := "true indicator"
    }
  ]

structure BackwardReachabilityPlotSpec where
  drakeFunction : String := "ComputeBackwardReachableSet"
  sourceStruct : String := "CallPython"
  backend : CubicPlotBackend := {}
  sampleCount : Nat := 1000
  xGridFormula : String := "x_bound * (2.0 * i / N - 1.0)"
  xLabel : String := "x"
  yLabel : String := "w, I_B"
  calls : Array PythonCallSpec := backwardReachabilityPythonCalls
  series : Array BackwardReachabilityPlotSeries := backwardReachabilityPlotSeries
  deriving Repr, BEq, Inhabited

namespace BackwardReachabilityPlotSpec

def validate? (spec : BackwardReachabilityPlotSpec) : Except String Unit := do
  if spec.drakeFunction != "ComputeBackwardReachableSet" then
    .error s!"backward reachability plot should mirror ComputeBackwardReachableSet, got {spec.drakeFunction}"
  if spec.sourceStruct != "CallPython" then
    .error s!"backward reachability output should record Drake CallPython calls, got {spec.sourceStruct}"
  spec.backend.validate?
  if spec.sampleCount != 1000 then
    .error s!"Drake backward_reachability uses N=1000 plot samples, got {spec.sampleCount}"
  if spec.xGridFormula != "x_bound * (2.0 * i / N - 1.0)" then
    .error s!"unexpected backward reachability x-grid formula: {spec.xGridFormula}"
  if spec.xLabel != "x" then
    .error s!"backward reachability x label should be x, got {spec.xLabel}"
  if spec.yLabel != "w, I_B" then
    .error s!"backward reachability y label should be w, I_B, got {spec.yLabel}"
  if spec.calls.size != 8 then
    .error s!"Drake backward_reachability has 8 CallPython calls after solving, got {spec.calls.size}"
  if spec.series.size != 2 then
    .error s!"Drake backward_reachability plots w_val and ground_val, got {spec.series.size} series"
  for call in spec.calls do
    call.validate?
  for series in spec.series do
    series.validate?

end BackwardReachabilityPlotSpec

def backwardReachabilityPlotSpec : BackwardReachabilityPlotSpec := {}

structure ReachabilitySample where
  x : Float
  wGroundTruth : Float
  dynamics : Float
  deriving Repr, Inhabited

structure ReachabilityPlotSample where
  index : Nat
  x : Float
  groundVal : Float
  dynamics : Float
  wValProducedBySolver : Bool := false
  deriving Repr, Inhabited

def reachabilitySamples (spec : BackwardReachabilitySpec := backwardReachabilitySpec)
    (count : Nat := 21) : Array ReachabilitySample := Id.run do
  let n := Nat.max 2 count
  let mut out : Array ReachabilitySample := #[]
  for i in [:n] do
    let s := i.toFloat / (n - 1).toFloat
    let x := -spec.xBound + 2.0 * spec.xBound * s
    out := out.push {
      x := x
      wGroundTruth := spec.indicator x
      dynamics := spec.dynamics.eval x
    }
  return out

def drakePlotSamples
    (spec : BackwardReachabilitySpec := backwardReachabilitySpec)
    (plotSpec : BackwardReachabilityPlotSpec := backwardReachabilityPlotSpec) :
    Array ReachabilityPlotSample := Id.run do
  let n := plotSpec.sampleCount
  let mut out : Array ReachabilityPlotSample := #[]
  for i in [:n] do
    let x := spec.xBound * (2.0 * i.toFloat / n.toFloat - 1.0)
    out := out.push {
      index := i
      x := x
      groundVal := spec.indicator x
      dynamics := spec.dynamics.eval x
      wValProducedBySolver := false
    }
  return out

def plotBoundaryVertex : VertexId := 8805

def plotOutputVertex : VertexId := 8806

def backwardReachabilityPlotMove
    (spec : BackwardReachabilityPlotSpec := backwardReachabilityPlotSpec) :
    SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[plotBoundaryVertex]
    reads := #[8804]
    writes := #[plotOutputVertex]
    exactness := .exact
    cost := { work := spec.sampleCount.toFloat, memory := 1.0 }
    label := s!"backward_reachability CallPython plot lowered to {spec.backend.exportFunction}"
  }

def optimizationGraph
    (plotSpec : BackwardReachabilityPlotSpec := backwardReachabilityPlotSpec) :
    SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8800, kind := .state .boundary, label := "cubic polynomial vector system" }
    |>.addVertex { id := 8801, kind := .state .interior, label := "symbolic polynomial dynamics" }
    |>.addVertex { id := 8802, kind := .state .interior, label := "SOS/SDP mathematical program" }
    |>.addVertex { id := 8803, kind := .opaque, label := "MathematicalProgramResult / SOS certificate" }
    |>.addVertex { id := 8804, kind := .state .checkpoint, label := "plot/evaluation samples" }
    |>.addVertex { id := plotBoundaryVertex, kind := .state .boundary, label := "backward_reachability CallPython render boundary" }
    |>.addVertex { id := plotOutputVertex, kind := .state .checkpoint, label := "lean-plot SVG output" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8800]
      reads := #[8800]
      writes := #[8801]
      exactness := .exact
      label := "extract symbolic cubic dynamics"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8801, 8802]
      reads := #[8801, 8802]
      writes := #[8803]
      exactness := .controlledApproximation
      label := "Solve SOS/SDP block represented by checked certificate metadata"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[8804]
      reads := #[8803]
      writes := #[8804]
      exactness := .exact
      label := "serialize cubic-polynomial plotting samples"
    }
    |>.addMove (backwardReachabilityPlotMove plotSpec)

structure CubicPolynomialResult where
  references : Array DrakeReference
  binaries : Array DrakeBinarySpec
  roaCertificate : RegionOfAttractionCertificate
  backwardSpec : BackwardReachabilitySpec
  samples : Array ReachabilitySample
  plotSpec : BackwardReachabilityPlotSpec
  plotSamples : Array ReachabilityPlotSample
  graph : SkeletonGraph
  deriving Repr, Inhabited

def buildEndToEnd? : Except String CubicPolynomialResult := do
  validateDrakeBinarySpecs?
  backwardReachabilityPlotSpec.validate?
  let roa := regionOfAttractionCertificate
  if !roa.verified then
    .error "cubic polynomial ROA certificate failed"
  let spec := backwardReachabilitySpec
  if spec.polynomialOrder < 2 then
    .error "backward reachability polynomial order must be at least 2"
  pure {
    references := drakeReferences
    binaries := drakeBinarySpecs
    roaCertificate := roa
    backwardSpec := spec
    samples := reachabilitySamples spec
    plotSpec := backwardReachabilityPlotSpec
    plotSamples := drakePlotSamples spec backwardReachabilityPlotSpec
    graph := optimizationGraph backwardReachabilityPlotSpec
  }

end Tyr.EventSkeleton.Examples.CubicPolynomial
