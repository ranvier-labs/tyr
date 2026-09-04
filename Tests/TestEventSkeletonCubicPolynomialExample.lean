import LeanTest
import Tyr.EventSkeleton.Examples.CubicPolynomial

namespace Tests.EventSkeletonCubicPolynomialExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.CubicPolynomial

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def assertPolyApprox (p : UniPoly) (expected : Array Float)
    (tol : Float) (label : String) : IO Unit := do
  LeanTest.assertEqual p.coeffs.size expected.size s!"{label}: coefficient size mismatch"
  for i in [:expected.size] do
    LeanTest.assertTrue (approx (p.coeffs.getD i 0.0) expected[i]! tol)
      s!"{label}: coeff[{i}] expected {expected[i]!}, got {p.coeffs.getD i 0.0}"

private def hasCall (calls : Array PythonCallSpec) (functionName : String)
    (args : Array String) : Bool :=
  calls.any (fun call => call.functionName == functionName && call.args == args)

private def hasPlotSeries (series : Array BackwardReachabilityPlotSeries)
    (yVar : String) : Bool :=
  series.any (fun item => item.xVar == "x_val" && item.yVar == yVar)

@[test]
def testDrakeReferencesAndCubicDynamicsAreRecorded : IO Unit := do
  let result ← assertOk buildEndToEnd? "cubic polynomial end-to-end build"
  LeanTest.assertTrue
    (result.references.any
      (fun ref => ref.path == "../drake/examples/cubic_polynomial/BUILD.bazel"))
    "Cubic polynomial example should reference Drake's BUILD.bazel"
  LeanTest.assertTrue
    (result.references.any
      (fun ref => ref.path == "../drake/examples/cubic_polynomial/region_of_attraction.cc"))
    "Cubic polynomial example should reference Drake's ROA executable"
  LeanTest.assertTrue
    (result.references.any
      (fun ref => ref.path == "../drake/examples/cubic_polynomial/backward_reachability.cc"))
    "Cubic polynomial example should reference Drake's backward reachability executable"
  assertPolyApprox roaDynamics #[0.0, -1.0, 0.0, 1.0] 1.0e-15
    "ROA dynamics xdot = -x + x^3"
  LeanTest.assertTrue (approx (roaDerivative 2.0) 6.0 1.0e-12)
    s!"ROA derivative should be -x + x^3, got {roaDerivative 2.0}"
  assertPolyApprox backwardReachabilityDynamics #[0.0, -25.0, 0.0, 100.0] 1.0e-15
    "Backward reachability dynamics xdot = 100x^3 - 25x"
  LeanTest.assertTrue (approx (backwardReachabilityDerivative 0.5) 0.0 1.0e-12)
    s!"x=0.5 should be an equilibrium for 100x^3 - 25x, got {backwardReachabilityDerivative 0.5}"

@[test]
def testBuildTargetsMatchDrakeBazelExecutables : IO Unit := do
  let result ← assertOk buildEndToEnd? "cubic polynomial end-to-end build"
  let _ ← assertOk (validateDrakeBinarySpecs? result.binaries)
    "cubic polynomial Drake binary metadata"
  LeanTest.assertEqual result.binaries.size 2
    "Drake BUILD.bazel declares region_of_attraction and backward_reachability"

  let region? := result.binaries.find? (fun spec => spec.name == "region_of_attraction")
  let backward? := result.binaries.find? (fun spec => spec.name == "backward_reachability")
  match region?, backward? with
  | some region, some backward =>
      LeanTest.assertEqual region.source "region_of_attraction.cc"
        "region_of_attraction should compile the matching C++ source"
      LeanTest.assertTrue region.addTestRule
        "region_of_attraction should keep add_test_rule=True"
      LeanTest.assertTrue (region.hasDep "//common:add_text_logging_gflags")
        "region_of_attraction should include the add_text_logging_gflags dependency"
      LeanTest.assertFalse (region.hasDep "//common/proto:call_python")
        "region_of_attraction does not use the CallPython plotting boundary"

      LeanTest.assertEqual backward.source "backward_reachability.cc"
        "backward_reachability should compile the matching C++ source"
      LeanTest.assertTrue backward.addTestRule
        "backward_reachability should keep add_test_rule=True"
      LeanTest.assertTrue (backward.hasDep "//common/proto:call_python")
        "backward_reachability should include Drake's CallPython dependency"
  | _, _ => LeanTest.fail "Expected both cubic_polynomial Drake binaries"

@[test]
def testRegionOfAttractionSosCertificateMatchesDrakeCheck : IO Unit := do
  let result ← assertOk buildEndToEnd? "cubic polynomial end-to-end build"
  let cert := result.roaCertificate
  LeanTest.assertTrue cert.verified
    "Analytic certificate should satisfy Drake's rho ≃ 1 check"
  LeanTest.assertTrue (approx cert.rho 1.0 1.0e-12)
    s!"Drake checks rho near 1.0, got {cert.rho}"
  assertPolyApprox cert.V #[0.0, 0.0, 1.0] 1.0e-15
    "Lyapunov function V=x^2"
  assertPolyApprox cert.Vdot #[0.0, 0.0, -2.0, 0.0, 2.0] 1.0e-15
    "Vdot = 2x(-x+x^3)"
  assertPolyApprox cert.lambda #[0.5] 1.0e-15
    "lambda constant SOS multiplier"
  LeanTest.assertTrue cert.sosExpression.isNearZero
    s!"(V-rho)x^2 - lambda*Vdot should collapse to zero, got {reprStr cert.sosExpression.coeffs}"
  LeanTest.assertTrue (cert.solverBlockExactness == MoveExactness.exact)
    "The analytic ROA certificate is exact for the Drake one-dimensional example"

@[test]
def testBackwardReachabilityProgramShapeAndMoments : IO Unit := do
  let result ← assertOk buildEndToEnd? "cubic polynomial end-to-end build"
  let spec := result.backwardSpec
  LeanTest.assertEqual spec.polynomialOrder 8
    "Drake chooses d=8 for v and w"
  LeanTest.assertEqual spec.sosMultiplierOrder 6
    "Drake constructs d-2 SOS multipliers"
  LeanTest.assertEqual spec.freePolynomialCount 2
    "Drake declares free polynomials v(t,x) and w(x)"
  LeanTest.assertEqual spec.sosMultiplierCount 5
    "Drake declares qx, qt, qT, q0, and sx"
  LeanTest.assertEqual spec.sosConstraintCount 4
    "Drake adds four nonnegativity/SOS constraints around the multipliers"
  LeanTest.assertTrue (spec.solverBlockExactness == MoveExactness.controlledApproximation)
    "The local implementation records the SDP solve boundary rather than claiming to run Drake's solver"
  assertPolyApprox spec.domainPolynomial #[1.0, 0.0, -1.0] 1.0e-15
    "Domain polynomial gx = 1 - x^2"
  assertPolyApprox spec.terminalPolynomial #[0.01, 0.0, -1.0] 1.0e-15
    "Terminal polynomial gxT = 0.01 - x^2"
  assertPolyApprox spec.groundTruthPolynomial #[0.25, 0.0, -1.0] 1.0e-15
    "Known reachable set polynomial gx0 = 0.25 - x^2"
  LeanTest.assertTrue (approx spec.volumeOfConstantOne 2.0 1.0e-12)
    s!"Integral of 1 over [-1,1] should be 2, got {spec.volumeOfConstantOne}"
  LeanTest.assertTrue (approx spec.volumeOfGroundTruthInterval 1.0 1.0e-12)
    s!"Ground-truth interval |x| <= 0.5 should have volume 1, got {spec.volumeOfGroundTruthInterval}"

@[test]
def testReachabilitySamplesAndOptimizationGraphExposeSdpBoundary : IO Unit := do
  let result ← assertOk buildEndToEnd? "cubic polynomial end-to-end build"
  LeanTest.assertEqual result.samples.size 21
    "Default plotting/check samples should cover the domain with 21 points"
  let midpoint := result.samples[10]!
  LeanTest.assertTrue (approx midpoint.x 0.0 1.0e-12)
    s!"Middle sample should be x=0, got {midpoint.x}"
  LeanTest.assertTrue (approx midpoint.wGroundTruth 1.0 1.0e-12)
    "x=0 should be inside the known backward reachable set"
  LeanTest.assertTrue (approx result.samples[0]!.wGroundTruth 0.0 1.0e-12)
    "x=-1 should be outside the known backward reachable set"
  LeanTest.assertTrue (approx result.samples[result.samples.size - 1]!.wGroundTruth 0.0 1.0e-12)
    "x=1 should be outside the known backward reachable set"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "Symbolic extraction and SDP solve should be local Schur blocks"
  LeanTest.assertTrue
    (result.graph.moves.any
      (fun move =>
        move.kind == .localSchurBlock &&
        move.exactness == MoveExactness.controlledApproximation &&
        move.label.contains "SOS/SDP"))
    "The SOS/SDP solve should remain visible as an approximate local solver boundary"
  LeanTest.assertTrue (result.graph.containsMoveKind .checkpointBoundary)
    "Plotting samples should be checkpointed as the output boundary"

@[test]
def testBackwardReachabilityPlotBoundaryMatchesDrakeCallPython : IO Unit := do
  let result ← assertOk buildEndToEnd? "cubic polynomial end-to-end build"
  let plot := result.plotSpec
  let _ ← assertOk plot.validate? "backward reachability plot spec validation"
  LeanTest.assertEqual plot.sourceStruct "CallPython"
    "Drake backward_reachability serializes through CallPython"
  LeanTest.assertTrue (plot.backend.kind == CubicPlotBackendKind.leanPlotSvg)
    "The Lean port should stage this plotting boundary for lean-plot SVG output"
  LeanTest.assertEqual plot.backend.libraryPath "../lean-plot"
    "The lean-plot backend should point at the sibling checkout"
  LeanTest.assertEqual plot.backend.exportFunction "LeanPlot.Export.writeSvg"
    "The plotting backend should identify the lightweight LeanPlot SVG exporter"
  LeanTest.assertEqual plot.sampleCount 1000
    "Drake backward_reachability uses N=1000 for x_val, w_val, and ground_val"
  LeanTest.assertEqual plot.calls.size 8
    "Drake emits 8 CallPython operations after solving"
  LeanTest.assertTrue (hasCall plot.calls "figure" #["1"])
    "Plotting should open figure 1"
  LeanTest.assertTrue (hasCall plot.calls "plot" #["x_val", "w_val"])
    "Plotting should include the SDP-produced w_val channel"
  LeanTest.assertTrue (hasCall plot.calls "setvars" #["x_val", "x_val", "w_val", "w_val"])
    "Plotting should expose x_val and w_val to the Python session"
  LeanTest.assertTrue (hasCall plot.calls "plot" #["x_val", "ground_val"])
    "Plotting should include the true indicator channel"
  LeanTest.assertTrue (hasCall plot.calls "plt.xlabel" #["x"])
    "Plotting should set the Drake x-axis label"
  LeanTest.assertTrue (hasCall plot.calls "plt.ylabel" #["w, I_B"])
    "Plotting should set the Drake y-axis label"
  LeanTest.assertTrue (hasPlotSeries plot.series "w_val")
    "The plot spec should expose w_val as the polynomial outer approximation"
  LeanTest.assertTrue (hasPlotSeries plot.series "ground_val")
    "The plot spec should expose ground_val as the true indicator"

  LeanTest.assertEqual result.plotSamples.size 1000
    "The port should materialize Drake's 1000-point plotting grid"
  LeanTest.assertTrue (approx result.plotSamples[0]!.x (-1.0) 1.0e-12)
    s!"First Drake plot sample should be -1, got {result.plotSamples[0]!.x}"
  LeanTest.assertTrue (approx result.plotSamples[500]!.x 0.0 1.0e-12)
    s!"Middle Drake plot sample should be 0, got {result.plotSamples[500]!.x}"
  LeanTest.assertTrue (approx result.plotSamples[999]!.x 0.998 1.0e-12)
    s!"Drake's right-open grid should end at 0.998 for N=1000, got {result.plotSamples[999]!.x}"
  LeanTest.assertTrue (approx result.plotSamples[500]!.groundVal 1.0 1.0e-12)
    "x=0 should be inside the known backward reachable set"
  LeanTest.assertTrue (approx result.plotSamples[751]!.groundVal 0.0 1.0e-12)
    "x=0.502 should be outside the known backward reachable set"
  LeanTest.assertFalse (result.plotSamples.any (fun sample => sample.wValProducedBySolver))
    "The current Lean surrogate should not fabricate the SDP-produced w_val channel"

  LeanTest.assertTrue
    (result.graph.vertices.any (fun vertex =>
      vertex.id == plotOutputVertex && vertex.label.contains "lean-plot SVG"))
    "Skeleton graph should expose the lean-plot SVG output checkpoint"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.targets == #[plotBoundaryVertex] &&
      move.reads == #[8804] &&
      move.writes == #[plotOutputVertex] &&
      move.label.contains "LeanPlot.Export.writeSvg"))
    "Skeleton graph should include the exact CallPython-to-lean-plot render boundary"

end Tests.EventSkeletonCubicPolynomialExample
