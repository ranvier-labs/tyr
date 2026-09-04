import LeanTest
import Tyr.EventSkeleton.Examples.Zmp

namespace Tests.EventSkeletonZmpExample

open LeanTest
open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.Zmp

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertOk {α : Type} (res : Except String α) (label : String) : IO α := do
  match res with
  | .ok value => pure value
  | .error msg => LeanTest.fail s!"{label}: expected ok, got {msg}"

private def hasPlotSeries
    (series : Array ZmpPlotSeries)
    (figure subplot row : Nat)
    (source style legend : String) : Bool :=
  series.any (fun item =>
    item.figure == figure &&
    item.subplot == subplot &&
    item.source == source &&
    item.row == row &&
    item.style == style &&
    item.legend == legend)

@[test]
def testDrakeReferencesAndCoordinateConventionsAreRecorded : IO Unit := do
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/examples/zmp/zmp_example.cc"))
    "ZMP example should reference Drake's plotting/demo wrapper"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/planning/locomotion/zmp_planner.cc"))
    "ZMP example should reference Drake's HJB/LQR planner implementation"
  LeanTest.assertTrue
    (drakeReferences.any (fun ref => ref.path == "../drake/planning/locomotion/test_utilities/zmp_test_util.cc"))
    "ZMP example should reference Drake's rollout and desired-ZMP utilities"
  LeanTest.assertEqual stateCoordinateNames #["com_x", "com_y", "comd_x", "comd_y"]
    "ZMP LIPM state order should match Drake's Vector4d convention"
  LeanTest.assertEqual inputCoordinateNames #["comdd_x", "comdd_y"]
    "ZMP control should be CoM acceleration"
  LeanTest.assertEqual outputCoordinateNames #["cop_x", "cop_y"]
    "ZMP output should be center of pressure"

@[test]
def testGenerateDesiredZmpTrajsMatchesDrakeFootstepTiming : IO Unit := do
  let trajs ← assertOk (generateDesiredZmpTrajs? defaultFootsteps 0.5 1.0)
    "desired ZMP trajectory generation"
  LeanTest.assertEqual trajs.size 3
    "Drake GenerateDesiredZmpTrajs returns ZOH, FOH, and cubic trajectories"
  LeanTest.assertTrue (trajs[0]!.kind == ZmpInterpolationKind.zeroOrderHold)
    "First desired-ZMP trajectory should be zero-order hold"
  LeanTest.assertTrue (trajs[1]!.kind == ZmpInterpolationKind.firstOrderHold)
    "Second desired-ZMP trajectory should be first-order hold"
  LeanTest.assertTrue (trajs[2]!.kind == ZmpInterpolationKind.cubicShapePreserving)
    "Third desired-ZMP trajectory should be cubic shape preserving"

  let expectedTimes := #[0.0, 1.0, 1.5, 2.5, 3.0, 4.0, 4.5, 5.5, 6.0, 7.0, 7.5, 8.5]
  LeanTest.assertEqual trajs[0]!.times expectedTimes
    "Desired-ZMP knot times should follow Drake's ss/ds footstep schedule"
  LeanTest.assertTrue (trajs[0]!.knots[0]! == defaultFootsteps[0]!)
    s!"First knot should be the first footstep, got {reprStr trajs[0]!.knots[0]!}"
  LeanTest.assertTrue (trajs[0]!.knots[1]! == defaultFootsteps[0]!)
    s!"Second knot should hold the first footstep through single support, got {reprStr trajs[0]!.knots[1]!}"
  LeanTest.assertTrue (trajs[0]!.knots[2]! == defaultFootsteps[1]!)
    s!"Third knot should move to the second footstep after double support, got {reprStr trajs[0]!.knots[2]!}"
  LeanTest.assertTrue (approx trajs[0]!.endTime 8.5 1.0e-12)
    s!"Drake default six-footstep ZMP trajectory should end at 8.5s, got {trajs[0]!.endTime}"

@[test]
def testLipmMatricesAndCopEquationMatchDrakePlannerDocs : IO Unit := do
  let m := lipmMatrices params
  LeanTest.assertEqual m.A #[
      #[0.0, 0.0, 1.0, 0.0],
      #[0.0, 0.0, 0.0, 1.0],
      #[0.0, 0.0, 0.0, 0.0],
      #[0.0, 0.0, 0.0, 0.0]]
    "LIPM A should be the double-integrator matrix"
  LeanTest.assertEqual m.B #[
      #[0.0, 0.0],
      #[0.0, 0.0],
      #[1.0, 0.0],
      #[0.0, 1.0]]
    "LIPM B should inject CoM acceleration into velocity derivatives"
  LeanTest.assertTrue (approx ((m.D[0]!).getD 0 99.0) (-params.height / params.gravity) 1.0e-12)
    s!"D should encode cop = com - z/g*comdd, got {reprStr m.D}"

  let trajs ← assertOk (generateDesiredZmpTrajs? defaultFootsteps 0.5 1.0)
    "desired ZMP trajectory generation"
  let planner ← assertOk (planTrackingPolicy? trajs[0]! params) "ZMP tracking planner"
  let x : ZmpState := { com := { x := 1.0, y := -0.5 }, comd := {} }
  let u : Vec2 := { x := 2.0, y := -3.0 }
  let cop := planner.comddToCop x u
  LeanTest.assertTrue (approx cop.x (1.0 - params.height / params.gravity * 2.0) 1.0e-12)
    s!"CoP x should equal com_x - z/g*comdd_x, got {reprStr cop}"
  LeanTest.assertTrue (approx cop.y (-0.5 + params.height / params.gravity * 3.0) 1.0e-12)
    s!"CoP y should equal com_y - z/g*comdd_y, got {reprStr cop}"

@[test]
def testEndToEndZmpRolloutExecutesLipmPolicy : IO Unit := do
  let result ← assertOk buildEndToEnd? "ZMP end-to-end build"
  LeanTest.assertEqual result.footsteps.size 6
    "Drake zmp_example uses six footsteps"
  LeanTest.assertEqual result.rollout.samples.size 1050
    "Drake SimulateZmpPolicy sample count is int((8.5 + 2.0) / 0.01)"
  LeanTest.assertTrue (result.graph.containsMoveKind .localSchurBlock)
    "ZmpPlanner.Plan should remain visible as a local planner block"
  LeanTest.assertTrue (result.graph.containsMoveKind .intervalAdjoint)
    "LIPM rollout should be represented as an interval"
  LeanTest.assertTrue (result.rollout.moves.any
      (fun move => move.kind == .localSchurBlock && move.exactness == .controlledApproximation))
    "The current planner block should be marked as controlled approximation until the HJB pass is ported"
  LeanTest.assertTrue (result.rollout.moves.any (fun move => move.kind == .intervalAdjoint))
    "Rollout moves should include the LIPM interval adjoint"

  match result.rollout.finalSample? with
  | none => LeanTest.fail "ZMP rollout should contain samples"
  | some final =>
      let finalFootstep := defaultFootsteps[defaultFootsteps.size - 1]!
      LeanTest.assertTrue ((final.state.com.sub finalFootstep).normInf < 5.0e-2)
        s!"Stabilizing LIPM policy should converge near the final footstep, got {reprStr final.state}"
      LeanTest.assertTrue ((final.state.comd).normInf < 5.0e-2)
        s!"Final LIPM velocity should decay near zero, got {reprStr final.state.comd}"
      LeanTest.assertTrue ((final.cop.sub finalFootstep).normInf < 5.0e-2)
        s!"Final CoP should be near the final desired ZMP, got {reprStr final.cop}"

@[test]
def testPlotResultsBoundaryCanLowerToLeanPlot : IO Unit := do
  let result ← assertOk buildEndToEnd? "ZMP end-to-end build"
  let spec := result.plotSpec
  let _ ← assertOk spec.validate? "ZMP plot spec validation"
  LeanTest.assertEqual spec.sourceStruct "planning::ZmpTestTraj"
    "PlotResults should consume Drake's ZmpTestTraj structure"
  LeanTest.assertEqual spec.drakeFunction "PlotResults"
    "Plot spec should mirror Drake's PlotResults helper"
  LeanTest.assertTrue (spec.backend.kind == ZmpPlotBackendKind.leanPlotSvg)
    "ZMP plotting should be staged for the lean-plot SVG backend"
  LeanTest.assertEqual spec.backend.libraryPath "../lean-plot"
    "The lean-plot backend should point at the sibling checkout"
  LeanTest.assertEqual spec.backend.exportFunction "LeanPlot.Export.writeSvg"
    "The backend should identify the lightweight LeanPlot SVG exporter"
  LeanTest.assertEqual spec.series.size 16
    "Drake PlotResults has 16 CallPython plot series"
  LeanTest.assertEqual (spec.series.filter (fun item => item.figure == 1)).size 8
    "Figure 1 should plot desired ZMP, nominal CoM, CoP, and actual CoM for x/y"
  LeanTest.assertEqual (spec.series.filter (fun item => item.figure == 2)).size 4
    "Figure 2 should plot planned and actual CoM velocity for x/y"
  LeanTest.assertEqual (spec.series.filter (fun item => item.figure == 3)).size 4
    "Figure 3 should plot policy and nominal CoM acceleration for x/y"

  LeanTest.assertTrue
    (hasPlotSeries spec.series 1 1 0 "desired_zmp" "r" "desired zmp")
    "Figure 1 x subplot should include desired_zmp row 0 in red"
  LeanTest.assertTrue
    (hasPlotSeries spec.series 1 2 1 "cop" "g" "planned cop")
    "Figure 1 y subplot should include cop row 1 in green"
  LeanTest.assertTrue
    (hasPlotSeries spec.series 2 1 2 "x" "c" "actual comd")
    "Figure 2 xd subplot should include actual state row 2 in cyan"
  LeanTest.assertTrue
    (hasPlotSeries spec.series 3 1 0 "u" "r" "comdd from policy")
    "Figure 3 xdd subplot should include policy control row 0 in red"
  LeanTest.assertTrue
    (hasPlotSeries spec.series 3 2 5 "nominal_com" "b." "nominal comdd")
    "Figure 3 ydd subplot should include nominal_com row 5 with Drake's b. style"

  LeanTest.assertTrue
    (result.graph.vertices.any (fun vertex =>
      vertex.id == plotOutputVertex && vertex.label.contains "lean-plot SVG"))
    "Skeleton graph should expose the lean-plot SVG output checkpoint"
  LeanTest.assertTrue
    (result.graph.moves.any (fun move =>
      move.targets == #[plotBoundaryVertex] &&
      move.reads == #[7004] &&
      move.writes == #[plotOutputVertex] &&
      move.label.contains "LeanPlot.Export.writeSvg"))
    "Skeleton graph should contain the exact PlotResults render boundary"

end Tests.EventSkeletonZmpExample
