import Tyr.EventSkeleton.Trace

/-!
# Drake ZMP Event-Skeleton Example

This ports the executable shape of `../drake/examples/zmp/zmp_example.cc` and
the helper routines in `../drake/planning/locomotion/test_utilities`.

The implemented physics primitive is the linear inverted pendulum model (LIPM):
the state is `[com_x, com_y, comd_x, comd_y]`, the control is CoM
acceleration, and the center of pressure is

`cop = com - height / gravity * comdd`.

Drake's full `ZmpPlanner::Plan` computes an HJB/LQR backward pass with matrix
exponentials.  That optimization block is kept visible as a local Schur block
here.  The executable policy below is a stabilizing LIPM tracking policy, so
the trace marks the planner block as a controlled approximation rather than
pretending to have ported the entire HJB primitive.
-/

namespace Tyr.EventSkeleton.Examples.Zmp

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/zmp/zmp_example.cc"
      concept := "builds footsteps, desired ZMP trajectories, plans a ZMP policy, simulates LIPM, and plots CoM/COP"
    },
    {
      path := "../drake/planning/locomotion/zmp_planner.h"
      concept := "declares the LIPM dynamics, CoP output, HJB value function, and linear policy API"
    },
    {
      path := "../drake/planning/locomotion/zmp_planner.cc"
      concept := "implements the HJB/LQR backward pass and nominal CoM trajectory"
    },
    {
      path := "../drake/planning/locomotion/test_utilities/zmp_test_util.cc"
      concept := "implements GenerateDesiredZmpTrajs and explicit-Euler SimulateZmpPolicy"
    },
    {
      path := "../drake/planning/locomotion/test/zmp_planner_test.cc"
      concept := "checks HJB equations, policy consistency, and convergence of perturbed LIPM rollouts"
    }
  ]

structure Vec2 where
  x : Float := 0.0
  y : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace Vec2

def add (a b : Vec2) : Vec2 :=
  { x := a.x + b.x, y := a.y + b.y }

def sub (a b : Vec2) : Vec2 :=
  { x := a.x - b.x, y := a.y - b.y }

def scale (s : Float) (v : Vec2) : Vec2 :=
  { x := s * v.x, y := s * v.y }

def normInf (v : Vec2) : Float :=
  max (Float.abs v.x) (Float.abs v.y)

def isFinite (v : Vec2) : Bool :=
  Float.isFinite v.x && Float.isFinite v.y

def toArray (v : Vec2) : Array Float :=
  #[v.x, v.y]

end Vec2

structure ZmpState where
  com : Vec2 := {}
  comd : Vec2 := {}
  deriving Repr, BEq, Inhabited

namespace ZmpState

def zero : ZmpState := {}

def toArray (x : ZmpState) : Array Float :=
  #[x.com.x, x.com.y, x.comd.x, x.comd.y]

def isFinite (x : ZmpState) : Bool :=
  x.com.isFinite && x.comd.isFinite

end ZmpState

def stateCoordinateNames : Array String :=
  #["com_x", "com_y", "comd_x", "comd_y"]

def inputCoordinateNames : Array String :=
  #["comdd_x", "comdd_y"]

def outputCoordinateNames : Array String :=
  #["cop_x", "cop_y"]

def nominalCoordinateNames : Array String :=
  #["nominal_com_x", "nominal_com_y", "nominal_comd_x", "nominal_comd_y",
    "nominal_comdd_x", "nominal_comdd_y"]

inductive ZmpPlotBackendKind where
  | drakeCallPython
  | leanPlotSvg
  deriving Repr, BEq, Inhabited

structure ZmpPlotBackend where
  kind : ZmpPlotBackendKind := .leanPlotSvg
  libraryPath : String := "../lean-plot"
  moduleName : String := "LeanPlot"
  exportFunction : String := "LeanPlot.Export.writeSvg"
  outputStem : String := "zmp-rollout"
  deriving Repr, BEq, Inhabited

namespace ZmpPlotBackend

def validate? (backend : ZmpPlotBackend) : Except String Unit := do
  if backend.kind == .leanPlotSvg then
    if backend.libraryPath != "../lean-plot" then
      .error s!"ZMP lean-plot backend should point at ../lean-plot, got {backend.libraryPath}"
    if backend.moduleName != "LeanPlot" then
      .error s!"ZMP lean-plot backend should import LeanPlot, got {backend.moduleName}"
    if backend.exportFunction != "LeanPlot.Export.writeSvg" then
      .error s!"ZMP lean-plot backend should render SVG through LeanPlot.Export.writeSvg, got {backend.exportFunction}"
    if backend.outputStem.isEmpty then
      .error "ZMP lean-plot backend requires an output stem"

end ZmpPlotBackend

structure ZmpPlotSeries where
  figure : Nat
  subplot : Nat
  yLabel : String
  source : String
  row : Nat
  style : String
  legend : String
  leanPlotMark : String := "Mark.lineSeries"
  deriving Repr, BEq, Inhabited

namespace ZmpPlotSeries

def validate? (series : ZmpPlotSeries) : Except String Unit := do
  if series.figure == 0 then
    .error "ZMP plot figure numbers are one-based"
  if series.subplot == 0 then
    .error "ZMP plot subplot numbers are one-based"
  if series.source.isEmpty then
    .error "ZMP plot source must name a ZmpTestTraj matrix"
  if series.style.isEmpty then
    .error "ZMP plot style must match the Drake CallPython plot style"
  if series.legend.isEmpty then
    .error "ZMP plot legend must match the Drake PlotResults legend"
  if series.leanPlotMark != "Mark.lineSeries" then
    .error s!"ZMP plot series should lower to LeanPlot Mark.lineSeries, got {series.leanPlotMark}"

end ZmpPlotSeries

def plotSeriesSpec : Array ZmpPlotSeries :=
  #[
    { figure := 1, subplot := 1, yLabel := "x [m]", source := "desired_zmp", row := 0, style := "r", legend := "desired zmp" },
    { figure := 1, subplot := 1, yLabel := "x [m]", source := "nominal_com", row := 0, style := "b", legend := "planned com" },
    { figure := 1, subplot := 1, yLabel := "x [m]", source := "cop", row := 0, style := "g", legend := "planned cop" },
    { figure := 1, subplot := 1, yLabel := "x [m]", source := "x", row := 0, style := "c", legend := "actual com" },
    { figure := 1, subplot := 2, yLabel := "y [m]", source := "desired_zmp", row := 1, style := "r", legend := "desired zmp" },
    { figure := 1, subplot := 2, yLabel := "y [m]", source := "nominal_com", row := 1, style := "b", legend := "planned com" },
    { figure := 1, subplot := 2, yLabel := "y [m]", source := "cop", row := 1, style := "g", legend := "planned cop" },
    { figure := 1, subplot := 2, yLabel := "y [m]", source := "x", row := 1, style := "c", legend := "actual com" },
    { figure := 2, subplot := 1, yLabel := "xd [m/s]", source := "nominal_com", row := 2, style := "b", legend := "planned comd" },
    { figure := 2, subplot := 1, yLabel := "xd [m/s]", source := "x", row := 2, style := "c", legend := "actual comd" },
    { figure := 2, subplot := 2, yLabel := "yd [m/s]", source := "nominal_com", row := 3, style := "b", legend := "planned comd" },
    { figure := 2, subplot := 2, yLabel := "yd [m/s]", source := "x", row := 3, style := "c", legend := "actual comd" },
    { figure := 3, subplot := 1, yLabel := "xdd [m/s2]", source := "u", row := 0, style := "r", legend := "comdd from policy" },
    { figure := 3, subplot := 1, yLabel := "xdd [m/s2]", source := "nominal_com", row := 4, style := "b.", legend := "nominal comdd" },
    { figure := 3, subplot := 2, yLabel := "ydd [m/s2]", source := "u", row := 1, style := "r", legend := "comdd from policy" },
    { figure := 3, subplot := 2, yLabel := "ydd [m/s2]", source := "nominal_com", row := 5, style := "b.", legend := "nominal comdd" }
  ]

structure ZmpPlotSpec where
  sourceStruct : String := "planning::ZmpTestTraj"
  drakeFunction : String := "PlotResults"
  backend : ZmpPlotBackend := {}
  figures : Nat := 3
  subplotsPerFigure : Nat := 2
  series : Array ZmpPlotSeries := plotSeriesSpec
  deriving Repr, BEq, Inhabited

namespace ZmpPlotSpec

def validate? (spec : ZmpPlotSpec) : Except String Unit := do
  if spec.sourceStruct != "planning::ZmpTestTraj" then
    .error s!"ZMP plot source struct should be planning::ZmpTestTraj, got {spec.sourceStruct}"
  if spec.drakeFunction != "PlotResults" then
    .error s!"ZMP plot should mirror Drake PlotResults, got {spec.drakeFunction}"
  spec.backend.validate?
  if spec.figures != 3 then
    .error s!"Drake PlotResults should have 3 figures, got {spec.figures}"
  if spec.subplotsPerFigure != 2 then
    .error s!"Drake PlotResults should have 2 subplots per figure, got {spec.subplotsPerFigure}"
  if spec.series.size != 16 then
    .error s!"Drake PlotResults should have 16 plotted ZmpTestTraj series, got {spec.series.size}"
  for item in spec.series do
    item.validate?

end ZmpPlotSpec

def zmpPlotSpec : ZmpPlotSpec := {}

structure ZmpParams where
  height : Float := 1.0
  gravity : Float := 9.81
  qy : Float := 1.0
  r : Float := 0.1
  sampleDt : Float := 0.01
  extraTimeAtEnd : Float := 2.0
  doubleSupportDuration : Float := 0.5
  singleSupportDuration : Float := 1.0
  deriving Repr, Inhabited

namespace ZmpParams

def validate? (p : ZmpParams) : Except String Unit := do
  if !(Float.isFinite p.height) || p.height <= 0.0 then
    .error s!"ZMP height must be positive and finite, got {p.height}"
  if !(Float.isFinite p.gravity) || p.gravity <= 0.0 then
    .error s!"ZMP gravity must be positive and finite, got {p.gravity}"
  if !(Float.isFinite p.qy) || p.qy <= 0.0 then
    .error s!"ZMP Qy scalar must be positive and finite, got {p.qy}"
  if !(Float.isFinite p.r) || p.r <= 0.0 then
    .error s!"ZMP R scalar must be positive and finite, got {p.r}"
  if !(Float.isFinite p.sampleDt) || p.sampleDt <= 0.0 then
    .error s!"ZMP sample_dt must be positive and finite, got {p.sampleDt}"
  if !(Float.isFinite p.extraTimeAtEnd) || p.extraTimeAtEnd < 0.0 then
    .error s!"ZMP extra_time_at_the_end must be nonnegative and finite, got {p.extraTimeAtEnd}"
  if !(Float.isFinite p.doubleSupportDuration) || p.doubleSupportDuration < 0.0 then
    .error s!"ZMP double support duration must be nonnegative and finite, got {p.doubleSupportDuration}"
  if !(Float.isFinite p.singleSupportDuration) || p.singleSupportDuration <= 0.0 then
    .error s!"ZMP single support duration must be positive and finite, got {p.singleSupportDuration}"

def omega (p : ZmpParams) : Float :=
  Float.sqrt (p.gravity / p.height)

def positionGain (p : ZmpParams) : Float :=
  let w := p.omega
  w * w

def velocityGain (p : ZmpParams) : Float :=
  2.0 * p.omega

end ZmpParams

def params : ZmpParams := {}

inductive ZmpInterpolationKind where
  | zeroOrderHold
  | firstOrderHold
  | cubicShapePreserving
  deriving Repr, BEq, Inhabited

namespace ZmpInterpolationKind

def label : ZmpInterpolationKind → String
  | .zeroOrderHold => "zero-order-hold"
  | .firstOrderHold => "first-order-hold"
  | .cubicShapePreserving => "cubic-shape-preserving"

end ZmpInterpolationKind

structure ZmpTrajectory where
  kind : ZmpInterpolationKind := .zeroOrderHold
  times : Array Float := #[]
  knots : Array Vec2 := #[]
  deriving Repr, Inhabited

namespace ZmpTrajectory

def startTime (traj : ZmpTrajectory) : Float :=
  traj.times.getD 0 0.0

def endTime (traj : ZmpTrajectory) : Float :=
  if traj.times.isEmpty then 0.0 else traj.times[traj.times.size - 1]!

def validate? (traj : ZmpTrajectory) : Except String Unit := do
  if traj.times.size != traj.knots.size then
    .error s!"ZMP trajectory time count {traj.times.size} != knot count {traj.knots.size}"
  if traj.times.size < 2 then
    .error "ZMP trajectory requires at least two knots"
  for i in [:traj.times.size] do
    let t := traj.times[i]!
    if !(Float.isFinite t) then
      .error s!"ZMP trajectory time {i} must be finite, got {t}"
    if i > 0 && !(traj.times[i - 1]! < t) then
      .error s!"ZMP trajectory times must be strictly increasing at index {i}"
    if !traj.knots[i]!.isFinite then
      .error s!"ZMP trajectory knot {i} is not finite"

private def findSegment (traj : ZmpTrajectory) (time : Float) : Nat := Id.run do
  if time <= traj.startTime then
    return 0
  if time >= traj.endTime then
    return traj.times.size - 2
  let lastSegment := traj.times.size - 2
  for i in [:lastSegment + 1] do
    if time < traj.times[i + 1]! then
      return i
  return lastSegment

private def smoothStep (s : Float) : Float :=
  s * s * (3.0 - 2.0 * s)

def value (traj : ZmpTrajectory) (time : Float) : Vec2 :=
  if traj.times.size < 2 || traj.knots.size < 2 then
    {}
  else if time <= traj.startTime then
    traj.knots[0]!
  else if time >= traj.endTime then
    traj.knots[traj.knots.size - 1]!
  else
    let i := traj.findSegment time
    let t0 := traj.times[i]!
    let t1 := traj.times[i + 1]!
    let y0 := traj.knots[i]!
    let y1 := traj.knots[i + 1]!
    let s := (time - t0) / (t1 - t0)
    match traj.kind with
    | .zeroOrderHold => y0
    | .firstOrderHold =>
        y0.add ((y1.sub y0).scale s)
    | .cubicShapePreserving =>
        y0.add ((y1.sub y0).scale (smoothStep s))

def velocity (traj : ZmpTrajectory) (time : Float) : Vec2 :=
  if traj.times.size < 2 || traj.knots.size < 2 then
    {}
  else if time < traj.startTime || time >= traj.endTime then
    {}
  else
    let i := traj.findSegment time
    let t0 := traj.times[i]!
    let t1 := traj.times[i + 1]!
    let y0 := traj.knots[i]!
    let y1 := traj.knots[i + 1]!
    let dt := t1 - t0
    let s := (time - t0) / dt
    match traj.kind with
    | .zeroOrderHold => {}
    | .firstOrderHold => (y1.sub y0).scale (1.0 / dt)
    | .cubicShapePreserving =>
        (y1.sub y0).scale ((6.0 * s * (1.0 - s)) / dt)

end ZmpTrajectory

def defaultFootsteps : Array Vec2 :=
  #[
    { x := 0.0, y := 0.0 },
    { x := 0.5, y := 0.1 },
    { x := 1.0, y := -0.1 },
    { x := 1.5, y := 0.1 },
    { x := 2.0, y := -0.1 },
    { x := 2.5, y := 0.0 }
  ]

private def zmpTimesAndKnots?
    (footsteps : Array Vec2)
    (doubleSupportDuration singleSupportDuration : Float) :
    Except String (Array Float × Array Vec2) := do
  if footsteps.isEmpty then
    .error "GenerateDesiredZmpTrajs requires at least one footstep"
  if !(Float.isFinite doubleSupportDuration) || doubleSupportDuration < 0.0 then
    .error s!"double_support_duration must be nonnegative and finite, got {doubleSupportDuration}"
  if !(Float.isFinite singleSupportDuration) || singleSupportDuration <= 0.0 then
    .error s!"single_support_duration must be positive and finite, got {singleSupportDuration}"
  for i in [:footsteps.size] do
    if !footsteps[i]!.isFinite then
      .error s!"footstep {i} is not finite"
  let mut time := 0.0
  let mut times := #[time]
  let mut knots := #[footsteps[0]!]
  time := time + singleSupportDuration
  times := times.push time
  knots := knots.push footsteps[0]!
  for i in [1:footsteps.size] do
    time := time + doubleSupportDuration
    times := times.push time
    knots := knots.push footsteps[i]!
    time := time + singleSupportDuration
    times := times.push time
    knots := knots.push footsteps[i]!
  pure (times, knots)

def generateDesiredZmpTrajs?
    (footsteps : Array Vec2 := defaultFootsteps)
    (doubleSupportDuration : Float := params.doubleSupportDuration)
    (singleSupportDuration : Float := params.singleSupportDuration) :
    Except String (Array ZmpTrajectory) := do
  let (times, knots) ← zmpTimesAndKnots? footsteps doubleSupportDuration singleSupportDuration
  pure #[
    { kind := .zeroOrderHold, times := times, knots := knots },
    { kind := .firstOrderHold, times := times, knots := knots },
    { kind := .cubicShapePreserving, times := times, knots := knots }
  ]

structure LipmMatrices where
  A : Array (Array Float)
  B : Array (Array Float)
  C : Array (Array Float)
  D : Array (Array Float)
  deriving Repr, Inhabited

def lipmMatrices (p : ZmpParams := params) : LipmMatrices :=
  {
    A := #[
      #[0.0, 0.0, 1.0, 0.0],
      #[0.0, 0.0, 0.0, 1.0],
      #[0.0, 0.0, 0.0, 0.0],
      #[0.0, 0.0, 0.0, 0.0]
    ]
    B := #[
      #[0.0, 0.0],
      #[0.0, 0.0],
      #[1.0, 0.0],
      #[0.0, 1.0]
    ]
    C := #[
      #[1.0, 0.0, 0.0, 0.0],
      #[0.0, 1.0, 0.0, 0.0]
    ]
    D := #[
      #[-p.height / p.gravity, 0.0],
      #[0.0, -p.height / p.gravity]
    ]
  }

structure ZmpPlannerState where
  params : ZmpParams := {}
  desired : ZmpTrajectory := {}
  planned : Bool := false
  policyExactness : MoveExactness := .controlledApproximation
  deriving Repr, Inhabited

namespace ZmpPlannerState

def validate? (planner : ZmpPlannerState) : Except String Unit := do
  planner.params.validate?
  planner.desired.validate?
  if !planner.planned then
    .error "ZMP planner has not been planned"

def desiredZmp (planner : ZmpPlannerState) (time : Float) : Vec2 :=
  planner.desired.value time

def desiredZmpVelocity (planner : ZmpPlannerState) (time : Float) : Vec2 :=
  planner.desired.velocity time

def nominalState (planner : ZmpPlannerState) (time : Float) : ZmpState :=
  {
    com := planner.desiredZmp time
    comd := planner.desiredZmpVelocity time
  }

def computeCoMdd (planner : ZmpPlannerState) (time : Float) (x : ZmpState) :
    Vec2 :=
  let desired := planner.desiredZmp time
  let desiredVelocity := planner.desiredZmpVelocity time
  let kp := planner.params.positionGain
  let kd := planner.params.velocityGain
  (desired.sub x.com).scale kp |>.add ((desiredVelocity.sub x.comd).scale kd)

def comddToCop (planner : ZmpPlannerState) (x : ZmpState) (u : Vec2) : Vec2 :=
  x.com.sub (u.scale (planner.params.height / planner.params.gravity))

def nominalComdd (planner : ZmpPlannerState) (time : Float) : Vec2 :=
  planner.computeCoMdd time (planner.nominalState time)

def localPlannerMove : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[7001]
    reads := #[7000]
    writes := #[7001]
    exactness := .controlledApproximation
    label := "ZmpPlanner.Plan HJB policy block represented by LIPM tracking policy"
  }

end ZmpPlannerState

def planTrackingPolicy?
    (desired : ZmpTrajectory)
    (p : ZmpParams := params) : Except String ZmpPlannerState := do
  p.validate?
  desired.validate?
  pure {
    params := p
    desired := desired
    planned := true
    policyExactness := .controlledApproximation
  }

structure ZmpSample where
  time : Float
  desiredZmp : Vec2
  nominalState : ZmpState
  nominalComdd : Vec2
  state : ZmpState
  control : Vec2
  cop : Vec2
  deriving Repr, Inhabited

namespace ZmpSample

def nominalArray (s : ZmpSample) : Array Float :=
  #[s.nominalState.com.x, s.nominalState.com.y,
    s.nominalState.comd.x, s.nominalState.comd.y,
    s.nominalComdd.x, s.nominalComdd.y]

end ZmpSample

structure ZmpRollout where
  samples : Array ZmpSample := #[]
  trace : DynamicEventTrace := {}
  moves : Array SkeletonMove := #[]
  deriving Repr, Inhabited

namespace ZmpRollout

def finalSample? (rollout : ZmpRollout) : Option ZmpSample :=
  if rollout.samples.isEmpty then none else some rollout.samples[rollout.samples.size - 1]!

end ZmpRollout

private def stepState (dt : Float) (x : ZmpState) (u : Vec2) : ZmpState :=
  {
    com := x.com.add (x.comd.scale dt)
    comd := x.comd.add (u.scale dt)
  }

def simulateZmpPolicy?
    (planner : ZmpPlannerState)
    (x0 : ZmpState)
    (dt : Float)
    (extraTimeAtEnd : Float) :
    Except String ZmpRollout := do
  planner.validate?
  if !x0.isFinite then
    .error "ZMP initial state is not finite"
  if !(Float.isFinite dt) || dt <= 0.0 then
    .error s!"ZMP simulation dt must be positive and finite, got {dt}"
  if !(Float.isFinite extraTimeAtEnd) || extraTimeAtEnd < 0.0 then
    .error s!"ZMP extra time must be nonnegative and finite, got {extraTimeAtEnd}"
  let totalTime := planner.desired.endTime - planner.desired.startTime + extraTimeAtEnd
  let n := Nat.max 1 (Float.toUInt64 (totalTime / dt)).toNat
  let mut x := x0
  let mut samples : Array ZmpSample := #[]
  for i in [:n] do
    let time := planner.desired.startTime + i.toFloat * dt
    let u := planner.computeCoMdd time x
    let sample : ZmpSample := {
      time := time
      desiredZmp := planner.desiredZmp time
      nominalState := planner.nominalState time
      nominalComdd := planner.nominalComdd time
      state := x
      control := u
      cop := planner.comddToCop x u
    }
    samples := samples.push sample
    x := stepState dt x u
  let segment : AcceptedStepSegment := {
    id := 0
    attemptIndex := 0
    tStart := planner.desired.startTime
    tAttempt := planner.desired.endTime + extraTimeAtEnd
    tAfter := planner.desired.endTime + extraTimeAtEnd
    label := "zmp-lipm-policy-rollout"
  }
  let trace := DynamicEventTrace.empty.push (.interval segment)
  trace.validate?
  pure {
    samples := samples
    trace := trace
    moves := #[ZmpPlannerState.localPlannerMove] ++ trace.moves
  }

structure SimulationResult where
  references : Array DrakeReference
  footsteps : Array Vec2
  desiredTrajectories : Array ZmpTrajectory
  planner : ZmpPlannerState
  initialPlanState : ZmpState
  initialSimulationState : ZmpState
  rollout : ZmpRollout
  plotSpec : ZmpPlotSpec
  graph : SkeletonGraph
  deriving Repr, Inhabited

def plotBoundaryVertex : VertexId := 7005

def plotOutputVertex : VertexId := 7006

def plotBoundaryMove (spec : ZmpPlotSpec := zmpPlotSpec) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[plotBoundaryVertex]
    reads := #[7004]
    writes := #[plotOutputVertex]
    exactness := .exact
    cost := { work := spec.series.size.toFloat, memory := 1.0 }
    label := s!"{spec.drakeFunction} lowered to {spec.backend.exportFunction} over {spec.sourceStruct}"
  }

def exampleGraph : SkeletonGraph :=
  {
    vertices := #[
      { id := 7000, kind := .state .boundary, label := "desired ZMP trajectory" },
      { id := 7001, kind := .opaque, label := "ZmpPlanner policy/value block" },
      { id := 7002, kind := .state .checkpoint, label := "perturbed LIPM initial state" },
      { id := 7003, kind := .interval, label := "explicit-Euler LIPM rollout" },
      { id := 7004, kind := .state .checkpoint, label := "ZMP rollout samples" },
      { id := plotBoundaryVertex, kind := .state .boundary, label := "ZMP PlotResults render boundary" },
      { id := plotOutputVertex, kind := .state .checkpoint, label := "lean-plot SVG output" }
    ]
    moves := #[
      ZmpPlannerState.localPlannerMove,
      {
        kind := .intervalAdjoint
        targets := #[7003]
        reads := #[7001, 7002]
        writes := #[7004]
        label := "SimulateZmpPolicy explicit-Euler LIPM interval"
      },
      {
        kind := .checkpointBoundary
        targets := #[7004]
        reads := #[7003]
        writes := #[7004]
        label := "store ZmpTestTraj samples"
      },
      plotBoundaryMove zmpPlotSpec
    ]
  }

def buildEndToEnd? (p : ZmpParams := params) :
    Except String SimulationResult := do
  zmpPlotSpec.validate?
  let trajectories ← generateDesiredZmpTrajs?
    defaultFootsteps p.doubleSupportDuration p.singleSupportDuration
  let desired := trajectories[0]!
  let initialPlanState : ZmpState := ZmpState.zero
  let planner ← planTrackingPolicy? desired p
  let initialSimulationState : ZmpState := {
    com := { x := 0.0, y := 0.0 }
    comd := { x := 0.2, y := -0.1 }
  }
  let rollout ← simulateZmpPolicy?
    planner initialSimulationState p.sampleDt p.extraTimeAtEnd
  pure {
    references := drakeReferences
    footsteps := defaultFootsteps
    desiredTrajectories := trajectories
    planner := planner
    initialPlanState := initialPlanState
    initialSimulationState := initialSimulationState
    rollout := rollout
    plotSpec := zmpPlotSpec
    graph := exampleGraph
  }

end Tyr.EventSkeleton.Examples.Zmp
