import Tyr.EventSkeleton.SceneGraph

/-!
# Tyr.EventSkeleton.HardwareSim

Typed scenario and setup-plan primitives for Drake-style hardware simulation.

Drake's `examples/hardware_sim` is a diagram boundary, not a new dynamics
equation: it loads a typed scenario, builds a MultibodyPlant plus SceneGraph,
processes model directives, applies named initial positions, installs LCM
buses, drivers, cameras, and visualization, then runs a Simulator.

This module captures that boundary in EventSkeleton terms.  A future YAML
provider can populate `HardwareScenario`; the simulation setup logic can remain
typed and testable.
-/

namespace Tyr.EventSkeleton

structure HardwareSimulatorConfig where
  maxStepSize : Float := 1.0e-3
  accuracy : Float := 1.0e-2
  targetRealtimeRate : Float := 1.0
  deriving Repr, BEq, Inhabited

namespace HardwareSimulatorConfig

def validate? (config : HardwareSimulatorConfig) : Except String Unit := do
  if !(Float.isFinite config.maxStepSize) || config.maxStepSize <= 0.0 then
    .error s!"simulator max_step_size must be positive and finite, got {config.maxStepSize}"
  if !(Float.isFinite config.accuracy) || config.accuracy <= 0.0 then
    .error s!"simulator accuracy must be positive and finite, got {config.accuracy}"
  if !(Float.isFinite config.targetRealtimeRate) || config.targetRealtimeRate < 0.0 then
    .error s!"simulator target_realtime_rate must be nonnegative and finite, got {config.targetRealtimeRate}"

end HardwareSimulatorConfig

structure HardwarePlantConfig where
  timeStep : Float := 0.0
  stictionTolerance : Float := 0.001
  contactModel : String := ""
  deriving Repr, BEq, Inhabited

namespace HardwarePlantConfig

def validate? (config : HardwarePlantConfig) : Except String Unit := do
  if !(Float.isFinite config.timeStep) || config.timeStep < 0.0 then
    .error s!"plant time_step must be nonnegative and finite, got {config.timeStep}"
  if !(Float.isFinite config.stictionTolerance) || config.stictionTolerance < 0.0 then
    .error s!"plant stiction_tolerance must be nonnegative and finite, got {config.stictionTolerance}"

end HardwarePlantConfig

inductive HardwareComplianceType where
  | rigid
  | compliant
  deriving Repr, BEq, Inhabited

structure HardwareSceneGraphConfig where
  defaultProximityCompliance? : Option HardwareComplianceType := none
  deriving Repr, BEq, Inhabited

structure HardwareLcmBusConfig where
  name : String := "default"
  lcmUrl : String := ""
  deriving Repr, BEq, Inhabited

namespace HardwareLcmBusConfig

def validate? (bus : HardwareLcmBusConfig) : Except String Unit := do
  if bus.name.isEmpty then
    .error "LCM bus name cannot be empty"

end HardwareLcmBusConfig

inductive HardwareDriverConfig where
  | iiwa (handModelName : String := "") (lcmBus : String := "default")
  | schunkWsg (lcmBus : String := "default")
  | zeroForce
  deriving Repr, BEq, Inhabited

namespace HardwareDriverConfig

def lcmBusName : HardwareDriverConfig → Option String
  | .iiwa _ bus => some bus
  | .schunkWsg bus => some bus
  | .zeroForce => none

def label : HardwareDriverConfig → String
  | .iiwa hand bus => s!"IiwaDriver(hand={hand}, bus={bus})"
  | .schunkWsg bus => s!"SchunkWsgDriver(bus={bus})"
  | .zeroForce => "ZeroForceDriver"

def validate? (driver : HardwareDriverConfig) : Except String Unit := do
  match driver.lcmBusName with
  | some bus =>
      if bus.isEmpty then
        .error s!"{driver.label}: lcm_bus cannot be empty"
  | none => pure ()

end HardwareDriverConfig

structure HardwareModelDriver where
  modelName : String
  config : HardwareDriverConfig
  deriving Repr, BEq, Inhabited

namespace HardwareModelDriver

def validate? (driver : HardwareModelDriver) : Except String Unit := do
  if driver.modelName.isEmpty then
    .error "model driver key cannot be empty"
  driver.config.validate?

end HardwareModelDriver

structure HardwareJointPosition where
  modelName : String := ""
  jointName : String
  positions : Array Float
  deriving Repr, BEq, Inhabited

namespace HardwareJointPosition

def validate? (position : HardwareJointPosition) : Except String Unit := do
  if position.jointName.isEmpty then
    .error s!"named position for model {position.modelName}: joint name cannot be empty"
  if position.positions.isEmpty then
    .error s!"named position {position.modelName}::{position.jointName}: positions cannot be empty"
  for i in [:position.positions.size] do
    let q := position.positions[i]!
    if !(Float.isFinite q) then
      .error s!"named position {position.modelName}::{position.jointName}[{i}] must be finite, got {q}"

end HardwareJointPosition

inductive HardwareDirectiveKind where
  | addModel
  | addFrame
  | addWeld
  deriving Repr, BEq, Inhabited

structure HardwareModelDirective where
  kind : HardwareDirectiveKind
  name : String := ""
  file : String := ""
  parent : String := ""
  child : String := ""
  translation : SceneVec3 := {}
  rotationRpyDeg : SceneVec3 := {}
  defaultJointPositions : Array HardwareJointPosition := #[]
  deriving Repr, BEq, Inhabited

namespace HardwareModelDirective

def addModel
    (name file : String)
    (defaultJointPositions : Array HardwareJointPosition := #[]) :
    HardwareModelDirective :=
  { kind := .addModel, name := name, file := file, defaultJointPositions := defaultJointPositions }

def addFrame
    (name parent : String)
    (translation : SceneVec3 := {})
    (rotationRpyDeg : SceneVec3 := {}) : HardwareModelDirective :=
  { kind := .addFrame, name := name, parent := parent, translation := translation,
    rotationRpyDeg := rotationRpyDeg }

def addWeld (parent child : String) : HardwareModelDirective :=
  { kind := .addWeld, parent := parent, child := child }

def validate? (directive : HardwareModelDirective) : Except String Unit := do
  match directive.kind with
  | .addModel =>
      if directive.name.isEmpty then
        .error "add_model directive requires a model name"
      if directive.file.isEmpty then
        .error s!"add_model {directive.name}: file cannot be empty"
      for joint in directive.defaultJointPositions do
        joint.validate?
  | .addFrame =>
      if directive.name.isEmpty then
        .error "add_frame directive requires a frame name"
      if directive.parent.isEmpty then
        .error s!"add_frame {directive.name}: parent/base frame cannot be empty"
      if !directive.translation.isFinite || !directive.rotationRpyDeg.isFinite then
        .error s!"add_frame {directive.name}: transform entries must be finite"
  | .addWeld =>
      if directive.parent.isEmpty || directive.child.isEmpty then
        .error "add_weld directive requires nonempty parent and child"

end HardwareModelDirective

structure HardwareCameraConfig where
  key : String
  name : String
  lcmBus : String := "default"
  X_PB : ScenePose3 := {}
  deriving Repr, BEq, Inhabited

namespace HardwareCameraConfig

def validate? (camera : HardwareCameraConfig) : Except String Unit := do
  if camera.key.isEmpty then
    .error "camera map key cannot be empty"
  if camera.name.isEmpty then
    .error s!"camera {camera.key}: name cannot be empty"
  if camera.lcmBus.isEmpty then
    .error s!"camera {camera.key}: lcm_bus cannot be empty"
  camera.X_PB.validate? s!"camera {camera.key} pose"

end HardwareCameraConfig

structure HardwareVisualizationConfig where
  lcmBus : String := "default"
  publishPeriod : Float := 1.0 / 64.0
  defaultIllustrationColor? : Option SceneRgba := none
  deriving Repr, BEq, Inhabited

namespace HardwareVisualizationConfig

def validate? (config : HardwareVisualizationConfig) : Except String Unit := do
  if config.lcmBus.isEmpty then
    .error "visualization lcm_bus cannot be empty"
  if !(Float.isFinite config.publishPeriod) || config.publishPeriod <= 0.0 then
    .error s!"visualization publish_period must be positive and finite, got {config.publishPeriod}"
  match config.defaultIllustrationColor? with
  | some color => color.validate? "visualization default illustration color"
  | none => pure ()

end HardwareVisualizationConfig

structure HardwareScenario where
  randomSeed : Nat := 0
  simulationDuration : Float := 1.0 / 0.0
  simulatorConfig : HardwareSimulatorConfig := {}
  plantConfig : HardwarePlantConfig := {}
  sceneGraphConfig : HardwareSceneGraphConfig := {}
  directives : Array HardwareModelDirective := #[]
  lcmBuses : Array HardwareLcmBusConfig := #[{ name := "default" }]
  modelDrivers : Array HardwareModelDriver := #[]
  cameras : Array HardwareCameraConfig := #[]
  visualization : HardwareVisualizationConfig := {}
  initialPosition : Array HardwareJointPosition := #[]
  label : String := ""
  deriving Repr, Inhabited

namespace HardwareScenario

private def containsString (needle : String) (xs : Array String) : Bool :=
  xs.any (fun x => x == needle)

private def hasDuplicateString (xs : Array String) : Bool := Id.run do
  let mut seen : Array String := #[]
  for x in xs do
    if containsString x seen then
      return true
    seen := seen.push x
  return false

def busNames (scenario : HardwareScenario) : Array String :=
  scenario.lcmBuses.map (fun bus => bus.name)

def hasBus (scenario : HardwareScenario) (name : String) : Bool :=
  containsString name scenario.busNames

def directiveCountByKind (scenario : HardwareScenario) (kind : HardwareDirectiveKind) : Nat :=
  (scenario.directives.filter (fun directive => directive.kind == kind)).size

def validate? (scenario : HardwareScenario) : Except String Unit := do
  if !(scenario.simulationDuration >= 0.0) then
    .error s!"scenario {scenario.label}: simulation_duration must be nonnegative or +infinity, got {scenario.simulationDuration}"
  scenario.simulatorConfig.validate?
  scenario.plantConfig.validate?
  if scenario.lcmBuses.isEmpty then
    .error s!"scenario {scenario.label}: at least one LCM bus is required"
  if hasDuplicateString scenario.busNames then
    .error s!"scenario {scenario.label}: duplicate LCM bus name"
  for bus in scenario.lcmBuses do
    bus.validate?
  for directive in scenario.directives do
    directive.validate?
  for driver in scenario.modelDrivers do
    driver.validate?
    match driver.config.lcmBusName with
    | some bus =>
        if !scenario.hasBus bus then
          .error s!"scenario {scenario.label}: driver {driver.modelName} references missing LCM bus {bus}"
    | none => pure ()
  for camera in scenario.cameras do
    camera.validate?
    if !scenario.hasBus camera.lcmBus then
      .error s!"scenario {scenario.label}: camera {camera.key} references missing LCM bus {camera.lcmBus}"
  scenario.visualization.validate?
  if !scenario.hasBus scenario.visualization.lcmBus then
    .error s!"scenario {scenario.label}: visualization references missing LCM bus {scenario.visualization.lcmBus}"
  for position in scenario.initialPosition do
    if position.modelName.isEmpty then
      .error s!"scenario {scenario.label}: initial_position entries require model names"
    position.validate?

def withSimulationDuration (scenario : HardwareScenario) (duration : Float) :
    HardwareScenario :=
  { scenario with simulationDuration := duration }

end HardwareScenario

inductive HardwareSetupStepKind where
  | addPlantAndSceneGraph
  | processDirectives
  | applyInitialPositions
  | finalizePlant
  | applyLcmBuses
  | applyDriverConfigs
  | applyCameraConfigs
  | applyVisualization
  | buildDiagram
  | applySimulatorConfig
  | setRandomContext
  | writeGraphviz
  | advanceTo
  deriving Repr, BEq, Inhabited

structure HardwareSetupStep where
  kind : HardwareSetupStepKind
  label : String := ""
  count : Nat := 0
  deriving Repr, BEq, Inhabited

structure HardwareSimulationPlan where
  scenario : HardwareScenario
  steps : Array HardwareSetupStep
  graph : SkeletonGraph
  graphvizRequested : Bool := false
  deriving Repr, Inhabited

namespace HardwareSimulationPlan

def stepKinds (plan : HardwareSimulationPlan) : Array HardwareSetupStepKind :=
  plan.steps.map (fun step => step.kind)

def containsStep (plan : HardwareSimulationPlan) (kind : HardwareSetupStepKind) : Bool :=
  plan.steps.any (fun step => step.kind == kind)

end HardwareSimulationPlan

namespace HardwareScenario

def setupSteps (scenario : HardwareScenario) (graphvizRequested : Bool := false) :
    Array HardwareSetupStep := Id.run do
  let mut steps : Array HardwareSetupStep := #[
    { kind := .addPlantAndSceneGraph, label := "AddMultibodyPlant", count := 2 },
    { kind := .processDirectives, label := "ProcessModelDirectives", count := scenario.directives.size },
    { kind := .applyInitialPositions, label := "ApplyNamedPositionsAsDefaults", count := scenario.initialPosition.size },
    { kind := .finalizePlant, label := "MultibodyPlant.Finalize", count := 1 },
    { kind := .applyLcmBuses, label := "ApplyLcmBusConfig", count := scenario.lcmBuses.size },
    { kind := .applyDriverConfigs, label := "ApplyDriverConfigs", count := scenario.modelDrivers.size },
    { kind := .applyCameraConfigs, label := "ApplyCameraConfig", count := scenario.cameras.size },
    { kind := .applyVisualization, label := "ApplyVisualizationConfig", count := 1 },
    { kind := .buildDiagram, label := "DiagramBuilder.Build", count := 1 },
    { kind := .applySimulatorConfig, label := "ApplySimulatorConfig", count := 1 },
    { kind := .setRandomContext, label := "Diagram.SetRandomContext", count := scenario.randomSeed }
  ]
  if graphvizRequested then
    steps := steps.push { kind := .writeGraphviz, label := "Diagram.GetGraphvizString", count := 1 }
  steps := steps.push { kind := .advanceTo, label := "Simulator.AdvanceTo", count := 1 }
  return steps

def skeletonGraph (_scenario : HardwareScenario) (graphvizRequested : Bool := false) :
    SkeletonGraph :=
  let graphvizMove :=
    if graphvizRequested then
      #[{
        kind := .checkpointBoundary
        targets := #[112]
        reads := #[109]
        writes := #[112]
        label := "write hardware_sim Graphviz diagram"
      }]
    else
      #[]
  {
    vertices := #[
      { id := 100, kind := .opaque, label := "typed HardwareScenario" },
      { id := 101, kind := .opaque, label := "MultibodyPlant" },
      { id := 102, kind := .opaque, label := "SceneGraph" },
      { id := 103, kind := .opaque, label := "model directives" },
      { id := 104, kind := .state .boundary, label := "initial positions" },
      { id := 105, kind := .opaque, label := "LCM buses" },
      { id := 106, kind := .opaque, label := "model drivers" },
      { id := 107, kind := .opaque, label := "cameras" },
      { id := 108, kind := .opaque, label := "visualization" },
      { id := 109, kind := .opaque, label := "Diagram" },
      { id := 110, kind := .state .checkpoint, label := "randomized simulator context" },
      { id := 111, kind := .interval, label := "hardware simulation interval" },
      { id := 112, kind := .checkpoint, label := "Graphviz output" }
    ]
    moves := #[
      {
        kind := .localSchurBlock
        targets := #[101, 102]
        reads := #[100]
        writes := #[101, 102]
        label := "AddMultibodyPlant with SceneGraph"
      },
      {
        kind := .localSchurBlock
        targets := #[103]
        reads := #[100, 101]
        writes := #[101, 103]
        label := "ProcessModelDirectives"
      },
      {
        kind := .checkpointBoundary
        targets := #[104]
        reads := #[100, 101]
        writes := #[101]
        label := "ApplyNamedPositionsAsDefaults before Finalize"
      },
      {
        kind := .freezeControl
        targets := #[105]
        reads := #[100]
        writes := #[105]
        label := "ApplyLcmBusConfig external hardware boundary"
      },
      {
        kind := .localSchurBlock
        targets := #[106]
        reads := #[100, 101, 103, 105]
        writes := #[106]
        label := "ApplyDriverConfigs"
      },
      {
        kind := .localSchurBlock
        targets := #[107]
        reads := #[100, 102, 105]
        writes := #[107]
        label := "ApplyCameraConfig"
      },
      {
        kind := .checkpointBoundary
        targets := #[108]
        reads := #[100, 102, 105]
        writes := #[108]
        label := "ApplyVisualizationConfig"
      },
      {
        kind := .localSchurBlock
        targets := #[109]
        reads := #[101, 102, 105, 106, 107, 108]
        writes := #[109]
        label := "DiagramBuilder.Build"
      },
      {
        kind := .checkpointBoundary
        targets := #[110]
        reads := #[100, 109]
        writes := #[110]
        label := "ApplySimulatorConfig and SetRandomContext"
      }
    ] ++ graphvizMove ++ #[
      {
        kind := .intervalAdjoint
        targets := #[111]
        reads := #[109, 110]
        writes := #[110]
        label := "Simulator.AdvanceTo simulation_duration"
      }
    ]
  }

def plan? (scenario : HardwareScenario) (graphvizRequested : Bool := false) :
    Except String HardwareSimulationPlan := do
  scenario.validate?
  pure {
    scenario := scenario
    steps := scenario.setupSteps graphvizRequested
    graph := scenario.skeletonGraph graphvizRequested
    graphvizRequested := graphvizRequested
  }

end HardwareScenario

end Tyr.EventSkeleton
