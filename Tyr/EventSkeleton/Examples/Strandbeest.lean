import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Trace

/-!
# Drake-Style Strandbeest Example

This ports the structure of
`../drake/examples/multibody/strandbeest/run_with_motor.cc`.

The Xacro-derived loop closures, motor law, plant configuration, and initial
context are explicit primitives.  The plant advance is represented by the
shared full-physics primitive equation so the example remains executable through
the event-skeleton physics layer while retaining a clear boundary for future
Drake-grade SAP and loop-closure solvers.
-/

namespace Tyr.EventSkeleton.Examples.Strandbeest

open Tyr.EventSkeleton

private def pi : Float := 3.14159265358979323846

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/multibody/strandbeest/BUILD.bazel"
      concept := "declares the StrandbeestModels xacro_filegroup, run_with_motor binary, and Python smoke test"
    },
    {
      path := "../drake/examples/multibody/strandbeest/run_with_motor.cc"
      concept := "builds MultibodyPlant/SceneGraph, selects constraints or bushings, runs IK, installs the desired-velocity motor, and advances the simulator"
    },
    {
      path := "../drake/examples/multibody/strandbeest/test/run_with_motor_test.py"
      concept := "smoke-tests the run_with_motor executable by calling it with --simulation_time=0.1 and requiring subprocess.check_call to return"
    },
    {
      path := "../drake/examples/multibody/strandbeest/README.md"
      concept := "describes the Theo Jansen Strandbeest demo and Xacro-generated URDF models"
    },
    {
      path := "../drake/examples/multibody/strandbeest/model/Strandbeest.xacro"
      concept := "declares floor, crossbar, crank axle, the crank transmission, and six phase-shifted leg pairs"
    },
    {
      path := "../drake/examples/multibody/strandbeest/model/StrandbeestConstraints.urdf.xacro"
      concept := "generated-URDF entry point that expands Strandbeest.xacro with ball constraints enabled"
    },
    {
      path := "../drake/examples/multibody/strandbeest/model/StrandbeestBushings.urdf.xacro"
      concept := "generated-URDF entry point that expands Strandbeest.xacro with LinearBushingRollPitchYaw loops enabled"
    },
    {
      path := "../drake/examples/multibody/strandbeest/model/LegPair.xacro"
      concept := "declares pair-level A-C loop closures or equivalent LinearBushingRollPitchYaw elements"
    },
    {
      path := "../drake/examples/multibody/strandbeest/model/LegAssembly.xacro"
      concept := "declares per-leg F-G and B-C loop closures or equivalent LinearBushingRollPitchYaw elements"
    },
    {
      path := "../drake/examples/multibody/strandbeest/model/Macros.xacro"
      concept := "declares PVC mass density and the bushing stiffness/damping constants"
    }
  ]

inductive LoopClosureMode where
  | constraints
  | bushings
  deriving Repr, BEq, Inhabited

namespace LoopClosureMode

def label : LoopClosureMode → String
  | .constraints => "constraints"
  | .bushings => "bushings"

def withConstraints : LoopClosureMode → Bool
  | .constraints => true
  | .bushings => false

def urdfUrl : LoopClosureMode → String
  | .constraints =>
      "package://drake/examples/multibody/strandbeest/model/StrandbeestConstraints.urdf"
  | .bushings =>
      "package://drake/examples/multibody/strandbeest/model/StrandbeestBushings.urdf"

end LoopClosureMode

def strandbeestExampleRoot : String :=
  "../drake/examples/multibody/strandbeest"

inductive StrandbeestExampleAssetKind where
  | metadata
  | source
  | test
  | xacro
  | generatedUrdfEntry
  deriving Repr, BEq, Inhabited

inductive StrandbeestExampleAssetFormat where
  | bazel
  | markdown
  | cpp
  | python
  | xacro
  deriving Repr, BEq, Inhabited

namespace StrandbeestExampleAssetFormat

def matchesPath (format : StrandbeestExampleAssetFormat) (path : String) : Bool :=
  match format with
  | .bazel => path == "BUILD.bazel"
  | .markdown => path.endsWith ".md"
  | .cpp => path.endsWith ".cc"
  | .python => path.endsWith ".py"
  | .xacro => path.endsWith ".xacro"

end StrandbeestExampleAssetFormat

/--
File manifest for Drake's `examples/multibody/strandbeest` tree.

The manifest records the xacro expansion graph that chooses between the
ball-constraint and LinearBushingRollPitchYaw URDF variants.  The expanded model
still lowers into the shared full-physics primitives below; this catalog is the
model-provider boundary, not a separate simulator.
-/
structure StrandbeestExampleAsset where
  relativePath : String
  format : StrandbeestExampleAssetFormat
  kind : StrandbeestExampleAssetKind
  component : String
  feedsModelExpansion : Bool := false
  loopClosureMode? : Option LoopClosureMode := none
  generatedUrdfUrl? : Option String := none
  localDependencies : Array String := #[]
  concept : String := ""
  deriving Repr, Inhabited

namespace StrandbeestExampleAsset

def fullPath (asset : StrandbeestExampleAsset) : String :=
  strandbeestExampleRoot ++ "/" ++ asset.relativePath

def validate? (asset : StrandbeestExampleAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "Strandbeest asset path cannot be empty"
  if !asset.format.matchesPath asset.relativePath then
    .error s!"Strandbeest asset {asset.relativePath}: format does not match path"
  if asset.component.isEmpty then
    .error s!"Strandbeest asset {asset.relativePath}: component cannot be empty"
  if asset.concept.isEmpty then
    .error s!"Strandbeest asset {asset.relativePath}: concept cannot be empty"
  for dep in asset.localDependencies do
    if dep.isEmpty then
      .error s!"Strandbeest asset {asset.relativePath}: local dependency cannot be empty"
    if dep == asset.relativePath then
      .error s!"Strandbeest asset {asset.relativePath}: cannot depend on itself"
  match asset.kind with
  | .generatedUrdfEntry =>
      if asset.loopClosureMode?.isNone then
        .error s!"Strandbeest generated URDF entry {asset.relativePath}: missing loop-closure mode"
      if asset.generatedUrdfUrl?.isNone then
        .error s!"Strandbeest generated URDF entry {asset.relativePath}: missing generated URDF URL"
      if !asset.feedsModelExpansion then
        .error s!"Strandbeest generated URDF entry {asset.relativePath}: should feed model expansion"
  | _ =>
      if asset.generatedUrdfUrl?.isSome then
        .error s!"Strandbeest asset {asset.relativePath}: only generated URDF entries should declare generated URLs"

end StrandbeestExampleAsset

def strandbeestExampleAssets : Array StrandbeestExampleAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      format := .bazel
      kind := .metadata
      component := "build"
      localDependencies := #[
        "run_with_motor.cc",
        "test/run_with_motor_test.py",
        "model/StrandbeestBushings.urdf.xacro",
        "model/StrandbeestConstraints.urdf.xacro",
        "model/LegAssembly.xacro",
        "model/LegPair.xacro",
        "model/Macros.xacro",
        "model/Strandbeest.xacro"
      ]
      concept := "Bazel xacro_filegroup plus run_with_motor and smoke-test targets"
    },
    {
      relativePath := "README.md"
      format := .markdown
      kind := .metadata
      component := "docs"
      concept := "overview of the Strandbeest demo and its constraint/bushing model variants"
    },
    {
      relativePath := "run_with_motor.cc"
      format := .cpp
      kind := .source
      component := "simulation"
      localDependencies := #[
        "model/StrandbeestConstraints.urdf.xacro",
        "model/StrandbeestBushings.urdf.xacro"
      ]
      concept := "full-plant simulation executable that selects a generated URDF model and advances Simulator"
    },
    {
      relativePath := "test/run_with_motor_test.py"
      format := .python
      kind := .test
      component := "smoke_test"
      localDependencies := #["run_with_motor.cc"]
      concept := "Python subprocess.check_call no-crash smoke test for run_with_motor"
    },
    {
      relativePath := "model/StrandbeestConstraints.urdf.xacro"
      format := .xacro
      kind := .generatedUrdfEntry
      component := "model"
      feedsModelExpansion := true
      loopClosureMode? := some LoopClosureMode.constraints
      generatedUrdfUrl? := some LoopClosureMode.constraints.urdfUrl
      localDependencies := #["model/Strandbeest.xacro"]
      concept := "entry xacro that expands the Strandbeest model with ball constraints"
    },
    {
      relativePath := "model/StrandbeestBushings.urdf.xacro"
      format := .xacro
      kind := .generatedUrdfEntry
      component := "model"
      feedsModelExpansion := true
      loopClosureMode? := some LoopClosureMode.bushings
      generatedUrdfUrl? := some LoopClosureMode.bushings.urdfUrl
      localDependencies := #["model/Strandbeest.xacro"]
      concept := "entry xacro that expands the Strandbeest model with compliant bushing loops"
    },
    {
      relativePath := "model/Strandbeest.xacro"
      format := .xacro
      kind := .xacro
      component := "model"
      feedsModelExpansion := true
      localDependencies := #["model/Macros.xacro", "model/LegPair.xacro"]
      concept := "root macro with floor, crossbar, crank, transmission, and six leg-pair instances"
    },
    {
      relativePath := "model/LegPair.xacro"
      format := .xacro
      kind := .xacro
      component := "model"
      feedsModelExpansion := true
      localDependencies := #["model/Macros.xacro", "model/LegAssembly.xacro"]
      concept := "leg-pair macro with mirrored legs and pair-level A-C loop closure"
    },
    {
      relativePath := "model/LegAssembly.xacro"
      format := .xacro
      kind := .xacro
      component := "model"
      feedsModelExpansion := true
      localDependencies := #["model/Macros.xacro"]
      concept := "single-leg macro with revolute links and F-G/B-C loop closures"
    },
    {
      relativePath := "model/Macros.xacro"
      format := .xacro
      kind := .xacro
      component := "model"
      feedsModelExpansion := true
      concept := "shared PVC mass and bushing stiffness/damping constants"
    }
  ]

private def hasDuplicateStrandbeestAssetPath : Bool := Id.run do
  let mut seen : Array String := #[]
  for asset in strandbeestExampleAssets do
    if seen.contains asset.relativePath then
      return true
    seen := seen.push asset.relativePath
  return false

def strandbeestExampleAssetPaths : Array String :=
  strandbeestExampleAssets.map (fun asset => asset.relativePath)

def strandbeestXacroAssets : Array StrandbeestExampleAsset :=
  strandbeestExampleAssets.filter (fun asset => asset.format == .xacro)

def strandbeestGeneratedUrdfEntryAssets : Array StrandbeestExampleAsset :=
  strandbeestExampleAssets.filter (fun asset => asset.kind == .generatedUrdfEntry)

def strandbeestModelExpansionAssets : Array StrandbeestExampleAsset :=
  strandbeestExampleAssets.filter (fun asset => asset.feedsModelExpansion)

def findStrandbeestExampleAsset? (relativePath : String) :
    Option StrandbeestExampleAsset :=
  strandbeestExampleAssets.find? (fun asset => asset.relativePath == relativePath)

def requiredStrandbeestExampleAssetPaths : Array String :=
  #[
    "BUILD.bazel",
    "README.md",
    "run_with_motor.cc",
    "test/run_with_motor_test.py",
    "model/LegAssembly.xacro",
    "model/LegPair.xacro",
    "model/Macros.xacro",
    "model/Strandbeest.xacro",
    "model/StrandbeestBushings.urdf.xacro",
    "model/StrandbeestConstraints.urdf.xacro"
  ]

def validateStrandbeestExampleAssetCatalog? : Except String Unit := do
  if strandbeestExampleAssets.size != 10 then
    .error s!"Strandbeest asset catalog should contain 10 files, got {strandbeestExampleAssets.size}"
  if hasDuplicateStrandbeestAssetPath then
    .error "Strandbeest asset catalog contains duplicate paths"
  for asset in strandbeestExampleAssets do
    asset.validate?
  for path in requiredStrandbeestExampleAssetPaths do
    if !(strandbeestExampleAssetPaths.contains path) then
      .error s!"Strandbeest asset catalog is missing {path}"
  for asset in strandbeestExampleAssets do
    for dep in asset.localDependencies do
      if !(strandbeestExampleAssetPaths.contains dep) then
        .error s!"Strandbeest asset {asset.relativePath}: missing local dependency {dep}"
  if strandbeestXacroAssets.size != 6 then
    .error s!"Strandbeest xacro asset count should be 6, got {strandbeestXacroAssets.size}"
  if strandbeestGeneratedUrdfEntryAssets.size != 2 then
    .error s!"Strandbeest generated URDF entry count should be 2, got {strandbeestGeneratedUrdfEntryAssets.size}"
  if strandbeestModelExpansionAssets.size != 6 then
    .error s!"Strandbeest model-expansion asset count should be 6, got {strandbeestModelExpansionAssets.size}"

structure Vec3 where
  x : Float := 0.0
  y : Float := 0.0
  z : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace Vec3

def toArray (v : Vec3) : Array Float :=
  #[v.x, v.y, v.z]

def isFinite (v : Vec3) : Bool :=
  v.x.isFinite && v.y.isFinite && v.z.isFinite

end Vec3

structure LegPairSpec where
  name : String
  offset : Float
  phase : Float
  deriving Repr, BEq, Inhabited

def legPairs : Array LegPairSpec :=
  #[
    { name := "pair01", offset := 0.6, phase := 0.0 },
    { name := "pair02", offset := 0.36, phase := 2.0 * pi / 3.0 },
    { name := "pair03", offset := 0.12, phase := 4.0 * pi / 3.0 },
    { name := "pair04", offset := -0.12, phase := pi / 3.0 },
    { name := "pair05", offset := -0.36, phase := pi },
    { name := "pair06", offset := -0.6, phase := 5.0 * pi / 3.0 }
  ]

def legNames : Array String := #["leg1", "leg2"]

structure BallConstraintSpec where
  id : Nat
  pairName : String
  legName : String := ""
  loopName : String
  bodyA : String
  p_AP : Vec3
  bodyB : String
  p_BQ : Vec3
  deriving Repr, Inhabited

structure BushingLoopSpec where
  id : Nat
  pairName : String
  legName : String := ""
  loopName : String
  frameA : String
  frameC : String
  bodyA : String
  p_AP : Vec3
  bodyC : String
  p_CQ : Vec3
  params : LinearBushingRollPitchYawParams
  deriving Repr, Inhabited

def strandbeestBushingParams : LinearBushingRollPitchYawParams :=
  {
    torqueStiffness := #[20000.0, 20000.0, 0.0]
    torqueDamping := #[2000.0, 2000.0, 0.0]
    forceStiffness := #[20000.0, 20000.0, 20000.0]
    forceDamping := #[2000.0, 2000.0, 2000.0]
    label := "strandbeest-linear-bushing-rpy"
  }

private def legPrefix (pairName legName : String) : String :=
  s!"{pairName}_{legName}"

private def legLoopConstraints (pairName legName : String) (baseId : Nat) :
    Array BallConstraintSpec :=
  let legId := legPrefix pairName legName
  #[
    {
      id := baseId
      pairName := pairName
      legName := legName
      loopName := "loop_f_g"
      bodyA := s!"{legId}_bar_f"
      p_AP := { z := 0.394 }
      bodyB := s!"{legId}_bar_g"
      p_BQ := { x := 0.3624, z := -0.0582 }
    },
    {
      id := baseId + 1
      pairName := pairName
      legName := legName
      loopName := "loop_b_c"
      bodyA := s!"{legId}_bar_b"
      p_AP := { x := -0.2978, z := 0.2892 }
      bodyB := s!"{legId}_bar_c"
      p_BQ := { z := 0.393 }
    }
  ]

private def pairLoopConstraints (pairName : String) (baseId : Nat) :
    Array BallConstraintSpec :=
  #[
    {
      id := baseId
      pairName := pairName
      legName := "leg1"
      loopName := "loop_a_c"
      bodyA := s!"{pairName}_bar_a"
      p_AP := { z := 0.38 }
      bodyB := s!"{pairName}_leg1_bar_c"
      p_BQ := { z := 0.393 }
    },
    {
      id := baseId + 1
      pairName := pairName
      legName := "leg2"
      loopName := "loop_a_c"
      bodyA := s!"{pairName}_bar_a"
      p_AP := { z := -0.38 }
      bodyB := s!"{pairName}_leg2_bar_c"
      p_BQ := { z := 0.393 }
    }
  ]

def ballConstraintsForPair (pairIndex : Nat) (pair : LegPairSpec) :
    Array BallConstraintSpec :=
  let base := 7000 + 6 * pairIndex
  legLoopConstraints pair.name "leg1" base ++
    legLoopConstraints pair.name "leg2" (base + 2) ++
    pairLoopConstraints pair.name (base + 4)

def ballConstraints : Array BallConstraintSpec := Id.run do
  let mut out : Array BallConstraintSpec := #[]
  for i in [:legPairs.size] do
    out := out ++ ballConstraintsForPair i legPairs[i]!
  return out

private def legLoopBushings (pairName legName : String) (baseId : Nat) :
    Array BushingLoopSpec :=
  let legId := legPrefix pairName legName
  #[
    {
      id := baseId
      pairName := pairName
      legName := legName
      loopName := "loop_f_g"
      frameA := s!"{legId}_loop_f_g_bushing_frameA"
      frameC := s!"{legId}_loop_f_g_bushing_frameC"
      bodyA := s!"{legId}_bar_f"
      p_AP := { z := 0.394 }
      bodyC := s!"{legId}_bar_g"
      p_CQ := { x := 0.3624, z := -0.0582 }
      params := strandbeestBushingParams
    },
    {
      id := baseId + 1
      pairName := pairName
      legName := legName
      loopName := "loop_b_c"
      frameA := s!"{legId}_loop_b_c_bushing_frameA"
      frameC := s!"{legId}_loop_b_c_bushing_frameC"
      bodyA := s!"{legId}_bar_b"
      p_AP := { x := -0.2978, z := 0.2892 }
      bodyC := s!"{legId}_bar_c"
      p_CQ := { z := 0.393 }
      params := strandbeestBushingParams
    }
  ]

private def pairLoopBushings (pairName : String) (baseId : Nat) :
    Array BushingLoopSpec :=
  #[
    {
      id := baseId
      pairName := pairName
      legName := "leg1"
      loopName := "loop_a_c"
      frameA := s!"{pairName}_leg1_loop_a_c_bushing_frameA"
      frameC := s!"{pairName}_leg1_loop_a_c_bushing_frameC"
      bodyA := s!"{pairName}_bar_a"
      p_AP := { z := 0.38 }
      bodyC := s!"{pairName}_leg1_bar_c"
      p_CQ := { z := 0.393 }
      params := strandbeestBushingParams
    },
    {
      id := baseId + 1
      pairName := pairName
      legName := "leg2"
      loopName := "loop_a_c"
      frameA := s!"{pairName}_leg2_loop_a_c_bushing_frameA"
      frameC := s!"{pairName}_leg2_loop_a_c_bushing_frameC"
      bodyA := s!"{pairName}_bar_a"
      p_AP := { z := -0.38 }
      bodyC := s!"{pairName}_leg2_bar_c"
      p_CQ := { z := 0.393 }
      params := strandbeestBushingParams
    }
  ]

def bushingLoopsForPair (pairIndex : Nat) (pair : LegPairSpec) :
    Array BushingLoopSpec :=
  let base := 8000 + 6 * pairIndex
  legLoopBushings pair.name "leg1" base ++
    legLoopBushings pair.name "leg2" (base + 2) ++
    pairLoopBushings pair.name (base + 4)

def bushingLoops : Array BushingLoopSpec := Id.run do
  let mut out : Array BushingLoopSpec := #[]
  for i in [:legPairs.size] do
    out := out ++ bushingLoopsForPair i legPairs[i]!
  return out

def perLegRevoluteJointCount : Nat := 5

def jointsPerLegPair : Nat :=
  2 * perLegRevoluteJointCount + 2

def legPairJointCount : Nat :=
  legPairs.size * jointsPerLegPair

def floatingBasePositionCount : Nat := 7

def floatingBaseVelocityCount : Nat := 6

def crankJointPositionIndex : Nat := floatingBasePositionCount

def crankJointVelocityStartInV : Nat := floatingBaseVelocityCount

def numPositions : Nat :=
  floatingBasePositionCount + 1 + legPairJointCount

def numVelocities : Nat :=
  floatingBaseVelocityCount + 1 + legPairJointCount

def numActuatedDofs : Nat := 1

def pvcKgPerM : Float := 0.476212462

def legTubeLengthSum : Float :=
  0.50 + 0.619 + 0.393 + 0.49 + 0.367 + 0.657 + 0.558 + 0.401 + 0.415 + 0.394

def pairSharedTubeLengthSum : Float :=
  0.078 + 0.76 + 0.15

def tubeMass (length : Float) : Float :=
  pvcKgPerM * length

def bodyMassEstimate : Float :=
  1.0 + tubeMass 1.2 + tubeMass 1.2 +
    (legPairs.size.toFloat * tubeMass (2.0 * legTubeLengthSum + pairSharedTubeLengthSum))

structure DesiredVelocityMotorParams where
  jointName : String := "joint_crossbar_crank"
  actuatorName : String := "crossbar_crank_motor"
  desiredOmega : Float := 5.0
  proportionalGain : Float := bodyMassEstimate
  velocityStartInV : Nat := crankJointVelocityStartInV
  numPositions : Nat := Strandbeest.numPositions
  deriving Repr, Inhabited

namespace DesiredVelocityMotorParams

def velocityIndex (motor : DesiredVelocityMotorParams) : Nat :=
  motor.numPositions + motor.velocityStartInV

def validate? (motor : DesiredVelocityMotorParams) : Except String Unit := do
  if motor.jointName.isEmpty then
    .error "desired velocity motor joint name is empty"
  if motor.actuatorName.isEmpty then
    .error "desired velocity motor actuator name is empty"
  if !motor.desiredOmega.isFinite then
    .error s!"desired velocity motor desired omega must be finite, got {motor.desiredOmega}"
  if !motor.proportionalGain.isFinite || motor.proportionalGain <= 0.0 then
    .error s!"desired velocity motor gain must be positive and finite, got {motor.proportionalGain}"

def torque (motor : DesiredVelocityMotorParams) (measuredOmega : Float) : Float :=
  motor.proportionalGain * (motor.desiredOmega - measuredOmega)

end DesiredVelocityMotorParams

structure CrossbarInitialPose where
  quaternionWxyz : Array Float := #[1.0, 0.0, 0.0, 0.0]
  translation : Vec3 := { x := -2.0, y := 0.0, z := 1.35 }
  deriving Repr, Inhabited

namespace CrossbarInitialPose

def validate? (pose : CrossbarInitialPose) : Except String Unit := do
  if pose.quaternionWxyz.size != 4 then
    .error s!"crossbar quaternion size {pose.quaternionWxyz.size} != 4"
  for i in [:pose.quaternionWxyz.size] do
    if !(pose.quaternionWxyz[i]!).isFinite then
      .error s!"crossbar quaternion[{i}] must be finite, got {pose.quaternionWxyz[i]!}"
  if !pose.translation.isFinite then
    .error s!"crossbar translation must be finite, got {reprStr pose.translation}"

end CrossbarInitialPose

structure StrandbeestParams where
  mode : LoopClosureMode := .constraints
  simulationTime : Float := 20.0
  initialVelocity : Float := 5.0
  mbtDt : Float := 5.0e-2
  penetrationAllowance : Float := 5.0e-3
  stictionTolerance : Float := 5.0e-2
  crossbarPose : CrossbarInitialPose := {}
  crankInitialPosition : Float := 0.0
  deriving Repr, Inhabited

def params : StrandbeestParams := {}

structure SimulatorGflagsBoundary where
  targetRealtimeRate : Float := 1.0
  accuracy : Float := 1.0e-2
  maxTimeStep : Float := 1.0e-1
  integrationScheme : String := "implicit_euler"
  printStatistics : Bool := true
  deriving Repr, Inhabited

namespace SimulatorGflagsBoundary

def validate? (flags : SimulatorGflagsBoundary) : Except String Unit := do
  if !flags.targetRealtimeRate.isFinite || flags.targetRealtimeRate < 0.0 then
    .error s!"Strandbeest simulator target realtime rate must be nonnegative and finite, got {flags.targetRealtimeRate}"
  if !flags.accuracy.isFinite || flags.accuracy <= 0.0 then
    .error s!"Strandbeest simulator accuracy must be positive and finite, got {flags.accuracy}"
  if !flags.maxTimeStep.isFinite || flags.maxTimeStep <= 0.0 then
    .error s!"Strandbeest simulator max time step must be positive and finite, got {flags.maxTimeStep}"
  if flags.integrationScheme != "implicit_euler" then
    .error s!"Strandbeest run_with_motor.cc sets implicit_euler, got {flags.integrationScheme}"

end SimulatorGflagsBoundary

def simulatorGflagsBoundary : SimulatorGflagsBoundary := {}

namespace StrandbeestParams

def selectedUrdfUrl (p : StrandbeestParams) : String :=
  p.mode.urdfUrl

def motor (p : StrandbeestParams) : DesiredVelocityMotorParams :=
  {
    desiredOmega := p.initialVelocity
    proportionalGain := bodyMassEstimate
  }

def plantConfig (p : StrandbeestParams) : MultibodyPlantConfigPrimitive :=
  {
    timeStep := p.mbtDt
    penetrationAllowance := p.penetrationAllowance
    stictionTolerance := p.stictionTolerance
    contactApproximation := .sap
  }

def validate? (p : StrandbeestParams) : Except String Unit := do
  if !p.simulationTime.isFinite || p.simulationTime <= 0.0 then
    .error s!"Strandbeest simulation_time must be positive and finite, got {p.simulationTime}"
  if !p.initialVelocity.isFinite then
    .error s!"Strandbeest initial_velocity must be finite, got {p.initialVelocity}"
  if !p.crankInitialPosition.isFinite then
    .error s!"Strandbeest crank initial position must be finite, got {p.crankInitialPosition}"
  p.crossbarPose.validate?
  (p.motor).validate?
  (p.plantConfig).validate?
  match p.mode with
  | .constraints =>
      if p.mbtDt <= 0.0 then
        .error "Strandbeest constraints mode follows Drake's SAP discrete plant path and requires mbt_dt > 0"
  | .bushings =>
      if p.mbtDt != 0.0 then
        .error "Strandbeest bushing mode follows Drake's continuous plant path and requires mbt_dt == 0"

end StrandbeestParams

def strandbeestModel (p : StrandbeestParams) : FullMultibodyPlantModel :=
  {
    modelName := "Strandbeest"
    modelUri := p.selectedUrdfUrl
    numPositions := numPositions
    numVelocities := numVelocities
    numActuatedDofs := numActuatedDofs
    floatingBases := #[
      {
        bodyName := "crossbar"
        convention := .quaternion
        floatingPositionsStart := 0
        floatingVelocitiesStartInV := 0
      }
    ]
    finalized := true
    label := s!"Strandbeest {p.mode.label} URDF"
  }

def ground : HalfSpaceContactEnvironment :=
  {
    visualName := "floor"
    collisionName := "floor"
    friction := { staticFriction := 1.0, dynamicFriction := 0.6 }
  }

def initialPositions (p : StrandbeestParams) : Array Float := Id.run do
  let mut q := Array.replicate numPositions 0.0
  q := q.set! 0 (p.crossbarPose.quaternionWxyz.getD 0 1.0)
  q := q.set! 1 (p.crossbarPose.quaternionWxyz.getD 1 0.0)
  q := q.set! 2 (p.crossbarPose.quaternionWxyz.getD 2 0.0)
  q := q.set! 3 (p.crossbarPose.quaternionWxyz.getD 3 0.0)
  q := q.set! 4 p.crossbarPose.translation.x
  q := q.set! 5 p.crossbarPose.translation.y
  q := q.set! 6 p.crossbarPose.translation.z
  q := q.set! crankJointPositionIndex p.crankInitialPosition
  return q

def initialVelocities : Array Float :=
  Array.replicate numVelocities 0.0

def actuationFromMotor (p : StrandbeestParams) (measuredOmega : Float) :
    Array Float :=
  #[(p.motor).torque measuredOmega]

def plantStep (p : StrandbeestParams) (measuredOmega : Float := 0.0) :
    FullMultibodyPlantStep :=
  {
    model := strandbeestModel p
    config := p.plantConfig
    q0 := initialPositions p
    v0 := initialVelocities
    actuation := actuationFromMotor p measuredOmega
    t0 := 0.0
    t1 := p.simulationTime
    ground? := some ground
    label := s!"strandbeest-{p.mode.label}-full-multibody-plant"
  }

private def unitVector (n index : Nat) : Array Float := Id.run do
  let mut out := Array.replicate n 0.0
  if index < n then
    out := out.set! index 1.0
  return out

def strandbeestMassDiagonal : Array Float := Id.run do
  let mut diag := Array.replicate numVelocities 1.0
  for i in [:3] do
    diag := diag.set! i bodyMassEstimate
  for i in [3:6] do
    diag := diag.set! i bodyMassEstimate
  return diag

def strandbeestMassMatrix : Array (Array Float) :=
  FloatMatrix.diagonal strandbeestMassDiagonal

def strandbeestGeneralizedActuation (step : FullMultibodyPlantStep) :
    Array Float :=
  let torque := step.actuation.getD 0 0.0
  (unitVector numVelocities crankJointVelocityStartInV).map
    (fun x => torque * x)

def strandbeestGravityBias : Array Float :=
  (unitVector numVelocities 5).map
    (fun x => bodyMassEstimate * 9.81 * x)

private def closureVelocityIndex (id offset : Nat) : Nat :=
  (crankJointVelocityStartInV + 1 + id + offset) % numVelocities

def ballConstraintContactCandidate (constraint : BallConstraintSpec) :
    ContactCandidate :=
  {
    id := constraint.id
    bodyA := constraint.bodyA
    bodyB := constraint.bodyB
    point_W := constraint.p_AP.toArray
    normal_W := #[1.0, 0.0, 0.0]
    signedDistance := 0.0
    normalVelocity := 0.0
    tangentVelocity := 0.0
    tangentVelocity2 := 0.0
    normalJacobian := unitVector numVelocities (closureVelocityIndex constraint.id 0)
    tangentJacobian := unitVector numVelocities (closureVelocityIndex constraint.id 1)
    tangentJacobian2 := unitVector numVelocities (closureVelocityIndex constraint.id 2)
    mode := .sticking
    label := s!"strandbeest-ball-constraint:{constraint.pairName}:{constraint.legName}:{constraint.loopName}"
  }

def ballConstraintContactCandidates : Array ContactCandidate :=
  ballConstraints.map ballConstraintContactCandidate

def ballConstraintPrimitive (constraint : BallConstraintSpec) :
    BilateralConstraintPrimitive :=
  let candidate := ballConstraintContactCandidate constraint
  {
    id := constraint.id
    jacobian := candidate.constraintJacobianRows true
    targetAcceleration := #[0.0, 0.0, 0.0]
    label := s!"strandbeest-ball-constraint:{constraint.pairName}:{constraint.legName}:{constraint.loopName}"
  }

def ballConstraintPrimitives : Array BilateralConstraintPrimitive :=
  ballConstraints.map ballConstraintPrimitive

def zeroContactForcesForCandidates
    (candidates : Array ContactCandidate) : Array ContactForceScalars :=
  candidates.map (fun candidate =>
    ContactForceScalars.fromCandidate3D candidate 0.0 0.0 0.0)

def bushingState (loop : BushingLoopSpec) : LinearBushingRollPitchYawState :=
  {
    rpyError := #[0.0, 0.0, 0.0]
    angularVelocityError := #[0.0, 0.0, 0.0]
    translationError := #[0.0, 0.0, 0.0]
    translationVelocityError := #[0.0, 0.0, 0.0]
    rpyJacobian := #[
      unitVector numVelocities (closureVelocityIndex loop.id 0),
      unitVector numVelocities (closureVelocityIndex loop.id 1),
      unitVector numVelocities (closureVelocityIndex loop.id 2)
    ]
    translationJacobian := #[
      unitVector numVelocities (closureVelocityIndex loop.id 3),
      unitVector numVelocities (closureVelocityIndex loop.id 4),
      unitVector numVelocities (closureVelocityIndex loop.id 5)
    ]
    label := s!"strandbeest-bushing:{loop.pairName}:{loop.legName}:{loop.loopName}"
  }

def bushingResults? :
    Except String (Array LinearBushingRollPitchYawResult) := do
  let mut out : Array LinearBushingRollPitchYawResult := #[]
  for loop in bushingLoops do
    out := out.push (← LinearBushingRollPitchYaw.evaluate?
      numVelocities loop.params (bushingState loop))
  pure out

def sumBushingGeneralizedForces
    (results : Array LinearBushingRollPitchYawResult) : Array Float :=
  results.foldl
    (fun acc result => FloatArray.add acc result.generalizedForce)
    (Array.replicate numVelocities 0.0)

structure StrandbeestPrimitivePhysics where
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  bushingResults : Array LinearBushingRollPitchYawResult := #[]
  deriving Repr, Inhabited

structure StrandbeestPhysicsState where
  params : StrandbeestParams := Strandbeest.params
  measuredOmega : Float := 0.0
  deriving Repr, Inhabited

namespace StrandbeestPhysicsState

def validate? (snapshot : StrandbeestPhysicsState) : Except String Unit := do
  snapshot.params.validate?
  if !snapshot.measuredOmega.isFinite then
    .error s!"Strandbeest measured motor velocity must be finite, got {snapshot.measuredOmega}"

def step (snapshot : StrandbeestPhysicsState) : FullMultibodyPlantStep :=
  plantStep snapshot.params snapshot.measuredOmega

end StrandbeestPhysicsState

def physicsState
    (p : StrandbeestParams := params)
    (measuredOmega : Float := 0.0) : StrandbeestPhysicsState :=
  {
    params := p
    measuredOmega := measuredOmega
  }

def strandbeestFullPhysicsPrimitives?
    (p : StrandbeestParams)
    (step : FullMultibodyPlantStep)
    (label : String := s!"Strandbeest {p.mode.label} primitive plant dynamics") :
    Except String (FullPhysicsPrimitives × Array LinearBushingRollPitchYawResult) := do
  p.validate?
  step.validate?
  let baseActuation := strandbeestGeneralizedActuation step
  match p.mode with
  | .constraints =>
      pure ({
        massMatrix := strandbeestMassMatrix
        qdot := step.v0
        actuationForces := baseActuation
        biasForces := strandbeestGravityBias
        contactCandidates := #[]
        supportPolicy := .fullSupport
        contactForceSource := .precomputed
        contactForces := #[]
        bilateralConstraints := ballConstraintPrimitives
        distanceTol := p.penetrationAllowance
        tangentVelocityTol := p.stictionTolerance
        label := label
      }, #[])
  | .bushings =>
      let results ← bushingResults?
      pure ({
        massMatrix := strandbeestMassMatrix
        qdot := step.v0
        actuationForces := baseActuation
        generalizedForceContributions := #[
          GeneralizedForceContribution.ofForce
            (sumBushingGeneralizedForces results)
            "Strandbeest LinearBushingRollPitchYaw loop closure generalized force"
            "LinearBushingRollPitchYaw"
        ]
        biasForces := strandbeestGravityBias
        contactCandidates := #[]
        supportPolicy := .fullSupport
        contactForceSource := .precomputed
        contactForces := #[]
        distanceTol := p.penetrationAllowance
        tangentVelocityTol := p.stictionTolerance
        label := label
      }, results)

namespace StrandbeestPhysicsState

def primitivesWithLoopClosures?
    (snapshot : StrandbeestPhysicsState)
    (label : String := s!"Strandbeest {snapshot.params.mode.label} primitive plant dynamics") :
    Except String (FullMultibodyPlantStep × FullPhysicsPrimitives ×
      Array LinearBushingRollPitchYawResult) := do
  snapshot.validate?
  let step := snapshot.step
  step.validate?
  let (primitives, bushingResults) ←
    strandbeestFullPhysicsPrimitives? snapshot.params step label
  pure (step, primitives, bushingResults)

end StrandbeestPhysicsState

def fullPhysicsPrimitiveProvider
    (label : String := "Strandbeest primitive full physics provider") :
    FullPhysicsPrimitiveProvider StrandbeestPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      let (_, primitives, _) ← snapshot.primitivesWithLoopClosures? label
      pure primitives
  }

def solveStrandbeestPrimitivePhysics?
    (p : StrandbeestParams)
    (step : FullMultibodyPlantStep)
    (intervalVertex : VertexId := 8905) :
    Except String StrandbeestPrimitivePhysics := do
  let (primitives, bushingResults) ← strandbeestFullPhysicsPrimitives? p step
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := step
    primitives := primitives
    intervalVertex := intervalVertex
    label := primitives.label
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    primitivePlant := primitivePlant
    fullPhysics := fullPhysics
    bushingResults := bushingResults
  }

def solveStrandbeestPrimitivePhysicsAt?
    (snapshot : StrandbeestPhysicsState)
    (intervalVertex : VertexId := 8905)
    (label : String := s!"Strandbeest {snapshot.params.mode.label} primitive plant dynamics") :
    Except String StrandbeestPrimitivePhysics := do
  let (step, primitives, bushingResults) ←
    snapshot.primitivesWithLoopClosures? label
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := step
    primitives := primitives
    intervalVertex := intervalVertex
    label := primitives.label
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    primitivePlant := primitivePlant
    fullPhysics := fullPhysics
    bushingResults := bushingResults
  }

def acceptedSegment (p : StrandbeestParams) : AcceptedStepSegment :=
  {
    id := 8905
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.simulationTime
    tAfter := p.simulationTime
    label := s!"Strandbeest Simulator.AdvanceTo {p.mode.label}"
  }

private def localMove (vertex : VertexId) (label : String)
    (exactness : MoveExactness := .exact) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := exactness
    label := label
  }

def loopClosureSupport (p : StrandbeestParams) : RuntimeSupport :=
  match p.mode with
  | .constraints =>
      { RuntimeSupport.full ballConstraints.size with label := "all Strandbeest ball constraints" }
  | .bushings =>
      { RuntimeSupport.full bushingLoops.size with label := "all Strandbeest bushing loops" }

def strandbeestGraph (p : StrandbeestParams) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8900, kind := .state .boundary, label := p.selectedUrdfUrl }
    |>.addVertex { id := 8901, kind := .state .interior, label := "Strandbeest Xacro-expanded loop closures" }
    |>.addVertex { id := 8902, kind := .state .interior, label := "MultibodyPlantConfig SAP/contact parameters" }
    |>.addVertex { id := 8903, kind := .state .checkpoint, label := "IK-initialized Strandbeest context" }
    |>.addVertex { id := 8904, kind := .frozen, label := "DesiredVelocityMotor torque input" }
    |>.addVertex { id := 8905, kind := .interval, label := "Simulator.AdvanceTo Strandbeest full plant" }
    |>.addVertex { id := 8906, kind := .state .checkpoint, label := "Strandbeest final context checkpoint" }
    |>.addMove (localMove 8900 "Parser.AddModelsFromUrl + MultibodyPlant.Finalize")
    |>.addMove (localMove 8901 s!"expand {p.mode.label} loop-closure primitives from Xacro")
    |>.addMove (localMove 8903 "InverseKinematics: crossbar pose, crank top-dead-center, and loop closures")
    |>.addMove (localMove 8904 "DesiredVelocityMotor total-mass gain torque law")
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[8905]
      reads := #[8900, 8901, 8902, 8903, 8904]
      writes := #[8906]
      exactness := .exact
      label := "Simulator.AdvanceTo full Strandbeest MultibodyPlant interval"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[8906]
      reads := #[8905]
      writes := #[8906]
      label := "store Strandbeest final context checkpoint"
    }

structure StrandbeestResult where
  references : Array DrakeReference
  assetCatalog : Array StrandbeestExampleAsset
  params : StrandbeestParams
  model : FullMultibodyPlantModel
  config : MultibodyPlantConfigPrimitive
  motor : DesiredVelocityMotorParams
  support : RuntimeSupport
  ballConstraints : Array BallConstraintSpec
  bushingLoops : Array BushingLoopSpec
  step : FullMultibodyPlantStep
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  bushingResults : Array LinearBushingRollPitchYawResult := #[]
  trace : DynamicEventTrace
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd? (p : StrandbeestParams := params)
    (measuredOmega : Float := 0.0) : Except String StrandbeestResult := do
  validateStrandbeestExampleAssetCatalog?
  p.validate?
  strandbeestBushingParams.validate?
  let snapshot := physicsState p measuredOmega
  snapshot.validate?
  let primitivePhysics ← solveStrandbeestPrimitivePhysicsAt? snapshot
  let step := primitivePhysics.primitivePlant.step
  let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment p))
  trace.validate?
  let graph := strandbeestGraph p
  let closureMove :=
    match p.mode with
    | .constraints =>
        localMove 8907 "SAP ball-constraint full-physics loop closure boundary"
    | .bushings =>
        localMove 8907 "LinearBushingRollPitchYaw full-physics loop closure boundary"
  pure {
    references := drakeReferences
    assetCatalog := strandbeestExampleAssets
    params := p
    model := strandbeestModel p
    config := p.plantConfig
    motor := p.motor
    support := loopClosureSupport p
    ballConstraints := ballConstraints
    bushingLoops := bushingLoops
    step := step
    primitivePlant := primitivePhysics.primitivePlant
    fullPhysics := primitivePhysics.fullPhysics
    bushingResults := primitivePhysics.bushingResults
    trace := trace
    graph := graph
    moves :=
      #[
        localMove 8900 "Strandbeest Xacro/URDF provider",
        localMove 8901 "Strandbeest IK initial context provider",
        localMove 8902 "DesiredVelocityMotor controller",
        closureMove,
        primitivePhysics.fullPhysics.supportMove,
        primitivePhysics.fullPhysics.move
      ] ++ trace.moves
  }

structure RunWithMotorSmokeTestBoundary where
  smokeTestPath : String :=
    "../drake/examples/multibody/strandbeest/test/run_with_motor_test.py"
  executablePath : String := "examples/multibody/strandbeest/run_with_motor"
  simulationTimeOverride : Float := 0.1
  commandLine : Array String :=
    #["examples/multibody/strandbeest/run_with_motor", "--simulation_time=0.1"]
  simulatorFlags : SimulatorGflagsBoundary := simulatorGflagsBoundary
  params : StrandbeestParams := { Strandbeest.params with simulationTime := 0.1 }
  expectsNoCrash : Bool := true
  deriving Repr, Inhabited

namespace RunWithMotorSmokeTestBoundary

def validate? (boundary : RunWithMotorSmokeTestBoundary) : Except String Unit := do
  if boundary.smokeTestPath != "../drake/examples/multibody/strandbeest/test/run_with_motor_test.py" then
    .error s!"Strandbeest smoke test path mismatch: {boundary.smokeTestPath}"
  if boundary.executablePath != "examples/multibody/strandbeest/run_with_motor" then
    .error s!"Strandbeest smoke test executable mismatch: {boundary.executablePath}"
  if !boundary.simulationTimeOverride.isFinite || boundary.simulationTimeOverride <= 0.0 then
    .error s!"Strandbeest smoke test simulation_time override must be positive and finite, got {boundary.simulationTimeOverride}"
  if boundary.commandLine.size != 2 then
    .error s!"Strandbeest smoke test should pass executable plus one flag, got {reprStr boundary.commandLine}"
  if boundary.commandLine.getD 0 "" != boundary.executablePath then
    .error "Strandbeest smoke test command should start with run_with_motor executable"
  if boundary.commandLine.getD 1 "" != "--simulation_time=0.1" then
    .error s!"Strandbeest smoke test should pass --simulation_time=0.1, got {boundary.commandLine.getD 1 ""}"
  if boundary.params.simulationTime != boundary.simulationTimeOverride then
    .error s!"Strandbeest smoke boundary params should use simulation_time={boundary.simulationTimeOverride}, got {boundary.params.simulationTime}"
  if !boundary.expectsNoCrash then
    .error "Strandbeest run_with_motor_test.py expects subprocess.check_call to return"
  boundary.simulatorFlags.validate?
  boundary.params.validate?

def graph (boundary : RunWithMotorSmokeTestBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8920, kind := .state .boundary, label := boundary.smokeTestPath }
    |>.addVertex { id := 8921, kind := .state .boundary, label := boundary.executablePath }
    |>.addVertex { id := 8922, kind := .state .boundary, label := s!"--simulation_time={boundary.simulationTimeOverride}" }
    |>.addVertex { id := 8923, kind := .state .interior, label := "run_with_motor simulator gflags" }
    |>.addVertex { id := 8924, kind := .interval, label := "subprocess.check_call no-crash smoke run" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8923]
      reads := #[8921]
      writes := #[8923]
      label := s!"MakeSimulatorFromGflags target={boundary.simulatorFlags.targetRealtimeRate}, accuracy={boundary.simulatorFlags.accuracy}, max_step={boundary.simulatorFlags.maxTimeStep}, scheme={boundary.simulatorFlags.integrationScheme}"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[8924]
      reads := #[8920, 8921, 8922, 8923]
      writes := #[8924]
      cost := { work := boundary.simulationTimeOverride }
      label := "Python unittest subprocess.check_call run_with_motor smoke boundary"
    }

end RunWithMotorSmokeTestBoundary

def runWithMotorSmokeTestBoundary : RunWithMotorSmokeTestBoundary := {}

structure RunWithMotorSmokeTestResult where
  boundary : RunWithMotorSmokeTestBoundary
  result : StrandbeestResult
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildRunWithMotorSmokeTest?
    (boundary : RunWithMotorSmokeTestBoundary := runWithMotorSmokeTestBoundary)
    (measuredOmega : Float := 0.0) :
    Except String RunWithMotorSmokeTestResult := do
  boundary.validate?
  let result ← buildEndToEnd? boundary.params measuredOmega
  let graph := boundary.graph
  pure {
    boundary := boundary
    result := result
    graph := graph
    moves := graph.moves ++ result.moves
  }

end Tyr.EventSkeleton.Examples.Strandbeest
