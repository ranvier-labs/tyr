import Tyr.EventSkeleton.Manipulator

/-!
# Drake Allegro Hand Examples

This ports the controller and plant-facing boundary of
`../drake/examples/allegro_hand`.

The Drake examples contain two related systems:

* a constant-load hand simulation, and
* a joint-position LCM demo that grasps and twists a mug.

This file keeps the physics-facing data explicit: hand model URLs, preferred
joint ordering, reflected actuator inertia, PID gains, command/status message
semantics, finger-stuck detection, and the twisting target sequence.  A future
SDF/URDF-backed multibody provider can replace the fixture plant data while
preserving these controller and event-skeleton boundaries.
-/

namespace Tyr.EventSkeleton.Examples.AllegroHand

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/allegro_hand/allegro_common.cc"
      concept := "defines preferred joint order, oscillator PID gains, grasp/open targets, and stuck-finger detection"
    },
    {
      path := "../drake/examples/allegro_hand/allegro_common.h"
      concept := "declares the 16-joint Allegro hand helpers and motion-state classifier"
    },
    {
      path := "../drake/examples/allegro_hand/allegro_lcm.cc"
      concept := "implements command receiver latching and status sender message construction"
    },
    {
      path := "../drake/examples/allegro_hand/allegro_lcm.h"
      concept := "declares LCM period, command receiver ports, and status sender ports"
    },
    {
      path := "../drake/examples/allegro_hand/joint_control/allegro_single_object_simulation.cc"
      concept := "builds a hand plus mug plant, applies reflected inertia, contact material, PID control, and LCM wiring"
    },
    {
      path := "../drake/examples/allegro_hand/joint_control/run_twisting_mug.cc"
      concept := "generates grasp and alternating twist joint-position commands from status feedback"
    },
    {
      path := "../drake/examples/allegro_hand/joint_control/test/run_twisting_mug_test.py"
      concept := "runs the single-object hand/mug simulator and one-cycle twisting controller under an isolated LCM URL"
    },
    {
      path := "../drake/examples/allegro_hand/run_allegro_constant_load_demo.cc"
      concept := "runs a welded Allegro hand under constant joint torque"
    },
    {
      path := "../drake/examples/allegro_hand/test/allegro_lcm_test.cc"
      concept := "checks command receiver initial state, command latching, zero velocities, and status message fields"
    },
    {
      path := "../drake/examples/allegro_hand/test/parse_test.cc"
      concept := "checks SDF and URDF parser outputs: model instances, actuators, joints, bodies, and hand state dimensions"
    },
    {
      path := "../drake/examples/allegro_hand/joint_control/simple_mug.sdf"
      concept := "object model used by the twisting-mug demo"
    }
  ]

def numJoints : Nat := 16
def fingerCount : Nat := 4
def jointsPerFinger : Nat := 4

def hardwareStatusPeriod : Float := 0.003
def statusChannel : String := "ALLEGRO_STATUS"
def commandChannel : String := "ALLEGRO_COMMAND"

def rightHandModelUri : String :=
  "package://drake_models/allegro_hand_description/sdf/allegro_hand_description_right.sdf"

def leftHandModelUri : String :=
  "package://drake_models/allegro_hand_description/sdf/allegro_hand_description_left.sdf"

def rightHandUrdfModelUri : String :=
  "package://drake_models/allegro_hand_description/urdf/allegro_hand_description_right.urdf"

def leftHandUrdfModelUri : String :=
  "package://drake_models/allegro_hand_description/urdf/allegro_hand_description_left.urdf"

def mugModelUri : String :=
  "package://drake/examples/allegro_hand/joint_control/simple_mug.sdf"

def preferredJointOrdering : Array String :=
  #["joint_12", "joint_13", "joint_14", "joint_15",
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
    "joint_8", "joint_9", "joint_10", "joint_11"]

def fingerTipLinks : Array String :=
  #["link_3", "link_7", "link_11", "link_15"]

def zeroJoints : Array Float :=
  Array.replicate numJoints 0.0

inductive AllegroModelFormat where
  | sdf
  | urdf
  deriving Repr, BEq, Inhabited

namespace AllegroModelFormat

def extension : AllegroModelFormat → String
  | .sdf => "sdf"
  | .urdf => "urdf"

def rightUri : AllegroModelFormat → String
  | .sdf => rightHandModelUri
  | .urdf => rightHandUrdfModelUri

def leftUri : AllegroModelFormat → String
  | .sdf => leftHandModelUri
  | .urdf => leftHandUrdfModelUri

def expectedNumJoints : AllegroModelFormat → Nat
  | .sdf => 2 * numJoints + 2
  | .urdf => 2 * 21 + 2

def expectedNumBodies : AllegroModelFormat → Nat
  | .sdf => 2 * 17 + 1
  | .urdf => 2 * 22 + 1

end AllegroModelFormat

def allegroParseQuantities
    (format : AllegroModelFormat) : ParsedMultibodyPlantQuantities :=
  {
    modelUris := #[format.rightUri, format.leftUri]
    builtInModelInstances := 2
    numModelInstances := 4
    numActuators := 2 * numJoints
    numJoints := format.expectedNumJoints
    numBodies := format.expectedNumBodies
    modelInstances := #[
      {
        name := "right_hand"
        modelUri := format.rightUri
        numPositions := 23
        numVelocities := 22
      },
      {
        name := "left_hand"
        modelUri := format.leftUri
        numPositions := 23
        numVelocities := 22
      }
    ]
    finalized := true
    label := s!"allegro ParseTest {format.extension}"
  }

structure AllegroParseTestBoundary where
  testPath : String := "../drake/examples/allegro_hand/test/parse_test.cc"
  formats : Array AllegroModelFormat := #[.sdf, .urdf]
  quantities : Array ParsedMultibodyPlantQuantities :=
    #[allegroParseQuantities .sdf, allegroParseQuantities .urdf]
  deriving Repr, Inhabited

namespace AllegroParseTestBoundary

def validate? (boundary : AllegroParseTestBoundary) :
    Except String Unit := do
  if boundary.testPath != "../drake/examples/allegro_hand/test/parse_test.cc" then
    .error s!"Allegro parse test path mismatch: {boundary.testPath}"
  if boundary.formats != #[.sdf, .urdf] then
    .error s!"Allegro parse test should instantiate SDF and URDF, got {reprStr boundary.formats}"
  if boundary.quantities.size != boundary.formats.size then
    .error s!"Allegro parse test quantities size {boundary.quantities.size} != formats size {boundary.formats.size}"
  for i in [:boundary.formats.size] do
    let format := boundary.formats[i]!
    let quantities := boundary.quantities[i]!
    quantities.validate?
    if quantities.modelUris != #[format.rightUri, format.leftUri] then
      .error s!"Allegro {format.extension} parser URIs mismatch: {quantities.modelUris}"
    if quantities.numModelInstances != 4 then
      .error s!"Allegro {format.extension} parser should produce 4 model instances, got {quantities.numModelInstances}"
    if quantities.numActuators != 2 * numJoints then
      .error s!"Allegro {format.extension} parser should produce {2 * numJoints} actuators, got {quantities.numActuators}"
    if quantities.numJoints != format.expectedNumJoints then
      .error s!"Allegro {format.extension} parser joints mismatch: got {quantities.numJoints}, expected {format.expectedNumJoints}"
    if quantities.numBodies != format.expectedNumBodies then
      .error s!"Allegro {format.extension} parser bodies mismatch: got {quantities.numBodies}, expected {format.expectedNumBodies}"
    for inst in quantities.modelInstances do
      if inst.numPositions != 23 then
        .error s!"Allegro {format.extension} {inst.name}: expected 23 positions, got {inst.numPositions}"
      if inst.numVelocities != 22 then
        .error s!"Allegro {format.extension} {inst.name}: expected 22 velocities, got {inst.numVelocities}"

def graph (boundary : AllegroParseTestBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8560, kind := .state .boundary, label := boundary.testPath }
    |>.addVertex { id := 8561, kind := .state .interior, label := "allegro SDF parser quantities" }
    |>.addVertex { id := 8562, kind := .state .interior, label := "allegro URDF parser quantities" }
    |>.addMove
      (ParsedMultibodyPlantQuantities.parserMove 8561
        "Parser.AddModelsFromUrl right+left Allegro SDF; plant.Finalize; check quantities")
    |>.addMove
      (ParsedMultibodyPlantQuantities.parserMove 8562
        "Parser.AddModelsFromUrl right+left Allegro URDF; plant.Finalize; check quantities")

end AllegroParseTestBoundary

def allegroParseTestBoundary : AllegroParseTestBoundary := {}

structure ConstantLoadParams where
  constantLoad : Float := 0.0
  simulationTime : Float := 5.0
  maxTimeStep : Float := 1.0e-4
  addGravity : Bool := true
  targetRealtimeRate : Float := 1.0
  useRightHand : Bool := true
  deriving Repr, Inhabited

def constantLoadParams : ConstantLoadParams := {}

namespace ConstantLoadParams

def handModelUri (p : ConstantLoadParams) : String :=
  if p.useRightHand then rightHandModelUri else leftHandModelUri

def validate? (p : ConstantLoadParams) : Except String Unit := do
  if !p.constantLoad.isFinite then
    .error s!"Allegro constant-load params: constant load must be finite, got {p.constantLoad}"
  if !p.simulationTime.isFinite || p.simulationTime <= 0.0 then
    .error s!"Allegro constant-load params: simulation time must be positive and finite, got {p.simulationTime}"
  if !p.maxTimeStep.isFinite || p.maxTimeStep < 0.0 then
    .error s!"Allegro constant-load params: max time step must be nonnegative and finite, got {p.maxTimeStep}"
  if !p.targetRealtimeRate.isFinite || p.targetRealtimeRate < 0.0 then
    .error s!"Allegro constant-load params: target realtime rate must be nonnegative and finite, got {p.targetRealtimeRate}"

end ConstantLoadParams

structure JointControlPlantParams where
  useRightHand : Bool := true
  addGravity : Bool := false
  mbpDiscreteUpdatePeriod : Float := 1.0e-2
  gearRatio : Float := 369.0
  rotorInertia : Float := 1.0e-6
  pidFrequency : Float := 10.0
  pointContactStiffness : Float := 1.0e4
  huntCrossleyDissipation : Float := 1.0
  fingerMassEstimate : Float := 0.17 / 3.0
  fingerLengthEstimate : Float := 0.05
  dampingRatio : Float := 0.4
  deriving Repr, Inhabited

def jointControlParams : JointControlPlantParams := {}

def mugFloatingPositions : Nat := 7
def mugFloatingVelocities : Nat := 6

def allegroJointControlPlantModel : FullMultibodyPlantModel :=
  {
    modelName := "allegro_hand_with_simple_mug"
    modelUri := rightHandModelUri ++ " + " ++ mugModelUri
    numPositions := numJoints + mugFloatingPositions
    numVelocities := numJoints + mugFloatingVelocities
    numActuatedDofs := numJoints
    floatingBases := #[
      {
        bodyName := "simple_mug"
        convention := .quaternion
        floatingPositionsStart := numJoints
        floatingVelocitiesStartInV := numJoints
      }
    ]
    finalized := true
    label := "allegro_single_object_simulation MultibodyPlant"
  }

def allegroJointControlActuationMap : GeneralizedActuationMap :=
  GeneralizedActuationMap.contiguousOffset
    (numJoints + mugFloatingVelocities) numJoints 0
    "Allegro hand actuators before floating mug velocities"

def allegroJointControlPlantConfig
    (p : JointControlPlantParams := jointControlParams) :
    MultibodyPlantConfigPrimitive :=
  {
    timeStep := p.mbpDiscreteUpdatePeriod
    penetrationAllowance := 1.0e-3
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def initialMugPoseQ : Array Float :=
  -- Drake initializes the mug with RollPitchYaw(pi/2, 0, 0) and translation
  -- p_WHand + (0.095, 0.062, 0.095).  The floating-base boundary records the
  -- quaternion-plus-translation generalized positions.
  #[0.7071067811865476, 0.7071067811865476, 0.0, 0.0,
    0.095, 0.062, 0.095]

def allegroJointControlInitialQ : Array Float :=
  zeroJoints ++ initialMugPoseQ

def allegroJointControlInitialV : Array Float :=
  Array.replicate (numJoints + mugFloatingVelocities) 0.0

def allegroJointControlPlantStep
    (p : JointControlPlantParams := jointControlParams)
    (simulationTime : Float := 30.0)
    (actuation : Array Float := zeroJoints) :
    FullMultibodyPlantStep :=
  {
    model := allegroJointControlPlantModel
    config := allegroJointControlPlantConfig p
    q0 := allegroJointControlInitialQ
    v0 := allegroJointControlInitialV
    actuation := actuation
    t0 := 0.0
    t1 := simulationTime
    ground? := none
    label := "allegro_single_object_simulation full plant advance"
  }

def mugMass : Float := 0.094

def mugSpatialInertiaDiagonal : Array Float :=
  -- Drake's simple_mug.sdf gives the rotational inertia first; quaternion
  -- floating-base velocities then append translational velocity components.
  #[0.000156, 0.000156, 0.00015, mugMass, mugMass, mugMass]

namespace JointControlPlantParams

def handModelUri (p : JointControlPlantParams) : String :=
  if p.useRightHand then rightHandModelUri else leftHandModelUri

def actuatorModel (p : JointControlPlantParams) : JointActuatorModel :=
  {
    gearRatio := p.gearRatio
    rotorInertia := p.rotorInertia
    label := "allegro reflected actuator inertia"
  }

def reflectedInertia? (p : JointControlPlantParams) : Except String Float :=
  (p.actuatorModel).reflectedInertia?

def fingerBodyInertia (p : JointControlPlantParams) : Float :=
  (p.fingerMassEstimate / 3.0) * p.fingerLengthEstimate * p.fingerLengthEstimate

def effectiveFingerInertia? (p : JointControlPlantParams) : Except String Float := do
  let reflected ← p.reflectedInertia?
  pure (p.fingerBodyInertia + reflected)

def validate? (p : JointControlPlantParams) : Except String Unit := do
  (p.actuatorModel).validate?
  if !p.mbpDiscreteUpdatePeriod.isFinite || p.mbpDiscreteUpdatePeriod < 0.0 then
    .error s!"Allegro plant params: discrete update period must be nonnegative and finite, got {p.mbpDiscreteUpdatePeriod}"
  if !p.pidFrequency.isFinite || p.pidFrequency <= 0.0 then
    .error s!"Allegro plant params: pid frequency must be positive and finite, got {p.pidFrequency}"
  if !p.pointContactStiffness.isFinite || p.pointContactStiffness < 0.0 then
    .error s!"Allegro plant params: point contact stiffness must be nonnegative and finite, got {p.pointContactStiffness}"
  if !p.huntCrossleyDissipation.isFinite || p.huntCrossleyDissipation < 0.0 then
    .error s!"Allegro plant params: Hunt-Crossley dissipation must be nonnegative and finite, got {p.huntCrossleyDissipation}"
  if !p.fingerMassEstimate.isFinite || p.fingerMassEstimate <= 0.0 then
    .error s!"Allegro plant params: finger mass estimate must be positive and finite, got {p.fingerMassEstimate}"
  if !p.fingerLengthEstimate.isFinite || p.fingerLengthEstimate <= 0.0 then
    .error s!"Allegro plant params: finger length estimate must be positive and finite, got {p.fingerLengthEstimate}"
  if !p.dampingRatio.isFinite || p.dampingRatio < 0.0 then
    .error s!"Allegro plant params: damping ratio must be nonnegative and finite, got {p.dampingRatio}"

def positionControlledGains? (p : JointControlPlantParams) :
    Except String JointPidGains := do
  p.validate?
  let iEff ← p.effectiveFingerInertia?
  let gainProp := p.pidFrequency * p.pidFrequency * iEff
  let gainDer := 2.0 * p.pidFrequency * iEff * p.dampingRatio
  let mut kp := Array.replicate numJoints gainProp
  kp := kp.set! 0 (kp[0]! * 1.6)
  pure {
    kp := kp
    kd := Array.replicate numJoints gainDer
    ki := Array.replicate numJoints 0.0
    label := "Allegro SetPositionControlledGains"
  }

end JointControlPlantParams

structure AllegroCommandMessage where
  jointPosition : Array Float := #[]
  jointTorque : Array Float := #[]
  deriving Repr, Inhabited

structure AllegroReceiverState where
  commandedPosition : Array Float
  commandedVelocity : Array Float
  commandedTorque : Array Float
  deriving Repr, Inhabited

namespace AllegroReceiverState

def commandedState (state : AllegroReceiverState) : Array Float :=
  state.commandedPosition ++ state.commandedVelocity

end AllegroReceiverState

structure AllegroStatusMessage where
  utime : Float
  jointPositionMeasured : Array Float
  jointVelocityEstimated : Array Float
  jointPositionCommanded : Array Float
  jointTorqueCommanded : Array Float
  deriving Repr, Inhabited

private def validateVector? (xs : Array Float) (n : Nat) (label : String) :
    Except String Unit := do
  if xs.size != n then
    .error s!"{label}: size {xs.size} != expected {n}"
  for i in [:xs.size] do
    if !(xs[i]!).isFinite then
      .error s!"{label}: entry {i} must be finite, got {xs[i]!}"

def constantLoadHandModel
    (p : ConstantLoadParams := constantLoadParams) : FullMultibodyPlantModel :=
  {
    modelName :=
      if p.useRightHand then
        "allegro_right_hand_constant_load"
      else
        "allegro_left_hand_constant_load"
    modelUri := p.handModelUri
    numPositions := numJoints
    numVelocities := numJoints
    numActuatedDofs := numJoints
    floatingBases := #[]
    finalized := true
    label := "run_allegro_constant_load_demo MultibodyPlant"
  }

def constantLoadActuationMap : GeneralizedActuationMap :=
  GeneralizedActuationMap.identity numJoints
    "Allegro constant-load joint actuation map"

def constantLoadInitialQ : Array Float :=
  #[0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, -0.1, 0.0,
    0.0, 0.0, 0.0, 0.5,
    0.0, 0.0, 0.0, 0.0]

def constantLoadInitialV : Array Float :=
  zeroJoints

def constantLoadActuation
    (p : ConstantLoadParams := constantLoadParams) : Array Float :=
  Array.replicate numJoints p.constantLoad

def constantLoadPlantConfig
    (p : ConstantLoadParams := constantLoadParams) :
    MultibodyPlantConfigPrimitive :=
  {
    timeStep := p.maxTimeStep
    penetrationAllowance := 1.0e-3
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def constantLoadPlantStep
    (p : ConstantLoadParams := constantLoadParams) :
    FullMultibodyPlantStep :=
  {
    model := constantLoadHandModel p
    config := constantLoadPlantConfig p
    q0 := constantLoadInitialQ
    v0 := constantLoadInitialV
    actuation := constantLoadActuation p
    t0 := 0.0
    t1 := p.simulationTime
    ground? := none
    label := "run_allegro_constant_load_demo full plant advance"
  }

structure ConstantLoadPhysicsProvider where
  massDiagonal : Array Float
  gravityBiasForces : Array Float := #[]
  label : String := ""
  deriving Repr, Inhabited

namespace ConstantLoadPhysicsProvider

def validate? (provider : ConstantLoadPhysicsProvider)
    (p : ConstantLoadParams := constantLoadParams) : Except String Unit := do
  validateVector? provider.massDiagonal numJoints
    s!"Allegro constant-load mass diagonal {provider.label}"
  for i in [:provider.massDiagonal.size] do
    if provider.massDiagonal[i]! <= 0.0 then
      .error s!"Allegro constant-load mass diagonal {provider.label}: entry {i} must be positive, got {provider.massDiagonal[i]!}"
  if p.addGravity || !provider.gravityBiasForces.isEmpty then
    validateVector? provider.gravityBiasForces numJoints
      s!"Allegro constant-load gravity bias {provider.label}"

def biasForces? (provider : ConstantLoadPhysicsProvider)
    (p : ConstantLoadParams := constantLoadParams) : Except String (Array Float) := do
  provider.validate? p
  if p.addGravity then
    pure provider.gravityBiasForces
  else
    pure (Array.replicate numJoints 0.0)

end ConstantLoadPhysicsProvider

def nominalConstantLoadMassDiagonal : Array Float :=
  Array.replicate numJoints jointControlParams.fingerBodyInertia

def nominalConstantLoadGravityBias
    (q : Array Float := constantLoadInitialQ) : Array Float :=
  let scale :=
    jointControlParams.fingerMassEstimate * 9.81 *
      (jointControlParams.fingerLengthEstimate / 2.0)
  q.map (fun theta => scale * Float.sin theta)

def nominalConstantLoadPhysicsProvider : ConstantLoadPhysicsProvider :=
  {
    massDiagonal := nominalConstantLoadMassDiagonal
    gravityBiasForces := nominalConstantLoadGravityBias constantLoadInitialQ
    label := "nominal Allegro constant-load mass/gravity provider"
  }

def constantLoadFullPhysicsPrimitives?
    (p : ConstantLoadParams := constantLoadParams)
    (step : FullMultibodyPlantStep := constantLoadPlantStep p)
    (provider : ConstantLoadPhysicsProvider := nominalConstantLoadPhysicsProvider)
    (label : String := "allegro constant-load primitive full physics") :
    Except String FullPhysicsPrimitives := do
  p.validate?
  step.validate?
  provider.validate? p
  if step.model.numVelocities != numJoints then
    .error s!"Allegro constant-load plant velocities {step.model.numVelocities} != {numJoints}"
  if step.model.numPositions != numJoints then
    .error s!"Allegro constant-load plant positions {step.model.numPositions} != {numJoints}"
  if step.model.numActuatedDofs != numJoints then
    .error s!"Allegro constant-load actuated dofs {step.model.numActuatedDofs} != {numJoints}"
  let generalizedActuation ← constantLoadActuationMap.generalizedForcesFromStep? step
  let bias ← provider.biasForces? p
  pure {
    massMatrix := FloatMatrix.diagonal provider.massDiagonal
    qdot := step.v0
    actuationForces := generalizedActuation
    biasForces := bias
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := label
  }

def constantLoadIntervalVertex : VertexId := 8573

structure AllegroConstantLoadPrimitivePhysics where
  params : ConstantLoadParams
  provider : ConstantLoadPhysicsProvider
  plantStep : FullMultibodyPlantStep
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  graph : SkeletonGraph
  deriving Repr, Inhabited

def constantLoadPhysicsGraph (physics : FullPhysicsResult) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8570, kind := .state .boundary, label := "Allegro hand model URL" }
    |>.addVertex { id := 8571, kind := .state .interior, label := "Parser.AddModelsFromUrl + WeldJoint(hand_root)" }
    |>.addVertex { id := 8572, kind := .state .interior, label := "ConstantVectorSource constant_load" }
    |>.addVertex { id := constantLoadIntervalVertex, kind := .interval, label := "Allegro constant-load full physics plant interval" }
    |>.addVertex { id := 8574, kind := .state .checkpoint, label := "next Allegro constant-load hand state" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8571]
      reads := #[8570]
      writes := #[8571]
      label := "Parser.AddModelsFromUrl Allegro SDF; WeldJoint weld_hand; plant.Finalize"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8572]
      reads := #[8572]
      writes := #[8572]
      label := "ConstantVectorSource ones(num_actuators) * constant_load"
    }
    |>.addMove physics.supportMove
    |>.addMove physics.move

def solveAllegroConstantLoadPrimitivePhysics?
    (p : ConstantLoadParams := constantLoadParams)
    (provider : ConstantLoadPhysicsProvider := nominalConstantLoadPhysicsProvider)
    (intervalVertex : VertexId := constantLoadIntervalVertex)
    (label : String := "allegro constant-load primitive full physics") :
    Except String AllegroConstantLoadPrimitivePhysics := do
  p.validate?
  let step := constantLoadPlantStep p
  let primitives ← constantLoadFullPhysicsPrimitives? p step provider label
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := step
    primitives := primitives
    intervalVertex := intervalVertex
    label := label
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    params := p
    provider := provider
    plantStep := step
    primitivePlant := primitivePlant
    fullPhysics := fullPhysics
    graph := constantLoadPhysicsGraph fullPhysics
  }

def initialReceiverState (position : Array Float := zeroJoints) :
    Except String AllegroReceiverState := do
  validateVector? position numJoints "Allegro initial receiver position"
  pure {
    commandedPosition := position
    commandedVelocity := zeroJoints
    commandedTorque := zeroJoints
  }

def receiveCommand? (state : AllegroReceiverState)
    (command : AllegroCommandMessage) : Except String AllegroReceiverState := do
  validateVector? state.commandedPosition numJoints "Allegro receiver commanded position"
  validateVector? state.commandedVelocity numJoints "Allegro receiver commanded velocity"
  validateVector? state.commandedTorque numJoints "Allegro receiver commanded torque"
  let position ←
    if command.jointPosition.isEmpty then
      pure state.commandedPosition
    else
      validateVector? command.jointPosition numJoints "Allegro command joint_position"
      pure command.jointPosition
  let torque ←
    if command.jointTorque.isEmpty then
      pure zeroJoints
    else
      validateVector? command.jointTorque numJoints "Allegro command joint_torque"
      pure command.jointTorque
  pure {
    commandedPosition := position
    commandedVelocity := zeroJoints
    commandedTorque := torque
  }

def allegroJointControlMassDiagonal?
    (p : JointControlPlantParams := jointControlParams) :
    Except String (Array Float) := do
  let iEff ← p.effectiveFingerInertia?
  pure (Array.replicate numJoints iEff ++ mugSpatialInertiaDiagonal)

def allegroJointPidInput?
    (step : FullMultibodyPlantStep)
    (commanded : AllegroReceiverState) :
    Except String JointPidInput := do
  step.validate?
  validateVector? commanded.commandedPosition numJoints
    "Allegro commanded receiver position"
  validateVector? commanded.commandedVelocity numJoints
    "Allegro commanded receiver velocity"
  let estimated :=
    step.q0.extract 0 numJoints ++ step.v0.extract 0 numJoints
  pure {
    estimatedState := estimated
    desiredState := commanded.commandedState
    label := "allegro hand PID desired-state input"
  }

def allegroJointControlFullPhysicsPrimitives?
    (p : JointControlPlantParams)
    (step : FullMultibodyPlantStep)
    (controllerOutput : JointPidOutput)
    (label : String := "allegro hand+mug primitive full physics") :
    Except String FullPhysicsPrimitives := do
  step.validate?
  if controllerOutput.feedback.size != numJoints then
    .error s!"Allegro PID feedback size {controllerOutput.feedback.size} != {numJoints}"
  if step.actuation.size != numJoints then
    .error s!"Allegro plant step actuation size {step.actuation.size} != {numJoints}"
  for i in [:numJoints] do
    if step.actuation[i]! != controllerOutput.feedback[i]! then
      .error s!"Allegro plant step actuation[{i}] {step.actuation[i]!} != PID feedback {controllerOutput.feedback[i]!}"
  let massDiagonal ← allegroJointControlMassDiagonal? p
  let n := step.model.numVelocities
  if massDiagonal.size != n then
    .error s!"Allegro mass diagonal size {massDiagonal.size} != plant velocities {n}"
  let generalizedActuation ← allegroJointControlActuationMap.generalizedForcesFromStep? step
  pure {
    massMatrix := FloatMatrix.diagonal massDiagonal
    qdot := step.v0
    actuationForces := generalizedActuation
    biasForces := Array.replicate n 0.0
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := label
  }

structure AllegroJointControlPhysicsState where
  params : JointControlPlantParams := jointControlParams
  plantStep : FullMultibodyPlantStep :=
    allegroJointControlPlantStep jointControlParams 30.0
  commanded : AllegroReceiverState := {
    commandedPosition := zeroJoints
    commandedVelocity := zeroJoints
    commandedTorque := zeroJoints
  }
  deriving Repr, Inhabited

namespace AllegroJointControlPhysicsState

def validate? (snapshot : AllegroJointControlPhysicsState) :
    Except String Unit := do
  snapshot.params.validate?
  snapshot.plantStep.validate?
  validateVector? snapshot.commanded.commandedPosition numJoints
    "Allegro joint-control snapshot commanded position"
  validateVector? snapshot.commanded.commandedVelocity numJoints
    "Allegro joint-control snapshot commanded velocity"
  validateVector? snapshot.commanded.commandedTorque numJoints
    "Allegro joint-control snapshot commanded torque"
  discard (allegroJointPidInput? snapshot.plantStep snapshot.commanded)

def controllerOutput? (snapshot : AllegroJointControlPhysicsState) :
    Except String JointPidOutput := do
  snapshot.validate?
  let gains ← snapshot.params.positionControlledGains?
  let input ← allegroJointPidInput? snapshot.plantStep snapshot.commanded
  input.evaluate? gains

def actuatedStep? (snapshot : AllegroJointControlPhysicsState) :
    Except String FullMultibodyPlantStep := do
  let controllerOutput ← snapshot.controllerOutput?
  pure { snapshot.plantStep with actuation := controllerOutput.feedback }

def primitivesWithController?
    (snapshot : AllegroJointControlPhysicsState)
    (label : String := "allegro hand+mug primitive full physics") :
    Except String (JointPidOutput × FullMultibodyPlantStep × FullPhysicsPrimitives) := do
  let controllerOutput ← snapshot.controllerOutput?
  let actuatedStep := { snapshot.plantStep with actuation := controllerOutput.feedback }
  let primitives ← allegroJointControlFullPhysicsPrimitives?
    snapshot.params actuatedStep controllerOutput label
  pure (controllerOutput, actuatedStep, primitives)

end AllegroJointControlPhysicsState

def allegroJointControlPhysicsState
    (p : JointControlPlantParams := jointControlParams)
    (step : FullMultibodyPlantStep := allegroJointControlPlantStep jointControlParams 30.0)
    (commanded : AllegroReceiverState := {
      commandedPosition := zeroJoints
      commandedVelocity := zeroJoints
      commandedTorque := zeroJoints
    }) :
    AllegroJointControlPhysicsState :=
  {
    params := p
    plantStep := step
    commanded := commanded
  }

def allegroJointControlFullPhysicsPrimitiveProvider
    (label : String := "allegro hand+mug primitive full physics provider") :
    FullPhysicsPrimitiveProvider AllegroJointControlPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      let (_, _, primitives) ← snapshot.primitivesWithController? label
      pure primitives
  }

structure AllegroJointControlPrimitivePhysics where
  controllerOutput : JointPidOutput
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  deriving Repr, Inhabited

def solveAllegroJointControlPrimitivePhysics?
    (p : JointControlPlantParams)
    (step : FullMultibodyPlantStep)
    (commanded : AllegroReceiverState)
    (intervalVertex : VertexId := 8509)
    (label : String := "allegro hand+mug primitive full physics") :
    Except String AllegroJointControlPrimitivePhysics := do
  let snapshot := allegroJointControlPhysicsState p step commanded
  let (controllerOutput, actuatedStep, primitives) ←
    snapshot.primitivesWithController? label
  let primitivePlant : FullPlantPrimitivePhysics := {
    step := actuatedStep
    primitives := primitives
    intervalVertex := intervalVertex
    label := label
  }
  let fullPhysics ← primitivePlant.solve?
  pure {
    controllerOutput := controllerOutput
    primitivePlant := primitivePlant
    fullPhysics := fullPhysics
  }

def statusMessage? (timeSeconds : Float)
    (commandedState measuredState commandedTorque : Array Float) :
    Except String AllegroStatusMessage := do
  if !timeSeconds.isFinite then
    .error s!"Allegro status time must be finite, got {timeSeconds}"
  validateVector? commandedState (2 * numJoints) "Allegro commanded state"
  validateVector? measuredState (2 * numJoints) "Allegro measured state"
  validateVector? commandedTorque numJoints "Allegro commanded torque"
  pure {
    utime := timeSeconds * 1000000.0
    jointPositionMeasured := measuredState.extract 0 numJoints
    jointVelocityEstimated := measuredState.extract numJoints (2 * numJoints)
    jointPositionCommanded := commandedState.extract 0 numJoints
    jointTorqueCommanded := commandedTorque
  }

def fingerGraspJointPosition (fingerIndex : Nat) : Array Float :=
  if fingerIndex == 0 then
    #[1.396, 0.85, 0.0, 1.3]
  else if fingerIndex == 1 then
    #[0.08, 0.9, 0.75, 1.5]
  else if fingerIndex == 2 then
    #[0.1, 0.9, 0.75, 1.5]
  else
    #[0.12, 0.9, 0.75, 1.5]

def fingerOpenJointPosition (fingerIndex : Nat) : Array Float :=
  if fingerIndex == 0 then
    #[0.263, 1.1, 0.0, 0.0]
  else
    #[0.0, 0.0, 0.0, 0.0]

def openHandPosition : Array Float :=
  fingerOpenJointPosition 0 ++ fingerOpenJointPosition 1 ++
    fingerOpenJointPosition 2 ++ fingerOpenJointPosition 3

def graspHandPosition : Array Float :=
  fingerGraspJointPosition 0 ++ fingerGraspJointPosition 1 ++
    fingerGraspJointPosition 2 ++ fingerGraspJointPosition 3

private def setAtD (xs : Array Float) (i : Nat) (x : Float) : Array Float :=
  if i < xs.size then xs.set! i x else xs

private def writeSegment (xs : Array Float) (start : Nat)
    (values : Array Float) : Array Float := Id.run do
  let mut out := xs
  for i in [:values.size] do
    out := setAtD out (start + i) values[i]!
  return out

private def addSegment (xs : Array Float) (start : Nat)
    (values : Array Float) : Array Float := Id.run do
  let mut out := xs
  for i in [:values.size] do
    let idx := start + i
    out := setAtD out idx (out.getD idx 0.0 + values[i]!)
  return out

def closeThumbTarget : Array Float :=
  let withThumbBase := setAtD zeroJoints 0 1.396
  setAtD withThumbBase 1 0.3

def indexTwistTarget (closed : Array Float := graspHandPosition) : Array Float :=
  let withMiddlePivot := addSegment closed 9 #[0.1, 0.1, 0.05]
  let withThumb := writeSegment withMiddlePivot 0 (fingerGraspJointPosition 0)
  addSegment withThumb 5 #[0.6, 0.18, 0.3]

def ringTwistTarget (closed : Array Float := graspHandPosition) : Array Float :=
  let withMiddlePivot := addSegment closed 9 #[0.1, 0.1, 0.05]
  let withThumb := writeSegment withMiddlePivot 0 (fingerGraspJointPosition 0)
  addSegment withThumb 13 #[0.6, 0.18, 0.3]

structure TwistingMugPlan where
  initialOpen : Array Float
  closeThumb : Array Float
  closedGrasp : Array Float
  indexTwist : Array Float
  ringTwist : Array Float
  deriving Repr, Inhabited

def twistingMugPlan : TwistingMugPlan :=
  {
    initialOpen := zeroJoints
    closeThumb := closeThumbTarget
    closedGrasp := graspHandPosition
    indexTwist := indexTwistTarget graspHandPosition
    ringTwist := ringTwistTarget graspHandPosition
  }

structure MotionState where
  jointStuck : Array Bool
  fingerStuck : Array Bool
  motorReverse : Array Bool
  deriving Repr, Inhabited

namespace MotionState

def isFingerStuck (state : MotionState) (fingerIndex : Nat) : Bool :=
  state.fingerStuck.getD fingerIndex false

def isAllFingersStuck (state : MotionState) : Bool :=
  state.fingerStuck.size == fingerCount && state.fingerStuck.all id

def isAnyHighFingerStuck (state : MotionState) : Bool :=
  (state.fingerStuck.getD 1 false) ||
    (state.fingerStuck.getD 2 false) ||
    (state.fingerStuck.getD 3 false)

end MotionState

def velocityThreshold : Float := 0.07
def motorReverseThreshold : Float := -0.001

private def allTrueSegment (xs : Array Bool) (start len : Nat) : Bool :=
  (Array.range len).all (fun i => xs.getD (start + i) false)

private def anyTrueSegment (xs : Array Bool) (start len : Nat) : Bool :=
  (Array.range len).any (fun i => xs.getD (start + i) false)

def classifyMotion? (status : AllegroStatusMessage) : Except String MotionState := do
  validateVector? status.jointVelocityEstimated numJoints "Allegro status velocity"
  validateVector? status.jointTorqueCommanded numJoints "Allegro status commanded torque"
  let mut jointStuck : Array Bool := #[]
  let mut motorReverse : Array Bool := #[]
  for i in [:numJoints] do
    let v := status.jointVelocityEstimated[i]!
    let tau := status.jointTorqueCommanded[i]!
    let reverse := v * tau < motorReverseThreshold
    motorReverse := motorReverse.push reverse
    jointStuck := jointStuck.push (Float.abs v < velocityThreshold || reverse)
  let thumbStuck := allTrueSegment jointStuck 0 4
  let indexStuck :=
    allTrueSegment jointStuck 5 3 || anyTrueSegment motorReverse 5 3
  let middleStuck :=
    allTrueSegment jointStuck 9 3 || anyTrueSegment motorReverse 9 3
  let ringStuck :=
    allTrueSegment jointStuck 13 3 || anyTrueSegment motorReverse 13 3
  pure {
    jointStuck := jointStuck
    fingerStuck := #[thumbStuck, indexStuck, middleStuck, ringStuck]
    motorReverse := motorReverse
  }

def controllerGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8500, kind := .state .boundary, label := "ALLEGRO_COMMAND" }
    |>.addVertex { id := 8501, kind := .state .interior, label := "command_receiver_discrete_state" }
    |>.addVertex { id := 8502, kind := .state .boundary, label := "plant_hand_state_preferred_order" }
    |>.addVertex { id := 8503, kind := .state .interior, label := "preferred_joint_selector" }
    |>.addVertex { id := 8504, kind := .state .interior, label := "reflected_actuator_inertia" }
    |>.addVertex { id := 8505, kind := .state .boundary, label := "plant_actuation" }
    |>.addVertex { id := 8506, kind := .state .boundary, label := "ALLEGRO_STATUS" }
    |>.addVertex { id := 8507, kind := .eventTime, label := "hardware_status_period" }
    |>.addVertex { id := 8508, kind := .state .interior, label := "motion_state_classifier" }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[8500, 8507]
      reads := #[8500, 8501]
      writes := #[8501]
      label := "allegro-command-receiver-latched-update"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8503, 8504]
      reads := #[8501, 8502, 8503, 8504]
      writes := #[8505]
      label := "allegro-position-pid-with-reflected-inertia"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[8507]
      reads := #[8501, 8502, 8505]
      writes := #[8506]
      label := "allegro-status-sender-periodic-publish"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8506]
      reads := #[8506]
      writes := #[8508]
      label := "allegro-motion-state-stuck-finger-classifier"
    }

def fullPhysicsIntervalVertex : VertexId := 8509

def closedLoopPhysicsGraph (physics : FullPhysicsResult) : SkeletonGraph :=
  controllerGraph
    |>.addVertex { id := fullPhysicsIntervalVertex, kind := .interval, label := "allegro hand+mug full physics plant interval" }
    |>.addVertex { id := 8510, kind := .state .checkpoint, label := "next allegro hand+mug state" }
    |>.addMove physics.supportMove
    |>.addMove physics.move

structure AllegroHandResult where
  references : Array DrakeReference
  parseTest : AllegroParseTestBoundary
  params : JointControlPlantParams
  gains : JointPidGains
  controllerOutput : JointPidOutput
  plantStep : FullMultibodyPlantStep
  primitivePlant : FullPlantPrimitivePhysics
  fullPhysics : FullPhysicsResult
  initialReceiver : AllegroReceiverState
  commandedReceiver : AllegroReceiverState
  status : AllegroStatusMessage
  motionState : MotionState
  plan : TwistingMugPlan
  graph : SkeletonGraph
  deriving Repr, Inhabited

def samplePosition : Array Float :=
  #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

def sampleVelocity : Array Float :=
  #[1.6, 1.5, 1.4, 1.3, 1.2, 0.01, 0.02, 0.03,
    0.8, 0.7, 0.6, 0.5, 0.4, -0.1, -0.1, -0.1]

def sampleTorque : Array Float :=
  #[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]

def buildEndToEnd? : Except String AllegroHandResult := do
  allegroParseTestBoundary.validate?
  let initial ← initialReceiverState zeroJoints
  let commanded ← receiveCommand? initial { jointPosition := graspHandPosition }
  let seedPlantStep := allegroJointControlPlantStep jointControlParams 30.0
  let primitivePhysics ←
    solveAllegroJointControlPrimitivePhysics?
      jointControlParams seedPlantStep commanded fullPhysicsIntervalVertex
  let plantStep := primitivePhysics.primitivePlant.step
  let gains ← jointControlParams.positionControlledGains?
  let status ← statusMessage? hardwareStatusPeriod
    commanded.commandedState (samplePosition ++ sampleVelocity)
    primitivePhysics.controllerOutput.feedback
  let motion ← classifyMotion? status
  pure {
    references := drakeReferences
    parseTest := allegroParseTestBoundary
    params := jointControlParams
    gains := gains
    controllerOutput := primitivePhysics.controllerOutput
    plantStep := plantStep
    primitivePlant := primitivePhysics.primitivePlant
    fullPhysics := primitivePhysics.fullPhysics
    initialReceiver := initial
    commandedReceiver := commanded
    status := status
    motionState := motion
    plan := twistingMugPlan
    graph := closedLoopPhysicsGraph primitivePhysics.fullPhysics
  }

structure RunTwistingMugPythonTestBoundary where
  testPath : String :=
    "../drake/examples/allegro_hand/joint_control/test/run_twisting_mug_test.py"
  simResource : String :=
    "drake/examples/allegro_hand/joint_control/allegro_single_object_simulation"
  controlResource : String :=
    "drake/examples/allegro_hand/joint_control/run_twisting_mug"
  testTmpdirEnv : String := "TEST_TMPDIR"
  lcmUrlEnv : String := "LCM_DEFAULT_URL"
  lcmUrlSource : String :=
    "udpm://239.{sha256(TEST_TMPDIR)[0]}.{sha256(TEST_TMPDIR)[1]}.{sha256(TEST_TMPDIR)[2]}:{20000 + sha256(TEST_TMPDIR)[3]}?ttl=0"
  onlySimCommand : Array String :=
    #["allegro_single_object_simulation", "--simulation_time=0.01"]
  coupledSimCommand : Array String :=
    #["allegro_single_object_simulation", "--simulation_time=30"]
  coupledControlCommand : Array String :=
    #["run_twisting_mug", "--max_cycles=1"]
  onlySimSimulationTime : Float := 0.01
  coupledSimulationTime : Float := 30.0
  maxCyclesDefault : Nat := 1000000000
  smokeMaxCycles : Nat := 1
  initialStatusWaitCount : Nat := 60
  handleSubscriptionsTimeoutMs : Nat := 10
  statusPollPeriod : Float := 0.1
  killTimeout : Float := 10.0
  skipOnDarwin : Bool := true
  skipInDebugBuild : Bool := true
  expectedControlReturnCode : Option Int := some 0
  expectedSimStillRunningWhenControlExits : Bool := true
  plantStep : FullMultibodyPlantStep :=
    allegroJointControlPlantStep jointControlParams 30.0
  deriving Repr, Inhabited

namespace RunTwistingMugPythonTestBoundary

private def requireCommand? (actual expected : Array String) (label : String) :
    Except String Unit := do
  if actual != expected then
    .error s!"{label} command mismatch: expected {reprStr expected}, got {reprStr actual}"

def validate? (boundary : RunTwistingMugPythonTestBoundary) :
    Except String Unit := do
  if boundary.testPath != "../drake/examples/allegro_hand/joint_control/test/run_twisting_mug_test.py" then
    .error s!"Allegro twisting mug Python test path mismatch: {boundary.testPath}"
  if boundary.simResource != "drake/examples/allegro_hand/joint_control/allegro_single_object_simulation" then
    .error s!"Allegro twisting mug simulator resource mismatch: {boundary.simResource}"
  if boundary.controlResource != "drake/examples/allegro_hand/joint_control/run_twisting_mug" then
    .error s!"Allegro twisting mug controller resource mismatch: {boundary.controlResource}"
  if boundary.testTmpdirEnv != "TEST_TMPDIR" then
    .error s!"Allegro twisting mug test should derive cwd and LCM URL from TEST_TMPDIR, got {boundary.testTmpdirEnv}"
  if boundary.lcmUrlEnv != "LCM_DEFAULT_URL" then
    .error s!"Allegro twisting mug test should write LCM_DEFAULT_URL, got {boundary.lcmUrlEnv}"
  if !(boundary.lcmUrlSource.contains "sha256") || !(boundary.lcmUrlSource.contains "ttl=0") then
    .error s!"Allegro twisting mug LCM URL source should record the TEST_TMPDIR sha256 udpm formula, got {boundary.lcmUrlSource}"
  requireCommand? boundary.onlySimCommand
    #["allegro_single_object_simulation", "--simulation_time=0.01"]
    "Allegro only-sim smoke"
  requireCommand? boundary.coupledSimCommand
    #["allegro_single_object_simulation", "--simulation_time=30"]
    "Allegro coupled simulator"
  requireCommand? boundary.coupledControlCommand
    #["run_twisting_mug", "--max_cycles=1"]
    "Allegro coupled controller"
  if !boundary.onlySimSimulationTime.isFinite || boundary.onlySimSimulationTime <= 0.0 then
    .error s!"Allegro only-sim simulation time must be positive and finite, got {boundary.onlySimSimulationTime}"
  if !boundary.coupledSimulationTime.isFinite || boundary.coupledSimulationTime <= 0.0 then
    .error s!"Allegro coupled simulation time must be positive and finite, got {boundary.coupledSimulationTime}"
  if boundary.maxCyclesDefault != 1000000000 then
    .error s!"Allegro run_twisting_mug default max_cycles should match Drake, got {boundary.maxCyclesDefault}"
  if boundary.smokeMaxCycles != 1 then
    .error s!"Allegro Python smoke controller should run one twist cycle, got {boundary.smokeMaxCycles}"
  if boundary.initialStatusWaitCount != 60 then
    .error s!"Allegro PositionCommander initial status wait should be 60, got {boundary.initialStatusWaitCount}"
  if boundary.handleSubscriptionsTimeoutMs != 10 then
    .error s!"Allegro LCM HandleSubscriptions timeout should be 10 ms, got {boundary.handleSubscriptionsTimeoutMs}"
  if !boundary.statusPollPeriod.isFinite || boundary.statusPollPeriod <= 0.0 then
    .error s!"Allegro Python polling period must be positive and finite, got {boundary.statusPollPeriod}"
  if !boundary.killTimeout.isFinite || boundary.killTimeout <= 0.0 then
    .error s!"Allegro process kill timeout must be positive and finite, got {boundary.killTimeout}"
  if !boundary.skipOnDarwin then
    .error "Allegro coupled sim/controller regression is skipped on Darwin in Drake"
  if !boundary.skipInDebugBuild then
    .error "Allegro coupled sim/controller regression is skipped in Debug builds in Drake"
  if boundary.expectedControlReturnCode != some 0 then
    .error s!"Allegro controller should exit cleanly with code 0, got {reprStr boundary.expectedControlReturnCode}"
  if !boundary.expectedSimStillRunningWhenControlExits then
    .error "Allegro simulator should still be running when the one-cycle controller exits"
  if boundary.plantStep.t1 != boundary.coupledSimulationTime then
    .error s!"Allegro full-plant step should use coupled simulation time {boundary.coupledSimulationTime}, got {boundary.plantStep.t1}"
  boundary.plantStep.validate?

def graph (boundary : RunTwistingMugPythonTestBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8550, kind := .state .boundary, label := boundary.testPath }
    |>.addVertex { id := 8551, kind := .state .boundary, label := boundary.testTmpdirEnv }
    |>.addVertex { id := 8552, kind := .state .interior, label := boundary.lcmUrlEnv }
    |>.addVertex { id := 8553, kind := .state .boundary, label := boundary.simResource }
    |>.addVertex { id := 8554, kind := .state .boundary, label := boundary.controlResource }
    |>.addVertex { id := 8555, kind := .state .interior, label := "allegro hand + floating mug FullMultibodyPlantStep" }
    |>.addVertex { id := 8556, kind := .interval, label := "test_only_sim subprocess.run check=True" }
    |>.addVertex { id := 8557, kind := .interval, label := "coupled allegro_single_object_simulation process" }
    |>.addVertex { id := 8558, kind := .interval, label := "run_twisting_mug --max_cycles=1 process" }
    |>.addVertex { id := 8559, kind := .checkpoint, label := "Darwin/debug unittest skip gates" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8553, 8554]
      reads := #[8550]
      writes := #[8553, 8554]
      label := "python.runfiles resolves simulator and controller resources"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8552]
      reads := #[8551]
      writes := #[8552]
      label := "TEST_TMPDIR sha256 unique LCM_DEFAULT_URL"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8555]
      reads := #[8553]
      writes := #[8555]
      label := "Parser.AddModelsFromUrl hand+mug, contact material, reflected inertia, PID, LCM diagram"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[8556]
      reads := #[8552, 8553, 8555]
      writes := #[8556]
      cost := { work := boundary.onlySimSimulationTime }
      label := "subprocess.run allegro_single_object_simulation --simulation_time=0.01"
    }
    |>.addMove {
      kind := .checkpointBoundary
      targets := #[8559]
      reads := #[8550]
      writes := #[8559]
      label := "unittest skipIf Darwin or debug build"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[8557]
      reads := #[8552, 8553, 8555, 8559]
      writes := #[8557]
      cost := { work := boundary.coupledSimulationTime }
      label := "Popen allegro_single_object_simulation --simulation_time=30 full-physics process"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[8558]
      reads := #[8552, 8554, 8557, 8559]
      writes := #[8558]
      cost := { work := boundary.statusPollPeriod * Float.ofNat boundary.smokeMaxCycles }
      label := "Popen run_twisting_mug --max_cycles=1; expect controller exit 0 while sim remains live"
    }

end RunTwistingMugPythonTestBoundary

def runTwistingMugPythonTestBoundary : RunTwistingMugPythonTestBoundary := {}

structure RunTwistingMugPythonTestResult where
  boundary : RunTwistingMugPythonTestBoundary
  controller : AllegroHandResult
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildRunTwistingMugPythonTest?
    (boundary : RunTwistingMugPythonTestBoundary := runTwistingMugPythonTestBoundary) :
    Except String RunTwistingMugPythonTestResult := do
  boundary.validate?
  let controller ← buildEndToEnd?
  let boundaryGraph := boundary.graph
  let graph : SkeletonGraph := {
    vertices := boundaryGraph.vertices ++ controller.graph.vertices
    moves := boundaryGraph.moves ++ controller.graph.moves
  }
  pure {
    boundary := boundary
    controller := controller
    graph := graph
    moves := graph.moves
  }

end Tyr.EventSkeleton.Examples.AllegroHand
