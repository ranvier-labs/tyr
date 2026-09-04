import Tyr.EventSkeleton.Physics
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Trace

/-!
# Drake-Style Mass-Spring Cloth Example

This ports the continuous force side of
`../drake/examples/mass_spring_cloth`: a rectangular particle grid with
stretching and shearing springs, two fixed corners, spring damping, and gravity.

The example keeps the topology and Drake defaults visible while delegating the
spring-graph force accumulation to `ParticleSpringSystem`.
-/

namespace Tyr.EventSkeleton.Examples.MassSpringCloth

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/mass_spring_cloth/cloth_spring_model.h"
      concept := "declares continuous and discrete ClothSpringModel ports, pinned-corner boundary conditions, and implicit damping solver hooks"
    },
    {
      path := "../drake/examples/mass_spring_cloth/cloth_spring_model.cc"
      concept := "initializes the nx-by-ny grid, builds stretch/shear springs, applies pinned corners, and evaluates spring forces"
    },
    {
      path := "../drake/examples/mass_spring_cloth/cloth_spring_model_params.h"
      concept := "defines mass, stiffness, damping, and gravity defaults"
    },
    {
      path := "../drake/examples/mass_spring_cloth/cloth_spring_model_params.cc"
      concept := "defines ClothSpringModelParams coordinate names"
    },
    {
      path := "../drake/examples/mass_spring_cloth/run_cloth_spring_model.cc"
      concept := "declares the nx/ny/h/dt flags and visualizes particle positions through SceneGraph"
    },
    {
      path := "../drake/examples/mass_spring_cloth/cloth_spring_model_geometry.h"
      concept := "declares the ClothSpringModelGeometry system with particle_positions input and geometry_pose output"
    },
    {
      path := "../drake/examples/mass_spring_cloth/cloth_spring_model_geometry.cc"
      concept := "registers one magenta sphere frame per particle with radius 0.8*h"
    },
    {
      path := "../drake/examples/mass_spring_cloth/README.md"
      concept := "documents fixed-corner cloth dynamics and the continuous/discrete solver split"
    },
    {
      path := "../drake/examples/mass_spring_cloth/test/cloth_spring_model_test.cc"
      concept := "regresses continuous simulation and discrete implicit-damping convergence"
    }
  ]

def parameterCoordinateNames : Array String :=
  #["mass", "k", "d", "gravity"]

structure ClothParams where
  nx : Nat := 20
  ny : Nat := 20
  spacing : Float := 0.05
  dt : Float := 0.01
  simulationTime? : Option Float := none
  physical : ParticleSpringParams := {}
  deriving Repr, Inhabited

def params : ClothParams := {}

namespace ClothParams

def particleCount (p : ClothParams) : Nat :=
  p.nx * p.ny

def positionDim (p : ClothParams) : Nat :=
  ParticleSpringSystem.stateDim p.particleCount

def bottomLeftCorner (_p : ClothParams) : Nat :=
  0

def topLeftCorner (p : ClothParams) : Nat :=
  if p.ny == 0 then 0 else p.ny - 1

def pinnedParticles (p : ClothParams) : Array Nat :=
  #[p.bottomLeftCorner, p.topLeftCorner]

def stretchingSpringCount (p : ClothParams) : Nat :=
  (p.nx - 1) * p.ny + (p.ny - 1) * p.nx

def shearingSpringCount (p : ClothParams) : Nat :=
  2 * (p.nx - 1) * (p.ny - 1)

def springCount (p : ClothParams) : Nat :=
  p.stretchingSpringCount + p.shearingSpringCount

def validate? (p : ClothParams) : Except String Unit := do
  if p.nx == 0 then
    .error "cloth nx must be positive"
  if p.ny == 0 then
    .error "cloth ny must be positive"
  if !(Float.isFinite p.spacing) || p.spacing <= 0.0 then
    .error s!"cloth spacing must be positive and finite, got {p.spacing}"
  if !(Float.isFinite p.dt) then
    .error s!"cloth dt must be finite, got {p.dt}"
  p.physical.validate? p.particleCount

def traceStepSize (p : ClothParams) : Float :=
  if p.dt > 0.0 then p.dt else 0.001

end ClothParams

structure ClothState where
  q : Array Float
  v : Array Float
  deriving Repr, Inhabited

namespace ClothState

private def finiteArray (xs : Array Float) : Bool :=
  xs.all Float.isFinite

def isFinite (x : ClothState) : Bool :=
  finiteArray x.q && finiteArray x.v

def validate? (p : ClothParams) (x : ClothState) : Except String Unit := do
  ParticleSpringSystem.validateStateSize? p.particleCount x.q "cloth q"
  ParticleSpringSystem.validateStateSize? p.particleCount x.v "cloth v"

def particlePosition (x : ClothState) (particle : Nat) : Array Float :=
  ParticleSpringSystem.particleState particle x.q

def particleVelocity (x : ClothState) (particle : Nat) : Array Float :=
  ParticleSpringSystem.particleState particle x.v

end ClothState

def springs (p : ClothParams := params) : Array ParticleSpring :=
  ParticleSpringSystem.gridSprings p.nx p.ny p.spacing true

def defaultState (p : ClothParams := params) : ClothState :=
  {
    q := ParticleSpringSystem.initialGridPositions p.nx p.ny p.spacing
    v := ParticleSpringSystem.zeroVelocities p.particleCount
  }

def visualParticleRadius (p : ClothParams := params) : Float :=
  0.8 * p.spacing

def particleFrameName (particle : Nat) : String :=
  s!"particle{particle}"

/-! ## ClothSpringModelGeometry SceneGraph provider -/

def clothGeometrySourceId : Nat := 4400
def clothGeometryFrameBaseId : Nat := 4401
def clothGeometryIdBase : Nat := 4801

def clothGeometryStateInputVertex : VertexId := 4900
def clothGeometryProviderVertex : VertexId := 4901
def clothGeometryPoseOutputVertex : VertexId := 4902
def clothDiscreteUpdateVertex : VertexId := 4903
def clothFullPhysicsIntervalVertex : VertexId := 4904

def particleFrameId (particle : Nat) : Nat :=
  clothGeometryFrameBaseId + particle

def particleGeometryId (particle : Nat) : Nat :=
  clothGeometryIdBase + particle

private def clothIllustrationProperties : SceneGeometryProperties :=
  { roles := #[.illustration], diffuseRgba? := some { r := 1.0, g := 0.0, b := 1.0, a := 1.0 } }

private def particleFrames (p : ClothParams) : Array SceneFrame := Id.run do
  let mut out : Array SceneFrame := #[]
  for i in [:p.particleCount] do
    out := out.push {
      id := particleFrameId i
      sourceId := clothGeometrySourceId
      name := particleFrameName i
    }
  return out

private def particleGeometries (p : ClothParams) : Array SceneGeometry := Id.run do
  let mut out : Array SceneGeometry := #[]
  for i in [:p.particleCount] do
    out := out.push {
      id := particleGeometryId i
      sourceId := clothGeometrySourceId
      frameId? := some (particleFrameId i)
      X_FG := ScenePose3.identity
      shape := .sphere (visualParticleRadius p)
      name := "sphere_visual"
      properties := clothIllustrationProperties
    }
  return out

def clothGeometryProvider (p : ClothParams := params) : SceneGraphProvider :=
  {
    sources := #[
      { id := clothGeometrySourceId, name := "cloth_spring_model_geometry" }
    ]
    frames := particleFrames p
    geometries := particleGeometries p
    label := "ClothSpringModelGeometry SceneGraph provider"
  }

def clothGeometryPoseOutput (p : ClothParams) (particlePositions : Array Float) :
    SceneFramePoseVector := Id.run do
  let mut poses : Array SceneFramePose := #[]
  for i in [:p.particleCount] do
    poses := poses.push {
      frameId := particleFrameId i
      X_WF := {
        translation := {
          x := particlePositions.getD (3 * i) 0.0
          y := particlePositions.getD (3 * i + 1) 0.0
          z := particlePositions.getD (3 * i + 2) 0.0
        }
        rotationAxis := SceneVec3.unitZ
        rotationAngle := 0.0
      }
    }
  return { poses := poses }

private def clothGeometryMove
    (target : VertexId) (label : String) (reads : Array VertexId := #[])
    (writes : Array VertexId := #[]) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target]
    reads := reads
    writes := writes
    exactness := .exact
    label := label
  }

def clothGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := clothGeometryStateInputVertex
      kind := .state .boundary
      label := "ClothSpringModelGeometry particle_positions input"
    }
    |>.addVertex {
      id := clothGeometryProviderVertex
      kind := .state .boundary
      label := "ClothSpringModelGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := clothGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "ClothSpringModelGeometry geometry_pose output"
    }
    |>.addMove (clothGeometryMove clothGeometryProviderVertex
      "Register particle frames and magenta sphere visuals"
      #[] #[clothGeometryProviderVertex])
    |>.addMove (clothGeometryMove clothGeometryPoseOutputVertex
      "OutputGeometryPose: particle_positions -> particle FramePoseVector"
      #[clothGeometryStateInputVertex, clothGeometryProviderVertex]
      #[clothGeometryPoseOutputVertex])

structure ClothSpringModelGeometryResult where
  references : Array DrakeReference
  params : ClothParams
  inputPortName : String := "particle_positions"
  inputPortSize : Nat
  outputPortName : String := "geometry_pose"
  particleRadius : Float
  provider : SceneGraphProvider
  samplePositions : Array Float
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildClothSpringModelGeometry?
    (p : ClothParams := params)
    (x : ClothState := defaultState p) :
    Except String ClothSpringModelGeometryResult := do
  p.validate?
  x.validate? p
  if !x.q.all Float.isFinite then
    .error "ClothSpringModelGeometry particle_positions input must be finite"
  let provider := clothGeometryProvider p
  provider.validate?
  let poses := clothGeometryPoseOutput p x.q
  poses.validate? provider
  pure {
    references := drakeReferences
    params := p
    inputPortSize := p.positionDim
    particleRadius := visualParticleRadius p
    provider := provider
    samplePositions := x.q
    poses := poses
    graph := clothGeometryGraph
    moves := clothGeometryGraph.moves
  }

structure ClothDerivative where
  qdot : Array Float
  vdot : Array Float
  springForces : ParticleSpringForceResult
  deriving Repr, Inhabited

structure ClothDiscreteSolverSettings where
  maxIterations? : Option Nat := none
  accuracy : Float := 0.0001
  deriving Repr, Inhabited

namespace ClothDiscreteSolverSettings

def effectiveMaxIterations (p : ClothParams) (settings : ClothDiscreteSolverSettings) :
    Nat :=
  match settings.maxIterations? with
  | some n => n
  | none => p.positionDim

def validate? (p : ClothParams) (settings : ClothDiscreteSolverSettings) :
    Except String Unit := do
  if let some n := settings.maxIterations? then
    if n == 0 then
      .error "cloth discrete solver max iterations must be positive"
  if !(Float.isFinite settings.accuracy) || settings.accuracy <= 0.0 then
    .error s!"cloth discrete solver accuracy must be positive and finite, got {settings.accuracy}"
  if settings.effectiveMaxIterations p < p.positionDim then
    .error s!"cloth discrete solver max iterations {settings.effectiveMaxIterations p} is smaller than linear system size {p.positionDim}"

end ClothDiscreteSolverSettings

private def vec3Sub (a b : Array Float) : Array Float :=
  #[
    a.getD 0 0.0 - b.getD 0 0.0,
    a.getD 1 0.0 - b.getD 1 0.0,
    a.getD 2 0.0 - b.getD 2 0.0
  ]

private def vec3Scale (s : Float) (v : Array Float) : Array Float :=
  #[s * v.getD 0 0.0, s * v.getD 1 0.0, s * v.getD 2 0.0]

private def vec3Dot (a b : Array Float) : Float :=
  a.getD 0 0.0 * b.getD 0 0.0 +
  a.getD 1 0.0 * b.getD 1 0.0 +
  a.getD 2 0.0 * b.getD 2 0.0

private def vec3Norm (v : Array Float) : Float :=
  Float.sqrt (vec3Dot v v)

private def springRelativeTolerance : Float :=
  2.220446049250313e-15

private def springDirection?
    (q : Array Float) (spring : ParticleSpring) : Except String (Array Float) := do
  let x0 := ParticleSpringSystem.particleState spring.particle0 q
  let x1 := ParticleSpringSystem.particleState spring.particle1 q
  let dx := vec3Sub x1 x0
  let length := vec3Norm dx
  if length < springRelativeTolerance * spring.restLength then
    .error "two spring particles are nearly coincident; the state is invalid"
  pure (vec3Scale (1.0 / length) dx)

private def zeroMatrix (n : Nat) : Array (Array Float) :=
  Array.replicate n (Array.replicate n 0.0)

private def setMatrixEntry
    (m : Array (Array Float)) (i j : Nat) (value : Float) :
    Array (Array Float) :=
  let row := m.getD i #[]
  m.set! i (row.set! j value)

private def addMatrixEntry
    (m : Array (Array Float)) (i j : Nat) (delta : Float) :
    Array (Array Float) :=
  setMatrixEntry m i j ((m.getD i #[]).getD j 0.0 + delta)

private def addBlock
    (m : Array (Array Float)) (rowParticle colParticle : Nat)
    (block : Array (Array Float)) (scale : Float := 1.0) :
    Array (Array Float) := Id.run do
  let mut out := m
  for j in [:3] do
    for k in [:3] do
      out := addMatrixEntry out (3 * rowParticle + j) (3 * colParticle + k)
        (scale * (block.getD j #[]).getD k 0.0)
  return out

private def setBlock
    (m : Array (Array Float)) (rowParticle colParticle : Nat)
    (block : Array (Array Float)) (scale : Float := 1.0) :
    Array (Array Float) := Id.run do
  let mut out := m
  for j in [:3] do
    for k in [:3] do
      out := setMatrixEntry out (3 * rowParticle + j) (3 * colParticle + k)
        (scale * (block.getD j #[]).getD k 0.0)
  return out

private def setIdentityBlock (m : Array (Array Float)) (particle : Nat) :
    Array (Array Float) := Id.run do
  let mut out := m
  for j in [:3] do
    for k in [:3] do
      out := setMatrixEntry out (3 * particle + j) (3 * particle + k)
        (if j == k then 1.0 else 0.0)
  return out

private def dampingDerivativeBlock
    (p : ClothParams) (dt : Float) (n : Array Float) :
    Array (Array Float) := Id.run do
  let mut rows : Array (Array Float) := #[]
  for j in [:3] do
    let mut row : Array Float := #[]
    for k in [:3] do
      row := row.push (p.physical.damping * dt * n.getD j 0.0 * n.getD k 0.0)
    rows := rows.push row
  return rows

def applyDirichletBoundary (p : ClothParams) (xs : Array Float) : Array Float :=
  p.pinnedParticles.foldl
    (fun acc particle => ParticleSpringSystem.setParticleState acc particle #[0.0, 0.0, 0.0])
    xs

def elasticForces? (p : ClothParams) (q : Array Float) :
    Except String (Array Float × ParticleSpringForceResult) := do
  let zeros := ParticleSpringSystem.zeroVelocities p.particleCount
  let result ← ParticleSpringSystem.accumulateForces?
    p.particleCount p.physical (springs p) q zeros
  pure (result.elasticForces, result)

def dampingForces? (p : ClothParams) (q v : Array Float) :
    Except String (Array Float × ParticleSpringForceResult) := do
  let result ← ParticleSpringSystem.accumulateForces?
    p.particleCount p.physical (springs p) q v
  pure (result.dampingForces, result)

def explicitVelocityPrediction? (p : ClothParams) (x : ClothState) :
    Except String (Array Float × Array Float × ParticleSpringForceResult) := do
  p.validate?
  x.validate? p
  if p.dt <= 0.0 then
    .error s!"cloth discrete update requires dt > 0, got {p.dt}"
  let m := p.physical.massPerParticle p.particleCount
  if !(Float.isFinite m) || m <= 0.0 then
    .error s!"mass per particle must be positive and finite, got {m}"
  let (elastic, elasticResult) ← elasticForces? p x.q
  let gravityDv :=
    FloatArray.scale p.dt
      (ParticleSpringSystem.gravityAccelerations p.particleCount p.physical.gravityZ)
  let vHat :=
    applyDirichletBoundary p
      (FloatArray.add (FloatArray.add x.v gravityDv)
        (FloatArray.scale (p.dt / m) elastic))
  pure (vHat, elastic, elasticResult)

def implicitDampingMatrix? (p : ClothParams) (q : Array Float) :
    Except String (Array (Array Float)) := do
  p.validate?
  ParticleSpringSystem.validateStateSize? p.particleCount q "cloth q"
  if p.dt <= 0.0 then
    .error s!"cloth implicit damping matrix requires dt > 0, got {p.dt}"
  let m := p.physical.massPerParticle p.particleCount
  if !(Float.isFinite m) || m <= 0.0 then
    .error s!"mass per particle must be positive and finite, got {m}"
  let dim := p.positionDim
  let mut h := zeroMatrix dim
  for i in [:dim] do
    h := setMatrixEntry h i i m
  let pinned := p.pinnedParticles
  for spring in springs p do
    ParticleSpringSystem.validateSpring? p.particleCount spring
    let n ← springDirection? q spring
    let block := dampingDerivativeBlock p p.dt n
    h := addBlock h spring.particle0 spring.particle0 block
    h := addBlock h spring.particle1 spring.particle1 block
    if !(pinned.contains spring.particle0) && !(pinned.contains spring.particle1) then
      h := setBlock h spring.particle1 spring.particle0 block (-1.0)
      h := setBlock h spring.particle0 spring.particle1 block (-1.0)
  for particle in pinned do
    if particle >= p.particleCount then
      .error s!"pinned particle index {particle} >= particle count {p.particleCount}"
    h := setIdentityBlock h particle
  pure h

private def clothDiscreteSolverMove : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[clothDiscreteUpdateVertex]
    reads := #[clothGeometryStateInputVertex]
    writes := #[clothDiscreteUpdateVertex]
    exactness := .exact
    label := "ClothSpringModel discrete update: explicit elastic/gravity plus exact dense implicit damping solve"
  }

structure ClothDiscreteStep where
  settings : ClothDiscreteSolverSettings
  linearSystemSize : Nat
  maxIterations : Nat
  accuracy : Float
  vHat : Array Float
  elasticForces : Array Float
  dampingForces : Array Float
  dampingMatrix : Array (Array Float)
  dampingDv : Array Float
  nextState : ClothState
  move : SkeletonMove := clothDiscreteSolverMove
  deriving Repr, Inhabited

def massMatrix (p : ClothParams) : Array (Array Float) :=
  let m := p.physical.massPerParticle p.particleCount
  FloatMatrix.diagonal (Array.replicate p.positionDim m)

def gravityGeneralizedForce (p : ClothParams) : Array Float :=
  let m := p.physical.massPerParticle p.particleCount
  FloatArray.scale m
    (ParticleSpringSystem.gravityAccelerations p.particleCount p.physical.gravityZ)

def primitiveGeneralizedForce?
    (p : ClothParams) (x : ClothState) :
    Except String (Array Float × ParticleSpringForceResult) := do
  p.validate?
  x.validate? p
  let springForces ← ParticleSpringSystem.accumulateForces?
    p.particleCount p.physical (springs p) x.q x.v
  pure
    (applyDirichletBoundary p
      (FloatArray.add springForces.forces (gravityGeneralizedForce p)),
      springForces)

def fullPhysicsPrimitives?
    (p : ClothParams) (x : ClothState)
    (label : String := "mass-spring cloth primitive full physics") :
    Except String (FullPhysicsPrimitives × ParticleSpringForceResult) := do
  let (primitiveForce, springForces) ← primitiveGeneralizedForce? p x
  pure ({
    massMatrix := massMatrix p
    qdot := applyDirichletBoundary p x.v
    actuationForces := Array.replicate p.positionDim 0.0
    generalizedForceContributions := #[
      GeneralizedForceContribution.ofForce
        primitiveForce
        "mass-spring cloth spring+damping+gravity generalized force"
        "ParticleSpringSystem"
    ]
    biasForces := Array.replicate p.positionDim 0.0
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := label
  }, springForces)

def fullPhysicsPrimitiveProvider
    (p : ClothParams := params)
    (label : String := "mass-spring cloth full physics provider") :
    FullPhysicsPrimitiveProvider ClothState :=
  {
    label := label
    primitivesAt? := fun x => do
      let (primitives, _) ← fullPhysicsPrimitives? p x label
      pure primitives
  }

def solveFullPhysics?
    (p : ClothParams) (x : ClothState)
    (intervalVertex : VertexId := clothFullPhysicsIntervalVertex)
    (label : String := "mass-spring cloth primitive full physics") :
    Except String (FullPhysicsResult × ParticleSpringForceResult) := do
  let (primitives, springForces) ← fullPhysicsPrimitives? p x label
  let fullPhysics ← primitives.solve? intervalVertex
  pure (fullPhysics, springForces)

def discreteStep?
    (p : ClothParams) (x : ClothState)
    (settings : ClothDiscreteSolverSettings := {}) :
    Except String ClothDiscreteStep := do
  settings.validate? p
  let (vHat, elastic, _) ← explicitVelocityPrediction? p x
  let (damping, _) ← dampingForces? p x.q vHat
  let damping := applyDirichletBoundary p damping
  let h ← implicitDampingMatrix? p x.q
  let dv ← DenseLinearAlgebra.solveLinear? h (FloatArray.scale p.dt damping)
  let nextV := applyDirichletBoundary p (FloatArray.add vHat dv)
  let nextQ := FloatArray.addScaled x.q nextV p.dt
  pure {
    settings := settings
    linearSystemSize := p.positionDim
    maxIterations := settings.effectiveMaxIterations p
    accuracy := settings.accuracy
    vHat := vHat
    elasticForces := elastic
    dampingForces := damping
    dampingMatrix := h
    dampingDv := dv
    nextState := { q := nextQ, v := nextV }
  }

def derivative? (p : ClothParams) (x : ClothState) :
    Except String ClothDerivative := do
  let (fullPhysics, springForces) ← solveFullPhysics? p x
  pure {
    qdot := fullPhysics.derivative.qdot
    vdot := fullPhysics.derivative.vdot
    springForces := springForces
  }

def addScaled (xs dxs : Array Float) (dt : Float) : Array Float :=
  FloatArray.addScaled xs dxs dt

def eulerStep? (p : ClothParams) (dt : Float) (x : ClothState) :
    Except String (ClothState × ClothDerivative) := do
  let dx ← derivative? p x
  pure ({
    q := addScaled x.q dx.qdot dt
    v := addScaled x.v dx.vdot dt
  }, dx)

def simulateSteps? (p : ClothParams) (steps : Nat)
    (x0 : ClothState := defaultState p) : Except String ClothState := do
  let mut x := x0
  for _ in [:steps] do
    let (next, _) ← eulerStep? p p.traceStepSize x
    x := next
  pure x

def acceptedSegment (p : ClothParams) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := p.traceStepSize
    tAfter := p.traceStepSize
    label := "mass-spring cloth continuous interval"
  }

structure ClothResult where
  references : Array DrakeReference
  params : ClothParams
  initialState : ClothState
  derivative : ClothDerivative
  fullPhysics : FullPhysicsResult
  discreteStep : ClothDiscreteStep
  oneStepState : ClothState
  rolloutState : ClothState
  geometry : ClothSpringModelGeometryResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildEndToEnd? (p : ClothParams := params) : Except String ClothResult := do
  let x0 := defaultState p
  let dx ← derivative? p x0
  let (fullPhysics, _) ← solveFullPhysics? p x0
  let discrete ← discreteStep? p x0
  let (x1, _) ← eulerStep? p p.traceStepSize x0
  let rollout ← simulateSteps? p 2 x0
  let geometry ← buildClothSpringModelGeometry? p x0
  let trace :=
    DynamicEventTrace.empty
      |>.push (.interval (acceptedSegment p))
  trace.validate?
  pure {
    references := drakeReferences
    params := p
    initialState := x0
    derivative := dx
    fullPhysics := fullPhysics
    discreteStep := discrete
    oneStepState := x1
    rolloutState := rollout
    geometry := geometry
    trace := trace
    moves := trace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
  }

end Tyr.EventSkeleton.Examples.MassSpringCloth
