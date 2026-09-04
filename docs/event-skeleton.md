# Event-skeleton simulation

`Tyr/EventSkeleton/` is a differentiation surface for hybrid, stochastic, and branching dynamical systems: continuous ODE intervals interrupted by guard-localized events (impacts, resets), discrete marks (categorical choices such as contact modes), and multi-child branches. It records what happened on a forward execution as a typed event tape, projects that tape into a sparse "skeleton graph" of elimination moves, and provides pure-Lean numeric kernels that apply the corresponding reverse-mode updates (saltation adjoints, reset transposes, mark marginalization, branch aggregation).

Use it when you need gradients — or at least a structured elimination plan — through event discontinuities, rather than through smooth tensor computation. It is deliberately independent of `Tyr.AD.Elim` (the AlphaGrad vertex eliminator, see [autodiff.md](autodiff.md)) and does not touch libtorch, the FFI, or the tensor shape machinery: all numerics are `Array Float` / `Array (Array Float)` (`Tyr/EventSkeleton/Core.lean:1-11`).

Everything lives in `namespace Tyr.EventSkeleton`; the umbrella module is `Tyr.EventSkeleton` (`Tyr/EventSkeleton.lean`), which re-exports the 12 core modules and 33 example modules. The top-level `Tyr.lean` does **not** re-export it — import `Tyr.EventSkeleton` (or the specific submodule) directly.

## Architecture and main abstractions

The design is a move vocabulary, a set of local elimination kernels, and a runtime tape that connects them.

### Skeleton IR (`Tyr/EventSkeleton/Core.lean`)

A computation is described as a `SkeletonGraph` — vertices plus a proposed elimination schedule:

```lean
abbrev VertexId := Nat
abbrev SegmentId := Nat
abbrev EventId := Nat

structure SkeletonVertex where
  id : VertexId
  kind : SkeletonVertexKind
  label : String := ""

structure SkeletonMove where
  kind : SkeletonMoveKind
  targets : Array VertexId := #[]
  reads : Array VertexId := #[]
  writes : Array VertexId := #[]
  cost : MoveCost := {}
  exactness : MoveExactness := kind.defaultExactness
  label : String := ""

structure SkeletonGraph where
  vertices : Array SkeletonVertex := #[]
  moves : Array SkeletonMove := #[]
```

`SkeletonMoveKind` (`Core.lean:47`) has 12 constructors: `intervalAdjoint`, `saltationTime`, `resetTranspose`, `checkpointBoundary`, `rematerializeSegment`, `freezeControl`, `clockedUpdate`, `localSchurBlock`, `markMarginalize`, `markScoreSample`, `branchAggregate`, `learnedComplement`. Each move carries a `MoveExactness` (`.exact | .unbiasedEstimator | .learnedApproximation | .controlledApproximation`), defaulted per kind by `SkeletonMoveKind.defaultExactness`, and a unitless `MoveCost` (`work`/`memory`/`variance`/`bias`). `localSchurBlock` is the designated future bridge back to the AlphaGrad eliminator.

### Event kernels

All kernels are partial via `Except String` and operate on dense arrays.

- **Saltation** (`Tyr/EventSkeleton/Saltation.lean`) — reverse mode through one deterministic hybrid event without forming the dense saltation matrix. `SaltationData` stores the reset Jacobian `R_x`, guard gradient `g_x`, and the precomputed `a = f⁺ − R_x f⁻ − R_t`, `γ = g_t + g_x·f⁻`, `β`. The reverse update is `p⁻ = c_x + R_xᵀ p⁺ + g_xᵀ α` with `α = (aᵀp⁺ − β)/γ`; a zero `γ` is rejected as "not transverse" (`validateGamma`, `Saltation.lean:126`).
- **Marks** (`Tyr/EventSkeleton/Mark.lean`) — `CategoricalMarkData.exactEliminate?` marginalizes over explicit marks: `V⁻ = Σ_y π_y Q_y` plus the probability-message term `(Dπ)ᵀQ`. `SampledMarkData.eliminate` keeps one sampled mark live and adds the score-function term `(Q_y − b) ∇ log π_y`. `EventMessage` is the per-outcome value/adjoint payload. `simplexHitCotangent?` computes the hit-simplex cotangent `Q − 1·(vᵀQ)/(1ᵀv)`.
- **Branches** (`Tyr/EventSkeleton/Branch.lean`) — `BranchEventData.aggregate?` generalizes the saltation update to weighted children: `p⁻ = c_x + Σ_j w_j R_{j,x}ᵀ p_j⁺ + g_xᵀ α` with `α = (Σ_j w_j a_jᵀ p_j⁺ − β)/γ`.
- **Intervals** (`Tyr/EventSkeleton/Interval.lean`) — `AcceptedStepSegment` records one accepted integration attempt (`tStart`, `tAttempt`, `tAfter`, jump flags); `localizedByEvent` is true when the segment ended before the attempted time, i.e. an event root was localized. `planForAcceptedSegment` emits an `intervalAdjoint` move (plus an optional `checkpointBoundary`), and `graphFromAcceptedSegments` builds a `SkeletonGraph` from a run of segments.

### Runtime tape (`Tyr/EventSkeleton/Trace.lean`)

`DynamicEventTrace` is the forward-pass record; reverse mode reads it to know which kernel and which exactness class applies:

```lean
inductive EventTraceEntry where
  | interval (segment : AcceptedStepSegment)
  | clockedUpdate (updateVertex : VertexId) (data : ClockedUpdateData)
  | saltation (eventVertex : VertexId) (data : SaltationData)
  | categoricalMark (markVertex : VertexId) (support : RuntimeSupport) (data : CategoricalMarkData)
  | sampledMark (markVertex : VertexId) (support : RuntimeSupport) (data : SampledMarkData)
  | branch (branchVertex : VertexId) (support : RuntimeSupport) (data : BranchEventData)
```

`RuntimeSupport` records *how* a dynamic support set was produced via a `SupportPolicy` (`fullSupport | sampled | topK | threshold | learnedTail | deterministicPick`); the policy maps to a `MoveExactness` (`SupportPolicy.defaultExactness`, `Trace.lean:29`), so e.g. a `topK` support yields a `controlledApproximation` move. `trace.validate?` enforces support/payload coherence (a `categoricalMark` entry cannot use sampled support; payload sizes must match the selected support), and `trace.moves` projects every entry to its `SkeletonMove`s. `ClockedUpdateData` covers periodic discrete updates: deterministic resets with a prescribed clock, so there is a VJP but no saltation timing scalar.

### Physics and contact stack

Small dense numeric primitives, explicitly not a high-performance backend (`Physics.lean:4-12`):

- `Tyr/EventSkeleton/Physics.lean` — `DenseLinearAlgebra.solveLinear?` (Gauss–Jordan with pivoting), `LinearComplementarityProblem.solveByActiveSet?` (dense LCP by active-set enumeration — exponential, for small exact examples only), `CoulombFriction` (harmonic-mean pair combination), `VelocityProjection.project?` (mass-metric impulse projection via the Delassus operator), `NormalContactLcpProblem.solve?` (acceleration-level unilateral contact), `LinearBushingRollPitchYaw.evaluate?`, and `ParticleSpringSystem` (cloth spring graphs).
- `Tyr/EventSkeleton/Contact.lean` — the pipeline is `ContactCandidate` (one primitive contact view: signed distance, normal/tangent velocities, Jacobian rows in generalized-velocity coordinates) → `ContactSupport` (the selected active set, via `selectWithPolicy` driven by a `SupportPolicy`) → force scalars (`CompliantContactModel.forcesForSupport?`: penalty normal force with Coulomb-clipped damping tangents, or precomputed forces) → generalized force `Jᵀf`. `ContactCandidateSet` / `PackedContactCandidateBatch` are provider output formats; `HydroelasticContactPatch` / `HydroelasticPatchSupport` model hydroelastic surface patches with an `equivalentContactCandidate` projection.
- `Tyr/EventSkeleton/Manipulator.lean` — `ManipulatorEquation.solve?` solves `M(q)v̇ = τ − bias` densely. Around it: coupler constraints, actuator models, joint torque/PID controllers, `DiscreteContactApproximation` (`sap | similar | lagged`), `FullMultibodyPlantModel` / `FullMultibodyPlantStep` (parser-plant metadata), `BilateralConstraintPrimitive`, and the capstone composition `FullPhysicsPrimitives → FullPhysicsEquation → FullPhysicsResult`, plus the state-dependent boundary `FullPhysicsPrimitiveProvider (State : Type)` with `solveAt?`.

`FullPhysicsEquation.solve? (intervalVertex : VertexId := 0) : Except String FullPhysicsResult` (`Manipulator.lean:1193`) returns the derivative and two `SkeletonMove`s: `supportMove` (`contact-support-selection:…`, kind `markMarginalize`, exactness inherited from the support policy) and `move` (`full-physics-step:…`, kind `intervalAdjoint`).

### Geometry and scenario boundaries

- `Tyr/EventSkeleton/SceneGraph.lean` — a Drake-SceneGraph-style provider boundary. `SceneGraphProvider` owns sources/frames/geometries/roles (`illustration | perception | proximity`); `SceneGeometryShape` covers sphere/half-space/box/cylinder/capsule/model/mesh/convex; `ScenePose3` is translation plus axis-angle. `SceneContactQueryResult.solverContactCandidateSet?` projects hydroelastic patches to `ContactCandidate`s and *rejects* point-pair-only results, because point pairs lack generalized-velocity Jacobian rows (`SceneGraph.lean:489`). `sphereHalfSpaceContactCandidate?` is the narrow built-in query.
- `Tyr/EventSkeleton/HardwareSim.lean` — a typed `HardwareScenario` (LCM buses, drivers, cameras, model directives, initial positions) compiled by `HardwareScenario.plan?` into a `HardwareSimulationPlan`: an ordered list of `HardwareSetupStep`s plus a hand-built `SkeletonGraph` of the simulator construction (`HardwareSim.lean:473`).
- `Tyr/EventSkeleton/NamedVector.lean` — `NamedVectorBoundary` records Drake `BasicVector` coordinate order, defaults, bounds, and source paths for hand-written vector classes.

### DiffEq bridge

`Tyr/DiffEq/EventSkeleton.lean` (namespace `torch.DiffEq.EventSkeletonBridge`) is the read-only projection from the DiffEq solve loop: `acceptedSegmentsFromSolveLoopState` turns `SolveLoopState` attempts into `AcceptedStepSegment`s, `graphFromSolveLoopState` builds the `SkeletonGraph`, and `summaryFromSolveLoopState` counts localized segments and jump-flag crossings. The core EventSkeleton modules do not import DiffEq; only the examples and this bridge do. See [diffeq.md](diffeq.md) for the solver side.

### Examples and URDF integration

`Tyr/EventSkeleton/Examples/` contains 33 modules porting Drake `examples/` (bouncing ball, rimless wheel, compass gait, iiwa, Allegro hand, hydroelastics, Atlas, …). Each records its provenance in a machine-checkable `DrakeReference` structure (`path` + `concept` pointing into `../drake/examples/...`). `Examples/UrdfContact.lean` consumes a real URDF at compile time:

```lean
urdf_type_provider "Tyr/EventSkeleton/Examples/contact_probe.urdf" as ContactProbeUrdf
```

via the `LeanUrdfTypeProvider` dependency. Note the build wiring: `lakefile.lean:261` declares `require LeanUrdfTypeProvider from "../lean-urdf-typeprovider"` as a **sibling path** dependency, so a standalone checkout of `tyr` will not build the URDF example or its runner (`lean_exe RunUrdfContactExample`, `lakefile.lean:702`). See [ffi-and-build.md](ffi-and-build.md).

## Key APIs

### Skeleton graph

| Declaration | Signature / role |
|---|---|
| `SkeletonGraph.empty` | `SkeletonGraph` |
| `SkeletonGraph.addVertex` / `addMove` | append a vertex / move (no duplicate-ID check) |
| `SkeletonGraph.containsMoveKind` | `SkeletonGraph → SkeletonMoveKind → Bool` |
| `SkeletonGraph.totalCost` | sum of `MoveCost.add` over moves |
| `SkeletonMove.exact` | constructor forcing `exactness := .exact` |

### Saltation (`SaltationData`)

| Declaration | Signature |
|---|---|
| `SaltationData.mkFromFields` | `(resetJac : Array (Array Float)) (guardGrad : Array Float) (fMinus fPlus : Array Float) (resetTime := #[]) (guardTime := 0.0) (beta := 0.0) … → SaltationData` — computes `a` and `γ` from the vector fields |
| `SaltationData.timingAdjoint?` | `SaltationData → (pPlus : Array Float) → Except String Float` |
| `SaltationData.reverseState?` | `SaltationData → (pPlus : Array Float) → Except String (Array Float)` — `c_x + R_xᵀp⁺ + g_xᵀα` |
| `SaltationData.reverseTheta?` | same shape, for parameter cotangents (`R_θ`, `g_θ`, `c_θ`) |
| `SaltationData.saltationMatrix?` | dense `S = R_x + a g_x / γ`, mainly for tests |
| `SaltationData.saltationTimeMove` / `resetTransposeMove` | `VertexId → SkeletonMove` |

### Marks and branches

| Declaration | Signature |
|---|---|
| `CategoricalMarkData.exactEliminate?` | `CategoricalMarkData → Except String EventMessage` |
| `SampledMarkData.eliminate` | `SampledMarkData → EventMessage` (score-function estimator) |
| `simplexHitCotangent?` | `(velocity values : Array Float) → Except String (Array Float)` |
| `BranchEventData.aggregate?` | `BranchEventData → Except String BranchAggregateResult` |
| `EventMessage.add` / `EventMessage.scale` | message algebra used by all kernels |

### Tape and intervals

| Declaration | Signature |
|---|---|
| `DynamicEventTrace.empty` / `push` | build the tape on the forward pass |
| `DynamicEventTrace.validate?` | `DynamicEventTrace → Except String Unit` |
| `DynamicEventTrace.moves` | `DynamicEventTrace → Array SkeletonMove` |
| `planForAcceptedSegment` | `AcceptedStepSegment → (checkpoint : Bool := true) → Array SkeletonMove` |
| `graphFromAcceptedSegments` | `Array AcceptedStepSegment → (checkpoint : Bool := true) → SkeletonGraph` |
| `RuntimeSupport.full` / `sampled` / `topK` | support constructors |

### Contact and physics

| Declaration | Signature |
|---|---|
| `ContactCandidate.classify` | `(distanceTol tangentVelocityTol : Float) → ContactCandidate → ContactMode` (`separated / impacting / sticking / sliding`) |
| `ContactSupport.selectWithPolicy` | `(policy : SupportPolicy) → (candidates : Array ContactCandidate) → (label : String := "") → ContactSupport` |
| `ContactSupport.toRuntimeSupport?` | map selected local indices to stable candidate IDs for the trace |
| `ContactCandidateSet.selectWithPolicy` / `selectByDistance` / `selectClosestK` | set-level selection entry points |
| `CompliantContactModel.forcesForSupport?` | `CompliantContactModel → ContactSupport → Except String (Array ContactForceScalars)` |
| `ContactCandidate.generalizedForce` / `generalizedForce3D` | scalar forces → `Jᵀf` |
| `DenseLinearAlgebra.solveLinear?` | `(a : Array (Array Float)) (b : Array Float) → Except String (Array Float)` |
| `LinearComplementarityProblem.solveByActiveSet?` | `(m : Array (Array Float)) (q : Array Float) (tol := 1.0e-8) → Except String LinearComplementaritySolution` |
| `VelocityProjection.project?` | `(mass jac : Array (Array Float)) (vPre : Array Float) (target? := none) → Except String VelocityProjection` |
| `NormalContactLcpProblem.solve?` | `NormalContactLcpProblem → (tol := 1.0e-8) → Except String NormalContactLcpResult` |
| `ManipulatorEquation.solve?` | `ManipulatorEquation → Except String ManipulatorDerivative` |
| `FullPhysicsPrimitives.equation?` / `FullPhysicsPrimitives.solve?` | assemble / solve one full-physics step |
| `FullPhysicsPrimitiveProvider.solveAt?` | `provider → (state : State) → (intervalVertex : VertexId := 0) → Except String FullPhysicsResult` |
| `SceneContactQueryResult.solverContactCandidateSet?` / `solverContactSupport?` | SceneGraph → contact primitives |
| `sphereHalfSpaceContactCandidate?` | built-in sphere/half-space query (`SceneGraph.lean:558`) |
| `HardwareScenario.plan?` | `HardwareScenario → (graphvizRequested : Bool := false) → Except String HardwareSimulationPlan` |

Caveats worth knowing before building on these: `DenseLinearAlgebra.solveUnchecked` (`Physics.lean:115`) and `ManipulatorEquation.solveUnchecked` (`Manipulator.lean:770`) silently return zero vectors on solver failure — prefer the `?` variants. `solveByActiveSet?` enumerates all `2ⁿ` active subsets with no size guard. `SkeletonGraph.addVertex` does not check for duplicate vertex IDs; examples hand-assign ID ranges.

## Usage example

Reconstructed example (from `Tyr/EventSkeleton/Examples/UrdfContact.lean` and `Examples/EventSkeleton/RunUrdfContactExample.lean`, the only executable call site; run with `lake exe RunUrdfContactExample`):

```lean
import Tyr.EventSkeleton.Examples.UrdfContact

open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.UrdfContact

-- Saltation data for the impact event: reset Jacobian, guard gradient, and the
-- pre/post vector fields; `mkFromFields` computes `a` and `gamma`.
#check contactSaltationData
-- SaltationData.mkFromFields contactResetJac contactGuardGrad
--   (freeFlightVectorField preImpactState) (freeFlightVectorField postImpactState)
--   (resetTheta := contactResetTheta preImpactState)

-- Runtime tape: accepted ODE segment + impact event + dynamic contact-mode mark.
def trace : DynamicEventTrace :=
  DynamicEventTrace.empty
    |>.push (.interval acceptedContactSegment)
    |>.push (.saltation contactEventVertex contactSaltationData)
    |>.push (.categoricalMark contactModeVertex contactModeSupport contactModeMarkData)

def run : IO Unit := do
  match buildEndToEnd? with
  | .error msg => IO.eprintln msg
  | .ok result =>
      -- reverse saltation updates through the impact
      IO.println s!"saltation alpha = {result.saltationAlpha}"
      IO.println s!"p- = {result.preImpactAdjoint}"          -- reverseState?
      IO.println s!"dL/d restitution = {result.restitutionGrad}" -- reverseTheta?
      -- exact elimination of the contact-mode mark
      IO.println s!"mark value = {result.markMessage.value}"
      -- projected elimination schedule (trace moves + full-physics moves)
      for m in result.moves do
        IO.println s!"{m.label} : {reprStr m.exactness}"
```

`buildEndToEnd?` (`UrdfContact.lean:282`) validates the URDF tree and the trace, applies `timingAdjoint?` / `reverseState?` / `reverseTheta?` with a terminal adjoint, eliminates the categorical mark with `CategoricalMarkData.exactEliminate?`, and solves one compliant-contact full-physics step through `FullPhysicsEquation.solve?`.

The forward-simulation pattern lives in `Tyr/EventSkeleton/Examples/BouncingBall.lean`: define a state/params pair, an `ODETerm` and a guard `EventSpec` (`BouncingBall.lean:230-242`), then run `simulate? (p : BallParams := params) (tFinal : Float := 10.0) (x0 : BallState := initialState p) (maxImpacts : Nat := 128) : Except String SimulationResult` (`:484`). Each impact localizes an `AcceptedStepSegment` and pushes `.interval` + `.saltation` entries onto the trace; the returned `SimulationResult.moves` is the projected schedule.

## Related guides

- [diffeq.md](diffeq.md) — ODE solvers, `EventSpec`, and the solve loop the skeleton segments are projected from.
- [autodiff.md](autodiff.md) — the other AD surfaces, including the `Tyr.AD.Elim` eliminator that `localSchurBlock` is meant to bridge to.
- [ffi-and-build.md](ffi-and-build.md) — Lake wiring, including the sibling-path `LeanUrdfTypeProvider` dependency.
- [examples-and-testing.md](examples-and-testing.md) — how the `Tests/TestEventSkeleton*.lean` suites are registered and run.

Exhaustive per-symbol documentation for every structure and definition mentioned here is generated by doc-gen4 (see `docbuild/`); this chapter is a guide, not a reference.
