import Tyr.EventSkeleton.Trace

/-!
# Drake-Style Witness Contact Example

This is a small, executable end-to-end event-skeleton example modeled after
Drake's hybrid event pattern:

* `../drake/examples/rimless_wheel/rimless_wheel.cc` uses
  `MakeWitnessFunction` to pair a guard with an unrestricted reset.
* `../drake/systems/analysis/simulator.cc` isolates a witness crossing before
  dispatching the triggered event.

The code below does not depend on Drake at build time.  It records the same
kind of skeleton in Lean: an accepted free-flight interval, a witness/contact
saltation event, a dynamically selected contact-mode mark support, and a
dynamically selected branch support.
-/

namespace Tyr.EventSkeleton.Examples.DrakeWitness

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/rimless_wheel/rimless_wheel.cc"
      concept := "MakeWitnessFunction pairs a step guard with an unrestricted reset"
    },
    {
      path := "../drake/examples/compass_gait/compass_gait.cc"
      concept := "foot collision is represented as a witness-triggered reset"
    },
    {
      path := "../drake/systems/analysis/simulator.cc"
      concept := "the simulator isolates witness crossings before event dispatch"
    }
  ]

structure ContactParameters where
  gravity : Float := 9.81
  restitution : Float := 0.5
  deriving Repr, Inhabited

structure ContactState where
  height : Float
  velocity : Float
  deriving Repr, Inhabited

def defaultParameters : ContactParameters := {}

/-- Pre-impact witness state: the guard `height = 0` is transverse. -/
def preImpactState : ContactState :=
  { height := 0.0, velocity := -2.0 }

def freeFlightVectorField (p : ContactParameters) (x : ContactState) : Array Float :=
  #[x.velocity, -p.gravity]

def resetState (p : ContactParameters) (x : ContactState) : ContactState :=
  { height := x.height, velocity := -p.restitution * x.velocity }

def postResetVectorField (p : ContactParameters) (x : ContactState) : Array Float :=
  let y := resetState p x
  #[y.velocity, -p.gravity]

def contactResetJac (p : ContactParameters) : Array (Array Float) :=
  #[#[1.0, 0.0], #[0.0, -p.restitution]]

def contactResetTheta (x : ContactState) : Array (Array Float) :=
  #[#[0.0], #[-x.velocity]]

def contactGuardGrad : Array Float :=
  #[1.0, 0.0]

def contactSaltationData
    (p : ContactParameters := defaultParameters)
    (x : ContactState := preImpactState) :
    SaltationData :=
  SaltationData.mkFromFields
    (contactResetJac p)
    contactGuardGrad
    (freeFlightVectorField p x)
    (postResetVectorField p x)
    (resetTheta := contactResetTheta x)

/-- Downstream cotangent at the post-impact state. -/
def terminalPostImpactAdjoint : Array Float :=
  #[2.0, -0.5]

/--
Runtime contact-mode support.  The selected IDs are source candidate IDs, not
local array indices; this is the important dynamic-support detail.
-/
def contactModeSupport : RuntimeSupport :=
  let support := RuntimeSupport.topK #[101, 7] (some 4)
  { support with label := "top-k runtime contact modes" }

/--
Two retained contact-mode messages.  The probability Jacobian rows correspond
to the retained runtime support IDs in `contactModeSupport.selectedIds`.
-/
def contactModeMarkData : CategoricalMarkData :=
  {
    probs := #[0.7, 0.3]
    messages := #[
      { value := 1.0, stateAdjoint := #[0.1, 0.2] },
      { value := 4.0, stateAdjoint := #[0.5, -0.1] }
    ]
    probStateJac := #[#[0.2, -0.1], #[-0.2, 0.1]]
  }

def branchSupport : RuntimeSupport :=
  {
    policy := .threshold 0.25
    selectedIds := #[3, 8]
    totalCandidates? := some 5
    label := "threshold runtime branch children"
  }

def contactBranchData : BranchEventData :=
  {
    children := #[
      {
        weight := 0.6
        resetJac := #[#[1.0, 0.0], #[0.0, 1.0]]
        a := #[0.5, 0.0]
        message := { value := 5.0, stateAdjoint := #[1.0, 0.0] }
      },
      {
        weight := 0.4
        resetJac := #[#[1.0, 0.0], #[0.0, 2.0]]
        a := #[1.0, -1.0]
        message := { value := 2.0, stateAdjoint := #[0.5, -0.25] }
      }
    ]
    guardGrad := contactGuardGrad
    gamma := -2.0
  }

def acceptedWitnessSegment : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := 0.0
    tAttempt := 0.25
    tAfter := 0.2
    madeJumpAfter := true
    label := "drake-style free-flight witness interval"
  }

def contactEventVertex : VertexId := 100
def contactModeVertex : VertexId := 101
def contactBranchVertex : VertexId := 102

def contactTrace : DynamicEventTrace :=
  DynamicEventTrace.empty
    |>.push (.interval acceptedWitnessSegment)
    |>.push (.saltation contactEventVertex contactSaltationData)
    |>.push (.categoricalMark contactModeVertex contactModeSupport contactModeMarkData)
    |>.push (.branch contactBranchVertex branchSupport contactBranchData)

structure EndToEndResult where
  references : Array DrakeReference
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  saltationAlpha : Float
  preImpactAdjoint : Array Float
  restitutionGrad : Array Float
  markMessage : EventMessage
  branchResult : BranchAggregateResult
  deriving Repr, Inhabited

def buildEndToEnd? : Except String EndToEndResult := do
  contactTrace.validate?
  let saltationAlpha ← contactSaltationData.timingAdjoint? terminalPostImpactAdjoint
  let preImpactAdjoint ← contactSaltationData.reverseState? terminalPostImpactAdjoint
  let restitutionGrad ← contactSaltationData.reverseTheta? terminalPostImpactAdjoint
  let markMessage ← contactModeMarkData.exactEliminate?
  let branchResult ← contactBranchData.aggregate?
  pure {
    references := drakeReferences
    trace := contactTrace
    moves := contactTrace.moves
    saltationAlpha := saltationAlpha
    preImpactAdjoint := preImpactAdjoint
    restitutionGrad := restitutionGrad
    markMessage := markMessage
    branchResult := branchResult
  }

end Tyr.EventSkeleton.Examples.DrakeWitness
