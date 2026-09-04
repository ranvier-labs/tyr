import Tyr.EventSkeleton.Interval
import Tyr.EventSkeleton.Branch

/-!
# Tyr.EventSkeleton.Trace

A minimal runtime event tape for dynamically computed marks and branches.

The local elimination kernels in `Mark` and `Branch` work once a support set is
fixed.  This module records how that support was obtained on the forward pass
so reverse mode can distinguish exact full-support elimination from sampled,
top-k, thresholded, deterministic, or learned-tail fixed-trace moves.
-/

namespace Tyr.EventSkeleton

/-- How a runtime support set was produced. -/
inductive SupportPolicy where
  | fullSupport
  | sampled (sampleIndex : Nat)
  | topK (k : Nat)
  | threshold (threshold : Float)
  | learnedTail (explicitCount : Nat)
  | deterministicPick (index : Nat)
  deriving Repr, BEq, Inhabited

namespace SupportPolicy

def defaultExactness : SupportPolicy → MoveExactness
  | .fullSupport => .exact
  | .sampled _ => .unbiasedEstimator
  | .topK _ => .controlledApproximation
  | .threshold _ => .controlledApproximation
  | .learnedTail _ => .learnedApproximation
  | .deterministicPick _ => .controlledApproximation

def isSampled : SupportPolicy → Bool
  | .sampled _ => true
  | _ => false

def isFullSupport : SupportPolicy → Bool
  | .fullSupport => true
  | _ => false

def isFixedTraceApproximation : SupportPolicy → Bool
  | .topK _ => true
  | .threshold _ => true
  | .deterministicPick _ => true
  | .learnedTail _ => true
  | _ => false

end SupportPolicy

private def natRangeArray (n : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
  for i in [:n] do
    out := out.push i
  return out

/--
The support selected on one concrete forward execution.

`selectedIds` are IDs in the source dynamic candidate space, not just local
array indices.  For top-k token events this might be token IDs; for branch
events it might be scenario or child IDs.
-/
structure RuntimeSupport where
  policy : SupportPolicy
  selectedIds : Array Nat := #[]
  totalCandidates? : Option Nat := none
  label : String := ""
  deriving Repr, Inhabited

namespace RuntimeSupport

def full (count : Nat) : RuntimeSupport :=
  {
    policy := .fullSupport
    selectedIds := natRangeArray count
    totalCandidates? := some count
  }

def sampled (id : Nat) : RuntimeSupport :=
  {
    policy := .sampled id
    selectedIds := #[id]
  }

def topK (ids : Array Nat) (totalCandidates? : Option Nat := none) : RuntimeSupport :=
  {
    policy := .topK ids.size
    selectedIds := ids
    totalCandidates? := totalCandidates?
  }

def exactness (support : RuntimeSupport) : MoveExactness :=
  support.policy.defaultExactness

def retainedCount (support : RuntimeSupport) : Nat :=
  support.selectedIds.size

def validatePayloadSize
    (support : RuntimeSupport)
    (label : String)
    (payloadSize : Nat) : Except String Unit :=
  if support.selectedIds.size == payloadSize then
    .ok ()
  else
    .error s!"{label}: runtime support size {support.selectedIds.size} does not match payload size {payloadSize}"

end RuntimeSupport

/-!
## Clocked discrete updates

Periodic discrete updates are deterministic reset vertices with a prescribed
clock, not state-localized event times.  They therefore need a first-class
event-tape entry: reverse mode applies the update VJP, but there is no
saltation timing scalar.
-/

structure ClockedUpdateData where
  time : Float
  period : Float
  stateBefore : Array Float := #[]
  stateAfter : Array Float := #[]
  updateJac : Array (Array Float) := #[]
  label : String := ""
  deriving Repr, Inhabited

namespace ClockedUpdateData

def validate? (data : ClockedUpdateData) : Except String Unit := do
  if !(Float.isFinite data.time) then
    .error s!"clocked update time must be finite, got {data.time}"
  if !(Float.isFinite data.period) || data.period <= 0.0 then
    .error s!"clocked update period must be positive and finite, got {data.period}"
  if !data.stateBefore.isEmpty && !data.stateAfter.isEmpty &&
      data.stateBefore.size != data.stateAfter.size then
    .error s!"clocked update state sizes differ: before={data.stateBefore.size}, after={data.stateAfter.size}"
  if !data.updateJac.isEmpty then
    let width := data.stateBefore.size
    if data.updateJac.size != data.stateAfter.size then
      .error s!"clocked update Jacobian row count {data.updateJac.size} != state-after size {data.stateAfter.size}"
    for row in data.updateJac do
      if row.size != width then
        .error s!"clocked update Jacobian row width {row.size} != state-before size {width}"

def move (updateVertex : VertexId) : SkeletonMove :=
  {
    kind := .clockedUpdate
    targets := #[updateVertex]
    label := s!"clocked-update:{updateVertex}"
  }

end ClockedUpdateData

private def moveWithSupport (move : SkeletonMove) (support : RuntimeSupport) : SkeletonMove :=
  { move with exactness := support.exactness }

/-- One entry on the runtime event tape. -/
inductive EventTraceEntry where
  | interval (segment : AcceptedStepSegment)
  | clockedUpdate (updateVertex : VertexId) (data : ClockedUpdateData)
  | saltation (eventVertex : VertexId) (data : SaltationData)
  | categoricalMark
      (markVertex : VertexId)
      (support : RuntimeSupport)
      (data : CategoricalMarkData)
  | sampledMark
      (markVertex : VertexId)
      (support : RuntimeSupport)
      (data : SampledMarkData)
  | branch
      (branchVertex : VertexId)
      (support : RuntimeSupport)
      (data : BranchEventData)
  deriving Repr, Inhabited

namespace EventTraceEntry

def support? : EventTraceEntry → Option RuntimeSupport
  | .categoricalMark _ support _ => some support
  | .sampledMark _ support _ => some support
  | .branch _ support _ => some support
  | _ => none

def validate? : EventTraceEntry → Except String Unit
  | .interval _ => .ok ()
  | .clockedUpdate _ data => data.validate?
  | .saltation _ data => data.validateGamma
  | .categoricalMark _ support data => do
      if support.policy.isSampled then
        .error "categoricalMark trace entry cannot use sampled support; use sampledMark"
      else do
        data.validate
        support.validatePayloadSize "categoricalMark" data.messages.size
  | .sampledMark _ support _ => do
      if !support.policy.isSampled then
        .error "sampledMark trace entry requires sampled support"
      else if support.selectedIds.size != 1 then
        .error s!"sampledMark: expected exactly one selected id, got {support.selectedIds.size}"
      else
        .ok ()
  | .branch _ support data => do
      if support.policy.isSampled then
        .error "branch aggregate trace entry cannot use sampled support"
      else do
        data.validate
        support.validatePayloadSize "branch" data.children.size

def moves : EventTraceEntry → Array SkeletonMove
  | .interval segment =>
      planForAcceptedSegment segment
  | .clockedUpdate updateVertex _ =>
      #[ClockedUpdateData.move updateVertex]
  | .saltation eventVertex _ =>
      #[
        SaltationData.saltationTimeMove eventVertex,
        SaltationData.resetTransposeMove eventVertex
      ]
  | .categoricalMark markVertex support _ =>
      #[moveWithSupport (CategoricalMarkData.markMarginalizeMove markVertex) support]
  | .sampledMark markVertex support _ =>
      #[moveWithSupport (SampledMarkData.markScoreSampleMove markVertex) support]
  | .branch branchVertex support _ =>
      #[moveWithSupport (BranchEventData.branchAggregateMove branchVertex) support]

end EventTraceEntry

/-- Runtime tape produced by one forward execution. -/
structure DynamicEventTrace where
  entries : Array EventTraceEntry := #[]
  deriving Repr, Inhabited

namespace DynamicEventTrace

def empty : DynamicEventTrace := {}

def push (trace : DynamicEventTrace) (entry : EventTraceEntry) : DynamicEventTrace :=
  { trace with entries := trace.entries.push entry }

def validate? (trace : DynamicEventTrace) : Except String Unit := do
  for entry in trace.entries do
    entry.validate?

def moves (trace : DynamicEventTrace) : Array SkeletonMove := Id.run do
  let mut out : Array SkeletonMove := #[]
  for entry in trace.entries do
    out := out ++ entry.moves
  return out

def supports (trace : DynamicEventTrace) : Array RuntimeSupport := Id.run do
  let mut out : Array RuntimeSupport := #[]
  for entry in trace.entries do
    match entry.support? with
    | some support => out := out.push support
    | none => pure ()
  return out

end DynamicEventTrace

end Tyr.EventSkeleton
