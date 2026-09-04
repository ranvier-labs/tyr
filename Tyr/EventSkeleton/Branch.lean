import Tyr.EventSkeleton.Mark

/-!
# Tyr.EventSkeleton.Branch

Branch-event aggregation rules for event-skeleton differentiation.

For explicit children with weights `w_j`, the reverse event update is

`p^- = c_x + sum_j w_j R_{j,x}^T p_j^+ + g_x^T alpha`

with

`alpha = (sum_j w_j a_j^T p_j^+ - beta) / gamma`.
-/

namespace Tyr.EventSkeleton

structure BranchChild where
  weight : Float := 1.0
  resetJac : Array (Array Float)
  resetTheta : Array (Array Float) := #[]
  a : Array Float := #[]
  message : EventMessage
  deriving Repr, Inhabited

structure BranchEventData where
  children : Array BranchChild
  guardGrad : Array Float := #[]
  guardTheta : Array Float := #[]
  gamma : Float
  beta : Float := 0.0
  costStateGrad : Array Float := #[]
  costThetaGrad : Array Float := #[]
  deriving Repr, Inhabited

structure BranchAggregateResult where
  value : Float := 0.0
  alpha : Float := 0.0
  stateAdjoint : Array Float := #[]
  thetaGrad : Array Float := #[]
  deriving Repr, Inhabited

namespace BranchEventData

def validate (data : BranchEventData) : Except String Unit :=
  if data.gamma == 0.0 then
    .error "branch event is not transverse: gamma is zero"
  else
    .ok ()

private def weightedValue (children : Array BranchChild) : Float :=
  children.foldl (fun acc child => acc + child.weight * child.message.value) 0.0

private def weightedResetState (children : Array BranchChild) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for child in children do
    let localMsg := FloatMatrix.transposeVec child.resetJac child.message.stateAdjoint
    out := FloatArray.add out (FloatArray.scale child.weight localMsg)
  return out

private def weightedResetTheta (children : Array BranchChild) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for child in children do
    let localMsg := FloatMatrix.transposeVec child.resetTheta child.message.stateAdjoint
    out := FloatArray.add out (FloatArray.scale child.weight localMsg)
  return out

private def weightedTimingNumerator (children : Array BranchChild) : Float :=
  children.foldl
    (fun acc child =>
      acc + child.weight * FloatArray.dot child.a child.message.stateAdjoint)
    0.0

def timingAdjoint? (data : BranchEventData) : Except String Float := do
  data.validate
  pure ((weightedTimingNumerator data.children - data.beta) / data.gamma)

def aggregate? (data : BranchEventData) : Except String BranchAggregateResult := do
  let alpha ← data.timingAdjoint?
  let resetState := weightedResetState data.children
  let resetTheta := weightedResetTheta data.children
  let timingState := FloatArray.scale alpha data.guardGrad
  let timingTheta := FloatArray.scale alpha data.guardTheta
  pure {
    value := weightedValue data.children
    alpha := alpha
    stateAdjoint := FloatArray.add data.costStateGrad
      (FloatArray.add resetState timingState)
    thetaGrad := FloatArray.add data.costThetaGrad
      (FloatArray.add resetTheta timingTheta)
  }

def branchAggregateMove (branchVertex : VertexId) : SkeletonMove :=
  {
    kind := .branchAggregate
    targets := #[branchVertex]
    label := s!"branch-aggregate:{branchVertex}"
  }

end BranchEventData

end Tyr.EventSkeleton
