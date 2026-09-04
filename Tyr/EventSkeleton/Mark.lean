import Tyr.EventSkeleton.Saltation

/-!
# Tyr.EventSkeleton.Mark

Categorical mark elimination rules for event-skeleton differentiation.

The exact move marginalizes over explicit marks:

`V^- = sum_y pi_y Q_y`

and propagates both child adjoints and the probability-message term
`(D pi)^T Q`.  The sampled move keeps one mark live and adds the score term
`(Q_y - b) grad log pi_y`.
-/

namespace Tyr.EventSkeleton

/-- Downstream value and adjoint message associated with one event outcome. -/
structure EventMessage where
  value : Float := 0.0
  stateAdjoint : Array Float := #[]
  thetaGrad : Array Float := #[]
  deriving Repr, Inhabited

namespace EventMessage

def zero : EventMessage := {}

def add (a b : EventMessage) : EventMessage :=
  {
    value := a.value + b.value
    stateAdjoint := FloatArray.add a.stateAdjoint b.stateAdjoint
    thetaGrad := FloatArray.add a.thetaGrad b.thetaGrad
  }

def scale (w : Float) (msg : EventMessage) : EventMessage :=
  {
    value := w * msg.value
    stateAdjoint := FloatArray.scale w msg.stateAdjoint
    thetaGrad := FloatArray.scale w msg.thetaGrad
  }

end EventMessage

private def validateEqualSize (label : String) (lhs rhs : Nat) : Except String Unit :=
  if lhs == rhs then
    .ok ()
  else
    .error s!"{label}: size mismatch ({lhs} vs {rhs})"

private def valuesOfMessages (messages : Array EventMessage) : Array Float :=
  messages.map (fun msg => msg.value)

/--
Exact categorical-mark elimination result.

`probStateJac` is the hit-corrected probability Jacobian `D_x pi`, represented
as rows over marks and columns over state coordinates.  `probThetaJac` is
`D_theta pi` in the same row convention.
-/
structure CategoricalMarkData where
  probs : Array Float
  messages : Array EventMessage
  probStateJac : Array (Array Float) := #[]
  probThetaJac : Array (Array Float) := #[]
  deriving Repr, Inhabited

namespace CategoricalMarkData

def validate (data : CategoricalMarkData) : Except String Unit := do
  validateEqualSize "categorical probabilities/messages" data.probs.size data.messages.size
  if data.probStateJac.size != 0 then
    validateEqualSize "categorical probStateJac/messages" data.probStateJac.size data.messages.size
  if data.probThetaJac.size != 0 then
    validateEqualSize "categorical probThetaJac/messages" data.probThetaJac.size data.messages.size

def weightedChildMessage (data : CategoricalMarkData) : EventMessage := Id.run do
  let mut out := EventMessage.zero
  let n := Nat.min data.probs.size data.messages.size
  for i in [:n] do
    out := out.add (EventMessage.scale data.probs[i]! data.messages[i]!)
  return out

def exactEliminate? (data : CategoricalMarkData) : Except String EventMessage := do
  data.validate
  let weighted := data.weightedChildMessage
  let values := valuesOfMessages data.messages
  let probState := FloatMatrix.transposeVec data.probStateJac values
  let probTheta := FloatMatrix.transposeVec data.probThetaJac values
  pure {
    value := weighted.value
    stateAdjoint := FloatArray.add weighted.stateAdjoint probState
    thetaGrad := FloatArray.add weighted.thetaGrad probTheta
  }

def markMarginalizeMove (markVertex : VertexId) : SkeletonMove :=
  {
    kind := .markMarginalize
    targets := #[markVertex]
    label := s!"mark-marginalize:{markVertex}"
  }

end CategoricalMarkData

/-- Data for one sampled categorical mark score update. -/
structure SampledMarkData where
  message : EventMessage
  baseline : Float := 0.0
  logProbStateGrad : Array Float := #[]
  logProbThetaGrad : Array Float := #[]
  deriving Repr, Inhabited

namespace SampledMarkData

def eliminate (data : SampledMarkData) : EventMessage :=
  let scale := data.message.value - data.baseline
  {
    value := data.message.value
    stateAdjoint :=
      FloatArray.add data.message.stateAdjoint
        (FloatArray.scale scale data.logProbStateGrad)
    thetaGrad :=
      FloatArray.add data.message.thetaGrad
        (FloatArray.scale scale data.logProbThetaGrad)
  }

def markScoreSampleMove (markVertex : VertexId) : SkeletonMove :=
  {
    kind := .markScoreSample
    targets := #[markVertex]
    exactness := .unbiasedEstimator
    label := s!"mark-score-sample:{markVertex}"
  }

end SampledMarkData

/--
Hit-simplex cotangent `P_hit^T Q = Q - 1 * (v^T Q)/(1^T v)`.

This is the probability-message into the evidence vector when `pi = r / E`.
-/
def simplexHitCotangent? (velocity values : Array Float) : Except String (Array Float) := do
  validateEqualSize "simplex velocity/values" velocity.size values.size
  let denom := velocity.foldl (fun acc v => acc + v) 0.0
  if denom == 0.0 then
    .error "simplex hit is not transverse: total evidence velocity is zero"
  else
    let baseline := FloatArray.dot velocity values / denom
    pure (values.map (fun q => q - baseline))

end Tyr.EventSkeleton
