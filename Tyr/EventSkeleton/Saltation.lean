import Tyr.EventSkeleton.Core

/-!
# Tyr.EventSkeleton.Saltation

Numeric utilities for deterministic hybrid-event elimination.  The reverse
update avoids forming a dense saltation matrix:

`p^- = c_x + R_x^T p^+ + g_x^T alpha`,
where `alpha = (a^T p^+ - beta) / gamma`.
-/

namespace Tyr.EventSkeleton

namespace FloatArray

def dot (xs ys : Array Float) : Float := Id.run do
  let mut acc := 0.0
  let n := Nat.min xs.size ys.size
  for i in [:n] do
    acc := acc + xs[i]! * ys[i]!
  return acc

def add (xs ys : Array Float) : Array Float := Id.run do
  let n := Nat.max xs.size ys.size
  let mut out : Array Float := #[]
  for i in [:n] do
    out := out.push (xs.getD i 0.0 + ys.getD i 0.0)
  return out

def sub (xs ys : Array Float) : Array Float := Id.run do
  let n := Nat.max xs.size ys.size
  let mut out : Array Float := #[]
  for i in [:n] do
    out := out.push (xs.getD i 0.0 - ys.getD i 0.0)
  return out

def scale (s : Float) (xs : Array Float) : Array Float :=
  xs.map (fun x => s * x)

def addScaled (xs ys : Array Float) (s : Float) : Array Float :=
  add xs (scale s ys)

def maxAbsDiff (xs ys : Array Float) : Float := Id.run do
  let n := Nat.max xs.size ys.size
  let mut acc := 0.0
  for i in [:n] do
    let d := Float.abs (xs.getD i 0.0 - ys.getD i 0.0)
    if d > acc then
      acc := d
  return acc

end FloatArray

namespace FloatMatrix

def colCount (m : Array (Array Float)) : Nat :=
  m.foldl (fun acc row => Nat.max acc row.size) 0

def matVec (m : Array (Array Float)) (x : Array Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for row in m do
    out := out.push (FloatArray.dot row x)
  return out

/-- Apply `m^T p` without explicitly transposing `m`. -/
def transposeVec (m : Array (Array Float)) (p : Array Float) : Array Float := Id.run do
  let cols := colCount m
  let mut out := Array.replicate cols 0.0
  for i in [:m.size] do
    let row := m[i]!
    let pVal := p.getD i 0.0
    for j in [:row.size] do
      let cur := out.getD j 0.0
      out := out.set! j (cur + row[j]! * pVal)
  return out

end FloatMatrix

/--
Data required for one hybrid event reverse update.

`a = f^+ - R_x f^- - R_t` and `gamma = g_t + g_x f^-` are stored directly so
callers can provide them from analytic guards/resets or from local linearized
metadata.
-/
structure SaltationData where
  resetJac : Array (Array Float)
  guardGrad : Array Float
  a : Array Float
  gamma : Float
  beta : Float := 0.0
  costStateGrad : Array Float := #[]
  resetTheta : Array (Array Float) := #[]
  guardTheta : Array Float := #[]
  costThetaGrad : Array Float := #[]
  deriving Repr, Inhabited

namespace SaltationData

def mkFromFields
    (resetJac : Array (Array Float))
    (guardGrad : Array Float)
    (fMinus fPlus : Array Float)
    (resetTime : Array Float := #[])
    (guardTime : Float := 0.0)
    (beta : Float := 0.0)
    (costStateGrad : Array Float := #[])
    (resetTheta : Array (Array Float) := #[])
    (guardTheta : Array Float := #[])
    (costThetaGrad : Array Float := #[]) :
    SaltationData :=
  let resetFlow := FloatMatrix.matVec resetJac fMinus
  {
    resetJac := resetJac
    guardGrad := guardGrad
    a := FloatArray.sub (FloatArray.sub fPlus resetFlow) resetTime
    gamma := guardTime + FloatArray.dot guardGrad fMinus
    beta := beta
    costStateGrad := costStateGrad
    resetTheta := resetTheta
    guardTheta := guardTheta
    costThetaGrad := costThetaGrad
  }

def validateGamma (data : SaltationData) : Except String Unit :=
  if data.gamma == 0.0 then
    .error "saltation event is not transverse: gamma is zero"
  else
    .ok ()

def timingAdjoint? (data : SaltationData) (pPlus : Array Float) : Except String Float := do
  data.validateGamma
  pure ((FloatArray.dot data.a pPlus - data.beta) / data.gamma)

/-- Reverse state update `c_x + R_x^T p^+ + g_x^T alpha`. -/
def reverseState? (data : SaltationData) (pPlus : Array Float) :
    Except String (Array Float) := do
  let alpha ← data.timingAdjoint? pPlus
  let resetPart := FloatMatrix.transposeVec data.resetJac pPlus
  let timingPart := FloatArray.scale alpha data.guardGrad
  pure (FloatArray.add data.costStateGrad (FloatArray.add resetPart timingPart))

/-- Reverse parameter update `c_theta + R_theta^T p^+ + g_theta^T alpha`. -/
def reverseTheta? (data : SaltationData) (pPlus : Array Float) :
    Except String (Array Float) := do
  let alpha ← data.timingAdjoint? pPlus
  let resetPart := FloatMatrix.transposeVec data.resetTheta pPlus
  let timingPart := FloatArray.scale alpha data.guardTheta
  pure (FloatArray.add data.costThetaGrad (FloatArray.add resetPart timingPart))

/-- Dense saltation matrix `S = R_x + a g_x / gamma`, useful for tests. -/
def saltationMatrix? (data : SaltationData) : Except String (Array (Array Float)) := do
  data.validateGamma
  let rows := Nat.max data.resetJac.size data.a.size
  let cols := Nat.max (FloatMatrix.colCount data.resetJac) data.guardGrad.size
  let mut out : Array (Array Float) := #[]
  for i in [:rows] do
    let mut row : Array Float := #[]
    for j in [:cols] do
      let resetVal := (data.resetJac.getD i #[]).getD j 0.0
      let correction := (data.a.getD i 0.0 * data.guardGrad.getD j 0.0) / data.gamma
      row := row.push (resetVal + correction)
    out := out.push row
  return out

def saltationTransposeApply? (data : SaltationData) (pPlus : Array Float) :
    Except String (Array Float) := do
  let matrix ← data.saltationMatrix?
  pure (FloatMatrix.transposeVec matrix pPlus)

def saltationTimeMove (eventVertex : VertexId) : SkeletonMove :=
  {
    kind := .saltationTime
    targets := #[eventVertex]
    label := s!"saltation-time:{eventVertex}"
  }

def resetTransposeMove (eventVertex : VertexId) : SkeletonMove :=
  {
    kind := .resetTranspose
    targets := #[eventVertex]
    label := s!"reset-transpose:{eventVertex}"
  }

end SaltationData

end Tyr.EventSkeleton
