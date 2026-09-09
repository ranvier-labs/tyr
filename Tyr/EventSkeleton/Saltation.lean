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

/-- Maximum absolute difference for equally sized finite vectors. Invalid shapes
or nonfinite inputs return positive infinity so tolerance checks cannot pass. -/
def maxAbsDiff (xs ys : Array Float) : Float := Id.run do
  if xs.size != ys.size then
    return 1.0 / 0.0
  let mut acc := 0.0
  for i in [:xs.size] do
    if !xs[i]!.isFinite || !ys[i]!.isFinite then
      return 1.0 / 0.0
    let d := Float.abs (xs[i]! - ys[i]!)
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

/-- Unchecked algebraic constructor for statically known field shapes. Use
`mkFromFields?` to validate fields received at a dynamic input boundary. -/
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
  if !data.gamma.isFinite then
    .error "saltation gamma must be finite"
  else if data.gamma == 0.0 then
    .error "saltation event is not transverse: gamma is zero"
  else
    .ok ()

private def validateFiniteVector (name : String) (xs : Array Float) : Except String Unit := do
  for x in xs do
    if !x.isFinite then
      throw s!"saltation {name} must contain only finite values"

private def validateVector (name : String) (n : Nat) (xs : Array Float)
    (optional : Bool := false) : Except String Unit := do
  if optional && xs.isEmpty then return ()
  if xs.size != n then
    throw s!"saltation {name} has size {xs.size}, expected {n}"
  validateFiniteVector name xs

private def validateMatrix (name : String) (rows cols : Nat)
    (matrix : Array (Array Float)) (optional : Bool := false) : Except String Unit := do
  if optional && matrix.isEmpty then return ()
  if matrix.size != rows then
    throw s!"saltation {name} has {matrix.size} rows, expected {rows}"
  for i in [:matrix.size] do
    validateVector s!"{name} row {i}" cols matrix[i]!

/-- Validate state and parameter dimensions. Empty optional cost/parameter terms
denote zero; nonempty terms must agree on their full dimensions. -/
def validate (data : SaltationData) : Except String Unit := do
  data.validateGamma
  if !data.beta.isFinite then throw "saltation beta must be finite"
  let inputDim := data.guardGrad.size
  let outputDim := data.a.size
  let thetaDim := Nat.max (FloatMatrix.colCount data.resetTheta)
    (Nat.max data.guardTheta.size data.costThetaGrad.size)
  validateFiniteVector "guardGrad" data.guardGrad
  validateFiniteVector "a" data.a
  validateMatrix "resetJac" outputDim inputDim data.resetJac
  validateVector "costStateGrad" inputDim data.costStateGrad true
  validateMatrix "resetTheta" outputDim thetaDim data.resetTheta true
  validateVector "guardTheta" thetaDim data.guardTheta true
  validateVector "costThetaGrad" thetaDim data.costThetaGrad true

/-- Checked counterpart of `mkFromFields`, including the pre-contraction vector
fields whose dimensions cannot be recovered from the resulting saltation data. -/
def mkFromFields?
    (resetJac : Array (Array Float))
    (guardGrad : Array Float)
    (fMinus fPlus : Array Float)
    (resetTime : Array Float := #[])
    (guardTime : Float := 0.0)
    (beta : Float := 0.0)
    (costStateGrad : Array Float := #[])
    (resetTheta : Array (Array Float) := #[])
    (guardTheta : Array Float := #[])
    (costThetaGrad : Array Float := #[]) : Except String SaltationData := do
  validateVector "fMinus" guardGrad.size fMinus
  validateVector "fPlus" resetJac.size fPlus
  validateVector "resetTime" resetJac.size resetTime true
  if !guardTime.isFinite then throw "saltation guardTime must be finite"
  let data := mkFromFields resetJac guardGrad fMinus fPlus resetTime guardTime beta
    costStateGrad resetTheta guardTheta costThetaGrad
  data.validate
  return data

private def validateCotangent (data : SaltationData) (pPlus : Array Float) : Except String Unit := do
  data.validate
  validateVector "pPlus" data.a.size pPlus

def timingAdjoint? (data : SaltationData) (pPlus : Array Float) : Except String Float := do
  validateCotangent data pPlus
  let alpha := (FloatArray.dot data.a pPlus - data.beta) / data.gamma
  if !alpha.isFinite then throw "saltation timing adjoint is not finite"
  return alpha

/-- Reverse state update `c_x + R_x^T p^+ + g_x^T alpha`. -/
def reverseState? (data : SaltationData) (pPlus : Array Float) :
    Except String (Array Float) := do
  let alpha ← data.timingAdjoint? pPlus
  let resetPart := FloatMatrix.transposeVec data.resetJac pPlus
  let timingPart := FloatArray.scale alpha data.guardGrad
  let result := FloatArray.add data.costStateGrad (FloatArray.add resetPart timingPart)
  validateFiniteVector "reverse state result" result
  return result

/-- Reverse parameter update `c_theta + R_theta^T p^+ + g_theta^T alpha`. -/
def reverseTheta? (data : SaltationData) (pPlus : Array Float) :
    Except String (Array Float) := do
  let alpha ← data.timingAdjoint? pPlus
  let resetPart := FloatMatrix.transposeVec data.resetTheta pPlus
  let timingPart := FloatArray.scale alpha data.guardTheta
  let result := FloatArray.add data.costThetaGrad (FloatArray.add resetPart timingPart)
  validateFiniteVector "reverse theta result" result
  return result

/-- Dense saltation matrix `S = R_x + a g_x / gamma`, useful for tests. -/
def saltationMatrix? (data : SaltationData) : Except String (Array (Array Float)) := do
  data.validate
  let rows := data.a.size
  let cols := data.guardGrad.size
  let mut out : Array (Array Float) := #[]
  for i in [:rows] do
    let mut row : Array Float := #[]
    for j in [:cols] do
      let resetVal := data.resetJac[i]![j]!
      let correction := (data.a[i]! * data.guardGrad[j]!) / data.gamma
      row := row.push (resetVal + correction)
    validateFiniteVector "matrix result" row
    out := out.push row
  return out

def saltationTransposeApply? (data : SaltationData) (pPlus : Array Float) :
    Except String (Array Float) := do
  validateCotangent data pPlus
  let matrix ← data.saltationMatrix?
  let result := FloatMatrix.transposeVec matrix pPlus
  validateFiniteVector "transpose result" result
  return result

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
