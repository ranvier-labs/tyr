import Tyr.Modular.MetricFactor
import Tyr.Modular.Compose
import Tyr.Modular.Manifold
import Tyr.Module.Linear
import Tyr.Manifolds.Embedded

/-!
# Riemannian modules

The [`Tyr.Modular.RiemannianModule`] class exposes the local Jacobians
([`Tyr.Modular.LocalLinearization`]) needed for recursive metric-factor
propagation (see `Tyr.Modular.MetricFactor`), with instances for `torch.Linear`
and [`Tyr.Modular.ManifoldLinear`].
-/

namespace Tyr.Modular

open torch
open Tyr.AD

/-- Local linearization data for one module at one forward point. -/
structure LocalLinearization (inDim paramDim outDim : UInt64) where
  applyA : T #[inDim] → T #[outDim]
  applyAT : T #[outDim] → T #[inDim]
  applyB : T #[paramDim] → T #[outDim]
  applyBT : T #[outDim] → T #[paramDim]

/-- Basic diagnostics for one layerwise Riemannian update. -/
structure LayerStepDiagnostics where
  gradientNorm : Float := 0.0
  factorRank : UInt64 := 0
  innerConditionEstimate : Float := 0.0
  residualNorm : Float := 0.0
  updateNorm : Float := 0.0
  deriving Repr, Inhabited

/--
Leaf module that can expose the local Jacobians needed for recursive
metric-factor propagation.
-/
class RiemannianModule (M : Type) (inDim outDim : UInt64) where
  paramDim : UInt64
  Tape : Type
  forwardWithTape : M → T #[inDim] → T #[outDim] × Tape
  localLinearization : M → Tape → LocalLinearization inDim paramDim outDim
  paramMass : M → DiagonalMass paramDim
  applyAmbientUpdate : M → T #[paramDim] → Float → M

namespace RiemannianModule

private def vectorToColumn {n : UInt64} (v : T #[n]) : T #[n, 1] :=
  reshape v #[n, 1]

private def columnToVector {n : UInt64} (v : T #[n, 1]) : T #[n] :=
  reshape v #[n]

private def splitLinearUpdate {inDim outDim : UInt64}
    (update : T #[outDim * inDim + outDim]) : T #[outDim, inDim] × T #[outDim] :=
  let weightSize := outDim * inDim
  let dwVec : T #[weightSize] := data.slice1d update 0 weightSize.toInt64
  let dbVec : T #[outDim] :=
    data.slice1d update weightSize.toInt64 (weightSize + outDim).toInt64
  (reshape dwVec #[outDim, inDim], dbVec)

namespace LocalLinearization

private def rowAt {rows cols : UInt64} (matrix : T #[rows, cols]) (idx : UInt64) : T #[cols] :=
  reshape (data.slice2d matrix idx 1) #[cols]

private def mapRows {rows inDim outDim : UInt64}
    (matrix : T #[rows, inDim])
    (f : T #[inDim] → T #[outDim]) : T #[rows, outDim] :=
  let mapped : Array (T #[outDim]) := Id.run do
    let mut acc : Array (T #[outDim]) := #[]
    for i in [:rows.toNat] do
      acc := acc.push (f (rowAt matrix (UInt64.ofNat i)))
    pure acc
  torch.stack1d mapped

/-- Materialize the dense input Jacobian for debugging/tests. -/
def materializeA {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : T #[outDim, inDim] :=
  let basis : T #[inDim, inDim] := eye inDim
  nn.transpose2d (mapRows basis lin.applyA)

/-- Materialize the dense parameter Jacobian for debugging/tests. -/
def materializeB {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : T #[outDim, paramDim] :=
  let basis : T #[paramDim, paramDim] := eye paramDim
  nn.transpose2d (mapRows basis lin.applyB)

/-- Pull a metric factor back through the input Jacobian using rowwise adjoints. -/
def pullbackInputFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank inDim :=
  { matrix := mapRows L.matrix lin.applyAT }

/-- Pull a metric factor back through the parameter Jacobian using rowwise adjoints. -/
def pullbackParamFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank paramDim :=
  { matrix := mapRows L.matrix lin.applyBT }

end LocalLinearization

private def linearApplyInput {inDim outDim : UInt64}
    (weight : T #[outDim, inDim]) (x : T #[inDim]) : T #[outDim] :=
  columnToVector (nn.mm weight (vectorToColumn x))

private def linearApplyInputAdjoint {inDim outDim : UInt64}
    (weight : T #[outDim, inDim]) (y : T #[outDim]) : T #[inDim] :=
  columnToVector (nn.mm (nn.transpose2d weight) (vectorToColumn y))

private def linearApplyParam {inDim outDim : UInt64}
    (x : T #[inDim])
    (useBias : Bool)
    (update : T #[outDim * inDim + outDim]) : T #[outDim] :=
  let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
  let weightPart := linearApplyInput dW x
  if useBias then
    weightPart + dB
  else
    weightPart

private def linearApplyParamAdjoint {inDim outDim : UInt64}
    (x : T #[inDim])
    (useBias : Bool)
    (y : T #[outDim]) : T #[outDim * inDim + outDim] :=
  let outer : T #[outDim, inDim] := nn.mm (vectorToColumn y) (reshape x #[1, inDim])
  let weightPart : T #[outDim * inDim] := reshape outer #[outDim * inDim]
  let biasPart : T #[outDim] := if useBias then y else zeros #[outDim]
  nn.cat weightPart biasPart 0

private def makeManifoldLeaf [MatrixManifoldCarrier M] {m n : UInt64}
    [Tyr.Modular.ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) m n)]
    (w : M m n) : M m n :=
  MatrixManifoldCarrier.fromAmbientUnchecked
    (autograd.set_requires_grad (autograd.detach (MatrixManifoldCarrier.toMatrix w)) true)

/-- Single leaf backward step: metric solve, parameter update, and propagation. -/
def backwardLeafStep {M : Type} {inDim outDim : UInt64} [RiemannianModule M inDim outDim]
    {rank : UInt64}
    (m : M)
    (tape : RiemannianModule.Tape (M := M) (inDim := inDim) (outDim := outDim))
    (L : MetricFactor rank outDim)
    (lambda : T #[outDim])
    (lr : Float)
    : M × MetricFactor rank inDim × T #[inDim] × LayerStepDiagnostics := by
  let lin := RiemannianModule.localLinearization (M := M) (inDim := inDim) (outDim := outDim) m tape
  let g := lin.applyBT lambda
  let K := LocalLinearization.pullbackParamFactor lin L
  let mass := RiemannianModule.paramMass (M := M) (inDim := inDim) (outDim := outDim) m
  let update := MetricFactor.solveWoodbury mass K g
  let m' := RiemannianModule.applyAmbientUpdate (M := M) (inDim := inDim) (outDim := outDim) m update lr
  let LPrev := LocalLinearization.pullbackInputFactor lin L
  let lambdaPrev := lin.applyAT lambda
  let inner := MetricFactor.woodburyInner mass K
  let innerInv := linalg.inv inner
  let residual := MetricFactor.applyRegularized mass K update - g
  let stats : LayerStepDiagnostics := {
    gradientNorm := linalg.l2Norm g
    factorRank := rank
    innerConditionEstimate := linalg.spectralNorm inner * linalg.spectralNorm innerInv
    residualNorm := linalg.l2Norm residual
    updateNorm := linalg.l2Norm update
  }
  exact (m', LPrev, lambdaPrev, stats)

/-- [`Tyr.Modular.RiemannianModule`] for `torch.Linear` with unit diagonal parameter mass. -/
instance linearRiemannianModule (inDim outDim : UInt64) :
    RiemannianModule (torch.Linear inDim outDim) inDim outDim where
  paramDim := outDim * inDim + outDim
  Tape := T #[inDim]
  forwardWithTape lin x :=
    let x2 : T #[1, inDim] := reshape x #[1, inDim]
    let y2 : T #[1, outDim] :=
      match lin.bias with
      | some b => affine x2 lin.weight b
      | none => linear x2 lin.weight
    (reshape y2 #[outDim], x)
  localLinearization lin x := {
    applyA := linearApplyInput lin.weight
    applyAT := linearApplyInputAdjoint lin.weight
    applyB := linearApplyParam x lin.bias.isSome
    applyBT := linearApplyParamAdjoint x lin.bias.isSome
  }
  paramMass _ := DiagonalMass.ones (outDim * inDim + outDim)
  applyAmbientUpdate lin update lr :=
    let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
    let weight' := autograd.set_requires_grad (autograd.detach (lin.weight - mul_scalar dW lr)) true
    let bias' :=
      match lin.bias with
      | some b =>
        some (autograd.set_requires_grad (autograd.detach (b - mul_scalar dB lr)) true)
      | none => none
    { lin with weight := weight', bias := bias' }

/-- [`Tyr.Modular.RiemannianModule`] for [`Tyr.Modular.ManifoldLinear`] layers. -/
instance manifoldLinearRiemannianModule
    (M : UInt64 → UInt64 → Type)
    [MatrixManifoldCarrier M]
    {inDim outDim : UInt64}
    [Tyr.Modular.ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) outDim inDim)]
    :
    RiemannianModule (ManifoldLinear M inDim outDim) inDim outDim where
  paramDim := outDim * inDim + outDim
  Tape := T #[inDim]
  forwardWithTape lin x :=
    let weight := MatrixManifoldCarrier.toMatrix lin.weight
    let x2 : T #[1, inDim] := reshape x #[1, inDim]
    let y2 : T #[1, outDim] :=
      match lin.bias with
      | some b => affine x2 weight b
      | none => linear x2 weight
    (reshape y2 #[outDim], x)
  localLinearization lin x := {
    applyA := linearApplyInput (MatrixManifoldCarrier.toMatrix lin.weight)
    applyAT := linearApplyInputAdjoint (MatrixManifoldCarrier.toMatrix lin.weight)
    applyB := linearApplyParam x lin.bias.isSome
    applyBT := linearApplyParamAdjoint x lin.bias.isSome
  }
  paramMass _ := DiagonalMass.ones (outDim * inDim + outDim)
  applyAmbientUpdate lin update lr :=
    let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
    let updatedMatrix := MatrixManifoldCarrier.toMatrix lin.weight - mul_scalar dW lr
    let weight' := MatrixManifoldCarrier.project updatedMatrix
    let weight' := makeManifoldLeaf weight'
    let bias' :=
      match lin.bias with
      | some b =>
        some (autograd.set_requires_grad (autograd.detach (b - mul_scalar dB lr)) true)
      | none => none
    { lin with weight := weight', bias := bias' }

end RiemannianModule

namespace LocalLinearization

/-- Materialize the dense input Jacobian for debugging/tests. -/
def materializeA {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : T #[outDim, inDim] :=
  RiemannianModule.LocalLinearization.materializeA lin

/-- Materialize the dense parameter Jacobian for debugging/tests. -/
def materializeB {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : T #[outDim, paramDim] :=
  RiemannianModule.LocalLinearization.materializeB lin

/-- Pull a metric factor back through the input Jacobian using rowwise adjoints. -/
def pullbackInputFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank inDim :=
  RiemannianModule.LocalLinearization.pullbackInputFactor lin L

/-- Pull a metric factor back through the parameter Jacobian using rowwise adjoints. -/
def pullbackParamFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank paramDim :=
  RiemannianModule.LocalLinearization.pullbackParamFactor lin L

end LocalLinearization

/-- Forward tape for a two-layer sequential composition. -/
abbrev SequentialTape
    (M₁ : Type) (inDim midDim : UInt64)
    (M₂ : Type) (outDim : UInt64)
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    :=
  RiemannianModule.Tape (M := M₁) (inDim := inDim) (outDim := midDim) ×
    RiemannianModule.Tape (M := M₂) (inDim := midDim) (outDim := outDim)

/-- Forward pass with tapes for a two-layer sequential composition. -/
def sequentialForwardWithTape
    {M₁ M₂ : Type}
    {inDim midDim outDim : UInt64}
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    (net : Sequential M₁ M₂)
    (x : T #[inDim])
    : T #[outDim] × SequentialTape M₁ inDim midDim M₂ outDim := by
  let (mid, tape₁) :=
    RiemannianModule.forwardWithTape (M := M₁) (inDim := inDim) (outDim := midDim) net.first x
  let (y, tape₂) :=
    RiemannianModule.forwardWithTape (M := M₂) (inDim := midDim) (outDim := outDim) net.second mid
  exact (y, (tape₁, tape₂))

/-- Backward recursion for a two-layer sequential composition. -/
def sequentialBackwardMetricStep
    {M₁ M₂ : Type}
    {rank inDim midDim outDim : UInt64}
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    (net : Sequential M₁ M₂)
    (tape : SequentialTape M₁ inDim midDim M₂ outDim)
    (L : MetricFactor rank outDim)
    (lambda : T #[outDim])
    (lr : Float)
    : Sequential M₁ M₂ × MetricFactor rank inDim × T #[inDim] × Array LayerStepDiagnostics := by
  let (tape₁, tape₂) := tape
  let (second', midFactor, midCotangent, stats₂) :=
    RiemannianModule.backwardLeafStep net.second tape₂ L lambda lr
  let (first', inFactor, inCotangent, stats₁) :=
    RiemannianModule.backwardLeafStep net.first tape₁ midFactor midCotangent lr
  exact ({ first := first', second := second' }, inFactor, inCotangent, #[stats₁, stats₂])

end Tyr.Modular
