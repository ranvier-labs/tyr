import Tyr.Modular.Norm
import Tyr.Modular.Budget
import Tyr.Modular.Compose
import Tyr.Manifolds.Optimizer
import Tyr.Optim.DualOptimizer

/-!
# Manifold-native modules

Layers whose parameters live on matrix or vector manifolds (Stiefel, Grassmann,
orthogonal, hyperbolic), with [`Tyr.Modular.NormedModule`] instances and the
[`Tyr.Modular.ManifoldUpdatable`] class for manifold-native update steps through
products and [`Tyr.Modular.Sequential`] networks.
-/

namespace Tyr.Modular

open torch
open Tyr.AD

/-- Lightweight proposition wrapper for carrying shape-constraint witnesses. -/
class ShapeFact (p : Prop) : Prop where
  out : p

/-- Trivial shape witness. -/
instance : ShapeFact True := ⟨trivial⟩

/-- Matrix-manifold carrier interface for manifold-native linear layers. -/
class MatrixManifoldCarrier (M : UInt64 → UInt64 → Type) where
  /-- Legal shape predicate for this manifold family. -/
  ShapeOK : UInt64 → UInt64 → Prop := fun _ _ => True
  /-- Project an ambient matrix to the manifold. -/
  project : {m n : UInt64} → [ShapeFact (ShapeOK m n)] → T #[m, n] → M m n
  /-- Convert manifold value to its ambient matrix representation. -/
  toMatrix : {m n : UInt64} → [ShapeFact (ShapeOK m n)] → M m n → T #[m, n]
  /-- Wrap an already-valid ambient matrix without re-projecting. -/
  fromAmbientUnchecked : {m n : UInt64} → [ShapeFact (ShapeOK m n)] → T #[m, n] → M m n
  /-- Take one geometry-aware step and retract on-manifold. -/
  dualMapStep : {m n : UInt64} → [ShapeFact (ShapeOK m n)] → M m n → T #[m, n] → Float → M m n

/-- Reflexive equality witness for `UInt64` shapes. -/
instance factEqReflUInt64 (n : UInt64) : ShapeFact (n = n) := ⟨rfl⟩

/-- Stiefel-manifold carrier; steps project the gradient to the tangent space, then use `DualMapGeometry.dualMapStep`. -/
instance : MatrixManifoldCarrier Stiefel where
  ShapeOK := fun _ _ => True
  project := fun {_ _} _ x => Stiefel.project _ _ x
  toMatrix := fun {_ _} _ x => x.matrix
  fromAmbientUnchecked := fun {_ _} _ x => ⟨x⟩
  dualMapStep := by
    intro m n _ x g lr
    let tg := StiefelTangent.project x g
    exact DualMapGeometry.dualMapStep x tg lr

/-- Every shape is legal for `Stiefel`. -/
instance stiefelShapeOKFact (m n : UInt64) :
    ShapeFact (MatrixManifoldCarrier.ShapeOK (M := Stiefel) m n) := ⟨trivial⟩

/-- Grassmann-manifold carrier; steps project the gradient to the tangent space, then use `DualMapGeometry.dualMapStep`. -/
instance : MatrixManifoldCarrier Grassmann where
  ShapeOK := fun _ _ => True
  project := fun {_ _} _ x => Grassmann.project _ _ x
  toMatrix := fun {_ _} _ x => x.matrix
  fromAmbientUnchecked := fun {_ _} _ x => ⟨x⟩
  dualMapStep := by
    intro m n _ x g lr
    let tg := GrassmannTangent.project x g
    exact DualMapGeometry.dualMapStep x tg lr

/-- Every shape is legal for `Grassmann`. -/
instance grassmannShapeOKFact (m n : UInt64) :
    ShapeFact (MatrixManifoldCarrier.ShapeOK (M := Grassmann) m n) := ⟨trivial⟩

/-- Orthogonal manifold viewed in matrix-family form with square-shape legality. -/
abbrev OrthogonalMatrix (_m n : UInt64) := Orthogonal n

/-- Orthogonal-matrix carrier; legal only for square shapes (`m = n`). -/
instance : MatrixManifoldCarrier OrthogonalMatrix where
  ShapeOK := fun m n => m = n
  project := by
    intro m n h mat
    let hmn : m = n := h.out
    let square : T #[n, n] := by simpa [hmn] using mat
    exact Orthogonal.project n square
  toMatrix := by
    intro m n h x
    let hmn : m = n := h.out
    exact (by simpa [hmn] using x.matrix : T #[m, n])
  fromAmbientUnchecked := by
    intro m n h mat
    let hmn : m = n := h.out
    let square : T #[n, n] := by simpa [hmn] using mat
    exact ⟨square⟩
  dualMapStep := by
    intro m n h x g lr
    let hmn : m = n := h.out
    let gSquare : T #[n, n] := by simpa [hmn] using g
    let tg := OrthogonalTangent.fromAmbient x gSquare
    exact DualMapGeometry.dualMapStep x tg lr

/-- Square shapes are legal for [`Tyr.Modular.OrthogonalMatrix`]. -/
instance orthogonalShapeOKFact (n : UInt64) :
    ShapeFact (MatrixManifoldCarrier.ShapeOK (M := OrthogonalMatrix) n n) := ⟨rfl⟩

/-- Vector-manifold carrier interface where ambient dimension may differ from intrinsic one. -/
class VectorManifoldCarrier (V : UInt64 → Type) where
  /-- Ambient coordinate dimension for intrinsic manifold dimension `n`. -/
  ambientDim : UInt64 → UInt64
  /-- Project ambient vector coordinates to the manifold. -/
  project : (n : UInt64) → T #[ambientDim n] → V n
  /-- Convert manifold value to ambient coordinates. -/
  toVector : {n : UInt64} → V n → T #[ambientDim n]
  /-- Wrap already-valid ambient coordinates without re-projecting. -/
  fromAmbientUnchecked : {n : UInt64} → T #[ambientDim n] → V n
  /-- Take one geometry-aware step on the manifold. -/
  dualMapStep : {n : UInt64} → V n → T #[ambientDim n] → Float → V n

/-- Hyperbolic-space carrier with ambient dimension `n + 1`. -/
instance : VectorManifoldCarrier Hyperbolic where
  ambientDim n := n + 1
  project := Hyperbolic.project
  toVector := fun x => x.coords
  fromAmbientUnchecked := fun x => ⟨x⟩
  dualMapStep := by
    intro n x g lr
    let tg := HyperbolicTangent.project x g
    exact DualMapGeometry.dualMapStep x tg lr

/--
Linear layer whose weight is stored on a matrix manifold.

The weight is natively represented by `M out_dim in_dim`, so retractions are
part of the layer-level update path rather than an external afterthought.
-/
structure ManifoldLinear (M : UInt64 → UInt64 → Type) [MatrixManifoldCarrier M]
    (in_dim out_dim : UInt64) where
  weight : M out_dim in_dim
  bias : Option (T #[out_dim]) := none

namespace ManifoldLinear

private def makeLeafWeight [MatrixManifoldCarrier M] {in_dim out_dim : UInt64}
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)]
    (w : M out_dim in_dim) : M out_dim in_dim :=
  MatrixManifoldCarrier.fromAmbientUnchecked
    (autograd.set_requires_grad (autograd.detach (MatrixManifoldCarrier.toMatrix w)) true)

/-- Initialize manifold-linear weights by projecting a random matrix. -/
def init [MatrixManifoldCarrier M] (in_dim out_dim : UInt64)
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)]
    (withBias : Bool := true) : IO (ManifoldLinear M in_dim out_dim) := do
  let w0 ← randn #[out_dim, in_dim] false
  let w := MatrixManifoldCarrier.project w0
  let weight := makeLeafWeight w
  let bias ← if withBias then
      let b := autograd.set_requires_grad (zeros #[out_dim]) true
      pure (some b)
    else
      pure none
  pure { weight, bias }

/-- Forward pass for `[batch, in_dim] -> [batch, out_dim]`. -/
def forward2d [MatrixManifoldCarrier M] {in_dim out_dim batch : UInt64}
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)]
    (lin : ManifoldLinear M in_dim out_dim)
    (x : T #[batch, in_dim]) : T #[batch, out_dim] :=
  let y := linear x (MatrixManifoldCarrier.toMatrix lin.weight)
  match lin.bias with
  | some b => add y (nn.expand b #[batch, out_dim])
  | none => y

/-- Apply a manifold-native step to weights and an Euclidean step to bias. -/
def applyDualMapUpdate [MatrixManifoldCarrier M] {in_dim out_dim : UInt64}
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)]
    (lin : ManifoldLinear M in_dim out_dim)
    (weightGrad : T #[out_dim, in_dim])
    (biasGrad? : Option (T #[out_dim]))
    (lr : Float) : ManifoldLinear M in_dim out_dim :=
  let weight' := MatrixManifoldCarrier.dualMapStep lin.weight weightGrad lr
  let weight' := makeLeafWeight weight'
  let bias' :=
    match lin.bias, biasGrad? with
    | some b, some g =>
      some (autograd.set_requires_grad (autograd.detach (b - mul_scalar g lr)) true)
    | some b, none => some b
    | none, _ => none
  { lin with weight := weight', bias := bias' }

end ManifoldLinear

/-- Single vector parameter living natively on a vector manifold. -/
structure ManifoldVectorParam (V : UInt64 → Type) [VectorManifoldCarrier V] (n : UInt64) where
  value : V n

namespace ManifoldVectorParam

/-- Ensure manifold-vector value is represented by a trainable leaf tensor. -/
private def makeLeafValue [VectorManifoldCarrier V] {n : UInt64}
    (v : V n) : V n :=
  VectorManifoldCarrier.fromAmbientUnchecked
    (autograd.set_requires_grad (autograd.detach (VectorManifoldCarrier.toVector v)) true)

/-- Initialize from random ambient coordinates projected to manifold. -/
def init [VectorManifoldCarrier V] (n : UInt64) : IO (ManifoldVectorParam V n) := do
  let d := (VectorManifoldCarrier.ambientDim (V := V) n)
  let vRaw ← randn #[d] false
  let v := makeLeafValue (VectorManifoldCarrier.project n vRaw)
  pure { value := v }

/-- Apply one dual-map manifold update for this vector parameter. -/
def applyDualMapUpdate [VectorManifoldCarrier V] {n : UInt64}
    (p : ManifoldVectorParam V n)
    (grad : T #[(VectorManifoldCarrier.ambientDim (V := V) n)])
    (lr : Float) : ManifoldVectorParam V n :=
  let v' := VectorManifoldCarrier.dualMapStep p.value grad lr
  { value := makeLeafValue v' }

end ManifoldVectorParam

/-- `TensorStruct` over the ambient weight matrix and optional bias of a [`Tyr.Modular.ManifoldLinear`]. -/
instance [MatrixManifoldCarrier M] (in_dim out_dim : UInt64)
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)] :
    TensorStruct (ManifoldLinear M in_dim out_dim) where
  map f lin :=
    let w := MatrixManifoldCarrier.toMatrix lin.weight
    let weight := MatrixManifoldCarrier.fromAmbientUnchecked (f w)
    let bias := lin.bias.map f
    { weight, bias }
  mapM f lin := do
    let w' ← f (MatrixManifoldCarrier.toMatrix lin.weight)
    let weight := MatrixManifoldCarrier.fromAmbientUnchecked w'
    let bias ← match lin.bias with
      | some b => some <$> f b
      | none => pure none
    pure { weight, bias }
  zipWith f a b :=
    let w := f (MatrixManifoldCarrier.toMatrix a.weight) (MatrixManifoldCarrier.toMatrix b.weight)
    let weight := MatrixManifoldCarrier.fromAmbientUnchecked w
    let bias := match a.bias, b.bias with
      | some x, some y => some (f x y)
      | _, _ => none
    { weight, bias }
  fold f init lin :=
    let acc := f (MatrixManifoldCarrier.toMatrix lin.weight) init
    match lin.bias with
    | some b => f b acc
    | none => acc

/-- `TensorStruct` over the ambient coordinates of a [`Tyr.Modular.ManifoldVectorParam`]. -/
instance [VectorManifoldCarrier V] (n : UInt64) : TensorStruct (ManifoldVectorParam V n) where
  map f p :=
    let v := VectorManifoldCarrier.toVector p.value
    { value := VectorManifoldCarrier.fromAmbientUnchecked (f v) }
  mapM f p := do
    let v' ← f (VectorManifoldCarrier.toVector p.value)
    pure { value := VectorManifoldCarrier.fromAmbientUnchecked v' }
  zipWith f a b :=
    let av := VectorManifoldCarrier.toVector a.value
    let bv := VectorManifoldCarrier.toVector b.value
    { value := VectorManifoldCarrier.fromAmbientUnchecked (f av bv) }
  fold f init p := f (VectorManifoldCarrier.toVector p.value) init

/-- [`Tyr.Modular.NormedModule`] instance for [`Tyr.Modular.ManifoldLinear`]. -/
instance [MatrixManifoldCarrier M] (in_dim out_dim : UInt64)
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)] :
    NormedModule (ManifoldLinear M in_dim out_dim) where
  norm lin :=
    let weightNorm := linalg.spectralNorm (MatrixManifoldCarrier.toMatrix lin.weight)
    match lin.bias with
    | some b => floatMax weightNorm (linalg.l2Norm b)
    | none => weightNorm
  dualNorm lin :=
    let weightDual := linalg.nuclearNorm (MatrixManifoldCarrier.toMatrix lin.weight)
    match lin.bias with
    | some b => weightDual + linalg.l2Norm b
    | none => weightDual
  nu lin := linalg.spectralNorm (MatrixManifoldCarrier.toMatrix lin.weight)
  mu _ := 1.0
  normalize lin :=
    let w := MatrixManifoldCarrier.toMatrix lin.weight
    let σ := linalg.spectralNorm w
    let w' := if σ == 0.0 then w else mul_scalar w (1.0 / σ)
    let weight := MatrixManifoldCarrier.project w'
    let bias :=
      match lin.bias with
      | some b =>
        let bNorm := linalg.l2Norm b
        if bNorm == 0.0 then some b else some (mul_scalar b (1.0 / bNorm))
      | none => none
    { lin with weight, bias }
  normalizeDual lin :=
    let w := MatrixManifoldCarrier.toMatrix lin.weight
    let fnorm := linalg.frobeniusNorm w
    let w' := if fnorm == 0.0 then w else mul_scalar w (1.0 / fnorm)
    let weight := MatrixManifoldCarrier.project w'
    let bias :=
      match lin.bias with
      | some b =>
        let bNorm := linalg.l2Norm b
        if bNorm == 0.0 then some b else some (mul_scalar b (1.0 / bNorm))
      | none => none
    { lin with weight, bias }

/-- [`Tyr.Modular.NormedModule`] for [`Tyr.Modular.ManifoldVectorParam`]: $\ell_2$ norm of the ambient coordinates. -/
instance [VectorManifoldCarrier V] (n : UInt64) :
    NormedModule (ManifoldVectorParam V n) where
  norm p := linalg.l2Norm (VectorManifoldCarrier.toVector p.value)
  dualNorm p := linalg.l2Norm (VectorManifoldCarrier.toVector p.value)
  nu _ := 1.0
  mu _ := 1.0
  normalize p :=
    let n' := n
    let v := VectorManifoldCarrier.toVector p.value
    let d := linalg.l2Norm v
    let v' := if d == 0.0 then v else mul_scalar v (1.0 / d)
    { value := VectorManifoldCarrier.project n' v' }
  normalizeDual p :=
    let n' := n
    let v := VectorManifoldCarrier.toVector p.value
    let d := linalg.l2Norm v
    let v' := if d == 0.0 then v else mul_scalar v (1.0 / d)
    { value := VectorManifoldCarrier.project n' v' }

/-- Convenience alias: manifold-linear with Stiefel-constrained weights. -/
abbrev StiefelLinear (in_dim out_dim : UInt64) := ManifoldLinear Stiefel in_dim out_dim

/-- Convenience alias: manifold-linear with Grassmann-constrained weights. -/
abbrev GrassmannLinear (in_dim out_dim : UInt64) := ManifoldLinear Grassmann in_dim out_dim

/-- Convenience alias: manifold-linear with orthogonal square weights. -/
abbrev OrthogonalLinear (dim : UInt64) := ManifoldLinear OrthogonalMatrix dim dim

/-- Convenience alias: single hyperbolic vector parameter. -/
abbrev HyperbolicVector (n : UInt64) := ManifoldVectorParam Hyperbolic n

/-- Cotangent payload for `ManifoldLinear`: weight cotangent plus optional bias cotangent. -/
abbrev ManifoldLinearCotangent (in_dim out_dim : UInt64) :=
  T #[out_dim, in_dim] × Option (T #[out_dim])

/--
Composable manifold-update interface.

This decouples "what manifold-native step to run" from "how modules are nested",
so product/sequential structures can be updated without dedicated manifold types.
-/
class ManifoldUpdatable (P : Type) where
  Cotangent : P → Type
  applyUpdate : (params : P) → Cotangent params → Float → P

namespace ManifoldUpdatable

/-- Apply one manifold-native update step. -/
def step [ManifoldUpdatable P] (params : P) (grad : ManifoldUpdatable.Cotangent params) (lr : Float) : P :=
  ManifoldUpdatable.applyUpdate params grad lr

/-- Manifold update for [`Tyr.Modular.ManifoldLinear`] via `ManifoldLinear.applyDualMapUpdate`. -/
instance [MatrixManifoldCarrier M] (in_dim out_dim : UInt64)
    [ShapeFact (MatrixManifoldCarrier.ShapeOK (M := M) out_dim in_dim)] :
    ManifoldUpdatable (ManifoldLinear M in_dim out_dim) where
  Cotangent _ := ManifoldLinearCotangent in_dim out_dim
  applyUpdate lin g lr := ManifoldLinear.applyDualMapUpdate lin g.1 g.2 lr

/-- Manifold update for [`Tyr.Modular.ManifoldVectorParam`] via `ManifoldVectorParam.applyDualMapUpdate`. -/
instance [VectorManifoldCarrier V] (n : UInt64) :
    ManifoldUpdatable (ManifoldVectorParam V n) where
  Cotangent _ := T #[(VectorManifoldCarrier.ambientDim (V := V) n)]
  applyUpdate p g lr := ManifoldVectorParam.applyDualMapUpdate p g lr

/-- Products update componentwise. -/
instance [ManifoldUpdatable A] [ManifoldUpdatable B] : ManifoldUpdatable (A × B) where
  Cotangent p := ManifoldUpdatable.Cotangent p.1 × ManifoldUpdatable.Cotangent p.2
  applyUpdate p g lr :=
    (ManifoldUpdatable.applyUpdate p.1 g.1 lr, ManifoldUpdatable.applyUpdate p.2 g.2 lr)

/-- [`Tyr.Modular.Sequential`] networks update each stage independently. -/
instance [ManifoldUpdatable M₁] [ManifoldUpdatable M₂] :
    ManifoldUpdatable (Sequential M₁ M₂) where
  Cotangent net := ManifoldUpdatable.Cotangent net.first × ManifoldUpdatable.Cotangent net.second
  applyUpdate net g lr := {
    first := ManifoldUpdatable.applyUpdate net.first g.1 lr
    second := ManifoldUpdatable.applyUpdate net.second g.2 lr
  }

end ManifoldUpdatable

private def meanOrOne (xs : Array Float) : Float :=
  if xs.isEmpty then 1.0 else xs.foldl (fun acc x => acc + x) 0.0 / xs.size.toFloat

/--
Compile a matrix-group budget multiplier from modular sensitivities of
manifold-native modules and inject it into `DualOptimizer.Config`.
-/
def applyMatrixBudgetFromModules [NormedModule M]
    (optCfg : torch.Optim.DualOptimizer.Config)
    (budgetCfg : BudgetConfig := {})
    (modules : Array M)
    : torch.Optim.DualOptimizer.Config :=
  let scales := sequentialDownstreamScales budgetCfg modules
  let matrixMul := meanOrOne scales
  { optCfg with
    useModularBudget := true
    budget := { optCfg.budget with matrix := matrixMul } }

end Tyr.Modular
