import Tyr.Torch

/-!
# Metric factors

Low-rank factors of positive semidefinite pullback metrics
(`Tyr.Modular.MetricFactor`) and diagonal mass regularizers
(`Tyr.Modular.DiagonalMass`).
-/

namespace Tyr.Modular

open torch

/-- Euclidean-valued factor for a positive semidefinite pullback metric. -/
structure MetricFactor (rank dim : UInt64) where
  matrix : T #[rank, dim]

/-- Diagonal positive mass metric used to regularize a layer metric. -/
structure DiagonalMass (dim : UInt64) where
  diag : T #[dim]

namespace DiagonalMass

/-- Identity diagonal mass. -/
def ones (dim : UInt64) : DiagonalMass dim :=
  { diag := torch.ones #[dim] }

/-- Constant diagonal mass. -/
def ofScalar (dim : UInt64) (value : Float) : DiagonalMass dim :=
  { diag := torch.full #[dim] value }

/-- Dense diagonal matrix representation. -/
def toMatrix {dim : UInt64} (d : DiagonalMass dim) : T #[dim, dim] :=
  linalg.diagflat d.diag

/-- Elementwise inverse diagonal entries. -/
def invDiag {dim : UInt64} (d : DiagonalMass dim) : T #[dim] :=
  nn.div (ones_like d.diag) d.diag

/-- Dense inverse of the diagonal mass. -/
def invMatrix {dim : UInt64} (d : DiagonalMass dim) : T #[dim, dim] :=
  linalg.diagflat d.invDiag

/-- Apply the diagonal mass to a vector. -/
def apply {dim : UInt64} (d : DiagonalMass dim) (x : T #[dim]) : T #[dim] :=
  d.diag * x

/-- Apply the inverse diagonal mass to a vector. -/
def applyInv {dim : UInt64} (d : DiagonalMass dim) (x : T #[dim]) : T #[dim] :=
  d.invDiag * x

end DiagonalMass

namespace MetricFactor

/-- Identity metric factor on Euclidean output space. -/
def identity (dim : UInt64) : MetricFactor dim dim :=
  { matrix := eye dim }

/-- Construct a factor directly from its matrix representation. -/
def ofMatrix {rank dim : UInt64} (matrix : T #[rank, dim]) : MetricFactor rank dim :=
  { matrix }

/-- Pull a metric factor back along a local Jacobian. -/
def pullback {rank outDim inDim : UInt64}
    (L : MetricFactor rank outDim)
    (A : T #[outDim, inDim]) : MetricFactor rank inDim :=
  { matrix := nn.mm L.matrix A }

private def scaleColumns {rank dim : UInt64}
    (matrix : T #[rank, dim])
    (scale : T #[dim]) : T #[rank, dim] :=
  let scaleRow : T #[1, dim] := reshape scale #[1, dim]
  let scaleFull : T #[rank, dim] := nn.expand scaleRow #[rank, dim]
  matrix * scaleFull

private def vectorToColumn {n : UInt64} (v : T #[n]) : T #[n, 1] :=
  reshape v #[n, 1]

private def columnToVector {n : UInt64} (v : T #[n, 1]) : T #[n] :=
  reshape v #[n]

/-- Dense Gram form `K^T K`. -/
def gram {rank dim : UInt64} (K : MetricFactor rank dim) : T #[dim, dim] :=
  nn.mm (nn.transpose2d K.matrix) K.matrix

/-- Dense regularized layer metric `D + K^T K`. -/
def denseMetric {rank dim : UInt64}
    (D : DiagonalMass dim)
    (K : MetricFactor rank dim) : T #[dim, dim] :=
  D.toMatrix + K.gram

/-- Apply `K` to a vector. -/
def apply {rank dim : UInt64} (K : MetricFactor rank dim) (x : T #[dim]) : T #[rank] :=
  columnToVector (nn.mm K.matrix (vectorToColumn x))

/-- Apply `K^T` to a vector. -/
def applyTranspose {rank dim : UInt64} (K : MetricFactor rank dim) (x : T #[rank]) : T #[dim] :=
  columnToVector (nn.mm (nn.transpose2d K.matrix) (vectorToColumn x))

/-- Apply the regularized metric `D + K^T K` to a vector. -/
def applyRegularized {rank dim : UInt64}
    (D : DiagonalMass dim)
    (K : MetricFactor rank dim)
    (x : T #[dim]) : T #[dim] :=
  D.apply x + K.applyTranspose (K.apply x)

/-- Woodbury inner matrix `I + K D^{-1} K^T`. -/
def woodburyInner {rank dim : UInt64}
    (D : DiagonalMass dim)
    (K : MetricFactor rank dim) : T #[rank, rank] :=
  let kd := scaleColumns K.matrix D.invDiag
  eye rank + nn.mm kd (nn.transpose2d K.matrix)

/-- Solve `(D + K^T K) x = g` using the Woodbury identity. -/
def solveWoodbury {rank dim : UInt64}
    (D : DiagonalMass dim)
    (K : MetricFactor rank dim)
    (g : T #[dim]) : T #[dim] :=
  let inner := woodburyInner D K
  let innerInv := linalg.inv inner
  let base := D.applyInv g
  let correctionArg := K.apply base
  let correctionInner := columnToVector (nn.mm innerInv (vectorToColumn correctionArg))
  let correction := D.applyInv (K.applyTranspose correctionInner)
  base - correction

end MetricFactor

end Tyr.Modular
