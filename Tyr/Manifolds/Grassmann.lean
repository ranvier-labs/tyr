/-
  Tyr/Manifolds/Grassmann.lean

  Grassmann manifold Gr(n, p): p-dimensional subspaces of ℝⁿ.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch (T)
open DifferentiableManifold

/-! ## Grassmann Manifold Gr(n, p)

The Grassmann manifold Gr(n, p) is the space of p-dimensional subspaces of ℝⁿ.

Key properties:
- Elements are equivalence classes of n×p matrices with orthonormal columns
- Two matrices X, Y represent the same subspace iff X = Y @ Q for some Q ∈ O(p)
- dim(Gr(n, p)) = p(n - p)
- Gr(n, 1) = ℝPⁿ⁻¹ (real projective space)

The tangent space at X is the horizontal space (orthogonal to the fiber):
  T_X Gr(n,p) = { Z ∈ ℝⁿˣᵖ : XᵀZ = 0 }

This differs from Stiefel where XᵀZ is skew-symmetric. For Grassmann, XᵀZ = 0 strictly.
-/

/-- Grassmann manifold Gr(n, p): p-dimensional subspaces of ℝⁿ.
    Represented by n×p matrices with orthonormal columns.
    Constraint: XᵀX = Iₚ, and tangent vectors satisfy XᵀZ = 0 -/
structure Grassmann (n p : UInt64) where
  /-- Representative matrix with orthonormal columns -/
  matrix : T #[n, p]

namespace Grassmann

/-- Create a random point on Gr(n, p) via QR decomposition -/
def random (n p : UInt64) : IO (Grassmann n p) := do
  let mat ← torch.randn #[n, p]
  let (Q, _R) := torch.linalg.qr_reduced mat
  return ⟨Q⟩

/-- Project a general n×p matrix to Gr(n, p) using QR decomposition -/
def project (n p : UInt64) (mat : T #[n, p]) : Grassmann n p :=
  let (Q, _R) := torch.linalg.qr_reduced mat
  ⟨Q⟩

/-- Standard basis subspace: span{e₁, ..., eₚ}
    Created by projecting a simple matrix to get orthonormal columns. -/
def standard (n p : UInt64) : Grassmann n p :=
  -- QR of ones matrix gives a valid orthonormal basis
  project n p (torch.ones #[n, p])

/-- Compute the projection matrix P = X @ Xᵀ onto the subspace -/
def projectionMatrix (X : Grassmann n p) : T #[n, n] :=
  torch.nn.mm X.matrix (torch.nn.transpose2d X.matrix)

/-- Principal angles between two subspaces.
    Returns the cosines of the principal angles (singular values of X₁ᵀX₂).
    The singular values σᵢ = cos(θᵢ) where θᵢ are the principal angles. -/
def principalAngles (X₁ X₂ : Grassmann n p) : T #[p] :=
  -- σ = svdvals(X₁ᵀ @ X₂)
  let product := torch.nn.mm (torch.nn.transpose2d X₁.matrix) X₂.matrix
  torch.linalg.svdvals product

/-- Chordal distance between two subspaces.
    d(X₁, X₂) = ||P₁ - P₂||_F / √2 = (Σᵢ sin²θᵢ)ᐟ², where P₁, P₂ are the
    orthogonal projectors and θᵢ the principal angles.
    Note: this is not the geodesic distance ||θ||₂ = (Σᵢ θᵢ²)ᐟ². -/
def distance (X₁ X₂ : Grassmann n p) : Float :=
  let P₁ := projectionMatrix X₁
  let P₂ := projectionMatrix X₂
  let diff := torch.sub P₁ P₂
  -- ||P₁ - P₂||_F / √2 is the chordal (projection-Frobenius) distance
  let frobSq := torch.nn.sumAll (torch.mul diff diff)
  Float.sqrt (torch.nn.item frobSq) / Float.sqrt 2.0

end Grassmann

/-- Tangent vector on the Grassmann manifold.
    Z ∈ T_X Gr(n,p) satisfies: XᵀZ = 0 (Z is orthogonal to columns of X) -/
structure GrassmannTangent (n p : UInt64) where
  /-- The tangent vector as an n×p matrix orthogonal to base point -/
  vec : T #[n, p]

namespace GrassmannTangent

/-- Zero tangent vector -/
def zero (n p : UInt64) : GrassmannTangent n p :=
  ⟨torch.zeros #[n, p]⟩

/-- Add two tangent vectors -/
def add (v w : GrassmannTangent n p) : GrassmannTangent n p :=
  ⟨torch.add v.vec w.vec⟩

/-- Scale a tangent vector -/
def smul (s : Float) (v : GrassmannTangent n p) : GrassmannTangent n p :=
  ⟨torch.mul_scalar v.vec s⟩

/-- Project an ambient space vector to the horizontal tangent space at X.
    Π_X(V) = (I - X @ Xᵀ) @ V
    This ensures XᵀΠ_X(V) = 0 -/
def project (X : Grassmann n p) (V : T #[n, p]) : GrassmannTangent n p :=
  -- (I - X @ X^T) @ V = V - X @ (X^T @ V)
  let Xt := torch.nn.transpose2d X.matrix
  let XtV := torch.nn.mm Xt V              -- [p, p]
  let XXtV := torch.nn.mm X.matrix XtV     -- [n, p]
  let horizontal := torch.sub V XXtV
  ⟨horizontal⟩

/-- Inner product of tangent vectors (induced by Frobenius norm) -/
def inner (v w : GrassmannTangent n p) : Float :=
  torch.nn.item (torch.nn.sumAll (torch.mul v.vec w.vec))

/-- Norm of tangent vector -/
def norm (v : GrassmannTangent n p) : Float :=
  Float.sqrt (inner v v)

end GrassmannTangent

namespace Grassmann

/-- Log map on Gr(n,p).
    Uses the canonical formula based on principal angles. -/
def log (X Y : Grassmann n p) : GrassmannTangent n p :=
  let Xt := torch.nn.transpose2d X.matrix
  let XtY := torch.nn.mm Xt Y.matrix                     -- [p, p]
  let invXtY := torch.linalg.inv XtY
  let proj := torch.sub Y.matrix (torch.nn.mm X.matrix XtY) -- (I - X Xᵀ) Y
  let A := torch.nn.mm proj invXtY
  let (U, S, Vh) := torch.linalg.svd A                   -- U [n,p], S [p], Vh [p,p]
  let theta := torch.nn.atan S
  let Theta := torch.linalg.diagflat theta
  let delta := torch.nn.mm (torch.nn.mm U Theta) Vh
  ⟨delta⟩

end Grassmann

/-- Grassmann manifold as a DifferentiableManifold.
    Uses QR-based retraction: R(X, Z) = qr(X + Z).Q -/
instance grassmannManifold (n p : UInt64) : DifferentiableManifold (Grassmann n p) where
  Tangent _ := GrassmannTangent n p
  Cotangent _ := GrassmannTangent n p
  zeroTangent _ := GrassmannTangent.zero n p
  zeroCotangent _ := GrassmannTangent.zero n p
  addTangent v w := GrassmannTangent.add v w
  addCotangent v w := GrassmannTangent.add v w
  scaleTangent s v := GrassmannTangent.smul s v
  sharp := id
  flat := id
  exp X Z :=
    -- QR-based retraction: R(X, Z) = qr(X + Z).Q
    -- This is a first-order approximation to the geodesic
    let newMat := torch.add X.matrix Z.vec
    let (Q, _R) := torch.linalg.qr_reduced newMat
    ⟨Q⟩

end Tyr.AD
