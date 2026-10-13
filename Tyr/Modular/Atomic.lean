import Tyr.Modular.Norm
import Tyr.Module.Linear
import Tyr.Module.LayerNorm

/-!
# Atomic modules

`NormedModule` instances for atomic (leaf) neural network modules.

Atomic modules are those whose norms are defined directly rather than composed:
- `torch.Linear`: spectral norm (largest singular value)
- `torch.LayerNorm`: $\ell_2$ norm of weight and bias
- `Tyr.Modular.Embedding`: max row norm

These are the base cases for the recursive modular norm; see
`Tyr.Modular.NormedModule` and `Tyr.Modular.Sequential` for how they compose.
-/

namespace Tyr.Modular

open torch.linalg

/-! ## Linear Layer

For a linear layer $y = Wx$, the modular norm is the spectral norm:
$$\|W\|_\sigma = \sigma_{\max}(W)$$

The dual norm (for gradients) is the nuclear norm:
$$\|G\|_* = \sum_i \sigma_i(G)$$

Input sensitivity $\nu = \|W\|_\sigma$ since $\|Wx\|_2 \le \|W\|_\sigma \|x\|_2$.
-/

/-- Spectral-norm `NormedModule` for `torch.Linear`. If a bias is present, the
norm is the max of the weight's spectral norm and the bias's $\ell_2$ norm.
`normalizeDual` currently uses Frobenius normalization as an approximation. -/
instance (in_dim out_dim : UInt64) : NormedModule (torch.Linear in_dim out_dim) where
  norm m :=
    let weightNorm := spectralNorm m.weight
    match m.bias with
    | some b => floatMax weightNorm (l2Norm b)
    | none => weightNorm

  dualNorm m :=
    let weightDual := nuclearNorm m.weight
    match m.bias with
    | some b => weightDual + l2Norm b
    | none => weightDual

  nu m := spectralNorm m.weight

  mu _ := 1.0  -- For well-normed modules

  normalize m :=
    let σ := spectralNorm m.weight
    if σ == 0.0 then m else
    let normalizedWeight := torch.mul_scalar m.weight (1.0 / σ)
    match m.bias with
    | some b =>
      let bNorm := l2Norm b
      let normalizedBias := if bNorm == 0.0 then b else torch.mul_scalar b (1.0 / bNorm)
      { m with weight := normalizedWeight, bias := some normalizedBias }
    | none => { m with weight := normalizedWeight }

  normalizeDual m :=
    -- For spectral norm, the dual normalization involves SVD projection
    -- For now, use Frobenius normalization as an approximation
    let fNorm := frobeniusNorm m.weight
    if fNorm == 0.0 then m else
    let normalizedWeight := torch.mul_scalar m.weight (1.0 / fNorm)
    match m.bias with
    | some b =>
      let bNorm := l2Norm b
      let normalizedBias := if bNorm == 0.0 then b else torch.mul_scalar b (1.0 / bNorm)
      { m with weight := normalizedWeight, bias := some normalizedBias }
    | none => { m with weight := normalizedWeight }

/-! ## LayerNorm

LayerNorm has learnable scale (weight) and shift (bias) parameters.
The modular norm combines them via $\ell_2$:
$$\|(\gamma, \beta)\|_M = \sqrt{\|\gamma\|_2^2 + \|\beta\|_2^2}$$

Input sensitivity $\nu$ depends on the activation range, typically $O(1)$.
-/

/-- `NormedModule` for `torch.LayerNorm`: $\ell_2$ combination of scale and
shift norms; self-dual, with $\nu = \mu = 1$. -/
instance (dim : UInt64) : NormedModule (torch.LayerNorm dim) where
  norm m :=
    let wNorm := l2Norm m.weight
    let bNorm := l2Norm m.bias
    Float.sqrt (wNorm * wNorm + bNorm * bNorm)

  dualNorm m :=
    -- Dual of ℓ₂ is ℓ₂
    let wNorm := l2Norm m.weight
    let bNorm := l2Norm m.bias
    Float.sqrt (wNorm * wNorm + bNorm * bNorm)

  nu _ := 1.0  -- LayerNorm approximately preserves scale

  mu _ := 1.0

  normalize m :=
    -- Compute norm inline to avoid recursion
    let wNorm := l2Norm m.weight
    let bNorm := l2Norm m.bias
    let n := Float.sqrt (wNorm * wNorm + bNorm * bNorm)
    if n == 0.0 then m else
    { m with
      weight := torch.mul_scalar m.weight (1.0 / n)
      bias := torch.mul_scalar m.bias (1.0 / n) }

  normalizeDual m :=
    -- Compute dual norm inline
    let wNorm := l2Norm m.weight
    let bNorm := l2Norm m.bias
    let n := Float.sqrt (wNorm * wNorm + bNorm * bNorm)
    if n == 0.0 then m else
    { m with
      weight := torch.mul_scalar m.weight (1.0 / n)
      bias := torch.mul_scalar m.bias (1.0 / n) }

/-! ## Embedding Layer

For an embedding table $E \in \mathbb{R}^{V \times d}$, the modular norm is the max row norm:
$$\|E\|_M = \max_i \|e_i\|_2$$

This ensures that no single embedding vector dominates.
The dual norm is the sum of row norms ($\ell_1$ of row $\ell_2$ norms).
-/

/-- Embedding table: maps discrete tokens to continuous vectors -/
structure Embedding (vocabSize dim : UInt64) where
  weight : torch.T #[vocabSize, dim]

namespace Embedding

/-- Treats the embedding table as a single tensor field. -/
instance : torch.TensorStruct (Embedding vocabSize dim) where
  map f e := { weight := f e.weight }
  mapM f e := do
    let w' ← f e.weight
    pure { weight := w' }
  zipWith f e1 e2 := { weight := f e1.weight e2.weight }
  fold f init e := f e.weight init

/-- Initialize with $\mathcal{N}(0, 1/\mathrm{dim})$ entries (std $1/\sqrt{\mathrm{dim}}$), with gradients enabled. -/
def init (vocabSize dim : UInt64) : IO (Embedding vocabSize dim) := do
  let std := 1.0 / Float.sqrt dim.toFloat
  let w ← torch.randn #[vocabSize, dim]
  let weight := torch.mul_scalar w std
  let weight := torch.autograd.set_requires_grad weight true
  pure { weight }

/-- Lookup embeddings for a batch of token indices -/
def forward {vocabSize dim batch seq : UInt64}
    (emb : Embedding vocabSize dim)
    (indices : torch.T #[batch, seq]) : torch.T #[batch, seq, dim] :=
  torch.nn.embedding indices emb.weight

end Embedding

/-- Max-row-norm `NormedModule` for `Embedding`; the dual norm is the sum of row norms. -/
instance (vocabSize dim : UInt64) : NormedModule (Embedding vocabSize dim) where
  norm m := maxRowNorm m.weight

  dualNorm m :=
    -- Sum of row norms
    let norms := rowNorms m.weight
    torch.nn.item (torch.nn.sumAll norms)

  nu m := maxRowNorm m.weight

  mu _ := 1.0

  normalize m :=
    let n := maxRowNorm m.weight
    if n == 0.0 then m else
    { weight := torch.mul_scalar m.weight (1.0 / n) }

  normalizeDual m :=
    -- Compute dual norm inline to avoid recursion
    let norms := rowNorms m.weight
    let n := torch.nn.item (torch.nn.sumAll norms)
    if n == 0.0 then m else
    { weight := torch.mul_scalar m.weight (1.0 / n) }

end Tyr.Modular
