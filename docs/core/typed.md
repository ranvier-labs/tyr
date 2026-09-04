# Typed tensors: the `Tyr/Typed` facade

## Purpose and when to use

`Tyr/Typed` is a phantom-typed layer over Tyr's raw shape-indexed tensor `T s`.
It grades each tensor by a single `StaticSpec` index — shape, dtype, and device
policy — so that shape mismatches, non-broadcastable operands, and dtype- or
device-mismatched matmuls fail at compile time instead of inside libtorch.
Use it for new model code where you want the type checker to catch wiring
errors, and for boundary checks (weight loading, solver states) where you want
an explicit runtime audit of static claims. The raw `T s` FFI surface stays
fully available underneath; the two interoperate freely.

The design is LeanMLX-inspired: dtype promotion and result-spec computation
happen in a pure algebra, while device placement stays a runtime property that
the type only constrains through a policy.

## Architecture and main abstractions

The facade is six modules under `Tyr/Typed/`, aggregated by `Tyr/Typed.lean`
(itself re-exported by `Tyr.lean`). They build on each other in layers:

| Module | Contents |
| --- | --- |
| `Tyr/Typed/DType.lean` | Pure dtype algebra: promotion and reduction-result rules |
| `Tyr/Typed/Spec.lean` | `TensorSpec` (shape + dtype) and pure result-spec computation |
| `Tyr/Typed/Device.lean` | `DevicePolicy`, `TensorRole`, `TensorContract` placement contracts |
| `Tyr/Typed/Tensor.lean` | `StaticSpec`, the `Tensor σ` wrapper, checked constructors, `validate` |
| `Tyr/Typed/Ops.lean` | Typed wrappers over the `torch.*` / `torch.nn.*` FFI ops |
| `Tyr/Typed/Layers.lean` | `Typed.Linear`, `Typed.RMSNorm` — migration templates for model code |

Everything lives in `namespace torch` (the layers in `torch.Typed`).

### The spec index and the tensor

The central types (`Tyr/Typed/Tensor.lean:29`, `Tyr/Typed/Tensor.lean:65`):

```lean
structure StaticSpec where
  shape : Shape
  dtype : DType := .Unknown "dynamic"   -- default: no dtype claim
  device : DevicePolicy := .any         -- default: any placement

structure Tensor (σ : StaticSpec) where
  private mk ::
  raw : T σ.shape

abbrev DTensor (shape : Shape) (dtype : DType) : Type :=
  Tensor { shape, dtype }
abbrev TT (shape : Shape) (dtype : DType) : Type := DTensor shape dtype
```

`Tensor σ` is a single-field structure around the same opaque libtorch handle
`T σ.shape` used everywhere else in Tyr (`Tyr/Basic.lean:108`). The `σ` index
is phantom: it is erased at runtime, no FFI signature changes, and wrapping or
unwrapping is free. `Tensor.toTensor` (and a `CoeOut` instance) unwraps;
`Tensor.assumeSpec` wraps.

The defaults are graded claims: `Tensor { shape := #[2, 3] }` claims only the
shape, while `Tensor { shape := #[2, 3], dtype := .Float32, device := .exact (.CUDA 0) }`
pins all three. `DTensor s d` is the common shape+dtype instantiation.

### The pure spec algebra

Result types are computed by pure functions, so the type checker can see them:

- `DType.promote : DType → DType → DType` (`Tyr/Typed/DType.lean:80`) encodes
  PyTorch-flavored promotion (f32 absorbs f16/bf16, mixed f16+bf16 gives f32,
  integers promote by rank, `Unknown` is sticky). Helpers `sumResult`,
  `meanResult`, `divideResult`, `atLeastFloat`, and predicates `isFloating`,
  `isIntegral`, `isBool`, `isIndex` describe reduction and division results.
- `TensorSpec := { shape : Shape, dtype : DType }` (`Tyr/Typed/Spec.lean:12`)
  with `broadcastShapes?`, `broadcastList?`, `matmulShape?`, `pointwise?`,
  `division?`, `matmul?` — `Option`-returning checked computation — plus the
  total `broadcastShape` (junk `#[]` on failure) for use in type indices, and
  `checkShape` / `checkDType` / `checkCompatible` for runtime validation.
- `StaticSpec.pointwise`, `StaticSpec.division`, `StaticSpec.sumSpec`
  (`Tyr/Typed/Tensor.lean:47-62`) lift the same rules to full specs; the
  pointwise result inherits the *left* operand's device policy.

### Device policy and contracts

Device is deliberately not a hard type-level placement (`Tyr/Typed/Device.lean`):

```lean
inductive DevicePolicy where
  | any
  | exact (device : Device)

inductive TensorRole where
  | activation | parameter | index | mask | logits | loss | cache | quantScale

structure TensorContract where
  spec : TensorSpec
  role : TensorRole := .activation
  devicePolicy : DevicePolicy := .any
```

A policy states which placements the type admits; the actual placement is still
the runtime `device` argument of the creation ops. `TensorContract.check`
verifies shape, dtype, and device against runtime metadata — `role` is
documentation-only metadata and is not checked. Contracts are the boundary
mechanism used by the SafeTensors loader (see
[Serialization](../serialization.md)).

### Boundary API: checked constructors and validation

Crossing from raw `T s` into the facade (`Tyr/Typed/Tensor.lean`):

- `Tensor.assumeSpec` / `Tensor.assumeDType` — wrap unconditionally. Every
  typed op uses this internally, so each op encodes an assumption about
  libtorch behavior; the parity tests in `Tests/TestTyped.lean` exist to keep
  those assumptions honest. Treat it as the escape hatch it is in your own code.
- `Tensor.ofTensor?` / `Tensor.ofTensor` — wrap after checking the runtime
  dtype (`Option` / `Except String`).
- `Tensor.ofTensorWithContract` — full contract check (shape, dtype, device).
- `Tensor.validate : Tensor σ → Except String Unit` (`Tyr/Typed/Tensor.lean:118`)
  — audit every phantom claim against the runtime handle. Shape is always
  checked; an `.Unknown _` dtype matches anything; `.any` accepts any device.

### What is actually enforced

Compile time:

- Same-shape elementwise ops unify the shape index — mismatches do not compile.
- The matmul family requires both operands to share dtype *and* device policy.
- The specialized matmuls (`mm`, `linear`, `linear3d`, `matmul3d`, `bmm4d`,
  `affine3d`) encode inner-dimension agreement in the types.
- Broadcasting ops carry a `rfl`-discharged proof that the shapes broadcast —
  incompatible pairs fail to compile (for concrete shapes).
- Float-only math (`exp`, `sqrt`, `tanh`, …) requires an
  `σ.dtype.isFloating = true` proof, found by `decide` for literal dtypes.

Runtime only:

- Generic `Tensor.matmul` computes its result shape with the total
  `matmulShape` from `Tyr/Basic.lean:179` (unchecked, junk on bad input) — the
  checked `TensorSpec.matmulShape?` is *not* in its type. Prefer the
  specialized variants when you want static shape checking.
- Dtype claims of FFI results are assumed, not proven; they are validated by
  running the parity tests, not by the compiler.
- The device policy is propagated through types but only *enforced* by
  `validate` / contract checks; ops do not consult the runtime device.

Coverage is deliberately partial: creation, casts, elementwise, scalar and
float-unary math, the matmul family, broadcasting, two reductions, shape ops,
`affine3d`, `rmsNormWeighted`, and `setRequiresGrad`. There are no typed conv,
attention, or loss wrappers yet, and `Typed.Linear` / `Typed.RMSNorm` are the
only layers — they are templates to copy, not a module library.

## Key APIs

All of the following are in `namespace torch.Tensor` unless noted. Signatures
abridged (default arguments omitted where they only pass through to the FFI).

### Construction and casts

| Function | Signature | Notes |
| --- | --- | --- |
| `zeros`, `ones` | `(shape : Shape) → Tensor { shape, dtype := .Float32 }` | `requiresGrad`, `device` defaults |
| `full` | `(shape : Shape) (value : Float) → Tensor { shape, dtype := .Float32 }` | |
| `fullInt` | `(shape : Shape) (value : Int64) → Tensor { shape, dtype := .Int64 }` | |
| `randn` | `(shape : Shape) → IO (Tensor { shape, dtype := .Float32 })` | |
| `zerosLike`, `onesLike` | `Tensor σ → Tensor σ` | |
| `toFloat32`, `toBFloat16`, `toInt64` | `Tensor σ → Tensor (σ.withDType …)` | |

### Elementwise and scalar arithmetic

```lean
-- Same shape and device policy; dtypes promote (div additionally floats ints):
def add {s : Shape} {l r : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s, l, dev⟩) (b : Tensor ⟨s, r, dev⟩) :
    Tensor ⟨s, DType.promote l r, dev⟩
-- sub, mul analogous; div : Tensor ⟨s, (DType.promote l r).atLeastFloat, dev⟩
```

- `HAdd`/`HSub`/`HMul`/`HDiv` instances give `+ - * /` the promoting types.
- Homogeneous `Add`/`Sub`/`Mul` instances (same `σ`, dtype-preserving) exist
  for code generic over the spec, where `DType.promote d d` does not reduce.
  No homogeneous `Div` — true division changes integer dtypes.
- `addScalar`/`subScalar`/`mulScalar`/`divScalar : Tensor σ → Float → Tensor (σ.withDType σ.dtype.atLeastFloat)`,
  with matching `HAdd (Tensor σ) Float …` etc. instances.

### Unary math

`relu`, `abs` are spec-preserving for all dtypes. The float-only family takes
a proof auto-parameter `(_float : σ.dtype.isFloating = true := by decide)`:

```lean
def exp {σ : StaticSpec} (t : Tensor σ)
    (_float : σ.dtype.isFloating = true := by decide) : Tensor σ
-- same shape: sqrt, rsqrt, sigmoid, tanh, gelu, silu, pow (exponent : Float),
-- softmax (dim : Int32 := -1)
```

Integer tensors must be cast first (`.toFloat32`).

### Matmul family

Operands must share dtype `d` and device policy `dev`:

```lean
def matmul (a : Tensor ⟨s1, d, dev⟩) (b : Tensor ⟨s2, d, dev⟩) :
    Tensor ⟨matmulShape s1 s2, d, dev⟩                 -- unchecked result shape
def mm     (a : Tensor ⟨#[m, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩
def linear (x : Tensor ⟨#[b, m], d, dev⟩) (weight : Tensor ⟨#[n, m], d, dev⟩) :
    Tensor ⟨#[b, n], d, dev⟩                           -- x @ Wᵀ
def linear3d  (x : Tensor ⟨#[b, s, i], d, dev⟩) (w : Tensor ⟨#[o, i], d, dev⟩) :
    Tensor ⟨#[b, s, o], d, dev⟩
def matmul3d  (a : Tensor ⟨#[b, s, k], d, dev⟩) (b : Tensor ⟨#[k, n], d, dev⟩) :
    Tensor ⟨#[b, s, n], d, dev⟩
def bmm4d     (a : Tensor ⟨#[b, h, m, k], d, dev⟩) (x : Tensor ⟨#[b, h, k, n], d, dev⟩) :
    Tensor ⟨#[b, h, m, n], d, dev⟩
def affine3d  (x : Tensor ⟨#[b, s, i], d, dev⟩) (w : Tensor ⟨#[o, i], d, dev⟩)
    (bias : Tensor ⟨#[o], d, dev⟩) : Tensor ⟨#[b, s, o], d, dev⟩
```

### Broadcasting

```lean
def addB (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂)
-- subB, mulB analogous; divB : Tensor (σ₁.division σ₂)

def broadcastTo (t : Tensor σ) (target : Shape)
    (_h : TensorSpec.broadcastsTo σ.shape target := by rfl) :
    Tensor (σ.withShape target)     -- checked, expand view
def expand (t : Tensor σ) (target : Shape) : Tensor (σ.withShape target)
    -- unchecked, for symbolic dims; libtorch validates at runtime
```

Scoped operators expand to the `…B` functions (`Tyr/Typed/Ops.lean:376-379`),
available under `open torch`:

```lean
scoped infixl:65 " +ᵇ " => Tensor.addB   -- also -ᵇ
scoped infixl:70 " *ᵇ " => Tensor.mulB   -- also /ᵇ
```

They are notation rather than instances on purpose: instance resolution never
runs auto-param tactics, so only a plain application gets the broadcastability
proof discharged. The result spec is `σ₁.pointwise σ₂` — promoted dtype, left
operand's device policy. At runtime the operands pass through a `reshape #[]`
type erasure and libtorch broadcasts natively; no expansion is materialized.

### Reductions and shape ops

```lean
def sumAll  (t : Tensor σ) : Tensor σ.sumSpec        -- shape #[], dtype sumResult
def meanAll (t : Tensor σ) (_float : σ.dtype.isFloating = true := by decide) :
    Tensor (σ.withShape #[])

def reshape     (t : Tensor σ) (shape' : Shape) : Tensor (σ.withShape shape')
def permute     (t : Tensor σ) (p : Array UInt64) :
    Tensor (σ.withShape (permuteShape σ.shape p))
def transpose   (t : Tensor σ) (dim0 dim1 : UInt64) :
    Tensor (σ.withShape (transposeShape σ.shape dim0.toNat dim1.toNat))
def transpose2d (t : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[n, m], d, dev⟩
def unsqueeze   (t : Tensor σ) (dim : Nat) :
    Tensor (σ.withShape (unsqueezeShape σ.shape dim))   -- squeeze analogous
def cat (t1 : Tensor ⟨s1, d, dev⟩) (t2 : Tensor ⟨s2, d, dev⟩) (dim : Nat) :
    Tensor ⟨nn.catShape s1 s2 dim, d, dev⟩
```

### Layers (`namespace torch.Typed`, `Tyr/Typed/Layers.lean`)

```lean
structure Linear (inDim outDim : UInt64) (d : DType) (dev : DevicePolicy := .any) where
  weight : Tensor ⟨#[outDim, inDim], d, dev⟩
  bias : Option (Tensor ⟨#[outDim], d, dev⟩) := none

Linear.init (inDim outDim : UInt64) (withBias : Bool := true) :
    IO (Linear inDim outDim .Float32)          -- Kaiming-style, requiresGrad
Linear.forward2d (lin : Linear i o d dev) (x : Tensor ⟨#[b, i], d, dev⟩) :
    Tensor ⟨#[b, o], d, dev⟩
Linear.forward3d (lin : Linear i o d dev) (x : Tensor ⟨#[b, s, i], d, dev⟩) :
    Tensor ⟨#[b, s, o], d, dev⟩

structure RMSNorm (dim : UInt64) (d : DType) (dev : DevicePolicy := .any) where
  weight : Tensor ⟨#[dim], d, dev⟩
  eps : Float := 1e-6

RMSNorm.init (dim : UInt64) (eps : Float := 1e-6) : RMSNorm dim .Float32
-- forward2d / forward3d / forward4d take an input whose last dimension is
-- `dim` and return the same shape with the upcast dtype, e.g.:
RMSNorm.forward3d (rn : RMSNorm dim d dev) (x : Tensor ⟨#[b, s, dim], d, dev⟩)
    (float : d.isFloating = true := by decide) :
    Tensor ⟨#[b, s, dim], DType.promote .Float32 d, dev⟩
```

The RMSNorm result type advertises a real runtime behavior: `nn.rmsNormWeighted`
computes in Float32 for stability and does not cast back, so a bf16 layer
returns an f32 tensor (`promote .Float32 .BFloat16 = .Float32`). Cast back
explicitly (`.toBFloat16`) where HF-style dtype restoration is wanted.

## Usage example

Reconstructed example (from `Tests/TestTyped.lean` and
`launch/demos/shape_safety.lean`):

```lean
import Tyr.Typed

open torch

-- Batch and feature dimensions are visible in the type; the result shape is
-- computed by the type checker.
def project
    (x : Tensor { shape := #[32, 768], dtype := .Float32 })
    (weight : Tensor { shape := #[512, 768], dtype := .Float32 }) :
    Tensor { shape := #[32, 512], dtype := .Float32 } :=
  Tensor.linear x weight

-- A tiny typed stack: Linear followed by RMSNorm, Float32, any device.
def forward
    (lin : Typed.Linear 768 512 .Float32)
    (rn : Typed.RMSNorm 512 .Float32)
    (x : Tensor ⟨#[2, 16, 768], .Float32, .any⟩) :
    Tensor ⟨#[2, 16, 512], .Float32, .any⟩ :=
  rn.forward3d (lin.forward3d x)

-- Compile-time-checked broadcasting: bias add with native libtorch broadcast.
def addBias
    (x : Tensor ⟨#[2, 16, 512], .Float32, .any⟩)
    (bias : Tensor ⟨#[512], .Float32, .any⟩) :
    Tensor ⟨#[2, 16, 512], .Float32, .any⟩ :=
  x +ᵇ bias

def run : IO Unit := do
  let lin ← Typed.Linear.init 768 512
  let rn := Typed.RMSNorm.init 512
  let x ← Tensor.randn #[2, 16, 768]
  let y := addBias (forward lin rn x) (Tensor.zeros #[512])
  -- Audit the phantom claims against the live runtime handle.
  match y.validate with
  | .ok () => IO.println s!"ok: {y.raw.runtimeShape}, dtype {y.raw.dtype}"
  | .error err => throw (IO.userError err)
```

For the negative case, `launch/demos/shape_mismatch.lean` is an intentionally
rejected file: passing a `#[768, 512]` weight where `linear` expects
`#[n, 768]` fails to compile, with the expected and actual shapes in the error.

Boundary-checking a raw tensor (the pattern used by the SafeTensors loader in
`Tyr/SafeTensors/Schema.lean:62-71`):

```lean
def checkRaw (raw : T #[2, 3]) : Except String (DTensor #[2, 3] .Float32) :=
  Tensor.ofTensor .Float32 raw
```

## Related guides

- [Raw tensors and the FFI surface](tensors.md) — `T s`, `Shape`, `DType`, `Device`, and the `torch.*` ops this facade wraps.
- [TensorStruct](tensorstruct.md) — tree traversals over parameter structures (works on raw `T s`).
- [Core utilities](utilities.md) — PRNG, logging, widgets.
- [Serialization](../serialization.md) — SafeTensors schema introspection and contract-checked loading built on `TensorContract`.
- [Autodiff](../autodiff.md) — gradients; the typed layer only exposes `setRequiresGrad`.
- [DiffEq](../diffeq.md) — `Tyr/DiffEq/Typed.lean` adds `DiffEqSpace`/`DiffEqElem`/`DiffEqSeminorm` instances for `Tensor σ`.
- [Examples and testing](../examples-and-testing.md) — `Tests/TestTyped.lean` is the executable spec of the promotion algebra.

For an exhaustive symbol listing, see the doc-gen4 API reference generated from
`docbuild/`; this chapter is a guide, not a symbol dump.
