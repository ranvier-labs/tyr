# Shape-indexed tensors and the torch FFI

## Purpose & when to use

`Tyr/Basic.lean` and `Tyr/Torch.lean` are the tensor layer everything else in Tyr sits on. `Tyr/Basic.lean` defines the shape-indexed tensor type `T s` — an opaque handle to a libtorch tensor whose `Shape` index exists only at the Lean level — plus the pure shape arithmetic used to compute result types. `Tyr/Torch.lean` binds roughly 245 `@[extern]` declarations (about 243 unique `lean_torch_*` symbols, implemented under `cc/src/`) into shape-aware Lean declarations: creation, arithmetic, `nn` functional ops, autograd, SafeTensors IO, rotary embeddings, and linear algebra. Import `Tyr.Torch` directly when you want the low-level surface with a narrow dependency footprint; everything here is re-exported by `import Tyr`.

## Architecture & main abstractions

### The phantom shape index

Everything lives in `namespace torch` (`Tyr/Basic.lean:21`).

```lean
abbrev Shape := Array UInt64                       -- Tyr/Basic.lean:23

opaque TSpec : NonemptyType
def T (_ : Shape) : Type := TSpec.type             -- Tyr/Basic.lean:107-108
```

`T s` is a phantom-typed opaque handle: the shape index is erased at runtime and the C++ side never sees it. `Nonempty (T s)` comes from `TSpec.property` (`Tyr/Basic.lean:203`); `Inhabited (T s)` is synthesized as `zeros s` (`Tyr/Torch.lean:60`). Because the index is a compile-time value, shapes like `T #[batch, seq, n_embd]` appear directly in signatures and dimension mismatches are elaboration errors.

Supporting types:

```lean
inductive DType where                                -- Tyr/Basic.lean:25
  | UInt8 | Int8 | Int16 | Int32 | Int64
  | Float16 | BFloat16 | Float32 | Float64 | Bool
  | Float8E4M3FN | Float8E5M2
  | Unknown (raw : String)

inductive Device where                               -- Tyr/Basic.lean:101
  | CUDA : UInt64 → Device
  | CPU
  | MPS
```

`DType` carries parsers/normalizers for PyTorch and SafeTensors metadata (`DType.ofString?`, `DType.parse`, `DType.canonicalName`, `DType.safeTensorTag?`). The dtype and device of a tensor are *not* tracked in `T s` — only the shape is; dtype/device are runtime properties read back through the metadata FFI (below). For dtype/device in types, see the typed facade in [typed.md](typed.md).

### Pure shape arithmetic

Result shapes are computed by total, pure Lean functions, so the type checker evaluates them during elaboration (`Tyr/Basic.lean:113-202`):

| Function | Computes |
|---|---|
| `unsqueezeShape s dim` / `squeezeShape s dim` | insert/remove a size-1 dim |
| `transposeShape s dim0 dim1` | swap two dims |
| `reduceShape s dim keepdim` | reduction along `dim` |
| `replaceAtDim s dim newSize` | overwrite one dim (used by `topk_values`, `multinomial`) |
| `stackShape s n dim` / `unbindShape s dim` | stack / unbind |
| `matmulShape s1 s2` | full PyTorch matmul broadcast rules |

`matmulShape` (`Tyr/Basic.lean:179`) implements the complete PyTorch contract: 1D·1D → `#[]` (dot product), 1D·2D, 2D·1D, 2D·2D, and ND·ND with broadcast batch dims. It is total and returns `#[]` for invalid inputs — it does **not** prove the multiplication is legal; a bad call fails at runtime in C++, not at compile time.

`Tyr/Torch.lean` adds the conv/pool shape family as plain Lean functions: `convOutputSize`, `convTransposeOutputSize`, `conv1dShape`, `convTranspose1dShape`, `conv2dShape`, `conv3dShape`, `poolOutputSize`, `pool2dShape` (`Tyr/Torch.lean:179-246`), plus `permuteShape` (`:309`), `catShape` (`:657`), `sliceShape` (`:851`), `slicedShape` (`:171`).

### The FFI pattern: opaque externs and shape re-tagging

Every binding is an `@[extern "lean_torch_*"] opaque` declaration; the Lean signature is a *claim* about what the C++ function does. Two patterns recur:

1. **Direct typed bindings** — the result shape is a closed function of the inputs, e.g. `nn.mm : T #[m, k] → T #[k, n] → T #[m, n]` (`Tyr/Torch.lean:569`).
2. **Shape-erased binding + pure re-tag** — the C++ result comes back as `T #[]` and a Lean `def` re-tags it with `reshape`:

```lean
@[extern "lean_torch_matmul"] private opaque matmul_impl {s1 s2 : Shape}
    (a : @& T s1) (b : @& T s2) : T #[]

def matmul {s1 s2 : Shape} (a : T s1) (b : T s2) : T (matmulShape s1 s2) :=
  reshape (matmul_impl a b) (matmulShape s1 s2)    -- Tyr/Torch.lean:564-565
```

`cat`, `expand`, `sumDim`, `meanDim`, and `slice1d` follow the same pattern. This is the trust boundary of the whole design: if the Lean shape logic and the C++ behavior drift apart, you get a runtime wrong-shape bug the type system cannot catch.

A few bindings are genuinely effectful and live in `IO`: `rand`, `randn`, `randint`, `nn.dropout*`, `nn.multinomial`, the autograd graph controls, and all file IO. Creation functions that are deterministic (`zeros`, `ones`, `full`, `eye`, `arange`, `linspace`) are pure.

### Namespaces

`Tyr/Torch.lean` is one long `namespace torch` with nested scopes:

| Namespace | Lines | Contents |
|---|---|---|
| (top level) | 37-318, 1243-1483 | creation, arithmetic, `linear`/`affine*`, device/runtime, comparison/logical, indexing/gather/scatter, einsum, `focal_loss`/`triplet_margin_loss` |
| `torch.autograd` | 251-283, 767-771, 1053-1058 | gradient queries, graph control, backward |
| `torch.nn` | 320-764, 1108-1241 | conv/pool, norms, losses, activations, attention, sampling filters |
| `torch.data` | 774-936 | binary/token IO, WAV/PPM, media preprocessing, array conversion |
| `torch.signal` | 938-975 | STFT/ISTFT/RFFT, Hann window |
| `torch.safetensors` | 978-1050 | SafeTensors load/save, sharded loading |
| `torch.rotary` | 1067-1106 | RoPE frequency precomputation and application |
| `torch.linalg` | 1489-1565 | QR/SVD/exp/log/inv, matrix norms |

## Key APIs

Only what a user actually touches; signatures abbreviated where defaults are obvious. `@[extern]`/`opaque` omitted for readability.

### Creation

```lean
zeros  (s : Shape) (requires_grad : Bool := false) (device : Device := .CPU) : T s
ones   (s : Shape) (requires_grad : Bool := false) (device : Device := .CPU) : T s
full   (s : Shape) (value : Float) ... : T s
rand   (s : Shape) ... : IO (T s)        -- uniform [0, 1)
randn  (s : Shape) ... : IO (T s)        -- unit normal
randint (low high : Int64) (s : Shape) ... : IO (T s)
arange (start stop : UInt64) (step : UInt64 := 1) : T #[(stop - start)/step]
eye (n : UInt64) ... : T #[n, n]
linspace / logspace (start stop : Float) (steps : UInt64) ... : T #[steps]
zeros_like / ones_like {s} (t : T s) : T s
uniform (s : Shape) (min max : Float) : IO (T s)   -- pure-Lean composite, Tyr/Torch.lean:88
zerosOn / onesOn {s} (device : Device) : T s       -- Tyr/Torch.lean:1313-1318
```

### Arithmetic and scalar ops

Same-shape `add`/`sub`/`mul {s} (t t' : T s) : T s` with `Add`/`Sub`/`Mul` instances; scalar `mul_scalar`/`div_scalar`/`add_scalar`/`sub_scalar` with `HMul`/`HDiv`/`HAdd`/`HSub (T s) Float (T s)` instances (`Tyr/Torch.lean:62-85`). Elementwise math: `relu`, `relu6`, `rsqrt`, `sqrt`, `abs`, `pow`, `exp`, `log`, `log10`, `cos`, `sin`, `atan`, `atan2`, `floor`, `div`. Casts: `toFloat'`, `toBFloat16'`, `toLong`, `castLike`, `restoreInputDType`.

Broadcasting multiplication is the one quirk to know:

```lean
opaque mul' {s1 s2 : Shape} (t1 : @& T s1) (t2 : @& T s2) : T s2   -- Tyr/Torch.lean:103
instance {s1 s2 : Shape} : HMul (T s1) (T s2) (T s2)                -- Tyr/Torch.lean:105
```

The result is typed as the *right* operand's shape regardless of actual broadcast semantics — fine for the dominant `normed * weight` idiom (as in `nn.rmsNormWeighted`), a silent shape-lie in general. A side effect of having both this and the scalar `HMul (T s) Float (T s)` instance: bare numeric literals are captured by the tensor-tensor instance, so write `w * (0.02 : Float)`, not `w * 0.02` (an `OfScientific (T …)` synthesis error otherwise); named `Float` bindings like GPT's `q_proj * scale` work unannotated.

### Shape transforms

```lean
reshape {s} (t : T s) (s' : Shape) : T s'                  -- the universal escape hatch
permute {s} (t : T s) (permutation : Array UInt64) : T (permuteShape s permutation)
transpose {s} (input : T s) (dim0 dim1 : UInt64) : T (transposeShape s ...)
unsqueeze / squeeze {s} (input : T s) (dim : Nat) : T (unsqueezeShape s dim) / ...
expand {s} (input : T s) (targetShape : Shape) : T targetShape
tensor_repeat {s} (input : T s) (repeats : Array UInt64) : T s
cat {s1 s2} (t1 : T s1) (t2 : T s2) (dim : Nat) : T (catShape s1 s2 dim)
eraseShape {s} (t : T s) : T #[]                           -- Tyr/Torch.lean:648
T.slice {s} (self : T s) (dim start stop step) : T s       -- shape-preserving slice
T.getOp {s} (self : T s) (idx : Int) : T (s[1:].toArray)   -- index first dim
T.to {s} (self : T s) (device : Device) : T s              -- device move
unbind {s} (t : T s) (dim : Nat := 0) : Array (T (unbindShape s dim))
stack1d {n k} (tensors : Array (T #[n])) (dim : Int64 := 0)
  : T (if dim == 0 then #[k, n] else #[n, k])
```

### Matmul family

```lean
nn.matmul {s1 s2} (a : T s1) (b : T s2) : T (matmulShape s1 s2)   -- generic, reshape-retagged
nn.mm       {m n k} (input : T #[m, k]) (mat2 : T #[k, n]) : T #[m, n]
nn.bmm      {b m n k} : T #[b, m, k] → T #[b, k, n] → T #[b, m, n]
nn.bmm4d    {b h m k n} : T #[b, h, m, k] → T #[b, h, k, n] → T #[b, h, m, n]
nn.matmul3d {batch seq k n} : T #[batch, seq, k] → T #[k, n] → T #[batch, seq, n]
linear  {m n b} (x : T #[b, m]) (M : T #[n, m]) : T #[b, n]              -- top-level
affine  ... (bias : T #[n]) : T #[b, n]
linear3d / affine3d : T #[batch, seq, in_dim] → T #[out_dim, in_dim] → T #[batch, seq, out_dim]
```

Head-layout helpers for attention: `nn.transpose2d`, `nn.transpose3d_12`, `nn.reshape_to_heads`, `nn.reshape_from_heads`, `nn.transpose_for_attention`, `nn.transpose_from_attention` (`Tyr/Torch.lean:594-620`).

### Reductions and scalar extraction

```lean
nn.sumAll / nn.meanAll / nn.maxAll / nn.minAll {s} (t : T s) : T #[]
nn.sumDim / nn.meanDim {s} (t : T s) (dim : Nat) (keepdim : Bool := false)
  : T (reduceShape s dim keepdim)
nn.cumsum {s} (input : T s) (dim : Int64) : T s
nn.argmax {s} (t : T s) (dim : UInt64) : T (reduceShape s dim.toNat false)
nn.item {s} (t : T s) : Float        nn.itemInt {s} (t : T s) : Int64
allclose {s} (a b : T s) (rtol : Float := 1e-05) (atol : Float := 1e-08) : Bool   -- top-level
```

### Autograd (`torch.autograd`)

```lean
grad {sx sy} (y : T sy) (x : T sx) (dy : T sy) : T sx          -- VJP, Tyr/Torch.lean:253
pullback (f : T sx → T sy) (x : T sx) (dy : T sy) : T sx       -- pure-Lean wrapper
backward {s} (output : T s) (grad_output : T s) : IO Unit      -- Tyr/Torch.lean:769
backwardLoss {s} (loss : T s) : IO Unit                        -- scalar loss, grad = 1.0
grad_of / zero_grad / detach / clone / retain_grad {s} (x : T s) : T s
set_requires_grad {s} (x : T s) (requires_grad : Bool) : T s
accumulate_grad / set_grad {s} (x grad : T s) : T s
is_leaf / has_grad_fn {s} (x : T s) : Bool
set_grad_enabled (enabled : Bool) : IO Unit ; is_grad_enabled : IO Bool
no_grad {α} (action : IO α) : IO α    -- try/finally restore, Tyr/Torch.lean:275
grad_grad {sx sy sz} (z y x grad_z grad_y) : T sx              -- second-order
```

The standard training idiom is `autograd.backwardLoss loss` followed by `autograd.grad_of param` (usually via `TensorStruct.grads`, see [tensorstruct.md](tensorstruct.md)). Leaf parameters are created with `set_requires_grad (detach t) true` since arithmetic results are non-leaves. Gradient clipping lives in `nn`: `clip_grad_norm_ (param) (max_norm) : IO Float`, `clip_grad_value_ (param) (clip_value) : IO Unit`.

### Neural network functional (`torch.nn`)

Activations: `sigmoid`, `silu`, `gelu`, `elu`, `tanh`, `softplus`, `leaky_relu`, `softmax (dim : Int32 := -1)`, `log_softmax (dim : Int := -1)`, `softmax_dim (dim : Int64)`, `softcap (cap := 15.0)`; pure-Lean composites `rmsNorm`, `rmsNormWeighted`, `reluSquared` (`Tyr/Torch.lean:1212-1229`).

Normalization: `layer_norm` (3D, shape-aware), `layer_norm'` (generic), fused `layer_norm_gelu`/`layer_norm_relu`/`layer_norm_silu`, `batch_norm`, `group_norm`, `instance_norm`.

Losses — all return `T #[]` scalars unless noted:

```lean
nn.cross_entropy {n c} (logits : T #[n, c]) (targets : T #[n]) : T #[]
nn.cross_entropy_none {n c} (logits : T #[n, c]) (targets : T #[n]) : T #[n]
nn.nll_loss / nn.nll_loss_none, nn.mse_loss, nn.l1_loss, nn.smooth_l1_loss,
nn.binary_cross_entropy (input target : T s) (reduction : String := "mean")
focal_loss, triplet_margin_loss   -- top-level simplified wrappers, Tyr/Torch.lean:1443,1462
```

Conv/pool (result shapes from the pure helpers above): `conv1d`, `conv1d_group_bias`, `conv2d`, `conv2d_bias`, `conv3d`, `conv_transpose1d_bias`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`. `conv_transpose2d` returns `T #[]` — its shape depends on all parameters and is computed at runtime (`Tyr/Torch.lean:366`).

Other: `embedding`/`embedding1d`, `dropout`/`dropout2d`/`dropout3d` (in `IO`), `tril`, `masked_fill`, `masked_select`, `masked_scatter`, `topk_values`, `multinomial`, `topKFilter`, `topPFilter` (sampling), `squeezeDim`.

### Attention

```lean
nn.scaled_dot_product_attention  {batch n_head seq head_dim}      -- 4D, causal default
nn.scaledDotProductAttentionGQA         -- K/V heads ≠ Q heads
nn.scaledDotProductAttentionGQAQKV      -- q_seq ≠ kv_seq (KV-cache decode)
nn.scaledDotProductAttentionGQAWindow   -- sliding window
nn.scaledDotProductAttentionGQAMask     -- [batch, seq] padding mask
nn.scaledDotProductAttentionGQAMaskQKV  -- [batch, q_seq, kv_seq] edge mask
nn.tyrFlashAttn4d                       -- Tyr flash attention; ThunderKittens H100
                                        -- kernels for validated shapes, SDPA
                                        -- fallback otherwise (Tyr/Torch.lean:1197)
```

All take `dropout_p`, `is_causal`, `enable_gqa` arguments and return `T #[batch, n_head, q_seq, head_dim]`.

### Comparison, masking, indexing

`lt`, `lt_scalar`, `gt`, `ge`, `eq`, `eq_scalar` (elementwise, boolean tensors), `where_`, `logical_not`, `logical_and`, `logicalOr` (De Morgan composite), `any : T s → Bool`, `falseMask`, `full_int`. Max/indices: `max_dim_3d : T #[d0,d1,d2] → T #[d0,d1] × T #[d0,d1]`, `topk`, `topk_2d`. Indexing: `indexSelect`, `index_select_1d`, `gather`, `scatter`, `scatter_add`, `scatter_2d`, `clamp`, `clampFloat`, `einsum`/`einsum2` (shape-erased), `interpolate`/`interpolate_scale` (shape-erased). Note: `max_dim` (`Tyr/Torch.lean:1283`) carries its own "signature is incorrect" warning — use `max_dim_3d`.

### Devices and runtime

```lean
cuda_is_available : IO Bool          mps_is_available : IO Bool
mps_is_available_stable (retries := 4) (delayMs := 250) : IO Bool  -- retry wrapper
getBestDevice : IO Device            -- MPS > CUDA > CPU, Tyr/Torch.lean:152
cuda_current_stream : IO UInt64      cuda_synchronize : IO Unit
manualSeed (seed : UInt64) : IO Unit       -- CPU + CUDA generators
get_live_tensors : IO UInt64               -- leak/debug probe
```

### Metadata and introspection (`Tyr/Basic.lean`)

```lean
T.runtimeShape {s} (t : T s) : Array UInt64        -- actual runtime shape
T.dtype / T.dtypeStr {s} (t : T s) : DType / String
T.device / T.deviceStr {s} (t : T s) : Device / String
T.getValues {s} (t : T s) (maxElements : UInt64 := 1000) : FloatArray
T.stats {s} (t : T s) : String                     -- JSON {min, max, mean, std}
T.toString / T.print ; T.shape {s} (_t : T s) : Shape := s   -- the static index
```

These exist mainly for the infoview widgets, but are the right tool for debugging and test assertions (`t.runtimeShape == #[2, 3]`).

### Data, signal, and media

`torch.data`: `loadU16Bin`/`binFileTokenCount` (uint16 token corpora), `loadF32Bin`, `fromInt64Array`, `fromFloatArray` (both `T #[]`), `saveTensor`/`loadTensor` (raw single-tensor files), `saveWav`/`wavBegin`/`wavAppend`/`wavFinalize`, `savePPMExplicit`, `slice1d`/`slice`/`slice2d`/`sliceScatter`, `toLong`, `tensorToUInt64Array*`, `tensorToFloatArray'`, `findBosPositions`, `resampleSoxrHQ` (libsoxr HQ resampling), `fileExists`. Apple-only media preprocessing: `loadImagePatchified`, `loadVideoPatchified`, `loadGemma4ImagePatchGrid` (shape-erased, need macOS media frameworks).

`torch.signal`: `hannWindow (n) : T #[n]`, `stft1d`, `istft1d`, `rfft1d` (all shape-erased outputs).

### SafeTensors (`torch.safetensors`)

```lean
opaque SafeTensorsHandle : Type
openHandle (path : String) : IO SafeTensorsHandle
loadFromHandle (handle) (name : String) (shape : Shape) : IO (T shape)
loadTensor (path name : String) (s : Shape) : IO (T s)
loadTensorSharded (dir name : String) (s : Shape) : IO (T s)
saveTensor (path name : String) (t : T s) : IO Unit
saveTensors (path) (entries : Array (String × T #[]))
            (metadata : Array (String × String) := #[]) : IO Unit
```

`*OnDevice` variants (`loadFromHandleOnDevice`, `loadTensorOnDevice`, `loadTensorShardedOnDevice`) load then move. The caller supplies the expected shape; it is trusted, not checked against the file header. For the elaboration-time type provider that generates shape-correct loaders from a SafeTensors file, see [serialization.md](../serialization.md).

### Rotary embeddings (`torch.rotary`)

```lean
computeFreqs / computeFreqsPure (seqLen headDim : UInt64) (base : Float := 10000.0)
  : IO (T #[seqLen, headDim/2] × T #[seqLen, headDim/2])   -- (cos, sin)
computeFreqsOnDevice / ...Pure  -- with explicit device
applyRotaryEmb {batch seq n_head head_dim}
  (x : T #[batch, seq, n_head, head_dim])
  (cos sin : T #[seq, head_dim / 2]) : T #[batch, seq, n_head, head_dim]
```

### Linear algebra (`torch.linalg`)

Matrix decompositions and functions for manifold optimization (`Tyr/Torch.lean:1489-1565`): `qr`, `qr_reduced`, `svd : T #[m,n] → T #[m, min m n] × T #[min m n] × T #[min m n, n]`, `svdvals`, `matrix_exp`, `matrix_log`, `inv`, `diag`, `diagflat`. Norms returning `Float`: `spectralNorm`, `nuclearNorm`, `maxRowNorm`, `l2Norm`, `frobeniusNorm`; `rowNorms : T #[n,d] → T #[n]`.

### The dynamic corner: `T #[]`

A growing set of APIs gives up static shapes entirely and works with `T #[]`: `einsum`, `interpolate`, `conv_transpose2d`, `squeezeDim`, `masked_select`, `cat_dyn`, the media loaders, `stft1d`/`istft1d`/`rfft1d`, `fromInt64Array`/`fromFloatArray`. Convert with `eraseShape` (down) and `reshape` (up, runtime-trusted). When you control the shapes, prefer the typed variants.

## Usage example

Reconstructed example (from `Examples/GPT/GPT.lean:87-115,197-207`, `Examples/GPT/Train.lean:110-145`, `Tests/TestAutoGrad.lean:232-268`):

```lean
import Tyr.Torch

open torch

/-- Leaf-parameter idiom: detach, then require grad (GPT.lean:87-89). -/
def makeLeafParam {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (autograd.detach t) true

/-- One classifier step: forward, scalar loss, backward, clip, read loss. -/
def step {batch inDim numClasses : UInt64}
    (w : T #[numClasses, inDim]) (x : T #[batch, inDim]) (y : T #[batch])
    : IO Float := do
  let logits := linear x w                    -- T #[batch, numClasses]
  let loss := nn.cross_entropy logits y       -- T #[], scalar (GPT.lean:207)
  autograd.backwardLoss loss                  -- accumulates ∂loss/∂w (Train.lean:125)
  let _ ← nn.clip_grad_norm_ w 1.0            -- in-place grad clip (Train.lean:86)
  let _grad : T #[numClasses, inDim] := autograd.grad_of w
  pure (nn.item loss)                         -- scalar to Float (Train.lean:136)

def main : IO Unit := do
  manualSeed 42
  let device ← getBestDevice                  -- MPS > CUDA > CPU
  let w0 ← randn #[10, 32] false device       -- IO; non-leaf after scaling
  let w := makeLeafParam (w0 * (0.02 : Float))  -- scalar HMul instance
  let x ← randn #[4, 32] false device
  let y : T #[4] := reshape (data.fromInt64Array #[1, 2, 3, 4]) #[4]
  let loss ← step w x y
  -- Evaluation under no_grad: graph building disabled, restored on exit
  autograd.no_grad do
    let logits := linear x w
    IO.println s!"loss={loss}, logits shape={logits.runtimeShape}, stats={logits.stats}"
```

Shape-level checking happens at elaboration: `nn.matmul (x : T #[2, 3]) (w : T #[3, 4])` has type `T (matmulShape #[2, 3] #[3, 4])`, which reduces to `T #[2, 4]` — a `[2,3] @ [4,3]` mistake fails to typecheck.

## Pitfalls

- `matmulShape` is total and returns `#[]` on invalid input; shape *compatibility* is not proven, only shape *arithmetic* is tracked. Bad contractions fail in C++ at runtime.
- `mul'`'s `HMul (T s1) (T s2) (T s2)` instance types the result as the right operand even when broadcasting would produce something else (`Tyr/Torch.lean:103-106`).
- `reshape`, and wrappers built on it (`matmul`, `cat`, `expand`, `sumDim`, …), re-tag shapes without runtime verification — the Lean-side shape helper is the source of truth and must match C++ behavior.
- `rand`/`randn`/`randint`/`dropout` are in `IO`; deterministic creation and most pure math are not.
- Naming is inconsistent (snake_case `mul_scalar`, `no_grad` vs. camelCase `manualSeed`, `eraseShape`, `zerosOn`); `focal_loss`'s docstring points at a nonexistent `nn.cross_entropy_loss` — the real names are `nn.cross_entropy`/`nn.cross_entropy_none` (`Tyr/Torch.lean:1442`).
- `torch.class differentiable` (`Tyr/Torch.lean:289`) is a near-unused experiment, not the AD system — for Tyr's actual autodiff see [autodiff.md](../autodiff.md).

## Related guides

- [typed.md](typed.md) — `Tensor (σ : StaticSpec)`: dtype and device policy in the type, checked constructors.
- [tensorstruct.md](tensorstruct.md) — PyTree-style traversal over parameter structures (`TensorStruct.grads`, `zeroGrads`).
- [utilities.md](utilities.md) — PRNG, logging, and the `#tensor`/`#module` infoview widgets that consume the metadata FFI.
- [../autodiff.md](../autodiff.md) — the `Tyr/AD` forward/reverse-mode system layered above `torch.autograd`.
- [../serialization.md](../serialization.md) — SafeTensors type provider, checkpoints, HuggingFace hub.
- [../ffi-and-build.md](../ffi-and-build.md) — the C++ side (`cc/src/tyr*.cpp`), libtorch linkage, build configuration.
- [../modules.md](../modules.md) — `Linear`, `RMSNorm`, and other modules built on these ops.
- [../getting-started.md](../getting-started.md), [../examples-and-testing.md](../examples-and-testing.md) — end-to-end training examples and the test suite.

Exhaustive per-symbol documentation for everything referenced here is generated by doc-gen4; see `docbuild/`.
