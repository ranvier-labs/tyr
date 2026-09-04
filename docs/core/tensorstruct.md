# TensorStruct: traversable tensor trees

`TensorStruct` is Tyr's pytree typeclass: one uniform interface for traversing,
transforming, and combining the nested parameter/state structures that make up a real
model. It plays the role JAX's PyTrees play, specialized for Tyr's shape-indexed
`T s`. Use it whenever you write code that must work over *any* model's parameters —
training loops, optimizers, checkpointing, device moves — instead of hand-writing
per-model plumbing. Everything in this module lives in `namespace torch`.

Source: `Tyr/TensorStruct.lean`. Deriving handler and leaf instances:
`Tyr/Module/Derive.lean`.

## Architecture and main abstractions

### The typeclass

```lean
class TensorStruct (α : Type) where                -- Tyr/TensorStruct.lean:247
  map     : (∀ {s}, T s → T s) → α → α
  mapM    {m : Type → Type} [Monad m] : (∀ {s}, T s → m (T s)) → α → m α
  zipWith : (∀ {s}, T s → T s → T s) → α → α → α
  fold    {β : Type} : (∀ {s}, T s → β → β) → β → α → β
```

Four operations over the tensor leaves of a structure:

- `map` — pure tensor-to-tensor transform at every leaf.
- `mapM` — monadic variant; works in `IO` (device moves, loading), since the leaf
  function is rank-2 polymorphic over the shape `s`.
- `zipWith` — combines two structures of identical shape tree.
- `fold` — reduces all leaves into a summary value.

The leaf function is universally quantified over the shape index (`∀ {s}`), so a
single traversal handles leaves of different static shapes in one pass. The shape
index is erased at runtime; see [tensors](tensors.md) for what `T s` actually is.

### Wrappers that control traversal

```lean
structure Static (α : Type) where                  -- Tyr/TensorStruct.lean:136
  val : α
  deriving Repr, BEq, Hashable

structure Frozen (s : Shape) where                 -- Tyr/TensorStruct.lean:147
  tensor : T s
```

- `Static α` — non-tensor metadata (configs, hyperparameters, names). Completely
  skipped by every traversal: `map` returns it unchanged, `fold` contributes nothing.
  A `Coe α (Static α)` instance lets you write `name := "layer1"` directly.
- `Frozen s` — a tensor leaf that participates in the forward pass but is intended
  to be non-trainable. It **is** traversed like any other tensor, so `grads`,
  `scale`, `add`, and friends touch it. Nothing in this module excludes it from
  gradient updates — if you need that, filter it in your own leaf function. Helpers:
  `Frozen.map`, `Frozen.get`; a `Coe (T s) (Frozen s)` instance lets you write
  `runningMean := zeros #[n]` directly.

### `Vector n α`: length-indexed arrays

```lean
structure Vector (n : Nat) (α : Type) where        -- Tyr/TensorStruct.lean:169
  data : Array α
  size_eq : data.size = n
```

`Vector` carries its length in the type, which makes `zipWith` statically safe.
`Array` and `List` can't do that, so their `zipWith` instances check lengths at
runtime and **panic on mismatch** (`Tyr/TensorStruct.lean:287` and `:300`). If a
structure zips model parameters against gradients or optimizer state — i.e. always —
a length mismatch is a bug you want caught at compile time. Prefer
`Vector numLayers (BlockParams …)` over `Array (BlockParams …)` when the count is
fixed.

### Container instances

Provided in `Tyr/TensorStruct.lean`:

| Type | Traversal behavior |
|---|---|
| `Array α` | elementwise; `zipWith` panics on size mismatch |
| `List α` | elementwise; `zipWith` panics on length mismatch |
| `Vector n α` | elementwise; `zipWith` always safe |
| `Option α` | traverses `some`; `zipWith` of mismatched variants yields `none` (no panic) |
| `Static α` | skipped entirely (requires `Inhabited α`) |
| `Frozen s` | traversed as one tensor leaf |
| `Bool`, `Float`, `UInt8`, `UInt64`, `Nat`, `Int`, `String` | no-tensor leaves, passed through |

Two important instances live elsewhere, in `Tyr/Module/Derive.lean`:

```lean
instance {s : Shape} : TensorStruct (T s)          -- Tyr/Module/Derive.lean:78
instance [TensorStruct α] [TensorStruct β] : TensorStruct (α × β)  -- :89
```

The single-tensor leaf instance is what terminates every traversal. Note the import
consequence: `import Tyr.TensorStruct` alone gives you neither the `T s` leaf
instance nor the deriving handler — use `import Tyr.Module.Derive` (or just
`import Tyr`, which re-exports both, `Tyr.lean:7,14`).

### Deriving instances

For a structure whose fields all have `TensorStruct` instances, the handler in
`Tyr/Module/Derive.lean` generates the instance field-by-field:

```lean
import Tyr.Module.Derive

structure LinearParams (inDim outDim : UInt64) where
  weight : T #[outDim, inDim]
  bias : T #[outDim]
  name : Static String
  deriving TensorStruct
```

The generated code just delegates each field to its own instance:
`map f x := { weight := TensorStruct.map f x.weight, … }`. It handles parametric
structures (shape and type parameters become instance binders) but assumes every
field already has an instance; custom traversal policies require a manual instance.

`class Model (α : Type)` (`Tyr/TensorStruct.lean:243`) is a marker class whose
deriving handler (`Tyr/Module/Derive.lean:202`) bundles three derivations at once:
`TensorStruct`, `ToTensorStructSchema`, and `TensorStructFlatten` (the latter two
come from `Tyr/AD/TensorStructSchema`). Use `deriving Model` when a parameter type
also needs AD/serialization support, plain `deriving TensorStruct` otherwise.

## Key APIs

Tree utilities in `namespace TensorStruct` (`Tyr/TensorStruct.lean:390-440`). All
take `[TensorStruct α]`:

| Function | Signature | What it does |
|---|---|---|
| `count` | `(model : α) : Nat` | number of tensor leaves |
| `grads` | `(model : α) : α` | replace each leaf by its accumulated gradient (`autograd.grad_of`) |
| `zeroGrads` | `(model : α) : α` | zero every leaf's gradient |
| `detach` | `(model : α) : α` | detach every leaf from the computation graph |
| `requiresGrad` | `(model : α) (b : Bool) : α` | set `requires_grad` on every leaf |
| `makeLeafParams` | `(model : α) : α` | detach + `requires_grad = true`; makes leaves trainable roots |
| `scale` | `(model : α) (s : Float) : α` | multiply every leaf by a scalar |
| `add` / `sub` | `(a b : α) : α` | elementwise add/subtract two matching trees |
| `all` / `any` | `(p : ∀ {s}, T s → Bool) (model : α) : Bool` | predicate over leaves |

`Vector` operations (`Tyr/TensorStruct.lean:173-225`):

| Function | Signature |
|---|---|
| `Vector.toArray` | `Vector n α → Array α` |
| `Vector.get` | `(v : Vector n α) (i : Fin n) → α` — bounds-safe via `Fin` |
| `Vector.map` / `Vector.mapM` | size-preserving (proof-carrying) |
| `Vector.zipWith` | `(α → β → γ) → Vector n α → Vector n β → Vector n γ` |
| `Vector.foldl` | `(β → α → β) → β → Vector n α → β` |
| `Vector.replicate` | `(n : Nat) → α → Vector n α` |
| `Vector.empty` | `Vector 0 α` |
| `Vector.push` | `Vector n α → α → Vector (n + 1) α` — increments the type-level length |

`Repr` and `Inhabited` instances exist for `Vector n α` (the latter needs
`Inhabited α` and replicates the default).

## How the rest of Tyr consumes it

Brief orientation only — details live in the linked guides.

- **Modules.** `Module`/`ModuleIO`/`ModuleCtx` in `Tyr/Module/Core.lean` each carry a
  `[toTensorStruct : TensorStruct M]` constraint (`Tyr/Module/Core.lean:77-79`), so
  any module automatically supports generic parameter transforms. See
  [modules](../modules.md).
- **Optimizers.** `Tyr/Optim.lean` is Optax-style: a
  `GradientTransformation α S` (`Tyr/Optim.lean:105`) is a pair of `init : α → S` and
  `update : α → α → S → α × S`, written generically against `TensorStruct`. Adam
  state (`ScaleByAdamState`, `TraceState`, `ChainState`) has manual `TensorStruct`
  instances, so optimizer state is itself a traversable tree mirroring the
  parameters. `Optim.step` (`Tyr/Optim.lean:223`) and `Optim.adamw`
  (`Tyr/Optim.lean:190`) are the usual entry points. See
  [optimization](../optimization.md).
- **Checkpointing.** `torch.checkpoint.saveParams`/`loadParams`
  (`Tyr/Checkpoint.lean:118,138`) save/load any `TensorStruct` tree as sequentially
  numbered `.pt` files: save folds with an `IO.Ref` counter, load `mapM`s over a
  template structure (which supplies the shapes) and finishes with
  `makeLeafParams`. See [serialization](../serialization.md).
- **Autodiff.** The utilities are thin wrappers over `torch.autograd` leaf
  operations (`grad_of`, `zero_grad`, `detach`, `set_requires_grad`). See
  [autodiff](../autodiff.md).

## Usage example

Reconstructed example (from `Examples/GPT/GPT.lean`, `Examples/GPT/Train.lean`,
`Examples/NanoChat/ModdedTrain.lean`):

```lean
import Tyr
import Tyr.Module.Derive

open torch

-- 1. Parameter trees are plain structures with shape-indexed fields.
--    (Examples/GPT/GPT.lean:50,75)
structure BlockParams (n_embd : UInt64) where
  ln1_weight : T #[n_embd]
  ln1_bias   : T #[n_embd]
  q_proj     : T #[n_embd, n_embd]
  c_proj     : T #[n_embd, n_embd]
  deriving TensorStruct

structure GPTParams (cfg : Config) where
  wte        : T #[cfg.vocab_size, cfg.n_embd]
  wpe        : T #[cfg.block_size, cfg.n_embd]
  blocks     : Array (BlockParams cfg.n_embd)   -- Array instance recurses
  ln_f_weight : T #[cfg.n_embd]
  ln_f_bias   : T #[cfg.n_embd]
  deriving TensorStruct

-- 2. Generic device move over any tree; mapM runs in IO.
--    (Examples/NanoChat/ModdedTrain.lean:34)
def moveToDevice [TensorStruct α] (x : α) (device : Device) : IO α := do
  let moved ← TensorStruct.mapM (fun t => pure (t.to device)) x
  pure (TensorStruct.makeLeafParams moved)

-- 3. A training step is tree-generic plumbing around the model-specific loss.
--    (Examples/GPT/Train.lean:110)
def trainStep {modelCfg : Config} {batch seq : UInt64}
    (params : GPTParams modelCfg)
    (optState : Optim.AdamWState (GPTParams modelCfg))
    (x : T #[batch, seq]) (y : T #[batch, seq])
    (lr : Float)
    : IO (GPTParams modelCfg × Optim.AdamWState (GPTParams modelCfg) × Float) := do
  let params := TensorStruct.zeroGrads params   -- clear stale gradients
  let lossT ← gpt.loss params x y true          -- model-specific forward
  autograd.backwardLoss lossT                   -- populate .grad on every leaf
  let lossVal := nn.item lossT
  let grads := TensorStruct.grads params        -- extract gradient tree
  let opt := Optim.adamw (lr := lr)
  let (params', optState') := Optim.step opt params grads optState
  return (params', optState', lossVal)
```

The pattern to copy: parameter shapes live in the structure's type indices, the
`TensorStruct` instance is derived, and everything except the forward pass
(zeroing, gradient extraction, optimizer update, device moves, checkpointing) is
written once, generically, against the class.

## Related guides

- [Tensors and the raw FFI surface](tensors.md) — `T s`, `Shape`, `torch.autograd`
- [Typed tensors](typed.md) — dtype/device-tracked facade over the same handles
- [Core utilities](utilities.md) — PRNG, logging, widgets
- [Modules](../modules.md) — `Module` classes and the deriving machinery
- [Optimization](../optimization.md) — Optax-style transformations over trees
- [Serialization](../serialization.md) — checkpoints and SafeTensors
- [Autodiff](../autodiff.md) — gradient computation underneath `grads`/`zeroGrads`
- [Getting started](../getting-started.md)

---

This is a guide, not an exhaustive reference. Full per-symbol documentation for
`torch.TensorStruct`, `Vector`, and the container instances is generated by
doc-gen4 (see `docbuild/`).
