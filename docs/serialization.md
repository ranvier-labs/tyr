# Weights and serialization

## Purpose and when to use

Tyr has three cooperating serialization subsystems, each aimed at a different job:

- `Tyr/SafeTensors/` reads HuggingFace `.safetensors` checkpoints — single files,
  sharded directories, and `model.safetensors.index.json` layouts — and can generate
  shape-typed Lean loaders from a checkpoint at elaboration time. Use this for
  pretrained weight loading.
- `Tyr/Hub.lean` resolves a local directory or a HuggingFace `repo_id` to a local
  model directory, downloading and caching as needed. Use this as the front door
  for `loadFromPretrained`-style APIs.
- `Tyr/Checkpoint.lean` persists training state — any `TensorStruct` parameter tree
  plus mirrored Adam-style optimizer state — for resume. Use this for Tyr's own
  training checkpoints, not for HF interop.

Everything below lives in three namespaces: `torch.safetensors`, `torch.Hub`, and
`torch.checkpoint`.

## Architecture and main abstractions

### FFI layer

The bottom layer is a handful of `@[extern]` ops in `Tyr/Torch.lean:978-1050`
(namespace `torch.safetensors`), backed by the `lean_torch_safetensors_*` C++
functions under `cc/`:

```lean
opaque SafeTensorsHandle : Type
opaque openHandle (path : @& String) : IO SafeTensorsHandle
opaque loadFromHandle (handle : @& SafeTensorsHandle) (name : @& String) (shape : Shape) : IO (T shape)
opaque loadTensor (path : @& String) (name : @& String) (s : Shape) : IO (T s)
opaque loadTensorSharded (dir : @& String) (name : @& String) (s : Shape) : IO (T s)
opaque saveTensor {s : Shape} (path : @& String) (name : @& String) (t : @& T s) : IO Unit
def saveTensors (path : String) (entries : Array (String × T #[]))
    (metadata : Array (String × String) := #[]) : IO Unit
```

Each loader has an `...OnDevice` wrapper (`loadTensorOnDevice`,
`loadTensorShardedOnDevice`, `loadFromHandleOnDevice`) that moves the result to a
target `Device`, defaulting to CPU. Loading is name-based and shape-checked: you
pass the expected `Shape` and get back a `T s` whose type carries it. `openHandle`
parses a file's header once so repeated `loadFromHandle` calls skip re-parsing.

### Runtime schema introspection

`Tyr/SafeTensors/Schema.lean` builds a metadata surface on top of the FFI. The two
central types (`Tyr/SafeTensors/Schema.lean:27-56`):

```lean
structure TensorSchema where
  name : String
  dtype : DType
  shape : Shape
  /-- For directory sources this is the shard filename; for single-file sources it is empty. -/
  sourceFile : String := ""

structure Schema where
  source : String
  sourceIsDirectory : Bool
  tensors : Array TensorSchema
```

`introspect (source : String) : IO Schema` (`Tyr/SafeTensors/Schema.lean:413`) is
the central entry point. It accepts:

- a single `.safetensors` file — parses the 8-byte little-endian header size and
  the JSON header itself;
- a directory of `.safetensors` shards — parses every shard header;
- a directory with `model.safetensors.index.json` — follows the `weight_map`,
  rejects unsafe shard paths (absolute paths, `\` prefixes, Windows drive letters,
  `.`/`..` segments; `Tyr/SafeTensors/Schema.lean:345-350`), and cross-checks every
  index entry against the actual shard headers. Only index-mapped tensors are
  exposed;
- a `.schema.json` snapshot — a previously saved `Schema`, loaded via
  `loadSnapshot` without touching any weight files.

Results are sorted by name and duplicate tensor names are rejected.
`saveSnapshot`/`loadSnapshot` persist a `Schema` as JSON; this is how
`Tyr/Model/KittenTTS/kokoro_v1.schema.json` lets the type provider run at build
time without a multi-GB checkpoint on disk.

The schema layer also provides contract-checked typed loaders that return a
dtype-indexed `DTensor` (see [Typed tensors](core/typed.md)):

```lean
def loadTensorWithContract (path : String) (name : String)
    (contract : TensorContract) (device : Device := Device.CPU)
    : IO (DTensor contract.spec.shape contract.spec.dtype)
```

plus `loadFromHandleWithContract`, `loadTensorShardedWithContract`, `...WithSpec`
variants (which fix `devicePolicy := .exact device`), and `...WithSchema` variants
that take a discovered `TensorSchema`. Checking happens after the load via
`TensorContract.check` against the runtime shape/dtype/device
(`Tyr/SafeTensors/Schema.lean:62-82`).

### Try-load helpers

`Tyr/SafeTensors/Load.lean` is a thin error-handling layer over the FFI, used by
the per-model `Weights` modules (`Tyr/Model/Qwen35/Weights.lean`,
`Tyr/Model/Gemma4/Weights.lean`, etc.). The two flavors matter:

- `tryLoadTensor` / `tryLoadTensorSharded` return `none` on **any** failure —
  missing tensor, shape mismatch, unreadable file.
- `tryLoadOptionalTensor` / `tryLoadOptionalTensorSharded` return `none` only when
  the tensor is absent (detected by the `isMissingTensorError` message heuristic,
  `Tyr/SafeTensors/Load.lean:39-41`); other errors are rethrown.

On top of these sit candidate-name fallbacks — `tryLoadTensorCandidates`,
`tryLoadTensorShardedCandidates`, and the throwing `loadTensorCandidates` /
`loadTensorShardedCandidates` — which try an array of alternative names (model
families use `pushUnique` to build prefix-variant lists, e.g. with and without a
`language_model.` prefix).

### The `safetensors_type_provider` command

`Tyr/SafeTensors/TypeProvider.lean` defines a command that introspects a
checkpoint **at elaboration time** and generates typed loaders from it:

```lean
safetensors_type_provider "path/to/model.safetensors" as MyWeights
safetensors_type_provider "path/to/sharded_dir" as MyWeights
safetensors_type_provider "path/to/model.schema.json" as MyWeights  -- snapshot
```

For each tensor, inside `namespace MyWeights`, it emits (tensor `linear.weight`
shown; the declaration base name is sanitized and lowercased to `linear_weight`):

- `abbrev linear_weightShape : Shape` — the static shape;
- `def linear_weightSpec : TensorSchema` and `def linear_weightTensorSpec : TensorSpec`;
- `def load_linear_weight (source : String := defaultSource) : IO (T linear_weightShape)`;
- checked variants `load_linear_weightTyped` / `...TypedOnDevice` returning
  `IO (DTensor linear_weightShape ...)`;
- for single-file sources additionally `load_linear_weightFromHandle` and
  `load_linear_weightTypedFromHandle[OnDevice]` taking a `SafeTensorsHandle`.

Tensor names split on `.` form a hierarchy. Named segments become fields of
generated `structure Weights<Segment>` records; a uniform run of numeric segments
(`layers.0.*`, `layers.1.*`, ...) becomes an `Array` field; a non-uniform run
falls back to `i0`, `i1`, ... fields; non-contiguous numeric indices are a
compile-time error (`validateIndexedContiguous`, `Tyr/SafeTensors/TypeProvider.lean:265`).
The top level of the generated namespace contains:

- `abbrev Weights` — the full hierarchical record;
- `def loadAll (source : String := defaultSource) : IO Weights`;
- `def loadAllFromHandle (handle : SafeTensorsHandle) : IO Weights` (single-file sources);
- subtree namespaces with their own `load` / `loadFromHandle` — e.g.
  `MyWeights.model.layers.load`;
- metadata: `defaultSource`, `sourceIsDirectory`, `tensorCount`, `schema`,
  `hasTensor`, `find?`, `fieldToTensorName`.

Generated code is rendered as strings and re-elaborated via
`Parser.runParserCategory` (`Tyr/SafeTensors/TypeProvider.lean:116-124`), with all
references `_root_.`-qualified. KittenTTS is the in-tree consumer: the provider
runs against a checked-in schema snapshot (`Tyr/Model/KittenTTS/Checkpoint.lean:14`)
and `Tyr/Model/KittenTTS/Weights.lean:784-792` loads subtrees from one open handle.

### HuggingFace Hub resolution

`Tyr/Hub.lean` (namespace `torch.Hub`) is deliberately small and depends only on
`Lean.Data.Json` and an external `curl` binary — no `Tyr.Torch` import. The flow:

```lean
structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := defaultCacheDir  -- "~/.cache/huggingface/tyr-models"
  includeTokenizer : Bool := true

def resolvePretrainedDir (source : String) (opts : DownloadOptions := {})
    (tokenizerFiles : Array String := #[]) : IO String
```

`resolvePretrainedDir` resolves in order (`Tyr/Hub.lean:230-244`):

1. an existing local directory is returned as-is;
2. the standard HF hub cache is searched (`findCachedSnapshot?` scans
   `$HF_HOME/hub` or `~/.cache/huggingface/hub` for
   `models--<org>--<repo>` snapshots containing `config.json` plus weights);
3. otherwise files are downloaded into the Tyr cache
   `<cacheDir>/<org>__<repo>/<revision>`: `config.json` (required), weights via
   `ensureModelWeights` (prefers `model.safetensors.index.json` + its shards,
   falls back to a single `model.safetensors`), tokenizer files best-effort.

Downloads shell out to `curl -fL --retry 3` with `.tmp`-file resume and an
optional `HF_TOKEN` bearer header (`Tyr/Hub.lean:89-118`). The generic entry point
used by every `Tyr/Model/*/Pretrained.lean` is:

```lean
def loadModelFromPretrained
    {Cfg : Type} {Model : Cfg → Type}
    (source : String) (revision : String) (cacheDir : String)
    (tokenizerFiles : Array String)
    (loadConfig : String → Cfg → IO Cfg) (defaults : Cfg)
    (loadSharded : String → (cfg : Cfg) → IO (Model cfg))
    (loadSingle : String → (cfg : Cfg) → IO (Model cfg))
    : IO (Sigma (fun cfg => Model cfg))
```

It resolves the directory, loads the config, picks sharded vs single-file via
`detectWeightLayout`, and returns the config and model as a dependent pair.

### Training checkpoints

`Tyr/Checkpoint.lean` (namespace `torch.checkpoint`) is model-agnostic
persistence over the `TensorStruct` class (see
[TensorStruct](core/tensorstruct.md)):

```lean
structure CheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  optimCount : Nat := 0

def saveParams [TensorStruct α] (params : α) (dir : String)
    (namePrefix : String := "param") (log : Handlers := {}) : IO Unit
def loadParams [TensorStruct α] (template : α) (dir : String)
    (namePrefix : String := "param") (log : Handlers := {}) : IO α
```

A checkpoint directory contains:

- `{namePrefix}_{i}.pt` — one libtorch-pickle file per tensor, numbered by
  `TensorStruct.fold` traversal order, written via `torch.data.saveTensor`
  (`Tyr/Torch.lean:884`). **No tensor names, shapes, or dtypes are persisted**;
  `loadParams` walks a template value and pulls shapes from it
  (`Tyr/Checkpoint.lean:138-154`). Save and load must therefore traverse the same
  structure in the same order.
- `meta.txt` — flat `key=value` lines for `CheckpointMeta`
  (`saveCheckpointMeta`/`loadCheckpointMeta`).
- `optim_mu_{i}.pt`, `optim_nu_{i}.pt`, `optim_count.txt` — Adam-style mirrored
  trees written by `saveOptimizerState` under separate prefixes.

This is a Tyr-native format, not interchangeable with HF weights — note it is
libtorch `.pt` pickle even though a safetensors writer (`safetensors.saveTensors`)
exists for interop.

## Key APIs

### SafeTensors schema (`torch.safetensors`, `Tyr/SafeTensors/Schema.lean`)

| API | Signature | Notes |
| --- | --- | --- |
| `introspect` | `String → IO Schema` | file, shard dir, index dir, or `.schema.json` snapshot |
| `Schema.find?` | `Schema → String → Option TensorSchema` | exact-name lookup |
| `saveSnapshot` / `loadSnapshot` | `String → Schema → IO Unit` / `String → IO Schema` | persist/restore a schema |
| `TensorSchema.toSpec` | `TensorSchema → TensorSpec` | shape + dtype |
| `TensorSchema.contract` | `(role := .parameter) → (devicePolicy := .any) → TensorContract` | boundary contract |
| `loadTensor[Sharded]WithContract` | `... → IO (DTensor contract.spec.shape contract.spec.dtype)` | checked load from file/dir |
| `loadFromHandleWithContract` | `SafeTensorsHandle → ... → IO (DTensor ...)` | checked load from open handle |
| `...WithSpec`, `...WithSchema` | same shapes | spec/schema conveniences |

### Try-load helpers (`Tyr/SafeTensors/Load.lean`)

| API | Returns `none` when |
| --- | --- |
| `tryLoadTensor` / `tryLoadTensorSharded` | any error occurs |
| `tryLoadOptionalTensor` / `tryLoadOptionalTensorSharded` | tensor is missing (other errors rethrown) |
| `tryLoadTensorCandidates` / `tryLoadTensorShardedCandidates` | no candidate name loads |
| `loadTensorCandidates` / `loadTensorShardedCandidates` | throwing variants of the above |

All take `(path|dir) (name[s]) (s : Shape) (device : Device := Device.CPU)` and
return `IO (Option (T s))` resp. `IO (T s)`.

### Hub (`torch.Hub`, `Tyr/Hub.lean`)

| API | Signature | Notes |
| --- | --- | --- |
| `defaultCacheDir` | `String` | `~/.cache/huggingface/tyr-models` |
| `resolvePretrainedDir` | `String → DownloadOptions → Array String → IO String` | local dir, HF cache, or download |
| `findCachedSnapshot?` | `String → (revision : String := "main") → IO (Option String)` | standard HF hub layout |
| `detectWeightLayout` | `String → IO Bool` | `true` = sharded |
| `shardFilesFromIndexFile` | `String → IO (Array String)` | unique shards from `weight_map` |
| `ensureModelWeights` / `ensureTokenizerFiles` | `... → IO Unit` | explicit download steps |
| `loadModelFromPretrained` | see signature above | generic `from_pretrained` |

### Checkpoint (`torch.checkpoint`, `Tyr/Checkpoint.lean`)

| API | Signature | Notes |
| --- | --- | --- |
| `saveParams` / `loadParams` | see above | positional `{prefix}_{i}.pt` tree |
| `saveCheckpoint` | `(params : α) → (iteration : Nat) → (bestValLoss trainLoss : Float) → (dir : String) → ... → IO Unit` | params + `meta.txt` |
| `loadCheckpoint` | `(template : α) → (dir : String) → ... → IO (α × CheckpointMeta)` | |
| `saveOptimizerState` | `(mu nu : α) → (count : Nat) → (dir : String) → ... → IO Unit` | mirrored Adam trees |
| `loadOptimizerState` | `(template : α) → (dir : String) → ... → IO (α × α × Nat)` | |
| `checkpointExists` / `optimStateExists` | `String → IO Bool` | probe for `meta.txt` / `optim_count.txt` |
| `saveCheckpointMeta` / `loadCheckpointMeta` | `CheckpointMeta → String → IO Unit` / `String → IO CheckpointMeta` | `meta.txt` only |

## Usage examples

Reconstructed example (from `Tests/TestSafeTensorsTypeProvider.lean:13-219` and
`launch/demos/safetensors_schema.lean`):

```lean
import Tyr.SafeTensors

open torch

-- Elaboration-time introspection; the source must exist at build time.
safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as SingleSafe
safetensors_type_provider "Tests/fixtures/safetensors/sharded" as ShardedSafe

def demo : IO Unit := do
  -- Per-tensor loaders; shape is checked against the generated Shape abbrev.
  let w ← SingleSafe.load_linear_weight            -- T #[2, 3]
  -- Dtype-indexed checked loader.
  let wTyped ← SingleSafe.load_linear_weightTyped  -- DTensor #[2, 3] .Float32
  -- Full hierarchical record: fields follow tensor names split on '.'.
  let weights ← ShardedSafe.loadAll
  IO.println s!"{weights.embed.weight.runtimeShape}"   -- #[2, 2]
  -- Subtree namespace loader.
  let embed ← ShardedSafe.embed.load
  -- Metadata.
  IO.println s!"{ShardedSafe.tensorCount} tensors"     -- 2
  IO.println s!"{ShardedSafe.proj_biasSpec.sourceFile}" -- "part2.safetensors"
```

Reconstructed example (from `Tyr/Model/Qwen35/Pretrained.lean:54-69`):

```lean
/-- Load a text Qwen3.5 checkpoint from local dir or HF repo id. -/
def Qwen35ForCausalLM.loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen35_9B)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => Qwen35ForCausalLM cfg)) := do
  Hub.loadModelFromPretrained
    source revision cacheDir hub.tokenizerFiles
    (fun modelDir cfg => Config.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => Qwen35ForCausalLM.loadSharded modelDir cfg device)
    (fun path cfg => Qwen35ForCausalLM.load path cfg device)

-- `loadFromPretrained "Qwen/Qwen3.5-0.8B"` downloads on first use;
-- `loadFromPretrained "./my-local-qwen-dir"` uses the directory directly.
```

Reconstructed AdamW example (from `Examples/TrainGPT.lean`):

```lean
open torch.checkpoint

-- End of training: params + meta, then optimizer state into the same dir.
saveCheckpoint finalParams trainCfg.maxIters bestValLoss 0.0 "checkpoints/gpt"
saveOptimizerState optState.mu optState.nu optState.count "checkpoints/gpt"

-- Resume: the template (freshly initialized params) supplies the shapes.
def resume (initParams : α) [TensorStruct α] : IO (α × CheckpointMeta) := do
  let (params, meta) ← loadCheckpoint initParams "checkpoints/gpt"
  if ← optimStateExists "checkpoints/gpt" then
    let (mu, nu, count) ← loadOptimizerState params "checkpoints/gpt"
    IO.println s!"resumed optimizer at count={count}"
  pure (params, meta)
```

The molecule BranchingFlows executable uses the same parameter checkpoint API,
but stores its Julia-compatible Muon momentum leaves under
`optim_muon_momentum_*` and its step in `optim_muon_count.txt`; resume requires
both files to restore optimizer state.

## Caveats

- The type provider introspects at elaboration time: the checkpoint (or a
  `.schema.json` snapshot) must exist on the build machine, and introspection
  failures surface as a generic `safetensors_type_provider failed while
  introspecting source ...` error (`Tyr/SafeTensors/TypeProvider.lean:628-632`).
  `introspect` reads each shard fully into memory to parse its header
  (`Tyr/SafeTensors/Schema.lean:272-273`), so snapshots also save build time on
  large checkpoints.
- Generated `defaultSource` is the elaboration-time path string. Every generated
  loader takes `source := defaultSource`, so pass the runtime path explicitly
  when the weights live elsewhere (as KittenTTS does with `openHandle` +
  `loadFromHandle`).
- Tyr checkpoint dirs are positional and template-driven: no names, shapes, or
  dtypes on disk. Refactoring a parameter structure invalidates old checkpoints.

## Related guides

- [Core tensors](core/tensors.md) — `T s`, `Shape`, `DType`, `Device`
- [Typed tensors](core/typed.md) — `TensorSpec`, `TensorContract`, `DTensor`
- [TensorStruct](core/tensorstruct.md) — the traversal class behind checkpointing
- [Optimization](optimization.md) — the Adam state trees being persisted
- [LLM models](models/llms.md), [Audio and speech models](models/audio-speech.md) —
  per-family `Weights`/`Pretrained` modules built on these APIs
- [FFI and build](ffi-and-build.md) — the `lean_torch_safetensors_*` C++ side

Exhaustive per-symbol documentation is generated by doc-gen4 (see `docbuild/`);
this chapter is a guide, not a symbol dump.
