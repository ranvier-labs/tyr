# Monte Carlo tree search

`Tyr/Mctx/` is a pure-Lean port of the policy side of DeepMind's JAX `mctx`
library: MuZero-style PUCT search, an AlphaZero variant with persistent
subtrees, and Gumbel MuZero with sequential halving. `Tyr/MctxDag/` is a second
backend whose search structure is a DAG with a transposition table, so
different action sequences that reach the same state share a single node. Use
these modules for discrete planning over a learned or hand-written dynamics
model; everything is `Array Float`-based — no tensors and no libtorch FFI are
involved. The main in-tree consumer is the AlphaGrad vertex-elimination
planner (`Tyr/AD/Elim/AlphaGradMctx.lean`) with demos and trainers under
`Examples/AlphaGradPort/`.

Both stacks live in the `torch.mctx` / `torch.mctxdag` namespaces (a naming
leftover; the code is tensor-free). `Tyr.lean` re-exports `Tyr.Mctx` but not
`Tyr.MctxDag` — import the DAG backend explicitly.

```lean
import Tyr.Mctx      -- tree backend: torch.mctx
import Tyr.MctxDag   -- DAG backend: torch.mctxdag
```

## Architecture and main abstractions

### Model interface

The environment is injected as functions over an opaque embedding type `S`
and a params type `P` (`Tyr/Mctx/Base.lean`). You supply a root evaluation
(prior + value + embedding) and a recurrent dynamics step:

```lean
structure RootFnOutput (S : Type) where
  priorLogits : Array Float
  value : Float
  embedding : S

structure RecurrentFnOutput where
  reward : Float
  discount : Float
  priorLogits : Array Float
  value : Float

abbrev RecurrentFn (P S : Type) :=
  P → UInt64 → Action → S → RecurrentFnOutput × S
```

`Action` is `Nat`, the `UInt64` is a manually threaded RNG key (the search
derives per-simulation keys as `rngKey + sim + 1`). Search returns a
`PolicyOutput TreeType` carrying the chosen `action`, the `actionWeights`
training target, and the `searchTree` itself; `Tree.summary` exposes a
`SearchSummary` with root `visitCounts`, `visitProbs`, `value`, and `qvalues`.

### Search tree

`Tree S E` (`Tyr/Mctx/Tree.lean:7`) is a flat, fixed-capacity store — one
array per node attribute (`nodeVisits`, `rawValues`, `nodeValues`, `parents`,
`embeddings`) and one nested array per node-per-action edge attribute
(`childrenIndex`, `childrenPriorLogits`, `childrenVisits`, `childrenRewards`,
`childrenDiscounts`, `childrenValues`). Capacity is `numSimulations + 1`, so
the whole search is allocation-free after instantiation. Sentinels:
`ROOT_INDEX = 0`, `NO_PARENT = -1`, `UNVISITED = -1`. The `E` parameter is
policy-specific extra data — Gumbel MuZero uses it to stash the root Gumbel
sample (`GumbelMuZeroExtraData.rootGumbel`); the other policies use `Unit`.

Useful tree operations:

- `Tree.qvalues tree nodeIndex` — `r + γ · v` per action.
- `Tree.summary tree` — root statistics for policies.
- `resetSearchTree tree` — wipe a tree back to empty, keeping capacity.
- `getSubtree tree childAction` (`Tyr/Mctx/Tree.lean:133`) — extract the
  subtree under a root action with remapped indices; this is how AlphaZero
  reuses search across environment steps.

### Search loop

`search` (`Tyr/Mctx/Search.lean:225`) instantiates a tree from the root and
hands off to `searchWithTree` (:195), which repeats the classic three steps
`numSimulations` times:

1. `simulate` (:106) — descend from the root with the action-selection
   function until an unvisited edge or the depth cutoff. Root and interior
   selectors are combined by `switchingActionSelectionWrapper`.
2. `expand` (:134) — call `recurrentFn` on the chosen `(parent, action)`
   edge, allocate the next free slot (`Tree.nextNodeIndex`), store reward,
   discount, prior, value, embedding.
3. `backward` (:162) — back the leaf value up to the root along parent
   pointers, updating running-average node values and edge visit counts.

Because the tree is persistent (immutable Lean structures), passing an
existing tree to `searchWithTree` continues a previous search — the mechanism
behind AlphaZero subtree reuse.

### Action selection and Q-transforms

Selectors have two signatures (`Tyr/Mctx/ActionSelection.lean:10-15`):
`RootActionSelectionFn` (no depth) and `InteriorActionSelectionFn`. Provided
implementations:

- `muzeroActionSelection` (:29) — PUCT: `argmax` of normalized Q plus
  `√N · pbC · π / (n + 1)`, with `pbCInit = 1.25`, `pbCBase = 19652`.
  Invalid actions are masked at the root only.
- `gumbelMuZeroRootActionSelection` (:60) — sequential-halving schedule
  (`getTableOfConsideredVisits` from `Tyr/Mctx/SeqHalving.lean`) scoring
  `gumbel + logits + normalizedQ` among actions on the current visit round.
- `gumbelMuZeroInteriorActionSelection` (:80) — deterministic:
  `argmax` of `softmax(logits + completedQ) − visits / (1 + Σvisits)`.

Q-transforms normalize raw Q-values before selection
(`Tyr/Mctx/QTransforms.lean`): `qtransformByMinMax` (known value bounds),
`qtransformByParentAndSiblings` (default for MuZero/AlphaZero), and
`qtransformCompletedByMixValue` (default for Gumbel MuZero; completes
unvisited Q-values with the mixed value of Appendix D of the Gumbel MuZero
paper via `computeMixedValue`).

### Policies

The public entry points are in `Tyr/Mctx/Policies.lean`:

- `muzeroPolicy` (:61) — fresh tree each call, Dirichlet-style root noise,
  PUCT selection everywhere.
- `alphazeroPolicy` (:103) — same selection, but takes an optional
  `searchTree` to continue from and a `maxNodes` capacity override, enabling
  subtree persistence across environment steps.
- `gumbelMuZeroPolicy` (:155) — Gumbel root with sequential halving,
  deterministic interior; `maxNumConsideredActions := 16`,
  `gumbelScale := 1.0`. `actionWeights` are
  `softmax(priorLogits + completedQvalues)` rather than raw visit
  probabilities.

All three accept `invalidActions : Option (Array Bool)` (`true` = invalid),
`maxDepth`, a `qtransform` override, and exploration hyperparameters.

### Batched API

`Tyr/Mctx/Batched.lean` mirrors the stack for a batch of independent search
problems. `BatchedTree S E` is literally an `Array` of per-row trees; the
point is the `BatchedRecurrentFn`:

```lean
abbrev BatchedRecurrentFn (P S : Type) :=
  P → UInt64 → Array Action → Array S → BatchedRecurrentFnOutput × Array S
```

`searchBatchedWithTrees` advances every row one simulation per model call, so
a neural dynamics model sees one batched query per simulation step instead of
one per row. `muzeroPolicyBatched`, `alphazeroPolicyBatched`,
`gumbelMuZeroPolicyBatched`, `resetSearchTreeBatched`, and
`getSubtreeBatched` complete the surface.

### DAG backend

`Tyr/MctxDag/` re-implements the same pipeline over `DagTree S K E`
(`Tyr/MctxDag/Tree.lean:7`). The differences from `Tree`:

- A `keys : Array K` slot per node plus `keyToNode : Std.HashMap K NodeIndex`
  and a `numAllocated` bump counter (requires `[BEq K] [Hashable K]`).
- No `parents`/`actionFromParent` arrays: a node can have several parents, so
  backup uses the concrete simulated path (`simulatePath` / `backwardPath` in
  `Tyr/MctxDag/Search.lean:124,205`).
- `expandEdge` (:154) hashes the successor embedding with a user-supplied
  `keyFn : S → K`; if the key is already in the table, the existing node is
  reused as the edge target instead of allocating a new one.

The three policies `muzeroPolicyDag`, `alphazeroPolicyDag`,
`gumbelMuZeroPolicyDag` (`Tyr/MctxDag/Policies.lean:66,107,159`) take the same
arguments as their tree counterparts plus `keyFn`, and return
`PolicyOutput (DagTree S K _)`. There is no batched DAG API.

### AlphaGrad bridge

`Tyr/AD/Elim/AlphaGradMctx.lean` instantiates the framework for AlphaGrad-style
vertex elimination over local-Jacobian graphs: `S := AlphaGradState`
(elimination graph + action trace), `P := AlphaGradMctxConfig`,
`K := AlphaGradDagKey` (canonical edge/elimination key via `dagStateKey`, used
as the DAG transposition key). It provides a deterministic
`recurrentFn : RecurrentFn AlphaGradMctxConfig AlphaGradState`, heuristic
priors/values (`heuristicPriorLogits`, `heuristicValue`), an
`invalidActionMask`, and ready-made entry points:

- `searchStep?` (:698) — one tree-Gumbel-guided step.
- `searchStepDagWithPolicy?` (:758) — one DAG step with
  `AlphaGradDagMctsPolicy.alphaZero | .gumbelMuZero`; AlphaZero returns the
  tree for carry-over.
- `searchEpisode?` / `searchEpisodeDag?` / `searchEpisodeDagGumbel?`
  (:851/:942/:951) — run a full elimination episode, returning
  `Except String AlphaGradEpisodeResult` (`actions0`, `order1`, `stepRewards`,
  `totalReward`). `...FromEdges?` / `...FromGraph?` variants build the initial
  state for you; `replayActions?` replays a fixed action sequence.

## Key APIs

Policies (unbatched, `import Tyr.Mctx`, `open torch.mctx`):

| Function | Search tree | Distinctive arguments |
| --- | --- | --- |
| `muzeroPolicy` | `Tree S Unit` (fresh) | `dirichletFraction := 0.25`, `pbCInit`, `pbCBase`, `temperature` |
| `alphazeroPolicy` | `Tree S Unit` (persistent) | `searchTree : Option (Tree S Unit)`, `maxNodes` |
| `gumbelMuZeroPolicy` | `Tree S GumbelMuZeroExtraData` | `maxNumConsideredActions := 16`, `gumbelScale := 1.0` |

All take `params`, `rngKey : UInt64`, `root : RootFnOutput S`,
`recurrentFn`, `numSimulations`, and optional `invalidActions`, `maxDepth`,
`qtransform`; all return `PolicyOutput _`.

DAG policies (`import Tyr.MctxDag`, `open torch.mctxdag`):
`muzeroPolicyDag`, `alphazeroPolicyDag`, `gumbelMuZeroPolicyDag` — same shape,
plus `keyFn : S → K` and `[BEq K] [Hashable K] [Inhabited K]`.

Lower-level pieces you touch when writing a custom policy:

| Function | Location | Purpose |
| --- | --- | --- |
| `search` / `searchWithTree` | `Tyr/Mctx/Search.lean:225/195` | run search from a root / existing tree |
| `instantiateTreeFromRoot(WithCapacity)` | `Tyr/Mctx/Search.lean:71/40` | allocate a tree |
| `updateTreeWithRoot` | `Tyr/Mctx/Search.lean:82` | refresh root prior/value on a reused tree |
| `getSubtree` / `resetSearchTree` | `Tyr/Mctx/Tree.lean:133/110` | subtree reuse, tree wipe |
| `muzeroActionSelection` etc. | `Tyr/Mctx/ActionSelection.lean` | plug-in selectors |
| `qtransformBy*` | `Tyr/Mctx/QTransforms.lean` | plug-in Q normalizers |
| `searchDag` / `searchWithDag` | `Tyr/MctxDag/Search.lean:275/244` | DAG equivalents |
| `searchBatched(WithTrees)` | `Tyr/Mctx/Batched.lean:159/90` | batched search loop |

AlphaGrad configs (`Tyr/AD/Elim/AlphaGradMctx.lean`):

```lean
structure AlphaGradMctsConfig where
  numSimulations : Nat := 32
  maxDepth : Option Nat := none
  maxNumConsideredActions : Nat := 16
  gumbelScale : Float := 1.0
  dagMaxNodes : Option Nat := none        -- DAG AlphaZero capacity override
  dagDirichletFraction : Float := 0.0     -- DAG AlphaZero root noise
  dagTemperature : Float := 1.0
```

`AlphaGradMctxConfig` carries the environment knobs: `rewardMode`,
`discount := 1.0`, `invalidActionPenalty := -1.0e6`,
`infeasibleStatePenalty := -1.0e4`, `terminalBonus`, `maxEpisodeSteps`,
plus constraint/cost specifications.

## Usage example

Reconstructed example (from `Tests/TestMctx.lean:37-69`): a one-step bandit
where action 3 is masked invalid and the prior decides.

```lean
import Tyr.Mctx

open torch.mctx

def root : RootFnOutput Unit := {
  priorLogits := #[-1.0, 0.0, 2.0, 3.0]
  value := 0.0
  embedding := ()
}

def banditStep : RecurrentFn Unit Unit := fun _params _rng action _emb =>
  let rewards := #[0.0, 0.0, 0.0, 0.0]
  ({ reward := rewards.getD action 0.0
     discount := 0.0
     priorLogits := #[0.0, 0.0, 0.0, 0.0]
     value := 0.0 }, ())

-- The test asserts: out.action = 2 (highest-prior valid action) and
-- out.actionWeights = #[0.0, 0.0, 1.0, 0.0] after a single simulation.
def out :=
  muzeroPolicy
    (params := ()) (rngKey := 0) (root := root)
    (recurrentFn := banditStep)
    (numSimulations := 1)
    (invalidActions := some #[false, false, false, true])
    (dirichletFraction := 0.0)
```

Subtree reuse across environment steps, as the AlphaZero self-play loop does
(from `Examples/AlphaGradPort/PolicyTrain.lean:638-661`):

```lean
let out := alphazeroPolicy
  (params := searchParams) (rngKey := key)
  (root := root) (recurrentFn := recurrentFromNet)
  (numSimulations := cfg.numSimulations)
  (searchTree := tree?)                 -- continue from last step's subtree
  (invalidActions := some invalid)
-- ... apply out.action in the real environment ...
tree? := if t.done then none else some (getSubtree out.searchTree out.action)
```

Full AlphaGrad episode planning (from `Examples/AlphaGradPort/A0Train.lean:64-75`):

```lean
open Tyr.AD.Elim
-- task : Examples.AlphaGradPort.TaskSpec with envCfg, mctsCfg, graph, numVertices
let result := searchEpisodeDagGumbelFromGraph?   -- or searchEpisodeFromGraph? / searchEpisodeDagFromGraph?
  task.envCfg task.mctsCfg seed task.graph task.numVertices
-- result : Except String AlphaGradEpisodeResult
```

Runnable front ends (registered in `lakefile.lean:707-727`):

```bash
lake exe AlphaGradRoeFlux1dA0 [episodes]     # RoeFlux_1d elimination-planning demo
lake exe AlphaGradPortSweep                  # task sweep
lake exe AlphaGradPolicyTrain <mode> [task] [epochs] [episodes-per-epoch]
```

Tests live in `Tests/TestMctx*.lean` and `Tests/TestMctxDag.lean` (run via
`lake exe test_runner`); `Tests/MctxData/` holds JSON tree dumps recorded with
upstream mctx configuration conventions (`pb_c_base`, qtransform names).

## Semantics caveats

Worth knowing before you trust the exploration knobs:

- All "randomness" is deterministic hashing of the `UInt64` key
  (`pseudoUniform01`, `Tyr/Mctx/Policies.lean:11`). `addDirichletNoise`
  normalizes those values — it is not a true Dirichlet sample, and
  `_dirichletAlpha` is ignored in every policy signature.
- `temperature` does not change the chosen action in `muzeroPolicy` /
  `alphazeroPolicy`: the action is `argmax` of tempered log visit-probs, and
  `argmax` is invariant under positive scaling. `actionWeights` are the raw
  visit probabilities either way. Upstream mctx samples from the tempered
  distribution instead.

## Related guides

- [Getting started](getting-started.md) — build and run the test suite.
- [Tensors](core/tensors.md) — the tensor stack; Mctx is deliberately
  tensor-free, `Examples/AlphaGradPort/PolicyTrain.lean` shows the bridge
  (network outputs feed `RootFnOutput`).
- [Autodiff](autodiff.md) — the `Tyr/AD/Elim` elimination machinery the
  AlphaGrad bridge plans over.
- [Modules](modules.md) and [Optimization](optimization.md) — used by the
  policy trainer to fit the prior/value network.
- [Examples and testing](examples-and-testing.md) — how the `Tests/` suites
  and example executables are wired.

Exhaustive per-symbol documentation for `Tyr.Mctx`, `Tyr.MctxDag`, and the
AlphaGrad bridge is generated by doc-gen4 (see `docbuild/`).
