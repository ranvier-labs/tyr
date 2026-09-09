# MCTS reference data

`deterministic_reference.json` is the active numerical reference for
`Tests/TestMctxTree.lean`. Its expected results are produced by the **unmodified
upstream mctx 0.0.6 package**, using JAX/JAXlib 0.7.0 and Chex 0.1.91. The artifact
records those versions, a SHA-256 of the installed mctx Python source, the generator
path, and the complete finite model. It does not use Tyr to generate expected data.

The six-state, four-action model defines every transition, reward, prior logit,
and raw value independently of which transitions the search chooses. Its input
numbers are binary fractions. Four cases exercise MuZero min/max and
parent/siblings Q transforms, and Gumbel MuZero completed-Q transforms with and
without rescaling. They include positive and negative discounts, nonzero rewards,
nondefault PUCT constants, and different sequential-halving widths. Every
algorithm parameter is read from the artifact by the Lean test.

Generation enables JAX float64, disables Dirichlet/Gumbel exploration, and uses
temperature zero for MuZero. Upstream's small PUCT tie-breaking noise remains
unchanged; the generator requires identical search output for seeds 0 through 7.
This is a deterministic search regression, not a cross-language PRNG reference.
Independent sampling tests cover stochastic policies separately.

The test compares exact actions, visits, parent/action topology, child indices,
and embeddings, plus priors, raw and backed-up values, edge rewards/discounts,
Q values, transformed root Q values, and action weights at absolute tolerance
`1e-10`. Priors are compared as probabilities because policy normalization may
shift logits by a common constant. Floating-point comparisons check dimensions
and finiteness. Every simulation allocates a node; there are no ignored trailing
slots or truncated simulation budgets. Mutation regressions change expected
actions, visits, and Q values, and change PUCT and sequential-halving parameters,
to establish that those checks actually reject disagreement.

Regenerate in an isolated Python 3.12 environment:

```sh
python3.12 -m venv /tmp/tyr-mctx-reference
/tmp/tyr-mctx-reference/bin/pip install mctx==0.0.6 jax==0.7.0 jaxlib==0.7.0 chex==0.1.91
/tmp/tyr-mctx-reference/bin/python scripts/generate_mctx_reference.py
/tmp/tyr-mctx-reference/bin/python scripts/generate_mctx_reference.py --check
```

The generator is optional: normal Lean/CI tests only read the checked-in JSON,
so JAX and Python dependencies are not required to run them. Inspect changes to
the model, parameters, and resulting search statistics before updating the
reference. A Tyr test failure is not a reason to regenerate expected data.

## Preserved historical dumps

The four original files remain unchanged:

| File | SHA-256 |
| --- | --- |
| `muzero_tree.json` | `f1ed64668f9c9f1349afd5698c381ea6addab9207210d115a1e0cdc93a4f4827` |
| `muzero_qtransform_tree.json` | `66e4aec1354c60d7057fd6b36f6297c5cf3c95499bde8313fb6f7d325329c236` |
| `gumbel_muzero_tree.json` | `288400b970a08e1fbe40e5e90ab7e9b3b4605fef942ebcc88ce78f28fbd4634e` |
| `gumbel_muzero_reward_tree.json` | `8fd64aeac2a850a6e9a0c006bbf81b2773418364888f8c28e9ca84e792323bdb` |

They are byte-identical to [upstream mctx test data at commit
f8cd07bcc5d7ff736ae4c1e4217d2001508f8353](https://github.com/google-deepmind/mctx/tree/f8cd07bcc5d7ff736ae4c1e4217d2001508f8353/mctx/_src/tests/test_data),
originally introduced in upstream commit `1232e22`. Upstream's
[`tree_test.py`](https://github.com/google-deepmind/mctx/blob/f8cd07bcc5d7ff736ae4c1e4217d2001508f8353/mctx/_src/tests/tree_test.py)
replays them with predictions derived from JAX PRNG keys, and compares the complete
recorded tree. That source and data are under upstream's Apache-2.0 license.

The former Tyr test read their action count, discount and simulation budget,
capped the budget at 32, generated a different LCG-based model, and checked only
array width and total visits. It neither replayed nor validated those trees.
The new tests replace that ineffective gate; the historical files are retained
for provenance and possible future compatibility work, and are not claimed as
passing golden-tree comparisons. Reconstructing a model from the dumps alone
would use rounded predictions only for already-expanded nodes and leave unseen
transitions unspecified, coupling the model to the expected search trajectory.
