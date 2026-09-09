#!/usr/bin/env python3
"""Generate independent search references with the unmodified upstream mctx wheel.

Install into a disposable Python 3.12 environment:
  pip install mctx==0.0.6 jax==0.7.0 jaxlib==0.7.0 chex==0.1.91
Run from the repository root; --check verifies the checked-in artifact.
No Tyr implementation is imported or invoked.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx
import numpy as np


VERSIONS = {"mctx": "0.0.6", "jax": "0.7.0", "jaxlib": "0.7.0", "chex": "0.1.91"}
OUTPUT = Path(__file__).resolve().parents[1] / "Tests/MctxData/deterministic_reference.json"

# A complete six-state/four-action model. Cycles are model transitions, not
# transpositions in the search tree. All numbers are exactly representable in
# binary, and all policy logits within each state are distinct. Recurrent output
# is defined for every transition, including ones absent from the reference tree.
MODEL = {
    "root_state": 0,
    "prior_logits": [
        [0.125, -0.75, 0.625, -0.25],
        [-0.375, 0.875, 0.25, -0.625],
        [0.5, -0.125, -0.875, 0.375],
        [-0.5, 0.25, 0.75, -0.125],
        [0.875, -0.25, 0.125, -0.625],
        [0.25, 0.625, -0.5, -0.125],
    ],
    "values": [0.375, -0.625, 0.125, 0.75, -0.25, 0.5],
    "next_states": [[1, 2, 3, 4], [2, 4, 5, 0], [4, 3, 0, 5],
                    [5, 0, 4, 2], [0, 5, 2, 1], [3, 1, 0, 4]],
    "rewards": [[0.25, -0.125, 0.5, -0.375], [-0.5, 0.625, -0.25, 0.125],
                [0.375, -0.625, 0.25, 0.5], [-0.125, 0.375, -0.5, 0.625],
                [0.5, 0.125, -0.375, -0.25], [-0.625, 0.25, 0.375, -0.125]],
}

CASES = [
    {"name": "muzero_min_max", "algorithm": "muzero", "num_simulations": 24,
     "discount": -1.0, "algorithm_config": {
         "qtransform": "qtransform_by_min_max",
         "qtransform_kwargs": {"min_value": -4.0, "max_value": 4.0},
         "dirichlet_fraction": 0.0, "dirichlet_alpha": 0.3,
         "pb_c_init": 1.25, "pb_c_base": 32.0, "temperature": 0.0}},
    {"name": "muzero_parent_siblings", "algorithm": "muzero", "num_simulations": 24,
     "discount": 0.875, "algorithm_config": {
         "qtransform": "qtransform_by_parent_and_siblings",
         "qtransform_kwargs": {"epsilon": 1e-8},
         "dirichlet_fraction": 0.0, "dirichlet_alpha": 0.3,
         "pb_c_init": 0.75, "pb_c_base": 64.0, "temperature": 0.0}},
    {"name": "gumbel_no_rescale", "algorithm": "gumbel_muzero", "num_simulations": 32,
     "discount": -1.0, "algorithm_config": {
         "qtransform": "qtransform_completed_by_mix_value",
         "qtransform_kwargs": {"value_scale": 0.5, "maxvisit_init": 20.0,
                               "rescale_values": False, "use_mixed_value": True,
                               "epsilon": 1e-8},
         "max_num_considered_actions": 4, "gumbel_scale": 0.0}},
    {"name": "gumbel_rescale", "algorithm": "gumbel_muzero", "num_simulations": 24,
     "discount": 0.875, "algorithm_config": {
         "qtransform": "qtransform_completed_by_mix_value",
         "qtransform_kwargs": {"value_scale": 0.125, "maxvisit_init": 12.0,
                               "rescale_values": True, "use_mixed_value": True,
                               "epsilon": 1e-8},
         "max_num_considered_actions": 3, "gumbel_scale": 0.0}},
]


def generate_case(case):
    priors = jnp.asarray(MODEL["prior_logits"], dtype=jnp.float64)
    values = jnp.asarray(MODEL["values"], dtype=jnp.float64)
    next_states = jnp.asarray(MODEL["next_states"], dtype=jnp.int32)
    rewards = jnp.asarray(MODEL["rewards"], dtype=jnp.float64)
    state = jnp.asarray([MODEL["root_state"]], dtype=jnp.int32)
    root = mctx.RootFnOutput(prior_logits=priors[state], value=values[state], embedding=state)

    def recurrent(params, rng_key, action, embedding):
        del params, rng_key
        next_state = next_states[embedding, action]
        reward = rewards[embedding, action]
        return mctx.RecurrentFnOutput(
            reward=reward, discount=jnp.full_like(reward, case["discount"]),
            prior_logits=priors[next_state], value=values[next_state]), next_state

    config = dict(case["algorithm_config"])
    qtransform = functools.partial(getattr(mctx, config.pop("qtransform")),
                                   **config.pop("qtransform_kwargs"))
    policy = getattr(mctx, case["algorithm"] + "_policy")
    run = jax.jit(lambda seed: policy(
        params=(), rng_key=jax.random.PRNGKey(seed), root=root,
        recurrent_fn=recurrent, num_simulations=case["num_simulations"],
        max_depth=case["num_simulations"], qtransform=qtransform, **config))

    result = run(0)
    # Upstream MuZero adds a 1e-7 tie breaker even with zero Dirichlet noise.
    # Keep upstream unchanged and reject this fixture if nearby seeds change any
    # search output. This also rejects a tied maximum visit count at temperature 0.
    for seed in range(1, 8):
        other = run(seed)
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(other), strict=True):
            np.testing.assert_array_equal(a, b)

    tree = jax.tree.map(lambda x: x[0], result.search_tree)
    assert np.all(np.asarray(tree.node_visits) > 0), "Fixture must allocate every slot"
    expected = {field: np.asarray(getattr(tree, field)).tolist() for field in (
        "node_visits", "raw_values", "node_values", "parents", "action_from_parent",
        "children_index", "children_visits", "children_rewards", "children_discounts",
        "children_values", "embeddings")}
    expected["prior_probs"] = np.asarray(jax.nn.softmax(tree.children_prior_logits)).tolist()
    expected["qvalues"] = np.asarray(
        tree.children_rewards + tree.children_discounts * tree.children_values).tolist()
    expected["action"] = int(result.action[0])
    expected["action_weights"] = np.asarray(result.action_weights[0]).tolist()
    expected["transformed_root_qvalues"] = np.asarray(qtransform(tree, 0)).tolist()
    return dict(case, max_depth=case["num_simulations"], seed=0, expected=expected)


def generate():
    for package, version in VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != version:
            raise RuntimeError(f"Expected {package}=={version}, found {actual}")
    jax.config.update("jax_enable_x64", True)
    # Hash the actual upstream Python package to identify the oracle independently
    # of the editable generator. No timestamps or host paths enter the artifact.
    package_root = Path(mctx.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return {
        "schema_version": 1,
        "provenance": {
            "upstream": "https://github.com/google-deepmind/mctx",
            "distribution": "https://pypi.org/project/mctx/0.0.6/",
            "versions": VERSIONS, "mctx_source_sha256": digest.hexdigest(),
            "jax_enable_x64": True, "seed_invariance_checked": list(range(8)),
            "generator": "scripts/generate_mctx_reference.py",
        },
        "model": MODEL,
        "cases": [generate_case(case) for case in CASES],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    content = json.dumps(generate(), indent=2, allow_nan=False) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit(f"Reference differs: {args.output}")
        print(f"Reference matches upstream regeneration: {args.output}")
    else:
        args.output.write_text(content)
        print(f"Wrote {args.output} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
