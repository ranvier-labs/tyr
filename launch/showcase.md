# Demonstration guide

These short examples follow the same order as the research-preview article.
Each one states the boundary it exercises and points to a reproducible command
or recorded artifact.

## 1. Static tensor contracts

```bash
./scripts/launch/run_shape_safety.sh
```

The valid projection elaborates. The second source changes the weight shape and
is rejected before a tensor is allocated. This is the shortest demonstration
of the core typed-tensor contribution.

## 2. Differentiation through a contact event

```bash
lake exe RunUrdfContactExample --csv /tmp/contact.csv
```

The executable localizes an impact, applies a velocity reset, and propagates an
adjoint through the event with a saltation update. Its CSV retains both
one-sided states at impact, so the velocity discontinuity is present in the
data rather than added by the visualization.

## 3. Checkpoint schemas during elaboration

```bash
./scripts/launch/run_safetensors_schema.sh
```

The fixture header is read during elaboration and produces typed tensor
specifications, checked loaders, and a nested checkpoint record. No model
schema is handwritten in the example.

## 4. GPU programs and numerical parity

```bash
lake exe TestGPUDSL
```

This runs the 49-test compiler and code-generation suite. On a machine without
NVCC it uses CPU stubs, so it does not establish native CUDA execution. The
separate recorded native run generated five CUDA modules and passed 10/10
reference comparisons; its raw log and device manifest are in
`generated/gpu/`.

The distinction matters: tensor programs use a general explicit device layer,
while the lower-level kernel compiler currently targets NVIDIA CUDA. A
named-device parity run is evidence for that execution path, not a claim about
every GPU.

## 5. Practical model execution

The recorded transcript in
`generated/model-inference/qwen36-27b-gb10.txt` shows Tyr loading a Qwen3.6 27B
checkpoint and generating 32 tokens through `Device.CUDA 0`. It demonstrates
loader and runtime integration, not model quality or performance.

## 6. BranchingFlows mechanism trace

[Open the standalone technical demonstration](site/branchingflows-water.html).

The recorded water trajectory follows stable runtime identities through two
split events and exposes all 33 states as raw JSONL. It is deliberately a
target-conditioned mechanism trace: a hand-written predictor can inspect the
fixed endpoint, and no trained molecule model is loaded. The example therefore
demonstrates variable cardinality, state updates, event sampling, lineage, and
serialization without making a learned-generation claim.

```bash
lake build BranchingFlowsMoleculeGenerate
./scripts/launch/run_molecule_showcase.sh
```

## Further executable examples

The repository also contains ODE and SDE solvers, continuous-flow training,
speech and multimodal model paths, and a variable-cardinality branching-flow
sampler. They are useful follow-up examples, while the initial article remains
focused on the typed system boundaries above.
