# Launch evidence and publication gates

This is the internal claim-to-artifact ledger for the research preview. It
records rerun state and the evidence that must be collected before a claim is
promoted into the public article.

Status values: **verified** means rerun in the current checkout; **recorded**
means an artifact exists from a named external run but was not rerun locally;
**available** means the path exists but still needs a launch-quality rerun; and
**gated** means publication depends on missing provenance, named hardware,
weights, or branch policy.

| Candidate claim | Evidence | Status | Publication gate |
|---|---|---|---|
| Lean rejects an incompatible projection before execution | `./scripts/launch/run_shape_safety.sh` | **verified** | Capture clean terminal clip |
| A Tyr model trains and its loss decreases | `.lake/build/bin/BranchingFlowsContinuousTrain` | **verified** | Add loss chart/capture |
| Tyr can simulate and render scientific dynamics | `.lake/build/bin/RunVanDerPolLeanPlot` and `output/event-skeleton/plot_limit_cycle.svg` | **verified** on Event Skeleton branch | Label as emerging direction or merge deliberately |
| Tyr has ODE/SDE solver and adjoint infrastructure | `Tyr/DiffEq/` and its test executables | **available** | Re-run relevant suite serially through Lake |
| Tyr's typed GPU language checks capabilities and generates CUDA source | `lake exe TestGPUDSL` | **verified**: 49/49 compiler/code-generation tests | This local run used CPU stubs and did not execute CUDA |
| Generated GPU kernels execute through native CUDA dispatch | `scripts/gpu/test_leantest_gb10_e2e.sh` and `launch/generated/gpu/cuda-parity-gb10.txt` | **recorded**: five generated modules, 10/10 tests | Named-device result on one CUDA configuration; do not generalize it to every GPU |
| Native GPU results match numerical references | `launch/generated/gpu/cuda-parity-gb10.txt` | **recorded**: 10/10 reference comparisons passed | Preserve the companion manifest and raw, uncut log |
| Tyr loads and runs a pretrained checkpoint | `launch/generated/model-inference/qwen36-27b-gb10.txt` | **recorded; provenance incomplete** | Re-capture model ID and revision, Tyr commit, command, device metadata, and output together |
| SafeTensors schemas generate typed accessors during elaboration | `./scripts/launch/run_safetensors_schema.sh` | **verified** | Capture the generated declarations for the compiler clip |
| Tyr supports distributed training | `scripts/nanochat/` | **available** | Fresh multi-GPU run before making a launch claim |
| Molecule branching and XYZ export work end to end | `./scripts/launch/run_molecule_showcase.sh` | **verified** | Target-conditioned reference run; do not describe it as learned chemistry |
| Full molecule transformer trains on QM9 through the CUDA device path | `./scripts/launch/run_qm9_launch_demo.sh` | **recorded research diagnostic** on an NVIDIA GB10 | Improve split/label objectives before presenting samples as chemically meaningful |
| Speech generation/transcription paths exist | Qwen3-TTS, Qwen3-ASR, Whisper examples | **available** | Clearly state external bridge and model-weight boundaries |

## Verification notes

- Branch: `ranvier-labs/event-skeleton`
- Local training result: `1.484375 → 0.000481`
- Van der Pol result: 6,668 samples; closure `q=0.006748`,
  `qdot=0.004919`; SVG generated successfully.
- `lake exe TestGPUDSL` passed 49/49 tests in the current checkout. The local
  build reported `NVCC not found` and used CPU kernel stubs, so this verifies
  the typed DSL, capability checks, and code-generation tests, not native CUDA
  execution.
- The separate native suite generated five CUDA translation units and executed
  ten reference-comparison tests on `spark-e626`; all ten passed. The recorded
  device was an NVIDIA GB10 with driver `580.126.09` and CUDA toolkit 13.0.
  The raw log is `launch/generated/gpu/cuda-parity-gb10.txt` (SHA-256
  `1718a3278d31b05d902bbc023c911fab07a11cfade9bf60b830abcaf1ebc975b`).
  This is evidence for that CUDA path on that named device, not the definition
  of Tyr's GPU support.

## Claims deliberately excluded from the initial post

- Production readiness.
- Universal static checking of every tensor/runtime property.
- Performance leadership without a named benchmark, device, baseline, and log.
- GPU portability or performance generalized from a single named-device run.
- Full support for every model family or every mode mentioned in the examples.
- Formal verification of floating-point numerical behavior.
- Treating an optional fallback path as native generated-kernel execution.
