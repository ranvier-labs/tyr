# Examples and testing

## Purpose & when to use

`Examples/` is the collection of runnable end-to-end demonstrations: training
scripts (char-GPT, NanoChat, discrete diffusion, branching flows), pretrained-model
demos (Flux, Whisper, Qwen, Gemma, KittenTTS), and GPU kernel parity/benchmark
harnesses. `Tests/` is the LeanTest-based test suite — unit tests, parity tests
against reference implementations, and GPU end-to-end suites. Read this chapter
when you want to run something rather than build it, or when you are adding a
test or example of your own.

## Architecture & main abstractions

### Build wiring

Everything is declared in `lakefile.lean`:

- `lean_lib Tests` (`lakefile.lean:581`, roots `Tests`, no precompile),
  `lean_lib TestsExperimental` (`:586`), `lean_lib Examples` (`:591`).
- 63 `lean_exe` targets. Test-side executables root under `Tests/`, example
  executables root under `Examples/`.
- `@[test_driver] lean_exe test_runner` (`:598`) is Lake's test driver, so
  `lake test` builds and runs it.
- The test framework is an external dependency
  (`lakefile.lean:260`): `require LeanTest from git "https://github.com/cpehle/lean_test.git"`.

### The LeanTest framework

Tests are plain `IO Unit` definitions tagged with attributes
(`.lake/packages/LeanTest/LeanTest/Attr.lean`):

```lean
@[test] def testSomething : IO Unit := do ...        -- discovered and run
@[test_ignore] def testSlow : IO Unit := do ...      -- skipped unless --ignored
@[test_should_error] def testFails : IO Unit := do ... -- passes iff it throws
```

Assertions live in `LeanTest/Assert.lean` and throw on failure:

| Function | Purpose |
|---|---|
| `LeanTest.assertTrue / assertFalse` | boolean conditions |
| `LeanTest.assertEqual / assertNotEqual` | `[BEq α] [ToString α]` equality |
| `LeanTest.assertSome / assertNone` | `Option` shape |
| `LeanTest.assertThrows / assertNoThrow` | error behavior of an `IO` action |
| `LeanTest.assertSatisfies` | predicate with a message |
| `LeanTest.fail` | unconditional failure |

Discovery is environment-based. A runner imports modules with
`Lean.importModules` and hands the resulting `Environment` to the runner,
which scans it for tagged declarations
(`.lake/packages/LeanTest/LeanTest/Runner.lean:145`):

```lean
unsafe def LeanTest.runTestsAndExit
    (env : Environment) (opts : Options) (config : RunConfig := {}) : IO UInt32

structure LeanTest.RunConfig where
  includeIgnored : Bool := false
  failFast       : Bool := false
  filter         : Option String := none
```

Consequence worth remembering: **a test file that no runner imports never
executes**, even though it still compiles as part of `lean_lib Tests`. Several
files in the tree are in that state today (see "Caveats" below).

### Test tiers

Tests are organized in three tiers:

1. **Main suite** — `lake test` (alias: `lake run`). `test_runner`'s root is
   `Tests/RunTests.lean`, which imports the umbrella module `Tests.lean`
   (116 imports) plus `Tests.TestAdditionalManifolds` (special-cased there, not
   in `Tests.lean`). It parses `--filter PATTERN`, `--ignored`, `--fail-fast`,
   `--help` into a `LeanTest.RunConfig`.
2. **Experimental suite** — `lake exe test_runner_experimental`, root
   `Tests/RunTestsExperimental.lean`, which imports `TestsExperimental.lean`.
   That umbrella currently contains exactly one module,
   `Tests.TestBranchingFlows`, parked there until
   `Tyr/Model/BranchingFlows.lean` stabilizes (note at `Tests.lean:118`).
3. **Standalone executables** — focused or heavy suites built and run
   individually: `TestDiffEq`, `TestDiffEqAdjoint`, `TestDiffEqAdjointCore`,
   `TestDataLoader`, `TestGPUDSL`, `TestGPUKernels`, `TestGPUE2E`,
   `TestGPUGB10E2E`, `TestGPUTileIR`, `TestTileIRGenerateMain`, `TestDiffusion`,
   and `RunRiemannianNanoGPTTests`. Each is a thin runner module
   (`Tests/RunTest*.lean`, `Tests/Run*.lean` — the prefix split is historical)
   with the same copy-pasted `parseArgs` → `runTestsAndExit` body, except
   `TestDiffusion` whose executable roots at the test file directly
   (`lakefile.lean:762`), and the SDE/underdamped order-parity runners
   (`Tests/RunDiffEqSDEOrderParity.lean`, `RunDiffEqUnderdampedOrderParity.lean`)
   which just call a `run` function.

Hosted CI (`.github/workflows/ci.yml`) builds and runs only tiers 1 and 2
(`test_runner --fail-fast`, then `test_runner_experimental`). The GPU suites
run on self-hosted hardware via `.github/workflows/cuda-smoke.yml`
(`TestGPUGB10E2E --fail-fast`, `RunMhaH100Decode`) or manually through the
shell harnesses in `scripts/gpu/`.

There is also `lean_exe ffi_crash_probe` (`lakefile.lean:672`, root
`Tests/FfiCrashProbe.lean`): a manual probe that intentionally triggers a
libtorch CUDA error to check the process dies with an intelligible `c10::Error`
message instead of a bare SIGABRT. It is deliberately not part of any suite;
`.github/workflows/ffi-probe.yml` runs it on Linux for diagnostics.

### Fixtures

- `Tests/fixtures/safetensors/` — `single.safetensors`, `sharded/`,
  `indexed.safetensors`, `indexed_dir/`, `provider_errors/` for the SafeTensors
  loader and type-provider tests (see [serialization.md](serialization.md)).
- `Tests/MctxData/` — four recorded MuZero/Gumbel search trees as JSON
  (~9 MB) replayed by the Mctx tests.
- GPU kernel examples compare against `.pt` fixtures produced by libtorch
  reference implementations; see `Examples/GPU/FixtureRunner.lean` below.

### Example organization

Examples pair library modules with thin executable wrappers. The convention is
a 4-line entrypoint module, e.g. `Examples/GPU/RunRMSNormExe.lean`:

```lean
import Examples.GPU.RunRMSNorm

/-- Lake executable entrypoint for the fused residual + RMSNorm demo. -/
def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunRMSNorm.main args
```

The root module `Examples.lean` imports only library modules (GPT, NanoChat,
AlphaGradPort) — executables are excluded by design because they define global
`main`.

Shared infrastructure:

- `Examples/ModelRunner.lean` — CLI/device/streaming helpers for the
  pretrained-model demos: `resolveDevice : String → IO (Device × Option String)`
  (parses `auto`/`cpu`/`mps`/`cuda[:n]`, warns and falls back to CPU),
  `parseNatArg`, `loadPrompts`.
- `Examples/GPU/Parity.lean` — parity-check helpers: `gpuTarget`/`gpuFamily`
  (read `TYR_GPU_TARGET`/`TYR_GPU_FAMILY`), `isBlackwellFamily`, `requireCuda`,
  `seedFixtures`, and `TensorCheck` (label/ok/mae/maxErr/rtol/atol) with
  `compareTensors`.
- `Examples/GPU/FixtureRunner.lean` — `FixtureSpec` (fixture dir + tensor
  names), `CommonArgs`, `parseCommonArgs` (`--regen`, `--gen-only`, `--trials`,
  `--seed`, `--regen-every-trial`), and `runWithFixtures`, which regenerates
  missing `.pt` fixtures from libtorch references and then compares kernel
  outputs against them.
- `Examples/GPT/GPT.lean` — the canonical dependently-typed model demo:
  `torch.gpt.Config` with presets `Config.gpt2_small`, `gpt2_mini`,
  `gpt2_micro`, `tiny_shakespeare`, `nanogpt_cpu_shakespeare`,
  `nanogpt_gpu_shakespeare`, and shape-indexed parameters such as
  `BlockParams (n_embd : UInt64)` whose fields are `T #[n_embd]`
  (`Examples/GPT/GPT.lean:16`).

Lake scripts (`lakefile.lean:1166`–`1232`) wrap the common invocations so the
runtime library path is set correctly:

| Script | What it does |
|---|---|
| `lake run` | run `test_runner` |
| `lake run train` | run `TrainGPT` |
| `lake run buildGpuTarget -- <KernelModule> <Target>...` | build GPU-backed targets with one kernel module |
| `lake run runBuiltTarget -- <Exe> [args...]` | run a compiled exe from `.lake/build/bin` |
| `lake run buildMhaH100Examples` / `validateMhaH100Examples` | build / build+run the raw H100 MHA binaries |
| `lake run runMhaH100Exe` / `runMhaH100Seq768Exe` | run those binaries directly |

## Key APIs: the example tour

Per-directory summary of what each example demonstrates, with its executable
targets. `Examples/README.md` has per-example data/weights setup and CLI flags
and is the authoritative source for running them.

| Example | Demonstrates | Executables |
|---|---|---|
| `Examples/GPT/` | Char-level GPT-2 on Shakespeare (nanoGPT configs), shape-indexed model code, Riemannian variant | `TrainGPT`, `RunRiemannianNanoGPT` |
| `Examples/Diffusion/` | Masked discrete diffusion on ASCII text, checkpoint save/load, terminal animation | `TrainDiffusion` |
| `Examples/NanoChat/` | Modded-nanogpt port: NorMuon + DistAdam training, SFT (`ChatSFT`), GRPO, tool-augmented generation, eval tasks | `TrainNanoChat` (torchrun-capable), `NanoChatPipeline`, `NanoChatChat` |
| `Examples/BranchingFlows/` | Branching-flow generative models: toy demos, QM9 molecule training/generation (`qm9_xyz/` data) | `BranchingFlows{Continuous,Molecule,MoleculeTransformer}Train`, `BranchingFlowsMoleculeGenerate`, `BranchingFlowsMoleculeTrainGenerate` |
| `Examples/AlphaGradPort/` | PDE flux tasks with MCTS-driven search (see [mctx.md](mctx.md)) | `AlphaGradRoeFlux1dA0`, `AlphaGradPortSweep`, `AlphaGradPolicyTrain`, `AlphaGradPolicySweep` |
| `Examples/GPU/` | ThunderKittens-style kernel parity vs libtorch fixtures, H100 MHA, decode benchmarks, B200 GEMM | `RunCopy`, `RunRotary`, `RunLayerNorm`, `RunRMSNorm`, `RunFlashAttn{,3,Op,Bench}`, `RunMhaH100{,Decode,Train,Seq768}`, `RunDecodeBench`, `RunB200Bf16Gemm` |
| `Examples/EventSkeleton/` | URDF-backed hybrid contact simulation | `RunUrdfContactExample` |
| `Examples/Flux/` | Flux Klein 4B image generation (text encode → diffusion → VAE decode) | `FluxDemo`, `FluxDebug` |
| `Examples/Qwen35/`, `Qwen25Omni/`, `Gemma4/` | Pretrained HF-hub inference with a shared CLI (`--source`, `--prompt[-file]`, `--image`, `--video`, `--batch-size`, `--max-new-tokens`, `--stream`, `--device`, ...) | `Qwen35RunHF`, `Qwen25OmniRunHF`, `Gemma4RunHF` |
| `Examples/Qwen3ASR/` | Speech transcription, live microphone streaming | `Qwen3ASRTranscribe`, `Qwen3ASRLiveMic`, `Qwen3ASRLiveMicTrueStream` |
| `Examples/Qwen3TTS/` | Text-to-speech end to end | `Qwen3TTSEndToEnd` |
| `Examples/Whisper/` | Whisper transcription (file and in-memory), voice-mode loop | `WhisperTranscribe`, `WhisperTranscribeInMem`, `WhisperVoiceMode` |
| `Examples/` (root) | KittenTTS pretrained synthesis and duration/debug harnesses; `TestGuard.lean` is a Widget display demo despite the name | `KittenTTSPretrained`, `KittenTTSDurations`, `KittenTTSDebug`, `KittenTTSCompare` |

Test-side executables: `test_runner`, `test_runner_experimental`,
`RunRiemannianNanoGPTTests`, `ffi_crash_probe`, and the standalone suites
listed under "Test tiers" above. Two codegen tools (`GenerateGpuKernels`,
`GenerateTileIRKernels`) are covered in [gpu/dsl-codegen.md](gpu/dsl-codegen.md).

## Usage example

Reconstructed example (from `Tests/TestSafeTensorsTypeProvider.lean`):

```lean
import Tyr.SafeTensors
import LeanTest

open torch

-- Elaboration-time type provider over a checked-in fixture:
safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as SingleSafe

@[test]
def testSafeTensorsIntrospectionSingle : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/single.safetensors"
  LeanTest.assertEqual schema.sourceIsDirectory false "single source should not be a directory"
  LeanTest.assertEqual schema.tensors.size 1 "single fixture should have one tensor"
```

To make a new test file actually run, add `import Tests.MyNewTest` to
`Tests.lean` — the runner discovers tests by scanning the imported environment.

Running tests and examples:

```bash
lake test                                  # main suite (test_runner)
lake test -- --filter diffeq --fail-fast   # subset, stop on first failure
lake exe test_runner_experimental          # experimental suite
lake exe TestGPUE2E                        # standalone GPU suite (needs CUDA + fixtures)
lake exe ffi_crash_probe                   # manual FFI failure-mode probe

lake build TrainGPT && lake run train      # char-GPT on Shakespeare
lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B --prompt "Summarize dependent types." --stream
lake exe BranchingFlowsMoleculeTrainGenerate
lake run validateMhaH100Examples           # build + run H100 MHA fixture checks
```

## What is covered vs uncovered

The main suite (`Tests.lean`, 116 modules) is deepest in: autodiff
(`TestAD*`, `TestAutoGrad`), differential equations (`TestDiffEq*` —
`TestDiffEq.lean` alone is 3880 lines), event-skeleton physics (~40
`TestEventSkeleton*` files), Mctx (`TestMctx*`), GPU DSL/kernels/TileIR
(`TestGPUDSL`, `TestGPUKernels`, `TestGPUTileIR` — these also run standalone),
model configs and pretrained-weight loading (Qwen3/3.5/3.6, Qwen2.5-Omni,
Gemma-era FLM, KittenTTS, Qwen3TTS, Qwen3ASR, SileroVAD), the SafeTensors type
provider, checkpoints, manifolds/optimizers, and NanoChat tokenizer/tasks.

No test file imports these modules directly (verified by grep): `Tyr.Audio`,
`Tyr.Hub`, `Tyr.PRNG`, `Tyr.RL`, `Tyr.Widget`, `Tyr.Log`, `Tyr.Text`,
`Tyr.Inference`. Coverage is thin (one or two files) for `Tyr.Distributed`,
`Tyr.Sharding`, and `Tyr.Data`. Multi-process collective behavior is only
smoke-tested through the FFI surface in `Tests/TestModdedGPT.lean`.

Caveats to be aware of when extending the suite:

- **Orphaned examples.** `Examples/GPU/Run{BrownianSample,BrownianDescent,EulerMaruyamaFused,RKCombine,RKFusedSolve,MhaGB10}.lean`
  are well-documented CPU↔GPU parity harnesses with no executable target; they
  only get compiled via `lake build Examples`.
- `Tests/Test.lean` defines helpers (`encode`, `decode`, `charToInt`, ...) at
  global scope and imports `Examples.GPT.*`, so the test library depends on the
  examples library and those generic names leak into downstream test modules.

## Related guides

- [getting-started.md](getting-started.md) — environment setup required before any example
- [core/tensors.md](core/tensors.md) — the `T #[...]` shape-tracked tensors used throughout
- [serialization.md](serialization.md) — SafeTensors provider and fixtures used by the tests
- [data.md](data.md) — datasets and tokenizers behind TrainGPT/NanoChat
- [autodiff.md](autodiff.md), [modules.md](modules.md), [optimization.md](optimization.md) — subsystems under test
- [distributed.md](distributed.md) — torchrun-based NanoChat training
- [models/llms.md](models/llms.md), [models/generative.md](models/generative.md), [models/audio-speech.md](models/audio-speech.md) — models exercised by the RunHF/ASR/TTS demos
- [gpu/dsl-codegen.md](gpu/dsl-codegen.md), [gpu/kernels.md](gpu/kernels.md) — the GPU suites and kernel examples
- [diffeq.md](diffeq.md), [event-skeleton.md](event-skeleton.md), [mctx.md](mctx.md) — heavily tested subsystems
- [ffi-and-build.md](ffi-and-build.md) — the C++ FFI layer probed by `ffi_crash_probe`

This chapter is a guide, not a symbol dump: exhaustive API documentation is
generated separately by doc-gen4 (see `docbuild/`).
