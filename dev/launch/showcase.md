# Showcase production notes

This is the internal recording and publication plan. The stable reader-facing
demonstration guide is `../../launch/showcase.md`.

## Capture plan

Keep the main video under three minutes and publish full terminal captures
separately.

### Static tensor contracts

```bash
./scripts/launch/run_shape_safety.sh
```

Show the valid signature, change `#[512, 768]` to `#[768, 512]`, and let Lean
point to the incompatible weight. Capture a 16:9 terminal recording and a
cropped still with both source and diagnostic readable.

### Training through the typed interface

```bash
lake build BranchingFlowsContinuousTrain
.lake/build/bin/BranchingFlowsContinuousTrain
```

The locally verified run reported:

```text
step=0  anchor_loss=1.484375
step=20 anchor_loss=0.389688
step=40 anchor_loss=0.050998
step=60 anchor_loss=0.001118
init_loss=1.484375 final_loss=0.000481
```

### Scientific trajectory

The Van der Pol artifact was produced on the Event Skeleton branch with:

```bash
lake build RunVanDerPolLeanPlot
.lake/build/bin/RunVanDerPolLeanPlot
```

It reported 6,668 samples and generated
`output/event-skeleton/plot_limit_cycle.svg`. Decide whether to merge and
present this direction before including it in the public sequence.

### Checkpoint execution

Use the capture harness so the transcript and provenance are stored together:

```bash
TYR_LAUNCH_MODEL=Qwen/Qwen3.5-0.8B \
  ./scripts/launch/capture_model_inference.sh
```

The caption should state model ID and revision, Tyr commit, command, device,
and whether the weights were cached. The existing Qwen3.6 27B transcript lacks
this complete provenance and should be re-captured.

### GPU execution

Record `lake exe TestGPUDSL` as compiler and code-generation evidence. It does
not execute CUDA on the local machine. The recorded named-device run was
captured with:

```bash
./scripts/launch/capture_gb10_parity.sh
```

Publish only native rows that identify the device, generated kernel, reference,
tolerance, and actual dispatch route.

### Molecule branching

The target-conditioned reference mechanism is produced with
`./scripts/launch/run_molecule_showcase.sh`. It must not be described as a
learned molecule.

The first full QM9 CUDA run lowered fixed anchor loss from `35.161827` to
`24.044386`, but its sample reached the 32-atom cap with mostly masked labels.
Keep that artifact as a diagnostic. Current sampler work and promotion gates
are recorded in `branchingflows.md`.

## Suggested three-minute cut

| Time | Visual | Spoken point |
|---:|---|---|
| 0:00 | Invalid projection rejected | This tensor bug never reaches runtime. |
| 0:20 | Valid code and type signature | Shapes are part of the program Lean checks. |
| 0:40 | Loss falling | The typed surface reaches a real training loop. |
| 1:05 | Scientific trajectory | The same project connects ML and scientific dynamics. |
| 1:30 | Recorded checkpoint execution | A model run exercises the loader and CUDA runtime boundary. |
| 2:00 | GPU DSL tests and one named-device parity result | The compiler path is tested separately from hardware execution. |
| 2:35 | Architecture diagram | One Lean system connects types, solvers, loaders, and GPU code generation. |
| 2:50 | Repository and contact | Closing frame. |

## Recording checklist

- Use one clean terminal theme, 18–22 pt type, and a fixed 120×34 window.
- Begin every full capture with `git rev-parse --short HEAD`.
- Keep raw, uncut logs next to edited clips.
- Never crop away device name, parity tolerance, or fallback status.
- Export terminal clips as MP4/WebM and a GIF only when a platform needs it.
- Export social stills at 1600×900 and 1200×1200.
- Add alt text for every chart, diagnostic, and terminal recording.
