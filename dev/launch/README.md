# Launch development notes

This directory contains launch preparation state that should not be rendered as
part of the public article. The launch site should describe the stable technical
contribution and selected evidence; unfinished experiments, publication gates,
and run logs belong here.

- [`branchingflows.md`](branchingflows.md): molecule/point-set sampler audit,
  corrections, current learned experiments, and promotion gates.
- [`evidence.md`](evidence.md): internal claim-to-artifact ledger, rerun state,
  and publication gates.
- [`showcase.md`](showcase.md): capture plan, recording checklist, and diagnostic
  examples that are not part of the reader-facing guide.
- [`runs/`](runs/): concise manifests, logs, metadata, and cohort evaluations;
  large checkpoints and frame directories remain on the named compute host.

## Remaining publication tasks

- Replace the article URL placeholders in the announcement copy.
- Capture the exact checkpoint revision and source revision with the model
  inference transcript.
- Run CI for the eventual launch commit. The visible successful GitHub Actions
  runs currently predate the launch work.
- Decide which examples are part of the initial article and which are separate
  technical notes.

The GPU evidence is intentionally split in two: 49 compiler/code-generation
tests run locally, while a five-module native CUDA suite is a named-device
measurement recorded under `launch/generated/gpu/`.
