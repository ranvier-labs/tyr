# BranchingFlows launch experiment notes

This is the development log for the variable-cardinality examples. It is not
launch-site copy.

## Public boundary

The target-conditioned water trajectory is a sampler/mechanism example. It is
not a learned molecule. Its scientific figure and raw trajectory are:

- `launch/generated/molecule/branching-trajectory.{svg,png}`
- `launch/generated/molecule/water_trajectory.jsonl`

The first learned QM9 run is a diagnostic, not evidence of chemically valid
generation. It reached the 32-particle cap, retained mostly masked labels, and
kept descendants coincident. The earlier storyboard must not be used as a
capability figure.

## Correctness work completed

- Runtime trajectories now assign stable particle, parent, and birth-event IDs.
- JSONL export contains raw coordinates and explicit split/delete events.
- Lineage reconstruction rejects malformed frame/event alignment and
  contradictory events.
- Training roots are sampled from the Gaussian base distribution instead of
  always starting at zero.
- Training bridges and learned generation both sample the OU/DFM base process.
- Identical descendants receive independent stochastic updates after a split.
- Learned inference enables the configured deletion process.
- Inference tensors and parameters stay on the requested device.
- `--grad-clip` applies per-tensor gradient-norm clipping instead of being a
  no-op.
- Population-cap overflow raises an error instead of silently truncating state
  and invalidating lineage.
- The SDF preprocessor moves hydrogens next to their nearest heavy atom while
  preserving heavy-atom order.
- The train/evaluation split is seeded, shuffled, and reserves ten percent of
  datasets larger than a few records.

Focused result: 40/40 experimental BranchingFlows tests pass. A local learned
smoke run on the embedded four-molecule fixture produced real split and delete
events and emitted valid lineage JSONL. It is too small to interpret as a model
result.

## Learned constellation experiment

The intermediate experiment removes chemistry from the question. The dataset
contains 4,096 slightly perturbed regular polygons with either three or six
points. The existing molecule-shaped transformer is used as a generic labeled
3D point-set model, so the bridge, event heads, sampler, GPU training path, and
lineage exporter are the same ones used by the QM9 experiment.

Artifacts and commands:

- dataset generator: `scripts/launch/generate_branching_constellations.py`
- dataset: `launch/generated/branching-constellations/dataset.jsonl`
- training harness: `scripts/launch/run_branching_constellation_train.sh`
- generation/evaluation harness:
  `scripts/launch/run_branching_constellation_generate.sh`
- cohort evaluator: `scripts/launch/evaluate_branching_constellations.py`

The evaluator reproduces the executable's seeded 64-bit shuffle and compares
the generated cohort with the reserved ten-percent evaluation split.

The aligned control dataset is reproduced with:

```bash
python3 scripts/launch/generate_branching_constellations.py \
  --out launch/generated/branching-constellations/dataset-aligned-unlabeled.jsonl \
  --count 4096 --seed 20260709 --rotation-mode fixed --label 1 \
  --radius-jitter 0.05 --anisotropy-jitter 0.05 \
  --radial-noise 0.01 --z-noise 0.005
```

For the aligned point-set control, `--fixed-labels` removes the stochastic
discrete DFM bridge and step from a label that only denotes “point”. The model
still fits the mask-to-point logits with ordinary cross entropy and mode
updates; coordinates and branching events remain learned. This option is not
used for molecular data.

The first 3,000-step, batch-16 run in the isolated Spark checkout reduced fixed
bridge loss from `137.626389` to `1.885227` (held-out bridge loss `4.253782`). A
64-sample generation cohort was not suitable for presentation: unrestricted
coordinate targets diverged, while adding a bound only at inference prevented
divergence but collapsed spatial extent. This identifies a training/inference
architecture mismatch rather than a rendering problem.

A second 3,000-step run trained with the coordinate bound present in both
paths, reduced the label vocabulary to the two symbols used by the dataset, and
gave the split objective more weight. Fixed bridge loss decreased from
`89.404999` to `22.740419`; held-out bridge loss was `2.784503`. This did not
produce a useful generative cohort. At split-logit cap `-2.5`, all 16 screening
samples were finite and none hit the population cap, but mean radius was
`0.0720` against `0.9996` in the reserved split, mean nearest-neighbor distance
was `0.0587` against `1.3314`, and only `0.4444` of labels resolved. Raising the
split rate increased cardinality and caused cap hits without correcting the
spatial collapse. This run remains a diagnostic.

Concise records are stored under
[`runs/branching-constellations/rotated-cap2/`](runs/branching-constellations/rotated-cap2/):
the training manifest/log/metadata and both 16-sample screen evaluations. The
checkpoint and frame-by-frame cohorts remain in the isolated compute checkout.

The likely confound is rotational averaging: the first synthetic dataset
rotates every polygon uniformly, while coordinate targets are trained by
squared error. An aligned control therefore fixes polygon orientation and
treats the sole synthetic point type as a fixed label rather than a discrete
DFM variable. This control still learns coordinates and split events. It is a
point-set diagnostic, not a molecule result.

The aligned control reduced fixed bridge loss from `6.544919` to `0.325558`
and reported held-out bridge loss `0.864706`. Generation improved but did not
pass the cohort gates. With 96 integration steps and split cap `-2.25`, a
16-sample screen had mean radius `0.3247`, mean nearest-neighbor distance
`0.4263`, and resolved-label fraction `0.3158`. An eight-sample, 256-step
screen reached radius `0.6818` and nearest-neighbor distance `0.9058`, but
underpopulated the target and resolved only `0.5714` of labels. Results were
not monotone at 512 steps or at a higher split rate. All screens remained
finite; none justifies selecting an individual trajectory as a model result.

The aligned training record and four resolution/rate screens are under
[`runs/branching-constellations/aligned-fixedlabels/`](runs/branching-constellations/aligned-fixedlabels/).

Run manifests record the requested CUDA device, source and dataset hashes,
seed, and command. The machine happens to contain an NVIDIA GB10; this is run
metadata rather than part of the experiment definition.

## Promotion gates

Do not put a learned trajectory on the launch site until a held-out cohort has:

- finite coordinates for every generated sample;
- no population-cap hits;
- spatially separated descendants after split events;
- resolved labels;
- an atom/point-count distribution reported against held-out data;
- radius and nearest-neighbor statistics reported against held-out data;
- a raw trajectory, checkpoint, seed, source hashes, and device manifest.

For QM9, add chemical validity and connectedness over a cohort before making
any chemistry claim.
