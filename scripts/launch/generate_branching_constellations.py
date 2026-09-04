#!/usr/bin/env python3
"""Generate a deterministic variable-cardinality 3D point-set dataset.

Each item is a slightly perturbed regular polygon with either three or six
points.  The dataset is deliberately simpler than molecular generation: it
tests whether a BranchingFlows model learns motion and cardinality without
requiring a chemical validity model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def make_record(
    index: int,
    rng: random.Random,
    *,
    rotation_mode: str,
    label: int,
    radius_jitter: float,
    anisotropy_jitter: float,
    radial_noise: float,
    z_noise: float,
) -> dict[str, object]:
    count = 3 if rng.random() < 0.5 else 6
    rotation = rng.uniform(-math.pi, math.pi) if rotation_mode == "uniform" else 0.0
    radius = rng.uniform(1.0 - radius_jitter, 1.0 + radius_jitter)
    anisotropy = rng.uniform(1.0 - anisotropy_jitter, 1.0 + anisotropy_jitter)
    atoms = []
    for point in range(count):
        angle = rotation + 2.0 * math.pi * point / count
        r = radius + rng.gauss(0.0, radial_noise)
        x = r * math.cos(angle)
        y = anisotropy * r * math.sin(angle)
        z = rng.gauss(0.0, z_noise)
        atoms.append({"label": label, "coord": [x, y, z]})
    return {
        "name": f"constellation_{index:06d}_{count}",
        "family": f"polygon_{count}",
        "atoms": atoms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--rotation-mode", choices=("uniform", "fixed"), default="uniform")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--radius-jitter", type=float, default=0.15)
    parser.add_argument("--anisotropy-jitter", type=float, default=0.15)
    parser.add_argument("--radial-noise", type=float, default=0.025)
    parser.add_argument("--z-noise", type=float, default=0.015)
    args = parser.parse_args()
    if args.count < 2:
        raise SystemExit("--count must be at least 2")
    if args.label < 0:
        raise SystemExit("--label must be nonnegative")
    for name in ("radius_jitter", "anisotropy_jitter", "radial_noise", "z_noise"):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be nonnegative")
    if args.radius_jitter >= 1.0 or args.anisotropy_jitter >= 1.0:
        raise SystemExit("radius and anisotropy jitter must be less than one")
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as destination:
        for index in range(args.count):
            destination.write(
                json.dumps(
                    make_record(
                        index,
                        rng,
                        rotation_mode=args.rotation_mode,
                        label=args.label,
                        radius_jitter=args.radius_jitter,
                        anisotropy_jitter=args.anisotropy_jitter,
                        radial_noise=args.radial_noise,
                        z_noise=args.z_noise,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        f"wrote {args.count} constellation records to {args.out} "
        f"(seed={args.seed}, rotation={args.rotation_mode}, label={args.label})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
