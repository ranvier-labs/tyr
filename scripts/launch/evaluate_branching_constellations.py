#!/usr/bin/env python3
"""Evaluate a cohort of lineage-aware generated constellation trajectories."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path


LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
U64_MASK = (1 << 64) - 1


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def final_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    states = [row for row in rows if row.get("type") == "state"]
    final_step = max(int(row["step"]) for row in states)
    return [row for row in states if int(row["step"]) == final_step]


def point_statistics(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    center = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
    radii = [math.dist(point, center) for point in points]
    nearest = []
    if len(points) > 1:
        for i, point in enumerate(points):
            nearest.append(min(math.dist(point, other) for j, other in enumerate(points) if i != j))
    return sum(radii) / len(radii), (sum(nearest) / len(nearest) if nearest else 0.0)


def evaluation_split(rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    """Reproduce MoleculeTrainGenerate.splitTrainEval exactly."""
    if len(rows) <= 1:
        return rows
    shuffled = list(rows)
    state = seed & U64_MASK
    for offset in range(len(shuffled)):
        index = len(shuffled) - 1 - offset
        if index > 0:
            state = (state * LCG_MULTIPLIER + LCG_INCREMENT) & U64_MASK
            swap_index = state % (index + 1)
            shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    evaluation_count = max(1, len(shuffled) // 10)
    return shuffled[len(shuffled) - evaluation_count :]


def target_statistics(dataset: Path, split_seed: int) -> dict[str, object]:
    counts: Counter[int] = Counter()
    radii = []
    nearest = []
    all_rows = read_jsonl(dataset)
    rows = evaluation_split(all_rows, split_seed)
    for row in rows:
        points = [tuple(float(value) for value in atom["coord"]) for atom in row["atoms"]]
        counts[len(points)] += 1
        radius, nn = point_statistics(points)
        radii.append(radius)
        nearest.append(nn)
    total = sum(counts.values())
    return {
        "samples": total,
        "source_samples": len(all_rows),
        "split_seed": split_seed,
        "count_distribution": {str(key): value / total for key, value in sorted(counts.items())},
        "mean_radius": sum(radii) / len(radii),
        "mean_nearest_neighbor": sum(nearest) / len(nearest),
    }


def generated_statistics(paths: list[Path], max_len: int) -> dict[str, object]:
    counts: Counter[int] = Counter()
    radii = []
    nearest = []
    finite = 0
    resolved = 0
    labels = 0
    split_events = 0
    delete_events = 0
    descendant_trials = 0
    descendants_separated = 0
    max_abs_coordinate = 0.0
    for path in paths:
        rows = read_jsonl(path)
        states = [row for row in rows if row.get("type") == "state"]
        events = [row for row in rows if row.get("type") == "event"]
        final = final_rows(rows)
        points = [(float(row["x"]), float(row["y"]), float(row["z"])) for row in final]
        is_finite = all(math.isfinite(value) for point in points for value in point)
        finite += int(is_finite)
        if is_finite:
            radius, nn = point_statistics(points)
            radii.append(radius)
            nearest.append(nn)
            max_abs_coordinate = max(max_abs_coordinate, *(abs(value) for point in points for value in point))
        counts[len(points)] += 1
        labels += len(final)
        resolved += sum(int(int(row["label"]) != 0) for row in final)
        by_particle: dict[int, list[dict[str, object]]] = {}
        for row in states:
            by_particle.setdefault(int(row["particle_id"]), []).append(row)
        for event in events:
            if event["kind"] == "split":
                split_events += 1
                child_ids = [int(value) for value in event.get("child_ids", [])]
                if len(child_ids) >= 2:
                    common_steps = set(int(row["step"]) for row in by_particle.get(child_ids[0], []))
                    for child in child_ids[1:]:
                        common_steps &= set(int(row["step"]) for row in by_particle.get(child, []))
                    later = sorted(step for step in common_steps if step > 0)
                    if later:
                        step = later[min(1, len(later) - 1)]
                        child_points = []
                        for child in child_ids:
                            row = next(row for row in by_particle[child] if int(row["step"]) == step)
                            child_points.append((float(row["x"]), float(row["y"]), float(row["z"])))
                        descendant_trials += 1
                        if max(math.dist(child_points[0], point) for point in child_points[1:]) > 1.0e-5:
                            descendants_separated += 1
            elif event["kind"] == "delete":
                delete_events += 1
    total = len(paths)
    report = {
        "samples": total,
        "finite_fraction": finite / total,
        "max_len_hit_fraction": sum(value for key, value in counts.items() if key >= max_len) / total,
        "resolved_label_fraction": resolved / labels if labels else 0.0,
        "count_distribution": {str(key): value / total for key, value in sorted(counts.items())},
        "mean_radius": sum(radii) / len(radii) if radii else None,
        "mean_nearest_neighbor": sum(nearest) / len(nearest) if nearest else None,
        "max_abs_final_coordinate": max_abs_coordinate,
        "split_events": split_events,
        "delete_events": delete_events,
        "descendant_separation_fraction": (
            descendants_separated / descendant_trials if descendant_trials else None
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, help="Glob for trajectory JSONL files")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260709)
    args = parser.parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.trajectories))]
    if not paths:
        raise SystemExit(f"no trajectories match {args.trajectories!r}")
    report = {
        "schema": "tyr.branching-constellation-eval.v1",
        "target": target_statistics(args.data, args.split_seed),
        "generated": generated_statistics(paths, args.max_len),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
