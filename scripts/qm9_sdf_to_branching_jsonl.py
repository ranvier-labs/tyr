#!/usr/bin/env python3
"""Stream QM9 V2000 SDF records into Tyr's BranchingFlows JSONL schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LABELS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def squared_distance(a: dict[str, object], b: dict[str, object]) -> float:
    a_coord = a["coord"]
    b_coord = b["coord"]
    return sum((x - y) * (x - y) for x, y in zip(a_coord, b_coord, strict=True))


def reorder_hydrogens_near_heavy(atoms: list[dict[str, object]]) -> list[dict[str, object]]:
    """Preserve heavy-atom order and place each H immediately before its nearest heavy atom."""
    heavy = [atom for atom in atoms if atom["symbol"] != "H"]
    hydrogens = [atom for atom in atoms if atom["symbol"] == "H"]
    if not heavy or not hydrogens:
        return atoms
    by_heavy: dict[int, list[dict[str, object]]] = {i: [] for i in range(len(heavy))}
    for hydrogen in hydrogens:
        nearest = min(range(len(heavy)), key=lambda i: squared_distance(hydrogen, heavy[i]))
        by_heavy[nearest].append(hydrogen)
    ordered: list[dict[str, object]] = []
    for i, atom in enumerate(heavy):
        ordered.extend(sorted(by_heavy[i], key=lambda h: squared_distance(h, atom)))
        ordered.append(atom)
    return ordered


def parse_record(
    lines: list[str], record_index: int, keep_original_order: bool
) -> dict[str, object]:
    if len(lines) < 4:
        raise ValueError(f"record {record_index}: truncated mol block")
    name = lines[0].strip() or f"qm9_{record_index}"
    counts = lines[3].split()
    if not counts:
        raise ValueError(f"record {record_index}: missing counts line")
    atom_count = int(counts[0])
    atoms = []
    for atom_index, line in enumerate(lines[4 : 4 + atom_count]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"record {record_index} atom {atom_index}: malformed atom line")
        x, y, z = map(float, fields[:3])
        symbol = fields[3]
        if symbol not in LABELS:
            raise ValueError(f"record {record_index}: unsupported atom {symbol}")
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError(f"record {record_index}: non-finite coordinate")
        atoms.append({"symbol": symbol, "label": LABELS[symbol], "coord": [x, y, z]})
    if len(atoms) != atom_count:
        raise ValueError(f"record {record_index}: expected {atom_count} atoms, found {len(atoms)}")
    if not keep_original_order:
        atoms = reorder_hydrogens_near_heavy(atoms)
    return {"name": name, "atoms": atoms}


def records(path: Path):
    block: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.rstrip("\r\n") == "$$$$":
                if block:
                    yield block
                    block = []
            else:
                block.append(line.rstrip("\r\n"))
    if block:
        yield block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdf", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-molecules", type=int, default=0)
    parser.add_argument(
        "--keep-original-order",
        action="store_true",
        help="Do not move hydrogens next to their nearest heavy atom.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as destination:
        for index, block in enumerate(records(args.sdf)):
            if args.max_molecules > 0 and count >= args.max_molecules:
                break
            destination.write(
                json.dumps(
                    parse_record(block, index, args.keep_original_order), sort_keys=True
                )
                + "\n"
            )
            count += 1
    if count == 0:
        raise SystemExit("no molecule records found")
    print(f"wrote {count} molecule records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
