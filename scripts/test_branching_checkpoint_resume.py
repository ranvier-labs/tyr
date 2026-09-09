#!/usr/bin/env python3
"""Native CPU save/resume regression using the executable's embedded dataset.

Build BranchingFlowsMoleculeTrainGenerate first and supply its normal runtime
library environment. No downloads or builds are performed. Fixtures and logs
remain in the printed temporary directory for inspection, including on failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve_snapshot(checkpoint):
    pointer = checkpoint / "CURRENT"
    require(pointer.is_file(), f"snapshot pointer was not published: {pointer}")
    name = pointer.read_text().strip()
    require(re.fullmatch(r"snapshot-[A-Za-z0-9_-]+", name), f"invalid pointer: {name!r}")
    snapshot = checkpoint / ".snapshots" / name
    require(snapshot.is_dir(), f"committed snapshot does not exist: {snapshot}")
    return snapshot


def fingerprints(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def check_counts(snapshot, expected):
    metadata = json.loads((snapshot / "meta.txt").read_text())
    require(metadata.get("version") == 2, f"expected version 2 metadata in {snapshot}")
    require(metadata.get("iteration") == expected,
            f"metadata iteration should be {expected}: {metadata}")
    require(metadata.get("optimCount") == expected,
            f"metadata optimizer count should be {expected}: {metadata}")
    count_path = snapshot / "optim_muon_count.txt"
    require(count_path.read_text().strip() == str(expected),
            f"Muon count should be {expected}: {count_path}")
    require(any(snapshot.glob("param_*.pt")), f"missing model tensors in {snapshot}")
    require(any(snapshot.glob("optim_muon_momentum_*.pt")),
            f"missing Muon momentum tensors in {snapshot}")


def exercise(binary, fixtures, timeout):
    checkpoint = fixtures / "checkpoint"
    env = dict(os.environ)
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )})
    command = [
        str(binary), "--device", "cpu", "--no-generate",
        "--steps", "1", "--total-steps", "2", "--batch-size", "1",
        "--max-len", "6", "--heads", "1", "--head-dim", "2", "--mlp", "4",
        "--checkpoint-dir", str(checkpoint),
    ]

    def run(phase, resume=False, expected_error=None):
        args = command + ["--out-prefix", str(fixtures / phase)]
        if resume:
            args += ["--resume-checkpoint", str(checkpoint)]
        log = fixtures / f"{phase}.log"
        with log.open("w") as stream:
            stream.write("command=" + shlex.join(args) + "\n")
            stream.flush()
            try:
                result = subprocess.run(args, cwd=REPO, env=env, text=True,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"{phase} exceeded {timeout}s; inspect {log}") from error
        output = log.read_text()
        if expected_error is None:
            require(result.returncode == 0,
                    f"{phase} exited {result.returncode}; inspect {log}\n{output[-6000:]}")
        else:
            require(result.returncode != 0,
                    f"{phase} incorrectly accepted corrupt optimizer state; inspect {log}")
            require(expected_error in output,
                    f"{phase} did not report {expected_error!r}; inspect {log}\n{output[-6000:]}")
            require("molecule_train step=" not in output,
                    f"{phase} trained before rejecting corrupt optimizer state; inspect {log}")
        print(f"PASS {phase}", flush=True)
        return output

    run("save")
    original = resolve_snapshot(checkpoint)
    check_counts(original, 1)
    original_files = fingerprints(original)

    output = run("resume", resume=True)
    require("Muon optimizer checkpoint" in output and "at count=1" in output,
            f"resume did not load the saved optimizer; inspect {fixtures / 'resume.log'}")
    latest = resolve_snapshot(checkpoint)
    require(latest != original, "resume overwrote the committed snapshot directory")
    check_counts(latest, 2)
    require(original.is_dir() and fingerprints(original) == original_files,
            f"resume removed or modified the previous snapshot: {original}")
    snapshots_before = set((checkpoint / ".snapshots").iterdir())
    latest_files = fingerprints(latest)
    count_path = latest / "optim_muon_count.txt"
    saved_count = count_path.read_bytes()
    try:
        count_path.write_text("0")
        run("mismatched-count", resume=True, expected_error="Muon optimizer count mismatch")
        count_path.unlink()
        run("missing-count", resume=True, expected_error="missing Muon optimizer checkpoint")
    finally:
        count_path.write_bytes(saved_count)
    require(resolve_snapshot(checkpoint) == latest, "failed resume changed CURRENT")
    require(set((checkpoint / ".snapshots").iterdir()) == snapshots_before,
            "failed resume created an unexpected snapshot")
    require(fingerprints(latest) == latest_files, "failed resume modified checkpoint tensors")
    require(fingerprints(original) == original_files, "failed resume modified the previous snapshot")
    print("PASS snapshot retention and failure isolation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=REPO / ".lake/build/bin/BranchingFlowsMoleculeTrainGenerate")
    parser.add_argument("--timeout", type=float, default=120,
                        help="maximum seconds for each native invocation (default: 120)")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"missing executable {binary}; build BranchingFlowsMoleculeTrainGenerate first")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    fixtures = Path(tempfile.mkdtemp(prefix="tyr-branching-checkpoint-resume-"))
    print(f"Fixtures and logs: {fixtures}", flush=True)
    try:
        exercise(binary, fixtures, args.timeout)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}\nFixtures retained at {fixtures}", file=sys.stderr)
        return 1
    print("Branching checkpoint save/resume regression passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
