#!/usr/bin/env python3
"""Exercise the constellation launcher's checkpoint handling without Torch/GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).parent / "launch/run_branching_constellation_generate.sh"


class SnapshotLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tyr-branching-snapshot-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "training/checkpoint"
        self.output = self.root / "cohort"
        self.launcher = self.root / "scripts/launch" / LAUNCHER.name
        self.launcher.parent.mkdir(parents=True)
        shutil.copyfile(LAUNCHER, self.launcher)
        self.write("dataset.jsonl", "{}\n")
        for source in (
            "Tyr/Model/BranchingFlows.lean",
            "Tyr/Model/BranchingFlows/Molecule.lean",
            "Tyr/Model/BranchingFlows/MoleculeTransformer.lean",
            "Examples/BranchingFlows/MoleculeTrainGenerate.lean",
        ):
            self.write(source, "-- fixture\n")
        self.write(
            ".lake/build/bin/BranchingFlowsMoleculeTrainGenerate",
            """#!/usr/bin/env python3
import os
from pathlib import Path
import sys
args = sys.argv[1:]
def arg(name):
    return args[args.index(name) + 1]
checkpoint = Path(arg('--resume-checkpoint'))
assert str(checkpoint) == os.environ['EXPECTED_SNAPSHOT'], checkpoint
Path('loaded_snapshot.txt').write_text(str(checkpoint))
if os.environ.get('REPLACE_CURRENT'):
    Path(os.environ['TYR_CONSTELLATION_CHECKPOINT'], 'CURRENT').write_text(
        os.environ['REPLACE_CURRENT'])
if os.environ.get('MUTATE_PARAMETERS'):
    (checkpoint / 'param_0.pt').write_text('changed during generation')
for index in range(int(arg('--sample-count'))):
    Path(arg('--out-prefix') + str(index) + '_trajectory.jsonl').write_text('{}\\n')
""",
            executable=True,
        )
        self.write(
            "scripts/launch/evaluate_branching_constellations.py",
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[sys.argv.index('--out') + 1]).write_text('{}\\n')\n",
        )
        # Keep these fixtures independent of GNU coreutils and GPU availability.
        self.write(
            "test-bin/sha256sum",
            """#!/usr/bin/env python3
import hashlib
from pathlib import Path
import sys
if len(sys.argv) == 1:
    print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest() + '  -')
else:
    for filename in sys.argv[1:]:
        print(hashlib.sha256(Path(filename).read_bytes()).hexdigest() + '  ' + filename)
""",
            executable=True,
        )
        self.write("test-bin/nvidia-smi", "#!/usr/bin/env bash\nexit 0\n", executable=True)

    def write(self, relative, text, executable=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if executable:
            path.chmod(0o755)
        return path

    def make_checkpoint(self, name=None):
        path = self.checkpoint if name is None else self.checkpoint / ".snapshots" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.txt").write_text(f"metadata for {name or 'legacy'}\n")
        (path / "param_0.pt").write_text(f"tensor for {name or 'legacy'}\n")
        return path

    def run_launcher(self, expected_snapshot, **extra_env):
        env = dict(os.environ)
        env.update(
            PATH=str(self.root / "test-bin") + os.pathsep + env["PATH"],
            TYR_CONSTELLATION_DATA=str(self.root / "dataset.jsonl"),
            TYR_CONSTELLATION_CHECKPOINT=str(self.checkpoint),
            TYR_CONSTELLATION_OUTPUT=str(self.output),
            TYR_CONSTELLATION_DEVICE="cpu",
            TYR_CONSTELLATION_SAMPLES="1",
            TYR_CONSTELLATION_FIXED_LABELS="0",
            EXPECTED_SNAPSHOT=str(expected_snapshot),
            **extra_env,
        )
        return subprocess.run(
            ["bash", str(self.launcher)], env=env, text=True, capture_output=True, timeout=30
        )

    def assert_success(self, result, expected_snapshot):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = (self.output / "manifest.txt").read_text()
        self.assertIn(f"checkpoint_snapshot={expected_snapshot}\n", manifest)
        self.assertIn("checkpoint_unchanged=true\n", manifest)
        self.assertEqual((self.root / "loaded_snapshot.txt").read_text(), str(expected_snapshot))
        self.assertTrue((self.output / "evaluation.json").is_file())

    def test_legacy_flat_checkpoint(self):
        snapshot = self.make_checkpoint()
        self.assert_success(self.run_launcher(snapshot), snapshot)

    def test_snapshot_without_flat_files(self):
        snapshot = self.make_checkpoint("snapshot-1")
        (self.checkpoint / "CURRENT").write_text("snapshot-1\n")
        self.assert_success(self.run_launcher(snapshot), snapshot)

    def test_snapshot_stays_pinned_when_current_changes(self):
        self.make_checkpoint()  # Stale legacy files must not be loaded or hashed.
        snapshot = self.make_checkpoint("snapshot-1")
        self.make_checkpoint("snapshot-2")
        (self.checkpoint / "CURRENT").write_text("snapshot-1")
        result = self.run_launcher(snapshot, REPLACE_CURRENT="snapshot-2")
        self.assert_success(result, snapshot)
        self.assertEqual((self.checkpoint / "CURRENT").read_text(), "snapshot-2")

    def test_rejects_invalid_or_missing_snapshot(self):
        self.make_checkpoint()  # A broken pointer must not silently fall back.
        for pointer in (
            "", "../escape", "snapshot-a/other", "snapshot-a\\other",
            "snapshot-a.txt", "snapshot-a\nsnapshot-b", "snapshot-missing",
        ):
            with self.subTest(pointer=pointer):
                (self.checkpoint / "CURRENT").write_text(pointer)
                result = self.run_launcher(self.checkpoint)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("checkpoint snapshot", result.stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root / "loaded_snapshot.txt").exists())

    def test_detects_mutation_of_pinned_snapshot(self):
        snapshot = self.make_checkpoint("snapshot-1")
        (self.checkpoint / "CURRENT").write_text("snapshot-1")
        result = self.run_launcher(snapshot, MUTATE_PARAMETERS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checkpoint changed during generate-only execution", result.stderr)
        self.assertIn("checkpoint_unchanged=false\n", (self.output / "manifest.txt").read_text())
        self.assertFalse((self.output / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
