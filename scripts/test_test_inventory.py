#!/usr/bin/env python3
"""Regression checks for orphaned Lean tests and standalone test executables."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from test_inventory import validate
import test_inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tyr-test-inventory-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "Tests").mkdir()
        (self.root / "lakefile.lean").write_text(
            "lean_exe suite where\n  root := `Tests.Run\n")
        (self.root / "Tests/Run.lean").write_text("import Tests.TestA\ndef main := pure 0\n")
        (self.root / "Tests/TestA.lean").write_text("@[test]\ndef testA := pure ()\n")
        self.manifest = {"required": {"suite": []}, "optional": {}}

    def test_new_test_module_requires_routing(self):
        self.assertEqual(validate(self.root, self.manifest), [])
        (self.root / "Tests/TestB.lean").write_text("@[test]\ndef testB := pure ()\n")
        self.assertIn("Unrouted @[test] module: Tests.TestB", validate(self.root, self.manifest))

    def test_standalone_main_requires_assignment(self):
        with (self.root / "lakefile.lean").open("a") as out:
            out.write("lean_exe standalone where\n  root := `Tests.Standalone\n")
        (self.root / "Tests/Standalone.lean").write_text("def main := pure ()\n")
        self.assertTrue(any("Standalone" in error for error in validate(self.root, self.manifest)))
        self.manifest["optional"]["standalone"] = "Requires CUDA"
        self.assertEqual(validate(self.root, self.manifest), [])

    def test_missing_main_is_rejected(self):
        (self.root / "Tests/Run.lean").write_text("import Tests.TestA\n")
        self.assertTrue(any("no main" in error for error in validate(self.root, self.manifest)))

    def test_failed_suite_preserves_exit_status_log_and_report(self):
        (self.root / "scripts").mkdir()
        (self.root / "scripts/test_suites.json").write_text(json.dumps(self.manifest))
        binary = self.root / ".lake/build/bin/suite"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\necho retained-diagnostic\nexit 7\n")
        binary.chmod(0o755)
        report = self.root / "reports/suites.json"
        with patch.object(test_inventory, "REPO", self.root), \
                patch("sys.argv", ["test_inventory.py", "--run", "--report", str(report)]):
            self.assertEqual(test_inventory.main(), 7)
        entry = json.loads(report.read_text())["suites"]["suite"]
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["exit_code"], 7)
        self.assertIn("retained-diagnostic", (report.parent / "suite.log").read_text())


if __name__ == "__main__":
    unittest.main()
