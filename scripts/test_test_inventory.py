#!/usr/bin/env python3
"""Regression checks for orphaned Lean tests and standalone test executables."""

from pathlib import Path
import tempfile
import unittest

from test_inventory import validate


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


if __name__ == "__main__":
    unittest.main()
