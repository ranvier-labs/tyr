#!/usr/bin/env python3
"""Hardware-free regressions for fixture integrity and strict execution gates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare import ensure_file, safe_path, verify
from readiness import configuration
from run import validate_result
from summarize import summarize


class QualificationTests(unittest.TestCase):
    def test_empty_skipped_and_failed_gpu_runs_cannot_qualify(self):
        for output, code in [
            ("", 0), ("all good", 0),
            ("[gpu-coverage] executed=0 skipped=0 failed=0 selected=0 strict=true", 0),
            ("[gpu-coverage] executed=4 skipped=1 failed=0 selected=5 strict=true", 0),
            ("[gpu-coverage] executed=4 skipped=0 failed=1 selected=4 strict=true", 0),
            ("[gpu-coverage] executed=4 skipped=0 failed=0 selected=4 strict=true", 1),
            ("[gpu-coverage] executed=4 skipped=0 failed=0 selected=4 strict=false", 0),
        ]:
            with self.subTest(output=output, code=code), self.assertRaises(ValueError):
                validate_result("gpu", output, code)
        self.assertEqual(validate_result("gpu",
            "[gpu-coverage] executed=10 skipped=0 failed=0 selected=10 strict=true", 0), 10)

    def test_model_and_cache_checks_require_actual_success_markers(self):
        for kind in ("parity", "asr", "decode", "laguna", "native"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_result(kind, "", 0)
        with self.assertRaises(ValueError):
            validate_result("asr", "SKIP missing model\n[qwen3tts-asr] PASS", 0)
        with self.assertRaises(ValueError):
            validate_result("laguna", "CUDA not available; skipped CUDA cases.\nAll Laguna model tests passed.", 0)
        with self.assertRaises(ValueError):
            validate_result("laguna", "All Laguna model tests passed.", 0)
        self.assertEqual(validate_result("laguna", "CACHE_BENCH device=cuda:0 capacity=128\nAll Laguna model tests passed.", 0), 1)
        self.assertEqual(validate_result("laguna", "PASS: session rejects skipped positions without damaging state\nCACHE_BENCH device=cuda:0 capacity=128\nAll Laguna model tests passed.", 0), 1)

    def test_fixture_cache_rejects_corruption_missing_files_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fixture.bin"
            content = b"pinned fixture"
            spec = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            with self.assertRaises(RuntimeError):
                ensure_file(path, spec, "https://invalid.example/unused", False)
            path.write_bytes(content)
            self.assertTrue(verify(path, spec))
            blob = {"size": len(content), "git_blob_sha1": hashlib.sha1(b"blob 14\0" + content).hexdigest()}
            self.assertTrue(verify(path, blob))
            path.write_bytes(b"corrupt bytes!")
            self.assertFalse(verify(path, spec))
            with self.assertRaises(RuntimeError):
                ensure_file(path, spec, "https://invalid.example/unused", False)
            with self.assertRaises(ValueError):
                safe_path(root, "../escape")

    def test_runner_association_is_explicit(self):
        with self.assertRaises(ValueError):
            configuration("ranvier-labs/tyr", "", "")
        with self.assertRaises(ValueError):
            configuration("ranvier-labs/tyr", "/torch", "")
        labels = configuration("cpehle/tyr", "/torch", "")
        self.assertIn("gb10", json.loads(labels))
        self.assertIn("tyr-qualification", json.loads(configuration("ranvier-labs/tyr", "/torch",
            '["self-hosted", "Linux", "ARM64", "tyr-qualification", "gb10"]')))

    def test_final_summary_fails_missing_runtime_only_or_skipped_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(summarize(path, ["gpu"])["status"], "failed")
            for status, executed, skipped in [("runtime_ready", 0, 0), ("passed", 0, 0), ("passed", 1, 1)]:
                (path / "gpu.json").write_text(json.dumps({"status": status, "executed": executed, "skipped": skipped, "failed": 0}))
                self.assertEqual(summarize(path, ["gpu"])["status"], "failed")
            (path / "gpu.json").write_text(json.dumps({"status": "passed", "executed": 10, "skipped": 0, "failed": 0}))
            self.assertEqual(summarize(path, ["gpu"])["status"], "passed")
            self.assertEqual(summarize(path, ["gpu", "models"])["status"], "failed")


if __name__ == "__main__":
    unittest.main()
