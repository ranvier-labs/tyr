#!/usr/bin/env python3
"""Fail qualification if any requested report is missing, failed, empty or skipped."""
import argparse
import json
import os
from pathlib import Path


def summarize(directory, kinds):
    result = {"status": "passed", "executed": 0, "skipped": 0, "failed": 0, "reports": {}}
    for kind in kinds:
        path = directory / (kind + ".json")
        try:
            report = json.loads(path.read_text())
            valid = (report["status"] == "passed" and report["executed"] > 0
                     and report["skipped"] == 0 and report["failed"] == 0)
            result["reports"][kind] = report["status"]
            for count in ("executed", "skipped", "failed"):
                result[count] += report[count]
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
            result["reports"][kind] = "missing_or_invalid"
        if not valid:
            result["status"] = "failed"
            result["failed"] = max(1, result["failed"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("kinds", nargs="+", choices=("gpu", "models"))
    args = parser.parse_args()
    result = summarize(args.directory, args.kinds)
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write("Strict qualification: " + json.dumps(result) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
