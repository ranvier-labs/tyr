#!/usr/bin/env python3
"""Record effective Make configuration without invalidating unchanged builds."""

import argparse
import json
import shlex
from pathlib import Path


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependencies-output", type=Path)
    parser.add_argument("--dependency-files", nargs="*", default=[])
    parser.add_argument("entries", nargs="*")
    args = parser.parse_args()
    config = dict(entry.split("=", 1) for entry in args.entries)
    write_if_changed(args.output, json.dumps(config, indent=2, sort_keys=True) + "\n")
    if args.dependencies_output is not None:
        # GCC/Clang/NVCC depfiles escape spaces and continue long rules. Only
        # the first rule carries dependencies; later -MP rules have no inputs.
        paths: set[str] = set()
        for depfile in args.dependency_files:
            path = Path(depfile)
            if not path.exists():
                continue
            rules = path.read_text().replace("\\\n", " ").splitlines()
            if not rules:
                continue
            rule = rules[0]
            _, _, dependencies = rule.partition(":")
            for dependency in shlex.split(dependencies):
                dependency_path = Path(dependency)
                if dependency_path.exists():
                    paths.add(str(dependency_path.resolve()))
        write_if_changed(args.dependencies_output, "".join(path + "\n" for path in sorted(paths)))


if __name__ == "__main__":
    main()
