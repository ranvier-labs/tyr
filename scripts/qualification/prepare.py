#!/usr/bin/env python3
"""Fetch/check immutable real-model inputs in a dedicated qualification cache.

Defaults to verification only. --download permits public downloads into --cache;
existing checkout/weights directories are never touched. Every model file is
checked against the official pinned revision's LFS SHA-256 or Git blob SHA-1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

MANIFEST = Path(__file__).with_name("fixtures.json")


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe fixture path: {relative}")
    return root / relative


def verify(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    sha = hashlib.sha256() if "sha256" in spec else hashlib.sha1()
    if "sha256" not in spec:
        sha.update(f"blob {spec['size']}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest() == spec.get("sha256", spec.get("git_blob_sha1"))


def ensure_file(path, spec, url, download):
    if verify(path, spec):
        return
    if not download:
        raise RuntimeError(f"Missing or corrupt pinned fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"Downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if not verify(partial, spec):
            raise RuntimeError(f"Downloaded fixture failed pinned checksum: {path}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def prepare(cache, manifest, download=False):
    paths = {}
    for model in manifest["models"]:
        directory = cache / model["name"] / model["revision"]
        for spec in model["files"]:
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{spec['path']}"
            ensure_file(safe_path(directory, spec["path"]), spec, url, download)
        paths[model["name"]] = str(directory.resolve())
    audio = manifest["audio"]
    audio_path = safe_path(cache, audio["path"])
    ensure_file(audio_path, audio, audio["url"], download)
    paths["audio"] = str(audio_path.resolve())
    reference = manifest["qwen_repository"]
    directory = cache / ("qwen-reference-" + reference["revision"])
    if not directory.exists() and download:
        subprocess.run(["git", "clone", "--no-checkout", reference["url"], str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", reference["revision"]], check=True)
    if not directory.exists():
        raise RuntimeError(f"Pinned Qwen reference checkout missing: {directory}")
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != reference["revision"] or dirty:
        raise RuntimeError(f"Qwen reference must be clean at {reference['revision']}: {directory}")
    paths["qwen-reference"] = str(directory.resolve())
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--paths-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    paths = prepare(args.cache, manifest, args.download)
    args.paths_output.parent.mkdir(parents=True, exist_ok=True)
    args.paths_output.write_text(json.dumps(paths, indent=2) + "\n")
    print("All pinned model/reference/audio inputs verified", flush=True)


if __name__ == "__main__":
    main()
