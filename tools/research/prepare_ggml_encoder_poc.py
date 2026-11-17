"""Prepare an isolated b9090 GGML encoder candidate; never run a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True,
                        help="Pinned b9090 source with aligner and single-output overlays (the encoder DLL reuses the installed optimized decoder)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new, isolated source directory; existing paths are rejected")
    parser.add_argument("--frontend-batched", action="store_true",
                        help="Add fixed FE microbatches with one whole-utterance BE and a shared scheduler")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    base = args.base_source.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output == base or output in base.parents or base in output.parents:
        raise ValueError("Output must be new and separate from the input source")
    manifest_path = repo / "native/ggml-vulkan-encoder-base.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for name, expected in manifest["required_sha256"].items():
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Base source mismatch: {name}: {actual}")
    patch = repo / "native/ggml-vulkan-encoder.patch"
    shutil.copytree(base, output)
    for check in (True, False):
        command = ["git", "apply", "--no-index", "--unsafe-paths"]
        if check:
            command.append("--check")
        subprocess.run(command + [str(patch)], cwd=output, check=True,
                       capture_output=True, text=True)
    batched_patch_sha256 = None
    if args.frontend_batched:
        batch_manifest = json.loads((repo / "native/ggml-vulkan-encoder-batched-base.json").read_text("utf-8"))
        for name, expected in batch_manifest["required_sha256"].items():
            actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Batched overlay base mismatch: {name}: {actual}")
        batch_patch = repo / "native/ggml-vulkan-encoder-batched.patch"
        for check in (True, False):
            command = ["git", "apply", "--no-index", "--unsafe-paths"]
            if check:
                command.append("--check")
            subprocess.run(command + [str(batch_patch)], cwd=output, check=True,
                           capture_output=True, text=True)
        batched_patch_sha256 = hashlib.sha256(batch_patch.read_bytes()).hexdigest()
    receipt = {
        "base": manifest["base"],
        "source_input": str(base),
        "source_output": str(output),
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "base_verified_files": len(manifest["required_sha256"]),
        "frontend_batched": args.frontend_batched,
        "batched_patch_sha256": batched_patch_sha256,
        "model_executed": False,
        "deployment_changed": False,
    }
    (output.parent / (output.name + "-prepare-receipt.json")).write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
