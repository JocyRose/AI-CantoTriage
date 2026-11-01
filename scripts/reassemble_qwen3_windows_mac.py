#!/usr/bin/env python3
"""Reassemble the split first safetensors shard for the Windows/macOS model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("导诊模型/Qwen3-1.7B-windows-mac-compatible"),
        help="directory containing model_parts_manifest.json",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output shard")
    args = parser.parse_args()

    model_dir = args.model_dir
    manifest_path = model_dir / "model_parts_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = model_dir / manifest["output"]
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing file: {output}; use --force")

    total = 0
    with output.open("wb") as destination:
        for part in manifest["parts"]:
            path = model_dir / part["name"]
            if not path.is_file():
                raise SystemExit(f"missing model part: {path}")
            actual_size = path.stat().st_size
            if actual_size != part["size"]:
                raise SystemExit(f"size mismatch for {path}: {actual_size} != {part['size']}")
            actual_hash = sha256_file(path)
            if actual_hash != part["sha256"]:
                raise SystemExit(f"SHA-256 mismatch for {path}")
            with path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
            total += actual_size

    if total != manifest["output_size"]:
        output.unlink(missing_ok=True)
        raise SystemExit(f"reassembled size mismatch: {total} != {manifest['output_size']}")
    actual_output_hash = sha256_file(output)
    if actual_output_hash != manifest["output_sha256"]:
        output.unlink(missing_ok=True)
        raise SystemExit("reassembled output SHA-256 mismatch")
    print(f"Reassembled {output} ({total} bytes)")
    print(f"SHA-256: {actual_output_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
