"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.

This stage only reads Bronze and verifies it against the release manifest; it
never writes Silver/Gold/results data itself, so a second run has nothing to
duplicate.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import PurePosixPath
from typing import Any

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.models import StageResult

MANIFEST_KEY = "metadata/course-release/bundle-manifest.json"


def _source_suffix(path: str) -> str:
    """Drop the leading zone directory ("bronze/" or "raw/") for comparison."""
    return path.split("/", 1)[1] if "/" in path else path


def _load_manifest(settings: Settings) -> list[dict[str, Any]]:
    if settings.lake_backend == "local":
        manifest_path = (
            settings.local_lake_root / "metadata" / "course-release" / "bundle-manifest.json"
        )
        if not manifest_path.is_file():
            raise RuntimeError(f"Release manifest not found at {manifest_path}")
        raw = manifest_path.read_bytes()
    else:
        client = minio_client(settings)
        response = client.get_object(settings.s3_bucket, MANIFEST_KEY)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
    return json.loads(raw)["objects"]


def _object_bytes(settings: Settings, key: str) -> bytes:
    if settings.lake_backend == "local":
        return (settings.local_lake_root / key).read_bytes()
    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _unsafe_zip_members(data: bytes) -> list[str]:
    """Return archive member names unsafe to trust (path traversal, absolute paths)."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
    unsafe = []
    for name in names:
        posix = PurePosixPath(name)
        # PurePosixPath ignores backslashes, but a member using them would
        # still be interpreted as a path separator by a Windows extractor.
        if posix.is_absolute() or ".." in posix.parts or "\\" in name:
            unsafe.append(name)
    return unsafe


def run(run_id: str) -> StageResult:
    result = StageResult(stage="register_sources", run_id=run_id)
    settings = Settings.from_environment()

    manifest = _load_manifest(settings)
    inventory = dict(bronze_inventory(settings))
    inventory_by_suffix = {_source_suffix(key): key for key in inventory}

    result.input_count = len(manifest)
    matched_keys: set[str] = set()

    for entry in manifest:
        suffix = _source_suffix(entry["path"])
        key = inventory_by_suffix.get(suffix)
        if key is None:
            if entry.get("mandatory", True):
                raise RuntimeError(
                    f"Missing mandatory Bronze object for source "
                    f"'{entry['source']}': {suffix}"
                )
            continue
        matched_keys.add(key)

        actual_size = inventory[key]
        if actual_size != entry["bytes"]:
            raise RuntimeError(
                f"Size mismatch for {key}: expected {entry['bytes']} bytes, "
                f"found {actual_size}"
            )

        data = _object_bytes(settings, key)
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != entry["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {key}: expected {entry['sha256']}, "
                f"got {actual_sha256}"
            )

        if key.endswith(".zip"):
            unsafe = _unsafe_zip_members(data)
            if unsafe:
                raise RuntimeError(
                    f"Unsafe archive member path(s) in {key}: {unsafe[:5]}"
                )

    unexpected = sorted(set(inventory) - matched_keys)
    for key in unexpected:
        print(f"WARNING: unexpected Bronze object not in the release manifest: {key}")

    result.output_count = len(matched_keys)
    result.issue_count = len(unexpected)
    result.finish()
    return result

