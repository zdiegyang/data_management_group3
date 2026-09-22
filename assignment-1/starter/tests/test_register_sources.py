import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from quantum_lake_student.stages import register_sources


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _seed_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mandatory: bool = True) -> Path:
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(tmp_path))

    zip_path = tmp_path / "bronze" / "source=qec_syndromes" / "syndromes_dataset.zip"
    _write_zip(zip_path, {"d-3_pfr-0.001000_nb-10M.csv": b"labels,syndromes,quantity\n"})
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    manifest = {
        "bundle_version": 3,
        "release_name": "test-release",
        "objects": [
            {
                "path": "raw/source=qec_syndromes/syndromes_dataset.zip",
                "bytes": zip_path.stat().st_size,
                "sha256": sha256,
                "source": "qec_syndromes",
                "mandatory": mandatory,
            }
        ],
    }
    manifest_path = tmp_path / "metadata" / "course-release" / "bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return zip_path


def test_run_verifies_matching_checksum_and_registers_object(tmp_path, monkeypatch):
    _seed_release(tmp_path, monkeypatch)

    result = register_sources.run("run-1")

    assert result.input_count == 1
    assert result.output_count == 1
    assert result.issue_count == 0


def test_run_is_idempotent_across_repeated_calls(tmp_path, monkeypatch):
    _seed_release(tmp_path, monkeypatch)

    first = register_sources.run("run-1")
    second = register_sources.run("run-2")

    assert (first.input_count, first.output_count, first.issue_count) == (
        second.input_count,
        second.output_count,
        second.issue_count,
    )


def test_run_raises_on_missing_mandatory_object(tmp_path, monkeypatch):
    zip_path = _seed_release(tmp_path, monkeypatch)
    zip_path.unlink()

    with pytest.raises(RuntimeError, match="Missing mandatory"):
        register_sources.run("run-1")


def test_run_raises_on_checksum_mismatch(tmp_path, monkeypatch):
    zip_path = _seed_release(tmp_path, monkeypatch)
    original = bytearray(zip_path.read_bytes())
    original[-1] ^= 0xFF  # flip a byte without changing the file size
    zip_path.write_bytes(bytes(original))

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        register_sources.run("run-1")


def test_run_raises_on_unsafe_archive_member(tmp_path, monkeypatch):
    zip_path = _seed_release(tmp_path, monkeypatch)
    _write_zip(zip_path, {"../evil.csv": b"labels,syndromes,quantity\n"})

    manifest_path = tmp_path / "metadata" / "course-release" / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objects"][0]["bytes"] = zip_path.stat().st_size
    manifest["objects"][0]["sha256"] = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        register_sources.run("run-1")


def test_run_reports_unexpected_object_without_raising(tmp_path, monkeypatch):
    _seed_release(tmp_path, monkeypatch)
    extra_path = tmp_path / "bronze" / "source=extra" / "unexpected.zip"
    _write_zip(extra_path, {"note.txt": b"hi"})

    result = register_sources.run("run-1")

    assert result.output_count == 1
    assert result.issue_count == 1
