"""Tests for F3/F4: one run.json per run, written by the CLI, with the code revision."""

import json
from datetime import UTC, datetime

from quantum_lake_student import cli
from quantum_lake_student.models import StageResult


def _finished(stage, output_count=0, issue_count=0, details=None):
    result = StageResult(stage=stage, run_id="run-1", output_count=output_count, issue_count=issue_count)
    result.details.update(details or {})
    result.finish()
    return result


def _bronze():
    return _finished("register_sources", output_count=3,
                     details={"input_hashes": {"bronze/source=x/x.zip": "deadbeef"}})


def _silver():
    return _finished("prepare_data", output_count=7, issue_count=2, details={
        "output_table_counts": {"silver/x/table.parquet": 7},
        "checks": [{"rule_id": "RULE_X", "outcome": "passed"}],
    })


def test_run_record_combines_all_stages(tmp_path):
    (tmp_path / "row_counts.json").write_text(json.dumps({"silver_to_gold": {
        "google_shot": {
            "expected_silver_count": 7,
            "loaded_count": 7,
            "matches": True,
        },
        "syndrome_pattern": {"loaded_count": 11},
        "decoder": {"loaded_count": 4},
        "benchmark": {"loaded_count": 3},
        "check_data_qubit": {"loaded_count": 24},
    }}))
    stages = [_bronze(), _silver(), _finished("load_postgres", output_count=7)]

    cli.write_run_record("run-1", datetime.now(UTC), stages, results_dir=tmp_path)
    record = json.loads((tmp_path / "run.json").read_text())

    assert record["status"] == "succeeded"
    assert record["input_hashes"] == {"bronze/source=x/x.zip": "deadbeef"}
    assert [stage["stage"] for stage in record["stages"]] == [
        "register_sources", "prepare_data", "load_postgres",
    ]
    assert record["output_table_counts"] == {
        "silver/x/table.parquet": 7,
        "gold.google_shot": 7,
        "gold.syndrome_pattern": 11,
        "gold.decoder": 4,
        "gold.benchmark": 3,
        "gold.check_data_qubit": 24,
    }
    assert record["issue_count"] == 2
    assert record["checks"] == [{"rule_id": "RULE_X", "outcome": "passed"}]
    assert len(record["code_sha256"]) == 64


def test_failed_run_is_recorded_without_later_stage_outputs(tmp_path):
    (tmp_path / "row_counts.json").write_text(json.dumps({"silver_to_gold": {
        "google_shot": {
            "expected_silver_count": 7,
            "loaded_count": 7,
            "matches": True,
        },
    }}))

    cli.write_run_record(
        "run-1", datetime.now(UTC), [_bronze()],
        error=RuntimeError("checksum mismatch"), results_dir=tmp_path,
    )
    record = json.loads((tmp_path / "run.json").read_text())

    assert record["status"] == "failed"
    assert record["error"]["type"] == "RuntimeError"
    assert record["error"]["message"] == "checksum mismatch"
    assert record["output_table_counts"] == {}
    assert record["checks"] == []


def test_git_revision_is_read_from_git_files_without_git(tmp_path, monkeypatch):
    git_dir = tmp_path / "repo-git"
    (git_dir / "refs/heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs/heads/main").write_text("abc123\n")
    monkeypatch.setenv("QUANTUM_GIT_DIR", str(git_dir))
    monkeypatch.setattr(cli.subprocess, "run", _no_git)

    assert cli.git_revision(start=tmp_path / "elsewhere") == "abc123"


def test_git_revision_falls_back_to_packed_refs(tmp_path, monkeypatch):
    git_dir = tmp_path / "repo-git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text("# pack-refs\ndef456 refs/heads/main\n")
    monkeypatch.setenv("QUANTUM_GIT_DIR", str(git_dir))
    monkeypatch.setattr(cli.subprocess, "run", _no_git)

    assert cli.git_revision(start=tmp_path / "elsewhere") == "def456"


def test_code_sha256_changes_when_code_changes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    first = cli.code_sha256(tmp_path)

    assert cli.code_sha256(tmp_path) == first
    (tmp_path / "src/a.py").write_text("x = 2\n")
    assert cli.code_sha256(tmp_path) != first


def _no_git(*args, **kwargs):
    raise FileNotFoundError("git")
