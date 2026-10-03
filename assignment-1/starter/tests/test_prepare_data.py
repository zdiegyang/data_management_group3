"""Tests for Stage 2: prepare_data and source lineage tracing."""

from pathlib import Path
import csv
import io
import json
import pyarrow.parquet as pq
import pytest
import zipfile

from quantum_lake_student.formats import b8_record_bytes, iter_b8_records
from quantum_lake_student.stages import prepare_data
from quantum_lake_student.tracing import SOURCE_TRACE_SCHEMA, save_source_traces


def test_failed_prepare_run_records_failure_and_reraises(tmp_path, monkeypatch):
    original_record_failed_run = prepare_data._record_failed_run

    def fail_execution(run_id, settings, run_state):
        raise ValueError("synthetic stage failure")

    def record_to_tmp_path(run_id, result, run_state, error):
        run_state["base_dir"] = tmp_path
        original_record_failed_run(run_id, result, run_state, error)

    monkeypatch.setattr(prepare_data, "_execute_run", fail_execution)
    monkeypatch.setattr(prepare_data, "_record_failed_run", record_to_tmp_path)

    with pytest.raises(ValueError, match="synthetic stage failure"):
        prepare_data.run("failed-run")

    results_dir = tmp_path / "results/part1"
    row_counts = json.loads((results_dir / "row_counts.json").read_text(encoding="utf-8"))
    issues = pq.read_table(results_dir / "data_issues.parquet").to_pylist()

    # run.json is written by the CLI for the whole run, not by this stage.
    assert not (results_dir / "run.json").exists()
    assert row_counts["tables"]["results/part1/data_issues.parquet"]["rows_loaded"] == 1
    assert issues[0]["rule_id"] == "RUN_PREPARE_DATA_FAILED"
    assert issues[0]["observed_value"] == "ValueError: synthetic stage failure"


def _syndrome_archive(rows):
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["labels", "syndromes", "quantity"])
    for label, syndrome, quantity in rows:
        writer.writerow([label, repr(syndrome), quantity])

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            "d-3_pfr-0.001000_nb-10M.csv",
            csv_buffer.getvalue(),
        )
    return archive_buffer.getvalue()


def _google_shot_archive(
    *,
    shots=1,
    measurement_bits=8,
    detector_bits=8,
    measurement_bytes=None,
    detector_bytes=None,
    actual=b"0\n",
    predictions=None,
    omitted=(),
):
    experiment = "experiment"
    members = {
        f"{experiment}/properties.yml": (
            f"shots: {shots}\n"
            f"circuit_measurements: {measurement_bits}\n"
            f"circuit_detectors: {detector_bits}\n"
            "circuit_sweep_bits: 0\n"
        ).encode(),
        f"{experiment}/measurements.b8": measurement_bytes or bytes(
            shots * b8_record_bytes(measurement_bits)
        ),
        f"{experiment}/detection_events.b8": detector_bytes or bytes(
            shots * b8_record_bytes(detector_bits)
        ),
        f"{experiment}/obs_flips_actual.01": actual,
    }
    prediction_rows = predictions or {decoder: b"0\n" * shots for decoder in prepare_data.DECODER_NAMES}
    for decoder, values in prediction_rows.items():
        members[f"{experiment}/obs_flips_predicted_by_{decoder}.01"] = values

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for name, content in members.items():
            if name.rsplit("/", 1)[-1] not in omitted:
                archive.writestr(name, content)
    return archive_buffer.getvalue()


def _parse_google_shots(archive_bytes):
    issues = []
    table, _, rows_read = prepare_data.prepare_google_shots(
        archive_bytes,
        "google_qec.zip",
        "input-hash",
        "test-run",
        issues,
    )
    return table, rows_read, issues


def test_save_source_traces_overwrites_existing_file(tmp_path):
    trace_file = tmp_path / "source_trace.parquet"
    first_run = [{
        "source_record_id": "first",
        "source_name": "test",
        "bronze_object": "first.zip",
        "archive_member": "first.csv",
        "record_locator": "row=1",
        "input_sha256": "hash1",
    }]
    second_run = [{
        "source_record_id": "second",
        "source_name": "test",
        "bronze_object": "second.zip",
        "archive_member": "second.csv",
        "record_locator": "row=2",
        "input_sha256": "hash2",
    }]

    save_source_traces(first_run, "test", trace_file)
    save_source_traces(second_run, "test", trace_file)

    saved = pq.read_table(trace_file)
    assert saved.num_rows == 1
    assert saved.column("source_record_id").to_pylist() == ["second"]


def test_prepare_qasmbench_records_stable_parse_issues(tmp_path):
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(
            "benchmark/invalid.qasm",
            "OPENQASM 2.0; qreg q[1]; cx missing[0], q[0];",
        )

    issues_first = []
    prepare_data.prepare_qasmbench(
        archive_bytes.getvalue(), "qasmbench.zip", "hash", "run-1", tmp_path, issues_first
    )
    issues_second = []
    prepare_data.prepare_qasmbench(
        archive_bytes.getvalue(), "qasmbench.zip", "hash", "run-2", tmp_path, issues_second
    )

    assert {issue["rule_id"] for issue in issues_first} == {
        "RULE_QASM_CIRCUIT_PARSE",
        "RULE_QASM_STABILIZER_PARSE",
    }
    assert [issue["issue_id"] for issue in issues_first] == [
        issue["issue_id"] for issue in issues_second
    ]
    assert {issue["run_id"] for issue in issues_first} == {"run-1"}



def test_syndrome_rows_with_same_bits_and_different_labels_are_kept():
    syndrome = [[0, 1, 0, 1] for _ in range(4)]
    archive_bytes = _syndrome_archive([
        (0, syndrome, 2),
        (1, syndrome, 3),
    ])

    table, _, rows_read = prepare_data.prepare_syndrome_observations(
        archive_bytes,
        "syndromes.zip",
        "input-hash",
        "run-1",
        [],
    )

    assert rows_read == 2
    assert table.num_rows == 2
    assert table.column("logical_error_label").to_pylist() == [False, True]
    assert len(set(table.column("source_record_id").to_pylist())) == 2



def test_google_shot_b8_records_preserve_little_endian_order_and_padding():
    archive_bytes = _google_shot_archive(
        shots=2,
        measurement_bits=9,
        detector_bits=9,
        measurement_bytes=b"\x05\x01\x02\x00",
        detector_bytes=b"\x05\x01\x02\x00",
        actual=b"0\n1\n",
        predictions={
            decoder: b"1\n0\n"
            for decoder in prepare_data.DECODER_NAMES
        },
    )

    table, rows_read, issues = _parse_google_shots(archive_bytes)

    assert issues == []
    assert rows_read == 2
    assert table.num_rows == 2
    detector_records = table.column("detector_bits").to_pylist()
    assert detector_records == [b"\x05\x01", b"\x02\x00"]
    assert list(iter_b8_records(detector_records[0], bits_per_record=9))[0] == (
        1, 0, 1, 0, 0, 0, 0, 0, 1
    )
    assert table.column("actual_observable_flip").to_pylist() == [False, True]
    assert table.column("pymatching_prediction").to_pylist() == [True, False]
    assert table.column("shot_index").to_pylist() == [0, 1]


def test_google_shot_rejects_malformed_b8_length():
    archive_bytes = _google_shot_archive(
        shots=2,
        detector_bytes=b"\x00",
        actual=b"0\n1\n",
    )

    table, rows_read, issues = _parse_google_shots(archive_bytes)

    assert rows_read == 2
    assert table.num_rows == 0
    assert [issue["rule_id"] for issue in issues] == [
        "RULE_GOOGLE_COMPANION_STRIDE_MISMATCH"
    ]


def test_google_shot_rejects_nonzero_b8_padding_bits():
    archive_bytes = _google_shot_archive(
        detector_bits=9,
        detector_bytes=b"\x00\x02",
    )

    table, rows_read, issues = _parse_google_shots(archive_bytes)

    assert rows_read == 1
    assert table.num_rows == 0
    assert [issue["rule_id"] for issue in issues] == ["RULE_GOOGLE_PADDING_BITS"]


def test_google_shot_rejects_misaligned_01_records():
    archive_bytes = _google_shot_archive(
        shots=2,
        actual=b"0\n",
    )

    table, rows_read, issues = _parse_google_shots(archive_bytes)

    assert rows_read == 2
    assert table.num_rows == 0
    assert [issue["rule_id"] for issue in issues] == [
        "RULE_GOOGLE_COMPANION_STRIDE_MISMATCH"
    ]


def test_google_shot_missing_companion_file_stops_run():
    archive_bytes = _google_shot_archive(
        omitted=("obs_flips_predicted_by_pymatching.01",),
    )
    issues = []

    with pytest.raises(prepare_data.MissingCompanionFileError):
        prepare_data.prepare_google_shots(
            archive_bytes, "google_qec.zip", "input-hash", "test-run", issues
        )

    assert [issue["rule_id"] for issue in issues] == [
        "RULE_GOOGLE_MISSING_COMPANION"
    ]
    assert issues[0]["action"] == "stopped run"


def test_prepare_data_stage_execution():
    """Verify that Stage 2 runs end-to-end, producing verified Silver tables and traces."""
    res = prepare_data.run("test_verification_run")

    assert res.stage == "prepare_data"
    assert res.finished_at is not None
    assert res.finished_at >= res.started_at
    assert res.input_count >= 325603
    assert res.output_count >= 325603
    assert res.issue_count == 0

    base_dir = Path(__file__).resolve().parents[1]
    silver_dir = base_dir / "silver"
    results_dir = base_dir / "results/part1"

    row_counts_file = results_dir / "row_counts.json"
    assert row_counts_file.exists(), "row_counts.json missing"
    row_count_data = json.loads(row_counts_file.read_text(encoding="utf-8"))
    output_table_counts = res.details["output_table_counts"]
    assert output_table_counts
    for table_path, counts in row_count_data["tables"].items():
        assert counts["rows_read"] == counts["rows_accepted"] + counts["rows_rejected"]
        assert counts["rows_loaded"] == output_table_counts[table_path]
    qasm_row_units = {
        "circuit": "circuit row",
        "stabilizer_check": "stabilizer check row",
        "conditional_correction": "conditional correction row",
    }
    for table_name, unit in qasm_row_units.items():
        table_path = f"silver/qasmbench/{table_name}.parquet"
        counts = row_count_data["tables"][table_path]
        assert counts["unit"] == unit
        assert counts["rows_read"] == counts["rows_loaded"]
        assert counts["source_files_read"] >= counts["source_files_rejected"]
        assert counts["rows_loaded"] == pq.read_table(silver_dir / "qasmbench" / f"{table_name}.parquet").num_rows

    # Outcome of every check: all rules evaluated, none failed on this release.
    outcomes = {check["rule_id"]: check for check in res.details["checks"]}
    assert set(outcomes) == set(prepare_data.CHECK_CATALOG)
    assert all(check["failed"] == 0 for check in outcomes.values())
    assert outcomes["RULE_SYN_WEIGHTED_TOTAL"]["checked"] == 7
    assert outcomes["RULE_GOOGLE_PADDING_BITS"]["checked"] == 250000
    assert outcomes["RULE_SYN_HEADER_DOCUMENTED"]["outcome"] == "observed"

    # 1. Syndrome observations contract & invariants
    syn_file = silver_dir / "qec_syndromes/syndrome_observation.parquet"
    assert syn_file.exists(), "Syndrome Silver table missing"
    t_syn = pq.read_table(syn_file)
    df_syn = t_syn.to_pandas()
    assert len(df_syn) == 75598
    assert df_syn["quantity"].sum() == 70_000_000
    assert df_syn["source_record_id"].is_unique
    assert (df_syn["round_count"] == 4).all()
    assert (df_syn["check_count"] == 4).all()
    assert (df_syn["syndrome_bits"].str.len() == 16).all()
    assert df_syn.isna().sum().sum() == 0

    # 2. Google experiments contract & invariants
    exp_file = silver_dir / "google_qec/experiment.parquet"
    assert exp_file.exists(), "Google experiment Silver table missing"
    t_exp = pq.read_table(exp_file)
    df_exp = t_exp.to_pandas()
    assert len(df_exp) == 5
    assert df_exp["experiment_id"].is_unique
    assert df_exp["source_record_id"].is_unique
    assert (df_exp["basis"] == "X").all()
    assert (df_exp["rounds"] == 25).all()
    assert (df_exp["shots"] == 50000).all()
    assert (df_exp["distance"].isin([3, 5])).all()

    # 3. Google shots contract & invariants
    shot_file = silver_dir / "google_qec/shot.parquet"
    assert shot_file.exists(), "Google shot Silver table missing"
    t_shot = pq.read_table(shot_file)
    df_shot = t_shot.to_pandas()
    assert len(df_shot) == 250000
    assert df_shot["source_record_id"].is_unique
    assert df_shot.isna().sum().sum() == 0
    assert (df_shot["detector_event_count"] >= 0).all()

    # 4. QASMBench silver-table contract
    qasm_dir = silver_dir / "qasmbench"
    qasm_specs = {
        "circuit": [
            "source_record_id",
            "circuit_id",
            "benchmark_name",
            "variant",
            "register_declarations",
            "qubit_count",
            "measurement_count",
            "two_qubit_gate_count",
        ],
        "stabilizer_check": [
            "source_record_id",
            "circuit_id",
            "check_id",
            "ancilla_qubit",
            "data_qubits",
            "syndrome_bit",
        ],
        "conditional_correction": [
            "source_record_id",
            "circuit_id",
            "condition_register",
            "condition_value",
            "gate",
            "target_qubit",
        ],
    }
    for table_name, expected_columns in qasm_specs.items():
        table_path = qasm_dir / f"{table_name}.parquet"
        assert table_path.exists(), f"QASMBench {table_name} Silver table missing"
        t_qasm = pq.read_table(table_path)
        assert set(t_qasm.column_names) == set(expected_columns)
        assert t_qasm.num_rows > 0

    t_qasm_circuit = pq.read_table(qasm_dir / "circuit.parquet")
    df_qasm_circuit = t_qasm_circuit.to_pandas()
    assert df_qasm_circuit["source_record_id"].is_unique
    assert (df_qasm_circuit["qubit_count"] > 0).all()
    assert (df_qasm_circuit["measurement_count"] > 0).all()
    assert (df_qasm_circuit["two_qubit_gate_count"] >= 0).all()

    t_qasm_checks = pq.read_table(qasm_dir / "stabilizer_check.parquet")
    df_qasm_checks = t_qasm_checks.to_pandas()
    assert df_qasm_checks["source_record_id"].is_unique
    assert (df_qasm_checks["syndrome_bit"].str.len() > 0).all()

    t_qasm_corr = pq.read_table(qasm_dir / "conditional_correction.parquet")
    df_qasm_corr = t_qasm_corr.to_pandas()
    assert df_qasm_corr["source_record_id"].is_unique
    assert (df_qasm_corr["condition_value"].astype(int) >= 0).all()
    assert set(df_qasm_corr["gate"]).issubset({"cx", "cz", "swap", "x", "y", "z"})

    # 5. Source trace contract, valid column, and companion-file coverage
    trace_file = results_dir / "source_trace.parquet"
    assert trace_file.exists(), "source_trace.parquet missing"
    t_trace = pq.read_table(trace_file)
    assert t_trace.schema == SOURCE_TRACE_SCHEMA
    df_trace = t_trace.to_pandas()
    assert set(df_trace["valid"].unique()) == {"yes"}
    assert df_trace["source_record_id"].nunique() in (325603, 325623)

    # Composite primary key uniqueness in trace table
    assert not df_trace.duplicated(
        subset=["source_record_id", "archive_member", "record_locator"]
    ).any()

    # 5. Data issues schema validation
    issues_file = results_dir / "data_issues.parquet"
    assert issues_file.exists(), "data_issues.parquet missing"
    t_issues = pq.read_table(issues_file)
    assert t_issues.num_rows == 0
    assert "rule_id" in t_issues.column_names
    assert "severity" in t_issues.column_names
