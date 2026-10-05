"""Tests for the Silver checks added for F6-F8: companion files stop the run,
the brief's minimum checks, and the recorded outcome of every check."""

import csv
import io
import zipfile

import pytest

from quantum_lake_student.formats import b8_record_bytes
from quantum_lake_student.stages import prepare_data

SYNDROME = [[0, 1, 0, 1] for _ in range(4)]


def _syndrome_archive(rows, member="d-3_pfr-0.001000_nb-10M.csv", readme=None):
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["labels", "syndromes", "quantity"])
    for label, syndrome, quantity in rows:
        writer.writerow([label, repr(syndrome), quantity])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, csv_buffer.getvalue())
        if readme is not None:
            archive.writestr("README.txt", readme)
    return buffer.getvalue()


def _google_archive(
    *,
    experiment="experiment",
    properties=None,
    shots=1,
    sweep_bits=0,
    sweep_bytes=None,
    omitted=(),
):
    properties = properties or {}
    yaml_text = "".join(f"{key}: {value}\n" for key, value in {
        "shots": shots,
        "circuit_measurements": 8,
        "circuit_detectors": 8,
        "circuit_sweep_bits": sweep_bits,
        **properties,
    }.items())
    members = {
        "properties.yml": yaml_text.encode(),
        "measurements.b8": bytes(shots),
        "detection_events.b8": bytes(shots),
        "obs_flips_actual.01": b"0\n" * shots,
    }
    if sweep_bits:
        members["sweep.b8"] = sweep_bytes or bytes(shots * b8_record_bytes(sweep_bits))
    for decoder in prepare_data.DECODER_NAMES:
        members[f"obs_flips_predicted_by_{decoder}.01"] = b"0\n" * shots
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            if name not in omitted:
                archive.writestr(f"{experiment}/{name}", content)
    return buffer.getvalue()


def _syndromes(archive_bytes):
    issues, checks = [], {}
    table, _, _ = prepare_data.prepare_syndrome_observations(
        archive_bytes, "syndromes.zip", "input-hash", "run-1", issues, checks
    )
    return table, issues, checks


def _shots(archive_bytes):
    issues, checks = [], {}
    table, _, rows_read = prepare_data.prepare_google_shots(
        archive_bytes, "google_qec.zip", "input-hash", "run-1", issues, checks
    )
    return table, rows_read, issues, checks


# --- F6: required companion files ------------------------------------------

def test_missing_sweep_file_stops_run_when_sweep_bits_are_declared():
    archive_bytes = _google_archive(sweep_bits=9, omitted=("sweep.b8",))
    issues = []

    with pytest.raises(prepare_data.MissingCompanionFileError, match="sweep.b8"):
        prepare_data.prepare_google_shots(
            archive_bytes, "google_qec.zip", "input-hash", "run-1", issues
        )

    assert [issue["rule_id"] for issue in issues] == ["RULE_GOOGLE_MISSING_COMPANION"]


def test_experiment_without_sweep_bits_needs_no_sweep_file():
    table, rows_read, issues, _ = _shots(_google_archive(sweep_bits=0))

    assert issues == []
    assert table.num_rows == rows_read == 1
    assert table.column("sweep_bits").to_pylist() == [b""]


# --- F7: minimum checks from the brief --------------------------------------

def test_weighted_total_matching_file_name_passes():
    table, issues, checks = _syndromes(
        _syndrome_archive([(0, SYNDROME, 2), (1, SYNDROME, 3)], member="d-3_pfr-0.001000_nb-5.csv")
    )

    assert issues == []
    assert table.num_rows == 2
    assert checks["RULE_SYN_WEIGHTED_TOTAL"]["checked"] == 1


def test_weighted_total_mismatch_is_logged_and_rows_are_kept():
    table, issues, _ = _syndromes(_syndrome_archive([(0, SYNDROME, 2), (1, SYNDROME, 3)]))

    assert [issue["rule_id"] for issue in issues] == ["RULE_SYN_WEIGHTED_TOTAL"]
    assert issues[0]["observed_value"] == "sum=5;declared=10000000"
    assert issues[0]["action"] == "logged"
    assert table.num_rows == 2


def test_documented_label_header_versus_actual_labels_is_recorded():
    readme = "- label: binary label\n- syndromes: sequence\n- quantity: samples\n"
    _, issues, checks = _syndromes(
        _syndrome_archive([(0, SYNDROME, 5)], member="d-3_pfr-0.001000_nb-5.csv", readme=readme)
    )

    assert issues == []
    observations = checks["RULE_SYN_HEADER_DOCUMENTED"]["observations"]
    assert observations == [
        "README documents columns ['label', 'syndromes', 'quantity']; "
        "CSV header is ['labels', 'syndromes', 'quantity']"
    ]


def test_experiment_name_agreeing_with_properties_is_accepted():
    properties = {
        "type": "surface_code_memory_experiment", "basis": "X", "distance": 3,
        "rounds": 25, "center_data_qubit_row": 3, "center_data_qubit_col": 5,
    }
    assert prepare_data._experiment_name_mismatches(
        "surface_code_bX_d3_r25_center_3_5", properties
    ) == []


def test_experiment_name_disagreeing_with_properties_is_excluded():
    archive_bytes = _google_archive(
        experiment="surface_code_bX_d3_r25_center_3_5",
        properties={
            "type": "surface_code_memory_experiment", "basis": "X", "distance": 5,
            "rounds": 25, "center_data_qubit_row": 3, "center_data_qubit_col": 5,
        },
    )
    issues = []

    table, _, rows_read = prepare_data.prepare_google_experiments(
        archive_bytes, "google_qec.zip", "input-hash", "run-1", issues
    )

    assert rows_read == 1
    assert table.num_rows == 0
    assert [issue["rule_id"] for issue in issues] == ["RULE_GOOGLE_NAME_PROPERTIES"]
    assert issues[0]["observed_value"] == "distance: name=3 properties=5"


def test_shots_of_an_excluded_experiment_are_not_published():
    issues = []
    table, _, rows_read = prepare_data.prepare_google_shots(
        _google_archive(), "google_qec.zip", "input-hash", "run-1", issues,
        excluded_experiments={"experiment"},
    )

    assert rows_read == 1
    assert table.num_rows == 0


def test_nonzero_sweep_padding_bits_are_rejected():
    # 9 sweep bits -> 2 bytes; bit 9 (0x02 in the second byte) is padding.
    table, _, issues, _ = _shots(_google_archive(sweep_bits=9, sweep_bytes=b"\x00\x02"))

    assert table.num_rows == 0
    assert [issue["rule_id"] for issue in issues] == ["RULE_GOOGLE_PADDING_BITS"]
    assert issues[0]["observed_value"] == "sweep.b8:last_byte=2"


# --- F8: outcome of every check ---------------------------------------------

def test_check_outcomes_cover_every_rule_with_counts():
    _, issues, checks = _syndromes(
        _syndrome_archive([(0, SYNDROME, 2), (1, SYNDROME, 3)])
    )

    outcomes = {o["rule_id"]: o for o in prepare_data._check_outcomes(checks, issues)}

    assert set(outcomes) == set(prepare_data.CHECK_CATALOG)
    shape = outcomes["RULE_SYN_SHAPE_DOMAIN"]
    assert (shape["checked"], shape["failed"], shape["passed"], shape["outcome"]) == (2, 0, 2, "passed")
    assert outcomes["RULE_SYN_WEIGHTED_TOTAL"]["outcome"] == "failed"
    assert outcomes["RULE_SYN_WEIGHTED_TOTAL"]["failed"] == 1
    assert outcomes["RULE_GOOGLE_PADDING_BITS"]["outcome"] == "not evaluated"
