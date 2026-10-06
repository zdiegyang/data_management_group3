"""Tests for Stage 4: Gold -> the two required ML input tables (R1-R3)."""

import io
import zipfile
from math import ceil

import psycopg
import pyarrow.parquet as pq
import pytest
import yaml

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.stages import build_ml_tables as ml
from quantum_lake_student.stages import register_sources

GOOGLE_BRONZE_KEY = "bronze/source=google_qec/google-surface-code-curated.zip"

SYNDROME = bytes([0, 1, 0, 1] * 4)


def _syndrome_rows():
    rows = []
    for index, split in enumerate(["train", "validation", "test"]):
        rows.append({
            "example_id": f"s{index}", "experiment_id": f"exp{index}", "physical_fault_rate": 0.001,
            "syndrome_bits": SYNDROME, "round_count": 4, "check_count": 4,
            "logical_error_label": bool(index % 2), "sample_weight": 10, "data_split": split,
        })
    return rows


def _google_rows():
    detector_bits = bytes([0b00000101]) + bytes(24)  # 200 detectors, bits 0 and 2 set
    rows = []
    for index, split in enumerate(["train", "validation", "test"]):
        rows.append({
            "example_id": f"g{index}", "experiment_id": "exp", "shot_index": index,
            "distance": 3, "rounds": 25, "center_row": 3, "center_col": 5,
            "detector_count": 200, "detector_event_count": 2, "detector_bits": detector_bits,
            "belief_matching_prediction": True, "correlated_matching_prediction": False,
            "pymatching_prediction": True, "tensor_network_contraction_prediction": False,
            "actual_observable_flip": True, "data_split": split,
        })
    return rows


# --- contract checks --------------------------------------------------------

def test_valid_examples_pass_the_contract():
    ml.check_syndrome_examples(_syndrome_rows())
    ml.check_google_examples(_google_rows())


def test_duplicate_example_ids_are_rejected():
    rows = _syndrome_rows()
    rows[1]["example_id"] = rows[0]["example_id"]
    with pytest.raises(ml.MlContractError, match="not unique"):
        ml.check_syndrome_examples(rows)


def test_an_empty_split_is_rejected():
    rows = [row for row in _google_rows() if row["data_split"] != "validation"]
    with pytest.raises(ml.MlContractError, match="validation"):
        ml.check_google_examples(rows)


def test_syndrome_must_have_sixteen_binary_values():
    rows = _syndrome_rows()
    rows[0]["syndrome_bits"] = bytes([0, 1, 2] + [0] * 13)
    with pytest.raises(ml.MlContractError, match="16 binary values"):
        ml.check_syndrome_examples(rows)


def test_nonzero_detector_padding_is_rejected(monkeypatch):
    # The release's 200 and 600 detectors fill whole bytes, so pretend a
    # distance-3 record has 199 detectors: bit 7 of the last byte is padding.
    monkeypatch.setitem(ml.DETECTORS_PER_DISTANCE, 3, 199)
    rows = _google_rows()
    for row in rows:
        row["detector_count"] = 199
    rows[0]["detector_bits"] = rows[0]["detector_bits"][:-1] + bytes([0b10000000])
    with pytest.raises(ml.MlContractError, match="padding"):
        ml.check_google_examples(rows)


def test_event_count_must_equal_set_detector_bits():
    rows = _google_rows()
    rows[0]["detector_event_count"] = 3
    with pytest.raises(ml.MlContractError, match="detector_event_count"):
        ml.check_google_examples(rows)


def test_detector_count_must_match_distance():
    rows = _google_rows()
    rows[0]["distance"] = 5
    with pytest.raises(ml.MlContractError, match="distance 5 must have 600"):
        ml.check_google_examples(rows)


def test_missing_decoder_prediction_is_rejected():
    rows = _google_rows()
    rows[2]["pymatching_prediction"] = None
    with pytest.raises(ml.MlContractError, match="not aligned"):
        ml.check_google_examples(rows)


def test_tables_have_exactly_the_contract_columns_and_types():
    extra = [dict(row, source_record_id="lineage only") for row in _syndrome_rows()]
    syndrome = ml.to_table(extra, ml.SYNDROME_SCHEMA)
    google = ml.to_table(_google_rows(), ml.GOOGLE_SCHEMA)

    assert syndrome.schema == ml.SYNDROME_SCHEMA
    assert "source_record_id" not in syndrome.column_names
    assert google.schema == ml.GOOGLE_SCHEMA


# --- end to end against Gold -------------------------------------------------

def _gold_is_loaded(settings):
    try:
        with psycopg.connect(settings.postgres_dsn) as conn:
            return conn.execute("SELECT to_regclass('gold.google_shot') IS NOT NULL").fetchone()[0]
    except psycopg.Error:
        return False


def _read_lake(settings, key):
    if settings.lake_backend == "local":
        return pq.read_table(settings.local_lake_root / key)
    response = minio_client(settings).get_object(settings.s3_bucket, key)
    try:
        return pq.read_table(io.BytesIO(response.read()))
    finally:
        response.close()
        response.release_conn()


def test_stage_exports_both_tables_from_gold_and_is_repeatable():
    settings = Settings.from_environment()
    if not _gold_is_loaded(settings):
        pytest.skip("Gold is not loaded; run `make run` first")

    first = ml.run("ml-test-1", settings)
    second = ml.run("ml-test-2", settings)

    syndrome = _read_lake(settings, ml.SYNDROME_KEY)
    google = _read_lake(settings, ml.GOOGLE_KEY)
    assert syndrome.schema == ml.SYNDROME_SCHEMA
    assert google.schema == ml.GOOGLE_SCHEMA
    assert syndrome.num_rows == 75598
    assert sum(syndrome.column("sample_weight").to_pylist()) == 70_000_000
    assert google.num_rows == 250000
    # Repeatable: same bytes, same identifiers on a second run.
    assert first.details["ml_tables"] == second.details["ml_tables"]
    assert first.details["ml_tables"][ml.GOOGLE_KEY]["split_counts"] == {
        "train": 100000, "validation": 25000, "test": 125000,
    }


def test_ml_detector_bits_reproduce_the_bronze_records_exactly():
    """Brief: "prove that the ML export can reproduce the packed bytes without
    returning to Bronze or Silver". The export reads only Gold; this test (not
    the export) opens Bronze to show every exported record is byte-identical
    to the original Stim b8 detection_events record of the same shot."""
    settings = Settings.from_environment()
    if not _gold_is_loaded(settings):
        pytest.skip("Gold is not loaded; run `make run` first")
    try:
        exported = _read_lake(settings, ml.GOOGLE_KEY)
    except Exception:
        ml.run("ml-bytes-test", settings)
        exported = _read_lake(settings, ml.GOOGLE_KEY)

    archive = zipfile.ZipFile(io.BytesIO(register_sources._object_bytes(settings, GOOGLE_BRONZE_KEY)))
    rows = exported.select(["experiment_id", "shot_index", "detector_bits"]).to_pylist()
    bronze = {}
    for experiment in {row["experiment_id"] for row in rows}:
        detectors = yaml.safe_load(archive.read(f"{experiment}/properties.yml"))["circuit_detectors"]
        bronze[experiment] = (archive.read(f"{experiment}/detection_events.b8"), ceil(detectors / 8))

    mismatched = [
        (row["experiment_id"], row["shot_index"])
        for row in rows
        for records, stride in [bronze[row["experiment_id"]]]
        if row["detector_bits"] != records[row["shot_index"] * stride:(row["shot_index"] + 1) * stride]
    ]

    assert len(rows) == 250000
    assert mismatched == []
