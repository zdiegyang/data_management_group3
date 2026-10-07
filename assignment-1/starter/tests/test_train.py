"""Tests for Part II (make train).

Spec: "automated tests for input columns/types, split use, weights, packed-bit
order and event counts, repeatability, and metric calculation". These tests use
small synthetic ML tables, so they need neither PostgreSQL nor MinIO.
"""

import hashlib
import io
import json
import random
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    google_data_split,
    syndrome_data_split,
    weighted_logical_error_rate,
)
from quantum_lake_student.stages import part2_contracts as contracts
from quantum_lake_student.stages import part2_evaluation as evaluation
from quantum_lake_student.stages import part2_tasks as tasks

FAULT_RATES = [0.00001, 0.0005, 0.005, 0.01]  # train, validation, test, train


def _syndrome_rows(seed=1):
    rng = random.Random(seed)
    rows = []
    for rate in FAULT_RATES:
        for index in range(30):
            bits = [rng.randint(0, 1) for _ in range(16)]
            rows.append({
                "example_id": f"s-{rate}-{index}", "experiment_id": f"exp-{rate}",
                "physical_fault_rate": rate, "syndrome_bits": bytes(bits),
                "round_count": 4, "check_count": 4,
                "logical_error_label": bool(sum(bits) > 9 or rng.random() < 0.1),
                "sample_weight": rng.randint(1, 50), "data_split": syndrome_data_split(rate),
            })
    return rows


def _packed(bits):
    return bytes(sum(bit << position for position, bit in enumerate(bits[byte * 8:byte * 8 + 8]))
                 for byte in range((len(bits) + 7) // 8))


def _google_rows(seed=2, shots=120):
    rng = random.Random(seed)
    rows = []
    for experiment in ("exp-a", "exp-b"):
        for shot in range(shots):
            bits = [1 if rng.random() < 0.15 else 0 for _ in range(200)]
            actual = rng.random() < 0.5
            row = {
                "example_id": f"{experiment}-{shot}", "experiment_id": experiment, "shot_index": shot,
                "distance": 3, "rounds": 25, "center_row": 3, "center_col": 5,
                "detector_count": 200, "detector_event_count": sum(bits), "detector_bits": _packed(bits),
                "actual_observable_flip": actual, "data_split": google_data_split(shot),
            }
            for column in GOOGLE_META_PREDICTION_COLUMNS:
                row[column] = actual if rng.random() < 0.7 else not actual
            rows.append(row)
    return rows


def _without_timings(metrics):
    return {model: {k: v for k, v in values.items() if "time" not in k} for model, values in metrics.items()}


# --- input columns and types ------------------------------------------------

def _parquet(rows, schema, metadata=None):
    table = pa.Table.from_pylist(rows, schema=schema)
    if metadata is not None:
        table = table.replace_schema_metadata(metadata)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_matching_schema_is_read_and_hashed(monkeypatch):
    raw = _parquet(_syndrome_rows(), contracts.SYNDROME_SCHEMA)
    monkeypatch.setattr(contracts, "_object_bytes", lambda settings, key: raw)

    table, data, digest = contracts._read_ml_table(None, contracts.SYNDROME_KEY, contracts.SYNDROME_SCHEMA)

    assert table.num_rows == 120
    assert data == raw
    assert digest == hashlib.sha256(raw).hexdigest()


def test_wrong_column_type_is_rejected(monkeypatch):
    schema = contracts.SYNDROME_SCHEMA.set(
        contracts.SYNDROME_SCHEMA.get_field_index("sample_weight"), pa.field("sample_weight", pa.float64())
    )
    raw = _parquet(_syndrome_rows(), schema)
    monkeypatch.setattr(contracts, "_object_bytes", lambda settings, key: raw)

    with pytest.raises(contracts.MlContractError, match="schema mismatch"):
        contracts._read_ml_table(None, contracts.SYNDROME_KEY, contracts.SYNDROME_SCHEMA)


def test_data_release_is_read_from_both_tables():
    release = {"release_name": "quantum-data-core", "bundle_version": 3, "release_date": "2026-08-05"}
    meta = {contracts.RELEASE_METADATA_KEY: json.dumps(release).encode()}
    table = pa.table({"x": [1]}).replace_schema_metadata(meta)

    assert contracts._data_release(table, table) == release
    with pytest.raises(contracts.MlContractError, match="no data-release"):
        contracts._data_release(table, pa.table({"x": [1]}))
    other = pa.table({"x": [1]}).replace_schema_metadata(
        {contracts.RELEASE_METADATA_KEY: json.dumps({**release, "bundle_version": 4}).encode()}
    )
    with pytest.raises(contracts.MlContractError, match="different data releases"):
        contracts._data_release(table, other)


def test_invalid_rows_are_rejected_before_training():
    rows = _syndrome_rows()
    rows[0]["syndrome_bits"] = bytes(15)
    with pytest.raises(contracts.MlContractError, match="syndrome_bits"):
        contracts._validate_syndrome(rows)
    rows = [row for row in _syndrome_rows() if row["data_split"] != "validation"]
    with pytest.raises(contracts.MlContractError, match="validation"):
        contracts._validate_syndrome(rows)


# --- packed-bit order and event counts ---------------------------------------

def test_detector_bits_are_unpacked_little_endian_in_detector_order():
    bits = [0] * 200
    bits[0] = bits[9] = bits[199] = 1  # byte 0 bit 0, byte 1 bit 1, byte 24 bit 7
    row = {"example_id": "x", "detector_bits": _packed(bits), "detector_count": 200}

    assert row["detector_bits"][:2] == bytes([0b00000001, 0b00000010])
    features = tasks._detector_features(row)
    assert features.shape == (200,)
    assert np.flatnonzero(features).tolist() == [0, 9, 199]


def test_event_count_must_match_set_detector_bits():
    rows = _google_rows()
    rows[0]["detector_event_count"] += 1
    with pytest.raises(contracts.MlContractError, match="detector_event_count"):
        contracts._validate_google(rows)


def test_nonzero_padding_is_rejected(monkeypatch):
    # 200 and 600 detectors fill whole bytes; pretend 199 so bit 7 of the last byte is padding.
    monkeypatch.setitem(contracts.DETECTORS_PER_DISTANCE, 3, 199)
    rows = _google_rows()
    for row in rows:
        row["detector_count"] = 199
        row["detector_bits"] = row["detector_bits"][:-1] + bytes([row["detector_bits"][-1] & 0x7F])
        row["detector_event_count"] = sum(tasks.unpack_little_endian_bits(row["detector_bits"], 199))
    contracts._validate_google(rows)  # all padding zero: accepted
    rows[0]["detector_bits"] = rows[0]["detector_bits"][:-1] + bytes([0x80])
    with pytest.raises(contracts.MlContractError, match="padding"):
        contracts._validate_google(rows)


# --- split use -----------------------------------------------------------------

def test_task_a_fits_on_train_and_keeps_the_supplied_split():
    rows = _syndrome_rows()
    predictions, metrics, _ = tasks._run_task_a(rows)

    train = [row for row in rows if row["data_split"] == "train"]
    expected_prior = sum(r["sample_weight"] for r in train if r["logical_error_label"]) / sum(
        r["sample_weight"] for r in train)
    assert metrics["task_a_prior"]["positive_prior"] == pytest.approx(expected_prior)

    split_by_id = {row["example_id"]: row["data_split"] for row in rows}
    assert all(p["split"] == split_by_id[p["example_id"]] for p in predictions)
    assert {p["model_id"] for p in predictions} == {"task_a_prior", "task_a_logistic"}
    assert len(predictions) == 2 * len(rows)


def test_task_c_uses_only_the_fixed_subset():
    rows = _google_rows(shots=120)
    rows += [dict(row, example_id=row["example_id"] + "-late", shot_index=row["shot_index"] + 20_000)
             for row in rows[:20]]
    rows += [dict(row, example_id=row["example_id"] + "-d5", distance=5) for row in rows[:20]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions, _, _, diagnostics = tasks._run_task_c(rows)

    assert diagnostics["subset_rows"] == 240
    assert not any(p["example_id"].endswith(("-late", "-d5")) for p in predictions)


def test_threshold_is_chosen_from_the_validation_data_given():
    labels = np.array([False, False, True, True])
    probabilities = np.array([0.1, 0.2, 0.7, 0.9])
    assert evaluation._select_threshold(labels, probabilities) == pytest.approx(0.5)


# --- weights ---------------------------------------------------------------------

def test_task_a_metrics_use_physical_weights():
    labels = np.array([True, False, False])
    predictions = np.array([False, False, False])
    weights = np.array([1.0, 9.0, 90.0])

    metrics = evaluation._metrics(labels, predictions, None, weights)

    assert metrics["logical_error_rate"] == pytest.approx(0.01)
    assert metrics["logical_error_rate"] == pytest.approx(
        weighted_logical_error_rate(labels, predictions, weights))
    assert evaluation._metrics(labels, predictions, None)["logical_error_rate"] == pytest.approx(1 / 3)
    assert evaluation._positive_prior(labels, weights) == pytest.approx(0.01)


# --- metric calculation -------------------------------------------------------------

def test_metrics_match_hand_computed_values():
    labels = np.array([True, True, False, False])
    predictions = np.array([True, False, False, True])
    probabilities = np.array([0.9, 0.4, 0.2, 0.6])

    metrics = evaluation._metrics(labels, predictions, probabilities)

    assert metrics["logical_error_rate"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)  # (1/2 + 1/2) / 2
    assert metrics["brier_score"] == pytest.approx((0.01 + 0.36 + 0.04 + 0.36) / 4)
    assert evaluation._metrics(labels, predictions, None)["brier_score"] is None


# --- repeatability ----------------------------------------------------------------------

def test_tasks_are_repeatable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = (tasks._run_task_a(_syndrome_rows()), tasks._run_task_b_distance(_google_rows(), 3),
                 tasks._run_task_c(_google_rows()))
        second = (tasks._run_task_a(_syndrome_rows()), tasks._run_task_b_distance(_google_rows(), 3),
                  tasks._run_task_c(_google_rows()))

    for one, two in zip(first, second):
        assert one[0] == two[0]  # identical prediction records
        assert _without_timings(one[1]) == _without_timings(two[1])
