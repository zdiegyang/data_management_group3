"""Part II input contracts and ML-table loading.

This module owns the boundary between Part I's materialized ML Parquet tables
and the Part II models. It validates the fixed schemas, supplied split labels,
and packed detector representation before any model is fitted.
"""
from __future__ import annotations

import hashlib
import io
import json
from math import ceil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    MODEL_SPLITS,
    google_meta_model_input,
    syndrome_model_input,
    unpack_little_endian_bits,
)

SYNDROME_KEY = "ml/ml_syndrome_decoder_example.parquet"
GOOGLE_KEY = "ml/ml_google_decoder_example.parquet"

SEED = 20261006
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
REQUIRED_SPLITS = tuple(MODEL_SPLITS)
DETECTORS_PER_DISTANCE = {3: 200, 5: 600}
MLP_SUBSET_DISTANCE = 3
MLP_SHOT_LIMIT = 12_500

SYNDROME_SCHEMA = pa.schema([
    ("example_id", pa.string()),
    ("experiment_id", pa.string()),
    ("physical_fault_rate", pa.float64()),
    ("syndrome_bits", pa.binary()),
    ("round_count", pa.int32()),
    ("check_count", pa.int32()),
    ("logical_error_label", pa.bool_()),
    ("sample_weight", pa.int64()),
    ("data_split", pa.string()),
])

GOOGLE_SCHEMA = pa.schema([
    ("example_id", pa.string()),
    ("experiment_id", pa.string()),
    ("shot_index", pa.int64()),
    ("distance", pa.int32()),
    ("rounds", pa.int32()),
    ("center_row", pa.int32()),
    ("center_col", pa.int32()),
    ("detector_count", pa.int32()),
    ("detector_event_count", pa.int32()),
    ("detector_bits", pa.binary()),
    *[(name, pa.bool_()) for name in GOOGLE_META_PREDICTION_COLUMNS],
    ("actual_observable_flip", pa.bool_()),
    ("data_split", pa.string()),
])

class MlContractError(RuntimeError):
    """Raised when an input ML table violates the fixed Part II contract."""

def _object_bytes(settings: Settings, key: str) -> bytes:
    """Read exactly one ML object from the configured lake backend."""
    if settings.lake_backend == "local":
        path = settings.local_lake_root / key
        if not path.is_file():
            raise FileNotFoundError(f"Required ML input table not found: {path}")
        return path.read_bytes()

    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _read_ml_table(
    settings: Settings, key: str, expected_schema: pa.Schema
) -> tuple[pa.Table, bytes, str]:
    """Read, hash, and contract-check one ML Parquet object."""
    raw = _object_bytes(settings, key)
    table = pq.read_table(io.BytesIO(raw))

    actual = [(field.name, field.type) for field in table.schema]
    expected = [(field.name, field.type) for field in expected_schema]
    if actual != expected:
        raise MlContractError(
            f"{key}: schema mismatch. Expected {expected}, found {actual}"
        )

    return table, raw, hashlib.sha256(raw).hexdigest()


# Written by Part I's build_ml_tables into both ML tables' Parquet metadata.
RELEASE_METADATA_KEY = b"quantum_lake.data_release"


def _data_release(*tables: pa.Table) -> dict[str, Any]:
    """The course data release recorded inside the ML tables.

    Spec: training records "the course data-release identifier" while reading
    the two ML tables "and nothing else", so it comes from the tables
    themselves. Both tables must name the same release.
    """
    releases = []
    for table in tables:
        raw = (table.schema.metadata or {}).get(RELEASE_METADATA_KEY)
        if raw is None:
            raise MlContractError("ML table carries no data-release metadata; rerun `make run`")
        releases.append(json.loads(raw))
    if any(release != releases[0] for release in releases):
        raise MlContractError(f"ML tables come from different data releases: {releases}")
    return releases[0]


def _validate_ids_and_splits(rows: Sequence[dict[str, Any]], table_name: str) -> None:
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise MlContractError(f"{table_name}: example_id values are not unique")

    present = {row["data_split"] for row in rows}
    unexpected = present - set(REQUIRED_SPLITS)
    missing = set(REQUIRED_SPLITS) - present
    if unexpected:
        raise MlContractError(
            f"{table_name}: unexpected data_split values: {sorted(unexpected)}"
        )
    if missing:
        raise MlContractError(
            f"{table_name}: missing required data_split values: {sorted(missing)}"
        )


def _validate_syndrome(rows: list[dict[str, Any]]) -> None:
    _validate_ids_and_splits(rows, "ml_syndrome_decoder_example")
    for row in rows:
        if (row["round_count"], row["check_count"]) != (4, 4):
            raise MlContractError(
                f"syndrome example {row['example_id']}: round_count/check_count must be 4/4"
            )
        if row["sample_weight"] <= 0:
            raise MlContractError(
                f"syndrome example {row['example_id']}: sample_weight must be positive"
            )
        if row["logical_error_label"] is None:
            raise MlContractError(
                f"syndrome example {row['example_id']}: label is null"
            )
        try:
            features = syndrome_model_input(row["syndrome_bits"])
        except ValueError as exc:
            raise MlContractError(
                f"syndrome example {row['example_id']}: invalid syndrome_bits: {exc}"
            ) from exc
        values = np.asarray(features)
        if values.shape != (16,):
            raise MlContractError(
                f"syndrome example {row['example_id']}: helper returned shape {values.shape}, expected (16,)"
            )
        if not np.all(np.isin(values, [0, 1])):
            raise MlContractError(
                f"syndrome example {row['example_id']}: helper returned non-binary values"
            )


def _validate_google(rows: list[dict[str, Any]]) -> None:
    _validate_ids_and_splits(rows, "ml_google_decoder_example")

    for row in rows:
        distance = int(row["distance"])
        detector_count = int(row["detector_count"])
        expected_count = DETECTORS_PER_DISTANCE.get(distance)
        if expected_count != detector_count:
            raise MlContractError(
                f"Google example {row['example_id']}: distance {distance} must have "
                f"{expected_count} detectors, found {detector_count}"
            )

        packed = row["detector_bits"]
        expected_bytes = (detector_count + 7) // 8
        if len(packed) != expected_bytes:
            raise MlContractError(
                f"Google example {row['example_id']}: detector_bits has {len(packed)} bytes, "
                f"expected {expected_bytes}"
            )

        used_bits = detector_count % 8
        if used_bits and (packed[-1] >> used_bits):
            raise MlContractError(
                f"Google example {row['example_id']}: unused padding bits are not zero"
            )

        try:
            unpacked = unpack_little_endian_bits(packed, detector_count)
        except ValueError as exc:
            raise MlContractError(
                f"Google example {row['example_id']}: invalid detector_bits: {exc}"
            ) from exc

        if len(unpacked) != detector_count:
            raise MlContractError(
                f"Google example {row['example_id']}: unpacked {len(unpacked)} bits, "
                f"expected {detector_count}"
            )
        event_count = int(sum(unpacked))
        if event_count != int(row["detector_event_count"]):
            raise MlContractError(
                f"Google example {row['example_id']}: detector_event_count={row['detector_event_count']} "
                f"but helper found {event_count} set detector bits"
            )

        required = (*GOOGLE_META_PREDICTION_COLUMNS, "actual_observable_flip")
        if any(row[name] is None for name in required):
            raise MlContractError(
                f"Google example {row['example_id']}: decoder prediction or label is null"
            )

        try:
            features = google_meta_model_input(row)
        except ValueError as exc:
            raise MlContractError(
                f"Google example {row['example_id']}: invalid helper input: {exc}"
            ) from exc
        values = np.asarray(features, dtype=float)
        if values.shape != (5,):
            raise MlContractError(
                f"Google example {row['example_id']}: google_meta_model_input returned "
                f"shape {values.shape}, expected (5,)"
            )
        if not np.all(np.isfinite(values)):
            raise MlContractError(
                f"Google example {row['example_id']}: combined features contain non-finite values"
            )


def _split_rows(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["data_split"] == split]
