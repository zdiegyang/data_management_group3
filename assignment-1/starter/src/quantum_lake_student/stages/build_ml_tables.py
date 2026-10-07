"""Create the two required ML input tables from PostgreSQL Gold.

Required-ml-tables.md: "Build the content of both tables with committed SQL
queries or views against your Gold schema ... A small Python step may run the
SQL, call the supplied split functions, check the result, and write Parquet.
It may not re-read Bronze or Silver."

This stage therefore only:
1. creates the committed views in sql/ml_*.sql and queries them;
2. adds data_split with the supplied syndrome_data_split / google_data_split;
3. checks the documented contract and stops the run on any violation;
4. writes both Parquet tables to the lake (MinIO `ml/`, or LOCAL_LAKE_ROOT/ml/).
"""

from __future__ import annotations

import hashlib
import io
from math import ceil
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    MODEL_SPLITS,
    google_data_split,
    google_meta_model_input,
    syndrome_data_split,
    syndrome_model_input,
    unpack_little_endian_bits,
)
from quantum_lake_student.models import StageResult

SQL_DIR = Path(__file__).resolve().parents[3] / "sql"
# Order matters: the example -> Gold view is built on top of the two ML views.
VIEW_FILES = ("ml_syndrome_example.sql", "ml_google_example.sql", "ml_example_gold_record.sql")

SYNDROME_KEY = "ml/ml_syndrome_decoder_example.parquet"
GOOGLE_KEY = "ml/ml_google_decoder_example.parquet"

# Exact column names, order and types from required-ml-tables.md.
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
    *[(column, pa.bool_()) for column in GOOGLE_META_PREDICTION_COLUMNS],
    ("actual_observable_flip", pa.bool_()),
    ("data_split", pa.string()),
])

# required-ml-tables.md: "detector_count | 200 for distance 3; 600 for distance 5".
DETECTORS_PER_DISTANCE = {3: 200, 5: 600}


class MlContractError(RuntimeError):
    """An ML table violates required-ml-tables.md; the run must stop."""


# ==============================================================================
# Gold queries
# ==============================================================================

def _query(conn: psycopg.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = conn.execute(sql)
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _unresolved_examples(conn: psycopg.Connection) -> dict[str, int]:
    """Examples whose Gold records are missing from gold.v_ml_example_gold_record."""
    # A syndrome example resolves to exactly one existing syndrome_observation row.
    syndrome = conn.execute("""
        SELECT count(*) FROM gold.v_ml_syndrome_decoder_example v
        LEFT JOIN (
            SELECT r.example_id, count(*) AS records
            FROM gold.v_ml_example_gold_record r
            JOIN gold.syndrome_observation o ON o.source_record_id = r.source_record_id
            WHERE r.gold_table = 'syndrome_observation'
            GROUP BY r.example_id
        ) resolved USING (example_id)
        WHERE resolved.records IS DISTINCT FROM 1
    """).fetchone()[0]
    # A Google example resolves to its shot plus one prediction per decoder.
    google = conn.execute(f"""
        SELECT count(*) FROM gold.v_ml_google_decoder_example v
        LEFT JOIN (
            SELECT example_id, count(*) AS records
            FROM gold.v_ml_example_gold_record
            WHERE ml_table = 'ml_google_decoder_example'
            GROUP BY example_id
        ) resolved USING (example_id)
        WHERE resolved.records IS DISTINCT FROM {1 + len(GOOGLE_META_PREDICTION_COLUMNS)}
    """).fetchone()[0]
    return {"ml_syndrome_decoder_example": syndrome, "ml_google_decoder_example": google}


# ==============================================================================
# Contract checks (required-ml-tables.md "Checks your code must perform")
# ==============================================================================

def check_syndrome_examples(rows: list[dict[str, Any]]) -> None:
    """Raise MlContractError on the first violation of the syndrome contract."""
    _check_ids_and_splits("ml_syndrome_decoder_example", rows)
    for row in rows:
        where = f"syndrome example {row['example_id']}"
        if (row["round_count"], row["check_count"]) != (4, 4):
            raise MlContractError(f"{where}: round_count/check_count must be 4/4")
        if row["sample_weight"] <= 0:
            raise MlContractError(f"{where}: sample_weight must be positive")
        try:
            # Supplied helper: exactly 16 binary values, ordered by round then check.
            syndrome_model_input(row["syndrome_bits"])
        except ValueError as error:
            raise MlContractError(f"{where}: {error}") from error


def check_google_examples(rows: list[dict[str, Any]]) -> None:
    """Raise MlContractError on the first violation of the Google contract."""
    _check_ids_and_splits("ml_google_decoder_example", rows)
    for row in rows:
        where = f"Google example {row['example_id']} ({row['experiment_id']} shot {row['shot_index']})"
        detector_count = row["detector_count"]
        if DETECTORS_PER_DISTANCE.get(row["distance"]) != detector_count:
            raise MlContractError(
                f"{where}: distance {row['distance']} must have "
                f"{DETECTORS_PER_DISTANCE.get(row['distance'])} detectors, found {detector_count}"
            )
        packed = row["detector_bits"]
        if len(packed) != ceil(detector_count / 8):
            raise MlContractError(f"{where}: detector_bits has {len(packed)} bytes, expected {ceil(detector_count / 8)}")
        used_bits = detector_count % 8
        if used_bits and packed[-1] >> used_bits:
            raise MlContractError(f"{where}: unused padding bits are not zero")
        # Supplied helper: little-endian unpacking that discards padding.
        if sum(unpack_little_endian_bits(packed, detector_count)) != row["detector_event_count"]:
            raise MlContractError(f"{where}: detector_event_count does not equal the set detector bits")
        if any(row[column] is None for column in (*GOOGLE_META_PREDICTION_COLUMNS, "actual_observable_flip")):
            raise MlContractError(f"{where}: label or decoder prediction is missing (not aligned)")
        try:
            # Supplied helper: event density plus the four aligned predictions.
            google_meta_model_input(row)
        except ValueError as error:
            raise MlContractError(f"{where}: {error}") from error


def _check_ids_and_splits(table: str, rows: list[dict[str, Any]]) -> None:
    example_ids = [row["example_id"] for row in rows]
    if len(set(example_ids)) != len(example_ids):
        raise MlContractError(f"{table}: example_id values are not unique")
    present = {row["data_split"] for row in rows}
    empty = [split for split in MODEL_SPLITS if split not in present]
    if empty:
        raise MlContractError(f"{table}: empty data split(s) {empty}")
    unexpected = present - set(MODEL_SPLITS)
    if unexpected:
        raise MlContractError(f"{table}: unexpected data_split values {sorted(unexpected)}")


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {split: sum(1 for row in rows if row["data_split"] == split) for split in MODEL_SPLITS}


# ==============================================================================
# Writing to the lake
# ==============================================================================

def to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Exactly the contract columns, in contract order and types."""
    return pa.Table.from_pylist(
        [{field.name: row[field.name] for field in schema} for row in rows], schema=schema
    )


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def _write_to_lake(settings: Settings, key: str, data: bytes) -> str:
    """Write one object to the lake and return where it went."""
    if settings.lake_backend == "local":
        path = settings.local_lake_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)
    minio_client(settings).put_object(
        settings.s3_bucket, key, io.BytesIO(data), length=len(data),
        content_type="application/vnd.apache.parquet",
    )
    return f"s3://{settings.s3_bucket}/{key}"


# ==============================================================================
# Stage
# ==============================================================================

def run(run_id: str, settings: Settings | None = None) -> StageResult:
    if settings is None:
        settings = Settings.from_environment()
    result = StageResult(stage="build_ml_tables", run_id=run_id)

    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.transaction():
            for name in VIEW_FILES:
                conn.execute((SQL_DIR / name).read_text())
        syndrome_rows = _query(conn, """
            SELECT * FROM gold.v_ml_syndrome_decoder_example ORDER BY experiment_id, example_id
        """)
        google_rows = _query(conn, """
            SELECT * FROM gold.v_ml_google_decoder_example ORDER BY experiment_id, shot_index
        """)
        unresolved = _unresolved_examples(conn)

    for row in syndrome_rows:
        row["data_split"] = syndrome_data_split(row["physical_fault_rate"])
    for row in google_rows:
        row["data_split"] = google_data_split(row["shot_index"])

    check_syndrome_examples(syndrome_rows)
    check_google_examples(google_rows)
    if any(unresolved.values()):
        raise MlContractError(f"example_id values that do not resolve to Gold records: {unresolved}")

    outputs = {}
    for key, rows, schema in (
        (SYNDROME_KEY, syndrome_rows, SYNDROME_SCHEMA),
        (GOOGLE_KEY, google_rows, GOOGLE_SCHEMA),
    ):
        data = _parquet_bytes(to_table(rows, schema))
        outputs[key] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(data).hexdigest(),
            "location": _write_to_lake(settings, key, data),
            "split_counts": _split_counts(rows),
        }

    result.input_count = len(syndrome_rows) + len(google_rows)
    result.output_count = result.input_count
    result.details["output_table_counts"] = {key: output["rows"] for key, output in outputs.items()}
    result.details["ml_tables"] = outputs
    result.finish()
    return result
