"""
Measure alternative detector-event representations.

It is performed on the Google QEC dataset before the data is loaded into Gold.

Compared representations:
- Packed: one row per shot with detector_bits stored as binary.
- Sparse long: one row per fired detector event.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from io import BytesIO


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.models import StageResult
from quantum_lake_student.connections import minio_client

SILVER_SHOT_KEY = "silver/google_qec/shot.parquet"

def read_parquet_from_lake(
    settings: Settings,
    key: str,
    *,
    columns: list[str] | None = None,
) -> pa.Table:
    """Read a Parquet object from either the local lake or MinIO."""

    if settings.lake_backend == "local":
        path = settings.local_lake_root / key
        if not path.is_file():
            raise FileNotFoundError(f"Parquet object not found: {path}")
        return pq.read_table(path, columns=columns)

    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, key)

    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()

    return pq.read_table(
        BytesIO(data),
        columns=columns,
    )

def detector_event_rows(
    shot_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
    """Expand packed detector bits into one row per fired detector"""

    rows: list[dict[str, Any]] = []
    for row in shot_df.itertuples(index=False):
        # detector_bits stored as bytes in the Silver Parquet table
        detector_bytes = row.detector_bits
        if detector_bytes is None:
            continue

        detector_value = int.from_bytes(detector_bytes, byteorder="little")

        detector_index = 0

        while detector_value:
            if detector_value & 1:
                rows.append(
                    {
                        "experiment_id": row.experiment_id,
                        "shot_index": int(row.shot_index),
                        "detector_index": detector_index,
                    }
                )

            detector_value >>= 1
            detector_index += 1

    return rows

def measure_detector_storage(
    shot_table: pa.Table,
    output_path: Path | None = None,
    ) -> dict[str, Any]:
    """Measure packed and sparse-long detector representations fairly"""
    
    required_columns = {
        "experiment_id",
        "shot_index",
        "detector_bits",
        "detector_event_count",
    }

    missing = required_columns - set(shot_table.column_names)

    if missing:
        raise ValueError(f"Missing required Silver columns: {sorted(missing)}")
    if shot_table.num_rows == 0:
        raise ValueError("Google QEC shot table is empty.")

    shot_df = shot_table.to_pandas()

    shot_count = len(shot_df)

    total_events = int(shot_df["detector_event_count"].sum())

    average_events_per_shot = total_events / shot_count if shot_count else 0.0

    max_events_per_shot = int(shot_df["detector_event_count"].max())

    shots_with_events = int((shot_df["detector_event_count"] > 0).sum())

    shots_without_events = shot_count - shots_with_events

    # Packed Representation
    packed_df = shot_df[
        [
            "experiment_id",
            "shot_index",
            "detector_bits",
        ]
    ].copy()

    # Sparse-long Representation
    sparse_rows = detector_event_rows(shot_df)
    sparse_row_count = len(sparse_rows)

    if sparse_row_count != total_events:
        raise ValueError(
            "Detector event count mismatch: "
            f"Silver detector_event_count sums to {total_events}, "
            f"but detector_bits expand to {sparse_row_count} events."
        )

    sparse_df = pd.DataFrame(sparse_rows, columns=[
        "experiment_id",
        "shot_index",
        "detector_index",
        ],
    )

    with tempfile.TemporaryDirectory(
        prefix="detector_storage_"
    ) as tmp_dir:
        tmp_dir = Path(tmp_dir)

        packed_path = tmp_dir / "detector_packed.parquet"
        sparse_path = tmp_dir / "detector_sparse.parquet"

        tmp_dir.mkdir(parents=True, exist_ok=True)

        packed_table = pa.Table.from_pandas(packed_df, preserve_index=False)

        pq.write_table(
            packed_table,
            packed_path,
            compression="zstd",
        )

        sparse_table = pa.Table.from_pandas(sparse_df, preserve_index=False)

        pq.write_table(
            sparse_table,
            sparse_path,
            compression="zstd",
        )
    
        packed_parquet_bytes = packed_path.stat().st_size
        sparse_parquet_bytes = sparse_path.stat().st_size


    # Derived metrics
    sparse_row_count = len(sparse_df)

    result = {
        "input": {
            "lake_object": SILVER_SHOT_KEY,
        },
        "source_statistics": {
            "shot_count": shot_count,
            "total_detector_events": total_events,
            "average_detector_events_per_shot": average_events_per_shot,
            "max_detector_events_per_shot": max_events_per_shot,
            "shots_with_detector_events": shots_with_events,
            "shots_without_detector_events": shots_without_events,
        },
        "packed_representation": {
            "description": (
                "Detector-only: one row per shot with "
                "detector_bits stored as packed bytes."
            ),
            "rows": shot_count,
            "parquet_bytes": packed_parquet_bytes,
        },
        "sparse_long_representation": {
            "description": (
                "Detector-only: one row per fired detector event."
            ),
            "rows": sparse_row_count,
            "parquet_bytes": sparse_parquet_bytes,
        },
        "comparison": {
            "sparse_rows_per_packed_row": (
                sparse_row_count / shot_count
            ),
            "row_count_ratio_sparse_to_packed": (
                sparse_row_count / shot_count
            ),
            "serialized_size_ratio_sparse_to_packed": (
                sparse_parquet_bytes / packed_parquet_bytes
            ),
            "packed_smaller": (
                packed_parquet_bytes < sparse_parquet_bytes
            ),
        },
    }

    # Write JSON result
    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

    return result

def run(
        run_id: str,
        settings: Settings | None = None,
    ) -> StageResult:

    if settings is None:
        settings = Settings.from_environment()

    result = StageResult(
        stage="measure_detector_storage",
        run_id=run_id,
    )

    results_base = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "part1"
    )

    output_path = (
        results_base
        / "detector_storage_measurement.json"
    )

    shot_table = read_parquet_from_lake(
        settings,
        SILVER_SHOT_KEY,
        columns=[
            "experiment_id",
            "shot_index",
            "detector_bits",
            "detector_event_count",
        ],
    )

    measurement = measure_detector_storage(
        shot_table = shot_table,
        output_path=output_path,
    )

    # Treat the number of Silver shots as the input/output count
    # for this measurement stage.
    result.input_count = measurement["source_statistics"]["shot_count"]
    result.output_count = 1
    result.issue_count = 0

    result.finish()

    return result