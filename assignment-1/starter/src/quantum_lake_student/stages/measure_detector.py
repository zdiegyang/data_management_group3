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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.models import StageResult

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
    silver_shot_path: Path,
    silver_experiment_path: Path | None = None,
    output_path: Path | None = None,
    ) -> dict[str, Any]:
    """Measure packed and sparse-long detector representations"""

    shot_table = pq.read_table(silver_shot_path)
    shot_df = shot_table.to_pandas()

    if shot_df.empty:
        raise ValueError("Silver Google QEC shot table is empty.")

    required_columns = {
        "experiment_id",
        "shot_index",
        "detector_bits",
        "detector_event_count",
    }

    missing = required_columns - set(shot_df.columns)

    if missing:
        raise ValueError(f"Missing required Silver columns: {sorted(missing)}")

    shot_count = len(shot_df)

    total_events = int(shot_df["detector_event_count"].sum())

    average_events_per_shot = int(total_events / shot_count if shot_count else 0.0)

    max_events_per_shot = int(shot_df["detector_event_count"].max())

    shots_with_events = int((shot_df["detector_event_count"] > 0).sum())

    shots_without_events = shot_count - shots_with_events

    # Packed Representation
    packed_parquet_bytes = silver_shot_path.stat().st_size

    detector_payload_bytes = int(sum(len(value) for value in shot_df["detector_bits"] if value is not None))

    packed_rows = shot_count

    # Sparse-long Representation
    sparse_rows = detector_event_rows(shot_df)

    sparse_df = pd.DataFrame(sparse_rows, columns=[
        "experiment_id",
        "shot_index",
        "detector_index",
        ],
    )

    sparse_row_count = len(sparse_df)

    # Serialize sparse representation to a temporary Parquet file
    with tempfile.TemporaryDirectory(
        prefix="detector_storage_"
    ) as tmp_dir:
        
        sparse_path = Path(tmp_dir) / "detector_event_long.parquet"

        sparse_table = pa.Table.from_pandas(
            sparse_df,
            preserve_index = False,
        )

        pq.write_table(
            sparse_table,
            sparse_path,
            compression="zstd",
        )

        sparse_parquet_bytes = sparse_path.stat().st_size


    # Derived metrics
    if packed_rows:
        sparse_rows_per_shot = (
            sparse_row_count / packed_rows
        )
    else:
        sparse_rows_per_shot = 0.0

    if packed_parquet_bytes:
        storage_ratio = (
            sparse_parquet_bytes
            / packed_parquet_bytes
        )
    else:
        storage_ratio = None

    if packed_rows:
        row_ratio = (
            sparse_row_count / packed_rows
        )
    else:
        row_ratio = None

    result = {
        "input": {
            "silver_shot_path": str(silver_shot_path),
            "silver_shot_file_bytes": packed_parquet_bytes,
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
                "One row per shot with detector_bits stored as packed bytes."
            ),
            "rows": packed_rows,
            "detector_payload_bytes": detector_payload_bytes,
            "parquet_bytes": packed_parquet_bytes,
        },
        "sparse_long_representation": {
            "description": (
                "One row per fired detector event."
            ),
            "rows": sparse_row_count,
            "parquet_bytes": sparse_parquet_bytes,
        },
        "comparison": {
            "sparse_rows_per_packed_row": sparse_rows_per_shot,
            "row_count_ratio_sparse_to_packed": row_ratio,
            "serialized_size_ratio_sparse_to_packed": storage_ratio,
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

    # prepare_data.py currently writes Silver underneath the starter/
    # directory. Match the same project-root convention.
    base_dir = Path(__file__).resolve().parents[3]

    silver_shot_path = (
        base_dir
        / "silver"
        / "google_qec"
        / "shot.parquet"
    )

    results_base = (
        base_dir
        / "results"
        / "part1"
    )

    output_path = (
        results_base
        / "detector_storage_measurement.json"
    )

    if not silver_shot_path.exists():
        raise FileNotFoundError(
            f"Silver shot table not found: {silver_shot_path}"
        )

    measurement = measure_detector_storage(
        silver_shot_path=silver_shot_path,
        output_path=output_path,
    )

    # Treat the number of Silver shots as the input/output count
    # for this measurement stage.
    result.input_count = measurement["source_statistics"]["shot_count"]
    result.output_count = 1
    result.issue_count = 0

    result.finish()

    return result