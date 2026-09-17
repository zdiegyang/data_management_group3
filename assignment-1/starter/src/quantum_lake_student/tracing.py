"""Provenance and lineage tracing helpers for the course platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Settings
from .connections import minio_client

SOURCE_TRACE_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("source_name", pa.string()),
    ("bronze_object", pa.string()),
    ("archive_member", pa.string()),
    ("record_locator", pa.string()),
    ("input_sha256", pa.string()),
])


def save_source_traces(
    new_records: Sequence[Mapping[str, Any]],
    source_name: str,
    trace_file_path: Path,
    settings: Settings | None = None,
) -> int:
    """Save or update source trace records idempotently.

    Appends new records and deduplicates by (source_record_id, archive_member, record_locator),
    preserving traces across all tables and sources while ensuring zero duplicate rows upon re-runs.
    """
    trace_file_path = Path(trace_file_path)
    trace_file_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(new_records)

    if trace_file_path.exists():
        df_existing = pq.read_table(trace_file_path).to_pandas()
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Deduplicate across the full composite primary key of the trace table
        df_combined = df_combined.drop_duplicates(
            subset=["source_record_id", "archive_member", "record_locator"],
            keep="last",
        )
    else:
        df_combined = df_new

    table = pa.Table.from_pandas(
        df_combined,
        schema=SOURCE_TRACE_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, trace_file_path, compression="zstd")

    if settings and settings.lake_backend == "minio":
        try:
            client = minio_client(settings)
            client.fput_object(
                settings.s3_bucket,
                "results/part1/source_trace.parquet",
                str(trace_file_path),
            )
        except Exception:
            pass

    return table.num_rows
