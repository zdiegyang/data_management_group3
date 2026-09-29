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
    ("valid", pa.string()),  # 'yes' or 'no'
])


def normalize_trace_table(data: Any) -> pa.Table:
    """Convert input trace data (list of dicts, columnar dict, DataFrame, or Table)

    into a PyArrow Table adhering strictly to SOURCE_TRACE_SCHEMA.
    """
    if isinstance(data, pa.Table):
        table = data
        if "valid" not in table.column_names:
            valids = pa.array(["yes"] * table.num_rows, type=pa.string())
            table = table.append_column("valid", valids)
        return table.cast(SOURCE_TRACE_SCHEMA)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if "valid" not in df.columns:
            df["valid"] = "yes"
        return pa.Table.from_pandas(df, schema=SOURCE_TRACE_SCHEMA, preserve_index=False)

    if isinstance(data, dict):
        # Columnar dict
        cols = dict(data)
        num_rows = len(next(iter(cols.values()))) if cols else 0
        if "valid" not in cols:
            cols["valid"] = ["yes"] * num_rows
        return pa.Table.from_pydict(cols, schema=SOURCE_TRACE_SCHEMA)

    if isinstance(data, (list, tuple)):
        # List of row mappings
        if not data:
            return pa.Table.from_pylist([], schema=SOURCE_TRACE_SCHEMA)
        # Ensure 'valid' is populated
        rows = [
            {**r, "valid": r.get("valid", "yes")} for r in data
        ]
        df = pd.DataFrame(rows)
        return pa.Table.from_pandas(df, schema=SOURCE_TRACE_SCHEMA, preserve_index=False)

    raise TypeError(f"Unsupported trace data type: {type(data)}")


def save_source_traces(
    new_records: Any,
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

    table_new = normalize_trace_table(new_records)

    if trace_file_path.exists():
        table_existing = pq.read_table(trace_file_path)
        if "valid" not in table_existing.column_names:
            valids = pa.array(["yes"] * table_existing.num_rows, type=pa.string())
            table_existing = table_existing.append_column("valid", valids)
            table_existing = table_existing.cast(SOURCE_TRACE_SCHEMA)

        df_existing = table_existing.to_pandas()
        df_new = table_new.to_pandas()
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Deduplicate across the full composite primary key of the trace table
        df_combined = df_combined.drop_duplicates(
            subset=["source_record_id", "archive_member", "record_locator"],
            keep="last",
        )
        table_final = pa.Table.from_pandas(
            df_combined,
            schema=SOURCE_TRACE_SCHEMA,
            preserve_index=False,
        )
    else:
        df_new = table_new.to_pandas().drop_duplicates(
            subset=["source_record_id", "archive_member", "record_locator"],
            keep="last",
        )
        table_final = pa.Table.from_pandas(
            df_new,
            schema=SOURCE_TRACE_SCHEMA,
            preserve_index=False,
        )

    pq.write_table(table_final, trace_file_path, compression="zstd")

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

    return table_final.num_rows
