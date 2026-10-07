"""Part I demonstration: results/part1/trace_examples.json.

Brief: "Demonstrate the complete trace for one syndrome prediction and one
Google prediction in trace_examples.json." Checklist: "Trace one syndrome
prediction through ML, Gold, Silver, and Bronze" and "Trace one Google
prediction to its shot and aligned Bronze members."

Each trace starts from a real row of results/part2/predictions.parquet and
follows the identifiers back, one hop at a time:

    prediction (example_id, model_id)
      -> ML table row                      (ml/*.parquet, by example_id)
      -> Gold record(s)                    (gold.v_ml_example_gold_record)
      -> Silver row                        (silver/*.parquet, by source_record_id)
      -> source_trace.parquet row(s)       (by source_record_id)
      -> Bronze object, member, position, SHA-256
         (hash checked against the verified input hash in results/part1/run.json)

It needs Part II's predictions, so it runs after `make train` (`make trace`).
It only reads; it never writes to Bronze, Silver, Gold or source_trace.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import psycopg
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import StageResult

PROJECT = Path(__file__).resolve().parents[2]  # starter/
RESULTS = PROJECT / "results" / "part1"
PREDICTIONS = PROJECT / "results" / "part2" / "predictions.parquet"

# Fixed, test-split examples so the file is identical on every run.
TRACES = {
    "syndrome_prediction_trace": {
        "model_id": "task_a_logistic",
        "source_record_id": "qec_syndromes:d-3_pfr-0.005000_nb-10M.csv:row:0",
        "ml_key": "ml/ml_syndrome_decoder_example.parquet",
        "ml_view": "gold.v_ml_syndrome_decoder_example",
        "silver_key": "silver/qec_syndromes/syndrome_observation.parquet",
    },
    "google_prediction_trace": {
        "model_id": "task_b_d3_combined",
        "source_record_id": "google_qec:surface_code_bX_d3_r25_center_3_5:shot:1",
        "ml_key": "ml/ml_google_decoder_example.parquet",
        "ml_view": "gold.v_ml_google_decoder_example",
        "silver_key": "silver/google_qec/shot.parquet",
    },
}
TRACE_KEY = "results/part1/source_trace.parquet"
GOLD_ROW_QUERIES = {
    "syndrome_observation": "SELECT * FROM gold.syndrome_observation WHERE source_record_id = %s",
    "google_shot": "SELECT * FROM gold.google_shot WHERE source_record_id = %s",
    "decoder_prediction": (
        "SELECT * FROM gold.decoder_prediction WHERE shot_source_record_id = %s ORDER BY decoder_name"
    ),
}


class TraceError(RuntimeError):
    """A hop of a trace cannot be resolved; the demonstration must not be published."""


def _plain(value: Any) -> Any:
    """JSON-friendly values: packed bytes as hex, everything else unchanged."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return value


def _row(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _plain(value) for key, value in record.items()}


def _read_lake_table(settings: Settings, key: str):
    if settings.lake_backend == "local":
        return pq.read_table(settings.local_lake_root / key)
    response = minio_client(settings).get_object(settings.s3_bucket, key)
    try:
        return pq.read_table(io.BytesIO(response.read()))
    finally:
        response.close()
        response.release_conn()


def _rows_where(table, column: str, value: str) -> list[dict[str, Any]]:
    return table.filter(pc.equal(table[column], value)).to_pylist()


def _one(rows: list[dict[str, Any]], what: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise TraceError(f"expected exactly one {what}, found {len(rows)}")
    return rows[0]


def _query(conn: psycopg.Connection, sql: str, value: str) -> list[dict[str, Any]]:
    cursor = conn.execute(sql, (value,))
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def build_trace(
    spec: dict[str, str],
    conn: psycopg.Connection,
    predictions,
    trace_table,
    lake_tables: dict[str, Any],
    verified_hashes: dict[str, str],
) -> dict[str, Any]:
    """Follow one prediction back to the Bronze bytes it came from."""
    source_record_id = spec["source_record_id"]

    # Gold: which ML example was built from this source record.
    example = _one(_query(conn, f"SELECT example_id FROM {spec['ml_view']} WHERE source_record_id = %s",
                          source_record_id), f"ML example for {source_record_id}")
    example_id = example["example_id"]

    # 1. The prediction itself (Part II output).
    prediction = _one(
        [row for row in _rows_where(predictions, "example_id", example_id) if row["model_id"] == spec["model_id"]],
        f"{spec['model_id']} prediction for {example_id}",
    )
    # 2. The ML input row it was made from.
    ml_row = _one(_rows_where(lake_tables[spec["ml_key"]], "example_id", example_id), f"ML row {example_id}")
    # 3. The Gold record(s) the ML row was built from.
    links = _query(conn, "SELECT gold_table, source_record_id, decoder_name FROM gold.v_ml_example_gold_record "
                         "WHERE example_id = %s ORDER BY gold_table, decoder_name", example_id)
    if not links:
        raise TraceError(f"{example_id} does not resolve to any Gold record")
    gold_records = {table: [_row(r) for r in _query(conn, GOLD_ROW_QUERIES[table], source_record_id)]
                    for table in sorted({link["gold_table"] for link in links})}
    # 4. The Silver row.
    silver_row = _one(_rows_where(lake_tables[spec["silver_key"]], "source_record_id", source_record_id),
                      f"Silver row {source_record_id}")
    # 5. The source trace rows and 6. the Bronze object they point to.
    trace_rows = sorted(_rows_where(trace_table, "source_record_id", source_record_id),
                        key=lambda row: row["archive_member"])
    if not trace_rows:
        raise TraceError(f"{source_record_id} has no row in source_trace.parquet")
    bronze_objects = {(row["bronze_object"], row["input_sha256"]) for row in trace_rows}
    bronze = [
        {"object": obj, "sha256": sha, "matches_verified_input_hash": verified_hashes.get(obj) == sha}
        for obj, sha in sorted(bronze_objects)
    ]
    if not all(item["matches_verified_input_hash"] for item in bronze):
        raise TraceError(f"{source_record_id}: Bronze hash differs from the hash verified in run.json")

    return {
        "path": "prediction -> ML row -> Gold record(s) -> Silver row -> source_trace -> Bronze",
        "1_prediction": {"file": "results/part2/predictions.parquet", "row": _row(prediction)},
        "2_ml_row": {"table": spec["ml_key"], "row": _row(ml_row)},
        "3_gold": {"view": "gold.v_ml_example_gold_record", "links": [_row(link) for link in links],
                   "records": gold_records},
        "4_silver_row": {"table": spec["silver_key"], "row": _row(silver_row)},
        "5_source_trace": {"table": TRACE_KEY, "rows": [_row(row) for row in trace_rows]},
        "6_bronze": {"objects": bronze,
                     "members": [{"archive_member": row["archive_member"],
                                  "record_locator": row["record_locator"]} for row in trace_rows]},
    }


def run(run_id: str, settings: Settings | None = None) -> StageResult:
    settings = settings or Settings.from_environment()
    result = StageResult(stage="build_trace_examples", run_id=run_id)

    if not PREDICTIONS.exists():
        raise TraceError(f"{PREDICTIONS} not found: run `make train` before `make trace`")
    run_record = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    predictions = pq.read_table(PREDICTIONS)
    trace_table = _read_lake_table(settings, TRACE_KEY)
    lake_tables = {key: _read_lake_table(settings, key)
                   for spec in TRACES.values() for key in (spec["ml_key"], spec["silver_key"])}

    output = {}
    with psycopg.connect(settings.postgres_dsn) as conn:
        for name, spec in TRACES.items():
            output[name] = build_trace(spec, conn, predictions, trace_table, lake_tables,
                                       run_record.get("input_hashes", {}))

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "trace_examples.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if settings.lake_backend == "minio":
        minio_client(settings).fput_object(settings.s3_bucket, "results/part1/trace_examples.json", str(path))

    result.input_count = len(TRACES)
    result.output_count = sum(len(trace["5_source_trace"]["rows"]) for trace in output.values())
    result.details["trace_examples"] = str(path)
    result.finish()
    return result
