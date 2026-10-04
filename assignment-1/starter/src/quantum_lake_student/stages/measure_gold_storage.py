from __future__ import annotations

import json
from pathlib import Path

import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.models import StageResult

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results" / "part1"

def measure_gold_storage(
    conn: psycopg.Connection,
) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
            schemaname,
            relname,
            pg_total_relation_size(
                format('%I.%I', schemaname, relname)
            ) AS bytes
        FROM 
            pg_catalog.pg_stat_user_tables
        WHERE schemaname = 'gold'
        ORDER BY relname
        """
    ).fetchall()

    return [
        {
            "schema": row[0],
            "table": row[1],
            "bytes": int(row[2]),
            "megabytes": round(int(row[2]) / (1024**2), 2)
        }
        for row in rows
    ]

def run(
    run_id: str,
    settings: Settings | None = None,
) -> StageResult:
    if settings is None:
        settings.Settings.from_environment()

    result = StageResult(
        stage = "measure_gold_storage",
        run_id = run_id,
    )

    with psycopg.connect(settings.postgres_dsn) as conn:
        tables = measure_gold_storage(conn)

    total_bytes = sum(int(row["bytes"]) for row in tables)

    output = {
        "tables": tables,
        "total": {
            "bytes": total_bytes,
            "megabytes": round(total_bytes / (1024 ** 2), 2),
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)

    (RESULTS / "gold_storage_measurement.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    result.input_count = len(tables)
    result.output_count = len(tables)
    result.issue_count = 0
    result.finish()

    return result
