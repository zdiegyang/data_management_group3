"""Part I analyses: run the committed SQL against Gold and save the answers.

Brief: "Answer all three with reproducible SQL against Gold ... At least one
query must join three or more Gold tables. Describe associations in the data;
do not claim that one variable causes another." Results layout:
results/part1/analysis/. results/README.md: "do not fill them in manually".

Every query lives in sql/analysis/*.sql. This stage runs each one (read-only),
writes its result to results/part1/analysis/<query>.csv, and generates
answers.md with short interpretations that quote this run's numbers.
"""
from __future__ import annotations

import csv
import io
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import StageResult

PROJECT = Path(__file__).resolve().parents[3]  # starter/
SQL_DIR = PROJECT / "sql" / "analysis"
OUTPUT_DIR = PROJECT / "results" / "part1" / "analysis"


def _plain(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


def run_query(conn: psycopg.Connection, name: str) -> tuple[list[str], list[dict[str, Any]]]:
    cursor = conn.execute((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"))
    columns = [column.name for column in cursor.description]
    return columns, [{c: _plain(v) for c, v in zip(columns, row)} for row in cursor.fetchall()]


def _csv_text(columns: list[str], rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()



# ==============================================================================
# Stage
# ==============================================================================

def run(run_id: str, settings: Settings | None = None) -> StageResult:
    settings = settings or Settings.from_environment()
    result = StageResult(stage="run_analyses", run_id=run_id)

    names = sorted(path.stem for path in SQL_DIR.glob("*.sql"))
    results: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, str] = {}
    with psycopg.connect(settings.postgres_dsn) as conn:
        for name in names:
            columns, rows = run_query(conn, name)
            results[name] = rows
            files[f"{name}.csv"] = _csv_text(columns, rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("*"):
        if stale.is_file() and stale.name not in files:
            stale.unlink()  # keep only this run's outputs
    for filename, text in files.items():
        (OUTPUT_DIR / filename).write_text(text, encoding="utf-8", newline="\n")
        if settings.lake_backend == "minio":
            minio_client(settings).fput_object(
                settings.s3_bucket, f"results/part1/analysis/{filename}", str(OUTPUT_DIR / filename)
            )

    result.input_count = len(names)
    result.output_count = len(files)
    result.details["output_table_counts"] = {
        f"results/part1/analysis/{name}.csv": len(rows) for name, rows in results.items()
    }
    result.finish()
    return result
