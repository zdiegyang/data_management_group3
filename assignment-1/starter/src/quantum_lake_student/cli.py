"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform, minio_client
from .models import StageResult
from .provenance import STARTER_DIR, code_sha256, git_revision
from quantum_lake_student import build_trace
from .stages import (
    build_ml_tables,
    load_postgres,
    measure_detector,
    measure_gold_storage,
    prepare_data,
    register_sources,
    train
)


console = Console()

RESULTS_DIR = STARTER_DIR / "results" / "part1"


def command_check(settings: Settings) -> int:
    result = check_platform(settings)
    for service, message in result.items():
        console.print(f"[green]OK[/green] {service}: {message}")
    return 0


def command_inventory(settings: Settings) -> int:
    table = Table(title="Supplied course data files (kept unchanged)")
    table.add_column("Stored path")
    table.add_column("Bytes", justify="right")
    for key, size in bronze_inventory(settings):
        table.add_row(key, f"{size:,}")
    console.print(table)
    return 0


# ==============================================================================
# Run record: results/part1/run.json is written here, once per run.
# Spec: run.json records "input hashes, code revision, start/end times, and
# every output count".
# ==============================================================================

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_run_record(
    run_id: str,
    started_at: datetime,
    stage_results: list[StageResult],
    error: BaseException | None = None,
    settings: Settings | None = None,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write run.json for the whole Part I run from the stages' results."""
    details = {result.stage: result.details for result in stage_results}

    output_table_counts: dict[str, Any] = {}
    for stage_details in details.values():
        output_table_counts.update(stage_details.get("output_table_counts", {}))
    if "load_postgres" in details:
        gold_counts = _read_json(results_dir / "row_counts.json").get("silver_to_gold", {})
        for table, comparison in gold_counts.items():
            loaded_count = (
                comparison.get("loaded_count")
                if isinstance(comparison, dict)
                else comparison
            )
            output_table_counts[f"gold.{table}"] = loaded_count

    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "failed" if error else "succeeded",
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "code_sha256": code_sha256(),
        "input_hashes": details.get("register_sources", {}).get("input_hashes", {}),
        "stages": [
            {
                "stage": result.stage,
                "started_at": result.started_at.isoformat(),
                "ended_at": result.finished_at.isoformat() if result.finished_at else None,
                "input_count": result.input_count,
                "output_count": result.output_count,
                "issue_count": result.issue_count,
            }
            for result in stage_results
        ],
        "output_table_counts": output_table_counts,
        "issue_count": sum(result.issue_count for result in stage_results),
        "checks": details.get("prepare_data", {}).get("checks", []),
        # Part II records "the hashes of both input tables"; Part I records them too.
        "ml_tables": details.get("build_ml_tables", {}).get("ml_tables", {}),
    }
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "run.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if settings is not None and settings.lake_backend == "minio":
        try:
            minio_client(settings).fput_object(settings.s3_bucket, "results/part1/run.json", str(path))
        except Exception as upload_error:
            console.print(f"[yellow]Warning: run.json not uploaded to MinIO: {upload_error}[/yellow]")
    return path


def command_run(settings: Settings) -> int:
    started_at = datetime.now(UTC)
    run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"
    console.print(f"[bold blue]Starting Part I Pipeline (Run ID: {run_id})...[/bold blue]")

    stage_results: list[StageResult] = []
    try:
        _run_part1_stages(run_id, settings, stage_results)
    except Exception as exc:
        # A failed stage stops the run; run.json still records what happened.
        write_run_record(run_id, started_at, stage_results, error=exc, settings=settings)
        raise
    run_path = write_run_record(run_id, started_at, stage_results, settings=settings)
    console.print(f"[bold green]✓ Run record written to {run_path}[/bold green]")
    return 0


def _run_part1_stages(
    run_id: str,
    settings: Settings,
    stage_results: list[StageResult],
) -> None:
    # Stage 1: Bronze verification. Raises on a missing object, checksum
    # mismatch, or unsafe archive member, which stops the run before parsing.
    res_bronze = register_sources.run(run_id=run_id)
    stage_results.append(res_bronze)
    duration = (
        (res_bronze.finished_at - res_bronze.started_at).total_seconds()
        if res_bronze.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Stage 1 (register_sources):[/green] "
        f"{res_bronze.output_count}/{res_bronze.input_count} Bronze objects verified "
        f"({res_bronze.issue_count} unexpected objects) in {duration:.2f}s"
    )

    # Stage 2: Data Preparation & Silver
    res_prep = prepare_data.run(run_id=run_id, settings=settings)
    stage_results.append(res_prep)
    duration = (
        (res_prep.finished_at - res_prep.started_at).total_seconds()
        if res_prep.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Stage 2 (prepare_data):[/green] "
        f"{res_prep.input_count:,} inputs -> {res_prep.output_count:,} outputs "
        f"({res_prep.issue_count} issues) in {duration:.2f}s"
    )

    # Stage 2.5: Measure detection-event storage on Silver tables
    res_measure = measure_detector.run(run_id=run_id, settings=settings)
    stage_results.append(res_measure)
    duration = (
        (res_measure.finished_at - res_measure.started_at).total_seconds()
        if res_measure.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Measuring detector-event storage:[/green] "
        f"{res_measure.input_count:,} inputs -> {res_measure.output_count:,} outputs "
        f"({res_measure.issue_count} issues) in {duration:.2f}s"
    )
    console.print("[bold green]✓ Part I Detection-event storage measured and saved as JSON![/bold green]")

    # Stage 3: Silver -> Gold (PostgreSQL)
    res_gold = load_postgres.run(run_id=run_id)
    stage_results.append(res_gold)
    duration = (
        (res_gold.finished_at - res_gold.started_at).total_seconds()
        if res_gold.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Stage 3 (load_postgres):[/green] "
        f"{res_gold.input_count:,} inputs -> {res_gold.output_count:,} outputs "
        f"({res_gold.issue_count} issues) in {duration:.2f}s"
    )

    console.print("[bold green]✓ Part I Data Preparation completed successfully![/bold green]")

    # Stage 3.5: Measure Gold PostgreSQL table storage
    gold_measure = measure_gold_storage.run(run_id=run_id, settings=settings)
    stage_results.append(gold_measure)
    duration = (
        (gold_measure.finished_at - gold_measure.started_at).total_seconds()
        if gold_measure.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Measuring PostgreSQL storage:[/green] "
        f"{gold_measure.input_count:,} inputs -> {gold_measure.output_count:,} outputs "
        f"({gold_measure.issue_count} issues) in {duration:.2f}s"
    )
    console.print("[bold green]✓ Part I PostgreSQL storage measured and saved as JSON![/bold green]")

    # Stage 4: Gold -> ML. Reads only Gold views; writes both ML tables to the lake.
    res_ml = build_ml_tables.run(run_id=run_id, settings=settings)
    stage_results.append(res_ml)
    duration = (
        (res_ml.finished_at - res_ml.started_at).total_seconds()
        if res_ml.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Stage 4 (build_ml_tables):[/green] "
        f"{res_ml.output_count:,} ML examples written "
        + ", ".join(f"{key} ({out['rows']:,})" for key, out in res_ml.details["ml_tables"].items())
        + f" in {duration:.2f}s"
    )


def command_train(settings: Settings) -> int:
    """Run Part II from the two materialized ML Parquet tables only."""
    started_at = datetime.now(UTC)
    run_id = f"model_{started_at.strftime('%Y%m%d_%H%M%S')}"
    console.print(f"[bold blue]Starting Part II ML stage (Run ID: {run_id})...[/bold blue]")

    result = train.run(model_run_id=run_id, settings=settings)
    duration = (
        (result.finished_at - result.started_at).total_seconds()
        if result.finished_at
        else 0.0
    )
    console.print(
        f"[green]✓ Part II:[/green] {result.input_count:,} inputs -> "
        f"{result.output_count:,} prediction rows ({result.issue_count} issues) "
        f"in {duration:.2f}s"
    )
    console.print("[bold green]✓ results/part2 regenerated successfully![/bold green]")
    return 0


def command_trace(settings: Settings) -> int:
    """Part I demonstration, run after `make train`: trace one syndrome and one
    Google prediction back to Bronze into results/part1/trace_examples.json.

    Kept out of `make train` because Part II may read only the two ML tables,
    while a trace must read Gold, Silver and the source trace.
    """
    run_id = f"trace_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    result = build_trace.run(run_id=run_id, settings=settings)
    console.print(
        f"[green]✓ Trace examples:[/green] {result.input_count} predictions traced to "
        f"{result.output_count} Bronze trace rows -> {result.details['trace_examples']}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "train", "trace"),
        help="Action to perform",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    settings = Settings.from_environment()
    commands = {
        "check": command_check,
        "inventory": command_inventory,
        "run": command_run,
        "train": command_train,
        "trace": command_trace,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
