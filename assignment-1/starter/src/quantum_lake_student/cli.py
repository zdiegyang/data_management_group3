"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
import sys

from datetime import UTC, datetime

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform
from .stages import prepare_data, measure_detector


console = Console()


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


def command_run(settings: Settings) -> int:
    run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    console.print(f"[bold blue]Starting Part I Pipeline (Run ID: {run_id})...[/bold blue]")

    # Stage 2: Data Preparation & Silver
    res_prep = prepare_data.run(run_id=run_id, settings=settings)
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

    console.print("[bold green]✓ Part I Data Preparation completed successfully![/bold green]")

    res_measure = measure_detector.run(run_id=run_id, settings=settings)
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

    return 0


def command_train(_: Settings) -> int:
    console.print(
        "[yellow]The AI/ML stage is intentionally unimplemented.[/yellow]\n"
        "Consume the required ML input tables through the supplied helpers and "
        "write model files and the required results/part2 files."
    )
    return 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "train"),
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
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
