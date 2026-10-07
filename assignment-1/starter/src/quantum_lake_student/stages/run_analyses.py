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
# Interpretations (generated from the query results)
# ==============================================================================

def interpret_q1(summary: list[dict[str, Any]]) -> str:
    low, high = summary[0], summary[-1]
    return (
        f"Across the seven simulated files (physical fault rate {low['physical_fault_rate']:g} to "
        f"{high['physical_fault_rate']:g}; 10,000,000 weighted shots each), the weighted logical-error "
        f"rate rises from {low['weighted_logical_error_rate']:.6f} to {high['weighted_logical_error_rate']:.6f}. "
        f"Over the same range the share of shots whose syndrome is all zeros falls from "
        f"{100 * low['all_zero_syndrome_share']:.2f} % to {100 * high['all_zero_syndrome_share']:.2f} %, "
        f"and the weighted mean number of fired syndrome bits rises from "
        f"{low['weighted_mean_fired_bits']:.3f} to {high['weighted_mean_fired_bits']:.3f}. "
        f"The number of distinct syndromes grows from {low['distinct_syndromes']:,} to "
        f"{high['distinct_syndromes']:,}, and the number of syndromes seen with both labels from "
        f"{low['syndromes_with_both_labels']:,} to {high['syndromes_with_both_labels']:,}: at higher fault "
        "rates the same syndrome increasingly occurs both with and without a logical error, so a "
        "syndrome alone determines the label less and less. All rates are weighted by `quantity`. "
        "These are associations across seven simulation settings, not causal estimates."
    )


def interpret_q2(rows: list[dict[str, Any]]) -> str:
    d3 = [row for row in rows if row["distance"] == 3]
    d5 = [row for row in rows if row["distance"] == 5]
    locations = sorted({(row["center_row"], row["center_col"]) for row in d3})
    best_per_location = {
        loc: min((r for r in d3 if (r["center_row"], r["center_col"]) == loc),
                 key=lambda r: r["logical_error_rate"])["decoder_name"]
        for loc in locations
    }
    ranges = []
    for decoder in sorted({row["decoder_name"] for row in d3}):
        rates = [row["logical_error_rate"] for row in d3 if row["decoder_name"] == decoder]
        ranges.append(f"{decoder} {min(rates):.3f}–{max(rates):.3f}")
    best_d5 = min(d5, key=lambda r: r["logical_error_rate"]) if d5 else None
    worst_d5 = max(d5, key=lambda r: r["logical_error_rate"]) if d5 else None
    flip_rates = [row["actual_flip_rate"] for row in rows]
    same_best = len(set(best_per_location.values())) == 1
    text = (
        f"At distance 3, the logical-error rate of each decoder varies across the {len(locations)} "
        f"processor locations ({'; '.join(ranges)}). "
        + (f"{next(iter(best_per_location.values()))} has the lowest rate at every location. "
           if same_best else
           "The decoder with the lowest rate differs by location ("
           + ", ".join(f"centre {r}_{c}: {name}" for (r, c), name in best_per_location.items()) + "). ")
    )
    if best_d5:
        text += (
            f"At distance 5 (one experiment, centre {best_d5['center_row']}_{best_d5['center_col']}) the rates "
            f"range from {best_d5['logical_error_rate']:.3f} ({best_d5['decoder_name']}) to "
            f"{worst_d5['logical_error_rate']:.3f} ({worst_d5['decoder_name']}), similar to distance 3. "
        )
    text += (
        f"The actual flip rate is between {min(flip_rates):.3f} and {max(flip_rates):.3f} everywhere: over 25 "
        "rounds the observable flips in about half of the shots, so all decoders stay near 0.4. With one "
        "distance-5 experiment, distance and location cannot be separated; the differences are "
        "associations between experiments, not effects of location or distance. This query joins four "
        "Gold tables (google_experiment, google_shot, decoder_prediction, decoder)."
    )
    return text


def interpret_q3(checks: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> str:
    variant = "source"
    qubits = {row["ancilla_qubit"]: set(row["data_qubits"].split(", "))
              for row in checks if row["variant"] == variant}
    lines = []
    consistent = True
    for row in (r for r in corrections if r["variant"] == variant):
        fired = [name for name in qubits if row["fired_checks"] and name in row["fired_checks"]]
        silent = [name for name in qubits if name not in fired]
        explained = set.intersection(*(qubits[n] for n in fired)) - set().union(*(qubits[n] for n in silent))
        consistent &= explained == {row["target_qubit"]}
        lines.append(
            f"`{row['condition_register']} = {row['condition_value']}`: {' and '.join(fired)} fire"
            f"{'s' if len(fired) == 1 else ''} → `{row['gate']} {row['target_qubit']}`"
        )
    check_text = "; ".join(
        f"{row['ancilla_qubit']} measures the parity of {row['data_qubits']} into {row['syndrome_bit']}"
        for row in checks if row["variant"] == variant
    )
    variants_equal = (
        [{k: v for k, v in r.items() if k != "variant"} for r in corrections if r["variant"] == "source"]
        == [{k: v for k, v in r.items() if k != "variant"} for r in corrections if r["variant"] == "transpiled"]
    )
    return (
        f"In `qec_sm_n5`, {check_text}. A correction is conditioned on the whole two-bit syndrome "
        "register, and bit k of its value is the syndrome bit written by check k: "
        + "; ".join(lines) + ". "
        + ("In every case the corrected qubit is the only data qubit that belongs to all fired checks "
           "and to no silent check — the flipped qubit the syndrome points to. "
           if consistent else "The corrected qubit does not always follow from the fired checks. ")
        + ("The transpiled variant has the same mapping." if variants_equal
           else "The transpiled variant differs from the source variant.")
    )


def interpret_q4(rows: list[dict[str, Any]]) -> str:
    matches = sum(row["matching_circuits"] for row in rows)
    return (
        f"Six candidate join keys between QASMBench circuits and the Google or simulated experiments "
        f"were tested; together they match {matches} circuit rows. "
        + ("No identifier, name or qubit count links a circuit to an experiment, so the relationship is "
           "rejected and Gold has no foreign key between them." if matches == 0 else
           "At least one candidate key matches; review before rejecting the relationship.")
    )


QUESTIONS = [
    ("Q1. How do weighted syndrome frequency and logical-error labels change with physical fault rate?",
     ["q1_fault_rate_summary", "q1_by_fired_bits"]),
    ("Q2. How do the supplied decoder logical-error rates compare by code distance and distance-three "
     "processor location?", ["q2_decoder_error_rates"]),
    ("Q3. How does the repetition-code circuit map data qubits to parity-check ancillas, syndrome bits, "
     "and conditional corrections?", ["q3_parity_checks", "q3_corrections"]),
    ("Rejected relationship: can QASMBench circuits be joined to the experiments?",
     ["q4_rejected_relationship"]),
]


def answers_markdown(results: dict[str, list[dict[str, Any]]]) -> str:
    interpretations = [
        interpret_q1(results["q1_fault_rate_summary"]),
        interpret_q2(results["q2_decoder_error_rates"]),
        interpret_q3(results["q3_parity_checks"], results["q3_corrections"]),
        interpret_q4(results["q4_rejected_relationship"]),
    ]
    lines = [
        "# Part I analyses",
        "",
        "Generated by `make run` (stage `run_analyses`): each query in `sql/analysis/` runs against",
        "Gold and its result is saved next to this file as `<query>.csv`. Do not edit by hand.",
        "",
    ]
    for (question, queries), interpretation in zip(QUESTIONS, interpretations):
        lines += [f"## {question}", ""]
        lines += [f"Queries: " + ", ".join(f"`sql/analysis/{q}.sql` → `{q}.csv`" for q in queries), ""]
        lines += [interpretation, ""]
    return "\n".join(lines)


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
    files["answers.md"] = answers_markdown(results)

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
