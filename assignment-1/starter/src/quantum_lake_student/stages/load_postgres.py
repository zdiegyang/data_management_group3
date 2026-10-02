"""Relational integration stage.

Create the student-designed PostgreSQL tables and load them as one all-or-
nothing update. Primary/foreign keys, value checks, and indexes are part of the
deliverable. Repeated loads must not create duplicate records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import psycopg
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.models import StageResult

PROJECT = Path(__file__).resolve().parents[3]
SQL_DIR = PROJECT / "sql"
SILVER = PROJECT / "silver"
RESULTS = PROJECT / "results/part1"

DECODER_NAMES = [
    "belief_matching",
    "correlated_matching",
    "pymatching",
    "tensor_network_contraction",
]


def _rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _copy_upsert(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    conflict_cols: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> int:
    """Load `rows` into `table` via COPY -> temp table -> INSERT ... ON CONFLICT DO UPDATE.

    Returns the number of rows sent. Update-on-conflict (not DO NOTHING) so a
    rerun after a parser fix corrects existing Gold rows instead of leaving
    them stale, while never creating a duplicate business record.
    """
    rows = list(rows)
    if not rows:
        return 0
    tmp = f"tmp_{table.replace('.', '_')}"
    col_list = ", ".join(columns)
    conn.execute(f"CREATE TEMP TABLE {tmp} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
    with conn.cursor().copy(f"COPY {tmp} ({col_list}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(tuple(row[c] for c in columns))
    update_cols = [c for c in columns if c not in conflict_cols]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    conn.execute(f"""
        INSERT INTO {table} ({col_list})
        SELECT {col_list} FROM {tmp}
        ON CONFLICT ({", ".join(conflict_cols)}) DO UPDATE SET {set_clause}
    """)
    return len(rows)


def _copy_load(
        conn: psycopg.Connection,
        table: str,
        columns: Sequence[str],
        rows: Iterable[dict[str, Any]],
) -> int:
    """Bulk-load `rows` into `table` via COPY. Returns the number of rows sent.

    No upsert logic is needed: gold_schema.sql drops and recreates every Gold
    table at the start of this same transaction, so every table is empty when
    this runs. "No duplicate business records on rerun" (brief) is satisfied
    because each run fully replaces Gold from Silver rather than merging into
    whatever was there before.
    """
    rows = list(rows)
    if not rows:
        return 0
    col_list = ", ".join(columns)
    with conn.cursor().copy(f"COPY {table} ({col_list}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(tuple(row[c] for c in columns))
    return len(rows)


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()

    result = StageResult(stage="load_postgres", run_id=run_id)
    counts: dict[str, int] = {}

    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.transaction():  # all-or-nothing: whole Gold load commits or none of it does
            conn.execute((SQL_DIR / "gold_schema.sql").read_text())

            # ---- qec_syndromes ------------------------------------------------
            syn_rows = _rows(SILVER / "qec_syndromes/syndrome_observation.parquet")
            experiments = {
                r["experiment_id"]: {
                    "experiment_id": r["experiment_id"],
                    "physical_fault_rate": r["physical_fault_rate"],
                    "round_count": r["round_count"],
                    "check_count": r["check_count"],
                }
                for r in syn_rows
            }
            counts["sim_experiment"] = _copy_load(
                conn, "gold.sim_experiment",
                ["experiment_id", "physical_fault_rate", "round_count", "check_count"],
                experiments.values(),
            )
            counts["syndrome_observation"] = _copy_load(
                conn, "gold.syndrome_observation",
                ["source_record_id", "experiment_id", "syndrome_bits",
                 "logical_error_label", "quantity"],
                syn_rows,
            )

            # ---- google_qec -----------------------------------------------------
            exp_rows = _rows(SILVER / "google_qec/experiment.parquet")
            counts["google_experiment"] = _copy_load(
                conn, "gold.google_experiment",
                ["experiment_id", "source_record_id", "basis", "distance", "rounds",
                 "shots", "center_row", "center_col", "measurement_count", "detector_count"],
                exp_rows,
            )

            shot_rows = _rows(SILVER / "google_qec/shot.parquet")
            counts["google_shot"] = _copy_load(
                conn, "gold.google_shot",
                ["source_record_id", "experiment_id", "shot_index", "measurement_bits",
                 "sweep_bits", "detector_bits", "detector_event_count",
                 "actual_observable_flip"],
                shot_rows,
            )

            prediction_rows = [
                {
                    "shot_source_record_id": r["source_record_id"],
                    "decoder_name": d,
                    "predicted_flip": r[f"{d}_prediction"],
                }
                for r in shot_rows
                for d in DECODER_NAMES
            ]
            counts["decoder_prediction"] = _copy_load(
                conn, "gold.decoder_prediction",
                ["shot_source_record_id", "decoder_name", "predicted_flip"],
                prediction_rows,
            )

            # ---- qasmbench -------------------------------------------------------
            circuit_rows = _rows(SILVER / "qasmbench/circuit.parquet")
            counts["circuit"] = _copy_load(
                conn, "gold.circuit",
                ["circuit_id", "source_record_id", "benchmark_name", "variant",
                 "register_declarations", "qubit_count", "measurement_count",
                 "two_qubit_gate_count"],
                circuit_rows,
            )
            sc_rows = _rows(SILVER / "qasmbench/stabilizer_check.parquet")
            counts["stabilizer_check"] = _copy_load(
                conn, "gold.stabilizer_check",
                ["source_record_id", "circuit_id", "check_id", "ancilla_qubit",
                 "data_qubits", "syndrome_bit"],
                sc_rows,
            )
            cc_rows = _rows(SILVER / "qasmbench/conditional_correction.parquet")
            counts["conditional_correction"] = _copy_load(
                conn, "gold.conditional_correction",
                ["source_record_id", "circuit_id", "condition_register",
                 "condition_value", "gate", "target_qubit"],
                cc_rows,
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    row_counts_path = RESULTS / "row_counts.json"
    existing = json.loads(row_counts_path.read_text()) if row_counts_path.exists() else {}
    existing["silver_to_gold"] = counts
    row_counts_path.write_text(json.dumps(existing, indent=2, sort_keys=True))

    result.input_count = len(syn_rows) + len(exp_rows) + len(shot_rows) + len(circuit_rows) \
                         + len(sc_rows) + len(cc_rows)
    result.output_count = sum(counts.values())
    result.issue_count = 0
    result.finish()
    return result

