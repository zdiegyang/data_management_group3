"""Integration tests for Stage 3: Silver -> Gold PostgreSQL."""
from pathlib import Path

import psycopg
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.stages import load_postgres

GOLD_TABLE_KEYS = {
    "sim_experiment": ("experiment_id",),
    "syndrome_pattern": ("syndrome_bits",),
    "syndrome_observation": ("source_record_id",),
    "google_experiment": ("experiment_id",),
    "google_shot": ("source_record_id",),
    "decoder": ("decoder_name",),
    "decoder_prediction": ("shot_source_record_id", "decoder_name"),
    "benchmark": ("benchmark_name",),
    "circuit": ("circuit_id",),
    "stabilizer_check": ("source_record_id",),
    "check_data_qubit": ("check_source_record_id", "position"),
    "conditional_correction": ("source_record_id",),
}

def _gold_snapshot(settings: Settings) -> dict:
    """Return row counts and identifying keys for every Gold table."""

    snapshot = {}

    with postgres_connection(settings) as conn:
        for table, key_columns in GOLD_TABLE_KEYS.items():
            columns = ", ".join(key_columns)
            order_by = ", ".join(key_columns)

            rows = conn.execute(
                f"""
                SELECT {columns}
                FROM gold.{table}
                ORDER BY {order_by}
                """
            ).fetchall()

            keys = [tuple(row) for row in rows]

            snapshot[table] = {
                "rows": len(keys),
                "keys": keys,
            }

            assert len(keys) == len(set(keys)), (
                f"""gold.{table} contains duplicate key values."""
            )

    return snapshot

def test_load_postgres_is_repeatable():
    """Two Gold loads produce identical tables with no duplicate keys."""
    settings = Settings.from_environment()

    first_result = load_postgres.run("gold-repeatability-1")
    first = _gold_snapshot(settings)

    second_result = load_postgres.run("gold-repeatability-2")
    second = _gold_snapshot(settings)

    assert first_result.issue_count == 0
    assert second_result.issue_count == 0
    assert first == second


def _silver_row_count(base_dir: Path, relative_path: str) -> int:
    path = base_dir / "silver" / relative_path
    assert path.exists(), f"Silver table missing: {path}"
    return pq.read_table(path).num_rows

def _gold_row_count(settings: Settings, table: str) -> int:
    with postgres_connection(settings) as conn:
        return conn.execute(
            f"SELECT COUNT(*) FROM gold.{table}"
        ).fetchone()[0]


def test_gold_counts_match_silver():
    """Gold preserves the expected Silver row counts for relational tables."""
    settings = Settings.from_environment()
    base_dir = Path(__file__).resolve().parents[1]

    load_postgres.run("gold-count-check")

    expected_counts = {
        "syndrome_observation": (
            "qec_syndromes/syndrome_observation.parquet"
        ),
        "google_experiment": (
            "google_qec/experiment.parquet"
        ),
        "google_shot": (
            "google_qec/shot.parquet"
        ),
        "circuit": (
            "qasmbench/circuit.parquet"
        ),
        "stabilizer_check": (
            "qasmbench/stabilizer_check.parquet"
        ),
        "conditional_correction": (
            "qasmbench/conditional_correction.parquet"
        ),
    }

    for gold_table, silver_path in expected_counts.items():
        silver_count = _silver_row_count(base_dir, silver_path)
        gold_count = _gold_row_count(settings, gold_table)

        assert silver_count == gold_count, (
            f"gold.{gold_table}: expected {silver_count} rows from Silver, "
            f"found {gold_count}"
        )
