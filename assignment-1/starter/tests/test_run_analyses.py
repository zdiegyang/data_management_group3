"""Tests for the Part I analyses (stage run_analyses, part of make run)."""

import re

import psycopg
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages import run_analyses as analyses

CHECKS = [
    {"variant": v, "ancilla_qubit": "a[0]", "data_qubits": "q[0], q[1]", "syndrome_bit": "syn[0]"}
    for v in ("source", "transpiled")
] + [
    {"variant": v, "ancilla_qubit": "a[1]", "data_qubits": "q[1], q[2]", "syndrome_bit": "syn[1]"}
    for v in ("source", "transpiled")
]


def _correction(variant, value, fired, target):
    return {"variant": variant, "condition_register": "syn", "condition_value": value,
            "fired_checks": fired, "silent_checks": None, "gate": "x", "target_qubit": target}


def _corrections(target_for_three="q[1]"):
    rows = []
    for variant in ("source", "transpiled"):
        rows += [
            _correction(variant, 1, "a[0] (q[0], q[1]) -> syn[0]", "q[0]"),
            _correction(variant, 2, "a[1] (q[1], q[2]) -> syn[1]", "q[2]"),
            _correction(variant, 3, "a[0] (q[0], q[1]) -> syn[0]; a[1] (q[1], q[2]) -> syn[1]",
                        target_for_three),
        ]
    return rows




def test_at_least_one_query_joins_three_or_more_gold_tables():
    # Brief: "At least one query must join three or more Gold tables."
    def gold_tables(sql):
        return set(re.findall(r"\bgold\.(\w+)", sql))
    widest = max(len(gold_tables(path.read_text())) for path in analyses.SQL_DIR.glob("*.sql"))
    assert widest >= 3



def _gold_is_loaded(settings):
    try:
        with psycopg.connect(settings.postgres_dsn) as conn:
            return conn.execute("SELECT to_regclass('gold.check_data_qubit') IS NOT NULL").fetchone()[0]
    except psycopg.Error:
        return False


def test_stage_writes_the_same_answers_on_every_run():
    settings = Settings.from_environment()
    if not _gold_is_loaded(settings):
        pytest.skip("Gold is not loaded; run `make run` first")

    analyses.run("analysis-test-1", settings)
    first = {p.name: p.read_text(encoding="utf-8") for p in analyses.OUTPUT_DIR.iterdir()}
    analyses.run("analysis-test-2", settings)
    second = {p.name: p.read_text(encoding="utf-8") for p in analyses.OUTPUT_DIR.iterdir()}

    assert first == second
    # The rejected relationship: no candidate key matches any circuit.
    assert all(line.endswith(",0") for line in second["q4_rejected_relationship.csv"].splitlines()[1:])
