"""Tests for Stage 2: prepare_data and source lineage tracing."""

from pathlib import Path
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.stages import prepare_data
from quantum_lake_student.tracing import SOURCE_TRACE_SCHEMA


def test_prepare_data_stage_execution():
    """Verify that Stage 2 runs end-to-end, producing verified Silver tables and traces."""
    res = prepare_data.run("test_verification_run")

    assert res.stage == "prepare_data"
    assert res.finished_at is not None
    assert res.finished_at >= res.started_at
    assert res.input_count >= 325603
    assert res.output_count >= 325603
    assert res.issue_count == 0

    base_dir = Path(__file__).resolve().parents[1]
    silver_dir = base_dir / "silver"
    results_dir = base_dir / "results/part1"

    # 1. Syndrome observations contract & invariants
    syn_file = silver_dir / "qec_syndromes/syndrome_observation.parquet"
    assert syn_file.exists(), "Syndrome Silver table missing"
    t_syn = pq.read_table(syn_file)
    df_syn = t_syn.to_pandas()
    assert len(df_syn) == 75598
    assert df_syn["quantity"].sum() == 70_000_000
    assert df_syn["source_record_id"].is_unique
    assert (df_syn["round_count"] == 4).all()
    assert (df_syn["check_count"] == 4).all()
    assert (df_syn["syndrome_bits"].str.len() == 16).all()
    assert df_syn.isna().sum().sum() == 0

    # 2. Google experiments contract & invariants
    exp_file = silver_dir / "google_qec/experiment.parquet"
    assert exp_file.exists(), "Google experiment Silver table missing"
    t_exp = pq.read_table(exp_file)
    df_exp = t_exp.to_pandas()
    assert len(df_exp) == 5
    assert df_exp["experiment_id"].is_unique
    assert df_exp["source_record_id"].is_unique
    assert (df_exp["basis"] == "X").all()
    assert (df_exp["rounds"] == 25).all()
    assert (df_exp["shots"] == 50000).all()
    assert (df_exp["distance"].isin([3, 5])).all()

    # 3. Google shots contract & invariants
    shot_file = silver_dir / "google_qec/shot.parquet"
    assert shot_file.exists(), "Google shot Silver table missing"
    t_shot = pq.read_table(shot_file)
    df_shot = t_shot.to_pandas()
    assert len(df_shot) == 250000
    assert df_shot["source_record_id"].is_unique
    assert df_shot.isna().sum().sum() == 0
    assert (df_shot["detector_event_count"] >= 0).all()

    # 4. Source trace contract, valid column, and companion-file coverage
    trace_file = results_dir / "source_trace.parquet"
    assert trace_file.exists(), "source_trace.parquet missing"
    t_trace = pq.read_table(trace_file)
    assert t_trace.schema == SOURCE_TRACE_SCHEMA
    df_trace = t_trace.to_pandas()
    assert len(df_trace) == 2075603
    assert set(df_trace["valid"].unique()) == {"yes"}
    assert df_trace["source_record_id"].nunique() == 325603

    # Composite primary key uniqueness in trace table
    assert not df_trace.duplicated(
        subset=["source_record_id", "archive_member", "record_locator"]
    ).any()

    # 5. Data issues schema validation
    issues_file = results_dir / "data_issues.parquet"
    assert issues_file.exists(), "data_issues.parquet missing"
    t_issues = pq.read_table(issues_file)
    assert t_issues.num_rows == 0
    assert "rule_id" in t_issues.column_names
    assert "severity" in t_issues.column_names
