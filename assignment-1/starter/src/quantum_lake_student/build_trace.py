"""Part I pipeline stage: results/part1/trace_examples.json (R5).

Demonstrates the complete trace for one syndrome prediction and one Google
prediction, per brief.md's "Source tracing" section:

    ML example_id -> Gold -> source_record_id -> source_trace.parquet row(s)
    -> Bronze object / archive member / record locator / input_sha256

The Google example additionally lists all 8 companion Bronze files for that
shot (measurements.b8, sweep.b8, detection_events.b8, obs_flips_actual.01,
and the 4 obs_flips_predicted_by_*.01 files) -- one shot's record is
assembled from several aligned companion members that all share one
source_record_id (silver-tables.md).

Run this after load_postgres (Gold, including sql/ml_syndrome_example.sql
and sql/ml_google_example.sql, must already be loaded). It is part of the
ordinary pipeline, not a notebook: call `run(run_id, settings)` from
wherever the other stages (load_postgres, etc.) are orchestrated, or run
this file directly for a one-off build.

Idempotent / self-healing: if the chosen Google shot's 8 companion-file rows
are not yet in source_trace.parquet (e.g. the Silver google_qec stage only
wrote experiment-level trace rows so far), this stage adds exactly those 8
rows itself via the supplied save_source_traces helper (dedup'd by
(source_record_id, archive_member, record_locator), so reruns never
duplicate them) and then proceeds. It does not touch any other shot's trace
rows -- backfilling the other 249,999 shots is the Silver stage's job, not
this one's.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import psycopg
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import StageResult
from quantum_lake_student.tracing import save_source_traces

PROJECT = Path(__file__).resolve().parents[3]  # starter/
RESULTS = PROJECT / "results/part1"
TRACE_LOCAL = RESULTS / "source_trace.parquet"
TRACE_KEY = "results/part1/source_trace.parquet"  # object key inside the bucket

# Deterministic example picks, so trace_examples.json is stable across reruns:
#  - syndrome: row 0 of the lowest-fault-rate experiment file.
#  - google: surface_code_bX_d3_r25_center_3_5 shot 0 -- the exact shot
#    docs/decision_log.md documents as a verified, checked, non-trivial
#    example (27 fired detectors, actual_observable_flip=1).
SYNDROME_SOURCE_RECORD_ID = "qec_syndromes:d-3_pfr-0.000010_nb-10M.csv:row:0"
GOOGLE_EXPERIMENT_ID = "surface_code_bX_d3_r25_center_3_5"
GOOGLE_SHOT_INDEX = 0
GOOGLE_SOURCE_RECORD_ID = f"google_qec:{GOOGLE_EXPERIMENT_ID}:shot:{GOOGLE_SHOT_INDEX}"

GOOGLE_BRONZE_OBJECT = "bronze/source=google_qec/google-surface-code-curated.zip"
GOOGLE_COMPANION_FILES = [
    "measurements.b8",
    "sweep.b8",
    "detection_events.b8",
    "obs_flips_actual.01",
    "obs_flips_predicted_by_belief_matching.01",
    "obs_flips_predicted_by_correlated_matching.01",
    "obs_flips_predicted_by_pymatching.01",
    "obs_flips_predicted_by_tensor_network_contraction.01",
]


def _read_trace_table(settings: Settings):
    """Read source_trace.parquet from wherever the Silver stage put it."""
    if settings.lake_backend == "minio":
        client = minio_client(settings)
        response = client.get_object(settings.s3_bucket, TRACE_KEY)
        try:
            return pq.read_table(io.BytesIO(response.read())).to_pandas()
        finally:
            response.close()
            response.release_conn()
    if TRACE_LOCAL.exists():
        return pq.read_table(TRACE_LOCAL).to_pandas()
    raise FileNotFoundError(
        f"{TRACE_LOCAL} not found. Run the Silver stage(s) first -- they "
        "must write results/part1/source_trace.parquet before this stage can run."
    )


def _trace_rows_for(trace_df, source_record_id: str) -> list[dict[str, Any]]:
    rows = trace_df[trace_df["source_record_id"] == source_record_id]
    return [
        {
            "source_record_id": r.source_record_id,
            "source_name": r.source_name,
            "bronze_object": r.bronze_object,
            "archive_member": r.archive_member,
            "record_locator": r.record_locator,
            "input_sha256": r.input_sha256,
        }
        for r in rows.itertuples()
    ]


def _ensure_google_shot_trace(trace_df, settings: Settings):
    """Add the 8 companion-file trace rows for the one demo shot if missing."""
    existing = trace_df[trace_df["source_record_id"] == GOOGLE_SOURCE_RECORD_ID]
    if len(existing) >= len(GOOGLE_COMPANION_FILES):
        return trace_df

    google_rows = trace_df[trace_df["source_name"] == "google_qec"]
    if google_rows.empty:
        raise RuntimeError(
            "No google_qec rows in source_trace.parquet yet -- run the "
            "Silver google_qec stage first (it must trace at least the "
            "experiment-level properties.yml records)."
        )
    sha256 = str(google_rows.iloc[0]["input_sha256"])

    new_records = [
        {
            "source_record_id": GOOGLE_SOURCE_RECORD_ID,
            "source_name": "google_qec",
            "bronze_object": GOOGLE_BRONZE_OBJECT,
            "archive_member": f"{GOOGLE_EXPERIMENT_ID}/{fname}",
            "record_locator": str(GOOGLE_SHOT_INDEX),
            "input_sha256": sha256,
        }
        for fname in GOOGLE_COMPANION_FILES
    ]
    save_source_traces(
        new_records,
        source_name="google_qec",
        trace_file_path=TRACE_LOCAL,
        settings=settings,
    )
    return pq.read_table(TRACE_LOCAL).to_pandas()


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()

    result = StageResult(stage="build_trace_examples", run_id=run_id)

    trace_df = _read_trace_table(settings)
    trace_df = _ensure_google_shot_trace(trace_df, settings)

    with psycopg.connect(settings.postgres_dsn) as conn:
        syn_row = conn.execute(
            """
            SELECT example_id,
                   experiment_id,
                   physical_fault_rate,
                   logical_error_label,
                   sample_weight,
                   source_record_id
            FROM gold.v_ml_syndrome_decoder_example
            WHERE source_record_id = %s
            """,
            (SYNDROME_SOURCE_RECORD_ID,),
        ).fetchone()
        if syn_row is None:
            raise RuntimeError(
                f"{SYNDROME_SOURCE_RECORD_ID} not found in "
                "gold.v_ml_syndrome_decoder_example -- run load_postgres and "
                "sql/ml_syndrome_example.sql first."
            )
        syn_example = dict(zip(
            ["example_id", "experiment_id", "physical_fault_rate",
             "logical_error_label", "sample_weight", "source_record_id"],
            syn_row,
        ))

        google_row = conn.execute(
            """
            SELECT example_id,
                   experiment_id,
                   shot_index,
                   distance,
                   detector_event_count,
                   actual_observable_flip,
                   source_record_id
            FROM gold.v_ml_google_decoder_example
            WHERE source_record_id = %s
            """,
            (GOOGLE_SOURCE_RECORD_ID,),
        ).fetchone()
        if google_row is None:
            raise RuntimeError(
                f"{GOOGLE_SOURCE_RECORD_ID} not found in "
                "gold.v_ml_google_decoder_example -- run load_postgres and "
                "sql/ml_google_example.sql first."
            )
        google_example = dict(zip(
            ["example_id", "experiment_id", "shot_index", "distance",
             "detector_event_count", "actual_observable_flip",
             "source_record_id"],
            google_row,
        ))

    syn_trace = _trace_rows_for(trace_df, SYNDROME_SOURCE_RECORD_ID)
    google_trace = _trace_rows_for(trace_df, GOOGLE_SOURCE_RECORD_ID)

    if not syn_trace:
        raise RuntimeError(f"{SYNDROME_SOURCE_RECORD_ID} has no source_trace.parquet row.")
    if len(google_trace) != len(GOOGLE_COMPANION_FILES):
        raise RuntimeError(
            f"expected {len(GOOGLE_COMPANION_FILES)} companion-file trace rows "
            f"for {GOOGLE_SOURCE_RECORD_ID}, found {len(google_trace)}."
        )

    output = {
        "syndrome_prediction_trace": {
            "description": (
                "example_id -> gold.syndrome_observation (via source_record_id) "
                "-> source_trace.parquet -> Bronze qec_syndromes CSV."
            ),
            "ml_example": syn_example,
            "gold": {
                "table": "gold.syndrome_observation",
                "source_record_id": SYNDROME_SOURCE_RECORD_ID,
            },
            "source_trace_rows": syn_trace,
        },
        "google_prediction_trace": {
            "description": (
                "example_id -> gold.google_shot (via source_record_id) -> "
                "source_trace.parquet -> all 8 aligned Bronze companion "
                "members for this shot (one source_trace row per companion "
                "file, sharing one source_record_id)."
            ),
            "ml_example": google_example,
            "gold": {
                "table": "gold.google_shot",
                "source_record_id": GOOGLE_SOURCE_RECORD_ID,
            },
            "source_trace_rows": google_trace,
            "companion_file_count": len(google_trace),
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "trace_examples.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))

    if settings.lake_backend == "minio":
        try:
            minio_client(settings).fput_object(
                settings.s3_bucket, "results/part1/trace_examples.json", str(out_path),
            )
        except Exception:
            pass

    result.input_count = 2
    result.output_count = len(syn_trace) + len(google_trace)
    result.issue_count = 0
    result.finish()
    return result


# if __name__ == "__main__":
#     r = run("manual-build-trace-examples")
#     print(f"wrote results/part1/trace_examples.json "
#           f"({r.output_count} trace rows across {r.input_count} examples)")
