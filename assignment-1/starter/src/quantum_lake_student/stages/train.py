"""Required Part II model-training stage.

Consume the two required ML input tables produced by Part I through the
supplied model-input and partition helpers. Publish repeatable model files,
predictions, metrics, run settings, the exact feature order, and the concise
Part II report under ``results/part2/``. Numerical performance is not graded.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter

from quantum_lake_student.config import Settings
from quantum_lake_student.models import StageResult

from .part2_contracts import (
    GOOGLE_KEY,
    GOOGLE_SCHEMA,
    SYNDROME_KEY,
    SYNDROME_SCHEMA,
    _read_ml_table,
    _validate_google,
    _validate_syndrome,
)
from .part2_outputs import (
    RESULTS_DIR,
    _data_release_version,
    _git_revision,
    _code_sha256,
    _package_versions,
    _report_markdown,
    _save_models,
    _write_json,
    _write_predictions,
)
from .part2_tasks import _run_task_a, _run_task_b_distance, _run_task_c
from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS


def run(model_run_id: str, settings: Settings | None = None) -> StageResult:
    """Run all required Part II tasks and publish ``results/part2``."""
    settings = settings or Settings.from_environment()
    result = StageResult(stage="train", run_id=model_run_id)
    started_at = datetime.now(UTC)
    wall_start = perf_counter()

    syndrome_table, syndrome_raw, syndrome_hash = _read_ml_table(
        settings, SYNDROME_KEY, SYNDROME_SCHEMA
    )
    google_table, google_raw, google_hash = _read_ml_table(
        settings, GOOGLE_KEY, GOOGLE_SCHEMA
    )
    syndrome_rows = syndrome_table.to_pylist()
    google_rows = google_table.to_pylist()

    _validate_syndrome(syndrome_rows)
    _validate_google(google_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = {}
    all_artifacts = {}
    diagnostics = {}

    predictions, metrics, artifacts = _run_task_a(syndrome_rows)
    all_predictions.extend(predictions)
    all_metrics.update(metrics)
    all_artifacts.update(artifacts)

    for distance in (3, 5):
        predictions, metrics, artifacts, task_diagnostics = _run_task_b_distance(
            google_rows, distance
        )
        all_predictions.extend(predictions)
        all_metrics.update(metrics)
        all_artifacts.update(artifacts)
        diagnostics[f"task_b_d{distance}"] = task_diagnostics

    predictions, metrics, artifacts, task_diagnostics = _run_task_c(google_rows)
    all_predictions.extend(predictions)
    all_metrics.update(metrics)
    all_artifacts.update(artifacts)
    diagnostics["task_c"] = task_diagnostics

    model_locations = _save_models(all_artifacts)
    _write_predictions(RESULTS_DIR / "predictions.parquet", all_predictions)

    timings = {
        model_id: {
            "training_time_seconds": value.get("training_time_seconds"),
            "prediction_time_seconds": value.get("prediction_time_seconds"),
        }
        for model_id, value in all_metrics.items()
    }

    run_record = {
        "model_run_id": model_run_id,
        "stage": "train",
        "status": "succeeded",
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "elapsed_wall_time_seconds": perf_counter() - wall_start,
        "data_release_version": _data_release_version(settings),
        "input_tables": {
            SYNDROME_KEY: {"rows": len(syndrome_rows), "sha256": syndrome_hash, "bytes": len(syndrome_raw)},
            GOOGLE_KEY: {"rows": len(google_rows), "sha256": google_hash, "bytes": len(google_raw)},
        },
        "code_revision": _git_revision(),
        "code_sha256": _code_sha256(),
        "dependencies": _package_versions(),
        "random_seed": 20261006,
        "splits": {
            "syndrome": "supplied data_split from syndrome_data_split",
            "google": "supplied data_split from google_data_split",
            "task_c_subset": "distance = 3 AND shot_index < 12_500",
        },
        "feature_order": {
            "task_a_logistic": [f"syndrome_bit_{i}" for i in range(16)],
            "task_b_combined": ["normalized_detector_event_density", *GOOGLE_META_PREDICTION_COLUMNS],
            "task_c_raw_mlp": [f"detector_bit_{i}" for i in range(200)],
        },
        "model_files": model_locations,
        "timings": timings,
        "prediction_rows": len(all_predictions),
        "diagnostics": diagnostics,
    }

    _write_json(RESULTS_DIR / "metrics.json", {"metrics": all_metrics})
    _write_json(RESULTS_DIR / "run.json", run_record)
    (RESULTS_DIR / "report.md").write_text(
        _report_markdown(all_metrics, diagnostics, {"syndrome": len(syndrome_rows), "google": len(google_rows)}),
        encoding="utf-8",
    )

    result.input_count = len(syndrome_rows) + len(google_rows)
    result.output_count = len(all_predictions)
    result.issue_count = 0
    result.details.update({
        "input_tables": run_record["input_tables"],
        "model_files": model_locations,
        "metrics_file": str(RESULTS_DIR / "metrics.json"),
        "predictions_file": str(RESULTS_DIR / "predictions.parquet"),
        "timings": timings,
    })
    result.finish()
    return result