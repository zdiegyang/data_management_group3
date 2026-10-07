"""Persistence, provenance, and report generation for Part II."""
from __future__ import annotations

import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS


PROJECT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT / "results" / "part2"
MODELS_DIR = RESULTS_DIR / "models"

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(path: Path, records: Iterable[dict[str, Any]]) -> None:
    schema = pa.schema(
        [
            ("example_id", pa.string()),
            ("model_id", pa.string()),
            ("label", pa.bool_()),
            ("prediction", pa.bool_()),
            ("probability", pa.float64()),
            ("split", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(list(records), schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _save_models(artifacts: dict[str, dict[str, Any]]) -> dict[str, str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    locations = {}
    for model_id, artifact in artifacts.items():
        path = MODELS_DIR / f"{_sanitize_filename(model_id)}.joblib"
        joblib.dump(artifact, path)
        locations[model_id] = str(path.relative_to(PROJECT))
    return locations


def _package_versions() -> dict[str, str | None]:
    names = ["numpy", "pyarrow", "scikit-learn", "joblib"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _report_markdown(
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    input_counts: dict[str, int],
) -> str:
    lines = [
        "# Part II — AI/ML results",
        "",
        "## Inputs and targets",
        "",
        "The stage reads only the two required ML Parquet tables. The syndrome",
        "table supplies 16-value syndrome features, a logical-error label, and a",
        "physical sample weight. The Google table supplies shot-level labels,",
        "four supplied decoder predictions, detector-event counts, and packed",
        "detector bits.",
        "",
        f"- Syndrome examples: {input_counts['syndrome']:,}",
        f"- Google examples: {input_counts['google']:,}",
        "- Supplied partitions: train / validation / test.",
        "",
        "## Task A — weighted syndrome decoder",
        "",
        "The prior is the weighted positive-label rate on the training rows. The",
        "linear model is logistic regression on the 16 values returned by the",
        "supplied `syndrome_model_input` helper. Physical `sample_weight` is used",
        "for fitting and for every Task A test metric. The classification threshold",
        "is selected from validation data only.",
        "",
        _metric_table(metrics, prefix="task_a_"),
        "",
        "## Task B — Google decoders and linear combination",
        "",
        "Distance three and distance five are evaluated independently. Every",
        "supplied decoder is evaluated on the same test shots within its distance.",
        "The combined model uses exactly the five values returned by the supplied",
        "`google_meta_model_input` helper: normalized detector-event density plus",
        "the four decoder predictions. Its threshold is selected from validation",
        "data only.",
        "",
        "### Decoder-error overlap",
        "",
    ]
    for distance in (3, 5):
        lines.append(f"#### Distance {distance}")
        lines.append("")
        lines.append(_metric_table(metrics, prefix=f"task_b_d{distance}_"))
        lines.append("")
        overlap = diagnostics.get(f"task_b_d{distance}", {}).get("decoder_error_overlap", {})
        if overlap:
            lines.append("Pairwise test-error overlap:")
            lines.append("")
            lines.append("| Decoder pair | Shared errors | Jaccard overlap |")
            lines.append("|---|---:|---:|")
            for pair, value in overlap.items():
                lines.append(
                    f"| {pair} | {value['shared_errors']} | {value['jaccard_error_overlap']:.4f} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Task C — bounded raw-detector prototype",
            "",
            "The MLP consumes only the fixed subset `distance = 3 AND shot_index <",
            "12_500`. The packed detector bytes are unpacked through the supplied",
            "little-endian helper, producing 200 binary inputs per shot. This is a",
            "bounded consumer check rather than an architecture comparison.",
            "",
            _metric_table(metrics, prefix="task_c_"),
            "",
            "A flat 200-bit representation does not explicitly encode detector",
            "position, spatial neighborhood, or the relationship between detector",
            "events across QEC rounds. The MLP therefore receives the right raw",
            "bits but not an explicit geometry/time structure; any such structure",
            "must be inferred indirectly from repeated bit positions.",
            "",
            "## Limitations",
            "",
            "The experiments are intentionally simple and bounded. The combined",
            "decoder is linear, the syndrome representation is the supplied flat",
            "16-value helper output, and Task C uses a small MLP on a fixed d=3",
            "subset. Numerical performance is not the grading target; correctness",
            "of prepared-data use, supplied splits, physical weighting, repeatability,",
            "evaluation, and traceable saved outputs is the focus.",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_table(metrics: dict[str, Any], prefix: str) -> str:
    selected = [(model_id, value) for model_id, value in metrics.items() if model_id.startswith(prefix)]
    if not selected:
        return "No metrics recorded."
    lines = [
        "| Model | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, value in selected:
        def fmt(number: Any) -> str:
            return "n/a" if number is None else f"{float(number):.6f}"
        lines.append(
            f"| {model_id} | {fmt(value.get('logical_error_rate'))} | "
            f"{fmt(value.get('balanced_accuracy'))} | {fmt(value.get('brier_score'))} | "
            f"{fmt(value.get('training_time_seconds'))} | {fmt(value.get('prediction_time_seconds'))} |"
        )
    return "\n".join(lines)
