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


def _pct(value: float) -> str:
    return f"{100 * value:.1f} %"


def _decoder_label(model_id: str, value: dict[str, Any]) -> str:
    column = value.get("source_column")
    return f"{model_id} ({column.removesuffix('_prediction')})" if column else model_id


def _report_markdown(
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    input_counts: dict[str, int],
) -> str:
    """Part II report, generated from this run's numbers.
    """
    prior_a, logistic_a = metrics["task_a_prior"], metrics["task_a_logistic"]
    share = diagnostics["task_a"]["positive_weight_share"]
    mlp = metrics["task_c_raw_mlp"]
    task_c = diagnostics["task_c"]

    lines = [
        "# Part II — AI/ML results",
        "",
        "## Data and prediction targets",
        "",
        f"- **Syndrome decoder dataset:** Contains {input_counts['syndrome']:,} rows. "
        "Each row contains 16 syndrome bits collected over four rounds and four checks. "
        "The models use these bits to predict whether a logical error occurred "
        "(`logical_error_label`). Each row represents a number of simulated shots "
        "given by `sample_weight`, covering 70 million shots in total.",
        "",
        f"- **Google decoder dataset:** Contains {input_counts['google']:,} rows, "
        "with one hardware shot per row. The available inputs include the number "
        "of detector events, the packed detector bits and predictions from four "
        "existing decoders. The goal is to predict whether the observable flipped "
        "(`actual_observable_flip`).",
        "",
        "All models use the original train, validation and test splits. "
        "The training data is used to fit the models, the validation data is used "
        "to select prediction thresholds, and the final results are measured on "
        "the test data. Models within the same task are evaluated on the same test rows, "
        "so their results can be compared directly.",
        "",
        "## Why these models were chosen",
        "",
        "- **Task A — Syndrome decoder:** The data is already grouped and weighted, "
        "so a weighted baseline provides a straightforward starting point. "
        "Logistic regression can also use the sample weights directly, avoiding "
        "the need to expand the dataset into 70 million individual observations. "
        "With only 16 binary inputs, a linear model is a reasonable first choice.",
        "",
        "- **Task B — Google decoders:** Four existing decoders already provide "
        "predictions. Logistic regression is used to combine those predictions "
        "with the detector-event density and learn when each signal is useful. "
        "The distance-3 and distance-5 datasets are modelled separately because "
        "they represent different codes.",
        "",
        "- **Task C — Raw detector data:** A small multilayer perceptron (MLP) "
        "is used to look for non-linear patterns in the 200 raw detector bits. "
        "This provides a simple starting point for testing whether the raw inputs "
        "contain patterns that can help predict logical errors.",
        "",
        "## Task A — Weighted syndrome decoder",
        "",
        _metric_table(metrics, prefix="task_a_"),
        "",
        "**Understanding the class imbalance.** Logical errors are uncommon in "
        "this dataset. They account for "
        f"{_pct(share['train'])} of the total sample weight in the training set, "
        f"{_pct(share['validation'])} in the validation set and "
        f"{_pct(share['test'])} in the test set. The simulation uses a fault rate "
        "of 0.005.",
        "",
        f"The weighted prior is {prior_a['positive_prior']:.4f}, which is below "
        "0.5. As a result, the baseline predicts that every shot has no logical "
        "error. Its logical-error rate is therefore "
        f"{prior_a['logical_error_rate']:.3f}, while its balanced accuracy is "
        "0.5. This gives us a simple reference point for judging whether the "
        "logistic regression model provides a useful improvement.",
        "",
    ]
    if logistic_a["logical_error_rate"] > prior_a["logical_error_rate"]:
        lines.append(
            "**Results.** The logistic regression model uses a threshold of "
            f"{logistic_a['threshold']:.3f}, selected on the validation set to "
            "maximise balanced accuracy. This threshold helps the model identify "
            "more of the actual logical errors, but it also causes more incorrect "
            "predictions.",
            )
        lines.append(
            f"Its balanced accuracy is {logistic_a['balanced_accuracy']:.3f}, "
            f"but its logical-error rate is {logistic_a['logical_error_rate']:.3f}, "
            f"compared with {prior_a['logical_error_rate']:.3f} for the baseline. "
            "In other words, the model is better at distinguishing between the "
            "two classes according to balanced accuracy, but this does not translate "
            "into a lower overall logical-error rate.",
        )
        lines.append(
            f"The Brier score is {logistic_a['brier_score']:.3f} for logistic "
            f"regression and {prior_a['brier_score']:.3f} for the baseline. "
            f"The logistic model's probabilities are "
            f"{'better' if logistic_a['brier_score'] < prior_a['brier_score'] else 'not better'} "
            "according to this measure. Which model is more useful depends on "
            "the practical cost of missing a real error compared with raising "
            "a false alarm."
        )
    else:
        lines.append(
            "**Results.** The logistic regression model uses a threshold of "
            f"{logistic_a['threshold']:.3f}, selected on the validation set. "
            f"It achieves a logical-error rate of {logistic_a['logical_error_rate']:.3f}, "
            f"compared with {prior_a['logical_error_rate']:.3f} for the baseline. "
            f"Its balanced accuracy is {logistic_a['balanced_accuracy']:.3f}. "
            "These results show how the model performs relative to the simple "
            "baseline on the test data."
        )
    lines += ["",
        "## Task B — Comparing the Google decoders",
        "",
        "This task evaluates the four supplied decoders and tests whether their "
        "predictions can be combined to improve performance. The results are "
        "reported separately for distances 3 and 5.",
        "",]

    for distance in (3, 5):
        prefix = f"task_b_d{distance}_"
        decoders = {k: v for k, v in metrics.items() if k.startswith(f"{prefix}decoder_")}
        best_id = min(decoders, key=lambda k: decoders[k]["logical_error_rate"])
        best = decoders[best_id]
        combined = metrics[f"{prefix}combined"]
        overlap = diagnostics[f"task_b_d{distance}"]["decoder_error_overlap"]
        jaccards = [value["jaccard_error_overlap"] for value in overlap.values()]
        difference = combined["logical_error_rate"] - best["logical_error_rate"]
        lines += [
            f"### Distance {distance}",
            "",
            _metric_table(metrics, prefix=prefix),
            "",
            "The table below shows how often pairs of decoders make the same "
            "mistakes on the test set. The shared-error count gives the number "
            "of errors made by both decoders, while the Jaccard overlap measures "
            "how similar their sets of errors are.",
            "",
            "| Decoder pair (test errors) | Shared errors | Jaccard overlap |",
            "|---|---:|---:|",
            *[
                f"| {pair.replace('_prediction', '').replace('__', ' / ')} | "
                f"{value['shared_errors']:,} | "
                f"{value['jaccard_error_overlap']:.3f} |"
                for pair, value in overlap.items()
            ],
            "",
            f"The baseline logical-error rate is "
            f"{metrics[f'{prefix}prior']['logical_error_rate']:.3f}, "
            "which is close to 0.5. Over 25 rounds, the observable flips in "
            "approximately half of the shots.",
            "",
            f"Among the supplied decoders, {_decoder_label(best_id, best)} "
            f"performs best, with a logical-error rate of "
            f"{best['logical_error_rate']:.3f}. The overlap results show that "
            f"decoder pairs share between {min(jaccards):.2f} and "
            f"{max(jaccards):.2f} of their errors according to the Jaccard "
            "measure. This means that the decoders sometimes make different "
            "mistakes, but they also get many of the same shots wrong.",
            "",
            f"The combined model has a logical-error rate of "
            f"{combined['logical_error_rate']:.3f}, which is "
            f"{abs(difference):.3f} "
            f"{'lower' if difference < 0 else 'higher' if difference > 0 else 'the same as'} "
            "the rate of the best individual decoder. The shared mistakes help "
            "explain why combining the predictions may not provide a large "
            "improvement. The combined model's threshold was selected using "
            "the validation set, not the test set.",
            "",
        ]

    lines += [
        "## Task C — Testing a model on raw detector data",
        "",
        "This experiment uses a limited subset of the raw detector data: only "
        "`distance = 3` and shots with `shot_index < 12_500` are included. "
        f"This gives {task_c['subset_rows']:,} shots in total, split into "
        f"{task_c['subset_rows_by_split']['train']:,} training shots, "
        f"{task_c['subset_rows_by_split']['validation']:,} validation shots "
        f"and {task_c['subset_rows_by_split']['test']:,} test shots. "
        "Each shot contains 200 detector bits, unpacked using the supplied "
        "little-endian helper.",
        "",
        _metric_table(metrics, prefix="task_c_"),
        "",
        "**Results.** The MLP "
        f"{'converged' if task_c['mlp_converged'] else 'did not converge'} "
        f"after {task_c['mlp_iterations']} iterations. Early stopping was used "
        "to monitor performance on 10% of the training rows and help prevent "
        "the model from fitting the training data too closely.",
        "",
        f"The model achieves a balanced accuracy of "
        f"{task_c['train_balanced_accuracy']:.3f} on the training data, "
        f"compared with {mlp['balanced_accuracy']:.3f} on the test data. "
        + (
            "This test result is close to chance level. "
            if abs(mlp["balanced_accuracy"] - 0.5) < 0.02
            else ""
        )
        + "The gap between training and test performance suggests that the "
        "model has not learned a pattern that generalises well to unseen shots. "
        "Without early stopping, it fits the training data more closely, "
        "but its test performance remains around chance level.",
        "",
        "**Why the detector layout matters.** Although the input contains "
        "200 bits representing 8 detectors across 25 rounds, the model receives "
        "them as a flat vector. It is not explicitly told which bit corresponds "
        "to each detector or round, which detectors are neighbours on the chip, "
        "or which detector events tend to occur near each other in space and time.",
        "",
        "These relationships can be important when decoding errors. A fully "
        "connected MLP must learn them from the training examples on its own. "
        "Given the limited size of this experiment, the model may not have enough "
        "information to discover these patterns reliably.",
        "",
        "## Information not used by the models",
        "",
        "- **Task A:** The model uses only the 16 syndrome bits. The fault rate "
        "is used to define the data split rather than as a prediction feature, "
        "and the original four-round, four-check structure is flattened into "
        "a single sequence of bits.",
        "",
        "- **Task B:** The combined model uses the overall proportion of fired "
        "detectors as its raw-data feature, along with the supplied decoder "
        "predictions. It does not use the identities of the fired detectors, "
        "their locations on the processor, or the measurement and sweep bits.",
        "",
        "- **Task C:** The model uses raw detector bits, but only for distance 3 "
        "and the first 12,500 shots of each experiment. The flat input also "
        "does not explicitly preserve the spatial and temporal relationships "
        "between detectors.",
        "",
        "## Limitations",
        "",
        "There are several limitations to keep in mind when interpreting these "
        "results:",
        "",
        "- Each table uses a single fixed "
        "split. Repeated runs and confidence intervals are not available, so "
        "small differences between models may not reflect a reliable improvement.",
        "",
        "- Thresholds are selected to maximise balanced "
        "accuracy. A different threshold could change the balance between "
        "missing real errors and raising false alarms.",
        "",
        "- The experiments use linear models and "
        "one small MLP, with no extensive hyperparameter tuning beyond selecting "
        "the prediction thresholds on validation data.",
        "",
        "- Training and prediction times depend on the "
        "machine and the conditions under which the experiments were run. "
        "They should therefore be treated as approximate rather than universal "
        "performance figures.",
        "",
    ]
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
            f"| {_decoder_label(model_id, value)} | {fmt(value.get('logical_error_rate'))} | "
            f"{fmt(value.get('balanced_accuracy'))} | {fmt(value.get('brier_score'))} | "
            f"{fmt(value.get('training_time_seconds'))} | {fmt(value.get('prediction_time_seconds'))} |"
        )
    return "\n".join(lines)
