"""Task A, B and C model implementations.

Each task consumes already-validated rows and returns prediction records,
final test metrics, and learned-model artifacts. Thresholds are selected from
validation data only.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    google_meta_model_input,
    syndrome_model_input,
    unpack_little_endian_bits,
)

from .part2_contracts import (
    REQUIRED_SPLITS,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    MLP_SUBSET_DISTANCE,
    MLP_SHOT_LIMIT,
    SEED,
    MlContractError,
    _split_rows,
)
from .part2_evaluation import (
    _positive_prior,
    _constant_predictions,
    _select_threshold,
    _metrics,
    _prediction_records,
    _fit_logistic,
)

def _run_task_a(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Weighted prior and weighted logistic regression on the 16 syndrome features."""
    by_split = {split: _split_rows(rows, split) for split in REQUIRED_SPLITS}
    train = by_split[TRAIN_SPLIT]
    validation = by_split[VALIDATION_SPLIT]
    test = by_split[TEST_SPLIT]

    y_train = np.asarray([bool(row["logical_error_label"]) for row in train])
    w_train = np.asarray([float(row["sample_weight"]) for row in train])
    prior = _positive_prior(y_train, w_train)
    prior_predictions: list[dict[str, Any]] = []
    prior_metrics: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        split_rows = by_split[split]
        y = np.asarray([bool(row["logical_error_label"]) for row in split_rows])
        weights = np.asarray([float(row["sample_weight"]) for row in split_rows])
        pred, prob = _constant_predictions(prior, len(split_rows))
        prior_predictions.extend(_prediction_records(split_rows, "task_a_prior", pred, prob))
        if split == TEST_SPLIT:
            prior_metrics = _metrics(y, pred, prob, weights)
    prior_metrics["positive_prior"] = prior
    prior_metrics["training_time_seconds"] = 0.0
    prior_metrics["prediction_time_seconds"] = None

    x_train = np.asarray([syndrome_model_input(row["syndrome_bits"]) for row in train], dtype=float)
    x_validation = np.asarray([syndrome_model_input(row["syndrome_bits"]) for row in validation], dtype=float)
    y_validation = np.asarray([bool(row["logical_error_label"]) for row in validation])
    w_validation = np.asarray([float(row["sample_weight"]) for row in validation])

    model, training_time, _ = _fit_logistic(x_train, y_train, w_train)
    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold = _select_threshold(y_validation, validation_probability, w_validation)

    logistic_predictions: list[dict[str, Any]] = []
    logistic_metrics: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        split_rows = by_split[split]
        x = np.asarray([syndrome_model_input(row["syndrome_bits"]) for row in split_rows], dtype=float)
        y = np.asarray([bool(row["logical_error_label"]) for row in split_rows])
        weights = np.asarray([float(row["sample_weight"]) for row in split_rows])
        start = perf_counter()
        probability = model.predict_proba(x)[:, 1]
        prediction = probability >= threshold
        prediction_time = perf_counter() - start
        logistic_predictions.extend(
            _prediction_records(split_rows, "task_a_logistic", prediction, probability)
        )
        if split == TEST_SPLIT:
            logistic_metrics = _metrics(y, prediction, probability, weights)
            logistic_metrics["threshold"] = float(threshold)
            logistic_metrics["training_time_seconds"] = float(training_time)
            logistic_metrics["prediction_time_seconds"] = float(prediction_time)

    model_artifact = {
        "model": model,
        "threshold": threshold,
        "feature_order": [f"syndrome_bit_{i}" for i in range(16)],
        "seed": SEED,
    }
    return (
        prior_predictions + logistic_predictions,
        {
            "task_a_prior": prior_metrics,
            "task_a_logistic": logistic_metrics,
        },
        {"task_a_logistic": model_artifact},
    )


def _run_task_b_distance(
    rows: list[dict[str, Any]], distance: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the required Google decoder comparison independently for one distance."""
    distance_rows = [row for row in rows if int(row["distance"]) == distance]
    if not distance_rows:
        raise MlContractError(f"No Google rows for distance {distance}")

    by_split = {split: _split_rows(distance_rows, split) for split in REQUIRED_SPLITS}
    if any(not by_split[split] for split in REQUIRED_SPLITS):
        raise MlContractError(f"Distance {distance}: one or more splits are empty")

    y_train = np.asarray([bool(row["actual_observable_flip"]) for row in by_split[TRAIN_SPLIT]])
    prior = _positive_prior(y_train)

    predictions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    training_times: dict[str, float] = {}
    prediction_times: dict[str, float] = {}

    prior_id = f"task_b_d{distance}_prior"
    for split in REQUIRED_SPLITS:
        split_rows = by_split[split]
        y = np.asarray([bool(row["actual_observable_flip"]) for row in split_rows])
        start = perf_counter()
        pred, prob = _constant_predictions(prior, len(split_rows))
        prediction_time = perf_counter() - start
        predictions.extend(_prediction_records(split_rows, prior_id, pred, prob))
        if split == TEST_SPLIT:
            metrics[prior_id] = _metrics(y, pred, prob)
            metrics[prior_id]["positive_prior"] = prior
            metrics[prior_id]["training_time_seconds"] = 0.0
            metrics[prior_id]["prediction_time_seconds"] = prediction_time

    # The four supplied decoders are evaluated on the exact same test rows.
    for index, column in enumerate(GOOGLE_META_PREDICTION_COLUMNS, start=1):
        model_id = f"task_b_d{distance}_decoder_{index}"
        for split in REQUIRED_SPLITS:
            split_rows = by_split[split]
            y = np.asarray([bool(row["actual_observable_flip"]) for row in split_rows])
            start = perf_counter()
            pred = np.asarray([bool(row[column]) for row in split_rows])
            prediction_time = perf_counter() - start
            predictions.extend(_prediction_records(split_rows, model_id, pred, None))
            if split == TEST_SPLIT:
                metrics[model_id] = _metrics(y, pred, None)
                metrics[model_id]["training_time_seconds"] = 0.0
                metrics[model_id]["prediction_time_seconds"] = prediction_time
                metrics[model_id]["source_column"] = column

    # Required five-feature helper: event density + four supplied predictions.
    x_by_split = {
        split: np.asarray([google_meta_model_input(row) for row in by_split[split]], dtype=float)
        for split in REQUIRED_SPLITS
    }
    y_train = np.asarray([bool(row["actual_observable_flip"]) for row in by_split[TRAIN_SPLIT]])
    y_validation = np.asarray([bool(row["actual_observable_flip"]) for row in by_split[VALIDATION_SPLIT]])
    x_train = x_by_split[TRAIN_SPLIT]
    x_validation = x_by_split[VALIDATION_SPLIT]

    model, training_time, _ = _fit_logistic(x_train, y_train, None)
    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold = _select_threshold(y_validation, validation_probability)

    combined_id = f"task_b_d{distance}_combined"
    for split in REQUIRED_SPLITS:
        split_rows = by_split[split]
        y = np.asarray([bool(row["actual_observable_flip"]) for row in split_rows])
        start = perf_counter()
        probability = model.predict_proba(x_by_split[split])[:, 1]
        prediction = probability >= threshold
        prediction_time = perf_counter() - start
        predictions.extend(_prediction_records(split_rows, combined_id, prediction, probability))
        if split == TEST_SPLIT:
            metrics[combined_id] = _metrics(y, prediction, probability)
            metrics[combined_id]["threshold"] = float(threshold)
            metrics[combined_id]["training_time_seconds"] = float(training_time)
            metrics[combined_id]["prediction_time_seconds"] = float(prediction_time)

    # Decoder-error overlap on the common test shots, for the report.
    test_rows = by_split[TEST_SPLIT]
    error_sets = {
        column: {
            str(row["example_id"])
            for row in test_rows
            if bool(row[column]) != bool(row["actual_observable_flip"])
        }
        for column in GOOGLE_META_PREDICTION_COLUMNS
    }
    overlap = {}
    columns = list(GOOGLE_META_PREDICTION_COLUMNS)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            union = error_sets[left] | error_sets[right]
            intersection = error_sets[left] & error_sets[right]
            overlap[f"{left}__{right}"] = {
                "left_errors": len(error_sets[left]),
                "right_errors": len(error_sets[right]),
                "shared_errors": len(intersection),
                "jaccard_error_overlap": (
                    len(intersection) / len(union) if union else 1.0
                ),
            }

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_order": [
            "normalized_detector_event_density",
            *GOOGLE_META_PREDICTION_COLUMNS,
        ],
        "seed": SEED,
        "distance": distance,
    }
    diagnostics = {"decoder_error_overlap": overlap}
    return predictions, metrics, {combined_id: artifact}, diagnostics


def _detector_features(row: dict[str, Any]) -> np.ndarray:
    """The 200 unpacked detector bits of one distance-3 shot, in detector order.

    Uses the supplied little-endian helper, which also drops byte padding.
    """
    values = np.asarray(
        unpack_little_endian_bits(row["detector_bits"], int(row["detector_count"])),
        dtype=float,
    )
    if values.shape != (200,):
        raise MlContractError(
            f"Task C: expected 200 detector bits, got {values.shape} for {row['example_id']}"
        )
    return values


def _run_task_c(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bounded d=3 MLP over exactly the 200 unpacked detector bits."""
    subset = [
        row
        for row in rows
        if int(row["distance"]) == MLP_SUBSET_DISTANCE
        and int(row["shot_index"]) < MLP_SHOT_LIMIT
    ]
    if not subset:
        raise MlContractError("Task C fixed subset is empty")

    by_split = {split: _split_rows(subset, split) for split in REQUIRED_SPLITS}
    if any(not by_split[split] for split in REQUIRED_SPLITS):
        raise MlContractError("Task C fixed subset does not contain all supplied splits")

    bits = _detector_features

    x_train = np.asarray([bits(row) for row in by_split[TRAIN_SPLIT]], dtype=float)
    y_train = np.asarray([bool(row["actual_observable_flip"]) for row in by_split[TRAIN_SPLIT]])
    x_validation = np.asarray([bits(row) for row in by_split[VALIDATION_SPLIT]], dtype=float)
    y_validation = np.asarray([bool(row["actual_observable_flip"]) for row in by_split[VALIDATION_SPLIT]])

    start = perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=SEED,
        early_stopping=False,
    )
    model.fit(x_train, y_train)
    training_time = perf_counter() - start

    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold = _select_threshold(y_validation, validation_probability)

    predictions: list[dict[str, Any]] = []
    test_metrics: dict[str, Any] = {}
    test_prediction_time = 0.0
    model_id = "task_c_raw_mlp"

    for split in REQUIRED_SPLITS:
        split_rows = by_split[split]
        x = np.asarray([bits(row) for row in split_rows], dtype=float)
        y = np.asarray([bool(row["actual_observable_flip"]) for row in split_rows])
        start = perf_counter()
        probability = model.predict_proba(x)[:, 1]
        prediction = probability >= threshold
        prediction_time = perf_counter() - start
        predictions.extend(_prediction_records(split_rows, model_id, prediction, probability))
        if split == TEST_SPLIT:
            test_prediction_time = prediction_time
            test_metrics = _metrics(y, prediction, probability)
            test_metrics["threshold"] = float(threshold)
            test_metrics["training_time_seconds"] = float(training_time)
            test_metrics["prediction_time_seconds"] = float(prediction_time)

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_order": [f"detector_bit_{i}" for i in range(200)],
        "seed": SEED,
        "subset": "distance = 3 AND shot_index < 12500",
    }
    diagnostics = {
        "subset_rows": len(subset),
        "subset_rows_by_split": {split: len(by_split[split]) for split in REQUIRED_SPLITS},
        "test_prediction_time_seconds": test_prediction_time,
    }
    return predictions, {model_id: test_metrics}, {model_id: artifact}, diagnostics
