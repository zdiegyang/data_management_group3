"""Shared evaluation utilities for Part II models."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

from .part2_contracts import (
    REQUIRED_SPLITS,
    SEED,
    TEST_SPLIT,
)

def _positive_prior(labels: np.ndarray, weights: np.ndarray | None = None) -> float:
    labels = np.asarray(labels, dtype=float)
    if weights is None:
        return float(np.mean(labels))
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("total sample weight must be positive")
    return float(np.sum(weights * labels) / total)


def _constant_predictions(
    prior: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.full(n, prior, dtype=float)
    prediction = probability >= 0.5
    return prediction.astype(bool), probability


def _select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Select a binary threshold using validation data only.

    Candidate thresholds are the validation probabilities plus the two class
    boundaries.  Ties are resolved by choosing the threshold closest to 0.5,
    making the result deterministic and conservative.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if probabilities.size == 0:
        raise ValueError("cannot select a threshold from empty validation data")

    candidates = np.unique(
        np.concatenate(([0.0, 0.5, 1.0], probabilities))
    )
    best_score = -np.inf
    best_threshold = 0.5
    best_distance = np.inf

    for threshold in candidates:
        predictions = probabilities >= threshold
        score = balanced_accuracy_score(labels, predictions, sample_weight=weights)
        distance = abs(float(threshold) - 0.5)
        if score > best_score + 1e-15 or (
            abs(score - best_score) <= 1e-15 and distance < best_distance
        ):
            best_score = float(score)
            best_threshold = float(threshold)
            best_distance = distance

    return best_threshold


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    weights: np.ndarray | None = None,
) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if weights is None:
        weights = np.ones(labels.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions have different shapes")
    if weights.shape != labels.shape:
        raise ValueError("weights and labels have different shapes")

    logical_error_rate = float(
        np.average((predictions != labels).astype(float), weights=weights)
    )
    balanced_accuracy = float(
        balanced_accuracy_score(labels, predictions, sample_weight=weights)
    )
    brier = None
    if probabilities is not None:
        brier = float(
            brier_score_loss(labels, np.asarray(probabilities, dtype=float), sample_weight=weights)
        )

    return {
        "logical_error_rate": logical_error_rate,
        "balanced_accuracy": balanced_accuracy,
        "brier_score": brier,
    }


def _prediction_records(
    rows: Sequence[dict[str, Any]],
    model_id: str,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> list[dict[str, Any]]:
    if len(rows) != len(predictions):
        raise ValueError(f"{model_id}: prediction length does not match rows")
    if probabilities is not None and len(rows) != len(probabilities):
        raise ValueError(f"{model_id}: probability length does not match rows")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        probability = None if probabilities is None else float(probabilities[index])
        records.append(
            {
                "example_id": str(row["example_id"]),
                "model_id": model_id,
                "label": bool(row["logical_error_label"] if "logical_error_label" in row else row["actual_observable_flip"]),
                "prediction": bool(predictions[index]),
                "probability": probability,
                "split": str(row["data_split"]),
            }
        )
    return records


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[LogisticRegression, float, float]:
    start = perf_counter()
    model = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
        solver="lbfgs",
    )
    if weights is None:
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train, sample_weight=weights)
    return model, perf_counter() - start, 0.0


def _evaluate_probabilistic_model(
    model: Any,
    rows_by_split: dict[str, list[dict[str, Any]]],
    feature_builder: Any,
    model_id: str,
    threshold: float,
    weight_column: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    for split in REQUIRED_SPLITS:
        rows = rows_by_split[split]
        x = np.asarray([feature_builder(row) for row in rows], dtype=float)
        y = np.asarray([bool(row["logical_error_label"] if "logical_error_label" in row else row["actual_observable_flip"]) for row in rows])
        weights = None if weight_column is None else np.asarray([float(row[weight_column]) for row in rows])

        start = perf_counter()
        probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        prediction = probability >= threshold
        prediction_time = perf_counter() - start

        predictions.extend(_prediction_records(rows, model_id, prediction, probability))
        if split == TEST_SPLIT:
            metrics = _metrics(y, prediction, probability, weights)
            metrics["threshold"] = float(threshold)
            metrics["prediction_time_seconds"] = float(prediction_time)
        else:
            # Prediction timing is not a test metric, but retain validation/training
            # diagnostics for the run record without using validation as final scoring.
            metrics.setdefault("prediction_time_by_split", {})[split] = float(prediction_time)

    return predictions, metrics
