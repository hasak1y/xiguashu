"""Metric helpers for classification and regression."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root mean squared error."""

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return binary classification accuracy."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return binary precision, handling empty positive predictions safely."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    if predicted_positive == 0:
        return 0.0
    return float(true_positive / predicted_positive)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return binary recall, handling empty positive labels safely."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    if actual_positive == 0:
        return 0.0
    return float(true_positive / actual_positive)


def evaluate_predictions(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a metric dictionary for one prediction result."""

    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MSE": mse,
            "RMSE": root_mean_squared_error(y_true, y_pred),
        }
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def primary_metric_name(task_type: str) -> str:
    """Return the metric mainly used to compare methods."""

    return "RMSE" if task_type == "regression" else "Accuracy"
