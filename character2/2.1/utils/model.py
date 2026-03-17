"""Closed-form and iterative polynomial fitting models used by the demo."""

from __future__ import annotations

import numpy as np


def fit_ordinary_least_squares(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float | str]:
    """Fit w by minimizing ||Xw - y||^2."""

    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "weights": weights,
        "method": "ordinary_least_squares",
        "condition_number": float(np.linalg.cond(X)),
    }


def fit_l2_regularized(
    X: np.ndarray,
    y: np.ndarray,
    lambda_value: float,
    *,
    regularize_bias: bool = False,
) -> dict[str, np.ndarray | float | str]:
    """Fit w by minimizing ||Xw - y||^2 + lambda * ||w||^2."""

    n_features = X.shape[1]
    identity = np.eye(n_features)
    if not regularize_bias:
        identity[0, 0] = 0.0

    gram_matrix = X.T @ X
    rhs = X.T @ y
    stabilized_matrix = gram_matrix + lambda_value * identity

    try:
        weights = np.linalg.solve(stabilized_matrix, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(stabilized_matrix) @ rhs

    return {
        "weights": weights,
        "method": "l2_regularized",
        "condition_number": float(np.linalg.cond(stabilized_matrix)),
    }


def _soft_threshold(value: float, threshold: float) -> float:
    """Apply the soft-threshold operator used by L1 optimization."""

    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_l1_regularized(
    X: np.ndarray,
    y: np.ndarray,
    lambda_value: float,
    *,
    regularize_bias: bool = False,
    max_iter: int = 5000,
    tolerance: float = 1e-6,
) -> dict[str, np.ndarray | float | str]:
    """
    Fit w by minimizing ||Xw - y||^2 + lambda * |w|.

    This uses coordinate descent so the algorithm stays transparent for teaching.
    """

    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=float)
    column_norms = np.sum(X * X, axis=0)
    column_norms = np.where(column_norms < 1e-12, 1.0, column_norms)

    for _ in range(max_iter):
        previous_weights = weights.copy()
        for feature_index in range(n_features):
            residual = y - X @ weights + X[:, feature_index] * weights[feature_index]
            rho = float(np.dot(X[:, feature_index], residual))

            if feature_index == 0 and not regularize_bias:
                weights[feature_index] = rho / column_norms[feature_index]
            else:
                weights[feature_index] = _soft_threshold(rho, lambda_value / 2.0) / column_norms[feature_index]

        if np.max(np.abs(weights - previous_weights)) < tolerance:
            break

    condition_number = float(np.linalg.cond(X))
    if n_samples < n_features:
        condition_number = float("inf")

    return {
        "weights": weights,
        "method": "l1_regularized",
        "condition_number": condition_number,
    }


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Generate predictions from a design matrix and weight vector."""

    return X @ weights
