"""Feature engineering helpers for polynomial regression."""

from __future__ import annotations

import numpy as np


def standardize_feature(
    x: np.ndarray,
    *,
    enabled: bool,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """
    Standardize x to reduce numerical instability for high-degree powers.

    When enabled, z = (x - mean) / std.
    We keep mean/std so the same transformation can be reused for test data.
    """

    x = np.asarray(x, dtype=float)
    if not enabled:
        return x, {"enabled": False, "mean": 0.0, "std": 1.0}

    computed_mean = float(np.mean(x) if mean is None else mean)
    computed_std = float(np.std(x) if std is None else std)
    if computed_std < 1e-12:
        computed_std = 1.0

    x_scaled = (x - computed_mean) / computed_std
    return x_scaled, {"enabled": True, "mean": computed_mean, "std": computed_std}


def apply_standardization(x: np.ndarray, stats: dict[str, float | bool]) -> np.ndarray:
    """Apply previously fitted standardization stats."""

    x = np.asarray(x, dtype=float)
    if not bool(stats.get("enabled", False)):
        return x
    return (x - float(stats["mean"])) / float(stats["std"])


def build_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Build the polynomial design matrix [1, x, x^2, ..., x^degree].

    Each row corresponds to one sample.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    return np.vander(x, N=degree + 1, increasing=True)
