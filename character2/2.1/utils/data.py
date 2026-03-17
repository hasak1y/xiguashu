"""Data generation helpers for the polynomial fitting demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    """Describe how to sample x and evaluate a teaching dataset."""

    label: str
    x_min: float
    x_max: float


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sin(x)": DatasetConfig(label="sin(x)", x_min=0.0, x_max=2.0 * np.pi),
    "cos(x)": DatasetConfig(label="cos(x)", x_min=0.0, x_max=2.0 * np.pi),
    "sin(x) + cos(x)": DatasetConfig(
        label="sin(x) + cos(x)", x_min=0.0, x_max=2.0 * np.pi
    ),
    "x^2": DatasetConfig(label="x^2", x_min=-3.0, x_max=3.0),
    "x^3 - x": DatasetConfig(label="x^3 - x", x_min=-3.0, x_max=3.0),
    "custom mix": DatasetConfig(label="custom mix", x_min=-3.0, x_max=3.0),
}


def list_dataset_names() -> list[str]:
    """Return supported dataset names in UI order."""

    return list(DATASET_CONFIGS.keys())


def evaluate_function(x: np.ndarray, dataset_name: str) -> np.ndarray:
    """Evaluate the selected ground-truth function on x."""

    if dataset_name == "sin(x)":
        return np.sin(x)
    if dataset_name == "cos(x)":
        return np.cos(x)
    if dataset_name == "sin(x) + cos(x)":
        return np.sin(x) + np.cos(x)
    if dataset_name == "x^2":
        return x**2
    if dataset_name == "x^3 - x":
        return x**3 - x
    if dataset_name == "custom mix":
        return 0.5 * np.sin(2.0 * x) + 0.3 * x**2 - 0.8 * x

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def sample_x(
    n_samples: int,
    dataset_name: str,
    rng: np.random.Generator,
    *,
    evenly_spaced: bool = False,
) -> np.ndarray:
    """Sample x values inside the configured range."""

    config = DATASET_CONFIGS[dataset_name]
    if evenly_spaced:
        x = np.linspace(config.x_min, config.x_max, n_samples)
    else:
        x = rng.uniform(config.x_min, config.x_max, size=n_samples)
    return np.sort(x)


def generate_dataset(
    dataset_name: str,
    train_size: int,
    test_size: int,
    noise_strength: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Generate train/test samples and a dense curve for plotting."""

    rng = np.random.default_rng(random_seed)

    x_train = sample_x(train_size, dataset_name, rng)
    y_train_true = evaluate_function(x_train, dataset_name)
    noise = rng.normal(loc=0.0, scale=noise_strength, size=train_size)
    y_train = y_train_true + noise

    x_test = sample_x(test_size, dataset_name, rng, evenly_spaced=True)
    y_test = evaluate_function(x_test, dataset_name)

    x_curve = np.linspace(
        DATASET_CONFIGS[dataset_name].x_min,
        DATASET_CONFIGS[dataset_name].x_max,
        400,
    )
    y_curve = evaluate_function(x_curve, dataset_name)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_true": y_train_true,
        "x_test": x_test,
        "y_test": y_test,
        "x_curve": x_curve,
        "y_curve": y_curve,
    }
