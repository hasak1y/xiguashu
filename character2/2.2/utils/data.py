"""Dataset generation helpers for evaluation method demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris, make_circles, make_moons


@dataclass(frozen=True)
class DatasetBundle:
    """Container that keeps the dataset and metadata together."""

    name: str
    task_type: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    description: str


def list_datasets() -> list[str]:
    """Return dataset names shown in the UI."""

    return [
        "二分类线性可分数据",
        "二分类非线性数据（moons）",
        "二分类非线性数据（circles）",
        "回归数据（sin(x)+noise）",
        "Iris 二分类子集",
    ]


def _generate_linear_classification(
    sample_count: int,
    noise_strength: float,
    random_seed: int,
) -> DatasetBundle:
    """Generate a linearly separable binary classification dataset."""

    rng = np.random.default_rng(random_seed)
    class_size = sample_count // 2
    remainder = sample_count - 2 * class_size

    center_distance = 2.2
    spread = 0.35 + 0.9 * noise_strength
    positive = rng.normal(
        loc=[center_distance, center_distance],
        scale=spread,
        size=(class_size + remainder, 2),
    )
    negative = rng.normal(
        loc=[-center_distance, -center_distance],
        scale=spread,
        size=(class_size, 2),
    )

    X = np.vstack([positive, negative])
    y = np.concatenate([np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)])

    return DatasetBundle(
        name="二分类线性可分数据",
        task_type="classification",
        X=X,
        y=y,
        feature_names=["x1", "x2"],
        target_name="类别",
        description="两团高斯点云大体可以被一条直线分开，适合演示逻辑回归在简单分类问题上的表现。",
    )


def _generate_moons(sample_count: int, noise_strength: float, random_seed: int) -> DatasetBundle:
    """Generate a two-moons classification dataset."""

    X, y = make_moons(n_samples=sample_count, noise=0.05 + 0.35 * noise_strength, random_state=random_seed)
    return DatasetBundle(
        name="二分类非线性数据（moons）",
        task_type="classification",
        X=X.astype(float),
        y=y.astype(int),
        feature_names=["x1", "x2"],
        target_name="类别",
        description="两个月牙形分布不能被一条直线很好分开，适合观察线性模型与非线性模型的差异。",
    )


def _generate_circles(sample_count: int, noise_strength: float, random_seed: int) -> DatasetBundle:
    """Generate a concentric-circles classification dataset."""

    X, y = make_circles(
        n_samples=sample_count,
        noise=0.02 + 0.2 * noise_strength,
        factor=0.45,
        random_state=random_seed,
    )
    return DatasetBundle(
        name="二分类非线性数据（circles）",
        task_type="classification",
        X=X.astype(float),
        y=y.astype(int),
        feature_names=["x1", "x2"],
        target_name="类别",
        description="同心圆分类任务是典型的非线性问题，单纯的线性边界通常难以给出好结果。",
    )


def _generate_sine_regression(sample_count: int, noise_strength: float, random_seed: int) -> DatasetBundle:
    """Generate a noisy one-dimensional sine regression dataset."""

    rng = np.random.default_rng(random_seed)
    X = np.linspace(0.0, 2.0 * np.pi, sample_count).reshape(-1, 1)
    signal = np.sin(X[:, 0])
    noise = rng.normal(loc=0.0, scale=0.05 + 0.45 * noise_strength, size=sample_count)
    y = signal + noise
    return DatasetBundle(
        name="回归数据（sin(x)+noise）",
        task_type="regression",
        X=X.astype(float),
        y=y.astype(float),
        feature_names=["x"],
        target_name="y",
        description="真实函数是 sin(x)，加入噪声后可以观察不同划分方法对回归误差估计的影响。",
    )


def _load_iris_binary(sample_count: int, noise_strength: float, random_seed: int) -> DatasetBundle:
    """Load a small binary subset of the iris dataset."""

    iris = load_iris()
    mask = iris.target < 2
    X = iris.data[mask][:, :2].astype(float)
    y = iris.target[mask].astype(int)

    rng = np.random.default_rng(random_seed)
    indices = rng.choice(len(X), size=min(sample_count, len(X)), replace=False)
    X = X[indices]
    y = y[indices]

    if noise_strength > 0:
        X = X + rng.normal(scale=0.15 * noise_strength, size=X.shape)

    return DatasetBundle(
        name="Iris 二分类子集",
        task_type="classification",
        X=X,
        y=y,
        feature_names=list(iris.feature_names[:2]),
        target_name="品种",
        description="使用 Iris 的前两类样本与前两个特征，数据量小，适合演示小样本下评估结果的不稳定性。",
    )


def generate_dataset(
    dataset_name: str,
    sample_count: int,
    noise_strength: float,
    random_seed: int,
    shuffle_data: bool,
) -> DatasetBundle:
    """Create one dataset according to the UI configuration."""

    builders = {
        "二分类线性可分数据": _generate_linear_classification,
        "二分类非线性数据（moons）": _generate_moons,
        "二分类非线性数据（circles）": _generate_circles,
        "回归数据（sin(x)+noise）": _generate_sine_regression,
        "Iris 二分类子集": _load_iris_binary,
    }
    dataset = builders[dataset_name](sample_count, noise_strength, random_seed)

    if shuffle_data:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(len(dataset.X))
        return DatasetBundle(
            name=dataset.name,
            task_type=dataset.task_type,
            X=dataset.X[indices],
            y=dataset.y[indices],
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            description=dataset.description,
        )
    return dataset
