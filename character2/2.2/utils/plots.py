"""Plot helpers for Streamlit visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.models import build_model


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_split_visualization(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_name: str,
    degree: int,
    knn_k: int,
    use_l2_regularization: bool,
    regularization_strength: float,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    title: str,
):
    """Draw how the current round splits the dataset."""

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    if task_type == "classification":
        _plot_classification_split(
            ax=ax,
            X=X,
            y=y,
            train_indices=train_indices,
            valid_indices=valid_indices,
            model_name=model_name,
            degree=degree,
            knn_k=knn_k,
            use_l2_regularization=use_l2_regularization,
            regularization_strength=regularization_strength,
        )
    else:
        _plot_regression_split(
            ax=ax,
            X=X,
            y=y,
            train_indices=train_indices,
            valid_indices=valid_indices,
            model_name=model_name,
            degree=degree,
            knn_k=knn_k,
            use_l2_regularization=use_l2_regularization,
            regularization_strength=regularization_strength,
        )

    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def _plot_classification_split(
    ax,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    model_name: str,
    degree: int,
    knn_k: int,
    use_l2_regularization: bool,
    regularization_strength: float,
) -> None:
    """Plot the classification train/validation split with decision boundary."""

    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[train_indices] = True
    valid_mask = np.zeros(len(X), dtype=bool)
    valid_mask[valid_indices] = True

    model = build_model(
        model_name=model_name,
        task_type="classification",
        degree=degree,
        knn_k=knn_k,
        use_l2_regularization=use_l2_regularization,
        regularization_strength=regularization_strength,
    )
    model.fit(X[train_indices], y[train_indices])

    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.15, cmap="coolwarm")

    ax.scatter(
        X[train_mask, 0],
        X[train_mask, 1],
        c=y[train_mask],
        cmap="coolwarm",
        edgecolor="black",
        s=70,
        label="训练样本",
    )
    if np.any(valid_mask):
        ax.scatter(
            X[valid_mask, 0],
            X[valid_mask, 1],
            c=y[valid_mask],
            cmap="coolwarm",
            marker="^",
            edgecolor="black",
            s=90,
            label="验证样本",
        )

    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")


def _plot_regression_split(
    ax,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    model_name: str,
    degree: int,
    knn_k: int,
    use_l2_regularization: bool,
    regularization_strength: float,
) -> None:
    """Plot the regression split and fitted curve."""

    sorted_indices = np.argsort(X[:, 0])
    x_sorted = X[sorted_indices, 0]
    ax.plot(x_sorted, np.sin(x_sorted), color="#1565C0", linewidth=2.5, label="真实函数 sin(x)")

    model = build_model(
        model_name=model_name,
        task_type="regression",
        degree=degree,
        knn_k=knn_k,
        use_l2_regularization=use_l2_regularization,
        regularization_strength=regularization_strength,
    )
    model.fit(X[train_indices], y[train_indices])
    x_curve = np.linspace(X[:, 0].min(), X[:, 0].max(), 400).reshape(-1, 1)
    y_curve = model.predict(x_curve)
    ax.plot(x_curve[:, 0], y_curve, color="#FB8C00", linestyle="--", linewidth=2.5, label="模型拟合曲线")

    ax.scatter(X[train_indices, 0], y[train_indices], color="#E53935", s=60, label="训练样本")
    if len(valid_indices) > 0:
        ax.scatter(X[valid_indices, 0], y[valid_indices], color="#43A047", marker="^", s=70, label="验证样本")

    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_score_distribution(round_table: pd.DataFrame, primary_metric_name: str):
    """Draw per-round score bars to compare fluctuations."""

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(round_table["轮次"], round_table["验证集主指标"], color="#4C78A8")
    ax.set_xlabel("轮次")
    ax.set_ylabel(primary_metric_name)
    ax.set_title("每一轮验证结果")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_method_comparison(summary_table: pd.DataFrame, primary_metric_name: str, task_type: str):
    """Draw a mean/std comparison chart for different methods."""

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    means = summary_table["验证均值"].to_numpy(dtype=float)
    stds = summary_table["验证标准差"].to_numpy(dtype=float)
    methods = summary_table["评估方法"].tolist()

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"][: len(methods)]
    ax.bar(methods, means, yerr=stds, capsize=6, color=colors)
    direction = "越小越好" if task_type == "regression" else "越大越好"
    ax.set_ylabel(primary_metric_name)
    ax.set_title(f"不同评估方法的均值与波动对比（{direction}）")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    return fig
