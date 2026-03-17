"""Model builders used by the evaluation visualizer."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    """UI-ready description of a model."""

    name: str
    task_type: str
    description: str


def list_models(task_type: str) -> list[ModelConfig]:
    """Return models that match the current task type."""

    if task_type == "regression":
        return [
            ModelConfig("线性回归", "regression", "用一条直线拟合数据，适合做基线比较。"),
            ModelConfig("多项式回归", "regression", "先构造多项式特征，再做线性回归，可表达更复杂曲线。"),
        ]
    return [
        ModelConfig("逻辑回归", "classification", "学习一条线性决策边界，适合线性可分或近似线性问题。"),
        ModelConfig("KNN", "classification", "根据邻近样本投票，适合局部结构明显的非线性分类问题。"),
    ]


def build_model(
    model_name: str,
    task_type: str,
    degree: int,
    knn_k: int,
    use_l2_regularization: bool,
    regularization_strength: float,
):
    """Create a sklearn model object with beginner-friendly defaults."""

    if task_type == "regression":
        if model_name == "线性回归":
            if use_l2_regularization:
                return Ridge(alpha=max(regularization_strength, 1e-6))
            return LinearRegression()

        if model_name == "多项式回归":
            regression_model = Ridge(alpha=max(regularization_strength, 1e-6)) if use_l2_regularization else LinearRegression()
            return Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("regressor", regression_model),
                ]
            )

    if task_type == "classification":
        if model_name == "逻辑回归":
            c_value = 1.0 / max(regularization_strength, 1e-3) if use_l2_regularization else 1e6
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c_value,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=1000,
                        ),
                    ),
                ]
            )
        if model_name == "KNN":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_neighbors=knn_k)),
                ]
            )

    raise ValueError(f"不支持的模型配置：{task_type=} {model_name=}")


def fit_and_predict(model, X_train, y_train, X_eval):
    """Fit on the training subset and predict on another subset."""

    model.fit(X_train, y_train)
    return model.predict(X_eval)
