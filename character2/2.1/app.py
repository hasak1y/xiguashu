"""Streamlit app for visualizing polynomial fitting and regularization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.data import generate_dataset, list_dataset_names
from utils.features import apply_standardization, build_polynomial_features, standardize_feature
from utils.metrics import explain_model_behavior, mean_squared_error
from utils.model import (
    fit_l1_regularized,
    fit_l2_regularized,
    fit_ordinary_least_squares,
    predict,
)


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="多项式拟合与正则化可视化实验",
    page_icon="📈",
    layout="wide",
)


def fit_polynomial_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    degree: int,
    regularization_name: str,
    lambda_value: float,
    standardize_x: bool,
) -> dict[str, object]:
    """Fit one model and return predictions plus diagnostics."""

    x_train_processed, scaling_stats = standardize_feature(x_train, enabled=standardize_x)
    x_eval_processed = apply_standardization(x_eval, scaling_stats)

    X_train = build_polynomial_features(x_train_processed, degree)
    X_eval = build_polynomial_features(x_eval_processed, degree)

    if regularization_name == "L2" and lambda_value > 0:
        model_info = fit_l2_regularized(X_train, y_train, lambda_value, regularize_bias=False)
    elif regularization_name == "L1" and lambda_value > 0:
        model_info = fit_l1_regularized(X_train, y_train, lambda_value, regularize_bias=False)
    else:
        model_info = fit_ordinary_least_squares(X_train, y_train)

    weights = np.asarray(model_info["weights"], dtype=float)
    y_eval_pred = predict(X_eval, weights)

    return {
        "weights": weights,
        "predictions": y_eval_pred,
        "scaling_stats": scaling_stats,
        "condition_number": float(model_info["condition_number"]),
        "method": str(model_info["method"]),
    }


def plot_main_figure(
    dataset: dict[str, np.ndarray],
    y_curve_pred: np.ndarray,
    train_mse: float,
    test_mse: float,
) -> plt.Figure:
    """Create the main fitting visualization."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataset["x_curve"], dataset["y_curve"], label="真实函数", linewidth=2.5, color="#1565C0")
    ax.scatter(
        dataset["x_train"],
        dataset["y_train"],
        label="训练样本（带噪声）",
        color="#E53935",
        s=40,
        alpha=0.85,
    )
    ax.scatter(
        dataset["x_test"],
        dataset["y_test"],
        label="测试集真实点",
        color="#43A047",
        s=18,
        alpha=0.35,
    )
    ax.plot(
        dataset["x_curve"],
        y_curve_pred,
        label="模型拟合曲线",
        linewidth=2.5,
        linestyle="--",
        color="#FB8C00",
    )
    ax.set_title(
        f"当前拟合结果 | 训练集 MSE = {train_mse:.4f} | 测试集 MSE = {test_mse:.4f}",
        fontsize=13,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def plot_complexity_curve(
    dataset: dict[str, np.ndarray],
    max_degree: int,
    regularization_name: str,
    lambda_value: float,
    standardize_x: bool,
) -> plt.Figure:
    """Plot train/test error as model complexity changes."""

    degrees = list(range(1, max_degree + 1))
    train_errors: list[float] = []
    test_errors: list[float] = []

    for degree in degrees:
        train_fit = fit_polynomial_model(
            dataset["x_train"],
            dataset["y_train"],
            dataset["x_train"],
            degree,
            regularization_name,
            lambda_value,
            standardize_x,
        )
        test_fit = fit_polynomial_model(
            dataset["x_train"],
            dataset["y_train"],
            dataset["x_test"],
            degree,
            regularization_name,
            lambda_value,
            standardize_x,
        )

        train_errors.append(mean_squared_error(dataset["y_train"], train_fit["predictions"]))
        test_errors.append(mean_squared_error(dataset["y_test"], test_fit["predictions"]))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(degrees, train_errors, marker="o", linewidth=2, label="训练集 MSE")
    ax.plot(degrees, test_errors, marker="o", linewidth=2, label="测试集 MSE")
    ax.set_xlabel("多项式次数 degree")
    ax.set_ylabel("MSE")
    ax.set_title("误差 - 模型复杂度曲线")
    ax.set_xticks(degrees)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


st.title("多项式拟合与正则化可视化实验")
st.caption("通过交互式调参，观察经验误差、测试误差、欠拟合、过拟合，以及 L1/L2 正则化对拟合曲线的影响。")

with st.sidebar:
    st.header("实验设置")

    dataset_name = st.selectbox("数据集类型", list_dataset_names(), index=0)

    st.subheader("样本参数")
    train_size = st.slider("训练集样本数", min_value=5, max_value=100, value=18, step=1)
    test_size = st.slider("测试集样本数", min_value=50, max_value=500, value=200, step=10)
    noise_strength = st.slider("训练集噪声强度", min_value=0.0, max_value=3.0, value=0.15, step=0.05)
    random_seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)

    st.subheader("多项式拟合设置")
    degree = st.slider("多项式次数 degree", min_value=1, max_value=15, value=5, step=1)
    standardize_x = st.checkbox("拟合前标准化 x", value=True)
    enable_regularization = st.checkbox("启用正则化", value=False)

    st.subheader("正则化设置")
    regularization_name = st.radio(
        "正则化类型",
        options=["None", "L1", "L2"],
        index=2 if enable_regularization else 0,
    )
    if not enable_regularization:
        regularization_name = "None"
    lambda_value = st.slider("正则化系数 lambda", min_value=0.0, max_value=10.0, value=0.5, step=0.05)

dataset = generate_dataset(
    dataset_name=dataset_name,
    train_size=train_size,
    test_size=test_size,
    noise_strength=noise_strength,
    random_seed=int(random_seed),
)

curve_fit = fit_polynomial_model(
    dataset["x_train"],
    dataset["y_train"],
    dataset["x_curve"],
    degree,
    regularization_name,
    lambda_value,
    standardize_x,
)
train_fit = fit_polynomial_model(
    dataset["x_train"],
    dataset["y_train"],
    dataset["x_train"],
    degree,
    regularization_name,
    lambda_value,
    standardize_x,
)
test_fit = fit_polynomial_model(
    dataset["x_train"],
    dataset["y_train"],
    dataset["x_test"],
    degree,
    regularization_name,
    lambda_value,
    standardize_x,
)

train_mse = mean_squared_error(dataset["y_train"], train_fit["predictions"])
test_mse = mean_squared_error(dataset["y_test"], test_fit["predictions"])
weights = np.asarray(curve_fit["weights"], dtype=float)
weight_norm = float(np.linalg.norm(weights))
nonzero_weight_count = int(np.sum(np.abs(weights[1:]) > 1e-6)) if len(weights) > 1 else 0

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("训练集 MSE", f"{train_mse:.4f}")
metric_col2.metric("测试集 MSE", f"{test_mse:.4f}")
metric_col3.metric("参数范数 ||w||", f"{weight_norm:.4f}")
metric_col4.metric("非零高次项个数", f"{nonzero_weight_count}")

left_col, right_col = st.columns([1.7, 1.0])

with left_col:
    st.pyplot(
        plot_main_figure(
            dataset,
            np.asarray(curve_fit["predictions"], dtype=float),
            train_mse,
            test_mse,
        ),
        clear_figure=True,
    )

with right_col:
    st.subheader("当前模型参数")
    st.write(f"- degree: `{degree}`")
    st.write(f"- 正则化类型: `{regularization_name}`")
    st.write(f"- lambda: `{lambda_value:.2f}`")
    st.write(f"- 是否标准化 x: `{standardize_x}`")
    st.write("- 截距项是否正则化: `否`")
    st.write(f"- 条件数: `{curve_fit['condition_number']:.2e}`")
    st.write(f"- 权重向量 w: `{np.round(weights, 4).tolist()}`")

    st.subheader("概念解释")
    st.markdown(
        """
经验误差：模型在训练集上的误差，也就是它对见过的数据拟合得怎么样。  
测试误差：模型在测试集上的误差，用来近似衡量泛化能力，也就是它对没见过的数据表现如何。  
L1 正则化：会推动一部分参数直接变成 0，因此常常让模型更稀疏。  
L2 正则化：会把参数整体压小，通常让曲线更平滑。
"""
    )

    st.subheader("自动解读")
    for message in explain_model_behavior(
        degree=degree,
        train_mse=train_mse,
        test_mse=test_mse,
        regularization_name=regularization_name,
        lambda_value=lambda_value,
        weight_norm=weight_norm,
        nonzero_weight_count=nonzero_weight_count,
    ):
        st.write(f"- {message}")

st.subheader("误差 - 复杂度观察")
st.write("点击按钮后，系统会遍历 `degree = 1 ~ 15`，帮助你观察模型复杂度变化时，训练误差和测试误差如何变化。")

if st.button("生成误差-复杂度曲线", use_container_width=True):
    st.pyplot(
        plot_complexity_curve(
            dataset=dataset,
            max_degree=15,
            regularization_name=regularization_name,
            lambda_value=lambda_value,
            standardize_x=standardize_x,
        ),
        clear_figure=True,
    )

st.subheader("学习提示")
st.info(
    "当 degree 很小时，曲线可能过于简单，训练误差和测试误差都会偏高，这通常是欠拟合。"
    " 当 degree 很高时，模型可能把噪声也当成规律去学习，训练误差下降，但测试误差反而上升，这通常是过拟合。"
    " L1 有助于压缩一部分参数，L2 则更倾向于把参数整体变小并让曲线更平滑。"
)
