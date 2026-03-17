"""Metrics and lightweight teaching heuristics."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean squared error."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def explain_model_behavior(
    degree: int,
    train_mse: float,
    test_mse: float,
    regularization_name: str,
    lambda_value: float,
    weight_norm: float,
    nonzero_weight_count: int,
) -> list[str]:
    """Generate short teaching messages from the current result."""

    messages: list[str] = []
    gap = test_mse - train_mse

    if degree <= 2 and train_mse > 0.2:
        messages.append("当前多项式次数较低，模型表达能力有限，可能出现欠拟合。")

    if train_mse < 0.03 and gap > max(0.05, train_mse * 1.5):
        messages.append("训练误差很小但测试误差更大，说明模型可能在记忆训练样本，存在过拟合风险。")

    if abs(gap) <= 0.03:
        messages.append("训练误差与测试误差接近，说明模型的泛化表现比较稳定。")

    if regularization_name == "L2" and lambda_value > 0:
        messages.append("L2 正则化会惩罚过大的参数，通常会让曲线更平滑，降低高次项带来的摆动。")

    if regularization_name == "L1" and lambda_value > 0:
        messages.append("L1 正则化会推动部分参数变成 0，因此模型往往更稀疏，也更容易观察哪些阶数真的在起作用。")

    if regularization_name == "L1" and nonzero_weight_count <= max(1, degree // 3):
        messages.append("当前 L1 让不少高次项被压到了 0，说明模型复杂度正在被主动裁剪。")

    if weight_norm > 50:
        messages.append("当前参数范数较大，模型对样本扰动会更敏感，适当增大正则化系数可能更稳。")

    if not messages:
        messages.append("可以继续调节次数、噪声或正则化系数，观察经验误差与测试误差如何变化。")

    return messages
