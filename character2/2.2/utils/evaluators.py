"""Evaluation orchestration for hold-out, cross validation, LOOCV, and bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.metrics import evaluate_predictions, primary_metric_name
from utils.models import build_model, fit_and_predict
from utils.splitters import (
    SplitResult,
    bootstrap_split,
    hold_out_split,
    k_fold_split,
    loocv_split,
    repeated_hold_out_split,
    train_test_split_indices,
)


@dataclass
class EvaluationArtifacts:
    """All outputs needed by the Streamlit UI."""

    splits: list[SplitResult]
    round_records: list[dict[str, object]]
    round_table: pd.DataFrame
    summary: dict[str, object]
    development_indices: np.ndarray
    test_indices: np.ndarray | None
    method_explanation: str
    experiment_note: str


def _select_splits(
    sample_count: int,
    evaluation_method: str,
    train_ratio: float,
    k_folds: int,
    bootstrap_rounds: int,
    repeated_holdout_rounds: int,
    random_seed: int,
    shuffle_data: bool,
) -> list[SplitResult]:
    """Dispatch to the requested evaluation method."""

    if evaluation_method == "留出法":
        return hold_out_split(sample_count, train_ratio, random_seed, shuffle=shuffle_data)
    if evaluation_method == "重复留出法":
        return repeated_hold_out_split(
            sample_count,
            train_ratio,
            repeated_holdout_rounds,
            random_seed,
            shuffle=shuffle_data,
        )
    if evaluation_method == "k折交叉验证":
        return k_fold_split(sample_count, k_folds, random_seed, shuffle=shuffle_data)
    if evaluation_method == "留一法":
        return loocv_split(sample_count)
    if evaluation_method == "自助法":
        return bootstrap_split(sample_count, bootstrap_rounds, random_seed)
    raise ValueError(f"未知评估方法：{evaluation_method}")


def _evaluate_one_split(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    split: SplitResult,
    task_type: str,
    model_name: str,
    degree: int,
    knn_k: int,
    use_l2_regularization: bool,
    regularization_strength: float,
) -> dict[str, object]:
    """Train one model on one split and compute train/valid/test metrics."""

    X_train = X_dev[split.train_indices]
    y_train = y_dev[split.train_indices]
    X_valid = X_dev[split.valid_indices] if len(split.valid_indices) > 0 else None
    y_valid = y_dev[split.valid_indices] if len(split.valid_indices) > 0 else None

    model = build_model(
        model_name=model_name,
        task_type=task_type,
        degree=degree,
        knn_k=knn_k,
        use_l2_regularization=use_l2_regularization,
        regularization_strength=regularization_strength,
    )

    train_predictions = fit_and_predict(model, X_train, y_train, X_train)
    train_metrics = evaluate_predictions(task_type, y_train, train_predictions)

    valid_metrics: dict[str, float] = {}
    if X_valid is not None and len(X_valid) > 0:
        valid_predictions = model.predict(X_valid)
        valid_metrics = evaluate_predictions(task_type, y_valid, valid_predictions)

    test_metrics: dict[str, float] = {}
    if X_test is not None and y_test is not None and len(X_test) > 0:
        test_predictions = model.predict(X_test)
        test_metrics = evaluate_predictions(task_type, y_test, test_predictions)

    unique_train, counts = np.unique(split.train_indices, return_counts=True)
    duplicate_counter = {int(index): int(count) for index, count in zip(unique_train, counts, strict=False) if count > 1}

    return {
        "round_id": split.round_id,
        "train_indices": split.train_indices,
        "valid_indices": split.valid_indices,
        "train_size": int(len(split.train_indices)),
        "valid_size": int(len(split.valid_indices)),
        "unique_train_size": int(len(unique_train)),
        "duplicate_counter": duplicate_counter,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "note": split.note,
    }


def _build_round_table(
    task_type: str,
    round_records: list[dict[str, object]],
) -> pd.DataFrame:
    """Convert raw round records to a flat DataFrame for display."""

    rows: list[dict[str, object]] = []
    primary_name = primary_metric_name(task_type)

    for record in round_records:
        row = {
            "轮次": int(record["round_id"]),
            "训练样本数": int(record["train_size"]),
            "验证样本数": int(record["valid_size"]),
            "训练集主指标": float(record["train_metrics"].get(primary_name, np.nan)),
            "验证集主指标": float(record["valid_metrics"].get(primary_name, np.nan)),
            "固定测试集主指标": float(record["test_metrics"].get(primary_name, np.nan)),
            "训练去重后样本数": int(record["unique_train_size"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize(
    task_type: str,
    evaluation_method: str,
    round_records: list[dict[str, object]],
    has_fixed_test_set: bool,
) -> dict[str, object]:
    """Aggregate mean and standard deviation for the key metric."""

    primary_name = primary_metric_name(task_type)
    train_scores = np.array([record["train_metrics"].get(primary_name, np.nan) for record in round_records], dtype=float)
    valid_scores = np.array([record["valid_metrics"].get(primary_name, np.nan) for record in round_records], dtype=float)
    test_scores = np.array([record["test_metrics"].get(primary_name, np.nan) for record in round_records], dtype=float)

    summary = {
        "primary_metric": primary_name,
        "train_mean": float(np.nanmean(train_scores)),
        "train_std": float(np.nanstd(train_scores)),
        "valid_mean": float(np.nanmean(valid_scores)),
        "valid_std": float(np.nanstd(valid_scores)),
        "n_rounds": len(round_records),
        "has_fixed_test_set": has_fixed_test_set,
    }
    if has_fixed_test_set:
        summary["test_mean"] = float(np.nanmean(test_scores))
        summary["test_std"] = float(np.nanstd(test_scores))

    summary["stability_hint"] = explain_stability(
        task_type=task_type,
        evaluation_method=evaluation_method,
        valid_std=summary["valid_std"],
        n_rounds=len(round_records),
    )
    return summary


def explain_method(evaluation_method: str) -> str:
    """Return a concise teaching explanation for the selected method."""

    explanations = {
        "留出法": "留出法把数据一次性分成训练集和验证集，实现简单，但性能估计会明显依赖这一刀怎么切。",
        "重复留出法": "重复留出法把留出法做很多次，用多次结果的均值和波动来说明单次切分为什么可能不稳定。",
        "k折交叉验证": "k 折交叉验证让每个样本轮流做一次验证，通常比单次留出法更稳健，但需要训练 k 次模型。",
        "留一法": "留一法是极端的交叉验证：每次只留 1 个样本做验证，样本利用率很高，但计算代价最大。",
        "自助法": "自助法通过有放回采样构造训练集，因此训练集中会出现重复样本，同时会留下袋外样本可作验证参考。",
    }
    return explanations[evaluation_method]


def explain_stability(task_type: str, evaluation_method: str, valid_std: float, n_rounds: int) -> str:
    """Generate a dynamic explanation based on score variance."""

    metric_name = primary_metric_name(task_type)
    if n_rounds <= 1:
        return f"当前只有 1 次估计，无法直接观察 {metric_name} 的波动，这正是单次留出法容易让人误判稳定性的原因。"
    if valid_std < 0.03:
        return f"当前 {metric_name} 的轮次标准差较小，说明这组设置下不同划分给出的估计比较接近，方法更稳定。"
    if valid_std < 0.08:
        return f"当前 {metric_name} 的轮次标准差中等，说明数据划分会带来一定波动，但还不算特别剧烈。"
    return f"当前 {metric_name} 的轮次标准差较大，说明样本较少、噪声较大或模型与数据不匹配时，不同划分会给出明显不同的性能估计。"


def build_experiment_note(
    task_type: str,
    evaluation_method: str,
    summary: dict[str, object],
    has_fixed_test_set: bool,
) -> str:
    """Generate the natural-language summary shown at the bottom of the page."""

    metric_name = summary["primary_metric"]
    valid_mean = summary["valid_mean"]
    valid_std = summary["valid_std"]
    method_text = explain_method(evaluation_method)

    if task_type == "regression":
        metric_sentence = f"当前验证集 {metric_name} 平均为 {valid_mean:.4f}，标准差为 {valid_std:.4f}；数值越小通常表示拟合越好。"
    else:
        metric_sentence = f"当前验证集 {metric_name} 平均为 {valid_mean:.4f}，标准差为 {valid_std:.4f}；数值越高通常表示分类效果越好。"

    test_sentence = ""
    if has_fixed_test_set:
        test_sentence = (
            f" 你启用了固定测试集，所以测试集只用于最后旁观模型表现，"
            f"更适合讲清楚“不要反复拿测试集调参”的原则。"
        )

    return method_text + " " + metric_sentence + " " + str(summary["stability_hint"]) + test_sentence


def evaluate_experiment(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_name: str,
    evaluation_method: str,
    degree: int,
    knn_k: int,
    use_l2_regularization: bool,
    regularization_strength: float,
    train_ratio: float,
    k_folds: int,
    bootstrap_rounds: int,
    repeated_holdout_rounds: int,
    random_seed: int,
    shuffle_data: bool,
    use_fixed_test_set: bool,
    fixed_test_ratio: float,
) -> EvaluationArtifacts:
    """Run the selected evaluation method and collect outputs for the UI."""

    all_indices = np.arange(len(X))
    development_indices = all_indices
    test_indices: np.ndarray | None = None

    if use_fixed_test_set:
        development_indices, test_indices = train_test_split_indices(
            sample_count=len(X),
            train_ratio=1.0 - fixed_test_ratio,
            random_seed=random_seed + 999,
            shuffle=shuffle_data,
        )

    X_dev = X[development_indices]
    y_dev = y[development_indices]
    X_test = X[test_indices] if test_indices is not None else None
    y_test = y[test_indices] if test_indices is not None else None

    splits = _select_splits(
        sample_count=len(X_dev),
        evaluation_method=evaluation_method,
        train_ratio=train_ratio,
        k_folds=k_folds,
        bootstrap_rounds=bootstrap_rounds,
        repeated_holdout_rounds=repeated_holdout_rounds,
        random_seed=random_seed,
        shuffle_data=shuffle_data,
    )

    round_records = [
        _evaluate_one_split(
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            split=split,
            task_type=task_type,
            model_name=model_name,
            degree=degree,
            knn_k=knn_k,
            use_l2_regularization=use_l2_regularization,
            regularization_strength=regularization_strength,
        )
        for split in splits
    ]

    round_table = _build_round_table(task_type, round_records)
    summary = _summarize(task_type, evaluation_method, round_records, has_fixed_test_set=use_fixed_test_set)
    experiment_note = build_experiment_note(
        task_type=task_type,
        evaluation_method=evaluation_method,
        summary=summary,
        has_fixed_test_set=use_fixed_test_set,
    )

    return EvaluationArtifacts(
        splits=splits,
        round_records=round_records,
        round_table=round_table,
        summary=summary,
        development_indices=development_indices,
        test_indices=test_indices,
        method_explanation=explain_method(evaluation_method),
        experiment_note=experiment_note,
    )
