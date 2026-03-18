"""Utilities for the chapter 2.3 metric visualization demo."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPSILON = 1e-12


@dataclass(frozen=True)
class ScenarioConfig:
    """Preset configuration for synthetic binary classification data."""

    sample_size: int
    positive_ratio: float
    positive_mean: float
    negative_mean: float
    score_std: float
    threshold: float
    seed: int
    description: str


SCENARIOS: dict[str, ScenarioConfig] = {
    "场景 A：类别平衡": ScenarioConfig(
        sample_size=300,
        positive_ratio=0.5,
        positive_mean=0.72,
        negative_mean=0.28,
        score_std=0.16,
        threshold=0.5,
        seed=42,
        description="正负样本数量接近，适合先建立对混淆矩阵和指标的直觉。",
    ),
    "场景 B：类别不平衡": ScenarioConfig(
        sample_size=500,
        positive_ratio=0.12,
        positive_mean=0.70,
        negative_mean=0.24,
        score_std=0.16,
        threshold=0.5,
        seed=7,
        description="正类很少，Accuracy 容易显得很高，但这不一定说明模型真正识别出了正类。",
    ),
    "场景 C：模型效果较好": ScenarioConfig(
        sample_size=320,
        positive_ratio=0.4,
        positive_mean=0.84,
        negative_mean=0.18,
        score_std=0.11,
        threshold=0.5,
        seed=21,
        description="正类分数整体更高，正负样本更容易被分开，AUC 和 F1 往往更高。",
    ),
    "场景 D：模型效果一般": ScenarioConfig(
        sample_size=320,
        positive_ratio=0.4,
        positive_mean=0.66,
        negative_mean=0.34,
        score_std=0.20,
        threshold=0.5,
        seed=21,
        description="正负类分数有明显重叠，阈值变化会更明显地影响 Precision 与 Recall。",
    ),
    "场景 E：模型效果很差": ScenarioConfig(
        sample_size=320,
        positive_ratio=0.5,
        positive_mean=0.54,
        negative_mean=0.46,
        score_std=0.24,
        threshold=0.5,
        seed=21,
        description="分数几乎接近随机，ROC 曲线会更接近对角线，AUC 也会接近 0.5。",
    ),
    "场景 F：自定义参数生成": ScenarioConfig(
        sample_size=300,
        positive_ratio=0.35,
        positive_mean=0.75,
        negative_mean=0.25,
        score_std=0.15,
        threshold=0.5,
        seed=42,
        description="手动调节样本规模、类别比例、分数分布和阈值，观察指标如何联动。",
    ),
}


def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0 when the denominator is too close to 0."""

    if abs(denominator) < EPSILON:
        return 0.0
    return numerator / denominator


def generate_synthetic_scores(
    sample_size: int,
    positive_ratio: float,
    positive_mean: float,
    negative_mean: float,
    score_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary labels and model scores from clipped normal distributions."""

    rng = np.random.default_rng(seed)
    positive_count = int(round(sample_size * positive_ratio))
    positive_count = max(1, min(sample_size - 1, positive_count))
    negative_count = sample_size - positive_count

    y_true = np.concatenate(
        [np.ones(positive_count, dtype=int), np.zeros(negative_count, dtype=int)]
    )
    positive_scores = rng.normal(loc=positive_mean, scale=score_std, size=positive_count)
    negative_scores = rng.normal(loc=negative_mean, scale=score_std, size=negative_count)
    y_score = np.concatenate([positive_scores, negative_scores])
    y_score = np.clip(y_score, 0.0, 1.0)

    order = rng.permutation(sample_size)
    return y_true[order], y_score[order]


def predict_by_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Convert scores into hard predictions."""

    return (y_score >= threshold).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Compute binary confusion matrix counts."""

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def compute_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute accuracy, precision, recall and F1 from confusion counts."""

    tp = counts["TP"]
    fp = counts["FP"]
    tn = counts["TN"]
    fn = counts["FN"]
    total = tp + fp + tn + fn

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    accuracy = safe_divide(tp + tn, total)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def trapezoid_area(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute area under a curve with the trapezoid rule."""

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) < 2:
        return 0.0

    widths = x_values[1:] - x_values[:-1]
    heights = (y_values[1:] + y_values[:-1]) / 2
    return float(np.sum(widths * heights))


def compute_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC points from scratch."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    positive_total = int(np.sum(y_true == 1))
    negative_total = int(np.sum(y_true == 0))
    if positive_total == 0 or negative_total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    tpr_values = [0.0]
    fpr_values = [0.0]
    thresholds = [np.inf]

    tp = 0
    fp = 0
    index = 0
    total = len(sorted_scores)

    while index < total:
        current_score = sorted_scores[index]
        while index < total and sorted_scores[index] == current_score:
            if sorted_true[index] == 1:
                tp += 1
            else:
                fp += 1
            index += 1

        tpr_values.append(safe_divide(tp, positive_total))
        fpr_values.append(safe_divide(fp, negative_total))
        thresholds.append(float(current_score))

    if tpr_values[-1] != 1.0 or fpr_values[-1] != 1.0:
        tpr_values.append(1.0)
        fpr_values.append(1.0)
        thresholds.append(-np.inf)

    return np.array(fpr_values), np.array(tpr_values), np.array(thresholds)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return ROC-AUC, guarded for degenerate label distributions."""

    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = compute_roc_curve(y_true, y_score)
    return trapezoid_area(fpr, tpr)


def summarize_current_state(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, object]:
    """Bundle the current threshold-based predictions, counts and metrics."""

    y_pred = predict_by_threshold(y_score, threshold)
    counts = confusion_counts(y_true, y_pred)
    metrics = compute_metrics_from_counts(counts)
    metrics["auc"] = compute_auc(y_true, y_score)
    return {"y_pred": y_pred, "counts": counts, "metrics": metrics}


def build_confusion_matrix_table(counts: dict[str, int]) -> pd.DataFrame:
    """Build a labeled confusion matrix table for display."""

    return pd.DataFrame(
        [
            [counts["TP"], counts["FN"]],
            [counts["FP"], counts["TN"]],
        ],
        index=["真实为正 (1)", "真实为负 (0)"],
        columns=["预测为正 (1)", "预测为负 (0)"],
    )


def scan_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Evaluate core metrics over thresholds from 0 to 1."""

    rows: list[dict[str, float]] = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for threshold in thresholds:
        y_pred = predict_by_threshold(y_score, threshold)
        counts = confusion_counts(y_true, y_pred)
        metrics = compute_metrics_from_counts(counts)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


def build_sample_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    sample_count: int = 12,
) -> pd.DataFrame:
    """Return a few samples for teaching use."""

    df = pd.DataFrame(
        {
            "真实标签 y_true": y_true,
            "预测分数 y_score": y_score,
            "预测类别 y_pred": y_pred,
        }
    )
    return df.sort_values("预测分数 y_score", ascending=False).head(sample_count).reset_index(drop=True)


def describe_metrics(
    positive_ratio: float,
    threshold: float,
    metrics: dict[str, float],
    counts: dict[str, int],
) -> list[str]:
    """Generate dynamic teaching explanations from the current result."""

    explanations: list[str] = []

    predicted_positive_ratio = safe_divide(counts["TP"] + counts["FP"], sum(counts.values()))
    if threshold <= 0.35:
        explanations.append("当前阈值较低，模型更容易把样本判成正类，因此 Recall 往往更高，但也更容易引入 FP，导致 Precision 下降。")
    elif threshold >= 0.65:
        explanations.append("当前阈值较高，模型只有在分数足够高时才判为正类，这通常会压低 FP、提高 Precision，但可能漏掉更多真正的正类。")
    else:
        explanations.append("当前阈值处于中间区域，模型没有明显偏向“多报正类”或“少报正类”，更适合观察 Precision 与 Recall 的平衡。")

    if positive_ratio <= 0.2 and metrics["accuracy"] - metrics["recall"] >= 0.2:
        explanations.append("当前数据存在明显类别不平衡，Accuracy 看起来可能不错，但它掩盖了正类样本较少、且可能没有被充分识别的问题。")
    elif positive_ratio >= 0.45:
        explanations.append("当前类别较平衡，此时 Accuracy 的参考价值比极端不平衡场景更高。")

    if metrics["f1"] >= 0.75:
        explanations.append("当前 F1 较高，说明 Precision 和 Recall 之间达成了较好的折中。")
    elif metrics["precision"] > metrics["recall"] + 0.15:
        explanations.append("Precision 明显高于 Recall，说明模型更保守，宁可少报正类，也希望一旦报正就尽量准确。")
    elif metrics["recall"] > metrics["precision"] + 0.15:
        explanations.append("Recall 明显高于 Precision，说明模型更激进，倾向于尽量找出正类，但会带来更多误报。")
    else:
        explanations.append("Precision 和 Recall 较为接近，模型当前阈值附近的取舍相对均衡。")

    if metrics["auc"] >= 0.85:
        explanations.append("AUC 较高，说明不论阈值怎样调整，模型整体上都有较强的正负样本排序能力。")
    elif metrics["auc"] <= 0.6:
        explanations.append("AUC 接近 0.5，说明 ROC 曲线会靠近对角线，模型整体区分能力有限。")
    else:
        explanations.append("AUC 处于中等水平，表示模型能提供一定排序信息，但正负样本仍有较多重叠。")

    if predicted_positive_ratio >= 0.7:
        explanations.append("当前被判为正类的样本很多，通常意味着 FN 下降，但 FP 更值得重点关注。")
    elif predicted_positive_ratio <= 0.15:
        explanations.append("当前被判为正类的样本较少，模型更保守，此时需要留意是否漏掉了过多真正的正类。")

    return explanations


def build_step_explanation(counts: dict[str, int], threshold: float) -> list[str]:
    """Create a compact step-by-step explanation for the current threshold."""

    steps = [
        f"当 threshold = {threshold:.2f} 时，所有预测分数大于等于阈值的样本都会被判为正类。",
        f"此时共有 {counts['TP'] + counts['FP']} 个样本被判为正类，其中真正例 TP = {counts['TP']}，假正例 FP = {counts['FP']}。",
        f"同时有 {counts['TN'] + counts['FN']} 个样本被判为负类，其中真反例 TN = {counts['TN']}，假反例 FN = {counts['FN']}。",
    ]
    if counts["FP"] > counts["FN"]:
        steps.append("目前误报多于漏报，说明阈值相对偏低，模型更愿意把不确定样本归为正类。")
    elif counts["FN"] > counts["FP"]:
        steps.append("目前漏报多于误报，说明阈值相对偏高，模型更倾向于谨慎地给出正类判断。")
    else:
        steps.append("目前误报和漏报数量接近，可以对照阈值曲线继续观察哪一种代价更值得优先控制。")
    return steps


def plot_score_distribution(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> plt.Figure:
    """Plot score distributions for positive and negative classes."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.65, label="正类样本分数", density=True)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.65, label="负类样本分数", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("预测分数")
    ax.set_ylabel("相对频率")
    ax.set_title("分数分布图")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_roc_curve_with_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> plt.Figure:
    """Plot ROC curve and mark the point implied by the current threshold."""

    fpr, tpr, _ = compute_roc_curve(y_true, y_score)
    auc_value = compute_auc(y_true, y_score)

    current_state = summarize_current_state(y_true, y_score, threshold)
    counts = current_state["counts"]
    current_tpr = safe_divide(counts["TP"], counts["TP"] + counts["FN"])
    current_fpr = safe_divide(counts["FP"], counts["FP"] + counts["TN"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC 曲线 (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="随机分类参考线")
    ax.scatter([current_fpr], [current_tpr], color="black", s=40, label=f"当前阈值点 ({threshold:.2f})")
    ax.set_xlabel("假正例率 FPR")
    ax.set_ylabel("真正例率 TPR")
    ax.set_title("ROC 曲线")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_threshold_curves(metrics_df: pd.DataFrame, threshold: float) -> plt.Figure:
    """Plot metric-threshold relationships."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(metrics_df["threshold"], metrics_df["precision"], label="Precision")
    ax.plot(metrics_df["threshold"], metrics_df["recall"], label="Recall")
    ax.plot(metrics_df["threshold"], metrics_df["f1"], label="F1")
    ax.plot(metrics_df["threshold"], metrics_df["accuracy"], label="Accuracy", linestyle=":")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"当前阈值={threshold:.2f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("指标值")
    ax.set_title("指标随阈值变化")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig
