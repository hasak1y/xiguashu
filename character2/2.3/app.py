"""Streamlit teaching demo for chapter 2.3 performance metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import streamlit as st

from utils import (
    SCENARIOS,
    build_confusion_matrix_table,
    build_sample_table,
    build_step_explanation,
    describe_metrics,
    generate_synthetic_scores,
    plot_roc_curve_with_threshold,
    plot_score_distribution,
    plot_threshold_curves,
    scan_threshold_metrics,
    summarize_current_state,
)


def configure_matplotlib_for_chinese() -> str:
    """Try to select a Chinese-capable font for matplotlib."""

    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    selected_font = "sans-serif"
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    plt.rcParams["font.sans-serif"] = [selected_font] + candidate_fonts
    plt.rcParams["axes.unicode_minus"] = False
    return selected_font


SELECTED_FONT = configure_matplotlib_for_chinese()

st.set_page_config(
    page_title="西瓜书 2.3 性能度量可视化",
    page_icon="📊",
    layout="wide",
)


METRIC_TEXT = {
    "accuracy": "整体预测正确的比例",
    "precision": "预测为正的样本中，真正为正的比例",
    "recall": "真实为正的样本中，被找出来的比例",
    "f1": "Precision 和 Recall 的折中",
    "auc": "模型整体区分正负样本的能力",
}


def render_metric_card(title: str, value: float, caption: str) -> None:
    """Render a metric with a short explanation."""

    st.metric(title, f"{value:.4f}")
    st.caption(caption)


def load_sidebar_inputs() -> dict[str, float | int | str]:
    """Read sidebar controls and apply scenario presets."""

    st.sidebar.header("参数设置")
    scenario_name = st.sidebar.selectbox("数据场景", list(SCENARIOS.keys()), index=0)
    scenario = SCENARIOS[scenario_name]
    st.sidebar.caption(scenario.description)

    sample_size = st.sidebar.slider("总样本数", min_value=50, max_value=2000, value=scenario.sample_size, step=50)
    positive_ratio = st.sidebar.slider(
        "正类比例", min_value=0.05, max_value=0.95, value=float(scenario.positive_ratio), step=0.01
    )
    positive_mean = st.sidebar.slider(
        "正类分数均值", min_value=0.0, max_value=1.0, value=float(scenario.positive_mean), step=0.01
    )
    negative_mean = st.sidebar.slider(
        "负类分数均值", min_value=0.0, max_value=1.0, value=float(scenario.negative_mean), step=0.01
    )
    score_std = st.sidebar.slider(
        "分数分布标准差", min_value=0.02, max_value=0.35, value=float(scenario.score_std), step=0.01
    )
    threshold = st.sidebar.slider(
        "分类阈值 threshold", min_value=0.0, max_value=1.0, value=float(scenario.threshold), step=0.01
    )
    seed = st.sidebar.number_input("随机种子", min_value=0, max_value=99999, value=scenario.seed, step=1)

    if scenario_name != "场景 F：自定义参数生成":
        st.sidebar.info("当前是预设场景，你仍然可以继续拖动参数，观察结果如何偏离预设。")

    return {
        "scenario_name": scenario_name,
        "sample_size": int(sample_size),
        "positive_ratio": float(positive_ratio),
        "positive_mean": float(positive_mean),
        "negative_mean": float(negative_mean),
        "score_std": float(score_std),
        "threshold": float(threshold),
        "seed": int(seed),
    }


def render_confusion_matrix_heatmap(confusion_df: pd.DataFrame) -> None:
    """Render confusion matrix heatmap using matplotlib only."""

    fig, ax = plt.subplots(figsize=(4.8, 4))
    image = ax.imshow(confusion_df.values, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(confusion_df.columns)), confusion_df.columns)
    ax.set_yticks(range(len(confusion_df.index)), confusion_df.index)
    ax.set_title("混淆矩阵")

    max_value = confusion_df.values.max() if confusion_df.values.size else 1
    threshold_value = max_value / 2
    for row in range(confusion_df.shape[0]):
        for col in range(confusion_df.shape[1]):
            value = int(confusion_df.iloc[row, col])
            color = "white" if value > threshold_value else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def main() -> None:
    """Render the full teaching demo."""

    st.title("西瓜书 2.3 性能度量可视化")
    st.write(
        "这是一个面向教学的二分类性能指标演示应用。你可以修改数据分布和分类阈值，"
        "直观看到混淆矩阵、Accuracy、Precision、Recall、F1、ROC 曲线和 AUC 如何一起变化。"
    )

    if SELECTED_FONT == "sans-serif":
        st.warning("未检测到常见中文字体，页面文字正常，但图中的中文可能仍显示为方框。")
    else:
        st.caption(f"图表字体已自动设置为：{SELECTED_FONT}")

    params = load_sidebar_inputs()
    y_true, y_score = generate_synthetic_scores(
        sample_size=params["sample_size"],
        positive_ratio=params["positive_ratio"],
        positive_mean=params["positive_mean"],
        negative_mean=params["negative_mean"],
        score_std=params["score_std"],
        seed=params["seed"],
    )

    current = summarize_current_state(y_true, y_score, params["threshold"])
    counts = current["counts"]
    metrics = current["metrics"]
    confusion_df = build_confusion_matrix_table(counts)
    threshold_metrics = scan_threshold_metrics(y_true, y_score)
    sample_df = build_sample_table(y_true, y_score, current["y_pred"])

    st.subheader("当前样本概览")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本总数", f"{len(y_true)}")
    col2.metric("正类样本数", f"{int(y_true.sum())}")
    col3.metric("负类样本数", f"{int((1 - y_true).sum())}")
    col4.metric("当前阈值", f"{params['threshold']:.2f}")

    st.caption(
        "教学重点：阈值不会改变 AUC，因为 AUC 反映的是模型对样本排序的整体能力；"
        "阈值会直接改变 y_pred，因此会影响混淆矩阵、Accuracy、Precision、Recall 和 F1。"
    )

    st.subheader("当前混淆矩阵与计数")
    left, right = st.columns([1.1, 1])
    with left:
        render_confusion_matrix_heatmap(confusion_df)
    with right:
        st.dataframe(confusion_df, use_container_width=True)
        st.write(f"TP = {counts['TP']}，FP = {counts['FP']}，TN = {counts['TN']}，FN = {counts['FN']}")
        st.markdown(
            """
            - `TP`：真实为正，也预测为正
            - `FP`：真实为负，却预测为正
            - `TN`：真实为负，也预测为负
            - `FN`：真实为正，却预测为负
            """
        )

    st.subheader("当前核心指标")
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Accuracy", metrics["accuracy"], METRIC_TEXT["accuracy"])
    with metric_cols[1]:
        render_metric_card("Precision", metrics["precision"], METRIC_TEXT["precision"])
    with metric_cols[2]:
        render_metric_card("Recall", metrics["recall"], METRIC_TEXT["recall"])
    with metric_cols[3]:
        render_metric_card("F1 Score", metrics["f1"], METRIC_TEXT["f1"])
    with metric_cols[4]:
        render_metric_card("AUC", metrics["auc"], METRIC_TEXT["auc"])

    st.info(
        "公式提示：Accuracy = (TP + TN) / N，Precision = TP / (TP + FP)，"
        "Recall = TP / (TP + FN)，F1 = 2PR / (P + R)。当分母为 0 时，本应用按 0 处理。"
    )

    st.subheader("分数分布图")
    st.pyplot(plot_score_distribution(y_true, y_score, params["threshold"]), use_container_width=True)
    st.caption("观察黑色竖线与两类分数分布的位置关系：阈值左移时，更多样本会被判为正类；阈值右移时，模型会更保守。")

    st.subheader("ROC 曲线")
    st.pyplot(plot_roc_curve_with_threshold(y_true, y_score, params["threshold"]), use_container_width=True)
    st.caption("ROC 曲线横轴是假正例率 FPR，纵轴是真正例率 TPR。曲线越靠近左上角，说明模型越能在不同阈值下兼顾较高 TPR 和较低 FPR。")

    st.subheader("指标随阈值变化")
    st.pyplot(plot_threshold_curves(threshold_metrics, params["threshold"]), use_container_width=True)
    st.caption("这张图最适合教学展示 Precision 与 Recall 的此消彼长，以及为什么 F1 常被用作两者平衡时的参考指标。")

    st.subheader("部分样本预览")
    st.dataframe(
        sample_df.style.format({"预测分数 y_score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("可结合阈值观察：分数高于 threshold 的样本会被判成 1，低于 threshold 的样本会被判成 0。")

    st.subheader("逐步解释")
    for step in build_step_explanation(counts, params["threshold"]):
        st.write(f"- {step}")

    st.subheader("当前结果解读")
    for explanation in describe_metrics(
        positive_ratio=params["positive_ratio"],
        threshold=params["threshold"],
        metrics=metrics,
        counts=counts,
    ):
        st.write(f"- {explanation}")


if __name__ == "__main__":
    main()
