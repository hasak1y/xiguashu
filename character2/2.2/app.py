"""Streamlit app for chapter 2.2 evaluation method visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.data import generate_dataset, list_datasets
from utils.evaluators import evaluate_experiment
from utils.metrics import primary_metric_name
from utils.models import list_models
from utils.plots import plot_method_comparison, plot_score_distribution, plot_split_visualization


st.set_page_config(
    page_title="评估方法可视化实验",
    page_icon="📊",
    layout="wide",
)


def build_method_summary_table(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_name: str,
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
) -> pd.DataFrame:
    """Run a lightweight comparison across several evaluation methods."""

    methods = ["留出法", "重复留出法", "k折交叉验证", "留一法", "自助法"]
    rows: list[dict[str, object]] = []
    for method in methods:
        if method == "留一法" and len(X) > 120:
            continue
        artifacts = evaluate_experiment(
            X=X,
            y=y,
            task_type=task_type,
            model_name=model_name,
            evaluation_method=method,
            degree=degree,
            knn_k=knn_k,
            use_l2_regularization=use_l2_regularization,
            regularization_strength=regularization_strength,
            train_ratio=train_ratio,
            k_folds=k_folds,
            bootstrap_rounds=min(bootstrap_rounds, 30),
            repeated_holdout_rounds=min(repeated_holdout_rounds, 12),
            random_seed=random_seed,
            shuffle_data=shuffle_data,
            use_fixed_test_set=use_fixed_test_set,
            fixed_test_ratio=fixed_test_ratio,
        )
        rows.append(
            {
                "评估方法": method,
                "验证均值": artifacts.summary["valid_mean"],
                "验证标准差": artifacts.summary["valid_std"],
                "轮次数": artifacts.summary["n_rounds"],
            }
        )
    return pd.DataFrame(rows)


def format_index_preview(indices: np.ndarray, max_items: int = 18) -> str:
    """Format indices for the split explanation panel."""

    preview = indices[:max_items].tolist()
    suffix = " ..." if len(indices) > max_items else ""
    return f"{preview}{suffix}"


def main() -> None:
    """Render the Streamlit application."""

    st.title("机器学习评估方法可视化实验：留出法、交叉验证、自助法")
    st.caption("围绕《机器学习》（西瓜书）第 2 章 2.2 节，交互式观察数据怎么划分、模型怎么被评估，以及为什么不同方法会得到不同的性能估计。")

    with st.sidebar:
        st.header("实验设置")

        dataset_name = st.selectbox("数据集选择", list_datasets(), index=0)
        sample_count = st.slider("总样本数", min_value=20, max_value=500, value=120, step=10)
        noise_strength = st.slider("噪声强度", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        random_seed = st.number_input("随机种子", min_value=0, max_value=99999, value=42, step=1)
        shuffle_data = st.checkbox("是否打乱数据", value=True)

        dataset = generate_dataset(
            dataset_name=dataset_name,
            sample_count=int(sample_count),
            noise_strength=float(noise_strength),
            random_seed=int(random_seed),
            shuffle_data=shuffle_data,
        )

        st.markdown("---")
        st.subheader("模型选择")
        model_options = list_models(dataset.task_type)
        model_name = st.selectbox("模型", [config.name for config in model_options], index=0)

        degree = 3
        knn_k = 5
        if model_name == "多项式回归":
            degree = st.slider("多项式次数 degree", min_value=2, max_value=10, value=3, step=1)
        if model_name == "KNN":
            knn_k = st.slider("KNN 的 k 值", min_value=1, max_value=15, value=5, step=1)

        use_l2_regularization = st.checkbox("启用 L2 正则", value=False)
        regularization_strength = 1.0
        if use_l2_regularization:
            regularization_strength = st.slider("正则强度", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

        st.markdown("---")
        st.subheader("评估方法")
        evaluation_method = st.selectbox(
            "评估方法选择",
            ["留出法", "重复留出法", "k折交叉验证", "留一法", "自助法"],
            index=0,
        )

        train_ratio = 0.7
        k_folds = 5
        bootstrap_rounds = 30
        repeated_holdout_rounds = 10

        if evaluation_method in {"留出法", "重复留出法"}:
            train_ratio = st.slider("训练集比例", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        if evaluation_method == "重复留出法":
            repeated_holdout_rounds = st.slider("重复次数", min_value=5, max_value=40, value=10, step=1)
        if evaluation_method == "k折交叉验证":
            max_k = min(10, len(dataset.X))
            k_folds = st.slider("k 值", min_value=2, max_value=max_k, value=min(5, max_k), step=1)
        if evaluation_method == "自助法":
            bootstrap_rounds = st.slider("Bootstrap 重采样次数", min_value=10, max_value=200, value=30, step=5)

        use_fixed_test_set = st.checkbox("启用固定测试集（教学加分项）", value=False)
        fixed_test_ratio = 0.2
        if use_fixed_test_set:
            fixed_test_ratio = st.slider("固定测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    artifacts = evaluate_experiment(
        X=dataset.X,
        y=dataset.y,
        task_type=dataset.task_type,
        model_name=model_name,
        evaluation_method=evaluation_method,
        degree=degree,
        knn_k=knn_k,
        use_l2_regularization=use_l2_regularization,
        regularization_strength=regularization_strength,
        train_ratio=train_ratio,
        k_folds=k_folds,
        bootstrap_rounds=bootstrap_rounds,
        repeated_holdout_rounds=repeated_holdout_rounds,
        random_seed=int(random_seed),
        shuffle_data=shuffle_data,
        use_fixed_test_set=use_fixed_test_set,
        fixed_test_ratio=fixed_test_ratio,
    )

    primary_name = primary_metric_name(dataset.task_type)
    round_options = [record["round_id"] for record in artifacts.round_records]
    selected_round_id = st.slider("查看第几轮划分", min_value=min(round_options), max_value=max(round_options), value=min(round_options))
    selected_record = next(record for record in artifacts.round_records if record["round_id"] == selected_round_id)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("任务类型", "分类" if dataset.task_type == "classification" else "回归")
    metric_col2.metric("主指标", primary_name)
    metric_col3.metric("验证均值", f"{artifacts.summary['valid_mean']:.4f}")
    metric_col4.metric("验证标准差", f"{artifacts.summary['valid_std']:.4f}")

    info_col, summary_col = st.columns([1.3, 1.0])
    with info_col:
        st.subheader("数据集与方法简介")
        st.write(dataset.description)
        st.write(artifacts.method_explanation)
        st.info(
            "训练集用于拟合模型参数；验证集用于比较模型或调参；测试集应该尽量只在最后使用，"
            "否则你会把测试集也“学进去”，失去对泛化性能的公正估计。"
        )

    with summary_col:
        st.subheader("当前实验概览")
        st.write(f"- 当前模型：`{model_name}`")
        st.write(f"- 当前评估方法：`{evaluation_method}`")
        st.write(f"- 总样本数：`{len(dataset.X)}`")
        st.write(f"- 开发集样本数：`{len(artifacts.development_indices)}`")
        if artifacts.test_indices is not None:
            st.write(f"- 固定测试集样本数：`{len(artifacts.test_indices)}`")
        st.write(f"- 轮次数：`{artifacts.summary['n_rounds']}`")
        st.write(f"- 当前稳定性判断：{artifacts.summary['stability_hint']}")

    st.subheader("主图：本轮数据是如何被划分的")
    st.pyplot(
        plot_split_visualization(
            X=dataset.X[artifacts.development_indices],
            y=dataset.y[artifacts.development_indices],
            task_type=dataset.task_type,
            model_name=model_name,
            degree=degree,
            knn_k=knn_k,
            use_l2_regularization=use_l2_regularization,
            regularization_strength=regularization_strength,
            train_indices=selected_record["train_indices"],
            valid_indices=selected_record["valid_indices"],
            title=f"{evaluation_method} 第 {selected_round_id} 轮划分示意",
        ),
        clear_figure=True,
    )

    split_col, result_col = st.columns([1.1, 1.1])
    with split_col:
        st.subheader("本轮划分细节")
        st.write(f"- 训练样本数：`{selected_record['train_size']}`")
        st.write(f"- 验证样本数：`{selected_record['valid_size']}`")
        st.write(f"- 训练集去重后样本数：`{selected_record['unique_train_size']}`")
        duplicate_counter = selected_record["duplicate_counter"]
        if duplicate_counter:
            duplicate_text = ", ".join([f"{index}:{count}次" for index, count in list(duplicate_counter.items())[:8]])
            st.write(f"- 训练集中重复采样示例：`{duplicate_text}`")
        else:
            st.write("- 当前训练集没有重复采样样本")
        st.write(f"- 训练索引预览：`{format_index_preview(selected_record['train_indices'])}`")
        st.write(f"- 验证索引预览：`{format_index_preview(selected_record['valid_indices'])}`")
        st.write(f"- 解释：{selected_record['note']}")

    with result_col:
        st.subheader("本轮得分")
        train_metrics = selected_record["train_metrics"]
        valid_metrics = selected_record["valid_metrics"]
        test_metrics = selected_record["test_metrics"]
        st.write(f"- 训练集：`{train_metrics}`")
        st.write(f"- 验证集：`{valid_metrics if valid_metrics else '本轮没有可用验证样本'}`")
        st.write(f"- 固定测试集：`{test_metrics if test_metrics else '未启用固定测试集'}`")
        st.write(
            "训练分数通常会比验证分数乐观，因为模型已经见过训练集；"
            "我们真正关心的是模型在未见样本上的表现，也就是泛化性能。"
        )

    st.subheader("评估结果面板")
    st.dataframe(artifacts.round_table, use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_score_distribution(artifacts.round_table, primary_name), clear_figure=True)
    with chart_col2:
        comparison_table = build_method_summary_table(
            X=dataset.X,
            y=dataset.y,
            task_type=dataset.task_type,
            model_name=model_name,
            degree=degree,
            knn_k=knn_k,
            use_l2_regularization=use_l2_regularization,
            regularization_strength=regularization_strength,
            train_ratio=train_ratio,
            k_folds=k_folds,
            bootstrap_rounds=bootstrap_rounds,
            repeated_holdout_rounds=repeated_holdout_rounds,
            random_seed=int(random_seed),
            shuffle_data=shuffle_data,
            use_fixed_test_set=use_fixed_test_set,
            fixed_test_ratio=fixed_test_ratio,
        )
        st.pyplot(
            plot_method_comparison(comparison_table, primary_name, dataset.task_type),
            clear_figure=True,
        )
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    st.subheader("方法解释区")
    st.markdown(
        f"""
1. **为什么不同评估方法会得到不同结果？**  
   因为它们看到的训练集与验证集并不完全一样。样本少、噪声大、模型不稳定时，这种差异会更明显。

2. **为什么样本量少时，单次切分可能不稳定？**  
   如果一次划分恰好把“难样本”几乎全分到验证集，结果就会偏悲观；反过来则可能偏乐观。

3. **为什么交叉验证通常更稳健？**  
   因为每个样本都参与过训练，也都轮流做过验证，估计不再依赖某一次偶然切分。

4. **为什么 Bootstrap 会出现重复样本？**  
   因为它是有放回采样。同一个样本被抽中多次是正常现象，同时也会有一些样本一次都没被抽到。
"""
    )

    st.subheader("当前实验说明")
    st.success(artifacts.experiment_note)

    st.markdown("---")
    st.markdown(
        """
**教学提醒**

- 评估方法的目标不是把当前数据“背下来”，而是尽量估计模型面对未来新样本时的泛化性能。
- 如果你反复查看测试集结果并据此调参，那么测试集就不再“独立”，最终得到的测试分数会偏乐观。
- 小样本时建议多观察“均值 + 标准差”，不要只盯住某一次偶然很高或很低的结果。
"""
    )


if __name__ == "__main__":
    main()
