from __future__ import annotations

from copy import deepcopy

import pandas as pd
import streamlit as st

from builder import BuildConfig, DecisionTreeBuilder
from datasets import builtin_datasets, load_uploaded_csv
from pruning import evaluate_pruning_step, prune_to_leaf, summary_stats
from utils import infer_default_features, train_val_split
from visualization import build_tree_graph, dataframe_download_csv, tree_to_json


st.set_page_config(page_title="决策树教学可视化", layout="wide")


ALGO_DESC = {
    "ID3": "ID3：使用信息增益选择划分特征，适合演示“熵减少最多”的思路。",
    "C4.5": "C4.5：在信息增益基础上引入信息增益率，缓解取值数多的特征偏好。",
    "CART": "CART：使用基尼指数，构建二叉树；离散特征会做二分，数值特征可做阈值切分。",
}


def init_session() -> None:
    defaults = {
        "builder": None,
        "selected_node_id": None,
        "post_prune_steps": [],
        "post_prune_index": 0,
        "highlight_prune_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_builder() -> DecisionTreeBuilder | None:
    return st.session_state.get("builder")


def reset_builder() -> None:
    st.session_state["builder"] = None
    st.session_state["selected_node_id"] = None
    st.session_state["post_prune_steps"] = []
    st.session_state["post_prune_index"] = 0
    st.session_state["highlight_prune_id"] = None


def prepare_builder(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    algorithm: str,
    pruning_mode: str,
    val_ratio: float,
    max_depth: int,
    min_samples_split: int,
    min_gain: float,
    show_formulas: bool,
) -> None:
    train_df, val_df = train_val_split(df, val_ratio if pruning_mode != "不剪枝" else 0.0)
    config = BuildConfig(
        algorithm=algorithm,
        target=target,
        feature_columns=features,
        pruning_mode=pruning_mode,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_gain=min_gain,
        show_formulas=show_formulas,
    )
    st.session_state["builder"] = DecisionTreeBuilder(train_df, val_df, config)
    st.session_state["selected_node_id"] = 0
    st.session_state["post_prune_steps"] = []
    st.session_state["post_prune_index"] = 0
    st.session_state["highlight_prune_id"] = None


def render_sidebar() -> tuple[pd.DataFrame, str, list[str], dict]:
    datasets = builtin_datasets()

    st.sidebar.title("控制栏")
    algorithm = st.sidebar.selectbox("算法模式", ["ID3", "C4.5", "CART"])
    dataset_choice = st.sidebar.selectbox("数据集选择", list(datasets.keys()) + ["上传 CSV"])

    if dataset_choice == "上传 CSV":
        uploaded_file = st.sidebar.file_uploader("上传 CSV", type=["csv"])
        if uploaded_file is not None:
            df = load_uploaded_csv(uploaded_file)
            default_target = df.columns[-1]
        else:
            df, default_target = datasets["西瓜数据集 2.0（离散版）"]
    else:
        df, default_target = datasets[dataset_choice]

    target = st.sidebar.selectbox("目标列", df.columns.tolist(), index=df.columns.tolist().index(default_target))
    default_features = infer_default_features(df, target)
    features = st.sidebar.multiselect("特征列选择", [c for c in df.columns if c != target], default=default_features)

    pruning_mode = st.sidebar.radio("是否启用剪枝", ["不剪枝", "预剪枝", "后剪枝"])
    val_ratio = st.sidebar.slider("训练集 / 验证集划分比例（验证集）", 0.1, 0.5, 0.3, 0.05) if pruning_mode != "不剪枝" else 0.0
    max_depth = st.sidebar.slider("最大深度", 1, 8, 5)
    min_samples_split = st.sidebar.slider("最小样本数", 1, 6, 2)
    default_gain = 0.0 if algorithm in {"ID3", "C4.5"} else 0.5
    min_gain = st.sidebar.number_input(
        "最小增益 / 最小增益率 / 最大允许基尼",
        min_value=0.0,
        max_value=1.0,
        value=default_gain,
        step=0.01,
    )
    show_formulas = st.sidebar.checkbox("显示详细计算提示", value=True)

    if st.sidebar.button("Reset"):
        reset_builder()
        st.rerun()

    options = {
        "algorithm": algorithm,
        "pruning_mode": pruning_mode,
        "val_ratio": val_ratio,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_gain": min_gain,
        "show_formulas": show_formulas,
    }
    return df, target, features, options


def render_tree_area(builder: DecisionTreeBuilder) -> None:
    st.subheader("A. 当前树结构可视化")
    current_node_id = builder.last_step.node_id if builder.last_step else builder.root_id
    graph = build_tree_graph(
        builder.nodes,
        builder.root_id,
        current_node_id=current_node_id,
        highlight_prune_id=st.session_state.get("highlight_prune_id"),
    )
    st.graphviz_chart(graph)

    json_text = tree_to_json(builder.nodes, builder.root_id)
    st.download_button(
        "导出当前树结构 JSON",
        data=json_text.encode("utf-8"),
        file_name="tree_structure.json",
        mime="application/json",
    )


def render_node_detail(builder: DecisionTreeBuilder) -> None:
    st.subheader("B. 当前节点详细信息")
    node_ids = sorted(builder.nodes.keys())
    selected_node_id = st.selectbox(
        "选择要查看的节点",
        node_ids,
        index=node_ids.index(st.session_state["selected_node_id"]) if st.session_state["selected_node_id"] in node_ids else 0,
    )
    st.session_state["selected_node_id"] = selected_node_id
    detail = builder.node_detail(selected_node_id)

    col1, col2, col3 = st.columns(3)
    metric_name = "熵" if builder.config.algorithm in {"ID3", "C4.5"} else "基尼"
    col1.metric("样本数", detail["sample_count"])
    col2.metric("多数类", str(detail["majority_label"]))
    col3.metric(metric_name, f"{detail['metric_value']:.4f}")
    st.caption(f"深度：{detail['depth']}；可选特征：{', '.join(detail['available_features']) or '无'}")
    st.dataframe(detail["samples"], use_container_width=True)


def render_step_area(builder: DecisionTreeBuilder) -> None:
    st.subheader("C. 候选特征评分表")
    last_step = builder.last_step
    if last_step and last_step.score_rows:
        st.dataframe(pd.DataFrame(last_step.score_rows), use_container_width=True)
        if builder.config.show_formulas:
            st.info(ALGO_DESC[builder.config.algorithm])
    else:
        st.info("点击 Next Step 或 Build Full Tree 后，这里会显示当前处理节点的候选特征评分。")

    st.subheader("D. 分裂结果展示")
    if last_step and last_step.split_rows:
        st.dataframe(pd.DataFrame(last_step.split_rows), use_container_width=True)
        if last_step.selected_feature:
            st.success(f"本步选择了特征：{last_step.selected_feature}")
    else:
        st.info("当前还没有分裂结果。")

    if last_step and last_step.pruning_detail and last_step.pruning_detail.get("enabled"):
        detail = last_step.pruning_detail
        st.subheader("剪枝决策展示")
        col1, col2 = st.columns(2)
        col1.metric("不分裂时验证集准确率", f"{detail['before_accuracy']:.3f}")
        col2.metric("尝试分裂后验证集准确率", f"{detail['after_accuracy']:.3f}")
        st.caption(f"决策：{'继续分裂' if detail['decision'] == 'split' else '停止并剪枝'}")


def render_logs(builder: DecisionTreeBuilder) -> None:
    st.subheader("E. 日志 / 时间线")
    for line in builder.logs[::-1]:
        st.markdown(f"- {line}")


def prepare_post_pruning(builder: DecisionTreeBuilder) -> None:
    from pruning import collect_internal_nodes_postorder

    st.session_state["post_prune_steps"] = collect_internal_nodes_postorder(builder.nodes, builder.root_id)
    st.session_state["post_prune_index"] = 0
    st.session_state["highlight_prune_id"] = None


def do_next_post_prune(builder: DecisionTreeBuilder) -> None:
    steps = st.session_state["post_prune_steps"]
    idx = st.session_state["post_prune_index"]
    if idx >= len(steps):
        return

    if builder.before_prune_stats is None:
        builder.before_prune_stats = summary_stats(
            builder.train_df, builder.val_df, builder.config.target, deepcopy(builder.nodes), builder.root_id
        )

    node_id = steps[idx]
    st.session_state["highlight_prune_id"] = node_id
    detail = evaluate_pruning_step(builder.nodes, node_id, builder.val_df, builder.config.target, builder.root_id)
    message = (
        f"后剪枝检查节点 {node_id}：保留子树验证集准确率 {detail['keep_accuracy']:.3f}，"
        f"剪成叶节点后为 {detail['prune_accuracy']:.3f}，决策："
        f"{'剪枝' if detail['decision'] == 'prune' else '保留'}。"
    )
    builder.logs.append(message)
    if detail["decision"] == "prune":
        prune_to_leaf(builder.nodes, node_id)

    st.session_state["post_prune_index"] = idx + 1
    if st.session_state["post_prune_index"] >= len(steps):
        builder.after_prune_stats = summary_stats(
            builder.train_df, builder.val_df, builder.config.target, builder.nodes, builder.root_id
        )


def main() -> None:
    init_session()
    st.title("《机器学习（周志华）》第四章 决策树教学可视化")

    df, target, features, options = render_sidebar()

    st.markdown(f"**当前模式说明**：{ALGO_DESC[options['algorithm']]}")
    st.caption("目标是教学演示：你可以一步一步看见当前节点、候选特征评分、分裂结果，以及剪枝判断。")

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.subheader("数据预览")
        st.dataframe(df, use_container_width=True, height=260)
        st.download_button(
            "导出当前数据 CSV",
            data=dataframe_download_csv(df),
            file_name="dataset.csv",
            mime="text/csv",
        )

        if st.button("Initialize / Rebuild"):
            prepare_builder(
                df=df,
                target=target,
                features=features,
                algorithm=options["algorithm"],
                pruning_mode=options["pruning_mode"],
                val_ratio=options["val_ratio"],
                max_depth=options["max_depth"],
                min_samples_split=options["min_samples_split"],
                min_gain=options["min_gain"],
                show_formulas=options["show_formulas"],
            )
            st.rerun()

        builder = current_builder()
        if builder is not None:
            col_a, col_b = st.columns(2)
            if col_a.button("Next Step"):
                builder.next_step()
                st.rerun()
            if col_b.button("Build Full Tree"):
                builder.build_full_tree()
                st.rerun()

            if builder.config.pruning_mode == "后剪枝" and not builder.queue:
                if st.button("准备后剪枝"):
                    prepare_post_pruning(builder)
                    st.rerun()
                if st.session_state["post_prune_steps"]:
                    col_c, col_d = st.columns(2)
                    if col_c.button("Next Prune Step"):
                        do_next_post_prune(builder)
                        st.rerun()
                    if col_d.button("Execute All Pruning"):
                        while st.session_state["post_prune_index"] < len(st.session_state["post_prune_steps"]):
                            do_next_post_prune(builder)
                        st.rerun()

    with col_right:
        builder = current_builder()
        if builder is None:
            st.info("先在左侧设置参数，然后点击 Initialize / Rebuild。")
            return

        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("训练集样本数", len(builder.train_df))
        stats_col2.metric("验证集样本数", len(builder.val_df))
        stats_col3.metric("待处理节点数", len(builder.queue))

        render_tree_area(builder)
        render_node_detail(builder)
        render_step_area(builder)

        if builder.before_prune_stats and builder.after_prune_stats:
            st.subheader("剪枝前后对比")
            comparison = pd.DataFrame(
                [
                    {"指标": "节点数", "剪枝前": builder.before_prune_stats["node_count"], "剪枝后": builder.after_prune_stats["node_count"]},
                    {"指标": "深度", "剪枝前": builder.before_prune_stats["depth"], "剪枝后": builder.after_prune_stats["depth"]},
                    {"指标": "训练集准确率", "剪枝前": round(builder.before_prune_stats["train_accuracy"], 3), "剪枝后": round(builder.after_prune_stats["train_accuracy"], 3)},
                    {
                        "指标": "验证集准确率",
                        "剪枝前": None if builder.before_prune_stats["val_accuracy"] is None else round(builder.before_prune_stats["val_accuracy"], 3),
                        "剪枝后": None if builder.after_prune_stats["val_accuracy"] is None else round(builder.after_prune_stats["val_accuracy"], 3),
                    },
                ]
            )
            st.dataframe(comparison, use_container_width=True)

        render_logs(builder)


if __name__ == "__main__":
    main()
