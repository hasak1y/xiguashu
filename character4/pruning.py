from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd

from tree_node import TreeNode, count_tree_nodes, tree_depth


def predict_sample(row: pd.Series, nodes: Dict[int, TreeNode], root_id: int) -> Any:
    node = nodes[root_id]
    while not node.is_leaf and node.children:
        next_child: Optional[int] = None
        for child_id in node.children:
            child = nodes[child_id]
            if child.edge_label.startswith("in {"):
                values = child.edge_label[4:-1].split(", ")
                if str(row[node.split_feature]) in values:
                    next_child = child_id
                    break
            elif child.edge_label.startswith("<= "):
                threshold = float(child.edge_label.replace("<= ", ""))
                if float(row[node.split_feature]) <= threshold:
                    next_child = child_id
                    break
            elif child.edge_label.startswith("> "):
                threshold = float(child.edge_label.replace("> ", ""))
                if float(row[node.split_feature]) > threshold:
                    next_child = child_id
                    break
            elif str(row[node.split_feature]) == child.edge_label:
                next_child = child_id
                break
        if next_child is None:
            return node.majority_label
        node = nodes[next_child]
    return node.prediction


def accuracy(df: pd.DataFrame, target: str, nodes: Dict[int, TreeNode], root_id: int) -> float:
    if df.empty:
        return 0.0
    preds = df.apply(lambda row: predict_sample(row, nodes, root_id), axis=1)
    return float((preds == df[target]).mean())


def temporary_leaf_accuracy(
    df_val: pd.DataFrame,
    target: str,
    nodes: Dict[int, TreeNode],
    root_id: int,
    node_id: int,
) -> float:
    copied_nodes = deepcopy(nodes)
    prune_to_leaf(copied_nodes, node_id)
    return accuracy(df_val, target, copied_nodes, root_id)


def prune_to_leaf(nodes: Dict[int, TreeNode], node_id: int) -> None:
    node = nodes[node_id]
    for child_id in list(node.children):
        if child_id in nodes:
            prune_to_leaf(nodes, child_id)
    node.children = []
    node.is_leaf = True
    node.prediction = node.majority_label
    node.split_feature = None
    node.split_rule = None
    node.split_summary = []


def collect_internal_nodes_postorder(nodes: Dict[int, TreeNode], root_id: int) -> List[int]:
    order: List[int] = []

    def dfs(node_id: int) -> None:
        node = nodes[node_id]
        for child_id in node.children:
            dfs(child_id)
        if node.children:
            order.append(node_id)

    if root_id in nodes:
        dfs(root_id)
    return order


def evaluate_pruning_step(
    nodes: Dict[int, TreeNode],
    node_id: int,
    df_val: pd.DataFrame,
    target: str,
    root_id: int,
) -> Dict[str, Any]:
    keep_acc = accuracy(df_val, target, nodes, root_id)
    prune_acc = temporary_leaf_accuracy(df_val, target, nodes, root_id, node_id)
    decision = "prune" if prune_acc >= keep_acc else "keep"
    return {
        "node_id": node_id,
        "keep_accuracy": keep_acc,
        "prune_accuracy": prune_acc,
        "decision": decision,
    }


def summary_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    nodes: Dict[int, TreeNode],
    root_id: int,
) -> Dict[str, Any]:
    return {
        "node_count": count_tree_nodes(nodes, root_id),
        "depth": tree_depth(nodes, root_id),
        "train_accuracy": accuracy(train_df, target, nodes, root_id),
        "val_accuracy": accuracy(val_df, target, nodes, root_id) if not val_df.empty else None,
    }
