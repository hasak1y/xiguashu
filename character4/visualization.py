from __future__ import annotations

import json
from typing import Dict, Optional

import graphviz
import pandas as pd

from tree_node import TreeNode


def node_label(node: TreeNode) -> str:
    counts = "\\n".join(f"{k}:{v}" for k, v in node.class_counts.items())
    parts = [
        f"Node {node.node_id}",
        f"samples={len(node.sample_indices)}",
        f"depth={node.depth}",
        f"majority={node.majority_label}",
        f"classes:\\n{counts}",
    ]
    if node.is_leaf:
        parts.append(f"leaf={node.prediction}")
    elif node.split_feature:
        parts.append(f"split={node.split_feature}")
        if node.split_rule:
            parts.append(node.split_rule)
    return "\\n".join(parts)


def build_tree_graph(
    nodes: Dict[int, TreeNode],
    root_id: int,
    current_node_id: Optional[int] = None,
    highlight_prune_id: Optional[int] = None,
) -> graphviz.Digraph:
    graph = graphviz.Digraph()
    graph.attr(rankdir="TB", splines="polyline")
    if root_id not in nodes:
        return graph

    for node_id, node in nodes.items():
        fillcolor = "#E8F3FF"
        shape = "ellipse"
        if node.is_leaf:
            fillcolor = "#E9F7EF"
            shape = "box"
        if current_node_id == node_id:
            fillcolor = "#FFE08A"
        if highlight_prune_id == node_id:
            fillcolor = "#F8B4B4"
        graph.node(
            str(node_id),
            label=node_label(node),
            shape=shape,
            style="rounded,filled",
            fillcolor=fillcolor,
            color="#4B5563",
        )
        for child_id in node.children:
            child = nodes[child_id]
            graph.edge(str(node_id), str(child_id), label=child.edge_label)
    return graph


def tree_to_json(nodes: Dict[int, TreeNode], root_id: int) -> str:
    def build(node_id: int):
        node = nodes[node_id]
        return {
            "node_id": node.node_id,
            "edge_label": node.edge_label,
            "is_leaf": node.is_leaf,
            "prediction": node.prediction,
            "majority_label": node.majority_label,
            "split_feature": node.split_feature,
            "split_rule": node.split_rule,
            "class_counts": node.class_counts,
            "children": [build(child_id) for child_id in node.children],
        }

    if root_id not in nodes:
        return "{}"
    return json.dumps(build(root_id), ensure_ascii=False, indent=2)


def dataframe_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
