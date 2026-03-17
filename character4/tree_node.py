from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeNode:
    """Simple decision tree node used by the teaching visualizer."""

    node_id: int
    depth: int
    sample_indices: List[int]
    parent_id: Optional[int] = None
    edge_label: str = "root"
    available_features: List[str] = field(default_factory=list)
    is_leaf: bool = False
    prediction: Optional[Any] = None
    majority_label: Optional[Any] = None
    split_feature: Optional[str] = None
    split_score: Optional[float] = None
    split_rule: Optional[str] = None
    metric_value: Optional[float] = None
    class_counts: Dict[Any, int] = field(default_factory=dict)
    children: List[int] = field(default_factory=list)
    split_summary: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "sample_indices": self.sample_indices,
            "parent_id": self.parent_id,
            "edge_label": self.edge_label,
            "available_features": self.available_features,
            "is_leaf": self.is_leaf,
            "prediction": self.prediction,
            "majority_label": self.majority_label,
            "split_feature": self.split_feature,
            "split_score": self.split_score,
            "split_rule": self.split_rule,
            "metric_value": self.metric_value,
            "class_counts": self.class_counts,
            "children": self.children,
            "split_summary": self.split_summary,
            "stop_reason": self.stop_reason,
        }


def count_tree_nodes(nodes: Dict[int, TreeNode], root_id: int) -> int:
    if root_id not in nodes:
        return 0
    root = nodes[root_id]
    return 1 + sum(count_tree_nodes(nodes, child_id) for child_id in root.children)


def tree_depth(nodes: Dict[int, TreeNode], root_id: int) -> int:
    if root_id not in nodes:
        return 0
    root = nodes[root_id]
    if not root.children:
        return root.depth
    return max(tree_depth(nodes, child_id) for child_id in root.children)
