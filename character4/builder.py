from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from metrics import entropy, gini, label_counts, majority_label
from pruning import accuracy
from splitters import evaluate_c45, evaluate_cart, evaluate_id3, split_cart_subset, split_discrete
from tree_node import TreeNode


@dataclass
class BuildConfig:
    algorithm: str
    target: str
    feature_columns: List[str]
    pruning_mode: str
    max_depth: int
    min_samples_split: int
    min_gain: float
    show_formulas: bool = False


@dataclass
class StepResult:
    step_index: int
    action: str
    message: str
    node_id: Optional[int] = None
    selected_feature: Optional[str] = None
    score_rows: List[Dict[str, Any]] = field(default_factory=list)
    split_rows: List[Dict[str, Any]] = field(default_factory=list)
    node_snapshot: Dict[str, Any] = field(default_factory=dict)
    pruning_detail: Optional[Dict[str, Any]] = None


class DecisionTreeBuilder:
    """Stateful step-by-step builder for teaching decision trees."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: BuildConfig,
    ) -> None:
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.config = config
        self.nodes: Dict[int, TreeNode] = {}
        self.root_id = 0
        self.next_node_id = 1
        self.step_counter = 0
        self.queue: Deque[int] = deque()
        self.logs: List[str] = []
        self.last_step: Optional[StepResult] = None
        self.post_prune_queue: List[int] = []
        self.before_prune_stats: Optional[Dict[str, Any]] = None
        self.after_prune_stats: Optional[Dict[str, Any]] = None
        self._init_root()

    def _init_root(self) -> None:
        root = self._create_node(
            sample_indices=self.train_df.index.tolist(),
            depth=0,
            parent_id=None,
            edge_label="root",
            available_features=self.config.feature_columns.copy(),
        )
        self.root_id = root.node_id
        self.nodes[root.node_id] = root
        self.queue.append(root.node_id)
        self._log(
            f"Step 0: 创建根节点，包含 {len(root.sample_indices)} 个样本，"
            f"类别分布为 {self._format_counts(root.class_counts)}。"
        )

    def _format_counts(self, counts: Dict[Any, int]) -> str:
        return " / ".join(f"{label}:{count}" for label, count in counts.items())

    def _create_node(
        self,
        sample_indices: List[int],
        depth: int,
        parent_id: Optional[int],
        edge_label: str,
        available_features: List[str],
    ) -> TreeNode:
        labels = self.train_df.loc[sample_indices, self.config.target]
        counts = label_counts(labels)
        metric_value = entropy(labels) if self.config.algorithm in {"ID3", "C4.5"} else gini(labels)
        return TreeNode(
            node_id=(0 if parent_id is None and not self.nodes else self.next_node_id),
            depth=depth,
            sample_indices=sample_indices,
            parent_id=parent_id,
            edge_label=edge_label,
            available_features=available_features,
            majority_label=majority_label(labels),
            metric_value=metric_value,
            class_counts=counts,
        )

    def _allocate_child_id(self) -> int:
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def _stop_reason(self, node: TreeNode) -> Optional[str]:
        labels = self.train_df.loc[node.sample_indices, self.config.target]
        if len(node.sample_indices) == 0:
            return "空节点"
        if len(set(labels.tolist())) == 1:
            return "节点已纯"
        if not node.available_features:
            return "没有可用特征"
        if node.depth >= self.config.max_depth:
            return "达到最大深度"
        if len(node.sample_indices) < self.config.min_samples_split:
            return "样本数不足以继续划分"
        return None

    def _make_leaf(self, node: TreeNode, reason: str) -> StepResult:
        node.is_leaf = True
        node.prediction = node.majority_label
        node.stop_reason = reason
        self.step_counter += 1
        message = (
            f"Step {self.step_counter}: 节点 {node.node_id} 停止划分，原因：{reason}；"
            f"预测类别设为多数类 {node.prediction}。"
        )
        self._log(message)
        result = StepResult(
            step_index=self.step_counter,
            action="make_leaf",
            message=message,
            node_id=node.node_id,
            node_snapshot=self.node_detail(node.node_id),
        )
        self.last_step = result
        return result

    def node_detail(self, node_id: int) -> Dict[str, Any]:
        node = self.nodes[node_id]
        samples = self.train_df.loc[node.sample_indices]
        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "sample_count": len(node.sample_indices),
            "class_counts": node.class_counts,
            "metric_value": node.metric_value,
            "available_features": node.available_features,
            "samples": samples,
            "majority_label": node.majority_label,
            "prediction": node.prediction,
            "split_feature": node.split_feature,
            "is_leaf": node.is_leaf,
        }

    def _evaluate_node(self, node: TreeNode) -> Dict[str, Any]:
        features = node.available_features
        if self.config.algorithm == "ID3":
            return evaluate_id3(self.train_df, node.sample_indices, self.config.target, features)
        if self.config.algorithm == "C4.5":
            return evaluate_c45(self.train_df, node.sample_indices, self.config.target, features)
        return evaluate_cart(self.train_df, node.sample_indices, self.config.target, features)

    def _score_rows(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if self.config.algorithm == "ID3":
            best_feature = evaluation["best"]["feature"]
            for item in evaluation["scores"]:
                rows.append(
                    {
                        "特征": item["feature"],
                        "当前节点熵": round(item["base_entropy"], 4),
                        "条件熵": round(item["conditional_entropy"], 4),
                        "信息增益": round(item["information_gain"], 4),
                        "是否选中": "是" if item["feature"] == best_feature else "",
                    }
                )
        elif self.config.algorithm == "C4.5":
            best_feature = evaluation["best"]["feature"]
            for item in evaluation["scores"]:
                rows.append(
                    {
                        "特征": item["feature"],
                        "信息增益": round(item["information_gain"], 4),
                        "SplitInfo": round(item["split_info"], 4),
                        "信息增益率": round(item["gain_ratio"], 4),
                        "是否高于平均增益": "是" if item["information_gain"] >= evaluation["avg_gain"] else "否",
                        "是否选中": "是" if item["feature"] == best_feature else "",
                    }
                )
        else:
            best_feature = evaluation["best"]["feature"]
            best_rule = evaluation["best"]["best_candidate"]["rule"]
            for item in evaluation["scores"]:
                rows.append(
                    {
                        "特征": item["feature"],
                        "最优二分方式": item["best_candidate"]["rule"],
                        "加权基尼": round(item["weighted_gini"], 4),
                        "是否选中": "是" if item["feature"] == best_feature and item["best_candidate"]["rule"] == best_rule else "",
                    }
                )
        return rows

    def _split_rows(self, split_map: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for branch, indices in split_map.items():
            labels = self.train_df.loc[indices, self.config.target]
            counts = label_counts(labels)
            rows.append(
                {
                    "分支": branch,
                    "样本数": len(indices),
                    "类别分布": self._format_counts(counts),
                    "是否纯": "是" if len(counts) <= 1 else "否",
                    "是否继续递归": "是" if len(counts) > 1 and len(indices) >= self.config.min_samples_split else "否",
                }
            )
        return rows

    def _should_stop_by_score(self, evaluation: Dict[str, Any]) -> bool:
        if self.config.algorithm == "CART":
            return evaluation["best"]["weighted_gini"] >= self.config.min_gain
        if self.config.algorithm == "ID3":
            return evaluation["best"]["information_gain"] <= self.config.min_gain
        return evaluation["best"]["gain_ratio"] <= self.config.min_gain

    def _pre_prune_decision(
        self,
        node: TreeNode,
        split_map: Dict[str, List[int]],
        selected_feature: str,
    ) -> Dict[str, Any]:
        if self.val_df.empty:
            return {"enabled": False}

        before_nodes = self._leaf_override_nodes(node.node_id)
        before_acc = accuracy(self.val_df, self.config.target, before_nodes, self.root_id)

        simulated_nodes = self._simulate_split(node, split_map, selected_feature)
        after_acc = accuracy(self.val_df, self.config.target, simulated_nodes, self.root_id)
        decision = "split" if after_acc > before_acc else "stop"
        return {
            "enabled": True,
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "decision": decision,
        }

    def _leaf_override_nodes(self, node_id: int) -> Dict[int, TreeNode]:
        import copy

        copied = copy.deepcopy(self.nodes)
        copied[node_id].is_leaf = True
        copied[node_id].prediction = copied[node_id].majority_label
        copied[node_id].children = []
        copied[node_id].split_feature = None
        return copied

    def _simulate_split(
        self,
        node: TreeNode,
        split_map: Dict[str, List[int]],
        selected_feature: str,
    ) -> Dict[int, TreeNode]:
        import copy

        copied = copy.deepcopy(self.nodes)
        sim_node = copied[node.node_id]
        sim_node.split_feature = selected_feature
        sim_node.children = []
        for branch, indices in split_map.items():
            labels = self.train_df.loc[indices, self.config.target]
            child = TreeNode(
                node_id=max(copied.keys()) + 1 + len(sim_node.children),
                depth=node.depth + 1,
                sample_indices=indices,
                parent_id=node.node_id,
                edge_label=branch,
                available_features=[f for f in node.available_features if f != selected_feature],
                is_leaf=True,
                prediction=majority_label(labels),
                majority_label=majority_label(labels),
                class_counts=label_counts(labels),
            )
            copied[child.node_id] = child
            sim_node.children.append(child.node_id)
        return copied

    def next_step(self) -> Optional[StepResult]:
        if not self.queue:
            return None

        node_id = self.queue.popleft()
        node = self.nodes[node_id]
        stop_reason = self._stop_reason(node)
        if stop_reason:
            return self._make_leaf(node, stop_reason)

        evaluation = self._evaluate_node(node)
        if self._should_stop_by_score(evaluation):
            return self._make_leaf(node, "最优划分指标未达到阈值")

        if self.config.algorithm == "CART":
            best_feature = evaluation["best"]["feature"]
            best_candidate = evaluation["best"]["best_candidate"]
            split_map = split_cart_subset(self.train_df, node.sample_indices, best_feature, best_candidate)
            reason_text = f"条件基尼最小，为 {best_candidate['weighted_gini']:.4f}"
            score_value = best_candidate["weighted_gini"]
            split_rule = best_candidate["rule"]
        else:
            best_feature = evaluation["best"]["feature"]
            split_map = split_discrete(self.train_df, node.sample_indices, best_feature)
            if self.config.algorithm == "ID3":
                reason_text = f"信息增益最大，为 {evaluation['best']['information_gain']:.4f}"
                score_value = evaluation["best"]["information_gain"]
            else:
                reason_text = f"信息增益率最大，为 {evaluation['best']['gain_ratio']:.4f}"
                score_value = evaluation["best"]["gain_ratio"]
            split_rule = None

        pruning_detail = None
        if self.config.pruning_mode == "预剪枝":
            pruning_detail = self._pre_prune_decision(node, split_map, best_feature)
            if pruning_detail.get("enabled") and pruning_detail["decision"] == "stop":
                message = (
                    f"Step {self.step_counter + 1}: 节点 {node.node_id} 原本拟按 {best_feature} 划分，"
                    f"但验证集准确率由 {pruning_detail['before_accuracy']:.3f} 到 "
                    f"{pruning_detail['after_accuracy']:.3f} 未提升，因此执行预剪枝。"
                )
                self._log(message)
                self.step_counter += 1
                node.is_leaf = True
                node.prediction = node.majority_label
                node.stop_reason = "预剪枝：分裂后验证集表现未提升"
                result = StepResult(
                    step_index=self.step_counter,
                    action="pre_prune_stop",
                    message=message,
                    node_id=node.node_id,
                    selected_feature=best_feature,
                    score_rows=self._score_rows(evaluation),
                    split_rows=self._split_rows(split_map),
                    node_snapshot=self.node_detail(node.node_id),
                    pruning_detail=pruning_detail,
                )
                self.last_step = result
                return result

        node.split_feature = best_feature
        node.split_score = score_value
        node.split_rule = split_rule
        node.split_summary = self._split_rows(split_map)

        remaining_features = [feature for feature in node.available_features if feature != best_feature]
        for branch, indices in split_map.items():
            child = self._create_node(indices, node.depth + 1, node.node_id, branch, remaining_features.copy())
            child.node_id = self._allocate_child_id()
            self.nodes[child.node_id] = child
            node.children.append(child.node_id)
            self.queue.append(child.node_id)

        self.step_counter += 1
        message = (
            f"Step {self.step_counter}: 处理节点 {node.node_id}，选择特征“{best_feature}”进行划分，原因：{reason_text}；"
            f"生成 {len(split_map)} 个子节点。"
        )
        self._log(message)
        result = StepResult(
            step_index=self.step_counter,
            action="split_node",
            message=message,
            node_id=node.node_id,
            selected_feature=best_feature,
            score_rows=self._score_rows(evaluation),
            split_rows=self._split_rows(split_map),
            node_snapshot=self.node_detail(node.node_id),
            pruning_detail=pruning_detail,
        )
        self.last_step = result
        return result

    def build_full_tree(self) -> List[StepResult]:
        results: List[StepResult] = []
        while self.queue:
            step = self.next_step()
            if step:
                results.append(step)
        return results
