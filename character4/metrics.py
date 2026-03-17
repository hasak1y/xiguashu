from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def label_counts(series: pd.Series) -> Dict[Any, int]:
    return dict(Counter(series.tolist()))


def majority_label(series: pd.Series) -> Any:
    counts = label_counts(series)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def entropy(series: pd.Series) -> float:
    counts = label_counts(series)
    total = len(series)
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            value -= prob * math.log2(prob)
    return value


def gini(series: pd.Series) -> float:
    counts = label_counts(series)
    total = len(series)
    if total == 0:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def conditional_entropy(
    df: pd.DataFrame,
    feature: str,
    target: str,
    sample_indices: List[int],
) -> Tuple[float, List[Dict[str, Any]]]:
    subset = df.loc[sample_indices]
    total = len(subset)
    detail: List[Dict[str, Any]] = []
    value = 0.0
    for feature_value, group in subset.groupby(feature):
        weight = len(group) / total
        group_entropy = entropy(group[target])
        value += weight * group_entropy
        detail.append(
            {
                "branch": str(feature_value),
                "samples": len(group),
                "weight": weight,
                "label_counts": label_counts(group[target]),
                "metric": group_entropy,
            }
        )
    return value, detail


def split_info(
    df: pd.DataFrame,
    feature: str,
    sample_indices: List[int],
) -> float:
    subset = df.loc[sample_indices]
    total = len(subset)
    value = 0.0
    for _, group in subset.groupby(feature):
        prob = len(group) / total
        if prob > 0:
            value -= prob * math.log2(prob)
    return value


def information_gain(
    df: pd.DataFrame,
    feature: str,
    target: str,
    sample_indices: List[int],
) -> Dict[str, Any]:
    subset = df.loc[sample_indices]
    base_entropy = entropy(subset[target])
    cond_entropy, detail = conditional_entropy(df, feature, target, sample_indices)
    gain = base_entropy - cond_entropy
    return {
        "feature": feature,
        "base_entropy": base_entropy,
        "conditional_entropy": cond_entropy,
        "information_gain": gain,
        "detail": detail,
    }


def gain_ratio(
    df: pd.DataFrame,
    feature: str,
    target: str,
    sample_indices: List[int],
) -> Dict[str, Any]:
    info = information_gain(df, feature, target, sample_indices)
    iv = split_info(df, feature, sample_indices)
    ratio = info["information_gain"] / iv if iv > 0 else 0.0
    return {
        **info,
        "split_info": iv,
        "gain_ratio": ratio,
    }


def weighted_gini_for_groups(groups: Iterable[pd.DataFrame], target: str) -> float:
    groups = list(groups)
    total = sum(len(group) for group in groups)
    if total == 0:
        return 0.0
    return sum((len(group) / total) * gini(group[target]) for group in groups)
