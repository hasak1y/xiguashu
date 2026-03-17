from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Sequence, Set

import pandas as pd
from pandas.api.types import is_numeric_dtype

from metrics import gain_ratio, information_gain, weighted_gini_for_groups


def split_discrete(df: pd.DataFrame, sample_indices: List[int], feature: str) -> Dict[str, List[int]]:
    subset = df.loc[sample_indices]
    result: Dict[str, List[int]] = {}
    for value, group in subset.groupby(feature):
        result[str(value)] = group.index.tolist()
    return result


def _candidate_subsets(values: Sequence[Any]) -> List[Set[Any]]:
    values = list(values)
    n = len(values)
    candidates: List[Set[Any]] = []
    if n <= 1:
        return candidates
    max_size = n // 2
    for size in range(1, max_size + 1):
        for combo in combinations(values, size):
            subset = set(combo)
            complement = set(values) - subset
            if not complement:
                continue
            if n % 2 == 0 and size == n // 2:
                if tuple(sorted(map(str, subset))) > tuple(sorted(map(str, complement))):
                    continue
            candidates.append(subset)
    return candidates


def evaluate_id3(df: pd.DataFrame, sample_indices: List[int], target: str, features: List[str]) -> Dict[str, Any]:
    scores = [information_gain(df, feature, target, sample_indices) for feature in features]
    best = max(scores, key=lambda item: item["information_gain"])
    return {"scores": scores, "best": best}


def evaluate_c45(df: pd.DataFrame, sample_indices: List[int], target: str, features: List[str]) -> Dict[str, Any]:
    scores = [gain_ratio(df, feature, target, sample_indices) for feature in features]
    avg_gain = sum(item["information_gain"] for item in scores) / len(scores) if scores else 0.0
    filtered = [item for item in scores if item["information_gain"] >= avg_gain]
    best_pool = filtered or scores
    best = max(best_pool, key=lambda item: item["gain_ratio"])
    return {"scores": scores, "best": best, "avg_gain": avg_gain}


def evaluate_cart(df: pd.DataFrame, sample_indices: List[int], target: str, features: List[str]) -> Dict[str, Any]:
    subset = df.loc[sample_indices]
    scores: List[Dict[str, Any]] = []

    for feature in features:
        feature_scores: List[Dict[str, Any]] = []
        if is_numeric_dtype(subset[feature]):
            sorted_values = sorted(subset[feature].dropna().unique().tolist())
            thresholds = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
            for threshold in thresholds:
                left = subset[subset[feature] <= threshold]
                right = subset[subset[feature] > threshold]
                weighted = weighted_gini_for_groups([left, right], target)
                feature_scores.append(
                    {
                        "feature": feature,
                        "split_type": "continuous",
                        "rule": f"<= {threshold:.3f} / > {threshold:.3f}",
                        "left_values": None,
                        "threshold": threshold,
                        "weighted_gini": weighted,
                        "branches": [
                            {
                                "branch": f"{feature} <= {threshold:.3f}",
                                "samples": len(left),
                                "label_counts": left[target].value_counts().to_dict(),
                            },
                            {
                                "branch": f"{feature} > {threshold:.3f}",
                                "samples": len(right),
                                "label_counts": right[target].value_counts().to_dict(),
                            },
                        ],
                    }
                )
        else:
            values = sorted(subset[feature].dropna().unique().tolist(), key=str)
            for left_values in _candidate_subsets(values):
                left = subset[subset[feature].isin(left_values)]
                right = subset[~subset[feature].isin(left_values)]
                weighted = weighted_gini_for_groups([left, right], target)
                left_label = "{" + ", ".join(map(str, sorted(left_values, key=str))) + "}"
                right_values = set(values) - left_values
                right_label = "{" + ", ".join(map(str, sorted(right_values, key=str))) + "}"
                feature_scores.append(
                    {
                        "feature": feature,
                        "split_type": "discrete",
                        "rule": f"in {left_label} / in {right_label}",
                        "left_values": sorted(left_values, key=str),
                        "threshold": None,
                        "weighted_gini": weighted,
                        "branches": [
                            {
                                "branch": f"in {left_label}",
                                "samples": len(left),
                                "label_counts": left[target].value_counts().to_dict(),
                            },
                            {
                                "branch": f"in {right_label}",
                                "samples": len(right),
                                "label_counts": right[target].value_counts().to_dict(),
                            },
                        ],
                    }
                )
        if feature_scores:
            best_feature_score = min(feature_scores, key=lambda item: item["weighted_gini"])
            scores.append(
                {
                    "feature": feature,
                    "candidates": feature_scores,
                    "best_candidate": best_feature_score,
                    "weighted_gini": best_feature_score["weighted_gini"],
                }
            )

    best = min(scores, key=lambda item: item["weighted_gini"])
    return {"scores": scores, "best": best}


def split_cart_subset(
    df: pd.DataFrame,
    sample_indices: List[int],
    feature: str,
    best_candidate: Dict[str, Any],
) -> Dict[str, List[int]]:
    subset = df.loc[sample_indices]
    if best_candidate["split_type"] == "continuous":
        threshold = best_candidate["threshold"]
        left = subset[subset[feature] <= threshold].index.tolist()
        right = subset[subset[feature] > threshold].index.tolist()
        return {
            f"<= {threshold:.3f}": left,
            f"> {threshold:.3f}": right,
        }

    left_values = set(best_candidate["left_values"])
    left = subset[subset[feature].isin(left_values)].index.tolist()
    right = subset[~subset[feature].isin(left_values)].index.tolist()
    left_label = "{" + ", ".join(map(str, sorted(left_values, key=str))) + "}"
    right_label = "{" + ", ".join(
        map(str, sorted(set(subset[feature].unique().tolist()) - left_values, key=str))
    ) + "}"
    return {
        f"in {left_label}": left,
        f"in {right_label}": right,
    }
