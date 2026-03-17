"""Manual data splitters that correspond to chapter 2.2 evaluation methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitResult:
    """One split produced by an evaluation method."""

    round_id: int
    train_indices: np.ndarray
    valid_indices: np.ndarray
    method_name: str
    note: str


def _normalize_indices(indices: np.ndarray | list[int]) -> np.ndarray:
    """Convert indices to an integer numpy array."""

    return np.asarray(indices, dtype=int)


def train_test_split_indices(
    sample_count: int,
    train_ratio: float,
    random_seed: int,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split once into train and test parts for the optional fixed test set."""

    indices = np.arange(sample_count)
    rng = np.random.default_rng(random_seed)
    if shuffle:
        indices = rng.permutation(indices)

    train_size = max(1, min(sample_count - 1, int(round(sample_count * train_ratio))))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return _normalize_indices(train_indices), _normalize_indices(test_indices)


def hold_out_split(
    sample_count: int,
    train_ratio: float,
    random_seed: int,
    shuffle: bool = True,
) -> list[SplitResult]:
    """Implement the hold-out method manually.

    对应《西瓜书》2.2 的“留出法”：一次性把数据划成训练集和验证集。
    """

    train_indices, valid_indices = train_test_split_indices(sample_count, train_ratio, random_seed, shuffle=shuffle)
    return [
        SplitResult(
            round_id=1,
            train_indices=train_indices,
            valid_indices=valid_indices,
            method_name="留出法",
            note="一次随机切分，简单直接，但结果容易受这一次切分影响。",
        )
    ]


def repeated_hold_out_split(
    sample_count: int,
    train_ratio: float,
    repeats: int,
    random_seed: int,
    shuffle: bool = True,
) -> list[SplitResult]:
    """Repeat the hold-out split several times for stability comparison."""

    splits: list[SplitResult] = []
    for repeat_id in range(repeats):
        train_indices, valid_indices = train_test_split_indices(
            sample_count,
            train_ratio,
            random_seed + repeat_id,
            shuffle=shuffle,
        )
        splits.append(
            SplitResult(
                round_id=repeat_id + 1,
                train_indices=train_indices,
                valid_indices=valid_indices,
                method_name="重复留出法",
                note="重复多次留出法，可以观察单次切分带来的波动。",
            )
        )
    return splits


def k_fold_split(
    sample_count: int,
    k: int,
    random_seed: int,
    shuffle: bool = True,
) -> list[SplitResult]:
    """Implement k-fold cross validation manually.

    对应《西瓜书》2.2 的“交叉验证法”：把数据分成 k 份，轮流拿一份验证。
    """

    if k < 2:
        raise ValueError("k-fold 需要 k >= 2。")
    if k > sample_count:
        raise ValueError("k 不能大于样本数。")

    indices = np.arange(sample_count)
    rng = np.random.default_rng(random_seed)
    if shuffle:
        indices = rng.permutation(indices)

    fold_sizes = np.full(k, sample_count // k, dtype=int)
    fold_sizes[: sample_count % k] += 1

    splits: list[SplitResult] = []
    current = 0
    for fold_id, fold_size in enumerate(fold_sizes, start=1):
        start = current
        stop = current + fold_size
        valid_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        current = stop

        splits.append(
            SplitResult(
                round_id=fold_id,
                train_indices=_normalize_indices(train_indices),
                valid_indices=_normalize_indices(valid_indices),
                method_name=f"{k} 折交叉验证",
                note=f"第 {fold_id} 折把第 {fold_id} 份作为验证集，其余 {k - 1} 份用于训练。",
            )
        )
    return splits


def loocv_split(sample_count: int) -> list[SplitResult]:
    """Implement leave-one-out cross validation manually.

    对应《西瓜书》2.2 的“留一法”：每次只留 1 个样本做验证。
    """

    splits: list[SplitResult] = []
    for index in range(sample_count):
        train_indices = np.delete(np.arange(sample_count), index)
        valid_indices = np.array([index], dtype=int)
        splits.append(
            SplitResult(
                round_id=index + 1,
                train_indices=train_indices,
                valid_indices=valid_indices,
                method_name="留一法",
                note="每轮只拿 1 个样本验证，因此样本利用率高，但训练次数很多。",
            )
        )
    return splits


def bootstrap_split(
    sample_count: int,
    n_bootstrap: int,
    random_seed: int,
) -> list[SplitResult]:
    """Implement bootstrap sampling manually.

    对应《西瓜书》2.2 的“自助法”：有放回采样形成训练集，未采中的样本作为袋外验证参考。
    """

    rng = np.random.default_rng(random_seed)
    indices = np.arange(sample_count)
    splits: list[SplitResult] = []

    for round_id in range(1, n_bootstrap + 1):
        train_indices = rng.choice(indices, size=sample_count, replace=True)
        sampled_mask = np.zeros(sample_count, dtype=bool)
        sampled_mask[np.unique(train_indices)] = True
        valid_indices = indices[~sampled_mask]

        splits.append(
            SplitResult(
                round_id=round_id,
                train_indices=_normalize_indices(train_indices),
                valid_indices=_normalize_indices(valid_indices),
                method_name="自助法",
                note="训练集允许重复样本；没有被抽到的袋外样本可以作为验证参考。",
            )
        )
    return splits
