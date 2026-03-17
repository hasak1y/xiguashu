from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def train_val_split(
    df: pd.DataFrame,
    val_ratio: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0 or len(df) <= 1:
        return df.copy(), pd.DataFrame(columns=df.columns)
    val_size = max(1, int(round(len(df) * val_ratio)))
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    train_df = shuffled.iloc[val_size:].reset_index(drop=True)
    if train_df.empty:
        train_df = val_df.iloc[:-1].reset_index(drop=True)
        val_df = val_df.iloc[-1:].reset_index(drop=True)
    return train_df, val_df


def infer_default_features(df: pd.DataFrame, target: str) -> List[str]:
    return [column for column in df.columns if column != target]
