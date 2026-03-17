from __future__ import annotations

import io
from typing import Dict, Tuple

import pandas as pd


def load_watermelon_dataset() -> Tuple[pd.DataFrame, str]:
    """西瓜数据集 2.0（离散版）。"""

    data = [
        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
        ["浅白", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
        ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "是"],
        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", "是"],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", "是"],
        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", "否"],
        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", "否"],
        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", "否"],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", "否"],
        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", "否"],
        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", "否"],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "否"],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", "否"],
        ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", "否"],
    ]
    columns = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "好瓜"]
    return pd.DataFrame(data, columns=columns), "好瓜"


def load_play_tennis_dataset() -> Tuple[pd.DataFrame, str]:
    data = [
        ["Sunny", "Hot", "High", "Weak", "No"],
        ["Sunny", "Hot", "High", "Strong", "No"],
        ["Overcast", "Hot", "High", "Weak", "Yes"],
        ["Rain", "Mild", "High", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Strong", "No"],
        ["Overcast", "Cool", "Normal", "Strong", "Yes"],
        ["Sunny", "Mild", "High", "Weak", "No"],
        ["Sunny", "Cool", "Normal", "Weak", "Yes"],
        ["Rain", "Mild", "Normal", "Weak", "Yes"],
        ["Sunny", "Mild", "Normal", "Strong", "Yes"],
        ["Overcast", "Mild", "High", "Strong", "Yes"],
        ["Overcast", "Hot", "Normal", "Weak", "Yes"],
        ["Rain", "Mild", "High", "Strong", "No"],
    ]
    columns = ["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"]
    return pd.DataFrame(data, columns=columns), "PlayTennis"


def builtin_datasets() -> Dict[str, Tuple[pd.DataFrame, str]]:
    return {
        "西瓜数据集 2.0（离散版）": load_watermelon_dataset(),
        "Play Tennis 示例": load_play_tennis_dataset(),
    }


def load_uploaded_csv(file) -> pd.DataFrame:
    raw = file.getvalue()
    text = io.BytesIO(raw)
    return pd.read_csv(text)
