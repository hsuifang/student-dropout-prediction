"""前處理 pipeline: 確保訓練與推論使用「同一個」scaler，避免 Training-serving Skew。

placeholder 版本將所有欄位視為數值並套用 StandardScaler。
組員 A/B 若有更完整的 encoder，可替換 build_preprocessor 並維持 save/load 介面。
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schema import FEATURE_ORDER


def build_preprocessor(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X[FEATURE_ORDER].values)
    return scaler


def save_preprocessor(scaler, path: str = "models/preprocessor.joblib") -> None:
    joblib.dump(scaler, path)


def load_preprocessor(path: str = "models/preprocessor.joblib"):
    return joblib.load(path)


def record_to_array(record: dict, scaler) -> np.ndarray:
    """把單筆 dict 依照 FEATURE_ORDER 排序、套用 scaler，回傳 (1, n_features)。"""
    row = np.array([[float(record[name]) for name in FEATURE_ORDER]], dtype=np.float32)
    return scaler.transform(row).astype(np.float32)
