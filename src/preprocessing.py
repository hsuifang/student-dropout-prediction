"""前處理 pipeline: 確保訓練與推論使用「同一個」scaler，避免 Training-serving Skew。

流程與訓練程式 (notebook) 一致:
    raw 11 欄 → 推導 3 個自創特徵 → 合成 14 維 → StandardScaler。
scaler 在訓練集上 fit、推論時只 transform，避免資訊洩漏。
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schema import ENGINEERED_FEATURES, FEATURE_ORDER, RAW_FEATURES

_EPS = 1e-8


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """由 11 個原始欄位推導 3 個自創特徵，回傳含 14 欄 (FEATURE_ORDER) 的 DataFrame。

    與訓練程式公式完全一致:
        1st_sem_pass_rate = approved_1 / (enrolled_1 + eps)
        2nd_sem_pass_rate = approved_2 / (enrolled_2 + eps)
        grade_change      = grade_2 - grade_1
    """
    out = X[RAW_FEATURES].copy()
    out["1st_sem_pass_rate"] = (
        X["Curricular units 1st sem (approved)"]
        / (X["Curricular units 1st sem (enrolled)"] + _EPS)
    )
    out["2nd_sem_pass_rate"] = (
        X["Curricular units 2nd sem (approved)"]
        / (X["Curricular units 2nd sem (enrolled)"] + _EPS)
    )
    out["grade_change"] = (
        X["Curricular units 2nd sem (grade)"] - X["Curricular units 1st sem (grade)"]
    )
    return out[FEATURE_ORDER]


def build_preprocessor(X: pd.DataFrame) -> StandardScaler:
    """X 為含 11 個原始欄位的訓練集；在推導完整 14 維後 fit scaler。"""
    X_eng = add_engineered_features(X)
    scaler = StandardScaler()
    scaler.fit(X_eng[FEATURE_ORDER].values)
    return scaler


def save_preprocessor(scaler, path: str = "models/preprocessor.joblib") -> None:
    joblib.dump(scaler, path)


def load_preprocessor(path: str = "models/preprocessor.joblib"):
    return joblib.load(path)


def record_to_array(record: dict, scaler) -> np.ndarray:
    """把單筆 raw dict (11 欄) 推導成 14 維、套用 scaler，回傳 (1, 14)。"""
    raw = {name: float(record[name]) for name in RAW_FEATURES}
    X_eng = add_engineered_features(pd.DataFrame([raw]))
    row = X_eng[FEATURE_ORDER].values.astype(np.float32)
    return scaler.transform(row).astype(np.float32)


__all__ = [
    "add_engineered_features",
    "build_preprocessor",
    "save_preprocessor",
    "load_preprocessor",
    "record_to_array",
    "ENGINEERED_FEATURES",
]
