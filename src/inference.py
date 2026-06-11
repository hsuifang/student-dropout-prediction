"""推論層: 載入模型 + preprocessor，提供單筆預測。

對應需求書第 10 節的流程:
輸入 -> 檢查格式 -> 套用 Scaler -> 推論 -> 機率 -> 風險等級

二分類: 模型輸出單一 logit，sigmoid 後即 P(Dropout)。
"""
from __future__ import annotations

from functools import lru_cache

import torch

from .model import load_checkpoint
from .preprocessing import load_preprocessor, record_to_array
from .schema import LABELS


def risk_level(dropout_prob: float) -> str:
    if dropout_prob >= 0.66:
        return "High"
    if dropout_prob >= 0.33:
        return "Medium"
    return "Low"


@lru_cache(maxsize=1)
def _load(model_path: str = "models/model.pt",
          preprocessor_path: str = "models/preprocessor.joblib"):
    model, ckpt = load_checkpoint(model_path)
    scaler = load_preprocessor(preprocessor_path)
    return model, ckpt, scaler


def predict(record: dict, model_path: str = "models/model.pt",
            preprocessor_path: str = "models/preprocessor.joblib") -> dict:
    """回傳預測結果 dict: outcome / probabilities / risk_level / model_version。"""
    model, ckpt, scaler = _load(model_path, preprocessor_path)
    x = record_to_array(record, scaler)
    with torch.no_grad():
        logit = model(torch.from_numpy(x))
        dropout_prob = float(torch.sigmoid(logit).item())

    # label_names = [負類(Graduate), 正類(Dropout)]
    neg_label, pos_label = ckpt.get("label_names", LABELS)
    prob_map = {pos_label: dropout_prob, neg_label: 1.0 - dropout_prob}
    outcome = pos_label if dropout_prob >= 0.5 else neg_label

    return {
        "outcome": outcome,
        "probabilities": prob_map,
        "dropout_probability": dropout_prob,
        "risk_level": risk_level(dropout_prob),
        "model_version": ckpt.get("model_version", "unknown"),
    }
