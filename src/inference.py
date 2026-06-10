"""推論層: 載入模型 + preprocessor，提供單筆預測。

對應需求書第 10 節的流程:
輸入 -> 檢查格式 -> 套用 Scaler -> 推論 -> 機率 -> 風險等級
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
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
        logits = model(torch.from_numpy(x))
        probs = torch.softmax(logits, dim=1).numpy()[0]

    labels = ckpt.get("label_names", LABELS)
    prob_map = {label: float(p) for label, p in zip(labels, probs)}
    top_idx = int(np.argmax(probs))
    dropout_prob = prob_map.get("Dropout", float(probs[0]))

    return {
        "outcome": labels[top_idx],
        "probabilities": prob_map,
        "dropout_probability": dropout_prob,
        "risk_level": risk_level(dropout_prob),
        "model_version": ckpt.get("model_version", "unknown"),
    }
