"""Explainability (對應需求書第 7 節)。

提供 Local Explanation: 對單筆預測，找出推高/降低該結果的主要特徵。
優先使用 SHAP；若環境未安裝 SHAP，退回梯度近似法，讓介面仍可 demo。

注意: SHAP 顯示的是模型判斷依據，不代表真正的因果關係。
"""
from __future__ import annotations

import numpy as np
import torch

from .inference import _load
from .preprocessing import record_to_array
from .schema import FEATURE_ORDER


def _gradient_attribution(model, x: np.ndarray, target_idx: int) -> np.ndarray:
    """以輸入梯度 x 輸入值 近似各特徵貢獻 (SHAP 不可用時的後援)。"""
    t = torch.from_numpy(x).clone().requires_grad_(True)
    logit = model(t)[0, target_idx]
    logit.backward()
    return (t.grad.numpy()[0] * x[0])


def explain_record(record: dict, target_label: str | None = None, top_k: int = 5,
                   model_path: str = "models/model.pt",
                   preprocessor_path: str = "models/preprocessor.joblib") -> dict:
    """回傳 {'method', 'target', 'top_features': [(feature, signed_contribution), ...]}。

    contribution > 0 表示推高該類別的機率，< 0 表示降低。
    """
    model, ckpt, scaler = _load(model_path, preprocessor_path)
    labels = ckpt.get("label_names")
    x = record_to_array(record, scaler)

    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()[0]
    target_idx = labels.index(target_label) if target_label else int(np.argmax(probs))

    try:
        import shap  # noqa: WPS433

        background = np.zeros_like(x)
        explainer = shap.GradientExplainer(model, torch.from_numpy(background))
        shap_values = explainer.shap_values(torch.from_numpy(x))
        # shap_values: list[per-class] 或 (n, features, n_classes)
        if isinstance(shap_values, list):
            contrib = shap_values[target_idx][0]
        else:
            contrib = shap_values[0, :, target_idx]
        method = "SHAP (GradientExplainer)"
    except Exception:  # noqa: BLE001 - 後援機制，任何 SHAP 問題都退回梯度法
        contrib = _gradient_attribution(model, x, target_idx)
        method = "Gradient x Input (SHAP fallback)"

    pairs = list(zip(FEATURE_ORDER, contrib.tolist()))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return {
        "method": method,
        "target": labels[target_idx],
        "top_features": pairs[:top_k],
    }
