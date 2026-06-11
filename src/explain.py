"""Explainability (對應需求書第 7 節)。

提供 Local Explanation: 對單筆預測，找出推高/降低「退學風險」的主要特徵。
優先使用 SHAP；若環境未安裝 SHAP，退回梯度近似法，讓介面仍可 demo。

二分類: 模型只有單一 logit (P(Dropout))，因此解釋對象固定為退學風險:
    contribution > 0 → 推高退學機率, < 0 → 降低退學機率。

注意: SHAP 顯示的是模型判斷依據，不代表真正的因果關係。
"""
from __future__ import annotations

import numpy as np
import torch

from .inference import _load
from .preprocessing import record_to_array
from .schema import FEATURE_ORDER


def _gradient_attribution(model, x: np.ndarray) -> np.ndarray:
    """以輸入梯度 x 輸入值 近似各特徵對退學 logit 的貢獻 (SHAP 不可用時的後援)。"""
    t = torch.from_numpy(x).clone().requires_grad_(True)
    logit = model(t)[0, 0]
    logit.backward()
    return t.grad.numpy()[0] * x[0]


def explain_record(record: dict, target_label: str | None = None, top_k: int = 5,
                   model_path: str = "models/model.pt",
                   preprocessor_path: str = "models/preprocessor.joblib") -> dict:
    """回傳 {'method', 'target', 'top_features': [(feature, signed_contribution), ...]}。

    target_label 參數保留以相容呼叫端，但二分類解釋對象固定為退學風險 (Dropout)。
    contribution > 0 表示推高退學機率，< 0 表示降低。
    """
    model, ckpt, scaler = _load(model_path, preprocessor_path)
    pos_label = ckpt.get("label_names", ["Graduate", "Dropout"])[-1]
    x = record_to_array(record, scaler)

    try:
        import shap  # noqa: WPS433

        background = np.zeros_like(x)
        explainer = shap.GradientExplainer(model, torch.from_numpy(background))
        shap_values = explainer.shap_values(torch.from_numpy(x))
        contrib = np.asarray(shap_values).reshape(-1)[: len(FEATURE_ORDER)]
        method = "SHAP (GradientExplainer)"
    except Exception:  # noqa: BLE001 - 後援機制，任何 SHAP 問題都退回梯度法
        contrib = _gradient_attribution(model, x)
        method = "Gradient x Input (SHAP fallback)"

    pairs = list(zip(FEATURE_ORDER, np.asarray(contrib).tolist()))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return {
        "method": method,
        "target": pos_label,
        "top_features": pairs[:top_k],
    }


def global_importance(X_scaled, top_k: int | None = None,
                      model_path: str = "models/model.pt",
                      preprocessor_path: str = "models/preprocessor.joblib") -> dict:
    """Global Explanation: 在一批資料上聚合各特徵的整體重要度。

    做法：對每一筆樣本計算退學 logit 對輸入的梯度 × 輸入值，取「絕對值的平均」
    作為該特徵的全域重要度（mean |grad × input|，數值越大代表整體影響越大）。

    參數:
        X_scaled : 已標準化的特徵矩陣 (N, 14)，欄位順序須等於 schema.FEATURE_ORDER。
                   例如直接讀入訓練程式產出的 `test_scaled.csv`（去掉 Target_Label 欄）。
        top_k    : 只回傳前 k 個；None 表示全部。

    回傳 {'method', 'target', 'n_samples', 'importances': [(feature, mean_abs_contribution), ...]}。
    """
    model, ckpt, _ = _load(model_path, preprocessor_path)
    pos_label = ckpt.get("label_names", ["Graduate", "Dropout"])[-1]

    X = np.asarray(X_scaled, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"X_scaled 形狀需為 (N, {len(FEATURE_ORDER)})，收到 {X.shape}")

    t = torch.from_numpy(X).clone().requires_grad_(True)
    logits = model(t)                 # (N, 1)；BatchNorm 於 eval 模式用 running stats，逐列獨立
    logits.sum().backward()           # 對每列而言即 d(該列 logit)/d(該列輸入)
    contrib = np.abs(t.grad.numpy() * X).mean(axis=0)   # 各特徵的 mean |grad × input|

    pairs = sorted(zip(FEATURE_ORDER, contrib.tolist()), key=lambda p: p[1], reverse=True)
    return {
        "method": "Gradient × Input (mean |·| over dataset)",
        "target": pos_label,
        "n_samples": int(X.shape[0]),
        "importances": pairs[:top_k] if top_k else pairs,
    }
