"""Security 與 Input Validation (對應需求書第 11 節)。

組員 C 負責: Input Validation / Inference Log / 不紀錄完整個資 / 顯示使用限制。
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .schema import FEATURE_SCHEMA, FEATURE_ORDER, SENSITIVE_ATTRIBUTES

# 顯示於介面的使用限制警告 (需求書原文)
HUMAN_REVIEW_WARNING = (
    "This prediction is intended for early intervention support only.\n"
    "It must not be used as the sole basis for academic, "
    "disciplinary, or enrollment decisions."
)

# 每個欄位的合理範圍 (超出視為異常輸入)
_NUMERIC_BOUNDS = {
    "Admission grade": (0, 200),
    "Previous qualification (grade)": (0, 200),
    "Age at enrollment": (15, 80),
    "Curricular units 1st sem (grade)": (0, 20),
    "Curricular units 2nd sem (grade)": (0, 20),
}
_DEFAULT_NUM_BOUNDS = (0, 1000)
_DEFAULT_CAT_BOUNDS = (0, 99999)


def validate_input(record: dict) -> list[str]:
    """檢查輸入是否合法，回傳錯誤訊息清單 (空清單代表通過)。"""
    errors: list[str] = []
    for feat in FEATURE_SCHEMA:
        name = feat["name"]
        if name not in record:
            errors.append(f"缺少欄位: {name}")
            continue
        value = record[name]
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} 必須是數值，收到: {value!r}")
            continue
        if feat["kind"] == "cat" and num != int(num):
            errors.append(f"{name} 為類別欄位，必須是整數，收到: {value!r}")
        lo, hi = _NUMERIC_BOUNDS.get(
            name, _DEFAULT_CAT_BOUNDS if feat["kind"] == "cat" else _DEFAULT_NUM_BOUNDS
        )
        if not (lo <= num <= hi):
            errors.append(f"{name} 超出合理範圍 [{lo}, {hi}]，收到: {num}")
    return errors


def deidentify(record: dict) -> dict:
    """去識別化: 不在 log 中保留敏感屬性的原始值，改以遮罩標記。"""
    masked = dict(record)
    for attr in SENSITIVE_ATTRIBUTES:
        if attr in masked:
            masked[attr] = "***"
    return masked


def log_inference(record: dict, prediction: dict, *, model_version: str,
                  log_path: str = "results/inference_log.jsonl") -> None:
    """寫入推論紀錄 (僅記錄去識別化後的輸入 + 預測結果 + 版本 + 時間)。"""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "input": deidentify(record),
        "prediction": prediction,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")