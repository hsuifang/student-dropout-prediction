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
    "Curricular units 1st sem (grade)": (0, 20),
    "Curricular units 2nd sem (grade)": (0, 20),
    "Curricular units 1st sem (approved)": (0, 30),
    "Curricular units 2nd sem (approved)": (0, 30),
    "Curricular units 1st sem (enrolled)": (0, 30),
    "Curricular units 2nd sem (enrolled)": (0, 30),
    "Scholarship holder": (0, 1),
    "Tuition fees up to date": (0, 1),
}
_DEFAULT_NUM_BOUNDS = (0, 1000)
_DEFAULT_CAT_BOUNDS = (0, 99999)

# 跨欄位邏輯規則: (子集欄位, 母集欄位) — 通過科目數不可超過修課科目數。
_SUBSET_RULES = [
    ("Curricular units 1st sem (approved)", "Curricular units 1st sem (enrolled)"),
    ("Curricular units 2nd sem (approved)", "Curricular units 2nd sem (enrolled)"),
]


def bounds_for(feat: dict) -> tuple[float, float]:
    """回傳欄位的 (min, max) 合理範圍；前端與驗證共用同一份來源，避免分歧。"""
    default = _DEFAULT_CAT_BOUNDS if feat["kind"] == "cat" else _DEFAULT_NUM_BOUNDS
    return _NUMERIC_BOUNDS.get(feat["name"], default)


def validate_input(record: dict) -> list[str]:
    """檢查輸入是否合法，回傳錯誤訊息清單 (空清單代表通過)。"""
    errors: list[str] = []
    for feat in FEATURE_SCHEMA:
        name = feat["name"]
        if name not in record:
            errors.append(f"Missing field: {name} · 缺少欄位")
            continue
        value = record[name]
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number (got {value!r}) · 必須是數值")
            continue
        if feat["kind"] == "cat" and num != int(num):
            errors.append(f"{name} must be a whole number (got {value!r}) · 類別欄位需為整數")
        lo, hi = bounds_for(feat)
        if not (lo <= num <= hi):
            errors.append(f"{name} is out of range [{lo}, {hi}] (got {num}) · 超出合理範圍")

    # 跨欄位邏輯檢查: 通過科目數不可大於修課科目數。
    for approved, enrolled in _SUBSET_RULES:
        try:
            if float(record[approved]) > float(record[enrolled]):
                errors.append(
                    f"{approved} cannot exceed {enrolled} (approved can't be more than enrolled) "
                    f"· 通過科目數不可大於修課科目數"
                )
        except (KeyError, TypeError, ValueError):
            continue  # 缺值或非數值已在上方迴圈回報
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