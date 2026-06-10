"""Inference 介面 (對應需求書第 10 節)。

流程: 輸入 -> 檢查格式 -> 套用 Scaler -> 推論 -> 顯示機率 -> SHAP -> Human Review 警告
執行: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# 讓 app 能 import 專案根目錄的 src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import explain_record
from src.inference import predict
from src.schema import FEATURE_SCHEMA
from src.security import HUMAN_REVIEW_WARNING, log_inference, validate_input

st.set_page_config(page_title="學生退學預測系統", page_icon="🎓", layout="centered")
st.title("🎓 學生退學與學業成果預測")
st.caption("A Secure Deep Learning System for Predicting Students' Dropout and Academic Success")

MODEL_PATH = "models/model.pt"
if not Path(MODEL_PATH).exists():
    st.error("找不到 models/model.pt，請先執行: `python -m scripts.train_placeholder`")
    st.stop()


def render_input(feat: dict):
    label = feat["name"] + (f" — {feat['help']}" if feat.get("help") else "")
    if feat["kind"] == "cat":
        return st.number_input(label, value=int(feat["default"]), step=1, format="%d")
    return st.number_input(label, value=float(feat["default"]))


# ---- 輸入表單 ----
st.subheader("1. 輸入學生資料")
record: dict = {}
key_feats = [f for f in FEATURE_SCHEMA if f.get("key")]
other_feats = [f for f in FEATURE_SCHEMA if not f.get("key")]

for feat in key_feats:
    record[feat["name"]] = render_input(feat)

with st.expander(f"進階欄位 ({len(other_feats)} 項，預設值即可)"):
    for feat in other_feats:
        record[feat["name"]] = render_input(feat)

# ---- 預測 ----
if st.button("執行預測", type="primary"):
    errors = validate_input(record)  # Security: Input Validation
    if errors:
        st.error("輸入驗證失敗：")
        for e in errors:
            st.write(f"- {e}")
        st.stop()

    result = predict(record, model_path=MODEL_PATH)

    st.subheader("2. 預測結果")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Outcome", result["outcome"])
    risk = result["risk_level"]
    c2.metric("Risk Level", risk, delta="退學風險" if risk == "High" else None,
              delta_color="inverse")

    st.write("**各類別機率**")
    for label, prob in result["probabilities"].items():
        st.progress(min(max(prob, 0.0), 1.0), text=f"{label}: {prob:.1%}")

    # ---- Explainability ----
    st.subheader("3. Explainability")
    exp = explain_record(record, target_label=result["outcome"])
    st.caption(f"方法: {exp['method']}（顯示判斷依據，非因果關係）")
    st.write(f"影響 **{exp['target']}** 的主要特徵：")
    for name, contrib in exp["top_features"]:
        arrow = "🔺 推高" if contrib > 0 else "🔻 降低"
        st.write(f"- {arrow}　{name}　({contrib:+.3f})")

    # ---- Security / Human Review ----
    st.subheader("4. 使用限制與人工審核")
    st.warning(HUMAN_REVIEW_WARNING)
    st.caption(
        f"Model Version: {result['model_version']}　|　"
        f"Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Inference Log (去識別化後寫入)
    log_inference(record, {
        "outcome": result["outcome"],
        "risk_level": result["risk_level"],
        "probabilities": result["probabilities"],
    }, model_version=result["model_version"])
    st.caption("✅ 已寫入去識別化推論紀錄 (results/inference_log.jsonl)")
