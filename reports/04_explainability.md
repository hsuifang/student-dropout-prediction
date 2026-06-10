# 04 — Explainability（組員 C）

對應需求書第 7 節。實作於 `src/explain.py`，呈現於 `app/streamlit_app.py`。

## 1. 方法
- 主要：**SHAP**（`GradientExplainer`，適用 PyTorch MLP）
- 後援：環境未安裝 SHAP 時，自動退回 **Gradient × Input** 近似，確保介面仍可 demo
- 程式入口：`explain_record(record, target_label, top_k)`

## 2. Global Explanation（全域）
- 目的：哪些特徵整體最重要
- 做法：對測試集計算 SHAP 值後取平均絕對值排序
- 結果圖（summary plot）：_TODO 放 results/_
- 預期重要特徵：第二學期通過科目數、學費是否繳清、入學成績等

## 3. Local Explanation（局部）
- 目的：單一學生為何被預測為某類
- 做法：對該筆輸入計算各特徵 signed contribution
  - contribution > 0 → 推高該類別機率
  - contribution < 0 → 降低該類別機率
- 介面顯示 top-k 特徵與方向（🔺推高 / 🔻降低）

## 4. 介面呈現範例
```
Predicted Outcome: Dropout
Main Contributing Factors:
- Low number of approved courses
- Tuition fees not up to date
- Low admission grade
```

## 5. 限制
> SHAP 顯示的是模型判斷依據，**不代表真正的因果關係**；解釋僅供人工審核參考。
