# 05 — Security Risk Register（組員 C）

對應需求書第 11 節。實作於 `src/security.py`，呈現於 `app/streamlit_app.py`。
README 有摘要表，本檔放完整風險說明與控制細節。

> 本檔的 Input Validation 與 Inference Log 屬於完整推論流程的一環；
> 整體流程（輸入 → 驗證 → 推論 → 解釋 → 顯示）見根目錄 README「推論流程 · Inference Flow」。

## 風險登錄表

| # | 風險 | 說明 | 控制方式 | 對應實作 | 狀態 |
| --- | --- | --- | --- | --- | --- |
| 1 | 學生個資外洩 | 推論輸入含敏感屬性，log 若保留原值會外洩 | 去識別化、不紀錄完整個資 | `security.deidentify` / `log_inference` | ✅ |
| 2 | 未授權存取 | 任何人可呼叫介面取得預測 | 部署層加存取控制 | Streamlit 部署設定 | ⬜ TODO |
| 3 | 輸入異常資料 | 超範圍 / 型別錯誤 / 邏輯矛盾導致錯誤預測 | Input Validation（範圍 + 型別 + 跨欄位邏輯檢查） | `security.validate_input` | ✅ |
| 4 | Data Leakage | 用到未來資訊的特徵 | 前處理階段檢查欄位 | [01_data_preprocessing.md](01_data_preprocessing.md) | ⬜ 協作 |
| 5 | Model Extraction | 大量查詢重建模型 | 限流、紀錄查詢來源 | inference log | ⬜ TODO |
| 6 | Bias Amplification | 對特定群體系統性誤判 | Fairness 評估 + MinDiff | [03_fairness.md](03_fairness.md) | ⬜ 協作 |
| 7 | Training-serving Skew | 訓練/推論前處理不一致 | 訓練與推論共用同一 preprocessor | `preprocessing.load_preprocessor` | ✅ |
| 8 | 過度依賴模型結果 | 把預測當唯一決策依據 | 顯示使用限制 + Human Review 警告 | `security.HUMAN_REVIEW_WARNING` | ✅ |

## 控制措施細節
- **Input Validation**：`validate_input` 檢查欄位齊全、型別（類別需整數）、數值範圍
  （如入學成績 / 前一學歷成績 0–200、學期成績 0–20、通過/修課科目數 0–30、
  `Scholarship holder` 與 `Tuition fees up to date` 限 0/1），並外加**跨欄位邏輯規則**：
  通過科目數不可超過修課科目數（`approved ≤ enrolled`，第一、二學期各一條）。
  範圍由 `bounds_for()` 統一提供，前端表單與後端驗證共用同一份來源，避免分歧。
- **去識別化**：`deidentify` 在寫 log 前，把本專案宣告之敏感屬性
  **`Tuition fees up to date`（學費繳納狀態，代理社經地位）** 遮罩為 `***`
  （由 `schema.SENSITIVE_ATTRIBUTES` 統一定義；若日後新增敏感屬性，於該處增列即可同步生效）。
- **Inference Log**：`log_inference` 以 JSONL 寫入 `results/inference_log.jsonl`，含時間戳、模型版本、去識別化輸入、預測結果。
- **共用 preprocessor**：避免 training-serving skew，確保前端與訓練使用相同 scaler。

## 系統警告（顯示於介面）
```
This prediction is intended for early intervention support only.
It must not be used as the sole basis for academic,
disciplinary, or enrollment decisions.
```

## 待辦
- [ ] 部署層存取控制（風險 #2）
- [ ] 查詢限流與異常偵測（風險 #5）
- [ ] 與組員 A 確認 Data Leakage 欄位清單（風險 #4）
