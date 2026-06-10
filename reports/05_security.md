# 05 — Security Risk Register（組員 C）

對應需求書第 11 節。實作於 `src/security.py`，呈現於 `app/streamlit_app.py`。
README 有摘要表，本檔放完整風險說明與控制細節。

## 風險登錄表

| # | 風險 | 說明 | 控制方式 | 對應實作 | 狀態 |
| --- | --- | --- | --- | --- | --- |
| 1 | 學生個資外洩 | 推論輸入含敏感屬性，log 若保留原值會外洩 | 去識別化、不紀錄完整個資 | `security.deidentify` / `log_inference` | ✅ |
| 2 | 未授權存取 | 任何人可呼叫介面取得預測 | 部署層加存取控制 | Streamlit 部署設定 | ⬜ TODO |
| 3 | 輸入異常資料 | 超範圍 / 型別錯誤導致錯誤預測 | Input Validation（範圍 + 型別檢查） | `security.validate_input` | ✅ |
| 4 | Data Leakage | 用到未來資訊的特徵 | 前處理階段檢查欄位 | [01_data_preprocessing.md](01_data_preprocessing.md) | ⬜ 協作 |
| 5 | Model Extraction | 大量查詢重建模型 | 限流、紀錄查詢來源 | inference log | ⬜ TODO |
| 6 | Bias Amplification | 對特定群體系統性誤判 | Fairness 評估 + MinDiff | [03_fairness.md](03_fairness.md) | ⬜ 協作 |
| 7 | Training-serving Skew | 訓練/推論前處理不一致 | 訓練與推論共用同一 preprocessor | `preprocessing.load_preprocessor` | ✅ |
| 8 | 過度依賴模型結果 | 把預測當唯一決策依據 | 顯示使用限制 + Human Review 警告 | `security.HUMAN_REVIEW_WARNING` | ✅ |

## 控制措施細節
- **Input Validation**：`validate_input` 檢查欄位齊全、型別（類別需整數）、數值範圍（如入學成績 0–200、年齡 15–80）。
- **去識別化**：`deidentify` 在寫 log 前把敏感屬性（Gender / Age / Nacionality / Scholarship）遮罩為 `***`。
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
