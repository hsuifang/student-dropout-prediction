# 學生退學與學業成果預測系統

**A Secure Deep Learning System for Predicting Students' Dropout and Academic Success**

使用學生的入學、課業與背景資料，預測學生最後可能屬於 **Dropout / Enrolled / Graduate**，
並涵蓋 Explainability、Security、Fairness 與可操作的 Inference 介面。

> 資料集：[UCI – Predict students' dropout and academic success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

## 快速開始

```bash
# 1. 安裝套件 (建議使用虛擬環境)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 訓練 placeholder 模型 -> 產出 models/model.pt
python -m scripts.train_placeholder

# 3. 啟動 Inference 介面
streamlit run app/streamlit_app.py
```

> ⚠️ `models/model.pt` 目前為 **placeholder**（佔位模型），僅供串接介面與 demo。
> 正式效能以組員 B 訓練的模型為準，替換 `models/model.pt` 即可，介面不需修改。

---

## 專案結構

```
student-dropout-prediction/
├── README.md                # 含 Data Card / Model Card / Security（見下方）
├── requirements.txt
├── src/                     # 共用模組 (模型契約)
│   ├── schema.py            # ★ 唯一真實來源：特徵順序、標籤、敏感屬性、版本
│   ├── model.py             # MLP 定義 + checkpoint 存取格式
│   ├── preprocessing.py     # scaler 建立/載入 (防 training-serving skew)
│   ├── inference.py         # 載入模型 + 預測 + 風險等級
│   ├── explain.py           # SHAP / 梯度後援 (Explainability)
│   └── security.py          # Input Validation / 去識別化 / Inference Log
├── scripts/
│   └── train_placeholder.py # 訓練佔位模型 -> models/model.pt
├── app/
│   └── streamlit_app.py     # Inference 介面 (組員 C 主要交付)
├── models/                  # model.pt / preprocessor.joblib (生成物)
├── results/                 # 推論紀錄、評估結果
├── notebooks/               # 探索性分析
├── reports/                 # 各階段詳細報告 (前處理/評估/fairness/explainability/security)
└── docs/                    # 作業需求書等參考文件
```

---

## 模型契約（給組員 B 替換真實模型）

只要遵守以下兩點，`models/model.pt` 可直接替換、前端無需改動：

1. **特徵順序與標籤** 來自 `src/schema.py`（`FEATURE_ORDER`、`LABELS`）。
2. **checkpoint 格式** 使用 `src/model.py` 的 `save_checkpoint(...)`，包含
   `state_dict / input_dim / hidden_dims / num_classes / dropout / model_version / label_names`，
   並一併輸出對應的 `models/preprocessor.joblib`。

---

## 分工對應

| 模組 | 負責人 | 
| --- | --- |
| 資料前處理、Data Card、Fairness 資料 | 113AB8049 | 
| Baseline、MLP、MinDiff、評估、Model Card | 113AB8046 | 
| Explainability、Security、Inference 介面、Deployment | 113AB8050 |

---

## 詳細報告

以下為重點摘要（Data Card / Model Card / Security）；完整方法與結果見 [`reports/`](reports/README.md)：

| 報告 | 內容 |
| --- | --- |
| [01 前處理](reports/01_data_preprocessing.md) | 資料分析、編碼、切分、Data Leakage |
| [02 模型與評估](reports/02_model_evaluation.md) | Baseline / MLP / MinDiff、指標比較 |
| [03 Fairness](reports/03_fairness.md) | 群體指標、MinDiff 前後比較 |
| [04 Explainability](reports/04_explainability.md) | SHAP 全域 / 局部解釋 |
| [05 Security](reports/05_security.md) | 完整風險登錄與控制細節 |

---

# 📋 Data Card（組員 A 113AB8049 填寫）

> 以下為待填樣板，可自行調整。

- **Dataset name**: Predict Students' Dropout and Academic Success
- **Dataset source**: UCI ML Repository (ID 697)
- **Dataset size**: _TODO（筆數、欄位數）_
- **Feature description**: 36 個特徵，見 `src/schema.py`（入學、課業、背景、總體經濟）
- **Target labels**: Dropout / Enrolled / Graduate
- **Data preprocessing**: _TODO（編碼、標準化、切分方式）_
- **Sensitive attributes**: Gender, Age at enrollment, Nacionality, Scholarship holder
- **Privacy risks**: _TODO_
- **Bias risks**: _TODO_
- **Data leakage risks**: _TODO（第二學期成績是否洩漏結果？）_
- **Intended use**: 早期介入輔導
- **Prohibited use**: 不可作為學業/紀律/註冊決策唯一依據
- **Dataset limitations**: _TODO_

---

# 📋 Model Card（組員 B 113AB8046 填寫）

> 以下為待填樣板，可自行調整。

- **Model name**: _TODO_
- **Model version**: 目前介面顯示 `0.1.0-placeholder`，正式版請更新 `src/schema.py` 的 `MODEL_VERSION`
- **Model architecture**: MLP（Input → Hidden(ReLU+Dropout) → Softmax 3 類），見 `src/model.py`
- **Training data**: _TODO_
- **Evaluation results**: _TODO_

  | Model | Accuracy | Macro F1 | Dropout Recall |
  | --- | --- | --- | --- |
  | Baseline | TBD | TBD | TBD |
  | MLP | TBD | TBD | TBD |
  | MLP + MinDiff | TBD | TBD | TBD |

- **Fairness results**: _TODO（FPR Gap / FNR Gap）_
- **Intended use**: 早期介入輔導
- **Out-of-scope use**: _TODO_
- **Model limitations**: _TODO_
- **Ethical risks**: _TODO_
- **Security risks**: 見下方 Security 章節
- **Human oversight**: 預測僅供參考，需保留人工審核
- **Deployment status**: _TODO_

---

# 🔒 Security（組員 C 113AB8050）

重點控制：**Input Validation**、敏感屬性**去識別化**、訓練/推論**共用 preprocessor**（防 training-serving skew）、**Human Review 警告**、去識別化 **inference log**。

> 完整風險登錄（8 項風險 × 說明 × 控制 × 對應實作 × 狀態）與待辦見 [`reports/05_security.md`](reports/05_security.md)。

## 系統警告（顯示於介面）

```
This prediction is intended for early intervention support only.
It must not be used as the sole basis for academic,
disciplinary, or enrollment decisions.
```