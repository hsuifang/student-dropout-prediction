# AI Final Project

[https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## 一、簡單版

### 題目

**A Secure Deep Learning System for Predicting Students' Dropout and Academic Success**

中文：

**具安全性與公平性的學生退學與學業成果預測系統**

### 我們要做什麼

使用學生的入學、課業與背景資料，建立模型預測學生最後可能屬於：

- Dropout
- Enrolled
- Graduate

除了建立深度學習模型，也要處理：

- Data Card
- Model Card
- Fairness
- Explainability
- Security
- Inference 介面
- Deployment Video
- GitHub
- 8–10 分鐘簡報

### 模型規劃

預計建立三個模型：

1. **Baseline**
    - Logistic Regression 或 Random Forest
2. **Deep Learning Model**
    - MLP
3. **Fairness Model**
    - MLP + MinDiff

主要比較：

- 哪個模型預測表現較好
- MLP 是否比 baseline 好
- 加入 MinDiff 後，群體偏誤是否降低
- Fairness 改善後，模型效能是否下降

---

## 二、任務分配簡表

| 組員 | 主要負責內容 |
| --- | --- |
| 組員 A | 資料分析、資料前處理、Data Card、Fairness 資料整理 |
| 組員 B | Baseline、MLP、MinDiff、模型評估、Model Card |
| 組員 C | Explainability、Security、Inference 介面、Deployment Video |
| 共同 | Paper 閱讀、GitHub 整理、簡報製作、報告練習 |

---

# 三、詳細版

## 1. 資料集分析與前處理

### 簡單說明

先了解資料集，整理成模型可以使用的格式。

### 需要完成的內容

- 確認資料筆數與欄位
- 確認預測標籤：
    - Dropout
    - Enrolled
    - Graduate
- 檢查缺失值、重複值與異常值
- 處理類別欄位編碼
- 處理數值欄位標準化
- 分割 Training、Validation、Test Set
- 檢查是否存在 Data Leakage
- 找出敏感屬性，例如：
    - Gender
    - Age
    - Nationality
    - Scholarship status

### 產出

- 前處理程式
- 資料欄位說明
- 類別分布圖
- Train、Validation、Test 資料

---

## 2. 建立 Baseline Model

### 簡單說明

建立一個較簡單的機器學習模型，作為 MLP 的比較基準。

### 建議模型

- Logistic Regression
- Random Forest

擇一即可。

### 需要完成的內容

- 使用相同資料切分訓練 baseline
- 計算模型評估指標
- 和 MLP 使用相同測試資料比較

### 目的

確認深度學習模型是否真的比傳統模型表現更好。

---

## 3. 建立 MLP 深度學習模型

### 簡單說明

使用 MLP 處理表格型學生資料，預測三種學業結果。

### 需要完成的內容

- 建立 Input Layer
- 建立 Hidden Layers
- 使用 ReLU
- 加入 Dropout
- 使用 Softmax 輸出三個類別
- 設定 Cross-Entropy Loss
- 使用 Adam Optimizer
- 加入 Early Stopping
- 儲存最佳模型

### 模型輸出

```
Dropout Probability
```

---

## 4. 建立 MLP + MinDiff

### 簡單說明

在 MLP 訓練時加入公平性限制，降低不同群體之間的預測差異。

### 預計做法

選擇一個敏感屬性，例如 Gender。

比較不同 Gender 群體的：

- Dropout Recall
- False Positive Rate
- False Negative Rate

MinDiff 會在原本的分類 Loss 之外加入 Fairness Loss：

```
Total Loss
=
Classification Loss
+
λ × Fairness Loss
```

### 需要比較

- 原始 MLP
- MLP + MinDiff

觀察：

- 公平性差距是否降低
- Accuracy 或 F1 是否下降
- 是否存在 performance–fairness trade-off

---

## 5. 模型評估

### 簡單說明

比較三個模型的預測表現。

### 評估指標

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Dropout Recall
- Confusion Matrix

### 最重要指標

**Dropout Recall**

原因是如果真正可能退學的學生沒有被模型找出來，學校可能錯過協助學生的機會。

### 模型比較表

| Model | Accuracy | Macro F1 | Dropout Recall |
| --- | --- | --- | --- |
| Baseline | TBD | TBD | TBD |
| MLP | TBD | TBD | TBD |
| MLP + MinDiff | TBD | TBD | TBD |

---

## 6. Fairness Evaluation

### 簡單說明

檢查模型是否對某一群學生比較容易誤判。

### 建議敏感屬性

優先選擇：

- Gender

也可以考慮：

- Age group
- Scholarship status
- Nationality

### 評估內容

- 各群體 Accuracy
- 各群體 Dropout Recall
- False Positive Rate
- False Negative Rate
- 群體間的 FPR Gap
- 群體間的 FNR Gap

### 比較方式

| Model | Group FPR Gap | Group FNR Gap |
| --- | --- | --- |
| MLP | TBD | TBD |
| MLP + MinDiff | TBD | TBD |

### 注意

不能宣稱模型完全沒有 bias，只能說：

> MinDiff 是否降低了目前定義與量測到的群體差異。
> 

---

## 7. Explainability

### 簡單說明

說明模型為什麼將某位學生預測成 Dropout、Enrolled 或 Graduate。

### 建議方法

- SHAP

### 需要完成的內容

1. Global Explanation
    - 哪些特徵整體最重要
2. Local Explanation
    - 一筆學生資料中，哪些特徵提高或降低退學風險

### 介面顯示範例

```
Predicted Outcome: Dropout

Main Contributing Factors:
- Low number of approved courses
- Tuition fees not up to date
- Low admission grade
```

### 注意

SHAP 顯示的是模型判斷依據，不代表真正的因果關係。

---

## 8. Data Card

### 簡單說明

說明資料集從哪裡來、如何處理、有哪些限制與風險。

### 內容

- Dataset name
- Dataset source
- Dataset size
- Feature description
- Target labels
- Data preprocessing
- Sensitive attributes
- Privacy risks
- Bias risks
- Data leakage risks
- Intended use
- Prohibited use
- Dataset limitations

### 檔案

```
DATA_CARD.md
```

---

## 9. Model Card

### 簡單說明

記錄模型的設計、用途、效能、限制與安全風險。

### 內容

- Model name
- Model version
- Model architecture
- Training data
- Evaluation results
- Fairness results
- Intended use
- Out-of-scope use
- Model limitations
- Ethical risks
- Security risks
- Human oversight
- Deployment status

### 檔案

```
MODEL_CARD.md
```

---

## 10. Inference 介面

### 簡單說明

建立一個可以輸入學生資料並執行模型預測的介面。

### 建議工具

- Streamlit

### 介面流程

```
輸入學生資料
    ↓
檢查輸入格式
    ↓
套用 Encoder 與 Scaler
    ↓
執行模型推論
    ↓
顯示預測機率
    ↓
顯示 SHAP Explanation
    ↓
顯示人工審核提醒
```

### 顯示內容

- Predicted Outcome
- Dropout Probability
- Enrolled Probability
- Graduate Probability
- Risk Level
- Important Features
- Model Version
- Prediction Time

---

## 11. Security

### 簡單說明

說明系統可能面臨的安全與使用風險，以及如何降低風險。

### 主要風險

- 學生個資外洩
- 未授權存取
- 輸入異常資料
- Data Leakage
- Model Extraction
- Bias Amplification
- Training-serving Skew
- 過度依賴模型結果

### 預計控制方式

- 資料去識別化
- Input Validation
- 不紀錄完整學生個資
- 使用相同 preprocessing pipeline
- 紀錄模型版本
- 紀錄 inference log
- 加入 Human Review
- 顯示使用限制

### 系統警告

```
This prediction is intended for early intervention support only.

It must not be used as the sole basis for academic,
disciplinary, or enrollment decisions.
```

---

## 12. Deployment Video

### 簡單說明

錄製模型從啟動到完成推論的操作流程。

### 影片內容

1. 開啟推論介面
2. 輸入學生資料
3. 執行預測
4. 顯示三類機率
5. 顯示風險等級
6. 顯示 Explainability
7. 顯示 Human Review 警告
8. 顯示模型版本與推論時間

---

## 13. GitHub

### 必須包含

```
README.md
DATA_CARD.md
MODEL_CARD.md
requirements.txt
模型訓練程式
模型評估程式
Fairness 程式
Inference 程式
Streamlit App
Deployment Video Link
```

### 建議結構

```
student-dropout-project/
├── README.md
├── requirements.txt
├── notebooks/
├── src/
├── app/
├── models/
├── results/
└── reports/
    ├── DATA_CARD.md
    ├── MODEL_CARD.md
    └── SECURITY_RISK_REGISTER.md
```

---

# 四、三人詳細分工

## 組員 A：Data 與 Fairness

負責：

- 資料集理解
- 資料前處理
- 資料切分
- Data Leakage 檢查
- 敏感屬性分析
- Fairness metrics
- Data Card
- 閱讀學生退學公平性 paper

報告內容：

- 問題背景
- Dataset
- Data Card
- Fairness 問題

---

## 組員 B：Model 與 Evaluation

負責：

- Baseline
- MLP
- MLP + MinDiff
- Hyperparameter tuning
- 模型評估
- Confusion Matrix
- Model Card
- 閱讀 MinDiff paper

報告內容：

- 模型架構
- MinDiff
- 模型比較
- Model Card

---

## 組員 C：Explainability、Security 與 Deployment

負責：

- SHAP
- Inference 介面
- Input Validation
- Security Risk Register
- Inference Log
- Streamlit Deployment
- Deployment Video
- GitHub 整理

報告內容：

- Explainability
- Security
- Inference flow
- Deployment Demo

---

# 五、8–10 分鐘簡報分配

| 組員 | 時間 | 內容 |
| --- | --- | --- |
| 組員 A | 約 2.5–3 分鐘 | 背景、資料集、Data Card、Fairness |
| 組員 B | 約 2.5–3 分鐘 | Baseline、MLP、MinDiff、評估、Model Card |
| 組員 C | 約 2.5–3 分鐘 | Explainability、Security、Inference、Deployment |
| 結尾 | 約 30 秒 | 結論與 GitHub、分工 |

---

# 六、Paper

讀讀幾篇Paper 作為支持各階段作法

# 七、最終要回答的問題

我們的專案最後希望回答三個問題：

1. MLP 是否比傳統 Baseline 更適合預測學生退學與學業成果？
2. 加入 MinDiff 後，是否能降低不同學生群體之間的預測差異？
3. 如何透過 Explainability、Security、Data Card、Model Card 與 Human Review，讓模型更適合被安全地使用？


----
REFERNCE SAMPLE
https://sites.google.com/mail.ntut.edu.tw/mlsecops/courses/mlsec-project