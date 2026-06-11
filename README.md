# 學生退學與學業成果預測系統

**A Secure Deep Learning System for Predicting Students' Dropout and Academic Success**

使用學生的入學、課業與背景資料，預測學生的學業結果。建模階段聚焦於最關鍵的
**二分類任務（Dropout 退學 vs Graduate 畢業，已移除 Enrolled 在學樣本）**，
並涵蓋 Explainability、Security、Fairness 與可操作的 Inference 介面。

> 資料集：[UCI – Predict students' dropout and academic success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

## 快速開始 · How to Use

### Step 1 — 安裝環境 (Setup)
```bash
python3 -m venv .venv && source .venv/bin/activate   # 建議使用虛擬環境
pip install -r requirements.txt
```

### Step 2 — 準備模型 (Get a model) — **二選一**
兩種方式都會產出 `models/model.pt` + `models/preprocessor.joblib`，皆符合推論契約、`load_checkpoint` 可直接載入，**介面不需修改**。

```bash
# A) 佔位模型：不需真實資料，最快，先把介面跑起來看 / demo 流程
python -m scripts.train_placeholder

# B) 正式模型：需要真實 data.csv（含 Target 欄；不在 repo 內，請自行放置）
python -m scripts.export_model --data notebooks/data.csv
```

| | A. `train_placeholder` | B. `export_model` |
| --- | --- | --- |
| 資料 | 合成 / UCI（不需 `data.csv`） | 真實 `data.csv` |
| 模型 | 佔位「假模型」，預測無意義 | 正式模型 |
| 方法 | 普通 BCE | Focal + MinDiff，**5-fold soft-voting ensemble**（= 報告中那一顆，方法與組員 B 的 notebook 一致） |
| 用途 | 沒資料也能跑介面 | 正式展示 / 上線 |

> 💡 `export_model` 產出的 5 顆成員以 soft-voting 整合、序列化為單一 `models/model.pt`；
> 推論層 (`src/inference`) 透過 `EnsembleMLP` 自動做機率平均，介面與推論程式皆不需修改。

### Step 3 — 啟動介面 (Run the app)
```bash
streamlit run app/streamlit_app.py     # 開瀏覽器 http://localhost:8501
```
操作：左側填入學生資料 → 按 **Run Assessment** → 右側即時呈現
**風險等級 → 建議行動 → 主要風險因子 → 各類別機率**，並顯示使用限制與人工審核提醒。

<p align="center">
  <img src="results/main.png" alt="Dropout Risk Prediction interface" width="527">
</p>

### Step 4 —（選用）產生全域解釋圖
```bash
python -m scripts.plot_global_importance       # → results/global_importance.png
```

> ⚠️ 用 A 跑出來的 `models/model.pt` 為 **placeholder**，僅供串接介面與 demo；
> 正式效能請用 B（或替換成組員 B 訓練好的權重），介面與推論層皆不需更動。

---

## 推論流程 · Inference Flow

介面（`app/`）不做 ML，只收輸入、串流程、顯示；推論與解釋都委派給 `src/`：

```
填表(11 欄) → ① validate_input → ② predict → ③ explain_record → interpret 翻成人話 → 顯示＋log
              (security)         (inference)   (explain)
```

| 模組 | 職責 |
| --- | --- |
| `inference.py` | **風險多高** — 前處理(11→14)＋模型(DropoutMLP/EnsembleMLP) → 機率 / 風險等級 |
| `explain.py` | **為什麼** — 各特徵推高/降低退學風險的貢獻（SHAP / 梯度後援） |
| `interpret.py`／`security.py` | 數字翻成人話 ／ 輸入驗證＋去識別化 log |

> 💡 介面只依賴 `predict()` / `explain_record()` 兩個介面 → 模型從單顆換成 **5-fold ensemble**，`inference / explain / app` 一行未改。

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
│   ├── interpret.py         # 把模型輸出翻成教師可讀的風險因子 / 建議行動
│   └── security.py          # Input Validation / 去識別化 / Inference Log
├── scripts/
│   ├── train_placeholder.py # 佔位模型 (合成資料, demo 用)
│   ├── export_model.py      # 正式模型 (真實資料, 5-fold MinDiff ensemble)
│   └── plot_global_importance.py # 全域解釋圖 -> results/global_importance.png
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
2. **checkpoint 格式** 使用 `src/model.py` 的存檔函式，並一併輸出對應的 `models/preprocessor.joblib`：
   - 單一模型 `save_checkpoint(...)`：
     `state_dict / input_dim / dropout_1 / dropout_2 / num_classes / model_version / label_names`
   - 5-fold ensemble `save_ensemble_checkpoint(...)`：
     `ensemble（各成員 state_dict）/ num_members / input_dim / dropout_1 / dropout_2 / num_classes / model_version / label_names`
   - `load_checkpoint(...)` 會自動辨識上述兩種格式，前端與推論層皆不需更動。

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

## Data Card（資料卡）

* **資料集名稱（Dataset name）**：
  Predict Students' Dropout and Academic Success

* **資料來源（Dataset source）**：
  UCI Machine Learning Repository（Dataset ID 697）

* **原始資料集規模（Dataset size）**：

  * 資料筆數（Instances）：4,424 位學生
  * 特徵數量（Features）：36 個輸入變數
  * 目標類別（Target classes）：Dropout、Enrolled、Graduate

* **特徵描述（Feature description）**：
  36 個特徵涵蓋以下面向：

  * 學生人口統計資訊（例如：性別、入學年齡、國籍）
  * 學術背景（例如：入學成績、前一學歷成績）
  * 家庭與社經背景資訊
  * 學業表現（第一、二學期課程相關資訊）
  * 總體經濟指標（GDP、通貨膨脹率、失業率）

* **原始目標標籤（Target labels）**：

  * Dropout（退學）
  * Enrolled（在學）
  * Graduate（畢業）

* **資料前處理（Data preprocessing）**：

  1. 移除缺失值與重複資料。
  2. 將原始三分類目標轉換為二元分類問題：

     * Dropout（退學）
     * Non-Dropout（僅包含 Graduate）
  3. 對數值型特徵進行標準化處理（Feature Scaling）。
  4. 使用 SMOTETomek 方法處理類別不平衡問題。
  5. 在模型開發過程中進行訓練集／測試集切分，以及 5 折分層交叉驗證（5-Fold Stratified Cross-Validation）。
  6. 建立額外的特徵工程變數，包括：

     * 第一學期課程通過率（1st semester pass rate）
     * 第二學期課程通過率（2nd semester pass rate）
     * 成績變化量（Grade change）

* **敏感屬性（Sensitive attributes）**：

  * Tuition fees up to date（學費是否按時繳交）

* **隱私風險（Privacy risks）**：
  「學費是否按時繳交（Tuition fees up to date）」反映學生的財務狀況，因此可視為敏感個人資訊。雖然資料集中不包含姓名、學號等直接識別資訊，但若與其他人口統計或學業資料結合，仍可能提高推測個人經濟狀況的風險。因此，在資料使用與分享過程中，應採取適當的存取控制與資料保護措施。

* **偏誤風險（Bias risks）**：
  「學費是否按時繳交（Tuition fees up to date）」可能間接反映學生的社經地位。若模型過度依賴此特徵，經濟弱勢學生可能被系統性地判定為較高的退學風險。因此，模型預測結果應作為提供額外支持與介入措施的參考依據，而非作為限制教育機會或施加處分的工具。

* **預期用途（Intended use）**：
  用於及早識別具有退學風險的學生，以協助學校提供學業輔導、經濟補助及其他針對性的支持措施。

* **禁止用途（Prohibited use）**：
  本模型不得作為以下決策的唯一依據：

  * 學生退學或開除決策
  * 入學資格審核
  * 獎學金分配
  * 紀律處分
  * 任何影響學生權益且未經人工審查的自動化決策

* **資料集限制（Dataset limitations）**：

  1. 資料來自單一高等教育機構，因此模型結果未必能推廣至其他學校或國家。
  2. 資料反映特定教育制度與社會經濟環境，可能無法完全代表其他學生族群。
  3. 影響退學的重要因素（如心理健康、學習動機、家庭支持及學習投入程度）並未納入資料集中。
  4. 總體經濟變數屬於宏觀指標，無法精確反映個別學生的財務狀況。
  5. 第二學期學業表現相關特徵的存在，可能降低模型在真正早期預警情境中的適用性。
  6. 原始資料存在類別不平衡問題，因此需透過重採樣與加權方法進行調整，可能影響模型的穩定性與泛化能力。

---

# 📋 Model Card（組員 B 113AB8046 填寫）

- **Model name**: Fair-Predict Student Retention Model (公平導向型學生退學預警模型)
- **Model version**: `1.0.0` *(已同步更新至 `src/schema.py` 中的 `MODEL_VERSION`)*
- **Model architecture**: 深度前饋神經網路 (Multi-Layer Perceptron, MLP)
  * **Layer Stack**: `Input (14 features) → Dense(64) + BatchNorm + ReLU + Dropout(0.4) → Dense(32) + BatchNorm + ReLU + Dropout(0.3) → Dense(16) + ReLU → Linear(1)`
  * **Loss Function**: `Focal Loss` (處理標籤不平衡) + `MMD MinDiff Loss` (公平性約束)
  * **Optimization**: `Optuna` 自動化貝氏超參數尋優 + `5-Fold` 交叉驗證
- **Training data**: 
  使用經去識別化之校園學生學術與社會經濟特徵數據 (`train_scaled.csv`)。內含 **14 個核心特徵**：包含經 SHAP 篩選之 11 個原始精選特徵，以及本團隊手動創造之 3 個核心特徵（第一、二學期學分通過率、成績變動率），未執行任何破壞資料分佈之 SMOTE 過採樣。

- **Evaluation results**: 
  本專案所有模型均在嚴格的 **5-Fold 交叉驗證** 架構下訓練，並於獨立的黃金測試集（Test Set）進行整合預測（Soft Voting），最終定量評估結果如下：

| Model Stage | Accuracy | Macro F1 | Dropout Recall | ROC-AUC (整合測試集) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** (5-Fold 邏輯回歸) | 0.94 | 0.93 | 0.90 | 0.9685 |
| **MLP** (5-Fold) | 0.94 | 0.92 | 0.90 | **0.9730** |
| **MLP + MinDiff** (Optuna調整參數) | 0.92 | 0.91 | 0.90 | 0.9485 |

> 💡 **5-Fold 內部驗證穩定度穩定度**
> * **MLP + MinDiff 內部平均驗證 AUC**: $0.9268 \pm 0.0146$
> * **MLP + MinDiff 內部平均驗證 Gap**: $22.67\% \pm 18.32\%$

- **Fairness results**: 
  本專案選定 `Tuition fees up to date`（學費是否按時繳納）作為受保護之**敏感屬性（Protected Attribute）**。用以衡量模型是否對經濟弱勢學生產生嚴重的系統性歧視。
  
  * **Baseline (FPR Gap)**: **62.54%** 🚨 *(敏感群體 FPR: 66.67% / 對照群體 FPR: 4.13%)*
  * **Pure MLP (FPR Gap)**: **78.52%** 🚨 *(敏感群體 FPR: 83.33% / 對照群體 FPR: 4.82%)*
  * **MLP + MinDiff (FPR Gap)**: 👑 **8.18%** ✨ *(敏感群體 FPR: 16.67% / 對照群體 FPR: 8.49%)*
  
  **去偏誤成效洞察：**  
  透過 Optuna 最大化自訂綜合指標 $\text{Score} = \text{Mean AUC} - \text{Mean FPR Gap}$。實驗證實，最終模型成功**消滅了 86.7% 的歷史財務階層偏見**，且僅犧牲了極微幅（2.39%）的預測上限（AUC），達成了工業級極致完美的 *Fairness-Accuracy Trade-off*。

- **Intended use**: 
  專門部署於大專院校第一學年結束時之教務數據自動化審查。旨在篩選出具備高退學風險（Dropout）的學生，提供教務處、輔導中心與各班導師作為**「主動發起關懷訪談、心理輔導與全方位校園資源分配」**的早期介入核心依據。

- **Out-of-scope use**: 
  * ❌ **絕對禁止**將本模型之預測標籤或機率，直接用於學生獎學金評定、助學金發放、優良學生選拔之扣分依據。
  * ❌ **絕對禁止**任何校園行政單位在未經人工覆核前，利用此自動化模型直接對學生進行強制退學、留級或任何處罰性之行政決策。

- **Model limitations**: 
  * **時間滯後性**: 核心特徵高度依賴學期末結算之學分與成績，對於學期中途因財務突發危機或志趣不合而「突發性退學」的學生，本模型存在預警延遲。
  * **二元簡化偏誤**: 目前模型將學費繳納狀態簡化為二元（低於中位數 vs 高於中位數），無法精準捕捉更為動態與連續性的家庭財務波動。

- **Ethical risks**: 
  若放任未經 MinDiff 校正的模型（如 Pure MLP）直接上線，模型會產生嚴重的**財務走捷徑（Shortcut Learning）**惡性偏誤。系統將僅憑學生「家庭清寒/學費未按時繳納」此一與學術能力無關的欄位，就給予高達 66.67% 的機率盲目誤判其必定退學，這將導致校園行政資源對特定經濟階層產生結構性歧視與二次標籤化傷害。

- **Security risks**: 見下方 Security 章節
- **Human oversight**: 
  **負責任 AI 核心原則（Human-in-the-loop）**：模型的預測結果與風險機率僅作為校園一線輔導人員的「輔助參考線索」。最終的關懷介入決策、實質因應措施與行政判斷，必須保留 100% 的人工審核與專業導師評估。

- **Deployment status**: 
  * 🟡 **Status**: `Contract-ready`（模型已訓練、推論介面已串通，尚差 1 步落盤）
  * **已完成**：5-Fold 交叉驗證與去偏誤評估通過（整合 FPR Gap < 10%）；`src/` 模型契約（14 特徵、二分類、checkpoint 格式）與 Streamlit 推論介面已用 placeholder 跑通。
  * **待完成（1 步）**：將訓練好的權重以 `save_checkpoint(...)` 輸出為 `models/model.pt`、並存出對應 `preprocessor.joblib`，替換 placeholder 後即可上線，介面無需修改。

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
