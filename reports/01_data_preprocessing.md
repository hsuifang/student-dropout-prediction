# 01 — 資料分析與前處理（組員 A 113AB8049）

---

## 1. 資料集概覽
- **來源**：UCI Machine Learning Repository (Dataset ID: 697) - "Predict students' dropout and academic success"。
- **總筆數 / 原始欄位數**：原始資料包含 4,424 筆樣本與 37 個欄位（36 個特徵 + 1 個預測標籤 `Target`）。
- **36 個特徵清單與型別**：主要涵蓋：學生個人背景、入學前學業表現、社會經濟因素（如是否按時繳納學費 `Tuition fees up to date`、是否享有獎學金 `Scholarship holder`）及第一、二學期學業表現。
- **預測標籤（Target）**：
  - 原始標籤包含三類：`Dropout`（退學）、`Graduate`（畢業）、`Enrolled`（在讀）。
  - **專業處理決策**：為了聚焦於預測學生是否會走向「退學」的終極命運，本專案在載入資料時**實施了嚴格的資料純淨化，過濾移除處於模糊狀態的 `Enrolled` 樣本**。
  - **二元分類轉換**：將標籤轉換為正負樣本：`Dropout`（退學）為 `1`，`Graduate`（畢業）為 `0`。
  - **過濾後退學比例**：經移除 `Enrolled` 後，純淨訓練集的真實退學率（Base Rate）為 **39.15%**，屬於輕微不平衡資料，後續機器學習流程將透過分層切分與損失函數進行優化。

---

## 2. 資料品質檢查與敘述統計（Descriptive Statistics）
經對原始 4,424 筆資料進行檢查與敘述統計分析，特徵的分佈表現如下：

- **缺失值與重複值**：全欄位無缺失值（Count 皆為 4,424），且無重複學生紀錄。
- **入學特徵分佈**：
  - `Age at enrollment`（入學年齡）：平均值為 **23.27 歲**，標準差 **7.59**。最小年齡為 **17 歲**，但最大年齡達 **70 歲**（第 75 百分位數為 25 歲），顯示存在高齡在職進修之極端值。
  - `Admission grade`（入學成績）：平均值為 **126.98 分**，分佈介於 **95 分至 190 分**之間。
- **學期修課表現（第一、二學期對比異常值檢查）**：
  - `Curricular units 1st sem (approved)`（第一學期通過學分數）：平均通過 **4.71 門**，最高達 **26 門**。
  - `Curricular units 2nd sem (approved)`（第二學期通過學分數）：平均通過 **4.44 門**，最高達 **20 門**。
  - 成績部分，第一學期平均成績為 **10.64 分**，第二學期平均成績為 **10.23 分**。在修課通過率計算中，由於部分學生註冊學分數為 0（Min 分佈為 0），存在分母為零之風險，已於特徵工程進行處理。
- **總體經濟與外部環境特徵**：
  - `Unemployment rate`（失業率）：平均高達 **11.57%**（區間 7.6% - 16.2%）。
  - `Inflation rate`（通貨膨脹率）：平均為 **1.23%**。
  - `GDP`（國內生產總值變動率）：平均接近於零（**0.002**），最低曾跌至 **-4.06%**，反映出外部經濟波動對學生就學穩定度之潛在衝擊。

---

## 3. 前處理與特徵篩選步驟

本專案拒絕盲目餵入原始特徵，規劃了完整的「特徵工程設計」、「進階特徵篩選」到「資料切分與標準化」的流程：

### 3.1 特徵工程設計（Feature Engineering Design）
為了深化模型對學生學業連續性與經濟壓力的捕捉，自創了以下 3 組核心特徵：

| 衍生特徵 (Feature) | 核心核心含意 (Meaning) | 建構計算方法 (Method) |
| :--- | :--- | :--- |
| `1st_sem_pass_rate`<br>`2nd_sem_pass_rate` | 學期課程通過率<br>(Course Pass Rate) | $\frac{\text{該學期通過之學分數 (approved)}}{\text{該學期註冊之總學分數 (enrolled)} + 10^{-8}}$ |
| `grade_change` | 學期成績趨勢變動<br>(Grade Variation) | $\text{第二學期總分 (2nd sem grade)} - \text{第一學期總分 (1st sem grade)}$<br>*捕捉學習曲線是進步或下滑（高風險）。* |
| `financial_status` | 學生財務綜合狀態<br>(Financial Status) | $\text{是否為獎學金持有者 (Scholarship holder)} + \text{學費是否按時繳納 (Tuition fees paid)}$ |

### 3.2 機器學習特徵篩選（Feature Selection）
為了確保模型的輕量化與高解釋性，專案採取了結合「機器學習解釋性」與「統計學資訊量」的四階段嚴格篩選流程：

1. **階段一（訓練基準 XGBoost 模型）**：首先建立一個基於樹模型的初始分類器（XGBoost Model），以此作為特徵重要性分析的算力底座與基礎。
2. **階段二（計算 SHAP 歸因值排名）**：引入 SHAP (SHapley Additive exPlanations) 賽局理論歸因分析，導出前 20 大對模型預測影響力最深遠的特徵排名（Top 20 Features）。由 SHAP 密集度分佈圖可觀察到：`Curricular units 2nd sem (approved)` 以及自創的財務狀態與學分變動，其 SHAP 值極化表現最為顯著。
3. **階段三（人工篩選可控特徵）**：在 Top 20 變數中，主動過濾掉無法透過學校政策改變的既定事實（如父母職業、出生地等），**手動精選出精選出 11 個具有「人類或政策干預潛力」（Potential for human or policy intervention）的學業表現與財務變數**。
4. **階段四（互資訊分析驗證）**：為強化篩選特徵的可靠度並支持後續的交叉驗證，進一步計算精選特徵與目標標籤（Target Variable）之間的**互資訊得分（Mutual Information Scores）**，最終敲定進入核心訓練模型的特徵矩陣：

| 篩選特徵 (Feature) | 互資訊得分 (Mutual Information) |
| :--- | :--- |
| `Curricular units 2nd sem (approved)` | 0.309975 |
| `Curricular units 1st sem (approved)` | 0.246007 |
| `Curricular units 2nd sem (grade)` | 0.232004 |
| `Curricular units 1st sem (grade)` | 0.180949 |
| `Tuition fees up to date` | 0.080529 |
| `Scholarship holder` | 0.049612 |
| `Application mode` | 0.049026 |
| `Previous qualification (grade)` | 0.043111 |
| `Admission grade` | 0.038124 |

### 3.3 資料標準化（Normalization）
- 使用 `sklearn.preprocessing.StandardScaler` 進行 $Z$-Score 標準化（將資料平移至均值 $\mu=0$，標準差 $\sigma=1$）。
- 轉換後保留 DataFrame 的欄位名稱（Column Names），以利後續階段進行 MinDiff 敏感欄位對齊與公平性審計。

### 3.4 嚴格的 Train / Test 切分
為了模擬真實世界的泛化表現，資料集進行了一次性黃金獨立測試集切分：
- **切分比例**：Train (80%) / Test (20%)。
- **分層抽樣（Stratified Split）**：啟用 `stratify=y`，確保訓練集與測試集中的退學比例完全一致，皆維持在 **39.15%**。
- **隨記憶種子（Random State）**：統一設定為 `42`，確保數據流可重複再現。
- **切分後規模**：
  - ➔ 成功生成 `train_scaled.csv` (樣本數: **2,904**)
  - ➔ 成功生成 `test_scaled.csv` (樣本數: **726**)

---

## 4. 特徵篩選與選擇策略（Feature Selection）
為了確保模型的輕量化與高解釋性，專案採取了結合「機器學習解釋性」與「統計學資訊量」的四階段嚴格篩選流程：

### 4.1 階段一：訓練基準 XGBoost 模型
- 首先建立一個基於樹模型的初始分類器（XGBoost Model），以此作為特徵重要性分析的算力底座與基礎。

### 4.2 階段二：計算 SHAP 歸因值排名（SHAP Value Rankings）
- 引入 SHAP (SHapley Additive exPlanations) 賽局理論歸因分析，導出前 20 大對模型預測影響力最深遠的特徵排名（Top 20 Features）。
- 由 SHAP 密集度分佈圖（SHAP Summary Plot）可觀察到：`Curricular units 2nd sem (approved)` 以及自創的財務狀態與學分變動，其 SHAP 值極化表現最為顯著。

### 4.3 階段三：人工篩選具備干預價值的「可控特徵」（Actionable Features）
- **專業決策**：在 Top 20 變數中，主動過濾掉無法透過學校政策改變的既定事實（如父母職業、出生地等），**手動精選出具有「人類或政策干預潛力」（Potential for human or policy intervention）的學業表現與財務變數**。

### 4.4 階段四：互資訊分析（Mutual Information Analysis）
- 為強化篩選特徵的可靠度並支持後續的交叉驗證，進一步計算精選特徵與目標標籤（Target Variable）之間的**互資訊得分（Mutual Information Scores）**，最終敲定進入核心訓練模型的特徵矩陣：

| 篩選特徵 (Feature) | 互資訊得分 (Mutual Information) |
| :--- | :--- |
| `Curricular units 2nd sem (approved)` | 0.309975 |
| `Curricular units 1st sem (approved)` | 0.246007 |
| `Curricular units 2nd sem (grade)` | 0.232004 |
| `Curricular units 1st sem (grade)` | 0.180949 |
| `Tuition fees up to date` | 0.080529 |
| `Scholarship holder` | 0.049612 |
| `Application mode` | 0.049026 |
| `Previous qualification (grade)` | 0.043111 |
| `Admission grade` | 0.038124 |

---

## 5. Data Leakage（資訊洩漏）嚴格檢查
資料洩漏是機器學習專案失效的首要原因。本專案進行了最高規格的防禦：

- **標準化洩漏防禦（關鍵步驟）**：
  - **【鐵律】**：`StandardScaler` **只在訓練集 (`X_train`) 上進行 `fit_transform`**，計算出訓練集的均值與標準差。
  - **【鐵律】**：測試集 (`X_test`) **絕對只能進行 `transform`**。測試集偷偷借用訓練集的統計量，嚴格禁止參與 `fit` 過程。
- **未來特徵審查**：第二學期的學業表現特徵（如 `Curricular units 2nd sem` 系列）在學期末是已知數據，若用於預測「該學期中途退學」可能存在時序上的洩漏。但由於目標標籤定義為「最終是否成功畢業或退學」，在全學程回溯性分析中屬於合規特徵。

---

## 6. 敏感屬性與公平性基礎定義
為響應需求書對於模型公平性（Fairness）的嚴格要求，組員 A 在前處理階段確立了關鍵的社會經濟敏感屬性：

- **核心敏感特徵**：`Tuition fees up to date`（學費是否按時繳納）。
- **基準值（Median）界定**：在訓練集上計算該欄位的中央基準值（Median），以此作為切分弱勢群體與對照群體的指標。
- **群體劃分邏輯**：
  - **敏感群體（Sens Group）**：標準化後的學費特徵值 < 訓練集基準值。代表「**未能按時繳納學費的經濟弱勢學生**」。
  - **對照群體（Ref Group）**：標準化後的學費特徵值 $\ge$ 訓練集基準值。代表「**有按時繳納學費的學生**」。
- **前處理產出承接**：此劃分矩陣與標準化特徵將直接落盤儲存，承接給後續的邏輯斯迴歸（Baseline）、深度神經網路（MLP）以及終極的 **MinDiff 仿射對齊公平性修復演算法**，用於追蹤與消除兩組之間的偽陽性率差距（FPR Gap）。

---

## 產出檔案與路徑
- **特徵規格結構定義**：[src/schema.py](src/schema.py)
- **前處理與特徵工程主程式**：`src/prepare_data.py` (執行 `prepare_and_save_pure_datasets('data.csv')`)
- **落盤落庫數據**（純淨且標準化完成，含特徵工程欄位）：
  - 訓練總集：`train_scaled.csv`（規模：2904 行 $\times$ 15 欄，含 `Target_Label`）
  - 黃金獨立測試集：`test_scaled.csv`（規模：726 行 $\times$ 15 欄，含 `Target_Label`）
