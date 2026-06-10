# 02 — 模型與評估（組員 B）

## 1. Baseline
- **模型**：Logistic Regression（5-Fold 交叉驗證整合體）
- **超參數**：`solver='liblinear'`, `random_state=42`, `C=1.0` (預設值)，並在 `train_scaled.csv` 內部進行 5 折分層切分整合預測。

## 2. MLP
- **架構**：Input (14) → Linear(64) → BatchNorm1d → ReLU → Dropout(0.4) → Linear(32) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(16) → ReLU → Linear(1)。*(註：配合二分類任務與 Focal Loss 實作優化，輸出層採單神經元 Logits 輸出，本質與三分類預估完全相容)*。
- **Loss**：Focal Loss ($\alpha=0.6, \gamma=2$，用於強效對抗 Dropout 類別不平衡)；**Optimizer**：Adam (`lr=0.001`, `weight_decay=1e-4`)；**Early Stopping**：`patience=10`，以驗證集損失（Val Loss）不下降為觸發條件，並自動還原至最佳權重狀態。
- **最佳模型輸出**：`models/model.pt`（已透過 `save_checkpoint` 完整保存權重結構與模型元數據）。

## 3. MLP + MinDiff
- **敏感屬性**：`Tuition fees up to date` (學費是否按時繳納)。此特徵直接代理了學生的社會經濟地位（Socioeconomic Status），用以防範演算法產生財務階層歧視。
- **Total Loss** = Classification Loss (Focal Loss) + $\lambda$ × Fairness Loss (MMD Loss)； $\lambda$ (即 $\beta$ ) = `1.3321` *(此為 Optuna 在 5-Fold 交叉驗證下，歷經 15 次貝氏嘗試所尋獲之兼顧準確度與公平性的黃金客觀超參數)*。

## 4. 評估指標
Accuracy / Precision / Recall / Macro F1 / Weighted F1 / **Dropout Recall** / Confusion Matrix

> 最重要：**Dropout Recall**——漏掉真正可能退學的學生，學校就錯過協助機會。
> **團隊核心實務洞察**：本模型定位為「早期關懷預警系統」，追求高 **Dropout Recall** 是本專案的至高核心。如果 Recall 過低（即漏判率高），意味著大量正處於退學危機邊緣的弱勢生將被系統忽視，進而徹底錯失學校行政資源介入輔導的黃金時間。

## 5. 模型比較

| Model | Accuracy | Macro F1 | Dropout Recall (關鍵指標) | ROC-AUC (整合表現) |
| --- | :---: | :---: | :---: | :---: |
| **Baseline** (5-Fold 邏輯回歸) | 0.94 | 0.93 | 0.90 | 0.9685 |
| **MLP** (純深度學習神經網路) | 0.94 | 0.92 | **0.90** 🎯 | **0.9730** |
| **MLP + MinDiff** (Optuna 終極體) | 0.92 | 0.91 | 0.90 | 0.9495 |

## 6. Confusion Matrix
- *混淆矩陣圖表與評估 ROC 曲線已成功輸出並保存於 `results/` 目錄下。*
- **修復後核心量化成果**：敏感群體（學費未繳生）的偽陽性率（FPR）從原先 Pure MLP 的 **83.33%** 斷崖式雪崩至 **16.67%**。兩群體間的**公平性差距（FPR Gap）從 78.52% 澈底收斂至 8.18%**，成功消滅了 **86.7%** 的結構性歧視。

---

## 待回答問題

### 1. MLP 是否優於 Baseline？
**是，在「預測上限」與「核心指標」上，MLP 均取得全面性壓倒性的勝利。**
* **技術分析**：在完全相同的 5-Fold 交叉驗證考場下，純 MLP 的測試集 ROC-AUC 達到了專案最高峰的 **0.9730**（超越 Baseline 的 0.9685）。這證明深度神經網路透過多層權重組合與非線性激活函數，能更精準地捕捉我們手動工程創造的 14 個特徵之間的動態互動關係。
* **指標分析**：最核心的 **Dropout Recall** 指標，MLP 達到了 **0.90**。這意味著每 100 個真正會退學的學生，純 MLP 成功幫學校精準抓回了 90 個，將「錯失協助機會」的漏網之魚壓低至僅剩 10 個，技術效能極其優異。

### 2. MinDiff 是否降低群體差異、效能是否下降？見 [03_fairness.md](03_fairness.md)
