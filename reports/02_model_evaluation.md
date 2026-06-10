# 02 — 模型與評估（組員 B）

對應需求書第 2–5、9 節。README 的 Model Card 為摘要，本檔放詳細架構、訓練與評估。

## 1. Baseline
- 模型：Logistic Regression 或 Random Forest（擇一）_TODO_
- 超參數：_TODO_

## 2. MLP
- 架構：Input → Hidden(ReLU + Dropout) → Softmax(3)，參考 `src/model.py`
- Loss：Cross-Entropy；Optimizer：Adam；Early Stopping：_TODO_
- 最佳模型輸出：`models/model.pt`（請用 `src/model.py` 的 `save_checkpoint` 維持格式）

## 3. MLP + MinDiff
- 敏感屬性：Gender（或其他）
- Total Loss = Classification Loss + λ × Fairness Loss；λ = _TODO_

## 4. 評估指標
Accuracy / Precision / Recall / Macro F1 / Weighted F1 / **Dropout Recall** / Confusion Matrix

> 最重要：**Dropout Recall**——漏掉真正可能退學的學生，學校就錯過協助機會。

## 5. 模型比較

| Model | Accuracy | Macro F1 | Dropout Recall |
| --- | --- | --- | --- |
| Baseline | TBD | TBD | TBD |
| MLP | TBD | TBD | TBD |
| MLP + MinDiff | TBD | TBD | TBD |

## 6. Confusion Matrix
- _TODO 放圖於 results/_

## 待回答問題
1. MLP 是否優於 Baseline？_TODO_
2. MinDiff 是否降低群體差異、效能是否下降？見 [03_fairness.md](03_fairness.md)
