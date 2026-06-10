# 03 — Fairness Evaluation（組員 A/B）

對應需求書第 6 節。

## 1. 敏感屬性
優先：Gender。其他可考慮：Age group / Scholarship status / Nacionality。

## 2. 群體評估指標
各群體：Accuracy、Dropout Recall、False Positive Rate、False Negative Rate
群體間：FPR Gap、FNR Gap

## 3. MinDiff 前後比較

| Model | Group FPR Gap | Group FNR Gap |
| --- | --- | --- |
| MLP | TBD | TBD |
| MLP + MinDiff | TBD | TBD |

## 4. 結論
- 公平性差距是否降低？_TODO_
- Accuracy / F1 是否下降（performance–fairness trade-off）？_TODO_

> 注意：不能宣稱模型完全沒有 bias，只能說 MinDiff 是否降低了**目前定義與量測到**的群體差異。
