# 01 — 資料分析與前處理（組員 A）

對應需求書第 1、8 節。README 的 Data Card 為摘要，本檔放詳細方法與結果。

## 1. 資料集概覽
- 來源：UCI ML Repository (ID 697)
- 筆數 / 欄位數：_TODO_
- 36 個特徵清單與型別：見 `src/schema.py` 的 `FEATURE_SCHEMA`
- 預測標籤：Dropout / Enrolled / Graduate（類別分布圖：_TODO 放 results/_）

## 2. 資料品質檢查
- 缺失值：_TODO_
- 重複值：_TODO_
- 異常值（超出合理範圍）：_TODO_

## 3. 前處理步驟
- 類別欄位編碼方式：_TODO_
- 數值欄位標準化：StandardScaler（與推論共用，見 `src/preprocessing.py`）
- 切分 Train / Val / Test 比例與隨機種子：_TODO_

## 4. Data Leakage 檢查
- 哪些欄位可能洩漏未來資訊？（例：第二學期成績）_TODO_
- 處理方式：_TODO_

## 5. 敏感屬性
- Gender / Age at enrollment / Nacionality / Scholarship holder
- 各群體分布：_TODO_

## 產出
- 前處理程式：_TODO 路徑_
- 類別分布圖：_TODO_
- Train / Val / Test 資料：_TODO_
