"""Single source of truth for the model's input/output contract.

組員 B 實際交付的模型是 **二分類 (Dropout vs Graduate)**，輸入為
**14 個特徵 = 11 個原始精選特徵 + 3 個自創特徵**（已移除 Enrolled 類別）。
只要遵守同一份 RAW_FEATURES / FEATURE_ORDER / LABELS，
就能直接替換 models/model.pt，前端 (app) 與推論層 (inference) 不需修改。

特徵分兩層:
    1. RAW_FEATURES (11)     : 使用者在介面實際輸入的原始欄位 (= FEATURE_SCHEMA)。
    2. ENGINEERED_FEATURES(3): 由原始欄位推導，使用者不需輸入 (見 preprocessing.py)。
    FEATURE_ORDER = RAW_FEATURES + ENGINEERED_FEATURES (14)
    → 即模型 / scaler 真正吃進去的向量順序，務必與訓練時一致。

每個 raw feature 欄位:
    name    : 與 UCI 資料集欄位名稱一致
    kind    : "num" (數值) 或 "cat" (類別, 以整數 code 表示)
    default : 前端表單預設值
    key     : True 表示在 Streamlit 主畫面顯示, False 收進「進階欄位」
    help    : 簡短說明 (可選)
"""

# 二分類任務: 0 = Graduate (對照/負類), 1 = Dropout (退學/正類)
# 與訓練程式 (Target == 'Dropout').astype(int) 一致。
LABELS = ["Graduate", "Dropout"]

# ---- 11 個原始精選特徵 (使用者實際輸入；順序須與訓練程式一致) ----
FEATURE_SCHEMA = [
    {"name": "Curricular units 1st sem (approved)", "kind": "num", "default": 5, "key": True,
     "help": "1st-semester approved units · 第一學期通過科目數"},
    {"name": "Curricular units 2nd sem (approved)", "kind": "num", "default": 5, "key": True,
     "help": "2nd-semester approved units · 第二學期通過科目數"},
    {"name": "Curricular units 1st sem (enrolled)", "kind": "num", "default": 6, "key": False,
     "help": "1st-semester enrolled units · 第一學期修課科目數"},
    {"name": "Curricular units 2nd sem (enrolled)", "kind": "num", "default": 6, "key": False,
     "help": "2nd-semester enrolled units · 第二學期修課科目數"},
    {"name": "Curricular units 1st sem (grade)", "kind": "num", "default": 12.0, "key": False,
     "help": "1st-semester average grade, 0–20 · 第一學期平均成績"},
    {"name": "Curricular units 2nd sem (grade)", "kind": "num", "default": 12.0, "key": False,
     "help": "2nd-semester average grade, 0–20 · 第二學期平均成績"},
    {"name": "Scholarship holder", "kind": "cat", "default": 0, "key": True,
     "help": "Scholarship holder, 0/1 · 是否有獎學金"},
    {"name": "Tuition fees up to date", "kind": "cat", "default": 1, "key": True,
     "help": "Sensitive attribute — tuition paid on time, 1=yes 0=no · 敏感屬性：學費是否按時繳清"},
    {"name": "Application mode", "kind": "cat", "default": 1, "key": False,
     "help": "Application route code · 入學管道代碼"},
    {"name": "Admission grade", "kind": "num", "default": 130.0, "key": True,
     "help": "Admission grade, 0–200 · 入學成績"},
    {"name": "Previous qualification (grade)", "kind": "num", "default": 130.0, "key": False,
     "help": "Previous qualification grade, 0–200 · 前一學歷成績"},
]

RAW_FEATURES = [f["name"] for f in FEATURE_SCHEMA]

# ---- 3 個自創特徵 (由 raw 推導，見 preprocessing.add_engineered_features) ----
ENGINEERED_FEATURES = ["1st_sem_pass_rate", "2nd_sem_pass_rate", "grade_change"]

# 模型 / scaler 真正吃進去的 14 維向量順序
FEATURE_ORDER = RAW_FEATURES + ENGINEERED_FEATURES

# 敏感屬性 (與 Fairness / Security 章節對應)：學費繳納狀態代理社經地位
SENSITIVE_ATTRIBUTES = ["Tuition fees up to date"]

MODEL_VERSION = "1.0.0"
