"""Single source of truth for the model's input/output contract.

組員 B 訓練的真實模型只要遵守同一份 FEATURE_ORDER 與 LABELS，
就能直接替換 models/model.pt，前端 (app) 與推論層 (inference) 不需修改。

每個 feature 欄位:
    name    : 與 UCI 資料集欄位名稱一致
    kind    : "num" (數值) 或 "cat" (類別, 以整數 code 表示)
    default : 前端表單預設值
    key     : True 表示在 Streamlit 主畫面顯示, False 收進「進階欄位」
    help    : 簡短說明 (可選)
"""

LABELS = ["Dropout", "Enrolled", "Graduate"]

# 與 UCI "Predict students' dropout and academic success" 36 個特徵一致
FEATURE_SCHEMA = [
    {"name": "Marital status", "kind": "cat", "default": 1, "key": False},
    {"name": "Application mode", "kind": "cat", "default": 1, "key": False},
    {"name": "Application order", "kind": "num", "default": 1, "key": False},
    {"name": "Course", "kind": "cat", "default": 9500, "key": False},
    {"name": "Daytime/evening attendance", "kind": "cat", "default": 1, "key": False},
    {"name": "Previous qualification", "kind": "cat", "default": 1, "key": False},
    {"name": "Previous qualification (grade)", "kind": "num", "default": 130.0, "key": False},
    {"name": "Nacionality", "kind": "cat", "default": 1, "key": False},
    {"name": "Mother's qualification", "kind": "cat", "default": 1, "key": False},
    {"name": "Father's qualification", "kind": "cat", "default": 1, "key": False},
    {"name": "Mother's occupation", "kind": "cat", "default": 1, "key": False},
    {"name": "Father's occupation", "kind": "cat", "default": 1, "key": False},
    {"name": "Admission grade", "kind": "num", "default": 130.0, "key": True,
     "help": "入學成績 (0-200)"},
    {"name": "Displaced", "kind": "cat", "default": 0, "key": False},
    {"name": "Educational special needs", "kind": "cat", "default": 0, "key": False},
    {"name": "Debtor", "kind": "cat", "default": 0, "key": True, "help": "是否欠費 (0/1)"},
    {"name": "Tuition fees up to date", "kind": "cat", "default": 1, "key": True,
     "help": "學費是否繳清 (1=是, 0=否)"},
    {"name": "Gender", "kind": "cat", "default": 1, "key": True, "help": "敏感屬性 (1=男, 0=女)"},
    {"name": "Scholarship holder", "kind": "cat", "default": 0, "key": True,
     "help": "是否有獎學金 (0/1)"},
    {"name": "Age at enrollment", "kind": "num", "default": 20, "key": True, "help": "入學年齡"},
    {"name": "International", "kind": "cat", "default": 0, "key": False},
    {"name": "Curricular units 1st sem (credited)", "kind": "num", "default": 0, "key": False},
    {"name": "Curricular units 1st sem (enrolled)", "kind": "num", "default": 6, "key": False},
    {"name": "Curricular units 1st sem (evaluations)", "kind": "num", "default": 8, "key": False},
    {"name": "Curricular units 1st sem (approved)", "kind": "num", "default": 5, "key": True,
     "help": "第一學期通過科目數"},
    {"name": "Curricular units 1st sem (grade)", "kind": "num", "default": 12.0, "key": False},
    {"name": "Curricular units 1st sem (without evaluations)", "kind": "num", "default": 0, "key": False},
    {"name": "Curricular units 2nd sem (credited)", "kind": "num", "default": 0, "key": False},
    {"name": "Curricular units 2nd sem (enrolled)", "kind": "num", "default": 6, "key": False},
    {"name": "Curricular units 2nd sem (evaluations)", "kind": "num", "default": 8, "key": False},
    {"name": "Curricular units 2nd sem (approved)", "kind": "num", "default": 5, "key": True,
     "help": "第二學期通過科目數"},
    {"name": "Curricular units 2nd sem (grade)", "kind": "num", "default": 12.0, "key": False},
    {"name": "Curricular units 2nd sem (without evaluations)", "kind": "num", "default": 0, "key": False},
    {"name": "Unemployment rate", "kind": "num", "default": 11.0, "key": False},
    {"name": "Inflation rate", "kind": "num", "default": 1.0, "key": False},
    {"name": "GDP", "kind": "num", "default": 0.0, "key": False},
]

FEATURE_ORDER = [f["name"] for f in FEATURE_SCHEMA]

# 敏感屬性 (與 Fairness / Security 章節對應)
SENSITIVE_ATTRIBUTES = ["Gender", "Age at enrollment", "Nacionality", "Scholarship holder"]

MODEL_VERSION = "0.1.0-placeholder"
