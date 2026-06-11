"""把模型輸出翻譯成「教師看得懂」的決策支援語言 (對應 UI 改版需求)。

這層不做任何推論，只負責解讀:
    - 風險等級 → 標籤 / 圖示 / 顏色
    - SHAP 貢獻 → 自然語言的 Risk Factor (名稱 / 強度 / 方向)
    - 風險 + 學生狀況 → 建議後續行動 (Recommended Action)

讓非技術背景的使用者能在 5 秒內理解「風險高低 / 主要原因 / 該做什麼」。
"""
from __future__ import annotations

_EPS = 1e-8

# 技術欄位名 → 教師可讀名稱
FRIENDLY_NAMES = {
    "Curricular units 1st sem (approved)": "1st Semester Approved Units",
    "Curricular units 2nd sem (approved)": "2nd Semester Approved Units",
    "Curricular units 1st sem (enrolled)": "1st Semester Enrolled Units",
    "Curricular units 2nd sem (enrolled)": "2nd Semester Enrolled Units",
    "Curricular units 1st sem (grade)": "1st Semester Grade",
    "Curricular units 2nd sem (grade)": "2nd Semester Grade",
    "Scholarship holder": "Scholarship",
    "Tuition fees up to date": "Tuition Payment",
    "Application mode": "Application Route",
    "Admission grade": "Admission Grade",
    "Previous qualification (grade)": "Previous Qualification Grade",
    "1st_sem_pass_rate": "1st Semester Pass Rate",
    "2nd_sem_pass_rate": "2nd Semester Pass Rate",
    "grade_change": "Grade Trend",
}

# 風險等級 → (圖示, 標籤, 顏色 token)
RISK_BADGE = {
    "High": ("⚠", "High Risk", "risk"),
    "Medium": ("◆", "Medium Risk", "warn"),
    "Low": ("✓", "Low Risk", "safe"),
}


def _feature_value(name: str, record: dict) -> float | None:
    """取得欄位的實際值；自創特徵 (pass rate / grade trend) 即時推導。"""
    try:
        if name == "1st_sem_pass_rate":
            return float(record["Curricular units 1st sem (approved)"]) / (
                float(record["Curricular units 1st sem (enrolled)"]) + _EPS)
        if name == "2nd_sem_pass_rate":
            return float(record["Curricular units 2nd sem (approved)"]) / (
                float(record["Curricular units 2nd sem (enrolled)"]) + _EPS)
        if name == "grade_change":
            return float(record["Curricular units 2nd sem (grade)"]) - \
                float(record["Curricular units 1st sem (grade)"])
        return float(record[name])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _readable_label(name: str, value: float | None) -> str:
    """把欄位 + 實際值組成自然語言，例如「Low 2nd Semester Pass Rate」。"""
    base = FRIENDLY_NAMES.get(name, name)

    # 狀態型欄位: 直接描述狀態
    if name == "Tuition fees up to date":
        return "Tuition Up to Date" if (value or 0) >= 1 else "Tuition Overdue"
    if name == "Scholarship holder":
        return "Has Scholarship" if (value or 0) >= 1 else "No Scholarship"
    if name == "Application mode":
        return base

    if value is None:
        return base

    if name in ("1st_sem_pass_rate", "2nd_sem_pass_rate"):
        q = "Low" if value < 0.6 else "Moderate" if value < 0.85 else "High"
    elif "(approved)" in name:
        q = "Few" if value < 4 else "Some" if value < 7 else "Many"
    elif "(grade)" in name and "sem" in name:          # 學期成績 0-20
        q = "Low" if value < 10 else "Moderate" if value < 14 else "High"
    elif name in ("Admission grade", "Previous qualification (grade)"):  # 0-200
        q = "Low" if value < 120 else "Moderate" if value < 150 else "High"
    elif name == "grade_change":
        return ("Declining" if value <= -1 else "Improving" if value >= 1 else "Stable") + " Grade Trend"
    else:
        return base

    return f"{q} {base}"


def describe_factor(name: str, contrib: float, record: dict, scale: float) -> dict:
    """單一 Risk Factor 的可讀描述。

    contrib > 0 → 推高退學風險; < 0 → 降低。scale 為這批因子中最大的 |contrib|。
    顏色只給「強烈推高 (紅)」與「強烈降低 (青綠)」，其餘維持灰色以降低紅色用量。
    """
    value = _feature_value(name, record)
    ratio = abs(contrib) / scale if scale else 0.0
    impact = "Strong" if ratio >= 0.66 else "Moderate" if ratio >= 0.33 else "Minor"
    raises = contrib > 0
    if impact == "Strong":
        role = "danger" if raises else "good"
    else:
        role = "neutral"
    return {
        "label": _readable_label(name, value),
        "impact": impact,
        "direction": "Raises risk" if raises else "Lowers risk",
        "role": role,
        "ratio": ratio,
        "shap": contrib,
    }


def recommended_actions(result: dict, record: dict) -> list[str]:
    """依風險等級 + 學生狀況給出建議後續行動 (給教師/導師的 next step)。"""
    risk = result["risk_level"]
    tuition_ok = _feature_value("Tuition fees up to date", record)
    tuition_ok = (tuition_ok or 0) >= 1

    if risk == "High":
        actions = ["Schedule academic counseling",
                   "Review attendance and engagement",
                   "Assess financial support needs"]
    elif risk == "Medium":
        actions = ["Arrange an early check-in with the student",
                   "Monitor next-semester performance"]
    else:
        actions = ["Continue routine monitoring",
                   "No immediate intervention needed"]

    if not tuition_ok and not any("financial" in a.lower() for a in actions):
        actions.append("Follow up on overdue tuition")
    return actions
