"""Inference 介面 (對應需求書第 10 節)。

定位: 教育早期預警與決策支援系統 (不是 ML demo dashboard)。
非技術背景的教師/導師/行政人員應能在 5 秒內看懂「風險高低 / 主要原因 / 建議行動」。

流程: 輸入 -> 檢查格式 -> 套用 Scaler -> 推論 -> 風險與建議 -> Risk Factors -> 使用限制
執行: streamlit run app/streamlit_app.py

版面: Step 01 Intake (Academic / Financial / Advanced) → Step 02 Result。
Result 以決策為先: Risk badge → Recommended action → Risk factors → Probabilities。
英文為主、中文為輔。桌機雙欄、手機單欄 + sticky 行動按鈕。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# 讓 app 能 import 專案根目錄的 src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import explain_record
from src.inference import predict
from src.interpret import RISK_BADGE, describe_factor, recommended_actions
from src.schema import FEATURE_SCHEMA, LABELS, MODEL_VERSION, SENSITIVE_ATTRIBUTES
from src.security import HUMAN_REVIEW_WARNING, bounds_for, log_inference, validate_input

DROPOUT_LABEL = LABELS[-1]  # 正類 = "Dropout"

# 輸入分組 (其餘欄位自動收進 Advanced)
ACADEMIC_FIELDS = [
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
]
FINANCIAL_FIELDS = ["Scholarship holder", "Tuition fees up to date"]
_BY_NAME = {f["name"]: f for f in FEATURE_SCHEMA}

st.set_page_config(page_title="Dropout Risk Prediction", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- #
# 視覺主題 (CSS)
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    :root{
      --ink:#15233B; --ink-2:#46566F; --paper:#F2F4F7; --card:#FFFFFF;
      --line:#DDE2E9; --grey:#C6CDD8; --safe:#137A63; --warn:#C9890A; --risk:#C42A3B;
    }

    html { font-size:17px; }   /* 整體放大一點點 (rem 連同間距等比例) */
    html, body, [class*="css"], .stMarkdown, .stApp { font-family:'Inter', sans-serif; letter-spacing:.012em; }
    .stApp { background:var(--paper); }
    .block-container { max-width:1180px; padding-top:1.1rem; padding-bottom:4rem; }
    h1,h2,h3 { font-family:'Archivo', sans-serif; letter-spacing:-0.02em; color:var(--ink); }

    /* ---- 標題列 (精簡, 讓兩欄內容更靠上) ---- */
    .masthead{ background:var(--ink); color:#EAF0F8; border-radius:14px;
      padding:1rem 1.5rem; margin-bottom:1.1rem; display:flex; align-items:baseline;
      flex-wrap:wrap; gap:.2rem 1rem; }
    .masthead .eyebrow{ flex:1 1 100%; font-family:'JetBrains Mono', monospace; font-size:.62rem; font-weight:700;
      letter-spacing:.22em; color:#7E93B5; text-transform:uppercase; margin-bottom:.2rem; }
    .masthead h1{ color:#F4F8FF; font-size:1.5rem; font-weight:800; margin:0; line-height:1.05; }
    .masthead .zh-kicker{ color:#9DB0CE; font-size:.85rem; font-weight:500; }
    .masthead .desc{ flex:1 1 100%; color:#7E93B5; font-size:.74rem; margin-top:.25rem; }

    /* 兩欄: 右側結果在桌機 sticky, 點擊預測後免向下捲動 */
    @media (min-width:769px){
      .st-key-resultcol{ position:sticky; top:.7rem; align-self:flex-start; }
    }
    .st-key-resultcol .sec{ margin-top:0; }

    /* 結果尚未產生時的 empty state */
    .empty{ border:1.5px dashed var(--line); border-radius:16px; background:#FBFCFE;
      padding:2.6rem 1.6rem; text-align:center; min-height:340px;
      display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.5rem; }
    .empty .ic{ font-size:2rem; opacity:.55; }
    .empty .t{ font-family:'Archivo', sans-serif; font-weight:700; font-size:1.05rem; color:var(--ink); }
    .empty .s{ font-size:.84rem; color:var(--ink-2); max-width:30ch; line-height:1.55; }

    /* ---- 區段標題 ---- */
    .sec{ display:flex; align-items:center; gap:.7rem; margin:1.6rem 0 .7rem 0; }
    .sec .step-tag{ font-family:'JetBrains Mono', monospace; font-size:.62rem; font-weight:700;
      letter-spacing:.12em; text-transform:uppercase; color:#FFF; background:var(--ink);
      padding:.22rem .55rem; border-radius:5px; }
    .sec .sec-en{ font-family:'Archivo', sans-serif; font-size:1.05rem; font-weight:700; color:var(--ink); }
    .sec .sec-zh{ font-size:.78rem; color:var(--ink-2); font-weight:500; }
    .sec::after{ content:""; flex:1; height:1px; background:var(--line); }

    /* ---- Intake console (藍圖點陣) ---- */
    .st-key-console{
      background-color:#FBFCFE;
      background-image:radial-gradient(rgba(21,35,59,.07) 1px, transparent 1px);
      background-size:20px 20px; background-position:-1px -1px;
      border:1px solid var(--line) !important; border-radius:18px !important;
      padding:1.5rem 1.6rem 1.4rem !important; box-shadow:0 6px 22px rgba(21,35,59,.06);
    }
    .console-head{ margin-bottom:1rem; }
    .console-head .step{ display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.62rem;
      font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#FFF; background:var(--ink);
      padding:.24rem .6rem; border-radius:6px; margin-bottom:.55rem; }
    .console-head .ctitle{ font-family:'Archivo', sans-serif; font-size:1.25rem; font-weight:800; color:var(--ink); }
    .console-head .ctitle span{ font-size:.88rem; font-weight:500; color:var(--ink-2); margin-left:.5rem; }
    .console-head .chint{ font-size:.8rem; color:var(--ink-2); margin-top:.25rem; }

    .group-label{ font-family:'JetBrains Mono', monospace; font-size:.66rem; font-weight:700;
      letter-spacing:.14em; text-transform:uppercase; color:var(--ink-2);
      margin:1.1rem 0 .1rem 0; padding-bottom:.35rem; border-bottom:1px solid var(--line); }
    .st-key-console .stNumberInput label, .st-key-console [data-testid="stWidgetLabel"] label{
      font-size:.82rem !important; font-weight:600 !important; color:var(--ink) !important; }
    .st-key-console .stNumberInput input{ font-family:'JetBrains Mono', monospace !important; }
    .st-key-console [data-baseweb="input"]{ border-radius:9px !important; background:#FFF !important; }

    /* ---- Run 按鈕 (手機 sticky 置底) ---- */
    .st-key-runbar{ margin-top:.6rem; }
    .stButton > button[kind="primary"]{ background:var(--ink); border:none; border-radius:11px;
      font-family:'Archivo', sans-serif; font-weight:700; letter-spacing:.02em; padding:.7rem 1.2rem;
      transition:transform .06s ease, background .15s ease; }
    .stButton > button[kind="primary"]:hover{ background:#0E1A2E; transform:translateY(-1px); }
    .stButton > button[kind="primary"]:focus-visible{ outline:3px solid #9DB0CE; outline-offset:2px; }

    /* ---- Step 02: 結果 ---- */
    .hero, .factors, .bars, .notice{ animation:rise .4s ease both; }
    @keyframes rise{ from{ opacity:0; transform:translateY(8px);} to{ opacity:1; transform:none;} }

    /* Hero result card */
    .hero{ background:var(--card); border:1px solid var(--line); border-radius:16px;
      padding:1.5rem 1.7rem; margin:.2rem 0 1rem; box-shadow:0 6px 22px rgba(21,35,59,.06); }
    .hero-top{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
    .badge{ display:inline-flex; align-items:center; gap:.5rem; font-family:'Archivo', sans-serif;
      font-weight:800; font-size:1.05rem; padding:.45rem .9rem; border-radius:10px; }
    .badge.risk{ background:#FBEAEC; color:var(--risk); }
    .badge.warn{ background:#FBF1DC; color:var(--warn); }
    .badge.safe{ background:#E2F1ED; color:var(--safe); }
    .hero-prob{ text-align:right; }
    .hero-prob .num{ font-family:'JetBrains Mono', monospace; font-size:2.6rem; font-weight:700; line-height:1; }
    .hero-prob .num span{ font-size:1.1rem; font-weight:500; }
    .hero-prob .lab{ font-family:'JetBrains Mono', monospace; font-size:.64rem; letter-spacing:.12em;
      text-transform:uppercase; color:var(--ink-2); margin-top:.3rem; }

    .track{ position:relative; height:12px; border-radius:7px; margin:1.2rem 0 .5rem;
      background:linear-gradient(90deg, var(--safe) 0%, var(--warn) 52%, var(--risk) 100%); }
    .track .mk{ position:absolute; top:-6px; width:4px; height:24px; border-radius:3px;
      background:var(--ink); box-shadow:0 0 0 3px var(--card); transform:translateX(-50%); }
    .track-scale{ display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace;
      font-size:.7rem; font-weight:700; }
    .track-scale .g{ color:var(--safe);} .track-scale .d{ color:var(--risk);}

    .actions{ margin-top:1.3rem; border-top:1px dashed var(--line); padding-top:1rem; }
    .actions .alab{ font-family:'JetBrains Mono', monospace; font-size:.66rem; font-weight:700;
      letter-spacing:.14em; text-transform:uppercase; color:var(--ink-2); margin-bottom:.55rem; }
    .actions ul{ margin:0; padding:0; list-style:none; }
    .actions li{ display:flex; align-items:center; gap:.6rem; font-size:.92rem; color:var(--ink);
      padding:.32rem 0; }
    .actions li::before{ content:""; width:7px; height:7px; border-radius:50%; background:var(--ink); flex:0 0 auto; }

    /* ---- Use Limitation (移到結果上方, 提高可見度) ---- */
    .notice{ background:#FBF1DC; border:1px solid #EAD7A6; border-left:4px solid var(--warn);
      border-radius:12px; padding:1rem 1.2rem; margin:.2rem 0 1.2rem; }
    .notice-tag{ font-family:'Archivo', sans-serif; font-size:.92rem; font-weight:800; color:#9A6A05;
      margin-bottom:.4rem; display:flex; align-items:center; gap:.45rem; }
    .notice p{ font-size:.84rem; color:#6E5418; margin:0; line-height:1.6; }

    /* ---- Risk Factors (卡片列表; 多為灰色, 重點才上色) ---- */
    .factors{ display:flex; flex-direction:column; gap:.55rem; }
    .factor{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; }
    .factor-top{ display:flex; align-items:center; gap:.7rem; }
    .factor .rank{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:.8rem;
      color:var(--ink-2); width:1.4rem; flex:0 0 auto; }
    .factor .fname{ flex:1; font-weight:600; font-size:.92rem; color:var(--ink); }
    .factor .ftag{ font-size:.72rem; font-weight:700; padding:.2rem .55rem; border-radius:20px; white-space:nowrap; }
    .ftag.danger{ background:#FBEAEC; color:var(--risk); }
    .ftag.good{ background:#E2F1ED; color:var(--safe); }
    .ftag.neutral{ background:#EEF1F5; color:var(--ink-2); }
    .factor .fbar{ height:7px; background:#EDF0F4; border-radius:5px; overflow:hidden; margin-top:.6rem; }
    .factor .fbar > div{ height:100%; border-radius:5px; }
    .fill-danger{ background:var(--risk);} .fill-good{ background:var(--safe);} .fill-neutral{ background:var(--grey);}
    .factor .fsub{ font-size:.74rem; color:var(--ink-2); margin-top:.35rem; }

    /* ---- 機率條 ---- */
    .bars{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.3rem; }
    .bar-row{ display:flex; align-items:center; gap:.85rem; padding:.42rem 0; }
    .bar-name{ flex:0 0 30%; font-size:.88rem; font-weight:600; color:var(--ink); }
    .bar-track{ flex:1; height:11px; background:#EDF0F4; border-radius:6px; overflow:hidden; }
    .bar-fill{ height:100%; border-radius:6px; }
    .bar-val{ flex:0 0 3.6rem; text-align:right; font-family:'JetBrains Mono', monospace;
      font-size:.9rem; font-weight:700; }

    .meta{ font-family:'JetBrains Mono', monospace; font-size:.7rem; color:#8A98AC; letter-spacing:.04em; }

    /* ============ RESPONSIVE ============ */
    @media (max-width:768px){
      .block-container{ padding-left:.7rem; padding-right:.7rem; }
      /* 表單一律單欄 */
      [data-testid="stHorizontalBlock"]{ flex-direction:column !important; gap:0 !important; }
      [data-testid="stHorizontalBlock"] > div{ width:100% !important; flex:1 1 100% !important; }
      /* Hero 置中、結果優先 */
      .hero-top{ flex-direction:column; align-items:center; text-align:center; }
      .hero-prob{ text-align:center; }
      .hero-prob .num{ font-size:3.1rem; }
      /* Run 按鈕 sticky 置底 */
      .st-key-runbar{ position:sticky; bottom:0; z-index:60; padding:.6rem 0 .4rem;
        background:linear-gradient(180deg, rgba(242,244,247,0) 0%, var(--paper) 38%); }
    }
    @media (prefers-reduced-motion: reduce){ *{ animation:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="masthead">
      <div class="eyebrow">Early-Intervention Decision Support · v{MODEL_VERSION}</div>
      <h1>Dropout Risk Prediction</h1>
      <div class="zh-kicker">學生退學與學業成果預測</div>
      <div class="desc">Early-warning support for teachers and advisors — not an automated decision.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


MODEL_PATH = "models/model.pt"
if not Path(MODEL_PATH).exists():
    st.error("Model not found at models/model.pt — run `python -m scripts.train_placeholder` first.")
    st.stop()


# --------------------------------------------------------------------------- #
# 共用元件
# --------------------------------------------------------------------------- #
def section_label(en: str, zh: str, step: str | None = None) -> None:
    step_html = f'<span class="step-tag">{step}</span>' if step else ""
    st.markdown(
        f'<div class="sec">{step_html}<span class="sec-en">{en}</span>'
        f'<span class="sec-zh">{zh}</span></div>',
        unsafe_allow_html=True,
    )


def render_field(feat: dict):
    """依欄位型別渲染輸入；bounds 取自 security.py 的單一真相來源 (防呆)。"""
    lo, hi = bounds_for(feat)
    name = feat["name"]
    help_text = feat.get("help")
    label = ("🔒 " if name in SENSITIVE_ATTRIBUTES else "") + name

    if feat["kind"] == "cat" and (hi - lo) == 1:           # 0/1 旗標 → 開關
        return int(st.toggle(label, value=bool(feat["default"]), help=help_text))
    if feat["kind"] == "cat":                              # 類別代碼 → 整數
        return st.number_input(label, min_value=int(lo), max_value=int(hi),
                               value=int(feat["default"]), step=1, help=help_text)
    if "grade" in name.lower():                            # 成績 → 小數步進
        return st.number_input(label, min_value=float(lo), max_value=float(hi),
                               value=float(feat["default"]), step=0.5, help=help_text)
    return st.number_input(label, min_value=int(lo), max_value=int(hi),   # 科目數 → 整數
                           value=int(feat["default"]), step=1, help=help_text)


def render_group(names: list[str], record: dict, cols_per_row: int = 2) -> None:
    feats = [_BY_NAME[n] for n in names if n in _BY_NAME]
    for i in range(0, len(feats), cols_per_row):
        row = feats[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, feat in zip(cols, row):
            with col:
                record[feat["name"]] = render_field(feat)


def render_hero(result: dict, record: dict) -> None:
    p = max(0.0, min(1.0, result["dropout_probability"]))
    pct = p * 100
    icon, label, tok = RISK_BADGE[result["risk_level"]]
    actions = "".join(f"<li>{a}</li>" for a in recommended_actions(result, record))
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-top">
            <div class="badge {tok}">{icon} {label}</div>
            <div class="hero-prob">
              <div class="num" style="color:var(--{tok})">{pct:.0f}<span>%</span></div>
              <div class="lab">Dropout Probability · {result['outcome']}</div>
            </div>
          </div>
          <div class="track"><div class="mk" style="left:{pct:.1f}%"></div></div>
          <div class="track-scale"><span class="g">◀ Graduate</span><span class="d">Dropout ▶</span></div>
          <div class="actions">
            <div class="alab">Recommended Action · 建議行動</div>
            <ul>{actions}</ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_use_limitation() -> None:
    st.markdown(
        f"""
        <div class="notice">
          <div class="notice-tag">⚠ Important · 使用限制</div>
          <p>{HUMAN_REVIEW_WARNING.replace(chr(10), ' ')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factors(top_features: list, record: dict, method: str = "") -> None:
    attr = "SHAP" if "SHAP" in method else "Contribution"   # 梯度後援時別誤標成 SHAP
    scale = max((abs(c) for _, c in top_features), default=1.0) or 1.0
    cards = ""
    for i, (name, contrib) in enumerate(top_features, start=1):
        f = describe_factor(name, contrib, record, scale)
        cards += (
            f'<div class="factor"><div class="factor-top">'
            f'<span class="rank">{i:02d}</span>'
            f'<span class="fname">{f["label"]}</span>'
            f'<span class="ftag {f["role"]}">{f["impact"]} Impact</span></div>'
            f'<div class="fbar"><div class="fill-{f["role"]}" style="width:{max(f["ratio"] * 100, 6):.0f}%"></div></div>'
            f'<div class="fsub">{f["direction"]} · {attr} {f["shap"]:+.3f}</div></div>'
        )
    st.markdown(f'<div class="factors">{cards}</div>', unsafe_allow_html=True)


def render_prob_bars(probs: dict) -> None:
    rows = ""
    for label, prob in sorted(probs.items(), key=lambda kv: kv[0] != DROPOUT_LABEL):
        color = "var(--risk)" if label == DROPOUT_LABEL else "var(--safe)"
        rows += (
            f'<div class="bar-row"><div class="bar-name">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{prob * 100:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-val" style="color:{color}">{prob * 100:.1f}%</div></div>'
        )
    st.markdown(f'<div class="bars">{rows}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Step 01 · Student Information
# --------------------------------------------------------------------------- #
def render_results(result: dict, record: dict) -> None:
    """Step 02 結果區 (決策為先: 風險 → 建議 → 原因 → 機率)。"""
    section_label("Result", "預測結果", step="Step 02")
    render_hero(result, record)
    render_use_limitation()

    section_label("Top Risk Factors", "主要風險因子")
    exp = explain_record(record, target_label=result["outcome"])
    render_factors(exp["top_features"], record, exp["method"])

    section_label("Class Probabilities", "各類別機率")
    render_prob_bars(result["probabilities"])

    with st.expander("Technical information · 技術資訊"):
        st.caption(f"Explainability method: {exp['method']} — shows model reasoning, not causation.")
        st.caption(
            f"Model {result['model_version']} · "
            f"assessed {result.get('assessed_at', '—')} · "
            f"de-identified record logged → results/inference_log.jsonl"
        )
        st.markdown(
            '<div class="meta">Feature contributions: positive raises dropout risk, negative lowers it.</div>',
            unsafe_allow_html=True,
        )


record: dict = {}
grouped = set(ACADEMIC_FIELDS + FINANCIAL_FIELDS)
advanced_names = [f["name"] for f in FEATURE_SCHEMA if f["name"] not in grouped]

# 左: 輸入 / 右: 結果 —— 桌機並排, 手機自動堆疊 (見 responsive CSS)
left_col, right_col = st.columns([1, 1.05], gap="large")

with left_col:
    with st.container(key="console", border=True):
        st.markdown(
            """
            <div class="console-head">
              <div class="step">Step 01 · Student Information</div>
              <div class="ctitle">Student record<span>學生資料</span></div>
              <div class="chint">Fill in what you know, then run the assessment. Defaults are reasonable.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="group-label">Academic Information · 學業</div>', unsafe_allow_html=True)
        render_group(ACADEMIC_FIELDS, record, cols_per_row=1)

        st.markdown('<div class="group-label">Financial Information · 財務</div>', unsafe_allow_html=True)
        render_group(FINANCIAL_FIELDS, record, cols_per_row=1)

        with st.expander(f"Advanced fields ({len(advanced_names)} items, defaults are fine)"):
            render_group(advanced_names, record, cols_per_row=1)

    with st.container(key="runbar"):
        run = st.button("Run Assessment · 執行評估", type="primary", use_container_width=True)

# 只在「按下 Run 當下」驗證 / 推論 / 寫 log（log 僅記一次），結果存入 session_state，
# 之後即使使用者調整輸入觸發 rerun，結果仍保留直到下一次 Run。
if run:
    errors = validate_input(record)  # Security: Input Validation
    if errors:
        st.session_state["assessment"] = {"errors": errors}
    else:
        pred = predict(record, model_path=MODEL_PATH)
        pred["assessed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_inference(record, {
            "outcome": pred["outcome"],
            "risk_level": pred["risk_level"],
            "probabilities": pred["probabilities"],
        }, model_version=pred["model_version"])
        st.session_state["assessment"] = {"prediction": pred, "record": dict(record)}

with right_col:
    with st.container(key="resultcol"):
        state = st.session_state.get("assessment")
        if not state:
            st.markdown(
                """
                <div class="empty">
                  <div class="ic">🎓</div>
                  <div class="t">No assessment yet · 尚未評估</div>
                  <div class="s">Fill in the student record on the left and run the assessment.
                  The risk level, recommended actions, and main factors appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif state.get("errors"):
            st.error("Please fix these fields before running the assessment · 請修正以下欄位：")
            for e in state["errors"]:
                st.write(f"- {e}")
        else:
            render_results(state["prediction"], state["record"])
