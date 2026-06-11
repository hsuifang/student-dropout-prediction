"""產生 Global Explanation 重要度長條圖 -> results/global_importance.png。

用法:
    python -m scripts.plot_global_importance
    python -m scripts.plot_global_importance --data notebooks/test_scaled.csv --top 14

資料來源 (依序嘗試): --data 指定路徑 -> notebooks/test_scaled.csv -> test_scaled.csv。
找不到時退回以 placeholder scaler 產生的合成樣本，仍可輸出一張 demo 圖。

注意: 圖反映「目前 models/model.pt」的行為。若要呈現正式結果，請先以 notebook Cell 4
輸出真實模型，再執行本腳本。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 無視窗環境也能存檔
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explain import global_importance
from src.schema import FEATURE_ORDER, FEATURE_SCHEMA


def _load_scaled(data_arg: str | None) -> tuple[np.ndarray, str]:
    candidates = [data_arg] if data_arg else []
    candidates += ["notebooks/test_scaled.csv", "test_scaled.csv"]
    for path in candidates:
        if path and Path(path).exists():
            df = pd.read_csv(path)
            cols = [c for c in df.columns if c != "Target_Label"]
            if cols != FEATURE_ORDER:
                raise ValueError(f"{path} 欄位與 FEATURE_ORDER 不一致")
            return df[FEATURE_ORDER].values.astype(np.float32), f"{path} (N={len(df)})"

    # 退回合成樣本
    from src.preprocessing import add_engineered_features, load_preprocessor
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        r = {f["name"]: float(f["default"]) for f in FEATURE_SCHEMA}
        r["Curricular units 1st sem (approved)"] = rng.integers(0, 7)
        r["Curricular units 2nd sem (approved)"] = rng.integers(0, 7)
        r["Tuition fees up to date"] = rng.integers(0, 2)
        r["Admission grade"] = rng.uniform(95, 190)
        rows.append(r)
    scaler = load_preprocessor()
    X = scaler.transform(add_engineered_features(pd.DataFrame(rows))[FEATURE_ORDER].values)
    return X.astype(np.float32), "synthetic demo (N=200)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="已標準化的 14 維 CSV 路徑")
    parser.add_argument("--top", type=int, default=14, help="顯示前 k 個特徵")
    parser.add_argument("--out", default="results/global_importance.png")
    args = parser.parse_args()

    X, source = _load_scaled(args.data)
    result = global_importance(X, top_k=args.top)
    names = [n for n, _ in result["importances"]][::-1]      # 反轉讓最大值在最上
    vals = [v for _, v in result["importances"]][::-1]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(names, vals, color="#d9534f")
    plt.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.xlabel("Global importance  (mean |grad x input|)")
    plt.title(f"Global Feature Importance for Dropout Risk\nsource: {source}")
    plt.margins(x=0.12)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"✅ 已輸出 {out}")
    print(f"   資料來源: {source} | 模型: models/model.pt")
    print("   前 5 重要特徵:")
    for name, imp in result["importances"][:5]:
        print(f"     {name:38s} {imp:.4f}")


if __name__ == "__main__":
    main()
