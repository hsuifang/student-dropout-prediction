"""訓練一個「佔位用」二分類 MLP 並輸出 models/model.pt + models/preprocessor.joblib。

目的不是追求準確率，而是讓組員 C 能先把 Inference 介面 / SHAP / Security 串起來。
組員 B 完成真正的模型後，只要輸出相同格式的 model.pt + preprocessor.joblib 即可替換
(架構與 src/model.py 一致、scaler 在 14 維 FEATURE_ORDER 上 fit)。

任務: 二分類 (Graduate=0 / Dropout=1)，輸入 11 個原始特徵 → 推導 3 個自創特徵 → 14 維。

資料來源:
    1. 優先嘗試 UCI 真實資料 (需 `pip install ucimlrepo`)。
    2. 取得失敗則產生合成資料，仍可訓練出可運作的 placeholder。

用法:
    python -m scripts.train_placeholder            # 自動選資料來源
    python -m scripts.train_placeholder --synthetic  # 強制使用合成資料
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model import DropoutMLP, save_checkpoint
from src.preprocessing import add_engineered_features, build_preprocessor, save_preprocessor
from src.schema import FEATURE_ORDER, LABELS, MODEL_VERSION, RAW_FEATURES

_DROPOUT_1, _DROPOUT_2 = 0.4, 0.3


def load_uci() -> tuple[pd.DataFrame, np.ndarray]:
    from ucimlrepo import fetch_ucirepo  # 可能未安裝

    ds = fetch_ucirepo(id=697)
    X = ds.data.features
    X = X.rename(columns={c: c.strip() for c in X.columns})
    y_raw = ds.data.targets.iloc[:, 0].astype(str).str.strip()

    # 二分類: 移除 Enrolled，只保留 Dropout / Graduate
    mask = y_raw != "Enrolled"
    X, y_raw = X[mask], y_raw[mask]

    for name in RAW_FEATURES:  # 缺欄位則無法對齊，直接報錯
        if name not in X.columns:
            raise KeyError(f"UCI 資料缺少欄位: {name}")
    X = X[RAW_FEATURES].reset_index(drop=True)
    y = (y_raw == "Dropout").astype(int).values
    return X, y


def make_synthetic(n: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = {
        "Curricular units 1st sem (approved)": rng.integers(0, 8, n),
        "Curricular units 2nd sem (approved)": rng.integers(0, 8, n),
        "Curricular units 1st sem (enrolled)": rng.integers(4, 9, n),
        "Curricular units 2nd sem (enrolled)": rng.integers(4, 9, n),
        "Curricular units 1st sem (grade)": rng.uniform(8, 16, n),
        "Curricular units 2nd sem (grade)": rng.uniform(8, 16, n),
        "Scholarship holder": rng.integers(0, 2, n),
        "Tuition fees up to date": rng.integers(0, 2, n),
        "Application mode": rng.integers(1, 18, n),
        "Admission grade": rng.uniform(95, 190, n),
        "Previous qualification (grade)": rng.uniform(95, 190, n),
    }
    X = pd.DataFrame(data)[RAW_FEATURES]
    # 用「通過科目數 + 學費繳清」造一個有意義的訊號，讓 SHAP demo 有東西看
    score = (
        0.5 * X["Curricular units 2nd sem (approved)"]
        + 0.3 * X["Curricular units 1st sem (approved)"]
        + 1.0 * X["Tuition fees up to date"]
        + rng.normal(0, 1.0, n)
    )
    y = (score < np.median(score)).astype(int).to_numpy()  # 分數低 → Dropout(1)
    return X, y


def train(X: pd.DataFrame, y: np.ndarray, epochs: int = 60) -> DropoutMLP:
    scaler = build_preprocessor(X)  # 內部會推導 14 維後 fit
    Xs = scaler.transform(add_engineered_features(X)[FEATURE_ORDER].values).astype(np.float32)
    Xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    model = DropoutMLP(input_dim=len(FEATURE_ORDER), dropout_1=_DROPOUT_1, dropout_2=_DROPOUT_2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}  loss={loss.item():.4f}")

    save_preprocessor(scaler)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="強制使用合成資料")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)

    if args.synthetic:
        X, y = make_synthetic()
        print("使用合成資料訓練 placeholder。")
    else:
        try:
            X, y = load_uci()
            print(f"使用 UCI 真實資料: {len(X)} 筆 (已移除 Enrolled)。")
        except Exception as exc:  # noqa: BLE001
            print(f"無法取得 UCI 資料 ({exc})，改用合成資料。")
            X, y = make_synthetic()

    model = train(X, y)
    save_checkpoint(
        "models/model.pt", model,
        input_dim=len(FEATURE_ORDER), dropout_1=_DROPOUT_1, dropout_2=_DROPOUT_2,
        model_version=MODEL_VERSION, label_names=LABELS,
    )
    print("已輸出 models/model.pt 與 models/preprocessor.joblib")


if __name__ == "__main__":
    main()
