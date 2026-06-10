"""訓練一個「佔位用」MLP 並輸出 models/model.pt + models/preprocessor.joblib。

目的不是追求準確率，而是讓組員 C 能先把 Inference 介面 / SHAP / Security 串起來。
組員 B 完成真正的模型後，只要輸出相同格式的 model.pt + preprocessor.joblib 即可替換。

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
from src.preprocessing import build_preprocessor, save_preprocessor
from src.schema import FEATURE_ORDER, FEATURE_SCHEMA, LABELS, MODEL_VERSION


def load_uci() -> tuple[pd.DataFrame, np.ndarray]:
    from ucimlrepo import fetch_ucirepo  # 可能未安裝

    ds = fetch_ucirepo(id=697)
    X = ds.data.features
    X = X.rename(columns={c: c.strip() for c in X.columns})
    # 對齊 schema 欄位順序 (缺的補預設值)
    for feat in FEATURE_SCHEMA:
        if feat["name"] not in X.columns:
            X[feat["name"]] = feat["default"]
    X = X[FEATURE_ORDER]
    y_raw = ds.data.targets.iloc[:, 0].astype(str).str.strip()
    y = y_raw.map({label: i for i, label in enumerate(LABELS)}).values
    return X, y


def make_synthetic(n: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = {}
    for feat in FEATURE_SCHEMA:
        base = float(feat["default"]) or 1.0
        data[feat["name"]] = rng.normal(base, abs(base) * 0.3 + 1.0, n)
    X = pd.DataFrame(data)[FEATURE_ORDER]
    # 用「通過科目數 + 學費繳清」造一個有意義的訊號，讓 SHAP demo 有東西看
    score = (
        0.5 * X["Curricular units 2nd sem (approved)"]
        + 0.3 * X["Curricular units 1st sem (approved)"]
        + 0.2 * X["Tuition fees up to date"]
        - 0.1 * X["Debtor"]
    )
    q1, q2 = np.quantile(score, [0.33, 0.66])
    y = np.where(score < q1, 0, np.where(score < q2, 1, 2))  # 0=Dropout,1=Enrolled,2=Graduate
    return X, y


def train(X: pd.DataFrame, y: np.ndarray, epochs: int = 40) -> DropoutMLP:
    scaler = build_preprocessor(X)
    Xs = scaler.transform(X[FEATURE_ORDER].values).astype(np.float32)
    Xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y.astype(np.int64))

    model = DropoutMLP(input_dim=Xs.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

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
            print(f"使用 UCI 真實資料: {len(X)} 筆。")
        except Exception as exc:  # noqa: BLE001
            print(f"無法取得 UCI 資料 ({exc})，改用合成資料。")
            X, y = make_synthetic()

    model = train(X, y)
    save_checkpoint(
        "models/model.pt", model,
        input_dim=len(FEATURE_ORDER), hidden_dims=(64, 32), num_classes=len(LABELS),
        dropout=0.3, model_version=MODEL_VERSION, label_names=LABELS,
    )
    print("已輸出 models/model.pt 與 models/preprocessor.joblib")


if __name__ == "__main__":
    main()
