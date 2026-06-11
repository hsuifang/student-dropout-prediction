"""輸出「正式」可部署模型 → models/model.pt + models/preprocessor.joblib。

與 train_placeholder.py 的差別:
    train_placeholder.py : 合成/UCI 資料 + 普通 BCE → 佔位「假模型」(demo 用，不需真資料)。
    export_model.py(本檔): 真實 data.csv + Focal + MinDiff(MMD) + Optuna best params
                           → **5-fold soft-voting ensemble**，完全比照組員 B 的 notebook
                             (Cell 3 階段二)：每折早停、還原最佳權重、軟投票整合。
                             即報告中那一顆模型本身。

序列化:5 顆成員以 save_ensemble_checkpoint 存成單一 models/model.pt；推論端 (src.inference)
       透過 EnsembleMLP 自動還原並做機率平均，介面/推論層皆不需改。

注意: 訓練與調參(Optuna)本身仍以組員 B 的 notebook 為準；本檔只把該設計「重訓並序列化」。

用法:
    python -m scripts.export_model --data notebooks/data.csv
    python -m scripts.export_model --data data.csv --epochs 100 --n-splits 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.model import DropoutMLP, save_ensemble_checkpoint
from src.preprocessing import add_engineered_features, build_preprocessor
from src.schema import FEATURE_ORDER, LABELS, MODEL_VERSION, RAW_FEATURES

# Optuna 在組員 B 的 notebook (Cell 3) 尋獲之黃金超參數；若重跑搜參請以最新值更新。
BEST_PARAMS = {"lr": 0.000221, "beta": 1.332106, "dropout_1": 0.292656, "dropout_2": 0.322018}


class FocalLoss(nn.Module):
    """與 notebook 一致的 Focal Loss(處理 Dropout 類別不平衡)。"""

    def __init__(self, alpha: float = 0.6, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        a_t = torch.where(targets == 1,
                          torch.tensor(self.alpha, device=inputs.device),
                          torch.tensor(1 - self.alpha, device=inputs.device))
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


class MmdLoss(nn.Module):
    """與 notebook 一致的 MMD MinDiff Loss(拉近敏感/對照群體的輸出分布)。"""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def _rbf(self, x, y):
        x, y = x.unsqueeze(1), y.unsqueeze(0)
        return torch.exp(-(x - y).pow(2).sum(2) / (2 * self.sigma ** 2))

    def forward(self, x, y):
        return self._rbf(x, x).mean() + self._rbf(y, y).mean() - 2 * self._rbf(x, y).mean()


class EarlyStopping:
    """與 notebook 一致的早停；val loss 連續 patience 次未改善即停，並還原最佳權重。

    註：notebook 版用 `best_model_state = model.state_dict()`（未深拷貝），後續訓練會
    就地覆寫到該 dict，導致「還原最佳權重」其實還原成停止當下的權重。此處以 clone
    深拷貝修正，確保真正還原到最佳。
    """

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience, self.delta = patience, delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model) -> None:
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.best_model_state = {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_real(data_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """讀真實 data.csv，複製 notebook Cell 0 的切分，回傳 raw 訓練集(11 欄)與標籤。"""
    df = pd.read_csv(data_path)
    df = df[df["Target"] != "Enrolled"].copy()          # 二分類：移除 Enrolled
    X = df[RAW_FEATURES].copy()
    y = (df["Target"] == "Dropout").astype(int).values  # 0=Graduate, 1=Dropout
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train.reset_index(drop=True), y_train


def _train_fold(X: np.ndarray, y: np.ndarray, sens: np.ndarray,
                tr_idx: np.ndarray, va_idx: np.ndarray, *,
                epochs: int, patience: int, device) -> DropoutMLP:
    """訓練單一折模型：MinDiff + 早停 + 還原最佳權重（= notebook 每折流程）。"""
    torch.manual_seed(42)
    model = DropoutMLP(input_dim=len(FEATURE_ORDER),
                       dropout_1=BEST_PARAMS["dropout_1"],
                       dropout_2=BEST_PARAMS["dropout_2"]).to(device)
    opt = optim.Adam(model.parameters(), lr=BEST_PARAMS["lr"], weight_decay=1e-4)
    focal, mmd = FocalLoss(0.6, 2), MmdLoss(1.0)
    stopper = EarlyStopping(patience=patience)

    ds = TensorDataset(
        torch.tensor(X[tr_idx]),
        torch.tensor(y[tr_idx], dtype=torch.float32).unsqueeze(1),
        torch.tensor(sens[tr_idx], dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)  # drop_last 防 BatchNorm 遇 batch=1
    val_x = torch.tensor(X[va_idx]).to(device)
    val_y = torch.tensor(y[va_idx], dtype=torch.float32).unsqueeze(1).to(device)

    for _ in range(epochs):
        model.train()
        for b_x, b_y, b_s in loader:
            b_x, b_y, b_s = b_x.to(device), b_y.to(device), b_s.to(device)
            opt.zero_grad()
            logits = model(b_x)
            loss = focal(logits, b_y)
            probs = torch.sigmoid(logits)
            sm, rm = (b_s == 1).squeeze(), (b_s == 0).squeeze()
            if sm.sum() > 1 and rm.sum() > 1:
                loss = loss + BEST_PARAMS["beta"] * mmd(probs[sm], probs[rm])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_loss = focal(model(val_x), val_y).item()
        stopper(v_loss, model)
        if stopper.early_stop:
            break

    model.load_state_dict(stopper.best_model_state)  # 還原最佳權重
    return model.eval().cpu()


def train_ensemble(X_train_raw: pd.DataFrame, y_train: np.ndarray, *,
                   epochs: int, patience: int, n_splits: int, device):
    """5-fold soft-voting ensemble：scaler fit on 全 train(== Cell 0)，每折早停。回傳 (members, scaler)。"""
    scaler = build_preprocessor(X_train_raw)  # fit on 完整 train(scaler 仍用全 train，與 notebook 一致)
    X = scaler.transform(
        add_engineered_features(X_train_raw)[FEATURE_ORDER].values
    ).astype(np.float32)

    median = X_train_raw["Tuition fees up to date"].median()
    sens = (X_train_raw["Tuition fees up to date"].values < median).astype(np.float32)  # 1=學費未繳

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    members = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_train), start=1):
        model = _train_fold(X, y_train, sens, tr_idx, va_idx,
                            epochs=epochs, patience=patience, device=device)
        members.append(model)
        print(f"  ✓ fold {fold}/{n_splits} 完成")
    return members, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="輸出正式 MinDiff 5-fold ensemble 至 models/")
    parser.add_argument("--data", default="data.csv", help="真實資料 CSV 路徑(含 Target 欄)")
    parser.add_argument("--epochs", type=int, default=100, help="每折最大 epoch（早停可能提前結束）")
    parser.add_argument("--patience", type=int, default=10, help="早停 patience（比照 notebook）")
    parser.add_argument("--n-splits", type=int, default=5, help="ensemble 折數（soft-voting 成員數）")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"找不到資料: {args.data}\n"
            "(這是真實學生資料、不在 repo 內；請用 --data 指定路徑，例如 notebooks/data.csv)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device} | 資料: {args.data}")
    print(f"best params: {BEST_PARAMS}")

    X_train, y_train = load_real(args.data)
    print(f"訓練集: {X_train.shape} | 退學比例: {y_train.mean():.2%} | {args.n_splits}-fold ensemble")
    members, scaler = train_ensemble(X_train, y_train, epochs=args.epochs,
                                     patience=args.patience, n_splits=args.n_splits, device=device)

    Path("models").mkdir(exist_ok=True)
    save_ensemble_checkpoint("models/model.pt", members,
                             input_dim=len(FEATURE_ORDER),
                             dropout_1=BEST_PARAMS["dropout_1"], dropout_2=BEST_PARAMS["dropout_2"],
                             model_version=MODEL_VERSION, label_names=LABELS)
    joblib.dump(scaler, "models/preprocessor.joblib")
    print(f"✅ 已輸出 {len(members)}-fold ensemble → models/model.pt + models/preprocessor.joblib "
          f"(version={MODEL_VERSION})")

    # 即時驗證：用 src.inference(會自動以 EnsembleMLP soft-voting)對一筆訓練樣本推論
    try:
        from src.inference import predict
        sample = {n: float(X_train.iloc[0][n]) for n in RAW_FEATURES}
        r = predict(sample)
        print(f"🔎 推論驗證: {r['outcome']} | dropout_prob={r['dropout_probability']:.4f} | risk={r['risk_level']}")
    except Exception as exc:  # noqa: BLE001
        print(f"(推論驗證略過: {exc})")


if __name__ == "__main__":
    main()
