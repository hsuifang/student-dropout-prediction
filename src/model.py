"""MLP 模型定義 (與需求書第 3 節 / 訓練程式對應)。

二分類版本：架構與訓練程式的 `DropoutNN` 完全一致，輸出單一 logit
(經 sigmoid → P(Dropout))。屬性名稱維持 `layer_stack`，使 state_dict 的
參數鍵 (layer_stack.0.weight ...) 能直接載入組員 B 訓練好的真實權重。

組員 B 若調整 dropout 比例 (例如 Optuna 找到的值)，不影響參數鍵，
仍可由 load_checkpoint 載入；dropout 值僅作為 metadata 記錄。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DropoutMLP(nn.Module):
    """Input(14) → 64+BN+ReLU+Drop → 32+BN+ReLU+Drop → 16+ReLU → Linear(1)。

    輸出單一 logit；二分類，正類為 Dropout。
    """

    def __init__(self, input_dim: int, dropout_1: float = 0.4, dropout_2: float = 0.3):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 回傳 (N, 1) logits
        return self.layer_stack(x)


def save_checkpoint(path: str, model: DropoutMLP, *, input_dim: int,
                    dropout_1: float, dropout_2: float, model_version: str,
                    label_names):
    """以統一格式儲存模型，inference 層只認得這個格式。

    num_classes 固定為 1 (單 logit 二分類)，label_names = [負類, 正類]。
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "dropout_1": dropout_1,
            "dropout_2": dropout_2,
            "num_classes": 1,
            "model_version": model_version,
            "label_names": list(label_names),
        },
        path,
    )


def load_checkpoint(path: str, map_location="cpu"):
    """回傳 (model, checkpoint_dict)。"""
    ckpt = torch.load(path, map_location=map_location)
    model = DropoutMLP(
        input_dim=ckpt["input_dim"],
        dropout_1=ckpt.get("dropout_1", 0.4),
        dropout_2=ckpt.get("dropout_2", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
