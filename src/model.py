"""MLP 模型定義 (與需求書第 3 節對應)。

組員 B 若使用相同架構，可直接沿用此檔案；
若架構不同，只需在 save_checkpoint / load_checkpoint 維持相同的 checkpoint 格式即可。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DropoutMLP(nn.Module):
    """Input -> Hidden(ReLU+Dropout) x N -> Softmax(3 classes) 的輸出 logits。"""

    def __init__(self, input_dim: int, hidden_dims=(64, 32), num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 回傳 logits
        return self.net(x)


def save_checkpoint(path: str, model: DropoutMLP, *, input_dim: int, hidden_dims,
                    num_classes: int, dropout: float, model_version: str, label_names):
    """以統一格式儲存模型，inference 層只認得這個格式。"""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "num_classes": num_classes,
            "dropout": dropout,
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
        hidden_dims=ckpt["hidden_dims"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
