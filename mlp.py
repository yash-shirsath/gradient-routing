from dataclasses import dataclass

import torch as t
import torch.nn as nn
from jaxtyping import Float, Int


@dataclass(frozen=True)
class MNISTConfig:
    img_dim: int = 28
    log_wandb: bool = False
    lr: float = 1e-3
    epochs: int = 10

    val_split: float = 0.1
    batch_size: int = 1024


class MNISTClassifier(nn.Module):
    def __init__(self, config: MNISTConfig) -> None:
        super().__init__()
        self.config = config

        in_dim = self.config.img_dim * self.config.img_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(
        self, x: Float[t.Tensor, "batch img_size"]
    ) -> Float[t.Tensor, "batch 10"]:
        return self.net(x)

    def loss(
        self, preds: Float[t.Tensor, "batch 10"], y: Int[t.Tensor, "batch"]
    ) -> Float[t.Tensor, ""]:
        return t.nn.functional.cross_entropy(preds, y)
