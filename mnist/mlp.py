from dataclasses import dataclass

import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from tqdm import tqdm

from checkpoint import Checkpoint
from data import DataManager


@dataclass(frozen=True)
class MLPConfig:
    img_dim: int = 28
    log_wandb: bool = False
    lr: float = 1e-3
    epochs: int = 10

    val_split: float = 0.1
    batch_size: int = 1024


class MNISTClassifier(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
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


class Trainer:
    def __init__(
        self,
        config: MLPConfig,
        data_manager: DataManager,
        model: MNISTClassifier,
        run_name: str,
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.model = model
        self.opt = t.optim.AdamW(model.parameters(), lr=config.lr)
        self.checkpoint = Checkpoint(
            run_name=run_name,
            model=model,
            optimizer=self.opt,
        )

    def evaluate(self) -> float:
        with t.no_grad():
            self.model.eval()
            loss = 0
            for x, y in self.data_manager.val_loader:
                pred = self.model(x)
                loss += self.model.loss(pred, y)
            avg_loss = loss / len(self.data_manager.val_loader)
            print(f"Validation loss: {avg_loss:.4f}")
            self.model.train()
            assert isinstance(avg_loss, t.Tensor)
            return avg_loss.item()

    def train_step(
        self, x: Float[t.Tensor, "batch img_size"], y: Int[t.Tensor, "batch"]
    ):
        self.opt.zero_grad()
        pred = self.model(x)
        loss = self.model.loss(pred, y)
        loss.backward()
        self.opt.step()
        return loss

    def train(self):
        self.model.train()
        assert self.data_manager.train_loader is not None
        total_steps = 0
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1} of {self.config.epochs}")
            for i, (x, y) in tqdm(
                enumerate(self.data_manager.train_loader), desc="Training"
            ):
                self.train_step(x, y)
                total_steps += 1

                if total_steps % 100 == 0:
                    val_loss = self.evaluate()
                if total_steps % 1000 == 0:
                    self.checkpoint.save_checkpoint(val_loss, epoch)


def train():
    config = MLPConfig(epochs=10)
    data_manager = DataManager()
    data_manager.prepare_data(
        ["mnist", "synthetic"], val_split=config.val_split, batch_size=config.batch_size
    )

    model = MNISTClassifier(config)
    trainer = Trainer(config, data_manager, model, "mnist_25_03_24_14_30")
    trainer.train()


if __name__ == "__main__":
    train()
