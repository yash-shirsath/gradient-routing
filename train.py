import os

import torch as t
from jaxtyping import Float, Int
from tqdm import tqdm

from data import DataManager
from mlp import MNISTClassifier, MNISTConfig


class ModelManager:
    def __init__(
        self, config: MNISTConfig, data_manager: DataManager, model: MNISTClassifier
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.model = model
        self.opt = t.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.best_val_loss = float("inf")

        # Checkpoints
        self.checkpoint_dir = "checkpoints"
        self.best_model_path = "checkpoints/best_model.pt"
        self.latest_model_path = "checkpoints/latest_model.pt"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, path: str):
        """Save model and optimizer state to disk."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        t.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, load_best: bool = False):
        """Load model and optimizer state from disk."""
        path = self.best_model_path if load_best else self.latest_model_path
        if os.path.exists(path):
            checkpoint = t.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded checkpoint from {path}")
            return True
        return False

    def evaluate(self):
        with t.no_grad():
            self.model.eval()
            loss = 0
            for x, y in self.data_manager.val_loader:
                pred = self.model(x)
                loss += self.model.loss(pred, y)
            avg_loss = loss / len(self.data_manager.val_loader)
            print(f"Validation loss: {avg_loss:.4f}")
            self.model.train()
            return avg_loss

    def train_step(
        self, x: Float[t.Tensor, "batch img_size"], y: Int[t.Tensor, "batch"]
    ):
        assert self.data_manager.train_loader is not None
        self.opt.zero_grad()
        pred = self.model(x)
        loss = self.model.loss(pred, y)
        loss.backward()
        self.opt.step()
        return loss

    def train(self):
        self.model.train()
        assert self.data_manager.train_loader is not None
        for _ in range(self.config.epochs):
            for i, (x, y) in tqdm(
                enumerate(self.data_manager.train_loader), desc="Training"
            ):
                self.train_step(x, y)

                if i % 100 == 0:
                    val_loss = self.evaluate()
                    if i % 1000 == 0:
                        self.save_checkpoint(self.latest_model_path)
                        # Save best model if validation loss improved
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(self.best_model_path)


def main():
    config = MNISTConfig(epochs=10)
    data_manager = DataManager()
    data_manager.prepare_data(
        ["mnist", "synthetic"], val_split=config.val_split, batch_size=config.batch_size
    )

    model = MNISTClassifier(config)
    trainer = ModelManager(config, data_manager, model)
    trainer.train()


if __name__ == "__main__":
    main()
