from datetime import datetime
import os

import torch as t
import torch.nn as nn


class Checkpoint:
    def __init__(
        self,
        run_name: str,
        model: nn.Module,
        optimizer: t.optim.Optimizer,
    ) -> None:
        self.run_name = run_name
        self.model = model
        self.optimizer = optimizer
        self.best_val_loss = float("inf")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_dir = os.path.join(script_dir, "checkpoints", self.run_name)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        self.latest_model_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, val_loss: float, epoch: int):
        """Save model and optimizer state to disk."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": val_loss,
            "epoch": epoch,
            "timestamp": datetime.now(),
        }
        t.save(checkpoint, self.latest_model_path)
        print(f"Saved checkpoint to {self.latest_model_path}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            t.save(checkpoint, self.best_model_path)
            print(f"Saved best model to {self.best_model_path}")

    def load_checkpoint(self, load_best: bool = False):
        """Load model and optimizer state from disk."""
        path = self.best_model_path if load_best else self.latest_model_path
        if os.path.exists(path):
            checkpoint = t.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_loss = checkpoint["best_val_loss"]
            self.timestamp = checkpoint["timestamp"]
            print(f"Loaded checkpoint from {path}")
            return True
        else:
            print(f"No checkpoint found at {path}")
            return False
