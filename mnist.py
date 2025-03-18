from typing import Optional
import torch as t
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from dataclasses import dataclass
from jaxtyping import Float, Int


@dataclass(frozen=True)
class MNISTConfig:
    img_dim = 28
    log_wandb = False
    lr = 1e-3
    val_split = 0.1
    batch_size = 512
    epochs = 10
    checkpoint_dir = "checkpoints"
    best_model_path = "checkpoints/best_model.pt"
    latest_model_path = "checkpoints/latest_model.pt"


class DataManager:
    def __init__(self, config: MNISTConfig):
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.train_cache_path = os.path.join(self.cache_dir, "train_tensors.pt")
        self.test_cache_path = os.path.join(self.cache_dir, "test_tensors.pt")
        self.train_labels_cache_path = os.path.join(self.cache_dir, "train_labels.pt")
        self.test_labels_cache_path = os.path.join(self.cache_dir, "test_labels.pt")
        self.config = config

        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None

    def _load_from_cache(self):
        """Load tensors from cache if they exist."""
        if all(
            os.path.exists(path)
            for path in [
                self.train_cache_path,
                self.test_cache_path,
                self.train_labels_cache_path,
                self.test_labels_cache_path,
            ]
        ):
            print("Loading data from cache...")
            train_tensors = t.load(self.train_cache_path)
            test_tensors = t.load(self.test_cache_path)
            train_labels = t.load(self.train_labels_cache_path)
            test_labels = t.load(self.test_labels_cache_path)
            return train_tensors, test_tensors, train_labels, test_labels
        return None

    def _save_to_cache(self, train_tensors, test_tensors, train_labels, test_labels):
        """Save processed tensors to cache."""
        print("Saving data to cache...")
        t.save(train_tensors, self.train_cache_path)
        t.save(test_tensors, self.test_cache_path)
        t.save(train_labels, self.train_labels_cache_path)
        t.save(test_labels, self.test_labels_cache_path)

    def load_mnist(self):
        """Return MNIST data using the provided Tensor class."""
        # Try to load from cache first
        cached_data = self._load_from_cache()
        if cached_data is not None:
            train_tensors, test_tensors, train_labels, test_labels = cached_data
        else:
            mnist_train = datasets.MNIST("data", train=True, download=True)
            mnist_test = datasets.MNIST("data", train=False, download=True)

            print("Preprocessing data...")
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.28,), (0.35,))]
            )

            train_tensors = t.stack(
                [
                    transform(img)
                    for img, _ in tqdm(mnist_train, desc="Processing Training Data")  # type: ignore
                ]
            )
            test_tensors = t.stack(
                [
                    transform(img)
                    for img, _ in tqdm(mnist_test, desc="Processing Test Data")  # type: ignore
                ]
            )
            train_labels = t.tensor([label for _, label in mnist_train])
            test_labels = t.tensor([label for _, label in mnist_test])

            # Save to cache
            self._save_to_cache(train_tensors, test_tensors, train_labels, test_labels)

        train_dataset = torch.utils.data.TensorDataset(train_tensors, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_tensors, test_labels)

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [1 - self.config.val_split, self.config.val_split],
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=self.config.batch_size
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=True, batch_size=self.config.batch_size
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=True
        )


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


class ModelManager:
    def __init__(
        self, config: MNISTConfig, data_manager: DataManager, model: MNISTClassifier
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.model = model
        self.opt = t.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.best_val_loss = float("inf")
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

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
        path = (
            self.config.best_model_path if load_best else self.config.latest_model_path
        )
        if os.path.exists(path):
            checkpoint = t.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded checkpoint from {path}")
            return True
        return False

    def train_setup(self):
        """
        get data
        initialize model weights? (todo)
        initialize w&b (todo)
        initialize optimizer
        """
        self.data_manager.load_mnist()

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
                loss = self.train_step(x, y)
                print(f"Batch loss: {loss:.4f}")
                if i % 100 == 0:
                    val_loss = self.evaluate()
                    self.save_checkpoint(self.config.latest_model_path)
                    # Save best model if validation loss improved
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(self.config.best_model_path)


def main():
    config = MNISTConfig()
    data_manager = DataManager(config)
    model = MNISTClassifier(config)
    trainer = ModelManager(config, data_manager, model)
    trainer.train_setup()
    trainer.train()


if __name__ == "__main__":
    main()
