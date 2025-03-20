from typing import Literal, Optional
import torch as t
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from dataclasses import dataclass
from jaxtyping import Float, Int
from synthetic import resize_and_reposition


@dataclass(frozen=True)
class MNISTConfig:
    img_dim = 28
    log_wandb = False
    lr = 1e-3
    val_split = 0.1
    batch_size = 1024
    epochs = 10
    checkpoint_dir = "checkpoints"
    best_model_path = "checkpoints/best_model.pt"
    latest_model_path = "checkpoints/latest_model.pt"
    cache_dir = "data/cache"
    synthetic_dir = "data/synthetic"


class DataManager:
    def __init__(self, config: MNISTConfig):
        self.config = config
        os.makedirs(self.config.cache_dir, exist_ok=True)

        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None

    def _load_from_cache(
        self,
        data_type: Literal["synthetic", "base"],
        split: Literal["train", "test"],
        suffix: Literal["tensors", "labels"],
    ):
        """Load tensors from cache if they exist."""
        if os.path.exists(self._build_path(data_type, split, suffix)):
            print(f"Loading {data_type} {split} {suffix} from cache...")
            data = t.load(self._build_path(data_type, split, suffix))
            return data
        return None

    def _build_path(
        self,
        data_type: Literal["synthetic", "base"],
        split: Literal["train", "test"],
        suffix: Literal["tensors", "labels"],
    ):
        return os.path.join(self.config.cache_dir, data_type, f"{split}_{suffix}.pt")

    def _save_to_cache(
        self,
        data_type: Literal["synthetic", "base"],
        split: Literal["train", "test"],
        suffix: Literal["tensors", "labels"],
        data: t.Tensor,
    ):
        """Save processed tensors to cache."""
        print(f"Saving {data_type} {split} {suffix} to cache...")
        os.makedirs(
            os.path.dirname(self._build_path(data_type, split, suffix)), exist_ok=True
        )
        t.save(data, self._build_path(data_type, split, suffix))

    def load_mnist(self) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """
        Attempts to load from cache if available, otherwise preprocesses and caches data.

        Returns:
            train_tensors: t.Tensor
            test_tensors: t.Tensor
            train_labels: t.Tensor
            test_labels: t.Tensor
        """

        train_tensors, test_tensors, train_labels, test_labels = (
            self._load_from_cache("base", "train", "tensors"),
            self._load_from_cache("base", "test", "tensors"),
            self._load_from_cache("base", "train", "labels"),
            self._load_from_cache("base", "test", "labels"),
        )
        if any(
            data is None
            for data in [train_tensors, test_tensors, train_labels, test_labels]
        ):
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
            self._save_to_cache("base", "train", "tensors", train_tensors)
            self._save_to_cache("base", "test", "tensors", test_tensors)
            self._save_to_cache("base", "train", "labels", train_labels)
            self._save_to_cache("base", "test", "labels", test_labels)

        return train_tensors, test_tensors, train_labels, test_labels  # type: ignore

    def prepare_data(
        self,
        recipe: list[Literal["mnist", "synthetic"]],
    ):
        """
        Loads preprocessed data based on recipe
        Assembles train, val, and test loaders
        """
        val_split = self.config.val_split
        batch_size = self.config.batch_size

        if "mnist" in recipe:
            train_tensors, test_tensors, train_labels, test_labels = self.load_mnist()

        if "synthetic" in recipe:
            (
                synthetic_train_tensors,
                synthetic_train_labels,
                synthetic_test_tensors,
                synthetic_test_labels,
            ) = self.load_synthetic()

            train_tensors = t.cat([train_tensors, synthetic_train_tensors])
            train_labels = t.cat([train_labels, synthetic_train_labels])
            test_tensors = t.cat([test_tensors, synthetic_test_tensors])
            test_labels = t.cat([test_labels, synthetic_test_labels])

        train_dataset = torch.utils.data.TensorDataset(train_tensors, train_labels)  # type: ignore
        test_dataset = torch.utils.data.TensorDataset(test_tensors, test_labels)  # type: ignore

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [1 - val_split, val_split],
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

    def load_synthetic(self) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """
        Attempts to load from cache if available, otherwise generates and caches data.
        Returns train, train_labels, test, test_labels
        """
        train_tensors, train_labels, test_tensors, test_labels = (
            self._load_from_cache("synthetic", "train", "tensors"),
            self._load_from_cache("synthetic", "train", "labels"),
            self._load_from_cache("synthetic", "test", "tensors"),
            self._load_from_cache("synthetic", "test", "labels"),
        )
        if any(
            data is None
            for data in [train_tensors, train_labels, test_tensors, test_labels]
        ):
            train_tensors, train_labels, test_tensors, test_labels = (
                self.generate_synthetic()
            )

        return train_tensors, train_labels, test_tensors, test_labels  # type: ignore

    def generate_synthetic(self, n=5):
        """
        Creates and caches n augments of each element in mnist
        """
        train_tensors, test_tensors, train_labels, test_labels = self.load_mnist()

        def generate(x: t.Tensor, y: t.Tensor, n: int):
            synthetic_tensors = []
            for i in tqdm(
                range(len(x)),
                desc=f"Generating {n} augments per item in mnist",
                total=len(x),
            ):
                for _ in range(n):
                    synthetic_tensors.append(
                        resize_and_reposition(x[i].squeeze(0)).unsqueeze(0)
                    )
            return t.stack(synthetic_tensors), y.repeat_interleave(n)

        synthetic_tensors, synthetic_labels = generate(train_tensors, train_labels, n)
        self._save_to_cache("synthetic", "train", "tensors", synthetic_tensors)
        self._save_to_cache("synthetic", "train", "labels", synthetic_labels)

        synthetic_tensors, synthetic_labels = generate(test_tensors, test_labels, n)
        self._save_to_cache("synthetic", "test", "tensors", synthetic_tensors)
        self._save_to_cache("synthetic", "test", "labels", synthetic_labels)

        return synthetic_tensors, synthetic_labels, test_tensors, test_labels


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
        self.opt = t.optim.AdamW(self.model.parameters(), lr=self.config.lr)
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
                        self.save_checkpoint(self.config.latest_model_path)
                        # Save best model if validation loss improved
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(self.config.best_model_path)


def main():
    config = MNISTConfig()
    data_manager = DataManager(config)
    data_manager.prepare_data(["mnist", "synthetic"])

    model = MNISTClassifier(config)
    trainer = ModelManager(config, data_manager, model)
    trainer.train()


if __name__ == "__main__":
    main()
