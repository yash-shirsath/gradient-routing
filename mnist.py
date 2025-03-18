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


class DataManager:
    def __init__(self, config: MNISTConfig):
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.train_cache_path = os.path.join(self.cache_dir, "train_tensors.pt")
        self.test_cache_path = os.path.join(self.cache_dir, "test_tensors.pt")
        self.train_labels_cache_path = os.path.join(self.cache_dir, "train_labels.pt")
        self.test_labels_cache_path = os.path.join(self.cache_dir, "test_labels.pt")
        self.config = config

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

    def get_mnist(self):
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
                    transform(img).reshape(self.config.img_dim * self.config.img_dim)
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

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=512
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

        return train_loader, test_loader


d = DataManager(MNISTConfig())
d.get_mnist()


class MNISTClassifier(nn.Module):
    def __init__(self, config: MNISTConfig) -> None:
        super().__init__()
        self.config = config

        in_dim = self.config.img_dim * self.config.img_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(
        self, x: Float[t.Tensor, "batch img_size"]
    ) -> Float[t.Tensor, "batch 10"]:
        return self.net(x)


class Trainer:
    def __init__(
        self, config: MNISTConfig, data_manager: DataManager, model: MNISTClassifier
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.model = model

    def train_setup(self):
        pass

    def evaluate(self):
        pass

    def train(self):
        pass
