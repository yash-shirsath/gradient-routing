# %%
import os
from typing import Literal, Optional

import torch as t
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import datasets, transforms
from tqdm import tqdm

from synthetic import resize_and_reposition

import matplotlib.pyplot as plt


class DataManager:
    def __init__(self):
        self.cache_dir = "data/cache"
        self.synthetic_dir = "data/synthetic"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.synthetic_dir, exist_ok=True)

        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None

    def prepare_data(
        self,
        recipe: list[Literal["mnist", "synthetic"]],
        val_split: float = 0.1,
        batch_size: int = 1024,
    ):
        """
        Loads preprocessed data based on recipe
        Assembles train, val, and test loaders
        """

        if "mnist" in recipe:
            train_tensors, test_tensors, train_labels, test_labels = self._load_mnist()

        if "synthetic" in recipe:
            (
                synthetic_train_tensors,
                synthetic_train_labels,
                synthetic_test_tensors,
                synthetic_test_labels,
            ) = self._load_synthetic()

            train_tensors = t.cat([train_tensors, synthetic_train_tensors])
            train_labels = t.cat([train_labels, synthetic_train_labels])
            test_tensors = t.cat([test_tensors, synthetic_test_tensors])
            test_labels = t.cat([test_labels, synthetic_test_labels])

        train_dataset = torch.utils.data.TensorDataset(train_tensors, train_labels)  # type: ignore
        test_dataset = torch.utils.data.TensorDataset(test_tensors, test_labels)  # type: ignore

        if val_split > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [1 - val_split, val_split],
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True
            )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

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
        return os.path.join(self.cache_dir, data_type, f"{split}_{suffix}.pt")

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

    def _load_mnist(self) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
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
                [
                    transforms.ToTensor(),
                ]
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

    def _load_synthetic(self) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
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
                self._generate_synthetic()
            )

        return train_tensors, train_labels, test_tensors, test_labels  # type: ignore

    def _generate_synthetic(self, n=5):
        """
        Creates and caches n augments of each element in mnist
        """
        train_tensors, test_tensors, train_labels, test_labels = self._load_mnist()

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


# %%

data_manager = DataManager()
data_manager.prepare_data(["mnist"], val_split=0.0, batch_size=1024)
assert data_manager.train_loader is not None
x, y = next(iter(data_manager.train_loader))
print(x.shape)
print(y.shape)

# %%

print(f"Mean: {x[0].mean():.4f}")
print(f"Std: {x[0].std():.4f}")
print(f"Min: {x[0].min():.4f}")
print(f"Max: {x[0].max():.4f}")


# %%
mnist_train = datasets.MNIST("data", train=True, download=True)
transform_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
tensors = t.stack(
    [transform_to_tensor(mnist_train[i][0]) for i in range(len(mnist_train))]
)
transform_normalize = transforms.Normalize(2, 1)

# Create figure with 4 rows and 10 columns
fig, axes = plt.subplots(4, 10, figsize=(20, 8))

for i in range(10):
    # Get image and convert to tensor
    tensor_img = tensors[i]
    normalized = transform_normalize(tensor_img)

    # Row 1: Original tensor image
    axes[0, i].imshow(tensor_img.squeeze())
    axes[0, i].set_title(f"Image {i}")
    axes[0, i].axis("off")

    # Row 2: Original tensor summary stats
    axes[1, i].axis("off")
    stats_text = f"Mean: {tensor_img.mean():.2f}\nStd: {tensor_img.std():.2f}\nMin: {tensor_img.min():.2f}\nMax: {tensor_img.max():.2f}"
    axes[1, i].text(
        0.5, 0.5, stats_text, ha="center", va="center", transform=axes[1, i].transAxes
    )

    # Row 3: Normalized image
    axes[2, i].imshow(normalized.squeeze())
    axes[2, i].set_title(f"Normalized {i}")
    axes[2, i].axis("off")

    # Row 4: Normalized summary stats
    axes[3, i].axis("off")
    norm_stats_text = f"Mean: {normalized.mean():.2f}\nStd: {normalized.std():.2f}\nMin: {normalized.min():.2f}\nMax: {normalized.max():.2f}"
    axes[3, i].text(
        0.5,
        0.5,
        norm_stats_text,
        ha="center",
        va="center",
        transform=axes[3, i].transAxes,
    )

plt.tight_layout()
plt.show()

print(f"Mean: {tensors.mean():.4f}")
print(f"Std: {tensors.std():.4f}")
print(f"Min: {tensors.min():.4f}")
print(f"Max: {tensors.max():.4f}")

# %%


# %%
# Histogram of distributions before and after normalization
mnist_train = datasets.MNIST("data", train=True, download=True)
mnist_test = datasets.MNIST("data", train=False, download=True)

# Convert to tensors without normalization (before)
transform_before = transforms.Compose([transforms.ToTensor()])
tensors_before_train = t.stack(
    [transform_before(mnist_train[i][0]) for i in range(len(mnist_train))]
)
tensors_before_test = t.stack(
    [transform_before(mnist_test[i][0]) for i in range(len(mnist_test))]
)

# With normalization (after)
transform_after = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.13, 0.3)]
)
tensors_after_train = t.stack(
    [transform_after(mnist_train[i][0]) for i in range(len(mnist_train))]
)
tensors_after_test = t.stack(
    [transform_after(mnist_test[i][0]) for i in range(len(mnist_test))]
)

# Create figure with 2x2 subplots for the histograms
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Distribution of Pixel Values Before and After Normalization", fontsize=16)

# Plot histograms
axes[0, 0].hist(tensors_before_train.flatten().numpy(), bins=50, alpha=0.7)
axes[0, 0].set_title("Train Data - Before Normalization")
axes[0, 0].set_xlabel("Pixel Value")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(tensors_before_test.flatten().numpy(), bins=50, alpha=0.7)
axes[0, 1].set_title("Test Data - Before Normalization")
axes[0, 1].set_xlabel("Pixel Value")
axes[0, 1].set_ylabel("Frequency")

axes[1, 0].hist(tensors_after_train.flatten().numpy(), bins=50, alpha=0.7)
axes[1, 0].set_title("Train Data - After Normalization")
axes[1, 0].set_xlabel("Pixel Value")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(tensors_after_test.flatten().numpy(), bins=50, alpha=0.7)
axes[1, 1].set_title("Test Data - After Normalization")
axes[1, 1].set_xlabel("Pixel Value")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# Print summary statistics
print("Before Normalization:")
print(
    f"Train Mean: {tensors_before_train.mean():.4f}, Std: {tensors_before_train.std():.4f}"
)
print(
    f"Test Mean: {tensors_before_test.mean():.4f}, Std: {tensors_before_test.std():.4f}"
)

print("\nAfter Normalization:")
print(
    f"Train Mean: {tensors_after_train.mean():.4f}, Std: {tensors_after_train.std():.4f}"
)
print(
    f"Test Mean: {tensors_after_test.mean():.4f}, Std: {tensors_after_test.std():.4f}"
)

# %%
