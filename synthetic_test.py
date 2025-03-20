from synthetic import resize_and_reposition
from mnist import DataManager, MNISTConfig
import matplotlib.pyplot as plt


def visualize_resize_and_reposition():
    config = MNISTConfig()
    data_manager = DataManager(config)
    data_manager.load_mnist()

    # Get 4 random images from test set
    test_loader = data_manager.test_loader
    assert test_loader is not None
    images, labels = next(iter(test_loader))
    images, labels = images[:4], labels[:4]

    # Create figure with 4x6 subplots
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    axes = axes.ravel()

    # For each original image, show it and 5 augmented versions
    for i in range(4):
        # Show original image
        axes[i * 6].imshow(images[i].squeeze(), cmap="gray")
        axes[i * 6].set_title(f"Original\nLabel: {labels[i].item()}")
        axes[i * 6].axis("off")

        # Show 5 augmented versions
        for j in range(5):
            augmented = resize_and_reposition(images[i].squeeze())
            axes[i * 6 + j + 1].imshow(augmented, cmap="gray")
            axes[i * 6 + j + 1].set_title(f"Augmented {j + 1}")
            axes[i * 6 + j + 1].axis("off")

    plt.tight_layout()
    plt.show()


visualize_resize_and_reposition()
