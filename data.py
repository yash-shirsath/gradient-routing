# %%
from matplotlib import pyplot as plt
import torch as t
from torchvision.transforms import functional as F

from mnist import DataManager, MNISTConfig


def resize_and_reposition(
    image: t.Tensor, max_rot_deg=30, resize_factor=0.5
) -> t.Tensor:
    """
    Takes a 28x28 MNIST digit image, resizes it to 60% of original size,
    and places it at a random position within the 28x28 grid.

    Args:
        image: A tensor of shape (28, 28) containing the MNIST digit

    Returns:
        A tensor of shape (28, 28) with the resized and repositioned digit
    """
    # Calculate new size (60% of original)
    new_size = int(28 * resize_factor)
    fill = image.min().item()

    # Resize the image using interpolation
    resized = t.nn.functional.interpolate(
        image.unsqueeze(0).unsqueeze(0),
        size=(new_size, new_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Add random rotation between -MAX_ROTATION_DEG and +MAX_ROTATION_DEG degrees
    angle = t.rand(1).item() * (2 * max_rot_deg) - max_rot_deg
    resized = F.rotate(
        resized.unsqueeze(0),
        angle,
        interpolation=F.InterpolationMode.BILINEAR,
        expand=False,
        fill=[fill],
    ).squeeze()

    # Create output tensor filled with minimum value from input image
    output = t.full((28, 28), fill)

    # Calculate maximum valid position for top-left corner
    max_pos = 28 - new_size

    # Generate random position
    top = t.randint(0, max_pos + 1, (1,)).item()
    left = t.randint(0, max_pos + 1, (1,)).item()

    # Place the resized image at the random position
    output[top : top + new_size, left : left + new_size] = resized

    return output


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

# %%
