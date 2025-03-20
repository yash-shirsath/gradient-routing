# %%
import torch as t
from torchvision.transforms import functional as F


def resize_and_reposition(
    image: t.Tensor, max_rot_deg=30, resize_factor=0.5
) -> t.Tensor:
    """
    Takes a preprocessed MNIST digit image of shape (28*28), resizes it to 60% of original size,
    and places it at a random position within the 28x28 grid.

    Args:
        image: A tensor of shape (28, 28) containing the MNIST digit

    Returns:
        A tensor of shape (28, 28) with the resized and repositioned digit
    """
    assert image.shape == (28, 28)
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


# %%
