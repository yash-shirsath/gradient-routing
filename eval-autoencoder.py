# %%
import torch as t
import matplotlib.pyplot as plt
import random

from autoencoder import Autoencoder, AutoencoderConfig
from checkpoint import Checkpoint
from torch.utils.data import DataLoader
from data import DataManager

# %%
config = AutoencoderConfig()
data_manager = DataManager()
data_manager.prepare_data(recipe=["mnist", "synthetic"], val_split=0.0, batch_size=1024)

model = Autoencoder(config)
checkpoint = Checkpoint(
    run_name="autoencoder_reconstruction_loss_only_no_scheduler_25_03_25_06_53",
    model=model,
    optimizer=t.optim.AdamW(model.parameters(), lr=config.lr),
    postfix_date=False,
)
assert checkpoint.load_checkpoint(load_best=True), "Failed to load checkpoint"

assert data_manager.train_loader is not None
assert data_manager.test_loader is not None


# %%
def calculate_loss(model: Autoencoder, loader: DataLoader) -> float:
    model.eval()
    with t.no_grad():
        loss = 0
        for x, y in loader:
            pred = model(x)
            loss += model.loss(pred, x).item()
    return loss / len(loader)


test_loss = calculate_loss(model, data_manager.test_loader)
print(f"Test loss: {test_loss:.4f}")
train_loss = calculate_loss(model, data_manager.train_loader)
print(f"Train loss: {train_loss:.4f}")

# %%


def visualize_samples(model: Autoencoder, loader: DataLoader, num_samples: int = 6):
    model.eval()
    with t.no_grad():
        # Get a random batch
        x, y = next(iter(loader))
        # Randomly select indices
        indices = random.sample(range(len(x)), min(num_samples, len(x)))
        x_samples = x[indices]

        # Get reconstructions
        reconstructions = model(x_samples)

        # Calculate losses
        losses = [
            model.loss(recon.unsqueeze(0), orig.unsqueeze(0)).item()
            for recon, orig in zip(reconstructions, x_samples)
        ]

        # Create subplot grid
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))

        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(x_samples[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original")

            # Reconstruction
            axes[1, i].imshow(reconstructions[i].squeeze(), cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Loss: {losses[i]:.4f}")
            if i == 0:
                axes[1, i].set_title("Reconstruction")

        plt.tight_layout()
        plt.show()


visualize_samples(model, data_manager.test_loader)

# %%
