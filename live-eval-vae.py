# %%
import torch as t
import matplotlib.pyplot as plt
import random
from ipywidgets import interactive, FloatSlider, VBox, HBox
import numpy as np
from IPython.display import display

from vae import VAE
from checkpoint import Checkpoint
from torch.utils.data import DataLoader
from data import DataManager

# %%
data_manager = DataManager()
data_manager.prepare_data(recipe=["mnist", "synthetic"], val_split=0.0, batch_size=1024)

model = VAE()
checkpoint = Checkpoint(
    run_name="vae_no_routing_l1_only_25_03_26_10_52",
    model=model,
    optimizer=t.optim.AdamW(model.parameters(), lr=1),
    postfix_date=False,
)
assert checkpoint.load_checkpoint(load_best=True), "Failed to load checkpoint"

assert data_manager.train_loader is not None
assert data_manager.test_loader is not None


# %%
def visualize_samples(model: VAE, loader: DataLoader, num_samples: int = 6):
    model.eval()
    with t.no_grad():
        # Get a random batch
        x, y = next(iter(loader))
        # Randomly select indices
        indices = random.sample(range(len(x)), min(num_samples, len(x)))
        x_samples = x[indices]

        # Get reconstructions
        reconstructions, mu, logvar = model(x_samples)

        # Calculate losses
        losses = [
            model.loss(recon.unsqueeze(0), orig.unsqueeze(0), mu, logvar).item()
            for recon, orig, mu, logvar in zip(reconstructions, x_samples, mu, logvar)
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


def manipulate_latent(model: VAE, loader: DataLoader):
    model.eval()
    with t.no_grad():
        x, y = next(iter(loader))
        original_img = x[0].flatten().unsqueeze(0)
        z_original = model.reparametrize(*model.encode(original_img))

        # Create the figure with subplots
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2)

        # Image subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Create initial plots
        img1 = ax1.imshow(original_img.reshape(28, 28), cmap="gray")
        ax1.set_title("Original Image")
        ax1.axis("off")

        img2 = ax2.imshow(original_img.reshape(28, 28), cmap="gray")
        ax2.set_title("Reconstructed Image")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

        def update_plot(**kwargs):
            # Create a new latent vector with slider values
            z_new = t.tensor(
                [kwargs[f"z{i}"] for i in range(10)], dtype=t.float32
            ).unsqueeze(0)

            # Decode the new latent vector
            reconstructed = model.decode(z_new).squeeze()

            # Update the reconstructed image data
            img2.set_data(reconstructed.detach().numpy().reshape(28, 28))

            # Force the plot to update
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Print latent vectors
            print(f"Original latent: {z_original[0].numpy().round(2)}")
            print(f"Current latent:  {z_new[0].numpy().round(2)}")
            print("-" * 50)

        # Create sliders for each dimension
        sliders = {
            f"z{i}": FloatSlider(
                value=float(z_original[0, i].item()),
                min=-3.0,
                max=3.0,
                step=0.1,
                description=f"z{i}",
            )
            for i in range(10)
        }

        # Create interactive plot
        interactive_plot = interactive(update_plot, **sliders)

        # Display the interactive plot
        display(interactive_plot)


manipulate_latent(model, data_manager.test_loader)

# %%
