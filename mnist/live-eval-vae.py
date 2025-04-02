# %%
import torch as t
import matplotlib.pyplot as plt
import random
from ipywidgets import interactive, FloatSlider, VBox, HBox
import numpy as np
from IPython.display import display
import torch.nn.functional as F

from vae import VAE, VAEConfig
from checkpoint import Checkpoint
from torch.utils.data import DataLoader
from data import DataManager
from typing import Union, Optional, Tuple, Any
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Try to import Autoencoder
try:
    from autoencoder import Autoencoder, AutoencoderConfig

    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False

# %%
data_manager = DataManager()


data_manager = DataManager()
mnist_data = datasets.MNIST(
    "data", train=True, download=True, transform=transforms.ToTensor()
)
dataloader = DataLoader(mnist_data, batch_size=128, shuffle=True)  # type: ignore
validation_data = datasets.MNIST(
    "data", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(validation_data, batch_size=128, shuffle=True)  # type: ignore
data_manager.train_loader = dataloader
data_manager.test_loader = test_loader

config = VAEConfig()
model = VAE(config)
checkpoint = Checkpoint(
    run_name="vae_chris_no_norm__vae__25_04_01_15_07",
    model=model,
    optimizer=t.optim.AdamW(model.parameters(), lr=1),
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
        losses = [
            model.loss(
                reconstructions[i].unsqueeze(0),
                x_samples[i].unsqueeze(0),
                mu[i].unsqueeze(0),
                logvar[i].unsqueeze(0),
            )[0]
            for i in range(num_samples)
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


# %%
def check_reconstruction_similarity(
    model: Union[VAE, Any],
    loader: DataLoader,
    num_samples: int = 10,
    model_type: str = "vae",
):
    """
    Check if reconstructions are similar across different inputs

    Parameters:
    -----------
    model: Union[VAE, Autoencoder]
        The model to evaluate
    loader: DataLoader
        DataLoader containing test or validation data
    num_samples: int
        Number of samples to compare
    model_type: str
        Type of model - either "vae" or "autoencoder"
    """
    assert model_type in ["vae", "autoencoder"], (
        "model_type must be 'vae' or 'autoencoder'"
    )

    model.eval()
    with t.no_grad():
        # Get a random batch
        x, y = next(iter(loader))
        # Randomly select indices
        indices = random.sample(range(len(x)), min(num_samples, len(x)))
        x_samples = x[indices]

        # Get reconstructions based on model type
        if model_type == "vae":
            reconstructions, mu, logvar = model(x_samples)
            # Compute loss for VAE
            losses = [
                model.loss(
                    reconstructions[i].unsqueeze(0),
                    x_samples[i].unsqueeze(0),
                    mu[i].unsqueeze(0),
                    logvar[i].unsqueeze(0),
                )[0].item()
                for i in range(num_samples)
            ]
        else:  # autoencoder
            reconstructions = model(x_samples)
            # Compute loss for autoencoder
            losses = [
                model.loss(
                    reconstructions[i].unsqueeze(0),
                    x_samples[i].unsqueeze(0),
                ).item()
                for i in range(num_samples)
            ]

        # Flatten the images for easier comparison
        flat_reconstructions = reconstructions.view(num_samples, -1)
        flat_originals = x_samples.view(num_samples, -1)

        # Check if reconstructions are identical
        identical_check = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                if i != j:
                    identical_check.append(
                        t.allclose(reconstructions[i], reconstructions[j], rtol=1e-4)
                    )

        print(f"Are any reconstructions bit-identical? {any(identical_check)}")
        if any(identical_check):
            identical_pairs = [
                (i, j)
                for idx, (i, j) in enumerate(
                    [
                        (i, j)
                        for i in range(num_samples)
                        for j in range(i + 1, num_samples)
                    ]
                )
                if identical_check[idx]
            ]
            print(f"Identical pairs: {identical_pairs}")

        # Compute pairwise cosine similarity for reconstructions
        recon_sim_matrix = []
        for i in range(num_samples):
            row = []
            for j in range(num_samples):
                sim = F.cosine_similarity(
                    flat_reconstructions[i].unsqueeze(0),
                    flat_reconstructions[j].unsqueeze(0),
                )
                row.append(sim.item())
            recon_sim_matrix.append(row)
        recon_sim_matrix = np.array(recon_sim_matrix)

        # Compute pairwise cosine similarity for original images
        orig_sim_matrix = []
        for i in range(num_samples):
            row = []
            for j in range(num_samples):
                sim = F.cosine_similarity(
                    flat_originals[i].unsqueeze(0), flat_originals[j].unsqueeze(0)
                )
                row.append(sim.item())
            orig_sim_matrix.append(row)
        orig_sim_matrix = np.array(orig_sim_matrix)

        # Compute L1 distance matrix for reconstructions
        recon_l1_matrix = []
        for i in range(num_samples):
            row = []
            for j in range(num_samples):
                l1 = (
                    (flat_reconstructions[i] - flat_reconstructions[j])
                    .abs()
                    .mean()
                    .item()
                )
                row.append(l1)
            recon_l1_matrix.append(row)
        recon_l1_matrix = np.array(recon_l1_matrix)

        # Compute mean similarity
        mean_recon_sim = np.mean(recon_sim_matrix[~np.eye(num_samples, dtype=bool)])
        mean_orig_sim = np.mean(orig_sim_matrix[~np.eye(num_samples, dtype=bool)])
        mean_recon_l1 = np.mean(recon_l1_matrix[~np.eye(num_samples, dtype=bool)])

        # Calculate standard deviation
        std_recon_sim = np.std(recon_sim_matrix[~np.eye(num_samples, dtype=bool)])
        std_orig_sim = np.std(orig_sim_matrix[~np.eye(num_samples, dtype=bool)])
        std_recon_l1 = np.std(recon_l1_matrix[~np.eye(num_samples, dtype=bool)])

        # Plot similarity matrices
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        im1 = ax1.imshow(orig_sim_matrix, cmap="viridis", vmin=-1, vmax=1)
        ax1.set_title(
            f"Original Images Similarity\nMean: {mean_orig_sim:.4f}, Std: {std_orig_sim:.4f}"
        )
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(recon_sim_matrix, cmap="viridis", vmin=-1, vmax=1)
        ax2.set_title(
            f"Reconstructions Similarity\nMean: {mean_recon_sim:.4f}, Std: {std_recon_sim:.4f}"
        )
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.imshow(recon_l1_matrix, cmap="viridis")
        ax3.set_title(
            f"Reconstructions L1 Distance\nMean: {mean_recon_l1:.4f}, Std: {std_recon_l1:.4f}"
        )
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.show()

        # Also display sample reconstructions with losses
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))

        for i in range(num_samples // 2):
            # Original image
            axes[0, i].imshow(x_samples[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Original {i}")

            # Reconstruction
            axes[1, i].imshow(reconstructions[i].squeeze(), cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Recon {i}, Loss: {losses[i]:.4f}")

        plt.tight_layout()
        plt.show()

        # Compute pixel statistics
        mean_pixel_val_orig = flat_originals.mean(dim=0)
        std_pixel_val_orig = flat_originals.std(dim=0)

        mean_pixel_val_recon = flat_reconstructions.mean(dim=0)
        std_pixel_val_recon = flat_reconstructions.std(dim=0)

        print(f"Original images pixel mean: {mean_pixel_val_orig.mean().item():.4f}")
        print(f"Original images pixel std: {std_pixel_val_orig.mean().item():.4f}")
        print(f"Reconstructions pixel mean: {mean_pixel_val_recon.mean().item():.4f}")
        print(f"Reconstructions pixel std: {std_pixel_val_recon.mean().item():.4f}")

        # If mean similarity is high, the reconstructions are similar across different inputs
        if mean_recon_sim > 0.9 and std_recon_sim < 0.1:
            print(
                "WARNING: Reconstructions appear to be very similar across different inputs!"
            )
            print(
                "This suggests the model might be reconstructing a mean image rather than the specific input."
            )

        return {
            "recon_sim_matrix": recon_sim_matrix,
            "orig_sim_matrix": orig_sim_matrix,
            "recon_l1_matrix": recon_l1_matrix,
            "mean_recon_sim": mean_recon_sim,
            "std_recon_sim": std_recon_sim,
            "mean_recon_l1": mean_recon_l1,
            "losses": losses,
        }


# Load and test VAE model
print("Testing VAE model:")
vae_results = check_reconstruction_similarity(
    model, data_manager.test_loader, num_samples=10, model_type="vae"
)

# Load and test Autoencoder model if available
if AUTOENCODER_AVAILABLE:
    print("\nTesting Autoencoder model:")
    autoencoder_config = AutoencoderConfig()
    autoencoder = Autoencoder(autoencoder_config)
    autoencoder_checkpoint = Checkpoint(
        run_name="autoencoder_reconstruction_loss_only_no_scheduler_25_03_25_06_53",
        model=autoencoder,
        optimizer=t.optim.AdamW(autoencoder.parameters(), lr=1e-3),
    )
    if autoencoder_checkpoint.load_checkpoint(load_best=True):
        autoencoder_results = check_reconstruction_similarity(
            autoencoder,
            data_manager.test_loader,
            num_samples=10,
            model_type="autoencoder",
        )

        # Compare the two models
        print("\nComparison between VAE and Autoencoder:")
        print(
            f"VAE mean reconstruction similarity: {vae_results['mean_recon_sim']:.4f}"
        )
        print(
            f"Autoencoder mean reconstruction similarity: {autoencoder_results['mean_recon_sim']:.4f}"
        )
        print(f"VAE mean L1 distance: {vae_results['mean_recon_l1']:.4f}")
        print(
            f"Autoencoder mean L1 distance: {autoencoder_results['mean_recon_l1']:.4f}"
        )
        print(f"VAE mean loss: {np.mean(vae_results['losses']):.4f}")
        print(f"Autoencoder mean loss: {np.mean(autoencoder_results['losses']):.4f}")
    else:
        print("Failed to load autoencoder checkpoint")
else:
    print("Autoencoder module not available")

# %%

# %%
# def manipulate_latent(model: VAE, loader: DataLoader):
#     model.eval()
#     with t.no_grad():
#         x, y = next(iter(loader))
#         original_img = x[0].flatten().unsqueeze(0)
#         z_original = model.reparametrize(*model.encode(original_img))

#         # Create the figure with subplots
#         fig = plt.figure(figsize=(12, 6))
#         gs = fig.add_gridspec(1, 2)

#         # Image subplots
#         ax1 = fig.add_subplot(gs[0, 0])
#         ax2 = fig.add_subplot(gs[0, 1])

#         # Create initial plots
#         img1 = ax1.imshow(original_img.reshape(28, 28), cmap="gray")
#         ax1.set_title("Original Image")
#         ax1.axis("off")

#         img2 = ax2.imshow(original_img.reshape(28, 28), cmap="gray")
#         ax2.set_title("Reconstructed Image")
#         ax2.axis("off")

#         plt.tight_layout()
#         plt.show()

#         def update_plot(**kwargs):
#             # Create a new latent vector with slider values
#             z_new = t.tensor(
#                 [kwargs[f"z{i}"] for i in range(10)], dtype=t.float32
#             ).unsqueeze(0)

#             # Decode the new latent vector
#             reconstructed = model.decode(z_new).squeeze()

#             # Update the reconstructed image data
#             img2.set_data(reconstructed.detach().numpy().reshape(28, 28))

#             # Force the plot to update
#             fig.canvas.draw()
#             fig.canvas.flush_events()

#             # Print latent vectors
#             print(f"Original latent: {z_original[0].numpy().round(2)}")
#             print(f"Current latent:  {z_new[0].numpy().round(2)}")
#             print("-" * 50)

#         # Create sliders for each dimension
#         sliders = {
#             f"z{i}": FloatSlider(
#                 value=float(z_original[0, i].item()),
#                 min=-3.0,
#                 max=3.0,
#                 step=0.1,
#                 description=f"z{i}",
#             )
#             for i in range(10)
#         }

#         # Create interactive plot
#         interactive_plot = interactive(update_plot, **sliders)

#         # Display the interactive plot
#         display(interactive_plot)


# manipulate_latent(model, data_manager.test_loader)

# # %%
