from dataclasses import dataclass

import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from checkpoint import Checkpoint
from data import DataManager
from jaxtyping import Float, Int

# adapted from https://github.com/g-w1/gradient-routed-vae/blob/main/vae.py. Thanks!


@dataclass
class VAEConfig:
    image_features: int = 28 * 28
    encoder_size_1: int = 2048
    encoder_size_2: int = 512
    latent_size: int = 10

    val_split: float = 0.1
    batch_size: int = 512
    epochs: int = 100
    start_lr: float = 1e-3

    def lr(self, epoch: int, start_lr: float) -> float:
        return start_lr * 0.9**epoch


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(
                self.config.image_features,
                self.config.encoder_size_1,
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.encoder_size_1,
                self.config.encoder_size_2,
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.encoder_size_2,
                self.config.latent_size,
            ),
        )
        self.mean_from_encoded = nn.Linear(
            self.config.latent_size,
            self.config.latent_size,
        )
        self.cov_diag_from_encoded = nn.Linear(
            self.config.latent_size,
            self.config.latent_size,
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                self.config.latent_size,
                self.config.encoder_size_2,
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.encoder_size_2,
                self.config.encoder_size_1,
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.encoder_size_1,
                self.config.image_features,
            ),
            nn.Sigmoid(),
        )

    def encode(self, images: t.Tensor):
        latent = self.encoder(images)
        assert latent.shape[-1] == self.config.latent_size
        mean_from_encoded = self.mean_from_encoded(latent)
        zeta = t.randn_like(mean_from_encoded)
        cov_diag_from_encoded = self.cov_diag_from_encoded(latent)
        z = mean_from_encoded + cov_diag_from_encoded * zeta
        z.relu_()
        return z, zeta, mean_from_encoded, cov_diag_from_encoded

    # def encode_and_mask(self, images: t.Tensor, labels: t.Tensor):
    #     encoded_unmasked, zeta, mean_from_encoded, cov_diag_from_encoded = self.encode(
    #         images
    #     )
    #     mask_one_hot = F.one_hot(labels, num_classes=self.config.latent_size).float()  # type: ignore
    #     encoded = (
    #         mask_one_hot * encoded_unmasked
    #         + (1 - mask_one_hot) * encoded_unmasked.detach()
    #     )
    #     return encoded, zeta, mean_from_encoded, cov_diag_from_encoded

    def loss(self, images: t.Tensor, labels=None):
        images = einops.rearrange(
            images, "batch chan width height -> batch (chan width height)"
        )
        if labels is not None:
            encoded, zeta, mean_from_encoded, cov_diag_from_encoded = (
                self.encode_and_mask(images, labels)
            )

        else:
            encoded, zeta, mean_from_encoded, cov_diag_from_encoded = self.encode(
                images
            )

        decoded = self.decoder(encoded)

        mse_loss = ((images - decoded).norm(dim=-1) ** 2).mean()
        kl_div_loss = (
            mean_from_encoded**2 + cov_diag_from_encoded.exp() - cov_diag_from_encoded
        ).mean()
        l1_reconstruction_loss = (images - decoded).abs().mean()
        loss = l1_reconstruction_loss
        return loss, mse_loss, kl_div_loss, l1_reconstruction_loss


class Trainer:
    def __init__(
        self,
        config: VAEConfig,
        data_manager: DataManager,
        model: VAE,
        run_name: str,
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.model = model
        self.opt = t.optim.AdamW(model.parameters(), lr=config.start_lr)

        self.checkpoint = Checkpoint(
            run_name=run_name,
            model=model,
            optimizer=self.opt,
        )

    def evaluate(self) -> float:
        with t.no_grad():
            self.model.eval()
            losses = {"total": [], "mse": [], "kl": [], "l1": []}
            for x, _ in self.data_manager.val_loader:
                loss, mse_loss, kl_div_loss, l1_reconstruction_loss = self.model.loss(x)
                losses["total"].append(loss)
                losses["mse"].append(mse_loss)
                losses["kl"].append(kl_div_loss)
                losses["l1"].append(l1_reconstruction_loss)

            avg_losses = {k: t.tensor(v).mean() for k, v in losses.items()}
            for name, loss in avg_losses.items():
                print(f"Validation {name.upper()} loss: {loss:.4f}")

            self.model.train()
            assert isinstance(avg_losses["total"], t.Tensor)
            return avg_losses["total"].item()

    def train_step(
        self, x: Float[t.Tensor, "batch img_size"], y: Int[t.Tensor, "batch"]
    ):
        assert self.data_manager.train_loader is not None
        self.opt.zero_grad()
        loss, *_ = self.model.loss(x)
        loss.backward()
        self.opt.step()
        return loss

    def train(self):
        self.model.train()
        assert self.data_manager.train_loader is not None
        total_steps = 0
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1} of {self.config.epochs}")
            for param_group in self.opt.param_groups:
                param_group["lr"] = self.config.lr(epoch, self.config.start_lr)
            for i, (x, y) in tqdm(
                enumerate(self.data_manager.train_loader), desc="Training"
            ):
                training_batch_loss = self.train_step(x, y)
                total_steps += 1

                if total_steps % 100 == 0:
                    print(f"Training loss: {training_batch_loss:.4f}")
                    val_loss = self.evaluate()
                if total_steps % 500 == 0:
                    self.checkpoint.save_checkpoint(val_loss, epoch)


def train():
    config = VAEConfig()
    data_manager = DataManager()
    data_manager.prepare_data(
        ["mnist", "synthetic"], val_split=config.val_split, batch_size=config.batch_size
    )

    model = VAE(config)
    trainer = Trainer(config, data_manager, model, "vae_no_routing_l1_only")
    trainer.train()


if __name__ == "__main__":
    train()
