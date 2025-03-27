from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from checkpoint import Checkpoint
from data import DataManager
from jaxtyping import Float, Int
from torch.autograd import Variable

# adapted from https://github.com/g-w1/gradient-routed-vae/blob/main/vae.py. Thanks!


@dataclass
class VAEConfig:
    image_features: int = 28 * 28
    hidden_sizes: int = 400
    latent_size: int = 10

    val_split: float = 0.1
    batch_size: int = 512
    epochs: int = 100
    start_lr: float = 1e-3

    def lr(self, epoch: int, start_lr: float) -> float:
        return start_lr * 0.9**epoch


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.image_features, self.config.hidden_sizes),
            nn.ReLU(),
            nn.Linear(self.config.hidden_sizes, self.config.latent_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(self.config.latent_size, self.config.latent_size)
        self.logvar_head = nn.Linear(self.config.latent_size, self.config.latent_size)

        # Initialize logvar to -2.0
        self.logvar_head.bias.data.fill_(-2.0)
        self.decoder = nn.Sequential(
            nn.Linear(self.config.latent_size, self.config.hidden_sizes),
            nn.ReLU(),
            nn.Linear(self.config.hidden_sizes, self.config.image_features),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

    def reparametrize(
        self,
        mu: Float[t.Tensor, "batch latent_size"],
        logvar: Float[t.Tensor, "batch latent_size"],
    ) -> Float[t.Tensor, "batch latent_size"]:
        std = (logvar / 2).exp()
        eps = t.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)

    def forward(
        self,
        x: Float[t.Tensor, "batch img_size"],
        labels: Optional[Int[t.Tensor, "batch"]] = None,
    ) -> Tuple[
        Float[t.Tensor, "batch img_size"],
        Float[t.Tensor, "batch latent_size"],
        Float[t.Tensor, "batch latent_size"],
    ]:
        x = x.view(-1, self.config.image_features)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        if labels is not None:
            mask_one_hot = F.one_hot(
                labels, num_classes=self.config.latent_size
            ).float()  # type: ignore
            z = mask_one_hot * z + (1 - mask_one_hot) * z.detach()
        y = self.decode(z)
        return y.view(-1, 1, 28, 28), mu, logvar

    def loss(
        self,
        recon_x: Float[t.Tensor, "batch img_size"],
        x: Float[t.Tensor, "batch img_size"],
        mu: Float[t.Tensor, "batch latent_size"],
        logvar: Float[t.Tensor, "batch latent_size"],
        epoch: int = 0,
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor, float]:
        MSE = (recon_x - x).pow(2).mean()
        KLD = -0.5 * t.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Increase beta from 1000 to 5000 over first 20 epochs, then stay at 5000
        beta = 1000 + min(epoch / 20, 1.0) * 4000
        total = MSE + beta * KLD
        return total, MSE, KLD, beta


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
        self.epoch = 0

    def evaluate(self) -> float:
        with t.no_grad():
            self.model.eval()
            losses = {
                "total": [],
                "MSE": [],
                "KLD": [],
                "beta": [],
                "mu_mean": [],
                "logvar_mean": [],
                "mu_std": [],
                "logvar_std": [],
            }
            for x, _ in self.data_manager.val_loader:
                recon_batch, mu, logvar = self.model(x)
                total, MSE, KLD, beta = self.model.loss(
                    recon_batch, x, mu, logvar, epoch=self.epoch
                )
                losses["total"].append(total)
                losses["MSE"].append(MSE)
                losses["KLD"].append(KLD)
                losses["beta"].append(beta)
                losses["mu_mean"].append(mu.mean())
                losses["logvar_mean"].append(logvar.mean())
                losses["mu_std"].append(mu.std())
                losses["logvar_std"].append(logvar.std())

            avg_losses = {k: t.tensor(v).mean() for k, v in losses.items()}
            for name, loss in avg_losses.items():
                print(f"Validation {name.upper()}: {loss:.4f}")

            self.model.train()
            assert isinstance(avg_losses["total"], t.Tensor)
            return avg_losses["total"].item()

    def train_step(
        self, x: Float[t.Tensor, "batch img_size"], y: Int[t.Tensor, "batch"]
    ):
        assert self.data_manager.train_loader is not None
        self.opt.zero_grad()
        recon_batch, mu, logvar = self.model(x, y)
        loss, *_ = self.model.loss(recon_batch, x, mu, logvar, epoch=self.epoch)
        loss.backward()
        self.opt.step()
        return loss

    def train(self):
        self.model.train()
        assert self.data_manager.train_loader is not None
        total_steps = 0
        for epoch in range(self.config.epochs):
            self.epoch = epoch
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
    trainer = Trainer(config, data_manager, model, "vae_initialize_logvar")
    trainer.train()


if __name__ == "__main__":
    train()
