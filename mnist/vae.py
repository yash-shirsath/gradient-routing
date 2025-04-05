from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict
import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from checkpoint import Checkpoint
import wandb
from data import DataManager
from jaxtyping import Float, Int
from torch.autograd import Variable
import time
# adapted from https://github.com/g-w1/gradient-routed-vae/blob/main/vae.py. Thanks!


@dataclass
class VAEConfig:
    image_features: int = 28 * 28
    hidden_size_1: int = 2048
    hidden_size_2: int = 512
    latent_size: int = 10

    val_split: float = 0.1
    batch_size: int = 128
    epochs: int = 100
    start_lr: float = 1e-3

    use_wandb: bool = True
    wandb_project: str = "vae"

    def lr(self, epoch: int, start_lr: float) -> float:
        return start_lr * 0.9**epoch


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.image_features, self.config.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size_1, self.config.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size_2, self.config.latent_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(self.config.latent_size, self.config.latent_size)
        self.logvar_head = nn.Linear(self.config.latent_size, self.config.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.config.latent_size, self.config.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size_2, self.config.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size_1, self.config.image_features),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

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
        z = eps * std + mu
        return self.relu(z)

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
        # if labels is not None:
        #     mask_one_hot = F.one_hot(
        #         labels, num_classes=self.config.latent_size
        #     ).float()  # type: ignore
        #     z = mask_one_hot * z + (1 - mask_one_hot) * z.detach()
        y = self.decode(z)
        return y.view(-1, 1, 28, 28), mu, logvar

    def loss(
        self,
        recon_x: Float[t.Tensor, "batch img_size"],
        x: Float[t.Tensor, "batch img_size"],
        mu: Float[t.Tensor, "batch latent_size"],
        logvar: Float[t.Tensor, "batch latent_size"],
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        MSE = ((recon_x - x) ** 2).mean()

        KLD = (
            mu**2 + logvar.exp() - logvar
        ).mean()  # todo not having a kld loss term. todo: look and see if encoding
        B = 0.3
        total = MSE * B + KLD
        return total, MSE, KLD


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
        self.epoch = 0
        self.total_steps = 0
        self.opt = t.optim.AdamW(model.parameters(), lr=config.start_lr)

        self.run_name = (
            f"{run_name}__{config.wandb_project}__{time.strftime('%y_%m_%d_%H_%M')}"
        )

        self.checkpoint = Checkpoint(
            run_name=self.run_name,
            model=model,
            optimizer=self.opt,
        )

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.run_name,
            )
            wandb.watch(self.model, log="all", log_freq=50)
            wandb.log(self.config.__dict__)

    def evaluate(self) -> float:
        with t.no_grad():
            self.model.eval()
            losses = defaultdict(list)
            for x, _ in self.data_manager.val_loader:
                recon_batch, mu, logvar = self.model(x)
                total, MSE, KLD = self.model.loss(
                    recon_batch,
                    x,
                    mu,
                    logvar,
                )
                losses["val_total"].append(total)
                losses["val_MSE"].append(MSE)
                losses["val_KLD"].append(KLD)
                losses["val_mu_mean"].append(mu.mean())
                losses["val_mu_std"].append(mu.std())
                losses["val_logvar_mean"].append(logvar.mean())
                losses["val_logvar_std"].append(logvar.std())

            avg_losses = {k: t.tensor(v).mean() for k, v in losses.items()}
            print(avg_losses)
            if self.config.use_wandb:
                wandb.log(avg_losses, step=self.total_steps)

            self.model.train()
            assert isinstance(avg_losses["val_total"], t.Tensor)
            return avg_losses["val_total"].item()

    def train_step(
        self, x: Float[t.Tensor, "batch img_size"], y: Int[t.Tensor, "batch"]
    ):
        assert self.data_manager.train_loader is not None
        self.opt.zero_grad()
        recon_batch, mu, logvar = self.model(x, y)
        loss, *_ = self.model.loss(recon_batch, x, mu, logvar)
        loss.backward()
        self.opt.step()
        return loss

    def train(self):
        self.model.train()
        assert self.data_manager.train_loader is not None

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            print(f"Epoch {epoch + 1} of {self.config.epochs}")
            for param_group in self.opt.param_groups:
                param_group["lr"] = self.config.lr(epoch, self.config.start_lr)
            for i, (x, y) in tqdm(
                enumerate(self.data_manager.train_loader),
                desc="Training",
                total=len(self.data_manager.train_loader),
            ):
                training_batch_loss = self.train_step(x, y)
                self.total_steps += 1

                if self.total_steps % 100 == 0:
                    print(f"Training loss: {training_batch_loss:.4f}")
                    val_loss = self.evaluate()
                if self.total_steps % 500 == 0:
                    self.checkpoint.save_checkpoint(val_loss, epoch)


def train():
    config = VAEConfig()
    data_manager = DataManager()
    mnist_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(mnist_data, batch_size=128, shuffle=True)  # type: ignore
    validation_data = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )
    validation_dataloader = DataLoader(validation_data, batch_size=128, shuffle=True)  # type: ignore
    data_manager.train_loader = dataloader
    data_manager.val_loader = validation_dataloader
    model = VAE(config)
    trainer = Trainer(config, data_manager, model, "vae_chris_no_norm")
    trainer.train()


if __name__ == "__main__":
    train()
