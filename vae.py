from dataclasses import dataclass

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
    def __init__(self):
        super(VAE, self).__init__()
        self.hidden_size = 10

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.hidden_size)
        self.fc22 = nn.Linear(400, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 400)
        self.fc4 = nn.Linear(400, 784)
        self.reconstruction_function = nn.MSELoss(size_average=False)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = t.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x, labels=None):
        x = x.reshape(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        if labels is not None:
            mask_one_hot = F.one_hot(labels, num_classes=self.hidden_size).float()  # type: ignore
            z = mask_one_hot * z + (1 - mask_one_hot) * z.detach()
        y = self.decode(z)
        return y.reshape(-1, 1, 28, 28), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        BCE = self.reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = t.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return 0.3 * BCE + KLD


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
            losses = {"total": []}
            for x, _ in self.data_manager.val_loader:
                recon_batch, mu, logvar = self.model(x)
                loss = self.model.loss(recon_batch, x, mu, logvar)
                losses["total"].append(loss)

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
        recon_batch, mu, logvar = self.model(x, y)
        loss = self.model.loss(recon_batch, x, mu, logvar)
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

    model = VAE()
    trainer = Trainer(config, data_manager, model, "vae_no_routing_l1_only")
    trainer.train()


if __name__ == "__main__":
    train()
