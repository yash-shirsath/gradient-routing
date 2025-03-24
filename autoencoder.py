import torch as t
from jaxtyping import Float
from torch import nn
from tqdm import tqdm

from checkpoint import Checkpoint
from data import DataManager


class AutoencoderConfig:
    hidden_size: int = 128
    encoder_size_1: int = 2048
    encoder_size_2: int = 256

    val_split: float = 0.1
    batch_size: int = 1024
    lr: float = 1e-3
    epochs: int = 10


class Autoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super(Autoencoder, self).__init__()
        self.config = config
        assert self.config.hidden_size % 2 == 0

        size_1 = self.config.encoder_size_1
        size_2 = self.config.encoder_size_2

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, self.config.hidden_size, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, 28 * 28),
        )

    def encode(self, x):
        batch_size = x.shape[0]
        encoding = self.encoder(x.reshape((batch_size, 784)))
        return encoding

    def forward(
        self, x: Float[t.Tensor, "batch img_size"]
    ) -> Float[t.Tensor, "batch img_size"]:
        batch_size = x.shape[0]
        encoding = self.encode(x)
        out = self.decoder(encoding).reshape((batch_size, 1, 28, 28))
        return out

    def loss(
        self,
        pred: Float[t.Tensor, "batch img_size"],
        x: Float[t.Tensor, "batch img_size"],
    ):
        l1_reconstruction_loss = (pred - x).abs().mean()
        return l1_reconstruction_loss


class Trainer:
    def __init__(
        self,
        config: AutoencoderConfig,
        data_manager: DataManager,
        model: Autoencoder,
        run_name: str,
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.model = model
        self.opt = t.optim.AdamW(model.parameters(), lr=config.lr)
        self.checkpoint = Checkpoint(
            run_name=run_name,
            model=model,
            optimizer=self.opt,
        )

    def evaluate(self) -> float:
        with t.no_grad():
            self.model.eval()
            loss = 0
            for x, _ in self.data_manager.val_loader:
                pred = self.model(x)
                loss += self.model.loss(pred, x)
            avg_loss = loss / len(self.data_manager.val_loader)
            print(f"Validation loss: {avg_loss:.4f}")
            self.model.train()
            assert isinstance(avg_loss, t.Tensor)
            return avg_loss.item()

    def train_step(self, x: Float[t.Tensor, "batch img_size"]):
        assert self.data_manager.train_loader is not None
        self.opt.zero_grad()
        pred = self.model(x)
        loss = self.model.loss(pred, x)
        loss.backward()
        self.opt.step()
        return loss

    def train(self):
        self.model.train()
        assert self.data_manager.train_loader is not None
        total_steps = 0
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1} of {self.config.epochs}")
            for i, (x, y) in tqdm(
                enumerate(self.data_manager.train_loader), desc="Training"
            ):
                self.train_step(x)
                total_steps += 1

            if total_steps % 100 == 0:
                val_loss = self.evaluate()
            if total_steps % 1000 == 0:
                self.checkpoint.save_checkpoint(val_loss, epoch)


def train():
    config = AutoencoderConfig()
    data_manager = DataManager()
    data_manager.prepare_data(
        ["mnist", "synthetic"], val_split=config.val_split, batch_size=config.batch_size
    )

    model = Autoencoder(config)
    trainer = Trainer(
        config, data_manager, model, "autoencoder_reconstruction_loss_only"
    )
    trainer.train()


if __name__ == "__main__":
    train()
