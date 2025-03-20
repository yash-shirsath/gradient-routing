import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torchvision import datasets, transforms

from data import DataManager


class Autoencoder(nn.Module):
    def __init__(self, hidden_size):
        super(Autoencoder, self).__init__()
        self.hidden_size = hidden_size
        assert self.hidden_size % 2 == 0

        size_1 = 2048
        size_2 = 256

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, self.hidden_size, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, size_2),
            nn.ReLU(True),
            nn.Linear(size_2, size_1),
            nn.ReLU(True),
            nn.Linear(size_1, 28 * 28),
        )

    def encode(self, x):
        batch_size = x.shape[0]
        encoding = self.encoder(x.reshape((batch_size, 784)))
        return encoding

    def forward(self, x):
        batch_size = x.shape[0]
        encoding = self.encode(x)
        out = self.decoder(encoding).reshape((batch_size, 1, 28, 28))
        return out, encoding


def train():
    data_manager = DataManager()
    data_manager.prepare_data(["mnist"])
    trainloader = data_manager.train_loader
    assert trainloader is not None
    num_epochs = 100
    model = Autoencoder(hidden_size=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    for epoch in tqdm(range(num_epochs)):
        for i, (x, y) in tqdm(enumerate(trainloader), desc="Training"):
            optimizer.zero_grad()
            pred, encodings = model(x)
            l1_reconstruction_loss = (pred - x).abs().mean()
            loss = l1_reconstruction_loss
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")


if __name__ == "__main__":
    train()
