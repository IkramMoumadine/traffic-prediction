# src/models/cnn1d_model.py

import torch
import torch.nn as nn


class CNN1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, padding="same")
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNN1D(nn.Module):
    """
    CNN 1D multi-canaux.
    Entrée : (batch, 5, T_in)
    Sortie : (batch, T_out)
    """
    def __init__(self, T_in: int, T_out: int,
                 n_channels: int = 5, dropout: float = 0.3):
        super().__init__()
        self.block1 = CNN1DBlock(n_channels, 32,  kernel_size=3, pool=True)
        self.block2 = CNN1DBlock(32,         64,  kernel_size=3, pool=True)
        self.block3 = CNN1DBlock(64,         128, kernel_size=3, pool=False)

        fc_in = 128 * (T_in // 4)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc1     = nn.Linear(fc_in, 128)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, T_out)
        self.relu    = nn.ReLU()

    def forward(self, x):
        last_val = x[:, 0, -1:]              # dernier pas connu (batch, 1)
        x_in = x.clone()
        x_in[:, 0, :] = x[:, 0, :] - last_val  # centre le signal → variations
        x_in = self.block1(x_in)
        x_in = self.block2(x_in)
        x_in = self.block3(x_in)
        x_in = self.flatten(x_in)
        x_in = self.dropout(x_in)
        x_in = self.relu(self.fc1(x_in))
        x_in = self.relu(self.fc2(x_in))
        delta = self.fc3(x_in)               # prédit le delta
        return last_val + delta              # valeur absolue reconstituée