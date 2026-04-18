# src/models/cnn2d_model.py

import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """
    CNN 2D spatio-temporel.
    Entrée : (batch, 1, n_sensors, T_in)
    Sortie : (batch, T_out)
    """
    def __init__(self, n_sensors: int, T_in: int,
                 T_out: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool  = nn.MaxPool2d((2, 2))
        self.bn1   = nn.BatchNorm2d(32)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(p=dropout)

        h = (n_sensors // 2) // 2
        w = (T_in      // 2) // 2
        self.fc1 = nn.Linear(64 * h * w, 128)
        self.fc2 = nn.Linear(128, T_out)

    def forward(self, x):
        # x : (batch, 1, n_sensors, T_in)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.flatten(start_dim=1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)