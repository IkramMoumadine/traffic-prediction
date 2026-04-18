# src/models/gru_model.py

import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU multivarié.
    Entrée : (batch, T_in, n_sensors)
    Sortie : (batch, T_out)
    """
    def __init__(self, input_size: int, T_out: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, T_out)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])   # dernier timestep