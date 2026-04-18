# src/models/lstm_model.py
# ============================================================
# Architecture LSTM multivariée
# Identique au GRU en termes d'interface pour garantir
# la comparabilité — seule la cellule interne diffère.
# ============================================================

import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM multivarié — Long Short-Term Memory.

    Par rapport au GRU, le LSTM ajoute une cellule mémoire
    (cell state) séparée de l'état caché, contrôlée par
    trois portes (input, forget, output) au lieu de deux.
    Cela lui confère une capacité théoriquement supérieure
    à modéliser les dépendances long terme.

    Entrée  : (batch, T_in, input_size)
              input_size = nombre de capteurs (3 par défaut)
    Sortie  : (batch, T_out)
    """

    def __init__(self,
                 input_size:  int,
                 T_out:       int,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 dropout:     float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            # dropout entre les couches (ignoré si num_layers=1)
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, T_out)

    def forward(self, x):
        # x : (batch, T_in, input_size)
        # LSTM retourne (output, (h_n, c_n))
        # On prend uniquement le dernier état caché
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (batch, T_out)