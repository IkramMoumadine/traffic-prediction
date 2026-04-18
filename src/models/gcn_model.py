# src/models/gcn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Couche de convolution sur graphe (GCN simple)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        """
        x   : (batch, N, in_features)
        adj : (N, N) — matrice d'adjacence normalisée
        """
        return F.relu(self.W(torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)))


class GCN(nn.Module):
    """
    Graph Convolutional Network pour prédiction de trafic.
    Entrée  : x (batch, N, T_in), adj (N, N)
    Sortie  : (batch, T_out)
    """
    def __init__(self, n_nodes: int, T_in: int, T_out: int,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.gc1  = GraphConvLayer(T_in,   hidden)
        self.gc2  = GraphConvLayer(hidden, hidden)
        self.drop = nn.Dropout(p=dropout)
        self.fc   = nn.Linear(n_nodes * hidden, T_out)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.drop(x)
        x = self.gc2(x, adj)
        x = x.flatten(start_dim=1)
        return self.fc(x)