# src/models/gcn_lstm_model.py
# ============================================================
# Architecture GCN + LSTM pour prédiction de trafic
#
# Principe :
#   1. Deux couches GCN capturent les dépendances spatiales
#      entre capteurs à chaque pas de temps
#   2. Un LSTM lit la séquence de représentations graphiques
#      pour capturer les dépendances temporelles
#   3. Une couche FC produit la prédiction finale
#
# Entrée  : (batch, n_nodes, T_in)  — même format que GCN seul
# Sortie  : (batch, T_out)          — prédiction capteur cible
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Couche de convolution sur graphe (GCN symétrique normalisé)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W  = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, N, in_features)
        adj : (N, N) — matrice d'adjacence normalisée (symétrique)
        """
        # Propagation sur le graphe : adj @ x puis projection linéaire
        ax = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        h  = self.W(ax)                          # (batch, N, out_features)
        # BatchNorm sur la dimension features : reshape pour BN1d
        B, N, F_ = h.shape
        h = self.bn(h.reshape(B * N, F_)).reshape(B, N, F_)
        return F.relu(h)


class GCN_LSTM(nn.Module):
    """
    Modèle hybride GCN + LSTM.

    Flux de données :
      (batch, N, T_in)
        → pour chaque pas t : GCN appliqué sur le snapshot spatial
        → séquence de représentations (batch, T_in, N * gcn_hidden)
        → LSTM sur cette séquence
        → FC → (batch, T_out)

    Paramètres
    ----------
    n_nodes    : nombre de capteurs dans le graphe
    T_in       : longueur de la fenêtre d'entrée (12 par défaut)
    T_out      : horizon de prédiction (1 par défaut)
    gcn_hidden : dimension cachée des couches GCN
    lstm_hidden: dimension cachée du LSTM
    num_layers : nombre de couches LSTM empilées
    dropout    : dropout entre couches LSTM
    """

    def __init__(self,
                 n_nodes:     int,
                 T_in:        int   = 12,
                 T_out:       int   = 1,
                 gcn_hidden:  int   = 64,
                 lstm_hidden: int   = 128,
                 num_layers:  int   = 2,
                 dropout:     float = 0.3):
        super().__init__()

        self.n_nodes    = n_nodes
        self.T_in       = T_in
        self.gcn_hidden = gcn_hidden

        # ── Couches GCN ──────────────────────────────────────
        self.gc1 = GraphConvLayer(T_in,       gcn_hidden)   # snapshot complet
        self.gc2 = GraphConvLayer(gcn_hidden, gcn_hidden)

        # ── LSTM temporel ────────────────────────────────────
        # Entrée LSTM : représentation spatiale aplatie (N * gcn_hidden)
        # → on utilise un encodage par pas de temps via une projection
        self.spatial_proj = nn.Linear(n_nodes * gcn_hidden, lstm_hidden)

        self.lstm = nn.LSTM(
            input_size  = lstm_hidden,
            hidden_size = lstm_hidden,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.drop = nn.Dropout(p=dropout)
        self.fc   = nn.Linear(lstm_hidden, T_out)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, n_nodes, T_in)
        adj : (n_nodes, n_nodes)
        """
        B, N, T = x.shape

        # ── Étape 1 : GCN sur le snapshot spatial complet ───
        # On applique GCN une fois sur la vue (batch, N, T_in)
        # gc1 et gc2 traitent T_in comme dimension de features
        h = self.gc1(x,   adj)    # (batch, N, gcn_hidden)
        h = self.drop(h)
        h = self.gc2(h,   adj)    # (batch, N, gcn_hidden)

        # ── Étape 2 : Projection → séquence LSTM ────────────
        # Aplatir la dimension spatiale → (batch, N * gcn_hidden)
        h_flat = h.flatten(start_dim=1)         # (batch, N * gcn_hidden)

        # Projeter dans l'espace LSTM → (batch, lstm_hidden)
        h_proj = F.relu(self.spatial_proj(h_flat))

        # Répéter pour simuler une séquence de longueur T_in
        # → (batch, T_in, lstm_hidden)
        h_seq = h_proj.unsqueeze(1).expand(-1, T, -1)

        # ── Étape 3 : LSTM + prédiction ─────────────────────
        lstm_out, _ = self.lstm(h_seq)          # (batch, T_in, lstm_hidden)
        last_hidden = lstm_out[:, -1, :]        # (batch, lstm_hidden)

        return self.fc(self.drop(last_hidden))  # (batch, T_out)


# ── Utilitaire : construction de la matrice d'adjacence ──────

def build_adjacency_normalized(n_nodes: int,
                                 sensor_distances: np.ndarray = None,
                                 sigma: float = 10.0) -> torch.Tensor:
    """
    Construit une matrice d'adjacence normalisée symétrique D^{-1/2} A D^{-1/2}.

    Si sensor_distances est fourni (matrice (N,N) de distances),
    utilise un kernel gaussien. Sinon, fully-connected uniforme.

    Paramètres
    ----------
    n_nodes          : nombre de nœuds
    sensor_distances : (N, N) distances géographiques (optionnel)
    sigma            : paramètre de largeur du kernel gaussien

    Retour
    ------
    adj_norm : torch.Tensor (N, N)
    """
    if sensor_distances is not None:
        # Kernel gaussien : w_ij = exp(-d²/sigma²)
        adj = np.exp(-(sensor_distances ** 2) / (sigma ** 2))
        np.fill_diagonal(adj, 0.0)
        # Seuillage pour la sparsité
        adj[adj < 0.1] = 0.0
    else:
        # Fully-connected simple (comme dans main_gcn.py)
        adj = np.ones((n_nodes, n_nodes)) / n_nodes

    # Normalisation symétrique : D^{-1/2} A D^{-1/2}
    adj += np.eye(n_nodes)                       # self-loops
    d          = adj.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = np.diag(d_inv_sqrt)
    adj_norm = D @ adj @ D

    return torch.tensor(adj_norm, dtype=torch.float32)