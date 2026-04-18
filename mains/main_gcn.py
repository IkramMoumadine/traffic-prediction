# mains/main_gcn.py
# ============================================================
# Pipeline GCN — Graph Convolutional Network
# ============================================================

import os
import numpy as np
import torch

import config as C
from src.data.load_data       import load_pems07
from src.data.preprocessing   import (normalize_sensor, temporal_split,
                                       denormalize)
from src.models.gcn_model     import GCN
from src.training.trainer     import train_model
from src.utils.metrics        import plot_all_diagnostics
from src.utils.visualization  import plot_loss_curves

OUT = os.path.join(C.RESULTS_DIR, "gcn")

# Sous-ensemble de capteurs pour le graphe
GCN_SENSORS = C.SELECTED_SENSORS   # [13, 537, 864]
N_NODES = len(GCN_SENSORS)


def build_adjacency(n: int) -> torch.Tensor:
    """
    Matrice d'adjacence normalisée simple (fully connected).
    À remplacer par une vraie matrice de distance si disponible.
    """
    adj = torch.ones(n, n) / n
    return adj


def make_windows_gcn(data_norm, sensor_indices, target_idx,
                     T_in, T_out):
    """Fenêtres (N, n_nodes, T_in) pour GCN."""
    X, y = [], []
    for i in range(len(data_norm) - T_in - T_out + 1):
        X.append(data_norm[i:i+T_in, sensor_indices].T)   # (n_nodes, T_in)
        y.append(data_norm[i+T_in:i+T_in+T_out, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}\n  GCN — {N_NODES} nœuds\n{'='*60}")

    data = load_pems07(C.DATA_PATH)
    data_norm, scaler = normalize_sensor(data, C.SENSOR_IDX, C.TRAIN_RATIO)

    X, y = make_windows_gcn(data_norm, GCN_SENSORS,
                              C.SENSOR_IDX, C.T_IN, C.T_OUT)
    splits = temporal_split(X, y, C.TRAIN_RATIO, C.VAL_RATIO)

    adj   = build_adjacency(N_NODES)
    model = GCN(n_nodes=N_NODES, T_in=C.T_IN,
                T_out=C.T_OUT, hidden=64, dropout=C.DROPOUT)

    # Wrapper pour passer adj au modèle dans trainer
    class GCNWrapper(torch.nn.Module):
        def __init__(self, gcn, adj):
            super().__init__()
            self.gcn = gcn
            self.register_buffer("adj", adj)

        def forward(self, x):
            return self.gcn(x, self.adj)

    wrapped = GCNWrapper(model, adj)

    model_out, history, y_pred_norm, y_test_norm = train_model(
        wrapped, splits,
        checkpoint_path = f"{OUT}/best_model.pt",
        epochs     = C.EPOCHS,
        batch_size = C.BATCH_SIZE,
        lr         = C.LR,
        patience   = C.PATIENCE,
        seed       = C.SEED,
    )

    plot_loss_curves(history, "GCN", f"{OUT}/loss.png")
    y_true_d = denormalize(y_test_norm, scaler)
    y_pred_d = denormalize(y_pred_norm, scaler)
    plot_all_diagnostics(
        y_true_denorm  = y_true_d,
        y_pred_denorm  = y_pred_d,
        model_name     = "GCN",
        save_dir       = OUT,
        steps_per_hour = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ GCN terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()