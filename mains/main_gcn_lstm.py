# mains/main_gcn_lstm.py
# ============================================================
# Pipeline GCN + LSTM — modèle hybride spatio-temporel
#
# Architecture :
#   GCN (2 couches) → capture les dépendances spatiales
#   LSTM (2 couches) → capture les dépendances temporelles
#
# Par rapport au GCN seul du projet :
#   - Même interface (train_model, plot_all_diagnostics, config)
#   - Utilise normalize_sensors_multi (comme GRU/LSTM)
#     pour exploiter les 3 capteurs voisins
#   - Matrice d'adjacence normalisée (D^-1/2 A D^-1/2)
# ============================================================

import os
import torch
import numpy as np

import config as C
from src.data.load_data       import load_pems07
from src.data.preprocessing   import (normalize_sensors_multi,
                                       temporal_split, denormalize)
from src.models.gcn_lstm_model import GCN_LSTM, build_adjacency_normalized
from src.training.trainer      import train_model
from src.utils.metrics         import plot_all_diagnostics
from src.utils.visualization   import plot_loss_curves

OUT = os.path.join(C.RESULTS_DIR, "gcn_lstm")

GCN_SENSORS = C.SELECTED_SENSORS   # [13, 537, 864]
N_NODES     = len(GCN_SENSORS)


def make_windows_gcn_lstm(data_norm: np.ndarray,
                           sensor_indices: list,
                           target_idx: int,
                           T_in: int,
                           T_out: int) -> tuple:
    """
    Fenêtres glissantes pour GCN+LSTM.

    Retour
    ------
    X : (N, n_nodes, T_in)  — vue spatiale par fenêtre
    y : (N, T_out)          — cible
    """
    X, y = [], []
    for i in range(len(data_norm) - T_in - T_out + 1):
        # (n_nodes, T_in) : chaque nœud = une ligne, chaque colonne = un pas
        X.append(data_norm[i : i + T_in, sensor_indices].T)
        y.append(data_norm[i + T_in : i + T_in + T_out, target_idx])

    X = np.array(X, dtype=np.float32)   # (N, n_nodes, T_in)
    y = np.array(y, dtype=np.float32)   # (N, T_out)

    print(f"  Fenêtres GCN+LSTM : X={X.shape}  y={y.shape}")
    return X, y


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}\n  GCN+LSTM — {N_NODES} nœuds\n{'='*60}")

    # ── 1. Chargement ─────────────────────────────────────────
    data = load_pems07(C.DATA_PATH)

    # ── 2. Normalisation multi-capteurs (fit sur train seul ✅) ─
    data_norm, scalers = normalize_sensors_multi(
        data, GCN_SENSORS, C.TRAIN_RATIO
    )

    # ── 3. Fenêtres (n_nodes, T_in) ──────────────────────────
    X, y = make_windows_gcn_lstm(
        data_norm, GCN_SENSORS, C.SENSOR_IDX, C.T_IN, C.T_OUT
    )

    # ── 4. Split chronologique ────────────────────────────────
    splits = temporal_split(X, y, C.TRAIN_RATIO, C.VAL_RATIO)

    # ── 5. Matrice d'adjacence normalisée ─────────────────────
    adj = build_adjacency_normalized(N_NODES)

    # ── 6. Modèle GCN+LSTM ────────────────────────────────────
    # Wrapper pour injecter adj dans le forward (compatible train_model)
    class GCNLSTMWrapper(torch.nn.Module):
        def __init__(self, model, adj):
            super().__init__()
            self.model = model
            self.register_buffer("adj", adj)

        def forward(self, x):
            return self.model(x, self.adj)

    base_model = GCN_LSTM(
        n_nodes     = N_NODES,
        T_in        = C.T_IN,
        T_out       = C.T_OUT,
        gcn_hidden  = 64,
        lstm_hidden = 128,
        num_layers  = 2,
        dropout     = C.DROPOUT,
    )
    model = GCNLSTMWrapper(base_model, adj)

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"\n  Architecture GCN+LSTM :")
    print(f"    n_nodes     = {N_NODES} capteurs {GCN_SENSORS}")
    print(f"    gcn_hidden  = 64")
    print(f"    lstm_hidden = 128")
    print(f"    num_layers  = 2")
    print(f"    dropout     = {C.DROPOUT}")
    print(f"    Paramètres  = {n_params:,}")

    # ── 7. Entraînement (boucle unifiée) ──────────────────────
    model, history, y_pred_norm, y_test_norm = train_model(
        model, splits,
        checkpoint_path = f"{OUT}/best_model.pt",
        epochs          = C.EPOCHS,
        batch_size      = C.BATCH_SIZE,
        lr              = C.LR,
        patience        = C.PATIENCE,
        seed            = C.SEED,
    )

    # ── 8. Courbe de loss ─────────────────────────────────────
    plot_loss_curves(history, "GCN+LSTM", f"{OUT}/loss.png")

    # ── 9. Dénormalisation ─────────────────────────────────────
    sc       = scalers[C.SENSOR_IDX]
    y_true_d = denormalize(y_test_norm, sc)
    y_pred_d = denormalize(y_pred_norm, sc)


    # ── 10. Diagnostics complets ──────────────────────────────
    plot_all_diagnostics(
        y_true_denorm  = y_true_d,
        y_pred_denorm  = y_pred_d,
        model_name     = "GCN+LSTM",
        save_dir       = OUT,
        steps_per_hour = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ GCN+LSTM terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()