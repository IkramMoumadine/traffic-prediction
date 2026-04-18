# mains/main_gru.py
# ============================================================
# Pipeline GRU multivarié
# ============================================================

import os
import torch
import numpy as np

import config as C
from src.data.load_data       import load_pems07
from src.data.preprocessing   import (normalize_sensors_multi,
                                       create_windows_multivariate,
                                       temporal_split, denormalize)
from src.models.gru_model     import GRUModel
from src.training.trainer     import train_model
from src.utils.metrics        import plot_all_diagnostics
from src.utils.visualization  import plot_loss_curves

OUT = os.path.join(C.RESULTS_DIR, "gru")


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}\n  GRU — capteurs {C.SELECTED_SENSORS}\n{'='*60}")

    # 1. Chargement
    data = load_pems07(C.DATA_PATH)

    # 2. Normalisation multi-capteurs (fit sur train uniquement ✅)
    data_norm, scalers = normalize_sensors_multi(
        data, C.SELECTED_SENSORS, C.TRAIN_RATIO
    )

    # 3. Fenêtres multivariées
    X, y = create_windows_multivariate(
        data_norm, C.SELECTED_SENSORS, C.SENSOR_IDX, C.T_IN, C.T_OUT
    )

    # 4. Split
    splits = temporal_split(X, y, C.TRAIN_RATIO, C.VAL_RATIO)

    # 5. Modèle
    n_sensors = len(C.SELECTED_SENSORS)
    model = GRUModel(input_size=n_sensors, T_out=C.T_OUT,
                     hidden_size=128, num_layers=2, dropout=C.DROPOUT)

    # 6. Entraînement
    model, history, y_pred_norm, y_test_norm = train_model(
        model, splits,
        checkpoint_path = f"{OUT}/best_model.pt",
        epochs     = C.EPOCHS,
        batch_size = C.BATCH_SIZE,
        lr         = C.LR,
        patience   = C.PATIENCE,
        seed       = C.SEED,
    )

    # 7. Loss curve
    plot_loss_curves(history, "GRU", f"{OUT}/loss.png")

    # 8. Dénormalisation
    sc = scalers[C.SENSOR_IDX]
    y_true_d = denormalize(y_test_norm, sc)
    y_pred_d = denormalize(y_pred_norm, sc)

    # 9. Diagnostics
    plot_all_diagnostics(
        y_true_denorm  = y_true_d,
        y_pred_denorm  = y_pred_d,
        model_name     = "GRU",
        save_dir       = OUT,
        steps_per_hour = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ GRU terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()