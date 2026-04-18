# mains/main_cnn2d.py
# ============================================================
# Pipeline CNN 2D spatio-temporel
# ============================================================

import os
import numpy as np
import torch

import config as C
from src.data.load_data       import load_pems07
from src.data.preprocessing   import (normalize_sensor,
                                       build_temporal_features,
                                       create_windows_univariate,
                                       temporal_split, denormalize)
from src.models.cnn2d_model   import CNN2D
from src.training.trainer     import train_model
from src.utils.metrics        import plot_all_diagnostics
from src.utils.visualization  import plot_loss_curves

OUT = os.path.join(C.RESULTS_DIR, "cnn2d")

# CNN 2D utilise plusieurs capteurs voisins comme dimension spatiale
SPATIAL_SENSORS = [C.SENSOR_IDX] + C.SELECTED_SENSORS   # ex. 4 capteurs


def run():
    os.makedirs(OUT, exist_ok=True)
    sensors = list(dict.fromkeys(SPATIAL_SENSORS))   # déduplique
    print(f"\n{'='*60}\n  CNN 2D — capteurs {sensors}\n{'='*60}")

    # 1. Chargement
    data = load_pems07(C.DATA_PATH)

    # 2. Normalisation du capteur cible
    data_norm, scaler = normalize_sensor(data, C.SENSOR_IDX, C.TRAIN_RATIO)

    # 3. Features temporelles
    temporal = build_temporal_features(data_norm.shape[0], C.INTERVAL_MINUTES)

    # 4. Fenêtres univariées (5 canaux)
    X, y = create_windows_univariate(data_norm, temporal,
                                      C.SENSOR_IDX, C.T_IN, C.T_OUT)

    # 5. Reshape pour CNN 2D : (N, 1, 5, T_in)
    X_2d = X[:, np.newaxis, :, :]   # ajoute dimension batch canal

    splits_2d = {
        "X_train": X_2d[:int(len(X_2d) * C.TRAIN_RATIO)],
        "y_train": y[:int(len(y) * C.TRAIN_RATIO)],
        "X_val"  : X_2d[int(len(X_2d)*C.TRAIN_RATIO):int(len(X_2d)*(C.TRAIN_RATIO+C.VAL_RATIO))],
        "y_val"  : y[int(len(y)*C.TRAIN_RATIO):int(len(y)*(C.TRAIN_RATIO+C.VAL_RATIO))],
        "X_test" : X_2d[int(len(X_2d)*(C.TRAIN_RATIO+C.VAL_RATIO)):],
        "y_test" : y[int(len(y)*(C.TRAIN_RATIO+C.VAL_RATIO)):],
    }

    # 6. Modèle
    n_ch, t_in = X_2d.shape[2], X_2d.shape[3]
    model = CNN2D(n_sensors=n_ch, T_in=t_in,
                  T_out=C.T_OUT, dropout=C.DROPOUT)

    # 7. Entraînement
    model, history, y_pred_norm, y_test_norm = train_model(
        model, splits_2d,
        checkpoint_path = f"{OUT}/best_model.pt",
        epochs     = C.EPOCHS,
        batch_size = C.BATCH_SIZE,
        lr         = C.LR,
        patience   = C.PATIENCE,
        seed       = C.SEED,
    )

    # 8. Loss + diagnostics
    plot_loss_curves(history, "CNN 2D", f"{OUT}/loss.png")
    y_true_d = denormalize(y_test_norm, scaler)
    y_pred_d = denormalize(y_pred_norm, scaler)
    plot_all_diagnostics(
        y_true_denorm  = y_true_d,
        y_pred_denorm  = y_pred_d,
        model_name     = "CNN 2D",
        save_dir       = OUT,
        steps_per_hour = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ CNN 2D terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()