# mains/main_cnn1d.py
# ============================================================
# Pipeline CNN 1D — utilise config.py et modules src/
# ============================================================

import os
import torch
import numpy as np

import config as C
from src.data.load_data        import load_pems07
from src.data.preprocessing    import (normalize_sensor,
                                        build_temporal_features,
                                        create_windows_univariate,
                                        temporal_split, denormalize)
from src.models.cnn1d_model    import CNN1D
from src.training.trainer      import train_model
from src.utils.metrics         import plot_all_diagnostics
from src.utils.visualization   import plot_loss_curves, plot_sensor_3days

OUT = os.path.join(C.RESULTS_DIR, "cnn1d")


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}\n  CNN 1D — Capteur {C.SENSOR_IDX}\n{'='*60}")

    # 1. Chargement
    data = load_pems07(C.DATA_PATH)

    # 2. Signal brut 3 jours
    plot_sensor_3days(data, C.SENSOR_IDX, C.INTERVAL_MINUTES,
                      f"{OUT}/00_signal_3jours.png")

    # 3. Normalisation (fit sur train uniquement)
    data_norm, scaler = normalize_sensor(data, C.SENSOR_IDX, C.TRAIN_RATIO)

    # 4. Features temporelles
    temporal = build_temporal_features(data_norm.shape[0], C.INTERVAL_MINUTES)

    # 5. Fenêtres glissantes
    X, y = create_windows_univariate(data_norm, temporal,
                                      C.SENSOR_IDX, C.T_IN, C.T_OUT)

    # 6. Split
    splits = temporal_split(X, y, C.TRAIN_RATIO, C.VAL_RATIO)

    # 7. Modèle + entraînement
    model = CNN1D(T_in=C.T_IN, T_out=C.T_OUT,
                  n_channels=5, dropout=C.DROPOUT)

    model, history, y_pred_norm, y_test_norm = train_model(
        model, splits,
        checkpoint_path = f"{OUT}/best_model.pt",
        epochs     = C.EPOCHS,
        batch_size = C.BATCH_SIZE,
        lr         = C.LR,
        patience   = C.PATIENCE,
        seed       = C.SEED,
    )

    # 8. Loss curve
    plot_loss_curves(history, "CNN 1D", f"{OUT}/loss.png")

    # 9. Dénormalisation
    y_true_d = denormalize(y_test_norm, scaler)
    y_pred_d = denormalize(y_pred_norm, scaler)

    # 10. Diagnostics complets
    plot_all_diagnostics(
        y_true_denorm  = y_true_d,
        y_pred_denorm  = y_pred_d,
        model_name     = "CNN 1D",
        save_dir       = OUT,
        steps_per_hour = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ CNN 1D terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()