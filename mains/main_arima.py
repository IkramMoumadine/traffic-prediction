# mains/main_arima.py
# ============================================================
# Pipeline ARIMA — modèle statistique de référence
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import config as C
from src.data.load_data      import load_pems07
from src.data.preprocessing  import normalize_sensor, denormalize
from src.models.arima_model  import fit_arima, predict_arima
from src.utils.metrics       import plot_all_diagnostics, compute_metrics

OUT = os.path.join(C.RESULTS_DIR, "arima")


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}\n  ARIMA — Capteur {C.SENSOR_IDX}\n{'='*60}")

    # 1. Chargement
    data = load_pems07(C.DATA_PATH)

    # 2. Normalisation
    data_norm, scaler = normalize_sensor(data, C.SENSOR_IDX, C.TRAIN_RATIO)
    signal = data_norm[:, C.SENSOR_IDX]

    # 3. Split chronologique
    n       = len(signal)
    i_train = int(n * C.TRAIN_RATIO)
    i_val   = int(n * (C.TRAIN_RATIO + C.VAL_RATIO))

    train_series = signal[:i_train]
    test_series  = signal[i_val:]

    print(f"  Train : {len(train_series)} | Test : {len(test_series)}")

    # 4. Fit ARIMA
    fitted = fit_arima(train_series, order=(12, 1, 6))

    # 5. Prédictions sur le test (pas à pas)
    print("  Prédictions en cours (peut être long)...")
    preds_norm = predict_arima(fitted, n_steps=len(test_series))

    # 6. Dénormalisation
    y_true_d = denormalize(test_series, scaler)
    y_pred_d = denormalize(preds_norm,  scaler)

    # 7. Diagnostics
    plot_all_diagnostics(
        y_true_denorm  = y_true_d,
        y_pred_denorm  = y_pred_d,
        model_name     = "ARIMA(12,1,6)",
        save_dir       = OUT,
        steps_per_hour = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ ARIMA terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()