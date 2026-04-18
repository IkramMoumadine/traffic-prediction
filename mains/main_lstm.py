# mains/main_lstm.py
# ============================================================
# Pipeline LSTM multivarié — symétrique à main_gru.py
# Lance via : python run_all.py --models lstm
# ============================================================

import os

import config as C
from src.data.load_data       import load_pems07
from src.data.preprocessing   import (normalize_sensors_multi,
                                       create_windows_multivariate,
                                       temporal_split,
                                       denormalize)
from src.models.lstm_model    import LSTMModel
from src.training.trainer     import train_model
from src.utils.metrics        import plot_all_diagnostics
from src.utils.visualization  import plot_loss_curves

OUT = os.path.join(C.RESULTS_DIR, "lstm")


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  LSTM — capteurs {C.SELECTED_SENSORS}")
    print(f"{'='*60}")

    # ── 1. Chargement ──────────────────────────────────────────
    data = load_pems07(C.DATA_PATH)

    # ── 2. Normalisation multi-capteurs (fit sur train seul ✅) ─
    data_norm, scalers = normalize_sensors_multi(
        data, C.SELECTED_SENSORS, C.TRAIN_RATIO
    )

    # ── 3. Fenêtres multivariées ───────────────────────────────
    X, y = create_windows_multivariate(
        data_norm,
        sensor_indices = C.SELECTED_SENSORS,
        target_idx     = C.SENSOR_IDX,
        T_in           = C.T_IN,
        T_out          = C.T_OUT,
    )

    # ── 4. Split chronologique ─────────────────────────────────
    splits = temporal_split(X, y, C.TRAIN_RATIO, C.VAL_RATIO)

    # ── 5. Modèle LSTM ─────────────────────────────────────────
    n_sensors = len(C.SELECTED_SENSORS)

    model = LSTMModel(
        input_size  = n_sensors,
        T_out       = C.T_OUT,
        hidden_size = 128,
        num_layers  = 2,
        dropout     = C.DROPOUT,
    )

    print(f"\n  Architecture LSTM :")
    print(f"    input_size  = {n_sensors} capteurs {C.SELECTED_SENSORS}")
    print(f"    hidden_size = 128")
    print(f"    num_layers  = 2")
    print(f"    dropout     = {C.DROPOUT}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Paramètres  = {total_params:,}")

    # ── 6. Entraînement (boucle unifiée trainer.py) ────────────
    model, history, y_pred_norm, y_test_norm = train_model(
        model,
        splits,
        checkpoint_path = f"{OUT}/best_model.pt",
        epochs          = C.EPOCHS,
        batch_size      = C.BATCH_SIZE,
        lr              = C.LR,
        patience        = C.PATIENCE,
        seed            = C.SEED,
    )

    # ── 7. Courbe de loss ──────────────────────────────────────
    plot_loss_curves(history, "LSTM", f"{OUT}/loss.png")

    # ── 8. Dénormalisation ─────────────────────────────────────
    sc       = scalers[C.SENSOR_IDX]
    y_true_d = denormalize(y_test_norm, sc)
    y_pred_d = denormalize(y_pred_norm, sc)

    # ── 9. Diagnostics complets (9 figures standardisées) ──────
    plot_all_diagnostics(
        y_true_denorm   = y_true_d,
        y_pred_denorm   = y_pred_d,
        model_name      = "LSTM",
        save_dir        = OUT,
        steps_per_hour  = 60 // C.INTERVAL_MINUTES,
    )

    print(f"\n✅ LSTM terminé — résultats dans {OUT}/")


if __name__ == "__main__":
    run()