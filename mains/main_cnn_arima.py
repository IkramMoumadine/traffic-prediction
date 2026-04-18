# mains/main_cnn_arima.py
# ============================================================
# Pipeline hybride CNN 1D + ARIMA rolling forecast
#
# Étapes :
#   1-6  : identiques à main_cnn1d.py  (données → CNN entraîné)
#   7    : calcul des résidus CNN (train et test)
#   8    : ARIMA rolling forecast sur les résidus
#   9    : prédictions hybrides = CNN + correction ARIMA
#   10   : métriques comparatives (Baseline / CNN / CNN+ARIMA)
#   11   : visualisations dédiées + diagnostics standards
#
# Paramètre clé : ROLLING_N_MAX
#   Nombre de pas du test traités en rolling.
#   500 pts ≈ 42h de trafic — prend ~2-5 min CPU.
#   Mettre None pour traiter tout le test (plus long).
# ============================================================

import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

import config as C
from src.data.load_data        import load_pems07
from src.data.preprocessing    import (normalize_sensor,
                                        build_temporal_features,
                                        create_windows_univariate,
                                        temporal_split, denormalize)
from src.models.cnn1d_model    import CNN1D
from src.models.cnn_arima_model import (fit_arima_on_residuals,
                                         arima_rolling_forecast)
from src.training.trainer      import train_model
from src.utils.metrics         import compute_metrics, print_metrics
from src.utils.visualization   import plot_loss_curves, plot_sensor_3days

OUT = os.path.join(C.RESULTS_DIR, "cnn_arima")

# ── Paramètre spécifique CNN+ARIMA ────────────────────────────
ARIMA_ORDER   = (6, 0, 3)   # d=0 : résidus déjà stationnaires
ROLLING_N_MAX = 500          # pts test en rolling (None = tout)


def _predict_cnn(model: torch.nn.Module,
                 X_np: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """Inférence CNN sur un tableau numpy."""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_np, dtype=torch.float32).to(device)
        return model(t).cpu().numpy().flatten()


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule RMSE, MAE, R², MAPE."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    ss_r = float(np.sum((y_true - y_pred) ** 2))
    ss_t = float(np.sum((y_true - y_true.mean()) ** 2))
    r2   = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
           if mask.sum() > 0 else float("nan")
    return dict(RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)


def run():
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}\n  CNN 1D + ARIMA Rolling — Capteur {C.SENSOR_IDX}\n{'='*60}")

    # ── 1. Chargement ─────────────────────────────────────────
    data = load_pems07(C.DATA_PATH)
    plot_sensor_3days(data, C.SENSOR_IDX, C.INTERVAL_MINUTES,
                      f"{OUT}/00_signal_3jours.png")

    # ── 2. Normalisation (fit sur train seul ✅) ──────────────
    data_norm, scaler = normalize_sensor(data, C.SENSOR_IDX, C.TRAIN_RATIO)

    # ── 3. Features temporelles ───────────────────────────────
    temporal = build_temporal_features(data_norm.shape[0], C.INTERVAL_MINUTES)

    # ── 4. Fenêtres (même format que CNN 1D) ──────────────────
    X, y = create_windows_univariate(
        data_norm, temporal, C.SENSOR_IDX, C.T_IN, C.T_OUT
    )

    # ── 5. Split chronologique ────────────────────────────────
    splits = temporal_split(X, y, C.TRAIN_RATIO, C.VAL_RATIO)

    # ── 6. Entraînement CNN 1D (boucle unifiée) ───────────────
    model = CNN1D(T_in=C.T_IN, T_out=C.T_OUT,
                  n_channels=5, dropout=C.DROPOUT)

    model, history, _, _ = train_model(
        model, splits,
        checkpoint_path = f"{OUT}/best_cnn_model.pt",
        epochs          = C.EPOCHS,
        batch_size      = C.BATCH_SIZE,
        lr              = C.LR,
        patience        = C.PATIENCE,
        seed            = C.SEED,
    )
    plot_loss_curves(history, "CNN 1D (pour CNN+ARIMA)", f"{OUT}/loss_cnn.png")

    # ── 7. Résidus CNN ─────────────────────────────────────────
    print("\n  Calcul des résidus CNN ...")
    device = next(model.parameters()).device

    y_train_pred  = _predict_cnn(model, splits["X_train"], device)
    y_train_true  = splits["y_train"].flatten()
    residus_train = y_train_true - y_train_pred

    y_test_pred  = _predict_cnn(model, splits["X_test"], device)
    y_test_true  = splits["y_test"].flatten()
    residus_test = y_test_true - y_test_pred

    fit_arima_on_residuals(residus_train, order=ARIMA_ORDER)

    # ── 8. ARIMA rolling forecast sur les résidus ─────────────
    corrections = arima_rolling_forecast(
        residuals_train = residus_train,
        residuals_test  = residus_test,
        order           = ARIMA_ORDER,
        n_max           = ROLLING_N_MAX,
        verbose         = True,
    )

    # ── 9. Prédictions hybrides ────────────────────────────────
    n_roll       = len(corrections)
    y_true_sub   = y_test_true[:n_roll]
    y_cnn_sub    = y_test_pred[:n_roll]
    y_hybrid_sub = y_cnn_sub + corrections
    # Baseline naïf : dernière valeur connue de la fenêtre d'entrée
    y_base_sub   = splits["X_test"][:n_roll, 0, -1]   # canal 0, dernier pas

    # ── 10. Dénormalisation ────────────────────────────────────
    y_true_d   = denormalize(y_true_sub,   scaler)
    y_cnn_d    = denormalize(y_cnn_sub,    scaler)
    y_hybrid_d = denormalize(y_hybrid_sub, scaler)
    y_base_d   = denormalize(y_base_sub,   scaler)

    # ── 11. Métriques comparatives ────────────────────────────
    results = {}
    print(f"\n  {'Modèle':<22} {'RMSE':>8} {'MAE':>8} {'R²':>7} {'MAPE':>9}")
    print(f"  {'─'*58}")
    for label, yt, yp in [
        ("Baseline (naïf)", y_true_d, y_base_d),
        ("CNN-1D",          y_true_d, y_cnn_d),
        ("CNN + ARIMA",     y_true_d, y_hybrid_d),
    ]:
        m = _metrics_dict(yt, yp)
        results[label] = m
        print(f"  {label:<22} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} "
              f"{m['R2']:>7.4f} {m['MAPE']:>8.2f}%")

    gain_rmse = (results["CNN-1D"]["RMSE"] - results["CNN + ARIMA"]["RMSE"]) \
                / results["CNN-1D"]["RMSE"] * 100
    gain_mae  = (results["CNN-1D"]["MAE"]  - results["CNN + ARIMA"]["MAE"]) \
                / results["CNN-1D"]["MAE"]  * 100
    print(f"\n  Gain CNN+ARIMA vs CNN seul :")
    print(f"  RMSE : {gain_rmse:+.2f}%  |  MAE : {gain_mae:+.2f}%")
    if gain_rmse > 0:
        print("  ✅ ARIMA améliore le CNN")
    else:
        print("  ⚠️  Résidus trop bruités pour ARIMA — gain nul/négatif")

    # ── 12. Visualisations ────────────────────────────────────
    _plot_comparison(y_true_d, y_cnn_d, y_hybrid_d,
                     residus_test[:n_roll], corrections, scaler, OUT)
    _plot_scatter(y_true_d, y_cnn_d, y_hybrid_d, results, OUT)
    _plot_barplot(results, OUT)

    print(f"\n✅ CNN+ARIMA terminé — résultats dans {OUT}/")


# ── Fonctions de visualisation dédiées ───────────────────────

def _plot_comparison(y_true, y_cnn, y_hybrid,
                     residus, corrections, scaler, out_dir):
    """Fig 1 — Prédictions + résidus vs corrections."""
    from sklearn.preprocessing import MinMaxScaler
    N_SHOW = min(200, len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.plot(y_true[:N_SHOW],   label="Réel",        color="#1a237e", lw=1.5)
    ax.plot(y_cnn[:N_SHOW],    label="CNN-1D",      color="#f57c00", lw=1.2, ls="--")
    ax.plot(y_hybrid[:N_SHOW], label="CNN + ARIMA", color="#6a1b9a", lw=1.2, ls="-.")
    ax.set_title(f"Capteur {C.SENSOR_IDX} — CNN seul vs CNN+ARIMA ({N_SHOW} premiers pts)")
    ax.set_ylabel("Débit (veh/h)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Dénormaliser les résidus (différence dans l'espace normalisé → veh/h)
    zero_norm   = scaler.inverse_transform([[0]])[0, 0]
    residu_veh  = y_true[:N_SHOW] - y_cnn[:N_SHOW]
    correct_veh = scaler.inverse_transform(
        corrections[:N_SHOW].reshape(-1, 1)
    ).ravel() - zero_norm

    ax.plot(residu_veh,  label="Résidu CNN (veh/h)",       color="#f57c00", lw=1.0)
    ax.plot(correct_veh, label="Correction ARIMA (veh/h)", color="#6a1b9a", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Résidus CNN vs Correction ARIMA rolling")
    ax.set_xlabel("Pas de temps")
    ax.set_ylabel("Erreur / Correction (veh/h)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/01_predictions_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {path}")


def _plot_scatter(y_true, y_cnn, y_hybrid, results, out_dir):
    """Fig 2 — Scatter plots côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (yp, label, key) in zip(axes, [
        (y_cnn,    "CNN-1D",      "CNN-1D"),
        (y_hybrid, "CNN + ARIMA", "CNN + ARIMA"),
    ]):
        m = results[key]
        ax.scatter(y_true, yp, alpha=0.3, s=5, color="#2196F3")
        lim = [min(y_true.min(), yp.min()), max(y_true.max(), yp.max())]
        ax.plot(lim, lim, "r--", lw=1.5, label="Parfait y=x")
        ax.set_title(f"{label}  (R²={m['R2']:.4f}  MAE={m['MAE']:.1f})")
        ax.set_xlabel("Réel (veh/h)"); ax.set_ylabel("Prédit (veh/h)")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Scatter — Capteur {C.SENSOR_IDX}", fontsize=13)
    plt.tight_layout()
    path = f"{out_dir}/02_scatter_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {path}")


def _plot_barplot(results: dict, out_dir: str):
    """Fig 3 — Barplot RMSE / MAE pour les 3 modèles."""
    labels = ["Baseline (naïf)", "CNN-1D", "CNN + ARIMA"]
    colors = ["#9E9E9E", "#f57c00", "#6a1b9a"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in zip(axes,
                                   ["RMSE", "MAE"],
                                   ["RMSE (veh/h)", "MAE (veh/h)"]):
        vals = [results[k][metric] for k in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(f"{ylabel} — plus bas = mieux")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle(f"Comparaison — Capteur {C.SENSOR_IDX}", fontsize=13)
    plt.tight_layout()
    path = f"{out_dir}/03_metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {path}")


if __name__ == "__main__":
    run()