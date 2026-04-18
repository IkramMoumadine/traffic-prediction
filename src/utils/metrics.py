# src/utils/metrics.py
# ============================================================
# Métriques et diagnostics — module unique partagé par tous
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, ConfusionMatrixDisplay
)
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings("ignore")


# ── Métriques de régression ──────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    label:  str = "") -> dict:
    """Calcule toutes les métriques de régression."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mse    = mean_squared_error(y_true, y_pred)
    rmse   = np.sqrt(mse)
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    max_er = float(np.max(np.abs(y_true - y_pred)))

    mask = y_true != 0
    mape = float(np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100) if mask.sum() > 0 else float("nan")

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(
        np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
    ) * 100)

    return dict(label=label, MSE=mse, RMSE=rmse, MAE=mae,
                MedAE=med_ae, MaxErr=max_er,
                MAPE=mape, SMAPE=smape, R2=r2)


def print_metrics(m: dict):
    """Affiche les métriques proprement."""
    print(f"\n{'='*50}")
    print(f"  Métriques — {m['label']}")
    print(f"{'='*50}")
    print(f"  MAE   : {m['MAE']:.4f}")
    print(f"  RMSE  : {m['RMSE']:.4f}")
    print(f"  MAPE  : {m['MAPE']:.2f} %")
    print(f"  SMAPE : {m['SMAPE']:.2f} %")
    print(f"  R²    : {m['R2']:.4f}")
    print(f"{'='*50}\n")


# ── Figures de diagnostic ────────────────────────────────────

def plot_all_diagnostics(y_true_denorm: np.ndarray,
                          y_pred_denorm: np.ndarray,
                          model_name:   str,
                          save_dir:     str,
                          steps_per_hour: int = 12):
    """
    Génère 9 figures de diagnostic standardisées.
    Appelé par TOUS les scripts main_*.py.
    """
    os.makedirs(save_dir, exist_ok=True)
    y_true = y_true_denorm.ravel()
    y_pred = y_pred_denorm.ravel()
    residuals = y_true - y_pred

    # ── 1. Prédictions ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    N = min(576, len(y_true))
    ax.plot(y_true[:N], label="Réel",   color="black",   lw=1.2, alpha=0.8)
    ax.plot(y_pred[:N], label="Prédit", color="#2196F3",  lw=1.2, alpha=0.85)
    ax.set_title(f"{model_name} — Prédictions")
    ax.set_xlabel("Pas de temps")
    ax.set_ylabel("Débit (veh/h)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Zoom 200 points ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    N = min(200, len(y_true))
    ax.plot(y_true[:N], label="Réel",   color="black",   lw=1.5)
    ax.plot(y_pred[:N], label="Prédit", color="#F44336",  lw=1.5, ls="--")
    ax.set_title(f"{model_name} — Zoom 200 points")
    ax.set_xlabel("Pas de temps")
    ax.set_ylabel("Débit (veh/h)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_prediction_zoom.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Scatter réel vs prédit ─────────────────────────────
    m = compute_metrics(y_true, y_pred, model_name)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="#2196F3")
    mn, mx = y_true.min(), y_true.max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Parfait")
    ax.set_title(f"Scatter — {model_name} (R²={m['R2']:.4f})")
    ax.set_xlabel("Réel (veh/h)"); ax.set_ylabel("Prédit (veh/h)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. Résidus dans le temps ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(residuals, color="#9C27B0", lw=0.8, alpha=0.7)
    ax.axhline(0,               color="black", lw=1.0, ls="--")
    ax.axhline(residuals.mean(), color="red",   lw=1.2, ls="--",
               label=f"μ = {residuals.mean():.2f} veh/h")
    ax.set_title(f"Résidus — {model_name} (σ={residuals.std():.2f} veh/h)")
    ax.set_xlabel("Pas de temps"); ax.set_ylabel("Erreur (veh/h)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/04_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Distribution des résidus ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=60, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(0,               color="black", lw=1.5, ls="--")
    ax.axvline(residuals.mean(), color="red",   lw=1.5, ls="--",
               label=f"μ = {residuals.mean():.2f}")
    ax.set_title(f"Distribution des résidus — {model_name}")
    ax.set_xlabel("Erreur (veh/h)"); ax.set_ylabel("Fréquence")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/05_residuals_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. ACF des résidus ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=48, ax=ax, color="#2196F3", alpha=0.05)
    ax.set_title(f"ACF Résidus — {model_name} (lags 0–48)")
    ax.set_xlabel("Lag"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/06_residuals_acf.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Q-Q plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.4, s=5, color="#2196F3")
    ax.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
    ax.set_title(f"Q-Q Plot — {model_name} (R={r:.3f})")
    ax.set_xlabel("Quantiles théoriques"); ax.set_ylabel("Quantiles observés")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/07_qq_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 8. MAE par heure ──────────────────────────────────────
    n_test = len(y_true)
    hours  = np.array([i % (24 * steps_per_hour) // steps_per_hour
                       for i in range(n_test)])
    mae_h  = [mean_absolute_error(y_true[hours == h], y_pred[hours == h])
              if (hours == h).sum() > 0 else 0
              for h in range(24)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(24), mae_h, color="#F44336", alpha=0.8, edgecolor="white")
    ax.set_title(f"MAE par heure — {model_name}")
    ax.set_xlabel("Heure"); ax.set_ylabel("MAE (veh/h)")
    ax.set_xticks(range(24)); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/08_mae_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 9. Comparaison vs Baseline (persistance) ─────────────
    baseline_mae  = mean_absolute_error(y_true[1:], y_true[:-1])
    baseline_rmse = np.sqrt(mean_squared_error(y_true[1:], y_true[:-1]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (mname, mval, bval) in zip(axes, [
        ("MAE (veh/h)",  m["MAE"],  baseline_mae),
        ("RMSE (veh/h)", m["RMSE"], baseline_rmse),
    ]):
        bars = ax.bar(["Baseline", model_name], [bval, mval],
                      color=["#9E9E9E", "#2196F3"], alpha=0.85,
                      edgecolor="white")
        ax.set_title(mname); ax.set_ylabel(mname)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, [bval, mval]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"{model_name} vs Baseline", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/09_vs_baseline.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 9 figures sauvegardées dans {save_dir}/")
    print_metrics(m)
    return m