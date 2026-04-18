# src/utils/visualization.py
# ============================================================
# Figures supplémentaires (loss, signal brut)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(history: dict, model_name: str, save_path: str):
    train_l = history["train_loss"]
    val_l   = history["val_loss"]
    best_ep = int(np.argmin(val_l)) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_l) + 1), train_l,
            label="Train", color="#2196F3", lw=2)
    ax.plot(range(1, len(val_l) + 1),   val_l,
            label="Val",   color="#F44336", lw=2, ls="--")
    ax.axvline(best_ep, color="green", ls=":", lw=1.5,
               label=f"Best epoch={best_ep} (val={min(val_l):.6f})")
    ax.set_title(f"Courbe de loss — {model_name}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("L1 Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Loss curve → {save_path}")


def plot_sensor_3days(data: np.ndarray, sensor_idx: int,
                      interval: int, save_path: str):
    signal = data[:, sensor_idx]
    steps  = int(3 * 24 * 60 / interval)   # 3 jours en pas de temps

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(signal[:steps], color="#2196F3", lw=1.2)
    ax.set_title(f"Capteur {sensor_idx} — 3 premiers jours")
    ax.set_xlabel("Pas de temps (5 min)")
    ax.set_ylabel("Débit (veh/h)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Signal 3j → {save_path}")

    