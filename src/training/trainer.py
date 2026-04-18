# src/training/trainer.py
# ============================================================
# Boucle d'entraînement unifiée pour CNN et GRU
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5,
                 checkpoint_path: str = "best_model.pt"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_loss        = float("inf")
        self.counter          = 0
        self.stop             = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_model(model: nn.Module,
                splits: dict,
                checkpoint_path: str,
                epochs: int     = 50,
                batch_size: int = 64,
                lr: float       = 1e-3,
                patience: int   = 10,
                seed: int       = 42) -> tuple:
    """
    Boucle d'entraînement générique (CNN ou GRU).

    Paramètres
    ----------
    model           : nn.Module — modèle PyTorch initialisé
    splits          : dict avec X_train/y_train/X_val/y_val/X_test/y_test
    checkpoint_path : chemin pour sauvegarder le meilleur modèle

    Retour
    ------
    model        : modèle avec les meilleurs poids chargés
    history      : dict {"train_loss": [...], "val_loss": [...]}
    y_pred       : np.ndarray — prédictions sur le test set
    y_test       : np.ndarray — valeurs réelles du test set
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # ── DataLoader ──
    X_train_t = torch.tensor(splits["X_train"], dtype=torch.float32)
    y_train_t = torch.tensor(splits["y_train"], dtype=torch.float32)
    loader    = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=False
    )

    X_val_t  = torch.tensor(splits["X_val"],  dtype=torch.float32).to(device)
    y_val_t  = torch.tensor(splits["y_val"],  dtype=torch.float32).to(device)
    X_test_t = torch.tensor(splits["X_test"], dtype=torch.float32).to(device)

    model = model.to(device)

    # ── Optimiseur & scheduler ──
    criterion = nn.L1Loss()   # MAE loss — validée expérimentalement
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper   = EarlyStopping(patience=patience,
                               checkpoint_path=checkpoint_path)

    train_losses, val_losses = [], []

    # ── Boucle ──
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        avg_train = running / len(loader.dataset)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            avg_val = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(avg_val)

        scheduler.step(avg_val)
        stopper(avg_val, model)

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train: {avg_train:.6f} | Val: {avg_val:.6f}"
              + (" ← best" if stopper.counter == 0 else ""))

        if stopper.stop:
            print(f"  Early stopping à l'époque {epoch+1}")
            break

    # ── Charge les meilleurs poids ──
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, history, y_pred, splits["y_test"]