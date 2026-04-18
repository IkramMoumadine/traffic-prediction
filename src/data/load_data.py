# src/data/load_data.py
# ============================================================
# Chargement des données PEMS07
# Source unique : PEMS07.npz  (plus besoin de data.npy)
# ============================================================

import numpy as np


def load_pems07(path: str = "data/PEMS07/PEMS07.npz") -> np.ndarray:
    """
    Charge le dataset PEMS07 depuis un fichier .npz ou .npy.

    Retour
    ------
    np.ndarray shape (T, N)
        T = nombre de pas de temps
        N = nombre de capteurs (883 pour PEMS07)
    """
    if path.endswith(".npz"):
        archive = np.load(path)
        # Cherche la clé 'data' ou prend la première disponible
        key = "data" if "data" in archive else list(archive.files)[0]
        raw = archive[key]
    else:
        raw = np.load(path)

    # Normalise vers shape (T, N)
    if raw.ndim == 3:
        # (T, N, 1)  → (T, N)
        data = raw[:, :, 0].copy()
    elif raw.ndim == 2:
        # Déjà (T, N)
        data = raw.copy()
    else:
        raise ValueError(f"Shape inattendue : {raw.shape}")

    print(f"✅ Données chargées : shape={data.shape} | "
          f"min={data.min():.2f} | max={data.max():.2f} | "
          f"mean={data.mean():.2f}")
    return data.astype(np.float32)