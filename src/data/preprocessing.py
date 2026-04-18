# src/data/preprocessing.py
# ============================================================
# Prétraitement commun à tous les modèles
# ============================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ── Normalisation ────────────────────────────────────────────

def normalize_sensor(data: np.ndarray,
                     sensor_idx: int,
                     train_ratio: float = 0.70) -> tuple:
    """
    Normalise MinMax UN capteur — fit sur train uniquement.
    Évite le data leakage sur val/test.

    Paramètres
    ----------
    data        : (T, N) — données brutes
    sensor_idx  : index du capteur cible
    train_ratio : fraction pour fitter le scaler

    Retour
    ------
    data_norm : (T, N) — colonne capteur normalisée [0,1]
    scaler    : MinMaxScaler fitté (pour dénormaliser)
    """
    signal    = data[:, sensor_idx].reshape(-1, 1)
    n_train   = int(len(signal) * train_ratio)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(signal[:n_train])               # fit sur train seul ✅

    signal_norm = scaler.transform(signal)

    data_norm = data.copy()
    data_norm[:, sensor_idx] = signal_norm.ravel()

    print(f"  Capteur {sensor_idx} normalisé — "
          f"min_train={scaler.data_min_[0]:.2f} | "
          f"max_train={scaler.data_max_[0]:.2f}")
    return data_norm, scaler


def normalize_sensors_multi(data: np.ndarray,
                             sensor_indices: list,
                             train_ratio: float = 0.70) -> tuple:
    """
    Normalise plusieurs capteurs indépendamment.
    Utilisé par le GRU multivarié.

    Retour
    ------
    data_norm : (T, N) — colonnes capteurs normalisées
    scalers   : dict {sensor_idx: MinMaxScaler}
    """
    n_train   = int(len(data) * train_ratio)
    data_norm = data.copy()
    scalers   = {}

    for idx in sensor_indices:
        signal  = data[:, idx].reshape(-1, 1)
        sc      = MinMaxScaler(feature_range=(0, 1))
        sc.fit(signal[:n_train])               # fit sur train seul ✅
        data_norm[:, idx] = sc.transform(signal).ravel()
        scalers[idx] = sc
        print(f"  Capteur {idx} normalisé")

    return data_norm, scalers


def denormalize(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse transform — ramène en veh/h."""
    return scaler.inverse_transform(
        values.reshape(-1, 1)
    ).ravel()


# ── Features temporelles sin/cos ─────────────────────────────

def build_temporal_features(n_timestamps: int,
                             interval_minutes: int = 5) -> np.ndarray:
    """
    4 features cycliques : sin/cos heure + sin/cos jour de semaine.
    Préserve la continuité circulaire (23h45 proche de 00h00).

    Retour : (T, 4)  float32
    """
    steps_per_day  = int(24 * 60 / interval_minutes)   # 288
    steps_per_week = 7 * steps_per_day                  # 2016

    t = np.arange(n_timestamps)

    hour_angle = 2 * np.pi * (t % steps_per_day)  / steps_per_day
    day_angle  = 2 * np.pi * (t % steps_per_week) / steps_per_week

    features = np.stack([
        np.sin(hour_angle), np.cos(hour_angle),
        np.sin(day_angle),  np.cos(day_angle)
    ], axis=1).astype(np.float32)

    print(f"  Features temporelles : {features.shape}")
    return features


# ── Fenêtres glissantes ──────────────────────────────────────

def create_windows_univariate(data_norm: np.ndarray,
                               temporal_feats: np.ndarray,
                               sensor_idx: int,
                               T_in: int = 12,
                               T_out: int = 1) -> tuple:
    """
    Fenêtres glissantes 5 canaux pour CNN 1D / 2D.
    Canal 0 : vitesse normalisée
    Canaux 1-4 : sin/cos heure + sin/cos jour

    Retour : X (N, 5, T_in), y (N, T_out)
    """
    signal   = data_norm[:, sensor_idx].astype(np.float32)
    combined = np.concatenate(
        [signal.reshape(-1, 1), temporal_feats], axis=1
    )   # (T, 5)

    X, y = [], []
    for i in range(len(signal) - T_in - T_out + 1):
        X.append(combined[i : i + T_in].T)           # (5, T_in)
        y.append(signal[i + T_in : i + T_in + T_out])

    X = np.array(X, dtype=np.float32)   # (N, 5, T_in)
    y = np.array(y, dtype=np.float32)   # (N, T_out)

    print(f"  Fenêtres univariées : X={X.shape}  y={y.shape}")
    return X, y


def create_windows_multivariate(data_norm: np.ndarray,
                                 sensor_indices: list,
                                 target_idx: int,
                                 T_in: int = 12,
                                 T_out: int = 1) -> tuple:
    """
    Fenêtres glissantes multivariées pour GRU.

    Retour : X (N, T_in, n_sensors), y (N, T_out)
    """
    X, y = [], []
    for i in range(len(data_norm) - T_in - T_out + 1):
        X.append(data_norm[i : i + T_in, sensor_indices])
        y.append(data_norm[i + T_in : i + T_in + T_out, target_idx])

    X = np.array(X, dtype=np.float32)   # (N, T_in, n_sensors)
    y = np.array(y, dtype=np.float32)   # (N, T_out)

    print(f"  Fenêtres multivariées : X={X.shape}  y={y.shape}")
    return X, y


# ── Split temporel ───────────────────────────────────────────

def temporal_split(X: np.ndarray, y: np.ndarray,
                   train_ratio: float = 0.70,
                   val_ratio:   float = 0.10) -> dict:
    """
    Split chronologique strict — aucun shuffle.
    """
    n       = len(X)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    splits = {
        "X_train": X[:i_train],       "y_train": y[:i_train],
        "X_val"  : X[i_train:i_val],  "y_val"  : y[i_train:i_val],
        "X_test" : X[i_val:],         "y_test" : y[i_val:],
    }
    print(f"  Split → train={i_train} | "
          f"val={i_val - i_train} | "
          f"test={n - i_val}")
    return splits