# config.py
# ============================================================
# Configuration centralisée — tous les hyperparamètres
# ============================================================

# ── Reproductibilité ────────────────────────────────────────
SEED = 42

# ── Données ─────────────────────────────────────────────────
DATA_PATH        = "data/PEMS07/PEMS07.npz"   # source unique
INTERVAL_MINUTES = 5                           # résolution temporelle
SENSOR_IDX       = 13                          # capteur cible
SELECTED_SENSORS = [13, 537, 864]              # capteurs multivariés (GRU)

# ── Fenêtres temporelles ─────────────────────────────────────
T_IN  = 12    # 12 pas × 5 min = 1h d'historique
T_OUT = 1     # prédiction à 5 min

# ── Split temporel ───────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
# test = 1 - TRAIN_RATIO - VAL_RATIO = 0.20

# ── Entraînement ─────────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-3
DROPOUT    = 0.3
PATIENCE   = 10   # early stopping

# ── Chemins de sortie ────────────────────────────────────────
RESULTS_DIR = "results"