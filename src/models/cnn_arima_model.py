# src/models/cnn_arima_model.py
# ============================================================
# Modèle hybride CNN 1D + ARIMA rolling forecast
#
# Principe (correction des résidus) :
#   1. CNN prédit le débit → résidu = réel - prédit_CNN
#   2. ARIMA(p,d,q) est fitté sur les résidus du train
#   3. En test : rolling forecast — à chaque pas t :
#        a) ARIMA est fitté sur résidus_train + résidus_test[:t]
#        b) forecast(1) → correction du prochain pas
#   4. Prédiction hybride = prédit_CNN + correction_ARIMA
#
# Avantage vs ARIMA statique :
#   ARIMA statique diverge (prédit → moyenne asymptotiquement).
#   Le rolling forecast reste ancré dans le présent à chaque pas.
#
# Ce module ne contient que la logique ARIMA.
# Le CNN est réutilisé depuis src/models/cnn1d_model.py.
# ============================================================

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def fit_arima_on_residuals(residuals_train: np.ndarray,
                            order: tuple = (6, 0, 3)) -> None:
    """
    Fonction utilitaire — vérifie que l'ordre ARIMA est cohérent
    avec des résidus déjà stationnaires (d=0 recommandé).

    Paramètre d=0 car les résidus CNN sont déjà centrés ≈ 0
    et ne présentent pas de tendance → pas besoin de différenciation.

    Paramètres
    ----------
    residuals_train : résidus du CNN sur le jeu d'entraînement
    order           : (p, d, q) — d=0 recommandé pour les résidus

    Retour : None (affichage informatif uniquement)
    """
    if order[1] != 0:
        print(f"  ⚠️  d={order[1]} sur des résidus : risque de sur-différenciation.")
    print(f"  Résidus train — μ={residuals_train.mean():.5f} | "
          f"σ={residuals_train.std():.5f}")


def arima_rolling_forecast(residuals_train: np.ndarray,
                            residuals_test:  np.ndarray,
                            order:           tuple = (6, 0, 3),
                            n_max:           int   = None,
                            verbose:         bool  = True) -> np.ndarray:
    """
    Rolling forecast ARIMA sur les résidus du test.

    À chaque pas t :
      - historique = résidus_train + résidus_test[:t] (connus)
      - ARIMA fitté sur cet historique
      - forecast(1) → correction pour le pas t+1
      - résidu_test[t] révélé → ajouté à l'historique

    Paramètres
    ----------
    residuals_train : résidus CNN sur le train (np.ndarray 1D)
    residuals_test  : résidus CNN sur le test  (np.ndarray 1D)
    order           : (p, d, q) ARIMA
    n_max           : nb de pas à traiter (None = tout le test)
                      ↑ réduire pour accélérer les tests
    verbose         : afficher la progression

    Retour
    ------
    corrections : np.ndarray (n_max,) — corrections ARIMA par pas
    """
    n_rolling = len(residuals_test) if n_max is None else min(n_max, len(residuals_test))
    corrections   = np.zeros(n_rolling)
    history_arima = list(residuals_train)   # historique courant

    if verbose:
        print(f"  ARIMA{order} rolling forecast — {n_rolling} pas")

    for i in range(n_rolling):
        if verbose and i % 100 == 0:
            print(f"    Step {i:>4}/{n_rolling} ...", end="\r")

        try:
            fit = ARIMA(history_arima, order=order).fit(
                method="innovations_mle"
            )
            correction = float(fit.forecast(steps=1).iloc[0])
        except Exception:
            correction = 0.0   # fallback si ARIMA diverge

        corrections[i] = correction
        # Révéler le vrai résidu → ancrage pour le prochain pas
        history_arima.append(float(residuals_test[i]))

    if verbose:
        print(f"\n  ✅ Rolling terminé — "
              f"μ_corr={corrections.mean():.5f} | "
              f"σ_corr={corrections.std():.5f}")

    return corrections