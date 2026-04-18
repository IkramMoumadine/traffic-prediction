# src/models/arima_model.py
# Wrapper léger autour de statsmodels ARIMA

from statsmodels.tsa.arima.model import ARIMA as _ARIMA
import numpy as np


def fit_arima(train_series: np.ndarray,
              order: tuple = (12, 1, 6)) -> object:
    """
    Fitte un modèle ARIMA sur la série d'entraînement.

    Paramètres
    ----------
    train_series : 1D array — valeurs normalisées
    order        : (p, d, q)

    Retour : ARIMAResultsWrapper fitté
    """
    model = _ARIMA(train_series, order=order)
    result = model.fit()
    print(f"  ARIMA{order} fitté — AIC={result.aic:.2f}")
    return result


def predict_arima(fitted_model, n_steps: int) -> np.ndarray:
    """Prédit n_steps pas en avant."""
    forecast = fitted_model.forecast(steps=n_steps)
    return np.array(forecast)