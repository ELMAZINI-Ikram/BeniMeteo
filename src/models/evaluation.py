"""
evaluation.py — Métriques d'évaluation des modèles.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(actual: pd.Series, forecast: pd.Series, model_name: str = "Modèle") -> dict:
    common = actual.index.intersection(forecast.index)
    if len(common) == 0:
        n = min(len(actual), len(forecast))
        y_true, y_pred = actual.values[:n], forecast.values[:n]
    else:
        y_true, y_pred = actual.loc[common].values, forecast.loc[common].values

    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"model": model_name, "RMSE": None, "MAE": None, "MAPE": None, "R2": None}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "RMSE": round(rmse, 3), "MAE": round(mae, 3),
        "MAPE": round(mape, 2), "R2": round(r2, 4),
        "n_points": len(y_true),
    }
