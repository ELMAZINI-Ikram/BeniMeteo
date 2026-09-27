"""
models/xgboost_model.py — Prédiction climatique XGBoost.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, target: str = "temp_mean") -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat[target] = df[target]
    for lag in [1, 3, 7, 14, 30, 60, 90, 180, 365]:
        feat[f"lag_{lag}"] = df[target].shift(lag)
    for win in [7, 14, 30, 90]:
        feat[f"roll_mean_{win}"] = df[target].rolling(win).mean()
        feat[f"roll_std_{win}"]  = df[target].rolling(win).std()
    feat["month"]       = df.index.month
    feat["day_of_year"] = df.index.dayofyear
    feat["sin_month"]   = np.sin(2 * np.pi * feat["month"] / 12)
    feat["cos_month"]   = np.cos(2 * np.pi * feat["month"] / 12)
    for col in ["precip", "wind_max", "humidity_mean", "pressure", "uv_max"]:
        if col in df.columns:
            feat[col]          = df[col]
            feat[f"{col}_lag1"] = df[col].shift(1)
    return feat.dropna()


def train_xgboost(df: pd.DataFrame, target: str = "temp_mean",
                  test_size: float = 0.15, forecast_steps: int = 90):
    """Entraîne XGBoost. Retourne (model, forecast_series, metrics_dict)."""
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error, r2_score

        feat   = build_features(df, target)
        X      = feat.drop(columns=[target])
        y      = feat[target]
        split  = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.85, colsample_bytree=0.85,
            early_stopping_rounds=30, eval_metric="rmse",
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)

        forecast = _iterative_forecast(model, df, feat, target, forecast_steps)
        return model, forecast, {"RMSE": round(rmse, 3), "R2": round(r2, 4)}

    except ImportError:
        _, pred, _ = _fallback_forecast(df, target, forecast_steps)
        return None, pred, {}
    except Exception as e:
        print(f"XGBoost échoué ({e})")
        _, pred, _ = _fallback_forecast(df, target, forecast_steps)
        return None, pred, {}


def _iterative_forecast(model, df, feat, target, steps):
    try:
        last_date    = df.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        df_ext       = df.copy()
        future_df    = pd.DataFrame(index=future_dates, columns=df.columns)
        df_ext       = pd.concat([df_ext, future_df])
        preds        = []
        for d in future_dates:
            fr = build_features(df_ext.loc[:d], target)
            if len(fr) == 0:
                preds.append(df[target].iloc[-1])
                continue
            row = fr.drop(columns=[target]).iloc[[-1]]
            p   = model.predict(row)[0]
            df_ext.loc[d, target] = p
            preds.append(p)
        return pd.Series(preds, index=future_dates, name=target)
    except Exception:
        return _fallback_forecast(df, target, steps)[1]


def _fallback_forecast(df, target, steps):
    y          = df[target].dropna()
    last_date  = y.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    monthly_mean = y.groupby(y.index.month).mean()
    pred = pd.Series([monthly_mean.get(d.month, y.mean()) for d in future_idx], index=future_idx)
    return None, pred, {}
