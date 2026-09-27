"""
models/ml_advanced.py — Modèles ML avancés BeniMeteo
- XGBoost avec optimisation GridSearchCV
- Prophet (Meta) pour séries saisonnières
- LSTM (TensorFlow/Keras) pour deep learning
- Isolation Forest pour détection d'anomalies
"""

import pandas as pd
import numpy as np
from datetime import date


# ══════════════════════════════════════════════════════════════════════════════
# 1. XGBOOST OPTIMISÉ (GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════════

def build_features_v2(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Features enrichies pour XGBoost v2."""
    feat = pd.DataFrame(index=df.index)
    feat[target] = df[target]

    # Lags temporels
    for lag in [1, 2, 3, 7, 14, 21, 30, 60, 90, 180, 365]:
        if lag < len(df):
            feat[f"lag_{lag}"] = df[target].shift(lag)

    # Moyennes mobiles
    for win in [7, 14, 30, 60, 90, 180]:
        feat[f"roll_mean_{win}"] = df[target].rolling(win, min_periods=3).mean()
        feat[f"roll_std_{win}"]  = df[target].rolling(win, min_periods=3).std()
        feat[f"roll_min_{win}"]  = df[target].rolling(win, min_periods=3).min()
        feat[f"roll_max_{win}"]  = df[target].rolling(win, min_periods=3).max()

    # Caractéristiques temporelles
    feat["month"]        = df.index.month
    feat["day_of_year"]  = df.index.dayofyear
    feat["day_of_week"]  = df.index.dayofweek
    feat["quarter"]      = df.index.quarter
    feat["sin_doy"]      = np.sin(2 * np.pi * feat["day_of_year"] / 365)
    feat["cos_doy"]      = np.cos(2 * np.pi * feat["day_of_year"] / 365)
    feat["sin_month"]    = np.sin(2 * np.pi * feat["month"] / 12)
    feat["cos_month"]    = np.cos(2 * np.pi * feat["month"] / 12)

    # Variables météo exogènes
    for col in ["precip", "wind_max", "humidity_max", "pressure", "uv_max",
                "tmax", "tmin", "tmean"]:
        if col in df.columns and col != target:
            feat[col]          = df[col]
            feat[f"{col}_lag1"] = df[col].shift(1)
            feat[f"{col}_lag7"] = df[col].shift(7)

    # Tendance linéaire
    feat["trend"] = np.arange(len(df))

    return feat.dropna()


def train_xgboost_optimized(df: pd.DataFrame, target: str = "temp_mean",
                              test_size: float = 0.15,
                              forecast_steps: int = 90,
                              optimize: bool = True) -> tuple:
    """
    XGBoost avec GridSearchCV optionnel.
    optimize=True : cherche les meilleurs hyperparamètres (plus lent)
    optimize=False : utilise des paramètres pré-optimisés pour la région BM
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.model_selection import TimeSeriesSplit

        # Alias colonnes
        if target not in df.columns:
            for alias in ["tmean", "temperature_mean"]:
                if alias in df.columns:
                    df = df.copy(); df[target] = df[alias]; break

        if target not in df.columns:
            return None, None, {"error": f"Colonne {target} introuvable"}

        feat   = build_features_v2(df, target)
        X      = feat.drop(columns=[target])
        y      = feat[target]
        split  = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if optimize:
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                "n_estimators":    [300, 500],
                "max_depth":       [4, 6],
                "learning_rate":   [0.03, 0.05],
                "subsample":       [0.8, 0.9],
                "colsample_bytree":[0.8, 0.9],
            }
            tscv = TimeSeriesSplit(n_splits=3)
            base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
            gs = GridSearchCV(base_model, param_grid, cv=tscv,
                              scoring="neg_root_mean_squared_error",
                              n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best_params = gs.best_params_
            print(f"✅ XGBoost best params: {best_params}")
        else:
            # Paramètres pré-optimisés pour séries climatiques Béni Mellal
            model = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.04, max_depth=5,
                subsample=0.85, colsample_bytree=0.85,
                min_child_weight=3, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0,
                early_stopping_rounds=30, eval_metric="rmse",
                random_state=42, verbosity=0,
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], verbose=False)
            best_params = {}

        y_pred = model.predict(X_test)
        rmse   = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
        mae    = round(float(mean_absolute_error(y_test, y_pred)), 3)
        r2     = round(float(r2_score(y_test, y_pred)), 3)
        mape   = round(float(np.mean(np.abs((y_test - y_pred) / y_test.replace(0,1))) * 100), 1)

        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": f"{mape}%",
                   "best_params": best_params}

        # Prévision récursive
        last_feat = X.iloc[[-1]].copy()
        predictions = []
        last_idx    = y.index[-1]
        future_idx  = pd.date_range(start=last_idx + pd.Timedelta(days=1),
                                     periods=forecast_steps, freq="D")

        for i, dt in enumerate(future_idx):
            pred_val = float(model.predict(last_feat)[0])
            predictions.append(pred_val)
            # Mise à jour lag_1
            if "lag_1" in last_feat.columns:
                last_feat["lag_1"] = pred_val
            if "trend" in last_feat.columns:
                last_feat["trend"] = last_feat["trend"].values[0] + 1
            if "sin_doy" in last_feat.columns:
                doy = dt.timetuple().tm_yday
                last_feat["sin_doy"] = np.sin(2 * np.pi * doy / 365)
                last_feat["cos_doy"] = np.cos(2 * np.pi * doy / 365)
            if "month" in last_feat.columns:
                last_feat["month"] = dt.month
            if "day_of_year" in last_feat.columns:
                last_feat["day_of_year"] = dt.timetuple().tm_yday

        forecast = pd.Series(predictions, index=future_idx)

        # Contraintes physiques
        if "temp" in target or target in ["tmax", "tmin", "tmean", "temp_mean"]:
            forecast = forecast.clip(lower=-5, upper=50)

        print(f"✅ XGBoost optimisé — RMSE={rmse} | R²={r2} | {forecast_steps} jours")
        return model, forecast, metrics

    except ImportError as e:
        print(f"⚠️ XGBoost non disponible ({e})")
        return None, None, {"error": str(e)}
    except Exception as e:
        print(f"⚠️ XGBoost échoué ({e})")
        return None, None, {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROPHET (Meta) — Séries saisonnières
# ══════════════════════════════════════════════════════════════════════════════

def train_prophet(df: pd.DataFrame, target: str = "temp_mean",
                   forecast_steps: int = 90) -> tuple:
    """
    Modèle Prophet de Meta — excellent pour les séries avec saisonnalité forte.
    Avantages : gère automatiquement les tendances, saisonnalités annuelle/hebdo.
    """
    try:
        from prophet import Prophet
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        if target not in df.columns:
            for alias in ["tmean", "temperature_mean"]:
                if alias in df.columns:
                    df = df.copy(); df[target] = df[alias]; break

        if target not in df.columns:
            return None, None, {"error": f"Colonne {target} introuvable"}

        # Format Prophet : colonnes ds et y
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y":  df[target].values,
        }).dropna()

        # Configuration pour météo Béni Mellal-Khénifra
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,   # Pas de saisonnalité hebdo pour météo
            daily_seasonality=False,
            seasonality_mode="multiplicative",   # Mieux pour températures
            changepoint_prior_scale=0.05,        # Flexibilité tendance
            seasonality_prior_scale=10,
            interval_width=0.95,
        )

        # Régression sur variables exogènes si disponibles
        for col in ["precip", "wind_max", "humidity_max"]:
            if col in df.columns:
                prophet_df[col] = df[col].values[:len(prophet_df)]
                model.add_regressor(col)

        model.fit(prophet_df)

        # Prévision
        future = model.make_future_dataframe(periods=forecast_steps)

        # Ajouter régresseurs dans le futur (valeurs moyennes historiques)
        for col in ["precip", "wind_max", "humidity_max"]:
            if col in prophet_df.columns:
                future[col] = prophet_df[col].mean()

        forecast_df = model.predict(future)
        forecast_future = forecast_df.iloc[-forecast_steps:]

        future_idx = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_steps, freq="D")

        fc_mean  = pd.Series(forecast_future["yhat"].values, index=future_idx)
        fc_lower = pd.Series(forecast_future["yhat_lower"].values, index=future_idx)
        fc_upper = pd.Series(forecast_future["yhat_upper"].values, index=future_idx)
        conf     = pd.DataFrame({"lower": fc_lower, "upper": fc_upper})

        # Contraintes physiques
        if "temp" in target:
            fc_mean  = fc_mean.clip(lower=-5, upper=50)
            conf["lower"] = conf["lower"].clip(lower=-5)
            conf["upper"] = conf["upper"].clip(upper=50)
        elif "precip" in target:
            fc_mean  = fc_mean.clip(lower=0)
            conf["lower"] = conf["lower"].clip(lower=0)

        # Métriques sur données historiques
        train_fc = forecast_df.iloc[:len(prophet_df)]
        rmse = round(float(np.sqrt(mean_squared_error(
            prophet_df["y"], train_fc["yhat"]))), 3)
        mae  = round(float(mean_absolute_error(
            prophet_df["y"], train_fc["yhat"])), 3)

        metrics = {"RMSE": rmse, "MAE": mae, "modele": "Prophet"}
        print(f"✅ Prophet — RMSE={rmse} | {forecast_steps} jours")
        return model, (fc_mean, conf), metrics

    except ImportError:
        print("⚠️ Prophet non installé — pip install prophet")
        return None, None, {"error": "pip install prophet requis"}
    except Exception as e:
        print(f"⚠️ Prophet échoué ({e})")
        return None, None, {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. LSTM (TensorFlow/Keras)
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(df: pd.DataFrame, target: str = "temp_mean",
                forecast_steps: int = 90,
                lookback: int = 60,
                epochs: int = 50) -> tuple:
    """
    LSTM (Long Short-Term Memory) — Deep learning pour séries temporelles.
    Plus précis sur les longues dépendances temporelles.
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error

        tf.get_logger().setLevel("ERROR")

        if target not in df.columns:
            for alias in ["tmean", "temperature_mean"]:
                if alias in df.columns:
                    df = df.copy(); df[target] = df[alias]; break

        series = df[target].dropna().values.reshape(-1, 1)
        if len(series) < lookback + 50:
            return None, None, {"error": f"Historique trop court pour LSTM (min {lookback + 50} jours)"}

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(series)

        # Création des séquences
        X_seq, y_seq = [], []
        for i in range(lookback, len(scaled)):
            X_seq.append(scaled[i-lookback:i, 0])
            y_seq.append(scaled[i, 0])
        X_seq = np.array(X_seq).reshape(-1, lookback, 1)
        y_seq = np.array(y_seq)

        split = int(len(X_seq) * 0.85)
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        # Architecture LSTM
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                  input_shape=(lookback, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss="mse")

        cb = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                  validation_data=(X_te, y_te),
                  callbacks=cb, verbose=0)

        # Prévision récursive
        last_seq    = scaled[-lookback:].copy()
        predictions = []
        for _ in range(forecast_steps):
            inp      = last_seq.reshape(1, lookback, 1)
            pred_sc  = model.predict(inp, verbose=0)[0, 0]
            predictions.append(pred_sc)
            last_seq = np.append(last_seq[1:], [[pred_sc]], axis=0)

        fc_arr  = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        last_dt = df[target].dropna().index[-1]
        future_idx = pd.date_range(start=last_dt + pd.Timedelta(days=1),
                                    periods=forecast_steps, freq="D")
        forecast = pd.Series(fc_arr, index=future_idx)

        # Contraintes physiques
        if "temp" in target:
            forecast = forecast.clip(lower=-5, upper=50)

        # Métriques test
        y_te_pred = model.predict(X_te, verbose=0).flatten()
        y_te_inv  = scaler.inverse_transform(y_te.reshape(-1,1)).flatten()
        yp_inv    = scaler.inverse_transform(y_te_pred.reshape(-1,1)).flatten()
        rmse = round(float(np.sqrt(mean_squared_error(y_te_inv, yp_inv))), 3)

        metrics = {"RMSE": rmse, "modele": "LSTM", "epochs": epochs, "lookback": lookback}
        print(f"✅ LSTM — RMSE={rmse} | {forecast_steps} jours")
        return model, forecast, metrics

    except ImportError:
        print("⚠️ TensorFlow non installé — pip install tensorflow")
        return None, None, {"error": "pip install tensorflow requis"}
    except Exception as e:
        print(f"⚠️ LSTM échoué ({e})")
        return None, None, {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. DÉTECTION D'ANOMALIES (Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(df: pd.DataFrame, variables: list = None,
                      contamination: float = 0.05) -> pd.DataFrame:
    """
    Détecte les événements météorologiques extrêmes / anomalies.
    Utilise Isolation Forest (non supervisé).
    contamination : proportion d'anomalies attendues (5% par défaut)
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        if variables is None:
            variables = [c for c in ["tmax", "tmin", "tmean", "temp_mean",
                                      "precip", "wind_max", "humidity_max"]
                         if c in df.columns]

        if not variables:
            return pd.DataFrame()

        data = df[variables].dropna()
        if len(data) < 50:
            return pd.DataFrame()

        scaler   = StandardScaler()
        data_sc  = scaler.fit_transform(data)

        iso = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        labels  = iso.fit_predict(data_sc)
        scores  = iso.score_samples(data_sc)

        anomalies_df = data.copy()
        anomalies_df["anomalie"]      = labels == -1
        anomalies_df["score_anomalie"]= (-scores).round(4)   # Plus élevé = plus anormal
        anomalies_df = anomalies_df[anomalies_df["anomalie"]].sort_values(
            "score_anomalie", ascending=False)

        # Catégoriser le type d'anomalie
        def categorize(row):
            cats = []
            if "tmax" in row.index and row.get("tmax", 0) > 42:
                cats.append("Canicule extrême")
            if "tmin" in row.index and row.get("tmin", 0) < 0:
                cats.append("Gel")
            if "precip" in row.index and row.get("precip", 0) > 50:
                cats.append("Précipitations extrêmes")
            if "wind_max" in row.index and row.get("wind_max", 0) > 80:
                cats.append("Vent violent")
            return " + ".join(cats) if cats else "Anomalie composite"

        anomalies_df["type_anomalie"] = anomalies_df.apply(categorize, axis=1)

        print(f"✅ Isolation Forest — {len(anomalies_df)} anomalies détectées "
              f"sur {len(data)} jours ({contamination*100:.0f}% seuil)")
        return anomalies_df.reset_index()

    except ImportError:
        print("⚠️ scikit-learn requis pour Isolation Forest")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Détection anomalies échouée ({e})")
        return pd.DataFrame()


def compare_all_models(df: pd.DataFrame, target: str = "temp_mean",
                        steps: int = 90) -> dict:
    """
    Lance tous les modèles disponibles et retourne les prévisions + métriques.
    Retourne {"sarima": ..., "xgboost": ..., "prophet": ..., "lstm": ...}
    """
    results = {}

    print("=== Comparaison tous modèles ===")

    # SARIMA
    try:
        from src.models.sarima import train_sarima
        _, fc, ci = train_sarima(df, target=target, steps=steps)
        results["sarima"] = {"forecast": fc, "conf": ci, "metrics": {}}
    except Exception as e:
        results["sarima"] = {"error": str(e)}

    # XGBoost optimisé
    _, fc, metrics = train_xgboost_optimized(df, target=target,
                                               forecast_steps=steps, optimize=False)
    if fc is not None:
        results["xgboost"] = {"forecast": fc, "metrics": metrics}

    # Prophet
    _, fc_tuple, metrics = train_prophet(df, target=target, forecast_steps=steps)
    if fc_tuple is not None:
        fc_mean, conf = fc_tuple
        results["prophet"] = {"forecast": fc_mean, "conf": conf, "metrics": metrics}

    # LSTM (optionnel — ne bloque pas si TF absent)
    _, fc, metrics = train_lstm(df, target=target, forecast_steps=steps)
    if fc is not None:
        results["lstm"] = {"forecast": fc, "metrics": metrics}

    print(f"Modèles disponibles : {[k for k in results if 'error' not in results[k]]}")
    return results
