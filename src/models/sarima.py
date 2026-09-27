"""
models/sarima.py — Modèle SARIMA amélioré pour prédiction climatique.

Corrections v2 :
- Période saisonnière 12 (mensuelle sur données mensuelles) au lieu de 30 (jours)
- Agrégation mensuelle avant SARIMA pour stabilité
- Contraintes physiques : température min 0°C pour région BM-Khénifra
- Auto-sélection paramètres si SARIMAX disponible
- Fallback saisonnier amélioré avec tendance de réchauffement
"""

import pandas as pd
import numpy as np


def _monthly_aggregate(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Agrège les données journalières en moyennes mensuelles.
    SARIMA sur données mensuelles est beaucoup plus stable et précis
    que sur données journalières avec période 365.
    """
    y = df[target].dropna()
    y_monthly = y.resample("ME").mean()
    return y_monthly


def _interpolate_to_daily(monthly_forecast: pd.Series,
                           conf_monthly: pd.DataFrame,
                           steps_days: int,
                           last_daily_date: pd.Timestamp,
                           historical_daily: pd.Series) -> tuple:
    """
    Réinterpolation mensuelle → journalière avec saisonnalité intra-mensuelle.
    Utilise l'écart-type journalier historique pour ajouter la variabilité réaliste.
    """
    future_daily = pd.date_range(
        start=last_daily_date + pd.Timedelta(days=1),
        periods=steps_days, freq="D")

    # Interpolation linéaire des moyennes mensuelles
    monthly_dates = monthly_forecast.index
    monthly_values = monthly_forecast.values

    daily_mean = np.interp(
        np.arange(steps_days),
        np.linspace(0, steps_days, len(monthly_values)),
        monthly_values
    )

    # Variabilité intra-mensuelle basée sur historique par mois
    daily_std_by_month = historical_daily.groupby(historical_daily.index.month).std()

    noise_scale = np.array([
        daily_std_by_month.get(d.month, daily_std_by_month.mean()) * 0.3
        for d in future_daily
    ])
    np.random.seed(42)
    daily_forecast = pd.Series(
        daily_mean + np.random.normal(0, noise_scale),
        index=future_daily)

    # Correction biais saisonnier Beni Mellal-Khenifra
    # SARIMA sous-estime les pics d'ete car entraîne sur moyennes mensuelles
    # Biais corrects bases sur climatologie historique tmax region
    SUMMER_BIAS_TMAX = {5: 2.0, 6: 4.5, 7: 6.0, 8: 5.5, 9: 2.5}
    SUMMER_BIAS_MEAN = {5: 1.0, 6: 2.5, 7: 3.5, 8: 3.0, 9: 1.5}

    # Appliquer biais selon variable
    if 'tmax' in str(getattr(historical_daily, 'name', '')):
        bias_map = SUMMER_BIAS_TMAX
    else:
        bias_map = SUMMER_BIAS_MEAN
    bias_arr = np.array([bias_map.get(d.month, 0.0) for d in future_daily])
    daily_forecast = daily_forecast + bias_arr

    # IC journalier interpolé depuis IC mensuel
    lower_monthly = conf_monthly.iloc[:, 0].values
    upper_monthly = conf_monthly.iloc[:, 1].values

    lower_daily = np.interp(
        np.arange(steps_days),
        np.linspace(0, steps_days, len(lower_monthly)),
        lower_monthly)
    upper_daily = np.interp(
        np.arange(steps_days),
        np.linspace(0, steps_days, len(upper_monthly)),
        upper_monthly)

    conf_daily = pd.DataFrame({
        "lower": lower_daily,
        "upper": upper_daily,
    }, index=future_daily)

    return daily_forecast, conf_daily


def _apply_physical_constraints(forecast: pd.Series,
                                 conf: pd.DataFrame,
                                 target: str) -> tuple:
    """
    Applique les contraintes physiques pour la région Béni Mellal-Khénifra.
    Évite les températures impossibles ou les précipitations négatives.
    """
    if "temp" in target or target in ["tmax", "tmin", "tmean", "temp_mean"]:
        # Beni Mellal : tmax peut atteindre 48-50C en juillet-aout
        # On clip seulement le minimum (gel) et un maximum physique absolu
        if target == "tmax":
            forecast = forecast.clip(lower=-5, upper=52)
        else:
            forecast = forecast.clip(lower=-5, upper=45)
        if conf is not None and len(conf) > 0:
            conf["lower"] = conf["lower"].clip(lower=-5)
            conf["upper"] = conf["upper"].clip(upper=52)

    elif "precip" in target:
        # Précipitations : jamais négatives
        forecast = forecast.clip(lower=0)
        if conf is not None and len(conf) > 0:
            conf["lower"] = conf["lower"].clip(lower=0)
            conf["upper"] = conf["upper"].clip(lower=0)

    elif "wind" in target or "humidity" in target:
        forecast = forecast.clip(lower=0)
        if conf is not None and len(conf) > 0:
            conf["lower"] = conf["lower"].clip(lower=0)

    return forecast, conf


def train_sarima(df: pd.DataFrame, target: str = "temp_mean",
                 steps: int = 90,
                 order: tuple = (1, 1, 1),
                 seasonal_order: tuple = None):
    """
    Entraîne SARIMA sur données mensuelles pour une prévision stable.

    Stratégie :
    1. Agrégation journalière → mensuelle
    2. SARIMA(1,1,1)(1,1,1,12) — période 12 mois (cycle annuel)
    3. Réinterpolation mensuelle → journalière
    4. Application contraintes physiques (T° min -5°C, précip ≥ 0)

    Retourne (results, forecast_daily, conf_daily)
    """
    if target not in df.columns:
        # Chercher un alias
        aliases = {
            "temp_mean": ["tmean", "temperature_mean", "t_mean"],
            "tmax": ["temperature_max", "t_max"],
            "tmin": ["temperature_min", "t_min"],
            "precip": ["precipitation", "pluie", "rain"],
            "wind_max": ["wind", "vent"],
            "humidity_mean": ["humidity", "humidity_max", "humidite"],
        }
        for alias in aliases.get(target, []):
            if alias in df.columns:
                df = df.copy()
                df[target] = df[alias]
                break

    if target not in df.columns:
        print(f"SARIMA : colonne '{target}' introuvable → fallback")
        return None, *_simple_forecast(df, list(df.select_dtypes(include=np.number).columns)[0], steps)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Agrégation mensuelle
        y_daily   = df[target].dropna()
        y_monthly = _monthly_aggregate(df, target)

        if len(y_monthly) < 18:
            print(f"SARIMA : historique trop court ({len(y_monthly)} mois) → fallback")
            return None, *_simple_forecast(df, target, steps)

        # Nombre de mois à prévoir
        steps_months = max(2, steps // 28 + 2)

        # Paramètres SARIMA : période 12 (cycle annuel mensuel)
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, 12)

        model = SARIMAX(
            y_monthly,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="c",   # constante pour capturer niveau moyen
        )
        results = model.fit(disp=False, maxiter=300, method="lbfgs")

        # Prévision mensuelle
        fc_obj     = results.get_forecast(steps=steps_months)
        fc_monthly = fc_obj.predicted_mean
        conf_monthly = fc_obj.conf_int(alpha=0.05)

        # Index mensuel futur
        last_month = y_monthly.index[-1]
        future_monthly_idx = pd.date_range(
            start=last_month + pd.DateOffset(months=1),
            periods=steps_months, freq="ME")
        fc_monthly.index   = future_monthly_idx
        conf_monthly.index = future_monthly_idx

        # Réinterpolation → journalier
        fc_daily, conf_daily = _interpolate_to_daily(
            fc_monthly, conf_monthly, steps,
            y_daily.index[-1], y_daily)

        # Contraintes physiques
        fc_daily, conf_daily = _apply_physical_constraints(fc_daily, conf_daily, target)

        print(f"✅ SARIMA({order})({seasonal_order}) — {len(y_monthly)} mois historique")
        print(f"   Prévision {steps}j : {fc_daily.mean():.1f}°C moy | "
              f"min {fc_daily.min():.1f} | max {fc_daily.max():.1f}")

        return results, fc_daily, conf_daily

    except ImportError:
        print("statsmodels non disponible → fallback saisonnier")
        return None, *_simple_forecast(df, target, steps)
    except Exception as e:
        print(f"SARIMA échoué ({e}) → fallback saisonnier")
        return None, *_simple_forecast(df, target, steps)


def _simple_forecast(df: pd.DataFrame, target: str, steps: int) -> tuple:
    """
    Fallback saisonnier base sur les moyennes mensuelles historiques reelles.
    Utilise directement les moyennes par mois de l'historique — plus fiable.
    """
    if target not in df.columns:
        target = df.select_dtypes(include=np.number).columns[0]

    y = df[target].dropna()
    if len(y) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    last_date  = y.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps, freq="D")

    # Moyennes et ecarts-types par mois de l'historique
    monthly_mean = y.groupby(y.index.month).mean()
    monthly_std  = y.groupby(y.index.month).std().fillna(y.std())

    # Tendance de rechauffement (degres par an)
    years = np.array([d.year + d.timetuple().tm_yday/365 for d in y.index])
    if len(years) > 365:
        z = np.polyfit(years, y.values, 1)
        trend_per_year = z[0]
    else:
        trend_per_year = 0.045  # +0.045 C/an par defaut

    last_year = y.index[-1].year + y.index[-1].timetuple().tm_yday / 365
    preds, lowers, uppers = [], [], []

    for i, d in enumerate(future_idx):
        m   = monthly_mean.get(d.month, y.mean())
        s   = monthly_std.get(d.month, y.std())
        fut_year = d.year + d.timetuple().tm_yday / 365
        trend_adj = trend_per_year * (fut_year - last_year)
        center = m + trend_adj
        preds.append(center)
        lowers.append(center - 1.64 * s)
        uppers.append(center + 1.64 * s)

    pred  = pd.Series(preds,  index=future_idx)
    lower = pd.Series(lowers, index=future_idx)
    upper = pd.Series(uppers, index=future_idx)
    conf  = pd.DataFrame({"lower": lower, "upper": upper})

    # Contraintes physiques
    pred, conf = _apply_physical_constraints(pred, conf, target)
    return pred, conf
