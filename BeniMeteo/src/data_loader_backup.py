"""
data_loader.py — BeniMeteo | Béni Mellal-Khénifra
Récupère données historiques + prévisions via Open-Meteo (gratuit, sans clé API).
Supporte aussi des coordonnées personnalisées pour intégration QGIS.
"""

import requests
import pandas as pd
import numpy as np
from datetime import date


# Coordonnées par défaut — Béni Mellal
DEFAULT_LAT = 32.337
DEFAULT_LON = -6.349
TIMEZONE    = "Africa/Casablanca"

VARIABLES_HISTORIQUES = ",".join([
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "windspeed_10m_max", "windgusts_10m_max",
    "winddirection_10m_dominant",
    "relative_humidity_2m_max", "relative_humidity_2m_min",
    "precipitation_hours", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "sunshine_duration",
    "uv_index_max", "surface_pressure_max", "snowfall_sum",
])


def fetch_data(start_date: str = "2010-01-01", end_date: str = None,
               lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> pd.DataFrame:
    """Données journalières historiques depuis Open-Meteo Archive API."""
    if end_date is None:
        end_date = str(date.today())

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={VARIABLES_HISTORIQUES}&timezone={TIMEZONE}"
    )

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        d = r.json()["daily"]

        df = pd.DataFrame({
            "date":         d.get("time", []),
            "tmax":         d.get("temperature_2m_max", []),
            "tmin":         d.get("temperature_2m_min", []),
            "tmean":        d.get("temperature_2m_mean", []),
            "precip":       d.get("precipitation_sum", []),
            "wind_max":     d.get("windspeed_10m_max", []),
            "wind_gusts":   d.get("windgusts_10m_max", []),
            "wind_dir":     d.get("winddirection_10m_dominant", []),
            "humidity_max": d.get("relative_humidity_2m_max", []),
            "humidity_min": d.get("relative_humidity_2m_min", []),
            "precip_hours": d.get("precipitation_hours", []),
            "radiation":    d.get("shortwave_radiation_sum", []),
            "evapotransp":  d.get("et0_fao_evapotranspiration", []),
            "sunshine_sec": d.get("sunshine_duration", []),
            "uv_max":       d.get("uv_index_max", []),
            "pressure":     d.get("surface_pressure_max", []),
            "snowfall":     d.get("snowfall_sum", []),
        })
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    except Exception as e:
        print(f"⚠️  API Open-Meteo indisponible ({e}) — données simulées.")
        return _fallback(start_date, end_date)


def fetch_forecast(days: int = 14, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> pd.DataFrame:
    """Prévisions météo pour les prochains jours."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,windspeed_10m_max,winddirection_10m_dominant,"
        f"relative_humidity_2m_max,uv_index_max,precipitation_probability_max,weathercode"
        f"&forecast_days={days}&timezone={TIMEZONE}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date":        d.get("time", []),
            "tmax":        d.get("temperature_2m_max", []),
            "tmin":        d.get("temperature_2m_min", []),
            "tmean":       d.get("temperature_2m_mean", []),
            "precip":      d.get("precipitation_sum", []),
            "wind_max":    d.get("windspeed_10m_max", []),
            "wind_dir":    d.get("winddirection_10m_dominant", []),
            "humidity":    d.get("relative_humidity_2m_max", []),
            "uv_max":      d.get("uv_index_max", []),
            "precip_prob": d.get("precipitation_probability_max", []),
            "wmo_code":    d.get("weathercode", []),
        })
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        print(f"⚠️  Prévisions indisponibles ({e})")
        return pd.DataFrame()


def fetch_multi_location(locations: list) -> dict:
    """
    Charge les données pour plusieurs coordonnées (intégration QGIS).
    locations = [{"nom": "Béni Mellal", "lat": 32.337, "lon": -6.349}, ...]
    Retourne un dict {nom: DataFrame}.
    """
    results = {}
    for loc in locations:
        try:
            df = fetch_data(lat=loc["lat"], lon=loc["lon"])
            results[loc["nom"]] = df
        except Exception as e:
            print(f"Erreur pour {loc['nom']}: {e}")
    return results


def _fallback(start_date: str, end_date: str) -> pd.DataFrame:
    """Données simulées réalistes pour Béni Mellal si API inaccessible."""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n     = len(dates)
    doy   = np.array([d.timetuple().tm_yday for d in dates])
    year  = np.array([d.year for d in dates])
    trend = (year - 2010) * 0.045

    base  = 24 + 14 * np.sin((doy - 100) * 2 * np.pi / 365)
    tmax  = (base + trend + np.random.normal(0, 2.5, n)).round(1)
    tmin  = (base - 12 + trend + np.random.normal(0, 1.8, n)).round(1)
    tmean = ((tmax + tmin) / 2).round(1)

    rain_p = np.clip(0.08 + 0.15 * np.cos((doy - 15) * 2 * np.pi / 365), 0.02, 0.3)
    precip = np.where(np.random.random(n) < rain_p, np.random.exponential(6, n), 0).clip(0, 80).round(1)
    hmax   = np.clip(58 + 18 * np.cos((doy - 30) * 2 * np.pi / 365) + np.random.normal(0, 7, n), 20, 98).round(1)
    hmin   = np.clip(hmax - 20 + np.random.normal(0, 4, n), 10, 90).round(1)
    wind   = np.clip(12 + 8 * np.sin((doy - 60) * 2 * np.pi / 365) + np.random.exponential(4, n), 0, 90).round(1)
    uv     = np.clip(3 + 7 * np.sin((doy - 80) * 2 * np.pi / 365) + np.random.normal(0, 0.8, n), 0, 11).round(1)
    press  = (1013 + 5 * np.cos((doy - 30) * 2 * np.pi / 365) + np.random.normal(0, 2.5, n)).round(1)
    sun_h  = np.clip(6 + 5 * np.sin((doy - 80) * 2 * np.pi / 365) + np.random.normal(0, 1, n), 0, 13)

    return pd.DataFrame({
        "tmax": tmax, "tmin": tmin, "tmean": tmean,
        "precip": precip, "wind_max": wind, "wind_gusts": (wind * 1.4).round(1),
        "wind_dir": np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], n),
        "humidity_max": hmax, "humidity_min": hmin,
        "precip_hours": ((precip > 0) * np.random.uniform(1, 8, n)).round(1),
        "radiation": np.clip(sun_h * 2.5 + np.random.normal(0, 1, n), 0, 30).round(2),
        "evapotransp": np.clip((tmax - tmin) * 0.15 + np.random.normal(0, 0.5, n), 0, 10).round(2),
        "sunshine_sec": (sun_h * 3600).round(0),
        "uv_max": uv, "pressure": press,
        "snowfall": np.where((tmin < 2) & (precip > 0), np.random.uniform(0, 5, n), 0).round(1),
    }, index=dates)
