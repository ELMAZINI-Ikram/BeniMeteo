"""
meteo_multipoints.py — Météo multi-points par province
Appel Open-Meteo pour chaque province au lieu d'un seul point central
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEZONE = "Africa/Casablanca"

# Coordonnées représentatives par province
PROVINCE_POINTS = {
    "Azilal":          {"lat": 31.967, "lon": -6.572, "altitude": 1200},
    "Béni Mellal":     {"lat": 32.337, "lon": -6.349, "altitude": 480},
    "Fqih Ben Saleh":  {"lat": 32.502, "lon": -6.687, "altitude": 400},
    "Khénifra":        {"lat": 32.934, "lon": -5.671, "altitude": 900},
    "Khouribga":       {"lat": 32.886, "lon": -6.906, "altitude": 600},
}


def _fetch_one_province(province: str, info: dict, days: int = 7) -> dict:
    """Récupère les prévisions pour une province."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,windspeed_10m_max,relative_humidity_2m_max,"
        f"uv_index_max,precipitation_probability_max"
        f"&forecast_days={days}&timezone={TIMEZONE}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        d = r.json().get("daily", {})
        dates = d.get("time", [])
        return {
            "province":  province,
            "lat":       info["lat"],
            "lon":       info["lon"],
            "altitude":  info["altitude"],
            "forecasts": [
                {
                    "date":     dates[i],
                    "tmax":     round(d.get("temperature_2m_max",    [25]*len(dates))[i] or 25, 1),
                    "tmin":     round(d.get("temperature_2m_min",    [15]*len(dates))[i] or 15, 1),
                    "temp":     round(d.get("temperature_2m_mean",   [20]*len(dates))[i] or 20, 1),
                    "precip":   round(d.get("precipitation_sum",     [0] *len(dates))[i] or 0,  1),
                    "wind":     round(d.get("windspeed_10m_max",     [10]*len(dates))[i] or 10, 1),
                    "humidity": round(d.get("relative_humidity_2m_max",[50]*len(dates))[i] or 50, 1),
                    "uv":       round(d.get("uv_index_max",          [5] *len(dates))[i] or 5,  1),
                    "precip_prob": round(d.get("precipitation_probability_max",[0]*len(dates))[i] or 0, 0),
                }
                for i in range(len(dates))
            ],
            "source": "open-meteo",
        }
    except Exception as e:
        print(f"⚠️ Météo {province} indisponible ({e})")
        return _fallback_province(province, info, days)


def _fallback_province(province: str, info: dict, days: int) -> dict:
    """Données simulées si l'API est inaccessible."""
    today = date.today()
    doy   = today.timetuple().tm_yday
    alt   = info.get("altitude", 500)
    base  = 24 + 14 * np.sin((doy - 100) * 2 * 3.14159 / 365) - alt * 0.006
    return {
        "province": province, "lat": info["lat"], "lon": info["lon"],
        "altitude": alt, "source": "simulé",
        "forecasts": [
            {
                "date": str(today + timedelta(days=i)),
                "tmax": round(base + 6 + np.random.normal(0, 1.5), 1),
                "tmin": round(base - 6 + np.random.normal(0, 1.2), 1),
                "temp": round(base + np.random.normal(0, 1), 1),
                "precip": 0, "wind": 15, "humidity": 50, "uv": 6,
                "precip_prob": 10,
            }
            for i in range(days)
        ],
    }


def fetch_all_provinces(days: int = 7, max_workers: int = 5) -> dict:
    """
    Récupère les prévisions météo pour les 5 provinces en parallèle.
    Retourne un dict {province: {lat, lon, forecasts, ...}}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_province, prov, info, days): prov
            for prov, info in PROVINCE_POINTS.items()
        }
        for future in as_completed(futures):
            prov = futures[future]
            try:
                results[prov] = future.result()
            except Exception as e:
                results[prov] = _fallback_province(prov, PROVINCE_POINTS[prov], days)

    print(f"✅ Météo chargée pour {len(results)} provinces")
    return results


def get_province_conditions_today(province_data: dict, target_date: str = None) -> dict:
    """
    Extrait les conditions d'aujourd'hui pour chaque province.
    Retourne {province: conditions_dict}
    """
    if target_date is None:
        target_date = str(date.today())

    conditions_by_province = {}
    for prov, data in province_data.items():
        forecasts = data.get("forecasts", [])
        # Chercher la date cible ou prendre la première
        cond = next(
            (f for f in forecasts if f.get("date") == target_date),
            forecasts[0] if forecasts else {}
        )
        conditions_by_province[prov] = {
            **cond,
            "province": prov,
            "lat":      data.get("lat", 32.337),
            "lon":      data.get("lon", -6.349),
            "altitude": data.get("altitude", 500),
            "source":   data.get("source", "open-meteo"),
        }
    return conditions_by_province


def compute_regional_scores_multipoint(hcp_df: pd.DataFrame,
                                        province_data: dict,
                                        target_date: str = None) -> pd.DataFrame:
    """
    Calcule les scores de vulnérabilité avec la météo précise de chaque province.
    Beaucoup plus précis qu'un seul point central.
    """
    from src.alert_engine import (compute_score_v2, alert_level_v2,
                                   compute_spi, get_seasonal_context,
                                   fetch_hopitaux_osm, HOPITAUX_REGION)
    from src.demographics import aggregate_by_commune

    conditions_prov = get_province_conditions_today(province_data, target_date)
    seasonal        = get_seasonal_context()

    # Hôpitaux (on utilise la liste locale comme fallback rapide)
    hopitaux = HOPITAUX_REGION

    communes = aggregate_by_commune(hcp_df)
    scores   = []

    for _, comm in communes.iterrows():
        prov_fr = comm.get("province_fr", "Béni Mellal")
        # Météo spécifique à la province de cette commune
        cond = conditions_prov.get(prov_fr, list(conditions_prov.values())[0])
        sc   = compute_score_v2(comm.to_dict(), cond, hopitaux, 0.0, seasonal)
        lvl, em, _ = alert_level_v2(sc)
        scores.append({
            **comm.to_dict(),
            "score":          sc,
            "niveau_alerte":  lvl,
            "emoji":          em,
            "tmax":           cond.get("tmax", 25),
            "precip":         cond.get("precip", 0),
            "wind":           cond.get("wind", 10),
            "source_meteo":   cond.get("source", "open-meteo"),
        })

    df = pd.DataFrame(scores).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"✅ Scores multi-points : {len(df)} communes | "
          f"ROUGE={int((df['niveau_alerte']=='ROUGE').sum())} | "
          f"ORANGE={int((df['niveau_alerte']=='ORANGE').sum())}")
    return df


def get_province_summary(province_data: dict) -> pd.DataFrame:
    """Résumé météo par province pour affichage tableau de bord."""
    rows = []
    for prov, data in province_data.items():
        fc = data.get("forecasts", [{}])
        today = fc[0] if fc else {}
        rows.append({
            "province":  prov,
            "tmax":      today.get("tmax", "—"),
            "tmin":      today.get("tmin", "—"),
            "temp":      today.get("temp", "—"),
            "precip":    today.get("precip", 0),
            "wind":      today.get("wind", "—"),
            "humidity":  today.get("humidity", "—"),
            "uv":        today.get("uv", "—"),
            "source":    data.get("source", "open-meteo"),
            "altitude":  data.get("altitude", 0),
        })
    return pd.DataFrame(rows)
