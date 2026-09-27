"""
data_loader.py — BeniMeteo | Béni Mellal-Khénifra
═══════════════════════════════════════════════════
Architecture double source :

  Open-Meteo   → Tableau de bord + prévisions 14 jours (physique atmosphérique)
  Meteostat    → Historique long terme → SARIMA + XGBoost (30/90/180/365 jours)
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Chemins cache local ────────────────────────────────────────────────────────
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(os.path.dirname(_SRC_DIR), "data")
METEOSTAT_CACHE = os.path.join(DATA_DIR, "meteostat_cache.csv")
OPENMETEO_CACHE = os.path.join(DATA_DIR, "openmeteo_cache.csv")


# ── Coordonnées par défaut — Béni Mellal ─────────────────────────────────────
DEFAULT_LAT = 32.337
DEFAULT_LON = -6.349
TIMEZONE    = "Africa/Casablanca"

# Meteostat — bibliothèque Python gratuite (pip install meteostat)

# ── Variables Open-Meteo ──────────────────────────────────────────────────────
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
        df["source"] = "open-meteo"
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


# ══════════════════════════════════════════════════════════════════════════════
# METEOSTAT — Historique long terme pour entraînement ML uniquement
# ══════════════════════════════════════════════════════════════════════════════

def _parse_meteostat_df(raw, start_dt, end_dt):
    """Convertit les données brutes Meteostat en DataFrame standard BeniMeteo."""
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.rename(columns={
        "tavg": "tmean", "tmin": "tmin", "tmax": "tmax",
        "prcp": "precip", "wspd": "wind_max", "wpgt": "wind_gusts",
        "wdir": "wind_dir", "pres": "pressure", "snow": "snowfall",
    })
    for col in ["humidity_max","humidity_min","uv_max","radiation",
                "evapotransp","sunshine_sec","precip_hours"]:
        if col not in df.columns:
            df[col] = np.nan
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="linear", limit=7,
                                            limit_direction="both")
    df["source"] = "meteostat"
    return df


def _fetch_meteostat_lib(lat: float, lon: float, start_dt, end_dt) -> pd.DataFrame:
    """
    Télécharge via la bibliothèque Python 'meteostat' — GRATUIT, SANS CLÉ.
    pip install meteostat
    """
    from meteostat import Point, Daily
    location = Point(lat, lon)
    data = Daily(location, start_dt, end_dt)
    df = data.fetch()
    if df is None or len(df) == 0:
        raise ValueError("Aucune donnée retournée par meteostat")
    # Renommage colonnes → standard BeniMeteo
    df = df.rename(columns={
        "tavg": "tmean", "tmin": "tmin", "tmax": "tmax",
        "prcp": "precip", "wspd": "wind_max", "wpgt": "wind_gusts",
        "wdir": "wind_dir", "pres": "pressure", "snow": "snowfall",
    })
    for col in ["humidity_max","humidity_min","uv_max","radiation",
                "evapotransp","sunshine_sec","precip_hours"]:
        if col not in df.columns:
            df[col] = np.nan
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].interpolate(method="linear", limit=7, limit_direction="both")
    df["source"] = "meteostat-lib"
    return df


def fetch_meteostat_for_ml(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON,
                            years: int = 12) -> pd.DataFrame:
    """
    Récupère l'historique pour les modèles ML.
    Priorité :
      1) Cache local (CSV)          → hors ligne, instant
      2) Bibliothèque meteostat     → pip install meteostat (gratuit, sans clé)
      3) Open-Meteo ERA5            → si bibliothèque meteostat non installée
      4) Cache Open-Meteo local     → si disponible
      5) API Open-Meteo ERA5        → anti-429
      6) Simulation DMN Maroc       → dernier recours
    """
    end_dt   = date.today() - timedelta(days=3)
    start_dt = date(end_dt.year - years, 1, 1)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 1. Cache local ────────────────────────────────────────────────────────
    if os.path.exists(METEOSTAT_CACHE):
        try:
            df_c = pd.read_csv(METEOSTAT_CACHE, sep=";", index_col=0,
                               parse_dates=True, encoding="utf-8-sig")
            if len(df_c) > 365 and "tmax" in df_c.columns:
                cache_end = df_c.index[-1].date()
                days_old  = (date.today() - cache_end).days
                jul = df_c[df_c.index.month == 7]["tmax"].dropna()
                tmax_jul = f"{jul.mean():.1f}°C" if len(jul) > 0 else "—"
                print(f"✅ Cache Meteostat : {len(df_c):,} jours "
                      f"(màj il y a {days_old}j) | Tmax juillet={tmax_jul}")
                df_c["source"] = "meteostat-cache"
                return df_c
        except Exception as e:
            print(f"⚠️  Cache corrompu ({e}) — re-téléchargement")

    # ── 2. Bibliothèque Python meteostat (gratuit, sans clé) ──────────────────
    try:
        from datetime import datetime as _dt
        start_py = _dt(start_dt.year, start_dt.month, start_dt.day)
        end_py   = _dt(end_dt.year,   end_dt.month,   end_dt.day)
        print(f"📥 Tentative bibliothèque meteostat (gratuit, sans clé)...")
        df = _fetch_meteostat_lib(lat, lon, start_py, end_py)
        if len(df) > 365:
            df.to_csv(METEOSTAT_CACHE, sep=";", encoding="utf-8-sig")
            jul = df[df.index.month == 7]["tmax"].dropna()
            print(f"✅ meteostat (lib) : {len(df):,} jours mis en cache")
            if len(jul) > 0:
                print(f"   Tmax juillet : moy={jul.mean():.1f}°C  max={jul.max():.1f}°C")
            return df
    except ImportError:
        print("⚠️  Bibliothèque meteostat non installée → pip install meteostat")
    except Exception as e:
        print(f"⚠️  meteostat lib : {e}")

    # ── 3. Open-Meteo ERA5 (si bibliothèque meteostat non dispo) ────────────────
    print("⚠️  Bibliothèque meteostat non dispo → tentative Open-Meteo ERA5...")

    # ── 4. Cache Open-Meteo local ─────────────────────────────────────────────
    if os.path.exists(OPENMETEO_CACHE):
        try:
            df_om = pd.read_csv(OPENMETEO_CACHE, sep=";", index_col=0,
                                parse_dates=True, encoding="utf-8-sig")
            if len(df_om) > 365:
                print(f"✅ Cache Open-Meteo ERA5 : {len(df_om):,} jours (hors ligne)")
                df_om["source"] = "openmeteo-cache"
                return df_om
        except:
            pass

    # ── 5. API Open-Meteo ERA5 ────────────────────────────────────────────────
    try:
        df_om = fetch_data(str(start_dt), str(end_dt), lat, lon)
        src   = df_om.get("source", pd.Series(["?"])).iloc[0] if hasattr(df_om, "get") else "?"
        if "open-meteo" in str(src):
            print(f"✅ Open-Meteo ERA5 : {len(df_om):,} jours")
            return df_om
    except:
        pass

    # ── 6. Simulation DMN ─────────────────────────────────────────────────────
    print("⚠️  Toutes sources indisponibles — simulation DMN Maroc (Tmax juillet ~43°C)")
    return fetch_data(str(start_dt), str(end_dt), lat, lon)


def fetch_meteostat_all_provinces(years: int = 12) -> dict:
    """
    Télécharge l'historique Meteostat pour les 5 provinces en parallèle
    via la bibliothèque Python (gratuit, sans clé).
    Retourne {province: DataFrame}
    """
    import time
    results = {}
    print(f"📥 Téléchargement historique pour 5 provinces (meteostat lib)...")

    for i, (prov, info) in enumerate(PROVINCE_POINTS.items()):
        if i > 0:
            time.sleep(0.5)
        try:
            df = fetch_meteostat_for_ml(lat=info["lat"], lon=info["lon"], years=years)
            results[prov] = df
            if "tmax" in df.columns:
                jul = df[df.index.month==7]["tmax"].dropna()
                tmax_j = f"{jul.mean():.1f}°C" if len(jul)>0 else "—"
                print(f"  ✅ {prov} : {len(df):,} jours | Tmax juillet={tmax_j}")
        except Exception as e:
            print(f"  ⚠️  {prov} : {e}")
            results[prov] = _fallback(
                str(date(date.today().year-years, 1, 1)),
                str(date.today()))

    return results


def get_ml_training_data(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON,
                          years: int = 12) -> pd.DataFrame:
    """
    Point d'entrée principal pour les modèles ML.
    Essaie Meteostat d'abord, complète avec Open-Meteo si couverture < 70%.
    """
    print("📊 Chargement données ML (Meteostat prioritaire)...")
    df = fetch_meteostat_for_ml(lat, lon, years)

    total    = len(df)
    valid    = df["tmean"].notna().sum() if "tmean" in df.columns else 0
    coverage = valid / total * 100 if total > 0 else 0
    print(f"   Couverture : {coverage:.1f}% ({valid:,}/{total:,} jours)")

    if coverage < 70:
        print("⚠️  Fusion avec Open-Meteo ERA5 pour combler les lacunes...")
        start = str(df.index.min().date()) if len(df) > 0 else f"{date.today().year - years}-01-01"
        df_om = fetch_data(start, str(date.today()), lat, lon)
        for col in ["tmax", "tmin", "tmean", "precip", "wind_max",
                    "humidity_max", "uv_max", "pressure"]:
            if col in df.columns and col in df_om.columns:
                mask = df[col].isna()
                df.loc[mask, col] = df_om.reindex(df.index).loc[mask, col]
        print(f"   ✅ Fusion terminée — {df['tmean'].notna().sum():,} jours valides")

    return df




# ── Climatologie officielle DMN Maroc — Béni Mellal (normales 1991-2020) ─────
# Source : Direction de la Météorologie Nationale Maroc, station Béni Mellal
CLIM_TMAX = {1:17.5,2:19.5,3:23.0,4:27.5,5:32.5,6:38.5,
             7:43.5,8:42.5,9:36.5,10:29.0,11:22.5,12:17.5}
CLIM_TMIN = {1:3.5,2:4.5,3:7.0,4:10.5,5:14.5,6:19.0,
             7:22.5,8:22.0,9:17.5,10:12.0,11:7.0,12:3.5}
CLIM_STD  = {1:3.5,2:4.0,3:4.5,4:5.0,5:5.5,6:4.5,
             7:3.5,8:3.8,9:4.2,10:4.8,11:4.0,12:3.5}


def _fallback(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Simulation basée sur la climatologie officielle DMN Maroc — Béni Mellal.
    Tmax juillet : moy 43.5°C, max possible ~48°C (conforme réalité terrain).
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n     = len(dates)

    tmax_arr, tmin_arr = [], []
    for d in dates:
        m   = d.month
        tr  = (d.year - 2010) * 0.045  # tendance réchauffement
        std = CLIM_STD[m]
        tmax_arr.append(float(np.clip(CLIM_TMAX[m] + tr + np.random.normal(0, std), -5, 49)))
        tmin_arr.append(float(np.clip(CLIM_TMIN[m] + tr + np.random.normal(0, std * 0.7), -8, 30)))

    tmax  = np.array(tmax_arr).round(1)
    tmin  = np.array(tmin_arr).round(1)
    tmean = ((tmax + tmin) / 2).round(1)
    doy   = np.array([d.timetuple().tm_yday for d in dates])

    rain_p = np.clip(0.08 + 0.15 * np.cos((doy - 15) * 2 * np.pi / 365), 0.02, 0.28)
    precip = np.where(np.random.random(n) < rain_p, np.random.exponential(7, n), 0).clip(0, 80).round(1)
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
        "source": "simulation-DMN",
    }, index=dates)


# ── Provinces ─────────────────────────────────────────────────────────────────
PROVINCE_POINTS = {
    "Azilal":         {"lat": 31.967, "lon": -6.572},
    "Béni Mellal":    {"lat": 32.337, "lon": -6.349},
    "Fqih Ben Saleh": {"lat": 32.502, "lon": -6.687},
    "Khénifra":       {"lat": 32.934, "lon": -5.671},
    "Khouribga":      {"lat": 32.886, "lon": -6.906},
}


def fetch_all_provinces(days: int = 7) -> dict:
    """
    Récupère les prévisions pour les 5 provinces.
    Appels SÉQUENTIELS avec délai anti-429 (pas parallèles).
    """
    import time
    results = {}
    for i, (prov, info) in enumerate(PROVINCE_POINTS.items()):
        if i > 0:
            time.sleep(0.8)  # délai anti-429 entre provinces
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={info['lat']}&longitude={info['lon']}"
            f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            f"precipitation_sum,windspeed_10m_max,relative_humidity_2m_max,"
            f"uv_index_max,precipitation_probability_max"
            f"&forecast_days={days}&timezone={TIMEZONE}"
        )
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 429:
                    wait = 3 * (2 ** attempt)
                    print(f"⏳ {prov} rate limit — attente {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                d = r.json().get("daily", {})
                dates_list = d.get("time", [])
                results[prov] = {
                    "province": prov, "lat": info["lat"], "lon": info["lon"],
                    "source": "open-meteo",
                    "forecasts": [{
                        "date":     dates_list[j],
                        "tmax":     round(d.get("temperature_2m_max",    [30]*len(dates_list))[j] or 30, 1),
                        "tmin":     round(d.get("temperature_2m_min",    [15]*len(dates_list))[j] or 15, 1),
                        "temp":     round(d.get("temperature_2m_mean",   [22]*len(dates_list))[j] or 22, 1),
                        "precip":   round(d.get("precipitation_sum",     [0] *len(dates_list))[j] or 0,  1),
                        "wind":     round(d.get("windspeed_10m_max",     [10]*len(dates_list))[j] or 10, 1),
                        "humidity": round(d.get("relative_humidity_2m_max",[50]*len(dates_list))[j] or 50, 1),
                        "uv":       round(d.get("uv_index_max",          [5] *len(dates_list))[j] or 5,  1),
                    } for j in range(len(dates_list))],
                }
                break
            except Exception as e:
                if attempt == 2:
                    print(f"⚠️ {prov} indisponible ({e})")
                    results[prov] = {
                        "province": prov, "lat": info["lat"], "lon": info["lon"],
                        "source": "simulé", "forecasts": [],
                    }
    return results


def get_data_source_info(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> dict:
    """Vérifie la disponibilité des sources météo."""
    import os
    info = {
        "open_meteo": {"disponible": False, "label": "Open-Meteo"},
        "meteostat":  {"disponible": False, "label": "Meteostat lib Python"},
    }
    # Vérifier Open-Meteo
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max&forecast_days=1&timezone=Africa/Casablanca",
            timeout=8)
        info["open_meteo"]["disponible"] = (r.status_code == 200)
    except:
        pass
    # Vérifier meteostat lib
    try:
        import meteostat
        info["meteostat"]["disponible"] = True
        info["meteostat"]["version"] = getattr(meteostat, "__version__", "installé")
    except ImportError:
        info["meteostat"]["disponible"] = False
        info["meteostat"]["label"] = "meteostat (pip install meteostat)"
    # Vérifier cache local
    info["cache_meteostat"] = os.path.exists(METEOSTAT_CACHE)
    info["cache_openmeteo"] = os.path.exists(OPENMETEO_CACHE)
    return info
