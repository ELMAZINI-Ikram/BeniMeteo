"""
demographics.py — Données démographiques HCP douar
Région Béni Mellal-Khénifra : 5 provinces, 119 communes, 2485 douars, ~1.24M habitants
"""

import pandas as pd
import numpy as np
import os
from src.commune_coords import get_commune_coords, PROVINCE_CENTER

HCP_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hcp_douar_bnk.csv")

PROVINCE_INFO = {
    "أزيلال":        {"nom_fr": "Azilal",          "chef_lieu": "Azilal",          "lat": 31.97, "lon": -6.57},
    "بني ملال":      {"nom_fr": "Béni Mellal",      "chef_lieu": "Béni Mellal",     "lat": 32.34, "lon": -6.35},
    "الفقيه بن صالح":{"nom_fr": "Fqih Ben Saleh",   "chef_lieu": "Fqih Ben Saleh",  "lat": 32.50, "lon": -6.68},
    "خنيفرة":        {"nom_fr": "Khénifra",          "chef_lieu": "Khénifra",        "lat": 32.93, "lon": -5.67},
    "خريبكة":        {"nom_fr": "Khouribga",         "chef_lieu": "Khouribga",       "lat": 32.88, "lon": -6.90},
}

# Couleurs thématiques par province
PROVINCE_COLORS = {
    "Azilal":         "#2D8B57",
    "Béni Mellal":    "#2D5A8E",
    "Fqih Ben Saleh": "#C4622D",
    "Khénifra":       "#7BB8D4",
    "Khouribga":      "#FCBF49",
}


def load_hcp_data(path=None):
    """Charge et enrichit le CSV HCP avec coordonnées précises par commune."""
    if path is None:
        path = HCP_CSV_PATH
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={
            "code_geo": "code_geo", "province": "province", "cercle": "cercle",
            "commune": "commune", "douar": "douar", "population": "population",
            "age_m15": "pct_moins_15", "age_15_59": "pct_15_59", "age_60_plus": "pct_60_plus",
        })
        df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
        for col in ["pct_moins_15", "pct_15_59", "pct_60_plus"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["pop_moins_15"]   = (df["population"] * df["pct_moins_15"] / 100).round(0).astype(int)
        df["pop_15_59"]      = (df["population"] * df["pct_15_59"]   / 100).round(0).astype(int)
        df["pop_60_plus"]    = (df["population"] * df["pct_60_plus"]  / 100).round(0).astype(int)
        df["pop_vulnerable"] = df["pop_moins_15"] + df["pop_60_plus"]
        df["pct_vulnerable"] = (df["pop_vulnerable"] / df["population"].replace(0, np.nan) * 100).round(1)

        # Ajout infos province
        df["province_fr"] = df["province"].map(lambda p: PROVINCE_INFO.get(p, {}).get("nom_fr", "Béni Mellal"))

        # Coordonnées précises par commune (pas seulement par province)
        coords = df.apply(lambda r: get_commune_coords(r["commune"], r["province"]), axis=1)
        df["lat"]      = coords.apply(lambda c: c[0])
        df["lon"]      = coords.apply(lambda c: c[1])
        df["altitude"] = coords.apply(lambda c: c[2])

        return df

    except Exception as e:
        print(f"⚠️ Erreur HCP ({e})")
        return _fallback_demo()


def aggregate_by_douar(df):
    """Données brutes par douar."""
    return df.copy()


def aggregate_by_fraction(df):
    """Agrégation par fraction (niveau RGPH 2024 entre commune et douar)."""
    grp = df.groupby(["province", "cercle", "commune", "fraction", "province_fr"]).agg(
        population    =("population",    "sum"),
        pop_moins_15  =("pop_moins_15",  "sum"),
        pop_15_59     =("pop_15_59",     "sum"),
        pop_60_plus   =("pop_60_plus",   "sum"),
        pop_vulnerable=("pop_vulnerable","sum"),
        nb_menages    =("nb_menages",    "sum"),
        nb_douars     =("douar",         "count"),
        lat           =("lat",           "mean"),
        lon           =("lon",           "mean"),
        altitude      =("altitude",      "mean"),
    ).reset_index()
    grp["pct_moins_15"]   = (grp["pop_moins_15"]   / grp["population"].replace(0, np.nan) * 100).round(1)
    grp["pct_60_plus"]    = (grp["pop_60_plus"]    / grp["population"].replace(0, np.nan) * 100).round(1)
    grp["pct_vulnerable"] = (grp["pop_vulnerable"] / grp["population"].replace(0, np.nan) * 100).round(1)
    return grp


def aggregate_by_commune(df):
    """Agrégation par commune avec coords moyennées."""
    grp = df.groupby(["province", "cercle", "commune", "province_fr"]).agg(
        population   =("population",    "sum"),
        pop_moins_15 =("pop_moins_15",  "sum"),
        pop_15_59    =("pop_15_59",     "sum"),
        pop_60_plus  =("pop_60_plus",   "sum"),
        pop_vulnerable=("pop_vulnerable","sum"),
        nb_douars    =("douar",         "count"),
        lat          =("lat",           "mean"),
        lon          =("lon",           "mean"),
        altitude     =("altitude",      "mean"),
    ).reset_index()
    grp["pct_moins_15"]  = (grp["pop_moins_15"]  / grp["population"].replace(0, np.nan) * 100).round(1)
    grp["pct_60_plus"]   = (grp["pop_60_plus"]   / grp["population"].replace(0, np.nan) * 100).round(1)
    grp["pct_vulnerable"]= (grp["pop_vulnerable"] / grp["population"].replace(0, np.nan) * 100).round(1)
    return grp


def aggregate_by_cercle(df):
    """Agrégation par cercle."""
    grp = df.groupby(["province", "cercle", "province_fr"]).agg(
        population   =("population",    "sum"),
        pop_moins_15 =("pop_moins_15",  "sum"),
        pop_60_plus  =("pop_60_plus",   "sum"),
        pop_vulnerable=("pop_vulnerable","sum"),
        nb_communes  =("commune",       "nunique"),
        nb_douars    =("douar",         "count"),
        lat          =("lat",           "mean"),
        lon          =("lon",           "mean"),
        altitude     =("altitude",      "mean"),
    ).reset_index()
    grp["pct_vulnerable"] = (grp["pop_vulnerable"] / grp["population"].replace(0, np.nan) * 100).round(1)
    grp["pct_moins_15"]   = (grp["pop_moins_15"]   / grp["population"].replace(0, np.nan) * 100).round(1)
    grp["pct_60_plus"]    = (grp["pop_60_plus"]    / grp["population"].replace(0, np.nan) * 100).round(1)
    return grp


def aggregate_by_province(df):
    """Agrégation par province."""
    grp = df.groupby(["province", "province_fr"]).agg(
        population   =("population",    "sum"),
        pop_moins_15 =("pop_moins_15",  "sum"),
        pop_15_59    =("pop_15_59",     "sum"),
        pop_60_plus  =("pop_60_plus",   "sum"),
        pop_vulnerable=("pop_vulnerable","sum"),
        nb_communes  =("commune",       "nunique"),
        nb_douars    =("douar",         "count"),
    ).reset_index()
    grp["pct_moins_15"]   = (grp["pop_moins_15"]  / grp["population"] * 100).round(1)
    grp["pct_60_plus"]    = (grp["pop_60_plus"]   / grp["population"] * 100).round(1)
    grp["pct_vulnerable"] = (grp["pop_vulnerable"] / grp["population"] * 100).round(1)
    # Ajouter coords centre province
    grp["lat"] = grp["province"].map(lambda p: PROVINCE_INFO.get(p, {}).get("lat", 32.337))
    grp["lon"] = grp["province"].map(lambda p: PROVINCE_INFO.get(p, {}).get("lon", -6.349))
    return grp


def get_top_vulnerable_zones(df, n=20, level="commune"):
    """Top N zones avec le plus de personnes vulnérables."""
    if level == "commune":
        df_agg = aggregate_by_commune(df)
    elif level == "cercle":
        df_agg = aggregate_by_cercle(df)
    elif level == "province":
        df_agg = aggregate_by_province(df)
    else:
        df_agg = df.copy()
    return df_agg.nlargest(n, "pop_vulnerable").reset_index(drop=True)


def compute_vulnerability_score_hcp(row, conditions):
    """Score 0-100 basé sur démographie HCP réelle + conditions météo."""
    pop = row.get("population", 1)
    if pop == 0:
        return 0.0
    pct_60    = float(row.get("pct_60_plus", 0))
    pct_moins15 = float(row.get("pct_moins_15", 0))
    alt       = float(row.get("altitude", 500))
    pct_vuln  = pct_60 + pct_moins15 * 0.6
    score     = 0.0

    temp     = conditions.get("temp", 20.0)
    precip   = conditions.get("precip", 0.0)
    wind     = conditions.get("wind", 10.0)
    humidity = conditions.get("humidity", 50.0)
    uv       = conditions.get("uv", 5.0)

    demo_w = pct_vuln / 40

    # Chaleur (modulée par altitude: zones basses = plus chaud)
    alt_factor = max(0.5, 1.0 - alt / 3000)
    if temp >= 42:   score += 40 * demo_w * alt_factor
    elif temp >= 38: score += 25 * demo_w * alt_factor
    elif temp >= 35: score += 12 * demo_w * alt_factor

    # Froid / gel (zones haute altitude)
    if temp <= 0 and alt > 1200:
        score += 15 * demo_w
    elif temp <= 5 and alt > 1500:
        score += 10 * demo_w

    # Humidex
    if temp >= 32 and humidity >= 70:
        score += 15 * demo_w

    # Précipitations (altitude = risque crues/torrents)
    flood_factor = 1.0 + alt / 2000
    if precip >= 40:   score += 20 * flood_factor
    elif precip >= 20: score += 10

    # Vent
    if wind >= 70:   score += 15
    elif wind >= 50: score += 7

    # UV
    if uv >= 8: score += 8 * demo_w

    return min(round(score, 1), 100.0)


def build_region_geojson(df, conditions):
    """GeoJSON FeatureCollection toute la région par commune."""
    comm = aggregate_by_commune(df)
    features = []
    for _, row in comm.iterrows():
        sc = compute_vulnerability_score_hcp(row.to_dict(), conditions)
        if sc >= 60:   niveau = "ROUGE"
        elif sc >= 35: niveau = "ORANGE"
        elif sc >= 15: niveau = "JAUNE"
        else:          niveau = "VERT"

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": {
                "commune":       row.get("commune", ""),
                "province":      row.get("province_fr", ""),
                "cercle":        row.get("cercle", ""),
                "population":    int(row["population"]),
                "pop_moins_15":  int(row["pop_moins_15"]),
                "pop_60_plus":   int(row["pop_60_plus"]),
                "pop_vulnerable":int(row["pop_vulnerable"]),
                "pct_vulnerable":float(row.get("pct_vulnerable", 0)),
                "altitude":      int(row.get("altitude", 0)),
                "score_vulnerabilite": float(sc),
                "niveau_alerte": niveau,
                "nb_douars":     int(row["nb_douars"]),
            }
        })
    return {"type": "FeatureCollection", "features": features,
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}}


def _fallback_demo():
    return pd.DataFrame({
        "province": ["بني ملال"], "province_fr": ["Béni Mellal"],
        "cercle": ["Centre"], "commune": ["Béni Mellal"], "douar": ["Centre"],
        "population": [45000], "pct_moins_15": [28.0], "pct_15_59": [60.0], "pct_60_plus": [12.0],
        "pop_moins_15": [12600], "pop_15_59": [27000], "pop_60_plus": [5400],
        "pop_vulnerable": [18000], "pct_vulnerable": [40.0],
        "lat": [32.337], "lon": [-6.349], "altitude": [480],
    })
