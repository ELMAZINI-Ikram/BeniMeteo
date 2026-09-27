"""
alerts.py — Alertes météo avec hiérarchie complète
Région → Province → Cercle → Commune → Douar
"""

import pandas as pd
import numpy as np
import json


def get_zones_hierarchy(hcp_df: pd.DataFrame) -> dict:
    """
    Construit la hiérarchie complète :
    { province: { cercle: { commune: [douars] } } }
    avec scores de vulnérabilité à chaque niveau.
    """
    hierarchy = {}
    for prov_fr in hcp_df["province_fr"].unique():
        df_p = hcp_df[hcp_df["province_fr"] == prov_fr]
        hierarchy[prov_fr] = {
            "population":    int(df_p["population"].sum()),
            "pop_vulnerable":int(df_p["pop_vulnerable"].sum()),
            "pct_moins_15":  float(df_p["pop_moins_15"].sum() / df_p["population"].sum() * 100),
            "pct_60_plus":   float(df_p["pop_60_plus"].sum()  / df_p["population"].sum() * 100),
            "lat": float(df_p["lat"].mean()),
            "lon": float(df_p["lon"].mean()),
            "cercles": {}
        }
        for cercle in df_p["cercle"].unique():
            df_c = df_p[df_p["cercle"] == cercle]
            hierarchy[prov_fr]["cercles"][cercle] = {
                "population":    int(df_c["population"].sum()),
                "pop_vulnerable":int(df_c["pop_vulnerable"].sum()),
                "pct_moins_15":  float(df_c["pop_moins_15"].sum() / df_c["population"].sum() * 100),
                "pct_60_plus":   float(df_c["pop_60_plus"].sum()  / df_c["population"].sum() * 100),
                "lat": float(df_c["lat"].mean()),
                "lon": float(df_c["lon"].mean()),
                "altitude": float(df_c["altitude"].mean()),
                "nb_communes": int(df_c["commune"].nunique()),
                "nb_douars":   int(len(df_c)),
                "communes": {}
            }
            for commune in df_c["commune"].unique():
                df_comm = df_c[df_c["commune"] == commune]
                pop_comm = int(df_comm["population"].sum())
                hierarchy[prov_fr]["cercles"][cercle]["communes"][commune] = {
                    "population":    pop_comm,
                    "pop_vulnerable":int(df_comm["pop_vulnerable"].sum()),
                    "pop_moins_15":  int(df_comm["pop_moins_15"].sum()),
                    "pop_60_plus":   int(df_comm["pop_60_plus"].sum()),
                    "pct_moins_15":  float(df_comm["pop_moins_15"].sum() / pop_comm * 100) if pop_comm else 0,
                    "pct_60_plus":   float(df_comm["pop_60_plus"].sum()  / pop_comm * 100) if pop_comm else 0,
                    "lat": float(df_comm["lat"].mean()),
                    "lon": float(df_comm["lon"].mean()),
                    "altitude": float(df_comm["altitude"].mean()),
                    "nb_douars": int(len(df_comm)),
                    "douars": [
                        {
                            "nom":       str(row["douar"]),
                            "population":int(row["population"]),
                            "pop_vulnerable": int(row["pop_vulnerable"]),
                            "pct_moins_15": float(row["pct_moins_15"]),
                            "pct_60_plus":  float(row["pct_60_plus"]),
                            "lat": float(row["lat"]),
                            "lon": float(row["lon"]),
                        }
                        for _, row in df_comm.iterrows()
                    ]
                }
    return hierarchy


def compute_score(entity: dict, conditions: dict) -> float:
    """
    Score vulnérabilité 0–100 applicable à tout niveau (province/cercle/commune/douar).
    entity doit avoir: population, pct_moins_15, pct_60_plus, altitude (optionnel)
    """
    pop = entity.get("population", 1)
    if pop == 0:
        return 0.0

    pct_60    = float(entity.get("pct_60_plus", 0))
    pct_m15   = float(entity.get("pct_moins_15", 0))
    altitude  = float(entity.get("altitude", 500))
    pct_vuln  = pct_60 + pct_m15 * 0.6

    temp     = conditions.get("temp", 20.0)
    precip   = conditions.get("precip", 0.0)
    wind     = conditions.get("wind", 10.0)
    humidity = conditions.get("humidity", 50.0)
    uv       = conditions.get("uv", 5.0)

    score = 0.0
    demo_w    = pct_vuln / 40  # normalisé
    alt_factor = max(0.5, 1.0 - altitude / 3000)

    # Chaleur (modulée par altitude)
    if temp >= 42:
        score += 40 * demo_w * alt_factor
    elif temp >= 38:
        score += 25 * demo_w * alt_factor
    elif temp >= 35:
        score += 12 * demo_w * alt_factor

    # Froid extrême (zones Atlas > 1200m)
    if temp <= 0 and altitude > 1200:
        score += 20 * demo_w
    elif temp <= 5 and altitude > 1500:
        score += 12 * demo_w

    # Humidex
    if temp >= 32 and humidity >= 70:
        score += 15 * demo_w

    # Précipitations
    alt_flood = 1.0 + altitude / 2000
    if precip >= 40:
        score += 20 * alt_flood
    elif precip >= 20:
        score += 10

    # Vent
    if wind >= 70:
        score += 15
    elif wind >= 50:
        score += 7

    # UV
    if uv >= 8:
        score += 8 * demo_w

    return min(round(score, 1), 100.0)


def generate_alerts(entity: dict, conditions: dict, level: str = "commune") -> list:
    """Génère la liste d'alertes textuelles pour une entité."""
    alerts = []
    temp     = conditions.get("temp", 20.0)
    precip   = conditions.get("precip", 0.0)
    wind     = conditions.get("wind", 10.0)
    humidity = conditions.get("humidity", 50.0)
    uv       = conditions.get("uv", 5.0)
    altitude = float(entity.get("altitude", 500))
    vp       = entity.get("pop_vulnerable", 0)
    pop      = entity.get("population", 0)

    if temp >= 42:
        alerts.append(("🌡️ Canicule critique",
            f"{vp:,} personnes vulnérables exposées — hydratation et évacuation urgente", "ROUGE"))
    elif temp >= 38:
        alerts.append(("🌡️ Forte chaleur",
            f"Vigilance pour {vp:,} personnes fragiles (âgées + enfants)", "ORANGE"))
    elif temp >= 35:
        alerts.append(("🌡️ Chaleur élevée",
            "Précautions pour nourrissons et personnes âgées, éviter sortie 11h–16h", "JAUNE"))

    if temp <= 0 and altitude > 1200:
        alerts.append(("❄️ Gel / Froid extrême",
            f"Altitude {int(altitude)}m — risque hypothermie pour {vp:,} personnes, routes enneigées", "ROUGE"))
    elif temp <= 5 and altitude > 1500:
        alerts.append(("❄️ Froid intense Atlas",
            f"Zone montagneuse {int(altitude)}m — chauffage et approvisionnement à assurer", "ORANGE"))

    if temp >= 32 and humidity >= 70:
        alerts.append(("💧 Humidex élevé",
            "Sensation thermique dangereuse — éviter les efforts en plein air", "ORANGE"))

    if precip >= 40:
        risk = "élevé" if altitude > 800 else "très élevé"
        alerts.append(("🌊 Risque inondation / crues",
            f"Pluies intenses — risque {risk} (ruissellement, oueds) — {int(pop * 0.3):,} personnes à risque", "ROUGE"))
    elif precip >= 20:
        alerts.append(("🌧️ Pluie intense",
            "Surveiller les bas-fonds et oueds, routes potentiellement coupées", "ORANGE"))

    if wind >= 70:
        alerts.append(("💨 Vents violents",
            f"Danger chutes d'objets — personnes âgées et enfants à l'intérieur", "ROUGE"))
    elif wind >= 50:
        alerts.append(("💨 Vent fort",
            "Difficultés déplacement pour personnes âgées et à mobilité réduite", "ORANGE"))

    if uv >= 8:
        alerts.append(("☀️ UV Extrême",
            "Éviter exposition 11h–16h, nourrissons et enfants absolument à l'ombre", "ORANGE"))

    return alerts


def alert_level(score: float) -> tuple:
    """Retourne (label, emoji, css_class)."""
    if score >= 60:
        return "ROUGE",  "🔴", "alert-rouge"
    elif score >= 35:
        return "ORANGE", "🟠", "alert-orange"
    elif score >= 15:
        return "JAUNE",  "🟡", "alert-jaune"
    else:
        return "VERT",   "🟢", "alert-vert"


def get_all_communes_with_scores(hcp_df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """
    Retourne un DataFrame de toutes les communes avec scores calculés.
    Utilisé pour la carte et les tableaux.
    """
    from src.demographics import aggregate_by_commune
    comm = aggregate_by_commune(hcp_df)
    comm["score"]         = comm.apply(lambda r: compute_score(r.to_dict(), conditions), axis=1)
    comm["niveau_alerte"] = comm["score"].apply(lambda s: alert_level(s)[0])
    comm["emoji"]         = comm["score"].apply(lambda s: alert_level(s)[1])
    return comm.sort_values("score", ascending=False).reset_index(drop=True)


def get_all_douars_with_scores(hcp_df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Retourne tous les douars avec scores."""
    df = hcp_df.copy()
    df["score"]         = df.apply(lambda r: compute_score(r.to_dict(), conditions), axis=1)
    df["niveau_alerte"] = df["score"].apply(lambda s: alert_level(s)[0])
    df["emoji"]         = df["score"].apply(lambda s: alert_level(s)[1])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def hierarchy_to_geojson(hcp_df: pd.DataFrame, conditions: dict, level: str = "commune") -> dict:
    """
    Génère GeoJSON pour QGIS à n'importe quel niveau de la hiérarchie.
    level: 'douar' | 'commune' | 'cercle' | 'province'
    """
    from src.demographics import aggregate_by_commune, aggregate_by_cercle, aggregate_by_province

    if level == "douar":
        df_agg = hcp_df.copy()
        name_col = "douar"
    elif level == "commune":
        df_agg = aggregate_by_commune(hcp_df)
        name_col = "commune"
    elif level == "cercle":
        df_agg = aggregate_by_cercle(hcp_df)
        name_col = "cercle"
    else:
        df_agg = aggregate_by_province(hcp_df)
        name_col = "province_fr"

    features = []
    for _, row in df_agg.iterrows():
        sc  = compute_score(row.to_dict(), conditions)
        lvl = alert_level(sc)[0]
        prop = {
            "nom":           str(row.get(name_col, "")),
            "province":      str(row.get("province_fr", "")),
            "cercle":        str(row.get("cercle", "")),
            "commune":       str(row.get("commune", "")),
            "niveau":        level,
            "population":    int(row.get("population", 0)),
            "pop_vulnerable":int(row.get("pop_vulnerable", 0)),
            "pop_moins_15":  int(row.get("pop_moins_15", 0)),
            "pop_60_plus":   int(row.get("pop_60_plus", 0)),
            "pct_moins_15":  float(row.get("pct_moins_15", 0)),
            "pct_60_plus":   float(row.get("pct_60_plus", 0)),
            "altitude":      int(row.get("altitude", 0)),
            "score_vulnerabilite": float(sc),
            "niveau_alerte": lvl,
            "nb_douars":     int(row.get("nb_douars", 1)),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": prop
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}
    }


# ── Compatibilité avec l'ancien code ─────────────────────────────────────────
def get_zones(hcp_df=None):
    """Retourne un DataFrame de communes (compatibilité app.py)."""
    if hcp_df is not None:
        from src.demographics import aggregate_by_commune
        comm = aggregate_by_commune(hcp_df)
        comm["id"]                = comm["commune"]
        comm["nom"]               = comm["commune"]
        comm["vieux_pct"]         = comm["pct_60_plus"]
        comm["bebes_pct"]         = (comm["pct_moins_15"] * 0.15).round(1)
        comm["enfants_pct"]       = (comm["pct_moins_15"] * 0.85).round(1)
        comm["handicap_pct"]      = 3.5
        comm["maladies_chron_pct"]= 12.0
        comm["espaces_verts_pct"] = 10.0
        comm["drainage"]          = "Moyen"
        comm["risque_inondation"] = comm["altitude"].apply(
            lambda a: "Élevé" if a > 800 else "Moyen" if a > 400 else "Très Élevé")
        comm["risque_chaleur"]    = comm["altitude"].apply(
            lambda a: "Faible" if a > 1200 else "Élevé" if a < 500 else "Moyen")
        comm["acces_sante"]       = "Moyen"
        return comm
    return pd.DataFrame()


def compute_vulnerability_score(zone: dict, conditions: dict) -> tuple:
    """Compatibilité — retourne (score, pop_vulnerable, alerts)."""
    score  = compute_score(zone, conditions)
    vp     = int(zone.get("pop_vulnerable", 0))
    alerts = generate_alerts(zone, conditions)
    return score, vp, alerts


def zones_to_geojson(df_zones: pd.DataFrame, conditions: dict) -> dict:
    """Compatibilité export QGIS."""
    features = []
    for _, z in df_zones.iterrows():
        sc, vp, _ = compute_vulnerability_score(z.to_dict(), conditions)
        lvl       = alert_level(sc)[0]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(z["lon"]), float(z["lat"])]},
            "properties": {
                "nom": str(z.get("nom", z.get("commune", ""))),
                "province": str(z.get("province_fr", "")),
                "cercle": str(z.get("cercle", "")),
                "population": int(z.get("population", 0)),
                "pop_vulnerable": int(vp),
                "score_vulnerabilite": float(sc),
                "niveau_alerte": lvl,
            }
        })
    return {"type": "FeatureCollection", "features": features,
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}}
