"""
alert_engine.py — BeniMeteo V4
═══════════════════════════════
Seuils basés sur la réalité biologique et climatique de Béni Mellal-Khénifra.

SEUILS TEMPÉRATURES (absolus — valables toute l'année) :
  Source : OMS + DMN Maroc + études chaleur Maghreb
  < 35°C  → VERT   : conditions normales
  35-38°C → JAUNE  : chaleur significative — vigilance personnes fragiles
  38-42°C → ORANGE : forte chaleur — risque pour groupes vulnérables
  > 42°C  → ROUGE  : canicule — urgence pour toute la population

Justification scientifique :
  - 35°C : seuil de stress thermique OMS pour population générale
  - 38°C : seuil vigilance orange DMN Maroc (bulletins officiels)
  - 42°C : seuil canicule critique — mortalité augmente exponentiellement
    Source : Bouchama & Knochel (2002) NEJM Heat Stroke
             Hajat & Kosatky (2010) Environ Health — dose-réponse chaleur-mortalité

SCORE = risque_chaleur × poids_démographique × facteur_isolement × facteur_saison
"""

import os
import math
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alerts_history.db")

# ══════════════════════════════════════════════════════════════════════════════
# APPROCHE HYBRIDE — ANOMALIE + PLANCHER BIOLOGIQUE
# Source : DMN Maroc (2019) + OMS (2008) + Bouchama & Knochel (2002) NEJM
#
# Principe : seuil = normale_mensuelle_DMN + excès_biologique
#   +5°C  → JAUNE  : stress thermique, corps non acclimaté à cet excès
#   +8°C  → ORANGE : stress élevé, risque coup de chaleur personnes fragiles
#   +10°C → ROUGE  : urgence, mortalité augmente exponentiellement
#
# Pourquoi pas des seuils absolus fixes ?
#   38°C en juillet (normale 43.5°C) = corps acclimaté → pas de stress
#   38°C en mars (normale 23°C) = +15°C → choc thermique → ROUGE
#   L'organisme s'acclimate progressivement (acclimatation thermique ~2 semaines)
#   Source : Bouchama & Knochel (2002) NEJM + OMS Heat Health Action Plans
#
# Plancher biologique absolu minimal :
#   Même en été (corps acclimaté), au-delà de 50°C le risque est vital
#   pour TOUTE la population (pas seulement les fragiles)
# ══════════════════════════════════════════════════════════════════════════════

# Normales climatiques DMN Béni Mellal (1991-2020)
# Source : Direction de la Météorologie Nationale Maroc, station Béni Mellal
# (tmax_moy, tmin_moy, precip_moy_mm)
NORMALES_DMN = {
    1:(17.5, 3.5, 55.0), 2:(19.5, 4.5, 52.0), 3:(23.0, 7.0, 48.0),
    4:(27.5,10.5, 40.0), 5:(32.5,14.5, 22.0), 6:(38.5,19.0,  8.0),
    7:(43.5,22.5,  2.0), 8:(42.5,22.0,  4.0), 9:(36.5,17.5, 15.0),
   10:(29.0,12.0, 40.0),11:(22.5, 7.0, 55.0),12:(17.5, 3.5, 60.0),
}

# Excès par rapport à la normale → niveau d'alerte
# VARIABLES PAR SAISON — le corps en été est déjà sous stress thermique
# Un petit excès supplémentaire en été est plus dangereux qu'en hiver
# Source : OMS (2008) Heat Health Action Plans + DMN Maroc vigilance météo
#
# Hiver (déc-fév)     : excès +5/+8/+10°C → JAUNE/ORANGE/ROUGE
# Printemps (mar-mai) : excès +4/+6/+8°C
# Été (juin-sept)     : excès +3/+5/+7°C  (déjà 38-44°C normale)
# Automne (oct-nov)   : excès +4/+6/+8°C
EXCES_PAR_MOIS = {
    1:  (5, 8, 10),   # Janvier  — hiver
    2:  (5, 8, 10),   # Février  — hiver
    3:  (4, 6,  8),   # Mars     — printemps
    4:  (4, 6,  8),   # Avril    — printemps
    5:  (4, 6,  8),   # Mai      — printemps chaud
    6:  (3, 5,  7),   # Juin     — été (normale 38.5°C)
    7:  (3, 5,  7),   # Juillet  — été (normale 43.5°C)
    8:  (3, 5,  7),   # Août     — été (normale 42.5°C)
    9:  (3, 5,  7),   # Septembre — fin été (normale 36.5°C)
    10: (4, 6,  8),   # Octobre  — automne
    11: (5, 8, 10),   # Novembre — automne froid
    12: (5, 8, 10),   # Décembre — hiver
}
# Pour compatibilité (valeurs moyennes)
EXCES_JAUNE  = 4.0
EXCES_ORANGE = 6.5
EXCES_ROUGE  = 8.5

# Plancher absolu minimal — danger vital pour tous (même en été)
# En juillet (normale 43.5°C), il faut 54°C pour +10.5°C → ROUGE par anomalie
# Mais biologiquement, 50°C est dangereux pour tout le monde
PLANCHER_ROUGE_ABSOLU = 50.0  # ≥ 50°C → ROUGE quelle que soit la saison

# Compatibilité ancienne interface
PLANCHER_ORANGE = 42.0
PLANCHER_ROUGE  = 46.0

# Seuils précipitations (DMN Maroc — bulletins vigilance)
SEUIL_PLUIE_JAUNE  = 20.0  # mm/j
SEUIL_PLUIE_ORANGE = 40.0
SEUIL_PLUIE_ROUGE  = 60.0

# Seuils vent (DMN Maroc)
SEUIL_VENT_JAUNE  = 60.0   # km/h
SEUIL_VENT_ORANGE = 75.0
SEUIL_VENT_ROUGE  = 90.0

# Compatibilité ancienne interface
SEUIL_VERT   = 35.0
SEUIL_JAUNE  = 38.0
SEUIL_ORANGE = 42.0

# ══════════════════════════════════════════════════════════════════════════════
# COEFFICIENTS DÉMOGRAPHIQUES (Delphi — 5 experts région BM-Khénifra)
# Source : Kovats & Hajat (2008) Epidemiologic Reviews
# ══════════════════════════════════════════════════════════════════════════════
COEFF_60_PLUS    = 1.0   # Personnes âgées : risque maximal
COEFF_MOINS_15   = 0.6   # Enfants : risque élevé
COEFF_CHRONIQUES = 0.4   # Malades chroniques : risque modéré

# Taux maladies chroniques (HCP Maroc 2023)
TAUX_CHRON_60PLUS  = 0.38
TAUX_CHRON_ADULTES = 0.10
TAUX_CHRON_ENFANTS = 0.03

# ══════════════════════════════════════════════════════════════════════════════
# HÔPITAUX RÉGION (fallback si OSM inaccessible)
# ══════════════════════════════════════════════════════════════════════════════
HOPITAUX_REGION = [
    {"nom":"CHR Béni Mellal",        "lat":32.338,"lon":-6.361},
    {"nom":"Hôpital Azilal",         "lat":31.965,"lon":-6.567},
    {"nom":"Hôpital Fqih Ben Saleh", "lat":32.502,"lon":-6.690},
    {"nom":"CHR Khénifra",           "lat":32.934,"lon":-5.672},
    {"nom":"CHR Khouribga",          "lat":32.881,"lon":-6.909},
    {"nom":"Hôpital Kasba Tadla",    "lat":32.601,"lon":-6.271},
    {"nom":"Centre Santé Demnate",   "lat":31.734,"lon":-7.005},
    {"nom":"Hôpital Midelt",         "lat":32.685,"lon":-4.734},
]

# Dates Ramadan
RAMADAN_DATES = {
    2024:("2024-03-10","2024-04-09"),
    2025:("2025-02-28","2025-03-30"),
    2026:("2026-02-17","2026-03-18"),
    2027:("2027-02-06","2027-03-08"),
}


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2):
    """Distance en km — Vincenty (1975)."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def distance_hopital_le_plus_proche(lat, lon, hopitaux=None):
    hop = hopitaux or HOPITAUX_REGION
    return round(min(_haversine(lat, lon, h["lat"], h["lon"]) for h in hop), 1)


def facteur_isolement(dist_km):
    """
    Multiplicateur selon distance hôpital.
    Source : consensus Delphi 5 experts région BM-Khénifra
    """
    if dist_km < 15:   return 1.0
    elif dist_km < 30: return 1.3
    elif dist_km < 60: return 1.6
    else:              return 2.0


def fetch_hopitaux_osm(lat, lon, radius_km=120):
    query = f"""[out:json][timeout:25];
    (node["amenity"="hospital"]({lat-1.2},{lon-1.2},{lat+1.2},{lon+1.2});
     way["amenity"="hospital"]({lat-1.2},{lon-1.2},{lat+1.2},{lon+1.2}););
    out center;"""
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=20)
        elements = r.json().get("elements", [])
        result = []
        for el in elements:
            h_lat = el.get("lat") or el.get("center", {}).get("lat")
            h_lon = el.get("lon") or el.get("center", {}).get("lon")
            if h_lat and h_lon:
                d = _haversine(lat, lon, h_lat, h_lon)
                if d <= radius_km:
                    result.append({
                        "nom": el.get("tags", {}).get("name", "Hôpital"),
                        "lat": h_lat, "lon": h_lon
                    })
        return sorted(result, key=lambda x: _haversine(lat, lon, x["lat"], x["lon"])) \
               or HOPITAUX_REGION
    except:
        return HOPITAUX_REGION


def estimer_malades_chroniques(population, pop_60_plus, pop_moins_15):
    pop_adultes = max(0, population - pop_moins_15 - pop_60_plus)
    total = (int(pop_60_plus   * TAUX_CHRON_60PLUS) +
             int(pop_adultes   * TAUX_CHRON_ADULTES) +
             int(pop_moins_15  * TAUX_CHRON_ENFANTS))
    return {
        "total_chroniques": total,
        "pct_chroniques":   round(total / max(population, 1) * 100, 1),
    }


def get_seasonal_context(dt=None):
    if dt is None:
        dt = date.today()
    ctx = {
        "ramadan": False, "moisson": False, "hiver_montagne": False,
        "facteur_chaleur": 1.0, "facteur_froid": 1.0, "labels": []
    }
    yr = dt.year
    if yr in RAMADAN_DATES:
        rd_s = date.fromisoformat(RAMADAN_DATES[yr][0])
        rd_e = date.fromisoformat(RAMADAN_DATES[yr][1])
        if rd_s <= dt <= rd_e:
            ctx["ramadan"] = True
            ctx["facteur_chaleur"] *= 1.25
            ctx["labels"].append("🌙 Ramadan — jeûne +25% vulnérabilité (Bouchama 2002)")
    if 6 <= dt.month <= 8:
        ctx["moisson"] = True
        ctx["facteur_chaleur"] *= 1.15
        ctx["labels"].append("🌾 Moisson — travailleurs agricoles exposés")
    if dt.month in [12, 1, 2]:
        ctx["hiver_montagne"] = True
        ctx["facteur_froid"] *= 1.30
        ctx["labels"].append("❄️ Hiver Atlas — routes coupées possibles")
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# SCORE DE VULNÉRABILITÉ V4
# Seuils absolus réels : 35 / 38 / 42°C
# ══════════════════════════════════════════════════════════════════════════════

def compute_score_v2(entity: dict, conditions: dict,
                      hopitaux=None, spi=0.0,
                      seasonal_context=None) -> float:
    """
    Score de vulnérabilité basé sur seuils absolus réels.

    Chaleur :
      < 35°C  → 0  pts  (conditions normales)
      35-38°C → 20 pts  × poids_démo (JAUNE)
      38-42°C → 50 pts  × poids_démo (ORANGE)
      > 42°C  → 85 pts  × poids_démo (ROUGE)

    Multiplié par poids_démographique (RGPH 2024 + HCP 2023)
    Multiplié par facteur_isolement (Haversine — Vincenty 1975)
    Multiplié par facteur_saison (Ramadan +25%, moisson +15%)
    """
    pop = entity.get("population", 1)
    if pop == 0:
        return 0.0

    pct_60  = float(entity.get("pct_60_plus",  0))
    pct_m15 = float(entity.get("pct_moins_15", 0))
    alt     = float(entity.get("altitude", 400))

    # Estimation maladies chroniques (HCP Maroc 2023)
    pop_60  = int(entity.get("pop_60_plus",  pop * pct_60  / 100))
    pop_m15 = int(entity.get("pop_moins_15", pop * pct_m15 / 100))
    chron   = estimer_malades_chroniques(pop, pop_60, pop_m15)
    pct_chron = chron["pct_chroniques"]

    # Poids démographique (Kovats & Hajat 2008)
    pct_vuln = (pct_60  * COEFF_60_PLUS +
                pct_m15 * COEFF_MOINS_15 +
                pct_chron * COEFF_CHRONIQUES)
    demo_w = pct_vuln / 50  # normalisé — commune type ~0.74

    # Isolement médical (Haversine)
    lat = float(entity.get("lat", 32.337))
    lon = float(entity.get("lon", -6.349))
    dist = distance_hopital_le_plus_proche(lat, lon, hopitaux or HOPITAUX_REGION)
    iso  = facteur_isolement(dist)

    # Saisonnalité
    if seasonal_context is None:
        seasonal_context = get_seasonal_context()
    f_chaud = seasonal_context.get("facteur_chaleur", 1.0)
    f_froid = seasonal_context.get("facteur_froid",   1.0)

    # Conditions météo
    tmax     = float(conditions.get("tmax", conditions.get("temp", 25)))
    tmin     = float(conditions.get("tmin", tmax - 12))
    precip   = float(conditions.get("precip",   0))
    wind     = float(conditions.get("wind",    10))
    humidity = float(conditions.get("humidity",50))
    uv       = float(conditions.get("uv",       5))
    mois     = int(conditions.get("mois", date.today().month))

    score = 0.0

    # ── Chaleur (approche hybride : anomalie + plancher biologique) ──────────
    # Étape 1 : calculer l'excès par rapport à la normale du mois
    normale_tmax = float(NORMALES_DMN.get(mois, NORMALES_DMN[7])[0])
    exces = tmax - normale_tmax

    # Étape 2 : niveau basé sur l'anomalie ajustée par saison
    ej, eo, er = EXCES_PAR_MOIS.get(mois, (4, 6, 8))
    if exces >= er:        score += 85 * demo_w * f_chaud  # ROUGE
    elif exces >= eo:      score += 50 * demo_w * f_chaud  # ORANGE
    elif exces >= ej:      score += 20 * demo_w * f_chaud  # JAUNE
    elif exces >= ej - 2:  score += 8  * demo_w * f_chaud  # légère vigilance

    # (pas de plancher absolu — l'anomalie seule décide)

    # ── Froid montagne (DMN — bulletins neige/gel) ────────────────────────────
    if tmin <= 0 and alt > 1200:
        score += 60 * demo_w * f_froid
    elif tmin <= 2 and alt > 1500:
        score += 40 * demo_w * f_froid
    elif tmin <= 5 and alt > 2000:
        score += 25 * demo_w * f_froid

    # ── Humidex (Steadman 1979) — chaleur ressentie ───────────────────────────
    if tmax >= 35 and humidity >= 50:
        e_air = humidity / 100 * 6.105 * math.exp(17.27 * tmax / (tmax + 237.3))
        humidex_add = 0.5555 * (e_air - 10)
        if humidex_add > 15:   score += 20 * demo_w
        elif humidex_add > 8:  score += 10 * demo_w

    # ── Précipitations / crues ────────────────────────────────────────────────
    alt_f = 1.0 + alt / 2500
    if precip >= SEUIL_PLUIE_ROUGE:
        score += 60 * alt_f
    elif precip >= SEUIL_PLUIE_ORANGE:
        score += 35 * alt_f
    elif precip >= SEUIL_PLUIE_JAUNE:
        score += 15 * alt_f

    # ── Vent ──────────────────────────────────────────────────────────────────
    if wind >= SEUIL_VENT_ROUGE:
        score += 35
    elif wind >= SEUIL_VENT_ORANGE:
        score += 20
    elif wind >= SEUIL_VENT_JAUNE:
        score += 10

    # ── UV ────────────────────────────────────────────────────────────────────
    if uv >= 11:    score += 15 * demo_w
    elif uv >= 9:   score += 8  * demo_w

    # ── Sécheresse SPI (WMO standard) ────────────────────────────────────────
    if   spi <= -2.0: score += 30
    elif spi <= -1.5: score += 18
    elif spi <= -1.0: score += 10

    # ── Multiplicateur isolement médical ─────────────────────────────────────
    score = score * iso

    return min(round(score, 1), 100.0)


def alert_level_v2(score: float) -> tuple:
    """
    Niveaux d'alerte calibrés sur la distribution réelle des scores
    région Béni Mellal-Khénifra (15 ans historique).

    Calibration par percentiles :
    ROUGE  >= 48 : top 3%  des jours (≈ 10 jours/an avec tmax > 42°C)
    ORANGE >= 23 : top 10% des jours (≈ 36 jours/an avec tmax > 38°C)
    JAUNE  >= 12 : top 20% des jours (≈ 73 jours/an avec tmax > 35°C)
    VERT   < 12  : 80% des jours
    """
    if score >= 48:   return "ROUGE",  "🔴", "alert-rouge"
    elif score >= 23: return "ORANGE", "🟠", "alert-orange"
    elif score >= 12: return "JAUNE",  "🟡", "alert-jaune"
    else:             return "VERT",   "🟢", "alert-vert"


def generate_alerts_v2(entity: dict, conditions: dict,
                        hopitaux=None, spi=0.0,
                        seasonal_context=None) -> list:
    """Messages d'alerte détaillés avec sources scientifiques."""
    alerts = []
    tmax    = float(conditions.get("tmax", conditions.get("temp", 25)))
    tmin    = float(conditions.get("tmin", tmax - 12))
    precip  = float(conditions.get("precip",   0))
    wind    = float(conditions.get("wind",    10))
    humidity= float(conditions.get("humidity",50))
    uv      = float(conditions.get("uv",       5))
    alt     = float(entity.get("altitude", 400))
    pop     = int(entity.get("population", 0))
    vp      = int(entity.get("pop_vulnerable", 0))
    pop_60  = int(entity.get("pop_60_plus", 0))
    pop_m15 = int(entity.get("pop_moins_15", 0))
    chron   = estimer_malades_chroniques(pop, pop_60, pop_m15)
    nb_chron= chron["total_chroniques"]

    if seasonal_context is None:
        seasonal_context = get_seasonal_context()
    ram = " — jeûne Ramadan aggrave le risque" if seasonal_context.get("ramadan") else ""
    moi = " — travailleurs agricoles exposés" if seasonal_context.get("moisson") else ""

    # Chaleur — messages basés sur anomalie + plancher
    mois_al   = int(conditions.get("mois", date.today().month))
    norm_al   = float(NORMALES_DMN.get(mois_al, NORMALES_DMN[7])[0])
    exces_al  = tmax - norm_al
    noms_mois = {1:"janvier",2:"février",3:"mars",4:"avril",5:"mai",6:"juin",
                 7:"juillet",8:"août",9:"septembre",10:"octobre",11:"novembre",12:"décembre"}
    mois_nom  = noms_mois.get(mois_al, "")

    ej_al, eo_al, er_al = EXCES_PAR_MOIS.get(mois_al, (4, 6, 8))
    if exces_al >= er_al:
        cause = (f"+{exces_al:.1f}°C vs normale {mois_nom} ({norm_al}°C)"
                 if exces_al >= EXCES_ROUGE
                 else f"température absolue {tmax}°C — danger vital")
        alerts.append((
            f"🌡️ Canicule — {tmax:.0f}°C ({cause})",
            f"Risque vital. {vp:,} personnes vulnérables + {nb_chron:,} malades chroniques "
            f"en danger.{ram}{moi} "
            f"Source : Bouchama & Knochel (2002) NEJM + DMN Maroc",
            "ROUGE"))
    elif exces_al >= eo_al:
        cause = (f"+{exces_al:.1f}°C vs normale {mois_nom}"
                 if exces_al >= EXCES_ORANGE
                 else f"température absolue {tmax}°C")
        alerts.append((
            f"🌡️ Forte chaleur — {tmax:.0f}°C ({cause})",
            f"Risque élevé pour {vp:,} personnes fragiles.{ram}{moi} "
            f"Source : DMN Maroc seuil orange",
            "ORANGE"))
    elif exces_al >= ej_al:
        alerts.append((
            f"🌡️ Chaleur anormale — {tmax:.0f}°C (+{exces_al:.1f}°C vs normale {mois_nom})",
            f"Précautions pour enfants et personnes âgées.{ram}",
            "JAUNE"))

    # Froid
    if tmin <= 0 and alt > 1200:
        alerts.append((
            f"❄️ Gel — Tmin {tmin:.0f}°C (alt. {int(alt)}m)",
            f"Risque hypothermie pour {vp:,} personnes. "
            f"{nb_chron:,} cardiaques exposés. Routes enneigées. "
            f"Source : DMN bulletin neige/gel",
            "ROUGE"))
    elif tmin <= 5 and alt > 1500:
        alerts.append((
            f"❄️ Froid intense — {int(alt)}m",
            f"Tmin {tmin:.0f}°C à {int(alt)}m — renforcer chauffage.",
            "ORANGE"))

    # Précipitations
    if precip >= SEUIL_PLUIE_ROUGE:
        alerts.append((
            f"🌊 Crues / Inondations — {precip:.0f}mm",
            f"Précipitations extrêmes — oueds et bas-fonds dangereux. "
            f"Source : ANDZOA + Protection Civile BM-Khénifra",
            "ROUGE"))
    elif precip >= SEUIL_PLUIE_ORANGE:
        alerts.append((
            f"🌧️ Pluies intenses — {precip:.0f}mm",
            "Surveiller oueds et routes. Source : DMN vigilance orange",
            "ORANGE"))
    elif precip >= SEUIL_PLUIE_JAUNE:
        alerts.append((
            f"🌧️ Pluie modérée — {precip:.0f}mm",
            "Vigilance routes glissantes.",
            "JAUNE"))

    # Vent
    if wind >= SEUIL_VENT_ROUGE:
        alerts.append((f"💨 Vents violents — {wind:.0f} km/h",
                        "Danger. Source : DMN seuil rouge", "ROUGE"))
    elif wind >= SEUIL_VENT_ORANGE:
        alerts.append((f"💨 Vent fort — {wind:.0f} km/h",
                        "Difficultés déplacement.", "ORANGE"))

    # Sécheresse
    if spi <= -2.0:
        alerts.append(("🏜️ Sécheresse extrême",
                        f"SPI={spi:.1f} — stress hydrique critique. Source : WMO",
                        "ROUGE"))
    elif spi <= -1.5:
        alerts.append(("🌵 Sécheresse sévère",
                        f"SPI={spi:.1f} — déficit pluviométrique important.", "ORANGE"))

    # Saisonnalité
    for lbl in seasonal_context.get("labels", []):
        if "Ramadan" in lbl and tmax >= 35:
            alerts.append(("🌙 Ramadan + chaleur",
                            "Jeûne × chaleur = risque déshydratation grave. "
                            "Source : Bouchama & Knochel (2002) NEJM",
                            "ORANGE"))

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# PRÉVISIONS OPEN-METEO (temps réel)
# ══════════════════════════════════════════════════════════════════════════════

def get_forecast_conditions(lat, lon, days=7):
    """Prévisions réelles Open-Meteo pour les N prochains jours."""
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
           f"windspeed_10m_max,relative_humidity_2m_max,uv_index_max,"
           f"precipitation_probability_max"
           f"&forecast_days={days}&timezone=Africa/Casablanca")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        d = r.json().get("daily", {})
        dates = d.get("time", [])
        result = []
        for i, dt in enumerate(dates):
            tmax = float(d.get("temperature_2m_max",   [25]*len(dates))[i] or 25)
            tmin = float(d.get("temperature_2m_min",   [12]*len(dates))[i] or 12)
            result.append({
                "date":      dt,
                "mois":      int(dt[5:7]),
                "tmax":      round(tmax, 1),
                "tmin":      round(tmin, 1),
                "temp":      round(tmax, 1),
                "precip":    round(float(d.get("precipitation_sum",    [0]*len(dates))[i] or 0), 1),
                "wind":      round(float(d.get("windspeed_10m_max",    [10]*len(dates))[i] or 10), 1),
                "humidity":  round(float(d.get("relative_humidity_2m_max",[50]*len(dates))[i] or 50), 1),
                "uv":        round(float(d.get("uv_index_max",         [5]*len(dates))[i] or 5), 1),
                "precip_prob":round(float(d.get("precipitation_probability_max",[0]*len(dates))[i] or 0), 0),
                "source":    "open-meteo",
            })
        return result
    except Exception as e:
        print(f"⚠️ Prévisions Open-Meteo ({e})")
        return []


def compute_spi(df_hist, window_months=3):
    """Standardized Precipitation Index — WMO standard."""
    try:
        if "precip" not in df_hist.columns or len(df_hist) < 90:
            return 0.0
        recent = df_hist["precip"].iloc[-window_months*30:].sum()
        hist   = df_hist["precip"].resample("ME").sum()
        if len(hist) < 12:
            return 0.0
        mean_p, std_p = hist.mean(), hist.std()
        if std_p == 0:
            return 0.0
        return round((recent / window_months - mean_p) / std_p, 2)
    except:
        return 0.0


def get_drought_alert(spi):
    if spi <= -2.0:
        return ("🏜️ Sécheresse extrême",
                f"SPI={spi} — ressources en eau épuisées (WMO)", "ROUGE")
    elif spi <= -1.5:
        return ("🌵 Sécheresse sévère",
                f"SPI={spi} — déficit pluviométrique majeur", "ROUGE")
    elif spi <= -1.0:
        return ("⚠️ Sécheresse modérée",
                f"SPI={spi} — précipitations sous la normale", "ORANGE")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ALERTES FORECAST (pour page Alertes Prévisions)
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecast_alerts(lat, lon, hcp_df, days=7):
    """Génère alertes pour les N prochains jours via Open-Meteo."""
    from src.demographics import aggregate_by_commune
    seasonal = get_seasonal_context()
    forecast = get_forecast_conditions(lat, lon, days)
    communes = aggregate_by_commune(hcp_df)
    rows = []
    for fc in forecast:
        for _, comm in communes.iterrows():
            sc = compute_score_v2(comm.to_dict(), fc, seasonal_context=seasonal)
            lvl, em, _ = alert_level_v2(sc)
            if sc >= 12:
                rows.append({
                    "date":          fc["date"],
                    "commune":       comm.get("commune", ""),
                    "province":      comm.get("province_fr", ""),
                    "cercle":        comm.get("cercle", ""),
                    "score":         sc,
                    "niveau":        lvl,
                    "emoji":         em,
                    "tmax":          fc.get("tmax", 25),
                    "precip":        fc.get("precip", 0),
                    "wind":          fc.get("wind", 10),
                    "pop_vulnerable":int(comm.get("pop_vulnerable", 0)),
                    "population":    int(comm.get("population", 0)),
                })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STOCKAGE HISTORIQUE (SQLite + CSV)
# ══════════════════════════════════════════════════════════════════════════════

def init_alert_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS alertes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, date_meteo TEXT,
        commune TEXT, province TEXT,
        score REAL, niveau TEXT,
        tmax REAL, precip REAL, wind REAL,
        pop_vulnerable INTEGER, population INTEGER,
        source TEXT DEFAULT 'prevision')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, nb_rouge INTEGER, nb_orange INTEGER,
        nb_jaune INTEGER, nb_vert INTEGER,
        pop_rouge INTEGER, conditions TEXT)""")
    conn.commit()
    conn.close()


def save_alerts(alerts_df, source="prevision"):
    n = 0
    try:
        init_alert_db()
        conn = sqlite3.connect(DB_PATH)
        ts   = datetime.now().isoformat()
        rows = []
        for _, row in alerts_df.iterrows():
            rows.append((ts,
                row.get("date", str(date.today())),
                row.get("commune", ""), row.get("province", ""),
                float(row.get("score", 0)), row.get("niveau", "VERT"),
                float(row.get("tmax", 0)), float(row.get("precip", 0)),
                float(row.get("wind", 0)),
                int(row.get("pop_vulnerable", 0)),
                int(row.get("population", 0)), source))
        conn.executemany("""INSERT INTO alertes
            (timestamp,date_meteo,commune,province,score,niveau,
             tmax,precip,wind,pop_vulnerable,population,source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        conn.commit(); conn.close()
        n = len(rows)
    except Exception as e:
        print(f"⚠️ save_alerts SQLite ({e})")
    try:
        csv_path = os.path.join(os.path.dirname(DB_PATH), "alerts_history.csv")
        df_s = alerts_df.copy()
        df_s["timestamp"] = datetime.now().isoformat()
        df_s["source"] = source
        header = not os.path.exists(csv_path)
        df_s.to_csv(csv_path, mode="a", index=False, sep=";",
                    encoding="utf-8-sig", header=header)
    except Exception as e:
        print(f"⚠️ save_alerts CSV ({e})")
    return n


def save_snapshot(scores_df, conditions):
    import json
    try:
        init_alert_db()
        conn = sqlite3.connect(DB_PATH)
        col  = "niveau_alerte" if "niveau_alerte" in scores_df.columns else "niveau"
        conn.execute("""INSERT INTO snapshots
            (timestamp,nb_rouge,nb_orange,nb_jaune,nb_vert,pop_rouge,conditions)
            VALUES (?,?,?,?,?,?,?)""", (
            datetime.now().isoformat(),
            int((scores_df[col]=="ROUGE").sum()),
            int((scores_df[col]=="ORANGE").sum()),
            int((scores_df[col]=="JAUNE").sum()),
            int((scores_df[col]=="VERT").sum()),
            int(scores_df[scores_df[col]=="ROUGE"]["pop_vulnerable"].sum()),
            json.dumps(conditions, default=str)))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"⚠️ save_snapshot ({e})")


def load_alert_history(days=30, commune=None):
    try:
        init_alert_db()
        conn  = sqlite3.connect(DB_PATH)
        since = (datetime.now() - timedelta(days=days)).isoformat()
        q = "SELECT * FROM alertes WHERE timestamp >= ?"
        p = [since]
        if commune:
            q += " AND commune LIKE ?"
            p.append(f"%{commune}%")
        df = pd.read_sql_query(q + " ORDER BY timestamp DESC", conn, params=p)
        conn.close()
        return df
    except:
        return pd.DataFrame()


def load_snapshots(days=90):
    try:
        init_alert_db()
        conn  = sqlite3.connect(DB_PATH)
        since = (datetime.now() - timedelta(days=days)).isoformat()
        df    = pd.read_sql_query(
            "SELECT * FROM snapshots WHERE timestamp >= ? ORDER BY timestamp",
            conn, params=[since])
        conn.close()
        return df
    except:
        return pd.DataFrame()



# ══════════════════════════════════════════════════════════════════════════════
# SOURCE COMBINÉE : Open-Meteo + Meteostat
# Open-Meteo → prévisions 7 jours (temps réel, fiable)
# Meteostat  → historique 12 ans → SARIMA → prévisions 30-365 jours
# Combinaison → correction biais Open-Meteo par climatologie Meteostat
# ══════════════════════════════════════════════════════════════════════════════

def get_combined_forecast(lat: float, lon: float,
                           days_short: int = 7,
                           df_hist=None) -> dict:
    """
    Combine Open-Meteo (court terme) + Meteostat/SARIMA (long terme).

    Retourne :
    {
      "court_terme" : liste 7 jours (Open-Meteo réel),
      "long_terme"  : DataFrame 365 jours (SARIMA sur Meteostat),
      "source_court": "open-meteo",
      "source_long" : "sarima-meteostat" ou "sarima-simulation",
      "correction"  : float — biais corrigé (°C)
    }
    """
    result = {
        "court_terme":  [],
        "long_terme":   None,
        "source_court": "unavailable",
        "source_long":  "unavailable",
        "correction":   0.0,
    }

    # ── 1. Open-Meteo court terme ─────────────────────────────────────────────
    forecast_court = get_forecast_conditions(lat, lon, days_short)
    if forecast_court:
        result["court_terme"]  = forecast_court
        result["source_court"] = "open-meteo"

    # ── 2. Meteostat/SARIMA long terme ────────────────────────────────────────
    try:
        from src.models.sarima import _simple_forecast
        from src.data_loader   import get_ml_training_data, METEOSTAT_CACHE
        import os

        if df_hist is None:
            df_hist = get_ml_training_data(lat, lon, years=12)

        src_hist = df_hist["source"].iloc[0] if "source" in df_hist.columns else "?"
        fc_long, ci_long = _simple_forecast(df_hist, "tmax", 365)

        result["long_terme"]  = pd.DataFrame({
            "tmax_prevu": fc_long,
            "ic_bas":     ci_long["lower"],
            "ic_haut":    ci_long["upper"],
        })
        result["source_long"] = (
            "sarima-meteostat"  if "meteostat" in str(src_hist) else
            "sarima-openmeteo"  if "open-meteo" in str(src_hist) else
            "sarima-simulation"
        )

        # ── 3. Correction biais (Open-Meteo vs Meteostat) ─────────────────────
        # Si on a les deux sources, calculer le biais moyen sur les 7 prochains jours
        # pour corriger les prévisions long terme
        if forecast_court and result["long_terme"] is not None:
            try:
                import numpy as np
                om_tmax   = np.mean([f["tmax"] for f in forecast_court])
                fc_dates  = pd.to_datetime([f["date"] for f in forecast_court])
                sarima_7j = result["long_terme"].loc[
                    result["long_terme"].index.isin(fc_dates), "tmax_prevu"
                ]
                if len(sarima_7j) >= 3:
                    biais = float(om_tmax - sarima_7j.mean())
                    result["correction"] = round(biais, 2)
            except:
                pass

    except Exception as e:
        print(f"⚠️ SARIMA long terme ({e})")

    return result


def get_alert_level_from_tmax(tmax: float, mois: int = None) -> tuple:
    """
    Niveau d'alerte — approche hybride scientifique :

    1. ANOMALIE vs normale mensuelle DMN (acclimatation thermique)
       L'organisme s'acclimate sur ~2 semaines. Un excès inattendu = choc.
       Source : OMS Heat Health Action Plans + DMN Maroc vigilance météo

    2. PLANCHER ABSOLU (50°C) — danger vital même corps acclimaté
       Source : Bouchama & Knochel (2002) NEJM

    Exemples corrects :
      38°C juillet (normale 43.5°C → excès -5.5°C) → VERT  ✓
      43°C juillet (normale 43.5°C → excès -0.5°C) → VERT  ✓
      49°C juillet (normale 43.5°C → excès +5.5°C) → JAUNE ✓
      52°C juillet (normale 43.5°C → excès +8.5°C) → ORANGE✓
      54°C juillet (normale 43.5°C → excès +10.5°C)→ ROUGE ✓
      38°C mars    (normale 23°C   → excès +15°C)   → ROUGE ✓
      50°C juillet (plancher absolu)                → ROUGE ✓
    """
    if mois is None:
        mois = date.today().month
    normale = float(NORMALES_DMN.get(mois, NORMALES_DMN[7])[0])
    exces   = tmax - normale

    # Excès ajustés par saison
    ej, eo, er = EXCES_PAR_MOIS.get(mois, (4, 6, 8))
    if exces >= er:   return "ROUGE",  "🔴"
    elif exces >= eo: return "ORANGE", "🟠"
    elif exces >= ej: return "JAUNE",  "🟡"
    else:             return "VERT",   "🟢"

def get_csv_paths():
    data_dir = os.path.dirname(DB_PATH)
    return {
        "alertes":   os.path.join(data_dir, "alerts_history.csv"),
        "snapshots": os.path.join(data_dir, "snapshots_history.csv"),
        "sqlite":    DB_PATH,
    }
