"""
water_needs.py — Calcul des besoins en eau par douar — BeniMeteo
═══════════════════════════════════════════════════════════════════
Sources scientifiques :
  OMS (2011). Domestic Water Quantity, Service Level and Health.
              WHO/SDE/WSH/03.02. Geneva.
  Sphere Handbook (2018). Humanitarian Charter and Minimum Standards.
              Chapitre Eau, Assainissement et Hygiène.
  OMS (2008). Heat Health Action Plans — supplément chaleur.
  Norme marocaine NM 03.7.001 — eau potable (IMANOR).
  ONEE Maroc (2022) — niveaux de service eau potable milieu rural.
"""

import pandas as pd
import numpy as np
from datetime import date

# ══════════════════════════════════════════════════════════════════════════════
# BESOINS JOURNALIERS PAR PERSONNE (litres/personne/jour)
# ══════════════════════════════════════════════════════════════════════════════

# Eau potable (boisson)
# Source : OMS (2011) — 2L minimum absolu, 2.5L standard, 3L temps chaud
EAU_BOISSON_BASE         = 2.5   # litres/pers/jour (OMS standard)
EAU_BOISSON_ENFANT_SUPPL = 0.5   # supplément enfants < 15 ans (surface corporelle)
EAU_BOISSON_VIEUX_SUPPL  = 0.3   # supplément 60+ (moins de sensation de soif = risque)
EAU_BOISSON_CANICULE     = 2.0   # supplément si Tmax > 35°C (OMS 2008)
EAU_BOISSON_RAMADAN      = 1.0   # supplément réhydratation nocturne (Ramadan)

# Eau cuisine
# Source : Sphere Handbook (2018) — 3-6L/pers/jour milieu rural Maroc
EAU_CUISINE              = 3.0   # litres/pers/jour

# Eau hygiène corporelle
# Source : Sphere (2018) — 6L minimum hygiène de base, 15L acceptable
EAU_HYGIENE_MIN          = 5.0   # minimum
EAU_HYGIENE_STANDARD     = 10.0  # standard acceptable rural

# Eau animaux (estimation milieu rural Maroc)
# Source : FAO (2006) + données ORMVAT région BM-Khénifra
# Vache : 40L/j | Brebis : 5L/j | Chèvre : 4L/j | Âne : 20L/j | Poule : 0.3L/j
EAU_ANIMAUX_PAR_MENAGE   = 20.0  # estimation moyenne ménage rural Atlas

# Eau agriculture (potager/jardin familial)
# Source : FAO — besoins irrigation petite agriculture familiale
EAU_AGRI_PAR_MENAGE      = 10.0  # litres/ménage/jour (jardins familiaux)

# ══════════════════════════════════════════════════════════════════════════════
# NORMES RÉSERVOIRS
# Source : Sphere Handbook (2018) + ONEE Maroc
# ══════════════════════════════════════════════════════════════════════════════

JOURS_AUTONOMIE_MIN    = 3    # Sphere minimum absolu
JOURS_AUTONOMIE_RURAL  = 7    # Standard ONEE zones rurales isolées
JOURS_AUTONOMIE_ALERTE = 14   # Durant alerte canicule/sécheresse

# Seuils types de réservoirs (m³)
SEUILS_RESERVOIR = [
    (5,    "Citerne plastique",      "< 5m³ — petit douar",         "Économique"),
    (20,   "Réservoir métallique",   "5-20m³ — douar moyen",         "Standard"),
    (50,   "Château d'eau béton",    "20-50m³ — grand douar",        "Durable"),
    (999,  "Infrastructure ONEE",    "> 50m³ — grand centre",        "Institutionnel"),
]

# ══════════════════════════════════════════════════════════════════════════════
# CALCUL PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def calculer_besoins_douar(row: dict, conditions: dict = None) -> dict:
    """
    Calcule les besoins en eau complets pour un douar.

    Paramètres :
      row        : données démographiques du douar (RGPH 2024)
      conditions : dict météo optionnel (tmax, ramadan, etc.)

    Retourne un dict avec tous les besoins calculés et les recommandations
    de réservoir.
    """
    if conditions is None:
        conditions = {}

    pop     = int(row.get('population', 0))
    menages = int(row.get('nb_menages', max(1, pop // 5)))
    if pop == 0:
        return {}

    pct_m15 = float(row.get('pct_moins_15', 30)) / 100
    pct_60  = float(row.get('pct_60_plus',  12)) / 100
    alt     = float(row.get('altitude', 400))

    pop_enfants = int(pop * pct_m15)
    pop_vieux   = int(pop * pct_60)

    tmax      = float(conditions.get('tmax', 25))
    is_canicule = tmax > 35
    is_ramadan  = _is_ramadan()

    # ── 1. Eau potable (boisson) ──────────────────────────────────────────────
    eau_boisson = (
        pop          * EAU_BOISSON_BASE +
        pop_enfants  * EAU_BOISSON_ENFANT_SUPPL +
        pop_vieux    * EAU_BOISSON_VIEUX_SUPPL
    )
    if is_canicule:
        eau_boisson += pop * EAU_BOISSON_CANICULE
    if is_ramadan:
        eau_boisson += pop * EAU_BOISSON_RAMADAN

    # ── 2. Eau cuisine ───────────────────────────────────────────────────────
    eau_cuisine = pop * EAU_CUISINE

    # ── 3. Eau hygiène ───────────────────────────────────────────────────────
    eau_hygiene = pop * EAU_HYGIENE_STANDARD

    # ── 4. Eau animaux ───────────────────────────────────────────────────────
    eau_animaux = menages * EAU_ANIMAUX_PAR_MENAGE

    # ── 5. Eau agriculture ───────────────────────────────────────────────────
    eau_agri = menages * EAU_AGRI_PAR_MENAGE

    # ── Totaux ───────────────────────────────────────────────────────────────
    eau_total_jour = (eau_boisson + eau_cuisine +
                      eau_hygiene + eau_animaux + eau_agri)

    # ── Volumes réservoirs (m³) ──────────────────────────────────────────────
    res_min_m3    = round(eau_total_jour * JOURS_AUTONOMIE_MIN    / 1000, 1)
    res_rural_m3  = round(eau_total_jour * JOURS_AUTONOMIE_RURAL  / 1000, 1)
    res_alerte_m3 = round(eau_total_jour * JOURS_AUTONOMIE_ALERTE / 1000, 1)

    # ── Type de réservoir ─────────────────────────────────────────────────────
    type_res, desc_res, type_tech = _type_reservoir(res_rural_m3)

    # ── Score de priorité d'intervention ─────────────────────────────────────
    pct_vuln = float(row.get('pct_vulnerable', 40))
    score_urgence = _score_priorite(pop, alt, pct_vuln, is_canicule)

    return {
        # Identification
        'province':    row.get('province_fr', ''),
        'cercle':      row.get('cercle', ''),
        'commune':     row.get('commune', ''),
        'douar':       row.get('douar', ''),
        'lat':         float(row.get('lat', 32.337)),
        'lon':         float(row.get('lon', -6.349)),

        # Démographie
        'population':      pop,
        'nb_menages':      menages,
        'altitude_m':      int(alt),
        'pct_vulnerable':  round(pct_vuln, 1),
        'pop_enfants':     pop_enfants,
        'pop_vieux':       pop_vieux,

        # Besoins journaliers (litres)
        'eau_boisson_L_j':  round(eau_boisson),
        'eau_cuisine_L_j':  round(eau_cuisine),
        'eau_hygiene_L_j':  round(eau_hygiene),
        'eau_animaux_L_j':  round(eau_animaux),
        'eau_agri_L_j':     round(eau_agri),
        'eau_total_L_j':    round(eau_total_jour),

        # Besoins par personne/jour
        'litre_pers_boisson': round(eau_boisson / pop, 1),
        'litre_pers_total':   round(eau_total_jour / pop, 1),

        # Volumes réservoirs (m³)
        'res_min_3j_m3':    res_min_m3,
        'res_ideal_7j_m3':  res_rural_m3,
        'res_alerte_14j_m3': res_alerte_m3,

        # Recommandation
        'type_reservoir': type_res,
        'description_reservoir': desc_res,
        'type_technique': type_tech,
        'score_urgence':  score_urgence,

        # Contexte
        'canicule': is_canicule,
        'ramadan':  is_ramadan,
    }


def calculer_besoins_region(hcp_df: pd.DataFrame,
                              conditions: dict = None) -> pd.DataFrame:
    """
    Calcule les besoins en eau pour tous les douars de la région.
    Retourne un DataFrame trié par score d'urgence décroissant.
    """
    rows = []
    for _, row in hcp_df.iterrows():
        result = calculer_besoins_douar(row.to_dict(), conditions)
        if result:
            rows.append(result)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('score_urgence', ascending=False).reset_index(drop=True)
    return df


def get_resume_region(df_eau: pd.DataFrame) -> dict:
    """Résumé statistique des besoins en eau de la région."""
    if len(df_eau) == 0:
        return {}

    total_pop  = df_eau['population'].sum()
    total_jour = df_eau['eau_total_L_j'].sum()
    total_boisson = df_eau['eau_boisson_L_j'].sum()

    types = df_eau['type_reservoir'].value_counts().to_dict()

    return {
        'nb_douars':           len(df_eau),
        'population_totale':   int(total_pop),
        'eau_boisson_m3_j':    round(total_boisson / 1000, 1),
        'eau_totale_m3_j':     round(total_jour / 1000, 1),
        'eau_totale_annuelle_m3': round(total_jour * 365 / 1000, 0),
        'vol_reservoirs_min_m3':   round(df_eau['res_min_3j_m3'].sum(), 0),
        'vol_reservoirs_ideal_m3': round(df_eau['res_ideal_7j_m3'].sum(), 0),
        'types_reservoir': types,
        'douars_prioritaires': df_eau.nlargest(10, 'score_urgence')[
            ['douar','province','population','altitude_m',
             'eau_total_L_j','res_ideal_7j_m3','score_urgence']
        ].to_dict('records'),
    }


def generer_plan_intervention(df_eau: pd.DataFrame,
                               budget_total_dh: float = None) -> pd.DataFrame:
    """
    Génère un plan d'intervention priorisé.
    Optionnel : avec un budget total en DH, estime quels douars couvrir.

    Coût estimatif (ONEE + entreprises BTP Maroc 2024) :
      Citerne plastique 5m³   : ~8 000 DH
      Réservoir métallique 20m³: ~35 000 DH
      Château eau béton 50m³  : ~120 000 DH
      Infrastructure ONEE     : ~500 000 DH+
    """
    COUT_DH = {
        "Citerne plastique":   8_000,
        "Réservoir métallique": 35_000,
        "Château d'eau béton": 120_000,
        "Infrastructure ONEE": 500_000,
    }

    df = df_eau.copy()
    df['cout_estime_dh'] = df['type_reservoir'].apply(
        lambda t: next((v for k,v in COUT_DH.items() if k in t), 35_000)
    )
    df['cout_cumule_dh'] = df['cout_estime_dh'].cumsum()

    if budget_total_dh:
        df['dans_budget'] = df['cout_cumule_dh'] <= budget_total_dh
    else:
        df['dans_budget'] = True

    df['rang_priorite'] = range(1, len(df) + 1)

    return df[['rang_priorite', 'douar', 'province', 'cercle', 'commune',
               'population', 'altitude_m', 'pct_vulnerable',
               'eau_total_L_j', 'res_ideal_7j_m3', 'type_reservoir',
               'cout_estime_dh', 'score_urgence', 'dans_budget']]


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _is_ramadan() -> bool:
    from src.alert_engine import RAMADAN_DATES
    today = date.today()
    yr    = today.year
    if yr in RAMADAN_DATES:
        s = date.fromisoformat(RAMADAN_DATES[yr][0])
        e = date.fromisoformat(RAMADAN_DATES[yr][1])
        return s <= today <= e
    return False


def _type_reservoir(vol_m3: float) -> tuple:
    for seuil, nom, desc, tech in SEUILS_RESERVOIR:
        if vol_m3 < seuil:
            return nom, desc, tech
    return SEUILS_RESERVOIR[-1][1], SEUILS_RESERVOIR[-1][2], SEUILS_RESERVOIR[-1][3]


def _score_priorite(pop: int, alt: float, pct_vuln: float,
                     is_canicule: bool) -> float:
    """
    Score de priorité d'intervention (0-100).
    Composantes :
      - Vulnérabilité démographique (40%)
      - Isolement géographique / altitude (30%)
      - Taille population à risque (20%)
      - Contexte canicule (10%)
    """
    score_vuln = (pct_vuln / 100) * 40
    score_alt  = min(alt / 3000, 1.0) * 30
    score_pop  = (1 - min(pop, 3000) / 3000) * 20  # petits douars isolés = plus urgents
    score_can  = 10 if is_canicule else 0
    return round(score_vuln + score_alt + score_pop + score_can, 1)
