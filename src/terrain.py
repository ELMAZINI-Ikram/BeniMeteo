"""
terrain.py — Validation terrain et seuils personnalisés par province
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, date

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH    = os.path.join(DATA_DIR, "alerts_history.db")
SEUILS_CFG = os.path.join(DATA_DIR, "seuils_personnalises.json")

DEFAULT_SEUILS = {
    "_region": {"rouge": 60, "orange": 35, "jaune": 15},
    "Azilal":          {"rouge": 55, "orange": 30, "jaune": 12},
    "Béni Mellal":     {"rouge": 60, "orange": 35, "jaune": 15},
    "Fqih Ben Saleh":  {"rouge": 58, "orange": 33, "jaune": 14},
    "Khénifra":        {"rouge": 55, "orange": 30, "jaune": 12},
    "Khouribga":       {"rouge": 60, "orange": 35, "jaune": 15},
}


def load_seuils() -> dict:
    if os.path.exists(SEUILS_CFG):
        try:
            with open(SEUILS_CFG) as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_SEUILS.copy()


def save_seuils(seuils: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SEUILS_CFG, "w") as f:
        json.dump(seuils, f, indent=2, ensure_ascii=False)


def alert_level_custom(score: float, province: str = None) -> tuple:
    """Niveau d'alerte avec seuils personnalisés par province."""
    seuils = load_seuils()
    cfg = seuils.get(province, seuils.get("_region", DEFAULT_SEUILS["_region"]))
    r, o, j = cfg["rouge"], cfg["orange"], cfg["jaune"]

    if score >= r:   return "ROUGE",  "🔴", "alert-rouge"
    elif score >= o: return "ORANGE", "🟠", "alert-orange"
    elif score >= j: return "JAUNE",  "🟡", "alert-jaune"
    else:            return "VERT",   "🟢", "alert-vert"


def init_validation_db():
    """Initialise la table de validations terrain."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS validations_terrain (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            commune     TEXT,
            province    TEXT,
            date_alerte TEXT,
            score_prevu REAL,
            niveau_prevu TEXT,
            confirme    INTEGER DEFAULT 0,   -- 1=oui, 0=non, -1=inconnu
            gravite_reelle TEXT,             -- VERT/JAUNE/ORANGE/ROUGE observée
            agent       TEXT,
            notes       TEXT,
            source      TEXT DEFAULT 'app'
        )
    """)
    conn.commit()
    conn.close()


def save_validation(commune: str, province: str, date_alerte: str,
                     score_prevu: float, niveau_prevu: str,
                     confirme: int, gravite_reelle: str,
                     agent: str = "", notes: str = "") -> bool:
    """Sauvegarde une validation terrain."""
    try:
        init_validation_db()
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO validations_terrain
            (timestamp, commune, province, date_alerte, score_prevu,
             niveau_prevu, confirme, gravite_reelle, agent, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (datetime.now().isoformat(), commune, province, date_alerte,
              float(score_prevu), niveau_prevu, confirme,
              gravite_reelle, agent, notes))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ Validation terrain ({e})")
        return False


def load_validations(days: int = 90) -> pd.DataFrame:
    """Charge les validations terrain."""
    try:
        init_validation_db()
        conn = sqlite3.connect(DB_PATH)
        since = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM validations_terrain WHERE timestamp >= ? ORDER BY timestamp DESC",
            conn, params=[since])
        conn.close()
        return df
    except:
        return pd.DataFrame()


def compute_model_accuracy(validations_df: pd.DataFrame) -> dict:
    """Calcule la précision du modèle à partir des validations terrain."""
    if len(validations_df) == 0:
        return {}

    df = validations_df[validations_df["confirme"] != -1].copy()
    if len(df) == 0:
        return {}

    total        = len(df)
    confirmes    = (df["confirme"] == 1).sum()
    non_confirmes= (df["confirme"] == 0).sum()

    # Vrais positifs (alerte ROUGE/ORANGE confirmée sur le terrain)
    alertes_hautes = df[df["niveau_prevu"].isin(["ROUGE", "ORANGE"])]
    vp = (alertes_hautes["confirme"] == 1).sum()
    fp = (alertes_hautes["confirme"] == 0).sum()

    # Faux négatifs (alerte VERT/JAUNE mais terrain grave)
    alertes_basses = df[df["niveau_prevu"].isin(["VERT", "JAUNE"])]
    fn = alertes_basses[alertes_basses["gravite_reelle"].isin(["ROUGE","ORANGE"])].shape[0]

    precision = round(vp / max(vp + fp, 1) * 100, 1)
    recall    = round(vp / max(vp + fn, 1) * 100, 1)

    return {
        "total_validations": total,
        "confirmees":        int(confirmes),
        "non_confirmees":    int(non_confirmes),
        "precision":         precision,
        "recall":            recall,
        "f1":                round(2 * precision * recall / max(precision + recall, 1), 1),
    }
