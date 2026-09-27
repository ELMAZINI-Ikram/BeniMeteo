"""
scheduler.py — Planification automatique des alertes BeniMeteo
Utilise APScheduler pour exécuter les tâches en arrière-plan
"""

import threading
import os
import json
from datetime import datetime, date
import pandas as pd

SCHEDULER_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scheduler_log.csv")


def _log(event: str, message: str):
    line = f"{datetime.now().isoformat()};{event};{message}\n"
    os.makedirs(os.path.dirname(SCHEDULER_LOG), exist_ok=True)
    with open(SCHEDULER_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def daily_alert_job(lat: float, lon: float, hcp_df: pd.DataFrame,
                     hist_df: pd.DataFrame):
    """
    Tâche quotidienne : calcule les alertes et envoie l'email si nécessaire.
    Exécutée chaque matin à l'heure configurée.
    """
    try:
        from src.alert_engine import (get_forecast_conditions, compute_score_v2,
                                       alert_level_v2, compute_spi, get_seasonal_context,
                                       save_snapshot, fetch_hopitaux_osm, HOPITAUX_REGION)
        from src.demographics import aggregate_by_commune
        from src.notifications import check_and_notify, load_config

        _log("scheduler_start", f"Calcul alertes journalières lat={lat} lon={lon}")

        # Récupération conditions météo
        forecast = get_forecast_conditions(lat, lon, days=3)
        if not forecast:
            _log("scheduler_error", "Prévisions indisponibles")
            return

        cond_today = forecast[0]

        # Calcul scores
        hopitaux = fetch_hopitaux_osm(lat, lon) or HOPITAUX_REGION
        spi      = compute_spi(hist_df) if len(hist_df) > 90 else 0.0
        seasonal = get_seasonal_context()
        communes = aggregate_by_commune(hcp_df)
        communes["score"]  = communes.apply(
            lambda r: compute_score_v2(r.to_dict(), cond_today, hopitaux, spi, seasonal), axis=1)
        communes["niveau"] = communes["score"].apply(lambda s: alert_level_v2(s)[0])

        snapshot = {
            "date":    str(date.today()),
            "nb_rouge":  int((communes["niveau"]=="ROUGE").sum()),
            "nb_orange": int((communes["niveau"]=="ORANGE").sum()),
            "pop_vulnerable_rouge": int(communes[communes["niveau"]=="ROUGE"]["pop_vulnerable"].sum()),
        }

        # Sauvegarde snapshot
        save_snapshot(communes.rename(columns={"niveau":"niveau_alerte"}), cond_today)

        # Notification email
        config = load_config()
        result = check_and_notify(communes, snapshot, config)
        _log("scheduler_done", f"Snapshot sauvé | Email: {result['message']}")

    except Exception as e:
        _log("scheduler_error", str(e))


def start_scheduler(lat: float, lon: float, hcp_df: pd.DataFrame,
                     hist_df: pd.DataFrame, heure: str = "07:00"):
    """
    Démarre le scheduler APScheduler en arrière-plan.
    Lance la tâche tous les jours à l'heure configurée.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        hh, mm = heure.split(":")
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            daily_alert_job,
            trigger=CronTrigger(hour=int(hh), minute=int(mm)),
            args=[lat, lon, hcp_df, hist_df],
            id="daily_alerts",
            name="Alertes quotidiennes BeniMeteo",
            replace_existing=True,
        )
        scheduler.start()
        _log("scheduler_started", f"Tâche planifiée à {heure} chaque jour")
        return scheduler

    except ImportError:
        _log("scheduler_import_error", "APScheduler non installé — pip install apscheduler")
        return None
    except Exception as e:
        _log("scheduler_error", str(e))
        return None


def get_scheduler_logs(n: int = 50) -> pd.DataFrame:
    """Retourne les N dernières lignes du journal scheduler."""
    try:
        if not os.path.exists(SCHEDULER_LOG):
            return pd.DataFrame(columns=["timestamp", "event", "message"])
        df = pd.read_csv(SCHEDULER_LOG, sep=";", header=None,
                         names=["timestamp", "event", "message"])
        return df.tail(n).sort_values("timestamp", ascending=False).reset_index(drop=True)
    except:
        return pd.DataFrame()
