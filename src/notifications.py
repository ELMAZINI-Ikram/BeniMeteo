"""
notifications.py — Notifications automatiques BeniMeteo
Email (Gmail SMTP) + logs locaux si email non configuré
"""

import smtplib
import os
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date
import pandas as pd

# ── Configuration (à remplir dans config.json ou variables d'environnement) ──
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "notif_config.json")

DEFAULT_CONFIG = {
    "email_enabled":   False,
    "smtp_server":     "smtp.gmail.com",
    "smtp_port":       587,
    "email_sender":    "",
    "email_password":  "",        # Mot de passe d'application Gmail (16 caractères)
    "email_recipients": [],       # Liste des destinataires
    "seuil_notification": "ORANGE",  # ROUGE, ORANGE, JAUNE
    "notification_heure": "07:00",
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            return {**DEFAULT_CONFIG, **cfg}
        except:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _build_html_email(alerts_df: pd.DataFrame, snapshot: dict) -> str:
    """Construit le corps HTML de l'email d'alerte."""
    nb_rouge  = int((alerts_df["niveau"] == "ROUGE").sum())  if "niveau" in alerts_df.columns else 0
    nb_orange = int((alerts_df["niveau"] == "ORANGE").sum()) if "niveau" in alerts_df.columns else 0
    nb_jaune  = int((alerts_df["niveau"] == "JAUNE").sum())  if "niveau" in alerts_df.columns else 0

    color_map = {"ROUGE": "#D62828", "ORANGE": "#F77F00", "JAUNE": "#E8A020"}

    rows_html = ""
    if len(alerts_df) > 0:
        top = alerts_df.sort_values("score", ascending=False).head(10) if "score" in alerts_df.columns else alerts_df.head(10)
        for _, row in top.iterrows():
            niv    = row.get("niveau", "VERT")
            color  = color_map.get(niv, "#4A7C59")
            commune = row.get("commune", row.get("nom", "—"))
            province = row.get("province_fr", row.get("province", "—"))
            score  = row.get("score", 0)
            vp     = row.get("pop_vulnerable", 0)
            rows_html += f"""
            <tr>
              <td style="padding:8px;border-bottom:1px solid #eee;">
                <span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{niv}</span>
              </td>
              <td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold">{commune}</td>
              <td style="padding:8px;border-bottom:1px solid #eee;color:#666">{province}</td>
              <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{score:.0f}/100</td>
              <td style="padding:8px;border-bottom:1px solid #eee;text-align:center">{int(vp):,}</td>
            </tr>"""

    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;color:#333">
  <div style="background:#1A3D6B;color:white;padding:20px;border-radius:8px 8px 0 0">
    <h1 style="margin:0;font-size:22px">🌦️ BeniMeteo — Bulletin d'Alertes</h1>
    <p style="margin:8px 0 0;opacity:0.8">Région Béni Mellal-Khénifra · {date_str}</p>
  </div>

  <div style="background:#f8f9fa;padding:16px;display:flex;gap:12px;flex-wrap:wrap">
    <div style="background:#D62828;color:white;padding:12px 20px;border-radius:8px;text-align:center">
      <div style="font-size:28px;font-weight:bold">{nb_rouge}</div>
      <div style="font-size:12px">communes ROUGE</div>
    </div>
    <div style="background:#F77F00;color:white;padding:12px 20px;border-radius:8px;text-align:center">
      <div style="font-size:28px;font-weight:bold">{nb_orange}</div>
      <div style="font-size:12px">communes ORANGE</div>
    </div>
    <div style="background:#E8A020;color:white;padding:12px 20px;border-radius:8px;text-align:center">
      <div style="font-size:28px;font-weight:bold">{nb_jaune}</div>
      <div style="font-size:12px">communes JAUNE</div>
    </div>
    <div style="background:#4A7C59;color:white;padding:12px 20px;border-radius:8px;text-align:center">
      <div style="font-size:28px;font-weight:bold">{snapshot.get("pop_vulnerable_rouge", 0):,}</div>
      <div style="font-size:12px">vulnérables exposés</div>
    </div>
  </div>

  <div style="padding:16px">
    <h2 style="color:#1A3D6B;border-bottom:2px solid #C4622D;padding-bottom:8px">
      Top communes à risque
    </h2>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
      <thead>
        <tr style="background:#f0f0f0">
          <th style="padding:8px;text-align:left">Niveau</th>
          <th style="padding:8px;text-align:left">Commune</th>
          <th style="padding:8px;text-align:left">Province</th>
          <th style="padding:8px;text-align:center">Score</th>
          <th style="padding:8px;text-align:center">Vulnérables</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div style="background:{'#D1FAE5' if nb_rouge==0 and nb_orange==0 and nb_jaune==0 else '#FEF3C7'};padding:12px 16px;margin:0 16px 16px;border-radius:8px;font-size:13px;color:{'#065F46' if nb_rouge==0 and nb_orange==0 and nb_jaune==0 else '#92400E'}">
    <strong>{'✅' if nb_rouge==0 and nb_orange==0 and nb_jaune==0 else '⚠️'} Ce bulletin est généré automatiquement par BeniMeteo.</strong><br>
    {'Aucune anomalie climatique détectée — conditions dans les normes saisonnières DMN Maroc.' if nb_rouge==0 and nb_orange==0 and nb_jaune==0 else 'Vérifiez les prévisions météo réelles avant toute décision opérationnelle.'}
  </div>

  <div style="background:#1A3D6B;color:rgba(255,255,255,0.6);padding:12px;font-size:11px;text-align:center;border-radius:0 0 8px 8px">
    BeniMeteo · Université Sultan Moulay Slimane · Béni Mellal-Khénifra, Maroc
  </div>
</body>
</html>"""


def send_alert_email(alerts_df: pd.DataFrame, snapshot: dict = None,
                     config: dict = None) -> dict:
    """
    Envoie un email d'alerte.
    Retourne {"success": bool, "message": str}
    """
    if config is None:
        config = load_config()

    if not config.get("email_enabled"):
        return {"success": False, "message": "Email non configuré"}
    if not config.get("email_sender") or not config.get("email_password"):
        return {"success": False, "message": "Identifiants manquants"}
    if not config.get("email_recipients"):
        return {"success": False, "message": "Aucun destinataire"}

    if snapshot is None:
        snapshot = {}

    nb_rouge = int((alerts_df["niveau"] == "ROUGE").sum()) if "niveau" in alerts_df.columns else 0
    subject  = f"🚨 BeniMeteo — {nb_rouge} commune(s) ROUGE · {date.today().strftime('%d/%m/%Y')}"

    html_body  = _build_html_email(alerts_df, snapshot)
    text_body  = f"BeniMeteo Alertes — {nb_rouge} zones ROUGE. Consultez l'application pour les détails."

    # Valider les adresses destinataires
    import re as _re
    email_regex = r'^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$'
    invalid_recipients = [r for r in config["email_recipients"]
                          if '@' not in r.strip() or '.' not in r.strip().split('@')[-1]]
    if invalid_recipients:
        return {"success": False,
                "message": f"Adresse(s) invalide(s) : {', '.join(invalid_recipients)}\n"
                           f"Vérifiez l'orthographe des adresses destinataires."}

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"BeniMeteo <{config['email_sender']}>"
        msg["To"]      = ", ".join(config["email_recipients"])

        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html",  "utf-8"))

        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["email_sender"], config["email_password"])
            server.sendmail(
                config["email_sender"],
                config["email_recipients"],
                msg.as_string()
            )

        _log_notification("email_sent", f"Email envoyé à {len(config['email_recipients'])} destinataire(s)")
        return {"success": True, "message": f"Email envoyé à {', '.join(config['email_recipients'])}"}

    except smtplib.SMTPAuthenticationError as e:
        msg_err = (
            "Mot de passe incorrect. \n"
            "Vous devez utiliser un MOT DE PASSE D'APPLICATION Gmail, pas votre vrai mot de passe.\n"
            "Pour le créer : myaccount.google.com → Sécurité → "
            "Vérification en 2 étapes (activez) → Mots de passe des applications → "
            "Autre → 'BeniMeteo' → Générer.\n"
            "Copiez le code 16 lettres généré (ex: abcd efgh ijkl mnop)."
        )
        _log_notification("email_auth_error", str(e))
        return {"success": False, "message": msg_err}
    except smtplib.SMTPRecipientsRefused as e:
        bad = list(e.recipients.keys())
        _log_notification("email_recipient_error", str(e))
        return {"success": False,
                "message": f"Adresse destinataire introuvable : {', '.join(bad)}\n"
                           f"Vérifiez que l'adresse existe et est correctement orthographiée."}
    except smtplib.SMTPException as e:
        _log_notification("email_smtp_error", str(e))
        return {"success": False, "message": f"Erreur SMTP : {e}"}
    except Exception as e:
        _log_notification("email_error", str(e))
        return {"success": False, "message": f"Erreur : {e}"}


def test_email_connection(config: dict) -> dict:
    """Teste la connexion SMTP sans envoyer d'email."""
    try:
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["email_sender"], config["email_password"])
        return {"success": True, "message": "Connexion SMTP réussie"}
    except smtplib.SMTPAuthenticationError:
        return {"success": False, "message": (
            "Authentification échouée. Utilisez un mot de passe d'APPLICATION Gmail "
            "(pas votre vrai mot de passe). "
            "myaccount.google.com → Sécurité → Mots de passe des applications."
        )}
    except Exception as e:
        return {"success": False, "message": f"Erreur SMTP : {e}"}


def _log_notification(event: str, message: str):
    """Journal local des notifications."""
    log_path = os.path.join(os.path.dirname(CONFIG_PATH), "notifications_log.csv")
    line = f"{datetime.now().isoformat()};{event};{message}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def check_and_notify(alerts_df: pd.DataFrame, snapshot: dict, config: dict = None):
    """
    Vérifie si des alertes dépassent le seuil et envoie l'email si nécessaire.
    Appelé automatiquement par le scheduler.
    """
    if config is None:
        config = load_config()

    seuil = config.get("seuil_notification", "ORANGE")
    seuils = {"ROUGE": ["ROUGE"], "ORANGE": ["ROUGE", "ORANGE"], "JAUNE": ["ROUGE", "ORANGE", "JAUNE"]}
    niveaux_cibles = seuils.get(seuil, ["ROUGE", "ORANGE"])

    if "niveau" in alerts_df.columns:
        alerte_trouvee = alerts_df["niveau"].isin(niveaux_cibles).any()
    else:
        alerte_trouvee = False

    if alerte_trouvee:
        result = send_alert_email(alerts_df, snapshot, config)
        return result
    return {"success": False, "message": "Aucune alerte au-dessus du seuil"}
