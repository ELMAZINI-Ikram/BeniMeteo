"""
report_generator.py — Génération de rapports PDF BeniMeteo
Utilise ReportLab pour créer des rapports professionnels
"""

import os
import io
from datetime import date, datetime
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rapports")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_alert_report(communes_df: pd.DataFrame,
                           province_data: dict = None,
                           snapshot: dict = None,
                           title: str = "Bulletin d'Alertes") -> bytes:
    """
    Génère un rapport PDF des alertes.
    Retourne les bytes du PDF (compatible st.download_button).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        BLEU    = colors.HexColor("#1A3D6B")
        ORANGE  = colors.HexColor("#C4622D")
        ROUGE   = colors.HexColor("#D62828")
        VERT    = colors.HexColor("#4A7C59")
        JAUNE   = colors.HexColor("#E8A020")
        GRIS_C  = colors.HexColor("#F5F5F5")

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                      textColor=BLEU, fontSize=18, spaceAfter=6)
        h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                      textColor=ORANGE, fontSize=13, spaceBefore=12, spaceAfter=4)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                      fontSize=10, spaceAfter=4)
        center_style= ParagraphStyle("center", parent=styles["Normal"],
                                      alignment=TA_CENTER, fontSize=9,
                                      textColor=colors.HexColor("#666666"))

        story = []

        # En-tête
        story.append(Paragraph("🌦️ BeniMeteo", title_style))
        story.append(Paragraph(f"{title} — Région Béni Mellal-Khénifra", h2_style))
        story.append(Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} · "
            f"Source : Open-Meteo + HCP RGPH 2024", center_style))
        story.append(HRFlowable(width="100%", thickness=2, color=ORANGE,
                                 spaceAfter=12))

        # KPIs
        if len(communes_df) > 0 and "niveau_alerte" in communes_df.columns:
            nb_r = int((communes_df["niveau_alerte"]=="ROUGE").sum())
            nb_o = int((communes_df["niveau_alerte"]=="ORANGE").sum())
            nb_j = int((communes_df["niveau_alerte"]=="JAUNE").sum())
            nb_v = int((communes_df["niveau_alerte"]=="VERT").sum())
            vp_r = int(communes_df[communes_df["niveau_alerte"]=="ROUGE"]["pop_vulnerable"].sum()) \
                   if "pop_vulnerable" in communes_df.columns else 0

            kpi_data = [
                ["Indicateur", "Valeur", "Interprétation"],
                ["Communes ROUGE", str(nb_r), "Risque critique — action immédiate"],
                ["Communes ORANGE", str(nb_o), "Vigilance renforcée"],
                ["Communes JAUNE", str(nb_j), "Précaution"],
                ["Communes VERT", str(nb_v), "Situation normale"],
                ["Personnes vulnérables en zone ROUGE", f"{vp_r:,}", "Enfants + âgés exposés"],
            ]
            kpi_colors = [
                [BLEU,      BLEU,      BLEU],
                [ROUGE,     ROUGE,     ROUGE],
                [ORANGE,    ORANGE,    ORANGE],
                [JAUNE,     JAUNE,     JAUNE],
                [VERT,      VERT,      VERT],
                [colors.white, colors.white, colors.white],
            ]

            story.append(Paragraph("Résumé des niveaux d'alerte", h2_style))
            t = Table(kpi_data, colWidths=[7*cm, 3*cm, 7*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  BLEU),
                ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [GRIS_C, colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                ("ALIGN",       (1,0), (1,-1),  "CENTER"),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",  (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*cm))

        # Météo par province
        if province_data:
            story.append(Paragraph("Conditions météo par province", h2_style))
            prov_rows = [["Province", "T°Max", "T°Min", "Pluie", "Vent", "UV"]]
            for prov, data in province_data.items():
                fc = data.get("forecasts", [{}])
                today_fc = fc[0] if fc else {}
                prov_rows.append([
                    prov,
                    f"{today_fc.get('tmax','—')}°C",
                    f"{today_fc.get('tmin','—')}°C",
                    f"{today_fc.get('precip',0)} mm",
                    f"{today_fc.get('wind','—')} km/h",
                    str(today_fc.get('uv','—')),
                ])
            t_prov = Table(prov_rows, colWidths=[5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm])
            t_prov.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  BLEU),
                ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[GRIS_C, colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                ("ALIGN",       (1,0), (-1,-1), "CENTER"),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ]))
            story.append(t_prov)
            story.append(Spacer(1, 0.4*cm))

        # Top 20 communes à risque
        story.append(Paragraph("Top 20 communes les plus vulnérables", h2_style))
        col_map = "niveau_alerte" if "niveau_alerte" in communes_df.columns else None
        if col_map:
            top = communes_df[communes_df[col_map].isin(["ROUGE","ORANGE"])].head(20)
        else:
            top = communes_df.head(20)

        if len(top) > 0:
            comm_rows = [["Commune", "Province", "Niveau", "Score", "Vulnérables", "Alt.(m)"]]
            for _, row in top.iterrows():
                niv   = row.get("niveau_alerte", "—")
                comm_rows.append([
                    str(row.get("commune", "—")),
                    str(row.get("province_fr", "—")),
                    niv,
                    f"{row.get('score', 0):.0f}/100",
                    f"{int(row.get('pop_vulnerable', 0)):,}",
                    str(int(row.get("altitude", 0))),
                ])

            niv_colors_map = {"ROUGE": ROUGE, "ORANGE": ORANGE,
                               "JAUNE": JAUNE, "VERT": VERT}
            t_comm = Table(comm_rows, colWidths=[5*cm, 3.5*cm, 2*cm, 2*cm, 2.5*cm, 2*cm])
            ts_list = [
                ("BACKGROUND",   (0,0), (-1,0),  BLEU),
                ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
                ("FONTSIZE",     (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [GRIS_C, colors.white]),
                ("GRID",         (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
                ("ALIGN",        (2,0), (-1,-1), "CENTER"),
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",   (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ]
            # Colorier colonne niveau
            for i, (_, row) in enumerate(top.iterrows(), 1):
                niv  = row.get("niveau_alerte", "VERT")
                col  = niv_colors_map.get(niv, VERT)
                ts_list.append(("BACKGROUND", (2,i), (2,i), col))
                ts_list.append(("TEXTCOLOR",  (2,i), (2,i), colors.white))
            t_comm.setStyle(TableStyle(ts_list))
            story.append(t_comm)

        # Avertissement
        story.append(Spacer(1, 0.8*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=GRIS_C))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "⚠️ Ce rapport est généré automatiquement par BeniMeteo sur la base de données "
            "météorologiques et démographiques. Il est destiné à l'aide à la décision et ne "
            "remplace pas une évaluation terrain par les autorités compétentes.",
            ParagraphStyle("warn", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#888888"))))
        story.append(Paragraph(
            f"BeniMeteo · Université Sultan Moulay Slimane · Béni Mellal-Khénifra",
            center_style))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        # Fallback si ReportLab non installé — PDF simple via fpdf2
        return _simple_pdf_fallback(communes_df, title)
    except Exception as e:
        print(f"⚠️ Erreur génération PDF ({e})")
        return b""


def _simple_pdf_fallback(communes_df: pd.DataFrame, title: str) -> bytes:
    """Fallback si ReportLab non dispo — génère un CSV encodé."""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "BeniMeteo - Rapport Alertes", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Date : {date.today()}", ln=True)
        pdf.ln(5)

        if len(communes_df) > 0:
            cols = ["commune", "niveau_alerte", "score", "pop_vulnerable"]
            cols = [c for c in cols if c in communes_df.columns]
            for col in cols:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(45, 7, str(col), border=1)
            pdf.ln()
            for _, row in communes_df.head(30).iterrows():
                pdf.set_font("Helvetica", "", 9)
                for col in cols:
                    pdf.cell(45, 6, str(row.get(col, ""))[:20], border=1)
                pdf.ln()

        return pdf.output()
    except:
        return b""


def save_report_to_file(pdf_bytes: bytes, filename: str = None) -> str:
    """Sauvegarde le PDF dans le dossier rapports."""
    _ensure_output_dir()
    if filename is None:
        filename = f"rapport_alertes_{date.today()}.pdf"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path


def list_reports() -> list:
    """Liste tous les rapports disponibles."""
    _ensure_output_dir()
    files = []
    for fn in os.listdir(OUTPUT_DIR):
        if fn.endswith(".pdf"):
            path = os.path.join(OUTPUT_DIR, fn)
            files.append({
                "nom":    fn,
                "path":   path,
                "taille": os.path.getsize(path),
                "date":   datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d/%m/%Y %H:%M"),
            })
    return sorted(files, key=lambda x: x["date"], reverse=True)
